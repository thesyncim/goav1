// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package transform

import (
	"encoding/binary"
	"math/rand"
	"testing"
)

// TestInverseAddDCTRect2HBDMatchesStagedPath: every accepted 10- and 12-bit
// 16x32 and 32x16 DCT_DCT block reproduces the staged transform plus the raw
// add sample for sample, in place and with a separate prediction, without
// writing past the block; a declined block is left untouched. A block with
// a single coefficient whose 1/sqrt(2)-scaled value leaves int16 but stays
// inside the 10- and 12-bit row clamps (bd + 8 bits) is kept by the staged
// path; the fused load saturates it to +-32767, above the L1 guard, so the
// fused path must decline it.
func TestInverseAddDCTRect2HBDMatchesStagedPath(t *testing.T) {
	rng := rand.New(rand.NewSource(1032))
	for _, shape := range [][2]int{{16, 32}, {32, 16}} {
		w, h := shape[0], shape[1]
		fused := InverseAddDCT16x32HBD
		if w == 32 {
			fused = InverseAddDCT32x16HBD
		}
		cs := min(h, 32)
		for _, bd := range []uint8{10, 12} {
			stride := 2 * (w + 8)
			coeff := make([]int32, w*cs)
			raw := make([]int32, w*h)
			pred := make([]byte, h*stride)
			got := make([]byte, h*stride)
			want := make([]byte, h*stride)
			maxV := 1<<bd - 1
			accepted, declined, lowDeclined, low := 0, 0, 0, 0
			for trial := 0; trial < 8000; trial++ {
				clear(coeff)
				activeRows, activeCols := 1+rng.Intn(h), 1+rng.Intn(w)
				switch trial % 4 {
				case 0:
					activeRows, activeCols = 1+rng.Intn(8), 1+rng.Intn(8)
				case 1:
					activeRows, activeCols = 1+rng.Intn(16), 1+rng.Intn(16)
				}
				mode := trial % 7
				mag := []int{64, 1024, 4096, 32767, 0, 300000, 1 << 23}[mode]
				// Mode 4: one coefficient whose scaled value |c|*181/256 lies
				// between int16 and the row clamp (46341 < |c| < 185000 keeps it
				// under 2^17 - 1, the 10-bit clamp).
				between := mode == 4
				if between {
					v := int32(47000 + rng.Intn(133000))
					if rng.Intn(2) == 0 {
						v = -v
					}
					coeff[rng.Intn(activeCols)*cs+rng.Intn(activeRows)] = v
				}
				for c := 0; c < activeCols && !between; c++ {
					for r := 0; r < activeRows; r++ {
						if rng.Intn(3) == 0 {
							continue
						}
						coeff[c*cs+r] = int32(rng.Intn(2*mag+1) - mag)
					}
				}
				for i := 0; i < len(pred); i += 2 {
					v := rng.Intn(maxV + 1)
					if trial%5 == 0 {
						v = maxV * rng.Intn(2)
					}
					binary.LittleEndian.PutUint16(pred[i:], uint16(v))
				}
				if err := InverseBlockBitDepthRawBounded(raw, coeff, cs, Size{Width: uint8(w), Height: uint8(h)}, TypeDCTDCT, bd, activeRows, activeCols); err != nil {
					t.Fatal(err)
				}
				copy(want, pred)
				for r := 0; r < h; r++ {
					for c := 0; c < w; c++ {
						p := int(binary.LittleEndian.Uint16(pred[r*stride+2*c:]))
						v := min(max(p+int((raw[r*w+c]+8)>>4), 0), maxV)
						binary.LittleEndian.PutUint16(want[r*stride+2*c:], uint16(v))
					}
				}
				copy(got, pred)
				ok := fused(got, stride, got, stride, coeff, cs, activeRows, activeCols, bd)
				if mode <= 1 {
					low++
				}
				if between && ok {
					t.Fatalf("%dx%d bd %d trial %d: a coefficient past the int16 range after the rect2 scale was accepted", w, h, bd, trial)
				}
				if !ok {
					declined++
					if mode <= 1 {
						lowDeclined++
					}
					for i := range got {
						if got[i] != pred[i] {
							t.Fatalf("%dx%d bd %d trial %d: declined block modified dst at %d", w, h, bd, trial, i)
						}
					}
					continue
				}
				accepted++
				for i := range got {
					if got[i] != want[i] {
						t.Fatalf("%dx%d bd %d trial %d (mode %d, rows %d cols %d): mismatch at row %d byte %d", w, h, bd, trial, mode, activeRows, activeCols, i/stride, i%stride)
					}
				}
				for i := range got {
					got[i] = 0xa5
				}
				if !fused(got, stride, pred, stride, coeff, cs, activeRows, activeCols, bd) {
					t.Fatalf("%dx%d bd %d trial %d: separate-pred call declined after the in-place call accepted", w, h, bd, trial)
				}
				for r := 0; r < h; r++ {
					for c := 0; c < stride; c++ {
						wv := want[r*stride+c]
						if c >= 2*w {
							wv = 0xa5
						}
						if got[r*stride+c] != wv {
							t.Fatalf("%dx%d bd %d trial %d: separate-pred mismatch at row %d byte %d", w, h, bd, trial, r, c)
						}
					}
				}
			}
			t.Logf("%dx%d bd %d: accepted %d, declined %d (low magnitudes: %d of %d declined)", w, h, bd, accepted, declined, lowDeclined, low)
			if lowDeclined != 0 || accepted == 0 {
				t.Fatalf("%dx%d bd %d: %d low-magnitude declines, %d accepted", w, h, bd, lowDeclined, accepted)
			}
		}
	}
}
