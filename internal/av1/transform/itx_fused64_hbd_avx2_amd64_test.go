// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && amd64 && !purego

package transform

import (
	"encoding/binary"
	"math/rand"
	"testing"
)

// TestInverseAddDCT64x64HBDMatchesStagedPath: every accepted 10- and 12-bit
// 64x64 block (every network form pairing within the 32x32 corner)
// reproduces the staged transform plus the raw add sample for sample, in
// place and with a separate prediction, without writing past the block; a
// declined block is left untouched and low magnitudes never decline.
func TestInverseAddDCT64x64HBDMatchesStagedPath(t *testing.T) {
	rng := rand.New(rand.NewSource(1064))
	const n = 64
	const stride = 2 * (n + 8)
	for _, bd := range []uint8{10, 12} {
		coeff := make([]int32, 32*32)
		raw := make([]int32, n*n)
		pred := make([]byte, n*stride)
		got := make([]byte, n*stride)
		want := make([]byte, n*stride)
		maxV := 1<<bd - 1
		accepted, declined, lowDeclined, low := 0, 0, 0, 0
		for trial := 0; trial < 4000; trial++ {
			clear(coeff)
			activeRows, activeCols := 1+rng.Intn(32), 1+rng.Intn(32)
			switch trial % 4 {
			case 0:
				activeRows, activeCols = 1+rng.Intn(8), 1+rng.Intn(8)
			case 1:
				activeRows, activeCols = 1+rng.Intn(16), 1+rng.Intn(16)
			}
			mode := trial % 6
			mag := []int{64, 1024, 4096, 32767, 70000, 300000}[mode]
			for c := 0; c < activeCols; c++ {
				for r := 0; r < activeRows; r++ {
					if rng.Intn(3) == 0 {
						continue
					}
					coeff[c*32+r] = int32(rng.Intn(2*mag+1) - mag)
				}
			}
			for i := 0; i < len(pred); i += 2 {
				v := rng.Intn(maxV + 1)
				if trial%7 == 0 {
					v = maxV * rng.Intn(2)
				}
				binary.LittleEndian.PutUint16(pred[i:], uint16(v))
			}
			if err := InverseBlockBitDepthRawBounded(raw, coeff, 32, Size{Width: n, Height: n}, TypeDCTDCT, bd, activeRows, activeCols); err != nil {
				t.Fatal(err)
			}
			copy(want, pred)
			for r := 0; r < n; r++ {
				for c := 0; c < n; c++ {
					p := int(binary.LittleEndian.Uint16(pred[r*stride+2*c:]))
					v := min(max(p+int((raw[r*n+c]+8)>>4), 0), maxV)
					binary.LittleEndian.PutUint16(want[r*stride+2*c:], uint16(v))
				}
			}
			copy(got, pred)
			ok := InverseAddDCT64x64HBD(got, stride, got, stride, coeff, 32, activeRows, activeCols, bd)
			if mode <= 1 {
				low++
			}
			if !ok {
				declined++
				if mode <= 1 {
					lowDeclined++
				}
				for i := range got {
					if got[i] != pred[i] {
						t.Fatalf("bd %d trial %d: declined block modified dst at %d", bd, trial, i)
					}
				}
				continue
			}
			accepted++
			for i := range got {
				if got[i] != want[i] {
					t.Fatalf("bd %d trial %d (mode %d, rows %d cols %d): mismatch at row %d byte %d", bd, trial, mode, activeRows, activeCols, i/stride, i%stride)
				}
			}
			for i := range got {
				got[i] = 0xa5
			}
			if !InverseAddDCT64x64HBD(got, stride, pred, stride, coeff, 32, activeRows, activeCols, bd) {
				t.Fatalf("bd %d trial %d: separate-pred call declined after the in-place call accepted", bd, trial)
			}
			for r := 0; r < n; r++ {
				for c := 0; c < stride; c++ {
					w := want[r*stride+c]
					if c >= 2*n {
						w = 0xa5
					}
					if got[r*stride+c] != w {
						t.Fatalf("bd %d trial %d: separate-pred mismatch at row %d byte %d", bd, trial, r, c)
					}
				}
			}
		}
		t.Logf("bd %d: accepted %d, declined %d (low magnitudes: %d of %d declined)", bd, accepted, declined, lowDeclined, low)
		if lowDeclined != 0 || accepted == 0 {
			t.Fatalf("bd %d: %d low-magnitude declines, %d accepted", bd, lowDeclined, accepted)
		}
	}
}
