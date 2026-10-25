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

// itxFusedHBDOracle is the staged transform at bitDepth plus the raw 16bpc
// add: dst = clip(pred + Round2(raw, 4), 0, 2^bd - 1).
func itxFusedHBDOracle(t *testing.T, n int, bitDepth uint8, dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, activeRows, activeCols int) {
	t.Helper()
	raw := make([]int32, n*n)
	if err := InverseBlockBitDepthRawBounded(raw, coeff, coeffStride, Size{Width: uint8(n), Height: uint8(n)}, TypeDCTDCT, bitDepth, activeRows, activeCols); err != nil {
		t.Fatal(err)
	}
	maxV := 1<<bitDepth - 1
	for r := 0; r < n; r++ {
		for c := 0; c < n; c++ {
			p := int(binary.LittleEndian.Uint16(pred[r*predStride+2*c:]))
			v := min(max(p+int((raw[r*n+c]+8)>>4), 0), maxV)
			binary.LittleEndian.PutUint16(dst[r*dstStride+2*c:], uint16(v))
		}
	}
}

func itxFusedHBD(n int, dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, activeRows, activeCols int, bitDepth uint8) bool {
	switch n {
	case 8:
		return InverseAddDCT8x8HBD(dst, dstStride, pred, predStride, coeff, coeffStride, activeRows, activeCols, bitDepth)
	case 16:
		return InverseAddDCT16x16HBD(dst, dstStride, pred, predStride, coeff, coeffStride, activeRows, activeCols, bitDepth)
	default:
		return InverseAddDCT32x32HBD(dst, dstStride, pred, predStride, coeff, coeffStride, activeRows, activeCols, bitDepth)
	}
}

// TestInverseAddDCTHBDMatchesStagedPath: every accepted 10- and 12-bit
// block reproduces the staged transform plus the raw add sample for sample,
// in place and with a separate prediction, without writing past the block;
// a declined block is left untouched.
func TestInverseAddDCTHBDMatchesStagedPath(t *testing.T) {
	rng := rand.New(rand.NewSource(1016))
	for _, n := range []int{8, 16, 32} {
		for _, bd := range []uint8{10, 12} {
			stride := 2 * (n + 8)
			coeff := make([]int32, n*n)
			pred := make([]byte, n*stride)
			got := make([]byte, n*stride)
			want := make([]byte, n*stride)
			maxV := 1<<bd - 1
			trials := map[int]int{8: 6000, 16: 8000, 32: 12000}[n]
			accepted, declined, lowDeclined, low := 0, 0, 0, 0
			for trial := 0; trial < trials; trial++ {
				clear(coeff)
				activeRows, activeCols := 1+rng.Intn(n), 1+rng.Intn(n)
				switch trial % 4 {
				case 0:
					activeRows, activeCols = 1+rng.Intn(min(n, 8)), 1+rng.Intn(min(n, 8))
				case 1:
					activeRows, activeCols = 1+rng.Intn(min(n, 16)), 1+rng.Intn(min(n, 16))
				}
				mode := trial % 6
				mag := []int{64, 256, 1024, 4096, 32767, 300000}[mode]
				for c := 0; c < activeCols; c++ {
					for r := 0; r < activeRows; r++ {
						if rng.Intn(3) == 0 {
							continue
						}
						coeff[c*n+r] = int32(rng.Intn(2*mag+1) - mag)
					}
				}
				for i := 0; i < len(pred); i += 2 {
					v := rng.Intn(maxV + 1)
					if trial%7 == 0 {
						v = maxV * rng.Intn(2)
					}
					binary.LittleEndian.PutUint16(pred[i:], uint16(v))
				}
				copy(want, pred)
				itxFusedHBDOracle(t, n, bd, want, stride, pred, stride, coeff, n, activeRows, activeCols)
				copy(got, pred)
				ok := itxFusedHBD(n, got, stride, got, stride, coeff, n, activeRows, activeCols, bd)
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
							t.Fatalf("n %d bd %d trial %d: declined block modified dst at %d", n, bd, trial, i)
						}
					}
					continue
				}
				accepted++
				for i := range got {
					if got[i] != want[i] {
						t.Fatalf("n %d bd %d trial %d (mode %d, rows %d cols %d): mismatch at row %d byte %d", n, bd, trial, mode, activeRows, activeCols, i/stride, i%stride)
					}
				}
				for i := range got {
					got[i] = 0xa5
				}
				if !itxFusedHBD(n, got, stride, pred, stride, coeff, n, activeRows, activeCols, bd) {
					t.Fatalf("n %d bd %d trial %d: separate-pred call declined after the in-place call accepted", n, bd, trial)
				}
				for r := 0; r < n; r++ {
					for c := 0; c < stride; c++ {
						w := want[r*stride+c]
						if c >= 2*n {
							w = 0xa5
						}
						if got[r*stride+c] != w {
							t.Fatalf("n %d bd %d trial %d: separate-pred mismatch at row %d byte %d", n, bd, trial, r, c)
						}
					}
				}
			}
			t.Logf("n %d bd %d: accepted %d, declined %d (low magnitudes: %d of %d declined)", n, bd, accepted, declined, lowDeclined, low)
			if lowDeclined != 0 || accepted == 0 {
				t.Fatalf("n %d bd %d: %d low-magnitude declines, %d accepted", n, bd, lowDeclined, accepted)
			}
		}
	}
}
