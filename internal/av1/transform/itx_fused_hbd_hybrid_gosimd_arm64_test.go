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

// TestInverseAddHybridHBDMatchesStagedPath: every accepted 10- and 12-bit
// 16x16 and 8x8 DCT/ADST/flipped-ADST block reproduces the staged transform
// plus the raw add sample for sample, in place and with a separate
// prediction, without writing past the block; a declined block is left
// untouched, and the other transform types are declined.
func TestInverseAddHybridHBDMatchesStagedPath(t *testing.T) {
	rng := rand.New(rand.NewSource(1017))
	types := []Type{TypeADSTDCT, TypeDCTADST, TypeADSTADST, TypeFlipADSTDCT, TypeDCTFlipADST,
		TypeFlipADSTFlipADST, TypeADSTFlipADST, TypeFlipADSTADST}
	fused := func(n int, dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, txType Type, activeRows, activeCols int, bd uint8) bool {
		if n == 8 {
			return InverseAdd8x8HybridHBD(dst, dstStride, pred, predStride, coeff, n, txType, activeRows, activeCols, bd)
		}
		return InverseAdd16x16HybridHBD(dst, dstStride, pred, predStride, coeff, n, txType, activeRows, activeCols, bd)
	}
	for _, n := range []int{8, 16} {
		for _, bd := range []uint8{10, 12} {
			stride := 2 * (n + 8)
			coeff := make([]int32, n*n)
			raw := make([]int32, n*n)
			pred := make([]byte, n*stride)
			got := make([]byte, n*stride)
			want := make([]byte, n*stride)
			maxV := 1<<bd - 1
			accepted, declined, lowDeclined, low := 0, 0, 0, 0
			for trial := 0; trial < 12000; trial++ {
				txType := types[trial%len(types)]
				clear(coeff)
				activeRows, activeCols := 1+rng.Intn(n), 1+rng.Intn(n)
				if trial%3 == 0 {
					activeRows, activeCols = 1+rng.Intn(min(n, 4)), 1+rng.Intn(min(n, 8))
				}
				mode := (trial / len(types)) % 6
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
				if err := InverseBlockBitDepthRawBounded(raw, coeff, n, Size{Width: uint8(n), Height: uint8(n)}, txType, bd, activeRows, activeCols); err != nil {
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
				ok := fused(n, got, stride, got, stride, coeff, txType, activeRows, activeCols, bd)
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
						t.Fatalf("n %d bd %d type %d trial %d (mode %d, rows %d cols %d): mismatch at row %d byte %d", n, bd, txType, trial, mode, activeRows, activeCols, i/stride, i%stride)
					}
				}
				for i := range got {
					got[i] = 0xa5
				}
				if !fused(n, got, stride, pred, stride, coeff, txType, activeRows, activeCols, bd) {
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
			for _, other := range []Type{TypeDCTDCT, TypeIDTX, TypeVDCT, TypeHDCT} {
				if fused(n, got, stride, pred, stride, coeff, other, 1, 1, bd) {
					t.Fatalf("n %d: type %d accepted", n, other)
				}
			}
		}
	}
}
