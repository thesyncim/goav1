// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && amd64 && !purego

package transform

import (
	"math/rand"
	"testing"
)

// TestInverseAdd4x4MatchesStaged checks the fused 4x4 kernel against the
// staged transform and residual add for every type but IDTX, every coded
// row extent, magnitudes past the int16 clamps, garbage below the coded
// rows, and separate pred/dst buffers.
func TestInverseAdd4x4MatchesStaged(t *testing.T) {
	if !itxAVX2 {
		t.Skip("no AVX2")
	}
	rng := rand.New(rand.NewSource(44))
	const n, stride = 4, 4 + 8
	coeff := make([]int32, n*n)
	raw := make([]int32, n*n)
	pred := make([]byte, n*stride)
	got := make([]byte, n*stride)
	want := make([]byte, n*stride)
	for trial := 0; trial < 200000; trial++ {
		txType := Type(trial % int(TypeCount))
		if txType == TypeIDTX {
			continue
		}
		activeRows := 1 + rng.Intn(n)
		mag := []int{16, 256, 4096, 32767, 70000, 1 << 30}[(trial/int(TypeCount))%6]
		for c := 0; c < n; c++ {
			for r := 0; r < n; r++ {
				coeff[c*n+r] = 0
				if rng.Intn(3) != 0 {
					coeff[c*n+r] = int32(rng.Intn(2*mag+1) - mag)
				}
			}
		}
		for i := range pred {
			pred[i] = byte(rng.Intn(256))
		}
		if err := InverseBlockBitDepthRawBounded(raw, coeff, n, Size{Width: n, Height: n}, txType, 8, activeRows, n); err != nil {
			t.Fatal(err)
		}
		copy(want, pred)
		for r := 0; r < n; r++ {
			for c := 0; c < n; c++ {
				v := int(pred[r*stride+c]) + int((int64(raw[r*n+c])+8)>>4)
				want[r*stride+c] = byte(min(max(v, 0), 255))
			}
		}
		copy(got, pred)
		if !InverseAdd4x4(got, stride, got, stride, coeff, n, txType, activeRows) {
			t.Fatalf("trial %d: type %d declined", trial, txType)
		}
		for i := range got {
			if got[i] != want[i] {
				t.Fatalf("trial %d (type %d, mag %d, rows %d): mismatch at row %d col %d: fused %d staged %d",
					trial, txType, mag, activeRows, i/stride, i%stride, got[i], want[i])
			}
		}
		for i := range got {
			got[i] = 0xa5
		}
		if !InverseAdd4x4(got, stride, pred, stride, coeff, n, txType, activeRows) {
			t.Fatalf("trial %d: separate-pred call declined", trial)
		}
		for r := 0; r < n; r++ {
			for c := 0; c < stride; c++ {
				w := want[r*stride+c]
				if c >= n {
					w = 0xa5
				}
				if got[r*stride+c] != w {
					t.Fatalf("trial %d: separate-pred mismatch at row %d col %d", trial, r, c)
				}
			}
		}
	}
	if InverseAdd4x4(got, stride, pred, stride, coeff, n, TypeIDTX, 4) {
		t.Fatal("IDTX accepted")
	}
	if InverseAdd4x4(got, stride, pred, stride, coeff, n, TypeDCTDCT, 0) {
		t.Fatal("activeRows 0 accepted")
	}
}
