// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && amd64 && !purego

package transform

import (
	"math/rand"
	"testing"
)

// itxFused64Oracle is the staged path the fused 64x64 kernel replaces.
func itxFused64Oracle(t *testing.T, dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, activeRows, activeCols int) {
	t.Helper()
	raw := make([]int32, 64*64)
	if err := InverseBlockBitDepthRawBounded(raw, coeff, coeffStride, Size{Width: 64, Height: 64}, TypeDCTDCT, 8, activeRows, activeCols); err != nil {
		t.Fatal(err)
	}
	for r := 0; r < 64; r++ {
		for c := 0; c < 64; c++ {
			v := int(pred[r*predStride+c]) + int((raw[r*64+c]+8)>>4)
			dst[r*dstStride+c] = byte(min(max(v, 0), 255))
		}
	}
}

// TestInverseAddDCT64x64MatchesStagedPath: every (rows, cols) shape within
// the 32x32 corner, every network form pairing, byte-exact against the
// staged path; declined blocks untouched; no declines at +-64 / +-1024.
func TestInverseAddDCT64x64MatchesStagedPath(t *testing.T) {
	rng := rand.New(rand.NewSource(43))
	const stride = 64 + 8
	coeff := make([]int32, 32*32)
	pred := make([]byte, 64*stride)
	got := make([]byte, 64*stride)
	want := make([]byte, 64*stride)
	accepted, declined, realistic, realisticDeclined := 0, 0, 0, 0
	for trial := 0; trial < 6000; trial++ {
		clear(coeff)
		activeRows, activeCols := 1+rng.Intn(32), 1+rng.Intn(32)
		switch trial % 4 {
		case 0:
			activeRows, activeCols = 1+rng.Intn(8), 1+rng.Intn(8)
		case 1:
			activeRows, activeCols = 1+rng.Intn(16), 1+rng.Intn(16)
		}
		mode := trial % 5
		mag := []int{64, 1024, 4096, 32767, 70000}[mode]
		for c := 0; c < activeCols; c++ {
			for r := 0; r < activeRows; r++ {
				if rng.Intn(3) == 0 {
					continue
				}
				coeff[c*32+r] = int32(rng.Intn(2*mag+1) - mag)
			}
		}
		for i := range pred {
			pred[i] = byte(rng.Intn(256))
		}
		copy(want, pred)
		itxFused64Oracle(t, want, stride, pred, stride, coeff, 32, activeRows, activeCols)
		copy(got, pred)
		ok := InverseAddDCT64x64(got, stride, got, stride, coeff, 32, activeRows, activeCols)
		if mode <= 1 {
			realistic++
		}
		if !ok {
			declined++
			if mode <= 1 {
				realisticDeclined++
			}
			for i := range got {
				if got[i] != pred[i] {
					t.Fatalf("trial %d: declined block modified dst at %d", trial, i)
				}
			}
			continue
		}
		accepted++
		for i := range got {
			if got[i] != want[i] {
				t.Fatalf("trial %d (mode %d, rows %d cols %d): mismatch at row %d col %d: fused %d staged %d",
					trial, mode, activeRows, activeCols, i/stride, i%stride, got[i], want[i])
			}
		}
		for i := range got {
			got[i] = byte(0xa5)
		}
		if !InverseAddDCT64x64(got, stride, pred, stride, coeff, 32, activeRows, activeCols) {
			t.Fatalf("trial %d: separate-pred call declined after the in-place call accepted", trial)
		}
		for r := 0; r < 64; r++ {
			for c := 0; c < 64; c++ {
				if got[r*stride+c] != want[r*stride+c] {
					t.Fatalf("trial %d: separate-pred mismatch at row %d col %d", trial, r, c)
				}
			}
			for c := 64; c < stride; c++ {
				if got[r*stride+c] != 0xa5 {
					t.Fatalf("trial %d: wrote past the block at row %d col %d", trial, r, c)
				}
			}
		}
	}
	t.Logf("accepted %d, declined %d (realistic magnitudes: %d of %d declined)", accepted, declined, realisticDeclined, realistic)
	if realisticDeclined != 0 {
		t.Fatalf("%d realistic blocks declined", realisticDeclined)
	}
	if accepted == 0 {
		t.Fatal("no block accepted")
	}
}

func BenchmarkInverseAddDCT64x64(b *testing.B) {
	for _, shape := range []struct {
		name       string
		rows, cols int
	}{{"low8", 8, 8}, {"low16", 16, 16}, {"low32", 32, 32}} {
		rng := rand.New(rand.NewSource(5))
		coeff := make([]int32, 32*32)
		for c := 0; c < shape.cols; c++ {
			for r := 0; r < shape.rows; r++ {
				coeff[c*32+r] = int32(rng.Intn(2049) - 1024)
			}
		}
		dst := make([]byte, 64*64)
		raw := make([]int32, 64*64)
		b.Run(shape.name+"/fused", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				if !InverseAddDCT64x64(dst, 64, dst, 64, coeff, 32, shape.rows, shape.cols) {
					b.Fatal("declined")
				}
			}
		})
		b.Run(shape.name+"/staged-transform-only", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				if err := InverseBlockBitDepthRawBounded(raw, coeff, 32, Size{Width: 64, Height: 64}, TypeDCTDCT, 8, shape.rows, shape.cols); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
