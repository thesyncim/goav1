// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package transform

import (
	"math/rand"
	"testing"
)

// TestInverseAddDCT32x32FullMatchesStagedPath: every (rows, cols) shape up
// to 32x32, every network form pairing, byte-exact against the staged path;
// declined blocks untouched; no declines at +-64 / +-1024.
func TestInverseAddDCT32x32FullMatchesStagedPath(t *testing.T) {
	rng := rand.New(rand.NewSource(47))
	const stride = 32 + 8
	coeff := make([]int32, 32*32)
	pred := make([]byte, 32*stride)
	got := make([]byte, 32*stride)
	want := make([]byte, 32*stride)
	accepted, declined, realistic, realisticDeclined := 0, 0, 0, 0
	for trial := 0; trial < 20000; trial++ {
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
		itxFused32Oracle(t, want, stride, pred, stride, coeff, 32, activeRows, activeCols)
		copy(got, pred)
		ok := InverseAddDCT32x32Full(got, stride, got, stride, coeff, 32, activeRows, activeCols)
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
		if !InverseAddDCT32x32Full(got, stride, pred, stride, coeff, 32, activeRows, activeCols) {
			t.Fatalf("trial %d: separate-pred call declined after the in-place call accepted", trial)
		}
		for r := 0; r < 32; r++ {
			for c := 0; c < 32; c++ {
				if got[r*stride+c] != want[r*stride+c] {
					t.Fatalf("trial %d: separate-pred mismatch at row %d col %d", trial, r, c)
				}
			}
			for c := 32; c < stride; c++ {
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

func BenchmarkInverseAddDCT32x32Full(b *testing.B) {
	for _, shape := range []struct {
		name       string
		rows, cols int
	}{{"low16", 16, 16}, {"full", 32, 32}} {
		rng := rand.New(rand.NewSource(5))
		coeff := make([]int32, 32*32)
		for c := 0; c < shape.cols; c++ {
			for r := 0; r < shape.rows; r++ {
				coeff[c*32+r] = int32(rng.Intn(2049) - 1024)
			}
		}
		dst := make([]byte, 32*32)
		raw := make([]int32, 32*32)
		b.Run(shape.name+"/fused", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				if !InverseAddDCT32x32Full(dst, 32, dst, 32, coeff, 32, shape.rows, shape.cols) {
					b.Fatal("declined")
				}
			}
		})
		b.Run(shape.name+"/staged-transform-only", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				if err := InverseBlockBitDepthRawBounded(raw, coeff, 32, Size{Width: 32, Height: 32}, TypeDCTDCT, 8, shape.rows, shape.cols); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
