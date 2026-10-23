// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package transform

import (
	"math/rand"
	"testing"
)

var itxHybridTypesUnderTest = []Type{
	TypeADSTDCT, TypeDCTADST, TypeADSTADST, TypeFlipADSTDCT, TypeDCTFlipADST,
	TypeFlipADSTFlipADST, TypeADSTFlipADST, TypeFlipADSTADST,
}

// itxFusedHybridOracle is the staged path the fused hybrid kernels replace.
func itxFusedHybridOracle(t *testing.T, n int, dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, txType Type, activeRows, activeCols int) {
	t.Helper()
	raw := make([]int32, n*n)
	if err := InverseBlockBitDepthRawBounded(raw, coeff, n, Size{Width: uint8(n), Height: uint8(n)}, txType, 8, activeRows, activeCols); err != nil {
		t.Fatal(err)
	}
	for r := 0; r < n; r++ {
		for c := 0; c < n; c++ {
			v := int(pred[r*predStride+c]) + int((raw[r*n+c]+8)>>4)
			dst[r*dstStride+c] = byte(min(max(v, 0), 255))
		}
	}
}

func testInverseAddHybrid(t *testing.T, n int, trials int, fused func(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, txType Type, activeRows, activeCols int) bool) {
	rng := rand.New(rand.NewSource(int64(53 + n)))
	stride := n + 8
	coeff := make([]int32, n*n)
	pred := make([]byte, n*stride)
	got := make([]byte, n*stride)
	want := make([]byte, n*stride)
	accepted, declined, realistic, realisticDeclined := 0, 0, 0, 0
	for trial := 0; trial < trials; trial++ {
		clear(coeff)
		txType := itxHybridTypesUnderTest[trial%len(itxHybridTypesUnderTest)]
		activeRows, activeCols := 1+rng.Intn(n), 1+rng.Intn(n)
		switch (trial / len(itxHybridTypesUnderTest)) % 3 {
		case 0:
			activeRows, activeCols = 1+rng.Intn(4), 1+rng.Intn(4)
		case 1:
			activeRows, activeCols = 1+rng.Intn(8), 1+rng.Intn(8)
		}
		mode := (trial / 7) % 5
		mag := []int{64, 1024, 4096, 32767, 70000}[mode]
		for c := 0; c < activeCols; c++ {
			for r := 0; r < activeRows; r++ {
				if rng.Intn(3) == 0 {
					continue
				}
				coeff[c*n+r] = int32(rng.Intn(2*mag+1) - mag)
			}
		}
		for i := range pred {
			pred[i] = byte(rng.Intn(256))
		}
		copy(want, pred)
		itxFusedHybridOracle(t, n, want, stride, pred, stride, coeff, txType, activeRows, activeCols)
		copy(got, pred)
		ok := fused(got, stride, got, stride, coeff, n, txType, activeRows, activeCols)
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
				t.Fatalf("trial %d (type %d, mode %d, rows %d cols %d): mismatch at row %d col %d: fused %d staged %d",
					trial, txType, mode, activeRows, activeCols, i/stride, i%stride, got[i], want[i])
			}
		}
		for i := range got {
			got[i] = byte(0xa5)
		}
		if !fused(got, stride, pred, stride, coeff, n, txType, activeRows, activeCols) {
			t.Fatalf("trial %d: separate-pred call declined after the in-place call accepted", trial)
		}
		for r := 0; r < n; r++ {
			for c := 0; c < n; c++ {
				if got[r*stride+c] != want[r*stride+c] {
					t.Fatalf("trial %d: separate-pred mismatch at row %d col %d", trial, r, c)
				}
			}
			for c := n; c < stride; c++ {
				if got[r*stride+c] != 0xa5 {
					t.Fatalf("trial %d: wrote past the block at row %d col %d", trial, r, c)
				}
			}
		}
	}
	t.Logf("%dx%d: accepted %d, declined %d (realistic magnitudes: %d of %d declined)", n, n, accepted, declined, realisticDeclined, realistic)
	if realisticDeclined != 0 {
		t.Fatalf("%d realistic blocks declined", realisticDeclined)
	}
	if accepted == 0 {
		t.Fatal("no block accepted")
	}
	if fused(got, stride, pred, stride, coeff, n, TypeDCTDCT, 1, 1) {
		t.Fatal("DCT_DCT accepted by the hybrid kernel")
	}
	if fused(got, stride, pred, stride, coeff, n, TypeIDTX, 1, 1) {
		t.Fatal("IDTX accepted by the hybrid kernel")
	}
}

// TestInverseAdd16x16HybridMatchesStagedPath: every hybrid type, every
// (rows, cols) shape and form pairing, byte-exact against the staged path.
func TestInverseAdd16x16HybridMatchesStagedPath(t *testing.T) {
	testInverseAddHybrid(t, 16, 40000, InverseAdd16x16Hybrid)
}

// TestInverseAdd8x8HybridMatchesStagedPath is the 8x8 counterpart.
func TestInverseAdd8x8HybridMatchesStagedPath(t *testing.T) {
	testInverseAddHybrid(t, 8, 40000, InverseAdd8x8Hybrid)
}

func BenchmarkInverseAddHybrid(b *testing.B) {
	for _, shape := range []struct {
		name       string
		n          int
		rows, cols int
	}{{"16x16/low4", 16, 4, 4}, {"16x16/full", 16, 16, 16}, {"8x8/full", 8, 8, 8}} {
		rng := rand.New(rand.NewSource(5))
		n := shape.n
		coeff := make([]int32, n*n)
		for c := 0; c < shape.cols; c++ {
			for r := 0; r < shape.rows; r++ {
				coeff[c*n+r] = int32(rng.Intn(2049) - 1024)
			}
		}
		dst := make([]byte, n*n)
		raw := make([]int32, n*n)
		b.Run(shape.name+"/adst-adst/fused", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				var ok bool
				if n == 16 {
					ok = InverseAdd16x16Hybrid(dst, n, dst, n, coeff, n, TypeADSTADST, shape.rows, shape.cols)
				} else {
					ok = InverseAdd8x8Hybrid(dst, n, dst, n, coeff, n, TypeADSTADST, shape.rows, shape.cols)
				}
				if !ok {
					b.Fatal("declined")
				}
			}
		})
		b.Run(shape.name+"/adst-adst/staged-transform-only", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				if err := InverseBlockBitDepthRawBounded(raw, coeff, n, Size{Width: uint8(n), Height: uint8(n)}, TypeADSTADST, 8, shape.rows, shape.cols); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
