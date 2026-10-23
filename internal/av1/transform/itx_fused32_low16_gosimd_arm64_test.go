//go:build goexperiment.simd && arm64 && !purego

package transform

import (
	"math/rand"
	"testing"
)

// TestInverseAddDCT32x32Low16MatchesStagedPath: every block the fused Low16
// path accepts reproduces the staged transform plus the raw add byte for
// byte over the (rows, cols) shapes up to 16x16, including the eight-row
// shapes that skip the second row strip; declined blocks leave dst untouched
// and no block at realistic magnitudes declines.
func TestInverseAddDCT32x32Low16MatchesStagedPath(t *testing.T) {
	rng := rand.New(rand.NewSource(37))
	const stride = 32 + 8
	coeff := make([]int32, 32*32)
	pred := make([]byte, 32*stride)
	got := make([]byte, 32*stride)
	want := make([]byte, 32*stride)
	accepted, declined, realistic, realisticDeclined := 0, 0, 0, 0
	for trial := 0; trial < 20000; trial++ {
		clear(coeff)
		activeRows, activeCols := 1+rng.Intn(16), 1+rng.Intn(16)
		if trial%4 == 0 {
			activeRows = 1 + rng.Intn(8)
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
		ok := InverseAddDCT32x32Low16(got, stride, got, stride, coeff, 32, activeRows)
		// Sixteen coefficients of +-4096 in one row exceed the per-lane
		// guard, so only the +-64 and +-1024 classes must never decline.
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
		if !InverseAddDCT32x32Low16(got, stride, pred, stride, coeff, 32, activeRows) {
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
		t.Fatalf("%d blocks at realistic magnitudes declined", realisticDeclined)
	}
}

func BenchmarkInverseAddDCT32x32Low16(b *testing.B) {
	rng := rand.New(rand.NewSource(3))
	coeff := make([]int32, 32*32)
	for c := 0; c < 16; c++ {
		for r := 0; r < 16; r++ {
			coeff[c*32+r] = int32(rng.Intn(2049) - 1024)
		}
	}
	dst := make([]byte, 32*32)
	for i := range dst {
		dst[i] = byte(rng.Intn(256))
	}
	b.Run("fused", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			if !InverseAddDCT32x32Low16(dst, 32, dst, 32, coeff, 32, 16) {
				b.Fatal("declined")
			}
		}
	})
	raw := make([]int32, 32*32)
	b.Run("staged-transform-only", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			if err := InverseBlockBitDepthRawBounded(raw, coeff, 32, Size{Width: 32, Height: 32}, TypeDCTDCT, 8, 16, 16); err != nil {
				b.Fatal(err)
			}
		}
	})
}
