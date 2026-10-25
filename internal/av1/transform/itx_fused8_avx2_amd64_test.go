//go:build goexperiment.simd && amd64 && !purego

package transform

import (
	"math/rand"
	"testing"
)

// itxFused8Oracle is the staged path the fused 8x8 kernel replaces.
func itxFused8Oracle(t *testing.T, dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, activeRows, activeCols int) {
	t.Helper()
	raw := make([]int32, 8*8)
	if err := InverseBlockBitDepthRawBounded(raw, coeff, coeffStride, Size{Width: 8, Height: 8}, TypeDCTDCT, 8, activeRows, activeCols); err != nil {
		t.Fatal(err)
	}
	for r := 0; r < 8; r++ {
		for c := 0; c < 8; c++ {
			v := int(pred[r*predStride+c]) + int((raw[r*8+c]+8)>>4)
			dst[r*dstStride+c] = byte(min(max(v, 0), 255))
		}
	}
}

// TestInverseAddDCT8x8MatchesStagedPath: every (rows, cols) shape, byte-exact
// against the staged path; declined blocks untouched; no declines at
// +-64 / +-1024 / +-4096 (eight coefficients of +-4096 stay under the guard).
func TestInverseAddDCT8x8MatchesStagedPath(t *testing.T) {
	rng := rand.New(rand.NewSource(43))
	const stride = 8 + 8
	coeff := make([]int32, 8*8)
	pred := make([]byte, 8*stride)
	got := make([]byte, 8*stride)
	want := make([]byte, 8*stride)
	accepted, declined, realistic, realisticDeclined := 0, 0, 0, 0
	for trial := 0; trial < 30000; trial++ {
		clear(coeff)
		activeRows, activeCols := 1+rng.Intn(8), 1+rng.Intn(8)
		mode := trial % 5
		mag := []int{64, 1024, 4096, 32767, 70000}[mode]
		for c := 0; c < activeCols; c++ {
			for r := 0; r < activeRows; r++ {
				if rng.Intn(3) == 0 {
					continue
				}
				coeff[c*8+r] = int32(rng.Intn(2*mag+1) - mag)
			}
		}
		for i := range pred {
			pred[i] = byte(rng.Intn(256))
		}
		copy(want, pred)
		itxFused8Oracle(t, want, stride, pred, stride, coeff, 8, activeRows, activeCols)
		copy(got, pred)
		ok := InverseAddDCT8x8(got, stride, got, stride, coeff, 8, activeRows, activeCols)
		if mode <= 2 {
			realistic++
		}
		if !ok {
			declined++
			if mode <= 2 {
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
		if !InverseAddDCT8x8(got, stride, pred, stride, coeff, 8, activeRows, activeCols) {
			t.Fatalf("trial %d: separate-pred call declined after the in-place call accepted", trial)
		}
		for r := 0; r < 8; r++ {
			for c := 0; c < 8; c++ {
				if got[r*stride+c] != want[r*stride+c] {
					t.Fatalf("trial %d: separate-pred mismatch at row %d col %d", trial, r, c)
				}
			}
			for c := 8; c < stride; c++ {
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

func BenchmarkInverseAddDCT8x8(b *testing.B) {
	rng := rand.New(rand.NewSource(3))
	coeff := make([]int32, 8*8)
	for c := 0; c < 8; c++ {
		for r := 0; r < 8; r++ {
			coeff[c*8+r] = int32(rng.Intn(2049) - 1024)
		}
	}
	dst := make([]byte, 8*8)
	for i := range dst {
		dst[i] = byte(rng.Intn(256))
	}
	raw := make([]int32, 8*8)
	b.Run("fused", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			if !InverseAddDCT8x8(dst, 8, dst, 8, coeff, 8, 8, 8) {
				b.Fatal("declined")
			}
		}
	})
	b.Run("staged-transform-only", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			if err := InverseBlockBitDepthRawBounded(raw, coeff, 8, Size{Width: 8, Height: 8}, TypeDCTDCT, 8, 8, 8); err != nil {
				b.Fatal(err)
			}
		}
	})
}
