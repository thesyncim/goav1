//go:build goexperiment.simd && arm64 && !purego

package transform

import (
	"fmt"
	"math/rand"
	"testing"
)

// itxFused32Oracle is the staged path the fused kernel replaces: the
// transform into the raw int32 scratch, then Round2(., 4) added to the
// prediction and clipped to bytes.
func itxFused32Oracle(t *testing.T, dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, activeRows, activeCols int) {
	t.Helper()
	raw := make([]int32, 32*32)
	if err := InverseBlockBitDepthRawBounded(raw, coeff, coeffStride, Size{Width: 32, Height: 32}, TypeDCTDCT, 8, activeRows, activeCols); err != nil {
		t.Fatal(err)
	}
	for r := 0; r < 32; r++ {
		for c := 0; c < 32; c++ {
			v := int(pred[r*predStride+c]) + int((raw[r*32+c]+8)>>4)
			dst[r*dstStride+c] = byte(min(max(v, 0), 255))
		}
	}
}

// TestInverseAddDCT32x32Low8MatchesStagedPath: every block the fused path
// accepts reproduces the staged transform plus the raw add byte for byte, a
// declined block leaves dst untouched, and the decline rate at the
// magnitudes real content codes is zero.
func TestInverseAddDCT32x32Low8MatchesStagedPath(t *testing.T) {
	rng := rand.New(rand.NewSource(31))
	const stride = 32 + 8
	coeff := make([]int32, 32*32)
	pred := make([]byte, 32*stride)
	got := make([]byte, 32*stride)
	want := make([]byte, 32*stride)
	accepted, declined, realistic, realisticDeclined := 0, 0, 0, 0
	for trial := 0; trial < 20000; trial++ {
		clear(coeff)
		activeRows, activeCols := 1+rng.Intn(8), 1+rng.Intn(8)
		// The Low4 network serves a pass whose coded extent is at most 4.
		switch trial % 4 {
		case 0:
			activeRows, activeCols = 1+rng.Intn(4), 1+rng.Intn(4)
		case 1:
			activeRows = 1 + rng.Intn(4)
		case 2:
			activeCols = 1 + rng.Intn(4)
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
		if rng.Intn(4) == 0 {
			coeff[0] = int32(rng.Intn(2*mag+1) - mag)
		}
		for i := range pred {
			pred[i] = byte(rng.Intn(256))
		}
		copy(want, pred)
		itxFused32Oracle(t, want, stride, pred, stride, coeff, 32, activeRows, activeCols)
		// In place (the decoder: the prediction already sits in dst).
		copy(got, pred)
		ok := InverseAddDCT32x32Low8(got, stride, got, stride, coeff, 32, activeRows, activeCols)
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
		// Separate prediction buffer (the encoder's reconstruction).
		for i := range got {
			got[i] = byte(0xa5)
		}
		if !InverseAddDCT32x32Low8(got, stride, pred, stride, coeff, 32, activeRows, activeCols) {
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

func BenchmarkInverseAddDCT32x32Low8(b *testing.B) {
	rng := rand.New(rand.NewSource(3))
	coeff := make([]int32, 32*32)
	for c := 0; c < 4; c++ {
		for r := 0; r < 4; r++ {
			coeff[c*32+r] = int32(rng.Intn(2049) - 1024)
		}
	}
	dst := make([]byte, 32*32)
	for i := range dst {
		dst[i] = byte(rng.Intn(256))
	}
	// The same 4x4-corner coefficients through the Low4 networks (extents 4)
	// and the Low8 networks (extents 8).
	for _, ext := range []int{4, 8} {
		b.Run(fmt.Sprintf("extent%d/fused", ext), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				if !InverseAddDCT32x32Low8(dst, 32, dst, 32, coeff, 32, ext, ext) {
					b.Fatal("declined")
				}
			}
		})
	}
	raw := make([]int32, 32*32)
	b.Run("staged-transform-only", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			if err := InverseBlockBitDepthRawBounded(raw, coeff, 32, Size{Width: 32, Height: 32}, TypeDCTDCT, 8, 4, 4); err != nil {
				b.Fatal(err)
			}
		}
	})
}
