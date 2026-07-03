//go:build amd64 && !purego

package encoder

import (
	"math/rand"
	"testing"
)

// The AVX2 residual test calls the kernel directly (never gated on
// cpu.Detected.AVX2) so hosts that hide AVX2 (Rosetta 2) still prove
// bit-exactness against the portable reference.
func TestResidualBlockAVX2MatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0x9E51D))
	widths := []int{8, 16, 24, 32, 64}
	heights := []int{4, 8, 16, 32, 64}
	for _, w := range widths {
		for _, h := range heights {
			for trial := range 200 {
				stride := w + rng.Intn(48)
				predStride := w + rng.Intn(48)
				srcOff := rng.Intn(32)
				src := make([]byte, srcOff+stride*h+64)
				pred := make([]byte, predStride*h+64)
				fillBytes(rng, src)
				fillBytes(rng, pred)
				got := make([]int16, w*h)
				want := make([]int16, w*h)
				residualBlockAVX2(got, src, srcOff, stride, pred, predStride, w, h)
				residualBlockPureGo(want, src, srcOff, stride, pred, predStride, w, h)
				for i := range want {
					if got[i] != want[i] {
						t.Fatalf("w%d h%d trial %d idx %d: avx2 %d want %d",
							w, h, trial, i, got[i], want[i])
					}
				}
			}
		}
	}
}
