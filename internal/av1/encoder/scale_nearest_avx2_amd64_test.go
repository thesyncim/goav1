//go:build amd64 && !purego

package encoder

import (
	"math/rand"
	"testing"
)

// The AVX2 nearest-scale tests call the kernels directly (never gated on
// cpu.Detected.AVX2) so hosts that hide AVX2 (Rosetta 2) still prove
// bit-exactness against the portable reference.

func TestScalePlaneNearestAVX2MatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0x5CA1E))
	// Cover both vectorized ratios (2x, 4x) plus a non-multiple width to exercise
	// the scalar tail, and a fallback ratio (3x) that must route to pure-Go.
	cases := []struct{ dstW, dstH, ratio int }{
		{32, 24, 2}, {40, 18, 2}, {16, 16, 4}, {28, 12, 4}, {30, 10, 3},
	}
	for _, tc := range cases {
		srcW, srcH := tc.dstW*tc.ratio, tc.dstH*tc.ratio
		for trial := range 200 {
			dstStride := tc.dstW + rng.Intn(24)
			srcStride := srcW + rng.Intn(24)
			src := make([]byte, srcStride*srcH+64)
			fillBytes(rng, src)
			got := make([]byte, dstStride*tc.dstH)
			want := make([]byte, dstStride*tc.dstH)
			scalePlaneNearestAVX2(got, dstStride, tc.dstW, tc.dstH, src, srcStride, srcW, srcH)
			scalePlaneNearestPureGo(want, dstStride, tc.dstW, tc.dstH, src, srcStride, srcW, srcH)
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("dstW%d dstH%d ratio%d trial %d idx %d: avx2 %d want %d",
						tc.dstW, tc.dstH, tc.ratio, trial, i, got[i], want[i])
				}
			}
		}
	}
}

func TestScalePlaneNearest16AVX2MatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0x5CA16))
	cases := []struct{ dstW, dstH, ratio int }{
		{16, 20, 2}, {24, 10, 2}, {8, 12, 4}, {20, 8, 4}, {14, 6, 3},
	}
	for _, tc := range cases {
		srcW, srcH := tc.dstW*tc.ratio, tc.dstH*tc.ratio
		for trial := range 200 {
			dstStride := tc.dstW + rng.Intn(24)
			srcStride := srcW + rng.Intn(24)
			src := make([]uint16, srcStride*srcH+64)
			for i := range src {
				src[i] = uint16(rng.Intn(1 << 16))
			}
			got := make([]uint16, dstStride*tc.dstH)
			want := make([]uint16, dstStride*tc.dstH)
			scalePlaneNearest16AVX2(got, dstStride, tc.dstW, tc.dstH, src, srcStride, srcW, srcH)
			scalePlaneNearest16PureGo(want, dstStride, tc.dstW, tc.dstH, src, srcStride, srcW, srcH)
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("dstW%d dstH%d ratio%d trial %d idx %d: avx2 %d want %d",
						tc.dstW, tc.dstH, tc.ratio, trial, i, got[i], want[i])
				}
			}
		}
	}
}
