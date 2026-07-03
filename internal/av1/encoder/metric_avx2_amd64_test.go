//go:build amd64 && !purego

package encoder

import (
	"math/rand"
	"testing"
)

// These AVX2 metric tests call the kernels directly (never gated on
// cpu.Detected.AVX2) so hosts whose CPUID hides AVX2 — notably Rosetta 2, which
// executes VEX correctly but reports AVX2=false — still prove bit-exactness
// against the portable reference.

func TestPixelStatsAVX2MatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0x9151))
	cases := []struct {
		name string
		w, h int
		got  func(src []byte, srcStride int, ref []byte, refStride int) (uint32, int32)
		want func(src []byte, srcStride int, ref []byte, refStride int) (uint32, int32)
	}{
		{"4x4", 4, 4, pixelStats4x4AVX2, pixelStats4x4PureGo},
		{"4x8", 4, 8, pixelStats4x8AVX2, pixelStats4x8PureGo},
		{"4x16", 4, 16, pixelStats4x16AVX2, pixelStats4x16PureGo},
		{"8x4", 8, 4, pixelStats8x4AVX2, pixelStats8x4PureGo},
		{"8x8", 8, 8, pixelStats8x8AVX2, pixelStats8x8PureGo},
		{"8x16", 8, 16, pixelStats8x16AVX2, pixelStats8x16PureGo},
		{"8x32", 8, 32, pixelStats8x32AVX2, pixelStats8x32PureGo},
		{"16x4", 16, 4, pixelStats16x4AVX2, pixelStats16x4PureGo},
		{"16x8", 16, 8, pixelStats16x8AVX2, pixelStats16x8PureGo},
		{"16x16", 16, 16, pixelStats16x16AVX2, pixelStats16x16PureGo},
		{"16x32", 16, 32, pixelStats16x32AVX2, pixelStats16x32PureGo},
		{"16x64", 16, 64, pixelStats16x64AVX2, pixelStats16x64PureGo},
		{"32x8", 32, 8, pixelStats32x8AVX2, pixelStats32x8PureGo},
		{"32x16", 32, 16, pixelStats32x16AVX2, pixelStats32x16PureGo},
		{"32x32", 32, 32, pixelStats32x32AVX2, pixelStats32x32PureGo},
		{"32x64", 32, 64, pixelStats32x64AVX2, pixelStats32x64PureGo},
		{"64x16", 64, 16, pixelStats64x16AVX2, pixelStats64x16PureGo},
		{"64x32", 64, 32, pixelStats64x32AVX2, pixelStats64x32PureGo},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			for trial := range 1500 {
				srcStride := tc.w + rng.Intn(40)
				refStride := tc.w + rng.Intn(40)
				src := make([]byte, srcStride*tc.h)
				ref := make([]byte, refStride*tc.h)
				fillBytes(rng, src)
				fillBytes(rng, ref)
				gSSE, gSum := tc.got(src, srcStride, ref, refStride)
				wSSE, wSum := tc.want(src, srcStride, ref, refStride)
				if gSSE != wSSE || gSum != wSum {
					t.Fatalf("trial %d ss %d rs %d: avx2 (sse=%d,sum=%d) want (sse=%d,sum=%d)",
						trial, srcStride, refStride, gSSE, gSum, wSSE, wSum)
				}
			}
		})
	}
}

func TestSATDCoeffsAVX2MatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0x5A7D))
	// AV1 TX coefficient counts of interest are multiples of 16; also probe a
	// couple of non-multiple-of-8 counts to exercise the scalar tail.
	counts := []int{16, 64, 256, 1024, 8, 24, 40, 15, 17, 33}
	for _, count := range counts {
		for trial := range 500 {
			coeff := make([]int32, count)
			for i := range coeff {
				coeff[i] = int32(rng.Intn(1<<17) - (1 << 16))
			}
			g := satdCoeffsAVX2(coeff, count)
			w := satdCoeffsPureGo(coeff, count)
			if g != w {
				t.Fatalf("count %d trial %d: avx2 %d want %d", count, trial, g, w)
			}
		}
	}
}
