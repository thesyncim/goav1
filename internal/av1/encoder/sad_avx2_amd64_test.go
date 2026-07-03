//go:build amd64 && !purego

package encoder

import (
	"math/rand"
	"testing"
)

// The AVX2 SAD tests call the kernels directly (never gated on
// cpu.Detected.AVX2) so hosts whose CPUID hides AVX2 — notably Rosetta 2, which
// executes VEX correctly but reports AVX2=false — still prove bit-exactness
// against the portable reference.

func fillBytes(rng *rand.Rand, b []byte) {
	for i := range b {
		b[i] = byte(rng.Intn(256))
	}
}

func TestSADx4AVX2MatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0x5AD4))
	cases := []struct {
		name string
		size int
		got  func(src, r0, r1, r2, r3 []byte, stride int) (int, int, int, int)
		want func(src, r0, r1, r2, r3 []byte, stride int) (int, int, int, int)
	}{
		{"8x8x4", 8, sad8x8x4AVX2, sad8x8x4PureGo},
		{"16x16x4", 16, sad16x16x4AVX2, sad16x16x4PureGo},
		{"32x32x4", 32, sad32x32x4AVX2, sad32x32x4PureGo},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			for trial := range 2000 {
				stride := tc.size + rng.Intn(40)
				n := stride * tc.size
				src := make([]byte, n)
				r0 := make([]byte, n)
				r1 := make([]byte, n)
				r2 := make([]byte, n)
				r3 := make([]byte, n)
				fillBytes(rng, src)
				fillBytes(rng, r0)
				fillBytes(rng, r1)
				fillBytes(rng, r2)
				fillBytes(rng, r3)
				g0, g1, g2, g3 := tc.got(src, r0, r1, r2, r3, stride)
				w0, w1, w2, w3 := tc.want(src, r0, r1, r2, r3, stride)
				if g0 != w0 || g1 != w1 || g2 != w2 || g3 != w3 {
					t.Fatalf("trial %d stride %d: avx2 (%d,%d,%d,%d) want (%d,%d,%d,%d)",
						trial, stride, g0, g1, g2, g3, w0, w1, w2, w3)
				}
			}
		})
	}
}

func TestSADx4Step4AVX2MatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0x57E4))
	cases := []struct {
		name string
		size int
		got  func(src, ref []byte, stride int) (int, int, int, int)
		want func(src, ref []byte, stride int) (int, int, int, int)
	}{
		{"8x8x4step4", 8, sad8x8x4Step4AVX2, sad8x8x4Step4PureGo},
		{"16x16x4step4", 16, sad16x16x4Step4AVX2, sad16x16x4Step4PureGo},
		{"32x32x4step4", 32, sad32x32x4Step4AVX2, sad32x32x4Step4PureGo},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			for trial := range 2000 {
				stride := tc.size + 12 + rng.Intn(40)
				n := stride*tc.size + 16
				src := make([]byte, n)
				ref := make([]byte, n)
				fillBytes(rng, src)
				fillBytes(rng, ref)
				g0, g1, g2, g3 := tc.got(src, ref, stride)
				w0, w1, w2, w3 := tc.want(src, ref, stride)
				if g0 != w0 || g1 != w1 || g2 != w2 || g3 != w3 {
					t.Fatalf("trial %d stride %d: avx2 (%d,%d,%d,%d) want (%d,%d,%d,%d)",
						trial, stride, g0, g1, g2, g3, w0, w1, w2, w3)
				}
			}
		})
	}
}
