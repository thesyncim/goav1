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

func TestSAD32x32AVX2MatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0x3232))
	for trial := range 3000 {
		stride := 32 + rng.Intn(40)
		n := stride * 32
		src := make([]byte, n)
		ref := make([]byte, n)
		fillBytes(rng, src)
		fillBytes(rng, ref)
		if g, w := sad32x32AVX2(src, ref, stride), sad32x32PureGo(src, ref, stride); g != w {
			t.Fatalf("trial %d stride %d: avx2 %d want %d", trial, stride, g, w)
		}
	}
}

func TestSADDualAVX2MatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0xD0A1))
	cases := []struct {
		name string
		size int
		got  func(src []byte, srcStride int, ref []byte, refStride int) int
		want func(src []byte, srcStride int, ref []byte, refStride int) int
	}{
		{"16x16dual", 16, sad16x16DualAVX2, sad16x16DualPureGo},
		{"32x32dual", 32, sad32x32DualAVX2, sad32x32DualPureGo},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			for trial := range 3000 {
				srcStride := tc.size + rng.Intn(40)
				refStride := tc.size + rng.Intn(40)
				src := make([]byte, srcStride*tc.size)
				ref := make([]byte, refStride*tc.size)
				fillBytes(rng, src)
				fillBytes(rng, ref)
				g := tc.got(src, srcStride, ref, refStride)
				w := tc.want(src, srcStride, ref, refStride)
				if g != w {
					t.Fatalf("trial %d ss %d rs %d: avx2 %d want %d", trial, srcStride, refStride, g, w)
				}
			}
		})
	}
}

func TestSAD8x8CompoundAvgBlockAVX2MatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0xC0FFEE))
	for trial := range 3000 {
		srcStride := 8 + rng.Intn(40)
		r0Stride := 8 + rng.Intn(40)
		r1Stride := 8 + rng.Intn(40)
		src := make([]byte, srcStride*8)
		ref0 := make([]byte, r0Stride*8)
		ref1 := make([]byte, r1Stride*8)
		fillBytes(rng, src)
		fillBytes(rng, ref0)
		fillBytes(rng, ref1)
		g := sad8x8CompoundAvgBlockAVX2(src, srcStride, ref0, r0Stride, ref1, r1Stride)
		w := sad8x8CompoundAvgBlockPureGo(src, srcStride, ref0, r0Stride, ref1, r1Stride)
		if g != w {
			t.Fatalf("trial %d: avx2 %d want %d", trial, g, w)
		}
	}
}
