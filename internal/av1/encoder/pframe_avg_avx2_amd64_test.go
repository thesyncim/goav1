//go:build amd64 && !purego

package encoder

import (
	"math/rand"
	"testing"
)

// The AVX2 realtime-average tests call the kernels directly (never gated on
// cpu.Detected.AVX2) so hosts that hide AVX2 (Rosetta 2) still prove
// bit-exactness against the portable reference.
func TestRealtimeAvg8x8AVX2MatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0xA7A8))
	for trial := range 5000 {
		stride := 8 + rng.Intn(56)
		src := make([]byte, stride*8+8)
		fillBytes(rng, src)
		if g, w := realtimeAvg8x8AVX2(src, stride), realtimeAvg8x8PureGo(src, stride); g != w {
			t.Fatalf("trial %d stride %d: avx2 %d want %d", trial, stride, g, w)
		}
	}
}

func TestRealtimeAvg8x8QuadAVX2MatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0xA7A9))
	for trial := range 5000 {
		stride := 16 + rng.Intn(56)
		src := make([]byte, stride*16+16)
		fillBytes(rng, src)
		g0, g1, g2, g3 := realtimeAvg8x8QuadAVX2(src, stride)
		w0, w1, w2, w3 := realtimeAvg8x8QuadPureGo(src, stride)
		if g0 != w0 || g1 != w1 || g2 != w2 || g3 != w3 {
			t.Fatalf("trial %d stride %d: avx2 (%d,%d,%d,%d) want (%d,%d,%d,%d)",
				trial, stride, g0, g1, g2, g3, w0, w1, w2, w3)
		}
	}
}
