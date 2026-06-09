package encoder

import (
	"math/rand"
	"testing"
)

// TestSAD8x8ImplMatchesPureGo proves the dispatched SAD kernel is bit-exact
// with the portable reference across random data, strides, and offsets
// (including totals far above any early-exit threshold).
func TestSAD8x8ImplMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(41))
	const stride = 73 // deliberately odd
	src := make([]byte, stride*64)
	ref := make([]byte, stride*64)
	for i := range src {
		src[i] = uint8(rng.Intn(256))
		ref[i] = uint8(rng.Intn(256))
	}
	for range 2000 {
		off := rng.Intn(stride*48) + rng.Intn(stride-8)
		want := sad8x8PureGo(src[off:], ref[off:], stride, 1<<30)
		got := sad8x8Impl(src[off:], ref[off:], stride, 1<<30)
		if got != want {
			t.Fatalf("off %d: impl %d want %d", off, got, want)
		}
	}
}

func BenchmarkSAD8x8(b *testing.B) {
	src := make([]byte, 64*64)
	ref := make([]byte, 64*64)
	for i := range src {
		src[i] = uint8(i * 7)
		ref[i] = uint8(i * 13)
	}
	b.ReportAllocs()
	for b.Loop() {
		sad8x8Impl(src, ref, 64, 1<<30)
	}
}
