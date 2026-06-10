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

// TestSAD16x16ImplMatchesPureGo proves the 16x16 kernel bit-exact with the
// portable reference.
func TestSAD16x16ImplMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(43))
	const stride = 91
	src := make([]byte, stride*96)
	ref := make([]byte, stride*96)
	for i := range src {
		src[i] = uint8(rng.Intn(256))
		ref[i] = uint8(rng.Intn(256))
	}
	for range 2000 {
		off := rng.Intn(stride*64) + rng.Intn(stride-16)
		want := sad16x16PureGo(src[off:], ref[off:], stride)
		got := sad16x16Impl(src[off:], ref[off:], stride)
		if got != want {
			t.Fatalf("off %d: impl %d want %d", off, got, want)
		}
	}
}

// TestSAD8x8DualImplMatchesPureGo proves the two-stride kernel bit-exact
// with the portable reference.
func TestSAD8x8DualImplMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(47))
	const srcStride, refStride = 83, 19
	src := make([]byte, srcStride*64)
	ref := make([]byte, refStride*64)
	for i := range src {
		src[i] = uint8(rng.Intn(256))
	}
	for i := range ref {
		ref[i] = uint8(rng.Intn(256))
	}
	for range 2000 {
		so := rng.Intn(srcStride*48) + rng.Intn(srcStride-8)
		ro := rng.Intn(refStride*48) + rng.Intn(refStride-8)
		want := sad8x8DualPureGo(src[so:], srcStride, ref[ro:], refStride)
		got := sad8x8DualImpl(src[so:], srcStride, ref[ro:], refStride)
		if got != want {
			t.Fatalf("so %d ro %d: impl %d want %d", so, ro, got, want)
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
