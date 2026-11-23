package encoder

import (
	"math/rand"
	"testing"
)

func benchPixelStats(b *testing.B, fn func([]byte, int, []byte, int) (uint32, int32)) {
	rng := rand.New(rand.NewSource(6103))
	const (
		srcStride = 1920
		refStride = 1920
		height    = 80
	)
	src := make([]byte, srcStride*height)
	ref := make([]byte, refStride*height)
	for i := range src {
		src[i] = byte(rng.Intn(256))
	}
	for i := range ref {
		ref[i] = byte(rng.Intn(256))
	}
	src = src[srcStride+3:]
	ref = ref[2*refStride+5:]
	var sse uint32
	var sum int32
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sse, sum = fn(src, srcStride, ref, refStride)
	}
	if sse == 0 && sum == 0 {
		b.Fatal("unexpected zero metric")
	}
}

func BenchmarkPixelStats8x8Scalar(b *testing.B) {
	benchPixelStats(b, pixelStats8x8PureGo)
}

func BenchmarkPixelStats4x4Scalar(b *testing.B) {
	benchPixelStats(b, pixelStats4x4PureGo)
}

func BenchmarkPixelStats16x8Scalar(b *testing.B) {
	benchPixelStats(b, pixelStats16x8PureGo)
}

func BenchmarkPixelStats8x16Scalar(b *testing.B) {
	benchPixelStats(b, pixelStats8x16PureGo)
}

func BenchmarkPixelStats16x16Scalar(b *testing.B) {
	benchPixelStats(b, pixelStats16x16PureGo)
}

func BenchmarkPixelStats32x16Scalar(b *testing.B) {
	benchPixelStats(b, pixelStats32x16PureGo)
}

func BenchmarkPixelStats16x32Scalar(b *testing.B) {
	benchPixelStats(b, pixelStats16x32PureGo)
}

func BenchmarkPixelStats32x32Scalar(b *testing.B) {
	benchPixelStats(b, pixelStats32x32PureGo)
}

func BenchmarkPixelStats64x32Scalar(b *testing.B) {
	benchPixelStats(b, pixelStats64x32PureGo)
}

func BenchmarkPixelStats32x64Scalar(b *testing.B) {
	benchPixelStats(b, pixelStats32x64PureGo)
}
