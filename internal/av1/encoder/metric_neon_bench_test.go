//go:build arm64 && !purego

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

func BenchmarkPixelStats8x8NEON(b *testing.B) {
	benchPixelStats(b, pixelStats8x8NEON)
}

func BenchmarkPixelStats16x16Scalar(b *testing.B) {
	benchPixelStats(b, pixelStats16x16PureGo)
}

func BenchmarkPixelStats16x16NEON(b *testing.B) {
	benchPixelStats(b, pixelStats16x16NEON)
}

func BenchmarkPixelStats32x32Scalar(b *testing.B) {
	benchPixelStats(b, pixelStats32x32PureGo)
}

func BenchmarkPixelStats32x32NEON(b *testing.B) {
	benchPixelStats(b, pixelStats32x32NEON)
}

func BenchmarkSSEVariance64x64Scalar(b *testing.B) {
	benchSSEVariance64x64(b, func(src []byte, srcStride int, ref []byte, refStride int) (uint32, uint32) {
		sse, sum := pixelStatsPureGo(src, srcStride, ref, refStride, 64, 64)
		return sse, varianceFromStats(sse, sum, 12)
	})
}

func BenchmarkSSEVariance64x64NEON(b *testing.B) {
	benchSSEVariance64x64(b, sseVariance64x64)
}

func benchSSEVariance64x64(b *testing.B, fn func([]byte, int, []byte, int) (uint32, uint32)) {
	rng := rand.New(rand.NewSource(6104))
	const (
		srcStride = 1920
		refStride = 1920
		height    = 96
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
	var variance uint32
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sse, variance = fn(src, srcStride, ref, refStride)
	}
	if sse == 0 && variance == 0 {
		b.Fatal("unexpected zero metric")
	}
}

func BenchmarkSATDCoeffs16Scalar(b *testing.B) {
	benchSATDCoeffs(b, satdCoeffsPureGo, 16)
}

func BenchmarkSATDCoeffs16NEON(b *testing.B) {
	benchSATDCoeffs(b, satdCoeffsNEON, 16)
}

func BenchmarkSATDCoeffs64Scalar(b *testing.B) {
	benchSATDCoeffs(b, satdCoeffsPureGo, 64)
}

func BenchmarkSATDCoeffs64NEON(b *testing.B) {
	benchSATDCoeffs(b, satdCoeffsNEON, 64)
}

func BenchmarkSATDCoeffs256Scalar(b *testing.B) {
	benchSATDCoeffs(b, satdCoeffsPureGo, 256)
}

func BenchmarkSATDCoeffs256NEON(b *testing.B) {
	benchSATDCoeffs(b, satdCoeffsNEON, 256)
}

func BenchmarkSATDCoeffs1024Scalar(b *testing.B) {
	benchSATDCoeffs(b, satdCoeffsPureGo, 1024)
}

func BenchmarkSATDCoeffs1024NEON(b *testing.B) {
	benchSATDCoeffs(b, satdCoeffsNEON, 1024)
}

func benchSATDCoeffs(b *testing.B, fn func([]int32, int) int, count int) {
	rng := rand.New(rand.NewSource(6107))
	coeff := make([]int32, count)
	for i := range coeff {
		coeff[i] = int32(rng.Intn(65281) - 32640)
	}
	var satd int
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		satd = fn(coeff, count)
	}
	if satd == 0 {
		b.Fatal("unexpected zero SATD")
	}
}

func BenchmarkHadamard4x4Scalar(b *testing.B) {
	benchHadamard4x4(b, hadamard4x4PureGo)
}

func BenchmarkHadamard4x4NEON(b *testing.B) {
	benchHadamard4x4(b, hadamard4x4NEON)
}

func benchHadamard4x4(b *testing.B, fn func([]int16, int, []int32)) {
	rng := rand.New(rand.NewSource(6119))
	const (
		stride = 16
		height = 8
	)
	src := make([]int16, stride*height)
	for i := range src {
		src[i] = int16(rng.Intn(511) - 255)
	}
	src = src[stride+3:]
	var coeff [16]int32
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fn(src, stride, coeff[:])
	}
	if coeff[0] == 0 && coeff[1] == 0 {
		b.Fatal("unexpected zero Hadamard")
	}
}

func BenchmarkHadamard8x8Scalar(b *testing.B) {
	benchHadamard8x8(b, hadamard8x8PureGo)
}

func BenchmarkHadamard8x8NEON(b *testing.B) {
	benchHadamard8x8(b, hadamard8x8NEON)
}

func benchHadamard8x8(b *testing.B, fn func([]int16, int, []int32)) {
	rng := rand.New(rand.NewSource(6110))
	const (
		stride = 24
		height = 16
	)
	src := make([]int16, stride*height)
	for i := range src {
		src[i] = int16(rng.Intn(511) - 255)
	}
	src = src[stride+3:]
	var coeff [64]int32
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fn(src, stride, coeff[:])
	}
	if coeff[0] == 0 && coeff[1] == 0 {
		b.Fatal("unexpected zero Hadamard")
	}
}

func BenchmarkHadamard16x16Scalar(b *testing.B) {
	benchHadamard16x16(b, hadamard16x16PureGo)
}

func BenchmarkHadamard16x16NEON(b *testing.B) {
	benchHadamard16x16(b, hadamard16x16NEON)
}

func benchHadamard16x16(b *testing.B, fn func([]int16, int, []int32)) {
	rng := rand.New(rand.NewSource(6113))
	const (
		stride = 32
		height = 24
	)
	src := make([]int16, stride*height)
	for i := range src {
		src[i] = int16(rng.Intn(511) - 255)
	}
	src = src[stride+5:]
	var coeff [256]int32
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fn(src, stride, coeff[:])
	}
	if coeff[0] == 0 && coeff[1] == 0 {
		b.Fatal("unexpected zero Hadamard")
	}
}

func BenchmarkHadamard32x32Scalar(b *testing.B) {
	benchHadamard32x32(b, hadamard32x32PureGo)
}

func BenchmarkHadamard32x32NEON(b *testing.B) {
	benchHadamard32x32(b, hadamard32x32NEON)
}

func benchHadamard32x32(b *testing.B, fn func([]int16, int, []int32)) {
	rng := rand.New(rand.NewSource(6116))
	const (
		stride = 64
		height = 40
	)
	src := make([]int16, stride*height)
	for i := range src {
		src[i] = int16(rng.Intn(511) - 255)
	}
	src = src[stride+7:]
	var coeff [1024]int32
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fn(src, stride, coeff[:])
	}
	if coeff[0] == 0 && coeff[1] == 0 {
		b.Fatal("unexpected zero Hadamard")
	}
}
