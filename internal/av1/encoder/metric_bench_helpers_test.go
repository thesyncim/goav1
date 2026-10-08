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

func BenchmarkPixelStats8x4Scalar(b *testing.B) {
	benchPixelStats(b, pixelStats8x4PureGo)
}

func BenchmarkPixelStats4x8Scalar(b *testing.B) {
	benchPixelStats(b, pixelStats4x8PureGo)
}

func BenchmarkPixelStats16x8Scalar(b *testing.B) {
	benchPixelStats(b, pixelStats16x8PureGo)
}

func BenchmarkPixelStats8x16Scalar(b *testing.B) {
	benchPixelStats(b, pixelStats8x16PureGo)
}

func BenchmarkPixelStats16x4Scalar(b *testing.B) {
	benchPixelStats(b, pixelStats16x4PureGo)
}

func BenchmarkPixelStats4x16Scalar(b *testing.B) {
	benchPixelStats(b, pixelStats4x16PureGo)
}

func BenchmarkPixelStats16x16Scalar(b *testing.B) {
	benchPixelStats(b, pixelStats16x16PureGo)
}

func BenchmarkPixelStats32x8Scalar(b *testing.B) {
	benchPixelStats(b, pixelStats32x8PureGo)
}

func BenchmarkPixelStats8x32Scalar(b *testing.B) {
	benchPixelStats(b, pixelStats8x32PureGo)
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

func BenchmarkPixelStats64x16Scalar(b *testing.B) {
	benchPixelStats(b, pixelStats64x16PureGo)
}

func BenchmarkPixelStats16x64Scalar(b *testing.B) {
	benchPixelStats(b, pixelStats16x64PureGo)
}

func BenchmarkPixelStats64x32Scalar(b *testing.B) {
	benchPixelStats(b, pixelStats64x32PureGo)
}

func BenchmarkPixelStats32x64Scalar(b *testing.B) {
	benchPixelStats(b, pixelStats32x64PureGo)
}

func BenchmarkSATDCoeffs16Scalar(b *testing.B) {
	benchSATDCoeffs(b, satdCoeffsPureGo, 16)
}

func BenchmarkSATDCoeffs64Scalar(b *testing.B) {
	benchSATDCoeffs(b, satdCoeffsPureGo, 64)
}

func BenchmarkSATDCoeffs256Scalar(b *testing.B) {
	benchSATDCoeffs(b, satdCoeffsPureGo, 256)
}

func BenchmarkSATDCoeffs1024Scalar(b *testing.B) {
	benchSATDCoeffs(b, satdCoeffsPureGo, 1024)
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
