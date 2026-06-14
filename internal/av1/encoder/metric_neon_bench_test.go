//go:build arm64 && !purego

package encoder

import (
	"math/rand"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/dsp/cpu"
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

func BenchmarkPixelStats4x4Scalar(b *testing.B) {
	benchPixelStats(b, pixelStats4x4PureGo)
}

func BenchmarkPixelStats4x4NEON(b *testing.B) {
	benchPixelStats(b, pixelStats4x4NEON)
}

func BenchmarkPixelStats8x4Scalar(b *testing.B) {
	benchPixelStats(b, pixelStats8x4PureGo)
}

func BenchmarkPixelStats8x4NEON(b *testing.B) {
	benchPixelStats(b, pixelStats8x4NEON)
}

func BenchmarkPixelStats4x8Scalar(b *testing.B) {
	benchPixelStats(b, pixelStats4x8PureGo)
}

func BenchmarkPixelStats4x8NEON(b *testing.B) {
	benchPixelStats(b, pixelStats4x8NEON)
}

func BenchmarkPixelStats16x8Scalar(b *testing.B) {
	benchPixelStats(b, pixelStats16x8PureGo)
}

func BenchmarkPixelStats16x8NEON(b *testing.B) {
	benchPixelStats(b, pixelStats16x8NEON)
}

func BenchmarkPixelStats8x16Scalar(b *testing.B) {
	benchPixelStats(b, pixelStats8x16PureGo)
}

func BenchmarkPixelStats8x16NEON(b *testing.B) {
	benchPixelStats(b, pixelStats8x16NEON)
}

func BenchmarkPixelStats16x4Scalar(b *testing.B) {
	benchPixelStats(b, pixelStats16x4PureGo)
}

func BenchmarkPixelStats16x4NEON(b *testing.B) {
	benchPixelStats(b, pixelStats16x4NEON)
}

func BenchmarkPixelStats4x16Scalar(b *testing.B) {
	benchPixelStats(b, pixelStats4x16PureGo)
}

func BenchmarkPixelStats4x16NEON(b *testing.B) {
	benchPixelStats(b, pixelStats4x16NEON)
}

func BenchmarkPixelStats16x16Scalar(b *testing.B) {
	benchPixelStats(b, pixelStats16x16PureGo)
}

func BenchmarkPixelStats16x16NEON(b *testing.B) {
	benchPixelStats(b, pixelStats16x16NEON)
}

func BenchmarkPixelStats32x8Scalar(b *testing.B) {
	benchPixelStats(b, pixelStats32x8PureGo)
}

func BenchmarkPixelStats32x8NEON(b *testing.B) {
	benchPixelStats(b, pixelStats32x8NEON)
}

func BenchmarkPixelStats8x32Scalar(b *testing.B) {
	benchPixelStats(b, pixelStats8x32PureGo)
}

func BenchmarkPixelStats8x32NEON(b *testing.B) {
	benchPixelStats(b, pixelStats8x32NEON)
}

func BenchmarkPixelStats32x16Scalar(b *testing.B) {
	benchPixelStats(b, pixelStats32x16PureGo)
}

func BenchmarkPixelStats32x16NEON(b *testing.B) {
	benchPixelStats(b, pixelStats32x16NEON)
}

func BenchmarkPixelStats16x32Scalar(b *testing.B) {
	benchPixelStats(b, pixelStats16x32PureGo)
}

func BenchmarkPixelStats16x32NEON(b *testing.B) {
	benchPixelStats(b, pixelStats16x32NEON)
}

func BenchmarkPixelStats32x32Scalar(b *testing.B) {
	benchPixelStats(b, pixelStats32x32PureGo)
}

func BenchmarkPixelStats32x32NEON(b *testing.B) {
	benchPixelStats(b, pixelStats32x32NEON)
}

func BenchmarkPixelStats64x16Scalar(b *testing.B) {
	benchPixelStats(b, pixelStats64x16PureGo)
}

func BenchmarkPixelStats64x16NEON(b *testing.B) {
	benchPixelStats(b, pixelStats64x16NEON)
}

func BenchmarkPixelStats16x64Scalar(b *testing.B) {
	benchPixelStats(b, pixelStats16x64PureGo)
}

func BenchmarkPixelStats16x64NEON(b *testing.B) {
	benchPixelStats(b, pixelStats16x64NEON)
}

func BenchmarkPixelStats64x32Scalar(b *testing.B) {
	benchPixelStats(b, pixelStats64x32PureGo)
}

func BenchmarkPixelStats64x32NEON(b *testing.B) {
	benchPixelStats(b, pixelStats64x32NEON)
}

func BenchmarkPixelStats32x64Scalar(b *testing.B) {
	benchPixelStats(b, pixelStats32x64PureGo)
}

func BenchmarkPixelStats32x64NEON(b *testing.B) {
	benchPixelStats(b, pixelStats32x64NEON)
}

func BenchmarkPixelStatsDotProd(b *testing.B) {
	if !cpu.Detected.DOTPROD {
		b.Skip("DOTPROD not detected")
	}
	for _, tc := range []struct {
		name string
		neon func([]byte, int, []byte, int) (uint32, int32)
		dp   func([]byte, int, []byte, int) (uint32, int32)
	}{
		{"8x4", pixelStats8x4NEON, pixelStats8x4DotProd},
		{"8x8", pixelStats8x8NEON, func(src []byte, srcStride int, ref []byte, refStride int) (uint32, int32) {
			return pixelStatsDotProd(src, srcStride, ref, refStride, 8, 8)
		}},
		{"16x4", pixelStats16x4NEON, pixelStats16x4DotProd},
		{"16x8", pixelStats16x8NEON, pixelStats16x8DotProd},
		{"8x16", pixelStats8x16NEON, pixelStats8x16DotProd},
		{"16x16", pixelStats16x16NEON, pixelStats16x16DotProd},
		{"32x8", pixelStats32x8NEON, pixelStats32x8DotProd},
		{"8x32", pixelStats8x32NEON, pixelStats8x32DotProd},
		{"32x16", pixelStats32x16NEON, pixelStats32x16DotProd},
		{"16x32", pixelStats16x32NEON, pixelStats16x32DotProd},
		{"32x32", pixelStats32x32NEON, pixelStats32x32DotProd},
		{"64x16", pixelStats64x16NEON, pixelStats64x16DotProd},
		{"16x64", pixelStats16x64NEON, pixelStats16x64DotProd},
		{"64x32", pixelStats64x32NEON, pixelStats64x32DotProd},
		{"32x64", pixelStats32x64NEON, pixelStats32x64DotProd},
	} {
		b.Run(tc.name+"/NEON", func(b *testing.B) {
			benchPixelStats(b, tc.neon)
		})
		b.Run(tc.name+"/DOTPROD", func(b *testing.B) {
			benchPixelStats(b, tc.dp)
		})
	}
}

func BenchmarkSSEVarianceDotProd(b *testing.B) {
	if !cpu.Detected.DOTPROD {
		b.Skip("DOTPROD not detected")
	}
	for _, tc := range []struct {
		name  string
		shift uint
		neon  func([]byte, int, []byte, int) (uint32, int32)
		dp    func([]byte, int, []byte, int) (uint32, int32)
	}{
		{"8x4", 5, pixelStats8x4NEON, pixelStats8x4DotProd},
		{"8x8", 6, pixelStats8x8NEON, func(src []byte, srcStride int, ref []byte, refStride int) (uint32, int32) {
			return pixelStatsDotProd(src, srcStride, ref, refStride, 8, 8)
		}},
		{"16x4", 6, pixelStats16x4NEON, pixelStats16x4DotProd},
		{"16x8", 7, pixelStats16x8NEON, pixelStats16x8DotProd},
		{"8x16", 7, pixelStats8x16NEON, pixelStats8x16DotProd},
		{"16x16", 8, pixelStats16x16NEON, pixelStats16x16DotProd},
		{"32x8", 8, pixelStats32x8NEON, pixelStats32x8DotProd},
		{"8x32", 8, pixelStats8x32NEON, pixelStats8x32DotProd},
		{"32x16", 9, pixelStats32x16NEON, pixelStats32x16DotProd},
		{"16x32", 9, pixelStats16x32NEON, pixelStats16x32DotProd},
		{"32x32", 10, pixelStats32x32NEON, pixelStats32x32DotProd},
		{"64x16", 10, pixelStats64x16NEON, pixelStats64x16DotProd},
		{"16x64", 10, pixelStats16x64NEON, pixelStats16x64DotProd},
		{"64x32", 11, pixelStats64x32NEON, pixelStats64x32DotProd},
		{"32x64", 11, pixelStats32x64NEON, pixelStats32x64DotProd},
	} {
		b.Run(tc.name+"/NEON", func(b *testing.B) {
			benchSSEVarianceFromStats(b, tc.neon, tc.shift)
		})
		b.Run(tc.name+"/DOTPROD", func(b *testing.B) {
			benchSSEVarianceFromStats(b, tc.dp, tc.shift)
		})
	}
}

func benchSSEVarianceFromStats(b *testing.B, fn func([]byte, int, []byte, int) (uint32, int32), shift uint) {
	benchSSEVariance(b, func(src []byte, srcStride int, ref []byte, refStride int) (uint32, uint32) {
		sse, sum := fn(src, srcStride, ref, refStride)
		return sse, varianceFromStats(sse, sum, shift)
	})
}

func BenchmarkSSEVariance4x4Scalar(b *testing.B) {
	benchSSEVariance(b, func(src []byte, srcStride int, ref []byte, refStride int) (uint32, uint32) {
		sse, sum := pixelStatsPureGo(src, srcStride, ref, refStride, 4, 4)
		return sse, varianceFromStats(sse, sum, 4)
	})
}

func BenchmarkSSEVariance4x4NEON(b *testing.B) {
	benchSSEVariance(b, sseVariance4x4)
}

func BenchmarkSSEVariance8x4Scalar(b *testing.B) {
	benchSSEVariance(b, func(src []byte, srcStride int, ref []byte, refStride int) (uint32, uint32) {
		sse, sum := pixelStatsPureGo(src, srcStride, ref, refStride, 8, 4)
		return sse, varianceFromStats(sse, sum, 5)
	})
}

func BenchmarkSSEVariance8x4NEON(b *testing.B) {
	benchSSEVariance(b, sseVariance8x4)
}

func BenchmarkSSEVariance4x8Scalar(b *testing.B) {
	benchSSEVariance(b, func(src []byte, srcStride int, ref []byte, refStride int) (uint32, uint32) {
		sse, sum := pixelStatsPureGo(src, srcStride, ref, refStride, 4, 8)
		return sse, varianceFromStats(sse, sum, 5)
	})
}

func BenchmarkSSEVariance4x8NEON(b *testing.B) {
	benchSSEVariance(b, sseVariance4x8)
}

func BenchmarkSSEVariance16x8Scalar(b *testing.B) {
	benchSSEVariance(b, func(src []byte, srcStride int, ref []byte, refStride int) (uint32, uint32) {
		sse, sum := pixelStatsPureGo(src, srcStride, ref, refStride, 16, 8)
		return sse, varianceFromStats(sse, sum, 7)
	})
}

func BenchmarkSSEVariance16x8NEON(b *testing.B) {
	benchSSEVariance(b, sseVariance16x8)
}

func BenchmarkSSEVariance8x16Scalar(b *testing.B) {
	benchSSEVariance(b, func(src []byte, srcStride int, ref []byte, refStride int) (uint32, uint32) {
		sse, sum := pixelStatsPureGo(src, srcStride, ref, refStride, 8, 16)
		return sse, varianceFromStats(sse, sum, 7)
	})
}

func BenchmarkSSEVariance8x16NEON(b *testing.B) {
	benchSSEVariance(b, sseVariance8x16)
}

func BenchmarkSSEVariance16x4Scalar(b *testing.B) {
	benchSSEVariance(b, func(src []byte, srcStride int, ref []byte, refStride int) (uint32, uint32) {
		sse, sum := pixelStatsPureGo(src, srcStride, ref, refStride, 16, 4)
		return sse, varianceFromStats(sse, sum, 6)
	})
}

func BenchmarkSSEVariance16x4NEON(b *testing.B) {
	benchSSEVariance(b, sseVariance16x4)
}

func BenchmarkSSEVariance4x16Scalar(b *testing.B) {
	benchSSEVariance(b, func(src []byte, srcStride int, ref []byte, refStride int) (uint32, uint32) {
		sse, sum := pixelStatsPureGo(src, srcStride, ref, refStride, 4, 16)
		return sse, varianceFromStats(sse, sum, 6)
	})
}

func BenchmarkSSEVariance4x16NEON(b *testing.B) {
	benchSSEVariance(b, sseVariance4x16)
}

func BenchmarkSSEVariance32x8Scalar(b *testing.B) {
	benchSSEVariance(b, func(src []byte, srcStride int, ref []byte, refStride int) (uint32, uint32) {
		sse, sum := pixelStatsPureGo(src, srcStride, ref, refStride, 32, 8)
		return sse, varianceFromStats(sse, sum, 8)
	})
}

func BenchmarkSSEVariance32x8NEON(b *testing.B) {
	benchSSEVariance(b, sseVariance32x8)
}

func BenchmarkSSEVariance8x32Scalar(b *testing.B) {
	benchSSEVariance(b, func(src []byte, srcStride int, ref []byte, refStride int) (uint32, uint32) {
		sse, sum := pixelStatsPureGo(src, srcStride, ref, refStride, 8, 32)
		return sse, varianceFromStats(sse, sum, 8)
	})
}

func BenchmarkSSEVariance8x32NEON(b *testing.B) {
	benchSSEVariance(b, sseVariance8x32)
}

func BenchmarkSSEVariance32x16Scalar(b *testing.B) {
	benchSSEVariance(b, func(src []byte, srcStride int, ref []byte, refStride int) (uint32, uint32) {
		sse, sum := pixelStatsPureGo(src, srcStride, ref, refStride, 32, 16)
		return sse, varianceFromStats(sse, sum, 9)
	})
}

func BenchmarkSSEVariance32x16NEON(b *testing.B) {
	benchSSEVariance(b, sseVariance32x16)
}

func BenchmarkSSEVariance16x32Scalar(b *testing.B) {
	benchSSEVariance(b, func(src []byte, srcStride int, ref []byte, refStride int) (uint32, uint32) {
		sse, sum := pixelStatsPureGo(src, srcStride, ref, refStride, 16, 32)
		return sse, varianceFromStats(sse, sum, 9)
	})
}

func BenchmarkSSEVariance16x32NEON(b *testing.B) {
	benchSSEVariance(b, sseVariance16x32)
}

func BenchmarkSSEVariance64x16Scalar(b *testing.B) {
	benchSSEVariance(b, func(src []byte, srcStride int, ref []byte, refStride int) (uint32, uint32) {
		sse, sum := pixelStatsPureGo(src, srcStride, ref, refStride, 64, 16)
		return sse, varianceFromStats(sse, sum, 10)
	})
}

func BenchmarkSSEVariance64x16NEON(b *testing.B) {
	benchSSEVariance(b, sseVariance64x16)
}

func BenchmarkSSEVariance16x64Scalar(b *testing.B) {
	benchSSEVariance(b, func(src []byte, srcStride int, ref []byte, refStride int) (uint32, uint32) {
		sse, sum := pixelStatsPureGo(src, srcStride, ref, refStride, 16, 64)
		return sse, varianceFromStats(sse, sum, 10)
	})
}

func BenchmarkSSEVariance16x64NEON(b *testing.B) {
	benchSSEVariance(b, sseVariance16x64)
}

func BenchmarkSSEVariance64x32Scalar(b *testing.B) {
	benchSSEVariance(b, func(src []byte, srcStride int, ref []byte, refStride int) (uint32, uint32) {
		sse, sum := pixelStatsPureGo(src, srcStride, ref, refStride, 64, 32)
		return sse, varianceFromStats(sse, sum, 11)
	})
}

func BenchmarkSSEVariance64x32NEON(b *testing.B) {
	benchSSEVariance(b, sseVariance64x32)
}

func BenchmarkSSEVariance32x64Scalar(b *testing.B) {
	benchSSEVariance(b, func(src []byte, srcStride int, ref []byte, refStride int) (uint32, uint32) {
		sse, sum := pixelStatsPureGo(src, srcStride, ref, refStride, 32, 64)
		return sse, varianceFromStats(sse, sum, 11)
	})
}

func BenchmarkSSEVariance32x64NEON(b *testing.B) {
	benchSSEVariance(b, sseVariance32x64)
}

func BenchmarkSSEVariance64x64Scalar(b *testing.B) {
	benchSSEVariance(b, func(src []byte, srcStride int, ref []byte, refStride int) (uint32, uint32) {
		sse, sum := pixelStatsPureGo(src, srcStride, ref, refStride, 64, 64)
		return sse, varianceFromStats(sse, sum, 12)
	})
}

func BenchmarkSSEVariance64x64NEON(b *testing.B) {
	benchSSEVariance(b, sseVariance64x64)
}

func benchSSEVariance(b *testing.B, fn func([]byte, int, []byte, int) (uint32, uint32)) {
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
