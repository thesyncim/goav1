//go:build arm64 && !purego

package encoder

import (
	"math/rand"
	"testing"
)

func benchResidual(b *testing.B, fn func([]int16, []byte, int, int, []byte, int, int, int), w, h int) {
	rng := rand.New(rand.NewSource(1))
	stride := 1920
	src := make([]byte, stride*64)
	pred := make([]byte, w*h)
	for i := range src {
		src[i] = byte(rng.Intn(256))
	}
	for i := range pred {
		pred[i] = byte(rng.Intn(256))
	}
	dst := make([]int16, w*h)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fn(dst, src, stride+3, stride, pred, w, w, h)
	}
}

func BenchmarkResidual32x32Scalar(b *testing.B) { benchResidual(b, residualBlockPureGo, 32, 32) }
func BenchmarkResidual32x32NEON(b *testing.B)   { benchResidual(b, residualBlockNEON, 32, 32) }
func BenchmarkResidual16x16Scalar(b *testing.B) { benchResidual(b, residualBlockPureGo, 16, 16) }
func BenchmarkResidual16x16NEON(b *testing.B)   { benchResidual(b, residualBlockNEON, 16, 16) }
func BenchmarkResidual8x8Scalar(b *testing.B)   { benchResidual(b, residualBlockPureGo, 8, 8) }
func BenchmarkResidual8x8NEON(b *testing.B)     { benchResidual(b, residualBlockNEON, 8, 8) }

func benchRDStats(b *testing.B, fn func([]int32, []int16, int, int32, uint8) (int64, int64, int64, bool), n int) {
	rng := rand.New(rand.NewSource(2))
	tran := make([]int32, n)
	qcoeff := make([]int16, n)
	for i := range tran {
		tran[i] = int32(rng.Intn(1<<16) - 1<<15)
		if rng.Intn(2) == 0 {
			qcoeff[i] = int16(rng.Intn(40) - 20)
		}
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fn(tran, qcoeff, n, 1365, 1)
	}
}

func BenchmarkRDStats1024Scalar(b *testing.B) { benchRDStats(b, rdStatsBlockPureGo, 1024) }
func BenchmarkRDStats1024NEON(b *testing.B)   { benchRDStats(b, rdStatsBlockNEON, 1024) }
func BenchmarkRDStats256Scalar(b *testing.B)  { benchRDStats(b, rdStatsBlockPureGo, 256) }
func BenchmarkRDStats256NEON(b *testing.B)    { benchRDStats(b, rdStatsBlockNEON, 256) }
func BenchmarkRDStats64Scalar(b *testing.B)   { benchRDStats(b, rdStatsBlockPureGo, 64) }
func BenchmarkRDStats64NEON(b *testing.B)     { benchRDStats(b, rdStatsBlockNEON, 64) }

func benchBlockError(b *testing.B, fn func([]int32, []int32, int) (int64, int64), n int) {
	rng := rand.New(rand.NewSource(3))
	coeff := make([]int32, n)
	dqcoeff := make([]int32, n)
	for i := range coeff {
		coeff[i] = int32(rng.Intn(1<<16) - 1<<15)
		dqcoeff[i] = int32(rng.Intn(1<<16) - 1<<15)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fn(coeff, dqcoeff, n)
	}
}

func BenchmarkBlockError1024Scalar(b *testing.B) { benchBlockError(b, blockErrorPureGo, 1024) }
func BenchmarkBlockError1024NEON(b *testing.B)   { benchBlockError(b, blockErrorNEON, 1024) }
func BenchmarkBlockError256Scalar(b *testing.B)  { benchBlockError(b, blockErrorPureGo, 256) }
func BenchmarkBlockError256NEON(b *testing.B)    { benchBlockError(b, blockErrorNEON, 256) }
func BenchmarkBlockError64Scalar(b *testing.B)   { benchBlockError(b, blockErrorPureGo, 64) }
func BenchmarkBlockError64NEON(b *testing.B)     { benchBlockError(b, blockErrorNEON, 64) }
