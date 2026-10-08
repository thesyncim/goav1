package encoder

import "testing"

// Benchmarks run through the dispatch vars, so they measure the kernel bound
// for the current build (Go SIMD under GOEXPERIMENT=simd, scalar otherwise).

func benchResidualDispatch(b *testing.B, w, h int) {
	stride := w + 8
	src := make([]byte, stride*h)
	pred := make([]byte, w*h)
	for i := range src {
		src[i] = byte(i * 7)
	}
	for i := range pred {
		pred[i] = byte(i * 3)
	}
	dst := make([]int16, w*h)
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		residualBlockImpl(dst, src, 0, stride, pred, w, w, h)
	}
}

func benchRDStatsDispatch(b *testing.B, n int) {
	tran := make([]int32, n)
	qcoeff := make([]int16, n)
	for i := range tran {
		tran[i] = int32(i*37 - 4000)
		qcoeff[i] = int16(i%7 - 3)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		_, _, _, _ = rdStatsBlockImpl(tran, qcoeff, n, 21, 1)
	}
}

func benchBlockErrorDispatch(b *testing.B, n int) {
	coeff := make([]int32, n)
	dq := make([]int32, n)
	for i := range coeff {
		coeff[i] = int32(i*13 - 500)
		dq[i] = int32(i*11 - 400)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		_, _ = blockErrorImpl(coeff, dq, n)
	}
}

func BenchmarkResidualDispatch32x32(b *testing.B) { benchResidualDispatch(b, 32, 32) }
func BenchmarkResidualDispatch16x16(b *testing.B) { benchResidualDispatch(b, 16, 16) }
func BenchmarkResidualDispatch8x8(b *testing.B)   { benchResidualDispatch(b, 8, 8) }
func BenchmarkRDStatsDispatch1024(b *testing.B)   { benchRDStatsDispatch(b, 1024) }
func BenchmarkRDStatsDispatch256(b *testing.B)    { benchRDStatsDispatch(b, 256) }
func BenchmarkRDStatsDispatch64(b *testing.B)     { benchRDStatsDispatch(b, 64) }
func BenchmarkBlockErrorDispatch1024(b *testing.B) { benchBlockErrorDispatch(b, 1024) }
func BenchmarkBlockErrorDispatch256(b *testing.B)  { benchBlockErrorDispatch(b, 256) }
func BenchmarkBlockErrorDispatch64(b *testing.B)   { benchBlockErrorDispatch(b, 64) }
