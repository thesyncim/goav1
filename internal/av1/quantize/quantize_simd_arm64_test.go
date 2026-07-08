//go:build goexperiment.simd && arm64 && !purego

package quantize

import (
	"math/rand"
	"reflect"
	"runtime"
	"strings"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/dsp/cpu"
)

func TestQuantizeSIMDDispatchBindings(t *testing.T) {
	if !cpu.Detected.NEON {
		t.Skip("NEON not detected")
	}
	requireQuantizeBinding(t, "quantizeBlockImpl", quantizeBlockImpl, "quantizeBlockSIMD")
	requireQuantizeBinding(t, "quantizeFPBlockImpl", quantizeFPBlockImpl, "quantizeFPBlockNEON")
	requireQuantizeBinding(t, "quantizeBBlockImpl", quantizeBBlockImpl, "quantizeBBlockNEON")
	requireQuantizeBinding(t, "quantizeFPNoQMatrixImpl", quantizeFPNoQMatrixImpl, "quantizeFPNoQMatrixSIMD")
}

func requireQuantizeBinding(t *testing.T, name string, fn any, want string) {
	t.Helper()
	v := reflect.ValueOf(fn)
	if v.IsNil() {
		t.Fatalf("%s is nil", name)
	}
	got := runtime.FuncForPC(v.Pointer()).Name()
	if !strings.Contains(got, want) {
		t.Fatalf("%s bound to %s, want %s", name, got, want)
	}
}

func TestQuantizeFPBlockSIMDMatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(73))
	edges := []int32{
		-(1 << 20), -32768, -32767, -1, 0, 1, 32767, 32768, 1 << 20,
		// extremes past maxSafe: verify the scan-free clamp path matches scalar
		minInt32, maxInt32, 1 << 30, -(1 << 30), 1 << 28, -(1 << 28),
	}
	for _, n := range []int{4, 8, 16, 32} {
		for _, ts := range []uint8{0, 1, 2} {
			for trial := 0; trial < 200; trial++ {
				q := Quantizer{
					DC: int32(4 + rng.Intn(8000)),
					AC: int32(4 + rng.Intn(8000)),
				}
				coeff := make([]int32, n*n)
				for i := range coeff {
					if i < len(edges) {
						coeff[i] = edges[i]
					} else {
						coeff[i] = int32(rng.Intn(1<<22)) - 1<<21
					}
				}
				want := make([]int16, n*n)
				got := make([]int16, n*n)
				quantDC := int64(1<<16) / int64(q.DC)
				roundDC := roundPowerOfTwo((64*q.DC)>>7, ts)
				quantAC := int64(1<<16) / int64(q.AC)
				roundAC := roundPowerOfTwo((64*q.AC)>>7, ts)
				for i := range coeff {
					if i == 0 {
						want[i] = quantizeScalarFP(coeff[i], q.DC, quantDC, roundDC, ts)
					} else {
						want[i] = quantizeScalarFP(coeff[i], q.AC, quantAC, roundAC, ts)
					}
				}
				if !quantizeFPBlockSIMD(got, coeff, n, q, ts) {
					t.Fatalf("n=%d ts=%d: kernel refused", n, ts)
				}
				for i := range want {
					if want[i] != got[i] {
						t.Fatalf("n=%d ts=%d trial=%d q[%d] simd %d want %d coeff=%d dc=%v q=%+v",
							n, ts, trial, i, got[i], want[i], coeff[i], i == 0, q)
					}
				}
			}
		}
	}
}

func benchQuantizeBlock(b *testing.B, n int, fn func([]int16, []int32, int, Quantizer, uint8) bool) {
	rng := rand.New(rand.NewSource(79))
	q := Quantizer{DC: 107, AC: 130}
	coeff := make([]int32, n*n)
	for i := range coeff {
		coeff[i] = int32(rng.Intn(1<<18)) - 1<<17
	}
	qcoeff := make([]int16, n*n)
	b.ReportAllocs()
	b.SetBytes(int64(n * n * 4))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = fn(qcoeff, coeff, n, q, 1)
	}
}

func quantizeBlockScalarBench(qcoeff []int16, coeff []int32, n int, q Quantizer, txScale uint8) bool {
	count := n * n
	for i := range count {
		scale := q.AC
		if i == 0 {
			scale = q.DC
		}
		qcoeff[i] = quantizeScalar(coeff[i], scale, txScale)
	}
	return true
}

func BenchmarkQuantizeBlock16x16_Scalar(b *testing.B) {
	benchQuantizeBlock(b, 16, quantizeBlockScalarBench)
}
func BenchmarkQuantizeBlock16x16_SIMD(b *testing.B) { benchQuantizeBlock(b, 16, quantizeBlockSIMD) }
func BenchmarkQuantizeBlock32x32_Scalar(b *testing.B) {
	benchQuantizeBlock(b, 32, quantizeBlockScalarBench)
}
func BenchmarkQuantizeBlock32x32_SIMD(b *testing.B) { benchQuantizeBlock(b, 32, quantizeBlockSIMD) }

func benchQuantizeFPBlock(b *testing.B, n int, fn func([]int16, []int32, int, Quantizer, uint8) bool) {
	rng := rand.New(rand.NewSource(83))
	q := Quantizer{DC: 107, AC: 130}
	coeff := make([]int32, n*n)
	for i := range coeff {
		coeff[i] = int32(rng.Intn(1<<18)) - 1<<17
	}
	qcoeff := make([]int16, n*n)
	b.ReportAllocs()
	b.SetBytes(int64(n * n * 4))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = fn(qcoeff, coeff, n, q, 1)
	}
}

func quantizeFPBlockScalarBench(qcoeff []int16, coeff []int32, n int, q Quantizer, txScale uint8) bool {
	count := n * n
	quantDC := int64(1<<16) / int64(q.DC)
	roundDC := roundPowerOfTwo((64*q.DC)>>7, txScale)
	quantAC := int64(1<<16) / int64(q.AC)
	roundAC := roundPowerOfTwo((64*q.AC)>>7, txScale)
	for i := range count {
		if i == 0 {
			qcoeff[i] = quantizeScalarFP(coeff[i], q.DC, quantDC, roundDC, txScale)
		} else {
			qcoeff[i] = quantizeScalarFP(coeff[i], q.AC, quantAC, roundAC, txScale)
		}
	}
	return true
}

func BenchmarkQuantizeFPBlock16x16_Scalar(b *testing.B) {
	benchQuantizeFPBlock(b, 16, quantizeFPBlockScalarBench)
}
func BenchmarkQuantizeFPBlock16x16_SIMD(b *testing.B) {
	benchQuantizeFPBlock(b, 16, quantizeFPBlockSIMD)
}
func BenchmarkQuantizeFPBlock16x16_ASM(b *testing.B) {
	benchQuantizeFPBlock(b, 16, quantizeFPBlockNEON)
}
func BenchmarkQuantizeFPBlock32x32_Scalar(b *testing.B) {
	benchQuantizeFPBlock(b, 32, quantizeFPBlockScalarBench)
}
func BenchmarkQuantizeFPBlock32x32_SIMD(b *testing.B) {
	benchQuantizeFPBlock(b, 32, quantizeFPBlockSIMD)
}
func BenchmarkQuantizeFPBlock32x32_ASM(b *testing.B) {
	benchQuantizeFPBlock(b, 32, quantizeFPBlockNEON)
}

func benchQuantizeFPNoQMatrix(b *testing.B, count int, fn func([]int32, []int32, []int32, []int16, FPQuantizer) (int, bool)) {
	rng := rand.New(rand.NewSource(89))
	q := FPQuantizer{
		Quant:    [2]int16{840, 704},
		Dequant:  [2]int16{78, 93},
		Round:    [2]int16{39, 46},
		LogScale: 1,
	}
	coeff := make([]int32, count)
	for i := range coeff {
		coeff[i] = int32(rng.Intn(1<<18)) - 1<<17
	}
	scan := make([]int16, count)
	for i := range scan {
		scan[i] = int16(i)
	}
	qcoeff := make([]int32, count)
	dqcoeff := make([]int32, count)
	b.ReportAllocs()
	b.SetBytes(int64(count * 4))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = fn(qcoeff, dqcoeff, coeff, scan, q)
	}
}

func quantizeFPNoQMatrixScalarBench(qcoeff []int32, dqcoeff []int32, coeff []int32, scan []int16, q FPQuantizer) (int, bool) {
	rounding := [2]int32{
		roundPowerOfTwo(int32(q.Round[0]), q.LogScale),
		roundPowerOfTwo(int32(q.Round[1]), q.LogScale),
	}
	eob := 0
	for i, rawRC := range scan {
		rc := int(rawRC)
		idx := 0
		if rc != 0 {
			idx = 1
		}
		qc, dqc, nonzero := quantizeFPNoQMatrixScalarCoeff(coeff[rc], idx, q, rounding)
		qcoeff[rc] = qc
		dqcoeff[rc] = dqc
		if nonzero {
			eob = i + 1
		}
	}
	return eob, true
}

func BenchmarkQuantizeFPNoQMatrix256_Scalar(b *testing.B) {
	benchQuantizeFPNoQMatrix(b, 256, quantizeFPNoQMatrixScalarBench)
}
func BenchmarkQuantizeFPNoQMatrix256_SIMD(b *testing.B) {
	benchQuantizeFPNoQMatrix(b, 256, quantizeFPNoQMatrixSIMD)
}
func BenchmarkQuantizeFPNoQMatrix1024_Scalar(b *testing.B) {
	benchQuantizeFPNoQMatrix(b, 1024, quantizeFPNoQMatrixScalarBench)
}
func BenchmarkQuantizeFPNoQMatrix1024_SIMD(b *testing.B) {
	benchQuantizeFPNoQMatrix(b, 1024, quantizeFPNoQMatrixSIMD)
}
