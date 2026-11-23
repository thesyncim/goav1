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
	requireQuantizeBinding(t, "quantizeFPBlockImpl", quantizeFPBlockImpl, "quantizeFPBlockSIMD")
	requireQuantizeBinding(t, "quantizeBBlockImpl", quantizeBBlockImpl, "quantizeBBlockSIMD")
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
func BenchmarkQuantizeBBlock16x16_SIMD(b *testing.B) {
	benchQuantizeFPBlock(b, 16, quantizeBBlockSIMD)
}
func BenchmarkQuantizeBBlock32x32_SIMD(b *testing.B) {
	benchQuantizeFPBlock(b, 32, quantizeBBlockSIMD)
}

func BenchmarkQuantizeFPBlock16x16_SIMD(b *testing.B) {
	benchQuantizeFPBlock(b, 16, quantizeFPBlockSIMD)
}
func BenchmarkQuantizeFPBlock32x32_Scalar(b *testing.B) {
	benchQuantizeFPBlock(b, 32, quantizeFPBlockScalarBench)
}
func BenchmarkQuantizeFPBlock32x32_SIMD(b *testing.B) {
	benchQuantizeFPBlock(b, 32, quantizeFPBlockSIMD)
}
