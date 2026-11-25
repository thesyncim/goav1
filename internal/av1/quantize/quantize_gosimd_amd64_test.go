//go:build goexperiment.simd && amd64 && !purego

package quantize

import (
	"reflect"
	"runtime"
	"simd/archsimd"
	"strings"
	"testing"
)

func dequantSIMDSupported() bool {
	return archsimd.X86.AVX2()
}

func TestDequantColumnSIMDBinding(t *testing.T) {
	if !dequantSIMDSupported() {
		t.Skip("AVX2 not available")
	}
	got := runtime.FuncForPC(reflect.ValueOf(dequantColumnImpl).Pointer()).Name()
	if !strings.Contains(got, "dequantColumnSIMD") {
		t.Fatalf("dequantColumnImpl bound to %s, want dequantColumnSIMD", got)
	}
}

func quantizeSIMDSupported() bool {
	return archsimd.X86.AVX2()
}

func TestQuantizeSIMDDispatchBindingsAMD64(t *testing.T) {
	if !quantizeSIMDSupported() {
		t.Skip("AVX2 not available")
	}
	for _, b := range []struct {
		name string
		fn   any
		want string
	}{
		{"quantizeFPBlockImpl", quantizeFPBlockImpl, "quantizeFPBlockSIMD"},
		{"quantizeBBlockImpl", quantizeBBlockImpl, "quantizeBBlockSIMD"},
	} {
		got := runtime.FuncForPC(reflect.ValueOf(b.fn).Pointer()).Name()
		if !strings.Contains(got, b.want) {
			t.Fatalf("%s bound to %s, want %s", b.name, got, b.want)
		}
	}
}
