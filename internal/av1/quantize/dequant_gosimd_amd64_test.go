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
