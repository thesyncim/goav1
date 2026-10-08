//go:build goexperiment.simd && amd64 && !purego

package quantize

import (
	"reflect"
	"runtime"
	"simd/archsimd"
	"strings"
	"testing"
)

func quantizeSIMDSupported() bool {
	return archsimd.X86.AVX2()
}

func TestQuantizeSIMDDispatchBindingsAMD64(t *testing.T) {
	if !quantizeSIMDSupported() {
		t.Skip("AVX2 not available")
	}
	name := runtime.FuncForPC(reflect.ValueOf(quantizeFPBlockImpl).Pointer()).Name()
	if !strings.Contains(name, "quantizeFPBlockSIMD") {
		t.Fatalf("quantizeFPBlockImpl bound to %s, want quantizeFPBlockSIMD", name)
	}
}
