//go:build goexperiment.simd && arm64 && !purego

package quantize

import (
	"reflect"
	"runtime"
	"strings"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/dsp/cpu"
)

func dequantSIMDSupported() bool {
	return cpu.Detected.NEON
}

func TestDequantColumnSIMDBinding(t *testing.T) {
	if !dequantSIMDSupported() {
		t.Skip("NEON not detected")
	}
	got := runtime.FuncForPC(reflect.ValueOf(dequantColumnImpl).Pointer()).Name()
	if !strings.Contains(got, "dequantColumnSIMD") {
		t.Fatalf("dequantColumnImpl bound to %s, want dequantColumnSIMD", got)
	}
}
