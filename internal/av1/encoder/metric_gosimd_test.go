//go:build goexperiment.simd && (arm64 || amd64) && !purego

package encoder

import (
	"reflect"
	"runtime"
	"strings"
	"testing"
)

// TestMetricSIMDBinding confirms the goexperiment.simd init bound the
// pixelStats dispatch to the pixelStatsSIMD wrappers.
func TestMetricSIMDBinding(t *testing.T) {
	if !gosimdKernelsSupported() {
		t.Skip("Go SIMD kernels unsupported on this CPU")
	}
	psName := runtime.FuncForPC(reflect.ValueOf(pixelStats8x8Impl).Pointer()).Name()
	if !strings.Contains(psName, "pixelStats8x8SIMD") {
		t.Errorf("pixelStats8x8Impl bound to %q, want pixelStats8x8SIMD", psName)
	}
}
