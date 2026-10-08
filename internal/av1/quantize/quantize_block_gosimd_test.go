//go:build goexperiment.simd && (amd64 || arm64) && !purego

package quantize

import "testing"

// TestQuantizeBlockSIMDZeroAlloc asserts the fp and quantize-b vector block
// kernels, which both arches dispatch, do not allocate on the hot path.
func TestQuantizeBlockSIMDZeroAlloc(t *testing.T) {
	if !quantizeSIMDSupported() {
		t.Skip("SIMD quantize kernels not supported on this CPU")
	}
	q := Quantizer{DC: 107, AC: 130}
	coeff := make([]int32, 16*16)
	for i := range coeff {
		coeff[i] = int32(i*37 - 4096)
	}
	qcoeff := make([]int16, 16*16)
	if allocs := testing.AllocsPerRun(1000, func() {
		if !quantizeFPBlockSIMD(qcoeff, coeff, 16, q, 1) {
			t.Fatal("quantizeFPBlockSIMD refused")
		}
	}); allocs != 0 {
		t.Fatalf("quantizeFPBlockSIMD allocated %f objects/run, want 0", allocs)
	}
	if allocs := testing.AllocsPerRun(1000, func() {
		if !quantizeBBlockSIMD(qcoeff, coeff, 16, q, 1) {
			t.Fatal("quantizeBBlockSIMD refused")
		}
	}); allocs != 0 {
		t.Fatalf("quantizeBBlockSIMD allocated %f objects/run, want 0", allocs)
	}
}
