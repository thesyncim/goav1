//go:build goexperiment.simd && arm64 && !purego

package encoder

import (
	"math/rand"
	"reflect"
	"runtime"
	"strings"
	"testing"
)

// TestMetricSIMDBinding confirms the goexperiment.simd init bound the SATD and
// Hadamard dispatch vars to the archsimd ports (not the NEON asm or pure-Go),
// while the pixelStats kernels stay on NEON.
func TestMetricSIMDBinding(t *testing.T) {
	cases := []struct {
		name string
		fn   any
		want string
	}{
		{"satdCoeffsImpl", satdCoeffsImpl, "satdCoeffsSIMD"},
		{"hadamard4x4Impl", hadamard4x4Impl, "hadamard4x4SIMD"},
		{"hadamard8x8Impl", hadamard8x8Impl, "hadamard8x8SIMD"},
		{"hadamard16x16Impl", hadamard16x16Impl, "hadamard16x16SIMD"},
		{"hadamard32x32Impl", hadamard32x32Impl, "hadamard32x32SIMD"},
	}
	for _, tc := range cases {
		pc := reflect.ValueOf(tc.fn).Pointer()
		name := runtime.FuncForPC(pc).Name()
		if !strings.Contains(name, tc.want) {
			t.Errorf("%s bound to %q, want a %q function", tc.name, name, tc.want)
		}
	}
	// pixelStats stays on NEON (or DOTPROD) under the simd build.
	psName := runtime.FuncForPC(reflect.ValueOf(pixelStats8x8Impl).Pointer()).Name()
	if !strings.Contains(psName, "NEON") && !strings.Contains(psName, "DotProd") {
		t.Errorf("pixelStats8x8Impl bound to %q, want NEON/DotProd", psName)
	}
}

// TestMetricSIMDNoAlloc asserts the SATD/Hadamard SIMD kernels are zero-alloc.
func TestMetricSIMDNoAlloc(t *testing.T) {
	rng := rand.New(rand.NewSource(918))
	var src [64 * 64]int16
	for i := range src {
		src[i] = int16(rng.Intn(511) - 255)
	}
	var coeff [1024]int32
	for i := range coeff {
		coeff[i] = int32(rng.Intn(65281) - 32640)
	}

	checks := []struct {
		name string
		fn   func()
	}{
		{"satdCoeffs1024", func() { _ = satdCoeffsSIMD(coeff[:], 1024) }},
		{"hadamard4x4", func() { hadamard4x4SIMD(src[:], 64, coeff[:16]) }},
		{"hadamard8x8", func() { hadamard8x8SIMD(src[:], 64, coeff[:64]) }},
		{"hadamard16x16", func() { hadamard16x16SIMD(src[:], 64, coeff[:256]) }},
		{"hadamard32x32", func() { hadamard32x32SIMD(src[:], 64, coeff[:1024]) }},
	}
	for _, c := range checks {
		if a := testing.AllocsPerRun(100, c.fn); a != 0 {
			t.Errorf("%s: %.1f allocs/op, want 0", c.name, a)
		}
	}
}
