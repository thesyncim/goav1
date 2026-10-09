// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && amd64 && !purego

package transform

import (
	"math/rand"
	"reflect"
	"runtime"
	"simd/archsimd"
	"strings"
	"testing"
)

// TestForwardDCT8x8SIMDMatchesPureGo proves the AVX2 8x8 kernel bit-exact with
// the portable reference across random 8-bit residual ranges and strides. It
// calls the kernel directly so hosts whose CPUID hides AVX2 (Rosetta) still
// exercise the VEX path.
func TestForwardDCT8x8SIMDMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(67))
	const resStride, coeffStride = 23, 17
	residual := make([]int16, resStride*16)
	for trial := range 3000 {
		for i := range residual {
			residual[i] = int16(rng.Intn(511)) - 255
		}
		want := make([]int32, coeffStride*16)
		got := make([]int32, coeffStride*16)
		forwardDCT8x8PureGo(want, coeffStride, residual, resStride)
		forwardDCT8x8SIMD(got, coeffStride, residual, resStride)
		for i := range want {
			if want[i] != got[i] {
				t.Fatalf("trial %d: coeff[%d] simd %d want %d", trial, i, got[i], want[i])
			}
		}
	}
}

// TestForwardDCT8x8SIMDBinding checks that the dispatcher selects the AVX2
// kernel whenever the host advertises AVX2.
func TestForwardDCTSIMDBindingsAMD64(t *testing.T) {
	if !archsimd.X86.AVX2() {
		t.Skip("host has no AVX2")
	}
	check := func(name string, got any, want string) {
		t.Helper()
		fn := runtime.FuncForPC(reflect.ValueOf(got).Pointer())
		if fn == nil || !strings.Contains(fn.Name(), want) {
			t.Fatalf("%s forward DCT bound to %v, want %s", name, fn, want)
		}
	}
	check("4x4", forwardDCT4x4Impl, "forwardDCT4x4SIMD")
	check("8x8", forwardDCT8x8Impl, "forwardDCT8x8SIMDGuarded")
	check("16x16", forwardDCT16x16Impl, "forwardDCT16x16SIMDGuarded")
	check("32x32", forwardDCT32x32Impl, "forwardDCT32x32SIMDGuarded")
}

// TestForwardDCT4x4SIMDMatchesPureGo proves the AVX2 4x4 kernel bit-exact with
// the portable reference over random residuals across the guard range and
// beyond it (the guarded entry routes out-of-range inputs to the scalar path).
func TestForwardDCT4x4SIMDMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(71))
	const resStride, coeffStride = 13, 9
	residual := make([]int16, resStride*4)
	for trial := range 3000 {
		span := 511
		if trial%2 == 1 {
			span = 4097
		}
		for i := range residual {
			residual[i] = int16(rng.Intn(span) - span/2)
		}
		want := make([]int32, coeffStride*4)
		got := make([]int32, coeffStride*4)
		forwardDCT4x4PureGo(want, coeffStride, residual, resStride)
		forwardDCT4x4SIMD(got, coeffStride, residual, resStride)
		for i := range want {
			if want[i] != got[i] {
				t.Fatalf("trial %d: coeff[%d] simd %d want %d", trial, i, got[i], want[i])
			}
		}
	}
}
