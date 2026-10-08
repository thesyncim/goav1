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
func TestForwardDCT8x8SIMDBinding(t *testing.T) {
	if !archsimd.X86.AVX2() {
		t.Skip("host has no AVX2")
	}
	fn := runtime.FuncForPC(reflect.ValueOf(forwardDCT8x8Impl).Pointer())
	if fn == nil || !strings.Contains(fn.Name(), "forwardDCT8x8SIMDGuarded") {
		t.Fatalf("8x8 forward DCT bound to %v, want forwardDCT8x8SIMDGuarded", fn)
	}
}
