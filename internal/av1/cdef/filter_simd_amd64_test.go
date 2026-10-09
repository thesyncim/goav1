// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && amd64 && !purego

package cdef

import (
	"reflect"
	"simd/archsimd"
	"testing"
)

// TestFilterBlockSIMDDispatchBound asserts that the AVX2 Go SIMD kernel is the
// one resolved into the dispatch slot whenever the CPU advertises AVX2. The
// bit-exactness of the kernel itself is covered by TestFilterBlockSIMDMatchesPureGo,
// which calls it directly so it runs regardless of feature detection.
func TestFilterBlockSIMDDispatchBound(t *testing.T) {
	if !archsimd.X86.AVX2() {
		t.Skip("CPU does not advertise AVX2")
	}
	if got, want := reflect.ValueOf(filterBlockImpl).Pointer(), reflect.ValueOf(filterBlockSIMD).Pointer(); got != want {
		t.Fatalf("filterBlockImpl is not bound to the AVX2 Go SIMD kernel")
	}
	if got, want := reflect.ValueOf(filterBlockU8Impl).Pointer(), reflect.ValueOf(filterBlockU8SIMD).Pointer(); got != want {
		t.Fatalf("filterBlockU8Impl is not bound to the AVX2 Go SIMD kernel")
	}
}
