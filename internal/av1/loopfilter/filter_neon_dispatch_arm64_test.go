// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego && !goexperiment.simd

package loopfilter

import (
	"reflect"
	"runtime"
	"testing"
)

// TestFilterNEONDispatchBound is the FuncForPC probe for the non-SIMD build: with
// the goexperiment.simd tag off, the narrow and wide dispatch slots must resolve
// to the hand-written NEON asm (the Go-native SIMD kernels are not compiled in
// this build). Its SIMD-build counterpart is TestFilterSIMDDispatchBound.
func TestFilterNEONDispatchBound(t *testing.T) {
	nameOf := func(v interface{}) string {
		return runtime.FuncForPC(reflect.ValueOf(v).Pointer()).Name()
	}
	checks := []struct {
		name      string
		got, want interface{}
	}{
		{"filter4EdgeImpl", filter4EdgeImpl, filter4EdgeNEON},
		{"filter4Edge16Impl", filter4Edge16Impl, filter4Edge16NEON},
		{"filter6EdgeImpl", filter6EdgeImpl, filter6EdgeNEON},
		{"filter8EdgeImpl", filter8EdgeImpl, filter8EdgeNEON},
		{"filter14EdgeImpl", filter14EdgeImpl, filter14EdgeNEON},
		{"filter6Edge16Impl", filter6Edge16Impl, filter6Edge16NEON},
		{"filter8Edge16Impl", filter8Edge16Impl, filter8Edge16NEON},
		{"filter14Edge16Impl", filter14Edge16Impl, filter14Edge16NEON},
	}
	for _, c := range checks {
		if got, want := nameOf(c.got), nameOf(c.want); got != want {
			t.Errorf("%s = %s, want %s", c.name, got, want)
		}
	}
}
