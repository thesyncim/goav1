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

// TestFilterNEONDispatchBound verifies that non-SIMD dispatch uses NEON for the
// retained kernels and scalar Go for the promoted HBD6/HBD14 SIMD kernels,
// whose replaced horizontal assembly bodies are no longer present.
func TestFilterNEONDispatchBound(t *testing.T) {
	nameOf := func(v interface{}) string {
		return runtime.FuncForPC(reflect.ValueOf(v).Pointer()).Name()
	}
	checks := []struct {
		name      string
		got, want interface{}
	}{
		{"filter6EdgeImpl", filter6EdgeImpl, filter6EdgeNEON},
		{"filter8EdgeImpl", filter8EdgeImpl, filter8EdgeNEON},
		{"filter14EdgeImpl", filter14EdgeImpl, filter14EdgeNEON},
		{"filter6Edge16Impl", filter6Edge16Impl, filter6Edge16PureGo},
		{"filter8Edge16Impl", filter8Edge16Impl, filter8Edge16NEON},
		{"filter14Edge16Impl", filter14Edge16Impl, filter14Edge16PureGoFallback},
	}
	for _, c := range checks {
		if got, want := nameOf(c.got), nameOf(c.want); got != want {
			t.Errorf("%s = %s, want %s", c.name, got, want)
		}
	}
}
