// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && amd64 && !purego

package loopfilter

import (
	"reflect"
	"runtime"
	"simd/archsimd"
	"testing"
)

// TestFilterSIMDDispatchBoundAMD64 checks that the amd64 dispatch slots resolve
// to the Go-native SIMD kernels whenever the CPU reports AVX2. Rosetta does not
// advertise AVX2, so the check skips there; the direct kernel tests still run.
func TestFilterSIMDDispatchBoundAMD64(t *testing.T) {
	if !archsimd.X86.AVX2() {
		t.Skip("AVX2 not reported by CPUID; dispatch stays on pure Go")
	}
	nameOf := func(v interface{}) string {
		return runtime.FuncForPC(reflect.ValueOf(v).Pointer()).Name()
	}
	checks := []struct {
		name      string
		got, want interface{}
	}{
		{"filter4EdgeImpl", filter4EdgeImpl, filter4EdgeSIMD},
		{"filter4Edge16Impl", filter4Edge16Impl, filter4Edge16SIMD},
		{"filter6EdgeImpl", filter6EdgeImpl, filter6EdgeSIMD},
		{"filter8EdgeImpl", filter8EdgeImpl, filter8EdgeSIMD},
		{"filter14EdgeImpl", filter14EdgeImpl, filter14EdgeSIMD},
	}
	for _, c := range checks {
		if got, want := nameOf(c.got), nameOf(c.want); got != want {
			t.Errorf("%s = %s, want %s", c.name, got, want)
		}
	}
}
