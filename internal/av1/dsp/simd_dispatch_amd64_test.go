// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && amd64 && !purego

package dsp

import (
	"reflect"
	"simd/archsimd"
	"testing"
)

// TestSIMDDispatchBindsAVX2Kernels confirms the amd64 SIMD build binds the Go SIMD
// kernels whenever archsimd reports AVX2, and the pure-Go references otherwise.
func TestSIMDDispatchBindsAVX2Kernels(t *testing.T) {
	if !archsimd.X86.AVX2() {
		t.Skip("AVX2 not reported by archsimd on this host")
	}
	cases := []struct {
		name string
		got  any
		want any
	}{
		{"blend", blendA64MaskImpl, blendA64MaskSIMD},
		{"minmax", minMaxAbsDiff8x8Impl, minMaxAbsDiff8x8SIMD},
		{"raw add", addRawTransformPlaneBlockImpl, addRawTransformPlaneBlockSIMD},
		{"residual add", addResidualPlaneBlockImpl, addResidualPlaneBlockSIMD},
	}
	for _, tc := range cases {
		if got, want := reflect.ValueOf(tc.got).Pointer(), reflect.ValueOf(tc.want).Pointer(); got != want {
			t.Errorf("%s dispatch points to %#x, want AVX2 target %#x", tc.name, got, want)
		}
	}
}
