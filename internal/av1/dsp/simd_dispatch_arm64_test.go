// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package dsp

import (
	"reflect"
	"testing"
)

func TestSIMDDispatchKeepsMeasuredNEONKernels(t *testing.T) {
	cases := []struct {
		name string
		got  any
		want any
	}{
		{"blend", blendA64MaskImpl, blendA64MaskSIMD},
		{"raw add", addRawTransformPlaneBlockImpl, addRawTransformPlaneBlockNEON},
		{"minmax", minMaxAbsDiff8x8Impl, minMaxAbsDiff8x8SIMD},
	}
	for _, tc := range cases {
		if got, want := reflect.ValueOf(tc.got).Pointer(), reflect.ValueOf(tc.want).Pointer(); got != want {
			t.Errorf("%s dispatch points to %#x, want NEON target %#x", tc.name, got, want)
		}
	}
}

func TestResidualSIMDDispatchUsesWrapper(t *testing.T) {
	got := reflect.ValueOf(addResidualPlaneBlockImpl).Pointer()
	want := reflect.ValueOf(addResidualPlaneBlockSIMDDispatch).Pointer()
	if got != want {
		t.Fatalf("residual dispatch points to %#x, want SIMD width selector %#x", got, want)
	}
	if got == reflect.ValueOf(addResidualPlaneBlockNEON).Pointer() {
		t.Fatalf("residual dispatch bypasses SIMD width selector")
	}
}
