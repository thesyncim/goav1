// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package dsp

import (
	"reflect"
	"testing"
)

func TestSIMDDispatchBindsGoSIMDKernels(t *testing.T) {
	cases := []struct {
		name string
		got  any
		want any
	}{
		{"raw add", addRawTransformPlaneBlockImpl, addRawTransformPlaneBlockSIMD},
		{"residual add", addResidualPlaneBlockImpl, addResidualPlaneBlockSIMD},
	}
	for _, tc := range cases {
		if got, want := reflect.ValueOf(tc.got).Pointer(), reflect.ValueOf(tc.want).Pointer(); got != want {
			t.Errorf("%s dispatch points to %#x, want Go SIMD target %#x", tc.name, got, want)
		}
	}
}
