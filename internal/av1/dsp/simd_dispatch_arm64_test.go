// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package dsp

import (
	"reflect"
	"testing"
)

func TestSIMDDispatchTargetsAreDistinctFromNEONBenchmarks(t *testing.T) {
	cases := []struct {
		name           string
		dispatch, simd any
		neon           any
	}{
		{"blend", blendA64MaskImpl, blendA64MaskSIMD, blendA64MaskNEON},
		{"residual add", addResidualPlaneBlockImpl, addResidualPlaneBlockSIMD, addResidualPlaneBlockNEON},
		{"raw add", addRawTransformPlaneBlockImpl, addRawTransformPlaneBlockSIMD, addRawTransformPlaneBlockNEON},
		{"minmax", minMaxAbsDiff8x8Impl, minMaxAbsDiff8x8SIMD, minMaxAbsDiff8x8NEON},
	}
	for _, tc := range cases {
		dispatchPC := reflect.ValueOf(tc.dispatch).Pointer()
		if simdPC := reflect.ValueOf(tc.simd).Pointer(); dispatchPC != simdPC {
			t.Errorf("%s dispatch points to %#x, want SIMD kernel %#x", tc.name, dispatchPC, simdPC)
		}
		if neonPC := reflect.ValueOf(tc.neon).Pointer(); dispatchPC == neonPC {
			t.Errorf("%s dispatch points to NEON benchmark target %#x", tc.name, neonPC)
		}
	}
}
