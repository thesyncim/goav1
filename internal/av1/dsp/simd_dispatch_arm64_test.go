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
		name                   string
		dispatch, wantDispatch any
		neon                   any
	}{
		{"blend", blendA64MaskImpl, blendA64MaskSIMD, blendA64MaskNEON},
		{"residual add", addResidualPlaneBlockImpl, addResidualPlaneBlockSIMDDispatch, addResidualPlaneBlockNEON},
		{"raw add", addRawTransformPlaneBlockImpl, addRawTransformPlaneBlockSIMD, addRawTransformPlaneBlockNEON},
		{"minmax", minMaxAbsDiff8x8Impl, minMaxAbsDiff8x8SIMD, minMaxAbsDiff8x8NEON},
	}
	for _, tc := range cases {
		dispatchPC := reflect.ValueOf(tc.dispatch).Pointer()
		if wantPC := reflect.ValueOf(tc.wantDispatch).Pointer(); dispatchPC != wantPC {
			t.Errorf("%s dispatch points to %#x, want selected target %#x", tc.name, dispatchPC, wantPC)
		}
		if neonPC := reflect.ValueOf(tc.neon).Pointer(); dispatchPC == neonPC {
			t.Errorf("%s dispatch points to NEON benchmark target %#x", tc.name, neonPC)
		}
	}
}
