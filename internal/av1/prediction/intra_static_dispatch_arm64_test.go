// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego && !goexperiment.simd

package prediction

import (
	"reflect"
	"runtime"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/dsp/cpu"
)

// TestIntraStaticNEONIsBound guards that outside the goexperiment.simd build the
// PAETH/SMOOTH dispatch slots keep resolving to the hand-written NEON asm (NEON
// is mandatory on every arm64 chip Go runs on). The mirror-image check for the
// SIMD build lives in intra_static_gosimd_arm64_test.go.
func TestIntraStaticNEONIsBound(t *testing.T) {
	if !cpu.Detected.NEON {
		t.Skip("NEON not detected")
	}
	nameOf := func(v any) string {
		return runtime.FuncForPC(reflect.ValueOf(v).Pointer()).Name()
	}
	cases := []struct {
		name     string
		got      any
		wantFunc any
	}{
		{"paeth", predictPaethImpl, predictPaethNEON},
		{"smooth", predictSmoothImpl, predictSmoothNEON},
		{"smooth_v", predictSmoothVerticalImpl, predictSmoothVerticalNEON},
		{"smooth_h", predictSmoothHorizontalImpl, predictSmoothHorizontalNEON},
	}
	for _, c := range cases {
		if got, want := nameOf(c.got), nameOf(c.wantFunc); got != want {
			t.Errorf("%s dispatch = %s, want %s", c.name, got, want)
		}
	}
}
