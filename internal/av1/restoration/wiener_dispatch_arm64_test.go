// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego && !goexperiment.simd

package restoration

import (
	"reflect"
	"runtime"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/dsp/cpu"
)

// TestWienerVerticalNEONIsBound guards that outside the goexperiment.simd build
// the vertical dispatch slot keeps resolving to the hand-written NEON asm (on any
// arm64 chip Go runs on, where NEON is mandatory). The mirror-image check for the
// SIMD build lives in wiener_gosimd_arm64_test.go.
func TestWienerVerticalNEONIsBound(t *testing.T) {
	if !cpu.Detected.NEON {
		t.Skip("NEON not detected")
	}
	nameOf := func(v any) string {
		return runtime.FuncForPC(reflect.ValueOf(v).Pointer()).Name()
	}
	if got := nameOf(wienerVerticalImpl); got != nameOf(wienerVerticalNEON) {
		t.Fatalf("wienerVerticalImpl = %s, want wienerVerticalNEON", got)
	}
}
