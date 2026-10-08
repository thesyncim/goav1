// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && (arm64 || amd64) && !purego

package restoration

import (
	"reflect"
	"runtime"
	"testing"
)

func TestWienerDispatchUsesSIMD(t *testing.T) {
	if !wienerSIMDDetected() {
		t.Skip("Go SIMD detection failed on this host")
	}
	nameOf := func(v any) string {
		return runtime.FuncForPC(reflect.ValueOf(v).Pointer()).Name()
	}
	for _, slot := range []struct {
		name string
		got  any
		want any
	}{
		{"wienerHorizontalImpl", wienerHorizontalImpl, wienerHorizontalSIMD},
		{"wienerHorizontalTrustedImpl", wienerHorizontalTrustedImpl, wienerHorizontalSIMDTrusted},
		{"wienerVerticalImpl", wienerVerticalImpl, wienerVerticalSIMD},
		{"wienerHorizontalU8Impl", wienerHorizontalU8Impl, wienerHorizontalU8SIMDChecked},
		{"wienerVerticalU8Impl", wienerVerticalU8Impl, wienerVerticalU8SIMD},
	} {
		if got, want := nameOf(slot.got), nameOf(slot.want); got != want {
			t.Errorf("%s = %s, want %s", slot.name, got, want)
		}
	}
}
