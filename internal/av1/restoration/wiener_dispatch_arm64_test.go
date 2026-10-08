// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

package restoration

import (
	"reflect"
	"runtime"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/dsp/cpu"
)

func TestArm64WienerDispatchUsesNEON(t *testing.T) {
	if !cpu.Detected.NEON {
		t.Skip("arm64 NEON not detected")
	}
	nameOf := func(v any) string {
		return runtime.FuncForPC(reflect.ValueOf(v).Pointer()).Name()
	}
	for _, slot := range []struct {
		name string
		got  any
		want any
	}{
		{"wienerHorizontalImpl", wienerHorizontalImpl, wienerHorizontalNEON},
		{"wienerHorizontalTrustedImpl", wienerHorizontalTrustedImpl, wienerHorizontalNEONTrusted},
		{"wienerVerticalImpl", wienerVerticalImpl, wienerVerticalNEON},
		{"wienerHorizontalU8Impl", wienerHorizontalU8Impl, wienerHorizontalU8NEON},
		{"wienerVerticalU8Impl", wienerVerticalU8Impl, wienerVerticalU8NEON},
	} {
		if got, want := nameOf(slot.got), nameOf(slot.want); got != want {
			t.Errorf("%s = %s, want %s", slot.name, got, want)
		}
	}
}
