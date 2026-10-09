// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package restoration

import "github.com/thesyncim/goav1/internal/av1/dsp/cpu"

// init binds the Go-native SIMD Wiener passes on arm64 (NEON is mandatory on
// every arm64 chip Go runs on). The assignment happens once, before any decoder
// goroutine starts, so the steady-state cost is a single indirect call. Every
// binding is byte-identical to the pure-Go reference; the wrappers fall back to
// it for widths that are not 8-aligned.
func init() {
	_ = cpu.Detected // ensure cpu package init runs before this point
	if cpu.Detected.NEON {
		wienerHorizontalImpl = wienerHorizontalSIMD
		wienerHorizontalTrustedImpl = wienerHorizontalSIMDTrusted
		wienerVerticalImpl = wienerVerticalSIMD
		wienerHorizontalU8Impl = wienerHorizontalU8SIMDChecked
		wienerVerticalU8Impl = wienerVerticalU8SIMD
		return
	}
	wienerHorizontalImpl = wienerHorizontal
	wienerHorizontalTrustedImpl = wienerHorizontalTrusted
	wienerVerticalImpl = wienerVertical
	wienerHorizontalU8Impl = wienerHorizontalU8
	wienerVerticalU8Impl = wienerVerticalU8
}

// wienerSIMDDetected reports whether the Go-native SIMD Wiener kernels are bound
// on this host (NEON is mandatory on arm64).
func wienerSIMDDetected() bool { return cpu.Detected.NEON }
