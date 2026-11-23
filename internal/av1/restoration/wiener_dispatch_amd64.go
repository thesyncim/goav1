// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && amd64 && !purego

package restoration

import "simd/archsimd"

// init binds the Go-native SIMD Wiener passes on amd64 when AVX2 is available.
// Without AVX2 the dispatch keeps the pure-Go reference. The assignment happens
// once, before any decoder goroutine starts, so the steady-state cost is a single
// indirect call. Every binding is byte-identical to the pure-Go reference.
func init() {

	if archsimd.X86.AVX2() {
		wienerHorizontalTrustedImpl = wienerHorizontalSIMDTrusted
		wienerVerticalImpl = wienerVerticalSIMD
		wienerHorizontalU8Impl = wienerHorizontalU8SIMDChecked
		wienerVerticalU8Impl = wienerVerticalU8SIMD
		return
	}
	wienerHorizontalTrustedImpl = wienerHorizontalTrusted
	wienerVerticalImpl = wienerVertical
	wienerHorizontalU8Impl = wienerHorizontalU8
	wienerVerticalU8Impl = wienerVerticalU8
}
