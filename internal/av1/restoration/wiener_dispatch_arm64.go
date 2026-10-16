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
		wienerHorizontalU8Impl = wienerHorizontalU8SIMDTaps
		wienerVerticalU8Impl = wienerVerticalU8SIMDTaps
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

// wienerFilterFiveTap reports a symmetric filter whose outer tap pair is zero
// (WienerFilter is [f0 f1 f2 f3 f2 f1 f0 0]): every chroma filter and most luma
// ones, dav1d's wiener_filter5 case.
func wienerFilterFiveTap(filter WienerFilter) bool {
	return filter[0] == 0 && filter[WienerSubpelTaps-2] == 0
}

// wienerHorizontalU8SIMDTaps and wienerVerticalU8SIMDTaps route five-tap
// filters to the kernels that skip the zero outer tap pair.
func wienerHorizontalU8SIMDTaps(src []uint8, srcStride int, srcOrigin int, width int, height int, filter WienerFilter, round0 int, temp []uint16) {
	if wienerFilterFiveTap(filter) {
		wienerHorizontalU8SIMD5(src, srcStride, srcOrigin, width, height, filter, round0, temp)
		return
	}
	wienerHorizontalU8SIMDChecked(src, srcStride, srcOrigin, width, height, filter, round0, temp)
}

func wienerVerticalU8SIMDTaps(temp []uint16, tempStride int, dst []uint8, dstStride int, width int, height int, filter WienerFilter, round1 int) {
	if wienerFilterFiveTap(filter) {
		wienerVerticalU8SIMD5(temp, tempStride, dst, dstStride, width, height, filter, round1)
		return
	}
	wienerVerticalU8SIMD(temp, tempStride, dst, dstStride, width, height, filter, round1)
}
