// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && (arm64 || amd64) && !purego

package restoration

// wienerHorizontalU8SIMDCanRun reports whether the 8-bit SIMD horizontal pass may
// run on this block. Its 16-byte window load for the last 8-column group reaches
// two samples past the three-sample Wiener border, so the source must hold that
// trailing pad; otherwise the caller uses the scalar reference.
func wienerHorizontalU8SIMDCanRun(srcLen int, srcStride int, srcOrigin int, width int, height int) bool {
	if width < 8 || width%8 != 0 {
		return false
	}
	loadWidth, ok := checkedAdd(width, 2)
	return ok && borderedBlockFits(srcLen, srcStride, srcOrigin, loadWidth, height, WienerHalfwin, WienerHalfwin)
}

// wienerHorizontalU8SIMDChecked is the dispatch entry for the 8-bit horizontal
// pass: the SIMD kernel when the padded load footprint is resident, else the
// scalar reference.
func wienerHorizontalU8SIMDChecked(src []uint8, srcStride int, srcOrigin int, width int, height int, filter WienerFilter, round0 int, temp []uint16) {
	if !wienerHorizontalU8SIMDCanRun(len(src), srcStride, srcOrigin, width, height) {
		wienerHorizontalU8(src, srcStride, srcOrigin, width, height, filter, round0, temp)
		return
	}
	wienerHorizontalU8SIMD(src, srcStride, srcOrigin, width, height, filter, round0, temp)
}
