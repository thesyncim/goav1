// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package frame

import (
	"simd/archsimd"
	"unsafe"
)

// loadSampleRows8 binds the Go SIMD 8-bit widen on arm64. NEON (AdvSIMD) is
// mandatory on arm64 so this is a static binding; see the dispatch note on
// loadSampleRows8PureGo for why it is not a func variable.
func loadSampleRows8(dst []uint16, dstStride int, src []byte, srcStride int, width int, height int) {
	loadSampleRows8SIMD(dst, dstStride, src, srcStride, width, height)
}

// loadSampleRows8SIMD widens 8-bit sample rows into uint16 staging storage, 16
// columns per iteration: one 16-byte load, the low and high eight bytes each
// zero-extended to eight uint16 lanes (USHLL/USHLL2 on arm64), and two 8-lane
// stores. The width
// remainder (< 16 columns) is finished by a scalar tail in the same row, so no
// vector access ever leaves the row and the output is bit-identical to
// loadSampleRows8PureGo.
//
// The vector body addresses the rows through raw pointers: the bounds check of
// the last row (below) covers every row, and each access is provably inside
// [0, width) of its row. Per-chunk slice bounds checks cost more than the widen
// itself and measured 4x slower than the asm kernel.
func loadSampleRows8SIMD(dst []uint16, dstStride int, src []byte, srcStride int, width int, height int) {
	if width <= 0 || height <= 0 {
		return
	}
	// Bounds contract identical to the pure-Go reference slices: the last
	// row's [0, width) region must be resident in both buffers.
	_ = src[(height-1)*srcStride+width-1]
	_ = dst[(height-1)*dstStride+width-1]
	vecWidth := width &^ 15
	srcOff := 0
	dstOff := 0
	for y := 0; y < height; y++ {
		sp := unsafe.Pointer(&src[srcOff])
		dp := unsafe.Pointer(&dst[dstOff])
		for x := 0; x < vecWidth; x += 16 {
			v := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(sp, x)))
			d := unsafe.Add(dp, 2*x)
			v.ExtendLo8ToUint16().StoreArray((*[8]uint16)(d))
			v.ConcatShiftBytesRight(v, 8).ExtendLo8ToUint16().StoreArray((*[8]uint16)(unsafe.Add(d, 16)))
		}
		for x := vecWidth; x < width; x++ {
			dst[dstOff+x] = uint16(src[srcOff+x])
		}
		srcOff += srcStride
		dstOff += dstStride
	}
}
