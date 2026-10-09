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
// zero-extended to eight uint16 lanes, and two 8-lane stores. A partial row
// ends with an overlapping 16- or 8-column group; widths below eight use
// scalar stores. No access leaves the row and output matches loadSampleRows8PureGo.
//
// The vector body addresses rows through raw pointers after checking the last
// row's extent. Every vector access stays inside [0, width) of its row.
func loadSampleRows8SIMD(dst []uint16, dstStride int, src []byte, srcStride int, width int, height int) {
	if width <= 0 || height <= 0 {
		return
	}
	// Bounds contract identical to the pure-Go reference slices: the last
	// row's [0, width) region must be resident in both buffers.
	_ = src[(height-1)*srcStride+width-1]
	_ = dst[(height-1)*dstStride+width-1]
	sp := unsafe.Pointer(unsafe.SliceData(src))
	dp := unsafe.Pointer(unsafe.SliceData(dst))
	dstByteStride := 2 * dstStride
	if width < 16 {
		if width == 8 {
			var word [2]uint64
			for y := 0; y < height; y++ {
				word[0] = *(*uint64)(sp)
				v := archsimd.LoadUint64x2Array(&word).ReshapeToUint8s()
				v.ExtendLo8ToUint16().StoreArray((*[8]uint16)(dp))
				if y+1 < height {
					sp = unsafe.Add(sp, srcStride)
					dp = unsafe.Add(dp, dstByteStride)
				}
			}
			return
		}
		if width > 8 {
			var word [2]uint64
			off := width - 8
			for y := 0; y < height; y++ {
				word[0] = *(*uint64)(sp)
				v := archsimd.LoadUint64x2Array(&word).ReshapeToUint8s()
				v.ExtendLo8ToUint16().StoreArray((*[8]uint16)(dp))
				word[0] = *(*uint64)(unsafe.Add(sp, off))
				v = archsimd.LoadUint64x2Array(&word).ReshapeToUint8s()
				v.ExtendLo8ToUint16().StoreArray((*[8]uint16)(unsafe.Add(dp, 2*off)))
				if y+1 < height {
					sp = unsafe.Add(sp, srcStride)
					dp = unsafe.Add(dp, dstByteStride)
				}
			}
			return
		}
		for y := 0; y < height; y++ {
			for x := 0; x < width; x++ {
				*(*uint16)(unsafe.Add(dp, 2*x)) = uint16(*(*byte)(unsafe.Add(sp, x)))
			}
			if y+1 < height {
				sp = unsafe.Add(sp, srcStride)
				dp = unsafe.Add(dp, dstByteStride)
			}
		}
		return
	}
	for y := 0; y < height; y++ {
		x := 0
		for ; x+16 <= width; x += 16 {
			v := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(sp, x)))
			d := unsafe.Add(dp, 2*x)
			v.ExtendLo8ToUint16().StoreArray((*[8]uint16)(d))
			v.HiToLo().ExtendLo8ToUint16().StoreArray((*[8]uint16)(unsafe.Add(d, 16)))
		}
		if x < width {
			off := width - 16
			v := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(sp, off)))
			d := unsafe.Add(dp, 2*off)
			v.ExtendLo8ToUint16().StoreArray((*[8]uint16)(d))
			v.HiToLo().ExtendLo8ToUint16().StoreArray((*[8]uint16)(unsafe.Add(d, 16)))
		}
		if y+1 < height {
			sp = unsafe.Add(sp, srcStride)
			dp = unsafe.Add(dp, dstByteStride)
		}
	}
}
