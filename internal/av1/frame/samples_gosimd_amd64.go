// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && amd64 && !purego

package frame

import "simd/archsimd"

// loadSampleRows8 binds the Go SIMD 8-bit widen on amd64 when the CPU
// advertises AVX2, falling back to the pure-Go reference otherwise. The check
// is an inline runtime branch rather than a func variable: an indirect call
// would make the compiler leak dst/src at every Load*SamplePlane entry point and
// break the decoder's zero-alloc steady-state budget (see the dispatch note on
// loadSampleRows8PureGo).
func loadSampleRows8(dst []uint16, dstStride int, src []byte, srcStride int, width int, height int) {
	if sampleRows8UseSIMD(width, height) {
		loadSampleRows8SIMD(dst, dstStride, src, srcStride, width, height)
		return
	}
	loadSampleRows8PureGo(dst, dstStride, src, srcStride, width, height)
}

// sampleRows8UseSIMD reports whether loadSampleRows8 takes the AVX2 kernel for
// a rectangle of the given shape: the CPU must advertise AVX2 and the rows must
// hold at least one 16-column vector body.
func sampleRows8UseSIMD(width int, height int) bool {
	return archsimd.X86.AVX2() && width >= 16 && height > 0
}

// loadSampleRows8SIMD widens 8-bit sample rows into uint16 staging storage, 16
// columns per iteration: one 16-byte load zero-extended to sixteen uint16 lanes
// (VPMOVZXBW) and one 32-byte store. The width remainder (< 16 columns) is
// finished by a scalar tail in the same row, so no vector access ever leaves
// the row and the output is bit-identical to loadSampleRows8PureGo. The caller
// must have AVX2 (see loadSampleRows8).
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
		for x := 0; x < vecWidth; x += 16 {
			v := archsimd.LoadUint8x16Array((*[16]uint8)(src[srcOff+x:]))
			v.ExtendToUint16().StoreArray((*[16]uint16)(dst[dstOff+x:]))
		}
		for x := vecWidth; x < width; x++ {
			dst[dstOff+x] = uint16(src[srcOff+x])
		}
		srcOff += srcStride
		dstOff += dstStride
	}
}
