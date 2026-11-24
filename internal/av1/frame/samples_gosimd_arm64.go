// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package frame

import (
	"encoding/binary"
	"simd/archsimd"
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
// scalar stores. Each row is one checked window of each buffer, so no access
// leaves the row and output matches loadSampleRows8PureGo.
func loadSampleRows8SIMD(dst []uint16, dstStride int, src []byte, srcStride int, width int, height int) {
	if width <= 0 || height <= 0 {
		return
	}
	srcOff := 0
	dstOff := 0
	if width < 8 {
		for y := 0; y < height; y++ {
			s := src[srcOff : srcOff+width]
			d := dst[dstOff : dstOff+width]
			for x := 0; x < len(s) && x < len(d); x++ {
				d[x] = uint16(s[x])
			}
			srcOff += srcStride
			dstOff += dstStride
		}
		return
	}
	if width < 16 {
		// One 8-byte group, plus an overlapping one when width > 8.
		var word [2]uint64
		off := width - 8
		for y := 0; y < height; y++ {
			word[0] = binary.LittleEndian.Uint64(src[srcOff : srcOff+8])
			archsimd.LoadUint64x2Array(&word).ReshapeToUint8s().ExtendLo8ToUint16().StoreArray((*[8]uint16)(dst[dstOff : dstOff+8]))
			if off > 0 {
				word[0] = binary.LittleEndian.Uint64(src[srcOff+off : srcOff+width])
				archsimd.LoadUint64x2Array(&word).ReshapeToUint8s().ExtendLo8ToUint16().StoreArray((*[8]uint16)(dst[dstOff+off : dstOff+width]))
			}
			srcOff += srcStride
			dstOff += dstStride
		}
		return
	}
	for y := 0; y < height; y++ {
		s := src[srcOff : srcOff+width]
		d := dst[dstOff : dstOff+width]
		for x := 0; x < len(s)-16 && x < len(d)-16; x += 16 {
			loadSampleRows8Widen16((*[16]uint8)(s[x:x+16]), (*[16]uint16)(d[x:x+16]))
		}
		// The last group overlaps the previous one when width%16 != 0.
		loadSampleRows8Widen16((*[16]uint8)(s[len(s)-16:]), (*[16]uint16)(d[len(d)-16:]))
		srcOff += srcStride
		dstOff += dstStride
	}
}

// loadSampleRows8Widen16 zero-extends sixteen bytes to sixteen uint16s.
func loadSampleRows8Widen16(s *[16]uint8, d *[16]uint16) {
	v := archsimd.LoadUint8x16Array(s)
	v.ExtendLo8ToUint16().StoreArray((*[8]uint16)(d[0:8]))
	v.HiToLo().ExtendLo8ToUint16().StoreArray((*[8]uint16)(d[8:16]))
}

// sampleRows16UseSIMD reports whether the 16-bit staging rows take the Go
// SIMD body; AdvSIMD is mandatory on arm64.
func sampleRows16UseSIMD() bool { return true }
