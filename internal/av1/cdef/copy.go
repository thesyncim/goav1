// Ported from libaom: av1/common/cdef_block.c
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

package cdef

// CopyRect8To16 ports libaom's cdef_copy_rect8_8bit_to_16bit_c.
func CopyRect8To16(dst []uint16, dstStride int, src []uint8, srcStride int, width int, height int) error {
	if width <= 0 || height <= 0 || dstStride < width || srcStride < width ||
		!blockFits(len(dst), dstStride, width, height) ||
		!blockFits(len(src), srcStride, width, height) {
		return ErrInvalidCDEF
	}
	for row := range height {
		dstOff := row * dstStride
		srcOff := row * srcStride
		dstRow := dst[dstOff : dstOff+width]
		srcRow := src[srcOff : srcOff+width]
		col := 0
		for ; col+8 <= width; col += 8 {
			dstRow[col+0] = uint16(srcRow[col+0])
			dstRow[col+1] = uint16(srcRow[col+1])
			dstRow[col+2] = uint16(srcRow[col+2])
			dstRow[col+3] = uint16(srcRow[col+3])
			dstRow[col+4] = uint16(srcRow[col+4])
			dstRow[col+5] = uint16(srcRow[col+5])
			dstRow[col+6] = uint16(srcRow[col+6])
			dstRow[col+7] = uint16(srcRow[col+7])
		}
		for ; col < width; col++ {
			dstRow[col] = uint16(srcRow[col])
		}
	}
	return nil
}

// CopyRect16To16 ports libaom's cdef_copy_rect8_16bit_to_16bit_c.
func CopyRect16To16(dst []uint16, dstStride int, src []uint16, srcStride int, width int, height int) error {
	if width <= 0 || height <= 0 || dstStride < width || srcStride < width ||
		!blockFits(len(dst), dstStride, width, height) ||
		!blockFits(len(src), srcStride, width, height) {
		return ErrInvalidCDEF
	}
	for row := range height {
		dstOff := row * dstStride
		srcOff := row * srcStride
		copy(dst[dstOff:dstOff+width], src[srcOff:srcOff+width])
	}
	return nil
}
