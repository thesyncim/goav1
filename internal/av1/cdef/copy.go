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
		for col := range width {
			dst[row*dstStride+col] = uint16(src[row*srcStride+col])
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
