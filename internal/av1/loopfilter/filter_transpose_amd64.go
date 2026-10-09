// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && amd64 && !purego

package loopfilter

func lfTranspose8x8U8(src []byte, srcOff, srcStride int, dst []byte, dstOff, dstStride int) {
	for y := 0; y < 8; y++ {
		for x := 0; x < 8; x++ {
			dst[dstOff+x*dstStride+y] = src[srcOff+y*srcStride+x]
		}
	}
}

func lfTranspose8x8U8Short(src []byte, srcOff, srcStride int, dst []byte, dstOff, dstStride int, readBytes, writeBytes int) {
	for y := 0; y < 8; y++ {
		for x := 0; x < writeBytes; x++ {
			if y < readBytes {
				dst[dstOff+y*dstStride+x] = src[srcOff+x*srcStride+y]
			}
		}
	}
}

func lfTranspose8x8U16(src []byte, srcOff, srcStride int, dst []byte, dstOff, dstStride int) {
	for y := 0; y < 8; y++ {
		for x := 0; x < 8; x++ {
			d := dstOff + 2*y + x*dstStride
			s := srcOff + 2*x + y*srcStride
			dst[d], dst[d+1] = src[s], src[s+1]
		}
	}
}

func lfTranspose8x8U16Short(src []byte, srcOff, srcStride int, dst []byte, dstOff, dstStride int, readWidth, writeWidth int) {
	for y := 0; y < 8; y++ {
		for x := 0; x < writeWidth; x++ {
			if y < readWidth {
				d := dstOff + 2*x + y*dstStride
				s := srcOff + 2*y + x*srcStride
				dst[d], dst[d+1] = src[s], src[s+1]
			}
		}
	}
}
