// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build !goexperiment.simd || !(arm64 || amd64) || purego

package loopfilter

func filterSBVertical8(pix []byte, off, stride, rowBytes int, vm [3]uint32, luma bool, lvl []uint8, lvlOff, lvlStride, comp int, lut *SBLUT) {
	_ = rowBytes
	filterSBVertical8Scalar(pix, off, stride, vm, luma, lvl, lvlOff, lvlStride, comp, lut)
}

func filterSBHorizontal8(pix []byte, off, stride, _ int, vm [3]uint32, luma bool, lvl []uint8, lvlOff, lvlStride, comp int, lut *SBLUT) {
	filterSBHorizontal8Scalar(pix, off, stride, vm, luma, lvl, lvlOff, lvlStride, comp, lut)
}

func filterSBVertical16(pix []byte, off, stride, _ int, bitDepth uint8, vm [3]uint32, luma bool, lvl []uint8, lvlOff, lvlStride, comp int, lut *SBLUT) {
	filterSBVertical16Scalar(pix, off, stride, bitDepth, vm, luma, lvl, lvlOff, lvlStride, comp, lut)
}

func filterSBHorizontal16(pix []byte, off, stride, _ int, bitDepth uint8, vm [3]uint32, luma bool, lvl []uint8, lvlOff, lvlStride, comp int, lut *SBLUT) {
	filterSBHorizontal16Scalar(pix, off, stride, bitDepth, vm, luma, lvl, lvlOff, lvlStride, comp, lut)
}
