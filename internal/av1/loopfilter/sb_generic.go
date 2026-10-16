// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build !goexperiment.simd || !arm64 || purego

package loopfilter

func filterSBVertical8(pix []byte, off, stride int, vm [3]uint32, luma bool, lvl [][4]uint8, lvlOff, lvlStride, comp int, lut *SBLUT) SBStats {
	return filterSBVertical8Scalar(pix, off, stride, vm, luma, lvl, lvlOff, lvlStride, comp, lut)
}

func filterSBHorizontal8(pix []byte, off, stride, _ int, vm [3]uint32, luma bool, lvl [][4]uint8, lvlOff, lvlStride, comp int, lut *SBLUT) SBStats {
	return filterSBHorizontal8Scalar(pix, off, stride, vm, luma, lvl, lvlOff, lvlStride, comp, lut)
}
