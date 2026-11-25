// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build !(goexperiment.simd && (arm64 || amd64) && !purego)

package motion

import (
	"github.com/thesyncim/goav1/internal/av1/frame"
)

const scaledStep2Half = ScaleSubpelScale / 2

func convolveScale2D8Step2GoSIMD(frame.Plane, frame.Plane, int, int, int, int, int64, int64, *SubpelKernelTable, *SubpelKernelTable, int, *scaledIM) bool {
	return false
}

// Targets without the vector 16bpc copy always run dsp.CopyPlaneBlock.
func mc16CopyFast(dst *frame.Plane, ref *frame.Plane, dstX, dstY, refX, refY, w, h int) bool {
	return false
}

// Targets without the 16bpc blend kernels (mc16mask_gosimd_arm64.go) always
// run the general blend paths.

func mc16BlendAvgFast(dst *frame.Plane, a, b []uint16, bitDepth, dstX, dstY, w, h int) bool {
	return false
}

func mc16BlendMaskFast(dst frame.Plane, a, b []uint16, bitDepth, dstX, dstY, w, h int, mask []byte, maskStride int, subX, subY bool) bool {
	return false
}

func mc16DiffWtdMaskFast(mask []byte, maskStride int, a, b []uint16, bitDepth, w, h int, invert bool) bool {
	return false
}

// Targets without the dav1d-style 8-bit kernels (mc8_gosimd_arm64.go) always
// run the general convolve paths.

// Put8 is the dav1d-style 8-bit put; without the kernels it always declines.
func Put8(dst *frame.Plane, ref *frame.Plane, dstX, dstY, refX, refY, w, h, filterW, filterH, subX, subY int, filters InterpFilters, scratch *ConvolveScratch) bool {
	return false
}

func mc8PrepFast(out []uint16, ref *frame.Plane, refX, refY, w, h, filterW, filterH, subX, subY int, filters InterpFilters, scratch *CompoundConvolveScratch) bool {
	return false
}

func mc8CopyFast(dst *frame.Plane, ref *frame.Plane, dstX, dstY, refX, refY, w, h int) bool {
	return false
}

func mc8BlendMaskFast(dst frame.Plane, a, b []uint16, dstX, dstY, w, h int, mask []byte, maskStride int, subX, subY bool) bool {
	return false
}

func mc8DiffWtdMaskFast(mask []byte, maskStride int, a, b []uint16, w, h int, invert bool) bool {
	return false
}

func mc8BlendAvgFast(dst *frame.Plane, a, b []uint16, dstX, dstY, w, h int) bool {
	return false
}

func mc16PrepFast(out []uint16, ref *frame.Plane, bitDepth, refX, refY, w, h, filterW, filterH, subX, subY int, filters InterpFilters, scratch *CompoundConvolveScratch) bool {
	return false
}

func mc16PutFast(dst *frame.Plane, ref *frame.Plane, bitDepth, dstX, dstY, refX, refY, w, h, filterW, filterH, subX, subY int, filters InterpFilters, scratch *ConvolveScratch) bool {
	return false
}

// Targets without a Go SIMD warp kernel keep the pure-Go references. These are the
// static counterparts of warp_dispatch_arm64.go, so warp.go's call sites resolve
// to a direct call on every platform (see warp_dispatch.go for why static
// dispatch matters to the tmp scratch's stack residency).

func warpHorizontal8ResidentDispatch(tmp *warpTmp, ref frame.Plane, ix4, sx4, iy4, sy4, alpha, beta, reduceBitsHoriz, offsetBitsHoriz int) int {
	return warpHorizontal8Resident(tmp, ref, ix4, sx4, iy4, sy4, alpha, beta, reduceBitsHoriz, offsetBitsHoriz)
}

func warpVertical8FullDispatch(dst frame.Plane, tmp *warpTmp, i, j, rowShift, colShift, baseSY, gamma, delta, reduceBitsVert, offsetBitsVert int) {
	warpVertical8Full(dst, tmp, i, j, rowShift, colShift, baseSY, gamma, delta, reduceBitsVert, offsetBitsVert)
}

func warpVertical8FullGamma0Dispatch(dst frame.Plane, tmp *warpTmp, i, j, rowShift, colShift, baseSY, delta, reduceBitsVert, offsetBitsVert int) {
	warpVertical8FullGamma0(dst, tmp, i, j, rowShift, colShift, baseSY, delta, reduceBitsVert, offsetBitsVert)
}

// warpCompound8x8Dispatch has no kernel here: the scalar compound block runs.
func warpCompound8x8Dispatch(out []uint16, stride int, ref frame.Plane, ix4, sx4, iy4, sy4, alpha, beta, gamma, delta int) bool {
	return false
}

func warpHorizontalHighBDResidentDispatch(tmp *[warpedIntermediateRows * warpedIntermediateColumns]int32, ref frame.Plane, ix4 int, sx4 int, iy4 int, sy4 int, alpha int, beta int, reduceBitsHoriz int, offsetBitsHoriz int) int {
	return warpHorizontalHighBDResident(tmp, ref, ix4, sx4, iy4, sy4, alpha, beta, reduceBitsHoriz, offsetBitsHoriz)
}

// Without the amd64 AVX2 or arm64 NEON kernels the 16bpc vertical warp pass is the scalar
// one.

func warpVerticalHighBDFullDispatch(dst frame.Plane, bitDepth uint8, max uint16, tmp *warpTmp, i, j, rowShift, colShift, baseSY, gamma, delta, reduceBitsVert, offsetBitsVert int) {
	warpVerticalHighBDFull(dst, bitDepth, max, tmp, i, j, rowShift, colShift, baseSY, gamma, delta, reduceBitsVert, offsetBitsVert)
}

func warpVerticalHighBDFullGamma0Dispatch(dst frame.Plane, bitDepth uint8, max uint16, tmp *warpTmp, i, j, rowShift, colShift, baseSY, delta, reduceBitsVert, offsetBitsVert int) {
	warpVerticalHighBDFullGamma0(dst, bitDepth, max, tmp, i, j, rowShift, colShift, baseSY, delta, reduceBitsVert, offsetBitsVert)
}
