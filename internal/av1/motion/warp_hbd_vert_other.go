// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build !(goexperiment.simd && amd64 && !purego)

package motion

import "github.com/thesyncim/goav1/internal/av1/frame"

// Without the amd64 AVX2 kernels the 16bpc vertical warp pass is the scalar
// one.

func warpVerticalHighBDFullDispatch(dst frame.Plane, bitDepth uint8, max uint16, tmp *warpTmp, i, j, rowShift, colShift, baseSY, gamma, delta, reduceBitsVert, offsetBitsVert int) {
	warpVerticalHighBDFull(dst, bitDepth, max, tmp, i, j, rowShift, colShift, baseSY, gamma, delta, reduceBitsVert, offsetBitsVert)
}

func warpVerticalHighBDFullGamma0Dispatch(dst frame.Plane, bitDepth uint8, max uint16, tmp *warpTmp, i, j, rowShift, colShift, baseSY, delta, reduceBitsVert, offsetBitsVert int) {
	warpVerticalHighBDFullGamma0(dst, bitDepth, max, tmp, i, j, rowShift, colShift, baseSY, delta, reduceBitsVert, offsetBitsVert)
}
