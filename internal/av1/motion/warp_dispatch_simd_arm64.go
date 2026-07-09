// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package motion

import "github.com/thesyncim/goav1/internal/av1/frame"

// SIMD dispatch for the hot 8-bit warped-motion inner kernels: routes to the
// Go-native SIMD forms (warp_h_gosimd_arm64.go / warp_v_gosimd_arm64.go). The
// wrappers are thin forwarders so they inline and keep the caller's `tmp`
// scratch array on the stack (direct calls propagate the non-escaping summary a
// func-value would erase).

func warpHorizontal8ResidentDispatch(tmp *[warpedIntermediateRows * warpedIntermediateColumns]int32, ref frame.Plane, ix4, sx4, iy4, sy4, alpha, beta, reduceBitsHoriz, offsetBitsHoriz int) int {
	return warpHorizontal8ResidentSIMD(tmp, ref, ix4, sx4, iy4, sy4, alpha, beta, reduceBitsHoriz, offsetBitsHoriz)
}

func warpVertical8FullDispatch(dst frame.Plane, tmp *[warpedIntermediateRows * warpedIntermediateColumns]int32, i, j, rowShift, colShift, baseSY, gamma, delta, reduceBitsVert, offsetBitsVert int) {
	warpVertical8FullSIMD(dst, tmp, i, j, rowShift, colShift, baseSY, gamma, delta, reduceBitsVert, offsetBitsVert)
}

func warpVertical8FullGamma0Dispatch(dst frame.Plane, tmp *[warpedIntermediateRows * warpedIntermediateColumns]int32, i, j, rowShift, colShift, baseSY, delta, reduceBitsVert, offsetBitsVert int) {
	warpVertical8FullGamma0SIMD(dst, tmp, i, j, rowShift, colShift, baseSY, delta, reduceBitsVert, offsetBitsVert)
}
