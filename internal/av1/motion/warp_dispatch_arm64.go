// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

package motion

import (
	"os"

	"github.com/thesyncim/goav1/internal/av1/dsp/cpu"
	"github.com/thesyncim/goav1/internal/av1/frame"
)

// warpAsmEnabled selects the NEON warp kernels. NEON is mandatory on every arm64
// chip Go runs on; the GOAV1_DISABLE_WARP_ASM escape hatch keeps the scalar path
// for A/B timing and bisection (measurement scaffolding, not a shipped switch).
// It is a package var (resolved once) so the warp*Dispatch calls stay static and
// escape analysis keeps warpAffine8Offset's tmp scratch on the stack.
var warpAsmEnabled = cpu.Detected.NEON && os.Getenv("GOAV1_DISABLE_WARP_ASM") == ""

func warpHorizontal8ResidentDispatch(tmp *warpTmp, ref frame.Plane, ix4, sx4, iy4, sy4, alpha, beta, reduceBitsHoriz, offsetBitsHoriz int) int {
	if warpAsmEnabled {
		return warpHorizontal8ResidentNEON(tmp, ref, ix4, sx4, iy4, sy4, alpha, beta, reduceBitsHoriz, offsetBitsHoriz)
	}
	return warpHorizontal8Resident(tmp, ref, ix4, sx4, iy4, sy4, alpha, beta, reduceBitsHoriz, offsetBitsHoriz)
}

func warpVertical8FullDispatch(dst frame.Plane, tmp *warpTmp, i, j, rowShift, colShift, baseSY, gamma, delta, reduceBitsVert, offsetBitsVert int) {
	if warpAsmEnabled {
		warpVertical8FullNEON(dst, tmp, i, j, rowShift, colShift, baseSY, gamma, delta, reduceBitsVert, offsetBitsVert)
		return
	}
	warpVertical8Full(dst, tmp, i, j, rowShift, colShift, baseSY, gamma, delta, reduceBitsVert, offsetBitsVert)
}

func warpVertical8FullGamma0Dispatch(dst frame.Plane, tmp *warpTmp, i, j, rowShift, colShift, baseSY, delta, reduceBitsVert, offsetBitsVert int) {
	if warpAsmEnabled {
		warpVertical8FullGamma0NEON(dst, tmp, i, j, rowShift, colShift, baseSY, delta, reduceBitsVert, offsetBitsVert)
		return
	}
	warpVertical8FullGamma0(dst, tmp, i, j, rowShift, colShift, baseSY, delta, reduceBitsVert, offsetBitsVert)
}
