// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

package motion

import "github.com/thesyncim/goav1/internal/av1/frame"

// NEON warp prediction for the 8-bit interior kernels, mirroring the pure-Go
// warpHorizontal8Resident / warpVertical8Full(Gamma0) in warp.go. The asm
// (warp_neon_arm64.s) reproduces goav1's own staged arithmetic bit-for-bit; it
// borrows the SMULL/EXT horizontal shape and the int8 filter transpose from
// dav1d (src/arm/64/mc.S warp_affine_8x8) but not dav1d's intermediate
// representation. The exact bias mapping is documented on each kernel below.
//
// Every scalar constant except reduce/offset bits is passed through the ctx so
// the asm stays a pure evaluator; the wrappers pick the NEON path only when the
// 8-bit reduce/offset bits are in force and the selected warpedFilter indices
// stay in range (see warp_dispatch.go), otherwise they defer to scalar, which
// owns the index clamp / out-of-range skip.

// warpH8ResidentNEONCtx is the calling context for warpHorizontalU8NEONAsm.
// Field order and sizes are the ABI shared with warp_neon_arm64.s; do not
// reorder.
type warpH8ResidentNEONCtx struct {
	src    *uint8  // &ref.Pix[(iy4-7)*stride + (ix4-7)] (first tap of row k=-7)
	tmp    *int32  // intermediate base (&tmp[0]), 15 rows x 8 int32
	filter *int8   // &warpedFilterI8[warpedPixelPrecShifts][0] (index-0 anchor)
	srcStr uintptr // ref.Stride in bytes
	mx0    int32   // sx4 + beta*(-3): the row-0 filter phase accumulator
	alpha  int32   // per-column filter-phase step
	beta   int32   // per-row filter-phase step
}

// warpV8FullNEONCtx is the calling context for warpVerticalU8NEONAsm. Field
// order and sizes are the ABI shared with warp_neon_arm64.s; do not reorder.
type warpV8FullNEONCtx struct {
	dst    *uint8  // &dst.Pix[(i+rowShift)*stride + (j+colShift)] (output r=0,c=0)
	tmp    *int32  // intermediate base (&tmp[0])
	filter *int8   // &warpedFilterI8[warpedPixelPrecShifts][0]
	dstStr uintptr // dst.Stride in bytes
	my0    int32   // baseSY: the output-row-0 filter phase accumulator
	gamma  int32   // per-column filter-phase step
	delta  int32   // per-row filter-phase step
	seed   int32   // accumulator seed folding offset + rounding bias - clip offset
}

//go:noescape
func warpHorizontalU8NEONAsm(ctx *warpH8ResidentNEONCtx)

//go:noescape
func warpVerticalU8NEONAsm(ctx *warpV8FullNEONCtx)

// warpHorizontal8ResidentNEON reproduces warpHorizontal8Resident. It runs the
// NEON kernel only when the 8-bit reduce/offset bits are in force and no
// selected filter index needs the scalar clamp; both conditions hold for every
// conformant 8-bit stream. Otherwise it defers to the scalar reference.
func warpHorizontal8ResidentNEON(tmp *warpTmp, ref frame.Plane, ix4, sx4, iy4, sy4, alpha, beta, reduceBitsHoriz, offsetBitsHoriz int) int {
	if reduceBitsHoriz != round0Bits || offsetBitsHoriz != 8+filterBits-1 ||
		!warpHorizResidentOffsInRange(sx4, alpha, beta) {
		return warpHorizontal8Resident(tmp, ref, ix4, sx4, iy4, sy4, alpha, beta, reduceBitsHoriz, offsetBitsHoriz)
	}
	ctx := warpH8ResidentNEONCtx{
		src:    &ref.Pix[(iy4-7)*ref.Stride+(ix4-7)],
		tmp:    &tmp[0],
		filter: &warpedFilterI8[warpedPixelPrecShifts][0],
		srcStr: uintptr(ref.Stride),
		mx0:    int32(sx4 - 3*beta),
		alpha:  int32(alpha),
		beta:   int32(beta),
	}
	warpHorizontalU8NEONAsm(&ctx)
	return sy4
}

// warpVertical8FullNEON reproduces warpVertical8Full. Because the scalar
// vertical *skips* out-of-range indices, the NEON kernel — which has no per-lane
// skip — is used only when every index is in range; otherwise scalar handles it.
func warpVertical8FullNEON(dst frame.Plane, tmp *warpTmp, i, j, rowShift, colShift, baseSY, gamma, delta, reduceBitsVert, offsetBitsVert int) {
	if reduceBitsVert != round1Bits || offsetBitsVert != 8+2*filterBits-round0Bits ||
		!warpVertFullOffsInRange(baseSY, gamma, delta) {
		warpVertical8Full(dst, tmp, i, j, rowShift, colShift, baseSY, gamma, delta, reduceBitsVert, offsetBitsVert)
		return
	}
	warpVerticalNEONCommon(dst, tmp, i, j, rowShift, colShift, baseSY, gamma, delta, reduceBitsVert, offsetBitsVert)
}

// warpVertical8FullGamma0NEON reproduces warpVertical8FullGamma0. gamma==0 makes
// every column in a row share one filter, which the same kernel expresses as a
// degenerate (broadcast) transpose, so it routes through the common path with
// gamma=0. The scalar gamma0 reference owns the out-of-range skip.
func warpVertical8FullGamma0NEON(dst frame.Plane, tmp *warpTmp, i, j, rowShift, colShift, baseSY, delta, reduceBitsVert, offsetBitsVert int) {
	if reduceBitsVert != round1Bits || offsetBitsVert != 8+2*filterBits-round0Bits ||
		!warpVertFullOffsInRange(baseSY, 0, delta) {
		warpVertical8FullGamma0(dst, tmp, i, j, rowShift, colShift, baseSY, delta, reduceBitsVert, offsetBitsVert)
		return
	}
	warpVerticalNEONCommon(dst, tmp, i, j, rowShift, colShift, baseSY, 0, delta, reduceBitsVert, offsetBitsVert)
}

func warpVerticalNEONCommon(dst frame.Plane, tmp *warpTmp, i, j, rowShift, colShift, baseSY, gamma, delta, reduceBitsVert, offsetBitsVert int) {
	// seed folds the vertical offset (1<<offsetBitsVert), the SRSHR rounding
	// bias, and the final -((1<<7)+(1<<8)) clip offset into one accumulator
	// preload. Since (1<<7)+(1<<8) << reduceBitsVert is an exact multiple of
	// 1<<reduceBitsVert, folding it before the shift matches
	// roundPowerOfTwo(sum, reduceBitsVert) - ((1<<7)+(1<<8)) bit-for-bit; the
	// SQXTUN+UQXTN narrow then applies clipPixel's [0,255] clamp.
	seed := (1 << offsetBitsVert) - ((1 << 7) + (1 << 8)) << reduceBitsVert
	ctx := warpV8FullNEONCtx{
		dst:    &dst.Pix[(i+rowShift)*dst.Stride+(j+colShift)],
		tmp:    &tmp[0],
		filter: &warpedFilterI8[warpedPixelPrecShifts][0],
		dstStr: uintptr(dst.Stride),
		my0:    int32(baseSY),
		gamma:  int32(gamma),
		delta:  int32(delta),
		seed:   int32(seed),
	}
	warpVerticalU8NEONAsm(&ctx)
}
