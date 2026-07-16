// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

package motion

import (
	"github.com/thesyncim/goav1/internal/av1/dsp/cpu"
	"github.com/thesyncim/goav1/internal/av1/frame"
)

// warpHorizontal8DotProdCtx carries the resident 8-bit horizontal warp
// arguments. Field offsets are mirrored by warp_dotprod_arm64.s.
type warpHorizontal8DotProdCtx struct {
	tmp     *int32
	ref     *byte
	filter  *int64
	permute *byte
	refStr  uintptr
	sxStart int64
	alpha   int64
	beta    int64
}

//go:noescape
func warpHorizontal8DotProdAsm(ctx *warpHorizontal8DotProdCtx)

// The four tables form two adjacent eight-tap source windows per vector. This
// is the same layout used by libaom's horizontal_filter_8x1_f8 warp kernel.
var warpHorizontalPermute = [64]byte{
	0, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 8,
	2, 3, 4, 5, 6, 7, 8, 9, 3, 4, 5, 6, 7, 8, 9, 10,
	4, 5, 6, 7, 8, 9, 10, 11, 5, 6, 7, 8, 9, 10, 11, 12,
	6, 7, 8, 9, 10, 11, 12, 13, 7, 8, 9, 10, 11, 12, 13, 14,
}

// All AV1 warp taps fit in int8. Packing one filter row into an int64 lets the
// assembly gather two per-column filters into each vector with two scalar
// loads, matching dav1d's int8 warp-filter representation.
var warpedFilterI8Packed = func() [len(warpedFilter)]int64 {
	var packed [len(warpedFilter)]int64
	for row := range warpedFilter {
		var bits uint64
		for tap := range filterTaps {
			bits |= uint64(uint8(int8(warpedFilter[row][tap]))) << (8 * uint(tap))
		}
		packed[row] = int64(bits)
	}
	return packed
}()

func warpHorizontal8ResidentDispatch(tmp *[warpedIntermediateRows * warpedIntermediateColumns]int32, ref frame.Plane, ix4, sx4, iy4, sy4, alpha, beta, reduceBitsHoriz, offsetBitsHoriz int) int {
	// The kernel is specialized to the normative 8-bit warp rounding. A strict
	// right-edge check makes its single 16-byte row load stay inside the plane;
	// the scalar resident path handles the one-pixel-tight edge case.
	if !cpu.Detected.DOTPROD || reduceBitsHoriz != round0Bits || offsetBitsHoriz != 8+filterBits-1 ||
		ix4+8 >= ref.Width || !warpHorizontalFilterRangeOK(sx4, alpha, beta) {
		return warpHorizontal8Resident(tmp, ref, ix4, sx4, iy4, sy4, alpha, beta, reduceBitsHoriz, offsetBitsHoriz)
	}
	ctx := warpHorizontal8DotProdCtx{
		tmp:     &tmp[0],
		ref:     &ref.Pix[(iy4-7)*ref.Stride+ix4-7],
		filter:  &warpedFilterI8Packed[0],
		permute: &warpHorizontalPermute[0],
		refStr:  uintptr(ref.Stride),
		sxStart: int64(sx4 - 3*beta),
		alpha:   int64(alpha),
		beta:    int64(beta),
	}
	warpHorizontal8DotProdAsm(&ctx)
	return sy4
}

// Filter phases are affine in row and column, so their extrema are at the four
// corners of the 15x8 intermediate tile. Keeping the rare out-of-table case on
// the scalar path removes eight clamps from every assembly row without changing
// its semantics.
func warpHorizontalFilterRangeOK(sx4, alpha, beta int) bool {
	row0 := sx4 - 3*beta
	row14 := sx4 + 11*beta
	lo := min(row0, row14)
	hi := max(row0, row14)
	if alpha < 0 {
		lo += 7 * alpha
	} else {
		hi += 7 * alpha
	}
	lo = roundPowerOfTwo(lo, warpedDiffPrecBits) + warpedPixelPrecShifts
	hi = roundPowerOfTwo(hi, warpedDiffPrecBits) + warpedPixelPrecShifts
	return lo >= 0 && hi < len(warpedFilter)
}
