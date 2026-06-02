// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

package cdef

// NEON-accelerated CDEF block filter. The .s file implements the per-row inner
// loop for 8-wide blocks (every luma block and the 4:4:4 chroma block). The Go
// wrapper here routes narrower shapes (8x4 / 4x8 / 4x4 chroma) to the pure-Go
// reference, which keeps the asm to a single code path and the byte-exactness
// contract easy to audit.
//
// Bit-exactness with filterBlockPureGo:
//   - diff = sample - x is computed in int16 lanes (pixels and the 0x4000
//     VeryLarge sentinel both fit, and the difference stays within int16).
//   - constrain() is reproduced lane-wise: abs, variable right shift by the
//     precomputed shift, threshold - shifted clamped to [0, abs], then the sign
//     of diff is reapplied. Identical to constrainShifted.
//   - taps are applied with signed widening multiply-accumulate into int32
//     lanes, matching the int accumulator sum.
//   - the final term x + ((8 + sum - (sum<0)) >> 4) is reproduced exactly,
//     including the -1 bias when sum is negative (an arithmetic shift of
//     (sum*2 + 16 - sign) folded as in the reference is avoided: the asm
//     computes sum, then the bias and shift in scalar-equivalent vector ops).
//   - when both strengths are non-zero the result is clamped to [min, max] over
//     the tap neighbourhood, with the VeryLarge sentinel skipped for max
//     exactly as maxClip does.

// filterBlockNEONCtx is the asm calling context. Field order and sizes are part
// of the ABI shared with filter_neon_arm64.s; do not reorder.
type filterBlockNEONCtx struct {
	dst    *uint16
	input  *uint16 // pointer to input[inputOrigin]
	dstStr uintptr // dst stride in elements
	height uintptr

	pri0 int64 // direction offsets in elements (signed)
	pri1 int64
	sec0 int64
	sec1 int64
	sec2 int64
	sec3 int64

	priTap0 int64
	priTap1 int64
	secTap0 int64
	secTap1 int64

	priStrength int64
	secStrength int64
	priShift    int64
	secShift    int64

	enablePrimary   int64
	enableSecondary int64
	clipping        int64
}

//go:noescape
func filterBlockNEONAsm(ctx *filterBlockNEONCtx)

func filterBlockNEON(dst []uint16, dstStride int, dstOrigin int, input []uint16, inputOrigin int, params BlockFilterParams) {
	if params.Width == 4 {
		var tmp [8 * 8]uint16
		filterBlockNEONWide(tmp[:], 8, 0, input, inputOrigin, params)
		for row := 0; row < int(params.Height); row++ {
			copy(dst[dstOrigin+row*dstStride:dstOrigin+row*dstStride+4], tmp[row*8:row*8+4])
		}
		return
	}
	if params.Width != 8 {
		filterBlockPureGo(dst, dstStride, dstOrigin, input, inputOrigin, params)
		return
	}
	filterBlockNEONWide(dst, dstStride, dstOrigin, input, inputOrigin, params)
}

func filterBlockNEONWide(dst []uint16, dstStride int, dstOrigin int, input []uint16, inputOrigin int, params BlockFilterParams) {
	primaryStrength := int(params.PrimaryStrength)
	secondaryStrength := int(params.SecondaryStrength)
	direction := int(params.Direction)
	primaryDamping := int(params.PrimaryDamping)
	secondaryDamping := int(params.SecondaryDamping)
	coeffShift := int(params.CoeffShift)
	enablePrimary := primaryStrength != 0
	enableSecondary := secondaryStrength != 0
	clippingRequired := enablePrimary && enableSecondary
	priTaps := cdefPrimaryTaps[(primaryStrength>>coeffShift)&1]

	ctx := filterBlockNEONCtx{
		dst:    &dst[dstOrigin],
		input:  &input[inputOrigin],
		dstStr: uintptr(dstStride),
		height: uintptr(params.Height),

		pri0: int64(cdefDirections[direction+2][0]),
		pri1: int64(cdefDirections[direction+2][1]),
		sec0: int64(cdefDirections[direction+4][0]),
		sec1: int64(cdefDirections[direction+4][1]),
		sec2: int64(cdefDirections[direction][0]),
		sec3: int64(cdefDirections[direction][1]),

		priTap0: int64(priTaps[0]),
		priTap1: int64(priTaps[1]),
		secTap0: int64(cdefSecondaryTaps[0]),
		secTap1: int64(cdefSecondaryTaps[1]),

		priStrength: int64(primaryStrength),
		secStrength: int64(secondaryStrength),
		priShift:    int64(constrainShift(primaryStrength, primaryDamping)),
		secShift:    int64(constrainShift(secondaryStrength, secondaryDamping)),

		enablePrimary:   boolToInt64(enablePrimary),
		enableSecondary: boolToInt64(enableSecondary),
		clipping:        boolToInt64(clippingRequired),
	}
	filterBlockNEONAsm(&ctx)
}

func boolToInt64(v bool) int64 {
	if v {
		return 1
	}
	return 0
}
