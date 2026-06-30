// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

package cdef

// NEON-accelerated CDEF block filter. The .s file implements the per-row
// inner loop for 8-wide blocks (every luma block and the 4:4:4 chroma
// block); narrower shapes route to the pure-Go reference, keeping the asm a
// single auditable code path. One CDEF row is eight 16-bit samples in one
// vector, and all bit depths share the path because the CDEF buffer is
// 16-bit regardless of source depth.
//
// Bit-exactness with filterBlockPureGo:
//   - diff = sample - x is computed in int16 lanes (pixels and the 0x4000
//     VeryLarge sentinel both fit, and the difference stays within int16).
//   - constrain() is reproduced lane-wise: ABS, variable logical right shift
//     (USHL by a negative count), strength minus shifted clamped to
//     [0, abs] with SMAX/SMIN, then the sign of diff reapplied branchlessly
//     as (limit ^ sign) - sign with sign = diff >> 15.
//   - tap weights accumulate through SMLAL/SMLAL2 into int32 lanes.
//   - the final x + ((8 + sum - (sum<0)) >> 4) folds the negative-sum bias
//     in as sum + (sum >> 31), then +8, then an arithmetic >>4.
//   - when both strengths are active the result clamps to [min, max] over
//     the tap neighbourhood; sentinel lanes are replaced by the running min
//     before the max fold, which skips them exactly like maxClip (the
//     sentinel exceeds every sample so the plain min fold is unaffected).
type filterBlockNEONCtx struct {
	dst    *uint16
	input  *uint16 // pointer to input[inputOrigin]
	dstStr int64   // dst stride in elements
	height int64

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
func cdefFilterBlock8NEON(ctx *filterBlockNEONCtx)

//go:noescape
func cdefFilterBlock8PrimaryNEON(ctx *filterBlockNEONCtx)

//go:noescape
func cdefFilterBlock8SecondaryNEON(ctx *filterBlockNEONCtx)

//go:noescape
func cdefFilterBlock4NEON(ctx *filterBlockNEONCtx)

func filterBlockNEON(dst []uint16, dstStride int, dstOrigin int, input []uint16, inputOrigin int, params BlockFilterParams) {
	if w := int(params.Width); w != 8 && w != 4 {
		filterBlockPureGo(dst, dstStride, dstOrigin, input, inputOrigin, params)
		return
	}
	primaryStrength := int(params.PrimaryStrength)
	secondaryStrength := int(params.SecondaryStrength)
	ctx := makeFilterBlockNEONCtx(dst, dstStride, dstOrigin, input, inputOrigin, params, primaryStrength, secondaryStrength)
	if int(params.Width) == 8 {
		switch {
		case primaryStrength != 0 && secondaryStrength == 0:
			cdefFilterBlock8PrimaryNEON(&ctx)
		case primaryStrength == 0 && secondaryStrength != 0:
			cdefFilterBlock8SecondaryNEON(&ctx)
		default:
			cdefFilterBlock8NEON(&ctx)
		}
	} else {
		cdefFilterBlock4NEON(&ctx)
	}
}

func makeFilterBlockNEONCtx(dst []uint16, dstStride int, dstOrigin int, input []uint16, inputOrigin int, params BlockFilterParams, primaryStrength, secondaryStrength int) filterBlockNEONCtx {
	direction := int(params.Direction)
	coeffShift := int(params.CoeffShift)
	priTaps := cdefPrimaryTaps[(primaryStrength>>coeffShift)&1]

	ctx := filterBlockNEONCtx{
		dst:    &dst[dstOrigin],
		input:  &input[inputOrigin],
		dstStr: int64(dstStride),
		height: int64(params.Height),

		pri0: int64(cdefDirections[direction+2][0]),
		pri1: int64(cdefDirections[direction+2][1]),
		sec0: int64(cdefDirections[direction+4][0]),
		sec1: int64(cdefDirections[direction][0]),
		sec2: int64(cdefDirections[direction+4][1]),
		sec3: int64(cdefDirections[direction][1]),

		priTap0: int64(priTaps[0]),
		priTap1: int64(priTaps[1]),
		secTap0: int64(cdefSecondaryTaps[0]),
		secTap1: int64(cdefSecondaryTaps[1]),

		priStrength: int64(primaryStrength),
		secStrength: int64(secondaryStrength),
		priShift:    int64(constrainShift(primaryStrength, int(params.PrimaryDamping))),
		secShift:    int64(constrainShift(secondaryStrength, int(params.SecondaryDamping))),
	}
	if primaryStrength != 0 {
		ctx.enablePrimary = 1
	}
	if secondaryStrength != 0 {
		ctx.enableSecondary = 1
	}
	if primaryStrength != 0 && secondaryStrength != 0 {
		ctx.clipping = 1
	}
	return ctx
}
