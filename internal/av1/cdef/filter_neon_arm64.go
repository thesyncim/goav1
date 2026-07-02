// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

package cdef

// NEON-accelerated CDEF block filter, ported from dav1d's strength-split
// kernels (src/arm/64/cdef_tmpl.S: cdef_filter{8,4}_{pri,sec,pri_sec}_neon).
// 8-wide blocks run one row per vector; 4-wide blocks pack two rows into one
// vector per iteration like dav1d's load_px w=4 path. All bit depths share
// the path because the CDEF buffer is 16-bit regardless of source depth.
//
// Bit-exactness with filterBlockPureGo:
//   - constrain() is dav1d's handle_pixel: clip = uqsub(threshold,
//     uabd(p, x) >> shift) followed by smax(smin(p - x, clip), -clip),
//     algebraically identical to the scalar sign-fold form.
//   - tap weights accumulate through 16-bit MLA lanes; the worst-case
//     |sum| = 2*(4+2)*pri_strength + 4*(2+1)*sec_strength = 3648 at 12-bit,
//     so int16 accumulation never wraps.
//   - the final x + ((8 + sum - (sum<0)) >> 4) is dav1d's cmlt/add/srshr #4.
//   - when both strengths are active the result clamps to [min, max] over
//     the tap neighbourhood; see filter_neon_arm64.s for how the 0x4000
//     VeryLarge sentinel (dav1d pads with INT16_MIN instead) is skipped in
//     the max fold via an xor-domain umax.
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

//go:noescape
func cdefFilterBlock4PrimaryNEON(ctx *filterBlockNEONCtx)

//go:noescape
func cdefFilterBlock4SecondaryNEON(ctx *filterBlockNEONCtx)

func filterBlockNEON(dst []uint16, dstStride int, dstOrigin int, input []uint16, inputOrigin int, params BlockFilterParams) {
	if w := int(params.Width); (w != 8 && w != 4) || (w == 4 && params.Height&1 != 0) {
		// The 4-wide kernels process two rows per iteration (dav1d's
		// load_px w=4 packing), so odd heights fall back.
		filterBlockPureGo(dst, dstStride, dstOrigin, input, inputOrigin, params)
		return
	}
	primaryStrength := int(params.PrimaryStrength)
	secondaryStrength := int(params.SecondaryStrength)
	ctx := makeFilterBlockNEONCtx(dst, dstStride, dstOrigin, input, inputOrigin, params, primaryStrength, secondaryStrength)
	dispatchFilterBlockNEON(&ctx, int(params.Width), primaryStrength, secondaryStrength)
}

// dispatchFilterBlockNEON routes a prepared ctx to the width- and
// strength-specialized kernel, mirroring dav1d's pri/sec/pri+sec kernel split
// (src/arm/64/cdef.S filter_func pri/sec/pri_sec instantiations).
func dispatchFilterBlockNEON(ctx *filterBlockNEONCtx, width int, primaryStrength int, secondaryStrength int) {
	if width == 8 {
		switch {
		case primaryStrength != 0 && secondaryStrength == 0:
			cdefFilterBlock8PrimaryNEON(ctx)
		case primaryStrength == 0 && secondaryStrength != 0:
			cdefFilterBlock8SecondaryNEON(ctx)
		default:
			cdefFilterBlock8NEON(ctx)
		}
		return
	}
	switch {
	case primaryStrength != 0 && secondaryStrength == 0:
		cdefFilterBlock4PrimaryNEON(ctx)
	case primaryStrength == 0 && secondaryStrength != 0:
		cdefFilterBlock4SecondaryNEON(ctx)
	default:
		cdefFilterBlock4NEON(ctx)
	}
}

// filterUnitBlocks binds the NEON unit-level loop on arm64 (NEON is
// architecturally mandatory there). See filter_dispatch.go for why this is a
// build-tag binding instead of a func variable.
func filterUnitBlocks(dst []uint16, dstStride int, input []uint16, inputOrigin int, blocks []BlockPosition, directions *DirectionGrid, variances *VarianceGrid, u unitFilterParams, trusted bool) error {
	return filterUnitBlocksNEON(dst, dstStride, input, inputOrigin, blocks, directions, variances, u, trusted)
}

// filterUnitBlocksNEON mirrors dav1d's per-superblock cdef apply loop
// (src/cdef_apply_tmpl.c): the per-unit invariants (dst stride, block
// geometry, secondary strength/shift/taps, and for chroma the primary
// constants too) are computed once per filter unit, each block fills only the
// per-block fields, and blocks whose adjusted primary strength and secondary
// strength are both zero skip the filter kernel entirely (identity copy, as
// dav1d skips the fb call for its in-place buffer). Output is bit-identical
// to filterUnitBlocksPureGo.
func filterUnitBlocksNEON(dst []uint16, dstStride int, input []uint16, inputOrigin int, blocks []BlockPosition, directions *DirectionGrid, variances *VarianceGrid, u unitFilterParams, trusted bool) error {
	if !trusted || (u.blockWidth != 8 && u.blockWidth != 4) {
		return filterUnitBlocksPureGo(dst, dstStride, input, inputOrigin, blocks, directions, variances, u, trusted)
	}
	secondaryStrength := u.secondaryStrength
	ctx := filterBlockNEONCtx{
		dstStr:      int64(dstStride),
		height:      int64(u.blockHeight),
		secTap0:     int64(cdefSecondaryTaps[0]),
		secTap1:     int64(cdefSecondaryTaps[1]),
		secStrength: int64(secondaryStrength),
		secShift:    int64(constrainShift(secondaryStrength, u.damping)),
	}
	if secondaryStrength != 0 {
		ctx.enableSecondary = 1
	}
	strength := u.primaryStrength
	if !u.lumaAdjust {
		setFilterBlockNEONCtxPrimary(&ctx, strength, secondaryStrength, u.damping, u.coeffShift)
	}
	for _, block := range blocks {
		by := int(block.BY)
		bx := int(block.BX)
		if u.lumaAdjust {
			strength = adjustStrength(u.primaryStrength, variances[by][bx])
			setFilterBlockNEONCtxPrimary(&ctx, strength, secondaryStrength, u.damping, u.coeffShift)
		}
		srcOrigin := inputOrigin + ((by * BStride) << u.bhLog2) + (bx << u.bwLog2)
		dstOrigin := (by<<u.bhLog2)*dstStride + (bx << u.bwLog2)
		if strength == 0 && secondaryStrength == 0 {
			// See filterUnitBlocksPureGo: dav1d's zero-strength skip.
			copyBlockIdentity(dst, dstStride, dstOrigin, input, srcOrigin, u.blockWidth, u.blockHeight)
			continue
		}
		dir := 0
		if u.primaryStrength != 0 {
			dir = int(directions[by][bx])
		}
		ctx.pri0 = int64(cdefDirections[dir+2][0])
		ctx.pri1 = int64(cdefDirections[dir+2][1])
		ctx.sec0 = int64(cdefDirections[dir+4][0])
		ctx.sec1 = int64(cdefDirections[dir][0])
		ctx.sec2 = int64(cdefDirections[dir+4][1])
		ctx.sec3 = int64(cdefDirections[dir][1])
		ctx.dst = &dst[dstOrigin]
		ctx.input = &input[srcOrigin]
		dispatchFilterBlockNEON(&ctx, u.blockWidth, strength, secondaryStrength)
	}
	return nil
}

// setFilterBlockNEONCtxPrimary fills the primary-strength-dependent ctx
// fields (taps, strength, constrain shift, enable/clip flags).
func setFilterBlockNEONCtxPrimary(ctx *filterBlockNEONCtx, primaryStrength int, secondaryStrength int, damping int, coeffShift int) {
	priTaps := cdefPrimaryTaps[(primaryStrength>>coeffShift)&1]
	ctx.priTap0 = int64(priTaps[0])
	ctx.priTap1 = int64(priTaps[1])
	ctx.priStrength = int64(primaryStrength)
	ctx.priShift = int64(constrainShift(primaryStrength, damping))
	ctx.enablePrimary = 0
	ctx.clipping = 0
	if primaryStrength != 0 {
		ctx.enablePrimary = 1
		if secondaryStrength != 0 {
			ctx.clipping = 1
		}
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
