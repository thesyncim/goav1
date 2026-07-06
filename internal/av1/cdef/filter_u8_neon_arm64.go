// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

package cdef

// NEON 8-bit-dst CDEF block filters, the dav1d 8bpc epilogue
// (src/arm/64/cdef.S cdef_filter{8,4}_8bpc_neon: xtn to bytes + st1 {v0.8b} /
// st1 {v0.s}[lane]) applied to goav1's strength-split kernels. The tap math
// is byte-for-byte the uint16 kernels in filter_neon_arm64.s /
// filter_neon_arm64_split.s — same loads from the uint16 CDEF input buffer,
// same constrain/accumulate/finalize, same 0x4000 VeryLarge handling — only
// the final store narrows the int16 result to uint8 (lossless under the
// 8-bit contract documented in filter_u8.go) and the dst stride is in bytes.
type filterBlockU8NEONCtx struct {
	dst    *byte
	input  *uint16 // pointer to input[inputOrigin]
	dstStr int64   // dst stride in BYTES
	height int64

	pri0 int64 // direction offsets in input elements (signed)
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
func cdefFilterBlock8U8NEON(ctx *filterBlockU8NEONCtx)

//go:noescape
func cdefFilterBlock8PrimaryU8NEON(ctx *filterBlockU8NEONCtx)

//go:noescape
func cdefFilterBlock8SecondaryU8NEON(ctx *filterBlockU8NEONCtx)

//go:noescape
func cdefFilterBlock4U8NEON(ctx *filterBlockU8NEONCtx)

//go:noescape
func cdefFilterBlock4PrimaryU8NEON(ctx *filterBlockU8NEONCtx)

//go:noescape
func cdefFilterBlock4SecondaryU8NEON(ctx *filterBlockU8NEONCtx)

// dispatchFilterBlockU8NEON routes a prepared ctx to the width- and
// strength-specialized kernel, mirroring dispatchFilterBlockNEON (dav1d's
// pri/sec/pri_sec split of src/arm/64/cdef.S).
func dispatchFilterBlockU8NEON(ctx *filterBlockU8NEONCtx, width int, primaryStrength int, secondaryStrength int) {
	if width == 8 {
		switch {
		case primaryStrength != 0 && secondaryStrength == 0:
			cdefFilterBlock8PrimaryU8NEON(ctx)
		case primaryStrength == 0 && secondaryStrength != 0:
			cdefFilterBlock8SecondaryU8NEON(ctx)
		default:
			cdefFilterBlock8U8NEON(ctx)
		}
		return
	}
	switch {
	case primaryStrength != 0 && secondaryStrength == 0:
		cdefFilterBlock4PrimaryU8NEON(ctx)
	case primaryStrength == 0 && secondaryStrength != 0:
		cdefFilterBlock4SecondaryU8NEON(ctx)
	default:
		cdefFilterBlock4U8NEON(ctx)
	}
}

// filterUnitBlocksU8 binds the NEON 8-bit unit-level loop on arm64. See
// filter_u8_dispatch.go for why this is a build-tag binding.
func filterUnitBlocksU8(dst []byte, dstStride int, input []uint16, inputOrigin int, blocks []BlockPosition, directions *DirectionGrid, variances *VarianceGrid, u unitFilterParams) error {
	return filterUnitBlocksU8NEON(dst, dstStride, input, inputOrigin, blocks, directions, variances, u)
}

// filterUnitBlocksU8NEON mirrors filterUnitBlocksNEON with in-place 8-bit
// stores: per-unit invariants hoisted, per-block strength adjust, and blocks
// whose adjusted primary strength and secondary strength are both zero are
// skipped outright (in place there is nothing to copy, dav1d's
// cdef_apply_tmpl.c skip). Output is bit-identical to
// filterUnitBlocksU8PureGo.
func filterUnitBlocksU8NEON(dst []byte, dstStride int, input []uint16, inputOrigin int, blocks []BlockPosition, directions *DirectionGrid, variances *VarianceGrid, u unitFilterParams) error {
	if u.blockWidth != 8 && u.blockWidth != 4 {
		return filterUnitBlocksU8PureGo(dst, dstStride, input, inputOrigin, blocks, directions, variances, u)
	}
	secondaryStrength := u.secondaryStrength
	ctx := filterBlockU8NEONCtx{
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
		setFilterBlockU8NEONCtxPrimary(&ctx, strength, secondaryStrength, u.damping, u.coeffShift)
	}
	for _, block := range blocks {
		by := int(block.BY)
		bx := int(block.BX)
		if u.lumaAdjust {
			strength = adjustStrength(u.primaryStrength, variances[by][bx])
			setFilterBlockU8NEONCtxPrimary(&ctx, strength, secondaryStrength, u.damping, u.coeffShift)
		}
		if strength == 0 && secondaryStrength == 0 {
			continue
		}
		srcOrigin := inputOrigin + ((by * BStride) << u.bhLog2) + (bx << u.bwLog2)
		dstOrigin := (by<<u.bhLog2)*dstStride + (bx << u.bwLog2)
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
		dispatchFilterBlockU8NEON(&ctx, u.blockWidth, strength, secondaryStrength)
	}
	return nil
}

// setFilterBlockU8NEONCtxPrimary fills the primary-strength-dependent ctx
// fields; identical to setFilterBlockNEONCtxPrimary.
func setFilterBlockU8NEONCtxPrimary(ctx *filterBlockU8NEONCtx, primaryStrength int, secondaryStrength int, damping int, coeffShift int) {
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

// filterBlockU8NEON is the single-block NEON entry backing the
// filterBlockU8Impl dispatch slot on arm64; narrow shapes fall back to the
// pure-Go reference exactly like filterBlockNEON.
func filterBlockU8NEON(dst []byte, dstStride int, dstOrigin int, input []uint16, inputOrigin int, params BlockFilterParams) {
	if w := int(params.Width); (w != 8 && w != 4) || (w == 4 && params.Height&1 != 0) {
		filterBlockU8PureGo(dst, dstStride, dstOrigin, input, inputOrigin, params)
		return
	}
	primaryStrength := int(params.PrimaryStrength)
	secondaryStrength := int(params.SecondaryStrength)
	direction := int(params.Direction)
	coeffShift := int(params.CoeffShift)
	priTaps := cdefPrimaryTaps[(primaryStrength>>coeffShift)&1]
	ctx := filterBlockU8NEONCtx{
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
	dispatchFilterBlockU8NEON(&ctx, int(params.Width), primaryStrength, secondaryStrength)
}
