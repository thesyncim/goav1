// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package cdef

import (
	"simd/archsimd"
	"unsafe"
)

// cdefShiftCount is the per-block constrain() shift in the form the NEON
// variable shift consumes: a broadcast negative count, so USHL shifts right.
type cdefShiftCount = archsimd.Int16x8

func cdefShiftCountOf(n int) cdefShiftCount {
	return archsimd.BroadcastInt16x8(-int16(n))
}

// cdefShrU16 is a logical right shift of each lane by the per-block count.
func cdefShrU16(v archsimd.Uint16x8, sh cdefShiftCount) archsimd.Uint16x8 {
	return v.Shift(sh)
}

// filterUnitBlocks binds the Go SIMD unit-level loop on arm64 (NEON is
// architecturally mandatory there). See filter_dispatch.go for why this is a
// build-tag binding instead of a func variable.
func filterUnitBlocks(dst []uint16, dstStride int, input []uint16, inputOrigin int, blocks []BlockPosition, directions *DirectionGrid, variances *VarianceGrid, u unitFilterParams, trusted bool) error {
	return filterUnitBlocksSIMD(dst, dstStride, input, inputOrigin, blocks, directions, variances, u, trusted)
}

// filterUnitBlocksSIMD mirrors dav1d's per-superblock cdef apply loop
// (src/cdef_apply_tmpl.c): the per-unit invariants (dst stride, block
// geometry, secondary strength/shift/taps, and for chroma the primary
// constants too) are computed once per filter unit, each block fills only the
// per-block fields, and blocks whose adjusted primary strength and secondary
// strength are both zero skip the filter kernel entirely (identity copy, as
// dav1d skips the fb call for its in-place buffer). Output is bit-identical
// to filterUnitBlocksPureGo.
func filterUnitBlocksSIMD(dst []uint16, dstStride int, input []uint16, inputOrigin int, blocks []BlockPosition, directions *DirectionGrid, variances *VarianceGrid, u unitFilterParams, trusted bool) error {
	if !trusted || (u.blockWidth != 8 && u.blockWidth != 4) {
		return filterUnitBlocksPureGo(dst, dstStride, input, inputOrigin, blocks, directions, variances, u, trusted)
	}
	secondaryStrength := u.secondaryStrength
	ctx := cdefSIMDCtx{
		dstStr: dstStride * 2,
		height: u.blockHeight,
	}
	ctx.setSecondary(secondaryStrength, u.damping)
	strength := u.primaryStrength
	if !u.lumaAdjust {
		ctx.setPrimary(strength, u.damping, u.coeffShift)
	}
	for _, block := range blocks {
		by := int(block.BY)
		bx := int(block.BX)
		if u.lumaAdjust {
			strength = adjustStrength(u.primaryStrength, variances[by][bx])
			ctx.setPrimary(strength, u.damping, u.coeffShift)
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
		ctx.setDirection(dir)
		ctx.dst = unsafe.Pointer(&dst[dstOrigin])
		ctx.input = unsafe.Pointer(&input[srcOrigin])
		cdefFilterBlockSIMD(&ctx, u.blockWidth)
	}
	return nil
}

// filterUnitBlocksU8 binds the Go SIMD 8-bit unit-level loop on arm64. See
// filter_u8_dispatch.go for why this is a build-tag binding.
func filterUnitBlocksU8(dst []byte, dstStride int, input []uint16, inputOrigin int, blocks []BlockPosition, directions *DirectionGrid, variances *VarianceGrid, u unitFilterParams) error {
	return filterUnitBlocksU8SIMD(dst, dstStride, input, inputOrigin, blocks, directions, variances, u)
}

// filterUnitBlocksU8SIMD mirrors filterUnitBlocksSIMD with in-place 8-bit
// stores: per-unit invariants hoisted, per-block strength adjust, and blocks
// whose adjusted primary strength and secondary strength are both zero are
// skipped outright (in place there is nothing to copy, dav1d's
// cdef_apply_tmpl.c skip). Output is bit-identical to
// filterUnitBlocksU8PureGo.
func filterUnitBlocksU8SIMD(dst []byte, dstStride int, input []uint16, inputOrigin int, blocks []BlockPosition, directions *DirectionGrid, variances *VarianceGrid, u unitFilterParams) error {
	if u.blockWidth != 8 && u.blockWidth != 4 {
		return filterUnitBlocksU8PureGo(dst, dstStride, input, inputOrigin, blocks, directions, variances, u)
	}
	secondaryStrength := u.secondaryStrength
	ctx := cdefSIMDCtx{
		dstStr: dstStride,
		height: u.blockHeight,
		u8:     true,
	}
	ctx.setSecondary(secondaryStrength, u.damping)
	strength := u.primaryStrength
	if !u.lumaAdjust {
		ctx.setPrimary(strength, u.damping, 0)
	}
	// Interior routing (dav1d edges == 0xf): the sentinel-free footprint is
	// proven once, at the first fused block, and reused for the unit.
	interiorEligible := u.blockWidth == 8 && u.blockHeight%2 == 0
	interiorChecked := false
	useInterior := false
	for _, block := range blocks {
		by := int(block.BY)
		bx := int(block.BX)
		if u.lumaAdjust {
			strength = adjustStrength(u.primaryStrength, variances[by][bx])
			ctx.setPrimary(strength, u.damping, 0)
		}
		if strength == 0 && secondaryStrength == 0 {
			continue
		}
		if !interiorChecked && interiorEligible && strength != 0 && secondaryStrength != 0 {
			interiorChecked = true
			useInterior = cdefUnitInteriorU8(input, inputOrigin, blocks, u.bwLog2, u.bhLog2)
		}
		srcOrigin := inputOrigin + ((by * BStride) << u.bhLog2) + (bx << u.bwLog2)
		dstOrigin := (by<<u.bhLog2)*dstStride + (bx << u.bwLog2)
		dir := 0
		if u.primaryStrength != 0 {
			dir = int(directions[by][bx])
		}
		ctx.setDirection(dir)
		ctx.dst = unsafe.Pointer(&dst[dstOrigin])
		ctx.input = unsafe.Pointer(&input[srcOrigin])
		if useInterior && strength != 0 && secondaryStrength != 0 {
			cdefFilterBlock8InteriorSIMD(&ctx)
		} else {
			cdefFilterBlockSIMD(&ctx, u.blockWidth)
		}
	}
	return nil
}

// cdefNarrowU8 keeps the low byte of each lane, the same bits as byte(y) in
// the pure-Go reference (XTN on arm64).
func cdefNarrowU8(y archsimd.Int16x8) archsimd.Uint8x16 {
	return y.ToBits().TruncToUint8()
}
