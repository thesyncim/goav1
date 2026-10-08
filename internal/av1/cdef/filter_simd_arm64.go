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
