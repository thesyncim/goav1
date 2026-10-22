// Ported from SVT-AV1 v4.1.0: Source/Lib/Codec/cdef.c (svt_cdef_filter_fb
// and svt_cdef_filter_block's subsampling_factor row step, as the encoder
// CDEF search in cdef_process.c cdef_seg_search calls them).
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package cdef

// FilterBlocksU8SubsampledTrusted filters the blocks of one filter block in
// place in an 8-bit trial buffer, every rowStep-th row of each block only,
// with direction and variance grids the caller already derived (the search's
// dirinit). It is svt_cdef_filter_fb with a subsampling_factor: SVT's search
// filters and measures only those rows. Each filtered row is bit-identical
// to the same row of FilterFrameBlocksU8TrustedNoPrimaryDirection's output;
// with rowStep <= 1 it is that call.
//
// Callers own the trusted-geometry contract of FilterFrameBlocksU8Trusted.
func FilterBlocksU8SubsampledTrusted(dst []byte, dstStride int, input []uint16, inputOrigin int, blocks []BlockPosition, directions *DirectionGrid, variances *VarianceGrid, params FrameFilterParams, rowStep int) error {
	if rowStep <= 1 {
		return filterFrameBlocksU8Trusted(dst, dstStride, input, inputOrigin, blocks, directions, variances, params, false)
	}
	if len(blocks) == 0 {
		return nil
	}
	if params.CoeffShift != 0 {
		return ErrInvalidCDEF
	}
	primary := int(params.Level)
	secondary := int(params.SecondaryStrength)
	damping := int(params.Damping)
	if params.Plane != PlaneY {
		damping--
	}
	if damping < 0 {
		return ErrInvalidCDEF
	}
	if primary == 0 && secondary == 0 {
		return nil
	}
	bwLog2 := 3 - int(params.XDec)
	bhLog2 := 3 - int(params.YDec)
	lumaAdjust := params.Plane == PlaneY && primary != 0
	bp := BlockFilterParams{
		SecondaryStrength: uint8(secondary),
		PrimaryDamping:    uint8(damping),
		SecondaryDamping:  uint8(damping),
		Width:             uint8(1 << bwLog2),
		Height:            1,
	}
	for _, block := range blocks {
		by := int(block.BY)
		bx := int(block.BX)
		strength := primary
		if lumaAdjust {
			strength = adjustStrength(primary, variances[by][bx])
		}
		if strength == 0 && secondary == 0 {
			continue
		}
		bp.PrimaryStrength = uint8(strength)
		bp.Direction = 0
		if primary != 0 {
			bp.Direction = directions[by][bx]
		}
		srcOrigin := inputOrigin + ((by * BStride) << bhLog2) + (bx << bwLog2)
		dstOrigin := (by<<bhLog2)*dstStride + (bx << bwLog2)
		for i := 0; i < 1<<bhLog2; i += rowStep {
			filterBlockU8Impl(dst, dstStride, dstOrigin+i*dstStride, input, srcOrigin+i*BStride, bp)
		}
	}
	return nil
}
