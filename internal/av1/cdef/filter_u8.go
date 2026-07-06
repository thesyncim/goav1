// Ported from libaom:
//   av1/common/cdef_block.c
// with the 8bpc pixel-domain split taken from dav1d:
//   src/cdef_tmpl.c (cdef_filter_block_c, pixel=uint8_t)
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

package cdef

// 8-bit-pixel CDEF kernel family. dav1d's 8bpc CDEF filter
// (src/cdef_tmpl.c cdef_filter_block_c with pixel=uint8_t, src/arm/64/cdef.S)
// keeps its padded tap buffer in 16 bits (padding() builds an int16 tmp with
// large-value sentinels from uint8 reads) but the FRAME side is uint8: the
// center pixel is read from the 8-bit dst and the filtered result is written
// straight back as a byte, in place. This file is that split mapped onto
// goav1's kernel family: the taps still come from the uint16 CDEF input
// buffer with libaom's 0x4000 VeryLarge sentinel (identical semantics to the
// uint16 kernels in filter.go), while dst is the frame's 8-bit plane.
//
// Contract (checked by the FilterFrameBlocksU8Trusted entry):
//   - params.CoeffShift == 0 (8-bit streams only);
//   - every interior sample of the current unit inside input holds a real
//     8-bit pixel (0..255); the VeryLarge sentinel appears only in the halo.
//     Under that contract the filter output always stays in [0, 255] (the
//     result moves at most max|tap-x| toward the real-tap neighbourhood:
//     the tap weights sum to 12 < 16 and constrain() bounds each term by
//     |tap-x|, while sentinel taps contribute 0), so narrowing to uint8 is
//     lossless and the family is bit-identical to the uint16 kernels fed the
//     same input followed by a uint8 store. u8_differential_test.go proves
//     it across strengths, dampings, directions, edge-sentinel patterns and
//     block sizes.
//
// The direction search reads the 8-bit frame directly (dav1d's
// cdef_find_dir_8bpc takes the pixel dst pointer), not the padded buffer; an
// 8x8 direction block never touches the halo, so the values are identical to
// the widened interior.

// filterBlockU8PureGo is the canonical bit-exact 8-bit-dst CDEF block filter:
// filterBlockPureGo (filter.go) with the final store narrowed to a byte. It
// assumes the FilterFrameBlocksU8Trusted contract; the dispatch slot in
// filter_u8_dispatch.go defaults to it so every build has a working
// reference path.
func filterBlockU8PureGo(dst []byte, dstStride int, dstOrigin int, input []uint16, inputOrigin int, params BlockFilterParams) {
	primaryStrength := int(params.PrimaryStrength)
	secondaryStrength := int(params.SecondaryStrength)
	direction := int(params.Direction)
	primaryDamping := int(params.PrimaryDamping)
	secondaryDamping := int(params.SecondaryDamping)
	coeffShift := int(params.CoeffShift)
	width := int(params.Width)
	height := int(params.Height)

	enablePrimary := primaryStrength != 0
	enableSecondary := secondaryStrength != 0
	clippingRequired := enablePrimary && enableSecondary
	priTaps := cdefPrimaryTaps[(primaryStrength>>coeffShift)&1]

	pri0 := int(cdefDirections[direction+2][0])
	pri1 := int(cdefDirections[direction+2][1])
	sec0a := int(cdefDirections[direction+4][0])
	sec0b := int(cdefDirections[direction+4][1])
	sec1a := int(cdefDirections[direction][0])
	sec1b := int(cdefDirections[direction][1])
	priTap0 := int(priTaps[0])
	priTap1 := int(priTaps[1])
	secTap0 := int(cdefSecondaryTaps[0])
	secTap1 := int(cdefSecondaryTaps[1])
	priStrength := primaryStrength
	secStrength := secondaryStrength
	priShift := constrainShift(priStrength, primaryDamping)
	secShift := constrainShift(secStrength, secondaryDamping)

	for row := 0; row < height; row++ {
		rowBase := inputOrigin + row*BStride
		dstRow := dstOrigin + row*dstStride
		for col := 0; col < width; col++ {
			base := rowBase + col
			sum := 0
			x := int(input[base])
			maxSample := x
			minSample := x
			if enablePrimary {
				p0 := int(input[base+pri0])
				p1 := int(input[base-pri0])
				p2 := int(input[base+pri1])
				p3 := int(input[base-pri1])
				sum += priTap0 * constrainShifted(p0-x, priStrength, priShift)
				sum += priTap0 * constrainShifted(p1-x, priStrength, priShift)
				sum += priTap1 * constrainShifted(p2-x, priStrength, priShift)
				sum += priTap1 * constrainShifted(p3-x, priStrength, priShift)
				if clippingRequired {
					maxSample = maxClip(maxSample, p0, p1, p2, p3)
					minSample = min4(minSample, p0, p1, p2, p3)
				}
			}
			if enableSecondary {
				s0 := int(input[base+sec0a])
				s1 := int(input[base-sec0a])
				s2 := int(input[base+sec1a])
				s3 := int(input[base-sec1a])
				s4 := int(input[base+sec0b])
				s5 := int(input[base-sec0b])
				s6 := int(input[base+sec1b])
				s7 := int(input[base-sec1b])
				if clippingRequired {
					maxSample = maxClip(maxSample, s0, s1, s2, s3)
					maxSample = maxClip(maxSample, s4, s5, s6, s7)
					minSample = min4(minSample, s0, s1, s2, s3)
					minSample = min4(minSample, s4, s5, s6, s7)
				}
				sum += secTap0 * constrainShifted(s0-x, secStrength, secShift)
				sum += secTap0 * constrainShifted(s1-x, secStrength, secShift)
				sum += secTap0 * constrainShifted(s2-x, secStrength, secShift)
				sum += secTap0 * constrainShifted(s3-x, secStrength, secShift)
				sum += secTap1 * constrainShifted(s4-x, secStrength, secShift)
				sum += secTap1 * constrainShifted(s5-x, secStrength, secShift)
				sum += secTap1 * constrainShifted(s6-x, secStrength, secShift)
				sum += secTap1 * constrainShifted(s7-x, secStrength, secShift)
			}
			y := x + ((8 + sum - boolToInt(sum < 0)) >> 4)
			if clippingRequired {
				y = clampInt(y, minSample, maxSample)
			}
			dst[dstRow+col] = byte(y)
		}
	}
}

// FilterFrameBlocksU8Trusted is the 8-bit-frame counterpart of
// FilterFrameBlocksTrusted: dst is the frame's uint8 plane view positioned at
// the unit origin and the kernels write it IN PLACE (dav1d's 8bpc
// cdef_filter_block shape), so zero-strength blocks are skipped outright
// instead of copied. The luma direction search reads dst directly (dav1d
// cdef_find_dir_8bpc); when the unit is a direction-only luma pass (both
// strengths zero after the coeffShift-0 decode) input is never read, so
// callers may pass an empty tap buffer for such units.
//
// Callers own the trusted-geometry contract of FilterFrameBlocksTrusted plus
// the 8-bit contract documented at the top of this file. Output bytes are
// bit-identical to running FilterFrameBlocksTrusted on the widened unit and
// narrowing the written blocks back to uint8.
func FilterFrameBlocksU8Trusted(dst []byte, dstStride int, input []uint16, inputOrigin int, blocks []BlockPosition, directions *DirectionGrid, variances *VarianceGrid, params FrameFilterParams) error {
	if len(blocks) == 0 {
		return nil
	}
	if params.CoeffShift != 0 {
		return ErrInvalidCDEF
	}
	primaryStrength := int(params.Level)
	secondaryStrength := int(params.SecondaryStrength)
	damping := int(params.Damping)
	if params.Plane != PlaneY {
		damping--
	}
	if damping < 0 {
		return ErrInvalidCDEF
	}
	xDec := int(params.XDec)
	yDec := int(params.YDec)
	bwLog2 := 3 - xDec
	bhLog2 := 3 - yDec

	if params.Plane == PlaneY {
		// The direction pass runs for every block before any in-place write:
		// block interiors are disjoint from other blocks' filtered output, so
		// the values seen here are the pre-CDEF pixels either way.
		for i := 0; i < len(blocks); {
			block := blocks[i]
			by := int(block.BY)
			bx := int(block.BX)
			srcOrigin := ((by * dstStride) << bhLog2) + (bx << bwLog2)
			if bwLog2 == 3 && i+1 < len(blocks) && blocks[i+1].BY == block.BY && blocks[i+1].BX == block.BX+1 {
				dir1, variance1, dir2, variance2 := findDirectionDualU8Unchecked(dst[srcOrigin:], dst[srcOrigin+8:], dstStride)
				directions[by][bx] = uint8(dir1)
				variances[by][bx] = variance1
				directions[by][bx+1] = uint8(dir2)
				variances[by][bx+1] = variance2
				i += 2
				continue
			}
			dir, variance := findDirectionU8Unchecked(dst[srcOrigin:], dstStride)
			directions[by][bx] = uint8(dir)
			variances[by][bx] = variance
			i++
		}
	}
	if params.Plane == PlaneU && params.XDec != params.YDec {
		if !convertChromaDirections(blocks, directions, xDec) {
			return ErrInvalidCDEF
		}
	}

	unitParams := unitFilterParams{
		primaryStrength:   primaryStrength,
		secondaryStrength: secondaryStrength,
		damping:           damping,
		coeffShift:        0,
		bwLog2:            bwLog2,
		bhLog2:            bhLog2,
		blockWidth:        1 << bwLog2,
		blockHeight:       1 << bhLog2,
		lumaAdjust:        params.Plane == PlaneY,
	}
	if unitParams.primaryStrength == 0 && unitParams.secondaryStrength == 0 {
		// Direction-only pass: dav1d never calls the filter with both levels
		// zero; in place there is nothing to write.
		return nil
	}
	return filterUnitBlocksU8(dst, dstStride, input, inputOrigin, blocks, directions, variances, unitParams)
}

// filterUnitBlocksU8PureGo is the canonical 8-bit-dst per-unit block loop:
// filterUnitBlocksPureGo with in-place skip semantics (a block whose adjusted
// primary strength and secondary strength are both zero keeps its frame
// bytes, exactly dav1d's cdef_apply_tmpl.c skip).
func filterUnitBlocksU8PureGo(dst []byte, dstStride int, input []uint16, inputOrigin int, blocks []BlockPosition, directions *DirectionGrid, variances *VarianceGrid, u unitFilterParams) error {
	for _, block := range blocks {
		by := int(block.BY)
		bx := int(block.BX)
		strength := u.primaryStrength
		if u.lumaAdjust {
			strength = adjustStrength(u.primaryStrength, variances[by][bx])
		}
		if strength == 0 && u.secondaryStrength == 0 {
			continue
		}
		dir := 0
		if u.primaryStrength != 0 {
			dir = int(directions[by][bx])
		}
		srcOrigin := inputOrigin + ((by * BStride) << u.bhLog2) + (bx << u.bwLog2)
		dstOrigin := (by<<u.bhLog2)*dstStride + (bx << u.bwLog2)
		blockParams := BlockFilterParams{
			PrimaryStrength:   uint8(strength),
			SecondaryStrength: uint8(u.secondaryStrength),
			Direction:         uint8(dir),
			PrimaryDamping:    uint8(u.damping),
			SecondaryDamping:  uint8(u.damping),
			CoeffShift:        0,
			Width:             uint8(u.blockWidth),
			Height:            uint8(u.blockHeight),
		}
		filterBlockU8Impl(dst, dstStride, dstOrigin, input, srcOrigin, blockParams)
	}
	return nil
}
