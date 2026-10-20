// Ported from libaom:
//   av1/common/cdef_block.c
// with the 8bpc pixel-domain split taken from dav1d:
//   src/cdef_tmpl.c (cdef_filter_block_c, pixel=uint8_t)
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

package cdef

import "math/bits"

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
//
// Like dav1d's cdef_filter_block_c it runs one loop per strength mode
// (primary only, secondary only, both), so the per-pixel work has no mode
// branches, and constrain is branch-free. Row slices are cut once per row so
// the tap loads carry no per-pixel bounds checks.
func filterBlockU8PureGo(dst []byte, dstStride int, dstOrigin int, input []uint16, inputOrigin int, params BlockFilterParams) {
	primaryStrength := int(params.PrimaryStrength)
	secondaryStrength := int(params.SecondaryStrength)
	direction := int(params.Direction)
	t := cdefU8Taps{
		priTap0:  int(cdefPrimaryTaps[(primaryStrength>>int(params.CoeffShift))&1][0]),
		priTap1:  int(cdefPrimaryTaps[(primaryStrength>>int(params.CoeffShift))&1][1]),
		priStr:   primaryStrength,
		priShift: constrainShift(primaryStrength, int(params.PrimaryDamping)),
		secStr:   secondaryStrength,
		secShift: constrainShift(secondaryStrength, int(params.SecondaryDamping)),
		pri0:     int(cdefDirections[direction+2][0]),
		pri1:     int(cdefDirections[direction+2][1]),
		sec0:     int(cdefDirections[direction+4][0]),
		sec1:     int(cdefDirections[direction+4][1]),
		sec2:     int(cdefDirections[direction][0]),
		sec3:     int(cdefDirections[direction][1]),
	}
	width := int(params.Width)
	height := int(params.Height)
	switch {
	case primaryStrength != 0 && secondaryStrength != 0:
		plut, slut := cdefLUT(t.priStr, t.priShift), cdefLUT(t.secStr, t.secShift)
		if plut != nil && slut != nil {
			for row := 0; row < height; row++ {
				filterRowU8PriSecLUT(dst[dstOrigin+row*dstStride:][:width], input, inputOrigin+row*BStride, &t, plut, slut)
			}
			return
		}
		for row := 0; row < height; row++ {
			filterRowU8PriSec(dst[dstOrigin+row*dstStride:][:width], input, inputOrigin+row*BStride, &t)
		}
	case primaryStrength != 0:
		if lut := cdefLUT(t.priStr, t.priShift); lut != nil {
			for row := 0; row < height; row++ {
				filterRowU8PriLUT(dst[dstOrigin+row*dstStride:][:width], input, inputOrigin+row*BStride, &t, lut)
			}
			return
		}
		for row := 0; row < height; row++ {
			filterRowU8Pri(dst[dstOrigin+row*dstStride:][:width], input, inputOrigin+row*BStride, &t)
		}
	case secondaryStrength != 0:
		if lut := cdefLUT(t.secStr, t.secShift); lut != nil {
			for row := 0; row < height; row++ {
				filterRowU8SecLUT(dst[dstOrigin+row*dstStride:][:width], input, inputOrigin+row*BStride, &t, lut)
			}
			return
		}
		for row := 0; row < height; row++ {
			filterRowU8Sec(dst[dstOrigin+row*dstStride:][:width], input, inputOrigin+row*BStride, &t)
		}
	default:
		for row := 0; row < height; row++ {
			d := dst[dstOrigin+row*dstStride:][:width]
			src := input[inputOrigin+row*BStride:][:len(d)]
			for i := range d {
				d[i] = byte(src[i])
			}
		}
	}
}

// cdefU8Taps holds the per-block constants of filterBlockU8PureGo: tap
// weights, constrain thresholds and shifts, and the six direction offsets
// into the BStride-pitched input.
type cdefU8Taps struct {
	priTap0, priTap1 int
	priStr, priShift int
	secStr, secShift int
	pri0, pri1       int
	sec0, sec1       int
	sec2, sec3       int
}

const cdefSignShift = bits.UintSize - 1

// constrainU8 is constrainShifted for a nonzero threshold, without branches.
func constrainU8(diff int, threshold int, shift int) int {
	s := diff >> cdefSignShift
	a := (diff ^ s) - s
	l := min(max(threshold-(a>>shift), 0), a)
	return (l ^ s) - s
}

// cdefRound applies the CDEF output rounding: x + (8 + sum - (sum < 0)) >> 4.
func cdefRound(x int, sum int) int {
	return x + ((8 + sum + (sum >> cdefSignShift)) >> 4)
}

// cdefTapRow returns the width-long run of input starting off past base.
func cdefTapRow(input []uint16, base int, off int, width int) []uint16 {
	return input[base+off:][:width]
}

func filterRowU8Pri(d []byte, input []uint16, base int, t *cdefU8Taps) {
	n := len(d)
	c := input[base:][:n]
	p0, p1 := cdefTapRow(input, base, t.pri0, n), cdefTapRow(input, base, -t.pri0, n)
	p2, p3 := cdefTapRow(input, base, t.pri1, n), cdefTapRow(input, base, -t.pri1, n)
	tap0, tap1, str, sh := t.priTap0, t.priTap1, t.priStr, t.priShift
	for i := range d {
		x := int(c[i])
		sum := tap0*(constrainU8(int(p0[i])-x, str, sh)+constrainU8(int(p1[i])-x, str, sh)) +
			tap1*(constrainU8(int(p2[i])-x, str, sh)+constrainU8(int(p3[i])-x, str, sh))
		d[i] = byte(cdefRound(x, sum))
	}
}

func filterRowU8Sec(d []byte, input []uint16, base int, t *cdefU8Taps) {
	n := len(d)
	c := input[base:][:n]
	s0, s1 := cdefTapRow(input, base, t.sec0, n), cdefTapRow(input, base, -t.sec0, n)
	s2, s3 := cdefTapRow(input, base, t.sec2, n), cdefTapRow(input, base, -t.sec2, n)
	s4, s5 := cdefTapRow(input, base, t.sec1, n), cdefTapRow(input, base, -t.sec1, n)
	s6, s7 := cdefTapRow(input, base, t.sec3, n), cdefTapRow(input, base, -t.sec3, n)
	str, sh := t.secStr, t.secShift
	for i := range d {
		x := int(c[i])
		sum := 2*(constrainU8(int(s0[i])-x, str, sh)+constrainU8(int(s1[i])-x, str, sh)+
			constrainU8(int(s2[i])-x, str, sh)+constrainU8(int(s3[i])-x, str, sh)) +
			constrainU8(int(s4[i])-x, str, sh) + constrainU8(int(s5[i])-x, str, sh) +
			constrainU8(int(s6[i])-x, str, sh) + constrainU8(int(s7[i])-x, str, sh)
		d[i] = byte(cdefRound(x, sum))
	}
}

// filterRowU8PriSec also clamps the result to the range of the real taps.
// The VeryLarge (0x4000) halo sentinel is the only tap value above 255, so
// masking it with 0x3fff turns it into 0 for the max (x itself already
// bounds the max from below) and it never wins the min.
func filterRowU8PriSec(d []byte, input []uint16, base int, t *cdefU8Taps) {
	n := len(d)
	c := input[base:][:n]
	p0, p1 := cdefTapRow(input, base, t.pri0, n), cdefTapRow(input, base, -t.pri0, n)
	p2, p3 := cdefTapRow(input, base, t.pri1, n), cdefTapRow(input, base, -t.pri1, n)
	s0, s1 := cdefTapRow(input, base, t.sec0, n), cdefTapRow(input, base, -t.sec0, n)
	s2, s3 := cdefTapRow(input, base, t.sec2, n), cdefTapRow(input, base, -t.sec2, n)
	s4, s5 := cdefTapRow(input, base, t.sec1, n), cdefTapRow(input, base, -t.sec1, n)
	s6, s7 := cdefTapRow(input, base, t.sec3, n), cdefTapRow(input, base, -t.sec3, n)
	tap0, tap1, pstr, psh := t.priTap0, t.priTap1, t.priStr, t.priShift
	sstr, ssh := t.secStr, t.secShift
	const mask = VeryLarge - 1
	for i := range d {
		x := int(c[i])
		a0, a1, a2, a3 := int(p0[i]), int(p1[i]), int(p2[i]), int(p3[i])
		b0, b1, b2, b3 := int(s0[i]), int(s1[i]), int(s2[i]), int(s3[i])
		b4, b5, b6, b7 := int(s4[i]), int(s5[i]), int(s6[i]), int(s7[i])
		sum := tap0*(constrainU8(a0-x, pstr, psh)+constrainU8(a1-x, pstr, psh)) +
			tap1*(constrainU8(a2-x, pstr, psh)+constrainU8(a3-x, pstr, psh)) +
			2*(constrainU8(b0-x, sstr, ssh)+constrainU8(b1-x, sstr, ssh)+
				constrainU8(b2-x, sstr, ssh)+constrainU8(b3-x, sstr, ssh)) +
			constrainU8(b4-x, sstr, ssh) + constrainU8(b5-x, sstr, ssh) +
			constrainU8(b6-x, sstr, ssh) + constrainU8(b7-x, sstr, ssh)
		hi := max(x, a0&mask, a1&mask, a2&mask, a3&mask, b0&mask, b1&mask, b2&mask, b3&mask,
			b4&mask, b5&mask, b6&mask, b7&mask)
		lo := min(x, a0, a1, a2, a3, b0, b1, b2, b3, b4, b5, b6, b7)
		d[i] = byte(min(max(cdefRound(x, sum), lo), hi))
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
	return filterFrameBlocksU8Trusted(dst, dstStride, input, inputOrigin, blocks, directions, variances, params, true)
}

// FilterFrameBlocksU8TrustedNoPrimaryDirection filters a decoder-owned unit
// when the caller has proved that this plane's primary strength is zero and
// no shared luma direction is needed. Secondary-only filtering uses direction
// 0, and the direction and variance grids remain untouched. The regular
// trusted entry point retains its direction-grid side effects.
func FilterFrameBlocksU8TrustedNoPrimaryDirection(dst []byte, dstStride int, input []uint16, inputOrigin int, blocks []BlockPosition, directions *DirectionGrid, variances *VarianceGrid, params FrameFilterParams) error {
	return filterFrameBlocksU8Trusted(dst, dstStride, input, inputOrigin, blocks, directions, variances, params, false)
}

func filterFrameBlocksU8Trusted(dst []byte, dstStride int, input []uint16, inputOrigin int, blocks []BlockPosition, directions *DirectionGrid, variances *VarianceGrid, params FrameFilterParams, deriveDirections bool) error {
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

	if params.Plane == PlaneY && deriveDirections {
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
	if deriveDirections && params.Plane == PlaneU && params.XDec != params.YDec {
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
		lumaAdjust:        params.Plane == PlaneY && (deriveDirections || primaryStrength != 0),
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
