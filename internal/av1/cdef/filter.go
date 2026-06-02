// Ported from libaom:
//   av1/common/cdef_block.c
//   av1/common/cdef.c
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

package cdef

import "math/bits"

type Plane uint8

const (
	PlaneY Plane = iota
	PlaneU
	PlaneV
)

type BlockPosition struct {
	BY uint8
	BX uint8
}

type DirectionGrid [NBlocks][NBlocks]uint8
type VarianceGrid [NBlocks][NBlocks]int32

type BlockFilterParams struct {
	PrimaryStrength   uint8
	SecondaryStrength uint8
	Direction         uint8
	PrimaryDamping    uint8
	SecondaryDamping  uint8
	CoeffShift        uint8
	Width             uint8
	Height            uint8
}

type FrameFilterParams struct {
	XDec              int
	YDec              int
	Plane             Plane
	Level             int
	SecondaryStrength int
	Damping           int
	CoeffShift        int
}

var cdefDirections = [12][2]int{
	{1 * BStride, 2 * BStride},
	{1 * BStride, 2*BStride - 1},
	{-1*BStride + 1, -2*BStride + 2},
	{1, -1*BStride + 2},
	{1, 2},
	{1, 1*BStride + 2},
	{1*BStride + 1, 2*BStride + 2},
	{1 * BStride, 2*BStride + 1},
	{1 * BStride, 2 * BStride},
	{1 * BStride, 2*BStride - 1},
	{-1*BStride + 1, -2*BStride + 2},
	{1, -1*BStride + 2},
}

var cdefPrimaryTaps = [2][2]int{{4, 2}, {3, 3}}
var cdefSecondaryTaps = [2]int{2, 1}

// FilterBlock ports libaom's cdef_filter_block_internal for 16-bit output.
// inputOrigin and dstOrigin mirror the C pointer arguments; input is normally a
// CDEF buffer whose current block starts at VerticalBorder*BStride+HorizontalBorder.
func FilterBlock(dst []uint16, dstStride int, dstOrigin int, input []uint16, inputOrigin int, params BlockFilterParams) error {
	if err := validateBlockFilter(dst, dstStride, dstOrigin, input, inputOrigin, params); err != nil {
		return err
	}
	filterBlockImpl(dst, dstStride, dstOrigin, input, inputOrigin, params)
	return nil
}

// filterBlockPureGo is the canonical bit-exact CDEF block filter. It assumes
// its arguments were already validated by FilterBlock; the dispatch slot in
// filter_dispatch.go defaults to this function so every build has a working
// reference path.
func filterBlockPureGo(dst []uint16, dstStride int, dstOrigin int, input []uint16, inputOrigin int, params BlockFilterParams) {
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

	// Hoist the per-call invariants out of the per-pixel loop: direction
	// offsets, primary taps, and the constrain() shift amounts (which depend
	// only on threshold/damping). This keeps the inner loop free of repeated
	// table lookups and bits.Len calls.
	pri0 := cdefDirections[direction+2][0]
	pri1 := cdefDirections[direction+2][1]
	sec0a := cdefDirections[direction+4][0]
	sec0b := cdefDirections[direction+4][1]
	sec1a := cdefDirections[direction][0]
	sec1b := cdefDirections[direction][1]
	priTap0 := priTaps[0]
	priTap1 := priTaps[1]
	secTap0 := cdefSecondaryTaps[0]
	secTap1 := cdefSecondaryTaps[1]
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
			dst[dstRow+col] = uint16(y)
		}
	}
}

// constrainShift computes the per-call shift amount used by constrain. It is
// invariant across all pixels for a given strength/damping pair.
func constrainShift(threshold int, damping int) int {
	if threshold == 0 {
		return 0
	}
	return max(damping-(bits.Len(uint(threshold))-1), 0)
}

// constrainShifted is constrain with the shift precomputed by constrainShift.
func constrainShifted(diff int, threshold int, shift int) int {
	if threshold == 0 {
		return 0
	}
	a := absInt(diff)
	limit := threshold - (a >> shift)
	if limit < 0 {
		limit = 0
	} else if limit > a {
		limit = a
	}
	if diff < 0 {
		return -limit
	}
	return limit
}

// maxClip folds four CDEF tap samples into the running max, skipping the
// VeryLarge sentinel exactly as libaom does.
func maxClip(cur, a, b, c, d int) int {
	if a != VeryLarge && a > cur {
		cur = a
	}
	if b != VeryLarge && b > cur {
		cur = b
	}
	if c != VeryLarge && c > cur {
		cur = c
	}
	if d != VeryLarge && d > cur {
		cur = d
	}
	return cur
}

// min4 folds four CDEF tap samples into the running min.
func min4(cur, a, b, c, d int) int {
	if a < cur {
		cur = a
	}
	if b < cur {
		cur = b
	}
	if c < cur {
		cur = c
	}
	if d < cur {
		cur = d
	}
	return cur
}

// FilterFrameBlocks ports libaom's av1_cdef_filter_fb decode path for 16-bit
// frame buffers. Direction and variance grids are shared between luma and
// chroma, matching libaom's CDEF frame-block flow.
func FilterFrameBlocks(dst []uint16, dstStride int, input []uint16, inputOrigin int, blocks []BlockPosition, directions *DirectionGrid, variances *VarianceGrid, params FrameFilterParams) error {
	if err := validateFrameFilter(dst, dstStride, input, inputOrigin, directions, variances, params); err != nil {
		return err
	}
	if len(blocks) == 0 {
		return nil
	}
	primaryStrength, ok := checkedShift(params.Level, params.CoeffShift)
	if !ok {
		return ErrInvalidCDEF
	}
	secondaryStrength, ok := checkedShift(params.SecondaryStrength, params.CoeffShift)
	if !ok {
		return ErrInvalidCDEF
	}
	damping := params.Damping + params.CoeffShift
	if params.Plane != PlaneY {
		damping--
	}
	bwLog2 := 3 - params.XDec
	bhLog2 := 3 - params.YDec
	blockWidth := 1 << bwLog2
	blockHeight := 1 << bhLog2

	for _, block := range blocks {
		by := int(block.BY)
		bx := int(block.BX)
		if by >= NBlocks || bx >= NBlocks {
			return ErrInvalidCDEF
		}
	}
	if params.Plane == PlaneY {
		for i := 0; i < len(blocks); {
			block := blocks[i]
			by := int(block.BY)
			bx := int(block.BX)
			srcOrigin := inputOrigin + ((by * BStride) << bhLog2) + (bx << bwLog2)
			if srcOrigin < 0 || srcOrigin >= len(input) {
				return ErrInvalidCDEF
			}
			nextOrigin := srcOrigin + blockWidth
			if blockWidth == 8 && nextOrigin <= len(input) &&
				i+1 < len(blocks) && blocks[i+1].BY == block.BY && blocks[i+1].BX == block.BX+1 {
				dir1, variance1, dir2, variance2, err := FindDirectionDual(input[srcOrigin:], input[nextOrigin:], BStride, params.CoeffShift)
				if err != nil {
					return err
				}
				directions[by][bx] = uint8(dir1)
				variances[by][bx] = variance1
				directions[by][bx+1] = uint8(dir2)
				variances[by][bx+1] = variance2
				i += 2
				continue
			}
			dir, variance, err := FindDirection(input[srcOrigin:], BStride, params.CoeffShift)
			if err != nil {
				return err
			}
			directions[by][bx] = uint8(dir)
			variances[by][bx] = variance
			i++
		}
	}
	if params.Plane == PlaneU && params.XDec != params.YDec {
		if !convertChromaDirections(blocks, directions, params.XDec) {
			return ErrInvalidCDEF
		}
	}

	for _, block := range blocks {
		by := int(block.BY)
		bx := int(block.BX)
		strength := primaryStrength
		if params.Plane == PlaneY {
			strength = adjustStrength(primaryStrength, variances[by][bx])
		}
		dir := 0
		if primaryStrength != 0 {
			dir = int(directions[by][bx])
		}
		srcOrigin := inputOrigin + ((by * BStride) << bhLog2) + (bx << bwLog2)
		dstOrigin := (by<<bhLog2)*dstStride + (bx << bwLog2)
		if strength > int(^uint8(0)) ||
			secondaryStrength > int(^uint8(0)) ||
			damping > int(^uint8(0)) ||
			params.CoeffShift > int(^uint8(0)) ||
			blockWidth > int(^uint8(0)) ||
			blockHeight > int(^uint8(0)) {
			return ErrInvalidCDEF
		}
		err := FilterBlock(dst, dstStride, dstOrigin, input, srcOrigin, BlockFilterParams{
			PrimaryStrength:   uint8(strength),
			SecondaryStrength: uint8(secondaryStrength),
			Direction:         uint8(dir),
			PrimaryDamping:    uint8(damping),
			SecondaryDamping:  uint8(damping),
			CoeffShift:        uint8(params.CoeffShift),
			Width:             uint8(blockWidth),
			Height:            uint8(blockHeight),
		})
		if err != nil {
			return err
		}
	}
	return nil
}

func validateBlockFilter(dst []uint16, dstStride int, dstOrigin int, input []uint16, inputOrigin int, params BlockFilterParams) error {
	width := int(params.Width)
	height := int(params.Height)
	if params.CoeffShift > 4 ||
		params.Direction > 7 ||
		!validBlockDimension(width) || !validBlockDimension(height) ||
		!blockFitsAt(len(dst), dstOrigin, dstStride, width, height) {
		return ErrInvalidCDEF
	}
	if !cdefInputFits(len(input), inputOrigin, params) {
		return ErrInvalidCDEF
	}
	return nil
}

func validateFrameFilter(dst []uint16, dstStride int, input []uint16, inputOrigin int, directions *DirectionGrid, variances *VarianceGrid, params FrameFilterParams) error {
	if directions == nil || variances == nil ||
		params.XDec < 0 || params.XDec > 1 ||
		params.YDec < 0 || params.YDec > 1 ||
		params.Plane > PlaneV ||
		params.Level < 0 || params.SecondaryStrength < 0 ||
		params.Damping < 0 ||
		params.CoeffShift < 0 || params.CoeffShift > 4 ||
		dstStride <= 0 || inputOrigin < 0 || inputOrigin >= len(input) {
		return ErrInvalidCDEF
	}
	return nil
}

func validBlockDimension(v int) bool {
	return v == 4 || v == 8
}

func cdefInputFits(length int, origin int, params BlockFilterParams) bool {
	if origin < 0 || origin >= length {
		return false
	}
	enablePrimary := params.PrimaryStrength != 0
	enableSecondary := params.SecondaryStrength != 0
	width := int(params.Width)
	height := int(params.Height)
	direction := int(params.Direction)

	// The block touches input[base+off] and input[base-off] for a fixed set of
	// direction offsets, where base ranges linearly over
	// [origin, origin+(Height-1)*BStride+(Width-1)]. Because every access uses
	// both +off and -off, the lowest accessed index is loBase-maxOff and the
	// highest is hiBase+maxOff. Checking just those two extremes is exact and
	// equivalent to walking every pixel, but O(1) instead of O(Width*Height).
	loBase := origin
	hiBase := origin + (height-1)*BStride + (width - 1)

	maxOff := 0
	if enablePrimary {
		maxOff = maxAbs(maxOff, cdefDirections[direction+2][0])
		maxOff = maxAbs(maxOff, cdefDirections[direction+2][1])
	}
	if enableSecondary {
		maxOff = maxAbs(maxOff, cdefDirections[direction+4][0])
		maxOff = maxAbs(maxOff, cdefDirections[direction+4][1])
		maxOff = maxAbs(maxOff, cdefDirections[direction][0])
		maxOff = maxAbs(maxOff, cdefDirections[direction][1])
	}

	return indexFits(length, loBase-maxOff) && indexFits(length, hiBase+maxOff)
}

func maxAbs(cur int, off int) int {
	if off < 0 {
		off = -off
	}
	if off > cur {
		return off
	}
	return cur
}

func indexFits(length int, index int) bool {
	return index >= 0 && index < length
}

func constrain(diff int, threshold int, damping int) int {
	if threshold == 0 {
		return 0
	}
	shift := max(damping-(bits.Len(uint(threshold))-1), 0)
	limit := threshold - (absInt(diff) >> shift)
	return signInt(diff) * clampInt(limit, 0, absInt(diff))
}

func adjustStrength(strength int, variance int32) int {
	if variance == 0 {
		return 0
	}
	i := 0
	if v := variance >> 6; v != 0 {
		i = min(bits.Len(uint(v))-1, 12)
	}
	return (strength*(4+i) + 8) >> 4
}

func convertChromaDirections(blocks []BlockPosition, directions *DirectionGrid, xdec int) bool {
	conv422 := [8]uint8{7, 0, 2, 4, 5, 6, 6, 6}
	conv440 := [8]uint8{1, 2, 2, 2, 3, 4, 6, 0}
	for _, block := range blocks {
		by := int(block.BY)
		bx := int(block.BX)
		if by >= NBlocks || bx >= NBlocks {
			return false
		}
		if directions[by][bx] > 7 {
			return false
		}
		if xdec != 0 {
			directions[by][bx] = conv422[directions[by][bx]]
		} else {
			directions[by][bx] = conv440[directions[by][bx]]
		}
	}
	return true
}

func checkedShift(value int, shift int) (int, bool) {
	if shift < 0 || value < 0 || value > int(^uint(0)>>1)>>shift {
		return 0, false
	}
	return value << shift, true
}

func blockFitsAt(length int, origin int, stride int, width int, height int) bool {
	if origin < 0 || stride <= 0 || width <= 0 || height <= 0 || stride < width {
		return false
	}
	lastRow, ok := checkedMul(height-1, stride)
	if !ok {
		return false
	}
	lastOffset, ok := checkedAdd(origin, lastRow)
	if !ok {
		return false
	}
	needed, ok := checkedAdd(lastOffset, width)
	return ok && needed <= length
}

func clampInt(v int, lo int, hi int) int {
	return min(max(v, lo), hi)
}

func signInt(v int) int {
	if v < 0 {
		return -1
	}
	return 1
}

func absInt(v int) int {
	if v < 0 {
		return -v
	}
	return v
}

func boolToInt(v bool) int {
	if v {
		return 1
	}
	return 0
}
