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

type DirectionGrid [NBlocks][NBlocks]int
type VarianceGrid [NBlocks][NBlocks]int32

type BlockFilterParams struct {
	PrimaryStrength   int
	SecondaryStrength int
	Direction         int
	PrimaryDamping    int
	SecondaryDamping  int
	CoeffShift        int
	Width             int
	Height            int
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
	enablePrimary := params.PrimaryStrength != 0
	enableSecondary := params.SecondaryStrength != 0
	clippingRequired := enablePrimary && enableSecondary
	priTaps := cdefPrimaryTaps[(params.PrimaryStrength>>params.CoeffShift)&1]

	for row := 0; row < params.Height; row++ {
		for col := 0; col < params.Width; col++ {
			base := inputOrigin + row*BStride + col
			sum := 0
			x := int(input[base])
			maxSample := x
			minSample := x
			for k := range 2 {
				if enablePrimary {
					p0 := int(input[base+cdefDirections[params.Direction+2][k]])
					p1 := int(input[base-cdefDirections[params.Direction+2][k]])
					sum += priTaps[k] * constrain(p0-x, params.PrimaryStrength, params.PrimaryDamping)
					sum += priTaps[k] * constrain(p1-x, params.PrimaryStrength, params.PrimaryDamping)
					if clippingRequired {
						if p0 != VeryLarge && p0 > maxSample {
							maxSample = p0
						}
						if p1 != VeryLarge && p1 > maxSample {
							maxSample = p1
						}
						if p0 < minSample {
							minSample = p0
						}
						if p1 < minSample {
							minSample = p1
						}
					}
				}
				if enableSecondary {
					s0 := int(input[base+cdefDirections[params.Direction+4][k]])
					s1 := int(input[base-cdefDirections[params.Direction+4][k]])
					s2 := int(input[base+cdefDirections[params.Direction][k]])
					s3 := int(input[base-cdefDirections[params.Direction][k]])
					if clippingRequired {
						if s0 != VeryLarge && s0 > maxSample {
							maxSample = s0
						}
						if s1 != VeryLarge && s1 > maxSample {
							maxSample = s1
						}
						if s2 != VeryLarge && s2 > maxSample {
							maxSample = s2
						}
						if s3 != VeryLarge && s3 > maxSample {
							maxSample = s3
						}
						if s0 < minSample {
							minSample = s0
						}
						if s1 < minSample {
							minSample = s1
						}
						if s2 < minSample {
							minSample = s2
						}
						if s3 < minSample {
							minSample = s3
						}
					}
					sum += cdefSecondaryTaps[k] * constrain(s0-x, params.SecondaryStrength, params.SecondaryDamping)
					sum += cdefSecondaryTaps[k] * constrain(s1-x, params.SecondaryStrength, params.SecondaryDamping)
					sum += cdefSecondaryTaps[k] * constrain(s2-x, params.SecondaryStrength, params.SecondaryDamping)
					sum += cdefSecondaryTaps[k] * constrain(s3-x, params.SecondaryStrength, params.SecondaryDamping)
				}
			}
			y := x + ((8 + sum - boolToInt(sum < 0)) >> 4)
			if clippingRequired {
				y = clampInt(y, minSample, maxSample)
			}
			dst[dstOrigin+row*dstStride+col] = uint16(y)
		}
	}
	return nil
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
		for _, block := range blocks {
			by := int(block.BY)
			bx := int(block.BX)
			srcOrigin := inputOrigin + ((by * BStride) << bhLog2) + (bx << bwLog2)
			if srcOrigin < 0 || srcOrigin >= len(input) {
				return ErrInvalidCDEF
			}
			dir, variance, err := FindDirection(input[srcOrigin:], BStride, params.CoeffShift)
			if err != nil {
				return err
			}
			directions[by][bx] = dir
			variances[by][bx] = variance
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
			dir = directions[by][bx]
		}
		srcOrigin := inputOrigin + ((by * BStride) << bhLog2) + (bx << bwLog2)
		dstOrigin := (by<<bhLog2)*dstStride + (bx << bwLog2)
		err := FilterBlock(dst, dstStride, dstOrigin, input, srcOrigin, BlockFilterParams{
			PrimaryStrength:   strength,
			SecondaryStrength: secondaryStrength,
			Direction:         dir,
			PrimaryDamping:    damping,
			SecondaryDamping:  damping,
			CoeffShift:        params.CoeffShift,
			Width:             blockWidth,
			Height:            blockHeight,
		})
		if err != nil {
			return err
		}
	}
	return nil
}

func validateBlockFilter(dst []uint16, dstStride int, dstOrigin int, input []uint16, inputOrigin int, params BlockFilterParams) error {
	if params.PrimaryStrength < 0 || params.SecondaryStrength < 0 ||
		params.PrimaryDamping < 0 || params.SecondaryDamping < 0 ||
		params.CoeffShift < 0 || params.CoeffShift > 4 ||
		params.Direction < 0 || params.Direction > 7 ||
		!validBlockDimension(params.Width) || !validBlockDimension(params.Height) ||
		!blockFitsAt(len(dst), dstOrigin, dstStride, params.Width, params.Height) {
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
	for row := 0; row < params.Height; row++ {
		for col := 0; col < params.Width; col++ {
			base := origin + row*BStride + col
			if base < 0 || base >= length {
				return false
			}
			for k := range 2 {
				if enablePrimary && (!indexFits(length, base+cdefDirections[params.Direction+2][k]) || !indexFits(length, base-cdefDirections[params.Direction+2][k])) {
					return false
				}
				if enableSecondary &&
					(!indexFits(length, base+cdefDirections[params.Direction+4][k]) ||
						!indexFits(length, base-cdefDirections[params.Direction+4][k]) ||
						!indexFits(length, base+cdefDirections[params.Direction][k]) ||
						!indexFits(length, base-cdefDirections[params.Direction][k])) {
					return false
				}
			}
		}
	}
	return true
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
	conv422 := [8]int{7, 0, 2, 4, 5, 6, 6, 6}
	conv440 := [8]int{1, 2, 2, 2, 3, 4, 6, 0}
	for _, block := range blocks {
		by := int(block.BY)
		bx := int(block.BX)
		if by >= NBlocks || bx >= NBlocks {
			return false
		}
		if directions[by][bx] < 0 || directions[by][bx] > 7 {
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
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
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
