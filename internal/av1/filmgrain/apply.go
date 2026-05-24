package filmgrain

// LumaRowParams contains the per-stripe inputs for placing luma grain onto a
// decoded luma row band.
type LumaRowParams struct {
	Seed uint16

	Width  int
	Height int
	Stride int
	Row    int

	BitDepth              uint8
	ScalingShift          uint8
	Overlap               bool
	ClipToRestrictedRange bool
}

// ApplyLumaRow places generated AV1 luma grain on one 32-line luma row band.
// src and dst are uint16 sample views of the row band and may alias.
func ApplyLumaRow(dst []uint16, src []uint16, grain []int16, scaling []uint8, params LumaRowParams) error {
	if err := validateLumaRow(dst, src, grain, scaling, params); err != nil {
		return err
	}

	rows := 1
	if params.Overlap && params.Row > 0 {
		rows = 2
	}
	var randoms [2]Random
	for i := 0; i < rows; i++ {
		rng, err := NewStripeRandom(params.Seed, (params.Row-i)*LumaBlockSize)
		if err != nil {
			return err
		}
		randoms[i] = rng
	}

	var offsets [2][2]uint8
	for blockX := 0; blockX < params.Width; blockX += LumaBlockSize {
		blockWidth := LumaBlockSize
		if remaining := params.Width - blockX; remaining < blockWidth {
			blockWidth = remaining
		}
		if params.Overlap && blockX > 0 {
			for i := 0; i < rows; i++ {
				offsets[1][i] = offsets[0][i]
			}
		}
		for i := 0; i < rows; i++ {
			offset, _ := randoms[i].Number(8)
			offsets[0][i] = uint8(offset)
		}

		yStart := 0
		if params.Overlap && params.Row > 0 {
			yStart = LumaOverlapSamples
			if params.Height < yStart {
				yStart = params.Height
			}
		}
		xStart := 0
		if params.Overlap && blockX > 0 {
			xStart = LumaOverlapSamples
			if blockWidth < xStart {
				xStart = blockWidth
			}
		}

		applyLumaBlock(dst, src, grain, scaling, params, offsets, blockX, blockWidth, xStart, yStart)
	}
	return nil
}

func applyLumaBlock(dst []uint16, src []uint16, grain []int16, scaling []uint8, params LumaRowParams, offsets [2][2]uint8, blockX int, blockWidth int, xStart int, yStart int) {
	for y := yStart; y < params.Height; y++ {
		for x := xStart; x < blockWidth; x++ {
			g := lumaGrainSample(grain, offsets[0][0], 0, 0, x, y)
			applyLumaRowSample(dst, src, scaling, params, blockX+x, y, g)
		}
		for x := 0; x < xStart; x++ {
			current := lumaGrainSample(grain, offsets[0][0], 0, 0, x, y)
			left := lumaGrainSample(grain, offsets[1][0], 1, 0, x, y)
			g := blendLumaOverlap(left, current, x, params.BitDepth)
			applyLumaRowSample(dst, src, scaling, params, blockX+x, y, g)
		}
	}

	for y := 0; y < yStart; y++ {
		for x := xStart; x < blockWidth; x++ {
			current := lumaGrainSample(grain, offsets[0][0], 0, 0, x, y)
			top := lumaGrainSample(grain, offsets[0][1], 0, 1, x, y)
			g := blendLumaOverlap(top, current, y, params.BitDepth)
			applyLumaRowSample(dst, src, scaling, params, blockX+x, y, g)
		}
		for x := 0; x < xStart; x++ {
			top := lumaGrainSample(grain, offsets[0][1], 0, 1, x, y)
			topLeft := lumaGrainSample(grain, offsets[1][1], 1, 1, x, y)
			top = blendLumaOverlap(topLeft, top, x, params.BitDepth)

			current := lumaGrainSample(grain, offsets[0][0], 0, 0, x, y)
			left := lumaGrainSample(grain, offsets[1][0], 1, 0, x, y)
			current = blendLumaOverlap(left, current, x, params.BitDepth)

			g := blendLumaOverlap(top, current, y, params.BitDepth)
			applyLumaRowSample(dst, src, scaling, params, blockX+x, y, g)
		}
	}
}

func applyLumaRowSample(dst []uint16, src []uint16, scaling []uint8, params LumaRowParams, x int, y int, grain int16) {
	i := y*params.Stride + x
	dst[i] = applyLumaSample(src[i], grain, scaling, params.BitDepth, params.ScalingShift, params.ClipToRestrictedRange)
}

func validateLumaRow(dst []uint16, src []uint16, grain []int16, scaling []uint8, params LumaRowParams) error {
	if len(grain) < LumaGrainSamples ||
		len(scaling) < ScalingLUTSize ||
		params.Width <= 0 ||
		params.Height <= 0 ||
		params.Height > LumaBlockSize ||
		params.Stride < params.Width ||
		params.Row < 0 ||
		(params.BitDepth != 8 && params.BitDepth != 10 && params.BitDepth != 12) ||
		params.ScalingShift < 8 ||
		params.ScalingShift > 11 {
		return ErrInvalidParams
	}
	need := (params.Height-1)*params.Stride + params.Width
	if len(dst) < need || len(src) < need {
		return ErrInvalidParams
	}
	return nil
}
