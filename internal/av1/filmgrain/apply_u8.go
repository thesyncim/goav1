// Ported from libaom: av1/decoder/grain_synthesis.c
// SPDX-License-Identifier: BSD-2-Clause
// See LICENSE and NOTICE.

package filmgrain

// ApplyLumaRowU8 places generated AV1 grain directly into an 8-bit display row.
// It uses the same RNG, overlap and rounding as ApplyLumaRow.
// src and dst are byte sample views of the row band and may alias.
func ApplyLumaRowU8(dst []uint8, src []uint8, grain []int16, scaling []uint8, params LumaRowParams) error {
	if err := validateLumaRowU8(dst, src, grain, scaling, params); err != nil {
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
			yStart = min(params.Height, LumaOverlapSamples)
		}
		xStart := 0
		if params.Overlap && blockX > 0 {
			xStart = min(blockWidth, LumaOverlapSamples)
		}

		applyLumaBlockU8(dst, src, grain, scaling, params, offsets, blockX, blockWidth, xStart, yStart)
	}
	return nil
}

// ApplyChromaRowU8 places generated AV1 grain directly into an 8-bit chroma row.
// luma must be the original, ungrained source plane.
// src and dst are byte sample views of the chroma band and may alias.
func ApplyChromaRowU8(dst []uint8, src []uint8, luma []uint8, grain []int16, scaling []uint8, params ChromaRowParams) error {
	if err := validateChromaRowU8(dst, src, luma, grain, scaling, params); err != nil {
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

	shiftX := chromaSubsamplingShift(params.SubsamplingX)
	shiftY := chromaSubsamplingShift(params.SubsamplingY)
	chromaBlockWidth := LumaBlockSize >> shiftX
	var offsets [2][2]uint8
	for blockX := 0; blockX < params.Width; blockX += chromaBlockWidth {
		blockWidth := chromaBlockWidth
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
			yStart = min(params.Height, LumaOverlapSamples>>shiftY)
		}
		xStart := 0
		if params.Overlap && blockX > 0 {
			xStart = min(blockWidth, LumaOverlapSamples>>shiftX)
		}

		applyChromaBlockU8(dst, src, luma, grain, scaling, params, shiftX, shiftY, offsets, blockX, blockWidth, xStart, yStart)
	}
	return nil
}

func applyLumaBlockU8(dst []uint8, src []uint8, grain []int16, scaling []uint8, params LumaRowParams, offsets [2][2]uint8, blockX int, blockWidth int, xStart int, yStart int) {
	minValue, maxValue := lumaClipBounds(params)
	var scaleBuf [LumaBlockSize]uint16
	var blendBuf [LumaBlockSize]int16
	bitDepth := params.BitDepth
	grainMin := -(1 << (bitDepth - 1))
	grainMax := (1 << (bitDepth - 1)) - 1
	for y := yStart; y < params.Height; y++ {
		// Non-overlap interior: dst, src and the grain run are all
		// contiguous in x, so gather the scaling-LUT values once and let the
		// dispatched apply kernel do the scaled-grain add+clip.
		if n := blockWidth - xStart; n > 0 {
			base := y*params.Stride + blockX + xStart
			dstRow := dst[base : base+n]
			srcRow := src[base : base+n]
			gbase := lumaGrainSampleIndex(offsets[0][0], 0, 0, xStart, y)
			grainRow := grain[gbase : gbase+n]
			scale := scaleBuf[:n]
			gatherLumaScaleRowU8(scale, srcRow, scaling)
			applyGrainSegmentU8(dstRow, srcRow, scale, grainRow, int(params.ScalingShift), minValue, maxValue)
		}
		for x := range xStart {
			current := lumaGrainSample(grain, offsets[0][0], 0, 0, x, y)
			left := lumaGrainSample(grain, offsets[1][0], 1, 0, x, y)
			g := blendLumaOverlap(left, current, x, params.BitDepth)
			applyLumaRowSampleU8(dst, src, scaling, params, blockX+x, y, g)
		}
	}

	for y := range yStart {
		// Top overlap: the vertical blend weight is fixed by the overlap row y,
		// so the whole [xStart,blockWidth) run blends two contiguous grain rows
		// (top, current) with one weight pair, then takes the interior
		// gather+apply path. This fuses the blend into the scaling gather+add as
		// dav1d's fgy_32x32xn does (filmgrain.S). The left seam / corner columns
		// (<=2 wide) stay scalar below.
		if n := blockWidth - xStart; n > 0 {
			base := y*params.Stride + blockX + xStart
			dstRow := dst[base : base+n]
			srcRow := src[base : base+n]
			curBase := lumaGrainSampleIndex(offsets[0][0], 0, 0, xStart, y)
			topBase := lumaGrainSampleIndex(offsets[0][1], 0, 1, xStart, y)
			// blendLumaOverlap(top, current, y): previous=top gets 27, current
			// 17, swapped when the overlap offset is 1.
			prevWeight, curWeight := 27, 17
			if y == 1 {
				prevWeight, curWeight = 17, 27
			}
			blended := blendBuf[:n]
			blendGrainRow(blended, grain[topBase:topBase+n], grain[curBase:curBase+n], prevWeight, curWeight, grainMin, grainMax)
			scale := scaleBuf[:n]
			gatherLumaScaleRowU8(scale, srcRow, scaling)
			applyGrainSegmentU8(dstRow, srcRow, scale, blended, int(params.ScalingShift), minValue, maxValue)
		}
		for x := range xStart {
			top := lumaGrainSample(grain, offsets[0][1], 0, 1, x, y)
			topLeft := lumaGrainSample(grain, offsets[1][1], 1, 1, x, y)
			top = blendLumaOverlap(topLeft, top, x, params.BitDepth)

			current := lumaGrainSample(grain, offsets[0][0], 0, 0, x, y)
			left := lumaGrainSample(grain, offsets[1][0], 1, 0, x, y)
			current = blendLumaOverlap(left, current, x, params.BitDepth)

			g := blendLumaOverlap(top, current, y, params.BitDepth)
			applyLumaRowSampleU8(dst, src, scaling, params, blockX+x, y, g)
		}
	}
}

func applyChromaBlockU8(dst []uint8, src []uint8, luma []uint8, grain []int16, scaling []uint8, params ChromaRowParams, shiftX int, shiftY int, offsets [2][2]uint8, blockX int, blockWidth int, xStart int, yStart int) {
	minValue, maxValue := chromaClipBounds(params)
	var scaleBuf [LumaBlockSize]uint16
	var blendBuf [LumaBlockSize]int16
	bitDepth := params.BitDepth
	grainMin := -(1 << (bitDepth - 1))
	grainMax := (1 << (bitDepth - 1)) - 1
	for y := yStart; y < params.Height; y++ {
		// Non-overlap interior: dst, src and the grain run are contiguous in
		// x. The scaling index still depends on the (possibly subsampled)
		// luma plane, so gather the scale values scalar, then run the
		// dispatched add+clip kernel.
		if n := blockWidth - xStart; n > 0 {
			base := y*params.Stride + blockX + xStart
			dstRow := dst[base : base+n]
			srcRow := src[base : base+n]
			gbase := chromaGrainSampleIndex(offsets[0][0], shiftX, shiftY, 0, 0, xStart, y)
			grainRow := grain[gbase : gbase+n]
			scale := scaleBuf[:n]
			gatherChromaScaleRowU8(scale, srcRow, luma, scaling, params, shiftX, shiftY, blockX+xStart, y)
			applyGrainSegmentU8(dstRow, srcRow, scale, grainRow, int(params.ScalingShift), minValue, maxValue)
		}
		for x := range xStart {
			current := chromaGrainSample(grain, offsets[0][0], shiftX, shiftY, 0, 0, x, y)
			left := chromaGrainSample(grain, offsets[1][0], shiftX, shiftY, 1, 0, x, y)
			g := blendChromaOverlap(left, current, x, shiftX, params.BitDepth)
			applyChromaRowSampleU8(dst, src, luma, scaling, params, shiftX, blockX+x, y, g)
		}
	}

	for y := range yStart {
		// Top overlap: the vertical blend weight is fixed by the overlap row y
		// and the vertical subsampling, so the [xStart,blockWidth) run blends two
		// contiguous grain rows (top, current) with one weight pair and then
		// takes the interior scaling gather+add path — the same fusion dav1d's
		// fguv applies (filmgrain16.S). The left seam / corner (<=1-2 wide) stay
		// scalar below.
		if n := blockWidth - xStart; n > 0 {
			base := y*params.Stride + blockX + xStart
			dstRow := dst[base : base+n]
			srcRow := src[base : base+n]
			curBase := chromaGrainSampleIndex(offsets[0][0], shiftX, shiftY, 0, 0, xStart, y)
			topBase := chromaGrainSampleIndex(offsets[0][1], shiftX, shiftY, 0, 1, xStart, y)
			prevWeight, curWeight := chromaOverlapWeights(y, shiftY)
			blended := blendBuf[:n]
			blendGrainRow(blended, grain[topBase:topBase+n], grain[curBase:curBase+n], prevWeight, curWeight, grainMin, grainMax)
			scale := scaleBuf[:n]
			gatherChromaScaleRowU8(scale, srcRow, luma, scaling, params, shiftX, shiftY, blockX+xStart, y)
			applyGrainSegmentU8(dstRow, srcRow, scale, blended, int(params.ScalingShift), minValue, maxValue)
		}
		for x := range xStart {
			top := chromaGrainSample(grain, offsets[0][1], shiftX, shiftY, 0, 1, x, y)
			topLeft := chromaGrainSample(grain, offsets[1][1], shiftX, shiftY, 1, 1, x, y)
			top = blendChromaOverlap(topLeft, top, x, shiftX, params.BitDepth)

			current := chromaGrainSample(grain, offsets[0][0], shiftX, shiftY, 0, 0, x, y)
			left := chromaGrainSample(grain, offsets[1][0], shiftX, shiftY, 1, 0, x, y)
			current = blendChromaOverlap(left, current, x, shiftX, params.BitDepth)

			g := blendChromaOverlap(top, current, y, shiftY, params.BitDepth)
			applyChromaRowSampleU8(dst, src, luma, scaling, params, shiftX, blockX+x, y, g)
		}
	}
}

func gatherLumaScaleRowU8(scale []uint16, src []uint8, scaling []uint8) {
	for i := range scale {
		scale[i] = uint16(scaling[src[i]])
	}
}

func gatherChromaScaleRowU8(scale []uint16, src []uint8, luma []uint8, scaling []uint8, params ChromaRowParams, shiftX, shiftY, xBase, y int) {
	base := (y<<shiftY)*params.LumaStride + (xBase << shiftX)
	if params.ChromaScalingFromLuma {
		if shiftX != 0 {
			for i := range scale {
				j := base + (i << 1)
				scale[i] = uint16(scaling[(int(luma[j])+int(luma[j+1])+1)>>1])
			}
		} else {
			for i := range scale {
				scale[i] = uint16(scaling[luma[base+i]])
			}
		}
		return
	}
	for i := range scale {
		idx := chromaScalingIndexU8(src[i], luma, params, shiftX, xBase+i, y)
		scale[i] = uint16(scaling[idx])
	}
}

func applyLumaRowSampleU8(dst, src []uint8, scaling []uint8, params LumaRowParams, x, y int, grain int16) {
	i := y*params.Stride + x
	dst[i] = uint8(applyLumaSample(uint16(src[i]), grain, scaling, 8, params.ScalingShift, params.ClipToRestrictedRange))
}

func applyChromaRowSampleU8(dst, src, luma []uint8, scaling []uint8, params ChromaRowParams, shiftX, x, y int, grain int16) {
	i := y*params.Stride + x
	val := chromaScalingIndexU8(src[i], luma, params, shiftX, x, y)
	dst[i] = uint8(applyChromaSample(uint16(src[i]), val, grain, scaling, params))
}

func chromaScalingIndexU8(src uint8, luma []uint8, params ChromaRowParams, shiftX, x, y int) int {
	lumaIndex := (y<<chromaSubsamplingShift(params.SubsamplingY))*params.LumaStride + (x << shiftX)
	avg := int(luma[lumaIndex])
	if shiftX != 0 {
		avg = (avg + int(luma[lumaIndex+1]) + 1) >> 1
	}
	if params.ChromaScalingFromLuma {
		return avg
	}
	combined := avg*(int(params.ChromaLumaMult)-128) + int(src)*(int(params.ChromaMult)-128)
	return clipInt((combined>>6)+int(params.ChromaOffset)-256, 0, 255)
}

func validateLumaRowU8(dst []uint8, src []uint8, grain []int16, scaling []uint8, params LumaRowParams) error {
	if len(grain) < LumaGrainSamples ||
		len(scaling) < ScalingLUTSize ||
		params.Width <= 0 ||
		params.Height <= 0 ||
		params.Height > LumaBlockSize ||
		params.Stride < params.Width ||
		params.Row < 0 ||
		params.BitDepth != 8 ||
		params.ScalingShift < 8 ||
		params.ScalingShift > 11 {
		return ErrInvalidParams
	}
	// Use overflow-checked arithmetic so a hostile Stride value cannot pass
	// the slice-length compare via a wrapped product.
	rowStride, ok := checkedMulNonNegative(params.Height-1, params.Stride)
	if !ok {
		return ErrInvalidParams
	}
	need, ok := checkedAdd(rowStride, params.Width)
	if !ok {
		return ErrInvalidParams
	}
	if len(dst) < need || len(src) < need {
		return ErrInvalidParams
	}
	return nil
}

func validateChromaRowU8(dst []uint8, src []uint8, luma []uint8, grain []int16, scaling []uint8, params ChromaRowParams) error {
	shiftX := chromaSubsamplingShift(params.SubsamplingX)
	shiftY := chromaSubsamplingShift(params.SubsamplingY)
	blockHeight := LumaBlockSize >> shiftY
	if len(grain) < ChromaGrainSamples ||
		len(scaling) < ScalingLUTSize ||
		params.Width <= 0 ||
		params.Height <= 0 ||
		params.Height > blockHeight ||
		params.Stride < params.Width ||
		params.LumaStride < ((params.Width-1)<<shiftX)+1+shiftX ||
		params.Row < 0 ||
		params.BitDepth != 8 ||
		params.ScalingShift < 8 ||
		params.ScalingShift > 11 {
		return ErrInvalidParams
	}
	rowStride, ok := checkedMulNonNegative(params.Height-1, params.Stride)
	if !ok {
		return ErrInvalidParams
	}
	need, ok := checkedAdd(rowStride, params.Width)
	if !ok {
		return ErrInvalidParams
	}
	lumaRows := ((params.Height - 1) << shiftY) + 1
	lumaRowStride, ok := checkedMulNonNegative(lumaRows-1, params.LumaStride)
	if !ok {
		return ErrInvalidParams
	}
	lumaTail, ok := checkedAdd(((params.Width-1)<<shiftX)+1, shiftX)
	if !ok {
		return ErrInvalidParams
	}
	lumaNeed, ok := checkedAdd(lumaRowStride, lumaTail)
	if !ok {
		return ErrInvalidParams
	}
	if len(dst) < need || len(src) < need || len(luma) < lumaNeed {
		return ErrInvalidParams
	}
	return nil
}
