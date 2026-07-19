// Ported from libaom: av1/decoder/grain_synthesis.c
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

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

// ChromaRowParams contains the per-stripe inputs for placing chroma grain onto
// a decoded chroma row band.
type ChromaRowParams struct {
	Seed uint16

	Width      int
	Height     int
	Stride     int
	LumaStride int
	Row        int

	BitDepth              uint8
	ScalingShift          uint8
	SubsamplingX          bool
	SubsamplingY          bool
	Overlap               bool
	ClipToRestrictedRange bool
	MatrixIdentity        bool
	ChromaScalingFromLuma bool

	ChromaMult     uint8
	ChromaLumaMult uint8
	ChromaOffset   uint16
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
			yStart = min(params.Height, LumaOverlapSamples)
		}
		xStart := 0
		if params.Overlap && blockX > 0 {
			xStart = min(blockWidth, LumaOverlapSamples)
		}

		applyLumaBlock(dst, src, grain, scaling, params, offsets, blockX, blockWidth, xStart, yStart)
	}
	return nil
}

// ApplyChromaRow places generated AV1 chroma grain on one chroma row band.
// src and dst are uint16 sample views of the chroma band and may alias.
func ApplyChromaRow(dst []uint16, src []uint16, luma []uint16, grain []int16, scaling []uint8, params ChromaRowParams) error {
	if err := validateChromaRow(dst, src, luma, grain, scaling, params); err != nil {
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

		applyChromaBlock(dst, src, luma, grain, scaling, params, shiftX, shiftY, offsets, blockX, blockWidth, xStart, yStart)
	}
	return nil
}

func applyLumaBlock(dst []uint16, src []uint16, grain []int16, scaling []uint8, params LumaRowParams, offsets [2][2]uint8, blockX int, blockWidth int, xStart int, yStart int) {
	minValue, maxValue := lumaClipBounds(params)
	var scaleBuf [LumaBlockSize]uint16
	bitDepth := params.BitDepth
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
			switch bitDepth {
			case 8:
				// 8-bit is a direct lut[index] gather with no interpolation;
				// the scalar load beats a NEON per-lane gather (measured), so
				// it stays scalar. The 10-bit interpolate is vectorized below.
				for x := 0; x < n; x++ {
					scale[x] = uint16(scaleLUT8(scaling, int(srcRow[x])))
				}
			case 10:
				// Vectorized scaling-LUT gather+interpolate
				// (build_scale_dispatch_*.go); bit-exact with the scaleLUT10
				// per-pixel loop.
				buildScaleRow10(scale, srcRow, scaling)
			case 12:
				for x := 0; x < n; x++ {
					scale[x] = uint16(scaleLUT12(scaling, int(srcRow[x])))
				}
			default:
				for x := 0; x < n; x++ {
					scale[x] = uint16(scaleLUT(scaling, int(srcRow[x]), bitDepth))
				}
			}
			applyGrainSegment(dstRow, srcRow, scale, grainRow, int(params.ScalingShift), minValue, maxValue)
		}
		for x := range xStart {
			current := lumaGrainSample(grain, offsets[0][0], 0, 0, x, y)
			left := lumaGrainSample(grain, offsets[1][0], 1, 0, x, y)
			g := blendLumaOverlap(left, current, x, params.BitDepth)
			applyLumaRowSample(dst, src, scaling, params, blockX+x, y, g)
		}
	}

	for y := range yStart {
		for x := xStart; x < blockWidth; x++ {
			current := lumaGrainSample(grain, offsets[0][0], 0, 0, x, y)
			top := lumaGrainSample(grain, offsets[0][1], 0, 1, x, y)
			g := blendLumaOverlap(top, current, y, params.BitDepth)
			applyLumaRowSample(dst, src, scaling, params, blockX+x, y, g)
		}
		for x := range xStart {
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

func applyChromaBlock(dst []uint16, src []uint16, luma []uint16, grain []int16, scaling []uint8, params ChromaRowParams, shiftX int, shiftY int, offsets [2][2]uint8, blockX int, blockWidth int, xStart int, yStart int) {
	minValue, maxValue := chromaClipBounds(params)
	var scaleBuf [LumaBlockSize]uint16
	// idxBuf collects the per-pixel chroma scaling indices (subsampled-luma
	// average, or the cross-plane chromaScalingIndex) so the 10-bit scaling-LUT
	// gather+interpolate can run through the vectorized buildScaleRow10 kernel.
	// Both stay on the stack (the apply path is zero-alloc).
	var idxBuf [LumaBlockSize]uint16
	bitDepth := params.BitDepth
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
			if params.ChromaScalingFromLuma {
				lumaY := y << shiftY
				lumaBase := lumaY*params.LumaStride + ((blockX + xStart) << shiftX)
				switch bitDepth {
				case 8:
					if shiftX != 0 {
						for x := 0; x < n; x++ {
							idx := lumaBase + (x << 1)
							avg := (int(luma[idx]) + int(luma[idx+1]) + 1) >> 1
							scale[x] = uint16(scaleLUT8(scaling, avg))
						}
					} else {
						for x := 0; x < n; x++ {
							scale[x] = uint16(scaleLUT8(scaling, int(luma[lumaBase+x])))
						}
					}
				case 10:
					if shiftX != 0 {
						idx := idxBuf[:n]
						for x := 0; x < n; x++ {
							li := lumaBase + (x << 1)
							idx[x] = uint16((int(luma[li]) + int(luma[li+1]) + 1) >> 1)
						}
						buildScaleRow10(scale, idx, scaling)
					} else {
						buildScaleRow10(scale, luma[lumaBase:lumaBase+n], scaling)
					}
				case 12:
					if shiftX != 0 {
						for x := 0; x < n; x++ {
							idx := lumaBase + (x << 1)
							avg := (int(luma[idx]) + int(luma[idx+1]) + 1) >> 1
							scale[x] = uint16(scaleLUT12(scaling, avg))
						}
					} else {
						for x := 0; x < n; x++ {
							scale[x] = uint16(scaleLUT12(scaling, int(luma[lumaBase+x])))
						}
					}
				default:
					if shiftX != 0 {
						for x := 0; x < n; x++ {
							idx := lumaBase + (x << 1)
							avg := (int(luma[idx]) + int(luma[idx+1]) + 1) >> 1
							scale[x] = uint16(scaleLUT(scaling, avg, bitDepth))
						}
					} else {
						for x := 0; x < n; x++ {
							scale[x] = uint16(scaleLUT(scaling, int(luma[lumaBase+x]), bitDepth))
						}
					}
				}
			} else {
				switch bitDepth {
				case 8:
					for x := 0; x < n; x++ {
						idx := chromaScalingIndex(srcRow[x], luma, params, shiftX, blockX+xStart+x, y)
						scale[x] = uint16(scaleLUT8(scaling, idx))
					}
				case 10:
					idx := idxBuf[:n]
					for x := 0; x < n; x++ {
						idx[x] = uint16(chromaScalingIndex(srcRow[x], luma, params, shiftX, blockX+xStart+x, y))
					}
					buildScaleRow10(scale, idx, scaling)
				case 12:
					for x := 0; x < n; x++ {
						idx := chromaScalingIndex(srcRow[x], luma, params, shiftX, blockX+xStart+x, y)
						scale[x] = uint16(scaleLUT12(scaling, idx))
					}
				default:
					for x := 0; x < n; x++ {
						idx := chromaScalingIndex(srcRow[x], luma, params, shiftX, blockX+xStart+x, y)
						scale[x] = uint16(scaleLUT(scaling, idx, bitDepth))
					}
				}
			}
			applyGrainSegment(dstRow, srcRow, scale, grainRow, int(params.ScalingShift), minValue, maxValue)
		}
		for x := range xStart {
			current := chromaGrainSample(grain, offsets[0][0], shiftX, shiftY, 0, 0, x, y)
			left := chromaGrainSample(grain, offsets[1][0], shiftX, shiftY, 1, 0, x, y)
			g := blendChromaOverlap(left, current, x, shiftX, params.BitDepth)
			applyChromaRowSample(dst, src, luma, scaling, params, shiftX, blockX+x, y, g)
		}
	}

	for y := range yStart {
		for x := xStart; x < blockWidth; x++ {
			current := chromaGrainSample(grain, offsets[0][0], shiftX, shiftY, 0, 0, x, y)
			top := chromaGrainSample(grain, offsets[0][1], shiftX, shiftY, 0, 1, x, y)
			g := blendChromaOverlap(top, current, y, shiftY, params.BitDepth)
			applyChromaRowSample(dst, src, luma, scaling, params, shiftX, blockX+x, y, g)
		}
		for x := range xStart {
			top := chromaGrainSample(grain, offsets[0][1], shiftX, shiftY, 0, 1, x, y)
			topLeft := chromaGrainSample(grain, offsets[1][1], shiftX, shiftY, 1, 1, x, y)
			top = blendChromaOverlap(topLeft, top, x, shiftX, params.BitDepth)

			current := chromaGrainSample(grain, offsets[0][0], shiftX, shiftY, 0, 0, x, y)
			left := chromaGrainSample(grain, offsets[1][0], shiftX, shiftY, 1, 0, x, y)
			current = blendChromaOverlap(left, current, x, shiftX, params.BitDepth)

			g := blendChromaOverlap(top, current, y, shiftY, params.BitDepth)
			applyChromaRowSample(dst, src, luma, scaling, params, shiftX, blockX+x, y, g)
		}
	}
}

func applyLumaRowSample(dst []uint16, src []uint16, scaling []uint8, params LumaRowParams, x int, y int, grain int16) {
	i := y*params.Stride + x
	dst[i] = applyLumaSample(src[i], grain, scaling, params.BitDepth, params.ScalingShift, params.ClipToRestrictedRange)
}

func applyChromaRowSample(dst []uint16, src []uint16, luma []uint16, scaling []uint8, params ChromaRowParams, shiftX int, x int, y int, grain int16) {
	i := y*params.Stride + x
	val := chromaScalingIndex(src[i], luma, params, shiftX, x, y)
	dst[i] = applyChromaSample(src[i], val, grain, scaling, params)
}

func chromaScalingIndex(src uint16, luma []uint16, params ChromaRowParams, shiftX int, x int, y int) int {
	lumaX := x << shiftX
	lumaY := y << chromaSubsamplingShift(params.SubsamplingY)
	lumaIndex := lumaY*params.LumaStride + lumaX
	avg := int(luma[lumaIndex])
	if shiftX != 0 {
		avg = (avg + int(luma[lumaIndex+1]) + 1) >> 1
	}
	if params.ChromaScalingFromLuma {
		return avg
	}
	// libaom add_noise_to_block: cb_mult/cb_luma_mult carry a fixed +128 bias
	// and cb_offset a +256 bias (scaled to bit depth as (offset<<(bd-8))-(1<<bd)).
	lumaMult := int(params.ChromaLumaMult) - 128
	chromaMult := int(params.ChromaMult) - 128
	offset := (int(params.ChromaOffset) << (params.BitDepth - 8)) - (1 << params.BitDepth)
	combined := avg*lumaMult + int(src)*chromaMult
	v := (combined >> 6) + offset
	return clipInt(v, 0, (1<<params.BitDepth)-1)
}

func applyChromaSample(src uint16, scalingIndex int, grain int16, scaling []uint8, params ChromaRowParams) uint16 {
	scale := scaleLUT(scaling, scalingIndex, params.BitDepth)
	noise := roundPowerOfTwo(int(scale)*int(grain), int(params.ScalingShift))
	minValue := 0
	maxValue := (256 << (params.BitDepth - 8)) - 1
	if params.ClipToRestrictedRange {
		minValue = LumaLegalMin << (params.BitDepth - 8)
		maxLegal := ChromaLegalMax
		if params.MatrixIdentity {
			maxLegal = ChromaIdentityMax
		}
		maxValue = maxLegal << (params.BitDepth - 8)
	}
	return uint16(clipInt(int(src)+noise, minValue, maxValue))
}

func blendChromaOverlap(previous int16, current int16, offset int, shift int, bitDepth uint8) int16 {
	if shift != 0 {
		v := roundPowerOfTwo(int(previous)*23+int(current)*22, 5)
		grainMin := -(1 << (bitDepth - 1))
		grainMax := (1 << (bitDepth - 1)) - 1
		return int16(clipInt(v, grainMin, grainMax))
	}
	return blendLumaOverlap(previous, current, offset, bitDepth)
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

func validateChromaRow(dst []uint16, src []uint16, luma []uint16, grain []int16, scaling []uint8, params ChromaRowParams) error {
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
		(params.BitDepth != 8 && params.BitDepth != 10 && params.BitDepth != 12) ||
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

func checkedAdd(a int, b int) (int, bool) {
	c := a + b
	if (b > 0 && c < a) || (b < 0 && c > a) {
		return 0, false
	}
	return c, true
}

func checkedMulNonNegative(a int, b int) (int, bool) {
	if a < 0 || b < 0 {
		return 0, false
	}
	if a != 0 && b > int(^uint(0)>>1)/a {
		return 0, false
	}
	return a * b, true
}
