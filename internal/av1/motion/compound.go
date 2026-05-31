// Ported from libaom: av1/common/convolve.c (av1_dist_wtd_convolve_* and the
// highbd variants), aom_dsp/blend_a64_mask.c (aom_*_blend_a64_d16_mask_c) and
// av1/common/reconinter.c (diffwtd_mask_d16). These implement the AV1 compound
// inter-prediction path, which keeps each reference predictor at the un-rounded
// 16-bit CONV_BUF precision (round_1 = COMPOUND_ROUND1_BITS) and only rounds to
// the final pixel after blending. Blending two already-rounded 8-bit predictors
// (the single-prediction path) loses up to a few LSBs of precision, so compound
// blocks must go through this dedicated path.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

package motion

import "github.com/thesyncim/goav1/internal/av1/frame"

const (
	// libaom convolve.h: ROUND0_BITS / COMPOUND_ROUND1_BITS for the compound
	// (do_average) convolve. round_0 stays 3 and round_1 stays 7 for all bit
	// depths <= 10 (the highbd intbufrange adjustment only fires at bd == 12).
	compoundRound0Bits = round0Bits
	compoundRound1Bits = 7
	// aom_dsp/blend.h
	compoundBlendA64RoundBits = 6
	compoundBlendA64MaxAlpha  = 64
	// CONV_BUF intermediate scratch sizing.
	compoundMaxConvSamples = maxBlockSize * maxBlockSize
)

// compoundRound0 ports get_conv_params_no_round() round_0 selection for the
// compound (is_compound) convolve. round_0 = ROUND0_BITS (3) for bd <= 10 and
// is raised to 5 at bd == 12 by the intbufrange adjustment (intbufrange =
// bd + FILTER_BITS - round_0 + 2 > 16). Unlike the single-prediction path,
// COMPOUND_ROUND1_BITS is left unchanged for compound (libaom only lowers
// round_1 when !is_compound), so callers keep compoundRound1Bits.
func compoundRound0(bitDepth uint8) int {
	round0 := compoundRound0Bits
	intbufrange := int(bitDepth) + filterBits - round0 + 2
	if intbufrange > 16 {
		round0 += intbufrange - 16
	}
	return round0
}

// CompoundConvBuf holds one reference's un-rounded 16-bit CONV_BUF predictor
// for compound inter prediction.
type CompoundConvBuf struct {
	Data   [compoundMaxConvSamples]uint16
	Width  int
	Height int
}

func compoundConvBufView(buf *CompoundConvBuf, width int, height int) ([]uint16, bool) {
	if width <= 0 || height <= 0 || width > maxBlockSize || height > maxBlockSize {
		return nil, false
	}
	buf.Width = width
	buf.Height = height
	return buf.Data[:width*height], true
}

// PredictInterCompoundRefToConvBuf fills buf with the un-rounded 16-bit CONV_BUF
// predictor (av1_dist_wtd_convolve_* with do_average == 0) for one reference.
// The block origin (refX, refY) and subpel phases match the single-prediction
// path; the result is later combined by a compound blend.
func PredictInterCompoundRefToConvBuf(buf *CompoundConvBuf, ref frame.Plane, bytesPerSample int, bitDepth uint8, refX int, refY int, width int, height int, subX int, subY int, filters InterpFilters) error {
	if buf == nil || !filters.X.Valid() || !filters.Y.Valid() {
		return ErrInvalidMotion
	}
	if subX < 0 || subX > subpelQ4Mask || subY < 0 || subY > subpelQ4Mask {
		return ErrInvalidMotion
	}
	if !bitDepthMatchesSampleWidth(bytesPerSample, bitDepth) {
		return ErrInvalidMotion
	}
	out, ok := compoundConvBufView(buf, width, height)
	if !ok {
		return ErrInvalidMotion
	}
	bd := int(bitDepth)
	xKernel, err := interpKernel(filters.X, width, subX)
	if err != nil {
		return err
	}
	yKernel, err := interpKernel(filters.Y, height, subY)
	if err != nil {
		return err
	}
	load := func(x, y int) int {
		switch bytesPerSample {
		case 1:
			return int(loadSample8Clamped(ref, x, y))
		default:
			return int(loadHighBDSampleClamped(ref, x, y))
		}
	}
	foX := filterTaps/2 - 1
	foY := filterTaps/2 - 1
	round0 := compoundRound0(bitDepth)
	offsetBits := bd + 2*filterBits - round0
	roundOffset := (1 << (offsetBits - compoundRound1Bits)) + (1 << (offsetBits - compoundRound1Bits - 1))
	switch {
	case subX != 0 && subY != 0:
		const imStride = maxBlockSize
		var im [((maxBlockSize + filterTaps - 1) * maxBlockSize)]int32
		imH := height + filterTaps - 1
		for y := range imH {
			for x := range width {
				sum := 1 << (bd + filterBits - 1)
				for k := range filterTaps {
					sum += int(xKernel[k]) * load(refX+x-foX+k, refY-foY+y)
				}
				im[y*imStride+x] = int32(roundPowerOfTwo(sum, round0))
			}
		}
		for y := range height {
			for x := range width {
				sum := 1 << offsetBits
				for k := range filterTaps {
					sum += int(yKernel[k]) * int(im[(y+k)*imStride+x])
				}
				out[y*width+x] = uint16(roundPowerOfTwo(sum, compoundRound1Bits))
			}
		}
	case subX != 0:
		// av1_dist_wtd_convolve_x: bits = FILTER_BITS - round_1.
		bits := filterBits - compoundRound1Bits
		for y := range height {
			for x := range width {
				res := 0
				for k := range filterTaps {
					res += int(xKernel[k]) * load(refX+x-foX+k, refY+y)
				}
				res = (1 << bits) * roundPowerOfTwo(res, round0)
				res += roundOffset
				out[y*width+x] = uint16(res)
			}
		}
	case subY != 0:
		// av1_dist_wtd_convolve_y: bits = FILTER_BITS - round_0.
		bits := filterBits - round0
		for y := range height {
			for x := range width {
				res := 0
				for k := range filterTaps {
					res += int(yKernel[k]) * load(refX+x, refY+y-foY+k)
				}
				res *= (1 << bits)
				res = roundPowerOfTwo(res, compoundRound1Bits) + roundOffset
				out[y*width+x] = uint16(res)
			}
		}
	default:
		// av1_dist_wtd_convolve_2d_copy: bits = 2*FILTER_BITS - round_1 - round_0.
		bits := 2*filterBits - compoundRound1Bits - round0
		for y := range height {
			for x := range width {
				res := load(refX+x, refY+y) << bits
				res += roundOffset
				out[y*width+x] = uint16(res)
			}
		}
	}
	return nil
}

// PredictScaledCompoundRefToConvBuf fills buf with the un-rounded 16-bit
// CONV_BUF predictor for one scaled reference. It mirrors
// av1_dist_wtd_convolve_2d_scale_c / av1_highbd_dist_wtd_convolve_2d_scale_c
// with do_average == 0, so compound blends can keep the scaled predictor at
// the same precision as the same-size translational path.
func PredictScaledCompoundRefToConvBuf(buf *CompoundConvBuf, ref frame.Plane, bytesPerSample int, bitDepth uint8, width int, height int,
	startX int64, xStep int64, startY int64, yStep int64,
	xTable SubpelKernelTable, yTable SubpelKernelTable) error {
	return PredictScaledCompoundRefToConvBufWithScratch(buf, ref, bytesPerSample, bitDepth, width, height,
		startX, xStep, startY, yStep, xTable, yTable, nil)
}

// PredictScaledCompoundRefToConvBufWithScratch is
// PredictScaledCompoundRefToConvBuf using optional caller-owned scratch for the
// large scaled intermediate block.
func PredictScaledCompoundRefToConvBufWithScratch(buf *CompoundConvBuf, ref frame.Plane, bytesPerSample int, bitDepth uint8, width int, height int,
	startX int64, xStep int64, startY int64, yStep int64,
	xTable SubpelKernelTable, yTable SubpelKernelTable, scratch *ScaledConvolveScratch) error {
	if buf == nil || xStep <= 0 || yStep <= 0 ||
		!bitDepthMatchesSampleWidth(bytesPerSample, bitDepth) {
		return ErrInvalidMotion
	}
	out, ok := compoundConvBufView(buf, width, height)
	if !ok {
		return ErrInvalidMotion
	}
	imH, ok := scaledIMHeight(height, startY, yStep)
	if !ok {
		return ErrInvalidMotion
	}
	if !planeRegionFits(ref, bytesPerSample, 0, 0, ref.Width, ref.Height) {
		return ErrInvalidMotion
	}

	const imStride = maxBlockSize
	im, pooled := scaledHighBDIMForScratch(scratch)
	foX := filterTaps/2 - 1
	foY := filterTaps/2 - 1
	startRow := int(scaledIntFloor(startY)) - foY
	round0 := compoundRound0(bitDepth)
	for y := range imH {
		srcRow := startRow + y
		xPos := startX
		for x := range width {
			xInt := int(scaledIntFloor(xPos)) - foX
			xFilterIdx := int(scaledSubpel(xPos) >> ScaleExtraBits)
			kernel := xTable[xFilterIdx]
			sum := 1 << (int(bitDepth) + filterBits - 1)
			for k := range filterTaps {
				var sample int
				if bytesPerSample == 1 {
					sample = int(loadSample8Clamped(ref, xInt+k, srcRow))
				} else {
					sample = int(loadHighBDSampleClamped(ref, xInt+k, srcRow))
				}
				sum += int(kernel[k]) * sample
			}
			im[y*imStride+x] = int32(roundPowerOfTwo(sum, round0))
			xPos += xStep
		}
	}

	offsetBits := int(bitDepth) + 2*filterBits - round0
	baseY := int(scaledIntFloor(startY))
	for x := range width {
		yPos := startY
		for y := range height {
			yInt := int(scaledIntFloor(yPos)) - baseY
			yFilterIdx := int(scaledSubpel(yPos) >> ScaleExtraBits)
			kernel := yTable[yFilterIdx]
			sum := 1 << offsetBits
			for k := range filterTaps {
				sum += int(kernel[k]) * int(im[(yInt+k)*imStride+x])
			}
			out[y*width+x] = uint16(roundPowerOfTwo(sum, compoundRound1Bits))
			yPos += yStep
		}
	}
	putScaledHighBDIM(im, pooled)
	return nil
}

// PredictWarpedCompoundToConvBuf fills buf with the un-rounded 16-bit CONV_BUF
// warp predictor for one compound reference (av1_warp_affine_c with
// is_compound && !do_average). It mirrors the warp filter in warp.go but emits
// the intermediate value at COMPOUND_ROUND1_BITS precision instead of the final
// clipped pixel, so the compound blend can round once after combining the two
// references. matX/matY are the block's true plane coordinates (libaom
// p_col/p_row); the result is written into buf at (0,0).
func PredictWarpedCompoundToConvBuf(buf *CompoundConvBuf, ref frame.Plane, bytesPerSample int, bitDepth uint8, matX int, matY int, width int, height int, matrix [6]int32, alpha int16, beta int16, gamma int16, delta int16, subsamplingX bool, subsamplingY bool) error {
	if buf == nil || width <= 0 || height <= 0 || width > maxBlockSize || height > maxBlockSize {
		return ErrInvalidMotion
	}
	if !planeRegionFits(ref, bytesPerSample, 0, 0, ref.Width, ref.Height) ||
		!bitDepthMatchesSampleWidth(bytesPerSample, bitDepth) {
		return ErrInvalidMotion
	}
	out, ok := compoundConvBufView(buf, width, height)
	if !ok {
		return ErrInvalidMotion
	}
	ssX := 0
	if subsamplingX {
		ssX = 1
	}
	ssY := 0
	if subsamplingY {
		ssY = 1
	}
	bd := int(bitDepth)
	a, be, g, d := int(alpha), int(beta), int(gamma), int(delta)
	// av1_highbd_warp_affine_c with is_compound: reduce_bits_horiz = round_0
	// (bd-dependent, 5 at bd == 12) and reduce_bits_vert = round_1
	// (COMPOUND_ROUND1_BITS, unchanged for compound).
	reduceBitsHoriz := compoundRound0(bitDepth)
	reduceBitsVert := compoundRound1Bits
	offsetBitsHoriz := bd + filterBits - 1
	offsetBitsVert := bd + 2*filterBits - reduceBitsHoriz
	var tmp [warpedIntermediateRows * warpedIntermediateColumns]int32
	for i := matY; i < matY+height; i += 8 {
		for j := matX; j < matX+width; j += 8 {
			if bytesPerSample == 1 {
				warpHorizontal8(&tmp, ref, i, j, matrix, a, be, g, d, ssX, ssY, reduceBitsHoriz, offsetBitsHoriz)
			} else {
				warpHorizontalHighBD(&tmp, ref, i, j, matrix, a, be, g, d, ssX, ssY, reduceBitsHoriz, offsetBitsHoriz)
			}
			for k := -4; k < minWarpInt(4, matY+height-i-4); k++ {
				sy := warpBlockSY(i, j, matrix, a, be, g, d, ssX, ssY) + d*(k+4)
				for l := -4; l < minWarpInt(4, matX+width-j-4); l++ {
					offs := roundPowerOfTwo(sy, warpedDiffPrecBits) + warpedPixelPrecShifts
					if offs < 0 || offs >= len(warpedFilter) {
						sy += g
						continue
					}
					coeffs := warpedFilter[offs]
					sum := 1 << offsetBitsVert
					for m := range filterTaps {
						sum += int(coeffs[m]) * int(tmp[(k+m+4)*warpedIntermediateColumns+(l+4)])
					}
					sum = roundPowerOfTwo(sum, reduceBitsVert)
					out[(i-matY+k+4)*width+(j-matX+l+4)] = uint16(sum)
					sy += g
				}
			}
		}
	}
	return nil
}

// BlendCompoundAvg blends two CONV_BUF predictors with distance weights
// (fwdOffset/bckOffset; both 8 for a plain average) and writes the final pixels
// (av1_dist_wtd_convolve_* do_average branch).
func BlendCompoundAvg(dst frame.Plane, buf0 *CompoundConvBuf, buf1 *CompoundConvBuf, bytesPerSample int, bitDepth uint8, dstX int, dstY int, width int, height int, fwdOffset int, bckOffset int) error {
	if buf0 == nil || buf1 == nil ||
		buf0.Width != width || buf0.Height != height ||
		buf1.Width != width || buf1.Height != height {
		return ErrInvalidMotion
	}
	if !planeRegionFits(dst, bytesPerSample, dstX, dstY, width, height) ||
		!bitDepthMatchesSampleWidth(bytesPerSample, bitDepth) {
		return ErrInvalidMotion
	}
	bd := int(bitDepth)
	round0 := compoundRound0(bitDepth)
	offsetBits := bd + 2*filterBits - round0
	roundOffset := (1 << (offsetBits - compoundRound1Bits)) + (1 << (offsetBits - compoundRound1Bits - 1))
	roundBits := 2*filterBits - round0 - compoundRound1Bits
	src0 := buf0.Data[:width*height]
	src1 := buf1.Data[:width*height]
	store := compoundPixelStorer(dst, bytesPerSample, bitDepth)
	for y := range height {
		for x := range width {
			i := y*width + x
			tmp := int(src0[i])*fwdOffset + int(src1[i])*bckOffset
			tmp >>= 4 // DIST_PRECISION_BITS
			tmp -= roundOffset
			store(dstX+x, dstY+y, roundPowerOfTwo(tmp, roundBits))
		}
	}
	return nil
}

// BlendCompoundMaskD16 blends two CONV_BUF predictors through a per-pixel A64
// soft mask (wedge / diff-weighted) at the d16 precision and writes the final
// pixels (aom_*_blend_a64_d16_mask_c).
func BlendCompoundMaskD16(dst frame.Plane, buf0 *CompoundConvBuf, buf1 *CompoundConvBuf, bytesPerSample int, bitDepth uint8, dstX int, dstY int, width int, height int, mask []byte, maskStride int, subX bool, subY bool) error {
	if buf0 == nil || buf1 == nil ||
		buf0.Width != width || buf0.Height != height ||
		buf1.Width != width || buf1.Height != height {
		return ErrInvalidMotion
	}
	if !planeRegionFits(dst, bytesPerSample, dstX, dstY, width, height) ||
		!bitDepthMatchesSampleWidth(bytesPerSample, bitDepth) {
		return ErrInvalidMotion
	}
	bd := int(bitDepth)
	round0 := compoundRound0(bitDepth)
	offsetBits := bd + 2*filterBits - round0
	roundOffset := (1 << (offsetBits - compoundRound1Bits)) + (1 << (offsetBits - compoundRound1Bits - 1))
	roundBits := 2*filterBits - round0 - compoundRound1Bits
	src0 := buf0.Data[:width*height]
	src1 := buf1.Data[:width*height]
	store := compoundPixelStorer(dst, bytesPerSample, bitDepth)
	for y := range height {
		for x := range width {
			m, ok := compoundMaskSample(mask, maskStride, y, x, subX, subY)
			if !ok {
				return ErrInvalidMotion
			}
			i := y*width + x
			res := (m*int(src0[i]) + (compoundBlendA64MaxAlpha-m)*int(src1[i])) >> compoundBlendA64RoundBits
			res -= roundOffset
			store(dstX+x, dstY+y, roundPowerOfTwo(res, roundBits))
		}
	}
	return nil
}

// BuildDiffWtdMaskD16 builds the difference-weighted compound mask from two
// CONV_BUF predictors (av1_build_compound_diffwtd_mask_d16_c).
func BuildDiffWtdMaskD16(mask []byte, maskStride int, buf0 *CompoundConvBuf, buf1 *CompoundConvBuf, bitDepth uint8, width int, height int, invert bool) error {
	if buf0 == nil || buf1 == nil ||
		buf0.Width != width || buf0.Height != height ||
		buf1.Width != width || buf1.Height != height {
		return ErrInvalidMotion
	}
	if maskStride < width || len(mask) < (height-1)*maskStride+width {
		return ErrInvalidMotion
	}
	bd := int(bitDepth)
	round := 2*filterBits - compoundRound0(bitDepth) - compoundRound1Bits + (bd - 8)
	const maskBase = 38
	const diffFactor = 16
	src0 := buf0.Data[:width*height]
	src1 := buf1.Data[:width*height]
	for y := range height {
		for x := range width {
			i := y*width + x
			diff := int(src0[i]) - int(src1[i])
			if diff < 0 {
				diff = -diff
			}
			diff = roundPowerOfTwo(diff, round)
			m := max(maskBase+diff/diffFactor, 0)
			if m > compoundBlendA64MaxAlpha {
				m = compoundBlendA64MaxAlpha
			}
			if invert {
				m = compoundBlendA64MaxAlpha - m
			}
			mask[y*maskStride+x] = byte(m)
		}
	}
	return nil
}

func compoundPixelStorer(dst frame.Plane, bytesPerSample int, bitDepth uint8) func(x, y, v int) {
	if bytesPerSample == 1 {
		return func(x, y, v int) {
			dst.Pix[y*dst.Stride+x] = clipPixel(v)
		}
	}
	max, _ := highBDMax(bitDepth)
	return func(x, y, v int) {
		storeHighBDSample(dst, x, y, clipPixelHighBD(v, max))
	}
}

// compoundMaskSample reads a (possibly subsampled) A64 mask sample, mirroring
// aom_*_blend_a64_d16_mask_c subw/subh handling.
func compoundMaskSample(mask []byte, stride int, row int, col int, subX bool, subY bool) (int, bool) {
	switch {
	case !subX && !subY:
		idx := row*stride + col
		if idx < 0 || idx >= len(mask) {
			return 0, false
		}
		return int(mask[idx]), true
	case subX && subY:
		base := (2 * row) * stride
		base2 := (2*row + 1) * stride
		if base2+2*col+1 >= len(mask) {
			return 0, false
		}
		sum := int(mask[base+2*col]) + int(mask[base2+2*col]) +
			int(mask[base+2*col+1]) + int(mask[base2+2*col+1])
		return roundPowerOfTwoCompound(sum, 2), true
	case subX:
		idx := row*stride + 2*col
		if idx+1 >= len(mask) {
			return 0, false
		}
		return (int(mask[idx]) + int(mask[idx+1]) + 1) >> 1, true
	default:
		base := (2 * row) * stride
		base2 := (2*row + 1) * stride
		if base2+col >= len(mask) {
			return 0, false
		}
		return (int(mask[base+col]) + int(mask[base2+col]) + 1) >> 1, true
	}
}

func roundPowerOfTwoCompound(value int, bits int) int {
	return (value + (1 << (bits - 1))) >> bits
}
