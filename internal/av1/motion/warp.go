// Ported from libaom: av1/common/warped_motion.c
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

package motion

import "github.com/thesyncim/goav1/internal/av1/frame"

const (
	warpedModelPrecBits       = 16
	warpedModelOne            = 1 << warpedModelPrecBits
	warpedPixelPrecBits       = 6
	warpedPixelPrecShifts     = 1 << warpedPixelPrecBits
	warpParamReduceBits       = 6
	warpedDiffPrecBits        = warpedModelPrecBits - warpedPixelPrecBits
	warpedIntermediateRows    = 15
	warpedIntermediateColumns = 8
)

// PredictWarpedPlaneBlockBitDepth predicts a single-reference affine warped
// block. The destination coordinates are absolute plane coordinates, matching
// libaom's p_col/p_row convention.
func PredictWarpedPlaneBlockBitDepth(dst frame.Plane, ref frame.Plane, bytesPerSample int, bitDepth uint8, dstX int, dstY int, width int, height int, matrix [6]int32, alpha int16, beta int16, gamma int16, delta int16, subsamplingX bool, subsamplingY bool) error {
	if width <= 0 || height <= 0 || width > maxBlockSize || height > maxBlockSize {
		return ErrInvalidMotion
	}
	if !planeRegionFits(dst, bytesPerSample, dstX, dstY, width, height) ||
		!planeRegionFits(ref, bytesPerSample, 0, 0, ref.Width, ref.Height) {
		return ErrInvalidMotion
	}
	if !bitDepthMatchesSampleWidth(bytesPerSample, bitDepth) {
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
	if bytesPerSample == 1 {
		warpAffine8(dst, ref, dstX, dstY, width, height, matrix, int(alpha), int(beta), int(gamma), int(delta), ssX, ssY)
		return nil
	}
	max, ok := highBDMax(bitDepth)
	if !ok {
		return ErrInvalidMotion
	}
	warpAffineHighBD(dst, ref, bitDepth, max, dstX, dstY, width, height, matrix, int(alpha), int(beta), int(gamma), int(delta), ssX, ssY)
	return nil
}

// PredictWarpedPlaneBlockToScratchBitDepth writes a warped block prediction into
// a block-sized scratch plane (origin 0,0) while sampling the affine model at the
// block's true plane coordinates (matX, matY). It mirrors PredictWarpedPlaneBlockBitDepth
// but stages the result for masked blending (inter-intra / masked compound) rather
// than writing in place. dst must be at least width x height.
func PredictWarpedPlaneBlockToScratchBitDepth(dst frame.Plane, ref frame.Plane, bytesPerSample int, bitDepth uint8, matX int, matY int, width int, height int, matrix [6]int32, alpha int16, beta int16, gamma int16, delta int16, subsamplingX bool, subsamplingY bool) error {
	if width <= 0 || height <= 0 || width > maxBlockSize || height > maxBlockSize {
		return ErrInvalidMotion
	}
	if !planeRegionFits(dst, bytesPerSample, 0, 0, width, height) ||
		!planeRegionFits(ref, bytesPerSample, 0, 0, ref.Width, ref.Height) {
		return ErrInvalidMotion
	}
	if !bitDepthMatchesSampleWidth(bytesPerSample, bitDepth) {
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
	if bytesPerSample == 1 {
		warpAffine8Offset(dst, ref, matX, matY, 0, 0, width, height, matrix, int(alpha), int(beta), int(gamma), int(delta), ssX, ssY)
		return nil
	}
	max, ok := highBDMax(bitDepth)
	if !ok {
		return ErrInvalidMotion
	}
	warpAffineHighBDOffset(dst, ref, bitDepth, max, matX, matY, 0, 0, width, height, matrix, int(alpha), int(beta), int(gamma), int(delta), ssX, ssY)
	return nil
}

func warpAffine8(dst frame.Plane, ref frame.Plane, dstX int, dstY int, width int, height int, matrix [6]int32, alpha int, beta int, gamma int, delta int, ssX int, ssY int) {
	warpAffine8Offset(dst, ref, dstX, dstY, dstX, dstY, width, height, matrix, alpha, beta, gamma, delta, ssX, ssY)
}

// warpAffine8Offset is warpAffine8 with the destination write origin decoupled
// from the affine sampling origin. The matrix is evaluated at absolute plane
// coordinates (matX, matY) (libaom's p_col/p_row), while output samples are
// written at (writeX, writeY) in dst. Passing writeX==matX, writeY==matY gives
// the in-place behaviour used by full-frame warp prediction; the inter-intra
// inter part passes writeX=writeY=0 to stage the warp in a block-sized scratch
// plane while still sampling from the block's true frame position.
func warpAffine8Offset(dst frame.Plane, ref frame.Plane, matX int, matY int, writeX int, writeY int, width int, height int, matrix [6]int32, alpha int, beta int, gamma int, delta int, ssX int, ssY int) {
	var tmp [warpedIntermediateRows * warpedIntermediateColumns]int32
	const bd = 8
	reduceBitsHoriz := round0Bits
	reduceBitsVert := 2*filterBits - reduceBitsHoriz
	offsetBitsHoriz := bd + filterBits - 1
	offsetBitsVert := bd + 2*filterBits - reduceBitsHoriz
	colShift := writeX - matX
	rowShift := writeY - matY
	for i := matY; i < matY+height; i += 8 {
		for j := matX; j < matX+width; j += 8 {
			baseSY := warpHorizontal8(&tmp, ref, i, j, matrix, alpha, beta, gamma, delta, ssX, ssY, reduceBitsHoriz, offsetBitsHoriz)
			for k := -4; k < minWarpInt(4, matY+height-i-4); k++ {
				sy := baseSY + delta*(k+4)
				for l := -4; l < minWarpInt(4, matX+width-j-4); l++ {
					offs := roundPowerOfTwo(sy, warpedDiffPrecBits) + warpedPixelPrecShifts
					if offs < 0 || offs >= len(warpedFilter) {
						continue
					}
					coeffs := warpedFilter[offs]
					sum := 1 << offsetBitsVert
					for m := range filterTaps {
						sum += int(coeffs[m]) * int(tmp[(k+m+4)*warpedIntermediateColumns+(l+4)])
					}
					sum = roundPowerOfTwo(sum, reduceBitsVert)
					dst.Pix[(i+rowShift+k+4)*dst.Stride+j+colShift+l+4] = byte(clipPixel(sum - (1 << (bd - 1)) - (1 << bd)))
					sy += gamma
				}
			}
		}
	}
}

func warpHorizontal8(tmp *[warpedIntermediateRows * warpedIntermediateColumns]int32, ref frame.Plane, i int, j int, matrix [6]int32, alpha int, beta int, gamma int, delta int, ssX int, ssY int, reduceBitsHoriz int, offsetBitsHoriz int) int {
	ix4, sx4, iy4, sy4 := warpBlockOrigin(i, j, matrix, alpha, beta, gamma, delta, ssX, ssY)
	for k := -7; k < 8; k++ {
		iy := clampInt(iy4+k, 0, ref.Height-1)
		row := iy * ref.Stride
		sx := sx4 + beta*(k+4)
		for l := -4; l < 4; l++ {
			ix := ix4 + l - 3
			offs := roundPowerOfTwo(sx, warpedDiffPrecBits) + warpedPixelPrecShifts
			if offs < 0 || offs >= len(warpedFilter) {
				offs = clampInt(offs, 0, len(warpedFilter)-1)
			}
			coeffs := warpedFilter[offs]
			sum := 1 << offsetBitsHoriz
			if ix >= 0 && ix+filterTaps <= ref.Width {
				base := row + ix
				sum += int(ref.Pix[base+0]) * int(coeffs[0])
				sum += int(ref.Pix[base+1]) * int(coeffs[1])
				sum += int(ref.Pix[base+2]) * int(coeffs[2])
				sum += int(ref.Pix[base+3]) * int(coeffs[3])
				sum += int(ref.Pix[base+4]) * int(coeffs[4])
				sum += int(ref.Pix[base+5]) * int(coeffs[5])
				sum += int(ref.Pix[base+6]) * int(coeffs[6])
				sum += int(ref.Pix[base+7]) * int(coeffs[7])
			} else {
				for m := range filterTaps {
					sampleX := clampInt(ix+m, 0, ref.Width-1)
					sum += int(ref.Pix[row+sampleX]) * int(coeffs[m])
				}
			}
			tmp[(k+7)*warpedIntermediateColumns+(l+4)] = int32(roundPowerOfTwo(sum, reduceBitsHoriz))
			sx += alpha
		}
	}
	return sy4
}

func warpAffineHighBD(dst frame.Plane, ref frame.Plane, bitDepth uint8, max uint16, dstX int, dstY int, width int, height int, matrix [6]int32, alpha int, beta int, gamma int, delta int, ssX int, ssY int) {
	warpAffineHighBDOffset(dst, ref, bitDepth, max, dstX, dstY, dstX, dstY, width, height, matrix, alpha, beta, gamma, delta, ssX, ssY)
}

// warpAffineHighBDOffset is warpAffineHighBD with the write origin decoupled
// from the affine sampling origin (see warpAffine8Offset).
func warpAffineHighBDOffset(dst frame.Plane, ref frame.Plane, bitDepth uint8, max uint16, matX int, matY int, writeX int, writeY int, width int, height int, matrix [6]int32, alpha int, beta int, gamma int, delta int, ssX int, ssY int) {
	var tmp [warpedIntermediateRows * warpedIntermediateColumns]int32
	// av1_highbd_warp_affine_c (single prediction): reduce_bits_horiz =
	// conv_params->round_0 and reduce_bits_vert = 2*FILTER_BITS -
	// reduce_bits_horiz, where round_0 follows get_conv_params_no_round (3 for
	// bd <= 10, 5 at bd == 12). highBDRoundBits returns exactly that round_0
	// and the matching 2*FILTER_BITS - round_0 vertical reduce.
	reduceBitsHoriz, reduceBitsVert := highBDRoundBits(bitDepth)
	offsetBitsHoriz := int(bitDepth) + filterBits - 1
	offsetBitsVert := int(bitDepth) + 2*filterBits - reduceBitsHoriz
	colShift := writeX - matX
	rowShift := writeY - matY
	for i := matY; i < matY+height; i += 8 {
		for j := matX; j < matX+width; j += 8 {
			baseSY := warpHorizontalHighBD(&tmp, ref, i, j, matrix, alpha, beta, gamma, delta, ssX, ssY, reduceBitsHoriz, offsetBitsHoriz)
			for k := -4; k < minWarpInt(4, matY+height-i-4); k++ {
				sy := baseSY + delta*(k+4)
				for l := -4; l < minWarpInt(4, matX+width-j-4); l++ {
					offs := roundPowerOfTwo(sy, warpedDiffPrecBits) + warpedPixelPrecShifts
					if offs < 0 || offs >= len(warpedFilter) {
						continue
					}
					coeffs := warpedFilter[offs]
					sum := 1 << offsetBitsVert
					for m := range filterTaps {
						sum += int(coeffs[m]) * int(tmp[(k+m+4)*warpedIntermediateColumns+(l+4)])
					}
					sum = roundPowerOfTwo(sum, reduceBitsVert)
					storeHighBDSample(dst, j+colShift+l+4, i+rowShift+k+4, clipPixelHighBD(sum-(1<<(bitDepth-1))-(1<<bitDepth), max))
					sy += gamma
				}
			}
		}
	}
}

func warpHorizontalHighBD(tmp *[warpedIntermediateRows * warpedIntermediateColumns]int32, ref frame.Plane, i int, j int, matrix [6]int32, alpha int, beta int, gamma int, delta int, ssX int, ssY int, reduceBitsHoriz int, offsetBitsHoriz int) int {
	ix4, sx4, iy4, sy4 := warpBlockOrigin(i, j, matrix, alpha, beta, gamma, delta, ssX, ssY)
	for k := -7; k < 8; k++ {
		iy := clampInt(iy4+k, 0, ref.Height-1)
		sx := sx4 + beta*(k+4)
		for l := -4; l < 4; l++ {
			ix := ix4 + l - 3
			offs := roundPowerOfTwo(sx, warpedDiffPrecBits) + warpedPixelPrecShifts
			if offs < 0 || offs >= len(warpedFilter) {
				offs = clampInt(offs, 0, len(warpedFilter)-1)
			}
			coeffs := warpedFilter[offs]
			sum := 1 << offsetBitsHoriz
			if ix >= 0 && ix+filterTaps <= ref.Width {
				sum += int(loadHighBDSample(ref, ix+0, iy)) * int(coeffs[0])
				sum += int(loadHighBDSample(ref, ix+1, iy)) * int(coeffs[1])
				sum += int(loadHighBDSample(ref, ix+2, iy)) * int(coeffs[2])
				sum += int(loadHighBDSample(ref, ix+3, iy)) * int(coeffs[3])
				sum += int(loadHighBDSample(ref, ix+4, iy)) * int(coeffs[4])
				sum += int(loadHighBDSample(ref, ix+5, iy)) * int(coeffs[5])
				sum += int(loadHighBDSample(ref, ix+6, iy)) * int(coeffs[6])
				sum += int(loadHighBDSample(ref, ix+7, iy)) * int(coeffs[7])
			} else {
				for m := range filterTaps {
					sampleX := clampInt(ix+m, 0, ref.Width-1)
					sum += int(loadHighBDSample(ref, sampleX, iy)) * int(coeffs[m])
				}
			}
			tmp[(k+7)*warpedIntermediateColumns+(l+4)] = int32(roundPowerOfTwo(sum, reduceBitsHoriz))
			sx += alpha
		}
	}
	return sy4
}

func warpBlockOrigin(i int, j int, matrix [6]int32, alpha int, beta int, gamma int, delta int, ssX int, ssY int) (int, int, int, int) {
	srcX := int64((j + 4) << ssX)
	srcY := int64((i + 4) << ssY)
	dstX := int64(matrix[2])*srcX + int64(matrix[3])*srcY + int64(matrix[0])
	dstY := int64(matrix[4])*srcX + int64(matrix[5])*srcY + int64(matrix[1])
	x4 := dstX >> ssX
	y4 := dstY >> ssY
	ix4 := int(x4 >> warpedModelPrecBits)
	sx4 := int(x4 & (warpedModelOne - 1))
	iy4 := int(y4 >> warpedModelPrecBits)
	sy4 := int(y4 & (warpedModelOne - 1))
	sx4 += alpha*(-4) + beta*(-4)
	sy4 += gamma*(-4) + delta*(-4)
	sx4 &^= (1 << warpParamReduceBits) - 1
	sy4 &^= (1 << warpParamReduceBits) - 1
	return ix4, sx4, iy4, sy4
}

func minWarpInt(a int, b int) int {
	if a < b {
		return a
	}
	return b
}
