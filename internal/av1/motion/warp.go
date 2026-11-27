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

// affineWarp holds the validated plane views, sampling/write origins, and
// local affine model parameters for one warped-motion prediction block.
//
// Methods take a value receiver so Go's escape analysis keeps the 15x8
// intermediate filter buffer on the stack with zero heap allocations.
type affineWarp struct {
	dst, ref       frame.Plane
	matX, matY     int
	writeX, writeY int
	width, height  int
	matrix         [6]int32
	alpha, beta    int
	gamma, delta   int
	ssX, ssY       int
}

func newAffineWarp(dst, ref frame.Plane, matX, matY, writeX, writeY, width, height int, matrix [6]int32, alpha, beta, gamma, delta int16, subsamplingX, subsamplingY bool) affineWarp {
	ssX, ssY := 0, 0
	if subsamplingX {
		ssX = 1
	}
	if subsamplingY {
		ssY = 1
	}
	return affineWarp{
		dst: dst, ref: ref,
		matX: matX, matY: matY,
		writeX: writeX, writeY: writeY,
		width: width, height: height,
		matrix: matrix,
		alpha:  int(alpha), beta: int(beta),
		gamma: int(gamma), delta: int(delta),
		ssX: ssX, ssY: ssY,
	}
}

func (w affineWarp) predict(bytesPerSample int, bitDepth uint8) error {
	if w.width <= 0 || w.height <= 0 || w.width > maxBlockSize || w.height > maxBlockSize {
		return ErrInvalidMotion
	}
	if !planeRegionFits(w.dst, bytesPerSample, w.writeX, w.writeY, w.width, w.height) ||
		!planeRegionFits(w.ref, bytesPerSample, 0, 0, w.ref.Width, w.ref.Height) {
		return ErrInvalidMotion
	}
	if !bitDepthMatchesSampleWidth(bytesPerSample, bitDepth) {
		return ErrInvalidMotion
	}
	if bytesPerSample == 1 {
		w.apply8()
		return nil
	}
	maxVal, ok := highBDMax(bitDepth)
	if !ok {
		return ErrInvalidMotion
	}
	w.applyHighBD(bitDepth, maxVal)
	return nil
}

// PredictWarpedPlaneBlockBitDepth predicts a single-reference affine warped
// block. The destination coordinates are absolute plane coordinates, matching
// libaom's p_col/p_row convention.
//
// Accesses stay inside ref.Pix (taps outside the plane are edge-clamped,
// never read) and inside the width x height block of the destination.
func PredictWarpedPlaneBlockBitDepth(dst frame.Plane, ref frame.Plane, bytesPerSample int, bitDepth uint8, dstX int, dstY int, width int, height int, matrix [6]int32, alpha int16, beta int16, gamma int16, delta int16, subsamplingX bool, subsamplingY bool) error {
	return newAffineWarp(dst, ref, dstX, dstY, dstX, dstY, width, height, matrix, alpha, beta, gamma, delta, subsamplingX, subsamplingY).predict(bytesPerSample, bitDepth)
}

// PredictWarpedPlaneBlockToScratchBitDepth writes a warped block prediction into
// a block-sized scratch plane (origin 0,0) while sampling the affine model at the
// block's true plane coordinates (matX, matY). It mirrors PredictWarpedPlaneBlockBitDepth
// but stages the result for masked blending (inter-intra / masked compound) rather
// than writing in place. dst must be at least width x height.
func PredictWarpedPlaneBlockToScratchBitDepth(dst frame.Plane, ref frame.Plane, bytesPerSample int, bitDepth uint8, matX int, matY int, width int, height int, matrix [6]int32, alpha int16, beta int16, gamma int16, delta int16, subsamplingX bool, subsamplingY bool) error {
	return newAffineWarp(dst, ref, matX, matY, 0, 0, width, height, matrix, alpha, beta, gamma, delta, subsamplingX, subsamplingY).predict(bytesPerSample, bitDepth)
}

// apply8 runs 8-bit two-pass separable affine warp filtering over 8x8 tiles.
// The affine model is evaluated at absolute plane coordinates (w.matX, w.matY)
// while samples are stored at (w.writeX, w.writeY) in w.dst.
func (w affineWarp) apply8() {
	var tmp [warpedIntermediateRows * warpedIntermediateColumns]int32
	const bd = 8
	reduceBitsHoriz := round0Bits
	reduceBitsVert := 2*filterBits - reduceBitsHoriz
	offsetBitsHoriz := bd + filterBits - 1
	offsetBitsVert := bd + 2*filterBits - reduceBitsHoriz
	colShift := w.writeX - w.matX
	rowShift := w.writeY - w.matY
	for i := w.matY; i < w.matY+w.height; i += 8 {
		for j := w.matX; j < w.matX+w.width; j += 8 {
			if i+8 <= w.matY+w.height && j+8 <= w.matX+w.width {
				ix4, sx4, iy4, sy4 := warpBlockOrigin(i, j, w.matrix, w.alpha, w.beta, w.gamma, w.delta, w.ssX, w.ssY)
				o := (i+rowShift)*w.dst.Stride + j + colShift
				if warpPut8x8Dispatch(w.dst.Pix[o:], w.dst.Stride, w.ref, ix4, sx4, iy4, sy4, w.alpha, w.beta, w.gamma, w.delta) {
					continue
				}
			}
			baseSY := warpHorizontal8(&tmp, w.ref, i, j, w.matrix, w.alpha, w.beta, w.gamma, w.delta, w.ssX, w.ssY, reduceBitsHoriz, offsetBitsHoriz)
			if i+8 <= w.matY+w.height && j+8 <= w.matX+w.width {
				if w.gamma == 0 {
					warpVertical8FullGamma0Dispatch(w.dst, &tmp, i, j, rowShift, colShift, baseSY, w.delta, reduceBitsVert, offsetBitsVert)
				} else {
					warpVertical8FullDispatch(w.dst, &tmp, i, j, rowShift, colShift, baseSY, w.gamma, w.delta, reduceBitsVert, offsetBitsVert)
				}
				continue
			}
			for k := -4; k < minWarpInt(4, w.matY+w.height-i-4); k++ {
				sy := baseSY + w.delta*(k+4)
				for l := -4; l < minWarpInt(4, w.matX+w.width-j-4); l++ {
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
					w.dst.Pix[(i+rowShift+k+4)*w.dst.Stride+j+colShift+l+4] = byte(clipPixel(sum - (1 << (bd - 1)) - (1 << bd)))
					sy += w.gamma
				}
			}
		}
	}
}

func warpVertical8Full(dst frame.Plane, tmp *[warpedIntermediateRows * warpedIntermediateColumns]int32, i int, j int, rowShift int, colShift int, baseSY int, gamma int, delta int, reduceBitsVert int, offsetBitsVert int) {
	for k := -4; k < 4; k++ {
		sy := baseSY + delta*(k+4)
		dstRow := (i+rowShift+k+4)*dst.Stride + j + colShift
		tmpRow := (k + 4) * warpedIntermediateColumns
		for l := -4; l < 4; l++ {
			offs := roundPowerOfTwo(sy, warpedDiffPrecBits) + warpedPixelPrecShifts
			if offs >= 0 && offs < len(warpedFilter) {
				coeffs := warpedFilter[offs]
				col := l + 4
				sum := 1 << offsetBitsVert
				sum += int(coeffs[0]) * int(tmp[tmpRow+0*warpedIntermediateColumns+col])
				sum += int(coeffs[1]) * int(tmp[tmpRow+1*warpedIntermediateColumns+col])
				sum += int(coeffs[2]) * int(tmp[tmpRow+2*warpedIntermediateColumns+col])
				sum += int(coeffs[3]) * int(tmp[tmpRow+3*warpedIntermediateColumns+col])
				sum += int(coeffs[4]) * int(tmp[tmpRow+4*warpedIntermediateColumns+col])
				sum += int(coeffs[5]) * int(tmp[tmpRow+5*warpedIntermediateColumns+col])
				sum += int(coeffs[6]) * int(tmp[tmpRow+6*warpedIntermediateColumns+col])
				sum += int(coeffs[7]) * int(tmp[tmpRow+7*warpedIntermediateColumns+col])
				sum = roundPowerOfTwo(sum, reduceBitsVert)
				dst.Pix[dstRow+l+4] = byte(clipPixel(sum - (1 << 7) - (1 << 8)))
			}
			sy += gamma
		}
	}
}

func warpVertical8FullGamma0(dst frame.Plane, tmp *[warpedIntermediateRows * warpedIntermediateColumns]int32, i int, j int, rowShift int, colShift int, baseSY int, delta int, reduceBitsVert int, offsetBitsVert int) {
	for k := -4; k < 4; k++ {
		sy := baseSY + delta*(k+4)
		dstRow := (i+rowShift+k+4)*dst.Stride + j + colShift
		tmpRow := (k + 4) * warpedIntermediateColumns
		offs := roundPowerOfTwo(sy, warpedDiffPrecBits) + warpedPixelPrecShifts
		if offs < 0 || offs >= len(warpedFilter) {
			continue
		}
		coeffs := warpedFilter[offs]
		c0 := int(coeffs[0])
		c1 := int(coeffs[1])
		c2 := int(coeffs[2])
		c3 := int(coeffs[3])
		c4 := int(coeffs[4])
		c5 := int(coeffs[5])
		c6 := int(coeffs[6])
		c7 := int(coeffs[7])
		for col := 0; col < 8; col++ {
			sum := 1 << offsetBitsVert
			sum += c0 * int(tmp[tmpRow+0*warpedIntermediateColumns+col])
			sum += c1 * int(tmp[tmpRow+1*warpedIntermediateColumns+col])
			sum += c2 * int(tmp[tmpRow+2*warpedIntermediateColumns+col])
			sum += c3 * int(tmp[tmpRow+3*warpedIntermediateColumns+col])
			sum += c4 * int(tmp[tmpRow+4*warpedIntermediateColumns+col])
			sum += c5 * int(tmp[tmpRow+5*warpedIntermediateColumns+col])
			sum += c6 * int(tmp[tmpRow+6*warpedIntermediateColumns+col])
			sum += c7 * int(tmp[tmpRow+7*warpedIntermediateColumns+col])
			sum = roundPowerOfTwo(sum, reduceBitsVert)
			dst.Pix[dstRow+col] = byte(clipPixel(sum - (1 << 7) - (1 << 8)))
		}
	}
}

func warpHorizontal8(tmp *[warpedIntermediateRows * warpedIntermediateColumns]int32, ref frame.Plane, i int, j int, matrix [6]int32, alpha int, beta int, gamma int, delta int, ssX int, ssY int, reduceBitsHoriz int, offsetBitsHoriz int) int {
	ix4, sx4, iy4, sy4 := warpBlockOrigin(i, j, matrix, alpha, beta, gamma, delta, ssX, ssY)
	if iy4 >= 7 && iy4+7 < ref.Height && ix4 >= 7 && ix4+8 <= ref.Width {
		return warpHorizontal8ResidentDispatch(tmp, ref, ix4, sx4, iy4, sy4, alpha, beta, reduceBitsHoriz, offsetBitsHoriz)
	}
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

func warpHorizontal8Resident(tmp *[warpedIntermediateRows * warpedIntermediateColumns]int32, ref frame.Plane, ix4 int, sx4 int, iy4 int, sy4 int, alpha int, beta int, reduceBitsHoriz int, offsetBitsHoriz int) int {
	for k := -7; k < 8; k++ {
		row := (iy4 + k) * ref.Stride
		sx := sx4 + beta*(k+4)
		for l := -4; l < 4; l++ {
			ix := ix4 + l - 3
			offs := roundPowerOfTwo(sx, warpedDiffPrecBits) + warpedPixelPrecShifts
			if offs < 0 || offs >= len(warpedFilter) {
				offs = clampInt(offs, 0, len(warpedFilter)-1)
			}
			coeffs := warpedFilter[offs]
			base := row + ix
			sum := 1 << offsetBitsHoriz
			sum += int(ref.Pix[base+0]) * int(coeffs[0])
			sum += int(ref.Pix[base+1]) * int(coeffs[1])
			sum += int(ref.Pix[base+2]) * int(coeffs[2])
			sum += int(ref.Pix[base+3]) * int(coeffs[3])
			sum += int(ref.Pix[base+4]) * int(coeffs[4])
			sum += int(ref.Pix[base+5]) * int(coeffs[5])
			sum += int(ref.Pix[base+6]) * int(coeffs[6])
			sum += int(ref.Pix[base+7]) * int(coeffs[7])
			tmp[(k+7)*warpedIntermediateColumns+(l+4)] = int32(roundPowerOfTwo(sum, reduceBitsHoriz))
			sx += alpha
		}
	}
	return sy4
}

// applyHighBD runs 10/12-bit two-pass separable affine warp filtering over 8x8
// tiles (libaom av1_highbd_warp_affine_c).
func (w affineWarp) applyHighBD(bitDepth uint8, maxVal uint16) {
	var tmp [warpedIntermediateRows * warpedIntermediateColumns]int32
	// reduce_bits_horiz = conv_params->round_0 (3 for bd <= 10, 5 at bd == 12)
	// and reduce_bits_vert = 2*FILTER_BITS - reduce_bits_horiz.
	reduceBitsHoriz, reduceBitsVert := highBDRoundBits(bitDepth)
	offsetBitsHoriz := int(bitDepth) + filterBits - 1
	offsetBitsVert := int(bitDepth) + 2*filterBits - reduceBitsHoriz
	colShift := w.writeX - w.matX
	rowShift := w.writeY - w.matY
	for i := w.matY; i < w.matY+w.height; i += 8 {
		for j := w.matX; j < w.matX+w.width; j += 8 {
			baseSY := warpHorizontalHighBD(&tmp, w.ref, i, j, w.matrix, w.alpha, w.beta, w.gamma, w.delta, w.ssX, w.ssY, reduceBitsHoriz, offsetBitsHoriz)
			if i+8 <= w.matY+w.height && j+8 <= w.matX+w.width {
				if w.gamma == 0 {
					warpVerticalHighBDFullGamma0Dispatch(w.dst, bitDepth, maxVal, &tmp, i, j, rowShift, colShift, baseSY, w.delta, reduceBitsVert, offsetBitsVert)
				} else {
					warpVerticalHighBDFullDispatch(w.dst, bitDepth, maxVal, &tmp, i, j, rowShift, colShift, baseSY, w.gamma, w.delta, reduceBitsVert, offsetBitsVert)
				}
				continue
			}
			for k := -4; k < minWarpInt(4, w.matY+w.height-i-4); k++ {
				sy := baseSY + w.delta*(k+4)
				for l := -4; l < minWarpInt(4, w.matX+w.width-j-4); l++ {
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
					storeHighBDSample(w.dst, j+colShift+l+4, i+rowShift+k+4, clipPixelHighBD(sum-(1<<(bitDepth-1))-(1<<bitDepth), maxVal))
					sy += w.gamma
				}
			}
		}
	}
}

func warpVerticalHighBDFull(dst frame.Plane, bitDepth uint8, max uint16, tmp *[warpedIntermediateRows * warpedIntermediateColumns]int32, i int, j int, rowShift int, colShift int, baseSY int, gamma int, delta int, reduceBitsVert int, offsetBitsVert int) {
	for k := -4; k < 4; k++ {
		sy := baseSY + delta*(k+4)
		dstY := i + rowShift + k + 4
		tmpRow := (k + 4) * warpedIntermediateColumns
		for l := -4; l < 4; l++ {
			offs := roundPowerOfTwo(sy, warpedDiffPrecBits) + warpedPixelPrecShifts
			if offs >= 0 && offs < len(warpedFilter) {
				coeffs := warpedFilter[offs]
				col := l + 4
				sum := 1 << offsetBitsVert
				sum += int(coeffs[0]) * int(tmp[tmpRow+0*warpedIntermediateColumns+col])
				sum += int(coeffs[1]) * int(tmp[tmpRow+1*warpedIntermediateColumns+col])
				sum += int(coeffs[2]) * int(tmp[tmpRow+2*warpedIntermediateColumns+col])
				sum += int(coeffs[3]) * int(tmp[tmpRow+3*warpedIntermediateColumns+col])
				sum += int(coeffs[4]) * int(tmp[tmpRow+4*warpedIntermediateColumns+col])
				sum += int(coeffs[5]) * int(tmp[tmpRow+5*warpedIntermediateColumns+col])
				sum += int(coeffs[6]) * int(tmp[tmpRow+6*warpedIntermediateColumns+col])
				sum += int(coeffs[7]) * int(tmp[tmpRow+7*warpedIntermediateColumns+col])
				sum = roundPowerOfTwo(sum, reduceBitsVert)
				storeHighBDSample(dst, j+colShift+l+4, dstY, clipPixelHighBD(sum-(1<<(bitDepth-1))-(1<<bitDepth), max))
			}
			sy += gamma
		}
	}
}

func warpVerticalHighBDFullGamma0(dst frame.Plane, bitDepth uint8, max uint16, tmp *[warpedIntermediateRows * warpedIntermediateColumns]int32, i int, j int, rowShift int, colShift int, baseSY int, delta int, reduceBitsVert int, offsetBitsVert int) {
	for k := -4; k < 4; k++ {
		sy := baseSY + delta*(k+4)
		dstY := i + rowShift + k + 4
		tmpRow := (k + 4) * warpedIntermediateColumns
		offs := roundPowerOfTwo(sy, warpedDiffPrecBits) + warpedPixelPrecShifts
		if offs < 0 || offs >= len(warpedFilter) {
			continue
		}
		coeffs := warpedFilter[offs]
		c0 := int(coeffs[0])
		c1 := int(coeffs[1])
		c2 := int(coeffs[2])
		c3 := int(coeffs[3])
		c4 := int(coeffs[4])
		c5 := int(coeffs[5])
		c6 := int(coeffs[6])
		c7 := int(coeffs[7])
		for col := 0; col < 8; col++ {
			sum := 1 << offsetBitsVert
			sum += c0 * int(tmp[tmpRow+0*warpedIntermediateColumns+col])
			sum += c1 * int(tmp[tmpRow+1*warpedIntermediateColumns+col])
			sum += c2 * int(tmp[tmpRow+2*warpedIntermediateColumns+col])
			sum += c3 * int(tmp[tmpRow+3*warpedIntermediateColumns+col])
			sum += c4 * int(tmp[tmpRow+4*warpedIntermediateColumns+col])
			sum += c5 * int(tmp[tmpRow+5*warpedIntermediateColumns+col])
			sum += c6 * int(tmp[tmpRow+6*warpedIntermediateColumns+col])
			sum += c7 * int(tmp[tmpRow+7*warpedIntermediateColumns+col])
			sum = roundPowerOfTwo(sum, reduceBitsVert)
			storeHighBDSample(dst, j+colShift+col, dstY, clipPixelHighBD(sum-(1<<(bitDepth-1))-(1<<bitDepth), max))
		}
	}
}

func warpHorizontalHighBD(tmp *[warpedIntermediateRows * warpedIntermediateColumns]int32, ref frame.Plane, i int, j int, matrix [6]int32, alpha int, beta int, gamma int, delta int, ssX int, ssY int, reduceBitsHoriz int, offsetBitsHoriz int) int {
	ix4, sx4, iy4, sy4 := warpBlockOrigin(i, j, matrix, alpha, beta, gamma, delta, ssX, ssY)
	if iy4 >= 7 && iy4+7 < ref.Height && ix4 >= 7 && ix4+8 <= ref.Width {
		return warpHorizontalHighBDResidentDispatch(tmp, ref, ix4, sx4, iy4, sy4, alpha, beta, reduceBitsHoriz, offsetBitsHoriz)
	}
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
				base := row + ix*2
				sum += int(uint16(ref.Pix[base+0])|uint16(ref.Pix[base+1])<<8) * int(coeffs[0])
				sum += int(uint16(ref.Pix[base+2])|uint16(ref.Pix[base+3])<<8) * int(coeffs[1])
				sum += int(uint16(ref.Pix[base+4])|uint16(ref.Pix[base+5])<<8) * int(coeffs[2])
				sum += int(uint16(ref.Pix[base+6])|uint16(ref.Pix[base+7])<<8) * int(coeffs[3])
				sum += int(uint16(ref.Pix[base+8])|uint16(ref.Pix[base+9])<<8) * int(coeffs[4])
				sum += int(uint16(ref.Pix[base+10])|uint16(ref.Pix[base+11])<<8) * int(coeffs[5])
				sum += int(uint16(ref.Pix[base+12])|uint16(ref.Pix[base+13])<<8) * int(coeffs[6])
				sum += int(uint16(ref.Pix[base+14])|uint16(ref.Pix[base+15])<<8) * int(coeffs[7])
			} else {
				for m := range filterTaps {
					sampleX := clampInt(ix+m, 0, ref.Width-1)
					sample := row + sampleX*2
					sum += int(uint16(ref.Pix[sample])|uint16(ref.Pix[sample+1])<<8) * int(coeffs[m])
				}
			}
			tmp[(k+7)*warpedIntermediateColumns+(l+4)] = int32(roundPowerOfTwo(sum, reduceBitsHoriz))
			sx += alpha
		}
	}
	return sy4
}

func warpHorizontalHighBDResident(tmp *[warpedIntermediateRows * warpedIntermediateColumns]int32, ref frame.Plane, ix4 int, sx4 int, iy4 int, sy4 int, alpha int, beta int, reduceBitsHoriz int, offsetBitsHoriz int) int {
	for k := -7; k < 8; k++ {
		row := (iy4 + k) * ref.Stride
		sx := sx4 + beta*(k+4)
		for l := -4; l < 4; l++ {
			ix := ix4 + l - 3
			offs := roundPowerOfTwo(sx, warpedDiffPrecBits) + warpedPixelPrecShifts
			if offs < 0 || offs >= len(warpedFilter) {
				offs = clampInt(offs, 0, len(warpedFilter)-1)
			}
			coeffs := warpedFilter[offs]
			base := row + ix*2
			sum := 1 << offsetBitsHoriz
			sum += int(uint16(ref.Pix[base+0])|uint16(ref.Pix[base+1])<<8) * int(coeffs[0])
			sum += int(uint16(ref.Pix[base+2])|uint16(ref.Pix[base+3])<<8) * int(coeffs[1])
			sum += int(uint16(ref.Pix[base+4])|uint16(ref.Pix[base+5])<<8) * int(coeffs[2])
			sum += int(uint16(ref.Pix[base+6])|uint16(ref.Pix[base+7])<<8) * int(coeffs[3])
			sum += int(uint16(ref.Pix[base+8])|uint16(ref.Pix[base+9])<<8) * int(coeffs[4])
			sum += int(uint16(ref.Pix[base+10])|uint16(ref.Pix[base+11])<<8) * int(coeffs[5])
			sum += int(uint16(ref.Pix[base+12])|uint16(ref.Pix[base+13])<<8) * int(coeffs[6])
			sum += int(uint16(ref.Pix[base+14])|uint16(ref.Pix[base+15])<<8) * int(coeffs[7])
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
