// Ported from libaom:
//   av1/common/convolve.c (av1_convolve_2d_scale_c,
//                          av1_highbd_convolve_2d_scale_c)
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

package motion

import "github.com/thesyncim/goav1/internal/av1/frame"

// scaledIMMaxHeight bounds the per-block intermediate row count consumed by
// the scaled 2D 8-tap convolver. AV1 caps downscaling at 2x (libaom: max
// y_step = 2 * ScaleSubpelScale), so the worst case is
//
//	im_h = (((h-1)*y_step + subpel_y) >> ScaleSubpelBits) + filterTaps
//	     <= ((h-1)*2*ScaleSubpelScale + ScaleSubpelMask) >> ScaleSubpelBits + 8
//	     <= 2*(h-1) + 1 + 8
//
// for h = maxBlockSize. Round up to a stable compile-time bound.
const scaledIMMaxHeight = 2*maxBlockSize + filterTaps

// SubpelKernelTable is a complete sub-pel filter table for one axis of the
// scaled 8-tap convolver. Index k is the kernel for Q4 phase k.
type SubpelKernelTable [1 << subpelQ4Bits][filterTaps]int16

// SubpelKernelTableFor returns the AV1 sub-pel kernel table for filter at the
// requested block-size axis. Smaller blocks select libaom's 4-tap-equivalent
// table; everything 8 and up uses the standard 8-tap table.
func SubpelKernelTableFor(filter InterpFilter, blockSize int) (SubpelKernelTable, error) {
	if !filter.Valid() {
		return SubpelKernelTable{}, ErrInvalidMotion
	}
	var table SubpelKernelTable
	for k := range len(table) {
		kern, err := interpKernel(filter, blockSize, k)
		if err != nil {
			return SubpelKernelTable{}, err
		}
		table[k] = kern
	}
	return table, nil
}

// ConvolveScale2D8 performs AV1 scaled 8-bit 8-tap 2D interpolation on a
// rectangular block. dst is the output plane; (dstX, dstY) is the top-left
// destination sample. ref is the reference plane.
//
// (startX, startY) is the absolute Q10 source position of the first output
// sample (see ScaleFactors.ScaledBlockOrigin), and (xStep, yStep) the per-
// output Q10 stride through the reference plane. The caller supplies the full
// Q4 sub-pel kernel tables for both axes; the scaled stepper indexes into
// them on every output sample.
//
// Source tap reads must lie within ref. Use the *Clamped variant when block
// taps may step outside the reference plane.
func ConvolveScale2D8(dst frame.Plane, ref frame.Plane, dstX int, dstY int, width int, height int,
	startX int64, xStep int64, startY int64, yStep int64,
	xTable SubpelKernelTable, yTable SubpelKernelTable) error {
	if width <= 0 || height <= 0 || width > maxBlockSize || height > maxBlockSize {
		return ErrInvalidMotion
	}
	if xStep <= 0 || yStep <= 0 {
		return ErrInvalidMotion
	}
	imH, ok := scaledIMHeight(height, startY, yStep)
	if !ok {
		return ErrInvalidMotion
	}
	if !planeRegionFits(dst, 1, dstX, dstY, width, height) {
		return ErrInvalidMotion
	}
	if !planeRegionFits(ref, 1, 0, 0, ref.Width, ref.Height) {
		return ErrInvalidMotion
	}
	if !scaledRefRegionFits(ref, width, imH, startX, xStep, startY) {
		return ErrInvalidMotion
	}
	convolveScale2D8(dst, ref, dstX, dstY, width, height, startX, xStep, startY, yStep, xTable, yTable, imH)
	return nil
}

// ConvolveScale2D8Clamped is ConvolveScale2D8 but replicates the reference
// plane at its edges when a tap falls outside the plane.
func ConvolveScale2D8Clamped(dst frame.Plane, ref frame.Plane, dstX int, dstY int, width int, height int,
	startX int64, xStep int64, startY int64, yStep int64,
	xTable SubpelKernelTable, yTable SubpelKernelTable) error {
	if width <= 0 || height <= 0 || width > maxBlockSize || height > maxBlockSize {
		return ErrInvalidMotion
	}
	if xStep <= 0 || yStep <= 0 {
		return ErrInvalidMotion
	}
	imH, ok := scaledIMHeight(height, startY, yStep)
	if !ok {
		return ErrInvalidMotion
	}
	if !planeRegionFits(dst, 1, dstX, dstY, width, height) {
		return ErrInvalidMotion
	}
	if !planeRegionFits(ref, 1, 0, 0, ref.Width, ref.Height) {
		return ErrInvalidMotion
	}
	if scaledRefRegionFits(ref, width, imH, startX, xStep, startY) {
		convolveScale2D8(dst, ref, dstX, dstY, width, height, startX, xStep, startY, yStep, xTable, yTable, imH)
		return nil
	}
	convolveScale2D8Clamped(dst, ref, dstX, dstY, width, height, startX, xStep, startY, yStep, xTable, yTable, imH)
	return nil
}

// ConvolveScale2DHighBD is the 10/12-bit variant of ConvolveScale2D8.
func ConvolveScale2DHighBD(dst frame.Plane, ref frame.Plane, bitDepth uint8, dstX int, dstY int, width int, height int,
	startX int64, xStep int64, startY int64, yStep int64,
	xTable SubpelKernelTable, yTable SubpelKernelTable) error {
	return ConvolveScale2DHighBDWithScratch(dst, ref, bitDepth, dstX, dstY, width, height,
		startX, xStep, startY, yStep, xTable, yTable, nil)
}

// ConvolveScale2DHighBDWithScratch is ConvolveScale2DHighBD using optional
// caller-owned scratch for the large intermediate block.
func ConvolveScale2DHighBDWithScratch(dst frame.Plane, ref frame.Plane, bitDepth uint8, dstX int, dstY int, width int, height int,
	startX int64, xStep int64, startY int64, yStep int64,
	xTable SubpelKernelTable, yTable SubpelKernelTable, scratch *ScaledConvolveScratch) error {
	max, ok := highBDMax(bitDepth)
	if !ok || width <= 0 || height <= 0 || width > maxBlockSize || height > maxBlockSize {
		return ErrInvalidMotion
	}
	if xStep <= 0 || yStep <= 0 {
		return ErrInvalidMotion
	}
	imH, ok := scaledIMHeight(height, startY, yStep)
	if !ok {
		return ErrInvalidMotion
	}
	if !planeRegionFits(dst, 2, dstX, dstY, width, height) {
		return ErrInvalidMotion
	}
	if !planeRegionFits(ref, 2, 0, 0, ref.Width, ref.Height) {
		return ErrInvalidMotion
	}
	if !scaledRefRegionFits(ref, width, imH, startX, xStep, startY) {
		return ErrInvalidMotion
	}
	im, pooled := scaledHighBDIMForScratch(scratch)
	convolveScale2DHighBD(dst, ref, bitDepth, max, dstX, dstY, width, height, startX, xStep, startY, yStep, xTable, yTable, imH, im)
	putScaledHighBDIM(im, pooled)
	return nil
}

// ConvolveScale2DHighBDClamped is ConvolveScale2DHighBD but replicates the
// reference plane at its edges when a tap falls outside the plane.
func ConvolveScale2DHighBDClamped(dst frame.Plane, ref frame.Plane, bitDepth uint8, dstX int, dstY int, width int, height int,
	startX int64, xStep int64, startY int64, yStep int64,
	xTable SubpelKernelTable, yTable SubpelKernelTable) error {
	return ConvolveScale2DHighBDClampedWithScratch(dst, ref, bitDepth, dstX, dstY, width, height,
		startX, xStep, startY, yStep, xTable, yTable, nil)
}

// ConvolveScale2DHighBDClampedWithScratch is ConvolveScale2DHighBDClamped using
// optional caller-owned scratch for the large intermediate block.
func ConvolveScale2DHighBDClampedWithScratch(dst frame.Plane, ref frame.Plane, bitDepth uint8, dstX int, dstY int, width int, height int,
	startX int64, xStep int64, startY int64, yStep int64,
	xTable SubpelKernelTable, yTable SubpelKernelTable, scratch *ScaledConvolveScratch) error {
	max, ok := highBDMax(bitDepth)
	if !ok || width <= 0 || height <= 0 || width > maxBlockSize || height > maxBlockSize {
		return ErrInvalidMotion
	}
	if xStep <= 0 || yStep <= 0 {
		return ErrInvalidMotion
	}
	imH, ok := scaledIMHeight(height, startY, yStep)
	if !ok {
		return ErrInvalidMotion
	}
	if !planeRegionFits(dst, 2, dstX, dstY, width, height) {
		return ErrInvalidMotion
	}
	if !planeRegionFits(ref, 2, 0, 0, ref.Width, ref.Height) {
		return ErrInvalidMotion
	}
	im, pooled := scaledHighBDIMForScratch(scratch)
	if scaledRefRegionFits(ref, width, imH, startX, xStep, startY) {
		convolveScale2DHighBD(dst, ref, bitDepth, max, dstX, dstY, width, height, startX, xStep, startY, yStep, xTable, yTable, imH, im)
	} else {
		convolveScale2DHighBDClamped(dst, ref, bitDepth, max, dstX, dstY, width, height, startX, xStep, startY, yStep, xTable, yTable, imH, im)
	}
	putScaledHighBDIM(im, pooled)
	return nil
}

// scaledIMHeight computes libaom's intermediate-buffer row count for the
// scaled 2D 8-tap convolver, expressed in im_block rows (== libaom's im_h).
//
// im_h = ((subpel_y + (h-1)*y_step) >> SCALE_SUBPEL_BITS) + taps,
// where subpel_y is the Q10 fractional part of startY.
func scaledIMHeight(height int, startY int64, yStep int64) (int, bool) {
	if height <= 0 || yStep <= 0 {
		return 0, false
	}
	subpelY := scaledSubpel(startY)
	last := subpelY + int64(height-1)*yStep
	if last < 0 {
		return 0, false
	}
	im := int(last>>ScaleSubpelBits) + filterTaps
	if im <= 0 || im > scaledIMMaxHeight {
		return 0, false
	}
	return im, true
}

// scaledSubpel returns the libaom Q10 fractional component of a Q10 scaled
// position. For non-negative values this is `pos & ScaleSubpelMask`, matching
// SubpelParams.subpel_x. For negative positions, libaom's
// dec_calc_subpel_params clamps pos to a non-negative range before masking, so
// callers feeding scaled-prediction kernels should normally see
// non-negative startY here.
func scaledSubpel(pos int64) int64 {
	r := pos & ScaleSubpelMask
	if r < 0 {
		r += ScaleSubpelScale
	}
	return r
}

// scaledIntFloor returns ⌊pos / ScaleSubpelScale⌋ matching libaom's
// `pos >> SCALE_SUBPEL_BITS` arithmetic shift.
func scaledIntFloor(pos int64) int64 {
	return pos >> ScaleSubpelBits
}

// scaledRefRegionFits checks that every tap read by the scaled 2D convolver
// stays inside ref. fo = filterTaps/2 - 1 is the leading half of each kernel.
//
// Horizontal pass reads source columns [intX - fo, intX - fo + 7] for each
// output sample x, where intX = (startX + x*xStep) >> SCALE_SUBPEL_BITS.
//
// Vertical pass uses im rows [0, imH); the source row of im row 0 is
// (startY >> SCALE_SUBPEL_BITS) - fo (libaom's src_horiz = src - fo*stride).
func scaledRefRegionFits(ref frame.Plane, width int, imH int,
	startX int64, xStep int64, startY int64) bool {
	const fo = filterTaps/2 - 1
	if width <= 0 || imH <= 0 {
		return false
	}
	// Horizontal taps. Compute min/max of (startX + x*xStep) >> SCALE_SUBPEL_BITS
	// using monotonic positivity of xStep.
	minCol := int(scaledIntFloor(startX))
	maxCol := int(scaledIntFloor(startX + int64(width-1)*xStep))
	leftTap := minCol - fo
	rightTap := maxCol - fo + filterTaps - 1
	if leftTap < 0 || rightTap >= ref.Width {
		return false
	}
	// Vertical extent (source plane row indices).
	startRow := int(scaledIntFloor(startY)) - fo
	endRow := startRow + imH - 1
	if startRow < 0 || endRow >= ref.Height {
		return false
	}
	return true
}

func convolveScale2D8(dst frame.Plane, ref frame.Plane, dstX int, dstY int, width int, height int,
	startX int64, xStep int64, startY int64, yStep int64,
	xTable SubpelKernelTable, yTable SubpelKernelTable, imH int) {
	const imStride = maxBlockSize
	var im [scaledIMMaxHeight * maxBlockSize]int16
	foX := filterTaps/2 - 1
	foY := filterTaps/2 - 1
	startRow := int(scaledIntFloor(startY)) - foY
	for y := range imH {
		srcRow := startRow + y
		base := srcRow * ref.Stride
		xPos := startX
		for x := range width {
			xInt := int(scaledIntFloor(xPos)) - foX
			xFilterIdx := int(scaledSubpel(xPos) >> ScaleExtraBits)
			kernel := xTable[xFilterIdx]
			sum := 1 << (8 + filterBits - 1)
			off := base + xInt
			for k := range filterTaps {
				sum += int(kernel[k]) * int(ref.Pix[off+k])
			}
			im[y*imStride+x] = int16(roundPowerOfTwo(sum, round0Bits))
			xPos += xStep
		}
	}
	offsetBits := 8 + 2*filterBits - round0Bits
	roundOffset := (1 << (offsetBits - round1Bits)) + (1 << (offsetBits - round1Bits - 1))
	bits := 2*filterBits - round0Bits - round1Bits
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
			res := roundPowerOfTwo(sum, round1Bits) - roundOffset
			dst.Pix[(dstY+y)*dst.Stride+dstX+x] = byte(clipPixel(roundPowerOfTwo(res, bits)))
			yPos += yStep
		}
	}
}

func convolveScale2D8Clamped(dst frame.Plane, ref frame.Plane, dstX int, dstY int, width int, height int,
	startX int64, xStep int64, startY int64, yStep int64,
	xTable SubpelKernelTable, yTable SubpelKernelTable, imH int) {
	const imStride = maxBlockSize
	var im [scaledIMMaxHeight * maxBlockSize]int16
	foX := filterTaps/2 - 1
	foY := filterTaps/2 - 1
	startRow := int(scaledIntFloor(startY)) - foY
	for y := range imH {
		srcRow := startRow + y
		xPos := startX
		for x := range width {
			xInt := int(scaledIntFloor(xPos)) - foX
			xFilterIdx := int(scaledSubpel(xPos) >> ScaleExtraBits)
			kernel := xTable[xFilterIdx]
			sum := 1 << (8 + filterBits - 1)
			for k := range filterTaps {
				sum += int(kernel[k]) * int(loadSample8Clamped(ref, xInt+k, srcRow))
			}
			im[y*imStride+x] = int16(roundPowerOfTwo(sum, round0Bits))
			xPos += xStep
		}
	}
	offsetBits := 8 + 2*filterBits - round0Bits
	roundOffset := (1 << (offsetBits - round1Bits)) + (1 << (offsetBits - round1Bits - 1))
	bits := 2*filterBits - round0Bits - round1Bits
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
			res := roundPowerOfTwo(sum, round1Bits) - roundOffset
			dst.Pix[(dstY+y)*dst.Stride+dstX+x] = byte(clipPixel(roundPowerOfTwo(res, bits)))
			yPos += yStep
		}
	}
}

func convolveScale2DHighBD(dst frame.Plane, ref frame.Plane, bitDepth uint8, max uint16, dstX int, dstY int, width int, height int,
	startX int64, xStep int64, startY int64, yStep int64,
	xTable SubpelKernelTable, yTable SubpelKernelTable, imH int, im *scaledHighBDIM) {
	const imStride = maxBlockSize
	foX := filterTaps/2 - 1
	foY := filterTaps/2 - 1
	startRow := int(scaledIntFloor(startY)) - foY
	for y := range imH {
		srcRow := startRow + y
		xPos := startX
		for x := range width {
			xInt := int(scaledIntFloor(xPos)) - foX
			xFilterIdx := int(scaledSubpel(xPos) >> ScaleExtraBits)
			kernel := xTable[xFilterIdx]
			sum := 1 << (int(bitDepth) + filterBits - 1)
			for k := range filterTaps {
				sum += int(kernel[k]) * int(loadHighBDSample(ref, xInt+k, srcRow))
			}
			im[y*imStride+x] = int32(roundPowerOfTwo(sum, round0Bits))
			xPos += xStep
		}
	}
	offsetBits := int(bitDepth) + 2*filterBits - round0Bits
	roundOffset := (1 << (offsetBits - round1Bits)) + (1 << (offsetBits - round1Bits - 1))
	bits := 2*filterBits - round0Bits - round1Bits
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
			res := roundPowerOfTwo(sum, round1Bits) - roundOffset
			storeHighBDSample(dst, dstX+x, dstY+y, clipPixelHighBD(roundPowerOfTwo(res, bits), max))
			yPos += yStep
		}
	}
}

func convolveScale2DHighBDClamped(dst frame.Plane, ref frame.Plane, bitDepth uint8, max uint16, dstX int, dstY int, width int, height int,
	startX int64, xStep int64, startY int64, yStep int64,
	xTable SubpelKernelTable, yTable SubpelKernelTable, imH int, im *scaledHighBDIM) {
	const imStride = maxBlockSize
	foX := filterTaps/2 - 1
	foY := filterTaps/2 - 1
	startRow := int(scaledIntFloor(startY)) - foY
	for y := range imH {
		srcRow := startRow + y
		xPos := startX
		for x := range width {
			xInt := int(scaledIntFloor(xPos)) - foX
			xFilterIdx := int(scaledSubpel(xPos) >> ScaleExtraBits)
			kernel := xTable[xFilterIdx]
			sum := 1 << (int(bitDepth) + filterBits - 1)
			for k := range filterTaps {
				sum += int(kernel[k]) * int(loadHighBDSampleClamped(ref, xInt+k, srcRow))
			}
			im[y*imStride+x] = int32(roundPowerOfTwo(sum, round0Bits))
			xPos += xStep
		}
	}
	offsetBits := int(bitDepth) + 2*filterBits - round0Bits
	roundOffset := (1 << (offsetBits - round1Bits)) + (1 << (offsetBits - round1Bits - 1))
	bits := 2*filterBits - round0Bits - round1Bits
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
			res := roundPowerOfTwo(sum, round1Bits) - roundOffset
			storeHighBDSample(dst, dstX+x, dstY+y, clipPixelHighBD(roundPowerOfTwo(res, bits), max))
			yPos += yStep
		}
	}
}
