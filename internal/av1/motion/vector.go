// Ported from libaom:
//   av1/common/reconinter.c
//   aom_dsp/aom_convolve.c
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

package motion

import (
	"github.com/thesyncim/goav1/internal/av1/dsp"
	"github.com/thesyncim/goav1/internal/av1/frame"
)

const (
	// SubpelBits is the AV1 motion-vector fractional precision in luma sample
	// units. A value of 8 represents one full-sample offset.
	SubpelBits  = 3
	SubpelScale = 1 << SubpelBits
)

// Vector stores an AV1-style motion vector as row/column offsets in eighth
// sample units.
type Vector struct {
	Row int32
	Col int32
}

// FullpelVector converts a full-sample column/row offset into a subpel motion
// vector.
func FullpelVector(col int, row int) (Vector, error) {
	scaledCol, ok := scaleFullpel(col)
	if !ok {
		return Vector{}, ErrInvalidMotion
	}
	scaledRow, ok := scaleFullpel(row)
	if !ok {
		return Vector{}, ErrInvalidMotion
	}
	return Vector{Row: scaledRow, Col: scaledCol}, nil
}

// IsFullpel reports whether v can be applied without subpel interpolation.
func (v Vector) IsFullpel() bool {
	return v.Row&(SubpelScale-1) == 0 && v.Col&(SubpelScale-1) == 0
}

// LowerPrecision applies libaom's lower_mv_precision() rule to v.
func LowerPrecision(v Vector, allowHighPrecision bool, forceInteger bool) Vector {
	if forceInteger {
		return Vector{
			Row: lowerIntegerPrecision(v.Row),
			Col: lowerIntegerPrecision(v.Col),
		}
	}
	if !allowHighPrecision {
		if v.Row&1 != 0 {
			if v.Row > 0 {
				v.Row--
			} else {
				v.Row++
			}
		}
		if v.Col&1 != 0 {
			if v.Col > 0 {
				v.Col--
			} else {
				v.Col++
			}
		}
	}
	return v
}

// FullpelOffset returns v's full-sample column/row offset. Fractional vectors
// are rejected until interpolation filters are wired in.
func (v Vector) FullpelOffset() (int, int, error) {
	if !v.IsFullpel() {
		return 0, 0, ErrInvalidMotion
	}
	return int(v.Col >> SubpelBits), int(v.Row >> SubpelBits), nil
}

// FullpelReferenceOrigin returns the reference-plane origin for a block whose
// output-plane origin is (dstX, dstY). The vector must be full-pixel aligned.
func FullpelReferenceOrigin(dstX int, dstY int, mv Vector) (int, int, error) {
	dx, dy, err := mv.FullpelOffset()
	if err != nil {
		return 0, 0, err
	}
	refX, ok := checkedAdd(dstX, dx)
	if !ok {
		return 0, 0, ErrInvalidMotion
	}
	refY, ok := checkedAdd(dstY, dy)
	if !ok {
		return 0, 0, ErrInvalidMotion
	}
	return refX, refY, nil
}

// ReferenceOrigin returns the integer reference-plane origin plus AV1 Q4
// subpel offsets for a block whose output-plane origin is (dstX, dstY).
func ReferenceOrigin(dstX int, dstY int, mv Vector) (refX int, refY int, subX int, subY int, err error) {
	dx := int(mv.Col >> SubpelBits)
	dy := int(mv.Row >> SubpelBits)
	refX, ok := checkedAdd(dstX, dx)
	if !ok {
		return 0, 0, 0, 0, ErrInvalidMotion
	}
	refY, ok = checkedAdd(dstY, dy)
	if !ok {
		return 0, 0, 0, 0, ErrInvalidMotion
	}
	subX = int(mv.Col&(SubpelScale-1)) << 1
	subY = int(mv.Row&(SubpelScale-1)) << 1
	return refX, refY, subX, subY, nil
}

// ReferenceOriginSubsampled returns the reference-plane origin and AV1 Q4
// subpel offsets for a plane that may be chroma-subsampled. This ports
// libaom's init_subpel_params() position rule: luma uses mv*2 to convert Q3
// motion vectors to Q4 filter offsets, while a subsampled chroma axis uses mv
// directly in that plane's Q4 units.
func ReferenceOriginSubsampled(dstX int, dstY int, mv Vector, subsamplingX bool, subsamplingY bool) (refX int, refY int, subX int, subY int, err error) {
	scaleX := int64(2)
	if subsamplingX {
		scaleX = 1
	}
	scaleY := int64(2)
	if subsamplingY {
		scaleY = 1
	}
	refX, subX, err = referenceOriginQ4(dstX, int64(mv.Col)*scaleX)
	if err != nil {
		return 0, 0, 0, 0, err
	}
	refY, subY, err = referenceOriginQ4(dstY, int64(mv.Row)*scaleY)
	if err != nil {
		return 0, 0, 0, 0, err
	}
	return refX, refY, subX, subY, nil
}

// PredictInterPlaneBlock predicts a block using AV1's regular translational
// interpolation filter for fractional vectors.
func PredictInterPlaneBlock(dst frame.Plane, ref frame.Plane, bytesPerSample int, dstX int, dstY int, width int, height int, mv Vector) error {
	return PredictInterPlaneBlockWithFilter(dst, ref, bytesPerSample, dstX, dstY, width, height, mv, RegularFilters)
}

// PredictInterPlaneBlockBitDepth predicts a block with explicit bit depth.
// Use this entry point for high-bit-depth fractional vectors so clipping can
// distinguish 10-bit from 12-bit output.
func PredictInterPlaneBlockBitDepth(dst frame.Plane, ref frame.Plane, bytesPerSample int, bitDepth uint8, dstX int, dstY int, width int, height int, mv Vector) error {
	return PredictInterPlaneBlockWithFilterBitDepth(dst, ref, bytesPerSample, bitDepth, dstX, dstY, width, height, mv, RegularFilters)
}

// PredictInterPlaneBlockWithFilter predicts a translational single-reference
// inter block. Low-bit-depth fractional vectors use libaom's av1_convolve_*_sr_c
// filter path. High-bit-depth fractional callers should use
// PredictInterPlaneBlockWithFilterBitDepth.
func PredictInterPlaneBlockWithFilter(dst frame.Plane, ref frame.Plane, bytesPerSample int, dstX int, dstY int, width int, height int, mv Vector, filters InterpFilters) error {
	return predictInterPlaneBlockWithFilter(dst, ref, bytesPerSample, 8, false, dstX, dstY, width, height, mv, filters)
}

// PredictInterPlaneBlockWithFilterBitDepth predicts a translational
// single-reference inter block with explicit clipping bit depth.
func PredictInterPlaneBlockWithFilterBitDepth(dst frame.Plane, ref frame.Plane, bytesPerSample int, bitDepth uint8, dstX int, dstY int, width int, height int, mv Vector, filters InterpFilters) error {
	return predictInterPlaneBlockWithFilter(dst, ref, bytesPerSample, bitDepth, true, dstX, dstY, width, height, mv, filters)
}

func predictInterPlaneBlockWithFilter(dst frame.Plane, ref frame.Plane, bytesPerSample int, bitDepth uint8, explicitBitDepth bool, dstX int, dstY int, width int, height int, mv Vector, filters InterpFilters) error {
	refX, refY, subX, subY, err := referenceOrigin(dstX, dstY, mv)
	if err != nil {
		return err
	}
	return predictInterPlaneBlockFromOriginWithFilter(dst, ref, bytesPerSample, bitDepth, explicitBitDepth, dstX, dstY, refX, refY, width, height, subX, subY, filters)
}

// PredictInterPlaneBlockFromOrigin predicts a translational inter block from an
// already-resolved reference origin and AV1 Q4 subpel offsets. This is useful
// for compound and scaled-reference paths that predict into caller-owned
// scratch buffers whose destination origin differs from the current frame.
func PredictInterPlaneBlockFromOrigin(dst frame.Plane, ref frame.Plane, bytesPerSample int, dstX int, dstY int, refX int, refY int, width int, height int, subX int, subY int) error {
	return PredictInterPlaneBlockFromOriginWithFilter(dst, ref, bytesPerSample, dstX, dstY, refX, refY, width, height, subX, subY, RegularFilters)
}

// PredictInterPlaneBlockFromOriginBitDepth is PredictInterPlaneBlockFromOrigin
// with explicit high-bit-depth clipping.
func PredictInterPlaneBlockFromOriginBitDepth(dst frame.Plane, ref frame.Plane, bytesPerSample int, bitDepth uint8, dstX int, dstY int, refX int, refY int, width int, height int, subX int, subY int) error {
	return PredictInterPlaneBlockFromOriginWithFilterBitDepth(dst, ref, bytesPerSample, bitDepth, dstX, dstY, refX, refY, width, height, subX, subY, RegularFilters)
}

// PredictInterPlaneBlockFromOriginWithFilter is PredictInterPlaneBlockFromOrigin
// with explicit interpolation filters.
func PredictInterPlaneBlockFromOriginWithFilter(dst frame.Plane, ref frame.Plane, bytesPerSample int, dstX int, dstY int, refX int, refY int, width int, height int, subX int, subY int, filters InterpFilters) error {
	return predictInterPlaneBlockFromOriginWithFilter(dst, ref, bytesPerSample, 8, false, dstX, dstY, refX, refY, width, height, subX, subY, filters)
}

// PredictInterPlaneBlockFromOriginWithFilterBitDepth is
// PredictInterPlaneBlockFromOriginWithFilter with explicit high-bit-depth
// clipping.
func PredictInterPlaneBlockFromOriginWithFilterBitDepth(dst frame.Plane, ref frame.Plane, bytesPerSample int, bitDepth uint8, dstX int, dstY int, refX int, refY int, width int, height int, subX int, subY int, filters InterpFilters) error {
	return predictInterPlaneBlockFromOriginWithFilter(dst, ref, bytesPerSample, bitDepth, true, dstX, dstY, refX, refY, width, height, subX, subY, filters)
}

func predictInterPlaneBlockFromOriginWithFilter(dst frame.Plane, ref frame.Plane, bytesPerSample int, bitDepth uint8, explicitBitDepth bool, dstX int, dstY int, refX int, refY int, width int, height int, subX int, subY int, filters InterpFilters) error {
	if !filters.X.Valid() || !filters.Y.Valid() {
		return ErrInvalidMotion
	}
	if subX < 0 || subX > subpelQ4Mask || subY < 0 || subY > subpelQ4Mask {
		return ErrInvalidMotion
	}
	if explicitBitDepth && !bitDepthMatchesSampleWidth(bytesPerSample, bitDepth) {
		return ErrInvalidMotion
	}
	if subX == 0 && subY == 0 {
		if planeRegionFits(ref, bytesPerSample, refX, refY, width, height) {
			if err := dsp.CopyPlaneBlock(dst, ref, bytesPerSample, dstX, dstY, refX, refY, width, height); err != nil {
				return ErrInvalidMotion
			}
			return nil
		}
		return copyPlaneBlockClamped(dst, ref, bytesPerSample, dstX, dstY, refX, refY, width, height)
	}
	if bytesPerSample != 1 {
		if bytesPerSample == 2 && explicitBitDepth {
			if err := predictInterPlaneBlockHighBD(dst, ref, bitDepth, dstX, dstY, refX, refY, width, height, subX, subY, filters); err != nil {
				return ErrInvalidMotion
			}
			return nil
		}
		return ErrInvalidMotion
	}
	if err := predictInterPlaneBlock8(dst, ref, dstX, dstY, refX, refY, width, height, subX, subY, filters); err != nil {
		return ErrInvalidMotion
	}
	return nil
}

func referenceOrigin(dstX int, dstY int, mv Vector) (refX int, refY int, subX int, subY int, err error) {
	return ReferenceOrigin(dstX, dstY, mv)
}

func referenceOriginQ4(dst int, mvQ4 int64) (int, int, error) {
	pos := int64(dst)*16 + mvQ4
	ref := pos >> 4
	if ref < int64(minInt) || ref > int64(maxInt) {
		return 0, 0, ErrInvalidMotion
	}
	return int(ref), int(pos & 15), nil
}

func predictInterPlaneBlock8(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, subX int, subY int, filters InterpFilters) error {
	if width <= 0 || height <= 0 || width > maxBlockSize || height > maxBlockSize {
		return ErrInvalidMotion
	}
	if !planeRegionFits(dst, 1, dstX, dstY, width, height) {
		return ErrInvalidMotion
	}
	if !planeRegionFits(ref, 1, 0, 0, ref.Width, ref.Height) {
		return ErrInvalidMotion
	}
	xKernel, err := interpKernel(filters.X, width, subX)
	if err != nil {
		return err
	}
	yKernel, err := interpKernel(filters.Y, height, subY)
	if err != nil {
		return err
	}
	foX := filterTaps/2 - 1
	foY := filterTaps/2 - 1
	switch {
	case subX != 0 && subY != 0:
		if planeRegionFits(ref, 1, refX-foX, refY-foY, width+filterTaps-1, height+filterTaps-1) {
			convolve2D8(dst, ref, dstX, dstY, refX, refY, width, height, xKernel, yKernel)
		} else {
			convolve2D8Clamped(dst, ref, dstX, dstY, refX, refY, width, height, xKernel, yKernel)
		}
	case subX != 0:
		if planeRegionFits(ref, 1, refX-foX, refY, width+filterTaps-1, height) {
			convolveX8(dst, ref, dstX, dstY, refX, refY, width, height, xKernel)
		} else {
			convolveX8Clamped(dst, ref, dstX, dstY, refX, refY, width, height, xKernel)
		}
	case subY != 0:
		if planeRegionFits(ref, 1, refX, refY-foY, width, height+filterTaps-1) {
			convolveY8(dst, ref, dstX, dstY, refX, refY, width, height, yKernel)
		} else {
			convolveY8Clamped(dst, ref, dstX, dstY, refX, refY, width, height, yKernel)
		}
	}
	return nil
}

func predictInterPlaneBlockHighBD(dst frame.Plane, ref frame.Plane, bitDepth uint8, dstX int, dstY int, refX int, refY int, width int, height int, subX int, subY int, filters InterpFilters) error {
	max, ok := highBDMax(bitDepth)
	if !ok || width <= 0 || height <= 0 || width > maxBlockSize || height > maxBlockSize {
		return ErrInvalidMotion
	}
	if !planeRegionFits(dst, 2, dstX, dstY, width, height) {
		return ErrInvalidMotion
	}
	if !planeRegionFits(ref, 2, 0, 0, ref.Width, ref.Height) {
		return ErrInvalidMotion
	}
	xKernel, err := interpKernel(filters.X, width, subX)
	if err != nil {
		return err
	}
	yKernel, err := interpKernel(filters.Y, height, subY)
	if err != nil {
		return err
	}
	foX := filterTaps/2 - 1
	foY := filterTaps/2 - 1
	switch {
	case subX != 0 && subY != 0:
		if planeRegionFits(ref, 2, refX-foX, refY-foY, width+filterTaps-1, height+filterTaps-1) {
			convolve2DHighBD(dst, ref, bitDepth, max, dstX, dstY, refX, refY, width, height, xKernel, yKernel)
		} else {
			convolve2DHighBDClamped(dst, ref, bitDepth, max, dstX, dstY, refX, refY, width, height, xKernel, yKernel)
		}
	case subX != 0:
		if planeRegionFits(ref, 2, refX-foX, refY, width+filterTaps-1, height) {
			convolveXHighBD(dst, ref, max, dstX, dstY, refX, refY, width, height, xKernel)
		} else {
			convolveXHighBDClamped(dst, ref, max, dstX, dstY, refX, refY, width, height, xKernel)
		}
	case subY != 0:
		if planeRegionFits(ref, 2, refX, refY-foY, width, height+filterTaps-1) {
			convolveYHighBD(dst, ref, max, dstX, dstY, refX, refY, width, height, yKernel)
		} else {
			convolveYHighBDClamped(dst, ref, max, dstX, dstY, refX, refY, width, height, yKernel)
		}
	}
	return nil
}

func copyPlaneBlockClamped(dst frame.Plane, ref frame.Plane, bytesPerSample int, dstX int, dstY int, refX int, refY int, width int, height int) error {
	if !planeRegionFits(dst, bytesPerSample, dstX, dstY, width, height) ||
		!planeRegionFits(ref, bytesPerSample, 0, 0, ref.Width, ref.Height) {
		return ErrInvalidMotion
	}
	for y := range height {
		sy := clampInt(refY+y, 0, ref.Height-1)
		for x := range width {
			sx := clampInt(refX+x, 0, ref.Width-1)
			switch bytesPerSample {
			case 1:
				dst.Pix[(dstY+y)*dst.Stride+dstX+x] = ref.Pix[sy*ref.Stride+sx]
			case 2:
				dstOffset := (dstY+y)*dst.Stride + (dstX+x)*2
				refOffset := sy*ref.Stride + sx*2
				dst.Pix[dstOffset] = ref.Pix[refOffset]
				dst.Pix[dstOffset+1] = ref.Pix[refOffset+1]
			default:
				return ErrInvalidMotion
			}
		}
	}
	return nil
}

func convolveX8(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, kernel [filterTaps]int16) {
	fo := filterTaps/2 - 1
	for y := range height {
		for x := range width {
			sum := 0
			for k := range filterTaps {
				sum += int(kernel[k]) * int(ref.Pix[(refY+y)*ref.Stride+refX+x-fo+k])
			}
			res := roundPowerOfTwo(sum, round0Bits)
			dst.Pix[(dstY+y)*dst.Stride+dstX+x] = byte(clipPixel(roundPowerOfTwo(res, filterBits-round0Bits)))
		}
	}
}

func convolveX8Clamped(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, kernel [filterTaps]int16) {
	fo := filterTaps/2 - 1
	for y := range height {
		for x := range width {
			sum := 0
			for k := range filterTaps {
				sum += int(kernel[k]) * int(loadSample8Clamped(ref, refX+x-fo+k, refY+y))
			}
			res := roundPowerOfTwo(sum, round0Bits)
			dst.Pix[(dstY+y)*dst.Stride+dstX+x] = byte(clipPixel(roundPowerOfTwo(res, filterBits-round0Bits)))
		}
	}
}

func convolveY8(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, kernel [filterTaps]int16) {
	fo := filterTaps/2 - 1
	for y := range height {
		for x := range width {
			sum := 0
			for k := range filterTaps {
				sum += int(kernel[k]) * int(ref.Pix[(refY+y-fo+k)*ref.Stride+refX+x])
			}
			dst.Pix[(dstY+y)*dst.Stride+dstX+x] = byte(clipPixel(roundPowerOfTwo(sum, filterBits)))
		}
	}
}

func convolveY8Clamped(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, kernel [filterTaps]int16) {
	fo := filterTaps/2 - 1
	for y := range height {
		for x := range width {
			sum := 0
			for k := range filterTaps {
				sum += int(kernel[k]) * int(loadSample8Clamped(ref, refX+x, refY+y-fo+k))
			}
			dst.Pix[(dstY+y)*dst.Stride+dstX+x] = byte(clipPixel(roundPowerOfTwo(sum, filterBits)))
		}
	}
}

func convolve2D8(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16) {
	const imStride = maxBlockSize
	var im [((maxBlockSize + filterTaps - 1) * maxBlockSize)]int16
	foX := filterTaps/2 - 1
	foY := filterTaps/2 - 1
	imH := height + filterTaps - 1
	for y := range imH {
		for x := range width {
			sum := 1 << (8 + filterBits - 1)
			for k := range filterTaps {
				sum += int(xKernel[k]) * int(ref.Pix[(refY-foY+y)*ref.Stride+refX+x-foX+k])
			}
			im[y*imStride+x] = int16(roundPowerOfTwo(sum, round0Bits))
		}
	}
	offsetBits := 8 + 2*filterBits - round0Bits
	roundOffset := (1 << (offsetBits - round1Bits)) + (1 << (offsetBits - round1Bits - 1))
	bits := 2*filterBits - round0Bits - round1Bits
	for y := range height {
		for x := range width {
			sum := 1 << offsetBits
			for k := range filterTaps {
				sum += int(yKernel[k]) * int(im[(y+k)*imStride+x])
			}
			res := roundPowerOfTwo(sum, round1Bits) - roundOffset
			dst.Pix[(dstY+y)*dst.Stride+dstX+x] = byte(clipPixel(roundPowerOfTwo(res, bits)))
		}
	}
}

func convolve2D8Clamped(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16) {
	const imStride = maxBlockSize
	var im [((maxBlockSize + filterTaps - 1) * maxBlockSize)]int16
	foX := filterTaps/2 - 1
	foY := filterTaps/2 - 1
	imH := height + filterTaps - 1
	for y := range imH {
		for x := range width {
			sum := 1 << (8 + filterBits - 1)
			for k := range filterTaps {
				sum += int(xKernel[k]) * int(loadSample8Clamped(ref, refX+x-foX+k, refY-foY+y))
			}
			im[y*imStride+x] = int16(roundPowerOfTwo(sum, round0Bits))
		}
	}
	offsetBits := 8 + 2*filterBits - round0Bits
	roundOffset := (1 << (offsetBits - round1Bits)) + (1 << (offsetBits - round1Bits - 1))
	bits := 2*filterBits - round0Bits - round1Bits
	for y := range height {
		for x := range width {
			sum := 1 << offsetBits
			for k := range filterTaps {
				sum += int(yKernel[k]) * int(im[(y+k)*imStride+x])
			}
			res := roundPowerOfTwo(sum, round1Bits) - roundOffset
			dst.Pix[(dstY+y)*dst.Stride+dstX+x] = byte(clipPixel(roundPowerOfTwo(res, bits)))
		}
	}
}

func convolveXHighBD(dst frame.Plane, ref frame.Plane, max uint16, dstX int, dstY int, refX int, refY int, width int, height int, kernel [filterTaps]int16) {
	fo := filterTaps/2 - 1
	for y := range height {
		for x := range width {
			sum := 0
			for k := range filterTaps {
				sum += int(kernel[k]) * int(loadHighBDSample(ref, refX+x-fo+k, refY+y))
			}
			res := roundPowerOfTwo(sum, round0Bits)
			storeHighBDSample(dst, dstX+x, dstY+y, clipPixelHighBD(roundPowerOfTwo(res, filterBits-round0Bits), max))
		}
	}
}

func convolveXHighBDClamped(dst frame.Plane, ref frame.Plane, max uint16, dstX int, dstY int, refX int, refY int, width int, height int, kernel [filterTaps]int16) {
	fo := filterTaps/2 - 1
	for y := range height {
		for x := range width {
			sum := 0
			for k := range filterTaps {
				sum += int(kernel[k]) * int(loadHighBDSampleClamped(ref, refX+x-fo+k, refY+y))
			}
			res := roundPowerOfTwo(sum, round0Bits)
			storeHighBDSample(dst, dstX+x, dstY+y, clipPixelHighBD(roundPowerOfTwo(res, filterBits-round0Bits), max))
		}
	}
}

func convolveYHighBD(dst frame.Plane, ref frame.Plane, max uint16, dstX int, dstY int, refX int, refY int, width int, height int, kernel [filterTaps]int16) {
	fo := filterTaps/2 - 1
	for y := range height {
		for x := range width {
			sum := 0
			for k := range filterTaps {
				sum += int(kernel[k]) * int(loadHighBDSample(ref, refX+x, refY+y-fo+k))
			}
			storeHighBDSample(dst, dstX+x, dstY+y, clipPixelHighBD(roundPowerOfTwo(sum, filterBits), max))
		}
	}
}

func convolveYHighBDClamped(dst frame.Plane, ref frame.Plane, max uint16, dstX int, dstY int, refX int, refY int, width int, height int, kernel [filterTaps]int16) {
	fo := filterTaps/2 - 1
	for y := range height {
		for x := range width {
			sum := 0
			for k := range filterTaps {
				sum += int(kernel[k]) * int(loadHighBDSampleClamped(ref, refX+x, refY+y-fo+k))
			}
			storeHighBDSample(dst, dstX+x, dstY+y, clipPixelHighBD(roundPowerOfTwo(sum, filterBits), max))
		}
	}
}

func convolve2DHighBD(dst frame.Plane, ref frame.Plane, bitDepth uint8, max uint16, dstX int, dstY int, refX int, refY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16) {
	const imStride = maxBlockSize
	var im [((maxBlockSize + filterTaps - 1) * maxBlockSize)]int32
	foX := filterTaps/2 - 1
	foY := filterTaps/2 - 1
	imH := height + filterTaps - 1
	for y := range imH {
		for x := range width {
			sum := 1 << (int(bitDepth) + filterBits - 1)
			for k := range filterTaps {
				sum += int(xKernel[k]) * int(loadHighBDSample(ref, refX+x-foX+k, refY-foY+y))
			}
			im[y*imStride+x] = int32(roundPowerOfTwo(sum, round0Bits))
		}
	}
	offsetBits := int(bitDepth) + 2*filterBits - round0Bits
	roundOffset := (1 << (offsetBits - round1Bits)) + (1 << (offsetBits - round1Bits - 1))
	bits := 2*filterBits - round0Bits - round1Bits
	for y := range height {
		for x := range width {
			sum := 1 << offsetBits
			for k := range filterTaps {
				sum += int(yKernel[k]) * int(im[(y+k)*imStride+x])
			}
			res := roundPowerOfTwo(sum, round1Bits) - roundOffset
			storeHighBDSample(dst, dstX+x, dstY+y, clipPixelHighBD(roundPowerOfTwo(res, bits), max))
		}
	}
}

func convolve2DHighBDClamped(dst frame.Plane, ref frame.Plane, bitDepth uint8, max uint16, dstX int, dstY int, refX int, refY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16) {
	const imStride = maxBlockSize
	var im [((maxBlockSize + filterTaps - 1) * maxBlockSize)]int32
	foX := filterTaps/2 - 1
	foY := filterTaps/2 - 1
	imH := height + filterTaps - 1
	for y := range imH {
		for x := range width {
			sum := 1 << (int(bitDepth) + filterBits - 1)
			for k := range filterTaps {
				sum += int(xKernel[k]) * int(loadHighBDSampleClamped(ref, refX+x-foX+k, refY-foY+y))
			}
			im[y*imStride+x] = int32(roundPowerOfTwo(sum, round0Bits))
		}
	}
	offsetBits := int(bitDepth) + 2*filterBits - round0Bits
	roundOffset := (1 << (offsetBits - round1Bits)) + (1 << (offsetBits - round1Bits - 1))
	bits := 2*filterBits - round0Bits - round1Bits
	for y := range height {
		for x := range width {
			sum := 1 << offsetBits
			for k := range filterTaps {
				sum += int(yKernel[k]) * int(im[(y+k)*imStride+x])
			}
			res := roundPowerOfTwo(sum, round1Bits) - roundOffset
			storeHighBDSample(dst, dstX+x, dstY+y, clipPixelHighBD(roundPowerOfTwo(res, bits), max))
		}
	}
}

func loadSample8Clamped(plane frame.Plane, x int, y int) byte {
	x = clampInt(x, 0, plane.Width-1)
	y = clampInt(y, 0, plane.Height-1)
	return plane.Pix[y*plane.Stride+x]
}

func loadHighBDSample(plane frame.Plane, x int, y int) uint16 {
	offset := y*plane.Stride + x*2
	return uint16(plane.Pix[offset]) | uint16(plane.Pix[offset+1])<<8
}

func loadHighBDSampleClamped(plane frame.Plane, x int, y int) uint16 {
	x = clampInt(x, 0, plane.Width-1)
	y = clampInt(y, 0, plane.Height-1)
	return loadHighBDSample(plane, x, y)
}

func storeHighBDSample(plane frame.Plane, x int, y int, value uint16) {
	offset := y*plane.Stride + x*2
	plane.Pix[offset] = byte(value)
	plane.Pix[offset+1] = byte(value >> 8)
}

func planeRegionFits(plane frame.Plane, bytesPerSample int, x int, y int, width int, height int) bool {
	if bytesPerSample != 1 && bytesPerSample != 2 {
		return false
	}
	if plane.Stride <= 0 || plane.Width <= 0 || plane.Height <= 0 || x < 0 || y < 0 || width <= 0 || height <= 0 {
		return false
	}
	if x > plane.Width-width || y > plane.Height-height {
		return false
	}
	rowBytes, ok := checkedMulNonNegative(width, bytesPerSample)
	if !ok || rowBytes > plane.Stride {
		return false
	}
	minStride, ok := checkedMulNonNegative(plane.Width, bytesPerSample)
	if !ok || plane.Stride < minStride {
		return false
	}
	rowOffset, ok := checkedMulNonNegative(y, plane.Stride)
	if !ok {
		return false
	}
	colOffset, ok := checkedMulNonNegative(x, bytesPerSample)
	if !ok {
		return false
	}
	offset, ok := checkedAdd(rowOffset, colOffset)
	if !ok {
		return false
	}
	lastRowOffset, ok := checkedMulNonNegative(height-1, plane.Stride)
	if !ok {
		return false
	}
	windowLen, ok := checkedAdd(lastRowOffset, rowBytes)
	if !ok {
		return false
	}
	end, ok := checkedAdd(offset, windowLen)
	return ok && offset >= 0 && end <= len(plane.Pix)
}

func roundPowerOfTwo(value int, bits int) int {
	if bits <= 0 {
		return value
	}
	return (value + (1 << (bits - 1))) >> bits
}

func clipPixel(v int) uint8 {
	if v < 0 {
		return 0
	}
	if v > 255 {
		return 255
	}
	return uint8(v)
}

func clipPixelHighBD(v int, max uint16) uint16 {
	if v < 0 {
		return 0
	}
	if v > int(max) {
		return max
	}
	return uint16(v)
}

func highBDMax(bitDepth uint8) (uint16, bool) {
	switch bitDepth {
	case 10, 12:
		return uint16((1 << bitDepth) - 1), true
	default:
		return 0, false
	}
}

func bitDepthMatchesSampleWidth(bytesPerSample int, bitDepth uint8) bool {
	return (bytesPerSample == 1 && bitDepth == 8) ||
		(bytesPerSample == 2 && (bitDepth == 10 || bitDepth == 12))
}

func scaleFullpel(v int) (int32, bool) {
	scaled := int64(v) * SubpelScale
	if scaled < minInt32 || scaled > maxInt32 {
		return 0, false
	}
	return int32(scaled), true
}

func lowerIntegerPrecision(v int32) int32 {
	mod := v % SubpelScale
	if mod == 0 {
		return v
	}
	v -= mod
	if absInt32(mod) > SubpelScale/2 {
		if mod > 0 {
			v += SubpelScale
		} else {
			v -= SubpelScale
		}
	}
	return v
}

func absInt32(v int32) int32 {
	if v < 0 {
		return -v
	}
	return v
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

func checkedAdd(a int, b int) (int, bool) {
	if b > 0 && a > maxInt-b {
		return 0, false
	}
	if b < 0 && a < minInt-b {
		return 0, false
	}
	return a + b, true
}

func checkedMulNonNegative(a int, b int) (int, bool) {
	if a < 0 || b < 0 {
		return 0, false
	}
	if a != 0 && b > maxInt/a {
		return 0, false
	}
	return a * b, true
}

const (
	maxInt32 = int64(1<<31 - 1)
	minInt32 = -1 << 31
	maxInt   = int(^uint(0) >> 1)
	minInt   = -maxInt - 1
)
