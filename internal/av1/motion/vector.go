// Ported from libaom:
//   av1/common/reconinter.c
//   aom_dsp/aom_convolve.c
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

package motion

import (
	"encoding/binary"

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
	Row int16
	Col int16
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

// TranslationOrigin is ReferenceOriginSubsampled for a plane position whose
// Q4 coordinate fits in an int (any position inside a frame), so it cannot
// fail: dav1d's mc() derives the same origin from the block position and mv.
func TranslationOrigin(dstX int, dstY int, mv Vector, subsamplingX bool, subsamplingY bool) (refX int, refY int, subX int, subY int) {
	mvX, mvY := int(mv.Col), int(mv.Row)
	if !subsamplingX {
		mvX *= 2
	}
	if !subsamplingY {
		mvY *= 2
	}
	posX, posY := dstX*16+mvX, dstY*16+mvY
	return posX >> 4, posY >> 4, posX & 15, posY & 15
}

// PredictInterPlaneBlockFromOriginWithFilterBitDepth predicts a block from an
// explicit reference origin and subpel phase, clipping to bitDepth.
func PredictInterPlaneBlockFromOriginWithFilterBitDepth(dst frame.Plane, ref frame.Plane, bytesPerSample int, bitDepth uint8, dstX int, dstY int, refX int, refY int, width int, height int, subX int, subY int, filters InterpFilters) error {
	return predictInterPlaneBlockFromOriginWithFilterSize(dst, ref, bytesPerSample, bitDepth, true, dstX, dstY, refX, refY, width, height, width, height, subX, subY, filters)
}

// PredictInterPlaneBlockFromOriginWithFilterBitDepthFilterSizeScratch is
// PredictInterPlaneBlockFromOriginWithFilterBitDepth with an explicit filter
// block size and optional caller-owned 2D convolution scratch for hot decode
// paths.
func PredictInterPlaneBlockFromOriginWithFilterBitDepthFilterSizeScratch(dst frame.Plane, ref frame.Plane, bytesPerSample int, bitDepth uint8, dstX int, dstY int, refX int, refY int, width int, height int, filterW int, filterH int, subX int, subY int, filters InterpFilters, scratch *ConvolveScratch) error {
	return predictInterPlaneBlockFromOriginWithFilterSizeScratch(dst, ref, bytesPerSample, bitDepth, true, dstX, dstY, refX, refY, width, height, filterW, filterH, subX, subY, filters, scratch)
}

// PredictTranslation8 is the decoder's single-reference 8-bit translational
// predictor: ReferenceOriginSubsampled followed by
// PredictInterPlaneBlockFromOriginWithFilterBitDepthFilterSizeScratch at bit
// depth 8, with the dav1d-style copy/put kernels tried first.
//
// Accesses stay inside ref.Pix (taps outside the plane are edge-clamped,
// never read) and inside the width x height block of the destination.
func PredictTranslation8(dst *frame.Plane, ref *frame.Plane, dstX int, dstY int, width int, height int, filterW int, filterH int, mv Vector, subsamplingX bool, subsamplingY bool, filters InterpFilters, scratch *ConvolveScratch) error {
	if !filters.X.Valid() || !filters.Y.Valid() {
		return ErrInvalidMotion
	}
	refX, refY, subX, subY, err := ReferenceOriginSubsampled(dstX, dstY, mv, subsamplingX, subsamplingY)
	if err != nil {
		return err
	}
	if subX|subY == 0 {
		if mc8CopyFast(dst, ref, dstX, dstY, refX, refY, width, height) {
			return nil
		}
	} else if filterW > 0 && filterH > 0 &&
		Put8(dst, ref, dstX, dstY, refX, refY, width, height, filterW, filterH, subX, subY, filters, scratch) {
		return nil
	}
	return predictInterPlaneBlockFromOriginWithFilterSizeScratch(*dst, *ref, 1, 8, true, dstX, dstY, refX, refY, width, height, filterW, filterH, subX, subY, filters, scratch)
}

// PredictTranslation16 is PredictTranslation8 for 10- and 12-bit planes:
// ReferenceOriginSubsampled followed by the 2-byte-sample single-reference
// predictor, for the decoder's lean (dav1d recon_b_inter mc()) route.
//
// Accesses stay inside ref.Pix (taps outside the plane are edge-clamped,
// never read) and inside the width x height block of the destination.
func PredictTranslation16(dst *frame.Plane, ref *frame.Plane, bitDepth uint8, dstX int, dstY int, width int, height int, filterW int, filterH int, mv Vector, subsamplingX bool, subsamplingY bool, filters InterpFilters, scratch *ConvolveScratch) error {
	refX, refY, subX, subY, err := ReferenceOriginSubsampled(dstX, dstY, mv, subsamplingX, subsamplingY)
	if err != nil {
		return err
	}
	return predictInterPlaneBlockFromOriginWithFilterSizeScratch(*dst, *ref, 2, bitDepth, true, dstX, dstY, refX, refY, width, height, filterW, filterH, subX, subY, filters, scratch)
}

func predictInterPlaneBlockFromOriginWithFilterSize(dst frame.Plane, ref frame.Plane, bytesPerSample int, bitDepth uint8, explicitBitDepth bool, dstX int, dstY int, refX int, refY int, width int, height int, filterW int, filterH int, subX int, subY int, filters InterpFilters) error {
	return predictInterPlaneBlockFromOriginWithFilterSizeScratch(dst, ref, bytesPerSample, bitDepth, explicitBitDepth, dstX, dstY, refX, refY, width, height, filterW, filterH, subX, subY, filters, nil)
}

func predictInterPlaneBlockFromOriginWithFilterSizeScratch(dst frame.Plane, ref frame.Plane, bytesPerSample int, bitDepth uint8, explicitBitDepth bool, dstX int, dstY int, refX int, refY int, width int, height int, filterW int, filterH int, subX int, subY int, filters InterpFilters, scratch *ConvolveScratch) error {
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
		if bytesPerSample == 1 && mc8CopyFast(&dst, &ref, dstX, dstY, refX, refY, width, height) {
			return nil
		}
		if bytesPerSample == 2 && mc16CopyFast(&dst, &ref, dstX, dstY, refX, refY, width, height) {
			return nil
		}
		if planeRegionFits(ref, bytesPerSample, refX, refY, width, height) {
			if err := dsp.CopyPlaneBlock(dst, ref, bytesPerSample, dstX, dstY, refX, refY, width, height); err != nil {
				return ErrInvalidMotion
			}
			return nil
		}
		return copyPlaneBlockClamped(dst, ref, bytesPerSample, dstX, dstY, refX, refY, width, height)
	}
	if filterW <= 0 {
		filterW = width
	}
	if filterH <= 0 {
		filterH = height
	}
	if bytesPerSample != 1 {
		if bytesPerSample == 2 && explicitBitDepth {
			if err := predictInterPlaneBlockHighBD(dst, ref, bitDepth, dstX, dstY, refX, refY, width, height, filterW, filterH, subX, subY, filters, scratch); err != nil {
				return ErrInvalidMotion
			}
			return nil
		}
		return ErrInvalidMotion
	}
	if err := predictInterPlaneBlock8(dst, ref, dstX, dstY, refX, refY, width, height, filterW, filterH, subX, subY, filters, scratch); err != nil {
		return ErrInvalidMotion
	}
	return nil
}

func referenceOriginQ4(dst int, mvQ4 int64) (int, int, error) {
	pos := int64(dst)*16 + mvQ4
	ref := pos >> 4
	if ref < int64(minInt) || ref > int64(maxInt) {
		return 0, 0, ErrInvalidMotion
	}
	return int(ref), int(pos & 15), nil
}

func predictInterPlaneBlock8(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, filterW int, filterH int, subX int, subY int, filters InterpFilters, scratch *ConvolveScratch) error {
	if width <= 0 || height <= 0 || width > maxBlockSize || height > maxBlockSize {
		return ErrInvalidMotion
	}
	if Put8(&dst, &ref, dstX, dstY, refX, refY, width, height, filterW, filterH, subX, subY, filters, scratch) {
		return nil
	}
	if !planeRegionFits(dst, 1, dstX, dstY, width, height) {
		return ErrInvalidMotion
	}
	if !planeRegionFits(ref, 1, 0, 0, ref.Width, ref.Height) {
		return ErrInvalidMotion
	}
	xKernel, err := interpKernel(filters.X, filterW, subX)
	if err != nil {
		return err
	}
	yKernel, err := interpKernel(filters.Y, filterH, subY)
	if err != nil {
		return err
	}
	foX := filterTaps/2 - 1
	foY := filterTaps/2 - 1
	switch {
	case subX != 0 && subY != 0:
		if planeRegionFits(ref, 1, refX-foX, refY-foY, width+filterTaps-1, height+filterTaps-1) {
			convolve2D8WithScratchImpl(dst, ref, dstX, dstY, refX, refY, width, height, xKernel, yKernel, scratch)
		} else {
			convolve2D8ClampedWithScratchImpl(dst, ref, dstX, dstY, refX, refY, width, height, xKernel, yKernel, scratch)
		}
	case subX != 0:
		if planeRegionFits(ref, 1, refX-foX, refY, width+filterTaps-1, height) {
			convolveX8Impl(dst, ref, dstX, dstY, refX, refY, width, height, xKernel)
		} else {
			convolveX8ClampedWithScratchImpl(dst, ref, dstX, dstY, refX, refY, width, height, xKernel, scratch)
		}
	case subY != 0:
		if planeRegionFits(ref, 1, refX, refY-foY, width, height+filterTaps-1) {
			convolveY8Impl(dst, ref, dstX, dstY, refX, refY, width, height, yKernel)
		} else {
			convolveY8ClampedWithScratchImpl(dst, ref, dstX, dstY, refX, refY, width, height, yKernel, scratch)
		}
	}
	return nil
}

func predictInterPlaneBlockHighBD(dst frame.Plane, ref frame.Plane, bitDepth uint8, dstX int, dstY int, refX int, refY int, width int, height int, filterW int, filterH int, subX int, subY int, filters InterpFilters, scratch *ConvolveScratch) error {
	if mc16PutFast(&dst, &ref, int(bitDepth), dstX, dstY, refX, refY, width, height, filterW, filterH, subX, subY, filters, scratch) {
		return nil
	}
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
	xKernel, err := interpKernel(filters.X, filterW, subX)
	if err != nil {
		return err
	}
	yKernel, err := interpKernel(filters.Y, filterH, subY)
	if err != nil {
		return err
	}
	foX := filterTaps/2 - 1
	foY := filterTaps/2 - 1
	switch {
	case subX != 0 && subY != 0:
		if planeRegionFits(ref, 2, refX-foX, refY-foY, width+filterTaps-1, height+filterTaps-1) {
			convolve2DHighBDWithScratchImpl(dst, ref, bitDepth, max, dstX, dstY, refX, refY, width, height, xKernel, yKernel, scratch)
		} else {
			convolve2DHighBDClampedWithScratchImpl(dst, ref, bitDepth, max, dstX, dstY, refX, refY, width, height, xKernel, yKernel, scratch)
		}
	case subX != 0:
		if planeRegionFits(ref, 2, refX-foX, refY, width+filterTaps-1, height) {
			convolveXHighBDImpl(dst, ref, bitDepth, max, dstX, dstY, refX, refY, width, height, xKernel)
		} else {
			convolveXHighBDClampedImpl(dst, ref, bitDepth, max, dstX, dstY, refX, refY, width, height, xKernel)
		}
	case subY != 0:
		if planeRegionFits(ref, 2, refX, refY-foY, width, height+filterTaps-1) {
			convolveYHighBDImpl(dst, ref, bitDepth, max, dstX, dstY, refX, refY, width, height, yKernel)
		} else {
			convolveYHighBDClampedImpl(dst, ref, bitDepth, max, dstX, dstY, refX, refY, width, height, yKernel)
		}
	}
	return nil
}

func copyPlaneBlockClamped(dst frame.Plane, ref frame.Plane, bytesPerSample int, dstX int, dstY int, refX int, refY int, width int, height int) error {
	if !planeRegionFits(dst, bytesPerSample, dstX, dstY, width, height) ||
		!planeRegionFits(ref, bytesPerSample, 0, 0, ref.Width, ref.Height) {
		return ErrInvalidMotion
	}
	if bytesPerSample != 1 && bytesPerSample != 2 {
		return ErrInvalidMotion
	}
	// emu_edge (dav1d mc.c): the horizontal source clamp splits each row into
	// three row-invariant regions -- a left pad reading ref column 0, an
	// in-bounds middle that maps 1:1 to a contiguous ref span, and a right pad
	// reading ref column W-1. Deriving the region geometry once (refX/width are
	// the same on every row) turns the inner loop into a single memmove for the
	// bulk in-bounds span and drops the per-pixel horizontal clamp entirely;
	// only the vertical source row still clamps per row.
	bps := bytesPerSample
	leftW := clampInt(-refX, 0, width)               // x < leftW  => sx = 0
	rightStart := clampInt(ref.Width-refX, 0, width) // x >= rightStart => sx = W-1
	midW := rightStart - leftW                       // >= 0; middle sx = refX+x
	srcStartX := refX + leftW                        // in [0, ref.Width) when midW>0
	for y := range height {
		sy := clampInt(refY+y, 0, ref.Height-1)
		refBase := sy * ref.Stride
		dstBase := (dstY+y)*dst.Stride + dstX*bps
		if bps == 1 {
			dstRow := dst.Pix[dstBase : dstBase+width]
			if midW > 0 {
				copy(dstRow[leftW:rightStart], ref.Pix[refBase+srcStartX:refBase+srcStartX+midW])
			}
			if leftW > 0 {
				lv := ref.Pix[refBase]
				for x := range leftW {
					dstRow[x] = lv
				}
			}
			if rightStart < width {
				rv := ref.Pix[refBase+ref.Width-1]
				for x := rightStart; x < width; x++ {
					dstRow[x] = rv
				}
			}
			continue
		}
		// bps == 2
		dstRow := dst.Pix[dstBase : dstBase+width*2]
		if midW > 0 {
			copy(dstRow[leftW*2:rightStart*2], ref.Pix[refBase+srcStartX*2:refBase+(srcStartX+midW)*2])
		}
		if leftW > 0 {
			l0, l1 := ref.Pix[refBase], ref.Pix[refBase+1]
			for x := range leftW {
				dstRow[x*2] = l0
				dstRow[x*2+1] = l1
			}
		}
		if rightStart < width {
			ro := refBase + (ref.Width-1)*2
			r0, r1 := ref.Pix[ro], ref.Pix[ro+1]
			for x := rightStart; x < width; x++ {
				dstRow[x*2] = r0
				dstRow[x*2+1] = r1
			}
		}
	}
	return nil
}

func convolveX8PureGo(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, kernel [filterTaps]int16) {
	if width <= 0 || height <= 0 {
		return
	}
	fo := filterTaps/2 - 1
	k0, k1, k2, k3, k4, k5, k6, k7 := int(kernel[0]), int(kernel[1]), int(kernel[2]), int(kernel[3]), int(kernel[4]), int(kernel[5]), int(kernel[6]), int(kernel[7])
	// 4-tap kernels (small blocks / bilinear) have zero end taps, so skip those MACs.
	fourTap := k0 == 0 && k1 == 0 && k6 == 0 && k7 == 0
	for y := range height {
		srcBase := (refY+y)*ref.Stride + refX - fo
		srcRow := ref.Pix[srcBase : srcBase+width+filterTaps-1]
		dstBase := (dstY+y)*dst.Stride + dstX
		dstRow := dst.Pix[dstBase : dstBase+width]
		if fourTap {
			s2, s3, s4 := int(srcRow[2]), int(srcRow[3]), int(srcRow[4])
			x := 0
			for ; x+3 < width; x += 4 {
				s5 := int(srcRow[x+5])
				s6 := int(srcRow[x+6])
				s7 := int(srcRow[x+7])
				s8 := int(srcRow[x+8])
				dstRow[x] = clipPixel((((k2*s2 + k3*s3 + k4*s4 + k5*s5 + 4) >> 3) + 8) >> 4)
				dstRow[x+1] = clipPixel((((k2*s3 + k3*s4 + k4*s5 + k5*s6 + 4) >> 3) + 8) >> 4)
				dstRow[x+2] = clipPixel((((k2*s4 + k3*s5 + k4*s6 + k5*s7 + 4) >> 3) + 8) >> 4)
				dstRow[x+3] = clipPixel((((k2*s5 + k3*s6 + k4*s7 + k5*s8 + 4) >> 3) + 8) >> 4)
				s2, s3, s4 = s6, s7, s8
			}
			for ; x < width; x++ {
				s5 := int(srcRow[x+5])
				dstRow[x] = clipPixel((((k2*s2 + k3*s3 + k4*s4 + k5*s5 + 4) >> 3) + 8) >> 4)
				s2, s3, s4 = s3, s4, s5
			}
			continue
		}
		s0, s1, s2, s3, s4, s5, s6 := int(srcRow[0]), int(srcRow[1]), int(srcRow[2]), int(srcRow[3]), int(srcRow[4]), int(srcRow[5]), int(srcRow[6])
		x := 0
		for ; x+3 < width; x += 4 {
			s7 := int(srcRow[x+7])
			s8 := int(srcRow[x+8])
			s9 := int(srcRow[x+9])
			s10 := int(srcRow[x+10])
			sum0 := k0*s0 + k1*s1 + k2*s2 + k3*s3 + k4*s4 + k5*s5 + k6*s6 + k7*s7
			sum1 := k0*s1 + k1*s2 + k2*s3 + k3*s4 + k4*s5 + k5*s6 + k6*s7 + k7*s8
			sum2 := k0*s2 + k1*s3 + k2*s4 + k3*s5 + k4*s6 + k5*s7 + k6*s8 + k7*s9
			sum3 := k0*s3 + k1*s4 + k2*s5 + k3*s6 + k4*s7 + k5*s8 + k6*s9 + k7*s10
			dstRow[x] = clipPixel((((sum0 + 4) >> 3) + 8) >> 4)
			dstRow[x+1] = clipPixel((((sum1 + 4) >> 3) + 8) >> 4)
			dstRow[x+2] = clipPixel((((sum2 + 4) >> 3) + 8) >> 4)
			dstRow[x+3] = clipPixel((((sum3 + 4) >> 3) + 8) >> 4)
			s0, s1, s2, s3, s4, s5, s6 = s4, s5, s6, s7, s8, s9, s10
		}
		for ; x < width; x++ {
			s7 := int(srcRow[x+7])
			sum := k0*s0 + k1*s1 + k2*s2 + k3*s3 + k4*s4 + k5*s5 + k6*s6 + k7*s7
			dstRow[x] = clipPixel((((sum + 4) >> 3) + 8) >> 4)
			s0, s1, s2, s3, s4, s5, s6 = s1, s2, s3, s4, s5, s6, s7
		}
	}
}

func convolveX8ClampedPureGo(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, kernel [filterTaps]int16) {
	fo := filterTaps/2 - 1
	k0, k1, k2, k3, k4, k5, k6, k7 := int(kernel[0]), int(kernel[1]), int(kernel[2]), int(kernel[3]), int(kernel[4]), int(kernel[5]), int(kernel[6]), int(kernel[7])
	fourTap := k0 == 0 && k1 == 0 && k6 == 0 && k7 == 0
	firstTap := refX - fo
	xLo, xHi := clampedXInterior(firstTap, filterTaps, ref.Width, width)
	for y := range height {
		cy := clampInt(refY+y, 0, ref.Height-1)
		rowBase := cy * ref.Stride
		dstRow := dst.Pix[(dstY+y)*dst.Stride+dstX:]
		if fourTap {
			for x := 0; x < xLo; x++ {
				sx := firstTap + x
				sum := k2*int(loadSample8ClampedRow(ref, sx+2, rowBase)) +
					k3*int(loadSample8ClampedRow(ref, sx+3, rowBase)) +
					k4*int(loadSample8ClampedRow(ref, sx+4, rowBase)) +
					k5*int(loadSample8ClampedRow(ref, sx+5, rowBase))
				res := roundPowerOfTwo(sum, round0Bits)
				dstRow[x] = byte(clipPixel(roundPowerOfTwo(res, filterBits-round0Bits)))
			}
			if xHi > xLo {
				srcRow := ref.Pix[rowBase+firstTap+xLo:]
				for x := xLo; x < xHi; x++ {
					i := x - xLo
					s := srcRow[i : i+filterTaps]
					sum := k2*int(s[2]) + k3*int(s[3]) + k4*int(s[4]) + k5*int(s[5])
					res := roundPowerOfTwo(sum, round0Bits)
					dstRow[x] = byte(clipPixel(roundPowerOfTwo(res, filterBits-round0Bits)))
				}
			}
			for x := xHi; x < width; x++ {
				sx := firstTap + x
				sum := k2*int(loadSample8ClampedRow(ref, sx+2, rowBase)) +
					k3*int(loadSample8ClampedRow(ref, sx+3, rowBase)) +
					k4*int(loadSample8ClampedRow(ref, sx+4, rowBase)) +
					k5*int(loadSample8ClampedRow(ref, sx+5, rowBase))
				res := roundPowerOfTwo(sum, round0Bits)
				dstRow[x] = byte(clipPixel(roundPowerOfTwo(res, filterBits-round0Bits)))
			}
			continue
		}
		for x := 0; x < xLo; x++ {
			sx := firstTap + x
			sum := k0*int(loadSample8ClampedRow(ref, sx, rowBase)) +
				k1*int(loadSample8ClampedRow(ref, sx+1, rowBase)) +
				k2*int(loadSample8ClampedRow(ref, sx+2, rowBase)) +
				k3*int(loadSample8ClampedRow(ref, sx+3, rowBase)) +
				k4*int(loadSample8ClampedRow(ref, sx+4, rowBase)) +
				k5*int(loadSample8ClampedRow(ref, sx+5, rowBase)) +
				k6*int(loadSample8ClampedRow(ref, sx+6, rowBase)) +
				k7*int(loadSample8ClampedRow(ref, sx+7, rowBase))
			res := roundPowerOfTwo(sum, round0Bits)
			dstRow[x] = byte(clipPixel(roundPowerOfTwo(res, filterBits-round0Bits)))
		}
		if xHi > xLo {
			srcRow := ref.Pix[rowBase+firstTap+xLo:]
			for x := xLo; x < xHi; x++ {
				i := x - xLo
				s := srcRow[i : i+filterTaps]
				sum := k0*int(s[0]) + k1*int(s[1]) + k2*int(s[2]) + k3*int(s[3]) +
					k4*int(s[4]) + k5*int(s[5]) + k6*int(s[6]) + k7*int(s[7])
				res := roundPowerOfTwo(sum, round0Bits)
				dstRow[x] = byte(clipPixel(roundPowerOfTwo(res, filterBits-round0Bits)))
			}
		}
		for x := xHi; x < width; x++ {
			sx := firstTap + x
			sum := k0*int(loadSample8ClampedRow(ref, sx, rowBase)) +
				k1*int(loadSample8ClampedRow(ref, sx+1, rowBase)) +
				k2*int(loadSample8ClampedRow(ref, sx+2, rowBase)) +
				k3*int(loadSample8ClampedRow(ref, sx+3, rowBase)) +
				k4*int(loadSample8ClampedRow(ref, sx+4, rowBase)) +
				k5*int(loadSample8ClampedRow(ref, sx+5, rowBase)) +
				k6*int(loadSample8ClampedRow(ref, sx+6, rowBase)) +
				k7*int(loadSample8ClampedRow(ref, sx+7, rowBase))
			res := roundPowerOfTwo(sum, round0Bits)
			dstRow[x] = byte(clipPixel(roundPowerOfTwo(res, filterBits-round0Bits)))
		}
	}
}

func convolveY8PureGo(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, kernel [filterTaps]int16) {
	if width <= 0 || height <= 0 {
		return
	}
	fo := filterTaps/2 - 1
	stride := ref.Stride
	k0, k1, k2, k3, k4, k5, k6, k7 := int(kernel[0]), int(kernel[1]), int(kernel[2]), int(kernel[3]), int(kernel[4]), int(kernel[5]), int(kernel[6]), int(kernel[7])
	fourTap := k0 == 0 && k1 == 0 && k6 == 0 && k7 == 0
	for y := range height {
		base := (refY+y-fo)*stride + refX
		dstBase := (dstY+y)*dst.Stride + dstX
		dstRow := dst.Pix[dstBase : dstBase+width]
		if fourTap {
			r2 := ref.Pix[base+2*stride : base+2*stride+width]
			r3 := ref.Pix[base+3*stride : base+3*stride+width]
			r4 := ref.Pix[base+4*stride : base+4*stride+width]
			r5 := ref.Pix[base+5*stride : base+5*stride+width]
			x := 0
			for ; x+3 < width; x += 4 {
				dstRow[x] = clipPixel((k2*int(r2[x]) + k3*int(r3[x]) + k4*int(r4[x]) + k5*int(r5[x]) + 64) >> 7)
				dstRow[x+1] = clipPixel((k2*int(r2[x+1]) + k3*int(r3[x+1]) + k4*int(r4[x+1]) + k5*int(r5[x+1]) + 64) >> 7)
				dstRow[x+2] = clipPixel((k2*int(r2[x+2]) + k3*int(r3[x+2]) + k4*int(r4[x+2]) + k5*int(r5[x+2]) + 64) >> 7)
				dstRow[x+3] = clipPixel((k2*int(r2[x+3]) + k3*int(r3[x+3]) + k4*int(r4[x+3]) + k5*int(r5[x+3]) + 64) >> 7)
			}
			for ; x < width; x++ {
				dstRow[x] = clipPixel((k2*int(r2[x]) + k3*int(r3[x]) + k4*int(r4[x]) + k5*int(r5[x]) + 64) >> 7)
			}
			continue
		}
		r0 := ref.Pix[base : base+width]
		r1 := ref.Pix[base+stride : base+stride+width]
		r2 := ref.Pix[base+2*stride : base+2*stride+width]
		r3 := ref.Pix[base+3*stride : base+3*stride+width]
		r4 := ref.Pix[base+4*stride : base+4*stride+width]
		r5 := ref.Pix[base+5*stride : base+5*stride+width]
		r6 := ref.Pix[base+6*stride : base+6*stride+width]
		r7 := ref.Pix[base+7*stride : base+7*stride+width]
		x := 0
		for ; x+3 < width; x += 4 {
			sum0 := k0*int(r0[x]) + k1*int(r1[x]) + k2*int(r2[x]) + k3*int(r3[x]) + k4*int(r4[x]) + k5*int(r5[x]) + k6*int(r6[x]) + k7*int(r7[x])
			sum1 := k0*int(r0[x+1]) + k1*int(r1[x+1]) + k2*int(r2[x+1]) + k3*int(r3[x+1]) + k4*int(r4[x+1]) + k5*int(r5[x+1]) + k6*int(r6[x+1]) + k7*int(r7[x+1])
			sum2 := k0*int(r0[x+2]) + k1*int(r1[x+2]) + k2*int(r2[x+2]) + k3*int(r3[x+2]) + k4*int(r4[x+2]) + k5*int(r5[x+2]) + k6*int(r6[x+2]) + k7*int(r7[x+2])
			sum3 := k0*int(r0[x+3]) + k1*int(r1[x+3]) + k2*int(r2[x+3]) + k3*int(r3[x+3]) + k4*int(r4[x+3]) + k5*int(r5[x+3]) + k6*int(r6[x+3]) + k7*int(r7[x+3])
			dstRow[x] = clipPixel((sum0 + 64) >> 7)
			dstRow[x+1] = clipPixel((sum1 + 64) >> 7)
			dstRow[x+2] = clipPixel((sum2 + 64) >> 7)
			dstRow[x+3] = clipPixel((sum3 + 64) >> 7)
		}
		for ; x < width; x++ {
			sum := k0*int(r0[x]) + k1*int(r1[x]) + k2*int(r2[x]) + k3*int(r3[x]) + k4*int(r4[x]) + k5*int(r5[x]) + k6*int(r6[x]) + k7*int(r7[x])
			dstRow[x] = clipPixel((sum + 64) >> 7)
		}
	}
}

func convolveY8ClampedPureGo(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, kernel [filterTaps]int16) {
	fo := filterTaps/2 - 1
	stride := ref.Stride
	k0, k1, k2, k3, k4, k5, k6, k7 := int(kernel[0]), int(kernel[1]), int(kernel[2]), int(kernel[3]), int(kernel[4]), int(kernel[5]), int(kernel[6]), int(kernel[7])
	fourTap := k0 == 0 && k1 == 0 && k6 == 0 && k7 == 0
	// Interior output-column range where the horizontal coordinate refX+x lies
	// within [0, Width-1] for every column, so the per-column x-clamp is a no-op.
	xLo, xHi := clampedXInterior(refX, 1, ref.Width, width)
	for y := range height {
		sy := refY + y - fo
		dstRow := dst.Pix[(dstY+y)*dst.Stride+dstX:]
		// When the whole vertical tap window is resident the y-clamp is a no-op
		// and we can walk the column with contiguous strided loads. Inside the
		// resident-x interior the x-clamp is a no-op too, so that span matches
		// the non-clamped reference bit-for-bit.
		yResident := sy >= 0 && sy+filterTaps-1 <= ref.Height-1
		if yResident && xHi > xLo {
			if fourTap {
				col := ref.Pix[(sy+2)*stride+refX+xLo:]
				for x := xLo; x < xHi; x++ {
					i := x - xLo
					sum := k2*int(col[i]) + k3*int(col[i+stride]) +
						k4*int(col[i+2*stride]) + k5*int(col[i+3*stride])
					dstRow[x] = byte(clipPixel(roundPowerOfTwo(sum, filterBits)))
				}
			} else {
				col := ref.Pix[sy*stride+refX+xLo:]
				for x := xLo; x < xHi; x++ {
					i := x - xLo
					sum := k0*int(col[i]) + k1*int(col[i+stride]) + k2*int(col[i+2*stride]) +
						k3*int(col[i+3*stride]) + k4*int(col[i+4*stride]) + k5*int(col[i+5*stride]) +
						k6*int(col[i+6*stride]) + k7*int(col[i+7*stride])
					dstRow[x] = byte(clipPixel(roundPowerOfTwo(sum, filterBits)))
				}
			}
			// Edge columns (x in [0,xLo) and [xHi,width)) still need x-clamping.
			for x := 0; x < xLo; x++ {
				dstRow[x] = convolveY8ClampedColumn(ref, refX+x, sy, fourTap, k0, k1, k2, k3, k4, k5, k6, k7)
			}
			for x := xHi; x < width; x++ {
				dstRow[x] = convolveY8ClampedColumn(ref, refX+x, sy, fourTap, k0, k1, k2, k3, k4, k5, k6, k7)
			}
			continue
		}
		for x := range width {
			dstRow[x] = convolveY8ClampedColumn(ref, refX+x, sy, fourTap, k0, k1, k2, k3, k4, k5, k6, k7)
		}
	}
}

// convolveY8ClampedColumn computes one vertical-convolve output sample at
// source column sx and tap base row sy, clamping both coordinates per tap. It
// is the per-column fallback for off-frame tap windows and is bit-identical to
// the inline loadSample8Clamped form it replaces.
func convolveY8ClampedColumn(ref frame.Plane, sx int, sy int, fourTap bool, k0, k1, k2, k3, k4, k5, k6, k7 int) byte {
	if fourTap {
		sum := k2*int(loadSample8Clamped(ref, sx, sy+2)) +
			k3*int(loadSample8Clamped(ref, sx, sy+3)) +
			k4*int(loadSample8Clamped(ref, sx, sy+4)) +
			k5*int(loadSample8Clamped(ref, sx, sy+5))
		return byte(clipPixel(roundPowerOfTwo(sum, filterBits)))
	}
	sum := k0*int(loadSample8Clamped(ref, sx, sy)) +
		k1*int(loadSample8Clamped(ref, sx, sy+1)) +
		k2*int(loadSample8Clamped(ref, sx, sy+2)) +
		k3*int(loadSample8Clamped(ref, sx, sy+3)) +
		k4*int(loadSample8Clamped(ref, sx, sy+4)) +
		k5*int(loadSample8Clamped(ref, sx, sy+5)) +
		k6*int(loadSample8Clamped(ref, sx, sy+6)) +
		k7*int(loadSample8Clamped(ref, sx, sy+7))
	return byte(clipPixel(roundPowerOfTwo(sum, filterBits)))
}

func convolve2D8PureGo(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16) {
	convolve2D8PureGoWithScratch(dst, ref, dstX, dstY, refX, refY, width, height, xKernel, yKernel, nil)
}

// convolve2D8PureGoWithScratch is convolve2D8PureGo with the intermediate in
// caller scratch when given, which skips zero-filling a ~34KB stack array on
// every block.
func convolve2D8PureGoWithScratch(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16, scratch *ConvolveScratch) {
	if width > 0 && width <= convolve2D8NarrowStride {
		if scratch != nil {
			convolve2D8NarrowPureGo(dst, ref, dstX, dstY, refX, refY, width, height, xKernel, yKernel, scratch.narrowIM())
			return
		}
		var im narrowIM
		convolve2D8NarrowPureGo(dst, ref, dstX, dstY, refX, refY, width, height, xKernel, yKernel, &im)
		return
	}
	if scratch != nil {
		convolve2D8PureGoWithIM(dst, ref, dstX, dstY, refX, refY, width, height, xKernel, yKernel, &scratch.im)
		return
	}
	var im [(maxBlockSize + filterTaps - 1) * maxBlockSize]int16
	convolve2D8PureGoWithIM(dst, ref, dstX, dstY, refX, refY, width, height, xKernel, yKernel, &im)
}

func convolve2D8PureGoWithIM(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16, im *[(maxBlockSize + filterTaps - 1) * maxBlockSize]int16) {
	if width <= 0 || height <= 0 {
		return
	}
	const imStride = maxBlockSize
	foX := filterTaps/2 - 1
	foY := filterTaps/2 - 1
	imH := height + filterTaps - 1
	const xBiasRound = (1 << (8 + filterBits - 1)) + (1 << (round0Bits - 1))

	xk0, xk1, xk2, xk3, xk4, xk5, xk6, xk7 := int(xKernel[0]), int(xKernel[1]), int(xKernel[2]), int(xKernel[3]), int(xKernel[4]), int(xKernel[5]), int(xKernel[6]), int(xKernel[7])
	xFourTap := xk0 == 0 && xk1 == 0 && xk6 == 0 && xk7 == 0
	xSixTapAlt := !xFourTap && xk0 == 0 && xk7 == 0 && xk1 >= 0 && xk2 <= 0 && xk3 >= 0 && xk4 >= 0 && xk5 <= 0 && xk6 >= 0 && (xk1+xk3+xk4+xk6) <= 255 && (-xk2-xk5) <= 255
	for y := range imH {
		srcBase := (refY-foY+y)*ref.Stride + refX - foX
		srcRow := ref.Pix[srcBase : srcBase+width+filterTaps-1]
		imRow := im[y*imStride : y*imStride+width]
		if xSixTapAlt {
			const loMsk = 0x00FF00FF00FF00FF
			pk1, pk3, pk4, pk6 := uint64(xk1), uint64(xk3), uint64(xk4), uint64(xk6)
			nk2, nk5 := uint64(-xk2), uint64(-xk5)
			x := 0
			for ; x+7 < width; x += 8 {
				w0 := binary.LittleEndian.Uint64(srcRow[x+1 : x+9])
				w1 := binary.LittleEndian.Uint64(srcRow[x+6 : x+14])
				e0 := w0 & loMsk
				o0 := (w0 >> 8) & loMsk
				o1 := w1 & loMsk
				e1 := (w1 >> 8) & loMsk
				t3 := (e0 >> 16) | ((e1 >> 16) << 48)
				t4 := (o0 >> 16) | ((o1 >> 32) << 48)
				t5 := (e0 >> 32) | ((e1 >> 16) << 32)

				posE := pk1*e0 + pk3*t3 + pk4*t4 + pk6*o1
				negE := nk2*o0 + nk5*t5
				posO := pk1*o0 + pk3*t4 + pk4*t5 + pk6*e1
				negO := nk2*t3 + nk5*o1

				imRow[x] = int16((xBiasRound + int(uint16(posE)) - int(uint16(negE))) >> round0Bits)
				imRow[x+1] = int16((xBiasRound + int(uint16(posO)) - int(uint16(negO))) >> round0Bits)
				imRow[x+2] = int16((xBiasRound + int(uint16(posE>>16)) - int(uint16(negE>>16))) >> round0Bits)
				imRow[x+3] = int16((xBiasRound + int(uint16(posO>>16)) - int(uint16(negO>>16))) >> round0Bits)
				imRow[x+4] = int16((xBiasRound + int(uint16(posE>>32)) - int(uint16(negE>>32))) >> round0Bits)
				imRow[x+5] = int16((xBiasRound + int(uint16(posO>>32)) - int(uint16(negO>>32))) >> round0Bits)
				imRow[x+6] = int16((xBiasRound + int(uint16(posE>>48)) - int(uint16(negE>>48))) >> round0Bits)
				imRow[x+7] = int16((xBiasRound + int(uint16(posO>>48)) - int(uint16(negO>>48))) >> round0Bits)
			}
			for ; x < width; x++ {
				sum := xBiasRound + xk1*int(srcRow[x+1]) + xk2*int(srcRow[x+2]) + xk3*int(srcRow[x+3]) +
					xk4*int(srcRow[x+4]) + xk5*int(srcRow[x+5]) + xk6*int(srcRow[x+6])
				imRow[x] = int16(sum >> round0Bits)
			}
			continue
		}
		if xFourTap {
			s2, s3, s4 := int(srcRow[2]), int(srcRow[3]), int(srcRow[4])
			x := 0
			for ; x+3 < width; x += 4 {
				s5 := int(srcRow[x+5])
				s6 := int(srcRow[x+6])
				s7 := int(srcRow[x+7])
				s8 := int(srcRow[x+8])
				imRow[x] = int16((xBiasRound + xk2*s2 + xk3*s3 + xk4*s4 + xk5*s5) >> round0Bits)
				imRow[x+1] = int16((xBiasRound + xk2*s3 + xk3*s4 + xk4*s5 + xk5*s6) >> round0Bits)
				imRow[x+2] = int16((xBiasRound + xk2*s4 + xk3*s5 + xk4*s6 + xk5*s7) >> round0Bits)
				imRow[x+3] = int16((xBiasRound + xk2*s5 + xk3*s6 + xk4*s7 + xk5*s8) >> round0Bits)
				s2, s3, s4 = s6, s7, s8
			}
			for ; x < width; x++ {
				s5 := int(srcRow[x+5])
				imRow[x] = int16((xBiasRound + xk2*s2 + xk3*s3 + xk4*s4 + xk5*s5) >> round0Bits)
				s2, s3, s4 = s3, s4, s5
			}
			continue
		}
		s0, s1, s2, s3, s4, s5, s6 := int(srcRow[0]), int(srcRow[1]), int(srcRow[2]), int(srcRow[3]), int(srcRow[4]), int(srcRow[5]), int(srcRow[6])
		x := 0
		for ; x+3 < width; x += 4 {
			s7 := int(srcRow[x+7])
			s8 := int(srcRow[x+8])
			s9 := int(srcRow[x+9])
			s10 := int(srcRow[x+10])
			sum0 := xBiasRound + xk0*s0 + xk1*s1 + xk2*s2 + xk3*s3 + xk4*s4 + xk5*s5 + xk6*s6 + xk7*s7
			sum1 := xBiasRound + xk0*s1 + xk1*s2 + xk2*s3 + xk3*s4 + xk4*s5 + xk5*s6 + xk6*s7 + xk7*s8
			sum2 := xBiasRound + xk0*s2 + xk1*s3 + xk2*s4 + xk3*s5 + xk4*s6 + xk5*s7 + xk6*s8 + xk7*s9
			sum3 := xBiasRound + xk0*s3 + xk1*s4 + xk2*s5 + xk3*s6 + xk4*s7 + xk5*s8 + xk6*s9 + xk7*s10
			imRow[x] = int16(sum0 >> round0Bits)
			imRow[x+1] = int16(sum1 >> round0Bits)
			imRow[x+2] = int16(sum2 >> round0Bits)
			imRow[x+3] = int16(sum3 >> round0Bits)
			s0, s1, s2, s3, s4, s5, s6 = s4, s5, s6, s7, s8, s9, s10
		}
		for ; x < width; x++ {
			s7 := int(srcRow[x+7])
			sum := xBiasRound + xk0*s0 + xk1*s1 + xk2*s2 + xk3*s3 + xk4*s4 + xk5*s5 + xk6*s6 + xk7*s7
			imRow[x] = int16(sum >> round0Bits)
			s0, s1, s2, s3, s4, s5, s6 = s1, s2, s3, s4, s5, s6, s7
		}
	}
	const (
		offsetBits  = 8 + 2*filterBits - round0Bits
		roundOffset = (1 << (offsetBits - round1Bits)) + (1 << (offsetBits - round1Bits - 1))
		yBiasRound  = (1 << offsetBits) + (1 << (round1Bits - 1))
	)

	yk0, yk1, yk2, yk3, yk4, yk5, yk6, yk7 := int(yKernel[0]), int(yKernel[1]), int(yKernel[2]), int(yKernel[3]), int(yKernel[4]), int(yKernel[5]), int(yKernel[6]), int(yKernel[7])
	yFourTap := yk0 == 0 && yk1 == 0 && yk6 == 0 && yk7 == 0
	ySixTap := !yFourTap && yk0 == 0 && yk7 == 0
	for y := range height {
		dstBase := (dstY+y)*dst.Stride + dstX
		dstRow := dst.Pix[dstBase : dstBase+width]
		if ySixTap {
			r1 := im[(y+1)*imStride : (y+1)*imStride+width]
			r2 := im[(y+2)*imStride : (y+2)*imStride+width]
			r3 := im[(y+3)*imStride : (y+3)*imStride+width]
			r4 := im[(y+4)*imStride : (y+4)*imStride+width]
			r5 := im[(y+5)*imStride : (y+5)*imStride+width]
			r6 := im[(y+6)*imStride : (y+6)*imStride+width]
			x := 0
			for ; x+3 < width; x += 4 {
				sum0 := yBiasRound + yk1*int(r1[x]) + yk2*int(r2[x]) + yk3*int(r3[x]) + yk4*int(r4[x]) + yk5*int(r5[x]) + yk6*int(r6[x])
				sum1 := yBiasRound + yk1*int(r1[x+1]) + yk2*int(r2[x+1]) + yk3*int(r3[x+1]) + yk4*int(r4[x+1]) + yk5*int(r5[x+1]) + yk6*int(r6[x+1])
				sum2 := yBiasRound + yk1*int(r1[x+2]) + yk2*int(r2[x+2]) + yk3*int(r3[x+2]) + yk4*int(r4[x+2]) + yk5*int(r5[x+2]) + yk6*int(r6[x+2])
				sum3 := yBiasRound + yk1*int(r1[x+3]) + yk2*int(r2[x+3]) + yk3*int(r3[x+3]) + yk4*int(r4[x+3]) + yk5*int(r5[x+3]) + yk6*int(r6[x+3])
				v0 := (sum0 >> round1Bits) - roundOffset
				v1 := (sum1 >> round1Bits) - roundOffset
				v2 := (sum2 >> round1Bits) - roundOffset
				v3 := (sum3 >> round1Bits) - roundOffset
				if uint(v0|v1|v2|v3) <= 255 {
					dstRow[x] = byte(v0)
					dstRow[x+1] = byte(v1)
					dstRow[x+2] = byte(v2)
					dstRow[x+3] = byte(v3)
				} else {
					dstRow[x] = clipPixel(v0)
					dstRow[x+1] = clipPixel(v1)
					dstRow[x+2] = clipPixel(v2)
					dstRow[x+3] = clipPixel(v3)
				}
			}
			for ; x < width; x++ {
				sum := yBiasRound + yk1*int(r1[x]) + yk2*int(r2[x]) + yk3*int(r3[x]) + yk4*int(r4[x]) + yk5*int(r5[x]) + yk6*int(r6[x])
				dstRow[x] = clipPixel((sum >> round1Bits) - roundOffset)
			}
			continue
		}
		if yFourTap {
			r2 := im[(y+2)*imStride : (y+2)*imStride+width]
			r3 := im[(y+3)*imStride : (y+3)*imStride+width]
			r4 := im[(y+4)*imStride : (y+4)*imStride+width]
			r5 := im[(y+5)*imStride : (y+5)*imStride+width]
			x := 0
			for ; x+3 < width; x += 4 {
				sum0 := yBiasRound + yk2*int(r2[x]) + yk3*int(r3[x]) + yk4*int(r4[x]) + yk5*int(r5[x])
				sum1 := yBiasRound + yk2*int(r2[x+1]) + yk3*int(r3[x+1]) + yk4*int(r4[x+1]) + yk5*int(r5[x+1])
				sum2 := yBiasRound + yk2*int(r2[x+2]) + yk3*int(r3[x+2]) + yk4*int(r4[x+2]) + yk5*int(r5[x+2])
				sum3 := yBiasRound + yk2*int(r2[x+3]) + yk3*int(r3[x+3]) + yk4*int(r4[x+3]) + yk5*int(r5[x+3])
				v0 := (sum0 >> round1Bits) - roundOffset
				v1 := (sum1 >> round1Bits) - roundOffset
				v2 := (sum2 >> round1Bits) - roundOffset
				v3 := (sum3 >> round1Bits) - roundOffset
				if uint(v0|v1|v2|v3) <= 255 {
					dstRow[x] = byte(v0)
					dstRow[x+1] = byte(v1)
					dstRow[x+2] = byte(v2)
					dstRow[x+3] = byte(v3)
				} else {
					dstRow[x] = clipPixel(v0)
					dstRow[x+1] = clipPixel(v1)
					dstRow[x+2] = clipPixel(v2)
					dstRow[x+3] = clipPixel(v3)
				}
			}
			for ; x < width; x++ {
				sum := yBiasRound + yk2*int(r2[x]) + yk3*int(r3[x]) + yk4*int(r4[x]) + yk5*int(r5[x])
				dstRow[x] = clipPixel((sum >> round1Bits) - roundOffset)
			}
			continue
		}
		r0 := im[y*imStride : y*imStride+width]
		r1 := im[(y+1)*imStride : (y+1)*imStride+width]
		r2 := im[(y+2)*imStride : (y+2)*imStride+width]
		r3 := im[(y+3)*imStride : (y+3)*imStride+width]
		r4 := im[(y+4)*imStride : (y+4)*imStride+width]
		r5 := im[(y+5)*imStride : (y+5)*imStride+width]
		r6 := im[(y+6)*imStride : (y+6)*imStride+width]
		r7 := im[(y+7)*imStride : (y+7)*imStride+width]
		x := 0
		for ; x+3 < width; x += 4 {
			sum0 := yBiasRound + yk0*int(r0[x]) + yk1*int(r1[x]) + yk2*int(r2[x]) + yk3*int(r3[x]) +
				yk4*int(r4[x]) + yk5*int(r5[x]) + yk6*int(r6[x]) + yk7*int(r7[x])
			sum1 := yBiasRound + yk0*int(r0[x+1]) + yk1*int(r1[x+1]) + yk2*int(r2[x+1]) + yk3*int(r3[x+1]) +
				yk4*int(r4[x+1]) + yk5*int(r5[x+1]) + yk6*int(r6[x+1]) + yk7*int(r7[x+1])
			sum2 := yBiasRound + yk0*int(r0[x+2]) + yk1*int(r1[x+2]) + yk2*int(r2[x+2]) + yk3*int(r3[x+2]) +
				yk4*int(r4[x+2]) + yk5*int(r5[x+2]) + yk6*int(r6[x+2]) + yk7*int(r7[x+2])
			sum3 := yBiasRound + yk0*int(r0[x+3]) + yk1*int(r1[x+3]) + yk2*int(r2[x+3]) + yk3*int(r3[x+3]) +
				yk4*int(r4[x+3]) + yk5*int(r5[x+3]) + yk6*int(r6[x+3]) + yk7*int(r7[x+3])
			dstRow[x] = clipPixel((sum0 >> round1Bits) - roundOffset)
			dstRow[x+1] = clipPixel((sum1 >> round1Bits) - roundOffset)
			dstRow[x+2] = clipPixel((sum2 >> round1Bits) - roundOffset)
			dstRow[x+3] = clipPixel((sum3 >> round1Bits) - roundOffset)
		}
		for ; x < width; x++ {
			sum := yBiasRound + yk0*int(r0[x]) + yk1*int(r1[x]) + yk2*int(r2[x]) + yk3*int(r3[x]) +
				yk4*int(r4[x]) + yk5*int(r5[x]) + yk6*int(r6[x]) + yk7*int(r7[x])
			dstRow[x] = clipPixel((sum >> round1Bits) - roundOffset)
		}
	}
}

const convolve2D8NarrowStride = 4

func convolve2D8NarrowPureGo(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16, im *narrowIM) {
	const imStride = convolve2D8NarrowStride
	foX := filterTaps/2 - 1
	foY := filterTaps/2 - 1
	imH := height + filterTaps - 1
	const xBias = 1 << (8 + filterBits - 1)

	xk0, xk1, xk2, xk3, xk4, xk5, xk6, xk7 := int(xKernel[0]), int(xKernel[1]), int(xKernel[2]), int(xKernel[3]), int(xKernel[4]), int(xKernel[5]), int(xKernel[6]), int(xKernel[7])
	xFourTap := xk0 == 0 && xk1 == 0 && xk6 == 0 && xk7 == 0
	for y := range imH {
		srcRow := ref.Pix[(refY-foY+y)*ref.Stride+refX-foX:]
		imRow := im[y*imStride:]
		if xFourTap {
			for x := range width {
				s := srcRow[x : x+filterTaps]
				sum := xBias + xk2*int(s[2]) + xk3*int(s[3]) + xk4*int(s[4]) + xk5*int(s[5])
				imRow[x] = int16(roundPowerOfTwo(sum, round0Bits))
			}
			continue
		}
		for x := range width {
			s := srcRow[x : x+filterTaps]
			sum := xBias + xk0*int(s[0]) + xk1*int(s[1]) + xk2*int(s[2]) + xk3*int(s[3]) +
				xk4*int(s[4]) + xk5*int(s[5]) + xk6*int(s[6]) + xk7*int(s[7])
			imRow[x] = int16(roundPowerOfTwo(sum, round0Bits))
		}
	}
	offsetBits := 8 + 2*filterBits - round0Bits
	roundOffset := (1 << (offsetBits - round1Bits)) + (1 << (offsetBits - round1Bits - 1))
	bits := 2*filterBits - round0Bits - round1Bits
	yBias := 1 << offsetBits

	yk0, yk1, yk2, yk3, yk4, yk5, yk6, yk7 := int(yKernel[0]), int(yKernel[1]), int(yKernel[2]), int(yKernel[3]), int(yKernel[4]), int(yKernel[5]), int(yKernel[6]), int(yKernel[7])
	yFourTap := yk0 == 0 && yk1 == 0 && yk6 == 0 && yk7 == 0
	for y := range height {
		dstRow := dst.Pix[(dstY+y)*dst.Stride+dstX:]
		if yFourTap {
			col := im[(y+2)*imStride : (y+6)*imStride+width]
			for x := range width {
				sum := yBias + yk2*int(col[x]) + yk3*int(col[x+imStride]) +
					yk4*int(col[x+2*imStride]) + yk5*int(col[x+3*imStride])
				res := roundPowerOfTwo(sum, round1Bits) - roundOffset
				dstRow[x] = byte(clipPixel(roundPowerOfTwo(res, bits)))
			}
			continue
		}
		col := im[y*imStride : (y+7)*imStride+width]
		for x := range width {
			sum := yBias + yk0*int(col[x]) + yk1*int(col[x+imStride]) + yk2*int(col[x+2*imStride]) +
				yk3*int(col[x+3*imStride]) + yk4*int(col[x+4*imStride]) + yk5*int(col[x+5*imStride]) +
				yk6*int(col[x+6*imStride]) + yk7*int(col[x+7*imStride])
			res := roundPowerOfTwo(sum, round1Bits) - roundOffset
			dstRow[x] = byte(clipPixel(roundPowerOfTwo(res, bits)))
		}
	}
}

func convolve2D8ClampedPureGo(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16) {
	convolve2D8ClampedPureGoWithScratch(dst, ref, dstX, dstY, refX, refY, width, height, xKernel, yKernel, nil)
}

func convolve2D8ClampedPureGoWithScratch(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16, scratch *ConvolveScratch) {
	if width > 0 && width <= convolve2D8NarrowStride {
		if scratch != nil {
			convolve2D8ClampedNarrowPureGo(dst, ref, dstX, dstY, refX, refY, width, height, xKernel, yKernel, scratch.narrowIM())
			return
		}
		var im narrowIM
		convolve2D8ClampedNarrowPureGo(dst, ref, dstX, dstY, refX, refY, width, height, xKernel, yKernel, &im)
		return
	}
	if scratch != nil {
		convolve2D8ClampedPureGoWithIM(dst, ref, dstX, dstY, refX, refY, width, height, xKernel, yKernel, &scratch.im)
		return
	}
	var im [((maxBlockSize + filterTaps - 1) * maxBlockSize)]int16
	convolve2D8ClampedPureGoWithIM(dst, ref, dstX, dstY, refX, refY, width, height, xKernel, yKernel, &im)
}

func convolve2D8ClampedPureGoWithIM(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16, im *[(maxBlockSize + filterTaps - 1) * maxBlockSize]int16) {
	if width <= 0 || height <= 0 {
		return
	}
	const imStride = maxBlockSize
	foX := filterTaps/2 - 1
	foY := filterTaps/2 - 1
	imH := height + filterTaps - 1
	const xBiasRound = (1 << (8 + filterBits - 1)) + (1 << (round0Bits - 1))

	xk0, xk1, xk2, xk3, xk4, xk5, xk6, xk7 := int(xKernel[0]), int(xKernel[1]), int(xKernel[2]), int(xKernel[3]), int(xKernel[4]), int(xKernel[5]), int(xKernel[6]), int(xKernel[7])
	xFourTap := xk0 == 0 && xk1 == 0 && xk6 == 0 && xk7 == 0
	xSixTapAlt := !xFourTap && xk0 == 0 && xk7 == 0 && xk1 >= 0 && xk2 <= 0 && xk3 >= 0 && xk4 >= 0 && xk5 <= 0 && xk6 >= 0 && (xk1+xk3+xk4+xk6) <= 255 && (-xk2-xk5) <= 255
	firstTap := refX - foX
	xLo, xHi := clampedXInterior(firstTap, filterTaps, ref.Width, width)
	for y := range imH {
		cy := clampInt(refY-foY+y, 0, ref.Height-1)
		rowBase := cy * ref.Stride
		imRow := im[y*imStride : y*imStride+width]
		if xSixTapAlt {
			const loMsk = 0x00FF00FF00FF00FF
			pk1, pk3, pk4, pk6 := uint64(xk1), uint64(xk3), uint64(xk4), uint64(xk6)
			nk2, nk5 := uint64(-xk2), uint64(-xk5)
			for x := 0; x < xLo; x++ {
				sx := firstTap + x
				sum := xBiasRound +
					xk1*int(loadSample8ClampedRow(ref, sx+1, rowBase)) +
					xk2*int(loadSample8ClampedRow(ref, sx+2, rowBase)) +
					xk3*int(loadSample8ClampedRow(ref, sx+3, rowBase)) +
					xk4*int(loadSample8ClampedRow(ref, sx+4, rowBase)) +
					xk5*int(loadSample8ClampedRow(ref, sx+5, rowBase)) +
					xk6*int(loadSample8ClampedRow(ref, sx+6, rowBase))
				imRow[x] = int16(sum >> round0Bits)
			}
			if n := xHi - xLo; n > 0 {
				srcBase := rowBase + firstTap + xLo
				srcRow := ref.Pix[srcBase : srcBase+n+filterTaps-1]
				dstSpan := imRow[xLo:xHi]
				i := 0
				for ; i+7 < n; i += 8 {
					w0 := binary.LittleEndian.Uint64(srcRow[i+1 : i+9])
					w1 := binary.LittleEndian.Uint64(srcRow[i+6 : i+14])
					e0 := w0 & loMsk
					o0 := (w0 >> 8) & loMsk
					o1 := w1 & loMsk
					e1 := (w1 >> 8) & loMsk
					t3 := (e0 >> 16) | ((e1 >> 16) << 48)
					t4 := (o0 >> 16) | ((o1 >> 32) << 48)
					t5 := (e0 >> 32) | ((e1 >> 16) << 32)

					posE := pk1*e0 + pk3*t3 + pk4*t4 + pk6*o1
					negE := nk2*o0 + nk5*t5
					posO := pk1*o0 + pk3*t4 + pk4*t5 + pk6*e1
					negO := nk2*t3 + nk5*o1

					dstSpan[i] = int16((xBiasRound + int(uint16(posE)) - int(uint16(negE))) >> round0Bits)
					dstSpan[i+1] = int16((xBiasRound + int(uint16(posO)) - int(uint16(negO))) >> round0Bits)
					dstSpan[i+2] = int16((xBiasRound + int(uint16(posE>>16)) - int(uint16(negE>>16))) >> round0Bits)
					dstSpan[i+3] = int16((xBiasRound + int(uint16(posO>>16)) - int(uint16(negO>>16))) >> round0Bits)
					dstSpan[i+4] = int16((xBiasRound + int(uint16(posE>>32)) - int(uint16(negE>>32))) >> round0Bits)
					dstSpan[i+5] = int16((xBiasRound + int(uint16(posO>>32)) - int(uint16(negO>>32))) >> round0Bits)
					dstSpan[i+6] = int16((xBiasRound + int(uint16(posE>>48)) - int(uint16(negE>>48))) >> round0Bits)
					dstSpan[i+7] = int16((xBiasRound + int(uint16(posO>>48)) - int(uint16(negO>>48))) >> round0Bits)
				}
				for ; i < n; i++ {
					sum := xBiasRound + xk1*int(srcRow[i+1]) + xk2*int(srcRow[i+2]) + xk3*int(srcRow[i+3]) +
						xk4*int(srcRow[i+4]) + xk5*int(srcRow[i+5]) + xk6*int(srcRow[i+6])
					dstSpan[i] = int16(sum >> round0Bits)
				}
			}
			for x := xHi; x < width; x++ {
				sx := firstTap + x
				sum := xBiasRound +
					xk1*int(loadSample8ClampedRow(ref, sx+1, rowBase)) +
					xk2*int(loadSample8ClampedRow(ref, sx+2, rowBase)) +
					xk3*int(loadSample8ClampedRow(ref, sx+3, rowBase)) +
					xk4*int(loadSample8ClampedRow(ref, sx+4, rowBase)) +
					xk5*int(loadSample8ClampedRow(ref, sx+5, rowBase)) +
					xk6*int(loadSample8ClampedRow(ref, sx+6, rowBase))
				imRow[x] = int16(sum >> round0Bits)
			}
			continue
		}
		if xFourTap {
			for x := 0; x < xLo; x++ {
				sx := firstTap + x
				sum := xBiasRound +
					xk2*int(loadSample8ClampedRow(ref, sx+2, rowBase)) +
					xk3*int(loadSample8ClampedRow(ref, sx+3, rowBase)) +
					xk4*int(loadSample8ClampedRow(ref, sx+4, rowBase)) +
					xk5*int(loadSample8ClampedRow(ref, sx+5, rowBase))
				imRow[x] = int16(sum >> round0Bits)
			}
			if n := xHi - xLo; n > 0 {
				srcBase := rowBase + firstTap + xLo
				srcRow := ref.Pix[srcBase : srcBase+n+filterTaps-1]
				dstSpan := imRow[xLo:xHi]
				s2, s3, s4 := int(srcRow[2]), int(srcRow[3]), int(srcRow[4])
				i := 0
				for ; i+3 < n; i += 4 {
					s5 := int(srcRow[i+5])
					s6 := int(srcRow[i+6])
					s7 := int(srcRow[i+7])
					s8 := int(srcRow[i+8])
					dstSpan[i] = int16((xBiasRound + xk2*s2 + xk3*s3 + xk4*s4 + xk5*s5) >> round0Bits)
					dstSpan[i+1] = int16((xBiasRound + xk2*s3 + xk3*s4 + xk4*s5 + xk5*s6) >> round0Bits)
					dstSpan[i+2] = int16((xBiasRound + xk2*s4 + xk3*s5 + xk4*s6 + xk5*s7) >> round0Bits)
					dstSpan[i+3] = int16((xBiasRound + xk2*s5 + xk3*s6 + xk4*s7 + xk5*s8) >> round0Bits)
					s2, s3, s4 = s6, s7, s8
				}
				for ; i < n; i++ {
					s5 := int(srcRow[i+5])
					dstSpan[i] = int16((xBiasRound + xk2*s2 + xk3*s3 + xk4*s4 + xk5*s5) >> round0Bits)
					s2, s3, s4 = s3, s4, s5
				}
			}
			for x := xHi; x < width; x++ {
				sx := firstTap + x
				sum := xBiasRound +
					xk2*int(loadSample8ClampedRow(ref, sx+2, rowBase)) +
					xk3*int(loadSample8ClampedRow(ref, sx+3, rowBase)) +
					xk4*int(loadSample8ClampedRow(ref, sx+4, rowBase)) +
					xk5*int(loadSample8ClampedRow(ref, sx+5, rowBase))
				imRow[x] = int16(sum >> round0Bits)
			}
			continue
		}
		for x := 0; x < xLo; x++ {
			sx := firstTap + x
			sum := xBiasRound +
				xk0*int(loadSample8ClampedRow(ref, sx, rowBase)) +
				xk1*int(loadSample8ClampedRow(ref, sx+1, rowBase)) +
				xk2*int(loadSample8ClampedRow(ref, sx+2, rowBase)) +
				xk3*int(loadSample8ClampedRow(ref, sx+3, rowBase)) +
				xk4*int(loadSample8ClampedRow(ref, sx+4, rowBase)) +
				xk5*int(loadSample8ClampedRow(ref, sx+5, rowBase)) +
				xk6*int(loadSample8ClampedRow(ref, sx+6, rowBase)) +
				xk7*int(loadSample8ClampedRow(ref, sx+7, rowBase))
			imRow[x] = int16(sum >> round0Bits)
		}
		if n := xHi - xLo; n > 0 {
			srcBase := rowBase + firstTap + xLo
			srcRow := ref.Pix[srcBase : srcBase+n+filterTaps-1]
			dstSpan := imRow[xLo:xHi]
			s0, s1, s2, s3, s4, s5, s6 := int(srcRow[0]), int(srcRow[1]), int(srcRow[2]), int(srcRow[3]), int(srcRow[4]), int(srcRow[5]), int(srcRow[6])
			i := 0
			for ; i+3 < n; i += 4 {
				s7 := int(srcRow[i+7])
				s8 := int(srcRow[i+8])
				s9 := int(srcRow[i+9])
				s10 := int(srcRow[i+10])
				dstSpan[i] = int16((xBiasRound + xk0*s0 + xk1*s1 + xk2*s2 + xk3*s3 + xk4*s4 + xk5*s5 + xk6*s6 + xk7*s7) >> round0Bits)
				dstSpan[i+1] = int16((xBiasRound + xk0*s1 + xk1*s2 + xk2*s3 + xk3*s4 + xk4*s5 + xk5*s6 + xk6*s7 + xk7*s8) >> round0Bits)
				dstSpan[i+2] = int16((xBiasRound + xk0*s2 + xk1*s3 + xk2*s4 + xk3*s5 + xk4*s6 + xk5*s7 + xk6*s8 + xk7*s9) >> round0Bits)
				dstSpan[i+3] = int16((xBiasRound + xk0*s3 + xk1*s4 + xk2*s5 + xk3*s6 + xk4*s7 + xk5*s8 + xk6*s9 + xk7*s10) >> round0Bits)
				s0, s1, s2, s3, s4, s5, s6 = s4, s5, s6, s7, s8, s9, s10
			}
			for ; i < n; i++ {
				s7 := int(srcRow[i+7])
				dstSpan[i] = int16((xBiasRound + xk0*s0 + xk1*s1 + xk2*s2 + xk3*s3 + xk4*s4 + xk5*s5 + xk6*s6 + xk7*s7) >> round0Bits)
				s0, s1, s2, s3, s4, s5, s6 = s1, s2, s3, s4, s5, s6, s7
			}
		}
		for x := xHi; x < width; x++ {
			sx := firstTap + x
			sum := xBiasRound +
				xk0*int(loadSample8ClampedRow(ref, sx, rowBase)) +
				xk1*int(loadSample8ClampedRow(ref, sx+1, rowBase)) +
				xk2*int(loadSample8ClampedRow(ref, sx+2, rowBase)) +
				xk3*int(loadSample8ClampedRow(ref, sx+3, rowBase)) +
				xk4*int(loadSample8ClampedRow(ref, sx+4, rowBase)) +
				xk5*int(loadSample8ClampedRow(ref, sx+5, rowBase)) +
				xk6*int(loadSample8ClampedRow(ref, sx+6, rowBase)) +
				xk7*int(loadSample8ClampedRow(ref, sx+7, rowBase))
			imRow[x] = int16(sum >> round0Bits)
		}
	}
	const (
		offsetBits  = 8 + 2*filterBits - round0Bits
		roundOffset = (1 << (offsetBits - round1Bits)) + (1 << (offsetBits - round1Bits - 1))
		yBiasRound  = (1 << offsetBits) + (1 << (round1Bits - 1))
	)

	yk0, yk1, yk2, yk3, yk4, yk5, yk6, yk7 := int(yKernel[0]), int(yKernel[1]), int(yKernel[2]), int(yKernel[3]), int(yKernel[4]), int(yKernel[5]), int(yKernel[6]), int(yKernel[7])
	yFourTap := yk0 == 0 && yk1 == 0 && yk6 == 0 && yk7 == 0
	ySixTap := !yFourTap && yk0 == 0 && yk7 == 0
	for y := range height {
		dstBase := (dstY+y)*dst.Stride + dstX
		dstRow := dst.Pix[dstBase : dstBase+width]
		if ySixTap {
			r1 := im[(y+1)*imStride : (y+1)*imStride+width]
			r2 := im[(y+2)*imStride : (y+2)*imStride+width]
			r3 := im[(y+3)*imStride : (y+3)*imStride+width]
			r4 := im[(y+4)*imStride : (y+4)*imStride+width]
			r5 := im[(y+5)*imStride : (y+5)*imStride+width]
			r6 := im[(y+6)*imStride : (y+6)*imStride+width]
			x := 0
			for ; x+3 < width; x += 4 {
				v0 := ((yBiasRound + yk1*int(r1[x]) + yk2*int(r2[x]) + yk3*int(r3[x]) + yk4*int(r4[x]) + yk5*int(r5[x]) + yk6*int(r6[x])) >> round1Bits) - roundOffset
				v1 := ((yBiasRound + yk1*int(r1[x+1]) + yk2*int(r2[x+1]) + yk3*int(r3[x+1]) + yk4*int(r4[x+1]) + yk5*int(r5[x+1]) + yk6*int(r6[x+1])) >> round1Bits) - roundOffset
				v2 := ((yBiasRound + yk1*int(r1[x+2]) + yk2*int(r2[x+2]) + yk3*int(r3[x+2]) + yk4*int(r4[x+2]) + yk5*int(r5[x+2]) + yk6*int(r6[x+2])) >> round1Bits) - roundOffset
				v3 := ((yBiasRound + yk1*int(r1[x+3]) + yk2*int(r2[x+3]) + yk3*int(r3[x+3]) + yk4*int(r4[x+3]) + yk5*int(r5[x+3]) + yk6*int(r6[x+3])) >> round1Bits) - roundOffset
				if uint(v0|v1|v2|v3) <= 255 {
					dstRow[x] = byte(v0)
					dstRow[x+1] = byte(v1)
					dstRow[x+2] = byte(v2)
					dstRow[x+3] = byte(v3)
				} else {
					dstRow[x] = clipPixel(v0)
					dstRow[x+1] = clipPixel(v1)
					dstRow[x+2] = clipPixel(v2)
					dstRow[x+3] = clipPixel(v3)
				}
			}
			for ; x < width; x++ {
				sum := yBiasRound + yk1*int(r1[x]) + yk2*int(r2[x]) + yk3*int(r3[x]) + yk4*int(r4[x]) + yk5*int(r5[x]) + yk6*int(r6[x])
				dstRow[x] = clipPixel((sum >> round1Bits) - roundOffset)
			}
			continue
		}
		if yFourTap {
			r2 := im[(y+2)*imStride : (y+2)*imStride+width]
			r3 := im[(y+3)*imStride : (y+3)*imStride+width]
			r4 := im[(y+4)*imStride : (y+4)*imStride+width]
			r5 := im[(y+5)*imStride : (y+5)*imStride+width]
			x := 0
			for ; x+3 < width; x += 4 {
				v0 := ((yBiasRound + yk2*int(r2[x]) + yk3*int(r3[x]) + yk4*int(r4[x]) + yk5*int(r5[x])) >> round1Bits) - roundOffset
				v1 := ((yBiasRound + yk2*int(r2[x+1]) + yk3*int(r3[x+1]) + yk4*int(r4[x+1]) + yk5*int(r5[x+1])) >> round1Bits) - roundOffset
				v2 := ((yBiasRound + yk2*int(r2[x+2]) + yk3*int(r3[x+2]) + yk4*int(r4[x+2]) + yk5*int(r5[x+2])) >> round1Bits) - roundOffset
				v3 := ((yBiasRound + yk2*int(r2[x+3]) + yk3*int(r3[x+3]) + yk4*int(r4[x+3]) + yk5*int(r5[x+3])) >> round1Bits) - roundOffset
				if uint(v0|v1|v2|v3) <= 255 {
					dstRow[x] = byte(v0)
					dstRow[x+1] = byte(v1)
					dstRow[x+2] = byte(v2)
					dstRow[x+3] = byte(v3)
				} else {
					dstRow[x] = clipPixel(v0)
					dstRow[x+1] = clipPixel(v1)
					dstRow[x+2] = clipPixel(v2)
					dstRow[x+3] = clipPixel(v3)
				}
			}
			for ; x < width; x++ {
				sum := yBiasRound + yk2*int(r2[x]) + yk3*int(r3[x]) + yk4*int(r4[x]) + yk5*int(r5[x])
				dstRow[x] = clipPixel((sum >> round1Bits) - roundOffset)
			}
			continue
		}
		r0 := im[y*imStride : y*imStride+width]
		r1 := im[(y+1)*imStride : (y+1)*imStride+width]
		r2 := im[(y+2)*imStride : (y+2)*imStride+width]
		r3 := im[(y+3)*imStride : (y+3)*imStride+width]
		r4 := im[(y+4)*imStride : (y+4)*imStride+width]
		r5 := im[(y+5)*imStride : (y+5)*imStride+width]
		r6 := im[(y+6)*imStride : (y+6)*imStride+width]
		r7 := im[(y+7)*imStride : (y+7)*imStride+width]
		x := 0
		for ; x+3 < width; x += 4 {
			v0 := ((yBiasRound + yk0*int(r0[x]) + yk1*int(r1[x]) + yk2*int(r2[x]) + yk3*int(r3[x]) + yk4*int(r4[x]) + yk5*int(r5[x]) + yk6*int(r6[x]) + yk7*int(r7[x])) >> round1Bits) - roundOffset
			v1 := ((yBiasRound + yk0*int(r0[x+1]) + yk1*int(r1[x+1]) + yk2*int(r2[x+1]) + yk3*int(r3[x+1]) + yk4*int(r4[x+1]) + yk5*int(r5[x+1]) + yk6*int(r6[x+1]) + yk7*int(r7[x+1])) >> round1Bits) - roundOffset
			v2 := ((yBiasRound + yk0*int(r0[x+2]) + yk1*int(r1[x+2]) + yk2*int(r2[x+2]) + yk3*int(r3[x+2]) + yk4*int(r4[x+2]) + yk5*int(r5[x+2]) + yk6*int(r6[x+2]) + yk7*int(r7[x+2])) >> round1Bits) - roundOffset
			v3 := ((yBiasRound + yk0*int(r0[x+3]) + yk1*int(r1[x+3]) + yk2*int(r2[x+3]) + yk3*int(r3[x+3]) + yk4*int(r4[x+3]) + yk5*int(r5[x+3]) + yk6*int(r6[x+3]) + yk7*int(r7[x+3])) >> round1Bits) - roundOffset
			if uint(v0|v1|v2|v3) <= 255 {
				dstRow[x] = byte(v0)
				dstRow[x+1] = byte(v1)
				dstRow[x+2] = byte(v2)
				dstRow[x+3] = byte(v3)
			} else {
				dstRow[x] = clipPixel(v0)
				dstRow[x+1] = clipPixel(v1)
				dstRow[x+2] = clipPixel(v2)
				dstRow[x+3] = clipPixel(v3)
			}
		}
		for ; x < width; x++ {
			sum := yBiasRound + yk0*int(r0[x]) + yk1*int(r1[x]) + yk2*int(r2[x]) + yk3*int(r3[x]) +
				yk4*int(r4[x]) + yk5*int(r5[x]) + yk6*int(r6[x]) + yk7*int(r7[x])
			dstRow[x] = clipPixel((sum >> round1Bits) - roundOffset)
		}
	}
}

func convolve2D8ClampedNarrowPureGo(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16, im *narrowIM) {
	const imStride = convolve2D8NarrowStride
	foX := filterTaps/2 - 1
	foY := filterTaps/2 - 1
	imH := height + filterTaps - 1
	const xBias = 1 << (8 + filterBits - 1)

	xk0, xk1, xk2, xk3, xk4, xk5, xk6, xk7 := int(xKernel[0]), int(xKernel[1]), int(xKernel[2]), int(xKernel[3]), int(xKernel[4]), int(xKernel[5]), int(xKernel[6]), int(xKernel[7])
	xFourTap := xk0 == 0 && xk1 == 0 && xk6 == 0 && xk7 == 0
	firstTap := refX - foX
	xLo, xHi := clampedXInterior(firstTap, filterTaps, ref.Width, width)
	for y := range imH {
		cy := clampInt(refY-foY+y, 0, ref.Height-1)
		rowBase := cy * ref.Stride
		imRow := im[y*imStride:]
		if xFourTap {
			for x := 0; x < xLo; x++ {
				sx := firstTap + x
				sum := xBias +
					xk2*int(loadSample8ClampedRow(ref, sx+2, rowBase)) +
					xk3*int(loadSample8ClampedRow(ref, sx+3, rowBase)) +
					xk4*int(loadSample8ClampedRow(ref, sx+4, rowBase)) +
					xk5*int(loadSample8ClampedRow(ref, sx+5, rowBase))
				imRow[x] = int16(roundPowerOfTwo(sum, round0Bits))
			}
			if xHi > xLo {
				srcRow := ref.Pix[rowBase+firstTap+xLo:]
				for x := xLo; x < xHi; x++ {
					i := x - xLo
					s := srcRow[i : i+filterTaps]
					sum := xBias + xk2*int(s[2]) + xk3*int(s[3]) + xk4*int(s[4]) + xk5*int(s[5])
					imRow[x] = int16(roundPowerOfTwo(sum, round0Bits))
				}
			}
			for x := xHi; x < width; x++ {
				sx := firstTap + x
				sum := xBias +
					xk2*int(loadSample8ClampedRow(ref, sx+2, rowBase)) +
					xk3*int(loadSample8ClampedRow(ref, sx+3, rowBase)) +
					xk4*int(loadSample8ClampedRow(ref, sx+4, rowBase)) +
					xk5*int(loadSample8ClampedRow(ref, sx+5, rowBase))
				imRow[x] = int16(roundPowerOfTwo(sum, round0Bits))
			}
			continue
		}
		for x := 0; x < xLo; x++ {
			sx := firstTap + x
			sum := xBias +
				xk0*int(loadSample8ClampedRow(ref, sx, rowBase)) +
				xk1*int(loadSample8ClampedRow(ref, sx+1, rowBase)) +
				xk2*int(loadSample8ClampedRow(ref, sx+2, rowBase)) +
				xk3*int(loadSample8ClampedRow(ref, sx+3, rowBase)) +
				xk4*int(loadSample8ClampedRow(ref, sx+4, rowBase)) +
				xk5*int(loadSample8ClampedRow(ref, sx+5, rowBase)) +
				xk6*int(loadSample8ClampedRow(ref, sx+6, rowBase)) +
				xk7*int(loadSample8ClampedRow(ref, sx+7, rowBase))
			imRow[x] = int16(roundPowerOfTwo(sum, round0Bits))
		}
		if xHi > xLo {
			srcRow := ref.Pix[rowBase+firstTap+xLo:]
			for x := xLo; x < xHi; x++ {
				i := x - xLo
				s := srcRow[i : i+filterTaps]
				sum := xBias + xk0*int(s[0]) + xk1*int(s[1]) + xk2*int(s[2]) + xk3*int(s[3]) +
					xk4*int(s[4]) + xk5*int(s[5]) + xk6*int(s[6]) + xk7*int(s[7])
				imRow[x] = int16(roundPowerOfTwo(sum, round0Bits))
			}
		}
		for x := xHi; x < width; x++ {
			sx := firstTap + x
			sum := xBias +
				xk0*int(loadSample8ClampedRow(ref, sx, rowBase)) +
				xk1*int(loadSample8ClampedRow(ref, sx+1, rowBase)) +
				xk2*int(loadSample8ClampedRow(ref, sx+2, rowBase)) +
				xk3*int(loadSample8ClampedRow(ref, sx+3, rowBase)) +
				xk4*int(loadSample8ClampedRow(ref, sx+4, rowBase)) +
				xk5*int(loadSample8ClampedRow(ref, sx+5, rowBase)) +
				xk6*int(loadSample8ClampedRow(ref, sx+6, rowBase)) +
				xk7*int(loadSample8ClampedRow(ref, sx+7, rowBase))
			imRow[x] = int16(roundPowerOfTwo(sum, round0Bits))
		}
	}
	offsetBits := 8 + 2*filterBits - round0Bits
	roundOffset := (1 << (offsetBits - round1Bits)) + (1 << (offsetBits - round1Bits - 1))
	bits := 2*filterBits - round0Bits - round1Bits
	yBias := 1 << offsetBits

	yk0, yk1, yk2, yk3, yk4, yk5, yk6, yk7 := int(yKernel[0]), int(yKernel[1]), int(yKernel[2]), int(yKernel[3]), int(yKernel[4]), int(yKernel[5]), int(yKernel[6]), int(yKernel[7])
	yFourTap := yk0 == 0 && yk1 == 0 && yk6 == 0 && yk7 == 0
	for y := range height {
		dstRow := dst.Pix[(dstY+y)*dst.Stride+dstX:]
		if yFourTap {
			col := im[(y+2)*imStride : (y+6)*imStride+width]
			for x := range width {
				sum := yBias + yk2*int(col[x]) + yk3*int(col[x+imStride]) +
					yk4*int(col[x+2*imStride]) + yk5*int(col[x+3*imStride])
				res := roundPowerOfTwo(sum, round1Bits) - roundOffset
				dstRow[x] = byte(clipPixel(roundPowerOfTwo(res, bits)))
			}
			continue
		}
		col := im[y*imStride : (y+7)*imStride+width]
		for x := range width {
			sum := yBias + yk0*int(col[x]) + yk1*int(col[x+imStride]) + yk2*int(col[x+2*imStride]) +
				yk3*int(col[x+3*imStride]) + yk4*int(col[x+4*imStride]) + yk5*int(col[x+5*imStride]) +
				yk6*int(col[x+6*imStride]) + yk7*int(col[x+7*imStride])
			res := roundPowerOfTwo(sum, round1Bits) - roundOffset
			dstRow[x] = byte(clipPixel(roundPowerOfTwo(res, bits)))
		}
	}
}

func convolveXHighBDPureGo(dst frame.Plane, ref frame.Plane, bitDepth uint8, max uint16, dstX int, dstY int, refX int, refY int, width int, height int, kernel [filterTaps]int16) {
	if width <= 0 || height <= 0 {
		return
	}
	fo := filterTaps/2 - 1
	// av1_highbd_convolve_x_sr_c: round_0 then bits = FILTER_BITS - round_0.
	// round_0 is bd-dependent (5 at bd == 12), matching get_conv_params_no_round.
	round0, _ := highBDRoundBits(bitDepth)
	bits := filterBits - round0
	r0Bias := 1 << (round0 - 1)
	r0Shift := uint(round0)
	bBias := 1 << (bits - 1)
	bShift := uint(bits)
	k0, k1, k2, k3, k4, k5, k6, k7 := int(kernel[0]), int(kernel[1]), int(kernel[2]), int(kernel[3]), int(kernel[4]), int(kernel[5]), int(kernel[6]), int(kernel[7])
	fourTap := k0 == 0 && k1 == 0 && k6 == 0 && k7 == 0
	for y := range height {
		base := (refY+y)*ref.Stride + (refX-fo)*2
		srcRow := ref.Pix[base : base+(width+filterTaps-1)*2]
		dstBase := (dstY+y)*dst.Stride + dstX*2
		dstRow := dst.Pix[dstBase : dstBase+width*2]
		if fourTap {
			s2 := int(binary.LittleEndian.Uint16(srcRow[4:6]))
			s3 := int(binary.LittleEndian.Uint16(srcRow[6:8]))
			s4 := int(binary.LittleEndian.Uint16(srcRow[8:10]))
			x := 0
			for ; x+3 < width; x += 4 {
				o := x * 2
				s5 := int(binary.LittleEndian.Uint16(srcRow[o+10 : o+12]))
				s6 := int(binary.LittleEndian.Uint16(srcRow[o+12 : o+14]))
				s7 := int(binary.LittleEndian.Uint16(srcRow[o+14 : o+16]))
				s8 := int(binary.LittleEndian.Uint16(srcRow[o+16 : o+18]))
				binary.LittleEndian.PutUint16(dstRow[o:o+2], clipPixelHighBD((((k2*s2+k3*s3+k4*s4+k5*s5+r0Bias)>>r0Shift)+bBias)>>bShift, max))
				binary.LittleEndian.PutUint16(dstRow[o+2:o+4], clipPixelHighBD((((k2*s3+k3*s4+k4*s5+k5*s6+r0Bias)>>r0Shift)+bBias)>>bShift, max))
				binary.LittleEndian.PutUint16(dstRow[o+4:o+6], clipPixelHighBD((((k2*s4+k3*s5+k4*s6+k5*s7+r0Bias)>>r0Shift)+bBias)>>bShift, max))
				binary.LittleEndian.PutUint16(dstRow[o+6:o+8], clipPixelHighBD((((k2*s5+k3*s6+k4*s7+k5*s8+r0Bias)>>r0Shift)+bBias)>>bShift, max))
				s2, s3, s4 = s6, s7, s8
			}
			for ; x < width; x++ {
				o := x * 2
				s5 := int(binary.LittleEndian.Uint16(srcRow[o+10 : o+12]))
				binary.LittleEndian.PutUint16(dstRow[o:o+2], clipPixelHighBD((((k2*s2+k3*s3+k4*s4+k5*s5+r0Bias)>>r0Shift)+bBias)>>bShift, max))
				s2, s3, s4 = s3, s4, s5
			}
			continue
		}
		s0 := int(binary.LittleEndian.Uint16(srcRow[0:2]))
		s1 := int(binary.LittleEndian.Uint16(srcRow[2:4]))
		s2 := int(binary.LittleEndian.Uint16(srcRow[4:6]))
		s3 := int(binary.LittleEndian.Uint16(srcRow[6:8]))
		s4 := int(binary.LittleEndian.Uint16(srcRow[8:10]))
		s5 := int(binary.LittleEndian.Uint16(srcRow[10:12]))
		s6 := int(binary.LittleEndian.Uint16(srcRow[12:14]))
		x := 0
		for ; x+3 < width; x += 4 {
			o := x * 2
			s7 := int(binary.LittleEndian.Uint16(srcRow[o+14 : o+16]))
			s8 := int(binary.LittleEndian.Uint16(srcRow[o+16 : o+18]))
			s9 := int(binary.LittleEndian.Uint16(srcRow[o+18 : o+20]))
			s10 := int(binary.LittleEndian.Uint16(srcRow[o+20 : o+22]))
			sum0 := k0*s0 + k1*s1 + k2*s2 + k3*s3 + k4*s4 + k5*s5 + k6*s6 + k7*s7 + r0Bias
			sum1 := k0*s1 + k1*s2 + k2*s3 + k3*s4 + k4*s5 + k5*s6 + k6*s7 + k7*s8 + r0Bias
			sum2 := k0*s2 + k1*s3 + k2*s4 + k3*s5 + k4*s6 + k5*s7 + k6*s8 + k7*s9 + r0Bias
			sum3 := k0*s3 + k1*s4 + k2*s5 + k3*s6 + k4*s7 + k5*s8 + k6*s9 + k7*s10 + r0Bias
			binary.LittleEndian.PutUint16(dstRow[o:o+2], clipPixelHighBD(((sum0>>r0Shift)+bBias)>>bShift, max))
			binary.LittleEndian.PutUint16(dstRow[o+2:o+4], clipPixelHighBD(((sum1>>r0Shift)+bBias)>>bShift, max))
			binary.LittleEndian.PutUint16(dstRow[o+4:o+6], clipPixelHighBD(((sum2>>r0Shift)+bBias)>>bShift, max))
			binary.LittleEndian.PutUint16(dstRow[o+6:o+8], clipPixelHighBD(((sum3>>r0Shift)+bBias)>>bShift, max))
			s0, s1, s2, s3, s4, s5, s6 = s4, s5, s6, s7, s8, s9, s10
		}
		for ; x < width; x++ {
			o := x * 2
			s7 := int(binary.LittleEndian.Uint16(srcRow[o+14 : o+16]))
			sum := k0*s0 + k1*s1 + k2*s2 + k3*s3 + k4*s4 + k5*s5 + k6*s6 + k7*s7 + r0Bias
			binary.LittleEndian.PutUint16(dstRow[o:o+2], clipPixelHighBD(((sum>>r0Shift)+bBias)>>bShift, max))
			s0, s1, s2, s3, s4, s5, s6 = s1, s2, s3, s4, s5, s6, s7
		}
	}
}

func convolveXHighBDClampedPureGo(dst frame.Plane, ref frame.Plane, bitDepth uint8, max uint16, dstX int, dstY int, refX int, refY int, width int, height int, kernel [filterTaps]int16) {
	fo := filterTaps/2 - 1
	round0, _ := highBDRoundBits(bitDepth)
	bits := filterBits - round0
	k0, k1, k2, k3, k4, k5, k6, k7 := int(kernel[0]), int(kernel[1]), int(kernel[2]), int(kernel[3]), int(kernel[4]), int(kernel[5]), int(kernel[6]), int(kernel[7])
	fourTap := k0 == 0 && k1 == 0 && k6 == 0 && k7 == 0
	for y := range height {
		sy := refY + y
		if fourTap {
			for x := range width {
				sx := refX + x - fo
				sum := k2*int(loadHighBDSampleClamped(ref, sx+2, sy)) +
					k3*int(loadHighBDSampleClamped(ref, sx+3, sy)) +
					k4*int(loadHighBDSampleClamped(ref, sx+4, sy)) +
					k5*int(loadHighBDSampleClamped(ref, sx+5, sy))
				res := roundPowerOfTwo(sum, round0)
				storeHighBDSample(dst, dstX+x, dstY+y, clipPixelHighBD(roundPowerOfTwo(res, bits), max))
			}
			continue
		}
		for x := range width {
			sx := refX + x - fo
			sum := k0*int(loadHighBDSampleClamped(ref, sx, sy)) +
				k1*int(loadHighBDSampleClamped(ref, sx+1, sy)) +
				k2*int(loadHighBDSampleClamped(ref, sx+2, sy)) +
				k3*int(loadHighBDSampleClamped(ref, sx+3, sy)) +
				k4*int(loadHighBDSampleClamped(ref, sx+4, sy)) +
				k5*int(loadHighBDSampleClamped(ref, sx+5, sy)) +
				k6*int(loadHighBDSampleClamped(ref, sx+6, sy)) +
				k7*int(loadHighBDSampleClamped(ref, sx+7, sy))
			res := roundPowerOfTwo(sum, round0)
			storeHighBDSample(dst, dstX+x, dstY+y, clipPixelHighBD(roundPowerOfTwo(res, bits), max))
		}
	}
}

func convolveYHighBDPureGo(dst frame.Plane, ref frame.Plane, bitDepth uint8, max uint16, dstX int, dstY int, refX int, refY int, width int, height int, kernel [filterTaps]int16) {
	_ = bitDepth
	fo := filterTaps/2 - 1
	stride := ref.Stride
	k0, k1, k2, k3, k4, k5, k6, k7 := int(kernel[0]), int(kernel[1]), int(kernel[2]), int(kernel[3]), int(kernel[4]), int(kernel[5]), int(kernel[6]), int(kernel[7])
	fourTap := k0 == 0 && k1 == 0 && k6 == 0 && k7 == 0
	for y := range height {
		base := (refY+y-fo)*stride + refX*2
		if fourTap {
			b := base + 2*stride
			for x := range width {
				o := b + x*2
				s2 := int(uint16(ref.Pix[o]) | uint16(ref.Pix[o+1])<<8)
				s3 := int(uint16(ref.Pix[o+stride]) | uint16(ref.Pix[o+stride+1])<<8)
				s4 := int(uint16(ref.Pix[o+2*stride]) | uint16(ref.Pix[o+2*stride+1])<<8)
				s5 := int(uint16(ref.Pix[o+3*stride]) | uint16(ref.Pix[o+3*stride+1])<<8)
				sum := k2*s2 + k3*s3 + k4*s4 + k5*s5
				storeHighBDSample(dst, dstX+x, dstY+y, clipPixelHighBD(roundPowerOfTwo(sum, filterBits), max))
			}
			continue
		}
		for x := range width {
			o := base + x*2
			s0 := int(uint16(ref.Pix[o]) | uint16(ref.Pix[o+1])<<8)
			s1 := int(uint16(ref.Pix[o+stride]) | uint16(ref.Pix[o+stride+1])<<8)
			s2 := int(uint16(ref.Pix[o+2*stride]) | uint16(ref.Pix[o+2*stride+1])<<8)
			s3 := int(uint16(ref.Pix[o+3*stride]) | uint16(ref.Pix[o+3*stride+1])<<8)
			s4 := int(uint16(ref.Pix[o+4*stride]) | uint16(ref.Pix[o+4*stride+1])<<8)
			s5 := int(uint16(ref.Pix[o+5*stride]) | uint16(ref.Pix[o+5*stride+1])<<8)
			s6 := int(uint16(ref.Pix[o+6*stride]) | uint16(ref.Pix[o+6*stride+1])<<8)
			s7 := int(uint16(ref.Pix[o+7*stride]) | uint16(ref.Pix[o+7*stride+1])<<8)
			sum := k0*s0 + k1*s1 + k2*s2 + k3*s3 + k4*s4 + k5*s5 + k6*s6 + k7*s7
			storeHighBDSample(dst, dstX+x, dstY+y, clipPixelHighBD(roundPowerOfTwo(sum, filterBits), max))
		}
	}
}

func convolveYHighBDClampedPureGo(dst frame.Plane, ref frame.Plane, bitDepth uint8, max uint16, dstX int, dstY int, refX int, refY int, width int, height int, kernel [filterTaps]int16) {
	_ = bitDepth
	fo := filterTaps/2 - 1
	k0, k1, k2, k3, k4, k5, k6, k7 := int(kernel[0]), int(kernel[1]), int(kernel[2]), int(kernel[3]), int(kernel[4]), int(kernel[5]), int(kernel[6]), int(kernel[7])
	fourTap := k0 == 0 && k1 == 0 && k6 == 0 && k7 == 0
	for y := range height {
		sy := refY + y - fo
		if fourTap {
			for x := range width {
				sx := refX + x
				sum := k2*int(loadHighBDSampleClamped(ref, sx, sy+2)) +
					k3*int(loadHighBDSampleClamped(ref, sx, sy+3)) +
					k4*int(loadHighBDSampleClamped(ref, sx, sy+4)) +
					k5*int(loadHighBDSampleClamped(ref, sx, sy+5))
				storeHighBDSample(dst, dstX+x, dstY+y, clipPixelHighBD(roundPowerOfTwo(sum, filterBits), max))
			}
			continue
		}
		for x := range width {
			sx := refX + x
			sum := k0*int(loadHighBDSampleClamped(ref, sx, sy)) +
				k1*int(loadHighBDSampleClamped(ref, sx, sy+1)) +
				k2*int(loadHighBDSampleClamped(ref, sx, sy+2)) +
				k3*int(loadHighBDSampleClamped(ref, sx, sy+3)) +
				k4*int(loadHighBDSampleClamped(ref, sx, sy+4)) +
				k5*int(loadHighBDSampleClamped(ref, sx, sy+5)) +
				k6*int(loadHighBDSampleClamped(ref, sx, sy+6)) +
				k7*int(loadHighBDSampleClamped(ref, sx, sy+7))
			storeHighBDSample(dst, dstX+x, dstY+y, clipPixelHighBD(roundPowerOfTwo(sum, filterBits), max))
		}
	}
}

func convolve2DHighBDPureGo(dst frame.Plane, ref frame.Plane, bitDepth uint8, max uint16, dstX int, dstY int, refX int, refY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16) {
	var im [((maxBlockSize + filterTaps - 1) * maxBlockSize)]int32
	convolve2DHighBDPureGoWithIM(dst, ref, bitDepth, max, dstX, dstY, refX, refY, width, height, xKernel, yKernel, &im)
}

// convolve2DHighBDPureGoWithScratch is convolve2DHighBDPureGo using optional
// caller-owned scratch for the large int32 intermediate block, so hot decode
// paths do not zero-fill a ~69KB stack array per call.
func convolve2DHighBDPureGoWithScratch(dst frame.Plane, ref frame.Plane, bitDepth uint8, max uint16, dstX int, dstY int, refX int, refY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16, scratch *ConvolveScratch) {
	if scratch != nil {
		convolve2DHighBDPureGoWithIM(dst, ref, bitDepth, max, dstX, dstY, refX, refY, width, height, xKernel, yKernel, &scratch.imHBD)
		return
	}
	convolve2DHighBDPureGo(dst, ref, bitDepth, max, dstX, dstY, refX, refY, width, height, xKernel, yKernel)
}

func convolve2DHighBDPureGoWithIM(dst frame.Plane, ref frame.Plane, bitDepth uint8, max uint16, dstX int, dstY int, refX int, refY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16, im *[(maxBlockSize + filterTaps - 1) * maxBlockSize]int32) {
	if width <= 0 || height <= 0 {
		return
	}
	const imStride = maxBlockSize
	foX := filterTaps/2 - 1
	foY := filterTaps/2 - 1
	imH := height + filterTaps - 1
	round0, round1 := highBDRoundBits(bitDepth)
	r0Shift := uint(round0)
	xBiasRound := (1 << (int(bitDepth) + filterBits - 1)) + (1 << (round0 - 1))

	xk0, xk1, xk2, xk3, xk4, xk5, xk6, xk7 := int(xKernel[0]), int(xKernel[1]), int(xKernel[2]), int(xKernel[3]), int(xKernel[4]), int(xKernel[5]), int(xKernel[6]), int(xKernel[7])
	xFourTap := xk0 == 0 && xk1 == 0 && xk6 == 0 && xk7 == 0
	for y := range imH {
		base := (refY-foY+y)*ref.Stride + (refX-foX)*2
		srcRow := ref.Pix[base : base+(width+filterTaps-1)*2]
		imRow := im[y*imStride : y*imStride+width]
		if xFourTap {
			s2 := int(binary.LittleEndian.Uint16(srcRow[4:6]))
			s3 := int(binary.LittleEndian.Uint16(srcRow[6:8]))
			s4 := int(binary.LittleEndian.Uint16(srcRow[8:10]))
			x := 0
			for ; x+3 < width; x += 4 {
				o := x * 2
				s5 := int(binary.LittleEndian.Uint16(srcRow[o+10 : o+12]))
				s6 := int(binary.LittleEndian.Uint16(srcRow[o+12 : o+14]))
				s7 := int(binary.LittleEndian.Uint16(srcRow[o+14 : o+16]))
				s8 := int(binary.LittleEndian.Uint16(srcRow[o+16 : o+18]))
				imRow[x] = int32((xBiasRound + xk2*s2 + xk3*s3 + xk4*s4 + xk5*s5) >> r0Shift)
				imRow[x+1] = int32((xBiasRound + xk2*s3 + xk3*s4 + xk4*s5 + xk5*s6) >> r0Shift)
				imRow[x+2] = int32((xBiasRound + xk2*s4 + xk3*s5 + xk4*s6 + xk5*s7) >> r0Shift)
				imRow[x+3] = int32((xBiasRound + xk2*s5 + xk3*s6 + xk4*s7 + xk5*s8) >> r0Shift)
				s2, s3, s4 = s6, s7, s8
			}
			for ; x < width; x++ {
				o := x * 2
				s5 := int(binary.LittleEndian.Uint16(srcRow[o+10 : o+12]))
				imRow[x] = int32((xBiasRound + xk2*s2 + xk3*s3 + xk4*s4 + xk5*s5) >> r0Shift)
				s2, s3, s4 = s3, s4, s5
			}
			continue
		}
		s0 := int(binary.LittleEndian.Uint16(srcRow[0:2]))
		s1 := int(binary.LittleEndian.Uint16(srcRow[2:4]))
		s2 := int(binary.LittleEndian.Uint16(srcRow[4:6]))
		s3 := int(binary.LittleEndian.Uint16(srcRow[6:8]))
		s4 := int(binary.LittleEndian.Uint16(srcRow[8:10]))
		s5 := int(binary.LittleEndian.Uint16(srcRow[10:12]))
		s6 := int(binary.LittleEndian.Uint16(srcRow[12:14]))
		x := 0
		for ; x+3 < width; x += 4 {
			o := x * 2
			s7 := int(binary.LittleEndian.Uint16(srcRow[o+14 : o+16]))
			s8 := int(binary.LittleEndian.Uint16(srcRow[o+16 : o+18]))
			s9 := int(binary.LittleEndian.Uint16(srcRow[o+18 : o+20]))
			s10 := int(binary.LittleEndian.Uint16(srcRow[o+20 : o+22]))
			sum0 := xBiasRound + xk0*s0 + xk1*s1 + xk2*s2 + xk3*s3 + xk4*s4 + xk5*s5 + xk6*s6 + xk7*s7
			sum1 := xBiasRound + xk0*s1 + xk1*s2 + xk2*s3 + xk3*s4 + xk4*s5 + xk5*s6 + xk6*s7 + xk7*s8
			sum2 := xBiasRound + xk0*s2 + xk1*s3 + xk2*s4 + xk3*s5 + xk4*s6 + xk5*s7 + xk6*s8 + xk7*s9
			sum3 := xBiasRound + xk0*s3 + xk1*s4 + xk2*s5 + xk3*s6 + xk4*s7 + xk5*s8 + xk6*s9 + xk7*s10
			imRow[x] = int32(sum0 >> r0Shift)
			imRow[x+1] = int32(sum1 >> r0Shift)
			imRow[x+2] = int32(sum2 >> r0Shift)
			imRow[x+3] = int32(sum3 >> r0Shift)
			s0, s1, s2, s3, s4, s5, s6 = s4, s5, s6, s7, s8, s9, s10
		}
		for ; x < width; x++ {
			o := x * 2
			s7 := int(binary.LittleEndian.Uint16(srcRow[o+14 : o+16]))
			sum := xBiasRound + xk0*s0 + xk1*s1 + xk2*s2 + xk3*s3 + xk4*s4 + xk5*s5 + xk6*s6 + xk7*s7
			imRow[x] = int32(sum >> r0Shift)
			s0, s1, s2, s3, s4, s5, s6 = s1, s2, s3, s4, s5, s6, s7
		}
	}
	offsetBits := int(bitDepth) + 2*filterBits - round0
	roundOffset := (1 << (offsetBits - round1)) + (1 << (offsetBits - round1 - 1))
	r1Shift := uint(round1)
	yBiasRound := (1 << offsetBits) + (1 << (round1 - 1))

	yk0, yk1, yk2, yk3, yk4, yk5, yk6, yk7 := int(yKernel[0]), int(yKernel[1]), int(yKernel[2]), int(yKernel[3]), int(yKernel[4]), int(yKernel[5]), int(yKernel[6]), int(yKernel[7])
	yFourTap := yk0 == 0 && yk1 == 0 && yk6 == 0 && yk7 == 0
	for y := range height {
		dstBase := (dstY+y)*dst.Stride + dstX*2
		dstRow := dst.Pix[dstBase : dstBase+width*2]
		if yFourTap {
			r2 := im[(y+2)*imStride : (y+2)*imStride+width]
			r3 := im[(y+3)*imStride : (y+3)*imStride+width]
			r4 := im[(y+4)*imStride : (y+4)*imStride+width]
			r5 := im[(y+5)*imStride : (y+5)*imStride+width]
			x := 0
			for ; x+3 < width; x += 4 {
				o := x * 2
				sum0 := yBiasRound + yk2*int(r2[x]) + yk3*int(r3[x]) + yk4*int(r4[x]) + yk5*int(r5[x])
				sum1 := yBiasRound + yk2*int(r2[x+1]) + yk3*int(r3[x+1]) + yk4*int(r4[x+1]) + yk5*int(r5[x+1])
				sum2 := yBiasRound + yk2*int(r2[x+2]) + yk3*int(r3[x+2]) + yk4*int(r4[x+2]) + yk5*int(r5[x+2])
				sum3 := yBiasRound + yk2*int(r2[x+3]) + yk3*int(r3[x+3]) + yk4*int(r4[x+3]) + yk5*int(r5[x+3])
				binary.LittleEndian.PutUint16(dstRow[o:o+2], clipPixelHighBD((sum0>>r1Shift)-roundOffset, max))
				binary.LittleEndian.PutUint16(dstRow[o+2:o+4], clipPixelHighBD((sum1>>r1Shift)-roundOffset, max))
				binary.LittleEndian.PutUint16(dstRow[o+4:o+6], clipPixelHighBD((sum2>>r1Shift)-roundOffset, max))
				binary.LittleEndian.PutUint16(dstRow[o+6:o+8], clipPixelHighBD((sum3>>r1Shift)-roundOffset, max))
			}
			for ; x < width; x++ {
				o := x * 2
				sum := yBiasRound + yk2*int(r2[x]) + yk3*int(r3[x]) + yk4*int(r4[x]) + yk5*int(r5[x])
				binary.LittleEndian.PutUint16(dstRow[o:o+2], clipPixelHighBD((sum>>r1Shift)-roundOffset, max))
			}
			continue
		}
		r0 := im[y*imStride : y*imStride+width]
		r1 := im[(y+1)*imStride : (y+1)*imStride+width]
		r2 := im[(y+2)*imStride : (y+2)*imStride+width]
		r3 := im[(y+3)*imStride : (y+3)*imStride+width]
		r4 := im[(y+4)*imStride : (y+4)*imStride+width]
		r5 := im[(y+5)*imStride : (y+5)*imStride+width]
		r6 := im[(y+6)*imStride : (y+6)*imStride+width]
		r7 := im[(y+7)*imStride : (y+7)*imStride+width]
		x := 0
		for ; x+3 < width; x += 4 {
			o := x * 2
			sum0 := yBiasRound + yk0*int(r0[x]) + yk1*int(r1[x]) + yk2*int(r2[x]) + yk3*int(r3[x]) +
				yk4*int(r4[x]) + yk5*int(r5[x]) + yk6*int(r6[x]) + yk7*int(r7[x])
			sum1 := yBiasRound + yk0*int(r0[x+1]) + yk1*int(r1[x+1]) + yk2*int(r2[x+1]) + yk3*int(r3[x+1]) +
				yk4*int(r4[x+1]) + yk5*int(r5[x+1]) + yk6*int(r6[x+1]) + yk7*int(r7[x+1])
			sum2 := yBiasRound + yk0*int(r0[x+2]) + yk1*int(r1[x+2]) + yk2*int(r2[x+2]) + yk3*int(r3[x+2]) +
				yk4*int(r4[x+2]) + yk5*int(r5[x+2]) + yk6*int(r6[x+2]) + yk7*int(r7[x+2])
			sum3 := yBiasRound + yk0*int(r0[x+3]) + yk1*int(r1[x+3]) + yk2*int(r2[x+3]) + yk3*int(r3[x+3]) +
				yk4*int(r4[x+3]) + yk5*int(r5[x+3]) + yk6*int(r6[x+3]) + yk7*int(r7[x+3])
			binary.LittleEndian.PutUint16(dstRow[o:o+2], clipPixelHighBD((sum0>>r1Shift)-roundOffset, max))
			binary.LittleEndian.PutUint16(dstRow[o+2:o+4], clipPixelHighBD((sum1>>r1Shift)-roundOffset, max))
			binary.LittleEndian.PutUint16(dstRow[o+4:o+6], clipPixelHighBD((sum2>>r1Shift)-roundOffset, max))
			binary.LittleEndian.PutUint16(dstRow[o+6:o+8], clipPixelHighBD((sum3>>r1Shift)-roundOffset, max))
		}
		for ; x < width; x++ {
			o := x * 2
			sum := yBiasRound + yk0*int(r0[x]) + yk1*int(r1[x]) + yk2*int(r2[x]) + yk3*int(r3[x]) +
				yk4*int(r4[x]) + yk5*int(r5[x]) + yk6*int(r6[x]) + yk7*int(r7[x])
			binary.LittleEndian.PutUint16(dstRow[o:o+2], clipPixelHighBD((sum>>r1Shift)-roundOffset, max))
		}
	}
}

func convolve2DHighBDClampedPureGo(dst frame.Plane, ref frame.Plane, bitDepth uint8, max uint16, dstX int, dstY int, refX int, refY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16) {
	var im [((maxBlockSize + filterTaps - 1) * maxBlockSize)]int32
	convolve2DHighBDClampedPureGoWithIM(dst, ref, bitDepth, max, dstX, dstY, refX, refY, width, height, xKernel, yKernel, &im)
}

// convolve2DHighBDClampedPureGoWithScratch is convolve2DHighBDClampedPureGo
// using optional caller-owned scratch for the large int32 intermediate block.
func convolve2DHighBDClampedPureGoWithScratch(dst frame.Plane, ref frame.Plane, bitDepth uint8, max uint16, dstX int, dstY int, refX int, refY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16, scratch *ConvolveScratch) {
	if scratch != nil {
		convolve2DHighBDClampedPureGoWithIM(dst, ref, bitDepth, max, dstX, dstY, refX, refY, width, height, xKernel, yKernel, &scratch.imHBD)
		return
	}
	convolve2DHighBDClampedPureGo(dst, ref, bitDepth, max, dstX, dstY, refX, refY, width, height, xKernel, yKernel)
}

func convolve2DHighBDClampedPureGoWithIM(dst frame.Plane, ref frame.Plane, bitDepth uint8, max uint16, dstX int, dstY int, refX int, refY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16, im *[(maxBlockSize + filterTaps - 1) * maxBlockSize]int32) {
	const imStride = maxBlockSize
	foX := filterTaps/2 - 1
	foY := filterTaps/2 - 1
	imH := height + filterTaps - 1
	round0, round1 := highBDRoundBits(bitDepth)
	xBias := 1 << (int(bitDepth) + filterBits - 1)

	xk0, xk1, xk2, xk3, xk4, xk5, xk6, xk7 := int(xKernel[0]), int(xKernel[1]), int(xKernel[2]), int(xKernel[3]), int(xKernel[4]), int(xKernel[5]), int(xKernel[6]), int(xKernel[7])
	xFourTap := xk0 == 0 && xk1 == 0 && xk6 == 0 && xk7 == 0
	for y := range imH {
		sy := refY - foY + y
		imRow := im[y*imStride:]
		if xFourTap {
			for x := range width {
				sx := refX + x - foX
				sum := xBias +
					xk2*int(loadHighBDSampleClamped(ref, sx+2, sy)) +
					xk3*int(loadHighBDSampleClamped(ref, sx+3, sy)) +
					xk4*int(loadHighBDSampleClamped(ref, sx+4, sy)) +
					xk5*int(loadHighBDSampleClamped(ref, sx+5, sy))
				imRow[x] = int32(roundPowerOfTwo(sum, round0))
			}
			continue
		}
		for x := range width {
			sx := refX + x - foX
			sum := xBias +
				xk0*int(loadHighBDSampleClamped(ref, sx, sy)) +
				xk1*int(loadHighBDSampleClamped(ref, sx+1, sy)) +
				xk2*int(loadHighBDSampleClamped(ref, sx+2, sy)) +
				xk3*int(loadHighBDSampleClamped(ref, sx+3, sy)) +
				xk4*int(loadHighBDSampleClamped(ref, sx+4, sy)) +
				xk5*int(loadHighBDSampleClamped(ref, sx+5, sy)) +
				xk6*int(loadHighBDSampleClamped(ref, sx+6, sy)) +
				xk7*int(loadHighBDSampleClamped(ref, sx+7, sy))
			imRow[x] = int32(roundPowerOfTwo(sum, round0))
		}
	}
	offsetBits := int(bitDepth) + 2*filterBits - round0
	roundOffset := (1 << (offsetBits - round1)) + (1 << (offsetBits - round1 - 1))
	bits := 2*filterBits - round0 - round1
	yBias := 1 << offsetBits

	yk0, yk1, yk2, yk3, yk4, yk5, yk6, yk7 := int(yKernel[0]), int(yKernel[1]), int(yKernel[2]), int(yKernel[3]), int(yKernel[4]), int(yKernel[5]), int(yKernel[6]), int(yKernel[7])
	yFourTap := yk0 == 0 && yk1 == 0 && yk6 == 0 && yk7 == 0
	for y := range height {
		if yFourTap {
			col := im[(y+2)*imStride : (y+6)*imStride+width]
			for x := range width {
				sum := yBias + yk2*int(col[x]) + yk3*int(col[x+imStride]) +
					yk4*int(col[x+2*imStride]) + yk5*int(col[x+3*imStride])
				res := roundPowerOfTwo(sum, round1) - roundOffset
				storeHighBDSample(dst, dstX+x, dstY+y, clipPixelHighBD(roundPowerOfTwo(res, bits), max))
			}
			continue
		}
		col := im[y*imStride : (y+7)*imStride+width]
		for x := range width {
			sum := yBias + yk0*int(col[x]) + yk1*int(col[x+imStride]) + yk2*int(col[x+2*imStride]) +
				yk3*int(col[x+3*imStride]) + yk4*int(col[x+4*imStride]) + yk5*int(col[x+5*imStride]) +
				yk6*int(col[x+6*imStride]) + yk7*int(col[x+7*imStride])
			res := roundPowerOfTwo(sum, round1) - roundOffset
			storeHighBDSample(dst, dstX+x, dstY+y, clipPixelHighBD(roundPowerOfTwo(res, bits), max))
		}
	}
}

func loadSample8Clamped(plane frame.Plane, x int, y int) byte {
	x = clampInt(x, 0, plane.Width-1)
	y = clampInt(y, 0, plane.Height-1)
	return plane.Pix[y*plane.Stride+x]
}

// loadSample8ClampedRow loads a sample whose row offset (clamped y times
// stride) is already resolved, clamping only the horizontal coordinate. It is
// bit-identical to loadSample8Clamped(plane, x, y) when rowBase ==
// clampInt(y,0,Height-1)*Stride.
func loadSample8ClampedRow(plane frame.Plane, x int, rowBase int) byte {
	x = clampInt(x, 0, plane.Width-1)
	return plane.Pix[rowBase+x]
}

// clampedXInterior returns the half-open output-column range [lo, hi) for a
// horizontal tap window of tapsWide samples whose leftmost tap for output
// column x is at source column firstTap+x. Inside [lo, hi) every tap source
// column lies within [0, width-1], so the per-tap horizontal clamp is a no-op
// and a contiguous slice load is bit-identical to loadSample8Clamped. Columns
// outside the range still need per-tap clamping. lo<=hi is always returned
// (an empty range when the whole row needs clamping).
func clampedXInterior(firstTap int, tapsWide int, planeWidth int, width int) (lo int, hi int) {
	// firstTap+x >= 0  -> x >= -firstTap
	lo = -firstTap
	if lo < 0 {
		lo = 0
	}
	if lo > width {
		lo = width
	}
	// firstTap+x+tapsWide-1 <= planeWidth-1 -> x <= planeWidth-tapsWide-firstTap
	hi = planeWidth - tapsWide - firstTap + 1
	if hi > width {
		hi = width
	}
	if hi < lo {
		hi = lo
	}
	return lo, hi
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
	if uint(v) > 255 {
		if v < 0 {
			return 0
		}
		return 255
	}
	return uint8(v)
}

func clipPixelHighBD(v int, max uint16) uint16 {
	if uint(v) > uint(max) {
		if v < 0 {
			return 0
		}
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

func lowerIntegerPrecision(v int16) int16 {
	mod := v % SubpelScale
	if mod == 0 {
		return v
	}
	v -= mod
	if absInt16(mod) > SubpelScale/2 {
		if mod > 0 {
			v += SubpelScale
		} else {
			v -= SubpelScale
		}
	}
	return v
}

func absInt16(v int16) int16 {
	if v < 0 {
		return -v
	}
	return v
}

func clampInt(v int, lo int, hi int) int {
	return min(max(v, lo), hi)
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
	maxInt16 = int64(1<<15 - 1)
	minInt16 = -1 << 15
	maxInt   = int(^uint(0) >> 1)
	minInt   = -maxInt - 1
)
