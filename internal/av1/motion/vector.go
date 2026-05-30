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
	return predictInterPlaneBlockFromOriginWithFilterSize(dst, ref, bytesPerSample, bitDepth, true, dstX, dstY, refX, refY, width, height, width, height, subX, subY, filters)
}

// PredictInterPlaneBlockFromOriginWithFilterBitDepthFilterSize is
// PredictInterPlaneBlockFromOriginWithFilterBitDepth but selects the
// interpolation-filter kernel from filterW / filterH instead of the output
// width / height. libaom's av1_get_interp_filter_params_with_block_size()
// chooses the narrow 4-tap chroma/luma filter (sub_pel_filters_4 /
// sub_pel_filters_4smooth) only when the *un-clipped* plane block side is <= 4
// (block_size_wide[plane_bsize] / block_size_high[plane_bsize]). When a chroma
// block straddles the right/bottom frame edge its visible extent can shrink to
// <= 4 while the plane block stays wider, and libaom keeps using the 8-tap
// filter. Callers that clip the output extent to the visible edge pass the
// un-clipped plane block dimensions here so the kernel choice matches libaom.
func PredictInterPlaneBlockFromOriginWithFilterBitDepthFilterSize(dst frame.Plane, ref frame.Plane, bytesPerSample int, bitDepth uint8, dstX int, dstY int, refX int, refY int, width int, height int, filterW int, filterH int, subX int, subY int, filters InterpFilters) error {
	return predictInterPlaneBlockFromOriginWithFilterSize(dst, ref, bytesPerSample, bitDepth, true, dstX, dstY, refX, refY, width, height, filterW, filterH, subX, subY, filters)
}

func predictInterPlaneBlockFromOriginWithFilter(dst frame.Plane, ref frame.Plane, bytesPerSample int, bitDepth uint8, explicitBitDepth bool, dstX int, dstY int, refX int, refY int, width int, height int, subX int, subY int, filters InterpFilters) error {
	return predictInterPlaneBlockFromOriginWithFilterSize(dst, ref, bytesPerSample, bitDepth, explicitBitDepth, dstX, dstY, refX, refY, width, height, width, height, subX, subY, filters)
}

func predictInterPlaneBlockFromOriginWithFilterSize(dst frame.Plane, ref frame.Plane, bytesPerSample int, bitDepth uint8, explicitBitDepth bool, dstX int, dstY int, refX int, refY int, width int, height int, filterW int, filterH int, subX int, subY int, filters InterpFilters) error {
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
	if filterW <= 0 {
		filterW = width
	}
	if filterH <= 0 {
		filterH = height
	}
	if bytesPerSample != 1 {
		if bytesPerSample == 2 && explicitBitDepth {
			if err := predictInterPlaneBlockHighBD(dst, ref, bitDepth, dstX, dstY, refX, refY, width, height, filterW, filterH, subX, subY, filters); err != nil {
				return ErrInvalidMotion
			}
			return nil
		}
		return ErrInvalidMotion
	}
	if err := predictInterPlaneBlock8(dst, ref, dstX, dstY, refX, refY, width, height, filterW, filterH, subX, subY, filters); err != nil {
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

func predictInterPlaneBlock8(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, filterW int, filterH int, subX int, subY int, filters InterpFilters) error {
	if width <= 0 || height <= 0 || width > maxBlockSize || height > maxBlockSize {
		return ErrInvalidMotion
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
			convolve2D8Impl(dst, ref, dstX, dstY, refX, refY, width, height, xKernel, yKernel)
		} else {
			convolve2D8ClampedImpl(dst, ref, dstX, dstY, refX, refY, width, height, xKernel, yKernel)
		}
	case subX != 0:
		if planeRegionFits(ref, 1, refX-foX, refY, width+filterTaps-1, height) {
			convolveX8Impl(dst, ref, dstX, dstY, refX, refY, width, height, xKernel)
		} else {
			convolveX8ClampedImpl(dst, ref, dstX, dstY, refX, refY, width, height, xKernel)
		}
	case subY != 0:
		if planeRegionFits(ref, 1, refX, refY-foY, width, height+filterTaps-1) {
			convolveY8Impl(dst, ref, dstX, dstY, refX, refY, width, height, yKernel)
		} else {
			convolveY8ClampedImpl(dst, ref, dstX, dstY, refX, refY, width, height, yKernel)
		}
	}
	return nil
}

func predictInterPlaneBlockHighBD(dst frame.Plane, ref frame.Plane, bitDepth uint8, dstX int, dstY int, refX int, refY int, width int, height int, filterW int, filterH int, subX int, subY int, filters InterpFilters) error {
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
			convolve2DHighBDImpl(dst, ref, bitDepth, max, dstX, dstY, refX, refY, width, height, xKernel, yKernel)
		} else {
			convolve2DHighBDClampedImpl(dst, ref, bitDepth, max, dstX, dstY, refX, refY, width, height, xKernel, yKernel)
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

func convolveX8PureGo(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, kernel [filterTaps]int16) {
	fo := filterTaps/2 - 1
	k0, k1, k2, k3, k4, k5, k6, k7 := int(kernel[0]), int(kernel[1]), int(kernel[2]), int(kernel[3]), int(kernel[4]), int(kernel[5]), int(kernel[6]), int(kernel[7])
	// 4-tap kernels (small blocks / bilinear) have zero end taps, so skip those MACs.
	fourTap := k0 == 0 && k1 == 0 && k6 == 0 && k7 == 0
	for y := range height {
		srcRow := ref.Pix[(refY+y)*ref.Stride+refX-fo:]
		dstRow := dst.Pix[(dstY+y)*dst.Stride+dstX:]
		if fourTap {
			for x := range width {
				s := srcRow[x : x+filterTaps]
				sum := k2*int(s[2]) + k3*int(s[3]) + k4*int(s[4]) + k5*int(s[5])
				res := roundPowerOfTwo(sum, round0Bits)
				dstRow[x] = byte(clipPixel(roundPowerOfTwo(res, filterBits-round0Bits)))
			}
			continue
		}
		for x := range width {
			s := srcRow[x : x+filterTaps]
			sum := k0*int(s[0]) + k1*int(s[1]) + k2*int(s[2]) + k3*int(s[3]) +
				k4*int(s[4]) + k5*int(s[5]) + k6*int(s[6]) + k7*int(s[7])
			res := roundPowerOfTwo(sum, round0Bits)
			dstRow[x] = byte(clipPixel(roundPowerOfTwo(res, filterBits-round0Bits)))
		}
	}
}

func convolveX8ClampedPureGo(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, kernel [filterTaps]int16) {
	fo := filterTaps/2 - 1
	k0, k1, k2, k3, k4, k5, k6, k7 := int(kernel[0]), int(kernel[1]), int(kernel[2]), int(kernel[3]), int(kernel[4]), int(kernel[5]), int(kernel[6]), int(kernel[7])
	fourTap := k0 == 0 && k1 == 0 && k6 == 0 && k7 == 0
	for y := range height {
		sy := refY + y
		dstRow := dst.Pix[(dstY+y)*dst.Stride+dstX:]
		if fourTap {
			for x := range width {
				sx := refX + x - fo
				sum := k2*int(loadSample8Clamped(ref, sx+2, sy)) +
					k3*int(loadSample8Clamped(ref, sx+3, sy)) +
					k4*int(loadSample8Clamped(ref, sx+4, sy)) +
					k5*int(loadSample8Clamped(ref, sx+5, sy))
				res := roundPowerOfTwo(sum, round0Bits)
				dstRow[x] = byte(clipPixel(roundPowerOfTwo(res, filterBits-round0Bits)))
			}
			continue
		}
		for x := range width {
			sx := refX + x - fo
			sum := k0*int(loadSample8Clamped(ref, sx, sy)) +
				k1*int(loadSample8Clamped(ref, sx+1, sy)) +
				k2*int(loadSample8Clamped(ref, sx+2, sy)) +
				k3*int(loadSample8Clamped(ref, sx+3, sy)) +
				k4*int(loadSample8Clamped(ref, sx+4, sy)) +
				k5*int(loadSample8Clamped(ref, sx+5, sy)) +
				k6*int(loadSample8Clamped(ref, sx+6, sy)) +
				k7*int(loadSample8Clamped(ref, sx+7, sy))
			res := roundPowerOfTwo(sum, round0Bits)
			dstRow[x] = byte(clipPixel(roundPowerOfTwo(res, filterBits-round0Bits)))
		}
	}
}

func convolveY8PureGo(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, kernel [filterTaps]int16) {
	fo := filterTaps/2 - 1
	stride := ref.Stride
	k0, k1, k2, k3, k4, k5, k6, k7 := int(kernel[0]), int(kernel[1]), int(kernel[2]), int(kernel[3]), int(kernel[4]), int(kernel[5]), int(kernel[6]), int(kernel[7])
	fourTap := k0 == 0 && k1 == 0 && k6 == 0 && k7 == 0
	for y := range height {
		base := (refY + y - fo) * stride
		dstRow := dst.Pix[(dstY+y)*dst.Stride+dstX:]
		if fourTap {
			// Only taps 2..5 contribute; reslice that contiguous column window.
			col := ref.Pix[base+2*stride+refX:]
			for x := range width {
				sum := k2*int(col[x]) + k3*int(col[x+stride]) +
					k4*int(col[x+2*stride]) + k5*int(col[x+3*stride])
				dstRow[x] = byte(clipPixel(roundPowerOfTwo(sum, filterBits)))
			}
			continue
		}
		col := ref.Pix[base+refX:]
		for x := range width {
			sum := k0*int(col[x]) + k1*int(col[x+stride]) + k2*int(col[x+2*stride]) +
				k3*int(col[x+3*stride]) + k4*int(col[x+4*stride]) + k5*int(col[x+5*stride]) +
				k6*int(col[x+6*stride]) + k7*int(col[x+7*stride])
			dstRow[x] = byte(clipPixel(roundPowerOfTwo(sum, filterBits)))
		}
	}
}

func convolveY8ClampedPureGo(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, kernel [filterTaps]int16) {
	fo := filterTaps/2 - 1
	k0, k1, k2, k3, k4, k5, k6, k7 := int(kernel[0]), int(kernel[1]), int(kernel[2]), int(kernel[3]), int(kernel[4]), int(kernel[5]), int(kernel[6]), int(kernel[7])
	fourTap := k0 == 0 && k1 == 0 && k6 == 0 && k7 == 0
	for y := range height {
		sy := refY + y - fo
		dstRow := dst.Pix[(dstY+y)*dst.Stride+dstX:]
		if fourTap {
			for x := range width {
				sx := refX + x
				sum := k2*int(loadSample8Clamped(ref, sx, sy+2)) +
					k3*int(loadSample8Clamped(ref, sx, sy+3)) +
					k4*int(loadSample8Clamped(ref, sx, sy+4)) +
					k5*int(loadSample8Clamped(ref, sx, sy+5))
				dstRow[x] = byte(clipPixel(roundPowerOfTwo(sum, filterBits)))
			}
			continue
		}
		for x := range width {
			sx := refX + x
			sum := k0*int(loadSample8Clamped(ref, sx, sy)) +
				k1*int(loadSample8Clamped(ref, sx, sy+1)) +
				k2*int(loadSample8Clamped(ref, sx, sy+2)) +
				k3*int(loadSample8Clamped(ref, sx, sy+3)) +
				k4*int(loadSample8Clamped(ref, sx, sy+4)) +
				k5*int(loadSample8Clamped(ref, sx, sy+5)) +
				k6*int(loadSample8Clamped(ref, sx, sy+6)) +
				k7*int(loadSample8Clamped(ref, sx, sy+7))
			dstRow[x] = byte(clipPixel(roundPowerOfTwo(sum, filterBits)))
		}
	}
}

func convolve2D8PureGo(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16) {
	const imStride = maxBlockSize
	var im [((maxBlockSize + filterTaps - 1) * maxBlockSize)]int16
	foX := filterTaps/2 - 1
	foY := filterTaps/2 - 1
	imH := height + filterTaps - 1
	const xBias = 1 << (8 + filterBits - 1)

	xk0, xk1, xk2, xk3, xk4, xk5, xk6, xk7 := int(xKernel[0]), int(xKernel[1]), int(xKernel[2]), int(xKernel[3]), int(xKernel[4]), int(xKernel[5]), int(xKernel[6]), int(xKernel[7])
	xFourTap := xk0 == 0 && xk1 == 0 && xk6 == 0 && xk7 == 0
	for y := range imH {
		// Reslice the source row once per row: the tap window srcRow[x:x+filterTaps]
		// has a statically-known length so the compiler drops the per-tap bounds
		// check on the inner loads.
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
	const imStride = maxBlockSize
	var im [((maxBlockSize + filterTaps - 1) * maxBlockSize)]int16
	foX := filterTaps/2 - 1
	foY := filterTaps/2 - 1
	imH := height + filterTaps - 1
	const xBias = 1 << (8 + filterBits - 1)

	xk0, xk1, xk2, xk3, xk4, xk5, xk6, xk7 := int(xKernel[0]), int(xKernel[1]), int(xKernel[2]), int(xKernel[3]), int(xKernel[4]), int(xKernel[5]), int(xKernel[6]), int(xKernel[7])
	xFourTap := xk0 == 0 && xk1 == 0 && xk6 == 0 && xk7 == 0
	for y := range imH {
		sy := refY - foY + y
		imRow := im[y*imStride:]
		if xFourTap {
			for x := range width {
				sx := refX + x - foX
				sum := xBias +
					xk2*int(loadSample8Clamped(ref, sx+2, sy)) +
					xk3*int(loadSample8Clamped(ref, sx+3, sy)) +
					xk4*int(loadSample8Clamped(ref, sx+4, sy)) +
					xk5*int(loadSample8Clamped(ref, sx+5, sy))
				imRow[x] = int16(roundPowerOfTwo(sum, round0Bits))
			}
			continue
		}
		for x := range width {
			sx := refX + x - foX
			sum := xBias +
				xk0*int(loadSample8Clamped(ref, sx, sy)) +
				xk1*int(loadSample8Clamped(ref, sx+1, sy)) +
				xk2*int(loadSample8Clamped(ref, sx+2, sy)) +
				xk3*int(loadSample8Clamped(ref, sx+3, sy)) +
				xk4*int(loadSample8Clamped(ref, sx+4, sy)) +
				xk5*int(loadSample8Clamped(ref, sx+5, sy)) +
				xk6*int(loadSample8Clamped(ref, sx+6, sy)) +
				xk7*int(loadSample8Clamped(ref, sx+7, sy))
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
	fo := filterTaps/2 - 1
	// av1_highbd_convolve_x_sr_c: round_0 then bits = FILTER_BITS - round_0.
	// round_0 is bd-dependent (5 at bd == 12), matching get_conv_params_no_round.
	round0, _ := highBDRoundBits(bitDepth)
	bits := filterBits - round0
	k0, k1, k2, k3, k4, k5, k6, k7 := int(kernel[0]), int(kernel[1]), int(kernel[2]), int(kernel[3]), int(kernel[4]), int(kernel[5]), int(kernel[6]), int(kernel[7])
	fourTap := k0 == 0 && k1 == 0 && k6 == 0 && k7 == 0
	for y := range height {
		base := (refY+y)*ref.Stride + (refX-fo)*2
		srcRow := ref.Pix[base : base+(width+filterTaps-1)*2]
		if fourTap {
			for x := range width {
				o := x * 2
				s2 := int(uint16(srcRow[o+4]) | uint16(srcRow[o+5])<<8)
				s3 := int(uint16(srcRow[o+6]) | uint16(srcRow[o+7])<<8)
				s4 := int(uint16(srcRow[o+8]) | uint16(srcRow[o+9])<<8)
				s5 := int(uint16(srcRow[o+10]) | uint16(srcRow[o+11])<<8)
				sum := k2*s2 + k3*s3 + k4*s4 + k5*s5
				res := roundPowerOfTwo(sum, round0)
				storeHighBDSample(dst, dstX+x, dstY+y, clipPixelHighBD(roundPowerOfTwo(res, bits), max))
			}
			continue
		}
		for x := range width {
			o := x * 2
			s0 := int(uint16(srcRow[o]) | uint16(srcRow[o+1])<<8)
			s1 := int(uint16(srcRow[o+2]) | uint16(srcRow[o+3])<<8)
			s2 := int(uint16(srcRow[o+4]) | uint16(srcRow[o+5])<<8)
			s3 := int(uint16(srcRow[o+6]) | uint16(srcRow[o+7])<<8)
			s4 := int(uint16(srcRow[o+8]) | uint16(srcRow[o+9])<<8)
			s5 := int(uint16(srcRow[o+10]) | uint16(srcRow[o+11])<<8)
			s6 := int(uint16(srcRow[o+12]) | uint16(srcRow[o+13])<<8)
			s7 := int(uint16(srcRow[o+14]) | uint16(srcRow[o+15])<<8)
			sum := k0*s0 + k1*s1 + k2*s2 + k3*s3 + k4*s4 + k5*s5 + k6*s6 + k7*s7
			res := roundPowerOfTwo(sum, round0)
			storeHighBDSample(dst, dstX+x, dstY+y, clipPixelHighBD(roundPowerOfTwo(res, bits), max))
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
	const imStride = maxBlockSize
	var im [((maxBlockSize + filterTaps - 1) * maxBlockSize)]int32
	foX := filterTaps/2 - 1
	foY := filterTaps/2 - 1
	imH := height + filterTaps - 1
	round0, round1 := highBDRoundBits(bitDepth)
	xBias := 1 << (int(bitDepth) + filterBits - 1)

	xk0, xk1, xk2, xk3, xk4, xk5, xk6, xk7 := int(xKernel[0]), int(xKernel[1]), int(xKernel[2]), int(xKernel[3]), int(xKernel[4]), int(xKernel[5]), int(xKernel[6]), int(xKernel[7])
	xFourTap := xk0 == 0 && xk1 == 0 && xk6 == 0 && xk7 == 0
	for y := range imH {
		// Reslice the high-bit-depth source row (2 bytes/sample) so the per-tap
		// loads index a length-known slice and drop bounds checks.
		base := (refY-foY+y)*ref.Stride + (refX-foX)*2
		srcRow := ref.Pix[base : base+(width+filterTaps-1)*2]
		imRow := im[y*imStride:]
		if xFourTap {
			for x := range width {
				o := x * 2
				s2 := int(uint16(srcRow[o+4]) | uint16(srcRow[o+5])<<8)
				s3 := int(uint16(srcRow[o+6]) | uint16(srcRow[o+7])<<8)
				s4 := int(uint16(srcRow[o+8]) | uint16(srcRow[o+9])<<8)
				s5 := int(uint16(srcRow[o+10]) | uint16(srcRow[o+11])<<8)
				sum := xBias + xk2*s2 + xk3*s3 + xk4*s4 + xk5*s5
				imRow[x] = int32(roundPowerOfTwo(sum, round0))
			}
			continue
		}
		for x := range width {
			o := x * 2
			s0 := int(uint16(srcRow[o]) | uint16(srcRow[o+1])<<8)
			s1 := int(uint16(srcRow[o+2]) | uint16(srcRow[o+3])<<8)
			s2 := int(uint16(srcRow[o+4]) | uint16(srcRow[o+5])<<8)
			s3 := int(uint16(srcRow[o+6]) | uint16(srcRow[o+7])<<8)
			s4 := int(uint16(srcRow[o+8]) | uint16(srcRow[o+9])<<8)
			s5 := int(uint16(srcRow[o+10]) | uint16(srcRow[o+11])<<8)
			s6 := int(uint16(srcRow[o+12]) | uint16(srcRow[o+13])<<8)
			s7 := int(uint16(srcRow[o+14]) | uint16(srcRow[o+15])<<8)
			sum := xBias + xk0*s0 + xk1*s1 + xk2*s2 + xk3*s3 + xk4*s4 + xk5*s5 + xk6*s6 + xk7*s7
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

func convolve2DHighBDClampedPureGo(dst frame.Plane, ref frame.Plane, bitDepth uint8, max uint16, dstX int, dstY int, refX int, refY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16) {
	const imStride = maxBlockSize
	var im [((maxBlockSize + filterTaps - 1) * maxBlockSize)]int32
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
	maxInt32 = int64(1<<31 - 1)
	minInt32 = -1 << 31
	maxInt   = int(^uint(0) >> 1)
	minInt   = -maxInt - 1
)
