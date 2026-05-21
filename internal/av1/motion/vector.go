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

// PredictInterPlaneBlock predicts a block using AV1's regular translational
// interpolation filter for fractional vectors.
func PredictInterPlaneBlock(dst frame.Plane, ref frame.Plane, bytesPerSample int, dstX int, dstY int, width int, height int, mv Vector) error {
	return PredictInterPlaneBlockWithFilter(dst, ref, bytesPerSample, dstX, dstY, width, height, mv, RegularFilters)
}

// PredictInterPlaneBlockWithFilter predicts a translational single-reference
// inter block. Low-bit-depth fractional vectors use libaom's av1_convolve_*_sr_c
// filter path; high-bit-depth fractional interpolation is kept for the highbd
// convolve port.
func PredictInterPlaneBlockWithFilter(dst frame.Plane, ref frame.Plane, bytesPerSample int, dstX int, dstY int, width int, height int, mv Vector, filters InterpFilters) error {
	if !filters.X.Valid() || !filters.Y.Valid() {
		return ErrInvalidMotion
	}
	refX, refY, subX, subY, err := referenceOrigin(dstX, dstY, mv)
	if err != nil {
		return err
	}
	if subX == 0 && subY == 0 {
		if err := dsp.CopyPlaneBlock(dst, ref, bytesPerSample, dstX, dstY, refX, refY, width, height); err != nil {
			return ErrInvalidMotion
		}
		return nil
	}
	if bytesPerSample != 1 {
		return ErrInvalidMotion
	}
	if err := predictInterPlaneBlock8(dst, ref, dstX, dstY, refX, refY, width, height, subX, subY, filters); err != nil {
		return ErrInvalidMotion
	}
	return nil
}

func referenceOrigin(dstX int, dstY int, mv Vector) (refX int, refY int, subX int, subY int, err error) {
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

func predictInterPlaneBlock8(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, subX int, subY int, filters InterpFilters) error {
	if width <= 0 || height <= 0 || width > maxBlockSize || height > maxBlockSize {
		return ErrInvalidMotion
	}
	if !planeRegionFits(dst, 1, dstX, dstY, width, height) {
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
		if !planeRegionFits(ref, 1, refX-foX, refY-foY, width+filterTaps-1, height+filterTaps-1) {
			return ErrInvalidMotion
		}
		convolve2D8(dst, ref, dstX, dstY, refX, refY, width, height, xKernel, yKernel)
	case subX != 0:
		if !planeRegionFits(ref, 1, refX-foX, refY, width+filterTaps-1, height) {
			return ErrInvalidMotion
		}
		convolveX8(dst, ref, dstX, dstY, refX, refY, width, height, xKernel)
	case subY != 0:
		if !planeRegionFits(ref, 1, refX, refY-foY, width, height+filterTaps-1) {
			return ErrInvalidMotion
		}
		convolveY8(dst, ref, dstX, dstY, refX, refY, width, height, yKernel)
	}
	return nil
}

func convolveX8(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, kernel [filterTaps]int16) {
	fo := filterTaps/2 - 1
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			sum := 0
			for k := 0; k < filterTaps; k++ {
				sum += int(kernel[k]) * int(ref.Pix[(refY+y)*ref.Stride+refX+x-fo+k])
			}
			res := roundPowerOfTwo(sum, round0Bits)
			dst.Pix[(dstY+y)*dst.Stride+dstX+x] = byte(clipPixel(roundPowerOfTwo(res, filterBits-round0Bits)))
		}
	}
}

func convolveY8(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, kernel [filterTaps]int16) {
	fo := filterTaps/2 - 1
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			sum := 0
			for k := 0; k < filterTaps; k++ {
				sum += int(kernel[k]) * int(ref.Pix[(refY+y-fo+k)*ref.Stride+refX+x])
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
	for y := 0; y < imH; y++ {
		for x := 0; x < width; x++ {
			sum := 1 << (8 + filterBits - 1)
			for k := 0; k < filterTaps; k++ {
				sum += int(xKernel[k]) * int(ref.Pix[(refY-foY+y)*ref.Stride+refX+x-foX+k])
			}
			im[y*imStride+x] = int16(roundPowerOfTwo(sum, round0Bits))
		}
	}
	offsetBits := 8 + 2*filterBits - round0Bits
	roundOffset := (1 << (offsetBits - round1Bits)) + (1 << (offsetBits - round1Bits - 1))
	bits := 2*filterBits - round0Bits - round1Bits
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			sum := 1 << offsetBits
			for k := 0; k < filterTaps; k++ {
				sum += int(yKernel[k]) * int(im[(y+k)*imStride+x])
			}
			res := roundPowerOfTwo(sum, round1Bits) - roundOffset
			dst.Pix[(dstY+y)*dst.Stride+dstX+x] = byte(clipPixel(roundPowerOfTwo(res, bits)))
		}
	}
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
