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

// PredictInterPlaneBlock copies a full-pixel inter prediction block from ref
// into dst. Fractional vectors return ErrInvalidMotion and are reserved for the
// interpolation-filter path.
func PredictInterPlaneBlock(dst frame.Plane, ref frame.Plane, bytesPerSample int, dstX int, dstY int, width int, height int, mv Vector) error {
	refX, refY, err := FullpelReferenceOrigin(dstX, dstY, mv)
	if err != nil {
		return err
	}
	if err := dsp.CopyPlaneBlock(dst, ref, bytesPerSample, dstX, dstY, refX, refY, width, height); err != nil {
		return ErrInvalidMotion
	}
	return nil
}

func scaleFullpel(v int) (int32, bool) {
	scaled := int64(v) * SubpelScale
	if scaled < minInt32 || scaled > maxInt32 {
		return 0, false
	}
	return int32(scaled), true
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

const (
	maxInt32 = int64(1<<31 - 1)
	minInt32 = -1 << 31
	maxInt   = int(^uint(0) >> 1)
	minInt   = -maxInt - 1
)
