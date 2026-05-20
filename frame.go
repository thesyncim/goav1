package goav1

import internalframe "github.com/thesyncim/goav1/internal/av1/frame"

type FrameFormat = internalframe.Format
type FrameLayout = internalframe.Layout
type FramePlane = internalframe.Plane
type Frame = internalframe.Frame
type FramePool = internalframe.Pool

var (
	ErrFrameInvalidFormat = internalframe.ErrInvalidFormat
	ErrFrameInvalidPool   = internalframe.ErrInvalidPool
	ErrFramePoolEmpty     = internalframe.ErrPoolEmpty
	ErrFrameInvalidSlot   = internalframe.ErrInvalidSlot
	ErrFrameShortBuffer   = internalframe.ErrShortBuffer
)

func FrameRequiredSize(format FrameFormat) (FrameLayout, error) {
	return internalframe.RequiredSize(format)
}

func FrameFormatFromHeaders(sequence SequenceHeader, size FrameSize, align int) (FrameFormat, error) {
	if size.CodedWidth == 0 || size.Height == 0 || sequence.ColorConfig.BitDepth == 0 {
		return FrameFormat{}, ErrFrameInvalidFormat
	}
	maxInt := uint64(^uint(0) >> 1)
	if uint64(size.CodedWidth) > maxInt || uint64(size.Height) > maxInt {
		return FrameFormat{}, ErrFrameInvalidFormat
	}
	return FrameFormat{
		Width:        int(size.CodedWidth),
		Height:       int(size.Height),
		BitDepth:     sequence.ColorConfig.BitDepth,
		MonoChrome:   sequence.ColorConfig.MonoChrome,
		SubsamplingX: sequence.ColorConfig.SubsamplingX,
		SubsamplingY: sequence.ColorConfig.SubsamplingY,
		Align:        align,
	}, nil
}

func BindFrame(buffer []byte, format FrameFormat) (Frame, error) {
	return internalframe.Bind(buffer, format)
}

func BindFramePool(backing []byte, format FrameFormat, frames []Frame, free []int, used []bool) (FramePool, error) {
	return internalframe.BindPool(backing, format, frames, free, used)
}
