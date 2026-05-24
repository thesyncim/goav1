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
	return FrameCodedFormatFromHeaders(sequence, size, align)
}

func FrameCodedFormatFromHeaders(sequence SequenceHeader, size FrameSize, align int) (FrameFormat, error) {
	return frameFormatFromHeaderWidth(sequence, size.CodedWidth, size.Height, align)
}

func FrameOutputFormatFromHeaders(sequence SequenceHeader, size FrameSize, align int) (FrameFormat, error) {
	return frameFormatFromHeaderWidth(sequence, size.UpscaledWidth, size.Height, align)
}

func frameFormatFromHeaderWidth(sequence SequenceHeader, width uint32, height uint32, align int) (FrameFormat, error) {
	if width == 0 || height == 0 || sequence.ColorConfig.BitDepth == 0 {
		return FrameFormat{}, ErrFrameInvalidFormat
	}
	maxInt := uint64(^uint(0) >> 1)
	if uint64(width) > maxInt || uint64(height) > maxInt {
		return FrameFormat{}, ErrFrameInvalidFormat
	}
	return FrameFormat{
		Width:        int(width),
		Height:       int(height),
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
