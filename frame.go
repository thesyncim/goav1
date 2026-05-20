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

func BindFrame(buffer []byte, format FrameFormat) (Frame, error) {
	return internalframe.Bind(buffer, format)
}

func BindFramePool(backing []byte, format FrameFormat, frames []Frame, free []int, used []bool) (FramePool, error) {
	return internalframe.BindPool(backing, format, frames, free, used)
}
