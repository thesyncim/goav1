package goav1

import internalframe "github.com/thesyncim/goav1/internal/av1/frame"

type FrameFormat = internalframe.Format
type FrameLayout = internalframe.Layout
type FramePlane = internalframe.Plane
type Frame = internalframe.Frame
type FramePool = internalframe.Pool
type FrameSamplePlane = internalframe.SamplePlane
type FrameBorderedSamplePlane = internalframe.BorderedSamplePlane
type FrameBorderedSamplePlaneLayout = internalframe.BorderedSamplePlaneLayout

var (
	ErrFrameInvalidFormat = internalframe.ErrInvalidFormat
	ErrFrameInvalidPool   = internalframe.ErrInvalidPool
	ErrFramePoolEmpty     = internalframe.ErrPoolEmpty
	ErrFrameInvalidSlot   = internalframe.ErrInvalidSlot
	ErrFrameInvalidPlane  = internalframe.ErrInvalidPlane
	ErrFrameShortBuffer   = internalframe.ErrShortBuffer
)

func FrameRequiredSize(format FrameFormat) (FrameLayout, error) {
	return internalframe.RequiredSize(format)
}

func FramePoolRequiredSize(format FrameFormat, count int) (FrameLayout, int, error) {
	if count <= 0 {
		return FrameLayout{}, 0, ErrFrameInvalidPool
	}
	layout, err := internalframe.RequiredSize(format)
	if err != nil {
		return FrameLayout{}, 0, err
	}
	maxInt := int(^uint(0) >> 1)
	if layout.Size != 0 && count > maxInt/layout.Size {
		return FrameLayout{}, 0, ErrFrameInvalidFormat
	}
	return layout, layout.Size * count, nil
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

func FrameSamplePlaneLen(plane FramePlane, bytesPerSample int) (int, error) {
	return internalframe.SamplePlaneLen(plane, bytesPerSample)
}

func FrameBorderedSamplePlaneLen(plane FramePlane, bytesPerSample int, borderHorz int, borderVert int, align int) (FrameBorderedSamplePlaneLayout, error) {
	return internalframe.BorderedSamplePlaneLen(plane, bytesPerSample, borderHorz, borderVert, align)
}

func BindFrameBorderedSamplePlane(dst []uint16, plane FramePlane, bytesPerSample int, borderHorz int, borderVert int, align int) (FrameBorderedSamplePlane, error) {
	return internalframe.BindBorderedSamplePlane(dst, plane, bytesPerSample, borderHorz, borderVert, align)
}

func LoadFrameSamplePlane(dst []uint16, src FramePlane, bytesPerSample int) (FrameSamplePlane, error) {
	return internalframe.LoadSamplePlane(dst, src, bytesPerSample)
}

func LoadFrameBorderedSamplePlane(dst []uint16, src FramePlane, bytesPerSample int, borderHorz int, borderVert int, align int) (FrameBorderedSamplePlane, error) {
	return internalframe.LoadBorderedSamplePlane(dst, src, bytesPerSample, borderHorz, borderVert, align)
}

func StoreFrameSamplePlane(dst FramePlane, bytesPerSample int, src FrameSamplePlane) error {
	return internalframe.StoreSamplePlane(dst, bytesPerSample, src)
}

func StoreFrameBorderedSamplePlane(dst FramePlane, bytesPerSample int, src FrameBorderedSamplePlane) error {
	return internalframe.StoreBorderedSamplePlane(dst, bytesPerSample, src)
}
