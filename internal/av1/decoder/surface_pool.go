package decoder

import (
	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/parser"
)

// AcquireFrameSurface validates that pool storage matches the parsed AV1 frame
// format before acquiring an output surface.
func AcquireFrameSurface(pool *frame.Pool, sequence parser.SequenceHeader, size parser.FrameSize, align int) (int, *frame.Frame, error) {
	format, err := codedFrameFormatFromHeaders(sequence, size, align)
	if err != nil {
		return -1, nil, err
	}
	return pool.AcquireFormat(format)
}

// BeginFrameSurface resolves inter-frame references and acquires the output
// frame surface for a frame-header or frame event. Reference output is written
// before the pool slot is acquired, so invalid references leave the pool
// untouched.
func BeginFrameSurface(refs *SurfaceReferences, pool *frame.Pool, sequence parser.SequenceHeader, event Event, align int, references []int) (int, *frame.Frame, int, error) {
	if event.Kind != EventFrameHeader && event.Kind != EventFrame {
		return -1, nil, 0, ErrInvalidSurfaceEvent
	}
	refCount, err := refs.FrameReferences(event, references)
	if err != nil {
		return -1, nil, 0, err
	}
	surface, output, err := AcquireFrameSurface(pool, sequence, event.FrameSize, align)
	if err != nil {
		return -1, nil, 0, err
	}
	return surface, output, refCount, nil
}

// ResolveFrameReferences converts frame-pool surface indices into checked
// frame pointers. dst is published only after every referenced surface has
// validated, so failures leave any previous caller-owned contents untouched.
func ResolveFrameReferences(pool *frame.Pool, surfaces []int, dst []*frame.Frame) (int, error) {
	count := len(surfaces)
	if count > parser.InterRefsPerFrame {
		return 0, ErrInvalidSurfaceReference
	}
	if len(dst) < count {
		return 0, ErrSurfaceReferenceBufferTooSmall
	}

	var resolved [parser.InterRefsPerFrame]*frame.Frame
	for i := 0; i < count; i++ {
		if surfaces[i] < 0 {
			return 0, ErrInvalidSurfaceReference
		}
		ref, err := pool.Frame(surfaces[i])
		if err != nil {
			return 0, err
		}
		resolved[i] = ref
	}
	for i := 0; i < count; i++ {
		dst[i] = resolved[i]
	}
	return count, nil
}

func frameFormatFromHeaders(sequence parser.SequenceHeader, size parser.FrameSize, align int) (frame.Format, error) {
	return codedFrameFormatFromHeaders(sequence, size, align)
}

func codedFrameFormatFromHeaders(sequence parser.SequenceHeader, size parser.FrameSize, align int) (frame.Format, error) {
	return frameFormatFromHeaderWidth(sequence, size.CodedWidth, size.Height, align)
}

func outputFrameFormatFromHeaders(sequence parser.SequenceHeader, size parser.FrameSize, align int) (frame.Format, error) {
	return frameFormatFromHeaderWidth(sequence, size.UpscaledWidth, size.Height, align)
}

func frameFormatFromHeaderWidth(sequence parser.SequenceHeader, width uint32, height uint32, align int) (frame.Format, error) {
	if width == 0 || height == 0 || sequence.ColorConfig.BitDepth == 0 {
		return frame.Format{}, frame.ErrInvalidFormat
	}
	maxInt := uint64(^uint(0) >> 1)
	if uint64(width) > maxInt || uint64(height) > maxInt {
		return frame.Format{}, frame.ErrInvalidFormat
	}
	return frame.Format{
		Width:        int(width),
		Height:       int(height),
		BitDepth:     sequence.ColorConfig.BitDepth,
		MonoChrome:   sequence.ColorConfig.MonoChrome,
		SubsamplingX: sequence.ColorConfig.SubsamplingX,
		SubsamplingY: sequence.ColorConfig.SubsamplingY,
		Align:        align,
	}, nil
}

// FinishFrameSurface applies a final decoded frame event to refs and releases
// overwritten frame-pool slots. Reference state is published only after the
// pool accepts the release batch.
func FinishFrameSurface(refs *SurfaceReferences, pool *frame.Pool, event Event, surface int, releases []int) (int, error) {
	if refs == nil {
		return 0, ErrInvalidSurfaceReference
	}

	next := *refs
	count, err := next.FinishFrame(event, surface, releases)
	if err != nil {
		return 0, err
	}
	if count != 0 {
		if pool == nil {
			return 0, frame.ErrInvalidPool
		}
		if err := pool.ReleaseMany(releases[:count]); err != nil {
			return 0, err
		}
	}

	*refs = next
	return count, nil
}

// ShowExistingFrameSurface resolves a show-existing-frame event, applies any
// key-frame reference reset, and releases overwritten frame-pool slots.
// Reference state is published only after the pool accepts the release batch.
func ShowExistingFrameSurface(refs *SurfaceReferences, pool *frame.Pool, event Event, releases []int) (int, int, error) {
	if refs == nil {
		return -1, 0, ErrInvalidSurfaceReference
	}

	next := *refs
	surface, count, err := next.ShowExistingFrame(event, releases)
	if err != nil {
		return -1, 0, err
	}
	if count != 0 {
		if pool == nil {
			return -1, 0, frame.ErrInvalidPool
		}
		if err := pool.ReleaseMany(releases[:count]); err != nil {
			return -1, 0, err
		}
	}

	*refs = next
	return surface, count, nil
}
