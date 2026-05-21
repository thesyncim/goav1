package decoder

import (
	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/parser"
)

// AcquireFrameSurface validates that pool storage matches the parsed AV1 frame
// format before acquiring an output surface.
func AcquireFrameSurface(pool *frame.Pool, sequence parser.SequenceHeader, size parser.FrameSize, align int) (int, *frame.Frame, error) {
	format, err := frameFormatFromHeaders(sequence, size, align)
	if err != nil {
		return -1, nil, err
	}
	return pool.AcquireFormat(format)
}

func frameFormatFromHeaders(sequence parser.SequenceHeader, size parser.FrameSize, align int) (frame.Format, error) {
	if size.CodedWidth == 0 || size.Height == 0 || sequence.ColorConfig.BitDepth == 0 {
		return frame.Format{}, frame.ErrInvalidFormat
	}
	maxInt := uint64(^uint(0) >> 1)
	if uint64(size.CodedWidth) > maxInt || uint64(size.Height) > maxInt {
		return frame.Format{}, frame.ErrInvalidFormat
	}
	return frame.Format{
		Width:        int(size.CodedWidth),
		Height:       int(size.Height),
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
