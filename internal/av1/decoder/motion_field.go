package decoder

import (
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/tile"
)

// PublishTemporalMotionReference stores the decoded frame's MV_REF side data
// and order-hint metadata under its frame-pool surface index.
func PublishTemporalMotionReference(event Event, surface int, current *tile.ReferenceMVFrame, store []tile.TemporalMotionReferenceFrame) error {
	if (event.Kind != EventFrame && event.Kind != EventTileGroup) || !event.TileGroup.Final {
		return ErrInvalidSurfaceEvent
	}
	if surface < 0 || current == nil {
		return ErrInvalidSurfaceReference
	}
	if surface >= len(store) {
		return ErrSurfaceReferenceBufferTooSmall
	}
	if err := current.Validate(); err != nil {
		return ErrInvalidSurfaceReference
	}
	store[surface] = tile.TemporalMotionReferenceFrame{
		Frame:         current,
		OrderHint:     event.FrameHeader.OrderHint,
		RefOrderHints: event.ReferenceOrderHints,
		IntraOnly: event.FrameHeader.FrameType == parser.FrameTypeKey ||
			event.FrameHeader.FrameType == parser.FrameTypeIntraOnly,
	}
	return nil
}

// ResolveTemporalMotionReferences converts frame-pool surface indices into
// checked TemporalMotionReferenceFrame metadata. dst is published atomically
// after every referenced surface has validated.
func ResolveTemporalMotionReferences(surfaces []int, store []tile.TemporalMotionReferenceFrame, dst []tile.TemporalMotionReferenceFrame) (int, error) {
	count := len(surfaces)
	if count > parser.InterRefsPerFrame {
		return 0, ErrInvalidSurfaceReference
	}
	if len(dst) < count {
		return 0, ErrSurfaceReferenceBufferTooSmall
	}

	var resolved [parser.InterRefsPerFrame]tile.TemporalMotionReferenceFrame
	for i := range count {
		surface := surfaces[i]
		if surface < 0 || surface >= len(store) || store[surface].Frame == nil {
			return 0, ErrInvalidSurfaceReference
		}
		if err := store[surface].Frame.Validate(); err != nil {
			return 0, ErrInvalidSurfaceReference
		}
		resolved[i] = store[surface]
	}
	for i := range count {
		dst[i] = resolved[i]
	}
	return count, nil
}
