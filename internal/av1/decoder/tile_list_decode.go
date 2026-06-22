package decoder

import (
	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/obu"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/threading"
	"github.com/thesyncim/goav1/internal/av1/tile"
)

// TileListEntryDecodePlan is the frame-work view needed to decode one
// tile_list_obu() entry. Event is a synthetic tile-group event whose payload is
// the entry's raw coded tile bytes; Step binds the single-tile work plan and
// exposes exactly one reference, LAST_FRAME, backed by the entry's external
// anchor frame.
type TileListEntryDecodePlan struct {
	Event          Event
	Step           FrameWorkStep
	Payload        []byte
	Geometry       parser.TileListOutputGeometry
	Region         parser.TileListTileRegion
	AnchorFrameIdx uint8
	ReferenceCount uint8
}

// PlanTileListEntryDecode prepares one tile-list entry for residual decoding.
// references receives the libaom-shaped reference table for the synthetic entry
// decode: references[0] is LAST_FRAME and points at anchors[anchor_frame_idx].
// The returned Event and Step can be supplied to the frame-work residual runner
// with output set to a caller-owned scratch reconstruction frame; after the
// entry decodes, copy the returned Region into the tile-list output mosaic.
func PlanTileListEntryDecode(event Event, entryIndex int, surface int, anchors []*frame.Frame, references []*frame.Frame, workers int, spans []parser.TileSpan, jobs []tile.Job, batches []threading.Batch) (TileListEntryDecodePlan, error) {
	if event.Kind != EventTileList {
		return TileListEntryDecodePlan{}, ErrInvalidSurfaceEvent
	}
	if event.TileListErr != nil {
		return TileListEntryDecodePlan{}, event.TileListErr
	}
	if surface < 0 {
		return TileListEntryDecodePlan{}, ErrInvalidSurfaceReference
	}
	if len(references) < 1 {
		return TileListEntryDecodePlan{}, ErrSurfaceReferenceBufferTooSmall
	}

	tilePlan, err := PlanTileListEntryWork(event.TileList, event.TileInfo, entryIndex, workers, spans, jobs, batches)
	if err != nil {
		return TileListEntryDecodePlan{}, err
	}
	geometry, err := parser.TileListOutputGeometryForGrid(event.TileList, event.TileInfo, event.SequenceHeader.Use128x128Superblock)
	if err != nil {
		return TileListEntryDecodePlan{}, err
	}
	region, err := parser.TileListOutputTileRegion(event.TileList, geometry, entryIndex)
	if err != nil {
		return TileListEntryDecodePlan{}, err
	}

	entry := event.TileList.Entries[entryIndex]
	anchor := int(entry.AnchorFrameIdx)
	if anchor < 0 || anchor >= parser.TileListMaxExternalReferences || anchor >= len(anchors) || anchors[anchor] == nil {
		return TileListEntryDecodePlan{}, parser.ErrTileListInvalidAnchorIndex
	}

	references[0] = anchors[anchor]
	decodeEvent := event
	decodeEvent.Kind = EventTileGroup
	decodeEvent.Type = obu.TypeTileGroup
	decodeEvent.Unit.Payload = entry.TileData
	decodeEvent.Unit.Raw = entry.TileData
	decodeEvent.Unit.Size = uint32(len(entry.TileData))
	decodeEvent.Unit.Header = obu.Header{Type: obu.TypeTileGroup}
	decodeEvent.TileGroup = parser.TileGroup{}

	referenceCount, err := frameWorkReferenceCount8(1)
	if err != nil {
		return TileListEntryDecodePlan{}, err
	}
	step := FrameWorkStep{
		Kind: FrameWorkStepTile,
		Tile: FrameTileWorkPlan{
			Surface:        surface,
			ReferenceCount: referenceCount,
			Tile:           tilePlan,
		},
	}
	return TileListEntryDecodePlan{
		Event:          decodeEvent,
		Step:           step,
		Payload:        entry.TileData,
		Geometry:       geometry,
		Region:         region,
		AnchorFrameIdx: entry.AnchorFrameIdx,
		ReferenceCount: referenceCount,
	}, nil
}
