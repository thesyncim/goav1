package goav1

import (
	"errors"
	"testing"
)

func TestDecoderFinishFrameSurface(t *testing.T) {
	format := FrameFormat{Width: 16, Height: 16, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32}
	layout, err := FrameRequiredSize(format)
	if err != nil {
		t.Fatal(err)
	}
	backing := make([]byte, layout.Size*2)
	var frames [2]Frame
	var free [2]int
	var used [2]bool
	pool, err := BindFramePool(backing, format, frames[:], free[:], used[:])
	if err != nil {
		t.Fatal(err)
	}
	sequence := SequenceHeader{ColorConfig: ColorConfig{
		BitDepth:     8,
		SubsamplingX: true,
		SubsamplingY: true,
	}}
	size := FrameSize{CodedWidth: 16, Height: 16}

	var refs DecoderSurfaceReferences
	plan0, _, err := BeginDecoderFrameWork(&refs, &pool, sequence, DecoderEvent{
		Kind:        DecoderEventFrameHeader,
		FrameHeader: FrameHeaderPrefix{FrameType: FrameTypeKey},
		FrameSize:   size,
	}, 32, nil, 1, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if plan0.Surface != 0 || plan0.ReferenceCount != 0 || plan0.Tile != (DecoderTileWorkPlan{}) {
		t.Fatalf("plan0=%+v", plan0)
	}
	plan1, _, err := BeginDecoderFrameWork(&refs, &pool, sequence, DecoderEvent{
		Kind:        DecoderEventFrameHeader,
		FrameHeader: FrameHeaderPrefix{FrameType: FrameTypeKey},
		FrameSize:   size,
	}, 32, nil, 1, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if plan1.Surface != 1 || plan1.ReferenceCount != 0 || plan1.Tile != (DecoderTileWorkPlan{}) {
		t.Fatalf("plan1=%+v", plan1)
	}

	var releases [RefFrames]int
	if _, err := refs.Refresh(0xff, plan0.Surface, releases[:]); err != nil {
		t.Fatal(err)
	}

	count, err := DecoderFinishFrameSurface(&refs, &pool, DecoderEvent{
		Kind:      DecoderEventTileGroup,
		FrameSize: FrameSize{RefreshFrameFlags: 0xff},
		TileGroup: TileGroup{Final: true},
	}, plan1.Surface, releases[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 1 || releases[0] != plan0.Surface {
		t.Fatalf("count=%d release=%d want 1,%d", count, releases[0], plan0.Surface)
	}
	if _, err := pool.Frame(plan0.Surface); !errors.Is(err, ErrFrameInvalidSlot) {
		t.Fatalf("released frame err=%v want %v", err, ErrFrameInvalidSlot)
	}
}

func TestPlanDecoderFrameTileWork(t *testing.T) {
	event := DecoderEvent{
		Kind: DecoderEventTileGroup,
		Unit: OBUUnit{Payload: []byte{0x80}},
		TileInfo: TileInfo{
			SBCols:     1,
			SBRows:     1,
			Cols:       1,
			Rows:       1,
			ColStartSB: [MaxTileCols + 1]uint16{0, 1},
			RowStartSB: [MaxTileRows + 1]uint16{0, 1},
		},
		TileGroup: TileGroup{
			TileCount:  1,
			DataOffset: 0,
			DataSize:   1,
			Final:      true,
		},
	}
	var spans [1]TileSpan
	var jobs [1]TileJob
	var batches [1]TileBatch
	plan, err := PlanDecoderFrameTileWork(event, 3, 0, 1, spans[:], jobs[:], batches[:])
	if err != nil {
		t.Fatal(err)
	}
	if plan.Surface != 3 || plan.ReferenceCount != 0 || plan.Tile != (DecoderTileWorkPlan{SpanCount: 1, JobCount: 1, BatchCount: 1}) {
		t.Fatalf("plan=%+v", plan)
	}
}
