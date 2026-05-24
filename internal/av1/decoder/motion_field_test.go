package decoder

import (
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/tile"
)

func TestTemporalMotionReferencePublishAndResolve(t *testing.T) {
	mvFrame := testDecoderReferenceMVFrame(t)
	store := make([]tile.TemporalMotionReferenceFrame, 4)
	event := Event{
		Kind:      EventFrame,
		TileGroup: parser.TileGroup{Final: true},
		FrameHeader: parser.FrameHeaderPrefix{
			FrameType: parser.FrameTypeInter,
			OrderHint: 9,
		},
		ReferenceOrderHints: [parser.InterRefsPerFrame]uint32{1, 2, 3, 4, 5, 6, 7},
	}
	if err := PublishTemporalMotionReference(event, 2, mvFrame, store); err != nil {
		t.Fatal(err)
	}
	if store[2].Frame != mvFrame || store[2].OrderHint != 9 ||
		store[2].RefOrderHints != event.ReferenceOrderHints ||
		store[2].IntraOnly {
		t.Fatalf("stored reference=%+v", store[2])
	}

	var out [parser.InterRefsPerFrame]tile.TemporalMotionReferenceFrame
	count, err := ResolveTemporalMotionReferences([]int{2}, store, out[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 1 || out[0].Frame != mvFrame || out[0].OrderHint != 9 {
		t.Fatalf("count=%d out=%+v", count, out[0])
	}
}

func TestTemporalMotionReferencePublishMarksIntraOnly(t *testing.T) {
	mvFrame := testDecoderReferenceMVFrame(t)
	store := make([]tile.TemporalMotionReferenceFrame, 1)
	err := PublishTemporalMotionReference(Event{
		Kind:        EventTileGroup,
		TileGroup:   parser.TileGroup{Final: true},
		FrameHeader: parser.FrameHeaderPrefix{FrameType: parser.FrameTypeKey},
	}, 0, mvFrame, store)
	if err != nil {
		t.Fatal(err)
	}
	if !store[0].IntraOnly {
		t.Fatalf("reference=%+v want intra-only", store[0])
	}
}

func TestTemporalMotionReferenceResolveIsAtomic(t *testing.T) {
	store := make([]tile.TemporalMotionReferenceFrame, 2)
	store[0] = tile.TemporalMotionReferenceFrame{Frame: testDecoderReferenceMVFrame(t), OrderHint: 4}
	out := [2]tile.TemporalMotionReferenceFrame{
		{OrderHint: 99},
		{OrderHint: 100},
	}
	_, err := ResolveTemporalMotionReferences([]int{0, 1}, store, out[:])
	if !errors.Is(err, ErrInvalidSurfaceReference) {
		t.Fatalf("err=%v want %v", err, ErrInvalidSurfaceReference)
	}
	if out[0].OrderHint != 99 || out[1].OrderHint != 100 {
		t.Fatalf("resolve modified output on error: %+v", out)
	}
}

func TestTemporalMotionReferenceRejectsInvalidInputs(t *testing.T) {
	mvFrame := testDecoderReferenceMVFrame(t)
	store := make([]tile.TemporalMotionReferenceFrame, 1)
	if err := PublishTemporalMotionReference(Event{Kind: EventFrame}, 0, mvFrame, store); !errors.Is(err, ErrInvalidSurfaceEvent) {
		t.Fatalf("non-final publish err=%v want %v", err, ErrInvalidSurfaceEvent)
	}
	if err := PublishTemporalMotionReference(Event{Kind: EventFrame, TileGroup: parser.TileGroup{Final: true}}, -1, mvFrame, store); !errors.Is(err, ErrInvalidSurfaceReference) {
		t.Fatalf("bad surface err=%v want %v", err, ErrInvalidSurfaceReference)
	}
	if err := PublishTemporalMotionReference(Event{Kind: EventFrame, TileGroup: parser.TileGroup{Final: true}}, 1, mvFrame, store); !errors.Is(err, ErrSurfaceReferenceBufferTooSmall) {
		t.Fatalf("short store err=%v want %v", err, ErrSurfaceReferenceBufferTooSmall)
	}
	if _, err := ResolveTemporalMotionReferences([]int{0}, store, nil); !errors.Is(err, ErrSurfaceReferenceBufferTooSmall) {
		t.Fatalf("short dst err=%v want %v", err, ErrSurfaceReferenceBufferTooSmall)
	}
	tooMany := make([]int, parser.InterRefsPerFrame+1)
	if _, err := ResolveTemporalMotionReferences(tooMany, store, nil); !errors.Is(err, ErrInvalidSurfaceReference) {
		t.Fatalf("too many err=%v want %v", err, ErrInvalidSurfaceReference)
	}
}

func TestTemporalMotionReferenceAllocs(t *testing.T) {
	mvFrame := testDecoderReferenceMVFrame(t)
	store := make([]tile.TemporalMotionReferenceFrame, 1)
	event := Event{
		Kind:        EventFrame,
		TileGroup:   parser.TileGroup{Final: true},
		FrameHeader: parser.FrameHeaderPrefix{FrameType: parser.FrameTypeInter},
	}
	var out [parser.InterRefsPerFrame]tile.TemporalMotionReferenceFrame
	allocs := testing.AllocsPerRun(1000, func() {
		if err := PublishTemporalMotionReference(event, 0, mvFrame, store); err != nil {
			t.Fatal(err)
		}
		if _, err := ResolveTemporalMotionReferences([]int{0}, store, out[:]); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("temporal motion reference helpers allocated: %f", allocs)
	}
}

func testDecoderReferenceMVFrame(t *testing.T) *tile.ReferenceMVFrame {
	t.Helper()
	need, err := tile.ReferenceMVFrameEntries(16, 16)
	if err != nil {
		t.Fatal(err)
	}
	frame := &tile.ReferenceMVFrame{}
	if err := frame.Init(16, 16, make([]tile.ReferenceMVEntry, need)); err != nil {
		t.Fatal(err)
	}
	return frame
}
