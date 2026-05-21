package decoder

import (
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/parser"
)

func TestSurfaceReferencesRefresh(t *testing.T) {
	var refs SurfaceReferences
	if _, ok := refs.ReferenceSlot(0); ok {
		t.Fatal("zero-value references should start empty")
	}

	var releases [parser.RefFrames]int
	count, err := refs.Refresh(0x03, 5, releases[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 0 {
		t.Fatalf("release count=%d", count)
	}
	if slot, ok := refs.ReferenceSlot(0); !ok || slot != 5 {
		t.Fatalf("ref[0]=%d ok=%v", slot, ok)
	}
	if slot, ok := refs.ReferenceSlot(1); !ok || slot != 5 {
		t.Fatalf("ref[1]=%d ok=%v", slot, ok)
	}

	count, err = refs.Refresh(0x01, 6, releases[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 0 || refs.Holds(5) != true || refs.Holds(6) != true {
		t.Fatalf("count=%d holds5=%v holds6=%v", count, refs.Holds(5), refs.Holds(6))
	}

	count, err = refs.Refresh(0x02, 7, releases[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 1 || releases[0] != 5 || refs.Holds(5) || !refs.Holds(6) || !refs.Holds(7) {
		t.Fatalf("count=%d releases=%v holds5=%v holds6=%v holds7=%v", count, releases[:count], refs.Holds(5), refs.Holds(6), refs.Holds(7))
	}
}

func TestSurfaceReferencesReleaseUniqueOnce(t *testing.T) {
	var refs SurfaceReferences
	var releases [parser.RefFrames]int
	if _, err := refs.Refresh(0x03, 5, releases[:]); err != nil {
		t.Fatal(err)
	}

	count, err := refs.Refresh(0x03, 6, releases[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 1 || releases[0] != 5 {
		t.Fatalf("count=%d releases=%v", count, releases[:count])
	}
}

func TestSurfaceReferencesFinishFrame(t *testing.T) {
	var refs SurfaceReferences
	var releases [parser.RefFrames]int
	event := Event{
		Kind: EventTileGroup,
		FrameSize: parser.FrameSize{
			RefreshFrameFlags: 0x03,
		},
		TileGroup: parser.TileGroup{Final: true},
	}

	count, err := refs.FinishFrame(event, 5, releases[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 0 {
		t.Fatalf("count=%d", count)
	}
	if slot, ok := refs.ReferenceSlot(0); !ok || slot != 5 {
		t.Fatalf("ref[0]=%d ok=%v", slot, ok)
	}
	if slot, ok := refs.ReferenceSlot(1); !ok || slot != 5 {
		t.Fatalf("ref[1]=%d ok=%v", slot, ok)
	}
}

func TestSurfaceReferencesFinishFrameRejectsIncompleteEvent(t *testing.T) {
	var refs SurfaceReferences
	var releases [parser.RefFrames]int
	_, err := refs.FinishFrame(Event{Kind: EventTileGroup}, 5, releases[:])
	if !errors.Is(err, ErrInvalidSurfaceEvent) {
		t.Fatalf("FinishFrame non-final err=%v want %v", err, ErrInvalidSurfaceEvent)
	}
	_, err = refs.FinishFrame(Event{Kind: EventFrameHeader, TileGroup: parser.TileGroup{Final: true}}, 5, releases[:])
	if !errors.Is(err, ErrInvalidSurfaceEvent) {
		t.Fatalf("FinishFrame wrong kind err=%v want %v", err, ErrInvalidSurfaceEvent)
	}
}

func TestSurfaceReferencesRefreshRejectsShortReleaseBuffer(t *testing.T) {
	var refs SurfaceReferences
	var releases [parser.RefFrames]int
	if _, err := refs.Refresh(0x01, 5, releases[:]); err != nil {
		t.Fatal(err)
	}

	_, err := refs.Refresh(0x01, 6, nil)
	if !errors.Is(err, ErrSurfaceReleaseBufferTooSmall) {
		t.Fatalf("Refresh err=%v want %v", err, ErrSurfaceReleaseBufferTooSmall)
	}
	if slot, ok := refs.ReferenceSlot(0); !ok || slot != 5 {
		t.Fatalf("ref[0]=%d ok=%v", slot, ok)
	}
}

func TestSurfaceReferencesShowExistingFrameEvent(t *testing.T) {
	var refs SurfaceReferences
	var releases [parser.RefFrames]int
	if _, err := refs.Refresh(0x05, 5, releases[:]); err != nil {
		t.Fatal(err)
	}
	if _, err := refs.Refresh(0x02, 6, releases[:]); err != nil {
		t.Fatal(err)
	}

	surface, count, err := refs.ShowExistingFrame(Event{
		Kind:        EventExistingFrame,
		FrameHeader: parser.FrameHeaderPrefix{ShowExistingFrame: true, ExistingFrameIdx: 1},
		ExistingFrame: parser.ReferenceFrame{
			FrameType: parser.FrameTypeKey,
		},
	}, releases[:])
	if err != nil {
		t.Fatal(err)
	}
	if surface != 6 || count != 1 || releases[0] != 5 {
		t.Fatalf("surface=%d count=%d releases=%v", surface, count, releases[:count])
	}
	for i := 0; i < parser.RefFrames; i++ {
		slot, ok := refs.ReferenceSlot(i)
		if !ok || slot != 6 {
			t.Fatalf("ref[%d]=%d ok=%v", i, slot, ok)
		}
	}
}

func TestSurfaceReferencesShowExistingFrameEventRejectsMissingSurface(t *testing.T) {
	var refs SurfaceReferences
	var releases [parser.RefFrames]int
	_, _, err := refs.ShowExistingFrame(Event{
		Kind:        EventExistingFrame,
		FrameHeader: parser.FrameHeaderPrefix{ShowExistingFrame: true, ExistingFrameIdx: 1},
	}, releases[:])
	if !errors.Is(err, ErrInvalidSurfaceReference) {
		t.Fatalf("ShowExistingFrame err=%v want %v", err, ErrInvalidSurfaceReference)
	}
	_, _, err = refs.ShowExistingFrame(Event{Kind: EventFrameHeader}, releases[:])
	if !errors.Is(err, ErrInvalidSurfaceEvent) {
		t.Fatalf("ShowExistingFrame wrong kind err=%v want %v", err, ErrInvalidSurfaceEvent)
	}
}

func TestSurfaceReferencesShowExistingKeyResetsReferences(t *testing.T) {
	var refs SurfaceReferences
	var releases [parser.RefFrames]int
	if _, err := refs.Refresh(0x05, 5, releases[:]); err != nil {
		t.Fatal(err)
	}
	if _, err := refs.Refresh(0x02, 6, releases[:]); err != nil {
		t.Fatal(err)
	}

	count, err := refs.ShowExisting(6, parser.FrameTypeKey, releases[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 1 || releases[0] != 5 {
		t.Fatalf("count=%d releases=%v", count, releases[:count])
	}
	for i := 0; i < parser.RefFrames; i++ {
		slot, ok := refs.ReferenceSlot(i)
		if !ok || slot != 6 {
			t.Fatalf("ref[%d]=%d ok=%v", i, slot, ok)
		}
	}
}

func TestSurfaceReferencesShowExistingInterLeavesReferences(t *testing.T) {
	var refs SurfaceReferences
	var releases [parser.RefFrames]int
	if _, err := refs.Refresh(0x01, 5, releases[:]); err != nil {
		t.Fatal(err)
	}

	count, err := refs.ShowExisting(6, parser.FrameTypeInter, releases[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 0 {
		t.Fatalf("count=%d", count)
	}
	if slot, ok := refs.ReferenceSlot(0); !ok || slot != 5 {
		t.Fatalf("ref[0]=%d ok=%v", slot, ok)
	}
	if refs.Holds(6) {
		t.Fatal("inter show-existing should not retain new surface")
	}
}

func TestSurfaceReferencesRejectsInvalidSurface(t *testing.T) {
	var refs SurfaceReferences
	var releases [parser.RefFrames]int
	_, err := refs.Refresh(0x01, -1, releases[:])
	if !errors.Is(err, ErrInvalidSurfaceReference) {
		t.Fatalf("Refresh err=%v want %v", err, ErrInvalidSurfaceReference)
	}
	_, err = refs.ShowExisting(-1, parser.FrameTypeKey, releases[:])
	if !errors.Is(err, ErrInvalidSurfaceReference) {
		t.Fatalf("ShowExisting err=%v want %v", err, ErrInvalidSurfaceReference)
	}
}

func TestSurfaceReferencesAllocs(t *testing.T) {
	var refs SurfaceReferences
	var releases [parser.RefFrames]int
	allocs := testing.AllocsPerRun(1000, func() {
		_, err := refs.Refresh(0xff, 5, releases[:])
		if err != nil {
			t.Fatal(err)
		}
		_, err = refs.ShowExisting(6, parser.FrameTypeKey, releases[:])
		if err != nil {
			t.Fatal(err)
		}
		_, err = refs.FinishFrame(Event{
			Kind:      EventFrame,
			FrameSize: parser.FrameSize{RefreshFrameFlags: 0xff},
			TileGroup: parser.TileGroup{Final: true},
		}, 5, releases[:])
		if err != nil {
			t.Fatal(err)
		}
		refs.Reset()
	})
	if allocs != 0 {
		t.Fatalf("SurfaceReferences allocated: %f", allocs)
	}
}

func BenchmarkSurfaceReferencesRefresh(b *testing.B) {
	var refs SurfaceReferences
	var releases [parser.RefFrames]int
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, _ = refs.Refresh(0xff, i&7, releases[:])
	}
}
