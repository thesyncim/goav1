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

func TestSurfaceReferencesReleaseAll(t *testing.T) {
	var refs SurfaceReferences
	var releases [parser.RefFrames]int
	if _, err := refs.Refresh(0x05, 5, releases[:]); err != nil {
		t.Fatal(err)
	}
	if _, err := refs.Refresh(0x02, 6, releases[:]); err != nil {
		t.Fatal(err)
	}

	count, err := refs.ReleaseAll(releases[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 2 || releases[0] != 5 || releases[1] != 6 || refs.Holds(5) || refs.Holds(6) {
		t.Fatalf("count=%d releases=%v holds5=%v holds6=%v", count, releases[:count], refs.Holds(5), refs.Holds(6))
	}
	if _, ok := refs.ReferenceSlot(0); ok {
		t.Fatal("reference slots were not reset")
	}
}

func TestSurfaceReferencesReleaseAllRejectsShortReleaseBuffer(t *testing.T) {
	var refs SurfaceReferences
	var releases [parser.RefFrames]int
	if _, err := refs.Refresh(0x05, 5, releases[:]); err != nil {
		t.Fatal(err)
	}
	if _, err := refs.Refresh(0x02, 6, releases[:]); err != nil {
		t.Fatal(err)
	}

	_, err := refs.ReleaseAll(releases[:1])
	if !errors.Is(err, ErrSurfaceReleaseBufferTooSmall) {
		t.Fatalf("ReleaseAll err=%v want %v", err, ErrSurfaceReleaseBufferTooSmall)
	}
	if !refs.Holds(5) || !refs.Holds(6) {
		t.Fatalf("references changed on error: holds5=%v holds6=%v", refs.Holds(5), refs.Holds(6))
	}
}

func TestSurfaceReferencesFrameReferences(t *testing.T) {
	var refs SurfaceReferences
	var releases [parser.RefFrames]int
	for i := range parser.RefFrames {
		if _, err := refs.Refresh(1<<uint(i), 10+i, releases[:]); err != nil {
			t.Fatal(err)
		}
	}

	event := Event{
		Kind:        EventFrameHeader,
		FrameHeader: parser.FrameHeaderPrefix{FrameType: parser.FrameTypeInter},
		FrameSize: parser.FrameSize{
			RefFrameIdx: [parser.InterRefsPerFrame]uint8{0, 1, 2, 3, 4, 5, 6},
		},
	}
	var surfaces [parser.InterRefsPerFrame]int
	count, err := refs.FrameReferences(event, surfaces[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != parser.InterRefsPerFrame {
		t.Fatalf("count=%d", count)
	}
	for i := range parser.InterRefsPerFrame {
		if surfaces[i] != 10+i {
			t.Fatalf("surface[%d]=%d want %d", i, surfaces[i], 10+i)
		}
	}
}

func TestSurfaceReferencesFrameReferencesIntra(t *testing.T) {
	var refs SurfaceReferences
	count, err := refs.FrameReferences(Event{
		Kind:        EventFrameHeader,
		FrameHeader: parser.FrameHeaderPrefix{FrameType: parser.FrameTypeKey},
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if count != 0 {
		t.Fatalf("count=%d", count)
	}
}

func TestSurfaceReferencesFrameReferencesRejectsInvalidInputs(t *testing.T) {
	var refs SurfaceReferences
	var surfaces [parser.InterRefsPerFrame]int
	for i := range len(surfaces) {
		surfaces[i] = -99
	}

	_, err := refs.FrameReferences(Event{Kind: EventExistingFrame}, surfaces[:])
	if !errors.Is(err, ErrInvalidSurfaceEvent) {
		t.Fatalf("FrameReferences event err=%v want %v", err, ErrInvalidSurfaceEvent)
	}
	_, err = refs.FrameReferences(Event{
		Kind:        EventFrameHeader,
		FrameHeader: parser.FrameHeaderPrefix{ShowExistingFrame: true, FrameType: parser.FrameTypeInter},
	}, surfaces[:])
	if !errors.Is(err, ErrInvalidSurfaceEvent) {
		t.Fatalf("FrameReferences show-existing err=%v want %v", err, ErrInvalidSurfaceEvent)
	}
	_, err = refs.FrameReferences(Event{
		Kind:        EventFrameHeader,
		FrameHeader: parser.FrameHeaderPrefix{FrameType: parser.FrameTypeInter},
	}, surfaces[:parser.InterRefsPerFrame-1])
	if !errors.Is(err, ErrSurfaceReferenceBufferTooSmall) {
		t.Fatalf("FrameReferences short buffer err=%v want %v", err, ErrSurfaceReferenceBufferTooSmall)
	}
	_, err = refs.FrameReferences(Event{
		Kind:        EventFrameHeader,
		FrameHeader: parser.FrameHeaderPrefix{FrameType: parser.FrameTypeInter},
	}, surfaces[:])
	if !errors.Is(err, ErrInvalidSurfaceReference) {
		t.Fatalf("FrameReferences missing surface err=%v want %v", err, ErrInvalidSurfaceReference)
	}
	for i := range len(surfaces) {
		if surfaces[i] != -99 {
			t.Fatalf("surface[%d]=%d changed on error", i, surfaces[i])
		}
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
	for i := range parser.RefFrames {
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
	for i := range parser.RefFrames {
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
		var surfaces [parser.InterRefsPerFrame]int
		_, err = refs.FrameReferences(Event{
			Kind:        EventFrameHeader,
			FrameHeader: parser.FrameHeaderPrefix{FrameType: parser.FrameTypeInter},
			FrameSize: parser.FrameSize{
				RefFrameIdx: [parser.InterRefsPerFrame]uint8{0, 1, 2, 3, 4, 5, 6},
			},
		}, surfaces[:])
		if err != nil {
			t.Fatal(err)
		}
		_, err = refs.ReleaseAll(releases[:])
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
