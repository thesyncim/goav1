package decoder

import (
	"reflect"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/parser"
)

func TestPostFilterPrivateQueriesPreserveValueSnapshots(t *testing.T) {
	const all = FrameWorkPostFilterLoopFilter | FrameWorkPostFilterCDEF |
		FrameWorkPostFilterSuperRes | FrameWorkPostFilterLoopRestoration | FrameWorkPostFilterFilmGrain
	masks := []FrameWorkPostFilterStage{0, FrameWorkPostFilterLoopFilter, FrameWorkPostFilterCDEF, all, 0x80, 0xff}
	for active := FrameWorkPostFilterStage(0); active <= all; active++ {
		var event Event
		if active.Has(FrameWorkPostFilterLoopFilter) {
			event.LoopFilter.LevelY[1] = 1
		}
		if active.Has(FrameWorkPostFilterCDEF) {
			event.CDEF.Bits = 1
		}
		event.FrameSize.SuperResEnabled = active.Has(FrameWorkPostFilterSuperRes)
		if active.Has(FrameWorkPostFilterLoopRestoration) {
			event.Restoration.Type[2] = parser.RestorationSGRProj
		}
		event.FilmGrain.Apply = active.Has(FrameWorkPostFilterFilmGrain)
		for _, completed := range masks {
			ctx := FrameWorkPostFilterContext{Event: event, ReferenceCount: 7,
				ExecutedTileWork: true, completedPostFilters: completed, detachedPostFilterOutput: true}
			before := ctx
			if got := ctx.ActivePostFilters(); got != active {
				t.Fatalf("active=%#x want=%#x", got, active)
			}
			if got := frameWorkActivePostFilters(&ctx); got != active {
				t.Fatalf("private active=%#x want=%#x", got, active)
			}
			wantRemaining := active &^ completed
			if ctx.RemainingPostFilters() != wantRemaining || frameWorkRemainingPostFilters(&ctx) != wantRemaining {
				t.Fatalf("remaining differs: active=%#x completed=%#x", active, completed)
			}
			for _, mark := range masks {
				got := ctx.WithCompletedPostFilters(mark)
				private := ctx
				frameWorkMarkCompletedPostFilters(&private, mark)
				want := before
				want.completedPostFilters |= mark & active
				if !reflect.DeepEqual(got, want) || !reflect.DeepEqual(private, want) {
					t.Fatalf("completion changed fields: active=%#x completed=%#x mark=%#x", active, completed, mark)
				}
			}
			if !reflect.DeepEqual(ctx, before) {
				t.Fatal("value methods or read-only private queries mutated the original context")
			}
		}
	}
}

func TestPostFilterPrivateQueriesObserveEventChanges(t *testing.T) {
	ctx := FrameWorkPostFilterContext{Event: Event{CDEF: parser.CDEFParams{Bits: 1}}}
	snapshot := ctx.WithCompletedPostFilters(FrameWorkPostFilterCDEF)
	ctx.Event.CDEF.Bits = 0
	ctx.Event.FrameSize.SuperResEnabled = true
	if frameWorkActivePostFilters(&ctx) != FrameWorkPostFilterSuperRes ||
		frameWorkRemainingPostFilters(&ctx) != FrameWorkPostFilterSuperRes {
		t.Fatal("private query reused a stale event mask")
	}
	if snapshot.ActivePostFilters() != FrameWorkPostFilterCDEF || snapshot.RemainingPostFilters() != 0 {
		t.Fatal("editing the original event changed the returned value snapshot")
	}
}
