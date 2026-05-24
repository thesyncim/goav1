package decoder

import (
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/parser"
)

func TestFrameWorkPostFilterContextRequireNoActivePostFiltersAllowsInactiveSyntax(t *testing.T) {
	tests := []struct {
		name  string
		event Event
	}{
		{
			name: "loopfilter-tuning-without-levels",
			event: Event{
				LoopFilter: parser.LoopFilterParams{
					Sharpness:           7,
					ModeRefDeltaEnabled: true,
					ModeRefDeltaUpdate:  true,
					Deltas: parser.LoopFilterDeltas{
						Ref:  [parser.RefFrames]int8{1},
						Mode: [parser.LoopFilterModeDeltas]int8{1},
					},
				},
			},
		},
		{
			name: "cdef-selection-with-zero-strengths",
			event: Event{
				CDEF: parser.CDEFParams{
					Damping:       3,
					StrengthCount: 1,
				},
			},
		},
		{
			name: "superres-denominator-without-superres",
			event: Event{
				FrameSize: parser.FrameSize{SuperResDenominator: 8},
			},
		},
		{
			name: "restoration-units-without-restoration-type",
			event: Event{
				Restoration: parser.RestorationParams{
					UnitSizeYLog2:  6,
					UnitSizeUVLog2: 6,
					UnitSizeY:      64,
					UnitSizeUV:     64,
				},
			},
		},
		{
			name: "film-grain-params-not-applied",
			event: Event{
				FilmGrain: parser.FilmGrainParams{
					ParamsPresent: true,
					Update:        true,
					Seed:          1,
					NumYPoints:    1,
					YPoints:       [parser.MaxFilmGrainYPoints][2]uint8{{0, 16}},
				},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := (FrameWorkPostFilterContext{Event: tt.event}).RequireNoActivePostFilters(); err != nil {
				t.Fatalf("RequireNoActivePostFilters err=%v", err)
			}
		})
	}
}

func TestFrameWorkPostFilterContextRequireNoActivePostFiltersRejectsActiveComponents(t *testing.T) {
	tests := []struct {
		name  string
		event Event
	}{
		{
			name: "loopfilter-y0",
			event: Event{
				LoopFilter: parser.LoopFilterParams{LevelY: [2]uint8{1, 0}},
			},
		},
		{
			name: "loopfilter-y1",
			event: Event{
				LoopFilter: parser.LoopFilterParams{LevelY: [2]uint8{0, 1}},
			},
		},
		{
			name: "loopfilter-u",
			event: Event{
				LoopFilter: parser.LoopFilterParams{LevelU: 1},
			},
		},
		{
			name: "loopfilter-v",
			event: Event{
				LoopFilter: parser.LoopFilterParams{LevelV: 1},
			},
		},
		{
			name: "cdef-index-bits",
			event: Event{
				CDEF: parser.CDEFParams{Bits: 1},
			},
		},
		{
			name: "cdef-y-strength",
			event: Event{
				CDEF: parser.CDEFParams{StrengthCount: 1, YStrength: [parser.MaxCDEFStrengths]uint8{1}},
			},
		},
		{
			name: "cdef-uv-strength",
			event: Event{
				CDEF: parser.CDEFParams{StrengthCount: 1, UVStrength: [parser.MaxCDEFStrengths]uint8{1}},
			},
		},
		{
			name: "superres",
			event: Event{
				FrameSize: parser.FrameSize{SuperResEnabled: true},
			},
		},
		{
			name: "restoration-y",
			event: Event{
				Restoration: parser.RestorationParams{Type: [3]parser.RestorationType{parser.RestorationWiener}},
			},
		},
		{
			name: "restoration-u",
			event: Event{
				Restoration: parser.RestorationParams{Type: [3]parser.RestorationType{1: parser.RestorationWiener}},
			},
		},
		{
			name: "restoration-v",
			event: Event{
				Restoration: parser.RestorationParams{Type: [3]parser.RestorationType{2: parser.RestorationWiener}},
			},
		},
		{
			name: "film-grain-apply",
			event: Event{
				FilmGrain: parser.FilmGrainParams{Apply: true},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := (FrameWorkPostFilterContext{Event: tt.event}).RequireNoActivePostFilters()
			if !errors.Is(err, ErrUnsupportedPostFilter) {
				t.Fatalf("RequireNoActivePostFilters err=%v want %v", err, ErrUnsupportedPostFilter)
			}
		})
	}
}

func TestFrameWorkPostFilterContextRemainingPostFilters(t *testing.T) {
	active := FrameWorkPostFilterLoopFilter |
		FrameWorkPostFilterCDEF |
		FrameWorkPostFilterSuperRes |
		FrameWorkPostFilterLoopRestoration |
		FrameWorkPostFilterFilmGrain
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			LoopFilter:  parser.LoopFilterParams{LevelY: [2]uint8{1}},
			CDEF:        parser.CDEFParams{Bits: 1},
			FrameSize:   parser.FrameSize{SuperResEnabled: true},
			Restoration: parser.RestorationParams{Type: [3]parser.RestorationType{parser.RestorationWiener}},
			FilmGrain:   parser.FilmGrainParams{Apply: true},
		},
	}
	if got := ctx.ActivePostFilters(); got != active {
		t.Fatalf("active=%b want %b", got, active)
	}
	if got := ctx.RemainingPostFilters(); got != active {
		t.Fatalf("remaining=%b want %b", got, active)
	}

	ctx = ctx.WithCompletedPostFilters(FrameWorkPostFilterLoopFilter | FrameWorkPostFilterCDEF | FrameWorkPostFilterSuperRes | FrameWorkPostFilterStage(0x80))
	want := FrameWorkPostFilterLoopRestoration | FrameWorkPostFilterFilmGrain
	if got := ctx.RemainingPostFilters(); got != want {
		t.Fatalf("remaining after pre-restoration stages=%b want %b", got, want)
	}
	if err := ctx.RequireNoActivePostFilters(); !errors.Is(err, ErrUnsupportedPostFilter) {
		t.Fatalf("RequireNoActivePostFilters err=%v want %v", err, ErrUnsupportedPostFilter)
	}
	if err := ctx.RequireNoRemainingPostFilters(); !errors.Is(err, ErrUnsupportedPostFilter) {
		t.Fatalf("RequireNoRemainingPostFilters err=%v want %v", err, ErrUnsupportedPostFilter)
	}

	ctx = ctx.WithCompletedPostFilters(FrameWorkPostFilterLoopRestoration | FrameWorkPostFilterFilmGrain)
	if got := ctx.RemainingPostFilters(); got != 0 {
		t.Fatalf("remaining after all stages=%b want 0", got)
	}
	if err := ctx.RequireNoRemainingPostFilters(); err != nil {
		t.Fatalf("RequireNoRemainingPostFilters err=%v", err)
	}
	if err := ctx.RequireNoActivePostFilters(); !errors.Is(err, ErrUnsupportedPostFilter) {
		t.Fatalf("RequireNoActivePostFilters err=%v want %v", err, ErrUnsupportedPostFilter)
	}
}

func TestFrameWorkStateRunStepWithPostFilterGateRejectsBeforePublish(t *testing.T) {
	pool := testFramePool(t, 1)
	surface, output, err := pool.Acquire()
	if err != nil {
		t.Fatal(err)
	}
	output.Y.Pix[0] = 0x55

	var refs SurfaceReferences
	var releases [parser.RefFrames]int
	state := FrameWorkState{Surface: surface, active: true}
	event := finalFrameEvent(0xff)
	event.LoopFilter = parser.LoopFilterParams{LevelY: [2]uint8{1, 0}}
	step := FrameWorkStep{
		Kind: FrameWorkStepTile,
		Tile: FrameTileWorkPlan{Surface: surface},
	}

	result, err := state.RunStepWithPostFilter(&refs, &pool, event, step, nil, nil, nil, releases[:], nil, func(ctx FrameWorkPostFilterContext) error {
		return ctx.RequireNoActivePostFilters()
	})
	if !errors.Is(err, ErrUnsupportedPostFilter) {
		t.Fatalf("RunStepWithPostFilter err=%v want %v", err, ErrUnsupportedPostFilter)
	}
	if result != (FrameWorkStepResult{}) || !state.Active() {
		t.Fatalf("result=%+v active=%v", result, state.Active())
	}
	for i := 0; i < parser.RefFrames; i++ {
		if slot, ok := refs.ReferenceSlot(i); ok || slot != -1 {
			t.Fatalf("reference[%d] slot=%d ok=%v want unpublished", i, slot, ok)
		}
	}
	got, err := pool.Frame(surface)
	if err != nil {
		t.Fatal(err)
	}
	if got != output || got.Y.Pix[0] != 0x55 {
		t.Fatalf("active output=%p sample=%d want %p sample=85", got, got.Y.Pix[0], output)
	}
}

func TestFrameWorkStateRunStepWithPostFilterSkipsHookUntilFinalTileGroup(t *testing.T) {
	pool := testFramePool(t, 1)
	surface, _, err := pool.Acquire()
	if err != nil {
		t.Fatal(err)
	}

	var refs SurfaceReferences
	var releases [parser.RefFrames]int
	state := FrameWorkState{Surface: surface, active: true}
	event := Event{
		Kind:      EventTileGroup,
		FrameSize: parser.FrameSize{RefreshFrameFlags: 0xff},
		TileGroup: parser.TileGroup{Final: false},
	}
	step := FrameWorkStep{
		Kind: FrameWorkStepTile,
		Tile: FrameTileWorkPlan{Surface: surface},
	}

	called := false
	result, err := state.RunStepWithPostFilter(&refs, &pool, event, step, nil, nil, nil, releases[:], nil, func(FrameWorkPostFilterContext) error {
		called = true
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if called {
		t.Fatal("postfilter hook ran before the final tile group")
	}
	if result != (FrameWorkStepResult{}) || !state.Active() {
		t.Fatalf("result=%+v active=%v", result, state.Active())
	}
	for i := 0; i < parser.RefFrames; i++ {
		if slot, ok := refs.ReferenceSlot(i); ok || slot != -1 {
			t.Fatalf("reference[%d] slot=%d ok=%v want unpublished", i, slot, ok)
		}
	}
}
