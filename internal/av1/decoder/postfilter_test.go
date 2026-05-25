package decoder

import (
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/tile"
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

func TestFrameWorkPostFilterScratchSizeMax(t *testing.T) {
	a := FrameWorkPostFilterScratchSize{
		LoopFilter: FrameWorkLoopFilterPostFilterScratchSize{Edges: 1},
		CDEF: FrameWorkCDEFPostFilterScratchSize{
			Samples:       [3]int{1, 8, 3},
			Dst:           [3]int{4, 2, 6},
			DirectionGrid: 5,
			VarianceGrid:  9,
			Input:         7,
			UnitDst:       11,
		},
		SuperRes: FrameWorkSuperResPostFilterScratchSize{
			OutputFrame:   32,
			CodedSamples:  [3]int{6, 2, 8},
			OutputSamples: [3]int{3, 10, 4},
		},
		Restoration: FrameWorkRestorationPostFilterScratchSize{
			Samples: tile.RestorationFrameSampleScratchSize{
				Data:    [3]frame.BorderedSamplePlaneLayout{{Stride: 4, Origin: 2, Rows: 3, Len: 12}},
				Dst:     [3]frame.BorderedSamplePlaneLayout{{Len: 1}, {Stride: 2, Origin: 3, Rows: 4, Len: 5}},
				DataLen: 20,
				DstLen:  7,
			},
			Apply: tile.RestorationUnitRecordBoundaryScratchSize{
				Unit:     tile.RestorationUnitScratchSize{Wiener: 14, SGRProj: 2},
				Boundary: tile.RestorationStripeBoundaryScratchSize{Above: 6, Below: 1},
			},
		},
		FilmGrain: FrameWorkFilmGrainPostFilterScratchSize{
			ScalingPoints: [3]int{1, 7, 3},
			ARCoeffs:      [3]int{5, 2, 9},
			LumaGrain:     12,
			ChromaGrain:   [2]int{6, 4},
			LumaSamples:   10,
			ChromaSamples: [2]int{8, 2},
			LumaLine:      3,
			LumaColumn:    11,
		},
	}
	b := FrameWorkPostFilterScratchSize{
		LoopFilter: FrameWorkLoopFilterPostFilterScratchSize{Edges: 3},
		CDEF: FrameWorkCDEFPostFilterScratchSize{
			Samples:       [3]int{2, 4, 9},
			Dst:           [3]int{5, 7, 1},
			DirectionGrid: 4,
			VarianceGrid:  10,
			Input:         12,
			UnitDst:       8,
		},
		SuperRes: FrameWorkSuperResPostFilterScratchSize{
			OutputFrame:   64,
			CodedSamples:  [3]int{7, 1, 9},
			OutputSamples: [3]int{8, 6, 5},
		},
		Restoration: FrameWorkRestorationPostFilterScratchSize{
			Samples: tile.RestorationFrameSampleScratchSize{
				Data:    [3]frame.BorderedSamplePlaneLayout{{Stride: 5, Origin: 1, Rows: 6, Len: 9}, {Len: 3}},
				Dst:     [3]frame.BorderedSamplePlaneLayout{{Len: 2}, {Stride: 1, Origin: 4, Rows: 3, Len: 8}},
				DataLen: 18,
				DstLen:  9,
			},
			Apply: tile.RestorationUnitRecordBoundaryScratchSize{
				Unit:     tile.RestorationUnitScratchSize{Wiener: 9, SGRProj: 5},
				Boundary: tile.RestorationStripeBoundaryScratchSize{Above: 3, Below: 8},
			},
		},
		FilmGrain: FrameWorkFilmGrainPostFilterScratchSize{
			ScalingPoints: [3]int{2, 5, 4},
			ARCoeffs:      [3]int{3, 8, 7},
			LumaGrain:     9,
			ChromaGrain:   [2]int{7, 3},
			LumaSamples:   11,
			ChromaSamples: [2]int{5, 6},
			LumaLine:      13,
			LumaColumn:    10,
		},
	}
	want := FrameWorkPostFilterScratchSize{
		LoopFilter: FrameWorkLoopFilterPostFilterScratchSize{Edges: 3},
		CDEF: FrameWorkCDEFPostFilterScratchSize{
			Samples:       [3]int{2, 8, 9},
			Dst:           [3]int{5, 7, 6},
			DirectionGrid: 5,
			VarianceGrid:  10,
			Input:         12,
			UnitDst:       11,
		},
		SuperRes: FrameWorkSuperResPostFilterScratchSize{
			OutputFrame:   64,
			CodedSamples:  [3]int{7, 2, 9},
			OutputSamples: [3]int{8, 10, 5},
		},
		Restoration: FrameWorkRestorationPostFilterScratchSize{
			Samples: tile.RestorationFrameSampleScratchSize{
				Data:    [3]frame.BorderedSamplePlaneLayout{{Stride: 5, Origin: 2, Rows: 6, Len: 12}, {Len: 3}},
				Dst:     [3]frame.BorderedSamplePlaneLayout{{Len: 2}, {Stride: 2, Origin: 4, Rows: 4, Len: 8}},
				DataLen: 20,
				DstLen:  9,
			},
			Apply: tile.RestorationUnitRecordBoundaryScratchSize{
				Unit:     tile.RestorationUnitScratchSize{Wiener: 14, SGRProj: 5},
				Boundary: tile.RestorationStripeBoundaryScratchSize{Above: 6, Below: 8},
			},
		},
		FilmGrain: FrameWorkFilmGrainPostFilterScratchSize{
			ScalingPoints: [3]int{2, 7, 4},
			ARCoeffs:      [3]int{5, 8, 9},
			LumaGrain:     12,
			ChromaGrain:   [2]int{7, 4},
			LumaSamples:   11,
			ChromaSamples: [2]int{8, 6},
			LumaLine:      13,
			LumaColumn:    11,
		},
	}
	if got := a.Max(b); got != want {
		t.Fatalf("a.Max(b)=%+v want %+v", got, want)
	}
	if got := b.Max(a); got != want {
		t.Fatalf("b.Max(a)=%+v want %+v", got, want)
	}
}

func TestFrameWorkPostFilterScratchSizeBindRequest(t *testing.T) {
	size := testFrameWorkPostFilterScratchSize()
	scratch := testFrameWorkPostFilterScratchStorage(size)
	var outputView frame.Frame
	options := FrameWorkPostFilterBindOptions{
		CDEFIndexMap: FrameWorkCDEFIndexMap{
			Index:  make([]uint8, 4),
			Read:   make([]bool, 4),
			Stride: 2,
			Rows:   2,
		},
		SuperResOutputView: &outputView,
		RestorationRecords: [3][]tile.RestorationUnitRecord{
			{{Index: 3}},
		},
		RestorationBoundaries: [3]tile.RestorationStripeBoundaries{
			{Stride: 16},
		},
		RestorationOptimized: true,
	}
	req, err := size.BindRequest(options, scratch)
	if err != nil {
		t.Fatal(err)
	}
	if len(req.LoopFilter.Edges) != size.LoopFilter.Edges ||
		len(req.CDEF.SampleScratch[0]) != size.CDEF.Samples[0] ||
		len(req.SuperRes.OutputFrame) != size.SuperRes.OutputFrame ||
		req.SuperRes.OutputView != &outputView ||
		len(req.Restoration.DataScratch) != size.Restoration.Samples.DataLen ||
		len(req.Restoration.Scratch.Unit.Wiener) != size.Restoration.Apply.Unit.Wiener ||
		!req.Restoration.Optimized ||
		len(req.Restoration.Records[0]) != 1 ||
		req.Restoration.Boundaries[0].Stride != 16 ||
		len(req.FilmGrain.LumaGrain) != size.FilmGrain.LumaGrain ||
		len(req.FilmGrain.ChromaSamples[1]) != size.FilmGrain.ChromaSamples[1] {
		t.Fatalf("request=%+v", req)
	}
}

func TestFrameWorkPostFilterScratchSizeBindRequestRejectsShortStageScratch(t *testing.T) {
	size := testFrameWorkPostFilterScratchSize()
	scratch := testFrameWorkPostFilterScratchStorage(size)
	tests := []struct {
		name   string
		mutate func(*FrameWorkPostFilterScratch)
		want   error
	}{
		{name: "loop-filter", mutate: func(s *FrameWorkPostFilterScratch) { s.LoopFilterEdges = s.LoopFilterEdges[:len(s.LoopFilterEdges)-1] }, want: frame.ErrShortBuffer},
		{name: "cdef", mutate: func(s *FrameWorkPostFilterScratch) { s.CDEFInput = s.CDEFInput[:len(s.CDEFInput)-1] }, want: frame.ErrShortBuffer},
		{name: "superres", mutate: func(s *FrameWorkPostFilterScratch) {
			s.SuperResOutputFrame = s.SuperResOutputFrame[:len(s.SuperResOutputFrame)-1]
		}, want: frame.ErrShortBuffer},
		{name: "restoration", mutate: func(s *FrameWorkPostFilterScratch) { s.RestorationData = s.RestorationData[:len(s.RestorationData)-1] }, want: tile.ErrJobBufferTooSmall},
		{name: "film-grain", mutate: func(s *FrameWorkPostFilterScratch) {
			s.FilmGrainLumaSamples = s.FilmGrainLumaSamples[:len(s.FilmGrainLumaSamples)-1]
		}, want: frame.ErrShortBuffer},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			short := scratch
			tt.mutate(&short)
			if _, err := size.BindRequest(FrameWorkPostFilterBindOptions{}, short); !errors.Is(err, tt.want) {
				t.Fatalf("BindRequest err=%v want %v", err, tt.want)
			}
		})
	}
}

func TestFrameWorkPostFilterScratchSizeBindRequestAllocs(t *testing.T) {
	size := testFrameWorkPostFilterScratchSize()
	scratch := testFrameWorkPostFilterScratchStorage(size)
	allocs := testing.AllocsPerRun(1000, func() {
		if _, err := size.BindRequest(FrameWorkPostFilterBindOptions{}, scratch); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("BindRequest allocated: %f", allocs)
	}
}

func BenchmarkFrameWorkPostFilterScratchSizeBindRequest(b *testing.B) {
	size := testFrameWorkPostFilterScratchSize()
	scratch := testFrameWorkPostFilterScratchStorage(size)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := size.BindRequest(FrameWorkPostFilterBindOptions{}, scratch); err != nil {
			b.Fatal(err)
		}
	}
}

func testFrameWorkPostFilterScratchSize() FrameWorkPostFilterScratchSize {
	return FrameWorkPostFilterScratchSize{
		LoopFilter: FrameWorkLoopFilterPostFilterScratchSize{Edges: 8},
		CDEF: FrameWorkCDEFPostFilterScratchSize{
			Samples:       [3]int{64, 32, 32},
			Dst:           [3]int{64, 32, 32},
			DirectionGrid: 4,
			VarianceGrid:  4,
			Input:         16,
			UnitDst:       16,
		},
		SuperRes: FrameWorkSuperResPostFilterScratchSize{
			OutputFrame:   128,
			CodedSamples:  [3]int{64, 32, 32},
			OutputSamples: [3]int{96, 48, 48},
		},
		Restoration: FrameWorkRestorationPostFilterScratchSize{
			Samples: tile.RestorationFrameSampleScratchSize{
				DataLen: 256,
				DstLen:  256,
			},
			Apply: tile.RestorationUnitRecordBoundaryScratchSize{
				Unit:     tile.RestorationUnitScratchSize{Wiener: 32, SGRProj: 24},
				Boundary: tile.RestorationStripeBoundaryScratchSize{Above: 16, Below: 16},
			},
		},
		FilmGrain: FrameWorkFilmGrainPostFilterScratchSize{
			LumaGrain:     64,
			ChromaGrain:   [2]int{32, 32},
			LumaSamples:   128,
			ChromaSamples: [2]int{64, 64},
		},
	}
}

func testFrameWorkPostFilterScratchStorage(size FrameWorkPostFilterScratchSize) FrameWorkPostFilterScratch {
	cdefSamples, cdefDst, directionGrid, varianceGrid, input, unitDst := testFrameWorkCDEFScratchStorage(size.CDEF)
	superResOutputFrame, superResCoded, superResOutput := testFrameWorkSuperResScratchStorage(size.SuperRes)
	restorationData, restorationDst, restorationWiener, restorationSGR, restorationAbove, restorationBelow := testFrameWorkRestorationPostFilterScratchStorage(size.Restoration)
	lumaGrain, chromaGrain, lumaSamples, chromaSamples := testFrameWorkFilmGrainScratchStorage(size.FilmGrain)
	return FrameWorkPostFilterScratch{
		LoopFilterEdges: make([]FrameWorkLoopFilterPostFilterEdge, maxInt(size.LoopFilter.Edges, 0)),

		CDEFSamples:       cdefSamples,
		CDEFDst:           cdefDst,
		CDEFDirectionGrid: directionGrid,
		CDEFVarianceGrid:  varianceGrid,
		CDEFInput:         input,
		CDEFUnitDst:       unitDst,

		SuperResOutputFrame: superResOutputFrame,
		SuperResCoded:       superResCoded,
		SuperResOutput:      superResOutput,

		RestorationData:   restorationData,
		RestorationDst:    restorationDst,
		RestorationWiener: restorationWiener,
		RestorationSGR:    restorationSGR,
		RestorationAbove:  restorationAbove,
		RestorationBelow:  restorationBelow,

		FilmGrainLumaGrain:     lumaGrain,
		FilmGrainChromaGrain:   chromaGrain,
		FilmGrainLumaSamples:   lumaSamples,
		FilmGrainChromaSamples: chromaSamples,
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
