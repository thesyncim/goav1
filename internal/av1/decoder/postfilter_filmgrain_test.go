package decoder

import (
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/filmgrain"
	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/parser"
)

func TestFrameWorkPostFilterContextFilmGrainPostFilterPlanSkipsInactive(t *testing.T) {
	plan, err := (FrameWorkPostFilterContext{}).FilmGrainPostFilterPlan()
	if err != nil {
		t.Fatal(err)
	}
	if plan != (FrameWorkFilmGrainPostFilterPlan{}) {
		t.Fatalf("plan=%+v want zero", plan)
	}
	size, err := (FrameWorkPostFilterContext{}).FilmGrainPostFilterScratchLen()
	if err != nil {
		t.Fatal(err)
	}
	if size != (FrameWorkFilmGrainPostFilterScratchSize{}) {
		t.Fatalf("scratch=%+v want zero", size)
	}
}

func TestFrameWorkPostFilterContextFilmGrainPostFilterPlanReportsLumaInputs(t *testing.T) {
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: 64, Height: 32, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32})
	event := Event{
		FilmGrain: parser.FilmGrainParams{
			ParamsPresent:         true,
			Apply:                 true,
			Update:                true,
			Seed:                  0x1234,
			BitDepth:              8,
			NumYPoints:            1,
			YPoints:               [parser.MaxFilmGrainYPoints][2]uint8{{16, 32}},
			ScalingShift:          9,
			ARCoeffLag:            1,
			ARCoeffShift:          8,
			GrainScaleShift:       1,
			Overlap:               true,
			ClipToRestrictedRange: true,
		},
	}
	ctx := FrameWorkPostFilterContext{Event: event, Output: output}
	plan, err := ctx.FilmGrainPostFilterPlan()
	if err != nil {
		t.Fatal(err)
	}
	if !plan.Active || plan.Seed != 0x1234 || plan.BitDepth != 8 || !plan.Overlap || !plan.ClipToRestrictedRange {
		t.Fatalf("plan=%+v", plan)
	}
	if plan.Planes[0] != (FrameWorkFilmGrainPostFilterPlanePlan{Active: true, ScalingPoints: 1, ARCoeffs: 4, Width: 64, Height: 32, Stride: 64}) {
		t.Fatalf("luma plane=%+v", plan.Planes[0])
	}
	if plan.Planes[1].Active || plan.Planes[2].Active {
		t.Fatalf("chroma planes=%+v %+v", plan.Planes[1], plan.Planes[2])
	}
	size, err := ctx.FilmGrainPostFilterScratchLen()
	if err != nil {
		t.Fatal(err)
	}
	if size.ScalingPoints != [3]int{1, 0, 0} || size.ARCoeffs != [3]int{4, 0, 0} {
		t.Fatalf("scratch=%+v", size)
	}
	if size.LumaGrain != filmgrain.LumaGrainSamples ||
		size.LumaLine != filmgrain.LumaOverlapSamples*plan.Planes[0].Stride ||
		size.LumaColumn != filmgrain.LumaColumnScratchRows*filmgrain.LumaOverlapSamples {
		t.Fatalf("luma scratch=%+v", size)
	}
}

func TestFrameWorkPostFilterContextFilmGrainPostFilterPlanReportsChromaScalingFromLuma(t *testing.T) {
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: 64, Height: 32, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32})
	event := Event{
		FilmGrain: parser.FilmGrainParams{
			ParamsPresent:         true,
			Apply:                 true,
			Seed:                  0x4433,
			BitDepth:              8,
			ChromaScalingFromLuma: true,
			NumYPoints:            1,
			YPoints:               [parser.MaxFilmGrainYPoints][2]uint8{{4, 12}},
			ARCoeffLag:            1,
		},
	}
	ctx := FrameWorkPostFilterContext{Event: event, Output: output}
	plan, err := ctx.FilmGrainPostFilterPlan()
	if err != nil {
		t.Fatal(err)
	}
	if !plan.ChromaScalingFromLuma || !plan.Planes[1].Active || !plan.Planes[2].Active {
		t.Fatalf("plan=%+v", plan)
	}
	if plan.Planes[1].Width != 32 || plan.Planes[1].Height != 16 || plan.Planes[1].ARCoeffs != 5 {
		t.Fatalf("cb plane=%+v", plan.Planes[1])
	}
	if plan.Planes[2].Width != 32 || plan.Planes[2].Height != 16 || plan.Planes[2].ARCoeffs != 5 {
		t.Fatalf("cr plane=%+v", plan.Planes[2])
	}
	size, err := ctx.FilmGrainPostFilterScratchLen()
	if err != nil {
		t.Fatal(err)
	}
	if size.ScalingPoints != [3]int{1, 0, 0} || size.ARCoeffs != [3]int{4, 5, 5} {
		t.Fatalf("scratch=%+v", size)
	}
}

func TestFrameWorkPostFilterContextFilmGrainPostFilterPlanReportsSampleStride(t *testing.T) {
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: 48, Height: 16, BitDepth: 10, Align: 64})
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			FilmGrain: parser.FilmGrainParams{
				ParamsPresent: true,
				Apply:         true,
				BitDepth:      10,
				NumYPoints:    1,
				YPoints:       [parser.MaxFilmGrainYPoints][2]uint8{{0, 16}},
			},
		},
		Output: output,
	}
	plan, err := ctx.FilmGrainPostFilterPlan()
	if err != nil {
		t.Fatal(err)
	}
	if plan.Planes[0].Width != 48 || plan.Planes[0].Stride != 64 {
		t.Fatalf("luma plane=%+v", plan.Planes[0])
	}
}

func TestFrameWorkPostFilterContextFilmGrainPostFilterScalingLUTs(t *testing.T) {
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: 64, Height: 32, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32})
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			FilmGrain: parser.FilmGrainParams{
				ParamsPresent: true,
				Apply:         true,
				BitDepth:      8,
				NumYPoints:    2,
				YPoints:       [parser.MaxFilmGrainYPoints][2]uint8{{10, 20}, {20, 40}},
				NumCbPoints:   1,
				CbPoints:      [parser.MaxFilmGrainUVPoints][2]uint8{{0, 7}},
				NumCrPoints:   2,
				CrPoints:      [parser.MaxFilmGrainUVPoints][2]uint8{{0, 100}, {10, 0}},
			},
		},
		Output: output,
	}
	luts, err := ctx.FilmGrainPostFilterScalingLUTs()
	if err != nil {
		t.Fatal(err)
	}
	if !luts.Active {
		t.Fatalf("luts=%+v", luts)
	}
	if luts.LUTs[0][0] != 20 || luts.LUTs[0][15] != 30 || luts.LUTs[0][255] != 40 {
		t.Fatalf("y lut samples=%d %d %d", luts.LUTs[0][0], luts.LUTs[0][15], luts.LUTs[0][255])
	}
	if luts.LUTs[1][0] != 7 || luts.LUTs[1][255] != 7 {
		t.Fatalf("cb lut samples=%d %d", luts.LUTs[1][0], luts.LUTs[1][255])
	}
	if luts.LUTs[2][5] != 50 || luts.LUTs[2][255] != 0 {
		t.Fatalf("cr lut samples=%d %d", luts.LUTs[2][5], luts.LUTs[2][255])
	}
}

func TestFrameWorkPostFilterContextFilmGrainPostFilterScalingLUTsCopiesLumaForChroma(t *testing.T) {
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: 64, Height: 32, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32})
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			FilmGrain: parser.FilmGrainParams{
				ParamsPresent:         true,
				Apply:                 true,
				BitDepth:              8,
				ChromaScalingFromLuma: true,
				NumYPoints:            2,
				YPoints:               [parser.MaxFilmGrainYPoints][2]uint8{{10, 20}, {20, 40}},
			},
		},
		Output: output,
	}
	luts, err := ctx.FilmGrainPostFilterScalingLUTs()
	if err != nil {
		t.Fatal(err)
	}
	for _, x := range []int{0, 15, 255} {
		if luts.LUTs[1][x] != luts.LUTs[0][x] || luts.LUTs[2][x] != luts.LUTs[0][x] {
			t.Fatalf("lut[%d] y=%d cb=%d cr=%d", x, luts.LUTs[0][x], luts.LUTs[1][x], luts.LUTs[2][x])
		}
	}
}

func TestFrameWorkPostFilterContextFilmGrainPostFilterScalingLUTsRejectsInvalidPoints(t *testing.T) {
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: 64, Height: 32, BitDepth: 8, Align: 32})
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			FilmGrain: parser.FilmGrainParams{
				ParamsPresent: true,
				Apply:         true,
				BitDepth:      8,
				NumYPoints:    2,
				YPoints:       [parser.MaxFilmGrainYPoints][2]uint8{{10, 20}, {10, 40}},
			},
		},
		Output: output,
	}
	if _, err := ctx.FilmGrainPostFilterScalingLUTs(); !errors.Is(err, frame.ErrInvalidFormat) {
		t.Fatalf("FilmGrainPostFilterScalingLUTs err=%v want %v", err, frame.ErrInvalidFormat)
	}
}

func TestFrameWorkPostFilterContextApplySupportedPostFiltersCompletesNoOpFilmGrain(t *testing.T) {
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: 64, Height: 32, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32})
	output.Y.Pix[0] = 0x5a
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			FilmGrain: parser.FilmGrainParams{
				ParamsPresent: true,
				Apply:         true,
				BitDepth:      8,
			},
		},
		Output: output,
	}
	size, err := ctx.SupportedPostFilterScratchLen(FrameWorkPostFilterRequest{})
	if err != nil {
		t.Fatal(err)
	}
	if size.FilmGrain != (FrameWorkFilmGrainPostFilterScratchSize{}) {
		t.Fatalf("scratch=%+v", size.FilmGrain)
	}
	next, result, err := ctx.ApplySupportedPostFilters(FrameWorkPostFilterRequest{})
	if err != nil {
		t.Fatal(err)
	}
	if result.Completed != FrameWorkPostFilterFilmGrain || !result.FilmGrain.NoOp || !result.FilmGrain.Plan.Active {
		t.Fatalf("result=%+v", result)
	}
	if got := next.RemainingPostFilters(); got != 0 {
		t.Fatalf("remaining=%b want 0", got)
	}
	if output.Y.Pix[0] != 0x5a {
		t.Fatalf("output sample=%d want 0x5a", output.Y.Pix[0])
	}
}

func TestFrameWorkPostFilterContextApplySupportedPostFiltersRejectsActiveFilmGrainBeforeMutation(t *testing.T) {
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: 64, Height: 32, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32})
	output.Y.Pix[0] = 0x44
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			FrameSize: parser.FrameSize{
				CodedWidth:          64,
				UpscaledWidth:       64,
				Height:              32,
				SuperResDenominator: 8,
			},
			CDEF: parser.CDEFParams{
				Damping:       5,
				StrengthCount: 1,
				YStrength:     [parser.MaxCDEFStrengths]uint8{63},
			},
			FilmGrain: parser.FilmGrainParams{
				ParamsPresent: true,
				Apply:         true,
				BitDepth:      8,
				NumYPoints:    1,
				YPoints:       [parser.MaxFilmGrainYPoints][2]uint8{{0, 16}},
			},
		},
		Output: output,
	}
	if _, err := ctx.SupportedPostFilterScratchLen(FrameWorkPostFilterRequest{}); !errors.Is(err, ErrUnsupportedPostFilter) {
		t.Fatalf("SupportedPostFilterScratchLen err=%v want %v", err, ErrUnsupportedPostFilter)
	}
	next, result, err := ctx.ApplySupportedPostFilters(FrameWorkPostFilterRequest{})
	if !errors.Is(err, ErrUnsupportedPostFilter) {
		t.Fatalf("ApplySupportedPostFilters err=%v want %v", err, ErrUnsupportedPostFilter)
	}
	if next.RemainingPostFilters() != ctx.RemainingPostFilters() || result != (FrameWorkPostFilterResult{}) {
		t.Fatalf("next remaining=%b result=%+v", next.RemainingPostFilters(), result)
	}
	if output.Y.Pix[0] != 0x44 {
		t.Fatalf("output sample=%d want 0x44", output.Y.Pix[0])
	}
}

func TestFrameWorkPostFilterContextFilmGrainPostFilterPlanRejectsNilOutput(t *testing.T) {
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			FilmGrain: parser.FilmGrainParams{Apply: true, BitDepth: 8},
		},
	}
	if _, err := ctx.FilmGrainPostFilterPlan(); !errors.Is(err, frame.ErrInvalidSlot) {
		t.Fatalf("FilmGrainPostFilterPlan err=%v want %v", err, frame.ErrInvalidSlot)
	}
}

func TestFrameWorkPostFilterContextFilmGrainPostFilterPlanRejectsInvalidActiveState(t *testing.T) {
	tests := []struct {
		name   string
		format frame.Format
		grain  parser.FilmGrainParams
	}{
		{
			name:   "bitdepth",
			format: frame.Format{Width: 64, Height: 32, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32},
			grain:  parser.FilmGrainParams{Apply: true, BitDepth: 10},
		},
		{
			name:   "monochrome-chroma",
			format: frame.Format{Width: 64, Height: 32, BitDepth: 8, MonoChrome: true, Align: 32},
			grain:  parser.FilmGrainParams{Apply: true, BitDepth: 8, ChromaScalingFromLuma: true},
		},
		{
			name:   "lag",
			format: frame.Format{Width: 64, Height: 32, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32},
			grain:  parser.FilmGrainParams{Apply: true, BitDepth: 8, ARCoeffLag: 4},
		},
		{
			name:   "420-one-sided-chroma",
			format: frame.Format{Width: 64, Height: 32, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32},
			grain:  parser.FilmGrainParams{Apply: true, BitDepth: 8, NumCbPoints: 1},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			output := testFrameWorkCDEFFrame(t, tt.format)
			ctx := FrameWorkPostFilterContext{Event: Event{FilmGrain: tt.grain}, Output: output}
			if _, err := ctx.FilmGrainPostFilterPlan(); !errors.Is(err, frame.ErrInvalidFormat) {
				t.Fatalf("FilmGrainPostFilterPlan err=%v want %v", err, frame.ErrInvalidFormat)
			}
		})
	}
}
