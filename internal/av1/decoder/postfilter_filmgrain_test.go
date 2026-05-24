package decoder

import (
	"errors"
	"testing"

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
