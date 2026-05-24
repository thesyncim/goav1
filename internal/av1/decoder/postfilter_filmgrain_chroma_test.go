package decoder

import (
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/filmgrain"
	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/parser"
)

func TestFrameWorkPostFilterContextGenerateFilmGrainChromaGrain(t *testing.T) {
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: 64, Height: 32, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32})
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			FilmGrain: parser.FilmGrainParams{
				ParamsPresent: true,
				Apply:         true,
				Seed:          0x1234,
				BitDepth:      8,
				NumYPoints:    1,
				YPoints:       [parser.MaxFilmGrainYPoints][2]uint8{{0, 16}},
				NumCbPoints:   1,
				CbPoints:      [parser.MaxFilmGrainUVPoints][2]uint8{{0, 16}},
				NumCrPoints:   1,
				CrPoints:      [parser.MaxFilmGrainUVPoints][2]uint8{{0, 16}},
				ARCoeffShift:  6,
				ARCoeffsCb:    [parser.MaxFilmGrainUVCoeffs]int8{0: 64},
			},
		},
		Output: output,
	}
	luma := testFrameWorkFilmGrainLumaTemplate(32)
	result, err := ctx.GenerateFilmGrainChromaGrain(make([]int16, filmgrain.ChromaGrainSamples), luma, filmgrain.ChromaPlaneCb)
	if err != nil {
		t.Fatal(err)
	}
	if !result.Active || result.Plane != filmgrain.ChromaPlaneCb ||
		result.Width != filmgrain.ChromaSubsampledGrainWidth ||
		result.Height != filmgrain.ChromaSubsampledGrainHeight ||
		result.Stride != filmgrain.ChromaGrainWidth {
		t.Fatalf("result=%+v", result)
	}
	if result.Grain[0] != 12 || result.Grain[15] != 13 || result.Grain[3*result.Stride+3] != 46 {
		t.Fatalf("cb grain samples=%d %d %d", result.Grain[0], result.Grain[15], result.Grain[3*result.Stride+3])
	}

	cr, err := ctx.GenerateFilmGrainChromaGrain(make([]int16, filmgrain.ChromaGrainSamples), luma, filmgrain.ChromaPlaneCr)
	if err != nil {
		t.Fatal(err)
	}
	if !cr.Active || cr.Plane != filmgrain.ChromaPlaneCr || cr.Grain[0] != 5 || cr.Grain[15] != 3 {
		t.Fatalf("cr result=%+v samples=%d %d", cr, cr.Grain[0], cr.Grain[15])
	}
}

func TestFrameWorkPostFilterContextGenerateFilmGrainChromaGrainSkipsInactive(t *testing.T) {
	result, err := (FrameWorkPostFilterContext{}).GenerateFilmGrainChromaGrain(nil, nil, filmgrain.ChromaPlaneCb)
	if err != nil {
		t.Fatal(err)
	}
	if result.Active || result.Plane != 0 || result.Grain != nil || result.Width != 0 || result.Height != 0 || result.Stride != 0 {
		t.Fatalf("result=%+v want zero", result)
	}
}

func TestFrameWorkPostFilterContextGenerateFilmGrainChromaGrainRejectsInvalidPlane(t *testing.T) {
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: 64, Height: 32, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32})
	ctx := FrameWorkPostFilterContext{
		Event: Event{FilmGrain: parser.FilmGrainParams{
			Apply:        true,
			BitDepth:     8,
			NumCbPoints:  1,
			NumCrPoints:  1,
			ARCoeffShift: 6,
		}},
		Output: output,
	}
	if _, err := ctx.GenerateFilmGrainChromaGrain(make([]int16, filmgrain.ChromaGrainSamples), nil, 0); !errors.Is(err, frame.ErrInvalidFormat) {
		t.Fatalf("GenerateFilmGrainChromaGrain err=%v want %v", err, frame.ErrInvalidFormat)
	}
}

func TestFrameWorkPostFilterContextGenerateFilmGrainChromaGrainRejectsShortScratch(t *testing.T) {
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: 64, Height: 32, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32})
	ctx := FrameWorkPostFilterContext{
		Event: Event{FilmGrain: parser.FilmGrainParams{
			Apply:        true,
			BitDepth:     8,
			NumYPoints:   1,
			NumCbPoints:  1,
			NumCrPoints:  1,
			ARCoeffShift: 6,
		}},
		Output: output,
	}
	luma := testFrameWorkFilmGrainLumaTemplate(32)
	if _, err := ctx.GenerateFilmGrainChromaGrain(make([]int16, filmgrain.ChromaGrainSamples-1), luma, filmgrain.ChromaPlaneCb); !errors.Is(err, frame.ErrShortBuffer) {
		t.Fatalf("short chroma scratch err=%v want %v", err, frame.ErrShortBuffer)
	}
	if _, err := ctx.GenerateFilmGrainChromaGrain(make([]int16, filmgrain.ChromaGrainSamples), luma[:filmgrain.LumaGrainSamples-1], filmgrain.ChromaPlaneCb); !errors.Is(err, frame.ErrShortBuffer) {
		t.Fatalf("short luma scratch err=%v want %v", err, frame.ErrShortBuffer)
	}
}

func TestFrameWorkPostFilterContextGenerateFilmGrainChromaGrainAllocs(t *testing.T) {
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: 64, Height: 32, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32})
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			FilmGrain: parser.FilmGrainParams{
				ParamsPresent: true,
				Apply:         true,
				Seed:          0x1234,
				BitDepth:      8,
				NumYPoints:    1,
				YPoints:       [parser.MaxFilmGrainYPoints][2]uint8{{0, 16}},
				NumCbPoints:   1,
				CbPoints:      [parser.MaxFilmGrainUVPoints][2]uint8{{0, 16}},
				NumCrPoints:   1,
				CrPoints:      [parser.MaxFilmGrainUVPoints][2]uint8{{0, 16}},
				ARCoeffLag:    1,
				ARCoeffShift:  6,
				ARCoeffsCb:    [parser.MaxFilmGrainUVCoeffs]int8{3: 64, 4: 32},
			},
		},
		Output: output,
	}
	grain := make([]int16, filmgrain.ChromaGrainSamples)
	luma := testFrameWorkFilmGrainLumaTemplate(32)
	allocs := testing.AllocsPerRun(1000, func() {
		result, err := ctx.GenerateFilmGrainChromaGrain(grain, luma, filmgrain.ChromaPlaneCb)
		if err != nil {
			t.Fatal(err)
		}
		if !result.Active || len(result.Grain) != filmgrain.ChromaGrainSamples {
			t.Fatalf("result=%+v", result)
		}
	})
	if allocs != 0 {
		t.Fatalf("GenerateFilmGrainChromaGrain allocated: %f", allocs)
	}
}

func testFrameWorkFilmGrainLumaTemplate(value int16) []int16 {
	grain := make([]int16, filmgrain.LumaGrainSamples)
	for i := range grain {
		grain[i] = value
	}
	return grain
}
