package decoder

import (
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/filmgrain"
	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/parser"
)

func TestFrameWorkPostFilterContextGenerateFilmGrainLumaGrain(t *testing.T) {
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
				ARCoeffShift:  6,
			},
		},
		Output: output,
	}
	result, err := ctx.GenerateFilmGrainLumaGrain(make([]int16, filmgrain.LumaGrainSamples))
	if err != nil {
		t.Fatal(err)
	}
	if !result.Active || result.Width != filmgrain.LumaGrainWidth ||
		result.Height != filmgrain.LumaGrainHeight ||
		result.Stride != filmgrain.LumaGrainWidth {
		t.Fatalf("result=%+v", result)
	}
	if result.Grain[0] != -29 || result.Grain[15] != 26 || result.Grain[3*result.Stride+3] != -14 {
		t.Fatalf("grain samples=%d %d %d", result.Grain[0], result.Grain[15], result.Grain[3*result.Stride+3])
	}
}

func TestFrameWorkPostFilterContextGenerateFilmGrainLumaGrainSkipsInactive(t *testing.T) {
	result, err := (FrameWorkPostFilterContext{}).GenerateFilmGrainLumaGrain(nil)
	if err != nil {
		t.Fatal(err)
	}
	if result.Active || result.Grain != nil || result.Width != 0 || result.Height != 0 || result.Stride != 0 {
		t.Fatalf("result=%+v want zero", result)
	}
}

func TestFrameWorkPostFilterContextGenerateFilmGrainLumaGrainRejectsShortScratch(t *testing.T) {
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: 64, Height: 32, BitDepth: 8, Align: 32})
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			FilmGrain: parser.FilmGrainParams{
				Apply:        true,
				BitDepth:     8,
				NumYPoints:   1,
				ARCoeffShift: 6,
			},
		},
		Output: output,
	}
	_, err := ctx.GenerateFilmGrainLumaGrain(make([]int16, filmgrain.LumaGrainSamples-1))
	if !errors.Is(err, frame.ErrShortBuffer) {
		t.Fatalf("GenerateFilmGrainLumaGrain err=%v want %v", err, frame.ErrShortBuffer)
	}
}
