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

func TestFrameWorkPostFilterContextGenerateFilmGrainLumaGrainAllocs(t *testing.T) {
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
				ARCoeffLag:    1,
				ARCoeffShift:  6,
				ARCoeffsY:     [parser.MaxFilmGrainYCoeffs]int8{3: 64},
			},
		},
		Output: output,
	}
	grain := make([]int16, filmgrain.LumaGrainSamples)
	allocs := testing.AllocsPerRun(1000, func() {
		result, err := ctx.GenerateFilmGrainLumaGrain(grain)
		if err != nil {
			t.Fatal(err)
		}
		if !result.Active || len(result.Grain) != filmgrain.LumaGrainSamples {
			t.Fatalf("result=%+v", result)
		}
	})
	if allocs != 0 {
		t.Fatalf("GenerateFilmGrainLumaGrain allocated: %f", allocs)
	}
}

func TestFrameWorkPostFilterContextApplyFilmGrainLumaRow(t *testing.T) {
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: 34, Height: 32, BitDepth: 8, Align: 32})
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			FilmGrain: parser.FilmGrainParams{
				ParamsPresent: true,
				Apply:         true,
				Seed:          0,
				BitDepth:      8,
				NumYPoints:    1,
				YPoints:       [parser.MaxFilmGrainYPoints][2]uint8{{0, 64}},
				ScalingShift:  8,
				Overlap:       true,
			},
		},
		Output: output,
	}
	plan, err := ctx.FilmGrainPostFilterPlan()
	if err != nil {
		t.Fatal(err)
	}
	luts, err := ctx.FilmGrainPostFilterScalingLUTs()
	if err != nil {
		t.Fatal(err)
	}
	dst, src := testFrameWorkFilmGrainLumaRowBuffers(int(plan.Planes[0].Width), filmgrain.LumaBlockSize, int(plan.Planes[0].Stride), 100)
	grain := make([]int16, filmgrain.LumaGrainSamples)
	setTestFrameWorkFilmGrainLumaRowGrain(t, grain, 0xec, 0, 0, 0, 0, 20)
	setTestFrameWorkFilmGrainLumaRowGrain(t, grain, 0xd9, 1, 0, 0, 0, 100)

	result, err := ctx.ApplyFilmGrainLumaRow(dst, src, grain, luts.LUTs[0][:], 0)
	if err != nil {
		t.Fatal(err)
	}
	if !result.Active || result.Row != 0 || result.Width != 34 || result.Height != filmgrain.LumaBlockSize || result.Stride != int(plan.Planes[0].Stride) {
		t.Fatalf("result=%+v", result)
	}
	if got := dst[32]; got != 124 {
		t.Fatalf("luma row sample=%d want 124", got)
	}
}

func TestFrameWorkPostFilterContextApplyFilmGrainLumaRowSkipsInactive(t *testing.T) {
	result, err := (FrameWorkPostFilterContext{}).ApplyFilmGrainLumaRow(nil, nil, nil, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	if result != (FrameWorkFilmGrainPostFilterLumaRow{}) {
		t.Fatalf("result=%+v want zero", result)
	}
}

func TestFrameWorkPostFilterContextApplyFilmGrainLumaRowRejectsInvalidRow(t *testing.T) {
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: 34, Height: 32, BitDepth: 8, Align: 32})
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			FilmGrain: parser.FilmGrainParams{
				Apply:        true,
				BitDepth:     8,
				NumYPoints:   1,
				ScalingShift: 8,
			},
		},
		Output: output,
	}
	if _, err := ctx.ApplyFilmGrainLumaRow(nil, nil, nil, nil, 1); !errors.Is(err, frame.ErrInvalidFormat) {
		t.Fatalf("ApplyFilmGrainLumaRow err=%v want %v", err, frame.ErrInvalidFormat)
	}
}

func TestFrameWorkPostFilterContextApplyFilmGrainLumaRowRejectsShortScratch(t *testing.T) {
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: 34, Height: 32, BitDepth: 8, Align: 32})
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			FilmGrain: parser.FilmGrainParams{
				Apply:        true,
				BitDepth:     8,
				NumYPoints:   1,
				ScalingShift: 8,
			},
		},
		Output: output,
	}
	plan, err := ctx.FilmGrainPostFilterPlan()
	if err != nil {
		t.Fatal(err)
	}
	dst, src := testFrameWorkFilmGrainLumaRowBuffers(int(plan.Planes[0].Width), filmgrain.LumaBlockSize, int(plan.Planes[0].Stride), 100)
	grain := make([]int16, filmgrain.LumaGrainSamples)
	var lut [filmgrain.ScalingLUTSize]uint8
	need := (filmgrain.LumaBlockSize-1)*int(plan.Planes[0].Stride) + int(plan.Planes[0].Width)
	if _, err := ctx.ApplyFilmGrainLumaRow(dst[:need-1], src, grain, lut[:], 0); !errors.Is(err, frame.ErrShortBuffer) {
		t.Fatalf("ApplyFilmGrainLumaRow err=%v want %v", err, frame.ErrShortBuffer)
	}
}

func TestFrameWorkPostFilterContextApplyFilmGrainLumaRowAllocs(t *testing.T) {
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: 34, Height: 32, BitDepth: 8, Align: 32})
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			FilmGrain: parser.FilmGrainParams{
				ParamsPresent: true,
				Apply:         true,
				Seed:          0,
				BitDepth:      8,
				NumYPoints:    1,
				YPoints:       [parser.MaxFilmGrainYPoints][2]uint8{{0, 64}},
				ScalingShift:  8,
				Overlap:       true,
			},
		},
		Output: output,
	}
	plan, err := ctx.FilmGrainPostFilterPlan()
	if err != nil {
		t.Fatal(err)
	}
	luts, err := ctx.FilmGrainPostFilterScalingLUTs()
	if err != nil {
		t.Fatal(err)
	}
	dst, src := testFrameWorkFilmGrainLumaRowBuffers(int(plan.Planes[0].Width), filmgrain.LumaBlockSize, int(plan.Planes[0].Stride), 100)
	grain := make([]int16, filmgrain.LumaGrainSamples)
	setTestFrameWorkFilmGrainLumaRowGrain(t, grain, 0xec, 0, 0, 0, 0, 20)
	setTestFrameWorkFilmGrainLumaRowGrain(t, grain, 0xd9, 1, 0, 0, 0, 100)

	allocs := testing.AllocsPerRun(1000, func() {
		result, err := ctx.ApplyFilmGrainLumaRow(dst, src, grain, luts.LUTs[0][:], 0)
		if err != nil {
			t.Fatal(err)
		}
		if !result.Active {
			t.Fatalf("result=%+v", result)
		}
	})
	if allocs != 0 {
		t.Fatalf("ApplyFilmGrainLumaRow allocated: %f", allocs)
	}
}

func testFrameWorkFilmGrainLumaRowBuffers(width int, height int, stride int, value uint16) ([]uint16, []uint16) {
	dst := make([]uint16, stride*height)
	src := make([]uint16, stride*height)
	for y := range height {
		for x := range width {
			src[y*stride+x] = value
		}
	}
	return dst, src
}

func setTestFrameWorkFilmGrainLumaRowGrain(t *testing.T, grain []int16, offset uint8, blockCol int, blockRow int, x int, y int, value int16) {
	t.Helper()
	col := 3 + filmgrain.LumaOverlapSamples*(3+int(offset>>4)) + x + filmgrain.LumaBlockSize*blockCol
	row := 3 + filmgrain.LumaOverlapSamples*(3+int(offset&0x0f)) + y + filmgrain.LumaBlockSize*blockRow
	if col < 0 || col >= filmgrain.LumaGrainWidth || row < 0 || row >= filmgrain.LumaGrainHeight {
		t.Fatalf("grain coordinate out of range: col=%d row=%d", col, row)
	}
	grain[row*filmgrain.LumaGrainWidth+col] = value
}
