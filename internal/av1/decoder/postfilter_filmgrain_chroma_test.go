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

func TestFrameWorkPostFilterContextApplyFilmGrainChromaRow(t *testing.T) {
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: 34, Height: 32, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32})
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			FilmGrain: parser.FilmGrainParams{
				ParamsPresent: true,
				Apply:         true,
				Seed:          0,
				BitDepth:      8,
				NumCbPoints:   1,
				CbPoints:      [parser.MaxFilmGrainUVPoints][2]uint8{{0, 64}},
				NumCrPoints:   1,
				CrPoints:      [parser.MaxFilmGrainUVPoints][2]uint8{{0, 64}},
				ScalingShift:  8,
				CbMult:        64,
				CrMult:        64,
				Overlap:       true,
			},
		},
		Output: output,
	}
	plan, err := ctx.FilmGrainPostFilterPlan()
	if err != nil {
		t.Fatal(err)
	}
	dst, src := testFrameWorkFilmGrainChromaRowBuffers(int(plan.Planes[filmgrain.ChromaPlaneCb].Width), filmgrain.LumaBlockSize>>1, int(plan.Planes[filmgrain.ChromaPlaneCb].Stride), 100)
	lumaStride := frameWorkFilmGrainSampleStride(output.Layout.YStride, output.Layout.BytesPerSample)
	luma := testFrameWorkFilmGrainSampleBuffer(output.Y.Width, output.Y.Height, lumaStride, 96)
	grain := make([]int16, filmgrain.ChromaGrainSamples)
	setTestFrameWorkFilmGrainChromaRowGrain(t, grain, 0xd9, true, true, 0, 0, 0, 0, 20)
	lut := testFrameWorkFilmGrainScaling(64)

	result, err := ctx.ApplyFilmGrainChromaRow(dst, src, luma, grain, lut[:], filmgrain.ChromaPlaneCb, 0)
	if err != nil {
		t.Fatal(err)
	}
	if !result.Active ||
		result.Plane != filmgrain.ChromaPlaneCb ||
		result.Row != 0 ||
		result.Width != plan.Planes[filmgrain.ChromaPlaneCb].Width ||
		result.Height != filmgrain.LumaBlockSize>>1 ||
		result.Stride != plan.Planes[filmgrain.ChromaPlaneCb].Stride ||
		result.LumaStride != uint32(lumaStride) {
		t.Fatalf("result=%+v", result)
	}
	if got := dst[0]; got != 105 {
		t.Fatalf("chroma row sample=%d want 105", got)
	}
}

func TestFrameWorkPostFilterContextApplyFilmGrainChromaRowSkipsInactive(t *testing.T) {
	result, err := (FrameWorkPostFilterContext{}).ApplyFilmGrainChromaRow(nil, nil, nil, nil, nil, filmgrain.ChromaPlaneCb, 0)
	if err != nil {
		t.Fatal(err)
	}
	if result != (FrameWorkFilmGrainPostFilterChromaRow{}) {
		t.Fatalf("result=%+v want zero", result)
	}
}

func TestFrameWorkPostFilterContextApplyFilmGrainChromaRowRejectsInvalidPlane(t *testing.T) {
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: 34, Height: 32, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32})
	ctx := FrameWorkPostFilterContext{
		Event: Event{FilmGrain: parser.FilmGrainParams{
			Apply:        true,
			BitDepth:     8,
			NumCbPoints:  1,
			NumCrPoints:  1,
			ScalingShift: 8,
		}},
		Output: output,
	}
	if _, err := ctx.ApplyFilmGrainChromaRow(nil, nil, nil, nil, nil, 0, 0); !errors.Is(err, frame.ErrInvalidFormat) {
		t.Fatalf("ApplyFilmGrainChromaRow err=%v want %v", err, frame.ErrInvalidFormat)
	}
}

func TestFrameWorkPostFilterContextApplyFilmGrainChromaRowRejectsInvalidRow(t *testing.T) {
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: 34, Height: 32, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32})
	ctx := FrameWorkPostFilterContext{
		Event: Event{FilmGrain: parser.FilmGrainParams{
			Apply:        true,
			BitDepth:     8,
			NumCbPoints:  1,
			NumCrPoints:  1,
			ScalingShift: 8,
		}},
		Output: output,
	}
	if _, err := ctx.ApplyFilmGrainChromaRow(nil, nil, nil, nil, nil, filmgrain.ChromaPlaneCb, 1); !errors.Is(err, frame.ErrInvalidFormat) {
		t.Fatalf("ApplyFilmGrainChromaRow err=%v want %v", err, frame.ErrInvalidFormat)
	}
}

func TestFrameWorkPostFilterContextApplyFilmGrainChromaRowRejectsShortScratch(t *testing.T) {
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: 34, Height: 32, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32})
	ctx := FrameWorkPostFilterContext{
		Event: Event{FilmGrain: parser.FilmGrainParams{
			Apply:        true,
			BitDepth:     8,
			NumCbPoints:  1,
			NumCrPoints:  1,
			ScalingShift: 8,
			CbMult:       64,
			CrMult:       64,
		}},
		Output: output,
	}
	plan, err := ctx.FilmGrainPostFilterPlan()
	if err != nil {
		t.Fatal(err)
	}
	height := filmgrain.LumaBlockSize >> 1
	dst, src := testFrameWorkFilmGrainChromaRowBuffers(int(plan.Planes[filmgrain.ChromaPlaneCb].Width), height, int(plan.Planes[filmgrain.ChromaPlaneCb].Stride), 100)
	lumaStride := frameWorkFilmGrainSampleStride(output.Layout.YStride, output.Layout.BytesPerSample)
	luma := testFrameWorkFilmGrainSampleBuffer(output.Y.Width, output.Y.Height, lumaStride, 96)
	grain := make([]int16, filmgrain.ChromaGrainSamples)
	lut := testFrameWorkFilmGrainScaling(64)
	need := (height-1)*int(plan.Planes[filmgrain.ChromaPlaneCb].Stride) + int(plan.Planes[filmgrain.ChromaPlaneCb].Width)
	if _, err := ctx.ApplyFilmGrainChromaRow(dst[:need-1], src, luma, grain, lut[:], filmgrain.ChromaPlaneCb, 0); !errors.Is(err, frame.ErrShortBuffer) {
		t.Fatalf("ApplyFilmGrainChromaRow err=%v want %v", err, frame.ErrShortBuffer)
	}
}

func TestFrameWorkPostFilterContextApplyFilmGrainChromaRowAllocs(t *testing.T) {
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: 34, Height: 32, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32})
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			FilmGrain: parser.FilmGrainParams{
				ParamsPresent: true,
				Apply:         true,
				Seed:          0,
				BitDepth:      8,
				NumCbPoints:   1,
				CbPoints:      [parser.MaxFilmGrainUVPoints][2]uint8{{0, 64}},
				NumCrPoints:   1,
				CrPoints:      [parser.MaxFilmGrainUVPoints][2]uint8{{0, 64}},
				ScalingShift:  8,
				CbMult:        64,
				CrMult:        64,
				Overlap:       true,
			},
		},
		Output: output,
	}
	plan, err := ctx.FilmGrainPostFilterPlan()
	if err != nil {
		t.Fatal(err)
	}
	dst, src := testFrameWorkFilmGrainChromaRowBuffers(int(plan.Planes[filmgrain.ChromaPlaneCb].Width), filmgrain.LumaBlockSize>>1, int(plan.Planes[filmgrain.ChromaPlaneCb].Stride), 100)
	lumaStride := frameWorkFilmGrainSampleStride(output.Layout.YStride, output.Layout.BytesPerSample)
	luma := testFrameWorkFilmGrainSampleBuffer(output.Y.Width, output.Y.Height, lumaStride, 96)
	grain := make([]int16, filmgrain.ChromaGrainSamples)
	setTestFrameWorkFilmGrainChromaRowGrain(t, grain, 0xd9, true, true, 0, 0, 0, 0, 20)
	lut := testFrameWorkFilmGrainScaling(64)

	allocs := testing.AllocsPerRun(1000, func() {
		result, err := ctx.ApplyFilmGrainChromaRow(dst, src, luma, grain, lut[:], filmgrain.ChromaPlaneCb, 0)
		if err != nil {
			t.Fatal(err)
		}
		if !result.Active {
			t.Fatalf("result=%+v", result)
		}
	})
	if allocs != 0 {
		t.Fatalf("ApplyFilmGrainChromaRow allocated: %f", allocs)
	}
}

func testFrameWorkFilmGrainLumaTemplate(value int16) []int16 {
	grain := make([]int16, filmgrain.LumaGrainSamples)
	for i := range grain {
		grain[i] = value
	}
	return grain
}

func testFrameWorkFilmGrainChromaRowBuffers(width int, height int, stride int, value uint16) ([]uint16, []uint16) {
	dst := make([]uint16, stride*height)
	src := make([]uint16, stride*height)
	for y := range height {
		for x := range width {
			src[y*stride+x] = value
		}
	}
	return dst, src
}

func testFrameWorkFilmGrainSampleBuffer(width int, height int, stride int, value uint16) []uint16 {
	samples := make([]uint16, stride*height)
	for y := range height {
		for x := range width {
			samples[y*stride+x] = value
		}
	}
	return samples
}

func testFrameWorkFilmGrainScaling(value uint8) [filmgrain.ScalingLUTSize]uint8 {
	var lut [filmgrain.ScalingLUTSize]uint8
	for i := range lut {
		lut[i] = value
	}
	return lut
}

func setTestFrameWorkFilmGrainChromaRowGrain(t *testing.T, grain []int16, offset uint8, subsamplingX bool, subsamplingY bool, blockCol int, blockRow int, x int, y int, value int16) {
	t.Helper()
	shiftX := frameWorkFilmGrainSubsamplingShift(subsamplingX)
	shiftY := frameWorkFilmGrainSubsamplingShift(subsamplingY)
	col := 3 + (filmgrain.LumaOverlapSamples>>shiftX)*(3+int(offset>>4)) + x + (filmgrain.LumaBlockSize>>shiftX)*blockCol
	row := 3 + (filmgrain.LumaOverlapSamples>>shiftY)*(3+int(offset&0x0f)) + y + (filmgrain.LumaBlockSize>>shiftY)*blockRow
	width := filmgrain.ChromaGrainWidth
	if subsamplingX {
		width = filmgrain.ChromaSubsampledGrainWidth
	}
	height := filmgrain.ChromaGrainHeight
	if subsamplingY {
		height = filmgrain.ChromaSubsampledGrainHeight
	}
	if col < 0 || col >= width || row < 0 || row >= height {
		t.Fatalf("grain coordinate out of range: col=%d row=%d", col, row)
	}
	grain[row*filmgrain.ChromaGrainWidth+col] = value
}
