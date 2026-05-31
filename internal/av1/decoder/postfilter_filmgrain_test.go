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
		size.LumaSamples != plan.Planes[0].Stride*plan.Planes[0].Height ||
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
	if size.ChromaGrain != [2]int{filmgrain.ChromaGrainSamples, filmgrain.ChromaGrainSamples} {
		t.Fatalf("chroma grain scratch=%+v", size.ChromaGrain)
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

func TestFrameWorkPostFilterContextApplySupportedPostFiltersRejectsChromaFilmGrainShortScratchBeforeMutation(t *testing.T) {
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
				NumCbPoints:   1,
				CbPoints:      [parser.MaxFilmGrainUVPoints][2]uint8{{0, 16}},
				NumCrPoints:   1,
				CrPoints:      [parser.MaxFilmGrainUVPoints][2]uint8{{0, 16}},
				ScalingShift:  8,
			},
		},
		Output: output,
	}
	size, err := ctx.SupportedPostFilterScratchLen(FrameWorkPostFilterRequest{})
	if err != nil {
		t.Fatal(err)
	}
	if size.FilmGrain.LumaSamples == 0 ||
		size.FilmGrain.ChromaGrain != [2]int{filmgrain.ChromaGrainSamples, filmgrain.ChromaGrainSamples} ||
		size.FilmGrain.ChromaSamples[0] == 0 ||
		size.FilmGrain.ChromaSamples[1] == 0 {
		t.Fatalf("film grain scratch=%+v", size.FilmGrain)
	}
	next, result, err := ctx.ApplySupportedPostFilters(FrameWorkPostFilterRequest{})
	if !errors.Is(err, frame.ErrShortBuffer) {
		t.Fatalf("ApplySupportedPostFilters err=%v want %v", err, frame.ErrShortBuffer)
	}
	if next.RemainingPostFilters() != ctx.RemainingPostFilters() || result != (FrameWorkPostFilterResult{}) {
		t.Fatalf("next remaining=%b result=%+v", next.RemainingPostFilters(), result)
	}
	if output.Y.Pix[0] != 0x44 {
		t.Fatalf("output sample=%d want 0x44", output.Y.Pix[0])
	}
}

func TestFrameWorkPostFilterContextApplySupportedPostFiltersAppliesLumaFilmGrain(t *testing.T) {
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: 34, Height: 32, BitDepth: 8, Align: 32})
	for y := 0; y < output.Y.Height; y++ {
		for x := 0; x < output.Y.Width; x++ {
			output.Y.Pix[y*output.Y.Stride+x] = 100
		}
	}
	before := testCopyFrameWorkCDEFPlane(output.Y)
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			FilmGrain: parser.FilmGrainParams{
				ParamsPresent: true,
				Apply:         true,
				Seed:          0x1234,
				BitDepth:      8,
				NumYPoints:    1,
				YPoints:       [parser.MaxFilmGrainYPoints][2]uint8{{0, 64}},
				ScalingShift:  8,
				Overlap:       true,
			},
		},
		Output: output,
	}
	size, err := ctx.SupportedPostFilterScratchLen(FrameWorkPostFilterRequest{})
	if err != nil {
		t.Fatal(err)
	}
	if size.FilmGrain.LumaGrain != filmgrain.LumaGrainSamples || size.FilmGrain.LumaSamples == 0 {
		t.Fatalf("film grain scratch=%+v", size.FilmGrain)
	}
	req := FrameWorkPostFilterRequest{
		FilmGrain: FrameWorkFilmGrainPostFilterRequest{
			LumaGrain:   make([]int16, size.FilmGrain.LumaGrain),
			LumaSamples: make([]uint16, size.FilmGrain.LumaSamples),
		},
	}
	next, result, err := ctx.ApplySupportedPostFilters(req)
	if err != nil {
		t.Fatal(err)
	}
	if result.Completed != FrameWorkPostFilterFilmGrain ||
		result.FilmGrain.NoOp ||
		result.FilmGrain.LumaRows != 1 ||
		next.RemainingPostFilters() != 0 {
		t.Fatalf("next remaining=%b result=%+v", next.RemainingPostFilters(), result)
	}
	if !testFrameWorkCDEFPlaneChanged(output.Y, before) {
		t.Fatal("film grain did not change luma output")
	}
}

func TestFrameWorkPostFilterContextApplySupportedPostFiltersAppliesChromaFilmGrain(t *testing.T) {
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: 64, Height: 32, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32})
	for y := 0; y < output.Y.Height; y++ {
		for x := 0; x < output.Y.Width; x++ {
			output.Y.Pix[y*output.Y.Stride+x] = 96
		}
	}
	for y := 0; y < output.U.Height; y++ {
		for x := 0; x < output.U.Width; x++ {
			output.U.Pix[y*output.U.Stride+x] = 100
			output.V.Pix[y*output.V.Stride+x] = 110
		}
	}
	beforeY := testCopyFrameWorkCDEFPlane(output.Y)
	beforeU := testCopyFrameWorkCDEFPlane(output.U)
	beforeV := testCopyFrameWorkCDEFPlane(output.V)
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			FilmGrain: parser.FilmGrainParams{
				ParamsPresent: true,
				Apply:         true,
				Seed:          0x1234,
				BitDepth:      8,
				NumYPoints:    1,
				YPoints:       [parser.MaxFilmGrainYPoints][2]uint8{{0, 64}},
				NumCbPoints:   1,
				CbPoints:      [parser.MaxFilmGrainUVPoints][2]uint8{{0, 64}},
				NumCrPoints:   1,
				CrPoints:      [parser.MaxFilmGrainUVPoints][2]uint8{{0, 64}},
				ScalingShift:  8,
				ARCoeffShift:  6,
				CbMult:        64,
				CrMult:        64,
				Overlap:       true,
			},
		},
		Output: output,
	}
	size, err := ctx.SupportedPostFilterScratchLen(FrameWorkPostFilterRequest{})
	if err != nil {
		t.Fatal(err)
	}
	req := testFrameWorkFilmGrainPostFilterRequest(size.FilmGrain)
	next, result, err := ctx.ApplySupportedPostFilters(FrameWorkPostFilterRequest{FilmGrain: req})
	if err != nil {
		t.Fatal(err)
	}
	if result.Completed != FrameWorkPostFilterFilmGrain ||
		result.FilmGrain.LumaRows != 1 ||
		result.FilmGrain.ChromaRows != [2]int{1, 1} ||
		next.RemainingPostFilters() != 0 {
		t.Fatalf("next remaining=%b result=%+v", next.RemainingPostFilters(), result)
	}
	if !testFrameWorkCDEFPlaneChanged(output.Y, beforeY) {
		t.Fatal("film grain did not change luma output")
	}
	if !testFrameWorkCDEFPlaneChanged(output.U, beforeU) {
		t.Fatal("film grain did not change cb output")
	}
	if !testFrameWorkCDEFPlaneChanged(output.V, beforeV) {
		t.Fatal("film grain did not change cr output")
	}
}

func TestFrameWorkPostFilterContextApplySupportedPostFiltersAppliesHighBitDepthChromaFilmGrain(t *testing.T) {
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: 64, Height: 32, BitDepth: 10, SubsamplingX: true, SubsamplingY: true, Align: 64})
	for y := 0; y < output.Y.Height; y++ {
		for x := 0; x < output.Y.Width; x++ {
			setTestFrameWorkFilmGrainPlaneSample(output.Y, output.Layout.BytesPerSample, x, y, 384)
		}
	}
	for y := 0; y < output.U.Height; y++ {
		for x := 0; x < output.U.Width; x++ {
			setTestFrameWorkFilmGrainPlaneSample(output.U, output.Layout.BytesPerSample, x, y, 400)
			setTestFrameWorkFilmGrainPlaneSample(output.V, output.Layout.BytesPerSample, x, y, 420)
		}
	}
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			FilmGrain: parser.FilmGrainParams{
				ParamsPresent: true,
				Apply:         true,
				Seed:          0x1234,
				BitDepth:      10,
				NumYPoints:    1,
				YPoints:       [parser.MaxFilmGrainYPoints][2]uint8{{0, 64}},
				NumCbPoints:   1,
				CbPoints:      [parser.MaxFilmGrainUVPoints][2]uint8{{0, 64}},
				NumCrPoints:   1,
				CrPoints:      [parser.MaxFilmGrainUVPoints][2]uint8{{0, 64}},
				ScalingShift:  8,
				ARCoeffShift:  6,
				CbMult:        64,
				CrMult:        64,
				Overlap:       true,
			},
		},
		Output: output,
	}
	size, err := ctx.SupportedPostFilterScratchLen(FrameWorkPostFilterRequest{})
	if err != nil {
		t.Fatal(err)
	}
	req := testFrameWorkFilmGrainPostFilterRequest(size.FilmGrain)
	next, result, err := ctx.ApplySupportedPostFilters(FrameWorkPostFilterRequest{FilmGrain: req})
	if err != nil {
		t.Fatal(err)
	}
	if result.Completed != FrameWorkPostFilterFilmGrain ||
		result.FilmGrain.LumaRows != 1 ||
		result.FilmGrain.ChromaRows != [2]int{1, 1} ||
		next.RemainingPostFilters() != 0 {
		t.Fatalf("next remaining=%b result=%+v", next.RemainingPostFilters(), result)
	}
	if !testFrameWorkFilmGrainPlaneChangedFromValue(output.U, output.Layout.BytesPerSample, 400) {
		t.Fatal("film grain did not change high-bit-depth cb output")
	}
	if !testFrameWorkFilmGrainPlaneChangedFromValue(output.V, output.Layout.BytesPerSample, 420) {
		t.Fatal("film grain did not change high-bit-depth cr output")
	}
	for _, sample := range []uint16{
		getTestFrameWorkFilmGrainPlaneSample(output.U, output.Layout.BytesPerSample, 0, 0),
		getTestFrameWorkFilmGrainPlaneSample(output.V, output.Layout.BytesPerSample, 0, 0),
	} {
		if sample > 1023 {
			t.Fatalf("high-bit-depth chroma sample=%d outside 10-bit range", sample)
		}
	}
}

func TestFrameWorkPostFilterContextApplySupportedPostFiltersChromaFilmGrainAllocs(t *testing.T) {
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: 64, Height: 32, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32})
	for y := 0; y < output.Y.Height; y++ {
		for x := 0; x < output.Y.Width; x++ {
			output.Y.Pix[y*output.Y.Stride+x] = 96
		}
	}
	for y := 0; y < output.U.Height; y++ {
		for x := 0; x < output.U.Width; x++ {
			output.U.Pix[y*output.U.Stride+x] = 100
			output.V.Pix[y*output.V.Stride+x] = 110
		}
	}
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			FilmGrain: parser.FilmGrainParams{
				ParamsPresent: true,
				Apply:         true,
				Seed:          0x1234,
				BitDepth:      8,
				NumYPoints:    1,
				YPoints:       [parser.MaxFilmGrainYPoints][2]uint8{{0, 64}},
				NumCbPoints:   1,
				CbPoints:      [parser.MaxFilmGrainUVPoints][2]uint8{{0, 64}},
				NumCrPoints:   1,
				CrPoints:      [parser.MaxFilmGrainUVPoints][2]uint8{{0, 64}},
				ScalingShift:  8,
				ARCoeffShift:  6,
				CbMult:        64,
				CrMult:        64,
				Overlap:       true,
			},
		},
		Output: output,
	}
	size, err := ctx.SupportedPostFilterScratchLen(FrameWorkPostFilterRequest{})
	if err != nil {
		t.Fatal(err)
	}
	req := testFrameWorkFilmGrainPostFilterRequest(size.FilmGrain)
	allocs := testing.AllocsPerRun(1000, func() {
		next, result, err := ctx.ApplySupportedPostFilters(FrameWorkPostFilterRequest{FilmGrain: req})
		if err != nil {
			t.Fatal(err)
		}
		if result.Completed != FrameWorkPostFilterFilmGrain ||
			result.FilmGrain.LumaRows != 1 ||
			result.FilmGrain.ChromaRows != [2]int{1, 1} ||
			next.RemainingPostFilters() != 0 {
			t.Fatalf("next remaining=%b result=%+v", next.RemainingPostFilters(), result)
		}
	})
	if allocs != 0 {
		t.Fatalf("chroma film grain postfilter pipeline allocated: %f", allocs)
	}
}

func TestFrameWorkPostFilterContextApplySupportedPostFiltersAppliesPartialFinalChromaRow(t *testing.T) {
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: 64, Height: 33, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32})
	for y := 0; y < output.Y.Height; y++ {
		for x := 0; x < output.Y.Width; x++ {
			output.Y.Pix[y*output.Y.Stride+x] = 96
		}
	}
	for y := 0; y < output.U.Height; y++ {
		for x := 0; x < output.U.Width; x++ {
			output.U.Pix[y*output.U.Stride+x] = 100
			output.V.Pix[y*output.V.Stride+x] = 110
		}
	}
	beforeU := testCopyFrameWorkCDEFPlane(output.U)
	beforeV := testCopyFrameWorkCDEFPlane(output.V)
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			FilmGrain: parser.FilmGrainParams{
				ParamsPresent: true,
				Apply:         true,
				Seed:          0x1234,
				BitDepth:      8,
				NumYPoints:    1,
				YPoints:       [parser.MaxFilmGrainYPoints][2]uint8{{0, 64}},
				NumCbPoints:   1,
				CbPoints:      [parser.MaxFilmGrainUVPoints][2]uint8{{0, 64}},
				NumCrPoints:   1,
				CrPoints:      [parser.MaxFilmGrainUVPoints][2]uint8{{0, 64}},
				ScalingShift:  8,
				ARCoeffShift:  6,
				CbMult:        64,
				CrMult:        64,
				Overlap:       true,
			},
		},
		Output: output,
	}
	size, err := ctx.SupportedPostFilterScratchLen(FrameWorkPostFilterRequest{})
	if err != nil {
		t.Fatal(err)
	}
	req := testFrameWorkFilmGrainPostFilterRequest(size.FilmGrain)
	next, result, err := ctx.ApplySupportedPostFilters(FrameWorkPostFilterRequest{FilmGrain: req})
	if err != nil {
		t.Fatal(err)
	}
	if result.Completed != FrameWorkPostFilterFilmGrain ||
		result.FilmGrain.LumaRows != 2 ||
		result.FilmGrain.ChromaRows != [2]int{2, 2} ||
		next.RemainingPostFilters() != 0 {
		t.Fatalf("next remaining=%b result=%+v", next.RemainingPostFilters(), result)
	}
	if !testFrameWorkCDEFPlaneChanged(output.U, beforeU) {
		t.Fatal("film grain did not change partial-row cb output")
	}
	if !testFrameWorkCDEFPlaneChanged(output.V, beforeV) {
		t.Fatal("film grain did not change partial-row cr output")
	}
}

func TestFrameWorkPostFilterContextApplySupportedPostFiltersAppliesHighBitDepthLumaFilmGrain(t *testing.T) {
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: 34, Height: 32, BitDepth: 10, Align: 64})
	for y := 0; y < output.Y.Height; y++ {
		for x := 0; x < output.Y.Width; x++ {
			setTestFrameWorkFilmGrainPlaneSample(output.Y, output.Layout.BytesPerSample, x, y, 400)
		}
	}
	before := testCopyFrameWorkCDEFPlane(output.Y)
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			FilmGrain: parser.FilmGrainParams{
				ParamsPresent: true,
				Apply:         true,
				Seed:          0x1234,
				BitDepth:      10,
				NumYPoints:    1,
				YPoints:       [parser.MaxFilmGrainYPoints][2]uint8{{0, 64}},
				ScalingShift:  8,
				Overlap:       true,
			},
		},
		Output: output,
	}
	size, err := ctx.SupportedPostFilterScratchLen(FrameWorkPostFilterRequest{})
	if err != nil {
		t.Fatal(err)
	}
	req := FrameWorkPostFilterRequest{
		FilmGrain: FrameWorkFilmGrainPostFilterRequest{
			LumaGrain:   make([]int16, size.FilmGrain.LumaGrain),
			LumaSamples: make([]uint16, size.FilmGrain.LumaSamples),
		},
	}
	next, result, err := ctx.ApplySupportedPostFilters(req)
	if err != nil {
		t.Fatal(err)
	}
	if result.Completed != FrameWorkPostFilterFilmGrain ||
		result.FilmGrain.LumaRows != 1 ||
		next.RemainingPostFilters() != 0 {
		t.Fatalf("next remaining=%b result=%+v", next.RemainingPostFilters(), result)
	}
	if !testFrameWorkCDEFPlaneChanged(output.Y, before) {
		t.Fatal("film grain did not change high-bit-depth luma output")
	}
	if got := getTestFrameWorkFilmGrainPlaneSample(output.Y, output.Layout.BytesPerSample, 0, 0); got > 1023 {
		t.Fatalf("high-bit-depth sample=%d outside 10-bit range", got)
	}
}

func TestFrameWorkPostFilterContextApplySupportedPostFiltersAppliesPartialFinalLumaRow(t *testing.T) {
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: 34, Height: 33, BitDepth: 8, Align: 32})
	for y := 0; y < output.Y.Height; y++ {
		for x := 0; x < output.Y.Width; x++ {
			output.Y.Pix[y*output.Y.Stride+x] = 100
		}
	}
	before := testCopyFrameWorkCDEFPlane(output.Y)
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			FilmGrain: parser.FilmGrainParams{
				ParamsPresent: true,
				Apply:         true,
				Seed:          0x1234,
				BitDepth:      8,
				NumYPoints:    1,
				YPoints:       [parser.MaxFilmGrainYPoints][2]uint8{{0, 64}},
				ScalingShift:  8,
				Overlap:       true,
			},
		},
		Output: output,
	}
	size, err := ctx.SupportedPostFilterScratchLen(FrameWorkPostFilterRequest{})
	if err != nil {
		t.Fatal(err)
	}
	req := FrameWorkPostFilterRequest{
		FilmGrain: FrameWorkFilmGrainPostFilterRequest{
			LumaGrain:   make([]int16, size.FilmGrain.LumaGrain),
			LumaSamples: make([]uint16, size.FilmGrain.LumaSamples),
		},
	}
	next, result, err := ctx.ApplySupportedPostFilters(req)
	if err != nil {
		t.Fatal(err)
	}
	if result.Completed != FrameWorkPostFilterFilmGrain ||
		result.FilmGrain.LumaRows != 2 ||
		next.RemainingPostFilters() != 0 {
		t.Fatalf("next remaining=%b result=%+v", next.RemainingPostFilters(), result)
	}
	if !testFrameWorkCDEFPlaneChanged(output.Y, before) {
		t.Fatal("film grain did not change partial-row luma output")
	}
}

func TestFrameWorkPostFilterContextApplySupportedPostFiltersRejectsLumaFilmGrainShortScratchBeforeMutation(t *testing.T) {
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
				ScalingShift:  8,
			},
		},
		Output: output,
	}
	next, result, err := ctx.ApplySupportedPostFilters(FrameWorkPostFilterRequest{})
	if !errors.Is(err, frame.ErrShortBuffer) {
		t.Fatalf("ApplySupportedPostFilters err=%v want %v", err, frame.ErrShortBuffer)
	}
	if next.RemainingPostFilters() != ctx.RemainingPostFilters() || result != (FrameWorkPostFilterResult{}) {
		t.Fatalf("next remaining=%b result=%+v", next.RemainingPostFilters(), result)
	}
	if output.Y.Pix[0] != 0x44 {
		t.Fatalf("output sample=%d want 0x44", output.Y.Pix[0])
	}
}

func TestFrameWorkPostFilterContextApplySupportedPostFiltersLumaFilmGrainAllocs(t *testing.T) {
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: 34, Height: 32, BitDepth: 8, Align: 32})
	for y := 0; y < output.Y.Height; y++ {
		for x := 0; x < output.Y.Width; x++ {
			output.Y.Pix[y*output.Y.Stride+x] = 100
		}
	}
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			FilmGrain: parser.FilmGrainParams{
				ParamsPresent: true,
				Apply:         true,
				Seed:          0x1234,
				BitDepth:      8,
				NumYPoints:    1,
				YPoints:       [parser.MaxFilmGrainYPoints][2]uint8{{0, 64}},
				ScalingShift:  8,
				Overlap:       true,
			},
		},
		Output: output,
	}
	size, err := ctx.SupportedPostFilterScratchLen(FrameWorkPostFilterRequest{})
	if err != nil {
		t.Fatal(err)
	}
	req := FrameWorkPostFilterRequest{
		FilmGrain: FrameWorkFilmGrainPostFilterRequest{
			LumaGrain:   make([]int16, size.FilmGrain.LumaGrain),
			LumaSamples: make([]uint16, size.FilmGrain.LumaSamples),
		},
	}
	allocs := testing.AllocsPerRun(1000, func() {
		next, result, err := ctx.ApplySupportedPostFilters(req)
		if err != nil {
			t.Fatal(err)
		}
		if result.Completed != FrameWorkPostFilterFilmGrain || result.FilmGrain.LumaRows != 1 || next.RemainingPostFilters() != 0 {
			t.Fatalf("next remaining=%b result=%+v", next.RemainingPostFilters(), result)
		}
	})
	if allocs != 0 {
		t.Fatalf("luma film grain postfilter pipeline allocated: %f", allocs)
	}
}

func TestFrameWorkFilmGrainPostFilterScratchSizeBindRequest(t *testing.T) {
	size := FrameWorkFilmGrainPostFilterScratchSize{
		ScalingPoints: [3]int{1, 2, 3},
		ARCoeffs:      [3]int{4, 5, 6},
		LumaGrain:     8,
		ChromaGrain:   [2]int{4, 2},
		LumaSamples:   16,
		ChromaSamples: [2]int{9, 7},
		LumaLine:      3,
		LumaColumn:    2,
	}
	outputFrame, lumaGrain, chromaGrain, lumaSamples, chromaSamples := testFrameWorkFilmGrainScratchStorage(size)
	req, err := size.BindRequest(outputFrame, nil, lumaGrain, chromaGrain, lumaSamples, chromaSamples)
	if err != nil {
		t.Fatal(err)
	}
	if len(req.LumaGrain) != size.LumaGrain || len(req.LumaSamples) != size.LumaSamples ||
		len(req.OutputFrame) != size.OutputFrame ||
		len(req.ChromaGrain[0]) != size.ChromaGrain[0] || len(req.ChromaGrain[1]) != size.ChromaGrain[1] ||
		len(req.ChromaSamples[0]) != size.ChromaSamples[0] || len(req.ChromaSamples[1]) != size.ChromaSamples[1] {
		t.Fatalf("request lumaGrain=%d lumaSamples=%d chromaGrain=%v chromaSamples=%v",
			len(req.LumaGrain), len(req.LumaSamples),
			[2]int{len(req.ChromaGrain[0]), len(req.ChromaGrain[1])},
			[2]int{len(req.ChromaSamples[0]), len(req.ChromaSamples[1])})
	}
}

func TestFrameWorkFilmGrainPostFilterScratchSizeBindRequestRejectsShortBuffers(t *testing.T) {
	size := FrameWorkFilmGrainPostFilterScratchSize{
		ScalingPoints: [3]int{1},
		ARCoeffs:      [3]int{4},
		LumaGrain:     8,
		ChromaGrain:   [2]int{4, 2},
		LumaSamples:   16,
		ChromaSamples: [2]int{9, 7},
		LumaLine:      3,
		LumaColumn:    2,
	}
	outputFrame, lumaGrain, chromaGrain, lumaSamples, chromaSamples := testFrameWorkFilmGrainScratchStorage(size)
	tests := []struct {
		name          string
		size          FrameWorkFilmGrainPostFilterScratchSize
		outputFrame   []byte
		lumaGrain     []int16
		chromaGrain   [2][]int16
		lumaSamples   []uint16
		chromaSamples [2][]uint16
	}{
		{name: "output-frame", size: FrameWorkFilmGrainPostFilterScratchSize{OutputFrame: 1}, outputFrame: nil, lumaGrain: lumaGrain, chromaGrain: chromaGrain, lumaSamples: lumaSamples, chromaSamples: chromaSamples},
		{name: "luma-grain", size: size, outputFrame: outputFrame, lumaGrain: lumaGrain[:7], chromaGrain: chromaGrain, lumaSamples: lumaSamples, chromaSamples: chromaSamples},
		{name: "luma-samples", size: size, outputFrame: outputFrame, lumaGrain: lumaGrain, chromaGrain: chromaGrain, lumaSamples: lumaSamples[:15], chromaSamples: chromaSamples},
		{name: "chroma-grain", size: size, outputFrame: outputFrame, lumaGrain: lumaGrain, chromaGrain: [2][]int16{chromaGrain[0][:3], chromaGrain[1]}, lumaSamples: lumaSamples, chromaSamples: chromaSamples},
		{name: "chroma-samples", size: size, outputFrame: outputFrame, lumaGrain: lumaGrain, chromaGrain: chromaGrain, lumaSamples: lumaSamples, chromaSamples: [2][]uint16{chromaSamples[0], chromaSamples[1][:6]}},
		{name: "negative-requested", size: FrameWorkFilmGrainPostFilterScratchSize{LumaGrain: -1}, outputFrame: outputFrame, lumaGrain: lumaGrain, chromaGrain: chromaGrain, lumaSamples: lumaSamples, chromaSamples: chromaSamples},
		{name: "negative-plan", size: FrameWorkFilmGrainPostFilterScratchSize{ScalingPoints: [3]int{-1}}, outputFrame: outputFrame, lumaGrain: lumaGrain, chromaGrain: chromaGrain, lumaSamples: lumaSamples, chromaSamples: chromaSamples},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := tt.size.BindRequest(tt.outputFrame, nil, tt.lumaGrain, tt.chromaGrain, tt.lumaSamples, tt.chromaSamples); !errors.Is(err, frame.ErrShortBuffer) {
				t.Fatalf("BindRequest err=%v want %v", err, frame.ErrShortBuffer)
			}
		})
	}
}

func TestFrameWorkFilmGrainPostFilterScratchSizeBindRequestAllocs(t *testing.T) {
	size := FrameWorkFilmGrainPostFilterScratchSize{
		LumaGrain:     filmgrain.LumaGrainSamples,
		ChromaGrain:   [2]int{filmgrain.ChromaGrainSamples, filmgrain.ChromaGrainSamples},
		LumaSamples:   1920 * 1080,
		ChromaSamples: [2]int{960 * 540, 960 * 540},
		LumaLine:      filmgrain.LumaOverlapSamples * 1920,
		LumaColumn:    filmgrain.LumaColumnScratchRows * filmgrain.LumaOverlapSamples,
	}
	outputFrame, lumaGrain, chromaGrain, lumaSamples, chromaSamples := testFrameWorkFilmGrainScratchStorage(size)
	allocs := testing.AllocsPerRun(1000, func() {
		if _, err := size.BindRequest(outputFrame, nil, lumaGrain, chromaGrain, lumaSamples, chromaSamples); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("BindRequest allocated: %f", allocs)
	}
}

func BenchmarkFrameWorkFilmGrainPostFilterScratchSizeBindRequest(b *testing.B) {
	size := FrameWorkFilmGrainPostFilterScratchSize{
		LumaGrain:     filmgrain.LumaGrainSamples,
		ChromaGrain:   [2]int{filmgrain.ChromaGrainSamples, filmgrain.ChromaGrainSamples},
		LumaSamples:   1920 * 1080,
		ChromaSamples: [2]int{960 * 540, 960 * 540},
		LumaLine:      filmgrain.LumaOverlapSamples * 1920,
		LumaColumn:    filmgrain.LumaColumnScratchRows * filmgrain.LumaOverlapSamples,
	}
	outputFrame, lumaGrain, chromaGrain, lumaSamples, chromaSamples := testFrameWorkFilmGrainScratchStorage(size)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := size.BindRequest(outputFrame, nil, lumaGrain, chromaGrain, lumaSamples, chromaSamples); err != nil {
			b.Fatal(err)
		}
	}
}

func testFrameWorkFilmGrainPostFilterRequest(size FrameWorkFilmGrainPostFilterScratchSize) FrameWorkFilmGrainPostFilterRequest {
	outputFrame, lumaGrain, chromaGrain, lumaSamples, chromaSamples := testFrameWorkFilmGrainScratchStorage(size)
	req, err := size.BindRequest(outputFrame, nil, lumaGrain, chromaGrain, lumaSamples, chromaSamples)
	if err != nil {
		panic(err)
	}
	return req
}

func testFrameWorkFilmGrainScratchStorage(size FrameWorkFilmGrainPostFilterScratchSize) ([]byte, []int16, [2][]int16, []uint16, [2][]uint16) {
	var chromaGrain [2][]int16
	var chromaSamples [2][]uint16
	for plane := range len(chromaGrain) {
		if size.ChromaGrain[plane] > 0 {
			chromaGrain[plane] = make([]int16, size.ChromaGrain[plane])
		}
		if size.ChromaSamples[plane] > 0 {
			chromaSamples[plane] = make([]uint16, size.ChromaSamples[plane])
		}
	}
	return make([]byte, maxInt(size.OutputFrame, 0)),
		make([]int16, maxInt(size.LumaGrain, 0)), chromaGrain,
		make([]uint16, maxInt(size.LumaSamples, 0)), chromaSamples
}

func getTestFrameWorkFilmGrainPlaneSample(plane frame.Plane, bytesPerSample int, x int, y int) uint16 {
	offset := y*plane.Stride + x*bytesPerSample
	if bytesPerSample == 1 {
		return uint16(plane.Pix[offset])
	}
	return uint16(plane.Pix[offset]) | uint16(plane.Pix[offset+1])<<8
}

func setTestFrameWorkFilmGrainPlaneSample(plane frame.Plane, bytesPerSample int, x int, y int, value uint16) {
	offset := y*plane.Stride + x*bytesPerSample
	if bytesPerSample == 1 {
		plane.Pix[offset] = byte(value)
		return
	}
	plane.Pix[offset] = byte(value)
	plane.Pix[offset+1] = byte(value >> 8)
}

func testFrameWorkFilmGrainPlaneChangedFromValue(plane frame.Plane, bytesPerSample int, value uint16) bool {
	for y := 0; y < plane.Height; y++ {
		for x := 0; x < plane.Width; x++ {
			if getTestFrameWorkFilmGrainPlaneSample(plane, bytesPerSample, x, y) != value {
				return true
			}
		}
	}
	return false
}

func TestFrameWorkPostFilterContextApplySupportedPostFiltersRunsCDEFThenNoOpFilmGrain(t *testing.T) {
	const width = 64
	const height = 64

	seq := testSequence()
	seq.EnableCDEF = true
	event := Event{
		SequenceHeader: seq,
		FrameSize: parser.FrameSize{
			CodedWidth:          width,
			UpscaledWidth:       width,
			Height:              height,
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
			Seed:          0x4321,
			BitDepth:      8,
		},
	}
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: width, Height: height, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32})
	testFillFrameWorkCDEFPlane(output.Y)
	before := testCopyFrameWorkCDEFPlane(output.Y)

	ctx := FrameWorkPostFilterContext{Event: event, Output: output}
	req := testFrameWorkCDEFPostFilterRequest(t, ctx, event)
	size, err := ctx.SupportedPostFilterScratchLen(FrameWorkPostFilterRequest{CDEF: req})
	if err != nil {
		t.Fatal(err)
	}
	if size.CDEF.Input == 0 || size.CDEF.UnitDst == 0 || size.FilmGrain != (FrameWorkFilmGrainPostFilterScratchSize{}) {
		t.Fatalf("scratch=%+v", size)
	}
	next, result, err := ctx.ApplySupportedPostFilters(FrameWorkPostFilterRequest{CDEF: req})
	if err != nil {
		t.Fatal(err)
	}
	if result.Completed != FrameWorkPostFilterCDEF|FrameWorkPostFilterFilmGrain ||
		result.CDEF.Planes != 1 ||
		!result.FilmGrain.NoOp ||
		result.FilmGrain.Plan.Seed != 0x4321 {
		t.Fatalf("result=%+v", result)
	}
	if err := next.RequireNoRemainingPostFilters(); err != nil {
		t.Fatalf("RequireNoRemainingPostFilters err=%v", err)
	}
	if !testFrameWorkCDEFPlaneChanged(output.Y, before) {
		t.Fatal("CDEF stage did not run before film grain")
	}
}

func TestFrameWorkPostFilterContextApplySupportedPostFiltersNoOpFilmGrainAllocs(t *testing.T) {
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: 64, Height: 32, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32})
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			FilmGrain: parser.FilmGrainParams{
				ParamsPresent: true,
				Apply:         true,
				Seed:          0x1234,
				BitDepth:      8,
			},
		},
		Output: output,
	}
	allocs := testing.AllocsPerRun(1000, func() {
		if _, err := ctx.SupportedPostFilterScratchLen(FrameWorkPostFilterRequest{}); err != nil {
			t.Fatal(err)
		}
		next, result, err := ctx.ApplySupportedPostFilters(FrameWorkPostFilterRequest{})
		if err != nil {
			t.Fatal(err)
		}
		if result.Completed != FrameWorkPostFilterFilmGrain || !result.FilmGrain.NoOp || next.RemainingPostFilters() != 0 {
			t.Fatalf("next remaining=%b result=%+v", next.RemainingPostFilters(), result)
		}
	})
	if allocs != 0 {
		t.Fatalf("no-op film grain postfilter pipeline allocated: %f", allocs)
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
