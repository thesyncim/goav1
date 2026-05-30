package decoder

import (
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/parser"
)

func TestFrameWorkPostFilterContextSuperResPostFilterPlanSkipsInactive(t *testing.T) {
	plan, err := (FrameWorkPostFilterContext{}).SuperResPostFilterPlan()
	if err != nil {
		t.Fatal(err)
	}
	if plan != (FrameWorkSuperResPostFilterPlan{}) {
		t.Fatalf("plan=%+v want zero", plan)
	}
	size, err := (FrameWorkPostFilterContext{}).SuperResPostFilterScratchLen()
	if err != nil {
		t.Fatal(err)
	}
	if size != (FrameWorkSuperResPostFilterScratchSize{}) {
		t.Fatalf("scratch=%+v want zero", size)
	}
}

func TestFrameWorkPostFilterContextSuperResPostFilterPlanReportsOutputGeometry(t *testing.T) {
	seq := testSequence()
	event := Event{
		SequenceHeader: seq,
		FrameSize: parser.FrameSize{
			CodedWidth:          16,
			UpscaledWidth:       32,
			Height:              16,
			SuperResEnabled:     true,
			SuperResDenominator: 16,
		},
	}
	codedFormat, err := codedFrameFormatFromHeaders(seq, event.FrameSize, 32)
	if err != nil {
		t.Fatal(err)
	}
	output := testFrameWorkCDEFFrame(t, codedFormat)

	ctx := FrameWorkPostFilterContext{Event: event, Output: output}
	plan, err := ctx.SuperResPostFilterPlan()
	if err != nil {
		t.Fatal(err)
	}
	if !plan.Active || plan.Denominator != 16 || plan.CodedFormat.Width != 16 || plan.OutputFormat.Width != 32 {
		t.Fatalf("plan=%+v", plan)
	}
	// The coded/output frames are allocated at the superblock-aligned extent
	// (16/32 -> 64), so OutputStride is align(64, Align=32) = 64.
	if plan.Planes[0] != (FrameWorkSuperResPostFilterPlanePlan{CodedWidth: 16, OutputWidth: 32, Height: 16, WidthDelta: 16, BytesPerRow: 32, OutputStride: 64}) {
		t.Fatalf("luma plane=%+v", plan.Planes[0])
	}
	if plan.Planes[1].CodedWidth != 8 || plan.Planes[1].OutputWidth != 16 || plan.Planes[1].Height != 8 || plan.Planes[1].WidthDelta != 8 {
		t.Fatalf("chroma plane=%+v", plan.Planes[1])
	}
	outputLayout, err := frame.RequiredSize(plan.OutputFormat)
	if err != nil {
		t.Fatal(err)
	}
	if plan.OutputSize != outputLayout.Size {
		t.Fatalf("output size=%d want %d", plan.OutputSize, outputLayout.Size)
	}
	scratch, err := ctx.SuperResPostFilterScratchLen()
	if err != nil {
		t.Fatal(err)
	}
	if scratch.OutputFrame != outputLayout.Size {
		t.Fatalf("scratch=%+v want output frame %d", scratch, outputLayout.Size)
	}
	if scratch.CodedSamples[0] == 0 || scratch.OutputSamples[0] == 0 {
		t.Fatalf("scratch samples=%+v", scratch)
	}
}

func TestFrameWorkPostFilterContextSuperResPostFilterPlanRejectsInvalidActiveState(t *testing.T) {
	seq := testSequence()
	event := Event{
		SequenceHeader: seq,
		FrameSize: parser.FrameSize{
			CodedWidth:          16,
			UpscaledWidth:       32,
			Height:              16,
			SuperResEnabled:     true,
			SuperResDenominator: 8,
		},
	}
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: 16, Height: 16, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32})
	ctx := FrameWorkPostFilterContext{Event: event, Output: output}
	if _, err := ctx.SuperResPostFilterPlan(); !errors.Is(err, frame.ErrInvalidFormat) {
		t.Fatalf("SuperResPostFilterPlan err=%v want %v", err, frame.ErrInvalidFormat)
	}
}

func TestFrameWorkPostFilterContextSuperResPostFilterPlanRejectsOutputMismatch(t *testing.T) {
	seq := testSequence()
	event := Event{
		SequenceHeader: seq,
		FrameSize: parser.FrameSize{
			CodedWidth:          16,
			UpscaledWidth:       32,
			Height:              16,
			SuperResEnabled:     true,
			SuperResDenominator: 16,
		},
	}
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: 32, Height: 16, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32})
	ctx := FrameWorkPostFilterContext{Event: event, Output: output}
	if _, err := ctx.SuperResPostFilterPlan(); !errors.Is(err, frame.ErrInvalidFormat) {
		t.Fatalf("SuperResPostFilterPlan err=%v want %v", err, frame.ErrInvalidFormat)
	}
}

func TestFrameWorkPostFilterContextApplySuperResPostFilterWritesOutputScratch(t *testing.T) {
	seq := testSequence()
	event := Event{
		SequenceHeader: seq,
		FrameSize: parser.FrameSize{
			CodedWidth:          8,
			UpscaledWidth:       13,
			Height:              2,
			SuperResEnabled:     true,
			SuperResDenominator: 13,
		},
	}
	codedFormat, err := codedFrameFormatFromHeaders(seq, event.FrameSize, 32)
	if err != nil {
		t.Fatal(err)
	}
	output := testFrameWorkCDEFFrame(t, codedFormat)
	for x, value := range []byte{0, 32, 64, 96, 128, 160, 192, 224} {
		output.Y.Pix[x] = value
		output.Y.Pix[output.Y.Stride+x] = 100
	}
	for x, value := range []byte{0, 64, 128, 192} {
		output.U.Pix[x] = value
		output.V.Pix[x] = 55
	}

	ctx := FrameWorkPostFilterContext{Event: event, Output: output}
	req := testFrameWorkSuperResPostFilterRequest(t, ctx)
	result, err := ctx.ApplySuperResPostFilter(req)
	if err != nil {
		t.Fatal(err)
	}
	if result.Planes != 3 || result.Output.Format.Width != 13 || result.OutputSize != len(req.OutputFrame) {
		t.Fatalf("result=%+v", result)
	}
	for x, want := range []byte{0, 11, 33, 54, 73, 92, 112, 132, 151, 171, 191, 214, 226} {
		if got := result.Output.Y.Pix[x]; got != want {
			t.Fatalf("Y[%d]=%d want %d", x, got, want)
		}
	}
	for x := 0; x < 13; x++ {
		if got := result.Output.Y.Pix[result.Output.Y.Stride+x]; got != 100 {
			t.Fatalf("Y row1[%d]=%d want 100", x, got)
		}
	}
	for x, want := range []byte{0, 19, 59, 96, 134, 174, 197} {
		if got := result.Output.U.Pix[x]; got != want {
			t.Fatalf("U[%d]=%d want %d", x, got, want)
		}
		if got := result.Output.V.Pix[x]; got != 55 {
			t.Fatalf("V[%d]=%d want 55", x, got)
		}
	}
	if output.Y.Pix[1] != 32 {
		t.Fatalf("coded output mutated: %d", output.Y.Pix[1])
	}
}

func TestFrameWorkPostFilterContextApplySuperResPostFilterToContextCompletesStage(t *testing.T) {
	seq := testSequence()
	event := Event{
		SequenceHeader: seq,
		FrameSize: parser.FrameSize{
			CodedWidth:          8,
			UpscaledWidth:       13,
			Height:              2,
			SuperResEnabled:     true,
			SuperResDenominator: 13,
		},
	}
	codedFormat, err := codedFrameFormatFromHeaders(seq, event.FrameSize, 32)
	if err != nil {
		t.Fatal(err)
	}
	output := testFrameWorkCDEFFrame(t, codedFormat)
	for x, value := range []byte{0, 32, 64, 96, 128, 160, 192, 224} {
		output.Y.Pix[x] = value
	}

	ctx := FrameWorkPostFilterContext{Event: event, Output: output}
	req := testFrameWorkSuperResPostFilterRequest(t, ctx)
	next, result, err := ctx.ApplySuperResPostFilterToContext(req)
	if err != nil {
		t.Fatal(err)
	}
	if result.Planes != 3 || next.RemainingPostFilters() != 0 {
		t.Fatalf("result=%+v remaining=%b", result, next.RemainingPostFilters())
	}
	if !next.DetachedPostFilterOutput() {
		t.Fatal("superres context did not report detached output")
	}
	if next.Output == nil {
		t.Fatal("next output is nil")
	}
	if next.Output == output || next.Output.Format.Width != 13 {
		t.Fatalf("next output=%p format=%+v coded=%p", next.Output, next.Output.Format, output)
	}
	if got := next.Output.Y.Pix[1]; got != 11 {
		t.Fatalf("upscaled Y[1]=%d want 11", got)
	}
	if output.Y.Pix[1] != 32 {
		t.Fatalf("coded output mutated: %d", output.Y.Pix[1])
	}
	if err := next.RequireNoRemainingPostFilters(); err != nil {
		t.Fatalf("RequireNoRemainingPostFilters err=%v", err)
	}
	if err := next.RequirePublishablePostFilterOutput(); !errors.Is(err, ErrUnsupportedPostFilter) {
		t.Fatalf("RequirePublishablePostFilterOutput err=%v want %v", err, ErrUnsupportedPostFilter)
	}
}

func TestFrameWorkPostFilterContextApplySuperResPostFilterToContextFeedsNoOpFilmGrain(t *testing.T) {
	seq := testSequence()
	event := Event{
		SequenceHeader: seq,
		FrameSize: parser.FrameSize{
			CodedWidth:          8,
			UpscaledWidth:       13,
			Height:              2,
			SuperResEnabled:     true,
			SuperResDenominator: 13,
		},
		FilmGrain: parser.FilmGrainParams{
			ParamsPresent: true,
			Apply:         true,
			BitDepth:      8,
		},
	}
	codedFormat, err := codedFrameFormatFromHeaders(seq, event.FrameSize, 32)
	if err != nil {
		t.Fatal(err)
	}
	output := testFrameWorkCDEFFrame(t, codedFormat)
	for x, value := range []byte{0, 32, 64, 96, 128, 160, 192, 224} {
		output.Y.Pix[x] = value
	}

	ctx := FrameWorkPostFilterContext{Event: event, Output: output}
	req := testFrameWorkSuperResPostFilterRequest(t, ctx)
	next, result, err := ctx.ApplySuperResPostFilterToContext(req)
	if err != nil {
		t.Fatal(err)
	}
	if result.Output.Format.Width != 13 || next.RemainingPostFilters() != FrameWorkPostFilterFilmGrain {
		t.Fatalf("superres result=%+v remaining=%b", result, next.RemainingPostFilters())
	}
	filmGrainResult, err := next.ApplyFilmGrainPostFilter(FrameWorkFilmGrainPostFilterRequest{})
	if err != nil {
		t.Fatal(err)
	}
	if !filmGrainResult.NoOp || filmGrainResult.Plan.Format.Width != 13 {
		t.Fatalf("film grain result=%+v", filmGrainResult)
	}
	next = next.WithCompletedPostFilters(FrameWorkPostFilterFilmGrain)
	if err := next.RequireNoRemainingPostFilters(); err != nil {
		t.Fatalf("RequireNoRemainingPostFilters err=%v", err)
	}
}

func TestFrameWorkPostFilterContextApplyCallerPostFiltersRunsSuperResThenFilmGrain(t *testing.T) {
	seq := testSequence()
	event := Event{
		SequenceHeader: seq,
		FrameSize: parser.FrameSize{
			CodedWidth:          8,
			UpscaledWidth:       13,
			Height:              32,
			SuperResEnabled:     true,
			SuperResDenominator: 13,
		},
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
	}
	codedFormat, err := codedFrameFormatFromHeaders(seq, event.FrameSize, 32)
	if err != nil {
		t.Fatal(err)
	}
	output := testFrameWorkCDEFFrame(t, codedFormat)
	for y := 0; y < output.Y.Height; y++ {
		for x := 0; x < output.Y.Width; x++ {
			output.Y.Pix[y*output.Y.Stride+x] = 100
		}
	}

	ctx := FrameWorkPostFilterContext{Event: event, Output: output}
	size, err := ctx.CallerPostFilterScratchLen(FrameWorkPostFilterRequest{})
	if err != nil {
		t.Fatal(err)
	}
	if size.SuperRes.OutputFrame == 0 || size.FilmGrain != (FrameWorkFilmGrainPostFilterScratchSize{}) {
		t.Fatalf("first scratch=%+v", size)
	}
	req := FrameWorkPostFilterRequest{
		SuperRes: testFrameWorkSuperResPostFilterRequest(t, ctx),
	}
	size, err = ctx.CallerPostFilterScratchLen(req)
	if err != nil {
		t.Fatal(err)
	}
	if size.SuperRes.OutputFrame == 0 || size.FilmGrain.LumaGrain == 0 || size.FilmGrain.LumaSamples == 0 {
		t.Fatalf("second scratch=%+v", size)
	}
	req.FilmGrain = testFrameWorkFilmGrainPostFilterRequest(size.FilmGrain)

	next, result, err := ctx.ApplyCallerPostFilters(req)
	if err != nil {
		t.Fatal(err)
	}
	wantCompleted := FrameWorkPostFilterSuperRes | FrameWorkPostFilterFilmGrain
	if result.Completed != wantCompleted ||
		result.SuperRes.Output.Format.Width != 13 ||
		result.FilmGrain.NoOp ||
		result.FilmGrain.LumaRows != 1 ||
		next.RemainingPostFilters() != 0 ||
		!next.DetachedPostFilterOutput() {
		t.Fatalf("next remaining=%b detached=%v result=%+v", next.RemainingPostFilters(), next.DetachedPostFilterOutput(), result)
	}
	if !testFrameWorkFilmGrainPlaneChangedFromValue(next.Output.Y, next.Output.Layout.BytesPerSample, 100) {
		t.Fatal("film grain did not change detached superres output")
	}
	if output.Y.Pix[0] != 100 {
		t.Fatalf("coded output mutated after detached stages: %d", output.Y.Pix[0])
	}
	if err := next.RequirePublishablePostFilterOutput(); !errors.Is(err, ErrUnsupportedPostFilter) {
		t.Fatalf("RequirePublishablePostFilterOutput err=%v want %v", err, ErrUnsupportedPostFilter)
	}
}

func TestFrameWorkCallerPostFilterRunnerApplyRunsSuperRes(t *testing.T) {
	seq := testSequence()
	event := Event{
		SequenceHeader: seq,
		FrameSize: parser.FrameSize{
			CodedWidth:          8,
			UpscaledWidth:       13,
			Height:              16,
			SuperResEnabled:     true,
			SuperResDenominator: 13,
		},
	}
	codedFormat, err := codedFrameFormatFromHeaders(seq, event.FrameSize, 32)
	if err != nil {
		t.Fatal(err)
	}
	output := testFrameWorkCDEFFrame(t, codedFormat)
	for y := 0; y < output.Y.Height; y++ {
		for x := 0; x < output.Y.Width; x++ {
			output.Y.Pix[y*output.Y.Stride+x] = byte(64 + x)
		}
	}

	ctx := FrameWorkPostFilterContext{Event: event, Output: output}
	req := FrameWorkPostFilterRequest{
		SuperRes: testFrameWorkSuperResPostFilterRequest(t, ctx),
	}
	runner := FrameWorkCallerPostFilterRunner{Request: req}
	size, err := runner.ScratchLen(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if size.SuperRes.OutputFrame == 0 || size.CDEF != (FrameWorkCDEFPostFilterScratchSize{}) ||
		size.Restoration != (FrameWorkRestorationPostFilterScratchSize{}) ||
		size.FilmGrain != (FrameWorkFilmGrainPostFilterScratchSize{}) {
		t.Fatalf("runner scratch=%+v", size)
	}
	if err := runner.Apply(ctx); err != nil {
		t.Fatal(err)
	}
	if runner.Result.Completed != FrameWorkPostFilterSuperRes ||
		runner.Result.SuperRes.Output.Format.Width != 13 ||
		runner.Context.RemainingPostFilters() != 0 ||
		!runner.Context.DetachedPostFilterOutput() ||
		runner.Context.Output.Format.Width != 13 {
		t.Fatalf("runner context detached=%v remaining=%b result=%+v", runner.Context.DetachedPostFilterOutput(), runner.Context.RemainingPostFilters(), runner.Result)
	}
	if err := runner.Context.RequireNoRemainingPostFilters(); err != nil {
		t.Fatalf("RequireNoRemainingPostFilters err=%v", err)
	}
	if err := runner.Context.RequirePublishablePostFilterOutput(); !errors.Is(err, ErrUnsupportedPostFilter) {
		t.Fatalf("RequirePublishablePostFilterOutput err=%v want %v", err, ErrUnsupportedPostFilter)
	}
	if output.Format.Width != 8 || output.Y.Pix[0] != 64 {
		t.Fatalf("coded output mutated: width=%d first=%d", output.Format.Width, output.Y.Pix[0])
	}
}

func TestFrameWorkPostFilterContextApplySuperResPostFilterRejectsShortOutputScratch(t *testing.T) {
	seq := testSequence()
	event := Event{
		SequenceHeader: seq,
		FrameSize: parser.FrameSize{
			CodedWidth:          8,
			UpscaledWidth:       13,
			Height:              2,
			SuperResEnabled:     true,
			SuperResDenominator: 13,
		},
	}
	codedFormat, err := codedFrameFormatFromHeaders(seq, event.FrameSize, 32)
	if err != nil {
		t.Fatal(err)
	}
	output := testFrameWorkCDEFFrame(t, codedFormat)
	ctx := FrameWorkPostFilterContext{Event: event, Output: output}
	size, err := ctx.SuperResPostFilterScratchLen()
	if err != nil {
		t.Fatal(err)
	}
	req := FrameWorkSuperResPostFilterRequest{OutputFrame: make([]byte, size.OutputFrame-1)}
	if _, err := ctx.ApplySuperResPostFilter(req); !errors.Is(err, frame.ErrShortBuffer) {
		t.Fatalf("ApplySuperResPostFilter err=%v want %v", err, frame.ErrShortBuffer)
	}
}

func TestFrameWorkPostFilterContextApplySuperResPostFilterRejectsShortPlaneScratchBeforeOutputMutation(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*FrameWorkSuperResPostFilterRequest)
	}{
		{
			name: "coded-chroma",
			mutate: func(req *FrameWorkSuperResPostFilterRequest) {
				req.CodedScratch[1] = req.CodedScratch[1][:len(req.CodedScratch[1])-1]
			},
		},
		{
			name: "output-chroma",
			mutate: func(req *FrameWorkSuperResPostFilterRequest) {
				req.OutputScratch[1] = req.OutputScratch[1][:len(req.OutputScratch[1])-1]
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			seq := testSequence()
			event := Event{
				SequenceHeader: seq,
				FrameSize: parser.FrameSize{
					CodedWidth:          8,
					UpscaledWidth:       13,
					Height:              2,
					SuperResEnabled:     true,
					SuperResDenominator: 13,
				},
			}
			codedFormat, err := codedFrameFormatFromHeaders(seq, event.FrameSize, 32)
			if err != nil {
				t.Fatal(err)
			}
			output := testFrameWorkCDEFFrame(t, codedFormat)
			for x, value := range []byte{0, 32, 64, 96, 128, 160, 192, 224} {
				output.Y.Pix[x] = value
			}
			for x, value := range []byte{0, 64, 128, 192} {
				output.U.Pix[x] = value
				output.V.Pix[x] = value
			}

			ctx := FrameWorkPostFilterContext{Event: event, Output: output}
			req := testFrameWorkSuperResPostFilterRequest(t, ctx)
			for i := range req.OutputFrame {
				req.OutputFrame[i] = 0xaa
			}
			tt.mutate(&req)

			if _, err := ctx.ApplySuperResPostFilter(req); !errors.Is(err, frame.ErrShortBuffer) {
				t.Fatalf("ApplySuperResPostFilter err=%v want %v", err, frame.ErrShortBuffer)
			}
			for i, got := range req.OutputFrame {
				if got != 0xaa {
					t.Fatalf("output frame byte %d mutated to %#x", i, got)
				}
			}
		})
	}
}

func TestFrameWorkPostFilterContextApplySuperResPostFilterRejectsEarlierStages(t *testing.T) {
	seq := testSequence()
	event := Event{
		SequenceHeader: seq,
		FrameSize: parser.FrameSize{
			CodedWidth:          8,
			UpscaledWidth:       13,
			Height:              2,
			SuperResEnabled:     true,
			SuperResDenominator: 13,
		},
		CDEF: parser.CDEFParams{StrengthCount: 1, YStrength: [parser.MaxCDEFStrengths]uint8{4}},
	}
	codedFormat, err := codedFrameFormatFromHeaders(seq, event.FrameSize, 32)
	if err != nil {
		t.Fatal(err)
	}
	output := testFrameWorkCDEFFrame(t, codedFormat)
	ctx := FrameWorkPostFilterContext{Event: event, Output: output}
	if _, err := ctx.ApplySuperResPostFilter(FrameWorkSuperResPostFilterRequest{}); !errors.Is(err, ErrUnsupportedPostFilter) {
		t.Fatalf("ApplySuperResPostFilter err=%v want %v", err, ErrUnsupportedPostFilter)
	}
}

func TestFrameWorkSuperResPostFilterScratchSizeBindRequest(t *testing.T) {
	size := FrameWorkSuperResPostFilterScratchSize{
		OutputFrame:   16,
		CodedSamples:  [3]int{8, 4, 2},
		OutputSamples: [3]int{9, 5, 3},
	}
	outputFrame, coded, output := testFrameWorkSuperResScratchStorage(size)
	var outputView frame.Frame
	req, err := size.BindRequest(outputFrame, coded, output, &outputView)
	if err != nil {
		t.Fatal(err)
	}
	if len(req.OutputFrame) != size.OutputFrame || req.OutputView != &outputView ||
		len(req.CodedScratch[0]) != size.CodedSamples[0] || len(req.CodedScratch[1]) != size.CodedSamples[1] ||
		len(req.OutputScratch[1]) != size.OutputSamples[1] || len(req.OutputScratch[2]) != size.OutputSamples[2] {
		t.Fatalf("request outputFrame=%d outputView=%p coded=%v output=%v",
			len(req.OutputFrame), req.OutputView,
			[3]int{len(req.CodedScratch[0]), len(req.CodedScratch[1]), len(req.CodedScratch[2])},
			[3]int{len(req.OutputScratch[0]), len(req.OutputScratch[1]), len(req.OutputScratch[2])})
	}
}

func TestFrameWorkSuperResPostFilterScratchSizeBindRequestRejectsShortBuffers(t *testing.T) {
	size := FrameWorkSuperResPostFilterScratchSize{
		OutputFrame:   16,
		CodedSamples:  [3]int{8, 4, 2},
		OutputSamples: [3]int{9, 5, 3},
	}
	outputFrame, coded, output := testFrameWorkSuperResScratchStorage(size)
	tests := []struct {
		name        string
		size        FrameWorkSuperResPostFilterScratchSize
		outputFrame []byte
		coded       [3][]uint16
		output      [3][]uint16
	}{
		{name: "output-frame", size: size, outputFrame: outputFrame[:15], coded: coded, output: output},
		{name: "coded", size: size, outputFrame: outputFrame, coded: [3][]uint16{coded[0][:7], coded[1], coded[2]}, output: output},
		{name: "output", size: size, outputFrame: outputFrame, coded: coded, output: [3][]uint16{output[0], output[1][:4], output[2]}},
		{name: "negative", size: FrameWorkSuperResPostFilterScratchSize{OutputFrame: -1}, outputFrame: outputFrame, coded: coded, output: output},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := tt.size.BindRequest(tt.outputFrame, tt.coded, tt.output, nil); !errors.Is(err, frame.ErrShortBuffer) {
				t.Fatalf("BindRequest err=%v want %v", err, frame.ErrShortBuffer)
			}
		})
	}
}

func TestFrameWorkSuperResPostFilterScratchSizeBindRequestAllocs(t *testing.T) {
	size := FrameWorkSuperResPostFilterScratchSize{
		OutputFrame:   1920 * 1080 * 3 / 2,
		CodedSamples:  [3]int{1280 * 720, 640 * 360, 640 * 360},
		OutputSamples: [3]int{1920 * 1080, 960 * 540, 960 * 540},
	}
	outputFrame, coded, output := testFrameWorkSuperResScratchStorage(size)
	allocs := testing.AllocsPerRun(1000, func() {
		if _, err := size.BindRequest(outputFrame, coded, output, nil); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("BindRequest allocated: %f", allocs)
	}
}

func BenchmarkFrameWorkSuperResPostFilterScratchSizeBindRequest(b *testing.B) {
	size := FrameWorkSuperResPostFilterScratchSize{
		OutputFrame:   1920 * 1080 * 3 / 2,
		CodedSamples:  [3]int{1280 * 720, 640 * 360, 640 * 360},
		OutputSamples: [3]int{1920 * 1080, 960 * 540, 960 * 540},
	}
	outputFrame, coded, output := testFrameWorkSuperResScratchStorage(size)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := size.BindRequest(outputFrame, coded, output, nil); err != nil {
			b.Fatal(err)
		}
	}
}

func testFrameWorkSuperResPostFilterRequest(t *testing.T, ctx FrameWorkPostFilterContext) FrameWorkSuperResPostFilterRequest {
	t.Helper()
	size, err := ctx.SuperResPostFilterScratchLen()
	if err != nil {
		t.Fatal(err)
	}
	outputFrame, coded, output := testFrameWorkSuperResScratchStorage(size)
	req, err := size.BindRequest(outputFrame, coded, output, nil)
	if err != nil {
		t.Fatal(err)
	}
	return req
}

func testFrameWorkSuperResScratchStorage(size FrameWorkSuperResPostFilterScratchSize) ([]byte, [3][]uint16, [3][]uint16) {
	var coded [3][]uint16
	var output [3][]uint16
	for plane := 0; plane < 3; plane++ {
		if size.CodedSamples[plane] > 0 {
			coded[plane] = make([]uint16, size.CodedSamples[plane])
		}
		if size.OutputSamples[plane] > 0 {
			output[plane] = make([]uint16, size.OutputSamples[plane])
		}
	}
	return make([]byte, maxInt(size.OutputFrame, 0)), coded, output
}
