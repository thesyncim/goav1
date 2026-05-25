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
	if plan.Planes[0] != (FrameWorkSuperResPostFilterPlanePlan{CodedWidth: 16, OutputWidth: 32, Height: 16, WidthDelta: 16, BytesPerRow: 32, OutputStride: 32}) {
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

func testFrameWorkSuperResPostFilterRequest(t *testing.T, ctx FrameWorkPostFilterContext) FrameWorkSuperResPostFilterRequest {
	t.Helper()
	size, err := ctx.SuperResPostFilterScratchLen()
	if err != nil {
		t.Fatal(err)
	}
	req := FrameWorkSuperResPostFilterRequest{
		OutputFrame: make([]byte, size.OutputFrame),
	}
	for plane := 0; plane < 3; plane++ {
		req.CodedScratch[plane] = make([]uint16, size.CodedSamples[plane])
		req.OutputScratch[plane] = make([]uint16, size.OutputSamples[plane])
	}
	return req
}
