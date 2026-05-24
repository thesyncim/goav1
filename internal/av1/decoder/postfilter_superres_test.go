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
