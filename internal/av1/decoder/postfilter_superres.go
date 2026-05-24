package decoder

import (
	"github.com/thesyncim/goav1/internal/av1/frame"
)

// FrameWorkSuperResPostFilterPlanePlan describes one plane's coded and
// upscaled dimensions.
type FrameWorkSuperResPostFilterPlanePlan struct {
	CodedWidth   int
	OutputWidth  int
	Height       int
	WidthDelta   int
	BytesPerRow  int
	OutputStride int
}

// FrameWorkSuperResPostFilterPlan summarizes the geometry transition from the
// coded reconstruction surface to AV1's upscaled output surface.
type FrameWorkSuperResPostFilterPlan struct {
	Active bool

	Denominator uint8

	CodedFormat  frame.Format
	OutputFormat frame.Format

	CodedSize  int
	OutputSize int

	Planes [3]FrameWorkSuperResPostFilterPlanePlan
}

// FrameWorkSuperResPostFilterScratchSize reports caller-owned scratch needed
// by the eventual superres resampler.
type FrameWorkSuperResPostFilterScratchSize struct {
	OutputFrame int
}

// SuperResPostFilterPlan returns the frame-level superres geometry for the
// current output. It does not resample or mutate ctx.Output.
func (ctx FrameWorkPostFilterContext) SuperResPostFilterPlan() (FrameWorkSuperResPostFilterPlan, error) {
	if !ctx.RemainingPostFilters().Has(FrameWorkPostFilterSuperRes) {
		return FrameWorkSuperResPostFilterPlan{}, nil
	}
	if ctx.Output == nil {
		return FrameWorkSuperResPostFilterPlan{}, frame.ErrInvalidSlot
	}
	if ctx.Event.FrameSize.SuperResDenominator < 9 || ctx.Event.FrameSize.SuperResDenominator > 16 ||
		ctx.Event.FrameSize.CodedWidth == 0 ||
		ctx.Event.FrameSize.UpscaledWidth <= ctx.Event.FrameSize.CodedWidth ||
		ctx.Event.FrameSize.Height == 0 {
		return FrameWorkSuperResPostFilterPlan{}, frame.ErrInvalidFormat
	}

	codedFormat, err := codedFrameFormatFromHeaders(ctx.Event.SequenceHeader, ctx.Event.FrameSize, ctx.Output.Format.Align)
	if err != nil {
		return FrameWorkSuperResPostFilterPlan{}, err
	}
	if ctx.Output.Format != codedFormat {
		return FrameWorkSuperResPostFilterPlan{}, frame.ErrInvalidFormat
	}
	codedLayout, err := frame.RequiredSize(codedFormat)
	if err != nil {
		return FrameWorkSuperResPostFilterPlan{}, err
	}
	outputFormat, err := outputFrameFormatFromHeaders(ctx.Event.SequenceHeader, ctx.Event.FrameSize, ctx.Output.Format.Align)
	if err != nil {
		return FrameWorkSuperResPostFilterPlan{}, err
	}
	outputLayout, err := frame.RequiredSize(outputFormat)
	if err != nil {
		return FrameWorkSuperResPostFilterPlan{}, err
	}

	plan := FrameWorkSuperResPostFilterPlan{
		Active:       true,
		Denominator:  ctx.Event.FrameSize.SuperResDenominator,
		CodedFormat:  codedFormat,
		OutputFormat: outputFormat,
		CodedSize:    codedLayout.Size,
		OutputSize:   outputLayout.Size,
	}
	plan.Planes[0] = frameWorkSuperResPlanePlan(codedFormat.Width, outputFormat.Width, codedFormat.Height, codedLayout.BytesPerSample, outputLayout.YStride)
	if !codedFormat.MonoChrome {
		plan.Planes[1] = frameWorkSuperResPlanePlan(codedLayout.ChromaWidth, outputLayout.ChromaWidth, codedLayout.ChromaHeight, codedLayout.BytesPerSample, outputLayout.UStride)
		plan.Planes[2] = frameWorkSuperResPlanePlan(codedLayout.ChromaWidth, outputLayout.ChromaWidth, codedLayout.ChromaHeight, codedLayout.BytesPerSample, outputLayout.VStride)
	}
	return plan, nil
}

// SuperResPostFilterScratchLen reports scratch lengths needed to hold the
// upscaled output frame for the current superres postfilter.
func (ctx FrameWorkPostFilterContext) SuperResPostFilterScratchLen() (FrameWorkSuperResPostFilterScratchSize, error) {
	plan, err := ctx.SuperResPostFilterPlan()
	if err != nil {
		return FrameWorkSuperResPostFilterScratchSize{}, err
	}
	if !plan.Active {
		return FrameWorkSuperResPostFilterScratchSize{}, nil
	}
	return FrameWorkSuperResPostFilterScratchSize{OutputFrame: plan.OutputSize}, nil
}

func frameWorkSuperResPlanePlan(codedWidth int, outputWidth int, height int, bytesPerSample int, outputStride int) FrameWorkSuperResPostFilterPlanePlan {
	return FrameWorkSuperResPostFilterPlanePlan{
		CodedWidth:   codedWidth,
		OutputWidth:  outputWidth,
		Height:       height,
		WidthDelta:   outputWidth - codedWidth,
		BytesPerRow:  outputWidth * bytesPerSample,
		OutputStride: outputStride,
	}
}
