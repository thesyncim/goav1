package decoder

import (
	"fmt"

	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/superres"
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

	CodedSamples  [3]int
	OutputSamples [3]int
}

// FrameWorkSuperResPostFilterRequest carries caller-owned scratch for
// ApplySuperResPostFilter.
type FrameWorkSuperResPostFilterRequest struct {
	OutputFrame []byte

	CodedScratch  [3][]uint16
	OutputScratch [3][]uint16
}

// FrameWorkSuperResPostFilterResult summarizes a scratch-targeted superres
// upscale. Output aliases the caller-owned request buffer.
type FrameWorkSuperResPostFilterResult struct {
	Plan       FrameWorkSuperResPostFilterPlan
	Output     frame.Frame
	OutputSize int
	Planes     int
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
	size := FrameWorkSuperResPostFilterScratchSize{OutputFrame: plan.OutputSize}
	for plane := 0; plane < len(plan.Planes); plane++ {
		planePlan := plan.Planes[plane]
		if planePlan.OutputWidth == 0 {
			continue
		}
		srcPlane, ok := frameWorkCDEFPlane(*ctx.Output, plane)
		if !ok {
			continue
		}
		bytesPerSample := frameWorkSuperResBytesPerSample(plan.CodedFormat)
		codedSamples, err := frame.SamplePlaneLen(srcPlane, bytesPerSample)
		if err != nil {
			return FrameWorkSuperResPostFilterScratchSize{}, err
		}
		outputSamples, err := frameWorkSuperResSampleScratchLen(planePlan.OutputWidth, planePlan.Height, planePlan.OutputStride, bytesPerSample)
		if err != nil {
			return FrameWorkSuperResPostFilterScratchSize{}, err
		}
		size.CodedSamples[plane] = codedSamples
		size.OutputSamples[plane] = outputSamples
	}
	return size, nil
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

// ApplySuperResPostFilter upscales ctx.Output into caller-owned scratch and
// returns an output-format frame view. It does not replace the published frame
// surface or mark superres complete in the normal postfilter runner.
func (ctx FrameWorkPostFilterContext) ApplySuperResPostFilter(req FrameWorkSuperResPostFilterRequest) (FrameWorkSuperResPostFilterResult, error) {
	remaining := ctx.RemainingPostFilters()
	preSuperRes := FrameWorkPostFilterLoopFilter | FrameWorkPostFilterCDEF
	if remaining&preSuperRes != 0 {
		return FrameWorkSuperResPostFilterResult{}, ErrUnsupportedPostFilter
	}
	if !remaining.Has(FrameWorkPostFilterSuperRes) {
		return FrameWorkSuperResPostFilterResult{}, nil
	}
	plan, err := ctx.SuperResPostFilterPlan()
	if err != nil {
		return FrameWorkSuperResPostFilterResult{}, err
	}
	if len(req.OutputFrame) < plan.OutputSize {
		return FrameWorkSuperResPostFilterResult{}, frame.ErrShortBuffer
	}
	output, err := frame.Bind(req.OutputFrame[:plan.OutputSize], plan.OutputFormat)
	if err != nil {
		return FrameWorkSuperResPostFilterResult{}, err
	}
	result := FrameWorkSuperResPostFilterResult{
		Plan:       plan,
		Output:     output,
		OutputSize: plan.OutputSize,
	}
	for plane := 0; plane < len(plan.Planes); plane++ {
		planePlan := plan.Planes[plane]
		if planePlan.OutputWidth == 0 {
			continue
		}
		srcPlane, ok := frameWorkCDEFPlane(*ctx.Output, plane)
		if !ok {
			continue
		}
		dstPlane, ok := frameWorkCDEFPlane(output, plane)
		if !ok {
			continue
		}
		srcSamples, err := frame.LoadSamplePlane(req.CodedScratch[plane], srcPlane, ctx.Output.Layout.BytesPerSample)
		if err != nil {
			return FrameWorkSuperResPostFilterResult{}, fmt.Errorf("decoder: superres coded scratch plane %d: %w", plane, err)
		}
		dstSamples, err := frame.LoadSamplePlane(req.OutputScratch[plane], dstPlane, output.Layout.BytesPerSample)
		if err != nil {
			return FrameWorkSuperResPostFilterResult{}, fmt.Errorf("decoder: superres output scratch plane %d: %w", plane, err)
		}
		if err := superres.UpscalePlane(srcSamples, dstSamples, plan.OutputFormat.BitDepth); err != nil {
			return FrameWorkSuperResPostFilterResult{}, fmt.Errorf("decoder: superres upscale plane %d: %w", plane, err)
		}
		if err := frame.StoreSamplePlane(dstPlane, output.Layout.BytesPerSample, dstSamples); err != nil {
			return FrameWorkSuperResPostFilterResult{}, fmt.Errorf("decoder: superres store plane %d: %w", plane, err)
		}
		result.Planes++
	}
	return result, nil
}

func frameWorkSuperResBytesPerSample(format frame.Format) int {
	if format.BitDepth > 8 {
		return 2
	}
	return 1
}

func frameWorkSuperResSampleScratchLen(width int, height int, stride int, bytesPerSample int) (int, error) {
	if width <= 0 || height <= 0 || bytesPerSample <= 0 || stride <= 0 || stride%bytesPerSample != 0 {
		return 0, frame.ErrInvalidPlane
	}
	strideSamples := stride / bytesPerSample
	if strideSamples < width {
		return 0, frame.ErrInvalidPlane
	}
	maxInt := int(^uint(0) >> 1)
	if height > maxInt/strideSamples {
		return 0, frame.ErrInvalidPlane
	}
	return height * strideSamples, nil
}
