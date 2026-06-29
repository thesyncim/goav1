package decoder

import "github.com/thesyncim/goav1/internal/av1/frame"

// FrameWorkLayerPoolPostFilterRunner applies AV1 postfilters for callers that
// publish references through a frame.LayerPool. It preserves the coded
// frame-pool path for ordinary frames and, when superres is active, acquires the
// upscaled output surface from Pool and reports that global surface ID through
// PublishedFrameWorkGlobalSurface.
type FrameWorkLayerPoolPostFilterRunner struct {
	Pool *frame.LayerPool

	Options FrameWorkPostFilterBindOptions
	Scratch FrameWorkPostFilterScratch

	Size          FrameWorkPostFilterScratchSize
	Request       FrameWorkPostFilterRequest
	Context       FrameWorkPostFilterContext
	Result        FrameWorkCallerPostFilterResult
	DisplayOutput *frame.Frame

	PublishedGlobalSurface int

	displayOutputView frame.Frame
}

// Apply runs the postfilter chain for ctx. Superres frames are written to an
// upscaled LayerPool surface so reference publication can follow libaom's
// realloc-and-upscale semantics without retaining the coded reconstruction
// surface.
func (r *FrameWorkLayerPoolPostFilterRunner) Apply(ctx FrameWorkPostFilterContext) error {
	if r == nil {
		return ErrInvalidFrameWorkState
	}
	r.PublishedGlobalSurface = -1
	r.DisplayOutput = nil
	r.Size = FrameWorkPostFilterScratchSize{}
	r.Request = FrameWorkPostFilterRequest{}
	r.Context = FrameWorkPostFilterContext{}
	r.Result = FrameWorkCallerPostFilterResult{}

	if !ctx.RemainingPostFilters().Has(FrameWorkPostFilterSuperRes) {
		return r.applySupported(ctx)
	}
	return r.applySuperRes(ctx)
}

// PublishedFrameWorkGlobalSurface implements FrameWorkPostFilterGlobalSurfacePublisher.
func (r *FrameWorkLayerPoolPostFilterRunner) PublishedFrameWorkGlobalSurface() (int, bool) {
	if r == nil || r.PublishedGlobalSurface < 0 {
		return -1, false
	}
	return r.PublishedGlobalSurface, true
}

func (r *FrameWorkLayerPoolPostFilterRunner) applySupported(ctx FrameWorkPostFilterContext) error {
	options := r.optionsFromContext(ctx)
	post := FrameWorkBoundSupportedPostFilterRunner{
		Options: options,
		Scratch: r.Scratch,
	}
	if err := post.Apply(ctx); err != nil {
		return err
	}
	r.Size = post.Size
	r.Request = post.Request
	r.Context = post.Context
	r.Result = FrameWorkCallerPostFilterResult{FrameWorkPostFilterResult: post.Result}
	r.DisplayOutput = post.DisplayOutput
	return nil
}

func (r *FrameWorkLayerPoolPostFilterRunner) applySuperRes(ctx FrameWorkPostFilterContext) error {
	if r.Pool == nil {
		return frame.ErrInvalidPool
	}
	if ctx.Output == nil {
		return frame.ErrInvalidSlot
	}
	format, err := outputFrameFormatFromHeaders(ctx.Event.SequenceHeader, ctx.Event.FrameSize, ctx.Output.Format.Align)
	if err != nil {
		return err
	}
	sub, err := r.Pool.SubPool(format)
	if err != nil {
		return err
	}
	local, output, err := sub.AcquireFormat(format)
	if err != nil {
		return err
	}
	published := false
	defer func() {
		if !published {
			_ = sub.Release(local)
		}
	}()

	globalID, err := r.Pool.EncodeGlobalID(sub, local)
	if err != nil || globalID < 0 {
		return ErrInvalidSurfaceReference
	}
	backing, err := frameWorkPostFilterFrameBacking(output)
	if err != nil {
		return err
	}

	options := r.optionsFromContext(ctx)
	options.SuperResOutputView = output
	probe := frameWorkCallerPostFilterProbeRequest(options)
	probe.SuperRes.OutputFrame = backing
	probe.SuperRes.OutputView = output
	size, err := ctx.CallerPostFilterScratchLen(probe)
	if err != nil {
		return err
	}
	scratch := r.Scratch
	scratch.SuperResOutputFrame = backing
	req, err := size.BindRequest(options, scratch)
	if err != nil {
		return err
	}
	next, display, result, err := ctx.ApplyCallerPostFiltersForPublication(req)
	if err != nil {
		return err
	}

	r.Size = size
	r.Request = req
	r.Context = next
	r.Result = result
	r.DisplayOutput = display
	r.PublishedGlobalSurface = globalID
	published = true
	return nil
}

func (r *FrameWorkLayerPoolPostFilterRunner) optionsFromContext(ctx FrameWorkPostFilterContext) FrameWorkPostFilterBindOptions {
	options := frameWorkPostFilterBindOptionsFromContext(ctx, r.Options)
	if options.FilmGrainOutputView == nil {
		options.FilmGrainOutputView = &r.displayOutputView
	}
	return options
}

func frameWorkCallerPostFilterProbeRequest(options FrameWorkPostFilterBindOptions) FrameWorkPostFilterRequest {
	return FrameWorkPostFilterRequest{
		LoopFilter: FrameWorkLoopFilterPostFilterRequest{
			Map:             options.LoopFilterMap,
			TrustedCoverage: options.LoopFilterTrustedCoverage,
		},
		CDEF: FrameWorkCDEFPostFilterRequest{IndexMap: options.CDEFIndexMap},
		Restoration: FrameWorkRestorationPostFilterRequest{
			Records:   options.RestorationRecords,
			Optimized: options.RestorationOptimized,
		},
	}
}

func frameWorkPostFilterFrameBacking(output *frame.Frame) ([]byte, error) {
	if output == nil {
		return nil, frame.ErrInvalidSlot
	}
	if output.Layout.Size < 0 || cap(output.Y.Pix) < output.Layout.Size {
		return nil, frame.ErrShortBuffer
	}
	return output.Y.Pix[:output.Layout.Size], nil
}
