package decoder

import (
	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/tile"
)

// FrameWorkPostFilterStage identifies a frame-level postfilter that must run
// after tile reconstruction and before reference publication.
type FrameWorkPostFilterStage uint8

const (
	FrameWorkPostFilterLoopFilter FrameWorkPostFilterStage = 1 << iota
	FrameWorkPostFilterCDEF
	FrameWorkPostFilterSuperRes
	FrameWorkPostFilterLoopRestoration
	FrameWorkPostFilterFilmGrain
)

// Empty reports whether no postfilter stages are active.
func (s FrameWorkPostFilterStage) Empty() bool {
	return s == 0
}

// Has reports whether all stages in stage are present.
func (s FrameWorkPostFilterStage) Has(stage FrameWorkPostFilterStage) bool {
	return s&stage == stage
}

// FrameWorkRestorationPostFilterScratchSize reports caller-owned scratch needed
// by ApplyLoopRestorationPostFilter.
type FrameWorkRestorationPostFilterScratchSize struct {
	Samples tile.RestorationFrameSampleScratchSize
	Apply   tile.RestorationUnitRecordBoundaryScratchSize
}

// FrameWorkPostFilterScratchSize reports caller-owned scratch needed by the
// supported frame-level postfilter pipeline.
type FrameWorkPostFilterScratchSize struct {
	LoopFilter  FrameWorkLoopFilterPostFilterScratchSize
	CDEF        FrameWorkCDEFPostFilterScratchSize
	SuperRes    FrameWorkSuperResPostFilterScratchSize
	Restoration FrameWorkRestorationPostFilterScratchSize
	FilmGrain   FrameWorkFilmGrainPostFilterScratchSize
}

// Max returns the per-field maximum scratch size needed to satisfy either
// postfilter scratch plan. Callers can accumulate this across frames to keep a
// reusable realtime scratch arena.
func (s FrameWorkPostFilterScratchSize) Max(other FrameWorkPostFilterScratchSize) FrameWorkPostFilterScratchSize {
	return FrameWorkPostFilterScratchSize{
		LoopFilter:  s.LoopFilter.Max(other.LoopFilter),
		CDEF:        s.CDEF.Max(other.CDEF),
		SuperRes:    s.SuperRes.Max(other.SuperRes),
		Restoration: s.Restoration.Max(other.Restoration),
		FilmGrain:   s.FilmGrain.Max(other.FilmGrain),
	}
}

// Max returns the per-field maximum restoration scratch size.
func (s FrameWorkRestorationPostFilterScratchSize) Max(other FrameWorkRestorationPostFilterScratchSize) FrameWorkRestorationPostFilterScratchSize {
	return FrameWorkRestorationPostFilterScratchSize{
		Samples: frameWorkRestorationFrameSampleScratchSizeMax(s.Samples, other.Samples),
		Apply:   frameWorkRestorationUnitRecordBoundaryScratchSizeMax(s.Apply, other.Apply),
	}
}

// FrameWorkRestorationPostFilterRequest carries decoded loop-restoration state
// and caller-owned scratch for ApplyLoopRestorationPostFilter.
type FrameWorkRestorationPostFilterRequest struct {
	Records    [3][]tile.RestorationUnitRecord
	Boundaries [3]tile.RestorationStripeBoundaries

	DataScratch []uint16
	DstScratch  []uint16
	Scratch     tile.RestorationUnitRecordBoundaryScratch

	Optimized bool
}

// FrameWorkPostFilterRequest carries caller-owned scratch and side data for the
// supported frame-level postfilter stages.
type FrameWorkPostFilterRequest struct {
	LoopFilter  FrameWorkLoopFilterPostFilterRequest
	CDEF        FrameWorkCDEFPostFilterRequest
	SuperRes    FrameWorkSuperResPostFilterRequest
	Restoration FrameWorkRestorationPostFilterRequest
	FilmGrain   FrameWorkFilmGrainPostFilterRequest
}

// FrameWorkPostFilterResult summarizes supported frame-level postfilter work.
type FrameWorkPostFilterResult struct {
	Completed   FrameWorkPostFilterStage
	LoopFilter  FrameWorkLoopFilterPostFilterApplyResult
	CDEF        FrameWorkCDEFPostFilterResult
	Restoration tile.RestorationFrameApplyResult
	FilmGrain   FrameWorkFilmGrainPostFilterResult
}

// FrameWorkCallerPostFilterResult summarizes a caller-owned full postfilter
// chain. It embeds the publishable-stage result and adds detached superres
// output metadata.
type FrameWorkCallerPostFilterResult struct {
	FrameWorkPostFilterResult
	SuperRes FrameWorkSuperResPostFilterResult
}

// FrameWorkSupportedPostFilterRunner adapts ApplySupportedPostFilters to the
// FrameWorkPostFilterFunc callback shape.
type FrameWorkSupportedPostFilterRunner struct {
	Request FrameWorkPostFilterRequest
	Context FrameWorkPostFilterContext
	Result  FrameWorkPostFilterResult
}

// FrameWorkCallerPostFilterRunner adapts ApplyCallerPostFilters to the
// FrameWorkPostFilterFunc callback shape for display/output consumers that can
// retain caller-owned detached output.
type FrameWorkCallerPostFilterRunner struct {
	Request FrameWorkPostFilterRequest
	Context FrameWorkPostFilterContext
	Result  FrameWorkCallerPostFilterResult
}

// ScratchLen reports caller-owned scratch needed by Apply for the runner's
// request.
func (r *FrameWorkSupportedPostFilterRunner) ScratchLen(ctx FrameWorkPostFilterContext) (FrameWorkPostFilterScratchSize, error) {
	if r == nil {
		return FrameWorkPostFilterScratchSize{}, ErrInvalidFrameWorkState
	}
	return ctx.SupportedPostFilterScratchLen(r.Request)
}

// Apply runs supported postfilters and rejects any remaining active stage.
func (r *FrameWorkSupportedPostFilterRunner) Apply(ctx FrameWorkPostFilterContext) error {
	if r == nil {
		return ErrInvalidFrameWorkState
	}
	next, result, err := ctx.ApplySupportedPostFilters(r.Request)
	if err != nil {
		return err
	}
	if err := next.RequirePublishablePostFilterOutput(); err != nil {
		return err
	}
	r.Context = next
	r.Result = result
	return nil
}

// ScratchLen reports caller-owned scratch needed by Apply for the runner's
// request.
func (r *FrameWorkCallerPostFilterRunner) ScratchLen(ctx FrameWorkPostFilterContext) (FrameWorkPostFilterScratchSize, error) {
	if r == nil {
		return FrameWorkPostFilterScratchSize{}, ErrInvalidFrameWorkState
	}
	return ctx.CallerPostFilterScratchLen(r.Request)
}

// Apply runs the full caller-owned postfilter chain and allows detached output.
func (r *FrameWorkCallerPostFilterRunner) Apply(ctx FrameWorkPostFilterContext) error {
	if r == nil {
		return ErrInvalidFrameWorkState
	}
	next, result, err := ctx.ApplyCallerPostFilters(r.Request)
	if err != nil {
		return err
	}
	if err := next.RequireNoRemainingPostFilters(); err != nil {
		return err
	}
	r.Context = next
	r.Result = result
	return nil
}

// ActivePostFilters returns the frame-level postfilter stages signaled by this
// final-frame context.
func (ctx FrameWorkPostFilterContext) ActivePostFilters() FrameWorkPostFilterStage {
	var stages FrameWorkPostFilterStage
	if frameWorkLoopFilterActive(ctx.Event.LoopFilter) {
		stages |= FrameWorkPostFilterLoopFilter
	}
	if frameWorkCDEFActive(ctx.Event.CDEF) {
		stages |= FrameWorkPostFilterCDEF
	}
	if ctx.Event.FrameSize.SuperResEnabled {
		stages |= FrameWorkPostFilterSuperRes
	}
	if frameWorkRestorationActive(ctx.Event.Restoration) {
		stages |= FrameWorkPostFilterLoopRestoration
	}
	if ctx.Event.FilmGrain.Apply {
		stages |= FrameWorkPostFilterFilmGrain
	}
	return stages
}

// WithCompletedPostFilters returns a callback-local view of ctx with stages
// marked complete. It does not change ActivePostFilters, which remains the syntax
// capability gate for callers that cannot run postfilters.
func (ctx FrameWorkPostFilterContext) WithCompletedPostFilters(stages FrameWorkPostFilterStage) FrameWorkPostFilterContext {
	ctx.completedPostFilters |= stages & ctx.ActivePostFilters()
	return ctx
}

// RemainingPostFilters returns active stages that have not been marked complete
// in this callback-local context.
func (ctx FrameWorkPostFilterContext) RemainingPostFilters() FrameWorkPostFilterStage {
	return ctx.ActivePostFilters() &^ ctx.completedPostFilters
}

// SupportedPostFilters returns the remaining active stages that the supported
// frame-pool publication pipeline can run.
func (ctx FrameWorkPostFilterContext) SupportedPostFilters() (FrameWorkPostFilterStage, error) {
	remaining := ctx.RemainingPostFilters()
	supported, err := ctx.supportedPostFilterStages(remaining)
	if err != nil {
		return 0, err
	}
	return remaining & supported, nil
}

// UnsupportedPostFilters returns the remaining active stages that require the
// caller-owned full postfilter path or are not currently integrated.
func (ctx FrameWorkPostFilterContext) UnsupportedPostFilters() (FrameWorkPostFilterStage, error) {
	remaining := ctx.RemainingPostFilters()
	supported, err := ctx.supportedPostFilterStages(remaining)
	if err != nil {
		return 0, err
	}
	return remaining &^ supported, nil
}

// RequireNoActivePostFilters is a capability gate for callers that consume the
// current reconstructed frame directly. It accepts frames whose postfilter plan
// is a no-op and rejects frames that need loop filter, CDEF, superres, loop
// restoration, or film grain before MD5/reference use.
func (ctx FrameWorkPostFilterContext) RequireNoActivePostFilters() error {
	if !ctx.ActivePostFilters().Empty() {
		return ErrUnsupportedPostFilter
	}
	return nil
}

// RequireSupportedPostFilters rejects contexts with remaining stages that the
// frame-pool publication pipeline cannot complete.
func (ctx FrameWorkPostFilterContext) RequireSupportedPostFilters() error {
	unsupported, err := ctx.UnsupportedPostFilters()
	if err != nil {
		return err
	}
	if !unsupported.Empty() {
		return ErrUnsupportedPostFilter
	}
	return nil
}

// RequireNoRemainingPostFilters rejects active postfilters that have not been
// marked complete in this callback-local context.
func (ctx FrameWorkPostFilterContext) RequireNoRemainingPostFilters() error {
	if !ctx.RemainingPostFilters().Empty() {
		return ErrUnsupportedPostFilter
	}
	return nil
}

// DetachedPostFilterOutput reports whether ctx.Output points at caller-owned
// postfilter scratch instead of the frame-pool surface that the framework will
// publish to reference slots.
func (ctx FrameWorkPostFilterContext) DetachedPostFilterOutput() bool {
	return ctx.detachedPostFilterOutput
}

// RequirePublishablePostFilterOutput rejects contexts whose filters are done
// but whose current output is detached caller-owned scratch. Use this guard from
// frame-pool publication hooks; display/caller-owned consumers can instead use
// RequireNoRemainingPostFilters and then read ctx.Output directly.
func (ctx FrameWorkPostFilterContext) RequirePublishablePostFilterOutput() error {
	if err := ctx.RequireNoRemainingPostFilters(); err != nil {
		return err
	}
	if ctx.detachedPostFilterOutput {
		return ErrUnsupportedPostFilter
	}
	return nil
}

// SupportedPostFilterScratchLen reports scratch lengths needed by
// ApplySupportedPostFilters. It rejects frames requiring unsupported stages
// before callers allocate scratch for a pipeline that cannot complete.
func (ctx FrameWorkPostFilterContext) SupportedPostFilterScratchLen(req FrameWorkPostFilterRequest) (FrameWorkPostFilterScratchSize, error) {
	remaining := ctx.RemainingPostFilters()
	supported, err := ctx.supportedPostFilterStages(remaining)
	if err != nil {
		return FrameWorkPostFilterScratchSize{}, err
	}
	if remaining&^supported != 0 {
		return FrameWorkPostFilterScratchSize{}, ErrUnsupportedPostFilter
	}
	var size FrameWorkPostFilterScratchSize
	if remaining.Has(FrameWorkPostFilterLoopFilter) {
		loopFilterSize, err := ctx.LoopFilterPostFilterScratchLen(req.LoopFilter)
		if err != nil {
			return FrameWorkPostFilterScratchSize{}, err
		}
		size.LoopFilter = loopFilterSize
	}
	if remaining.Has(FrameWorkPostFilterCDEF) {
		cdefSize, err := ctx.CDEFPostFilterScratchLen()
		if err != nil {
			return FrameWorkPostFilterScratchSize{}, err
		}
		size.CDEF = cdefSize
	}
	if remaining.Has(FrameWorkPostFilterLoopRestoration) {
		records := req.Restoration.Records
		if frameWorkRestorationRecordsEmpty(records) && ctx.RestorationFrameBuffers != nil {
			records = ctx.RestorationFrameBuffers.Records
		}
		restorationSize, err := ctx.LoopRestorationPostFilterScratchLen(records, req.Restoration.Optimized)
		if err != nil {
			return FrameWorkPostFilterScratchSize{}, err
		}
		size.Restoration = restorationSize
	}
	if remaining.Has(FrameWorkPostFilterFilmGrain) {
		filmGrainSize, err := ctx.FilmGrainPostFilterScratchLen()
		if err != nil {
			return FrameWorkPostFilterScratchSize{}, err
		}
		size.FilmGrain = filmGrainSize
	}
	return size, nil
}

// PreSuperResPostFilterScratchLen reports scratch for the coded-surface
// postfilter stages that AV1 runs before superres. It intentionally ignores
// superres and later stages so callers can allocate and run this prefix before
// switching to caller-owned upscaled output scratch.
func (ctx FrameWorkPostFilterContext) PreSuperResPostFilterScratchLen(req FrameWorkPostFilterRequest) (FrameWorkPostFilterScratchSize, error) {
	remaining := ctx.RemainingPostFilters()
	var size FrameWorkPostFilterScratchSize
	if remaining.Has(FrameWorkPostFilterLoopFilter) {
		loopFilterSize, err := ctx.LoopFilterPostFilterScratchLen(req.LoopFilter)
		if err != nil {
			return FrameWorkPostFilterScratchSize{}, err
		}
		size.LoopFilter = loopFilterSize
	}
	if remaining.Has(FrameWorkPostFilterCDEF) {
		cdefSize, err := ctx.CDEFPostFilterScratchLen()
		if err != nil {
			return FrameWorkPostFilterScratchSize{}, err
		}
		size.CDEF = cdefSize
	}
	return size, nil
}

// CallerPostFilterScratchLen reports scratch for a caller-owned full postfilter
// chain. When superres is active, callers can use the returned SuperRes sizes to
// allocate req.SuperRes first; a second call with req.SuperRes.OutputFrame sized
// will also report post-superres restoration and film-grain scratch.
func (ctx FrameWorkPostFilterContext) CallerPostFilterScratchLen(req FrameWorkPostFilterRequest) (FrameWorkPostFilterScratchSize, error) {
	size, err := ctx.PreSuperResPostFilterScratchLen(req)
	if err != nil {
		return FrameWorkPostFilterScratchSize{}, err
	}

	tailCtx := ctx.WithCompletedPostFilters(FrameWorkPostFilterLoopFilter | FrameWorkPostFilterCDEF)
	if tailCtx.RemainingPostFilters().Has(FrameWorkPostFilterSuperRes) {
		superResSize, err := tailCtx.SuperResPostFilterScratchLen()
		if err != nil {
			return FrameWorkPostFilterScratchSize{}, err
		}
		size.SuperRes = superResSize
		if len(req.SuperRes.OutputFrame) < superResSize.OutputFrame {
			return size, nil
		}
		tailCtx, err = tailCtx.bindSuperResPostFilterOutputContext(req.SuperRes)
		if err != nil {
			return FrameWorkPostFilterScratchSize{}, err
		}
	}
	if tailCtx.RemainingPostFilters().Has(FrameWorkPostFilterLoopRestoration) {
		records := req.Restoration.Records
		if frameWorkRestorationRecordsEmpty(records) && tailCtx.RestorationFrameBuffers != nil {
			records = tailCtx.RestorationFrameBuffers.Records
		}
		restorationSize, err := tailCtx.LoopRestorationPostFilterScratchLen(records, req.Restoration.Optimized)
		if err != nil {
			return FrameWorkPostFilterScratchSize{}, err
		}
		size.Restoration = restorationSize
	}
	if tailCtx.RemainingPostFilters().Has(FrameWorkPostFilterFilmGrain) {
		filmGrainSize, err := tailCtx.FilmGrainPostFilterScratchLen()
		if err != nil {
			return FrameWorkPostFilterScratchSize{}, err
		}
		size.FilmGrain = filmGrainSize
	}
	return size, nil
}

// ApplySupportedPostFilters runs the currently integrated postfilter stages in
// normal AV1 order. It rejects frames requiring unsupported stages before
// mutating ctx.Output.
func (ctx FrameWorkPostFilterContext) ApplySupportedPostFilters(req FrameWorkPostFilterRequest) (FrameWorkPostFilterContext, FrameWorkPostFilterResult, error) {
	var result FrameWorkPostFilterResult
	remaining := ctx.RemainingPostFilters()
	supported, err := ctx.supportedPostFilterStages(remaining)
	if err != nil {
		return ctx, result, err
	}
	if remaining&^supported != 0 {
		return ctx, result, ErrUnsupportedPostFilter
	}
	if remaining.Has(FrameWorkPostFilterFilmGrain) {
		if err := ctx.validateFilmGrainPostFilterRequest(req.FilmGrain); err != nil {
			return ctx, result, err
		}
	}
	if remaining.Has(FrameWorkPostFilterCDEF) {
		if err := ctx.validateCDEFPostFilterRequest(req.CDEF); err != nil {
			return ctx, result, err
		}
	}
	if remaining.Has(FrameWorkPostFilterLoopRestoration) {
		if err := ctx.validateLoopRestorationPostFilterRequest(req.Restoration); err != nil {
			return ctx, result, err
		}
	}
	if remaining.Has(FrameWorkPostFilterLoopFilter) {
		loopFilterResult, err := ctx.ApplyLoopFilterEdges(req.LoopFilter)
		if err != nil {
			return ctx, result, err
		}
		ctx = ctx.WithCompletedPostFilters(FrameWorkPostFilterLoopFilter)
		result.Completed |= FrameWorkPostFilterLoopFilter
		result.LoopFilter = loopFilterResult
	}
	if remaining.Has(FrameWorkPostFilterCDEF) {
		cdefResult, err := ctx.ApplyCDEFPostFilter(req.CDEF)
		if err != nil {
			return ctx, result, err
		}
		ctx = ctx.WithCompletedPostFilters(FrameWorkPostFilterCDEF)
		result.Completed |= FrameWorkPostFilterCDEF
		result.CDEF = cdefResult
	}
	if ctx.RemainingPostFilters().Has(FrameWorkPostFilterLoopRestoration) {
		restorationResult, err := ctx.ApplyLoopRestorationPostFilter(req.Restoration)
		if err != nil {
			return ctx, result, err
		}
		ctx = ctx.WithCompletedPostFilters(FrameWorkPostFilterLoopRestoration)
		result.Completed |= FrameWorkPostFilterLoopRestoration
		result.Restoration = restorationResult
	}
	if ctx.RemainingPostFilters().Has(FrameWorkPostFilterFilmGrain) {
		filmGrainResult, err := ctx.ApplyFilmGrainPostFilter(req.FilmGrain)
		if err != nil {
			return ctx, result, err
		}
		ctx = ctx.WithCompletedPostFilters(FrameWorkPostFilterFilmGrain)
		result.Completed |= FrameWorkPostFilterFilmGrain
		result.FilmGrain = filmGrainResult
	}
	return ctx, result, nil
}

// ApplyCallerPostFilters runs all active postfilters in AV1 order. Unlike the
// supported frame-pool publication runner, this path may switch ctx.Output to
// caller-owned superres scratch and is therefore intended for display/output
// consumers that can keep that detached frame alive.
func (ctx FrameWorkPostFilterContext) ApplyCallerPostFilters(req FrameWorkPostFilterRequest) (FrameWorkPostFilterContext, FrameWorkCallerPostFilterResult, error) {
	var result FrameWorkCallerPostFilterResult
	remaining := ctx.RemainingPostFilters()
	if remaining.Has(FrameWorkPostFilterCDEF) {
		if err := ctx.validateCDEFPostFilterRequest(req.CDEF); err != nil {
			return ctx, result, err
		}
	}
	validateTailCtx := ctx.WithCompletedPostFilters(FrameWorkPostFilterLoopFilter | FrameWorkPostFilterCDEF)
	if validateTailCtx.RemainingPostFilters().Has(FrameWorkPostFilterSuperRes) {
		if err := validateTailCtx.validateSuperResPostFilterRequest(req.SuperRes); err != nil {
			return ctx, result, err
		}
		var err error
		validateTailCtx, err = validateTailCtx.bindSuperResPostFilterOutputContext(req.SuperRes)
		if err != nil {
			return ctx, result, err
		}
	}
	if validateTailCtx.RemainingPostFilters().Has(FrameWorkPostFilterLoopRestoration) {
		if err := validateTailCtx.validateLoopRestorationPostFilterRequest(req.Restoration); err != nil {
			return ctx, result, err
		}
	}
	if validateTailCtx.RemainingPostFilters().Has(FrameWorkPostFilterFilmGrain) {
		if err := validateTailCtx.validateFilmGrainPostFilterRequest(req.FilmGrain); err != nil {
			return ctx, result, err
		}
	}

	next, prefixResult, err := ctx.ApplyPreSuperResPostFilters(req)
	if err != nil {
		return ctx, result, err
	}
	result.Completed |= prefixResult.Completed
	result.LoopFilter = prefixResult.LoopFilter
	result.CDEF = prefixResult.CDEF

	if next.RemainingPostFilters().Has(FrameWorkPostFilterSuperRes) {
		superResNext, superResResult, err := next.ApplySuperResPostFilterToContext(req.SuperRes)
		if err != nil {
			return ctx, result, err
		}
		next = superResNext
		result.Completed |= FrameWorkPostFilterSuperRes
		result.SuperRes = superResResult
	}

	next, tailResult, err := next.ApplySupportedPostFilters(req)
	if err != nil {
		return ctx, result, err
	}
	result.Completed |= tailResult.Completed
	result.Restoration = tailResult.Restoration
	result.FilmGrain = tailResult.FilmGrain
	return next, result, nil
}

// ApplyPreSuperResPostFilters runs the coded-surface postfilter prefix in AV1
// order: loop filter, then CDEF. It leaves superres, loop restoration, and film
// grain remaining for caller-owned follow-up stages.
func (ctx FrameWorkPostFilterContext) ApplyPreSuperResPostFilters(req FrameWorkPostFilterRequest) (FrameWorkPostFilterContext, FrameWorkPostFilterResult, error) {
	var result FrameWorkPostFilterResult
	remaining := ctx.RemainingPostFilters()
	if remaining.Has(FrameWorkPostFilterCDEF) {
		if err := ctx.validateCDEFPostFilterRequest(req.CDEF); err != nil {
			return ctx, result, err
		}
	}
	if remaining.Has(FrameWorkPostFilterLoopFilter) {
		loopFilterResult, err := ctx.ApplyLoopFilterEdges(req.LoopFilter)
		if err != nil {
			return ctx, result, err
		}
		ctx = ctx.WithCompletedPostFilters(FrameWorkPostFilterLoopFilter)
		result.Completed |= FrameWorkPostFilterLoopFilter
		result.LoopFilter = loopFilterResult
	}
	if ctx.RemainingPostFilters().Has(FrameWorkPostFilterCDEF) {
		cdefResult, err := ctx.ApplyCDEFPostFilter(req.CDEF)
		if err != nil {
			return ctx, result, err
		}
		ctx = ctx.WithCompletedPostFilters(FrameWorkPostFilterCDEF)
		result.Completed |= FrameWorkPostFilterCDEF
		result.CDEF = cdefResult
	}
	return ctx, result, nil
}

func (ctx FrameWorkPostFilterContext) supportedPostFilterStages(remaining FrameWorkPostFilterStage) (FrameWorkPostFilterStage, error) {
	supported := FrameWorkPostFilterLoopFilter | FrameWorkPostFilterCDEF | FrameWorkPostFilterLoopRestoration
	if remaining&^(supported|FrameWorkPostFilterFilmGrain) != 0 {
		return supported, nil
	}
	if !remaining.Has(FrameWorkPostFilterFilmGrain) {
		return supported, nil
	}
	ok, err := ctx.filmGrainPostFilterSupported()
	if err != nil {
		return 0, err
	}
	if ok {
		supported |= FrameWorkPostFilterFilmGrain
	}
	return supported, nil
}

func frameWorkRestorationFrameSampleScratchSizeMax(a tile.RestorationFrameSampleScratchSize, b tile.RestorationFrameSampleScratchSize) tile.RestorationFrameSampleScratchSize {
	result := tile.RestorationFrameSampleScratchSize{
		DataLen: maxInt(a.DataLen, b.DataLen),
		DstLen:  maxInt(a.DstLen, b.DstLen),
	}
	for plane := 0; plane < len(result.Data); plane++ {
		result.Data[plane] = frameWorkBorderedSamplePlaneLayoutMax(a.Data[plane], b.Data[plane])
		result.Dst[plane] = frameWorkBorderedSamplePlaneLayoutMax(a.Dst[plane], b.Dst[plane])
	}
	return result
}

func frameWorkBorderedSamplePlaneLayoutMax(a frame.BorderedSamplePlaneLayout, b frame.BorderedSamplePlaneLayout) frame.BorderedSamplePlaneLayout {
	return frame.BorderedSamplePlaneLayout{
		Stride: maxInt(a.Stride, b.Stride),
		Origin: maxInt(a.Origin, b.Origin),
		Rows:   maxInt(a.Rows, b.Rows),
		Len:    maxInt(a.Len, b.Len),
	}
}

func frameWorkRestorationUnitRecordBoundaryScratchSizeMax(a tile.RestorationUnitRecordBoundaryScratchSize, b tile.RestorationUnitRecordBoundaryScratchSize) tile.RestorationUnitRecordBoundaryScratchSize {
	return tile.RestorationUnitRecordBoundaryScratchSize{
		Unit: tile.RestorationUnitScratchSize{
			Wiener:  maxInt(a.Unit.Wiener, b.Unit.Wiener),
			SGRProj: maxInt(a.Unit.SGRProj, b.Unit.SGRProj),
		},
		Boundary: tile.RestorationStripeBoundaryScratchSize{
			Above: maxInt(a.Boundary.Above, b.Boundary.Above),
			Below: maxInt(a.Boundary.Below, b.Boundary.Below),
		},
	}
}

// LoopRestorationPostFilterPlan returns the frame-level loop-restoration plan
// for the current event.
func (ctx FrameWorkPostFilterContext) LoopRestorationPostFilterPlan() (tile.RestorationFramePlan, error) {
	return tile.BuildRestorationFramePlan(ctx.Event.Restoration, ctx.Event.FrameSize, ctx.Event.SequenceHeader.ColorConfig)
}

// LoopRestorationPostFilterScratchLen reports scratch lengths needed to apply
// decoded loop-restoration records to ctx.Output.
func (ctx FrameWorkPostFilterContext) LoopRestorationPostFilterScratchLen(records [3][]tile.RestorationUnitRecord, optimized bool) (FrameWorkRestorationPostFilterScratchSize, error) {
	plan, err := ctx.LoopRestorationPostFilterPlan()
	if err != nil {
		return FrameWorkRestorationPostFilterScratchSize{}, err
	}
	if !plan.Active {
		return FrameWorkRestorationPostFilterScratchSize{}, nil
	}
	if ctx.Output == nil {
		return FrameWorkRestorationPostFilterScratchSize{}, frame.ErrInvalidSlot
	}
	samples, err := tile.RestorationFrameSampleScratchLen(plan, *ctx.Output)
	if err != nil {
		return FrameWorkRestorationPostFilterScratchSize{}, err
	}
	var apply tile.RestorationUnitRecordBoundaryScratchSize
	for plane := 0; plane < int(plan.Planes); plane++ {
		planeSize, err := tile.RestorationFramePlaneScratchLen(plan.Grids[plane], records[plane], optimized)
		if err != nil {
			return FrameWorkRestorationPostFilterScratchSize{}, err
		}
		if planeSize.Unit.Wiener > apply.Unit.Wiener {
			apply.Unit.Wiener = planeSize.Unit.Wiener
		}
		if planeSize.Unit.SGRProj > apply.Unit.SGRProj {
			apply.Unit.SGRProj = planeSize.Unit.SGRProj
		}
		if planeSize.Boundary.Above > apply.Boundary.Above {
			apply.Boundary.Above = planeSize.Boundary.Above
		}
		if planeSize.Boundary.Below > apply.Boundary.Below {
			apply.Boundary.Below = planeSize.Boundary.Below
		}
	}
	return FrameWorkRestorationPostFilterScratchSize{
		Samples: samples,
		Apply:   apply,
	}, nil
}

// ApplyLoopRestorationPostFilter applies loop restoration to ctx.Output. Earlier
// postfilter stages that feed restoration must already be inactive or marked
// complete with WithCompletedPostFilters.
func (ctx FrameWorkPostFilterContext) ApplyLoopRestorationPostFilter(req FrameWorkRestorationPostFilterRequest) (tile.RestorationFrameApplyResult, error) {
	remaining := ctx.RemainingPostFilters()
	preRestoration := FrameWorkPostFilterLoopFilter | FrameWorkPostFilterCDEF | FrameWorkPostFilterSuperRes
	if remaining&preRestoration != 0 {
		return tile.RestorationFrameApplyResult{}, ErrUnsupportedPostFilter
	}
	if !remaining.Has(FrameWorkPostFilterLoopRestoration) {
		return tile.RestorationFrameApplyResult{}, nil
	}
	if ctx.Output == nil {
		return tile.RestorationFrameApplyResult{}, frame.ErrInvalidSlot
	}
	plan, err := ctx.LoopRestorationPostFilterPlan()
	if err != nil {
		return tile.RestorationFrameApplyResult{}, err
	}
	if ctx.RestorationFrameBuffers != nil {
		if frameWorkRestorationRecordsEmpty(req.Records) {
			req.Records = ctx.RestorationFrameBuffers.Records
		}
		if frameWorkRestorationBoundariesEmpty(req.Boundaries) {
			req.Boundaries = ctx.RestorationFrameBuffers.Boundaries
		}
	}
	if err := ctx.validateLoopRestorationPostFilterRequest(req); err != nil {
		return tile.RestorationFrameApplyResult{}, err
	}
	return tile.ApplyRestorationFrameToFrame(plan, *ctx.Output, req.Records, req.Boundaries, req.DataScratch, req.DstScratch, req.Scratch, req.Optimized)
}

func (ctx FrameWorkPostFilterContext) validateLoopRestorationPostFilterRequest(req FrameWorkRestorationPostFilterRequest) error {
	if !ctx.RemainingPostFilters().Has(FrameWorkPostFilterLoopRestoration) {
		return nil
	}
	if ctx.Output == nil {
		return frame.ErrInvalidSlot
	}
	records := req.Records
	if frameWorkRestorationRecordsEmpty(records) && ctx.RestorationFrameBuffers != nil {
		records = ctx.RestorationFrameBuffers.Records
	}
	size, err := ctx.LoopRestorationPostFilterScratchLen(records, req.Optimized)
	if err != nil {
		return err
	}
	if len(req.DataScratch) < size.Samples.DataLen || len(req.DstScratch) < size.Samples.DstLen {
		return tile.ErrJobBufferTooSmall
	}
	if len(req.DataScratch) != size.Samples.DataLen || len(req.DstScratch) != size.Samples.DstLen {
		return tile.ErrInvalidPlan
	}
	if len(req.Scratch.Unit.Wiener) < size.Apply.Unit.Wiener ||
		len(req.Scratch.Unit.SGRProj) < size.Apply.Unit.SGRProj ||
		len(req.Scratch.Boundary.Above) < size.Apply.Boundary.Above ||
		len(req.Scratch.Boundary.Below) < size.Apply.Boundary.Below {
		return tile.ErrInvalidPlan
	}
	return nil
}

func frameWorkRestorationRecordsEmpty(records [3][]tile.RestorationUnitRecord) bool {
	for plane := range records {
		if len(records[plane]) != 0 {
			return false
		}
	}
	return true
}

func frameWorkRestorationBoundariesEmpty(boundaries [3]tile.RestorationStripeBoundaries) bool {
	for plane := range boundaries {
		if len(boundaries[plane].Above) != 0 || len(boundaries[plane].Below) != 0 || boundaries[plane].Stride != 0 {
			return false
		}
	}
	return true
}
