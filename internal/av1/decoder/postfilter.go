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
	CDEF        FrameWorkCDEFPostFilterRequest
	Restoration FrameWorkRestorationPostFilterRequest
}

// FrameWorkPostFilterResult summarizes supported frame-level postfilter work.
type FrameWorkPostFilterResult struct {
	Completed   FrameWorkPostFilterStage
	CDEF        FrameWorkCDEFPostFilterResult
	Restoration tile.RestorationFrameApplyResult
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

// RequireNoRemainingPostFilters rejects active postfilters that have not been
// marked complete in this callback-local context.
func (ctx FrameWorkPostFilterContext) RequireNoRemainingPostFilters() error {
	if !ctx.RemainingPostFilters().Empty() {
		return ErrUnsupportedPostFilter
	}
	return nil
}

// ApplySupportedPostFilters runs the currently integrated postfilter stages in
// normal AV1 order. It rejects frames requiring unsupported stages before
// mutating ctx.Output.
func (ctx FrameWorkPostFilterContext) ApplySupportedPostFilters(req FrameWorkPostFilterRequest) (FrameWorkPostFilterContext, FrameWorkPostFilterResult, error) {
	var result FrameWorkPostFilterResult
	remaining := ctx.RemainingPostFilters()
	supported := FrameWorkPostFilterCDEF | FrameWorkPostFilterLoopRestoration
	if remaining&^supported != 0 {
		return ctx, result, ErrUnsupportedPostFilter
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
	return ctx, result, nil
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
	return tile.ApplyRestorationFrameToFrame(plan, *ctx.Output, req.Records, req.Boundaries, req.DataScratch, req.DstScratch, req.Scratch, req.Optimized)
}
