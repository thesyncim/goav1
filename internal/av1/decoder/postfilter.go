package decoder

import (
	"github.com/thesyncim/goav1/internal/av1/cdef"
	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/parser"
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

// FrameWorkPostFilterBindOptions carries decoded side data that is not part of
// the reusable scratch arena.
type FrameWorkPostFilterBindOptions struct {
	LoopFilterMap             FrameWorkLoopFilterMap
	LoopFilterTrustedCoverage bool
	CDEFIndexMap              FrameWorkCDEFIndexMap

	SuperResOutputView *frame.Frame

	FilmGrainOutputView *frame.Frame

	RestorationRecords    [3][]tile.RestorationUnitRecord
	RestorationBoundaries [3]tile.RestorationStripeBoundaries
	RestorationOptimized  bool
}

// FrameWorkPostFilterScratch carries caller-owned backing storage for the full
// postfilter request. Callers can reuse one instance across frames after sizing
// it with FrameWorkPostFilterScratchSize.Max.
type FrameWorkPostFilterScratch struct {
	LoopFilterEdges    []FrameWorkLoopFilterPostFilterEdge
	LoopFilterSchedule []uint32

	CDEFSamples       [3][]uint16
	CDEFDst           [3][]uint16
	CDEFDirectionGrid []cdef.DirectionGrid
	CDEFVarianceGrid  []cdef.VarianceGrid
	CDEFInput         []uint16
	CDEFUnitDst       []uint16

	SuperResOutputFrame []byte
	SuperResCoded       [3][]uint16
	SuperResOutput      [3][]uint16

	RestorationData   []uint16
	RestorationDst    []uint16
	RestorationWiener []uint16
	RestorationSGR    []int32
	RestorationAbove  []uint16
	RestorationBelow  []uint16

	FilmGrainOutputFrame   []byte
	FilmGrainLumaGrain     []int16
	FilmGrainChromaGrain   [2][]int16
	FilmGrainLumaSamples   []uint16
	FilmGrainChromaSamples [2][]uint16
}

// BindRequest validates and slices caller-owned storage for the full
// postfilter request surface.
func (s FrameWorkPostFilterScratchSize) BindRequest(options FrameWorkPostFilterBindOptions, scratch FrameWorkPostFilterScratch) (FrameWorkPostFilterRequest, error) {
	edges, err := s.LoopFilter.BindEdges(scratch.LoopFilterEdges)
	if err != nil {
		return FrameWorkPostFilterRequest{}, err
	}
	schedule, err := s.LoopFilter.BindSchedule(scratch.LoopFilterSchedule)
	if err != nil {
		return FrameWorkPostFilterRequest{}, err
	}
	cdefReq, err := s.CDEF.BindRequest(options.CDEFIndexMap, scratch.CDEFSamples, scratch.CDEFDst, scratch.CDEFDirectionGrid, scratch.CDEFVarianceGrid, scratch.CDEFInput, scratch.CDEFUnitDst)
	if err != nil {
		return FrameWorkPostFilterRequest{}, err
	}
	superResReq, err := s.SuperRes.BindRequest(scratch.SuperResOutputFrame, scratch.SuperResCoded, scratch.SuperResOutput, options.SuperResOutputView)
	if err != nil {
		return FrameWorkPostFilterRequest{}, err
	}
	restorationReq, err := s.Restoration.BindRequest(options.RestorationRecords, options.RestorationBoundaries, scratch.RestorationData, scratch.RestorationDst, scratch.RestorationWiener, scratch.RestorationSGR, scratch.RestorationAbove, scratch.RestorationBelow, options.RestorationOptimized)
	if err != nil {
		return FrameWorkPostFilterRequest{}, err
	}
	filmGrainReq, err := s.FilmGrain.BindRequest(scratch.FilmGrainOutputFrame, options.FilmGrainOutputView, scratch.FilmGrainLumaGrain, scratch.FilmGrainChromaGrain, scratch.FilmGrainLumaSamples, scratch.FilmGrainChromaSamples)
	if err != nil {
		return FrameWorkPostFilterRequest{}, err
	}
	return FrameWorkPostFilterRequest{
		LoopFilter: FrameWorkLoopFilterPostFilterRequest{
			Map:             options.LoopFilterMap,
			Edges:           edges,
			Schedule:        schedule,
			TrustedCoverage: options.LoopFilterTrustedCoverage,
		},
		CDEF:        cdefReq,
		SuperRes:    superResReq,
		Restoration: restorationReq,
		FilmGrain:   filmGrainReq,
	}, nil
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

// BindRequest validates and slices caller-owned scratch for loop restoration.
func (s FrameWorkRestorationPostFilterScratchSize) BindRequest(records [3][]tile.RestorationUnitRecord, boundaries [3]tile.RestorationStripeBoundaries, dataScratch []uint16, dstScratch []uint16, wienerScratch []uint16, sgrprojScratch []int32, aboveScratch []uint16, belowScratch []uint16, optimized bool) (FrameWorkRestorationPostFilterRequest, error) {
	if s.Samples.DataLen < 0 || s.Samples.DstLen < 0 ||
		s.Apply.Unit.Wiener < 0 || s.Apply.Unit.SGRProj < 0 ||
		s.Apply.Boundary.Above < 0 || s.Apply.Boundary.Below < 0 {
		return FrameWorkRestorationPostFilterRequest{}, tile.ErrInvalidPlan
	}
	if len(dataScratch) < s.Samples.DataLen || len(dstScratch) < s.Samples.DstLen {
		return FrameWorkRestorationPostFilterRequest{}, tile.ErrJobBufferTooSmall
	}
	if len(wienerScratch) < s.Apply.Unit.Wiener || len(sgrprojScratch) < s.Apply.Unit.SGRProj ||
		len(aboveScratch) < s.Apply.Boundary.Above || len(belowScratch) < s.Apply.Boundary.Below {
		return FrameWorkRestorationPostFilterRequest{}, tile.ErrInvalidPlan
	}
	return FrameWorkRestorationPostFilterRequest{
		Records:     records,
		Boundaries:  boundaries,
		DataScratch: dataScratch[:s.Samples.DataLen],
		DstScratch:  dstScratch[:s.Samples.DstLen],
		Scratch: tile.RestorationUnitRecordBoundaryScratch{
			Unit: tile.RestorationUnitScratch{
				Wiener:  wienerScratch[:s.Apply.Unit.Wiener],
				SGRProj: sgrprojScratch[:s.Apply.Unit.SGRProj],
			},
			Boundary: tile.RestorationStripeBoundaryScratch{
				Above: aboveScratch[:s.Apply.Boundary.Above],
				Below: belowScratch[:s.Apply.Boundary.Below],
			},
		},
		Optimized: optimized,
	}, nil
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

// FrameWorkPostFilterBanding selects row-band sizes for the supported
// postfilter prefix. LoopFilterMIRows is expressed in 4x4 MI rows; CDEFUnitRows
// is expressed in 64x64 CDEF unit rows. A zero field keeps the whole-frame
// path for that stage.
type FrameWorkPostFilterBanding struct {
	LoopFilterMIRows int
	CDEFUnitRows     int
}

func frameWorkDav1dPostFilterBanding(ctx FrameWorkPostFilterContext) FrameWorkPostFilterBanding {
	if ctx.Event.SequenceHeader.Use128x128Superblock {
		return FrameWorkPostFilterBanding{
			LoopFilterMIRows: 32,
			CDEFUnitRows:     2,
		}
	}
	return FrameWorkPostFilterBanding{
		LoopFilterMIRows: 16,
		CDEFUnitRows:     1,
	}
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

// FrameWorkBoundSupportedPostFilterRunner adapts ApplySupportedPostFilters for
// event runners that only know exact scratch sizing after tile residuals have
// filled frame-level side data.
type FrameWorkBoundSupportedPostFilterRunner struct {
	Options FrameWorkPostFilterBindOptions
	Scratch FrameWorkPostFilterScratch

	Size    FrameWorkPostFilterScratchSize
	Request FrameWorkPostFilterRequest
	Context FrameWorkPostFilterContext
	Result  FrameWorkPostFilterResult

	DisplayOutput *frame.Frame
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

// Apply binds the runner's caller-owned scratch against the completed side-data
// context, runs supported postfilters, and rejects detached output.
func (r *FrameWorkBoundSupportedPostFilterRunner) Apply(ctx FrameWorkPostFilterContext) error {
	if r == nil {
		return ErrInvalidFrameWorkState
	}
	options := frameWorkPostFilterBindOptionsFromContext(ctx, r.Options)
	probe := FrameWorkPostFilterRequest{
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
	size, err := ctx.boundSupportedPostFilterScratchLen(probe, r.Scratch)
	if err != nil {
		return err
	}
	req, err := size.BindRequest(options, r.Scratch)
	if err != nil {
		return err
	}
	next, display, result, err := ctx.ApplySupportedPostFiltersForPublication(req)
	if err != nil {
		return err
	}
	if err := next.RequirePublishablePostFilterOutput(); err != nil {
		return err
	}
	r.Size = size
	r.Request = req
	r.Context = next
	r.Result = result
	r.DisplayOutput = display
	return nil
}

func frameWorkPostFilterBindOptionsFromContext(ctx FrameWorkPostFilterContext, options FrameWorkPostFilterBindOptions) FrameWorkPostFilterBindOptions {
	if frameWorkLoopFilterMapEmpty(options.LoopFilterMap) && ctx.LoopFilterMap != nil {
		options.LoopFilterMap = *ctx.LoopFilterMap
		options.LoopFilterTrustedCoverage = true
	}
	if frameWorkCDEFIndexMapEmpty(options.CDEFIndexMap) && ctx.CDEFIndexMap != nil {
		options.CDEFIndexMap = *ctx.CDEFIndexMap
	}
	if frameWorkRestorationRecordsEmpty(options.RestorationRecords) && ctx.RestorationFrameBuffers != nil {
		options.RestorationRecords = ctx.RestorationFrameBuffers.Records
		options.RestorationBoundaries = ctx.RestorationFrameBuffers.Boundaries
	}
	return options
}

// boundSupportedPostFilterScratchLen sizes a bound runner from the scratch it
// already owns. Loop-filter edge planning is intentionally deferred to
// ApplyLoopFilterEdges: that path computes the exact edge count and reports a
// short buffer before mutating the output, so the bound runner does not need to
// pay a duplicate full planning pass just to learn the same count.
func (ctx FrameWorkPostFilterContext) boundSupportedPostFilterScratchLen(req FrameWorkPostFilterRequest, scratch FrameWorkPostFilterScratch) (FrameWorkPostFilterScratchSize, error) {
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
		size.LoopFilter = FrameWorkLoopFilterPostFilterScratchSize{
			Edges:    len(scratch.LoopFilterEdges),
			Schedule: len(scratch.LoopFilterSchedule),
		}
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
		loopFilterSize, err := ctx.LoopFilterPostFilterScratchUpperBound()
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
		loopFilterSize, err := ctx.LoopFilterPostFilterScratchUpperBound()
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
	return ctx.applySupportedPostFilters(req, FrameWorkPostFilterBanding{})
}

// ApplySupportedPostFiltersBanded runs the supported postfilter chain in AV1
// order while splitting the loop-filter and CDEF stages into row bands selected
// by banding. The banded stages preserve the same final plane pass order and
// restoration boundary-save points as ApplySupportedPostFilters.
func (ctx FrameWorkPostFilterContext) ApplySupportedPostFiltersBanded(req FrameWorkPostFilterRequest, banding FrameWorkPostFilterBanding) (FrameWorkPostFilterContext, FrameWorkPostFilterResult, error) {
	return ctx.applySupportedPostFilters(req, banding)
}

func (ctx FrameWorkPostFilterContext) applySupportedPostFilters(req FrameWorkPostFilterRequest, banding FrameWorkPostFilterBanding) (FrameWorkPostFilterContext, FrameWorkPostFilterResult, error) {
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
		var loopFilterResult FrameWorkLoopFilterPostFilterApplyResult
		var err error
		if banding.LoopFilterMIRows > 0 {
			loopFilterResult, err = ctx.ApplyLoopFilterEdgesBanded(req.LoopFilter, banding.LoopFilterMIRows)
		} else {
			loopFilterResult, err = ctx.ApplyLoopFilterEdges(req.LoopFilter)
		}
		if err != nil {
			return ctx, result, err
		}
		ctx = ctx.WithCompletedPostFilters(FrameWorkPostFilterLoopFilter)
		result.Completed |= FrameWorkPostFilterLoopFilter
		result.LoopFilter = loopFilterResult
	}
	// Save the deblock boundary lines (afterCDEF=false) for restoration before
	// CDEF mutates the inter-stripe rows libaom needs as filter context. When
	// CDEF is disabled the deblock-context is the same as the frame data; the
	// non-optimized restoration path still requires those rows to be saved.
	if remaining.Has(FrameWorkPostFilterLoopRestoration) && ctx.shouldSaveRestorationDeblockBoundaries() {
		if err := ctx.saveRestorationBoundariesForRequest(req.Restoration, false); err != nil {
			return ctx, result, err
		}
	}
	if remaining.Has(FrameWorkPostFilterCDEF) {
		var cdefResult FrameWorkCDEFPostFilterResult
		var err error
		if banding.CDEFUnitRows > 0 {
			cdefResult, err = ctx.ApplyCDEFPostFilterBanded(req.CDEF, banding.CDEFUnitRows)
		} else {
			// applyCDEFPostFilterMaybePooled fans CDEF unit rows across the idle
			// worker lanes when a multi-lane pool was threaded in, and otherwise
			// runs the byte-identical serial whole-frame apply.
			cdefResult, err = ctx.applyCDEFPostFilterMaybePooled(req.CDEF)
		}
		if err != nil {
			return ctx, result, err
		}
		ctx = ctx.WithCompletedPostFilters(FrameWorkPostFilterCDEF)
		result.Completed |= FrameWorkPostFilterCDEF
		result.CDEF = cdefResult
	}
	// Save the CDEF boundary lines (afterCDEF=true) for the frame-edge stripes
	// that have no neighboring deblock context.
	if ctx.RemainingPostFilters().Has(FrameWorkPostFilterLoopRestoration) {
		if err := ctx.saveRestorationBoundariesForRequest(req.Restoration, true); err != nil {
			return ctx, result, err
		}
	}
	if ctx.RemainingPostFilters().Has(FrameWorkPostFilterLoopRestoration) {
		cdefDebugLogChromaPreLR(ctx.Output)
		cdefDebugLogLRRecords(req.Restoration.Records)
		restorationResult, err := ctx.ApplyLoopRestorationPostFilter(req.Restoration)
		if err != nil {
			return ctx, result, err
		}
		ctx = ctx.WithCompletedPostFilters(FrameWorkPostFilterLoopRestoration)
		result.Completed |= FrameWorkPostFilterLoopRestoration
		result.Restoration = restorationResult
	}
	// Edge-replicate the reconstructed plane borders into the superblock-aligned
	// padding before the frame becomes a reference, mirroring libaom's
	// aom_extend_frame_borders (called after CDEF + loop restoration + superres
	// and before film grain, which is applied to a separate display buffer).
	// Inter prediction in later frames reads this buffer for motion-compensated
	// samples that land past the cropped edge; without the replicated edge those
	// reads would return the zero-filled padding instead of libaom's clamped
	// edge pixel. It is a no-op for frames whose planes fill their allocation.
	if ctx.Output != nil {
		ctx.Output.ExtendBorders()
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

func (ctx FrameWorkPostFilterContext) shouldSaveRestorationDeblockBoundaries() bool {
	if !ctx.ActivePostFilters().Has(FrameWorkPostFilterCDEF) {
		return true
	}
	return ctx.RemainingPostFilters().Has(FrameWorkPostFilterCDEF)
}

// ApplySupportedPostFiltersForPublication runs the supported postfilter chain
// while preserving the frame-pool surface as the clean publishable reference.
// Display-only film grain is synthesized into caller-owned request scratch and
// returned separately.
func (ctx FrameWorkPostFilterContext) ApplySupportedPostFiltersForPublication(req FrameWorkPostFilterRequest) (FrameWorkPostFilterContext, *frame.Frame, FrameWorkPostFilterResult, error) {
	var result FrameWorkPostFilterResult
	if !ctx.RemainingPostFilters().Has(FrameWorkPostFilterFilmGrain) {
		next, result, err := ctx.ApplySupportedPostFilters(req)
		return next, next.Output, result, err
	}
	if err := ctx.validateFilmGrainPostFilterRequest(req.FilmGrain); err != nil {
		return ctx, nil, result, err
	}

	publishCtx, result, err := ctx.WithCompletedPostFilters(FrameWorkPostFilterFilmGrain).ApplySupportedPostFilters(req)
	if err != nil {
		return ctx, nil, result, err
	}

	displayCtx := publishCtx
	displayCtx.completedPostFilters &^= FrameWorkPostFilterFilmGrain
	displayOutput, err := displayCtx.BindFilmGrainPostFilterOutput(req.FilmGrain)
	if err != nil {
		return ctx, nil, result, err
	}
	if displayOutput != nil {
		displayCtx.Output = displayOutput
		displayCtx.detachedPostFilterOutput = true
	}
	filmGrainResult, err := displayCtx.ApplyFilmGrainPostFilter(req.FilmGrain)
	if err != nil {
		return ctx, nil, result, err
	}
	if displayOutput == nil {
		displayOutput = publishCtx.Output
	} else if filmGrainResult.OutputSize == 0 {
		filmGrainResult.Output = displayOutput
		filmGrainResult.OutputSize = displayOutput.Layout.Size
	}
	publishCtx = publishCtx.WithCompletedPostFilters(FrameWorkPostFilterFilmGrain)
	result.Completed |= FrameWorkPostFilterFilmGrain
	result.FilmGrain = filmGrainResult
	return publishCtx, displayOutput, result, nil
}

// saveRestorationBoundariesForRequest is the AV1 loop-restoration boundary save
// pass that captures pre-CDEF (afterCDEF=false) and post-CDEF (afterCDEF=true)
// boundary samples from ctx.Output for use during restoration application.
func (ctx FrameWorkPostFilterContext) saveRestorationBoundariesForRequest(req FrameWorkRestorationPostFilterRequest, afterCDEF bool) error {
	if req.Optimized {
		return nil
	}
	if ctx.Output == nil {
		return frame.ErrInvalidSlot
	}
	boundaries := req.Boundaries
	if frameWorkRestorationBoundariesEmpty(boundaries) && ctx.RestorationFrameBuffers != nil {
		boundaries = ctx.RestorationFrameBuffers.Boundaries
	}
	if frameWorkRestorationBoundariesEmpty(boundaries) {
		return nil
	}
	plan, err := ctx.LoopRestorationPostFilterPlan()
	if err != nil {
		return err
	}
	if !plan.Active {
		return nil
	}
	bytesPerSample := ctx.Output.Layout.BytesPerSample
	if bytesPerSample == 0 {
		bytesPerSample = 1
	}
	// When super-res is active the pre-CDEF (deblock) boundary save runs on the
	// coded (downscaled) surface, but loop restoration consumes boundary rows at
	// the upscaled width. Mirror libaom's save_deblock_boundary_lines by
	// upscaling each saved row on the fly (av1/common/restoration.c ~line 1381).
	// The post-CDEF (afterCDEF=true) save runs after the surface is already
	// upscaled and uses the plain path.
	if !afterCDEF && ctx.superResActiveForRestorationSave() {
		return ctx.saveRestorationDeblockBoundariesSuperRes(plan, boundaries, bytesPerSample, req.DataScratch)
	}
	return tile.SaveRestorationFrameBoundaryLinesFromFrame(plan, *ctx.Output, bytesPerSample, boundaries, afterCDEF)
}

// superResActiveForRestorationSave reports whether the current output surface is
// still the coded (pre-super-res) surface and super-res will upscale it, so the
// pre-CDEF restoration boundary save must upscale its rows on the fly.
func (ctx FrameWorkPostFilterContext) superResActiveForRestorationSave() bool {
	if ctx.detachedPostFilterOutput {
		// Output already points at the upscaled super-res surface.
		return false
	}
	if !ctx.RemainingPostFilters().Has(FrameWorkPostFilterSuperRes) {
		return false
	}
	size := ctx.Event.FrameSize
	return size.SuperResEnabled && size.UpscaledWidth > size.CodedWidth &&
		size.SuperResDenominator >= 9 && size.SuperResDenominator <= 16
}

// saveRestorationDeblockBoundariesSuperRes performs the pre-CDEF restoration
// boundary save against the MI-aligned coded surface, upscaling each saved row
// to the upscaled plane width. It ports libaom's super-res deblock save branch.
func (ctx FrameWorkPostFilterContext) saveRestorationDeblockBoundariesSuperRes(plan tile.RestorationFramePlan, boundaries [3]tile.RestorationStripeBoundaries, bytesPerSample int, rowScratch []uint16) error {
	codedFormat, err := codedFrameFormatFromHeaders(ctx.Event.SequenceHeader, ctx.Event.FrameSize, ctx.Output.Format.Align)
	if err != nil {
		return err
	}
	var codedPlanes [3]frame.Plane
	upscale := tile.RestorationSuperResDeblockUpscale{BitDepth: ctx.Output.Format.BitDepth}
	if upscale.BitDepth == 0 {
		upscale.BitDepth = 8
	}
	maxAlignedWidth := 0
	for plane := 0; plane < int(plan.Planes); plane++ {
		if plan.Grids[plane].Type == parser.RestorationNone {
			continue
		}
		srcPlane, ok := frameWorkCDEFPlane(*ctx.Output, plane)
		if !ok {
			continue
		}
		upscale.CodedWidth[plane] = uint32(srcPlane.Width)
		xDec, yDec := frameWorkCDEFPlaneDecimation(codedFormat, plane)
		srcPlane = frameWorkCDEFAlignedPlane(srcPlane, ctx.Event.FrameSize, xDec, yDec, bytesPerSample)
		codedPlanes[plane] = srcPlane
		if srcPlane.Width > maxAlignedWidth {
			maxAlignedWidth = srcPlane.Width
		}
	}
	if maxAlignedWidth > 0 {
		if len(rowScratch) < maxAlignedWidth {
			return tile.ErrJobBufferTooSmall
		}
		upscale.RowScratch = rowScratch[:maxAlignedWidth]
	}
	return tile.SaveRestorationFrameBoundaryLinesFromFrameSuperRes(plan, codedPlanes, bytesPerSample, upscale, boundaries)
}

// ApplyCallerPostFilters runs all active postfilters in AV1 order. Unlike the
// supported frame-pool publication runner, this path may switch ctx.Output to
// caller-owned superres scratch and is therefore intended for display/output
// consumers that can keep that detached frame alive.
func (ctx FrameWorkPostFilterContext) ApplyCallerPostFilters(req FrameWorkPostFilterRequest) (FrameWorkPostFilterContext, FrameWorkCallerPostFilterResult, error) {
	var result FrameWorkCallerPostFilterResult
	if _, err := ctx.validateCallerPostFilterRequest(req); err != nil {
		return ctx, result, err
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

func (ctx FrameWorkPostFilterContext) validateCallerPostFilterRequest(req FrameWorkPostFilterRequest) (FrameWorkPostFilterContext, error) {
	remaining := ctx.RemainingPostFilters()
	if remaining.Has(FrameWorkPostFilterCDEF) {
		if err := ctx.validateCDEFPostFilterRequest(req.CDEF); err != nil {
			return ctx, err
		}
	}
	validateTailCtx := ctx.WithCompletedPostFilters(FrameWorkPostFilterLoopFilter | FrameWorkPostFilterCDEF)
	if validateTailCtx.RemainingPostFilters().Has(FrameWorkPostFilterSuperRes) {
		if err := validateTailCtx.validateSuperResPostFilterRequest(req.SuperRes); err != nil {
			return ctx, err
		}
		var err error
		validateTailCtx, err = validateTailCtx.bindSuperResPostFilterOutputContext(req.SuperRes)
		if err != nil {
			return ctx, err
		}
	}
	if validateTailCtx.RemainingPostFilters().Has(FrameWorkPostFilterLoopRestoration) {
		if err := validateTailCtx.validateLoopRestorationPostFilterRequest(req.Restoration); err != nil {
			return ctx, err
		}
	}
	if validateTailCtx.RemainingPostFilters().Has(FrameWorkPostFilterFilmGrain) {
		if err := validateTailCtx.validateFilmGrainPostFilterRequest(req.FilmGrain); err != nil {
			return ctx, err
		}
	}
	return validateTailCtx, nil
}

// ApplyCallerPostFiltersForPublication runs the full caller-owned AV1
// postfilter chain while preserving a clean publishable reference surface. When
// film grain is active, the returned context/output remain the post-superres and
// post-restoration reference frame, and display receives a separate grain-applied
// frame.
func (ctx FrameWorkPostFilterContext) ApplyCallerPostFiltersForPublication(req FrameWorkPostFilterRequest) (FrameWorkPostFilterContext, *frame.Frame, FrameWorkCallerPostFilterResult, error) {
	var result FrameWorkCallerPostFilterResult
	if !ctx.RemainingPostFilters().Has(FrameWorkPostFilterFilmGrain) {
		next, result, err := ctx.ApplyCallerPostFilters(req)
		return next, next.Output, result, err
	}
	validateTailCtx, err := ctx.validateCallerPostFilterRequest(req)
	if err != nil {
		return ctx, nil, result, err
	}
	if err := validateTailCtx.validateFilmGrainPostFilterOutputRequest(req.FilmGrain); err != nil {
		return ctx, nil, result, err
	}

	publishCtx, result, err := ctx.WithCompletedPostFilters(FrameWorkPostFilterFilmGrain).ApplyCallerPostFilters(req)
	if err != nil {
		return ctx, nil, result, err
	}

	displayCtx := publishCtx
	displayCtx.completedPostFilters &^= FrameWorkPostFilterFilmGrain
	displayOutput, err := displayCtx.BindFilmGrainPostFilterOutput(req.FilmGrain)
	if err != nil {
		return ctx, nil, result, err
	}
	if displayOutput != nil {
		displayCtx.Output = displayOutput
		displayCtx.detachedPostFilterOutput = true
	}
	filmGrainResult, err := displayCtx.ApplyFilmGrainPostFilter(req.FilmGrain)
	if err != nil {
		return ctx, nil, result, err
	}
	if displayOutput == nil {
		displayOutput = publishCtx.Output
	} else if filmGrainResult.OutputSize == 0 {
		filmGrainResult.Output = displayOutput
		filmGrainResult.OutputSize = displayOutput.Layout.Size
	}
	publishCtx = publishCtx.WithCompletedPostFilters(FrameWorkPostFilterFilmGrain)
	result.Completed |= FrameWorkPostFilterFilmGrain
	result.FilmGrain = filmGrainResult
	return publishCtx, displayOutput, result, nil
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
	if remaining.Has(FrameWorkPostFilterLoopRestoration) {
		if err := ctx.saveRestorationBoundariesForRequest(req.Restoration, false); err != nil {
			return ctx, result, err
		}
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
	for plane := range len(result.Data) {
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
	samples, err := tile.RestorationFrameSampleScratchLen(plan, *ctx.Output, optimized)
	if err != nil {
		return FrameWorkRestorationPostFilterScratchSize{}, err
	}
	rowScratch, err := ctx.loopRestorationSuperResBoundaryRowScratchLen(plan)
	if err != nil {
		return FrameWorkRestorationPostFilterScratchSize{}, err
	}
	if rowScratch > samples.DataLen {
		samples.DataLen = rowScratch
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

func (ctx FrameWorkPostFilterContext) loopRestorationSuperResBoundaryRowScratchLen(plan tile.RestorationFramePlan) (int, error) {
	size := ctx.Event.FrameSize
	if !size.SuperResEnabled || size.UpscaledWidth <= size.CodedWidth ||
		size.SuperResDenominator < 9 || size.SuperResDenominator > 16 {
		return 0, nil
	}
	if ctx.Output == nil {
		return 0, frame.ErrInvalidSlot
	}
	codedFormat, err := codedFrameFormatFromHeaders(ctx.Event.SequenceHeader, size, ctx.Output.Format.Align)
	if err != nil {
		return 0, err
	}
	alignedLumaW := (int(size.CodedWidth) + 7) &^ 7
	maxAlignedWidth := 0
	for plane := 0; plane < int(plan.Planes); plane++ {
		if plan.Grids[plane].Type == parser.RestorationNone {
			continue
		}
		xDec, _ := frameWorkCDEFPlaneDecimation(codedFormat, plane)
		width := alignedLumaW >> xDec
		if width > maxAlignedWidth {
			maxAlignedWidth = width
		}
	}
	return maxAlignedWidth, nil
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
	applySamples, err := tile.RestorationFrameSampleScratchLen(plan, *ctx.Output, req.Optimized)
	if err != nil {
		return tile.RestorationFrameApplyResult{}, err
	}
	dataScratch := req.DataScratch
	if len(dataScratch) > applySamples.DataLen {
		dataScratch = dataScratch[:applySamples.DataLen]
	}
	dstScratch := req.DstScratch
	if len(dstScratch) > applySamples.DstLen {
		dstScratch = dstScratch[:applySamples.DstLen]
	}
	return tile.ApplyRestorationFrameToFrame(plan, *ctx.Output, req.Records, req.Boundaries, dataScratch, dstScratch, req.Scratch, req.Optimized)
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
