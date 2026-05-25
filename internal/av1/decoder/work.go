package decoder

import (
	"errors"

	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/threading"
	"github.com/thesyncim/goav1/internal/av1/tile"
	decodework "github.com/thesyncim/goav1/internal/av1/work"
)

var ErrInvalidTileWork = errors.New("decoder: invalid tile work")

type TileWorkPlan = decodework.TilePlan
type FrameWorkPlan = decodework.FramePlan
type FrameTileWorkPlan = decodework.FrameTilePlan
type ShowExistingFrameWorkPlan = decodework.ShowExistingFramePlan
type FrameWorkStepKind = decodework.FrameStepKind
type FrameWorkStep = decodework.FrameStep
type FrameWorkStepResult = decodework.FrameStepResult
type FrameWorkBatch = threading.FrameWorkBatch
type FrameWorkSequenceContext = threading.FrameWorkSequenceContext
type FrameWorkFrameContext = threading.FrameWorkFrameContext
type FrameWorkJobRegion = threading.FrameWorkJobRegion
type FrameWorkPlane = threading.FrameWorkPlane
type FrameWorkPlaneRegion = threading.FrameWorkPlaneRegion
type FrameWorkReference = threading.FrameWorkReference
type FrameWorkLoopRestorationPlan = threading.FrameWorkLoopRestorationPlan
type FrameWorkBatchFunc = threading.FrameWorkBatchFunc

const (
	FrameWorkStepIgnored      FrameWorkStepKind = decodework.FrameStepIgnored
	FrameWorkStepDropped      FrameWorkStepKind = decodework.FrameStepDropped
	FrameWorkStepBegin        FrameWorkStepKind = decodework.FrameStepBegin
	FrameWorkStepTile         FrameWorkStepKind = decodework.FrameStepTile
	FrameWorkStepShowExisting FrameWorkStepKind = decodework.FrameStepShowExisting

	FrameWorkPlaneY FrameWorkPlane = threading.FrameWorkPlaneY
	FrameWorkPlaneU FrameWorkPlane = threading.FrameWorkPlaneU
	FrameWorkPlaneV FrameWorkPlane = threading.FrameWorkPlaneV

	FrameWorkReferenceLast    FrameWorkReference = threading.FrameWorkReferenceLast
	FrameWorkReferenceLast2   FrameWorkReference = threading.FrameWorkReferenceLast2
	FrameWorkReferenceLast3   FrameWorkReference = threading.FrameWorkReferenceLast3
	FrameWorkReferenceGolden  FrameWorkReference = threading.FrameWorkReferenceGolden
	FrameWorkReferenceBwd     FrameWorkReference = threading.FrameWorkReferenceBwd
	FrameWorkReferenceAltRef2 FrameWorkReference = threading.FrameWorkReferenceAltRef2
	FrameWorkReferenceAltRef  FrameWorkReference = threading.FrameWorkReferenceAltRef
)

// FrameWorkEventResult reports the planning, output, and execution result for
// one event-level frame-work run.
type FrameWorkEventResult struct {
	Step           FrameWorkStep
	Output         *frame.Frame
	ReferenceCount int
	Run            FrameWorkStepResult
}

// FrameWorkPostFilterContext is supplied after final tile work succeeds and
// before the decoded surface is published to reference slots.
type FrameWorkPostFilterContext struct {
	Event Event
	Step  FrameWorkStep

	Output           *frame.Frame
	ReferenceCount   int
	ExecutedTileWork bool

	CDEFIndexMap            *threading.FrameWorkCDEFIndexMap
	LoopFilterMap           *threading.FrameWorkLoopFilterMap
	RestorationFrameBuffers *threading.FrameWorkRestorationFrameBuffers

	completedPostFilters     FrameWorkPostFilterStage
	detachedPostFilterOutput bool
}

// FrameWorkPostFilterFunc applies final frame postfilters before reference
// publication. Returning an error keeps frame work active and unpublished.
type FrameWorkPostFilterFunc func(FrameWorkPostFilterContext) error

// FrameWorkPostFilterRunner applies final frame postfilters before reference
// publication without requiring callers to allocate a method-value callback.
type FrameWorkPostFilterRunner interface {
	Apply(FrameWorkPostFilterContext) error
}

// FrameWorkSideDataFunc binds caller-owned frame-level side data after event
// planning and before tile work executes.
type FrameWorkSideDataFunc func(*FrameWorkState, FrameWorkBatch) error

// FrameWorkSideDataRunner binds caller-owned frame-level side data after event
// planning and before tile work executes.
type FrameWorkSideDataRunner interface {
	BindFrameWorkSideData(*FrameWorkState, FrameWorkBatch) error
}

// FrameWorkBoundSideDataRunner binds caller-owned frame-level side-data buffers
// for active postfilter stages before tile work executes.
type FrameWorkBoundSideDataRunner struct {
	CDEFIndex []uint8
	CDEFRead  []bool

	LoopFilterRecords []threading.FrameWorkLoopFilterBlockRecord

	RestorationRecords []tile.RestorationUnitRecord
	RestorationAbove   []uint16
	RestorationBelow   []uint16

	CDEFIndexMap            threading.FrameWorkCDEFIndexMap
	LoopFilterMap           threading.FrameWorkLoopFilterMap
	RestorationFrameBuffers threading.FrameWorkRestorationFrameBuffers
}

// FrameWorkSideDataScratchSize reports caller-owned side-data storage needed by
// FrameWorkBoundSideDataRunner for one frame context.
type FrameWorkSideDataScratchSize struct {
	CDEF                int
	LoopFilterRecords   int
	RestorationRecords  int
	RestorationBoundary int
}

// FrameWorkSideDataScratch carries caller-owned backing storage for
// FrameWorkBoundSideDataRunner.
type FrameWorkSideDataScratch struct {
	CDEFIndex []uint8
	CDEFRead  []bool

	LoopFilterRecords []threading.FrameWorkLoopFilterBlockRecord

	RestorationRecords []tile.RestorationUnitRecord
	RestorationAbove   []uint16
	RestorationBelow   []uint16
}

// FrameWorkState is caller-owned lifecycle state for one in-flight frame. It
// records the acquired output surface between the frame begin event, any later
// tile-group continuation events, and the final reference publication step.
type FrameWorkState struct {
	Surface        int
	ReferenceCount int
	Sequence       threading.FrameWorkSequenceContext

	tileResidualFrameContexts     [parser.RefFrames]threading.FrameWorkTileResidualCDFStorage
	tileResidualFrameContextValid [parser.RefFrames]bool
	tileResidualCurrentCDFs       threading.FrameWorkTileResidualCDFStorage
	tileResidualCurrentCDFsValid  bool
	tileResidualRetainedCDFs      threading.FrameWorkTileResidualCDFStorage
	tileResidualRetainedCDFsValid bool
	cdefIndexMap                  threading.FrameWorkCDEFIndexMap
	cdefIndexMapValid             bool
	loopFilterMap                 threading.FrameWorkLoopFilterMap
	loopFilterMapValid            bool
	restorationFrameBuffers       threading.FrameWorkRestorationFrameBuffers
	restorationFrameBuffersValid  bool

	active bool
}

// EventDropsFrameWork reports whether a stream event implies that any active
// incomplete frame work should be discarded. This matches Stream's pending
// frame reset points.
func EventDropsFrameWork(event Event) bool {
	return event.NewCodedVideoSequence || event.NewTemporalUnit || event.Kind == EventExistingFrame
}

// EventCompletesFrameWork reports whether event carries the final tile group
// for an in-flight frame and is therefore ready for reference publication
// after tile decoding succeeds.
func EventCompletesFrameWork(event Event) bool {
	return (event.Kind == EventFrame || event.Kind == EventTileGroup) && event.TileGroup.Final
}

// EventOutputsFrame reports whether event can produce a display frame. Unlike
// EventCompletesFrameWork, this includes show-existing-frame events that output
// a retained reference without running tile work.
func EventOutputsFrame(event Event) bool {
	return EventCompletesFrameWork(event) || event.Kind == EventExistingFrame
}

// Active reports whether a frame has begun and not yet been finished, aborted,
// or reset.
func (s *FrameWorkState) Active() bool {
	return s != nil && s.active
}

// Reset clears any in-flight frame work state without touching frame-pool or
// reference ownership. Callers should normally use Finish for successfully
// decoded final tile groups and Abort for incomplete or discarded frames.
func (s *FrameWorkState) Reset() {
	if s == nil {
		return
	}
	*s = FrameWorkState{}
}

func (s *FrameWorkState) resetActive() {
	if s == nil {
		return
	}
	s.Surface = 0
	s.ReferenceCount = 0
	s.Sequence = threading.FrameWorkSequenceContext{}
	s.tileResidualCurrentCDFs = threading.FrameWorkTileResidualCDFStorage{}
	s.tileResidualCurrentCDFsValid = false
	s.tileResidualRetainedCDFs = threading.FrameWorkTileResidualCDFStorage{}
	s.tileResidualRetainedCDFsValid = false
	s.cdefIndexMap = threading.FrameWorkCDEFIndexMap{}
	s.cdefIndexMapValid = false
	s.loopFilterMap = threading.FrameWorkLoopFilterMap{}
	s.loopFilterMapValid = false
	s.restorationFrameBuffers = threading.FrameWorkRestorationFrameBuffers{}
	s.restorationFrameBuffersValid = false
	s.active = false
}

func (s *FrameWorkState) resetReferenceState() {
	if s == nil {
		return
	}
	s.tileResidualFrameContexts = [parser.RefFrames]threading.FrameWorkTileResidualCDFStorage{}
	s.tileResidualFrameContextValid = [parser.RefFrames]bool{}
}

// SetCDEFIndexMap attaches frame-level CDEF side data to active frame work.
// The caller owns the backing slices; the state passes this view into tile
// batches and drops it when the frame finishes or aborts.
func (s *FrameWorkState) SetCDEFIndexMap(cdefMap threading.FrameWorkCDEFIndexMap) error {
	if s == nil || !s.active {
		return ErrInvalidFrameWorkState
	}
	if err := cdefMap.Reset(); err != nil {
		return err
	}
	s.cdefIndexMap = cdefMap
	s.cdefIndexMapValid = true
	return nil
}

// SetLoopFilterMap attaches frame-level loop-filter side data to active frame
// work. The caller owns the backing slice; the state passes this view into tile
// batches and drops it when the frame finishes or aborts.
func (s *FrameWorkState) SetLoopFilterMap(lfMap threading.FrameWorkLoopFilterMap) error {
	if s == nil || !s.active {
		return ErrInvalidFrameWorkState
	}
	if err := lfMap.Reset(); err != nil {
		return err
	}
	s.loopFilterMap = lfMap
	s.loopFilterMapValid = true
	return nil
}

// SetRestorationFrameBuffers attaches frame-level loop-restoration side data to
// active frame work. The caller owns the backing slices; records are reset so
// tile work can fill decoded restoration units before the postfilter callback.
func (s *FrameWorkState) SetRestorationFrameBuffers(buffers threading.FrameWorkRestorationFrameBuffers) error {
	if s == nil || !s.active {
		return ErrInvalidFrameWorkState
	}
	if err := buffers.ResetRecords(); err != nil {
		return err
	}
	s.restorationFrameBuffers = buffers
	s.restorationFrameBuffersValid = true
	return nil
}

// SetSideData attaches all caller-owned frame-level side data to active frame
// work after validating and resetting every view. The state is left unchanged
// if any side-data view is invalid.
func (s *FrameWorkState) SetSideData(cdefMap threading.FrameWorkCDEFIndexMap, lfMap threading.FrameWorkLoopFilterMap, buffers threading.FrameWorkRestorationFrameBuffers) error {
	if s == nil || !s.active {
		return ErrInvalidFrameWorkState
	}
	if err := cdefMap.Reset(); err != nil {
		return err
	}
	if err := lfMap.Reset(); err != nil {
		return err
	}
	if err := buffers.ResetRecords(); err != nil {
		return err
	}
	s.cdefIndexMap = cdefMap
	s.cdefIndexMapValid = true
	s.loopFilterMap = lfMap
	s.loopFilterMapValid = true
	s.restorationFrameBuffers = buffers
	s.restorationFrameBuffersValid = true
	return nil
}

// Begin acquires the output surface and records the active frame work state.
func (s *FrameWorkState) Begin(refs *SurfaceReferences, pool *frame.Pool, sequence parser.SequenceHeader, event Event, align int, references []int, workers int, spans []parser.TileSpan, jobs []tile.Job, batches []threading.Batch) (FrameWorkPlan, *frame.Frame, error) {
	if s == nil || s.active {
		return FrameWorkPlan{}, nil, ErrInvalidFrameWorkState
	}
	tileResidualCDFs, err := s.initialTileResidualCDFs(event)
	if err != nil {
		return FrameWorkPlan{}, nil, err
	}
	plan, output, err := BeginFrameWork(refs, pool, sequence, event, align, references, workers, spans, jobs, batches)
	if err != nil {
		return FrameWorkPlan{}, nil, err
	}
	s.Surface = plan.Surface
	s.ReferenceCount = plan.ReferenceCount
	s.Sequence = threading.FrameWorkSequenceContextFromHeader(sequence)
	s.tileResidualCurrentCDFs = tileResidualCDFs
	s.tileResidualCurrentCDFsValid = true
	s.tileResidualRetainedCDFs = threading.FrameWorkTileResidualCDFStorage{}
	s.tileResidualRetainedCDFsValid = false
	s.active = true
	return plan, output, nil
}

// PlanTile plans tile work for a continuation tile group using the active
// frame's output surface and resolved reference count.
func (s *FrameWorkState) PlanTile(event Event, workers int, spans []parser.TileSpan, jobs []tile.Job, batches []threading.Batch) (FrameTileWorkPlan, error) {
	if s == nil || !s.active {
		return FrameTileWorkPlan{}, ErrInvalidFrameWorkState
	}
	return PlanFrameTileWork(event, s.Surface, s.ReferenceCount, workers, spans, jobs, batches)
}

// Finish applies the final frame event to surface references and releases any
// overwritten frame-pool slots atomically. State is cleared only after a
// successful finish so callers can recover or retry after transient release
// failures.
func (s *FrameWorkState) Finish(refs *SurfaceReferences, pool *frame.Pool, event Event, releases []int) (int, error) {
	if s == nil || !s.active {
		return 0, ErrInvalidFrameWorkState
	}
	count, err := FinishFrameSurface(refs, pool, event, s.Surface, releases)
	if err != nil {
		return 0, err
	}
	s.finishTileResidualCDFs(event)
	s.resetActive()
	return count, nil
}

func (s *FrameWorkState) initialTileResidualCDFs(event Event) (threading.FrameWorkTileResidualCDFStorage, error) {
	var storage threading.FrameWorkTileResidualCDFStorage
	if event.FrameHeader.PrimaryRefFrame != parser.PrimaryRefNone && event.FrameHeader.PrimaryRefFrame < parser.InterRefsPerFrame {
		slot := event.FrameSize.RefFrameIdx[event.FrameHeader.PrimaryRefFrame]
		if slot < parser.RefFrames && s.tileResidualFrameContextValid[slot] {
			return s.tileResidualFrameContexts[slot], nil
		}
	}
	if err := storage.InitDefault(event.Quantization.BaseQIdx); err != nil {
		return threading.FrameWorkTileResidualCDFStorage{}, err
	}
	return storage, nil
}

func (s *FrameWorkState) finishTileResidualCDFs(event Event) {
	if s == nil || !s.tileResidualCurrentCDFsValid {
		return
	}
	cdfs := s.tileResidualCurrentCDFs
	if s.tileResidualRetainedCDFsValid {
		cdfs = s.tileResidualRetainedCDFs
	}
	for i := 0; i < parser.RefFrames; i++ {
		if (event.FrameSize.RefreshFrameFlags & (1 << uint(i))) == 0 {
			continue
		}
		s.tileResidualFrameContexts[i] = cdfs
		s.tileResidualFrameContextValid[i] = true
	}
}

// Abort releases the active output surface without publishing it to reference
// slots. State is cleared only after the pool accepts the release.
func (s *FrameWorkState) Abort(pool *frame.Pool) error {
	if s == nil || !s.active {
		return ErrInvalidFrameWorkState
	}
	if err := pool.Release(s.Surface); err != nil {
		return err
	}
	s.resetActive()
	return nil
}

// AbortIfEventDropsFrameWork aborts active frame work when event marks a
// stream boundary that invalidates incomplete frame data. It is a no-op for
// inactive state and non-boundary events.
func (s *FrameWorkState) AbortIfEventDropsFrameWork(pool *frame.Pool, event Event) (bool, error) {
	if !EventDropsFrameWork(event) || s == nil || !s.active {
		return false, nil
	}
	if err := s.Abort(pool); err != nil {
		return false, err
	}
	return true, nil
}

// FinishIfEventCompletesFrameWork publishes active frame work only when event
// carries a final tile group. It is a no-op for non-final events.
func (s *FrameWorkState) FinishIfEventCompletesFrameWork(refs *SurfaceReferences, pool *frame.Pool, event Event, releases []int) (bool, int, error) {
	if !EventCompletesFrameWork(event) {
		return false, 0, nil
	}
	if s == nil || !s.active {
		return false, 0, ErrInvalidFrameWorkState
	}
	count, err := s.Finish(refs, pool, event, releases)
	if err != nil {
		return false, 0, err
	}
	return true, count, nil
}

// PostFilterContext returns the final-frame postfilter context that RunStep
// would pass to a postfilter hook, without executing tile work or publishing
// references. Non-final events return a zero context and nil error.
func (s *FrameWorkState) PostFilterContext(framePool *frame.Pool, event Event, step FrameWorkStep, executed bool) (FrameWorkPostFilterContext, error) {
	if !EventCompletesFrameWork(event) {
		return FrameWorkPostFilterContext{}, nil
	}
	if s == nil || !s.active {
		return FrameWorkPostFilterContext{}, ErrInvalidFrameWorkState
	}
	if !frameWorkStepMatchesEvent(event, step) {
		return FrameWorkPostFilterContext{}, ErrInvalidFrameWorkStep
	}
	_, referenceCount, _, err := frameWorkStepTilePlan(step)
	if err != nil {
		return FrameWorkPostFilterContext{}, err
	}
	output, err := frameWorkStepOutput(framePool, step)
	if err != nil {
		return FrameWorkPostFilterContext{}, err
	}
	cdefIndexMap, loopFilterMap, restorationFrameBuffers := s.postFilterSideData()
	return FrameWorkPostFilterContext{
		Event:                   event,
		Step:                    step,
		Output:                  output,
		ReferenceCount:          referenceCount,
		ExecutedTileWork:        executed,
		CDEFIndexMap:            cdefIndexMap,
		LoopFilterMap:           loopFilterMap,
		RestorationFrameBuffers: restorationFrameBuffers,
	}, nil
}

// ShowExisting resolves a show-existing-frame event to an output surface,
// aborts any active incomplete frame work, and applies any AV1 key-frame
// reference reset. Active frame work is dropped only after the show-existing
// event validates and the abort succeeds.
func (s *FrameWorkState) ShowExisting(refs *SurfaceReferences, pool *frame.Pool, event Event, releases []int) (ShowExistingFrameWorkPlan, error) {
	if event.Kind != EventExistingFrame || !event.FrameHeader.ShowExistingFrame {
		return ShowExistingFrameWorkPlan{}, ErrInvalidSurfaceEvent
	}
	if refs == nil {
		return ShowExistingFrameWorkPlan{}, ErrInvalidSurfaceReference
	}

	next := *refs
	surface, releaseCount, err := next.ShowExistingFrame(event, releases)
	if err != nil {
		return ShowExistingFrameWorkPlan{}, err
	}
	if releaseCount != 0 && pool == nil {
		return ShowExistingFrameWorkPlan{}, frame.ErrInvalidPool
	}

	dropped := false
	if s != nil && s.active {
		if err := s.Abort(pool); err != nil {
			return ShowExistingFrameWorkPlan{}, err
		}
		dropped = true
	}
	if releaseCount != 0 {
		if err := pool.ReleaseMany(releases[:releaseCount]); err != nil {
			return ShowExistingFrameWorkPlan{}, err
		}
	}
	*refs = next
	return ShowExistingFrameWorkPlan{
		Surface:          surface,
		ReleaseCount:     releaseCount,
		DroppedFrameWork: dropped,
	}, nil
}

// PlanEvent applies one parsed decoder event to frame-work state and returns
// the work that should be executed by the caller. It does not finish final tile
// groups; callers should execute the planned tile work first, then call
// FinishIfEventCompletesFrameWork with the same event.
func (s *FrameWorkState) PlanEvent(refs *SurfaceReferences, pool *frame.Pool, sequence parser.SequenceHeader, event Event, align int, references []int, workers int, spans []parser.TileSpan, jobs []tile.Job, batches []threading.Batch, releases []int) (FrameWorkStep, *frame.Frame, error) {
	if event.Kind == EventExistingFrame {
		plan, err := s.ShowExisting(refs, pool, event, releases)
		if err != nil {
			return FrameWorkStep{}, nil, err
		}
		return FrameWorkStep{
			Kind:             FrameWorkStepShowExisting,
			DroppedFrameWork: plan.DroppedFrameWork,
			ReleaseCount:     plan.ReleaseCount,
			ShowExisting:     plan,
		}, nil, nil
	}

	dropped := false
	releaseCount := 0
	if EventDropsFrameWork(event) {
		var err error
		dropped, err = s.AbortIfEventDropsFrameWork(pool, event)
		if err != nil {
			return FrameWorkStep{}, nil, err
		}
		if event.NewCodedVideoSequence {
			releaseCount, err = ResetFrameSurfaceReferences(refs, pool, releases)
			if err != nil {
				return FrameWorkStep{}, nil, err
			}
			s.resetReferenceState()
		}
		if event.Kind != EventFrameHeader && event.Kind != EventFrame {
			if dropped {
				return FrameWorkStep{Kind: FrameWorkStepDropped, DroppedFrameWork: true, ReleaseCount: releaseCount}, nil, nil
			}
			return FrameWorkStep{ReleaseCount: releaseCount}, nil, nil
		}
	}

	switch event.Kind {
	case EventFrameHeader, EventFrame:
		plan, output, err := s.Begin(refs, pool, sequence, event, align, references, workers, spans, jobs, batches)
		if err != nil {
			return FrameWorkStep{}, nil, err
		}
		return FrameWorkStep{
			Kind:             FrameWorkStepBegin,
			DroppedFrameWork: dropped,
			ReleaseCount:     releaseCount,
			Begin:            plan,
		}, output, nil

	case EventTileGroup:
		plan, err := s.PlanTile(event, workers, spans, jobs, batches)
		if err != nil {
			return FrameWorkStep{}, nil, err
		}
		return FrameWorkStep{
			Kind: FrameWorkStepTile,
			Tile: plan,
		}, nil, nil
	}

	return FrameWorkStep{}, nil, nil
}

// RunStep executes any tile work in step, then publishes the active frame only
// if event carries a final tile group. step must be the PlanEvent result for
// event, which prevents final references from being published for unexecuted
// or mismatched tile work.
func (s *FrameWorkState) RunStep(refs *SurfaceReferences, framePool *frame.Pool, event Event, step FrameWorkStep, workerPool *threading.Pool, jobs []tile.Job, batches []threading.Batch, releases []int, fn threading.BatchFunc) (FrameWorkStepResult, error) {
	return s.RunStepWithPostFilter(refs, framePool, event, step, workerPool, jobs, batches, releases, fn, nil)
}

// RunStepWithPostFilter is RunStep with a final-frame postfilter hook that
// runs after tile work succeeds and before reference publication.
func (s *FrameWorkState) RunStepWithPostFilter(refs *SurfaceReferences, framePool *frame.Pool, event Event, step FrameWorkStep, workerPool *threading.Pool, jobs []tile.Job, batches []threading.Batch, releases []int, fn threading.BatchFunc, post FrameWorkPostFilterFunc) (FrameWorkStepResult, error) {
	if !frameWorkStepMatchesEvent(event, step) {
		return FrameWorkStepResult{}, ErrInvalidFrameWorkStep
	}
	plan, referenceCount, hasTile, err := frameWorkStepTilePlan(step)
	if err != nil {
		return FrameWorkStepResult{}, err
	}
	executed := false
	if hasTile {
		if err := ExecuteTileWork(plan, workerPool, jobs, batches, fn); err != nil {
			return FrameWorkStepResult{}, err
		}
		executed = plan.JobCount != 0
	}
	output, err := frameWorkPostFilterOutput(event, framePool, step, post)
	if err != nil {
		return FrameWorkStepResult{}, err
	}
	cdefIndexMap, loopFilterMap, restorationFrameBuffers := s.postFilterSideData()
	if err := runFrameWorkPostFilter(event, step, output, referenceCount, executed, cdefIndexMap, loopFilterMap, restorationFrameBuffers, post); err != nil {
		return FrameWorkStepResult{ExecutedTileWork: executed}, err
	}
	completed, releaseCount, err := s.FinishIfEventCompletesFrameWork(refs, framePool, event, releases)
	if err != nil {
		return FrameWorkStepResult{ExecutedTileWork: executed}, err
	}
	return FrameWorkStepResult{
		ExecutedTileWork: executed,
		CompletedFrame:   completed,
		ReleaseCount:     step.ReleaseCount + releaseCount,
	}, nil
}

// RunStepWithContext is RunStep with decoder frame context attached to each
// executed tile batch.
func (s *FrameWorkState) RunStepWithContext(refs *SurfaceReferences, framePool *frame.Pool, event Event, step FrameWorkStep, workerPool *threading.Pool, output *frame.Frame, references []*frame.Frame, jobs []tile.Job, batches []threading.Batch, releases []int, fn FrameWorkBatchFunc) (FrameWorkStepResult, error) {
	return s.RunStepWithContextAndPostFilter(refs, framePool, event, step, workerPool, output, references, jobs, batches, releases, fn, nil)
}

// RunStepWithContextAndPostFilter is RunStepWithContext with a final-frame
// postfilter hook that runs before reference publication.
func (s *FrameWorkState) RunStepWithContextAndPostFilter(refs *SurfaceReferences, framePool *frame.Pool, event Event, step FrameWorkStep, workerPool *threading.Pool, output *frame.Frame, references []*frame.Frame, jobs []tile.Job, batches []threading.Batch, releases []int, fn FrameWorkBatchFunc, post FrameWorkPostFilterFunc) (FrameWorkStepResult, error) {
	return s.runStepWithPayloadContext(refs, framePool, event, step, workerPool, output, references, nil, false, jobs, batches, releases, fn, post)
}

// RunStepWithPayloadContext is RunStepWithContext with the tile-group payload
// attached to every executed tile batch.
func (s *FrameWorkState) RunStepWithPayloadContext(refs *SurfaceReferences, framePool *frame.Pool, event Event, step FrameWorkStep, workerPool *threading.Pool, output *frame.Frame, references []*frame.Frame, payload []byte, jobs []tile.Job, batches []threading.Batch, releases []int, fn FrameWorkBatchFunc) (FrameWorkStepResult, error) {
	return s.RunStepWithPayloadContextAndPostFilter(refs, framePool, event, step, workerPool, output, references, payload, jobs, batches, releases, fn, nil)
}

// RunStepWithPayloadContextAndPostFilter is RunStepWithPayloadContext with a
// final-frame postfilter hook that runs before reference publication.
func (s *FrameWorkState) RunStepWithPayloadContextAndPostFilter(refs *SurfaceReferences, framePool *frame.Pool, event Event, step FrameWorkStep, workerPool *threading.Pool, output *frame.Frame, references []*frame.Frame, payload []byte, jobs []tile.Job, batches []threading.Batch, releases []int, fn FrameWorkBatchFunc, post FrameWorkPostFilterFunc) (FrameWorkStepResult, error) {
	return s.runStepWithPayloadContext(refs, framePool, event, step, workerPool, output, references, payload, true, jobs, batches, releases, fn, post)
}

// RunStepWithPayloadContextAndPostFilterRunner is
// RunStepWithPayloadContextAndPostFilter using a frame-work batch runner
// directly instead of a callback adapter.
func (s *FrameWorkState) RunStepWithPayloadContextAndPostFilterRunner(refs *SurfaceReferences, framePool *frame.Pool, event Event, step FrameWorkStep, workerPool *threading.Pool, output *frame.Frame, references []*frame.Frame, payload []byte, jobs []tile.Job, batches []threading.Batch, releases []int, runner threading.FrameWorkBatchRunner, post FrameWorkPostFilterFunc) (FrameWorkStepResult, error) {
	return s.runStepWithPayloadContextRunner(refs, framePool, event, step, workerPool, output, references, payload, true, jobs, batches, releases, runner, post)
}

// RunStepWithPayloadContextAndPostFilterRunners is
// RunStepWithPayloadContextAndPostFilterRunner using direct runners for both
// tile batches and final-frame postfilters.
func (s *FrameWorkState) RunStepWithPayloadContextAndPostFilterRunners(refs *SurfaceReferences, framePool *frame.Pool, event Event, step FrameWorkStep, workerPool *threading.Pool, output *frame.Frame, references []*frame.Frame, payload []byte, jobs []tile.Job, batches []threading.Batch, releases []int, runner threading.FrameWorkBatchRunner, post FrameWorkPostFilterRunner) (FrameWorkStepResult, error) {
	return s.runStepWithPayloadContextRunners(refs, framePool, event, step, workerPool, output, references, payload, true, jobs, batches, releases, runner, post)
}

func (s *FrameWorkState) runStepWithPayloadContext(refs *SurfaceReferences, framePool *frame.Pool, event Event, step FrameWorkStep, workerPool *threading.Pool, output *frame.Frame, references []*frame.Frame, payload []byte, validatePayload bool, jobs []tile.Job, batches []threading.Batch, releases []int, fn FrameWorkBatchFunc, post FrameWorkPostFilterFunc) (FrameWorkStepResult, error) {
	if !frameWorkStepMatchesEvent(event, step) {
		return FrameWorkStepResult{}, ErrInvalidFrameWorkStep
	}
	_, referenceCount, _, err := frameWorkStepTilePlan(step)
	if err != nil {
		return FrameWorkStepResult{}, err
	}
	frameContext := frameWorkFrameContext(event, s.sequenceContext())
	var initialTileResidualCDFs *threading.FrameWorkTileResidualCDFStorage
	var retainedTileResidualCDFs *threading.FrameWorkTileResidualCDFStorage
	var retainedTileResidualCDFsValid *bool
	if s != nil && s.tileResidualCurrentCDFsValid {
		initialTileResidualCDFs = &s.tileResidualCurrentCDFs
		retainedTileResidualCDFs = &s.tileResidualRetainedCDFs
		retainedTileResidualCDFsValid = &s.tileResidualRetainedCDFsValid
	}
	cdefIndexMap, loopFilterMap, restorationFrameBuffers := s.postFilterSideData()
	executed, err := executeFrameWorkStepWithPayload(step, workerPool, output, references, payload, validatePayload, frameContext, event.FrameHeader.DisableCDFUpdate, initialTileResidualCDFs, retainedTileResidualCDFs, retainedTileResidualCDFsValid, cdefIndexMap, loopFilterMap, jobs, batches, fn)
	if err != nil {
		return FrameWorkStepResult{}, err
	}
	if output == nil {
		output, err = frameWorkPostFilterOutput(event, framePool, step, post)
		if err != nil {
			return FrameWorkStepResult{ExecutedTileWork: executed}, err
		}
	}
	if err := runFrameWorkPostFilter(event, step, output, referenceCount, executed, cdefIndexMap, loopFilterMap, restorationFrameBuffers, post); err != nil {
		return FrameWorkStepResult{ExecutedTileWork: executed}, err
	}
	completed, releaseCount, err := s.FinishIfEventCompletesFrameWork(refs, framePool, event, releases)
	if err != nil {
		return FrameWorkStepResult{ExecutedTileWork: executed}, err
	}
	return FrameWorkStepResult{
		ExecutedTileWork: executed,
		CompletedFrame:   completed,
		ReleaseCount:     step.ReleaseCount + releaseCount,
	}, nil
}

func (s *FrameWorkState) runStepWithPayloadContextRunner(refs *SurfaceReferences, framePool *frame.Pool, event Event, step FrameWorkStep, workerPool *threading.Pool, output *frame.Frame, references []*frame.Frame, payload []byte, validatePayload bool, jobs []tile.Job, batches []threading.Batch, releases []int, runner threading.FrameWorkBatchRunner, post FrameWorkPostFilterFunc) (FrameWorkStepResult, error) {
	if !frameWorkStepMatchesEvent(event, step) {
		return FrameWorkStepResult{}, ErrInvalidFrameWorkStep
	}
	_, referenceCount, _, err := frameWorkStepTilePlan(step)
	if err != nil {
		return FrameWorkStepResult{}, err
	}
	frameContext := frameWorkFrameContext(event, s.sequenceContext())
	var initialTileResidualCDFs *threading.FrameWorkTileResidualCDFStorage
	var retainedTileResidualCDFs *threading.FrameWorkTileResidualCDFStorage
	var retainedTileResidualCDFsValid *bool
	if s != nil && s.tileResidualCurrentCDFsValid {
		initialTileResidualCDFs = &s.tileResidualCurrentCDFs
		retainedTileResidualCDFs = &s.tileResidualRetainedCDFs
		retainedTileResidualCDFsValid = &s.tileResidualRetainedCDFsValid
	}
	cdefIndexMap, loopFilterMap, restorationFrameBuffers := s.postFilterSideData()
	executed, err := executeFrameWorkStepWithPayloadRunner(step, workerPool, output, references, payload, validatePayload, frameContext, event.FrameHeader.DisableCDFUpdate, initialTileResidualCDFs, retainedTileResidualCDFs, retainedTileResidualCDFsValid, cdefIndexMap, loopFilterMap, jobs, batches, runner)
	if err != nil {
		return FrameWorkStepResult{}, err
	}
	if output == nil {
		output, err = frameWorkPostFilterOutput(event, framePool, step, post)
		if err != nil {
			return FrameWorkStepResult{ExecutedTileWork: executed}, err
		}
	}
	if err := runFrameWorkPostFilter(event, step, output, referenceCount, executed, cdefIndexMap, loopFilterMap, restorationFrameBuffers, post); err != nil {
		return FrameWorkStepResult{ExecutedTileWork: executed}, err
	}
	completed, releaseCount, err := s.FinishIfEventCompletesFrameWork(refs, framePool, event, releases)
	if err != nil {
		return FrameWorkStepResult{ExecutedTileWork: executed}, err
	}
	return FrameWorkStepResult{
		ExecutedTileWork: executed,
		CompletedFrame:   completed,
		ReleaseCount:     step.ReleaseCount + releaseCount,
	}, nil
}

func (s *FrameWorkState) runStepWithPayloadContextRunners(refs *SurfaceReferences, framePool *frame.Pool, event Event, step FrameWorkStep, workerPool *threading.Pool, output *frame.Frame, references []*frame.Frame, payload []byte, validatePayload bool, jobs []tile.Job, batches []threading.Batch, releases []int, runner threading.FrameWorkBatchRunner, post FrameWorkPostFilterRunner) (FrameWorkStepResult, error) {
	if !frameWorkStepMatchesEvent(event, step) {
		return FrameWorkStepResult{}, ErrInvalidFrameWorkStep
	}
	_, referenceCount, _, err := frameWorkStepTilePlan(step)
	if err != nil {
		return FrameWorkStepResult{}, err
	}
	frameContext := frameWorkFrameContext(event, s.sequenceContext())
	var initialTileResidualCDFs *threading.FrameWorkTileResidualCDFStorage
	var retainedTileResidualCDFs *threading.FrameWorkTileResidualCDFStorage
	var retainedTileResidualCDFsValid *bool
	if s != nil && s.tileResidualCurrentCDFsValid {
		initialTileResidualCDFs = &s.tileResidualCurrentCDFs
		retainedTileResidualCDFs = &s.tileResidualRetainedCDFs
		retainedTileResidualCDFsValid = &s.tileResidualRetainedCDFsValid
	}
	cdefIndexMap, loopFilterMap, restorationFrameBuffers := s.postFilterSideData()
	executed, err := executeFrameWorkStepWithPayloadRunner(step, workerPool, output, references, payload, validatePayload, frameContext, event.FrameHeader.DisableCDFUpdate, initialTileResidualCDFs, retainedTileResidualCDFs, retainedTileResidualCDFsValid, cdefIndexMap, loopFilterMap, jobs, batches, runner)
	if err != nil {
		return FrameWorkStepResult{}, err
	}
	if output == nil {
		output, err = frameWorkPostFilterRunnerOutput(event, framePool, step, post)
		if err != nil {
			return FrameWorkStepResult{ExecutedTileWork: executed}, err
		}
	}
	if err := runFrameWorkPostFilterRunner(event, step, output, referenceCount, executed, cdefIndexMap, loopFilterMap, restorationFrameBuffers, post); err != nil {
		return FrameWorkStepResult{ExecutedTileWork: executed}, err
	}
	completed, releaseCount, err := s.FinishIfEventCompletesFrameWork(refs, framePool, event, releases)
	if err != nil {
		return FrameWorkStepResult{ExecutedTileWork: executed}, err
	}
	return FrameWorkStepResult{
		ExecutedTileWork: executed,
		CompletedFrame:   completed,
		ReleaseCount:     step.ReleaseCount + releaseCount,
	}, nil
}

// RunEventWithContext plans event, resolves any output/reference frame
// pointers needed by tile work, executes it, and finishes final tile groups.
// All working storage is caller-owned.
func (s *FrameWorkState) RunEventWithContext(refs *SurfaceReferences, framePool *frame.Pool, sequence parser.SequenceHeader, event Event, align int, referenceSurfaces []int, referenceFrames []*frame.Frame, workers int, spans []parser.TileSpan, jobs []tile.Job, batches []threading.Batch, releases []int, workerPool *threading.Pool, fn FrameWorkBatchFunc) (FrameWorkEventResult, error) {
	return s.RunEventWithContextAndPostFilter(refs, framePool, sequence, event, align, referenceSurfaces, referenceFrames, workers, spans, jobs, batches, releases, workerPool, fn, nil)
}

// RunEventWithContextRunner is RunEventWithContext using a frame-work batch
// runner directly instead of a callback adapter.
func (s *FrameWorkState) RunEventWithContextRunner(refs *SurfaceReferences, framePool *frame.Pool, sequence parser.SequenceHeader, event Event, align int, referenceSurfaces []int, referenceFrames []*frame.Frame, workers int, spans []parser.TileSpan, jobs []tile.Job, batches []threading.Batch, releases []int, workerPool *threading.Pool, runner threading.FrameWorkBatchRunner) (FrameWorkEventResult, error) {
	return s.RunEventWithContextAndPostFilterRunner(refs, framePool, sequence, event, align, referenceSurfaces, referenceFrames, workers, spans, jobs, batches, releases, workerPool, runner, nil)
}

// RunEventWithContextAndSideDataRunner is RunEventWithContextRunner with a
// side-data hook that runs after planning and before tile work executes.
func (s *FrameWorkState) RunEventWithContextAndSideDataRunner(refs *SurfaceReferences, framePool *frame.Pool, sequence parser.SequenceHeader, event Event, align int, referenceSurfaces []int, referenceFrames []*frame.Frame, workers int, spans []parser.TileSpan, jobs []tile.Job, batches []threading.Batch, releases []int, workerPool *threading.Pool, side FrameWorkSideDataRunner, runner threading.FrameWorkBatchRunner) (FrameWorkEventResult, error) {
	return s.RunEventWithContextAndSideDataAndPostFilterRunners(refs, framePool, sequence, event, align, referenceSurfaces, referenceFrames, workers, spans, jobs, batches, releases, workerPool, side, runner, nil)
}

// RunEventWithContextAndPostFilter is RunEventWithContext with a final-frame
// postfilter hook that runs after tile work and before reference publication.
func (s *FrameWorkState) RunEventWithContextAndPostFilter(refs *SurfaceReferences, framePool *frame.Pool, sequence parser.SequenceHeader, event Event, align int, referenceSurfaces []int, referenceFrames []*frame.Frame, workers int, spans []parser.TileSpan, jobs []tile.Job, batches []threading.Batch, releases []int, workerPool *threading.Pool, fn FrameWorkBatchFunc, post FrameWorkPostFilterFunc) (FrameWorkEventResult, error) {
	step, output, referenceCount, references, err := s.planEventWithResolvedContext(refs, framePool, sequence, event, align, referenceSurfaces, referenceFrames, workers, spans, jobs, batches, releases)
	if err != nil {
		return FrameWorkEventResult{}, err
	}

	run, err := s.RunStepWithPayloadContextAndPostFilter(refs, framePool, event, step, workerPool, output, references, event.Unit.Payload, jobs, batches, releases, fn, post)
	if err != nil {
		return FrameWorkEventResult{}, err
	}
	return FrameWorkEventResult{
		Step:           step,
		Output:         output,
		ReferenceCount: referenceCount,
		Run:            run,
	}, nil
}

// RunEventWithContextAndPostFilterRunner is RunEventWithContextAndPostFilter
// using a frame-work batch runner directly instead of a callback adapter.
func (s *FrameWorkState) RunEventWithContextAndPostFilterRunner(refs *SurfaceReferences, framePool *frame.Pool, sequence parser.SequenceHeader, event Event, align int, referenceSurfaces []int, referenceFrames []*frame.Frame, workers int, spans []parser.TileSpan, jobs []tile.Job, batches []threading.Batch, releases []int, workerPool *threading.Pool, runner threading.FrameWorkBatchRunner, post FrameWorkPostFilterFunc) (FrameWorkEventResult, error) {
	step, output, referenceCount, references, err := s.planEventWithResolvedContext(refs, framePool, sequence, event, align, referenceSurfaces, referenceFrames, workers, spans, jobs, batches, releases)
	if err != nil {
		return FrameWorkEventResult{}, err
	}

	run, err := s.RunStepWithPayloadContextAndPostFilterRunner(refs, framePool, event, step, workerPool, output, references, event.Unit.Payload, jobs, batches, releases, runner, post)
	if err != nil {
		return FrameWorkEventResult{}, err
	}
	return FrameWorkEventResult{
		Step:           step,
		Output:         output,
		ReferenceCount: referenceCount,
		Run:            run,
	}, nil
}

// RunEventWithContextAndPostFilterRunners is
// RunEventWithContextAndPostFilterRunner using direct runners for both tile
// batches and final-frame postfilters.
func (s *FrameWorkState) RunEventWithContextAndPostFilterRunners(refs *SurfaceReferences, framePool *frame.Pool, sequence parser.SequenceHeader, event Event, align int, referenceSurfaces []int, referenceFrames []*frame.Frame, workers int, spans []parser.TileSpan, jobs []tile.Job, batches []threading.Batch, releases []int, workerPool *threading.Pool, runner threading.FrameWorkBatchRunner, post FrameWorkPostFilterRunner) (FrameWorkEventResult, error) {
	step, output, referenceCount, references, err := s.planEventWithResolvedContext(refs, framePool, sequence, event, align, referenceSurfaces, referenceFrames, workers, spans, jobs, batches, releases)
	if err != nil {
		return FrameWorkEventResult{}, err
	}

	run, err := s.RunStepWithPayloadContextAndPostFilterRunners(refs, framePool, event, step, workerPool, output, references, event.Unit.Payload, jobs, batches, releases, runner, post)
	if err != nil {
		return FrameWorkEventResult{}, err
	}
	return FrameWorkEventResult{
		Step:           step,
		Output:         output,
		ReferenceCount: referenceCount,
		Run:            run,
	}, nil
}

// RunEventWithContextAndSideDataAndPostFilterRunners is
// RunEventWithContextAndPostFilterRunners with a side-data hook that runs after
// planning and before tile work executes.
func (s *FrameWorkState) RunEventWithContextAndSideDataAndPostFilterRunners(refs *SurfaceReferences, framePool *frame.Pool, sequence parser.SequenceHeader, event Event, align int, referenceSurfaces []int, referenceFrames []*frame.Frame, workers int, spans []parser.TileSpan, jobs []tile.Job, batches []threading.Batch, releases []int, workerPool *threading.Pool, side FrameWorkSideDataRunner, runner threading.FrameWorkBatchRunner, post FrameWorkPostFilterRunner) (FrameWorkEventResult, error) {
	step, output, referenceCount, references, err := s.planEventWithResolvedContext(refs, framePool, sequence, event, align, referenceSurfaces, referenceFrames, workers, spans, jobs, batches, releases)
	if err != nil {
		return FrameWorkEventResult{}, err
	}
	if err := s.bindFrameWorkEventSideData(event, step, output, references, event.Unit.Payload, side); err != nil {
		return FrameWorkEventResult{}, err
	}

	run, err := s.RunStepWithPayloadContextAndPostFilterRunners(refs, framePool, event, step, workerPool, output, references, event.Unit.Payload, jobs, batches, releases, runner, post)
	if err != nil {
		return FrameWorkEventResult{}, err
	}
	return FrameWorkEventResult{
		Step:           step,
		Output:         output,
		ReferenceCount: referenceCount,
		Run:            run,
	}, nil
}

func (s *FrameWorkState) planEventWithResolvedContext(refs *SurfaceReferences, framePool *frame.Pool, sequence parser.SequenceHeader, event Event, align int, referenceSurfaces []int, referenceFrames []*frame.Frame, workers int, spans []parser.TileSpan, jobs []tile.Job, batches []threading.Batch, releases []int) (FrameWorkStep, *frame.Frame, int, []*frame.Frame, error) {
	step, output, err := s.PlanEvent(refs, framePool, sequence, event, align, referenceSurfaces, workers, spans, jobs, batches, releases)
	if err != nil {
		return FrameWorkStep{}, nil, 0, nil, err
	}

	plan, referenceCount, hasTile, err := frameWorkStepTilePlan(step)
	if err != nil {
		return FrameWorkStep{}, nil, 0, nil, err
	}
	if step.Kind == FrameWorkStepShowExisting {
		output, err = framePool.Frame(step.ShowExisting.Surface)
		if err != nil {
			return FrameWorkStep{}, nil, 0, nil, err
		}
	}

	var references []*frame.Frame
	if hasTile && plan.JobCount != 0 {
		if output == nil {
			surface, err := frameWorkStepSurface(step)
			if err != nil {
				return FrameWorkStep{}, nil, 0, nil, err
			}
			output, err = framePool.Frame(surface)
			if err != nil {
				return FrameWorkStep{}, nil, 0, nil, err
			}
		}
		if referenceCount != 0 {
			if step.Kind == FrameWorkStepTile {
				count, err := refs.FrameReferences(event, referenceSurfaces)
				if err != nil {
					return FrameWorkStep{}, nil, 0, nil, err
				}
				if count != referenceCount {
					return FrameWorkStep{}, nil, 0, nil, ErrInvalidFrameWorkStep
				}
			}
			count, err := ResolveFrameReferences(framePool, referenceSurfaces[:referenceCount], referenceFrames)
			if err != nil {
				return FrameWorkStep{}, nil, 0, nil, err
			}
			if count != referenceCount {
				return FrameWorkStep{}, nil, 0, nil, ErrInvalidFrameWorkStep
			}
			references = referenceFrames[:referenceCount]
		}
	}
	return step, output, referenceCount, references, nil
}

// BindFrameWorkSideData implements FrameWorkSideDataRunner for function hooks.
func (fn FrameWorkSideDataFunc) BindFrameWorkSideData(s *FrameWorkState, b FrameWorkBatch) error {
	if fn == nil {
		return nil
	}
	return fn(s, b)
}

// FrameWorkSideDataScratchLen reports the caller-owned side-data scratch needed
// for active postfilter stages in b.
func FrameWorkSideDataScratchLen(b FrameWorkBatch) (FrameWorkSideDataScratchSize, error) {
	var size FrameWorkSideDataScratchSize
	if frameWorkCDEFActive(b.CDEF) {
		_, _, length, err := b.CDEFIndexMapShape()
		if err != nil {
			return FrameWorkSideDataScratchSize{}, err
		}
		size.CDEF = length
	}
	if frameWorkLoopFilterActive(b.LoopFilter) {
		_, _, length, err := b.LoopFilterMapShape()
		if err != nil {
			return FrameWorkSideDataScratchSize{}, err
		}
		size.LoopFilterRecords = length
	}
	if frameWorkRestorationActive(b.Restoration) {
		plan, err := b.RestorationFramePlan()
		if err != nil {
			return FrameWorkSideDataScratchSize{}, err
		}
		size.RestorationRecords = plan.UnitRecordLen()
		size.RestorationBoundary = plan.BoundaryBufferLen()
	}
	return size, nil
}

// BindRunner validates and slices caller-owned scratch for
// FrameWorkBoundSideDataRunner.
func (s FrameWorkSideDataScratchSize) BindRunner(scratch FrameWorkSideDataScratch) (FrameWorkBoundSideDataRunner, error) {
	if s.CDEF < 0 || s.LoopFilterRecords < 0 || s.RestorationRecords < 0 || s.RestorationBoundary < 0 ||
		len(scratch.CDEFIndex) < s.CDEF || len(scratch.CDEFRead) < s.CDEF ||
		len(scratch.LoopFilterRecords) < s.LoopFilterRecords ||
		len(scratch.RestorationRecords) < s.RestorationRecords ||
		len(scratch.RestorationAbove) < s.RestorationBoundary ||
		len(scratch.RestorationBelow) < s.RestorationBoundary {
		return FrameWorkBoundSideDataRunner{}, frame.ErrShortBuffer
	}
	return FrameWorkBoundSideDataRunner{
		CDEFIndex:          scratch.CDEFIndex[:s.CDEF],
		CDEFRead:           scratch.CDEFRead[:s.CDEF],
		LoopFilterRecords:  scratch.LoopFilterRecords[:s.LoopFilterRecords],
		RestorationRecords: scratch.RestorationRecords[:s.RestorationRecords],
		RestorationAbove:   scratch.RestorationAbove[:s.RestorationBoundary],
		RestorationBelow:   scratch.RestorationBelow[:s.RestorationBoundary],
	}, nil
}

// BindFrameWorkSideData binds side-data storage for any active CDEF,
// loop-filter, or loop-restoration stage and attaches the views to state.
func (r *FrameWorkBoundSideDataRunner) BindFrameWorkSideData(s *FrameWorkState, b FrameWorkBatch) error {
	if r == nil {
		return ErrInvalidFrameWorkState
	}
	if frameWorkCDEFActive(b.CDEF) {
		_, _, length, err := b.CDEFIndexMapShape()
		if err != nil {
			return err
		}
		if len(r.CDEFIndex) < length || len(r.CDEFRead) < length {
			return threading.ErrInvalidBatch
		}
		cdefMap, err := b.BindCDEFIndexMap(r.CDEFIndex, r.CDEFRead)
		if err != nil {
			return err
		}
		if err := s.SetCDEFIndexMap(cdefMap); err != nil {
			return err
		}
		r.CDEFIndexMap = cdefMap
	}
	if frameWorkLoopFilterActive(b.LoopFilter) {
		_, _, length, err := b.LoopFilterMapShape()
		if err != nil {
			return err
		}
		if len(r.LoopFilterRecords) < length {
			return threading.ErrInvalidBatch
		}
		lfMap, err := b.BindLoopFilterMap(r.LoopFilterRecords)
		if err != nil {
			return err
		}
		if err := s.SetLoopFilterMap(lfMap); err != nil {
			return err
		}
		r.LoopFilterMap = lfMap
	}
	if frameWorkRestorationActive(b.Restoration) {
		buffers, err := b.BindRestorationFrameBuffers(r.RestorationRecords, r.RestorationAbove, r.RestorationBelow)
		if err != nil {
			return err
		}
		if err := s.SetRestorationFrameBuffers(buffers); err != nil {
			return err
		}
		r.RestorationFrameBuffers = buffers
	}
	return nil
}

func (s *FrameWorkState) bindFrameWorkEventSideData(event Event, step FrameWorkStep, output *frame.Frame, references []*frame.Frame, payload []byte, side FrameWorkSideDataRunner) error {
	if side == nil {
		return nil
	}
	plan, referenceCount, hasTile, err := frameWorkStepTilePlan(step)
	if err != nil {
		return err
	}
	if !hasTile || plan.JobCount == 0 {
		return nil
	}
	if referenceCount < 0 || referenceCount > parser.InterRefsPerFrame {
		return ErrInvalidTileWork
	}
	if len(references) < referenceCount {
		return ErrSurfaceReferenceBufferTooSmall
	}
	ctx := FrameWorkBatch{
		Step:                  step,
		Output:                output,
		Payload:               payload,
		References:            references[:referenceCount],
		FrameWorkFrameContext: frameWorkFrameContext(event, s.sequenceContext()),
		DisableCDFUpdate:      event.FrameHeader.DisableCDFUpdate,
	}
	return side.BindFrameWorkSideData(s, ctx)
}

func frameWorkPostFilterOutput(event Event, pool *frame.Pool, step FrameWorkStep, post FrameWorkPostFilterFunc) (*frame.Frame, error) {
	if post == nil || !EventCompletesFrameWork(event) {
		return nil, nil
	}
	return frameWorkStepOutput(pool, step)
}

func frameWorkPostFilterRunnerOutput(event Event, pool *frame.Pool, step FrameWorkStep, post FrameWorkPostFilterRunner) (*frame.Frame, error) {
	if post == nil || !EventCompletesFrameWork(event) {
		return nil, nil
	}
	return frameWorkStepOutput(pool, step)
}

func frameWorkStepOutput(pool *frame.Pool, step FrameWorkStep) (*frame.Frame, error) {
	surface, err := frameWorkStepSurface(step)
	if err != nil {
		return nil, err
	}
	if pool == nil {
		return nil, frame.ErrInvalidPool
	}
	return pool.Frame(surface)
}

func (s *FrameWorkState) postFilterSideData() (*threading.FrameWorkCDEFIndexMap, *threading.FrameWorkLoopFilterMap, *threading.FrameWorkRestorationFrameBuffers) {
	if s == nil {
		return nil, nil, nil
	}
	var cdefIndexMap *threading.FrameWorkCDEFIndexMap
	var loopFilterMap *threading.FrameWorkLoopFilterMap
	var restorationFrameBuffers *threading.FrameWorkRestorationFrameBuffers
	if s.cdefIndexMapValid {
		cdefIndexMap = &s.cdefIndexMap
	}
	if s.loopFilterMapValid {
		loopFilterMap = &s.loopFilterMap
	}
	if s.restorationFrameBuffersValid {
		restorationFrameBuffers = &s.restorationFrameBuffers
	}
	return cdefIndexMap, loopFilterMap, restorationFrameBuffers
}

func runFrameWorkPostFilter(event Event, step FrameWorkStep, output *frame.Frame, referenceCount int, executed bool, cdefIndexMap *threading.FrameWorkCDEFIndexMap, loopFilterMap *threading.FrameWorkLoopFilterMap, restorationFrameBuffers *threading.FrameWorkRestorationFrameBuffers, post FrameWorkPostFilterFunc) error {
	if post == nil || !EventCompletesFrameWork(event) {
		return nil
	}
	if output == nil {
		return frame.ErrInvalidSlot
	}
	return post(FrameWorkPostFilterContext{
		Event:                   event,
		Step:                    step,
		Output:                  output,
		ReferenceCount:          referenceCount,
		ExecutedTileWork:        executed,
		CDEFIndexMap:            cdefIndexMap,
		LoopFilterMap:           loopFilterMap,
		RestorationFrameBuffers: restorationFrameBuffers,
	})
}

func runFrameWorkPostFilterRunner(event Event, step FrameWorkStep, output *frame.Frame, referenceCount int, executed bool, cdefIndexMap *threading.FrameWorkCDEFIndexMap, loopFilterMap *threading.FrameWorkLoopFilterMap, restorationFrameBuffers *threading.FrameWorkRestorationFrameBuffers, post FrameWorkPostFilterRunner) error {
	if post == nil || !EventCompletesFrameWork(event) {
		return nil
	}
	if output == nil {
		return frame.ErrInvalidSlot
	}
	return post.Apply(FrameWorkPostFilterContext{
		Event:                   event,
		Step:                    step,
		Output:                  output,
		ReferenceCount:          referenceCount,
		ExecutedTileWork:        executed,
		CDEFIndexMap:            cdefIndexMap,
		LoopFilterMap:           loopFilterMap,
		RestorationFrameBuffers: restorationFrameBuffers,
	})
}

// ExecuteTileWork dispatches the planned tile jobs through a reusable worker
// pool. Only the caller-owned job and batch ranges named by plan are used.
func ExecuteTileWork(plan TileWorkPlan, pool *threading.Pool, jobs []tile.Job, batches []threading.Batch, fn threading.BatchFunc) error {
	if err := validateTileWorkPlan(plan, jobs, batches); err != nil {
		return err
	}
	if plan.JobCount == 0 {
		return nil
	}
	return pool.Execute(batches[:plan.BatchCount], jobs[:plan.JobCount], fn)
}

// ExecuteFrameWorkStep dispatches any tile work carried by step. Steps that do
// not carry tile work are successful no-ops; the boolean reports whether tile
// work actually ran.
func ExecuteFrameWorkStep(step FrameWorkStep, pool *threading.Pool, jobs []tile.Job, batches []threading.Batch, fn threading.BatchFunc) (bool, error) {
	plan, _, hasTile, err := frameWorkStepTilePlan(step)
	if err != nil {
		return false, err
	}
	if !hasTile {
		return false, nil
	}

	if err := ExecuteTileWork(plan, pool, jobs, batches, fn); err != nil {
		return false, err
	}
	return plan.JobCount != 0, nil
}

// ExecuteFrameWorkStepWithContext dispatches frame-work tile batches while
// passing the output frame and resolved reference frames to each batch.
func ExecuteFrameWorkStepWithContext(step FrameWorkStep, pool *threading.Pool, output *frame.Frame, references []*frame.Frame, jobs []tile.Job, batches []threading.Batch, fn FrameWorkBatchFunc) (bool, error) {
	return executeFrameWorkStepWithPayload(step, pool, output, references, nil, false, FrameWorkFrameContext{}, false, nil, nil, nil, nil, nil, jobs, batches, fn)
}

// ExecuteFrameWorkStepWithPayload dispatches frame-work tile batches while
// passing the output frame, tile-group payload, and resolved reference frames
// to each batch.
func ExecuteFrameWorkStepWithPayload(step FrameWorkStep, pool *threading.Pool, output *frame.Frame, references []*frame.Frame, payload []byte, jobs []tile.Job, batches []threading.Batch, fn FrameWorkBatchFunc) (bool, error) {
	return executeFrameWorkStepWithPayload(step, pool, output, references, payload, true, FrameWorkFrameContext{}, false, nil, nil, nil, nil, nil, jobs, batches, fn)
}

func executeFrameWorkStepWithPayload(step FrameWorkStep, pool *threading.Pool, output *frame.Frame, references []*frame.Frame, payload []byte, validatePayload bool, frameContext FrameWorkFrameContext, disableCDFUpdate bool, initialTileResidualCDFs *threading.FrameWorkTileResidualCDFStorage, retainedTileResidualCDFs *threading.FrameWorkTileResidualCDFStorage, retainedTileResidualCDFsValid *bool, cdefIndexMap *threading.FrameWorkCDEFIndexMap, loopFilterMap *threading.FrameWorkLoopFilterMap, jobs []tile.Job, batches []threading.Batch, fn FrameWorkBatchFunc) (bool, error) {
	plan, referenceCount, hasTile, err := frameWorkStepTilePlan(step)
	if err != nil {
		return false, err
	}
	if !hasTile {
		return false, nil
	}
	if err := validateTileWorkPlan(plan, jobs, batches); err != nil {
		return false, err
	}
	if plan.JobCount == 0 {
		return false, nil
	}
	if referenceCount < 0 || referenceCount > parser.InterRefsPerFrame {
		return false, ErrInvalidTileWork
	}
	if len(references) < referenceCount {
		return false, ErrSurfaceReferenceBufferTooSmall
	}
	if validatePayload {
		if err := tile.ValidatePayloads(payload, jobs[:plan.JobCount]); err != nil {
			return false, err
		}
	}

	base := FrameWorkBatch{
		Step:                          step,
		Output:                        output,
		Payload:                       payload,
		References:                    references[:referenceCount],
		FrameWorkFrameContext:         frameContext,
		DisableCDFUpdate:              disableCDFUpdate,
		InitialTileResidualCDFs:       initialTileResidualCDFs,
		RetainedTileResidualCDFs:      retainedTileResidualCDFs,
		RetainedTileResidualCDFsValid: retainedTileResidualCDFsValid,
		CDEFIndexMap:                  cdefIndexMap,
		LoopFilterMap:                 loopFilterMap,
	}
	err = pool.ExecuteFrameWork(batches[:plan.BatchCount], jobs[:plan.JobCount], base, fn)
	if err != nil {
		return false, err
	}
	return true, nil
}

func executeFrameWorkStepWithPayloadRunner(step FrameWorkStep, pool *threading.Pool, output *frame.Frame, references []*frame.Frame, payload []byte, validatePayload bool, frameContext FrameWorkFrameContext, disableCDFUpdate bool, initialTileResidualCDFs *threading.FrameWorkTileResidualCDFStorage, retainedTileResidualCDFs *threading.FrameWorkTileResidualCDFStorage, retainedTileResidualCDFsValid *bool, cdefIndexMap *threading.FrameWorkCDEFIndexMap, loopFilterMap *threading.FrameWorkLoopFilterMap, jobs []tile.Job, batches []threading.Batch, runner threading.FrameWorkBatchRunner) (bool, error) {
	plan, referenceCount, hasTile, err := frameWorkStepTilePlan(step)
	if err != nil {
		return false, err
	}
	if !hasTile {
		return false, nil
	}
	if err := validateTileWorkPlan(plan, jobs, batches); err != nil {
		return false, err
	}
	if plan.JobCount == 0 {
		return false, nil
	}
	if referenceCount < 0 || referenceCount > parser.InterRefsPerFrame {
		return false, ErrInvalidTileWork
	}
	if len(references) < referenceCount {
		return false, ErrSurfaceReferenceBufferTooSmall
	}
	if validatePayload {
		if err := tile.ValidatePayloads(payload, jobs[:plan.JobCount]); err != nil {
			return false, err
		}
	}

	base := FrameWorkBatch{
		Step:                          step,
		Output:                        output,
		Payload:                       payload,
		References:                    references[:referenceCount],
		FrameWorkFrameContext:         frameContext,
		DisableCDFUpdate:              disableCDFUpdate,
		InitialTileResidualCDFs:       initialTileResidualCDFs,
		RetainedTileResidualCDFs:      retainedTileResidualCDFs,
		RetainedTileResidualCDFsValid: retainedTileResidualCDFsValid,
		CDEFIndexMap:                  cdefIndexMap,
		LoopFilterMap:                 loopFilterMap,
	}
	err = pool.ExecuteFrameWorkRunner(batches[:plan.BatchCount], jobs[:plan.JobCount], base, runner)
	if err != nil {
		return false, err
	}
	return true, nil
}

func frameWorkFrameContext(event Event, fallback threading.FrameWorkSequenceContext) FrameWorkFrameContext {
	sequence := threading.FrameWorkSequenceContextFromHeader(event.SequenceHeader)
	if !sequence.Valid() {
		sequence = fallback
	}
	return FrameWorkFrameContext{
		Sequence:            sequence,
		FrameHeader:         event.FrameHeader,
		FrameSize:           event.FrameSize,
		TileInfo:            event.TileInfo,
		Quantization:        event.Quantization,
		Segmentation:        event.Segmentation,
		Delta:               event.Delta,
		LoopFilter:          event.LoopFilter,
		CDEF:                event.CDEF,
		Restoration:         event.Restoration,
		TransformRef:        event.TransformRef,
		SkipMode:            event.SkipMode,
		FrameMode:           event.FrameMode,
		GlobalMotion:        event.GlobalMotion,
		FilmGrain:           event.FilmGrain,
		ReferenceOrderHints: event.ReferenceOrderHints,
	}
}

func (s *FrameWorkState) sequenceContext() threading.FrameWorkSequenceContext {
	if s == nil {
		return threading.FrameWorkSequenceContext{}
	}
	return s.Sequence
}

func frameWorkLoopFilterActive(lf parser.LoopFilterParams) bool {
	return lf.LevelY[0] != 0 || lf.LevelY[1] != 0 || lf.LevelU != 0 || lf.LevelV != 0
}

func frameWorkCDEFActive(cdef parser.CDEFParams) bool {
	return cdef.Bits != 0 || cdef.YStrength[0] != 0 || cdef.UVStrength[0] != 0
}

func frameWorkRestorationActive(restoration parser.RestorationParams) bool {
	return restoration.Type[0] != parser.RestorationNone ||
		restoration.Type[1] != parser.RestorationNone ||
		restoration.Type[2] != parser.RestorationNone
}

// BeginFrameWork resolves frame references, plans any inline tile work, and
// acquires the output frame surface. For frame OBUs, tile work is validated
// before acquiring a pool slot so malformed tile data leaves surface ownership
// untouched.
func BeginFrameWork(refs *SurfaceReferences, pool *frame.Pool, sequence parser.SequenceHeader, event Event, align int, references []int, workers int, spans []parser.TileSpan, jobs []tile.Job, batches []threading.Batch) (FrameWorkPlan, *frame.Frame, error) {
	if event.Kind != EventFrameHeader && event.Kind != EventFrame {
		return FrameWorkPlan{}, nil, ErrInvalidSurfaceEvent
	}

	var tilePlan TileWorkPlan
	var err error
	if event.Kind == EventFrame {
		tilePlan, err = PlanTileWork(event, workers, spans, jobs, batches)
		if err != nil {
			return FrameWorkPlan{}, nil, err
		}
	}

	surface, output, refCount, err := BeginFrameSurface(refs, pool, sequence, event, align, references)
	if err != nil {
		return FrameWorkPlan{}, nil, err
	}

	return FrameWorkPlan{
		Surface:        surface,
		ReferenceCount: refCount,
		Tile:           tilePlan,
	}, output, nil
}

// PlanFrameTileWork plans a later tile-group event for an already-begun frame.
// The caller carries surface and reference count from BeginFrameWork.
func PlanFrameTileWork(event Event, surface int, referenceCount int, workers int, spans []parser.TileSpan, jobs []tile.Job, batches []threading.Batch) (FrameTileWorkPlan, error) {
	if event.Kind != EventTileGroup {
		return FrameTileWorkPlan{}, ErrInvalidTileWork
	}
	if surface < 0 || referenceCount < 0 || referenceCount > parser.InterRefsPerFrame {
		return FrameTileWorkPlan{}, ErrInvalidTileWork
	}
	tilePlan, err := PlanTileWork(event, workers, spans, jobs, batches)
	if err != nil {
		return FrameTileWorkPlan{}, err
	}
	return FrameTileWorkPlan{
		Surface:        surface,
		ReferenceCount: referenceCount,
		Tile:           tilePlan,
	}, nil
}

// PlanTileWork turns an EventFrame or EventTileGroup into tile payload spans,
// tile jobs, and deterministic worker batches. All output storage is
// caller-owned.
func PlanTileWork(event Event, workers int, spans []parser.TileSpan, jobs []tile.Job, batches []threading.Batch) (TileWorkPlan, error) {
	if event.Kind != EventFrame && event.Kind != EventTileGroup {
		return TileWorkPlan{}, ErrInvalidTileWork
	}

	spanCount, err := parser.SplitTileGroup(event.Unit.Payload, event.TileInfo, event.TileGroup, spans)
	if err != nil {
		return TileWorkPlan{}, err
	}
	jobCount, err := tile.BuildJobs(jobs, event.TileInfo, spans[:spanCount])
	if err != nil {
		return TileWorkPlan{}, err
	}
	batchCount, err := threading.BuildBatches(batches, jobs[:jobCount], workers)
	if err != nil {
		return TileWorkPlan{}, err
	}
	return TileWorkPlan{
		SpanCount:  spanCount,
		JobCount:   jobCount,
		BatchCount: batchCount,
	}, nil
}

func validateTileWorkPlan(plan TileWorkPlan, jobs []tile.Job, batches []threading.Batch) error {
	if plan.SpanCount < 0 ||
		plan.JobCount < 0 ||
		plan.BatchCount < 0 ||
		plan.SpanCount != plan.JobCount ||
		plan.JobCount > len(jobs) ||
		plan.BatchCount > len(batches) ||
		plan.BatchCount > plan.JobCount {
		return ErrInvalidTileWork
	}
	if plan.JobCount == 0 {
		if plan.BatchCount != 0 {
			return ErrInvalidTileWork
		}
		return nil
	}
	if plan.BatchCount == 0 {
		return ErrInvalidTileWork
	}
	return nil
}

func frameWorkStepTilePlan(step FrameWorkStep) (TileWorkPlan, int, bool, error) {
	switch step.Kind {
	case FrameWorkStepIgnored, FrameWorkStepDropped, FrameWorkStepShowExisting:
		return TileWorkPlan{}, 0, false, nil
	case FrameWorkStepBegin:
		return step.Begin.Tile, step.Begin.ReferenceCount, true, nil
	case FrameWorkStepTile:
		return step.Tile.Tile, step.Tile.ReferenceCount, true, nil
	default:
		return TileWorkPlan{}, 0, false, ErrInvalidTileWork
	}
}

func frameWorkStepSurface(step FrameWorkStep) (int, error) {
	switch step.Kind {
	case FrameWorkStepBegin:
		return step.Begin.Surface, nil
	case FrameWorkStepTile:
		return step.Tile.Surface, nil
	case FrameWorkStepShowExisting:
		return step.ShowExisting.Surface, nil
	default:
		return -1, ErrInvalidFrameWorkStep
	}
}

func frameWorkStepMatchesEvent(event Event, step FrameWorkStep) bool {
	switch event.Kind {
	case EventFrameHeader, EventFrame:
		return step.Kind == FrameWorkStepBegin
	case EventTileGroup:
		return step.Kind == FrameWorkStepTile
	case EventExistingFrame:
		return step.Kind == FrameWorkStepShowExisting
	}
	if EventDropsFrameWork(event) {
		return step.Kind == FrameWorkStepDropped || step.Kind == FrameWorkStepIgnored
	}
	return step.Kind == FrameWorkStepIgnored
}
