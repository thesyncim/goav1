package goav1

import (
	internaldecoder "github.com/thesyncim/goav1/internal/av1/decoder"
	internalthreading "github.com/thesyncim/goav1/internal/av1/threading"
	internaltile "github.com/thesyncim/goav1/internal/av1/tile"
)

type DecoderStream = internaldecoder.Stream
type DecoderEvent = internaldecoder.Event
type DecoderEventKind = internaldecoder.EventKind
type DecoderTileWorkPlan = internaldecoder.TileWorkPlan
type DecoderFrameWorkPlan = internaldecoder.FrameWorkPlan
type DecoderFrameTileWorkPlan = internaldecoder.FrameTileWorkPlan
type DecoderShowExistingFrameWorkPlan = internaldecoder.ShowExistingFrameWorkPlan
type DecoderFrameWorkStepKind = internaldecoder.FrameWorkStepKind
type DecoderFrameWorkStep = internaldecoder.FrameWorkStep
type DecoderFrameWorkStepResult = internaldecoder.FrameWorkStepResult
type DecoderFrameWorkBatch = internaldecoder.FrameWorkBatch
type DecoderFrameWorkSequenceContext = internaldecoder.FrameWorkSequenceContext
type DecoderFrameWorkFrameContext = internaldecoder.FrameWorkFrameContext
type DecoderFrameWorkJobRegion = internaldecoder.FrameWorkJobRegion
type DecoderFrameWorkPlane = internaldecoder.FrameWorkPlane
type DecoderFrameWorkPlaneRegion = internaldecoder.FrameWorkPlaneRegion
type DecoderFrameWorkReference = internaldecoder.FrameWorkReference
type DecoderFrameWorkBatchFunc = internaldecoder.FrameWorkBatchFunc
type DecoderFrameWorkEventResult = internaldecoder.FrameWorkEventResult
type DecoderFrameWorkPostFilterContext = internaldecoder.FrameWorkPostFilterContext
type DecoderFrameWorkPostFilterFunc = internaldecoder.FrameWorkPostFilterFunc
type DecoderFrameWorkPostFilterStage = internaldecoder.FrameWorkPostFilterStage
type DecoderFrameWorkPostFilterScratchSize = internaldecoder.FrameWorkPostFilterScratchSize
type DecoderFrameWorkPostFilterRequest = internaldecoder.FrameWorkPostFilterRequest
type DecoderFrameWorkPostFilterResult = internaldecoder.FrameWorkPostFilterResult
type DecoderFrameWorkCallerPostFilterResult = internaldecoder.FrameWorkCallerPostFilterResult
type DecoderFrameWorkSupportedPostFilterRunner = internaldecoder.FrameWorkSupportedPostFilterRunner
type DecoderFrameWorkCDEFIndexMap = internaldecoder.FrameWorkCDEFIndexMap
type DecoderFrameWorkCDEFPostFilterScratchSize = internaldecoder.FrameWorkCDEFPostFilterScratchSize
type DecoderFrameWorkCDEFPostFilterRequest = internaldecoder.FrameWorkCDEFPostFilterRequest
type DecoderFrameWorkCDEFPostFilterResult = internaldecoder.FrameWorkCDEFPostFilterResult
type DecoderFrameWorkLoopFilterMap = internaldecoder.FrameWorkLoopFilterMap
type DecoderFrameWorkLoopFilterPostFilterRequest = internaldecoder.FrameWorkLoopFilterPostFilterRequest
type DecoderFrameWorkLoopFilterPostFilterEdge = internaldecoder.FrameWorkLoopFilterPostFilterEdge
type DecoderFrameWorkLoopFilterPostFilterScratchSize = internaldecoder.FrameWorkLoopFilterPostFilterScratchSize
type DecoderFrameWorkLoopFilterPostFilterLevelStats = internaldecoder.FrameWorkLoopFilterPostFilterLevelStats
type DecoderFrameWorkLoopFilterPostFilterPlan = internaldecoder.FrameWorkLoopFilterPostFilterPlan
type DecoderFrameWorkLoopFilterPostFilterApplyResult = internaldecoder.FrameWorkLoopFilterPostFilterApplyResult
type DecoderFrameWorkSuperResPostFilterPlanePlan = internaldecoder.FrameWorkSuperResPostFilterPlanePlan
type DecoderFrameWorkSuperResPostFilterPlan = internaldecoder.FrameWorkSuperResPostFilterPlan
type DecoderFrameWorkSuperResPostFilterScratchSize = internaldecoder.FrameWorkSuperResPostFilterScratchSize
type DecoderFrameWorkSuperResPostFilterRequest = internaldecoder.FrameWorkSuperResPostFilterRequest
type DecoderFrameWorkSuperResPostFilterResult = internaldecoder.FrameWorkSuperResPostFilterResult
type DecoderFrameWorkRestorationPostFilterRequest = internaldecoder.FrameWorkRestorationPostFilterRequest
type DecoderFrameWorkRestorationPostFilterScratchSize = internaldecoder.FrameWorkRestorationPostFilterScratchSize
type DecoderFrameWorkFilmGrainPostFilterPlanePlan = internaldecoder.FrameWorkFilmGrainPostFilterPlanePlan
type DecoderFrameWorkFilmGrainPostFilterPlan = internaldecoder.FrameWorkFilmGrainPostFilterPlan
type DecoderFrameWorkFilmGrainPostFilterScratchSize = internaldecoder.FrameWorkFilmGrainPostFilterScratchSize
type DecoderFrameWorkFilmGrainPostFilterRequest = internaldecoder.FrameWorkFilmGrainPostFilterRequest
type DecoderFrameWorkFilmGrainPostFilterResult = internaldecoder.FrameWorkFilmGrainPostFilterResult
type DecoderFrameWorkFilmGrainPostFilterScalingLUTs = internaldecoder.FrameWorkFilmGrainPostFilterScalingLUTs
type DecoderFrameWorkFilmGrainPostFilterLumaGrain = internaldecoder.FrameWorkFilmGrainPostFilterLumaGrain
type DecoderFrameWorkState = internaldecoder.FrameWorkState
type DecoderSurfaceReferences = internaldecoder.SurfaceReferences
type DecoderFrameWorkRestorationFrameBuffers = internalthreading.FrameWorkRestorationFrameBuffers
type TileJob = internaltile.Job
type TileBatch = internalthreading.Batch
type TileWorkerPool = internalthreading.Pool
type TileBatchFunc = internalthreading.BatchFunc

const (
	DecoderEventIgnored              DecoderEventKind = internaldecoder.EventIgnored
	DecoderEventSequenceHeader       DecoderEventKind = internaldecoder.EventSequenceHeader
	DecoderEventTemporalDelimiter    DecoderEventKind = internaldecoder.EventTemporalDelimiter
	DecoderEventFrameHeader          DecoderEventKind = internaldecoder.EventFrameHeader
	DecoderEventRedundantFrameHeader DecoderEventKind = internaldecoder.EventRedundantFrameHeader
	DecoderEventFrame                DecoderEventKind = internaldecoder.EventFrame
	DecoderEventTileGroup            DecoderEventKind = internaldecoder.EventTileGroup
	DecoderEventMetadata             DecoderEventKind = internaldecoder.EventMetadata
	DecoderEventTileList             DecoderEventKind = internaldecoder.EventTileList
	DecoderEventPadding              DecoderEventKind = internaldecoder.EventPadding
	DecoderEventReserved             DecoderEventKind = internaldecoder.EventReserved
	DecoderEventExistingFrame        DecoderEventKind = internaldecoder.EventExistingFrame

	DecoderFrameWorkStepIgnored      DecoderFrameWorkStepKind = internaldecoder.FrameWorkStepIgnored
	DecoderFrameWorkStepDropped      DecoderFrameWorkStepKind = internaldecoder.FrameWorkStepDropped
	DecoderFrameWorkStepBegin        DecoderFrameWorkStepKind = internaldecoder.FrameWorkStepBegin
	DecoderFrameWorkStepTile         DecoderFrameWorkStepKind = internaldecoder.FrameWorkStepTile
	DecoderFrameWorkStepShowExisting DecoderFrameWorkStepKind = internaldecoder.FrameWorkStepShowExisting

	DecoderFrameWorkPlaneY DecoderFrameWorkPlane = internaldecoder.FrameWorkPlaneY
	DecoderFrameWorkPlaneU DecoderFrameWorkPlane = internaldecoder.FrameWorkPlaneU
	DecoderFrameWorkPlaneV DecoderFrameWorkPlane = internaldecoder.FrameWorkPlaneV

	DecoderFrameWorkReferenceLast    DecoderFrameWorkReference = internaldecoder.FrameWorkReferenceLast
	DecoderFrameWorkReferenceLast2   DecoderFrameWorkReference = internaldecoder.FrameWorkReferenceLast2
	DecoderFrameWorkReferenceLast3   DecoderFrameWorkReference = internaldecoder.FrameWorkReferenceLast3
	DecoderFrameWorkReferenceGolden  DecoderFrameWorkReference = internaldecoder.FrameWorkReferenceGolden
	DecoderFrameWorkReferenceBwd     DecoderFrameWorkReference = internaldecoder.FrameWorkReferenceBwd
	DecoderFrameWorkReferenceAltRef2 DecoderFrameWorkReference = internaldecoder.FrameWorkReferenceAltRef2
	DecoderFrameWorkReferenceAltRef  DecoderFrameWorkReference = internaldecoder.FrameWorkReferenceAltRef

	DecoderFrameWorkPostFilterLoopFilter      DecoderFrameWorkPostFilterStage = internaldecoder.FrameWorkPostFilterLoopFilter
	DecoderFrameWorkPostFilterCDEF            DecoderFrameWorkPostFilterStage = internaldecoder.FrameWorkPostFilterCDEF
	DecoderFrameWorkPostFilterSuperRes        DecoderFrameWorkPostFilterStage = internaldecoder.FrameWorkPostFilterSuperRes
	DecoderFrameWorkPostFilterLoopRestoration DecoderFrameWorkPostFilterStage = internaldecoder.FrameWorkPostFilterLoopRestoration
	DecoderFrameWorkPostFilterFilmGrain       DecoderFrameWorkPostFilterStage = internaldecoder.FrameWorkPostFilterFilmGrain
)

var (
	ErrDecoderMissingSequenceHeader          = internaldecoder.ErrMissingSequenceHeader
	ErrDecoderMissingFrameHeader             = internaldecoder.ErrMissingFrameHeader
	ErrDecoderEventBufferTooSmall            = internaldecoder.ErrEventBufferTooSmall
	ErrDecoderInvalidFrameWorkState          = internaldecoder.ErrInvalidFrameWorkState
	ErrDecoderInvalidFrameWorkStep           = internaldecoder.ErrInvalidFrameWorkStep
	ErrDecoderInvalidTileWork                = internaldecoder.ErrInvalidTileWork
	ErrDecoderInvalidSurfaceEvent            = internaldecoder.ErrInvalidSurfaceEvent
	ErrDecoderInvalidSurfaceReference        = internaldecoder.ErrInvalidSurfaceReference
	ErrDecoderSurfaceReferenceBufferTooSmall = internaldecoder.ErrSurfaceReferenceBufferTooSmall
	ErrDecoderSurfaceReleaseBufferTooSmall   = internaldecoder.ErrSurfaceReleaseBufferTooSmall
	ErrDecoderUnsupportedPostFilter          = internaldecoder.ErrUnsupportedPostFilter

	ErrTileInvalidPlan       = internaltile.ErrInvalidPlan
	ErrTileJobBufferTooSmall = internaltile.ErrJobBufferTooSmall

	ErrThreadingInvalidWorkerCount  = internalthreading.ErrInvalidWorkerCount
	ErrThreadingBatchBufferTooSmall = internalthreading.ErrBatchBufferTooSmall
	ErrThreadingInvalidBatch        = internalthreading.ErrInvalidBatch
	ErrThreadingInvalidJobs         = internalthreading.ErrInvalidJobs
	ErrThreadingInvalidCallback     = internalthreading.ErrInvalidCallback
	ErrThreadingPoolClosed          = internalthreading.ErrPoolClosed
)

func PlanDecoderTileWork(event DecoderEvent, workers int, spans []TileSpan, jobs []TileJob, batches []TileBatch) (DecoderTileWorkPlan, error) {
	return internaldecoder.PlanTileWork(event, workers, spans, jobs, batches)
}

func BeginDecoderFrameWork(refs *DecoderSurfaceReferences, pool *FramePool, sequence SequenceHeader, event DecoderEvent, align int, references []int, workers int, spans []TileSpan, jobs []TileJob, batches []TileBatch) (DecoderFrameWorkPlan, *Frame, error) {
	return internaldecoder.BeginFrameWork(refs, pool, sequence, event, align, references, workers, spans, jobs, batches)
}

func PlanDecoderFrameTileWork(event DecoderEvent, surface int, referenceCount int, workers int, spans []TileSpan, jobs []TileJob, batches []TileBatch) (DecoderFrameTileWorkPlan, error) {
	return internaldecoder.PlanFrameTileWork(event, surface, referenceCount, workers, spans, jobs, batches)
}

func ExecuteDecoderTileWork(plan DecoderTileWorkPlan, pool *TileWorkerPool, jobs []TileJob, batches []TileBatch, fn TileBatchFunc) error {
	return internaldecoder.ExecuteTileWork(plan, pool, jobs, batches, fn)
}

func ExecuteDecoderFrameWorkStep(step DecoderFrameWorkStep, pool *TileWorkerPool, jobs []TileJob, batches []TileBatch, fn TileBatchFunc) (bool, error) {
	return internaldecoder.ExecuteFrameWorkStep(step, pool, jobs, batches, fn)
}

func ExecuteDecoderFrameWorkStepWithContext(step DecoderFrameWorkStep, pool *TileWorkerPool, output *Frame, references []*Frame, jobs []TileJob, batches []TileBatch, fn DecoderFrameWorkBatchFunc) (bool, error) {
	return internaldecoder.ExecuteFrameWorkStepWithContext(step, pool, output, references, jobs, batches, fn)
}

func ExecuteDecoderFrameWorkStepWithPayload(step DecoderFrameWorkStep, pool *TileWorkerPool, output *Frame, references []*Frame, payload []byte, jobs []TileJob, batches []TileBatch, fn DecoderFrameWorkBatchFunc) (bool, error) {
	return internaldecoder.ExecuteFrameWorkStepWithPayload(step, pool, output, references, payload, jobs, batches, fn)
}

func RunDecoderFrameWorkEventWithContext(state *DecoderFrameWorkState, refs *DecoderSurfaceReferences, framePool *FramePool, sequence SequenceHeader, event DecoderEvent, align int, referenceSurfaces []int, referenceFrames []*Frame, workers int, spans []TileSpan, jobs []TileJob, batches []TileBatch, releases []int, workerPool *TileWorkerPool, fn DecoderFrameWorkBatchFunc) (DecoderFrameWorkEventResult, error) {
	return state.RunEventWithContext(refs, framePool, sequence, event, align, referenceSurfaces, referenceFrames, workers, spans, jobs, batches, releases, workerPool, fn)
}

func RunDecoderFrameWorkEventWithContextAndPostFilter(state *DecoderFrameWorkState, refs *DecoderSurfaceReferences, framePool *FramePool, sequence SequenceHeader, event DecoderEvent, align int, referenceSurfaces []int, referenceFrames []*Frame, workers int, spans []TileSpan, jobs []TileJob, batches []TileBatch, releases []int, workerPool *TileWorkerPool, fn DecoderFrameWorkBatchFunc, post DecoderFrameWorkPostFilterFunc) (DecoderFrameWorkEventResult, error) {
	return state.RunEventWithContextAndPostFilter(refs, framePool, sequence, event, align, referenceSurfaces, referenceFrames, workers, spans, jobs, batches, releases, workerPool, fn, post)
}

func DecoderEventDropsFrameWork(event DecoderEvent) bool {
	return internaldecoder.EventDropsFrameWork(event)
}

func DecoderEventCompletesFrameWork(event DecoderEvent) bool {
	return internaldecoder.EventCompletesFrameWork(event)
}

func NewTileWorkerPool(workers int) (*TileWorkerPool, error) {
	return internalthreading.NewPool(workers)
}

func DecoderFrameWorkSequenceContextFromHeader(sequence SequenceHeader) DecoderFrameWorkSequenceContext {
	return internalthreading.FrameWorkSequenceContextFromHeader(sequence)
}

func DecoderAcquireFrameSurface(pool *FramePool, sequence SequenceHeader, size FrameSize, align int) (int, *Frame, error) {
	return internaldecoder.AcquireFrameSurface(pool, sequence, size, align)
}

func DecoderBeginFrameSurface(refs *DecoderSurfaceReferences, pool *FramePool, sequence SequenceHeader, event DecoderEvent, align int, references []int) (int, *Frame, int, error) {
	return internaldecoder.BeginFrameSurface(refs, pool, sequence, event, align, references)
}

func ResolveDecoderFrameReferences(pool *FramePool, surfaces []int, frames []*Frame) (int, error) {
	return internaldecoder.ResolveFrameReferences(pool, surfaces, frames)
}

func DecoderFinishFrameSurface(refs *DecoderSurfaceReferences, pool *FramePool, event DecoderEvent, surface int, releases []int) (int, error) {
	return internaldecoder.FinishFrameSurface(refs, pool, event, surface, releases)
}

func DecoderShowExistingFrameSurface(refs *DecoderSurfaceReferences, pool *FramePool, event DecoderEvent, releases []int) (int, int, error) {
	return internaldecoder.ShowExistingFrameSurface(refs, pool, event, releases)
}
