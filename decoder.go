package goav1

// This file re-exports the decoder entry points: the parsed event stream,
// the per-event frame-work plans, the post-filter scratch and request/result
// types, and the frame-work state object that ties them all together.
//
// The exported identifiers fall into the following groups:
//
//   - DecoderStream and DecoderEvent / DecoderEventKind: the OBU parser
//     output a caller drives the decoder from.
//   - DecoderTileWorkPlan, DecoderFrameWorkPlan, DecoderFrameTileWorkPlan,
//     DecoderShowExistingFrameWorkPlan, DecoderFrameWorkStep* and
//     DecoderFrameWorkBatch*: the per-event tile-work plans and the
//     callback signatures the executor invokes.
//   - DecoderFrameWork{Sequence,Frame}Context, DecoderFrameWorkPlane*,
//     DecoderFrameWorkJobRegion, DecoderFrameWorkPlaneRegion,
//     DecoderFrameWorkReference*: the per-batch context the executor
//     supplies to caller callbacks.
//   - DecoderFrameWorkPostFilter*, DecoderFrameWork{Loop,CDEF,SuperRes,
//     Restoration,FilmGrain}*PostFilter*: the per-stage post-filter
//     contexts, plans, requests, results, and scratch-size descriptors.
//   - DecoderFrameWorkTileResidual*, DecoderFrameWorkBlock*,
//     DecoderFrameWork{Intra,Inter,CFL}PredictionScratch: the per-tile and
//     per-block scratches and helper types the executor's per-batch
//     callbacks invoke.
//   - DecoderFrameWorkState, DecoderSurfaceReferences,
//     DecoderFrameWorkRestorationFrameBuffers: the long-lived caller-owned
//     state objects the run loop drives.
//
// Operations on these types live in this file, in decoder_residual_state.go,
// decoder_residual_event.go, decoder_residual_stream.go,
// decoder_postfilter_bind.go, decoder_prediction.go, decoder_motion_bind.go,
// and decoder_coeff_reconstruct.go.

import (
	internaldecoder "github.com/thesyncim/goav1/internal/av1/decoder"
	internalthreading "github.com/thesyncim/goav1/internal/av1/threading"
	internaltile "github.com/thesyncim/goav1/internal/av1/tile"
)

// DecoderStream wires an OBU/temporal-unit byte source to a decoder pipeline
// and yields parsed DecoderEvent values to the caller, who drives frame work
// from those events.
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
type DecoderFrameWorkPostFilterRunner = internaldecoder.FrameWorkPostFilterRunner
type DecoderFrameWorkPostFilterStage = internaldecoder.FrameWorkPostFilterStage
type DecoderFrameWorkPostFilterScratchSize = internaldecoder.FrameWorkPostFilterScratchSize
type DecoderFrameWorkPostFilterRequest = internaldecoder.FrameWorkPostFilterRequest
type DecoderFrameWorkPostFilterResult = internaldecoder.FrameWorkPostFilterResult
type DecoderFrameWorkCallerPostFilterResult = internaldecoder.FrameWorkCallerPostFilterResult
type DecoderFrameWorkSupportedPostFilterRunner = internaldecoder.FrameWorkSupportedPostFilterRunner
type DecoderFrameWorkCallerPostFilterRunner = internaldecoder.FrameWorkCallerPostFilterRunner
type DecoderFrameWorkCDEFIndexMap = internaldecoder.FrameWorkCDEFIndexMap
type DecoderFrameWorkCDEFPostFilterScratchSize = internaldecoder.FrameWorkCDEFPostFilterScratchSize
type DecoderFrameWorkCDEFPostFilterRequest = internaldecoder.FrameWorkCDEFPostFilterRequest
type DecoderFrameWorkCDEFPostFilterResult = internaldecoder.FrameWorkCDEFPostFilterResult
type DecoderFrameWorkLoopFilterMap = internaldecoder.FrameWorkLoopFilterMap
type DecoderFrameWorkLoopFilterBlockRecord = internalthreading.FrameWorkLoopFilterBlockRecord
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
type DecoderFrameWorkTileResidualCDFStorage = internalthreading.FrameWorkTileResidualCDFStorage
type DecoderFrameWorkTileResidualCDFs = internalthreading.FrameWorkTileResidualCDFs
type DecoderFrameWorkTileResidualScratch = internalthreading.FrameWorkTileResidualScratch
type DecoderFrameWorkTileResidualRequest = internalthreading.FrameWorkTileResidualRequest
type DecoderFrameWorkTileResidualStats = internalthreading.FrameWorkTileResidualStats
type DecoderFrameWorkBlockCoeffReconstruction = internalthreading.FrameWorkBlockCoeffReconstruction
type DecoderFrameWorkBlockTransforms = internalthreading.FrameWorkBlockTransforms
type DecoderFrameWorkBlockTransformSelector = internalthreading.FrameWorkBlockTransformSelector
type DecoderFrameWorkBlockPredictor = internalthreading.FrameWorkBlockPredictor
type DecoderFrameWorkTileRestorationRequest = internalthreading.FrameWorkTileRestorationRequest
type DecoderFrameWorkPredictionScratch = internalthreading.FrameWorkPredictionScratch
type DecoderFrameWorkIntraPredictionScratch = internalthreading.FrameWorkIntraPredictionScratch
type DecoderFrameWorkInterPredictionScratch = internalthreading.FrameWorkInterPredictionScratch
type DecoderFrameWorkCFLPredictionScratch = internalthreading.FrameWorkCFLPredictionScratch
type DecoderFrameWorkState = internaldecoder.FrameWorkState
type DecoderSurfaceReferences = internaldecoder.SurfaceReferences
type DecoderFrameWorkRestorationFrameBuffers = internalthreading.FrameWorkRestorationFrameBuffers
type TileJob = internaltile.Job
type TileBatch = internalthreading.Batch
type TileWorkerPool = internalthreading.Pool
type TileBatchFunc = internalthreading.BatchFunc

// AV1 decoder event and frame-work step kinds.
//
// The DecoderEvent* values classify the OBU/tile-group/etc. parse output
// produced by DecoderStream. The DecoderFrameWorkStep* values classify each
// step of frame-work execution. DecoderFrameWorkPlane* / Reference* /
// PostFilter* values identify planes, AV1 reference slots, and post-filter
// stages used by the frame-work and post-filter helpers.
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

// Decoder, tile, and threading error values. Each Err* sentinel is matched
// with errors.Is by the corresponding helper. Callers should use errors.Is
// rather than == when the error is produced indirectly (e.g. through a
// runner callback).
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

// PlanDecoderTileWork builds a bounded DecoderTileWorkPlan from a tile-group
// event, using caller-owned spans, jobs, and batches scratches and the
// requested worker count.
func PlanDecoderTileWork(event DecoderEvent, workers int, spans []TileSpan, jobs []TileJob, batches []TileBatch) (DecoderTileWorkPlan, error) {
	return internaldecoder.PlanTileWork(event, workers, spans, jobs, batches)
}

// BeginDecoderFrameWork acquires an output surface from pool, validates the
// reference set, and returns a DecoderFrameWorkPlan together with the
// acquired *Frame. align is the surface alignment requirement.
func BeginDecoderFrameWork(refs *DecoderSurfaceReferences, pool *FramePool, sequence SequenceHeader, event DecoderEvent, align int, references []int, workers int, spans []TileSpan, jobs []TileJob, batches []TileBatch) (DecoderFrameWorkPlan, *Frame, error) {
	return internaldecoder.BeginFrameWork(refs, pool, sequence, event, align, references, workers, spans, jobs, batches)
}

// PlanDecoderFrameTileWork builds a frame-scoped DecoderFrameTileWorkPlan
// targeting the specified surface index and reference count.
func PlanDecoderFrameTileWork(event DecoderEvent, surface int, referenceCount int, workers int, spans []TileSpan, jobs []TileJob, batches []TileBatch) (DecoderFrameTileWorkPlan, error) {
	return internaldecoder.PlanFrameTileWork(event, surface, referenceCount, workers, spans, jobs, batches)
}

// ExecuteDecoderTileWork dispatches plan onto pool, invoking fn once per
// scheduled batch. jobs and batches must be the same slices that were passed
// to the planner.
func ExecuteDecoderTileWork(plan DecoderTileWorkPlan, pool *TileWorkerPool, jobs []TileJob, batches []TileBatch, fn TileBatchFunc) error {
	return internaldecoder.ExecuteTileWork(plan, pool, jobs, batches, fn)
}

// ExecuteDecoderFrameWorkStep executes one DecoderFrameWorkStep, dispatching
// tile work onto pool when the step kind is tile-work.
func ExecuteDecoderFrameWorkStep(step DecoderFrameWorkStep, pool *TileWorkerPool, jobs []TileJob, batches []TileBatch, fn TileBatchFunc) (bool, error) {
	return internaldecoder.ExecuteFrameWorkStep(step, pool, jobs, batches, fn)
}

// ExecuteDecoderFrameWorkStepWithContext is ExecuteDecoderFrameWorkStep with
// the output frame and resolved reference frames threaded through to fn via
// DecoderFrameWorkBatchFunc.
func ExecuteDecoderFrameWorkStepWithContext(step DecoderFrameWorkStep, pool *TileWorkerPool, output *Frame, references []*Frame, jobs []TileJob, batches []TileBatch, fn DecoderFrameWorkBatchFunc) (bool, error) {
	return internaldecoder.ExecuteFrameWorkStepWithContext(step, pool, output, references, jobs, batches, fn)
}

// ExecuteDecoderFrameWorkStepWithPayload is
// ExecuteDecoderFrameWorkStepWithContext with the tile-group payload bytes
// supplied separately so they may live in caller-owned demux buffers.
func ExecuteDecoderFrameWorkStepWithPayload(step DecoderFrameWorkStep, pool *TileWorkerPool, output *Frame, references []*Frame, payload []byte, jobs []TileJob, batches []TileBatch, fn DecoderFrameWorkBatchFunc) (bool, error) {
	return internaldecoder.ExecuteFrameWorkStepWithPayload(step, pool, output, references, payload, jobs, batches, fn)
}

// RunDecoderFrameWorkEventWithContext drives one decoder event through the
// caller-owned DecoderFrameWorkState, performing surface acquisition,
// reference resolution, tile-work dispatch, and surface release in one call.
// It returns the event result describing what happened (frame completed,
// dropped, in flight).
func RunDecoderFrameWorkEventWithContext(state *DecoderFrameWorkState, refs *DecoderSurfaceReferences, framePool *FramePool, sequence SequenceHeader, event DecoderEvent, align int, referenceSurfaces []int, referenceFrames []*Frame, workers int, spans []TileSpan, jobs []TileJob, batches []TileBatch, releases []int, workerPool *TileWorkerPool, fn DecoderFrameWorkBatchFunc) (DecoderFrameWorkEventResult, error) {
	return state.RunEventWithContext(refs, framePool, sequence, event, align, referenceSurfaces, referenceFrames, workers, spans, jobs, batches, releases, workerPool, fn)
}

// RunDecoderFrameWorkEventWithContextAndPostFilter is
// RunDecoderFrameWorkEventWithContext with an additional post-filter
// callback invoked once per completed frame.
func RunDecoderFrameWorkEventWithContextAndPostFilter(state *DecoderFrameWorkState, refs *DecoderSurfaceReferences, framePool *FramePool, sequence SequenceHeader, event DecoderEvent, align int, referenceSurfaces []int, referenceFrames []*Frame, workers int, spans []TileSpan, jobs []TileJob, batches []TileBatch, releases []int, workerPool *TileWorkerPool, fn DecoderFrameWorkBatchFunc, post DecoderFrameWorkPostFilterFunc) (DecoderFrameWorkEventResult, error) {
	return state.RunEventWithContextAndPostFilter(refs, framePool, sequence, event, align, referenceSurfaces, referenceFrames, workers, spans, jobs, batches, releases, workerPool, fn, post)
}

// DecoderEventDropsFrameWork reports whether event drops the currently
// in-flight frame-work (e.g. a new sequence header or temporal delimiter
// arriving mid-frame).
func DecoderEventDropsFrameWork(event DecoderEvent) bool {
	return internaldecoder.EventDropsFrameWork(event)
}

// DecoderEventCompletesFrameWork reports whether event completes the
// currently in-flight frame-work (e.g. the last tile-group of a frame).
func DecoderEventCompletesFrameWork(event DecoderEvent) bool {
	return internaldecoder.EventCompletesFrameWork(event)
}

// DecoderEventOutputsFrame reports whether event will produce a visible
// output frame (i.e. show_frame=1 or show_existing_frame=1).
func DecoderEventOutputsFrame(event DecoderEvent) bool {
	return internaldecoder.EventOutputsFrame(event)
}

// NewTileWorkerPool returns a bounded TileWorkerPool with the requested
// worker goroutine count. The caller is responsible for closing the pool to
// release its goroutines.
func NewTileWorkerPool(workers int) (*TileWorkerPool, error) {
	return internalthreading.NewPool(workers)
}

// DecoderFrameWorkSequenceContextFromHeader builds a
// DecoderFrameWorkSequenceContext from sequence so it can be threaded into
// the frame-work helpers without re-parsing.
func DecoderFrameWorkSequenceContextFromHeader(sequence SequenceHeader) DecoderFrameWorkSequenceContext {
	return internalthreading.FrameWorkSequenceContextFromHeader(sequence)
}

// DecoderAcquireFrameSurface acquires a Frame from pool sized for (sequence,
// size, align). It is the building block used by BeginDecoderFrameWork.
func DecoderAcquireFrameSurface(pool *FramePool, sequence SequenceHeader, size FrameSize, align int) (int, *Frame, error) {
	return internaldecoder.AcquireFrameSurface(pool, sequence, size, align)
}

// DecoderBeginFrameSurface acquires the output surface implied by event,
// updates refs to map AV1 reference slots to pool indices, and returns the
// acquired surface index, frame, and reference count.
func DecoderBeginFrameSurface(refs *DecoderSurfaceReferences, pool *FramePool, sequence SequenceHeader, event DecoderEvent, align int, references []int) (int, *Frame, int, error) {
	return internaldecoder.BeginFrameSurface(refs, pool, sequence, event, align, references)
}

// ResolveDecoderFrameReferences fills frames with the *Frame pointers for the
// reference surface indices in surfaces. The frames slice must be at least
// the length of surfaces.
func ResolveDecoderFrameReferences(pool *FramePool, surfaces []int, frames []*Frame) (int, error) {
	return internaldecoder.ResolveFrameReferences(pool, surfaces, frames)
}

// DecoderFinishFrameSurface marks the frame work for event as complete,
// updates refs accordingly, and writes the indices of any surfaces that
// became eligible for release into releases.
func DecoderFinishFrameSurface(refs *DecoderSurfaceReferences, pool *FramePool, event DecoderEvent, surface int, releases []int) (int, error) {
	return internaldecoder.FinishFrameSurface(refs, pool, event, surface, releases)
}

// DecoderShowExistingFrameSurface processes a show_existing_frame event,
// updating refs and returning the displayed surface index together with any
// freed surface indices written into releases.
func DecoderShowExistingFrameSurface(refs *DecoderSurfaceReferences, pool *FramePool, event DecoderEvent, releases []int) (int, int, error) {
	return internaldecoder.ShowExistingFrameSurface(refs, pool, event, releases)
}
