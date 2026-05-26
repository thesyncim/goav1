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

// DecoderEvent is one parsed unit emitted by DecoderStream: a sequence
// header, frame header, tile group, or other OBU classification together
// with its payload span and parsed metadata.
type DecoderEvent = internaldecoder.Event

// DecoderEventKind tags a DecoderEvent value with the AV1 OBU category that
// produced it. See the DecoderEvent* constants for the complete enumeration.
type DecoderEventKind = internaldecoder.EventKind

// DecoderTileWorkPlan describes a bounded tile-group decode schedule built
// over caller-owned TileSpan, TileJob, and TileBatch slices.
//
// See also: PlanDecoderTileWork, ExecuteDecoderTileWork.
type DecoderTileWorkPlan = internaldecoder.TileWorkPlan

// DecoderFrameWorkPlan ties a DecoderTileWorkPlan to its acquired output
// surface, resolved reference set, and frame-work step list for one frame.
//
// See also: BeginDecoderFrameWork, DecoderFrameWorkStep.
type DecoderFrameWorkPlan = internaldecoder.FrameWorkPlan

// DecoderFrameTileWorkPlan is a frame-scoped DecoderTileWorkPlan that
// targets a specific surface index and reference count.
//
// See also: PlanDecoderFrameTileWork.
type DecoderFrameTileWorkPlan = internaldecoder.FrameTileWorkPlan

// DecoderShowExistingFrameWorkPlan describes a show_existing_frame event:
// the surface that should be displayed plus any surfaces eligible for
// release after the show.
type DecoderShowExistingFrameWorkPlan = internaldecoder.ShowExistingFrameWorkPlan

// DecoderFrameWorkStepKind tags one entry of a frame-work plan. See the
// DecoderFrameWorkStep* constants for the complete enumeration.
type DecoderFrameWorkStepKind = internaldecoder.FrameWorkStepKind

// DecoderFrameWorkStep is one step of a DecoderFrameWorkPlan: begin, a
// per-tile-group tile-work dispatch, a show-existing-frame action, dropped
// work, or ignored work.
//
// See also: ExecuteDecoderFrameWorkStep.
type DecoderFrameWorkStep = internaldecoder.FrameWorkStep

// DecoderFrameWorkStepResult reports the outcome of executing one
// DecoderFrameWorkStep, including whether the step completed in flight.
type DecoderFrameWorkStepResult = internaldecoder.FrameWorkStepResult

// DecoderFrameWorkBatch carries the per-batch context the executor passes
// to a DecoderFrameWorkBatchFunc: sequence/frame context, payload, jobs,
// and the worker index assigned to this batch.
type DecoderFrameWorkBatch = internaldecoder.FrameWorkBatch

// DecoderFrameWorkSequenceContext is a derived view of SequenceHeader
// optimized for hot-path lookups in the frame-work batch callback.
//
// See also: DecoderFrameWorkSequenceContextFromHeader.
type DecoderFrameWorkSequenceContext = internaldecoder.FrameWorkSequenceContext

// DecoderFrameWorkFrameContext is the per-frame derived geometry the
// executor threads into the per-batch callback (mi geometry, planes,
// reference setup, etc.).
type DecoderFrameWorkFrameContext = internaldecoder.FrameWorkFrameContext

// DecoderFrameWorkJobRegion locates a tile job's region within the frame:
// mi origin, mi size, and the bytes-per-sample-aligned plane regions.
type DecoderFrameWorkJobRegion = internaldecoder.FrameWorkJobRegion

// DecoderFrameWorkPlane identifies a plane (Y, U, or V) within frame-work
// helpers. See the DecoderFrameWorkPlane* constants for the enumeration.
type DecoderFrameWorkPlane = internaldecoder.FrameWorkPlane

// DecoderFrameWorkPlaneRegion describes a clipped per-plane reconstruction
// region within the output frame, ready for scratch binding.
type DecoderFrameWorkPlaneRegion = internaldecoder.FrameWorkPlaneRegion

// DecoderFrameWorkReference identifies one of the seven AV1 inter
// reference slots. See the DecoderFrameWorkReference* constants for the
// enumeration.
type DecoderFrameWorkReference = internaldecoder.FrameWorkReference

// DecoderFrameWorkBatchFunc is the per-batch callback invoked by the
// frame-work executor. It receives the DecoderFrameWorkBatch and may
// perform residual decode, reconstruction, and post-filter setup using
// the caller-owned scratch supplied at execution time.
type DecoderFrameWorkBatchFunc = internaldecoder.FrameWorkBatchFunc

// DecoderFrameWorkEventResult classifies the outcome of one event run:
// frame completed, dropped, or still in flight, and whether a surface
// was released.
type DecoderFrameWorkEventResult = internaldecoder.FrameWorkEventResult

// DecoderFrameWorkPostFilterContext bundles the per-frame post-filter
// inputs (frame, references, side data) supplied to a
// DecoderFrameWorkPostFilterFunc when a frame finishes.
type DecoderFrameWorkPostFilterContext = internaldecoder.FrameWorkPostFilterContext

// DecoderFrameWorkPostFilterFunc is the per-frame callback invoked after
// the final tile-group of a frame to apply loop-filter, CDEF, super-res,
// loop-restoration, and film-grain stages.
type DecoderFrameWorkPostFilterFunc = internaldecoder.FrameWorkPostFilterFunc

// DecoderFrameWorkPostFilterRunner is the executor-resolvable interface
// invoked by RunDecoderFrameWorkEventWithContextAndPostFilter to run the
// full post-filter chain.
type DecoderFrameWorkPostFilterRunner = internaldecoder.FrameWorkPostFilterRunner

// DecoderFrameWorkPostFilterStage identifies one stage of the post-filter
// pipeline (loop filter, CDEF, super-res, loop restoration, or film
// grain). See the DecoderFrameWorkPostFilter* constants.
type DecoderFrameWorkPostFilterStage = internaldecoder.FrameWorkPostFilterStage

// DecoderFrameWorkPostFilterScratchSize reports the caller-owned scratch
// lengths required by the full post-filter chain for one frame.
type DecoderFrameWorkPostFilterScratchSize = internaldecoder.FrameWorkPostFilterScratchSize

// DecoderFrameWorkPostFilterRequest bundles all caller-owned post-filter
// scratch and side data into one request value that runs the loop-filter,
// CDEF, super-res, loop-restoration, and film-grain stages.
//
// See also: BindDecoderFrameWorkPostFilterRequest, DecoderFrameWorkPostFilterResult.
type DecoderFrameWorkPostFilterRequest = internaldecoder.FrameWorkPostFilterRequest

// DecoderFrameWorkPostFilterResult reports the per-stage outcome of one
// DecoderFrameWorkPostFilterRequest execution.
//
// See also: DecoderFrameWorkPostFilterRequest.
type DecoderFrameWorkPostFilterResult = internaldecoder.FrameWorkPostFilterResult

// DecoderFrameWorkCallerPostFilterResult is the per-stage outcome of a
// caller-supplied post-filter runner that bypasses the built-in pipeline.
type DecoderFrameWorkCallerPostFilterResult = internaldecoder.FrameWorkCallerPostFilterResult

// DecoderFrameWorkSupportedPostFilterRunner is the built-in post-filter
// runner that applies the full chain when all stages are supported.
type DecoderFrameWorkSupportedPostFilterRunner = internaldecoder.FrameWorkSupportedPostFilterRunner

// DecoderFrameWorkCallerPostFilterRunner is the caller-supplied
// post-filter runner used when one or more stages are routed through a
// custom implementation.
type DecoderFrameWorkCallerPostFilterRunner = internaldecoder.FrameWorkCallerPostFilterRunner

// DecoderFrameWorkCDEFIndexMap is the per-block CDEF index/read map that
// post-filter helpers populate during reconstruction and consume during
// the CDEF stage.
type DecoderFrameWorkCDEFIndexMap = internaldecoder.FrameWorkCDEFIndexMap

// DecoderFrameWorkCDEFPostFilterScratchSize reports the caller-owned
// scratch lengths required by the CDEF stage.
type DecoderFrameWorkCDEFPostFilterScratchSize = internaldecoder.FrameWorkCDEFPostFilterScratchSize

// DecoderFrameWorkCDEFPostFilterRequest binds caller-owned CDEF scratch
// (samples, dst, direction grid, variance grid, input/unit-dst buffers)
// into one CDEF apply request.
//
// See also: BindDecoderFrameWorkCDEFPostFilterRequest.
type DecoderFrameWorkCDEFPostFilterRequest = internaldecoder.FrameWorkCDEFPostFilterRequest

// DecoderFrameWorkCDEFPostFilterResult reports per-plane CDEF statistics.
type DecoderFrameWorkCDEFPostFilterResult = internaldecoder.FrameWorkCDEFPostFilterResult

// DecoderFrameWorkLoopFilterMap is the per-block loop-filter record map
// populated during reconstruction and consumed during the loop-filter
// post-filter stage.
type DecoderFrameWorkLoopFilterMap = internaldecoder.FrameWorkLoopFilterMap

// DecoderFrameWorkLoopFilterBlockRecord is one entry of
// DecoderFrameWorkLoopFilterMap, recording the per-block edge state used
// by the loop-filter stage.
type DecoderFrameWorkLoopFilterBlockRecord = internalthreading.FrameWorkLoopFilterBlockRecord

// DecoderFrameWorkLoopFilterPostFilterRequest binds caller-owned
// loop-filter scratch and the loop-filter map into one apply request.
//
// See also: BindDecoderFrameWorkLoopFilterPostFilterRequest.
type DecoderFrameWorkLoopFilterPostFilterRequest = internaldecoder.FrameWorkLoopFilterPostFilterRequest

// DecoderFrameWorkLoopFilterPostFilterEdge is one entry in the
// loop-filter edge list consumed by the loop-filter apply request.
type DecoderFrameWorkLoopFilterPostFilterEdge = internaldecoder.FrameWorkLoopFilterPostFilterEdge

// DecoderFrameWorkLoopFilterPostFilterScratchSize reports the caller-owned
// scratch lengths required by the loop-filter stage.
type DecoderFrameWorkLoopFilterPostFilterScratchSize = internaldecoder.FrameWorkLoopFilterPostFilterScratchSize

// DecoderFrameWorkLoopFilterPostFilterLevelStats reports per-plane
// loop-filter level statistics observed during the apply pass.
type DecoderFrameWorkLoopFilterPostFilterLevelStats = internaldecoder.FrameWorkLoopFilterPostFilterLevelStats

// DecoderFrameWorkLoopFilterPostFilterPlan summarizes loop-filter
// planning output: edge count, per-plane level activity, and skip flags.
type DecoderFrameWorkLoopFilterPostFilterPlan = internaldecoder.FrameWorkLoopFilterPostFilterPlan

// DecoderFrameWorkLoopFilterPostFilterApplyResult reports per-edge and
// per-plane statistics produced by one loop-filter apply pass.
type DecoderFrameWorkLoopFilterPostFilterApplyResult = internaldecoder.FrameWorkLoopFilterPostFilterApplyResult

// DecoderFrameWorkSuperResPostFilterPlanePlan describes the per-plane
// super-res upscale geometry derived from FrameSize and the active
// SequenceHeader.
type DecoderFrameWorkSuperResPostFilterPlanePlan = internaldecoder.FrameWorkSuperResPostFilterPlanePlan

// DecoderFrameWorkSuperResPostFilterPlan aggregates the per-plane plans
// for one frame's super-res execution.
type DecoderFrameWorkSuperResPostFilterPlan = internaldecoder.FrameWorkSuperResPostFilterPlan

// DecoderFrameWorkSuperResPostFilterScratchSize reports the caller-owned
// scratch lengths required by the super-res stage.
type DecoderFrameWorkSuperResPostFilterScratchSize = internaldecoder.FrameWorkSuperResPostFilterScratchSize

// DecoderFrameWorkSuperResPostFilterRequest binds caller-owned super-res
// scratch (output frame and per-plane coded/output scratch) into one
// apply request.
//
// See also: BindDecoderFrameWorkSuperResPostFilterRequest.
type DecoderFrameWorkSuperResPostFilterRequest = internaldecoder.FrameWorkSuperResPostFilterRequest

// DecoderFrameWorkSuperResPostFilterResult reports per-plane super-res
// statistics from one apply pass.
type DecoderFrameWorkSuperResPostFilterResult = internaldecoder.FrameWorkSuperResPostFilterResult

// DecoderFrameWorkRestorationPostFilterRequest binds caller-owned
// loop-restoration scratch (records, boundaries, sample arenas, Wiener
// and SGR-proj scratch) into one apply request.
//
// See also: BindDecoderFrameWorkRestorationPostFilterRequest.
type DecoderFrameWorkRestorationPostFilterRequest = internaldecoder.FrameWorkRestorationPostFilterRequest

// DecoderFrameWorkRestorationPostFilterScratchSize reports the
// caller-owned scratch lengths required by the loop-restoration stage.
type DecoderFrameWorkRestorationPostFilterScratchSize = internaldecoder.FrameWorkRestorationPostFilterScratchSize

// DecoderFrameWorkFilmGrainPostFilterPlanePlan describes the per-plane
// film-grain noise geometry derived from FilmGrainParams.
type DecoderFrameWorkFilmGrainPostFilterPlanePlan = internaldecoder.FrameWorkFilmGrainPostFilterPlanePlan

// DecoderFrameWorkFilmGrainPostFilterPlan aggregates the per-plane plans
// for one frame's film-grain synthesis.
type DecoderFrameWorkFilmGrainPostFilterPlan = internaldecoder.FrameWorkFilmGrainPostFilterPlan

// DecoderFrameWorkFilmGrainPostFilterScratchSize reports the caller-owned
// scratch lengths required by the film-grain stage.
type DecoderFrameWorkFilmGrainPostFilterScratchSize = internaldecoder.FrameWorkFilmGrainPostFilterScratchSize

// DecoderFrameWorkFilmGrainPostFilterRequest binds caller-owned
// film-grain scratch (luma/chroma grain and sample arenas) into one
// apply request.
//
// See also: BindDecoderFrameWorkFilmGrainPostFilterRequest.
type DecoderFrameWorkFilmGrainPostFilterRequest = internaldecoder.FrameWorkFilmGrainPostFilterRequest

// DecoderFrameWorkFilmGrainPostFilterResult reports per-plane film-grain
// statistics from one apply pass.
type DecoderFrameWorkFilmGrainPostFilterResult = internaldecoder.FrameWorkFilmGrainPostFilterResult

// DecoderFrameWorkFilmGrainPostFilterScalingLUTs holds the caller-owned
// per-plane film-grain scaling lookup tables built from FilmGrainParams.
type DecoderFrameWorkFilmGrainPostFilterScalingLUTs = internaldecoder.FrameWorkFilmGrainPostFilterScalingLUTs

// DecoderFrameWorkFilmGrainPostFilterLumaGrain holds the caller-owned
// luma grain block used by the film-grain synthesis stage.
type DecoderFrameWorkFilmGrainPostFilterLumaGrain = internaldecoder.FrameWorkFilmGrainPostFilterLumaGrain

// DecoderFrameWorkTileResidualCDFStorage owns the per-job adapted CDF
// tables used by the residual decode pipeline. Callers retain it across
// tile-group boundaries to propagate context updates.
//
// See also: InitDecoderFrameWorkTileResidualCDFStorage,
// InitDecoderFrameWorkTileResidualCDFStorageDefault.
type DecoderFrameWorkTileResidualCDFStorage = internalthreading.FrameWorkTileResidualCDFStorage

// DecoderFrameWorkTileResidualCDFs is a view of the active per-job CDF
// tables bound from a DecoderFrameWorkTileResidualCDFStorage.
//
// See also: DecoderFrameWorkTileResidualCDFsFromStorage.
type DecoderFrameWorkTileResidualCDFs = internalthreading.FrameWorkTileResidualCDFs

// DecoderFrameWorkTileResidualScratch carries per-job caller-owned
// scratch (entropy reader state, loop context, residual scratch) used by
// the residual decode pipeline.
type DecoderFrameWorkTileResidualScratch = internalthreading.FrameWorkTileResidualScratch

// DecoderFrameWorkTileResidualRequest bundles the caller-owned inputs
// (CDF maps, scratches, prediction scratch, transform/loop overrides)
// the residual decode helper consumes for one job.
//
// See also: DecoderFrameWorkTileResidualStats.
type DecoderFrameWorkTileResidualRequest = internalthreading.FrameWorkTileResidualRequest

// DecoderFrameWorkTileResidualStats reports per-job decode counters
// (partitions, blocks, modes, motion vectors, coefficient totals,
// reconstructions, etc.) produced by the residual decode pipeline.
//
// See also: DecoderFrameWorkTileResidualRequest.
type DecoderFrameWorkTileResidualStats = internalthreading.FrameWorkTileResidualStats

// DecoderFrameWorkBlockCoeffReconstruction is the per-block reconstruction
// request that combines a decoded coefficient block with the prediction
// scratch needed to assemble the output sample plane.
type DecoderFrameWorkBlockCoeffReconstruction = internalthreading.FrameWorkBlockCoeffReconstruction

// DecoderFrameWorkBlockTransforms describes the per-block transform
// metadata (mode, type, size, partition) selected by the residual decode
// pipeline for one tile block.
type DecoderFrameWorkBlockTransforms = internalthreading.FrameWorkBlockTransforms

// DecoderFrameWorkBlockTransformSelector is the per-block transform
// selector callback the residual pipeline invokes to choose the
// transform type for caller-customized decode paths.
type DecoderFrameWorkBlockTransformSelector = internalthreading.FrameWorkBlockTransformSelector

// DecoderFrameWorkBlockPredictor is the per-block prediction callback the
// residual pipeline invokes to populate the prediction scratch before
// reconstruction.
type DecoderFrameWorkBlockPredictor = internalthreading.FrameWorkBlockPredictor

// DecoderFrameWorkTileRestorationRequest carries the caller-owned
// loop-restoration buffers and references threaded into the residual
// pipeline when restoration is active for the frame.
type DecoderFrameWorkTileRestorationRequest = internalthreading.FrameWorkTileRestorationRequest

// DecoderFrameWorkPredictionScratch is the per-batch prediction scratch
// shared by intra, inter, and CFL prediction helpers.
type DecoderFrameWorkPredictionScratch = internalthreading.FrameWorkPredictionScratch

// DecoderFrameWorkIntraPredictionScratch is the per-block intra
// prediction scratch (above/left edge buffers, scratch plane).
type DecoderFrameWorkIntraPredictionScratch = internalthreading.FrameWorkIntraPredictionScratch

// DecoderFrameWorkInterPredictionScratch is the per-block inter
// prediction scratch (interpolation samples, blend scratch, OBMC
// scratch).
type DecoderFrameWorkInterPredictionScratch = internalthreading.FrameWorkInterPredictionScratch

// DecoderFrameWorkCFLPredictionScratch is the per-block CFL prediction
// scratch (luma AC samples, reconstruction Q3 buffers).
type DecoderFrameWorkCFLPredictionScratch = internalthreading.FrameWorkCFLPredictionScratch

// DecoderFrameWorkState is the long-lived caller-owned object that ties
// together surface-reference tracking, frame-pool acquisition, and
// per-event frame-work execution. Callers drive the decoder by feeding
// DecoderEvent values through state.RunEvent* helpers.
//
// See also: DecoderSurfaceReferences, RunDecoderFrameWorkEventWithContext.
type DecoderFrameWorkState = internaldecoder.FrameWorkState

// DecoderSurfaceReferences tracks which FramePool surface indices are
// currently bound to each AV1 reference slot. Updates happen at frame
// begin/finish and during show_existing_frame events.
type DecoderSurfaceReferences = internaldecoder.SurfaceReferences

// DecoderFrameWorkRestorationFrameBuffers groups the caller-owned
// loop-restoration buffers (per-plane unit records and stripe
// boundaries) maintained across the frame's tile groups.
//
// See also: BindDecoderFrameWorkRestorationFrameBuffers.
type DecoderFrameWorkRestorationFrameBuffers = internalthreading.FrameWorkRestorationFrameBuffers

// TileJob is one scheduled tile decode (offset+size within the
// tile-group payload plus per-tile metadata).
//
// See also: PlanDecoderTileWork, TileBatch.
type TileJob = internaltile.Job

// TileBatch groups a set of TileJob entries assigned to one worker for
// execution by a TileWorkerPool.
//
// See also: TileBatchFunc, TileWorkerPool.
type TileBatch = internalthreading.Batch

// TileWorkerPool is the bounded goroutine pool that dispatches
// TileBatch values to caller-supplied TileBatchFunc callbacks.
//
// See also: NewTileWorkerPool.
type TileWorkerPool = internalthreading.Pool

// TileBatchFunc is the per-batch callback signature invoked by a
// TileWorkerPool. It receives the batch identifier and returns any
// error from the worker's decode pass.
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
