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
type DecoderFrameWorkState = internaldecoder.FrameWorkState
type DecoderSurfaceReferences = internaldecoder.SurfaceReferences
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
)

var (
	ErrDecoderMissingSequenceHeader          = internaldecoder.ErrMissingSequenceHeader
	ErrDecoderMissingFrameHeader             = internaldecoder.ErrMissingFrameHeader
	ErrDecoderEventBufferTooSmall            = internaldecoder.ErrEventBufferTooSmall
	ErrDecoderInvalidFrameWorkState          = internaldecoder.ErrInvalidFrameWorkState
	ErrDecoderInvalidTileWork                = internaldecoder.ErrInvalidTileWork
	ErrDecoderInvalidSurfaceEvent            = internaldecoder.ErrInvalidSurfaceEvent
	ErrDecoderInvalidSurfaceReference        = internaldecoder.ErrInvalidSurfaceReference
	ErrDecoderSurfaceReferenceBufferTooSmall = internaldecoder.ErrSurfaceReferenceBufferTooSmall
	ErrDecoderSurfaceReleaseBufferTooSmall   = internaldecoder.ErrSurfaceReleaseBufferTooSmall

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

func DecoderEventDropsFrameWork(event DecoderEvent) bool {
	return internaldecoder.EventDropsFrameWork(event)
}

func DecoderEventCompletesFrameWork(event DecoderEvent) bool {
	return internaldecoder.EventCompletesFrameWork(event)
}

func NewTileWorkerPool(workers int) (*TileWorkerPool, error) {
	return internalthreading.NewPool(workers)
}

func DecoderAcquireFrameSurface(pool *FramePool, sequence SequenceHeader, size FrameSize, align int) (int, *Frame, error) {
	return internaldecoder.AcquireFrameSurface(pool, sequence, size, align)
}

func DecoderBeginFrameSurface(refs *DecoderSurfaceReferences, pool *FramePool, sequence SequenceHeader, event DecoderEvent, align int, references []int) (int, *Frame, int, error) {
	return internaldecoder.BeginFrameSurface(refs, pool, sequence, event, align, references)
}

func DecoderFinishFrameSurface(refs *DecoderSurfaceReferences, pool *FramePool, event DecoderEvent, surface int, releases []int) (int, error) {
	return internaldecoder.FinishFrameSurface(refs, pool, event, surface, releases)
}

func DecoderShowExistingFrameSurface(refs *DecoderSurfaceReferences, pool *FramePool, event DecoderEvent, releases []int) (int, int, error) {
	return internaldecoder.ShowExistingFrameSurface(refs, pool, event, releases)
}
