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
)

var (
	ErrDecoderMissingSequenceHeader          = internaldecoder.ErrMissingSequenceHeader
	ErrDecoderMissingFrameHeader             = internaldecoder.ErrMissingFrameHeader
	ErrDecoderEventBufferTooSmall            = internaldecoder.ErrEventBufferTooSmall
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

func NewTileWorkerPool(workers int) (*TileWorkerPool, error) {
	return internalthreading.NewPool(workers)
}

func DecoderAcquireFrameSurface(pool *FramePool, sequence SequenceHeader, size FrameSize, align int) (int, *Frame, error) {
	return internaldecoder.AcquireFrameSurface(pool, sequence, size, align)
}

func DecoderFinishFrameSurface(refs *DecoderSurfaceReferences, pool *FramePool, event DecoderEvent, surface int, releases []int) (int, error) {
	return internaldecoder.FinishFrameSurface(refs, pool, event, surface, releases)
}

func DecoderShowExistingFrameSurface(refs *DecoderSurfaceReferences, pool *FramePool, event DecoderEvent, releases []int) (int, int, error) {
	return internaldecoder.ShowExistingFrameSurface(refs, pool, event, releases)
}
