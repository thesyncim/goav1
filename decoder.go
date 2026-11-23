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
// decoder_residual_event.go, decoder_residual_stream.go, decoder_tile_list.go,
// decoder_postfilter_bind.go, decoder_prediction.go, decoder_motion_bind.go,
// and decoder_coeff_reconstruct.go.

import (
	internaldecoder "github.com/thesyncim/goav1/internal/av1/decoder"
	internalframe "github.com/thesyncim/goav1/internal/av1/frame"
	internalparser "github.com/thesyncim/goav1/internal/av1/parser"
	internalthreading "github.com/thesyncim/goav1/internal/av1/threading"
	internaltile "github.com/thesyncim/goav1/internal/av1/tile"
)

// Decoder, tile, and threading error values. Each Err* sentinel is matched
// with errors.Is by the corresponding helper. Callers should use errors.Is
// rather than == when the error is produced indirectly (e.g. through a
// runner callback).
var (
	ErrDecoderMissingSequenceHeader = internaldecoder.ErrMissingSequenceHeader
	ErrDecoderMissingFrameHeader    = internaldecoder.ErrMissingFrameHeader
	ErrDecoderEventBufferTooSmall   = internaldecoder.ErrEventBufferTooSmall

	ErrDecoderUnsupportedPostFilter = internaldecoder.ErrUnsupportedPostFilter
)

func planDecoderTileWorkPtr(event *internaldecoder.Event, workers int, spans []internalparser.TileSpan, jobs []internaltile.Job, batches []internalthreading.Batch) (internaldecoder.TileWorkPlan, error) {
	return internaldecoder.PlanTileWorkPtr(event, workers, spans, jobs, batches)
}

// runDecoderFrameWorkEventWithContext drives one decoder event through the
// caller-owned DecoderFrameWorkState, performing surface acquisition,
// reference resolution, tile-work dispatch, and surface release in one call.
// It returns the event result describing what happened (frame completed,
// dropped, in flight).
func runDecoderFrameWorkEventWithContext(state *internaldecoder.FrameWorkState, refs *internaldecoder.SurfaceReferences, framePool *internalframe.Pool, sequence SequenceHeader, event *internaldecoder.Event, align int, referenceSurfaces []int, referenceFrames []*Frame, workers int, spans []internalparser.TileSpan, jobs []internaltile.Job, batches []internalthreading.Batch, releases []int, workerPool *internalthreading.Pool, fn internaldecoder.FrameWorkBatchFunc) (internaldecoder.FrameWorkEventResult, error) {
	return state.RunEventWithContext(refs, framePool, sequence, event, align, referenceSurfaces, referenceFrames, workers, spans, jobs, batches, releases, workerPool, fn)
}

// runDecoderFrameWorkEventWithContextAndPostFilter is
// runDecoderFrameWorkEventWithContext with an additional post-filter
// callback invoked once per completed frame.
func runDecoderFrameWorkEventWithContextAndPostFilter(state *internaldecoder.FrameWorkState, refs *internaldecoder.SurfaceReferences, framePool *internalframe.Pool, sequence SequenceHeader, event *internaldecoder.Event, align int, referenceSurfaces []int, referenceFrames []*Frame, workers int, spans []internalparser.TileSpan, jobs []internaltile.Job, batches []internalthreading.Batch, releases []int, workerPool *internalthreading.Pool, fn internaldecoder.FrameWorkBatchFunc, post internaldecoder.FrameWorkPostFilterFunc) (internaldecoder.FrameWorkEventResult, error) {
	return state.RunEventWithContextAndPostFilter(refs, framePool, sequence, event, align, referenceSurfaces, referenceFrames, workers, spans, jobs, batches, releases, workerPool, fn, post)
}
