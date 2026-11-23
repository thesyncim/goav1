package goav1

// This file exposes the caller-owned residual-decode state and the helpers
// that drive coefficient decode, block reconstruction, and the per-job
// residual runner.
//
// The helpers fall into these groups:
//
//   - decoderFrameWorkResidualScratchLen / MaxScratchLen /
//     MaxFrameScratchLen / BatchResidualScratchLen /
//     BatchResidualRunnerScratchLen: shape helpers that report the
//     caller-owned scratch lengths required for per-block, per-job, and
//     per-batch residual processing.
//   - decoderFrameWorkBlockQuantizer / BlockQIndex /
//     BlockCoeffPlanePosition: per-block helpers that resolve the
//     quantizer, q-index, and plane position for one block.
//   - initDecoderFrameWorkTileResidualCDFStorage{,Default} /
//     decoderFrameWorkTileResidualCDFsFromStorage: caller-owned CDF
//     storage and binding.
//   - initDecoderFrameWorkJobDecodeState /
//     retainDecoderFrameWorkTileResidualCDFStorage: per-job CDF lifecycle.
//   - initDecoderFrameWorkTileRestorationRequestReferences /
//     decoderFrameWorkJobBlockLoopContextRootColumns /
//     bindTileBlockLoopContextCarrier / decoderFrameWorkJobBlockLoopRequest:
//     per-job block-loop setup.
//   - DecodeAndReconstruct /
//     DecodeAndRetain*: the residual decode + reconstruct pipeline.
//   - decoderFrameWorkBatchResidualRunner and its
//     bindDecoderFrameWorkBatchResidualRunner, Run, SetSideData,
//     ResetStats, TotalStats helpers: the bounded per-batch executor that
//     ties the residual pipeline to the worker pool.

import (
	internaldecoder "github.com/thesyncim/goav1/internal/av1/decoder"
	internalframe "github.com/thesyncim/goav1/internal/av1/frame"
	internalparser "github.com/thesyncim/goav1/internal/av1/parser"
	internalreconstruct "github.com/thesyncim/goav1/internal/av1/reconstruct"
	internalthreading "github.com/thesyncim/goav1/internal/av1/threading"
	internaltile "github.com/thesyncim/goav1/internal/av1/tile"
	internaltransform "github.com/thesyncim/goav1/internal/av1/transform"
)

// decoderFrameWorkResidualScratchLen reports the int32 and int16 residual
// scratch lengths required to reconstruct one block at (size, typ) for the
// given quantizer context.
func decoderFrameWorkResidualScratchLen(batch internaldecoder.FrameWorkBatch, currentQIndex uint8, segmentID uint8, plane internaldecoder.FrameWorkPlane, size internaltransform.Size, typ internaltransform.Type) (int32Len int, int16Len int, err error) {
	_, lossless, err := batch.BlockQuantizer(currentQIndex, segmentID, plane)
	if err != nil {
		return 0, 0, err
	}
	return internalreconstruct.ScratchLen(internalreconstruct.Block{
		Size:      size,
		Transform: typ,
		Lossless:  lossless,
	})
}

// decoderFrameWorkResidualMaxScratchLen reports the worst-case int32 and
// int16 residual scratch lengths needed for any block in the given
// quantizer context. Callers use it to size persistent per-worker buffers.
func decoderFrameWorkResidualMaxScratchLen(batch internaldecoder.FrameWorkBatch, currentQIndex uint8, segmentID uint8, plane internaldecoder.FrameWorkPlane) (int32Len int, int16Len int, err error) {
	_, lossless, err := batch.BlockQuantizer(currentQIndex, segmentID, plane)
	if err != nil {
		return 0, 0, err
	}
	return reconstructBlockMaxScratchLen(lossless)
}

// initDecoderFrameWorkTileResidualCDFStorageDefault initializes storage
// with the AV1 default CDF tables for baseQIndex. Use it once before
// decoding the first frame of a coded video sequence.
func initDecoderFrameWorkTileResidualCDFStorageDefault(storage *internalthreading.FrameWorkTileResidualCDFStorage, baseQIndex uint8) error {
	return storage.InitDefault(baseQIndex)
}

// decoderFrameWorkTileResidualCDFsFromStorage returns the CDF view
// bound from storage, ready to thread into a residual decode call.
func decoderFrameWorkTileResidualCDFsFromStorage(storage *internalthreading.FrameWorkTileResidualCDFStorage) internalthreading.FrameWorkTileResidualCDFs {
	return storage.CDFs()
}

// initDecoderFrameWorkTileResidualCDFStorage initializes storage from
// the active frame's primary reference (or default CDFs when the frame
// is keyframe / has no primary reference) using the batch context.
func initDecoderFrameWorkTileResidualCDFStorage(batch internaldecoder.FrameWorkBatch, storage *internalthreading.FrameWorkTileResidualCDFStorage) error {
	return batch.InitTileResidualCDFStorage(storage)
}

// initDecoderFrameWorkJobDecodeState populates state from the per-job
// metadata in batch (tile payload span, entropy reader setup,
// segmentation/loop-filter cache, CDF retention flag).
func initDecoderFrameWorkJobDecodeState(batch internaldecoder.FrameWorkBatch, index int, state *internaltile.DecodeState) error {
	return batch.JobDecodeState(index, state)
}

// retainDecoderFrameWorkTileResidualCDFStorage copies the adapted CDF
// state from state into storage when index identifies the
// context-update tile for the frame. Other jobs leave storage
// unchanged.
func retainDecoderFrameWorkTileResidualCDFStorage(batch internaldecoder.FrameWorkBatch, index int, state *internaltile.DecodeState, storage *internalthreading.FrameWorkTileResidualCDFStorage) error {
	return batch.RetainTileResidualCDFStorage(index, state, storage)
}

// initDecoderFrameWorkTileRestorationRequestReferences populates the
// reference-frame pointers inside req so loop-restoration helpers can
// dispatch through them without re-resolving the references.
func initDecoderFrameWorkTileRestorationRequestReferences(req *internalthreading.FrameWorkTileRestorationRequest) error {
	return req.InitReferences()
}

// decoderFrameWorkJobBlockLoopContextRootColumns reports the number of
// root-column above-context entries one job's block-loop carrier needs
// to track for its tile.
func decoderFrameWorkJobBlockLoopContextRootColumns(batch internaldecoder.FrameWorkBatch, index int) (int, error) {
	return batch.JobBlockLoopContextRootColumns(index)
}

// bindTileBlockLoopContextCarrier wraps a caller-owned above-context
// slice (length >= rootColumns) into a TileBlockLoopContextCarrier
// ready for use with TileBlockLoopRequest. The first rootColumns
// entries are zeroed before return.
func bindTileBlockLoopContextCarrier(rootColumns int, above []internaltile.BlockLoopRootAboveContext) (internaltile.BlockLoopContextCarrier, error) {
	if rootColumns < 0 {
		return internaltile.BlockLoopContextCarrier{}, internaltile.ErrInvalidDecodeState
	}
	if len(above) < rootColumns {
		return internaltile.BlockLoopContextCarrier{}, internalframe.ErrShortBuffer
	}
	clear(above[:rootColumns])
	return internaltile.BlockLoopContextCarrier{
		Above: above[:rootColumns],
	}, nil
}

// decoderFrameWorkJobBlockLoopRequest builds a TileBlockLoopRequest
// for one job in batch, wiring the caller-owned segmentation maps,
// segment-map stride, and block-loop context carrier into the
// returned request.
func decoderFrameWorkJobBlockLoopRequest(batch internaldecoder.FrameWorkBatch, index int, currentSegmentMap []uint8, previousSegmentMap []uint8, segmentMapStride uint16, carrier *internaltile.BlockLoopContextCarrier) (internaltile.BlockLoopRequest, error) {
	return decoderFrameWorkJobBlockLoopRequestPtr(&batch, index, currentSegmentMap, previousSegmentMap, segmentMapStride, carrier)
}

func decoderFrameWorkJobBlockLoopRequestPtr(batch *internaldecoder.FrameWorkBatch, index int, currentSegmentMap []uint8, previousSegmentMap []uint8, segmentMapStride uint16, carrier *internaltile.BlockLoopContextCarrier) (internaltile.BlockLoopRequest, error) {
	if batch == nil {
		return internaltile.BlockLoopRequest{}, internalthreading.ErrInvalidBatch
	}
	req, err := batch.JobBlockLoopRequest(index, currentSegmentMap, previousSegmentMap, segmentMapStride)
	if err != nil {
		return internaltile.BlockLoopRequest{}, err
	}
	req.ContextCarrier = carrier
	// Decode the full block-mode syntax for every block. block_loop.go gates
	// the inter-only reads on whether a block is actually inter-coded, so
	// enabling all of these on an intra frame is harmless; without them the
	// block loop walks blocks with zero-valued prediction state, which
	// produces wrong reconstruction pixels (and a wrong frame MD5) for both
	// intra and inter frames. This mirrors the byte-exact framework dry-run
	// harness, which sets the same flags before DecodeAndReconstructJobResiduals.
	req.DecodePredictionModes = true
	req.DecodeInterModes = true
	req.DecodeMotionVectors = true
	req.DecodeInterIntra = true
	req.DecodeMotionModes = true
	req.DecodeCompoundBlend = true
	return req, nil
}

// decodeAndReconstructDecoderFrameWorkJobResiduals decodes and
// reconstructs every residual in one tile job. It uses cdfs as the
// per-job CDF view and scratch for residual int32/int16 scratch.
//
// See also: decodeAndRetainDecoderFrameWorkJobResiduals.
func decodeAndReconstructDecoderFrameWorkJobResiduals(batch internaldecoder.FrameWorkBatch, index int, state *internaltile.DecodeState, cdfs internalthreading.FrameWorkTileResidualCDFs, scratch *internalthreading.FrameWorkTileResidualScratch, req internalthreading.FrameWorkTileResidualRequest) (internalthreading.FrameWorkTileResidualStats, error) {
	return batch.DecodeAndReconstructJobResiduals(index, state, cdfs, scratch, req)
}

// decodeAndRetainDecoderFrameWorkJobResiduals is
// decodeAndReconstructDecoderFrameWorkJobResiduals with automatic
// per-job decode-state setup and CDF retention into storage when this
// job's tile is the context-update tile.
func decodeAndRetainDecoderFrameWorkJobResiduals(batch internaldecoder.FrameWorkBatch, index int, state *internaltile.DecodeState, storage *internalthreading.FrameWorkTileResidualCDFStorage, scratch *internalthreading.FrameWorkTileResidualScratch, req internalthreading.FrameWorkTileResidualRequest) (internalthreading.FrameWorkTileResidualStats, error) {
	if storage == nil {
		return internalthreading.FrameWorkTileResidualStats{}, internalthreading.ErrInvalidBatch
	}
	if err := initDecoderFrameWorkJobDecodeState(batch, index, state); err != nil {
		return internalthreading.FrameWorkTileResidualStats{}, err
	}
	stats, err := decodeAndReconstructDecoderFrameWorkJobResiduals(batch, index, state, decoderFrameWorkTileResidualCDFsFromStorage(storage), scratch, req)
	if err != nil {
		return stats, err
	}
	if err := retainDecoderFrameWorkTileResidualCDFStorage(batch, index, state, storage); err != nil {
		return stats, err
	}
	return stats, nil
}

// decoderFrameWorkBatchResidualRequest carries caller-owned state for decoding
// every job assigned to one frame-work worker batch.
type decoderFrameWorkBatchResidualRequest struct {
	Tile internalthreading.FrameWorkTileResidualRequest

	CurrentSegmentMap  []uint8
	PreviousSegmentMap []uint8
	SegmentMapStride   uint16

	// LoopContextAbove optionally backs one job's root-column context at a time.
	// When nil, the helper leaves Tile.Loop.ContextCarrier unchanged.
	LoopContextAbove []internaltile.BlockLoopRootAboveContext
}

// decoderFrameWorkBatchResidualScratchSize reports the caller-owned scratch
// needed to decode all jobs in one frame-work worker batch.
type decoderFrameWorkBatchResidualScratchSize struct {
	Int32Scratch     int
	ResidualScratch  int
	LoopContextAbove int
}

// decoderFrameWorkBatchResidualScratch carries typed caller-owned arenas for
// bindDecoderFrameWorkBatchResidualRequestScratch.
type decoderFrameWorkBatchResidualScratch struct {
	Int32Scratch     []int32
	ResidualScratch  []int16
	LoopContextAbove []internaltile.BlockLoopRootAboveContext
}

// Max returns per-arena maximum lengths for reusable residual batch scratch.
func (s decoderFrameWorkBatchResidualScratchSize) Max(other decoderFrameWorkBatchResidualScratchSize) decoderFrameWorkBatchResidualScratchSize {
	return decoderFrameWorkBatchResidualScratchSize{
		Int32Scratch:     max(s.Int32Scratch, other.Int32Scratch),
		ResidualScratch:  max(s.ResidualScratch, other.ResidualScratch),
		LoopContextAbove: max(s.LoopContextAbove, other.LoopContextAbove),
	}
}

// decoderFrameWorkBatchResidualRunnerScratchSize reports the caller-owned
// per-worker state and flat scratch needed by decoderFrameWorkBatchResidualRunner.
type decoderFrameWorkBatchResidualRunnerScratchSize struct {
	Workers int
	Batch   decoderFrameWorkBatchResidualScratchSize

	RestorationRequests int
	Int32Scratch        int
	ResidualScratch     int
	LoopContextAbove    int
	// LoopFilterMaskCols is the frame's MI-column count, used to pre-size each
	// worker's loop-filter mask-build above-context (aY/aUV) at bind time so the
	// first decode of a fresh decoder does not allocate them.
	LoopFilterMaskCols int
}

// decoderFrameWorkBatchResidualRunnerScratch carries caller-owned storage for
// bindDecoderFrameWorkBatchResidualRunner.
type decoderFrameWorkBatchResidualRunnerScratch struct {
	States                  []internaltile.DecodeState
	Storages                []internalthreading.FrameWorkTileResidualCDFStorage
	TileScratch             []internalthreading.FrameWorkTileResidualScratch
	RestorationRequests     []internalthreading.FrameWorkTileRestorationRequest
	PredictionScratch       []internalthreading.FrameWorkPredictionScratch
	InterPredictionScratch  []internalthreading.FrameWorkInterPredictionScratch
	Stats                   []internalthreading.FrameWorkTileResidualStats
	Int32Scratch            []int32
	ResidualScratch         []int16
	LoopContextAboveScratch []internaltile.BlockLoopRootAboveContext
}

// Max returns per-arena maximum lengths for reusable residual runner scratch.
func (s decoderFrameWorkBatchResidualRunnerScratchSize) Max(other decoderFrameWorkBatchResidualRunnerScratchSize) decoderFrameWorkBatchResidualRunnerScratchSize {
	return decoderFrameWorkBatchResidualRunnerScratchSize{
		Workers:             max(s.Workers, other.Workers),
		Batch:               s.Batch.Max(other.Batch),
		RestorationRequests: max(s.RestorationRequests, other.RestorationRequests),
		Int32Scratch:        max(s.Int32Scratch, other.Int32Scratch),
		ResidualScratch:     max(s.ResidualScratch, other.ResidualScratch),
		LoopContextAbove:    max(s.LoopContextAbove, other.LoopContextAbove),
		LoopFilterMaskCols:  max(s.LoopFilterMaskCols, other.LoopFilterMaskCols),
	}
}

// decoderFrameWorkBatchResidualRunner adapts the public residual batch decoder
// to DecoderFrameWorkBatchFunc with caller-owned per-worker state.
type decoderFrameWorkBatchResidualRunner struct {
	Request decoderFrameWorkBatchResidualRequest
	// UseDefaultPrediction wires each worker's PredictionScratch into Request
	// when Request.Tile.PredictionScratch is nil. Bound runners default this to
	// true so event-level decode paths perform reconstruction prediction without
	// a custom per-block callback.
	UseDefaultPrediction bool

	States                 []internaltile.DecodeState
	Storages               []internalthreading.FrameWorkTileResidualCDFStorage
	TileScratch            []internalthreading.FrameWorkTileResidualScratch
	RestorationRequests    []internalthreading.FrameWorkTileRestorationRequest
	PredictionScratch      []internalthreading.FrameWorkPredictionScratch
	InterPredictionScratch []internalthreading.FrameWorkInterPredictionScratch
	Stats                  []internalthreading.FrameWorkTileResidualStats

	BatchScratchSize decoderFrameWorkBatchResidualScratchSize
	Int32Scratch     []int32
	ResidualScratch  []int16
	LoopContextAbove []internaltile.BlockLoopRootAboveContext

	CDEFIndexMap              internaldecoder.FrameWorkCDEFIndexMap
	LoopFilterMap             internaldecoder.FrameWorkLoopFilterMap
	LoopFilterMasks           internalthreading.FrameWorkLoopFilterMasks
	sideDataValid             bool
	sideDataRestorationActive bool
	compactState              *internalthreading.FrameWorkLoopFilterCompactState
	compactOwner              *decoderFrameWorkBatchResidualRunner
	compactSideOwner          *decoderFrameWorkSideData
	compactBoundSide          *decoderFrameWorkSideData
}

// decoderFrameWorkResidualMaxFrameScratchLen reports the worst-case
// per-block residual scratch lengths needed by any plane/segment in
// the current frame. Callers size persistent per-worker scratch from
// the returned values.
func decoderFrameWorkResidualMaxFrameScratchLen(batch internaldecoder.FrameWorkBatch, currentQIndex uint8) (int32Len int, int16Len int, err error) {
	planes := [3]internaldecoder.FrameWorkPlane{internaldecoder.FrameWorkPlaneY}
	planeCount := 1
	if !batch.Sequence.ColorConfig.MonoChrome {
		planes[1] = internaldecoder.FrameWorkPlaneU
		planes[2] = internaldecoder.FrameWorkPlaneV
		planeCount = 3
	}
	segmentCount := 1
	if batch.Segmentation.Enabled {
		segmentCount = internalparser.MaxSegments
	}
	for segmentID := 0; segmentID < segmentCount; segmentID++ {
		for plane := 0; plane < planeCount; plane++ {
			next32, next16, err := decoderFrameWorkResidualMaxScratchLen(batch, currentQIndex, uint8(segmentID), planes[plane])
			if err != nil {
				return 0, 0, err
			}
			if next32 > int32Len {
				int32Len = next32
			}
			if next16 > int16Len {
				int16Len = next16
			}
		}
	}
	return int32Len, int16Len, nil
}

// decoderFrameWorkBatchResidualScratchLen reports the caller-owned
// scratch lengths required to decode every job in one worker batch
// (per-batch int32, residual, and loop-context-above lengths).
//
// See also: bindDecoderFrameWorkBatchResidualRequestScratch.
func decoderFrameWorkBatchResidualScratchLen(batch internaldecoder.FrameWorkBatch, currentQIndex uint8) (decoderFrameWorkBatchResidualScratchSize, error) {
	int32Len, residualLen, err := decoderFrameWorkResidualMaxFrameScratchLen(batch, currentQIndex)
	if err != nil {
		return decoderFrameWorkBatchResidualScratchSize{}, err
	}
	loopContextLen, err := decoderFrameWorkBatchResidualLoopContextAboveLen(batch)
	if err != nil {
		return decoderFrameWorkBatchResidualScratchSize{}, err
	}
	return decoderFrameWorkBatchResidualScratchSize{
		Int32Scratch:     int32Len,
		ResidualScratch:  residualLen,
		LoopContextAbove: loopContextLen,
	}, nil
}

// decoderFrameWorkBatchResidualRunnerScratchLen reports the
// caller-owned per-worker state and flat scratch lengths required by
// decoderFrameWorkBatchResidualRunner across workers workers.
//
// See also: bindDecoderFrameWorkBatchResidualRunner.
func decoderFrameWorkBatchResidualRunnerScratchLen(batch internaldecoder.FrameWorkBatch, workers int) (decoderFrameWorkBatchResidualRunnerScratchSize, error) {
	if workers <= 0 {
		return decoderFrameWorkBatchResidualRunnerScratchSize{}, internalthreading.ErrInvalidWorkerCount
	}
	batchSize, err := decoderFrameWorkBatchResidualScratchLen(batch, batch.Quantization.BaseQIdx)
	if err != nil {
		return decoderFrameWorkBatchResidualRunnerScratchSize{}, err
	}
	int32Len, ok := decoderFrameWorkResidualCheckedMul(batchSize.Int32Scratch, workers)
	if !ok {
		return decoderFrameWorkBatchResidualRunnerScratchSize{}, internalframe.ErrShortBuffer
	}
	residualLen, ok := decoderFrameWorkResidualCheckedMul(batchSize.ResidualScratch, workers)
	if !ok {
		return decoderFrameWorkBatchResidualRunnerScratchSize{}, internalframe.ErrShortBuffer
	}
	loopContextLen, ok := decoderFrameWorkResidualCheckedMul(batchSize.LoopContextAbove, workers)
	if !ok {
		return decoderFrameWorkBatchResidualRunnerScratchSize{}, internalframe.ErrShortBuffer
	}
	maskCols, _, _, err := batch.LoopFilterMapShape()
	if err != nil {
		return decoderFrameWorkBatchResidualRunnerScratchSize{}, err
	}
	return decoderFrameWorkBatchResidualRunnerScratchSize{
		Workers:             workers,
		Batch:               batchSize,
		RestorationRequests: workers,
		Int32Scratch:        int32Len,
		ResidualScratch:     residualLen,
		LoopContextAbove:    loopContextLen,
		LoopFilterMaskCols:  maskCols,
	}, nil
}

// bindDecoderFrameWorkBatchResidualRequestScratch slices caller-owned
// scratch into a decoderFrameWorkBatchResidualRequest sized for size.
// Callers fill the remaining request fields (segmentation maps,
// stride, tile loop overrides) before invoking the residual decoder.
func bindDecoderFrameWorkBatchResidualRequestScratch(size decoderFrameWorkBatchResidualScratchSize, scratch decoderFrameWorkBatchResidualScratch) (decoderFrameWorkBatchResidualRequest, error) {
	if decoderFrameWorkResidualScratchTooShort(scratch.Int32Scratch, size.Int32Scratch) ||
		decoderFrameWorkResidualScratchTooShort(scratch.ResidualScratch, size.ResidualScratch) ||
		decoderFrameWorkResidualScratchTooShort(scratch.LoopContextAbove, size.LoopContextAbove) {
		return decoderFrameWorkBatchResidualRequest{}, internalframe.ErrShortBuffer
	}
	return decoderFrameWorkBatchResidualRequest{
		Tile: internalthreading.FrameWorkTileResidualRequest{
			Int32Scratch:    scratch.Int32Scratch[:size.Int32Scratch],
			ResidualScratch: scratch.ResidualScratch[:size.ResidualScratch],
		},
		LoopContextAbove: scratch.LoopContextAbove[:size.LoopContextAbove],
	}, nil
}

// bindDecoderFrameWorkBatchResidualRunner slices caller-owned
// per-worker state and flat scratch into a
// decoderFrameWorkBatchResidualRunner. The returned runner is the
// DecoderFrameWorkBatchFunc-compatible executor that decodes every
// job in a batch using one worker's slot.
//
// See also: decoderFrameWorkBatchResidualRunner.Run.
func bindDecoderFrameWorkBatchResidualRunner(size decoderFrameWorkBatchResidualRunnerScratchSize, scratch decoderFrameWorkBatchResidualRunnerScratch) (decoderFrameWorkBatchResidualRunner, error) {
	if size.Workers <= 0 {
		return decoderFrameWorkBatchResidualRunner{}, internalthreading.ErrInvalidWorkerCount
	}
	if err := decoderFrameWorkBatchResidualRunnerScratchErr(size, scratch); err != nil {
		return decoderFrameWorkBatchResidualRunner{}, err
	}
	var restorationRequests []internalthreading.FrameWorkTileRestorationRequest
	if len(scratch.RestorationRequests) != 0 {
		restorationRequests = scratch.RestorationRequests[:size.RestorationRequests]
	}
	predictionScratch := scratch.PredictionScratch[:size.Workers]
	interScratch := scratch.InterPredictionScratch[:size.Workers]
	tileScratch := scratch.TileScratch[:size.Workers]
	for i := range predictionScratch {
		predictionScratch[i].Inter = &interScratch[i]
		tileScratch[i].PreallocCallbackScratch()
		tileScratch[i].PreallocLoopFilterMaskScratch(size.LoopFilterMaskCols)
		internaltile.PreallocBlockLoopContextCarrierScratch(&tileScratch[i].LoopContext, size.Batch.LoopContextAbove)
	}
	return decoderFrameWorkBatchResidualRunner{
		UseDefaultPrediction:   true,
		States:                 scratch.States[:size.Workers],
		Storages:               scratch.Storages[:size.Workers],
		TileScratch:            tileScratch,
		RestorationRequests:    restorationRequests,
		PredictionScratch:      predictionScratch,
		InterPredictionScratch: interScratch,
		Stats:                  scratch.Stats[:size.Workers],
		BatchScratchSize:       size.Batch,
		Int32Scratch:           scratch.Int32Scratch[:size.Int32Scratch],
		ResidualScratch:        scratch.ResidualScratch[:size.ResidualScratch],
		LoopContextAbove:       scratch.LoopContextAboveScratch[:size.LoopContextAbove],
	}, nil
}

func decoderFrameWorkBatchResidualRunnerScratchErr(size decoderFrameWorkBatchResidualRunnerScratchSize, scratch decoderFrameWorkBatchResidualRunnerScratch) error {
	if size.Workers == 0 {
		return nil
	}
	if size.Workers < 0 {
		return internalthreading.ErrInvalidWorkerCount
	}
	if size.RestorationRequests < 0 ||
		size.Int32Scratch < 0 ||
		size.ResidualScratch < 0 ||
		size.LoopContextAbove < 0 ||
		size.Batch.Int32Scratch < 0 ||
		size.Batch.ResidualScratch < 0 ||
		size.Batch.LoopContextAbove < 0 {
		return internalframe.ErrShortBuffer
	}
	if decoderFrameWorkResidualScratchTooShort(scratch.States, size.Workers) ||
		decoderFrameWorkResidualScratchTooShort(scratch.Storages, size.Workers) ||
		decoderFrameWorkResidualScratchTooShort(scratch.TileScratch, size.Workers) ||
		decoderFrameWorkResidualScratchTooShort(scratch.PredictionScratch, size.Workers) ||
		decoderFrameWorkResidualScratchTooShort(scratch.InterPredictionScratch, size.Workers) ||
		decoderFrameWorkResidualScratchTooShort(scratch.Stats, size.Workers) ||
		decoderFrameWorkResidualScratchTooShort(scratch.Int32Scratch, size.Int32Scratch) ||
		decoderFrameWorkResidualScratchTooShort(scratch.ResidualScratch, size.ResidualScratch) ||
		decoderFrameWorkResidualScratchTooShort(scratch.LoopContextAboveScratch, size.LoopContextAbove) {
		return internalframe.ErrShortBuffer
	}
	if len(scratch.RestorationRequests) != 0 && decoderFrameWorkResidualScratchTooShort(scratch.RestorationRequests, size.RestorationRequests) {
		return internalframe.ErrShortBuffer
	}
	return nil
}

// Run decodes every residual in batch using the worker slot identified
// by batch.Batch.Worker. It is the DecoderFrameWorkBatchFunc form of
// the runner; pass r.Run to ExecuteDecoderFrameWorkStepWithContext.
func (r *decoderFrameWorkBatchResidualRunner) Run(batch internaldecoder.FrameWorkBatch) error {
	return r.RunPtr(&batch)
}

// RunPtr decodes every residual in batch without copying the frame-work
// context into the runner call. It is used by the internal single-worker typed
// path; public callers should keep using Run.
func (r *decoderFrameWorkBatchResidualRunner) RunPtr(batch *internaldecoder.FrameWorkBatch) (runErr error) {
	if r != nil && internalthreading.FrameWorkLoopFilterCompactStateActive(r.compactState) {
		defer func() {
			if runErr != nil {
				internalthreading.InvalidateFrameWorkLoopFilterCompactState(r.compactState)
			}
		}()
		if r.compactOwner != r || r.compactSideOwner == nil || r.compactBoundSide != r.compactSideOwner ||
			!decoderFrameWorkResidualCompactBatchValid(r, batch) {
			return internalthreading.ErrInvalidBatch
		}
	}
	if batch == nil {
		return internalthreading.ErrInvalidBatch
	}
	worker, err := r.workerIndex(*batch)
	if err != nil {
		return err
	}
	r.Stats[worker] = internalthreading.FrameWorkTileResidualStats{}
	req, err := r.workerRequest(worker)
	if err != nil {
		return err
	}
	req.Tile.TransformMode = batch.TransformRef.TransformMode
	if err := initDecoderFrameWorkTileResidualCDFStorage(*batch, &r.Storages[worker]); err != nil {
		return err
	}
	stats, err := decodeAndRetainDecoderFrameWorkBatchResidualsPtr(batch, &r.States[worker], &r.Storages[worker], &r.TileScratch[worker], req, r.compactState)
	r.Stats[worker] = stats
	if err != nil {
		return err
	}
	if internalthreading.FrameWorkLoopFilterCompactStateActive(r.compactState) {
		if err := internalthreading.CompleteFrameWorkLoopFilterCompactState(r.compactState); err != nil {
			return err
		}
	}
	return nil
}

// SetSideData attaches bound frame-level side data (CDEF index map,
// loop-filter map, loop-restoration frame buffers) to the runner so
// the residual decode pipeline can populate them as it walks blocks.
// It also resets each side-data structure for a fresh frame.
func (r *decoderFrameWorkBatchResidualRunner) SetSideData(side decoderFrameWorkSideData) error {
	if r == nil {
		return internalthreading.ErrInvalidBatch
	}
	if side.RestorationFrameBuffers.Plan.Active && len(r.RestorationRequests) < len(r.States) {
		return internalframe.ErrShortBuffer
	}
	if err := side.CDEFIndexMap.Reset(); err != nil {
		return err
	}
	if err := side.LoopFilterMap.Reset(); err != nil {
		return err
	}
	if err := side.RestorationFrameBuffers.ResetRecords(); err != nil {
		return err
	}
	if r.compactState != nil {
		internalthreading.DeactivateFrameWorkLoopFilterCompactState(r.compactState)
	}
	if side.RestorationFrameBuffers.Plan.Active {
		for i := range r.States {
			req := &r.RestorationRequests[i]
			req.Buffers = side.RestorationFrameBuffers
			if err := initDecoderFrameWorkTileRestorationRequestReferences(req); err != nil {
				return err
			}
		}
	}
	r.CDEFIndexMap = side.CDEFIndexMap
	r.LoopFilterMap = side.LoopFilterMap
	r.LoopFilterMasks = side.LoopFilterMasks
	r.sideDataValid = true
	r.sideDataRestorationActive = side.RestorationFrameBuffers.Plan.Active
	r.compactBoundSide = nil
	return nil
}

func (r *decoderFrameWorkBatchResidualRunner) setOwnedCompactSideData(side *decoderFrameWorkSideData) error {
	if r == nil || side == nil || r.compactOwner != r || r.compactSideOwner == nil || side != r.compactSideOwner ||
		r.compactState == nil || !internalthreading.FrameWorkLoopFilterCompactStateActive(r.compactState) || !side.LoopFilterMasks.Valid() {
		return internalthreading.ErrInvalidBatch
	}
	if side.RestorationFrameBuffers.Plan.Active && len(r.RestorationRequests) < len(r.States) {
		return internalframe.ErrShortBuffer
	}
	if err := side.CDEFIndexMap.Reset(); err != nil {
		return err
	}
	if err := side.RestorationFrameBuffers.ResetRecords(); err != nil {
		return err
	}
	if side.RestorationFrameBuffers.Plan.Active {
		for i := range r.States {
			req := &r.RestorationRequests[i]
			req.Buffers = side.RestorationFrameBuffers
			if err := initDecoderFrameWorkTileRestorationRequestReferences(req); err != nil {
				return err
			}
		}
	}
	r.LoopFilterMap = internaldecoder.FrameWorkLoopFilterMap{}
	r.LoopFilterMasks = side.LoopFilterMasks
	r.CDEFIndexMap = side.CDEFIndexMap
	r.sideDataValid = true
	r.sideDataRestorationActive = side.RestorationFrameBuffers.Plan.Active
	r.compactBoundSide = side
	return nil
}

// setDecoderFrameWorkBatchResidualRunnerSideData is the function-style
// alias for (*decoderFrameWorkBatchResidualRunner).SetSideData provided
// for callers that prefer free functions.
func setDecoderFrameWorkBatchResidualRunnerSideData(runner *decoderFrameWorkBatchResidualRunner, side decoderFrameWorkSideData) error {
	return runner.SetSideData(side)
}

// ResetStats clears the per-worker statistics buckets so the next
// frame's residual decode starts with zeroed counters.
func (r *decoderFrameWorkBatchResidualRunner) ResetStats() error {
	if r == nil {
		return internalthreading.ErrInvalidBatch
	}
	clear(r.Stats)
	return nil
}

// TotalStats aggregates the per-worker counters into one
// DecoderFrameWorkTileResidualStats value for the current frame.
func (r *decoderFrameWorkBatchResidualRunner) TotalStats() (internalthreading.FrameWorkTileResidualStats, error) {
	if r == nil {
		return internalthreading.FrameWorkTileResidualStats{}, internalthreading.ErrInvalidBatch
	}
	var total internalthreading.FrameWorkTileResidualStats
	for _, stats := range r.Stats {
		decoderFrameWorkAccumulateResidualStats(&total, stats)
	}
	return total, nil
}

// decoderFrameWorkBatchResidualLoopContextAboveLen reports the
// worst-case root-column above-context length across the jobs in
// batch, used to size the per-batch loop-context above scratch.
func decoderFrameWorkBatchResidualLoopContextAboveLen(batch internaldecoder.FrameWorkBatch) (int, error) {
	maxRootColumns := 0
	for i := range batch.Jobs {
		rootColumns, err := decoderFrameWorkJobBlockLoopContextRootColumns(batch, i)
		if err != nil {
			return 0, err
		}
		if rootColumns > maxRootColumns {
			maxRootColumns = rootColumns
		}
	}
	return maxRootColumns, nil
}

// decodeAndRetainDecoderFrameWorkBatchResiduals decodes every job in
// batch in order, accumulating per-tile statistics and retaining the
// adapted CDF state for the context-update tile. It is the
// reusable, lower-level building block underneath
// decoderFrameWorkBatchResidualRunner.Run.
func decodeAndRetainDecoderFrameWorkBatchResiduals(batch internaldecoder.FrameWorkBatch, state *internaltile.DecodeState, storage *internalthreading.FrameWorkTileResidualCDFStorage, scratch *internalthreading.FrameWorkTileResidualScratch, req decoderFrameWorkBatchResidualRequest) (internalthreading.FrameWorkTileResidualStats, error) {
	return decodeAndRetainDecoderFrameWorkBatchResidualsPtr(&batch, state, storage, scratch, req, nil)
}

func decodeAndRetainDecoderFrameWorkBatchResidualsPtr(batch *internaldecoder.FrameWorkBatch, state *internaltile.DecodeState, storage *internalthreading.FrameWorkTileResidualCDFStorage, scratch *internalthreading.FrameWorkTileResidualScratch, req decoderFrameWorkBatchResidualRequest, compact *internalthreading.FrameWorkLoopFilterCompactState) (internalthreading.FrameWorkTileResidualStats, error) {
	if state == nil || storage == nil || scratch == nil {
		return internalthreading.FrameWorkTileResidualStats{}, internalthreading.ErrInvalidBatch
	}
	if batch == nil {
		return internalthreading.FrameWorkTileResidualStats{}, internalthreading.ErrInvalidBatch
	}
	var total internalthreading.FrameWorkTileResidualStats
	for i := range batch.Jobs {
		// Re-initialize the entropy (CDF) context from the frame context before
		// each tile in the batch. AV1 decodes every tile from the same starting
		// context (libaom: tile_data->tctx = *cm->fc inside the per-tile loop of
		// decode_tiles), so a tile must never inherit the adapted CDFs of an
		// earlier tile that shared this worker batch. Without this reset, the
		// first tile decodes correctly but every subsequent tile in the same
		// batch starts from desynced CDFs and produces wrong pixels. The per-job
		// retain that follows still captures the context_update_tile_id tile's
		// adapted CDFs (retainDecoderFrameWorkTileResidualCDFStorage copies them
		// out before the next iteration re-initializes the shared storage).
		if i > 0 {
			if err := batch.InitTileResidualCDFStorage(storage); err != nil {
				return total, err
			}
		}
		tileReq := req.Tile
		loopReq, err := decoderFrameWorkJobBlockLoopRequestPtr(batch, i, req.CurrentSegmentMap, req.PreviousSegmentMap, req.SegmentMapStride, req.Tile.Loop.ContextCarrier)
		if err != nil {
			return total, err
		}
		if req.LoopContextAbove != nil {
			rootColumns, err := batch.JobBlockLoopContextRootColumns(i)
			if err != nil {
				return total, err
			}
			carrier, err := bindTileBlockLoopContextCarrier(rootColumns, req.LoopContextAbove)
			if err != nil {
				return total, err
			}
			// Preserve already-allocated diagonal-corner scratch slices
			// across binds so the per-frame loop does not re-allocate
			// them. The carrier returned by Bind has Diagonal/PendingDiagonal
			// nil; reuse the persistent scratch from scratch.LoopContext.
			carrier.Diagonal = scratch.LoopContext.Diagonal
			carrier.PendingDiagonal = scratch.LoopContext.PendingDiagonal
			scratch.LoopContext = carrier
			loopReq.ContextCarrier = &scratch.LoopContext
		}
		decoderFrameWorkApplyBatchResidualLoopOverrides(&loopReq, req.Tile.Loop)
		tileReq.Loop = loopReq
		stats, err := decodeAndRetainDecoderFrameWorkJobResidualsPtr(batch, i, state, storage, scratch, tileReq, compact)
		decoderFrameWorkAccumulateResidualStats(&total, stats)
		if err != nil {
			return total, err
		}
	}
	return total, nil
}

func decodeAndRetainDecoderFrameWorkJobResidualsPtr(batch *internaldecoder.FrameWorkBatch, index int, state *internaltile.DecodeState, storage *internalthreading.FrameWorkTileResidualCDFStorage, scratch *internalthreading.FrameWorkTileResidualScratch, req internalthreading.FrameWorkTileResidualRequest, compact *internalthreading.FrameWorkLoopFilterCompactState) (internalthreading.FrameWorkTileResidualStats, error) {
	if storage == nil || batch == nil {
		return internalthreading.FrameWorkTileResidualStats{}, internalthreading.ErrInvalidBatch
	}
	if err := batch.JobDecodeState(index, state); err != nil {
		return internalthreading.FrameWorkTileResidualStats{}, err
	}
	var stats internalthreading.FrameWorkTileResidualStats
	var err error
	if internalthreading.FrameWorkLoopFilterCompactStateActive(compact) {
		stats, err = internalthreading.DecodeAndReconstructJobResidualsPtrOwned(batch, index, state, decoderFrameWorkTileResidualCDFsFromStorage(storage), scratch, req, compact)
	} else {
		stats, err = batch.DecodeAndReconstructJobResidualsPtr(index, state, decoderFrameWorkTileResidualCDFsFromStorage(storage), scratch, req)
	}
	if err != nil {
		return stats, err
	}
	if err := internalthreading.RetainTileResidualCDFStorageOwned(batch, index, state, storage, &req, compact); err != nil {
		return stats, err
	}
	return stats, nil
}

func decoderFrameWorkResidualCompactBatchValid(r *decoderFrameWorkBatchResidualRunner, batch *internaldecoder.FrameWorkBatch) bool {
	if r == nil || batch == nil || r.compactState == nil || !r.sideDataValid || len(r.States) != 1 ||
		batch.Batch.Worker != 0 || batch.Batch.Count != 1 || batch.Batch.FirstJob != 0 ||
		batch.WavefrontWorkers != 0 || len(batch.Jobs) != 1 ||
		batch.Jobs[0].Tile != 0 || batch.Jobs[0].Row != 0 || batch.Jobs[0].Col != 0 ||
		batch.TileInfo.Cols != 1 || batch.TileInfo.Rows != 1 ||
		batch.Sequence.ColorConfig.BitDepth != 8 || batch.FrameSize.SuperResEnabled || batch.FrameSize.AllowIntrabc ||
		batch.LoopFilterMap != nil || batch.LoopFilterMasks == nil ||
		!(batch.LoopFilter.LevelY[0] != 0 || batch.LoopFilter.LevelY[1] != 0) {
		return false
	}
	cols, rows, _, err := batch.LoopFilterMapShape()
	if err != nil || !internalthreading.FrameWorkLoopFilterCompactActiveShape(batch.LoopFilterMasks, r.compactState, cols, rows) {
		return false
	}
	color := batch.Sequence.ColorConfig
	ssHor, ssVer := 0, 0
	if color.SubsamplingX {
		ssHor = 1
	}
	if color.SubsamplingY {
		ssVer = 1
	}
	layout := batch.LoopFilterMasks.Layout
	return layout.Mono == color.MonoChrome && layout.SSHor == ssHor && layout.SSVer == ssVer &&
		batch.LoopFilterMasks.HasChroma == !color.MonoChrome
}

func (r *decoderFrameWorkBatchResidualRunner) workerIndex(batch internaldecoder.FrameWorkBatch) (int, error) {
	if r == nil {
		return 0, internalthreading.ErrInvalidBatch
	}
	worker := int(batch.Batch.Worker)
	if worker < 0 || worker >= len(r.States) ||
		worker >= len(r.Storages) ||
		worker >= len(r.TileScratch) ||
		worker >= len(r.PredictionScratch) ||
		worker >= len(r.InterPredictionScratch) ||
		worker >= len(r.Stats) {
		return 0, internalthreading.ErrInvalidBatch
	}
	return worker, nil
}

func (r *decoderFrameWorkBatchResidualRunner) workerRequest(worker int) (decoderFrameWorkBatchResidualRequest, error) {
	req := r.Request
	int32Offset, ok := decoderFrameWorkResidualCheckedMul(worker, r.BatchScratchSize.Int32Scratch)
	if !ok {
		return decoderFrameWorkBatchResidualRequest{}, internalframe.ErrShortBuffer
	}
	residualOffset, ok := decoderFrameWorkResidualCheckedMul(worker, r.BatchScratchSize.ResidualScratch)
	if !ok {
		return decoderFrameWorkBatchResidualRequest{}, internalframe.ErrShortBuffer
	}
	loopOffset, ok := decoderFrameWorkResidualCheckedMul(worker, r.BatchScratchSize.LoopContextAbove)
	if !ok {
		return decoderFrameWorkBatchResidualRequest{}, internalframe.ErrShortBuffer
	}
	if int32Offset > len(r.Int32Scratch) ||
		residualOffset > len(r.ResidualScratch) ||
		loopOffset > len(r.LoopContextAbove) {
		return decoderFrameWorkBatchResidualRequest{}, internalframe.ErrShortBuffer
	}
	scratch := decoderFrameWorkBatchResidualScratch{
		Int32Scratch:     r.Int32Scratch[int32Offset:],
		ResidualScratch:  r.ResidualScratch[residualOffset:],
		LoopContextAbove: r.LoopContextAbove[loopOffset:],
	}
	bound, err := bindDecoderFrameWorkBatchResidualRequestScratch(r.BatchScratchSize, scratch)
	if err != nil {
		return decoderFrameWorkBatchResidualRequest{}, err
	}
	req.Tile.Int32Scratch = bound.Tile.Int32Scratch
	req.Tile.ResidualScratch = bound.Tile.ResidualScratch
	req.LoopContextAbove = bound.LoopContextAbove
	if r.sideDataValid {
		req.Tile.CDEFIndexMap = &r.CDEFIndexMap
		if internalthreading.FrameWorkLoopFilterCompactStateActive(r.compactState) {
			if r.compactOwner != r || r.compactSideOwner == nil || r.compactBoundSide != r.compactSideOwner {
				return decoderFrameWorkBatchResidualRequest{}, internalthreading.ErrInvalidBatch
			}
			req.Tile.LoopFilterMap = nil
		} else {
			req.Tile.LoopFilterMap = &r.LoopFilterMap
		}
		req.Tile.LoopFilterMasks = &r.LoopFilterMasks
		if r.sideDataRestorationActive {
			if worker >= len(r.RestorationRequests) {
				return decoderFrameWorkBatchResidualRequest{}, internalframe.ErrShortBuffer
			}
			req.Tile.Restoration = &r.RestorationRequests[worker]
		}
	}
	if req.Tile.PredictionScratch == nil && r.UseDefaultPrediction {
		req.Tile.PredictionScratch = &r.PredictionScratch[worker]
	}
	return req, nil
}

func decoderFrameWorkResidualCheckedMul(a int, b int) (int, bool) {
	if a < 0 || b < 0 {
		return 0, false
	}
	if a != 0 && b > int(^uint(0)>>1)/a {
		return 0, false
	}
	return a * b, true
}

func decoderFrameWorkResidualScratchTooShort[T any](scratch []T, need int) bool {
	return need < 0 || len(scratch) < need
}

func decoderFrameWorkApplyBatchResidualLoopOverrides(dst *internaltile.BlockLoopRequest, overrides internaltile.BlockLoopRequest) {
	if overrides.ContextCarrier != nil {
		dst.ContextCarrier = overrides.ContextCarrier
	}
	dst.BeforeSuperblock = overrides.BeforeSuperblock
}

func decoderFrameWorkAccumulateResidualStats(total *internalthreading.FrameWorkTileResidualStats, next internalthreading.FrameWorkTileResidualStats) {
	total.Loop.PartitionReads += next.Loop.PartitionReads
	total.Loop.Blocks += next.Loop.Blocks
	total.Loop.CoefficientBlocks += next.Loop.CoefficientBlocks
	total.CoefficientBlocks += next.CoefficientBlocks
	total.SkippedBlocks += next.SkippedBlocks
	total.TXBs += next.TXBs
	total.NonZero += next.NonZero
	total.AllZero += next.AllZero
	total.EOBTotal += next.EOBTotal
	total.Residuals += next.Residuals
	total.Predictions += next.Predictions
	total.RestorationUnits += next.RestorationUnits
}
