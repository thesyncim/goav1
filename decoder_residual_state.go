package goav1

// This file exposes the caller-owned residual-decode state and the helpers
// that drive coefficient decode, block reconstruction, and the per-job
// residual runner.
//
// The helpers fall into these groups:
//
//   - DecoderFrameWorkResidualScratchLen / MaxScratchLen /
//     MaxFrameScratchLen / BatchResidualScratchLen /
//     BatchResidualRunnerScratchLen: shape helpers that report the
//     caller-owned scratch lengths required for per-block, per-job, and
//     per-batch residual processing.
//   - DecoderFrameWorkBlockQuantizer / BlockQIndex /
//     BlockCoeffPlanePosition: per-block helpers that resolve the
//     quantizer, q-index, and plane position for one block.
//   - InitDecoderFrameWorkTileResidualCDFStorage{,Default} /
//     DecoderFrameWorkTileResidualCDFsFromStorage: caller-owned CDF
//     storage and binding.
//   - InitDecoderFrameWorkJobDecodeState /
//     RetainDecoderFrameWorkTileResidualCDFStorage: per-job CDF lifecycle.
//   - InitDecoderFrameWorkTileRestorationRequestReferences /
//     DecoderFrameWorkJobBlockLoopContextRootColumns /
//     BindTileBlockLoopContextCarrier / DecoderFrameWorkJobBlockLoopRequest:
//     per-job block-loop setup.
//   - ReadDecoderFrameWorkIntraBlockTransforms / Inter / BlockTransforms /
//     ReconstructDecoderFrameWorkBlockCoeff / DecodeAndReconstruct /
//     DecodeAndRetain*: the residual decode + reconstruct pipeline.
//   - DecoderFrameWorkBatchResidualRunner and its
//     BindDecoderFrameWorkBatchResidualRunner, Run, SetSideData,
//     ResetStats, TotalStats helpers: the bounded per-batch executor that
//     ties the residual pipeline to the worker pool.

import internaltile "github.com/thesyncim/goav1/internal/av1/tile"

// DecoderFrameWorkResidualScratchLen reports the int32 and int16 residual
// scratch lengths required to reconstruct one block at (size, typ) for the
// given quantizer context.
func DecoderFrameWorkResidualScratchLen(batch DecoderFrameWorkBatch, currentQIndex uint8, segmentID uint8, plane DecoderFrameWorkPlane, size TransformSize, typ TransformType) (int32Len int, int16Len int, err error) {
	q, lossless, err := batch.BlockQuantizer(currentQIndex, segmentID, plane)
	if err != nil {
		return 0, 0, err
	}
	return ReconstructBlockScratchLen(ReconstructBlock{
		Size:      size,
		Transform: typ,
		Quantizer: q,
		Lossless:  lossless,
	})
}

// DecoderFrameWorkResidualMaxScratchLen reports the worst-case int32 and
// int16 residual scratch lengths needed for any block in the given
// quantizer context. Callers use it to size persistent per-worker buffers.
func DecoderFrameWorkResidualMaxScratchLen(batch DecoderFrameWorkBatch, currentQIndex uint8, segmentID uint8, plane DecoderFrameWorkPlane) (int32Len int, int16Len int, err error) {
	_, lossless, err := batch.BlockQuantizer(currentQIndex, segmentID, plane)
	if err != nil {
		return 0, 0, err
	}
	return ReconstructBlockMaxScratchLen(lossless)
}

// DecoderFrameWorkBlockQuantizer resolves the per-block Quantizer for the
// given (currentQIndex, segmentID, plane) tuple. The second return value
// reports whether the block is lossless.
func DecoderFrameWorkBlockQuantizer(batch DecoderFrameWorkBatch, currentQIndex uint8, segmentID uint8, plane DecoderFrameWorkPlane) (Quantizer, bool, error) {
	return batch.BlockQuantizer(currentQIndex, segmentID, plane)
}

// DecoderFrameWorkBlockQIndex resolves the per-block q-index for
// (currentQIndex, segmentID). The second return value reports whether
// the block is lossless.
func DecoderFrameWorkBlockQIndex(batch DecoderFrameWorkBatch, currentQIndex uint8, segmentID uint8) (uint8, bool, error) {
	return batch.BlockQIndex(currentQIndex, segmentID)
}

// DecoderFrameWorkBlockCoeffPlanePosition reports the plane and (x, y)
// origin of the coefficient block within the output sample plane.
func DecoderFrameWorkBlockCoeffPlanePosition(batch DecoderFrameWorkBatch, index int, visit TileBlockVisit, block TileBlockCoeffBlock) (DecoderFrameWorkPlane, int, int, error) {
	return batch.BlockCoeffPlanePosition(index, visit, block)
}

// ReconstructDecoderFrameWorkBlockCoeff reconstructs one decoded
// coefficient block into the output sample plane described by req,
// using the prediction scratch the caller already populated.
func ReconstructDecoderFrameWorkBlockCoeff(batch DecoderFrameWorkBatch, index int, req DecoderFrameWorkBlockCoeffReconstruction) error {
	return batch.ReconstructBlockCoeff(index, req)
}

// InitDecoderFrameWorkTileResidualCDFStorageDefault initializes storage
// with the AV1 default CDF tables for baseQIndex. Use it once before
// decoding the first frame of a coded video sequence.
func InitDecoderFrameWorkTileResidualCDFStorageDefault(storage *DecoderFrameWorkTileResidualCDFStorage, baseQIndex uint8) error {
	return storage.InitDefault(baseQIndex)
}

// DecoderFrameWorkTileResidualCDFsFromStorage returns the CDF view
// bound from storage, ready to thread into a residual decode call.
func DecoderFrameWorkTileResidualCDFsFromStorage(storage *DecoderFrameWorkTileResidualCDFStorage) DecoderFrameWorkTileResidualCDFs {
	return storage.CDFs()
}

// InitDecoderFrameWorkTileResidualCDFStorage initializes storage from
// the active frame's primary reference (or default CDFs when the frame
// is keyframe / has no primary reference) using the batch context.
func InitDecoderFrameWorkTileResidualCDFStorage(batch DecoderFrameWorkBatch, storage *DecoderFrameWorkTileResidualCDFStorage) error {
	return batch.InitTileResidualCDFStorage(storage)
}

// InitDecoderFrameWorkJobDecodeState populates state from the per-job
// metadata in batch (tile payload span, entropy reader setup,
// segmentation/loop-filter cache, CDF retention flag).
func InitDecoderFrameWorkJobDecodeState(batch DecoderFrameWorkBatch, index int, state *TileDecodeState) error {
	return batch.JobDecodeState(index, state)
}

// RetainDecoderFrameWorkTileResidualCDFStorage copies the adapted CDF
// state from state into storage when index identifies the
// context-update tile for the frame. Other jobs leave storage
// unchanged.
func RetainDecoderFrameWorkTileResidualCDFStorage(batch DecoderFrameWorkBatch, index int, state *TileDecodeState, storage *DecoderFrameWorkTileResidualCDFStorage) error {
	return batch.RetainTileResidualCDFStorage(index, state, storage)
}

// InitDecoderFrameWorkTileRestorationRequestReferences populates the
// reference-frame pointers inside req so loop-restoration helpers can
// dispatch through them without re-resolving the references.
func InitDecoderFrameWorkTileRestorationRequestReferences(req *DecoderFrameWorkTileRestorationRequest) error {
	return req.InitReferences()
}

// DecoderFrameWorkJobBlockLoopContextRootColumns reports the number of
// root-column above-context entries one job's block-loop carrier needs
// to track for its tile.
func DecoderFrameWorkJobBlockLoopContextRootColumns(batch DecoderFrameWorkBatch, index int) (int, error) {
	return batch.JobBlockLoopContextRootColumns(index)
}

// BindTileBlockLoopContextCarrier wraps a caller-owned above-context
// slice (length >= rootColumns) into a TileBlockLoopContextCarrier
// ready for use with TileBlockLoopRequest. The first rootColumns
// entries are zeroed before return.
func BindTileBlockLoopContextCarrier(rootColumns int, above []TileBlockLoopRootAboveContext) (TileBlockLoopContextCarrier, error) {
	if rootColumns < 0 {
		return TileBlockLoopContextCarrier{}, ErrTileInvalidDecodeState
	}
	if len(above) < rootColumns {
		return TileBlockLoopContextCarrier{}, ErrFrameShortBuffer
	}
	clear(above[:rootColumns])
	return TileBlockLoopContextCarrier{
		Above: above[:rootColumns],
	}, nil
}

// DecoderFrameWorkJobBlockLoopRequest builds a TileBlockLoopRequest
// for one job in batch, wiring the caller-owned segmentation maps,
// segment-map stride, and block-loop context carrier into the
// returned request.
func DecoderFrameWorkJobBlockLoopRequest(batch DecoderFrameWorkBatch, index int, currentSegmentMap []uint8, previousSegmentMap []uint8, segmentMapStride uint16, carrier *TileBlockLoopContextCarrier) (TileBlockLoopRequest, error) {
	req, err := batch.JobBlockLoopRequest(index, currentSegmentMap, previousSegmentMap, segmentMapStride)
	if err != nil {
		return TileBlockLoopRequest{}, err
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

// ReadDecoderFrameWorkIntraBlockTransforms reads the per-block
// transform metadata for an intra-coded block using state's entropy
// reader.
func ReadDecoderFrameWorkIntraBlockTransforms(batch DecoderFrameWorkBatch, state *TileDecodeState, visit TileBlockLoopVisit) (DecoderFrameWorkBlockTransforms, error) {
	return batch.ReadIntraBlockTransforms(state, visit)
}

// ReadDecoderFrameWorkInterBlockTransforms reads the per-block
// transform metadata for an inter-coded block using state's entropy
// reader.
func ReadDecoderFrameWorkInterBlockTransforms(batch DecoderFrameWorkBatch, state *TileDecodeState, visit TileBlockLoopVisit) (DecoderFrameWorkBlockTransforms, error) {
	return batch.ReadInterBlockTransforms(state, visit)
}

// ReadDecoderFrameWorkBlockTransforms reads the per-block transform
// metadata for the block visited at visit, dispatching to intra or
// inter as appropriate.
func ReadDecoderFrameWorkBlockTransforms(batch DecoderFrameWorkBatch, state *TileDecodeState, visit TileBlockLoopVisit) (DecoderFrameWorkBlockTransforms, error) {
	return batch.ReadBlockTransforms(state, visit)
}

// DecodeAndReconstructDecoderFrameWorkJobResiduals decodes and
// reconstructs every residual in one tile job. It uses cdfs as the
// per-job CDF view and scratch for residual int32/int16 scratch.
//
// See also: DecodeAndRetainDecoderFrameWorkJobResiduals.
func DecodeAndReconstructDecoderFrameWorkJobResiduals(batch DecoderFrameWorkBatch, index int, state *TileDecodeState, cdfs DecoderFrameWorkTileResidualCDFs, scratch *DecoderFrameWorkTileResidualScratch, req DecoderFrameWorkTileResidualRequest) (DecoderFrameWorkTileResidualStats, error) {
	return batch.DecodeAndReconstructJobResiduals(index, state, cdfs, scratch, req)
}

// DecodeAndRetainDecoderFrameWorkJobResiduals is
// DecodeAndReconstructDecoderFrameWorkJobResiduals with automatic
// per-job decode-state setup and CDF retention into storage when this
// job's tile is the context-update tile.
func DecodeAndRetainDecoderFrameWorkJobResiduals(batch DecoderFrameWorkBatch, index int, state *TileDecodeState, storage *DecoderFrameWorkTileResidualCDFStorage, scratch *DecoderFrameWorkTileResidualScratch, req DecoderFrameWorkTileResidualRequest) (DecoderFrameWorkTileResidualStats, error) {
	if storage == nil {
		return DecoderFrameWorkTileResidualStats{}, ErrThreadingInvalidBatch
	}
	if err := InitDecoderFrameWorkJobDecodeState(batch, index, state); err != nil {
		return DecoderFrameWorkTileResidualStats{}, err
	}
	stats, err := DecodeAndReconstructDecoderFrameWorkJobResiduals(batch, index, state, DecoderFrameWorkTileResidualCDFsFromStorage(storage), scratch, req)
	if err != nil {
		return stats, err
	}
	if err := RetainDecoderFrameWorkTileResidualCDFStorage(batch, index, state, storage); err != nil {
		return stats, err
	}
	return stats, nil
}

// DecoderFrameWorkBatchResidualRequest carries caller-owned state for decoding
// every job assigned to one frame-work worker batch.
type DecoderFrameWorkBatchResidualRequest struct {
	Tile DecoderFrameWorkTileResidualRequest

	CurrentSegmentMap  []uint8
	PreviousSegmentMap []uint8
	SegmentMapStride   uint16

	// LoopContextAbove optionally backs one job's root-column context at a time.
	// When nil, the helper leaves Tile.Loop.ContextCarrier unchanged.
	LoopContextAbove []TileBlockLoopRootAboveContext
}

// DecoderFrameWorkBatchResidualScratchSize reports the caller-owned scratch
// needed to decode all jobs in one frame-work worker batch.
type DecoderFrameWorkBatchResidualScratchSize struct {
	Int32Scratch     int
	ResidualScratch  int
	LoopContextAbove int
}

// DecoderFrameWorkBatchResidualScratch carries typed caller-owned arenas for
// BindDecoderFrameWorkBatchResidualRequestScratch.
type DecoderFrameWorkBatchResidualScratch struct {
	Int32Scratch     []int32
	ResidualScratch  []int16
	LoopContextAbove []TileBlockLoopRootAboveContext
}

// Max returns per-arena maximum lengths for reusable residual batch scratch.
func (s DecoderFrameWorkBatchResidualScratchSize) Max(other DecoderFrameWorkBatchResidualScratchSize) DecoderFrameWorkBatchResidualScratchSize {
	return DecoderFrameWorkBatchResidualScratchSize{
		Int32Scratch:     max(s.Int32Scratch, other.Int32Scratch),
		ResidualScratch:  max(s.ResidualScratch, other.ResidualScratch),
		LoopContextAbove: max(s.LoopContextAbove, other.LoopContextAbove),
	}
}

// DecoderFrameWorkBatchResidualRunnerScratchSize reports the caller-owned
// per-worker state and flat scratch needed by DecoderFrameWorkBatchResidualRunner.
type DecoderFrameWorkBatchResidualRunnerScratchSize struct {
	Workers int
	Batch   DecoderFrameWorkBatchResidualScratchSize

	RestorationRequests int
	Int32Scratch        int
	ResidualScratch     int
	LoopContextAbove    int
}

// DecoderFrameWorkBatchResidualRunnerScratch carries caller-owned storage for
// BindDecoderFrameWorkBatchResidualRunner.
type DecoderFrameWorkBatchResidualRunnerScratch struct {
	States                  []TileDecodeState
	Storages                []DecoderFrameWorkTileResidualCDFStorage
	TileScratch             []DecoderFrameWorkTileResidualScratch
	RestorationRequests     []DecoderFrameWorkTileRestorationRequest
	PredictionScratch       []DecoderFrameWorkPredictionScratch
	InterPredictionScratch  []DecoderFrameWorkInterPredictionScratch
	Stats                   []DecoderFrameWorkTileResidualStats
	Int32Scratch            []int32
	ResidualScratch         []int16
	LoopContextAboveScratch []TileBlockLoopRootAboveContext
}

// Max returns per-arena maximum lengths for reusable residual runner scratch.
func (s DecoderFrameWorkBatchResidualRunnerScratchSize) Max(other DecoderFrameWorkBatchResidualRunnerScratchSize) DecoderFrameWorkBatchResidualRunnerScratchSize {
	return DecoderFrameWorkBatchResidualRunnerScratchSize{
		Workers:             max(s.Workers, other.Workers),
		Batch:               s.Batch.Max(other.Batch),
		RestorationRequests: max(s.RestorationRequests, other.RestorationRequests),
		Int32Scratch:        max(s.Int32Scratch, other.Int32Scratch),
		ResidualScratch:     max(s.ResidualScratch, other.ResidualScratch),
		LoopContextAbove:    max(s.LoopContextAbove, other.LoopContextAbove),
	}
}

// DecoderFrameWorkBatchResidualRunner adapts the public residual batch decoder
// to DecoderFrameWorkBatchFunc with caller-owned per-worker state.
type DecoderFrameWorkBatchResidualRunner struct {
	Request DecoderFrameWorkBatchResidualRequest
	// UseDefaultPrediction wires each worker's PredictionScratch into Request
	// when Request.Tile.PredictionScratch is nil. Bound runners default this to
	// true so event-level decode paths perform reconstruction prediction without
	// a custom per-block callback.
	UseDefaultPrediction bool

	States                 []TileDecodeState
	Storages               []DecoderFrameWorkTileResidualCDFStorage
	TileScratch            []DecoderFrameWorkTileResidualScratch
	RestorationRequests    []DecoderFrameWorkTileRestorationRequest
	PredictionScratch      []DecoderFrameWorkPredictionScratch
	InterPredictionScratch []DecoderFrameWorkInterPredictionScratch
	Stats                  []DecoderFrameWorkTileResidualStats

	BatchScratchSize DecoderFrameWorkBatchResidualScratchSize
	Int32Scratch     []int32
	ResidualScratch  []int16
	LoopContextAbove []TileBlockLoopRootAboveContext

	CDEFIndexMap              DecoderFrameWorkCDEFIndexMap
	LoopFilterMap             DecoderFrameWorkLoopFilterMap
	sideDataValid             bool
	sideDataRestorationActive bool
}

// DecoderFrameWorkResidualMaxFrameScratchLen reports the worst-case
// per-block residual scratch lengths needed by any plane/segment in
// the current frame. Callers size persistent per-worker scratch from
// the returned values.
func DecoderFrameWorkResidualMaxFrameScratchLen(batch DecoderFrameWorkBatch, currentQIndex uint8) (int32Len int, int16Len int, err error) {
	planes := [3]DecoderFrameWorkPlane{DecoderFrameWorkPlaneY}
	planeCount := 1
	if !batch.Sequence.ColorConfig.MonoChrome {
		planes[1] = DecoderFrameWorkPlaneU
		planes[2] = DecoderFrameWorkPlaneV
		planeCount = 3
	}
	segmentCount := 1
	if batch.Segmentation.Enabled {
		segmentCount = MaxSegments
	}
	for segmentID := 0; segmentID < segmentCount; segmentID++ {
		for plane := 0; plane < planeCount; plane++ {
			next32, next16, err := DecoderFrameWorkResidualMaxScratchLen(batch, currentQIndex, uint8(segmentID), planes[plane])
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

// DecoderFrameWorkBatchResidualScratchLen reports the caller-owned
// scratch lengths required to decode every job in one worker batch
// (per-batch int32, residual, and loop-context-above lengths).
//
// See also: BindDecoderFrameWorkBatchResidualRequestScratch.
func DecoderFrameWorkBatchResidualScratchLen(batch DecoderFrameWorkBatch, currentQIndex uint8) (DecoderFrameWorkBatchResidualScratchSize, error) {
	int32Len, residualLen, err := DecoderFrameWorkResidualMaxFrameScratchLen(batch, currentQIndex)
	if err != nil {
		return DecoderFrameWorkBatchResidualScratchSize{}, err
	}
	loopContextLen, err := DecoderFrameWorkBatchResidualLoopContextAboveLen(batch)
	if err != nil {
		return DecoderFrameWorkBatchResidualScratchSize{}, err
	}
	return DecoderFrameWorkBatchResidualScratchSize{
		Int32Scratch:     int32Len,
		ResidualScratch:  residualLen,
		LoopContextAbove: loopContextLen,
	}, nil
}

// DecoderFrameWorkBatchResidualRunnerScratchLen reports the
// caller-owned per-worker state and flat scratch lengths required by
// DecoderFrameWorkBatchResidualRunner across workers workers.
//
// See also: BindDecoderFrameWorkBatchResidualRunner.
func DecoderFrameWorkBatchResidualRunnerScratchLen(batch DecoderFrameWorkBatch, workers int) (DecoderFrameWorkBatchResidualRunnerScratchSize, error) {
	if workers <= 0 {
		return DecoderFrameWorkBatchResidualRunnerScratchSize{}, ErrThreadingInvalidWorkerCount
	}
	batchSize, err := DecoderFrameWorkBatchResidualScratchLen(batch, batch.Quantization.BaseQIdx)
	if err != nil {
		return DecoderFrameWorkBatchResidualRunnerScratchSize{}, err
	}
	int32Len, ok := decoderFrameWorkResidualCheckedMul(batchSize.Int32Scratch, workers)
	if !ok {
		return DecoderFrameWorkBatchResidualRunnerScratchSize{}, ErrFrameShortBuffer
	}
	residualLen, ok := decoderFrameWorkResidualCheckedMul(batchSize.ResidualScratch, workers)
	if !ok {
		return DecoderFrameWorkBatchResidualRunnerScratchSize{}, ErrFrameShortBuffer
	}
	loopContextLen, ok := decoderFrameWorkResidualCheckedMul(batchSize.LoopContextAbove, workers)
	if !ok {
		return DecoderFrameWorkBatchResidualRunnerScratchSize{}, ErrFrameShortBuffer
	}
	return DecoderFrameWorkBatchResidualRunnerScratchSize{
		Workers:             workers,
		Batch:               batchSize,
		RestorationRequests: workers,
		Int32Scratch:        int32Len,
		ResidualScratch:     residualLen,
		LoopContextAbove:    loopContextLen,
	}, nil
}

// BindDecoderFrameWorkBatchResidualRequestScratch slices caller-owned
// scratch into a DecoderFrameWorkBatchResidualRequest sized for size.
// Callers fill the remaining request fields (segmentation maps,
// stride, tile loop overrides) before invoking the residual decoder.
func BindDecoderFrameWorkBatchResidualRequestScratch(size DecoderFrameWorkBatchResidualScratchSize, scratch DecoderFrameWorkBatchResidualScratch) (DecoderFrameWorkBatchResidualRequest, error) {
	if decoderFrameWorkResidualScratchTooShort(scratch.Int32Scratch, size.Int32Scratch) ||
		decoderFrameWorkResidualScratchTooShort(scratch.ResidualScratch, size.ResidualScratch) ||
		decoderFrameWorkResidualScratchTooShort(scratch.LoopContextAbove, size.LoopContextAbove) {
		return DecoderFrameWorkBatchResidualRequest{}, ErrFrameShortBuffer
	}
	return DecoderFrameWorkBatchResidualRequest{
		Tile: DecoderFrameWorkTileResidualRequest{
			Int32Scratch:    scratch.Int32Scratch[:size.Int32Scratch],
			ResidualScratch: scratch.ResidualScratch[:size.ResidualScratch],
		},
		LoopContextAbove: scratch.LoopContextAbove[:size.LoopContextAbove],
	}, nil
}

// BindDecoderFrameWorkBatchResidualRunner slices caller-owned
// per-worker state and flat scratch into a
// DecoderFrameWorkBatchResidualRunner. The returned runner is the
// DecoderFrameWorkBatchFunc-compatible executor that decodes every
// job in a batch using one worker's slot.
//
// See also: DecoderFrameWorkBatchResidualRunner.Run.
func BindDecoderFrameWorkBatchResidualRunner(size DecoderFrameWorkBatchResidualRunnerScratchSize, scratch DecoderFrameWorkBatchResidualRunnerScratch) (DecoderFrameWorkBatchResidualRunner, error) {
	if size.Workers <= 0 {
		return DecoderFrameWorkBatchResidualRunner{}, ErrThreadingInvalidWorkerCount
	}
	if err := decoderFrameWorkBatchResidualRunnerScratchErr(size, scratch); err != nil {
		return DecoderFrameWorkBatchResidualRunner{}, err
	}
	var restorationRequests []DecoderFrameWorkTileRestorationRequest
	if len(scratch.RestorationRequests) != 0 {
		restorationRequests = scratch.RestorationRequests[:size.RestorationRequests]
	}
	predictionScratch := scratch.PredictionScratch[:size.Workers]
	interScratch := scratch.InterPredictionScratch[:size.Workers]
	tileScratch := scratch.TileScratch[:size.Workers]
	for i := range predictionScratch {
		predictionScratch[i].Inter = &interScratch[i]
		tileScratch[i].PreallocCallbackScratch()
		internaltile.PreallocBlockLoopContextCarrierScratch(&tileScratch[i].LoopContext, size.Batch.LoopContextAbove)
	}
	return DecoderFrameWorkBatchResidualRunner{
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

func decoderFrameWorkBatchResidualRunnerScratchErr(size DecoderFrameWorkBatchResidualRunnerScratchSize, scratch DecoderFrameWorkBatchResidualRunnerScratch) error {
	if size.Workers == 0 {
		return nil
	}
	if size.Workers < 0 {
		return ErrThreadingInvalidWorkerCount
	}
	if size.RestorationRequests < 0 ||
		size.Int32Scratch < 0 ||
		size.ResidualScratch < 0 ||
		size.LoopContextAbove < 0 ||
		size.Batch.Int32Scratch < 0 ||
		size.Batch.ResidualScratch < 0 ||
		size.Batch.LoopContextAbove < 0 {
		return ErrFrameShortBuffer
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
		return ErrFrameShortBuffer
	}
	if len(scratch.RestorationRequests) != 0 && decoderFrameWorkResidualScratchTooShort(scratch.RestorationRequests, size.RestorationRequests) {
		return ErrFrameShortBuffer
	}
	return nil
}

// Run decodes every residual in batch using the worker slot identified
// by batch.Batch.Worker. It is the DecoderFrameWorkBatchFunc form of
// the runner; pass r.Run to ExecuteDecoderFrameWorkStepWithContext.
func (r *DecoderFrameWorkBatchResidualRunner) Run(batch DecoderFrameWorkBatch) error {
	worker, err := r.workerIndex(batch)
	if err != nil {
		return err
	}
	r.Stats[worker] = DecoderFrameWorkTileResidualStats{}
	req, err := r.workerRequest(worker)
	if err != nil {
		return err
	}
	req.Tile.TransformMode = batch.TransformRef.TransformMode
	if req.Tile.Transforms == nil {
		req.Tile.UseDefaultTransforms = true
	}
	if err := InitDecoderFrameWorkTileResidualCDFStorage(batch, &r.Storages[worker]); err != nil {
		return err
	}
	stats, err := DecodeAndRetainDecoderFrameWorkBatchResiduals(batch, &r.States[worker], &r.Storages[worker], &r.TileScratch[worker], req)
	r.Stats[worker] = stats
	return err
}

// SetSideData attaches bound frame-level side data (CDEF index map,
// loop-filter map, loop-restoration frame buffers) to the runner so
// the residual decode pipeline can populate them as it walks blocks.
// It also resets each side-data structure for a fresh frame.
func (r *DecoderFrameWorkBatchResidualRunner) SetSideData(side DecoderFrameWorkSideData) error {
	if r == nil {
		return ErrThreadingInvalidBatch
	}
	if side.RestorationFrameBuffers.Plan.Active && len(r.RestorationRequests) < len(r.States) {
		return ErrFrameShortBuffer
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
	if side.RestorationFrameBuffers.Plan.Active {
		for i := range r.States {
			req := &r.RestorationRequests[i]
			req.Buffers = side.RestorationFrameBuffers
			if err := InitDecoderFrameWorkTileRestorationRequestReferences(req); err != nil {
				return err
			}
		}
	}
	r.CDEFIndexMap = side.CDEFIndexMap
	r.LoopFilterMap = side.LoopFilterMap
	r.sideDataValid = true
	r.sideDataRestorationActive = side.RestorationFrameBuffers.Plan.Active
	return nil
}

// SetDecoderFrameWorkBatchResidualRunnerSideData is the function-style
// alias for (*DecoderFrameWorkBatchResidualRunner).SetSideData provided
// for callers that prefer free functions.
func SetDecoderFrameWorkBatchResidualRunnerSideData(runner *DecoderFrameWorkBatchResidualRunner, side DecoderFrameWorkSideData) error {
	return runner.SetSideData(side)
}

// ResetStats clears the per-worker statistics buckets so the next
// frame's residual decode starts with zeroed counters.
func (r *DecoderFrameWorkBatchResidualRunner) ResetStats() error {
	if r == nil {
		return ErrThreadingInvalidBatch
	}
	clear(r.Stats)
	return nil
}

// TotalStats aggregates the per-worker counters into one
// DecoderFrameWorkTileResidualStats value for the current frame.
func (r *DecoderFrameWorkBatchResidualRunner) TotalStats() (DecoderFrameWorkTileResidualStats, error) {
	if r == nil {
		return DecoderFrameWorkTileResidualStats{}, ErrThreadingInvalidBatch
	}
	var total DecoderFrameWorkTileResidualStats
	for _, stats := range r.Stats {
		decoderFrameWorkAccumulateResidualStats(&total, stats)
	}
	return total, nil
}

// DecoderFrameWorkBatchResidualLoopContextAboveLen reports the
// worst-case root-column above-context length across the jobs in
// batch, used to size the per-batch loop-context above scratch.
func DecoderFrameWorkBatchResidualLoopContextAboveLen(batch DecoderFrameWorkBatch) (int, error) {
	maxRootColumns := 0
	for i := range batch.Jobs {
		rootColumns, err := DecoderFrameWorkJobBlockLoopContextRootColumns(batch, i)
		if err != nil {
			return 0, err
		}
		if rootColumns > maxRootColumns {
			maxRootColumns = rootColumns
		}
	}
	return maxRootColumns, nil
}

// DecodeAndRetainDecoderFrameWorkBatchResiduals decodes every job in
// batch in order, accumulating per-tile statistics and retaining the
// adapted CDF state for the context-update tile. It is the
// reusable, lower-level building block underneath
// DecoderFrameWorkBatchResidualRunner.Run.
func DecodeAndRetainDecoderFrameWorkBatchResiduals(batch DecoderFrameWorkBatch, state *TileDecodeState, storage *DecoderFrameWorkTileResidualCDFStorage, scratch *DecoderFrameWorkTileResidualScratch, req DecoderFrameWorkBatchResidualRequest) (DecoderFrameWorkTileResidualStats, error) {
	if state == nil || storage == nil || scratch == nil {
		return DecoderFrameWorkTileResidualStats{}, ErrThreadingInvalidBatch
	}
	var total DecoderFrameWorkTileResidualStats
	for i := range batch.Jobs {
		// Re-initialize the entropy (CDF) context from the frame context before
		// each tile in the batch. AV1 decodes every tile from the same starting
		// context (libaom: tile_data->tctx = *cm->fc inside the per-tile loop of
		// decode_tiles), so a tile must never inherit the adapted CDFs of an
		// earlier tile that shared this worker batch. Without this reset, the
		// first tile decodes correctly but every subsequent tile in the same
		// batch starts from desynced CDFs and produces wrong pixels. The per-job
		// retain that follows still captures the context_update_tile_id tile's
		// adapted CDFs (RetainDecoderFrameWorkTileResidualCDFStorage copies them
		// out before the next iteration re-initializes the shared storage).
		if i > 0 {
			if err := InitDecoderFrameWorkTileResidualCDFStorage(batch, storage); err != nil {
				return total, err
			}
		}
		tileReq := req.Tile
		loopReq, err := DecoderFrameWorkJobBlockLoopRequest(batch, i, req.CurrentSegmentMap, req.PreviousSegmentMap, req.SegmentMapStride, req.Tile.Loop.ContextCarrier)
		if err != nil {
			return total, err
		}
		if req.LoopContextAbove != nil {
			rootColumns, err := DecoderFrameWorkJobBlockLoopContextRootColumns(batch, i)
			if err != nil {
				return total, err
			}
			carrier, err := BindTileBlockLoopContextCarrier(rootColumns, req.LoopContextAbove)
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
		stats, err := DecodeAndRetainDecoderFrameWorkJobResiduals(batch, i, state, storage, scratch, tileReq)
		decoderFrameWorkAccumulateResidualStats(&total, stats)
		if err != nil {
			return total, err
		}
	}
	return total, nil
}

func (r *DecoderFrameWorkBatchResidualRunner) workerIndex(batch DecoderFrameWorkBatch) (int, error) {
	if r == nil {
		return 0, ErrThreadingInvalidBatch
	}
	worker := int(batch.Batch.Worker)
	if worker < 0 || worker >= len(r.States) ||
		worker >= len(r.Storages) ||
		worker >= len(r.TileScratch) ||
		worker >= len(r.PredictionScratch) ||
		worker >= len(r.InterPredictionScratch) ||
		worker >= len(r.Stats) {
		return 0, ErrThreadingInvalidBatch
	}
	return worker, nil
}

func (r *DecoderFrameWorkBatchResidualRunner) workerRequest(worker int) (DecoderFrameWorkBatchResidualRequest, error) {
	req := r.Request
	int32Offset, ok := decoderFrameWorkResidualCheckedMul(worker, r.BatchScratchSize.Int32Scratch)
	if !ok {
		return DecoderFrameWorkBatchResidualRequest{}, ErrFrameShortBuffer
	}
	residualOffset, ok := decoderFrameWorkResidualCheckedMul(worker, r.BatchScratchSize.ResidualScratch)
	if !ok {
		return DecoderFrameWorkBatchResidualRequest{}, ErrFrameShortBuffer
	}
	loopOffset, ok := decoderFrameWorkResidualCheckedMul(worker, r.BatchScratchSize.LoopContextAbove)
	if !ok {
		return DecoderFrameWorkBatchResidualRequest{}, ErrFrameShortBuffer
	}
	if int32Offset > len(r.Int32Scratch) ||
		residualOffset > len(r.ResidualScratch) ||
		loopOffset > len(r.LoopContextAbove) {
		return DecoderFrameWorkBatchResidualRequest{}, ErrFrameShortBuffer
	}
	scratch := DecoderFrameWorkBatchResidualScratch{
		Int32Scratch:     r.Int32Scratch[int32Offset:],
		ResidualScratch:  r.ResidualScratch[residualOffset:],
		LoopContextAbove: r.LoopContextAbove[loopOffset:],
	}
	bound, err := BindDecoderFrameWorkBatchResidualRequestScratch(r.BatchScratchSize, scratch)
	if err != nil {
		return DecoderFrameWorkBatchResidualRequest{}, err
	}
	req.Tile.Int32Scratch = bound.Tile.Int32Scratch
	req.Tile.ResidualScratch = bound.Tile.ResidualScratch
	req.LoopContextAbove = bound.LoopContextAbove
	if r.sideDataValid {
		req.Tile.CDEFIndexMap = &r.CDEFIndexMap
		req.Tile.LoopFilterMap = &r.LoopFilterMap
		if r.sideDataRestorationActive {
			if worker >= len(r.RestorationRequests) {
				return DecoderFrameWorkBatchResidualRequest{}, ErrFrameShortBuffer
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

func decoderFrameWorkApplyBatchResidualLoopOverrides(dst *TileBlockLoopRequest, overrides TileBlockLoopRequest) {
	if overrides.ContextCarrier != nil {
		dst.ContextCarrier = overrides.ContextCarrier
	}
	dst.BeforeSuperblock = overrides.BeforeSuperblock
	dst.BeforeCoefficients = overrides.BeforeCoefficients
	dst.CoeffVisitor = overrides.CoeffVisitor
}

func decoderFrameWorkAccumulateResidualStats(total *DecoderFrameWorkTileResidualStats, next DecoderFrameWorkTileResidualStats) {
	total.Loop.PartitionReads += next.Loop.PartitionReads
	total.Loop.Blocks += next.Loop.Blocks
	total.Loop.SegmentPredictions += next.Loop.SegmentPredictions
	total.Loop.SegmentIDs += next.Loop.SegmentIDs
	total.Loop.Prefixes += next.Loop.Prefixes
	total.Loop.PredictionModes += next.Loop.PredictionModes
	total.Loop.IntraModes += next.Loop.IntraModes
	total.Loop.InterEntries += next.Loop.InterEntries
	total.Loop.InterReferences += next.Loop.InterReferences
	total.Loop.InterModes += next.Loop.InterModes
	total.Loop.RefMVStacks += next.Loop.RefMVStacks
	total.Loop.DRLIndices += next.Loop.DRLIndices
	total.Loop.InterMVReferences += next.Loop.InterMVReferences
	total.Loop.MotionVectors += next.Loop.MotionVectors
	total.Loop.MVResiduals += next.Loop.MVResiduals
	total.Loop.InterpFilters += next.Loop.InterpFilters
	total.Loop.InterIntras += next.Loop.InterIntras
	total.Loop.MotionModes += next.Loop.MotionModes
	total.Loop.CompoundBlends += next.Loop.CompoundBlends
	total.Loop.CoefficientBlocks += next.Loop.CoefficientBlocks
	total.Loop.CoefficientTXBs += next.Loop.CoefficientTXBs
	total.Loop.CoefficientNonZero += next.Loop.CoefficientNonZero
	total.Loop.CoefficientAllZero += next.Loop.CoefficientAllZero
	total.Loop.CoefficientEOBTotal += next.Loop.CoefficientEOBTotal
	total.Loop.DeltaReads += next.Loop.DeltaReads
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
