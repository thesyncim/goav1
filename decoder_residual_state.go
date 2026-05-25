package goav1

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

func DecoderFrameWorkBlockQuantizer(batch DecoderFrameWorkBatch, currentQIndex uint8, segmentID uint8, plane DecoderFrameWorkPlane) (Quantizer, bool, error) {
	return batch.BlockQuantizer(currentQIndex, segmentID, plane)
}

func DecoderFrameWorkBlockQIndex(batch DecoderFrameWorkBatch, currentQIndex uint8, segmentID uint8) (uint8, bool, error) {
	return batch.BlockQIndex(currentQIndex, segmentID)
}

func DecoderFrameWorkBlockCoeffPlanePosition(batch DecoderFrameWorkBatch, index int, visit TileBlockVisit, block TileBlockCoeffBlock) (DecoderFrameWorkPlane, int, int, error) {
	return batch.BlockCoeffPlanePosition(index, visit, block)
}

func ReconstructDecoderFrameWorkBlockCoeff(batch DecoderFrameWorkBatch, index int, req DecoderFrameWorkBlockCoeffReconstruction) error {
	return batch.ReconstructBlockCoeff(index, req)
}

func InitDecoderFrameWorkTileResidualCDFStorageDefault(storage *DecoderFrameWorkTileResidualCDFStorage, baseQIndex uint8) error {
	return storage.InitDefault(baseQIndex)
}

func DecoderFrameWorkTileResidualCDFsFromStorage(storage *DecoderFrameWorkTileResidualCDFStorage) DecoderFrameWorkTileResidualCDFs {
	return storage.CDFs()
}

func InitDecoderFrameWorkTileResidualCDFStorage(batch DecoderFrameWorkBatch, storage *DecoderFrameWorkTileResidualCDFStorage) error {
	return batch.InitTileResidualCDFStorage(storage)
}

func InitDecoderFrameWorkJobDecodeState(batch DecoderFrameWorkBatch, index int, state *TileDecodeState) error {
	return batch.JobDecodeState(index, state)
}

func RetainDecoderFrameWorkTileResidualCDFStorage(batch DecoderFrameWorkBatch, index int, state *TileDecodeState, storage *DecoderFrameWorkTileResidualCDFStorage) error {
	return batch.RetainTileResidualCDFStorage(index, state, storage)
}

func InitDecoderFrameWorkTileRestorationRequestReferences(req *DecoderFrameWorkTileRestorationRequest) error {
	return req.InitReferences()
}

func DecoderFrameWorkJobBlockLoopContextRootColumns(batch DecoderFrameWorkBatch, index int) (int, error) {
	return batch.JobBlockLoopContextRootColumns(index)
}

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

func DecoderFrameWorkJobBlockLoopRequest(batch DecoderFrameWorkBatch, index int, currentSegmentMap []uint8, previousSegmentMap []uint8, segmentMapStride int, carrier *TileBlockLoopContextCarrier) (TileBlockLoopRequest, error) {
	req, err := batch.JobBlockLoopRequest(index, currentSegmentMap, previousSegmentMap, segmentMapStride)
	if err != nil {
		return TileBlockLoopRequest{}, err
	}
	req.ContextCarrier = carrier
	return req, nil
}

func ReadDecoderFrameWorkIntraBlockTransforms(batch DecoderFrameWorkBatch, state *TileDecodeState, visit TileBlockLoopVisit) (DecoderFrameWorkBlockTransforms, error) {
	return batch.ReadIntraBlockTransforms(state, visit)
}

func ReadDecoderFrameWorkInterBlockTransforms(batch DecoderFrameWorkBatch, state *TileDecodeState, visit TileBlockLoopVisit) (DecoderFrameWorkBlockTransforms, error) {
	return batch.ReadInterBlockTransforms(state, visit)
}

func DecodeAndReconstructDecoderFrameWorkJobResiduals(batch DecoderFrameWorkBatch, index int, state *TileDecodeState, cdfs DecoderFrameWorkTileResidualCDFs, scratch *DecoderFrameWorkTileResidualScratch, req DecoderFrameWorkTileResidualRequest) (DecoderFrameWorkTileResidualStats, error) {
	return batch.DecodeAndReconstructJobResiduals(index, state, cdfs, scratch, req)
}

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
	SegmentMapStride   int

	// LoopContextAbove optionally backs one job's root-column context at a time.
	// When nil, the helper leaves Tile.Loop.ContextCarrier unchanged.
	LoopContextAbove []TileBlockLoopRootAboveContext
}

func DecodeAndRetainDecoderFrameWorkBatchResiduals(batch DecoderFrameWorkBatch, state *TileDecodeState, storage *DecoderFrameWorkTileResidualCDFStorage, scratch *DecoderFrameWorkTileResidualScratch, req DecoderFrameWorkBatchResidualRequest) (DecoderFrameWorkTileResidualStats, error) {
	if state == nil || storage == nil || scratch == nil {
		return DecoderFrameWorkTileResidualStats{}, ErrThreadingInvalidBatch
	}
	var total DecoderFrameWorkTileResidualStats
	for i := range batch.Jobs {
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
