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
