package goav1

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
