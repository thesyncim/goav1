package goav1

func PredictDecoderFrameWorkBlock(batch DecoderFrameWorkBatch, index int, visit TileBlockLoopVisit, scratch *DecoderFrameWorkPredictionScratch) error {
	return batch.PredictBlock(index, visit, scratch)
}

func PredictDecoderFrameWorkBlockLuma(batch DecoderFrameWorkBatch, index int, visit TileBlockLoopVisit, scratch *DecoderFrameWorkPredictionScratch) error {
	return batch.PredictBlockLuma(index, visit, scratch)
}

func PredictDecoderFrameWorkBlockInter(batch DecoderFrameWorkBatch, index int, visit TileBlockLoopVisit, scratch *DecoderFrameWorkInterPredictionScratch) error {
	return batch.PredictBlockInter(index, visit, scratch)
}

func PredictDecoderFrameWorkBlockInterWithFilters(batch DecoderFrameWorkBatch, index int, visit TileBlockLoopVisit, scratch *DecoderFrameWorkInterPredictionScratch, filters MotionInterpFilters) error {
	return batch.PredictBlockInterWithFilters(index, visit, scratch, filters)
}

func PredictDecoderFrameWorkBlockIntra(batch DecoderFrameWorkBatch, index int, visit TileBlockLoopVisit, scratch *DecoderFrameWorkIntraPredictionScratch) error {
	return batch.PredictBlockIntra(index, visit, scratch)
}

func PredictDecoderFrameWorkBlockChromaIntra(batch DecoderFrameWorkBatch, index int, visit TileBlockLoopVisit, scratch *DecoderFrameWorkIntraPredictionScratch) error {
	return batch.PredictBlockChromaIntra(index, visit, scratch)
}

func PredictDecoderFrameWorkBlockChromaCFL(batch DecoderFrameWorkBatch, index int, visit TileBlockLoopVisit, scratch *DecoderFrameWorkCFLPredictionScratch) error {
	return batch.PredictBlockChromaCFL(index, visit, scratch)
}

func PredictDecoderFrameWorkBlockIntraCoeff(batch DecoderFrameWorkBatch, index int, visit TileBlockLoopVisit, block TileBlockCoeffBlock, scratch *DecoderFrameWorkIntraPredictionScratch) error {
	return batch.PredictBlockIntraCoeff(index, visit, block, scratch)
}

func PredictDecoderFrameWorkBlockLumaIntra(batch DecoderFrameWorkBatch, index int, visit TileBlockLoopVisit, scratch *DecoderFrameWorkIntraPredictionScratch) error {
	return batch.PredictBlockLumaIntra(index, visit, scratch)
}

func PredictDecoderFrameWorkBlockLumaInter(batch DecoderFrameWorkBatch, index int, visit TileBlockLoopVisit) error {
	return batch.PredictBlockLumaInter(index, visit)
}

func PredictDecoderFrameWorkBlockLumaInterWithFilters(batch DecoderFrameWorkBatch, index int, visit TileBlockLoopVisit, filters MotionInterpFilters) error {
	return batch.PredictBlockLumaInterWithFilters(index, visit, filters)
}

func PredictDecoderFrameWorkBlockLumaInterOBMC(batch DecoderFrameWorkBatch, index int, visit TileBlockLoopVisit, scratch *DecoderFrameWorkInterPredictionScratch) error {
	return batch.PredictBlockLumaInterOBMC(index, visit, scratch)
}

func PredictDecoderFrameWorkBlockLumaInterOBMCWithFilters(batch DecoderFrameWorkBatch, index int, visit TileBlockLoopVisit, scratch *DecoderFrameWorkInterPredictionScratch, filters MotionInterpFilters) error {
	return batch.PredictBlockLumaInterOBMCWithFilters(index, visit, scratch, filters)
}

func PredictDecoderFrameWorkBlockLumaInterCompoundAverage(batch DecoderFrameWorkBatch, index int, visit TileBlockLoopVisit, scratch *DecoderFrameWorkInterPredictionScratch) error {
	return batch.PredictBlockLumaInterCompoundAverage(index, visit, scratch)
}

func PredictDecoderFrameWorkBlockLumaInterCompoundAverageWithFilters(batch DecoderFrameWorkBatch, index int, visit TileBlockLoopVisit, scratch *DecoderFrameWorkInterPredictionScratch, filters MotionInterpFilters) error {
	return batch.PredictBlockLumaInterCompoundAverageWithFilters(index, visit, scratch, filters)
}

func PredictDecoderFrameWorkBlockLumaInterCompound(batch DecoderFrameWorkBatch, index int, visit TileBlockLoopVisit, scratch *DecoderFrameWorkInterPredictionScratch) error {
	return batch.PredictBlockLumaInterCompound(index, visit, scratch)
}

func PredictDecoderFrameWorkBlockLumaInterCompoundWithFilters(batch DecoderFrameWorkBatch, index int, visit TileBlockLoopVisit, scratch *DecoderFrameWorkInterPredictionScratch, filters MotionInterpFilters) error {
	return batch.PredictBlockLumaInterCompoundWithFilters(index, visit, scratch, filters)
}
