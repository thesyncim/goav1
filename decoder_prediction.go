package goav1

// PredictDecoderFrameWorkBlock dispatches to the appropriate intra or inter
// prediction path for batch entry index, invoking visit once per generated
// block sample region. scratch carries the caller-owned per-block prediction
// buffers used by the underlying DSP and motion stages.
func PredictDecoderFrameWorkBlock(batch DecoderFrameWorkBatch, index int, visit TileBlockLoopVisit, scratch *DecoderFrameWorkPredictionScratch) error {
	return batch.PredictBlock(index, visit, scratch)
}

// PredictDecoderFrameWorkBlockLuma predicts only the luma plane of the block
// at batch entry index.
func PredictDecoderFrameWorkBlockLuma(batch DecoderFrameWorkBatch, index int, visit TileBlockLoopVisit, scratch *DecoderFrameWorkPredictionScratch) error {
	return batch.PredictBlockLuma(index, visit, scratch)
}

// PredictDecoderFrameWorkBlockInter runs inter prediction for the block at
// batch entry index using the default AV1 interpolation filters.
func PredictDecoderFrameWorkBlockInter(batch DecoderFrameWorkBatch, index int, visit TileBlockLoopVisit, scratch *DecoderFrameWorkInterPredictionScratch) error {
	return batch.PredictBlockInter(index, visit, scratch)
}

// PredictDecoderFrameWorkBlockInterWithFilters runs inter prediction for the
// block at batch entry index using the supplied per-axis interpolation
// filters.
func PredictDecoderFrameWorkBlockInterWithFilters(batch DecoderFrameWorkBatch, index int, visit TileBlockLoopVisit, scratch *DecoderFrameWorkInterPredictionScratch, filters MotionInterpFilters) error {
	return batch.PredictBlockInterWithFilters(index, visit, scratch, filters)
}

// PredictDecoderFrameWorkBlockIntra runs intra prediction for the block at
// batch entry index.
func PredictDecoderFrameWorkBlockIntra(batch DecoderFrameWorkBatch, index int, visit TileBlockLoopVisit, scratch *DecoderFrameWorkIntraPredictionScratch) error {
	return batch.PredictBlockIntra(index, visit, scratch)
}

// PredictDecoderFrameWorkBlockChromaIntra runs chroma-only intra prediction
// for the block at batch entry index.
func PredictDecoderFrameWorkBlockChromaIntra(batch DecoderFrameWorkBatch, index int, visit TileBlockLoopVisit, scratch *DecoderFrameWorkIntraPredictionScratch) error {
	return batch.PredictBlockChromaIntra(index, visit, scratch)
}

// PredictDecoderFrameWorkBlockChromaCFL runs chroma-from-luma intra
// prediction for the block at batch entry index.
func PredictDecoderFrameWorkBlockChromaCFL(batch DecoderFrameWorkBatch, index int, visit TileBlockLoopVisit, scratch *DecoderFrameWorkCFLPredictionScratch) error {
	return batch.PredictBlockChromaCFL(index, visit, scratch)
}

// PredictDecoderFrameWorkBlockIntraCoeff runs intra prediction for the block
// at batch entry index and additionally applies the decoded coefficient block
// to the predicted samples.
func PredictDecoderFrameWorkBlockIntraCoeff(batch DecoderFrameWorkBatch, index int, visit TileBlockLoopVisit, block TileBlockCoeffBlock, scratch *DecoderFrameWorkIntraPredictionScratch) error {
	return batch.PredictBlockIntraCoeff(index, visit, block, scratch)
}

// PredictDecoderFrameWorkBlockLumaIntra runs intra prediction restricted to
// the luma plane of the block at batch entry index.
func PredictDecoderFrameWorkBlockLumaIntra(batch DecoderFrameWorkBatch, index int, visit TileBlockLoopVisit, scratch *DecoderFrameWorkIntraPredictionScratch) error {
	return batch.PredictBlockLumaIntra(index, visit, scratch)
}

// PredictDecoderFrameWorkBlockLumaInter runs inter prediction restricted to
// the luma plane of the block at batch entry index using the default AV1
// filters.
func PredictDecoderFrameWorkBlockLumaInter(batch DecoderFrameWorkBatch, index int, visit TileBlockLoopVisit) error {
	return batch.PredictBlockLumaInter(index, visit)
}

// PredictDecoderFrameWorkBlockLumaInterWithFilters runs luma-only inter
// prediction with the supplied interpolation filters.
func PredictDecoderFrameWorkBlockLumaInterWithFilters(batch DecoderFrameWorkBatch, index int, visit TileBlockLoopVisit, filters MotionInterpFilters) error {
	return batch.PredictBlockLumaInterWithFilters(index, visit, filters)
}

// PredictDecoderFrameWorkBlockLumaInterOBMC runs luma-only inter prediction
// with overlapped block motion compensation (OBMC) applied along the top and
// left neighbors.
func PredictDecoderFrameWorkBlockLumaInterOBMC(batch DecoderFrameWorkBatch, index int, visit TileBlockLoopVisit, scratch *DecoderFrameWorkInterPredictionScratch) error {
	return batch.PredictBlockLumaInterOBMC(index, visit, scratch)
}

// PredictDecoderFrameWorkBlockLumaInterOBMCWithFilters is the
// PredictDecoderFrameWorkBlockLumaInterOBMC variant with explicit
// interpolation filters.
func PredictDecoderFrameWorkBlockLumaInterOBMCWithFilters(batch DecoderFrameWorkBatch, index int, visit TileBlockLoopVisit, scratch *DecoderFrameWorkInterPredictionScratch, filters MotionInterpFilters) error {
	return batch.PredictBlockLumaInterOBMCWithFilters(index, visit, scratch, filters)
}

// PredictDecoderFrameWorkBlockLumaInterCompoundAverage runs compound-average
// inter prediction (mean of two references) for the luma plane.
func PredictDecoderFrameWorkBlockLumaInterCompoundAverage(batch DecoderFrameWorkBatch, index int, visit TileBlockLoopVisit, scratch *DecoderFrameWorkInterPredictionScratch) error {
	return batch.PredictBlockLumaInterCompoundAverage(index, visit, scratch)
}

// PredictDecoderFrameWorkBlockLumaInterCompoundAverageWithFilters is the
// PredictDecoderFrameWorkBlockLumaInterCompoundAverage variant with explicit
// interpolation filters.
func PredictDecoderFrameWorkBlockLumaInterCompoundAverageWithFilters(batch DecoderFrameWorkBatch, index int, visit TileBlockLoopVisit, scratch *DecoderFrameWorkInterPredictionScratch, filters MotionInterpFilters) error {
	return batch.PredictBlockLumaInterCompoundAverageWithFilters(index, visit, scratch, filters)
}

// PredictDecoderFrameWorkBlockLumaInterCompound runs the full AV1 compound
// inter prediction (distance-weighted, wedge, diff-wtd, intra-inter) for the
// luma plane.
func PredictDecoderFrameWorkBlockLumaInterCompound(batch DecoderFrameWorkBatch, index int, visit TileBlockLoopVisit, scratch *DecoderFrameWorkInterPredictionScratch) error {
	return batch.PredictBlockLumaInterCompound(index, visit, scratch)
}

// PredictDecoderFrameWorkBlockLumaInterCompoundWithFilters is the
// PredictDecoderFrameWorkBlockLumaInterCompound variant with explicit
// interpolation filters.
func PredictDecoderFrameWorkBlockLumaInterCompoundWithFilters(batch DecoderFrameWorkBatch, index int, visit TileBlockLoopVisit, scratch *DecoderFrameWorkInterPredictionScratch, filters MotionInterpFilters) error {
	return batch.PredictBlockLumaInterCompoundWithFilters(index, visit, scratch, filters)
}
