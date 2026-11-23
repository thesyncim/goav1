package goav1

import (
	internaldecoder "github.com/thesyncim/goav1/internal/av1/decoder"
	internalthreading "github.com/thesyncim/goav1/internal/av1/threading"
	internaltile "github.com/thesyncim/goav1/internal/av1/tile"
)

// predictDecoderFrameWorkBlock dispatches to the appropriate intra or inter
// prediction path for batch entry index, invoking visit once per generated
// block sample region. scratch carries the caller-owned per-block prediction
// buffers used by the underlying DSP and motion stages.
func predictDecoderFrameWorkBlock(batch internaldecoder.FrameWorkBatch, index int, visit internaltile.BlockLoopVisit, scratch *internalthreading.FrameWorkPredictionScratch) error {
	return batch.PredictBlock(index, visit, scratch)
}

// predictDecoderFrameWorkBlockLuma predicts only the luma plane of the block
// at batch entry index.
func predictDecoderFrameWorkBlockLuma(batch internaldecoder.FrameWorkBatch, index int, visit internaltile.BlockLoopVisit, scratch *internalthreading.FrameWorkPredictionScratch) error {
	return batch.PredictBlockLuma(index, visit, scratch)
}

// predictDecoderFrameWorkBlockChromaCFL runs chroma-from-luma intra
// prediction for the block at batch entry index.
func predictDecoderFrameWorkBlockChromaCFL(batch internaldecoder.FrameWorkBatch, index int, visit internaltile.BlockLoopVisit, scratch *internalthreading.FrameWorkCFLPredictionScratch) error {
	return batch.PredictBlockChromaCFL(index, visit, scratch)
}

// predictDecoderFrameWorkBlockLumaIntra runs intra prediction restricted to
// the luma plane of the block at batch entry index.
func predictDecoderFrameWorkBlockLumaIntra(batch internaldecoder.FrameWorkBatch, index int, visit internaltile.BlockLoopVisit, scratch *internalthreading.FrameWorkIntraPredictionScratch) error {
	return batch.PredictBlockLumaIntra(index, visit, scratch)
}

// predictDecoderFrameWorkBlockLumaInter runs inter prediction restricted to
// the luma plane of the block at batch entry index using the default AV1
// filters.
func predictDecoderFrameWorkBlockLumaInter(batch internaldecoder.FrameWorkBatch, index int, visit internaltile.BlockLoopVisit) error {
	return batch.PredictBlockLumaInter(index, visit)
}
