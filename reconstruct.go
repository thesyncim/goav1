package goav1

import internalreconstruct "github.com/thesyncim/goav1/internal/av1/reconstruct"

type ReconstructBlock = internalreconstruct.Block

var ErrReconstructInvalidBlock = internalreconstruct.ErrInvalidBlock

func ReconstructBlockScratchLen(cfg ReconstructBlock) (int32Len int, int16Len int, err error) {
	return internalreconstruct.ScratchLen(cfg)
}

func ReconstructPlaneBlock(dst FramePlane, bytesPerSample int, bitDepth uint8, x int, y int, quantized []int16, quantizedStride int, int32Scratch []int32, residualScratch []int16, cfg ReconstructBlock) error {
	return internalreconstruct.ReconstructPlaneBlock(dst, bytesPerSample, bitDepth, x, y, quantized, quantizedStride, int32Scratch, residualScratch, cfg)
}
