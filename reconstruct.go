package goav1

import internalreconstruct "github.com/thesyncim/goav1/internal/av1/reconstruct"

type ReconstructBlock = internalreconstruct.Block

var ErrReconstructInvalidBlock = internalreconstruct.ErrInvalidBlock

func ReconstructBlockScratchLen(cfg ReconstructBlock) (int32Len int, int16Len int, err error) {
	return internalreconstruct.ScratchLen(cfg)
}

func ReconstructBlockMaxScratchLen(lossless bool) (int32Len int, int16Len int, err error) {
	var found bool
	for _, size := range reconstructBlockScratchSizes {
		for _, typ := range reconstructBlockScratchTypes {
			next32, next16, err := ReconstructBlockScratchLen(ReconstructBlock{
				Size:      size,
				Transform: typ,
				Lossless:  lossless,
			})
			if err != nil {
				continue
			}
			found = true
			if next32 > int32Len {
				int32Len = next32
			}
			if next16 > int16Len {
				int16Len = next16
			}
		}
	}
	if !found {
		return 0, 0, ErrReconstructInvalidBlock
	}
	return int32Len, int16Len, nil
}

func ReconstructPlaneBlock(dst FramePlane, bytesPerSample int, bitDepth uint8, x int, y int, quantized []int16, quantizedStride int, int32Scratch []int32, residualScratch []int16, cfg ReconstructBlock) error {
	return internalreconstruct.ReconstructPlaneBlock(dst, bytesPerSample, bitDepth, x, y, quantized, quantizedStride, int32Scratch, residualScratch, cfg)
}

var reconstructBlockScratchSizes = [...]TransformSize{
	{Width: 4, Height: 4},
	{Width: 4, Height: 8},
	{Width: 8, Height: 4},
	{Width: 4, Height: 16},
	{Width: 8, Height: 8},
	{Width: 8, Height: 16},
	{Width: 16, Height: 4},
	{Width: 16, Height: 8},
	{Width: 16, Height: 32},
	{Width: 32, Height: 16},
	{Width: 32, Height: 64},
	{Width: 64, Height: 32},
	{Width: 8, Height: 32},
	{Width: 16, Height: 16},
	{Width: 16, Height: 64},
	{Width: 32, Height: 8},
	{Width: 32, Height: 32},
	{Width: 64, Height: 16},
	{Width: 64, Height: 64},
}

var reconstructBlockScratchTypes = [...]TransformType{
	TransformTypeDCTDCT,
	TransformTypeADSTDCT,
	TransformTypeDCTADST,
	TransformTypeADSTADST,
	TransformTypeFlipADSTDCT,
	TransformTypeDCTFlipADST,
	TransformTypeFlipADSTFlipADST,
	TransformTypeADSTFlipADST,
	TransformTypeFlipADSTADST,
	TransformTypeIDTX,
	TransformTypeVDCT,
	TransformTypeHDCT,
	TransformTypeVADST,
	TransformTypeHADST,
	TransformTypeVFlipADST,
	TransformTypeHFlipADST,
}
