package goav1

import (
	internaltransform "github.com/thesyncim/goav1/internal/av1/transform"

	internalreconstruct "github.com/thesyncim/goav1/internal/av1/reconstruct"
)

// reconstructBlockMaxScratchLen reports the worst-case int32 and int16 scratch
// lengths required to reconstruct any supported AV1 block, optionally limited
// to the lossless transform path. The bounds are constants, as dav1d's
// fixed-size t->cf and itx scratch: they are computed once, not per frame.
func reconstructBlockMaxScratchLen(lossless bool) (int32Len int, int16Len int, err error) {
	i := 0
	if lossless {
		i = 1
	}
	m := &reconstructMaxScratch[i]
	return m.int32Len, m.int16Len, m.err
}

type reconstructScratchBound struct {
	int32Len, int16Len int
	err                error
}

var reconstructMaxScratch = [2]reconstructScratchBound{
	computeReconstructScratchBound(false),
	computeReconstructScratchBound(true),
}

func computeReconstructScratchBound(lossless bool) reconstructScratchBound {
	int32Len, int16Len, err := scanReconstructBlockMaxScratchLen(lossless)
	return reconstructScratchBound{int32Len, int16Len, err}
}

func scanReconstructBlockMaxScratchLen(lossless bool) (int32Len int, int16Len int, err error) {
	var found bool
	for _, size := range reconstructBlockScratchSizes {
		for _, typ := range reconstructBlockScratchTypes {
			next32, next16, err := internalreconstruct.ScratchLen(internalreconstruct.Block{
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
		return 0, 0, internalreconstruct.ErrInvalidBlock
	}
	return int32Len, int16Len, nil
}

var reconstructBlockScratchSizes = [...]internaltransform.Size{
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

var reconstructBlockScratchTypes = [...]internaltransform.Type{
	internaltransform.TypeDCTDCT,
	internaltransform.TypeADSTDCT,
	internaltransform.TypeDCTADST,
	internaltransform.TypeADSTADST,
	internaltransform.TypeFlipADSTDCT,
	internaltransform.TypeDCTFlipADST,
	internaltransform.TypeFlipADSTFlipADST,
	internaltransform.TypeADSTFlipADST,
	internaltransform.TypeFlipADSTADST,
	internaltransform.TypeIDTX,
	internaltransform.TypeVDCT,
	internaltransform.TypeHDCT,
	internaltransform.TypeVADST,
	internaltransform.TypeHADST,
	internaltransform.TypeVFlipADST,
	internaltransform.TypeHFlipADST,
}
