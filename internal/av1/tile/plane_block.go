package tile

import "github.com/thesyncim/goav1/internal/av1/parser"

var planeBlockSizeLookup = [blockSizeCount][2][2]BlockSize{
	BlockSize128x128: {{BlockSize128x128, BlockSize128x64}, {BlockSize64x128, BlockSize64x64}},
	BlockSize128x64:  {{BlockSize128x64, blockSizeCount}, {BlockSize64x64, BlockSize64x32}},
	BlockSize64x128:  {{BlockSize64x128, BlockSize64x64}, {blockSizeCount, BlockSize32x64}},
	BlockSize64x64:   {{BlockSize64x64, BlockSize64x32}, {BlockSize32x64, BlockSize32x32}},
	BlockSize64x32:   {{BlockSize64x32, blockSizeCount}, {BlockSize32x32, BlockSize32x16}},
	BlockSize64x16:   {{BlockSize64x16, blockSizeCount}, {BlockSize32x16, BlockSize32x8}},
	BlockSize32x64:   {{BlockSize32x64, BlockSize32x32}, {blockSizeCount, BlockSize16x32}},
	BlockSize32x32:   {{BlockSize32x32, BlockSize32x16}, {BlockSize16x32, BlockSize16x16}},
	BlockSize32x16:   {{BlockSize32x16, blockSizeCount}, {BlockSize16x16, BlockSize16x8}},
	BlockSize32x8:    {{BlockSize32x8, blockSizeCount}, {BlockSize16x8, BlockSize16x4}},
	BlockSize16x64:   {{BlockSize16x64, BlockSize16x32}, {blockSizeCount, BlockSize8x32}},
	BlockSize16x32:   {{BlockSize16x32, BlockSize16x16}, {blockSizeCount, BlockSize8x16}},
	BlockSize16x16:   {{BlockSize16x16, BlockSize16x8}, {BlockSize8x16, BlockSize8x8}},
	BlockSize16x8:    {{BlockSize16x8, blockSizeCount}, {BlockSize8x8, BlockSize8x4}},
	BlockSize16x4:    {{BlockSize16x4, blockSizeCount}, {BlockSize8x4, BlockSize8x4}},
	BlockSize8x32:    {{BlockSize8x32, BlockSize8x16}, {blockSizeCount, BlockSize4x16}},
	BlockSize8x16:    {{BlockSize8x16, BlockSize8x8}, {blockSizeCount, BlockSize4x8}},
	BlockSize8x8:     {{BlockSize8x8, BlockSize8x4}, {BlockSize4x8, BlockSize4x4}},
	BlockSize8x4:     {{BlockSize8x4, blockSizeCount}, {BlockSize4x4, BlockSize4x4}},
	BlockSize4x16:    {{BlockSize4x16, BlockSize4x8}, {blockSizeCount, BlockSize4x8}},
	BlockSize4x8:     {{BlockSize4x8, BlockSize4x4}, {blockSizeCount, BlockSize4x4}},
	BlockSize4x4:     {{BlockSize4x4, BlockSize4x4}, {BlockSize4x4, BlockSize4x4}},
}

func PlaneBlockSize(block BlockSize, color parser.ColorConfig, plane int) (BlockSize, error) {
	if block >= blockSizeCount || plane < 0 || plane > 2 {
		return 0, ErrInvalidDecodeState
	}
	if plane == 0 {
		return block, nil
	}
	if color.MonoChrome {
		return 0, ErrInvalidDecodeState
	}
	ssX := int(boolToShift(color.SubsamplingX))
	ssY := int(boolToShift(color.SubsamplingY))
	out := planeBlockSizeLookup[block][ssX][ssY]
	if out >= blockSizeCount {
		return 0, ErrInvalidDecodeState
	}
	return out, nil
}

func HasChromaBlock(req TransformTreeRequest, color parser.ColorConfig) bool {
	if color.MonoChrome {
		return false
	}
	dims, ok := req.Size.Dimensions()
	if !ok {
		return false
	}
	ssX := int(boolToShift(color.SubsamplingX))
	ssY := int(boolToShift(color.SubsamplingY))
	return (int(dims.W4) > ssX || req.X4&1 != 0) &&
		(int(dims.H4) > ssY || req.Y4&1 != 0)
}
