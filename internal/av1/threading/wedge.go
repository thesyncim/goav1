package threading

import "github.com/thesyncim/goav1/internal/av1/tile"

const (
	frameWorkWedgeMasterSize = 64
	frameWorkWedgeMaxAlpha   = 64
)

type frameWorkWedgeDirection uint8

const (
	frameWorkWedgeHorizontal frameWorkWedgeDirection = iota
	frameWorkWedgeVertical
	frameWorkWedgeOblique27
	frameWorkWedgeOblique63
	frameWorkWedgeOblique117
	frameWorkWedgeOblique153
)

type frameWorkWedgeCode struct {
	direction frameWorkWedgeDirection
	xOffset   uint8
	yOffset   uint8
}

var frameWorkWedgeMasterObliqueOdd = [frameWorkWedgeMasterSize]byte{
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 6, 18,
	37, 53, 60, 63, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
	64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
}

var frameWorkWedgeMasterObliqueEven = [frameWorkWedgeMasterSize]byte{
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 11, 27,
	46, 58, 62, 63, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
	64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
}

var frameWorkWedgeMasterVertical = [frameWorkWedgeMasterSize]byte{
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 7, 21,
	43, 57, 62, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
	64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
}

var frameWorkWedgeCodebook16HGTW = [tile.MaxWedgeTypes]frameWorkWedgeCode{
	{frameWorkWedgeOblique27, 4, 4}, {frameWorkWedgeOblique63, 4, 4},
	{frameWorkWedgeOblique117, 4, 4}, {frameWorkWedgeOblique153, 4, 4},
	{frameWorkWedgeHorizontal, 4, 2}, {frameWorkWedgeHorizontal, 4, 4},
	{frameWorkWedgeHorizontal, 4, 6}, {frameWorkWedgeVertical, 4, 4},
	{frameWorkWedgeOblique27, 4, 2}, {frameWorkWedgeOblique27, 4, 6},
	{frameWorkWedgeOblique153, 4, 2}, {frameWorkWedgeOblique153, 4, 6},
	{frameWorkWedgeOblique63, 2, 4}, {frameWorkWedgeOblique63, 6, 4},
	{frameWorkWedgeOblique117, 2, 4}, {frameWorkWedgeOblique117, 6, 4},
}

var frameWorkWedgeCodebook16HLTW = [tile.MaxWedgeTypes]frameWorkWedgeCode{
	{frameWorkWedgeOblique27, 4, 4}, {frameWorkWedgeOblique63, 4, 4},
	{frameWorkWedgeOblique117, 4, 4}, {frameWorkWedgeOblique153, 4, 4},
	{frameWorkWedgeVertical, 2, 4}, {frameWorkWedgeVertical, 4, 4},
	{frameWorkWedgeVertical, 6, 4}, {frameWorkWedgeHorizontal, 4, 4},
	{frameWorkWedgeOblique27, 4, 2}, {frameWorkWedgeOblique27, 4, 6},
	{frameWorkWedgeOblique153, 4, 2}, {frameWorkWedgeOblique153, 4, 6},
	{frameWorkWedgeOblique63, 2, 4}, {frameWorkWedgeOblique63, 6, 4},
	{frameWorkWedgeOblique117, 2, 4}, {frameWorkWedgeOblique117, 6, 4},
}

var frameWorkWedgeCodebook16HEQW = [tile.MaxWedgeTypes]frameWorkWedgeCode{
	{frameWorkWedgeOblique27, 4, 4}, {frameWorkWedgeOblique63, 4, 4},
	{frameWorkWedgeOblique117, 4, 4}, {frameWorkWedgeOblique153, 4, 4},
	{frameWorkWedgeHorizontal, 4, 2}, {frameWorkWedgeHorizontal, 4, 6},
	{frameWorkWedgeVertical, 2, 4}, {frameWorkWedgeVertical, 6, 4},
	{frameWorkWedgeOblique27, 4, 2}, {frameWorkWedgeOblique27, 4, 6},
	{frameWorkWedgeOblique153, 4, 2}, {frameWorkWedgeOblique153, 4, 6},
	{frameWorkWedgeOblique63, 2, 4}, {frameWorkWedgeOblique63, 6, 4},
	{frameWorkWedgeOblique117, 2, 4}, {frameWorkWedgeOblique117, 6, 4},
}

var frameWorkWedgeSignFlipHEQW = [tile.MaxWedgeTypes]byte{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1}
var frameWorkWedgeSignFlipHGTW = [tile.MaxWedgeTypes]byte{1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1}
var frameWorkWedgeSignFlip8x32 = [tile.MaxWedgeTypes]byte{1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1}
var frameWorkWedgeSignFlip32x8 = [tile.MaxWedgeTypes]byte{1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1}

func frameWorkBuildWedgeMask(mask []byte, stride int, size tile.BlockSize, wedgeIndex uint8, wedgeSign bool) error {
	dims, ok := size.Dimensions()
	if !ok || !tile.WedgeUsed(size) || int(wedgeIndex) >= tile.MaxWedgeTypes {
		return ErrInvalidBatch
	}
	width := int(dims.W4) * 4
	height := int(dims.H4) * 4
	if !frameWorkMaskBlockFits(len(mask), stride, width, height) {
		return ErrInvalidBatch
	}
	codebook, signFlip, ok := frameWorkWedgeParams(size)
	if !ok {
		return ErrInvalidBatch
	}

	code := codebook[wedgeIndex]
	neg := int(signFlip[wedgeIndex])
	if wedgeSign {
		neg ^= 1
	}
	woff := (int(code.xOffset) * width) >> 3
	hoff := (int(code.yOffset) * height) >> 3
	row0 := frameWorkWedgeMasterSize/2 - hoff
	col0 := frameWorkWedgeMasterSize/2 - woff
	for row := range height {
		for col := range width {
			sample, ok := frameWorkWedgeMasterSample(neg, code.direction, row0+row, col0+col)
			if !ok {
				return ErrInvalidBatch
			}
			mask[row*stride+col] = sample
		}
	}
	return nil
}

func frameWorkWedgeParams(size tile.BlockSize) (*[tile.MaxWedgeTypes]frameWorkWedgeCode, *[tile.MaxWedgeTypes]byte, bool) {
	switch size {
	case tile.BlockSize8x8, tile.BlockSize16x16, tile.BlockSize32x32:
		return &frameWorkWedgeCodebook16HEQW, &frameWorkWedgeSignFlipHEQW, true
	case tile.BlockSize8x16, tile.BlockSize16x32:
		return &frameWorkWedgeCodebook16HGTW, &frameWorkWedgeSignFlipHGTW, true
	case tile.BlockSize8x32:
		return &frameWorkWedgeCodebook16HGTW, &frameWorkWedgeSignFlip8x32, true
	case tile.BlockSize16x8, tile.BlockSize32x16:
		return &frameWorkWedgeCodebook16HLTW, &frameWorkWedgeSignFlipHGTW, true
	case tile.BlockSize32x8:
		return &frameWorkWedgeCodebook16HLTW, &frameWorkWedgeSignFlip32x8, true
	default:
		return nil, nil, false
	}
}

func frameWorkWedgeMasterSample(neg int, direction frameWorkWedgeDirection, row int, col int) (byte, bool) {
	if neg < 0 || neg > 1 || row < 0 || row >= frameWorkWedgeMasterSize || col < 0 || col >= frameWorkWedgeMasterSize {
		return 0, false
	}
	var sample byte
	switch direction {
	case frameWorkWedgeHorizontal:
		sample = frameWorkWedgeMasterVertical[row]
	case frameWorkWedgeVertical:
		sample = frameWorkWedgeMasterVertical[col]
	case frameWorkWedgeOblique27:
		sample = frameWorkWedgeMasterOblique63(col, row)
	case frameWorkWedgeOblique63:
		sample = frameWorkWedgeMasterOblique63(row, col)
	case frameWorkWedgeOblique117:
		sample = frameWorkWedgeMaxAlpha - frameWorkWedgeMasterOblique63(row, frameWorkWedgeMasterSize-1-col)
	case frameWorkWedgeOblique153:
		sample = frameWorkWedgeMaxAlpha - frameWorkWedgeMasterOblique63(col, frameWorkWedgeMasterSize-1-row)
	default:
		return 0, false
	}
	if neg != 0 {
		sample = frameWorkWedgeMaxAlpha - sample
	}
	return sample, true
}

func frameWorkWedgeMasterOblique63(row int, col int) byte {
	src := frameWorkWedgeMasterObliqueEven[:]
	if row&1 != 0 {
		src = frameWorkWedgeMasterObliqueOdd[:]
	}
	shift := frameWorkWedgeMasterSize/4 - ((row + 1) >> 1)
	if shift >= 0 {
		if col < shift {
			return src[0]
		}
		return src[col-shift]
	}
	shift = -shift
	if col >= frameWorkWedgeMasterSize-shift {
		return src[frameWorkWedgeMasterSize-1]
	}
	return src[col+shift]
}
