package threading

import (
	"sync"
	"sync/atomic"

	"github.com/thesyncim/goav1/internal/av1/tile"
)

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

	src := frameWorkWedgeMaskTable(size, codebook, signFlip, width, height, wedgeIndex, wedgeSign)
	if src == nil {
		return ErrInvalidBatch
	}
	for row := range height {
		copy(mask[row*stride:row*stride+width], src[row*width:row*width+width])
	}
	return nil
}

// frameWorkWedgeTableLen is the summed area of the nine wedge block sizes
// (8x8, 16x16, 32x32, 8x16, 16x32, 8x32, 16x8, 32x16, 32x8).
const frameWorkWedgeTableLen = 64 + 256 + 1024 + 128 + 512 + 256 + 128 + 512 + 256

// frameWorkWedgeMasks holds every wedge mask (block size x index x sign) at
// stride = block width, built once per block size on first use, as dav1d's
// dav1d_init_ii_wedge_masks precomputes its tables instead of rebuilding a
// mask per block.
var frameWorkWedgeMasks struct {
	mu    sync.Mutex
	ready [9]atomic.Bool
	bad   [9]bool
	pix   [frameWorkWedgeTableLen * tile.MaxWedgeTypes * 2]byte
}

// frameWorkWedgeSlot maps a wedge block size to its table slot and the byte
// offset of its masks in frameWorkWedgeMasks.pix.
func frameWorkWedgeSlot(size tile.BlockSize) (slot int, base int, ok bool) {
	const n = tile.MaxWedgeTypes * 2
	switch size {
	case tile.BlockSize8x8:
		return 0, 0, true
	case tile.BlockSize16x16:
		return 1, 64 * n, true
	case tile.BlockSize32x32:
		return 2, (64 + 256) * n, true
	case tile.BlockSize8x16:
		return 3, (64 + 256 + 1024) * n, true
	case tile.BlockSize16x32:
		return 4, (64 + 256 + 1024 + 128) * n, true
	case tile.BlockSize8x32:
		return 5, (64 + 256 + 1024 + 128 + 512) * n, true
	case tile.BlockSize16x8:
		return 6, (64 + 256 + 1024 + 128 + 512 + 256) * n, true
	case tile.BlockSize32x16:
		return 7, (64 + 256 + 1024 + 128 + 512 + 256 + 128) * n, true
	case tile.BlockSize32x8:
		return 8, (64 + 256 + 1024 + 128 + 512 + 256 + 128 + 512) * n, true
	}
	return 0, 0, false
}

// frameWorkWedgeMaskTable returns the width x height mask for (wedgeIndex,
// wedgeSign), building all 32 masks of the block size on first use.
func frameWorkWedgeMaskTable(size tile.BlockSize, codebook *[tile.MaxWedgeTypes]frameWorkWedgeCode, signFlip *[tile.MaxWedgeTypes]byte, width, height int, wedgeIndex uint8, wedgeSign bool) []byte {
	slot, base, ok := frameWorkWedgeSlot(size)
	if !ok {
		return nil
	}
	area := width * height
	all := frameWorkWedgeMasks.pix[base : base+area*tile.MaxWedgeTypes*2]
	if !frameWorkWedgeMasks.ready[slot].Load() {
		frameWorkWedgeMasks.mu.Lock()
		if !frameWorkWedgeMasks.ready[slot].Load() {
			for idx := range tile.MaxWedgeTypes {
				for sign := range 2 {
					m := all[(idx*2+sign)*area : (idx*2+sign+1)*area]
					if !frameWorkFillWedgeMask(m, codebook, signFlip, width, height, uint8(idx), sign != 0) {
						frameWorkWedgeMasks.bad[slot] = true
					}
				}
			}
			frameWorkWedgeMasks.ready[slot].Store(true)
		}
		frameWorkWedgeMasks.mu.Unlock()
	}
	if frameWorkWedgeMasks.bad[slot] {
		return nil
	}
	sign := 0
	if wedgeSign {
		sign = 1
	}
	i := int(wedgeIndex)*2 + sign
	return all[i*area : (i+1)*area]
}

// frameWorkFillWedgeMask computes one wedge mask from the master templates
// (libaom init_wedge_masks) at stride = width.
func frameWorkFillWedgeMask(mask []byte, codebook *[tile.MaxWedgeTypes]frameWorkWedgeCode, signFlip *[tile.MaxWedgeTypes]byte, width, height int, wedgeIndex uint8, wedgeSign bool) bool {
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
				return false
			}
			mask[row*width+col] = sample
		}
	}
	return true
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
