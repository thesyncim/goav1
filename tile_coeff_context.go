package goav1

import internaltile "github.com/thesyncim/goav1/internal/av1/tile"

// TileCoeffEntropyContext is caller-owned top/left coefficient entropy context
// storage for one superblock. Its zero value is ready for a fresh superblock.
type TileCoeffEntropyContext struct {
	context internaltile.CoeffEntropyContext
}

func (c *TileCoeffEntropyContext) Reset() {
	if c != nil {
		*c = TileCoeffEntropyContext{}
	}
}

func (c *TileCoeffEntropyContext) TXBContext(req TileCoeffContextRequest) (TileTXBContext, error) {
	if c == nil {
		return TileTXBContext{}, ErrTileInvalidDecodeState
	}
	return c.context.TXBContext(req)
}

func (c *TileCoeffEntropyContext) MarkTXB(req TileCoeffContextRequest, result TileTXBDecodeResult) error {
	if c == nil {
		return ErrTileInvalidDecodeState
	}
	return c.context.MarkTXB(req, result)
}

func (c *TileCoeffEntropyContext) ResetBlock(plane int, block TileBlockSize, x4 int, y4 int) error {
	if c == nil {
		return ErrTileInvalidDecodeState
	}
	return c.context.ResetBlock(plane, block, x4, y4)
}
