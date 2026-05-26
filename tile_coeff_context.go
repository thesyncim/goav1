package goav1

import internaltile "github.com/thesyncim/goav1/internal/av1/tile"

// TileCoeffEntropyContext is caller-owned top/left coefficient entropy context
// storage for one superblock. Its zero value is ready for a fresh superblock.
type TileCoeffEntropyContext struct {
	context internaltile.CoeffEntropyContext
}

// Reset returns c to its zero state, ready to track entropy contexts for a
// fresh superblock. It is a no-op when c is nil.
func (c *TileCoeffEntropyContext) Reset() {
	if c != nil {
		*c = TileCoeffEntropyContext{}
	}
}

// TXBContext returns the top/left coefficient entropy context that should
// drive entropy decoding for the transform block described by req.
func (c *TileCoeffEntropyContext) TXBContext(req TileCoeffContextRequest) (TileTXBContext, error) {
	if c == nil {
		return TileTXBContext{}, ErrTileInvalidDecodeState
	}
	return c.context.TXBContext(req)
}

// MarkTXB records the decoded transform block result back into the
// top/left coefficient entropy context so the next block's TXBContext call
// reflects it.
func (c *TileCoeffEntropyContext) MarkTXB(req TileCoeffContextRequest, result TileTXBDecodeResult) error {
	if c == nil {
		return ErrTileInvalidDecodeState
	}
	return c.context.MarkTXB(req, result)
}

// ResetBlock clears the entropy context entries that cover (plane, block) at
// the 4x4 grid coordinates (x4, y4), so an out-of-band reset (e.g. tile
// boundary) does not leak into subsequent blocks.
func (c *TileCoeffEntropyContext) ResetBlock(plane int, block TileBlockSize, x4 int, y4 int) error {
	if c == nil {
		return ErrTileInvalidDecodeState
	}
	return c.context.ResetBlock(plane, block, x4, y4)
}
