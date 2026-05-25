package goav1

import internaltile "github.com/thesyncim/goav1/internal/av1/tile"

// TileTransformContext is caller-owned top/left transform context storage for
// one superblock. Its zero value is ready for a fresh superblock.
type TileTransformContext struct {
	context internaltile.BlockModeContext
}

func (c *TileTransformContext) Reset() {
	if c != nil {
		*c = TileTransformContext{}
	}
}

func (c *TileTransformContext) SelectedTransformContext(max TileTransformSize, x4 int, y4 int) (int, error) {
	if c == nil {
		return 0, ErrTileInvalidDecodeState
	}
	return c.context.SelectedTransformContext(max, x4, y4)
}

func (c *TileTransformContext) TransformPartitionContext(req TileTransformPartitionRequest) (int, int, error) {
	if c == nil {
		return 0, 0, ErrTileInvalidDecodeState
	}
	return c.context.TransformPartitionContext(req)
}

func (c *TileTransformContext) MarkTransform(block TileBlockSize, x4 int, y4 int, size TileTransformSize, intra bool) error {
	if c == nil {
		return ErrTileInvalidDecodeState
	}
	return c.context.MarkTransform(block, x4, y4, size, intra)
}

func (c *TileTransformContext) MarkTransformArea(x4 int, y4 int, spanW4 int, spanH4 int, ctxLog2W uint8, ctxLog2H uint8, intra bool) error {
	if c == nil {
		return ErrTileInvalidDecodeState
	}
	return c.context.MarkTransformArea(x4, y4, spanW4, spanH4, ctxLog2W, ctxLog2H, intra)
}

func ForEachTileLumaTransformBlock(result TileTransformTreeResult, req TileTransformTreeRequest, visit TileTransformBlockVisitor) error {
	return result.ForEachLumaTXB(req, visit)
}
