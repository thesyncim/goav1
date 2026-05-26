package goav1

import internaltile "github.com/thesyncim/goav1/internal/av1/tile"

// TileTransformContext is caller-owned top/left transform context storage for
// one superblock. Its zero value is ready for a fresh superblock.
type TileTransformContext struct {
	context internaltile.BlockModeContext
}

// Reset returns c to its zero state, ready to track transform contexts for
// a fresh superblock. It is a no-op when c is nil.
func (c *TileTransformContext) Reset() {
	if c != nil {
		*c = TileTransformContext{}
	}
}

// SelectedTransformContext returns the AV1 selected-transform CDF context for
// a block whose maximum transform size is max at the 4x4 grid coordinates
// (x4, y4) within the superblock.
func (c *TileTransformContext) SelectedTransformContext(max TileTransformSize, x4 int, y4 int) (int, error) {
	if c == nil {
		return 0, ErrTileInvalidDecodeState
	}
	return c.context.SelectedTransformContext(max, x4, y4)
}

// SelectedTransformContextWithAvailability is SelectedTransformContext with
// explicit top/left availability flags, used at tile and frame edges.
func (c *TileTransformContext) SelectedTransformContextWithAvailability(max TileTransformSize, x4 int, y4 int, haveTop bool, haveLeft bool) (int, error) {
	if c == nil {
		return 0, ErrTileInvalidDecodeState
	}
	return c.context.SelectedTransformContextWithAvailability(max, x4, y4, haveTop, haveLeft)
}

// TransformPartitionContext returns the (above, left) transform-partition
// contexts for req used by the AV1 transform partition CDFs.
func (c *TileTransformContext) TransformPartitionContext(req TileTransformPartitionRequest) (int, int, error) {
	if c == nil {
		return 0, 0, ErrTileInvalidDecodeState
	}
	return c.context.TransformPartitionContext(req)
}

// MarkTransform records that a transform of the given size was applied at
// the (x4, y4) grid coordinates of a block of size block, so subsequent
// neighbors can read the correct context.
func (c *TileTransformContext) MarkTransform(block TileBlockSize, x4 int, y4 int, size TileTransformSize, intra bool) error {
	if c == nil {
		return ErrTileInvalidDecodeState
	}
	return c.context.MarkTransform(block, x4, y4, size, intra)
}

// MarkTransformArea fills a (spanW4, spanH4) area of the transform context
// starting at (x4, y4) with the supplied per-axis context log2 values.
func (c *TileTransformContext) MarkTransformArea(x4 int, y4 int, spanW4 int, spanH4 int, ctxLog2W uint8, ctxLog2H uint8, intra bool) error {
	if c == nil {
		return ErrTileInvalidDecodeState
	}
	return c.context.MarkTransformArea(x4, y4, spanW4, spanH4, ctxLog2W, ctxLog2H, intra)
}

// ForEachTileLumaTransformBlock walks the luma transform-block tree implied
// by result, invoking visit for each transform block.
func ForEachTileLumaTransformBlock(result TileTransformTreeResult, req TileTransformTreeRequest, visit TileTransformBlockVisitor) error {
	return result.ForEachLumaTXB(req, visit)
}
