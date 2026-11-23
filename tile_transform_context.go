package goav1

import internaltile "github.com/thesyncim/goav1/internal/av1/tile"

// tileTransformContext is caller-owned top/left transform context storage for
// one superblock. Its zero value is ready for a fresh superblock.
type tileTransformContext struct {
	context internaltile.BlockModeContext
}

// Reset returns c to its zero state, ready to track transform contexts for
// a fresh superblock. It is a no-op when c is nil.
func (c *tileTransformContext) Reset() {
	if c != nil {
		*c = tileTransformContext{}
	}
}

// SelectedTransformContext returns the AV1 selected-transform CDF context for
// a block whose maximum transform size is max at the 4x4 grid coordinates
// (x4, y4) within the superblock.
func (c *tileTransformContext) SelectedTransformContext(max internaltile.TransformSize, x4 int, y4 int) (int, error) {
	if c == nil {
		return 0, internaltile.ErrInvalidDecodeState
	}
	return c.context.SelectedTransformContext(max, x4, y4)
}

// SelectedTransformContextWithAvailability is SelectedTransformContext with
// explicit top/left availability flags, used at tile and frame edges.
func (c *tileTransformContext) SelectedTransformContextWithAvailability(max internaltile.TransformSize, x4 int, y4 int, haveTop bool, haveLeft bool) (int, error) {
	if c == nil {
		return 0, internaltile.ErrInvalidDecodeState
	}
	return c.context.SelectedTransformContextWithAvailability(max, x4, y4, haveTop, haveLeft)
}

// TransformPartitionContext returns the (above, left) transform-partition
// contexts for req used by the AV1 transform partition CDFs.
func (c *tileTransformContext) TransformPartitionContext(req internaltile.TransformPartitionRequest) (int, int, error) {
	if c == nil {
		return 0, 0, internaltile.ErrInvalidDecodeState
	}
	return c.context.TransformPartitionContext(req)
}

// MarkTransform records that a transform of the given size was applied at
// the (x4, y4) grid coordinates of a block of size block, so subsequent
// neighbors can read the correct context.
func (c *tileTransformContext) MarkTransform(block internaltile.BlockSize, x4 int, y4 int, size internaltile.TransformSize, intra bool) error {
	if c == nil {
		return internaltile.ErrInvalidDecodeState
	}
	return c.context.MarkTransform(block, x4, y4, size, intra)
}

// MarkTransformArea fills a (spanW4, spanH4) area of the transform context
// starting at (x4, y4) with the supplied per-axis context log2 values.
func (c *tileTransformContext) MarkTransformArea(x4 int, y4 int, spanW4 int, spanH4 int, ctxLog2W uint8, ctxLog2H uint8, intra bool) error {
	if c == nil {
		return internaltile.ErrInvalidDecodeState
	}
	return c.context.MarkTransformArea(x4, y4, spanW4, spanH4, ctxLog2W, ctxLog2H, intra)
}

// forEachTileLumaTransformBlock walks the luma transform-block tree implied
// by result, invoking visit for each transform block.
func forEachTileLumaTransformBlock(result internaltile.TransformTreeResult, req internaltile.TransformTreeRequest, visit internaltile.TransformBlockVisitor) error {
	return result.ForEachLumaTXB(req, visit)
}
