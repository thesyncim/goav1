package threading

import "github.com/thesyncim/goav1/internal/av1/tile"

// FrameWorkLoopFilterOrderedBuilder carries transform-edge contexts while an
// encoder visits one tile in coding order. The generic record map remains
// independent; this builder only replaces the later geometry reconstruction.
type FrameWorkLoopFilterOrderedBuilder struct {
	masks   *FrameWorkLoopFilterMasks
	scratch frameWorkLoopFilterMaskScratch
	sbLog2  int
	cells   int
}

// Reset starts one complete single-tile frame. Storage is reused after the
// first frame; callers must finish the coding-order walk before consuming masks.
func (b *FrameWorkLoopFilterOrderedBuilder) Reset(masks *FrameWorkLoopFilterMasks, sbLog2 int) error {
	if b == nil || masks == nil || !masks.Valid() || masks.Cols <= 0 || masks.Rows <= 0 ||
		(sbLog2 != 4 && sbLog2 != 5) {
		return ErrInvalidBatch
	}
	b.masks = masks
	b.sbLog2 = sbLog2
	b.cells = 0
	clear(masks.Masks)
	b.scratch.reset(masks)
	return nil
}

// MarkBlock commits the final transform geometry of a coded block. Call only
// once per block, after any decision pass, in superblock raster / block Z order.
func (b *FrameWorkLoopFilterOrderedBuilder) MarkBlock(block tile.BlockVisit, tree tile.TransformTreeResult, skip, intra bool) error {
	if b == nil || b.masks == nil {
		return ErrInvalidBatch
	}
	h := b.masks
	dims, ok := block.Size.Dimensions()
	bx, by := int(block.MICol), int(block.MIRow)
	endX, endY := int(block.MIColEnd), int(block.MIRowEnd)
	if !ok || bx < 0 || by < 0 || endX <= bx || endY <= by || endX > h.Cols || endY > h.Rows ||
		endX != min(bx+int(dims.W4), h.Cols) || endY != min(by+int(dims.H4), h.Rows) {
		return ErrInvalidBatch
	}
	h.buildBlock(&b.scratch, bx, by, block.Size, tree, skip, intra, b.sbLog2)
	b.cells += (endX - bx) * (endY - by)
	return nil
}

// Complete reports whether the caller supplied all cells of the bound frame.
func (b *FrameWorkLoopFilterOrderedBuilder) Complete(masks *FrameWorkLoopFilterMasks) bool {
	return b != nil && masks != nil && b.masks == masks && b.cells == masks.Cols*masks.Rows
}
