package tile

import (
	"testing"

	"github.com/thesyncim/goav1/internal/av1/motion"
)

// TestCollectSubChromaInterCellsSkipsNonSub4 pins the negative case: blocks
// whose width and height are both >= 8 (under chroma subsampling) never
// trigger libaom's build_inter_predictors_sub8x8 chroma path. The helper
// must return (zero, false) for such blocks so callers fall back to the
// standard single-MV chroma prediction with the anchor block's own MV.
func TestCollectSubChromaInterCellsSkipsNonSub4(t *testing.T) {
	ctx := &BlockModeContext{}
	// BlockSize8x8 with 4:2:0 chroma: chroma block is 4x4, no sub4 trigger
	// since W4>=2 and H4>=2.
	got, ok := ctx.CollectSubChromaInterCells(BlockSize8x8, 4, 4, true, true, motion.RegularFilters)
	if ok || got.Count != 0 {
		t.Fatalf("BlockSize8x8 should not trigger sub8x8, got ok=%v count=%d", ok, got.Count)
	}
	// BlockSize16x16: even less reason to trigger.
	got, ok = ctx.CollectSubChromaInterCells(BlockSize16x16, 4, 4, true, true, motion.RegularFilters)
	if ok || got.Count != 0 {
		t.Fatalf("BlockSize16x16 should not trigger sub8x8, got ok=%v count=%d", ok, got.Count)
	}
}

// TestCollectSubChromaInterCellsSkipsWhenNotSubsampled pins that sub8x8
// chroma is gated on the chroma subsampling flag - a sequence with 4:4:4
// chroma never triggers the path because is_sub4_x/is_sub4_y both require
// ss_x/ss_y respectively to be set.
func TestCollectSubChromaInterCellsSkipsWhenNotSubsampled(t *testing.T) {
	ctx := &BlockModeContext{}
	got, ok := ctx.CollectSubChromaInterCells(BlockSize4x4, 4, 4, false, false, motion.RegularFilters)
	if ok || got.Count != 0 {
		t.Fatalf("4:4:4 should not trigger sub8x8, got ok=%v count=%d", ok, got.Count)
	}
	got, ok = ctx.CollectSubChromaInterCells(BlockSize4x8, 4, 4, false, true, motion.RegularFilters)
	if ok || got.Count != 0 {
		t.Fatalf("ssY-only without ssX for BlockSize4x8 should not trigger sub8x8 (W4=1 needs ssX), got ok=%v count=%d", ok, got.Count)
	}
}

// TestCollectSubChromaInterCellsRequiresAllNeighborsInter pins libaom's
// is_sub8x8_inter() gate: every luma block inside the chroma footprint
// (NW, N, W, self for BlockSize4x4) must be inter. When any neighbor is
// intra (GridMotionValid==0) the helper returns (zero, false) so the
// caller falls back to the anchor's single-MV chroma prediction.
func TestCollectSubChromaInterCellsRequiresAllNeighborsInter(t *testing.T) {
	ctx := &BlockModeContext{}
	// Seed three of the four cells (NW, N, W) as valid inter; leave self
	// empty since the caller is expected to fill that one after
	// MarkInterMotion. Then seed only two of the three neighbors and
	// confirm the helper aborts.
	seedInter := func(x, y int) {
		setGridRecordCellForTest(ctx, x, y, blockModeGridRecord{
			Motion: InterMotionResult{
				References: InterReferencesResult{Ref: [2]ReferenceFrame{ReferenceFrameLast, ReferenceFrameNone}},
				MV:         [2]motion.Vector{{Row: 1, Col: 2}},
			},
			Flags: gridRecordMotionValid,
		})
	}
	seedInter(3, 3) // NW for block at (4, 4)
	seedInter(4, 3) // N
	// W not seeded - missing inter neighbor must cause failure.
	got, ok := ctx.CollectSubChromaInterCells(BlockSize4x4, 4, 4, true, true, motion.RegularFilters)
	if ok || got.Count != 0 {
		t.Fatalf("missing W neighbor must abort, got ok=%v count=%d", ok, got.Count)
	}

	seedInter(3, 4) // W
	got, ok = ctx.CollectSubChromaInterCells(BlockSize4x4, 4, 4, true, true, motion.RegularFilters)
	if !ok {
		t.Fatalf("with all neighbors inter, expected ok=true, got ok=%v count=%d", ok, got.Count)
	}
	if got.Count != 4 {
		t.Fatalf("BlockSize4x4 with full quadrant should emit 4 cells, got %d", got.Count)
	}
}

// TestCollectSubChromaInterCellsBlockSize4x4Quadrant pins the cell layout
// produced for a BlockSize4x4 chroma anchor: NW, N, W, self in libaom's
// row-then-col order, each 2x2 chroma samples covering the four 4x4 luma
// cells that share the chroma anchor's 4x4 footprint.
func TestCollectSubChromaInterCellsBlockSize4x4Quadrant(t *testing.T) {
	ctx := &BlockModeContext{}
	for _, p := range [][2]int{{3, 3}, {4, 3}, {3, 4}} {
		setGridRecordCellForTest(ctx, p[0], p[1], blockModeGridRecord{
			Motion: InterMotionResult{
				References: InterReferencesResult{Ref: [2]ReferenceFrame{ReferenceFrameLast, ReferenceFrameNone}},
				MV:         [2]motion.Vector{{Row: int16(p[0]), Col: int16(p[1])}},
			},
			Flags: gridRecordMotionValid,
		})
	}
	got, ok := ctx.CollectSubChromaInterCells(BlockSize4x4, 4, 4, true, true, motion.RegularFilters)
	if !ok || got.Count != 4 {
		t.Fatalf("BlockSize4x4 should emit 4 cells, got ok=%v count=%d", ok, got.Count)
	}
	// Cells iterate row=-1..0, col=-1..0 in libaom row-major order:
	// (NW), (N), (W), (self). The trailing self cell intentionally has
	// zero MV because the helper expects the caller to fill it with the
	// just-decoded anchor's MV/reference after MarkInterMotion runs.
	want := [][3]int{
		{0, 0, 0}, // NW: offset (0,0)
		{2, 0, 0}, // N : offset (2,0) chroma samples
		{0, 2, 0}, // W : offset (0,2)
		{2, 2, 1}, // self at offset (2,2), marker=1 (synthetic anchor MV)
	}
	for i, w := range want {
		if int(got.Cells[i].OffsetX) != w[0] || int(got.Cells[i].OffsetY) != w[1] {
			t.Errorf("cell %d offset=(%d,%d) want (%d,%d)", i, got.Cells[i].OffsetX, got.Cells[i].OffsetY, w[0], w[1])
		}
		if got.Cells[i].Width != 2 || got.Cells[i].Height != 2 {
			t.Errorf("cell %d size=(%d,%d) want (2,2)", i, got.Cells[i].Width, got.Cells[i].Height)
		}
	}
	// Self cell has zero MV by design - it is patched by the caller.
	if got.Cells[3].MV != (motion.Vector{}) {
		t.Errorf("self cell MV=%+v want zero (filled by caller)", got.Cells[3].MV)
	}
}

// TestCollectSubChromaInterCellsBlockSize4x8Pair pins the BlockSize4x8
// path: width 4 triggers sub8x8 but height 8 does not, so the cell list
// holds two horizontal cells (W neighbor, self) of size (2, 4) chroma
// samples each.
func TestCollectSubChromaInterCellsBlockSize4x8Pair(t *testing.T) {
	ctx := &BlockModeContext{}
	setGridRecordCellForTest(ctx, 3, 4, blockModeGridRecord{
		Motion: InterMotionResult{
			References: InterReferencesResult{Ref: [2]ReferenceFrame{ReferenceFrameLast, ReferenceFrameNone}},
			MV:         [2]motion.Vector{{Row: 10, Col: -7}},
		},
		Flags: gridRecordMotionValid,
	})
	got, ok := ctx.CollectSubChromaInterCells(BlockSize4x8, 4, 4, true, true, motion.RegularFilters)
	if !ok || got.Count != 2 {
		t.Fatalf("BlockSize4x8 should emit 2 cells, got ok=%v count=%d", ok, got.Count)
	}
	if got.Cells[0].OffsetX != 0 || got.Cells[0].OffsetY != 0 {
		t.Errorf("W cell offset=(%d,%d) want (0,0)", got.Cells[0].OffsetX, got.Cells[0].OffsetY)
	}
	if got.Cells[1].OffsetX != 2 || got.Cells[1].OffsetY != 0 {
		t.Errorf("self cell offset=(%d,%d) want (2,0)", got.Cells[1].OffsetX, got.Cells[1].OffsetY)
	}
	if got.Cells[0].Width != 2 || got.Cells[0].Height != 4 {
		t.Errorf("W cell size=(%d,%d) want (2,4)", got.Cells[0].Width, got.Cells[0].Height)
	}
	if got.Cells[0].MV != (motion.Vector{Row: 10, Col: -7}) {
		t.Errorf("W cell MV=%+v want (10,-7)", got.Cells[0].MV)
	}
}

// TestCollectSubChromaInterCellsBlockSize8x4Pair pins the BlockSize8x4
// path: height 4 triggers sub8x8 but width 8 does not, so the cell list
// holds two vertical cells (N neighbor, self) of size (4, 2) chroma
// samples each.
func TestCollectSubChromaInterCellsBlockSize8x4Pair(t *testing.T) {
	ctx := &BlockModeContext{}
	setGridRecordCellForTest(ctx, 4, 3, blockModeGridRecord{
		Motion: InterMotionResult{
			References: InterReferencesResult{Ref: [2]ReferenceFrame{ReferenceFrameLast, ReferenceFrameNone}},
			MV:         [2]motion.Vector{{Row: -1, Col: 28}},
		},
		Flags: gridRecordMotionValid,
	})
	got, ok := ctx.CollectSubChromaInterCells(BlockSize8x4, 4, 4, true, true, motion.RegularFilters)
	if !ok || got.Count != 2 {
		t.Fatalf("BlockSize8x4 should emit 2 cells, got ok=%v count=%d", ok, got.Count)
	}
	if got.Cells[0].OffsetX != 0 || got.Cells[0].OffsetY != 0 {
		t.Errorf("N cell offset=(%d,%d) want (0,0)", got.Cells[0].OffsetX, got.Cells[0].OffsetY)
	}
	if got.Cells[1].OffsetX != 0 || got.Cells[1].OffsetY != 2 {
		t.Errorf("self cell offset=(%d,%d) want (0,2)", got.Cells[1].OffsetX, got.Cells[1].OffsetY)
	}
	if got.Cells[0].Width != 4 || got.Cells[0].Height != 2 {
		t.Errorf("N cell size=(%d,%d) want (4,2)", got.Cells[0].Width, got.Cells[0].Height)
	}
	if got.Cells[0].MV != (motion.Vector{Row: -1, Col: 28}) {
		t.Errorf("N cell MV=%+v want (-1,28)", got.Cells[0].MV)
	}
}

// TestCollectSubChromaInterCellsAbortsAtTileEdge pins that a sub-block
// whose chroma footprint extends outside the tile (the block sits at the
// top row or left column of an SB) aborts the sub8x8 path. libaom's
// is_sub8x8_inter() gate dereferences xd->mi[row * mi_stride + col] at
// row_start/col_start <= 0; goav1 maps that to the MaxBlockModeSlots
// bounds check, returning (zero, false) when any neighbor cell would
// land at a negative grid index. The caller falls back to the standard
// single-MV chroma prediction in that case.
func TestCollectSubChromaInterCellsAbortsAtTileEdge(t *testing.T) {
	ctx := &BlockModeContext{}
	// BlockSize4x4 anchor at (0, 0): the chroma footprint would require
	// reading the (−1, −1) NW neighbor which does not exist.
	got, ok := ctx.CollectSubChromaInterCells(BlockSize4x4, 0, 0, true, true, motion.RegularFilters)
	if ok || got.Count != 0 {
		t.Fatalf("anchor at tile origin must abort sub8x8, got ok=%v count=%d", ok, got.Count)
	}
}
