package decoder

import (
	"fmt"
	"sort"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/lfmask"
	"github.com/thesyncim/goav1/internal/av1/loopfilter"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/threading"
	"github.com/thesyncim/goav1/internal/av1/tile"
)

// lfmask_diff_test.go cross-validates the internal/av1/lfmask port (dav1d's
// per-superblock bitmask edge planner) against the production
// LoopFilterPostFilterPlan edge builder. Both implement the same AV1 deblocking
// decisions; this test asserts the lfmask build+scan reproduces the production
// per-4x4 edge geometry, width class, and level on synthetic block layouts.

// lumaWidthByClass maps a dav1d strength class (imin(2, log2 of the tx dimension
// perpendicular to the edge)) to the production pixel filter width. Production
// stores 4/8/14 (frameWorkLoopFilterWidthByTX); dav1d's 4<<idx (4/8/16) is the
// same filter selection under a different encoding.
var lumaWidthByClass = [3]uint8{4, 8, 14}

type lfDiffCell struct {
	width uint8
	level uint8
}

// productionLumaCells expands the production run-merged luma edge list for one
// edge direction to a per-4x4 cell map keyed by (x4, y4). Frame-boundary edges
// (x4==0 for vertical, y4==0 for horizontal) are dropped to match dav1d's
// apply-time have_left / have_top skip.
func productionLumaCells(t *testing.T, ctx FrameWorkPostFilterContext, filterMap FrameWorkLoopFilterMap, edge loopfilter.Edge) map[[2]int]lfDiffCell {
	t.Helper()
	scratch, err := ctx.LoopFilterPostFilterScratchLen(FrameWorkLoopFilterPostFilterRequest{Map: filterMap})
	if err != nil {
		t.Fatalf("scratch len: %v", err)
	}
	edges := make([]FrameWorkLoopFilterPostFilterEdge, scratch.Edges)
	plan, err := ctx.LoopFilterPostFilterPlan(FrameWorkLoopFilterPostFilterRequest{Map: filterMap, Edges: edges})
	if err != nil {
		t.Fatalf("plan: %v", err)
	}
	cells := map[[2]int]lfDiffCell{}
	for _, e := range edges[:plan.StoredEdges] {
		if e.Plane != loopfilter.PlaneY || e.Edge != edge {
			continue
		}
		for i := 0; i < int(e.Length4); i++ {
			// A vertical edge is a vertical line at fixed X4 spanning rows; a
			// horizontal edge is a horizontal line at fixed Y4 spanning columns.
			x4, y4 := int(e.X4), int(e.Y4)
			if edge == loopfilter.EdgeVertical {
				y4 += i
			} else {
				x4 += i
			}
			if edge == loopfilter.EdgeVertical && x4 == 0 {
				continue
			}
			if edge == loopfilter.EdgeHorizontal && y4 == 0 {
				continue
			}
			cells[[2]int{x4, y4}] = lfDiffCell{width: e.Width, level: e.Level}
		}
	}
	return cells
}

// lfmaskLumaCells builds lfmask masks from the records in the given (decode)
// order and scans one luma edge direction into a per-4x4 cell map keyed by
// (x4, y4), dropping frame-boundary cells to match production.
func lfmaskLumaCells(t *testing.T, event Event, size parser.FrameSize, records []threading.FrameWorkLoopFilterBlockRecord, edge loopfilter.Edge) map[[2]int]lfDiffCell {
	t.Helper()
	cols, rows, err := frameWorkLoopFilterMapGrid(size)
	if err != nil {
		t.Fatalf("grid: %v", err)
	}
	if cols > 32 || rows > 32 {
		t.Fatalf("test only covers a single 128x128 region, got %dx%d", cols, rows)
	}
	levelCtx := frameWorkLoopFilterLevelContextFor(&event)

	var builder lfmask.Builder
	var mask lfmask.FilterMask
	lc := lfmask.LevelCache{Cells: make([][4]uint8, cols*rows), Stride: cols}
	aY := make([]uint8, cols)
	for i := range aY {
		aY[i] = 2
	}
	lY := make([]uint8, 32)
	for i := range lY {
		lY[i] = 2
	}

	for i := range records {
		rec := &records[i]
		levels, err := frameWorkResolveLoopFilterRecordLevels(levelCtx, rec)
		if err != nil {
			t.Fatalf("resolve levels: %v", err)
		}
		lv := lfmask.Levels{
			YVert: levels[loopfilter.PlaneY][loopfilter.EdgeVertical],
			YHorz: levels[loopfilter.PlaneY][loopfilter.EdgeHorizontal],
		}
		bx := int(rec.Block.MICol)
		by := int(rec.Block.MIRow)
		skip := 0
		if rec.SkipTransform {
			skip = 1
		}
		builder.CreateInter(&mask, lc, lv, bx, by, cols, rows, skip,
			rec.Block.Size, rec.TransformTree.Y, rec.TransformTree.Split, rec.TransformTree.UV,
			lfmask.I420(), aY, lY, nil, nil)
	}

	cells := map[[2]int]lfDiffCell{}
	if edge == loopfilter.EdgeVertical {
		// dir 0 (columns): position = column (x4), scan bits = rows (y4).
		for col := 1; col < cols; col++ {
			lfmask.ScanLuma(&mask.Y[0][col], 0, rows, func(off, widthClass int) {
				L := lc.Cells[off*cols+col][loopfilter.EdgeVertical]
				if L == 0 {
					L = lc.Cells[off*cols+col-1][loopfilter.EdgeVertical]
				}
				if L == 0 {
					return
				}
				cells[[2]int{col, off}] = lfDiffCell{width: lumaWidthByClass[widthClass], level: L}
			})
		}
	} else {
		// dir 1 (rows): position = row (y4), scan bits = columns (x4).
		for row := 1; row < rows; row++ {
			lfmask.ScanLuma(&mask.Y[1][row], 0, cols, func(off, widthClass int) {
				L := lc.Cells[row*cols+off][loopfilter.EdgeHorizontal]
				if L == 0 {
					L = lc.Cells[(row-1)*cols+off][loopfilter.EdgeHorizontal]
				}
				if L == 0 {
					return
				}
				cells[[2]int{off, row}] = lfDiffCell{width: lumaWidthByClass[widthClass], level: L}
			})
		}
	}
	return cells
}

func assertCellsEqual(t *testing.T, prod, got map[[2]int]lfDiffCell) {
	t.Helper()
	if len(prod) == len(got) {
		equal := true
		for k, v := range prod {
			if got[k] != v {
				equal = false
				break
			}
		}
		if equal {
			return
		}
	}
	keys := map[[2]int]struct{}{}
	for k := range prod {
		keys[k] = struct{}{}
	}
	for k := range got {
		keys[k] = struct{}{}
	}
	var sorted [][2]int
	for k := range keys {
		sorted = append(sorted, k)
	}
	sort.Slice(sorted, func(i, j int) bool {
		if sorted[i][0] != sorted[j][0] {
			return sorted[i][0] < sorted[j][0]
		}
		return sorted[i][1] < sorted[j][1]
	})
	var diff string
	for _, k := range sorted {
		p, pok := prod[k]
		g, gok := got[k]
		if pok && gok && p == g {
			continue
		}
		diff += fmt.Sprintf("  cell(x4=%d,y4=%d): production=%+v(present=%v) lfmask=%+v(present=%v)\n",
			k[0], k[1], p, pok, g, gok)
	}
	t.Fatalf("edge mismatch (production vs lfmask):\n%s", diff)
}

func lfDiffEvent(size parser.FrameSize) Event {
	return Event{
		SequenceHeader: testSequence(),
		FrameSize:      size,
		LoopFilter:     parser.LoopFilterParams{LevelY: [2]uint8{8, 8}, LevelU: 8, LevelV: 8},
	}
}

// chromaWidthByClass maps a dav1d chroma strength class (!!log2dim) to the
// production chroma pixel width (frameWorkLoopFilterWidthByTX chroma column:
// 4 for the 4x4 class, 6 otherwise).
var chromaWidthByClass = [2]uint8{4, 6}

// productionChromaCells expands the production run-merged chroma edge list for
// one plane (U or V) and edge direction to a per-4x4 chroma cell map, dropping
// chroma frame-boundary cells.
func productionChromaCells(t *testing.T, ctx FrameWorkPostFilterContext, filterMap FrameWorkLoopFilterMap, plane loopfilter.Plane, edge loopfilter.Edge) map[[2]int]lfDiffCell {
	t.Helper()
	scratch, err := ctx.LoopFilterPostFilterScratchLen(FrameWorkLoopFilterPostFilterRequest{Map: filterMap})
	if err != nil {
		t.Fatalf("scratch len: %v", err)
	}
	edges := make([]FrameWorkLoopFilterPostFilterEdge, scratch.Edges)
	plan, err := ctx.LoopFilterPostFilterPlan(FrameWorkLoopFilterPostFilterRequest{Map: filterMap, Edges: edges})
	if err != nil {
		t.Fatalf("plan: %v", err)
	}
	cells := map[[2]int]lfDiffCell{}
	for _, e := range edges[:plan.StoredEdges] {
		if e.Plane != plane || e.Edge != edge {
			continue
		}
		for i := 0; i < int(e.Length4); i++ {
			x4, y4 := int(e.X4), int(e.Y4)
			if edge == loopfilter.EdgeVertical {
				y4 += i
			} else {
				x4 += i
			}
			if edge == loopfilter.EdgeVertical && x4 == 0 {
				continue
			}
			if edge == loopfilter.EdgeHorizontal && y4 == 0 {
				continue
			}
			cells[[2]int{x4, y4}] = lfDiffCell{width: e.Width, level: e.Level}
		}
	}
	return cells
}

// lfmaskChromaCells builds lfmask masks (with chroma) from records in decode
// order and scans one chroma plane/edge direction into a per-4x4 chroma cell map.
// Test coverage is 4:2:0 (ss_hor=ss_ver=1), so the mask lo/hi lane split is at
// 16>>1 == 8 bits in both directions.
func lfmaskChromaCells(t *testing.T, event Event, size parser.FrameSize, records []threading.FrameWorkLoopFilterBlockRecord, plane loopfilter.Plane, edge loopfilter.Edge) map[[2]int]lfDiffCell {
	t.Helper()
	cols, rows, err := frameWorkLoopFilterMapGrid(size)
	if err != nil {
		t.Fatalf("grid: %v", err)
	}
	if cols > 32 || rows > 32 {
		t.Fatalf("test only covers a single 128x128 region, got %dx%d", cols, rows)
	}
	levelCtx := frameWorkLoopFilterLevelContextFor(&event)

	var builder lfmask.Builder
	var mask lfmask.FilterMask
	lc := lfmask.LevelCache{Cells: make([][4]uint8, cols*rows), Stride: cols}
	aY := make([]uint8, cols)
	auv := make([]uint8, cols)
	for i := range aY {
		aY[i] = 2
		auv[i] = 1
	}
	lY := make([]uint8, 32)
	luv := make([]uint8, 32)
	for i := range lY {
		lY[i] = 2
		luv[i] = 1
	}

	for i := range records {
		rec := &records[i]
		levels, err := frameWorkResolveLoopFilterRecordLevels(levelCtx, rec)
		if err != nil {
			t.Fatalf("resolve levels: %v", err)
		}
		lv := lfmask.Levels{
			YVert: levels[loopfilter.PlaneY][loopfilter.EdgeVertical],
			YHorz: levels[loopfilter.PlaneY][loopfilter.EdgeHorizontal],
			U:     levels[loopfilter.PlaneU][loopfilter.EdgeVertical],
			V:     levels[loopfilter.PlaneV][loopfilter.EdgeVertical],
		}
		bx := int(rec.Block.MICol)
		by := int(rec.Block.MIRow)
		skip := 0
		if rec.SkipTransform {
			skip = 1
		}
		builder.CreateInter(&mask, lc, lv, bx, by, cols, rows, skip,
			rec.Block.Size, rec.TransformTree.Y, rec.TransformTree.Split, rec.TransformTree.UV,
			lfmask.I420(), aY, lY, auv, luv)
	}

	levelIdx := 2 // level cache index: 2 = U, 3 = V
	if plane == loopfilter.PlaneV {
		levelIdx = 3
	}
	ccols := (cols + 1) >> 1
	crows := (rows + 1) >> 1
	cells := map[[2]int]lfDiffCell{}
	const laneBits = 8 // 16 >> ss for 4:2:0
	if edge == loopfilter.EdgeVertical {
		for ccol := 1; ccol < ccols; ccol++ {
			lfmask.ScanChroma(&mask.UV[0][ccol], 0, crows, laneBits, func(off, widthClass int) {
				L := lc.Cells[off*cols+ccol][levelIdx]
				if L == 0 {
					L = lc.Cells[off*cols+ccol-1][levelIdx]
				}
				if L == 0 {
					return
				}
				cells[[2]int{ccol, off}] = lfDiffCell{width: chromaWidthByClass[widthClass], level: L}
			})
		}
	} else {
		for crow := 1; crow < crows; crow++ {
			lfmask.ScanChroma(&mask.UV[1][crow], 0, ccols, laneBits, func(off, widthClass int) {
				L := lc.Cells[crow*cols+off][levelIdx]
				if L == 0 {
					L = lc.Cells[(crow-1)*cols+off][levelIdx]
				}
				if L == 0 {
					return
				}
				cells[[2]int{off, crow}] = lfDiffCell{width: chromaWidthByClass[widthClass], level: L}
			})
		}
	}
	return cells
}

func runChromaDiff(t *testing.T, size parser.FrameSize, records []threading.FrameWorkLoopFilterBlockRecord, plane loopfilter.Plane, edge loopfilter.Edge) {
	t.Helper()
	event := lfDiffEvent(size)
	ctx := FrameWorkPostFilterContext{Event: event}
	filterMap := testFrameWorkLoopFilterPostFilterMap(t, size, records...)
	prod := productionChromaCells(t, ctx, filterMap, plane, edge)
	got := lfmaskChromaCells(t, event, size, records, plane, edge)
	if len(prod) == 0 {
		t.Fatalf("production produced no chroma edges for plane=%d edge=%d; test setup is degenerate", plane, edge)
	}
	assertCellsEqual(t, prod, got)
}

// runLumaDiff builds the production and lfmask per-4x4 luma cell maps for one
// edge direction and asserts they are identical.
func runLumaDiff(t *testing.T, size parser.FrameSize, records []threading.FrameWorkLoopFilterBlockRecord, edge loopfilter.Edge) {
	t.Helper()
	runLumaDiffWithEvent(t, lfDiffEvent(size), size, records, edge)
}

func runLumaDiffWithEvent(t *testing.T, event Event, size parser.FrameSize, records []threading.FrameWorkLoopFilterBlockRecord, edge loopfilter.Edge) {
	t.Helper()
	ctx := FrameWorkPostFilterContext{Event: event}
	filterMap := testFrameWorkLoopFilterPostFilterMap(t, size, records...)
	prod := productionLumaCells(t, ctx, filterMap, edge)
	got := lfmaskLumaCells(t, event, size, records, edge)
	if len(prod) == 0 {
		t.Fatalf("production produced no luma edges for edge=%d; test setup is degenerate", edge)
	}
	assertCellsEqual(t, prod, got)
}

func TestLFMaskDiffLumaVerticalLevelFromPrevious(t *testing.T) {
	// Right block resolves to loop-filter level 0 (segment delta), left block to 8.
	// The shared vertical boundary edge must fall back to the previous (left) side
	// for its level (AV1 "level from previous"), exactly as production does. The
	// mask geometry is level-independent; the fallback happens during the scan.
	size := parser.FrameSize{CodedWidth: 32, UpscaledWidth: 32, Height: 16, SuperResDenominator: 8}
	event := lfDiffEvent(size)
	event.Segmentation.Enabled = true
	event.Segmentation.Data.Segments[1].DeltaLFYV = -63 // drives level_y_v to 0
	event.Segmentation.Data.Segments[1].DeltaLFYH = -63

	left := testFrameWorkLoopFilterPostFilterRecordAt(0, 0, 4, 4)
	left.SegmentID = 0
	left.TransformTree = tile.TransformTreeResult{Y: tile.TransformSize16x16, UV: tile.TransformSize8x8, HasUV: true}
	right := testFrameWorkLoopFilterPostFilterRecordAt(4, 0, 8, 4)
	right.SegmentID = 1
	right.TransformTree = tile.TransformTreeResult{Y: tile.TransformSize16x16, UV: tile.TransformSize8x8, HasUV: true}
	recs := []threading.FrameWorkLoopFilterBlockRecord{left, right}
	runLumaDiffWithEvent(t, event, size, recs, loopfilter.EdgeVertical)
}

func TestLFMaskDiffLumaSingleSplitBlock(t *testing.T) {
	// One 16x16 inter block filling a 16x16 frame, transform tree split to four
	// 8x8 sub-transforms. Only inner tx edges exist (block edges are the frame
	// boundary at x4=0 / y4=0, dropped on both sides).
	size := parser.FrameSize{CodedWidth: 16, UpscaledWidth: 16, Height: 16, SuperResDenominator: 8}
	rec := testFrameWorkLoopFilterPostFilterRecordAt(0, 0, 4, 4)
	rec.TransformTree = tile.TransformTreeResult{Y: tile.TransformSize16x16, UV: tile.TransformSize8x8, HasUV: true, Variable: true, Split: [2]uint16{1, 0}}
	recs := []threading.FrameWorkLoopFilterBlockRecord{rec}
	runLumaDiff(t, size, recs, loopfilter.EdgeVertical)
	runLumaDiff(t, size, recs, loopfilter.EdgeHorizontal)
}

func TestLFMaskDiffLumaTwoBlocksDecodeOrder(t *testing.T) {
	// Two horizontally adjacent inter blocks. The left block's right-edge tx must
	// flow through the carried l[] context so block2's left-edge strength resolves
	// as imin(block2 tx, block1 tx) exactly as the production neighbour lookup does.
	// Records are supplied in decode order (left then right).
	size := parser.FrameSize{CodedWidth: 32, UpscaledWidth: 32, Height: 16, SuperResDenominator: 8}
	left := testFrameWorkLoopFilterPostFilterRecordAt(0, 0, 4, 4)
	left.SkipTransform = true
	right := testFrameWorkLoopFilterPostFilterRecordAt(4, 0, 8, 4)
	right.TransformTree = tile.TransformTreeResult{Y: tile.TransformSize16x16, UV: tile.TransformSize8x8, HasUV: true, Variable: true, Split: [2]uint16{1, 0}}
	recs := []threading.FrameWorkLoopFilterBlockRecord{left, right}
	runLumaDiff(t, size, recs, loopfilter.EdgeVertical)
	runLumaDiff(t, size, recs, loopfilter.EdgeHorizontal)
}

func TestLFMaskDiffLumaTwoBlocksUnequalTx(t *testing.T) {
	// Left block uses a small (8x8) fixed transform, right block a large (16x16)
	// one, so the shared boundary strength is genuinely min(prev,cur) rather than
	// a single class. Exercises the l[] carry across differing transform sizes.
	size := parser.FrameSize{CodedWidth: 32, UpscaledWidth: 32, Height: 16, SuperResDenominator: 8}
	left := testFrameWorkLoopFilterPostFilterRecordAt(0, 0, 4, 4)
	left.TransformTree = tile.TransformTreeResult{Y: tile.TransformSize8x8, UV: tile.TransformSize4x4, HasUV: true, Variable: true, Split: [2]uint16{1, 0}}
	right := testFrameWorkLoopFilterPostFilterRecordAt(4, 0, 8, 4)
	right.TransformTree = tile.TransformTreeResult{Y: tile.TransformSize16x16, UV: tile.TransformSize8x8, HasUV: true}
	recs := []threading.FrameWorkLoopFilterBlockRecord{left, right}
	runLumaDiff(t, size, recs, loopfilter.EdgeVertical)
	runLumaDiff(t, size, recs, loopfilter.EdgeHorizontal)
}

func TestLFMaskDiffLumaVerticalStackedBlocks(t *testing.T) {
	// Two vertically stacked inter blocks with differing transforms. The upper
	// block's bottom-edge tx must flow through the carried a[] (above) context so
	// the lower block's top horizontal edge resolves as min(prev,cur). Records in
	// decode order (top then bottom).
	size := parser.FrameSize{CodedWidth: 16, UpscaledWidth: 16, Height: 32, SuperResDenominator: 8}
	top := testFrameWorkLoopFilterPostFilterRecordAt(0, 0, 4, 4)
	top.TransformTree = tile.TransformTreeResult{Y: tile.TransformSize8x8, UV: tile.TransformSize4x4, HasUV: true, Variable: true, Split: [2]uint16{1, 0}}
	bottom := testFrameWorkLoopFilterPostFilterRecordAt(0, 4, 4, 8)
	bottom.TransformTree = tile.TransformTreeResult{Y: tile.TransformSize16x16, UV: tile.TransformSize8x8, HasUV: true}
	recs := []threading.FrameWorkLoopFilterBlockRecord{top, bottom}
	runLumaDiff(t, size, recs, loopfilter.EdgeVertical)
	runLumaDiff(t, size, recs, loopfilter.EdgeHorizontal)
}

func TestLFMaskDiffChromaSingleBlock(t *testing.T) {
	// One 32x32 inter block filling a 32x32 frame with an 8x8 chroma transform, so
	// the 16x16 chroma area (4:2:0) tiles into a 2x2 grid of 8x8 chroma transforms
	// with interior chroma edges (the block's own boundary edges are at the frame
	// boundary and dropped on both sides).
	size := parser.FrameSize{CodedWidth: 32, UpscaledWidth: 32, Height: 32, SuperResDenominator: 8}
	rec := testFrameWorkLoopFilterPostFilterRecordAt(0, 0, 8, 8)
	rec.Block.Size = tile.BlockSize32x32
	rec.TransformTree = tile.TransformTreeResult{Y: tile.TransformSize32x32, UV: tile.TransformSize8x8, HasUV: true}
	recs := []threading.FrameWorkLoopFilterBlockRecord{rec}
	runChromaDiff(t, size, recs, loopfilter.PlaneU, loopfilter.EdgeVertical)
	runChromaDiff(t, size, recs, loopfilter.PlaneU, loopfilter.EdgeHorizontal)
	runChromaDiff(t, size, recs, loopfilter.PlaneV, loopfilter.EdgeVertical)
	runChromaDiff(t, size, recs, loopfilter.PlaneV, loopfilter.EdgeHorizontal)
}

func TestLFMaskDiffChromaTwoBlocksDecodeOrder(t *testing.T) {
	// Two horizontally adjacent 32x32 inter blocks; the shared chroma boundary
	// resolves through the carried chroma left context. 4:2:0.
	size := parser.FrameSize{CodedWidth: 64, UpscaledWidth: 64, Height: 32, SuperResDenominator: 8}
	left := testFrameWorkLoopFilterPostFilterRecordAt(0, 0, 8, 8)
	left.Block.Size = tile.BlockSize32x32
	left.TransformTree = tile.TransformTreeResult{Y: tile.TransformSize16x16, UV: tile.TransformSize8x8, HasUV: true}
	right := testFrameWorkLoopFilterPostFilterRecordAt(8, 0, 16, 8)
	right.Block.Size = tile.BlockSize32x32
	right.TransformTree = tile.TransformTreeResult{Y: tile.TransformSize32x32, UV: tile.TransformSize16x16, HasUV: true}
	recs := []threading.FrameWorkLoopFilterBlockRecord{left, right}
	runChromaDiff(t, size, recs, loopfilter.PlaneU, loopfilter.EdgeVertical)
	runChromaDiff(t, size, recs, loopfilter.PlaneV, loopfilter.EdgeVertical)
}
