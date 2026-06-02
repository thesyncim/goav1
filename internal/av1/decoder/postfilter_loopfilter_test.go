package decoder

import (
	"bytes"
	"encoding/binary"
	"errors"
	"math/rand"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/cdef"
	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/loopfilter"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/threading"
	"github.com/thesyncim/goav1/internal/av1/tile"
)

func TestFrameWorkPostFilterContextLoopFilterPostFilterPlanSkipsInactive(t *testing.T) {
	plan, err := (FrameWorkPostFilterContext{}).LoopFilterPostFilterPlan(FrameWorkLoopFilterPostFilterRequest{})
	if err != nil {
		t.Fatal(err)
	}
	if plan != (FrameWorkLoopFilterPostFilterPlan{}) {
		t.Fatalf("plan=%+v want zero", plan)
	}
	size, err := (FrameWorkPostFilterContext{}).LoopFilterPostFilterScratchLen(FrameWorkLoopFilterPostFilterRequest{})
	if err != nil {
		t.Fatal(err)
	}
	if size != (FrameWorkLoopFilterPostFilterScratchSize{}) {
		t.Fatalf("scratch=%+v want zero", size)
	}
	upper, err := (FrameWorkPostFilterContext{}).LoopFilterPostFilterScratchUpperBound()
	if err != nil {
		t.Fatal(err)
	}
	if upper != (FrameWorkLoopFilterPostFilterScratchSize{}) {
		t.Fatalf("upper=%+v want zero", upper)
	}
}

func TestFrameWorkPostFilterContextLoopFilterPostFilterScratchUpperBound(t *testing.T) {
	size := parser.FrameSize{
		CodedWidth:          16,
		UpscaledWidth:       16,
		Height:              16,
		SuperResDenominator: 8,
	}
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			SequenceHeader: testSequence(),
			FrameSize:      size,
			LoopFilter:     parser.LoopFilterParams{LevelY: [2]uint8{8}},
		},
	}
	cols, rows, err := frameWorkLoopFilterMapGrid(size)
	if err != nil {
		t.Fatal(err)
	}
	upper, err := ctx.LoopFilterPostFilterScratchUpperBound()
	if err != nil {
		t.Fatal(err)
	}
	if upper.Edges != cols*rows*8 {
		t.Fatalf("upper edges=%d want %d", upper.Edges, cols*rows*8)
	}
}

func TestFrameWorkPostFilterContextLoopFilterPostFilterPlanDefaultsMapAndResolvesLevels(t *testing.T) {
	size := parser.FrameSize{
		CodedWidth:          16,
		UpscaledWidth:       16,
		Height:              16,
		SuperResDenominator: 8,
	}
	cols, rows, err := frameWorkLoopFilterMapGrid(size)
	if err != nil {
		t.Fatal(err)
	}
	record := testFrameWorkLoopFilterPostFilterRecord(cols, rows)
	record.DeltaLF = [tile.FrameLoopFilterCount]int8{-2, 3, 4, -1}
	filterMap := testFrameWorkLoopFilterPostFilterMap(t, size, record)
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			SequenceHeader: testSequence(),
			FrameSize:      size,
			Delta: parser.DeltaParams{
				DeltaLFPresent: true,
				DeltaLFMulti:   true,
			},
			LoopFilter: parser.LoopFilterParams{
				LevelY: [2]uint8{16, 20},
				LevelU: 8,
				LevelV: 4,
			},
		},
		LoopFilterMap: &filterMap,
	}

	plan, err := ctx.LoopFilterPostFilterPlan(FrameWorkLoopFilterPostFilterRequest{})
	if err != nil {
		t.Fatal(err)
	}
	if !plan.Active || plan.MICols != cols || plan.MIRows != rows || plan.Cells != cols*rows || plan.Blocks != 1 || plan.Missing != 0 {
		t.Fatalf("plan=%+v", plan)
	}
	if plan.TransformReadyBlocks != 1 || plan.SkipTransformBlocks != 0 || plan.LumaTXBs != 1 {
		t.Fatalf("transform stats=%+v", plan)
	}
	wantYV := FrameWorkLoopFilterPostFilterLevelStats{Blocks: 1, NonZero: 1, MaxLevel: 14}
	if got := plan.Levels[loopfilter.PlaneY][loopfilter.EdgeVertical]; got != wantYV {
		t.Fatalf("Y vertical=%+v want %+v", got, wantYV)
	}
	wantYH := FrameWorkLoopFilterPostFilterLevelStats{Blocks: 1, NonZero: 1, MaxLevel: 23}
	if got := plan.Levels[loopfilter.PlaneY][loopfilter.EdgeHorizontal]; got != wantYH {
		t.Fatalf("Y horizontal=%+v want %+v", got, wantYH)
	}
	wantU := FrameWorkLoopFilterPostFilterLevelStats{Blocks: 1, NonZero: 1, MaxLevel: 12}
	if got := plan.Levels[loopfilter.PlaneU][loopfilter.EdgeHorizontal]; got != wantU {
		t.Fatalf("U horizontal=%+v want %+v", got, wantU)
	}
	wantV := FrameWorkLoopFilterPostFilterLevelStats{Blocks: 1, NonZero: 1, MaxLevel: 3}
	if got := plan.Levels[loopfilter.PlaneV][loopfilter.EdgeVertical]; got != wantV {
		t.Fatalf("V vertical=%+v want %+v", got, wantV)
	}
}

func TestFrameWorkPostFilterContextLoopFilterPostFilterPlanIgnoresPaddedMapCells(t *testing.T) {
	size := parser.FrameSize{
		CodedWidth:          16,
		UpscaledWidth:       16,
		Height:              16,
		SuperResDenominator: 8,
	}
	cols, rows, err := frameWorkLoopFilterMapGrid(size)
	if err != nil {
		t.Fatal(err)
	}
	record := testFrameWorkLoopFilterPostFilterRecord(cols, rows)
	filterMap := FrameWorkLoopFilterMap{
		Records: make([]threading.FrameWorkLoopFilterBlockRecord, (cols+2)*(rows+1)),
		Stride:  cols + 2,
		Rows:    rows + 1,
	}
	for miRow := record.Block.MIRow; miRow < record.Block.MIRowEnd; miRow++ {
		row := int(miRow) * filterMap.Stride
		for miCol := record.Block.MICol; miCol < record.Block.MIColEnd; miCol++ {
			filterMap.Records[row+int(miCol)] = record
		}
	}
	filterMap.Records[cols] = threading.FrameWorkLoopFilterBlockRecord{
		Valid: true,
		Block: tile.BlockVisit{MICol: uint32(cols), MIRow: 0, MIColEnd: uint32(cols + 1), MIRowEnd: 1},
	}
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			SequenceHeader: testSequence(),
			FrameSize:      size,
			LoopFilter:     parser.LoopFilterParams{LevelY: [2]uint8{8}},
		},
		LoopFilterMap: &filterMap,
	}

	plan, err := ctx.LoopFilterPostFilterPlan(FrameWorkLoopFilterPostFilterRequest{})
	if err != nil {
		t.Fatal(err)
	}
	if plan.Cells != cols*rows || plan.Blocks != 1 || plan.Missing != 0 {
		t.Fatalf("plan=%+v want cells=%d blocks=1 missing=0", plan, cols*rows)
	}
}

func TestFrameWorkPostFilterContextLoopFilterPostFilterPlanCountsTransformReplay(t *testing.T) {
	size := parser.FrameSize{
		CodedWidth:          32,
		UpscaledWidth:       32,
		Height:              16,
		SuperResDenominator: 8,
	}
	first := testFrameWorkLoopFilterPostFilterRecordAt(0, 0, 4, 4)
	first.SkipTransform = true
	second := testFrameWorkLoopFilterPostFilterRecordAt(4, 0, 8, 4)
	second.TransformTree = tile.TransformTreeResult{
		Y:        tile.TransformSize16x16,
		Variable: true,
		Split:    [2]uint16{1, 0},
	}
	filterMap := testFrameWorkLoopFilterPostFilterMap(t, size, first, second)
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			SequenceHeader: testSequence(),
			FrameSize:      size,
			LoopFilter:     parser.LoopFilterParams{LevelY: [2]uint8{8}},
		},
	}

	plan, err := ctx.LoopFilterPostFilterPlan(FrameWorkLoopFilterPostFilterRequest{Map: filterMap})
	if err != nil {
		t.Fatal(err)
	}
	if plan.Blocks != 2 || plan.TransformReadyBlocks != 2 || plan.SkipTransformBlocks != 1 || plan.LumaTXBs != 4 {
		t.Fatalf("plan=%+v", plan)
	}
}

func TestFrameWorkPostFilterContextLoopFilterPostFilterPlanStoresLumaEdges(t *testing.T) {
	size := parser.FrameSize{
		CodedWidth:          32,
		UpscaledWidth:       32,
		Height:              16,
		SuperResDenominator: 8,
	}
	first := testFrameWorkLoopFilterPostFilterRecordAt(0, 0, 4, 4)
	first.SkipTransform = true
	second := testFrameWorkLoopFilterPostFilterRecordAt(4, 0, 8, 4)
	second.TransformTree = tile.TransformTreeResult{
		Y:        tile.TransformSize16x16,
		Variable: true,
		Split:    [2]uint16{1, 0},
	}
	filterMap := testFrameWorkLoopFilterPostFilterMap(t, size, first, second)
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			SequenceHeader: testSequence(),
			FrameSize:      size,
			LoopFilter:     parser.LoopFilterParams{LevelY: [2]uint8{8}},
		},
	}
	edges := make([]FrameWorkLoopFilterPostFilterEdge, 4)
	scratchSize, err := ctx.LoopFilterPostFilterScratchLen(FrameWorkLoopFilterPostFilterRequest{Map: filterMap})
	if err != nil {
		t.Fatal(err)
	}
	if scratchSize.Edges != len(edges) {
		t.Fatalf("scratch=%+v want %d edges", scratchSize, len(edges))
	}

	plan, err := ctx.LoopFilterPostFilterPlan(FrameWorkLoopFilterPostFilterRequest{
		Map:   filterMap,
		Edges: edges,
	})
	if err != nil {
		t.Fatal(err)
	}
	if plan.EdgeCandidates != 4 || plan.StoredEdges != len(edges) || plan.DroppedEdges != 0 {
		t.Fatalf("plan=%+v", plan)
	}
	want := []FrameWorkLoopFilterPostFilterEdge{
		{Plane: loopfilter.PlaneY, Edge: loopfilter.EdgeVertical, X4: 4, Y4: 0, Length4: 2, Level: 8, Transform: tile.TransformSize8x8, Width: 8, BlockMICol: 4, BlockMIRow: 0},
		{Plane: loopfilter.PlaneY, Edge: loopfilter.EdgeVertical, X4: 6, Y4: 0, Length4: 2, Level: 8, Transform: tile.TransformSize8x8, Width: 8, BlockMICol: 4, BlockMIRow: 0},
		{Plane: loopfilter.PlaneY, Edge: loopfilter.EdgeVertical, X4: 4, Y4: 2, Length4: 2, Level: 8, Transform: tile.TransformSize8x8, Width: 8, BlockMICol: 4, BlockMIRow: 0},
		{Plane: loopfilter.PlaneY, Edge: loopfilter.EdgeVertical, X4: 6, Y4: 2, Length4: 2, Level: 8, Transform: tile.TransformSize8x8, Width: 8, BlockMICol: 4, BlockMIRow: 0},
	}
	for i := range want {
		if edges[i] != want[i] {
			t.Fatalf("edge[%d]=%+v want %+v", i, edges[i], want[i])
		}
	}
}

func TestFrameWorkLoopFilterPostFilterScratchSizeBindEdges(t *testing.T) {
	size := FrameWorkLoopFilterPostFilterScratchSize{Edges: 3}
	storage := make([]FrameWorkLoopFilterPostFilterEdge, 4)
	got, err := size.BindEdges(storage)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 3 || cap(got) != cap(storage) {
		t.Fatalf("bound len/cap=%d/%d want len 3 cap %d", len(got), cap(got), cap(storage))
	}
	if _, err := size.BindEdges(storage[:2]); !errors.Is(err, frame.ErrShortBuffer) {
		t.Fatalf("short bind err=%v want %v", err, frame.ErrShortBuffer)
	}
	if _, err := (FrameWorkLoopFilterPostFilterScratchSize{Edges: -1}).BindEdges(storage); !errors.Is(err, frame.ErrShortBuffer) {
		t.Fatalf("negative bind err=%v want %v", err, frame.ErrShortBuffer)
	}
}

func TestFrameWorkPostFilterContextLoopFilterPostFilterPlanStoresChromaEdges(t *testing.T) {
	size := parser.FrameSize{
		CodedWidth:          32,
		UpscaledWidth:       32,
		Height:              16,
		SuperResDenominator: 8,
	}
	first := testFrameWorkLoopFilterPostFilterRecordAt(0, 0, 4, 4)
	second := testFrameWorkLoopFilterPostFilterRecordAt(4, 0, 8, 4)
	filterMap := testFrameWorkLoopFilterPostFilterMap(t, size, first, second)
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			SequenceHeader: testSequence(),
			FrameSize:      size,
			LoopFilter: parser.LoopFilterParams{
				LevelY: [2]uint8{8},
				LevelU: 7,
				LevelV: 9,
			},
		},
	}
	scratchSize, err := ctx.LoopFilterPostFilterScratchLen(FrameWorkLoopFilterPostFilterRequest{Map: filterMap})
	if err != nil {
		t.Fatal(err)
	}
	if scratchSize.Edges != 3 {
		t.Fatalf("scratch=%+v want 3 edges", scratchSize)
	}
	edges := make([]FrameWorkLoopFilterPostFilterEdge, scratchSize.Edges)

	plan, err := ctx.LoopFilterPostFilterPlan(FrameWorkLoopFilterPostFilterRequest{
		Map:   filterMap,
		Edges: edges,
	})
	if err != nil {
		t.Fatal(err)
	}
	if plan.EdgeCandidates != 3 || plan.StoredEdges != 3 || plan.DroppedEdges != 0 {
		t.Fatalf("plan=%+v", plan)
	}
	if plan.PlaneEdgeCandidates != [3]int{1, 1, 1} {
		t.Fatalf("plane edge candidates=%v want [1 1 1]", plan.PlaneEdgeCandidates)
	}
	if plan.ChromaTXBs != 2 {
		t.Fatalf("chroma txbs=%d want 2", plan.ChromaTXBs)
	}
	want := []FrameWorkLoopFilterPostFilterEdge{
		{Plane: loopfilter.PlaneY, Edge: loopfilter.EdgeVertical, X4: 4, Y4: 0, Length4: 4, Level: 8, Transform: tile.TransformSize16x16, Width: 14, BlockMICol: 4, BlockMIRow: 0},
		{Plane: loopfilter.PlaneU, Edge: loopfilter.EdgeVertical, X4: 2, Y4: 0, Length4: 2, Level: 7, Transform: tile.TransformSize8x8, Width: 6, BlockMICol: 4, BlockMIRow: 0},
		{Plane: loopfilter.PlaneV, Edge: loopfilter.EdgeVertical, X4: 2, Y4: 0, Length4: 2, Level: 9, Transform: tile.TransformSize8x8, Width: 6, BlockMICol: 4, BlockMIRow: 0},
	}
	for i := range want {
		if edges[i] != want[i] {
			t.Fatalf("edge[%d]=%+v want %+v", i, edges[i], want[i])
		}
	}
}

func TestFrameWorkPostFilterContextLoopFilterPostFilterPlanChromaSubsamplingGeometries(t *testing.T) {
	tests := []struct {
		name    string
		ssX     bool
		ssY     bool
		wantX4  int
		wantY4  int
		wantLen int
	}{
		{name: "444", wantX4: 4, wantY4: 0, wantLen: 4},
		{name: "422", ssX: true, wantX4: 2, wantY4: 0, wantLen: 4},
		{name: "420", ssX: true, ssY: true, wantX4: 2, wantY4: 0, wantLen: 2},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			size := parser.FrameSize{
				CodedWidth:          32,
				UpscaledWidth:       32,
				Height:              16,
				SuperResDenominator: 8,
			}
			seq := testSequence()
			seq.ColorConfig.SubsamplingX = tt.ssX
			seq.ColorConfig.SubsamplingY = tt.ssY
			first := testFrameWorkLoopFilterPostFilterRecordAt(0, 0, 4, 4)
			first.SkipTransform = true
			second := testFrameWorkLoopFilterPostFilterRecordAt(4, 0, 8, 4)
			second.SkipTransform = true
			filterMap := testFrameWorkLoopFilterPostFilterMap(t, size, first, second)
			ctx := FrameWorkPostFilterContext{
				Event: Event{
					SequenceHeader: seq,
					FrameSize:      size,
					LoopFilter: parser.LoopFilterParams{
						LevelY: [2]uint8{8},
						LevelU: 7,
					},
				},
			}
			edges := make([]FrameWorkLoopFilterPostFilterEdge, 2)

			plan, err := ctx.LoopFilterPostFilterPlan(FrameWorkLoopFilterPostFilterRequest{
				Map:   filterMap,
				Edges: edges,
			})
			if err != nil {
				t.Fatal(err)
			}
			if plan.EdgeCandidates != 2 || plan.StoredEdges != 2 || plan.PlaneEdgeCandidates != [3]int{1, 1, 0} {
				t.Fatalf("plan=%+v", plan)
			}
			wantU := FrameWorkLoopFilterPostFilterEdge{
				Plane:      loopfilter.PlaneU,
				Edge:       loopfilter.EdgeVertical,
				X4:         int32(tt.wantX4),
				Y4:         int32(tt.wantY4),
				Length4:    int32(tt.wantLen),
				Level:      7,
				Transform:  tile.TransformSize8x8,
				Width:      6,
				BlockMICol: 4,
				BlockMIRow: 0,
			}
			if edges[1] != wantU {
				t.Fatalf("U edge=%+v want %+v", edges[1], wantU)
			}
		})
	}
}

func TestFrameWorkPostFilterContextLoopFilterPostFilterPlanSchedulesLumaEdgeWidths(t *testing.T) {
	size := parser.FrameSize{
		CodedWidth:          32,
		UpscaledWidth:       32,
		Height:              16,
		SuperResDenominator: 8,
	}
	cols, rows, err := frameWorkLoopFilterMapGrid(size)
	if err != nil {
		t.Fatal(err)
	}
	filler := testFrameWorkLoopFilterPostFilterRecord(cols, rows)
	filler.Block.Size = tile.BlockSize32x16
	filler.SkipTransform = true
	tx4 := testFrameWorkLoopFilterPostFilterRecordAt(1, 0, 2, 1)
	tx4.SkipTransform = true
	tx4.TransformTree = tile.TransformTreeResult{Y: tile.TransformSize4x4}
	tx8 := testFrameWorkLoopFilterPostFilterRecordAt(2, 0, 4, 2)
	tx8.SkipTransform = true
	tx8.TransformTree = tile.TransformTreeResult{Y: tile.TransformSize8x8}
	tx16 := testFrameWorkLoopFilterPostFilterRecordAt(4, 0, 8, 4)
	tx16.SkipTransform = true
	tx16.TransformTree = tile.TransformTreeResult{Y: tile.TransformSize16x16}
	filterMap := testFrameWorkLoopFilterPostFilterMap(t, size, filler, tx4, tx8, tx16)
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			SequenceHeader: testSequence(),
			FrameSize:      size,
			LoopFilter:     parser.LoopFilterParams{LevelY: [2]uint8{8}},
		},
	}
	edges := make([]FrameWorkLoopFilterPostFilterEdge, 3)
	plan, err := ctx.LoopFilterPostFilterPlan(FrameWorkLoopFilterPostFilterRequest{
		Map:   filterMap,
		Edges: edges,
	})
	if err != nil {
		t.Fatal(err)
	}
	if plan.StoredEdges != 3 {
		t.Fatalf("plan=%+v edges=%+v", plan, edges)
	}
	for i, want := range []int{4, 4, 8} {
		if int(edges[i].Width) != want {
			t.Fatalf("edge[%d]=%+v want width %d", i, edges[i], want)
		}
	}
}

// TestFrameWorkPostFilterContextLoopFilterPostFilterPlanSplitsLumaEdgeByPerCellWidth
// covers the 10-bit quantizer 32 LF regression: when a single luma transform
// edge spans multiple MI cells, each MI cell's filter width must come from the
// (current, previous) transform pair at THAT cell, matching libaom's
// set_one_param_for_line_luma. The previous implementation collapsed the edge
// down to min(previousWidth) across the whole length, applying the narrower
// filter at cells that should have used the wider one. The bottom 8x8 TX
// block here straddles two top-side TXs (TX_8X8 vs TX_4X4) and must emit two
// per-MI-cell width-8 / width-4 horizontal edges instead of one width-4 edge.
func TestFrameWorkPostFilterContextLoopFilterPostFilterPlanSplitsLumaEdgeByPerCellWidth(t *testing.T) {
	size := parser.FrameSize{
		CodedWidth:          16,
		UpscaledWidth:       16,
		Height:              16,
		SuperResDenominator: 8,
	}
	cols, rows, err := frameWorkLoopFilterMapGrid(size)
	if err != nil {
		t.Fatal(err)
	}
	filler := testFrameWorkLoopFilterPostFilterRecord(cols, rows)
	filler.Block.Size = tile.BlockSize16x16
	filler.SkipTransform = true
	// Top row: two side-by-side 1-MI-tall TX blocks with different sizes.
	// Top-left has TX_8X8 (filter width=8 for horizontal); top-right has
	// TX_4X4 (filter width=4). They share the y=4 horizontal boundary with
	// the bottom block below.
	topLeft := testFrameWorkLoopFilterPostFilterRecordAt(0, 0, 2, 1)
	topLeft.SkipTransform = true
	topLeft.TransformTree = tile.TransformTreeResult{Y: tile.TransformSize8x8}
	topRight := testFrameWorkLoopFilterPostFilterRecordAt(2, 0, 4, 1)
	topRight.SkipTransform = true
	topRight.TransformTree = tile.TransformTreeResult{Y: tile.TransformSize4x4}
	// Bottom block spans cols 0..3, rows 1..3 with TX_8X8 (width=8 horiz).
	bottom := testFrameWorkLoopFilterPostFilterRecordAt(0, 1, 4, 4)
	bottom.SkipTransform = true
	bottom.TransformTree = tile.TransformTreeResult{Y: tile.TransformSize8x8}
	filterMap := testFrameWorkLoopFilterPostFilterMap(t, size, filler, topLeft, topRight, bottom)
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			SequenceHeader: testSequence(),
			FrameSize:      size,
			LoopFilter:     parser.LoopFilterParams{LevelY: [2]uint8{8, 8}},
		},
	}
	edges := make([]FrameWorkLoopFilterPostFilterEdge, 16)
	plan, err := ctx.LoopFilterPostFilterPlan(FrameWorkLoopFilterPostFilterRequest{
		Map:   filterMap,
		Edges: edges,
	})
	if err != nil {
		t.Fatal(err)
	}
	// Find plane-Y horizontal edges at y=4 (Y4=1) emitted by `bottom`. The
	// fixed code splits the horizontal edge between cols 0..3 into two
	// segments: cols 0..1 (prev TX_8X8 -> width 8) and cols 2..3 (prev
	// TX_4X4 -> width 4). The old behaviour emitted a single length=4
	// width=4 edge.
	var horizY []FrameWorkLoopFilterPostFilterEdge
	for i := 0; i < plan.StoredEdges; i++ {
		e := edges[i]
		if e.Plane == loopfilter.PlaneY && e.Edge == loopfilter.EdgeHorizontal && e.Y4 == 1 {
			horizY = append(horizY, e)
		}
	}
	if len(horizY) != 2 {
		t.Fatalf("expected 2 per-MI-cell width segments, got %d: %+v plan=%+v", len(horizY), horizY, plan)
	}
	if horizY[0].X4 != 0 || horizY[0].Length4 != 2 || horizY[0].Width != 8 {
		t.Fatalf("edge[0]=%+v want X4=0 Length4=2 Width=8", horizY[0])
	}
	if horizY[1].X4 != 2 || horizY[1].Length4 != 2 || horizY[1].Width != 4 {
		t.Fatalf("edge[1]=%+v want X4=2 Length4=2 Width=4", horizY[1])
	}
}

func TestFrameWorkPostFilterContextLoopFilterPostFilterPlanLimitsHorizontalLumaWidthByPreviousTransform(t *testing.T) {
	size := parser.FrameSize{
		CodedWidth:          16,
		UpscaledWidth:       16,
		Height:              32,
		SuperResDenominator: 8,
	}
	cols, rows, err := frameWorkLoopFilterMapGrid(size)
	if err != nil {
		t.Fatal(err)
	}
	filler := testFrameWorkLoopFilterPostFilterRecord(cols, rows)
	filler.Block.Size = tile.BlockSize16x32
	filler.SkipTransform = true
	tx4 := testFrameWorkLoopFilterPostFilterRecordAt(0, 1, 1, 2)
	tx4.SkipTransform = true
	tx4.TransformTree = tile.TransformTreeResult{Y: tile.TransformSize4x4}
	tx8 := testFrameWorkLoopFilterPostFilterRecordAt(0, 2, 2, 4)
	tx8.SkipTransform = true
	tx8.TransformTree = tile.TransformTreeResult{Y: tile.TransformSize8x8}
	tx16 := testFrameWorkLoopFilterPostFilterRecordAt(0, 4, 4, 8)
	tx16.SkipTransform = true
	tx16.TransformTree = tile.TransformTreeResult{Y: tile.TransformSize16x16}
	filterMap := testFrameWorkLoopFilterPostFilterMap(t, size, filler, tx4, tx8, tx16)
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			SequenceHeader: testSequence(),
			FrameSize:      size,
			LoopFilter:     parser.LoopFilterParams{LevelY: [2]uint8{0, 8}},
		},
	}
	edges := make([]FrameWorkLoopFilterPostFilterEdge, 3)
	plan, err := ctx.LoopFilterPostFilterPlan(FrameWorkLoopFilterPostFilterRequest{
		Map:   filterMap,
		Edges: edges,
	})
	if err != nil {
		t.Fatal(err)
	}
	if plan.StoredEdges != 3 {
		t.Fatalf("plan=%+v edges=%+v", plan, edges)
	}
	for i, want := range []int{4, 4, 8} {
		if int(edges[i].Width) != want {
			t.Fatalf("edge[%d]=%+v want width %d", i, edges[i], want)
		}
	}
}

func TestFrameWorkPostFilterContextLoopFilterPostFilterPlanLimitsChromaWidthByPreviousTransform(t *testing.T) {
	size := parser.FrameSize{
		CodedWidth:          16,
		UpscaledWidth:       16,
		Height:              16,
		SuperResDenominator: 8,
	}
	seq := testSequence()
	seq.ColorConfig.SubsamplingX = false
	seq.ColorConfig.SubsamplingY = false
	left := testFrameWorkLoopFilterPostFilterRecordAt(0, 0, 2, 4)
	left.SkipTransform = true
	left.TransformTree = tile.TransformTreeResult{Y: tile.TransformSize16x16, UV: tile.TransformSize4x4, HasUV: true}
	right := testFrameWorkLoopFilterPostFilterRecordAt(2, 0, 4, 4)
	right.SkipTransform = true
	right.TransformTree = tile.TransformTreeResult{Y: tile.TransformSize16x16, UV: tile.TransformSize8x8, HasUV: true}
	filterMap := testFrameWorkLoopFilterPostFilterMap(t, size, left, right)
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			SequenceHeader: seq,
			FrameSize:      size,
			Segmentation: parser.SegmentationParams{
				Enabled: true,
				Data: parser.SegmentationData{
					Segments: [parser.MaxSegments]parser.SegmentData{
						0: {DeltaLFYV: -1},
					},
				},
			},
			LoopFilter: parser.LoopFilterParams{
				LevelY: [2]uint8{1},
				LevelU: 7,
			},
		},
	}
	edges := make([]FrameWorkLoopFilterPostFilterEdge, 1)
	plan, err := ctx.LoopFilterPostFilterPlan(FrameWorkLoopFilterPostFilterRequest{
		Map:   filterMap,
		Edges: edges,
	})
	if err != nil {
		t.Fatal(err)
	}
	if plan.EdgeCandidates != 1 || plan.StoredEdges != 1 || plan.PlaneEdgeCandidates != [3]int{0, 1, 0} {
		t.Fatalf("plan=%+v", plan)
	}
	want := FrameWorkLoopFilterPostFilterEdge{
		Plane:      loopfilter.PlaneU,
		Edge:       loopfilter.EdgeVertical,
		X4:         2,
		Y4:         0,
		Length4:    4,
		Level:      7,
		Transform:  tile.TransformSize8x8,
		Width:      4,
		BlockMICol: 2,
		BlockMIRow: 0,
	}
	if edges[0] != want {
		t.Fatalf("edge=%+v want %+v", edges[0], want)
	}
}

// TestFrameWorkPostFilterContextLoopFilterPostFilterPlanChromaPreviousLookupHonoursORAnchor
// covers the 10-bit quantizer 32 chroma deblock regression: when chroma is
// subsampled (4:2:0) the previous chroma cell's TX size must be read from the
// libaom OR-anchor (mi_row|=scale, mi_col|=scale) — i.e. the bottom-right luma
// MI of the previous chroma cluster — not from the cluster's top-left luma
// MI. The two MIs disagree whenever the previous chroma cluster contains a
// sub8x8 partition: only the bottom-right MI carries the chroma transform
// info, the other three luma MIs report HasChromaBlock=false. With the
// pre-fix top-left lookup the chroma edge was misclassified as a "no-chroma
// gap" and skipped, leaving the 10-bit quantizer 32 vector ~2000 chroma
// samples off after the loop filter. The fixed lookup lands on the
// bottom-right MI and recovers the chroma TX 4x4 -> 4-tap filter expected by
// libaom's set_one_param_for_line_chroma (av1/common/av1_loopfilter.c).
func TestFrameWorkPostFilterContextLoopFilterPostFilterPlanChromaPreviousLookupHonoursORAnchor(t *testing.T) {
	size := parser.FrameSize{
		CodedWidth:          16,
		UpscaledWidth:       16,
		Height:              16,
		SuperResDenominator: 8,
	}
	cols, rows, err := frameWorkLoopFilterMapGrid(size)
	if err != nil {
		t.Fatal(err)
	}
	// Filler block covers the rest of the frame so every MI cell carries a
	// valid record. The vertical chroma edge we exercise sits between the
	// sub8x8 cluster at luma MI (0..1, 0..1) and an 8x8 block at luma MI
	// (2..3, 0..1).
	filler := testFrameWorkLoopFilterPostFilterRecord(cols, rows)
	filler.Block.Size = tile.BlockSize16x16
	filler.SkipTransform = true
	// Sub8x8 cluster: four 4x4 luma blocks at (0,0), (1,0), (0,1), (1,1).
	// Only (1,1) — the bottom-right anchor — carries chroma per libaom's
	// HasChroma condition. We give it a small chroma TX (4x4) so the edge
	// filter width depends on the bottom-right MI lookup.
	subTL := testFrameWorkLoopFilterPostFilterRecordAt(0, 0, 1, 1)
	subTL.SkipTransform = true
	subTL.Block.Size = tile.BlockSize4x4
	subTL.TransformTree = tile.TransformTreeResult{Y: tile.TransformSize4x4, HasUV: false}
	subTR := testFrameWorkLoopFilterPostFilterRecordAt(1, 0, 2, 1)
	subTR.SkipTransform = true
	subTR.Block.Size = tile.BlockSize4x4
	subTR.TransformTree = tile.TransformTreeResult{Y: tile.TransformSize4x4, HasUV: false}
	subBL := testFrameWorkLoopFilterPostFilterRecordAt(0, 1, 1, 2)
	subBL.SkipTransform = true
	subBL.Block.Size = tile.BlockSize4x4
	subBL.TransformTree = tile.TransformTreeResult{Y: tile.TransformSize4x4, HasUV: false}
	// Bottom-right anchor block: carries the chroma TX (4x4) that the right
	// neighbour's previous-side lookup must read via the OR-anchor mapping.
	subBR := testFrameWorkLoopFilterPostFilterRecordAt(1, 1, 2, 2)
	subBR.SkipTransform = true
	subBR.Block.Size = tile.BlockSize4x4
	subBR.TransformTree = tile.TransformTreeResult{Y: tile.TransformSize4x4, UV: tile.TransformSize4x4, HasUV: true}
	// Right neighbour: 8x8 luma at MI (2..3, 0..1), chroma TX 4x4. Its
	// left chroma edge sits at chroma_4x4=1; the previous chroma cell is
	// the cluster (0..1, 0..1).
	right := testFrameWorkLoopFilterPostFilterRecordAt(2, 0, 4, 2)
	right.SkipTransform = true
	right.Block.Size = tile.BlockSize8x8
	right.TransformTree = tile.TransformTreeResult{Y: tile.TransformSize8x8, UV: tile.TransformSize4x4, HasUV: true}
	filterMap := testFrameWorkLoopFilterPostFilterMap(t, size, filler, subTL, subTR, subBL, subBR, right)
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			SequenceHeader: testSequence(),
			FrameSize:      size,
			LoopFilter: parser.LoopFilterParams{
				LevelY: [2]uint8{8, 8},
				LevelU: 7,
				LevelV: 5,
			},
		},
	}
	edges := make([]FrameWorkLoopFilterPostFilterEdge, 64)
	plan, err := ctx.LoopFilterPostFilterPlan(FrameWorkLoopFilterPostFilterRequest{
		Map:   filterMap,
		Edges: edges,
	})
	if err != nil {
		t.Fatal(err)
	}
	// Locate the U/V vertical edges emitted at chroma_4x4=1 (= luma 2),
	// the boundary between the sub8x8 cluster and the 8x8 right block.
	var chromaVerts []FrameWorkLoopFilterPostFilterEdge
	for i := 0; i < plan.StoredEdges; i++ {
		e := edges[i]
		if (e.Plane == loopfilter.PlaneU || e.Plane == loopfilter.PlaneV) &&
			e.Edge == loopfilter.EdgeVertical && e.X4 == 1 && e.BlockMICol == 2 {
			chromaVerts = append(chromaVerts, e)
		}
	}
	if len(chromaVerts) != 2 {
		t.Fatalf("expected 1 U + 1 V vertical chroma edge at the cluster boundary, got %d: %+v", len(chromaVerts), chromaVerts)
	}
	for _, e := range chromaVerts {
		// With the OR-anchor lookup the previous chroma TX is TX_4x4, the
		// current TX is also TX_4x4, so libaom selects the 4-tap chroma
		// filter (filter_length = 4) for both U and V.
		if e.Width != 4 {
			t.Fatalf("chroma edge=%+v want Width=4 (OR-anchor lookup should read TX_4x4 from sub-block (1,1))", e)
		}
	}
}

func TestFrameWorkPostFilterContextLoopFilterPostFilterPlanDowngradesBoundaryWidths(t *testing.T) {
	size := parser.FrameSize{
		CodedWidth:          16,
		UpscaledWidth:       16,
		Height:              16,
		SuperResDenominator: 8,
	}
	cols, rows, err := frameWorkLoopFilterMapGrid(size)
	if err != nil {
		t.Fatal(err)
	}
	filler := testFrameWorkLoopFilterPostFilterRecord(cols, rows)
	filler.SkipTransform = true
	nearLeft := testFrameWorkLoopFilterPostFilterRecordAt(1, 0, 4, 4)
	nearLeft.SkipTransform = true
	nearLeft.TransformTree = tile.TransformTreeResult{Y: tile.TransformSize16x16}
	filterMap := testFrameWorkLoopFilterPostFilterMap(t, size, filler, nearLeft)
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			SequenceHeader: testSequence(),
			FrameSize:      size,
			LoopFilter:     parser.LoopFilterParams{LevelY: [2]uint8{8}},
		},
	}
	edges := make([]FrameWorkLoopFilterPostFilterEdge, 1)
	plan, err := ctx.LoopFilterPostFilterPlan(FrameWorkLoopFilterPostFilterRequest{
		Map:   filterMap,
		Edges: edges,
	})
	if err != nil {
		t.Fatal(err)
	}
	if plan.StoredEdges != 1 || edges[0].Width != 8 {
		t.Fatalf("plan=%+v edge=%+v", plan, edges[0])
	}
}

func TestFrameWorkPostFilterContextLoopFilterPostFilterPlanCountsDroppedEdges(t *testing.T) {
	size := parser.FrameSize{
		CodedWidth:          32,
		UpscaledWidth:       32,
		Height:              32,
		SuperResDenominator: 8,
	}
	filterMap := testFrameWorkLoopFilterPostFilterMap(t, size,
		testFrameWorkLoopFilterPostFilterRecordAt(0, 0, 4, 4),
		testFrameWorkLoopFilterPostFilterRecordAt(4, 0, 8, 4),
		testFrameWorkLoopFilterPostFilterRecordAt(0, 4, 4, 8),
		testFrameWorkLoopFilterPostFilterRecordAt(4, 4, 8, 8),
	)
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			SequenceHeader: testSequence(),
			FrameSize:      size,
			LoopFilter:     parser.LoopFilterParams{LevelY: [2]uint8{8, 9}},
		},
	}

	plan, err := ctx.LoopFilterPostFilterPlan(FrameWorkLoopFilterPostFilterRequest{
		Map:   filterMap,
		Edges: make([]FrameWorkLoopFilterPostFilterEdge, 1),
	})
	if err != nil {
		t.Fatal(err)
	}
	if plan.EdgeCandidates != 4 || plan.StoredEdges != 1 || plan.DroppedEdges != 3 {
		t.Fatalf("plan=%+v", plan)
	}
}

func TestFrameWorkPostFilterContextLoopFilterPostFilterPlanUsesPreviousLumaLevelVertical(t *testing.T) {
	size := parser.FrameSize{
		CodedWidth:          32,
		UpscaledWidth:       32,
		Height:              16,
		SuperResDenominator: 8,
	}
	left := testFrameWorkLoopFilterPostFilterRecordAt(0, 0, 4, 4)
	left.SkipTransform = true
	right := testFrameWorkLoopFilterPostFilterRecordAt(4, 0, 8, 4)
	right.SkipTransform = true
	right.SegmentID = 1
	filterMap := testFrameWorkLoopFilterPostFilterMap(t, size, left, right)
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			SequenceHeader: testSequence(),
			FrameSize:      size,
			Segmentation: parser.SegmentationParams{
				Enabled: true,
				Data: parser.SegmentationData{
					Segments: [parser.MaxSegments]parser.SegmentData{
						1: {DeltaLFYV: -8},
					},
				},
			},
			LoopFilter: parser.LoopFilterParams{LevelY: [2]uint8{8}},
		},
	}
	edges := make([]FrameWorkLoopFilterPostFilterEdge, 1)

	plan, err := ctx.LoopFilterPostFilterPlan(FrameWorkLoopFilterPostFilterRequest{
		Map:   filterMap,
		Edges: edges,
	})
	if err != nil {
		t.Fatal(err)
	}
	if plan.EdgeCandidates != 1 || plan.StoredEdges != 1 || plan.PreviousLevelEdges != 1 {
		t.Fatalf("plan=%+v", plan)
	}
	want := FrameWorkLoopFilterPostFilterEdge{
		Plane:             loopfilter.PlaneY,
		Edge:              loopfilter.EdgeVertical,
		X4:                4,
		Y4:                0,
		Length4:           4,
		Level:             8,
		Transform:         tile.TransformSize16x16,
		Width:             14,
		LevelFromPrevious: true,
		BlockMICol:        4,
		BlockMIRow:        0,
	}
	if edges[0] != want {
		t.Fatalf("edge=%+v want %+v", edges[0], want)
	}
}

func TestFrameWorkPostFilterContextLoopFilterPostFilterPlanUsesPreviousLumaLevelHorizontal(t *testing.T) {
	size := parser.FrameSize{
		CodedWidth:          16,
		UpscaledWidth:       16,
		Height:              32,
		SuperResDenominator: 8,
	}
	top := testFrameWorkLoopFilterPostFilterRecordAt(0, 0, 4, 4)
	top.SkipTransform = true
	bottom := testFrameWorkLoopFilterPostFilterRecordAt(0, 4, 4, 8)
	bottom.SkipTransform = true
	bottom.SegmentID = 1
	filterMap := testFrameWorkLoopFilterPostFilterMap(t, size, top, bottom)
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			SequenceHeader: testSequence(),
			FrameSize:      size,
			Segmentation: parser.SegmentationParams{
				Enabled: true,
				Data: parser.SegmentationData{
					Segments: [parser.MaxSegments]parser.SegmentData{
						1: {DeltaLFYH: -9},
					},
				},
			},
			LoopFilter: parser.LoopFilterParams{LevelY: [2]uint8{0, 9}},
		},
	}
	edges := make([]FrameWorkLoopFilterPostFilterEdge, 1)

	plan, err := ctx.LoopFilterPostFilterPlan(FrameWorkLoopFilterPostFilterRequest{
		Map:   filterMap,
		Edges: edges,
	})
	if err != nil {
		t.Fatal(err)
	}
	if plan.EdgeCandidates != 1 || plan.StoredEdges != 1 || plan.PreviousLevelEdges != 1 {
		t.Fatalf("plan=%+v", plan)
	}
	want := FrameWorkLoopFilterPostFilterEdge{
		Plane:             loopfilter.PlaneY,
		Edge:              loopfilter.EdgeHorizontal,
		X4:                0,
		Y4:                4,
		Length4:           4,
		Level:             9,
		Transform:         tile.TransformSize16x16,
		Width:             14,
		LevelFromPrevious: true,
		BlockMICol:        0,
		BlockMIRow:        4,
	}
	if edges[0] != want {
		t.Fatalf("edge=%+v want %+v", edges[0], want)
	}
}

func TestFrameWorkPostFilterContextLoopFilterPostFilterPlanUsesPreviousChromaLevelVertical(t *testing.T) {
	size := parser.FrameSize{
		CodedWidth:          32,
		UpscaledWidth:       32,
		Height:              16,
		SuperResDenominator: 8,
	}
	left := testFrameWorkLoopFilterPostFilterRecordAt(0, 0, 4, 4)
	left.SkipTransform = true
	right := testFrameWorkLoopFilterPostFilterRecordAt(4, 0, 8, 4)
	right.SkipTransform = true
	right.SegmentID = 1
	filterMap := testFrameWorkLoopFilterPostFilterMap(t, size, left, right)
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			SequenceHeader: testSequence(),
			FrameSize:      size,
			Segmentation: parser.SegmentationParams{
				Enabled: true,
				Data: parser.SegmentationData{
					Segments: [parser.MaxSegments]parser.SegmentData{
						0: {DeltaLFYV: -1},
						1: {DeltaLFYV: -1, DeltaLFU: -7},
					},
				},
			},
			LoopFilter: parser.LoopFilterParams{
				LevelY: [2]uint8{1},
				LevelU: 7,
			},
		},
	}
	edges := make([]FrameWorkLoopFilterPostFilterEdge, 1)

	plan, err := ctx.LoopFilterPostFilterPlan(FrameWorkLoopFilterPostFilterRequest{
		Map:   filterMap,
		Edges: edges,
	})
	if err != nil {
		t.Fatal(err)
	}
	if plan.EdgeCandidates != 1 || plan.StoredEdges != 1 || plan.PreviousLevelEdges != 1 || plan.PlaneEdgeCandidates != [3]int{0, 1, 0} {
		t.Fatalf("plan=%+v", plan)
	}
	want := FrameWorkLoopFilterPostFilterEdge{
		Plane:             loopfilter.PlaneU,
		Edge:              loopfilter.EdgeVertical,
		X4:                2,
		Y4:                0,
		Length4:           2,
		Level:             7,
		Transform:         tile.TransformSize8x8,
		Width:             6,
		LevelFromPrevious: true,
		BlockMICol:        4,
		BlockMIRow:        0,
	}
	if edges[0] != want {
		t.Fatalf("edge=%+v want %+v", edges[0], want)
	}
}

func TestFrameWorkPostFilterContextLoopFilterPostFilterPlanUsesPreviousChromaLevelHorizontal(t *testing.T) {
	size := parser.FrameSize{
		CodedWidth:          16,
		UpscaledWidth:       16,
		Height:              32,
		SuperResDenominator: 8,
	}
	top := testFrameWorkLoopFilterPostFilterRecordAt(0, 0, 4, 4)
	top.SkipTransform = true
	bottom := testFrameWorkLoopFilterPostFilterRecordAt(0, 4, 4, 8)
	bottom.SkipTransform = true
	bottom.SegmentID = 1
	filterMap := testFrameWorkLoopFilterPostFilterMap(t, size, top, bottom)
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			SequenceHeader: testSequence(),
			FrameSize:      size,
			Segmentation: parser.SegmentationParams{
				Enabled: true,
				Data: parser.SegmentationData{
					Segments: [parser.MaxSegments]parser.SegmentData{
						0: {DeltaLFYH: -1},
						1: {DeltaLFYH: -1, DeltaLFV: -9},
					},
				},
			},
			LoopFilter: parser.LoopFilterParams{
				LevelY: [2]uint8{0, 1},
				LevelV: 9,
			},
		},
	}
	edges := make([]FrameWorkLoopFilterPostFilterEdge, 1)

	plan, err := ctx.LoopFilterPostFilterPlan(FrameWorkLoopFilterPostFilterRequest{
		Map:   filterMap,
		Edges: edges,
	})
	if err != nil {
		t.Fatal(err)
	}
	if plan.EdgeCandidates != 1 || plan.StoredEdges != 1 || plan.PreviousLevelEdges != 1 || plan.PlaneEdgeCandidates != [3]int{0, 0, 1} {
		t.Fatalf("plan=%+v", plan)
	}
	want := FrameWorkLoopFilterPostFilterEdge{
		Plane:             loopfilter.PlaneV,
		Edge:              loopfilter.EdgeHorizontal,
		X4:                0,
		Y4:                2,
		Length4:           2,
		Level:             9,
		Transform:         tile.TransformSize8x8,
		Width:             6,
		LevelFromPrevious: true,
		BlockMICol:        0,
		BlockMIRow:        4,
	}
	if edges[0] != want {
		t.Fatalf("edge=%+v want %+v", edges[0], want)
	}
}

func TestFrameWorkPostFilterContextApplyLoopFilterLumaEdges(t *testing.T) {
	size := parser.FrameSize{
		CodedWidth:          32,
		UpscaledWidth:       32,
		Height:              16,
		SuperResDenominator: 8,
	}
	first := testFrameWorkLoopFilterPostFilterRecordAt(0, 0, 4, 4)
	first.SkipTransform = true
	second := testFrameWorkLoopFilterPostFilterRecordAt(4, 0, 8, 4)
	second.TransformTree = tile.TransformTreeResult{
		Y:        tile.TransformSize16x16,
		Variable: true,
		Split:    [2]uint16{1, 0},
	}
	filterMap := testFrameWorkLoopFilterPostFilterMap(t, size, first, second)
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: 32, Height: 16, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32})
	testFillFrameWorkLoopFilterPattern(output.Y)
	before := append([]byte(nil), output.Y.Pix...)
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			SequenceHeader: testSequence(),
			FrameSize:      size,
			LoopFilter:     parser.LoopFilterParams{LevelY: [2]uint8{63}, Sharpness: 1},
		},
		Output: output,
	}

	edges := make([]FrameWorkLoopFilterPostFilterEdge, 4)
	wantY := append([]byte(nil), output.Y.Pix...)
	wantPlane := output.Y
	wantPlane.Pix = wantY
	plan, err := ctx.LoopFilterPostFilterPlan(FrameWorkLoopFilterPostFilterRequest{Map: filterMap, Edges: edges})
	if err != nil {
		t.Fatal(err)
	}
	for _, edge := range edges[:plan.StoredEdges] {
		thresholds, err := loopfilter.ThresholdsForLevel(edge.Level, ctx.Event.LoopFilter.Sharpness)
		if err != nil {
			t.Fatal(err)
		}
		if err := loopfilter.FilterEdgeByWidth(edge.Width, wantPlane, output.Layout.BytesPerSample, output.Format.BitDepth, edge.Edge, edge.X4*4, edge.Y4*4, edge.Length4*4, thresholds); err != nil {
			t.Fatal(err)
		}
	}

	result, err := ctx.ApplyLoopFilterLumaEdges(FrameWorkLoopFilterPostFilterRequest{
		Map:   filterMap,
		Edges: edges,
	})
	if err != nil {
		t.Fatal(err)
	}
	if !result.Active || result.Edges != 4 || result.Applied != 4 || result.MaxLevel != 63 || result.Plan.StoredEdges != 4 {
		t.Fatalf("result=%+v", result)
	}
	if bytes.Equal(output.Y.Pix, before) {
		t.Fatal("loop filter luma edge bridge did not change output")
	}
	if !bytes.Equal(output.Y.Pix, wantY) {
		t.Fatal("loop filter luma edge bridge output did not match direct edge application")
	}
}

func TestFrameWorkPostFilterContextApplyLoopFilterEdgesLumaAndChroma(t *testing.T) {
	size := parser.FrameSize{
		CodedWidth:          32,
		UpscaledWidth:       32,
		Height:              16,
		SuperResDenominator: 8,
	}
	first := testFrameWorkLoopFilterPostFilterRecordAt(0, 0, 4, 4)
	second := testFrameWorkLoopFilterPostFilterRecordAt(4, 0, 8, 4)
	filterMap := testFrameWorkLoopFilterPostFilterMap(t, size, first, second)
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: 32, Height: 16, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32})
	testFillFrameWorkLoopFilterPattern(output.Y)
	testFillFrameWorkLoopFilterPattern(output.U)
	testFillFrameWorkLoopFilterPattern(output.V)
	beforeY := append([]byte(nil), output.Y.Pix...)
	beforeU := append([]byte(nil), output.U.Pix...)
	beforeV := append([]byte(nil), output.V.Pix...)
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			SequenceHeader: testSequence(),
			FrameSize:      size,
			LoopFilter: parser.LoopFilterParams{
				LevelY:    [2]uint8{63},
				LevelU:    63,
				LevelV:    63,
				Sharpness: 1,
			},
		},
		Output: output,
	}

	edges := make([]FrameWorkLoopFilterPostFilterEdge, 3)
	wantY := append([]byte(nil), output.Y.Pix...)
	wantU := append([]byte(nil), output.U.Pix...)
	wantV := append([]byte(nil), output.V.Pix...)
	wantFrame := *output
	wantFrame.Y.Pix = wantY
	wantFrame.U.Pix = wantU
	wantFrame.V.Pix = wantV
	plan, err := ctx.LoopFilterPostFilterPlan(FrameWorkLoopFilterPostFilterRequest{Map: filterMap, Edges: edges})
	if err != nil {
		t.Fatal(err)
	}
	testApplyFrameWorkLoopFilterEdgesDirect(t, wantFrame, ctx.Event.LoopFilter.Sharpness, edges[:plan.StoredEdges])

	result, err := ctx.ApplyLoopFilterEdges(FrameWorkLoopFilterPostFilterRequest{
		Map:   filterMap,
		Edges: edges,
	})
	if err != nil {
		t.Fatal(err)
	}
	if !result.Active ||
		result.Edges != 3 ||
		result.Applied != 3 ||
		result.MaxLevel != 63 ||
		result.PlaneEdges != [3]int{1, 1, 1} ||
		result.PlaneApplied != [3]int{1, 1, 1} ||
		result.PlaneMaxLevel != [3]uint8{63, 63, 63} ||
		result.Plan.StoredEdges != 3 {
		t.Fatalf("result=%+v", result)
	}
	if bytes.Equal(output.Y.Pix, beforeY) || bytes.Equal(output.U.Pix, beforeU) || bytes.Equal(output.V.Pix, beforeV) {
		t.Fatal("loop filter all-plane bridge did not change every active plane")
	}
	if !bytes.Equal(output.Y.Pix, wantY) || !bytes.Equal(output.U.Pix, wantU) || !bytes.Equal(output.V.Pix, wantV) {
		t.Fatal("loop filter all-plane bridge output did not match direct edge application")
	}
}

func TestFrameWorkPostFilterContextApplyLoopFilterEdgesHighBitDepthChroma(t *testing.T) {
	size := parser.FrameSize{
		CodedWidth:          32,
		UpscaledWidth:       32,
		Height:              16,
		SuperResDenominator: 8,
	}
	seq := testSequence()
	seq.ColorConfig.BitDepth = 10
	first := testFrameWorkLoopFilterPostFilterRecordAt(0, 0, 4, 4)
	second := testFrameWorkLoopFilterPostFilterRecordAt(4, 0, 8, 4)
	filterMap := testFrameWorkLoopFilterPostFilterMap(t, size, first, second)
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: 32, Height: 16, BitDepth: 10, SubsamplingX: true, SubsamplingY: true, Align: 64})
	testFillFrameWorkLoopFilterSamplePattern(output.Y, output.Layout.BytesPerSample)
	testFillFrameWorkLoopFilterSamplePattern(output.U, output.Layout.BytesPerSample)
	testFillFrameWorkLoopFilterSamplePattern(output.V, output.Layout.BytesPerSample)
	beforeU := append([]byte(nil), output.U.Pix...)
	beforeV := append([]byte(nil), output.V.Pix...)
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			SequenceHeader: seq,
			FrameSize:      size,
			LoopFilter: parser.LoopFilterParams{
				LevelY:    [2]uint8{63},
				LevelU:    63,
				LevelV:    63,
				Sharpness: 1,
			},
		},
		Output: output,
	}

	edges := make([]FrameWorkLoopFilterPostFilterEdge, 3)
	wantFrame := testCloneFrameWorkLoopFilterFrame(output)
	plan, err := ctx.LoopFilterPostFilterPlan(FrameWorkLoopFilterPostFilterRequest{Map: filterMap, Edges: edges})
	if err != nil {
		t.Fatal(err)
	}
	testApplyFrameWorkLoopFilterEdgesDirect(t, wantFrame, ctx.Event.LoopFilter.Sharpness, edges[:plan.StoredEdges])

	result, err := ctx.ApplyLoopFilterEdges(FrameWorkLoopFilterPostFilterRequest{
		Map:   filterMap,
		Edges: edges,
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.PlaneApplied != [3]int{1, 1, 1} || result.MaxLevel != 63 {
		t.Fatalf("result=%+v", result)
	}
	if bytes.Equal(output.U.Pix, beforeU) || bytes.Equal(output.V.Pix, beforeV) {
		t.Fatal("high-bit-depth chroma loop filter did not change U and V")
	}
	if !bytes.Equal(output.Y.Pix, wantFrame.Y.Pix) ||
		!bytes.Equal(output.U.Pix, wantFrame.U.Pix) ||
		!bytes.Equal(output.V.Pix, wantFrame.V.Pix) {
		t.Fatal("high-bit-depth loop filter output did not match direct edge application")
	}
}

func TestFrameWorkPostFilterContextApplyLoopFilterEdgesUsesPlanePassOrder(t *testing.T) {
	size := parser.FrameSize{
		CodedWidth:          32,
		UpscaledWidth:       32,
		Height:              32,
		SuperResDenominator: 8,
	}
	filterMap := testFrameWorkLoopFilterPostFilterMap(t, size,
		testFrameWorkLoopFilterPostFilterRecordAt(0, 0, 4, 4),
		testFrameWorkLoopFilterPostFilterRecordAt(4, 0, 8, 4),
		testFrameWorkLoopFilterPostFilterRecordAt(0, 4, 4, 8),
		testFrameWorkLoopFilterPostFilterRecordAt(4, 4, 8, 8),
	)
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: 32, Height: 32, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32})
	rng := rand.New(rand.NewSource(1))
	testFillFrameWorkLoopFilterOrderPattern(output.Y, rng)
	testFillFrameWorkLoopFilterOrderPattern(output.U, rng)
	testFillFrameWorkLoopFilterOrderPattern(output.V, rng)
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			SequenceHeader: testSequence(),
			FrameSize:      size,
			LoopFilter: parser.LoopFilterParams{
				LevelY: [2]uint8{63, 63},
				LevelU: 63,
				LevelV: 63,
			},
		},
		Output: output,
	}

	edges := make([]FrameWorkLoopFilterPostFilterEdge, 12)
	wantFrame := testCloneFrameWorkLoopFilterFrame(output)
	storedOrderFrame := testCloneFrameWorkLoopFilterFrame(output)
	plan, err := ctx.LoopFilterPostFilterPlan(FrameWorkLoopFilterPostFilterRequest{Map: filterMap, Edges: edges})
	if err != nil {
		t.Fatal(err)
	}
	if plan.StoredEdges != len(edges) || plan.PlaneEdgeCandidates != [3]int{4, 4, 4} {
		t.Fatalf("plan=%+v", plan)
	}
	testApplyFrameWorkLoopFilterEdgesDirect(t, wantFrame, ctx.Event.LoopFilter.Sharpness, edges[:plan.StoredEdges])
	testApplyFrameWorkLoopFilterEdgesStoredOrder(t, storedOrderFrame, ctx.Event.LoopFilter.Sharpness, edges[:plan.StoredEdges])
	if bytes.Equal(storedOrderFrame.Y.Pix, wantFrame.Y.Pix) &&
		bytes.Equal(storedOrderFrame.U.Pix, wantFrame.U.Pix) &&
		bytes.Equal(storedOrderFrame.V.Pix, wantFrame.V.Pix) {
		t.Fatal("test fixture did not distinguish stored order from plane/pass order")
	}

	result, err := ctx.ApplyLoopFilterEdges(FrameWorkLoopFilterPostFilterRequest{
		Map:   filterMap,
		Edges: edges,
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Edges != 12 ||
		result.PlaneEdges != [3]int{4, 4, 4} ||
		result.PlaneApplied != [3]int{4, 4, 4} {
		t.Fatalf("result=%+v", result)
	}
	if !bytes.Equal(output.Y.Pix, wantFrame.Y.Pix) ||
		!bytes.Equal(output.U.Pix, wantFrame.U.Pix) ||
		!bytes.Equal(output.V.Pix, wantFrame.V.Pix) {
		t.Fatal("loop filter all-plane bridge did not use plane/pass order")
	}
}

func TestFrameWorkPostFilterContextApplyLoopFilterEdgesAllocs(t *testing.T) {
	size := parser.FrameSize{
		CodedWidth:          32,
		UpscaledWidth:       32,
		Height:              16,
		SuperResDenominator: 8,
	}
	first := testFrameWorkLoopFilterPostFilterRecordAt(0, 0, 4, 4)
	second := testFrameWorkLoopFilterPostFilterRecordAt(4, 0, 8, 4)
	filterMap := testFrameWorkLoopFilterPostFilterMap(t, size, first, second)
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: 32, Height: 16, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32})
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			SequenceHeader: testSequence(),
			FrameSize:      size,
			LoopFilter: parser.LoopFilterParams{
				LevelY:    [2]uint8{63},
				LevelU:    63,
				LevelV:    63,
				Sharpness: 1,
			},
		},
		Output: output,
	}
	edges := make([]FrameWorkLoopFilterPostFilterEdge, 3)
	allocs := testing.AllocsPerRun(1000, func() {
		testFillFrameWorkLoopFilterPattern(output.Y)
		testFillFrameWorkLoopFilterPattern(output.U)
		testFillFrameWorkLoopFilterPattern(output.V)
		result, err := ctx.ApplyLoopFilterEdges(FrameWorkLoopFilterPostFilterRequest{
			Map:   filterMap,
			Edges: edges,
		})
		if err != nil {
			t.Fatal(err)
		}
		if result.Applied != 3 || result.PlaneApplied != [3]int{1, 1, 1} {
			t.Fatalf("result=%+v", result)
		}
	})
	if allocs != 0 {
		t.Fatalf("ApplyLoopFilterEdges allocated: %f", allocs)
	}
}

func TestFrameWorkPostFilterContextApplySupportedPostFiltersRunsLoopFilterThenCDEF(t *testing.T) {
	const width = 32
	const height = 16

	seq := testSequence()
	seq.EnableCDEF = true
	size := parser.FrameSize{
		CodedWidth:          width,
		UpscaledWidth:       width,
		Height:              height,
		SuperResDenominator: 8,
	}
	first := testFrameWorkLoopFilterPostFilterRecordAt(0, 0, 4, 4)
	second := testFrameWorkLoopFilterPostFilterRecordAt(4, 0, 8, 4)
	filterMap := testFrameWorkLoopFilterPostFilterMap(t, size, first, second)
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: width, Height: height, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32})
	testFillFrameWorkLoopFilterPattern(output.Y)
	before := append([]byte(nil), output.Y.Pix...)
	event := Event{
		SequenceHeader: seq,
		FrameSize:      size,
		LoopFilter: parser.LoopFilterParams{
			LevelY:    [2]uint8{63},
			Sharpness: 1,
		},
		CDEF: parser.CDEFParams{
			Damping:       5,
			StrengthCount: 1,
			YStrength:     [parser.MaxCDEFStrengths]uint8{63},
		},
	}
	ctx := FrameWorkPostFilterContext{Event: event, Output: output, LoopFilterMap: &filterMap}
	cdefReq := testFrameWorkCDEFPostFilterRequest(t, ctx, event)
	scratchSize, err := ctx.SupportedPostFilterScratchLen(FrameWorkPostFilterRequest{CDEF: cdefReq})
	if err != nil {
		t.Fatal(err)
	}
	if scratchSize.LoopFilter.Edges < 1 || scratchSize.CDEF.Input != cdef.InputBufferSize {
		t.Fatalf("scratch=%+v", scratchSize)
	}

	next, result, err := ctx.ApplySupportedPostFilters(FrameWorkPostFilterRequest{
		LoopFilter: FrameWorkLoopFilterPostFilterRequest{
			Edges: make([]FrameWorkLoopFilterPostFilterEdge, scratchSize.LoopFilter.Edges),
		},
		CDEF: cdefReq,
	})
	if err != nil {
		t.Fatal(err)
	}
	wantCompleted := FrameWorkPostFilterLoopFilter | FrameWorkPostFilterCDEF
	if result.Completed != wantCompleted ||
		result.LoopFilter.Applied != 1 ||
		result.CDEF.Units != 1 ||
		next.RemainingPostFilters() != 0 {
		t.Fatalf("next remaining=%b result=%+v", next.RemainingPostFilters(), result)
	}
	if bytes.Equal(output.Y.Pix, before) {
		t.Fatal("supported postfilter pipeline did not change luma samples")
	}
}

func TestFrameWorkBoundSupportedPostFilterRunnerDoesNotRetainDefaultSideMaps(t *testing.T) {
	const width = 16
	const height = 16

	size := parser.FrameSize{
		CodedWidth:          width,
		UpscaledWidth:       width,
		Height:              height,
		SuperResDenominator: 8,
	}
	cols, rows, err := frameWorkLoopFilterMapGrid(size)
	if err != nil {
		t.Fatal(err)
	}
	filterMap := testFrameWorkLoopFilterPostFilterMap(t, size, testFrameWorkLoopFilterPostFilterRecord(cols, rows))
	event := Event{
		SequenceHeader: testSequence(),
		FrameSize:      size,
		LoopFilter: parser.LoopFilterParams{
			LevelY: [2]uint8{4},
		},
	}
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: width, Height: height, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32})
	runner := FrameWorkBoundSupportedPostFilterRunner{
		Scratch: FrameWorkPostFilterScratch{
			LoopFilterEdges: make([]FrameWorkLoopFilterPostFilterEdge, 7),
		},
	}
	if err := runner.Apply(FrameWorkPostFilterContext{Event: event, Output: output, LoopFilterMap: &filterMap}); err != nil {
		t.Fatal(err)
	}
	if runner.Result.Completed != FrameWorkPostFilterLoopFilter || runner.Context.RemainingPostFilters() != 0 {
		t.Fatalf("result=%+v remaining=%b", runner.Result, runner.Context.RemainingPostFilters())
	}
	if !frameWorkLoopFilterMapEmpty(runner.Options.LoopFilterMap) {
		t.Fatalf("runner retained defaulted loop-filter map: %+v", runner.Options.LoopFilterMap)
	}
	if runner.Size.LoopFilter.Edges != len(runner.Scratch.LoopFilterEdges) {
		t.Fatalf("bound runner loop-filter size=%+v want owned edge capacity %d", runner.Size.LoopFilter, len(runner.Scratch.LoopFilterEdges))
	}

	err = runner.Apply(FrameWorkPostFilterContext{Event: event, Output: output})
	if !errors.Is(err, threading.ErrInvalidBatch) {
		t.Fatalf("reused runner Apply err=%v want %v", err, threading.ErrInvalidBatch)
	}
}

func TestFrameWorkPostFilterContextApplyPreSuperResPostFiltersRunsLoopFilterThenCDEF(t *testing.T) {
	const width = 32
	const height = 16
	const upscaledWidth = 48

	seq := testSequence()
	seq.EnableCDEF = true
	size := parser.FrameSize{
		CodedWidth:          width,
		UpscaledWidth:       upscaledWidth,
		Height:              height,
		SuperResEnabled:     true,
		SuperResDenominator: 16,
	}
	first := testFrameWorkLoopFilterPostFilterRecordAt(0, 0, 4, 4)
	second := testFrameWorkLoopFilterPostFilterRecordAt(4, 0, 8, 4)
	filterMap := testFrameWorkLoopFilterPostFilterMap(t, size, first, second)
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: width, Height: height, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32})
	testFillFrameWorkLoopFilterPattern(output.Y)
	before := append([]byte(nil), output.Y.Pix...)
	event := Event{
		SequenceHeader: seq,
		FrameSize:      size,
		LoopFilter: parser.LoopFilterParams{
			LevelY:    [2]uint8{63},
			Sharpness: 1,
		},
		CDEF: parser.CDEFParams{
			Damping:       5,
			StrengthCount: 1,
			YStrength:     [parser.MaxCDEFStrengths]uint8{63},
		},
	}
	ctx := FrameWorkPostFilterContext{Event: event, Output: output, LoopFilterMap: &filterMap}
	cdefReq := testFrameWorkCDEFPostFilterRequest(t, ctx, event)
	scratchSize, err := ctx.PreSuperResPostFilterScratchLen(FrameWorkPostFilterRequest{CDEF: cdefReq})
	if err != nil {
		t.Fatal(err)
	}
	if scratchSize.LoopFilter.Edges < 1 || scratchSize.CDEF.Input != cdef.InputBufferSize || scratchSize.Restoration != (FrameWorkRestorationPostFilterScratchSize{}) {
		t.Fatalf("scratch=%+v", scratchSize)
	}

	next, result, err := ctx.ApplyPreSuperResPostFilters(FrameWorkPostFilterRequest{
		LoopFilter: FrameWorkLoopFilterPostFilterRequest{
			Edges: make([]FrameWorkLoopFilterPostFilterEdge, scratchSize.LoopFilter.Edges),
		},
		CDEF: cdefReq,
	})
	if err != nil {
		t.Fatal(err)
	}
	wantCompleted := FrameWorkPostFilterLoopFilter | FrameWorkPostFilterCDEF
	if result.Completed != wantCompleted ||
		result.LoopFilter.Applied != 1 ||
		result.CDEF.Units != 1 ||
		next.RemainingPostFilters() != FrameWorkPostFilterSuperRes {
		t.Fatalf("next remaining=%b result=%+v", next.RemainingPostFilters(), result)
	}
	if bytes.Equal(output.Y.Pix, before) {
		t.Fatal("pre-superres postfilter prefix did not change luma samples")
	}

	superResReq := testFrameWorkSuperResPostFilterRequest(t, next)
	next, superResResult, err := next.ApplySuperResPostFilterToContext(superResReq)
	if err != nil {
		t.Fatal(err)
	}
	if superResResult.Output.Format.Width != upscaledWidth || next.RemainingPostFilters() != 0 {
		t.Fatalf("superres output=%+v remaining=%b", superResResult.Output.Format, next.RemainingPostFilters())
	}
}

func TestFrameWorkPostFilterContextApplyCallerPostFiltersRunsPreSuperResAndSuperRes(t *testing.T) {
	const width = 32
	const height = 16
	const upscaledWidth = 48

	seq := testSequence()
	seq.EnableCDEF = true
	size := parser.FrameSize{
		CodedWidth:          width,
		UpscaledWidth:       upscaledWidth,
		Height:              height,
		SuperResEnabled:     true,
		SuperResDenominator: 16,
	}
	first := testFrameWorkLoopFilterPostFilterRecordAt(0, 0, 4, 4)
	second := testFrameWorkLoopFilterPostFilterRecordAt(4, 0, 8, 4)
	filterMap := testFrameWorkLoopFilterPostFilterMap(t, size, first, second)
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: width, Height: height, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32})
	testFillFrameWorkLoopFilterPattern(output.Y)
	before := append([]byte(nil), output.Y.Pix...)
	event := Event{
		SequenceHeader: seq,
		FrameSize:      size,
		LoopFilter: parser.LoopFilterParams{
			LevelY:    [2]uint8{63},
			Sharpness: 1,
		},
		CDEF: parser.CDEFParams{
			Damping:       5,
			StrengthCount: 1,
			YStrength:     [parser.MaxCDEFStrengths]uint8{63},
		},
	}
	ctx := FrameWorkPostFilterContext{Event: event, Output: output, LoopFilterMap: &filterMap}
	cdefReq := testFrameWorkCDEFPostFilterRequest(t, ctx, event)
	req := FrameWorkPostFilterRequest{CDEF: cdefReq}
	sizeReq, err := ctx.CallerPostFilterScratchLen(req)
	if err != nil {
		t.Fatal(err)
	}
	if sizeReq.LoopFilter.Edges < 1 || sizeReq.CDEF.Input != cdef.InputBufferSize || sizeReq.SuperRes.OutputFrame == 0 {
		t.Fatalf("scratch=%+v", sizeReq)
	}
	req.LoopFilter.Edges = make([]FrameWorkLoopFilterPostFilterEdge, sizeReq.LoopFilter.Edges)
	req.SuperRes = testFrameWorkSuperResPostFilterRequest(t, ctx)

	next, result, err := ctx.ApplyCallerPostFilters(req)
	if err != nil {
		t.Fatal(err)
	}
	wantCompleted := FrameWorkPostFilterLoopFilter | FrameWorkPostFilterCDEF | FrameWorkPostFilterSuperRes
	if result.Completed != wantCompleted ||
		result.LoopFilter.Applied != 1 ||
		result.CDEF.Units != 1 ||
		result.SuperRes.Output.Format.Width != upscaledWidth ||
		next.RemainingPostFilters() != 0 ||
		!next.DetachedPostFilterOutput() {
		t.Fatalf("next remaining=%b detached=%v result=%+v", next.RemainingPostFilters(), next.DetachedPostFilterOutput(), result)
	}
	if bytes.Equal(output.Y.Pix, before) {
		t.Fatal("caller postfilter pipeline did not change coded luma samples before superres")
	}
	if err := next.RequirePublishablePostFilterOutput(); !errors.Is(err, ErrUnsupportedPostFilter) {
		t.Fatalf("RequirePublishablePostFilterOutput err=%v want %v", err, ErrUnsupportedPostFilter)
	}
}

func TestFrameWorkPostFilterContextApplyPreSuperResPostFiltersRejectsShortCDEFScratchBeforeLoopFilterMutation(t *testing.T) {
	const width = 32
	const height = 16

	seq := testSequence()
	seq.EnableCDEF = true
	size := parser.FrameSize{
		CodedWidth:          width,
		UpscaledWidth:       48,
		Height:              height,
		SuperResEnabled:     true,
		SuperResDenominator: 16,
	}
	first := testFrameWorkLoopFilterPostFilterRecordAt(0, 0, 4, 4)
	second := testFrameWorkLoopFilterPostFilterRecordAt(4, 0, 8, 4)
	filterMap := testFrameWorkLoopFilterPostFilterMap(t, size, first, second)
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: width, Height: height, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32})
	testFillFrameWorkLoopFilterPattern(output.Y)
	beforeY := append([]byte(nil), output.Y.Pix...)
	event := Event{
		SequenceHeader: seq,
		FrameSize:      size,
		LoopFilter: parser.LoopFilterParams{
			LevelY:    [2]uint8{63},
			Sharpness: 1,
		},
		CDEF: parser.CDEFParams{
			Damping:       5,
			StrengthCount: 1,
			YStrength:     [parser.MaxCDEFStrengths]uint8{63},
		},
	}
	ctx := FrameWorkPostFilterContext{Event: event, Output: output, LoopFilterMap: &filterMap}
	cdefReq := testFrameWorkCDEFPostFilterRequest(t, ctx, event)
	scratchSize, err := ctx.PreSuperResPostFilterScratchLen(FrameWorkPostFilterRequest{CDEF: cdefReq})
	if err != nil {
		t.Fatal(err)
	}
	cdefReq.InputScratch = cdefReq.InputScratch[:len(cdefReq.InputScratch)-1]

	next, result, err := ctx.ApplyPreSuperResPostFilters(FrameWorkPostFilterRequest{
		LoopFilter: FrameWorkLoopFilterPostFilterRequest{
			Edges: make([]FrameWorkLoopFilterPostFilterEdge, scratchSize.LoopFilter.Edges),
		},
		CDEF: cdefReq,
	})
	if !errors.Is(err, frame.ErrShortBuffer) {
		t.Fatalf("ApplyPreSuperResPostFilters err=%v want %v", err, frame.ErrShortBuffer)
	}
	if result != (FrameWorkPostFilterResult{}) || next.RemainingPostFilters() != ctx.RemainingPostFilters() {
		t.Fatalf("next remaining=%b result=%+v", next.RemainingPostFilters(), result)
	}
	if !bytes.Equal(output.Y.Pix, beforeY) {
		t.Fatal("short CDEF scratch mutated output through earlier loop filter")
	}
}

func TestFrameWorkPostFilterContextApplySupportedPostFiltersRejectsShortCDEFScratchBeforeLoopFilterMutation(t *testing.T) {
	const width = 32
	const height = 16

	seq := testSequence()
	seq.EnableCDEF = true
	size := parser.FrameSize{
		CodedWidth:          width,
		UpscaledWidth:       width,
		Height:              height,
		SuperResDenominator: 8,
	}
	first := testFrameWorkLoopFilterPostFilterRecordAt(0, 0, 4, 4)
	second := testFrameWorkLoopFilterPostFilterRecordAt(4, 0, 8, 4)
	filterMap := testFrameWorkLoopFilterPostFilterMap(t, size, first, second)
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: width, Height: height, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32})
	testFillFrameWorkLoopFilterPattern(output.Y)
	beforeY := append([]byte(nil), output.Y.Pix...)
	event := Event{
		SequenceHeader: seq,
		FrameSize:      size,
		LoopFilter: parser.LoopFilterParams{
			LevelY:    [2]uint8{63},
			Sharpness: 1,
		},
		CDEF: parser.CDEFParams{
			Damping:       5,
			StrengthCount: 1,
			YStrength:     [parser.MaxCDEFStrengths]uint8{63},
		},
	}
	ctx := FrameWorkPostFilterContext{Event: event, Output: output, LoopFilterMap: &filterMap}
	cdefReq := testFrameWorkCDEFPostFilterRequest(t, ctx, event)
	scratchSize, err := ctx.SupportedPostFilterScratchLen(FrameWorkPostFilterRequest{CDEF: cdefReq})
	if err != nil {
		t.Fatal(err)
	}
	cdefReq.InputScratch = cdefReq.InputScratch[:len(cdefReq.InputScratch)-1]

	next, result, err := ctx.ApplySupportedPostFilters(FrameWorkPostFilterRequest{
		LoopFilter: FrameWorkLoopFilterPostFilterRequest{
			Edges: make([]FrameWorkLoopFilterPostFilterEdge, scratchSize.LoopFilter.Edges),
		},
		CDEF: cdefReq,
	})
	if !errors.Is(err, frame.ErrShortBuffer) {
		t.Fatalf("ApplySupportedPostFilters err=%v want %v", err, frame.ErrShortBuffer)
	}
	if result != (FrameWorkPostFilterResult{}) || next.RemainingPostFilters() != ctx.RemainingPostFilters() {
		t.Fatalf("next remaining=%b result=%+v", next.RemainingPostFilters(), result)
	}
	if !bytes.Equal(output.Y.Pix, beforeY) {
		t.Fatal("short CDEF scratch mutated output through earlier loop filter")
	}
}

func TestFrameWorkPostFilterContextApplySupportedPostFiltersRejectsShortRestorationScratchBeforeLoopFilterMutation(t *testing.T) {
	const width = 64
	const height = 64

	size := parser.FrameSize{
		CodedWidth:          width,
		UpscaledWidth:       width,
		Height:              height,
		SuperResDenominator: 8,
	}
	cols, rows, err := frameWorkLoopFilterMapGrid(size)
	if err != nil {
		t.Fatal(err)
	}
	filler := testFrameWorkLoopFilterPostFilterRecord(cols, rows)
	filler.Block.Size = tile.BlockSize64x64
	filler.SkipTransform = true
	filler.TransformTree = tile.TransformTreeResult{Y: tile.TransformSize64x64, UV: tile.TransformSize32x32, HasUV: true}
	second := testFrameWorkLoopFilterPostFilterRecordAt(4, 0, 8, 4)
	second.SkipTransform = true
	filterMap := testFrameWorkLoopFilterPostFilterMap(t, size, filler, second)
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: width, Height: height, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32})
	testFillFrameWorkLoopFilterPattern(output.Y)
	beforeY := append([]byte(nil), output.Y.Pix...)
	event := Event{
		SequenceHeader: testSequence(),
		FrameSize:      size,
		LoopFilter: parser.LoopFilterParams{
			LevelY:    [2]uint8{63},
			Sharpness: 1,
		},
		Restoration: parser.RestorationParams{
			Type:      [3]parser.RestorationType{parser.RestorationWiener, parser.RestorationNone, parser.RestorationNone},
			UnitSizeY: 64,
		},
	}
	ctx := FrameWorkPostFilterContext{Event: event, Output: output, LoopFilterMap: &filterMap}
	loopScratch, err := ctx.LoopFilterPostFilterScratchLen(FrameWorkLoopFilterPostFilterRequest{})
	if err != nil {
		t.Fatal(err)
	}
	plan, err := ctx.LoopRestorationPostFilterPlan()
	if err != nil {
		t.Fatal(err)
	}
	records, err := tile.BindRestorationFrameRecordBuffers(plan, make([]tile.RestorationUnitRecord, plan.UnitRecordLen()))
	if err != nil {
		t.Fatal(err)
	}
	if err := tile.ResetRestorationPlaneRecords(plan.Grids[0], records[0]); err != nil {
		t.Fatal(err)
	}
	restorationScratch, err := ctx.LoopRestorationPostFilterScratchLen(records, false)
	if err != nil {
		t.Fatal(err)
	}
	if restorationScratch.Samples.DataLen == 0 {
		t.Fatalf("restoration scratch=%+v", restorationScratch)
	}

	next, result, err := ctx.ApplySupportedPostFilters(FrameWorkPostFilterRequest{
		LoopFilter: FrameWorkLoopFilterPostFilterRequest{
			Edges: make([]FrameWorkLoopFilterPostFilterEdge, loopScratch.Edges),
		},
		Restoration: FrameWorkRestorationPostFilterRequest{
			Records:     records,
			DataScratch: make([]uint16, restorationScratch.Samples.DataLen-1),
			DstScratch:  make([]uint16, restorationScratch.Samples.DstLen),
			Scratch:     testFrameWorkRestorationPostFilterScratch(restorationScratch.Apply),
		},
	})
	if !errors.Is(err, tile.ErrJobBufferTooSmall) {
		t.Fatalf("ApplySupportedPostFilters err=%v want %v", err, tile.ErrJobBufferTooSmall)
	}
	if result != (FrameWorkPostFilterResult{}) || next.RemainingPostFilters() != ctx.RemainingPostFilters() {
		t.Fatalf("next remaining=%b result=%+v", next.RemainingPostFilters(), result)
	}
	if !bytes.Equal(output.Y.Pix, beforeY) {
		t.Fatal("short restoration scratch mutated output through earlier loop filter")
	}
}

func TestFrameWorkPostFilterContextApplyLoopFilterLumaEdgesSkipsInactive(t *testing.T) {
	result, err := (FrameWorkPostFilterContext{}).ApplyLoopFilterLumaEdges(FrameWorkLoopFilterPostFilterRequest{})
	if err != nil {
		t.Fatal(err)
	}
	if result != (FrameWorkLoopFilterPostFilterApplyResult{}) {
		t.Fatalf("result=%+v want zero", result)
	}
}

func TestFrameWorkPostFilterContextApplyLoopFilterLumaEdgesRejectsShortEdgeScratchBeforeMutation(t *testing.T) {
	size := parser.FrameSize{
		CodedWidth:          32,
		UpscaledWidth:       32,
		Height:              32,
		SuperResDenominator: 8,
	}
	filterMap := testFrameWorkLoopFilterPostFilterMap(t, size,
		testFrameWorkLoopFilterPostFilterRecordAt(0, 0, 4, 4),
		testFrameWorkLoopFilterPostFilterRecordAt(4, 0, 8, 4),
		testFrameWorkLoopFilterPostFilterRecordAt(0, 4, 4, 8),
		testFrameWorkLoopFilterPostFilterRecordAt(4, 4, 8, 8),
	)
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: 32, Height: 32, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32})
	testFillFrameWorkLoopFilterPattern(output.Y)
	before := append([]byte(nil), output.Y.Pix...)
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			SequenceHeader: testSequence(),
			FrameSize:      size,
			LoopFilter:     parser.LoopFilterParams{LevelY: [2]uint8{8, 9}},
		},
		Output: output,
	}

	result, err := ctx.ApplyLoopFilterLumaEdges(FrameWorkLoopFilterPostFilterRequest{
		Map:   filterMap,
		Edges: make([]FrameWorkLoopFilterPostFilterEdge, 1),
	})
	if !errors.Is(err, frame.ErrShortBuffer) {
		t.Fatalf("ApplyLoopFilterLumaEdges err=%v want %v", err, frame.ErrShortBuffer)
	}
	if !result.Active || result.Plan.DroppedEdges == 0 {
		t.Fatalf("result=%+v", result)
	}
	if !bytes.Equal(output.Y.Pix, before) {
		t.Fatal("short edge scratch mutated output")
	}
}

func TestFrameWorkPostFilterContextApplyLoopFilterLumaEdgesRejectsChromaCandidatesBeforeMutation(t *testing.T) {
	size := parser.FrameSize{
		CodedWidth:          32,
		UpscaledWidth:       32,
		Height:              16,
		SuperResDenominator: 8,
	}
	first := testFrameWorkLoopFilterPostFilterRecordAt(0, 0, 4, 4)
	second := testFrameWorkLoopFilterPostFilterRecordAt(4, 0, 8, 4)
	filterMap := testFrameWorkLoopFilterPostFilterMap(t, size, first, second)
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: 32, Height: 16, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32})
	testFillFrameWorkLoopFilterPattern(output.Y)
	beforeY := append([]byte(nil), output.Y.Pix...)
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			SequenceHeader: testSequence(),
			FrameSize:      size,
			LoopFilter: parser.LoopFilterParams{
				LevelY: [2]uint8{8},
				LevelU: 7,
				LevelV: 9,
			},
		},
		Output: output,
	}

	result, err := ctx.ApplyLoopFilterLumaEdges(FrameWorkLoopFilterPostFilterRequest{
		Map:   filterMap,
		Edges: make([]FrameWorkLoopFilterPostFilterEdge, 3),
	})
	if !errors.Is(err, ErrUnsupportedPostFilter) {
		t.Fatalf("ApplyLoopFilterLumaEdges err=%v want %v", err, ErrUnsupportedPostFilter)
	}
	if !result.Active || result.Plan.PlaneEdgeCandidates != [3]int{1, 1, 1} {
		t.Fatalf("result=%+v", result)
	}
	if !bytes.Equal(output.Y.Pix, beforeY) {
		t.Fatal("chroma candidate rejection mutated output")
	}
}

func TestFrameWorkPostFilterContextApplyLoopFilterLumaEdgesAllocs(t *testing.T) {
	size := parser.FrameSize{
		CodedWidth:          32,
		UpscaledWidth:       32,
		Height:              16,
		SuperResDenominator: 8,
	}
	first := testFrameWorkLoopFilterPostFilterRecordAt(0, 0, 4, 4)
	first.SkipTransform = true
	second := testFrameWorkLoopFilterPostFilterRecordAt(4, 0, 8, 4)
	second.TransformTree = tile.TransformTreeResult{
		Y:        tile.TransformSize16x16,
		Variable: true,
		Split:    [2]uint16{1, 0},
	}
	filterMap := testFrameWorkLoopFilterPostFilterMap(t, size, first, second)
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: 32, Height: 16, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32})
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			SequenceHeader: testSequence(),
			FrameSize:      size,
			LoopFilter:     parser.LoopFilterParams{LevelY: [2]uint8{8}},
		},
		Output: output,
	}
	edges := make([]FrameWorkLoopFilterPostFilterEdge, 4)
	allocs := testing.AllocsPerRun(1000, func() {
		testFillFrameWorkLoopFilterPattern(output.Y)
		result, err := ctx.ApplyLoopFilterLumaEdges(FrameWorkLoopFilterPostFilterRequest{
			Map:   filterMap,
			Edges: edges,
		})
		if err != nil {
			t.Fatal(err)
		}
		if result.Applied != 4 {
			t.Fatalf("result=%+v", result)
		}
	})
	if allocs != 0 {
		t.Fatalf("ApplyLoopFilterLumaEdges allocated: %f", allocs)
	}
}

func TestFrameWorkPostFilterContextLoopFilterPostFilterPlanRejectsMissingActiveMap(t *testing.T) {
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			FrameSize: parser.FrameSize{
				CodedWidth:          16,
				UpscaledWidth:       16,
				Height:              16,
				SuperResDenominator: 8,
			},
			LoopFilter: parser.LoopFilterParams{LevelY: [2]uint8{1}},
		},
	}

	plan, err := ctx.LoopFilterPostFilterPlan(FrameWorkLoopFilterPostFilterRequest{})
	if !errors.Is(err, threading.ErrInvalidBatch) {
		t.Fatalf("LoopFilterPostFilterPlan err=%v want %v", err, threading.ErrInvalidBatch)
	}
	if _, err := ctx.LoopFilterPostFilterScratchLen(FrameWorkLoopFilterPostFilterRequest{}); !errors.Is(err, threading.ErrInvalidBatch) {
		t.Fatalf("LoopFilterPostFilterScratchLen err=%v want %v", err, threading.ErrInvalidBatch)
	}
	if !plan.Active || plan.MICols != 4 || plan.MIRows != 4 {
		t.Fatalf("plan=%+v", plan)
	}
}

func TestFrameWorkPostFilterContextLoopFilterPostFilterPlanRejectsIncompleteMap(t *testing.T) {
	size := parser.FrameSize{
		CodedWidth:          16,
		UpscaledWidth:       16,
		Height:              16,
		SuperResDenominator: 8,
	}
	cols, rows, err := frameWorkLoopFilterMapGrid(size)
	if err != nil {
		t.Fatal(err)
	}
	filterMap := FrameWorkLoopFilterMap{
		Records: make([]threading.FrameWorkLoopFilterBlockRecord, cols*rows),
		Stride:  cols,
		Rows:    rows,
	}
	filterMap.Records[0] = testFrameWorkLoopFilterPostFilterRecord(cols, rows)
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			SequenceHeader: testSequence(),
			FrameSize:      size,
			LoopFilter:     parser.LoopFilterParams{LevelY: [2]uint8{8}},
		},
	}

	plan, err := ctx.LoopFilterPostFilterPlan(FrameWorkLoopFilterPostFilterRequest{Map: filterMap})
	if !errors.Is(err, threading.ErrInvalidBatch) {
		t.Fatalf("LoopFilterPostFilterPlan err=%v want %v", err, threading.ErrInvalidBatch)
	}
	if plan.Blocks != 1 || plan.Missing != cols*rows-1 {
		t.Fatalf("plan=%+v", plan)
	}
}

func BenchmarkLoopFilterPostFilterPlan(b *testing.B) {
	ctx, filterMap, edges := benchmarkLoopFilterPostFilterFixture(b)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		plan, err := ctx.LoopFilterPostFilterPlan(FrameWorkLoopFilterPostFilterRequest{
			Map:   filterMap,
			Edges: edges,
		})
		if err != nil {
			b.Fatal(err)
		}
		if plan.StoredEdges != 72 {
			b.Fatalf("stored edges=%d want 72", plan.StoredEdges)
		}
	}
}

func BenchmarkApplyLoopFilterEdges(b *testing.B) {
	ctx, filterMap, edges := benchmarkLoopFilterPostFilterFixture(b)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchmarkFillLoopFilterFrame(ctx.Output)
		result, err := ctx.ApplyLoopFilterEdges(FrameWorkLoopFilterPostFilterRequest{
			Map:   filterMap,
			Edges: edges,
		})
		if err != nil {
			b.Fatal(err)
		}
		if result.Applied != 72 {
			b.Fatalf("applied=%d want 72", result.Applied)
		}
	}
}

func BenchmarkApplySupportedPostFiltersLoopFilterCDEF(b *testing.B) {
	ctx, filterMap, edges := benchmarkLoopFilterPostFilterFixture(b)
	ctx.Event.SequenceHeader.EnableCDEF = true
	ctx.Event.CDEF = parser.CDEFParams{
		Damping:       5,
		StrengthCount: 1,
		YStrength:     [parser.MaxCDEFStrengths]uint8{63},
		UVStrength:    [parser.MaxCDEFStrengths]uint8{63},
	}
	cdefReq := testFrameWorkCDEFPostFilterRequest(b, ctx, ctx.Event)
	req := FrameWorkPostFilterRequest{
		LoopFilter: FrameWorkLoopFilterPostFilterRequest{
			Map:   filterMap,
			Edges: edges,
		},
		CDEF: cdefReq,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchmarkFillLoopFilterFrame(ctx.Output)
		next, result, err := ctx.ApplySupportedPostFilters(req)
		if err != nil {
			b.Fatal(err)
		}
		if result.Completed != FrameWorkPostFilterLoopFilter|FrameWorkPostFilterCDEF ||
			result.LoopFilter.Applied != 72 ||
			result.CDEF.Units == 0 ||
			next.RemainingPostFilters() != 0 {
			b.Fatalf("next remaining=%b result=%+v", next.RemainingPostFilters(), result)
		}
	}
}

func testFillFrameWorkLoopFilterPattern(plane frame.Plane) {
	for y := 0; y < plane.Height; y++ {
		for x := 0; x < plane.Width; x++ {
			value := byte(90)
			if (x/8+y/8)&1 != 0 {
				value = 100
			}
			plane.Pix[y*plane.Stride+x] = value
		}
	}
}

func testFillFrameWorkLoopFilterSamplePattern(plane frame.Plane, bytesPerSample int) {
	for y := 0; y < plane.Height; y++ {
		for x := 0; x < plane.Width; x++ {
			value := uint16(360)
			if (x/8+y/8)&1 != 0 {
				value = 400
			}
			testSetFrameWorkLoopFilterSample(plane, bytesPerSample, x, y, value)
		}
	}
}

func testSetFrameWorkLoopFilterSample(plane frame.Plane, bytesPerSample int, x int, y int, value uint16) {
	offset := y*plane.Stride + x*bytesPerSample
	if bytesPerSample == 1 {
		plane.Pix[offset] = byte(value)
		return
	}
	binary.LittleEndian.PutUint16(plane.Pix[offset:], value)
}

func testFillFrameWorkLoopFilterOrderPattern(plane frame.Plane, rng *rand.Rand) {
	for y := 0; y < plane.Height; y++ {
		for x := 0; x < plane.Width; x++ {
			value := 88 + rng.Intn(25)
			if x >= plane.Width/2 {
				value += rng.Intn(8)
			}
			if y >= plane.Height/2 {
				value += rng.Intn(8)
			}
			plane.Pix[y*plane.Stride+x] = byte(value)
		}
	}
}

func testCloneFrameWorkLoopFilterFrame(src *frame.Frame) frame.Frame {
	clone := *src
	clone.Y.Pix = append([]byte(nil), src.Y.Pix...)
	if !src.Format.MonoChrome {
		clone.U.Pix = append([]byte(nil), src.U.Pix...)
		clone.V.Pix = append([]byte(nil), src.V.Pix...)
	}
	return clone
}

func testApplyFrameWorkLoopFilterEdgesDirect(t *testing.T, output frame.Frame, sharpness uint8, edges []FrameWorkLoopFilterPostFilterEdge) {
	t.Helper()
	for plane := loopfilter.PlaneY; plane <= loopfilter.PlaneV; plane++ {
		for edgeKind := loopfilter.EdgeVertical; edgeKind <= loopfilter.EdgeHorizontal; edgeKind++ {
			for _, edge := range edges {
				if edge.Plane != plane || edge.Edge != edgeKind {
					continue
				}
				testApplyFrameWorkLoopFilterEdgeDirect(t, output, sharpness, edge)
			}
		}
	}
}

func testApplyFrameWorkLoopFilterEdgesStoredOrder(t *testing.T, output frame.Frame, sharpness uint8, edges []FrameWorkLoopFilterPostFilterEdge) {
	t.Helper()
	for _, edge := range edges {
		testApplyFrameWorkLoopFilterEdgeDirect(t, output, sharpness, edge)
	}
}

func testApplyFrameWorkLoopFilterEdgeDirect(t *testing.T, output frame.Frame, sharpness uint8, edge FrameWorkLoopFilterPostFilterEdge) {
	t.Helper()
	thresholds, err := loopfilter.ThresholdsForLevel(edge.Level, sharpness)
	if err != nil {
		t.Fatal(err)
	}
	plane, ok := frameWorkLoopFilterOutputPlane(output, edge.Plane)
	if !ok {
		t.Fatalf("missing output plane for edge=%+v", edge)
	}
	if err := loopfilter.FilterEdgeByWidth(edge.Width, plane, output.Layout.BytesPerSample, output.Format.BitDepth, edge.Edge, edge.X4*4, edge.Y4*4, edge.Length4*4, thresholds); err != nil {
		t.Fatal(err)
	}
}

func testFrameWorkLoopFilterPostFilterRecord(cols int, rows int) threading.FrameWorkLoopFilterBlockRecord {
	return testFrameWorkLoopFilterPostFilterRecordAt(0, 0, cols, rows)
}

func testFrameWorkLoopFilterPostFilterRecordAt(col0 int, row0 int, col1 int, row1 int) threading.FrameWorkLoopFilterBlockRecord {
	visibleW4 := uint8(col1 - col0)
	visibleH4 := uint8(row1 - row0)
	return threading.FrameWorkLoopFilterBlockRecord{
		Valid: true,
		Block: tile.BlockVisit{
			MICol:     uint32(col0),
			MIRow:     uint32(row0),
			MIColEnd:  uint32(col1),
			MIRowEnd:  uint32(row1),
			X4:        col0,
			Y4:        row0,
			Size:      tile.BlockSize16x16,
			VisibleW4: visibleW4,
			VisibleH4: visibleH4,
		},
		TransformTree: tile.TransformTreeResult{Y: tile.TransformSize16x16, UV: tile.TransformSize8x8, HasUV: true},
		SegmentID:     0,
		RefFrame:      0,
		Mode:          loopfilter.ModeDeltaClassZero,
	}
}

func testFrameWorkLoopFilterPostFilterMap(t testing.TB, size parser.FrameSize, records ...threading.FrameWorkLoopFilterBlockRecord) FrameWorkLoopFilterMap {
	t.Helper()
	cols, rows, err := frameWorkLoopFilterMapGrid(size)
	if err != nil {
		t.Fatal(err)
	}
	filterMap := FrameWorkLoopFilterMap{
		Records: make([]threading.FrameWorkLoopFilterBlockRecord, cols*rows),
		Stride:  cols,
		Rows:    rows,
	}
	for _, record := range records {
		for miRow := record.Block.MIRow; miRow < record.Block.MIRowEnd; miRow++ {
			row := int(miRow) * filterMap.Stride
			for miCol := record.Block.MICol; miCol < record.Block.MIColEnd; miCol++ {
				filterMap.Records[row+int(miCol)] = record
			}
		}
	}
	return filterMap
}

func benchmarkLoopFilterPostFilterFixture(b *testing.B) (FrameWorkPostFilterContext, FrameWorkLoopFilterMap, []FrameWorkLoopFilterPostFilterEdge) {
	b.Helper()
	const width = 64
	const height = 64
	size := parser.FrameSize{
		CodedWidth:          width,
		UpscaledWidth:       width,
		Height:              height,
		SuperResDenominator: 8,
	}
	records := make([]threading.FrameWorkLoopFilterBlockRecord, 0, 16)
	for row := 0; row < 16; row += 4 {
		for col := 0; col < 16; col += 4 {
			records = append(records, testFrameWorkLoopFilterPostFilterRecordAt(col, row, col+4, row+4))
		}
	}
	filterMap := testFrameWorkLoopFilterPostFilterMap(b, size, records...)
	output := testFrameWorkCDEFFrame(b, frame.Format{Width: width, Height: height, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 64})
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			SequenceHeader: testSequence(),
			FrameSize:      size,
			LoopFilter: parser.LoopFilterParams{
				LevelY:    [2]uint8{63, 63},
				LevelU:    63,
				LevelV:    63,
				Sharpness: 1,
			},
		},
		Output:        output,
		LoopFilterMap: &filterMap,
	}
	scratch, err := ctx.LoopFilterPostFilterScratchLen(FrameWorkLoopFilterPostFilterRequest{Map: filterMap})
	if err != nil {
		b.Fatal(err)
	}
	edges, err := scratch.BindEdges(make([]FrameWorkLoopFilterPostFilterEdge, scratch.Edges))
	if err != nil {
		b.Fatal(err)
	}
	if scratch.Edges != 72 {
		b.Fatalf("edge scratch=%d want 72", scratch.Edges)
	}
	return ctx, filterMap, edges
}

func benchmarkFillLoopFilterFrame(output *frame.Frame) {
	testFillFrameWorkLoopFilterPattern(output.Y)
	testFillFrameWorkLoopFilterPattern(output.U)
	testFillFrameWorkLoopFilterPattern(output.V)
}

// TestFrameWorkPostFilterContextApplyLoopFilterEdgesEnhancementLayerResolution
// pins loop-filter dispatch to the per-event frame size by exercising a
// 1280x720 frame end-to-end. SVC enhancement layers decode at their own
// resolution, so the loop filter must not assume a global frame size: plane
// sizes, the MI grid, edge fit checks, and chroma subsampling must all derive
// from the current event's FrameSize/SequenceHeader, not any stream-wide
// maximum.
func TestFrameWorkPostFilterContextApplyLoopFilterEdgesEnhancementLayerResolution(t *testing.T) {
	const width = 1280
	const height = 720
	const miBlock = 16 // BlockSize64x64 covers a 16x16 MI superblock.

	size := parser.FrameSize{
		CodedWidth:          width,
		UpscaledWidth:       width,
		Height:              height,
		SuperResDenominator: 8,
	}
	cols, rows, err := frameWorkLoopFilterMapGrid(size)
	if err != nil {
		t.Fatal(err)
	}
	if cols != 320 || rows != 180 {
		t.Fatalf("frameWorkLoopFilterMapGrid(1280x720)=(%d,%d) want (320,180)", cols, rows)
	}

	records := make([]threading.FrameWorkLoopFilterBlockRecord, 0, 256)
	for row := 0; row < rows; row += miBlock {
		blockRows := miBlock
		if row+blockRows > rows {
			blockRows = rows - row
		}
		for col := 0; col < cols; col += miBlock {
			blockCols := miBlock
			if col+blockCols > cols {
				blockCols = cols - col
			}
			records = append(records, testFrameWorkLoopFilterEnhancementLayerRecord(col, row, blockCols, blockRows))
		}
	}
	filterMap := testFrameWorkLoopFilterPostFilterMap(t, size, records...)

	output := testFrameWorkCDEFFrame(t, frame.Format{
		Width:        width,
		Height:       height,
		BitDepth:     8,
		SubsamplingX: true,
		SubsamplingY: true,
		Align:        64,
	})
	testFillFrameWorkLoopFilterPattern(output.Y)
	testFillFrameWorkLoopFilterPattern(output.U)
	testFillFrameWorkLoopFilterPattern(output.V)
	beforeY := append([]byte(nil), output.Y.Pix...)
	beforeU := append([]byte(nil), output.U.Pix...)
	beforeV := append([]byte(nil), output.V.Pix...)

	ctx := FrameWorkPostFilterContext{
		Event: Event{
			SequenceHeader: testSequence(),
			FrameSize:      size,
			LoopFilter: parser.LoopFilterParams{
				LevelY:    [2]uint8{32, 32},
				LevelU:    20,
				LevelV:    20,
				Sharpness: 1,
			},
		},
		Output:        output,
		LoopFilterMap: &filterMap,
	}

	scratch, err := ctx.LoopFilterPostFilterScratchLen(FrameWorkLoopFilterPostFilterRequest{Map: filterMap})
	if err != nil {
		t.Fatal(err)
	}
	if scratch.Edges == 0 {
		t.Fatalf("scratch=%+v want nonzero", scratch)
	}
	edges := make([]FrameWorkLoopFilterPostFilterEdge, scratch.Edges)

	plan, err := ctx.LoopFilterPostFilterPlan(FrameWorkLoopFilterPostFilterRequest{Map: filterMap, Edges: edges})
	if err != nil {
		t.Fatal(err)
	}
	if !plan.Active || plan.MICols != cols || plan.MIRows != rows {
		t.Fatalf("plan=%+v want active grid %dx%d", plan, cols, rows)
	}
	if plan.Cells != cols*rows {
		t.Fatalf("plan.Cells=%d want %d", plan.Cells, cols*rows)
	}
	if plan.Blocks != len(records) {
		t.Fatalf("plan.Blocks=%d want %d", plan.Blocks, len(records))
	}
	if plan.Missing != 0 {
		t.Fatalf("plan.Missing=%d want 0", plan.Missing)
	}
	if plan.DroppedEdges != 0 || plan.StoredEdges != plan.EdgeCandidates {
		t.Fatalf("plan storage=%+v want all stored", plan)
	}
	if plan.PlaneEdgeCandidates[loopfilter.PlaneY] == 0 ||
		plan.PlaneEdgeCandidates[loopfilter.PlaneU] == 0 ||
		plan.PlaneEdgeCandidates[loopfilter.PlaneV] == 0 {
		t.Fatalf("plan PlaneEdgeCandidates=%v want all planes nonzero", plan.PlaneEdgeCandidates)
	}

	// Verify the bottom-right edge candidates use the per-event frame size:
	// at least one stored vertical luma edge sits beyond MICol=128 (X=512
	// luma) and at least one horizontal edge sits beyond MIRow=128 (Y=512
	// luma). Without per-event sizing, edges in the >512 region could not
	// resolve a valid width and would be skipped or rejected.
	farLumaVertical := false
	farLumaHorizontal := false
	farChromaVertical := false
	for _, e := range edges[:plan.StoredEdges] {
		if e.Plane == loopfilter.PlaneY && e.Edge == loopfilter.EdgeVertical && e.X4 >= 128 {
			farLumaVertical = true
		}
		if e.Plane == loopfilter.PlaneY && e.Edge == loopfilter.EdgeHorizontal && e.Y4 >= 128 {
			farLumaHorizontal = true
		}
		if (e.Plane == loopfilter.PlaneU || e.Plane == loopfilter.PlaneV) &&
			e.Edge == loopfilter.EdgeVertical && e.X4 >= 64 {
			farChromaVertical = true
		}
	}
	if !farLumaVertical {
		t.Fatal("no luma vertical edge stored past MICol=128 (X=512 luma)")
	}
	if !farLumaHorizontal {
		t.Fatal("no luma horizontal edge stored past MIRow=128 (Y=512 luma)")
	}
	if !farChromaVertical {
		t.Fatal("no chroma vertical edge stored past chroma X4=64")
	}

	wantFrame := testCloneFrameWorkLoopFilterFrame(output)
	testApplyFrameWorkLoopFilterEdgesDirect(t, wantFrame, ctx.Event.LoopFilter.Sharpness, edges[:plan.StoredEdges])

	result, err := ctx.ApplyLoopFilterEdges(FrameWorkLoopFilterPostFilterRequest{
		Map:   filterMap,
		Edges: edges,
	})
	if err != nil {
		t.Fatal(err)
	}
	if !result.Active || result.Applied == 0 || result.Plan.StoredEdges != plan.StoredEdges {
		t.Fatalf("result=%+v", result)
	}
	if bytes.Equal(output.Y.Pix, beforeY) {
		t.Fatal("1280x720 loop filter did not change luma samples")
	}
	if bytes.Equal(output.U.Pix, beforeU) || bytes.Equal(output.V.Pix, beforeV) {
		t.Fatal("1280x720 loop filter did not change chroma samples")
	}
	if !bytes.Equal(output.Y.Pix, wantFrame.Y.Pix) ||
		!bytes.Equal(output.U.Pix, wantFrame.U.Pix) ||
		!bytes.Equal(output.V.Pix, wantFrame.V.Pix) {
		t.Fatal("1280x720 loop filter bridge output did not match direct edge application")
	}
}

// testFrameWorkLoopFilterEnhancementLayerRecord builds a loop-filter block
// record at a frame-relative MI position. Unlike the small-fixture helpers, X4
// and Y4 are pinned to the block's local superblock origin (0,0) so the
// transform-tree validator's MaxBlockModeSlots check passes regardless of how
// far into the frame the block sits. Each block is one SkipTransform
// superblock so the test exercises whole-block luma and chroma edges without
// recursing into variable transform splits.
func testFrameWorkLoopFilterEnhancementLayerRecord(miCol int, miRow int, miCols int, miRows int) threading.FrameWorkLoopFilterBlockRecord {
	return threading.FrameWorkLoopFilterBlockRecord{
		Valid: true,
		Block: tile.BlockVisit{
			MICol:     uint32(miCol),
			MIRow:     uint32(miRow),
			MIColEnd:  uint32(miCol + miCols),
			MIRowEnd:  uint32(miRow + miRows),
			X4:        0,
			Y4:        0,
			Size:      tile.BlockSize64x64,
			VisibleW4: uint8(miCols),
			VisibleH4: uint8(miRows),
		},
		TransformTree: tile.TransformTreeResult{
			Y:     tile.TransformSize32x32,
			UV:    tile.TransformSize16x16,
			HasUV: true,
		},
		SkipTransform: true,
		SegmentID:     0,
		RefFrame:      0,
		Mode:          loopfilter.ModeDeltaClassZero,
	}
}
