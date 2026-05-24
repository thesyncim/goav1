package decoder

import (
	"errors"
	"testing"

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
		{Plane: loopfilter.PlaneY, Edge: loopfilter.EdgeVertical, X4: 4, Y4: 0, Length4: 2, Level: 8, Transform: tile.TransformSize8x8, BlockMICol: 4, BlockMIRow: 0},
		{Plane: loopfilter.PlaneY, Edge: loopfilter.EdgeVertical, X4: 6, Y4: 0, Length4: 2, Level: 8, Transform: tile.TransformSize8x8, BlockMICol: 4, BlockMIRow: 0},
		{Plane: loopfilter.PlaneY, Edge: loopfilter.EdgeVertical, X4: 4, Y4: 2, Length4: 2, Level: 8, Transform: tile.TransformSize8x8, BlockMICol: 4, BlockMIRow: 0},
		{Plane: loopfilter.PlaneY, Edge: loopfilter.EdgeVertical, X4: 6, Y4: 2, Length4: 2, Level: 8, Transform: tile.TransformSize8x8, BlockMICol: 4, BlockMIRow: 0},
	}
	for i := range want {
		if edges[i] != want[i] {
			t.Fatalf("edge[%d]=%+v want %+v", i, edges[i], want[i])
		}
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
		TransformTree: tile.TransformTreeResult{Y: tile.TransformSize16x16},
		SegmentID:     0,
		RefFrame:      0,
		Mode:          loopfilter.ModeDeltaClassZero,
	}
}

func testFrameWorkLoopFilterPostFilterMap(t *testing.T, size parser.FrameSize, records ...threading.FrameWorkLoopFilterBlockRecord) FrameWorkLoopFilterMap {
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
