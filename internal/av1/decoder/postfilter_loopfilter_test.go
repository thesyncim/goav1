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
	return threading.FrameWorkLoopFilterBlockRecord{
		Valid: true,
		Block: tile.BlockVisit{
			MIColEnd: uint32(cols),
			MIRowEnd: uint32(rows),
		},
		SegmentID: 0,
		RefFrame:  0,
		Mode:      loopfilter.ModeDeltaClassZero,
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
