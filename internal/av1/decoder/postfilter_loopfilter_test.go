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
		{Plane: loopfilter.PlaneU, Edge: loopfilter.EdgeVertical, X4: 2, Y4: 0, Length4: 2, Level: 7, Transform: tile.TransformSize8x8, Width: 8, BlockMICol: 4, BlockMIRow: 0},
		{Plane: loopfilter.PlaneV, Edge: loopfilter.EdgeVertical, X4: 2, Y4: 0, Length4: 2, Level: 9, Transform: tile.TransformSize8x8, Width: 8, BlockMICol: 4, BlockMIRow: 0},
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
				X4:         tt.wantX4,
				Y4:         tt.wantY4,
				Length4:    tt.wantLen,
				Level:      7,
				Transform:  tile.TransformSize8x8,
				Width:      8,
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
		if edges[i].Width != want {
			t.Fatalf("edge[%d]=%+v want width %d", i, edges[i], want)
		}
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
		if edges[i].Width != want {
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
		Width:             8,
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
		Width:             8,
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
	if scratchSize.LoopFilter.Edges != 1 || scratchSize.CDEF.Input != cdef.InputBufferSize {
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
