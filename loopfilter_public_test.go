package goav1_test

import (
	"bytes"
	"errors"
	"testing"

	av1 "github.com/thesyncim/goav1"
)

func TestPublicLoopFilterLevelThresholdsAndBlockLevel(t *testing.T) {
	params := av1.LoopFilterParams{
		LevelY: [2]uint8{40, 20},
		LevelU: 12,
		LevelV: 40,
		Deltas: av1.LoopFilterDeltas{
			Ref:  [av1.RefFrames]int8{1, 0, 0, 0, -2},
			Mode: [av1.LoopFilterModeDeltas]int8{0, 3},
		},
		ModeRefDeltaEnabled: true,
	}
	base, err := av1.LoopFilterBaseLevel(params, av1.LoopFilterPlaneY, av1.LoopFilterEdgeVertical)
	if err != nil {
		t.Fatal(err)
	}
	if base != 40 || av1.LoopFilterClampLevel(99) != av1.LoopFilterMaxLevel {
		t.Fatalf("base=%d clamp=%d", base, av1.LoopFilterClampLevel(99))
	}
	if idx, err := av1.LoopFilterDeltaIndex(av1.LoopFilterPlaneV, av1.LoopFilterEdgeHorizontal); err != nil || idx != 3 {
		t.Fatalf("delta index=%d err=%v want 3,nil", idx, err)
	}

	multi := [av1.LoopFilterDeltaCount]int8{-2, 3, 4, -5}
	gotDelta, err := av1.SelectLoopFilterDelta(av1.DeltaParams{DeltaLFPresent: true, DeltaLFMulti: true}, 7, multi, av1.LoopFilterPlaneV, av1.LoopFilterEdgeHorizontal)
	if err != nil || gotDelta != -5 {
		t.Fatalf("delta=%d err=%v want -5,nil", gotDelta, err)
	}

	seg := av1.SegmentationParams{Enabled: true}
	seg.Data.Segments[2] = av1.SegmentData{
		DeltaLFYV: -5,
		DeltaLFYH: 2,
		DeltaLFU:  7,
		DeltaLFV:  9,
	}
	if got, err := av1.LoopFilterSegmentDelta(seg, 2, av1.LoopFilterPlaneU, av1.LoopFilterEdgeHorizontal); err != nil || got != 7 {
		t.Fatalf("segment delta=%d err=%v want 7,nil", got, err)
	}

	level, err := av1.ResolveLoopFilterLevel(params, seg, av1.LoopFilterLevelRequest{
		Plane:     av1.LoopFilterPlaneY,
		Edge:      av1.LoopFilterEdgeVertical,
		SegmentID: 2,
		RefFrame:  4,
		Mode:      av1.LoopFilterModeDeltaClassMotion,
		DeltaLF:   -3,
	})
	if err != nil {
		t.Fatal(err)
	}
	if level != 34 {
		t.Fatalf("level=%d want 34", level)
	}

	blockLevel, err := av1.ResolveLoopFilterBlockLevel(
		av1.LoopFilterParams{LevelY: [2]uint8{40, 20}, LevelU: 12},
		av1.SegmentationParams{},
		av1.DeltaParams{DeltaLFPresent: true, DeltaLFMulti: true},
		av1.LoopFilterDeltaState{FromBase: -3, Multi: [av1.LoopFilterDeltaCount]int8{1, 2, 4, 8}},
		av1.LoopFilterLevelRequest{Plane: av1.LoopFilterPlaneU, Edge: av1.LoopFilterEdgeHorizontal, SegmentID: 0, RefFrame: 0},
	)
	if err != nil {
		t.Fatal(err)
	}
	if blockLevel != 16 {
		t.Fatalf("block level=%d want 16", blockLevel)
	}

	thresholds, err := av1.LoopFilterThresholdsForLevel(63, 7)
	if err != nil {
		t.Fatal(err)
	}
	if thresholds != (av1.LoopFilterThresholds{Limit: 2, BlockLimit: 132, HighEdgeVariance: 3}) {
		t.Fatalf("thresholds=%+v", thresholds)
	}
}

func TestPublicApplyLoopFilterEdges(t *testing.T) {
	plane := publicPredictionPlane(4, 4, 1, 4)
	for x := range 4 {
		setPublicFrameSample(plane, 1, x, 0, 90)
		setPublicFrameSample(plane, 1, x, 1, 90)
		setPublicFrameSample(plane, 1, x, 2, 100)
		setPublicFrameSample(plane, 1, x, 3, 100)
	}
	thresholds := av1.LoopFilterThresholds{Limit: 20, BlockLimit: 25, HighEdgeVariance: 10}
	if err := av1.ApplyLoopFilter4Edge(plane, 1, 8, av1.LoopFilterEdgeHorizontal, 0, 2, 4, thresholds); err != nil {
		t.Fatal(err)
	}
	for x := range 4 {
		if got := getPublicFrameSample(plane, 1, x, 0); got != 92 {
			t.Fatalf("p1 sample=%d want 92", got)
		}
		if got := getPublicFrameSample(plane, 1, x, 1); got != 94 {
			t.Fatalf("p0 sample=%d want 94", got)
		}
		if got := getPublicFrameSample(plane, 1, x, 2); got != 96 {
			t.Fatalf("q0 sample=%d want 96", got)
		}
		if got := getPublicFrameSample(plane, 1, x, 3); got != 98 {
			t.Fatalf("q1 sample=%d want 98", got)
		}
	}

	wide := publicPredictionPlane(4, 8, 1, 4)
	for x := range 4 {
		for y := range 4 {
			setPublicFrameSample(wide, 1, x, y, 90)
		}
		for y := 4; y < 8; y++ {
			setPublicFrameSample(wide, 1, x, y, 100)
		}
	}
	if err := av1.ApplyLoopFilterEdgeByWidth(8, wide, 1, 8, av1.LoopFilterEdgeHorizontal, 0, 4, 4, thresholds); err != nil {
		t.Fatal(err)
	}
	if getPublicFrameSample(wide, 1, 0, 1) != 91 ||
		getPublicFrameSample(wide, 1, 0, 2) != 93 ||
		getPublicFrameSample(wide, 1, 0, 4) != 96 ||
		getPublicFrameSample(wide, 1, 0, 6) != 99 {
		t.Fatalf("unexpected 8-tap filtered column")
	}
}

func TestPublicApplyLoopFilterBlockEdges(t *testing.T) {
	plane := publicPredictionPlane(4, 4, 1, 4)
	for x := range 4 {
		setPublicFrameSample(plane, 1, x, 0, 90)
		setPublicFrameSample(plane, 1, x, 1, 90)
		setPublicFrameSample(plane, 1, x, 2, 100)
		setPublicFrameSample(plane, 1, x, 3, 100)
	}
	result, err := av1.ApplyLoopFilter4BlockEdge(plane, 1, 8,
		av1.LoopFilterParams{LevelY: [2]uint8{0, 16}},
		av1.SegmentationParams{},
		av1.DeltaParams{},
		av1.LoopFilterDeltaState{},
		av1.LoopFilter4Request{
			LevelRequest: av1.LoopFilterLevelRequest{Plane: av1.LoopFilterPlaneY, Edge: av1.LoopFilterEdgeHorizontal, SegmentID: 0, RefFrame: 0},
			X:            0,
			Y:            2,
			Length:       4,
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	if !result.Applied || result.Level != 16 || result.Thresholds != (av1.LoopFilterThresholds{Limit: 16, BlockLimit: 52, HighEdgeVariance: 1}) {
		t.Fatalf("result=%+v", result)
	}
	if getPublicFrameSample(plane, 1, 0, 1) != 94 || getPublicFrameSample(plane, 1, 0, 2) != 96 {
		t.Fatalf("resolved edge did not filter expected samples")
	}

	for _, width := range []int{6, 8, 14} {
		req := av1.LoopFilterEdgeRequest{
			LevelRequest: av1.LoopFilterLevelRequest{Plane: av1.LoopFilterPlaneY, Edge: av1.LoopFilterEdgeHorizontal, SegmentID: 0, RefFrame: 0},
			X:            0,
			Y:            16,
			Length:       32,
		}
		params := av1.LoopFilterParams{LevelY: [2]uint8{0, 16}}
		got := publicLoopFilterPatternPlane(32, 32)
		want := av1.FramePlane{Pix: append([]byte(nil), got.Pix...), Stride: got.Stride, Width: got.Width, Height: got.Height}
		if _, err := av1.ApplyLoopFilterBlockEdge(want, 1, 8, params, av1.SegmentationParams{}, av1.DeltaParams{}, av1.LoopFilterDeltaState{}, av1.LoopFilterBlockRequest{
			FilterEdgeRequest: req,
			Width:             width,
		}); err != nil {
			t.Fatalf("width=%d generic err=%v", width, err)
		}
		switch width {
		case 6:
			result, err = av1.ApplyLoopFilter6BlockEdge(got, 1, 8, params, av1.SegmentationParams{}, av1.DeltaParams{}, av1.LoopFilterDeltaState{}, req)
		case 8:
			result, err = av1.ApplyLoopFilter8BlockEdge(got, 1, 8, params, av1.SegmentationParams{}, av1.DeltaParams{}, av1.LoopFilterDeltaState{}, req)
		case 14:
			result, err = av1.ApplyLoopFilter14BlockEdge(got, 1, 8, params, av1.SegmentationParams{}, av1.DeltaParams{}, av1.LoopFilterDeltaState{}, req)
		}
		if err != nil {
			t.Fatalf("width=%d wrapper err=%v", width, err)
		}
		if !result.Applied || result.Level != 16 {
			t.Fatalf("width=%d result=%+v", width, result)
		}
		if !bytes.Equal(got.Pix, want.Pix) {
			t.Fatalf("width=%d wrapper output mismatch", width)
		}
	}
}

func TestPublicLoopFilterRejectsInvalid(t *testing.T) {
	params := av1.LoopFilterParams{LevelY: [2]uint8{1, 1}}
	if _, err := av1.ResolveLoopFilterLevel(params, av1.SegmentationParams{}, av1.LoopFilterLevelRequest{Plane: av1.LoopFilterPlane(9), Edge: av1.LoopFilterEdgeVertical, RefFrame: 0, SegmentID: 0}); !errors.Is(err, av1.ErrLoopFilterInvalidFilter) {
		t.Fatalf("ResolveLoopFilterLevel err=%v want %v", err, av1.ErrLoopFilterInvalidFilter)
	}
	if _, err := av1.LoopFilterThresholdsForLevel(av1.LoopFilterMaxLevel+1, 0); !errors.Is(err, av1.ErrLoopFilterInvalidFilter) {
		t.Fatalf("LoopFilterThresholdsForLevel err=%v want %v", err, av1.ErrLoopFilterInvalidFilter)
	}
	plane := publicPredictionPlane(4, 4, 1, 4)
	if err := av1.ApplyLoopFilter4Edge(plane, 1, 8, av1.LoopFilterEdgeHorizontal, 1, 2, 4, av1.LoopFilterThresholds{}); !errors.Is(err, av1.ErrLoopFilterInvalidFilter) {
		t.Fatalf("ApplyLoopFilter4Edge err=%v want %v", err, av1.ErrLoopFilterInvalidFilter)
	}
	if _, err := av1.ApplyLoopFilterBlockEdge(plane, 1, 8, params, av1.SegmentationParams{}, av1.DeltaParams{}, av1.LoopFilterDeltaState{}, av1.LoopFilterBlockRequest{
		FilterEdgeRequest: av1.LoopFilterEdgeRequest{LevelRequest: av1.LoopFilterLevelRequest{Plane: av1.LoopFilterPlaneY, Edge: av1.LoopFilterEdgeVertical, SegmentID: 0, RefFrame: 0}, X: 2, Y: 0, Length: 4},
		Width:             5,
	}); !errors.Is(err, av1.ErrLoopFilterInvalidFilter) {
		t.Fatalf("ApplyLoopFilterBlockEdge err=%v want %v", err, av1.ErrLoopFilterInvalidFilter)
	}
}

func TestPublicLoopFilterAllocs(t *testing.T) {
	plane := publicPredictionPlane(64, 64, 1, 64)
	for y := 0; y < plane.Height; y++ {
		for x := 0; x < plane.Width; x++ {
			setPublicFrameSample(plane, 1, x, y, 100)
		}
	}
	params := av1.LoopFilterParams{LevelY: [2]uint8{16, 16}}
	req := av1.LoopFilter4Request{
		LevelRequest: av1.LoopFilterLevelRequest{Plane: av1.LoopFilterPlaneY, Edge: av1.LoopFilterEdgeHorizontal, SegmentID: 0, RefFrame: 0},
		X:            0,
		Y:            32,
		Length:       64,
	}
	thresholds := av1.LoopFilterThresholds{Limit: 20, BlockLimit: 25, HighEdgeVariance: 10}
	allocs := testing.AllocsPerRun(1000, func() {
		if _, err := av1.ResolveLoopFilterLevel(params, av1.SegmentationParams{}, req.LevelRequest); err != nil {
			t.Fatalf("ResolveLoopFilterLevel err=%v", err)
		}
		if _, err := av1.LoopFilterThresholdsForLevel(16, 0); err != nil {
			t.Fatalf("LoopFilterThresholdsForLevel err=%v", err)
		}
		if err := av1.ApplyLoopFilter4Edge(plane, 1, 8, av1.LoopFilterEdgeHorizontal, 0, 32, 64, thresholds); err != nil {
			t.Fatalf("ApplyLoopFilter4Edge err=%v", err)
		}
		if _, err := av1.ApplyLoopFilter4BlockEdge(plane, 1, 8, params, av1.SegmentationParams{}, av1.DeltaParams{}, av1.LoopFilterDeltaState{}, req); err != nil {
			t.Fatalf("ApplyLoopFilter4BlockEdge err=%v", err)
		}
	})
	if allocs != 0 {
		t.Fatalf("allocs=%v want 0", allocs)
	}
}

func publicLoopFilterPatternPlane(width int, height int) av1.FramePlane {
	plane := publicPredictionPlane(width, height, 1, width)
	for y := 0; y < plane.Height; y++ {
		for x := 0; x < plane.Width; x++ {
			setPublicFrameSample(plane, 1, x, y, uint16((x*11+y*13)&0xff))
		}
	}
	return plane
}
