package goav1

import (
	"bytes"
	"testing"

	internalloopfilter "github.com/thesyncim/goav1/internal/av1/loopfilter"
	internalparser "github.com/thesyncim/goav1/internal/av1/parser"
)

func TestPublicApplyLoopFilterBlockEdges(t *testing.T) {
	plane := publicPredictionPlane(4, 4, 1, 4)
	for x := range 4 {
		setPublicFrameSample(plane, 1, x, 0, 90)
		setPublicFrameSample(plane, 1, x, 1, 90)
		setPublicFrameSample(plane, 1, x, 2, 100)
		setPublicFrameSample(plane, 1, x, 3, 100)
	}
	result, err := internalloopfilter.Filter4BlockEdge(plane, 1, 8,
		internalparser.LoopFilterParams{LevelY: [2]uint8{0, 16}},
		internalparser.SegmentationParams{},
		internalparser.DeltaParams{},
		internalloopfilter.DeltaState{},
		internalloopfilter.Filter4Request{
			LevelRequest: internalloopfilter.LevelRequest{Plane: internalloopfilter.PlaneY, Edge: internalloopfilter.EdgeHorizontal, SegmentID: 0, RefFrame: 0},
			X:            0,
			Y:            2,
			Length:       4,
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	if !result.Applied || result.Level != 16 || result.Thresholds != (internalloopfilter.Thresholds{Limit: 16, BlockLimit: 52, HighEdgeVariance: 1}) {
		t.Fatalf("result=%+v", result)
	}
	if getPublicFrameSample(plane, 1, 0, 1) != 94 || getPublicFrameSample(plane, 1, 0, 2) != 96 {
		t.Fatalf("resolved edge did not filter expected samples")
	}

	for _, width := range []uint8{6, 8, 14} {
		req := internalloopfilter.FilterEdgeRequest{
			LevelRequest: internalloopfilter.LevelRequest{Plane: internalloopfilter.PlaneY, Edge: internalloopfilter.EdgeHorizontal, SegmentID: 0, RefFrame: 0},
			X:            0,
			Y:            16,
			Length:       32,
		}
		params := internalparser.LoopFilterParams{LevelY: [2]uint8{0, 16}}
		got := publicLoopFilterPatternPlane(32, 32)
		want := FramePlane{Pix: append([]byte(nil), got.Pix...), Stride: got.Stride, Width: got.Width, Height: got.Height}
		if _, err := internalloopfilter.FilterBlockEdge(want, 1, 8, params, internalparser.SegmentationParams{}, internalparser.DeltaParams{}, internalloopfilter.DeltaState{}, internalloopfilter.FilterBlockRequest{
			FilterEdgeRequest: req,
			Width:             width,
		}); err != nil {
			t.Fatalf("width=%d generic err=%v", width, err)
		}
		switch width {
		case 6:
			result, err = internalloopfilter.Filter6BlockEdge(got, 1, 8, params, internalparser.SegmentationParams{}, internalparser.DeltaParams{}, internalloopfilter.DeltaState{}, req)
		case 8:
			result, err = internalloopfilter.Filter8BlockEdge(got, 1, 8, params, internalparser.SegmentationParams{}, internalparser.DeltaParams{}, internalloopfilter.DeltaState{}, req)
		case 14:
			result, err = internalloopfilter.Filter14BlockEdge(got, 1, 8, params, internalparser.SegmentationParams{}, internalparser.DeltaParams{}, internalloopfilter.DeltaState{}, req)
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

func publicLoopFilterPatternPlane(width int, height int) FramePlane {
	plane := publicPredictionPlane(width, height, 1, width)
	for y := 0; y < plane.Height; y++ {
		for x := 0; x < plane.Width; x++ {
			setPublicFrameSample(plane, 1, x, y, uint16((x*11+y*13)&0xff))
		}
	}
	return plane
}
