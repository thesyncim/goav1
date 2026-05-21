package loopfilter

import (
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/parser"
)

func TestResolveBlockLevelSelectsDeltaState(t *testing.T) {
	params := parser.LoopFilterParams{
		LevelY: [2]uint8{40, 20},
		LevelU: 12,
	}
	state := DeltaState{
		FromBase: -3,
		Multi:    [DeltaCount]int8{1, 2, 4, 8},
	}

	got, err := ResolveBlockLevel(params, parser.SegmentationParams{},
		parser.DeltaParams{DeltaLFPresent: true},
		state,
		LevelRequest{Plane: PlaneY, Edge: EdgeVertical, SegmentID: 0, RefFrame: 0},
	)
	if err != nil {
		t.Fatal(err)
	}
	if got != 37 {
		t.Fatalf("single-delta level=%d want 37", got)
	}

	got, err = ResolveBlockLevel(params, parser.SegmentationParams{},
		parser.DeltaParams{DeltaLFPresent: true, DeltaLFMulti: true},
		state,
		LevelRequest{Plane: PlaneU, Edge: EdgeHorizontal, SegmentID: 0, RefFrame: 0},
	)
	if err != nil {
		t.Fatal(err)
	}
	if got != 16 {
		t.Fatalf("multi-delta level=%d want 16", got)
	}
}

func TestFilter4BlockEdgeAppliesResolvedLevel(t *testing.T) {
	plane := testPlane(4, 4, 1, 4)
	for x := 0; x < 4; x++ {
		setSample(plane, 1, x, 0, 90)
		setSample(plane, 1, x, 1, 90)
		setSample(plane, 1, x, 2, 100)
		setSample(plane, 1, x, 3, 100)
	}

	result, err := Filter4BlockEdge(plane, 1, 8,
		parser.LoopFilterParams{LevelY: [2]uint8{0, 16}},
		parser.SegmentationParams{},
		parser.DeltaParams{},
		DeltaState{},
		Filter4Request{
			LevelRequest: LevelRequest{Plane: PlaneY, Edge: EdgeHorizontal, SegmentID: 0, RefFrame: 0},
			X:            0,
			Y:            2,
			Length:       4,
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	if !result.Applied || result.Level != 16 || result.Thresholds != (Thresholds{Limit: 16, BlockLimit: 52, HighEdgeVariance: 1}) {
		t.Fatalf("result=%+v", result)
	}
	for x := 0; x < 4; x++ {
		if got := getSample(plane, 1, x, 0); got != 92 {
			t.Fatalf("p1 sample=%d want 92", got)
		}
		if got := getSample(plane, 1, x, 1); got != 94 {
			t.Fatalf("p0 sample=%d want 94", got)
		}
		if got := getSample(plane, 1, x, 2); got != 96 {
			t.Fatalf("q0 sample=%d want 96", got)
		}
		if got := getSample(plane, 1, x, 3); got != 98 {
			t.Fatalf("q1 sample=%d want 98", got)
		}
	}
}

func TestFilter4BlockEdgeSkipsZeroLevel(t *testing.T) {
	plane := testPlane(4, 4, 1, 4)
	for x := 0; x < 4; x++ {
		setSample(plane, 1, x, 1, 90)
		setSample(plane, 1, x, 2, 100)
	}
	result, err := Filter4BlockEdge(plane, 1, 8,
		parser.LoopFilterParams{},
		parser.SegmentationParams{},
		parser.DeltaParams{},
		DeltaState{},
		Filter4Request{
			LevelRequest: LevelRequest{Plane: PlaneY, Edge: EdgeVertical, SegmentID: 0, RefFrame: 0},
			X:            0,
			Y:            2,
			Length:       4,
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	if result.Applied || result.Level != 0 {
		t.Fatalf("result=%+v", result)
	}
	for x := 0; x < 4; x++ {
		if got := getSample(plane, 1, x, 1); got != 90 {
			t.Fatalf("p0 sample=%d want 90", got)
		}
		if got := getSample(plane, 1, x, 2); got != 100 {
			t.Fatalf("q0 sample=%d want 100", got)
		}
	}
}

func TestFilter4BlockEdgeRejectsInvalidInputs(t *testing.T) {
	plane := testPlane(4, 4, 1, 4)
	_, err := Filter4BlockEdge(plane, 1, 8,
		parser.LoopFilterParams{LevelY: [2]uint8{16, 0}, Sharpness: MaxSharpness + 1},
		parser.SegmentationParams{},
		parser.DeltaParams{},
		DeltaState{},
		Filter4Request{LevelRequest: LevelRequest{Plane: PlaneY, Edge: EdgeVertical, SegmentID: 0, RefFrame: 0}, X: 0, Y: 2, Length: 4},
	)
	if !errors.Is(err, ErrInvalidFilter) {
		t.Fatalf("invalid sharpness err=%v want %v", err, ErrInvalidFilter)
	}

	_, err = Filter4BlockEdge(plane, 1, 8,
		parser.LoopFilterParams{LevelY: [2]uint8{16, 0}},
		parser.SegmentationParams{},
		parser.DeltaParams{},
		DeltaState{},
		Filter4Request{LevelRequest: LevelRequest{Plane: Plane(9), Edge: EdgeVertical, SegmentID: 0, RefFrame: 0}, X: 0, Y: 2, Length: 4},
	)
	if !errors.Is(err, ErrInvalidFilter) {
		t.Fatalf("invalid level request err=%v want %v", err, ErrInvalidFilter)
	}
}

func TestFilter4BlockEdgeAllocs(t *testing.T) {
	plane := testPlane(64, 64, 1, 64)
	for y := 0; y < plane.Height; y++ {
		for x := 0; x < plane.Width; x++ {
			setSample(plane, 1, x, y, 100)
		}
	}
	params := parser.LoopFilterParams{LevelY: [2]uint8{16, 16}}
	req := Filter4Request{
		LevelRequest: LevelRequest{Plane: PlaneY, Edge: EdgeHorizontal, SegmentID: 0, RefFrame: 0},
		X:            0,
		Y:            32,
		Length:       64,
	}
	allocs := testing.AllocsPerRun(1000, func() {
		if _, err := Filter4BlockEdge(plane, 1, 8, params, parser.SegmentationParams{}, parser.DeltaParams{}, DeltaState{}, req); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("Filter4BlockEdge allocated: %f", allocs)
	}
}

func FuzzFilter4BlockEdge(f *testing.F) {
	f.Add(uint8(16), uint8(0), int8(0), uint8(0), uint8(8), uint8(8))
	f.Add(uint8(32), uint8(3), int8(-4), uint8(1), uint8(12), uint8(10))
	f.Fuzz(func(t *testing.T, rawLevel uint8, rawSharpness uint8, rawDelta int8, rawEdge uint8, rawWidth uint8, rawHeight uint8) {
		width := int(rawWidth%32) + 4
		height := int(rawHeight%32) + 4
		edge := Edge(rawEdge & 1)
		plane := testPlane(width, height, 1, width)
		for y := 0; y < height; y++ {
			for x := 0; x < width; x++ {
				setSample(plane, 1, x, y, uint16((x*17+y*19)&0xff))
			}
		}

		x := 2
		y := 2
		length := width - x
		if edge == EdgeVertical {
			length = height - y
		}
		result, err := Filter4BlockEdge(plane, 1, 8,
			parser.LoopFilterParams{LevelY: [2]uint8{rawLevel % (MaxLevel + 1), rawLevel % (MaxLevel + 1)}, Sharpness: rawSharpness & MaxSharpness},
			parser.SegmentationParams{},
			parser.DeltaParams{DeltaLFPresent: true},
			DeltaState{FromBase: rawDelta},
			Filter4Request{
				LevelRequest: LevelRequest{Plane: PlaneY, Edge: edge, SegmentID: 0, RefFrame: 0},
				X:            x,
				Y:            y,
				Length:       length,
			},
		)
		if err != nil {
			t.Fatal(err)
		}
		if result.Level > MaxLevel {
			t.Fatalf("level=%d", result.Level)
		}
		for y := 0; y < height; y++ {
			for x := 0; x < width; x++ {
				if got := getSample(plane, 1, x, y); got > 0xff {
					t.Fatalf("sample(%d,%d)=%d", x, y, got)
				}
			}
		}
	})
}

func BenchmarkFilter4BlockEdge(b *testing.B) {
	plane := testPlane(64, 64, 1, 64)
	params := parser.LoopFilterParams{LevelY: [2]uint8{16, 16}}
	req := Filter4Request{
		LevelRequest: LevelRequest{Plane: PlaneY, Edge: EdgeHorizontal, SegmentID: 0, RefFrame: 0},
		X:            0,
		Y:            32,
		Length:       64,
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, _ = Filter4BlockEdge(plane, 1, 8, params, parser.SegmentationParams{}, parser.DeltaParams{}, DeltaState{}, req)
	}
}
