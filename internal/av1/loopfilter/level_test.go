package loopfilter

import (
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/parser"
)

func TestBaseLevelAndDeltaIndex(t *testing.T) {
	params := parser.LoopFilterParams{
		LevelY: [2]uint8{10, 20},
		LevelU: 30,
		LevelV: 40,
	}

	tests := []struct {
		name      string
		plane     Plane
		edge      Edge
		wantLevel uint8
		wantIndex int
	}{
		{name: "y vertical", plane: PlaneY, edge: EdgeVertical, wantLevel: 10, wantIndex: 0},
		{name: "y horizontal", plane: PlaneY, edge: EdgeHorizontal, wantLevel: 20, wantIndex: 1},
		{name: "u vertical", plane: PlaneU, edge: EdgeVertical, wantLevel: 30, wantIndex: 2},
		{name: "u horizontal", plane: PlaneU, edge: EdgeHorizontal, wantLevel: 30, wantIndex: 2},
		{name: "v vertical", plane: PlaneV, edge: EdgeVertical, wantLevel: 40, wantIndex: 3},
		{name: "v horizontal", plane: PlaneV, edge: EdgeHorizontal, wantLevel: 40, wantIndex: 3},
	}
	for _, tt := range tests {
		got, err := BaseLevel(params, tt.plane, tt.edge)
		if err != nil {
			t.Fatalf("%s BaseLevel err=%v", tt.name, err)
		}
		if got != tt.wantLevel {
			t.Fatalf("%s BaseLevel=%d want %d", tt.name, got, tt.wantLevel)
		}
		idx, err := DeltaIndex(tt.plane, tt.edge)
		if err != nil {
			t.Fatalf("%s DeltaIndex err=%v", tt.name, err)
		}
		if idx != tt.wantIndex {
			t.Fatalf("%s DeltaIndex=%d want %d", tt.name, idx, tt.wantIndex)
		}
	}

	if _, err := BaseLevel(params, Plane(9), EdgeVertical); !errors.Is(err, ErrInvalidFilter) {
		t.Fatalf("invalid plane BaseLevel err=%v want %v", err, ErrInvalidFilter)
	}
	if _, err := DeltaIndex(PlaneY, Edge(9)); !errors.Is(err, ErrInvalidFilter) {
		t.Fatalf("invalid edge DeltaIndex err=%v want %v", err, ErrInvalidFilter)
	}
}

func TestSelectDelta(t *testing.T) {
	multi := [DeltaCount]int8{-2, 3, 4, -5}
	got, err := SelectDelta(parser.DeltaParams{}, 7, multi, PlaneY, EdgeVertical)
	if err != nil || got != 0 {
		t.Fatalf("disabled SelectDelta=%d err=%v want 0 nil", got, err)
	}
	got, err = SelectDelta(parser.DeltaParams{DeltaLFPresent: true}, 7, multi, PlaneY, EdgeVertical)
	if err != nil || got != 7 {
		t.Fatalf("single SelectDelta=%d err=%v want 7 nil", got, err)
	}
	got, err = SelectDelta(parser.DeltaParams{DeltaLFPresent: true, DeltaLFMulti: true}, 7, multi, PlaneV, EdgeHorizontal)
	if err != nil || got != -5 {
		t.Fatalf("multi SelectDelta=%d err=%v want -5 nil", got, err)
	}
}

func TestSegmentDelta(t *testing.T) {
	seg := parser.SegmentationParams{Enabled: true}
	seg.Data.Segments[3] = parser.SegmentData{
		DeltaLFYV: -1,
		DeltaLFYH: 2,
		DeltaLFU:  3,
		DeltaLFV:  -4,
	}

	tests := []struct {
		name  string
		plane Plane
		edge  Edge
		want  int16
	}{
		{name: "y vertical", plane: PlaneY, edge: EdgeVertical, want: -1},
		{name: "y horizontal", plane: PlaneY, edge: EdgeHorizontal, want: 2},
		{name: "u", plane: PlaneU, edge: EdgeHorizontal, want: 3},
		{name: "v", plane: PlaneV, edge: EdgeVertical, want: -4},
	}
	for _, tt := range tests {
		got, err := SegmentDelta(seg, 3, tt.plane, tt.edge)
		if err != nil {
			t.Fatalf("%s SegmentDelta err=%v", tt.name, err)
		}
		if got != tt.want {
			t.Fatalf("%s SegmentDelta=%d want %d", tt.name, got, tt.want)
		}
	}

	got, err := SegmentDelta(parser.SegmentationParams{}, 3, PlaneY, EdgeVertical)
	if err != nil || got != 0 {
		t.Fatalf("disabled SegmentDelta=%d err=%v want 0 nil", got, err)
	}
	if _, err := SegmentDelta(seg, parser.MaxSegments, PlaneY, EdgeVertical); !errors.Is(err, ErrInvalidFilter) {
		t.Fatalf("invalid segment SegmentDelta err=%v want %v", err, ErrInvalidFilter)
	}
}

func TestResolveLevel(t *testing.T) {
	params := parser.LoopFilterParams{
		LevelY:              [2]uint8{40, 20},
		LevelU:              17,
		ModeRefDeltaEnabled: true,
		Deltas: parser.LoopFilterDeltas{
			Ref:  [parser.RefFrames]int8{1, 0, 0, 0, -2},
			Mode: [parser.LoopFilterModeDeltas]int8{0, 3},
		},
	}
	seg := parser.SegmentationParams{Enabled: true}
	seg.Data.Segments[2] = parser.SegmentData{
		DeltaLFYV: -5,
		DeltaLFYH: 2,
		DeltaLFU:  7,
		DeltaLFV:  9,
	}

	got, err := ResolveLevel(params, seg, LevelRequest{
		Plane:     PlaneY,
		Edge:      EdgeVertical,
		SegmentID: 2,
		RefFrame:  4,
		Mode:      ModeDeltaClassMotion,
		DeltaLF:   -3,
	})
	if err != nil {
		t.Fatal(err)
	}
	if got != 34 {
		t.Fatalf("resolved yv=%d want 34", got)
	}

	got, err = ResolveLevel(params, seg, LevelRequest{
		Plane:     PlaneY,
		Edge:      EdgeHorizontal,
		SegmentID: 2,
		RefFrame:  0,
		Mode:      ModeDeltaClassMotion,
	})
	if err != nil {
		t.Fatal(err)
	}
	if got != 23 {
		t.Fatalf("resolved intra=%d want 23", got)
	}

	params.LevelV = 0
	got, err = ResolveLevel(params, seg, LevelRequest{
		Plane:     PlaneV,
		Edge:      EdgeVertical,
		SegmentID: 2,
		RefFrame:  4,
		Mode:      ModeDeltaClassMotion,
		DeltaLF:   20,
	})
	if err != nil {
		t.Fatal(err)
	}
	if got != 0 {
		t.Fatalf("zero chroma level=%d want 0", got)
	}

	params.LevelY = [2]uint8{}
	params.LevelU = 63
	got, err = ResolveLevel(params, seg, LevelRequest{
		Plane:     PlaneU,
		Edge:      EdgeHorizontal,
		SegmentID: 2,
		RefFrame:  4,
		Mode:      ModeDeltaClassMotion,
		DeltaLF:   20,
	})
	if err != nil {
		t.Fatal(err)
	}
	if got != 0 {
		t.Fatalf("all-zero luma gate level=%d want 0", got)
	}
}

func TestResolveLevelRejectsInvalidInputs(t *testing.T) {
	params := parser.LoopFilterParams{LevelY: [2]uint8{1, 1}}
	tests := []struct {
		name string
		req  LevelRequest
	}{
		{name: "plane", req: LevelRequest{Plane: Plane(9), Edge: EdgeVertical, RefFrame: 0, SegmentID: 0}},
		{name: "edge", req: LevelRequest{Plane: PlaneY, Edge: Edge(9), RefFrame: 0, SegmentID: 0}},
		{name: "segment", req: LevelRequest{Plane: PlaneY, Edge: EdgeVertical, RefFrame: 0, SegmentID: parser.MaxSegments}},
		{name: "ref", req: LevelRequest{Plane: PlaneY, Edge: EdgeVertical, RefFrame: parser.RefFrames, SegmentID: 0}},
		{name: "mode", req: LevelRequest{Plane: PlaneY, Edge: EdgeVertical, RefFrame: 0, SegmentID: 0, Mode: ModeDeltaClass(9)}},
	}
	for _, tt := range tests {
		_, err := ResolveLevel(params, parser.SegmentationParams{}, tt.req)
		if !errors.Is(err, ErrInvalidFilter) {
			t.Fatalf("%s ResolveLevel err=%v want %v", tt.name, err, ErrInvalidFilter)
		}
	}
}

func TestThresholdsForLevel(t *testing.T) {
	tests := []struct {
		level     uint8
		sharpness uint8
		want      Thresholds
	}{
		{level: 0, sharpness: 0, want: Thresholds{Limit: 1, BlockLimit: 5, HighEdgeVariance: 0}},
		{level: 16, sharpness: 0, want: Thresholds{Limit: 16, BlockLimit: 52, HighEdgeVariance: 1}},
		{level: 63, sharpness: 7, want: Thresholds{Limit: 2, BlockLimit: 132, HighEdgeVariance: 3}},
	}
	for _, tt := range tests {
		got, err := ThresholdsForLevel(tt.level, tt.sharpness)
		if err != nil {
			t.Fatalf("ThresholdsForLevel(%d,%d) err=%v", tt.level, tt.sharpness, err)
		}
		if got != tt.want {
			t.Fatalf("ThresholdsForLevel(%d,%d)=%+v want %+v", tt.level, tt.sharpness, got, tt.want)
		}
	}
	if _, err := ThresholdsForLevel(MaxLevel+1, 0); !errors.Is(err, ErrInvalidFilter) {
		t.Fatalf("invalid level ThresholdsForLevel err=%v want %v", err, ErrInvalidFilter)
	}
	if _, err := ThresholdsForLevel(0, MaxSharpness+1); !errors.Is(err, ErrInvalidFilter) {
		t.Fatalf("invalid sharpness ThresholdsForLevel err=%v want %v", err, ErrInvalidFilter)
	}
}

func TestLevelHelpersAllocs(t *testing.T) {
	params := parser.LoopFilterParams{
		LevelY:              [2]uint8{32, 31},
		LevelU:              12,
		LevelV:              13,
		ModeRefDeltaEnabled: true,
		Deltas: parser.LoopFilterDeltas{
			Ref:  [parser.RefFrames]int8{1, 0, 0, 0, -1, 0, -1, -1},
			Mode: [parser.LoopFilterModeDeltas]int8{0, 1},
		},
	}
	seg := parser.SegmentationParams{Enabled: true}
	seg.Data.Segments[1].DeltaLFYV = -2
	req := LevelRequest{
		Plane:     PlaneY,
		Edge:      EdgeVertical,
		SegmentID: 1,
		RefFrame:  4,
		Mode:      ModeDeltaClassMotion,
		DeltaLF:   3,
	}

	allocs := testing.AllocsPerRun(1000, func() {
		if _, err := ResolveLevel(params, seg, req); err != nil {
			t.Fatal(err)
		}
		if _, err := ThresholdsForLevel(32, params.Sharpness); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("loopfilter helpers allocated: %f", allocs)
	}
}

func BenchmarkResolveLevel(b *testing.B) {
	params := parser.LoopFilterParams{
		LevelY:              [2]uint8{32, 31},
		LevelU:              12,
		LevelV:              13,
		ModeRefDeltaEnabled: true,
		Deltas: parser.LoopFilterDeltas{
			Ref:  [parser.RefFrames]int8{1, 0, 0, 0, -1, 0, -1, -1},
			Mode: [parser.LoopFilterModeDeltas]int8{0, 1},
		},
	}
	seg := parser.SegmentationParams{Enabled: true}
	seg.Data.Segments[1].DeltaLFYV = -2
	req := LevelRequest{
		Plane:     PlaneY,
		Edge:      EdgeVertical,
		SegmentID: 1,
		RefFrame:  4,
		Mode:      ModeDeltaClassMotion,
		DeltaLF:   3,
	}

	b.ReportAllocs()
	for b.Loop() {
		_, _ = ResolveLevel(params, seg, req)
	}
}

func BenchmarkThresholdsForLevel(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, _ = ThresholdsForLevel(uint8(i&MaxLevel), uint8(i&MaxSharpness))
	}
}
