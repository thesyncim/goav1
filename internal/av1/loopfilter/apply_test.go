package loopfilter

import (
	"bytes"
	"errors"
	"strconv"
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
	for x := range 4 {
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
	for x := range 4 {
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

func TestFilterBlockEdgeDispatchesResolvedWidths(t *testing.T) {
	for _, width := range []int32{4, 6, 8, 14} {
		plane := testPlane(32, 32, 1, 32)
		for y := 0; y < plane.Height; y++ {
			for x := 0; x < plane.Width; x++ {
				setSample(plane, 1, x, y, uint16((x*5+y*7)&0xff))
			}
		}
		want := cloneTestPlane(plane)
		thresholds, err := ThresholdsForLevel(16, 0)
		if err != nil {
			t.Fatal(err)
		}
		if err := FilterEdgeByWidth(width, want, 1, 8, EdgeHorizontal, 0, 16, 32, thresholds); err != nil {
			t.Fatalf("width=%d direct err=%v", width, err)
		}

		result, err := FilterBlockEdge(plane, 1, 8,
			parser.LoopFilterParams{LevelY: [2]uint8{0, 16}},
			parser.SegmentationParams{},
			parser.DeltaParams{},
			DeltaState{},
			FilterBlockRequest{
				FilterEdgeRequest: FilterEdgeRequest{
					LevelRequest: LevelRequest{Plane: PlaneY, Edge: EdgeHorizontal, SegmentID: 0, RefFrame: 0},
					X:            0,
					Y:            16,
					Length:       32,
				},
				Width: width,
			},
		)
		if err != nil {
			t.Fatalf("width=%d FilterBlockEdge err=%v", width, err)
		}
		if !result.Applied || result.Level != 16 || result.Thresholds != thresholds {
			t.Fatalf("width=%d result=%+v thresholds=%+v", width, result, thresholds)
		}
		if !bytes.Equal(plane.Pix, want.Pix) {
			t.Fatalf("width=%d dispatch output mismatch", width)
		}
	}
}

func TestSpecificBlockEdgeWrappersMatchGenericDispatcher(t *testing.T) {
	for _, width := range []int32{6, 8, 14} {
		plane := testPlane(32, 32, 1, 32)
		for y := 0; y < plane.Height; y++ {
			for x := 0; x < plane.Width; x++ {
				setSample(plane, 1, x, y, uint16((x*11+y*13)&0xff))
			}
		}
		want := cloneTestPlane(plane)
		req := FilterEdgeRequest{
			LevelRequest: LevelRequest{Plane: PlaneY, Edge: EdgeHorizontal, SegmentID: 0, RefFrame: 0},
			X:            0,
			Y:            16,
			Length:       32,
		}
		params := parser.LoopFilterParams{LevelY: [2]uint8{0, 16}}
		if _, err := FilterBlockEdge(want, 1, 8, params, parser.SegmentationParams{}, parser.DeltaParams{}, DeltaState{}, FilterBlockRequest{FilterEdgeRequest: req, Width: width}); err != nil {
			t.Fatalf("width=%d generic err=%v", width, err)
		}

		var result FilterResult
		var err error
		switch width {
		case 6:
			result, err = Filter6BlockEdge(plane, 1, 8, params, parser.SegmentationParams{}, parser.DeltaParams{}, DeltaState{}, req)
		case 8:
			result, err = Filter8BlockEdge(plane, 1, 8, params, parser.SegmentationParams{}, parser.DeltaParams{}, DeltaState{}, req)
		case 14:
			result, err = Filter14BlockEdge(plane, 1, 8, params, parser.SegmentationParams{}, parser.DeltaParams{}, DeltaState{}, req)
		}
		if err != nil {
			t.Fatalf("width=%d wrapper err=%v", width, err)
		}
		if !result.Applied || result.Level != 16 {
			t.Fatalf("width=%d result=%+v", width, result)
		}
		if !bytes.Equal(plane.Pix, want.Pix) {
			t.Fatalf("width=%d wrapper output mismatch", width)
		}
	}
}

func TestFilter4BlockEdgeSkipsZeroLevel(t *testing.T) {
	plane := testPlane(4, 4, 1, 4)
	for x := range 4 {
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
	for x := range 4 {
		if got := getSample(plane, 1, x, 1); got != 90 {
			t.Fatalf("p0 sample=%d want 90", got)
		}
		if got := getSample(plane, 1, x, 2); got != 100 {
			t.Fatalf("q0 sample=%d want 100", got)
		}
	}
}

func TestFilterBlockEdgeRejectsInvalidInputs(t *testing.T) {
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

	_, err = FilterBlockEdge(plane, 1, 8,
		parser.LoopFilterParams{LevelY: [2]uint8{16, 0}},
		parser.SegmentationParams{},
		parser.DeltaParams{},
		DeltaState{},
		FilterBlockRequest{
			FilterEdgeRequest: FilterEdgeRequest{LevelRequest: LevelRequest{Plane: PlaneY, Edge: EdgeVertical, SegmentID: 0, RefFrame: 0}, X: 2, Y: 0, Length: 4},
			Width:             5,
		},
	)
	if !errors.Is(err, ErrInvalidFilter) {
		t.Fatalf("invalid width err=%v want %v", err, ErrInvalidFilter)
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
		for y := range height {
			for x := range width {
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
				X:            int32(x),
				Y:            int32(y),
				Length:       int32(length),
			},
		)
		if err != nil {
			t.Fatal(err)
		}
		if result.Level > MaxLevel {
			t.Fatalf("level=%d", result.Level)
		}
		for y := range height {
			for x := range width {
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
	for b.Loop() {
		_, _ = Filter4BlockEdge(plane, 1, 8, params, parser.SegmentationParams{}, parser.DeltaParams{}, DeltaState{}, req)
	}
}

func BenchmarkFilterBlockEdge(b *testing.B) {
	plane := testPlane(64, 64, 1, 64)
	params := parser.LoopFilterParams{LevelY: [2]uint8{16, 16}}
	for _, width := range []int32{4, 6, 8, 14} {
		req := FilterBlockRequest{
			FilterEdgeRequest: FilterEdgeRequest{
				LevelRequest: LevelRequest{Plane: PlaneY, Edge: EdgeHorizontal, SegmentID: 0, RefFrame: 0},
				X:            0,
				Y:            32,
				Length:       64,
			},
			Width: width,
		}
		b.Run("width="+strconv.Itoa(int(width)), func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				_, _ = FilterBlockEdge(plane, 1, 8, params, parser.SegmentationParams{}, parser.DeltaParams{}, DeltaState{}, req)
			}
		})
	}
}

// TestFilterWideBlockEdgeAllocs pins every block-edge dispatcher (the
// width-specific wrappers and the generic FilterBlockEdge) to zero per-call
// allocations. The wrappers all funnel into filterBlockEdgeWidth, which
// resolves the level on the stack and dispatches to FilterEdgeByWidth; this
// test catches regressions in that fast path for every supported width.
func TestFilterWideBlockEdgeAllocs(t *testing.T) {
	plane := testPlane(64, 64, 1, 64)
	for y := 0; y < plane.Height; y++ {
		for x := 0; x < plane.Width; x++ {
			setSample(plane, 1, x, y, uint16((x*7+y*11)&0xff))
		}
	}
	params := parser.LoopFilterParams{LevelY: [2]uint8{16, 16}}
	edgeReq := FilterEdgeRequest{
		LevelRequest: LevelRequest{Plane: PlaneY, Edge: EdgeHorizontal, SegmentID: 0, RefFrame: 0},
		X:            0,
		Y:            32,
		Length:       64,
	}

	cases := []struct {
		name string
		call func() error
	}{
		{
			name: "Filter6BlockEdge",
			call: func() error {
				_, err := Filter6BlockEdge(plane, 1, 8, params, parser.SegmentationParams{}, parser.DeltaParams{}, DeltaState{}, edgeReq)
				return err
			},
		},
		{
			name: "Filter8BlockEdge",
			call: func() error {
				_, err := Filter8BlockEdge(plane, 1, 8, params, parser.SegmentationParams{}, parser.DeltaParams{}, DeltaState{}, edgeReq)
				return err
			},
		},
		{
			name: "Filter14BlockEdge",
			call: func() error {
				_, err := Filter14BlockEdge(plane, 1, 8, params, parser.SegmentationParams{}, parser.DeltaParams{}, DeltaState{}, edgeReq)
				return err
			},
		},
		{
			name: "FilterBlockEdge/4",
			call: func() error {
				_, err := FilterBlockEdge(plane, 1, 8, params, parser.SegmentationParams{}, parser.DeltaParams{}, DeltaState{}, FilterBlockRequest{FilterEdgeRequest: edgeReq, Width: 4})
				return err
			},
		},
		{
			name: "FilterBlockEdge/14",
			call: func() error {
				_, err := FilterBlockEdge(plane, 1, 8, params, parser.SegmentationParams{}, parser.DeltaParams{}, DeltaState{}, FilterBlockRequest{FilterEdgeRequest: edgeReq, Width: 14})
				return err
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if err := tc.call(); err != nil {
				t.Fatalf("warm-up err=%v", err)
			}
			allocs := testing.AllocsPerRun(1000, func() {
				if err := tc.call(); err != nil {
					t.Fatal(err)
				}
			})
			if allocs != 0 {
				t.Fatalf("%s allocated: %f", tc.name, allocs)
			}
		})
	}
}

// TestFilterEdgeByWidthAllocs covers the lower-level FilterEdgeByWidth used by
// callers that already hold a resolved level/thresholds pair.
func TestFilterEdgeByWidthAllocs(t *testing.T) {
	plane := testPlane(64, 64, 1, 64)
	for y := 0; y < plane.Height; y++ {
		for x := 0; x < plane.Width; x++ {
			setSample(plane, 1, x, y, uint16((x*7+y*11)&0xff))
		}
	}
	thresholds, err := ThresholdsForLevel(16, 0)
	if err != nil {
		t.Fatal(err)
	}
	for _, width := range []int32{4, 6, 8, 14} {
		t.Run("width="+strconv.Itoa(int(width)), func(t *testing.T) {
			if err := FilterEdgeByWidth(width, plane, 1, 8, EdgeHorizontal, 0, 32, 64, thresholds); err != nil {
				t.Fatalf("warm-up err=%v", err)
			}
			allocs := testing.AllocsPerRun(1000, func() {
				if err := FilterEdgeByWidth(width, plane, 1, 8, EdgeHorizontal, 0, 32, 64, thresholds); err != nil {
					t.Fatal(err)
				}
			})
			if allocs != 0 {
				t.Fatalf("FilterEdgeByWidth(%d) allocated: %f", width, allocs)
			}
		})
	}
}
