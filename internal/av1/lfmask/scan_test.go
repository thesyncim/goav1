package lfmask

import (
	"testing"

	"github.com/thesyncim/goav1/internal/av1/tile"
)

type scanEdge struct {
	offset int
	width  int
}

func collectLuma(mask *[3][2]uint16, starty4, endy4 int) []scanEdge {
	var out []scanEdge
	ScanLuma(mask, starty4, endy4, func(o, w int) {
		out = append(out, scanEdge{o, w})
	})
	return out
}

func TestScanLumaStrengthLanes(t *testing.T) {
	var m [3][2]uint16
	m[0][0] = 1 << 0 // strength 0 at offset 0
	m[1][0] = 1 << 2 // strength 1 at offset 2
	m[2][0] = 1 << 5 // strength 2 at offset 5
	got := collectLuma(&m, 0, 16)
	want := []scanEdge{{0, 0}, {2, 1}, {5, 2}}
	if len(got) != len(want) {
		t.Fatalf("got %v want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("edge %d: got %v want %v", i, got[i], want[i])
		}
	}
}

func TestScanLumaHighLane(t *testing.T) {
	var m [3][2]uint16
	m[1][1] = 1 << 3 // strength 1 at high-lane offset 3 -> global offset 19
	got := collectLuma(&m, 0, 32)
	if len(got) != 1 || got[0] != (scanEdge{19, 1}) {
		t.Fatalf("high lane: got %v want [{19 1}]", got)
	}
}

func TestScanLumaHighLaneIgnoredWhenWindowShort(t *testing.T) {
	var m [3][2]uint16
	m[0][1] = 1 << 0 // high lane bit, but window ends at row 16
	got := collectLuma(&m, 0, 16)
	if len(got) != 0 {
		t.Fatalf("high lane should be ignored for endy4<=16, got %v", got)
	}
}

func TestScanLumaBottomSBRow(t *testing.T) {
	var m [3][2]uint16
	m[1][1] = 1 << 3 // high lane; bottom SB-row uses lane[1] alone
	got := collectLuma(&m, 16, 32)
	if len(got) != 1 || got[0] != (scanEdge{3, 1}) {
		t.Fatalf("bottom SB-row: got %v want [{3 1}]", got)
	}
}

func TestScanChromaTwoLanes(t *testing.T) {
	var m [2][2]uint16
	m[0][0] = 1 << 1 // width class 0 at offset 1
	m[1][0] = 1 << 4 // width class 1 at offset 4
	var out []scanEdge
	// 4:2:0 vertical direction: laneBits = 16>>ss_ver = 8, chroma region 16 tall.
	ScanChroma(&m, 0, 16, 8, func(o, w int) { out = append(out, scanEdge{o, w}) })
	want := []scanEdge{{1, 0}, {4, 1}}
	if len(out) != len(want) {
		t.Fatalf("got %v want %v", out, want)
	}
	for i := range want {
		if out[i] != want[i] {
			t.Fatalf("chroma edge %d: got %v want %v", i, out[i], want[i])
		}
	}
}

// TestBuildScanRoundTripInter16x16 ties the builder and scanner together: it
// builds the luma vertical mask for a split 16x16 inter block, then scans each
// column position and checks the recovered edges match the block+interior edges
// the builder wrote (left block edge at col 0, interior 8px edge at col 2, each
// spanning the 4 rows at strength class 1).
func TestBuildScanRoundTripInter16x16(t *testing.T) {
	var m FilterMask
	var txa [2][2][32][32]uint8
	a := make([]uint8, 4)
	l := make([]uint8, 4)
	for i := range a {
		a[i], l[i] = initY, initY
	}
	maskEdgesInter(&m.Y, 0, 0, 4, 4, 0, tile.TransformSize16x16, [2]uint16{1, 0}, a, l, &txa)

	type colEdges struct{ edges []scanEdge }
	byCol := map[int]colEdges{}
	for col := 0; col < 4; col++ {
		e := collectLuma(&m.Y[0][col], 0, 16)
		if len(e) > 0 {
			byCol[col] = colEdges{e}
		}
	}
	// Expect edges only at column 0 (left block edge) and column 2 (interior).
	if len(byCol) != 2 {
		t.Fatalf("expected edges at 2 columns, got %d: %v", len(byCol), byCol)
	}
	for _, col := range []int{0, 2} {
		ce, ok := byCol[col]
		if !ok {
			t.Fatalf("missing edges at column %d", col)
		}
		if len(ce.edges) != 4 {
			t.Fatalf("column %d: expected 4 row edges, got %v", col, ce.edges)
		}
		for r, e := range ce.edges {
			if e.offset != r || e.width != 1 {
				t.Fatalf("column %d row %d: got %v want {%d 1}", col, r, e, r)
			}
		}
	}
}
