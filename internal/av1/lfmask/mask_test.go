package lfmask

import (
	"testing"

	"github.com/thesyncim/goav1/internal/av1/loopfilter"
	"github.com/thesyncim/goav1/internal/av1/tile"
)

// dav1d initialises the per-tile edge contexts to tx_lpf_y=2 and tx_lpf_uv=1
// (src/decode.c:2401). Tests seed the a[]/l[] arrays with these values to model
// a block whose neighbours are outside the current tile.
const (
	initY  = 2
	initUV = 1
)

// TestCalcEIHMatchesThresholds asserts the ported dav1d_calc_eih E/I limits equal
// the decoder's existing libaom-derived ThresholdsForLevel across every level and
// sharpness, proving the two limit derivations agree byte-for-byte.
func TestCalcEIHMatchesThresholds(t *testing.T) {
	for sharp := 0; sharp <= int(loopfilter.MaxSharpness); sharp++ {
		lut := CalcEIH(sharp)
		for level := 0; level <= int(loopfilter.MaxLevel); level++ {
			th, err := loopfilter.ThresholdsForLevel(uint8(level), uint8(sharp))
			if err != nil {
				t.Fatalf("ThresholdsForLevel(%d,%d): %v", level, sharp, err)
			}
			if lut.I[level] != th.Limit {
				t.Fatalf("I[%d] sharp=%d: got %d want %d", level, sharp, lut.I[level], th.Limit)
			}
			if lut.E[level] != th.BlockLimit {
				t.Fatalf("E[%d] sharp=%d: got %d want %d", level, sharp, lut.E[level], th.BlockLimit)
			}
		}
	}
}

func TestDecompTxLeaf8x8(t *testing.T) {
	var txa [2][2][32][32]uint8
	decompTx(&txa, 0, 0, tile.TransformSize8x8, 0, 0, 0, [2]uint16{})
	for y := 0; y < 2; y++ {
		for x := 0; x < 2; x++ {
			if txa[0][0][y][x] != 1 {
				t.Fatalf("txa[0][0][%d][%d]=%d want 1 (lw)", y, x, txa[0][0][y][x])
			}
			if txa[1][0][y][x] != 1 {
				t.Fatalf("txa[1][0][%d][%d]=%d want 1 (lh)", y, x, txa[1][0][y][x])
			}
		}
		if txa[0][1][y][0] != 2 {
			t.Fatalf("txa[0][1][%d][0]=%d want 2 (x-step)", y, txa[0][1][y][0])
		}
	}
	for x := 0; x < 2; x++ {
		if txa[1][1][0][x] != 2 {
			t.Fatalf("txa[1][1][0][%d]=%d want 2 (y-step)", x, txa[1][1][0][x])
		}
	}
}

func TestDecompTxSplit8x8(t *testing.T) {
	var txa [2][2][32][32]uint8
	// Split the root 8x8 into four 4x4 leaves via the depth-0 split bit.
	decompTx(&txa, 0, 0, tile.TransformSize8x8, 0, 0, 0, [2]uint16{1, 0})
	for y := 0; y < 2; y++ {
		for x := 0; x < 2; x++ {
			if txa[0][0][y][x] != 0 || txa[1][0][y][x] != 0 {
				t.Fatalf("split leaf size at (%d,%d): lw=%d lh=%d want 0,0", x, y, txa[0][0][y][x], txa[1][0][y][x])
			}
			if txa[0][1][y][x] != 1 {
				t.Fatalf("split leaf x-step at (%d,%d)=%d want 1", x, y, txa[0][1][y][x])
			}
			if txa[1][1][y][x] != 1 {
				t.Fatalf("split leaf y-step at (%d,%d)=%d want 1", x, y, txa[1][1][y][x])
			}
		}
	}
}

func TestMaskEdgesIntra4x4(t *testing.T) {
	var m FilterMask
	a := []uint8{initY}
	l := []uint8{initY}
	maskEdgesIntra(&m.Y, 0, 0, 1, 1, tile.TransformSize4x4, a, l)
	// left edge: strength imin(twl4c=0, l=2)=0, bit y=0
	if m.Y[0][0][0][0] != 1 {
		t.Fatalf("left edge Y[0][0][0][0]=%d want 1", m.Y[0][0][0][0])
	}
	// top edge: strength imin(thl4c=0, a=2)=0, bit x=0
	if m.Y[1][0][0][0] != 1 {
		t.Fatalf("top edge Y[1][0][0][0]=%d want 1", m.Y[1][0][0][0])
	}
	// no inner edges for a 4x4 block
	for s := 0; s < 3; s++ {
		if s != 0 && m.Y[0][0][s][0] != 0 {
			t.Fatalf("unexpected vert strength %d bits", s)
		}
	}
	if a[0] != 0 || l[0] != 0 {
		t.Fatalf("context update: a=%d l=%d want 0,0", a[0], l[0])
	}
}

func TestMaskEdgesInter8x8(t *testing.T) {
	var m FilterMask
	var txa [2][2][32][32]uint8
	a := []uint8{initY, initY}
	l := []uint8{initY, initY}
	maskEdgesInter(&m.Y, 0, 0, 2, 2, 0, tile.TransformSize8x8, [2]uint16{}, a, l, &txa)
	// left edge rows 0,1 at strength imin(lw=1, l=2)=1 -> bits 0b11
	if m.Y[0][0][1][0] != 0b11 {
		t.Fatalf("left edge Y[0][0][1][0]=%b want 11", m.Y[0][0][1][0])
	}
	// top edge cols 0,1 at strength 1 -> bits 0b11
	if m.Y[1][0][1][0] != 0b11 {
		t.Fatalf("top edge Y[1][0][1][0]=%b want 11", m.Y[1][0][1][0])
	}
	// no inner edges: step spans the whole block
	if m.Y[0][1][0][0] != 0 || m.Y[0][1][1][0] != 0 {
		t.Fatalf("unexpected inner vertical edge at col 1")
	}
	for x := 0; x < 2; x++ {
		if a[x] != 1 {
			t.Fatalf("a[%d]=%d want 1", x, a[x])
		}
	}
	for y := 0; y < 2; y++ {
		if l[y] != 1 {
			t.Fatalf("l[%d]=%d want 1", y, l[y])
		}
	}
}

// TestMaskEdgesInter16x16SplitInner verifies that a 16x16 inter block whose
// transform tree is split into 8x8 sub-transforms records the interior 8x8 edge.
func TestMaskEdgesInter16x16SplitInner(t *testing.T) {
	var m FilterMask
	var txa [2][2][32][32]uint8
	a := make([]uint8, 4)
	l := make([]uint8, 4)
	for i := range a {
		a[i] = initY
		l[i] = initY
	}
	// Depth-0 split bit at tree position 0 turns 16x16 into four 8x8.
	maskEdgesInter(&m.Y, 0, 0, 4, 4, 0, tile.TransformSize16x16, [2]uint16{1, 0}, a, l, &txa)
	// Interior vertical edge at column x=2 (8px), strength imin(1,1)=1, rows 0..3.
	if m.Y[0][2][1][0] != 0b1111 {
		t.Fatalf("inner vert edge Y[0][2][1][0]=%b want 1111", m.Y[0][2][1][0])
	}
	// Interior horizontal edge at row y=2, strength 1, cols 0..3.
	if m.Y[1][2][1][0] != 0b1111 {
		t.Fatalf("inner horz edge Y[1][2][1][0]=%b want 1111", m.Y[1][2][1][0])
	}
	// Left block edge rows 0..3 at strength 1.
	if m.Y[0][0][1][0] != 0b1111 {
		t.Fatalf("left edge Y[0][0][1][0]=%b want 1111", m.Y[0][0][1][0])
	}
}

func TestMaskEdgesChroma420(t *testing.T) {
	var m FilterMask
	a := []uint8{initUV}
	l := []uint8{initUV}
	// 4x4 chroma unit (from an 8x8 luma block), 4:2:0.
	maskEdgesChroma(&m.UV, 0, 0, 1, 1, 0, tile.TransformSize4x4, a, l, 1, 1)
	if m.UV[0][0][0][0] != 1 {
		t.Fatalf("chroma left edge UV[0][0][0][0]=%d want 1", m.UV[0][0][0][0])
	}
	if m.UV[1][0][0][0] != 1 {
		t.Fatalf("chroma top edge UV[1][0][0][0]=%d want 1", m.UV[1][0][0][0])
	}
	if a[0] != 0 || l[0] != 0 {
		t.Fatalf("chroma context update a=%d l=%d want 0,0", a[0], l[0])
	}
}

// TestMaskEdgesInterHighColumn exercises the 32-bit split (sidx=1) path by
// placing a block in the high half of the 128px region so the edge bits land in
// the high uint16 lane.
func TestMaskEdgesInterHighColumn(t *testing.T) {
	var m FilterMask
	var txa [2][2][32][32]uint8
	a := make([]uint8, 4)
	l := make([]uint8, 4)
	for i := range a {
		a[i] = initY
		l[i] = initY
	}
	// by4 = 20 -> rows 20..23, all with mask >= 0x10000 => sidx=1.
	maskEdgesInter(&m.Y, 20, 0, 1, 4, 0, tile.TransformSize4x16, [2]uint16{}, a, l, &txa)
	// left edge in high lane: rows 20..23 -> bits (1<<4)..(1<<7) after >>16.
	want := uint16(0b11110000)
	if m.Y[0][0][0][1] != want {
		t.Fatalf("high-lane left edge Y[0][0][0][1]=%b want %b", m.Y[0][0][0][1], want)
	}
	if m.Y[0][0][0][0] != 0 {
		t.Fatalf("low lane should be empty, got %b", m.Y[0][0][0][0])
	}
}

// TestBuilderCreateIntraLevelCache verifies CreateIntra writes resolved levels
// into the level cache for the covered luma and chroma cells.
func TestBuilderCreateIntraLevelCache(t *testing.T) {
	var b Builder
	var m FilterMask
	stride := 8
	lc := LevelCache{Cells: make([][4]uint8, stride*8), Stride: stride}
	ay := make([]uint8, 32)
	ly := make([]uint8, 32)
	auv := make([]uint8, 32)
	luv := make([]uint8, 32)
	for i := range ay {
		ay[i], ly[i] = initY, initY
		auv[i], luv[i] = initUV, initUV
	}
	lv := Levels{YVert: 10, YHorz: 11, U: 12, V: 13}
	// 8x8 intra block at frame origin, 4:2:0.
	b.CreateIntra(&m, lc, lv, 0, 0, stride, 8, tile.BlockSize8x8, tile.TransformSize8x8, tile.TransformSize4x4, I420(), ay, ly, auv, luv)
	for y := 0; y < 2; y++ {
		for x := 0; x < 2; x++ {
			cell := lc.Cells[y*stride+x]
			if cell[0] != 10 || cell[1] != 11 {
				t.Fatalf("luma level cache (%d,%d)=%v want [10 11 ..]", x, y, cell)
			}
		}
	}
	// One chroma cell at (0,0).
	if lc.Cells[0][2] != 12 || lc.Cells[0][3] != 13 {
		t.Fatalf("chroma level cache = %v want [.. 12 13]", lc.Cells[0])
	}
}
