package encoder

import (
	"testing"

	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/tile"
)

func TestSVTDimensionsRequire8x8(t *testing.T) {
	cases := []struct {
		w, h int
		want bool
	}{
		{64, 64, false},
		{64, 16, false}, // 720p bottom row: 16x16 covers it
		{64, 56, false}, // 1080p bottom row: the 32 half fits
		{64, 8, true},
		{8, 64, true},
		{40, 64, false},
		{24, 64, false},
	}
	for _, c := range cases {
		if got := svtDimensionsRequire8x8(c.w, c.h); got != c.want {
			t.Errorf("svtDimensionsRequire8x8(%d, %d) = %v, want %v", c.w, c.h, got, c.want)
		}
	}
}

func TestSVTInputResolution(t *testing.T) {
	cases := []struct {
		w, h, want int
	}{
		{320, 240, 0},
		{640, 360, 1}, // 0x38400: >= the 240p and < the 360p threshold
		{1280, 720, 3},
		{1920, 1080, 4},
		{3840, 2160, 5},
	}
	for _, c := range cases {
		if got := svtInputResolution(c.w, c.h); got != c.want {
			t.Errorf("svtInputResolution(%d, %d) = %d, want %d", c.w, c.h, got, c.want)
		}
	}
}

// TestSVTPD0PicksLargeBlocksOnStaticContent checks the VLPD0 tree end to end:
// a source identical to its reference has zero variance everywhere, so every
// split only adds partition rate and the 64x64 block stays whole.
func TestSVTPD0PicksLargeBlocksOnStaticContent(t *testing.T) {
	const w, h = 128, 128
	mk := func(fill func(i int) byte) SourceFrame420 {
		f := SourceFrame420{Y: make([]byte, w*h), U: make([]byte, w*h/4), V: make([]byte, w*h/4),
			YStride: w, ChromaStride: w / 2, Width: w, Height: h}
		for i := range f.Y {
			f.Y[i] = fill(i)
		}
		return f
	}
	flat := mk(func(int) byte { return 100 })
	var pc pframeCoder
	if err := pc.reset(120, 2, nil, parser.ColorConfig{BitDepth: 8, SubsamplingX: true, SubsamplingY: true}); err != nil {
		t.Fatal(err)
	}
	st := &pc.st
	st.prepareSVTPD0(2, 2)
	st.pd0Rates.estimate(&pc.partCDFs)
	st.setInterLambda()
	sb, err := st.svtPD0ForSB(flat, flat, 0, 0)
	if err != nil || sb == nil {
		t.Fatalf("svtPD0ForSB: %v %v", sb, err)
	}
	if sb.part64 != tile.PartitionNone {
		t.Fatalf("static SB partition %v, want None", sb.part64)
	}
}

// TestSVTPD0EdgeShapes checks the incomplete-SB shapes: a 16-row bottom SB
// cannot code the 64x32 PART_H block (it crosses the frame edge), so the
// 64x64 node splits and each 32x32 node, missing its lower half, codes the
// 32x16 PART_H block of a static source.
func TestSVTPD0EdgeShapes(t *testing.T) {
	const w, h = 128, 80
	f := SourceFrame420{Y: make([]byte, w*h), U: make([]byte, w*h/4), V: make([]byte, w*h/4),
		YStride: w, ChromaStride: w / 2, Width: w, Height: h}
	for i := range f.Y {
		f.Y[i] = 100
	}
	var pc pframeCoder
	if err := pc.reset(120, 2, nil, parser.ColorConfig{BitDepth: 8, SubsamplingX: true, SubsamplingY: true}); err != nil {
		t.Fatal(err)
	}
	st := &pc.st
	st.prepareSVTPD0(2, 2)
	st.pd0Rates.estimate(&pc.partCDFs)
	st.setInterLambda()
	sb, err := st.svtPD0ForSB(f, f, 0, 64)
	if err != nil || sb == nil {
		t.Fatalf("svtPD0ForSB: %v %v", sb, err)
	}
	if sb.part64 != tile.PartitionSplit || sb.part32[0] != tile.PartitionH || sb.part32[1] != tile.PartitionH {
		t.Fatalf("edge SB tree %v %v, want Split [H H ..]", sb.part64, sb.part32)
	}
}
