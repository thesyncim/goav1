package tile

import (
	"slices"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/entropy"
)

func TestPaletteDefaultsMatchLibaomShape(t *testing.T) {
	var cdfs IntraModeCDFs
	if err := cdfs.InitDefault(); err != nil {
		t.Fatal(err)
	}
	for sizeCtx := 0; sizeCtx < PaletteBSizeContexts; sizeCtx++ {
		if _, err := cdfs.PaletteYSizeCDF(sizeCtx); err != nil {
			t.Fatalf("size ctx %d: %v", sizeCtx, err)
		}
		for modeCtx := 0; modeCtx < PaletteYModeContexts; modeCtx++ {
			if _, err := cdfs.PaletteYModeCDF(sizeCtx, modeCtx); err != nil {
				t.Fatalf("mode ctx %d/%d: %v", sizeCtx, modeCtx, err)
			}
		}
	}
	for colors := PaletteMinSize; colors <= PaletteMaxSize; colors++ {
		for ctx := 0; ctx < PaletteColorContexts; ctx++ {
			if _, err := cdfs.PaletteYColorIndexCDF(colors, ctx); err != nil {
				t.Fatalf("color ctx colors=%d ctx=%d: %v", colors, ctx, err)
			}
		}
	}
	if _, err := cdfs.PaletteYColorIndexCDF(PaletteMinSize-1, 0); err != entropy.ErrInvalidCDF {
		t.Fatalf("bad colors err=%v want %v", err, entropy.ErrInvalidCDF)
	}
}

func TestPaletteContextAndCacheUseMarkedNeighbors(t *testing.T) {
	var ctx BlockModeContext
	above := PaletteModeResult{YSize: 3, YColors: [PaletteMaxSize]uint16{11, 23, 47}}
	left := PaletteModeResult{YSize: 3, YColors: [PaletteMaxSize]uint16{23, 42, 99}}
	if err := ctx.MarkPaletteY(BlockSize16x16, 4, 0, above); err != nil {
		t.Fatal(err)
	}
	if err := ctx.MarkPaletteY(BlockSize16x16, 0, 4, left); err != nil {
		t.Fatal(err)
	}
	modeCtx, err := ctx.PaletteYModeContext(4, 4, true, true)
	if err != nil {
		t.Fatal(err)
	}
	if modeCtx != 2 {
		t.Fatalf("mode ctx=%d want 2", modeCtx)
	}
	var cache [2 * PaletteMaxSize]uint16
	n, err := ctx.PaletteYCache(4, 4, true, true, &cache)
	if err != nil {
		t.Fatal(err)
	}
	want := []uint16{11, 23, 42, 47, 99}
	if n != len(want) || !slices.Equal(cache[:n], want) {
		t.Fatalf("cache[:%d]=%v want %v", n, cache[:n], want)
	}
}

func TestPaletteColorIndexContextMatchesLibaomLookup(t *testing.T) {
	colorMap := []uint8{
		0, 1, 0, 0,
		1, 0, 0, 0,
		0, 0, 0, 0,
		0, 0, 0, 0,
	}
	ctx, order, err := paletteColorIndexContext(colorMap, 4, 1, 1, 3)
	if err != nil {
		t.Fatal(err)
	}
	if ctx != 3 {
		t.Fatalf("ctx=%d want 3", ctx)
	}
	if got := order[:3]; !slices.Equal(got, []uint8{1, 0, 2}) {
		t.Fatalf("order=%v want [1 0 2]", got)
	}
}
