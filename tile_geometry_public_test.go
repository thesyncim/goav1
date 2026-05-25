package goav1_test

import (
	"errors"
	"testing"

	av1 "github.com/thesyncim/goav1"
)

func TestPublicTilePartitionGeometry(t *testing.T) {
	if got := av1.TileRootBlockLevel(true); got != av1.TileBlockLevel128x128 {
		t.Fatalf("128 root=%d", got)
	}
	if got := av1.TileRootBlockLevel(false); got != av1.TileBlockLevel64x64 {
		t.Fatalf("64 root=%d", got)
	}
	if got := av1.TileBlockLevelHalfSize4x4(av1.TileBlockLevel64x64); got != 8 {
		t.Fatalf("half=%d want 8", got)
	}
	if got := av1.TileBlockLevelSize4x4(av1.TileBlockLevel64x64); got != 16 {
		t.Fatalf("size=%d want 16", got)
	}

	dims, ok := av1.TileBlockDimensionsOf(av1.TileBlockSize16x8)
	if !ok || dims != (av1.TileBlockDimensions{W4: 4, H4: 2, Log2W: 2, Log2H: 1}) {
		t.Fatalf("dims=%+v ok=%v", dims, ok)
	}
	if _, ok := av1.TileBlockDimensionsOf(av1.TileBlockSize(99)); ok {
		t.Fatal("invalid block size reported dimensions")
	}

	if !av1.TilePartitionValidForLevel(av1.TilePartitionH4, av1.TileBlockLevel64x64) {
		t.Fatal("H4 should be valid at 64x64 level")
	}
	if av1.TilePartitionValidForLevel(av1.TilePartitionH4, av1.TileBlockLevel128x128) {
		t.Fatal("H4 should be invalid at 128x128 level")
	}
	a, b, ok := av1.TilePartitionBlockSizes(av1.TilePartitionTTopSplit, av1.TileBlockLevel32x32)
	if !ok || a != av1.TileBlockSize16x16 || b != av1.TileBlockSize32x16 {
		t.Fatalf("partition sizes=%d,%d ok=%v", a, b, ok)
	}
	if _, _, ok := av1.TilePartitionBlockSizes(av1.TilePartitionSplit, av1.TileBlockLevel32x32); ok {
		t.Fatal("recursive split should not report non-recursive block sizes")
	}
}

func TestPublicTileTransformGeometry(t *testing.T) {
	dims, ok := av1.TileTransformDimensionsOf(av1.TileTransformSize32x16)
	if !ok || dims.W4 != 8 || dims.H4 != 4 || dims.Log2W != 3 || dims.Log2H != 2 || dims.Sub != av1.TileTransformSize16x16 {
		t.Fatalf("dims=%+v ok=%v", dims, ok)
	}
	pixels, err := av1.TileTransformSizePixels(av1.TileTransformSize32x16)
	if err != nil {
		t.Fatal(err)
	}
	if pixels != (av1.TransformSize{Width: 32, Height: 16}) {
		t.Fatalf("pixels=%+v", pixels)
	}
	if _, err := av1.TileTransformSizePixels(av1.TileTransformSize(99)); !errors.Is(err, av1.ErrTileInvalidDecodeState) {
		t.Fatalf("invalid transform err=%v want %v", err, av1.ErrTileInvalidDecodeState)
	}

	color420 := av1.ColorConfig{SubsamplingX: true, SubsamplingY: true}
	maxY, err := av1.MaxTileTransformSize(av1.TileBlockSize64x64, color420, 0)
	if err != nil {
		t.Fatal(err)
	}
	if maxY != av1.TileTransformSize64x64 {
		t.Fatalf("maxY=%d want 64x64", maxY)
	}
	maxUV, err := av1.MaxTileTransformSize(av1.TileBlockSize64x64, color420, 1)
	if err != nil {
		t.Fatal(err)
	}
	if maxUV != av1.TileTransformSize32x32 {
		t.Fatalf("maxUV=%d want 32x32", maxUV)
	}
	if _, err := av1.MaxTileTransformSize(av1.TileBlockSize64x64, av1.ColorConfig{MonoChrome: true}, 1); !errors.Is(err, av1.ErrTileInvalidDecodeState) {
		t.Fatalf("monochrome chroma err=%v want %v", err, av1.ErrTileInvalidDecodeState)
	}
}

func TestPublicTileExtendedTransformSets(t *testing.T) {
	square, err := av1.TileTransformSizeSquare(av1.TileTransformSize16x8)
	if err != nil {
		t.Fatal(err)
	}
	if square != av1.TileTransformSize8x8 {
		t.Fatalf("square=%d want 8x8", square)
	}
	up, err := av1.TileTransformSizeSquareUp(av1.TileTransformSize16x8)
	if err != nil {
		t.Fatal(err)
	}
	if up != av1.TileTransformSize16x16 {
		t.Fatalf("square up=%d want 16x16", up)
	}

	set, err := av1.TileExtTXSetTypeFor(av1.TileTransformSize16x16, true, false)
	if err != nil {
		t.Fatal(err)
	}
	if set != av1.TileExtTXSetDTT9IDTX1DDCT {
		t.Fatalf("set=%d want DTT9/IDTX/1D-DCT", set)
	}
	index, err := av1.TileExtTXSetIndex(av1.TileTransformSize16x16, true, false)
	if err != nil {
		t.Fatal(err)
	}
	if index != 2 {
		t.Fatalf("index=%d want 2", index)
	}
	count, err := av1.TileExtTXTypeCount(set)
	if err != nil {
		t.Fatal(err)
	}
	if count != 12 {
		t.Fatalf("count=%d want 12", count)
	}
	typ, err := av1.TileExtTXTypeFromSymbol(set, 3)
	if err != nil {
		t.Fatal(err)
	}
	if typ != av1.TransformTypeDCTDCT {
		t.Fatalf("type=%d want DCT_DCT", typ)
	}
	symbol, err := av1.TileExtTXSymbolForType(set, av1.TransformTypeDCTDCT)
	if err != nil {
		t.Fatal(err)
	}
	if symbol != 3 {
		t.Fatalf("symbol=%d want 3", symbol)
	}
	allowed, err := av1.TileExtTXTypeAllowed(set, av1.TransformTypeVADST)
	if err != nil {
		t.Fatal(err)
	}
	if allowed {
		t.Fatal("VADST should be disallowed for the DTT9/IDTX/1D-DCT set")
	}
	if _, err := av1.TileExtTXTypeFromSymbol(set, count); !errors.Is(err, av1.ErrTileInvalidDecodeState) {
		t.Fatalf("invalid symbol err=%v want %v", err, av1.ErrTileInvalidDecodeState)
	}
}

func TestPublicTileGeometryAllocs(t *testing.T) {
	color420 := av1.ColorConfig{SubsamplingX: true, SubsamplingY: true}
	var err error
	allocs := testing.AllocsPerRun(1000, func() {
		_ = av1.TileRootBlockLevel(true)
		_ = av1.TileBlockLevelSize4x4(av1.TileBlockLevel64x64)
		_, _ = av1.TileBlockDimensionsOf(av1.TileBlockSize32x16)
		_, _, _ = av1.TilePartitionBlockSizes(av1.TilePartitionTTopSplit, av1.TileBlockLevel32x32)
		_, _ = av1.TileTransformDimensionsOf(av1.TileTransformSize16x8)
		_, err = av1.TileTransformSizePixels(av1.TileTransformSize16x8)
		_, err = av1.MaxTileTransformSize(av1.TileBlockSize64x64, color420, 1)
		_, err = av1.TileTransformSizeSquare(av1.TileTransformSize16x8)
		_, err = av1.TileTransformSizeSquareUp(av1.TileTransformSize16x8)
		_, err = av1.TileExtTXSetTypeFor(av1.TileTransformSize16x16, true, false)
		_, err = av1.TileExtTXSetIndex(av1.TileTransformSize16x16, true, false)
		_, err = av1.TileExtTXTypeCount(av1.TileExtTXSetDTT9IDTX1DDCT)
		_, err = av1.TileExtTXTypeFromSymbol(av1.TileExtTXSetDTT9IDTX1DDCT, 3)
		_, err = av1.TileExtTXTypeAllowed(av1.TileExtTXSetDTT9IDTX1DDCT, av1.TransformTypeDCTDCT)
		_, err = av1.TileExtTXSymbolForType(av1.TileExtTXSetDTT9IDTX1DDCT, av1.TransformTypeDCTDCT)
	})
	if err != nil {
		t.Fatal(err)
	}
	if allocs != 0 {
		t.Fatalf("allocs=%v want 0", allocs)
	}
}
