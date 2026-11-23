package goav1

import (
	"errors"
	"testing"

	internalparser "github.com/thesyncim/goav1/internal/av1/parser"
	internaltile "github.com/thesyncim/goav1/internal/av1/tile"
	internaltransform "github.com/thesyncim/goav1/internal/av1/transform"
)

func TestPublicTilePartitionGeometry(t *testing.T) {
	if got := internaltile.RootBlockLevel(true); got != internaltile.BlockLevel128x128 {
		t.Fatalf("128 root=%d", got)
	}
	if got := internaltile.RootBlockLevel(false); got != internaltile.BlockLevel64x64 {
		t.Fatalf("64 root=%d", got)
	}
	if got := tileBlockLevelHalfSize4x4(internaltile.BlockLevel64x64); got != 8 {
		t.Fatalf("half=%d want 8", got)
	}
	if got := tileBlockLevelSize4x4(internaltile.BlockLevel64x64); got != 16 {
		t.Fatalf("size=%d want 16", got)
	}

	dims, ok := tileBlockDimensionsOf(internaltile.BlockSize16x8)
	if !ok || dims != (internaltile.BlockDimensions{W4: 4, H4: 2, Log2W: 2, Log2H: 1}) {
		t.Fatalf("dims=%+v ok=%v", dims, ok)
	}
	if _, ok := tileBlockDimensionsOf(internaltile.BlockSize(99)); ok {
		t.Fatal("invalid block size reported dimensions")
	}

	if !tilePartitionValidForLevel(internaltile.PartitionH4, internaltile.BlockLevel64x64) {
		t.Fatal("H4 should be valid at 64x64 level")
	}
	if tilePartitionValidForLevel(internaltile.PartitionH4, internaltile.BlockLevel128x128) {
		t.Fatal("H4 should be invalid at 128x128 level")
	}
	a, b, ok := tilePartitionBlockSizes(internaltile.PartitionTTopSplit, internaltile.BlockLevel32x32)
	if !ok || a != internaltile.BlockSize16x16 || b != internaltile.BlockSize32x16 {
		t.Fatalf("partition sizes=%d,%d ok=%v", a, b, ok)
	}
	if _, _, ok := tilePartitionBlockSizes(internaltile.PartitionSplit, internaltile.BlockLevel32x32); ok {
		t.Fatal("recursive split should not report non-recursive block sizes")
	}
}

func TestPublicTileTransformGeometry(t *testing.T) {
	dims, ok := tileTransformDimensionsOf(internaltile.TransformSize32x16)
	if !ok || dims.W4 != 8 || dims.H4 != 4 || dims.Log2W != 3 || dims.Log2H != 2 || dims.Sub != internaltile.TransformSize16x16 {
		t.Fatalf("dims=%+v ok=%v", dims, ok)
	}
	pixels, err := tileTransformSizePixels(internaltile.TransformSize32x16)
	if err != nil {
		t.Fatal(err)
	}
	if pixels != (internaltransform.Size{Width: 32, Height: 16}) {
		t.Fatalf("pixels=%+v", pixels)
	}
	if _, err := tileTransformSizePixels(internaltile.TransformSize(99)); !errors.Is(err, internaltile.ErrInvalidDecodeState) {
		t.Fatalf("invalid transform err=%v want %v", err, internaltile.ErrInvalidDecodeState)
	}

	color420 := internalparser.ColorConfig{SubsamplingX: true, SubsamplingY: true}
	maxY, err := internaltile.MaxTransformSize(internaltile.BlockSize64x64, color420, 0)
	if err != nil {
		t.Fatal(err)
	}
	if maxY != internaltile.TransformSize64x64 {
		t.Fatalf("maxY=%d want 64x64", maxY)
	}
	maxUV, err := internaltile.MaxTransformSize(internaltile.BlockSize64x64, color420, 1)
	if err != nil {
		t.Fatal(err)
	}
	if maxUV != internaltile.TransformSize32x32 {
		t.Fatalf("maxUV=%d want 32x32", maxUV)
	}
	if _, err := internaltile.MaxTransformSize(internaltile.BlockSize64x64, internalparser.ColorConfig{MonoChrome: true}, 1); !errors.Is(err, internaltile.ErrInvalidDecodeState) {
		t.Fatalf("monochrome chroma err=%v want %v", err, internaltile.ErrInvalidDecodeState)
	}
}

func TestPublicTileExtendedTransformSets(t *testing.T) {
	square, err := internaltile.TransformSizeSquare(internaltile.TransformSize16x8)
	if err != nil {
		t.Fatal(err)
	}
	if square != internaltile.TransformSize8x8 {
		t.Fatalf("square=%d want 8x8", square)
	}
	up, err := internaltile.TransformSizeSquareUp(internaltile.TransformSize16x8)
	if err != nil {
		t.Fatal(err)
	}
	if up != internaltile.TransformSize16x16 {
		t.Fatalf("square up=%d want 16x16", up)
	}

	set, err := internaltile.ExtTXSetTypeFor(internaltile.TransformSize16x16, true, false)
	if err != nil {
		t.Fatal(err)
	}
	if set != internaltile.ExtTXSetDTT9IDTX1DDCT {
		t.Fatalf("set=%d want DTT9/IDTX/1D-DCT", set)
	}
	index, err := internaltile.ExtTXSetIndex(internaltile.TransformSize16x16, true, false)
	if err != nil {
		t.Fatal(err)
	}
	if index != 2 {
		t.Fatalf("index=%d want 2", index)
	}
	count, err := internaltile.ExtTXTypeCount(set)
	if err != nil {
		t.Fatal(err)
	}
	if count != 12 {
		t.Fatalf("count=%d want 12", count)
	}
	typ, err := internaltile.ExtTXTypeFromSymbol(set, 3)
	if err != nil {
		t.Fatal(err)
	}
	if typ != internaltransform.TypeDCTDCT {
		t.Fatalf("type=%d want DCT_DCT", typ)
	}
	symbol, err := internaltile.ExtTXSymbolForType(set, internaltransform.TypeDCTDCT)
	if err != nil {
		t.Fatal(err)
	}
	if symbol != 3 {
		t.Fatalf("symbol=%d want 3", symbol)
	}
	allowed, err := internaltile.ExtTXTypeAllowed(set, internaltransform.TypeVADST)
	if err != nil {
		t.Fatal(err)
	}
	if allowed {
		t.Fatal("VADST should be disallowed for the DTT9/IDTX/1D-DCT set")
	}
	if _, err := internaltile.ExtTXTypeFromSymbol(set, count); !errors.Is(err, internaltile.ErrInvalidDecodeState) {
		t.Fatalf("invalid symbol err=%v want %v", err, internaltile.ErrInvalidDecodeState)
	}
}

func TestPublicTileGeometryAllocs(t *testing.T) {
	color420 := internalparser.ColorConfig{SubsamplingX: true, SubsamplingY: true}
	var err error
	allocs := testing.AllocsPerRun(1000, func() {
		_ = internaltile.RootBlockLevel(true)
		_ = tileBlockLevelSize4x4(internaltile.BlockLevel64x64)
		_, _ = tileBlockDimensionsOf(internaltile.BlockSize32x16)
		_, _, _ = tilePartitionBlockSizes(internaltile.PartitionTTopSplit, internaltile.BlockLevel32x32)
		_, _ = tileTransformDimensionsOf(internaltile.TransformSize16x8)
		_, err = tileTransformSizePixels(internaltile.TransformSize16x8)
		_, err = internaltile.MaxTransformSize(internaltile.BlockSize64x64, color420, 1)
		_, err = internaltile.TransformSizeSquare(internaltile.TransformSize16x8)
		_, err = internaltile.TransformSizeSquareUp(internaltile.TransformSize16x8)
		_, err = internaltile.ExtTXSetTypeFor(internaltile.TransformSize16x16, true, false)
		_, err = internaltile.ExtTXSetIndex(internaltile.TransformSize16x16, true, false)
		_, err = internaltile.ExtTXTypeCount(internaltile.ExtTXSetDTT9IDTX1DDCT)
		_, err = internaltile.ExtTXTypeFromSymbol(internaltile.ExtTXSetDTT9IDTX1DDCT, 3)
		_, err = internaltile.ExtTXTypeAllowed(internaltile.ExtTXSetDTT9IDTX1DDCT, internaltransform.TypeDCTDCT)
		_, err = internaltile.ExtTXSymbolForType(internaltile.ExtTXSetDTT9IDTX1DDCT, internaltransform.TypeDCTDCT)
	})
	if err != nil {
		t.Fatal(err)
	}
	if allocs != 0 {
		t.Fatalf("allocs=%v want 0", allocs)
	}
}
