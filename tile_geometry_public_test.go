package goav1

import (
	"errors"
	"testing"

	internaltile "github.com/thesyncim/goav1/internal/av1/tile"
	internaltransform "github.com/thesyncim/goav1/internal/av1/transform"
)

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
