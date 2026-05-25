package goav1_test

import (
	"errors"
	"testing"

	av1 "github.com/thesyncim/goav1"
)

func TestPublicTileCoeffEntropyContext(t *testing.T) {
	var ctx av1.TileCoeffEntropyContext
	req := av1.TileCoeffContextRequest{
		Plane:      0,
		PlaneBlock: av1.TileBlockSize16x16,
		Size:       av1.TileTransformSize8x8,
		X4:         4,
		Y4:         5,
		VisibleW4:  2,
		VisibleH4:  2,
	}
	initial, err := ctx.TXBContext(req)
	if err != nil {
		t.Fatal(err)
	}
	if initial != (av1.TileTXBContext{TXBSkipContext: 1, DCSignContext: 0}) {
		t.Fatalf("initial context=%+v", initial)
	}
	if err := ctx.MarkTXB(req, av1.TileTXBDecodeResult{EOB: 3, CulLevel: 17}); err != nil {
		t.Fatal(err)
	}
	marked, err := ctx.TXBContext(req)
	if err != nil {
		t.Fatal(err)
	}
	if marked != (av1.TileTXBContext{TXBSkipContext: 4, DCSignContext: 2}) {
		t.Fatalf("marked context=%+v", marked)
	}
	if err := ctx.MarkTXB(req, av1.TileTXBDecodeResult{AllZero: true}); err != nil {
		t.Fatal(err)
	}
	cleared, err := ctx.TXBContext(req)
	if err != nil {
		t.Fatal(err)
	}
	if cleared != initial {
		t.Fatalf("all-zero context=%+v want %+v", cleared, initial)
	}
	if err := ctx.MarkTXB(req, av1.TileTXBDecodeResult{EOB: 1, CulLevel: 17}); err != nil {
		t.Fatal(err)
	}
	if err := ctx.ResetBlock(0, av1.TileBlockSize16x16, 4, 5); err != nil {
		t.Fatal(err)
	}
	reset, err := ctx.TXBContext(req)
	if err != nil {
		t.Fatal(err)
	}
	if reset != initial {
		t.Fatalf("reset context=%+v want %+v", reset, initial)
	}
	ctx.Reset()
	resetAll, err := ctx.TXBContext(req)
	if err != nil {
		t.Fatal(err)
	}
	if resetAll != initial {
		t.Fatalf("context reset=%+v want %+v", resetAll, initial)
	}
}

func TestPublicTileCoeffEntropyContextRejectsInvalidInputs(t *testing.T) {
	var nilCtx *av1.TileCoeffEntropyContext
	req := av1.TileCoeffContextRequest{
		Plane:      0,
		PlaneBlock: av1.TileBlockSize4x4,
		Size:       av1.TileTransformSize4x4,
		VisibleW4:  1,
		VisibleH4:  1,
	}
	if _, err := nilCtx.TXBContext(req); !errors.Is(err, av1.ErrTileInvalidDecodeState) {
		t.Fatalf("nil context err=%v want %v", err, av1.ErrTileInvalidDecodeState)
	}
	var ctx av1.TileCoeffEntropyContext
	if err := ctx.MarkTXB(req, av1.TileTXBDecodeResult{EOB: 1, CulLevel: 24}); !errors.Is(err, av1.ErrTileInvalidDecodeState) {
		t.Fatalf("bad cul level err=%v want %v", err, av1.ErrTileInvalidDecodeState)
	}
	if err := ctx.ResetBlock(3, av1.TileBlockSize4x4, 0, 0); !errors.Is(err, av1.ErrTileInvalidDecodeState) {
		t.Fatalf("bad reset plane err=%v want %v", err, av1.ErrTileInvalidDecodeState)
	}
}

func TestPublicTileCoeffEntropyContextAllocs(t *testing.T) {
	var ctx av1.TileCoeffEntropyContext
	req := av1.TileCoeffContextRequest{
		Plane:      0,
		PlaneBlock: av1.TileBlockSize16x16,
		Size:       av1.TileTransformSize8x8,
		X4:         4,
		Y4:         5,
		VisibleW4:  2,
		VisibleH4:  2,
	}
	var err error
	allocs := testing.AllocsPerRun(1000, func() {
		ctx.Reset()
		if _, err = ctx.TXBContext(req); err != nil {
			return
		}
		err = ctx.MarkTXB(req, av1.TileTXBDecodeResult{EOB: 3, CulLevel: 17})
		if err != nil {
			return
		}
		if _, err = ctx.TXBContext(req); err != nil {
			return
		}
		err = ctx.ResetBlock(0, av1.TileBlockSize16x16, 4, 5)
	})
	if err != nil {
		t.Fatal(err)
	}
	if allocs != 0 {
		t.Fatalf("allocs=%v want 0", allocs)
	}
}
