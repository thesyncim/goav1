package goav1

import (
	"errors"
	"testing"

	internaltile "github.com/thesyncim/goav1/internal/av1/tile"
)

func TestPublicTileCoeffEntropyContext(t *testing.T) {
	var ctx tileCoeffEntropyContext
	req := internaltile.CoeffContextRequest{
		Plane:      0,
		PlaneBlock: internaltile.BlockSize16x16,
		Size:       internaltile.TransformSize8x8,
		X4:         4,
		Y4:         5,
		VisibleW4:  2,
		VisibleH4:  2,
	}
	initial, err := ctx.TXBContext(req)
	if err != nil {
		t.Fatal(err)
	}
	if initial != (internaltile.TXBContext{TXBSkipContext: 1, DCSignContext: 0}) {
		t.Fatalf("initial context=%+v", initial)
	}
	if err := ctx.MarkTXB(req, internaltile.TXBDecodeResult{EOB: 3, CulLevel: 17}); err != nil {
		t.Fatal(err)
	}
	marked, err := ctx.TXBContext(req)
	if err != nil {
		t.Fatal(err)
	}
	if marked != (internaltile.TXBContext{TXBSkipContext: 4, DCSignContext: 2}) {
		t.Fatalf("marked context=%+v", marked)
	}
	if err := ctx.MarkTXB(req, internaltile.TXBDecodeResult{AllZero: true}); err != nil {
		t.Fatal(err)
	}
	cleared, err := ctx.TXBContext(req)
	if err != nil {
		t.Fatal(err)
	}
	if cleared != initial {
		t.Fatalf("all-zero context=%+v want %+v", cleared, initial)
	}
	if err := ctx.MarkTXB(req, internaltile.TXBDecodeResult{EOB: 1, CulLevel: 17}); err != nil {
		t.Fatal(err)
	}
	if err := ctx.ResetBlock(0, internaltile.BlockSize16x16, 4, 5); err != nil {
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
	var nilCtx *tileCoeffEntropyContext
	req := internaltile.CoeffContextRequest{
		Plane:      0,
		PlaneBlock: internaltile.BlockSize4x4,
		Size:       internaltile.TransformSize4x4,
		VisibleW4:  1,
		VisibleH4:  1,
	}
	if _, err := nilCtx.TXBContext(req); !errors.Is(err, internaltile.ErrInvalidDecodeState) {
		t.Fatalf("nil context err=%v want %v", err, internaltile.ErrInvalidDecodeState)
	}
	var ctx tileCoeffEntropyContext
	if err := ctx.MarkTXB(req, internaltile.TXBDecodeResult{EOB: 1, CulLevel: 24}); !errors.Is(err, internaltile.ErrInvalidDecodeState) {
		t.Fatalf("bad cul level err=%v want %v", err, internaltile.ErrInvalidDecodeState)
	}
	if err := ctx.ResetBlock(3, internaltile.BlockSize4x4, 0, 0); !errors.Is(err, internaltile.ErrInvalidDecodeState) {
		t.Fatalf("bad reset plane err=%v want %v", err, internaltile.ErrInvalidDecodeState)
	}
}

func TestPublicTileCoeffEntropyContextAllocs(t *testing.T) {
	var ctx tileCoeffEntropyContext
	req := internaltile.CoeffContextRequest{
		Plane:      0,
		PlaneBlock: internaltile.BlockSize16x16,
		Size:       internaltile.TransformSize8x8,
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
		err = ctx.MarkTXB(req, internaltile.TXBDecodeResult{EOB: 3, CulLevel: 17})
		if err != nil {
			return
		}
		if _, err = ctx.TXBContext(req); err != nil {
			return
		}
		err = ctx.ResetBlock(0, internaltile.BlockSize16x16, 4, 5)
	})
	if err != nil {
		t.Fatal(err)
	}
	if allocs != 0 {
		t.Fatalf("allocs=%v want 0", allocs)
	}
}
