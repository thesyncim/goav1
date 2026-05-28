package goav1_test

import (
	"errors"
	"testing"

	av1 "github.com/thesyncim/goav1"
)

func TestPublicTileTransformContext(t *testing.T) {
	var ctx av1.TileTransformContext
	selected, err := ctx.SelectedTransformContext(av1.TileTransformSize16x16, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if selected != 0 {
		t.Fatalf("initial selected context=%d want 0", selected)
	}
	if err := ctx.MarkTransformArea(0, 0, 4, 4, 2, 1, true); err != nil {
		t.Fatal(err)
	}
	selected, err = ctx.SelectedTransformContext(av1.TileTransformSize16x16, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if selected != 1 {
		t.Fatalf("selected context=%d want 1", selected)
	}

	category, partitionCtx, err := ctx.TransformPartitionContext(av1.TileTransformPartitionRequest{
		Size:     av1.TileBlockSize32x32,
		From:     av1.TileTransformSize32x32,
		Depth:    0,
		X4:       0,
		Y4:       0,
		HaveTop:  true,
		HaveLeft: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if category != 2 || partitionCtx != 2 {
		t.Fatalf("partition context=%d,%d want 2,2", category, partitionCtx)
	}
	if err := ctx.MarkTransformArea(0, 0, 8, 8, 3, 3, false); err != nil {
		t.Fatal(err)
	}
	category, partitionCtx, err = ctx.TransformPartitionContext(av1.TileTransformPartitionRequest{
		Size:     av1.TileBlockSize32x32,
		From:     av1.TileTransformSize32x32,
		Depth:    0,
		X4:       0,
		Y4:       0,
		HaveTop:  true,
		HaveLeft: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if category != 2 || partitionCtx != 0 {
		t.Fatalf("marked partition context=%d,%d want 2,0", category, partitionCtx)
	}
	ctx.Reset()
	selected, err = ctx.SelectedTransformContext(av1.TileTransformSize16x16, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if selected != 0 {
		t.Fatalf("reset selected context=%d want 0", selected)
	}

	var nilCtx *av1.TileTransformContext
	if _, err := nilCtx.SelectedTransformContext(av1.TileTransformSize16x16, 0, 0); !errors.Is(err, av1.ErrTileInvalidDecodeState) {
		t.Fatalf("nil context err=%v want %v", err, av1.ErrTileInvalidDecodeState)
	}
}

func TestPublicTileTransformTreeReplay(t *testing.T) {
	req := av1.TileTransformTreeRequest{
		Size:      av1.TileBlockSize16x16,
		X4:        4,
		Y4:        5,
		VisibleW4: 4,
		VisibleH4: 4,
	}
	result := av1.TileTransformTreeResult{Y: av1.TileTransformSize8x8}
	var blocks [4]av1.TileTransformBlock
	count := 0
	err := av1.ForEachTileLumaTransformBlock(result, req, func(block av1.TileTransformBlock) error {
		if count >= len(blocks) {
			t.Fatal("too many blocks")
		}
		blocks[count] = block
		count++
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	want := [...]av1.TileTransformBlock{
		{X4: 4, Y4: 5, Size: av1.TileTransformSize8x8, VisibleW4: 2, VisibleH4: 2},
		{X4: 6, Y4: 5, Size: av1.TileTransformSize8x8, VisibleW4: 2, VisibleH4: 2},
		{X4: 4, Y4: 7, Size: av1.TileTransformSize8x8, VisibleW4: 2, VisibleH4: 2},
		{X4: 6, Y4: 7, Size: av1.TileTransformSize8x8, VisibleW4: 2, VisibleH4: 2},
	}
	if count != len(want) {
		t.Fatalf("count=%d want %d", count, len(want))
	}
	for i := range want {
		if blocks[i] != want[i] {
			t.Fatalf("block %d=%+v want %+v", i, blocks[i], want[i])
		}
	}

	skipReq := req
	skipReq.SkipTransform = true
	count = 0
	if err := av1.ForEachTileLumaTransformBlock(result, skipReq, func(av1.TileTransformBlock) error {
		count++
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if count != 0 {
		t.Fatalf("skip emitted %d blocks", count)
	}
	if err := av1.ForEachTileLumaTransformBlock(result, req, nil); !errors.Is(err, av1.ErrTileInvalidDecodeState) {
		t.Fatalf("nil visitor err=%v want %v", err, av1.ErrTileInvalidDecodeState)
	}
}

func TestPublicTileTransformContextAllocs(t *testing.T) {
	var ctx av1.TileTransformContext
	req := av1.TileTransformPartitionRequest{
		Size: av1.TileBlockSize32x32,
		From: av1.TileTransformSize32x32,
	}
	treeReq := av1.TileTransformTreeRequest{Size: av1.TileBlockSize16x16, VisibleW4: 4, VisibleH4: 4}
	result := av1.TileTransformTreeResult{Y: av1.TileTransformSize8x8}
	var err error
	allocs := testing.AllocsPerRun(1000, func() {
		ctx.Reset()
		err = ctx.MarkTransformArea(0, 0, 8, 8, 3, 3, true)
		_, err = ctx.SelectedTransformContext(av1.TileTransformSize32x32, 0, 0)
		_, _, err = ctx.TransformPartitionContext(req)
		err = ctx.MarkTransform(av1.TileBlockSize16x16, 0, 0, av1.TileTransformSize8x8, true)
		err = av1.ForEachTileLumaTransformBlock(result, treeReq, func(av1.TileTransformBlock) error {
			return nil
		})
	})
	if err != nil {
		t.Fatal(err)
	}
	if allocs != 0 {
		t.Fatalf("allocs=%v want 0", allocs)
	}
}
