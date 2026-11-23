package goav1

import (
	"errors"
	"testing"

	internaltile "github.com/thesyncim/goav1/internal/av1/tile"
)

func TestPublicTileTransformContext(t *testing.T) {
	var ctx tileTransformContext
	selected, err := ctx.SelectedTransformContext(internaltile.TransformSize16x16, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if selected != 0 {
		t.Fatalf("initial selected context=%d want 0", selected)
	}
	if err := ctx.MarkTransformArea(0, 0, 4, 4, 2, 1, true); err != nil {
		t.Fatal(err)
	}
	selected, err = ctx.SelectedTransformContext(internaltile.TransformSize16x16, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if selected != 1 {
		t.Fatalf("selected context=%d want 1", selected)
	}

	category, partitionCtx, err := ctx.TransformPartitionContext(internaltile.TransformPartitionRequest{
		Size:     internaltile.BlockSize32x32,
		From:     internaltile.TransformSize32x32,
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
	category, partitionCtx, err = ctx.TransformPartitionContext(internaltile.TransformPartitionRequest{
		Size:     internaltile.BlockSize32x32,
		From:     internaltile.TransformSize32x32,
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
	selected, err = ctx.SelectedTransformContext(internaltile.TransformSize16x16, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if selected != 0 {
		t.Fatalf("reset selected context=%d want 0", selected)
	}

	var nilCtx *tileTransformContext
	if _, err := nilCtx.SelectedTransformContext(internaltile.TransformSize16x16, 0, 0); !errors.Is(err, internaltile.ErrInvalidDecodeState) {
		t.Fatalf("nil context err=%v want %v", err, internaltile.ErrInvalidDecodeState)
	}
}

func TestPublicTileTransformTreeReplay(t *testing.T) {
	req := internaltile.TransformTreeRequest{
		Size:      internaltile.BlockSize16x16,
		X4:        4,
		Y4:        5,
		VisibleW4: 4,
		VisibleH4: 4,
	}
	result := internaltile.TransformTreeResult{Y: internaltile.TransformSize8x8}
	var blocks [4]internaltile.TransformBlock
	count := 0
	err := forEachTileLumaTransformBlock(result, req, func(block internaltile.TransformBlock) error {
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
	want := [...]internaltile.TransformBlock{
		{X4: 4, Y4: 5, Size: internaltile.TransformSize8x8, VisibleW4: 2, VisibleH4: 2},
		{X4: 6, Y4: 5, Size: internaltile.TransformSize8x8, VisibleW4: 2, VisibleH4: 2},
		{X4: 4, Y4: 7, Size: internaltile.TransformSize8x8, VisibleW4: 2, VisibleH4: 2},
		{X4: 6, Y4: 7, Size: internaltile.TransformSize8x8, VisibleW4: 2, VisibleH4: 2},
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
	if err := forEachTileLumaTransformBlock(result, skipReq, func(internaltile.TransformBlock) error {
		count++
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if count != 0 {
		t.Fatalf("skip emitted %d blocks", count)
	}
	if err := forEachTileLumaTransformBlock(result, req, nil); !errors.Is(err, internaltile.ErrInvalidDecodeState) {
		t.Fatalf("nil visitor err=%v want %v", err, internaltile.ErrInvalidDecodeState)
	}
}

func TestPublicTileTransformContextAllocs(t *testing.T) {
	var ctx tileTransformContext
	req := internaltile.TransformPartitionRequest{
		Size: internaltile.BlockSize32x32,
		From: internaltile.TransformSize32x32,
	}
	treeReq := internaltile.TransformTreeRequest{Size: internaltile.BlockSize16x16, VisibleW4: 4, VisibleH4: 4}
	result := internaltile.TransformTreeResult{Y: internaltile.TransformSize8x8}
	var err error
	allocs := testing.AllocsPerRun(1000, func() {
		ctx.Reset()
		err = ctx.MarkTransformArea(0, 0, 8, 8, 3, 3, true)
		_, err = ctx.SelectedTransformContext(internaltile.TransformSize32x32, 0, 0)
		_, _, err = ctx.TransformPartitionContext(req)
		err = ctx.MarkTransform(internaltile.BlockSize16x16, 0, 0, internaltile.TransformSize8x8, true)
		err = forEachTileLumaTransformBlock(result, treeReq, func(internaltile.TransformBlock) error {
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
