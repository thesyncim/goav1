package goav1_test

import (
	"errors"
	"testing"

	av1 "github.com/thesyncim/goav1"
)

func TestPublicDecodeTileLumaCoefficients(t *testing.T) {
	var cdfs av1.TileCoeffCDFs
	if err := av1.InitTileCoeffCDFsDefault(&cdfs, 0); err != nil {
		t.Fatal(err)
	}
	var state av1.TileDecodeState
	if err := av1.ResetTileDecodeState(&state, make([]byte, 16), av1.TileJob{Offset: 0, Size: 16}, av1.TileDecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	var ctx av1.TileCoeffEntropyContext
	var scratch av1.TileCoeffTreeScratch
	var blocks [4]av1.TileTransformBlock
	count := 0
	stats, err := av1.DecodeTileLumaCoefficients(&state, &cdfs, &ctx, &scratch, av1.TileLumaCoeffTreeRequest{
		TreeRequest: av1.TileTransformTreeRequest{
			Size:          av1.TileBlockSize8x8,
			VisibleW4:     2,
			VisibleH4:     2,
			Inter:         true,
			TransformMode: av1.TransformModeSwitchable,
		},
		Tree: av1.TileTransformTreeResult{
			Y:        av1.TileTransformSize8x8,
			Variable: true,
			Split:    [2]uint16{1},
		},
		Class: av1.TransformClass2D,
	}, func(block av1.TileLumaCoeffBlock) error {
		if count >= len(blocks) {
			t.Fatal("too many luma coefficient blocks")
		}
		blocks[count] = block.Block
		count++
		if len(block.Coeffs) != 16 || len(block.Scan) != 16 {
			t.Fatalf("buffer lens coeffs=%d scan=%d want 16", len(block.Coeffs), len(block.Scan))
		}
		if block.Result.EOB < 0 || block.Result.EOB > len(block.Coeffs) {
			t.Fatalf("invalid eob=%d coeffs=%d", block.Result.EOB, len(block.Coeffs))
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if stats.TXBs != 4 || stats.TXBs != stats.NonZero+stats.AllZero || count != 4 {
		t.Fatalf("stats=%+v count=%d", stats, count)
	}
	want := [...]av1.TileTransformBlock{
		{X4: 0, Y4: 0, Size: av1.TileTransformSize4x4, VisibleW4: 1, VisibleH4: 1},
		{X4: 1, Y4: 0, Size: av1.TileTransformSize4x4, VisibleW4: 1, VisibleH4: 1},
		{X4: 0, Y4: 1, Size: av1.TileTransformSize4x4, VisibleW4: 1, VisibleH4: 1},
		{X4: 1, Y4: 1, Size: av1.TileTransformSize4x4, VisibleW4: 1, VisibleH4: 1},
	}
	for i := range want {
		if blocks[i] != want[i] {
			t.Fatalf("block[%d]=%+v want %+v", i, blocks[i], want[i])
		}
	}
}

func TestPublicDecodeTileChromaCoefficients(t *testing.T) {
	var cdfs av1.TileCoeffCDFs
	if err := av1.InitTileCoeffCDFsDefault(&cdfs, 0); err != nil {
		t.Fatal(err)
	}
	var state av1.TileDecodeState
	if err := av1.ResetTileDecodeState(&state, make([]byte, 8), av1.TileJob{Offset: 0, Size: 8}, av1.TileDecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	var ctx av1.TileCoeffEntropyContext
	var scratch av1.TileCoeffTreeScratch
	var got av1.TileChromaCoeffBlock
	stats, err := av1.DecodeTileChromaCoefficients(&state, &cdfs, &ctx, &scratch, av1.TileChromaCoeffTreeRequest{
		TreeRequest: av1.TileTransformTreeRequest{
			Size:      av1.TileBlockSize16x16,
			X4:        5,
			Y4:        7,
			VisibleW4: 2,
			VisibleH4: 2,
		},
		Tree:  av1.TileTransformTreeResult{UV: av1.TileTransformSize8x8, HasUV: true},
		Color: av1.ColorConfig{SubsamplingX: true, SubsamplingY: true},
		Plane: 1,
		Class: av1.TransformClass2D,
	}, func(block av1.TileChromaCoeffBlock) error {
		got = block
		if len(block.Coeffs) != 64 || len(block.Scan) != 64 {
			t.Fatalf("buffer lens coeffs=%d scan=%d want 64", len(block.Coeffs), len(block.Scan))
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if stats.TXBs != 1 || stats.TXBs != stats.NonZero+stats.AllZero {
		t.Fatalf("stats=%+v", stats)
	}
	want := av1.TileTransformBlock{X4: 2, Y4: 3, Size: av1.TileTransformSize8x8, VisibleW4: 2, VisibleH4: 2}
	if got.Plane != 1 || got.Block != want {
		t.Fatalf("chroma block plane=%d block=%+v want plane=1 block=%+v", got.Plane, got.Block, want)
	}
}

func TestPublicDecodeTileCoefficientsRejectsInvalidInputs(t *testing.T) {
	var cdfs av1.TileCoeffCDFs
	var state av1.TileDecodeState
	var ctx av1.TileCoeffEntropyContext
	var scratch av1.TileCoeffTreeScratch
	valid := av1.TileLumaCoeffTreeRequest{
		TreeRequest: av1.TileTransformTreeRequest{Size: av1.TileBlockSize4x4, VisibleW4: 1, VisibleH4: 1},
		Tree:        av1.TileTransformTreeResult{Y: av1.TileTransformSize4x4},
		Class:       av1.TransformClass2D,
	}
	visitor := func(av1.TileLumaCoeffBlock) error { return nil }

	if err := av1.InitTileCoeffCDFsDefault(nil, 0); !errors.Is(err, av1.ErrTileInvalidDecodeState) {
		t.Fatalf("nil cdfs init err=%v want %v", err, av1.ErrTileInvalidDecodeState)
	}
	if err := av1.ResetTileDecodeState(nil, nil, av1.TileJob{}, av1.TileDecodeOptions{}); !errors.Is(err, av1.ErrTileInvalidDecodeState) {
		t.Fatalf("nil state reset err=%v want %v", err, av1.ErrTileInvalidDecodeState)
	}
	if _, err := av1.DecodeTileLumaCoefficients(&state, &cdfs, nil, &scratch, valid, visitor); !errors.Is(err, av1.ErrTileInvalidDecodeState) {
		t.Fatalf("nil ctx err=%v want %v", err, av1.ErrTileInvalidDecodeState)
	}
	if _, err := av1.DecodeTileLumaCoefficients(&state, nil, &ctx, &scratch, valid, visitor); !errors.Is(err, av1.ErrTileInvalidDecodeState) {
		t.Fatalf("nil cdfs err=%v want %v", err, av1.ErrTileInvalidDecodeState)
	}
	if _, err := av1.DecodeTileLumaCoefficients(&state, &cdfs, &ctx, nil, valid, visitor); !errors.Is(err, av1.ErrTileInvalidDecodeState) {
		t.Fatalf("nil scratch err=%v want %v", err, av1.ErrTileInvalidDecodeState)
	}
	if _, err := av1.DecodeTileLumaCoefficients(&state, &cdfs, &ctx, &scratch, valid, nil); !errors.Is(err, av1.ErrTileInvalidDecodeState) {
		t.Fatalf("nil visitor err=%v want %v", err, av1.ErrTileInvalidDecodeState)
	}
}

func TestPublicDecodeTileCoefficientsAllocs(t *testing.T) {
	payload := make([]byte, 16)
	job := av1.TileJob{Offset: 0, Size: len(payload)}
	req := av1.TileLumaCoeffTreeRequest{
		TreeRequest: av1.TileTransformTreeRequest{Size: av1.TileBlockSize4x4, VisibleW4: 1, VisibleH4: 1},
		Tree:        av1.TileTransformTreeResult{Y: av1.TileTransformSize4x4},
		Class:       av1.TransformClass2D,
	}
	var cdfs av1.TileCoeffCDFs
	var state av1.TileDecodeState
	var ctx av1.TileCoeffEntropyContext
	var scratch av1.TileCoeffTreeScratch
	var err error
	allocs := testing.AllocsPerRun(1000, func() {
		err = av1.InitTileCoeffCDFsDefault(&cdfs, 0)
		if err != nil {
			return
		}
		err = av1.ResetTileDecodeState(&state, payload, job, av1.TileDecodeOptions{})
		if err != nil {
			return
		}
		ctx.Reset()
		_, err = av1.DecodeTileLumaCoefficients(&state, &cdfs, &ctx, &scratch, req, func(av1.TileLumaCoeffBlock) error {
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
