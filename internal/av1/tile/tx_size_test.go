package tile

import (
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/entropy"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/transform"
)

func TestTransformDimensionsAndMaxTablesMatchDav1d(t *testing.T) {
	dims, ok := TransformSize64x16.Dimensions()
	if !ok {
		t.Fatal("missing 64x16 transform dimensions")
	}
	wantDims := TransformDimensions{
		W4: 16, H4: 4,
		Log2W: 4, Log2H: 2,
		Min: 2, Max: 4,
		Sub: TransformSize32x16,
		Ctx: 3,
	}
	if dims != wantDims {
		t.Fatalf("64x16 dims=%+v want %+v", dims, wantDims)
	}

	tests := []struct {
		name string
		size TransformSize
		want transform.Size
	}{
		{name: "4x4", size: TransformSize4x4, want: transform.Size{Width: 4, Height: 4}},
		{name: "64x64", size: TransformSize64x64, want: transform.Size{Width: 64, Height: 64}},
		{name: "4x16", size: TransformSize4x16, want: transform.Size{Width: 4, Height: 16}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := tt.size.TransformSize()
			if err != nil {
				t.Fatal(err)
			}
			if got != tt.want {
				t.Fatalf("size=%+v want %+v", got, tt.want)
			}
		})
	}

	max, err := MaxTransformSize(BlockSize16x16, parser.ColorConfig{}, 0)
	if err != nil {
		t.Fatal(err)
	}
	if max != TransformSize16x16 {
		t.Fatalf("16x16 luma max=%d want %d", max, TransformSize16x16)
	}
	max, err = MaxTransformSize(BlockSize16x16, parser.ColorConfig{SubsamplingX: true, SubsamplingY: true}, 1)
	if err != nil {
		t.Fatal(err)
	}
	if max != TransformSize8x8 {
		t.Fatalf("16x16 4:2:0 chroma max=%d want %d", max, TransformSize8x8)
	}
	max, err = MaxTransformSize(BlockSize8x8, parser.ColorConfig{}, 1)
	if err != nil {
		t.Fatal(err)
	}
	if max != TransformSize8x8 {
		t.Fatalf("8x8 4:4:4 chroma max=%d want %d", max, TransformSize8x8)
	}
	if _, err := MaxTransformSize(BlockSize8x8, parser.ColorConfig{MonoChrome: true}, 1); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("monochrome chroma err=%v want %v", err, ErrInvalidDecodeState)
	}
}

func TestTransformCDFsInitDefaultMatchesDav1dAndLibaom(t *testing.T) {
	var cdfs TransformCDFs
	if err := cdfs.InitDefault(); err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name string
		cdf  *entropy.CDF
		want []uint16
	}{
		{name: "tx size cat0 ctx0", cdf: &cdfs.TxSize[0][0], want: []uint16{12800, 0, 0}},
		{name: "tx size cat1 ctx2", cdf: &cdfs.TxSize[1][2], want: []uint16{14091, 1920, 0, 0}},
		{name: "tx size cat3 ctx2", cdf: &cdfs.TxSize[3][2], want: []uint16{15965, 10009, 0, 0}},
		{name: "tx partition cat0 ctx0", cdf: &cdfs.TxPartition[0][0], want: []uint16{4187, 0, 0}},
		{name: "tx partition cat6 ctx2", cdf: &cdfs.TxPartition[6][2], want: []uint16{16680, 0, 0}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assertEntropyCDFValues(t, tt.cdf.Values(), tt.want)
		})
	}
}

func TestTransformContextsAndMarkMatchDav1d(t *testing.T) {
	var ctx BlockModeContext
	if got, err := ctx.SelectedTransformContextWithAvailability(TransformSize16x16, 0, 0, false, false); err != nil || got != 0 {
		t.Fatalf("empty selected tx ctx=%d err=%v want 0", got, err)
	}

	if err := ctx.MarkTransform(BlockSize16x16, 0, 0, TransformSize16x16, true); err != nil {
		t.Fatal(err)
	}
	if ctx.AboveTxIntra[0] != 2 || ctx.LeftTxIntra[0] != 2 || ctx.AboveTx[0] != 2 || ctx.LeftTx[0] != 2 {
		t.Fatalf("marked tx above_intra=%d left_intra=%d above=%d left=%d",
			ctx.AboveTxIntra[0], ctx.LeftTxIntra[0], ctx.AboveTx[0], ctx.LeftTx[0])
	}
	if got, err := ctx.SelectedTransformContextWithAvailability(TransformSize16x16, 0, 0, true, true); err != nil || got != 2 {
		t.Fatalf("marked selected tx ctx=%d err=%v want 2", got, err)
	}
	if err := ctx.MarkInter(BlockSize32x16, 4, 0, InterReferencesResult{Ref: [2]ReferenceFrame{ReferenceFrameLast, ReferenceFrameNone}}); err != nil {
		t.Fatal(err)
	}
	if got, err := ctx.SelectedTransformContextWithAvailability(TransformSize16x16, 4, 0, true, false); err != nil || got != 1 {
		t.Fatalf("inter above selected tx ctx=%d err=%v want 1", got, err)
	}

	ctx.AboveTx[0] = 0
	ctx.LeftTx[0] = 0
	category, context, err := ctx.TransformPartitionContext(TransformPartitionRequest{
		Size:  BlockSize16x16,
		From:  TransformSize16x16,
		Depth: 0,
	})
	if err != nil {
		t.Fatal(err)
	}
	if category != 4 || context != 2 {
		t.Fatalf("tx partition cat=%d ctx=%d want 4,2", category, context)
	}

	category, context, err = ctx.TransformPartitionContext(TransformPartitionRequest{
		Size:  BlockSize32x32,
		From:  TransformSize32x32,
		Depth: 1,
	})
	if err != nil {
		t.Fatal(err)
	}
	if category != 1 || context != 2 {
		t.Fatalf("tx partition depth1 cat=%d ctx=%d want 1,2", category, context)
	}
}

func TestReadSelectedTransformSize(t *testing.T) {
	var cdfs TransformCDFs
	if err := cdfs.InitDefault(); err != nil {
		t.Fatal(err)
	}
	var ctx BlockModeContext
	var state DecodeState

	if err := state.Reset([]byte{0xff}, Job{Offset: 0, Size: 1}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	tx, err := state.ReadSelectedTransformSize(&cdfs, &ctx, SelectedTransformRequest{
		Size:          BlockSize16x16,
		TransformMode: parser.TransformModeSwitchable,
		Lossless:      true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if tx != TransformSize4x4 || cdfs.TxSize[1][0].Values()[3] != 0 {
		t.Fatalf("lossless tx=%d cdf=%v", tx, cdfs.TxSize[1][0].Values())
	}

	if err := state.Reset([]byte{0xff}, Job{Offset: 0, Size: 1}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	tx, err = state.ReadSelectedTransformSize(&cdfs, &ctx, SelectedTransformRequest{
		Size:          BlockSize16x16,
		TransformMode: parser.TransformModeLargest,
	})
	if err != nil {
		t.Fatal(err)
	}
	if tx != TransformSize16x16 || cdfs.TxSize[1][0].Values()[3] != 0 {
		t.Fatalf("largest tx=%d cdf=%v", tx, cdfs.TxSize[1][0].Values())
	}

	if err := state.Reset([]byte{0x00}, Job{Offset: 0, Size: 1}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	tx, err = state.ReadSelectedTransformSize(&cdfs, &ctx, SelectedTransformRequest{
		Size:          BlockSize16x16,
		TransformMode: parser.TransformModeSwitchable,
	})
	if err != nil {
		t.Fatal(err)
	}
	if tx != TransformSize16x16 {
		t.Fatalf("switchable tx=%d want %d", tx, TransformSize16x16)
	}
	if got := cdfs.TxSize[1][0].Values()[3]; got != 1 {
		t.Fatalf("tx size cdf count=%d want 1", got)
	}
}

func TestReadTransformPartitionSplit(t *testing.T) {
	var cdfs TransformCDFs
	if err := cdfs.InitDefault(); err != nil {
		t.Fatal(err)
	}
	var ctx BlockModeContext
	var state DecodeState
	if err := state.Reset([]byte{0x00}, Job{Offset: 0, Size: 1}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}

	split, err := state.ReadTransformPartitionSplit(&cdfs, &ctx, TransformPartitionRequest{
		Size:  BlockSize16x16,
		From:  TransformSize16x16,
		Depth: 0,
	})
	if err != nil {
		t.Fatal(err)
	}
	if split {
		t.Fatal("zero payload split=true want false")
	}
	if got := cdfs.TxPartition[4][2].Values()[2]; got != 1 {
		t.Fatalf("tx partition cdf count=%d want 1", got)
	}

	before := cdfs.TxPartition[4][2]
	split, err = state.ReadTransformPartitionSplit(&cdfs, &ctx, TransformPartitionRequest{
		Size:  BlockSize16x16,
		From:  TransformSize16x16,
		Depth: 2,
	})
	if err != nil {
		t.Fatal(err)
	}
	if split || cdfs.TxPartition[4][2] != before {
		t.Fatalf("terminal split=%v cdf changed=%v", split, cdfs.TxPartition[4][2] != before)
	}
}

func TestTransformSizeRejectsInvalidInputs(t *testing.T) {
	var cdfs TransformCDFs
	if _, err := cdfs.TxSizeCDF(0, 0); !errors.Is(err, entropy.ErrInvalidCDF) {
		t.Fatalf("uninitialized tx size cdf err=%v want %v", err, entropy.ErrInvalidCDF)
	}
	if err := cdfs.InitDefault(); err != nil {
		t.Fatal(err)
	}
	if _, err := cdfs.TxSizeCDF(TransformSizeCategories, 0); !errors.Is(err, entropy.ErrInvalidCDF) {
		t.Fatalf("bad tx size cdf err=%v want %v", err, entropy.ErrInvalidCDF)
	}
	if _, err := cdfs.TxPartitionCDF(0, TransformPartitionContexts); !errors.Is(err, entropy.ErrInvalidCDF) {
		t.Fatalf("bad tx partition cdf err=%v want %v", err, entropy.ErrInvalidCDF)
	}
	if _, err := MaxTransformSize(blockSizeCount, parser.ColorConfig{}, 0); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("bad max tx block err=%v want %v", err, ErrInvalidDecodeState)
	}
	if _, err := TransformSize(transformSizeCount).TransformSize(); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("bad transform size err=%v want %v", err, ErrInvalidDecodeState)
	}

	var ctx BlockModeContext
	if _, err := ctx.SelectedTransformContextWithAvailability(TransformSize16x16, MaxBlockModeSlots, 0, true, true); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("bad selected tx ctx err=%v want %v", err, ErrInvalidDecodeState)
	}
	if _, _, err := ctx.TransformPartitionContext(TransformPartitionRequest{Size: BlockSize16x16, From: TransformSize4x4}); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("bad tx partition ctx err=%v want %v", err, ErrInvalidDecodeState)
	}
	if err := ctx.MarkTransform(blockSizeCount, 0, 0, TransformSize4x4, true); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("bad mark block err=%v want %v", err, ErrInvalidDecodeState)
	}
	if err := ctx.MarkTransform(BlockSize4x4, 0, 0, transformSizeCount, true); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("bad mark size err=%v want %v", err, ErrInvalidDecodeState)
	}

	var state DecodeState
	if err := state.Reset([]byte{0x00}, Job{Offset: 0, Size: 1}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	var nilState *DecodeState
	if _, err := nilState.ReadSelectedTransformSize(&cdfs, &ctx, SelectedTransformRequest{}); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("nil selected tx err=%v want %v", err, ErrInvalidDecodeState)
	}
	if _, err := state.ReadSelectedTransformSize(&cdfs, &ctx, SelectedTransformRequest{Size: BlockSize16x16, TransformMode: parser.TransformMode(99)}); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("bad selected tx mode err=%v want %v", err, ErrInvalidDecodeState)
	}
	if _, err := state.ReadTransformPartitionSplit(&cdfs, nil, TransformPartitionRequest{Size: BlockSize16x16, From: TransformSize16x16}); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("nil tx partition ctx err=%v want %v", err, ErrInvalidDecodeState)
	}
}

func TestTransformSizeAllocs(t *testing.T) {
	var cdfs TransformCDFs
	var ctx BlockModeContext
	var state DecodeState
	payload := []byte{0x00}

	allocs := testing.AllocsPerRun(1000, func() {
		if err := cdfs.InitDefault(); err != nil {
			t.Fatal(err)
		}
		ctx = BlockModeContext{}
		if err := state.Reset(payload, Job{Offset: 0, Size: len(payload)}, DecodeOptions{}); err != nil {
			t.Fatal(err)
		}
		tx, err := state.ReadSelectedTransformSize(&cdfs, &ctx, SelectedTransformRequest{
			Size:          BlockSize16x16,
			TransformMode: parser.TransformModeSwitchable,
		})
		if err != nil {
			t.Fatal(err)
		}
		if err := ctx.MarkTransform(BlockSize16x16, 0, 0, tx, true); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("transform-size decode allocated: %f", allocs)
	}
}

func FuzzReadSelectedTransformSize(f *testing.F) {
	f.Add([]byte{0x00}, uint8(BlockSize16x16), uint8(0), uint8(0), uint8(parser.TransformModeSwitchable), false, false)
	f.Add([]byte{0xff}, uint8(BlockSize32x32), uint8(3), uint8(4), uint8(parser.TransformModeLargest), false, true)
	f.Add([]byte{0xa5, 0x5a}, uint8(BlockSize8x8), uint8(8), uint8(7), uint8(parser.TransformMode4x4Only), true, false)

	f.Fuzz(func(t *testing.T, payload []byte, rawSize uint8, rawX uint8, rawY uint8, rawMode uint8, lossless bool, premark bool) {
		if len(payload) == 0 || len(payload) > 64 {
			return
		}
		size := BlockSize(rawSize % uint8(blockSizeCount))
		dims, ok := size.Dimensions()
		if !ok {
			t.Fatal("invalid generated block size")
		}
		x4 := int(rawX) % (MaxBlockModeSlots - int(dims.W4) + 1)
		y4 := int(rawY) % (MaxBlockModeSlots - int(dims.H4) + 1)
		mode := parser.TransformMode(rawMode % 3)

		var cdfs TransformCDFs
		if err := cdfs.InitDefault(); err != nil {
			t.Fatal(err)
		}
		var ctx BlockModeContext
		if premark {
			max, err := MaxTransformSize(size, parser.ColorConfig{}, 0)
			if err != nil {
				t.Fatal(err)
			}
			if err := ctx.MarkTransform(size, x4, y4, max, true); err != nil {
				t.Fatal(err)
			}
		}
		var state DecodeState
		if err := state.Reset(payload, Job{Offset: 0, Size: len(payload)}, DecodeOptions{}); err != nil {
			t.Fatal(err)
		}
		tx, err := state.ReadSelectedTransformSize(&cdfs, &ctx, SelectedTransformRequest{
			Size:          size,
			TransformMode: mode,
			Lossless:      lossless,
			X4:            x4,
			Y4:            y4,
		})
		if err != nil {
			t.Fatalf("ReadSelectedTransformSize err=%v size=%d mode=%d lossless=%v", err, size, mode, lossless)
		}
		if !tx.Valid() {
			t.Fatalf("tx=%d", tx)
		}
		if err := ctx.MarkTransform(size, x4, y4, tx, true); err != nil {
			t.Fatalf("MarkTransform err=%v", err)
		}
	})
}

func FuzzReadTransformPartitionSplit(f *testing.F) {
	f.Add([]byte{0x00}, uint8(0), uint8(0), uint8(0), false)
	f.Add([]byte{0xff}, uint8(3), uint8(7), uint8(5), true)
	f.Add([]byte{0xa5, 0x5a}, uint8(6), uint8(15), uint8(12), false)

	valid := [...]TransformPartitionRequest{
		{Size: BlockSize64x64, From: TransformSize64x64, Depth: 0},
		{Size: BlockSize32x32, From: TransformSize32x32, Depth: 1},
		{Size: BlockSize32x32, From: TransformSize32x32, Depth: 0},
		{Size: BlockSize16x16, From: TransformSize16x16, Depth: 1},
		{Size: BlockSize16x16, From: TransformSize16x16, Depth: 0},
		{Size: BlockSize8x8, From: TransformSize8x8, Depth: 1},
		{Size: BlockSize8x8, From: TransformSize8x8, Depth: 0},
	}

	f.Fuzz(func(t *testing.T, payload []byte, rawReq uint8, rawX uint8, rawY uint8, premark bool) {
		if len(payload) == 0 || len(payload) > 64 {
			return
		}
		req := valid[int(rawReq)%len(valid)]
		dims, ok := req.Size.Dimensions()
		if !ok {
			t.Fatal("invalid generated block size")
		}
		req.X4 = int(rawX) % (MaxBlockModeSlots - int(dims.W4) + 1)
		req.Y4 = int(rawY) % (MaxBlockModeSlots - int(dims.H4) + 1)

		var cdfs TransformCDFs
		if err := cdfs.InitDefault(); err != nil {
			t.Fatal(err)
		}
		var ctx BlockModeContext
		if premark {
			if err := ctx.MarkTransform(req.Size, req.X4, req.Y4, req.From, false); err != nil {
				t.Fatal(err)
			}
		}
		var state DecodeState
		if err := state.Reset(payload, Job{Offset: 0, Size: len(payload)}, DecodeOptions{}); err != nil {
			t.Fatal(err)
		}
		if _, err := state.ReadTransformPartitionSplit(&cdfs, &ctx, req); err != nil {
			t.Fatalf("ReadTransformPartitionSplit err=%v req=%+v", err, req)
		}
	})
}
