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
	if err := ctx.MarkInter(BlockSize32x16, 4, 0, InterReferencesResult{Ref: [2]ReferenceFrame{ReferenceFrameLast, ReferenceFrameNone}}, true); err != nil {
		t.Fatal(err)
	}
	if got, err := ctx.SelectedTransformContextWithAvailability(TransformSize16x16, 4, 0, true, false); err != nil || got != 1 {
		t.Fatalf("inter above selected tx ctx=%d err=%v want 1", got, err)
	}

	ctx.AboveTx[0] = 0
	ctx.LeftTx[0] = 0
	category, context, err := ctx.TransformPartitionContext(TransformPartitionRequest{
		Size:     BlockSize16x16,
		From:     TransformSize16x16,
		Depth:    0,
		HaveTop:  true,
		HaveLeft: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if category != 4 || context != 2 {
		t.Fatalf("tx partition cat=%d ctx=%d want 4,2", category, context)
	}

	category, context, err = ctx.TransformPartitionContext(TransformPartitionRequest{
		Size:     BlockSize32x32,
		From:     TransformSize32x32,
		Depth:    1,
		HaveTop:  true,
		HaveLeft: true,
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

// TestSelectedTransformContextSnapshotForInterFrameIntraBlock pins the
// libaom-aligned get_tx_size_context() neighbor-snapshot behaviour for an
// intra block decoded inside an inter frame. Before the fix, the snapshot was
// gated on AllowIntrabc, so inter frames read the current block's
// just-written AboveIntra/LeftIntra slots and selected the wrong tx_size
// CDF context (which then adapted divergently and broke frame 1+ MD5s).
func TestSelectedTransformContextSnapshotForInterFrameIntraBlock(t *testing.T) {
	var ctx BlockModeContext

	// Seed the above neighbor at column 4 as an INTER block (AboveIntra=0)
	// with block-size 16x16, and the left neighbor at row 0 as an INTER
	// block (LeftIntra=0) with block-size 16x16. tx_size_context() for an
	// 8x8 intra block at (x4=4, y4=0) should evaluate the inter-neighbor
	// branch: above = block_size_wide(16) >= max_tx_wide(8) → 1, and
	// left = block_size_high(16) >= max_tx_high(8) → 1, giving ctx=2.
	ctx.AboveIntra[4] = 0
	ctx.AboveBlockSize[4] = BlockSize16x16
	ctx.LeftIntra[0] = 0
	ctx.LeftBlockSize[0] = BlockSize16x16

	gotPre, err := ctx.SelectedTransformContextWithAvailability(TransformSize8x8, 4, 0, true, true)
	if err != nil || gotPre != 2 {
		t.Fatalf("pre-mark ctx=%d err=%v want 2", gotPre, err)
	}

	// Take the snapshot, then simulate the current block marking itself
	// intra at the same slot. The live AboveIntra/LeftIntra now report
	// the current block (intra=1), but the snapshot must preserve the
	// pre-mark neighbor values.
	ctx.TxNeighborValid = true
	ctx.TxAboveNeighborIntra = ctx.AboveIntra[4]
	ctx.TxAboveNeighborBlockSize = ctx.AboveBlockSize[4]
	ctx.TxLeftNeighborIntra = ctx.LeftIntra[0]
	ctx.TxLeftNeighborBlockSize = ctx.LeftBlockSize[0]

	ctx.AboveIntra[4] = 1
	ctx.AboveBlockSize[4] = BlockSize8x8
	ctx.LeftIntra[0] = 1
	ctx.LeftBlockSize[0] = BlockSize8x8

	gotPost, err := ctx.SelectedTransformContextWithAvailability(TransformSize8x8, 4, 0, true, true)
	if err != nil || gotPost != 2 {
		t.Fatalf("post-mark ctx=%d err=%v want 2 (snapshot must hold neighbor flags)", gotPost, err)
	}

	// Without the snapshot, the function reads the just-written current
	// block's flags (intra=1, size=8x8): the inter-neighbor branch is
	// skipped, so above = AboveTx[4] (0) >= dims.Log2W (1) → 0, similarly
	// for left. ctx degenerates to 0 — the buggy pre-fix value.
	ctx.TxNeighborValid = false
	gotNoSnap, err := ctx.SelectedTransformContextWithAvailability(TransformSize8x8, 4, 0, true, true)
	if err != nil || gotNoSnap != 0 {
		t.Fatalf("no-snapshot ctx=%d err=%v want 0 (post-mark stale read)", gotNoSnap, err)
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
		Size:     BlockSize16x16,
		From:     TransformSize16x16,
		Depth:    0,
		HaveTop:  true,
		HaveLeft: true,
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
		Size:     BlockSize16x16,
		From:     TransformSize16x16,
		Depth:    2,
		HaveTop:  true,
		HaveLeft: true,
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

// TestTransformPartitionCategoryMatchesLibaom pins TransformPartitionContext's
// category index to libaom's txfm_partition_context() formula for every valid
// (bsize_max_tx, depth, from) triple reachable by libaom's
// read_tx_size_vartx() (depth in [0..MAX_VARTX_DEPTH-1] = [0,1]).
//
// libaom formula (av1/common/av1_common_int.h: txfm_partition_context):
//
//	max_tx_size = get_sqr_tx_size(AOMMAX(block_size_wide[bsize],
//	                                     block_size_high[bsize]))
//	base  = (TX_SIZES - 1 - max_tx_size) * 2
//	bonus = (txsize_sqr_up_map[tx_size] != max_tx_size &&
//	         max_tx_size > TX_8X8) ? 1 : 0
//	cat   = base + bonus
//
// goav1's category = 2*(int(TransformSize64x64)-int(dims.Max)) - req.Depth.
// The two formulas converge for all reachable inputs because dims.Max collapses
// rectangular `from` sizes onto their containing square (e.g. TX_32X16 →
// Max=3 = TX_32X32) and Depth shifts the result to mirror the bonus +1 that
// libaom adds when txsize_sqr_up_map[tx_size] != max_tx_size.
func TestTransformPartitionCategoryMatchesLibaom(t *testing.T) {
	type chain struct {
		bsize    BlockSize
		root     TransformSize // libaom's max_txsize_rect_lookup[bsize]
		sqr      TransformSize // libaom's get_sqr_tx_size(AOMMAX(bw,bh))
		baseLog2 uint8         // libaom's max_tx_size as log2 (1..4 for 8..64)
	}
	chains := []chain{
		{BlockSize8x8, TransformSize8x8, TransformSize8x8, 1},
		{BlockSize16x16, TransformSize16x16, TransformSize16x16, 2},
		{BlockSize32x32, TransformSize32x32, TransformSize32x32, 3},
		{BlockSize64x64, TransformSize64x64, TransformSize64x64, 4},
		{BlockSize64x32, TransformSize64x32, TransformSize64x64, 4},
		{BlockSize32x16, TransformSize32x16, TransformSize32x32, 3},
		{BlockSize16x8, TransformSize16x8, TransformSize16x16, 2},
		{BlockSize8x4, TransformSize8x4, TransformSize8x8, 1},
		{BlockSize32x8, TransformSize32x8, TransformSize32x32, 3},
		{BlockSize16x4, TransformSize16x4, TransformSize16x16, 2},
	}
	sqrUpMap := map[TransformSize]TransformSize{
		TransformSize4x4:   TransformSize4x4,
		TransformSize8x8:   TransformSize8x8,
		TransformSize16x16: TransformSize16x16,
		TransformSize32x32: TransformSize32x32,
		TransformSize64x64: TransformSize64x64,
		TransformSize4x8:   TransformSize8x8,
		TransformSize8x4:   TransformSize8x8,
		TransformSize8x16:  TransformSize16x16,
		TransformSize16x8:  TransformSize16x16,
		TransformSize16x32: TransformSize32x32,
		TransformSize32x16: TransformSize32x32,
		TransformSize32x64: TransformSize64x64,
		TransformSize64x32: TransformSize64x64,
		TransformSize4x16:  TransformSize16x16,
		TransformSize16x4:  TransformSize16x16,
		TransformSize8x32:  TransformSize32x32,
		TransformSize32x8:  TransformSize32x32,
		TransformSize16x64: TransformSize64x64,
		TransformSize64x16: TransformSize64x64,
	}
	for _, ch := range chains {
		depths := []struct {
			depth int
			from  TransformSize
		}{
			{0, ch.root},
		}
		if dims, ok := ch.root.Dimensions(); ok && dims.Max > 1 {
			depths = append(depths, struct {
				depth int
				from  TransformSize
			}{1, dims.Sub})
		}
		for _, d := range depths {
			base := int(2 * (4 - ch.baseLog2))
			bonus := 0
			if sqrUpMap[d.from] != ch.sqr && ch.sqr > TransformSize8x8 {
				bonus = 1
			}
			wantCat := base + bonus
			var ctx BlockModeContext
			gotCat, _, err := ctx.TransformPartitionContext(TransformPartitionRequest{
				Size:     ch.bsize,
				From:     d.from,
				Depth:    d.depth,
				HaveTop:  true,
				HaveLeft: true,
			})
			if err != nil {
				t.Fatalf("bsize=%v from=%v depth=%d: %v", ch.bsize, d.from, d.depth, err)
			}
			if gotCat != wantCat {
				t.Fatalf("bsize=%v from=%v depth=%d: cat=%d want %d (libaom base=%d bonus=%d)",
					ch.bsize, d.from, d.depth, gotCat, wantCat, base, bonus)
			}
		}
	}
}

// TestTransformPartitionContextNeighborComparisonMatchesLibaom pins the
// above/left neighbor comparison in libaom's txfm_partition_context():
//
//	above = *above_ctx < tx_size_wide[tx_size]
//	left  = *left_ctx  < tx_size_high[tx_size]
//
// goav1 compares log2 widths/heights instead of pixel widths/heights; since
// the recorded values are powers of two the comparison must produce the same
// boolean outcome.
func TestTransformPartitionContextNeighborComparisonMatchesLibaom(t *testing.T) {
	cases := []struct {
		recordLog2 uint8
		fromSize   TransformSize
	}{
		{0, TransformSize8x8},
		{1, TransformSize8x8},
		{2, TransformSize8x8},
		{1, TransformSize16x16},
		{2, TransformSize16x16},
		{3, TransformSize16x16},
		{2, TransformSize32x32},
		{3, TransformSize32x32},
		{0, TransformSize64x64},
		{4, TransformSize64x64},
	}
	for _, c := range cases {
		var ctx BlockModeContext
		ctx.AboveTx[0] = c.recordLog2
		ctx.LeftTx[0] = c.recordLog2
		fromDims, ok := c.fromSize.Dimensions()
		if !ok {
			t.Fatalf("missing dims for %v", c.fromSize)
		}
		_, gotCtx, err := ctx.TransformPartitionContext(TransformPartitionRequest{
			Size:     BlockSize64x64,
			From:     c.fromSize,
			Depth:    0,
			HaveTop:  true,
			HaveLeft: true,
		})
		if err != nil {
			t.Fatalf("from=%v record=%d: %v", c.fromSize, c.recordLog2, err)
		}
		recordPixels := 4 << c.recordLog2
		fromW := int(fromDims.W4) << 2
		fromH := int(fromDims.H4) << 2
		want := 0
		if recordPixels < fromW {
			want++
		}
		if recordPixels < fromH {
			want++
		}
		if gotCtx != want {
			t.Fatalf("from=%v record=%d (px=%d) txw=%d txh=%d: ctx=%d want %d",
				c.fromSize, c.recordLog2, recordPixels, fromW, fromH, gotCtx, want)
		}
	}
}
