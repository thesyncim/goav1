package tile

import (
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/parser"
)

func TestDecodeTransformTreeIntraSelectedMarksContexts(t *testing.T) {
	var state DecodeState
	if err := state.Reset([]byte{0x00}, Job{Offset: 0, Size: 1}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	var cdfs TransformCDFs
	if err := cdfs.InitDefault(); err != nil {
		t.Fatal(err)
	}
	var ctx BlockModeContext
	result, err := state.DecodeTransformTree(&cdfs, &ctx, TransformTreeRequest{
		Size:          BlockSize16x16,
		VisibleW4:     4,
		VisibleH4:     4,
		TransformMode: parser.TransformModeSwitchable,
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Y != TransformSize16x16 || result.Variable {
		t.Fatalf("result=%+v want fixed 16x16", result)
	}
	if ctx.AboveTxIntra[0] != 2 || ctx.LeftTxIntra[0] != 2 || ctx.AboveTx[3] != 2 || ctx.LeftTx[3] != 2 {
		t.Fatalf("tx contexts above=%v left=%v aboveIntra=%v leftIntra=%v",
			ctx.AboveTx[:4], ctx.LeftTx[:4], ctx.AboveTxIntra[:4], ctx.LeftTxIntra[:4])
	}
}

func TestDecodeTransformTreeLosslessForcesChroma4x4(t *testing.T) {
	var state DecodeState
	if err := state.Reset([]byte{0x00}, Job{Offset: 0, Size: 1}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	var cdfs TransformCDFs
	if err := cdfs.InitDefault(); err != nil {
		t.Fatal(err)
	}
	var ctx BlockModeContext
	result, err := state.DecodeTransformTree(&cdfs, &ctx, TransformTreeRequest{
		Size:          BlockSize4x8,
		VisibleW4:     1,
		VisibleH4:     2,
		Color:         parser.ColorConfig{BitDepth: 8},
		TransformMode: parser.TransformModeSwitchable,
		Lossless:      true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Y != TransformSize4x4 || result.UV != TransformSize4x4 || !result.HasUV {
		t.Fatalf("lossless transform tree=%+v want y=4x4 uv=4x4", result)
	}
}

func TestMaxTransformSizeTableMatchesDav1d(t *testing.T) {
	tests := []struct {
		block BlockSize
		y     TransformSize
		uv420 TransformSize
		uv422 TransformSize
		uv444 TransformSize
	}{
		{BlockSize128x128, TransformSize64x64, TransformSize32x32, TransformSize32x32, TransformSize32x32},
		{BlockSize128x64, TransformSize64x64, TransformSize32x32, TransformSize32x32, TransformSize32x32},
		{BlockSize64x128, TransformSize64x64, TransformSize32x32, 0, TransformSize32x32},
		{BlockSize64x64, TransformSize64x64, TransformSize32x32, TransformSize32x32, TransformSize32x32},
		{BlockSize64x32, TransformSize64x32, TransformSize32x16, TransformSize32x32, TransformSize32x32},
		{BlockSize64x16, TransformSize64x16, TransformSize32x8, TransformSize32x16, TransformSize32x16},
		{BlockSize32x64, TransformSize32x64, TransformSize16x32, 0, TransformSize32x32},
		{BlockSize32x32, TransformSize32x32, TransformSize16x16, TransformSize16x32, TransformSize32x32},
		{BlockSize32x16, TransformSize32x16, TransformSize16x8, TransformSize16x16, TransformSize32x16},
		{BlockSize32x8, TransformSize32x8, TransformSize16x4, TransformSize16x8, TransformSize32x8},
		{BlockSize16x64, TransformSize16x64, TransformSize8x32, 0, TransformSize16x32},
		{BlockSize16x32, TransformSize16x32, TransformSize8x16, 0, TransformSize16x32},
		{BlockSize16x16, TransformSize16x16, TransformSize8x8, TransformSize8x16, TransformSize16x16},
		{BlockSize16x8, TransformSize16x8, TransformSize8x4, TransformSize8x8, TransformSize16x8},
		{BlockSize16x4, TransformSize16x4, TransformSize8x4, TransformSize8x4, TransformSize16x4},
		{BlockSize8x32, TransformSize8x32, TransformSize4x16, 0, TransformSize8x32},
		{BlockSize8x16, TransformSize8x16, TransformSize4x8, 0, TransformSize8x16},
		{BlockSize8x8, TransformSize8x8, TransformSize4x4, TransformSize4x8, TransformSize8x8},
		{BlockSize8x4, TransformSize8x4, TransformSize4x4, TransformSize4x4, TransformSize8x4},
		{BlockSize4x16, TransformSize4x16, TransformSize4x8, 0, TransformSize4x16},
		{BlockSize4x8, TransformSize4x8, TransformSize4x4, 0, TransformSize4x8},
		{BlockSize4x4, TransformSize4x4, TransformSize4x4, TransformSize4x4, TransformSize4x4},
	}
	for _, tt := range tests {
		got, err := MaxTransformSize(tt.block, parser.ColorConfig{}, 0)
		if err != nil || got != tt.y {
			t.Fatalf("block=%d y max=%d err=%v want %d", tt.block, got, err, tt.y)
		}
		checkChromaMaxTransform(t, tt.block, parser.ColorConfig{SubsamplingX: true, SubsamplingY: true}, tt.uv420, "420")
		checkChromaMaxTransform(t, tt.block, parser.ColorConfig{SubsamplingX: true}, tt.uv422, "422")
		checkChromaMaxTransform(t, tt.block, parser.ColorConfig{}, tt.uv444, "444")
	}
}

func TestBlockSignalsTransformSizeMatchesLibaom(t *testing.T) {
	for size := BlockSize(0); size < blockSizeCount; size++ {
		want := size != BlockSize4x4
		if got := blockSignalsTransformSize(size); got != want {
			t.Fatalf("blockSignalsTransformSize(%d)=%v want %v", size, got, want)
		}
	}
}

func TestDecodeTransformTreeInterNoSplitMarksLeaf(t *testing.T) {
	var state DecodeState
	if err := state.Reset([]byte{0x00}, Job{Offset: 0, Size: 1}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	var cdfs TransformCDFs
	if err := cdfs.InitDefault(); err != nil {
		t.Fatal(err)
	}
	var ctx BlockModeContext
	result, err := state.DecodeTransformTree(&cdfs, &ctx, TransformTreeRequest{
		Size:          BlockSize16x16,
		VisibleW4:     4,
		VisibleH4:     4,
		TransformMode: parser.TransformModeSwitchable,
		Inter:         true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Y != TransformSize16x16 || !result.Variable || result.Split != ([2]uint16{}) {
		t.Fatalf("result=%+v want variable unsplit 16x16", result)
	}
	if ctx.AboveTx[0] != 2 || ctx.LeftTx[0] != 2 {
		t.Fatalf("tx contexts above=%v left=%v", ctx.AboveTx[:4], ctx.LeftTx[:4])
	}
	var leaves []TransformBlock
	if err := result.ForEachLumaTXB(TransformTreeRequest{
		Size:          BlockSize16x16,
		VisibleW4:     4,
		VisibleH4:     4,
		TransformMode: parser.TransformModeSwitchable,
		Inter:         true,
	}, func(block TransformBlock) error {
		leaves = append(leaves, block)
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if len(leaves) != 1 || leaves[0].Size != TransformSize16x16 || leaves[0].VisibleW4 != 4 || leaves[0].VisibleH4 != 4 {
		t.Fatalf("leaves=%+v", leaves)
	}
}

func TestTransformTreeReplaySplitMasksMatchDav1dOrder(t *testing.T) {
	req := TransformTreeRequest{
		Size:          BlockSize16x16,
		VisibleW4:     4,
		VisibleH4:     4,
		TransformMode: parser.TransformModeSwitchable,
		Inter:         true,
	}
	result := TransformTreeResult{
		Y:        TransformSize16x16,
		Variable: true,
		Split:    [2]uint16{1, 0},
	}
	var leaves []TransformBlock
	if err := result.ForEachLumaTXB(req, func(block TransformBlock) error {
		leaves = append(leaves, block)
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	want := []TransformBlock{
		{X4: 0, Y4: 0, Size: TransformSize8x8, VisibleW4: 2, VisibleH4: 2},
		{X4: 2, Y4: 0, Size: TransformSize8x8, VisibleW4: 2, VisibleH4: 2},
		{X4: 0, Y4: 2, Size: TransformSize8x8, VisibleW4: 2, VisibleH4: 2},
		{X4: 2, Y4: 2, Size: TransformSize8x8, VisibleW4: 2, VisibleH4: 2},
	}
	if len(leaves) != len(want) {
		t.Fatalf("leaf count=%d leaves=%+v", len(leaves), leaves)
	}
	for i := range want {
		if leaves[i] != want[i] {
			t.Fatalf("leaf[%d]=+%v want %+v", i, leaves[i], want[i])
		}
	}
}

func TestTransformTreeReplayRectangularSplitOrderMatchesDav1d(t *testing.T) {
	tests := []struct {
		name string
		req  TransformTreeRequest
		tree TransformTreeResult
		want []TransformBlock
	}{
		{
			name: "16x8 splits horizontally",
			req: TransformTreeRequest{
				Size:          BlockSize16x8,
				VisibleW4:     4,
				VisibleH4:     2,
				TransformMode: parser.TransformModeSwitchable,
				Inter:         true,
			},
			tree: TransformTreeResult{Y: TransformSize16x8, Variable: true, Split: [2]uint16{1, 0}},
			want: []TransformBlock{
				{X4: 0, Y4: 0, Size: TransformSize8x8, VisibleW4: 2, VisibleH4: 2},
				{X4: 2, Y4: 0, Size: TransformSize8x8, VisibleW4: 2, VisibleH4: 2},
			},
		},
		{
			name: "8x16 splits vertically",
			req: TransformTreeRequest{
				Size:          BlockSize8x16,
				VisibleW4:     2,
				VisibleH4:     4,
				TransformMode: parser.TransformModeSwitchable,
				Inter:         true,
			},
			tree: TransformTreeResult{Y: TransformSize8x16, Variable: true, Split: [2]uint16{1, 0}},
			want: []TransformBlock{
				{X4: 0, Y4: 0, Size: TransformSize8x8, VisibleW4: 2, VisibleH4: 2},
				{X4: 0, Y4: 2, Size: TransformSize8x8, VisibleW4: 2, VisibleH4: 2},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var got []TransformBlock
			if err := tt.tree.ForEachLumaTXB(tt.req, func(block TransformBlock) error {
				got = append(got, block)
				return nil
			}); err != nil {
				t.Fatal(err)
			}
			if len(got) != len(tt.want) {
				t.Fatalf("leaf count=%d got=%+v want %+v", len(got), got, tt.want)
			}
			for i := range tt.want {
				if got[i] != tt.want[i] {
					t.Fatalf("leaf[%d]=%+v want %+v", i, got[i], tt.want[i])
				}
			}
		})
	}
}

func TestTransformTreeReplay8x8SplitEmits4x4Leaves(t *testing.T) {
	req := TransformTreeRequest{
		Size:          BlockSize8x8,
		VisibleW4:     2,
		VisibleH4:     2,
		TransformMode: parser.TransformModeSwitchable,
		Inter:         true,
	}
	result := TransformTreeResult{
		Y:        TransformSize8x8,
		Variable: true,
		Split:    [2]uint16{1, 0},
	}
	var leaves []TransformBlock
	if err := result.ForEachLumaTXB(req, func(block TransformBlock) error {
		leaves = append(leaves, block)
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if len(leaves) != 4 {
		t.Fatalf("leaf count=%d leaves=%+v", len(leaves), leaves)
	}
	for i, leaf := range leaves {
		if leaf.Size != TransformSize4x4 || leaf.VisibleW4 != 1 || leaf.VisibleH4 != 1 {
			t.Fatalf("leaf[%d]=%+v want 4x4", i, leaf)
		}
	}
}

func TestMarkTransformAreaMatchesLibaomTxfmPartitionUpdate(t *testing.T) {
	var ctx BlockModeContext
	if err := ctx.MarkTransformArea(4, 6, 2, 2, 0, 0, false); err != nil {
		t.Fatal(err)
	}
	if ctx.AboveTx[4] != 0 || ctx.AboveTx[5] != 0 || ctx.LeftTx[6] != 0 || ctx.LeftTx[7] != 0 {
		t.Fatalf("split-to-4x4 context above=%v left=%v", ctx.AboveTx[4:8], ctx.LeftTx[6:10])
	}
	ctx.AboveTx[0], ctx.LeftTx[0] = 9, 9
	if err := ctx.MarkTransformArea(0, 0, 4, 2, 2, 1, false); err != nil {
		t.Fatal(err)
	}
	if ctx.AboveTx[0] != 2 || ctx.AboveTx[3] != 2 || ctx.LeftTx[0] != 1 || ctx.LeftTx[1] != 1 {
		t.Fatalf("txfm partition update context above=%v left=%v", ctx.AboveTx[:4], ctx.LeftTx[:2])
	}
}

func TestTransformTreeFixedInterSkipMarksBlockLogContext(t *testing.T) {
	var state DecodeState
	if err := state.Reset([]byte{0x00}, Job{Offset: 0, Size: 1}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	var cdfs TransformCDFs
	if err := cdfs.InitDefault(); err != nil {
		t.Fatal(err)
	}
	var ctx BlockModeContext
	result, err := state.DecodeTransformTree(&cdfs, &ctx, TransformTreeRequest{
		Size:          BlockSize128x128,
		VisibleW4:     32,
		VisibleH4:     32,
		TransformMode: parser.TransformModeSwitchable,
		Inter:         true,
		SkipTransform: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Y != TransformSize64x64 || result.Variable {
		t.Fatalf("result=%+v want fixed max y tx", result)
	}
	if ctx.AboveTx[31] != 5 || ctx.LeftTx[31] != 5 {
		t.Fatalf("skip contexts above[31]=%d left[31]=%d want block log 5", ctx.AboveTx[31], ctx.LeftTx[31])
	}
	var called bool
	if err := result.ForEachLumaTXB(TransformTreeRequest{
		Size:          BlockSize128x128,
		VisibleW4:     32,
		VisibleH4:     32,
		TransformMode: parser.TransformModeSwitchable,
		Inter:         true,
		SkipTransform: true,
	}, func(TransformBlock) error {
		called = true
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if called {
		t.Fatal("skip transform replay called visitor")
	}
}

func checkChromaMaxTransform(t *testing.T, block BlockSize, color parser.ColorConfig, want TransformSize, name string) {
	t.Helper()
	got, err := MaxTransformSize(block, color, 1)
	if err != nil || got != want {
		t.Fatalf("block=%d %s max=%d err=%v want %d", block, name, got, err, want)
	}
}

func TestTransformTreeRejectsInvalidInputs(t *testing.T) {
	var cdfs TransformCDFs
	if err := cdfs.InitDefault(); err != nil {
		t.Fatal(err)
	}
	var state DecodeState
	if err := state.Reset([]byte{0x00}, Job{Offset: 0, Size: 1}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	if _, err := (*DecodeState)(nil).DecodeTransformTree(&cdfs, &BlockModeContext{}, TransformTreeRequest{Size: BlockSize4x4, VisibleW4: 1, VisibleH4: 1}); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("nil state err=%v want %v", err, ErrInvalidDecodeState)
	}
	if _, err := state.DecodeTransformTree(nil, &BlockModeContext{}, TransformTreeRequest{Size: BlockSize4x4, VisibleW4: 1, VisibleH4: 1}); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("nil cdfs err=%v want %v", err, ErrInvalidDecodeState)
	}
	if _, err := state.DecodeTransformTree(&cdfs, &BlockModeContext{}, TransformTreeRequest{Size: BlockSize4x4}); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("empty req err=%v want %v", err, ErrInvalidDecodeState)
	}
	if err := (TransformTreeResult{Y: transformSizeCount}).ForEachLumaTXB(TransformTreeRequest{Size: BlockSize4x4, VisibleW4: 1, VisibleH4: 1}, func(TransformBlock) error { return nil }); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("bad replay err=%v want %v", err, ErrInvalidDecodeState)
	}
}

func FuzzDecodeTransformTree(f *testing.F) {
	f.Add([]byte{0x00}, uint8(BlockSize16x16), uint8(parser.TransformModeSwitchable), true, false, false, uint8(4), uint8(4))
	f.Add([]byte{0xff}, uint8(BlockSize32x32), uint8(parser.TransformModeLargest), true, true, false, uint8(8), uint8(8))
	f.Add([]byte{0xa5, 0x5a}, uint8(BlockSize8x8), uint8(parser.TransformModeSwitchable), false, false, true, uint8(2), uint8(2))

	f.Fuzz(func(t *testing.T, payload []byte, rawSize uint8, rawMode uint8, inter bool, skip bool, lossless bool, rawW uint8, rawH uint8) {
		if len(payload) == 0 || len(payload) > 64 {
			return
		}
		size := BlockSize(rawSize % uint8(blockSizeCount))
		dims, ok := size.Dimensions()
		if !ok {
			t.Fatal("invalid generated block size")
		}
		visibleW := uint8((rawW % dims.W4) + 1)
		visibleH := uint8((rawH % dims.H4) + 1)
		mode := parser.TransformMode(rawMode % 3)

		var cdfs TransformCDFs
		if err := cdfs.InitDefault(); err != nil {
			t.Fatal(err)
		}
		var state DecodeState
		if err := state.Reset(payload, Job{Offset: 0, Size: len(payload)}, DecodeOptions{}); err != nil {
			t.Fatal(err)
		}
		req := TransformTreeRequest{
			Size:          size,
			VisibleW4:     visibleW,
			VisibleH4:     visibleH,
			TransformMode: mode,
			Inter:         inter,
			SkipTransform: skip,
			Lossless:      lossless,
		}
		result, err := state.DecodeTransformTree(&cdfs, &BlockModeContext{}, req)
		if err != nil {
			return
		}
		if !result.Y.Valid() {
			t.Fatalf("invalid y transform=%d", result.Y)
		}
		if err := result.ForEachLumaTXB(req, func(block TransformBlock) error {
			if block.X4 < 0 || block.Y4 < 0 || block.VisibleW4 == 0 || block.VisibleH4 == 0 {
				t.Fatalf("bad block=%+v", block)
			}
			return nil
		}); err != nil {
			t.Fatalf("ForEachLumaTXB err=%v result=%+v req=%+v", err, result, req)
		}
	})
}
