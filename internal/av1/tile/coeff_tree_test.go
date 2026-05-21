package tile

import (
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/transform"
)

func TestDecodeLumaCoefficientsFollowsDav1dTXBReplay(t *testing.T) {
	var cdfs CoeffCDFs
	if err := cdfs.InitDefault(0); err != nil {
		t.Fatal(err)
	}
	var state DecodeState
	if err := state.Reset(make([]byte, 16), Job{Offset: 0, Size: 16}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	treeReq := TransformTreeRequest{
		Size:          BlockSize8x8,
		VisibleW4:     2,
		VisibleH4:     2,
		Inter:         true,
		TransformMode: parser.TransformModeSwitchable,
	}
	tree := TransformTreeResult{
		Y:        TransformSize8x8,
		Variable: true,
		Split:    [2]uint16{1},
	}
	var ctx CoeffEntropyContext
	var scratch LumaCoeffTreeScratch
	var visits []TransformBlock
	stats, err := state.DecodeLumaCoefficients(&cdfs, &ctx, &scratch, LumaCoeffTreeRequest{
		TreeRequest:     treeReq,
		Tree:            tree,
		Class:           transform.Class2D,
		EOBMultiContext: 0,
	}, func(block LumaCoeffBlock) error {
		visits = append(visits, block.Block)
		if len(block.Coeffs) != 16 || len(block.Scan) != 16 {
			t.Fatalf("buffer lens coeffs=%d scan=%d want 16", len(block.Coeffs), len(block.Scan))
		}
		assertTXBDecodeInvariants(t, block.Result, block.Coeffs, block.Scan)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if stats.TXBs != 4 || stats.TXBs != stats.NonZero+stats.AllZero {
		t.Fatalf("stats=%+v want 4 txbs and consistent counts", stats)
	}
	want := []TransformBlock{
		{X4: 0, Y4: 0, Size: TransformSize4x4, VisibleW4: 1, VisibleH4: 1},
		{X4: 1, Y4: 0, Size: TransformSize4x4, VisibleW4: 1, VisibleH4: 1},
		{X4: 0, Y4: 1, Size: TransformSize4x4, VisibleW4: 1, VisibleH4: 1},
		{X4: 1, Y4: 1, Size: TransformSize4x4, VisibleW4: 1, VisibleH4: 1},
	}
	if len(visits) != len(want) {
		t.Fatalf("visits=%v want %v", visits, want)
	}
	for i := range want {
		if visits[i] != want[i] {
			t.Fatalf("visit[%d]=%+v want %+v", i, visits[i], want[i])
		}
	}
}

func TestDecodeLumaCoefficientsSelectsTransformPerTXB(t *testing.T) {
	var cdfs CoeffCDFs
	if err := cdfs.InitDefault(0); err != nil {
		t.Fatal(err)
	}
	var state DecodeState
	if err := state.Reset(make([]byte, 16), Job{Offset: 0, Size: 16}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	treeReq := TransformTreeRequest{
		Size:          BlockSize8x8,
		VisibleW4:     2,
		VisibleH4:     2,
		Inter:         true,
		TransformMode: parser.TransformModeSwitchable,
	}
	tree := TransformTreeResult{
		Y:        TransformSize8x8,
		Variable: true,
		Split:    [2]uint16{1},
	}
	selected := []transform.Type{
		transform.TypeDCTDCT,
		transform.TypeHDCT,
		transform.TypeVDCT,
		transform.TypeADSTDCT,
	}
	calls := 0
	var ctx CoeffEntropyContext
	var scratch LumaCoeffTreeScratch
	var got []transform.Type
	stats, err := state.DecodeLumaCoefficients(&cdfs, &ctx, &scratch, LumaCoeffTreeRequest{
		TreeRequest: treeReq,
		Tree:        tree,
		Class:       transform.Class2D,
		TransformSelect: CoeffTransformSelectorFunc(func(req CoeffTransformRequest) (transform.Type, error) {
			if req.Plane != 0 || req.Block.Size != TransformSize4x4 {
				t.Fatalf("selector req=%+v", req)
			}
			if calls >= len(selected) {
				t.Fatalf("unexpected selector call %d", calls)
			}
			typ := selected[calls]
			calls++
			return typ, nil
		}),
	}, func(block LumaCoeffBlock) error {
		got = append(got, block.Transform)
		assertTXBDecodeInvariants(t, block.Result, block.Coeffs, block.Scan)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if stats.TXBs != 4 || calls != 4 {
		t.Fatalf("stats=%+v calls=%d", stats, calls)
	}
	for i, want := range selected {
		if got[i] != want {
			t.Fatalf("transform[%d]=%d want %d", i, got[i], want)
		}
	}
}

func TestDecodeLumaCoefficientsSkipTransformResetsContext(t *testing.T) {
	var state DecodeState
	var cdfs CoeffCDFs
	var ctx CoeffEntropyContext
	for i := 4; i < 8; i++ {
		ctx.Above[0][i] = 17
		ctx.Left[0][i] = 17
	}
	var scratch LumaCoeffTreeScratch
	stats, err := state.DecodeLumaCoefficients(&cdfs, &ctx, &scratch, LumaCoeffTreeRequest{
		TreeRequest: TransformTreeRequest{
			Size:          BlockSize16x16,
			X4:            4,
			Y4:            4,
			VisibleW4:     4,
			VisibleH4:     4,
			SkipTransform: true,
		},
		Tree:  TransformTreeResult{Y: TransformSize16x16},
		Class: transform.Class2D,
	}, func(LumaCoeffBlock) error {
		t.Fatal("visitor called for skip_txfm block")
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if stats != (LumaCoeffStats{}) {
		t.Fatalf("stats=%+v want zero for skip_txfm", stats)
	}
	for i := 4; i < 8; i++ {
		if ctx.Above[0][i] != 0 || ctx.Left[0][i] != 0 {
			t.Fatalf("ctx slot %d above=%d left=%d want reset", i, ctx.Above[0][i], ctx.Left[0][i])
		}
	}
}

func TestDecodeChromaCoefficientsUsesSubsampledPlaneBlock(t *testing.T) {
	var cdfs CoeffCDFs
	if err := cdfs.InitDefault(0); err != nil {
		t.Fatal(err)
	}
	var state DecodeState
	if err := state.Reset(make([]byte, 8), Job{Offset: 0, Size: 8}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	treeReq := TransformTreeRequest{
		Size:      BlockSize16x16,
		X4:        5,
		Y4:        7,
		VisibleW4: 2,
		VisibleH4: 2,
	}
	tree := TransformTreeResult{UV: TransformSize8x8, HasUV: true}
	var ctx CoeffEntropyContext
	var scratch LumaCoeffTreeScratch
	var got ChromaCoeffBlock
	stats, err := state.DecodeChromaCoefficients(&cdfs, &ctx, &scratch, ChromaCoeffTreeRequest{
		TreeRequest: treeReq,
		Tree:        tree,
		Color:       parser.ColorConfig{SubsamplingX: true, SubsamplingY: true},
		Plane:       1,
		Class:       transform.Class2D,
	}, func(block ChromaCoeffBlock) error {
		got = block
		if len(block.Coeffs) != 64 || len(block.Scan) != 64 {
			t.Fatalf("buffer lens coeffs=%d scan=%d want 64", len(block.Coeffs), len(block.Scan))
		}
		assertTXBDecodeInvariants(t, block.Result, block.Coeffs, block.Scan)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if stats.TXBs != 1 || stats.TXBs != stats.NonZero+stats.AllZero {
		t.Fatalf("stats=%+v want one chroma txb and consistent counts", stats)
	}
	want := ChromaCoeffBlock{
		Plane: 1,
		Block: TransformBlock{X4: 2, Y4: 3, Size: TransformSize8x8, VisibleW4: 2, VisibleH4: 2},
	}
	if got.Plane != want.Plane || got.Block != want.Block {
		t.Fatalf("chroma block plane=%d block=%+v want plane=%d block=%+v", got.Plane, got.Block, want.Plane, want.Block)
	}
}

func TestDecodeChromaCoefficientsNoChromaRefAndSkipReset(t *testing.T) {
	color420 := parser.ColorConfig{SubsamplingX: true, SubsamplingY: true}
	var state DecodeState
	var cdfs CoeffCDFs
	var ctx CoeffEntropyContext
	var scratch LumaCoeffTreeScratch
	stats, err := state.DecodeChromaCoefficients(&cdfs, &ctx, &scratch, ChromaCoeffTreeRequest{
		TreeRequest: TransformTreeRequest{Size: BlockSize4x4, VisibleW4: 1, VisibleH4: 1},
		Tree:        TransformTreeResult{UV: TransformSize4x4, HasUV: true},
		Color:       color420,
		Plane:       1,
		Class:       transform.Class2D,
	}, func(ChromaCoeffBlock) error {
		t.Fatal("visitor called for no-chroma-ref block")
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if stats != (LumaCoeffStats{}) {
		t.Fatalf("stats=%+v want zero for no-chroma-ref block", stats)
	}

	for i := 2; i < 4; i++ {
		ctx.Above[2][i] = 17
		ctx.Left[2][i] = 17
	}
	stats, err = state.DecodeChromaCoefficients(&cdfs, &ctx, &scratch, ChromaCoeffTreeRequest{
		TreeRequest: TransformTreeRequest{
			Size:          BlockSize16x16,
			X4:            4,
			Y4:            4,
			VisibleW4:     4,
			VisibleH4:     4,
			SkipTransform: true,
		},
		Tree:  TransformTreeResult{UV: TransformSize8x8, HasUV: true},
		Color: color420,
		Plane: 2,
		Class: transform.Class2D,
	}, func(ChromaCoeffBlock) error {
		t.Fatal("visitor called for skip_txfm chroma block")
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if stats != (LumaCoeffStats{}) {
		t.Fatalf("stats=%+v want zero for skip_txfm chroma", stats)
	}
	for i := 2; i < 4; i++ {
		if ctx.Above[2][i] != 0 || ctx.Left[2][i] != 0 {
			t.Fatalf("ctx slot %d above=%d left=%d want reset", i, ctx.Above[2][i], ctx.Left[2][i])
		}
	}
}

func TestLumaCoeffScratchMatchesAdjustedLibaomScanBounds(t *testing.T) {
	var scratch LumaCoeffTreeScratch
	coeffs, scan, levels, err := scratch.coeffBuffers(TransformSize64x64, transform.Class2D)
	if err != nil {
		t.Fatal(err)
	}
	if len(coeffs) != 1024 || len(scan) != 1024 {
		t.Fatalf("scan buffers coeffs=%d scan=%d want adjusted 32x32", len(coeffs), len(scan))
	}
	if len(levels) != (32+txPadHorizontal)*(32+txPadHorizontal) {
		t.Fatalf("levels len=%d want adjusted padded 32x32", len(levels))
	}
}

func TestDecodeLumaCoefficientsRejectsInvalidInputs(t *testing.T) {
	valid := LumaCoeffTreeRequest{
		TreeRequest: TransformTreeRequest{Size: BlockSize4x4, VisibleW4: 1, VisibleH4: 1},
		Tree:        TransformTreeResult{Y: TransformSize4x4},
		Class:       transform.Class2D,
	}
	var state DecodeState
	var cdfs CoeffCDFs
	var ctx CoeffEntropyContext
	var scratch LumaCoeffTreeScratch
	visitor := func(LumaCoeffBlock) error { return nil }

	if _, err := (*DecodeState)(nil).DecodeLumaCoefficients(&cdfs, &ctx, &scratch, valid, visitor); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("nil state err=%v want %v", err, ErrInvalidDecodeState)
	}
	if _, err := state.DecodeLumaCoefficients(nil, &ctx, &scratch, valid, visitor); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("nil cdfs err=%v want %v", err, ErrInvalidDecodeState)
	}
	if _, err := state.DecodeLumaCoefficients(&cdfs, nil, &scratch, valid, visitor); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("nil ctx err=%v want %v", err, ErrInvalidDecodeState)
	}
	if _, err := state.DecodeLumaCoefficients(&cdfs, &ctx, nil, valid, visitor); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("nil scratch err=%v want %v", err, ErrInvalidDecodeState)
	}
	if _, err := state.DecodeLumaCoefficients(&cdfs, &ctx, &scratch, valid, nil); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("nil visitor err=%v want %v", err, ErrInvalidDecodeState)
	}
	badClass := valid
	badClass.Class = transform.Class(99)
	if _, err := state.DecodeLumaCoefficients(&cdfs, &ctx, &scratch, badClass, visitor); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("bad class err=%v want %v", err, ErrInvalidDecodeState)
	}
	badTree := valid
	badTree.Tree.Y = transformSizeCount
	if _, err := state.DecodeLumaCoefficients(&cdfs, &ctx, &scratch, badTree, visitor); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("bad tree err=%v want %v", err, ErrInvalidDecodeState)
	}
	if _, _, _, err := (*LumaCoeffTreeScratch)(nil).coeffBuffers(TransformSize4x4, transform.Class2D); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("nil scratch buffers err=%v want %v", err, ErrInvalidDecodeState)
	}

	chromaValid := ChromaCoeffTreeRequest{
		TreeRequest: TransformTreeRequest{Size: BlockSize8x8, VisibleW4: 2, VisibleH4: 2},
		Tree:        TransformTreeResult{UV: TransformSize4x4, HasUV: true},
		Color:       parser.ColorConfig{SubsamplingX: true, SubsamplingY: true},
		Plane:       1,
		Class:       transform.Class2D,
	}
	if _, err := state.DecodeChromaCoefficients(&cdfs, &ctx, &scratch, chromaValid, nil); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("nil chroma visitor err=%v want %v", err, ErrInvalidDecodeState)
	}
	badChroma := chromaValid
	badChroma.Plane = 0
	if _, err := state.DecodeChromaCoefficients(&cdfs, &ctx, &scratch, badChroma, func(ChromaCoeffBlock) error { return nil }); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("bad chroma plane err=%v want %v", err, ErrInvalidDecodeState)
	}
	badChroma = chromaValid
	badChroma.Tree.HasUV = false
	if _, err := state.DecodeChromaCoefficients(&cdfs, &ctx, &scratch, badChroma, func(ChromaCoeffBlock) error { return nil }); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("missing uv tree err=%v want %v", err, ErrInvalidDecodeState)
	}
}

func TestDecodeLumaCoefficientsAllocs(t *testing.T) {
	var cdfs CoeffCDFs
	if err := cdfs.InitDefault(0); err != nil {
		t.Fatal(err)
	}
	payload := []byte{0x00}
	req := LumaCoeffTreeRequest{
		TreeRequest: TransformTreeRequest{Size: BlockSize4x4, VisibleW4: 1, VisibleH4: 1},
		Tree:        TransformTreeResult{Y: TransformSize4x4},
		Class:       transform.Class2D,
	}
	var state DecodeState
	var ctx CoeffEntropyContext
	var scratch LumaCoeffTreeScratch
	visit := func(block LumaCoeffBlock) error {
		if block.Result.EOB != 1 || block.Coeffs[0] != 1 {
			t.Fatalf("block result=%+v coeff[0]=%d want eob=1 coeff=1", block.Result, block.Coeffs[0])
		}
		return nil
	}

	allocs := testing.AllocsPerRun(1000, func() {
		ctx = CoeffEntropyContext{}
		if err := state.Reset(payload, Job{Offset: 0, Size: len(payload)}, DecodeOptions{}); err != nil {
			t.Fatal(err)
		}
		if _, err := state.DecodeLumaCoefficients(&cdfs, &ctx, &scratch, req, visit); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("DecodeLumaCoefficients allocated: %f", allocs)
	}
}

func FuzzDecodeLumaCoefficients(f *testing.F) {
	f.Add([]byte{0x00}, uint8(BlockSize4x4), uint8(TransformSize4x4), uint8(transform.Class2D), false)
	f.Add([]byte{0x00, 0x00}, uint8(BlockSize8x8), uint8(TransformSize8x8), uint8(transform.Class2D), true)

	f.Fuzz(func(t *testing.T, payload []byte, rawBlock uint8, rawTX uint8, rawClass uint8, split bool) {
		if len(payload) == 0 || len(payload) > 64 {
			return
		}
		block := BlockSize(rawBlock % uint8(blockSizeCount))
		blockDims, ok := block.Dimensions()
		if !ok {
			t.Fatal("invalid normalized block")
		}
		tx := TransformSize(rawTX % uint8(transformSizeCount))
		txDims, ok := tx.Dimensions()
		if !ok || txDims.W4 > blockDims.W4 || txDims.H4 > blockDims.H4 {
			tx = TransformSize4x4
		}
		class := transform.Class(rawClass % 3)
		tree := TransformTreeResult{Y: tx}
		if split && tx > TransformSize4x4 {
			tree.Variable = true
			tree.Split[0] = 1
		}
		req := LumaCoeffTreeRequest{
			TreeRequest: TransformTreeRequest{
				Size:      block,
				VisibleW4: blockDims.W4,
				VisibleH4: blockDims.H4,
			},
			Tree:  tree,
			Class: class,
		}

		var cdfs CoeffCDFs
		if err := cdfs.InitDefault(0); err != nil {
			t.Fatal(err)
		}
		var state DecodeState
		if err := state.Reset(payload, Job{Offset: 0, Size: len(payload)}, DecodeOptions{}); err != nil {
			t.Fatal(err)
		}
		var ctx CoeffEntropyContext
		var scratch LumaCoeffTreeScratch
		stats, err := state.DecodeLumaCoefficients(&cdfs, &ctx, &scratch, req, func(block LumaCoeffBlock) error {
			assertTXBDecodeInvariants(t, block.Result, block.Coeffs, block.Scan)
			return nil
		})
		if err != nil {
			if errors.Is(err, ErrInvalidDecodeState) {
				return
			}
			t.Fatalf("DecodeLumaCoefficients err=%v", err)
		}
		if stats.TXBs != stats.NonZero+stats.AllZero {
			t.Fatalf("stats=%+v inconsistent counts", stats)
		}
	})
}

func FuzzDecodeChromaCoefficients(f *testing.F) {
	f.Add([]byte{0x00}, uint8(BlockSize16x16), uint8(1), uint8(1), uint8(1), uint8(transform.Class2D))
	f.Add([]byte{0xff}, uint8(BlockSize8x8), uint8(1), uint8(0), uint8(2), uint8(transform.ClassHoriz))

	f.Fuzz(func(t *testing.T, payload []byte, rawBlock uint8, rawSSX uint8, rawSSY uint8, rawPlane uint8, rawClass uint8) {
		if len(payload) == 0 || len(payload) > 64 {
			return
		}
		block := BlockSize(rawBlock % uint8(blockSizeCount))
		blockDims, ok := block.Dimensions()
		if !ok {
			t.Fatal("invalid normalized block")
		}
		color := parser.ColorConfig{SubsamplingX: rawSSX&1 != 0, SubsamplingY: rawSSY&1 != 0}
		plane := 1 + int(rawPlane%2)
		uv, err := MaxTransformSize(block, color, plane)
		if err != nil {
			return
		}
		req := ChromaCoeffTreeRequest{
			TreeRequest: TransformTreeRequest{
				Size:      block,
				VisibleW4: blockDims.W4,
				VisibleH4: blockDims.H4,
			},
			Tree:  TransformTreeResult{UV: uv, HasUV: true},
			Color: color,
			Plane: plane,
			Class: transform.Class(rawClass % 3),
		}

		var cdfs CoeffCDFs
		if err := cdfs.InitDefault(0); err != nil {
			t.Fatal(err)
		}
		var state DecodeState
		if err := state.Reset(payload, Job{Offset: 0, Size: len(payload)}, DecodeOptions{}); err != nil {
			t.Fatal(err)
		}
		var ctx CoeffEntropyContext
		var scratch LumaCoeffTreeScratch
		stats, err := state.DecodeChromaCoefficients(&cdfs, &ctx, &scratch, req, func(block ChromaCoeffBlock) error {
			assertTXBDecodeInvariants(t, block.Result, block.Coeffs, block.Scan)
			return nil
		})
		if err != nil {
			if errors.Is(err, ErrInvalidDecodeState) {
				return
			}
			t.Fatalf("DecodeChromaCoefficients err=%v", err)
		}
		if stats.TXBs != stats.NonZero+stats.AllZero {
			t.Fatalf("stats=%+v inconsistent counts", stats)
		}
	})
}
