package tile

import (
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/transform"
)

func TestDecodeBlockCoefficientsRunsTransformThenPlanes(t *testing.T) {
	transformCDFs, coeffCDFs := mustBlockCoeffCDFs(t)
	var state DecodeState
	if err := state.Reset(make([]byte, 32), Job{Offset: 0, Size: 32}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	var modeCtx BlockModeContext
	var coeffCtx CoeffEntropyContext
	var scratch BlockCoeffScratch
	var visits []BlockCoeffBlock
	result, err := state.DecodeBlockCoefficients(BlockCoeffCDFs{
		Transform: &transformCDFs,
		Coeff:     &coeffCDFs,
	}, &modeCtx, &coeffCtx, &scratch, BlockCoeffRequest{
		Transform: TransformTreeRequest{
			Size:          BlockSize16x16,
			VisibleW4:     4,
			VisibleH4:     4,
			Color:         parser.ColorConfig{SubsamplingX: true, SubsamplingY: true},
			TransformMode: parser.TransformModeLargest,
			Inter:         true,
		},
		LumaType:   transform.TypeDCTDCT,
		ChromaType: [2]transform.Type{transform.TypeDCTDCT, transform.TypeDCTDCT},
	}, func(block BlockCoeffBlock) error {
		visits = append(visits, block)
		assertTXBDecodeInvariants(t, block.Result, block.Coeffs, block.Scan)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Tree.Y != TransformSize16x16 || result.Tree.UV != TransformSize8x8 || !result.Tree.HasUV {
		t.Fatalf("tree=%+v want y=16x16 uv=8x8", result.Tree)
	}
	total := result.TotalStats()
	if total.TXBs != 3 || total.TXBs != total.NonZero+total.AllZero {
		t.Fatalf("total stats=%+v want 3 txbs and consistent counts", total)
	}
	wantPlanes := []uint8{0, 1, 2}
	if len(visits) != len(wantPlanes) {
		t.Fatalf("visits=%d want %d", len(visits), len(wantPlanes))
	}
	for i, want := range wantPlanes {
		if visits[i].Plane != want {
			t.Fatalf("visit[%d] plane=%d want %d", i, visits[i].Plane, want)
		}
	}
	if visits[0].Block.Size != TransformSize16x16 || visits[1].Block.Size != TransformSize8x8 || visits[2].Block.Size != TransformSize8x8 {
		t.Fatalf("visit sizes=%d,%d,%d want 16x16,8x8,8x8", visits[0].Block.Size, visits[1].Block.Size, visits[2].Block.Size)
	}
	for i, visit := range visits {
		if visit.Transform != transform.TypeDCTDCT {
			t.Fatalf("visit[%d] transform=%d want DCT_DCT", i, visit.Transform)
		}
	}
}

func TestDecodeBlockCoefficientsSkipTransformResetsAllPlaneContexts(t *testing.T) {
	transformCDFs, coeffCDFs := mustBlockCoeffCDFs(t)
	var state DecodeState
	if err := state.Reset([]byte{0x00}, Job{Offset: 0, Size: 1}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	var modeCtx BlockModeContext
	var coeffCtx CoeffEntropyContext
	for i := range 4 {
		coeffCtx.Above[0][i] = 17
		coeffCtx.Left[0][i] = 17
	}
	for plane := 1; plane <= 2; plane++ {
		for i := range 2 {
			coeffCtx.Above[plane][i] = 17
			coeffCtx.Left[plane][i] = 17
		}
	}
	var scratch BlockCoeffScratch
	result, err := state.DecodeBlockCoefficients(BlockCoeffCDFs{
		Transform: &transformCDFs,
		Coeff:     &coeffCDFs,
	}, &modeCtx, &coeffCtx, &scratch, BlockCoeffRequest{
		Transform: TransformTreeRequest{
			Size:          BlockSize16x16,
			VisibleW4:     4,
			VisibleH4:     4,
			Color:         parser.ColorConfig{SubsamplingX: true, SubsamplingY: true},
			TransformMode: parser.TransformModeLargest,
			Inter:         true,
			SkipTransform: true,
		},
		LumaType:   transform.TypeDCTDCT,
		ChromaType: [2]transform.Type{transform.TypeDCTDCT, transform.TypeDCTDCT},
	}, func(BlockCoeffBlock) error {
		t.Fatal("visitor called for skip_txfm block")
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.TotalStats() != (LumaCoeffStats{}) {
		t.Fatalf("stats=%+v want zero", result.TotalStats())
	}
	for i := range 4 {
		if coeffCtx.Above[0][i] != 0 || coeffCtx.Left[0][i] != 0 {
			t.Fatalf("luma ctx[%d] above=%d left=%d want reset", i, coeffCtx.Above[0][i], coeffCtx.Left[0][i])
		}
	}
	for plane := 1; plane <= 2; plane++ {
		for i := range 2 {
			if coeffCtx.Above[plane][i] != 0 || coeffCtx.Left[plane][i] != 0 {
				t.Fatalf("plane %d ctx[%d] above=%d left=%d want reset", plane, i, coeffCtx.Above[plane][i], coeffCtx.Left[plane][i])
			}
		}
	}
}

func TestDecodeBlockCoefficientsSkipAllZeroClearFastPath(t *testing.T) {
	transformCDFs, coeffCDFs := mustBlockCoeffCDFs(t)
	payload := mustTXBSkipPayload(t)
	var state DecodeState
	if err := state.Reset(payload, Job{Offset: 0, Size: uint32(len(payload))}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	var modeCtx BlockModeContext
	var coeffCtx CoeffEntropyContext
	var scratch BlockCoeffScratch
	var visits []BlockCoeffBlock
	result, err := state.DecodeBlockCoefficients(BlockCoeffCDFs{
		Transform: &transformCDFs,
		Coeff:     &coeffCDFs,
	}, &modeCtx, &coeffCtx, &scratch, BlockCoeffRequest{
		Transform: TransformTreeRequest{
			Size:          BlockSize4x4,
			VisibleW4:     1,
			VisibleH4:     1,
			Color:         parser.ColorConfig{MonoChrome: true},
			TransformMode: parser.TransformModeLargest,
		},
		LumaType:              transform.TypeDCTDCT,
		SkipAllZeroCoeffClear: true,
	}, func(block BlockCoeffBlock) error {
		visits = append(visits, block)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	total := result.TotalStats()
	if total.TXBs != 1 || total.AllZero != 1 || total.NonZero != 0 || total.EOBTotal != 0 {
		t.Fatalf("stats=%+v want one all-zero txb", total)
	}
	if len(visits) != 1 {
		t.Fatalf("visits=%d want 1", len(visits))
	}
	if !visits[0].Result.AllZero || visits[0].Result.EOB != 0 || len(visits[0].Coeffs) != 0 || len(visits[0].Scan) != 0 {
		t.Fatalf("visit=%+v want all-zero without coeff/scan slices", visits[0])
	}
	if coeffCtx.Above[0][0] != 0 || coeffCtx.Left[0][0] != 0 {
		t.Fatalf("coeff context above=%d left=%d want zero", coeffCtx.Above[0][0], coeffCtx.Left[0][0])
	}
}

func TestDecodeBlockCoefficientsDirtyCoeffClearFastPath(t *testing.T) {
	transformCDFs, coeffCDFs := mustBlockCoeffCDFs(t)
	var state DecodeState
	if err := state.Reset([]byte{0x00}, Job{Offset: 0, Size: 1}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	var modeCtx BlockModeContext
	var coeffCtx CoeffEntropyContext
	var scratch BlockCoeffScratch
	scratch.Coeff.Coeffs[0] = 9
	scratch.Coeff.Coeffs[17] = 11
	scratch.Coeff.InverseScan[0] = 0
	scratch.Coeff.InverseScan[1] = 17
	scratch.Coeff.coeffDirtyLen = 2

	result, err := state.DecodeBlockCoefficients(BlockCoeffCDFs{
		Transform: &transformCDFs,
		Coeff:     &coeffCDFs,
	}, &modeCtx, &coeffCtx, &scratch, BlockCoeffRequest{
		Transform: TransformTreeRequest{
			Size:          BlockSize4x4,
			VisibleW4:     1,
			VisibleH4:     1,
			Color:         parser.ColorConfig{MonoChrome: true},
			TransformMode: parser.TransformModeLargest,
		},
		LumaType:              transform.TypeDCTDCT,
		SkipAllZeroCoeffClear: true,
	}, func(block BlockCoeffBlock) error {
		if block.Result.EOB != 1 || len(block.Coeffs) != 16 {
			t.Fatalf("block=%+v coeffs=%d want eob=1 len=16", block, len(block.Coeffs))
		}
		for i, coeff := range block.Coeffs {
			want := int16(0)
			if i == 0 {
				want = 1
			}
			if coeff != want {
				t.Fatalf("coeff[%d]=%d want %d", i, coeff, want)
			}
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if total := result.TotalStats(); total.TXBs != 1 || total.NonZero != 1 || total.EOBTotal != 1 {
		t.Fatalf("stats=%+v want one nonzero txb", total)
	}
	if scratch.Coeff.Coeffs[17] != 0 {
		t.Fatalf("stale coeff[17]=%d want dirty clear", scratch.Coeff.Coeffs[17])
	}
	if scratch.Coeff.coeffDirtyLen != 1 || scratch.Coeff.InverseScan[0] != 0 {
		t.Fatalf("dirtyLen=%d dirty0=%d want only coeff 0 tracked", scratch.Coeff.coeffDirtyLen, scratch.Coeff.InverseScan[0])
	}
}

func TestDecodeBlockCoefficientsRejectsInvalidInputs(t *testing.T) {
	transformCDFs, coeffCDFs := mustBlockCoeffCDFs(t)
	valid := BlockCoeffRequest{
		Transform: TransformTreeRequest{Size: BlockSize4x4, VisibleW4: 1, VisibleH4: 1, TransformMode: parser.TransformModeLargest},
		LumaType:  transform.TypeDCTDCT,
	}
	var state DecodeState
	var modeCtx BlockModeContext
	var coeffCtx CoeffEntropyContext
	var scratch BlockCoeffScratch
	visitor := func(BlockCoeffBlock) error { return nil }
	cdfs := BlockCoeffCDFs{Transform: &transformCDFs, Coeff: &coeffCDFs}

	if _, err := (*DecodeState)(nil).DecodeBlockCoefficients(cdfs, &modeCtx, &coeffCtx, &scratch, valid, visitor); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("nil state err=%v want %v", err, ErrInvalidDecodeState)
	}
	if _, err := state.DecodeBlockCoefficients(BlockCoeffCDFs{Coeff: &coeffCDFs}, &modeCtx, &coeffCtx, &scratch, valid, visitor); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("nil transform cdfs err=%v want %v", err, ErrInvalidDecodeState)
	}
	if _, err := state.DecodeBlockCoefficients(cdfs, nil, &coeffCtx, &scratch, valid, visitor); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("nil mode ctx err=%v want %v", err, ErrInvalidDecodeState)
	}
	bad := valid
	bad.LumaType = transform.TypeCount
	if _, err := state.DecodeBlockCoefficients(cdfs, &modeCtx, &coeffCtx, &scratch, bad, visitor); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("bad luma class err=%v want %v", err, ErrInvalidDecodeState)
	}
	bad = valid
	bad.Transform.Color = parser.ColorConfig{SubsamplingX: true, SubsamplingY: true}
	bad.ChromaType = [2]transform.Type{transform.TypeDCTDCT, transform.TypeCount}
	if _, err := state.DecodeBlockCoefficients(cdfs, &modeCtx, &coeffCtx, &scratch, bad, visitor); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("bad chroma classes err=%v want %v", err, ErrInvalidDecodeState)
	}
}

func TestDecodeBlockCoefficientsAllocs(t *testing.T) {
	transformCDFs, coeffCDFs := mustBlockCoeffCDFs(t)
	payload := []byte{0x00}
	req := BlockCoeffRequest{
		Transform: TransformTreeRequest{Size: BlockSize4x4, VisibleW4: 1, VisibleH4: 1, Color: parser.ColorConfig{MonoChrome: true}, TransformMode: parser.TransformModeLargest},
		LumaType:  transform.TypeDCTDCT,
	}
	var state DecodeState
	var modeCtx BlockModeContext
	var coeffCtx CoeffEntropyContext
	var scratch BlockCoeffScratch
	cdfs := BlockCoeffCDFs{Transform: &transformCDFs, Coeff: &coeffCDFs}
	visit := func(block BlockCoeffBlock) error {
		if block.Plane != 0 || block.Result.EOB != 1 || block.Coeffs[0] != 1 {
			t.Fatalf("block=%+v coeff[0]=%d want luma eob=1 coeff=1", block, block.Coeffs[0])
		}
		return nil
	}

	allocs := testing.AllocsPerRun(1000, func() {
		modeCtx = BlockModeContext{}
		coeffCtx = CoeffEntropyContext{}
		if err := state.Reset(payload, Job{Offset: 0, Size: uint32(len(payload))}, DecodeOptions{}); err != nil {
			t.Fatal(err)
		}
		if _, err := state.DecodeBlockCoefficients(cdfs, &modeCtx, &coeffCtx, &scratch, req, visit); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("DecodeBlockCoefficients allocated: %f", allocs)
	}
}

func mustTXBSkipPayload(t *testing.T) []byte {
	t.Helper()
	for i := 0; i <= 0xff; i++ {
		var cdfs CoeffCDFs
		if err := cdfs.InitDefault(0); err != nil {
			t.Fatal(err)
		}
		var state DecodeState
		payload := []byte{byte(i)}
		if err := state.Reset(payload, Job{Offset: 0, Size: uint32(len(payload))}, DecodeOptions{}); err != nil {
			t.Fatal(err)
		}
		skip, err := state.ReadTXBSkip(&cdfs, TXBSkipRequest{Size: TransformSize4x4, Context: 0})
		if err != nil {
			t.Fatal(err)
		}
		if skip {
			return payload
		}
	}
	t.Fatal("no single-byte txb_skip payload found")
	return nil
}

func FuzzDecodeBlockCoefficients(f *testing.F) {
	f.Add([]byte{0x00}, uint8(BlockSize4x4), false, uint8(0), uint8(0), uint8(0))
	f.Add([]byte{0x00, 0x00}, uint8(BlockSize16x16), true, uint8(1), uint8(1), uint8(0))

	f.Fuzz(func(t *testing.T, payload []byte, rawBlock uint8, chroma bool, rawSSX uint8, rawSSY uint8, rawType uint8) {
		if len(payload) == 0 || len(payload) > 64 {
			return
		}
		block := BlockSize(rawBlock % uint8(blockSizeCount))
		dims, ok := block.Dimensions()
		if !ok {
			t.Fatal("invalid normalized block")
		}
		color := parser.ColorConfig{MonoChrome: !chroma, SubsamplingX: rawSSX&1 != 0, SubsamplingY: rawSSY&1 != 0}
		typ := transform.Type(rawType % uint8(transform.TypeCount))
		req := BlockCoeffRequest{
			Transform: TransformTreeRequest{
				Size:          block,
				VisibleW4:     dims.W4,
				VisibleH4:     dims.H4,
				Color:         color,
				TransformMode: parser.TransformModeLargest,
			},
			LumaType:   typ,
			ChromaType: [2]transform.Type{typ, typ},
		}
		transformCDFs, coeffCDFs := mustBlockCoeffCDFs(t)
		var state DecodeState
		if err := state.Reset(payload, Job{Offset: 0, Size: uint32(len(payload))}, DecodeOptions{}); err != nil {
			t.Fatal(err)
		}
		var modeCtx BlockModeContext
		var coeffCtx CoeffEntropyContext
		var scratch BlockCoeffScratch
		result, err := state.DecodeBlockCoefficients(BlockCoeffCDFs{Transform: &transformCDFs, Coeff: &coeffCDFs}, &modeCtx, &coeffCtx, &scratch, req, func(block BlockCoeffBlock) error {
			assertTXBDecodeInvariants(t, block.Result, block.Coeffs, block.Scan)
			return nil
		})
		if err != nil {
			if errors.Is(err, ErrInvalidDecodeState) {
				return
			}
			t.Fatalf("DecodeBlockCoefficients err=%v", err)
		}
		total := result.TotalStats()
		if total.TXBs != total.NonZero+total.AllZero {
			t.Fatalf("stats=%+v inconsistent counts", total)
		}
	})
}

func mustBlockCoeffCDFs(t *testing.T) (TransformCDFs, CoeffCDFs) {
	t.Helper()
	var transformCDFs TransformCDFs
	if err := transformCDFs.InitDefault(); err != nil {
		t.Fatal(err)
	}
	var coeffCDFs CoeffCDFs
	if err := coeffCDFs.InitDefault(0); err != nil {
		t.Fatal(err)
	}
	return transformCDFs, coeffCDFs
}
