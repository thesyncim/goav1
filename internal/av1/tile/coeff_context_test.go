package tile

import (
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/transform"
)

func TestCoeffEntropyContextTXBContextMatchesLibaom(t *testing.T) {
	tests := []struct {
		name string
		req  CoeffContextRequest
		seed func(*CoeffEntropyContext)
		want TXBContext
	}{
		{
			name: "luma same block uses ctx zero",
			req:  coeffContextReq(0, BlockSize4x4, TransformSize4x4, 0, 0),
			want: TXBContext{TXBSkipContext: 0, DCSignContext: 0},
		},
		{
			name: "luma larger block zero neighbors",
			req:  coeffContextReq(0, BlockSize8x8, TransformSize4x4, 0, 0),
			want: TXBContext{TXBSkipContext: 1, DCSignContext: 0},
		},
		{
			name: "luma top low left zero",
			req:  coeffContextReq(0, BlockSize8x8, TransformSize4x4, 0, 0),
			seed: func(ctx *CoeffEntropyContext) { ctx.Above[0][0] = 1 },
			want: TXBContext{TXBSkipContext: 2, DCSignContext: 0},
		},
		{
			name: "luma top high left zero",
			req:  coeffContextReq(0, BlockSize8x8, TransformSize4x4, 0, 0),
			seed: func(ctx *CoeffEntropyContext) { ctx.Above[0][0] = 4 },
			want: TXBContext{TXBSkipContext: 3, DCSignContext: 0},
		},
		{
			name: "luma top low left low",
			req:  coeffContextReq(0, BlockSize8x8, TransformSize4x4, 0, 0),
			seed: func(ctx *CoeffEntropyContext) {
				ctx.Above[0][0] = 1
				ctx.Left[0][0] = 1
			},
			want: TXBContext{TXBSkipContext: 4, DCSignContext: 0},
		},
		{
			name: "luma top high left low",
			req:  coeffContextReq(0, BlockSize8x8, TransformSize4x4, 0, 0),
			seed: func(ctx *CoeffEntropyContext) {
				ctx.Above[0][0] = 4
				ctx.Left[0][0] = 1
			},
			want: TXBContext{TXBSkipContext: 5, DCSignContext: 0},
		},
		{
			name: "luma top high left high",
			req:  coeffContextReq(0, BlockSize8x8, TransformSize4x4, 0, 0),
			seed: func(ctx *CoeffEntropyContext) {
				ctx.Above[0][0] = 4
				ctx.Left[0][0] = 4
			},
			want: TXBContext{TXBSkipContext: 6, DCSignContext: 0},
		},
		{
			name: "positive dc sign",
			req:  coeffContextReq(0, BlockSize4x4, TransformSize4x4, 0, 0),
			seed: func(ctx *CoeffEntropyContext) { ctx.Above[0][0] = 2 << CoeffContextBits },
			want: TXBContext{TXBSkipContext: 0, DCSignContext: 2},
		},
		{
			name: "negative dc sign",
			req:  coeffContextReq(0, BlockSize4x4, TransformSize4x4, 0, 0),
			seed: func(ctx *CoeffEntropyContext) { ctx.Left[0][0] = 1 << CoeffContextBits },
			want: TXBContext{TXBSkipContext: 0, DCSignContext: 1},
		},
		{
			name: "balanced dc signs",
			req:  coeffContextReq(0, BlockSize4x4, TransformSize4x4, 0, 0),
			seed: func(ctx *CoeffEntropyContext) {
				ctx.Above[0][0] = 2 << CoeffContextBits
				ctx.Left[0][0] = 1 << CoeffContextBits
			},
			want: TXBContext{TXBSkipContext: 0, DCSignContext: 0},
		},
		{
			name: "chroma same area offset",
			req:  coeffContextReq(1, BlockSize4x4, TransformSize4x4, 0, 0),
			want: TXBContext{TXBSkipContext: 7, DCSignContext: 0},
		},
		{
			name: "chroma same area both edges nonzero",
			req:  coeffContextReq(1, BlockSize4x4, TransformSize4x4, 0, 0),
			seed: func(ctx *CoeffEntropyContext) {
				ctx.Above[1][0] = 1
				ctx.Left[1][0] = 1
			},
			want: TXBContext{TXBSkipContext: 9, DCSignContext: 0},
		},
		{
			name: "chroma larger block offset",
			req:  coeffContextReq(1, BlockSize8x8, TransformSize4x4, 0, 0),
			seed: func(ctx *CoeffEntropyContext) { ctx.Above[1][0] = 1 },
			want: TXBContext{TXBSkipContext: 11, DCSignContext: 0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var ctx CoeffEntropyContext
			if tt.seed != nil {
				tt.seed(&ctx)
			}
			got, err := ctx.TXBContext(tt.req)
			if err != nil {
				t.Fatal(err)
			}
			if got != tt.want {
				t.Fatalf("TXBContext=%+v want %+v", got, tt.want)
			}
		})
	}
}

func TestCoeffEntropyContextMarkAndResetMatchesLibaom(t *testing.T) {
	var ctx CoeffEntropyContext
	req := coeffContextReq(0, BlockSize16x16, TransformSize8x8, 4, 5)
	req.VisibleW4 = 1
	req.VisibleH4 = 2

	if err := ctx.MarkTXB(req, TXBDecodeResult{EOB: 3, CulLevel: 17}); err != nil {
		t.Fatal(err)
	}
	if ctx.Above[0][4] != 17 || ctx.Above[0][5] != 0 {
		t.Fatalf("above context=[%d,%d] want [17,0]", ctx.Above[0][4], ctx.Above[0][5])
	}
	if ctx.Left[0][5] != 17 || ctx.Left[0][6] != 17 {
		t.Fatalf("left context=[%d,%d] want [17,17]", ctx.Left[0][5], ctx.Left[0][6])
	}

	if err := ctx.MarkTXB(req, TXBDecodeResult{AllZero: true}); err != nil {
		t.Fatal(err)
	}
	if ctx.Above[0][4] != 0 || ctx.Left[0][5] != 0 {
		t.Fatalf("all-zero mark left contexts above=%d left=%d want zero", ctx.Above[0][4], ctx.Left[0][5])
	}

	if err := ctx.MarkTXB(req, TXBDecodeResult{EOB: 1, CulLevel: 17}); err != nil {
		t.Fatal(err)
	}
	if err := ctx.ResetBlock(0, BlockSize16x16, 4, 5); err != nil {
		t.Fatal(err)
	}
	for i := 4; i < 8; i++ {
		if ctx.Above[0][i] != 0 {
			t.Fatalf("above[%d]=%d want zero after reset", i, ctx.Above[0][i])
		}
	}
	for i := 5; i < 9; i++ {
		if ctx.Left[0][i] != 0 {
			t.Fatalf("left[%d]=%d want zero after reset", i, ctx.Left[0][i])
		}
	}
}

func TestReadCoefficientsTXBWithContext(t *testing.T) {
	txSize, err := TransformSize4x4.TransformSize()
	if err != nil {
		t.Fatal(err)
	}
	scan, scratch := coeffScanAndScratch(t, TransformSize4x4, txSize, transform.Class2D)
	coeffs := make([]int16, len(scan))
	var cdfs CoeffCDFs
	if err := cdfs.InitDefault(0); err != nil {
		t.Fatal(err)
	}
	var state DecodeState
	if err := state.Reset([]byte{0x00}, Job{Offset: 0, Size: 1}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	var ctx CoeffEntropyContext
	ctxReq := coeffContextReq(0, BlockSize4x4, TransformSize4x4, 0, 0)
	result, err := state.ReadCoefficientsTXBWithContext(&cdfs, &ctx, ctxReq, TXBDecodeRequest{
		Class:           transform.Class2D,
		EOBMultiContext: 0,
	}, coeffs, scan, scratch)
	if err != nil {
		t.Fatal(err)
	}
	if result.EOB != 1 || result.CulLevel != 17 || coeffs[0] != 1 {
		t.Fatalf("result=%+v coeff[0]=%d want eob=1 cul=17 coeff=1", result, coeffs[0])
	}
	if ctx.Above[0][0] != 17 || ctx.Left[0][0] != 17 {
		t.Fatalf("context above=%d left=%d want 17", ctx.Above[0][0], ctx.Left[0][0])
	}
	next, err := ctx.TXBContext(ctxReq)
	if err != nil {
		t.Fatal(err)
	}
	if next.DCSignContext != 2 || next.TXBSkipContext != 0 {
		t.Fatalf("next context=%+v want dc sign 2 skip 0", next)
	}
}

func TestCoeffEntropyContextRejectsInvalidInputs(t *testing.T) {
	var nilCtx *CoeffEntropyContext
	if _, err := nilCtx.TXBContext(coeffContextReq(0, BlockSize4x4, TransformSize4x4, 0, 0)); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("nil context err=%v want %v", err, ErrInvalidDecodeState)
	}
	var ctx CoeffEntropyContext
	if _, err := ctx.TXBContext(coeffContextReq(3, BlockSize4x4, TransformSize4x4, 0, 0)); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("bad plane err=%v want %v", err, ErrInvalidDecodeState)
	}
	if _, err := ctx.TXBContext(coeffContextReq(0, blockSizeCount, TransformSize4x4, 0, 0)); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("bad block err=%v want %v", err, ErrInvalidDecodeState)
	}
	if _, err := ctx.TXBContext(coeffContextReq(0, BlockSize4x4, transformSizeCount, 0, 0)); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("bad tx err=%v want %v", err, ErrInvalidDecodeState)
	}
	if _, err := ctx.TXBContext(coeffContextReq(0, BlockSize4x4, TransformSize4x4, MaxBlockModeSlots, 0)); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("bad x err=%v want %v", err, ErrInvalidDecodeState)
	}
	badVisible := coeffContextReq(0, BlockSize4x4, TransformSize4x4, 0, 0)
	badVisible.VisibleW4 = 2
	if err := ctx.MarkTXB(badVisible, TXBDecodeResult{EOB: 1, CulLevel: 1}); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("bad visible err=%v want %v", err, ErrInvalidDecodeState)
	}
	if err := ctx.MarkTXB(coeffContextReq(0, BlockSize4x4, TransformSize4x4, 0, 0), TXBDecodeResult{EOB: 1, CulLevel: 3 << CoeffContextBits}); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("bad cul err=%v want %v", err, ErrInvalidDecodeState)
	}
	ctx.Above[0][0] = 3 << CoeffContextBits
	if _, err := ctx.TXBContext(coeffContextReq(0, BlockSize4x4, TransformSize4x4, 0, 0)); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("bad sign err=%v want %v", err, ErrInvalidDecodeState)
	}
}

func TestReadCoefficientsTXBWithContextAllocs(t *testing.T) {
	txSize, err := TransformSize4x4.TransformSize()
	if err != nil {
		t.Fatal(err)
	}
	scan, scratch := coeffScanAndScratch(t, TransformSize4x4, txSize, transform.Class2D)
	coeffs := make([]int16, len(scan))
	payload := []byte{0x00}
	ctxReq := coeffContextReq(0, BlockSize4x4, TransformSize4x4, 0, 0)
	var cdfs CoeffCDFs
	var state DecodeState
	var ctx CoeffEntropyContext

	allocs := testing.AllocsPerRun(1000, func() {
		ctx = CoeffEntropyContext{}
		if err := cdfs.InitDefault(0); err != nil {
			t.Fatal(err)
		}
		if err := state.Reset(payload, Job{Offset: 0, Size: uint32(len(payload))}, DecodeOptions{}); err != nil {
			t.Fatal(err)
		}
		result, err := state.ReadCoefficientsTXBWithContext(&cdfs, &ctx, ctxReq, TXBDecodeRequest{
			Class: transform.Class2D,
		}, coeffs, scan, scratch)
		if err != nil {
			t.Fatal(err)
		}
		if result.EOB != 1 || coeffs[0] != 1 {
			t.Fatalf("txb result=%+v coeff[0]=%d want eob=1 coeff=1", result, coeffs[0])
		}
	})
	if allocs != 0 {
		t.Fatalf("contextual txb coefficient decode allocated: %f", allocs)
	}
}

func FuzzCoeffEntropyContext(f *testing.F) {
	f.Add(uint8(0), uint8(BlockSize4x4), uint8(TransformSize4x4), uint8(0), uint8(0), uint8(1), uint8(1), uint8(17), false)
	f.Add(uint8(1), uint8(BlockSize8x8), uint8(TransformSize4x4), uint8(3), uint8(5), uint8(1), uint8(1), uint8(0), true)
	f.Add(uint8(2), uint8(BlockSize16x16), uint8(TransformSize8x8), uint8(7), uint8(11), uint8(2), uint8(1), uint8(9), false)

	f.Fuzz(func(t *testing.T, rawPlane uint8, rawBlock uint8, rawTX uint8, rawX uint8, rawY uint8, rawVisibleW uint8, rawVisibleH uint8, rawCul uint8, allZero bool) {
		plane := int(rawPlane % 3)
		block := BlockSize(rawBlock % uint8(blockSizeCount))
		size := TransformSize(rawTX % uint8(transformSizeCount))
		dims, ok := size.Dimensions()
		if !ok {
			t.Fatal("invalid normalized tx size")
		}
		blockDims, ok := block.Dimensions()
		if !ok {
			t.Fatal("invalid normalized block size")
		}
		spanW := max(int(blockDims.W4), int(dims.W4))
		spanH := max(int(blockDims.H4), int(dims.H4))
		xLimit := MaxBlockModeSlots - spanW + 1
		yLimit := MaxBlockModeSlots - spanH + 1
		req := coeffContextReq(plane, block, size, int(rawX)%xLimit, int(rawY)%yLimit)
		req.VisibleW4 = 1 + rawVisibleW%dims.W4
		req.VisibleH4 = 1 + rawVisibleH%dims.H4

		cul := rawCul % uint8(CoeffContextMask+(2<<CoeffContextBits)+1)
		for !validCoeffEntropyValue(cul) {
			cul--
		}
		result := TXBDecodeResult{EOB: 1, CulLevel: cul, AllZero: allZero}
		if allZero {
			result.EOB = 0
			result.CulLevel = 0
		}

		var ctx CoeffEntropyContext
		if err := ctx.MarkTXB(req, result); err != nil {
			t.Fatalf("MarkTXB err=%v", err)
		}
		txbCtx, err := ctx.TXBContext(req)
		if err != nil {
			t.Fatalf("TXBContext err=%v", err)
		}
		if txbCtx.TXBSkipContext < 0 || txbCtx.TXBSkipContext >= TXBSkipContexts || txbCtx.DCSignContext < 0 || txbCtx.DCSignContext > 2 {
			t.Fatalf("txb context out of range: %+v", txbCtx)
		}
		if err := ctx.ResetBlock(plane, block, int(req.X4), int(req.Y4)); err != nil {
			t.Fatalf("ResetBlock err=%v", err)
		}
	})
}

func coeffContextReq(plane int, block BlockSize, tx TransformSize, x4 int, y4 int) CoeffContextRequest {
	return CoeffContextRequest{
		Plane:      uint8(plane),
		PlaneBlock: block,
		Size:       tx,
		X4:         uint8(x4),
		Y4:         uint8(y4),
	}
}

func BenchmarkCoeffEntropyContextTXBContextTrustedLuma(b *testing.B) {
	benchmarkCoeffEntropyContextTXBContextTrusted(b, 0)
}

func BenchmarkCoeffEntropyContextTXBContextTrustedChroma(b *testing.B) {
	benchmarkCoeffEntropyContextTXBContextTrusted(b, 1)
}

func BenchmarkCoeffEntropyContextSetTXBContextKnownFull(b *testing.B) {
	benchmarkCoeffEntropyContextSetTXBContextKnown(b, 4, 4)
}

func BenchmarkCoeffEntropyContextSetTXBContextKnownClipped(b *testing.B) {
	benchmarkCoeffEntropyContextSetTXBContextKnown(b, 2, 3)
}

func benchmarkCoeffEntropyContextTXBContextTrusted(b *testing.B, plane int) {
	req := coeffContextReq(plane, BlockSize16x16, TransformSize4x4, 4, 5)
	txDims, ok := req.Size.Dimensions()
	if !ok {
		b.Fatal("invalid transform size")
	}
	blockDims, ok := req.PlaneBlock.Dimensions()
	if !ok {
		b.Fatal("invalid block size")
	}
	var ctx CoeffEntropyContext
	for i := 0; i < MaxBlockModeSlots; i++ {
		ctx.Above[plane][i] = uint8((i % 7) + 1)
		ctx.Left[plane][i] = uint8(((i + 3) % 7) + 1)
	}

	sum := 0
	b.ReportAllocs()
	for b.Loop() {
		txb := ctx.txbContextTrusted(req, txDims, blockDims)
		sum += int(txb.TXBSkipContext) + int(txb.DCSignContext)
	}
	benchmarkCoeffContextSink = sum
}

func benchmarkCoeffEntropyContextSetTXBContextKnown(b *testing.B, visibleW int, visibleH int) {
	req := coeffContextReq(0, BlockSize16x16, TransformSize4x4, 4, 5)
	txDims, ok := req.Size.Dimensions()
	if !ok {
		b.Fatal("invalid transform size")
	}
	var ctx CoeffEntropyContext
	value := uint8(CoeffContextMask + (2 << CoeffContextBits))

	b.ReportAllocs()
	for b.Loop() {
		ctx.setTXBContextKnown(req, txDims, visibleW, visibleH, value)
	}
	benchmarkCoeffContextSink = int(ctx.Above[0][4]) + int(ctx.Left[0][5])
}

var benchmarkCoeffContextSink int
