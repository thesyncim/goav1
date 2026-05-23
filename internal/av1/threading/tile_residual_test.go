package threading

import (
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/reconstruct"
	"github.com/thesyncim/goav1/internal/av1/tile"
	"github.com/thesyncim/goav1/internal/av1/transform"
)

func TestFrameWorkBatchJobBlockLoopRequest(t *testing.T) {
	ctx := FrameWorkBatch{
		FrameWorkFrameContext: FrameWorkFrameContext{
			Sequence: FrameWorkSequenceContextFromHeader(parser.SequenceHeader{
				Use128x128Superblock: true,
				ColorConfig:          parser.ColorConfig{BitDepth: 8, MonoChrome: true},
			}),
			FrameSize: parser.FrameSize{CodedWidth: 300, Height: 260},
			SkipMode:  parser.SkipModeParams{Allowed: true, Enabled: true},
			CDEF:      parser.CDEFParams{Bits: 2, StrengthCount: 4},
			Delta:     parser.DeltaParams{DeltaQPresent: true, DeltaQResLog2: 1},
		},
		Jobs: []tile.Job{{Tile: 3, Row: 1, Col: 1, SBX: 1, SBY: 1, SBCols: 2, SBRows: 2}},
	}
	segMap := make([]uint8, 96*96)
	req, err := ctx.JobBlockLoopRequest(0, segMap, nil, 96)
	if err != nil {
		t.Fatal(err)
	}
	if req.Walk.Root != tile.BlockLevel128x128 ||
		req.Walk.MIColStart != 32 || req.Walk.MIRowStart != 32 ||
		req.Walk.MIColEnd != 76 || req.Walk.MIRowEnd != 66 {
		t.Fatalf("walk=%+v", req.Walk)
	}
	if !req.SkipMode.Enabled || req.CDEF.Bits != 2 ||
		!req.Delta.DeltaQPresent || req.SBSizeMIB != 32 || !req.Monochrome ||
		len(req.CurrentSegmentMap) != len(segMap) || req.SegmentMapStride != 96 {
		t.Fatalf("request=%+v", req)
	}
}

func TestFrameWorkBatchDecodeAndReconstructJobResiduals(t *testing.T) {
	output := testBatchFrame(t, frame.Format{
		Width:      64,
		Height:     64,
		BitDepth:   8,
		MonoChrome: true,
		Align:      64,
	})
	testFillFrame(output, 128)
	ctx := FrameWorkBatch{
		Output:  output,
		Payload: make([]byte, 256),
		FrameWorkFrameContext: FrameWorkFrameContext{
			Sequence: FrameWorkSequenceContextFromHeader(parser.SequenceHeader{
				ColorConfig: parser.ColorConfig{BitDepth: 8, MonoChrome: true},
			}),
			FrameSize:    parser.FrameSize{CodedWidth: 64, Height: 64},
			Quantization: parser.QuantizationParams{BaseQIdx: 64},
			TransformRef: parser.TransformReferenceParams{TransformMode: parser.TransformModeLargest},
		},
		Jobs: []tile.Job{{SBCols: 1, SBRows: 1, Offset: 0, Size: 256}},
	}
	var state tile.DecodeState
	if err := ctx.JobDecodeState(0, &state); err != nil {
		t.Fatal(err)
	}
	cdfs := mustFrameWorkTileResidualCDFs(t, ctx.Quantization.BaseQIdx)
	var scratch FrameWorkTileResidualScratch
	loopReq, err := ctx.JobBlockLoopRequest(0, nil, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	int32Scratch, residualScratch := testFrameWorkResidualScratch(t, ctx, transform.Size{Width: 64, Height: 64}, transform.TypeDCTDCT)
	predictions := 0
	transforms := 0
	stats, err := ctx.DecodeAndReconstructJobResiduals(0, &state, cdfs, &scratch, FrameWorkTileResidualRequest{
		Loop:          loopReq,
		TransformMode: ctx.TransformRef.TransformMode,
		Predict: func(visit tile.BlockLoopVisit) error {
			predictions++
			if visit.Block.MICol != 0 || visit.Block.MIRow != 0 {
				t.Fatalf("visit=%+v", visit.Block)
			}
			return nil
		},
		Transforms: func(visit tile.BlockLoopVisit) (FrameWorkBlockTransforms, error) {
			transforms++
			return ctx.ReadInterBlockTransforms(&state, visit)
		},
		Int32Scratch:    int32Scratch,
		ResidualScratch: residualScratch,
	})
	if err != nil {
		t.Fatal(err)
	}
	if stats.Loop.Blocks != 1 || stats.Loop.PartitionReads != 1 ||
		predictions != 1 || transforms != 1 || stats.Predictions != 1 {
		t.Fatalf("stats=%+v predictions=%d transforms=%d", stats, predictions, transforms)
	}
	if stats.CoefficientBlocks != 1 || stats.SkippedBlocks != 0 ||
		stats.TXBs != 1 || stats.Residuals != 1 || stats.TXBs != stats.NonZero+stats.AllZero {
		t.Fatalf("residual stats=%+v", stats)
	}
	if stats.Loop.CoefficientBlocks != 1 || stats.Loop.CoefficientTXBs != stats.TXBs {
		t.Fatalf("loop coefficient stats=%+v residual stats=%+v", stats.Loop, stats)
	}
}

func TestFrameWorkBatchDecodeAndReconstructJobResidualsRejectsInvalidInputs(t *testing.T) {
	ctx, state, cdfs, scratch, req := testFrameWorkResidualDriver(t)

	if _, err := ctx.DecodeAndReconstructJobResiduals(0, nil, cdfs, &scratch, req); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("nil state err=%v want %v", err, ErrInvalidBatch)
	}
	if _, err := ctx.DecodeAndReconstructJobResiduals(0, state, cdfs, nil, req); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("nil scratch err=%v want %v", err, ErrInvalidBatch)
	}
	noTransforms := req
	noTransforms.Transforms = nil
	if _, err := ctx.DecodeAndReconstructJobResiduals(0, state, cdfs, &scratch, noTransforms); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("nil transforms err=%v want %v", err, ErrInvalidBatch)
	}
}

func TestFrameWorkBatchReadInterBlockTransforms(t *testing.T) {
	ctx := FrameWorkBatch{
		FrameWorkFrameContext: FrameWorkFrameContext{
			Sequence: FrameWorkSequenceContextFromHeader(parser.SequenceHeader{
				ColorConfig: parser.ColorConfig{BitDepth: 8},
			}),
			Quantization: parser.QuantizationParams{BaseQIdx: 64},
		},
	}
	var cdfs tile.TransformTypeCDFs
	if err := cdfs.InitDefault(); err != nil {
		t.Fatal(err)
	}
	var state tile.DecodeState
	if err := state.Reset([]byte{0x00}, tile.Job{Offset: 0, Size: 1}, tile.DecodeOptions{BaseQIdx: 64}); err != nil {
		t.Fatal(err)
	}
	transforms, err := ctx.ReadInterBlockTransforms(&state, tile.BlockLoopVisit{})
	if err != nil {
		t.Fatal(err)
	}
	if !transforms.Inter || !transforms.ReadInterTX || transforms.Luma != transform.TypeDCTDCT {
		t.Fatalf("transforms=%+v", transforms)
	}
	var selector tile.InterCoeffTransformSelector
	selector.Reset(&state, &cdfs, ctx.FrameMode.ReducedTxSet, false, false)
	got, err := selector.SelectCoeffTransform(tile.CoeffTransformRequest{
		Block: tile.TransformBlock{Size: tile.TransformSize32x32},
	})
	if err != nil {
		t.Fatal(err)
	}
	if got != transform.TypeIDTX {
		t.Fatalf("tx type=%d want %d", got, transform.TypeIDTX)
	}

	ctx.FrameMode.ReducedTxSet = true
	if err := state.Reset([]byte{0x00}, tile.Job{Offset: 0, Size: 1}, tile.DecodeOptions{BaseQIdx: 64}); err != nil {
		t.Fatal(err)
	}
	transforms, err = ctx.ReadInterBlockTransforms(&state, tile.BlockLoopVisit{})
	if err != nil {
		t.Fatal(err)
	}
	selector.Reset(&state, nil, ctx.FrameMode.ReducedTxSet, true, false)
	got, err = selector.SelectCoeffTransform(tile.CoeffTransformRequest{
		Block: tile.TransformBlock{Size: tile.TransformSize16x16},
	})
	if err != nil {
		t.Fatal(err)
	}
	if got != transform.TypeDCTDCT {
		t.Fatalf("reduced tx type=%d want %d", got, transform.TypeDCTDCT)
	}
}

func TestFrameWorkBatchDecodeAndReconstructJobResidualsPropagatesCallbacks(t *testing.T) {
	ctx, state, cdfs, scratch, req := testFrameWorkResidualDriver(t)
	errPredict := errors.New("predict")
	req.Predict = func(tile.BlockLoopVisit) error { return errPredict }
	if _, err := ctx.DecodeAndReconstructJobResiduals(0, state, cdfs, &scratch, req); !errors.Is(err, errPredict) {
		t.Fatalf("predict err=%v want %v", err, errPredict)
	}

	ctx, state, cdfs, scratch, req = testFrameWorkResidualDriver(t)
	errTransform := errors.New("transform")
	req.Transforms = func(tile.BlockLoopVisit) (FrameWorkBlockTransforms, error) {
		return FrameWorkBlockTransforms{}, errTransform
	}
	if _, err := ctx.DecodeAndReconstructJobResiduals(0, state, cdfs, &scratch, req); !errors.Is(err, errTransform) {
		t.Fatalf("transform err=%v want %v", err, errTransform)
	}
}

func TestFrameWorkBatchDecodeAndReconstructJobResidualsAllocs(t *testing.T) {
	ctx, state, cdfs, scratch, req := testFrameWorkResidualDriver(t)
	allocs := testing.AllocsPerRun(1000, func() {
		if err := state.Reset(ctx.Payload, ctx.Jobs[0], tile.DecodeOptions{BaseQIdx: ctx.Quantization.BaseQIdx}); err != nil {
			t.Fatal(err)
		}
		if _, err := ctx.DecodeAndReconstructJobResiduals(0, state, cdfs, &scratch, req); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("DecodeAndReconstructJobResiduals allocated: %f", allocs)
	}
}

func FuzzFrameWorkBatchDecodeAndReconstructJobResiduals(f *testing.F) {
	f.Add([]byte{0x00, 0x00, 0x00, 0x00}, uint8(64))
	f.Add([]byte{0xff, 0x80, 0x00, 0x7f}, uint8(3))
	f.Add([]byte{0xaa, 0x55, 0x11, 0xee, 0x00}, uint8(127))

	f.Fuzz(func(t *testing.T, payload []byte, rawQ uint8) {
		if len(payload) == 0 || len(payload) > 512 {
			return
		}
		ctx := FrameWorkBatch{
			Output:  testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, MonoChrome: true, Align: 64}),
			Payload: payload,
			FrameWorkFrameContext: FrameWorkFrameContext{
				Sequence: FrameWorkSequenceContextFromHeader(parser.SequenceHeader{
					ColorConfig: parser.ColorConfig{BitDepth: 8, MonoChrome: true},
				}),
				FrameSize:    parser.FrameSize{CodedWidth: 64, Height: 64},
				Quantization: parser.QuantizationParams{BaseQIdx: rawQ | 1},
				TransformRef: parser.TransformReferenceParams{TransformMode: parser.TransformModeLargest},
			},
			Jobs: []tile.Job{{SBCols: 1, SBRows: 1, Offset: 0, Size: len(payload)}},
		}
		testFillFrame(ctx.Output, 128)
		var state tile.DecodeState
		if err := ctx.JobDecodeState(0, &state); err != nil {
			t.Fatal(err)
		}
		cdfs := mustFrameWorkTileResidualCDFs(t, ctx.Quantization.BaseQIdx)
		var scratch FrameWorkTileResidualScratch
		loopReq, err := ctx.JobBlockLoopRequest(0, nil, nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		int32Scratch, residualScratch := testFrameWorkResidualScratch(t, ctx, transform.Size{Width: 64, Height: 64}, transform.TypeDCTDCT)
		_, _ = ctx.DecodeAndReconstructJobResiduals(0, &state, cdfs, &scratch, FrameWorkTileResidualRequest{
			Loop:          loopReq,
			TransformMode: ctx.TransformRef.TransformMode,
			Transforms: func(visit tile.BlockLoopVisit) (FrameWorkBlockTransforms, error) {
				return ctx.ReadInterBlockTransforms(&state, visit)
			},
			Int32Scratch:    int32Scratch,
			ResidualScratch: residualScratch,
		})
	})
}

func testFrameWorkResidualDriver(t *testing.T) (FrameWorkBatch, *tile.DecodeState, FrameWorkTileResidualCDFs, FrameWorkTileResidualScratch, FrameWorkTileResidualRequest) {
	t.Helper()
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, MonoChrome: true, Align: 64})
	testFillFrame(output, 128)
	ctx := FrameWorkBatch{
		Output:  output,
		Payload: make([]byte, 256),
		FrameWorkFrameContext: FrameWorkFrameContext{
			Sequence: FrameWorkSequenceContextFromHeader(parser.SequenceHeader{
				ColorConfig: parser.ColorConfig{BitDepth: 8, MonoChrome: true},
			}),
			FrameSize:    parser.FrameSize{CodedWidth: 64, Height: 64},
			Quantization: parser.QuantizationParams{BaseQIdx: 64},
			TransformRef: parser.TransformReferenceParams{TransformMode: parser.TransformModeLargest},
		},
		Jobs: []tile.Job{{SBCols: 1, SBRows: 1, Offset: 0, Size: 256}},
	}
	state := &tile.DecodeState{}
	if err := ctx.JobDecodeState(0, state); err != nil {
		t.Fatal(err)
	}
	cdfs := mustFrameWorkTileResidualCDFs(t, ctx.Quantization.BaseQIdx)
	var scratch FrameWorkTileResidualScratch
	loopReq, err := ctx.JobBlockLoopRequest(0, nil, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	int32Scratch, residualScratch := testFrameWorkResidualScratch(t, ctx, transform.Size{Width: 64, Height: 64}, transform.TypeDCTDCT)
	req := FrameWorkTileResidualRequest{
		Loop:          loopReq,
		TransformMode: ctx.TransformRef.TransformMode,
		Transforms: func(visit tile.BlockLoopVisit) (FrameWorkBlockTransforms, error) {
			return ctx.ReadInterBlockTransforms(state, visit)
		},
		Int32Scratch:    int32Scratch,
		ResidualScratch: residualScratch,
	}
	return ctx, state, cdfs, scratch, req
}

func mustFrameWorkTileResidualCDFs(t *testing.T, baseQIndex uint8) FrameWorkTileResidualCDFs {
	t.Helper()
	var partition tile.PartitionCDFs
	if err := partition.InitDefault(); err != nil {
		t.Fatal(err)
	}
	var mode tile.BlockModeCDFs
	if err := mode.InitDefault(); err != nil {
		t.Fatal(err)
	}
	var transformCDFs tile.TransformCDFs
	if err := transformCDFs.InitDefault(); err != nil {
		t.Fatal(err)
	}
	var transformType tile.TransformTypeCDFs
	if err := transformType.InitDefault(); err != nil {
		t.Fatal(err)
	}
	var coeff tile.CoeffCDFs
	if err := coeff.InitDefault(baseQIndex); err != nil {
		t.Fatal(err)
	}
	return FrameWorkTileResidualCDFs{
		Loop: tile.BlockLoopCDFs{
			Partition: &partition,
			Mode:      &mode,
		},
		Coeff: tile.BlockCoeffCDFs{
			Transform: &transformCDFs,
			Coeff:     &coeff,
		},
		TransformType: &transformType,
	}
}

func testFrameWorkDCTTransforms(tile.BlockLoopVisit) (FrameWorkBlockTransforms, error) {
	return FrameWorkBlockTransforms{
		Inter:  true,
		Luma:   transform.TypeDCTDCT,
		Chroma: [2]transform.Type{transform.TypeDCTDCT, transform.TypeDCTDCT},
	}, nil
}

func testFrameWorkResidualScratch(t *testing.T, ctx FrameWorkBatch, size transform.Size, typ transform.Type) ([]int32, []int16) {
	t.Helper()
	q, lossless, err := ctx.BlockQuantizer(ctx.Quantization.BaseQIdx, 0, FrameWorkPlaneY)
	if err != nil {
		t.Fatal(err)
	}
	int32Len, int16Len, err := reconstruct.ScratchLen(reconstruct.Block{
		Size:      size,
		Transform: typ,
		Quantizer: q,
		Lossless:  lossless,
	})
	if err != nil {
		t.Fatal(err)
	}
	return make([]int32, int32Len), make([]int16, int16Len)
}
