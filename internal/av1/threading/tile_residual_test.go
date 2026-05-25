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
				EnableDualFilter:     true,
				EnableFilterIntra:    true,
				EnableOrderHint:      true,
				OrderHintBits:        5,
				ColorConfig:          parser.ColorConfig{BitDepth: 8, MonoChrome: true},
			}),
			FrameHeader:         parser.FrameHeaderPrefix{OrderHint: 9},
			FrameSize:           parser.FrameSize{CodedWidth: 300, Height: 260},
			TileInfo:            parser.TileInfo{InterpolationFilter: parser.InterpolationSwitchable, UseRefFrameMVS: true},
			GlobalMotion:        parser.DefaultGlobalMotionParams(),
			ReferenceOrderHints: [parser.InterRefsPerFrame]uint32{1, 9, 10, 4, 5, 6, 7},
			SkipMode:            parser.SkipModeParams{Allowed: true, Enabled: true},
			CDEF:                parser.CDEFParams{Bits: 2, StrengthCount: 4},
			Delta:               parser.DeltaParams{DeltaQPresent: true, DeltaQResLog2: 1},
		},
		Jobs: []tile.Job{{Tile: 3, Row: 1, Col: 1, SBX: 1, SBY: 1, SBCols: 2, SBRows: 2}},
	}
	segMap := make([]uint8, 96*96)
	_, _, mvLength, err := ctx.ReferenceMVFrameShape()
	if err != nil {
		t.Fatal(err)
	}
	currentMVFrame, err := ctx.BindReferenceMVFrame(make([]tile.ReferenceMVEntry, mvLength))
	if err != nil {
		t.Fatal(err)
	}
	ctx.CurrentMVFrame = &currentMVFrame
	temporalMVs, err := ctx.BindTemporalMotionField(make([]tile.TemporalMotionEntry, mvLength))
	if err != nil {
		t.Fatal(err)
	}
	ctx.TemporalMVs = &temporalMVs
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
		len(req.CurrentSegmentMap) != len(segMap) || req.SegmentMapStride != 96 ||
		req.InterpolationFilter != parser.InterpolationSwitchable || !req.EnableDualFilter ||
		!req.EnableFilterIntra ||
		req.GlobalMotion[0] != parser.DefaultWarpedMotionParams() ||
		!req.EnableOrderHint || req.OrderHintBits != 5 ||
		req.CurrentOrderHint != 9 || req.ReferenceOrderHints[4] != 5 ||
		req.RefFrameSide[tile.ReferenceFrameLast2] != -1 ||
		req.RefFrameSide[tile.ReferenceFrameLast3] != 1 ||
		req.CurrentMVFrame != &currentMVFrame ||
		req.TemporalMVs != &temporalMVs ||
		!req.UseRefFrameMVS || !req.TemporalMVSampleUnavailable {
		t.Fatalf("request=%+v", req)
	}
	rootCols, err := ctx.JobBlockLoopContextRootColumns(0)
	if err != nil {
		t.Fatal(err)
	}
	if rootCols != 2 {
		t.Fatalf("root context cols=%d want 2", rootCols)
	}
}

func TestFrameWorkBatchReferenceMVFrameShapeAndBind(t *testing.T) {
	ctx := FrameWorkBatch{
		FrameWorkFrameContext: FrameWorkFrameContext{
			Sequence: FrameWorkSequenceContextFromHeader(parser.SequenceHeader{
				ColorConfig: parser.ColorConfig{BitDepth: 8},
			}),
			FrameSize: parser.FrameSize{CodedWidth: 130, Height: 65},
		},
	}
	cols, rows, length, err := ctx.ReferenceMVFrameShape()
	if err != nil {
		t.Fatal(err)
	}
	if cols != 17 || rows != 9 || length != 153 {
		t.Fatalf("shape cols=%d rows=%d length=%d want 17,9,153", cols, rows, length)
	}

	entries := make([]tile.ReferenceMVEntry, length+1)
	for i := range entries {
		entries[i] = tile.ReferenceMVEntry{
			Ref:   tile.ReferenceFrameGolden,
			Valid: true,
		}
	}
	got, err := ctx.BindReferenceMVFrame(entries)
	if err != nil {
		t.Fatal(err)
	}
	if got.Cols != cols || got.Rows != rows || got.Stride != cols || len(got.Entries) != length {
		t.Fatalf("mv frame=%+v len=%d", got, len(got.Entries))
	}
	for i, entry := range got.Entries {
		if entry != (tile.ReferenceMVEntry{Ref: tile.ReferenceFrameNone}) {
			t.Fatalf("entry %d=%+v want NONE", i, entry)
		}
	}
	if !entries[length].Valid || entries[length].Ref != tile.ReferenceFrameGolden {
		t.Fatalf("caller storage past MV_REF grid was modified: %+v", entries[length])
	}

	temporalEntries := make([]tile.TemporalMotionEntry, length+1)
	for i := range temporalEntries {
		temporalEntries[i] = tile.TemporalMotionEntry{
			Valid: true,
		}
	}
	temporal, err := ctx.BindTemporalMotionField(temporalEntries)
	if err != nil {
		t.Fatal(err)
	}
	if temporal.Cols != cols || temporal.Rows != rows || temporal.Stride != cols || len(temporal.Entries) != length {
		t.Fatalf("temporal field=%+v len=%d", temporal, len(temporal.Entries))
	}
	for i, entry := range temporal.Entries {
		if entry.Valid {
			t.Fatalf("temporal entry %d=%+v want invalid", i, entry)
		}
	}
	if !temporalEntries[length].Valid {
		t.Fatalf("caller temporal storage past grid was modified: %+v", temporalEntries[length])
	}

	if _, err := ctx.BindReferenceMVFrame(entries[:length-1]); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("short bind err=%v want %v", err, ErrInvalidBatch)
	}
	if _, err := ctx.BindTemporalMotionField(temporalEntries[:length-1]); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("short temporal bind err=%v want %v", err, ErrInvalidBatch)
	}
	if _, _, _, err := (FrameWorkBatch{}).ReferenceMVFrameShape(); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("invalid shape err=%v want %v", err, ErrInvalidBatch)
	}
}

func TestFrameWorkBatchReferenceMVFrameBindAllocs(t *testing.T) {
	ctx := FrameWorkBatch{
		FrameWorkFrameContext: FrameWorkFrameContext{
			Sequence: FrameWorkSequenceContextFromHeader(parser.SequenceHeader{
				ColorConfig: parser.ColorConfig{BitDepth: 8},
			}),
			FrameSize: parser.FrameSize{CodedWidth: 64, Height: 64},
		},
	}
	_, _, length, err := ctx.ReferenceMVFrameShape()
	if err != nil {
		t.Fatal(err)
	}
	entries := make([]tile.ReferenceMVEntry, length)
	temporalEntries := make([]tile.TemporalMotionEntry, length)
	allocs := testing.AllocsPerRun(1000, func() {
		if _, err := ctx.BindReferenceMVFrame(entries); err != nil {
			t.Fatal(err)
		}
		if _, err := ctx.BindTemporalMotionField(temporalEntries); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("MV frame binds allocated: %f", allocs)
	}
}

func TestFrameWorkBlockLoopRefFrameSideMatchesLibaom(t *testing.T) {
	refs := [parser.InterRefsPerFrame]uint32{7, 8, 9, 24, 4, 5, 6}
	side, err := frameWorkBlockLoopRefFrameSide(true, 5, 8, refs)
	if err != nil {
		t.Fatal(err)
	}
	if side[tile.ReferenceFrameLast] != 0 ||
		side[tile.ReferenceFrameLast2] != -1 ||
		side[tile.ReferenceFrameLast3] != 1 ||
		side[tile.ReferenceFrameGolden] != 0 {
		t.Fatalf("side=%v", side)
	}

	disabled, err := frameWorkBlockLoopRefFrameSide(false, 0, 8, refs)
	if err != nil {
		t.Fatal(err)
	}
	if disabled != ([parser.InterRefsPerFrame]int8{}) {
		t.Fatalf("disabled side=%v want zero", disabled)
	}

	if _, err := frameWorkBlockLoopRefFrameSide(true, 0, 8, refs); err == nil {
		t.Fatal("expected invalid order-hint bits error")
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

func TestFrameWorkBatchDecodeAndReconstructJobResidualsDefault(t *testing.T) {
	ctx, state, cdfs, scratch, req := testFrameWorkResidualDriver(t)
	afterBlocks := 0

	stats, err := ctx.DecodeAndReconstructJobResidualsDefault(0, state, cdfs, &scratch, FrameWorkTileResidualDefaultRequest{
		Int32Scratch:    req.Int32Scratch,
		ResidualScratch: req.ResidualScratch,
		AfterBlock: func(visit tile.BlockLoopVisit) error {
			afterBlocks++
			if !visit.CoefficientsValid {
				t.Fatalf("visit missing coefficients: %+v", visit)
			}
			return nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if stats.Loop.Blocks != 1 || stats.CoefficientBlocks != 1 || stats.TXBs != 1 || stats.Residuals != 1 || afterBlocks != 1 {
		t.Fatalf("stats=%+v afterBlocks=%d", stats, afterBlocks)
	}
}

func TestFrameWorkBatchDecodeAndReconstructJobResidualsUsesBatchCDEFIndexMap(t *testing.T) {
	ctx, state, cdfs, scratch, req := testFrameWorkResidualDriver(t)
	ctx.CDEF = parser.CDEFParams{Bits: 0, StrengthCount: 1}
	_, _, length, err := ctx.CDEFIndexMapShape()
	if err != nil {
		t.Fatal(err)
	}
	cdefMap, err := ctx.BindCDEFIndexMap(make([]uint8, length), make([]bool, length))
	if err != nil {
		t.Fatal(err)
	}
	ctx.CDEFIndexMap = &cdefMap

	stats, err := ctx.DecodeAndReconstructJobResiduals(0, state, cdfs, &scratch, req)
	if err != nil {
		t.Fatal(err)
	}
	if stats.CoefficientBlocks != 1 {
		t.Fatalf("stats=%+v", stats)
	}
	if !cdefMap.Read[0] || cdefMap.Index[0] != 0 {
		t.Fatalf("batch cdef map read=%v index=%d want true,0", cdefMap.Read[0], cdefMap.Index[0])
	}
}

func TestFrameWorkBatchDecodeAndReconstructJobResidualsUsesBatchLoopFilterMap(t *testing.T) {
	ctx, state, cdfs, scratch, req := testFrameWorkResidualDriver(t)
	_, _, length, err := ctx.LoopFilterMapShape()
	if err != nil {
		t.Fatal(err)
	}
	lfMap, err := ctx.BindLoopFilterMap(make([]FrameWorkLoopFilterBlockRecord, length))
	if err != nil {
		t.Fatal(err)
	}
	ctx.LoopFilterMap = &lfMap
	state.DeltaLFFromBase = -2
	state.DeltaLF = [tile.FrameLoopFilterCount]int8{1, 2, 3, 4}

	stats, err := ctx.DecodeAndReconstructJobResiduals(0, state, cdfs, &scratch, req)
	if err != nil {
		t.Fatal(err)
	}
	if stats.CoefficientBlocks != 1 {
		t.Fatalf("stats=%+v", stats)
	}
	record := lfMap.Records[0]
	if !record.Valid || record.Block.MICol != 0 || record.Block.MIRow != 0 ||
		record.DeltaLFFromBase != -2 || record.DeltaLF != state.DeltaLF {
		t.Fatalf("loop filter record=%+v state delta=%v", record, state.DeltaLF)
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
	shortCarrier := req
	scratch.LoopContext.Above = []tile.BlockLoopRootAboveContext{}
	if _, err := ctx.DecodeAndReconstructJobResiduals(0, state, cdfs, &scratch, shortCarrier); !errors.Is(err, tile.ErrInvalidDecodeState) {
		t.Fatalf("short scratch context carrier err=%v want %v", err, tile.ErrInvalidDecodeState)
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

	intrabc := tile.BlockLoopVisit{
		Prediction: tile.BlockPredictionModeResult{
			Valid:        true,
			Intrabc:      true,
			IntrabcValid: true,
		},
	}
	if _, err := ctx.ReadInterBlockTransforms(&state, intrabc); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("intrabc inter transforms err=%v want %v", err, ErrInvalidBatch)
	}
	if _, err := ctx.ReadBlockTransforms(&state, intrabc); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("intrabc default transforms err=%v want %v", err, ErrInvalidBatch)
	}
}

func TestFrameWorkBatchReadIntraBlockTransforms(t *testing.T) {
	ctx := FrameWorkBatch{
		FrameWorkFrameContext: FrameWorkFrameContext{
			Sequence: FrameWorkSequenceContextFromHeader(parser.SequenceHeader{
				ColorConfig: parser.ColorConfig{BitDepth: 8, SubsamplingX: true, SubsamplingY: true},
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
	visit := tile.BlockLoopVisit{
		Prediction: tile.BlockPredictionModeResult{
			Valid:           true,
			Intra:           true,
			LumaMode:        tile.IntraModeDC,
			ChromaMode:      tile.ChromaIntraModeVertical,
			ChromaModeValid: true,
		},
	}
	transforms, err := ctx.ReadIntraBlockTransforms(&state, visit)
	if err != nil {
		t.Fatal(err)
	}
	if transforms.Inter || !transforms.ReadIntraTX || transforms.Luma != transform.TypeDCTDCT {
		t.Fatalf("transforms=%+v", transforms)
	}
	var selector tile.IntraCoeffTransformSelector
	selector.Reset(&state, &cdfs, ctx.FrameMode.ReducedTxSet, false, false, 64, tile.IntraModeDC, tile.ChromaIntraModeVertical)
	got, err := selector.SelectCoeffTransform(tile.CoeffTransformRequest{
		Plane: 0,
		Block: tile.TransformBlock{Size: tile.TransformSize8x8},
	})
	if err != nil {
		t.Fatal(err)
	}
	if got != transform.TypeIDTX {
		t.Fatalf("intra luma tx type=%d want %d", got, transform.TypeIDTX)
	}
	got, err = selector.SelectCoeffTransform(tile.CoeffTransformRequest{
		Plane: 1,
		Block: tile.TransformBlock{Size: tile.TransformSize8x8},
	})
	if err != nil {
		t.Fatal(err)
	}
	if got != transform.TypeADSTDCT {
		t.Fatalf("intra chroma tx type=%d want %d", got, transform.TypeADSTDCT)
	}

	if _, err := ctx.ReadIntraBlockTransforms(nil, visit); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("nil state err=%v want %v", err, ErrInvalidBatch)
	}
	visit.Prediction.Intra = false
	if _, err := ctx.ReadIntraBlockTransforms(&state, visit); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("inter visit err=%v want %v", err, ErrInvalidBatch)
	}
}

func TestFrameWorkTileResidualControllerWiresIntraTransformSelector(t *testing.T) {
	ctx := FrameWorkBatch{
		FrameWorkFrameContext: FrameWorkFrameContext{
			Sequence: FrameWorkSequenceContextFromHeader(parser.SequenceHeader{
				ColorConfig: parser.ColorConfig{BitDepth: 8, SubsamplingX: true, SubsamplingY: true},
			}),
			Quantization: parser.QuantizationParams{BaseQIdx: 64},
			TransformRef: parser.TransformReferenceParams{TransformMode: parser.TransformModeLargest},
		},
	}
	var state tile.DecodeState
	if err := state.Reset([]byte{0x00}, tile.Job{Offset: 0, Size: 1}, tile.DecodeOptions{BaseQIdx: 64}); err != nil {
		t.Fatal(err)
	}
	cdfs := mustFrameWorkTileResidualCDFs(t, 64)
	var scratch FrameWorkTileResidualScratch
	controller := frameWorkTileResidualLoopController{
		batch:   ctx,
		state:   &state,
		cdfs:    cdfs,
		scratch: &scratch,
		req: FrameWorkTileResidualRequest{
			TransformMode: parser.TransformModeLargest,
			Transforms: func(visit tile.BlockLoopVisit) (FrameWorkBlockTransforms, error) {
				return ctx.ReadIntraBlockTransforms(&state, visit)
			},
		},
	}
	visit := tile.BlockLoopVisit{
		Block:     tile.BlockVisit{Size: tile.BlockSize16x16, VisibleW4: 4, VisibleH4: 4},
		Segment:   parser.SegmentData{RefFrame: -1},
		SegmentID: 0,
		Prediction: tile.BlockPredictionModeResult{
			Valid:           true,
			Intra:           true,
			LumaMode:        tile.IntraModeDC,
			ChromaMode:      tile.ChromaIntraModeVertical,
			ChromaModeValid: true,
		},
	}
	req, err := controller.SelectBlockCoeffRequest(visit)
	if err != nil {
		t.Fatal(err)
	}
	got, err := req.TransformSelect.SelectCoeffTransform(tile.CoeffTransformRequest{
		Plane: 0,
		Block: tile.TransformBlock{Size: tile.TransformSize8x8},
	})
	if err != nil {
		t.Fatal(err)
	}
	if got != transform.TypeIDTX {
		t.Fatalf("controller intra luma tx=%d want %d", got, transform.TypeIDTX)
	}
	got, err = req.TransformSelect.SelectCoeffTransform(tile.CoeffTransformRequest{
		Plane: 1,
		Block: tile.TransformBlock{Size: tile.TransformSize8x8},
	})
	if err != nil {
		t.Fatal(err)
	}
	if got != transform.TypeADSTDCT {
		t.Fatalf("controller intra chroma tx=%d want %d", got, transform.TypeADSTDCT)
	}
}

func TestFrameWorkTileResidualCDFStorageInitDefault(t *testing.T) {
	var storage FrameWorkTileResidualCDFStorage
	if err := storage.InitDefault(64); err != nil {
		t.Fatal(err)
	}
	cdfs := storage.CDFs()
	if cdfs.Loop.Partition != &storage.Partition ||
		cdfs.Loop.Mode != &storage.Mode ||
		cdfs.Loop.Intra != &storage.Intra ||
		cdfs.Loop.InterRef != &storage.InterRef ||
		cdfs.Loop.InterMode != &storage.InterMode ||
		cdfs.Loop.MV != &storage.MV ||
		cdfs.Loop.Interp != &storage.Interp ||
		cdfs.Loop.Motion != &storage.Motion ||
		cdfs.Loop.Blend != &storage.Blend ||
		cdfs.Loop.Transform != &storage.Transform ||
		cdfs.Loop.Coeff != &storage.Coeff ||
		cdfs.Coeff.Transform != &storage.Transform ||
		cdfs.Coeff.Coeff != &storage.Coeff ||
		cdfs.TransformType != &storage.TransformType ||
		cdfs.Restoration.Switchable != &storage.RestorationSwitchable ||
		cdfs.Restoration.Wiener != &storage.RestorationWiener ||
		cdfs.Restoration.SGRProj != &storage.RestorationSGRProj ||
		cdfs.Loop.Delta.Q != &storage.DeltaQ ||
		cdfs.Loop.Delta.LF != &storage.DeltaLF {
		t.Fatalf("cdf pointer view does not alias storage: %+v", cdfs)
	}
	for i := 0; i < tile.FrameLoopFilterCount; i++ {
		if cdfs.Loop.Delta.LFMulti[i] != &storage.DeltaLFMulti[i] {
			t.Fatalf("delta lf multi[%d] does not alias storage", i)
		}
	}
	if _, err := cdfs.Loop.Intra.YModeCDF(0); err != nil {
		t.Fatalf("intra cdf not initialized: %v", err)
	}
	if _, err := cdfs.Loop.InterRef.SingleRefCDF(0, 0); err != nil {
		t.Fatalf("inter ref cdf not initialized: %v", err)
	}
	if _, err := cdfs.Loop.MV.JointCDF(); err != nil {
		t.Fatalf("mv cdf not initialized: %v", err)
	}
	if _, err := cdfs.Loop.Interp.SwitchableCDF(0); err != nil {
		t.Fatalf("interp cdf not initialized: %v", err)
	}
	if _, err := cdfs.Loop.Motion.MotionModeCDF(tile.BlockSize16x16); err != nil {
		t.Fatalf("motion cdf not initialized: %v", err)
	}
	if _, err := cdfs.Loop.Blend.CompoundTypeCDF(tile.BlockSize16x16); err != nil {
		t.Fatalf("blend cdf not initialized: %v", err)
	}
	if err := cdfs.Restoration.Wiener.Validate(); err != nil {
		t.Fatalf("restoration cdf not initialized: %v", err)
	}
	set, err := tile.ExtTXSetTypeFor(tile.TransformSize16x16, true, false)
	if err != nil {
		t.Fatal(err)
	}
	index, err := tile.ExtTXSetIndex(tile.TransformSize16x16, true, false)
	if err != nil {
		t.Fatal(err)
	}
	symbols, err := tile.ExtTXTypeCount(set)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := cdfs.TransformType.InterCDF(index, tile.TransformSize16x16, symbols); err != nil {
		t.Fatalf("transform type cdf not initialized: %v", err)
	}
}

func TestFrameWorkTileResidualCDFStorageRejectsNil(t *testing.T) {
	if err := (*FrameWorkTileResidualCDFStorage)(nil).InitDefault(0); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("nil storage err=%v want %v", err, ErrInvalidBatch)
	}
	if got := (*FrameWorkTileResidualCDFStorage)(nil).CDFs(); got.Loop.Partition != nil || got.Coeff.Coeff != nil || got.TransformType != nil {
		t.Fatalf("nil storage cdfs=%+v", got)
	}
}

func TestFrameWorkTileResidualCDFStorageAllocs(t *testing.T) {
	var storage FrameWorkTileResidualCDFStorage
	allocs := testing.AllocsPerRun(1000, func() {
		if err := storage.InitDefault(64); err != nil {
			t.Fatal(err)
		}
		_ = storage.CDFs()
	})
	if allocs != 0 {
		t.Fatalf("FrameWorkTileResidualCDFStorage allocated: %f", allocs)
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

	ctx, state, cdfs, scratch, req = testFrameWorkResidualDriver(t)
	errAfterBlock := errors.New("after block")
	req.AfterBlock = func(tile.BlockLoopVisit) error { return errAfterBlock }
	if _, err := ctx.DecodeAndReconstructJobResiduals(0, state, cdfs, &scratch, req); !errors.Is(err, errAfterBlock) {
		t.Fatalf("after block err=%v want %v", err, errAfterBlock)
	}

	ctx, state, cdfs, scratch, req = testFrameWorkResidualDriver(t)
	errBeforeSuperblock := errors.New("before superblock")
	req.Loop.BeforeSuperblock = func(tile.BlockLoopSuperblockVisit) error { return errBeforeSuperblock }
	if _, err := ctx.DecodeAndReconstructJobResiduals(0, state, cdfs, &scratch, req); !errors.Is(err, errBeforeSuperblock) {
		t.Fatalf("before superblock err=%v want %v", err, errBeforeSuperblock)
	}
}

func TestFrameWorkBatchDecodeAndReconstructJobResidualsAfterBlock(t *testing.T) {
	ctx, state, cdfs, scratch, req := testFrameWorkResidualDriver(t)
	var order []string
	req.Loop.CoeffVisitor = func(parent tile.BlockLoopVisit, block tile.BlockCoeffBlock) error {
		order = append(order, "coeff")
		if parent.Block.MICol != 0 || parent.Block.MIRow != 0 || block.Plane != 0 {
			t.Fatalf("parent=%+v block=%+v", parent.Block, block)
		}
		return nil
	}
	req.AfterBlock = func(visit tile.BlockLoopVisit) error {
		order = append(order, "after")
		if !visit.CoefficientsValid || visit.Coefficients.TotalStats().TXBs == 0 {
			t.Fatalf("after block visit missing coefficients: %+v", visit)
		}
		return nil
	}
	stats, err := ctx.DecodeAndReconstructJobResiduals(0, state, cdfs, &scratch, req)
	if err != nil {
		t.Fatal(err)
	}
	if stats.Loop.Blocks != 1 || stats.CoefficientBlocks != 1 || stats.TXBs == 0 {
		t.Fatalf("stats=%+v", stats)
	}
	if len(order) != 2 || order[0] != "coeff" || order[1] != "after" {
		t.Fatalf("order=%v want [coeff after]", order)
	}
}

func TestFrameWorkBatchDecodeAndReconstructJobResidualsReadsRestorationUnits(t *testing.T) {
	ctx, state, cdfs, scratch, req := testFrameWorkResidualDriver(t)
	ctx.FrameSize.UpscaledWidth = 64
	ctx.FrameSize.CodedWidth = 64
	ctx.FrameSize.Height = 64
	ctx.Restoration = parser.RestorationParams{
		Type:      [3]parser.RestorationType{parser.RestorationWiener},
		UnitSizeY: 64,
	}
	plan, err := ctx.RestorationFramePlan()
	if err != nil {
		t.Fatal(err)
	}
	recordBacking := make([]tile.RestorationUnitRecord, plan.UnitRecordLen())
	above := make([]uint16, plan.BoundaryBufferLen())
	below := make([]uint16, plan.BoundaryBufferLen())
	buffers, err := ctx.BindRestorationFrameBuffers(recordBacking, above, below)
	if err != nil {
		t.Fatal(err)
	}
	if err := buffers.ResetRecords(); err != nil {
		t.Fatal(err)
	}
	restorationReq := FrameWorkTileRestorationRequest{Buffers: buffers}
	if err := restorationReq.InitReferences(); err != nil {
		t.Fatal(err)
	}
	req.Restoration = &restorationReq

	stats, err := ctx.DecodeAndReconstructJobResiduals(0, state, cdfs, &scratch, req)
	if err != nil {
		t.Fatal(err)
	}
	if stats.RestorationUnits != 1 {
		t.Fatalf("restoration units=%d stats=%+v", stats.RestorationUnits, stats)
	}
	got := buffers.Records[0][0]
	if got.Index != 0 || got.Col != 0 || got.Row != 0 || got.Unit.Type != parser.RestorationNone {
		t.Fatalf("record=%+v", got)
	}
	if got.StripeCount == 0 || got.Rect.Width() == 0 || got.Rect.Height() == 0 {
		t.Fatalf("record geometry=%+v", got)
	}
}

func TestFrameWorkTileResidualControllerUsesPredictionScratch(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, MonoChrome: true, Align: 64})
	ctx := testIntraPredictionBatch(output)
	for x := 16; x < 32; x++ {
		setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, x, 15, 10)
	}
	for y := 16; y < 32; y++ {
		setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, 15, y, 50)
	}

	var predictionScratch FrameWorkPredictionScratch
	var stats FrameWorkTileResidualStats
	controller := frameWorkTileResidualLoopController{
		batch: ctx,
		index: 0,
		req: FrameWorkTileResidualRequest{
			PredictionScratch: &predictionScratch,
		},
		stats: &stats,
	}
	visit := testIntraPredictionVisit(tile.IntraModeDC)
	visit.Prefix.SkipTransform = true
	if err := controller.BeforeBlockCoefficients(visit); err != nil {
		t.Fatal(err)
	}
	if stats.Predictions != 1 || stats.SkippedBlocks != 1 || stats.CoefficientBlocks != 0 {
		t.Fatalf("stats=%+v", stats)
	}
	if got := frameWorkTestSample(output.Y, output.Layout.BytesPerSample, 16, 16); got != 30 {
		t.Fatalf("predicted sample=%d want 30", got)
	}
}

func TestFrameWorkTileResidualControllerSkipsInvalidPredictionScratch(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, MonoChrome: true, Align: 64})
	ctx := testIntraPredictionBatch(output)

	var predictionScratch FrameWorkPredictionScratch
	var stats FrameWorkTileResidualStats
	controller := frameWorkTileResidualLoopController{
		batch: ctx,
		index: 0,
		req: FrameWorkTileResidualRequest{
			PredictionScratch: &predictionScratch,
		},
		stats: &stats,
	}
	visit := testIntraPredictionVisit(tile.IntraModeDC)
	visit.Prediction = tile.BlockPredictionModeResult{}
	visit.Prefix.SkipTransform = true
	if err := controller.BeforeBlockCoefficients(visit); err != nil {
		t.Fatal(err)
	}
	if stats.Predictions != 0 || stats.SkippedBlocks != 1 || stats.CoefficientBlocks != 0 {
		t.Fatalf("stats=%+v", stats)
	}
}

func TestFrameWorkTileResidualControllerPredictsCFLOnSkipTransform(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 64})
	want := testBatchFrame(t, output.Format)
	seedFrameWorkCFLPredictionFrame(output)
	seedFrameWorkCFLPredictionFrame(want)
	ctx := testIntraPredictionBatch(output)
	wantCtx := testIntraPredictionBatch(want)

	var predictionScratch FrameWorkPredictionScratch
	var scratch FrameWorkTileResidualScratch
	var stats FrameWorkTileResidualStats
	controller := frameWorkTileResidualLoopController{
		batch:   ctx,
		index:   0,
		scratch: &scratch,
		req: FrameWorkTileResidualRequest{
			PredictionScratch: &predictionScratch,
		},
		stats: &stats,
	}
	visit := testCFLPredictionVisit()
	visit.Prefix.SkipTransform = true
	if err := controller.BeforeBlockCoefficients(visit); err != nil {
		t.Fatal(err)
	}

	var wantIntraScratch FrameWorkIntraPredictionScratch
	if err := wantCtx.PredictBlockLumaIntra(0, visit, &wantIntraScratch); err != nil {
		t.Fatal(err)
	}
	testPredictFrameWorkCFLWant(t, want, visit, FrameWorkPlaneU)
	testPredictFrameWorkCFLWant(t, want, visit, FrameWorkPlaneV)
	assertFrameWorkPlaneBlockEqual(t, output.Y, want.Y, output.Layout.BytesPerSample, 16, 16, 16, 16)
	assertFrameWorkPlaneBlockEqual(t, output.U, want.U, output.Layout.BytesPerSample, 8, 8, 8, 8)
	assertFrameWorkPlaneBlockEqual(t, output.V, want.V, output.Layout.BytesPerSample, 8, 8, 8, 8)
	if stats.Predictions != 1 || stats.SkippedBlocks != 1 || stats.CoefficientBlocks != 0 {
		t.Fatalf("stats=%+v", stats)
	}
}

func TestFrameWorkTileResidualControllerDefersCFLUntilChroma(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 64})
	want := testBatchFrame(t, output.Format)
	seedFrameWorkCFLPredictionFrame(output)
	seedFrameWorkCFLPredictionFrame(want)
	ctx := testIntraPredictionBatch(output)
	ctx.Quantization = parser.QuantizationParams{BaseQIdx: 64}

	int32Scratch, residualScratch := testFrameWorkResidualScratch(t, ctx, transform.Size{Width: 8, Height: 8}, transform.TypeDCTDCT)
	var predictionScratch FrameWorkPredictionScratch
	var scratch FrameWorkTileResidualScratch
	var stats FrameWorkTileResidualStats
	state := &tile.DecodeState{CurrentBaseQIdx: ctx.Quantization.BaseQIdx}
	controller := frameWorkTileResidualLoopController{
		batch:   ctx,
		index:   0,
		state:   state,
		scratch: &scratch,
		req: FrameWorkTileResidualRequest{
			PredictionScratch: &predictionScratch,
			Int32Scratch:      int32Scratch,
			ResidualScratch:   residualScratch,
		},
		stats: &stats,
	}
	visit := testCFLPredictionVisit()
	if err := controller.BeforeBlockCoefficients(visit); err != nil {
		t.Fatal(err)
	}
	if got := frameWorkTestSample(output.U, output.Layout.BytesPerSample, 8, 8); got != 0 {
		t.Fatalf("chroma predicted before first chroma txb: sample=%d", got)
	}
	overwriteFrameWorkCFLLuma(output)
	overwriteFrameWorkCFLLuma(want)

	chroma := tile.BlockCoeffBlock{
		Plane:     1,
		Block:     tile.TransformBlock{X4: 2, Y4: 2, Size: tile.TransformSize8x8, VisibleW4: 2, VisibleH4: 2},
		Transform: transform.TypeDCTDCT,
		Result:    tile.TXBDecodeResult{AllZero: true},
		Coeffs:    make([]int16, 64),
	}
	if err := controller.VisitBlockCoeff(visit, chroma); err != nil {
		t.Fatal(err)
	}
	testPredictFrameWorkCFLWant(t, want, visit, FrameWorkPlaneU)
	assertFrameWorkPlaneBlockEqual(t, output.U, want.U, output.Layout.BytesPerSample, 8, 8, 8, 8)
	if !controller.cflPredictionDone || stats.Residuals != 1 {
		t.Fatalf("controller cflDone=%v stats=%+v", controller.cflPredictionDone, stats)
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

func TestFrameWorkBatchDecodeAndReconstructJobResidualsDefaultAllocs(t *testing.T) {
	ctx, state, cdfs, scratch, req := testFrameWorkResidualDriver(t)
	defaultReq := FrameWorkTileResidualDefaultRequest{
		Int32Scratch:    req.Int32Scratch,
		ResidualScratch: req.ResidualScratch,
	}
	allocs := testing.AllocsPerRun(1000, func() {
		if err := state.Reset(ctx.Payload, ctx.Jobs[0], tile.DecodeOptions{BaseQIdx: ctx.Quantization.BaseQIdx}); err != nil {
			t.Fatal(err)
		}
		if _, err := ctx.DecodeAndReconstructJobResidualsDefault(0, state, cdfs, &scratch, defaultReq); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("DecodeAndReconstructJobResidualsDefault allocated: %f", allocs)
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
	var storage FrameWorkTileResidualCDFStorage
	if err := storage.InitDefault(baseQIndex); err != nil {
		t.Fatal(err)
	}
	return storage.CDFs()
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

func overwriteFrameWorkCFLLuma(output *frame.Frame) {
	for y := 16; y < 32; y++ {
		for x := 16; x < 32; x++ {
			value := uint16(9 + ((x-16)*23+(y-16)*29)%220)
			setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, x, y, value)
		}
	}
}
