package goav1_test

import (
	"errors"
	"testing"

	av1 "github.com/thesyncim/goav1"
)

func TestPublicDecoderTileResidualCDFStorageLifecycle(t *testing.T) {
	var initial av1.DecoderFrameWorkTileResidualCDFStorage
	if err := av1.InitDecoderFrameWorkTileResidualCDFStorageDefault(&initial, 64); err != nil {
		t.Fatal(err)
	}
	if err := initial.DeltaQ.Update(2); err != nil {
		t.Fatal(err)
	}
	initialCDFs := av1.DecoderFrameWorkTileResidualCDFsFromStorage(&initial)
	if initialCDFs.Loop.Partition == nil || initialCDFs.Coeff.Coeff == nil || initialCDFs.TransformType == nil {
		t.Fatalf("initial CDF view=%+v", initialCDFs)
	}

	var retained av1.DecoderFrameWorkTileResidualCDFStorage
	retainedValid := false
	batch := av1.DecoderFrameWorkBatch{
		Payload: []byte{0x00, 0xff},
		FrameWorkFrameContext: av1.DecoderFrameWorkFrameContext{
			Quantization: av1.QuantizationParams{BaseQIdx: 91},
		},
		InitialTileResidualCDFs:       &initial,
		RetainedTileResidualCDFs:      &retained,
		RetainedTileResidualCDFsValid: &retainedValid,
		Jobs: []av1.TileJob{
			{Tile: 0, Offset: 0, Size: 1},
			{Tile: 1, Offset: 1, Size: 1, UpdatesFrameContext: true},
		},
	}

	var storage av1.DecoderFrameWorkTileResidualCDFStorage
	if err := av1.InitDecoderFrameWorkTileResidualCDFStorage(batch, &storage); err != nil {
		t.Fatal(err)
	}
	if !publicCDFValuesEqual(storage.DeltaQ.Values(), initial.DeltaQ.Values()) {
		t.Fatalf("storage delta q=%v want %v", storage.DeltaQ.Values(), initial.DeltaQ.Values())
	}
	if err := storage.DeltaQ.Update(1); err != nil {
		t.Fatal(err)
	}

	var state av1.TileDecodeState
	if err := av1.InitDecoderFrameWorkJobDecodeState(batch, 0, &state); err != nil {
		t.Fatal(err)
	}
	if state.RetainFrameContext || state.CurrentBaseQIdx != 91 {
		t.Fatalf("state=%+v", state)
	}
	if err := av1.RetainDecoderFrameWorkTileResidualCDFStorage(batch, 0, &state, &storage); err != nil {
		t.Fatal(err)
	}
	if retainedValid {
		t.Fatal("non-update tile retained frame context")
	}

	if err := av1.InitDecoderFrameWorkJobDecodeState(batch, 1, &state); err != nil {
		t.Fatal(err)
	}
	if !state.RetainFrameContext || state.CurrentBaseQIdx != 91 {
		t.Fatalf("update state=%+v", state)
	}
	if err := av1.RetainDecoderFrameWorkTileResidualCDFStorage(batch, 1, &state, &storage); err != nil {
		t.Fatal(err)
	}
	if !retainedValid {
		t.Fatal("update tile did not retain frame context")
	}
	if !publicCDFValuesEqual(retained.DeltaQ.Values(), storage.DeltaQ.Values()) {
		t.Fatalf("retained delta q=%v want %v", retained.DeltaQ.Values(), storage.DeltaQ.Values())
	}

	batch.InitialTileResidualCDFs = nil
	if err := av1.InitDecoderFrameWorkTileResidualCDFStorage(batch, &storage); err != nil {
		t.Fatal(err)
	}
	if publicCDFValuesEqual(storage.DeltaQ.Values(), initial.DeltaQ.Values()) {
		t.Fatal("default fallback unexpectedly reused initial frame context")
	}
}

func TestPublicDecoderTileResidualStateRejectsInvalidInputs(t *testing.T) {
	batch := av1.DecoderFrameWorkBatch{
		Payload: []byte{0xaa},
		Jobs:    []av1.TileJob{{Offset: 0, Size: 2}},
	}
	var state av1.TileDecodeState
	if err := av1.InitDecoderFrameWorkTileResidualCDFStorage(batch, nil); !errors.Is(err, av1.ErrThreadingInvalidBatch) {
		t.Fatalf("nil cdf storage err=%v want %v", err, av1.ErrThreadingInvalidBatch)
	}
	if err := av1.InitDecoderFrameWorkJobDecodeState(batch, -1, &state); !errors.Is(err, av1.ErrThreadingInvalidBatch) {
		t.Fatalf("negative index err=%v want %v", err, av1.ErrThreadingInvalidBatch)
	}
	if err := av1.InitDecoderFrameWorkJobDecodeState(batch, 0, nil); !errors.Is(err, av1.ErrThreadingInvalidBatch) {
		t.Fatalf("nil state err=%v want %v", err, av1.ErrThreadingInvalidBatch)
	}
	if err := av1.InitDecoderFrameWorkJobDecodeState(batch, 0, &state); !errors.Is(err, av1.ErrTileInvalidPlan) {
		t.Fatalf("invalid range err=%v want %v", err, av1.ErrTileInvalidPlan)
	}
	var storage av1.DecoderFrameWorkTileResidualCDFStorage
	if err := av1.RetainDecoderFrameWorkTileResidualCDFStorage(batch, 0, &state, nil); !errors.Is(err, av1.ErrThreadingInvalidBatch) {
		t.Fatalf("nil retain storage err=%v want %v", err, av1.ErrThreadingInvalidBatch)
	}
	if err := av1.InitDecoderFrameWorkTileResidualCDFStorageDefault(nil, 0); !errors.Is(err, av1.ErrThreadingInvalidBatch) {
		t.Fatalf("nil default cdf storage err=%v want %v", err, av1.ErrThreadingInvalidBatch)
	}
	if got := av1.DecoderFrameWorkTileResidualCDFsFromStorage(nil); got.Loop.Partition != nil || got.Coeff.Coeff != nil || got.TransformType != nil {
		t.Fatalf("nil CDF view=%+v", got)
	}
	if err := av1.RetainDecoderFrameWorkTileResidualCDFStorage(batch, 0, nil, &storage); !errors.Is(err, av1.ErrThreadingInvalidBatch) {
		t.Fatalf("nil retain state err=%v want %v", err, av1.ErrThreadingInvalidBatch)
	}
}

func TestPublicDecoderBlockLoopRequestAndContextBinding(t *testing.T) {
	batch := publicDecoderBlockLoopBatch()
	rootColumns, err := av1.DecoderFrameWorkJobBlockLoopContextRootColumns(batch, 0)
	if err != nil {
		t.Fatal(err)
	}
	if rootColumns != 2 {
		t.Fatalf("root columns=%d want 2", rootColumns)
	}

	above := make([]av1.TileBlockLoopRootAboveContext, rootColumns+1)
	above[0].Partition[0] = 11
	above[rootColumns].Partition[0] = 7
	carrier, err := av1.BindTileBlockLoopContextCarrier(rootColumns, above)
	if err != nil {
		t.Fatal(err)
	}
	if len(carrier.Above) != rootColumns {
		t.Fatalf("carrier above=%d want %d", len(carrier.Above), rootColumns)
	}
	if above[0].Partition[0] != 0 {
		t.Fatalf("carrier bind did not clear active above storage: %+v", above[0].Partition)
	}
	if above[rootColumns].Partition[0] != 7 {
		t.Fatal("carrier bind touched caller storage past active root columns")
	}

	segMap := make([]uint8, 96*96)
	req, err := av1.DecoderFrameWorkJobBlockLoopRequest(batch, 0, segMap, nil, 96, &carrier)
	if err != nil {
		t.Fatal(err)
	}
	if req.ContextCarrier != &carrier {
		t.Fatalf("context carrier=%p want %p", req.ContextCarrier, &carrier)
	}
	if req.Walk.Root != av1.TileBlockLevel128x128 ||
		req.Walk.MIColStart != 32 || req.Walk.MIRowStart != 32 ||
		req.Walk.MIColEnd != 76 || req.Walk.MIRowEnd != 66 {
		t.Fatalf("walk=%+v", req.Walk)
	}
	if !req.SkipMode.Enabled || req.CDEF.Bits != 2 ||
		!req.Delta.DeltaQPresent || req.SBSizeMIB != 32 || !req.Monochrome ||
		len(req.CurrentSegmentMap) != len(segMap) || req.SegmentMapStride != 96 ||
		req.InterpolationFilter != av1.InterpolationSwitchable || !req.EnableDualFilter ||
		!req.EnableFilterIntra || !req.EnableOrderHint || req.OrderHintBits != 5 ||
		req.CurrentOrderHint != 9 || req.ReferenceOrderHints[4] != 5 ||
		req.RefFrameSide[av1.TileReferenceFrameLast2] != -1 ||
		req.RefFrameSide[av1.TileReferenceFrameLast3] != 1 ||
		!req.UseRefFrameMVS || !req.TemporalMVSampleUnavailable {
		t.Fatalf("request=%+v", req)
	}

	if _, err := av1.BindTileBlockLoopContextCarrier(-1, nil); !errors.Is(err, av1.ErrTileInvalidDecodeState) {
		t.Fatalf("negative carrier err=%v want %v", err, av1.ErrTileInvalidDecodeState)
	}
	if _, err := av1.BindTileBlockLoopContextCarrier(rootColumns, above[:rootColumns-1]); !errors.Is(err, av1.ErrFrameShortBuffer) {
		t.Fatalf("short carrier err=%v want %v", err, av1.ErrFrameShortBuffer)
	}
	if _, err := av1.DecoderFrameWorkJobBlockLoopRequest(batch, 1, nil, nil, 0, nil); !errors.Is(err, av1.ErrThreadingInvalidBatch) {
		t.Fatalf("bad job err=%v want %v", err, av1.ErrThreadingInvalidBatch)
	}
}

func TestPublicDecodeAndReconstructDecoderFrameWorkJobResiduals(t *testing.T) {
	batch, state, cdfs, scratch, req := publicDecoderResidualDriver(t)
	predictions := 0
	transforms := 0
	req.Predict = func(visit av1.TileBlockLoopVisit) error {
		predictions++
		if visit.Block.MICol != 0 || visit.Block.MIRow != 0 {
			t.Fatalf("visit=%+v", visit.Block)
		}
		return nil
	}
	req.Transforms = func(visit av1.TileBlockLoopVisit) (av1.DecoderFrameWorkBlockTransforms, error) {
		transforms++
		return av1.ReadDecoderFrameWorkInterBlockTransforms(batch, state, visit)
	}

	stats, err := av1.DecodeAndReconstructDecoderFrameWorkJobResiduals(batch, 0, state, cdfs, &scratch, req)
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

func TestPublicDecodeAndReconstructDecoderFrameWorkJobResidualsMarksSideMaps(t *testing.T) {
	batch, state, cdfs, scratch, req := publicDecoderResidualDriver(t)
	batch.CDEF = av1.CDEFParams{Bits: 0, StrengthCount: 1}
	sequence := av1.SequenceHeader{
		Use128x128Superblock: batch.Sequence.SBSizeMIB == 32,
		ColorConfig:          batch.Sequence.ColorConfig,
	}
	_, _, cdefLength, err := av1.DecoderFrameWorkCDEFIndexMapShape(sequence, batch.FrameSize)
	if err != nil {
		t.Fatal(err)
	}
	cdefMap, err := av1.BindDecoderFrameWorkCDEFIndexMap(sequence, batch.FrameSize, batch.CDEF, make([]uint8, cdefLength), make([]bool, cdefLength))
	if err != nil {
		t.Fatal(err)
	}
	_, _, loopLength, err := av1.DecoderFrameWorkLoopFilterMapShape(sequence, batch.FrameSize)
	if err != nil {
		t.Fatal(err)
	}
	loopMap, err := av1.BindDecoderFrameWorkLoopFilterMap(sequence, batch.FrameSize, make([]av1.DecoderFrameWorkLoopFilterBlockRecord, loopLength))
	if err != nil {
		t.Fatal(err)
	}
	batch.CDEFIndexMap = &cdefMap
	batch.LoopFilterMap = &loopMap
	state.DeltaLFFromBase = -3
	state.DeltaLF = [av1.TileFrameLoopFilterCount]int8{4, 3, 2, 1}

	stats, err := av1.DecodeAndReconstructDecoderFrameWorkJobResiduals(batch, 0, state, cdfs, &scratch, req)
	if err != nil {
		t.Fatal(err)
	}
	if stats.CoefficientBlocks != 1 || stats.Loop.CoefficientBlocks != 1 {
		t.Fatalf("stats=%+v", stats)
	}
	if !cdefMap.Read[0] || cdefMap.Index[0] != 0 {
		t.Fatalf("cdef map read=%v index=%d want true,0", cdefMap.Read[0], cdefMap.Index[0])
	}
	record := loopMap.Records[0]
	if !record.Valid || record.Block.MICol != 0 || record.Block.MIRow != 0 ||
		record.DeltaLFFromBase != -3 || record.DeltaLF != state.DeltaLF {
		t.Fatalf("loop-filter record=%+v state delta=%v", record, state.DeltaLF)
	}
}

func TestPublicDecodeAndReconstructDecoderFrameWorkJobResidualsRejectsInvalidInputs(t *testing.T) {
	batch, state, cdfs, scratch, req := publicDecoderResidualDriver(t)
	if _, err := av1.DecodeAndReconstructDecoderFrameWorkJobResiduals(batch, 0, nil, cdfs, &scratch, req); !errors.Is(err, av1.ErrThreadingInvalidBatch) {
		t.Fatalf("nil state err=%v want %v", err, av1.ErrThreadingInvalidBatch)
	}
	if _, err := av1.DecodeAndReconstructDecoderFrameWorkJobResiduals(batch, 0, state, cdfs, nil, req); !errors.Is(err, av1.ErrThreadingInvalidBatch) {
		t.Fatalf("nil scratch err=%v want %v", err, av1.ErrThreadingInvalidBatch)
	}
	noTransforms := req
	noTransforms.Transforms = nil
	if _, err := av1.DecodeAndReconstructDecoderFrameWorkJobResiduals(batch, 0, state, cdfs, &scratch, noTransforms); !errors.Is(err, av1.ErrThreadingInvalidBatch) {
		t.Fatalf("nil transforms err=%v want %v", err, av1.ErrThreadingInvalidBatch)
	}
	shortCarrier := req
	scratch.LoopContext.Above = []av1.TileBlockLoopRootAboveContext{}
	if _, err := av1.DecodeAndReconstructDecoderFrameWorkJobResiduals(batch, 0, state, cdfs, &scratch, shortCarrier); !errors.Is(err, av1.ErrTileInvalidDecodeState) {
		t.Fatalf("short scratch carrier err=%v want %v", err, av1.ErrTileInvalidDecodeState)
	}
	if _, _, err := av1.DecoderFrameWorkResidualScratchLen(batch, batch.Quantization.BaseQIdx, 0, av1.DecoderFrameWorkPlane(99), av1.TransformSize{Width: 64, Height: 64}, av1.TransformTypeDCTDCT); !errors.Is(err, av1.ErrThreadingInvalidBatch) {
		t.Fatalf("bad scratch plane err=%v want %v", err, av1.ErrThreadingInvalidBatch)
	}
	if err := av1.InitDecoderFrameWorkTileRestorationRequestReferences(nil); !errors.Is(err, av1.ErrThreadingInvalidBatch) {
		t.Fatalf("nil restoration refs err=%v want %v", err, av1.ErrThreadingInvalidBatch)
	}
}

func TestPublicDecoderTileResidualStateAllocs(t *testing.T) {
	var initial av1.DecoderFrameWorkTileResidualCDFStorage
	if err := av1.InitDecoderFrameWorkTileResidualCDFStorageDefault(&initial, 64); err != nil {
		t.Fatal(err)
	}
	var retained av1.DecoderFrameWorkTileResidualCDFStorage
	retainedValid := false
	batch := av1.DecoderFrameWorkBatch{
		Payload: []byte{0x00, 0xff},
		FrameWorkFrameContext: av1.DecoderFrameWorkFrameContext{
			Quantization: av1.QuantizationParams{BaseQIdx: 73},
		},
		InitialTileResidualCDFs:       &initial,
		RetainedTileResidualCDFs:      &retained,
		RetainedTileResidualCDFsValid: &retainedValid,
		Jobs: []av1.TileJob{
			{Tile: 0, Offset: 0, Size: 1},
			{Tile: 1, Offset: 1, Size: 1, UpdatesFrameContext: true},
		},
	}
	var storage av1.DecoderFrameWorkTileResidualCDFStorage
	var state av1.TileDecodeState
	blockLoopBatch := publicDecoderBlockLoopBatch()
	rootColumns, err := av1.DecoderFrameWorkJobBlockLoopContextRootColumns(blockLoopBatch, 0)
	if err != nil {
		t.Fatal(err)
	}
	above := make([]av1.TileBlockLoopRootAboveContext, rootColumns)
	segMap := make([]uint8, 96*96)

	allocs := testing.AllocsPerRun(1000, func() {
		err = av1.InitDecoderFrameWorkTileResidualCDFStorage(batch, &storage)
		if err != nil {
			return
		}
		_ = av1.DecoderFrameWorkTileResidualCDFsFromStorage(&storage)
		err = av1.InitDecoderFrameWorkJobDecodeState(batch, 1, &state)
		if err != nil {
			return
		}
		err = av1.RetainDecoderFrameWorkTileResidualCDFStorage(batch, 1, &state, &storage)
		if err != nil {
			return
		}
		var carrier av1.TileBlockLoopContextCarrier
		carrier, err = av1.BindTileBlockLoopContextCarrier(rootColumns, above)
		if err != nil {
			return
		}
		_, err = av1.DecoderFrameWorkJobBlockLoopRequest(blockLoopBatch, 0, segMap, nil, 96, &carrier)
	})
	if err != nil {
		t.Fatal(err)
	}
	if allocs != 0 {
		t.Fatalf("allocs=%v want 0", allocs)
	}
}

func TestPublicDecodeAndReconstructDecoderFrameWorkJobResidualsAllocs(t *testing.T) {
	batch, state, cdfs, scratch, req := publicDecoderResidualDriver(t)
	allocs := testing.AllocsPerRun(1000, func() {
		if err := av1.InitDecoderFrameWorkJobDecodeState(batch, 0, state); err != nil {
			t.Fatal(err)
		}
		if _, err := av1.DecodeAndReconstructDecoderFrameWorkJobResiduals(batch, 0, state, cdfs, &scratch, req); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("allocs=%v want 0", allocs)
	}
}

func publicDecoderBlockLoopBatch() av1.DecoderFrameWorkBatch {
	return av1.DecoderFrameWorkBatch{
		FrameWorkFrameContext: av1.DecoderFrameWorkFrameContext{
			Sequence: av1.DecoderFrameWorkSequenceContextFromHeader(av1.SequenceHeader{
				Use128x128Superblock: true,
				EnableDualFilter:     true,
				EnableFilterIntra:    true,
				EnableOrderHint:      true,
				OrderHintBits:        5,
				ColorConfig:          av1.ColorConfig{BitDepth: 8, MonoChrome: true},
			}),
			FrameHeader:         av1.FrameHeaderPrefix{OrderHint: 9},
			FrameSize:           av1.FrameSize{CodedWidth: 300, UpscaledWidth: 300, Height: 260, SuperResDenominator: 8},
			TileInfo:            av1.TileInfo{InterpolationFilter: av1.InterpolationSwitchable, UseRefFrameMVS: true},
			ReferenceOrderHints: [av1.InterRefsPerFrame]uint32{1, 9, 10, 4, 5, 6, 7},
			SkipMode:            av1.SkipModeParams{Allowed: true, Enabled: true},
			CDEF:                av1.CDEFParams{Bits: 2, StrengthCount: 4},
			Delta:               av1.DeltaParams{DeltaQPresent: true, DeltaQResLog2: 1},
		},
		Jobs: []av1.TileJob{{Tile: 3, Row: 1, Col: 1, SBX: 1, SBY: 1, SBCols: 2, SBRows: 2}},
	}
}

func publicDecoderResidualDriver(t *testing.T) (av1.DecoderFrameWorkBatch, *av1.TileDecodeState, av1.DecoderFrameWorkTileResidualCDFs, av1.DecoderFrameWorkTileResidualScratch, av1.DecoderFrameWorkTileResidualRequest) {
	t.Helper()
	output := publicDecoderPostFilterFrame(t, av1.FrameFormat{Width: 64, Height: 64, BitDepth: 8, MonoChrome: true, Align: 64})
	publicFillDecoderPostFilterPlane(output.Y)
	batch := av1.DecoderFrameWorkBatch{
		Output:  output,
		Payload: make([]byte, 256),
		FrameWorkFrameContext: av1.DecoderFrameWorkFrameContext{
			Sequence: av1.DecoderFrameWorkSequenceContextFromHeader(av1.SequenceHeader{
				ColorConfig: av1.ColorConfig{BitDepth: 8, MonoChrome: true},
			}),
			FrameSize:    av1.FrameSize{CodedWidth: 64, UpscaledWidth: 64, Height: 64, SuperResDenominator: 8},
			Quantization: av1.QuantizationParams{BaseQIdx: 64},
			TransformRef: av1.TransformReferenceParams{TransformMode: av1.TransformModeLargest},
		},
		Jobs: []av1.TileJob{{SBCols: 1, SBRows: 1, Offset: 0, Size: 256}},
	}
	state := &av1.TileDecodeState{}
	if err := av1.InitDecoderFrameWorkJobDecodeState(batch, 0, state); err != nil {
		t.Fatal(err)
	}
	var storage av1.DecoderFrameWorkTileResidualCDFStorage
	if err := av1.InitDecoderFrameWorkTileResidualCDFStorageDefault(&storage, batch.Quantization.BaseQIdx); err != nil {
		t.Fatal(err)
	}
	loopReq, err := av1.DecoderFrameWorkJobBlockLoopRequest(batch, 0, nil, nil, 0, nil)
	if err != nil {
		t.Fatal(err)
	}
	int32Len, int16Len, err := av1.DecoderFrameWorkResidualScratchLen(batch, batch.Quantization.BaseQIdx, 0, av1.DecoderFrameWorkPlaneY, av1.TransformSize{Width: 64, Height: 64}, av1.TransformTypeDCTDCT)
	if err != nil {
		t.Fatal(err)
	}
	var scratch av1.DecoderFrameWorkTileResidualScratch
	req := av1.DecoderFrameWorkTileResidualRequest{
		Loop:          loopReq,
		TransformMode: batch.TransformRef.TransformMode,
		Transforms: func(visit av1.TileBlockLoopVisit) (av1.DecoderFrameWorkBlockTransforms, error) {
			return av1.ReadDecoderFrameWorkInterBlockTransforms(batch, state, visit)
		},
		Int32Scratch:    make([]int32, int32Len),
		ResidualScratch: make([]int16, int16Len),
	}
	return batch, state, av1.DecoderFrameWorkTileResidualCDFsFromStorage(&storage), scratch, req
}

func publicCDFValuesEqual(a []uint16, b []uint16) bool {
	if len(a) != len(b) {
		return false
	}
	for i := 0; i < len(a); i++ {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
