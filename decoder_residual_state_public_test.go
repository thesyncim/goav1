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

func TestPublicDecoderFrameWorkBlockTransformsDispatch(t *testing.T) {
	output := publicDecoderPostFilterFrame(t, av1.FrameFormat{Width: 64, Height: 64, BitDepth: 8, MonoChrome: true, Align: 64})
	batch := publicDecoderPredictionBatch(output)
	var state av1.TileDecodeState

	intra, err := av1.ReadDecoderFrameWorkBlockTransforms(batch, &state, publicDecoderPredictionIntraVisit(av1.TileIntraModeDC))
	if err != nil {
		t.Fatal(err)
	}
	if intra.Inter || !intra.ReadIntraTX || intra.ReadInterTX || intra.Luma != av1.TransformTypeDCTDCT {
		t.Fatalf("intra transforms=%+v", intra)
	}

	interVisit := publicDecoderPredictionInterVisit(av1.MotionVector{})
	interVisit.Prediction.Valid = true
	inter, err := av1.ReadDecoderFrameWorkBlockTransforms(batch, &state, interVisit)
	if err != nil {
		t.Fatal(err)
	}
	if !inter.Inter || inter.ReadIntraTX || !inter.ReadInterTX || inter.Luma != av1.TransformTypeDCTDCT {
		t.Fatalf("inter transforms=%+v", inter)
	}

	fallback, err := av1.ReadDecoderFrameWorkBlockTransforms(batch, &state, av1.TileBlockLoopVisit{})
	if err != nil {
		t.Fatal(err)
	}
	if !fallback.Inter || fallback.ReadIntraTX || !fallback.ReadInterTX {
		t.Fatalf("fallback transforms=%+v", fallback)
	}
	if _, err := av1.ReadDecoderFrameWorkBlockTransforms(batch, nil, publicDecoderPredictionIntraVisit(av1.TileIntraModeDC)); !errors.Is(err, av1.ErrThreadingInvalidBatch) {
		t.Fatalf("nil state err=%v want %v", err, av1.ErrThreadingInvalidBatch)
	}
}

func TestPublicDecoderFrameWorkResidualMaxScratchLen(t *testing.T) {
	batch, _, _, _, _ := publicDecoderResidualDriver(t)
	got32, got16, err := av1.DecoderFrameWorkResidualMaxScratchLen(batch, batch.Quantization.BaseQIdx, 0, av1.DecoderFrameWorkPlaneY)
	if err != nil {
		t.Fatal(err)
	}
	want32, want16, err := av1.ReconstructBlockMaxScratchLen(false)
	if err != nil {
		t.Fatal(err)
	}
	if got32 != want32 || got16 != want16 {
		t.Fatalf("decoder max scratch=%d/%d want %d/%d", got32, got16, want32, want16)
	}

	losslessBatch := batch
	losslessBatch.Quantization.BaseQIdx = 0
	lossless32, lossless16, err := av1.DecoderFrameWorkResidualMaxScratchLen(losslessBatch, 0, 0, av1.DecoderFrameWorkPlaneY)
	if err != nil {
		t.Fatal(err)
	}
	wantLossless32, wantLossless16, err := av1.ReconstructBlockMaxScratchLen(true)
	if err != nil {
		t.Fatal(err)
	}
	if lossless32 != wantLossless32 || lossless16 != wantLossless16 {
		t.Fatalf("decoder lossless max scratch=%d/%d want %d/%d", lossless32, lossless16, wantLossless32, wantLossless16)
	}

	if _, _, err := av1.DecoderFrameWorkResidualMaxScratchLen(batch, batch.Quantization.BaseQIdx, 0, av1.DecoderFrameWorkPlane(99)); !errors.Is(err, av1.ErrThreadingInvalidBatch) {
		t.Fatalf("invalid plane max scratch err=%v want %v", err, av1.ErrThreadingInvalidBatch)
	}

	colorBatch := batch
	colorBatch.Sequence.ColorConfig.MonoChrome = false
	colorBatch.Segmentation.Enabled = true
	colorBatch.Segmentation.Data.Segments[3].DeltaQ = -64
	frame32, frame16, err := av1.DecoderFrameWorkResidualMaxFrameScratchLen(colorBatch, colorBatch.Quantization.BaseQIdx)
	if err != nil {
		t.Fatal(err)
	}
	nonLossless32, nonLossless16, err := av1.ReconstructBlockMaxScratchLen(false)
	if err != nil {
		t.Fatal(err)
	}
	if frame32 != nonLossless32 || frame16 != nonLossless16 {
		t.Fatalf("frame scratch=%d/%d want %d/%d", frame32, frame16, nonLossless32, nonLossless16)
	}
}

func TestPublicDecoderFrameWorkBatchResidualScratchBinding(t *testing.T) {
	batch, state, storage, scratch, _ := publicDecoderResidualBatchDriver(t)
	size, err := av1.DecoderFrameWorkBatchResidualScratchLen(batch, batch.Quantization.BaseQIdx)
	if err != nil {
		t.Fatal(err)
	}
	want32, want16, err := av1.DecoderFrameWorkResidualMaxFrameScratchLen(batch, batch.Quantization.BaseQIdx)
	if err != nil {
		t.Fatal(err)
	}
	if size.Int32Scratch != want32 || size.ResidualScratch != want16 || size.LoopContextAbove != 1 {
		t.Fatalf("batch residual scratch=%+v want int32=%d residual=%d loop=1", size, want32, want16)
	}

	arena := av1.DecoderFrameWorkBatchResidualScratch{
		Int32Scratch:     make([]int32, size.Int32Scratch+1),
		ResidualScratch:  make([]int16, size.ResidualScratch+1),
		LoopContextAbove: make([]av1.TileBlockLoopRootAboveContext, size.LoopContextAbove+1),
	}
	arena.Int32Scratch[size.Int32Scratch] = 99
	arena.ResidualScratch[size.ResidualScratch] = 77
	arena.LoopContextAbove[size.LoopContextAbove].Partition[0] = 55
	bound, err := av1.BindDecoderFrameWorkBatchResidualRequestScratch(size, arena)
	if err != nil {
		t.Fatal(err)
	}
	if len(bound.Tile.Int32Scratch) != size.Int32Scratch ||
		len(bound.Tile.ResidualScratch) != size.ResidualScratch ||
		len(bound.LoopContextAbove) != size.LoopContextAbove {
		t.Fatalf("bound scratch lens int32=%d residual=%d loop=%d want %+v",
			len(bound.Tile.Int32Scratch), len(bound.Tile.ResidualScratch), len(bound.LoopContextAbove), size)
	}
	if arena.Int32Scratch[size.Int32Scratch] != 99 ||
		arena.ResidualScratch[size.ResidualScratch] != 77 ||
		arena.LoopContextAbove[size.LoopContextAbove].Partition[0] != 55 {
		t.Fatal("scratch bind touched storage past requested lengths")
	}

	bound.Tile.TransformMode = batch.TransformRef.TransformMode
	bound.Tile.UseDefaultTransforms = true
	stats, err := av1.DecodeAndRetainDecoderFrameWorkBatchResiduals(batch, state, &storage, &scratch, bound)
	if err != nil {
		t.Fatal(err)
	}
	if stats.Loop.Blocks != 2 || stats.TXBs != 2 || stats.Residuals != 2 {
		t.Fatalf("stats=%+v", stats)
	}

	shortArena := arena
	shortArena.ResidualScratch = shortArena.ResidualScratch[:size.ResidualScratch-1]
	if _, err := av1.BindDecoderFrameWorkBatchResidualRequestScratch(size, shortArena); !errors.Is(err, av1.ErrFrameShortBuffer) {
		t.Fatalf("short residual scratch err=%v want %v", err, av1.ErrFrameShortBuffer)
	}
	negativeSize := size
	negativeSize.Int32Scratch = -1
	if _, err := av1.BindDecoderFrameWorkBatchResidualRequestScratch(negativeSize, arena); !errors.Is(err, av1.ErrFrameShortBuffer) {
		t.Fatalf("negative scratch size err=%v want %v", err, av1.ErrFrameShortBuffer)
	}
}

func TestPublicDecoderFrameWorkBatchResidualRunner(t *testing.T) {
	batch, _, _, _, _ := publicDecoderResidualBatchDriver(t)
	size, err := av1.DecoderFrameWorkBatchResidualRunnerScratchLen(batch, 2)
	if err != nil {
		t.Fatal(err)
	}
	if size.Workers != 2 || size.Int32Scratch != size.Batch.Int32Scratch*2 ||
		size.ResidualScratch != size.Batch.ResidualScratch*2 ||
		size.LoopContextAbove != size.Batch.LoopContextAbove*2 ||
		size.RestorationRequests != 2 {
		t.Fatalf("runner scratch size=%+v", size)
	}
	scratch := publicDecoderBatchResidualRunnerScratch(size)
	scratch.Int32Scratch[size.Int32Scratch] = 33
	scratch.ResidualScratch[size.ResidualScratch] = 44
	scratch.LoopContextAboveScratch[size.LoopContextAbove].Partition[0] = 66
	runner, err := av1.BindDecoderFrameWorkBatchResidualRunner(size, scratch)
	if err != nil {
		t.Fatal(err)
	}
	if scratch.PredictionScratch[1].Inter != &scratch.InterPredictionScratch[1] {
		t.Fatal("runner did not bind inter prediction scratch")
	}

	batch.Batch = av1.TileBatch{Worker: 1}
	statsErr := runner.ResetStats()
	if statsErr != nil {
		t.Fatal(statsErr)
	}
	if err := runner.Run(batch); err != nil {
		t.Fatal(err)
	}
	stats := runner.Stats[1]
	if stats.Loop.Blocks != 2 || stats.TXBs != 2 || stats.Residuals != 2 {
		t.Fatalf("runner stats=%+v", stats)
	}
	total, err := runner.TotalStats()
	if err != nil {
		t.Fatal(err)
	}
	if total.Loop.Blocks != stats.Loop.Blocks || total.TXBs != stats.TXBs {
		t.Fatalf("total stats=%+v worker stats=%+v", total, stats)
	}
	if scratch.Int32Scratch[size.Int32Scratch] != 33 ||
		scratch.ResidualScratch[size.ResidualScratch] != 44 ||
		scratch.LoopContextAboveScratch[size.LoopContextAbove].Partition[0] != 66 {
		t.Fatal("runner bind touched storage past requested lengths")
	}

	badWorker := batch
	badWorker.Batch.Worker = 2
	if err := runner.Run(badWorker); !errors.Is(err, av1.ErrThreadingInvalidBatch) {
		t.Fatalf("bad worker err=%v want %v", err, av1.ErrThreadingInvalidBatch)
	}
	shortScratch := publicDecoderBatchResidualRunnerScratch(size)
	shortScratch.Int32Scratch = shortScratch.Int32Scratch[:size.Int32Scratch-1]
	if _, err := av1.BindDecoderFrameWorkBatchResidualRunner(size, shortScratch); !errors.Is(err, av1.ErrFrameShortBuffer) {
		t.Fatalf("short runner scratch err=%v want %v", err, av1.ErrFrameShortBuffer)
	}
	if _, err := av1.DecoderFrameWorkBatchResidualRunnerScratchLen(batch, 0); !errors.Is(err, av1.ErrThreadingInvalidWorkerCount) {
		t.Fatalf("zero-worker scratch err=%v want %v", err, av1.ErrThreadingInvalidWorkerCount)
	}
	negativeSize := size
	negativeSize.RestorationRequests = -1
	if _, err := av1.BindDecoderFrameWorkBatchResidualRunner(negativeSize, scratch); !errors.Is(err, av1.ErrFrameShortBuffer) {
		t.Fatalf("negative restoration request size err=%v want %v", err, av1.ErrFrameShortBuffer)
	}
	var nilRunner *av1.DecoderFrameWorkBatchResidualRunner
	if err := nilRunner.Run(batch); !errors.Is(err, av1.ErrThreadingInvalidBatch) {
		t.Fatalf("nil runner err=%v want %v", err, av1.ErrThreadingInvalidBatch)
	}
}

func TestPublicDecoderFrameWorkBatchResidualRunnerSideData(t *testing.T) {
	batch, _, _, _, _ := publicDecoderResidualBatchDriver(t)
	size, err := av1.DecoderFrameWorkBatchResidualRunnerScratchLen(batch, 1)
	if err != nil {
		t.Fatal(err)
	}
	runner, err := av1.BindDecoderFrameWorkBatchResidualRunner(size, publicDecoderBatchResidualRunnerScratch(size))
	if err != nil {
		t.Fatal(err)
	}

	sequence := av1.SequenceHeader{ColorConfig: av1.ColorConfig{BitDepth: 8, MonoChrome: true}}
	cdef := av1.CDEFParams{Bits: 1, StrengthCount: 2}
	sideSize, err := av1.DecoderFrameWorkSideDataScratchLen(sequence, batch.FrameSize, cdef, av1.RestorationParams{})
	if err != nil {
		t.Fatal(err)
	}
	side, err := av1.BindDecoderFrameWorkSideData(sequence, batch.FrameSize, cdef, av1.RestorationParams{}, av1.DecoderFrameWorkSideDataScratch{
		CDEFIndexMap:             make([]uint8, sideSize.CDEFIndexMap),
		CDEFReadMap:              make([]bool, sideSize.CDEFReadMap),
		LoopFilterMap:            make([]av1.DecoderFrameWorkLoopFilterBlockRecord, sideSize.LoopFilterMap),
		RestorationRecords:       make([]av1.TileRestorationUnitRecord, sideSize.RestorationRecords),
		RestorationBoundaryAbove: make([]uint16, sideSize.RestorationBoundaryAbove),
		RestorationBoundaryBelow: make([]uint16, sideSize.RestorationBoundaryBelow),
	})
	if err != nil {
		t.Fatal(err)
	}
	side.LoopFilterMap.Records[0].Valid = true
	if err := av1.SetDecoderFrameWorkBatchResidualRunnerSideData(&runner, side); err != nil {
		t.Fatal(err)
	}
	if side.LoopFilterMap.Records[0].Valid {
		t.Fatal("runner side-data attach did not reset loop-filter map")
	}
	if err := runner.Run(batch); err != nil {
		t.Fatal(err)
	}
	seenLoopRecord := false
	for _, record := range side.LoopFilterMap.Records {
		if record.Valid {
			seenLoopRecord = true
			break
		}
	}
	if !seenLoopRecord {
		t.Fatal("runner did not write through attached loop-filter side data")
	}

	activeRestoration := av1.RestorationParams{
		Type:      [3]av1.RestorationType{av1.RestorationWiener},
		UnitSizeY: 64,
	}
	activeSideSize, err := av1.DecoderFrameWorkSideDataScratchLen(sequence, batch.FrameSize, cdef, activeRestoration)
	if err != nil {
		t.Fatal(err)
	}
	activeSide, err := av1.BindDecoderFrameWorkSideData(sequence, batch.FrameSize, cdef, activeRestoration, av1.DecoderFrameWorkSideDataScratch{
		CDEFIndexMap:             make([]uint8, activeSideSize.CDEFIndexMap),
		CDEFReadMap:              make([]bool, activeSideSize.CDEFReadMap),
		LoopFilterMap:            make([]av1.DecoderFrameWorkLoopFilterBlockRecord, activeSideSize.LoopFilterMap),
		RestorationRecords:       make([]av1.TileRestorationUnitRecord, activeSideSize.RestorationRecords),
		RestorationBoundaryAbove: make([]uint16, activeSideSize.RestorationBoundaryAbove),
		RestorationBoundaryBelow: make([]uint16, activeSideSize.RestorationBoundaryBelow),
	})
	if err != nil {
		t.Fatal(err)
	}
	noRestorationScratch := publicDecoderBatchResidualRunnerScratch(size)
	noRestorationScratch.RestorationRequests = nil
	noRestorationRunner, err := av1.BindDecoderFrameWorkBatchResidualRunner(size, noRestorationScratch)
	if err != nil {
		t.Fatal(err)
	}
	if err := av1.SetDecoderFrameWorkBatchResidualRunnerSideData(&noRestorationRunner, activeSide); !errors.Is(err, av1.ErrFrameShortBuffer) {
		t.Fatalf("missing restoration requests err=%v want %v", err, av1.ErrFrameShortBuffer)
	}
	var nilRunner *av1.DecoderFrameWorkBatchResidualRunner
	if err := av1.SetDecoderFrameWorkBatchResidualRunnerSideData(nilRunner, side); !errors.Is(err, av1.ErrThreadingInvalidBatch) {
		t.Fatalf("nil side-data runner err=%v want %v", err, av1.ErrThreadingInvalidBatch)
	}
}

func TestPublicDecoderFrameWorkResidualEventRunner(t *testing.T) {
	workerPool, err := av1.NewTileWorkerPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	sequence := av1.SequenceHeader{
		EnableCDEF: true,
		ColorConfig: av1.ColorConfig{
			BitDepth:   8,
			MonoChrome: true,
		},
	}
	event := publicDecoderResidualRunnerFrameEvent()
	var scratchSpans [2]av1.TileSpan
	var scratchJobs [2]av1.TileJob
	var scratchBatches [1]av1.TileBatch
	eventScratch, err := av1.DecoderFrameWorkResidualEventScratchLen(sequence, event, 1, scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		t.Fatal(err)
	}
	runnerSize := eventScratch.Runner
	plan := eventScratch.Plan
	if plan != (av1.DecoderTileWorkPlan{SpanCount: 2, JobCount: 2, BatchCount: 1}) {
		t.Fatalf("event residual runner plan=%+v", plan)
	}
	runner, err := av1.BindDecoderFrameWorkBatchResidualRunner(runnerSize, publicDecoderBatchResidualRunnerScratch(runnerSize))
	if err != nil {
		t.Fatal(err)
	}
	side, err := av1.BindDecoderFrameWorkResidualEventSideData(sequence, event, publicDecoderFrameWorkSideDataScratch(eventScratch.SideData))
	if err != nil {
		t.Fatal(err)
	}
	side.LoopFilterMap.Records[0].Valid = true

	pool := publicDecoderPostFilterFramePool(t, av1.FrameFormat{
		Width:      int(event.FrameSize.CodedWidth),
		Height:     int(event.FrameSize.Height),
		BitDepth:   8,
		MonoChrome: true,
		Align:      64,
	}, 1)
	var refs av1.DecoderSurfaceReferences
	var state av1.DecoderFrameWorkState
	var referenceSurfaces [av1.InterRefsPerFrame]int
	var referenceFrames [av1.InterRefsPerFrame]*av1.Frame
	var spans [2]av1.TileSpan
	var jobs [2]av1.TileJob
	var batches [1]av1.TileBatch
	var releases [av1.RefFrames]int
	var postSawSideData bool
	eventRunner := av1.DecoderFrameWorkResidualEventRunner{
		State:             &state,
		Refs:              &refs,
		FramePool:         &pool,
		Align:             64,
		ReferenceSurfaces: referenceSurfaces[:],
		ReferenceFrames:   referenceFrames[:],
		Workers:           1,
		Spans:             spans[:],
		Jobs:              jobs[:],
		Batches:           batches[:],
		Releases:          releases[:],
		WorkerPool:        workerPool,
		BatchRunner:       &runner,
	}

	result, err := eventRunner.Run(sequence, event, &side, func(ctx av1.DecoderFrameWorkPostFilterContext) error {
		postSawSideData = ctx.CDEFIndexMap != nil && ctx.LoopFilterMap != nil && ctx.RestorationFrameBuffers != nil
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Step.Kind != av1.DecoderFrameWorkStepBegin ||
		result.Run != (av1.DecoderFrameWorkStepResult{ExecutedTileWork: true, CompletedFrame: true}) ||
		result.Output == nil ||
		state.Active() ||
		!postSawSideData {
		t.Fatalf("result=%+v output=%p active=%v postSide=%v", result, result.Output, state.Active(), postSawSideData)
	}
	seenLoopRecord := false
	for _, record := range side.LoopFilterMap.Records {
		if record.Valid {
			seenLoopRecord = true
			break
		}
	}
	if !seenLoopRecord {
		t.Fatal("event runner did not write loop-filter side data")
	}
	slot, ok := refs.ReferenceSlot(0)
	if !ok || slot < 0 {
		t.Fatalf("decoded frame was not published to reference slot 0: slot=%d ok=%v", slot, ok)
	}
	if _, err := av1.RunDecoderFrameWorkEventWithResidualRunner(av1.DecoderFrameWorkResidualEventRequest{Runner: nil}); !errors.Is(err, av1.ErrThreadingInvalidBatch) {
		t.Fatalf("nil residual event runner err=%v want %v", err, av1.ErrThreadingInvalidBatch)
	}
}

func TestPublicDecoderFrameWorkResidualEventRunnerScratchLen(t *testing.T) {
	sequence := av1.SequenceHeader{
		EnableCDEF: true,
		ColorConfig: av1.ColorConfig{
			BitDepth:   8,
			MonoChrome: true,
		},
	}
	event := publicDecoderResidualRunnerFrameEvent()
	var spans [2]av1.TileSpan
	var jobs [2]av1.TileJob
	var batches [1]av1.TileBatch
	size, plan, err := av1.DecoderFrameWorkResidualEventRunnerScratchLen(sequence, event, 1, spans[:], jobs[:], batches[:])
	if err != nil {
		t.Fatal(err)
	}
	if plan != (av1.DecoderTileWorkPlan{SpanCount: 2, JobCount: 2, BatchCount: 1}) ||
		size.Workers != 1 ||
		size.RestorationRequests != 1 ||
		size.Batch.LoopContextAbove == 0 {
		t.Fatalf("size=%+v plan=%+v", size, plan)
	}
	if spans[0] != (av1.TileSpan{Tile: 0, Row: 0, Col: 0, Offset: 1, Size: 256}) ||
		spans[1] != (av1.TileSpan{Tile: 1, Row: 0, Col: 1, Offset: 257, Size: 256}) ||
		jobs[0].Tile != 0 ||
		jobs[1].Tile != 1 ||
		batches[0].Count != 2 {
		t.Fatalf("spans=%+v jobs=%+v batches=%+v", spans, jobs, batches)
	}
	var shortSpans [1]av1.TileSpan
	if _, _, err := av1.DecoderFrameWorkResidualEventRunnerScratchLen(sequence, event, 1, shortSpans[:], jobs[:], batches[:]); !errors.Is(err, av1.ErrInvalidTileGroup) {
		t.Fatalf("short spans err=%v want %v", err, av1.ErrInvalidTileGroup)
	}
	if _, _, err := av1.DecoderFrameWorkResidualEventRunnerScratchLen(sequence, av1.DecoderEvent{Kind: av1.DecoderEventSequenceHeader}, 1, spans[:], jobs[:], batches[:]); !errors.Is(err, av1.ErrDecoderInvalidTileWork) {
		t.Fatalf("invalid event err=%v want %v", err, av1.ErrDecoderInvalidTileWork)
	}

	allocs := testing.AllocsPerRun(1000, func() {
		_, _, err = av1.DecoderFrameWorkResidualEventRunnerScratchLen(sequence, event, 1, spans[:], jobs[:], batches[:])
	})
	if err != nil {
		t.Fatal(err)
	}
	if allocs != 0 {
		t.Fatalf("DecoderFrameWorkResidualEventRunnerScratchLen allocated: %f", allocs)
	}

	combinedSize, err := av1.DecoderFrameWorkResidualEventScratchLen(sequence, event, 1, spans[:], jobs[:], batches[:])
	if err != nil {
		t.Fatal(err)
	}
	if combinedSize.Runner != size ||
		combinedSize.Plan != plan ||
		combinedSize.SideData.CDEFIndexMap == 0 ||
		combinedSize.SideData.LoopFilterMap == 0 {
		t.Fatalf("combined residual event scratch size=%+v runner=%+v plan=%+v", combinedSize, size, plan)
	}
	side, err := av1.BindDecoderFrameWorkResidualEventSideData(sequence, event, publicDecoderFrameWorkSideDataScratch(combinedSize.SideData))
	if err != nil {
		t.Fatal(err)
	}
	if len(side.CDEFIndexMap.Index) != combinedSize.SideData.CDEFIndexMap ||
		len(side.LoopFilterMap.Records) != combinedSize.SideData.LoopFilterMap {
		t.Fatalf("side=%+v size=%+v", side, combinedSize.SideData)
	}
	shortSideScratch := publicDecoderFrameWorkSideDataScratch(combinedSize.SideData)
	shortSideScratch.CDEFIndexMap = shortSideScratch.CDEFIndexMap[:combinedSize.SideData.CDEFIndexMap-1]
	if _, err := av1.BindDecoderFrameWorkResidualEventSideData(sequence, event, shortSideScratch); !errors.Is(err, av1.ErrFrameShortBuffer) {
		t.Fatalf("short event side-data err=%v want %v", err, av1.ErrFrameShortBuffer)
	}

	allocs = testing.AllocsPerRun(1000, func() {
		_, err = av1.DecoderFrameWorkResidualEventScratchLen(sequence, event, 1, spans[:], jobs[:], batches[:])
	})
	if err != nil {
		t.Fatal(err)
	}
	if allocs != 0 {
		t.Fatalf("DecoderFrameWorkResidualEventScratchLen allocated: %f", allocs)
	}

	sideScratch := publicDecoderFrameWorkSideDataScratch(combinedSize.SideData)
	allocs = testing.AllocsPerRun(1000, func() {
		_, err = av1.BindDecoderFrameWorkResidualEventSideData(sequence, event, sideScratch)
	})
	if err != nil {
		t.Fatal(err)
	}
	if allocs != 0 {
		t.Fatalf("BindDecoderFrameWorkResidualEventSideData allocated: %f", allocs)
	}
}

func TestPublicDecoderFrameWorkResidualEventRunnerAllocs(t *testing.T) {
	workerPool, err := av1.NewTileWorkerPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	sequence := av1.SequenceHeader{
		EnableCDEF: true,
		ColorConfig: av1.ColorConfig{
			BitDepth:   8,
			MonoChrome: true,
		},
	}
	event := publicDecoderResidualRunnerFrameEvent()
	var scratchSpans [2]av1.TileSpan
	var scratchJobs [2]av1.TileJob
	var scratchBatches [1]av1.TileBatch
	eventScratch, err := av1.DecoderFrameWorkResidualEventScratchLen(sequence, event, 1, scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		t.Fatal(err)
	}
	runnerSize := eventScratch.Runner
	runner, err := av1.BindDecoderFrameWorkBatchResidualRunner(runnerSize, publicDecoderBatchResidualRunnerScratch(runnerSize))
	if err != nil {
		t.Fatal(err)
	}
	side, err := av1.BindDecoderFrameWorkResidualEventSideData(sequence, event, publicDecoderFrameWorkSideDataScratch(eventScratch.SideData))
	if err != nil {
		t.Fatal(err)
	}

	pool := publicDecoderPostFilterFramePool(t, av1.FrameFormat{
		Width:      int(event.FrameSize.CodedWidth),
		Height:     int(event.FrameSize.Height),
		BitDepth:   8,
		MonoChrome: true,
		Align:      64,
	}, 1)
	var refs av1.DecoderSurfaceReferences
	var referenceSurfaces [av1.InterRefsPerFrame]int
	var referenceFrames [av1.InterRefsPerFrame]*av1.Frame
	var spans [2]av1.TileSpan
	var jobs [2]av1.TileJob
	var batches [1]av1.TileBatch
	var releases [av1.RefFrames]int
	var state av1.DecoderFrameWorkState
	post := func(av1.DecoderFrameWorkPostFilterContext) error { return nil }
	eventRunner := av1.DecoderFrameWorkResidualEventRunner{
		State:             &state,
		Refs:              &refs,
		FramePool:         &pool,
		Align:             64,
		ReferenceSurfaces: referenceSurfaces[:],
		ReferenceFrames:   referenceFrames[:],
		Workers:           1,
		Spans:             spans[:],
		Jobs:              jobs[:],
		Batches:           batches[:],
		Releases:          releases[:],
		WorkerPool:        workerPool,
		BatchRunner:       &runner,
	}

	allocs := testing.AllocsPerRun(1000, func() {
		pool.Reset()
		refs.Reset()
		state.Reset()
		result, runErr := eventRunner.Run(sequence, event, &side, post)
		if runErr != nil {
			err = runErr
			return
		}
		if result.Output == nil || result.Run != (av1.DecoderFrameWorkStepResult{ExecutedTileWork: true, CompletedFrame: true}) {
			err = av1.ErrDecoderInvalidFrameWorkStep
			return
		}
		if _, ok := refs.ReferenceSlot(0); !ok {
			err = av1.ErrDecoderInvalidSurfaceReference
		}
	})
	if err != nil {
		t.Fatal(err)
	}
	if allocs != 0 {
		t.Fatalf("DecoderFrameWorkResidualEventRunner.Run allocated: %f", allocs)
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
		return av1.ReadDecoderFrameWorkBlockTransforms(batch, state, visit)
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

func TestPublicDecodeAndRetainDecoderFrameWorkJobResiduals(t *testing.T) {
	batch, state, _, scratch, req := publicDecoderResidualDriver(t)
	var storage av1.DecoderFrameWorkTileResidualCDFStorage
	if err := av1.InitDecoderFrameWorkTileResidualCDFStorageDefault(&storage, batch.Quantization.BaseQIdx); err != nil {
		t.Fatal(err)
	}
	var retained av1.DecoderFrameWorkTileResidualCDFStorage
	retainedValid := false
	batch.RetainedTileResidualCDFs = &retained
	batch.RetainedTileResidualCDFsValid = &retainedValid
	batch.Jobs[0].UpdatesFrameContext = true

	stats, err := av1.DecodeAndRetainDecoderFrameWorkJobResiduals(batch, 0, state, &storage, &scratch, req)
	if err != nil {
		t.Fatal(err)
	}
	if stats.TXBs != 1 || stats.TXBs != stats.NonZero+stats.AllZero {
		t.Fatalf("stats=%+v", stats)
	}
	if !retainedValid {
		t.Fatal("update job did not retain residual CDF storage")
	}
	if !publicCDFValuesEqual(retained.DeltaQ.Values(), storage.DeltaQ.Values()) {
		t.Fatalf("retained delta q=%v want %v", retained.DeltaQ.Values(), storage.DeltaQ.Values())
	}

	if err := av1.InitDecoderFrameWorkTileResidualCDFStorageDefault(&storage, batch.Quantization.BaseQIdx); err != nil {
		t.Fatal(err)
	}
	retainedValid = false
	batch.Jobs[0].UpdatesFrameContext = false
	if _, err := av1.DecodeAndRetainDecoderFrameWorkJobResiduals(batch, 0, state, &storage, &scratch, req); err != nil {
		t.Fatal(err)
	}
	if retainedValid {
		t.Fatal("non-update job retained residual CDF storage")
	}
}

func TestPublicDecodeAndRetainDecoderFrameWorkJobResidualsDefaultTransforms(t *testing.T) {
	batch, state, _, scratch, req := publicDecoderResidualDriver(t)
	req.Transforms = nil
	req.UseDefaultTransforms = true
	var storage av1.DecoderFrameWorkTileResidualCDFStorage
	if err := av1.InitDecoderFrameWorkTileResidualCDFStorageDefault(&storage, batch.Quantization.BaseQIdx); err != nil {
		t.Fatal(err)
	}

	stats, err := av1.DecodeAndRetainDecoderFrameWorkJobResiduals(batch, 0, state, &storage, &scratch, req)
	if err != nil {
		t.Fatal(err)
	}
	if stats.Loop.Blocks != 1 || stats.TXBs != 1 || stats.Residuals != 1 {
		t.Fatalf("stats=%+v", stats)
	}
}

func TestPublicDecodeAndRetainDecoderFrameWorkBatchResiduals(t *testing.T) {
	batch, state, storage, scratch, req := publicDecoderResidualBatchDriver(t)
	var retained av1.DecoderFrameWorkTileResidualCDFStorage
	retainedValid := false
	batch.RetainedTileResidualCDFs = &retained
	batch.RetainedTileResidualCDFsValid = &retainedValid
	visits := 0
	req.Tile.Loop.BeforeCoefficients = func(visit av1.TileBlockLoopVisit) error {
		visits++
		if visits == 2 && visit.Block.MICol != 16 {
			t.Fatalf("second job block=%+v", visit.Block)
		}
		return nil
	}

	stats, err := av1.DecodeAndRetainDecoderFrameWorkBatchResiduals(batch, state, &storage, &scratch, req)
	if err != nil {
		t.Fatal(err)
	}
	if stats.Loop.Blocks != 2 || stats.Loop.PartitionReads != 2 || visits != 2 {
		t.Fatalf("stats=%+v visits=%d", stats, visits)
	}
	if stats.CoefficientBlocks != 2 || stats.Residuals != 2 ||
		stats.TXBs != 2 || stats.TXBs != stats.NonZero+stats.AllZero ||
		stats.Loop.CoefficientTXBs != stats.TXBs {
		t.Fatalf("residual stats=%+v", stats)
	}
	if !retainedValid {
		t.Fatal("update job did not retain residual CDF storage")
	}
	if !publicCDFValuesEqual(retained.DeltaQ.Values(), storage.DeltaQ.Values()) {
		t.Fatalf("retained delta q=%v want %v", retained.DeltaQ.Values(), storage.DeltaQ.Values())
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
	if _, err := av1.DecodeAndRetainDecoderFrameWorkJobResiduals(batch, 0, state, nil, &scratch, req); !errors.Is(err, av1.ErrThreadingInvalidBatch) {
		t.Fatalf("nil retain storage err=%v want %v", err, av1.ErrThreadingInvalidBatch)
	}
	batchReq := av1.DecoderFrameWorkBatchResidualRequest{Tile: req, LoopContextAbove: []av1.TileBlockLoopRootAboveContext{}}
	if _, err := av1.DecodeAndRetainDecoderFrameWorkBatchResiduals(batch, state, &av1.DecoderFrameWorkTileResidualCDFStorage{}, &scratch, batchReq); !errors.Is(err, av1.ErrFrameShortBuffer) {
		t.Fatalf("short batch carrier err=%v want %v", err, av1.ErrFrameShortBuffer)
	}
	if _, err := av1.DecodeAndRetainDecoderFrameWorkBatchResiduals(batch, nil, &av1.DecoderFrameWorkTileResidualCDFStorage{}, &scratch, batchReq); !errors.Is(err, av1.ErrThreadingInvalidBatch) {
		t.Fatalf("nil batch state err=%v want %v", err, av1.ErrThreadingInvalidBatch)
	}
	if _, _, err := av1.DecoderFrameWorkResidualScratchLen(batch, batch.Quantization.BaseQIdx, 0, av1.DecoderFrameWorkPlane(99), av1.TransformSize{Width: 64, Height: 64}, av1.TransformTypeDCTDCT); !errors.Is(err, av1.ErrThreadingInvalidBatch) {
		t.Fatalf("bad scratch plane err=%v want %v", err, av1.ErrThreadingInvalidBatch)
	}
	if n, err := av1.DecoderFrameWorkBatchResidualLoopContextAboveLen(av1.DecoderFrameWorkBatch{}); err != nil || n != 0 {
		t.Fatalf("empty batch loop context len=%d err=%v want 0,nil", n, err)
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

func TestPublicDecodeAndRetainDecoderFrameWorkJobResidualsAllocs(t *testing.T) {
	batch, state, _, scratch, req := publicDecoderResidualDriver(t)
	var storage av1.DecoderFrameWorkTileResidualCDFStorage
	allocs := testing.AllocsPerRun(1000, func() {
		if err := av1.InitDecoderFrameWorkTileResidualCDFStorageDefault(&storage, batch.Quantization.BaseQIdx); err != nil {
			t.Fatal(err)
		}
		if _, err := av1.DecodeAndRetainDecoderFrameWorkJobResiduals(batch, 0, state, &storage, &scratch, req); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("allocs=%v want 0", allocs)
	}
}

func TestPublicDecodeAndRetainDecoderFrameWorkBatchResidualsAllocs(t *testing.T) {
	batch, state, storage, scratch, _ := publicDecoderResidualBatchDriver(t)
	scratchSize, err := av1.DecoderFrameWorkBatchResidualScratchLen(batch, batch.Quantization.BaseQIdx)
	if err != nil {
		t.Fatal(err)
	}
	arena := av1.DecoderFrameWorkBatchResidualScratch{
		Int32Scratch:     make([]int32, scratchSize.Int32Scratch),
		ResidualScratch:  make([]int16, scratchSize.ResidualScratch),
		LoopContextAbove: make([]av1.TileBlockLoopRootAboveContext, scratchSize.LoopContextAbove),
	}
	var req av1.DecoderFrameWorkBatchResidualRequest
	allocs := testing.AllocsPerRun(1000, func() {
		req, err = av1.BindDecoderFrameWorkBatchResidualRequestScratch(scratchSize, arena)
		if err != nil {
			t.Fatal(err)
		}
		req.Tile.TransformMode = batch.TransformRef.TransformMode
		req.Tile.UseDefaultTransforms = true
		if err := av1.InitDecoderFrameWorkTileResidualCDFStorageDefault(&storage, batch.Quantization.BaseQIdx); err != nil {
			t.Fatal(err)
		}
		if _, err := av1.DecodeAndRetainDecoderFrameWorkBatchResiduals(batch, state, &storage, &scratch, req); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("allocs=%v want 0", allocs)
	}
}

func TestPublicDecoderFrameWorkBatchResidualRunnerAllocs(t *testing.T) {
	batch, _, _, _, _ := publicDecoderResidualBatchDriver(t)
	size, err := av1.DecoderFrameWorkBatchResidualRunnerScratchLen(batch, 1)
	if err != nil {
		t.Fatal(err)
	}
	runner, err := av1.BindDecoderFrameWorkBatchResidualRunner(size, publicDecoderBatchResidualRunnerScratch(size))
	if err != nil {
		t.Fatal(err)
	}
	allocs := testing.AllocsPerRun(1000, func() {
		if err := runner.Run(batch); err != nil {
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
	int32Len, int16Len, err := av1.DecoderFrameWorkResidualMaxScratchLen(batch, batch.Quantization.BaseQIdx, 0, av1.DecoderFrameWorkPlaneY)
	if err != nil {
		t.Fatal(err)
	}
	var scratch av1.DecoderFrameWorkTileResidualScratch
	req := av1.DecoderFrameWorkTileResidualRequest{
		Loop:          loopReq,
		TransformMode: batch.TransformRef.TransformMode,
		Transforms: func(visit av1.TileBlockLoopVisit) (av1.DecoderFrameWorkBlockTransforms, error) {
			return av1.ReadDecoderFrameWorkBlockTransforms(batch, state, visit)
		},
		Int32Scratch:    make([]int32, int32Len),
		ResidualScratch: make([]int16, int16Len),
	}
	return batch, state, av1.DecoderFrameWorkTileResidualCDFsFromStorage(&storage), scratch, req
}

func publicDecoderResidualBatchDriver(t *testing.T) (av1.DecoderFrameWorkBatch, *av1.TileDecodeState, av1.DecoderFrameWorkTileResidualCDFStorage, av1.DecoderFrameWorkTileResidualScratch, av1.DecoderFrameWorkBatchResidualRequest) {
	t.Helper()
	output := publicDecoderPostFilterFrame(t, av1.FrameFormat{Width: 128, Height: 64, BitDepth: 8, MonoChrome: true, Align: 64})
	publicFillDecoderPostFilterPlane(output.Y)
	batch := av1.DecoderFrameWorkBatch{
		Output:  output,
		Payload: make([]byte, 512),
		FrameWorkFrameContext: av1.DecoderFrameWorkFrameContext{
			Sequence: av1.DecoderFrameWorkSequenceContextFromHeader(av1.SequenceHeader{
				ColorConfig: av1.ColorConfig{BitDepth: 8, MonoChrome: true},
			}),
			FrameSize:    av1.FrameSize{CodedWidth: 128, UpscaledWidth: 128, Height: 64, SuperResDenominator: 8},
			Quantization: av1.QuantizationParams{BaseQIdx: 64},
			TransformRef: av1.TransformReferenceParams{TransformMode: av1.TransformModeLargest},
		},
		Jobs: []av1.TileJob{
			{SBX: 0, SBY: 0, SBCols: 1, SBRows: 1, Offset: 0, Size: 256},
			{SBX: 1, SBY: 0, SBCols: 1, SBRows: 1, Offset: 256, Size: 256, UpdatesFrameContext: true},
		},
	}
	var retained av1.DecoderFrameWorkTileResidualCDFStorage
	retainedValid := false
	batch.RetainedTileResidualCDFs = &retained
	batch.RetainedTileResidualCDFsValid = &retainedValid
	var storage av1.DecoderFrameWorkTileResidualCDFStorage
	if err := av1.InitDecoderFrameWorkTileResidualCDFStorageDefault(&storage, batch.Quantization.BaseQIdx); err != nil {
		t.Fatal(err)
	}
	int32Len, int16Len, err := av1.DecoderFrameWorkResidualMaxScratchLen(batch, batch.Quantization.BaseQIdx, 0, av1.DecoderFrameWorkPlaneY)
	if err != nil {
		t.Fatal(err)
	}
	loopContextLen, err := av1.DecoderFrameWorkBatchResidualLoopContextAboveLen(batch)
	if err != nil {
		t.Fatal(err)
	}
	if loopContextLen != 1 {
		t.Fatalf("batch loop context len=%d want 1", loopContextLen)
	}
	state := &av1.TileDecodeState{}
	var scratch av1.DecoderFrameWorkTileResidualScratch
	req := av1.DecoderFrameWorkBatchResidualRequest{
		Tile: av1.DecoderFrameWorkTileResidualRequest{
			TransformMode: batch.TransformRef.TransformMode,
			Transforms: func(visit av1.TileBlockLoopVisit) (av1.DecoderFrameWorkBlockTransforms, error) {
				return av1.ReadDecoderFrameWorkBlockTransforms(batch, state, visit)
			},
			Int32Scratch:    make([]int32, int32Len),
			ResidualScratch: make([]int16, int16Len),
		},
		LoopContextAbove: make([]av1.TileBlockLoopRootAboveContext, loopContextLen),
	}
	return batch, state, storage, scratch, req
}

func publicDecoderBatchResidualRunnerScratch(size av1.DecoderFrameWorkBatchResidualRunnerScratchSize) av1.DecoderFrameWorkBatchResidualRunnerScratch {
	return av1.DecoderFrameWorkBatchResidualRunnerScratch{
		States:                  make([]av1.TileDecodeState, size.Workers+1),
		Storages:                make([]av1.DecoderFrameWorkTileResidualCDFStorage, size.Workers+1),
		TileScratch:             make([]av1.DecoderFrameWorkTileResidualScratch, size.Workers+1),
		RestorationRequests:     make([]av1.DecoderFrameWorkTileRestorationRequest, size.RestorationRequests+1),
		PredictionScratch:       make([]av1.DecoderFrameWorkPredictionScratch, size.Workers+1),
		InterPredictionScratch:  make([]av1.DecoderFrameWorkInterPredictionScratch, size.Workers+1),
		Stats:                   make([]av1.DecoderFrameWorkTileResidualStats, size.Workers+1),
		Int32Scratch:            make([]int32, size.Int32Scratch+1),
		ResidualScratch:         make([]int16, size.ResidualScratch+1),
		LoopContextAboveScratch: make([]av1.TileBlockLoopRootAboveContext, size.LoopContextAbove+1),
	}
}

func publicDecoderFrameWorkSideDataScratch(size av1.DecoderFrameWorkSideDataScratchSize) av1.DecoderFrameWorkSideDataScratch {
	return av1.DecoderFrameWorkSideDataScratch{
		CDEFIndexMap:             make([]uint8, size.CDEFIndexMap+1),
		CDEFReadMap:              make([]bool, size.CDEFReadMap+1),
		LoopFilterMap:            make([]av1.DecoderFrameWorkLoopFilterBlockRecord, size.LoopFilterMap+1),
		RestorationRecords:       make([]av1.TileRestorationUnitRecord, size.RestorationRecords+1),
		RestorationBoundaryAbove: make([]uint16, size.RestorationBoundaryAbove+1),
		RestorationBoundaryBelow: make([]uint16, size.RestorationBoundaryBelow+1),
	}
}

func publicDecoderResidualRunnerFrameEvent() av1.DecoderEvent {
	payload := make([]byte, 513)
	payload[0] = 0xff
	return av1.DecoderEvent{
		Kind: av1.DecoderEventFrame,
		Unit: av1.OBUUnit{Payload: payload},
		FrameHeader: av1.FrameHeaderPrefix{
			FrameType: av1.FrameTypeKey,
		},
		FrameSize: av1.FrameSize{
			CodedWidth:          128,
			UpscaledWidth:       128,
			Height:              64,
			SuperResDenominator: 8,
			RefreshFrameFlags:   0xff,
		},
		TileInfo: av1.TileInfo{
			TileSizeBytes: 1,
			SBCols:        2,
			SBRows:        1,
			Cols:          2,
			Rows:          1,
			ColStartSB:    [av1.MaxTileCols + 1]uint16{0, 1, 2},
			RowStartSB:    [av1.MaxTileRows + 1]uint16{0, 1},
		},
		TileGroup: av1.TileGroup{
			StartTile: 0,
			EndTile:   1,
			TileCount: 2,
			Final:     true,
		},
		CDEF:         av1.CDEFParams{Bits: 0, StrengthCount: 1},
		Quantization: av1.QuantizationParams{BaseQIdx: 64},
		TransformRef: av1.TransformReferenceParams{TransformMode: av1.TransformModeLargest},
	}
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
