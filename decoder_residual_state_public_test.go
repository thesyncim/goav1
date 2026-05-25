package goav1_test

import (
	"errors"
	"testing"

	av1 "github.com/thesyncim/goav1"
)

type publicDecoderResidualEventPostRunner struct {
	calls  int
	side   bool
	output *av1.Frame
	err    error
	sample byte
	active bool
	seq    av1.SequenceHeader
}

func (r *publicDecoderResidualEventPostRunner) Apply(ctx av1.DecoderFrameWorkPostFilterContext) error {
	r.calls++
	r.side = ctx.CDEFIndexMap != nil && ctx.LoopFilterMap != nil && ctx.RestorationFrameBuffers != nil
	r.output = ctx.Output
	r.active = ctx.ExecutedTileWork && ctx.Step.Kind == av1.DecoderFrameWorkStepBegin
	r.seq = ctx.Event.SequenceHeader
	if ctx.Output != nil {
		ctx.Output.Y.Pix[0] = r.sample
	}
	return r.err
}

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
	if !runner.UseDefaultPrediction {
		t.Fatal("runner did not enable default prediction scratch")
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
	brokenRunner := runner
	brokenRunner.Stats[0] = av1.DecoderFrameWorkTileResidualStats{TXBs: 88, Residuals: 77}
	brokenRunner.Int32Scratch = nil
	brokenBatch := batch
	brokenBatch.Batch.Worker = 0
	if err := brokenRunner.Run(brokenBatch); !errors.Is(err, av1.ErrFrameShortBuffer) {
		t.Fatalf("broken runner err=%v want %v", err, av1.ErrFrameShortBuffer)
	}
	if brokenRunner.Stats[0] != (av1.DecoderFrameWorkTileResidualStats{}) {
		t.Fatalf("failed worker stats were not cleared: %+v", brokenRunner.Stats[0])
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

func TestPublicDecoderFrameWorkResidualScratchSizeMax(t *testing.T) {
	batchA := av1.DecoderFrameWorkBatchResidualScratchSize{
		Int32Scratch:     1,
		ResidualScratch:  7,
		LoopContextAbove: 3,
	}
	batchB := av1.DecoderFrameWorkBatchResidualScratchSize{
		Int32Scratch:     5,
		ResidualScratch:  2,
		LoopContextAbove: 4,
	}
	if got, want := batchA.Max(batchB), (av1.DecoderFrameWorkBatchResidualScratchSize{
		Int32Scratch:     5,
		ResidualScratch:  7,
		LoopContextAbove: 4,
	}); got != want {
		t.Fatalf("batch residual scratch max=%+v want %+v", got, want)
	}

	runnerA := av1.DecoderFrameWorkBatchResidualRunnerScratchSize{
		Workers:             1,
		Batch:               batchA,
		RestorationRequests: 1,
		Int32Scratch:        1,
		ResidualScratch:     7,
		LoopContextAbove:    3,
	}
	runnerB := av1.DecoderFrameWorkBatchResidualRunnerScratchSize{
		Workers:             3,
		Batch:               batchB,
		RestorationRequests: 2,
		Int32Scratch:        15,
		ResidualScratch:     6,
		LoopContextAbove:    12,
	}
	if got, want := runnerA.Max(runnerB), (av1.DecoderFrameWorkBatchResidualRunnerScratchSize{
		Workers:             3,
		Batch:               batchA.Max(batchB),
		RestorationRequests: 2,
		Int32Scratch:        15,
		ResidualScratch:     7,
		LoopContextAbove:    12,
	}); got != want {
		t.Fatalf("runner residual scratch max=%+v want %+v", got, want)
	}

	eventA := av1.DecoderFrameWorkResidualEventScratchSize{
		Runner:   runnerA,
		SideData: av1.DecoderFrameWorkSideDataScratchSize{CDEFIndexMap: 2, LoopFilterMap: 8},
		Plan:     av1.DecoderTileWorkPlan{SpanCount: 2, JobCount: 2, BatchCount: 1},
		Outputs:  1,
	}
	eventB := av1.DecoderFrameWorkResidualEventScratchSize{
		Runner:   runnerB,
		SideData: av1.DecoderFrameWorkSideDataScratchSize{CDEFReadMap: 4, LoopFilterMap: 3},
		Plan:     av1.DecoderTileWorkPlan{SpanCount: 1, JobCount: 5, BatchCount: 2},
		Outputs:  3,
	}
	got := eventA.Max(eventB)
	if got.Runner != runnerA.Max(runnerB) ||
		got.SideData != (av1.DecoderFrameWorkSideDataScratchSize{CDEFIndexMap: 2, CDEFReadMap: 4, LoopFilterMap: 8}) ||
		got.Plan != (av1.DecoderTileWorkPlan{SpanCount: 2, JobCount: 5, BatchCount: 2}) ||
		got.Outputs != 3 {
		t.Fatalf("event residual scratch max=%+v", got)
	}

	streamA := av1.DecoderFrameWorkResidualStreamScratchSize{
		Events:    2,
		RTPBuffer: 16,
		RTPSpans:  1,
		Event:     eventA,
	}
	streamB := av1.DecoderFrameWorkResidualStreamScratchSize{
		Events:    4,
		RTPBuffer: 8,
		RTPSpans:  3,
		Event:     eventB,
	}
	streamGot := streamA.Max(streamB)
	if streamGot.Events != 4 ||
		streamGot.RTPBuffer != 16 ||
		streamGot.RTPSpans != 3 ||
		streamGot.Event != eventA.Max(eventB) {
		t.Fatalf("stream residual scratch max=%+v", streamGot)
	}

	sequenceA := av1.SequenceHeader{ColorConfig: av1.ColorConfig{BitDepth: 8}}
	sequenceB := av1.SequenceHeader{ColorConfig: av1.ColorConfig{BitDepth: 10}}
	planA := av1.DecoderFrameWorkResidualStreamPlan{
		Size: streamA,
		Bind: av1.DecoderFrameWorkResidualEventBindPlan{
			Sequence:   sequenceA,
			Event:      av1.DecoderEvent{Kind: av1.DecoderEventExistingFrame},
			EventIndex: 0,
		},
	}
	planB := av1.DecoderFrameWorkResidualStreamPlan{
		Size: streamB,
		Bind: av1.DecoderFrameWorkResidualEventBindPlan{
			Sequence:   sequenceB,
			Event:      av1.DecoderEvent{Kind: av1.DecoderEventTileGroup},
			EventIndex: 1,
		},
	}
	planGot := planA.Max(planB)
	if !planGot.HasEvent() ||
		planGot.Size != streamGot ||
		planGot.Bind.Event.Kind != av1.DecoderEventTileGroup ||
		planGot.Bind.Sequence.ColorConfig.BitDepth != 10 ||
		planGot.Bind.EventIndex != 1 {
		t.Fatalf("stream residual plan max=%+v", planGot)
	}
}

func TestPublicDecoderFrameWorkResidualStreamScratchCheckAndState(t *testing.T) {
	size := av1.DecoderFrameWorkResidualStreamScratchSize{
		Events:    2,
		RTPBuffer: 16,
		RTPSpans:  2,
		Event: av1.DecoderFrameWorkResidualEventScratchSize{
			Runner: av1.DecoderFrameWorkBatchResidualRunnerScratchSize{
				Workers:         1,
				Int32Scratch:    4,
				ResidualScratch: 4,
			},
			SideData: av1.DecoderFrameWorkSideDataScratchSize{
				CDEFIndexMap:  4,
				LoopFilterMap: 4,
			},
			Plan:    av1.DecoderTileWorkPlan{SpanCount: 1, JobCount: 1, BatchCount: 1},
			Outputs: 2,
		},
	}
	scratch := publicDecoderResidualStreamScratch(size)
	if err := scratch.Check(size); err != nil {
		t.Fatal(err)
	}
	shortEvents := scratch
	shortEvents.Events = shortEvents.Events[:size.Events-1]
	if err := shortEvents.Check(size); !errors.Is(err, av1.ErrDecoderEventBufferTooSmall) {
		t.Fatalf("short events err=%v want %v", err, av1.ErrDecoderEventBufferTooSmall)
	}
	shortRTP := scratch
	shortRTP.RTPBuffer = shortRTP.RTPBuffer[:size.RTPBuffer-1]
	if err := shortRTP.Check(size); !errors.Is(err, av1.ErrRTPShortBuffer) {
		t.Fatalf("short RTP err=%v want %v", err, av1.ErrRTPShortBuffer)
	}
	shortOutputs := scratch
	shortOutputs.Outputs = shortOutputs.Outputs[:size.Event.Outputs-1]
	if err := shortOutputs.Check(size); !errors.Is(err, av1.ErrFrameShortBuffer) {
		t.Fatalf("short outputs err=%v want %v", err, av1.ErrFrameShortBuffer)
	}
	shortEventScratch := scratch
	shortEventScratch.Event.Spans = shortEventScratch.Event.Spans[:0]
	if err := shortEventScratch.Check(size); !errors.Is(err, av1.ErrFrameShortBuffer) {
		t.Fatalf("short event scratch err=%v want %v", err, av1.ErrFrameShortBuffer)
	}

	var nilRunner *av1.DecoderFrameWorkResidualStreamRunner
	if state := nilRunner.State(); state != (av1.DecoderFrameWorkResidualStreamRunnerState{}) {
		t.Fatalf("nil runner state=%+v", state)
	}
	if _, ok := nilRunner.SequenceHeader(); ok {
		t.Fatal("nil runner reported sequence header")
	}

	var stream av1.DecoderStream
	var events [4]av1.DecoderEvent
	if count, err := stream.PushLowOverhead(publicDecoderResidualLowOverheadStream(), events[:]); err != nil {
		t.Fatal(err)
	} else if count != 3 || !stream.HasSequenceHeader() {
		t.Fatalf("low-overhead count=%d hasSequence=%v", count, stream.HasSequenceHeader())
	}
	runner := av1.DecoderFrameWorkResidualStreamRunner{
		Stream: &stream,
		EventRunner: av1.DecoderFrameWorkResidualEventRunner{
			Outputs: scratch.Outputs[:size.Event.Outputs],
		},
		Events:    scratch.Events[:size.Events],
		RTPBuffer: scratch.RTPBuffer[:size.RTPBuffer],
		RTPSpans:  scratch.RTPSpans[:size.RTPSpans],
	}
	state := runner.State()
	if !state.Bound ||
		!state.HasSequenceHeader ||
		state.InRTPFragment ||
		state.RTPUsed != 0 ||
		state.EventCapacity != size.Events ||
		state.RTPBufferCapacity != size.RTPBuffer ||
		state.RTPSpanCapacity != size.RTPSpans ||
		state.OutputCapacity != size.Event.Outputs {
		t.Fatalf("runner state=%+v", state)
	}
	sequence, ok := runner.SequenceHeader()
	if !ok || sequence.ColorConfig.BitDepth != 8 {
		t.Fatalf("sequence ok=%v header=%+v", ok, sequence)
	}

	frameHeader := publicDecoderResidualRTPElement(av1.OBUFrameHeader, publicDecoderResidualFrameHeaderPayload())
	var packet [8]byte
	n, _, more, err := av1.PutRTPFragment(packet[:], frameHeader, 0, 2, false)
	if err != nil {
		t.Fatal(err)
	}
	if !more {
		t.Fatal("expected retained RTP fragment")
	}
	used, count, err := stream.PushRTPPayload(runner.RTPBuffer, 0, runner.RTPSpans, runner.Events, packet[:n])
	if err != nil {
		t.Fatal(err)
	}
	runner.RTPUsed = used
	state = runner.State()
	if count != 0 || !state.InRTPFragment || state.RTPUsed == 0 {
		t.Fatalf("fragment count=%d state=%+v", count, state)
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
	}, 2)
	var refs av1.DecoderSurfaceReferences
	var state av1.DecoderFrameWorkState
	var referenceSurfaces [av1.InterRefsPerFrame]int
	var referenceFrames [av1.InterRefsPerFrame]*av1.Frame
	var spans [2]av1.TileSpan
	var jobs [2]av1.TileJob
	var batches [1]av1.TileBatch
	var releases [av1.RefFrames]int
	var stats av1.DecoderFrameWorkTileResidualStats
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
		Stats:             &stats,
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
	totalStats, err := runner.TotalStats()
	if err != nil {
		t.Fatal(err)
	}
	if stats != totalStats || stats.TXBs == 0 || stats.Residuals == 0 {
		t.Fatalf("stats=%+v total=%+v", stats, totalStats)
	}
	slot, ok := refs.ReferenceSlot(0)
	if !ok || slot < 0 {
		t.Fatalf("decoded frame was not published to reference slot 0: slot=%d ok=%v", slot, ok)
	}
	if _, err := av1.RunDecoderFrameWorkEventWithResidualRunner(av1.DecoderFrameWorkResidualEventRequest{
		Event:  av1.DecoderEvent{Kind: av1.DecoderEventFrame},
		Runner: nil,
	}); !errors.Is(err, av1.ErrThreadingInvalidBatch) {
		t.Fatalf("nil residual event runner err=%v want %v", err, av1.ErrThreadingInvalidBatch)
	}
}

func TestPublicDecoderFrameWorkResidualEventPostFilterRunner(t *testing.T) {
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
	runner, err := av1.BindDecoderFrameWorkBatchResidualRunner(eventScratch.Runner, publicDecoderBatchResidualRunnerScratch(eventScratch.Runner))
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
	}, 2)
	var refs av1.DecoderSurfaceReferences
	var state av1.DecoderFrameWorkState
	var referenceSurfaces [av1.InterRefsPerFrame]int
	var referenceFrames [av1.InterRefsPerFrame]*av1.Frame
	var spans [2]av1.TileSpan
	var jobs [2]av1.TileJob
	var batches [1]av1.TileBatch
	var releases [av1.RefFrames]int
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
	postRunner := publicDecoderResidualEventPostRunner{sample: 0x5a}

	result, err := eventRunner.RunWithPostFilterRunner(sequence, event, &side, &postRunner)
	if err != nil {
		t.Fatal(err)
	}
	if result.Run != (av1.DecoderFrameWorkStepResult{ExecutedTileWork: true, CompletedFrame: true}) ||
		result.Output == nil ||
		state.Active() ||
		postRunner.calls != 1 ||
		!postRunner.side ||
		!postRunner.active ||
		postRunner.seq.ColorConfig.BitDepth != sequence.ColorConfig.BitDepth ||
		!postRunner.seq.EnableCDEF ||
		postRunner.output != result.Output {
		t.Fatalf("result=%+v active=%v post=%+v", result, state.Active(), postRunner)
	}
	slot, ok := refs.ReferenceSlot(0)
	if !ok || slot < 0 {
		t.Fatalf("decoded frame was not published to reference slot 0: slot=%d ok=%v", slot, ok)
	}
	published, err := pool.Frame(slot)
	if err != nil {
		t.Fatal(err)
	}
	if published.Y.Pix[0] != postRunner.sample {
		t.Fatalf("published sample=%d want %d", published.Y.Pix[0], postRunner.sample)
	}
	if _, err := av1.RunDecoderFrameWorkEventWithResidualRunner(av1.DecoderFrameWorkResidualEventRequest{
		Runner:     &runner,
		Post:       func(av1.DecoderFrameWorkPostFilterContext) error { return nil },
		PostRunner: &postRunner,
	}); !errors.Is(err, av1.ErrDecoderInvalidFrameWorkState) {
		t.Fatalf("mixed post callback/runner err=%v want %v", err, av1.ErrDecoderInvalidFrameWorkState)
	}
}

func TestPublicDecoderFrameWorkResidualEventRunnerResetsStats(t *testing.T) {
	workerPool, err := av1.NewTileWorkerPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	sequence := av1.SequenceHeader{
		ColorConfig: av1.ColorConfig{
			BitDepth:   8,
			MonoChrome: true,
		},
	}
	event := publicDecoderResidualRunnerFrameEvent()
	var scratchSpans [2]av1.TileSpan
	var scratchJobs [2]av1.TileJob
	var scratchBatches [2]av1.TileBatch
	runnerSize, _, err := av1.DecoderFrameWorkResidualEventRunnerScratchLen(sequence, event, 2, scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		t.Fatal(err)
	}
	runner, err := av1.BindDecoderFrameWorkBatchResidualRunner(runnerSize, publicDecoderBatchResidualRunnerScratch(runnerSize))
	if err != nil {
		t.Fatal(err)
	}
	runner.Stats[1] = av1.DecoderFrameWorkTileResidualStats{TXBs: 77, Residuals: 55}

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
	result, err := eventRunner.Run(sequence, event, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if result.Run != (av1.DecoderFrameWorkStepResult{ExecutedTileWork: true, CompletedFrame: true}) {
		t.Fatalf("result=%+v", result)
	}
	if runner.Stats[1] != (av1.DecoderFrameWorkTileResidualStats{}) {
		t.Fatalf("stale worker stats were not reset: %+v", runner.Stats[1])
	}
	total, err := runner.TotalStats()
	if err != nil {
		t.Fatal(err)
	}
	if total.TXBs != runner.Stats[0].TXBs || total.Residuals != runner.Stats[0].Residuals {
		t.Fatalf("total stats=%+v worker0=%+v worker1=%+v", total, runner.Stats[0], runner.Stats[1])
	}
}

func TestPublicDecoderFrameWorkResidualEventRunnerResetsStatsOnPlanError(t *testing.T) {
	sequence := av1.SequenceHeader{
		ColorConfig: av1.ColorConfig{
			BitDepth:   8,
			MonoChrome: true,
		},
	}
	event := publicDecoderResidualRunnerFrameEvent()
	var scratchSpans [2]av1.TileSpan
	var scratchJobs [2]av1.TileJob
	var scratchBatches [1]av1.TileBatch
	runnerSize, _, err := av1.DecoderFrameWorkResidualEventRunnerScratchLen(sequence, event, 1, scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		t.Fatal(err)
	}
	runner, err := av1.BindDecoderFrameWorkBatchResidualRunner(runnerSize, publicDecoderBatchResidualRunnerScratch(runnerSize))
	if err != nil {
		t.Fatal(err)
	}
	runner.Stats[0] = av1.DecoderFrameWorkTileResidualStats{TXBs: 33, Residuals: 22}
	stats := av1.DecoderFrameWorkTileResidualStats{TXBs: 44, Residuals: 11}

	_, err = av1.RunDecoderFrameWorkEventWithResidualRunner(av1.DecoderFrameWorkResidualEventRequest{
		Runner:   &runner,
		Sequence: sequence,
		Event:    event,
		Workers:  1,
		Stats:    &stats,
	})
	if !errors.Is(err, av1.ErrDecoderInvalidFrameWorkState) {
		t.Fatalf("plan error=%v want %v", err, av1.ErrDecoderInvalidFrameWorkState)
	}
	if runner.Stats[0] != (av1.DecoderFrameWorkTileResidualStats{}) {
		t.Fatalf("plan-error stats were not reset: %+v", runner.Stats[0])
	}
	if stats != (av1.DecoderFrameWorkTileResidualStats{}) {
		t.Fatalf("plan-error aggregate stats were not reset: %+v", stats)
	}
}

func TestPublicDecoderFrameWorkResidualEventSupportedPostFilterScratchRunner(t *testing.T) {
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
	event.CDEF = av1.CDEFParams{
		Damping:       5,
		StrengthCount: 1,
		YStrength:     [av1.MaxCDEFStrengths]uint8{63},
	}
	var scratchSpans [2]av1.TileSpan
	var scratchJobs [2]av1.TileJob
	var scratchBatches [1]av1.TileBatch
	eventScratch, err := av1.DecoderFrameWorkResidualEventScratchLen(sequence, event, 1, scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		t.Fatal(err)
	}
	runner, err := av1.BindDecoderFrameWorkBatchResidualRunner(eventScratch.Runner, publicDecoderBatchResidualRunnerScratch(eventScratch.Runner))
	if err != nil {
		t.Fatal(err)
	}
	side, err := av1.BindDecoderFrameWorkResidualEventSideData(sequence, event, publicDecoderFrameWorkSideDataScratch(eventScratch.SideData))
	if err != nil {
		t.Fatal(err)
	}

	var probeOutput av1.Frame
	probeCtx, err := av1.DecoderFrameWorkPostFilterScratchContext(sequence, event, 64, &side, &probeOutput)
	if err != nil {
		t.Fatal(err)
	}
	var probe av1.DecoderFrameWorkSupportedPostFilterScratchRunner
	exact, err := probe.ScratchLen(probeCtx)
	if err != nil {
		t.Fatal(err)
	}
	arenaSize := av1.DecoderFrameWorkPostFilterRequestScratchLen(exact)
	postRunner := av1.DecoderFrameWorkSupportedPostFilterScratchRunner{
		Scratch: publicDecoderPostFilterRequestScratch(arenaSize),
	}

	pool := publicDecoderPostFilterFramePool(t, probeOutput.Format, 1)
	var refs av1.DecoderSurfaceReferences
	var state av1.DecoderFrameWorkState
	var referenceSurfaces [av1.InterRefsPerFrame]int
	var referenceFrames [av1.InterRefsPerFrame]*av1.Frame
	var spans [2]av1.TileSpan
	var jobs [2]av1.TileJob
	var batches [1]av1.TileBatch
	var releases [av1.RefFrames]int
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

	result, err := eventRunner.RunWithPostFilterRunner(sequence, event, &side, &postRunner)
	if err != nil {
		t.Fatal(err)
	}
	cdefRead := false
	for _, read := range side.CDEFIndexMap.Read {
		cdefRead = cdefRead || read
	}
	if result.Run != (av1.DecoderFrameWorkStepResult{ExecutedTileWork: true, CompletedFrame: true}) ||
		result.Output == nil ||
		state.Active() ||
		!postRunner.Result.Completed.Has(av1.DecoderFrameWorkPostFilterCDEF) ||
		postRunner.Result.CDEF.Units == 0 ||
		postRunner.Context.RemainingPostFilters() != 0 ||
		postRunner.Size.CDEF.Input != exact.CDEF.Input ||
		postRunner.Request.CDEF.IndexMap.Rows != side.CDEFIndexMap.Rows ||
		!cdefRead {
		t.Fatalf("result=%+v active=%v post=%+v side=%+v", result, state.Active(), postRunner, side.CDEFIndexMap)
	}
	if postRunner.Context.Event.SequenceHeader.ColorConfig.BitDepth != sequence.ColorConfig.BitDepth ||
		!postRunner.Context.Event.SequenceHeader.EnableCDEF {
		t.Fatalf("postfilter sequence=%+v want %+v", postRunner.Context.Event.SequenceHeader, sequence)
	}
	slot, ok := refs.ReferenceSlot(0)
	if !ok || slot < 0 {
		t.Fatalf("decoded frame was not published to reference slot 0: slot=%d ok=%v", slot, ok)
	}
}

func TestPublicDecoderFrameWorkResidualEventCallerPostFilterScratchRunner(t *testing.T) {
	workerPool, err := av1.NewTileWorkerPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	sequence := av1.SequenceHeader{
		EnableCDEF:     true,
		EnableSuperRes: true,
		ColorConfig: av1.ColorConfig{
			BitDepth:   8,
			MonoChrome: true,
		},
	}
	event := publicDecoderResidualRunnerFrameEvent()
	event.FrameSize.UpscaledWidth = 160
	event.FrameSize.SuperResEnabled = true
	event.FrameSize.SuperResDenominator = 13
	event.CDEF = av1.CDEFParams{
		Damping:       5,
		StrengthCount: 1,
		YStrength:     [av1.MaxCDEFStrengths]uint8{63},
	}
	event.FilmGrain = av1.FilmGrainParams{
		ParamsPresent: true,
		Apply:         true,
		Seed:          0x1234,
		BitDepth:      8,
		NumYPoints:    1,
		YPoints:       [av1.MaxFilmGrainYPoints][2]uint8{{0, 64}},
		ScalingShift:  8,
		Overlap:       true,
	}
	var scratchSpans [2]av1.TileSpan
	var scratchJobs [2]av1.TileJob
	var scratchBatches [1]av1.TileBatch
	eventScratch, err := av1.DecoderFrameWorkResidualEventScratchLen(sequence, event, 1, scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		t.Fatal(err)
	}
	runner, err := av1.BindDecoderFrameWorkBatchResidualRunner(eventScratch.Runner, publicDecoderBatchResidualRunnerScratch(eventScratch.Runner))
	if err != nil {
		t.Fatal(err)
	}
	side, err := av1.BindDecoderFrameWorkResidualEventSideData(sequence, event, publicDecoderFrameWorkSideDataScratch(eventScratch.SideData))
	if err != nil {
		t.Fatal(err)
	}

	var scratchOutput av1.Frame
	scratchCtx, err := av1.DecoderFrameWorkPostFilterScratchContext(sequence, event, 64, &side, &scratchOutput)
	if err != nil {
		t.Fatal(err)
	}
	var postRunner av1.DecoderFrameWorkCallerPostFilterScratchRunner
	first, err := postRunner.ScratchLen(scratchCtx)
	if err != nil {
		t.Fatal(err)
	}
	postRunner.Scratch = publicDecoderPostFilterRequestScratch(av1.DecoderFrameWorkPostFilterRequestScratchLen(first))
	full, err := postRunner.ScratchLen(scratchCtx)
	if err != nil {
		t.Fatal(err)
	}
	postRunner.Scratch = publicDecoderPostFilterRequestScratch(av1.DecoderFrameWorkPostFilterRequestScratchLen(full))

	pool := publicDecoderPostFilterFramePool(t, scratchOutput.Format, 1)
	var refs av1.DecoderSurfaceReferences
	var state av1.DecoderFrameWorkState
	var referenceSurfaces [av1.InterRefsPerFrame]int
	var referenceFrames [av1.InterRefsPerFrame]*av1.Frame
	var spans [2]av1.TileSpan
	var jobs [2]av1.TileJob
	var batches [1]av1.TileBatch
	var releases [av1.RefFrames]int
	var outputs [2]*av1.Frame
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
		SideData:          &side,
		Outputs:           outputs[:],
	}

	result, err := eventRunner.RunWithPostFilterRunner(sequence, event, &side, &postRunner)
	if err != nil {
		t.Fatal(err)
	}
	wantCompleted := av1.DecoderFrameWorkPostFilterCDEF |
		av1.DecoderFrameWorkPostFilterSuperRes |
		av1.DecoderFrameWorkPostFilterFilmGrain
	if result.Run != (av1.DecoderFrameWorkStepResult{ExecutedTileWork: true, CompletedFrame: true}) ||
		result.Output == nil ||
		state.Active() ||
		postRunner.Result.Completed != wantCompleted ||
		postRunner.Result.SuperRes.Output.Format.Width != int(event.FrameSize.UpscaledWidth) ||
		postRunner.Context.Output != result.Output ||
		postRunner.Context.Output.Format.Width != int(event.FrameSize.UpscaledWidth) ||
		!postRunner.Context.DetachedPostFilterOutput() ||
		postRunner.Context.RemainingPostFilters() != 0 {
		t.Fatalf("result=%+v active=%v post=%+v detached=%v", result, state.Active(), postRunner, postRunner.Context.DetachedPostFilterOutput())
	}
	slot, ok := refs.ReferenceSlot(0)
	if !ok || slot < 0 {
		t.Fatalf("decoded frame was not published to reference slot 0: slot=%d ok=%v", slot, ok)
	}

	pool.Reset()
	refs.Reset()
	state.Reset()
	outputs[0] = nil
	events := [...]av1.DecoderEvent{
		{Kind: av1.DecoderEventSequenceHeader, SequenceHeader: sequence, NewCodedVideoSequence: true},
		event,
	}
	listResult, err := eventRunner.RunEventsWithPostFilterRunner(av1.SequenceHeader{}, events[:], publicDecoderFrameWorkSideDataScratch(eventScratch.SideData), &postRunner)
	if err != nil {
		t.Fatal(err)
	}
	if listResult.OutputCount != 1 ||
		listResult.Last.Output == nil ||
		outputs[0] != listResult.Last.Output ||
		outputs[0] != postRunner.Context.Output ||
		postRunner.Context.Output.Format.Width != int(event.FrameSize.UpscaledWidth) ||
		!postRunner.Context.DetachedPostFilterOutput() {
		t.Fatalf("list result=%+v output=%p postOutput=%p", listResult, outputs[0], postRunner.Context.Output)
	}

	allocs := testing.AllocsPerRun(100, func() {
		pool.Reset()
		refs.Reset()
		state.Reset()
		result, err = eventRunner.RunWithPostFilterRunner(sequence, event, &side, &postRunner)
		if err != nil {
			return
		}
		if result.Output == nil ||
			result.Run != (av1.DecoderFrameWorkStepResult{ExecutedTileWork: true, CompletedFrame: true}) ||
			postRunner.Result.Completed != wantCompleted ||
			postRunner.Context.RemainingPostFilters() != 0 ||
			!postRunner.Context.DetachedPostFilterOutput() {
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
		t.Fatalf("caller residual event runner allocated: %f", allocs)
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

func TestPublicBindDecoderFrameWorkResidualEventRunner(t *testing.T) {
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
	size, err := av1.DecoderFrameWorkResidualEventScratchLen(sequence, event, 1, scratchSpans[:], scratchJobs[:], scratchBatches[:])
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
	var state av1.DecoderFrameWorkState
	var referenceSurfaces [av1.InterRefsPerFrame]int
	var referenceFrames [av1.InterRefsPerFrame]*av1.Frame
	var releases [av1.RefFrames]int
	var stats av1.DecoderFrameWorkTileResidualStats
	runtime := av1.DecoderFrameWorkResidualEventRuntime{
		State:             &state,
		Refs:              &refs,
		FramePool:         &pool,
		Align:             64,
		ReferenceSurfaces: referenceSurfaces[:],
		ReferenceFrames:   referenceFrames[:],
		Releases:          releases[:],
		WorkerPool:        workerPool,
		Stats:             &stats,
	}
	scratch := publicDecoderResidualEventScratch(size)
	var batchRunner av1.DecoderFrameWorkBatchResidualRunner
	eventRunner, side, err := av1.BindDecoderFrameWorkResidualEventRunner(size, sequence, event, runtime, scratch, &batchRunner)
	if err != nil {
		t.Fatal(err)
	}
	if eventRunner.BatchRunner != &batchRunner ||
		eventRunner.Workers != size.Runner.Workers ||
		len(eventRunner.Spans) != size.Plan.SpanCount ||
		len(eventRunner.Jobs) != size.Plan.JobCount ||
		len(eventRunner.Batches) != size.Plan.BatchCount {
		t.Fatalf("event runner=%+v size=%+v", eventRunner, size)
	}

	result, err := eventRunner.Run(sequence, event, &side, nil)
	if err != nil {
		t.Fatal(err)
	}
	if result.Run != (av1.DecoderFrameWorkStepResult{ExecutedTileWork: true, CompletedFrame: true}) ||
		result.Output == nil ||
		state.Active() ||
		stats.TXBs == 0 ||
		stats.Residuals == 0 {
		t.Fatalf("result=%+v active=%v stats=%+v", result, state.Active(), stats)
	}
	if _, ok := refs.ReferenceSlot(0); !ok {
		t.Fatal("bound event runner did not publish decoded frame")
	}

	allocScratch := publicDecoderResidualEventScratch(size)
	var allocBatchRunner av1.DecoderFrameWorkBatchResidualRunner
	allocs := testing.AllocsPerRun(1000, func() {
		_, _, err = av1.BindDecoderFrameWorkResidualEventRunner(size, sequence, event, runtime, allocScratch, &allocBatchRunner)
	})
	if err != nil {
		t.Fatal(err)
	}
	if allocs != 0 {
		t.Fatalf("BindDecoderFrameWorkResidualEventRunner allocated: %f", allocs)
	}
}

func TestPublicBindDecoderFrameWorkResidualEventRunnerRejectsInvalidScratch(t *testing.T) {
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
	size, err := av1.DecoderFrameWorkResidualEventScratchLen(sequence, event, 1, scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		t.Fatal(err)
	}
	runtime := av1.DecoderFrameWorkResidualEventRuntime{Align: 64}
	scratch := publicDecoderResidualEventScratch(size)

	if _, _, err := av1.BindDecoderFrameWorkResidualEventRunner(size, sequence, event, runtime, scratch, nil); !errors.Is(err, av1.ErrThreadingInvalidBatch) {
		t.Fatalf("nil batch runner err=%v want %v", err, av1.ErrThreadingInvalidBatch)
	}

	var batchRunner av1.DecoderFrameWorkBatchResidualRunner
	shortPlanScratch := scratch
	shortPlanScratch.Jobs = shortPlanScratch.Jobs[:size.Plan.JobCount-1]
	if _, _, err := av1.BindDecoderFrameWorkResidualEventRunner(size, sequence, event, runtime, shortPlanScratch, &batchRunner); !errors.Is(err, av1.ErrFrameShortBuffer) {
		t.Fatalf("short plan scratch err=%v want %v", err, av1.ErrFrameShortBuffer)
	}

	shortRunnerScratch := scratch
	shortRunnerScratch.Runner.Stats = shortRunnerScratch.Runner.Stats[:size.Runner.Workers-1]
	if _, _, err := av1.BindDecoderFrameWorkResidualEventRunner(size, sequence, event, runtime, shortRunnerScratch, &batchRunner); !errors.Is(err, av1.ErrFrameShortBuffer) {
		t.Fatalf("short runner scratch err=%v want %v", err, av1.ErrFrameShortBuffer)
	}

	shortSideScratch := scratch
	shortSideScratch.SideData.LoopFilterMap = shortSideScratch.SideData.LoopFilterMap[:size.SideData.LoopFilterMap-1]
	if _, _, err := av1.BindDecoderFrameWorkResidualEventRunner(size, sequence, event, runtime, shortSideScratch, &batchRunner); !errors.Is(err, av1.ErrFrameShortBuffer) {
		t.Fatalf("short side-data scratch err=%v want %v", err, av1.ErrFrameShortBuffer)
	}
}

func TestPublicDecoderFrameWorkResidualEventRunnerRunEvents(t *testing.T) {
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
	frame := publicDecoderResidualRunnerFrameEvent()
	header := frame
	header.Kind = av1.DecoderEventFrameHeader
	header.Unit.Payload = nil
	tile := frame
	tile.Kind = av1.DecoderEventTileGroup
	events := [...]av1.DecoderEvent{
		{
			Kind:                  av1.DecoderEventSequenceHeader,
			SequenceHeader:        sequence,
			NewCodedVideoSequence: true,
		},
		header,
		tile,
	}

	var scratchSpans [2]av1.TileSpan
	var scratchJobs [2]av1.TileJob
	var scratchBatches [1]av1.TileBatch
	size, err := av1.DecoderFrameWorkResidualEventsScratchLen(sequence, events[:], 1, scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		t.Fatal(err)
	}
	if size.Plan != (av1.DecoderTileWorkPlan{SpanCount: 2, JobCount: 2, BatchCount: 1}) ||
		size.Runner.Workers != 1 ||
		size.SideData.CDEFIndexMap == 0 ||
		size.SideData.LoopFilterMap == 0 {
		t.Fatalf("events scratch size=%+v", size)
	}
	singleSize, err := av1.DecoderFrameWorkResidualEventScratchLen(sequence, tile, 1, scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		t.Fatal(err)
	}
	if size != singleSize {
		t.Fatalf("events scratch=%+v single=%+v", size, singleSize)
	}

	pool := publicDecoderPostFilterFramePool(t, av1.FrameFormat{
		Width:      int(frame.FrameSize.CodedWidth),
		Height:     int(frame.FrameSize.Height),
		BitDepth:   8,
		MonoChrome: true,
		Align:      64,
	}, 1)
	var refs av1.DecoderSurfaceReferences
	var state av1.DecoderFrameWorkState
	var referenceSurfaces [av1.InterRefsPerFrame]int
	var referenceFrames [av1.InterRefsPerFrame]*av1.Frame
	var releases [av1.RefFrames]int
	var stats av1.DecoderFrameWorkTileResidualStats
	var eventSideData av1.DecoderFrameWorkSideData
	var batchRunner av1.DecoderFrameWorkBatchResidualRunner
	var outputs [2]*av1.Frame
	scratch := publicDecoderResidualEventScratch(size)
	eventRunner, _, err := av1.BindDecoderFrameWorkResidualEventRunner(size, sequence, tile, av1.DecoderFrameWorkResidualEventRuntime{
		State:             &state,
		Refs:              &refs,
		FramePool:         &pool,
		Align:             64,
		ReferenceSurfaces: referenceSurfaces[:],
		ReferenceFrames:   referenceFrames[:],
		Releases:          releases[:],
		WorkerPool:        workerPool,
		SideData:          &eventSideData,
		Stats:             &stats,
		Outputs:           outputs[:],
	}, scratch, &batchRunner)
	if err != nil {
		t.Fatal(err)
	}
	postCalls := 0
	result, err := eventRunner.RunEvents(av1.SequenceHeader{}, events[:], scratch.SideData, func(ctx av1.DecoderFrameWorkPostFilterContext) error {
		postCalls++
		if ctx.Step.Kind != av1.DecoderFrameWorkStepTile || !ctx.ExecutedTileWork {
			t.Fatalf("postfilter ctx=%+v", ctx)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Count != len(events) ||
		result.ExecutedTileWork != 1 ||
		result.CompletedFrames != 1 ||
		result.OutputCount != 1 ||
		len(result.Outputs) != 1 ||
		result.Last.Step.Kind != av1.DecoderFrameWorkStepTile ||
		result.Last.Output == nil ||
		outputs[0] != result.Last.Output ||
		result.Outputs[0] != result.Last.Output ||
		state.Active() ||
		postCalls != 1 ||
		stats != result.Stats ||
		stats.TXBs == 0 ||
		stats.Residuals == 0 {
		t.Fatalf("result=%+v active=%v postCalls=%d stats=%+v", result, state.Active(), postCalls, stats)
	}
	if _, ok := refs.ReferenceSlot(0); !ok {
		t.Fatal("event list run did not publish decoded frame")
	}

	shortOutputRunner := eventRunner
	shortOutputRunner.Outputs = outputs[:0]
	if _, err := shortOutputRunner.RunEvents(sequence, events[:], scratch.SideData, nil); !errors.Is(err, av1.ErrFrameShortBuffer) {
		t.Fatalf("short event-list output err=%v want %v", err, av1.ErrFrameShortBuffer)
	}

	noSideRunner := eventRunner
	noSideRunner.SideData = nil
	if _, err := noSideRunner.RunEvents(sequence, events[:], scratch.SideData, nil); !errors.Is(err, av1.ErrDecoderInvalidFrameWorkState) {
		t.Fatalf("missing event-list side data err=%v want %v", err, av1.ErrDecoderInvalidFrameWorkState)
	}

	shortSideScratch := scratch.SideData
	shortSideScratch.CDEFIndexMap = shortSideScratch.CDEFIndexMap[:size.SideData.CDEFIndexMap-1]
	if _, err := eventRunner.RunEvents(sequence, events[:], shortSideScratch, nil); !errors.Is(err, av1.ErrFrameShortBuffer) {
		t.Fatalf("short event-list side scratch err=%v want %v", err, av1.ErrFrameShortBuffer)
	}
}

func TestPublicDecoderFrameWorkResidualEventRunnerShowExistingOutput(t *testing.T) {
	sequence := av1.SequenceHeader{ColorConfig: av1.ColorConfig{
		BitDepth:   8,
		MonoChrome: true,
	}}
	event := av1.DecoderEvent{
		Kind: av1.DecoderEventExistingFrame,
		FrameHeader: av1.FrameHeaderPrefix{
			ShowExistingFrame: true,
			ExistingFrameIdx:  0,
		},
		ExistingFrame: av1.ReferenceFrame{FrameType: av1.FrameTypeInter},
		FrameSize: av1.FrameSize{
			CodedWidth:    16,
			UpscaledWidth: 16,
			Height:        16,
		},
	}
	var scratchSpans [1]av1.TileSpan
	var scratchJobs [1]av1.TileJob
	var scratchBatches [1]av1.TileBatch
	size, err := av1.DecoderFrameWorkResidualEventScratchLen(sequence, event, 1, scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		t.Fatal(err)
	}
	if size.Runner != (av1.DecoderFrameWorkBatchResidualRunnerScratchSize{}) ||
		size.SideData != (av1.DecoderFrameWorkSideDataScratchSize{}) ||
		size.Plan != (av1.DecoderTileWorkPlan{}) ||
		size.Outputs != 1 {
		t.Fatalf("show-existing scratch size=%+v", size)
	}
	listSize, err := av1.DecoderFrameWorkResidualEventsScratchLen(sequence, []av1.DecoderEvent{event}, 1, scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		t.Fatal(err)
	}
	if listSize != size {
		t.Fatalf("show-existing list scratch=%+v single=%+v", listSize, size)
	}

	pool := publicDecoderPostFilterFramePool(t, av1.FrameFormat{
		Width:      16,
		Height:     16,
		BitDepth:   8,
		MonoChrome: true,
		Align:      64,
	}, 1)
	referenceSlot, referenceFrame, err := pool.Acquire()
	if err != nil {
		t.Fatal(err)
	}
	var refs av1.DecoderSurfaceReferences
	var releases [av1.RefFrames]int
	if _, err := refs.Refresh(1<<0, referenceSlot, releases[:]); err != nil {
		t.Fatal(err)
	}

	var state av1.DecoderFrameWorkState
	var outputs [1]*av1.Frame
	eventRunner, _, err := av1.BindDecoderFrameWorkResidualEventRunner(size, sequence, event, av1.DecoderFrameWorkResidualEventRuntime{
		State:     &state,
		Refs:      &refs,
		FramePool: &pool,
		Releases:  releases[:],
		Outputs:   outputs[:],
	}, publicDecoderResidualEventScratch(size), nil)
	if err != nil {
		t.Fatal(err)
	}

	result, err := eventRunner.RunEvents(sequence, []av1.DecoderEvent{event}, av1.DecoderFrameWorkSideDataScratch{}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if result.Count != 1 ||
		result.ExecutedTileWork != 0 ||
		result.CompletedFrames != 0 ||
		result.OutputCount != 1 ||
		len(result.Outputs) != 1 ||
		result.Last.Step.Kind != av1.DecoderFrameWorkStepShowExisting ||
		result.Last.Output != referenceFrame ||
		outputs[0] != referenceFrame ||
		result.Outputs[0] != referenceFrame ||
		state.Active() {
		t.Fatalf("show-existing result=%+v output=%p/%p active=%v", result, result.Last.Output, referenceFrame, state.Active())
	}

	shortOutputRunner := eventRunner
	shortOutputRunner.Outputs = outputs[:0]
	if _, err := shortOutputRunner.RunEvents(sequence, []av1.DecoderEvent{event}, av1.DecoderFrameWorkSideDataScratch{}, nil); !errors.Is(err, av1.ErrFrameShortBuffer) {
		t.Fatalf("short show-existing output err=%v want %v", err, av1.ErrFrameShortBuffer)
	}
}

func TestPublicDecoderFrameWorkResidualEventRunnerFrameHeaderWithoutBatchRunner(t *testing.T) {
	payload := publicDecoderResidualLowOverheadStream()
	var probeStream av1.DecoderStream
	var events [4]av1.DecoderEvent
	count, err := probeStream.PushLowOverhead(payload, events[:])
	if err != nil {
		t.Fatal(err)
	}
	if count < 2 || events[1].Kind != av1.DecoderEventFrameHeader {
		t.Fatalf("events=%+v", events[:count])
	}
	sequence, ok := probeStream.SequenceHeader()
	if !ok {
		t.Fatal("probe stream missing sequence")
	}
	var scratchSpans [1]av1.TileSpan
	var scratchJobs [1]av1.TileJob
	var scratchBatches [1]av1.TileBatch
	size, err := av1.DecoderFrameWorkResidualEventScratchLen(sequence, events[1], 1, scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		t.Fatal(err)
	}
	if size.Runner != (av1.DecoderFrameWorkBatchResidualRunnerScratchSize{}) ||
		size.Plan != (av1.DecoderTileWorkPlan{}) ||
		size.Outputs != 0 ||
		size.SideData.CDEFIndexMap == 0 ||
		size.SideData.LoopFilterMap == 0 {
		t.Fatalf("frame-header scratch size=%+v", size)
	}

	pool := publicDecoderPostFilterFramePool(t, av1.FrameFormat{
		Width:        int(events[1].FrameSize.CodedWidth),
		Height:       int(events[1].FrameSize.Height),
		BitDepth:     8,
		MonoChrome:   true,
		SubsamplingX: true,
		SubsamplingY: true,
		Align:        64,
	}, 1)
	var state av1.DecoderFrameWorkState
	var refs av1.DecoderSurfaceReferences
	var side av1.DecoderFrameWorkSideData
	var stats av1.DecoderFrameWorkTileResidualStats
	eventRunner, _, err := av1.BindDecoderFrameWorkResidualEventRunner(size, sequence, events[1], av1.DecoderFrameWorkResidualEventRuntime{
		State:     &state,
		Refs:      &refs,
		FramePool: &pool,
		Align:     64,
		SideData:  &side,
		Stats:     &stats,
	}, publicDecoderResidualEventScratch(size), nil)
	if err != nil {
		t.Fatal(err)
	}

	result, err := eventRunner.RunEvents(sequence, events[1:2], publicDecoderFrameWorkSideDataScratch(size.SideData), nil)
	if err != nil {
		t.Fatal(err)
	}
	if result.Count != 1 ||
		result.ExecutedTileWork != 0 ||
		result.CompletedFrames != 0 ||
		result.OutputCount != 0 ||
		len(result.Outputs) != 0 ||
		result.Last.Step.Kind != av1.DecoderFrameWorkStepBegin ||
		!state.Active() ||
		stats != (av1.DecoderFrameWorkTileResidualStats{}) {
		t.Fatalf("frame-header result=%+v active=%v stats=%+v", result, state.Active(), stats)
	}
}

func TestPublicDecoderFrameWorkResidualEventRunnerRunEventsAllocs(t *testing.T) {
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
	frame := publicDecoderResidualRunnerFrameEvent()
	events := [...]av1.DecoderEvent{
		{
			Kind:                  av1.DecoderEventSequenceHeader,
			SequenceHeader:        sequence,
			NewCodedVideoSequence: true,
		},
		frame,
	}
	var scratchSpans [2]av1.TileSpan
	var scratchJobs [2]av1.TileJob
	var scratchBatches [1]av1.TileBatch
	size, err := av1.DecoderFrameWorkResidualEventsScratchLen(sequence, events[:], 1, scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		t.Fatal(err)
	}
	pool := publicDecoderPostFilterFramePool(t, av1.FrameFormat{
		Width:      int(frame.FrameSize.CodedWidth),
		Height:     int(frame.FrameSize.Height),
		BitDepth:   8,
		MonoChrome: true,
		Align:      64,
	}, 1)
	var refs av1.DecoderSurfaceReferences
	var state av1.DecoderFrameWorkState
	var referenceSurfaces [av1.InterRefsPerFrame]int
	var referenceFrames [av1.InterRefsPerFrame]*av1.Frame
	var releases [av1.RefFrames]int
	var stats av1.DecoderFrameWorkTileResidualStats
	var eventSideData av1.DecoderFrameWorkSideData
	var batchRunner av1.DecoderFrameWorkBatchResidualRunner
	var outputs [1]*av1.Frame
	scratch := publicDecoderResidualEventScratch(size)
	eventRunner, _, err := av1.BindDecoderFrameWorkResidualEventRunner(size, sequence, frame, av1.DecoderFrameWorkResidualEventRuntime{
		State:             &state,
		Refs:              &refs,
		FramePool:         &pool,
		Align:             64,
		ReferenceSurfaces: referenceSurfaces[:],
		ReferenceFrames:   referenceFrames[:],
		Releases:          releases[:],
		WorkerPool:        workerPool,
		SideData:          &eventSideData,
		Stats:             &stats,
		Outputs:           outputs[:],
	}, scratch, &batchRunner)
	if err != nil {
		t.Fatal(err)
	}

	allocs := testing.AllocsPerRun(1000, func() {
		pool.Reset()
		refs.Reset()
		state.Reset()
		result, runErr := eventRunner.RunEvents(sequence, events[:], scratch.SideData, nil)
		if runErr != nil {
			err = runErr
			return
		}
		if result.Count != len(events) ||
			result.ExecutedTileWork != 1 ||
			result.CompletedFrames != 1 ||
			result.OutputCount != 1 ||
			outputs[0] != result.Last.Output ||
			stats.TXBs == 0 ||
			stats.Residuals == 0 {
			err = av1.ErrThreadingInvalidBatch
		}
	})
	if err != nil {
		t.Fatal(err)
	}
	if allocs != 0 {
		t.Fatalf("DecoderFrameWorkResidualEventRunner.RunEvents allocated: %f", allocs)
	}
}

func TestPublicDecoderFrameWorkResidualStreamRunnerLowOverhead(t *testing.T) {
	workerPool, err := av1.NewTileWorkerPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	payload := publicDecoderResidualLowOverheadStream()
	var probeStream av1.DecoderStream
	var probeEvents [4]av1.DecoderEvent
	count, err := probeStream.PushLowOverhead(payload, probeEvents[:])
	if err != nil {
		t.Fatal(err)
	}
	sequence, ok := probeStream.SequenceHeader()
	if !ok {
		t.Fatal("probe stream missing sequence")
	}
	var scratchSpans [1]av1.TileSpan
	var scratchJobs [1]av1.TileJob
	var scratchBatches [1]av1.TileBatch
	size, err := av1.DecoderFrameWorkResidualEventsScratchLen(sequence, probeEvents[:count], 1, scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		t.Fatal(err)
	}

	pool := publicDecoderPostFilterFramePool(t, av1.FrameFormat{
		Width:        int(probeEvents[count-1].FrameSize.CodedWidth),
		Height:       int(probeEvents[count-1].FrameSize.Height),
		BitDepth:     8,
		MonoChrome:   true,
		SubsamplingX: true,
		SubsamplingY: true,
		Align:        64,
	}, 2)
	var stream av1.DecoderStream
	var refs av1.DecoderSurfaceReferences
	var state av1.DecoderFrameWorkState
	var referenceSurfaces [av1.InterRefsPerFrame]int
	var referenceFrames [av1.InterRefsPerFrame]*av1.Frame
	var releases [av1.RefFrames]int
	var stats av1.DecoderFrameWorkTileResidualStats
	var side av1.DecoderFrameWorkSideData
	var batchRunner av1.DecoderFrameWorkBatchResidualRunner
	var outputs [2]*av1.Frame
	scratch := publicDecoderResidualEventScratch(size)
	eventRunner, _, err := av1.BindDecoderFrameWorkResidualEventRunner(size, sequence, probeEvents[count-1], av1.DecoderFrameWorkResidualEventRuntime{
		State:             &state,
		Refs:              &refs,
		FramePool:         &pool,
		Align:             64,
		ReferenceSurfaces: referenceSurfaces[:],
		ReferenceFrames:   referenceFrames[:],
		Releases:          releases[:],
		WorkerPool:        workerPool,
		SideData:          &side,
		Stats:             &stats,
		Outputs:           outputs[:],
	}, scratch, &batchRunner)
	if err != nil {
		t.Fatal(err)
	}
	runner := av1.DecoderFrameWorkResidualStreamRunner{
		Stream:          &stream,
		EventRunner:     eventRunner,
		Events:          make([]av1.DecoderEvent, len(probeEvents)),
		SideDataScratch: scratch.SideData,
	}

	result, err := runner.RunLowOverhead(payload, nil)
	if err != nil {
		t.Fatal(err)
	}
	if result.EventCount != count ||
		result.Run.Count != count ||
		result.Run.ExecutedTileWork != 1 ||
		result.Run.CompletedFrames != 1 ||
		result.Run.OutputCount != 1 ||
		len(result.Run.Outputs) != 1 ||
		result.Run.Last.Output == nil ||
		result.Run.Outputs[0] != result.Run.Last.Output ||
		state.Active() ||
		stats.TXBs == 0 ||
		stats.Residuals == 0 {
		t.Fatalf("stream result=%+v active=%v stats=%+v", result, state.Active(), stats)
	}
	if _, ok := refs.ReferenceSlot(0); !ok {
		t.Fatal("low-overhead stream runner did not publish decoded frame")
	}
}

func TestPublicDecoderFrameWorkResidualStreamRunnerLowOverheadShowExistingOutput(t *testing.T) {
	var stream av1.DecoderStream
	var prime []byte
	keyFrame := append([]byte{}, publicDecoderResidualShownKeyFrameHeaderPayload()...)
	keyFrame = append(keyFrame, 0xaa)
	interFrame := append([]byte{}, publicDecoderResidualInterFrameHeaderPayload()...)
	interFrame = append(interFrame, 0xbb)
	prime = appendPublicLowOverheadOBU(prime, av1.OBUSequenceHeader, publicDecoderResidualRealtimeSequenceHeaderPayload())
	prime = appendPublicLowOverheadOBU(prime, av1.OBUFrame, keyFrame)
	prime = appendPublicLowOverheadOBU(prime, av1.OBUFrame, interFrame)
	var primeEvents [3]av1.DecoderEvent
	if count, err := stream.PushLowOverhead(prime, primeEvents[:]); err != nil {
		t.Fatal(err)
	} else if count != len(primeEvents) || primeEvents[2].FrameHeader.FrameType != av1.FrameTypeInter {
		t.Fatalf("prime count=%d events=%+v", count, primeEvents)
	}

	var showExisting []byte
	showExisting = appendPublicLowOverheadOBU(showExisting, av1.OBUFrameHeader, publicDecoderResidualShowExistingFrameHeaderPayload(0))
	var events [1]av1.DecoderEvent
	var scratchSpans [1]av1.TileSpan
	var scratchJobs [1]av1.TileJob
	var scratchBatches [1]av1.TileBatch
	size, err := av1.DecoderFrameWorkResidualLowOverheadStreamScratchLen(stream, showExisting, 1, events[:], scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		t.Fatal(err)
	}
	if size.Events != 1 ||
		size.Event.Outputs != 1 ||
		size.Event.Runner != (av1.DecoderFrameWorkBatchResidualRunnerScratchSize{}) ||
		size.Event.SideData != (av1.DecoderFrameWorkSideDataScratchSize{}) ||
		size.Event.Plan != (av1.DecoderTileWorkPlan{}) ||
		events[0].Kind != av1.DecoderEventExistingFrame ||
		events[0].ExistingFrame.FrameType != av1.FrameTypeInter {
		t.Fatalf("show-existing stream scratch size=%+v event=%+v", size, events[0])
	}
	sequence, ok := stream.SequenceHeader()
	if !ok {
		t.Fatal("stream missing sequence")
	}

	pool := publicDecoderPostFilterFramePool(t, av1.FrameFormat{
		Width:        int(events[0].FrameSize.CodedWidth),
		Height:       int(events[0].FrameSize.Height),
		BitDepth:     8,
		SubsamplingX: true,
		SubsamplingY: true,
		Align:        64,
	}, 1)
	referenceSlot, referenceFrame, err := pool.Acquire()
	if err != nil {
		t.Fatal(err)
	}
	var refs av1.DecoderSurfaceReferences
	var releases [av1.RefFrames]int
	if _, err := refs.Refresh(1<<0, referenceSlot, releases[:]); err != nil {
		t.Fatal(err)
	}

	var state av1.DecoderFrameWorkState
	var stats av1.DecoderFrameWorkTileResidualStats
	eventRunner, _, err := av1.BindDecoderFrameWorkResidualEventRunner(size.Event, sequence, events[0], av1.DecoderFrameWorkResidualEventRuntime{
		State:     &state,
		Refs:      &refs,
		FramePool: &pool,
		Releases:  releases[:],
		Stats:     &stats,
	}, publicDecoderResidualEventScratch(size.Event), nil)
	if err != nil {
		t.Fatal(err)
	}
	streamScratch := publicDecoderResidualStreamScratch(size)
	runner, err := av1.BindDecoderFrameWorkResidualStreamRunner(size, &stream, eventRunner, streamScratch)
	if err != nil {
		t.Fatal(err)
	}

	result, err := runner.RunLowOverhead(showExisting, nil)
	if err != nil {
		t.Fatal(err)
	}
	if result.EventCount != 1 ||
		result.Run.Count != 1 ||
		result.Run.ExecutedTileWork != 0 ||
		result.Run.CompletedFrames != 0 ||
		result.Run.OutputCount != 1 ||
		len(result.Run.Outputs) != 1 ||
		result.Run.Last.Step.Kind != av1.DecoderFrameWorkStepShowExisting ||
		result.Run.Last.Output != referenceFrame ||
		result.Run.Outputs[0] != referenceFrame ||
		streamScratch.Outputs[0] != referenceFrame ||
		state.Active() ||
		stats != (av1.DecoderFrameWorkTileResidualStats{}) {
		t.Fatalf("show-existing stream result=%+v output=%p/%p active=%v stats=%+v", result, streamScratch.Outputs[0], referenceFrame, state.Active(), stats)
	}
}

func TestPublicDecoderFrameWorkResidualStreamRunnerReset(t *testing.T) {
	var stream av1.DecoderStream
	var events [4]av1.DecoderEvent
	if count, err := stream.PushLowOverhead(publicDecoderResidualLowOverheadStream(), events[:]); err != nil {
		t.Fatal(err)
	} else if count != 3 {
		t.Fatalf("count=%d events=%+v", count, events[:count])
	}
	runner := av1.DecoderFrameWorkResidualStreamRunner{
		Stream:  &stream,
		RTPUsed: 7,
	}
	if !stream.HasSequenceHeader() {
		t.Fatal("stream missing sequence before reset")
	}
	frameHeader := publicDecoderResidualRTPElement(av1.OBUFrameHeader, publicDecoderResidualFrameHeaderPayload())
	var packet [8]byte
	n, _, more, err := av1.PutRTPFragment(packet[:], frameHeader, 0, 2, false)
	if err != nil {
		t.Fatal(err)
	}
	if !more {
		t.Fatal("expected frame header to need a second fragment")
	}
	var rtpBuffer [128]byte
	var rtpSpans [1]av1.RTPObuSpan
	used, count, err := stream.PushRTPPayload(rtpBuffer[:], 0, rtpSpans[:], events[:1], packet[:n])
	if err != nil {
		t.Fatal(err)
	}
	if count != 0 || !stream.InRTPFragment() || used == 0 {
		t.Fatalf("fragment push used=%d count=%d inFragment=%v", used, count, stream.InRTPFragment())
	}
	runner.RTPUsed = used
	if err := runner.ResetRTP(); err != nil {
		t.Fatal(err)
	}
	if runner.RTPUsed != 0 || !stream.HasSequenceHeader() || stream.InRTPFragment() {
		t.Fatalf("ResetRTP runner=%+v hasSequence=%v inFragment=%v", runner, stream.HasSequenceHeader(), stream.InRTPFragment())
	}
	runner.RTPUsed = 9
	if err := runner.Reset(); err != nil {
		t.Fatal(err)
	}
	if runner.RTPUsed != 0 || stream.HasSequenceHeader() || stream.InRTPFragment() {
		t.Fatalf("Reset runner=%+v hasSequence=%v inFragment=%v", runner, stream.HasSequenceHeader(), stream.InRTPFragment())
	}

	var nilRunner *av1.DecoderFrameWorkResidualStreamRunner
	if err := nilRunner.Reset(); !errors.Is(err, av1.ErrDecoderInvalidFrameWorkState) {
		t.Fatalf("nil reset err=%v want %v", err, av1.ErrDecoderInvalidFrameWorkState)
	}
	if err := nilRunner.ResetRTP(); !errors.Is(err, av1.ErrDecoderInvalidFrameWorkState) {
		t.Fatalf("nil rtp reset err=%v want %v", err, av1.ErrDecoderInvalidFrameWorkState)
	}
}

func TestPublicDecoderFrameWorkResidualEventsBindPlan(t *testing.T) {
	payload := publicDecoderResidualLowOverheadStream()
	var stream av1.DecoderStream
	var events [4]av1.DecoderEvent
	count, err := stream.PushLowOverhead(payload, events[:])
	if err != nil {
		t.Fatal(err)
	}
	plan := av1.DecoderFrameWorkResidualEventsBindPlan(av1.SequenceHeader{}, events[:count])
	if !plan.HasEvent() ||
		plan.EventIndex != count-1 ||
		plan.Event.Kind != av1.DecoderEventTileGroup ||
		plan.Sequence.ColorConfig.BitDepth != 8 ||
		!plan.Sequence.ColorConfig.MonoChrome {
		t.Fatalf("low-overhead bind plan=%+v count=%d", plan, count)
	}

	var realtime av1.DecoderStream
	var prime []byte
	keyFrame := append([]byte{}, publicDecoderResidualShownKeyFrameHeaderPayload()...)
	keyFrame = append(keyFrame, 0xaa)
	interFrame := append([]byte{}, publicDecoderResidualInterFrameHeaderPayload()...)
	interFrame = append(interFrame, 0xbb)
	prime = appendPublicLowOverheadOBU(prime, av1.OBUSequenceHeader, publicDecoderResidualRealtimeSequenceHeaderPayload())
	prime = appendPublicLowOverheadOBU(prime, av1.OBUFrame, keyFrame)
	prime = appendPublicLowOverheadOBU(prime, av1.OBUFrame, interFrame)
	var primeEvents [3]av1.DecoderEvent
	if count, err := realtime.PushLowOverhead(prime, primeEvents[:]); err != nil {
		t.Fatal(err)
	} else if count != len(primeEvents) {
		t.Fatalf("prime count=%d events=%+v", count, primeEvents)
	}
	sequence, ok := realtime.SequenceHeader()
	if !ok {
		t.Fatal("realtime stream missing sequence")
	}
	var showExisting []byte
	showExisting = appendPublicLowOverheadOBU(showExisting, av1.OBUFrameHeader, publicDecoderResidualShowExistingFrameHeaderPayload(0))
	var outputEvents [1]av1.DecoderEvent
	outputCount, err := realtime.PushLowOverhead(showExisting, outputEvents[:])
	if err != nil {
		t.Fatal(err)
	}
	outputPlan := av1.DecoderFrameWorkResidualEventsBindPlan(sequence, outputEvents[:outputCount])
	if !outputPlan.HasEvent() ||
		outputPlan.EventIndex != 0 ||
		outputPlan.Event.Kind != av1.DecoderEventExistingFrame ||
		outputPlan.Event.ExistingFrame.FrameType != av1.FrameTypeInter ||
		outputPlan.Sequence.ColorConfig.BitDepth != 8 {
		t.Fatalf("show-existing bind plan=%+v count=%d", outputPlan, outputCount)
	}

	sequenceOnly := av1.SequenceHeader{ColorConfig: av1.ColorConfig{
		BitDepth:   10,
		MonoChrome: true,
	}}
	noEventPlan := av1.DecoderFrameWorkResidualEventsBindPlan(av1.SequenceHeader{}, []av1.DecoderEvent{
		{Kind: av1.DecoderEventSequenceHeader, SequenceHeader: sequenceOnly},
	})
	if noEventPlan.HasEvent() ||
		noEventPlan.EventIndex != -1 ||
		noEventPlan.Sequence.ColorConfig.BitDepth != 10 ||
		!noEventPlan.Sequence.ColorConfig.MonoChrome {
		t.Fatalf("sequence-only bind plan=%+v", noEventPlan)
	}
}

func TestPublicDecoderFrameWorkResidualStreamPlan(t *testing.T) {
	lowOverhead := publicDecoderResidualLowOverheadStream()
	lowOverheads := [...][]byte{
		publicDecoderResidualLowOverheadStream(),
		publicDecoderResidualLowOverheadFrameStream(),
	}
	rtpPayload := publicDecoderResidualRTPPayload()
	rtpPayloads := [...][]byte{
		publicDecoderResidualRTPPayload(),
		publicDecoderResidualRTPFramePayload(),
	}
	var stream av1.DecoderStream
	var events [4]av1.DecoderEvent
	var rtpBuffer [256]byte
	var rtpSpans [4]av1.RTPObuSpan
	var scratchSpans [1]av1.TileSpan
	var scratchJobs [1]av1.TileJob
	var scratchBatches [1]av1.TileBatch

	lowPlan, err := av1.DecoderFrameWorkResidualLowOverheadStreamPlan(stream, lowOverhead, 1, events[:], scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		t.Fatal(err)
	}
	if !lowPlan.HasEvent() ||
		lowPlan.Size.Events != 3 ||
		lowPlan.Size.Event.Outputs != 1 ||
		lowPlan.Bind.EventIndex != 2 ||
		lowPlan.Bind.Event.Kind != av1.DecoderEventTileGroup ||
		lowPlan.Bind.Sequence.ColorConfig.BitDepth != 8 {
		t.Fatalf("low-overhead stream plan=%+v", lowPlan)
	}
	lowSize, err := av1.DecoderFrameWorkResidualLowOverheadStreamScratchLen(stream, lowOverhead, 1, events[:], scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		t.Fatal(err)
	}
	if lowPlan.Size != lowSize {
		t.Fatalf("low-overhead plan size=%+v scratch size=%+v", lowPlan.Size, lowSize)
	}

	lowBatchPlan, err := av1.DecoderFrameWorkResidualLowOverheadStreamsPlan(stream, lowOverheads[:], 1, events[:], scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		t.Fatal(err)
	}
	if !lowBatchPlan.HasEvent() ||
		lowBatchPlan.Size.Events != 3 ||
		lowBatchPlan.Size.Event.Outputs != 2 ||
		lowBatchPlan.Bind.EventIndex != 1 ||
		lowBatchPlan.Bind.Event.Kind != av1.DecoderEventTileGroup {
		t.Fatalf("low-overhead stream batch plan=%+v", lowBatchPlan)
	}
	lowBatchSize, err := av1.DecoderFrameWorkResidualLowOverheadStreamsScratchLen(stream, lowOverheads[:], 1, events[:], scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		t.Fatal(err)
	}
	if lowBatchPlan.Size != lowBatchSize {
		t.Fatalf("low-overhead batch plan size=%+v scratch size=%+v", lowBatchPlan.Size, lowBatchSize)
	}

	rtpPlan, err := av1.DecoderFrameWorkResidualRTPPayloadStreamPlan(stream, 0, rtpPayload, 1, rtpBuffer[:], rtpSpans[:], events[:], scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		t.Fatal(err)
	}
	if !rtpPlan.HasEvent() ||
		rtpPlan.Size.Events != 3 ||
		rtpPlan.Size.RTPSpans != 3 ||
		rtpPlan.Size.Event.Outputs != 1 ||
		rtpPlan.Bind.EventIndex != 2 ||
		rtpPlan.Bind.Event.Kind != av1.DecoderEventTileGroup {
		t.Fatalf("rtp stream plan=%+v", rtpPlan)
	}

	batchPlan, err := av1.DecoderFrameWorkResidualRTPPayloadsStreamPlan(stream, 0, rtpPayloads[:], 1, rtpBuffer[:], rtpSpans[:], events[:], scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		t.Fatal(err)
	}
	if !batchPlan.HasEvent() ||
		batchPlan.Size.Events != 3 ||
		batchPlan.Size.RTPSpans != 3 ||
		batchPlan.Size.Event.Outputs != 2 ||
		batchPlan.Bind.EventIndex != 1 ||
		batchPlan.Bind.Event.Kind != av1.DecoderEventTileGroup ||
		batchPlan.Bind.Event.FrameHeader.OrderHint != rtpPlan.Bind.Event.FrameHeader.OrderHint {
		t.Fatalf("rtp payload batch stream plan=%+v", batchPlan)
	}
	batchSize, err := av1.DecoderFrameWorkResidualRTPPayloadsStreamScratchLen(stream, 0, rtpPayloads[:], 1, rtpBuffer[:], rtpSpans[:], events[:], scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		t.Fatal(err)
	}
	if batchPlan.Size != batchSize {
		t.Fatalf("batch plan size=%+v scratch size=%+v", batchPlan.Size, batchSize)
	}

	shortPlan, err := av1.DecoderFrameWorkResidualLowOverheadStreamPlan(stream, lowOverhead, 1, events[:2], scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if !errors.Is(err, av1.ErrDecoderEventBufferTooSmall) {
		t.Fatalf("short plan err=%v want %v", err, av1.ErrDecoderEventBufferTooSmall)
	}
	if shortPlan.Size.Events != 3 || shortPlan.HasEvent() {
		t.Fatalf("short low-overhead stream plan=%+v", shortPlan)
	}
}

func TestPublicDecoderFrameWorkResidualStreamRunnerRTPPayload(t *testing.T) {
	workerPool, err := av1.NewTileWorkerPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	lowOverhead := publicDecoderResidualLowOverheadStream()
	var probeStream av1.DecoderStream
	var probeEvents [4]av1.DecoderEvent
	count, err := probeStream.PushLowOverhead(lowOverhead, probeEvents[:])
	if err != nil {
		t.Fatal(err)
	}
	sequence, ok := probeStream.SequenceHeader()
	if !ok {
		t.Fatal("probe stream missing sequence")
	}
	var scratchSpans [1]av1.TileSpan
	var scratchJobs [1]av1.TileJob
	var scratchBatches [1]av1.TileBatch
	size, err := av1.DecoderFrameWorkResidualEventsScratchLen(sequence, probeEvents[:count], 1, scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		t.Fatal(err)
	}

	rtpPayload := publicDecoderResidualRTPPayload()
	pool := publicDecoderPostFilterFramePool(t, av1.FrameFormat{
		Width:        int(probeEvents[count-1].FrameSize.CodedWidth),
		Height:       int(probeEvents[count-1].FrameSize.Height),
		BitDepth:     8,
		MonoChrome:   true,
		SubsamplingX: true,
		SubsamplingY: true,
		Align:        64,
	}, 1)
	var stream av1.DecoderStream
	var refs av1.DecoderSurfaceReferences
	var state av1.DecoderFrameWorkState
	var referenceSurfaces [av1.InterRefsPerFrame]int
	var referenceFrames [av1.InterRefsPerFrame]*av1.Frame
	var releases [av1.RefFrames]int
	var stats av1.DecoderFrameWorkTileResidualStats
	var side av1.DecoderFrameWorkSideData
	var batchRunner av1.DecoderFrameWorkBatchResidualRunner
	var outputs [1]*av1.Frame
	scratch := publicDecoderResidualEventScratch(size)
	eventRunner, _, err := av1.BindDecoderFrameWorkResidualEventRunner(size, sequence, probeEvents[count-1], av1.DecoderFrameWorkResidualEventRuntime{
		State:             &state,
		Refs:              &refs,
		FramePool:         &pool,
		Align:             64,
		ReferenceSurfaces: referenceSurfaces[:],
		ReferenceFrames:   referenceFrames[:],
		Releases:          releases[:],
		WorkerPool:        workerPool,
		SideData:          &side,
		Stats:             &stats,
		Outputs:           outputs[:],
	}, scratch, &batchRunner)
	if err != nil {
		t.Fatal(err)
	}
	runner := av1.DecoderFrameWorkResidualStreamRunner{
		Stream:          &stream,
		EventRunner:     eventRunner,
		Events:          make([]av1.DecoderEvent, len(probeEvents)),
		SideDataScratch: scratch.SideData,
		RTPBuffer:       make([]byte, len(lowOverhead)),
		RTPSpans:        make([]av1.RTPObuSpan, len(probeEvents)),
	}

	result, err := runner.RunRTPPayload(rtpPayload, nil)
	if err != nil {
		t.Fatal(err)
	}
	if result.EventCount != count ||
		result.RTPUsed != 0 ||
		runner.RTPUsed != 0 ||
		result.Run.Count != count ||
		result.Run.ExecutedTileWork != 1 ||
		result.Run.CompletedFrames != 1 ||
		stats.TXBs == 0 ||
		stats.Residuals == 0 {
		t.Fatalf("rtp result=%+v retained=%d stats=%+v", result, runner.RTPUsed, stats)
	}
	if _, ok := refs.ReferenceSlot(0); !ok {
		t.Fatal("RTP stream runner did not publish decoded frame")
	}
}

func TestPublicDecoderFrameWorkResidualStreamRunnerLowOverheads(t *testing.T) {
	workerPool, err := av1.NewTileWorkerPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	payloads := [...][]byte{
		publicDecoderResidualLowOverheadStream(),
		publicDecoderResidualLowOverheadFrameStream(),
	}
	var probeStream av1.DecoderStream
	var probeEvents [4]av1.DecoderEvent
	var scratchSpans [1]av1.TileSpan
	var scratchJobs [1]av1.TileJob
	var scratchBatches [1]av1.TileBatch
	plan, err := av1.DecoderFrameWorkResidualLowOverheadStreamsPlan(probeStream, payloads[:], 1, probeEvents[:], scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		t.Fatal(err)
	}
	if plan.Size.Event.Outputs != 2 {
		t.Fatalf("low-overhead batch plan=%+v", plan)
	}

	pool := publicDecoderPostFilterFramePool(t, av1.FrameFormat{
		Width:        int(plan.Bind.Event.FrameSize.CodedWidth),
		Height:       int(plan.Bind.Event.FrameSize.Height),
		BitDepth:     8,
		MonoChrome:   true,
		SubsamplingX: true,
		SubsamplingY: true,
		Align:        64,
	}, 2)
	var stream av1.DecoderStream
	var refs av1.DecoderSurfaceReferences
	var state av1.DecoderFrameWorkState
	var referenceSurfaces [av1.InterRefsPerFrame]int
	var referenceFrames [av1.InterRefsPerFrame]*av1.Frame
	var releases [av1.RefFrames]int
	var stats av1.DecoderFrameWorkTileResidualStats
	var side av1.DecoderFrameWorkSideData
	var batchRunner av1.DecoderFrameWorkBatchResidualRunner
	streamScratch := publicDecoderResidualStreamScratch(plan.Size)
	runner, _, err := av1.BindDecoderFrameWorkResidualStreamPlanRunner(plan, &stream, av1.DecoderFrameWorkResidualEventRuntime{
		State:             &state,
		Refs:              &refs,
		FramePool:         &pool,
		Align:             64,
		ReferenceSurfaces: referenceSurfaces[:],
		ReferenceFrames:   referenceFrames[:],
		Releases:          releases[:],
		WorkerPool:        workerPool,
		SideData:          &side,
		Stats:             &stats,
	}, streamScratch, &batchRunner)
	if err != nil {
		t.Fatal(err)
	}

	var nilRunner *av1.DecoderFrameWorkResidualStreamRunner
	if _, err := nilRunner.RunLowOverheads(nil, nil); !errors.Is(err, av1.ErrDecoderInvalidFrameWorkState) {
		t.Fatalf("nil low-overhead batch runner err=%v want %v", err, av1.ErrDecoderInvalidFrameWorkState)
	}
	empty, err := runner.RunLowOverheads(nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if empty.EventCount != 0 || empty.Run.Count != 0 || empty.Run.OutputCount != 0 {
		t.Fatalf("empty low-overhead batch result=%+v", empty)
	}

	result, err := runner.RunLowOverheads(payloads[:], nil)
	if err != nil {
		t.Fatal(err)
	}
	if result.EventCount != 5 ||
		result.Run.Count != 5 ||
		result.Run.ExecutedTileWork != 2 ||
		result.Run.CompletedFrames != 2 ||
		result.Run.OutputCount != 2 ||
		len(result.Run.Outputs) != 2 ||
		streamScratch.Outputs[0] == nil ||
		streamScratch.Outputs[1] == nil ||
		result.Run.Outputs[0] != streamScratch.Outputs[0] ||
		result.Run.Outputs[1] != streamScratch.Outputs[1] ||
		result.Run.Last.Output != streamScratch.Outputs[1] ||
		stats.TXBs == 0 ||
		stats.Residuals == 0 {
		t.Fatalf("low-overhead batch result=%+v outputs=%p/%p stats=%+v", result, streamScratch.Outputs[0], streamScratch.Outputs[1], stats)
	}

	pool.Reset()
	refs.Reset()
	state.Reset()
	stats = av1.DecoderFrameWorkTileResidualStats{}
	if err := runner.Reset(); err != nil {
		t.Fatal(err)
	}
	result = av1.DecoderFrameWorkResidualStreamResult{}
	if err := runner.RunLowOverheadsInto(&result, payloads[:], nil); err != nil {
		t.Fatal(err)
	}
	if result.EventCount != 5 ||
		result.Run.CompletedFrames != 2 ||
		result.Run.OutputCount != 2 ||
		result.Run.Outputs[0] != streamScratch.Outputs[0] ||
		result.Run.Outputs[1] != streamScratch.Outputs[1] ||
		result.Run.Last.Output != streamScratch.Outputs[1] ||
		stats.TXBs == 0 ||
		stats.Residuals == 0 {
		t.Fatalf("low-overhead into result=%+v outputs=%p/%p stats=%+v", result, streamScratch.Outputs[0], streamScratch.Outputs[1], stats)
	}
}

func TestPublicDecoderFrameWorkResidualStreamRunnerRTPPayloadAfterLoss(t *testing.T) {
	workerPool, err := av1.NewTileWorkerPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	payloads := [...][]byte{
		publicDecoderResidualRTPPayload(),
		publicDecoderResidualRTPFramePayload(),
	}
	var probeStream av1.DecoderStream
	var probeEvents [4]av1.DecoderEvent
	var probeRTPBuffer [256]byte
	var probeRTPSpans [4]av1.RTPObuSpan
	var scratchSpans [1]av1.TileSpan
	var scratchJobs [1]av1.TileJob
	var scratchBatches [1]av1.TileBatch
	plan, err := av1.DecoderFrameWorkResidualRTPPayloadsStreamPlan(probeStream, 0, payloads[:], 1, probeRTPBuffer[:], probeRTPSpans[:], probeEvents[:], scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		t.Fatal(err)
	}

	pool := publicDecoderPostFilterFramePool(t, av1.FrameFormat{
		Width:        int(plan.Bind.Event.FrameSize.CodedWidth),
		Height:       int(plan.Bind.Event.FrameSize.Height),
		BitDepth:     8,
		MonoChrome:   true,
		SubsamplingX: true,
		SubsamplingY: true,
		Align:        64,
	}, 2)
	var stream av1.DecoderStream
	var refs av1.DecoderSurfaceReferences
	var state av1.DecoderFrameWorkState
	var referenceSurfaces [av1.InterRefsPerFrame]int
	var referenceFrames [av1.InterRefsPerFrame]*av1.Frame
	var releases [av1.RefFrames]int
	var stats av1.DecoderFrameWorkTileResidualStats
	var side av1.DecoderFrameWorkSideData
	var batchRunner av1.DecoderFrameWorkBatchResidualRunner
	streamScratch := publicDecoderResidualStreamScratch(plan.Size)
	runner, _, err := av1.BindDecoderFrameWorkResidualStreamPlanRunner(plan, &stream, av1.DecoderFrameWorkResidualEventRuntime{
		State:             &state,
		Refs:              &refs,
		FramePool:         &pool,
		Align:             64,
		ReferenceSurfaces: referenceSurfaces[:],
		ReferenceFrames:   referenceFrames[:],
		Releases:          releases[:],
		WorkerPool:        workerPool,
		SideData:          &side,
		Stats:             &stats,
	}, streamScratch, &batchRunner)
	if err != nil {
		t.Fatal(err)
	}

	fragmented := publicDecoderResidualFragmentedRTPPayloads()
	if _, err := runner.RunRTPPayload(fragmented[0], nil); err != nil {
		t.Fatal(err)
	}
	firstFragment, err := runner.RunRTPPayload(fragmented[1], nil)
	if err != nil {
		t.Fatal(err)
	}
	if firstFragment.EventCount != 0 || firstFragment.RTPUsed == 0 || runner.RTPUsed == 0 || !runner.State().InRTPFragment {
		t.Fatalf("first fragment result=%+v state=%+v", firstFragment, runner.State())
	}
	if _, err := runner.RunRTPPayload(payloads[1], nil); !errors.Is(err, av1.ErrRTPFragmentInterrupted) {
		t.Fatalf("interrupted fragment err=%v want %v", err, av1.ErrRTPFragmentInterrupted)
	}
	if runner.RTPUsed == 0 || !runner.State().InRTPFragment {
		t.Fatalf("interrupted fragment did not retain state=%+v", runner.State())
	}

	recovered, err := runner.RunRTPPayloadAfterLoss(payloads[1], nil)
	if err != nil {
		t.Fatal(err)
	}
	if recovered.EventCount != 2 ||
		recovered.RTPUsed != 0 ||
		runner.RTPUsed != 0 ||
		runner.State().InRTPFragment ||
		recovered.Run.CompletedFrames != 1 ||
		recovered.Run.OutputCount != 1 ||
		recovered.Run.Last.Output == nil ||
		stats.TXBs == 0 ||
		stats.Residuals == 0 {
		t.Fatalf("recovered result=%+v state=%+v stats=%+v", recovered, runner.State(), stats)
	}

	pool.Reset()
	refs.Reset()
	state.Reset()
	stats = av1.DecoderFrameWorkTileResidualStats{}
	if err := runner.Reset(); err != nil {
		t.Fatal(err)
	}
	var result av1.DecoderFrameWorkResidualStreamResult
	if err := runner.RunRTPPayloadsInto(&result, payloads[:1], nil); err != nil {
		t.Fatal(err)
	}
	if _, err := runner.RunRTPPayload(fragmented[1], nil); err != nil {
		t.Fatal(err)
	}
	if err := runner.RunRTPPayloadAfterLossInto(&result, payloads[1], nil); err != nil {
		t.Fatal(err)
	}
	if result.EventCount != 5 ||
		result.Run.CompletedFrames != 2 ||
		result.Run.OutputCount != 2 ||
		result.Run.Outputs[0] != streamScratch.Outputs[0] ||
		result.Run.Outputs[1] != streamScratch.Outputs[1] ||
		result.Run.Last.Output != streamScratch.Outputs[1] ||
		runner.RTPUsed != 0 ||
		runner.State().InRTPFragment ||
		stats.TXBs == 0 ||
		stats.Residuals == 0 {
		t.Fatalf("recovered into result=%+v state=%+v outputs=%p/%p stats=%+v", result, runner.State(), streamScratch.Outputs[0], streamScratch.Outputs[1], stats)
	}

	var nilRunner *av1.DecoderFrameWorkResidualStreamRunner
	if _, err := nilRunner.RunRTPPayloadAfterLoss(payloads[0], nil); !errors.Is(err, av1.ErrDecoderInvalidFrameWorkState) {
		t.Fatalf("nil after-loss err=%v want %v", err, av1.ErrDecoderInvalidFrameWorkState)
	}
	var nilResult *av1.DecoderFrameWorkResidualStreamResult
	if err := runner.RunRTPPayloadAfterLossInto(nilResult, payloads[0], nil); !errors.Is(err, av1.ErrDecoderInvalidFrameWorkState) {
		t.Fatalf("nil after-loss result err=%v want %v", err, av1.ErrDecoderInvalidFrameWorkState)
	}
}

func TestPublicDecoderFrameWorkResidualStreamRunnerRTPPayloads(t *testing.T) {
	workerPool, err := av1.NewTileWorkerPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	lowOverhead := publicDecoderResidualLowOverheadStream()
	rtpPayloads := publicDecoderResidualFragmentedRTPPayloads()
	var probeStream av1.DecoderStream
	var probeEvents [4]av1.DecoderEvent
	count, err := probeStream.PushLowOverhead(lowOverhead, probeEvents[:])
	if err != nil {
		t.Fatal(err)
	}
	sequence, ok := probeStream.SequenceHeader()
	if !ok {
		t.Fatal("probe stream missing sequence")
	}
	var scratchSpans [1]av1.TileSpan
	var scratchJobs [1]av1.TileJob
	var scratchBatches [1]av1.TileBatch
	size, err := av1.DecoderFrameWorkResidualEventsScratchLen(sequence, probeEvents[:count], 1, scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		t.Fatal(err)
	}

	pool := publicDecoderPostFilterFramePool(t, av1.FrameFormat{
		Width:        int(probeEvents[count-1].FrameSize.CodedWidth),
		Height:       int(probeEvents[count-1].FrameSize.Height),
		BitDepth:     8,
		MonoChrome:   true,
		SubsamplingX: true,
		SubsamplingY: true,
		Align:        64,
	}, 1)
	var stream av1.DecoderStream
	var refs av1.DecoderSurfaceReferences
	var state av1.DecoderFrameWorkState
	var referenceSurfaces [av1.InterRefsPerFrame]int
	var referenceFrames [av1.InterRefsPerFrame]*av1.Frame
	var releases [av1.RefFrames]int
	var stats av1.DecoderFrameWorkTileResidualStats
	var side av1.DecoderFrameWorkSideData
	var batchRunner av1.DecoderFrameWorkBatchResidualRunner
	var outputs [1]*av1.Frame
	scratch := publicDecoderResidualEventScratch(size)
	eventRunner, _, err := av1.BindDecoderFrameWorkResidualEventRunner(size, sequence, probeEvents[count-1], av1.DecoderFrameWorkResidualEventRuntime{
		State:             &state,
		Refs:              &refs,
		FramePool:         &pool,
		Align:             64,
		ReferenceSurfaces: referenceSurfaces[:],
		ReferenceFrames:   referenceFrames[:],
		Releases:          releases[:],
		WorkerPool:        workerPool,
		SideData:          &side,
		Stats:             &stats,
		Outputs:           outputs[:],
	}, scratch, &batchRunner)
	if err != nil {
		t.Fatal(err)
	}
	runner := av1.DecoderFrameWorkResidualStreamRunner{
		Stream:          &stream,
		EventRunner:     eventRunner,
		Events:          make([]av1.DecoderEvent, len(probeEvents)),
		SideDataScratch: scratch.SideData,
		RTPBuffer:       make([]byte, len(lowOverhead)),
		RTPSpans:        make([]av1.RTPObuSpan, len(probeEvents)),
	}

	var nilRunner *av1.DecoderFrameWorkResidualStreamRunner
	if _, err := nilRunner.RunRTPPayloads(nil, nil); !errors.Is(err, av1.ErrDecoderInvalidFrameWorkState) {
		t.Fatalf("nil RTP payload batch runner err=%v want %v", err, av1.ErrDecoderInvalidFrameWorkState)
	}
	empty, err := runner.RunRTPPayloads(nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if empty.EventCount != 0 || empty.RTPUsed != 0 || empty.Run.Count != 0 || empty.Run.OutputCount != 0 {
		t.Fatalf("empty RTP payload batch result=%+v", empty)
	}

	postCalls := 0
	result, err := runner.RunRTPPayloads(rtpPayloads, func(ctx av1.DecoderFrameWorkPostFilterContext) error {
		postCalls++
		if !ctx.ExecutedTileWork || ctx.Step.Kind != av1.DecoderFrameWorkStepTile {
			t.Fatalf("postfilter ctx=%+v", ctx)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.EventCount != count ||
		result.RTPUsed != 0 ||
		runner.RTPUsed != 0 ||
		result.Run.Count != count ||
		result.Run.ExecutedTileWork != 1 ||
		result.Run.CompletedFrames != 1 ||
		result.Run.OutputCount != 1 ||
		len(result.Run.Outputs) != 1 ||
		result.Run.Last.Output == nil ||
		outputs[0] != result.Run.Last.Output ||
		result.Run.Outputs[0] != result.Run.Last.Output ||
		postCalls != 1 ||
		stats.TXBs == 0 ||
		stats.Residuals == 0 {
		t.Fatalf("rtp batch result=%+v retained=%d postCalls=%d stats=%+v", result, runner.RTPUsed, postCalls, stats)
	}
	if _, ok := refs.ReferenceSlot(0); !ok {
		t.Fatal("RTP payload batch runner did not publish decoded frame")
	}
}

func TestPublicDecoderFrameWorkResidualStreamRunnerRTPPayloadsMultipleOutputs(t *testing.T) {
	workerPool, err := av1.NewTileWorkerPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	payloads := [...][]byte{
		publicDecoderResidualRTPPayload(),
		publicDecoderResidualRTPFramePayload(),
	}
	var probeStream av1.DecoderStream
	var probeEvents [4]av1.DecoderEvent
	var probeRTPBuffer [256]byte
	var probeRTPSpans [4]av1.RTPObuSpan
	var scratchSpans [1]av1.TileSpan
	var scratchJobs [1]av1.TileJob
	var scratchBatches [1]av1.TileBatch
	plan, err := av1.DecoderFrameWorkResidualRTPPayloadsStreamPlan(probeStream, 0, payloads[:], 1, probeRTPBuffer[:], probeRTPSpans[:], probeEvents[:], scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		t.Fatal(err)
	}
	size := plan.Size
	if size.Events != 3 || size.RTPSpans != 3 || size.Event.Outputs != 2 {
		t.Fatalf("multi-output RTP batch scratch size=%+v", size)
	}
	if probeEvents[0].Kind != av1.DecoderEventFrameHeader || probeEvents[1].Kind != av1.DecoderEventTileGroup {
		t.Fatalf("last RTP batch events=%+v", probeEvents[:2])
	}
	if !plan.HasEvent() ||
		plan.Bind.EventIndex != 1 ||
		plan.Bind.Event.Kind != av1.DecoderEventTileGroup {
		t.Fatalf("multi-output RTP batch bind plan=%+v", plan)
	}
	sequence := plan.Bind.Sequence
	tile := plan.Bind.Event

	pool := publicDecoderPostFilterFramePool(t, av1.FrameFormat{
		Width:        int(tile.FrameSize.CodedWidth),
		Height:       int(tile.FrameSize.Height),
		BitDepth:     8,
		MonoChrome:   true,
		SubsamplingX: true,
		SubsamplingY: true,
		Align:        64,
	}, 2)
	var stream av1.DecoderStream
	var refs av1.DecoderSurfaceReferences
	var state av1.DecoderFrameWorkState
	var referenceSurfaces [av1.InterRefsPerFrame]int
	var referenceFrames [av1.InterRefsPerFrame]*av1.Frame
	var releases [av1.RefFrames]int
	var stats av1.DecoderFrameWorkTileResidualStats
	var side av1.DecoderFrameWorkSideData
	var batchRunner av1.DecoderFrameWorkBatchResidualRunner
	eventScratch := publicDecoderResidualEventScratch(size.Event)
	eventRunner, _, err := av1.BindDecoderFrameWorkResidualEventRunner(size.Event, sequence, tile, av1.DecoderFrameWorkResidualEventRuntime{
		State:             &state,
		Refs:              &refs,
		FramePool:         &pool,
		Align:             64,
		ReferenceSurfaces: referenceSurfaces[:],
		ReferenceFrames:   referenceFrames[:],
		Releases:          releases[:],
		WorkerPool:        workerPool,
		SideData:          &side,
		Stats:             &stats,
	}, eventScratch, &batchRunner)
	if err != nil {
		t.Fatal(err)
	}
	streamScratch := publicDecoderResidualStreamScratch(size)
	runner, err := av1.BindDecoderFrameWorkResidualStreamRunner(size, &stream, eventRunner, streamScratch)
	if err != nil {
		t.Fatal(err)
	}

	result, err := runner.RunRTPPayloads(payloads[:], nil)
	if err != nil {
		t.Fatal(err)
	}
	if result.EventCount != 5 ||
		result.RTPUsed != 0 ||
		runner.RTPUsed != 0 ||
		result.Run.Count != 5 ||
		result.Run.ExecutedTileWork != 2 ||
		result.Run.CompletedFrames != 2 ||
		result.Run.OutputCount != 2 ||
		len(result.Run.Outputs) != 2 ||
		result.Run.Last.Output == nil ||
		streamScratch.Outputs[0] == nil ||
		streamScratch.Outputs[1] != result.Run.Last.Output ||
		result.Run.Outputs[0] != streamScratch.Outputs[0] ||
		result.Run.Outputs[1] != result.Run.Last.Output ||
		stats.TXBs == 0 ||
		stats.Residuals == 0 {
		t.Fatalf("multi-output RTP batch result=%+v retained=%d outputs=%p/%p stats=%+v", result, runner.RTPUsed, streamScratch.Outputs[0], streamScratch.Outputs[1], stats)
	}
	if _, ok := refs.ReferenceSlot(0); !ok {
		t.Fatal("multi-output RTP payload batch runner did not publish decoded frame")
	}
}

func TestPublicDecoderFrameWorkResidualStreamRunnerRTPPayloadInto(t *testing.T) {
	workerPool, err := av1.NewTileWorkerPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	payloads := [...][]byte{
		publicDecoderResidualRTPPayload(),
		publicDecoderResidualRTPFramePayload(),
	}
	var probeStream av1.DecoderStream
	var probeEvents [4]av1.DecoderEvent
	var probeRTPBuffer [256]byte
	var probeRTPSpans [4]av1.RTPObuSpan
	var scratchSpans [1]av1.TileSpan
	var scratchJobs [1]av1.TileJob
	var scratchBatches [1]av1.TileBatch
	plan, err := av1.DecoderFrameWorkResidualRTPPayloadsStreamPlan(probeStream, 0, payloads[:], 1, probeRTPBuffer[:], probeRTPSpans[:], probeEvents[:], scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		t.Fatal(err)
	}

	pool := publicDecoderPostFilterFramePool(t, av1.FrameFormat{
		Width:        int(plan.Bind.Event.FrameSize.CodedWidth),
		Height:       int(plan.Bind.Event.FrameSize.Height),
		BitDepth:     8,
		MonoChrome:   true,
		SubsamplingX: true,
		SubsamplingY: true,
		Align:        64,
	}, 2)
	var stream av1.DecoderStream
	var refs av1.DecoderSurfaceReferences
	var state av1.DecoderFrameWorkState
	var referenceSurfaces [av1.InterRefsPerFrame]int
	var referenceFrames [av1.InterRefsPerFrame]*av1.Frame
	var releases [av1.RefFrames]int
	var stats av1.DecoderFrameWorkTileResidualStats
	var side av1.DecoderFrameWorkSideData
	var batchRunner av1.DecoderFrameWorkBatchResidualRunner
	streamScratch := publicDecoderResidualStreamScratch(plan.Size)
	runner, _, err := av1.BindDecoderFrameWorkResidualStreamPlanRunner(plan, &stream, av1.DecoderFrameWorkResidualEventRuntime{
		State:             &state,
		Refs:              &refs,
		FramePool:         &pool,
		Align:             64,
		ReferenceSurfaces: referenceSurfaces[:],
		ReferenceFrames:   referenceFrames[:],
		Releases:          releases[:],
		WorkerPool:        workerPool,
		SideData:          &side,
		Stats:             &stats,
	}, streamScratch, &batchRunner)
	if err != nil {
		t.Fatal(err)
	}

	var nilResult *av1.DecoderFrameWorkResidualStreamResult
	if err := runner.RunRTPPayloadInto(nilResult, payloads[0], nil); !errors.Is(err, av1.ErrDecoderInvalidFrameWorkState) {
		t.Fatalf("nil result err=%v want %v", err, av1.ErrDecoderInvalidFrameWorkState)
	}
	if err := runner.RunRTPPayloadsInto(nilResult, nil, nil); !errors.Is(err, av1.ErrDecoderInvalidFrameWorkState) {
		t.Fatalf("nil batch result err=%v want %v", err, av1.ErrDecoderInvalidFrameWorkState)
	}
	var result av1.DecoderFrameWorkResidualStreamResult
	if err := runner.RunRTPPayloadsInto(&result, payloads[:], nil); err != nil {
		t.Fatal(err)
	}
	if result.RTPUsed != 0 || runner.RTPUsed != 0 {
		t.Fatalf("retained result=%d runner=%d", result.RTPUsed, runner.RTPUsed)
	}
	if result.EventCount != 5 ||
		result.Run.Count != 5 ||
		result.Run.ExecutedTileWork != 2 ||
		result.Run.CompletedFrames != 2 ||
		result.Run.OutputCount != 2 ||
		len(result.Run.Outputs) != 2 ||
		streamScratch.Outputs[0] == nil ||
		streamScratch.Outputs[1] == nil ||
		result.Run.Outputs[0] != streamScratch.Outputs[0] ||
		result.Run.Outputs[1] != streamScratch.Outputs[1] ||
		result.Run.Last.Output != streamScratch.Outputs[1] ||
		stats.TXBs == 0 ||
		stats.Residuals == 0 {
		t.Fatalf("RTP payload into result=%+v outputs=%p/%p stats=%+v", result, streamScratch.Outputs[0], streamScratch.Outputs[1], stats)
	}
	if _, ok := refs.ReferenceSlot(0); !ok {
		t.Fatal("RTP payload into runner did not publish decoded frame")
	}
}

func TestPublicDecoderFrameWorkResidualStreamResultAccumulate(t *testing.T) {
	var firstFrame av1.Frame
	var secondFrame av1.Frame
	var result av1.DecoderFrameWorkResidualStreamResult
	if err := result.Accumulate(av1.DecoderFrameWorkResidualStreamResult{
		EventCount: 2,
		RTPUsed:    3,
		Run: av1.DecoderFrameWorkResidualEventsResult{
			Count:            2,
			ExecutedTileWork: 1,
			CompletedFrames:  1,
			OutputCount:      1,
			Last: av1.DecoderFrameWorkEventResult{
				Output: &firstFrame,
			},
			Stats: av1.DecoderFrameWorkTileResidualStats{
				TXBs:      1,
				Residuals: 2,
			},
		},
	}); err != nil {
		t.Fatal(err)
	}
	if err := result.Accumulate(av1.DecoderFrameWorkResidualStreamResult{
		EventCount: 1,
		RTPUsed:    0,
		Run: av1.DecoderFrameWorkResidualEventsResult{
			Count:            1,
			ExecutedTileWork: 1,
			CompletedFrames:  1,
			OutputCount:      1,
			Last: av1.DecoderFrameWorkEventResult{
				Output: &secondFrame,
			},
			Stats: av1.DecoderFrameWorkTileResidualStats{
				TXBs:      3,
				Residuals: 5,
			},
		},
	}); err != nil {
		t.Fatal(err)
	}
	outputs := [...]*av1.Frame{&firstFrame, &secondFrame}
	if err := result.BindOutputs(outputs[:]); err != nil {
		t.Fatal(err)
	}
	if result.EventCount != 3 ||
		result.RTPUsed != 0 ||
		result.Run.Count != 3 ||
		result.Run.ExecutedTileWork != 2 ||
		result.Run.CompletedFrames != 2 ||
		result.Run.OutputCount != 2 ||
		result.Run.Last.Output != &secondFrame ||
		result.Run.Outputs[0] != &firstFrame ||
		result.Run.Outputs[1] != &secondFrame ||
		result.Run.Stats.TXBs != 4 ||
		result.Run.Stats.Residuals != 7 {
		t.Fatalf("accumulated result=%+v", result)
	}
	if err := result.BindOutputs(outputs[:1]); !errors.Is(err, av1.ErrFrameShortBuffer) {
		t.Fatalf("short outputs err=%v want %v", err, av1.ErrFrameShortBuffer)
	}
	var nilStreamResult *av1.DecoderFrameWorkResidualStreamResult
	if err := nilStreamResult.Accumulate(result); !errors.Is(err, av1.ErrDecoderInvalidFrameWorkState) {
		t.Fatalf("nil stream accumulate err=%v want %v", err, av1.ErrDecoderInvalidFrameWorkState)
	}
	var nilEventsResult *av1.DecoderFrameWorkResidualEventsResult
	if err := nilEventsResult.Accumulate(result.Run); !errors.Is(err, av1.ErrDecoderInvalidFrameWorkState) {
		t.Fatalf("nil event accumulate err=%v want %v", err, av1.ErrDecoderInvalidFrameWorkState)
	}
	if err := nilEventsResult.BindOutputs(outputs[:]); !errors.Is(err, av1.ErrDecoderInvalidFrameWorkState) {
		t.Fatalf("nil event outputs err=%v want %v", err, av1.ErrDecoderInvalidFrameWorkState)
	}
}

func TestPublicBindDecoderFrameWorkResidualStreamRunner(t *testing.T) {
	workerPool, err := av1.NewTileWorkerPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	lowOverhead := publicDecoderResidualLowOverheadStream()
	rtpPayload := publicDecoderResidualRTPPayload()
	var probeStream av1.DecoderStream
	var probeEvents [4]av1.DecoderEvent
	var probeRTPBuffer [128]byte
	var probeRTPSpans [4]av1.RTPObuSpan
	var scratchSpans [1]av1.TileSpan
	var scratchJobs [1]av1.TileJob
	var scratchBatches [1]av1.TileBatch
	lowPlan, err := av1.DecoderFrameWorkResidualLowOverheadStreamPlan(probeStream, lowOverhead, 1, probeEvents[:], scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		t.Fatal(err)
	}
	rtpPlan, err := av1.DecoderFrameWorkResidualRTPPayloadStreamPlan(probeStream, 0, rtpPayload, 1, probeRTPBuffer[:], probeRTPSpans[:], probeEvents[:], scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		t.Fatal(err)
	}
	streamPlan := lowPlan.Max(rtpPlan)
	streamSize := streamPlan.Size
	if !streamPlan.HasEvent() || streamPlan.Bind.Event.Kind != av1.DecoderEventTileGroup {
		t.Fatalf("stream bind plan=%+v", streamPlan)
	}
	sequence := streamPlan.Bind.Sequence
	tile := streamPlan.Bind.Event

	pool := publicDecoderPostFilterFramePool(t, av1.FrameFormat{
		Width:        int(tile.FrameSize.CodedWidth),
		Height:       int(tile.FrameSize.Height),
		BitDepth:     8,
		MonoChrome:   true,
		SubsamplingX: true,
		SubsamplingY: true,
		Align:        64,
	}, 1)
	var refs av1.DecoderSurfaceReferences
	var state av1.DecoderFrameWorkState
	var referenceSurfaces [av1.InterRefsPerFrame]int
	var referenceFrames [av1.InterRefsPerFrame]*av1.Frame
	var releases [av1.RefFrames]int
	var stats av1.DecoderFrameWorkTileResidualStats
	var side av1.DecoderFrameWorkSideData
	var batchRunner av1.DecoderFrameWorkBatchResidualRunner
	eventScratch := publicDecoderResidualEventScratch(streamSize.Event)
	eventRunner, _, err := av1.BindDecoderFrameWorkResidualEventRunner(streamSize.Event, sequence, tile, av1.DecoderFrameWorkResidualEventRuntime{
		State:             &state,
		Refs:              &refs,
		FramePool:         &pool,
		Align:             64,
		ReferenceSurfaces: referenceSurfaces[:],
		ReferenceFrames:   referenceFrames[:],
		Releases:          releases[:],
		WorkerPool:        workerPool,
		SideData:          &side,
		Stats:             &stats,
	}, eventScratch, &batchRunner)
	if err != nil {
		t.Fatal(err)
	}

	var stream av1.DecoderStream
	streamScratch := publicDecoderResidualStreamScratch(streamSize)
	runner, err := av1.BindDecoderFrameWorkResidualStreamRunner(streamSize, &stream, eventRunner, streamScratch)
	if err != nil {
		t.Fatal(err)
	}
	if runner.Stream != &stream ||
		len(runner.Events) != streamSize.Events ||
		len(runner.RTPBuffer) != streamSize.RTPBuffer ||
		len(runner.RTPSpans) != streamSize.RTPSpans ||
		len(runner.SideDataScratch.CDEFIndexMap) != streamSize.Event.SideData.CDEFIndexMap ||
		len(runner.EventRunner.Outputs) != streamSize.Event.Outputs {
		t.Fatalf("bound runner=%+v size=%+v", runner, streamSize)
	}

	result, err := runner.RunLowOverhead(lowOverhead, nil)
	if err != nil {
		t.Fatal(err)
	}
	if result.Run.CompletedFrames != 1 || stats.TXBs == 0 {
		t.Fatalf("low-overhead result=%+v stats=%+v", result, stats)
	}

	pool.Reset()
	refs.Reset()
	state.Reset()
	if err := runner.Reset(); err != nil {
		t.Fatal(err)
	}
	result, err = runner.RunRTPPayload(rtpPayload, nil)
	if err != nil {
		t.Fatal(err)
	}
	if result.Run.CompletedFrames != 1 || runner.RTPUsed != 0 || stats.TXBs == 0 {
		t.Fatalf("rtp result=%+v retained=%d stats=%+v", result, runner.RTPUsed, stats)
	}

	if _, err := av1.BindDecoderFrameWorkResidualStreamRunner(streamSize, nil, eventRunner, streamScratch); !errors.Is(err, av1.ErrDecoderInvalidFrameWorkState) {
		t.Fatalf("nil stream err=%v want %v", err, av1.ErrDecoderInvalidFrameWorkState)
	}
	shortEvents := streamScratch
	shortEvents.Events = shortEvents.Events[:streamSize.Events-1]
	if _, err := av1.BindDecoderFrameWorkResidualStreamRunner(streamSize, &stream, eventRunner, shortEvents); !errors.Is(err, av1.ErrDecoderEventBufferTooSmall) {
		t.Fatalf("short events err=%v want %v", err, av1.ErrDecoderEventBufferTooSmall)
	}
	shortRTP := streamScratch
	shortRTP.RTPBuffer = shortRTP.RTPBuffer[:streamSize.RTPBuffer-1]
	if _, err := av1.BindDecoderFrameWorkResidualStreamRunner(streamSize, &stream, eventRunner, shortRTP); !errors.Is(err, av1.ErrRTPShortBuffer) {
		t.Fatalf("short rtp err=%v want %v", err, av1.ErrRTPShortBuffer)
	}
	shortSide := streamScratch
	shortSide.SideData.CDEFIndexMap = shortSide.SideData.CDEFIndexMap[:streamSize.Event.SideData.CDEFIndexMap-1]
	if _, err := av1.BindDecoderFrameWorkResidualStreamRunner(streamSize, &stream, eventRunner, shortSide); !errors.Is(err, av1.ErrFrameShortBuffer) {
		t.Fatalf("short side-data err=%v want %v", err, av1.ErrFrameShortBuffer)
	}
	shortOutputs := streamScratch
	shortOutputs.Outputs = shortOutputs.Outputs[:streamSize.Event.Outputs-1]
	if _, err := av1.BindDecoderFrameWorkResidualStreamRunner(streamSize, &stream, eventRunner, shortOutputs); !errors.Is(err, av1.ErrFrameShortBuffer) {
		t.Fatalf("short output err=%v want %v", err, av1.ErrFrameShortBuffer)
	}
}

func TestPublicBindDecoderFrameWorkResidualStreamEventRunner(t *testing.T) {
	workerPool, err := av1.NewTileWorkerPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	payloads := [...][]byte{
		publicDecoderResidualRTPPayload(),
		publicDecoderResidualRTPFramePayload(),
	}
	var probeStream av1.DecoderStream
	var probeEvents [4]av1.DecoderEvent
	var probeRTPBuffer [256]byte
	var probeRTPSpans [4]av1.RTPObuSpan
	var scratchSpans [1]av1.TileSpan
	var scratchJobs [1]av1.TileJob
	var scratchBatches [1]av1.TileBatch
	plan, err := av1.DecoderFrameWorkResidualRTPPayloadsStreamPlan(probeStream, 0, payloads[:], 1, probeRTPBuffer[:], probeRTPSpans[:], probeEvents[:], scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		t.Fatal(err)
	}
	size := plan.Size
	if !plan.HasEvent() ||
		plan.Bind.EventIndex != 1 ||
		plan.Bind.Event.Kind != av1.DecoderEventTileGroup {
		t.Fatalf("stream event bind plan=%+v", plan)
	}
	sequence := plan.Bind.Sequence
	event := plan.Bind.Event

	pool := publicDecoderPostFilterFramePool(t, av1.FrameFormat{
		Width:        int(event.FrameSize.CodedWidth),
		Height:       int(event.FrameSize.Height),
		BitDepth:     8,
		MonoChrome:   true,
		SubsamplingX: true,
		SubsamplingY: true,
		Align:        64,
	}, 2)
	var stream av1.DecoderStream
	var refs av1.DecoderSurfaceReferences
	var state av1.DecoderFrameWorkState
	var referenceSurfaces [av1.InterRefsPerFrame]int
	var referenceFrames [av1.InterRefsPerFrame]*av1.Frame
	var releases [av1.RefFrames]int
	var stats av1.DecoderFrameWorkTileResidualStats
	var side av1.DecoderFrameWorkSideData
	var batchRunner av1.DecoderFrameWorkBatchResidualRunner
	scratch := publicDecoderResidualStreamScratch(size)
	runner, boundSide, err := av1.BindDecoderFrameWorkResidualStreamEventRunner(size, &stream, sequence, event, av1.DecoderFrameWorkResidualEventRuntime{
		State:             &state,
		Refs:              &refs,
		FramePool:         &pool,
		Align:             64,
		ReferenceSurfaces: referenceSurfaces[:],
		ReferenceFrames:   referenceFrames[:],
		Releases:          releases[:],
		WorkerPool:        workerPool,
		SideData:          &side,
		Stats:             &stats,
	}, scratch, &batchRunner)
	if err != nil {
		t.Fatal(err)
	}
	if runner.Stream != &stream ||
		runner.EventRunner.BatchRunner != &batchRunner ||
		len(runner.EventRunner.Spans) != size.Event.Plan.SpanCount ||
		len(runner.SideDataScratch.CDEFIndexMap) != size.Event.SideData.CDEFIndexMap ||
		len(runner.EventRunner.Outputs) != size.Event.Outputs ||
		len(boundSide.CDEFIndexMap.Index) == 0 ||
		len(boundSide.LoopFilterMap.Records) == 0 {
		t.Fatalf("combined runner=%+v side=%+v size=%+v", runner, boundSide, size)
	}

	result, err := runner.RunRTPPayloads(payloads[:], nil)
	if err != nil {
		t.Fatal(err)
	}
	if result.EventCount != 5 ||
		result.Run.CompletedFrames != 2 ||
		result.Run.OutputCount != 2 ||
		len(result.Run.Outputs) != 2 ||
		result.Run.Outputs[0] == nil ||
		result.Run.Outputs[1] != result.Run.Last.Output ||
		stats.TXBs == 0 ||
		stats.Residuals == 0 {
		t.Fatalf("combined run result=%+v stats=%+v", result, stats)
	}

	if _, _, err := av1.BindDecoderFrameWorkResidualStreamEventRunner(size, nil, sequence, event, av1.DecoderFrameWorkResidualEventRuntime{}, scratch, &batchRunner); !errors.Is(err, av1.ErrDecoderInvalidFrameWorkState) {
		t.Fatalf("nil stream err=%v want %v", err, av1.ErrDecoderInvalidFrameWorkState)
	}
	var untouchedBatch av1.DecoderFrameWorkBatchResidualRunner
	shortEvents := scratch
	shortEvents.Events = shortEvents.Events[:size.Events-1]
	if _, _, err := av1.BindDecoderFrameWorkResidualStreamEventRunner(size, &stream, sequence, event, av1.DecoderFrameWorkResidualEventRuntime{}, shortEvents, &untouchedBatch); !errors.Is(err, av1.ErrDecoderEventBufferTooSmall) {
		t.Fatalf("short stream event scratch err=%v want %v", err, av1.ErrDecoderEventBufferTooSmall)
	}
	if untouchedBatch.UseDefaultPrediction || len(untouchedBatch.States) != 0 {
		t.Fatalf("short stream scratch mutated batch runner: %+v", untouchedBatch)
	}
	shortEvent := scratch
	shortEvent.Event.Spans = shortEvent.Event.Spans[:size.Event.Plan.SpanCount-1]
	if _, _, err := av1.BindDecoderFrameWorkResidualStreamEventRunner(size, &stream, sequence, event, av1.DecoderFrameWorkResidualEventRuntime{}, shortEvent, &batchRunner); !errors.Is(err, av1.ErrFrameShortBuffer) {
		t.Fatalf("short event scratch err=%v want %v", err, av1.ErrFrameShortBuffer)
	}
	if _, _, err := av1.BindDecoderFrameWorkResidualStreamEventRunner(size, &stream, sequence, event, av1.DecoderFrameWorkResidualEventRuntime{}, scratch, nil); !errors.Is(err, av1.ErrThreadingInvalidBatch) {
		t.Fatalf("nil batch runner err=%v want %v", err, av1.ErrThreadingInvalidBatch)
	}
}

func TestPublicBindDecoderFrameWorkResidualStreamPlanRunner(t *testing.T) {
	workerPool, err := av1.NewTileWorkerPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	payloads := [...][]byte{
		publicDecoderResidualRTPPayload(),
		publicDecoderResidualRTPFramePayload(),
	}
	var probeStream av1.DecoderStream
	var probeEvents [4]av1.DecoderEvent
	var probeRTPBuffer [256]byte
	var probeRTPSpans [4]av1.RTPObuSpan
	var scratchSpans [1]av1.TileSpan
	var scratchJobs [1]av1.TileJob
	var scratchBatches [1]av1.TileBatch
	plan, err := av1.DecoderFrameWorkResidualRTPPayloadsStreamPlan(probeStream, 0, payloads[:], 1, probeRTPBuffer[:], probeRTPSpans[:], probeEvents[:], scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		t.Fatal(err)
	}
	if !plan.HasEvent() || plan.Bind.Event.Kind != av1.DecoderEventTileGroup || plan.Size.Event.Outputs != 2 {
		t.Fatalf("stream plan=%+v", plan)
	}

	pool := publicDecoderPostFilterFramePool(t, av1.FrameFormat{
		Width:        int(plan.Bind.Event.FrameSize.CodedWidth),
		Height:       int(plan.Bind.Event.FrameSize.Height),
		BitDepth:     8,
		MonoChrome:   true,
		SubsamplingX: true,
		SubsamplingY: true,
		Align:        64,
	}, 2)
	var stream av1.DecoderStream
	var refs av1.DecoderSurfaceReferences
	var state av1.DecoderFrameWorkState
	var referenceSurfaces [av1.InterRefsPerFrame]int
	var referenceFrames [av1.InterRefsPerFrame]*av1.Frame
	var releases [av1.RefFrames]int
	var stats av1.DecoderFrameWorkTileResidualStats
	var side av1.DecoderFrameWorkSideData
	var batchRunner av1.DecoderFrameWorkBatchResidualRunner
	scratch := publicDecoderResidualStreamScratch(plan.Size)
	runner, boundSide, err := av1.BindDecoderFrameWorkResidualStreamPlanRunner(plan, &stream, av1.DecoderFrameWorkResidualEventRuntime{
		State:             &state,
		Refs:              &refs,
		FramePool:         &pool,
		Align:             64,
		ReferenceSurfaces: referenceSurfaces[:],
		ReferenceFrames:   referenceFrames[:],
		Releases:          releases[:],
		WorkerPool:        workerPool,
		SideData:          &side,
		Stats:             &stats,
	}, scratch, &batchRunner)
	if err != nil {
		t.Fatal(err)
	}
	if runner.Stream != &stream ||
		runner.EventRunner.BatchRunner != &batchRunner ||
		len(runner.Events) != plan.Size.Events ||
		len(runner.EventRunner.Outputs) != plan.Size.Event.Outputs ||
		len(boundSide.CDEFIndexMap.Index) == 0 ||
		len(boundSide.LoopFilterMap.Records) == 0 {
		t.Fatalf("plan-bound runner=%+v side=%+v plan=%+v", runner, boundSide, plan)
	}

	result, err := runner.RunRTPPayloads(payloads[:], nil)
	if err != nil {
		t.Fatal(err)
	}
	if result.EventCount != 5 ||
		result.Run.CompletedFrames != 2 ||
		result.Run.OutputCount != 2 ||
		result.Run.Outputs[0] == nil ||
		result.Run.Outputs[1] != result.Run.Last.Output ||
		stats.TXBs == 0 ||
		stats.Residuals == 0 {
		t.Fatalf("plan-bound result=%+v stats=%+v", result, stats)
	}

	if _, _, err := av1.BindDecoderFrameWorkResidualStreamPlanRunner(plan, nil, av1.DecoderFrameWorkResidualEventRuntime{}, scratch, &batchRunner); !errors.Is(err, av1.ErrDecoderInvalidFrameWorkState) {
		t.Fatalf("nil stream plan bind err=%v want %v", err, av1.ErrDecoderInvalidFrameWorkState)
	}
}

func TestPublicDecoderFrameWorkResidualStreamPlanRunnerSupportedPostFilter(t *testing.T) {
	workerPool, err := av1.NewTileWorkerPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	payload := publicDecoderResidualRTPPayload()
	var probeStream av1.DecoderStream
	var probeEvents [4]av1.DecoderEvent
	var probeRTPBuffer [128]byte
	var probeRTPSpans [4]av1.RTPObuSpan
	var scratchSpans [1]av1.TileSpan
	var scratchJobs [1]av1.TileJob
	var scratchBatches [1]av1.TileBatch
	plan, err := av1.DecoderFrameWorkResidualRTPPayloadStreamPlan(probeStream, 0, payload, 1, probeRTPBuffer[:], probeRTPSpans[:], probeEvents[:], scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		t.Fatal(err)
	}
	if !plan.HasEvent() || plan.Bind.Event.Kind != av1.DecoderEventTileGroup || plan.Size.Event.Outputs != 1 {
		t.Fatalf("stream plan=%+v", plan)
	}

	pool := publicDecoderPostFilterFramePool(t, av1.FrameFormat{
		Width:        int(plan.Bind.Event.FrameSize.CodedWidth),
		Height:       int(plan.Bind.Event.FrameSize.Height),
		BitDepth:     8,
		MonoChrome:   true,
		SubsamplingX: true,
		SubsamplingY: true,
		Align:        64,
	}, 1)
	var stream av1.DecoderStream
	var refs av1.DecoderSurfaceReferences
	var state av1.DecoderFrameWorkState
	var referenceSurfaces [av1.InterRefsPerFrame]int
	var referenceFrames [av1.InterRefsPerFrame]*av1.Frame
	var releases [av1.RefFrames]int
	var stats av1.DecoderFrameWorkTileResidualStats
	var side av1.DecoderFrameWorkSideData
	var batchRunner av1.DecoderFrameWorkBatchResidualRunner
	scratch := publicDecoderResidualStreamScratch(plan.Size)
	runner, boundSide, err := av1.BindDecoderFrameWorkResidualStreamPlanRunner(plan, &stream, av1.DecoderFrameWorkResidualEventRuntime{
		State:             &state,
		Refs:              &refs,
		FramePool:         &pool,
		Align:             64,
		ReferenceSurfaces: referenceSurfaces[:],
		ReferenceFrames:   referenceFrames[:],
		Releases:          releases[:],
		WorkerPool:        workerPool,
		SideData:          &side,
		Stats:             &stats,
	}, scratch, &batchRunner)
	if err != nil {
		t.Fatal(err)
	}

	var probeOutput av1.Frame
	probeCtx, err := av1.DecoderFrameWorkPostFilterScratchContext(plan.Bind.Sequence, plan.Bind.Event, 64, &boundSide, &probeOutput)
	if err != nil {
		t.Fatal(err)
	}
	var probe av1.DecoderFrameWorkSupportedPostFilterScratchRunner
	exact, err := probe.ScratchLen(probeCtx)
	if err != nil {
		t.Fatal(err)
	}
	postRunner := av1.DecoderFrameWorkSupportedPostFilterScratchRunner{
		Scratch: publicDecoderPostFilterRequestScratch(av1.DecoderFrameWorkPostFilterRequestScratchLen(exact)),
	}

	result, err := runner.RunRTPPayloadWithPostFilterRunner(payload, &postRunner)
	if err != nil {
		t.Fatal(err)
	}
	cdefRead := false
	for _, read := range side.CDEFIndexMap.Read {
		cdefRead = cdefRead || read
	}
	if result.EventCount != 3 ||
		result.RTPUsed != 0 ||
		result.Run.CompletedFrames != 1 ||
		result.Run.OutputCount != 1 ||
		result.Run.Outputs[0] != result.Run.Last.Output ||
		result.Run.Last.Output == nil ||
		postRunner.Result.Completed != 0 ||
		postRunner.Context.RemainingPostFilters() != 0 ||
		postRunner.Context.Output != result.Run.Last.Output ||
		!cdefRead ||
		stats.TXBs == 0 ||
		stats.Residuals == 0 {
		t.Fatalf("postfilter stream result=%+v post=%+v stats=%+v cdefRead=%v", result, postRunner, stats, cdefRead)
	}
	if _, ok := refs.ReferenceSlot(0); !ok {
		t.Fatal("postfilter stream runner did not publish decoded frame")
	}
}

func TestPublicDecoderFrameWorkResidualStreamScratchLenLowOverhead(t *testing.T) {
	payload := publicDecoderResidualLowOverheadStream()
	var stream av1.DecoderStream
	var events [4]av1.DecoderEvent
	var scratchSpans [1]av1.TileSpan
	var scratchJobs [1]av1.TileJob
	var scratchBatches [1]av1.TileBatch
	size, err := av1.DecoderFrameWorkResidualLowOverheadStreamScratchLen(stream, payload, 1, events[:], scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		t.Fatal(err)
	}
	if size.Events != 3 ||
		size.RTPBuffer != 0 ||
		size.RTPSpans != 0 ||
		size.Event.Plan != (av1.DecoderTileWorkPlan{SpanCount: 1, JobCount: 1, BatchCount: 1}) ||
		size.Event.Outputs != 1 ||
		size.Event.Runner.Workers != 1 ||
		size.Event.SideData.CDEFIndexMap == 0 ||
		size.Event.SideData.LoopFilterMap == 0 {
		t.Fatalf("low-overhead stream scratch size=%+v", size)
	}
	if stream.HasSequenceHeader() {
		t.Fatal("low-overhead stream scratch sizing mutated caller stream")
	}
	if events[0].Kind != av1.DecoderEventSequenceHeader ||
		events[1].Kind != av1.DecoderEventFrameHeader ||
		events[2].Kind != av1.DecoderEventTileGroup {
		t.Fatalf("events=%+v", events[:3])
	}

	short, err := av1.DecoderFrameWorkResidualLowOverheadStreamScratchLen(stream, payload, 1, events[:2], scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if !errors.Is(err, av1.ErrDecoderEventBufferTooSmall) {
		t.Fatalf("short events err=%v want %v", err, av1.ErrDecoderEventBufferTooSmall)
	}
	if short.Events != 3 || short.Event != (av1.DecoderFrameWorkResidualEventScratchSize{}) {
		t.Fatalf("short low-overhead scratch size=%+v", short)
	}
}

func TestPublicDecoderFrameWorkResidualStreamScratchLenRTPPayload(t *testing.T) {
	payload := publicDecoderResidualRTPPayload()
	var stream av1.DecoderStream
	var events [4]av1.DecoderEvent
	var rtpBuffer [128]byte
	var rtpSpans [4]av1.RTPObuSpan
	var scratchSpans [1]av1.TileSpan
	var scratchJobs [1]av1.TileJob
	var scratchBatches [1]av1.TileBatch
	size, err := av1.DecoderFrameWorkResidualRTPPayloadStreamScratchLen(stream, 0, payload, 1, rtpBuffer[:], rtpSpans[:], events[:], scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		t.Fatal(err)
	}
	if size.Events != 3 ||
		size.RTPBuffer == 0 ||
		size.RTPSpans != 3 ||
		size.Event.Plan != (av1.DecoderTileWorkPlan{SpanCount: 1, JobCount: 1, BatchCount: 1}) ||
		size.Event.Outputs != 1 ||
		size.Event.Runner.Workers != 1 ||
		size.Event.SideData.CDEFIndexMap == 0 ||
		size.Event.SideData.LoopFilterMap == 0 {
		t.Fatalf("rtp stream scratch size=%+v", size)
	}
	if stream.HasSequenceHeader() || stream.InRTPFragment() {
		t.Fatal("RTP stream scratch sizing mutated caller stream")
	}
	if events[0].Kind != av1.DecoderEventSequenceHeader ||
		events[1].Kind != av1.DecoderEventFrameHeader ||
		events[2].Kind != av1.DecoderEventTileGroup ||
		rtpSpans[0].Length == 0 ||
		rtpSpans[2].Length == 0 {
		t.Fatalf("events=%+v spans=%+v", events[:3], rtpSpans[:3])
	}

	shortRTP, err := av1.DecoderFrameWorkResidualRTPPayloadStreamScratchLen(stream, 0, payload, 1, rtpBuffer[:size.RTPBuffer-1], rtpSpans[:], events[:], scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if !errors.Is(err, av1.ErrRTPShortBuffer) {
		t.Fatalf("short rtp buffer err=%v want %v", err, av1.ErrRTPShortBuffer)
	}
	if shortRTP.Events != 3 || shortRTP.RTPBuffer != size.RTPBuffer || shortRTP.RTPSpans != 3 {
		t.Fatalf("short rtp scratch size=%+v want buffer %d", shortRTP, size.RTPBuffer)
	}

	shortEvents, err := av1.DecoderFrameWorkResidualRTPPayloadStreamScratchLen(stream, 0, payload, 1, rtpBuffer[:], rtpSpans[:], events[:2], scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if !errors.Is(err, av1.ErrDecoderEventBufferTooSmall) {
		t.Fatalf("short events err=%v want %v", err, av1.ErrDecoderEventBufferTooSmall)
	}
	if shortEvents.Events != 3 || shortEvents.RTPBuffer != size.RTPBuffer || shortEvents.RTPSpans != 3 {
		t.Fatalf("short events scratch size=%+v want buffer %d", shortEvents, size.RTPBuffer)
	}
}

func TestPublicDecoderFrameWorkResidualStreamScratchLenRTPPayloads(t *testing.T) {
	payload := publicDecoderResidualRTPPayload()
	payloads := publicDecoderResidualFragmentedRTPPayloads()
	var stream av1.DecoderStream
	var events [4]av1.DecoderEvent
	var rtpBuffer [128]byte
	var rtpSpans [4]av1.RTPObuSpan
	var scratchSpans [1]av1.TileSpan
	var scratchJobs [1]av1.TileJob
	var scratchBatches [1]av1.TileBatch
	single, err := av1.DecoderFrameWorkResidualRTPPayloadStreamScratchLen(stream, 0, payload, 1, rtpBuffer[:], rtpSpans[:], events[:], scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		t.Fatal(err)
	}
	size, err := av1.DecoderFrameWorkResidualRTPPayloadsStreamScratchLen(stream, 0, payloads, 1, rtpBuffer[:], rtpSpans[:], events[:], scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		t.Fatal(err)
	}
	if size.Events != 1 ||
		size.RTPBuffer == 0 ||
		size.RTPSpans != 1 ||
		size.Event != single.Event {
		t.Fatalf("rtp payload batch scratch size=%+v single=%+v", size, single)
	}
	if stream.HasSequenceHeader() || stream.InRTPFragment() {
		t.Fatal("RTP payload batch scratch sizing mutated caller stream")
	}
	if events[0].Kind != av1.DecoderEventTileGroup {
		t.Fatalf("last sized event=%+v", events[0])
	}

	shortRTP, err := av1.DecoderFrameWorkResidualRTPPayloadsStreamScratchLen(stream, 0, payloads, 1, rtpBuffer[:size.RTPBuffer-1], rtpSpans[:], events[:], scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if !errors.Is(err, av1.ErrRTPShortBuffer) {
		t.Fatalf("short rtp buffer err=%v want %v", err, av1.ErrRTPShortBuffer)
	}
	if shortRTP.RTPBuffer != size.RTPBuffer || shortRTP.Events != 1 || shortRTP.RTPSpans != 1 {
		t.Fatalf("short rtp payload batch scratch size=%+v want buffer %d", shortRTP, size.RTPBuffer)
	}

	shortEvents, err := av1.DecoderFrameWorkResidualRTPPayloadsStreamScratchLen(stream, 0, payloads, 1, rtpBuffer[:], rtpSpans[:], events[:0], scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if !errors.Is(err, av1.ErrDecoderEventBufferTooSmall) {
		t.Fatalf("short events err=%v want %v", err, av1.ErrDecoderEventBufferTooSmall)
	}
	if shortEvents.Events != 1 || shortEvents.RTPBuffer == 0 || shortEvents.RTPSpans != 1 {
		t.Fatalf("short events payload batch scratch size=%+v", shortEvents)
	}
}

func TestPublicDecoderFrameWorkResidualStreamScratchLenRTPPayloadsMultipleOutputs(t *testing.T) {
	payloads := [...][]byte{
		publicDecoderResidualRTPPayload(),
		publicDecoderResidualRTPFramePayload(),
	}
	var stream av1.DecoderStream
	var events [4]av1.DecoderEvent
	var rtpBuffer [256]byte
	var rtpSpans [4]av1.RTPObuSpan
	var scratchSpans [1]av1.TileSpan
	var scratchJobs [1]av1.TileJob
	var scratchBatches [1]av1.TileBatch
	size, err := av1.DecoderFrameWorkResidualRTPPayloadsStreamScratchLen(stream, 0, payloads[:], 1, rtpBuffer[:], rtpSpans[:], events[:], scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		t.Fatal(err)
	}
	if size.Events != 3 ||
		size.RTPBuffer == 0 ||
		size.RTPSpans != 3 ||
		size.Event.Plan != (av1.DecoderTileWorkPlan{SpanCount: 1, JobCount: 1, BatchCount: 1}) ||
		size.Event.Outputs != 2 ||
		size.Event.Runner.Workers != 1 ||
		size.Event.SideData.CDEFIndexMap == 0 ||
		size.Event.SideData.LoopFilterMap == 0 {
		t.Fatalf("multi-output RTP payload batch scratch size=%+v", size)
	}
	if stream.HasSequenceHeader() || stream.InRTPFragment() {
		t.Fatal("multi-output RTP payload batch scratch sizing mutated caller stream")
	}
	if events[0].Kind != av1.DecoderEventFrameHeader || events[1].Kind != av1.DecoderEventTileGroup {
		t.Fatalf("last multi-output RTP batch events=%+v", events[:2])
	}
}

func TestPublicDecoderFrameWorkResidualStreamScratchLenRTPFragment(t *testing.T) {
	var stream av1.DecoderStream
	var events [2]av1.DecoderEvent
	var rtpBuffer [128]byte
	var rtpSpans [2]av1.RTPObuSpan

	sequenceElement := [...]av1.RTPElement{
		{Data: publicDecoderResidualRTPElement(av1.OBUSequenceHeader, publicDecoderResidualSequenceHeaderPayload())},
	}
	var sequencePayload [128]byte
	n, err := av1.PutRTPPayload(sequencePayload[:], av1.RTPAggregationHeader{
		ElementCount:                1,
		StartsNewCodedVideoSequence: true,
	}, sequenceElement[:])
	if err != nil {
		t.Fatal(err)
	}
	used, count, err := stream.PushRTPPayload(rtpBuffer[:], 0, rtpSpans[:], events[:], sequencePayload[:n])
	if err != nil {
		t.Fatal(err)
	}
	if used == 0 || count != 1 || events[0].Kind != av1.DecoderEventSequenceHeader || !stream.HasSequenceHeader() {
		t.Fatalf("sequence used=%d count=%d event=%+v hasSequence=%v", used, count, events[0], stream.HasSequenceHeader())
	}

	frameHeader := publicDecoderResidualRTPElement(av1.OBUFrameHeader, publicDecoderResidualFrameHeaderPayload())
	var packet [8]byte
	n, next, more, err := av1.PutRTPFragment(packet[:], frameHeader, 0, 2, false)
	if err != nil {
		t.Fatal(err)
	}
	if !more {
		t.Fatal("expected frame header to need a second fragment")
	}
	used = 0
	var scratchSpans [1]av1.TileSpan
	var scratchJobs [1]av1.TileJob
	var scratchBatches [1]av1.TileBatch
	first, err := av1.DecoderFrameWorkResidualRTPPayloadStreamScratchLen(stream, used, packet[:n], 1, rtpBuffer[:], rtpSpans[:], events[:], scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		t.Fatal(err)
	}
	if first.Events != 0 || first.RTPBuffer == 0 || first.RTPSpans != 0 || first.Event != (av1.DecoderFrameWorkResidualEventScratchSize{}) {
		t.Fatalf("first fragment scratch size=%+v", first)
	}
	if stream.InRTPFragment() {
		t.Fatal("first fragment sizing mutated stream RTP fragment state")
	}
	used, count, err = stream.PushRTPPayload(rtpBuffer[:], used, rtpSpans[:], events[:], packet[:n])
	if err != nil {
		t.Fatal(err)
	}
	if count != 0 || !stream.InRTPFragment() {
		t.Fatalf("first fragment push used=%d count=%d inFragment=%v", used, count, stream.InRTPFragment())
	}

	n, _, more, err = av1.PutRTPFragment(packet[:], frameHeader, next, len(packet), false)
	if err != nil {
		t.Fatal(err)
	}
	if more {
		t.Fatal("unexpected third fragment")
	}
	final, err := av1.DecoderFrameWorkResidualRTPPayloadStreamScratchLen(stream, used, packet[:n], 1, rtpBuffer[:], rtpSpans[:], events[:], scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		t.Fatal(err)
	}
	if final.Events != 1 ||
		final.RTPBuffer != len(frameHeader) ||
		final.RTPSpans != 1 ||
		final.Event.Plan != (av1.DecoderTileWorkPlan{}) ||
		final.Event.SideData.CDEFIndexMap == 0 ||
		events[0].Kind != av1.DecoderEventFrameHeader {
		t.Fatalf("final fragment scratch size=%+v event=%+v", final, events[0])
	}
	if !stream.InRTPFragment() {
		t.Fatal("final fragment sizing mutated live stream RTP state")
	}
}

func TestPublicDecoderFrameWorkResidualStreamRunnerAllocs(t *testing.T) {
	workerPool, err := av1.NewTileWorkerPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	lowOverhead := publicDecoderResidualLowOverheadStream()
	lowOverheads := [...][]byte{
		publicDecoderResidualLowOverheadStream(),
		publicDecoderResidualLowOverheadFrameStream(),
	}
	rtpPayload := publicDecoderResidualRTPPayload()
	rtpPayloads := publicDecoderResidualFragmentedRTPPayloads()
	var probeStream av1.DecoderStream
	var probeEvents [4]av1.DecoderEvent
	count, err := probeStream.PushLowOverhead(lowOverhead, probeEvents[:])
	if err != nil {
		t.Fatal(err)
	}
	sequence, ok := probeStream.SequenceHeader()
	if !ok {
		t.Fatal("probe stream missing sequence")
	}
	var scratchSpans [1]av1.TileSpan
	var scratchJobs [1]av1.TileJob
	var scratchBatches [1]av1.TileBatch
	size, err := av1.DecoderFrameWorkResidualEventsScratchLen(sequence, probeEvents[:count], 1, scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		t.Fatal(err)
	}
	pool := publicDecoderPostFilterFramePool(t, av1.FrameFormat{
		Width:        int(probeEvents[count-1].FrameSize.CodedWidth),
		Height:       int(probeEvents[count-1].FrameSize.Height),
		BitDepth:     8,
		MonoChrome:   true,
		SubsamplingX: true,
		SubsamplingY: true,
		Align:        64,
	}, 2)
	var stream av1.DecoderStream
	var refs av1.DecoderSurfaceReferences
	var state av1.DecoderFrameWorkState
	var referenceSurfaces [av1.InterRefsPerFrame]int
	var referenceFrames [av1.InterRefsPerFrame]*av1.Frame
	var releases [av1.RefFrames]int
	var stats av1.DecoderFrameWorkTileResidualStats
	var side av1.DecoderFrameWorkSideData
	var batchRunner av1.DecoderFrameWorkBatchResidualRunner
	var outputs [2]*av1.Frame
	scratch := publicDecoderResidualEventScratch(size)
	eventRunner, _, err := av1.BindDecoderFrameWorkResidualEventRunner(size, sequence, probeEvents[count-1], av1.DecoderFrameWorkResidualEventRuntime{
		State:             &state,
		Refs:              &refs,
		FramePool:         &pool,
		Align:             64,
		ReferenceSurfaces: referenceSurfaces[:],
		ReferenceFrames:   referenceFrames[:],
		Releases:          releases[:],
		WorkerPool:        workerPool,
		SideData:          &side,
		Stats:             &stats,
		Outputs:           outputs[:],
	}, scratch, &batchRunner)
	if err != nil {
		t.Fatal(err)
	}
	var events [4]av1.DecoderEvent
	var rtpBuffer [128]byte
	var rtpSpans [4]av1.RTPObuSpan
	runner := av1.DecoderFrameWorkResidualStreamRunner{
		Stream:          &stream,
		EventRunner:     eventRunner,
		Events:          events[:],
		SideDataScratch: scratch.SideData,
		RTPBuffer:       rtpBuffer[:],
		RTPSpans:        rtpSpans[:],
	}

	allocs := testing.AllocsPerRun(1000, func() {
		pool.Reset()
		refs.Reset()
		state.Reset()
		if resetErr := runner.Reset(); resetErr != nil {
			err = resetErr
			return
		}
		result, runErr := runner.RunLowOverhead(lowOverhead, nil)
		if runErr != nil {
			err = runErr
			return
		}
		if result.Run.CompletedFrames != 1 || result.Run.OutputCount != 1 || outputs[0] != result.Run.Last.Output || stats.TXBs == 0 {
			err = av1.ErrThreadingInvalidBatch
			return
		}

		pool.Reset()
		refs.Reset()
		state.Reset()
		if resetErr := runner.Reset(); resetErr != nil {
			err = resetErr
			return
		}
		result, runErr = runner.RunLowOverheads(lowOverheads[:], nil)
		if runErr != nil {
			err = runErr
			return
		}
		if result.Run.CompletedFrames != 2 || result.Run.OutputCount != 2 || outputs[1] != result.Run.Last.Output || stats.TXBs == 0 {
			err = av1.ErrThreadingInvalidBatch
			return
		}

		pool.Reset()
		refs.Reset()
		state.Reset()
		if resetErr := runner.Reset(); resetErr != nil {
			err = resetErr
			return
		}
		result = av1.DecoderFrameWorkResidualStreamResult{}
		if runErr = runner.RunLowOverheadsInto(&result, lowOverheads[:], nil); runErr != nil {
			err = runErr
			return
		}
		if result.Run.CompletedFrames != 2 || result.Run.OutputCount != 2 || outputs[1] != result.Run.Last.Output || stats.TXBs == 0 {
			err = av1.ErrThreadingInvalidBatch
			return
		}

		pool.Reset()
		refs.Reset()
		state.Reset()
		if resetErr := runner.Reset(); resetErr != nil {
			err = resetErr
			return
		}
		result, runErr = runner.RunRTPPayload(rtpPayload, nil)
		if runErr != nil {
			err = runErr
			return
		}
		if result.Run.CompletedFrames != 1 || result.Run.OutputCount != 1 || outputs[0] != result.Run.Last.Output || stats.TXBs == 0 || runner.RTPUsed != 0 {
			err = av1.ErrThreadingInvalidBatch
		}

		pool.Reset()
		refs.Reset()
		state.Reset()
		if resetErr := runner.Reset(); resetErr != nil {
			err = resetErr
			return
		}
		result, runErr = runner.RunRTPPayloads(rtpPayloads, nil)
		if runErr != nil {
			err = runErr
			return
		}
		if result.Run.CompletedFrames != 1 || result.Run.OutputCount != 1 || outputs[0] != result.Run.Last.Output || stats.TXBs == 0 || runner.RTPUsed != 0 {
			err = av1.ErrThreadingInvalidBatch
			return
		}

		pool.Reset()
		refs.Reset()
		state.Reset()
		if resetErr := runner.Reset(); resetErr != nil {
			err = resetErr
			return
		}
		result = av1.DecoderFrameWorkResidualStreamResult{}
		if runErr = runner.RunRTPPayloadsInto(&result, rtpPayloads, nil); runErr != nil {
			err = runErr
			return
		}
		if result.Run.CompletedFrames != 1 || result.Run.OutputCount != 1 || outputs[0] != result.Run.Last.Output || stats.TXBs == 0 || runner.RTPUsed != 0 {
			err = av1.ErrThreadingInvalidBatch
			return
		}

		pool.Reset()
		refs.Reset()
		state.Reset()
		if resetErr := runner.Reset(); resetErr != nil {
			err = resetErr
			return
		}
		result = av1.DecoderFrameWorkResidualStreamResult{}
		if runErr = runner.RunRTPPayloadsInto(&result, rtpPayloads[:2], nil); runErr != nil {
			err = runErr
			return
		}
		if runner.RTPUsed == 0 {
			err = av1.ErrThreadingInvalidBatch
			return
		}
		if runErr = runner.RunRTPPayloadAfterLossInto(&result, rtpPayload, nil); runErr != nil {
			err = runErr
			return
		}
		if result.Run.CompletedFrames != 1 || result.Run.OutputCount != 1 || outputs[0] != result.Run.Last.Output || stats.TXBs == 0 || runner.RTPUsed != 0 {
			err = av1.ErrThreadingInvalidBatch
		}
	})
	if err != nil {
		t.Fatal(err)
	}
	if allocs != 0 {
		t.Fatalf("DecoderFrameWorkResidualStreamRunner allocated: %f", allocs)
	}
}

func TestPublicDecoderFrameWorkResidualStreamPostFilterRunnerAllocs(t *testing.T) {
	workerPool, err := av1.NewTileWorkerPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	payload := publicDecoderResidualRTPPayload()
	var probeStream av1.DecoderStream
	var probeEvents [4]av1.DecoderEvent
	var probeRTPBuffer [128]byte
	var probeRTPSpans [4]av1.RTPObuSpan
	var scratchSpans [1]av1.TileSpan
	var scratchJobs [1]av1.TileJob
	var scratchBatches [1]av1.TileBatch
	plan, err := av1.DecoderFrameWorkResidualRTPPayloadStreamPlan(probeStream, 0, payload, 1, probeRTPBuffer[:], probeRTPSpans[:], probeEvents[:], scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		t.Fatal(err)
	}

	pool := publicDecoderPostFilterFramePool(t, av1.FrameFormat{
		Width:        int(plan.Bind.Event.FrameSize.CodedWidth),
		Height:       int(plan.Bind.Event.FrameSize.Height),
		BitDepth:     8,
		MonoChrome:   true,
		SubsamplingX: true,
		SubsamplingY: true,
		Align:        64,
	}, 1)
	var stream av1.DecoderStream
	var refs av1.DecoderSurfaceReferences
	var state av1.DecoderFrameWorkState
	var referenceSurfaces [av1.InterRefsPerFrame]int
	var referenceFrames [av1.InterRefsPerFrame]*av1.Frame
	var releases [av1.RefFrames]int
	var stats av1.DecoderFrameWorkTileResidualStats
	var side av1.DecoderFrameWorkSideData
	var batchRunner av1.DecoderFrameWorkBatchResidualRunner
	scratch := publicDecoderResidualStreamScratch(plan.Size)
	runner, boundSide, err := av1.BindDecoderFrameWorkResidualStreamPlanRunner(plan, &stream, av1.DecoderFrameWorkResidualEventRuntime{
		State:             &state,
		Refs:              &refs,
		FramePool:         &pool,
		Align:             64,
		ReferenceSurfaces: referenceSurfaces[:],
		ReferenceFrames:   referenceFrames[:],
		Releases:          releases[:],
		WorkerPool:        workerPool,
		SideData:          &side,
		Stats:             &stats,
	}, scratch, &batchRunner)
	if err != nil {
		t.Fatal(err)
	}

	var probeOutput av1.Frame
	probeCtx, err := av1.DecoderFrameWorkPostFilterScratchContext(plan.Bind.Sequence, plan.Bind.Event, 64, &boundSide, &probeOutput)
	if err != nil {
		t.Fatal(err)
	}
	var probe av1.DecoderFrameWorkSupportedPostFilterScratchRunner
	exact, err := probe.ScratchLen(probeCtx)
	if err != nil {
		t.Fatal(err)
	}
	postScratch := publicDecoderPostFilterRequestScratch(av1.DecoderFrameWorkPostFilterRequestScratchLen(exact))
	postRunner := av1.DecoderFrameWorkSupportedPostFilterScratchRunner{Scratch: postScratch}

	allocs := testing.AllocsPerRun(1000, func() {
		pool.Reset()
		refs.Reset()
		state.Reset()
		if resetErr := runner.Reset(); resetErr != nil {
			err = resetErr
			return
		}
		postRunner = av1.DecoderFrameWorkSupportedPostFilterScratchRunner{Scratch: postScratch}
		result, runErr := runner.RunRTPPayloadWithPostFilterRunner(payload, &postRunner)
		if runErr != nil {
			err = runErr
			return
		}
		if result.EventCount != 3 ||
			result.Run.CompletedFrames != 1 ||
			result.Run.OutputCount != 1 ||
			result.Run.Last.Output == nil ||
			postRunner.Result.Completed != 0 ||
			postRunner.Context.RemainingPostFilters() != 0 ||
			stats.TXBs == 0 ||
			stats.Residuals == 0 {
			err = av1.ErrThreadingInvalidBatch
		}
	})
	if err != nil {
		t.Fatal(err)
	}
	if allocs != 0 {
		t.Fatalf("DecoderFrameWorkResidualStreamRunner postfilter allocated: %f", allocs)
	}
}

func TestPublicDecoderFrameWorkResidualStreamScratchLenAllocs(t *testing.T) {
	lowOverhead := publicDecoderResidualLowOverheadStream()
	lowOverheads := [...][]byte{
		publicDecoderResidualLowOverheadStream(),
		publicDecoderResidualLowOverheadFrameStream(),
	}
	rtpPayload := publicDecoderResidualRTPPayload()
	rtpPayloads := publicDecoderResidualFragmentedRTPPayloads()
	var events [4]av1.DecoderEvent
	var rtpBuffer [128]byte
	var rtpSpans [4]av1.RTPObuSpan
	var scratchSpans [1]av1.TileSpan
	var scratchJobs [1]av1.TileJob
	var scratchBatches [1]av1.TileBatch
	var err error
	var lowSize av1.DecoderFrameWorkResidualStreamScratchSize
	var lowBatchSize av1.DecoderFrameWorkResidualStreamScratchSize
	var rtpSize av1.DecoderFrameWorkResidualStreamScratchSize
	var rtpPayloadsSize av1.DecoderFrameWorkResidualStreamScratchSize

	allocs := testing.AllocsPerRun(1000, func() {
		var stream av1.DecoderStream
		lowSize, err = av1.DecoderFrameWorkResidualLowOverheadStreamScratchLen(stream, lowOverhead, 1, events[:], scratchSpans[:], scratchJobs[:], scratchBatches[:])
		if err != nil {
			return
		}
		lowBatchSize, err = av1.DecoderFrameWorkResidualLowOverheadStreamsScratchLen(stream, lowOverheads[:], 1, events[:], scratchSpans[:], scratchJobs[:], scratchBatches[:])
		if err != nil {
			return
		}
		rtpSize, err = av1.DecoderFrameWorkResidualRTPPayloadStreamScratchLen(stream, 0, rtpPayload, 1, rtpBuffer[:], rtpSpans[:], events[:], scratchSpans[:], scratchJobs[:], scratchBatches[:])
		if err != nil {
			return
		}
		rtpPayloadsSize, err = av1.DecoderFrameWorkResidualRTPPayloadsStreamScratchLen(stream, 0, rtpPayloads, 1, rtpBuffer[:], rtpSpans[:], events[:], scratchSpans[:], scratchJobs[:], scratchBatches[:])
	})
	if err != nil {
		t.Fatal(err)
	}
	if lowSize.Events != 3 ||
		lowSize.Event.Outputs != 1 ||
		lowBatchSize.Events != 3 ||
		lowBatchSize.Event.Outputs != 2 ||
		rtpSize.Events != 3 ||
		rtpSize.RTPBuffer == 0 ||
		rtpSize.RTPSpans != 3 ||
		rtpSize.Event.Outputs != 1 ||
		rtpPayloadsSize.Events != 1 ||
		rtpPayloadsSize.RTPBuffer == 0 ||
		rtpPayloadsSize.RTPSpans != 1 ||
		rtpPayloadsSize.Event.Outputs != 1 {
		t.Fatalf("scratch sizes low=%+v lowBatch=%+v rtp=%+v payloads=%+v", lowSize, lowBatchSize, rtpSize, rtpPayloadsSize)
	}
	if allocs != 0 {
		t.Fatalf("DecoderFrameWorkResidualStreamScratchLen allocated: %f", allocs)
	}
}

func TestPublicDecoderFrameWorkResidualEventsBindPlanAllocs(t *testing.T) {
	sequence := av1.SequenceHeader{ColorConfig: av1.ColorConfig{
		BitDepth:   8,
		MonoChrome: true,
	}}
	events := [...]av1.DecoderEvent{
		{Kind: av1.DecoderEventSequenceHeader, SequenceHeader: sequence},
		{Kind: av1.DecoderEventFrameHeader, SequenceHeader: sequence},
		{Kind: av1.DecoderEventTileGroup, SequenceHeader: sequence},
	}
	var plan av1.DecoderFrameWorkResidualEventBindPlan
	allocs := testing.AllocsPerRun(1000, func() {
		plan = av1.DecoderFrameWorkResidualEventsBindPlan(av1.SequenceHeader{}, events[:])
	})
	if !plan.HasEvent() || plan.EventIndex != 2 || plan.Event.Kind != av1.DecoderEventTileGroup {
		t.Fatalf("bind plan=%+v", plan)
	}
	if allocs != 0 {
		t.Fatalf("DecoderFrameWorkResidualEventsBindPlan allocated: %f", allocs)
	}
}

func TestPublicDecoderFrameWorkResidualStreamPlanAllocs(t *testing.T) {
	lowOverhead := publicDecoderResidualLowOverheadStream()
	lowOverheads := [...][]byte{
		publicDecoderResidualLowOverheadStream(),
		publicDecoderResidualLowOverheadFrameStream(),
	}
	rtpPayload := publicDecoderResidualRTPPayload()
	rtpPayloads := [...][]byte{
		publicDecoderResidualRTPPayload(),
		publicDecoderResidualRTPFramePayload(),
	}
	var events [4]av1.DecoderEvent
	var rtpBuffer [256]byte
	var rtpSpans [4]av1.RTPObuSpan
	var scratchSpans [1]av1.TileSpan
	var scratchJobs [1]av1.TileJob
	var scratchBatches [1]av1.TileBatch
	var err error
	var lowPlan av1.DecoderFrameWorkResidualStreamPlan
	var lowBatchPlan av1.DecoderFrameWorkResidualStreamPlan
	var rtpPlan av1.DecoderFrameWorkResidualStreamPlan
	var batchPlan av1.DecoderFrameWorkResidualStreamPlan

	allocs := testing.AllocsPerRun(1000, func() {
		var stream av1.DecoderStream
		lowPlan, err = av1.DecoderFrameWorkResidualLowOverheadStreamPlan(stream, lowOverhead, 1, events[:], scratchSpans[:], scratchJobs[:], scratchBatches[:])
		if err != nil {
			return
		}
		lowBatchPlan, err = av1.DecoderFrameWorkResidualLowOverheadStreamsPlan(stream, lowOverheads[:], 1, events[:], scratchSpans[:], scratchJobs[:], scratchBatches[:])
		if err != nil {
			return
		}
		rtpPlan, err = av1.DecoderFrameWorkResidualRTPPayloadStreamPlan(stream, 0, rtpPayload, 1, rtpBuffer[:], rtpSpans[:], events[:], scratchSpans[:], scratchJobs[:], scratchBatches[:])
		if err != nil {
			return
		}
		batchPlan, err = av1.DecoderFrameWorkResidualRTPPayloadsStreamPlan(stream, 0, rtpPayloads[:], 1, rtpBuffer[:], rtpSpans[:], events[:], scratchSpans[:], scratchJobs[:], scratchBatches[:])
	})
	if err != nil {
		t.Fatal(err)
	}
	if !lowPlan.HasEvent() ||
		!lowBatchPlan.HasEvent() ||
		!rtpPlan.HasEvent() ||
		!batchPlan.HasEvent() ||
		lowBatchPlan.Size.Event.Outputs != 2 ||
		batchPlan.Size.Event.Outputs != 2 ||
		batchPlan.Bind.Event.Kind != av1.DecoderEventTileGroup {
		t.Fatalf("stream plans low=%+v lowBatch=%+v rtp=%+v batch=%+v", lowPlan, lowBatchPlan, rtpPlan, batchPlan)
	}
	if allocs != 0 {
		t.Fatalf("DecoderFrameWorkResidualStreamPlan allocated: %f", allocs)
	}

	var reusablePlan av1.DecoderFrameWorkResidualStreamPlan
	allocs = testing.AllocsPerRun(1000, func() {
		reusablePlan = lowPlan.Max(lowBatchPlan).Max(rtpPlan).Max(batchPlan)
	})
	if !reusablePlan.HasEvent() ||
		reusablePlan.Size.Event.Outputs != 2 ||
		reusablePlan.Bind.Event.Kind != av1.DecoderEventTileGroup {
		t.Fatalf("reusable stream plan=%+v", reusablePlan)
	}
	if allocs != 0 {
		t.Fatalf("DecoderFrameWorkResidualStreamPlan.Max allocated: %f", allocs)
	}
}

func TestPublicDecoderFrameWorkResidualStreamCheckAndStateAllocs(t *testing.T) {
	size := av1.DecoderFrameWorkResidualStreamScratchSize{
		Events:    3,
		RTPBuffer: 32,
		RTPSpans:  3,
		Event: av1.DecoderFrameWorkResidualEventScratchSize{
			Runner: av1.DecoderFrameWorkBatchResidualRunnerScratchSize{
				Workers:         1,
				Int32Scratch:    8,
				ResidualScratch: 8,
			},
			SideData: av1.DecoderFrameWorkSideDataScratchSize{
				CDEFIndexMap:  4,
				LoopFilterMap: 4,
			},
			Plan:    av1.DecoderTileWorkPlan{SpanCount: 1, JobCount: 1, BatchCount: 1},
			Outputs: 2,
		},
	}
	scratch := publicDecoderResidualStreamScratch(size)
	var stream av1.DecoderStream
	runner := av1.DecoderFrameWorkResidualStreamRunner{
		Stream: &stream,
		EventRunner: av1.DecoderFrameWorkResidualEventRunner{
			Outputs: scratch.Outputs[:size.Event.Outputs],
		},
		Events:    scratch.Events[:size.Events],
		RTPBuffer: scratch.RTPBuffer[:size.RTPBuffer],
		RTPSpans:  scratch.RTPSpans[:size.RTPSpans],
		RTPUsed:   7,
	}
	var state av1.DecoderFrameWorkResidualStreamRunnerState
	var err error
	var ok bool
	var sequence av1.SequenceHeader
	allocs := testing.AllocsPerRun(1000, func() {
		err = scratch.Check(size)
		state = runner.State()
		sequence, ok = runner.SequenceHeader()
	})
	if err != nil {
		t.Fatal(err)
	}
	if !state.Bound || state.RTPUsed != 7 || state.OutputCapacity != 2 || ok || sequence != (av1.SequenceHeader{}) {
		t.Fatalf("state=%+v ok=%v sequence=%+v", state, ok, sequence)
	}
	if allocs != 0 {
		t.Fatalf("DecoderFrameWorkResidualStream Check/State allocated: %f", allocs)
	}
}

func TestPublicBindDecoderFrameWorkResidualStreamRunnerAllocs(t *testing.T) {
	size := av1.DecoderFrameWorkResidualStreamScratchSize{
		Events:    3,
		RTPBuffer: 32,
		RTPSpans:  3,
		Event: av1.DecoderFrameWorkResidualEventScratchSize{
			SideData: av1.DecoderFrameWorkSideDataScratchSize{
				CDEFIndexMap:             4,
				CDEFReadMap:              4,
				LoopFilterMap:            4,
				RestorationRecords:       1,
				RestorationBoundaryAbove: 2,
				RestorationBoundaryBelow: 2,
			},
		},
	}
	scratch := publicDecoderResidualStreamScratch(size)
	var stream av1.DecoderStream
	var eventRunner av1.DecoderFrameWorkResidualEventRunner
	var runner av1.DecoderFrameWorkResidualStreamRunner
	var err error
	allocs := testing.AllocsPerRun(1000, func() {
		runner, err = av1.BindDecoderFrameWorkResidualStreamRunner(size, &stream, eventRunner, scratch)
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(runner.Events) != size.Events ||
		len(runner.RTPBuffer) != size.RTPBuffer ||
		len(runner.RTPSpans) != size.RTPSpans ||
		len(runner.SideDataScratch.RestorationBoundaryBelow) != size.Event.SideData.RestorationBoundaryBelow {
		t.Fatalf("bound runner=%+v size=%+v", runner, size)
	}
	if allocs != 0 {
		t.Fatalf("BindDecoderFrameWorkResidualStreamRunner allocated: %f", allocs)
	}

	sequence := av1.SequenceHeader{ColorConfig: av1.ColorConfig{
		BitDepth:   8,
		MonoChrome: true,
	}}
	event := av1.DecoderEvent{
		Kind: av1.DecoderEventFrameHeader,
		FrameSize: av1.FrameSize{
			CodedWidth:    16,
			UpscaledWidth: 16,
			Height:        16,
		},
	}
	var spans [1]av1.TileSpan
	var jobs [1]av1.TileJob
	var batches [1]av1.TileBatch
	eventSize, err := av1.DecoderFrameWorkResidualEventScratchLen(sequence, event, 1, spans[:], jobs[:], batches[:])
	if err != nil {
		t.Fatal(err)
	}
	combinedSize := av1.DecoderFrameWorkResidualStreamScratchSize{
		Events:    1,
		RTPBuffer: 8,
		RTPSpans:  1,
		Event:     eventSize,
	}
	combinedPlan := av1.DecoderFrameWorkResidualStreamPlan{
		Size: combinedSize,
		Bind: av1.DecoderFrameWorkResidualEventBindPlan{
			Sequence:   sequence,
			Event:      event,
			EventIndex: 0,
		},
	}
	combinedScratch := publicDecoderResidualStreamScratch(combinedSize)
	var combinedRunner av1.DecoderFrameWorkResidualStreamRunner
	var side av1.DecoderFrameWorkSideData
	allocs = testing.AllocsPerRun(1000, func() {
		combinedRunner, side, err = av1.BindDecoderFrameWorkResidualStreamEventRunner(combinedSize, &stream, sequence, event, av1.DecoderFrameWorkResidualEventRuntime{}, combinedScratch, nil)
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(combinedRunner.Events) != combinedSize.Events ||
		len(combinedRunner.RTPBuffer) != combinedSize.RTPBuffer ||
		len(combinedRunner.EventRunner.Spans) != 0 ||
		len(side.CDEFIndexMap.Index) == 0 {
		t.Fatalf("combined bound runner=%+v side=%+v size=%+v", combinedRunner, side, combinedSize)
	}
	if allocs != 0 {
		t.Fatalf("BindDecoderFrameWorkResidualStreamEventRunner allocated: %f", allocs)
	}

	allocs = testing.AllocsPerRun(1000, func() {
		combinedRunner, side, err = av1.BindDecoderFrameWorkResidualStreamPlanRunner(combinedPlan, &stream, av1.DecoderFrameWorkResidualEventRuntime{}, combinedScratch, nil)
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(combinedRunner.Events) != combinedSize.Events ||
		len(combinedRunner.RTPBuffer) != combinedSize.RTPBuffer ||
		len(combinedRunner.EventRunner.Spans) != 0 ||
		len(side.CDEFIndexMap.Index) == 0 {
		t.Fatalf("plan-bound runner=%+v side=%+v plan=%+v", combinedRunner, side, combinedPlan)
	}
	if allocs != 0 {
		t.Fatalf("BindDecoderFrameWorkResidualStreamPlanRunner allocated: %f", allocs)
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
	event.CDEF = av1.CDEFParams{
		Damping:       5,
		StrengthCount: 1,
		YStrength:     [av1.MaxCDEFStrengths]uint8{63},
	}
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
	var stats av1.DecoderFrameWorkTileResidualStats
	var probeOutput av1.Frame
	probeCtx, err := av1.DecoderFrameWorkPostFilterScratchContext(sequence, event, 64, &side, &probeOutput)
	if err != nil {
		t.Fatal(err)
	}
	var probe av1.DecoderFrameWorkSupportedPostFilterScratchRunner
	postScratchSize, err := probe.ScratchLen(probeCtx)
	if err != nil {
		t.Fatal(err)
	}
	postRunner := av1.DecoderFrameWorkSupportedPostFilterScratchRunner{
		Scratch: publicDecoderPostFilterRequestScratch(av1.DecoderFrameWorkPostFilterRequestScratchLen(postScratchSize)),
	}
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
		Stats:             &stats,
	}

	allocs := testing.AllocsPerRun(1000, func() {
		pool.Reset()
		refs.Reset()
		state.Reset()
		result, runErr := eventRunner.RunWithPostFilterRunner(sequence, event, &side, &postRunner)
		if runErr != nil {
			err = runErr
			return
		}
		if result.Output == nil || result.Run != (av1.DecoderFrameWorkStepResult{ExecutedTileWork: true, CompletedFrame: true}) {
			err = av1.ErrDecoderInvalidFrameWorkStep
			return
		}
		if !postRunner.Result.Completed.Has(av1.DecoderFrameWorkPostFilterCDEF) || postRunner.Context.RemainingPostFilters() != 0 {
			err = av1.ErrDecoderUnsupportedPostFilter
			return
		}
		if stats.TXBs == 0 || stats.Residuals == 0 {
			err = av1.ErrThreadingInvalidBatch
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
		t.Fatalf("DecoderFrameWorkResidualEventRunner.RunWithPostFilterRunner allocated: %f", allocs)
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

func publicDecoderResidualEventScratch(size av1.DecoderFrameWorkResidualEventScratchSize) av1.DecoderFrameWorkResidualEventScratch {
	return av1.DecoderFrameWorkResidualEventScratch{
		Runner:   publicDecoderBatchResidualRunnerScratch(size.Runner),
		SideData: publicDecoderFrameWorkSideDataScratch(size.SideData),
		Spans:    make([]av1.TileSpan, size.Plan.SpanCount+1),
		Jobs:     make([]av1.TileJob, size.Plan.JobCount+1),
		Batches:  make([]av1.TileBatch, size.Plan.BatchCount+1),
	}
}

func publicDecoderResidualStreamScratch(size av1.DecoderFrameWorkResidualStreamScratchSize) av1.DecoderFrameWorkResidualStreamScratch {
	return av1.DecoderFrameWorkResidualStreamScratch{
		Events:    make([]av1.DecoderEvent, size.Events+1),
		Event:     publicDecoderResidualEventScratch(size.Event),
		SideData:  publicDecoderFrameWorkSideDataScratch(size.Event.SideData),
		Outputs:   make([]*av1.Frame, size.Event.Outputs+1),
		RTPBuffer: make([]byte, size.RTPBuffer+1),
		RTPSpans:  make([]av1.RTPObuSpan, size.RTPSpans+1),
	}
}

type publicDecoderResidualBitWriter struct {
	buf [128]byte
	bit int
}

func (w *publicDecoderResidualBitWriter) writeBits(value uint64, n uint8) {
	for i := int(n) - 1; i >= 0; i-- {
		if (value>>uint(i))&1 != 0 {
			w.buf[w.bit>>3] |= 1 << uint(7-(w.bit&7))
		}
		w.bit++
	}
}

func (w *publicDecoderResidualBitWriter) writeBool(value bool) {
	if value {
		w.writeBits(1, 1)
		return
	}
	w.writeBits(0, 1)
}

func (w *publicDecoderResidualBitWriter) bytes() []byte {
	return w.buf[:(w.bit+7)>>3]
}

func (w *publicDecoderResidualBitWriter) trailingBits() []byte {
	w.writeBits(1, 1)
	for w.bit&7 != 0 {
		w.writeBits(0, 1)
	}
	return w.buf[:w.bit>>3]
}

func publicDecoderResidualLowOverheadStream() []byte {
	var stream []byte
	stream = appendPublicLowOverheadOBU(stream, av1.OBUSequenceHeader, publicDecoderResidualSequenceHeaderPayload())
	stream = appendPublicLowOverheadOBU(stream, av1.OBUFrameHeader, publicDecoderResidualFrameHeaderPayload())
	stream = appendPublicLowOverheadOBU(stream, av1.OBUTileGroup, []byte{0x80})
	return stream
}

func publicDecoderResidualLowOverheadFrameStream() []byte {
	var stream []byte
	stream = appendPublicLowOverheadOBU(stream, av1.OBUFrameHeader, publicDecoderResidualFrameHeaderPayload())
	stream = appendPublicLowOverheadOBU(stream, av1.OBUTileGroup, []byte{0x80})
	return stream
}

func publicDecoderResidualRTPPayload() []byte {
	elements := [...]av1.RTPElement{
		{Data: publicDecoderResidualRTPElement(av1.OBUSequenceHeader, publicDecoderResidualSequenceHeaderPayload())},
		{Data: publicDecoderResidualRTPElement(av1.OBUFrameHeader, publicDecoderResidualFrameHeaderPayload())},
		{Data: publicDecoderResidualRTPElement(av1.OBUTileGroup, []byte{0x80})},
	}
	payload := make([]byte, 128)
	n, err := av1.PutRTPPayload(payload, av1.RTPAggregationHeader{
		ElementCount:                uint8(len(elements)),
		StartsNewCodedVideoSequence: true,
	}, elements[:])
	if err != nil {
		panic(err)
	}
	return payload[:n]
}

func publicDecoderResidualRTPFramePayload() []byte {
	elements := [...]av1.RTPElement{
		{Data: publicDecoderResidualRTPElement(av1.OBUFrameHeader, publicDecoderResidualFrameHeaderPayload())},
		{Data: publicDecoderResidualRTPElement(av1.OBUTileGroup, []byte{0x80})},
	}
	payload := make([]byte, 128)
	n, err := av1.PutRTPPayload(payload, av1.RTPAggregationHeader{
		ElementCount: uint8(len(elements)),
	}, elements[:])
	if err != nil {
		panic(err)
	}
	return payload[:n]
}

func publicDecoderResidualFragmentedRTPPayloads() [][]byte {
	payloads := make([][]byte, 0, 4)

	sequence := [...]av1.RTPElement{
		{Data: publicDecoderResidualRTPElement(av1.OBUSequenceHeader, publicDecoderResidualSequenceHeaderPayload())},
	}
	var packet [128]byte
	n, err := av1.PutRTPPayload(packet[:], av1.RTPAggregationHeader{
		ElementCount:                1,
		StartsNewCodedVideoSequence: true,
	}, sequence[:])
	if err != nil {
		panic(err)
	}
	payloads = append(payloads, append([]byte(nil), packet[:n]...))

	frameHeader := publicDecoderResidualRTPElement(av1.OBUFrameHeader, publicDecoderResidualFrameHeaderPayload())
	n, next, more, err := av1.PutRTPFragment(packet[:], frameHeader, 0, 2, false)
	if err != nil {
		panic(err)
	}
	if !more {
		panic("expected frame header RTP fragment")
	}
	payloads = append(payloads, append([]byte(nil), packet[:n]...))
	n, _, more, err = av1.PutRTPFragment(packet[:], frameHeader, next, len(packet), false)
	if err != nil {
		panic(err)
	}
	if more {
		panic("unexpected third frame header RTP fragment")
	}
	payloads = append(payloads, append([]byte(nil), packet[:n]...))

	tile := [...]av1.RTPElement{
		{Data: publicDecoderResidualRTPElement(av1.OBUTileGroup, []byte{0x80})},
	}
	n, err = av1.PutRTPPayload(packet[:], av1.RTPAggregationHeader{ElementCount: 1}, tile[:])
	if err != nil {
		panic(err)
	}
	payloads = append(payloads, append([]byte(nil), packet[:n]...))

	return payloads
}

func publicDecoderResidualRTPElement(typ av1.OBUType, payload []byte) []byte {
	var header [2]byte
	n, err := av1.PutOBUHeader(header[:], av1.OBUHeader{Type: typ})
	if err != nil {
		panic(err)
	}
	element := make([]byte, 0, n+len(payload))
	element = append(element, header[:n]...)
	element = append(element, payload...)
	return element
}

func publicDecoderResidualSequenceHeaderPayload() []byte {
	var w publicDecoderResidualBitWriter
	w.writeBits(0, 3)  // seq_profile
	w.writeBool(true)  // still_picture
	w.writeBool(true)  // reduced_still_picture_header
	w.writeBits(5, 5)  // seq_level_idx[0]
	w.writeBits(7, 4)  // frame_width_bits_minus_1
	w.writeBits(7, 4)  // frame_height_bits_minus_1
	w.writeBits(15, 8) // max_frame_width_minus_1
	w.writeBits(15, 8) // max_frame_height_minus_1
	w.writeBool(false) // use_128x128_superblock
	w.writeBool(true)  // enable_filter_intra
	w.writeBool(true)  // enable_intra_edge_filter
	w.writeBool(false) // enable_superres
	w.writeBool(true)  // enable_cdef
	w.writeBool(false) // enable_restoration
	w.writeBool(false) // high_bitdepth
	w.writeBool(true)  // mono_chrome
	w.writeBool(false) // color_description_present_flag
	w.writeBool(false) // color_range
	w.writeBool(false) // film_grain_params_present
	return w.trailingBits()
}

func publicDecoderResidualRealtimeSequenceHeaderPayload() []byte {
	var w publicDecoderResidualBitWriter
	w.writeBits(0, 3)  // seq_profile
	w.writeBool(false) // still_picture
	w.writeBool(false) // reduced_still_picture_header
	w.writeBool(false) // timing_info_present_flag
	w.writeBool(false) // initial_display_delay_present_flag
	w.writeBits(0, 5)  // operating_points_cnt_minus_1
	w.writeBits(0, 12) // operating_point_idc[0]
	w.writeBits(5, 5)  // seq_level_idx[0]
	w.writeBits(3, 4)  // frame_width_bits_minus_1
	w.writeBits(3, 4)  // frame_height_bits_minus_1
	w.writeBits(15, 4) // max_frame_width_minus_1
	w.writeBits(8, 4)  // max_frame_height_minus_1
	w.writeBool(false) // frame_id_numbers_present_flag
	w.writeBool(false) // use_128x128_superblock
	w.writeBool(true)  // enable_filter_intra
	w.writeBool(true)  // enable_intra_edge_filter
	w.writeBool(true)  // enable_interintra_compound
	w.writeBool(true)  // enable_masked_compound
	w.writeBool(false) // enable_warped_motion
	w.writeBool(true)  // enable_dual_filter
	w.writeBool(false) // enable_order_hint
	w.writeBool(false) // seq_choose_screen_content_tools
	w.writeBits(0, 1)  // seq_force_screen_content_tools
	w.writeBool(false) // enable_superres
	w.writeBool(true)  // enable_cdef
	w.writeBool(false) // enable_restoration
	w.writeBool(false) // high_bitdepth
	w.writeBool(false) // mono_chrome
	w.writeBool(false) // color_description_present_flag
	w.writeBool(false) // color_range
	w.writeBits(0, 2)  // chroma_sample_position
	w.writeBool(true)  // separate_uv_delta_q
	w.writeBool(false) // film_grain_params_present
	return w.trailingBits()
}

func publicDecoderResidualShownKeyFrameHeaderPayload() []byte {
	var w publicDecoderResidualBitWriter
	w.writeBool(false)                       // show_existing_frame
	w.writeBits(uint64(av1.FrameTypeKey), 2) // frame_type
	w.writeBool(true)                        // show_frame
	w.writeBool(false)                       // disable_cdf_update
	w.writeBool(false)                       // frame_size_override_flag
	w.writeBool(false)                       // render_and_frame_size_different
	w.writeBool(false)                       // disable_frame_end_update_cdf
	w.writeBool(false)                       // uniform_tile_spacing_flag
	publicDecoderResidualWriteColorQuantParams(&w)
	publicDecoderResidualWriteZeroSegmentationParams(&w)
	w.writeBool(false) // reduced_tx_set
	return w.bytes()
}

func publicDecoderResidualInterFrameHeaderPayload() []byte {
	var w publicDecoderResidualBitWriter
	w.writeBool(false)                         // show_existing_frame
	w.writeBits(uint64(av1.FrameTypeInter), 2) // frame_type
	w.writeBool(true)                          // show_frame
	w.writeBool(false)                         // error_resilient_mode
	w.writeBool(false)                         // disable_cdf_update
	w.writeBool(false)                         // frame_size_override_flag
	w.writeBits(0, 3)                          // primary_ref_frame
	w.writeBits(0x01, 8)                       // refresh_frame_flags
	for i := 0; i < av1.InterRefsPerFrame; i++ {
		w.writeBits(0, 3) // ref_frame_idx[i]
	}
	w.writeBool(false) // render_and_frame_size_different
	w.writeBool(false) // allow_high_precision_mv
	w.writeBool(false) // interpolation_filter is fixed
	w.writeBits(0, 2)  // interpolation_filter = EIGHTTAP
	w.writeBool(false) // is_motion_mode_switchable
	w.writeBool(false) // disable_frame_end_update_cdf
	w.writeBool(false) // uniform_tile_spacing_flag
	publicDecoderResidualWriteColorQuantParams(&w)
	publicDecoderResidualWriteZeroSegmentationParams(&w)
	w.writeBool(false) // reference_select
	w.writeBool(false) // reduced_tx_set
	for i := 0; i < av1.InterRefsPerFrame; i++ {
		w.writeBool(false) // global_motion_is_global
	}
	return w.bytes()
}

func publicDecoderResidualShowExistingFrameHeaderPayload(index uint8) []byte {
	var w publicDecoderResidualBitWriter
	w.writeBool(true) // show_existing_frame
	w.writeBits(uint64(index&7), 3)
	return w.bytes()
}

func publicDecoderResidualWriteColorQuantParams(w *publicDecoderResidualBitWriter) {
	w.writeBits(0, 8)  // base_q_idx
	w.writeBool(false) // y_dc_delta_q
	w.writeBool(false) // diff_uv_delta
	w.writeBool(false) // u_dc_delta_q
	w.writeBool(false) // u_ac_delta_q
	w.writeBool(false) // using_qmatrix
}

func publicDecoderResidualWriteZeroSegmentationParams(w *publicDecoderResidualBitWriter) {
	w.writeBool(false) // segmentation_enabled
}

func publicDecoderResidualFrameHeaderPayload() []byte {
	var w publicDecoderResidualBitWriter
	w.writeBool(true)  // disable_cdf_update
	w.writeBool(false) // render_and_frame_size_different
	w.writeBool(false) // uniform_tile_spacing_flag
	w.writeBits(0, 8)  // base_q_idx
	w.writeBool(false) // y_dc_delta_q
	w.writeBool(false) // using_qmatrix
	w.writeBool(false) // segmentation_enabled
	w.writeBool(false) // reduced_tx_set
	return w.bytes()
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
