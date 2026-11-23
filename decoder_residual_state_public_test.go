package goav1

import (
	"bytes"
	"errors"
	"testing"

	internaldecoder "github.com/thesyncim/goav1/internal/av1/decoder"
	internalframe "github.com/thesyncim/goav1/internal/av1/frame"
	internalobu "github.com/thesyncim/goav1/internal/av1/obu"
	internalparser "github.com/thesyncim/goav1/internal/av1/parser"
	internalthreading "github.com/thesyncim/goav1/internal/av1/threading"
	internaltile "github.com/thesyncim/goav1/internal/av1/tile"
	internaltransform "github.com/thesyncim/goav1/internal/av1/transform"
)

type publicDecoderResidualEventPostRunner struct {
	calls  int
	side   bool
	output *Frame
	err    error
	sample byte
	active bool
	seq    SequenceHeader
}

func (r *publicDecoderResidualEventPostRunner) Apply(ctx internaldecoder.FrameWorkPostFilterContext) error {
	r.calls++
	r.side = ctx.CDEFIndexMap != nil && ctx.LoopFilterMap != nil && ctx.RestorationFrameBuffers != nil
	r.output = ctx.Output
	r.active = ctx.ExecutedTileWork && ctx.Step.Kind == internaldecoder.FrameWorkStepBegin
	r.seq = ctx.Event.SequenceHeader
	if ctx.Output != nil {
		ctx.Output.Y.Pix[0] = r.sample
	}
	return r.err
}

func TestPublicDecoderTileResidualCDFStorageLifecycle(t *testing.T) {
	var initial internalthreading.FrameWorkTileResidualCDFStorage
	if err := initDecoderFrameWorkTileResidualCDFStorageDefault(&initial, 64); err != nil {
		t.Fatal(err)
	}
	if err := initial.DeltaQ.Update(2); err != nil {
		t.Fatal(err)
	}
	initialCDFs := decoderFrameWorkTileResidualCDFsFromStorage(&initial)
	if initialCDFs.Loop.Partition == nil || initialCDFs.Coeff.Coeff == nil || initialCDFs.TransformType == nil {
		t.Fatalf("initial CDF view=%+v", initialCDFs)
	}

	var retained internalthreading.FrameWorkTileResidualCDFStorage
	retainedValid := false
	batch := internaldecoder.FrameWorkBatch{
		Payload: []byte{0x00, 0xff},
		FrameWorkFrameContext: internaldecoder.FrameWorkFrameContext{
			Quantization: internalparser.QuantizationParams{BaseQIdx: 91},
		},
		InitialTileResidualCDFs:       &initial,
		RetainedTileResidualCDFs:      &retained,
		RetainedTileResidualCDFsValid: &retainedValid,
		Jobs: []internaltile.Job{
			{Tile: 0, Offset: 0, Size: 1},
			{Tile: 1, Offset: 1, Size: 1, UpdatesFrameContext: true},
		},
	}

	var storage internalthreading.FrameWorkTileResidualCDFStorage
	if err := initDecoderFrameWorkTileResidualCDFStorage(batch, &storage); err != nil {
		t.Fatal(err)
	}
	if !publicCDFValuesEqual(storage.DeltaQ.Values(), initial.DeltaQ.Values()) {
		t.Fatalf("storage delta q=%v want %v", storage.DeltaQ.Values(), initial.DeltaQ.Values())
	}
	if err := storage.DeltaQ.Update(1); err != nil {
		t.Fatal(err)
	}

	var state internaltile.DecodeState
	if err := initDecoderFrameWorkJobDecodeState(batch, 0, &state); err != nil {
		t.Fatal(err)
	}
	if state.RetainFrameContext || state.CurrentBaseQIdx != 91 {
		t.Fatalf("state=%+v", state)
	}
	if err := retainDecoderFrameWorkTileResidualCDFStorage(batch, 0, &state, &storage); err != nil {
		t.Fatal(err)
	}
	if retainedValid {
		t.Fatal("non-update tile retained frame context")
	}

	if err := initDecoderFrameWorkJobDecodeState(batch, 1, &state); err != nil {
		t.Fatal(err)
	}
	if !state.RetainFrameContext || state.CurrentBaseQIdx != 91 {
		t.Fatalf("update state=%+v", state)
	}
	if err := retainDecoderFrameWorkTileResidualCDFStorage(batch, 1, &state, &storage); err != nil {
		t.Fatal(err)
	}
	if !retainedValid {
		t.Fatal("update tile did not retain frame context")
	}
	if !publicCDFValuesEqual(retained.DeltaQ.Values(), storage.DeltaQ.Values()) {
		t.Fatalf("retained delta q=%v want %v", retained.DeltaQ.Values(), storage.DeltaQ.Values())
	}

	batch.InitialTileResidualCDFs = nil
	if err := initDecoderFrameWorkTileResidualCDFStorage(batch, &storage); err != nil {
		t.Fatal(err)
	}
	if publicCDFValuesEqual(storage.DeltaQ.Values(), initial.DeltaQ.Values()) {
		t.Fatal("default fallback unexpectedly reused initial frame context")
	}
}

func TestPublicDecoderTileResidualStateRejectsInvalidInputs(t *testing.T) {
	batch := internaldecoder.FrameWorkBatch{
		Payload: []byte{0xaa},
		Jobs:    []internaltile.Job{{Offset: 0, Size: 2}},
	}
	var state internaltile.DecodeState
	if err := initDecoderFrameWorkTileResidualCDFStorage(batch, nil); !errors.Is(err, internalthreading.ErrInvalidBatch) {
		t.Fatalf("nil cdf storage err=%v want %v", err, internalthreading.ErrInvalidBatch)
	}
	if err := initDecoderFrameWorkJobDecodeState(batch, -1, &state); !errors.Is(err, internalthreading.ErrInvalidBatch) {
		t.Fatalf("negative index err=%v want %v", err, internalthreading.ErrInvalidBatch)
	}
	if err := initDecoderFrameWorkJobDecodeState(batch, 0, nil); !errors.Is(err, internalthreading.ErrInvalidBatch) {
		t.Fatalf("nil state err=%v want %v", err, internalthreading.ErrInvalidBatch)
	}
	if err := initDecoderFrameWorkJobDecodeState(batch, 0, &state); !errors.Is(err, internaltile.ErrInvalidPlan) {
		t.Fatalf("invalid range err=%v want %v", err, internaltile.ErrInvalidPlan)
	}
	var storage internalthreading.FrameWorkTileResidualCDFStorage
	if err := retainDecoderFrameWorkTileResidualCDFStorage(batch, 0, &state, nil); !errors.Is(err, internalthreading.ErrInvalidBatch) {
		t.Fatalf("nil retain storage err=%v want %v", err, internalthreading.ErrInvalidBatch)
	}
	if err := initDecoderFrameWorkTileResidualCDFStorageDefault(nil, 0); !errors.Is(err, internalthreading.ErrInvalidBatch) {
		t.Fatalf("nil default cdf storage err=%v want %v", err, internalthreading.ErrInvalidBatch)
	}
	if got := decoderFrameWorkTileResidualCDFsFromStorage(nil); got.Loop.Partition != nil || got.Coeff.Coeff != nil || got.TransformType != nil {
		t.Fatalf("nil CDF view=%+v", got)
	}
	if err := retainDecoderFrameWorkTileResidualCDFStorage(batch, 0, nil, &storage); !errors.Is(err, internalthreading.ErrInvalidBatch) {
		t.Fatalf("nil retain state err=%v want %v", err, internalthreading.ErrInvalidBatch)
	}
}

func TestPublicDecoderBlockLoopRequestAndContextBinding(t *testing.T) {
	batch := publicDecoderBlockLoopBatch()
	rootColumns, err := decoderFrameWorkJobBlockLoopContextRootColumns(batch, 0)
	if err != nil {
		t.Fatal(err)
	}
	if rootColumns != 2 {
		t.Fatalf("root columns=%d want 2", rootColumns)
	}

	above := make([]internaltile.BlockLoopRootAboveContext, rootColumns+1)
	above[0].Partition[0] = 11
	above[rootColumns].Partition[0] = 7
	carrier, err := bindTileBlockLoopContextCarrier(rootColumns, above)
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
	req, err := decoderFrameWorkJobBlockLoopRequest(batch, 0, segMap, nil, 96, &carrier)
	if err != nil {
		t.Fatal(err)
	}
	if req.ContextCarrier != &carrier {
		t.Fatalf("context carrier=%p want %p", req.ContextCarrier, &carrier)
	}
	if req.Walk.Root != internaltile.BlockLevel128x128 ||
		req.Walk.MIColStart != 32 || req.Walk.MIRowStart != 32 ||
		req.Walk.MIColEnd != 76 || req.Walk.MIRowEnd != 66 {
		t.Fatalf("walk=%+v", req.Walk)
	}
	if !req.SkipMode.Enabled || req.CDEF.Bits != 2 ||
		!req.Delta.DeltaQPresent || req.SBSizeMIB != 32 || !req.Monochrome ||
		len(req.CurrentSegmentMap) != len(segMap) || req.SegmentMapStride != 96 ||
		req.InterpolationFilter != internalparser.InterpolationSwitchable || !req.EnableDualFilter ||
		!req.EnableFilterIntra || !req.EnableOrderHint || req.OrderHintBits != 5 ||
		req.CurrentOrderHint != 9 || req.ReferenceOrderHints[4] != 5 ||
		req.RefFrameSide[internaltile.ReferenceFrameLast2] != -1 ||
		req.RefFrameSide[internaltile.ReferenceFrameLast3] != 1 ||
		!req.UseRefFrameMVS || !req.TemporalMVSampleUnavailable {
		t.Fatalf("request=%+v", req)
	}

	if _, err := bindTileBlockLoopContextCarrier(-1, nil); !errors.Is(err, internaltile.ErrInvalidDecodeState) {
		t.Fatalf("negative carrier err=%v want %v", err, internaltile.ErrInvalidDecodeState)
	}
	if _, err := bindTileBlockLoopContextCarrier(rootColumns, above[:rootColumns-1]); !errors.Is(err, internalframe.ErrShortBuffer) {
		t.Fatalf("short carrier err=%v want %v", err, internalframe.ErrShortBuffer)
	}
	if _, err := decoderFrameWorkJobBlockLoopRequest(batch, 1, nil, nil, 0, nil); !errors.Is(err, internalthreading.ErrInvalidBatch) {
		t.Fatalf("bad job err=%v want %v", err, internalthreading.ErrInvalidBatch)
	}
}

func TestPublicDecoderFrameWorkResidualMaxScratchLen(t *testing.T) {
	batch, _, _, _, _ := publicDecoderResidualDriver(t)
	got32, got16, err := decoderFrameWorkResidualMaxScratchLen(batch, batch.Quantization.BaseQIdx, 0, internaldecoder.FrameWorkPlaneY)
	if err != nil {
		t.Fatal(err)
	}
	want32, want16, err := reconstructBlockMaxScratchLen(false)
	if err != nil {
		t.Fatal(err)
	}
	if got32 != want32 || got16 != want16 {
		t.Fatalf("decoder max scratch=%d/%d want %d/%d", got32, got16, want32, want16)
	}

	losslessBatch := batch
	losslessBatch.Quantization.BaseQIdx = 0
	lossless32, lossless16, err := decoderFrameWorkResidualMaxScratchLen(losslessBatch, 0, 0, internaldecoder.FrameWorkPlaneY)
	if err != nil {
		t.Fatal(err)
	}
	wantLossless32, wantLossless16, err := reconstructBlockMaxScratchLen(true)
	if err != nil {
		t.Fatal(err)
	}
	if lossless32 != wantLossless32 || lossless16 != wantLossless16 {
		t.Fatalf("decoder lossless max scratch=%d/%d want %d/%d", lossless32, lossless16, wantLossless32, wantLossless16)
	}

	if _, _, err := decoderFrameWorkResidualMaxScratchLen(batch, batch.Quantization.BaseQIdx, 0, internaldecoder.FrameWorkPlane(99)); !errors.Is(err, internalthreading.ErrInvalidBatch) {
		t.Fatalf("invalid plane max scratch err=%v want %v", err, internalthreading.ErrInvalidBatch)
	}

	colorBatch := batch
	colorBatch.Sequence.ColorConfig.MonoChrome = false
	colorBatch.Segmentation.Enabled = true
	colorBatch.Segmentation.Data.Segments[3].DeltaQ = -64
	frame32, frame16, err := decoderFrameWorkResidualMaxFrameScratchLen(colorBatch, colorBatch.Quantization.BaseQIdx)
	if err != nil {
		t.Fatal(err)
	}
	nonLossless32, nonLossless16, err := reconstructBlockMaxScratchLen(false)
	if err != nil {
		t.Fatal(err)
	}
	if frame32 != nonLossless32 || frame16 != nonLossless16 {
		t.Fatalf("frame scratch=%d/%d want %d/%d", frame32, frame16, nonLossless32, nonLossless16)
	}
}

func TestPublicDecoderFrameWorkBatchResidualScratchBinding(t *testing.T) {
	batch, state, storage, scratch, _ := publicDecoderResidualBatchDriver(t)
	size, err := decoderFrameWorkBatchResidualScratchLen(batch, batch.Quantization.BaseQIdx)
	if err != nil {
		t.Fatal(err)
	}
	want32, want16, err := decoderFrameWorkResidualMaxFrameScratchLen(batch, batch.Quantization.BaseQIdx)
	if err != nil {
		t.Fatal(err)
	}
	if size.Int32Scratch != want32 || size.ResidualScratch != want16 || size.LoopContextAbove != 1 {
		t.Fatalf("batch residual scratch=%+v want int32=%d residual=%d loop=1", size, want32, want16)
	}

	arena := decoderFrameWorkBatchResidualScratch{
		Int32Scratch:     make([]int32, size.Int32Scratch+1),
		ResidualScratch:  make([]int16, size.ResidualScratch+1),
		LoopContextAbove: make([]internaltile.BlockLoopRootAboveContext, size.LoopContextAbove+1),
	}
	arena.Int32Scratch[size.Int32Scratch] = 99
	arena.ResidualScratch[size.ResidualScratch] = 77
	arena.LoopContextAbove[size.LoopContextAbove].Partition[0] = 55
	bound, err := bindDecoderFrameWorkBatchResidualRequestScratch(size, arena)
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
	stats, err := decodeAndRetainDecoderFrameWorkBatchResiduals(batch, state, &storage, &scratch, bound)
	if err != nil {
		t.Fatal(err)
	}
	if stats.Loop.Blocks != 2 || stats.TXBs != 2 || stats.Residuals != 2 {
		t.Fatalf("stats=%+v", stats)
	}

	shortArena := arena
	shortArena.ResidualScratch = shortArena.ResidualScratch[:size.ResidualScratch-1]
	if _, err := bindDecoderFrameWorkBatchResidualRequestScratch(size, shortArena); !errors.Is(err, internalframe.ErrShortBuffer) {
		t.Fatalf("short residual scratch err=%v want %v", err, internalframe.ErrShortBuffer)
	}
	negativeSize := size
	negativeSize.Int32Scratch = -1
	if _, err := bindDecoderFrameWorkBatchResidualRequestScratch(negativeSize, arena); !errors.Is(err, internalframe.ErrShortBuffer) {
		t.Fatalf("negative scratch size err=%v want %v", err, internalframe.ErrShortBuffer)
	}
}

func TestPublicDecoderFrameWorkBatchResidualRunner(t *testing.T) {
	batch, _, _, _, _ := publicDecoderResidualBatchDriver(t)
	size, err := decoderFrameWorkBatchResidualRunnerScratchLen(batch, 2)
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
	runner, err := bindDecoderFrameWorkBatchResidualRunner(size, scratch)
	if err != nil {
		t.Fatal(err)
	}
	if !runner.UseDefaultPrediction {
		t.Fatal("runner did not enable default prediction scratch")
	}
	if scratch.PredictionScratch[1].Inter != &scratch.InterPredictionScratch[1] {
		t.Fatal("runner did not bind inter prediction scratch")
	}

	batch.Batch = internalthreading.Batch{Worker: 1}
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
	if err := runner.Run(badWorker); !errors.Is(err, internalthreading.ErrInvalidBatch) {
		t.Fatalf("bad worker err=%v want %v", err, internalthreading.ErrInvalidBatch)
	}
	brokenRunner := runner
	brokenRunner.Stats[0] = internalthreading.FrameWorkTileResidualStats{TXBs: 88, Residuals: 77}
	brokenRunner.Int32Scratch = nil
	brokenBatch := batch
	brokenBatch.Batch.Worker = 0
	if err := brokenRunner.Run(brokenBatch); !errors.Is(err, internalframe.ErrShortBuffer) {
		t.Fatalf("broken runner err=%v want %v", err, internalframe.ErrShortBuffer)
	}
	if brokenRunner.Stats[0] != (internalthreading.FrameWorkTileResidualStats{}) {
		t.Fatalf("failed worker stats were not cleared: %+v", brokenRunner.Stats[0])
	}
	shortScratch := publicDecoderBatchResidualRunnerScratch(size)
	shortScratch.Int32Scratch = shortScratch.Int32Scratch[:size.Int32Scratch-1]
	if _, err := bindDecoderFrameWorkBatchResidualRunner(size, shortScratch); !errors.Is(err, internalframe.ErrShortBuffer) {
		t.Fatalf("short runner scratch err=%v want %v", err, internalframe.ErrShortBuffer)
	}
	if _, err := decoderFrameWorkBatchResidualRunnerScratchLen(batch, 0); !errors.Is(err, internalthreading.ErrInvalidWorkerCount) {
		t.Fatalf("zero-worker scratch err=%v want %v", err, internalthreading.ErrInvalidWorkerCount)
	}
	negativeSize := size
	negativeSize.RestorationRequests = -1
	if _, err := bindDecoderFrameWorkBatchResidualRunner(negativeSize, scratch); !errors.Is(err, internalframe.ErrShortBuffer) {
		t.Fatalf("negative restoration request size err=%v want %v", err, internalframe.ErrShortBuffer)
	}
	var nilRunner *decoderFrameWorkBatchResidualRunner
	if err := nilRunner.Run(batch); !errors.Is(err, internalthreading.ErrInvalidBatch) {
		t.Fatalf("nil runner err=%v want %v", err, internalthreading.ErrInvalidBatch)
	}
}

func TestPublicDecoderFrameWorkResidualScratchSizeMax(t *testing.T) {
	batchA := decoderFrameWorkBatchResidualScratchSize{
		Int32Scratch:     1,
		ResidualScratch:  7,
		LoopContextAbove: 3,
	}
	batchB := decoderFrameWorkBatchResidualScratchSize{
		Int32Scratch:     5,
		ResidualScratch:  2,
		LoopContextAbove: 4,
	}
	if got, want := batchA.Max(batchB), (decoderFrameWorkBatchResidualScratchSize{
		Int32Scratch:     5,
		ResidualScratch:  7,
		LoopContextAbove: 4,
	}); got != want {
		t.Fatalf("batch residual scratch max=%+v want %+v", got, want)
	}

	runnerA := decoderFrameWorkBatchResidualRunnerScratchSize{
		Workers:             1,
		Batch:               batchA,
		RestorationRequests: 1,
		Int32Scratch:        1,
		ResidualScratch:     7,
		LoopContextAbove:    3,
	}
	runnerB := decoderFrameWorkBatchResidualRunnerScratchSize{
		Workers:             3,
		Batch:               batchB,
		RestorationRequests: 2,
		Int32Scratch:        15,
		ResidualScratch:     6,
		LoopContextAbove:    12,
	}
	if got, want := runnerA.Max(runnerB), (decoderFrameWorkBatchResidualRunnerScratchSize{
		Workers:             3,
		Batch:               batchA.Max(batchB),
		RestorationRequests: 2,
		Int32Scratch:        15,
		ResidualScratch:     7,
		LoopContextAbove:    12,
	}); got != want {
		t.Fatalf("runner residual scratch max=%+v want %+v", got, want)
	}

	eventA := decoderFrameWorkResidualEventScratchSize{
		Runner:   runnerA,
		SideData: decoderFrameWorkSideDataScratchSize{CDEFIndexMap: 2, LoopFilterMap: 8},
		Plan:     internaldecoder.TileWorkPlan{SpanCount: 2, JobCount: 2, BatchCount: 1},
		Outputs:  1,
	}
	eventB := decoderFrameWorkResidualEventScratchSize{
		Runner:   runnerB,
		SideData: decoderFrameWorkSideDataScratchSize{CDEFReadMap: 4, LoopFilterMap: 3},
		Plan:     internaldecoder.TileWorkPlan{SpanCount: 1, JobCount: 5, BatchCount: 2},
		Outputs:  3,
	}
	got := eventA.Max(eventB)
	if got.Runner != runnerA.Max(runnerB) ||
		got.SideData != (decoderFrameWorkSideDataScratchSize{CDEFIndexMap: 2, CDEFReadMap: 4, LoopFilterMap: 8}) ||
		got.Plan != (internaldecoder.TileWorkPlan{SpanCount: 2, JobCount: 5, BatchCount: 2}) ||
		got.Outputs != 3 {
		t.Fatalf("event residual scratch max=%+v", got)
	}

	streamA := decoderFrameWorkResidualStreamScratchSize{
		Events:    2,
		RTPBuffer: 16,
		RTPSpans:  1,
		Event:     eventA,
	}
	streamB := decoderFrameWorkResidualStreamScratchSize{
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

	sequenceA := SequenceHeader{ColorConfig: internalparser.ColorConfig{BitDepth: 8}}
	sequenceB := SequenceHeader{ColorConfig: internalparser.ColorConfig{BitDepth: 10}}
	planA := decoderFrameWorkResidualStreamPlan{
		Size: streamA,
		Bind: decoderFrameWorkResidualEventBindPlan{
			Sequence:   sequenceA,
			Event:      internaldecoder.Event{Kind: internaldecoder.EventExistingFrame},
			EventIndex: 0,
		},
	}
	planB := decoderFrameWorkResidualStreamPlan{
		Size: streamB,
		Bind: decoderFrameWorkResidualEventBindPlan{
			Sequence:   sequenceB,
			Event:      internaldecoder.Event{Kind: internaldecoder.EventTileGroup},
			EventIndex: 1,
		},
	}
	planGot := planA.Max(planB)
	if !planGot.HasEvent() ||
		planGot.Size != streamGot ||
		planGot.Bind.Event.Kind != internaldecoder.EventTileGroup ||
		planGot.Bind.Sequence.ColorConfig.BitDepth != 10 ||
		planGot.Bind.EventIndex != 1 {
		t.Fatalf("stream residual plan max=%+v", planGot)
	}
}

func TestPublicDecoderFrameWorkResidualStreamScratchCheckAndState(t *testing.T) {
	size := decoderFrameWorkResidualStreamScratchSize{
		Events:    2,
		RTPBuffer: 16,
		RTPSpans:  2,
		Event: decoderFrameWorkResidualEventScratchSize{
			Runner: decoderFrameWorkBatchResidualRunnerScratchSize{
				Workers:         1,
				Int32Scratch:    4,
				ResidualScratch: 4,
			},
			SideData: decoderFrameWorkSideDataScratchSize{
				CDEFIndexMap:  4,
				LoopFilterMap: 4,
			},
			Plan:    internaldecoder.TileWorkPlan{SpanCount: 1, JobCount: 1, BatchCount: 1},
			Outputs: 2,
		},
	}
	scratch := publicDecoderResidualStreamScratch(size)
	if err := scratch.Check(size); err != nil {
		t.Fatal(err)
	}
	shortEvents := scratch
	shortEvents.Events = shortEvents.Events[:size.Events-1]
	if err := shortEvents.Check(size); !errors.Is(err, ErrDecoderEventBufferTooSmall) {
		t.Fatalf("short events err=%v want %v", err, ErrDecoderEventBufferTooSmall)
	}
	shortRTP := scratch
	shortRTP.RTPBuffer = shortRTP.RTPBuffer[:size.RTPBuffer-1]
	if err := shortRTP.Check(size); !errors.Is(err, ErrRTPShortBuffer) {
		t.Fatalf("short RTP err=%v want %v", err, ErrRTPShortBuffer)
	}
	shortOutputs := scratch
	shortOutputs.Outputs = shortOutputs.Outputs[:size.Event.Outputs-1]
	if err := shortOutputs.Check(size); !errors.Is(err, internalframe.ErrShortBuffer) {
		t.Fatalf("short outputs err=%v want %v", err, internalframe.ErrShortBuffer)
	}
	shortEventScratch := scratch
	shortEventScratch.Event.Spans = shortEventScratch.Event.Spans[:0]
	if err := shortEventScratch.Check(size); !errors.Is(err, internalframe.ErrShortBuffer) {
		t.Fatalf("short event scratch err=%v want %v", err, internalframe.ErrShortBuffer)
	}

	var nilRunner *decoderFrameWorkResidualStreamRunner
	if state := nilRunner.State(); state != (decoderFrameWorkResidualStreamRunnerState{}) {
		t.Fatalf("nil runner state=%+v", state)
	}
	if _, ok := nilRunner.SequenceHeader(); ok {
		t.Fatal("nil runner reported sequence header")
	}

	var stream internaldecoder.Stream
	var events [4]internaldecoder.Event
	if count, err := stream.PushLowOverhead(publicDecoderResidualLowOverheadStream(), events[:]); err != nil {
		t.Fatal(err)
	} else if count != 3 || !stream.HasSequenceHeader() {
		t.Fatalf("low-overhead count=%d hasSequence=%v", count, stream.HasSequenceHeader())
	}
	runner := decoderFrameWorkResidualStreamRunner{
		Stream: &stream,
		EventRunner: decoderFrameWorkResidualEventRunner{
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

	frameHeader := publicDecoderResidualRTPElement(internalobu.TypeFrameHeader, publicDecoderResidualFrameHeaderPayload())
	var packet [8]byte
	n, _, more, err := PutRTPFragment(packet[:], frameHeader, 0, 2, false)
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
	size, err := decoderFrameWorkBatchResidualRunnerScratchLen(batch, 1)
	if err != nil {
		t.Fatal(err)
	}
	runner, err := bindDecoderFrameWorkBatchResidualRunner(size, publicDecoderBatchResidualRunnerScratch(size))
	if err != nil {
		t.Fatal(err)
	}

	sequence := SequenceHeader{ColorConfig: internalparser.ColorConfig{BitDepth: 8, MonoChrome: true}}
	cdef := internalparser.CDEFParams{Bits: 1, StrengthCount: 2}
	sideSize, err := decoderFrameWorkSideDataScratchLen(sequence, batch.FrameSize, cdef, internalparser.RestorationParams{})
	if err != nil {
		t.Fatal(err)
	}
	side, err := bindDecoderFrameWorkSideData(sequence, batch.FrameSize, cdef, internalparser.RestorationParams{}, decoderFrameWorkSideDataScratch{
		CDEFIndexMap:             make([]uint8, sideSize.CDEFIndexMap),
		CDEFReadMap:              make([]bool, sideSize.CDEFReadMap),
		LoopFilterMap:            make([]internalthreading.FrameWorkLoopFilterBlockRecord, sideSize.LoopFilterMap),
		RestorationRecords:       make([]internaltile.RestorationUnitRecord, sideSize.RestorationRecords),
		RestorationBoundaryAbove: make([]uint16, sideSize.RestorationBoundaryAbove),
		RestorationBoundaryBelow: make([]uint16, sideSize.RestorationBoundaryBelow),
	})
	if err != nil {
		t.Fatal(err)
	}
	side.LoopFilterMap.Records[0].Valid = true
	if err := setDecoderFrameWorkBatchResidualRunnerSideData(&runner, side); err != nil {
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

	activeRestoration := internalparser.RestorationParams{
		Type:      [3]internalparser.RestorationType{internalparser.RestorationWiener},
		UnitSizeY: 64,
	}
	activeSideSize, err := decoderFrameWorkSideDataScratchLen(sequence, batch.FrameSize, cdef, activeRestoration)
	if err != nil {
		t.Fatal(err)
	}
	activeSide, err := bindDecoderFrameWorkSideData(sequence, batch.FrameSize, cdef, activeRestoration, decoderFrameWorkSideDataScratch{
		CDEFIndexMap:             make([]uint8, activeSideSize.CDEFIndexMap),
		CDEFReadMap:              make([]bool, activeSideSize.CDEFReadMap),
		LoopFilterMap:            make([]internalthreading.FrameWorkLoopFilterBlockRecord, activeSideSize.LoopFilterMap),
		RestorationRecords:       make([]internaltile.RestorationUnitRecord, activeSideSize.RestorationRecords),
		RestorationBoundaryAbove: make([]uint16, activeSideSize.RestorationBoundaryAbove),
		RestorationBoundaryBelow: make([]uint16, activeSideSize.RestorationBoundaryBelow),
	})
	if err != nil {
		t.Fatal(err)
	}
	noRestorationScratch := publicDecoderBatchResidualRunnerScratch(size)
	noRestorationScratch.RestorationRequests = nil
	noRestorationRunner, err := bindDecoderFrameWorkBatchResidualRunner(size, noRestorationScratch)
	if err != nil {
		t.Fatal(err)
	}
	if err := setDecoderFrameWorkBatchResidualRunnerSideData(&noRestorationRunner, activeSide); !errors.Is(err, internalframe.ErrShortBuffer) {
		t.Fatalf("missing restoration requests err=%v want %v", err, internalframe.ErrShortBuffer)
	}
	var nilRunner *decoderFrameWorkBatchResidualRunner
	if err := setDecoderFrameWorkBatchResidualRunnerSideData(nilRunner, side); !errors.Is(err, internalthreading.ErrInvalidBatch) {
		t.Fatalf("nil side-data runner err=%v want %v", err, internalthreading.ErrInvalidBatch)
	}
}

func TestPublicDecoderFrameWorkResidualEventRunner(t *testing.T) {
	workerPool, err := internalthreading.NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	sequence := SequenceHeader{
		EnableCDEF: true,
		ColorConfig: internalparser.ColorConfig{
			BitDepth:   8,
			MonoChrome: true,
		},
	}
	event := publicDecoderResidualRunnerFrameEvent()
	var scratchSpans [2]internalparser.TileSpan
	var scratchJobs [2]internaltile.Job
	var scratchBatches [1]internalthreading.Batch
	eventScratch, err := decoderFrameWorkResidualEventScratchLen(sequence, &event, 1, scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		t.Fatal(err)
	}
	runnerSize := eventScratch.Runner
	plan := eventScratch.Plan
	if plan != (internaldecoder.TileWorkPlan{SpanCount: 2, JobCount: 2, BatchCount: 1}) {
		t.Fatalf("event residual runner plan=%+v", plan)
	}
	runner, err := bindDecoderFrameWorkBatchResidualRunner(runnerSize, publicDecoderBatchResidualRunnerScratch(runnerSize))
	if err != nil {
		t.Fatal(err)
	}
	side, err := bindDecoderFrameWorkResidualEventSideData(sequence, &event, publicDecoderFrameWorkSideDataScratch(eventScratch.SideData))
	if err != nil {
		t.Fatal(err)
	}
	side.LoopFilterMap.Records[0].Valid = true

	pool := publicDecoderPostFilterFramePool(t, FrameFormat{
		Width:      int(event.FrameSize.CodedWidth),
		Height:     int(event.FrameSize.Height),
		BitDepth:   8,
		MonoChrome: true,
		Align:      64,
	}, 2)
	var refs internaldecoder.SurfaceReferences
	var state internaldecoder.FrameWorkState
	var referenceSurfaces [internalparser.InterRefsPerFrame]int
	var referenceFrames [internalparser.InterRefsPerFrame]*Frame
	var spans [2]internalparser.TileSpan
	var jobs [2]internaltile.Job
	var batches [1]internalthreading.Batch
	var releases [internalparser.RefFrames]int
	var stats internalthreading.FrameWorkTileResidualStats
	var postSawSideData bool
	eventRunner := decoderFrameWorkResidualEventRunner{
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

	result, err := eventRunner.Run(sequence, &event, &side, func(ctx internaldecoder.FrameWorkPostFilterContext) error {
		postSawSideData = ctx.CDEFIndexMap != nil && ctx.LoopFilterMap != nil && ctx.RestorationFrameBuffers != nil
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Step.Kind != internaldecoder.FrameWorkStepBegin ||
		result.Run != (internaldecoder.FrameWorkStepResult{ExecutedTileWork: true, CompletedFrame: true}) ||
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
	if _, err := runDecoderFrameWorkEventWithResidualRunner(decoderFrameWorkResidualEventRequest{
		Event:  internaldecoder.Event{Kind: internaldecoder.EventFrame},
		Runner: nil,
	}); !errors.Is(err, internalthreading.ErrInvalidBatch) {
		t.Fatalf("nil residual event runner err=%v want %v", err, internalthreading.ErrInvalidBatch)
	}
}

func TestPublicDecoderFrameWorkResidualEventPostFilterRunner(t *testing.T) {
	workerPool, err := internalthreading.NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	sequence := SequenceHeader{
		EnableCDEF: true,
		ColorConfig: internalparser.ColorConfig{
			BitDepth:   8,
			MonoChrome: true,
		},
	}
	event := publicDecoderResidualRunnerFrameEvent()
	var scratchSpans [2]internalparser.TileSpan
	var scratchJobs [2]internaltile.Job
	var scratchBatches [1]internalthreading.Batch
	eventScratch, err := decoderFrameWorkResidualEventScratchLen(sequence, &event, 1, scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		t.Fatal(err)
	}
	runner, err := bindDecoderFrameWorkBatchResidualRunner(eventScratch.Runner, publicDecoderBatchResidualRunnerScratch(eventScratch.Runner))
	if err != nil {
		t.Fatal(err)
	}
	side, err := bindDecoderFrameWorkResidualEventSideData(sequence, &event, publicDecoderFrameWorkSideDataScratch(eventScratch.SideData))
	if err != nil {
		t.Fatal(err)
	}

	pool := publicDecoderPostFilterFramePool(t, FrameFormat{
		Width:      int(event.FrameSize.CodedWidth),
		Height:     int(event.FrameSize.Height),
		BitDepth:   8,
		MonoChrome: true,
		Align:      64,
	}, 2)
	var refs internaldecoder.SurfaceReferences
	var state internaldecoder.FrameWorkState
	var referenceSurfaces [internalparser.InterRefsPerFrame]int
	var referenceFrames [internalparser.InterRefsPerFrame]*Frame
	var spans [2]internalparser.TileSpan
	var jobs [2]internaltile.Job
	var batches [1]internalthreading.Batch
	var releases [internalparser.RefFrames]int
	eventRunner := decoderFrameWorkResidualEventRunner{
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

	result, err := eventRunner.RunWithPostFilterRunner(sequence, &event, &side, &postRunner)
	if err != nil {
		t.Fatal(err)
	}
	if result.Run != (internaldecoder.FrameWorkStepResult{ExecutedTileWork: true, CompletedFrame: true}) ||
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
	if _, err := runDecoderFrameWorkEventWithResidualRunner(decoderFrameWorkResidualEventRequest{
		Runner:     &runner,
		Post:       func(internaldecoder.FrameWorkPostFilterContext) error { return nil },
		PostRunner: &postRunner,
	}); !errors.Is(err, internaldecoder.ErrInvalidFrameWorkState) {
		t.Fatalf("mixed post callback/runner err=%v want %v", err, internaldecoder.ErrInvalidFrameWorkState)
	}
}

func TestPublicDecoderFrameWorkResidualEventRunnerResetsStats(t *testing.T) {
	workerPool, err := internalthreading.NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	sequence := SequenceHeader{
		ColorConfig: internalparser.ColorConfig{
			BitDepth:   8,
			MonoChrome: true,
		},
	}
	event := publicDecoderResidualRunnerFrameEvent()
	var scratchSpans [2]internalparser.TileSpan
	var scratchJobs [2]internaltile.Job
	var scratchBatches [2]internalthreading.Batch
	runnerSize, _, err := decoderFrameWorkResidualEventRunnerScratchLen(sequence, &event, 2, scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		t.Fatal(err)
	}
	runner, err := bindDecoderFrameWorkBatchResidualRunner(runnerSize, publicDecoderBatchResidualRunnerScratch(runnerSize))
	if err != nil {
		t.Fatal(err)
	}
	runner.Stats[1] = internalthreading.FrameWorkTileResidualStats{TXBs: 77, Residuals: 55}

	pool := publicDecoderPostFilterFramePool(t, FrameFormat{
		Width:      int(event.FrameSize.CodedWidth),
		Height:     int(event.FrameSize.Height),
		BitDepth:   8,
		MonoChrome: true,
		Align:      64,
	}, 1)
	var refs internaldecoder.SurfaceReferences
	var state internaldecoder.FrameWorkState
	var referenceSurfaces [internalparser.InterRefsPerFrame]int
	var referenceFrames [internalparser.InterRefsPerFrame]*Frame
	var spans [2]internalparser.TileSpan
	var jobs [2]internaltile.Job
	var batches [1]internalthreading.Batch
	var releases [internalparser.RefFrames]int
	eventRunner := decoderFrameWorkResidualEventRunner{
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
	result, err := eventRunner.Run(sequence, &event, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if result.Run != (internaldecoder.FrameWorkStepResult{ExecutedTileWork: true, CompletedFrame: true}) {
		t.Fatalf("result=%+v", result)
	}
	if runner.Stats[1] != (internalthreading.FrameWorkTileResidualStats{}) {
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
	sequence := SequenceHeader{
		ColorConfig: internalparser.ColorConfig{
			BitDepth:   8,
			MonoChrome: true,
		},
	}
	event := publicDecoderResidualRunnerFrameEvent()
	var scratchSpans [2]internalparser.TileSpan
	var scratchJobs [2]internaltile.Job
	var scratchBatches [1]internalthreading.Batch
	runnerSize, _, err := decoderFrameWorkResidualEventRunnerScratchLen(sequence, &event, 1, scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		t.Fatal(err)
	}
	runner, err := bindDecoderFrameWorkBatchResidualRunner(runnerSize, publicDecoderBatchResidualRunnerScratch(runnerSize))
	if err != nil {
		t.Fatal(err)
	}
	runner.Stats[0] = internalthreading.FrameWorkTileResidualStats{TXBs: 33, Residuals: 22}
	stats := internalthreading.FrameWorkTileResidualStats{TXBs: 44, Residuals: 11}

	_, err = runDecoderFrameWorkEventWithResidualRunner(decoderFrameWorkResidualEventRequest{
		Runner:   &runner,
		Sequence: sequence,
		Event:    event,
		Workers:  1,
		Stats:    &stats,
	})
	if !errors.Is(err, internaldecoder.ErrInvalidFrameWorkState) {
		t.Fatalf("plan error=%v want %v", err, internaldecoder.ErrInvalidFrameWorkState)
	}
	if runner.Stats[0] != (internalthreading.FrameWorkTileResidualStats{}) {
		t.Fatalf("plan-error stats were not reset: %+v", runner.Stats[0])
	}
	if stats != (internalthreading.FrameWorkTileResidualStats{}) {
		t.Fatalf("plan-error aggregate stats were not reset: %+v", stats)
	}
}

func TestPublicDecoderFrameWorkResidualEventSupportedPostFilterScratchRunner(t *testing.T) {
	workerPool, err := internalthreading.NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	sequence := SequenceHeader{
		EnableCDEF: true,
		ColorConfig: internalparser.ColorConfig{
			BitDepth:   8,
			MonoChrome: true,
		},
	}
	event := publicDecoderResidualRunnerFrameEvent()
	event.CDEF = internalparser.CDEFParams{
		Damping:       5,
		StrengthCount: 1,
		YStrength:     [internalparser.MaxCDEFStrengths]uint8{63},
	}
	var scratchSpans [2]internalparser.TileSpan
	var scratchJobs [2]internaltile.Job
	var scratchBatches [1]internalthreading.Batch
	eventScratch, err := decoderFrameWorkResidualEventScratchLen(sequence, &event, 1, scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		t.Fatal(err)
	}
	runner, err := bindDecoderFrameWorkBatchResidualRunner(eventScratch.Runner, publicDecoderBatchResidualRunnerScratch(eventScratch.Runner))
	if err != nil {
		t.Fatal(err)
	}
	side, err := bindDecoderFrameWorkResidualEventSideData(sequence, &event, publicDecoderFrameWorkSideDataScratch(eventScratch.SideData))
	if err != nil {
		t.Fatal(err)
	}

	var probeOutput Frame
	probeCtx, err := decoderFrameWorkPostFilterScratchContext(sequence, &event, 64, &side, &probeOutput)
	if err != nil {
		t.Fatal(err)
	}
	var probe decoderFrameWorkSupportedPostFilterScratchRunner
	exact, err := probe.ScratchLen(probeCtx)
	if err != nil {
		t.Fatal(err)
	}
	arenaSize := decoderFrameWorkPostFilterRequestScratchLen(exact)
	postRunner := decoderFrameWorkSupportedPostFilterScratchRunner{
		Scratch: publicDecoderPostFilterRequestScratch(arenaSize),
	}

	pool := publicDecoderPostFilterFramePool(t, probeOutput.Format, 1)
	var refs internaldecoder.SurfaceReferences
	var state internaldecoder.FrameWorkState
	var referenceSurfaces [internalparser.InterRefsPerFrame]int
	var referenceFrames [internalparser.InterRefsPerFrame]*Frame
	var spans [2]internalparser.TileSpan
	var jobs [2]internaltile.Job
	var batches [1]internalthreading.Batch
	var releases [internalparser.RefFrames]int
	eventRunner := decoderFrameWorkResidualEventRunner{
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

	result, err := eventRunner.RunWithPostFilterRunner(sequence, &event, &side, &postRunner)
	if err != nil {
		t.Fatal(err)
	}
	cdefRead := false
	for _, read := range side.CDEFIndexMap.Read {
		cdefRead = cdefRead || read
	}
	if result.Run != (internaldecoder.FrameWorkStepResult{ExecutedTileWork: true, CompletedFrame: true}) ||
		result.Output == nil ||
		state.Active() ||
		!postRunner.Result.Completed.Has(internaldecoder.FrameWorkPostFilterCDEF) ||
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
	workerPool, err := internalthreading.NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	sequence := SequenceHeader{
		EnableCDEF:     true,
		EnableSuperRes: true,
		ColorConfig: internalparser.ColorConfig{
			BitDepth:   8,
			MonoChrome: true,
		},
	}
	event := publicDecoderResidualRunnerFrameEvent()
	event.FrameSize.UpscaledWidth = 160
	event.FrameSize.SuperResEnabled = true
	event.FrameSize.SuperResDenominator = 13
	event.CDEF = internalparser.CDEFParams{
		Damping:       5,
		StrengthCount: 1,
		YStrength:     [internalparser.MaxCDEFStrengths]uint8{63},
	}
	event.FilmGrain = internalparser.FilmGrainParams{
		ParamsPresent: true,
		Apply:         true,
		Seed:          0x1234,
		BitDepth:      8,
		NumYPoints:    1,
		YPoints:       [internalparser.MaxFilmGrainYPoints][2]uint8{{0, 64}},
		ScalingShift:  8,
		Overlap:       true,
	}
	var scratchSpans [2]internalparser.TileSpan
	var scratchJobs [2]internaltile.Job
	var scratchBatches [1]internalthreading.Batch
	eventScratch, err := decoderFrameWorkResidualEventScratchLen(sequence, &event, 1, scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		t.Fatal(err)
	}
	runner, err := bindDecoderFrameWorkBatchResidualRunner(eventScratch.Runner, publicDecoderBatchResidualRunnerScratch(eventScratch.Runner))
	if err != nil {
		t.Fatal(err)
	}
	side, err := bindDecoderFrameWorkResidualEventSideData(sequence, &event, publicDecoderFrameWorkSideDataScratch(eventScratch.SideData))
	if err != nil {
		t.Fatal(err)
	}

	var scratchOutput Frame
	scratchCtx, err := decoderFrameWorkPostFilterScratchContext(sequence, &event, 64, &side, &scratchOutput)
	if err != nil {
		t.Fatal(err)
	}
	var postRunner decoderFrameWorkCallerPostFilterScratchRunner
	first, err := postRunner.ScratchLen(scratchCtx)
	if err != nil {
		t.Fatal(err)
	}
	postRunner.Scratch = publicDecoderPostFilterRequestScratch(decoderFrameWorkPostFilterRequestScratchLen(first))
	full, err := postRunner.ScratchLen(scratchCtx)
	if err != nil {
		t.Fatal(err)
	}
	postRunner.Scratch = publicDecoderPostFilterRequestScratch(decoderFrameWorkPostFilterRequestScratchLen(full))

	pool := publicDecoderPostFilterFramePool(t, scratchOutput.Format, 1)
	var refs internaldecoder.SurfaceReferences
	var state internaldecoder.FrameWorkState
	var referenceSurfaces [internalparser.InterRefsPerFrame]int
	var referenceFrames [internalparser.InterRefsPerFrame]*Frame
	var spans [2]internalparser.TileSpan
	var jobs [2]internaltile.Job
	var batches [1]internalthreading.Batch
	var releases [internalparser.RefFrames]int
	var outputs [2]*Frame
	eventRunner := decoderFrameWorkResidualEventRunner{
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

	result, err := eventRunner.RunWithPostFilterRunner(sequence, &event, &side, &postRunner)
	if err != nil {
		t.Fatal(err)
	}
	wantCompleted := internaldecoder.FrameWorkPostFilterCDEF |
		internaldecoder.FrameWorkPostFilterSuperRes |
		internaldecoder.FrameWorkPostFilterFilmGrain
	if result.Run != (internaldecoder.FrameWorkStepResult{ExecutedTileWork: true, CompletedFrame: true}) ||
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
	events := [...]internaldecoder.Event{
		{Kind: internaldecoder.EventSequenceHeader, SequenceHeader: sequence, NewCodedVideoSequence: true},
		event,
	}
	listResult, err := eventRunner.RunEventsWithPostFilterRunner(SequenceHeader{}, events[:], publicDecoderFrameWorkSideDataScratch(eventScratch.SideData), &postRunner)
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
		result, err = eventRunner.RunWithPostFilterRunner(sequence, &event, &side, &postRunner)
		if err != nil {
			return
		}
		if result.Output == nil ||
			result.Run != (internaldecoder.FrameWorkStepResult{ExecutedTileWork: true, CompletedFrame: true}) ||
			postRunner.Result.Completed != wantCompleted ||
			postRunner.Context.RemainingPostFilters() != 0 ||
			!postRunner.Context.DetachedPostFilterOutput() {
			err = internaldecoder.ErrInvalidFrameWorkStep
			return
		}
		if _, ok := refs.ReferenceSlot(0); !ok {
			err = internaldecoder.ErrInvalidSurfaceReference
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
	sequence := SequenceHeader{
		EnableCDEF: true,
		ColorConfig: internalparser.ColorConfig{
			BitDepth:   8,
			MonoChrome: true,
		},
	}
	event := publicDecoderResidualRunnerFrameEvent()
	var spans [2]internalparser.TileSpan
	var jobs [2]internaltile.Job
	var batches [1]internalthreading.Batch
	size, plan, err := decoderFrameWorkResidualEventRunnerScratchLen(sequence, &event, 1, spans[:], jobs[:], batches[:])
	if err != nil {
		t.Fatal(err)
	}
	if plan != (internaldecoder.TileWorkPlan{SpanCount: 2, JobCount: 2, BatchCount: 1}) ||
		size.Workers != 1 ||
		size.RestorationRequests != 1 ||
		size.Batch.LoopContextAbove == 0 {
		t.Fatalf("size=%+v plan=%+v", size, plan)
	}
	if spans[0] != (internalparser.TileSpan{Tile: 0, Row: 0, Col: 0, Offset: 1, Size: 256}) ||
		spans[1] != (internalparser.TileSpan{Tile: 1, Row: 0, Col: 1, Offset: 257, Size: 256}) ||
		jobs[0].Tile != 0 ||
		jobs[1].Tile != 1 ||
		batches[0].Count != 2 {
		t.Fatalf("spans=%+v jobs=%+v batches=%+v", spans, jobs, batches)
	}
	var shortSpans [1]internalparser.TileSpan
	if _, _, err := decoderFrameWorkResidualEventRunnerScratchLen(sequence, &event, 1, shortSpans[:], jobs[:], batches[:]); !errors.Is(err, internalparser.ErrInvalidTileGroup) {
		t.Fatalf("short spans err=%v want %v", err, internalparser.ErrInvalidTileGroup)
	}
	if _, _, err := decoderFrameWorkResidualEventRunnerScratchLen(sequence, &internaldecoder.Event{Kind: internaldecoder.EventSequenceHeader}, 1, spans[:], jobs[:], batches[:]); !errors.Is(err, internaldecoder.ErrInvalidTileWork) {
		t.Fatalf("invalid event err=%v want %v", err, internaldecoder.ErrInvalidTileWork)
	}

	allocs := testing.AllocsPerRun(1000, func() {
		_, _, err = decoderFrameWorkResidualEventRunnerScratchLen(sequence, &event, 1, spans[:], jobs[:], batches[:])
	})
	if err != nil {
		t.Fatal(err)
	}
	if allocs != 0 {
		t.Fatalf("DecoderFrameWorkResidualEventRunnerScratchLen allocated: %f", allocs)
	}

	combinedSize, err := decoderFrameWorkResidualEventScratchLen(sequence, &event, 1, spans[:], jobs[:], batches[:])
	if err != nil {
		t.Fatal(err)
	}
	if combinedSize.Runner != size ||
		combinedSize.Plan != plan ||
		combinedSize.SideData.CDEFIndexMap == 0 ||
		combinedSize.SideData.LoopFilterMap == 0 {
		t.Fatalf("combined residual event scratch size=%+v runner=%+v plan=%+v", combinedSize, size, plan)
	}
	side, err := bindDecoderFrameWorkResidualEventSideData(sequence, &event, publicDecoderFrameWorkSideDataScratch(combinedSize.SideData))
	if err != nil {
		t.Fatal(err)
	}
	if len(side.CDEFIndexMap.Index) != combinedSize.SideData.CDEFIndexMap ||
		len(side.LoopFilterMap.Records) != combinedSize.SideData.LoopFilterMap {
		t.Fatalf("side=%+v size=%+v", side, combinedSize.SideData)
	}
	shortSideScratch := publicDecoderFrameWorkSideDataScratch(combinedSize.SideData)
	shortSideScratch.CDEFIndexMap = shortSideScratch.CDEFIndexMap[:combinedSize.SideData.CDEFIndexMap-1]
	if _, err := bindDecoderFrameWorkResidualEventSideData(sequence, &event, shortSideScratch); !errors.Is(err, internalframe.ErrShortBuffer) {
		t.Fatalf("short event side-data err=%v want %v", err, internalframe.ErrShortBuffer)
	}

	allocs = testing.AllocsPerRun(1000, func() {
		_, err = decoderFrameWorkResidualEventScratchLen(sequence, &event, 1, spans[:], jobs[:], batches[:])
	})
	if err != nil {
		t.Fatal(err)
	}
	if allocs != 0 {
		t.Fatalf("DecoderFrameWorkResidualEventScratchLen allocated: %f", allocs)
	}

	sideScratch := publicDecoderFrameWorkSideDataScratch(combinedSize.SideData)
	allocs = testing.AllocsPerRun(1000, func() {
		_, err = bindDecoderFrameWorkResidualEventSideData(sequence, &event, sideScratch)
	})
	if err != nil {
		t.Fatal(err)
	}
	if allocs != 0 {
		t.Fatalf("BindDecoderFrameWorkResidualEventSideData allocated: %f", allocs)
	}
}

func TestPublicBindDecoderFrameWorkResidualEventRunner(t *testing.T) {
	workerPool, err := internalthreading.NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	sequence := SequenceHeader{
		EnableCDEF: true,
		ColorConfig: internalparser.ColorConfig{
			BitDepth:   8,
			MonoChrome: true,
		},
	}
	event := publicDecoderResidualRunnerFrameEvent()
	var scratchSpans [2]internalparser.TileSpan
	var scratchJobs [2]internaltile.Job
	var scratchBatches [1]internalthreading.Batch
	size, err := decoderFrameWorkResidualEventScratchLen(sequence, &event, 1, scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		t.Fatal(err)
	}

	pool := publicDecoderPostFilterFramePool(t, FrameFormat{
		Width:      int(event.FrameSize.CodedWidth),
		Height:     int(event.FrameSize.Height),
		BitDepth:   8,
		MonoChrome: true,
		Align:      64,
	}, 1)
	var refs internaldecoder.SurfaceReferences
	var state internaldecoder.FrameWorkState
	var referenceSurfaces [internalparser.InterRefsPerFrame]int
	var referenceFrames [internalparser.InterRefsPerFrame]*Frame
	var releases [internalparser.RefFrames]int
	var stats internalthreading.FrameWorkTileResidualStats
	runtime := decoderFrameWorkResidualEventRuntime{
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
	var batchRunner decoderFrameWorkBatchResidualRunner
	eventRunner, side, err := bindDecoderFrameWorkResidualEventRunner(size, sequence, &event, runtime, scratch, &batchRunner)
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

	result, err := eventRunner.Run(sequence, &event, &side, nil)
	if err != nil {
		t.Fatal(err)
	}
	if result.Run != (internaldecoder.FrameWorkStepResult{ExecutedTileWork: true, CompletedFrame: true}) ||
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
	var allocBatchRunner decoderFrameWorkBatchResidualRunner
	allocs := testing.AllocsPerRun(1000, func() {
		_, _, err = bindDecoderFrameWorkResidualEventRunner(size, sequence, &event, runtime, allocScratch, &allocBatchRunner)
	})
	if err != nil {
		t.Fatal(err)
	}
	if allocs != 0 {
		t.Fatalf("BindDecoderFrameWorkResidualEventRunner allocated: %f", allocs)
	}
}

func TestPublicBindDecoderFrameWorkResidualEventRunnerRejectsInvalidScratch(t *testing.T) {
	sequence := SequenceHeader{
		EnableCDEF: true,
		ColorConfig: internalparser.ColorConfig{
			BitDepth:   8,
			MonoChrome: true,
		},
	}
	event := publicDecoderResidualRunnerFrameEvent()
	var scratchSpans [2]internalparser.TileSpan
	var scratchJobs [2]internaltile.Job
	var scratchBatches [1]internalthreading.Batch
	size, err := decoderFrameWorkResidualEventScratchLen(sequence, &event, 1, scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		t.Fatal(err)
	}
	runtime := decoderFrameWorkResidualEventRuntime{Align: 64}
	scratch := publicDecoderResidualEventScratch(size)

	if _, _, err := bindDecoderFrameWorkResidualEventRunner(size, sequence, &event, runtime, scratch, nil); !errors.Is(err, internalthreading.ErrInvalidBatch) {
		t.Fatalf("nil batch runner err=%v want %v", err, internalthreading.ErrInvalidBatch)
	}

	var batchRunner decoderFrameWorkBatchResidualRunner
	shortPlanScratch := scratch
	shortPlanScratch.Jobs = shortPlanScratch.Jobs[:size.Plan.JobCount-1]
	if _, _, err := bindDecoderFrameWorkResidualEventRunner(size, sequence, &event, runtime, shortPlanScratch, &batchRunner); !errors.Is(err, internalframe.ErrShortBuffer) {
		t.Fatalf("short plan scratch err=%v want %v", err, internalframe.ErrShortBuffer)
	}

	shortRunnerScratch := scratch
	shortRunnerScratch.Runner.Stats = shortRunnerScratch.Runner.Stats[:size.Runner.Workers-1]
	if _, _, err := bindDecoderFrameWorkResidualEventRunner(size, sequence, &event, runtime, shortRunnerScratch, &batchRunner); !errors.Is(err, internalframe.ErrShortBuffer) {
		t.Fatalf("short runner scratch err=%v want %v", err, internalframe.ErrShortBuffer)
	}

	shortSideScratch := scratch
	shortSideScratch.SideData.LoopFilterMap = shortSideScratch.SideData.LoopFilterMap[:size.SideData.LoopFilterMap-1]
	if _, _, err := bindDecoderFrameWorkResidualEventRunner(size, sequence, &event, runtime, shortSideScratch, &batchRunner); !errors.Is(err, internalframe.ErrShortBuffer) {
		t.Fatalf("short side-data scratch err=%v want %v", err, internalframe.ErrShortBuffer)
	}
}

func TestPublicDecoderFrameWorkResidualEventRunnerRunEvents(t *testing.T) {
	workerPool, err := internalthreading.NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	sequence := SequenceHeader{
		EnableCDEF: true,
		ColorConfig: internalparser.ColorConfig{
			BitDepth:   8,
			MonoChrome: true,
		},
	}
	frame := publicDecoderResidualRunnerFrameEvent()
	header := frame
	header.Kind = internaldecoder.EventFrameHeader
	header.Unit.Payload = nil
	tile := frame
	tile.Kind = internaldecoder.EventTileGroup
	events := [...]internaldecoder.Event{
		{
			Kind:                  internaldecoder.EventSequenceHeader,
			SequenceHeader:        sequence,
			NewCodedVideoSequence: true,
		},
		header,
		tile,
	}

	var scratchSpans [2]internalparser.TileSpan
	var scratchJobs [2]internaltile.Job
	var scratchBatches [1]internalthreading.Batch
	size, err := decoderFrameWorkResidualEventsScratchLen(sequence, events[:], 1, scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		t.Fatal(err)
	}
	if size.Plan != (internaldecoder.TileWorkPlan{SpanCount: 2, JobCount: 2, BatchCount: 1}) ||
		size.Runner.Workers != 1 ||
		size.SideData.CDEFIndexMap == 0 ||
		size.SideData.LoopFilterMap == 0 {
		t.Fatalf("events scratch size=%+v", size)
	}
	singleSize, err := decoderFrameWorkResidualEventScratchLen(sequence, &tile, 1, scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		t.Fatal(err)
	}
	if size != singleSize {
		t.Fatalf("events scratch=%+v single=%+v", size, singleSize)
	}

	pool := publicDecoderPostFilterFramePool(t, FrameFormat{
		Width:      int(frame.FrameSize.CodedWidth),
		Height:     int(frame.FrameSize.Height),
		BitDepth:   8,
		MonoChrome: true,
		Align:      64,
	}, 1)
	var refs internaldecoder.SurfaceReferences
	var state internaldecoder.FrameWorkState
	var referenceSurfaces [internalparser.InterRefsPerFrame]int
	var referenceFrames [internalparser.InterRefsPerFrame]*Frame
	var releases [internalparser.RefFrames]int
	var stats internalthreading.FrameWorkTileResidualStats
	var eventSideData decoderFrameWorkSideData
	var batchRunner decoderFrameWorkBatchResidualRunner
	var outputs [2]*Frame
	scratch := publicDecoderResidualEventScratch(size)
	eventRunner, _, err := bindDecoderFrameWorkResidualEventRunner(size, sequence, &tile, decoderFrameWorkResidualEventRuntime{
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
	result, err := eventRunner.RunEvents(SequenceHeader{}, events[:], scratch.SideData, func(ctx internaldecoder.FrameWorkPostFilterContext) error {
		postCalls++
		if ctx.Step.Kind != internaldecoder.FrameWorkStepTile || !ctx.ExecutedTileWork {
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
		result.Last.Step.Kind != internaldecoder.FrameWorkStepTile ||
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
	if _, err := shortOutputRunner.RunEvents(sequence, events[:], scratch.SideData, nil); !errors.Is(err, internalframe.ErrShortBuffer) {
		t.Fatalf("short event-list output err=%v want %v", err, internalframe.ErrShortBuffer)
	}

	noSideRunner := eventRunner
	noSideRunner.SideData = nil
	if _, err := noSideRunner.RunEvents(sequence, events[:], scratch.SideData, nil); !errors.Is(err, internaldecoder.ErrInvalidFrameWorkState) {
		t.Fatalf("missing event-list side data err=%v want %v", err, internaldecoder.ErrInvalidFrameWorkState)
	}

	shortSideScratch := scratch.SideData
	shortSideScratch.CDEFIndexMap = shortSideScratch.CDEFIndexMap[:size.SideData.CDEFIndexMap-1]
	if _, err := eventRunner.RunEvents(sequence, events[:], shortSideScratch, nil); !errors.Is(err, internalframe.ErrShortBuffer) {
		t.Fatalf("short event-list side scratch err=%v want %v", err, internalframe.ErrShortBuffer)
	}
}

func TestPublicDecoderFrameWorkResidualEventRunnerRequiresRunnerForTileList(t *testing.T) {
	var eventRunner decoderFrameWorkResidualEventRunner
	sequence := SequenceHeader{ColorConfig: internalparser.ColorConfig{BitDepth: 8, MonoChrome: true}}
	event := internaldecoder.Event{
		Kind: internaldecoder.EventTileList,
		FrameSize: FrameSize{
			CodedWidth:    64,
			UpscaledWidth: 64,
			Height:        64,
		},
		TileInfo: internalparser.TileInfo{
			Cols:       1,
			Rows:       1,
			ColStartSB: [internalparser.MaxTileCols + 1]uint16{0, 1},
			RowStartSB: [internalparser.MaxTileRows + 1]uint16{0, 1},
		},
		TileList: internalparser.TileList{
			OutputFrameWidthInTilesMinus1:  0,
			OutputFrameHeightInTilesMinus1: 0,
			TileCountMinus1:                0,
			Entries: []internalparser.TileListEntry{{
				AnchorFrameIdx:     0,
				AnchorTileRow:      0,
				AnchorTileCol:      0,
				TileDataSizeMinus1: 0,
				TileData:           []byte{0x80},
			}},
		},
	}

	result, err := eventRunner.RunEvents(sequence, []internaldecoder.Event{event}, decoderFrameWorkSideDataScratch{}, nil)
	if !errors.Is(err, internalthreading.ErrInvalidBatch) {
		t.Fatalf("RunEvents err=%v want %v", err, internalthreading.ErrInvalidBatch)
	}
	if result.Count != 0 || result.OutputCount != 0 || result.CompletedFrames != 0 {
		t.Fatalf("result after rejected tile list=%+v", result)
	}
}

func TestPublicDecoderFrameWorkResidualEventRunnerPropagatesTileListParseError(t *testing.T) {
	var eventRunner decoderFrameWorkResidualEventRunner

	_, err := eventRunner.RunEvents(SequenceHeader{}, []internaldecoder.Event{
		{Kind: internaldecoder.EventTileList, TileListErr: internalparser.ErrTileListShortEntry},
	}, decoderFrameWorkSideDataScratch{}, nil)
	if !errors.Is(err, internalparser.ErrTileListShortEntry) {
		t.Fatalf("RunEvents err=%v want %v", err, internalparser.ErrTileListShortEntry)
	}
}

func TestPublicDecoderFrameWorkResidualEventRunnerValidatesTileListLayout(t *testing.T) {
	var eventRunner decoderFrameWorkResidualEventRunner
	tileList := internalparser.TileList{
		OutputFrameWidthInTilesMinus1:  0,
		OutputFrameHeightInTilesMinus1: 0,
		TileCountMinus1:                0,
		Entries: []internalparser.TileListEntry{{
			AnchorFrameIdx:     0,
			AnchorTileRow:      1,
			AnchorTileCol:      1,
			TileDataSizeMinus1: 0,
			TileData:           []byte{0xaa},
		}},
	}
	validTiles := internalparser.TileInfo{
		Cols:       2,
		Rows:       2,
		ColStartSB: [internalparser.MaxTileCols + 1]uint16{0, 2, 4},
		RowStartSB: [internalparser.MaxTileRows + 1]uint16{0, 1, 2},
	}

	_, err := eventRunner.RunEvents(SequenceHeader{}, []internaldecoder.Event{
		{Kind: internaldecoder.EventTileList, TileList: tileList, TileInfo: validTiles},
	}, decoderFrameWorkSideDataScratch{}, nil)
	if !errors.Is(err, internalthreading.ErrInvalidBatch) {
		t.Fatalf("valid tile-list layout err=%v want %v", err, internalthreading.ErrInvalidBatch)
	}

	badAnchor := validTiles
	badAnchor.Cols = 1
	_, err = eventRunner.RunEvents(SequenceHeader{}, []internaldecoder.Event{
		{Kind: internaldecoder.EventTileList, TileList: tileList, TileInfo: badAnchor},
	}, decoderFrameWorkSideDataScratch{}, nil)
	if !errors.Is(err, internalparser.ErrTileListInvalidAnchorTile) {
		t.Fatalf("bad anchor layout err=%v want %v", err, internalparser.ErrTileListInvalidAnchorTile)
	}

	nonUniform := validTiles
	nonUniform.ColStartSB[2] = 5
	_, err = eventRunner.RunEvents(SequenceHeader{}, []internaldecoder.Event{
		{Kind: internaldecoder.EventTileList, TileList: tileList, TileInfo: nonUniform},
	}, decoderFrameWorkSideDataScratch{}, nil)
	if !errors.Is(err, internalparser.ErrTileListNonUniformTileSize) {
		t.Fatalf("non-uniform layout err=%v want %v", err, internalparser.ErrTileListNonUniformTileSize)
	}
}

func TestPublicDecoderFrameWorkResidualEventRunnerPlaysTileList(t *testing.T) {
	workerPool, err := internalthreading.NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	streamPayload := publicDecoderResidualLowOverheadStream()
	var stream internaldecoder.Stream
	var parsed [4]internaldecoder.Event
	count, err := stream.PushLowOverhead(streamPayload, parsed[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 3 || parsed[2].Kind != internaldecoder.EventTileGroup {
		t.Fatalf("parsed=%+v", parsed[:count])
	}
	sequence, ok := stream.SequenceHeader()
	if !ok {
		t.Fatal("missing sequence")
	}
	event := parsed[2]
	event.Kind = internaldecoder.EventTileList
	event.Type = internalobu.TypeTileList
	event.Unit = internalobu.Unit{}
	event.TileGroup = internalparser.TileGroup{}
	event.TileList = internalparser.TileList{
		OutputFrameWidthInTilesMinus1:  0,
		OutputFrameHeightInTilesMinus1: 0,
		TileCountMinus1:                0,
		Entries: []internalparser.TileListEntry{{
			AnchorFrameIdx:     0,
			AnchorTileRow:      0,
			AnchorTileCol:      0,
			TileDataSizeMinus1: 0,
			TileData:           []byte{0x80},
		}},
	}

	var scratchSpans [1]internalparser.TileSpan
	var scratchJobs [1]internaltile.Job
	var scratchBatches [1]internalthreading.Batch
	size, err := decoderFrameWorkResidualEventScratchLen(sequence, &event, 1, scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		t.Fatal(err)
	}
	if size.Outputs != 1 ||
		size.Plan != (internaldecoder.TileWorkPlan{SpanCount: 1, JobCount: 1, BatchCount: 1}) ||
		size.Runner.Workers != 1 ||
		size.SideData != (decoderFrameWorkSideDataScratchSize{}) {
		t.Fatalf("tile-list scratch size=%+v", size)
	}
	runnerSize, runnerPlan, err := decoderFrameWorkResidualEventRunnerScratchLen(sequence, &event, 1, scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		t.Fatal(err)
	}
	if runnerSize != size.Runner || runnerPlan != size.Plan {
		t.Fatalf("tile-list runner scratch=%+v/%+v want %+v/%+v", runnerSize, runnerPlan, size.Runner, size.Plan)
	}
	geom, err := internalparser.TileListOutputGeometryForGrid(event.TileList, event.TileInfo, event.SequenceHeader.Use128x128Superblock)
	if err != nil {
		t.Fatal(err)
	}

	format, err := tileListOutputFrameFormat(sequence, geom, 64)
	if err != nil {
		t.Fatal(err)
	}
	pool := publicDecoderPostFilterFramePool(t, format, 3)
	anchorSurface, anchorFrame, err := pool.Acquire()
	if err != nil {
		t.Fatal(err)
	}
	publicFillDecoderBlockCoeffFrame(anchorFrame, 0x21)
	var refs internaldecoder.SurfaceReferences
	var releases [internalparser.RefFrames]int
	if _, err := refs.Refresh(1<<0, anchorSurface, releases[:]); err != nil {
		t.Fatal(err)
	}

	var state internaldecoder.FrameWorkState
	var batchRunner decoderFrameWorkBatchResidualRunner
	var outputs [1]*Frame
	eventRunner, _, err := bindDecoderFrameWorkResidualEventRunner(size, sequence, &event, decoderFrameWorkResidualEventRuntime{
		State:      &state,
		Refs:       &refs,
		FramePool:  &pool,
		Align:      64,
		Releases:   releases[:],
		WorkerPool: workerPool,
		Outputs:    outputs[:],
	}, publicDecoderResidualEventScratch(size), &batchRunner)
	if err != nil {
		t.Fatal(err)
	}

	result, err := eventRunner.RunEvents(sequence, []internaldecoder.Event{event}, decoderFrameWorkSideDataScratch{}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if result.Count != 1 ||
		result.ExecutedTileWork != 1 ||
		result.CompletedFrames != 0 ||
		result.OutputCount != 1 ||
		len(result.Outputs) != 1 ||
		result.Last.Output == nil ||
		result.Last.Step.Kind != internaldecoder.FrameWorkStepTile ||
		result.Last.Run != (internaldecoder.FrameWorkStepResult{ExecutedTileWork: true}) ||
		result.Outputs[0] != result.Last.Output ||
		outputs[0] != result.Last.Output ||
		state.Active() {
		t.Fatalf("tile-list result=%+v active=%v", result, state.Active())
	}
	if surface, ok := refs.ReferenceSlot(0); !ok || surface != anchorSurface {
		t.Fatalf("tile-list mutated reference slot: surface=%d ok=%v want %d", surface, ok, anchorSurface)
	}
}

func TestPublicPlanDecoderTileListEntryWork(t *testing.T) {
	list := internalparser.TileList{
		OutputFrameWidthInTilesMinus1:  1,
		OutputFrameHeightInTilesMinus1: 0,
		TileCountMinus1:                1,
		Entries: []internalparser.TileListEntry{
			{
				AnchorFrameIdx:     0,
				AnchorTileRow:      0,
				AnchorTileCol:      1,
				TileDataSizeMinus1: 2,
				TileData:           []byte{0xaa, 0xbb, 0xcc},
			},
			{
				AnchorFrameIdx:     0,
				AnchorTileRow:      1,
				AnchorTileCol:      0,
				TileDataSizeMinus1: 1,
				TileData:           []byte{0xdd, 0xee},
			},
		},
	}
	tiles := internalparser.TileInfo{
		Cols:                2,
		Rows:                2,
		SBCols:              4,
		SBRows:              4,
		RefreshContext:      true,
		ContextUpdateTileID: 2,
		ColStartSB:          [internalparser.MaxTileCols + 1]uint16{0, 2, 4},
		RowStartSB:          [internalparser.MaxTileRows + 1]uint16{0, 2, 4},
	}
	var spans [1]internalparser.TileSpan
	var jobs [1]internaltile.Job
	var batches [1]internalthreading.Batch

	plan, err := internaldecoder.PlanTileListEntryWork(list, tiles, 1, 1, spans[:], jobs[:], batches[:])
	if err != nil {
		t.Fatalf("PlanDecoderTileListEntryWork: %v", err)
	}
	if plan.SpanCount != 1 || plan.JobCount != 1 || plan.BatchCount != 1 {
		t.Fatalf("plan=%+v want 1/1/1", plan)
	}
	if spans[0] != (internalparser.TileSpan{Tile: 2, Row: 1, Col: 0, Offset: 0, Size: 2}) {
		t.Fatalf("span=%+v", spans[0])
	}
	if jobs[0] != (internaltile.Job{
		Tile:                2,
		Row:                 1,
		Col:                 0,
		SBX:                 0,
		SBY:                 2,
		SBCols:              2,
		SBRows:              2,
		Offset:              0,
		Size:                2,
		LastRow:             true,
		UpdatesFrameContext: true,
	}) {
		t.Fatalf("job=%+v", jobs[0])
	}
	if batches[0] != (internalthreading.Batch{FirstJob: 0, Count: 1, Units: 4, Worker: 0, FirstTile: 2, LastTile: 2}) {
		t.Fatalf("batch=%+v", batches[0])
	}
	payload, err := jobs[0].Payload(list.Entries[1].TileData)
	if err != nil {
		t.Fatalf("job payload: %v", err)
	}
	if len(payload) != 2 || payload[0] != 0xdd || payload[1] != 0xee {
		t.Fatalf("payload=% x", payload)
	}
}

func TestPublicPlanDecoderTileListEntryWorkRejectsInvalid(t *testing.T) {
	list := internalparser.TileList{
		OutputFrameWidthInTilesMinus1:  0,
		OutputFrameHeightInTilesMinus1: 0,
		TileCountMinus1:                0,
		Entries: []internalparser.TileListEntry{{
			AnchorFrameIdx:     0,
			AnchorTileRow:      0,
			AnchorTileCol:      0,
			TileDataSizeMinus1: 0,
			TileData:           []byte{0xaa},
		}},
	}
	tiles := internalparser.TileInfo{
		Cols:       1,
		Rows:       1,
		SBCols:     2,
		SBRows:     2,
		ColStartSB: [internalparser.MaxTileCols + 1]uint16{0, 2},
		RowStartSB: [internalparser.MaxTileRows + 1]uint16{0, 2},
	}
	var spans [1]internalparser.TileSpan
	var jobs [1]internaltile.Job
	var batches [1]internalthreading.Batch

	tests := []struct {
		name  string
		list  internalparser.TileList
		tiles internalparser.TileInfo
		index int
		spans []internalparser.TileSpan
		want  error
	}{
		{name: "entry-index", list: list, tiles: tiles, index: 1, spans: spans[:], want: internalparser.ErrTileListInvalidTileCount},
		{name: "short-spans", list: list, tiles: tiles, index: 0, spans: spans[:0], want: internaldecoder.ErrInvalidTileWork},
		{name: "short-tile-data", list: func() internalparser.TileList {
			short := list
			short.Entries = append([]internalparser.TileListEntry(nil), list.Entries...)
			short.Entries[0].TileDataSizeMinus1 = 2
			short.Entries[0].TileData = []byte{0xaa, 0xbb}
			return short
		}(), tiles: tiles, index: 0, spans: spans[:], want: internalparser.ErrTileListShortTileData},
		{name: "long-tile-data", list: func() internalparser.TileList {
			long := list
			long.Entries = append([]internalparser.TileListEntry(nil), list.Entries...)
			long.Entries[0].TileDataSizeMinus1 = 0
			long.Entries[0].TileData = []byte{0xaa, 0xbb}
			return long
		}(), tiles: tiles, index: 0, spans: spans[:], want: internalparser.ErrTileListTrailingBytes},
		{name: "non-uniform-layout", list: list, tiles: func() internalparser.TileInfo {
			nonUniform := tiles
			nonUniform.Cols = 2
			nonUniform.SBCols = 5
			nonUniform.ColStartSB = [internalparser.MaxTileCols + 1]uint16{0, 2, 5}
			return nonUniform
		}(), index: 0, spans: spans[:], want: internalparser.ErrTileListNonUniformTileSize},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, err := internaldecoder.PlanTileListEntryWork(tc.list, tc.tiles, tc.index, 1, tc.spans, jobs[:], batches[:])
			if !errors.Is(err, tc.want) {
				t.Fatalf("PlanDecoderTileListEntryWork err=%v want %v", err, tc.want)
			}
		})
	}
}

func TestPublicPlanDecoderTileListEntryDecode(t *testing.T) {
	event := publicTileListEntryDecodeEvent()
	anchors := make([]*Frame, internalparser.TileListMaxExternalReferences)
	anchor := &Frame{Format: FrameFormat{Width: 256, Height: 256, BitDepth: 8, SubsamplingX: true, SubsamplingY: true}}
	anchors[7] = anchor
	references := []*Frame{nil, &Frame{Format: FrameFormat{Width: 1, Height: 1, BitDepth: 8}}}
	var spans [1]internalparser.TileSpan
	var jobs [1]internaltile.Job
	var batches [1]internalthreading.Batch

	plan, err := internaldecoder.PlanTileListEntryDecode(&event, 1, 3, anchors, references, 1, spans[:], jobs[:], batches[:])
	if err != nil {
		t.Fatalf("PlanDecoderTileListEntryDecode: %v", err)
	}
	if plan.Event.Kind != internaldecoder.EventTileGroup ||
		plan.Step.Kind != internaldecoder.FrameWorkStepTile ||
		plan.Step.Tile.Surface != 3 ||
		plan.ReferenceCount != 1 ||
		plan.Step.Tile.ReferenceCount != 1 ||
		plan.AnchorFrameIdx != 7 ||
		!bytes.Equal(plan.Payload, []byte{0xdd, 0xee}) ||
		!bytes.Equal(plan.Event.Unit.Payload, []byte{0xdd, 0xee}) {
		t.Fatalf("decode plan=%+v payload=% x", plan, plan.Event.Unit.Payload)
	}
	if references[0] != anchor {
		t.Fatalf("LAST reference was not bound to anchor frame: %p want %p", references[0], anchor)
	}
	if plan.Geometry.OutputFrameWidth != 256 || plan.Geometry.OutputFrameHeight != 128 {
		t.Fatalf("geometry=%+v", plan.Geometry)
	}
	if plan.Region != (internalparser.TileListTileRegion{SourceX: 0, SourceY: 128, DestX: 128, DestY: 0, Width: 128, Height: 128}) {
		t.Fatalf("region=%+v", plan.Region)
	}
	if spans[0] != (internalparser.TileSpan{Tile: 2, Row: 1, Col: 0, Offset: 0, Size: 2}) {
		t.Fatalf("span=%+v", spans[0])
	}
	if jobs[0] != (internaltile.Job{
		Tile:                2,
		Row:                 1,
		Col:                 0,
		SBX:                 0,
		SBY:                 2,
		SBCols:              2,
		SBRows:              2,
		Offset:              0,
		Size:                2,
		LastRow:             true,
		UpdatesFrameContext: true,
	}) {
		t.Fatalf("job=%+v", jobs[0])
	}

	size, scratchPlan, err := decoderTileListEntryDecodeRunnerScratchLen(&event, 1, 1, spans[:], jobs[:], batches[:])
	if err != nil {
		t.Fatalf("DecoderTileListEntryDecodeRunnerScratchLen: %v", err)
	}
	if scratchPlan != plan.Step.Tile.Tile {
		t.Fatalf("scratch plan=%+v want %+v", scratchPlan, plan.Step.Tile.Tile)
	}
	if size.Batch.Int32Scratch == 0 || size.Batch.ResidualScratch == 0 || size.Workers != 1 {
		t.Fatalf("scratch size=%+v", size)
	}
}

func TestPublicPlanDecoderTileListEntryDecodeRejectsInvalid(t *testing.T) {
	event := publicTileListEntryDecodeEvent()
	anchors := make([]*Frame, internalparser.TileListMaxExternalReferences)
	anchor := &Frame{Format: FrameFormat{Width: 256, Height: 256, BitDepth: 8, SubsamplingX: true, SubsamplingY: true}}
	anchors[7] = anchor
	stale := &Frame{}
	references := []*Frame{stale}
	var spans [1]internalparser.TileSpan
	var jobs [1]internaltile.Job
	var batches [1]internalthreading.Batch

	tests := []struct {
		name    string
		event   internaldecoder.Event
		index   int
		surface int
		anchors []*Frame
		refs    []*Frame
		want    error
	}{
		{name: "not-tile-list", event: internaldecoder.Event{Kind: internaldecoder.EventFrame}, index: 0, surface: 0, anchors: anchors, refs: references, want: internaldecoder.ErrInvalidSurfaceEvent},
		{name: "parse-error", event: func() internaldecoder.Event {
			bad := event
			bad.TileListErr = internalparser.ErrTileListShortEntry
			return bad
		}(), index: 0, surface: 0, anchors: anchors, refs: references, want: internalparser.ErrTileListShortEntry},
		{name: "negative-surface", event: event, index: 0, surface: -1, anchors: anchors, refs: references, want: internaldecoder.ErrInvalidSurfaceReference},
		{name: "short-reference-table", event: event, index: 0, surface: 0, anchors: anchors, refs: nil, want: internaldecoder.ErrSurfaceReferenceBufferTooSmall},
		{name: "missing-anchor", event: event, index: 1, surface: 0, anchors: anchors[:7], refs: references, want: internalparser.ErrTileListInvalidAnchorIndex},
		{name: "nil-anchor", event: event, index: 1, surface: 0, anchors: append([]*Frame(nil), anchors...), refs: references, want: internalparser.ErrTileListInvalidAnchorIndex},
		{name: "bad-index", event: event, index: 2, surface: 0, anchors: anchors, refs: references, want: internalparser.ErrTileListInvalidTileCount},
	}
	tests[5].anchors[7] = nil

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			references[0] = stale
			_, err := internaldecoder.PlanTileListEntryDecode(&tc.event, tc.index, tc.surface, tc.anchors, tc.refs, 1, spans[:], jobs[:], batches[:])
			if !errors.Is(err, tc.want) {
				t.Fatalf("PlanDecoderTileListEntryDecode err=%v want %v", err, tc.want)
			}
			if references[0] != stale {
				t.Fatalf("references mutated on error: %p want %p", references[0], stale)
			}
		})
	}

	if _, _, err := decoderTileListEntryDecodeRunnerScratchLen(&internaldecoder.Event{Kind: internaldecoder.EventFrame}, 0, 1, spans[:], jobs[:], batches[:]); !errors.Is(err, internaldecoder.ErrInvalidSurfaceEvent) {
		t.Fatalf("scratch non-tile-list err=%v want %v", err, internaldecoder.ErrInvalidSurfaceEvent)
	}
	parseErrEvent := event
	parseErrEvent.TileListErr = internalparser.ErrTileListShortEntry
	if _, _, err := decoderTileListEntryDecodeRunnerScratchLen(&parseErrEvent, 0, 1, spans[:], jobs[:], batches[:]); !errors.Is(err, internalparser.ErrTileListShortEntry) {
		t.Fatalf("scratch parse err=%v want %v", err, internalparser.ErrTileListShortEntry)
	}
}

func publicTileListEntryDecodeEvent() internaldecoder.Event {
	return internaldecoder.Event{
		Kind: internaldecoder.EventTileList,
		SequenceHeader: SequenceHeader{
			ColorConfig: internalparser.ColorConfig{BitDepth: 8, SubsamplingX: true, SubsamplingY: true},
		},
		FrameHeader: internalparser.FrameHeaderPrefix{FrameType: internalparser.FrameTypeInter},
		FrameSize: FrameSize{
			CodedWidth:        256,
			UpscaledWidth:     256,
			Height:            256,
			RenderWidth:       256,
			RenderHeight:      256,
			RefreshFrameFlags: 0,
		},
		TileInfo: internalparser.TileInfo{
			Cols:                2,
			Rows:                2,
			SBCols:              4,
			SBRows:              4,
			RefreshContext:      true,
			ContextUpdateTileID: 2,
			ColStartSB:          [internalparser.MaxTileCols + 1]uint16{0, 2, 4},
			RowStartSB:          [internalparser.MaxTileRows + 1]uint16{0, 2, 4},
		},
		TileList: internalparser.TileList{
			OutputFrameWidthInTilesMinus1:  1,
			OutputFrameHeightInTilesMinus1: 0,
			TileCountMinus1:                1,
			Entries: []internalparser.TileListEntry{
				{
					AnchorFrameIdx:     0,
					AnchorTileRow:      0,
					AnchorTileCol:      1,
					TileDataSizeMinus1: 2,
					TileData:           []byte{0xaa, 0xbb, 0xcc},
				},
				{
					AnchorFrameIdx:     7,
					AnchorTileRow:      1,
					AnchorTileCol:      0,
					TileDataSizeMinus1: 1,
					TileData:           []byte{0xdd, 0xee},
				},
			},
		},
	}
}

func TestPublicDecoderFrameWorkResidualEventRunnerShowExistingOutput(t *testing.T) {
	sequence := SequenceHeader{ColorConfig: internalparser.ColorConfig{
		BitDepth:   8,
		MonoChrome: true,
	}}
	event := internaldecoder.Event{
		Kind: internaldecoder.EventExistingFrame,
		FrameHeader: internalparser.FrameHeaderPrefix{
			ShowExistingFrame: true,
			ExistingFrameIdx:  0,
		},
		ExistingFrame: internalparser.ReferenceFrame{FrameType: internalparser.FrameTypeInter},
		FrameSize: FrameSize{
			CodedWidth:    16,
			UpscaledWidth: 16,
			Height:        16,
		},
	}
	var scratchSpans [1]internalparser.TileSpan
	var scratchJobs [1]internaltile.Job
	var scratchBatches [1]internalthreading.Batch
	size, err := decoderFrameWorkResidualEventScratchLen(sequence, &event, 1, scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		t.Fatal(err)
	}
	if size.Runner != (decoderFrameWorkBatchResidualRunnerScratchSize{}) ||
		size.SideData != (decoderFrameWorkSideDataScratchSize{}) ||
		size.Plan != (internaldecoder.TileWorkPlan{}) ||
		size.Outputs != 1 {
		t.Fatalf("show-existing scratch size=%+v", size)
	}
	listSize, err := decoderFrameWorkResidualEventsScratchLen(sequence, []internaldecoder.Event{event}, 1, scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		t.Fatal(err)
	}
	if listSize != size {
		t.Fatalf("show-existing list scratch=%+v single=%+v", listSize, size)
	}

	pool := publicDecoderPostFilterFramePool(t, FrameFormat{
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
	var refs internaldecoder.SurfaceReferences
	var releases [internalparser.RefFrames]int
	if _, err := refs.Refresh(1<<0, referenceSlot, releases[:]); err != nil {
		t.Fatal(err)
	}

	var state internaldecoder.FrameWorkState
	var outputs [1]*Frame
	eventRunner, _, err := bindDecoderFrameWorkResidualEventRunner(size, sequence, &event, decoderFrameWorkResidualEventRuntime{
		State:     &state,
		Refs:      &refs,
		FramePool: &pool,
		Releases:  releases[:],
		Outputs:   outputs[:],
	}, publicDecoderResidualEventScratch(size), nil)
	if err != nil {
		t.Fatal(err)
	}

	result, err := eventRunner.RunEvents(sequence, []internaldecoder.Event{event}, decoderFrameWorkSideDataScratch{}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if result.Count != 1 ||
		result.ExecutedTileWork != 0 ||
		result.CompletedFrames != 0 ||
		result.OutputCount != 1 ||
		len(result.Outputs) != 1 ||
		result.Last.Step.Kind != internaldecoder.FrameWorkStepShowExisting ||
		result.Last.Output != referenceFrame ||
		outputs[0] != referenceFrame ||
		result.Outputs[0] != referenceFrame ||
		state.Active() {
		t.Fatalf("show-existing result=%+v output=%p/%p active=%v", result, result.Last.Output, referenceFrame, state.Active())
	}

	shortOutputRunner := eventRunner
	shortOutputRunner.Outputs = outputs[:0]
	if _, err := shortOutputRunner.RunEvents(sequence, []internaldecoder.Event{event}, decoderFrameWorkSideDataScratch{}, nil); !errors.Is(err, internalframe.ErrShortBuffer) {
		t.Fatalf("short show-existing output err=%v want %v", err, internalframe.ErrShortBuffer)
	}
}

func TestPublicDecoderFrameWorkResidualEventRunnerFrameHeaderWithoutBatchRunner(t *testing.T) {
	payload := publicDecoderResidualLowOverheadStream()
	var probeStream internaldecoder.Stream
	var events [4]internaldecoder.Event
	count, err := probeStream.PushLowOverhead(payload, events[:])
	if err != nil {
		t.Fatal(err)
	}
	if count < 2 || events[1].Kind != internaldecoder.EventFrameHeader {
		t.Fatalf("events=%+v", events[:count])
	}
	sequence, ok := probeStream.SequenceHeader()
	if !ok {
		t.Fatal("probe stream missing sequence")
	}
	var scratchSpans [1]internalparser.TileSpan
	var scratchJobs [1]internaltile.Job
	var scratchBatches [1]internalthreading.Batch
	size, err := decoderFrameWorkResidualEventScratchLen(sequence, &events[1], 1, scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		t.Fatal(err)
	}
	if size.Runner != (decoderFrameWorkBatchResidualRunnerScratchSize{}) ||
		size.Plan != (internaldecoder.TileWorkPlan{}) ||
		size.Outputs != 0 ||
		size.SideData.CDEFIndexMap == 0 ||
		size.SideData.LoopFilterMap == 0 {
		t.Fatalf("frame-header scratch size=%+v", size)
	}

	pool := publicDecoderPostFilterFramePool(t, FrameFormat{
		Width:        int(events[1].FrameSize.CodedWidth),
		Height:       int(events[1].FrameSize.Height),
		BitDepth:     8,
		MonoChrome:   true,
		SubsamplingX: true,
		SubsamplingY: true,
		Align:        64,
	}, 1)
	var state internaldecoder.FrameWorkState
	var refs internaldecoder.SurfaceReferences
	var side decoderFrameWorkSideData
	var stats internalthreading.FrameWorkTileResidualStats
	eventRunner, _, err := bindDecoderFrameWorkResidualEventRunner(size, sequence, &events[1], decoderFrameWorkResidualEventRuntime{
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
		result.Last.Step.Kind != internaldecoder.FrameWorkStepBegin ||
		!state.Active() ||
		stats != (internalthreading.FrameWorkTileResidualStats{}) {
		t.Fatalf("frame-header result=%+v active=%v stats=%+v", result, state.Active(), stats)
	}
}

func TestPublicDecoderFrameWorkResidualEventRunnerRunEventsAllocs(t *testing.T) {
	workerPool, err := internalthreading.NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	sequence := SequenceHeader{
		EnableCDEF: true,
		ColorConfig: internalparser.ColorConfig{
			BitDepth:   8,
			MonoChrome: true,
		},
	}
	frame := publicDecoderResidualRunnerFrameEvent()
	events := [...]internaldecoder.Event{
		{
			Kind:                  internaldecoder.EventSequenceHeader,
			SequenceHeader:        sequence,
			NewCodedVideoSequence: true,
		},
		frame,
	}
	var scratchSpans [2]internalparser.TileSpan
	var scratchJobs [2]internaltile.Job
	var scratchBatches [1]internalthreading.Batch
	size, err := decoderFrameWorkResidualEventsScratchLen(sequence, events[:], 1, scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		t.Fatal(err)
	}
	pool := publicDecoderPostFilterFramePool(t, FrameFormat{
		Width:      int(frame.FrameSize.CodedWidth),
		Height:     int(frame.FrameSize.Height),
		BitDepth:   8,
		MonoChrome: true,
		Align:      64,
	}, 1)
	var refs internaldecoder.SurfaceReferences
	var state internaldecoder.FrameWorkState
	var referenceSurfaces [internalparser.InterRefsPerFrame]int
	var referenceFrames [internalparser.InterRefsPerFrame]*Frame
	var releases [internalparser.RefFrames]int
	var stats internalthreading.FrameWorkTileResidualStats
	var eventSideData decoderFrameWorkSideData
	var batchRunner decoderFrameWorkBatchResidualRunner
	var outputs [1]*Frame
	scratch := publicDecoderResidualEventScratch(size)
	eventRunner, _, err := bindDecoderFrameWorkResidualEventRunner(size, sequence, &frame, decoderFrameWorkResidualEventRuntime{
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
			err = internalthreading.ErrInvalidBatch
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
	workerPool, err := internalthreading.NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	payload := publicDecoderResidualLowOverheadStream()
	var probeStream internaldecoder.Stream
	var probeEvents [4]internaldecoder.Event
	count, err := probeStream.PushLowOverhead(payload, probeEvents[:])
	if err != nil {
		t.Fatal(err)
	}
	sequence, ok := probeStream.SequenceHeader()
	if !ok {
		t.Fatal("probe stream missing sequence")
	}
	var scratchSpans [1]internalparser.TileSpan
	var scratchJobs [1]internaltile.Job
	var scratchBatches [1]internalthreading.Batch
	size, err := decoderFrameWorkResidualEventsScratchLen(sequence, probeEvents[:count], 1, scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		t.Fatal(err)
	}

	pool := publicDecoderPostFilterFramePool(t, FrameFormat{
		Width:        int(probeEvents[count-1].FrameSize.CodedWidth),
		Height:       int(probeEvents[count-1].FrameSize.Height),
		BitDepth:     8,
		MonoChrome:   true,
		SubsamplingX: true,
		SubsamplingY: true,
		Align:        64,
	}, 2)
	var stream internaldecoder.Stream
	var refs internaldecoder.SurfaceReferences
	var state internaldecoder.FrameWorkState
	var referenceSurfaces [internalparser.InterRefsPerFrame]int
	var referenceFrames [internalparser.InterRefsPerFrame]*Frame
	var releases [internalparser.RefFrames]int
	var stats internalthreading.FrameWorkTileResidualStats
	var side decoderFrameWorkSideData
	var batchRunner decoderFrameWorkBatchResidualRunner
	var outputs [2]*Frame
	scratch := publicDecoderResidualEventScratch(size)
	eventRunner, _, err := bindDecoderFrameWorkResidualEventRunner(size, sequence, &probeEvents[count-1], decoderFrameWorkResidualEventRuntime{
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
	runner := decoderFrameWorkResidualStreamRunner{
		Stream:          &stream,
		EventRunner:     eventRunner,
		Events:          make([]internaldecoder.Event, len(probeEvents)),
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
	var stream internaldecoder.Stream
	var prime []byte
	keyFrame := append([]byte{}, publicDecoderResidualShownKeyFrameHeaderPayload()...)
	keyFrame = append(keyFrame, 0xaa)
	interFrame := append([]byte{}, publicDecoderResidualInterFrameHeaderPayload()...)
	interFrame = append(interFrame, 0xbb)
	prime = appendOBUForTest(prime, internalobu.TypeSequenceHeader, publicDecoderResidualRealtimeSequenceHeaderPayload())
	prime = appendOBUForTest(prime, internalobu.TypeFrame, keyFrame)
	prime = appendOBUForTest(prime, internalobu.TypeFrame, interFrame)
	var primeEvents [3]internaldecoder.Event
	if count, err := stream.PushLowOverhead(prime, primeEvents[:]); err != nil {
		t.Fatal(err)
	} else if count != len(primeEvents) || primeEvents[2].FrameHeader.FrameType != internalparser.FrameTypeInter {
		t.Fatalf("prime count=%d events=%+v", count, primeEvents)
	}

	var showExisting []byte
	showExisting = appendOBUForTest(showExisting, internalobu.TypeFrameHeader, publicDecoderResidualShowExistingFrameHeaderPayload(0))
	var events [1]internaldecoder.Event
	var scratchSpans [1]internalparser.TileSpan
	var scratchJobs [1]internaltile.Job
	var scratchBatches [1]internalthreading.Batch
	size, err := decoderFrameWorkResidualLowOverheadStreamScratchLen(stream, showExisting, 1, events[:], scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		t.Fatal(err)
	}
	if size.Events != 1 ||
		size.Event.Outputs != 1 ||
		size.Event.Runner != (decoderFrameWorkBatchResidualRunnerScratchSize{}) ||
		size.Event.SideData != (decoderFrameWorkSideDataScratchSize{}) ||
		size.Event.Plan != (internaldecoder.TileWorkPlan{}) ||
		events[0].Kind != internaldecoder.EventExistingFrame ||
		events[0].ExistingFrame.FrameType != internalparser.FrameTypeInter {
		t.Fatalf("show-existing stream scratch size=%+v event=%+v", size, events[0])
	}
	sequence, ok := stream.SequenceHeader()
	if !ok {
		t.Fatal("stream missing sequence")
	}

	pool := publicDecoderPostFilterFramePool(t, FrameFormat{
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
	var refs internaldecoder.SurfaceReferences
	var releases [internalparser.RefFrames]int
	if _, err := refs.Refresh(1<<0, referenceSlot, releases[:]); err != nil {
		t.Fatal(err)
	}

	var state internaldecoder.FrameWorkState
	var stats internalthreading.FrameWorkTileResidualStats
	eventRunner, _, err := bindDecoderFrameWorkResidualEventRunner(size.Event, sequence, &events[0], decoderFrameWorkResidualEventRuntime{
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
	runner, err := bindDecoderFrameWorkResidualStreamRunner(size, &stream, eventRunner, streamScratch)
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
		result.Run.Last.Step.Kind != internaldecoder.FrameWorkStepShowExisting ||
		result.Run.Last.Output != referenceFrame ||
		result.Run.Outputs[0] != referenceFrame ||
		streamScratch.Outputs[0] != referenceFrame ||
		state.Active() ||
		stats != (internalthreading.FrameWorkTileResidualStats{}) {
		t.Fatalf("show-existing stream result=%+v output=%p/%p active=%v stats=%+v", result, streamScratch.Outputs[0], referenceFrame, state.Active(), stats)
	}
}

func TestPublicDecoderFrameWorkResidualStreamRunnerReset(t *testing.T) {
	var stream internaldecoder.Stream
	var events [4]internaldecoder.Event
	if count, err := stream.PushLowOverhead(publicDecoderResidualLowOverheadStream(), events[:]); err != nil {
		t.Fatal(err)
	} else if count != 3 {
		t.Fatalf("count=%d events=%+v", count, events[:count])
	}
	runner := decoderFrameWorkResidualStreamRunner{
		Stream:  &stream,
		RTPUsed: 7,
	}
	if !stream.HasSequenceHeader() {
		t.Fatal("stream missing sequence before reset")
	}
	frameHeader := publicDecoderResidualRTPElement(internalobu.TypeFrameHeader, publicDecoderResidualFrameHeaderPayload())
	var packet [8]byte
	n, _, more, err := PutRTPFragment(packet[:], frameHeader, 0, 2, false)
	if err != nil {
		t.Fatal(err)
	}
	if !more {
		t.Fatal("expected frame header to need a second fragment")
	}
	var rtpBuffer [128]byte
	var rtpSpans [1]RTPObuSpan
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

	var nilRunner *decoderFrameWorkResidualStreamRunner
	if err := nilRunner.Reset(); !errors.Is(err, internaldecoder.ErrInvalidFrameWorkState) {
		t.Fatalf("nil reset err=%v want %v", err, internaldecoder.ErrInvalidFrameWorkState)
	}
	if err := nilRunner.ResetRTP(); !errors.Is(err, internaldecoder.ErrInvalidFrameWorkState) {
		t.Fatalf("nil rtp reset err=%v want %v", err, internaldecoder.ErrInvalidFrameWorkState)
	}
}

func TestPublicDecoderFrameWorkResidualEventsBindPlan(t *testing.T) {
	payload := publicDecoderResidualLowOverheadStream()
	var stream internaldecoder.Stream
	var events [4]internaldecoder.Event
	count, err := stream.PushLowOverhead(payload, events[:])
	if err != nil {
		t.Fatal(err)
	}
	plan := decoderFrameWorkResidualEventsBindPlan(SequenceHeader{}, events[:count])
	if !plan.HasEvent() ||
		plan.EventIndex != count-1 ||
		plan.Event.Kind != internaldecoder.EventTileGroup ||
		plan.Sequence.ColorConfig.BitDepth != 8 ||
		!plan.Sequence.ColorConfig.MonoChrome {
		t.Fatalf("low-overhead bind plan=%+v count=%d", plan, count)
	}

	var realtime internaldecoder.Stream
	var prime []byte
	keyFrame := append([]byte{}, publicDecoderResidualShownKeyFrameHeaderPayload()...)
	keyFrame = append(keyFrame, 0xaa)
	interFrame := append([]byte{}, publicDecoderResidualInterFrameHeaderPayload()...)
	interFrame = append(interFrame, 0xbb)
	prime = appendOBUForTest(prime, internalobu.TypeSequenceHeader, publicDecoderResidualRealtimeSequenceHeaderPayload())
	prime = appendOBUForTest(prime, internalobu.TypeFrame, keyFrame)
	prime = appendOBUForTest(prime, internalobu.TypeFrame, interFrame)
	var primeEvents [3]internaldecoder.Event
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
	showExisting = appendOBUForTest(showExisting, internalobu.TypeFrameHeader, publicDecoderResidualShowExistingFrameHeaderPayload(0))
	var outputEvents [1]internaldecoder.Event
	outputCount, err := realtime.PushLowOverhead(showExisting, outputEvents[:])
	if err != nil {
		t.Fatal(err)
	}
	outputPlan := decoderFrameWorkResidualEventsBindPlan(sequence, outputEvents[:outputCount])
	if !outputPlan.HasEvent() ||
		outputPlan.EventIndex != 0 ||
		outputPlan.Event.Kind != internaldecoder.EventExistingFrame ||
		outputPlan.Event.ExistingFrame.FrameType != internalparser.FrameTypeInter ||
		outputPlan.Sequence.ColorConfig.BitDepth != 8 {
		t.Fatalf("show-existing bind plan=%+v count=%d", outputPlan, outputCount)
	}

	sequenceOnly := SequenceHeader{ColorConfig: internalparser.ColorConfig{
		BitDepth:   10,
		MonoChrome: true,
	}}
	noEventPlan := decoderFrameWorkResidualEventsBindPlan(SequenceHeader{}, []internaldecoder.Event{
		{Kind: internaldecoder.EventSequenceHeader, SequenceHeader: sequenceOnly},
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
	var stream internaldecoder.Stream
	var events [4]internaldecoder.Event
	var rtpBuffer [256]byte
	var rtpSpans [4]RTPObuSpan
	var scratchSpans [1]internalparser.TileSpan
	var scratchJobs [1]internaltile.Job
	var scratchBatches [1]internalthreading.Batch

	lowPlan, err := decoderFrameWorkResidualLowOverheadStreamPlan(stream, lowOverhead, 1, events[:], scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		t.Fatal(err)
	}
	if !lowPlan.HasEvent() ||
		lowPlan.Size.Events != 3 ||
		lowPlan.Size.Event.Outputs != 1 ||
		lowPlan.Bind.EventIndex != 2 ||
		lowPlan.Bind.Event.Kind != internaldecoder.EventTileGroup ||
		lowPlan.Bind.Sequence.ColorConfig.BitDepth != 8 {
		t.Fatalf("low-overhead stream plan=%+v", lowPlan)
	}
	lowSize, err := decoderFrameWorkResidualLowOverheadStreamScratchLen(stream, lowOverhead, 1, events[:], scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		t.Fatal(err)
	}
	if lowPlan.Size != lowSize {
		t.Fatalf("low-overhead plan size=%+v scratch size=%+v", lowPlan.Size, lowSize)
	}

	lowBatchPlan, err := decoderFrameWorkResidualLowOverheadStreamsPlan(stream, lowOverheads[:], 1, events[:], scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		t.Fatal(err)
	}
	if !lowBatchPlan.HasEvent() ||
		lowBatchPlan.Size.Events != 3 ||
		lowBatchPlan.Size.Event.Outputs != 2 ||
		lowBatchPlan.Bind.EventIndex != 1 ||
		lowBatchPlan.Bind.Event.Kind != internaldecoder.EventTileGroup {
		t.Fatalf("low-overhead stream batch plan=%+v", lowBatchPlan)
	}
	lowBatchSize, err := decoderFrameWorkResidualLowOverheadStreamsScratchLen(stream, lowOverheads[:], 1, events[:], scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		t.Fatal(err)
	}
	if lowBatchPlan.Size != lowBatchSize {
		t.Fatalf("low-overhead batch plan size=%+v scratch size=%+v", lowBatchPlan.Size, lowBatchSize)
	}

	rtpPlan, err := decoderFrameWorkResidualRTPPayloadStreamPlan(stream, 0, rtpPayload, 1, rtpBuffer[:], rtpSpans[:], events[:], scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		t.Fatal(err)
	}
	if !rtpPlan.HasEvent() ||
		rtpPlan.Size.Events != 3 ||
		rtpPlan.Size.RTPSpans != 3 ||
		rtpPlan.Size.Event.Outputs != 1 ||
		rtpPlan.Bind.EventIndex != 2 ||
		rtpPlan.Bind.Event.Kind != internaldecoder.EventTileGroup {
		t.Fatalf("rtp stream plan=%+v", rtpPlan)
	}

	batchPlan, err := decoderFrameWorkResidualRTPPayloadsStreamPlan(stream, 0, rtpPayloads[:], 1, rtpBuffer[:], rtpSpans[:], events[:], scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		t.Fatal(err)
	}
	if !batchPlan.HasEvent() ||
		batchPlan.Size.Events != 3 ||
		batchPlan.Size.RTPSpans != 3 ||
		batchPlan.Size.Event.Outputs != 2 ||
		batchPlan.Bind.EventIndex != 1 ||
		batchPlan.Bind.Event.Kind != internaldecoder.EventTileGroup ||
		batchPlan.Bind.Event.FrameHeader.OrderHint != rtpPlan.Bind.Event.FrameHeader.OrderHint {
		t.Fatalf("rtp payload batch stream plan=%+v", batchPlan)
	}
	batchSize, err := decoderFrameWorkResidualRTPPayloadsStreamScratchLen(stream, 0, rtpPayloads[:], 1, rtpBuffer[:], rtpSpans[:], events[:], scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		t.Fatal(err)
	}
	if batchPlan.Size != batchSize {
		t.Fatalf("batch plan size=%+v scratch size=%+v", batchPlan.Size, batchSize)
	}

	shortPlan, err := decoderFrameWorkResidualLowOverheadStreamPlan(stream, lowOverhead, 1, events[:2], scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if !errors.Is(err, ErrDecoderEventBufferTooSmall) {
		t.Fatalf("short plan err=%v want %v", err, ErrDecoderEventBufferTooSmall)
	}
	if shortPlan.Size.Events != 3 || shortPlan.HasEvent() {
		t.Fatalf("short low-overhead stream plan=%+v", shortPlan)
	}
}

func TestPublicDecoderFrameWorkResidualStreamRunnerRTPPayload(t *testing.T) {
	workerPool, err := internalthreading.NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	lowOverhead := publicDecoderResidualLowOverheadStream()
	var probeStream internaldecoder.Stream
	var probeEvents [4]internaldecoder.Event
	count, err := probeStream.PushLowOverhead(lowOverhead, probeEvents[:])
	if err != nil {
		t.Fatal(err)
	}
	sequence, ok := probeStream.SequenceHeader()
	if !ok {
		t.Fatal("probe stream missing sequence")
	}
	var scratchSpans [1]internalparser.TileSpan
	var scratchJobs [1]internaltile.Job
	var scratchBatches [1]internalthreading.Batch
	size, err := decoderFrameWorkResidualEventsScratchLen(sequence, probeEvents[:count], 1, scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		t.Fatal(err)
	}

	rtpPayload := publicDecoderResidualRTPPayload()
	pool := publicDecoderPostFilterFramePool(t, FrameFormat{
		Width:        int(probeEvents[count-1].FrameSize.CodedWidth),
		Height:       int(probeEvents[count-1].FrameSize.Height),
		BitDepth:     8,
		MonoChrome:   true,
		SubsamplingX: true,
		SubsamplingY: true,
		Align:        64,
	}, 1)
	var stream internaldecoder.Stream
	var refs internaldecoder.SurfaceReferences
	var state internaldecoder.FrameWorkState
	var referenceSurfaces [internalparser.InterRefsPerFrame]int
	var referenceFrames [internalparser.InterRefsPerFrame]*Frame
	var releases [internalparser.RefFrames]int
	var stats internalthreading.FrameWorkTileResidualStats
	var side decoderFrameWorkSideData
	var batchRunner decoderFrameWorkBatchResidualRunner
	var outputs [1]*Frame
	scratch := publicDecoderResidualEventScratch(size)
	eventRunner, _, err := bindDecoderFrameWorkResidualEventRunner(size, sequence, &probeEvents[count-1], decoderFrameWorkResidualEventRuntime{
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
	runner := decoderFrameWorkResidualStreamRunner{
		Stream:          &stream,
		EventRunner:     eventRunner,
		Events:          make([]internaldecoder.Event, len(probeEvents)),
		SideDataScratch: scratch.SideData,
		RTPBuffer:       make([]byte, len(lowOverhead)),
		RTPSpans:        make([]RTPObuSpan, len(probeEvents)),
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
	workerPool, err := internalthreading.NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	payloads := [...][]byte{
		publicDecoderResidualLowOverheadStream(),
		publicDecoderResidualLowOverheadFrameStream(),
	}
	var probeStream internaldecoder.Stream
	var probeEvents [4]internaldecoder.Event
	var scratchSpans [1]internalparser.TileSpan
	var scratchJobs [1]internaltile.Job
	var scratchBatches [1]internalthreading.Batch
	plan, err := decoderFrameWorkResidualLowOverheadStreamsPlan(probeStream, payloads[:], 1, probeEvents[:], scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		t.Fatal(err)
	}
	if plan.Size.Event.Outputs != 2 {
		t.Fatalf("low-overhead batch plan=%+v", plan)
	}

	pool := publicDecoderPostFilterFramePool(t, FrameFormat{
		Width:        int(plan.Bind.Event.FrameSize.CodedWidth),
		Height:       int(plan.Bind.Event.FrameSize.Height),
		BitDepth:     8,
		MonoChrome:   true,
		SubsamplingX: true,
		SubsamplingY: true,
		Align:        64,
	}, 2)
	var stream internaldecoder.Stream
	var refs internaldecoder.SurfaceReferences
	var state internaldecoder.FrameWorkState
	var referenceSurfaces [internalparser.InterRefsPerFrame]int
	var referenceFrames [internalparser.InterRefsPerFrame]*Frame
	var releases [internalparser.RefFrames]int
	var stats internalthreading.FrameWorkTileResidualStats
	var side decoderFrameWorkSideData
	var batchRunner decoderFrameWorkBatchResidualRunner
	streamScratch := publicDecoderResidualStreamScratch(plan.Size)
	runner, _, err := bindDecoderFrameWorkResidualStreamPlanRunner(plan, &stream, decoderFrameWorkResidualEventRuntime{
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

	var nilRunner *decoderFrameWorkResidualStreamRunner
	if _, err := nilRunner.RunLowOverheads(nil, nil); !errors.Is(err, internaldecoder.ErrInvalidFrameWorkState) {
		t.Fatalf("nil low-overhead batch runner err=%v want %v", err, internaldecoder.ErrInvalidFrameWorkState)
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
	stats = internalthreading.FrameWorkTileResidualStats{}
	if err := runner.Reset(); err != nil {
		t.Fatal(err)
	}
	result = decoderFrameWorkResidualStreamResult{}
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
	workerPool, err := internalthreading.NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	payloads := [...][]byte{
		publicDecoderResidualRTPPayload(),
		publicDecoderResidualRTPFramePayload(),
	}
	var probeStream internaldecoder.Stream
	var probeEvents [4]internaldecoder.Event
	var probeRTPBuffer [256]byte
	var probeRTPSpans [4]RTPObuSpan
	var scratchSpans [1]internalparser.TileSpan
	var scratchJobs [1]internaltile.Job
	var scratchBatches [1]internalthreading.Batch
	plan, err := decoderFrameWorkResidualRTPPayloadsStreamPlan(probeStream, 0, payloads[:], 1, probeRTPBuffer[:], probeRTPSpans[:], probeEvents[:], scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		t.Fatal(err)
	}

	pool := publicDecoderPostFilterFramePool(t, FrameFormat{
		Width:        int(plan.Bind.Event.FrameSize.CodedWidth),
		Height:       int(plan.Bind.Event.FrameSize.Height),
		BitDepth:     8,
		MonoChrome:   true,
		SubsamplingX: true,
		SubsamplingY: true,
		Align:        64,
	}, 2)
	var stream internaldecoder.Stream
	var refs internaldecoder.SurfaceReferences
	var state internaldecoder.FrameWorkState
	var referenceSurfaces [internalparser.InterRefsPerFrame]int
	var referenceFrames [internalparser.InterRefsPerFrame]*Frame
	var releases [internalparser.RefFrames]int
	var stats internalthreading.FrameWorkTileResidualStats
	var side decoderFrameWorkSideData
	var batchRunner decoderFrameWorkBatchResidualRunner
	streamScratch := publicDecoderResidualStreamScratch(plan.Size)
	runner, _, err := bindDecoderFrameWorkResidualStreamPlanRunner(plan, &stream, decoderFrameWorkResidualEventRuntime{
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
	if _, err := runner.RunRTPPayload(payloads[1], nil); !errors.Is(err, ErrRTPFragmentInterrupted) {
		t.Fatalf("interrupted fragment err=%v want %v", err, ErrRTPFragmentInterrupted)
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
	stats = internalthreading.FrameWorkTileResidualStats{}
	if err := runner.Reset(); err != nil {
		t.Fatal(err)
	}
	var result decoderFrameWorkResidualStreamResult
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

	var nilRunner *decoderFrameWorkResidualStreamRunner
	if _, err := nilRunner.RunRTPPayloadAfterLoss(payloads[0], nil); !errors.Is(err, internaldecoder.ErrInvalidFrameWorkState) {
		t.Fatalf("nil after-loss err=%v want %v", err, internaldecoder.ErrInvalidFrameWorkState)
	}
	var nilResult *decoderFrameWorkResidualStreamResult
	if err := runner.RunRTPPayloadAfterLossInto(nilResult, payloads[0], nil); !errors.Is(err, internaldecoder.ErrInvalidFrameWorkState) {
		t.Fatalf("nil after-loss result err=%v want %v", err, internaldecoder.ErrInvalidFrameWorkState)
	}
}

func TestPublicDecoderFrameWorkResidualStreamRunnerRTPPayloads(t *testing.T) {
	workerPool, err := internalthreading.NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	lowOverhead := publicDecoderResidualLowOverheadStream()
	rtpPayloads := publicDecoderResidualFragmentedRTPPayloads()
	var probeStream internaldecoder.Stream
	var probeEvents [4]internaldecoder.Event
	count, err := probeStream.PushLowOverhead(lowOverhead, probeEvents[:])
	if err != nil {
		t.Fatal(err)
	}
	sequence, ok := probeStream.SequenceHeader()
	if !ok {
		t.Fatal("probe stream missing sequence")
	}
	var scratchSpans [1]internalparser.TileSpan
	var scratchJobs [1]internaltile.Job
	var scratchBatches [1]internalthreading.Batch
	size, err := decoderFrameWorkResidualEventsScratchLen(sequence, probeEvents[:count], 1, scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		t.Fatal(err)
	}

	pool := publicDecoderPostFilterFramePool(t, FrameFormat{
		Width:        int(probeEvents[count-1].FrameSize.CodedWidth),
		Height:       int(probeEvents[count-1].FrameSize.Height),
		BitDepth:     8,
		MonoChrome:   true,
		SubsamplingX: true,
		SubsamplingY: true,
		Align:        64,
	}, 1)
	var stream internaldecoder.Stream
	var refs internaldecoder.SurfaceReferences
	var state internaldecoder.FrameWorkState
	var referenceSurfaces [internalparser.InterRefsPerFrame]int
	var referenceFrames [internalparser.InterRefsPerFrame]*Frame
	var releases [internalparser.RefFrames]int
	var stats internalthreading.FrameWorkTileResidualStats
	var side decoderFrameWorkSideData
	var batchRunner decoderFrameWorkBatchResidualRunner
	var outputs [1]*Frame
	scratch := publicDecoderResidualEventScratch(size)
	eventRunner, _, err := bindDecoderFrameWorkResidualEventRunner(size, sequence, &probeEvents[count-1], decoderFrameWorkResidualEventRuntime{
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
	runner := decoderFrameWorkResidualStreamRunner{
		Stream:          &stream,
		EventRunner:     eventRunner,
		Events:          make([]internaldecoder.Event, len(probeEvents)),
		SideDataScratch: scratch.SideData,
		RTPBuffer:       make([]byte, len(lowOverhead)),
		RTPSpans:        make([]RTPObuSpan, len(probeEvents)),
	}

	var nilRunner *decoderFrameWorkResidualStreamRunner
	if _, err := nilRunner.RunRTPPayloads(nil, nil); !errors.Is(err, internaldecoder.ErrInvalidFrameWorkState) {
		t.Fatalf("nil RTP payload batch runner err=%v want %v", err, internaldecoder.ErrInvalidFrameWorkState)
	}
	empty, err := runner.RunRTPPayloads(nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if empty.EventCount != 0 || empty.RTPUsed != 0 || empty.Run.Count != 0 || empty.Run.OutputCount != 0 {
		t.Fatalf("empty RTP payload batch result=%+v", empty)
	}

	postCalls := 0
	result, err := runner.RunRTPPayloads(rtpPayloads, func(ctx internaldecoder.FrameWorkPostFilterContext) error {
		postCalls++
		if !ctx.ExecutedTileWork || ctx.Step.Kind != internaldecoder.FrameWorkStepTile {
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
	workerPool, err := internalthreading.NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	payloads := [...][]byte{
		publicDecoderResidualRTPPayload(),
		publicDecoderResidualRTPFramePayload(),
	}
	var probeStream internaldecoder.Stream
	var probeEvents [4]internaldecoder.Event
	var probeRTPBuffer [256]byte
	var probeRTPSpans [4]RTPObuSpan
	var scratchSpans [1]internalparser.TileSpan
	var scratchJobs [1]internaltile.Job
	var scratchBatches [1]internalthreading.Batch
	plan, err := decoderFrameWorkResidualRTPPayloadsStreamPlan(probeStream, 0, payloads[:], 1, probeRTPBuffer[:], probeRTPSpans[:], probeEvents[:], scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		t.Fatal(err)
	}
	size := plan.Size
	if size.Events != 3 || size.RTPSpans != 3 || size.Event.Outputs != 2 {
		t.Fatalf("multi-output RTP batch scratch size=%+v", size)
	}
	if probeEvents[0].Kind != internaldecoder.EventFrameHeader || probeEvents[1].Kind != internaldecoder.EventTileGroup {
		t.Fatalf("last RTP batch events=%+v", probeEvents[:2])
	}
	if !plan.HasEvent() ||
		plan.Bind.EventIndex != 1 ||
		plan.Bind.Event.Kind != internaldecoder.EventTileGroup {
		t.Fatalf("multi-output RTP batch bind plan=%+v", plan)
	}
	sequence := plan.Bind.Sequence
	tile := plan.Bind.Event

	pool := publicDecoderPostFilterFramePool(t, FrameFormat{
		Width:        int(tile.FrameSize.CodedWidth),
		Height:       int(tile.FrameSize.Height),
		BitDepth:     8,
		MonoChrome:   true,
		SubsamplingX: true,
		SubsamplingY: true,
		Align:        64,
	}, 2)
	var stream internaldecoder.Stream
	var refs internaldecoder.SurfaceReferences
	var state internaldecoder.FrameWorkState
	var referenceSurfaces [internalparser.InterRefsPerFrame]int
	var referenceFrames [internalparser.InterRefsPerFrame]*Frame
	var releases [internalparser.RefFrames]int
	var stats internalthreading.FrameWorkTileResidualStats
	var side decoderFrameWorkSideData
	var batchRunner decoderFrameWorkBatchResidualRunner
	eventScratch := publicDecoderResidualEventScratch(size.Event)
	eventRunner, _, err := bindDecoderFrameWorkResidualEventRunner(size.Event, sequence, &tile, decoderFrameWorkResidualEventRuntime{
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
	runner, err := bindDecoderFrameWorkResidualStreamRunner(size, &stream, eventRunner, streamScratch)
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
	workerPool, err := internalthreading.NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	payloads := [...][]byte{
		publicDecoderResidualRTPPayload(),
		publicDecoderResidualRTPFramePayload(),
	}
	var probeStream internaldecoder.Stream
	var probeEvents [4]internaldecoder.Event
	var probeRTPBuffer [256]byte
	var probeRTPSpans [4]RTPObuSpan
	var scratchSpans [1]internalparser.TileSpan
	var scratchJobs [1]internaltile.Job
	var scratchBatches [1]internalthreading.Batch
	plan, err := decoderFrameWorkResidualRTPPayloadsStreamPlan(probeStream, 0, payloads[:], 1, probeRTPBuffer[:], probeRTPSpans[:], probeEvents[:], scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		t.Fatal(err)
	}

	pool := publicDecoderPostFilterFramePool(t, FrameFormat{
		Width:        int(plan.Bind.Event.FrameSize.CodedWidth),
		Height:       int(plan.Bind.Event.FrameSize.Height),
		BitDepth:     8,
		MonoChrome:   true,
		SubsamplingX: true,
		SubsamplingY: true,
		Align:        64,
	}, 2)
	var stream internaldecoder.Stream
	var refs internaldecoder.SurfaceReferences
	var state internaldecoder.FrameWorkState
	var referenceSurfaces [internalparser.InterRefsPerFrame]int
	var referenceFrames [internalparser.InterRefsPerFrame]*Frame
	var releases [internalparser.RefFrames]int
	var stats internalthreading.FrameWorkTileResidualStats
	var side decoderFrameWorkSideData
	var batchRunner decoderFrameWorkBatchResidualRunner
	streamScratch := publicDecoderResidualStreamScratch(plan.Size)
	runner, _, err := bindDecoderFrameWorkResidualStreamPlanRunner(plan, &stream, decoderFrameWorkResidualEventRuntime{
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

	var nilResult *decoderFrameWorkResidualStreamResult
	if err := runner.RunRTPPayloadInto(nilResult, payloads[0], nil); !errors.Is(err, internaldecoder.ErrInvalidFrameWorkState) {
		t.Fatalf("nil result err=%v want %v", err, internaldecoder.ErrInvalidFrameWorkState)
	}
	if err := runner.RunRTPPayloadsInto(nilResult, nil, nil); !errors.Is(err, internaldecoder.ErrInvalidFrameWorkState) {
		t.Fatalf("nil batch result err=%v want %v", err, internaldecoder.ErrInvalidFrameWorkState)
	}
	var result decoderFrameWorkResidualStreamResult
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
	var firstFrame Frame
	var secondFrame Frame
	var result decoderFrameWorkResidualStreamResult
	if err := result.Accumulate(decoderFrameWorkResidualStreamResult{
		EventCount: 2,
		RTPUsed:    3,
		Run: decoderFrameWorkResidualEventsResult{
			Count:            2,
			ExecutedTileWork: 1,
			CompletedFrames:  1,
			OutputCount:      1,
			Last: internaldecoder.FrameWorkEventResult{
				Output: &firstFrame,
			},
			Stats: internalthreading.FrameWorkTileResidualStats{
				TXBs:      1,
				Residuals: 2,
			},
		},
	}); err != nil {
		t.Fatal(err)
	}
	if err := result.Accumulate(decoderFrameWorkResidualStreamResult{
		EventCount: 1,
		RTPUsed:    0,
		Run: decoderFrameWorkResidualEventsResult{
			Count:            1,
			ExecutedTileWork: 1,
			CompletedFrames:  1,
			OutputCount:      1,
			Last: internaldecoder.FrameWorkEventResult{
				Output: &secondFrame,
			},
			Stats: internalthreading.FrameWorkTileResidualStats{
				TXBs:      3,
				Residuals: 5,
			},
		},
	}); err != nil {
		t.Fatal(err)
	}
	outputs := [...]*Frame{&firstFrame, &secondFrame}
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
	if err := result.BindOutputs(outputs[:1]); !errors.Is(err, internalframe.ErrShortBuffer) {
		t.Fatalf("short outputs err=%v want %v", err, internalframe.ErrShortBuffer)
	}
	var nilStreamResult *decoderFrameWorkResidualStreamResult
	if err := nilStreamResult.Accumulate(result); !errors.Is(err, internaldecoder.ErrInvalidFrameWorkState) {
		t.Fatalf("nil stream accumulate err=%v want %v", err, internaldecoder.ErrInvalidFrameWorkState)
	}
	var nilEventsResult *decoderFrameWorkResidualEventsResult
	if err := nilEventsResult.Accumulate(result.Run); !errors.Is(err, internaldecoder.ErrInvalidFrameWorkState) {
		t.Fatalf("nil event accumulate err=%v want %v", err, internaldecoder.ErrInvalidFrameWorkState)
	}
	if err := nilEventsResult.BindOutputs(outputs[:]); !errors.Is(err, internaldecoder.ErrInvalidFrameWorkState) {
		t.Fatalf("nil event outputs err=%v want %v", err, internaldecoder.ErrInvalidFrameWorkState)
	}
}

func TestPublicBindDecoderFrameWorkResidualStreamRunner(t *testing.T) {
	workerPool, err := internalthreading.NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	lowOverhead := publicDecoderResidualLowOverheadStream()
	rtpPayload := publicDecoderResidualRTPPayload()
	var probeStream internaldecoder.Stream
	var probeEvents [4]internaldecoder.Event
	var probeRTPBuffer [128]byte
	var probeRTPSpans [4]RTPObuSpan
	var scratchSpans [1]internalparser.TileSpan
	var scratchJobs [1]internaltile.Job
	var scratchBatches [1]internalthreading.Batch
	lowPlan, err := decoderFrameWorkResidualLowOverheadStreamPlan(probeStream, lowOverhead, 1, probeEvents[:], scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		t.Fatal(err)
	}
	rtpPlan, err := decoderFrameWorkResidualRTPPayloadStreamPlan(probeStream, 0, rtpPayload, 1, probeRTPBuffer[:], probeRTPSpans[:], probeEvents[:], scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		t.Fatal(err)
	}
	streamPlan := lowPlan.Max(rtpPlan)
	streamSize := streamPlan.Size
	if !streamPlan.HasEvent() || streamPlan.Bind.Event.Kind != internaldecoder.EventTileGroup {
		t.Fatalf("stream bind plan=%+v", streamPlan)
	}
	sequence := streamPlan.Bind.Sequence
	tile := streamPlan.Bind.Event

	pool := publicDecoderPostFilterFramePool(t, FrameFormat{
		Width:        int(tile.FrameSize.CodedWidth),
		Height:       int(tile.FrameSize.Height),
		BitDepth:     8,
		MonoChrome:   true,
		SubsamplingX: true,
		SubsamplingY: true,
		Align:        64,
	}, 1)
	var refs internaldecoder.SurfaceReferences
	var state internaldecoder.FrameWorkState
	var referenceSurfaces [internalparser.InterRefsPerFrame]int
	var referenceFrames [internalparser.InterRefsPerFrame]*Frame
	var releases [internalparser.RefFrames]int
	var stats internalthreading.FrameWorkTileResidualStats
	var side decoderFrameWorkSideData
	var batchRunner decoderFrameWorkBatchResidualRunner
	eventScratch := publicDecoderResidualEventScratch(streamSize.Event)
	eventRunner, _, err := bindDecoderFrameWorkResidualEventRunner(streamSize.Event, sequence, &tile, decoderFrameWorkResidualEventRuntime{
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

	var stream internaldecoder.Stream
	streamScratch := publicDecoderResidualStreamScratch(streamSize)
	runner, err := bindDecoderFrameWorkResidualStreamRunner(streamSize, &stream, eventRunner, streamScratch)
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

	if _, err := bindDecoderFrameWorkResidualStreamRunner(streamSize, nil, eventRunner, streamScratch); !errors.Is(err, internaldecoder.ErrInvalidFrameWorkState) {
		t.Fatalf("nil stream err=%v want %v", err, internaldecoder.ErrInvalidFrameWorkState)
	}
	shortEvents := streamScratch
	shortEvents.Events = shortEvents.Events[:streamSize.Events-1]
	if _, err := bindDecoderFrameWorkResidualStreamRunner(streamSize, &stream, eventRunner, shortEvents); !errors.Is(err, ErrDecoderEventBufferTooSmall) {
		t.Fatalf("short events err=%v want %v", err, ErrDecoderEventBufferTooSmall)
	}
	shortRTP := streamScratch
	shortRTP.RTPBuffer = shortRTP.RTPBuffer[:streamSize.RTPBuffer-1]
	if _, err := bindDecoderFrameWorkResidualStreamRunner(streamSize, &stream, eventRunner, shortRTP); !errors.Is(err, ErrRTPShortBuffer) {
		t.Fatalf("short rtp err=%v want %v", err, ErrRTPShortBuffer)
	}
	shortSide := streamScratch
	shortSide.SideData.CDEFIndexMap = shortSide.SideData.CDEFIndexMap[:streamSize.Event.SideData.CDEFIndexMap-1]
	if _, err := bindDecoderFrameWorkResidualStreamRunner(streamSize, &stream, eventRunner, shortSide); !errors.Is(err, internalframe.ErrShortBuffer) {
		t.Fatalf("short side-data err=%v want %v", err, internalframe.ErrShortBuffer)
	}
	shortOutputs := streamScratch
	shortOutputs.Outputs = shortOutputs.Outputs[:streamSize.Event.Outputs-1]
	if _, err := bindDecoderFrameWorkResidualStreamRunner(streamSize, &stream, eventRunner, shortOutputs); !errors.Is(err, internalframe.ErrShortBuffer) {
		t.Fatalf("short output err=%v want %v", err, internalframe.ErrShortBuffer)
	}
}

func TestPublicBindDecoderFrameWorkResidualStreamEventRunner(t *testing.T) {
	workerPool, err := internalthreading.NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	payloads := [...][]byte{
		publicDecoderResidualRTPPayload(),
		publicDecoderResidualRTPFramePayload(),
	}
	var probeStream internaldecoder.Stream
	var probeEvents [4]internaldecoder.Event
	var probeRTPBuffer [256]byte
	var probeRTPSpans [4]RTPObuSpan
	var scratchSpans [1]internalparser.TileSpan
	var scratchJobs [1]internaltile.Job
	var scratchBatches [1]internalthreading.Batch
	plan, err := decoderFrameWorkResidualRTPPayloadsStreamPlan(probeStream, 0, payloads[:], 1, probeRTPBuffer[:], probeRTPSpans[:], probeEvents[:], scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		t.Fatal(err)
	}
	size := plan.Size
	if !plan.HasEvent() ||
		plan.Bind.EventIndex != 1 ||
		plan.Bind.Event.Kind != internaldecoder.EventTileGroup {
		t.Fatalf("stream event bind plan=%+v", plan)
	}
	sequence := plan.Bind.Sequence
	event := plan.Bind.Event

	pool := publicDecoderPostFilterFramePool(t, FrameFormat{
		Width:        int(event.FrameSize.CodedWidth),
		Height:       int(event.FrameSize.Height),
		BitDepth:     8,
		MonoChrome:   true,
		SubsamplingX: true,
		SubsamplingY: true,
		Align:        64,
	}, 2)
	var stream internaldecoder.Stream
	var refs internaldecoder.SurfaceReferences
	var state internaldecoder.FrameWorkState
	var referenceSurfaces [internalparser.InterRefsPerFrame]int
	var referenceFrames [internalparser.InterRefsPerFrame]*Frame
	var releases [internalparser.RefFrames]int
	var stats internalthreading.FrameWorkTileResidualStats
	var side decoderFrameWorkSideData
	var batchRunner decoderFrameWorkBatchResidualRunner
	scratch := publicDecoderResidualStreamScratch(size)
	runner, boundSide, err := bindDecoderFrameWorkResidualStreamEventRunner(size, &stream, sequence, &event, decoderFrameWorkResidualEventRuntime{
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

	if _, _, err := bindDecoderFrameWorkResidualStreamEventRunner(size, nil, sequence, &event, decoderFrameWorkResidualEventRuntime{}, scratch, &batchRunner); !errors.Is(err, internaldecoder.ErrInvalidFrameWorkState) {
		t.Fatalf("nil stream err=%v want %v", err, internaldecoder.ErrInvalidFrameWorkState)
	}
	var untouchedBatch decoderFrameWorkBatchResidualRunner
	shortEvents := scratch
	shortEvents.Events = shortEvents.Events[:size.Events-1]
	if _, _, err := bindDecoderFrameWorkResidualStreamEventRunner(size, &stream, sequence, &event, decoderFrameWorkResidualEventRuntime{}, shortEvents, &untouchedBatch); !errors.Is(err, ErrDecoderEventBufferTooSmall) {
		t.Fatalf("short stream event scratch err=%v want %v", err, ErrDecoderEventBufferTooSmall)
	}
	if untouchedBatch.UseDefaultPrediction || len(untouchedBatch.States) != 0 {
		t.Fatalf("short stream scratch mutated batch runner: %+v", untouchedBatch)
	}
	shortEvent := scratch
	shortEvent.Event.Spans = shortEvent.Event.Spans[:size.Event.Plan.SpanCount-1]
	if _, _, err := bindDecoderFrameWorkResidualStreamEventRunner(size, &stream, sequence, &event, decoderFrameWorkResidualEventRuntime{}, shortEvent, &batchRunner); !errors.Is(err, internalframe.ErrShortBuffer) {
		t.Fatalf("short event scratch err=%v want %v", err, internalframe.ErrShortBuffer)
	}
	if _, _, err := bindDecoderFrameWorkResidualStreamEventRunner(size, &stream, sequence, &event, decoderFrameWorkResidualEventRuntime{}, scratch, nil); !errors.Is(err, internalthreading.ErrInvalidBatch) {
		t.Fatalf("nil batch runner err=%v want %v", err, internalthreading.ErrInvalidBatch)
	}
}

func TestPublicBindDecoderFrameWorkResidualStreamPlanRunner(t *testing.T) {
	workerPool, err := internalthreading.NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	payloads := [...][]byte{
		publicDecoderResidualRTPPayload(),
		publicDecoderResidualRTPFramePayload(),
	}
	var probeStream internaldecoder.Stream
	var probeEvents [4]internaldecoder.Event
	var probeRTPBuffer [256]byte
	var probeRTPSpans [4]RTPObuSpan
	var scratchSpans [1]internalparser.TileSpan
	var scratchJobs [1]internaltile.Job
	var scratchBatches [1]internalthreading.Batch
	plan, err := decoderFrameWorkResidualRTPPayloadsStreamPlan(probeStream, 0, payloads[:], 1, probeRTPBuffer[:], probeRTPSpans[:], probeEvents[:], scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		t.Fatal(err)
	}
	if !plan.HasEvent() || plan.Bind.Event.Kind != internaldecoder.EventTileGroup || plan.Size.Event.Outputs != 2 {
		t.Fatalf("stream plan=%+v", plan)
	}

	pool := publicDecoderPostFilterFramePool(t, FrameFormat{
		Width:        int(plan.Bind.Event.FrameSize.CodedWidth),
		Height:       int(plan.Bind.Event.FrameSize.Height),
		BitDepth:     8,
		MonoChrome:   true,
		SubsamplingX: true,
		SubsamplingY: true,
		Align:        64,
	}, 2)
	var stream internaldecoder.Stream
	var refs internaldecoder.SurfaceReferences
	var state internaldecoder.FrameWorkState
	var referenceSurfaces [internalparser.InterRefsPerFrame]int
	var referenceFrames [internalparser.InterRefsPerFrame]*Frame
	var releases [internalparser.RefFrames]int
	var stats internalthreading.FrameWorkTileResidualStats
	var side decoderFrameWorkSideData
	var batchRunner decoderFrameWorkBatchResidualRunner
	scratch := publicDecoderResidualStreamScratch(plan.Size)
	runner, boundSide, err := bindDecoderFrameWorkResidualStreamPlanRunner(plan, &stream, decoderFrameWorkResidualEventRuntime{
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

	if _, _, err := bindDecoderFrameWorkResidualStreamPlanRunner(plan, nil, decoderFrameWorkResidualEventRuntime{}, scratch, &batchRunner); !errors.Is(err, internaldecoder.ErrInvalidFrameWorkState) {
		t.Fatalf("nil stream plan bind err=%v want %v", err, internaldecoder.ErrInvalidFrameWorkState)
	}
}

func TestPublicDecoderFrameWorkResidualStreamPlanRunnerSupportedPostFilter(t *testing.T) {
	workerPool, err := internalthreading.NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	payload := publicDecoderResidualRTPPayload()
	var probeStream internaldecoder.Stream
	var probeEvents [4]internaldecoder.Event
	var probeRTPBuffer [128]byte
	var probeRTPSpans [4]RTPObuSpan
	var scratchSpans [1]internalparser.TileSpan
	var scratchJobs [1]internaltile.Job
	var scratchBatches [1]internalthreading.Batch
	plan, err := decoderFrameWorkResidualRTPPayloadStreamPlan(probeStream, 0, payload, 1, probeRTPBuffer[:], probeRTPSpans[:], probeEvents[:], scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		t.Fatal(err)
	}
	if !plan.HasEvent() || plan.Bind.Event.Kind != internaldecoder.EventTileGroup || plan.Size.Event.Outputs != 1 {
		t.Fatalf("stream plan=%+v", plan)
	}

	pool := publicDecoderPostFilterFramePool(t, FrameFormat{
		Width:        int(plan.Bind.Event.FrameSize.CodedWidth),
		Height:       int(plan.Bind.Event.FrameSize.Height),
		BitDepth:     8,
		MonoChrome:   true,
		SubsamplingX: true,
		SubsamplingY: true,
		Align:        64,
	}, 1)
	var stream internaldecoder.Stream
	var refs internaldecoder.SurfaceReferences
	var state internaldecoder.FrameWorkState
	var referenceSurfaces [internalparser.InterRefsPerFrame]int
	var referenceFrames [internalparser.InterRefsPerFrame]*Frame
	var releases [internalparser.RefFrames]int
	var stats internalthreading.FrameWorkTileResidualStats
	var side decoderFrameWorkSideData
	var batchRunner decoderFrameWorkBatchResidualRunner
	scratch := publicDecoderResidualStreamScratch(plan.Size)
	runner, boundSide, err := bindDecoderFrameWorkResidualStreamPlanRunner(plan, &stream, decoderFrameWorkResidualEventRuntime{
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

	var probeOutput Frame
	probeCtx, err := decoderFrameWorkPostFilterScratchContext(plan.Bind.Sequence, &plan.Bind.Event, 64, &boundSide, &probeOutput)
	if err != nil {
		t.Fatal(err)
	}
	var probe decoderFrameWorkSupportedPostFilterScratchRunner
	exact, err := probe.ScratchLen(probeCtx)
	if err != nil {
		t.Fatal(err)
	}
	postRunner := decoderFrameWorkSupportedPostFilterScratchRunner{
		Scratch: publicDecoderPostFilterRequestScratch(decoderFrameWorkPostFilterRequestScratchLen(exact)),
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
	var stream internaldecoder.Stream
	var events [4]internaldecoder.Event
	var scratchSpans [1]internalparser.TileSpan
	var scratchJobs [1]internaltile.Job
	var scratchBatches [1]internalthreading.Batch
	size, err := decoderFrameWorkResidualLowOverheadStreamScratchLen(stream, payload, 1, events[:], scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		t.Fatal(err)
	}
	if size.Events != 3 ||
		size.RTPBuffer != 0 ||
		size.RTPSpans != 0 ||
		size.Event.Plan != (internaldecoder.TileWorkPlan{SpanCount: 1, JobCount: 1, BatchCount: 1}) ||
		size.Event.Outputs != 1 ||
		size.Event.Runner.Workers != 1 ||
		size.Event.SideData.CDEFIndexMap == 0 ||
		size.Event.SideData.LoopFilterMap == 0 {
		t.Fatalf("low-overhead stream scratch size=%+v", size)
	}
	if stream.HasSequenceHeader() {
		t.Fatal("low-overhead stream scratch sizing mutated caller stream")
	}
	if events[0].Kind != internaldecoder.EventSequenceHeader ||
		events[1].Kind != internaldecoder.EventFrameHeader ||
		events[2].Kind != internaldecoder.EventTileGroup {
		t.Fatalf("events=%+v", events[:3])
	}

	short, err := decoderFrameWorkResidualLowOverheadStreamScratchLen(stream, payload, 1, events[:2], scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if !errors.Is(err, ErrDecoderEventBufferTooSmall) {
		t.Fatalf("short events err=%v want %v", err, ErrDecoderEventBufferTooSmall)
	}
	if short.Events != 3 || short.Event != (decoderFrameWorkResidualEventScratchSize{}) {
		t.Fatalf("short low-overhead scratch size=%+v", short)
	}
}

func TestPublicDecoderFrameWorkResidualStreamScratchLenRTPPayload(t *testing.T) {
	payload := publicDecoderResidualRTPPayload()
	var stream internaldecoder.Stream
	var events [4]internaldecoder.Event
	var rtpBuffer [128]byte
	var rtpSpans [4]RTPObuSpan
	var scratchSpans [1]internalparser.TileSpan
	var scratchJobs [1]internaltile.Job
	var scratchBatches [1]internalthreading.Batch
	size, err := decoderFrameWorkResidualRTPPayloadStreamScratchLen(stream, 0, payload, 1, rtpBuffer[:], rtpSpans[:], events[:], scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		t.Fatal(err)
	}
	if size.Events != 3 ||
		size.RTPBuffer == 0 ||
		size.RTPSpans != 3 ||
		size.Event.Plan != (internaldecoder.TileWorkPlan{SpanCount: 1, JobCount: 1, BatchCount: 1}) ||
		size.Event.Outputs != 1 ||
		size.Event.Runner.Workers != 1 ||
		size.Event.SideData.CDEFIndexMap == 0 ||
		size.Event.SideData.LoopFilterMap == 0 {
		t.Fatalf("rtp stream scratch size=%+v", size)
	}
	if stream.HasSequenceHeader() || stream.InRTPFragment() {
		t.Fatal("RTP stream scratch sizing mutated caller stream")
	}
	if events[0].Kind != internaldecoder.EventSequenceHeader ||
		events[1].Kind != internaldecoder.EventFrameHeader ||
		events[2].Kind != internaldecoder.EventTileGroup ||
		rtpSpans[0].Length == 0 ||
		rtpSpans[2].Length == 0 {
		t.Fatalf("events=%+v spans=%+v", events[:3], rtpSpans[:3])
	}

	shortRTP, err := decoderFrameWorkResidualRTPPayloadStreamScratchLen(stream, 0, payload, 1, rtpBuffer[:size.RTPBuffer-1], rtpSpans[:], events[:], scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if !errors.Is(err, ErrRTPShortBuffer) {
		t.Fatalf("short rtp buffer err=%v want %v", err, ErrRTPShortBuffer)
	}
	if shortRTP.Events != 3 || shortRTP.RTPBuffer != size.RTPBuffer || shortRTP.RTPSpans != 3 {
		t.Fatalf("short rtp scratch size=%+v want buffer %d", shortRTP, size.RTPBuffer)
	}

	shortEvents, err := decoderFrameWorkResidualRTPPayloadStreamScratchLen(stream, 0, payload, 1, rtpBuffer[:], rtpSpans[:], events[:2], scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if !errors.Is(err, ErrDecoderEventBufferTooSmall) {
		t.Fatalf("short events err=%v want %v", err, ErrDecoderEventBufferTooSmall)
	}
	if shortEvents.Events != 3 || shortEvents.RTPBuffer != size.RTPBuffer || shortEvents.RTPSpans != 3 {
		t.Fatalf("short events scratch size=%+v want buffer %d", shortEvents, size.RTPBuffer)
	}
}

func TestPublicDecoderFrameWorkResidualStreamScratchLenRTPPayloads(t *testing.T) {
	payload := publicDecoderResidualRTPPayload()
	payloads := publicDecoderResidualFragmentedRTPPayloads()
	var stream internaldecoder.Stream
	var events [4]internaldecoder.Event
	var rtpBuffer [128]byte
	var rtpSpans [4]RTPObuSpan
	var scratchSpans [1]internalparser.TileSpan
	var scratchJobs [1]internaltile.Job
	var scratchBatches [1]internalthreading.Batch
	single, err := decoderFrameWorkResidualRTPPayloadStreamScratchLen(stream, 0, payload, 1, rtpBuffer[:], rtpSpans[:], events[:], scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		t.Fatal(err)
	}
	size, err := decoderFrameWorkResidualRTPPayloadsStreamScratchLen(stream, 0, payloads, 1, rtpBuffer[:], rtpSpans[:], events[:], scratchSpans[:], scratchJobs[:], scratchBatches[:])
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
	if events[0].Kind != internaldecoder.EventTileGroup {
		t.Fatalf("last sized event=%+v", events[0])
	}

	shortRTP, err := decoderFrameWorkResidualRTPPayloadsStreamScratchLen(stream, 0, payloads, 1, rtpBuffer[:size.RTPBuffer-1], rtpSpans[:], events[:], scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if !errors.Is(err, ErrRTPShortBuffer) {
		t.Fatalf("short rtp buffer err=%v want %v", err, ErrRTPShortBuffer)
	}
	if shortRTP.RTPBuffer != size.RTPBuffer || shortRTP.Events != 1 || shortRTP.RTPSpans != 1 {
		t.Fatalf("short rtp payload batch scratch size=%+v want buffer %d", shortRTP, size.RTPBuffer)
	}

	shortEvents, err := decoderFrameWorkResidualRTPPayloadsStreamScratchLen(stream, 0, payloads, 1, rtpBuffer[:], rtpSpans[:], events[:0], scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if !errors.Is(err, ErrDecoderEventBufferTooSmall) {
		t.Fatalf("short events err=%v want %v", err, ErrDecoderEventBufferTooSmall)
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
	var stream internaldecoder.Stream
	var events [4]internaldecoder.Event
	var rtpBuffer [256]byte
	var rtpSpans [4]RTPObuSpan
	var scratchSpans [1]internalparser.TileSpan
	var scratchJobs [1]internaltile.Job
	var scratchBatches [1]internalthreading.Batch
	size, err := decoderFrameWorkResidualRTPPayloadsStreamScratchLen(stream, 0, payloads[:], 1, rtpBuffer[:], rtpSpans[:], events[:], scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		t.Fatal(err)
	}
	if size.Events != 3 ||
		size.RTPBuffer == 0 ||
		size.RTPSpans != 3 ||
		size.Event.Plan != (internaldecoder.TileWorkPlan{SpanCount: 1, JobCount: 1, BatchCount: 1}) ||
		size.Event.Outputs != 2 ||
		size.Event.Runner.Workers != 1 ||
		size.Event.SideData.CDEFIndexMap == 0 ||
		size.Event.SideData.LoopFilterMap == 0 {
		t.Fatalf("multi-output RTP payload batch scratch size=%+v", size)
	}
	if stream.HasSequenceHeader() || stream.InRTPFragment() {
		t.Fatal("multi-output RTP payload batch scratch sizing mutated caller stream")
	}
	if events[0].Kind != internaldecoder.EventFrameHeader || events[1].Kind != internaldecoder.EventTileGroup {
		t.Fatalf("last multi-output RTP batch events=%+v", events[:2])
	}
}

func TestPublicDecoderFrameWorkResidualStreamScratchLenRTPFragment(t *testing.T) {
	var stream internaldecoder.Stream
	var events [2]internaldecoder.Event
	var rtpBuffer [128]byte
	var rtpSpans [2]RTPObuSpan

	sequenceElement := [...]RTPElement{
		{Data: publicDecoderResidualRTPElement(internalobu.TypeSequenceHeader, publicDecoderResidualSequenceHeaderPayload())},
	}
	var sequencePayload [128]byte
	n, err := PutRTPPayload(sequencePayload[:], RTPAggregationHeader{
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
	if used == 0 || count != 1 || events[0].Kind != internaldecoder.EventSequenceHeader || !stream.HasSequenceHeader() {
		t.Fatalf("sequence used=%d count=%d event=%+v hasSequence=%v", used, count, events[0], stream.HasSequenceHeader())
	}

	frameHeader := publicDecoderResidualRTPElement(internalobu.TypeFrameHeader, publicDecoderResidualFrameHeaderPayload())
	var packet [8]byte
	n, next, more, err := PutRTPFragment(packet[:], frameHeader, 0, 2, false)
	if err != nil {
		t.Fatal(err)
	}
	if !more {
		t.Fatal("expected frame header to need a second fragment")
	}
	used = 0
	var scratchSpans [1]internalparser.TileSpan
	var scratchJobs [1]internaltile.Job
	var scratchBatches [1]internalthreading.Batch
	first, err := decoderFrameWorkResidualRTPPayloadStreamScratchLen(stream, used, packet[:n], 1, rtpBuffer[:], rtpSpans[:], events[:], scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		t.Fatal(err)
	}
	if first.Events != 0 || first.RTPBuffer == 0 || first.RTPSpans != 0 || first.Event != (decoderFrameWorkResidualEventScratchSize{}) {
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

	n, _, more, err = PutRTPFragment(packet[:], frameHeader, next, len(packet), false)
	if err != nil {
		t.Fatal(err)
	}
	if more {
		t.Fatal("unexpected third fragment")
	}
	final, err := decoderFrameWorkResidualRTPPayloadStreamScratchLen(stream, used, packet[:n], 1, rtpBuffer[:], rtpSpans[:], events[:], scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		t.Fatal(err)
	}
	if final.Events != 1 ||
		final.RTPBuffer != len(frameHeader) ||
		final.RTPSpans != 1 ||
		final.Event.Plan != (internaldecoder.TileWorkPlan{}) ||
		final.Event.SideData.CDEFIndexMap == 0 ||
		events[0].Kind != internaldecoder.EventFrameHeader {
		t.Fatalf("final fragment scratch size=%+v event=%+v", final, events[0])
	}
	if !stream.InRTPFragment() {
		t.Fatal("final fragment sizing mutated live stream RTP state")
	}
}

func TestPublicDecoderFrameWorkResidualStreamRunnerAllocs(t *testing.T) {
	workerPool, err := internalthreading.NewPool(1)
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
	var probeStream internaldecoder.Stream
	var probeEvents [4]internaldecoder.Event
	count, err := probeStream.PushLowOverhead(lowOverhead, probeEvents[:])
	if err != nil {
		t.Fatal(err)
	}
	sequence, ok := probeStream.SequenceHeader()
	if !ok {
		t.Fatal("probe stream missing sequence")
	}
	var scratchSpans [1]internalparser.TileSpan
	var scratchJobs [1]internaltile.Job
	var scratchBatches [1]internalthreading.Batch
	size, err := decoderFrameWorkResidualEventsScratchLen(sequence, probeEvents[:count], 1, scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		t.Fatal(err)
	}
	pool := publicDecoderPostFilterFramePool(t, FrameFormat{
		Width:        int(probeEvents[count-1].FrameSize.CodedWidth),
		Height:       int(probeEvents[count-1].FrameSize.Height),
		BitDepth:     8,
		MonoChrome:   true,
		SubsamplingX: true,
		SubsamplingY: true,
		Align:        64,
	}, 2)
	var stream internaldecoder.Stream
	var refs internaldecoder.SurfaceReferences
	var state internaldecoder.FrameWorkState
	var referenceSurfaces [internalparser.InterRefsPerFrame]int
	var referenceFrames [internalparser.InterRefsPerFrame]*Frame
	var releases [internalparser.RefFrames]int
	var stats internalthreading.FrameWorkTileResidualStats
	var side decoderFrameWorkSideData
	var batchRunner decoderFrameWorkBatchResidualRunner
	var outputs [2]*Frame
	scratch := publicDecoderResidualEventScratch(size)
	eventRunner, _, err := bindDecoderFrameWorkResidualEventRunner(size, sequence, &probeEvents[count-1], decoderFrameWorkResidualEventRuntime{
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
	var events [4]internaldecoder.Event
	var rtpBuffer [128]byte
	var rtpSpans [4]RTPObuSpan
	runner := decoderFrameWorkResidualStreamRunner{
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
			err = internalthreading.ErrInvalidBatch
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
			err = internalthreading.ErrInvalidBatch
			return
		}

		pool.Reset()
		refs.Reset()
		state.Reset()
		if resetErr := runner.Reset(); resetErr != nil {
			err = resetErr
			return
		}
		result = decoderFrameWorkResidualStreamResult{}
		if runErr = runner.RunLowOverheadsInto(&result, lowOverheads[:], nil); runErr != nil {
			err = runErr
			return
		}
		if result.Run.CompletedFrames != 2 || result.Run.OutputCount != 2 || outputs[1] != result.Run.Last.Output || stats.TXBs == 0 {
			err = internalthreading.ErrInvalidBatch
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
			err = internalthreading.ErrInvalidBatch
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
			err = internalthreading.ErrInvalidBatch
			return
		}

		pool.Reset()
		refs.Reset()
		state.Reset()
		if resetErr := runner.Reset(); resetErr != nil {
			err = resetErr
			return
		}
		result = decoderFrameWorkResidualStreamResult{}
		if runErr = runner.RunRTPPayloadsInto(&result, rtpPayloads, nil); runErr != nil {
			err = runErr
			return
		}
		if result.Run.CompletedFrames != 1 || result.Run.OutputCount != 1 || outputs[0] != result.Run.Last.Output || stats.TXBs == 0 || runner.RTPUsed != 0 {
			err = internalthreading.ErrInvalidBatch
			return
		}

		pool.Reset()
		refs.Reset()
		state.Reset()
		if resetErr := runner.Reset(); resetErr != nil {
			err = resetErr
			return
		}
		result = decoderFrameWorkResidualStreamResult{}
		if runErr = runner.RunRTPPayloadsInto(&result, rtpPayloads[:2], nil); runErr != nil {
			err = runErr
			return
		}
		if runner.RTPUsed == 0 {
			err = internalthreading.ErrInvalidBatch
			return
		}
		if runErr = runner.RunRTPPayloadAfterLossInto(&result, rtpPayload, nil); runErr != nil {
			err = runErr
			return
		}
		if result.Run.CompletedFrames != 1 || result.Run.OutputCount != 1 || outputs[0] != result.Run.Last.Output || stats.TXBs == 0 || runner.RTPUsed != 0 {
			err = internalthreading.ErrInvalidBatch
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
	workerPool, err := internalthreading.NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	payload := publicDecoderResidualRTPPayload()
	var probeStream internaldecoder.Stream
	var probeEvents [4]internaldecoder.Event
	var probeRTPBuffer [128]byte
	var probeRTPSpans [4]RTPObuSpan
	var scratchSpans [1]internalparser.TileSpan
	var scratchJobs [1]internaltile.Job
	var scratchBatches [1]internalthreading.Batch
	plan, err := decoderFrameWorkResidualRTPPayloadStreamPlan(probeStream, 0, payload, 1, probeRTPBuffer[:], probeRTPSpans[:], probeEvents[:], scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		t.Fatal(err)
	}

	pool := publicDecoderPostFilterFramePool(t, FrameFormat{
		Width:        int(plan.Bind.Event.FrameSize.CodedWidth),
		Height:       int(plan.Bind.Event.FrameSize.Height),
		BitDepth:     8,
		MonoChrome:   true,
		SubsamplingX: true,
		SubsamplingY: true,
		Align:        64,
	}, 1)
	var stream internaldecoder.Stream
	var refs internaldecoder.SurfaceReferences
	var state internaldecoder.FrameWorkState
	var referenceSurfaces [internalparser.InterRefsPerFrame]int
	var referenceFrames [internalparser.InterRefsPerFrame]*Frame
	var releases [internalparser.RefFrames]int
	var stats internalthreading.FrameWorkTileResidualStats
	var side decoderFrameWorkSideData
	var batchRunner decoderFrameWorkBatchResidualRunner
	scratch := publicDecoderResidualStreamScratch(plan.Size)
	runner, boundSide, err := bindDecoderFrameWorkResidualStreamPlanRunner(plan, &stream, decoderFrameWorkResidualEventRuntime{
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

	var probeOutput Frame
	probeCtx, err := decoderFrameWorkPostFilterScratchContext(plan.Bind.Sequence, &plan.Bind.Event, 64, &boundSide, &probeOutput)
	if err != nil {
		t.Fatal(err)
	}
	var probe decoderFrameWorkSupportedPostFilterScratchRunner
	exact, err := probe.ScratchLen(probeCtx)
	if err != nil {
		t.Fatal(err)
	}
	postScratch := publicDecoderPostFilterRequestScratch(decoderFrameWorkPostFilterRequestScratchLen(exact))
	postRunner := decoderFrameWorkSupportedPostFilterScratchRunner{Scratch: postScratch}

	allocs := testing.AllocsPerRun(1000, func() {
		pool.Reset()
		refs.Reset()
		state.Reset()
		if resetErr := runner.Reset(); resetErr != nil {
			err = resetErr
			return
		}
		postRunner = decoderFrameWorkSupportedPostFilterScratchRunner{Scratch: postScratch}
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
			err = internalthreading.ErrInvalidBatch
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
	var events [4]internaldecoder.Event
	var rtpBuffer [128]byte
	var rtpSpans [4]RTPObuSpan
	var scratchSpans [1]internalparser.TileSpan
	var scratchJobs [1]internaltile.Job
	var scratchBatches [1]internalthreading.Batch
	var err error
	var lowSize decoderFrameWorkResidualStreamScratchSize
	var lowBatchSize decoderFrameWorkResidualStreamScratchSize
	var rtpSize decoderFrameWorkResidualStreamScratchSize
	var rtpPayloadsSize decoderFrameWorkResidualStreamScratchSize

	allocs := testing.AllocsPerRun(1000, func() {
		var stream internaldecoder.Stream
		lowSize, err = decoderFrameWorkResidualLowOverheadStreamScratchLen(stream, lowOverhead, 1, events[:], scratchSpans[:], scratchJobs[:], scratchBatches[:])
		if err != nil {
			return
		}
		lowBatchSize, err = decoderFrameWorkResidualLowOverheadStreamsScratchLen(stream, lowOverheads[:], 1, events[:], scratchSpans[:], scratchJobs[:], scratchBatches[:])
		if err != nil {
			return
		}
		rtpSize, err = decoderFrameWorkResidualRTPPayloadStreamScratchLen(stream, 0, rtpPayload, 1, rtpBuffer[:], rtpSpans[:], events[:], scratchSpans[:], scratchJobs[:], scratchBatches[:])
		if err != nil {
			return
		}
		rtpPayloadsSize, err = decoderFrameWorkResidualRTPPayloadsStreamScratchLen(stream, 0, rtpPayloads, 1, rtpBuffer[:], rtpSpans[:], events[:], scratchSpans[:], scratchJobs[:], scratchBatches[:])
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
	sequence := SequenceHeader{ColorConfig: internalparser.ColorConfig{
		BitDepth:   8,
		MonoChrome: true,
	}}
	events := [...]internaldecoder.Event{
		{Kind: internaldecoder.EventSequenceHeader, SequenceHeader: sequence},
		{Kind: internaldecoder.EventFrameHeader, SequenceHeader: sequence},
		{Kind: internaldecoder.EventTileGroup, SequenceHeader: sequence},
	}
	var plan decoderFrameWorkResidualEventBindPlan
	allocs := testing.AllocsPerRun(1000, func() {
		plan = decoderFrameWorkResidualEventsBindPlan(SequenceHeader{}, events[:])
	})
	if !plan.HasEvent() || plan.EventIndex != 2 || plan.Event.Kind != internaldecoder.EventTileGroup {
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
	var events [4]internaldecoder.Event
	var rtpBuffer [256]byte
	var rtpSpans [4]RTPObuSpan
	var scratchSpans [1]internalparser.TileSpan
	var scratchJobs [1]internaltile.Job
	var scratchBatches [1]internalthreading.Batch
	var err error
	var lowPlan decoderFrameWorkResidualStreamPlan
	var lowBatchPlan decoderFrameWorkResidualStreamPlan
	var rtpPlan decoderFrameWorkResidualStreamPlan
	var batchPlan decoderFrameWorkResidualStreamPlan

	allocs := testing.AllocsPerRun(1000, func() {
		var stream internaldecoder.Stream
		lowPlan, err = decoderFrameWorkResidualLowOverheadStreamPlan(stream, lowOverhead, 1, events[:], scratchSpans[:], scratchJobs[:], scratchBatches[:])
		if err != nil {
			return
		}
		lowBatchPlan, err = decoderFrameWorkResidualLowOverheadStreamsPlan(stream, lowOverheads[:], 1, events[:], scratchSpans[:], scratchJobs[:], scratchBatches[:])
		if err != nil {
			return
		}
		rtpPlan, err = decoderFrameWorkResidualRTPPayloadStreamPlan(stream, 0, rtpPayload, 1, rtpBuffer[:], rtpSpans[:], events[:], scratchSpans[:], scratchJobs[:], scratchBatches[:])
		if err != nil {
			return
		}
		batchPlan, err = decoderFrameWorkResidualRTPPayloadsStreamPlan(stream, 0, rtpPayloads[:], 1, rtpBuffer[:], rtpSpans[:], events[:], scratchSpans[:], scratchJobs[:], scratchBatches[:])
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
		batchPlan.Bind.Event.Kind != internaldecoder.EventTileGroup {
		t.Fatalf("stream plans low=%+v lowBatch=%+v rtp=%+v batch=%+v", lowPlan, lowBatchPlan, rtpPlan, batchPlan)
	}
	if allocs != 0 {
		t.Fatalf("DecoderFrameWorkResidualStreamPlan allocated: %f", allocs)
	}

	var reusablePlan decoderFrameWorkResidualStreamPlan
	allocs = testing.AllocsPerRun(1000, func() {
		reusablePlan = lowPlan.Max(lowBatchPlan).Max(rtpPlan).Max(batchPlan)
	})
	if !reusablePlan.HasEvent() ||
		reusablePlan.Size.Event.Outputs != 2 ||
		reusablePlan.Bind.Event.Kind != internaldecoder.EventTileGroup {
		t.Fatalf("reusable stream plan=%+v", reusablePlan)
	}
	if allocs != 0 {
		t.Fatalf("DecoderFrameWorkResidualStreamPlan.Max allocated: %f", allocs)
	}
}

func TestPublicDecoderFrameWorkResidualStreamCheckAndStateAllocs(t *testing.T) {
	size := decoderFrameWorkResidualStreamScratchSize{
		Events:    3,
		RTPBuffer: 32,
		RTPSpans:  3,
		Event: decoderFrameWorkResidualEventScratchSize{
			Runner: decoderFrameWorkBatchResidualRunnerScratchSize{
				Workers:         1,
				Int32Scratch:    8,
				ResidualScratch: 8,
			},
			SideData: decoderFrameWorkSideDataScratchSize{
				CDEFIndexMap:  4,
				LoopFilterMap: 4,
			},
			Plan:    internaldecoder.TileWorkPlan{SpanCount: 1, JobCount: 1, BatchCount: 1},
			Outputs: 2,
		},
	}
	scratch := publicDecoderResidualStreamScratch(size)
	var stream internaldecoder.Stream
	runner := decoderFrameWorkResidualStreamRunner{
		Stream: &stream,
		EventRunner: decoderFrameWorkResidualEventRunner{
			Outputs: scratch.Outputs[:size.Event.Outputs],
		},
		Events:    scratch.Events[:size.Events],
		RTPBuffer: scratch.RTPBuffer[:size.RTPBuffer],
		RTPSpans:  scratch.RTPSpans[:size.RTPSpans],
		RTPUsed:   7,
	}
	var state decoderFrameWorkResidualStreamRunnerState
	var err error
	var ok bool
	var sequence SequenceHeader
	allocs := testing.AllocsPerRun(1000, func() {
		err = scratch.Check(size)
		state = runner.State()
		sequence, ok = runner.SequenceHeader()
	})
	if err != nil {
		t.Fatal(err)
	}
	if !state.Bound || state.RTPUsed != 7 || state.OutputCapacity != 2 || ok || sequence != (SequenceHeader{}) {
		t.Fatalf("state=%+v ok=%v sequence=%+v", state, ok, sequence)
	}
	if allocs != 0 {
		t.Fatalf("DecoderFrameWorkResidualStream Check/State allocated: %f", allocs)
	}
}

func TestPublicBindDecoderFrameWorkResidualStreamRunnerAllocs(t *testing.T) {
	size := decoderFrameWorkResidualStreamScratchSize{
		Events:    3,
		RTPBuffer: 32,
		RTPSpans:  3,
		Event: decoderFrameWorkResidualEventScratchSize{
			SideData: decoderFrameWorkSideDataScratchSize{
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
	var stream internaldecoder.Stream
	var eventRunner decoderFrameWorkResidualEventRunner
	var runner decoderFrameWorkResidualStreamRunner
	var err error
	allocs := testing.AllocsPerRun(1000, func() {
		runner, err = bindDecoderFrameWorkResidualStreamRunner(size, &stream, eventRunner, scratch)
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

	sequence := SequenceHeader{ColorConfig: internalparser.ColorConfig{
		BitDepth:   8,
		MonoChrome: true,
	}}
	event := internaldecoder.Event{
		Kind: internaldecoder.EventFrameHeader,
		FrameSize: FrameSize{
			CodedWidth:    16,
			UpscaledWidth: 16,
			Height:        16,
		},
	}
	var spans [1]internalparser.TileSpan
	var jobs [1]internaltile.Job
	var batches [1]internalthreading.Batch
	eventSize, err := decoderFrameWorkResidualEventScratchLen(sequence, &event, 1, spans[:], jobs[:], batches[:])
	if err != nil {
		t.Fatal(err)
	}
	combinedSize := decoderFrameWorkResidualStreamScratchSize{
		Events:    1,
		RTPBuffer: 8,
		RTPSpans:  1,
		Event:     eventSize,
	}
	combinedPlan := decoderFrameWorkResidualStreamPlan{
		Size: combinedSize,
		Bind: decoderFrameWorkResidualEventBindPlan{
			Sequence:   sequence,
			Event:      event,
			EventIndex: 0,
		},
	}
	combinedScratch := publicDecoderResidualStreamScratch(combinedSize)
	var combinedRunner decoderFrameWorkResidualStreamRunner
	var side decoderFrameWorkSideData
	allocs = testing.AllocsPerRun(1000, func() {
		combinedRunner, side, err = bindDecoderFrameWorkResidualStreamEventRunner(combinedSize, &stream, sequence, &event, decoderFrameWorkResidualEventRuntime{}, combinedScratch, nil)
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
		combinedRunner, side, err = bindDecoderFrameWorkResidualStreamPlanRunner(combinedPlan, &stream, decoderFrameWorkResidualEventRuntime{}, combinedScratch, nil)
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
	workerPool, err := internalthreading.NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	sequence := SequenceHeader{
		EnableCDEF: true,
		ColorConfig: internalparser.ColorConfig{
			BitDepth:   8,
			MonoChrome: true,
		},
	}
	event := publicDecoderResidualRunnerFrameEvent()
	event.CDEF = internalparser.CDEFParams{
		Damping:       5,
		StrengthCount: 1,
		YStrength:     [internalparser.MaxCDEFStrengths]uint8{63},
	}
	var scratchSpans [2]internalparser.TileSpan
	var scratchJobs [2]internaltile.Job
	var scratchBatches [1]internalthreading.Batch
	eventScratch, err := decoderFrameWorkResidualEventScratchLen(sequence, &event, 1, scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		t.Fatal(err)
	}
	runnerSize := eventScratch.Runner
	runner, err := bindDecoderFrameWorkBatchResidualRunner(runnerSize, publicDecoderBatchResidualRunnerScratch(runnerSize))
	if err != nil {
		t.Fatal(err)
	}
	side, err := bindDecoderFrameWorkResidualEventSideData(sequence, &event, publicDecoderFrameWorkSideDataScratch(eventScratch.SideData))
	if err != nil {
		t.Fatal(err)
	}

	pool := publicDecoderPostFilterFramePool(t, FrameFormat{
		Width:      int(event.FrameSize.CodedWidth),
		Height:     int(event.FrameSize.Height),
		BitDepth:   8,
		MonoChrome: true,
		Align:      64,
	}, 1)
	var refs internaldecoder.SurfaceReferences
	var referenceSurfaces [internalparser.InterRefsPerFrame]int
	var referenceFrames [internalparser.InterRefsPerFrame]*Frame
	var spans [2]internalparser.TileSpan
	var jobs [2]internaltile.Job
	var batches [1]internalthreading.Batch
	var releases [internalparser.RefFrames]int
	var state internaldecoder.FrameWorkState
	var stats internalthreading.FrameWorkTileResidualStats
	var probeOutput Frame
	probeCtx, err := decoderFrameWorkPostFilterScratchContext(sequence, &event, 64, &side, &probeOutput)
	if err != nil {
		t.Fatal(err)
	}
	var probe decoderFrameWorkSupportedPostFilterScratchRunner
	postScratchSize, err := probe.ScratchLen(probeCtx)
	if err != nil {
		t.Fatal(err)
	}
	postRunner := decoderFrameWorkSupportedPostFilterScratchRunner{
		Scratch: publicDecoderPostFilterRequestScratch(decoderFrameWorkPostFilterRequestScratchLen(postScratchSize)),
	}
	eventRunner := decoderFrameWorkResidualEventRunner{
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
		result, runErr := eventRunner.RunWithPostFilterRunner(sequence, &event, &side, &postRunner)
		if runErr != nil {
			err = runErr
			return
		}
		if result.Output == nil || result.Run != (internaldecoder.FrameWorkStepResult{ExecutedTileWork: true, CompletedFrame: true}) {
			err = internaldecoder.ErrInvalidFrameWorkStep
			return
		}
		if !postRunner.Result.Completed.Has(internaldecoder.FrameWorkPostFilterCDEF) || postRunner.Context.RemainingPostFilters() != 0 {
			err = ErrDecoderUnsupportedPostFilter
			return
		}
		if stats.TXBs == 0 || stats.Residuals == 0 {
			err = internalthreading.ErrInvalidBatch
			return
		}
		if _, ok := refs.ReferenceSlot(0); !ok {
			err = internaldecoder.ErrInvalidSurfaceReference
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
	stats, err := decodeAndReconstructDecoderFrameWorkJobResiduals(batch, 0, state, cdfs, &scratch, req)
	if err != nil {
		t.Fatal(err)
	}
	if stats.Loop.Blocks != 1 || stats.Loop.PartitionReads != 1 {
		t.Fatalf("stats=%+v", stats)
	}
	if stats.CoefficientBlocks != 1 || stats.SkippedBlocks != 0 ||
		stats.TXBs != 1 || stats.Residuals != 1 || stats.TXBs != stats.NonZero+stats.AllZero {
		t.Fatalf("residual stats=%+v", stats)
	}
	if stats.Loop.CoefficientBlocks != 1 {
		t.Fatalf("loop coefficient stats=%+v residual stats=%+v", stats.Loop, stats)
	}
}

func TestPublicDecodeAndRetainDecoderFrameWorkJobResiduals(t *testing.T) {
	batch, state, _, scratch, req := publicDecoderResidualDriver(t)
	var storage internalthreading.FrameWorkTileResidualCDFStorage
	if err := initDecoderFrameWorkTileResidualCDFStorageDefault(&storage, batch.Quantization.BaseQIdx); err != nil {
		t.Fatal(err)
	}
	var retained internalthreading.FrameWorkTileResidualCDFStorage
	retainedValid := false
	batch.RetainedTileResidualCDFs = &retained
	batch.RetainedTileResidualCDFsValid = &retainedValid
	batch.Jobs[0].UpdatesFrameContext = true

	stats, err := decodeAndRetainDecoderFrameWorkJobResiduals(batch, 0, state, &storage, &scratch, req)
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

	if err := initDecoderFrameWorkTileResidualCDFStorageDefault(&storage, batch.Quantization.BaseQIdx); err != nil {
		t.Fatal(err)
	}
	retainedValid = false
	batch.Jobs[0].UpdatesFrameContext = false
	if _, err := decodeAndRetainDecoderFrameWorkJobResiduals(batch, 0, state, &storage, &scratch, req); err != nil {
		t.Fatal(err)
	}
	if retainedValid {
		t.Fatal("non-update job retained residual CDF storage")
	}
}

func TestPublicDecodeAndRetainDecoderFrameWorkJobResidualsDefaultTransforms(t *testing.T) {
	batch, state, _, scratch, req := publicDecoderResidualDriver(t)
	var storage internalthreading.FrameWorkTileResidualCDFStorage
	if err := initDecoderFrameWorkTileResidualCDFStorageDefault(&storage, batch.Quantization.BaseQIdx); err != nil {
		t.Fatal(err)
	}

	stats, err := decodeAndRetainDecoderFrameWorkJobResiduals(batch, 0, state, &storage, &scratch, req)
	if err != nil {
		t.Fatal(err)
	}
	if stats.Loop.Blocks != 1 || stats.TXBs != 1 || stats.Residuals != 1 {
		t.Fatalf("stats=%+v", stats)
	}
}

func TestPublicDecodeAndRetainDecoderFrameWorkBatchResiduals(t *testing.T) {
	batch, state, storage, scratch, req := publicDecoderResidualBatchDriver(t)
	var retained internalthreading.FrameWorkTileResidualCDFStorage
	retainedValid := false
	batch.RetainedTileResidualCDFs = &retained
	batch.RetainedTileResidualCDFsValid = &retainedValid
	visits := 0
	req.Tile.AfterBlock = func(visit internaltile.BlockLoopVisit) error {
		visits++
		if visits == 2 && visit.Block.MICol != 16 {
			t.Fatalf("second job block=%+v", visit.Block)
		}
		return nil
	}

	stats, err := decodeAndRetainDecoderFrameWorkBatchResiduals(batch, state, &storage, &scratch, req)
	if err != nil {
		t.Fatal(err)
	}
	if stats.Loop.Blocks != 2 || stats.Loop.PartitionReads != 2 || visits != 2 {
		t.Fatalf("stats=%+v visits=%d", stats, visits)
	}
	if stats.CoefficientBlocks != 2 || stats.Residuals != 2 ||
		stats.TXBs != 2 || stats.TXBs != stats.NonZero+stats.AllZero {
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
	batch.CDEF = internalparser.CDEFParams{Bits: 0, StrengthCount: 1}
	sequence := SequenceHeader{
		Use128x128Superblock: batch.Sequence.SBSizeMIB == 32,
		ColorConfig:          batch.Sequence.ColorConfig,
	}
	_, _, cdefLength, err := decoderFrameWorkCDEFIndexMapShape(sequence, batch.FrameSize)
	if err != nil {
		t.Fatal(err)
	}
	cdefMap, err := bindDecoderFrameWorkCDEFIndexMap(sequence, batch.FrameSize, batch.CDEF, make([]uint8, cdefLength), make([]bool, cdefLength))
	if err != nil {
		t.Fatal(err)
	}
	_, _, loopLength, err := decoderFrameWorkLoopFilterMapShape(sequence, batch.FrameSize)
	if err != nil {
		t.Fatal(err)
	}
	loopMap, err := bindDecoderFrameWorkLoopFilterMap(sequence, batch.FrameSize, make([]internalthreading.FrameWorkLoopFilterBlockRecord, loopLength))
	if err != nil {
		t.Fatal(err)
	}
	batch.CDEFIndexMap = &cdefMap
	batch.LoopFilterMap = &loopMap
	state.DeltaLFFromBase = -3
	state.DeltaLF = [internaltile.FrameLoopFilterCount]int8{4, 3, 2, 1}

	stats, err := decodeAndReconstructDecoderFrameWorkJobResiduals(batch, 0, state, cdfs, &scratch, req)
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
	if _, err := decodeAndReconstructDecoderFrameWorkJobResiduals(batch, 0, nil, cdfs, &scratch, req); !errors.Is(err, internalthreading.ErrInvalidBatch) {
		t.Fatalf("nil state err=%v want %v", err, internalthreading.ErrInvalidBatch)
	}
	if _, err := decodeAndReconstructDecoderFrameWorkJobResiduals(batch, 0, state, cdfs, nil, req); !errors.Is(err, internalthreading.ErrInvalidBatch) {
		t.Fatalf("nil scratch err=%v want %v", err, internalthreading.ErrInvalidBatch)
	}
	shortCarrier := req
	scratch.LoopContext.Above = []internaltile.BlockLoopRootAboveContext{}
	if _, err := decodeAndReconstructDecoderFrameWorkJobResiduals(batch, 0, state, cdfs, &scratch, shortCarrier); !errors.Is(err, internaltile.ErrInvalidDecodeState) {
		t.Fatalf("short scratch carrier err=%v want %v", err, internaltile.ErrInvalidDecodeState)
	}
	if _, err := decodeAndRetainDecoderFrameWorkJobResiduals(batch, 0, state, nil, &scratch, req); !errors.Is(err, internalthreading.ErrInvalidBatch) {
		t.Fatalf("nil retain storage err=%v want %v", err, internalthreading.ErrInvalidBatch)
	}
	batchReq := decoderFrameWorkBatchResidualRequest{Tile: req, LoopContextAbove: []internaltile.BlockLoopRootAboveContext{}}
	if _, err := decodeAndRetainDecoderFrameWorkBatchResiduals(batch, state, &internalthreading.FrameWorkTileResidualCDFStorage{}, &scratch, batchReq); !errors.Is(err, internalframe.ErrShortBuffer) {
		t.Fatalf("short batch carrier err=%v want %v", err, internalframe.ErrShortBuffer)
	}
	if _, err := decodeAndRetainDecoderFrameWorkBatchResiduals(batch, nil, &internalthreading.FrameWorkTileResidualCDFStorage{}, &scratch, batchReq); !errors.Is(err, internalthreading.ErrInvalidBatch) {
		t.Fatalf("nil batch state err=%v want %v", err, internalthreading.ErrInvalidBatch)
	}
	if _, _, err := decoderFrameWorkResidualScratchLen(batch, batch.Quantization.BaseQIdx, 0, internaldecoder.FrameWorkPlane(99), internaltransform.Size{Width: 64, Height: 64}, internaltransform.TypeDCTDCT); !errors.Is(err, internalthreading.ErrInvalidBatch) {
		t.Fatalf("bad scratch plane err=%v want %v", err, internalthreading.ErrInvalidBatch)
	}
	if n, err := decoderFrameWorkBatchResidualLoopContextAboveLen(internaldecoder.FrameWorkBatch{}); err != nil || n != 0 {
		t.Fatalf("empty batch loop context len=%d err=%v want 0,nil", n, err)
	}
	if err := initDecoderFrameWorkTileRestorationRequestReferences(nil); !errors.Is(err, internalthreading.ErrInvalidBatch) {
		t.Fatalf("nil restoration refs err=%v want %v", err, internalthreading.ErrInvalidBatch)
	}
}

func TestPublicDecoderTileResidualStateAllocs(t *testing.T) {
	var initial internalthreading.FrameWorkTileResidualCDFStorage
	if err := initDecoderFrameWorkTileResidualCDFStorageDefault(&initial, 64); err != nil {
		t.Fatal(err)
	}
	var retained internalthreading.FrameWorkTileResidualCDFStorage
	retainedValid := false
	batch := internaldecoder.FrameWorkBatch{
		Payload: []byte{0x00, 0xff},
		FrameWorkFrameContext: internaldecoder.FrameWorkFrameContext{
			Quantization: internalparser.QuantizationParams{BaseQIdx: 73},
		},
		InitialTileResidualCDFs:       &initial,
		RetainedTileResidualCDFs:      &retained,
		RetainedTileResidualCDFsValid: &retainedValid,
		Jobs: []internaltile.Job{
			{Tile: 0, Offset: 0, Size: 1},
			{Tile: 1, Offset: 1, Size: 1, UpdatesFrameContext: true},
		},
	}
	var storage internalthreading.FrameWorkTileResidualCDFStorage
	var state internaltile.DecodeState
	blockLoopBatch := publicDecoderBlockLoopBatch()
	rootColumns, err := decoderFrameWorkJobBlockLoopContextRootColumns(blockLoopBatch, 0)
	if err != nil {
		t.Fatal(err)
	}
	above := make([]internaltile.BlockLoopRootAboveContext, rootColumns)
	segMap := make([]uint8, 96*96)

	allocs := testing.AllocsPerRun(1000, func() {
		err = initDecoderFrameWorkTileResidualCDFStorage(batch, &storage)
		if err != nil {
			return
		}
		_ = decoderFrameWorkTileResidualCDFsFromStorage(&storage)
		err = initDecoderFrameWorkJobDecodeState(batch, 1, &state)
		if err != nil {
			return
		}
		err = retainDecoderFrameWorkTileResidualCDFStorage(batch, 1, &state, &storage)
		if err != nil {
			return
		}
		var carrier internaltile.BlockLoopContextCarrier
		carrier, err = bindTileBlockLoopContextCarrier(rootColumns, above)
		if err != nil {
			return
		}
		_, err = decoderFrameWorkJobBlockLoopRequest(blockLoopBatch, 0, segMap, nil, 96, &carrier)
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
		if err := initDecoderFrameWorkJobDecodeState(batch, 0, state); err != nil {
			t.Fatal(err)
		}
		if _, err := decodeAndReconstructDecoderFrameWorkJobResiduals(batch, 0, state, cdfs, &scratch, req); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("allocs=%v want 0", allocs)
	}
}

func TestPublicDecodeAndRetainDecoderFrameWorkJobResidualsAllocs(t *testing.T) {
	batch, state, _, scratch, req := publicDecoderResidualDriver(t)
	var storage internalthreading.FrameWorkTileResidualCDFStorage
	allocs := testing.AllocsPerRun(1000, func() {
		if err := initDecoderFrameWorkTileResidualCDFStorageDefault(&storage, batch.Quantization.BaseQIdx); err != nil {
			t.Fatal(err)
		}
		if _, err := decodeAndRetainDecoderFrameWorkJobResiduals(batch, 0, state, &storage, &scratch, req); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("allocs=%v want 0", allocs)
	}
}

func TestPublicDecodeAndRetainDecoderFrameWorkBatchResidualsAllocs(t *testing.T) {
	batch, state, storage, scratch, _ := publicDecoderResidualBatchDriver(t)
	scratchSize, err := decoderFrameWorkBatchResidualScratchLen(batch, batch.Quantization.BaseQIdx)
	if err != nil {
		t.Fatal(err)
	}
	arena := decoderFrameWorkBatchResidualScratch{
		Int32Scratch:     make([]int32, scratchSize.Int32Scratch),
		ResidualScratch:  make([]int16, scratchSize.ResidualScratch),
		LoopContextAbove: make([]internaltile.BlockLoopRootAboveContext, scratchSize.LoopContextAbove),
	}
	var req decoderFrameWorkBatchResidualRequest
	allocs := testing.AllocsPerRun(1000, func() {
		req, err = bindDecoderFrameWorkBatchResidualRequestScratch(scratchSize, arena)
		if err != nil {
			t.Fatal(err)
		}
		req.Tile.TransformMode = batch.TransformRef.TransformMode
		if err := initDecoderFrameWorkTileResidualCDFStorageDefault(&storage, batch.Quantization.BaseQIdx); err != nil {
			t.Fatal(err)
		}
		if _, err := decodeAndRetainDecoderFrameWorkBatchResiduals(batch, state, &storage, &scratch, req); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("allocs=%v want 0", allocs)
	}
}

func TestPublicDecoderFrameWorkBatchResidualRunnerAllocs(t *testing.T) {
	batch, _, _, _, _ := publicDecoderResidualBatchDriver(t)
	size, err := decoderFrameWorkBatchResidualRunnerScratchLen(batch, 1)
	if err != nil {
		t.Fatal(err)
	}
	runner, err := bindDecoderFrameWorkBatchResidualRunner(size, publicDecoderBatchResidualRunnerScratch(size))
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

func publicDecoderBlockLoopBatch() internaldecoder.FrameWorkBatch {
	return internaldecoder.FrameWorkBatch{
		FrameWorkFrameContext: internaldecoder.FrameWorkFrameContext{
			Sequence: internalthreading.FrameWorkSequenceContextFromHeader(SequenceHeader{
				Use128x128Superblock: true,
				EnableDualFilter:     true,
				EnableFilterIntra:    true,
				EnableOrderHint:      true,
				OrderHintBits:        5,
				ColorConfig:          internalparser.ColorConfig{BitDepth: 8, MonoChrome: true},
			}),
			FrameHeader:         internalparser.FrameHeaderPrefix{OrderHint: 9},
			FrameSize:           FrameSize{CodedWidth: 300, UpscaledWidth: 300, Height: 260, SuperResDenominator: 8},
			TileInfo:            internalparser.TileInfo{InterpolationFilter: internalparser.InterpolationSwitchable, UseRefFrameMVS: true},
			ReferenceOrderHints: [internalparser.InterRefsPerFrame]uint8{1, 9, 10, 4, 5, 6, 7},
			SkipMode:            internalparser.SkipModeParams{Allowed: true, Enabled: true},
			CDEF:                internalparser.CDEFParams{Bits: 2, StrengthCount: 4},
			Delta:               internalparser.DeltaParams{DeltaQPresent: true, DeltaQResLog2: 1},
		},
		Jobs: []internaltile.Job{{Tile: 3, Row: 1, Col: 1, SBX: 1, SBY: 1, SBCols: 2, SBRows: 2}},
	}
}

func publicDecoderResidualDriver(t *testing.T) (internaldecoder.FrameWorkBatch, *internaltile.DecodeState, internalthreading.FrameWorkTileResidualCDFs, internalthreading.FrameWorkTileResidualScratch, internalthreading.FrameWorkTileResidualRequest) {
	t.Helper()
	output := publicDecoderPostFilterFrame(t, FrameFormat{Width: 64, Height: 64, BitDepth: 8, MonoChrome: true, Align: 64})
	publicFillDecoderPostFilterPlane(output.Y)
	batch := internaldecoder.FrameWorkBatch{
		Output:  output,
		Payload: make([]byte, 256),
		FrameWorkFrameContext: internaldecoder.FrameWorkFrameContext{
			Sequence: internalthreading.FrameWorkSequenceContextFromHeader(SequenceHeader{
				ColorConfig: internalparser.ColorConfig{BitDepth: 8, MonoChrome: true},
			}),
			FrameSize:    FrameSize{CodedWidth: 64, UpscaledWidth: 64, Height: 64, SuperResDenominator: 8},
			Quantization: internalparser.QuantizationParams{BaseQIdx: 64},
			TransformRef: internalparser.TransformReferenceParams{TransformMode: internalparser.TransformModeLargest},
		},
		Jobs: []internaltile.Job{{SBCols: 1, SBRows: 1, Offset: 0, Size: 256}},
	}
	state := &internaltile.DecodeState{}
	if err := initDecoderFrameWorkJobDecodeState(batch, 0, state); err != nil {
		t.Fatal(err)
	}
	var storage internalthreading.FrameWorkTileResidualCDFStorage
	if err := initDecoderFrameWorkTileResidualCDFStorageDefault(&storage, batch.Quantization.BaseQIdx); err != nil {
		t.Fatal(err)
	}
	loopReq, err := decoderFrameWorkJobBlockLoopRequest(batch, 0, nil, nil, 0, nil)
	if err != nil {
		t.Fatal(err)
	}
	int32Len, int16Len, err := decoderFrameWorkResidualMaxScratchLen(batch, batch.Quantization.BaseQIdx, 0, internaldecoder.FrameWorkPlaneY)
	if err != nil {
		t.Fatal(err)
	}
	var scratch internalthreading.FrameWorkTileResidualScratch
	req := internalthreading.FrameWorkTileResidualRequest{
		Loop:            loopReq,
		TransformMode:   batch.TransformRef.TransformMode,
		Int32Scratch:    make([]int32, int32Len),
		ResidualScratch: make([]int16, int16Len),
	}
	return batch, state, decoderFrameWorkTileResidualCDFsFromStorage(&storage), scratch, req
}

func publicDecoderResidualBatchDriver(t *testing.T) (internaldecoder.FrameWorkBatch, *internaltile.DecodeState, internalthreading.FrameWorkTileResidualCDFStorage, internalthreading.FrameWorkTileResidualScratch, decoderFrameWorkBatchResidualRequest) {
	t.Helper()
	output := publicDecoderPostFilterFrame(t, FrameFormat{Width: 128, Height: 64, BitDepth: 8, MonoChrome: true, Align: 64})
	publicFillDecoderPostFilterPlane(output.Y)
	batch := internaldecoder.FrameWorkBatch{
		Output:  output,
		Payload: make([]byte, 512),
		FrameWorkFrameContext: internaldecoder.FrameWorkFrameContext{
			Sequence: internalthreading.FrameWorkSequenceContextFromHeader(SequenceHeader{
				ColorConfig: internalparser.ColorConfig{BitDepth: 8, MonoChrome: true},
			}),
			FrameSize:    FrameSize{CodedWidth: 128, UpscaledWidth: 128, Height: 64, SuperResDenominator: 8},
			Quantization: internalparser.QuantizationParams{BaseQIdx: 64},
			TransformRef: internalparser.TransformReferenceParams{TransformMode: internalparser.TransformModeLargest},
		},
		Jobs: []internaltile.Job{
			{SBX: 0, SBY: 0, SBCols: 1, SBRows: 1, Offset: 0, Size: 256},
			{SBX: 1, SBY: 0, SBCols: 1, SBRows: 1, Offset: 256, Size: 256, UpdatesFrameContext: true},
		},
	}
	var retained internalthreading.FrameWorkTileResidualCDFStorage
	retainedValid := false
	batch.RetainedTileResidualCDFs = &retained
	batch.RetainedTileResidualCDFsValid = &retainedValid
	var storage internalthreading.FrameWorkTileResidualCDFStorage
	if err := initDecoderFrameWorkTileResidualCDFStorageDefault(&storage, batch.Quantization.BaseQIdx); err != nil {
		t.Fatal(err)
	}
	int32Len, int16Len, err := decoderFrameWorkResidualMaxScratchLen(batch, batch.Quantization.BaseQIdx, 0, internaldecoder.FrameWorkPlaneY)
	if err != nil {
		t.Fatal(err)
	}
	loopContextLen, err := decoderFrameWorkBatchResidualLoopContextAboveLen(batch)
	if err != nil {
		t.Fatal(err)
	}
	if loopContextLen != 1 {
		t.Fatalf("batch loop context len=%d want 1", loopContextLen)
	}
	state := &internaltile.DecodeState{}
	var scratch internalthreading.FrameWorkTileResidualScratch
	req := decoderFrameWorkBatchResidualRequest{
		Tile: internalthreading.FrameWorkTileResidualRequest{
			TransformMode:   batch.TransformRef.TransformMode,
			Int32Scratch:    make([]int32, int32Len),
			ResidualScratch: make([]int16, int16Len),
		},
		LoopContextAbove: make([]internaltile.BlockLoopRootAboveContext, loopContextLen),
	}
	return batch, state, storage, scratch, req
}

func publicDecoderBatchResidualRunnerScratch(size decoderFrameWorkBatchResidualRunnerScratchSize) decoderFrameWorkBatchResidualRunnerScratch {
	return decoderFrameWorkBatchResidualRunnerScratch{
		States:                  make([]internaltile.DecodeState, size.Workers+1),
		Storages:                make([]internalthreading.FrameWorkTileResidualCDFStorage, size.Workers+1),
		TileScratch:             make([]internalthreading.FrameWorkTileResidualScratch, size.Workers+1),
		RestorationRequests:     make([]internalthreading.FrameWorkTileRestorationRequest, size.RestorationRequests+1),
		PredictionScratch:       make([]internalthreading.FrameWorkPredictionScratch, size.Workers+1),
		InterPredictionScratch:  make([]internalthreading.FrameWorkInterPredictionScratch, size.Workers+1),
		Stats:                   make([]internalthreading.FrameWorkTileResidualStats, size.Workers+1),
		Int32Scratch:            make([]int32, size.Int32Scratch+1),
		ResidualScratch:         make([]int16, size.ResidualScratch+1),
		LoopContextAboveScratch: make([]internaltile.BlockLoopRootAboveContext, size.LoopContextAbove+1),
	}
}

func publicDecoderFrameWorkSideDataScratch(size decoderFrameWorkSideDataScratchSize) decoderFrameWorkSideDataScratch {
	return decoderFrameWorkSideDataScratch{
		CDEFIndexMap:             make([]uint8, size.CDEFIndexMap+1),
		CDEFReadMap:              make([]bool, size.CDEFReadMap+1),
		LoopFilterMap:            make([]internalthreading.FrameWorkLoopFilterBlockRecord, size.LoopFilterMap+1),
		RestorationRecords:       make([]internaltile.RestorationUnitRecord, size.RestorationRecords+1),
		RestorationBoundaryAbove: make([]uint16, size.RestorationBoundaryAbove+1),
		RestorationBoundaryBelow: make([]uint16, size.RestorationBoundaryBelow+1),
	}
}

func publicDecoderResidualEventScratch(size decoderFrameWorkResidualEventScratchSize) decoderFrameWorkResidualEventScratch {
	return decoderFrameWorkResidualEventScratch{
		Runner:   publicDecoderBatchResidualRunnerScratch(size.Runner),
		SideData: publicDecoderFrameWorkSideDataScratch(size.SideData),
		Spans:    make([]internalparser.TileSpan, size.Plan.SpanCount+1),
		Jobs:     make([]internaltile.Job, size.Plan.JobCount+1),
		Batches:  make([]internalthreading.Batch, size.Plan.BatchCount+1),
	}
}

func publicDecoderResidualStreamScratch(size decoderFrameWorkResidualStreamScratchSize) decoderFrameWorkResidualStreamScratch {
	return decoderFrameWorkResidualStreamScratch{
		Events:    make([]internaldecoder.Event, size.Events+1),
		Event:     publicDecoderResidualEventScratch(size.Event),
		SideData:  publicDecoderFrameWorkSideDataScratch(size.Event.SideData),
		Outputs:   make([]*Frame, size.Event.Outputs+1),
		RTPBuffer: make([]byte, size.RTPBuffer+1),
		RTPSpans:  make([]RTPObuSpan, size.RTPSpans+1),
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
	stream = appendOBUForTest(stream, internalobu.TypeSequenceHeader, publicDecoderResidualSequenceHeaderPayload())
	stream = appendOBUForTest(stream, internalobu.TypeFrameHeader, publicDecoderResidualFrameHeaderPayload())
	stream = appendOBUForTest(stream, internalobu.TypeTileGroup, []byte{0x80})
	return stream
}

func publicDecoderResidualLowOverheadFrameStream() []byte {
	var stream []byte
	stream = appendOBUForTest(stream, internalobu.TypeFrameHeader, publicDecoderResidualFrameHeaderPayload())
	stream = appendOBUForTest(stream, internalobu.TypeTileGroup, []byte{0x80})
	return stream
}

func publicDecoderResidualRTPPayload() []byte {
	elements := [...]RTPElement{
		{Data: publicDecoderResidualRTPElement(internalobu.TypeSequenceHeader, publicDecoderResidualSequenceHeaderPayload())},
		{Data: publicDecoderResidualRTPElement(internalobu.TypeFrameHeader, publicDecoderResidualFrameHeaderPayload())},
		{Data: publicDecoderResidualRTPElement(internalobu.TypeTileGroup, []byte{0x80})},
	}
	payload := make([]byte, 128)
	n, err := PutRTPPayload(payload, RTPAggregationHeader{
		ElementCount:                uint8(len(elements)),
		StartsNewCodedVideoSequence: true,
	}, elements[:])
	if err != nil {
		panic(err)
	}
	return payload[:n]
}

func publicDecoderResidualRTPFramePayload() []byte {
	elements := [...]RTPElement{
		{Data: publicDecoderResidualRTPElement(internalobu.TypeFrameHeader, publicDecoderResidualFrameHeaderPayload())},
		{Data: publicDecoderResidualRTPElement(internalobu.TypeTileGroup, []byte{0x80})},
	}
	payload := make([]byte, 128)
	n, err := PutRTPPayload(payload, RTPAggregationHeader{
		ElementCount: uint8(len(elements)),
	}, elements[:])
	if err != nil {
		panic(err)
	}
	return payload[:n]
}

func publicDecoderResidualFragmentedRTPPayloads() [][]byte {
	payloads := make([][]byte, 0, 4)

	sequence := [...]RTPElement{
		{Data: publicDecoderResidualRTPElement(internalobu.TypeSequenceHeader, publicDecoderResidualSequenceHeaderPayload())},
	}
	var packet [128]byte
	n, err := PutRTPPayload(packet[:], RTPAggregationHeader{
		ElementCount:                1,
		StartsNewCodedVideoSequence: true,
	}, sequence[:])
	if err != nil {
		panic(err)
	}
	payloads = append(payloads, append([]byte(nil), packet[:n]...))

	frameHeader := publicDecoderResidualRTPElement(internalobu.TypeFrameHeader, publicDecoderResidualFrameHeaderPayload())
	n, next, more, err := PutRTPFragment(packet[:], frameHeader, 0, 2, false)
	if err != nil {
		panic(err)
	}
	if !more {
		panic("expected frame header RTP fragment")
	}
	payloads = append(payloads, append([]byte(nil), packet[:n]...))
	n, _, more, err = PutRTPFragment(packet[:], frameHeader, next, len(packet), false)
	if err != nil {
		panic(err)
	}
	if more {
		panic("unexpected third frame header RTP fragment")
	}
	payloads = append(payloads, append([]byte(nil), packet[:n]...))

	tile := [...]RTPElement{
		{Data: publicDecoderResidualRTPElement(internalobu.TypeTileGroup, []byte{0x80})},
	}
	n, err = PutRTPPayload(packet[:], RTPAggregationHeader{ElementCount: 1}, tile[:])
	if err != nil {
		panic(err)
	}
	payloads = append(payloads, append([]byte(nil), packet[:n]...))

	return payloads
}

func publicDecoderResidualRTPElement(typ internalobu.Type, payload []byte) []byte {
	var header [2]byte
	n, err := internalobu.PutHeader(header[:], internalobu.Header{Type: typ})
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
	w.writeBool(false)                                  // show_existing_frame
	w.writeBits(uint64(internalparser.FrameTypeKey), 2) // frame_type
	w.writeBool(true)                                   // show_frame
	w.writeBool(false)                                  // disable_cdf_update
	w.writeBool(false)                                  // frame_size_override_flag
	w.writeBool(false)                                  // render_and_frame_size_different
	w.writeBool(false)                                  // disable_frame_end_update_cdf
	w.writeBool(false)                                  // uniform_tile_spacing_flag
	publicDecoderResidualWriteColorQuantParams(&w)
	publicDecoderResidualWriteZeroSegmentationParams(&w)
	w.writeBool(false) // reduced_tx_set
	return w.bytes()
}

func publicDecoderResidualInterFrameHeaderPayload() []byte {
	var w publicDecoderResidualBitWriter
	w.writeBool(false)                                    // show_existing_frame
	w.writeBits(uint64(internalparser.FrameTypeInter), 2) // frame_type
	w.writeBool(true)                                     // show_frame
	w.writeBool(false)                                    // error_resilient_mode
	w.writeBool(false)                                    // disable_cdf_update
	w.writeBool(false)                                    // frame_size_override_flag
	w.writeBits(0, 3)                                     // primary_ref_frame
	w.writeBits(0x01, 8)                                  // refresh_frame_flags
	for range internalparser.InterRefsPerFrame {
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
	for range internalparser.InterRefsPerFrame {
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

func publicDecoderResidualRunnerFrameEvent() internaldecoder.Event {
	payload := make([]byte, 513)
	payload[0] = 0xff
	return internaldecoder.Event{
		Kind: internaldecoder.EventFrame,
		Unit: internalobu.Unit{Payload: payload},
		FrameHeader: internalparser.FrameHeaderPrefix{
			FrameType: internalparser.FrameTypeKey,
			ShowFrame: true,
		},
		FrameSize: FrameSize{
			CodedWidth:          128,
			UpscaledWidth:       128,
			Height:              64,
			SuperResDenominator: 8,
			RefreshFrameFlags:   0xff,
		},
		TileInfo: internalparser.TileInfo{
			TileSizeBytes: 1,
			SBCols:        2,
			SBRows:        1,
			Cols:          2,
			Rows:          1,
			ColStartSB:    [internalparser.MaxTileCols + 1]uint16{0, 1, 2},
			RowStartSB:    [internalparser.MaxTileRows + 1]uint16{0, 1},
		},
		TileGroup: internalparser.TileGroup{
			StartTile: 0,
			EndTile:   1,
			TileCount: 2,
			Final:     true,
		},
		CDEF:         internalparser.CDEFParams{Bits: 0, StrengthCount: 1},
		Quantization: internalparser.QuantizationParams{BaseQIdx: 64},
		TransformRef: internalparser.TransformReferenceParams{TransformMode: internalparser.TransformModeLargest},
	}
}

func publicCDFValuesEqual(a []uint16, b []uint16) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func publicFillDecoderBlockCoeffFrame(frame *Frame, sample uint16) {
	fillPublicReconstructPlane(frame.Y, frame.Layout.BytesPerSample, sample)
	if len(frame.U.Pix) != 0 {
		fillPublicReconstructPlane(frame.U, frame.Layout.BytesPerSample, sample)
	}
	if len(frame.V.Pix) != 0 {
		fillPublicReconstructPlane(frame.V, frame.Layout.BytesPerSample, sample)
	}
}
