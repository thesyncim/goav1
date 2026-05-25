package threading

import (
	"bytes"
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/quantize"
	"github.com/thesyncim/goav1/internal/av1/reconstruct"
	"github.com/thesyncim/goav1/internal/av1/tile"
	"github.com/thesyncim/goav1/internal/av1/transform"
)

func TestBuildBatchesBalancesContiguousJobs(t *testing.T) {
	jobs := testJobs()
	var batches [2]Batch

	n, err := BuildBatches(batches[:], jobs[:], 2)
	if err != nil {
		t.Fatal(err)
	}
	if n != 2 {
		t.Fatalf("n=%d", n)
	}
	if batches[0] != (Batch{Worker: 0, FirstJob: 0, Count: 2, FirstTile: 0, LastTile: 1, Units: 10}) {
		t.Fatalf("batch[0]=%+v", batches[0])
	}
	if batches[1] != (Batch{Worker: 1, FirstJob: 2, Count: 2, FirstTile: 2, LastTile: 3, Units: 15}) {
		t.Fatalf("batch[1]=%+v", batches[1])
	}
}

func TestBuildBatchesCapsWorkersToJobs(t *testing.T) {
	jobs := testJobs()
	var batches [8]Batch

	n, err := BuildBatches(batches[:], jobs[:], 8)
	if err != nil {
		t.Fatal(err)
	}
	if n != len(jobs) {
		t.Fatalf("n=%d want %d", n, len(jobs))
	}
	for i := 0; i < n; i++ {
		if batches[i].Worker != uint16(i) || batches[i].Count != 1 || batches[i].FirstJob != i {
			t.Fatalf("batch[%d]=%+v", i, batches[i])
		}
	}
}

func TestBuildBatchesRejectsInvalidWorkerCount(t *testing.T) {
	var batches [1]Batch
	jobs := testJobs()
	_, err := BuildBatches(batches[:], jobs[:], 0)
	if !errors.Is(err, ErrInvalidWorkerCount) {
		t.Fatalf("BuildBatches err=%v want %v", err, ErrInvalidWorkerCount)
	}
}

func TestBuildBatchesRejectsShortBuffer(t *testing.T) {
	var batches [1]Batch
	jobs := testJobs()
	_, err := BuildBatches(batches[:], jobs[:], 2)
	if !errors.Is(err, ErrBatchBufferTooSmall) {
		t.Fatalf("BuildBatches err=%v want %v", err, ErrBatchBufferTooSmall)
	}
}

func TestBuildBatchesRejectsZeroAreaJob(t *testing.T) {
	jobs := [1]tile.Job{{Tile: 0, SBCols: 0, SBRows: 1}}
	var batches [1]Batch
	_, err := BuildBatches(batches[:], jobs[:], 1)
	if !errors.Is(err, ErrInvalidJobs) {
		t.Fatalf("BuildBatches err=%v want %v", err, ErrInvalidJobs)
	}
}

func TestFrameWorkSequenceContextFromHeader(t *testing.T) {
	seq := parser.SequenceHeader{
		SeqProfile:                 1,
		Use128x128Superblock:       true,
		EnableFilterIntra:          true,
		EnableIntraEdgeFilter:      true,
		EnableInterIntraCompound:   true,
		EnableMaskedCompound:       true,
		EnableWarpedMotion:         true,
		EnableDualFilter:           true,
		EnableOrderHint:            true,
		EnableJNTComp:              true,
		EnableRefFrameMVS:          true,
		SeqForceScreenContentTools: parser.SelectScreenContentTools,
		SeqForceIntegerMV:          parser.SelectIntegerMV,
		OrderHintBits:              5,
		EnableSuperRes:             true,
		EnableCDEF:                 true,
		EnableRestoration:          true,
		ColorConfig:                parser.ColorConfig{BitDepth: 10, SubsamplingX: true, SubsamplingY: true},
		FilmGrainParamsPresent:     true,
	}

	ctx := FrameWorkSequenceContextFromHeader(seq)
	if !ctx.Valid() ||
		ctx.Profile != 1 ||
		!ctx.Use128x128Superblock ||
		ctx.SBSizeLog2 != 7 ||
		ctx.SBSizeMIB != 32 ||
		!ctx.EnableFilterIntra ||
		!ctx.EnableIntraEdgeFilter ||
		!ctx.EnableInterIntraCompound ||
		!ctx.EnableMaskedCompound ||
		!ctx.EnableWarpedMotion ||
		!ctx.EnableDualFilter ||
		!ctx.EnableOrderHint ||
		!ctx.EnableJNTComp ||
		!ctx.EnableRefFrameMVS ||
		ctx.SeqForceScreenContentTools != parser.SelectScreenContentTools ||
		ctx.SeqForceIntegerMV != parser.SelectIntegerMV ||
		ctx.OrderHintBits != 5 ||
		!ctx.EnableSuperRes ||
		!ctx.EnableCDEF ||
		!ctx.EnableRestoration ||
		ctx.ColorConfig != seq.ColorConfig ||
		!ctx.FilmGrainParamsPresent {
		t.Fatalf("sequence context=%+v", ctx)
	}

	seq.Use128x128Superblock = false
	ctx = FrameWorkSequenceContextFromHeader(seq)
	if ctx.SBSizeLog2 != 6 || ctx.SBSizeMIB != 16 {
		t.Fatalf("64x64 superblock context=%+v", ctx)
	}
}

func TestFrameWorkBatchJobPayload(t *testing.T) {
	payload := []byte{0xaa, 0xbb, 0xcc}
	ctx := FrameWorkBatch{
		Payload: payload,
		Jobs: []tile.Job{
			{Offset: 1, Size: 2},
			{Offset: 0, Size: 1},
		},
	}

	if err := ctx.ValidatePayloads(); err != nil {
		t.Fatal(err)
	}
	data, err := ctx.JobPayload(0)
	if err != nil {
		t.Fatal(err)
	}
	if len(data) != 2 || data[0] != 0xbb || data[1] != 0xcc {
		t.Fatalf("payload=%v", data)
	}
	if len(data) != 0 {
		data[0] = 0xdd
	}
	if payload[1] != 0xdd {
		t.Fatalf("payload did not alias: %v", payload)
	}
}

func TestFrameWorkBatchJobPayloadRejectsInvalidInputs(t *testing.T) {
	ctx := FrameWorkBatch{
		Payload: []byte{0xaa},
		Jobs:    []tile.Job{{Offset: 0, Size: 2}},
	}
	if _, err := ctx.JobPayload(-1); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("negative index err=%v want %v", err, ErrInvalidBatch)
	}
	if _, err := ctx.JobPayload(1); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("large index err=%v want %v", err, ErrInvalidBatch)
	}
	if _, err := ctx.JobPayload(0); !errors.Is(err, tile.ErrInvalidPlan) {
		t.Fatalf("invalid range err=%v want %v", err, tile.ErrInvalidPlan)
	}
	if err := ctx.ValidatePayloads(); !errors.Is(err, tile.ErrInvalidPlan) {
		t.Fatalf("ValidatePayloads err=%v want %v", err, tile.ErrInvalidPlan)
	}
}

func TestFrameWorkBatchJobEntropyReader(t *testing.T) {
	ctx := FrameWorkBatch{
		Payload:          []byte{0x00, 0xff, 0x00},
		DisableCDFUpdate: true,
		Jobs: []tile.Job{
			{Offset: 0, Size: 1},
			{Offset: 1, Size: 1},
		},
	}

	r, err := ctx.JobEntropyReader(1)
	if err != nil {
		t.Fatal(err)
	}
	if r.AllowCDFUpdate() {
		t.Fatal("CDF update enabled")
	}
	bit, err := r.ReadBit()
	if err != nil {
		t.Fatal(err)
	}
	if bit != 1 {
		t.Fatalf("bit=%d want 1", bit)
	}

	ctx.DisableCDFUpdate = false
	r, err = ctx.JobEntropyReader(0)
	if err != nil {
		t.Fatal(err)
	}
	if !r.AllowCDFUpdate() {
		t.Fatal("CDF update disabled")
	}
}

func TestFrameWorkBatchJobEntropyReaderRejectsInvalidInputs(t *testing.T) {
	ctx := FrameWorkBatch{
		Payload: []byte{0xaa},
		Jobs:    []tile.Job{{Offset: 0, Size: 2}},
	}
	if _, err := ctx.JobEntropyReader(-1); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("negative index err=%v want %v", err, ErrInvalidBatch)
	}
	if _, err := ctx.JobEntropyReader(1); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("large index err=%v want %v", err, ErrInvalidBatch)
	}
	if _, err := ctx.JobEntropyReader(0); !errors.Is(err, tile.ErrInvalidPlan) {
		t.Fatalf("invalid range err=%v want %v", err, tile.ErrInvalidPlan)
	}
}

func TestFrameWorkBatchJobDecodeState(t *testing.T) {
	wantSequence := testBatchSequenceContext()
	wantDelta := parser.DeltaParams{DeltaQPresent: true, DeltaQResLog2: 2}
	wantMotion := testBatchGlobalMotion()
	wantGrain := testBatchFilmGrain()
	ctx := FrameWorkBatch{
		Payload: []byte{0x00, 0xff, 0x00},
		FrameWorkFrameContext: FrameWorkFrameContext{
			Sequence:     wantSequence,
			FrameSize:    parser.FrameSize{CodedWidth: 128, Height: 64},
			TileInfo:     testBatchTileInfo(),
			Quantization: parser.QuantizationParams{BaseQIdx: 73},
			Segmentation: parser.SegmentationParams{AllLossless: false, QIndex: [parser.MaxSegments]uint8{73}},
			Delta:        wantDelta,
			LoopFilter:   parser.LoopFilterParams{LevelY: [2]uint8{4, 5}},
			CDEF:         parser.CDEFParams{Damping: 3, StrengthCount: 1},
			Restoration:  parser.RestorationParams{UnitSizeY: 64},
			TransformRef: parser.TransformReferenceParams{TransformMode: parser.TransformModeSwitchable, ReferenceMode: parser.ReferenceModeSelect},
			SkipMode:     parser.SkipModeParams{Allowed: true, Enabled: true, RefFrameIdx: [2]uint8{1, 2}},
			FrameMode:    parser.FrameModeParams{AllowWarpedMotion: true, ReducedTxSet: true},
			GlobalMotion: wantMotion,
			FilmGrain:    wantGrain,
		},
		Jobs: []tile.Job{
			{Tile: 0, Offset: 0, Size: 1},
			{Tile: 1, Offset: 1, Size: 1, UpdatesFrameContext: true},
		},
	}
	var state tile.DecodeState
	if ctx.Delta != wantDelta {
		t.Fatalf("Delta=%+v want %+v", ctx.Delta, wantDelta)
	}
	if ctx.Sequence != wantSequence || ctx.Sequence.SBSizeMIB != 32 || ctx.Sequence.ColorConfig.BitDepth != 10 {
		t.Fatalf("Sequence=%+v want %+v", ctx.Sequence, wantSequence)
	}
	if ctx.FrameSize.CodedWidth != 128 || ctx.TileInfo.Cols != 2 {
		t.Fatalf("frame context size=%+v tiles=%+v", ctx.FrameSize, ctx.TileInfo)
	}
	if ctx.Segmentation.QIndex[0] != 73 || ctx.LoopFilter.LevelY[0] != 4 ||
		ctx.CDEF.Damping != 3 || ctx.Restoration.UnitSizeY != 64 {
		t.Fatalf("filter context seg=%+v lf=%+v cdef=%+v restoration=%+v", ctx.Segmentation, ctx.LoopFilter, ctx.CDEF, ctx.Restoration)
	}
	if ctx.TransformRef.ReferenceMode != parser.ReferenceModeSelect || !ctx.SkipMode.Enabled ||
		!ctx.FrameMode.AllowWarpedMotion || ctx.GlobalMotion != wantMotion || ctx.FilmGrain != wantGrain {
		t.Fatalf("motion context transform=%+v skip=%+v frame=%+v global=%+v grain=%+v", ctx.TransformRef, ctx.SkipMode, ctx.FrameMode, ctx.GlobalMotion, ctx.FilmGrain)
	}

	if err := ctx.JobDecodeState(1, &state); err != nil {
		t.Fatal(err)
	}
	if state.Job.Tile != 1 {
		t.Fatalf("state job=%+v", state.Job)
	}
	if !state.Reader.AllowCDFUpdate() {
		t.Fatal("CDF update disabled")
	}
	if !state.RetainFrameContext {
		t.Fatal("frame context not retained")
	}
	if state.CurrentBaseQIdx != 73 {
		t.Fatalf("CurrentBaseQIdx=%d want 73", state.CurrentBaseQIdx)
	}
	bit, err := state.Reader.ReadBit()
	if err != nil {
		t.Fatal(err)
	}
	if bit != 1 {
		t.Fatalf("bit=%d want 1", bit)
	}

	ctx.DisableCDFUpdate = true
	if err := ctx.JobDecodeState(1, &state); err != nil {
		t.Fatal(err)
	}
	if state.Reader.AllowCDFUpdate() {
		t.Fatal("CDF update enabled")
	}
	if state.RetainFrameContext {
		t.Fatal("frame context retained")
	}
	if state.CurrentBaseQIdx != 73 {
		t.Fatalf("CurrentBaseQIdx=%d want 73", state.CurrentBaseQIdx)
	}
}

func TestFrameWorkBatchTileResidualCDFStorage(t *testing.T) {
	var initial FrameWorkTileResidualCDFStorage
	if err := initial.InitDefault(64); err != nil {
		t.Fatal(err)
	}
	if err := initial.DeltaQ.Update(2); err != nil {
		t.Fatal(err)
	}
	var retained FrameWorkTileResidualCDFStorage
	retainedValid := false
	ctx := FrameWorkBatch{
		Payload: []byte{0x00, 0xff},
		FrameWorkFrameContext: FrameWorkFrameContext{
			Quantization: parser.QuantizationParams{BaseQIdx: 91},
		},
		InitialTileResidualCDFs:       &initial,
		RetainedTileResidualCDFs:      &retained,
		RetainedTileResidualCDFsValid: &retainedValid,
		Jobs: []tile.Job{
			{Tile: 0, Offset: 0, Size: 1},
			{Tile: 1, Offset: 1, Size: 1, UpdatesFrameContext: true},
		},
	}
	var storage FrameWorkTileResidualCDFStorage
	if err := ctx.InitTileResidualCDFStorage(&storage); err != nil {
		t.Fatal(err)
	}
	if !testCDFValuesEqual(storage.DeltaQ.Values(), initial.DeltaQ.Values()) {
		t.Fatalf("storage delta q=%v want %v", storage.DeltaQ.Values(), initial.DeltaQ.Values())
	}
	if err := storage.DeltaQ.Update(1); err != nil {
		t.Fatal(err)
	}

	var state tile.DecodeState
	if err := ctx.JobDecodeState(0, &state); err != nil {
		t.Fatal(err)
	}
	if err := ctx.RetainTileResidualCDFStorage(0, &state, &storage); err != nil {
		t.Fatal(err)
	}
	if retainedValid {
		t.Fatal("non-update tile retained frame context")
	}

	if err := ctx.JobDecodeState(1, &state); err != nil {
		t.Fatal(err)
	}
	if err := ctx.RetainTileResidualCDFStorage(1, &state, &storage); err != nil {
		t.Fatal(err)
	}
	if !retainedValid {
		t.Fatal("update tile did not retain frame context")
	}
	if !testCDFValuesEqual(retained.DeltaQ.Values(), storage.DeltaQ.Values()) {
		t.Fatalf("retained delta q=%v want %v", retained.DeltaQ.Values(), storage.DeltaQ.Values())
	}

	ctx.InitialTileResidualCDFs = nil
	if err := ctx.InitTileResidualCDFStorage(&storage); err != nil {
		t.Fatal(err)
	}
	if testCDFValuesEqual(storage.DeltaQ.Values(), initial.DeltaQ.Values()) {
		t.Fatal("default fallback unexpectedly reused initial frame context")
	}
}

func TestFrameWorkBatchJobDecodeStateRejectsInvalidInputs(t *testing.T) {
	ctx := FrameWorkBatch{
		Payload: []byte{0xaa},
		Jobs:    []tile.Job{{Offset: 0, Size: 2}},
	}
	var state tile.DecodeState
	if err := ctx.JobDecodeState(-1, &state); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("negative index err=%v want %v", err, ErrInvalidBatch)
	}
	if err := ctx.JobDecodeState(1, &state); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("large index err=%v want %v", err, ErrInvalidBatch)
	}
	if err := ctx.JobDecodeState(0, nil); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("nil state err=%v want %v", err, ErrInvalidBatch)
	}
	if err := ctx.JobDecodeState(0, &state); !errors.Is(err, tile.ErrInvalidPlan) {
		t.Fatalf("invalid range err=%v want %v", err, tile.ErrInvalidPlan)
	}

	if err := ctx.InitTileResidualCDFStorage(nil); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("nil cdf storage err=%v want %v", err, ErrInvalidBatch)
	}
	if err := ctx.RetainTileResidualCDFStorage(0, &state, nil); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("nil retained cdf storage err=%v want %v", err, ErrInvalidBatch)
	}
}

func testCDFValuesEqual(a []uint16, b []uint16) bool {
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

func TestFrameWorkBatchJobRegion64SuperblockClipsFrameEdge(t *testing.T) {
	ctx := FrameWorkBatch{
		FrameWorkFrameContext: FrameWorkFrameContext{
			Sequence: FrameWorkSequenceContextFromHeader(parser.SequenceHeader{
				ColorConfig: parser.ColorConfig{BitDepth: 8},
			}),
			FrameSize: parser.FrameSize{CodedWidth: 130, Height: 65},
		},
		Jobs: []tile.Job{
			{Tile: 1, Row: 0, Col: 1, SBX: 1, SBY: 0, SBCols: 2, SBRows: 2},
		},
	}
	region, err := ctx.JobRegion(0)
	if err != nil {
		t.Fatal(err)
	}
	want := FrameWorkJobRegion{
		Tile:        1,
		Row:         0,
		Col:         1,
		SBX:         1,
		SBY:         0,
		SBCols:      2,
		SBRows:      2,
		PixelX:      64,
		PixelY:      0,
		PixelWidth:  66,
		PixelHeight: 65,
		MIColStart:  16,
		MIRowStart:  0,
		MIColEnd:    34,
		MIRowEnd:    18,
	}
	if region != want {
		t.Fatalf("region=%+v want %+v", region, want)
	}
}

func TestFrameWorkBatchJobRegion128SuperblockClipsFrameEdge(t *testing.T) {
	ctx := FrameWorkBatch{
		FrameWorkFrameContext: FrameWorkFrameContext{
			Sequence:  testBatchSequenceContext(),
			FrameSize: parser.FrameSize{CodedWidth: 300, Height: 260},
		},
		Jobs: []tile.Job{
			{Tile: 3, Row: 1, Col: 1, SBX: 1, SBY: 1, SBCols: 2, SBRows: 2},
		},
	}
	region, err := ctx.JobRegion(0)
	if err != nil {
		t.Fatal(err)
	}
	want := FrameWorkJobRegion{
		Tile:        3,
		Row:         1,
		Col:         1,
		SBX:         1,
		SBY:         1,
		SBCols:      2,
		SBRows:      2,
		PixelX:      128,
		PixelY:      128,
		PixelWidth:  172,
		PixelHeight: 132,
		MIColStart:  32,
		MIRowStart:  32,
		MIColEnd:    76,
		MIRowEnd:    66,
	}
	if region != want {
		t.Fatalf("region=%+v want %+v", region, want)
	}
}

func TestFrameWorkBatchJobBlockDeltaContext(t *testing.T) {
	ctx := FrameWorkBatch{
		FrameWorkFrameContext: FrameWorkFrameContext{
			Sequence: FrameWorkSequenceContextFromHeader(parser.SequenceHeader{
				Use128x128Superblock: true,
				ColorConfig:          parser.ColorConfig{BitDepth: 10, MonoChrome: true},
			}),
			FrameSize: parser.FrameSize{CodedWidth: 256, Height: 256},
		},
		Jobs: []tile.Job{
			{Tile: 0, SBX: 1, SBY: 1, SBCols: 1, SBRows: 1},
		},
	}
	block, err := ctx.JobBlockDeltaContext(0, 32, 32, true, false)
	if err != nil {
		t.Fatal(err)
	}
	want := tile.BlockDeltaContext{
		MICol:          32,
		MIRow:          32,
		SBSizeMIB:      32,
		FullSuperblock: true,
		Monochrome:     true,
	}
	if block != want {
		t.Fatalf("block=%+v want %+v", block, want)
	}
	if _, err := ctx.JobBlockDeltaContext(0, 31, 32, false, false); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("outside block err=%v want %v", err, ErrInvalidBatch)
	}
}

func TestFrameWorkBatchRestorationPlaneGridAndRange(t *testing.T) {
	ctx := FrameWorkBatch{
		FrameWorkFrameContext: FrameWorkFrameContext{
			Sequence: testBatchSequenceContext(),
			FrameSize: parser.FrameSize{
				UpscaledWidth:       300,
				CodedWidth:          300,
				Height:              260,
				SuperResDenominator: 8,
			},
			Restoration: parser.RestorationParams{
				Type:       [3]parser.RestorationType{parser.RestorationWiener, parser.RestorationSGRProj},
				UnitSizeY:  128,
				UnitSizeUV: 64,
			},
		},
		Jobs: []tile.Job{
			{Tile: 3, Row: 1, Col: 1, SBX: 1, SBY: 1, SBCols: 2, SBRows: 2},
		},
	}
	grid, err := ctx.RestorationPlaneGrid(FrameWorkPlaneY)
	if err != nil {
		t.Fatal(err)
	}
	if grid.Type != parser.RestorationWiener || grid.HorzUnits != 2 || grid.VertUnits != 2 {
		t.Fatalf("grid=%+v", grid)
	}
	r, ok, err := ctx.JobRestorationUnitRange(0, FrameWorkPlaneY, 1, 1)
	if err != nil {
		t.Fatal(err)
	}
	if !ok || r != (tile.RestorationUnitRange{Col0: 1, Col1: 2, Row0: 1, Row1: 2}) {
		t.Fatalf("job origin range=%+v ok=%v", r, ok)
	}
	r, ok, err = ctx.JobRestorationUnitRange(0, FrameWorkPlaneY, 2, 2)
	if err != nil {
		t.Fatal(err)
	}
	if ok || r != (tile.RestorationUnitRange{}) {
		t.Fatalf("frame-edge range=%+v ok=%v", r, ok)
	}
	uv, ok, err := ctx.JobRestorationUnitRange(0, FrameWorkPlaneU, 1, 1)
	if err != nil {
		t.Fatal(err)
	}
	if !ok || uv != (tile.RestorationUnitRange{Col0: 1, Col1: 2, Row0: 1, Row1: 2}) {
		t.Fatalf("uv range=%+v ok=%v", uv, ok)
	}
	if _, _, err := ctx.JobRestorationUnitRange(0, FrameWorkPlaneY, 0, 0); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("outside restoration range err=%v want %v", err, ErrInvalidBatch)
	}
	if _, err := ctx.RestorationPlaneGrid(FrameWorkPlane(9)); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("bad plane grid err=%v want %v", err, ErrInvalidBatch)
	}
}

func TestFrameWorkBatchRestorationFrameAndLoopPlans(t *testing.T) {
	ctx := FrameWorkBatch{
		FrameWorkFrameContext: FrameWorkFrameContext{
			Sequence: testBatchSequenceContext(),
			FrameSize: parser.FrameSize{
				UpscaledWidth:       300,
				CodedWidth:          280,
				Height:              260,
				SuperResEnabled:     true,
				SuperResDenominator: 16,
			},
			CDEF: parser.CDEFParams{Bits: 1, StrengthCount: 2, YStrength: [parser.MaxCDEFStrengths]uint8{4}},
			Restoration: parser.RestorationParams{
				Type:       [3]parser.RestorationType{parser.RestorationWiener, parser.RestorationSGRProj, parser.RestorationNone},
				UnitSizeY:  128,
				UnitSizeUV: 64,
			},
		},
	}
	framePlan, err := ctx.RestorationFramePlan()
	if err != nil {
		t.Fatal(err)
	}
	if !framePlan.Active || framePlan.Planes != 3 || framePlan.UnitRecords != ([3]int{4, 4, 0}) {
		t.Fatalf("frame plan=%+v", framePlan)
	}

	plan, err := ctx.LoopRestorationPlan(false)
	if err != nil {
		t.Fatal(err)
	}
	if !plan.DoCDEF || !plan.DoSuperRes || !plan.DoLoopRestoration ||
		plan.OptimizedLoopRestoration || !plan.SaveBoundariesBeforeCDEF || !plan.SaveBoundariesAfterCDEF {
		t.Fatalf("loop restoration plan=%+v", plan)
	}

	skip, err := ctx.LoopRestorationPlan(true)
	if err != nil {
		t.Fatal(err)
	}
	if skip.DoCDEF || !skip.DoSuperRes || skip.OptimizedLoopRestoration ||
		!skip.SaveBoundariesBeforeCDEF || !skip.SaveBoundariesAfterCDEF {
		t.Fatalf("skip plan=%+v", skip)
	}
}

func TestFrameWorkBatchLoopRestorationOptimizedAndInactive(t *testing.T) {
	ctx := FrameWorkBatch{
		FrameWorkFrameContext: FrameWorkFrameContext{
			Sequence: testBatchSequenceContext(),
			FrameSize: parser.FrameSize{
				UpscaledWidth:       128,
				CodedWidth:          128,
				Height:              128,
				SuperResDenominator: 8,
			},
			Restoration: parser.RestorationParams{
				Type:      [3]parser.RestorationType{parser.RestorationWiener},
				UnitSizeY: 64,
			},
		},
	}
	plan, err := ctx.LoopRestorationPlan(false)
	if err != nil {
		t.Fatal(err)
	}
	if !plan.DoLoopRestoration || !plan.OptimizedLoopRestoration ||
		plan.SaveBoundariesBeforeCDEF || plan.SaveBoundariesAfterCDEF {
		t.Fatalf("optimized plan=%+v", plan)
	}

	ctx.Restoration = parser.RestorationParams{UnitSizeY: 256, UnitSizeUV: 256}
	ctx.CDEF = parser.CDEFParams{Bits: 1}
	plan, err = ctx.LoopRestorationPlan(false)
	if err != nil {
		t.Fatal(err)
	}
	if !plan.DoCDEF || plan.DoLoopRestoration || plan.OptimizedLoopRestoration ||
		plan.SaveBoundariesBeforeCDEF || plan.SaveBoundariesAfterCDEF {
		t.Fatalf("inactive plan=%+v", plan)
	}
}

func TestFrameWorkBatchRestorationPlansRejectInvalidInputs(t *testing.T) {
	ctx := FrameWorkBatch{}
	if _, err := ctx.RestorationFramePlan(); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("frame plan err=%v want %v", err, ErrInvalidBatch)
	}
	if _, err := ctx.LoopRestorationPlan(false); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("loop plan err=%v want %v", err, ErrInvalidBatch)
	}
}

func TestFrameWorkBatchJobOutputPlane420(t *testing.T) {
	output := testBatchFrame(t, frame.Format{
		Width:        130,
		Height:       65,
		BitDepth:     8,
		SubsamplingX: true,
		SubsamplingY: true,
		Align:        64,
	})
	ctx := FrameWorkBatch{
		Output: output,
		FrameWorkFrameContext: FrameWorkFrameContext{
			Sequence: FrameWorkSequenceContextFromHeader(parser.SequenceHeader{
				ColorConfig: parser.ColorConfig{BitDepth: 8, SubsamplingX: true, SubsamplingY: true},
			}),
			FrameSize: parser.FrameSize{CodedWidth: 130, Height: 65},
		},
		Jobs: []tile.Job{
			{Tile: 1, Row: 0, Col: 1, SBX: 1, SBY: 0, SBCols: 2, SBRows: 2},
		},
	}
	y, err := ctx.JobOutputPlane(0, FrameWorkPlaneY)
	if err != nil {
		t.Fatal(err)
	}
	if y.Plane != FrameWorkPlaneY || y.X != 64 || y.Y != 0 || y.Width != 66 || y.Height != 65 ||
		y.Stride != output.Y.Stride || y.BytesPerSample != 1 || y.RowBytes != 66 {
		t.Fatalf("Y plane region=%+v", y)
	}
	if len(y.Pix) != (y.Height-1)*y.Stride+y.RowBytes {
		t.Fatalf("Y len=%d region=%+v", len(y.Pix), y)
	}
	y.Pix[0] = 0x7a
	if output.Y.Pix[64] != 0x7a {
		t.Fatalf("Y plane did not alias")
	}

	u, err := ctx.JobOutputPlane(0, FrameWorkPlaneU)
	if err != nil {
		t.Fatal(err)
	}
	if u.Plane != FrameWorkPlaneU || u.X != 32 || u.Y != 0 || u.Width != 33 || u.Height != 33 ||
		u.Stride != output.U.Stride || u.BytesPerSample != 1 || u.RowBytes != 33 {
		t.Fatalf("U plane region=%+v", u)
	}
	u.Pix[0] = 0x55
	if output.U.Pix[32] != 0x55 {
		t.Fatalf("U plane did not alias")
	}

	v, err := ctx.JobOutputPlane(0, FrameWorkPlaneV)
	if err != nil {
		t.Fatal(err)
	}
	if v.Plane != FrameWorkPlaneV || v.X != 32 || v.Y != 0 || v.Width != 33 || v.Height != 33 ||
		v.Stride != output.V.Stride || v.BytesPerSample != 1 || v.RowBytes != 33 {
		t.Fatalf("V plane region=%+v", v)
	}
	v.Pix[0] = 0x33
	if output.V.Pix[32] != 0x33 {
		t.Fatalf("V plane did not alias")
	}
}

func TestFrameWorkBatchJobOutputPlaneHighBitDepth444(t *testing.T) {
	output := testBatchFrame(t, frame.Format{
		Width:    300,
		Height:   260,
		BitDepth: 10,
		Align:    64,
	})
	ctx := FrameWorkBatch{
		Output: output,
		FrameWorkFrameContext: FrameWorkFrameContext{
			Sequence: FrameWorkSequenceContextFromHeader(parser.SequenceHeader{
				Use128x128Superblock: true,
				ColorConfig:          parser.ColorConfig{BitDepth: 10},
			}),
			FrameSize: parser.FrameSize{CodedWidth: 300, Height: 260},
		},
		Jobs: []tile.Job{
			{Tile: 3, Row: 1, Col: 1, SBX: 1, SBY: 1, SBCols: 2, SBRows: 2},
		},
	}
	y, err := ctx.JobOutputPlane(0, FrameWorkPlaneY)
	if err != nil {
		t.Fatal(err)
	}
	if y.X != 128 || y.Y != 128 || y.Width != 172 || y.Height != 132 ||
		y.BytesPerSample != 2 || y.RowBytes != 344 {
		t.Fatalf("Y plane region=%+v", y)
	}
	wantOffset := 128*output.Y.Stride + 128*2
	y.Pix[1] = 0x9c
	if output.Y.Pix[wantOffset+1] != 0x9c {
		t.Fatalf("Y high-bit plane did not alias")
	}

	u, err := ctx.JobOutputPlane(0, FrameWorkPlaneU)
	if err != nil {
		t.Fatal(err)
	}
	if u.X != 128 || u.Y != 128 || u.Width != 172 || u.Height != 132 ||
		u.BytesPerSample != 2 || u.RowBytes != 344 {
		t.Fatalf("U 4:4:4 plane region=%+v", u)
	}
}

func TestFrameWorkBatchJobOutputPlaneRejectsInvalidInputs(t *testing.T) {
	output := testBatchFrame(t, frame.Format{
		Width:      128,
		Height:     128,
		BitDepth:   8,
		MonoChrome: true,
		Align:      32,
	})
	ctx := FrameWorkBatch{
		Output: output,
		FrameWorkFrameContext: FrameWorkFrameContext{
			Sequence: FrameWorkSequenceContextFromHeader(parser.SequenceHeader{
				ColorConfig: parser.ColorConfig{BitDepth: 8, MonoChrome: true},
			}),
			FrameSize: parser.FrameSize{CodedWidth: 128, Height: 128},
		},
		Jobs: []tile.Job{{SBCols: 1, SBRows: 1}},
	}
	if _, err := ctx.JobOutputPlane(0, FrameWorkPlaneU); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("monochrome U err=%v want %v", err, ErrInvalidBatch)
	}
	if _, err := ctx.JobOutputPlane(0, FrameWorkPlane(99)); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("invalid plane err=%v want %v", err, ErrInvalidBatch)
	}
	withoutOutput := ctx
	withoutOutput.Output = nil
	if _, err := withoutOutput.JobOutputPlane(0, FrameWorkPlaneY); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("nil output err=%v want %v", err, ErrInvalidBatch)
	}
	badLayout := ctx
	badLayout.Output = &frame.Frame{
		Format: output.Format,
		Layout: frame.Layout{},
		Y:      output.Y,
	}
	if _, err := badLayout.JobOutputPlane(0, FrameWorkPlaneY); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("bad layout err=%v want %v", err, ErrInvalidBatch)
	}
}

func TestFrameWorkBatchBlockQuantizerUsesCurrentSegmentQ(t *testing.T) {
	ctx := FrameWorkBatch{
		FrameWorkFrameContext: FrameWorkFrameContext{
			Sequence: FrameWorkSequenceContextFromHeader(parser.SequenceHeader{
				ColorConfig: parser.ColorConfig{BitDepth: 8},
			}),
			Quantization: parser.QuantizationParams{
				BaseQIdx: 20,
				UDCDelta: 1,
				UACDelta: 2,
			},
			Segmentation: parser.SegmentationParams{Enabled: true},
		},
	}
	ctx.Segmentation.Data.Segments[3].DeltaQ = -9

	got, lossless, err := ctx.BlockQuantizer(20, 3, FrameWorkPlaneU)
	if err != nil {
		t.Fatal(err)
	}
	want, err := quantize.PlaneQuantizer(ctx.Quantization, 11, 8, quantize.PlaneU)
	if err != nil {
		t.Fatal(err)
	}
	if got != want || lossless {
		t.Fatalf("quantizer=%+v lossless=%v want %+v false", got, lossless, want)
	}

	losslessCtx := ctx
	losslessCtx.Quantization = parser.QuantizationParams{}
	_, lossless, err = losslessCtx.BlockQuantizer(3, 3, FrameWorkPlaneY)
	if err != nil {
		t.Fatal(err)
	}
	if !lossless {
		t.Fatalf("lossless=false want true")
	}

	ctx.Segmentation.Enabled = false
	if _, _, err := ctx.BlockQuantizer(20, 3, FrameWorkPlaneY); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("disabled segment err=%v want %v", err, ErrInvalidBatch)
	}
}

func TestFrameWorkBatchReconstructBlockCoeffLuma(t *testing.T) {
	got := testBatchFrame(t, frame.Format{Width: 160, Height: 128, BitDepth: 8, Align: 64})
	want := testBatchFrame(t, got.Format)
	testFillFrame(got, 128)
	testFillFrame(want, 128)

	ctx := FrameWorkBatch{
		Output: got,
		FrameWorkFrameContext: FrameWorkFrameContext{
			Sequence: FrameWorkSequenceContextFromHeader(parser.SequenceHeader{
				ColorConfig: parser.ColorConfig{BitDepth: 8},
			}),
			FrameSize:    parser.FrameSize{CodedWidth: 160, Height: 128},
			Quantization: parser.QuantizationParams{BaseQIdx: 40},
			Segmentation: parser.SegmentationParams{Enabled: true},
		},
		Jobs: []tile.Job{{SBX: 1, SBY: 0, SBCols: 1, SBRows: 1}},
	}
	ctx.Segmentation.Data.Segments[2].DeltaQ = 5
	coeffs := make([]int16, 16)
	coeffs[0] = 3
	req := FrameWorkBlockCoeffReconstruction{
		Visit: tile.BlockVisit{
			MICol: 16, MIRow: 0, MIColEnd: 20, MIRowEnd: 4,
			X4: 0, Y4: 0, Size: tile.BlockSize16x16, VisibleW4: 4, VisibleH4: 4,
		},
		Block: tile.BlockCoeffBlock{
			Plane: 0,
			Block: tile.TransformBlock{X4: 1, Y4: 2, Size: tile.TransformSize4x4, VisibleW4: 1, VisibleH4: 1},
			Result: tile.TXBDecodeResult{
				EOB: 1,
			},
			Coeffs: coeffs,
		},
		Transform:     transform.TypeDCTDCT,
		CurrentQIndex: 40,
		SegmentID:     2,
	}
	plane, x, y, err := ctx.BlockCoeffPlanePosition(0, req.Visit, req.Block)
	if err != nil {
		t.Fatal(err)
	}
	if plane != FrameWorkPlaneY || x != 68 || y != 8 {
		t.Fatalf("position plane=%d x=%d y=%d", plane, x, y)
	}

	req.Int32Scratch, req.ResidualScratch = testBlockCoeffScratch(t, ctx, req, plane)
	if err := ctx.ReconstructBlockCoeff(0, req); err != nil {
		t.Fatal(err)
	}
	testReconstructBlockCoeffDirect(t, ctx, want, plane, x, y, req)
	if !bytes.Equal(got.Y.Pix, want.Y.Pix) {
		t.Fatalf("luma reconstruction did not match direct block reconstruction")
	}
}

func TestFrameWorkBatchReconstructBlockCoeffChroma420(t *testing.T) {
	got := testBatchFrame(t, frame.Format{
		Width:        160,
		Height:       160,
		BitDepth:     8,
		SubsamplingX: true,
		SubsamplingY: true,
		Align:        64,
	})
	want := testBatchFrame(t, got.Format)
	testFillFrame(got, 96)
	testFillFrame(want, 96)

	ctx := FrameWorkBatch{
		Output: got,
		FrameWorkFrameContext: FrameWorkFrameContext{
			Sequence: FrameWorkSequenceContextFromHeader(parser.SequenceHeader{
				ColorConfig: parser.ColorConfig{BitDepth: 8, SubsamplingX: true, SubsamplingY: true},
			}),
			FrameSize:    parser.FrameSize{CodedWidth: 160, Height: 160},
			Quantization: parser.QuantizationParams{BaseQIdx: 64, VDCDelta: -1, VACDelta: 2},
		},
		Jobs: []tile.Job{{SBX: 1, SBY: 1, SBCols: 1, SBRows: 1}},
	}
	coeffs := make([]int16, 16)
	coeffs[0] = -2
	req := FrameWorkBlockCoeffReconstruction{
		Visit: tile.BlockVisit{
			MICol: 16, MIRow: 16, MIColEnd: 24, MIRowEnd: 24,
			X4: 0, Y4: 0, Size: tile.BlockSize32x32, VisibleW4: 8, VisibleH4: 8,
		},
		Block: tile.BlockCoeffBlock{
			Plane: 2,
			Block: tile.TransformBlock{X4: 1, Y4: 2, Size: tile.TransformSize4x4, VisibleW4: 1, VisibleH4: 1},
			Result: tile.TXBDecodeResult{
				EOB: 1,
			},
			Coeffs: coeffs,
		},
		Transform:     transform.TypeDCTDCT,
		CurrentQIndex: 64,
	}
	plane, x, y, err := ctx.BlockCoeffPlanePosition(0, req.Visit, req.Block)
	if err != nil {
		t.Fatal(err)
	}
	if plane != FrameWorkPlaneV || x != 36 || y != 40 {
		t.Fatalf("position plane=%d x=%d y=%d", plane, x, y)
	}

	req.Int32Scratch, req.ResidualScratch = testBlockCoeffScratch(t, ctx, req, plane)
	if err := ctx.ReconstructBlockCoeff(0, req); err != nil {
		t.Fatal(err)
	}
	testReconstructBlockCoeffDirect(t, ctx, want, plane, x, y, req)
	if !bytes.Equal(got.V.Pix, want.V.Pix) {
		t.Fatalf("chroma reconstruction did not match direct block reconstruction")
	}
}

func TestFrameWorkBatchReconstructBlockCoeffChromaEdgeVisible(t *testing.T) {
	output := testBatchFrame(t, frame.Format{
		Width:        64,
		Height:       64,
		BitDepth:     8,
		SubsamplingX: true,
		SubsamplingY: true,
		Align:        64,
	})
	testFillFrame(output, 96)
	ctx := FrameWorkBatch{
		Output: output,
		FrameWorkFrameContext: FrameWorkFrameContext{
			Sequence: FrameWorkSequenceContextFromHeader(parser.SequenceHeader{
				ColorConfig: parser.ColorConfig{BitDepth: 8, SubsamplingX: true, SubsamplingY: true},
			}),
			FrameSize:    parser.FrameSize{CodedWidth: 64, Height: 64},
			Quantization: parser.QuantizationParams{BaseQIdx: 64, VDCDelta: -1, VACDelta: 2},
		},
		Jobs: []tile.Job{{SBX: 0, SBY: 0, SBCols: 1, SBRows: 1}},
	}
	coeffs := make([]int16, 16*16)
	coeffs[0] = -2
	req := FrameWorkBlockCoeffReconstruction{
		Visit: tile.BlockVisit{
			MICol: 0, MIRow: 0, MIColEnd: 16, MIRowEnd: 16,
			X4: 0, Y4: 0, Size: tile.BlockSize64x64, VisibleW4: 16, VisibleH4: 16,
		},
		Block: tile.BlockCoeffBlock{
			Plane: 1,
			Block: tile.TransformBlock{X4: 6, Y4: 0, Size: tile.TransformSize16x16, VisibleW4: 2, VisibleH4: 4},
			Result: tile.TXBDecodeResult{
				EOB: 1,
			},
			Coeffs: coeffs,
		},
		Transform:     transform.TypeDCTDCT,
		CurrentQIndex: 64,
	}
	plane, x, y, err := ctx.BlockCoeffPlanePosition(0, req.Visit, req.Block)
	if err != nil {
		t.Fatal(err)
	}
	if plane != FrameWorkPlaneU || x != 24 || y != 0 {
		t.Fatalf("position plane=%d x=%d y=%d", plane, x, y)
	}
	req.Int32Scratch, req.ResidualScratch = testBlockCoeffScratch(t, ctx, req, plane)
	if err := ctx.ReconstructBlockCoeff(0, req); err != nil {
		t.Fatal(err)
	}
	for row := 0; row < 16; row++ {
		if got := output.U.Pix[row*output.U.Stride+23]; got != 96 {
			t.Fatalf("left guard row=%d got=%d want 96", row, got)
		}
	}
}

func TestFrameWorkBatchReconstructBlockCoeffRejectsInvalidInputs(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, Align: 64})
	ctx := FrameWorkBatch{
		Output: output,
		FrameWorkFrameContext: FrameWorkFrameContext{
			Sequence: FrameWorkSequenceContextFromHeader(parser.SequenceHeader{
				ColorConfig: parser.ColorConfig{BitDepth: 8},
			}),
			FrameSize:    parser.FrameSize{CodedWidth: 64, Height: 64},
			Quantization: parser.QuantizationParams{BaseQIdx: 32},
		},
		Jobs: []tile.Job{{SBCols: 1, SBRows: 1}},
	}
	valid := FrameWorkBlockCoeffReconstruction{
		Visit: tile.BlockVisit{
			MICol: 0, MIRow: 0, MIColEnd: 4, MIRowEnd: 4,
			X4: 0, Y4: 0, Size: tile.BlockSize16x16, VisibleW4: 4, VisibleH4: 4,
		},
		Block: tile.BlockCoeffBlock{
			Plane:  0,
			Block:  tile.TransformBlock{X4: 0, Y4: 0, Size: tile.TransformSize4x4, VisibleW4: 1, VisibleH4: 1},
			Result: tile.TXBDecodeResult{EOB: 1},
			Coeffs: make([]int16, 16),
		},
		Transform:     transform.TypeDCTDCT,
		CurrentQIndex: 32,
	}
	valid.Int32Scratch, valid.ResidualScratch = testBlockCoeffScratch(t, ctx, valid, FrameWorkPlaneY)

	tests := []struct {
		name string
		ctx  FrameWorkBatch
		req  FrameWorkBlockCoeffReconstruction
	}{
		{name: "invalid plane", ctx: ctx, req: func() FrameWorkBlockCoeffReconstruction {
			req := valid
			req.Block.Plane = 3
			return req
		}()},
		{name: "disabled nonzero segment", ctx: ctx, req: func() FrameWorkBlockCoeffReconstruction {
			req := valid
			req.SegmentID = 1
			return req
		}()},
		{name: "out of range segment", ctx: func() FrameWorkBatch {
			next := ctx
			next.Segmentation.Enabled = true
			return next
		}(), req: func() FrameWorkBlockCoeffReconstruction {
			req := valid
			req.SegmentID = parser.MaxSegments
			return req
		}()},
		{name: "outside job plane", ctx: ctx, req: func() FrameWorkBlockCoeffReconstruction {
			req := valid
			req.Block.Block.X4 = 16
			return req
		}()},
		{name: "unsupported transform", ctx: ctx, req: func() FrameWorkBlockCoeffReconstruction {
			req := valid
			req.Transform = transform.TypeCount
			return req
		}()},
		{name: "lossless non 4x4", ctx: func() FrameWorkBatch {
			next := ctx
			next.Quantization = parser.QuantizationParams{}
			return next
		}(), req: func() FrameWorkBlockCoeffReconstruction {
			req := valid
			req.Block.Block.Size = tile.TransformSize8x8
			req.Block.Coeffs = make([]int16, 64)
			req.CurrentQIndex = 0
			req.Int32Scratch = nil
			req.ResidualScratch = nil
			return req
		}()},
		{name: "short scratch", ctx: ctx, req: func() FrameWorkBlockCoeffReconstruction {
			req := valid
			req.Int32Scratch = nil
			return req
		}()},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := tt.ctx.ReconstructBlockCoeff(0, tt.req); !errors.Is(err, ErrInvalidBatch) {
				t.Fatalf("err=%v want %v", err, ErrInvalidBatch)
			}
		})
	}
}

func TestFrameWorkBatchReconstructBlockCoeffAllocs(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, Align: 64})
	testFillFrame(output, 128)
	ctx := FrameWorkBatch{
		Output: output,
		FrameWorkFrameContext: FrameWorkFrameContext{
			Sequence: FrameWorkSequenceContextFromHeader(parser.SequenceHeader{
				ColorConfig: parser.ColorConfig{BitDepth: 8},
			}),
			FrameSize:    parser.FrameSize{CodedWidth: 64, Height: 64},
			Quantization: parser.QuantizationParams{BaseQIdx: 32},
		},
		Jobs: []tile.Job{{SBCols: 1, SBRows: 1}},
	}
	req := FrameWorkBlockCoeffReconstruction{
		Visit: tile.BlockVisit{
			MICol: 0, MIRow: 0, MIColEnd: 4, MIRowEnd: 4,
			X4: 0, Y4: 0, Size: tile.BlockSize16x16, VisibleW4: 4, VisibleH4: 4,
		},
		Block: tile.BlockCoeffBlock{
			Plane:  0,
			Block:  tile.TransformBlock{X4: 0, Y4: 0, Size: tile.TransformSize4x4, VisibleW4: 1, VisibleH4: 1},
			Result: tile.TXBDecodeResult{EOB: 1},
			Coeffs: make([]int16, 16),
		},
		Transform:     transform.TypeDCTDCT,
		CurrentQIndex: 32,
	}
	req.Int32Scratch, req.ResidualScratch = testBlockCoeffScratch(t, ctx, req, FrameWorkPlaneY)
	allocs := testing.AllocsPerRun(1000, func() {
		if err := ctx.ReconstructBlockCoeff(0, req); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("FrameWorkBatch.ReconstructBlockCoeff allocated: %f", allocs)
	}
}

func TestFrameWorkBatchReferencePlane(t *testing.T) {
	last := testBatchFrame(t, frame.Format{
		Width:        64,
		Height:       64,
		BitDepth:     8,
		SubsamplingX: true,
		SubsamplingY: true,
		Align:        32,
	})
	golden := testBatchFrame(t, frame.Format{
		Width:        150,
		Height:       129,
		BitDepth:     10,
		SubsamplingX: true,
		SubsamplingY: true,
		Align:        64,
	})
	references := [parser.InterRefsPerFrame]*frame.Frame{}
	references[int(FrameWorkReferenceLast)] = last
	references[int(FrameWorkReferenceGolden)] = golden
	ctx := FrameWorkBatch{References: references[:]}

	ref, err := ctx.ReferenceFrame(FrameWorkReferenceGolden)
	if err != nil {
		t.Fatal(err)
	}
	if ref != golden {
		t.Fatalf("reference=%p want %p", ref, golden)
	}

	u, err := ctx.ReferencePlane(FrameWorkReferenceGolden, FrameWorkPlaneU)
	if err != nil {
		t.Fatal(err)
	}
	if u.Plane != FrameWorkPlaneU || u.X != 0 || u.Y != 0 || u.Width != 75 || u.Height != 65 ||
		u.Stride != golden.U.Stride || u.BytesPerSample != 2 || u.RowBytes != 150 {
		t.Fatalf("U reference plane=%+v", u)
	}
	if len(u.Pix) != (u.Height-1)*u.Stride+u.RowBytes {
		t.Fatalf("U len=%d plane=%+v", len(u.Pix), u)
	}
	u.Pix[1] = 0x5a
	if golden.U.Pix[1] != 0x5a {
		t.Fatalf("U reference plane did not alias")
	}

	y, err := ctx.ReferencePlane(FrameWorkReferenceLast, FrameWorkPlaneY)
	if err != nil {
		t.Fatal(err)
	}
	if y.Width != 64 || y.Height != 64 || y.RowBytes != 64 || y.BytesPerSample != 1 {
		t.Fatalf("Y reference plane=%+v", y)
	}
}

func TestFrameWorkBatchJobReferencePlaneWindow420(t *testing.T) {
	reference := testBatchFrame(t, frame.Format{
		Width:        130,
		Height:       65,
		BitDepth:     8,
		SubsamplingX: true,
		SubsamplingY: true,
		Align:        64,
	})
	ctx := FrameWorkBatch{
		References: []*frame.Frame{reference},
		FrameWorkFrameContext: FrameWorkFrameContext{
			Sequence: FrameWorkSequenceContextFromHeader(parser.SequenceHeader{
				ColorConfig: parser.ColorConfig{BitDepth: 8, SubsamplingX: true, SubsamplingY: true},
			}),
			FrameSize: parser.FrameSize{CodedWidth: 130, Height: 65},
		},
		Jobs: []tile.Job{
			{Tile: 1, Row: 0, Col: 1, SBX: 1, SBY: 0, SBCols: 2, SBRows: 2},
		},
	}
	y, err := ctx.JobReferencePlane(0, FrameWorkReferenceLast, FrameWorkPlaneY)
	if err != nil {
		t.Fatal(err)
	}
	if y.X != 64 || y.Y != 0 || y.Width != 66 || y.Height != 65 || y.RowBytes != 66 {
		t.Fatalf("Y reference job plane=%+v", y)
	}
	if len(y.Pix) != (y.Height-1)*y.Stride+y.RowBytes {
		t.Fatalf("Y len=%d plane=%+v", len(y.Pix), y)
	}
	y.Pix[0] = 0x91
	if reference.Y.Pix[64] != 0x91 {
		t.Fatalf("Y reference job plane did not alias")
	}

	u, err := ctx.JobReferencePlaneWindow(0, FrameWorkReferenceLast, FrameWorkPlaneU, 2, 3)
	if err != nil {
		t.Fatal(err)
	}
	if u.X != 30 || u.Y != 0 || u.Width != 35 || u.Height != 33 ||
		u.Stride != reference.U.Stride || u.BytesPerSample != 1 || u.RowBytes != 35 {
		t.Fatalf("U reference job window=%+v", u)
	}
	if len(u.Pix) != (u.Height-1)*u.Stride+u.RowBytes {
		t.Fatalf("U len=%d plane=%+v", len(u.Pix), u)
	}
	u.Pix[0] = 0x42
	if reference.U.Pix[30] != 0x42 {
		t.Fatalf("U reference job window did not alias")
	}
}

func TestFrameWorkBatchJobReferencePlaneWindowClipsReference(t *testing.T) {
	reference := testBatchFrame(t, frame.Format{
		Width:    200,
		Height:   180,
		BitDepth: 10,
		Align:    64,
	})
	ctx := FrameWorkBatch{
		References: []*frame.Frame{reference},
		FrameWorkFrameContext: FrameWorkFrameContext{
			Sequence: FrameWorkSequenceContextFromHeader(parser.SequenceHeader{
				Use128x128Superblock: true,
				ColorConfig:          parser.ColorConfig{BitDepth: 10},
			}),
			FrameSize: parser.FrameSize{CodedWidth: 300, Height: 260},
		},
		Jobs: []tile.Job{
			{Tile: 3, Row: 1, Col: 1, SBX: 1, SBY: 1, SBCols: 2, SBRows: 2},
		},
	}
	y, err := ctx.JobReferencePlaneWindow(0, FrameWorkReferenceLast, FrameWorkPlaneY, 16, 16)
	if err != nil {
		t.Fatal(err)
	}
	if y.X != 112 || y.Y != 112 || y.Width != 88 || y.Height != 68 ||
		y.BytesPerSample != 2 || y.RowBytes != 176 {
		t.Fatalf("clipped Y reference job window=%+v", y)
	}
	wantOffset := 112*reference.Y.Stride + 112*2
	y.Pix[1] = 0x37
	if reference.Y.Pix[wantOffset+1] != 0x37 {
		t.Fatalf("clipped Y reference job window did not alias")
	}
}

func TestFrameWorkBatchReferencePlaneRejectsInvalidInputs(t *testing.T) {
	mono := testBatchFrame(t, frame.Format{
		Width:      64,
		Height:     64,
		BitDepth:   8,
		MonoChrome: true,
		Align:      32,
	})
	ctx := FrameWorkBatch{References: []*frame.Frame{mono}}
	if _, err := ctx.ReferenceFrame(FrameWorkReferenceLast2); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("missing reference err=%v want %v", err, ErrInvalidBatch)
	}
	if _, err := ctx.ReferenceFrame(FrameWorkReference(parser.InterRefsPerFrame)); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("invalid reference err=%v want %v", err, ErrInvalidBatch)
	}
	if _, err := (FrameWorkBatch{References: []*frame.Frame{nil}}).ReferenceFrame(FrameWorkReferenceLast); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("nil reference err=%v want %v", err, ErrInvalidBatch)
	}
	if _, err := ctx.ReferencePlane(FrameWorkReferenceLast, FrameWorkPlaneU); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("monochrome U err=%v want %v", err, ErrInvalidBatch)
	}
	if _, err := ctx.ReferencePlane(FrameWorkReferenceLast, FrameWorkPlane(99)); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("invalid plane err=%v want %v", err, ErrInvalidBatch)
	}

	badLayout := *mono
	badLayout.Layout.BytesPerSample = 0
	if _, err := (FrameWorkBatch{References: []*frame.Frame{&badLayout}}).ReferencePlane(FrameWorkReferenceLast, FrameWorkPlaneY); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("bad layout err=%v want %v", err, ErrInvalidBatch)
	}

	if _, err := ctx.JobReferencePlane(0, FrameWorkReferenceLast, FrameWorkPlaneY); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("missing job context err=%v want %v", err, ErrInvalidBatch)
	}
	outside := FrameWorkBatch{
		References: []*frame.Frame{mono},
		FrameWorkFrameContext: FrameWorkFrameContext{
			Sequence: FrameWorkSequenceContextFromHeader(parser.SequenceHeader{
				ColorConfig: parser.ColorConfig{BitDepth: 8, MonoChrome: true},
			}),
			FrameSize: parser.FrameSize{CodedWidth: 128, Height: 128},
		},
		Jobs: []tile.Job{{SBX: 1, SBY: 0, SBCols: 1, SBRows: 1}},
	}
	if _, err := outside.JobReferencePlaneWindow(0, FrameWorkReferenceLast, FrameWorkPlaneY, 0, 0); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("outside reference err=%v want %v", err, ErrInvalidBatch)
	}
}

func TestFrameWorkBatchJobRegionRejectsInvalidInputs(t *testing.T) {
	validRegionBatch := func() FrameWorkBatch {
		return FrameWorkBatch{
			FrameWorkFrameContext: FrameWorkFrameContext{
				Sequence:  testBatchSequenceContext(),
				FrameSize: parser.FrameSize{CodedWidth: 128, Height: 128},
			},
			Jobs: []tile.Job{{SBCols: 1, SBRows: 1}},
		}
	}
	valid := validRegionBatch()
	if _, err := valid.JobRegion(-1); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("negative index err=%v want %v", err, ErrInvalidBatch)
	}
	if _, err := valid.JobRegion(1); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("large index err=%v want %v", err, ErrInvalidBatch)
	}

	invalid := validRegionBatch()
	invalid.Sequence = FrameWorkSequenceContext{}
	if _, err := invalid.JobRegion(0); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("invalid sequence err=%v want %v", err, ErrInvalidBatch)
	}
	invalid = validRegionBatch()
	invalid.FrameSize = parser.FrameSize{}
	if _, err := invalid.JobRegion(0); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("invalid frame size err=%v want %v", err, ErrInvalidBatch)
	}
	invalid = validRegionBatch()
	invalid.Jobs[0].SBCols = 0
	if _, err := invalid.JobRegion(0); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("zero job err=%v want %v", err, ErrInvalidBatch)
	}
	invalid = validRegionBatch()
	invalid.Jobs = []tile.Job{{SBX: 8, SBCols: 1, SBRows: 1}}
	if _, err := invalid.JobRegion(0); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("outside frame err=%v want %v", err, ErrInvalidBatch)
	}
}

func TestFrameWorkBatchJobUpdatesFrameContext(t *testing.T) {
	ctx := FrameWorkBatch{
		Jobs: []tile.Job{
			{Tile: 0},
			{Tile: 1, UpdatesFrameContext: true},
		},
	}
	updates, err := ctx.JobUpdatesFrameContext(0)
	if err != nil {
		t.Fatal(err)
	}
	if updates {
		t.Fatal("job 0 updates frame context")
	}
	updates, err = ctx.JobUpdatesFrameContext(1)
	if err != nil {
		t.Fatal(err)
	}
	if !updates {
		t.Fatal("job 1 does not update frame context")
	}
	if _, err := ctx.JobUpdatesFrameContext(-1); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("negative index err=%v want %v", err, ErrInvalidBatch)
	}
	if _, err := ctx.JobUpdatesFrameContext(2); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("large index err=%v want %v", err, ErrInvalidBatch)
	}
}

func TestBuildBatchesAllocs(t *testing.T) {
	jobs := testJobs()
	var batches [4]Batch

	allocs := testing.AllocsPerRun(1000, func() {
		_, err := BuildBatches(batches[:], jobs[:], 3)
		if err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("BuildBatches allocated: %f", allocs)
	}
}

func TestFrameWorkBatchJobPayloadAllocs(t *testing.T) {
	payload := []byte{0xaa, 0xbb, 0xcc}
	ctx := FrameWorkBatch{
		Payload: payload,
		Jobs: []tile.Job{
			{Offset: 0, Size: 1},
			{Offset: 1, Size: 2},
		},
	}
	allocs := testing.AllocsPerRun(1000, func() {
		data, err := ctx.JobPayload(1)
		if err != nil {
			t.Fatal(err)
		}
		if len(data) != 2 {
			t.Fatalf("payload=%v", data)
		}
		if err := ctx.ValidatePayloads(); err != nil {
			t.Fatal(err)
		}
		updates, err := ctx.JobUpdatesFrameContext(1)
		if err != nil {
			t.Fatal(err)
		}
		if updates {
			t.Fatal("job updates frame context")
		}
	})
	if allocs != 0 {
		t.Fatalf("FrameWorkBatch.JobPayload allocated: %f", allocs)
	}
}

func TestFrameWorkBatchJobEntropyReaderAllocs(t *testing.T) {
	ctx := FrameWorkBatch{
		Payload:          []byte{0x00, 0xff, 0x00},
		DisableCDFUpdate: true,
		Jobs: []tile.Job{
			{Offset: 0, Size: 1},
			{Offset: 1, Size: 1},
		},
	}
	allocs := testing.AllocsPerRun(1000, func() {
		r, err := ctx.JobEntropyReader(1)
		if err != nil {
			t.Fatal(err)
		}
		if r.AllowCDFUpdate() {
			t.Fatal("CDF update enabled")
		}
		bit, err := r.ReadBit()
		if err != nil {
			t.Fatal(err)
		}
		if bit != 1 {
			t.Fatalf("bit=%d want 1", bit)
		}
	})
	if allocs != 0 {
		t.Fatalf("FrameWorkBatch.JobEntropyReader allocated: %f", allocs)
	}
}

func TestFrameWorkBatchJobDecodeStateAllocs(t *testing.T) {
	ctx := FrameWorkBatch{
		Payload: []byte{0x00, 0xff, 0x00},
		FrameWorkFrameContext: FrameWorkFrameContext{
			Sequence:     testBatchSequenceContext(),
			FrameSize:    parser.FrameSize{CodedWidth: 128, Height: 64},
			TileInfo:     testBatchTileInfo(),
			Quantization: parser.QuantizationParams{BaseQIdx: 91},
			Segmentation: parser.SegmentationParams{QIndex: [parser.MaxSegments]uint8{91}},
			Delta:        parser.DeltaParams{DeltaQPresent: true, DeltaQResLog2: 1},
			LoopFilter:   parser.LoopFilterParams{LevelY: [2]uint8{6, 7}},
			CDEF:         parser.CDEFParams{Damping: 4, StrengthCount: 1},
			Restoration:  parser.RestorationParams{UnitSizeY: 128},
			TransformRef: parser.TransformReferenceParams{TransformMode: parser.TransformModeLargest},
			SkipMode:     parser.SkipModeParams{Allowed: true},
			FrameMode:    parser.FrameModeParams{ReducedTxSet: true},
			GlobalMotion: testBatchGlobalMotion(),
			FilmGrain:    testBatchFilmGrain(),
		},
		Jobs: []tile.Job{
			{Offset: 0, Size: 1},
			{Offset: 1, Size: 1, UpdatesFrameContext: true},
		},
	}
	var state tile.DecodeState
	allocs := testing.AllocsPerRun(1000, func() {
		if err := ctx.JobDecodeState(1, &state); err != nil {
			t.Fatal(err)
		}
		if !state.RetainFrameContext {
			t.Fatal("frame context not retained")
		}
		if state.CurrentBaseQIdx != 91 {
			t.Fatalf("CurrentBaseQIdx=%d want 91", state.CurrentBaseQIdx)
		}
		bit, err := state.Reader.ReadBit()
		if err != nil {
			t.Fatal(err)
		}
		if bit != 1 {
			t.Fatalf("bit=%d want 1", bit)
		}
	})
	if allocs != 0 {
		t.Fatalf("FrameWorkBatch.JobDecodeState allocated: %f", allocs)
	}
}

func TestFrameWorkBatchJobRegionAllocs(t *testing.T) {
	output := testBatchFrame(t, frame.Format{
		Width:        300,
		Height:       260,
		BitDepth:     10,
		SubsamplingX: true,
		SubsamplingY: true,
		Align:        64,
	})
	ctx := FrameWorkBatch{
		Output:     output,
		References: []*frame.Frame{output},
		FrameWorkFrameContext: FrameWorkFrameContext{
			Sequence: FrameWorkSequenceContextFromHeader(parser.SequenceHeader{
				Use128x128Superblock: true,
				ColorConfig: parser.ColorConfig{
					BitDepth:     10,
					SubsamplingX: true,
					SubsamplingY: true,
				},
			}),
			FrameSize: parser.FrameSize{CodedWidth: 300, Height: 260},
		},
		Jobs: []tile.Job{
			{Tile: 3, Row: 1, Col: 1, SBX: 1, SBY: 1, SBCols: 2, SBRows: 2},
		},
	}
	allocs := testing.AllocsPerRun(1000, func() {
		region, err := ctx.JobRegion(0)
		if err != nil {
			t.Fatal(err)
		}
		if region.PixelWidth != 172 || region.MIColEnd != 76 {
			t.Fatalf("region=%+v", region)
		}
		block, err := ctx.JobBlockDeltaContext(0, region.MIColStart, region.MIRowStart, false, true)
		if err != nil {
			t.Fatal(err)
		}
		if block.SBSizeMIB != 32 || !block.SkipTransform {
			t.Fatalf("block=%+v", block)
		}
		plane, err := ctx.JobOutputPlane(0, FrameWorkPlaneU)
		if err != nil {
			t.Fatal(err)
		}
		if plane.X != 64 || plane.Y != 64 || plane.Width != 86 || plane.Height != 66 || plane.RowBytes != 172 {
			t.Fatalf("plane=%+v", plane)
		}
		ref, err := ctx.ReferenceFrame(FrameWorkReferenceLast)
		if err != nil {
			t.Fatal(err)
		}
		if ref != output {
			t.Fatalf("reference=%p want %p", ref, output)
		}
		refPlane, err := ctx.ReferencePlane(FrameWorkReferenceLast, FrameWorkPlaneY)
		if err != nil {
			t.Fatal(err)
		}
		if refPlane.Width != 300 || refPlane.Height != 260 || refPlane.RowBytes != 600 {
			t.Fatalf("reference plane=%+v", refPlane)
		}
		jobRefPlane, err := ctx.JobReferencePlaneWindow(0, FrameWorkReferenceLast, FrameWorkPlaneU, 4, 5)
		if err != nil {
			t.Fatal(err)
		}
		if jobRefPlane.X != 60 || jobRefPlane.Y != 59 || jobRefPlane.Width != 90 || jobRefPlane.Height != 71 || jobRefPlane.RowBytes != 180 {
			t.Fatalf("job reference plane=%+v", jobRefPlane)
		}
	})
	if allocs != 0 {
		t.Fatalf("FrameWorkBatch.JobRegion allocated: %f", allocs)
	}
}

func TestFrameWorkBatchLoopRestorationPlanAllocs(t *testing.T) {
	ctx := FrameWorkBatch{
		FrameWorkFrameContext: FrameWorkFrameContext{
			Sequence: testBatchSequenceContext(),
			FrameSize: parser.FrameSize{
				UpscaledWidth:       300,
				CodedWidth:          280,
				Height:              260,
				SuperResEnabled:     true,
				SuperResDenominator: 16,
			},
			CDEF: parser.CDEFParams{Bits: 1, StrengthCount: 2, YStrength: [parser.MaxCDEFStrengths]uint8{4}},
			Restoration: parser.RestorationParams{
				Type:       [3]parser.RestorationType{parser.RestorationWiener, parser.RestorationSGRProj},
				UnitSizeY:  128,
				UnitSizeUV: 64,
			},
		},
	}
	allocs := testing.AllocsPerRun(1000, func() {
		plan, err := ctx.LoopRestorationPlan(false)
		if err != nil {
			t.Fatal(err)
		}
		if !plan.DoCDEF || !plan.DoSuperRes || !plan.DoLoopRestoration {
			t.Fatalf("plan=%+v", plan)
		}
	})
	if allocs != 0 {
		t.Fatalf("FrameWorkBatch.LoopRestorationPlan allocated: %f", allocs)
	}
}

func FuzzBuildBatches(f *testing.F) {
	f.Add([]byte{2, 3, 2, 2, 2, 3, 3, 2, 3})
	f.Add([]byte{8, 1, 1})

	f.Fuzz(func(t *testing.T, data []byte) {
		if len(data) < 3 {
			return
		}
		workers := int(data[0] & 7)
		count := int(data[1]&7) + 1
		if count > 8 {
			count = 8
		}
		if len(data) < 2+count {
			return
		}
		var jobs [8]tile.Job
		for i := 0; i < count; i++ {
			sb := data[2+i]
			jobs[i] = tile.Job{
				Tile:   uint16(i),
				SBCols: uint16(sb&7) + 1,
				SBRows: uint16((sb>>3)&7) + 1,
			}
		}
		var batches [8]Batch
		n, err := BuildBatches(batches[:], jobs[:count], workers)
		if err != nil {
			return
		}
		if n == 0 || n > workers || n > count {
			t.Fatalf("n=%d workers=%d count=%d", n, workers, count)
		}
		next := 0
		for i := 0; i < n; i++ {
			batch := batches[i]
			if batch.FirstJob != next || batch.Count <= 0 || batch.FirstJob+batch.Count > count || batch.Units == 0 {
				t.Fatalf("batch[%d]=%+v next=%d count=%d", i, batch, next, count)
			}
			next += batch.Count
		}
		if next != count {
			t.Fatalf("covered=%d count=%d", next, count)
		}
	})
}

func FuzzFrameWorkBatchJobPayload(f *testing.F) {
	f.Add(uint8(3), int16(0), int16(1))
	f.Add(uint8(3), int16(1), int16(2))
	f.Add(uint8(1), int16(0), int16(2))

	f.Fuzz(func(t *testing.T, payloadLen uint8, offset int16, size int16) {
		payload := make([]byte, int(payloadLen%64))
		job := tile.Job{Offset: int(offset), Size: int(size)}
		ctx := FrameWorkBatch{Payload: payload, Jobs: []tile.Job{job}}

		data, err := ctx.JobPayload(0)
		if err != nil {
			if _, _, rangeErr := job.PayloadRange(len(payload)); rangeErr == nil {
				t.Fatalf("JobPayload err=%v payloadLen=%d job=%+v", err, len(payload), job)
			}
			return
		}

		start, end, err := job.PayloadRange(len(payload))
		if err != nil {
			t.Fatalf("PayloadRange err=%v after JobPayload success", err)
		}
		if len(data) != end-start {
			t.Fatalf("len=%d want %d", len(data), end-start)
		}
		if len(data) != 0 && &data[0] != &payload[start] {
			t.Fatalf("payload does not alias")
		}
		if err := ctx.ValidatePayloads(); err != nil {
			t.Fatalf("ValidatePayloads err=%v", err)
		}
	})
}

func FuzzFrameWorkBatchJobEntropyReader(f *testing.F) {
	f.Add([]byte{0xff}, int16(0), int16(1), false, uint8(0), false, uint8(0))
	f.Add([]byte{0x00, 0xff, 0x00}, int16(1), int16(1), true, uint8(73), true, uint8(2))
	f.Add([]byte{0xaa}, int16(0), int16(2), false, uint8(255), true, uint8(7))

	f.Fuzz(func(t *testing.T, payload []byte, offset int16, size int16, disableCDFUpdate bool, baseQIdx uint8, deltaQPresent bool, deltaQResLog2 uint8) {
		if len(payload) > 64 {
			return
		}
		job := tile.Job{Offset: int(offset), Size: int(size)}
		delta := parser.DeltaParams{DeltaQPresent: deltaQPresent, DeltaQResLog2: deltaQResLog2 & 3}
		ctx := FrameWorkBatch{
			Payload: payload,
			FrameWorkFrameContext: FrameWorkFrameContext{
				Sequence:     testBatchSequenceContext(),
				FrameSize:    parser.FrameSize{CodedWidth: uint32(len(payload)) + 1, Height: 1},
				TileInfo:     testBatchTileInfo(),
				Quantization: parser.QuantizationParams{BaseQIdx: baseQIdx},
				Segmentation: parser.SegmentationParams{QIndex: [parser.MaxSegments]uint8{baseQIdx}},
				Delta:        delta,
			},
			DisableCDFUpdate: disableCDFUpdate,
			Jobs:             []tile.Job{job},
		}
		r, err := ctx.JobEntropyReader(0)
		if err != nil {
			if _, _, rangeErr := job.PayloadRange(len(payload)); rangeErr == nil {
				t.Fatalf("JobEntropyReader err=%v payloadLen=%d job=%+v", err, len(payload), job)
			}
			return
		}
		var state tile.DecodeState
		if err := ctx.JobDecodeState(0, &state); err != nil {
			t.Fatalf("JobDecodeState err=%v after JobEntropyReader success", err)
		}
		if r.AllowCDFUpdate() == disableCDFUpdate {
			t.Fatalf("AllowCDFUpdate=%v disableCDFUpdate=%v", r.AllowCDFUpdate(), disableCDFUpdate)
		}
		wantRetain := job.UpdatesFrameContext && !disableCDFUpdate
		if state.RetainFrameContext != wantRetain {
			t.Fatalf("RetainFrameContext=%v want %v", state.RetainFrameContext, wantRetain)
		}
		if state.CurrentBaseQIdx != baseQIdx {
			t.Fatalf("CurrentBaseQIdx=%d want %d", state.CurrentBaseQIdx, baseQIdx)
		}
		if ctx.Delta != delta {
			t.Fatalf("Delta=%+v want %+v", ctx.Delta, delta)
		}
		if _, err := r.ReadBit(); err != nil {
			t.Fatalf("ReadBit err=%v", err)
		}
	})
}

func FuzzFrameWorkBatchJobRegion(f *testing.F) {
	f.Add(uint16(130), uint16(65), false, uint16(1), uint16(0), uint16(2), uint16(2))
	f.Add(uint16(300), uint16(260), true, uint16(1), uint16(1), uint16(2), uint16(2))
	f.Add(uint16(16), uint16(16), false, uint16(8), uint16(0), uint16(1), uint16(1))

	f.Fuzz(func(t *testing.T, width uint16, height uint16, use128 bool, sbx uint16, sby uint16, sbCols uint16, sbRows uint16) {
		if width == 0 || height == 0 {
			return
		}
		format := frame.Format{
			Width:        int(width),
			Height:       int(height),
			BitDepth:     8,
			MonoChrome:   width&1 == 1,
			SubsamplingX: true,
			SubsamplingY: true,
			Align:        64,
		}
		output := testBatchFrame(t, format)
		references := [parser.InterRefsPerFrame]*frame.Frame{output}
		seq := parser.SequenceHeader{
			Use128x128Superblock: use128,
			ColorConfig: parser.ColorConfig{
				BitDepth:     8,
				MonoChrome:   format.MonoChrome,
				SubsamplingX: format.SubsamplingX,
				SubsamplingY: format.SubsamplingY,
			},
		}
		ctx := FrameWorkBatch{
			Output:     output,
			References: references[:1],
			FrameWorkFrameContext: FrameWorkFrameContext{
				Sequence: FrameWorkSequenceContextFromHeader(seq),
				FrameSize: parser.FrameSize{
					CodedWidth: uint32(width),
					Height:     uint32(height),
				},
			},
			Jobs: []tile.Job{{
				Tile:   0,
				SBX:    sbx & 63,
				SBY:    sby & 63,
				SBCols: sbCols & 7,
				SBRows: sbRows & 7,
			}},
		}
		region, err := ctx.JobRegion(0)
		if err != nil {
			return
		}
		if region.PixelWidth == 0 || region.PixelHeight == 0 ||
			region.PixelX+region.PixelWidth > uint32(width) ||
			region.PixelY+region.PixelHeight > uint32(height) ||
			region.MIColStart >= region.MIColEnd ||
			region.MIRowStart >= region.MIRowEnd {
			t.Fatalf("invalid region=%+v width=%d height=%d", region, width, height)
		}
		block, err := ctx.JobBlockDeltaContext(0, region.MIColStart, region.MIRowStart, false, false)
		if err != nil {
			t.Fatalf("block context err=%v region=%+v", err, region)
		}
		if block.SBSizeMIB != ctx.Sequence.SBSizeMIB || block.Monochrome != ctx.Sequence.ColorConfig.MonoChrome {
			t.Fatalf("block=%+v ctx=%+v", block, ctx.Sequence)
		}
		yPlane, err := ctx.JobOutputPlane(0, FrameWorkPlaneY)
		if err != nil {
			t.Fatalf("Y plane err=%v region=%+v", err, region)
		}
		if yPlane.Width == 0 || yPlane.Height == 0 || len(yPlane.Pix) == 0 {
			t.Fatalf("invalid Y plane=%+v", yPlane)
		}
		refPlane, err := ctx.ReferencePlane(FrameWorkReferenceLast, FrameWorkPlaneY)
		if err != nil {
			t.Fatalf("reference plane err=%v region=%+v", err, region)
		}
		if refPlane.Width != int(width) || refPlane.Height != int(height) || len(refPlane.Pix) == 0 {
			t.Fatalf("invalid reference plane=%+v", refPlane)
		}
		jobRefPlane, err := ctx.JobReferencePlaneWindow(0, FrameWorkReferenceLast, FrameWorkPlaneY, uint32(width&7), uint32(height&7))
		if err != nil {
			t.Fatalf("job reference plane err=%v region=%+v", err, region)
		}
		if jobRefPlane.Width == 0 || jobRefPlane.Height == 0 || len(jobRefPlane.Pix) == 0 ||
			jobRefPlane.X < 0 || jobRefPlane.Y < 0 ||
			jobRefPlane.X+jobRefPlane.Width > int(width) ||
			jobRefPlane.Y+jobRefPlane.Height > int(height) {
			t.Fatalf("invalid job reference plane=%+v", jobRefPlane)
		}
		if !format.MonoChrome {
			uPlane, err := ctx.JobOutputPlane(0, FrameWorkPlaneU)
			if err != nil {
				t.Fatalf("U plane err=%v region=%+v", err, region)
			}
			if uPlane.Width == 0 || uPlane.Height == 0 || len(uPlane.Pix) == 0 {
				t.Fatalf("invalid U plane=%+v", uPlane)
			}
		}
	})
}

func FuzzFrameWorkBatchLoopRestorationPlan(f *testing.F) {
	f.Add(uint16(300), uint16(260), uint8(2), uint8(3), uint8(0), uint8(1), uint8(4), false, true, true, true, false)
	f.Add(uint16(128), uint16(128), uint8(2), uint8(0), uint8(0), uint8(0), uint8(0), false, false, false, false, false)
	f.Add(uint16(64), uint16(64), uint8(0), uint8(0), uint8(0), uint8(1), uint8(9), true, true, true, false, true)

	f.Fuzz(func(t *testing.T, rawW uint16, rawH uint16, rawY uint8, rawU uint8, rawV uint8, rawCDEFBits uint8, rawCDEFStrength uint8, mono bool, ssX bool, ssY bool, superres bool, skipLoopFilter bool) {
		types := [...]parser.RestorationType{
			parser.RestorationNone,
			parser.RestorationSwitchable,
			parser.RestorationWiener,
			parser.RestorationSGRProj,
		}
		unitSizes := [...]uint16{64, 128, 256}
		width := uint32(rawW%512) + 1
		height := uint32(rawH%512) + 1
		size := parser.FrameSize{
			UpscaledWidth:       width,
			CodedWidth:          width,
			Height:              height,
			SuperResDenominator: 8,
		}
		if superres {
			size.SuperResEnabled = true
			size.SuperResDenominator = 16
		}
		ctx := FrameWorkBatch{
			FrameWorkFrameContext: FrameWorkFrameContext{
				Sequence: FrameWorkSequenceContextFromHeader(parser.SequenceHeader{
					Use128x128Superblock: rawW&1 == 1,
					EnableCDEF:           true,
					EnableRestoration:    true,
					ColorConfig: parser.ColorConfig{
						BitDepth:     8,
						MonoChrome:   mono,
						SubsamplingX: ssX,
						SubsamplingY: ssY,
					},
				}),
				FrameSize: size,
				CDEF: parser.CDEFParams{
					Bits:          rawCDEFBits & 3,
					StrengthCount: 1 << (rawCDEFBits & 3),
					YStrength:     [parser.MaxCDEFStrengths]uint8{rawCDEFStrength & 63},
					UVStrength:    [parser.MaxCDEFStrengths]uint8{(rawCDEFStrength >> 1) & 63},
				},
				Restoration: parser.RestorationParams{
					Type:       [3]parser.RestorationType{types[rawY&3], types[rawU&3], types[rawV&3]},
					UnitSizeY:  unitSizes[rawY%uint8(len(unitSizes))],
					UnitSizeUV: unitSizes[(rawU^rawV)%uint8(len(unitSizes))],
				},
			},
		}
		plan, err := ctx.LoopRestorationPlan(skipLoopFilter)
		if err != nil {
			t.Fatalf("LoopRestorationPlan err=%v", err)
		}
		wantCDEF := !skipLoopFilter && (ctx.CDEF.Bits != 0 || ctx.CDEF.YStrength[0] != 0 || ctx.CDEF.UVStrength[0] != 0)
		wantOptimized := !wantCDEF && !size.SuperResEnabled
		if plan.DoCDEF != wantCDEF ||
			plan.DoSuperRes != size.SuperResEnabled ||
			plan.OptimizedLoopRestoration != wantOptimized ||
			plan.DoLoopRestoration != plan.Restoration.Active ||
			plan.SaveBoundariesBeforeCDEF != (plan.Restoration.Active && !wantOptimized) ||
			plan.SaveBoundariesAfterCDEF != (plan.Restoration.Active && !wantOptimized) {
			t.Fatalf("plan=%+v wantCDEF=%v wantOptimized=%v", plan, wantCDEF, wantOptimized)
		}
		wantPlanes := uint8(3)
		if mono {
			wantPlanes = 1
		}
		if plan.Restoration.Planes != wantPlanes {
			t.Fatalf("planes=%d want %d", plan.Restoration.Planes, wantPlanes)
		}
	})
}

func FuzzFrameWorkBatchReconstructBlockCoeff(f *testing.F) {
	f.Add(uint8(0), uint8(0), uint8(0), uint8(64), int16(1), uint8(0))
	f.Add(uint8(2), uint8(7), uint8(9), uint8(1), int16(-3), uint8(1))
	f.Add(uint8(1), uint8(14), uint8(13), uint8(255), int16(4), uint8(0))

	f.Fuzz(func(t *testing.T, rawPlane uint8, rawX4 uint8, rawY4 uint8, qIndex uint8, coeff int16, rawSize uint8) {
		output := testBatchFrame(t, frame.Format{
			Width:        128,
			Height:       128,
			BitDepth:     8,
			SubsamplingX: true,
			SubsamplingY: true,
			Align:        64,
		})
		testFillFrame(output, 128)
		ctx := FrameWorkBatch{
			Output: output,
			FrameWorkFrameContext: FrameWorkFrameContext{
				Sequence: FrameWorkSequenceContextFromHeader(parser.SequenceHeader{
					ColorConfig: parser.ColorConfig{BitDepth: 8, SubsamplingX: true, SubsamplingY: true},
				}),
				FrameSize:    parser.FrameSize{CodedWidth: 128, Height: 128},
				Quantization: parser.QuantizationParams{BaseQIdx: qIndex},
			},
			Jobs: []tile.Job{{SBCols: 1, SBRows: 1}},
		}
		sizes := [...]tile.TransformSize{tile.TransformSize4x4, tile.TransformSize8x8}
		size := sizes[rawSize%uint8(len(sizes))]
		dims, ok := size.Dimensions()
		if !ok {
			t.Fatal("invalid transform size")
		}
		planeID := int(rawPlane % 3)
		slotsX := 16
		slotsY := 16
		if planeID > 0 {
			slotsX = 8
			slotsY = 8
		}
		xLimit := slotsX - int(dims.W4)
		yLimit := slotsY - int(dims.H4)
		scanSize, err := size.TransformSize()
		if err != nil {
			t.Fatal(err)
		}
		scan, err := transform.ScanSize(scanSize)
		if err != nil {
			t.Fatal(err)
		}
		coeffs := make([]int16, scan.Width*scan.Height)
		coeffs[0] = coeff % 8
		req := FrameWorkBlockCoeffReconstruction{
			Visit: tile.BlockVisit{
				MICol: 0, MIRow: 0, MIColEnd: 16, MIRowEnd: 16,
				X4: 0, Y4: 0, Size: tile.BlockSize64x64, VisibleW4: 16, VisibleH4: 16,
			},
			Block: tile.BlockCoeffBlock{
				Plane: planeID,
				Block: tile.TransformBlock{
					X4:        int(rawX4) % (xLimit + 1),
					Y4:        int(rawY4) % (yLimit + 1),
					Size:      size,
					VisibleW4: dims.W4,
					VisibleH4: dims.H4,
				},
				Result: tile.TXBDecodeResult{EOB: 1},
				Coeffs: coeffs,
			},
			Transform:     transform.TypeDCTDCT,
			CurrentQIndex: qIndex,
		}
		plane, _, _, err := ctx.BlockCoeffPlanePosition(0, req.Visit, req.Block)
		if err != nil {
			t.Fatalf("BlockCoeffPlanePosition err=%v", err)
		}
		txSize, err := req.Block.Block.Size.TransformSize()
		if err != nil {
			t.Fatal(err)
		}
		q, lossless, err := ctx.BlockQuantizer(req.CurrentQIndex, req.SegmentID, plane)
		if err != nil {
			t.Fatal(err)
		}
		int32Len, int16Len, err := reconstruct.ScratchLen(reconstruct.Block{
			Size:      txSize,
			Transform: req.Transform,
			Quantizer: q,
			Lossless:  lossless,
			EOB:       req.Block.Result.EOB,
		})
		if err != nil {
			return
		}
		req.Int32Scratch = make([]int32, int32Len)
		req.ResidualScratch = make([]int16, int16Len)
		if err := ctx.ReconstructBlockCoeff(0, req); err != nil {
			if errors.Is(err, ErrInvalidBatch) {
				return
			}
			t.Fatalf("ReconstructBlockCoeff err=%v", err)
		}
	})
}

func BenchmarkBuildBatches(b *testing.B) {
	jobs := testJobs()
	var batches [4]Batch

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, _ = BuildBatches(batches[:], jobs[:], 3)
	}
}

func BenchmarkFrameWorkBatchJobPayload(b *testing.B) {
	payload := []byte{0xaa, 0xbb, 0xcc}
	ctx := FrameWorkBatch{
		Payload: payload,
		Jobs: []tile.Job{
			{Offset: 0, Size: 1},
			{Offset: 1, Size: 2},
		},
	}

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, _ = ctx.JobPayload(1)
	}
}

func BenchmarkFrameWorkBatchJobEntropyReader(b *testing.B) {
	ctx := FrameWorkBatch{
		Payload:          []byte{0x00, 0xff, 0x00},
		DisableCDFUpdate: true,
		Jobs: []tile.Job{
			{Offset: 0, Size: 1},
			{Offset: 1, Size: 1},
		},
	}

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, _ = ctx.JobEntropyReader(1)
	}
}

func BenchmarkFrameWorkBatchJobDecodeState(b *testing.B) {
	ctx := FrameWorkBatch{
		Payload: []byte{0x00, 0xff, 0x00},
		FrameWorkFrameContext: FrameWorkFrameContext{
			Sequence:     testBatchSequenceContext(),
			FrameSize:    parser.FrameSize{CodedWidth: 128, Height: 64},
			TileInfo:     testBatchTileInfo(),
			Quantization: parser.QuantizationParams{BaseQIdx: 37},
			Segmentation: parser.SegmentationParams{QIndex: [parser.MaxSegments]uint8{37}},
			Delta:        parser.DeltaParams{DeltaQPresent: true},
			LoopFilter:   parser.LoopFilterParams{LevelY: [2]uint8{1, 2}},
			CDEF:         parser.CDEFParams{Damping: 3},
			Restoration:  parser.RestorationParams{UnitSizeY: 64},
			TransformRef: parser.TransformReferenceParams{TransformMode: parser.TransformModeSwitchable},
			SkipMode:     parser.SkipModeParams{Allowed: true},
			FrameMode:    parser.FrameModeParams{AllowWarpedMotion: true},
			GlobalMotion: testBatchGlobalMotion(),
			FilmGrain:    testBatchFilmGrain(),
		},
		Jobs: []tile.Job{
			{Offset: 0, Size: 1},
			{Offset: 1, Size: 1, UpdatesFrameContext: true},
		},
	}
	var state tile.DecodeState

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = ctx.JobDecodeState(1, &state)
	}
}

func BenchmarkFrameWorkBatchJobRegion(b *testing.B) {
	ctx := FrameWorkBatch{
		FrameWorkFrameContext: FrameWorkFrameContext{
			Sequence:  testBatchSequenceContext(),
			FrameSize: parser.FrameSize{CodedWidth: 300, Height: 260},
		},
		Jobs: []tile.Job{
			{Tile: 3, Row: 1, Col: 1, SBX: 1, SBY: 1, SBCols: 2, SBRows: 2},
		},
	}

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, _ = ctx.JobRegion(0)
	}
}

func BenchmarkFrameWorkBatchJobBlockDeltaContext(b *testing.B) {
	ctx := FrameWorkBatch{
		FrameWorkFrameContext: FrameWorkFrameContext{
			Sequence:  testBatchSequenceContext(),
			FrameSize: parser.FrameSize{CodedWidth: 300, Height: 260},
		},
		Jobs: []tile.Job{
			{Tile: 3, Row: 1, Col: 1, SBX: 1, SBY: 1, SBCols: 2, SBRows: 2},
		},
	}

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, _ = ctx.JobBlockDeltaContext(0, 32, 32, false, false)
	}
}

func BenchmarkFrameWorkBatchJobRestorationUnitRange(b *testing.B) {
	ctx := FrameWorkBatch{
		FrameWorkFrameContext: FrameWorkFrameContext{
			Sequence: testBatchSequenceContext(),
			FrameSize: parser.FrameSize{
				UpscaledWidth:       300,
				CodedWidth:          300,
				Height:              260,
				SuperResDenominator: 8,
			},
			Restoration: parser.RestorationParams{
				Type:       [3]parser.RestorationType{parser.RestorationWiener},
				UnitSizeY:  128,
				UnitSizeUV: 64,
			},
		},
		Jobs: []tile.Job{
			{Tile: 3, Row: 1, Col: 1, SBX: 1, SBY: 1, SBCols: 2, SBRows: 2},
		},
	}

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, _, _ = ctx.JobRestorationUnitRange(0, FrameWorkPlaneY, 2, 2)
	}
}

func BenchmarkFrameWorkBatchLoopRestorationPlan(b *testing.B) {
	ctx := FrameWorkBatch{
		FrameWorkFrameContext: FrameWorkFrameContext{
			Sequence: testBatchSequenceContext(),
			FrameSize: parser.FrameSize{
				UpscaledWidth:       300,
				CodedWidth:          280,
				Height:              260,
				SuperResEnabled:     true,
				SuperResDenominator: 16,
			},
			CDEF: parser.CDEFParams{Bits: 1, StrengthCount: 2, YStrength: [parser.MaxCDEFStrengths]uint8{4}},
			Restoration: parser.RestorationParams{
				Type:       [3]parser.RestorationType{parser.RestorationWiener, parser.RestorationSGRProj},
				UnitSizeY:  128,
				UnitSizeUV: 64,
			},
		},
	}

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, _ = ctx.LoopRestorationPlan(false)
	}
}

func BenchmarkFrameWorkBatchJobOutputPlane(b *testing.B) {
	output := benchmarkBatchFrame(b, frame.Format{
		Width:        300,
		Height:       260,
		BitDepth:     10,
		SubsamplingX: true,
		SubsamplingY: true,
		Align:        64,
	})
	ctx := FrameWorkBatch{
		Output: output,
		FrameWorkFrameContext: FrameWorkFrameContext{
			Sequence: FrameWorkSequenceContextFromHeader(parser.SequenceHeader{
				Use128x128Superblock: true,
				ColorConfig: parser.ColorConfig{
					BitDepth:     10,
					SubsamplingX: true,
					SubsamplingY: true,
				},
			}),
			FrameSize: parser.FrameSize{CodedWidth: 300, Height: 260},
		},
		Jobs: []tile.Job{
			{Tile: 3, Row: 1, Col: 1, SBX: 1, SBY: 1, SBCols: 2, SBRows: 2},
		},
	}

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, _ = ctx.JobOutputPlane(0, FrameWorkPlaneU)
	}
}

func BenchmarkFrameWorkBatchReferencePlane(b *testing.B) {
	reference := benchmarkBatchFrame(b, frame.Format{
		Width:        300,
		Height:       260,
		BitDepth:     10,
		SubsamplingX: true,
		SubsamplingY: true,
		Align:        64,
	})
	ctx := FrameWorkBatch{References: []*frame.Frame{reference}}

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, _ = ctx.ReferencePlane(FrameWorkReferenceLast, FrameWorkPlaneU)
	}
}

func BenchmarkFrameWorkBatchJobReferencePlaneWindow(b *testing.B) {
	reference := benchmarkBatchFrame(b, frame.Format{
		Width:        300,
		Height:       260,
		BitDepth:     10,
		SubsamplingX: true,
		SubsamplingY: true,
		Align:        64,
	})
	ctx := FrameWorkBatch{
		References: []*frame.Frame{reference},
		FrameWorkFrameContext: FrameWorkFrameContext{
			Sequence: FrameWorkSequenceContextFromHeader(parser.SequenceHeader{
				Use128x128Superblock: true,
				ColorConfig: parser.ColorConfig{
					BitDepth:     10,
					SubsamplingX: true,
					SubsamplingY: true,
				},
			}),
			FrameSize: parser.FrameSize{CodedWidth: 300, Height: 260},
		},
		Jobs: []tile.Job{
			{Tile: 3, Row: 1, Col: 1, SBX: 1, SBY: 1, SBCols: 2, SBRows: 2},
		},
	}

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, _ = ctx.JobReferencePlaneWindow(0, FrameWorkReferenceLast, FrameWorkPlaneU, 4, 5)
	}
}

func BenchmarkFrameWorkBatchJobUpdatesFrameContext(b *testing.B) {
	ctx := FrameWorkBatch{
		Jobs: []tile.Job{
			{Tile: 0},
			{Tile: 1, UpdatesFrameContext: true},
		},
	}

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, _ = ctx.JobUpdatesFrameContext(1)
	}
}

func testJobs() [4]tile.Job {
	return [4]tile.Job{
		{Tile: 0, SBCols: 3, SBRows: 2},
		{Tile: 1, SBCols: 2, SBRows: 2},
		{Tile: 2, SBCols: 3, SBRows: 3},
		{Tile: 3, SBCols: 2, SBRows: 3},
	}
}

func testBatchTileInfo() parser.TileInfo {
	tiles := parser.TileInfo{
		SBCols: 2,
		SBRows: 1,
		Cols:   2,
		Rows:   1,
	}
	tiles.ColStartSB[1] = 1
	tiles.ColStartSB[2] = 2
	tiles.RowStartSB[1] = 1
	return tiles
}

func testBatchFrame(t *testing.T, format frame.Format) *frame.Frame {
	t.Helper()
	layout, err := frame.RequiredSize(format)
	if err != nil {
		t.Fatal(err)
	}
	buffer := make([]byte, layout.Size)
	output, err := frame.Bind(buffer, format)
	if err != nil {
		t.Fatal(err)
	}
	return &output
}

func testFillFrame(dst *frame.Frame, value byte) {
	for i := range dst.Y.Pix {
		dst.Y.Pix[i] = value
	}
	for i := range dst.U.Pix {
		dst.U.Pix[i] = value
	}
	for i := range dst.V.Pix {
		dst.V.Pix[i] = value
	}
}

func testBlockCoeffScratch(t *testing.T, ctx FrameWorkBatch, req FrameWorkBlockCoeffReconstruction, plane FrameWorkPlane) ([]int32, []int16) {
	t.Helper()
	size, err := req.Block.Block.Size.TransformSize()
	if err != nil {
		t.Fatal(err)
	}
	q, lossless, err := ctx.BlockQuantizer(req.CurrentQIndex, req.SegmentID, plane)
	if err != nil {
		t.Fatal(err)
	}
	cfg := reconstruct.Block{
		Size:      size,
		Transform: req.Transform,
		Quantizer: q,
		Lossless:  lossless,
		EOB:       req.Block.Result.EOB,
	}
	int32Len, int16Len, err := reconstruct.ScratchLen(cfg)
	if err != nil {
		t.Fatal(err)
	}
	return make([]int32, int32Len), make([]int16, int16Len)
}

func testReconstructBlockCoeffDirect(t *testing.T, ctx FrameWorkBatch, dst *frame.Frame, plane FrameWorkPlane, x int, y int, req FrameWorkBlockCoeffReconstruction) {
	t.Helper()
	size, err := req.Block.Block.Size.TransformSize()
	if err != nil {
		t.Fatal(err)
	}
	scanSize, err := transform.ScanSize(size)
	if err != nil {
		t.Fatal(err)
	}
	q, lossless, err := ctx.BlockQuantizer(req.CurrentQIndex, req.SegmentID, plane)
	if err != nil {
		t.Fatal(err)
	}
	cfg := reconstruct.Block{
		Size:      size,
		Transform: req.Transform,
		Quantizer: q,
		Lossless:  lossless,
		EOB:       req.Block.Result.EOB,
	}
	int32Scratch, residualScratch := testBlockCoeffScratch(t, ctx, req, plane)
	dstPlane, _, _, ok := frameWorkFramePlane(dst, plane)
	if !ok {
		t.Fatal("invalid frame plane")
	}
	if err := reconstruct.ReconstructPlaneBlock(dstPlane, dst.Layout.BytesPerSample, ctx.Sequence.ColorConfig.BitDepth,
		x, y, req.Block.Coeffs, scanSize.Height, int32Scratch, residualScratch, cfg); err != nil {
		t.Fatal(err)
	}
}

func benchmarkBatchFrame(b *testing.B, format frame.Format) *frame.Frame {
	b.Helper()
	layout, err := frame.RequiredSize(format)
	if err != nil {
		b.Fatal(err)
	}
	buffer := make([]byte, layout.Size)
	output, err := frame.Bind(buffer, format)
	if err != nil {
		b.Fatal(err)
	}
	return &output
}

func testBatchSequenceContext() FrameWorkSequenceContext {
	return FrameWorkSequenceContextFromHeader(parser.SequenceHeader{
		SeqProfile:           1,
		Use128x128Superblock: true,
		EnableOrderHint:      true,
		OrderHintBits:        5,
		EnableCDEF:           true,
		EnableRestoration:    true,
		ColorConfig: parser.ColorConfig{
			BitDepth:     10,
			SubsamplingX: true,
			SubsamplingY: true,
		},
		FilmGrainParamsPresent: true,
	})
}

func testBatchGlobalMotion() parser.GlobalMotionParams {
	motion := parser.DefaultGlobalMotionParams()
	motion.Ref[0].Type = parser.GlobalMotionTranslation
	motion.Ref[0].Matrix[0] = 17
	motion.Ref[0].Matrix[1] = -9
	return motion
}

func testBatchFilmGrain() parser.FilmGrainParams {
	return parser.FilmGrainParams{
		ParamsPresent: true,
		Apply:         true,
		Seed:          99,
		BitDepth:      8,
		NumYPoints:    1,
		YPoints:       [parser.MaxFilmGrainYPoints][2]uint8{{16, 32}},
	}
}
