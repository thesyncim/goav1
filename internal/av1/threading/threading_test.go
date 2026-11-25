package threading

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"github.com/thesyncim/goav1/internal/av1/entropy"
	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/lfmask"
	"github.com/thesyncim/goav1/internal/av1/loopfilter"
	"github.com/thesyncim/goav1/internal/av1/motion"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/prediction"
	"github.com/thesyncim/goav1/internal/av1/quantize"
	"github.com/thesyncim/goav1/internal/av1/reconstruct"
	"github.com/thesyncim/goav1/internal/av1/tile"
	"github.com/thesyncim/goav1/internal/av1/transform"
	decodework "github.com/thesyncim/goav1/internal/av1/work"
	"github.com/thesyncim/goav1/internal/poison"
	"math/rand"
	"reflect"
	"slices"
	"sync"
	"sync/atomic"
	"testing"
	"time"
	"unsafe"
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
	for i := range n {
		if batches[i].Worker != uint16(i) || batches[i].Count != 1 || int(batches[i].FirstJob) != i {
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
	for i := range a {
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
	if !framePlan.Active || framePlan.Planes != 3 || framePlan.UnitRecords != ([3]uint32{4, 4, 0}) {
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
	// 130x65 frame, SBX=1 col, SBCols=2. The visible coded width is 130 so
	// the second SB contributes a visible 130-64=66 pixel strip (Width=66).
	// ReadWidth caps neighbor reads at the MI-aligned extent mi_cols*4 = 34*4 =
	// 136 luma pixels (mi_cols = ((130+7)>>3)<<1 = 34), so ReadWidth = 136-64 =
	// 72. ClipWidth is the writable extent at the superblock-aligned edge
	// (SBX+SBCols)*64 = 192, clamped to the allocation (YStride/bps = 192), so
	// ClipWidth = 192-64 = 128. ReadHeight caps at mi_rows*4 = 18*4 = 72
	// (mi_rows = ((65+7)>>3)<<1 = 18); ClipHeight is the SB-aligned writable
	// edge (SBRows*64 = 128, clamped to the SB-aligned allocation 128), = 128.
	if y.Plane != FrameWorkPlaneY || y.X != 64 || y.Y != 0 || y.Width != 66 || y.Height != 65 ||
		int(y.Stride) != output.Y.Stride || y.BytesPerSample != 1 || y.RowBytes != 66 ||
		y.ReadWidth != 72 || y.ReadHeight != 72 ||
		y.ClipWidth != 128 || y.ClipHeight != 128 || y.ClipRowBytes != 128 {
		t.Fatalf("Y plane region=%+v", y)
	}
	if len(y.Pix) != int((y.ClipHeight-1)*y.Stride+y.ClipRowBytes) {
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
	// Chroma 4:2:0: visible width 130/2 = 65 -> strip 65-32 = 33 (Width=33).
	// ReadWidth = MI-aligned (luma 136 -> chroma 68): 68-32 = 36. ClipWidth =
	// SB-aligned (luma 192 -> chroma 96): 96-32 = 64. ReadHeight = luma 72 >> 1
	// = 36; ClipHeight = luma 128 >> 1 = 64.
	if u.Plane != FrameWorkPlaneU || u.X != 32 || u.Y != 0 || u.Width != 33 || u.Height != 33 ||
		int(u.Stride) != output.U.Stride || u.BytesPerSample != 1 || u.RowBytes != 33 ||
		u.ReadWidth != 36 || u.ReadHeight != 36 ||
		u.ClipWidth != 64 || u.ClipHeight != 64 || u.ClipRowBytes != 64 {
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
		int(v.Stride) != output.V.Stride || v.BytesPerSample != 1 || v.RowBytes != 33 ||
		v.ReadWidth != 36 || v.ReadHeight != 36 ||
		v.ClipWidth != 64 || v.ClipHeight != 64 || v.ClipRowBytes != 64 {
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
		int(u.Stride) != golden.U.Stride || u.BytesPerSample != 2 || u.RowBytes != 150 {
		t.Fatalf("U reference plane=%+v", u)
	}
	if len(u.Pix) != int((u.Height-1)*u.Stride+u.RowBytes) {
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
	if len(y.Pix) != int((y.Height-1)*y.Stride+y.RowBytes) {
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
		int(u.Stride) != reference.U.Stride || u.BytesPerSample != 1 || u.RowBytes != 35 {
		t.Fatalf("U reference job window=%+v", u)
	}
	if len(u.Pix) != int((u.Height-1)*u.Stride+u.RowBytes) {
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
	nilRefBatch := FrameWorkBatch{References: []*frame.Frame{nil}}
	if _, err := nilRefBatch.ReferenceFrame(FrameWorkReferenceLast); !errors.Is(err, ErrInvalidBatch) {
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
	badLayoutBatch := FrameWorkBatch{References: []*frame.Frame{&badLayout}}
	if _, err := badLayoutBatch.ReferencePlane(FrameWorkReferenceLast, FrameWorkPlaneY); !errors.Is(err, ErrInvalidBatch) {
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
		block, err := ctx.JobBlockDeltaContext(0, uint32(region.MIColStart), uint32(region.MIRowStart), false, true)
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
		count := min(int(data[1]&7)+1, 8)
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
		for i := range n {
			batch := batches[i]
			firstJob := int(batch.FirstJob)
			batchCount := int(batch.Count)
			if firstJob != next || batchCount <= 0 || firstJob+batchCount > count || batch.Units == 0 {
				t.Fatalf("batch[%d]=%+v next=%d count=%d", i, batch, next, count)
			}
			next += batchCount
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
		job := tile.Job{Offset: uint32(offset), Size: uint32(size)}
		if offset < 0 || size < 0 {
			job = tile.Job{Offset: ^uint32(0), Size: 1}
		}
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
		job := tile.Job{Offset: uint32(offset), Size: uint32(size)}
		if offset < 0 || size < 0 {
			job = tile.Job{Offset: ^uint32(0), Size: 1}
		}
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
		block, err := ctx.JobBlockDeltaContext(0, uint32(region.MIColStart), uint32(region.MIRowStart), false, false)
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
		if int(refPlane.Width) != int(width) || int(refPlane.Height) != int(height) || len(refPlane.Pix) == 0 {
			t.Fatalf("invalid reference plane=%+v", refPlane)
		}
		jobRefPlane, err := ctx.JobReferencePlaneWindow(0, FrameWorkReferenceLast, FrameWorkPlaneY, uint32(width&7), uint32(height&7))
		if err != nil {
			t.Fatalf("job reference plane err=%v region=%+v", err, region)
		}
		if jobRefPlane.Width == 0 || jobRefPlane.Height == 0 || len(jobRefPlane.Pix) == 0 ||
			jobRefPlane.X+jobRefPlane.Width > uint32(width) ||
			jobRefPlane.Y+jobRefPlane.Height > uint32(height) {
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

func testBatchFrame(t testing.TB, format frame.Format) *frame.Frame {
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

func TestFrameWorkBlockCoeffTransformMetaMatchesHelpers(t *testing.T) {
	sizes := [...]tile.TransformSize{
		tile.TransformSize4x4,
		tile.TransformSize8x8,
		tile.TransformSize16x16,
		tile.TransformSize32x32,
		tile.TransformSize64x64,
		tile.TransformSize4x8,
		tile.TransformSize8x4,
		tile.TransformSize8x16,
		tile.TransformSize16x8,
		tile.TransformSize16x32,
		tile.TransformSize32x16,
		tile.TransformSize32x64,
		tile.TransformSize64x32,
		tile.TransformSize4x16,
		tile.TransformSize16x4,
		tile.TransformSize8x32,
		tile.TransformSize32x8,
		tile.TransformSize16x64,
		tile.TransformSize64x16,
	}
	for _, txSize := range sizes {
		meta, ok := frameWorkBlockCoeffTransformMeta(txSize)
		if !ok {
			t.Fatalf("metadata missing for tx size %d", txSize)
		}
		size, err := txSize.TransformSize()
		if err != nil {
			t.Fatalf("TransformSize(%d): %v", txSize, err)
		}
		if meta.size != size {
			t.Fatalf("size metadata for %d = %+v, want %+v", txSize, meta.size, size)
		}
		scanSize, err := transform.ScanSize(size)
		if err != nil {
			t.Fatalf("ScanSize(%+v): %v", size, err)
		}
		if meta.scanSize != scanSize {
			t.Fatalf("scan metadata for %d = %+v, want %+v", txSize, meta.scanSize, scanSize)
		}
		txScale, err := quantize.TransformScale(int(size.Width), int(size.Height))
		if err != nil {
			t.Fatalf("TransformScale(%+v): %v", size, err)
		}
		if meta.txScale != txScale {
			t.Fatalf("scale metadata for %d = %d, want %d", txSize, meta.txScale, txScale)
		}
	}
	if _, ok := frameWorkBlockCoeffTransformMeta(tile.TransformSize(255)); ok {
		t.Fatal("invalid transform size unexpectedly had metadata")
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

func TestFrameWorkBatchCDEFIndexMapShapeAndBind(t *testing.T) {
	ctx := testFrameWorkCDEFIndexBatch(130, 65, 2)
	cols, rows, length, err := ctx.CDEFIndexMapShape()
	if err != nil {
		t.Fatal(err)
	}
	if cols != 3 || rows != 2 || length != 6 {
		t.Fatalf("shape cols=%d rows=%d length=%d want 3,2,6", cols, rows, length)
	}

	index := make([]uint8, length+2)
	read := make([]bool, length+2)
	index[5] = 3
	read[5] = true
	got, err := ctx.BindCDEFIndexMap(index, read)
	if err != nil {
		t.Fatal(err)
	}
	if got.Stride != uint16(cols) || got.Rows != uint16(rows) || len(got.Index) != length || len(got.Read) != length {
		t.Fatalf("map=%+v len(index)=%d len(read)=%d", got, len(got.Index), len(got.Read))
	}

	short := make([]uint8, length-1)
	if _, err := ctx.BindCDEFIndexMap(short, read); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("short index err=%v want %v", err, ErrInvalidBatch)
	}

	badStored := make([]uint8, length)
	badRead := make([]bool, length)
	badStored[0] = 4
	badRead[0] = true
	if _, err := ctx.BindCDEFIndexMap(badStored, badRead); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("bad stored index err=%v want %v", err, ErrInvalidBatch)
	}

	badBits := ctx
	badBits.CDEF.Bits = 4
	if _, err := badBits.BindCDEFIndexMap(index, read); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("bad cdef bits err=%v want %v", err, ErrInvalidBatch)
	}

	emptyBatch := FrameWorkBatch{}
	if _, _, _, err := emptyBatch.CDEFIndexMapShape(); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("invalid batch shape err=%v want %v", err, ErrInvalidBatch)
	}

	oversized := testFrameWorkCDEFIndexBatch(1<<22, 64, 2)
	if _, _, _, err := oversized.CDEFIndexMapShape(); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("oversized cdef shape err=%v want %v", err, ErrInvalidBatch)
	}
}

func TestFrameWorkCDEFIndexMapMarkBlockMatchesLibaomUnits(t *testing.T) {
	ctx := testFrameWorkCDEFIndexBatch(130, 65, 2)
	_, _, length, err := ctx.CDEFIndexMapShape()
	if err != nil {
		t.Fatal(err)
	}
	cdefMap, err := ctx.BindCDEFIndexMap(make([]uint8, length), make([]bool, length))
	if err != nil {
		t.Fatal(err)
	}
	visit := tile.BlockLoopVisit{
		Block: tile.BlockVisit{
			MICol: 15, MIRow: 15, MIColEnd: 33, MIRowEnd: 18,
		},
		Prefix: tile.BlockModeResult{CDEFIndex: 3},
	}
	if err := cdefMap.MarkBlock(ctx.CDEF, visit); err != nil {
		t.Fatal(err)
	}
	for i := range cdefMap.Index {
		if !cdefMap.Read[i] || cdefMap.Index[i] != 3 {
			t.Fatalf("unit %d read=%v index=%d want true,3", i, cdefMap.Read[i], cdefMap.Index[i])
		}
	}
}

func TestFrameWorkCDEFIndexMapMarkBlockSkipsUncodedSyntax(t *testing.T) {
	ctx := testFrameWorkCDEFIndexBatch(128, 64, 2)
	_, _, length, err := ctx.CDEFIndexMapShape()
	if err != nil {
		t.Fatal(err)
	}
	cdefMap, err := ctx.BindCDEFIndexMap(make([]uint8, length), make([]bool, length))
	if err != nil {
		t.Fatal(err)
	}
	skip := tile.BlockLoopVisit{
		Block:  tile.BlockVisit{MICol: 0, MIRow: 0, MIColEnd: 16, MIRowEnd: 16},
		Prefix: tile.BlockModeResult{SkipTransform: true, CDEFIndex: 3},
	}
	if err := cdefMap.MarkBlock(ctx.CDEF, skip); err != nil {
		t.Fatal(err)
	}
	if cdefMap.Read[0] || cdefMap.Index[0] != 0 {
		t.Fatalf("skip marked read=%v index=%d want false,0", cdefMap.Read[0], cdefMap.Index[0])
	}

	singleStrength := ctx.CDEF
	singleStrength.Bits = 0
	nonSkip := skip
	nonSkip.Prefix.SkipTransform = false
	if err := cdefMap.MarkBlock(singleStrength, nonSkip); err != nil {
		t.Fatal(err)
	}
	if !cdefMap.Read[0] || cdefMap.Index[0] != 0 {
		t.Fatalf("single strength marked read=%v index=%d want true,0", cdefMap.Read[0], cdefMap.Index[0])
	}
}

func TestFrameWorkCDEFIndexMapMarkBlockRejectsInvalidInputs(t *testing.T) {
	ctx := testFrameWorkCDEFIndexBatch(64, 64, 2)
	cdefMap, err := ctx.BindCDEFIndexMap(make([]uint8, 1), make([]bool, 1))
	if err != nil {
		t.Fatal(err)
	}
	valid := tile.BlockLoopVisit{
		Block:  tile.BlockVisit{MICol: 0, MIRow: 0, MIColEnd: 16, MIRowEnd: 16},
		Prefix: tile.BlockModeResult{CDEFIndex: 3},
	}

	badIndex := valid
	badIndex.Prefix.CDEFIndex = 4
	if err := cdefMap.MarkBlock(ctx.CDEF, badIndex); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("bad index err=%v want %v", err, ErrInvalidBatch)
	}

	outside := valid
	outside.Block.MICol = 16
	outside.Block.MIColEnd = 32
	if err := cdefMap.MarkBlock(ctx.CDEF, outside); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("outside err=%v want %v", err, ErrInvalidBatch)
	}

	empty := valid
	empty.Block.MIColEnd = empty.Block.MICol
	if err := cdefMap.MarkBlock(ctx.CDEF, empty); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("empty block err=%v want %v", err, ErrInvalidBatch)
	}

	shortMap := FrameWorkCDEFIndexMap{Index: nil, Read: nil, Stride: 1, Rows: 1}
	if err := shortMap.MarkBlock(ctx.CDEF, valid); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("short map err=%v want %v", err, ErrInvalidBatch)
	}
	if err := shortMap.Reset(); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("short reset err=%v want %v", err, ErrInvalidBatch)
	}
}

func TestFrameWorkTileResidualControllerRecordsCDEFIndexMap(t *testing.T) {
	ctx := testFrameWorkCDEFIndexBatch(128, 64, 2)
	cdefMap, err := ctx.BindCDEFIndexMap(make([]uint8, 2), make([]bool, 2))
	if err != nil {
		t.Fatal(err)
	}
	var stats FrameWorkTileResidualStats
	controller := frameWorkTileResidualLoopController{
		batch: ctx,
		req:   FrameWorkTileResidualRequest{CDEFIndexMap: &cdefMap},
		stats: &stats,
	}

	first := tile.BlockLoopVisit{
		Block:  tile.BlockVisit{MICol: 0, MIRow: 0, MIColEnd: 16, MIRowEnd: 16},
		Prefix: tile.BlockModeResult{CDEFIndex: 2},
	}
	if err := controller.reconBBegin(&first); err != nil {
		t.Fatal(err)
	}
	if !cdefMap.Read[0] || cdefMap.Index[0] != 2 || stats.CoefficientBlocks != 1 {
		t.Fatalf("first map read=%v index=%d stats=%+v", cdefMap.Read[0], cdefMap.Index[0], stats)
	}

	skip := tile.BlockLoopVisit{
		Block:  tile.BlockVisit{MICol: 16, MIRow: 0, MIColEnd: 32, MIRowEnd: 16},
		Prefix: tile.BlockModeResult{SkipTransform: true, CDEFIndex: 3},
	}
	if err := controller.reconBBegin(&skip); err != nil {
		t.Fatal(err)
	}
	if cdefMap.Read[1] || cdefMap.Index[1] != 0 || stats.SkippedBlocks != 1 {
		t.Fatalf("skip map read=%v index=%d stats=%+v", cdefMap.Read[1], cdefMap.Index[1], stats)
	}

	second := skip
	second.Prefix.SkipTransform = false
	if err := controller.reconBBegin(&second); err != nil {
		t.Fatal(err)
	}
	if !cdefMap.Read[1] || cdefMap.Index[1] != 3 || stats.CoefficientBlocks != 2 {
		t.Fatalf("second map read=%v index=%d stats=%+v", cdefMap.Read[1], cdefMap.Index[1], stats)
	}
	if err := cdefMap.Reset(); err != nil {
		t.Fatal(err)
	}
	for i := range cdefMap.Index {
		if cdefMap.Read[i] || cdefMap.Index[i] != 0 {
			t.Fatalf("reset unit %d read=%v index=%d want false,0", i, cdefMap.Read[i], cdefMap.Index[i])
		}
	}
}

func TestFrameWorkCDEFIndexMapAllocs(t *testing.T) {
	ctx := testFrameWorkCDEFIndexBatch(128, 64, 2)
	index := make([]uint8, 2)
	read := make([]bool, 2)
	visit := tile.BlockLoopVisit{
		Block:  tile.BlockVisit{MICol: 16, MIRow: 0, MIColEnd: 32, MIRowEnd: 16},
		Prefix: tile.BlockModeResult{CDEFIndex: 3},
	}
	allocs := testing.AllocsPerRun(1000, func() {
		for i := range index {
			index[i] = 0
			read[i] = false
		}
		cdefMap, err := ctx.BindCDEFIndexMap(index, read)
		if err != nil {
			t.Fatal(err)
		}
		if err := cdefMap.MarkBlock(ctx.CDEF, visit); err != nil {
			t.Fatal(err)
		}
		if err := cdefMap.Reset(); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("FrameWorkCDEFIndexMap allocated: %f", allocs)
	}
}

func testFrameWorkCDEFIndexBatch(width uint32, height uint32, bits uint8) FrameWorkBatch {
	return FrameWorkBatch{
		FrameWorkFrameContext: FrameWorkFrameContext{
			Sequence: FrameWorkSequenceContextFromHeader(parser.SequenceHeader{
				ColorConfig: parser.ColorConfig{BitDepth: 8},
			}),
			FrameSize: parser.FrameSize{CodedWidth: width, Height: height},
			CDEF:      parser.CDEFParams{Bits: bits, StrengthCount: 1 << bits},
		},
	}
}

// Visit the storage structurally so the oracle covers fields independently of
// both handwritten reset walks, including any CDF field added in the future.
func visitResetTestCDFs(t *testing.T, v reflect.Value, name string, visit func(string, *entropy.CDF)) {
	t.Helper()
	if v.Type() == reflect.TypeFor[entropy.CDF]() {
		visit(name, v.Addr().Interface().(*entropy.CDF))
		return
	}
	switch v.Kind() {
	case reflect.Struct:
		for i := range v.NumField() {
			visitResetTestCDFs(t, v.Field(i), name+"."+v.Type().Field(i).Name, visit)
		}
	case reflect.Array:
		for i := range v.Len() {
			visitResetTestCDFs(t, v.Index(i), fmt.Sprintf("%s[%d]", name, i), visit)
		}
	default:
		t.Fatalf("unexpected storage field %s: %s", name, v.Type())
	}
}

func TestFrameWorkCDFResetTrustedAllFields(t *testing.T) {
	for _, q := range []uint8{0, 21, 61, 121} {
		t.Run(fmt.Sprint(q), func(t *testing.T) {
			var storage FrameWorkTileResidualCDFStorage
			if err := storage.InitDefault(q); err != nil {
				t.Fatal(err)
			}
			active, unused := 0, 0
			visitResetTestCDFs(t, reflect.ValueOf(&storage).Elem(), "storage", func(name string, cdf *entropy.CDF) {
				if cdf.Symbols() == 0 {
					unused++
					return
				}
				active++
				for i := 0; i <= active%40; i++ {
					if err := cdf.Update((active + i*7) % cdf.Symbols()); err != nil {
						t.Fatalf("%s: %v", name, err)
					}
				}
			})
			want, generic := storage, storage
			visitResetTestCDFs(t, reflect.ValueOf(&want).Elem(), "storage", func(_ string, cdf *entropy.CDF) {
				if values := cdf.Values(); len(values) != 0 {
					values[len(values)-1] = 0
				}
			})
			if err := generic.ResetCDFCounts(); err != nil {
				t.Fatal(err)
			}
			storage.resetCDFCountsTrusted()
			if storage != want || generic != want {
				t.Fatal("reset disagrees with independent counter-only oracle")
			}
			if active == 0 || unused == 0 {
				t.Fatalf("missing active/unused coverage: %d/%d", active, unused)
			}
			t.Logf("q=%d active=%d unused=%d", q, active, unused)
		})
	}
}

func ownedCDFResetTestBatch(t *testing.T, retained *FrameWorkTileResidualCDFStorage, valid *bool) (FrameWorkBatch, FrameWorkLoopFilterCompactState, FrameWorkTileResidualRequest) {
	t.Helper()
	masks := compactTestHandle(16, 16, lfmask.Layout{SSHor: 1, SSVer: 1})
	var compact FrameWorkLoopFilterCompactState
	if err := BindFrameWorkLoopFilterCompactState(masks, &compact, make([]byte, 8), parser.LoopFilterParams{}, parser.SegmentationParams{}, parser.DeltaParams{}, false); err != nil {
		t.Fatal(err)
	}
	b := FrameWorkBatch{
		Jobs:                          []tile.Job{{UpdatesFrameContext: true}},
		RetainedTileResidualCDFs:      retained,
		RetainedTileResidualCDFsValid: valid,
	}
	b.LoopFilterMasks = masks
	return b, compact, FrameWorkTileResidualRequest{}
}

func TestFrameWorkCDFResetOwnedRetention(t *testing.T) {
	var storage, retained FrameWorkTileResidualCDFStorage
	if err := storage.InitDefault(64); err != nil {
		t.Fatal(err)
	}
	if err := storage.DeltaQ.Update(2); err != nil {
		t.Fatal(err)
	}
	want := storage
	if err := want.ResetCDFCounts(); err != nil {
		t.Fatal(err)
	}
	valid := false
	b, compact, req := ownedCDFResetTestBatch(t, &retained, &valid)
	state := tile.DecodeState{RetainFrameContext: true}
	if err := RetainTileResidualCDFStorageOwned(&b, 0, &state, &storage, &req, &compact); err != nil {
		t.Fatal(err)
	}
	if !valid || retained != want || storage != want {
		t.Fatal("owned retention did not publish the reset probability state")
	}
}

func TestFrameWorkCDFResetFallbackValidation(t *testing.T) {
	for _, tc := range []struct {
		name  string
		alter func(*FrameWorkBatch, *FrameWorkTileResidualRequest, *FrameWorkLoopFilterCompactState)
	}{
		{"inactive capability", func(_ *FrameWorkBatch, _ *FrameWorkTileResidualRequest, c *FrameWorkLoopFilterCompactState) {
			c.active = false
		}},
		{"detached masks", func(b *FrameWorkBatch, _ *FrameWorkTileResidualRequest, _ *FrameWorkLoopFilterCompactState) {
			b.LoopFilterMasks = nil
		}},
		{"wavefront", func(b *FrameWorkBatch, _ *FrameWorkTileResidualRequest, _ *FrameWorkLoopFilterCompactState) {
			b.WavefrontWorkers = 2
		}},
		{"after block", func(_ *FrameWorkBatch, r *FrameWorkTileResidualRequest, _ *FrameWorkLoopFilterCompactState) {
			r.AfterBlock = func(tile.BlockLoopVisit) error { return nil }
		}},
		{"before superblock", func(_ *FrameWorkBatch, r *FrameWorkTileResidualRequest, _ *FrameWorkLoopFilterCompactState) {
			r.Loop.BeforeSuperblock = func(tile.BlockLoopSuperblockVisit) error { return nil }
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var storage, retained FrameWorkTileResidualCDFStorage
			if err := storage.InitDefault(64); err != nil {
				t.Fatal(err)
			}
			if err := storage.Mode.Skip[0].Update(1); err != nil {
				t.Fatal(err)
			}
			// Mutate a late CDF through the supported writable API. The earlier
			// count stores and failed publication must match generic retention.
			storage.DeltaQ.Values()[0] = 0
			want := storage
			if err := want.ResetCDFCounts(); !errors.Is(err, entropy.ErrInvalidCDF) {
				t.Fatalf("oracle: %v", err)
			}
			valid := false
			b, compact, req := ownedCDFResetTestBatch(t, &retained, &valid)
			tc.alter(&b, &req, &compact)
			state := tile.DecodeState{RetainFrameContext: true}
			if err := RetainTileResidualCDFStorageOwned(&b, 0, &state, &storage, &req, &compact); !errors.Is(err, entropy.ErrInvalidCDF) {
				t.Fatalf("fallback error: %v", err)
			}
			if storage != want || valid || retained != (FrameWorkTileResidualCDFStorage{}) {
				t.Fatal("fallback changed validation/mutation/publication ordering")
			}
		})
	}
}

func TestFrameWorkCDFResetGenericRetentionValidation(t *testing.T) {
	var storage, retained FrameWorkTileResidualCDFStorage
	if err := storage.InitDefault(64); err != nil {
		t.Fatal(err)
	}
	storage.DeltaQ.Values()[0] = 0
	valid := false
	b, compact, req := ownedCDFResetTestBatch(t, &retained, &valid)
	state := tile.DecodeState{RetainFrameContext: true}
	for _, ownedWithoutCapability := range []bool{false, true} {
		var err error
		if ownedWithoutCapability {
			err = RetainTileResidualCDFStorageOwned(&b, 0, &state, &storage, &req, nil)
		} else {
			err = b.RetainTileResidualCDFStorage(0, &state, &storage)
		}
		if !errors.Is(err, entropy.ErrInvalidCDF) || valid {
			t.Fatalf("generic writable CDF accepted: err=%v retained=%v", err, valid)
		}
	}
	if err := RetainTileResidualCDFStorageOwned(&b, 0, &state, &storage, nil, &compact); !errors.Is(err, entropy.ErrInvalidCDF) {
		t.Fatalf("nil request fallback: %v", err)
	}
	state.RetainFrameContext = false
	before := storage
	if err := RetainTileResidualCDFStorageOwned(&b, 0, &state, &storage, &req, &compact); err != nil || storage != before || valid {
		t.Fatal("disabled retention mutated state")
	}
	state.RetainFrameContext = true
	b.Jobs[0].UpdatesFrameContext = false
	if err := RetainTileResidualCDFStorageOwned(&b, 0, &state, &storage, &req, &compact); err != nil || storage != before || valid {
		t.Fatal("non-update job mutated state")
	}
}

// TestCopyCDFStorageCoversEveryField fills every byte of a source storage
// and checks that copyCDFStorage reproduces it, so a field added to the
// storage (or to one of its CDF tables) without a copy fails here.
func TestCopyCDFStorageCoversEveryField(t *testing.T) {
	src := new(FrameWorkTileResidualCDFStorage)
	// entropy.CDF keeps its fields unexported, so each CDF is filled through
	// its API: all MaxSymbols+1 lanes (the whole values array) get a value
	// unique to that CDF.
	cdfType := reflect.TypeFor[entropy.CDF]()
	lane := uint16(0)
	fillCDF := func(v reflect.Value) bool {
		if v.Type() != cdfType {
			return false
		}
		c := v.Addr().Interface().(*entropy.CDF)
		if err := c.InitUniform(entropy.MaxSymbols); err != nil {
			t.Fatal(err)
		}
		for i := range c.Values() {
			lane++
			c.Values()[i] = lane
		}
		return true
	}
	n := 0
	if ro := poison.LeavesWith(src, func(i int) byte { return byte(i*7 + 1) }, func() bool { n++; return n&1 == 1 }, fillCDF); len(ro) != 0 {
		t.Fatalf("storage leaves the poison cannot set: %q", ro)
	}
	dst := new(FrameWorkTileResidualCDFStorage)
	copyCDFStorage(dst, src)
	if *dst != *src {
		t.Fatal("copyCDFStorage left a field uncopied")
	}
}

// TestCDFStorageInitDefaultMatchesBuild checks the cached defaults against a
// fresh build for every coefficient q-context.
func TestCDFStorageInitDefaultMatchesBuild(t *testing.T) {
	for _, q := range []uint8{0, 20, 21, 60, 61, 120, 121, 255} {
		var want, got FrameWorkTileResidualCDFStorage
		if err := want.buildDefault(q); err != nil {
			t.Fatal(err)
		}
		other := uint8(255)
		if q > 120 {
			other = 0
		}
		if err := got.InitDefault(other); err != nil { // start from another context
			t.Fatal(err)
		}
		if err := got.InitDefault(q); err != nil {
			t.Fatal(err)
		}
		if got != want {
			t.Fatalf("baseQIndex %d: cached defaults differ from a fresh build", q)
		}
	}
}

func TestFrameWorkBlockDequantIsKeyed(t *testing.T) {
	batch, state, _, _, req := testFrameWorkResidualDriver(t)
	batch.Quantization.UDCDelta = 3
	batch.Quantization.UACDelta = -5
	batch.Quantization.VDCDelta = -7
	batch.Quantization.VACDelta = 11
	c := frameWorkTileResidualLoopController{batch: batch, state: state, req: req}
	check := func(segmentID uint8) {
		t.Helper()
		dq := c.blockDequant(segmentID)
		if dq == nil {
			t.Fatalf("segment %d: no dequant factors", segmentID)
		}
		_, dqMax := quantize.DequantCoefficientBoundsTrusted(c.batch.Sequence.ColorConfig.BitDepth)
		for plane, p := range []FrameWorkPlane{FrameWorkPlaneY, FrameWorkPlaneU, FrameWorkPlaneV} {
			want, lossless, err := c.batch.BlockQuantizer(state.CurrentBaseQIdx, segmentID, p)
			if err != nil || lossless {
				t.Fatalf("plane %d oracle quantizer q=%+v lossless=%v err=%v", plane, want, lossless, err)
			}
			if got := dq[plane]; got.DC != uint32(want.DC) || got.AC != uint32(want.AC) || got.Max != uint32(dqMax) {
				t.Fatalf("segment %d plane %d factors=%+v want DC/AC %d/%d max %d", segmentID, plane, got, want.DC, want.AC, dqMax)
			}
		}
	}
	check(0)
	state.CurrentBaseQIdx++
	check(0)
	c.batch.Segmentation.Enabled = true
	c.batch.Segmentation.Data.Segments[1].DeltaQ = -11
	check(1)
}

func TestFrameWorkBlockDequantLossless(t *testing.T) {
	batch, state, _, _, req := testFrameWorkResidualDriver(t)
	batch.Quantization = parser.QuantizationParams{UsingQMatrix: true, QMatrixLevelY: 3, QMatrixLevelU: 3, QMatrixLevelV: 3}
	state.CurrentBaseQIdx = 0
	c := frameWorkTileResidualLoopController{batch: batch, state: state, req: req}
	dq := c.blockDequant(0)
	if dq == nil {
		t.Fatal("lossless block: no dequant factors")
	}
	for plane, p := range dq {
		if p.DC != 4 || p.AC != 4 || p.QM != nil {
			t.Fatalf("lossless plane %d factors %+v, want DC/AC 4/4 without matrices", plane, p)
		}
	}
}

func TestFrameWorkBlockDequantSelectsQMatrices(t *testing.T) {
	batch, state, _, _, req := testFrameWorkResidualDriver(t)
	batch.Quantization.UsingQMatrix = true
	batch.Quantization.QMatrixLevelY = 3
	batch.Quantization.QMatrixLevelU = quantize.QMLevels - 1
	batch.Quantization.QMatrixLevelV = 7
	c := frameWorkTileResidualLoopController{batch: batch, state: state, req: req}
	dq := c.blockDequant(0)
	if dq == nil {
		t.Fatal("quantizer-matrix block: no dequant factors")
	}
	want := [3]*tile.CoefQMatrices{tile.QMatrices(3, false), nil, tile.QMatrices(7, true)}
	for plane := range dq {
		if dq[plane].QM != want[plane] {
			t.Fatalf("plane %d matrices %p, want %p", plane, dq[plane].QM, want[plane])
		}
	}
	if want[0] == nil || want[2] == nil || want[0] == want[2] {
		t.Fatalf("matrix selection %p %p", want[0], want[2])
	}
}

func refRowsTestVisit(miRow uint16, ref tile.ReferenceFrame, mvRow int16) tile.BlockLoopVisit {
	var v tile.BlockLoopVisit
	v.Block.MIRow = miRow
	v.Block.Size = tile.BlockSize16x16
	v.Prediction.Valid = true
	v.Prediction.InterMotion.References.Ref0, v.Prediction.InterMotion.References.Ref1 = ref, tile.ReferenceFrameNone
	v.Prediction.InterMotion.MV[0] = motion.Vector{Row: mvRow}
	return v
}

// TestFrameThreadRefNeed pins the per-reference row bound of each prediction
// kind: bottom edge + whole-row vertical motion + the filter / chroma margin.
func TestFrameThreadRefNeed(t *testing.T) {
	const bottom = (8 + 4) * 4 // MIRow 8, 16x16

	var n frameThreadRefNeed
	v := refRowsTestVisit(8, tile.ReferenceFrameLast, 8*10+3) // 10 rows + a fraction
	n.addVisit(&v, &testPayloads)
	if want := int32(bottom + 10 + refRowMargin); n[tile.ReferenceFrameLast] != want {
		t.Fatalf("translation: need %d, want %d", n[tile.ReferenceFrameLast], want)
	}

	// Upward motion never needs fewer than one row.
	n = frameThreadRefNeed{}
	v = refRowsTestVisit(0, tile.ReferenceFrameLast, -8*500)
	n.addVisit(&v, &testPayloads)
	if n[tile.ReferenceFrameLast] != 1 {
		t.Fatalf("upward: need %d, want 1", n[tile.ReferenceFrameLast])
	}

	// Compound: both references, each with its own vector.
	n = frameThreadRefNeed{}
	v = refRowsTestVisit(8, tile.ReferenceFrameLast, 8*4)
	v.Prediction.InterMotion.References.Ref1 = tile.ReferenceFrameAltref
	v.Prediction.InterMotion.References.Compound = true
	v.Prediction.InterMotion.MV[1] = motion.Vector{Row: 8 * 40}
	n.addVisit(&v, &testPayloads)
	if n[tile.ReferenceFrameLast] != bottom+4+refRowMargin || n[tile.ReferenceFrameAltref] != bottom+40+refRowMargin {
		t.Fatalf("compound: need %v", n)
	}

	// OBMC: a neighbour's vector applies over the block's own rows.
	n = frameThreadRefNeed{}
	v = refRowsTestVisit(8, tile.ReferenceFrameLast, 0)
	v.Prediction.MotionModeValid = true
	v.Prediction.MotionMode = tile.MotionModeOBMC
	v.Prediction.OverlappableNeighborsValid = true
	testMotionExtras(&v.Prediction).OverlappableNeighbors.AboveCount = 1
	testMotionExtras(&v.Prediction).OverlappableNeighbors.Above[0].MotionValid = true
	testMotionExtras(&v.Prediction).OverlappableNeighbors.Above[0].Motion.References = tile.InterReferencesResult{Ref0: tile.ReferenceFrameGolden, Ref1: tile.ReferenceFrameNone}
	testMotionExtras(&v.Prediction).OverlappableNeighbors.Above[0].Motion.MV[0] = motion.Vector{Row: 8 * 64}
	n.addVisit(&v, &testPayloads)
	if n[tile.ReferenceFrameGolden] != bottom+64+refRowMargin {
		t.Fatalf("obmc: golden need %d", n[tile.ReferenceFrameGolden])
	}

	// Sub-8x8 chroma cells use their own (neighbouring) vectors.
	n = frameThreadRefNeed{}
	v = refRowsTestVisit(8, tile.ReferenceFrameLast, 0)
	sub := testSubChroma(&v.Prediction)
	sub.Count = 1
	sub.Cells[0].Reference = tile.ReferenceFrameLast2
	sub.Cells[0].MV = motion.Vector{Row: 8 * 30}
	n.addVisit(&v, &testPayloads)
	if n[tile.ReferenceFrameLast2] != bottom+30+refRowMargin {
		t.Fatalf("sub-chroma: last2 need %d", n[tile.ReferenceFrameLast2])
	}

	// Warped and global-warp predictions wait for the whole reference.
	for _, mod := range []func(*tile.BlockLoopVisit){
		func(v *tile.BlockLoopVisit) {
			v.Prediction.MotionModeValid = true
			v.Prediction.MotionMode = tile.MotionModeWarp
		},
		func(v *tile.BlockLoopVisit) { v.Prediction.GlobalWarpedMotionValid = true },
	} {
		n = frameThreadRefNeed{}
		v = refRowsTestVisit(8, tile.ReferenceFrameLast, 0)
		mod(&v)
		n.addVisit(&v, &testPayloads)
		if n[tile.ReferenceFrameLast] != FrameProgressDone {
			t.Fatalf("warp: need %d, want the whole reference", n[tile.ReferenceFrameLast])
		}
	}

	// Intra and intra block copy read no reference.
	n = frameThreadRefNeed{}
	v = refRowsTestVisit(8, tile.ReferenceFrameLast, 0)
	v.Prediction.Intra = true
	n.addVisit(&v, &testPayloads)
	if n != (frameThreadRefNeed{}) {
		t.Fatalf("intra: need %v", n)
	}
}

// testPayloads backs the payload slots of test visits; the test prediction
// batches resolve through it. Tests in this package do not run in parallel.
var testPayloads = tile.BlockPayloads{
	Extras:    make([]tile.BlockPredictionMotionExtras, 1),
	SubChroma: make([]tile.SubChromaInterResult, 1),
	Palette:   make([]tile.PaletteModeScratch, 1),
}

// testMotionExtras returns p's out-of-line warp/OBMC payloads, naming a
// cleared test slot the first time.
func testMotionExtras(p *tile.BlockPredictionModeResult) *tile.BlockPredictionMotionExtras {
	if p.ExtrasSlot == 0 {
		testPayloads.Extras[0] = tile.BlockPredictionMotionExtras{}
		p.ExtrasSlot = 1
	}
	return &testPayloads.Extras[0]
}

// testSubChroma returns p's sub8x8 chroma cells, naming a cleared test slot
// the first time.
func testSubChroma(p *tile.BlockPredictionModeResult) *tile.SubChromaInterResult {
	if p.SubChromaSlot == 0 {
		testPayloads.SubChroma[0] = tile.SubChromaInterResult{}
		p.SubChromaSlot = 1
	}
	return &testPayloads.SubChroma[0]
}

// TestFrameThreadMotionRowsAggregatesTileGrid checks the multi-tile MV_REF
// progress against a brute-force reference: the published row must be the
// largest R such that every 8x8 row below R is saved in every tile column, for
// tile grids with one or several tile rows/columns and non-uniform spacing.
func TestFrameThreadMotionRowsAggregatesTileGrid(t *testing.T) {
	type grid struct {
		name     string
		rowStart []int32 // tile row starts in 8x8 rows, plus the frame end
		cols     int
	}
	grids := []grid{
		{"1x4", []int32{0, 34}, 4},
		{"4x1", []int32{0, 9, 18, 27, 34}, 1},
		{"4x2", []int32{0, 18, 34}, 4},
		{"nonuniform-3x3", []int32{0, 4, 24, 34}, 3},
		{"nonuniform-2x2", []int32{0, 30, 34}, 2},
	}
	rng := rand.New(rand.NewSource(1))
	for _, g := range grids {
		t.Run(g.name, func(t *testing.T) {
			var tiles []FrameThreadMotionTile
			for r := 0; r+1 < len(g.rowStart); r++ {
				for c := 0; c < g.cols; c++ {
					tiles = append(tiles, FrameThreadMotionTile{rows: g.rowStart[r], end: g.rowStart[r+1], used: true})
				}
			}
			frameEnd := g.rowStart[len(g.rowStart)-1]
			for step := 0; step < 2000; step++ {
				got := frameThreadMotionRows(tiles)
				// Reference: row y is saved when the tile covering it in
				// every column has rows > y.
				want := frameEnd
				for y := int32(0); y < frameEnd && want == frameEnd; y++ {
					for i := range tiles {
						if y >= tiles[i].rows && y < tiles[i].end && y >= startOf(tiles[i], g.rowStart) {
							want = y
							break
						}
					}
				}
				if want == frameEnd {
					want = FrameProgressDone
				}
				if got != want {
					t.Fatalf("step %d: got %d want %d tiles %+v", step, got, want, tiles)
				}
				if got == FrameProgressDone {
					break
				}
				// Advance a random unfinished tile by one SB row (2 8x8 rows).
				var open []int
				for i := range tiles {
					if tiles[i].rows < tiles[i].end {
						open = append(open, i)
					}
				}
				i := open[rng.Intn(len(open))]
				tiles[i].rows = min(tiles[i].rows+2, tiles[i].end+1)
			}
		})
	}
}

func startOf(tile FrameThreadMotionTile, rowStart []int32) int32 {
	for i := 0; i+1 < len(rowStart); i++ {
		if rowStart[i+1] == tile.end {
			return rowStart[i]
		}
	}
	return 0
}

const hotStructKiB = 1024

func TestHotStructSizes(t *testing.T) {
	tests := []struct {
		name string
		size uintptr
		max  uintptr
	}{
		{name: "Batch", size: unsafe.Sizeof(Batch{}), max: 20},
		// The wavefront lane channel (8 bytes) lets tiles reuse the pool's
		// persistent lane goroutines instead of spawning them per frame.
		// DeferredTemporalReferences (8 bytes) hands the tile loops the
		// references they project per superblock row (dav1d load_tmvs).
		{name: "FrameWorkBatch", size: unsafe.Sizeof(FrameWorkBatch{}), max: 1648}, // +8: FrameThread (dav1d frame-thread split); +8: SegmentMaps (dav1d cur/prev_segmap); +8: visit payload slots
		{name: "FrameWorkFrameContext", size: unsafe.Sizeof(FrameWorkFrameContext{}), max: 1152},
		{name: "FrameWorkPlaneRegion", size: unsafe.Sizeof(FrameWorkPlaneRegion{}), max: 72},
		{name: "FrameWorkTileResidualCDFStorage", size: unsafe.Sizeof(FrameWorkTileResidualCDFStorage{}), max: 57 * hotStructKiB}, // +508: DV, the intra block copy MV context
		// The fused coefficient path adds one compact job-owned dequant target
		// to the controller; its independent 80-byte layout is guarded in tile.
		// It embeds FrameWorkBatch copies, so the 8-byte wavefront lane channel
		// appears here too.
		// The tile refmvs state carries dav1d's mfmv setup and rp_proj window.
		// The three restoration stripe-boundary handles each add a pixel-row
		// (Bytes) flag, 8 bytes with alignment.
		{name: "FrameWorkTileResidualScratch", size: unsafe.Sizeof(FrameWorkTileResidualScratch{}), max: 99408}, // net smaller: BlockLoopScratch gained the recon_b binding and txtp_map, the controller lost the dequant target, coeffReq and the transform-type selectors; +64: the controller's coeffReq slot (the block coefficient request is read in place, not copied per block); +16: two FrameWorkBatch copies gained FrameThread; +16: BlockLoopRequest.MotionProgress; +16: BlockLoopScratch.leftEdge (resident SB-row left context); +16: two FrameWorkBatch copies gained SegmentMaps; +2032: two job geometry caches gained the f->refp reference-plane views; +1072: and the bound recon_b_inter plane state
		{name: "FrameWorkJobRegion", size: unsafe.Sizeof(FrameWorkJobRegion{}), max: 36},
		{name: "FrameWorkCDEFIndexMap", size: unsafe.Sizeof(FrameWorkCDEFIndexMap{}), max: 56},
		{name: "FrameWorkLoopFilterBlockRecord", size: unsafe.Sizeof(FrameWorkLoopFilterBlockRecord{}), max: 34},
		{name: "FrameWorkLoopFilterMapStats", size: unsafe.Sizeof(FrameWorkLoopFilterMapStats{}), max: 12},
		{name: "FrameWorkTileResidualStats", size: unsafe.Sizeof(FrameWorkTileResidualStats{}), max: 136},
		// poolTask carries one RangeRunner interface plus the low/high range for
		// row-banded postfilter work. The band ordinal reuses Batch.Worker, so the
		// typed runner adds no space beyond the former function-plus-range fields.
		// It also embeds a FrameWorkBatch, hence the wavefront lane channel and
		// the deferred temporal-reference pointer.
		{name: "poolTask", size: unsafe.Sizeof(poolTask{}), max: 1760}, // +8: FrameWorkBatch.FrameThread; +8: FrameWorkBatch.SegmentMaps; +8: FrameWorkBatch.payloads
		{name: "frameWorkReconCBI", size: unsafe.Sizeof(frameWorkReconCBI{}), max: 6},
		{name: "frameWorkReconSB", size: unsafe.Sizeof(frameWorkReconSB{}), max: 20}, // +8: the superblock's cbi and cf cursors
		// The wavefront is allocated once per scratch behind a pointer; it carries
		// the parse-side span cursor and the published row snapshot the lanes
		// read while the tile is still parsing.
		{name: "frameWorkReconWavefront", size: unsafe.Sizeof(frameWorkReconWavefront{}), max: 552}, // +8: frameThread, +8: reconTile; +24: the published view snapshots the cf arena; +72: the published view carries the payload arenas, -8: no palette install cursor
		{name: "frameWorkBlockCoeffGeometry", size: unsafe.Sizeof(frameWorkBlockCoeffGeometry{}), max: 88},
		// The geometry cache also binds each job's inter-prediction output planes
		// once (dav1d binds f->cur per frame); it lives in per-worker scratch.
		{name: "frameWorkJobGeometryCache", size: unsafe.Sizeof(frameWorkJobGeometryCache{}), max: 2224}, // +1016: dav1d f->refp views, 7 references x 3 planes; +536: the bound recon_b_inter planes and same-size reference views; +8: the frame filter pair
		{name: "frameWorkPredictionPlaneGeometry", size: unsafe.Sizeof(frameWorkPredictionPlaneGeometry{}), max: 152},
		{name: "frameWorkWedgeCode", size: unsafe.Sizeof(frameWorkWedgeCode{}), max: 3},
	}
	for _, tc := range tests {
		t.Logf("%s size=%d max=%d", tc.name, tc.size, tc.max)
		if tc.size > tc.max {
			t.Fatalf("%s grew to %d bytes, max %d", tc.name, tc.size, tc.max)
		}
	}
}

func TestFrameWorkOwnedIntraContextsSelection(t *testing.T) {
	var scratch FrameWorkTileResidualScratch
	batch := FrameWorkBatch{}
	batch.FrameHeader.FrameType = parser.FrameTypeKey
	request := FrameWorkTileResidualRequest{}
	loop := tile.BlockLoopRequest{FrameType: parser.FrameTypeKey, DecodePredictionModes: true, ContextCarrier: &scratch.LoopContext}
	for _, ft := range []parser.FrameType{parser.FrameTypeKey, parser.FrameTypeIntraOnly} {
		b, q := batch, loop
		b.FrameHeader.FrameType = ft
		q.FrameType = ft
		if !frameWorkOwnedIntraContexts(&b, &scratch, request, q) {
			t.Fatal("eligible private intra route rejected", ft)
		}
	}
	for _, tc := range []struct {
		name  string
		alter func(*FrameWorkBatch, *FrameWorkTileResidualRequest, *tile.BlockLoopRequest)
	}{
		{"inter", func(b *FrameWorkBatch, _ *FrameWorkTileResidualRequest, l *tile.BlockLoopRequest) {
			b.FrameHeader.FrameType = parser.FrameTypeInter
			l.FrameType = parser.FrameTypeInter
		}},
		{"switch", func(b *FrameWorkBatch, _ *FrameWorkTileResidualRequest, l *tile.BlockLoopRequest) {
			b.FrameHeader.FrameType = parser.FrameTypeSwitch
			l.FrameType = parser.FrameTypeSwitch
		}},
		{"frame mismatch", func(_ *FrameWorkBatch, _ *FrameWorkTileResidualRequest, l *tile.BlockLoopRequest) {
			l.FrameType = parser.FrameTypeIntraOnly
		}},
		{"frame IBC", func(b *FrameWorkBatch, _ *FrameWorkTileResidualRequest, _ *tile.BlockLoopRequest) {
			b.FrameSize.AllowIntrabc = true
		}},
		{"request IBC", func(_ *FrameWorkBatch, _ *FrameWorkTileResidualRequest, l *tile.BlockLoopRequest) {
			l.AllowIntrabc = true
		}},
		{"skip-mode override", func(_ *FrameWorkBatch, _ *FrameWorkTileResidualRequest, l *tile.BlockLoopRequest) {
			l.SkipMode.Enabled = true
		}},
		{"no prediction", func(_ *FrameWorkBatch, _ *FrameWorkTileResidualRequest, l *tile.BlockLoopRequest) {
			l.DecodePredictionModes = false
		}},
		{"copied carrier", func(_ *FrameWorkBatch, _ *FrameWorkTileResidualRequest, l *tile.BlockLoopRequest) {
			copied := scratch.LoopContext
			l.ContextCarrier = &copied
		}},
		{"caller neighborhood", func(_ *FrameWorkBatch, _ *FrameWorkTileResidualRequest, l *tile.BlockLoopRequest) {
			l.Walk.UseNeighborBounds = true
		}},
		{"wavefront", func(b *FrameWorkBatch, _ *FrameWorkTileResidualRequest, _ *tile.BlockLoopRequest) {
			b.WavefrontWorkers = 2
		}},
		{"block hook", func(_ *FrameWorkBatch, r *FrameWorkTileResidualRequest, _ *tile.BlockLoopRequest) {
			r.AfterBlock = func(tile.BlockLoopVisit) error { return nil }
		}},
		{"SB hook", func(_ *FrameWorkBatch, _ *FrameWorkTileResidualRequest, l *tile.BlockLoopRequest) {
			l.BeforeSuperblock = func(tile.BlockLoopSuperblockVisit) error { return nil }
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			b, r, l := batch, request, loop
			tc.alter(&b, &r, &l)
			if frameWorkOwnedIntraContexts(&b, &scratch, r, l) {
				t.Fatal("fallback incorrectly selected private contexts")
			}
		})
	}
}

func compactTestHandle(cols, rows int, layout lfmask.Layout) *FrameWorkLoopFilterMasks {
	sb128w, sb128h, count := FrameWorkLoopFilterMaskShape(cols, rows)
	return &FrameWorkLoopFilterMasks{
		Masks:      make([]lfmask.FilterMask, count),
		LevelCache: make([]uint8, 4*(cols*rows)),
		Cols:       cols,
		Rows:       rows,
		SB128W:     sb128w,
		SB128H:     sb128h,
		Layout:     layout,
		HasChroma:  !layout.Mono,
	}
}

func compactTestTransformForSize(size tile.BlockSize) tile.TransformSize {
	switch size {
	case tile.BlockSize128x128, tile.BlockSize128x64, tile.BlockSize64x128, tile.BlockSize64x64:
		return tile.TransformSize64x64
	case tile.BlockSize64x32, tile.BlockSize32x64, tile.BlockSize32x32:
		return tile.TransformSize32x32
	case tile.BlockSize64x16, tile.BlockSize16x64, tile.BlockSize32x16, tile.BlockSize16x32, tile.BlockSize16x16:
		return tile.TransformSize16x16
	case tile.BlockSize32x8, tile.BlockSize8x32, tile.BlockSize16x8, tile.BlockSize8x16:
		return tile.TransformSize8x8
	case tile.BlockSize16x4:
		return tile.TransformSize16x4
	case tile.BlockSize4x16:
		return tile.TransformSize4x16
	case tile.BlockSize8x4:
		return tile.TransformSize8x4
	case tile.BlockSize4x8:
		return tile.TransformSize4x8
	case tile.BlockSize8x8:
		return tile.TransformSize8x8
	default:
		return tile.TransformSize4x4
	}
}

func compactTestLegalBlocks(cols, rows, sbLog2 int) []orderTestBlock {
	var blocks []orderTestBlock
	appendBlock := func(x, y, w, h int, size tile.BlockSize, pattern int) {
		if x >= cols || y >= rows {
			return
		}
		if x+w > cols {
			w = cols - x
		}
		if y+h > rows {
			h = rows - y
		}
		ytx := compactTestTransformForSize(size)
		// Keep some transform variation within legal rectangles, including the
		// 8x4/4x8 shared-chroma siblings below.
		if pattern&1 != 0 && (size == tile.BlockSize8x4 || size == tile.BlockSize4x8) {
			ytx = tile.TransformSize4x4
		}
		blocks = append(blocks, orderTestBlock{
			bx: x, by: y, bw4: w, bh4: h, size: size,
			tree:  tile.TransformTreeResult{Y: ytx, UV: tile.TransformSize4x4, HasUV: true},
			intra: pattern%3 == 0,
		})
	}
	var walk func(x, y, size int)
	walk = func(x, y, size int) {
		if x >= cols || y >= rows {
			return
		}
		if size == 2 {
			choice := (x/2 + 3*(y/2)) % 4
			switch choice {
			case 0: // PartitionNone.
				appendBlock(x, y, 2, 2, tile.BlockSize8x8, choice)
			case 1: // PartitionH, with a smaller transform in the bottom sibling.
				appendBlock(x, y, 2, 1, tile.BlockSize8x4, 0)
				appendBlock(x, y+1, 2, 1, tile.BlockSize8x4, 1)
			case 2: // PartitionV, with a smaller transform in the right sibling.
				appendBlock(x, y, 1, 2, tile.BlockSize4x8, 0)
				appendBlock(x+1, y, 1, 2, tile.BlockSize4x8, 1)
			default: // PartitionSplit into four 4x4 leaves.
				appendBlock(x, y, 1, 1, tile.BlockSize4x4, 0)
				appendBlock(x+1, y, 1, 1, tile.BlockSize4x4, 1)
				appendBlock(x, y+1, 1, 1, tile.BlockSize4x4, 2)
				appendBlock(x+1, y+1, 1, 1, tile.BlockSize4x4, 3)
			}
			return
		}
		// At the 16x16 MI level, use legal H4/V4 partitions to include
		// 16x4 and 4x16 leaves; at larger levels, vary unsplit blocks and
		// recursive quad splits. Every recursive descent is TL, TR, BL, BR.
		if size == 4 {
			switch (x/4 + 3*(y/4)) % 6 {
			case 0: // PartitionH4.
				for dy := 0; dy < 4; dy++ {
					appendBlock(x, y+dy, 4, 1, tile.BlockSize16x4, dy)
				}
				return
			case 1: // PartitionV4.
				for dx := 0; dx < 4; dx++ {
					appendBlock(x+dx, y, 1, 4, tile.BlockSize4x16, dx)
				}
				return
			case 2:
				appendBlock(x, y, 4, 2, tile.BlockSize16x8, 0)
				appendBlock(x, y+2, 4, 2, tile.BlockSize16x8, 1)
				return
			case 3:
				appendBlock(x, y, 2, 4, tile.BlockSize8x16, 0)
				appendBlock(x+2, y, 2, 4, tile.BlockSize8x16, 1)
				return
			}
		}
		if size > 4 && (x+3*y+size)%5 == 0 {
			blockSize := tile.BlockSize16x16
			switch size {
			case 8:
				blockSize = tile.BlockSize32x32
			case 16:
				blockSize = tile.BlockSize64x64
			case 32:
				blockSize = tile.BlockSize128x128
			}
			appendBlock(x, y, size, size, blockSize, x+y)
			return
		}
		half := size / 2
		walk(x, y, half)
		walk(x+half, y, half)
		walk(x, y+half, half)
		walk(x+half, y+half, half)
	}
	sbMI := 1 << uint(sbLog2)
	for y := 0; y < rows; y += sbMI {
		for x := 0; x < cols; x += sbMI {
			walk(x, y, sbMI)
		}
	}
	return blocks
}

func compactTestVisit(b orderTestBlock, cols, rows, index int, skip bool) tile.BlockLoopVisit {
	endX, endY := b.bx+b.bw4, b.by+b.bh4
	if endX > cols {
		endX = cols
	}
	if endY > rows {
		endY = rows
	}
	visit := tile.BlockLoopVisit{
		Block: tile.BlockVisit{
			MICol: uint16(b.bx), MIRow: uint16(b.by),
			MIColEnd: uint16(endX), MIRowEnd: uint16(endY),
			X4: uint8(b.bx), Y4: uint8(b.by),
			Size: b.size, VisibleW4: uint8(endX - b.bx), VisibleH4: uint8(endY - b.by),
		},
		SegmentID:         uint8(index % parser.MaxSegments),
		Prefix:            tile.BlockModeResult{SkipTransform: skip},
		Coefficients:      tile.BlockCoeffResult{Tree: b.tree},
		CoefficientsValid: true,
	}
	if b.intra {
		visit.Prediction = tile.BlockPredictionModeResult{Valid: true, Intra: true}
	} else {
		mode := tile.InterModeNewMV
		if index%3 == 0 {
			mode = tile.InterModeGlobalMV
		}
		visit.Prediction = tile.BlockPredictionModeResult{
			Valid:                true,
			InterReferencesValid: true,
			InterReferences: tile.InterReferencesResult{
				Ref0: tile.ReferenceFrame(index % int(tile.ReferenceFrameAltref+1)), Ref1: tile.ReferenceFrameNone,
			},
			InterModeValid: true,
			InterMode:      tile.InterModeResult{Mode: mode},
		}
	}
	return visit
}

func compactTestDecodeState(index int) tile.DecodeState {
	return tile.DecodeState{
		DeltaLFFromBase: int8(index%11 - 5),
		DeltaLF: [tile.FrameLoopFilterCount]int8{
			int8(index%13 - 6), int8(index%9 - 4), int8(index%7 - 3), int8(index%5 - 2),
		},
	}
}

func compactTestExpectedNoSkip(records FrameWorkLoopFilterMap, cols, rows int) []byte {
	unitCols, unitRows := (cols+15)>>4, (rows+15)>>4
	got := make([]byte, unitCols*unitRows*8)
	stride := int(records.Stride)
	for miRow := 0; miRow < rows; miRow++ {
		for miCol := 0; miCol < cols; miCol++ {
			record := records.Records[miRow*stride+miCol]
			if !record.Valid || record.SkipTransform {
				continue
			}
			x, y := miCol>>1, miRow>>1
			off := ((y>>3)*unitCols+(x>>3))*8 + (y & 7)
			got[off] |= 1 << uint(x&7)
		}
	}
	validCols, validRows := cols>>1, rows>>1
	for y := 0; y < unitRows*8; y++ {
		for x := 0; x < unitCols*8; x++ {
			if x < validCols && y < validRows {
				continue
			}
			off := ((y>>3)*unitCols+(x>>3))*8 + (y & 7)
			got[off] |= 1 << uint(x&7)
		}
	}
	return got
}

func compactTestExpectedLevels(t *testing.T, records FrameWorkLoopFilterMap, cols, rows int, layout lfmask.Layout, hasChroma bool, params parser.LoopFilterParams, seg parser.SegmentationParams, delta parser.DeltaParams) []uint8 {
	t.Helper()
	levels := make([]uint8, 4*(cols*rows))
	for row := 0; row < rows; row++ {
		for col := 0; col < cols; {
			rec := records.Records[row*int(records.Stride)+col]
			if !rec.Valid {
				col++
				continue
			}
			nextCol := int(rec.Block.MIColEnd)
			if nextCol <= col || nextCol > cols {
				t.Fatalf("invalid next block column at (%d,%d): %d", col, row, nextCol)
			}
			if int(rec.Block.MICol) != col || int(rec.Block.MIRow) != row {
				col = nextCol
				continue
			}
			resolve := func(plane loopfilter.Plane, edge loopfilter.Edge) uint8 {
				blockDelta, err := loopfilter.SelectDelta(delta, rec.DeltaLFFromBase, rec.DeltaLF, plane, edge)
				if err != nil {
					t.Fatalf("select delta (%d,%d plane=%d edge=%d): %v", col, row, plane, edge, err)
				}
				level, err := loopfilter.ResolveLevel(params, seg, loopfilter.LevelRequest{
					Plane: plane, Edge: edge, SegmentID: rec.SegmentID,
					RefFrame: rec.RefFrame, Mode: rec.Mode, DeltaLF: blockDelta,
				})
				if err != nil {
					t.Fatalf("resolve level (%d,%d plane=%d edge=%d): %v", col, row, plane, edge, err)
				}
				return level
			}
			dims, ok := rec.Block.Size.Dimensions()
			if !ok {
				t.Fatalf("invalid block size at (%d,%d): %d", col, row, rec.Block.Size)
			}
			bx, by := col, row
			bw4 := min(cols-bx, int(dims.W4))
			bh4 := min(rows-by, int(dims.H4))
			yVert := resolve(loopfilter.PlaneY, loopfilter.EdgeVertical)
			yHorz := resolve(loopfilter.PlaneY, loopfilter.EdgeHorizontal)
			for y := 0; y < bh4; y++ {
				for x := 0; x < bw4; x++ {
					cell := (*[4]uint8)(levels[4*((by+y)*cols+bx+x):])
					cell[0], cell[1] = yVert, yHorz
				}
			}
			if hasChroma {
				cbCols := (cols + layout.SSHor) >> layout.SSHor
				cbRows := (rows + layout.SSVer) >> layout.SSVer
				cbw4 := min(cbCols-(bx>>layout.SSHor), (int(dims.W4)+layout.SSHor)>>layout.SSHor)
				cbh4 := min(cbRows-(by>>layout.SSVer), (int(dims.H4)+layout.SSVer)>>layout.SSVer)
				uLevel := resolve(loopfilter.PlaneU, loopfilter.EdgeVertical)
				vLevel := resolve(loopfilter.PlaneV, loopfilter.EdgeVertical)
				cbx, cby := bx>>layout.SSHor, by>>layout.SSVer
				for y := 0; y < cbh4; y++ {
					for x := 0; x < cbw4; x++ {
						cell := (*[4]uint8)(levels[4*((cby+y)*cols+cbx+x):])
						cell[2], cell[3] = uLevel, vLevel
					}
				}
			}
			col = nextCol
		}
	}
	return levels
}

func TestCompactLoopFilterMatchesMapRasterAndCDEFNoSkip(t *testing.T) {
	params := parser.LoopFilterParams{
		LevelY: [2]uint8{31, 42}, LevelU: 29, LevelV: 37,
		ModeRefDeltaEnabled: true,
		Deltas: parser.LoopFilterDeltas{
			Ref:  [parser.RefFrames]int8{-7, 3, 5, -2, 4, -3, 6, 1},
			Mode: [2]int8{-2, 5},
		},
	}
	seg := parser.SegmentationParams{Enabled: true}
	for i := range seg.Data.Segments {
		seg.Data.Segments[i] = parser.SegmentData{
			DeltaLFYV: int16(i*17 - 50), DeltaLFYH: int16(45 - i*13),
			DeltaLFU: int16(i*9 - 25), DeltaLFV: int16(30 - i*7),
		}
	}
	delta := parser.DeltaParams{DeltaLFPresent: true, DeltaLFMulti: true}

	cases := []struct {
		name   string
		cols   int
		rows   int
		sbLog2 int
		layout lfmask.Layout
		mono   bool
		blocks func(int, int, int) []orderTestBlock
	}{
		{name: "64sb-420-partial-crop", cols: 30, rows: 30, sbLog2: 4, layout: lfmask.Layout{SSHor: 1, SSVer: 1}, blocks: compactTestLegalBlocks},
		{name: "128sb-422", cols: 32, rows: 32, sbLog2: 5, layout: lfmask.Layout{SSHor: 1}, blocks: compactTestLegalBlocks},
		{name: "128sb-444", cols: 32, rows: 30, sbLog2: 5, layout: lfmask.Layout{}, blocks: compactTestLegalBlocks},
		{name: "shared-chroma-420-sub8-partitions", cols: 16, rows: 16, sbLog2: 4, layout: lfmask.Layout{SSHor: 1, SSVer: 1}, blocks: compactTestLegalBlocks},
		{name: "mono-sub8-partitions", cols: 16, rows: 16, sbLog2: 4, layout: lfmask.Layout{Mono: true}, mono: true, blocks: compactTestLegalBlocks},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			blocks := tc.blocks(tc.cols, tc.rows, tc.sbLog2)
			compactMasks := compactTestHandle(tc.cols, tc.rows, tc.layout)
			state := &FrameWorkLoopFilterCompactState{}
			unitCols, unitRows := (tc.cols+15)>>4, (tc.rows+15)>>4
			noSkip := make([]byte, unitCols*unitRows*8)
			if err := BindFrameWorkLoopFilterCompactState(compactMasks, state, noSkip, params, seg, delta, tc.mono); err != nil {
				t.Fatalf("bind compact state: %v", err)
			}
			if !FrameWorkLoopFilterCompactActiveShape(compactMasks, state, tc.cols, tc.rows) ||
				FrameWorkLoopFilterCompactReady(compactMasks, state, tc.cols, tc.rows) {
				t.Fatal("new compact binding must be active with exact shape but not ready")
			}

			mapHandle := FrameWorkLoopFilterMap{
				Records: make([]FrameWorkLoopFilterBlockRecord, tc.cols*tc.rows),
				Stride:  uint16(tc.cols), Rows: uint16(tc.rows),
			}
			var compactScratch frameWorkLoopFilterMaskScratch
			compactScratch.reset(compactMasks)
			for i, block := range blocks {
				skip := i%5 == 0
				if block.size != tile.BlockSize4x4 && i%3 == 1 {
					block.intra = false
				}
				if tc.mono {
					block.tree.HasUV = false
				}
				visit := compactTestVisit(block, tc.cols, tc.rows, i, skip)
				decodeState := compactTestDecodeState(i)
				if err := mapHandle.MarkBlockPtr(&visit, &decodeState); err != nil {
					t.Fatalf("map mark block %d: %v", i, err)
				}
				if err := compactMasks.buildCompact(state, &compactScratch, &visit, &decodeState, tc.sbLog2); err != nil {
					t.Fatalf("compact block %d: %v", i, err)
				}
			}

			mapStats, err := mapHandle.CoverageStats(tc.cols, tc.rows)
			if err != nil || mapStats.Cells != int32(tc.cols*tc.rows) || mapStats.Blocks != int32(len(blocks)) || mapStats.Missing != 0 {
				t.Fatalf("map coverage=%+v err=%v blocks=%d", mapStats, err, len(blocks))
			}
			refMasks := compactTestHandle(tc.cols, tc.rows, tc.layout)
			var rasterScratch FrameWorkLoopFilterMaskBuildScratch
			if err := refMasks.BuildFromMap(&rasterScratch, mapHandle, tc.sbLog2); err != nil {
				t.Fatalf("raster mask build: %v", err)
			}
			if !reflect.DeepEqual(compactMasks.MaskBits(), refMasks.MaskBits()) {
				t.Fatal("compact edge masks differ from independent record-map raster build")
			}

			if want := compactTestExpectedLevels(t, mapHandle, tc.cols, tc.rows, tc.layout, !tc.mono, params, seg, delta); !reflect.DeepEqual(compactMasks.LevelCache, want) {
				for i := range want {
					if compactMasks.LevelCache[i] != want[i] {
						t.Fatalf("level cell %d: compact=%v independent=%v", i, compactMasks.LevelCache[i], want[i])
					}
				}
			}
			if err := CompleteFrameWorkLoopFilterCompactState(state); err != nil {
				t.Fatalf("complete full compact frame: %v cells=%d blocks=%d", err, state.cells, state.blocks)
			}
			cells, blockCount, ok := FrameWorkLoopFilterCompactCoverage(compactMasks, state, tc.cols, tc.rows)
			if !ok || cells != int32(tc.cols*tc.rows) || blockCount != int32(len(blocks)) {
				t.Fatalf("compact coverage=(%d,%d,%v) want=(%d,%d,true)", cells, blockCount, ok, tc.cols*tc.rows, len(blocks))
			}
			wantNoSkip := compactTestExpectedNoSkip(mapHandle, tc.cols, tc.rows)
			if !reflect.DeepEqual(state.noskip, wantNoSkip) {
				t.Fatalf("compact no-skip bytes differ: got=%x want=%x", state.noskip, wantNoSkip)
			}
			for unitRow := 0; unitRow < unitRows; unitRow++ {
				for unitCol := 0; unitCol < unitCols; unitCol++ {
					got, ok := FrameWorkLoopFilterCompactUnitMask(compactMasks, state, unitRow, unitCol)
					off := (unitRow*unitCols + unitCol) * 8
					want := binary.LittleEndian.Uint64(wantNoSkip[off : off+8])
					if !ok || got != want {
						t.Fatalf("unit mask (%d,%d)=(%#x,%v) want %#x", unitCol, unitRow, got, ok, want)
					}
				}
			}
		})
	}
}

func TestCompactLoopFilterReadinessRequiresCompleteValidWalk(t *testing.T) {
	const cols, rows = 8, 8
	h := compactTestHandle(cols, rows, lfmask.Layout{SSHor: 1, SSVer: 1})
	state := &FrameWorkLoopFilterCompactState{ready: true, cells: 16, blocks: 1}
	noSkip := make([]byte, 8)
	for i := range noSkip {
		noSkip[i] = 0xff
	}
	if err := BindFrameWorkLoopFilterCompactState(h, state, noSkip, parser.LoopFilterParams{LevelY: [2]uint8{1, 1}}, parser.SegmentationParams{}, parser.DeltaParams{}, false); err != nil {
		t.Fatalf("rebind compact state: %v", err)
	}
	if state.ready || state.cells != 0 || state.blocks != 0 || !FrameWorkLoopFilterCompactStateActive(state) {
		t.Fatalf("bind did not reset compact state: %+v", *state)
	}
	if err := CompleteFrameWorkLoopFilterCompactState(state); err == nil {
		t.Fatal("incomplete frame unexpectedly became ready")
	}
	if FrameWorkLoopFilterCompactReady(h, state, cols, rows) {
		t.Fatal("incomplete frame reported ready")
	}

	bad := tile.BlockLoopVisit{Block: tile.BlockVisit{MICol: 0, MIRow: 0, MIColEnd: uint16(cols + 1), MIRowEnd: 2}}
	var scratch frameWorkLoopFilterMaskScratch
	scratch.reset(h)
	if err := h.buildCompact(state, &scratch, &bad, &tile.DecodeState{}, 4); err == nil {
		t.Fatal("out-of-frame block unexpectedly accepted")
	}
	if state.cells != 0 || state.blocks != 0 || state.ready {
		t.Fatalf("rejected block mutated readiness/coverage: %+v", *state)
	}
	InvalidateFrameWorkLoopFilterCompactState(state)
	if !FrameWorkLoopFilterCompactStateActive(state) || state.ready {
		t.Fatal("invalidate should leave active marker but clear readiness")
	}
	DeactivateFrameWorkLoopFilterCompactState(state)
	if FrameWorkLoopFilterCompactStateActive(state) || state.ready || state.noskip != nil {
		t.Fatal("deactivate should clear compact binding")
	}
}

func testFrameWorkLoopFilterBlock(t testing.TB, block tile.BlockVisit) FrameWorkLoopFilterBlock {
	t.Helper()
	out, err := FrameWorkLoopFilterBlockFromVisit(block)
	if err != nil {
		t.Fatal(err)
	}
	return out
}

func TestFrameWorkBatchLoopFilterMapShapeAndBind(t *testing.T) {
	ctx := testFrameWorkLoopFilterBatch(130, 65)
	cols, rows, length, err := ctx.LoopFilterMapShape()
	if err != nil {
		t.Fatal(err)
	}
	if cols != 34 || rows != 18 || length != 612 {
		t.Fatalf("shape cols=%d rows=%d length=%d want 34,18,612", cols, rows, length)
	}

	records := make([]FrameWorkLoopFilterBlockRecord, length+1)
	records[0].Valid = true
	records[length].Valid = true
	got, err := ctx.BindLoopFilterMap(records)
	if err != nil {
		t.Fatal(err)
	}
	if got.Stride != uint16(cols) || got.Rows != uint16(rows) || len(got.Records) != length {
		t.Fatalf("map=%+v len(records)=%d", got, len(got.Records))
	}
	if got.Records[0].Valid {
		t.Fatal("BindLoopFilterMap did not reset bound storage")
	}
	if !records[length].Valid {
		t.Fatal("BindLoopFilterMap reset storage beyond bound map length")
	}

	short := make([]FrameWorkLoopFilterBlockRecord, length-1)
	if _, err := ctx.BindLoopFilterMap(short); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("short records err=%v want %v", err, ErrInvalidBatch)
	}
	emptyBatch := FrameWorkBatch{}
	if _, _, _, err := emptyBatch.LoopFilterMapShape(); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("invalid batch shape err=%v want %v", err, ErrInvalidBatch)
	}
}

func TestFrameWorkLoopFilterMapMarkBlockRecordsInterMetadata(t *testing.T) {
	ctx := testFrameWorkLoopFilterBatch(32, 24)
	_, _, length, err := ctx.LoopFilterMapShape()
	if err != nil {
		t.Fatal(err)
	}
	filterMap, err := ctx.BindLoopFilterMap(make([]FrameWorkLoopFilterBlockRecord, length))
	if err != nil {
		t.Fatal(err)
	}

	state := &tile.DecodeState{
		DeltaLFFromBase: -3,
		DeltaLF:         [tile.FrameLoopFilterCount]int8{1, -2, 3, -4},
	}
	visit := tile.BlockLoopVisit{
		Block: tile.BlockVisit{
			MICol: 4, MIRow: 2, MIColEnd: 7, MIRowEnd: 5,
			Size: tile.BlockSize16x16,
		},
		SegmentID: 5,
		Prefix:    tile.BlockModeResult{SkipTransform: true},
		Prediction: tile.BlockPredictionModeResult{
			Valid:                true,
			InterReferencesValid: true,
			InterReferences: tile.InterReferencesResult{
				Ref0: tile.ReferenceFrameGolden, Ref1: tile.ReferenceFrameNone,
			},
			InterModeValid: true,
			InterMode:      tile.InterModeResult{Mode: tile.InterModeNewMV},
		},
		Coefficients: tile.BlockCoeffResult{
			Tree: tile.TransformTreeResult{Y: tile.TransformSize8x8},
		},
	}
	if err := filterMap.MarkBlock(visit, state); err != nil {
		t.Fatal(err)
	}
	wantBlock := testFrameWorkLoopFilterBlock(t, visit.Block)

	for miRow := uint32(2); miRow < 5; miRow++ {
		for miCol := uint32(4); miCol < 7; miCol++ {
			got := filterMap.Records[int(miRow)*int(filterMap.Stride)+int(miCol)]
			if !got.Valid || got.Block != wantBlock || got.TransformTree != visit.Coefficients.Tree {
				t.Fatalf("record[%d,%d]=%+v", miCol, miRow, got)
			}
			if got.SegmentID != 5 || !got.SkipTransform || got.Intra {
				t.Fatalf("record[%d,%d] flags=%+v", miCol, miRow, got)
			}
			if int(got.RefFrame) != int(tile.ReferenceFrameGolden)+1 || got.Mode != loopfilter.ModeDeltaClassMotion {
				t.Fatalf("record[%d,%d] ref/mode=%d/%d", miCol, miRow, got.RefFrame, got.Mode)
			}
			if got.DeltaLFFromBase != state.DeltaLFFromBase || got.DeltaLF != state.DeltaLF {
				t.Fatalf("record[%d,%d] delta=%d/%+v", miCol, miRow, got.DeltaLFFromBase, got.DeltaLF)
			}
		}
	}
	if filterMap.Records[0].Valid {
		t.Fatalf("outside record marked: %+v", filterMap.Records[0])
	}
}

func TestFrameWorkLoopFilterMapMarkBlockHandlesIntraAndCompound(t *testing.T) {
	ctx := testFrameWorkLoopFilterBatch(32, 16)
	_, _, length, err := ctx.LoopFilterMapShape()
	if err != nil {
		t.Fatal(err)
	}
	filterMap, err := ctx.BindLoopFilterMap(make([]FrameWorkLoopFilterBlockRecord, length))
	if err != nil {
		t.Fatal(err)
	}
	state := &tile.DecodeState{}

	intra := tile.BlockLoopVisit{
		Block:      tile.BlockVisit{MICol: 0, MIRow: 0, MIColEnd: 2, MIRowEnd: 2},
		Prediction: tile.BlockPredictionModeResult{Valid: true, Intra: true},
	}
	if err := filterMap.MarkBlock(intra, state); err != nil {
		t.Fatal(err)
	}
	if got := filterMap.Records[0]; !got.Valid || !got.Intra || got.RefFrame != 0 || got.Mode != loopfilter.ModeDeltaClassZero {
		t.Fatalf("intra record=%+v", got)
	}

	compound := tile.BlockLoopVisit{
		Block: tile.BlockVisit{MICol: 2, MIRow: 0, MIColEnd: 4, MIRowEnd: 2},
		Prediction: tile.BlockPredictionModeResult{
			Valid:                true,
			InterReferencesValid: true,
			InterReferences: tile.InterReferencesResult{
				Ref0: tile.ReferenceFrameLast, Ref1: tile.ReferenceFrameAltref,
				Compound: true,
			},
			InterModeValid: true,
			InterMode: tile.InterModeResult{
				Compound:     true,
				CompoundMode: tile.CompoundInterModeNearNew,
			},
		},
	}
	if err := filterMap.MarkBlock(compound, state); err != nil {
		t.Fatal(err)
	}
	if got := filterMap.Records[2]; !got.Valid || got.Intra || int(got.RefFrame) != int(tile.ReferenceFrameLast)+1 || got.Mode != loopfilter.ModeDeltaClassMotion {
		t.Fatalf("compound record=%+v", got)
	}

	intrabc := intra
	intrabc.Prediction.Intra = false
	intrabc.Prediction.Intrabc = true
	intrabc.Prediction.IntrabcValid = true
	if err := filterMap.MarkBlock(intrabc, state); err != nil {
		t.Fatal(err)
	}
	if got := filterMap.Records[0]; !got.Valid || got.Intra || got.RefFrame != 0 || got.Mode != loopfilter.ModeDeltaClassZero {
		t.Fatalf("intrabc record=%+v", got)
	}
}

func TestFrameWorkLoopFilterMapMarkBlockRejectsInvalidInputs(t *testing.T) {
	ctx := testFrameWorkLoopFilterBatch(16, 16)
	filterMap, err := ctx.BindLoopFilterMap(make([]FrameWorkLoopFilterBlockRecord, 16))
	if err != nil {
		t.Fatal(err)
	}
	state := &tile.DecodeState{}
	valid := tile.BlockLoopVisit{
		Block: tile.BlockVisit{MICol: 0, MIRow: 0, MIColEnd: 2, MIRowEnd: 2},
		Prediction: tile.BlockPredictionModeResult{
			Valid:                true,
			InterReferencesValid: true,
			InterReferences: tile.InterReferencesResult{
				Ref0: tile.ReferenceFrameLast, Ref1: tile.ReferenceFrameNone,
			},
			InterModeValid: true,
			InterMode:      tile.InterModeResult{Mode: tile.InterModeNearestMV},
		},
	}

	if err := filterMap.MarkBlock(valid, nil); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("nil state err=%v want %v", err, ErrInvalidBatch)
	}

	outside := valid
	outside.Block.MICol = 4
	outside.Block.MIColEnd = 6
	if err := filterMap.MarkBlock(outside, state); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("outside err=%v want %v", err, ErrInvalidBatch)
	}

	empty := valid
	empty.Block.MIColEnd = empty.Block.MICol
	if err := filterMap.MarkBlock(empty, state); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("empty err=%v want %v", err, ErrInvalidBatch)
	}

	badRef := valid
	badRef.Prediction.InterReferences.Ref0 = tile.ReferenceFrameNone
	if err := filterMap.MarkBlock(badRef, state); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("bad ref err=%v want %v", err, ErrInvalidBatch)
	}

	shortMap := FrameWorkLoopFilterMap{Records: nil, Stride: 1, Rows: 1}
	if err := shortMap.MarkBlock(valid, state); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("short map err=%v want %v", err, ErrInvalidBatch)
	}
	if err := shortMap.Reset(); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("short reset err=%v want %v", err, ErrInvalidBatch)
	}
}

func TestFrameWorkLoopFilterMapRecordAtAndForEachBlock(t *testing.T) {
	ctx := testFrameWorkLoopFilterBatch(32, 16)
	_, _, length, err := ctx.LoopFilterMapShape()
	if err != nil {
		t.Fatal(err)
	}
	filterMap, err := ctx.BindLoopFilterMap(make([]FrameWorkLoopFilterBlockRecord, length))
	if err != nil {
		t.Fatal(err)
	}
	state := &tile.DecodeState{}
	visit := tile.BlockLoopVisit{
		Block: tile.BlockVisit{MICol: 2, MIRow: 1, MIColEnd: 6, MIRowEnd: 3},
		Prediction: tile.BlockPredictionModeResult{
			Valid: true,
			Intra: true,
		},
	}
	if err := filterMap.MarkBlock(visit, state); err != nil {
		t.Fatal(err)
	}
	wantBlock := testFrameWorkLoopFilterBlock(t, visit.Block)

	record, ok, err := filterMap.RecordAt(2, 1)
	if err != nil {
		t.Fatal(err)
	}
	if !ok || !record.Valid || record.Block != wantBlock {
		t.Fatalf("top-left record=%+v ok=%v", record, ok)
	}
	record, ok, err = filterMap.RecordAt(5, 2)
	if err != nil {
		t.Fatal(err)
	}
	if !ok || record.Block != wantBlock {
		t.Fatalf("covered record=%+v ok=%v", record, ok)
	}
	if _, ok, err = filterMap.RecordAt(0, 0); err != nil || ok {
		t.Fatalf("missing record ok=%v err=%v", ok, err)
	}
	if _, _, err = filterMap.RecordAt(8, 0); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("out-of-range record err=%v want %v", err, ErrInvalidBatch)
	}
	stats, err := filterMap.CoverageStats(8, 4)
	if err != nil {
		t.Fatal(err)
	}
	if stats != (FrameWorkLoopFilterMapStats{Cells: 8, Blocks: 1, Missing: 24}) {
		t.Fatalf("coverage stats=%+v", stats)
	}

	var blocks int
	if err := filterMap.ForEachBlock(func(got FrameWorkLoopFilterBlockRecord) error {
		blocks++
		if got.Block != wantBlock {
			t.Fatalf("visited block=%+v want %+v", got.Block, wantBlock)
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if blocks != 1 {
		t.Fatalf("visited blocks=%d want 1", blocks)
	}
}

func TestFrameWorkLoopFilterMapGridHelpersIgnorePadding(t *testing.T) {
	filterMap := FrameWorkLoopFilterMap{
		Records: make([]FrameWorkLoopFilterBlockRecord, 8),
		Stride:  4,
		Rows:    2,
	}
	block := testFrameWorkLoopFilterBlock(t, tile.BlockVisit{MICol: 0, MIRow: 0, MIColEnd: 2, MIRowEnd: 2})
	record := FrameWorkLoopFilterBlockRecord{Valid: true, Block: block}
	filterMap.Records[0] = record
	filterMap.Records[1] = record
	filterMap.Records[4] = record
	filterMap.Records[5] = record
	filterMap.Records[2] = FrameWorkLoopFilterBlockRecord{
		Valid: true,
		Block: testFrameWorkLoopFilterBlock(t, tile.BlockVisit{MICol: 2, MIRow: 0, MIColEnd: 4, MIRowEnd: 2}),
	}

	stats, err := filterMap.CoverageStats(2, 2)
	if err != nil {
		t.Fatal(err)
	}
	if stats != (FrameWorkLoopFilterMapStats{Cells: 4, Blocks: 1}) {
		t.Fatalf("coverage stats=%+v", stats)
	}
	var blocks int
	if err := filterMap.ForEachBlockInGrid(2, 2, func(got FrameWorkLoopFilterBlockRecord) error {
		blocks++
		if got.Block != block {
			t.Fatalf("visited padding block=%+v", got.Block)
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if blocks != 1 {
		t.Fatalf("visited blocks=%d want 1", blocks)
	}
}

func TestFrameWorkLoopFilterMapForEachBlockRejectsInvalidInputs(t *testing.T) {
	if err := (FrameWorkLoopFilterMap{}).ForEachBlock(func(FrameWorkLoopFilterBlockRecord) error { return nil }); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("empty map err=%v want %v", err, ErrInvalidBatch)
	}
	ctx := testFrameWorkLoopFilterBatch(16, 16)
	filterMap, err := ctx.BindLoopFilterMap(make([]FrameWorkLoopFilterBlockRecord, 16))
	if err != nil {
		t.Fatal(err)
	}
	if err := filterMap.ForEachBlock(nil); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("nil visitor err=%v want %v", err, ErrInvalidBatch)
	}
	filterMap.Records[0] = FrameWorkLoopFilterBlockRecord{
		Valid: true,
		Block: testFrameWorkLoopFilterBlock(t, tile.BlockVisit{MICol: 0, MIRow: 0, MIColEnd: 5, MIRowEnd: 1}),
	}
	if err := filterMap.ForEachBlock(func(FrameWorkLoopFilterBlockRecord) error { return nil }); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("bad record err=%v want %v", err, ErrInvalidBatch)
	}
}

func TestFrameWorkLoopFilterMapAllocs(t *testing.T) {
	ctx := testFrameWorkLoopFilterBatch(16, 16)
	records := make([]FrameWorkLoopFilterBlockRecord, 16)
	state := &tile.DecodeState{DeltaLF: [tile.FrameLoopFilterCount]int8{1, 2, 3, 4}}
	visit := tile.BlockLoopVisit{
		Block: tile.BlockVisit{MICol: 0, MIRow: 0, MIColEnd: 2, MIRowEnd: 2},
		Prediction: tile.BlockPredictionModeResult{
			Valid:                true,
			InterReferencesValid: true,
			InterReferences: tile.InterReferencesResult{
				Ref0: tile.ReferenceFrameLast, Ref1: tile.ReferenceFrameNone,
			},
			InterModeValid: true,
			InterMode:      tile.InterModeResult{Mode: tile.InterModeNearestMV},
		},
	}
	allocs := testing.AllocsPerRun(1000, func() {
		filterMap, err := ctx.BindLoopFilterMap(records)
		if err != nil {
			t.Fatal(err)
		}
		if err := filterMap.MarkBlock(visit, state); err != nil {
			t.Fatal(err)
		}
		if _, ok, err := filterMap.RecordAt(0, 0); err != nil || !ok {
			t.Fatalf("RecordAt ok=%v err=%v", ok, err)
		}
		if _, err := filterMap.CoverageStats(4, 4); err != nil {
			t.Fatal(err)
		}
		if err := filterMap.ForEachBlock(func(FrameWorkLoopFilterBlockRecord) error {
			return nil
		}); err != nil {
			t.Fatal(err)
		}
		if err := filterMap.Reset(); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("FrameWorkLoopFilterMap allocated: %f", allocs)
	}
}

func BenchmarkFrameWorkLoopFilterMapMarkBlock(b *testing.B) {
	ctx := testFrameWorkLoopFilterBatch(64, 64)
	_, _, length, err := ctx.LoopFilterMapShape()
	if err != nil {
		b.Fatal(err)
	}
	filterMap, err := ctx.BindLoopFilterMap(make([]FrameWorkLoopFilterBlockRecord, length))
	if err != nil {
		b.Fatal(err)
	}
	state := &tile.DecodeState{DeltaLF: [tile.FrameLoopFilterCount]int8{1, -1, 2, -2}}
	visit := tile.BlockLoopVisit{
		Block: tile.BlockVisit{
			MICol:     0,
			MIRow:     0,
			MIColEnd:  16,
			MIRowEnd:  16,
			Size:      tile.BlockSize64x64,
			VisibleW4: 16,
			VisibleH4: 16,
		},
		Prediction: tile.BlockPredictionModeResult{
			Valid:                true,
			InterReferencesValid: true,
			InterReferences: tile.InterReferencesResult{
				Ref0: tile.ReferenceFrameLast, Ref1: tile.ReferenceFrameNone,
			},
			InterModeValid: true,
			InterMode:      tile.InterModeResult{Mode: tile.InterModeNearestMV},
		},
	}
	b.ReportAllocs()
	for b.Loop() {
		if err := filterMap.MarkBlock(visit, state); err != nil {
			b.Fatal(err)
		}
	}
}

func testFrameWorkLoopFilterBatch(width uint32, height uint32) FrameWorkBatch {
	return FrameWorkBatch{
		FrameWorkFrameContext: FrameWorkFrameContext{
			Sequence: FrameWorkSequenceContextFromHeader(parser.SequenceHeader{
				ColorConfig: parser.ColorConfig{BitDepth: 8},
			}),
			FrameSize: parser.FrameSize{CodedWidth: width, Height: height},
		},
	}
}

// loop_filter_masks_order_test.go pins the neighbour-derived BuildFromMap
// byte-for-byte against the decode-time carried-context build fed in true
// Z-order, for a staircase partition around a superblock boundary. This is
// content-independent: it compares the raw edge bitmasks, so a filter-width
// class divergence fails even when the deblocked pixels would coincide.

type orderTestBlock struct {
	bx, by, bw4, bh4 int
	size             tile.BlockSize
	tree             tile.TransformTreeResult
	intra            bool
}

func newOrderTestMasks(t *testing.T, cols, rows int) (*FrameWorkLoopFilterMasks, FrameWorkLoopFilterMap) {
	t.Helper()
	sb128w, sb128h, maskCount := FrameWorkLoopFilterMaskShape(cols, rows)
	h := &FrameWorkLoopFilterMasks{
		Masks:      make([]lfmask.FilterMask, maskCount),
		LevelCache: make([]uint8, 4*(cols*rows)),
		Cols:       cols, Rows: rows, SB128W: sb128w, SB128H: sb128h,
		Layout:    lfmask.Layout{SSHor: 1, SSVer: 1},
		HasChroma: true,
	}
	m := FrameWorkLoopFilterMap{
		Records: make([]FrameWorkLoopFilterBlockRecord, cols*rows),
		Stride:  uint16(cols), Rows: uint16(rows),
	}
	return h, m
}

func (b orderTestBlock) put(m FrameWorkLoopFilterMap) {
	stride := int(m.Stride)
	rec := FrameWorkLoopFilterBlockRecord{
		Valid: true, TransformTree: b.tree, Intra: b.intra,
		Block: FrameWorkLoopFilterBlock{
			MICol: uint16(b.bx), MIRow: uint16(b.by),
			MIColEnd: uint16(b.bx + b.bw4), MIRowEnd: uint16(b.by + b.bh4),
			X4: uint8(b.bx), Y4: uint8(b.by), Size: b.size,
			VisibleW4: uint8(b.bw4), VisibleH4: uint8(b.bh4),
		},
	}
	for r := b.by; r < b.by+b.bh4; r++ {
		for c := b.bx; c < b.bx+b.bw4; c++ {
			m.Records[r*stride+c] = rec
		}
	}
}

// zOrderStaircase returns blocks in true single-tile decode Z-order (superblocks
// raster, blocks Z within each SB) for a staircase around the SB boundary at
// col 16: the left SB's right column is 8x8s, the right SB is 16x16s.
func zOrderStaircase(cols, rows int) []orderTestBlock {
	tree8 := tile.TransformTreeResult{Y: tile.TransformSize8x8, UV: tile.TransformSize4x4, HasUV: true}
	tree16 := tile.TransformTreeResult{Y: tile.TransformSize16x16, UV: tile.TransformSize8x8, HasUV: true}
	tree32 := tile.TransformTreeResult{Y: tile.TransformSize32x32, UV: tile.TransformSize16x16, HasUV: true}
	var out []orderTestBlock
	// Superblocks are 64px = 16 MI. Iterate SB rows then SB cols (raster).
	for sbY := 0; sbY < rows; sbY += 16 {
		for sbX := 0; sbX < cols; sbX += 16 {
			bh := 16
			if sbY+bh > rows {
				bh = rows - sbY
			}
			bw := 16
			if sbX+bw > cols {
				bw = cols - sbX
			}
			switch sbX {
			case 0:
				// Left SB: 8x8 columns for cols 8..15, 32x32 for cols 0..7.
				for row := sbY; row < sbY+bh; row += 8 {
					out = append(out, orderTestBlock{0, row, 8, 8, tile.BlockSize32x32, tree32, true})
				}
				// 8x8s in Z-order within the right half column band.
				for row := sbY; row < sbY+bh; row += 2 {
					for col := 8; col < 16; col += 2 {
						out = append(out, orderTestBlock{col, row, 2, 2, tile.BlockSize8x8, tree8, true})
					}
				}
			case 16:
				// Right SB: 16x16 blocks in Z (raster within SB is fine for equal size).
				for row := sbY; row < sbY+bh; row += 4 {
					for col := 16; col < 32 && col < cols; col += 4 {
						out = append(out, orderTestBlock{col, row, 4, 4, tile.BlockSize16x16, tree16, true})
					}
				}
			default:
				for row := sbY; row < sbY+bh; row += 8 {
					for col := sbX; col < sbX+bw; col += 8 {
						out = append(out, orderTestBlock{col, row, 8, 8, tile.BlockSize32x32, tree32, true})
					}
				}
			}
		}
	}
	return out
}

func TestBuildFromMapMatchesZOrderStaircase(t *testing.T) {
	const cols, rows = 32, 32 // 128x128
	blocks := zOrderStaircase(cols, rows)

	// Reference: decode-time carried-context build fed in true Z-order.
	ref, _ := newOrderTestMasks(t, cols, rows)
	ob := make([]OrderedBlock, len(blocks))
	for i, b := range blocks {
		ob[i] = OrderedBlock{BX: b.bx, BY: b.by, Size: b.size, Tree: b.tree, Intra: b.intra}
	}
	ref.BuildFromOrderedBlocks(ob, 4)

	// Under test: neighbour-derived raster build from the record map.
	got, m := newOrderTestMasks(t, cols, rows)
	for _, b := range blocks {
		b.put(m)
	}
	var scratch FrameWorkLoopFilterMaskBuildScratch
	if err := got.BuildFromMap(&scratch, m, 4); err != nil {
		t.Fatalf("BuildFromMap: %v", err)
	}

	rb := ref.MaskBits()
	gb := got.MaskBits()
	if len(rb) != len(gb) {
		t.Fatalf("mask count mismatch %d vs %d", len(rb), len(gb))
	}
	for i := range rb {
		if rb[i] != gb[i] {
			// Find the first differing luma vertical-edge cell for a useful report.
			for pos := 0; pos < 32; pos++ {
				for str := 0; str < 3; str++ {
					for half := 0; half < 2; half++ {
						if rb[i].Y[0][pos][str][half] != gb[i].Y[0][pos][str][half] {
							t.Fatalf("region %d luma vert edge pos=%d strength=%d half=%d: ref=%#x mask=%#x",
								i, pos, str, half, rb[i].Y[0][pos][str][half], gb[i].Y[0][pos][str][half])
						}
					}
				}
			}
			t.Fatalf("region %d mask differs (non-luma-vert)", i)
		}
	}
}

func TestBuildFromMapMatchesInterZOrderStaircaseAtPartialFrame(t *testing.T) {
	const cols, rows = 31, 30 // Clip the final blocks at the right and bottom edges.
	raw := zOrderStaircase(32, 32)
	blocks := make([]orderTestBlock, 0, len(raw))
	for _, b := range raw {
		if b.bx >= cols || b.by >= rows {
			continue
		}
		if b.bx+b.bw4 > cols {
			b.bw4 = cols - b.bx
		}
		if b.by+b.bh4 > rows {
			b.bh4 = rows - b.by
		}
		b.intra = false
		switch b.size {
		case tile.BlockSize32x32:
			// Split the root, then only selected depth-1 children.
			b.tree.Split = [2]uint16{1, 0x0021}
		case tile.BlockSize16x16:
			b.tree.Split = [2]uint16{1, 0x0001}
		case tile.BlockSize8x8:
			// The root split reaches 4x4 leaves.
			b.tree.Split = [2]uint16{1, 0}
		}
		blocks = append(blocks, b)
	}

	ref, _ := newOrderTestMasks(t, cols, rows)
	ordered := make([]OrderedBlock, len(blocks))
	for i, b := range blocks {
		ordered[i] = OrderedBlock{BX: b.bx, BY: b.by, Size: b.size, Tree: b.tree, Intra: false}
	}
	ref.BuildFromOrderedBlocks(ordered, 4)

	got, m := newOrderTestMasks(t, cols, rows)
	for _, b := range blocks {
		b.put(m)
	}
	var scratch FrameWorkLoopFilterMaskBuildScratch
	if err := got.BuildFromMap(&scratch, m, 4); err != nil {
		t.Fatalf("BuildFromMap: %v", err)
	}
	refBits, gotBits := ref.MaskBits(), got.MaskBits()
	if len(refBits) != len(gotBits) {
		t.Fatalf("mask count mismatch %d vs %d", len(refBits), len(gotBits))
	}
	for i := range refBits {
		if refBits[i] != gotBits[i] {
			t.Fatalf("partial-frame inter mask differs in region %d", i)
		}
	}
}

func TestFrameWorkPaletteFillMatchesPerSampleStore(t *testing.T) {
	for _, bps := range []int{1, 2} {
		for _, size := range []uint8{2, 5, 8} {
			for _, dims := range [][2]int{{4, 4}, {8, 4}, {16, 16}, {12, 20}, {64, 64}} {
				w, h := dims[0], dims[1]
				const pw, ph = 80, 72
				mk := func() frame.Plane {
					p := frame.Plane{Pix: make([]byte, pw*ph*bps), Stride: pw * bps, Width: pw, Height: ph}
					for i := range p.Pix {
						p.Pix[i] = byte(i * 13)
					}
					return p
				}
				var colors [tile.PaletteMaxSize]uint16
				for i := range colors {
					colors[i] = uint16(17*i+3) << (8 * (bps - 1))
				}
				mapStride := w + 8
				idx := make([]uint8, mapStride*(h+4))
				for i := range idx {
					idx[i] = uint8(i*7) % size
				}
				got, want := mk(), mk()
				if err := frameWorkPaletteFill(got, bps, 3, 5, w, h, idx, mapStride, 4, 2, &colors, size); err != nil {
					t.Fatalf("bps%d size%d %dx%d: %v", bps, size, w, h, err)
				}
				for r := range h {
					for c := range w {
						frameWorkStoreSample(want, bps, 3+c, 5+r, colors[idx[(2+r)*mapStride+4+c]])
					}
				}
				if !bytes.Equal(got.Pix, want.Pix) {
					t.Fatalf("bps%d size%d %dx%d: mismatch", bps, size, w, h)
				}
				idx[(2+h-1)*mapStride+4+w-1] = size
				if err := frameWorkPaletteFill(mk(), bps, 3, 5, w, h, idx, mapStride, 4, 2, &colors, size); err == nil {
					t.Fatalf("bps%d size%d %dx%d: out-of-palette index accepted", bps, size, w, h)
				}
			}
		}
	}
}

func TestPoolExecute(t *testing.T) {
	jobs := testJobs()
	var batches [2]Batch
	n, err := BuildBatches(batches[:], jobs[:], 2)
	if err != nil {
		t.Fatal(err)
	}

	pool, err := NewPool(2)
	if err != nil {
		t.Fatal(err)
	}
	defer pool.Close()

	var seen [4]uint16
	err = pool.Execute(batches[:n], jobs[:], func(batch Batch, batchJobs []tile.Job) error {
		if len(batchJobs) != int(batch.Count) {
			t.Fatalf("jobs len=%d count=%d", len(batchJobs), batch.Count)
		}
		for i := range batchJobs {
			seen[int(batch.FirstJob)+i] = batchJobs[i].Tile + 1
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	for i := range len(jobs) {
		if seen[i] != jobs[i].Tile+1 {
			t.Fatalf("seen[%d]=%d job=%+v", i, seen[i], jobs[i])
		}
	}
}

func TestPoolExecuteFrameWork(t *testing.T) {
	jobs := testJobs()
	var batches [2]Batch
	n, err := BuildBatches(batches[:], jobs[:], 2)
	if err != nil {
		t.Fatal(err)
	}

	pool, err := NewPool(2)
	if err != nil {
		t.Fatal(err)
	}
	defer pool.Close()

	var output frame.Frame
	var reference frame.Frame
	payload := []byte{0xaa, 0xbb}
	base := FrameWorkBatch{
		Step:       decodework.FrameStep{Kind: decodework.FrameStepTile},
		Output:     &output,
		Payload:    payload,
		References: []*frame.Frame{&reference},
	}
	var seen [4]uint16
	err = pool.ExecuteFrameWork(batches[:n], jobs[:], base, func(ctx FrameWorkBatch) error {
		if ctx.Step.Kind != decodework.FrameStepTile {
			t.Fatalf("step=%+v", ctx.Step)
		}
		if ctx.Output != &output || len(ctx.References) != 1 || ctx.References[0] != &reference {
			t.Fatalf("ctx=%+v", ctx)
		}
		if len(ctx.Payload) != len(payload) || ctx.Payload[0] != payload[0] || ctx.Payload[1] != payload[1] {
			t.Fatalf("payload=%v want %v", ctx.Payload, payload)
		}
		if len(ctx.Jobs) != int(ctx.Batch.Count) {
			t.Fatalf("jobs len=%d count=%d", len(ctx.Jobs), ctx.Batch.Count)
		}
		for i := 0; i < len(ctx.Jobs); i++ {
			seen[int(ctx.Batch.FirstJob)+i] = ctx.Jobs[i].Tile + 1
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	for i := range len(jobs) {
		if seen[i] != jobs[i].Tile+1 {
			t.Fatalf("seen[%d]=%d job=%+v", i, seen[i], jobs[i])
		}
	}
}

func TestFrameWorkBatchSurface(t *testing.T) {
	tests := []struct {
		name string
		ctx  FrameWorkBatch
		want int
	}{
		{
			name: "begin",
			ctx: FrameWorkBatch{
				Step: decodework.FrameStep{
					Kind:  decodework.FrameStepBegin,
					Begin: decodework.FramePlan{Surface: 3},
				},
			},
			want: 3,
		},
		{
			name: "tile",
			ctx: FrameWorkBatch{
				Step: decodework.FrameStep{
					Kind: decodework.FrameStepTile,
					Tile: decodework.FrameTilePlan{Surface: 5},
				},
			},
			want: 5,
		},
	}
	for _, tt := range tests {
		got, err := tt.ctx.Surface()
		if err != nil {
			t.Fatalf("%s: Surface err=%v", tt.name, err)
		}
		if got != tt.want {
			t.Fatalf("%s: Surface=%d want %d", tt.name, got, tt.want)
		}
	}

	for _, ctx := range []FrameWorkBatch{
		{Step: decodework.FrameStep{Kind: decodework.FrameStepIgnored}},
		{Step: decodework.FrameStep{Kind: decodework.FrameStepBegin, Begin: decodework.FramePlan{Surface: -1}}},
		{Step: decodework.FrameStep{Kind: decodework.FrameStepTile, Tile: decodework.FrameTilePlan{Surface: -1}}},
	} {
		if _, err := ctx.Surface(); !errors.Is(err, ErrInvalidBatch) {
			t.Fatalf("Surface err=%v want %v for ctx=%+v", err, ErrInvalidBatch, ctx)
		}
	}
}

func TestPoolPropagatesFirstError(t *testing.T) {
	jobs := testJobs()
	var batches [2]Batch
	n, err := BuildBatches(batches[:], jobs[:], 2)
	if err != nil {
		t.Fatal(err)
	}
	want := errors.New("decode batch")

	pool, err := NewPool(2)
	if err != nil {
		t.Fatal(err)
	}
	defer pool.Close()

	err = pool.Execute(batches[:n], jobs[:], func(batch Batch, batchJobs []tile.Job) error {
		if batch.Worker == 1 {
			return want
		}
		return nil
	})
	if !errors.Is(err, want) {
		t.Fatalf("Execute err=%v want %v", err, want)
	}
}

func TestPoolRejectsInvalidInputs(t *testing.T) {
	jobs := testJobs()
	pool, err := NewPool(2)
	if err != nil {
		t.Fatal(err)
	}
	defer pool.Close()

	err = pool.Execute([]Batch{{Worker: 2, FirstJob: 0, Count: 1, FirstTile: 0, LastTile: 0}}, jobs[:], func(Batch, []tile.Job) error {
		return nil
	})
	if !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("Execute invalid batch err=%v want %v", err, ErrInvalidBatch)
	}

	err = pool.Execute([]Batch{{Worker: 0, FirstJob: 0, Count: 1, FirstTile: 0, LastTile: 0}}, jobs[:], nil)
	if !errors.Is(err, ErrInvalidCallback) {
		t.Fatalf("Execute nil callback err=%v want %v", err, ErrInvalidCallback)
	}

	err = pool.ExecuteFrameWork([]Batch{{Worker: 0, FirstJob: 0, Count: 1, FirstTile: 0, LastTile: 0}}, jobs[:], FrameWorkBatch{}, nil)
	if !errors.Is(err, ErrInvalidCallback) {
		t.Fatalf("ExecuteFrameWork nil callback err=%v want %v", err, ErrInvalidCallback)
	}
}

func TestPoolRejectsAfterClose(t *testing.T) {
	jobs := testJobs()
	batch := Batch{Worker: 0, FirstJob: 0, Count: 1, FirstTile: 0, LastTile: 0}
	pool, err := NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	pool.Close()
	pool.Close()

	err = pool.Execute([]Batch{batch}, jobs[:], func(Batch, []tile.Job) error {
		return nil
	})
	if !errors.Is(err, ErrPoolClosed) {
		t.Fatalf("Execute err=%v want %v", err, ErrPoolClosed)
	}
}

func TestPoolExecuteAllocs(t *testing.T) {
	jobs := testJobs()
	var batches [2]Batch
	n, err := BuildBatches(batches[:], jobs[:], 2)
	if err != nil {
		t.Fatal(err)
	}
	pool, err := NewPool(2)
	if err != nil {
		t.Fatal(err)
	}
	defer pool.Close()

	allocs := testing.AllocsPerRun(1000, func() {
		err := pool.Execute(batches[:n], jobs[:], func(Batch, []tile.Job) error {
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("Pool.Execute allocated: %f", allocs)
	}
}

func TestPoolExecuteFrameWorkAllocs(t *testing.T) {
	jobs := testJobs()
	var batches [2]Batch
	n, err := BuildBatches(batches[:], jobs[:], 2)
	if err != nil {
		t.Fatal(err)
	}
	pool, err := NewPool(2)
	if err != nil {
		t.Fatal(err)
	}
	defer pool.Close()

	base := FrameWorkBatch{Step: decodework.FrameStep{Kind: decodework.FrameStepTile}}
	allocs := testing.AllocsPerRun(1000, func() {
		err := pool.ExecuteFrameWork(batches[:n], jobs[:], base, noopFrameWorkBatch)
		if err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("Pool.ExecuteFrameWork allocated: %f", allocs)
	}
}

func BenchmarkPoolExecute(b *testing.B) {
	jobs := testJobs()
	var batches [2]Batch
	n, err := BuildBatches(batches[:], jobs[:], 2)
	if err != nil {
		b.Fatal(err)
	}
	pool, err := NewPool(2)
	if err != nil {
		b.Fatal(err)
	}
	defer pool.Close()

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = pool.Execute(batches[:n], jobs[:], func(Batch, []tile.Job) error {
			return nil
		})
	}
}

func BenchmarkPoolExecuteFrameWork(b *testing.B) {
	jobs := testJobs()
	var batches [2]Batch
	n, err := BuildBatches(batches[:], jobs[:], 2)
	if err != nil {
		b.Fatal(err)
	}
	pool, err := NewPool(2)
	if err != nil {
		b.Fatal(err)
	}
	defer pool.Close()

	base := FrameWorkBatch{Step: decodework.FrameStep{Kind: decodework.FrameStepTile}}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = pool.ExecuteFrameWork(batches[:n], jobs[:], base, noopFrameWorkBatch)
	}
}

func noopFrameWorkBatch(FrameWorkBatch) error {
	return nil
}

// TestFrameWorkPlaneRangeChromaSubsampleMatchesLibaomCeil pins the chroma
// sub-sampling formula used by frameWorkPlaneRange. AV1 / libaom compute the
// chroma extent from a luma extent as ROUND_POWER_OF_TWO(end, ss), i.e.
// (end + ss) >> ss for ss in {0, 1}. For ss=1 (4:2:0 / 4:2:2 / 4:4:0) the
// expanded form (end + 1) >> 1 is mathematically identical to the implemented
// (end >> 1) + (end & 1):
//
//	end even N:  N/2 == (N + 1) >> 1 == (N >> 1) + 0
//	end odd  N:  (N+1)/2 == (N + 1) >> 1 == (N >> 1) + 1
//
// This ceil rounding is REQUIRED at frame edges where CodedWidth/CodedHeight
// can be odd: the partially-covered last luma column (or row) MUST contribute
// its chroma sample to the visible/MD5 extent (frame/buffer.go allocates
// the chroma plane as (W + ssx) >> ssx samples wide). Switching to floor
// rounding (end >> 1) would silently drop the rightmost chroma column on
// odd-CodedWidth frames such as the 1x1 fast-suite 1280x720, leaking a
// stride-padding sample into the MD5.
//
// AV1 tile/SB boundaries are always MI-aligned (multiples of MI_SIZE=4 luma
// pixels => even), so inner boundaries are unaffected by ceil-vs-floor; the
// formula's odd-end branch only fires at the right/bottom frame edge.
func TestFrameWorkPlaneRangeChromaSubsampleMatchesLibaomCeil(t *testing.T) {
	tests := []struct {
		start uint32
		end   uint32
		// wantStart/wantEnd are the chroma extents when subsampled=true.
		wantStart uint32
		wantEnd   uint32
	}{
		// Even (MI-aligned) ends: ceil == floor.
		{start: 0, end: 0, wantStart: 0, wantEnd: 0},
		{start: 0, end: 8, wantStart: 0, wantEnd: 4},
		{start: 0, end: 64, wantStart: 0, wantEnd: 32},
		{start: 64, end: 128, wantStart: 32, wantEnd: 64},
		{start: 0, end: 640, wantStart: 0, wantEnd: 320}, // SVC L2T1 base width
		{start: 0, end: 1280, wantStart: 0, wantEnd: 640},
		// Odd ends (only valid at frame edges): ceil must round up.
		{start: 0, end: 1, wantStart: 0, wantEnd: 1},
		{start: 0, end: 17, wantStart: 0, wantEnd: 9}, // 34x34 +1 visible would be 17
		{start: 0, end: 33, wantStart: 0, wantEnd: 17},
		{start: 0, end: 35, wantStart: 0, wantEnd: 18}, // 35-wide frame edge
		{start: 0, end: 67, wantStart: 0, wantEnd: 34}, // 67-wide frame edge
		{start: 64, end: 67, wantStart: 32, wantEnd: 34},
	}
	for _, tt := range tests {
		// Subsampled axis: must match libaom's ROUND_POWER_OF_TWO(end, 1).
		gotStart, gotEnd := frameWorkPlaneRange(tt.start, tt.end, true)
		libaomEnd := (tt.end + 1) >> 1
		libaomStart := tt.start >> 1
		if gotStart != tt.wantStart || gotEnd != tt.wantEnd {
			t.Fatalf("frameWorkPlaneRange(%d,%d,true)=(%d,%d) want (%d,%d)", tt.start, tt.end, gotStart, gotEnd, tt.wantStart, tt.wantEnd)
		}
		if gotEnd != libaomEnd {
			t.Fatalf("frameWorkPlaneRange(%d,%d,true) end=%d != libaom (end+1)>>1=%d", tt.start, tt.end, gotEnd, libaomEnd)
		}
		// Start uses floor (start >> 1). Tile/SB starts are always MI-aligned
		// (multiples of MI_SIZE=4 luma => even), so floor and ceil agree, but
		// the formula's left edge must NEVER round up, otherwise two adjacent
		// jobs with MI-aligned boundaries would overlap on the chroma column.
		if gotStart != libaomStart {
			t.Fatalf("frameWorkPlaneRange(%d,%d,true) start=%d != floor (start>>1)=%d", tt.start, tt.end, gotStart, libaomStart)
		}
		// Non-subsampled axis: identity.
		gotStart, gotEnd = frameWorkPlaneRange(tt.start, tt.end, false)
		if gotStart != tt.start || gotEnd != tt.end {
			t.Fatalf("frameWorkPlaneRange(%d,%d,false)=(%d,%d) want (%d,%d)", tt.start, tt.end, gotStart, gotEnd, tt.start, tt.end)
		}
	}
}

// TestFrameWorkPlaneRangeAdjacentJobsNoChromaOverlap pins that two jobs whose
// MI-aligned luma extents abut at an even luma column (every valid AV1 tile/SB
// boundary) produce chroma ranges that abut without overlap: job A's chroma
// end equals job B's chroma start. This is the safety property that prevents
// the formula from leaking one chroma column of writes from one SB stripe
// into the next.
func TestFrameWorkPlaneRangeAdjacentJobsNoChromaOverlap(t *testing.T) {
	// Walk every plausible MI-aligned boundary up to a generous limit. MI_SIZE
	// is 4 luma pixels and SB sizes are >= 64, so any tile/SB-aligned boundary
	// is a multiple of 4 (hence even). Cover the broader "multiple of 4" set
	// to be safe.
	for boundary := uint32(0); boundary <= 4096; boundary += 4 {
		// Job A: [0, boundary). Job B: [boundary, boundary+8).
		_, aEnd := frameWorkPlaneRange(0, boundary, true)
		bStart, _ := frameWorkPlaneRange(boundary, boundary+8, true)
		if aEnd != bStart {
			t.Fatalf("boundary luma=%d chroma aEnd=%d bStart=%d (gap or overlap)", boundary, aEnd, bStart)
		}
	}
}

// TestPoolRunRanges verifies the row-band postfilter dispatch: bands tile the
// whole [0, n) range with no gap or overlap, the band count honours the
// WorkerCount/maxBands/n cap, every band index is unique, and worker errors
// propagate. This is the safety net under the pooled CDEF postfilter split.
func TestPoolRunRanges(t *testing.T) {
	for _, workers := range []int{1, 2, 3, 8} {
		pool, err := NewPool(workers)
		if err != nil {
			t.Fatalf("workers=%d: %v", workers, err)
		}
		for _, tc := range []struct {
			n        int
			maxBands int
		}{{0, 0}, {1, 0}, {5, 0}, {17, 0}, {17, 4}, {3, 8}, {64, 8}} {
			covered := make([]int32, tc.n)
			bandSeen := make([]int32, workers+1)
			var mu sync.Mutex
			bands := 0
			err := pool.RunRanges(tc.n, tc.maxBands, func(band, lo, hi int) error {
				if lo < 0 || hi > tc.n || lo > hi {
					return errors.New("bad range")
				}
				for i := lo; i < hi; i++ {
					atomic.AddInt32(&covered[i], 1)
				}
				mu.Lock()
				if band >= 0 && band < len(bandSeen) {
					bandSeen[band]++
				}
				bands++
				mu.Unlock()
				return nil
			})
			if err != nil {
				t.Fatalf("workers=%d n=%d maxBands=%d: %v", workers, tc.n, tc.maxBands, err)
			}
			for i := range covered {
				if covered[i] != 1 {
					t.Fatalf("workers=%d n=%d maxBands=%d: index %d covered %d times", workers, tc.n, tc.maxBands, i, covered[i])
				}
			}
			if tc.n > 0 {
				wantMax := min(workers, tc.n)
				if tc.maxBands > 0 && tc.maxBands < wantMax {
					wantMax = tc.maxBands
				}
				if bands != wantMax {
					t.Fatalf("workers=%d n=%d maxBands=%d: bands=%d want %d", workers, tc.n, tc.maxBands, bands, wantMax)
				}
			}
			for b, c := range bandSeen {
				if c > 1 {
					t.Fatalf("workers=%d n=%d maxBands=%d: band %d ran %d times", workers, tc.n, tc.maxBands, b, c)
				}
			}
		}

		sentinel := errors.New("boom")
		err = pool.RunRanges(16, 0, func(band, lo, hi int) error {
			if band == 0 {
				return sentinel
			}
			return nil
		})
		if !errors.Is(err, sentinel) {
			t.Fatalf("workers=%d: RunRanges error=%v want %v", workers, err, sentinel)
		}
		pool.Close()
	}
}

func TestPoolRunRangesNoWarmAllocs(t *testing.T) {
	pool, err := NewPool(2)
	if err != nil {
		t.Fatal(err)
	}
	defer pool.Close()

	var work atomic.Int64
	callback := RangeFunc(func(_, lo, hi int) error {
		work.Add(int64(hi - lo))
		return nil
	})
	for range 3 {
		if err := pool.RunRanges(64, 2, callback); err != nil {
			t.Fatalf("warm RunRanges: %v", err)
		}
	}
	const runs = 25
	allocs := testing.AllocsPerRun(runs, func() {
		if err := pool.RunRanges(64, 2, callback); err != nil {
			t.Fatalf("RunRanges: %v", err)
		}
	})
	if allocs != 0 {
		t.Fatalf("warm two-band RunRanges allocations/run=%g, want 0", allocs)
	}
	if got, want := work.Load(), int64(64*(3+runs+1)); got != want {
		t.Fatalf("callback work=%d, want %d", got, want)
	}
}

type rangeRunnerCoverage struct {
	covered []atomic.Int32
	bands   atomic.Int32
}

func (r *rangeRunnerCoverage) RunRange(band, lo, hi int) error {
	r.bands.Add(1)
	for i := lo; i < hi; i++ {
		r.covered[i].Add(1)
	}
	return nil
}

type rangeRunnerDrain struct {
	entered   chan int
	release   <-chan struct{}
	failBand  int
	fail      error
	completed atomic.Int32
}

func (r *rangeRunnerDrain) RunRange(band, _, _ int) error {
	r.entered <- band
	if band != r.failBand {
		<-r.release
	}
	r.completed.Add(1)
	if band == r.failBand {
		return r.fail
	}
	return nil
}

func TestPoolRunRangesRunnerDrainsErrorsAndReusesPool(t *testing.T) {
	pool, err := NewPool(2)
	if err != nil {
		t.Fatal(err)
	}
	defer pool.Close()

	sentinel := errors.New("range failed")
	release := make(chan struct{})
	drain := &rangeRunnerDrain{
		entered:  make(chan int, 2),
		release:  release,
		failBand: 0,
		fail:     sentinel,
	}
	finished := make(chan error, 1)
	go func() { finished <- pool.RunRangesRunner(8, 2, drain) }()
	<-drain.entered
	<-drain.entered
	select {
	case err := <-finished:
		t.Fatalf("RunRangesRunner returned before all submitted bands joined: %v", err)
	default:
	}
	close(release)
	if err := <-finished; !errors.Is(err, sentinel) {
		t.Fatalf("RunRangesRunner error=%v, want %v", err, sentinel)
	}
	if got := drain.completed.Load(); got != 2 {
		t.Fatalf("completed ranges=%d, want 2 after the error barrier", got)
	}

	coverage := &rangeRunnerCoverage{covered: make([]atomic.Int32, 17)}
	if err := pool.RunRangesRunner(len(coverage.covered), 2, coverage); err != nil {
		t.Fatalf("pool reuse after range error: %v", err)
	}
	if got := coverage.bands.Load(); got != 2 {
		t.Fatalf("reused runner bands=%d, want 2", got)
	}
	for i := range coverage.covered {
		if got := coverage.covered[i].Load(); got != 1 {
			t.Fatalf("reused runner index %d covered %d times, want once", i, got)
		}
	}
}

func TestFreshByteDirectionalEdgesIgnorePoisonOutsideConsumedSpans(t *testing.T) {
	for _, size := range []int{4, 8, 16, 32, 64} {
		for _, angle := range []int{45, 67, 90, 113, 135, 157, 180, 203, 225} {
			for available := 0; available < 4; available++ {
				for _, filter := range []bool{false, true} {
					var scratch FrameWorkIntraPredictionScratch
					for i := range scratch.Above {
						scratch.Above[i] = 0xffff
						scratch.Left[i] = 0xffff
					}
					for i := range scratch.Edge {
						scratch.Edge[i] = 0xffff
					}
					block := tile.BlockVisit{HaveTop: available&1 != 0, HaveLeft: available&2 != 0}
					x, y := 0, 0
					if block.HaveLeft {
						x = 1
					}
					if block.HaveTop {
						y = 1
					}
					plane := frame.Plane{Pix: make([]byte, 96*96), Stride: 96, Width: 96, Height: 96}
					for i := range plane.Pix {
						plane.Pix[i] = byte(i*79 + 3)
					}
					edges, err := frameWorkDirectionalPredictionEdges(plane, 1, 8, x, y, size, size, angle, block, &scratch, filter, false, true, true, x+size-1, y+size-1)
					if err != nil {
						t.Fatalf("gather size%d angle%d available%d filter%v: %v", size, angle, available, filter, err)
					}
					want, got := plane, plane
					want.Pix = slices.Clone(plane.Pix)
					got.Pix = slices.Clone(plane.Pix)
					// The general entry scans every consumed sample and therefore
					// rejects any poison left reachable by the fresh-gather route.
					we := prediction.PredictDirectionalIntraPlaneBlockWithExtent(want, 1, 8, x, y, size-1, size-1, size, size, angle, edges)
					ge := prediction.PredictDirectionalIntraPlaneBlockWithExtentBounded(got, 1, 8, x, y, size-1, size-1, size, size, angle, edges, true)
					if we != nil || ge != nil || !bytes.Equal(want.Pix, got.Pix) {
						t.Fatalf("size%d angle%d available%d filter%v: %v/%v or pixels differ", size, angle, available, filter, we, ge)
					}
				}
			}
		}
	}
}

func TestDirectDirectionalEdgeFilterStillRejectsUnboundedSamples(t *testing.T) {
	var scratch FrameWorkIntraPredictionScratch
	for i := range scratch.Above {
		scratch.Above[i] = 0xffff
		scratch.Left[i] = 0xffff
	}
	edges := prediction.DirectionalEdges{Above: scratch.Above[:], Left: scratch.Left[:], AboveOrigin: frameWorkDirectionalEdgeOrigin, LeftOrigin: frameWorkDirectionalEdgeOrigin}
	if err := frameWorkApplyDirectionalIntraEdgeFilterBounded(8, 16, 16, 135, tile.BlockVisit{HaveTop: true, HaveLeft: true}, false, &edges, &scratch, false); err == nil {
		t.Fatal("direct helper accepted unbounded caller-supplied samples")
	}
}

var benchmarkCFLPredictionSink int

func TestFrameWorkBatchPredictBlockLumaIntraDC(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, Align: 64})
	testFillFrame(output, 0)
	ctx := testIntraPredictionBatch(output)

	for x := 16; x < 32; x++ {
		setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, x, 15, 10)
	}
	for y := 16; y < 32; y++ {
		setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, 15, y, 50)
	}

	var scratch FrameWorkIntraPredictionScratch
	visit := testIntraPredictionVisit(tile.IntraModeDC)
	if err := ctx.PredictBlockLumaIntra(0, visit, &scratch); err != nil {
		t.Fatal(err)
	}
	for y := 16; y < 32; y++ {
		for x := 16; x < 32; x++ {
			if got := frameWorkTestSample(output.Y, output.Layout.BytesPerSample, x, y); got != 30 {
				t.Fatalf("sample(%d,%d)=%d want 30", x, y, got)
			}
		}
	}
}

func TestFrameWorkBatchPredictBlockPalette10Bit444(t *testing.T) {
	output := testBatchFrame(t, frame.Format{
		Width:        64,
		Height:       64,
		BitDepth:     10,
		SubsamplingX: false,
		SubsamplingY: false,
		Align:        64,
	})
	testFillFrame(output, 0)
	ctx := testIntraPredictionBatch(output)

	maps := &testPayloads.Palette[0]
	for y := range 16 {
		for x := range 16 {
			maps.Y[y*16+x] = uint8((x/5 + y/7) % 3)
			maps.UV[y*16+x] = uint8((x/4 + y/6) % 3)
		}
	}
	palette := tile.PaletteModeResult{
		YSize:     3,
		UVSize:    3,
		YColors:   [tile.PaletteMaxSize]uint16{17, 511, 1000},
		UColors:   [tile.PaletteMaxSize]uint16{33, 444, 901},
		VColors:   [tile.PaletteMaxSize]uint16{55, 666, 990},
		YMapSlot:  1,
		UVMapSlot: 1,
	}
	visit := testIntraPredictionVisit(tile.IntraModeDC)
	visit.Prediction.ChromaModeValid = true
	visit.Prediction.ChromaMode = tile.ChromaIntraModeDC
	visit.Prediction.Palette = palette

	var scratch FrameWorkIntraPredictionScratch
	if err := ctx.PredictBlockIntra(0, visit, &scratch); err != nil {
		t.Fatal(err)
	}
	for y := range 16 {
		for x := range 16 {
			yIndex := maps.Y[y*16+x]
			if got, want := frameWorkTestSample(output.Y, output.Layout.BytesPerSample, 16+x, 16+y), palette.YColors[yIndex]; got != want {
				t.Fatalf("Y sample(%d,%d)=%d want %d", 16+x, 16+y, got, want)
			}
			uvIndex := maps.UV[y*16+x]
			if got, want := frameWorkTestSample(output.U, output.Layout.BytesPerSample, 16+x, 16+y), palette.UColors[uvIndex]; got != want {
				t.Fatalf("U sample(%d,%d)=%d want %d", 16+x, 16+y, got, want)
			}
			if got, want := frameWorkTestSample(output.V, output.Layout.BytesPerSample, 16+x, 16+y), palette.VColors[uvIndex]; got != want {
				t.Fatalf("V sample(%d,%d)=%d want %d", 16+x, 16+y, got, want)
			}
		}
	}
}

func TestFrameWorkBatchPredictBlockLumaFilterIntra(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, Align: 64})
	testFillFrame(output, 0)
	ctx := testIntraPredictionBatch(output)

	for x := 16; x < 32; x++ {
		setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, x, 15, uint16(20+x))
	}
	for y := 16; y < 32; y++ {
		setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, 15, y, uint16(40+y))
	}
	setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, 15, 15, 33)

	edges := testFrameWorkIntraEdges(output, output.Y, 16, 16, 16, 16)
	wantPix := make([]byte, 16*16)
	want := frame.Plane{Pix: wantPix, Stride: 16, Width: 16, Height: 16}
	if err := prediction.PredictFilterIntraPlaneBlockWithExtentBounded(want, 1, 8, 0, 0, 16, 16, 16, 16, prediction.FilterIntraModePaeth, edges, false); err != nil {
		t.Fatal(err)
	}

	var scratch FrameWorkIntraPredictionScratch
	visit := testIntraPredictionVisit(tile.IntraModeDC)
	visit.Prediction.FilterIntraValid = true
	visit.Prediction.FilterIntraMode = tile.FilterIntraModePaeth
	if err := ctx.PredictBlockLumaIntra(0, visit, &scratch); err != nil {
		t.Fatal(err)
	}
	for y := range 16 {
		for x := range 16 {
			got := frameWorkTestSample(output.Y, output.Layout.BytesPerSample, 16+x, 16+y)
			if want := uint16(wantPix[y*16+x]); got != want {
				t.Fatalf("sample(%d,%d)=%d want %d", 16+x, 16+y, got, want)
			}
		}
	}
}

func TestFrameWorkBatchPredictBlockLumaIntraStaticModes(t *testing.T) {
	for _, tt := range []struct {
		name string
		mode tile.IntraMode
		want func(x int, y int) uint16
	}{
		{
			name: "vertical",
			mode: tile.IntraModeVertical,
			want: func(x int, _ int) uint16 { return uint16(200 + x - 16) },
		},
		{
			name: "horizontal",
			mode: tile.IntraModeHorizontal,
			want: func(_ int, y int) uint16 { return uint16(300 + y - 16) },
		},
		{
			name: "paeth",
			mode: tile.IntraModePaeth,
			want: func(_ int, y int) uint16 { return uint16(300 + y - 16) },
		},
		{
			name: "smooth",
			mode: tile.IntraModeSmooth,
			want: nil,
		},
		{
			name: "smooth-vertical",
			mode: tile.IntraModeSmoothVertical,
			want: nil,
		},
		{
			name: "smooth-horizontal",
			mode: tile.IntraModeSmoothHorizontal,
			want: nil,
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 10, Align: 128})
			ctx := testIntraPredictionBatch(output)
			for x := 16; x < 32; x++ {
				setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, x, 15, uint16(200+x-16))
			}
			for y := 16; y < 32; y++ {
				setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, 15, y, uint16(300+y-16))
			}
			setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, 15, 15, 180)

			var scratch FrameWorkIntraPredictionScratch
			visit := testIntraPredictionVisit(tt.mode)
			if err := ctx.PredictBlockLumaIntra(0, visit, &scratch); err != nil {
				t.Fatal(err)
			}
			for y := 16; y < 32; y++ {
				for x := 16; x < 32; x++ {
					got := frameWorkTestSample(output.Y, output.Layout.BytesPerSample, x, y)
					if got > 1023 {
						t.Fatalf("sample(%d,%d)=%d exceeds 10-bit max", x, y, got)
					}
					if tt.want != nil {
						if want := tt.want(x, y); got != want {
							t.Fatalf("sample(%d,%d)=%d want %d", x, y, got, want)
						}
					}
				}
			}
		})
	}
}

func TestFrameWorkBatchPredictBlockLumaIntraClipsFrameEdge(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 320, Height: 180, BitDepth: 8, MonoChrome: true, Align: 64})
	testFillFrame(output, 7)
	ctx := testIntraPredictionBatch(output)
	ctx.Jobs = []tile.Job{{SBCols: 5, SBRows: 3}}
	for x := 112; x < 128; x++ {
		setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, x, 175, uint16(30+x-112))
	}

	visit := testIntraPredictionVisit(tile.IntraModeVertical)
	visit.Block = tile.BlockVisit{
		MICol: 28, MIRow: 44, MIColEnd: 32, MIRowEnd: 46,
		X4: 12, Y4: 12, Size: tile.BlockSize16x8, VisibleW4: 4, VisibleH4: 2,
		HaveTop: true, HaveLeft: true,
	}
	var scratch FrameWorkIntraPredictionScratch
	if err := ctx.PredictBlockLumaIntra(0, visit, &scratch); err != nil {
		t.Fatal(err)
	}
	for y := 176; y < 180; y++ {
		for x := 112; x < 128; x++ {
			want := uint16(30 + x - 112)
			if got := frameWorkTestSample(output.Y, output.Layout.BytesPerSample, x, y); got != want {
				t.Fatalf("sample(%d,%d)=%d want %d", x, y, got, want)
			}
		}
	}
}

func TestFrameWorkBatchPredictBlockInterClipsFrameEdge(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 320, Height: 180, BitDepth: 8, MonoChrome: true, Align: 64})
	reference := testBatchFrame(t, output.Format)
	testFillFrame(output, 7)
	testFillFrame(reference, 0x55)
	ctx := testInterPredictionBatch(output, reference)
	ctx.Jobs = []tile.Job{{SBCols: 5, SBRows: 3}}

	visit := testInterPredictionVisit(motion.Vector{})
	visit.Block = tile.BlockVisit{
		MICol: 0, MIRow: 44, MIColEnd: 16, MIRowEnd: 46,
		X4: 0, Y4: 12, Size: tile.BlockSize64x16, VisibleW4: 16, VisibleH4: 2,
		HaveTop: true,
	}
	if err := ctx.PredictBlockInterWithFilters(0, visit, nil, motion.RegularFilters); err != nil {
		t.Fatal(err)
	}
	for y := 176; y < 180; y++ {
		for x := range 64 {
			if got := output.Y.Pix[y*output.Y.Stride+x]; got != 0x55 {
				t.Fatalf("sample(%d,%d)=%d want 0x55", x, y, got)
			}
		}
	}
}

// TestFrameWorkClipVisiblePixelsToWindow covers the boundary cases that show
// up in the 34x34 libaom-extended fast-suite vector: MI-aligned blocks at the
// right/bottom edge whose nominal extent (e.g. 4x4 or 8x8 transforms aligned
// to MI col 8 in a 34-pixel-wide luma plane) overshoots the coded frame and
// must clip down to the writable extent.
//
// When ClipWidth/ClipHeight are zero, the clamp falls back to Width/Height
// (visible coded-frame edge) — this is the older single-extent behaviour and
// is what the test harness in TestFrameWorkBatchPredictBlockLumaIntraDC and
// friends still relies on.
//
// When ClipWidth/ClipHeight are set, the clamp clamps to the MI-aligned
// writable extent (xd->mi_params.mi_cols * MI_SIZE in libaom; region.MIColEnd
// * 4 in goav1). This is the regression behaviour pinned by the structural
// past-visible-write fix: blocks that straddle the visible boundary keep
// writing into the past-visible padding so later blocks see the same
// neighbors libaom does. Frame-edge MD5 / output is still computed over the
// visible Width/Height range; the past-visible writes land in the stride
// padding that the MD5 harness ignores.
func TestFrameWorkClipVisiblePixelsToWindow(t *testing.T) {
	luma := FrameWorkPlaneRegion{X: 0, Y: 0, Width: 34, Height: 34}
	chroma := FrameWorkPlaneRegion{X: 0, Y: 0, Width: 17, Height: 17}
	// MI-aligned luma extent for a 34x34 frame: mi_cols = ((34+7)>>3)<<1 = 10,
	// luma write extent = 10*4 = 40. Chroma 4:2:0 = 20.
	lumaAligned := FrameWorkPlaneRegion{X: 0, Y: 0, Width: 34, Height: 34, ClipWidth: 40, ClipHeight: 40}
	chromaAligned := FrameWorkPlaneRegion{X: 0, Y: 0, Width: 17, Height: 17, ClipWidth: 20, ClipHeight: 20}
	tests := []struct {
		name         string
		window       FrameWorkPlaneRegion
		x, y, w, h   int
		wantW, wantH int
		wantOK       bool
	}{
		// Visible-extent fallback (legacy semantics).
		{"luma corner 8x8 visible 2x2", luma, 32, 32, 8, 8, 2, 2, true},
		{"luma right 4x8 visible 2x8", luma, 32, 8, 4, 8, 2, 8, true},
		{"luma right 4x4 visible 2x4", luma, 32, 16, 4, 4, 2, 4, true},
		{"luma bottom 8x4 visible 8x2", luma, 16, 32, 8, 4, 8, 2, true},
		{"luma bottom 16x8 visible 16x2", luma, 0, 32, 16, 8, 16, 2, true},
		{"luma right-most pixel 1x1 visible 1x1", luma, 33, 33, 1, 1, 1, 1, true},
		{"luma 4x4 at MI col 9 entirely past visible", luma, 36, 8, 4, 4, 0, 0, false},
		{"luma 8x8 at MI col 9 entirely past visible", luma, 36, 0, 8, 8, 0, 0, false},
		{"luma block at exact right boundary", luma, 34, 0, 4, 4, 0, 0, false},
		{"chroma corner 4x4 visible 1x1", chroma, 16, 16, 4, 4, 1, 1, true},
		{"chroma right 2x4 visible 1x4", chroma, 16, 8, 2, 4, 1, 4, true},
		{"chroma bottom 4x2 visible 4x1", chroma, 8, 16, 4, 2, 4, 1, true},
		{"chroma 2x4 entirely past visible", chroma, 18, 0, 2, 4, 0, 0, false},
		{"zero width rejects", luma, 0, 0, 0, 4, 0, 0, false},
		{"zero height rejects", luma, 0, 0, 4, 0, 0, 0, false},
		{"negative origin rejects", luma, -1, 0, 4, 4, 0, 0, false},
		// Aligned-extent semantics: ClipWidth/ClipHeight > Width/Height.
		// A block whose nominal extent fits inside the MI-aligned write
		// extent must return its full size, not the smaller visible clip.
		{"aligned luma corner 8x8 full 8x8", lumaAligned, 32, 32, 8, 8, 8, 8, true},
		{"aligned luma right 4x8 full 4x8", lumaAligned, 32, 8, 4, 8, 4, 8, true},
		{"aligned luma right 4x4 full 4x4", lumaAligned, 32, 16, 4, 4, 4, 4, true},
		{"aligned luma bottom 16x8 full 16x8", lumaAligned, 0, 32, 16, 8, 16, 8, true},
		{"aligned luma MI col 9 4x4 full 4x4", lumaAligned, 36, 8, 4, 4, 4, 4, true},
		{"aligned luma MI col 9 8x8 full 4x8", lumaAligned, 36, 0, 8, 8, 4, 8, true},
		{"aligned luma at exact aligned boundary", lumaAligned, 40, 0, 4, 4, 0, 0, false},
		{"aligned chroma corner 4x4 full 4x4", chromaAligned, 16, 16, 4, 4, 4, 4, true},
		{"aligned chroma right 2x4 full 2x4", chromaAligned, 16, 8, 2, 4, 2, 4, true},
		{"aligned chroma bottom 4x2 full 4x2", chromaAligned, 8, 16, 4, 2, 4, 2, true},
		{"aligned chroma 2x4 at MI col 9 full 2x4", chromaAligned, 18, 0, 2, 4, 2, 4, true},
		{"aligned chroma block at exact aligned boundary", chromaAligned, 20, 0, 2, 4, 0, 0, false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			gotW, gotH, ok := frameWorkClipVisiblePixelsToWindow(tc.window, tc.x, tc.y, tc.w, tc.h)
			if ok != tc.wantOK || gotW != tc.wantW || gotH != tc.wantH {
				t.Fatalf("clip(x=%d,y=%d,w=%d,h=%d) = (%d,%d,%v) want (%d,%d,%v)",
					tc.x, tc.y, tc.w, tc.h, gotW, gotH, ok, tc.wantW, tc.wantH, tc.wantOK)
			}
		})
	}
}

// TestFrameWorkJobOutputPlaneClipExtentMatchesMIAlignment pins the MI-aligned
// writable extent that JobOutputPlane records on a 34x34 frame's output
// region. The visible Width/Height stay at the coded-frame edge (34 / 17 for
// chroma 4:2:0) but ClipWidth/ClipHeight extend to mi_cols * MI_SIZE
// (40 luma, 20 chroma). The window's Pix slice extends through
// (ClipHeight-1)*Stride + ClipRowBytes so AV1 prediction and residual writes
// past the visible edge stay within the underlying plane buffer.
//
// This is the regression behaviour locked in for the structural
// past-visible-write fix: predictors that previously zero-clipped at col 34
// (luma) / 17 (chroma) now write up to col 39 / 19, exactly mirroring
// libaom's xd->mi_params.mi_cols * MI_SIZE write boundary.
func TestFrameWorkJobOutputPlaneClipExtentMatchesMIAlignment(t *testing.T) {
	output := testBatchFrame(t, frame.Format{
		Width:        34,
		Height:       34,
		BitDepth:     8,
		SubsamplingX: true,
		SubsamplingY: true,
		Align:        32,
	})
	ctx := FrameWorkBatch{
		Output: output,
		FrameWorkFrameContext: FrameWorkFrameContext{
			Sequence: FrameWorkSequenceContextFromHeader(parser.SequenceHeader{
				ColorConfig: parser.ColorConfig{BitDepth: 8, SubsamplingX: true, SubsamplingY: true},
			}),
			FrameSize: parser.FrameSize{CodedWidth: 34, Height: 34},
		},
		Jobs: []tile.Job{
			{Tile: 0, Row: 0, Col: 0, SBX: 0, SBY: 0, SBCols: 1, SBRows: 1},
		},
	}
	y, err := ctx.JobOutputPlane(0, FrameWorkPlaneY)
	if err != nil {
		t.Fatal(err)
	}
	// ReadWidth/ReadHeight cap neighbor reads at the MI-aligned 40x40 extent
	// (mi_cols = mi_rows = ((34+7)>>3)<<1 = 10 -> 40 luma px). ClipWidth/Height
	// are the superblock-aligned writable extent (SBCols*64 = 64), clamped to
	// the SB-aligned allocation (frame.RequiredSize rounds 34 -> 64). The
	// partial bottom/right superblock reconstructs the full transform into that
	// padding the way libaom writes into the YV12 border.
	if y.X != 0 || y.Y != 0 || y.Width != 34 || y.Height != 34 ||
		y.ReadWidth != 40 || y.ReadHeight != 40 ||
		y.ClipWidth != 64 || y.ClipHeight != 64 || y.ClipRowBytes != 64 {
		t.Fatalf("Y plane region=%+v", y)
	}
	if len(y.Pix) != int((y.ClipHeight-1)*y.Stride+y.ClipRowBytes) {
		t.Fatalf("Y len=%d region=%+v", len(y.Pix), y)
	}
	if y.Stride < y.ClipRowBytes {
		t.Fatalf("Y Stride=%d ClipRowBytes=%d", y.Stride, y.ClipRowBytes)
	}

	u, err := ctx.JobOutputPlane(0, FrameWorkPlaneU)
	if err != nil {
		t.Fatal(err)
	}
	if u.X != 0 || u.Y != 0 || u.Width != 17 || u.Height != 17 ||
		u.ReadWidth != 20 || u.ReadHeight != 20 ||
		u.ClipWidth != 32 || u.ClipHeight != 32 || u.ClipRowBytes != 32 {
		t.Fatalf("U plane region=%+v", u)
	}
	if len(u.Pix) != int((u.ClipHeight-1)*u.Stride+u.ClipRowBytes) {
		t.Fatalf("U len=%d region=%+v", len(u.Pix), u)
	}
	v, err := ctx.JobOutputPlane(0, FrameWorkPlaneV)
	if err != nil {
		t.Fatal(err)
	}
	if v.X != 0 || v.Y != 0 || v.Width != 17 || v.Height != 17 ||
		v.ReadWidth != 20 || v.ReadHeight != 20 ||
		v.ClipWidth != 32 || v.ClipHeight != 32 || v.ClipRowBytes != 32 {
		t.Fatalf("V plane region=%+v", v)
	}
	// The clip helper accepts past-visible writes up to the superblock-aligned
	// writable extent (ClipWidth=64): a 4x4 luma block at MI col 8 (luma pixel
	// 32) and MI col 9 (luma pixel 36) of a 34-wide frame both return the full
	// 4x4, and so does a block deeper in the SB padding (col 56).
	if w, h, ok := frameWorkClipVisiblePixelsToWindow(y, 32, 0, 4, 4); !ok || w != 4 || h != 4 {
		t.Fatalf("luma MI col 8 4x4 clip=(%d,%d,%v) want (4,4,true)", w, h, ok)
	}
	if w, h, ok := frameWorkClipVisiblePixelsToWindow(y, 36, 0, 4, 4); !ok || w != 4 || h != 4 {
		t.Fatalf("luma MI col 9 4x4 clip=(%d,%d,%v) want (4,4,true)", w, h, ok)
	}
	if w, h, ok := frameWorkClipVisiblePixelsToWindow(y, 56, 0, 4, 4); !ok || w != 4 || h != 4 {
		t.Fatalf("luma SB-padding 4x4 clip=(%d,%d,%v) want (4,4,true)", w, h, ok)
	}
	// A block past the superblock-aligned writable extent (64) still fails.
	if _, _, ok := frameWorkClipVisiblePixelsToWindow(y, 64, 0, 4, 4); ok {
		t.Fatalf("luma at SB-aligned boundary should reject")
	}
	// Chroma equivalents: blocks inside the aligned chroma extent (ClipWidth=32)
	// succeed; one at chroma pixel 30 clips its width to the writable edge.
	if w, h, ok := frameWorkClipVisiblePixelsToWindow(u, 16, 0, 4, 4); !ok || w != 4 || h != 4 {
		t.Fatalf("chroma MI col 8 4x4 clip=(%d,%d,%v) want (4,4,true)", w, h, ok)
	}
	if w, h, ok := frameWorkClipVisiblePixelsToWindow(u, 30, 0, 4, 4); !ok || w != 2 || h != 4 {
		t.Fatalf("chroma near edge 4x4 clip=(%d,%d,%v) want (2,4,true)", w, h, ok)
	}
}

// TestFrameWorkBatchPredictBlockLumaIntraWritesPastVisibleRightEdge pins the
// MI-aligned past-visible write behaviour locked in by a86e729 on a 34x34
// frame. An 8x8 DC intra block whose MI footprint covers luma columns 32..39
// (block at MI col 8) has only the leftmost two columns (32 and 33) inside
// the coded-frame edge; the remaining six columns (34..39) are past visible
// but inside the MI-aligned writable extent (mi_cols*MI_SIZE = 40).
//
// libaom writes the entire 8x8 transform regardless of where the visible
// boundary lands; later blocks read the past-visible samples as predictor
// neighbors. goav1 mirrors that by extending JobOutputPlane's window through
// the stride padding so the prediction kernel can write all 8 columns. This
// regression test confirms that the prediction call:
//
//  1. Succeeds at the right-edge MI position (cols 32..39 are inside ClipWidth=40).
//  2. Writes a uniform DC value across columns 32..39 of each row, including
//     the past-visible columns 34..39 (which previously stayed zero-clipped
//     under the visible-only write path and caused downstream blocks to see
//     uninitialised stride padding when fetching above neighbors at row 7+).
//
// The frame is allocated with Align=32 (matching the libaom oracle harness's
// frameFormatFromEvent stride alignment), so Y stride = 64 bytes, leaving 30
// bytes of past-visible stride padding per row that the write fills.
func TestFrameWorkBatchPredictBlockLumaIntraWritesPastVisibleRightEdge(t *testing.T) {
	output := testBatchFrame(t, frame.Format{
		Width:        34,
		Height:       34,
		BitDepth:     8,
		SubsamplingX: true,
		SubsamplingY: true,
		Align:        32,
	})
	testFillFrame(output, 0)

	// Seed the left-neighbor column at x=31 with a fixed value so the
	// 8x8 DC block at (col 32, row 0) has predictable Above (missing,
	// filled from dst[31, 0]) and Left (col 31 rows 0..7) neighbors.
	// DC = (8*seed + 8*seed) / 16 = seed.
	const seed uint16 = 96
	for y := range 8 {
		setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, 31, y, seed)
	}

	ctx := testIntraPredictionBatch(output)
	ctx.Sequence = FrameWorkSequenceContextFromHeader(parser.SequenceHeader{
		ColorConfig: parser.ColorConfig{
			BitDepth:     output.Format.BitDepth,
			MonoChrome:   false,
			SubsamplingX: output.Format.SubsamplingX,
			SubsamplingY: output.Format.SubsamplingY,
		},
	})
	ctx.FrameSize = parser.FrameSize{CodedWidth: 34, Height: 34}
	ctx.Jobs = []tile.Job{{SBCols: 1, SBRows: 1}}

	visit := tile.BlockLoopVisit{
		Block: tile.BlockVisit{
			MICol: 8, MIRow: 0, MIColEnd: 10, MIRowEnd: 2,
			X4: 0, Y4: 0, Size: tile.BlockSize8x8, VisibleW4: 2, VisibleH4: 2,
			HaveTop: false, HaveLeft: true,
		},
		Prediction: tile.BlockPredictionModeResult{
			Valid:    true,
			Intra:    true,
			LumaMode: tile.IntraModeDC,
		},
	}
	var scratch FrameWorkIntraPredictionScratch
	if err := ctx.PredictBlockLumaIntra(0, visit, &scratch); err != nil {
		t.Fatalf("PredictBlockLumaIntra at right-edge MI col 8: %v", err)
	}

	// All 8 rows x 8 cols of the block (cols 32..39, rows 0..7) must hold
	// the DC value, including past-visible cols 34..39. A regression that
	// re-clips writes to the visible 2-col strip would leave cols 34..39
	// at their initial 0, breaking downstream blocks that read row 7
	// cols 32..39 as above neighbors.
	//
	// Read raw bytes by stride (not via frameWorkLoadSample which bounds-
	// checks against plane.Width=34) so we can verify past-visible writes
	// in the stride padding.
	for y := range 8 {
		row := y * output.Y.Stride
		for x := 32; x < 40; x++ {
			got := uint16(output.Y.Pix[row+x])
			if got != seed {
				t.Fatalf("Y(%d,%d)=%d want %d (past-visible write missing or wrong)", x, y, got, seed)
			}
		}
	}
}

// TestFrameWorkPlaneBlockStartsBeyondOutput covers the rounded-up MI grid
// short-circuit. libaom reconstructs the bottom/right partial superblock into
// the MI-aligned padding of the YV12 buffer, so a 34x34 frame (allocated at the
// MI-aligned 40x40, stride-padded to 64 by Align=64) must reconstruct blocks
// addressed past the cropped 34x34 edge but inside that allocation. The helper
// therefore reports "beyond output" only for origins past the allocated buffer
// extent (Stride/BytesPerSample cols, len(Pix)/Stride rows), not past the
// cropped Width/Height. Negative coordinates are rejected so callers hit the
// invalid-batch path instead.
func TestFrameWorkPlaneBlockStartsBeyondOutput(t *testing.T) {
	// 34x34 4:2:0, Align 64: the buffer is superblock-aligned, so luma alloc =
	// 64 rows of stride 64 (64 samples/row); chroma alloc = 32 rows of stride
	// 32. Luma origins with x<64 and y<64 are within the allocation; chroma with
	// x<32 and y<32 likewise.
	output := testBatchFrame(t, frame.Format{Width: 34, Height: 34, BitDepth: 8, Align: 64, SubsamplingX: true, SubsamplingY: true})
	tests := []struct {
		name  string
		plane FrameWorkPlane
		x, y  int
		want  bool
	}{
		{"luma origin", FrameWorkPlaneY, 0, 0, false},
		{"luma last visible (33, 33)", FrameWorkPlaneY, 33, 33, false},
		{"luma right padding (34, 0) within alloc", FrameWorkPlaneY, 34, 0, false},
		{"luma bottom padding (0, 34) within alloc", FrameWorkPlaneY, 0, 34, false},
		{"luma MI col 9 (36, 8) within alloc", FrameWorkPlaneY, 36, 8, false},
		{"luma MI row 9 (8, 36) within alloc", FrameWorkPlaneY, 8, 36, false},
		{"luma SB padding (0, 40) within alloc", FrameWorkPlaneY, 0, 40, false},
		{"luma past alloc height (0, 64)", FrameWorkPlaneY, 0, 64, true},
		{"luma past alloc width (64, 0)", FrameWorkPlaneY, 64, 0, true},
		{"luma negative x rejected (not beyond)", FrameWorkPlaneY, -1, 0, false},
		{"luma negative y rejected (not beyond)", FrameWorkPlaneY, 0, -1, false},
		{"chroma U origin", FrameWorkPlaneU, 0, 0, false},
		{"chroma U last visible (16, 16)", FrameWorkPlaneU, 16, 16, false},
		{"chroma U right padding (17, 0) within alloc", FrameWorkPlaneU, 17, 0, false},
		{"chroma U MI col 9 (18, 4) within alloc", FrameWorkPlaneU, 18, 4, false},
		{"chroma U SB padding (0, 20) within alloc", FrameWorkPlaneU, 0, 20, false},
		{"chroma U past alloc height (0, 32)", FrameWorkPlaneU, 0, 32, true},
		{"chroma V last visible (16, 16)", FrameWorkPlaneV, 16, 16, false},
		{"chroma V bottom padding (8, 17) within alloc", FrameWorkPlaneV, 8, 17, false},
		{"chroma V past alloc height (8, 32)", FrameWorkPlaneV, 8, 32, true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := frameWorkPlaneBlockStartsBeyondOutput(output, tc.plane, tc.x, tc.y); got != tc.want {
				t.Fatalf("plane=%d (%d,%d) = %v want %v", tc.plane, tc.x, tc.y, got, tc.want)
			}
		})
	}
	if got := frameWorkPlaneBlockStartsBeyondOutput(nil, FrameWorkPlaneY, 0, 0); got != false {
		t.Fatalf("nil output: got=%v want false", got)
	}
}

func TestFrameWorkBatchPredictBlockLumaDirectionalKnownVectors(t *testing.T) {
	tests := []struct {
		name string
		mode tile.IntraMode
		seed func(*frame.Frame)
		want []uint16
	}{
		{
			name: "d45",
			mode: tile.IntraModeD45,
			seed: func(output *frame.Frame) {
				for i, sample := range []uint16{10, 20, 30, 40, 50, 60, 70, 80} {
					setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, 16+i, 15, sample)
				}
			},
			want: []uint16{
				20, 30, 40, 50,
				30, 40, 50, 60,
				40, 50, 60, 70,
				50, 60, 70, 80,
			},
		},
		{
			name: "d135",
			mode: tile.IntraModeD135,
			seed: func(output *frame.Frame) {
				for i, sample := range []uint16{21, 31, 41, 51} {
					setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, 16+i, 15, sample)
				}
				for i, sample := range []uint16{101, 111, 121, 131} {
					setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, 15, 16+i, sample)
				}
				setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, 15, 15, 11)
			},
			want: []uint16{
				11, 21, 31, 41,
				101, 11, 21, 31,
				111, 101, 11, 21,
				121, 111, 101, 11,
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, Align: 64})
			ctx := testIntraPredictionBatch(output)
			tt.seed(output)

			var scratch FrameWorkIntraPredictionScratch
			if err := ctx.PredictBlockLumaIntra(0, testIntraPrediction4x4Visit(tt.mode), &scratch); err != nil {
				t.Fatal(err)
			}
			got := make([]uint16, 0, 16)
			for y := 16; y < 20; y++ {
				for x := 16; x < 20; x++ {
					got = append(got, frameWorkTestSample(output.Y, output.Layout.BytesPerSample, x, y))
				}
			}
			for i := range got {
				if got[i] != tt.want[i] {
					t.Fatalf("prediction=%v want %v", got, tt.want)
				}
			}
		})
	}
}

func TestFrameWorkBatchPredictBlockLumaDirectionalModes(t *testing.T) {
	for _, tt := range []struct {
		name string
		mode tile.IntraMode
	}{
		{name: "d45", mode: tile.IntraModeD45},
		{name: "d67", mode: tile.IntraModeD67},
		{name: "d113", mode: tile.IntraModeD113},
		{name: "d135", mode: tile.IntraModeD135},
		{name: "d157", mode: tile.IntraModeD157},
		{name: "d203", mode: tile.IntraModeD203},
	} {
		t.Run(tt.name, func(t *testing.T) {
			output := testBatchFrame(t, frame.Format{Width: 96, Height: 96, BitDepth: 10, Align: 128})
			ctx := testIntraPredictionBatch(output)
			seedFrameWorkDirectionalEdges(output, 0x3ff, 19, 37, 101)

			var scratch FrameWorkIntraPredictionScratch
			if err := ctx.PredictBlockLumaIntra(0, testIntraPredictionVisit(tt.mode), &scratch); err != nil {
				t.Fatal(err)
			}
			for y := 16; y < 32; y++ {
				for x := 16; x < 32; x++ {
					if got := frameWorkTestSample(output.Y, output.Layout.BytesPerSample, x, y); got > 0x3ff {
						t.Fatalf("sample(%d,%d)=%d exceeds 10-bit max", x, y, got)
					}
				}
			}
		})
	}
}

func TestFrameWorkBatchPredictBlockLumaDirectionalIntraEdgeUpsample(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, Align: 64})
	ctx := testIntraPredictionBatch(output)
	ctx.Sequence.EnableIntraEdgeFilter = true
	leftSamples := []uint16{97, 100, 103, 107, 109, 111, 112, 113}
	for y, sample := range leftSamples {
		setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, 15, y, sample)
	}
	visit := testIntraPrediction4x4Visit(tile.IntraModeD203)
	visit.Block.MIRow = 0
	visit.Block.MIRowEnd = 1
	visit.Block.Y4 = 0
	visit.Block.HaveTop = false
	visit.Block.HaveLeft = true

	left := make([]uint16, frameWorkDirectionalEdgeSamples)
	origin := frameWorkDirectionalEdgeOrigin
	left[origin-1] = leftSamples[0]
	copy(left[origin:], leftSamples)
	scratch := make([]uint16, frameWorkIntraEdgeScratchSamples)
	if err := prediction.UpsampleIntraEdgeBounded(left, origin, 8, scratch, 8, false); err != nil {
		t.Fatal(err)
	}
	wantPix := make([]byte, 16)
	want := frame.Plane{Pix: wantPix, Stride: 4, Width: 4, Height: 4}
	edges := prediction.DirectionalEdges{
		Left:         left,
		LeftOrigin:   origin,
		UpsampleLeft: true,
	}
	if err := prediction.PredictDirectionalIntraPlaneBlock(want, 1, 8, 0, 0, 4, 4, 203, edges); err != nil {
		t.Fatal(err)
	}

	var predScratch FrameWorkIntraPredictionScratch
	if err := ctx.PredictBlockLumaIntra(0, visit, &predScratch); err != nil {
		t.Fatal(err)
	}
	for y := range 4 {
		for x := range 4 {
			got := frameWorkTestSample(output.Y, output.Layout.BytesPerSample, 16+x, y)
			if want := uint16(wantPix[y*4+x]); got != want {
				t.Fatalf("sample(%d,%d)=%d want %d", 16+x, y, got, want)
			}
		}
	}
}

func TestFrameWorkLumaTransformDirectionalExtendedEdgesMatchesLibaomCases(t *testing.T) {
	tests := []struct {
		name           string
		block          tile.BlockVisit
		miColEnd       uint32
		miRowEnd       uint32
		absX           int
		absY           int
		width          int
		height         int
		wantTopRight   bool
		wantBottomLeft bool
	}{
		{
			name: "d203 right-half 8x4 cannot use bottom-left",
			block: tile.BlockVisit{
				MICol: 56, MIRow: 0, MIColEnd: 58, MIRowEnd: 1,
				Size: tile.BlockSize8x4, VisibleW4: 2, VisibleH4: 1, HaveLeft: true,
			},
			absX: 228, absY: 0, width: 4, height: 4,
		},
		{
			name: "d203 left-half 8x4 table allows bottom-left",
			block: tile.BlockVisit{
				MICol: 56, MIRow: 0, MIColEnd: 58, MIRowEnd: 1,
				Size: tile.BlockSize8x4, VisibleW4: 2, VisibleH4: 1, HaveLeft: true,
			},
			absX: 224, absY: 0, width: 4, height: 4, wantBottomLeft: true,
		},
		{
			name: "d203 4x8 table denies bottom-left",
			block: tile.BlockVisit{
				MICol: 66, MIRow: 0, MIColEnd: 67, MIRowEnd: 2,
				Size: tile.BlockSize4x8, VisibleW4: 1, VisibleH4: 2, HaveLeft: true,
			},
			absX: 264, absY: 4, width: 4, height: 4,
		},
		{
			name: "d45 right-half 8x8 cannot use top-right",
			block: tile.BlockVisit{
				MICol: 74, MIRow: 2, MIColEnd: 76, MIRowEnd: 4,
				Size: tile.BlockSize8x8, VisibleW4: 2, VisibleH4: 2, HaveTop: true, HaveLeft: true,
			},
			absX: 300, absY: 8, width: 4, height: 4,
		},
		{
			name: "d45 left-half 8x8 has internal top-right",
			block: tile.BlockVisit{
				MICol: 74, MIRow: 2, MIColEnd: 76, MIRowEnd: 4,
				Size: tile.BlockSize8x8, VisibleW4: 2, VisibleH4: 2, HaveTop: true, HaveLeft: true,
			},
			absX: 296, absY: 8, width: 4, height: 4, wantTopRight: true, wantBottomLeft: true,
		},
		{
			name: "quantizer00 d67 4x4 table allows top-right",
			block: tile.BlockVisit{
				MICol: 8, MIRow: 2, MIColEnd: 9, MIRowEnd: 3,
				Partition: tile.PartitionSplit, Size: tile.BlockSize4x4,
				VisibleW4: 1, VisibleH4: 1, HaveTop: true, HaveLeft: true,
			},
			miColEnd: 88, miRowEnd: 72,
			absX: 32, absY: 8, width: 4, height: 4, wantTopRight: true, wantBottomLeft: true,
		},
		{
			name: "quantizer00 d67 4x4 denies top-right at tile edge",
			block: tile.BlockVisit{
				MICol: 8, MIRow: 2, MIColEnd: 9, MIRowEnd: 3,
				Partition: tile.PartitionSplit, Size: tile.BlockSize4x4,
				VisibleW4: 1, VisibleH4: 1, HaveTop: true, HaveLeft: true,
			},
			miColEnd: 9, miRowEnd: 72,
			absX: 32, absY: 8, width: 4, height: 4, wantBottomLeft: true,
		},
		{
			name: "quantizer00 internal transform top edge allows top-right",
			block: func() tile.BlockVisit {
				parent := tile.BlockVisit{
					MICol: 24, MIRow: 0, MIColEnd: 28, MIRowEnd: 2,
					X4: 24, Y4: 0, Partition: tile.PartitionTBottomSplit, Size: tile.BlockSize16x8,
					VisibleW4: 4, VisibleH4: 2, HaveLeft: true,
				}
				return frameWorkPredictionTransformEdgeBlock(parent, int(parent.X4), int(parent.Y4), 24, 1)
			}(),
			miColEnd: 88, miRowEnd: 72,
			absX: 96, absY: 4, width: 4, height: 4, wantTopRight: true, wantBottomLeft: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			miColEnd := tt.miColEnd
			if miColEnd == 0 {
				miColEnd = 88
			}
			miRowEnd := tt.miRowEnd
			if miRowEnd == 0 {
				miRowEnd = 72
			}
			gotTopRight, gotBottomLeft := frameWorkLumaDirectionalExtendedEdges(tt.block, 32, miColEnd, miRowEnd, tt.absX, tt.absY, tt.width, tt.height)
			if gotTopRight != tt.wantTopRight || gotBottomLeft != tt.wantBottomLeft {
				t.Fatalf("topRight=%v bottomLeft=%v want %v/%v", gotTopRight, gotBottomLeft, tt.wantTopRight, tt.wantBottomLeft)
			}
		})
	}
}

func TestFrameWorkDirectionalAvailabilityTablesCoverBlockSizes(t *testing.T) {
	sizes := []tile.BlockSize{
		tile.BlockSize4x4, tile.BlockSize4x8, tile.BlockSize8x4, tile.BlockSize8x8,
		tile.BlockSize8x16, tile.BlockSize16x8, tile.BlockSize16x16,
		tile.BlockSize16x32, tile.BlockSize32x16, tile.BlockSize32x32,
		tile.BlockSize32x64, tile.BlockSize64x32, tile.BlockSize64x64,
		tile.BlockSize64x128, tile.BlockSize128x64, tile.BlockSize128x128,
		tile.BlockSize4x16, tile.BlockSize16x4, tile.BlockSize8x32,
		tile.BlockSize32x8, tile.BlockSize16x64, tile.BlockSize64x16,
	}
	for _, size := range sizes {
		if table := frameWorkTopRightAvailabilityTable(tile.PartitionNone, size); len(table) == 0 {
			t.Fatalf("missing top-right table for size %d", size)
		}
		if table := frameWorkBottomLeftAvailabilityTable(tile.PartitionNone, size); len(table) == 0 {
			t.Fatalf("missing bottom-left table for size %d", size)
		}
	}
	if table := frameWorkTopRightAvailabilityTable(tile.PartitionTLeftSplit, tile.BlockSize8x8); len(table) == 0 || table[2] != 0 {
		t.Fatalf("mixed-vertical top-right table not selected: %v", table)
	}
	if table := frameWorkBottomLeftAvailabilityTable(tile.PartitionTRightSplit, tile.BlockSize8x8); len(table) == 0 || table[0] != 254 {
		t.Fatalf("mixed-vertical bottom-left table not selected: %v", table)
	}
}

func TestFrameWorkFillDirectionalAboveCapsTopRightToPrimaryWidth(t *testing.T) {
	// libaom's intra prediction loads at most primaryWidth additional top-right
	// samples (n_topright_px = min(txwpx, xr)) and extends the rest from the last
	// real sample. The fill helper must mirror that even when allow_top_right is
	// true so directional predictors see the same edge buffer libaom does.
	const W, H = 64, 64
	format := frame.Format{Width: W, Height: H, BitDepth: 8, Align: 64}
	output := testBatchFrame(t, format)
	for x := range W {
		// Distinct per-column markers above the block being predicted.
		setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, x, 15, uint16(x+1))
	}
	dst := frame.Plane{Pix: output.Y.Pix, Stride: output.Y.Stride, Width: W, Height: H}
	var scratch FrameWorkIntraPredictionScratch
	block := tile.BlockVisit{
		Size:    tile.BlockSize4x8,
		HaveTop: true,
	}
	const primaryWidth = 4
	if err := frameWorkFillDirectionalAbove(dst, 1, 8, 16, 16, 0, primaryWidth+8-1, primaryWidth, true, block, &scratch, 0); err != nil {
		t.Fatalf("fill: %v", err)
	}
	// Indices 0..3 read columns 16..19 verbatim.
	want := []uint16{17, 18, 19, 20}
	for i, v := range want {
		if got := scratch.Above[frameWorkDirectionalEdgeOrigin+i]; got != v {
			t.Fatalf("primary above[%d]=%d want %d", i, got, v)
		}
	}
	// Indices 4..7 are the top-right extension (n_topright_px = primaryWidth).
	for i := 4; i < 8; i++ {
		if got := scratch.Above[frameWorkDirectionalEdgeOrigin+i]; got != uint16(16+i+1) {
			t.Fatalf("topright above[%d]=%d want %d", i, got, 16+i+1)
		}
	}
	// Indices 8..10 must extend from the last real sample at column 16+7=23, not
	// pull additional decoded samples from columns 24+.
	last := uint16(16 + 7 + 1)
	for i := 8; i < primaryWidth+8-1; i++ {
		if got := scratch.Above[frameWorkDirectionalEdgeOrigin+i]; got != last {
			t.Fatalf("extension above[%d]=%d want %d", i, got, last)
		}
	}
}

// TestFrameWorkFillDirectionalAboveTruncatesToVisibleRightEdge pins the
// behavior of frameWorkFillDirectionalAbove when a block straddles the
// visible right edge. libaom's intra predictor caps the real-sample range
// to n_top_px = min(txwpx, xr + txwpx); past-visible columns are replicated
// from the last visible neighbor sample, never read from MI-padding bytes
// (which carry past-visible writes from earlier blocks that libaom never
// sees). The 34x34 libaom-extended vector exposed this divergence: an 8x8
// transform at x=32 in a 34-wide plane had n_top_px=2 in libaom but goav1
// previously loaded all 8 cols from MI-padding, yielding a different above
// row and a downstream MD5 mismatch.
func TestFrameWorkFillDirectionalAboveTruncatesToVisibleRightEdge(t *testing.T) {
	// Allocate a 40-wide buffer so we can seed both the visible cols (0..33)
	// and the MI-padding cols (34..39) with distinct values, then verify the
	// fill ignores the MI-padding cols when visibleW=34.
	const Stride, H = 40, 16
	const visibleW = 34
	pix := make([]byte, Stride*H)
	dst := frame.Plane{Pix: pix, Stride: Stride, Width: Stride, Height: H}
	// Above row (y=7): distinct per-column markers so the fill output records
	// exactly which cols were loaded. Marker(c) = c + 1.
	for c := range Stride {
		pix[7*Stride+c] = byte(c + 1)
	}
	var scratch FrameWorkIntraPredictionScratch
	block := tile.BlockVisit{
		Size:    tile.BlockSize8x8,
		HaveTop: true,
	}
	const x = 32
	const primaryWidth = 8
	// Range covers primary (0..primaryWidth-1) and top-right extension
	// (primaryWidth..primaryWidth+primaryWidth-1) per libaom's directional
	// above buffer layout.
	maxIndex := primaryWidth + primaryWidth - 1
	if err := frameWorkFillDirectionalAbove(dst, 1, 8, x, 8, 0, maxIndex, primaryWidth, true, block, &scratch, visibleW); err != nil {
		t.Fatalf("fill: %v", err)
	}
	// libaom: n_top_px = min(8, (34 - 32 - 8) + 8) = min(8, 2) = 2.
	//         n_topright_px = min(8, 34 - 32 - 8) = min(8, -6) -> 0.
	// Real samples land at cols 32, 33 (markers 33, 34); the remaining
	// primary slots (i=2..7) and the entire top-right extension
	// (i=8..15) must replicate col 33's marker (=34).
	want := []uint16{33, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34}
	for i, w := range want {
		got := scratch.Above[frameWorkDirectionalEdgeOrigin+i]
		if got != w {
			t.Fatalf("above[%d]=%d want %d (visibleW=%d should cap real loads to cols 32..33)", i, got, w, visibleW)
		}
	}
}

// TestFrameWorkIntraPredictionEdgesWithExtentTruncatesToVisibleRightEdge
// pins the equivalent libaom contract for the non-directional edge builder:
// the above buffer copies n_top_px real samples and replicates the last
// real one for the remaining edgeWidth-n_top_px slots. The 34x34 right-edge
// 8x8 DC / Smooth / Paeth block must see Above[0..1] = real cols and
// Above[2..7] = replicated last visible neighbor.
func TestFrameWorkIntraPredictionEdgesWithExtentTruncatesToVisibleRightEdge(t *testing.T) {
	const Stride, H = 40, 16
	const visibleW = 34
	pix := make([]byte, Stride*H)
	dst := frame.Plane{Pix: pix, Stride: Stride, Width: Stride, Height: H}
	for c := range Stride {
		// Distinct marker per column on the above row (y=7) and on the
		// left column (x=31) so we can assert which neighbors were read.
		pix[7*Stride+c] = byte(c + 1)
	}
	for r := range H {
		pix[r*Stride+31] = byte(r + 100)
	}
	var scratch FrameWorkIntraPredictionScratch
	block := tile.BlockVisit{
		Size:     tile.BlockSize8x8,
		HaveTop:  true,
		HaveLeft: true,
	}
	const x, y = 32, 8
	edges, err := frameWorkIntraPredictionEdgesWithExtent(dst, 1, 8, x, y, 8, 8, 8, 8, visibleW, 0, block, &scratch, true)
	if err != nil {
		t.Fatalf("edges: %v", err)
	}
	if !edges.AboveAvailable || len(edges.Above) < 8 {
		t.Fatalf("expected AboveAvailable with len>=8, got %v len=%d", edges.AboveAvailable, len(edges.Above))
	}
	// n_top_px = min(8, (34-32-8)+8) = 2. Above[0] = col 32, Above[1] = col 33,
	// Above[2..7] = replicated col 33 (the last visible).
	wantAbove := []uint16{33, 34, 34, 34, 34, 34, 34, 34}
	for i, w := range wantAbove {
		if edges.Above[i] != w {
			t.Fatalf("Above[%d]=%d want %d", i, edges.Above[i], w)
		}
	}
	// Left is unrestricted by the right-edge cap: 8 real samples from col 31.
	if !edges.LeftAvailable || len(edges.Left) < 8 {
		t.Fatalf("expected LeftAvailable with len>=8, got %v len=%d", edges.LeftAvailable, len(edges.Left))
	}
	for i := range 8 {
		want := uint16(y + i + 100)
		if edges.Left[i] != want {
			t.Fatalf("Left[%d]=%d want %d", i, edges.Left[i], want)
		}
	}
	// AboveLeft = load(x-1, y-1) since n_top_px>0 and n_left_px>0.
	wantAL := uint16((y-1)*1 + 100)
	if edges.AboveLeft != wantAL {
		t.Fatalf("AboveLeft=%d want %d", edges.AboveLeft, wantAL)
	}
}

func TestFrameWorkChromaDirectionalExtendedEdgesMatchesLibaomCases(t *testing.T) {
	tests := []struct {
		name           string
		block          tile.BlockVisit
		originX        int
		originY        int
		absX           int
		absY           int
		width          int
		height         int
		wantTopRight   bool
		wantBottomLeft bool
	}{
		{
			name: "quantizer00 horizontal right-half denies bottom-left",
			block: func() tile.BlockVisit {
				parent := tile.BlockVisit{
					MICol: 4, MIRow: 0, MIColEnd: 8, MIRowEnd: 2,
					X4: 4, Y4: 0, Partition: tile.PartitionH, Size: tile.BlockSize16x8,
					VisibleW4: 4, VisibleH4: 2, HaveLeft: true,
				}
				return frameWorkPredictionTransformEdgeBlock(parent, int(parent.X4)>>1, int(parent.Y4)>>1, 3, 0)
			}(),
			originX: 8, originY: 0,
			absX: 12, absY: 0, width: 4, height: 4,
		},
		{
			name: "quantizer00 dc next block has left-edge bottom-left",
			block: tile.BlockVisit{
				MICol: 8, MIRow: 1, MIColEnd: 10, MIRowEnd: 2,
				X4: 8, Y4: 1, Partition: tile.PartitionH, Size: tile.BlockSize8x4,
				VisibleW4: 2, VisibleH4: 1, HaveTop: true, HaveLeft: true,
			},
			originX: 16, originY: 0,
			absX: 16, absY: 0, width: 4, height: 4,
			wantTopRight: true, wantBottomLeft: true,
		},
		{
			name: "small 4x4 420 scales to 8x8 for table lookup",
			block: tile.BlockVisit{
				MICol: 8, MIRow: 2, MIColEnd: 9, MIRowEnd: 3,
				Partition: tile.PartitionSplit, Size: tile.BlockSize4x4,
				VisibleW4: 1, VisibleH4: 1, HaveTop: true, HaveLeft: true,
			},
			originX: 16, originY: 4,
			absX: 16, absY: 4, width: 4, height: 4,
			wantTopRight: true, wantBottomLeft: true,
		},
		{
			name: "rightmost 420 chroma tx denies top-right at tile edge",
			block: tile.BlockVisit{
				MICol: 84, MIRow: 8, MIColEnd: 88, MIRowEnd: 12,
				Partition: tile.PartitionNone, Size: tile.BlockSize16x16,
				VisibleW4: 4, VisibleH4: 4, HaveTop: true, HaveLeft: true,
			},
			originX: 168, originY: 16,
			absX: 172, absY: 16, width: 4, height: 4,
		},
		{
			name: "bottom 420 chroma tx denies bottom-left at tile edge",
			block: tile.BlockVisit{
				MICol: 8, MIRow: 68, MIColEnd: 12, MIRowEnd: 72,
				Partition: tile.PartitionNone, Size: tile.BlockSize16x16,
				VisibleW4: 4, VisibleH4: 4, HaveTop: true, HaveLeft: true,
			},
			originX: 16, originY: 136,
			absX: 16, absY: 140, width: 4, height: 4,
			wantTopRight: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotTopRight, gotBottomLeft := frameWorkChromaDirectionalExtendedEdges(tt.block, 32, 88, 72, tt.originX, tt.originY, tt.absX, tt.absY, tt.width, tt.height, true, true)
			if gotTopRight != tt.wantTopRight || gotBottomLeft != tt.wantBottomLeft {
				t.Fatalf("topRight=%v bottomLeft=%v want %v/%v", gotTopRight, gotBottomLeft, tt.wantTopRight, tt.wantBottomLeft)
			}
		})
	}
}

func TestFrameWorkChromaAvailabilityBlockSizeMatchesLibaom(t *testing.T) {
	tests := []struct {
		name         string
		size         tile.BlockSize
		subsamplingX bool
		subsamplingY bool
		want         tile.BlockSize
	}{
		{name: "4x4 420", size: tile.BlockSize4x4, subsamplingX: true, subsamplingY: true, want: tile.BlockSize8x8},
		{name: "4x4 422", size: tile.BlockSize4x4, subsamplingX: true, want: tile.BlockSize8x4},
		{name: "4x4 440", size: tile.BlockSize4x4, subsamplingY: true, want: tile.BlockSize4x8},
		{name: "4x8 422", size: tile.BlockSize4x8, subsamplingX: true, want: tile.BlockSize8x8},
		{name: "8x4 440", size: tile.BlockSize8x4, subsamplingY: true, want: tile.BlockSize8x8},
		{name: "4x16 420", size: tile.BlockSize4x16, subsamplingX: true, subsamplingY: true, want: tile.BlockSize8x16},
		{name: "16x4 420", size: tile.BlockSize16x4, subsamplingX: true, subsamplingY: true, want: tile.BlockSize16x8},
		{name: "16x16 420 unchanged", size: tile.BlockSize16x16, subsamplingX: true, subsamplingY: true, want: tile.BlockSize16x16},
		{name: "16x8 420 unchanged", size: tile.BlockSize16x8, subsamplingX: true, subsamplingY: true, want: tile.BlockSize16x8},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := frameWorkChromaAvailabilityBlockSize(tt.size, tt.subsamplingX, tt.subsamplingY); got != tt.want {
				t.Fatalf("size=%d want %d", got, tt.want)
			}
		})
	}
	got := frameWorkChromaAvailabilityBlockSize(tile.BlockSize16x16, true, true)
	plane, err := tile.PlaneBlockSize(tile.BlockSize16x16, parser.ColorConfig{SubsamplingX: true, SubsamplingY: true}, 1)
	if err != nil {
		t.Fatal(err)
	}
	if got == plane {
		t.Fatalf("availability size used residual plane size %d", plane)
	}
}

func TestFrameWorkBatchPredictBlockLumaIntraBoundaryEdges(t *testing.T) {
	tests := []struct {
		name  string
		mode  tile.IntraMode
		seed  func(*frame.Frame)
		visit func() tile.BlockLoopVisit
		want  uint16
	}{
		{
			name: "vertical-missing-top-uses-left",
			mode: tile.IntraModeVertical,
			seed: func(output *frame.Frame) {
				for y := range 16 {
					setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, 15, y, 70)
				}
			},
			visit: func() tile.BlockLoopVisit {
				visit := testIntraPredictionVisit(tile.IntraModeVertical)
				visit.Block.MIRow = 0
				visit.Block.MIRowEnd = 4
				visit.Block.Y4 = 0
				visit.Block.HaveTop = false
				return visit
			},
			want: 70,
		},
		{
			name: "directional-missing-top-uses-left",
			mode: tile.IntraModeD45,
			seed: func(output *frame.Frame) {
				for y := range 16 {
					setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, 15, y, 81)
				}
			},
			visit: func() tile.BlockLoopVisit {
				visit := testIntraPredictionVisit(tile.IntraModeD45)
				visit.Block.MIRow = 0
				visit.Block.MIRowEnd = 4
				visit.Block.Y4 = 0
				visit.Block.HaveTop = false
				return visit
			},
			want: 81,
		},
		{
			name: "horizontal-missing-left-uses-top",
			mode: tile.IntraModeHorizontal,
			seed: func(output *frame.Frame) {
				for x := range 16 {
					setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, x, 15, 93)
				}
			},
			visit: func() tile.BlockLoopVisit {
				visit := testIntraPredictionVisit(tile.IntraModeHorizontal)
				visit.Block.MICol = 0
				visit.Block.MIColEnd = 4
				visit.Block.X4 = 0
				visit.Block.HaveLeft = false
				return visit
			},
			want: 93,
		},
		{
			name: "vertical-missing-both-uses-mid-minus-one",
			mode: tile.IntraModeVertical,
			seed: func(*frame.Frame) {},
			visit: func() tile.BlockLoopVisit {
				visit := testIntraPredictionVisit(tile.IntraModeVertical)
				visit.Block.MICol = 0
				visit.Block.MIColEnd = 4
				visit.Block.MIRow = 0
				visit.Block.MIRowEnd = 4
				visit.Block.X4 = 0
				visit.Block.Y4 = 0
				visit.Block.HaveTop = false
				visit.Block.HaveLeft = false
				return visit
			},
			want: 127,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, Align: 64})
			ctx := testIntraPredictionBatch(output)
			tt.seed(output)
			visit := tt.visit()

			var scratch FrameWorkIntraPredictionScratch
			if err := ctx.PredictBlockLumaIntra(0, visit, &scratch); err != nil {
				t.Fatal(err)
			}
			x, y, err := frameWorkBlockLumaPosition(visit.Block)
			if err != nil {
				t.Fatal(err)
			}
			for row := range 16 {
				for col := range 16 {
					if got := frameWorkTestSample(output.Y, output.Layout.BytesPerSample, x+col, y+row); got != tt.want {
						t.Fatalf("sample(%d,%d)=%d want %d", x+col, y+row, got, tt.want)
					}
				}
			}
		})
	}
}

func TestFrameWorkBatchPredictBlockLumaInterFullpel(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, Align: 64})
	reference := testBatchFrame(t, output.Format)
	fillFrameWorkInterReference(reference, 0xff)
	ctx := testInterPredictionBatch(output, reference)
	visit := testInterPredictionVisit(motion.Vector{Col: 8, Row: -8})
	if err := ctx.PredictBlockLumaInter(0, visit); err != nil {
		t.Fatal(err)
	}
	for y := 16; y < 32; y++ {
		for x := 16; x < 32; x++ {
			got := frameWorkTestSample(output.Y, output.Layout.BytesPerSample, x, y)
			want := frameWorkTestSample(reference.Y, reference.Layout.BytesPerSample, x+1, y-1)
			if got != want {
				t.Fatalf("sample(%d,%d)=%d want %d", x, y, got, want)
			}
		}
	}
}

func TestFrameWorkBatchPredictBlockInterFullpelExtendsReferenceEdges(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 64})
	reference := testBatchFrame(t, output.Format)
	fillFrameWorkInterReferenceAllPlanes(reference, 0xff)
	ctx := testInterPredictionBatch(output, reference)

	visit := testInterPredictionVisit(motion.Vector{Col: -16, Row: 16})
	visit.Block = tile.BlockVisit{
		MICol: 0, MIRow: 0, MIColEnd: 4, MIRowEnd: 4,
		X4: 0, Y4: 0, Size: tile.BlockSize16x16, VisibleW4: 4, VisibleH4: 4,
	}
	if err := ctx.PredictBlockInter(0, visit, nil); err != nil {
		t.Fatal(err)
	}

	for y := range 16 {
		for x := range 16 {
			got := frameWorkTestSample(output.Y, output.Layout.BytesPerSample, x, y)
			want := frameWorkTestSampleClamped(reference.Y, output.Layout.BytesPerSample, x-2, y+2)
			if got != want {
				t.Fatalf("y sample(%d,%d)=%d want %d", x, y, got, want)
			}
		}
	}
	for y := range 8 {
		for x := range 8 {
			wantU := frameWorkTestSampleClamped(reference.U, output.Layout.BytesPerSample, x-1, y+1)
			if got := frameWorkTestSample(output.U, output.Layout.BytesPerSample, x, y); got != wantU {
				t.Fatalf("u sample(%d,%d)=%d want %d", x, y, got, wantU)
			}
			wantV := frameWorkTestSampleClamped(reference.V, output.Layout.BytesPerSample, x-1, y+1)
			if got := frameWorkTestSample(output.V, output.Layout.BytesPerSample, x, y); got != wantV {
				t.Fatalf("v sample(%d,%d)=%d want %d", x, y, got, wantV)
			}
		}
	}
}

func TestFrameWorkBatchPredictBlockLumaInterFractionalMatchesMotion(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, Align: 64})
	want := testBatchFrame(t, output.Format)
	reference := testBatchFrame(t, output.Format)
	fillFrameWorkInterReference(reference, 0xff)
	ctx := testInterPredictionBatch(output, reference)
	mv := motion.Vector{Col: 3, Row: 5}
	filters := motion.InterpFilters{X: motion.InterpMultiTapSharp, Y: motion.InterpEightTapSmooth}
	visit := testInterPredictionVisit(mv)
	if err := ctx.PredictBlockLumaInterWithFilters(0, visit, filters); err != nil {
		t.Fatal(err)
	}
	predictTestInterPlaneBlock(t, want.Y, reference.Y, want.Layout.BytesPerSample, want.Format.BitDepth, 16, 16, 16, 16, mv, filters)
	assertFrameWorkPlaneBlockEqual(t, output.Y, want.Y, output.Layout.BytesPerSample, 16, 16, 16, 16)
}

func TestFrameWorkBatchPredictBlockLumaInterInvalidWarpFallsBackToTranslation(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, Align: 64})
	want := testBatchFrame(t, output.Format)
	reference := testBatchFrame(t, output.Format)
	fillFrameWorkInterReference(reference, 0xff)
	ctx := testInterPredictionBatch(output, reference)
	mv := motion.Vector{Col: 3, Row: 5}
	filters := motion.InterpFilters{X: motion.InterpMultiTapSharp, Y: motion.InterpEightTapSmooth}
	visit := testInterPredictionVisit(mv)
	visit.Prediction.MotionModeValid = true
	visit.Prediction.MotionMode = tile.MotionModeWarp
	visit.Prediction.WarpedMotionInvalid = true

	if err := ctx.PredictBlockLumaInterWithFilters(0, visit, filters); err != nil {
		t.Fatal(err)
	}
	predictTestInterPlaneBlock(t, want.Y, reference.Y, want.Layout.BytesPerSample, want.Format.BitDepth, 16, 16, 16, 16, mv, filters)
	assertFrameWorkPlaneBlockEqual(t, output.Y, want.Y, output.Layout.BytesPerSample, 16, 16, 16, 16)
}

func TestFrameWorkBatchPredictBlockLumaInterWarpIdentityConstant(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, Align: 64})
	reference := testBatchFrame(t, output.Format)
	testFillFrame(reference, 0xab)
	ctx := testInterPredictionBatch(output, reference)
	visit := testInterPredictionVisit(motion.Vector{})
	visit.Prediction.MotionModeValid = true
	visit.Prediction.MotionMode = tile.MotionModeWarp
	params := parser.DefaultWarpedMotionParams()
	params.Type = parser.GlobalMotionAffine
	testMotionExtras(&visit.Prediction).WarpedMotion = tile.WarpedMotionModel{Params: params}
	visit.Prediction.WarpedMotionValid = true

	if err := ctx.PredictBlockLumaInterWithFilters(0, visit, motion.RegularFilters); err != nil {
		t.Fatal(err)
	}
	for y := 16; y < 32; y++ {
		for x := 16; x < 32; x++ {
			if got := output.Y.Pix[y*output.Y.Stride+x]; got != 0xab {
				t.Fatalf("sample(%d,%d)=%d want 0xab", x, y, got)
			}
		}
	}
}

func TestFrameWorkBatchPredictBlockInterWarpSmallChromaFallsBackToRegular(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 64})
	reference := testBatchFrame(t, output.Format)
	fillFrameWorkInterReferenceAllPlanes(reference, 0xff)
	ctx := testInterPredictionBatch(output, reference)
	filters := motion.InterpFilters{X: motion.InterpEightTapRegular, Y: motion.InterpEightTapRegular}
	visit := testInterPredictionVisit(motion.Vector{})
	visit.Block.MIColEnd = 6
	visit.Block.MIRowEnd = 6
	visit.Block.Size = tile.BlockSize8x8
	visit.Block.VisibleW4 = 2
	visit.Block.VisibleH4 = 2
	visit.Prediction.MotionModeValid = true
	visit.Prediction.MotionMode = tile.MotionModeWarp
	params := parser.DefaultWarpedMotionParams()
	params.Type = parser.GlobalMotionAffine
	params.Matrix[0] = 1 << 16
	testMotionExtras(&visit.Prediction).WarpedMotion = tile.WarpedMotionModel{Params: params}
	visit.Prediction.WarpedMotionValid = true

	if err := ctx.PredictBlockInterWithFilters(0, visit, nil, filters); err != nil {
		t.Fatal(err)
	}

	wantU := testFrameWorkMotionPredictionPlaneSubsampled(t, reference.U, output.Layout.BytesPerSample, output.Format.BitDepth, 8, 8, 4, 4, motion.Vector{}, true, true, filters)
	wantV := testFrameWorkMotionPredictionPlaneSubsampled(t, reference.V, output.Layout.BytesPerSample, output.Format.BitDepth, 8, 8, 4, 4, motion.Vector{}, true, true, filters)
	assertFrameWorkPlaneBlockEqualAt(t, output.U, 8, 8, wantU, 0, 0, output.Layout.BytesPerSample, 4, 4)
	assertFrameWorkPlaneBlockEqualAt(t, output.V, 8, 8, wantV, 0, 0, output.Layout.BytesPerSample, 4, 4)
}

func TestFrameWorkBatchPredictBlockLumaInterHighBitDepthFractionalMatchesMotion(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 10, Align: 128})
	want := testBatchFrame(t, output.Format)
	reference := testBatchFrame(t, output.Format)
	fillFrameWorkInterReference(reference, 0x3ff)
	ctx := testInterPredictionBatch(output, reference)
	mv := motion.Vector{Col: 4, Row: 6}
	filters := motion.InterpFilters{X: motion.InterpEightTapRegular, Y: motion.InterpEightTapSmooth}
	visit := testInterPredictionVisit(mv)
	if err := ctx.PredictBlockLumaInterWithFilters(0, visit, filters); err != nil {
		t.Fatal(err)
	}
	predictTestInterPlaneBlock(t, want.Y, reference.Y, want.Layout.BytesPerSample, want.Format.BitDepth, 16, 16, 16, 16, mv, filters)
	assertFrameWorkPlaneBlockEqual(t, output.Y, want.Y, output.Layout.BytesPerSample, 16, 16, 16, 16)
}

func TestFrameWorkBatchPredictBlockLumaInterOBMCLeftMatchesLibaomMask(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, Align: 64})
	reference := testBatchFrame(t, output.Format)
	fillFrameWorkInterReferenceVariant(reference, 0xff, 17)
	ctx := testInterPredictionBatch(output, reference)

	visit := testInterPredictionVisit(motion.Vector{})
	visit.Prediction.MotionMode = tile.MotionModeOBMC
	visit.Prediction.MotionModeValid = true
	visit.Prediction.OverlappableNeighborsValid = true
	testMotionExtras(&visit.Prediction).OverlappableNeighbors.LeftCount = 1
	testMotionExtras(&visit.Prediction).OverlappableNeighbors.Left[0] = tile.OverlappableNeighbor{
		RelY4: 0,
		Span4: 4,
		Size:  tile.BlockSize16x16,
		Motion: tile.InterMotionResult{
			References: visit.Prediction.InterMotion.References,
			MV:         [2]motion.Vector{{Col: 8}},
		},
		InterpFilters: motion.RegularFilters,
	}

	var scratch FrameWorkInterPredictionScratch
	if err := ctx.PredictBlockLumaInterOBMCWithFilters(0, visit, &scratch, motion.RegularFilters); err != nil {
		t.Fatal(err)
	}

	mask, ok := frameWorkOBMCMask(8)
	if !ok {
		t.Fatal("missing 8-wide obmc mask")
	}
	for y := 16; y < 32; y++ {
		for x := 16; x < 32; x++ {
			base := frameWorkTestSample(reference.Y, reference.Layout.BytesPerSample, x, y)
			want := base
			if x < 24 {
				neighbor := frameWorkTestSample(reference.Y, reference.Layout.BytesPerSample, x+1, y)
				want = frameWorkBlendA64(uint16(mask[x-16]), base, neighbor)
			}
			if got := frameWorkTestSample(output.Y, output.Layout.BytesPerSample, x, y); got != want {
				t.Fatalf("sample(%d,%d)=%d want %d", x, y, got, want)
			}
		}
	}
}

func TestFrameWorkBatchPredictBlockInterOBMCChromaSubsampledMatchesLibaomMasks(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 64})
	reference := testBatchFrame(t, output.Format)
	fillFrameWorkInterReferenceAllPlanes(reference, 0xff)
	ctx := testInterPredictionBatch(output, reference)

	visit := testOBMCInterPredictionVisit(motion.Vector{}, motion.Vector{Row: 16}, motion.Vector{Col: 16})
	var scratch FrameWorkInterPredictionScratch
	if err := ctx.PredictBlockInterOBMCWithFilters(0, visit, &scratch, motion.RegularFilters); err != nil {
		t.Fatal(err)
	}

	mask4, ok := frameWorkOBMCMask(4)
	if !ok {
		t.Fatal("missing 4-wide obmc mask")
	}
	for _, tt := range []struct {
		plane frame.Plane
		ref   frame.Plane
	}{
		{plane: output.U, ref: reference.U},
		{plane: output.V, ref: reference.V},
	} {
		for y := 8; y < 16; y++ {
			for x := 8; x < 16; x++ {
				want := frameWorkTestSample(tt.ref, reference.Layout.BytesPerSample, x, y)
				if y < 12 {
					neighbor := frameWorkTestSample(tt.ref, reference.Layout.BytesPerSample, x, y+1)
					want = frameWorkBlendA64(uint16(mask4[y-8]), want, neighbor)
				}
				if x < 12 {
					neighbor := frameWorkTestSample(tt.ref, reference.Layout.BytesPerSample, x+1, y)
					want = frameWorkBlendA64(uint16(mask4[x-8]), want, neighbor)
				}
				if got := frameWorkTestSample(tt.plane, output.Layout.BytesPerSample, x, y); got != want {
					t.Fatalf("sample(%d,%d)=%d want %d", x, y, got, want)
				}
			}
		}
	}
}

func TestFrameWorkBatchPredictBlockInterChromaSubsampledMatchesMotion(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 10, SubsamplingX: true, SubsamplingY: true, Align: 128})
	reference := testBatchFrame(t, output.Format)
	fillFrameWorkInterReferenceAllPlanes(reference, 0x3ff)
	ctx := testInterPredictionBatch(output, reference)
	mv := motion.Vector{Col: 3, Row: -5}
	filters := motion.InterpFilters{X: motion.InterpEightTapSmooth, Y: motion.InterpMultiTapSharp}
	visit := testInterPredictionVisit(mv)
	if err := ctx.PredictBlockInterWithFilters(0, visit, nil, filters); err != nil {
		t.Fatal(err)
	}

	wantY := testFrameWorkMotionPredictionPlane(t, reference.Y, output.Layout.BytesPerSample, output.Format.BitDepth, 16, 16, 16, 16, mv, filters)
	wantU := testFrameWorkMotionPredictionPlaneSubsampled(t, reference.U, output.Layout.BytesPerSample, output.Format.BitDepth, 8, 8, 8, 8, mv, true, true, filters)
	wantV := testFrameWorkMotionPredictionPlaneSubsampled(t, reference.V, output.Layout.BytesPerSample, output.Format.BitDepth, 8, 8, 8, 8, mv, true, true, filters)
	assertFrameWorkPlaneBlockEqualAt(t, output.Y, 16, 16, wantY, 0, 0, output.Layout.BytesPerSample, 16, 16)
	assertFrameWorkPlaneBlockEqualAt(t, output.U, 8, 8, wantU, 0, 0, output.Layout.BytesPerSample, 8, 8)
	assertFrameWorkPlaneBlockEqualAt(t, output.V, 8, 8, wantV, 0, 0, output.Layout.BytesPerSample, 8, 8)
}

func TestFrameWorkBatchPredictBlockInterIntraDCBlendsPredictors(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, MonoChrome: true, Align: 64})
	reference := testBatchFrame(t, output.Format)
	testFillFrame(reference, 16)
	for x := 16; x < 32; x++ {
		setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, x, 15, 80)
	}
	for y := 16; y < 32; y++ {
		setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, 15, y, 80)
	}
	setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, 15, 15, 80)
	ctx := testInterPredictionBatch(output, reference)
	visit := testInterIntraPredictionVisit(tile.InterIntraModeDC, false)
	var scratch FrameWorkInterPredictionScratch

	if err := ctx.PredictBlockInterWithFilters(0, visit, &scratch, motion.RegularFilters); err != nil {
		t.Fatal(err)
	}
	for y := 16; y < 32; y++ {
		for x := 16; x < 32; x++ {
			if got := frameWorkTestSample(output.Y, output.Layout.BytesPerSample, x, y); got != 48 {
				t.Fatalf("sample(%d,%d)=%d want 48", x, y, got)
			}
		}
	}
}

func TestFrameWorkBatchPredictBlockInterIntraWedgeBlendsPredictors(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, MonoChrome: true, Align: 64})
	reference := testBatchFrame(t, output.Format)
	testFillFrame(reference, 20)
	for x := 16; x < 32; x++ {
		setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, x, 15, 84)
	}
	ctx := testInterPredictionBatch(output, reference)
	visit := testInterIntraPredictionVisit(tile.InterIntraModeVertical, true)
	visit.Prediction.InterIntra.WedgeIndex = 0
	var scratch FrameWorkInterPredictionScratch

	if err := ctx.PredictBlockInterWithFilters(0, visit, &scratch, motion.RegularFilters); err != nil {
		t.Fatal(err)
	}
	var mask [16 * 16]byte
	if err := frameWorkBuildWedgeMask(mask[:], 16, tile.BlockSize16x16, 0, false); err != nil {
		t.Fatal(err)
	}
	for _, sample := range [][2]int{{0, 0}, {7, 8}, {15, 15}} {
		x := 16 + sample[0]
		y := 16 + sample[1]
		m := uint16(mask[sample[1]*16+sample[0]])
		want := frameWorkBlendA64(m, 84, 20)
		if got := frameWorkTestSample(output.Y, output.Layout.BytesPerSample, x, y); got != want {
			t.Fatalf("sample(%d,%d)=%d want %d mask=%d", x, y, got, want, m)
		}
	}
}

func TestFrameWorkBatchPredictBlockInterIntraAllocs(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, MonoChrome: true, Align: 64})
	reference := testBatchFrame(t, output.Format)
	testFillFrame(reference, 16)
	ctx := testInterPredictionBatch(output, reference)
	visit := testInterIntraPredictionVisit(tile.InterIntraModeSmooth, false)
	var scratch FrameWorkInterPredictionScratch

	allocs := testing.AllocsPerRun(1000, func() {
		if err := ctx.PredictBlockInterWithFilters(0, visit, &scratch, motion.RegularFilters); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("PredictBlockInter inter-intra allocated: %f", allocs)
	}
}

func TestFrameWorkBatchPredictBlockLumaInterCompoundAverageMatchesMotion(t *testing.T) {
	tests := []struct {
		name    string
		format  frame.Format
		max     uint16
		mv0     motion.Vector
		mv1     motion.Vector
		filters motion.InterpFilters
	}{
		{
			name:    "lowbd",
			format:  frame.Format{Width: 64, Height: 64, BitDepth: 8, Align: 64},
			max:     0xff,
			mv0:     motion.Vector{Col: 3, Row: 5},
			mv1:     motion.Vector{Col: -1, Row: 6},
			filters: motion.InterpFilters{X: motion.InterpMultiTapSharp, Y: motion.InterpEightTapSmooth},
		},
		{
			name:    "highbd-dist-wtd",
			format:  frame.Format{Width: 64, Height: 64, BitDepth: 10, Align: 128},
			max:     0x3ff,
			mv0:     motion.Vector{Col: 4, Row: 6},
			mv1:     motion.Vector{Col: -1, Row: 3},
			filters: motion.InterpFilters{X: motion.InterpEightTapRegular, Y: motion.InterpEightTapSmooth},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			output := testBatchFrame(t, tt.format)
			last := testBatchFrame(t, tt.format)
			bwd := testBatchFrame(t, tt.format)
			fillFrameWorkInterReference(last, tt.max)
			fillFrameWorkInterReferenceVariant(bwd, tt.max, 37)
			ctx := testCompoundInterPredictionBatch(output, last, bwd)
			compoundType := tile.CompoundTypeAverage
			if tt.name == "highbd-dist-wtd" {
				compoundType = tile.CompoundTypeDistWtd
			}
			visit := testCompoundInterPredictionVisit(tt.mv0, tt.mv1, compoundType)

			var scratch FrameWorkInterPredictionScratch
			if err := ctx.PredictBlockLumaInterCompoundWithFilters(0, visit, &scratch, tt.filters); err != nil {
				t.Fatal(err)
			}

			first := testFrameWorkCompoundConvBuf(t, last.Y, output.Layout.BytesPerSample, output.Format.BitDepth, 16, 16, 16, 16, tt.mv0, false, false, tt.filters)
			second := testFrameWorkCompoundConvBuf(t, bwd.Y, output.Layout.BytesPerSample, output.Format.BitDepth, 16, 16, 16, 16, tt.mv1, false, false, tt.filters)
			fwdOffset, bckOffset, err := ctx.frameWorkCompoundOffsets(visit.Prediction.InterMotion.References, visit.Prediction.CompoundBlend)
			if err != nil {
				t.Fatal(err)
			}
			assertFrameWorkCompoundBlendEqual(t, output.Y, first, second, output.Layout.BytesPerSample, output.Format.BitDepth, 16, 16, 16, 16, fwdOffset, bckOffset)
		})
	}
}

// TestFrameWorkBatchPredictBlockLumaInterCompoundAverageZeroMV pins the
// integer-pel compound-average path used by NEAREST_NEAREST blocks with
// MV=(0,0)/(0,0). With sub-pel offsets of zero the convolve degenerates to
// a copy, so the per-pixel blend is exactly (a+b+1)>>1. Pinning specific
// arithmetic samples here protects the libaom-equivalent rounding constant
// (1<<(DistPrecisionBits-1) = 8) at offsets 8/8 from silently regressing
// to a truncating average.
func TestFrameWorkBatchPredictBlockLumaInterCompoundAverageZeroMV(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, Align: 64})
	last := testBatchFrame(t, output.Format)
	bwd := testBatchFrame(t, output.Format)
	// Set up a constant-coloured 16x16 luma block on each reference so the
	// average degenerates to a deterministic per-pixel formula. Position
	// (16,16) matches testCompoundInterPredictionVisit's destination block.
	const blockX, blockY, blockW, blockH = 16, 16, 16, 16
	for row := range blockH {
		for col := range blockW {
			setFrameWorkTestSample(last.Y, last.Layout.BytesPerSample, blockX+col, blockY+row, 115)
			setFrameWorkTestSample(bwd.Y, bwd.Layout.BytesPerSample, blockX+col, blockY+row, 117)
		}
	}
	ctx := testCompoundInterPredictionBatch(output, last, bwd)
	visit := testCompoundInterPredictionVisit(motion.Vector{}, motion.Vector{}, tile.CompoundTypeAverage)
	var scratch FrameWorkInterPredictionScratch
	if err := ctx.PredictBlockLumaInterCompoundWithFilters(0, visit, &scratch, motion.RegularFilters); err != nil {
		t.Fatal(err)
	}
	for row := range blockH {
		for col := range blockW {
			got := frameWorkTestSample(output.Y, output.Layout.BytesPerSample, blockX+col, blockY+row)
			// (115 + 117 + 1) >> 1 = 116
			if got != 116 {
				t.Fatalf("compound avg(115,117) at (%d,%d) = %d, want 116", blockX+col, blockY+row, got)
			}
		}
	}
	// Also pin the asymmetric case where the average rounds up.
	for row := range blockH {
		for col := range blockW {
			setFrameWorkTestSample(last.Y, last.Layout.BytesPerSample, blockX+col, blockY+row, 164)
			setFrameWorkTestSample(bwd.Y, bwd.Layout.BytesPerSample, blockX+col, blockY+row, 165)
		}
	}
	if err := ctx.PredictBlockLumaInterCompoundWithFilters(0, visit, &scratch, motion.RegularFilters); err != nil {
		t.Fatal(err)
	}
	for row := range blockH {
		for col := range blockW {
			got := frameWorkTestSample(output.Y, output.Layout.BytesPerSample, blockX+col, blockY+row)
			// (164 + 165 + 1) >> 1 = 165
			if got != 165 {
				t.Fatalf("compound avg(164,165) at (%d,%d) = %d, want 165", blockX+col, blockY+row, got)
			}
		}
	}
	// Same-value inputs must blend back to themselves (no off-by-one drift).
	for row := range blockH {
		for col := range blockW {
			setFrameWorkTestSample(last.Y, last.Layout.BytesPerSample, blockX+col, blockY+row, 164)
			setFrameWorkTestSample(bwd.Y, bwd.Layout.BytesPerSample, blockX+col, blockY+row, 164)
		}
	}
	if err := ctx.PredictBlockLumaInterCompoundWithFilters(0, visit, &scratch, motion.RegularFilters); err != nil {
		t.Fatal(err)
	}
	for row := range blockH {
		for col := range blockW {
			got := frameWorkTestSample(output.Y, output.Layout.BytesPerSample, blockX+col, blockY+row)
			if got != 164 {
				t.Fatalf("compound avg(164,164) at (%d,%d) = %d, want 164", blockX+col, blockY+row, got)
			}
		}
	}
}

func TestFrameWorkBatchPredictBlockInterCompoundChromaSubsampledMatchesMotion(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 64})
	last := testBatchFrame(t, output.Format)
	bwd := testBatchFrame(t, output.Format)
	fillFrameWorkInterReferenceVariant(last, 0xff, 19)
	fillFrameWorkInterReferenceVariant(bwd, 0xff, 73)
	ctx := testCompoundInterPredictionBatch(output, last, bwd)
	mv0 := motion.Vector{Col: 3, Row: 5}
	mv1 := motion.Vector{Col: -5, Row: 1}
	filters := motion.InterpFilters{X: motion.InterpEightTapRegular, Y: motion.InterpEightTapSmooth}
	visit := testCompoundInterPredictionVisit(mv0, mv1, tile.CompoundTypeDistWtd)

	var scratch FrameWorkInterPredictionScratch
	if err := ctx.PredictBlockInterWithFilters(0, visit, &scratch, filters); err != nil {
		t.Fatal(err)
	}

	firstU := testFrameWorkCompoundConvBuf(t, last.U, output.Layout.BytesPerSample, output.Format.BitDepth, 8, 8, 8, 8, mv0, true, true, filters)
	secondU := testFrameWorkCompoundConvBuf(t, bwd.U, output.Layout.BytesPerSample, output.Format.BitDepth, 8, 8, 8, 8, mv1, true, true, filters)
	firstV := testFrameWorkCompoundConvBuf(t, last.V, output.Layout.BytesPerSample, output.Format.BitDepth, 8, 8, 8, 8, mv0, true, true, filters)
	secondV := testFrameWorkCompoundConvBuf(t, bwd.V, output.Layout.BytesPerSample, output.Format.BitDepth, 8, 8, 8, 8, mv1, true, true, filters)
	fwdOffset, bckOffset, err := ctx.frameWorkCompoundOffsets(visit.Prediction.InterMotion.References, visit.Prediction.CompoundBlend)
	if err != nil {
		t.Fatal(err)
	}
	assertFrameWorkCompoundBlendEqual(t, output.U, firstU, secondU, output.Layout.BytesPerSample, output.Format.BitDepth, 8, 8, 8, 8, fwdOffset, bckOffset)
	assertFrameWorkCompoundBlendEqual(t, output.V, firstV, secondV, output.Layout.BytesPerSample, output.Format.BitDepth, 8, 8, 8, 8, fwdOffset, bckOffset)
}

func TestFrameWorkBatchPredictBlockLumaInterCompoundDiffWtdMatchesLibaom(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 10, Align: 128})
	last := testBatchFrame(t, output.Format)
	bwd := testBatchFrame(t, output.Format)
	fillFrameWorkInterReferenceVariant(last, 0x3ff, 29)
	fillFrameWorkInterReferenceVariant(bwd, 0x3ff, 113)
	ctx := testCompoundInterPredictionBatch(output, last, bwd)
	mv0 := motion.Vector{Col: 3, Row: 5}
	mv1 := motion.Vector{Col: -5, Row: 1}
	filters := motion.InterpFilters{X: motion.InterpEightTapSmooth, Y: motion.InterpMultiTapSharp}
	visit := testCompoundInterPredictionVisit(mv0, mv1, tile.CompoundTypeDiffWtd)
	visit.Prediction.CompoundBlend.MaskType = tile.DiffWtdMaskType38Inv

	var scratch FrameWorkInterPredictionScratch
	if err := ctx.PredictBlockLumaInterCompoundWithFilters(0, visit, &scratch, filters); err != nil {
		t.Fatal(err)
	}

	first := testFrameWorkCompoundConvBuf(t, last.Y, output.Layout.BytesPerSample, output.Format.BitDepth, 16, 16, 16, 16, mv0, false, false, filters)
	second := testFrameWorkCompoundConvBuf(t, bwd.Y, output.Layout.BytesPerSample, output.Format.BitDepth, 16, 16, 16, 16, mv1, false, false, filters)
	mask := testFrameWorkDiffWtdMask(t, first, second, output.Format.BitDepth, 16, 16, tile.DiffWtdMaskType38Inv)
	assertFrameWorkMaskedCompoundEqual(t, output.Y, first, second, mask, 16, false, false, output.Layout.BytesPerSample, output.Format.BitDepth, 16, 16, 16, 16)
}

func TestFrameWorkBatchPredictBlockInterCompoundDiffWtdChromaSubsampledMatchesLibaom(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 64})
	last := testBatchFrame(t, output.Format)
	bwd := testBatchFrame(t, output.Format)
	fillFrameWorkInterReferenceVariant(last, 0xff, 17)
	fillFrameWorkInterReferenceVariant(bwd, 0xff, 191)
	ctx := testCompoundInterPredictionBatch(output, last, bwd)
	mv0 := motion.Vector{Col: 5, Row: -3}
	mv1 := motion.Vector{Col: -1, Row: 7}
	filters := motion.InterpFilters{X: motion.InterpEightTapRegular, Y: motion.InterpEightTapSmooth}
	visit := testCompoundInterPredictionVisit(mv0, mv1, tile.CompoundTypeDiffWtd)
	visit.Prediction.CompoundBlend.MaskType = tile.DiffWtdMaskType38

	var scratch FrameWorkInterPredictionScratch
	if err := ctx.PredictBlockInterWithFilters(0, visit, &scratch, filters); err != nil {
		t.Fatal(err)
	}

	firstY := testFrameWorkCompoundConvBuf(t, last.Y, output.Layout.BytesPerSample, output.Format.BitDepth, 16, 16, 16, 16, mv0, false, false, filters)
	secondY := testFrameWorkCompoundConvBuf(t, bwd.Y, output.Layout.BytesPerSample, output.Format.BitDepth, 16, 16, 16, 16, mv1, false, false, filters)
	mask := testFrameWorkDiffWtdMask(t, firstY, secondY, output.Format.BitDepth, 16, 16, tile.DiffWtdMaskType38)
	firstU := testFrameWorkCompoundConvBuf(t, last.U, output.Layout.BytesPerSample, output.Format.BitDepth, 8, 8, 8, 8, mv0, true, true, filters)
	secondU := testFrameWorkCompoundConvBuf(t, bwd.U, output.Layout.BytesPerSample, output.Format.BitDepth, 8, 8, 8, 8, mv1, true, true, filters)
	firstV := testFrameWorkCompoundConvBuf(t, last.V, output.Layout.BytesPerSample, output.Format.BitDepth, 8, 8, 8, 8, mv0, true, true, filters)
	secondV := testFrameWorkCompoundConvBuf(t, bwd.V, output.Layout.BytesPerSample, output.Format.BitDepth, 8, 8, 8, 8, mv1, true, true, filters)
	assertFrameWorkMaskedCompoundEqual(t, output.Y, firstY, secondY, mask, 16, false, false, output.Layout.BytesPerSample, output.Format.BitDepth, 16, 16, 16, 16)
	assertFrameWorkMaskedCompoundEqual(t, output.U, firstU, secondU, mask, 16, true, true, output.Layout.BytesPerSample, output.Format.BitDepth, 8, 8, 8, 8)
	assertFrameWorkMaskedCompoundEqual(t, output.V, firstV, secondV, mask, 16, true, true, output.Layout.BytesPerSample, output.Format.BitDepth, 8, 8, 8, 8)
}

func TestFrameWorkBuildWedgeMaskMatchesLibaomSamples(t *testing.T) {
	tests := []struct {
		name       string
		size       tile.BlockSize
		index      uint8
		sign       bool
		width      int
		height     int
		samples    [][2]int
		wantSample []byte
	}{
		{
			name:       "8x8 oblique27 neg",
			size:       tile.BlockSize8x8,
			index:      0,
			width:      8,
			height:     8,
			samples:    [][2]int{{0, 0}, {7, 7}},
			wantSample: []byte{64, 0},
		},
		{
			name:       "8x8 oblique27 positive",
			size:       tile.BlockSize8x8,
			index:      0,
			sign:       true,
			width:      8,
			height:     8,
			samples:    [][2]int{{0, 0}, {7, 7}},
			wantSample: []byte{0, 64},
		},
		{
			name:       "8x8 vertical neg",
			size:       tile.BlockSize8x8,
			index:      6,
			width:      8,
			height:     8,
			samples:    [][2]int{{0, 0}, {1, 0}, {2, 0}, {7, 0}},
			wantSample: []byte{57, 43, 21, 0},
		},
		{
			name:       "32x8 oblique63 positive",
			size:       tile.BlockSize32x8,
			index:      12,
			width:      32,
			height:     8,
			samples:    [][2]int{{0, 0}, {31, 7}},
			wantSample: []byte{0, 64},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mask := make([]byte, tt.width*tt.height)
			if err := frameWorkBuildWedgeMask(mask, tt.width, tt.size, tt.index, tt.sign); err != nil {
				t.Fatal(err)
			}
			for i, sample := range tt.samples {
				got := mask[sample[1]*tt.width+sample[0]]
				if got != tt.wantSample[i] {
					t.Fatalf("mask(%d,%d)=%d want %d", sample[0], sample[1], got, tt.wantSample[i])
				}
			}
		})
	}
}

func TestFrameWorkBuildWedgeMaskSignComplementsLibaom(t *testing.T) {
	sizes := []tile.BlockSize{
		tile.BlockSize8x8,
		tile.BlockSize8x16,
		tile.BlockSize16x8,
		tile.BlockSize16x16,
		tile.BlockSize8x32,
		tile.BlockSize32x8,
		tile.BlockSize16x32,
		tile.BlockSize32x16,
		tile.BlockSize32x32,
	}
	var mask [32 * 32]byte
	var complement [32 * 32]byte
	for _, size := range sizes {
		dims, ok := size.Dimensions()
		if !ok {
			t.Fatalf("missing dimensions for %v", size)
		}
		width := int(dims.W4) * 4
		height := int(dims.H4) * 4
		for index := range uint8(tile.MaxWedgeTypes) {
			if err := frameWorkBuildWedgeMask(mask[:], width, size, index, false); err != nil {
				t.Fatalf("size=%v index=%d neg err=%v", size, index, err)
			}
			if err := frameWorkBuildWedgeMask(complement[:], width, size, index, true); err != nil {
				t.Fatalf("size=%v index=%d pos err=%v", size, index, err)
			}
			for row := range height {
				for col := range width {
					got := int(mask[row*width+col]) + int(complement[row*width+col])
					if got != frameWorkWedgeMaxAlpha {
						t.Fatalf("size=%v index=%d mask(%d,%d) sum=%d want %d", size, index, col, row, got, frameWorkWedgeMaxAlpha)
					}
				}
			}
		}
	}
	if err := frameWorkBuildWedgeMask(mask[:], 4, tile.BlockSize4x4, 0, false); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("unsupported wedge err=%v want %v", err, ErrInvalidBatch)
	}
}

func TestFrameWorkBuildWedgeMaskAllocs(t *testing.T) {
	var mask [16 * 16]byte
	allocs := testing.AllocsPerRun(1000, func() {
		if err := frameWorkBuildWedgeMask(mask[:], 16, tile.BlockSize16x16, 0, false); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("frameWorkBuildWedgeMask allocated: %f", allocs)
	}
}

func TestFrameWorkBatchPredictBlockLumaInterCompoundWedgeMatchesLibaom(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 10, Align: 128})
	last := testBatchFrame(t, output.Format)
	bwd := testBatchFrame(t, output.Format)
	fillFrameWorkInterReferenceVariant(last, 0x3ff, 31)
	fillFrameWorkInterReferenceVariant(bwd, 0x3ff, 149)
	ctx := testCompoundInterPredictionBatch(output, last, bwd)
	mv0 := motion.Vector{Col: 3, Row: 5}
	mv1 := motion.Vector{Col: -5, Row: 1}
	filters := motion.InterpFilters{X: motion.InterpEightTapRegular, Y: motion.InterpEightTapSmooth}
	visit := testCompoundInterPredictionVisit(mv0, mv1, tile.CompoundTypeWedge)
	visit.Prediction.CompoundBlend.WedgeIndex = 0
	visit.Prediction.CompoundBlend.WedgeSign = false

	var scratch FrameWorkInterPredictionScratch
	if err := ctx.PredictBlockLumaInterCompoundWithFilters(0, visit, &scratch, filters); err != nil {
		t.Fatal(err)
	}

	first := testFrameWorkCompoundConvBuf(t, last.Y, output.Layout.BytesPerSample, output.Format.BitDepth, 16, 16, 16, 16, mv0, false, false, filters)
	second := testFrameWorkCompoundConvBuf(t, bwd.Y, output.Layout.BytesPerSample, output.Format.BitDepth, 16, 16, 16, 16, mv1, false, false, filters)
	mask := make([]byte, 16*16)
	if err := frameWorkBuildWedgeMask(mask, 16, visit.Block.Size, visit.Prediction.CompoundBlend.WedgeIndex, visit.Prediction.CompoundBlend.WedgeSign); err != nil {
		t.Fatal(err)
	}
	assertFrameWorkMaskedCompoundEqual(t, output.Y, first, second, mask, 16, false, false, output.Layout.BytesPerSample, output.Format.BitDepth, 16, 16, 16, 16)
}

func TestFrameWorkBatchPredictBlockInterCompoundWedgeChromaSubsampledMatchesLibaom(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 64})
	last := testBatchFrame(t, output.Format)
	bwd := testBatchFrame(t, output.Format)
	fillFrameWorkInterReferenceVariant(last, 0xff, 43)
	fillFrameWorkInterReferenceVariant(bwd, 0xff, 211)
	ctx := testCompoundInterPredictionBatch(output, last, bwd)
	mv0 := motion.Vector{Col: 5, Row: -3}
	mv1 := motion.Vector{Col: -1, Row: 7}
	filters := motion.InterpFilters{X: motion.InterpEightTapRegular, Y: motion.InterpEightTapSmooth}
	visit := testCompoundInterPredictionVisit(mv0, mv1, tile.CompoundTypeWedge)
	visit.Prediction.CompoundBlend.WedgeIndex = 6
	visit.Prediction.CompoundBlend.WedgeSign = false

	var scratch FrameWorkInterPredictionScratch
	if err := ctx.PredictBlockInterWithFilters(0, visit, &scratch, filters); err != nil {
		t.Fatal(err)
	}

	firstY := testFrameWorkCompoundConvBuf(t, last.Y, output.Layout.BytesPerSample, output.Format.BitDepth, 16, 16, 16, 16, mv0, false, false, filters)
	secondY := testFrameWorkCompoundConvBuf(t, bwd.Y, output.Layout.BytesPerSample, output.Format.BitDepth, 16, 16, 16, 16, mv1, false, false, filters)
	mask := make([]byte, 16*16)
	if err := frameWorkBuildWedgeMask(mask, 16, visit.Block.Size, visit.Prediction.CompoundBlend.WedgeIndex, visit.Prediction.CompoundBlend.WedgeSign); err != nil {
		t.Fatal(err)
	}
	firstU := testFrameWorkCompoundConvBuf(t, last.U, output.Layout.BytesPerSample, output.Format.BitDepth, 8, 8, 8, 8, mv0, true, true, filters)
	secondU := testFrameWorkCompoundConvBuf(t, bwd.U, output.Layout.BytesPerSample, output.Format.BitDepth, 8, 8, 8, 8, mv1, true, true, filters)
	firstV := testFrameWorkCompoundConvBuf(t, last.V, output.Layout.BytesPerSample, output.Format.BitDepth, 8, 8, 8, 8, mv0, true, true, filters)
	secondV := testFrameWorkCompoundConvBuf(t, bwd.V, output.Layout.BytesPerSample, output.Format.BitDepth, 8, 8, 8, 8, mv1, true, true, filters)
	assertFrameWorkMaskedCompoundEqual(t, output.Y, firstY, secondY, mask, 16, false, false, output.Layout.BytesPerSample, output.Format.BitDepth, 16, 16, 16, 16)
	assertFrameWorkMaskedCompoundEqual(t, output.U, firstU, secondU, mask, 16, true, true, output.Layout.BytesPerSample, output.Format.BitDepth, 8, 8, 8, 8)
	assertFrameWorkMaskedCompoundEqual(t, output.V, firstV, secondV, mask, 16, true, true, output.Layout.BytesPerSample, output.Format.BitDepth, 8, 8, 8, 8)
}

func TestFrameWorkCompoundDistanceWeightedOffsetsMatchLibaom(t *testing.T) {
	tests := []struct {
		name string
		cur  uint8
		ref0 uint8
		ref1 uint8
		want [2]int
	}{
		{name: "equal distance", cur: 8, ref0: 4, ref1: 12, want: [2]int{7, 9}},
		{name: "forward nearer", cur: 8, ref0: 2, ref1: 10, want: [2]int{4, 12}},
		{name: "backward nearer", cur: 8, ref0: 6, ref1: 15, want: [2]int{13, 3}},
		{name: "zero distance", cur: 8, ref0: 8, ref1: 12, want: [2]int{13, 3}},
		{name: "wraparound", cur: 1, ref0: 14, ref1: 4, want: [2]int{7, 9}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			fwd, bck, err := frameWorkDistanceWeightedCompoundOffsets(4, tt.cur, tt.ref0, tt.ref1)
			if err != nil {
				t.Fatal(err)
			}
			if got := [2]int{fwd, bck}; got != tt.want {
				t.Fatalf("offsets=%v want %v", got, tt.want)
			}
		})
	}
	if _, _, err := frameWorkDistanceWeightedCompoundOffsets(0, 0, 0, 0); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("bad bits err=%v want %v", err, ErrInvalidBatch)
	}
	if _, _, err := frameWorkDistanceWeightedCompoundOffsets(4, 16, 0, 1); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("out of range hint err=%v want %v", err, ErrInvalidBatch)
	}
}

func FuzzFrameWorkDistanceWeightedCompoundOffsets(f *testing.F) {
	f.Add(uint8(4), uint32(8), uint32(4), uint32(12))
	f.Add(uint8(4), uint32(8), uint32(2), uint32(10))
	f.Add(uint8(4), uint32(1), uint32(14), uint32(4))
	f.Add(uint8(5), uint32(16), uint32(3), uint32(28))

	f.Fuzz(func(t *testing.T, bits uint8, cur uint32, ref0 uint32, ref1 uint32) {
		bits = bits%8 + 1
		limit := uint32(1) << bits
		cur %= limit
		ref0 %= limit
		ref1 %= limit
		fwd, bck, err := frameWorkDistanceWeightedCompoundOffsets(bits, uint8(cur), uint8(ref0), uint8(ref1))
		if err != nil {
			t.Fatalf("frameWorkDistanceWeightedCompoundOffsets err=%v", err)
		}
		if fwd < 0 || bck < 0 || fwd+bck != 16 {
			t.Fatalf("offsets=%d,%d", fwd, bck)
		}
	})
}

func TestFrameWorkBatchPredictBlockLumaDispatchesCompoundAverage(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, Align: 64})
	last := testBatchFrame(t, output.Format)
	bwd := testBatchFrame(t, output.Format)
	fillFrameWorkInterReference(last, 0xff)
	fillFrameWorkInterReferenceVariant(bwd, 0xff, 91)
	ctx := testCompoundInterPredictionBatch(output, last, bwd)
	visit := testCompoundInterPredictionVisit(motion.Vector{Col: 8, Row: 0}, motion.Vector{Col: -8, Row: 0}, tile.CompoundTypeAverage)

	var interScratch FrameWorkInterPredictionScratch
	scratch := FrameWorkPredictionScratch{Inter: &interScratch}
	if err := ctx.PredictBlockLuma(0, visit, &scratch); err != nil {
		t.Fatal(err)
	}
	first := testFrameWorkCompoundConvBuf(t, last.Y, output.Layout.BytesPerSample, output.Format.BitDepth, 16, 16, 16, 16, visit.Prediction.InterMotion.MV[0], false, false, motion.RegularFilters)
	second := testFrameWorkCompoundConvBuf(t, bwd.Y, output.Layout.BytesPerSample, output.Format.BitDepth, 16, 16, 16, 16, visit.Prediction.InterMotion.MV[1], false, false, motion.RegularFilters)
	assertFrameWorkCompoundBlendEqual(t, output.Y, first, second, output.Layout.BytesPerSample, output.Format.BitDepth, 16, 16, 16, 16, 8, 8)
}

func TestFrameWorkBatchPredictBlockDispatchesCompletePlanes(t *testing.T) {
	mono := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, MonoChrome: true, Align: 64})
	monoCtx := testIntraPredictionBatch(mono)
	for x := 16; x < 32; x++ {
		setFrameWorkTestSample(mono.Y, mono.Layout.BytesPerSample, x, 15, 10)
	}
	for y := 16; y < 32; y++ {
		setFrameWorkTestSample(mono.Y, mono.Layout.BytesPerSample, 15, y, 50)
	}
	var predictionScratch FrameWorkPredictionScratch
	if err := monoCtx.PredictBlock(0, testIntraPredictionVisit(tile.IntraModeDC), &predictionScratch); err != nil {
		t.Fatal(err)
	}
	if got := frameWorkTestSample(mono.Y, mono.Layout.BytesPerSample, 16, 16); got != 30 {
		t.Fatalf("mono intra sample=%d want 30", got)
	}

	colorIntra := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 64})
	colorIntraCtx := testIntraPredictionBatch(colorIntra)
	for x := 16; x < 32; x++ {
		setFrameWorkTestSample(colorIntra.Y, colorIntra.Layout.BytesPerSample, x, 15, 10)
	}
	for y := 16; y < 32; y++ {
		setFrameWorkTestSample(colorIntra.Y, colorIntra.Layout.BytesPerSample, 15, y, 50)
	}
	for x := 8; x < 16; x++ {
		setFrameWorkTestSample(colorIntra.U, colorIntra.Layout.BytesPerSample, x, 7, 20)
		setFrameWorkTestSample(colorIntra.V, colorIntra.Layout.BytesPerSample, x, 7, 30)
	}
	for y := 8; y < 16; y++ {
		setFrameWorkTestSample(colorIntra.U, colorIntra.Layout.BytesPerSample, 7, y, 60)
		setFrameWorkTestSample(colorIntra.V, colorIntra.Layout.BytesPerSample, 7, y, 70)
	}
	colorVisit := testIntraPredictionVisit(tile.IntraModeDC)
	colorVisit.Prediction.ChromaMode = tile.ChromaIntraModeDC
	colorVisit.Prediction.ChromaModeValid = true
	if err := colorIntraCtx.PredictBlock(0, colorVisit, &predictionScratch); err != nil {
		t.Fatal(err)
	}
	if got := frameWorkTestSample(colorIntra.Y, colorIntra.Layout.BytesPerSample, 16, 16); got != 30 {
		t.Fatalf("color intra y sample=%d want 30", got)
	}
	if got := frameWorkTestSample(colorIntra.U, colorIntra.Layout.BytesPerSample, 8, 8); got != 40 {
		t.Fatalf("color intra u sample=%d want 40", got)
	}
	if got := frameWorkTestSample(colorIntra.V, colorIntra.Layout.BytesPerSample, 8, 8); got != 50 {
		t.Fatalf("color intra v sample=%d want 50", got)
	}

	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 64})
	reference := testBatchFrame(t, output.Format)
	fillFrameWorkInterReferenceAllPlanes(reference, 0xff)
	ctx := testInterPredictionBatch(output, reference)
	mv := motion.Vector{Col: 3, Row: -5}
	visit := testInterPredictionVisit(mv)
	if err := ctx.PredictBlock(0, visit, nil); err != nil {
		t.Fatal(err)
	}

	wantY := testFrameWorkMotionPredictionPlane(t, reference.Y, output.Layout.BytesPerSample, output.Format.BitDepth, 16, 16, 16, 16, mv, motion.RegularFilters)
	wantU := testFrameWorkMotionPredictionPlaneSubsampled(t, reference.U, output.Layout.BytesPerSample, output.Format.BitDepth, 8, 8, 8, 8, mv, true, true, motion.RegularFilters)
	wantV := testFrameWorkMotionPredictionPlaneSubsampled(t, reference.V, output.Layout.BytesPerSample, output.Format.BitDepth, 8, 8, 8, 8, mv, true, true, motion.RegularFilters)
	assertFrameWorkPlaneBlockEqualAt(t, output.Y, 16, 16, wantY, 0, 0, output.Layout.BytesPerSample, 16, 16)
	assertFrameWorkPlaneBlockEqualAt(t, output.U, 8, 8, wantU, 0, 0, output.Layout.BytesPerSample, 8, 8)
	assertFrameWorkPlaneBlockEqualAt(t, output.V, 8, 8, wantV, 0, 0, output.Layout.BytesPerSample, 8, 8)
}

func TestFrameWorkBatchPredictBlockIntraSubsampledChromaUsesPlaneEdges(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 64})
	ctx := testIntraPredictionBatch(output)
	for y := range 16 {
		setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, 3, y, 64)
	}

	visit := testIntraPredictionVisit(tile.IntraModeDC)
	visit.Block = tile.BlockVisit{
		MICol: 1, MIRow: 0, MIColEnd: 2, MIRowEnd: 4,
		X4: 1, Y4: 0, Size: tile.BlockSize4x16, VisibleW4: 1, VisibleH4: 4,
		HaveTop: false, HaveLeft: true,
	}
	visit.Prediction.ChromaMode = tile.ChromaIntraModeVertical
	visit.Prediction.ChromaModeValid = true

	var scratch FrameWorkPredictionScratch
	if err := ctx.PredictBlock(0, visit, &scratch); err != nil {
		t.Fatal(err)
	}
	for y := range 8 {
		for x := range 4 {
			if got := frameWorkTestSample(output.U, output.Layout.BytesPerSample, x, y); got != 127 {
				t.Fatalf("u sample(%d,%d)=%d want 127", x, y, got)
			}
			if got := frameWorkTestSample(output.V, output.Layout.BytesPerSample, x, y); got != 127 {
				t.Fatalf("v sample(%d,%d)=%d want 127", x, y, got)
			}
		}
	}
}

func TestFrameWorkBatchPredictBlockIntraCoeffClippedChromaSmooth(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 352, Height: 256, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 64})
	ctx := testIntraPredictionBatch(output)
	ctx.Sequence = FrameWorkSequenceContextFromHeader(parser.SequenceHeader{
		Use128x128Superblock: true,
		ColorConfig: parser.ColorConfig{
			BitDepth:     output.Format.BitDepth,
			SubsamplingX: true,
			SubsamplingY: true,
		},
	})
	ctx.Jobs = []tile.Job{{SBCols: 3, SBRows: 2}}
	for y := range 32 {
		setFrameWorkTestSample(output.U, output.Layout.BytesPerSample, 159, y, 77)
		setFrameWorkTestSample(output.V, output.Layout.BytesPerSample, 159, y, 91)
	}
	visit := testIntraPredictionVisit(tile.IntraModeDC)
	visit.Block = tile.BlockVisit{
		MICol: 64, MIRow: 0, MIColEnd: 88, MIRowEnd: 32,
		X4: 0, Y4: 0, Size: tile.BlockSize128x128, VisibleW4: 24, VisibleH4: 32,
		HaveTop: false, HaveLeft: true,
	}
	visit.Prediction.ChromaMode = tile.ChromaIntraModeSmooth
	visit.Prediction.ChromaModeValid = true
	block := tile.BlockCoeffBlock{
		Plane: 1,
		Block: tile.TransformBlock{X4: 8, Y4: 0, Size: tile.TransformSize32x32, VisibleW4: 4, VisibleH4: 8},
	}
	var scratch FrameWorkIntraPredictionScratch
	if err := ctx.PredictBlockIntraCoeff(0, visit, block, &scratch); err != nil {
		t.Fatal(err)
	}
	for y := range 32 {
		for x := 160; x < 176; x++ {
			if got := frameWorkTestSample(output.U, output.Layout.BytesPerSample, x, y); got != 77 {
				t.Fatalf("u sample(%d,%d)=%d want 77", x, y, got)
			}
		}
	}
	block.Plane = 2
	if err := ctx.PredictBlockIntraCoeff(0, visit, block, &scratch); err != nil {
		t.Fatal(err)
	}
	for y := range 32 {
		for x := 160; x < 176; x++ {
			if got := frameWorkTestSample(output.V, output.Layout.BytesPerSample, x, y); got != 91 {
				t.Fatalf("v sample(%d,%d)=%d want 91", x, y, got)
			}
		}
	}
}

func TestFrameWorkBatchPredictBlockIntraCoeffLumaPaddingNoOp(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 320, Height: 180, BitDepth: 8, MonoChrome: true, Align: 64})
	testFillFrame(output, 19)
	before := slices.Clone(output.Y.Pix)
	ctx := testIntraPredictionBatch(output)
	ctx.Jobs = []tile.Job{{SBCols: 5, SBRows: 3}}

	visit := testIntraPredictionVisit(tile.IntraModeSmooth)
	visit.Block = tile.BlockVisit{
		MICol: 4, MIRow: 44, MIColEnd: 5, MIRowEnd: 46,
		X4: 4, Y4: 12, Size: tile.BlockSize4x8, VisibleW4: 1, VisibleH4: 2,
		HaveTop: true, HaveLeft: true,
	}
	block := tile.BlockCoeffBlock{
		Plane: 0,
		Block: tile.TransformBlock{X4: 4, Y4: 13, Size: tile.TransformSize4x4, VisibleW4: 1, VisibleH4: 1},
	}
	var scratch FrameWorkIntraPredictionScratch
	if err := ctx.PredictBlockIntraCoeff(0, visit, block, &scratch); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(output.Y.Pix, before) {
		t.Fatalf("padding-only luma TXB prediction modified output")
	}
}

func TestFrameWorkBatchPredictBlockIntraClippedChromaSmooth(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 352, Height: 256, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 64})
	ctx := testIntraPredictionBatch(output)
	ctx.Sequence = FrameWorkSequenceContextFromHeader(parser.SequenceHeader{
		Use128x128Superblock: true,
		ColorConfig: parser.ColorConfig{
			BitDepth:     output.Format.BitDepth,
			SubsamplingX: true,
			SubsamplingY: true,
		},
	})
	ctx.Jobs = []tile.Job{{SBCols: 3, SBRows: 2}}
	for y := range 64 {
		setFrameWorkTestSample(output.U, output.Layout.BytesPerSample, 127, y, 77)
		setFrameWorkTestSample(output.V, output.Layout.BytesPerSample, 127, y, 91)
	}
	visit := testIntraPredictionVisit(tile.IntraModeDC)
	visit.Block = tile.BlockVisit{
		MICol: 64, MIRow: 0, MIColEnd: 88, MIRowEnd: 32,
		X4: 0, Y4: 0, Size: tile.BlockSize128x128, VisibleW4: 24, VisibleH4: 32,
		HaveTop: false, HaveLeft: true,
	}
	visit.Prediction.ChromaMode = tile.ChromaIntraModeSmooth
	visit.Prediction.ChromaModeValid = true

	var scratch FrameWorkPredictionScratch
	if err := ctx.PredictBlock(0, visit, &scratch); err != nil {
		t.Fatal(err)
	}
	for y := range 64 {
		for x := 128; x < 176; x++ {
			if got := frameWorkTestSample(output.U, output.Layout.BytesPerSample, x, y); got != 77 {
				t.Fatalf("u sample(%d,%d)=%d want 77", x, y, got)
			}
			if got := frameWorkTestSample(output.V, output.Layout.BytesPerSample, x, y); got != 91 {
				t.Fatalf("v sample(%d,%d)=%d want 91", x, y, got)
			}
		}
	}
}

func TestFrameWorkBatchPredictBlockRejectsIncompletePlaneSupport(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, Align: 64})
	ctx := testIntraPredictionBatch(output)
	var scratch FrameWorkPredictionScratch
	if err := ctx.PredictBlock(0, testIntraPredictionVisit(tile.IntraModeDC), &scratch); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("missing chroma mode err=%v want %v", err, ErrInvalidBatch)
	}
	cfl := testIntraPredictionVisit(tile.IntraModeDC)
	cfl.Prediction.ChromaMode = tile.ChromaIntraModeCFL
	cfl.Prediction.ChromaModeValid = true
	cfl.Prediction.CFLAlphaValid = true
	if err := ctx.PredictBlock(0, cfl, &scratch); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("cfl intra err=%v want %v", err, ErrInvalidBatch)
	}
	if err := ctx.PredictBlock(0, tile.BlockLoopVisit{}, &scratch); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("missing prediction err=%v want %v", err, ErrInvalidBatch)
	}
}

func TestFrameWorkBatchPredictBlockChromaCFLMatchesPrimitives(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 64})
	want := testBatchFrame(t, output.Format)
	visit := testCFLPredictionVisit()
	seedFrameWorkCFLPredictionFrame(output)
	seedFrameWorkCFLPredictionFrame(want)

	ctx := testIntraPredictionBatch(output)
	var scratch FrameWorkCFLPredictionScratch
	if err := ctx.PredictBlockChromaCFL(0, visit, &scratch); err != nil {
		t.Fatal(err)
	}

	testPredictFrameWorkCFLWant(t, want, visit, FrameWorkPlaneU)
	testPredictFrameWorkCFLWant(t, want, visit, FrameWorkPlaneV)
	assertFrameWorkPlaneBlockEqual(t, output.U, want.U, output.Layout.BytesPerSample, 8, 8, 8, 8)
	assertFrameWorkPlaneBlockEqual(t, output.V, want.V, output.Layout.BytesPerSample, 8, 8, 8, 8)
}

func TestFrameWorkBatchPredictBlockChromaCFLMatchesPerPlaneOracle(t *testing.T) {
	tests := []struct {
		name   string
		format frame.Format
		pad    bool
		zero   bool
	}{
		{name: "420-10", format: frame.Format{Width: 64, Height: 64, BitDepth: 10, SubsamplingX: true, SubsamplingY: true, Align: 128}},
		{name: "420-12-padding-alpha-zero", format: frame.Format{Width: 34, Height: 34, BitDepth: 12, SubsamplingX: true, SubsamplingY: true, Align: 128}, pad: true, zero: true},
		{name: "422-10", format: frame.Format{Width: 64, Height: 64, BitDepth: 10, SubsamplingX: true, Align: 128}},
		{name: "422-12", format: frame.Format{Width: 64, Height: 64, BitDepth: 12, SubsamplingX: true, Align: 128}},
		{name: "444-10", format: frame.Format{Width: 64, Height: 64, BitDepth: 10, Align: 128}},
		{name: "444-12", format: frame.Format{Width: 64, Height: 64, BitDepth: 12, Align: 128}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := testBatchFrame(t, tt.format)
			want := testBatchFrame(t, tt.format)
			testFillCFLFrame(got)
			testFillCFLFrame(want)
			visit := testCFLPredictionVisit()
			if tt.pad {
				visit.Block.MICol, visit.Block.MIRow = 6, 6
				visit.Block.MIColEnd, visit.Block.MIRowEnd = 10, 10
				visit.Block.X4, visit.Block.Y4 = 6, 6
			}
			if tt.zero {
				visit.Prediction.CFLAlpha.JointSign = 0
			}
			gotCtx := testIntraPredictionBatch(got)
			wantCtx := testIntraPredictionBatch(want)
			var gotScratch, wantScratch FrameWorkCFLPredictionScratch
			if err := gotCtx.PredictBlockChromaCFL(0, visit, &gotScratch); err != nil {
				t.Fatal(err)
			}
			if err := wantCtx.predictBlockChromaCFLPlanePtr(0, &visit, FrameWorkPlaneU, &wantScratch); err != nil {
				t.Fatal(err)
			}
			if err := wantCtx.predictBlockChromaCFLPlanePtr(0, &visit, FrameWorkPlaneV, &wantScratch); err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(got.U.Pix, want.U.Pix) || !slices.Equal(got.V.Pix, want.V.Pix) {
				t.Fatal("shared CfL output differs from the former per-plane path")
			}
		})
	}
}

func TestFrameWorkBatchPredictBlockChromaCFLPreservesAliasAndErrorOrder(t *testing.T) {
	t.Run("overlapping-luma-and-chroma", func(t *testing.T) {
		format := frame.Format{Width: 64, Height: 64, BitDepth: 8, Align: 64}
		got, want := testBatchFrame(t, format), testBatchFrame(t, format)
		gotShared, wantShared := make([]byte, 64*64), make([]byte, 64*64)
		got.Y.Pix, got.U.Pix = gotShared, gotShared
		want.Y.Pix, want.U.Pix = wantShared, wantShared
		got.Y.Stride, got.U.Stride = 64, 64
		want.Y.Stride, want.U.Stride = 64, 64
		got.Format.SubsamplingX, got.Format.SubsamplingY = false, false
		want.Format = got.Format
		gotCtx, wantCtx := testIntraPredictionBatch(got), testIntraPredictionBatch(want)
		testFillCFLFrame(got)
		testFillCFLFrame(want)
		if !frameWorkCFLYMayAliasChroma(got) {
			t.Fatal("overlapping plane storage was not detected")
		}
		visit := testCFLPredictionVisit()
		var gotScratch, wantScratch FrameWorkCFLPredictionScratch
		if err := gotCtx.PredictBlockChromaCFL(0, visit, &gotScratch); err != nil {
			t.Fatal(err)
		}
		if err := wantCtx.predictBlockChromaCFLPlanePtr(0, &visit, FrameWorkPlaneU, &wantScratch); err != nil {
			t.Fatal(err)
		}
		if err := wantCtx.predictBlockChromaCFLPlanePtr(0, &visit, FrameWorkPlaneV, &wantScratch); err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(gotShared, wantShared) || !slices.Equal(got.V.Pix, want.V.Pix) {
			t.Fatal("aliased output differs from the former per-plane path")
		}
	})

	t.Run("v-geometry-error-after-u-write", func(t *testing.T) {
		format := frame.Format{Width: 64, Height: 64, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 64}
		got, want := testBatchFrame(t, format), testBatchFrame(t, format)
		testFillCFLFrame(got)
		testFillCFLFrame(want)
		got.V.Stride, want.V.Stride = 0, 0
		gotCtx, wantCtx := testIntraPredictionBatch(got), testIntraPredictionBatch(want)
		visit := testCFLPredictionVisit()
		var gotScratch, wantScratch FrameWorkCFLPredictionScratch
		gotErr := gotCtx.PredictBlockChromaCFL(0, visit, &gotScratch)
		if !errors.Is(gotErr, ErrInvalidBatch) {
			t.Fatalf("shared error=%v want %v", gotErr, ErrInvalidBatch)
		}
		if err := wantCtx.predictBlockChromaCFLPlanePtr(0, &visit, FrameWorkPlaneU, &wantScratch); err != nil {
			t.Fatal(err)
		}
		wantErr := wantCtx.predictBlockChromaCFLPlanePtr(0, &visit, FrameWorkPlaneV, &wantScratch)
		if !errors.Is(wantErr, ErrInvalidBatch) {
			t.Fatalf("legacy error=%v want %v", wantErr, ErrInvalidBatch)
		}
		if !slices.Equal(got.U.Pix, want.U.Pix) || !slices.Equal(got.V.Pix, want.V.Pix) {
			t.Fatal("partial output before V error differs from the former per-plane path")
		}
	})
}

func TestFrameWorkSubsampleLumaCFLQ3HBDUnalignedFallbackAndMaxValidation(t *testing.T) {
	for _, tt := range []struct {
		name     string
		bitDepth uint8
		subX     bool
		subY     bool
	}{
		{name: "420-10", bitDepth: 10, subX: true, subY: true},
		{name: "422-12", bitDepth: 12, subX: true},
		{name: "444-10", bitDepth: 10},
	} {
		for _, layout := range []struct {
			name   string
			stride int
			base   int
		}{{name: "odd-stride-odd-base", stride: 65, base: 1}, {name: "even-stride-odd-base", stride: 66, base: 1}} {
			t.Run(tt.name+"/"+layout.name, func(t *testing.T) {
				const width, height = 16, 16
				stride := layout.stride
				backing := make([]byte, layout.base+stride*height)
				plane := frame.Plane{Pix: backing[layout.base:], Stride: stride, Width: 32, Height: 16}
				max := uint16((1 << tt.bitDepth) - 1)
				for row := 0; row < height; row++ {
					for col := 0; col < 32; col++ {
						value := (uint16(row*37+col*23+5) & max)
						offset := row*stride + col*2
						plane.Pix[offset], plane.Pix[offset+1] = byte(value), byte(value>>8)
					}
				}
				var got, want [prediction.CFLBufSquare]uint16
				if err := frameWorkSubsampleLumaCFLQ3(got[:], new([prediction.CFLBufSquare]uint16), plane, 2, tt.bitDepth, 2, 0, width, height, tt.subX, tt.subY); err != nil {
					t.Fatal(err)
				}
				outW, outH := width, height
				if tt.subX {
					outW /= 2
				}
				if tt.subY {
					outH /= 2
				}
				read := func(row, col int) uint16 {
					offset := row*stride + (2+col)*2
					return uint16(plane.Pix[offset]) | uint16(plane.Pix[offset+1])<<8
				}
				for row := 0; row < outH; row++ {
					for col := 0; col < outW; col++ {
						var value uint16
						switch {
						case tt.subX && tt.subY:
							top, bot := row*2, row*2+1
							left := col * 2
							value = (read(top, left) + read(top, left+1) + read(bot, left) + read(bot, left+1)) << 1
						case tt.subX:
							left := col * 2
							value = (read(row, left) + read(row, left+1)) << 2
						default:
							value = read(row, col) << 3
						}
						want[row*prediction.CFLBufLine+col] = value
					}
				}
				if !slices.Equal(got[:], want[:]) {
					t.Fatal("unaligned HBD fallback differs from scalar byte reference")
				}
				badOffset := 5*stride + (2+7)*2
				plane.Pix[badOffset], plane.Pix[badOffset+1] = byte(max+1), byte((max+1)>>8)
				if err := frameWorkSubsampleLumaCFLQ3(got[:], new([prediction.CFLBufSquare]uint16), plane, 2, tt.bitDepth, 2, 0, width, height, tt.subX, tt.subY); !errors.Is(err, ErrInvalidBatch) {
					t.Fatalf("sample above bit-depth maximum err=%v want %v", err, ErrInvalidBatch)
				}
			})
		}
	}
}

func TestFrameWorkSubsampleLumaCFLQ3HBDAlignedMaxValidation(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 10, SubsamplingX: true, SubsamplingY: true, Align: 128})
	if output.Y.Stride&1 != 0 {
		t.Skip("requires an even-stride HBD plane")
	}
	const x, y, width, height = 16, 16, 16, 16
	offset := y*output.Y.Stride + x*output.Layout.BytesPerSample
	output.Y.Pix[offset], output.Y.Pix[offset+1] = 0, 0x04 // 1024 is outside the 10-bit range.
	var got [prediction.CFLBufSquare]uint16
	if err := frameWorkSubsampleLumaCFLQ3(got[:], new([prediction.CFLBufSquare]uint16), output.Y, output.Layout.BytesPerSample, 10, x, y, width, height, true, true); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("aligned SIMD-route sample above maximum err=%v want %v", err, ErrInvalidBatch)
	}
}

// BenchmarkFrameWorkPredictBlockChromaCFLWarm isolates duplicate per-plane
// preparation versus shared preparation; both paths use the current HBD
// subsampler, so this is a block-level hoisting comparison, not a full prechange
// baseline.
func BenchmarkFrameWorkPredictBlockChromaCFLWarm(b *testing.B) {
	blockCases := []struct {
		name string
		size tile.BlockSize
	}{
		{name: "8x8", size: tile.BlockSize8x8},
		{name: "16x16", size: tile.BlockSize16x16},
		{name: "32x32", size: tile.BlockSize32x32},
	}
	for _, bitDepth := range []uint8{10, 12} {
		for _, chroma := range []struct {
			name string
			x, y bool
		}{{name: "420", x: true, y: true}, {name: "444"}} {
			for _, block := range blockCases {
				name := fmt.Sprintf("bd%d-%s-%s", bitDepth, chroma.name, block.name)
				b.Run(name, func(b *testing.B) {
					for _, method := range []string{"sequential-per-plane", "shared"} {
						b.Run(method, func(b *testing.B) {
							format := frame.Format{Width: 64, Height: 64, BitDepth: bitDepth, SubsamplingX: chroma.x, SubsamplingY: chroma.y, Align: 128}
							output := benchmarkCFLFrame(b, format)
							testFillCFLFrame(output)
							ctx := testIntraPredictionBatch(output)
							visit := testCFLPredictionVisit()
							visit.Block.Size = block.size
							dims, ok := block.size.Dimensions()
							if !ok {
								b.Fatalf("missing dimensions for %s", block.name)
							}
							visit.Block.MIColEnd = visit.Block.MICol + uint16(dims.W4)
							visit.Block.MIRowEnd = visit.Block.MIRow + uint16(dims.H4)
							visit.Block.VisibleW4, visit.Block.VisibleH4 = dims.W4, dims.H4
							chromaX, chromaY := 16, 16
							if chroma.x {
								chromaX >>= 1
							}
							if chroma.y {
								chromaY >>= 1
							}
							sinkIndex := chromaY*output.U.Stride + chromaX*output.Layout.BytesPerSample
							var scratch FrameWorkCFLPredictionScratch
							// Y is read-only here, and the repeated U/V writes do not touch
							// the current block's above/left DC edges, so setup remains
							// stable across warm iterations without frame resets.
							predict := func() {
								var err error
								if method == "shared" {
									err = ctx.PredictBlockChromaCFL(0, visit, &scratch)
								} else if err = ctx.predictBlockChromaCFLPlanePtr(0, &visit, FrameWorkPlaneU, &scratch); err == nil {
									err = ctx.predictBlockChromaCFLPlanePtr(0, &visit, FrameWorkPlaneV, &scratch)
								}
								if err != nil {
									b.Fatalf("CfL prediction: %v", err)
								}
								benchmarkCFLPredictionSink += int(output.U.Pix[sinkIndex]) + int(output.V.Pix[sinkIndex])
							}
							if allocs := testing.AllocsPerRun(100, predict); allocs != 0 {
								b.Fatalf("warm CfL path allocated: %f", allocs)
							}
							b.ReportAllocs()
							b.ResetTimer()
							for b.Loop() {
								predict()
							}
						})
					}
				})
			}
		}
	}
}

func benchmarkCFLFrame(b *testing.B, format frame.Format) *frame.Frame {
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

func TestFrameWorkBatchPredictBlockChromaCFLHBDNoAllocs(t *testing.T) {
	for _, bitDepth := range []uint8{10, 12} {
		for _, chroma := range []struct {
			name string
			x, y bool
		}{{name: "420", x: true, y: true}, {name: "444"}} {
			t.Run(fmt.Sprintf("bd%d-%s", bitDepth, chroma.name), func(t *testing.T) {
				output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: bitDepth, SubsamplingX: chroma.x, SubsamplingY: chroma.y, Align: 128})
				testFillCFLFrame(output)
				ctx := testIntraPredictionBatch(output)
				visit := testCFLPredictionVisit()
				var scratch FrameWorkCFLPredictionScratch
				allocs := testing.AllocsPerRun(100, func() {
					if err := ctx.PredictBlockChromaCFL(0, visit, &scratch); err != nil {
						t.Fatal(err)
					}
				})
				if allocs != 0 {
					t.Fatalf("HBD CfL path allocated: %f", allocs)
				}
			})
		}
	}
}

func testFillCFLFrame(output *frame.Frame) {
	bytesPerSample := output.Layout.BytesPerSample
	max := uint16((1 << output.Format.BitDepth) - 1)
	for planeIndex, plane := range []*frame.Plane{&output.Y, &output.U, &output.V} {
		if plane.Stride <= 0 {
			continue
		}
		for offset := 0; offset+bytesPerSample <= len(plane.Pix); offset += bytesPerSample {
			value := uint16((offset*17 + (planeIndex+1)*113) % (int(max) + 1))
			plane.Pix[offset] = byte(value)
			if bytesPerSample == 2 {
				plane.Pix[offset+1] = byte(value >> 8)
			}
		}
	}
}

func TestFrameWorkSubsampleLumaCFLQ3MatchesPrimitives(t *testing.T) {
	tests := []struct {
		name     string
		bitDepth uint8
		subX     bool
		subY     bool
	}{
		{name: "lowbd-420", bitDepth: 8, subX: true, subY: true},
		{name: "highbd-444", bitDepth: 10},
		{name: "highbd-422", bitDepth: 12, subX: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			format := frame.Format{Width: 64, Height: 64, BitDepth: tt.bitDepth, SubsamplingX: tt.subX, SubsamplingY: tt.subY, Align: 128}
			output := testBatchFrame(t, format)
			bytesPerSample := output.Layout.BytesPerSample
			const x = 16
			const y = 16
			const width = 16
			const height = 16
			stride := width + 5
			max := int((1 << tt.bitDepth) - 1)
			var got [prediction.CFLBufSquare]uint16
			var want [prediction.CFLBufSquare]uint16
			if bytesPerSample == 1 {
				src := make([]uint8, stride*height)
				for row := range height {
					for col := range width {
						value := uint8((17 + row*13 + col*19) & max)
						src[row*stride+col] = value
						setFrameWorkTestSample(output.Y, bytesPerSample, x+col, y+row, uint16(value))
					}
				}
				if err := prediction.SubsampleLuma8ToQ3(want[:], src, stride, width, height, tt.subX, tt.subY); err != nil {
					t.Fatal(err)
				}
			} else {
				src := make([]uint16, stride*height)
				for row := range height {
					for col := range width {
						value := uint16((257 + row*83 + col*41) & max)
						src[row*stride+col] = value
						setFrameWorkTestSample(output.Y, bytesPerSample, x+col, y+row, value)
					}
				}
				if err := prediction.SubsampleLuma16ToQ3(want[:], src, stride, width, height, tt.subX, tt.subY, tt.bitDepth); err != nil {
					t.Fatal(err)
				}
			}
			if err := frameWorkSubsampleLumaCFLQ3(got[:], new([prediction.CFLBufSquare]uint16), output.Y, bytesPerSample, tt.bitDepth, x, y, width, height, tt.subX, tt.subY); err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(got[:], want[:]) {
				t.Fatalf("subsample mismatch")
			}
		})
	}
}

func TestFrameWorkBatchPredictBlockChromaCFLRejectsInvalidInputs(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 64})
	ctx := testIntraPredictionBatch(output)
	valid := testCFLPredictionVisit()
	var scratch FrameWorkCFLPredictionScratch
	if err := ctx.PredictBlockChromaCFL(0, valid, nil); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("nil scratch err=%v want %v", err, ErrInvalidBatch)
	}
	if err := ctx.PredictBlockChromaCFL(0, testIntraPredictionVisit(tile.IntraModeDC), &scratch); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("non-cfl err=%v want %v", err, ErrInvalidBatch)
	}
	noAlpha := valid
	noAlpha.Prediction.CFLAlphaValid = false
	if err := ctx.PredictBlockChromaCFL(0, noAlpha, &scratch); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("missing alpha err=%v want %v", err, ErrInvalidBatch)
	}
}

func TestFrameWorkBatchPredictBlockChromaCFLAllocs(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 64})
	seedFrameWorkCFLPredictionFrame(output)
	ctx := testIntraPredictionBatch(output)
	visit := testCFLPredictionVisit()
	var scratch FrameWorkCFLPredictionScratch
	allocs := testing.AllocsPerRun(1000, func() {
		if err := ctx.PredictBlockChromaCFL(0, visit, &scratch); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("PredictBlockChromaCFL allocated: %f", allocs)
	}
}

func TestFrameWorkBatchPredictBlockAllocs(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, MonoChrome: true, Align: 64})
	reference := testBatchFrame(t, output.Format)
	fillFrameWorkInterReference(reference, 0xff)
	ctx := testInterPredictionBatch(output, reference)
	for x := 16; x < 32; x++ {
		setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, x, 15, 90)
	}
	for y := 16; y < 32; y++ {
		setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, 15, y, 92)
	}
	var scratch FrameWorkPredictionScratch
	intra := testIntraPredictionVisit(tile.IntraModeDC)
	inter := testInterPredictionVisit(motion.Vector{Col: 8, Row: 0})
	allocs := testing.AllocsPerRun(1000, func() {
		if err := ctx.PredictBlock(0, intra, &scratch); err != nil {
			t.Fatal(err)
		}
		if err := ctx.PredictBlock(0, inter, &scratch); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("PredictBlock allocated: %f", allocs)
	}
}

func TestFrameWorkBatchPredictBlockLumaInterCompoundAverageRejectsInvalidInputs(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, Align: 64})
	last := testBatchFrame(t, output.Format)
	bwd := testBatchFrame(t, output.Format)
	ctx := testCompoundInterPredictionBatch(output, last, bwd)
	valid := testCompoundInterPredictionVisit(motion.Vector{}, motion.Vector{}, tile.CompoundTypeAverage)
	var scratch FrameWorkInterPredictionScratch

	tests := []struct {
		name    string
		ctx     FrameWorkBatch
		visit   tile.BlockLoopVisit
		scratch *FrameWorkInterPredictionScratch
	}{
		{name: "nil scratch", ctx: ctx, visit: valid},
		{name: "missing blend", ctx: ctx, visit: func() tile.BlockLoopVisit {
			visit := valid
			visit.Prediction.CompoundBlendValid = false
			return visit
		}(), scratch: &scratch},
		{name: "wedge", ctx: ctx, visit: testCompoundInterPredictionVisit(motion.Vector{}, motion.Vector{}, tile.CompoundTypeWedge), scratch: &scratch},
		{name: "dist wtd without order hint", ctx: func() FrameWorkBatch {
			next := ctx
			next.Sequence.EnableOrderHint = false
			return next
		}(), visit: testCompoundInterPredictionVisit(motion.Vector{}, motion.Vector{}, tile.CompoundTypeDistWtd), scratch: &scratch},
		{name: "single ref", ctx: ctx, visit: func() tile.BlockLoopVisit {
			visit := valid
			refs := tile.InterReferencesResult{Ref0: tile.ReferenceFrameLast, Ref1: tile.ReferenceFrameNone}
			visit.Prediction.InterReferences = refs
			visit.Prediction.InterMotion.References = refs
			return visit
		}(), scratch: &scratch},
		{name: "inter intra", ctx: ctx, visit: func() tile.BlockLoopVisit {
			visit := valid
			visit.Prediction.InterIntraValid = true
			visit.Prediction.InterIntra.Enabled = true
			return visit
		}(), scratch: &scratch},
		{name: "intrabc", ctx: ctx, visit: func() tile.BlockLoopVisit {
			visit := valid
			visit.Prediction.Intrabc = true
			visit.Prediction.IntrabcValid = true
			return visit
		}(), scratch: &scratch},
		{name: "non translation motion mode", ctx: ctx, visit: func() tile.BlockLoopVisit {
			visit := valid
			visit.Prediction.MotionModeValid = true
			visit.Prediction.MotionMode = tile.MotionModeOBMC
			return visit
		}(), scratch: &scratch},
		{name: "missing second reference frame", ctx: func() FrameWorkBatch {
			next := ctx
			next.References = next.References[:int(FrameWorkReferenceBwd)]
			return next
		}(), visit: valid, scratch: &scratch},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := tt.ctx.PredictBlockLumaInterCompoundAverage(0, tt.visit, tt.scratch); !errors.Is(err, ErrInvalidBatch) {
				t.Fatalf("err=%v want %v", err, ErrInvalidBatch)
			}
		})
	}
}

func TestFrameWorkBatchPredictBlockLumaInterCompoundRejectsInvalidDiffWtdMask(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, Align: 64})
	last := testBatchFrame(t, output.Format)
	bwd := testBatchFrame(t, output.Format)
	fillFrameWorkInterReferenceVariant(last, 0xff, 11)
	fillFrameWorkInterReferenceVariant(bwd, 0xff, 97)
	ctx := testCompoundInterPredictionBatch(output, last, bwd)
	visit := testCompoundInterPredictionVisit(motion.Vector{}, motion.Vector{}, tile.CompoundTypeDiffWtd)
	visit.Prediction.CompoundBlend.MaskType = tile.DiffWtdMaskType(99)
	var scratch FrameWorkInterPredictionScratch
	if err := ctx.PredictBlockLumaInterCompoundWithFilters(0, visit, &scratch, motion.RegularFilters); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("err=%v want %v", err, ErrInvalidBatch)
	}
}

func TestFrameWorkBatchPredictBlockLumaInterCompoundAverageAllocs(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 10, Align: 128})
	last := testBatchFrame(t, output.Format)
	bwd := testBatchFrame(t, output.Format)
	fillFrameWorkInterReference(last, 0x3ff)
	fillFrameWorkInterReferenceVariant(bwd, 0x3ff, 17)
	ctx := testCompoundInterPredictionBatch(output, last, bwd)
	visit := testCompoundInterPredictionVisit(motion.Vector{Col: 4, Row: 6}, motion.Vector{Col: -1, Row: 3}, tile.CompoundTypeAverage)
	var scratch FrameWorkInterPredictionScratch
	filters := motion.InterpFilters{X: motion.InterpEightTapRegular, Y: motion.InterpEightTapSmooth}

	allocs := testing.AllocsPerRun(1000, func() {
		if err := ctx.PredictBlockLumaInterCompoundAverageWithFilters(0, visit, &scratch, filters); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("PredictBlockLumaInterCompoundAverage allocated: %f", allocs)
	}
}

func TestFrameWorkBatchPredictBlockInterCompoundDiffWtdAllocs(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 64})
	last := testBatchFrame(t, output.Format)
	bwd := testBatchFrame(t, output.Format)
	fillFrameWorkInterReferenceVariant(last, 0xff, 31)
	fillFrameWorkInterReferenceVariant(bwd, 0xff, 157)
	ctx := testCompoundInterPredictionBatch(output, last, bwd)
	visit := testCompoundInterPredictionVisit(motion.Vector{Col: 3, Row: 5}, motion.Vector{Col: -1, Row: 7}, tile.CompoundTypeDiffWtd)
	visit.Prediction.CompoundBlend.MaskType = tile.DiffWtdMaskType38
	var scratch FrameWorkInterPredictionScratch
	filters := motion.InterpFilters{X: motion.InterpEightTapRegular, Y: motion.InterpEightTapSmooth}

	allocs := testing.AllocsPerRun(1000, func() {
		if err := ctx.PredictBlockInterWithFilters(0, visit, &scratch, filters); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("PredictBlockInter diff-wtd allocated: %f", allocs)
	}
}

func TestFrameWorkBatchPredictBlockInterCompoundWedgeAllocs(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 64})
	last := testBatchFrame(t, output.Format)
	bwd := testBatchFrame(t, output.Format)
	fillFrameWorkInterReferenceVariant(last, 0xff, 31)
	fillFrameWorkInterReferenceVariant(bwd, 0xff, 157)
	ctx := testCompoundInterPredictionBatch(output, last, bwd)
	visit := testCompoundInterPredictionVisit(motion.Vector{Col: 3, Row: 5}, motion.Vector{Col: -1, Row: 7}, tile.CompoundTypeWedge)
	visit.Prediction.CompoundBlend.WedgeIndex = 0
	var scratch FrameWorkInterPredictionScratch
	filters := motion.InterpFilters{X: motion.InterpEightTapRegular, Y: motion.InterpEightTapSmooth}

	allocs := testing.AllocsPerRun(1000, func() {
		if err := ctx.PredictBlockInterWithFilters(0, visit, &scratch, filters); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("PredictBlockInter wedge allocated: %f", allocs)
	}
}

func TestFrameWorkBatchPredictBlockInterOBMCAllocs(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 64})
	reference := testBatchFrame(t, output.Format)
	fillFrameWorkInterReferenceAllPlanes(reference, 0xff)
	ctx := testInterPredictionBatch(output, reference)
	visit := testOBMCInterPredictionVisit(motion.Vector{}, motion.Vector{Row: 16}, motion.Vector{Col: 16})
	var scratch FrameWorkInterPredictionScratch
	allocs := testing.AllocsPerRun(1000, func() {
		if err := ctx.PredictBlockInterOBMCWithFilters(0, visit, &scratch, motion.RegularFilters); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("PredictBlockInter OBMC allocated: %f", allocs)
	}
}

func TestFrameWorkBatchPredictBlockLumaInterRejectsInvalidInputs(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, Align: 64})
	reference := testBatchFrame(t, output.Format)
	ctx := testInterPredictionBatch(output, reference)
	valid := testInterPredictionVisit(motion.Vector{})

	tests := []struct {
		name  string
		ctx   FrameWorkBatch
		visit tile.BlockLoopVisit
	}{
		{name: "missing prediction", ctx: ctx, visit: func() tile.BlockLoopVisit {
			visit := valid
			visit.Prediction.Valid = false
			return visit
		}()},
		{name: "intra", ctx: ctx, visit: func() tile.BlockLoopVisit {
			visit := valid
			visit.Prediction.Intra = true
			return visit
		}()},
		{name: "missing motion", ctx: ctx, visit: func() tile.BlockLoopVisit {
			visit := valid
			visit.Prediction.InterMotionValid = false
			return visit
		}()},
		{name: "compound", ctx: ctx, visit: func() tile.BlockLoopVisit {
			visit := valid
			visit.Prediction.InterMotion.References.Compound = true
			visit.Prediction.InterMotion.References.Ref1 = tile.ReferenceFrameBWD
			return visit
		}()},
		{name: "non translation motion mode", ctx: ctx, visit: func() tile.BlockLoopVisit {
			visit := valid
			visit.Prediction.MotionModeValid = true
			visit.Prediction.MotionMode = tile.MotionModeWarp
			return visit
		}()},
		{name: "inter intra", ctx: ctx, visit: func() tile.BlockLoopVisit {
			visit := valid
			visit.Prediction.InterIntraValid = true
			visit.Prediction.InterIntra.Enabled = true
			return visit
		}()},
		{name: "intrabc", ctx: ctx, visit: func() tile.BlockLoopVisit {
			visit := valid
			visit.Prediction.Intrabc = true
			visit.Prediction.IntrabcValid = true
			return visit
		}()},
		{name: "missing reference frame", ctx: func() FrameWorkBatch {
			next := ctx
			next.References = nil
			return next
		}(), visit: valid},
		{name: "switchable filter not decoded", ctx: func() FrameWorkBatch {
			next := ctx
			next.TileInfo.InterpolationFilter = parser.InterpolationSwitchable
			return next
		}(), visit: valid},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := tt.ctx.PredictBlockLumaInter(0, tt.visit); !errors.Is(err, ErrInvalidBatch) {
				t.Fatalf("err=%v want %v", err, ErrInvalidBatch)
			}
		})
	}
	// A block whose plane origin lands entirely beyond the output frame (for
	// example because the bitstream rounded the MI grid up past the visible
	// area) is silently skipped to match libaom's clip-to-visible behavior.
	t.Run("outside job", func(t *testing.T) {
		visit := valid
		visit.Block.MICol = 16
		visit.Block.MIColEnd = 20
		if err := ctx.PredictBlockLumaInter(0, visit); err != nil {
			t.Fatalf("outside job err=%v want nil", err)
		}
	})
	if err := ctx.PredictBlockLumaInterWithFilters(0, valid, motion.InterpFilters{X: motion.InterpFilter(99)}); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("bad filters err=%v want %v", err, ErrInvalidBatch)
	}
}

func TestFrameWorkBatchPredictBlockInterRejectsInterIntraBeforeMutation(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, Align: 64})
	reference := testBatchFrame(t, output.Format)
	output.Y.Pix[0] = 0x44
	output.U.Pix[0] = 0x55
	output.V.Pix[0] = 0x66

	ctx := testInterPredictionBatch(output, reference)
	visit := testInterPredictionVisit(motion.Vector{})
	visit.Prediction.InterIntraValid = true
	visit.Prediction.InterIntra.Enabled = true
	if err := ctx.PredictBlockInterWithFilters(0, visit, nil, motion.RegularFilters); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("PredictBlockInterWithFilters err=%v want %v", err, ErrInvalidBatch)
	}
	if output.Y.Pix[0] != 0x44 || output.U.Pix[0] != 0x55 || output.V.Pix[0] != 0x66 {
		t.Fatalf("output mutated y=%#x u=%#x v=%#x", output.Y.Pix[0], output.U.Pix[0], output.V.Pix[0])
	}
}

func TestFrameWorkBatchPredictBlockInterIntrabcCopiesOutput(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, MonoChrome: true, Align: 64})
	testFillFrame(output, 3)
	ctx := testInterPredictionBatch(output, output)
	for y := range 16 {
		for x := 16; x < 32; x++ {
			setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, x, y, uint16(40+y+x-16))
		}
	}

	visit := testInterPredictionVisit(motion.Vector{Row: -16 * 8})
	visit.Prediction.Intrabc = true
	visit.Prediction.IntrabcValid = true
	if err := ctx.PredictBlockInterWithFilters(0, visit, nil, motion.RegularFilters); err != nil {
		t.Fatal(err)
	}
	for y := 16; y < 32; y++ {
		for x := 16; x < 32; x++ {
			want := uint16(40 + y - 16 + x - 16)
			if got := frameWorkTestSample(output.Y, output.Layout.BytesPerSample, x, y); got != want {
				t.Fatalf("sample(%d,%d)=%d want %d", x, y, got, want)
			}
		}
	}
}

func TestFrameWorkBatchPredictBlockInterRoutesScaledReferenceWhenEnabled(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 64})
	reference := testBatchFrame(t, frame.Format{Width: 32, Height: 64, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 64})
	// Fill the reference with a uniform value: 8-tap kernels sum to 128 (1.0
	// in Q7), so a uniform reference must produce that same value at every
	// output sample regardless of scale factors or sub-pel phase.
	for y := 0; y < reference.Y.Height; y++ {
		for x := 0; x < reference.Y.Width; x++ {
			setFrameWorkTestSample(reference.Y, reference.Layout.BytesPerSample, x, y, 0xa5)
		}
	}
	for y := 0; y < reference.U.Height; y++ {
		for x := 0; x < reference.U.Width; x++ {
			setFrameWorkTestSample(reference.U, reference.Layout.BytesPerSample, x, y, 0x5a)
			setFrameWorkTestSample(reference.V, reference.Layout.BytesPerSample, x, y, 0x33)
		}
	}
	// Seed a sentinel byte that the scaled convolver must overwrite.
	output.Y.Pix[16*output.Y.Stride+16] = 0x00

	ctx := testInterPredictionBatch(output, reference)
	visit := testInterPredictionVisit(motion.Vector{})
	if err := ctx.PredictBlockInterWithFilters(0, visit, nil, motion.RegularFilters); err != nil {
		t.Fatalf("PredictBlockInterWithFilters err=%v", err)
	}
	if got := output.Y.Pix[16*output.Y.Stride+16]; got != 0xa5 {
		t.Fatalf("scaled prediction did not write uniform ref value at (16,16): got=%#x want=%#x", got, 0xa5)
	}
}

// TestFrameWorkBatchPredictBlockInterWarpFallsBackToScaledTranslation anchors
// the libaom behavior described in allow_warp() (av1/common/reconinter.c):
// when av1_is_scaled(sf) is true, allow_warp() returns 0 and the predictor
// mode stays TRANSLATION_PRED, so av1_make_inter_predictor() runs the scaled
// 8-tap convolver on the block-level MV instead of the warp matrix. The
// warp-dispatch output must therefore match the dedicated scaled translational
// helper byte-for-byte.
func TestFrameWorkBatchPredictBlockInterWarpFallsBackToScaledTranslation(t *testing.T) {
	format := frame.Format{Width: 64, Height: 64, BitDepth: 8, Align: 64}
	output := testBatchFrame(t, format)
	want := testBatchFrame(t, format)
	reference := testBatchFrame(t, frame.Format{Width: 32, Height: 64, BitDepth: 8, Align: 64})
	// Deterministic pseudo-random reference so the scaled convolver does
	// real work (no uniform shortcut); the equality check below catches
	// any drift between the warp-dispatch fallback and the libaom
	// reference (translational scaled convolver on the block-level MV).
	for y := 0; y < reference.Y.Height; y++ {
		for x := 0; x < reference.Y.Width; x++ {
			setFrameWorkTestSample(reference.Y, reference.Layout.BytesPerSample, x, y, uint16((y*31+x*17+5)&0xff))
		}
	}

	mv := motion.Vector{Row: 5, Col: -3}
	filters := motion.InterpFilters{X: motion.InterpEightTapRegular, Y: motion.InterpEightTapSmooth}

	// Drive the warp dispatch with a non-identity affine wm: per libaom,
	// the scaled-ref gate must downgrade this to translation and ignore
	// the warp matrix entirely.
	ctx := testInterPredictionBatch(output, reference)
	visit := testInterPredictionVisit(mv)
	visit.Prediction.MotionModeValid = true
	visit.Prediction.MotionMode = tile.MotionModeWarp
	params := parser.DefaultWarpedMotionParams()
	params.Type = parser.GlobalMotionAffine
	// Perturb the affine matrix so a real warp would diverge from
	// translation; the equality check below proves the matrix is unused.
	params.Matrix[2] = 1<<16 + 123
	params.Matrix[5] = 1<<16 - 123
	testMotionExtras(&visit.Prediction).WarpedMotion = tile.WarpedMotionModel{Params: params}
	visit.Prediction.WarpedMotionValid = true
	if err := ctx.PredictBlockLumaInterWithFilters(0, visit, filters); err != nil {
		t.Fatalf("PredictBlockLumaInterWithFilters err=%v", err)
	}

	// Build the libaom reference: translational scaled convolver on the
	// block-level MV with the same filters, written at the same plane
	// position.
	wantCtx := testInterPredictionBatch(want, reference)
	geom, ok, err := wantCtx.blockPredictionPlaneGeometry(0, visit.Block, FrameWorkPlaneY)
	if err != nil || !ok {
		t.Fatalf("plane geometry err=%v ok=%v", err, ok)
	}
	refPlane := frame.Plane{Pix: reference.Y.Pix, Stride: reference.Y.Stride, Width: reference.Y.Width, Height: reference.Y.Height}
	if err := frameWorkPredictScaledReferencePlane(geom, refPlane, geom.bytesPerSample(), want.Format.BitDepth,
		geom.X, geom.Y, geom.X, geom.Y, geom.width(), geom.height(), mv, geom.SubsamplingX, geom.SubsamplingY, filters); err != nil {
		t.Fatalf("frameWorkPredictScaledReferencePlane err=%v", err)
	}

	assertFrameWorkPlaneBlockEqual(t, output.Y, want.Y, output.Layout.BytesPerSample, geom.X, geom.Y, geom.width(), geom.height())
}

// TestFrameWorkBatchPredictBlockInterGlobalWarpFallsBackToScaledTranslation
// is the GLOBALMV analogue of the warp test above. allow_warp() applies the
// same is_scaled gate to frame-level global motion, so a non-translational
// global warp model paired with a scaled reference must also collapse to the
// translational scaled convolver on the block-level MV.
func TestFrameWorkBatchPredictBlockInterGlobalWarpFallsBackToScaledTranslation(t *testing.T) {
	format := frame.Format{Width: 64, Height: 64, BitDepth: 8, Align: 64}
	output := testBatchFrame(t, format)
	want := testBatchFrame(t, format)
	reference := testBatchFrame(t, frame.Format{Width: 32, Height: 64, BitDepth: 8, Align: 64})
	for y := 0; y < reference.Y.Height; y++ {
		for x := 0; x < reference.Y.Width; x++ {
			setFrameWorkTestSample(reference.Y, reference.Layout.BytesPerSample, x, y, uint16((y*23+x*13+7)&0xff))
		}
	}

	mv := motion.Vector{Row: -4, Col: 6}
	filters := motion.InterpFilters{X: motion.InterpEightTapRegular, Y: motion.InterpEightTapRegular}

	ctx := testInterPredictionBatch(output, reference)
	visit := testInterPredictionVisit(mv)
	// MotionMode stays SIMPLE_TRANSLATION so dispatch falls through to
	// the GlobalMV branch; the scaled-ref gate must override the global
	// warp model just like it does for block-level warp.
	visit.Prediction.MotionModeValid = true
	visit.Prediction.MotionMode = tile.MotionModeTranslation
	params := parser.DefaultWarpedMotionParams()
	params.Type = parser.GlobalMotionAffine
	params.Matrix[2] = 1<<16 + 77
	params.Matrix[5] = 1<<16 - 77
	testMotionExtras(&visit.Prediction).GlobalWarpedMotion = tile.WarpedMotionModel{Params: params}
	visit.Prediction.GlobalWarpedMotionValid = true
	if err := ctx.PredictBlockLumaInterWithFilters(0, visit, filters); err != nil {
		t.Fatalf("PredictBlockLumaInterWithFilters err=%v", err)
	}

	wantCtx := testInterPredictionBatch(want, reference)
	geom, ok, err := wantCtx.blockPredictionPlaneGeometry(0, visit.Block, FrameWorkPlaneY)
	if err != nil || !ok {
		t.Fatalf("plane geometry err=%v ok=%v", err, ok)
	}
	refPlane := frame.Plane{Pix: reference.Y.Pix, Stride: reference.Y.Stride, Width: reference.Y.Width, Height: reference.Y.Height}
	if err := frameWorkPredictScaledReferencePlane(geom, refPlane, geom.bytesPerSample(), want.Format.BitDepth,
		geom.X, geom.Y, geom.X, geom.Y, geom.width(), geom.height(), mv, geom.SubsamplingX, geom.SubsamplingY, filters); err != nil {
		t.Fatalf("frameWorkPredictScaledReferencePlane err=%v", err)
	}

	assertFrameWorkPlaneBlockEqual(t, output.Y, want.Y, output.Layout.BytesPerSample, geom.X, geom.Y, geom.width(), geom.height())
}

// TestFrameWorkBatchPredictBlockInterOBMCScaledRoutesNeighborThroughScaledConvolver
// pins the libaom behaviour described in dec_calc_subpel_params /
// av1_make_inter_predictor (av1/common/reconinter.c) for OBMC neighbor
// predictions against scaled references. The main block prediction already
// dispatches through the scaled convolver, but each OBMC overlappable
// neighbor prediction also has to go through the same scaled path —
// otherwise the neighbor area is filled from the wrong source-plane
// coordinates and the OBMC blend writes garbage into the leftmost / topmost
// overlap columns of the block.
//
// Concretely: with a 32x32 reference scaled into a 64x64 output, OBMC LEFT
// for the central 16x16 block at (16, 16) must produce the same leftmost
// 8 columns as a plain scaled translational predictor on the LEFT
// neighbor's MV, then blended into the base prediction via the 8-wide
// OBMC mask. Before the fix predictInterReferenceAreaToScratch dropped
// to the same-size convolver and corrupted those 8 columns; the SVC L2T1
// spatial=1 enhancement layer flagged it at dst (96..103, 0) of frame 0.
func TestFrameWorkBatchPredictBlockInterOBMCScaledRoutesNeighborThroughScaledConvolver(t *testing.T) {
	format := frame.Format{Width: 64, Height: 64, BitDepth: 8, MonoChrome: true, Align: 64}
	output := testBatchFrame(t, format)
	want := testBatchFrame(t, format)
	reference := testBatchFrame(t, frame.Format{Width: 32, Height: 32, BitDepth: 8, MonoChrome: true, Align: 64})
	// Deterministic pseudo-random reference so the scaled convolver does
	// real work and an off-by-scale neighbor read would diverge sample-
	// for-sample from the libaom reference.
	for y := 0; y < reference.Y.Height; y++ {
		for x := 0; x < reference.Y.Width; x++ {
			setFrameWorkTestSample(reference.Y, reference.Layout.BytesPerSample, x, y, uint16((y*29+x*19+11)&0xff))
		}
	}

	baseMV := motion.Vector{Row: 4, Col: -6}
	leftMV := motion.Vector{Row: -3, Col: 11}
	filters := motion.InterpFilters{X: motion.InterpEightTapRegular, Y: motion.InterpEightTapSmooth}

	ctx := testInterPredictionBatch(output, reference)
	// Single LEFT neighbor with a distinct MV so the OBMC blend reads
	// from a different source-plane area than the base prediction.
	visit := testInterPredictionVisit(baseMV)
	visit.Prediction.MotionMode = tile.MotionModeOBMC
	visit.Prediction.MotionModeValid = true
	visit.Prediction.OverlappableNeighborsValid = true
	testMotionExtras(&visit.Prediction).OverlappableNeighbors.LeftCount = 1
	testMotionExtras(&visit.Prediction).OverlappableNeighbors.Left[0] = tile.OverlappableNeighbor{
		RelY4: 0,
		Span4: 4,
		Size:  tile.BlockSize16x16,
		Motion: tile.InterMotionResult{
			References: visit.Prediction.InterMotion.References,
			MV:         [2]motion.Vector{leftMV},
		},
		InterpFilters: filters,
	}
	var scratch FrameWorkInterPredictionScratch
	if err := ctx.PredictBlockLumaInterOBMCWithFilters(0, visit, &scratch, filters); err != nil {
		t.Fatalf("PredictBlockLumaInterOBMCWithFilters err=%v", err)
	}

	// libaom reference: scaled translational predictor for the base block,
	// then a scaled neighbor prediction blended into the leftmost overlap
	// columns via the OBMC mask.
	wantCtx := testInterPredictionBatch(want, reference)
	geom, ok, err := wantCtx.blockPredictionPlaneGeometry(0, visit.Block, FrameWorkPlaneY)
	if err != nil || !ok {
		t.Fatalf("plane geometry err=%v ok=%v", err, ok)
	}
	refPlane := frame.Plane{Pix: reference.Y.Pix, Stride: reference.Y.Stride, Width: reference.Y.Width, Height: reference.Y.Height}
	if err := frameWorkPredictScaledReferencePlane(geom, refPlane, geom.bytesPerSample(), want.Format.BitDepth,
		geom.X, geom.Y, geom.X, geom.Y, geom.width(), geom.height(), baseMV, geom.SubsamplingX, geom.SubsamplingY, filters); err != nil {
		t.Fatalf("base scaled-ref err=%v", err)
	}
	overlap, err := frameWorkOBMCLeftWidth(visit.Block.Size, geom)
	if err != nil {
		t.Fatalf("OBMC left width err=%v", err)
	}
	if overlap <= 0 || overlap > geom.width() {
		t.Fatalf("unexpected overlap=%d width=%d", overlap, geom.width())
	}
	height, ok := frameWorkOBMCPlaneSpan(testMotionExtras(&visit.Prediction).OverlappableNeighbors.Left[0].Span4, geom.SubsamplingY)
	if !ok {
		t.Fatalf("OBMC plane span invalid")
	}
	if height > geom.height() {
		height = geom.height()
	}
	var neighborScratch FrameWorkInterPredictionScratch
	neighbor, err := frameWorkInterScratchPlane(neighborScratch.First[:], geom.bytesPerSample(), geom.width(), geom.height())
	if err != nil {
		t.Fatalf("neighbor scratch err=%v", err)
	}
	// Neighbor prediction must go through the scaled convolver anchored
	// to the same geom.Output dimensions as the base prediction; this is
	// what predictInterReferenceAreaToScratch must do under the fix.
	if err := frameWorkPredictScaledReferencePlaneWithDims(neighbor, refPlane, geom.Output.Width, geom.Output.Height,
		geom.bytesPerSample(), want.Format.BitDepth, 0, 0, geom.X, geom.Y, overlap, height, leftMV,
		geom.SubsamplingX, geom.SubsamplingY, filters); err != nil {
		t.Fatalf("neighbor scaled-ref err=%v", err)
	}
	mask, ok := frameWorkOBMCMask(overlap)
	if !ok {
		t.Fatalf("missing OBMC mask for overlap=%d", overlap)
	}
	if err := frameWorkBlendOBMCH(want.Y, neighbor, geom.bytesPerSample(), geom.X, geom.Y, 0, 0, overlap, height, mask); err != nil {
		t.Fatalf("blend OBMC H err=%v", err)
	}

	assertFrameWorkPlaneBlockEqual(t, output.Y, want.Y, output.Layout.BytesPerSample, geom.X, geom.Y, geom.width(), geom.height())
}

// TestFrameWorkBatchPredictBlockInterWarpScaledAllocs locks the steady-state
// allocation profile for the warp+scaled fallback path so the SVC vectors
// stay zero-alloc per visit.
func TestFrameWorkBatchPredictBlockInterWarpScaledAllocs(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, Align: 64})
	reference := testBatchFrame(t, frame.Format{Width: 32, Height: 64, BitDepth: 8, Align: 64})
	fillFrameWorkInterReference(reference, 0xff)
	ctx := testInterPredictionBatch(output, reference)
	visit := testInterPredictionVisit(motion.Vector{Row: 3, Col: -5})
	visit.Prediction.MotionModeValid = true
	visit.Prediction.MotionMode = tile.MotionModeWarp
	params := parser.DefaultWarpedMotionParams()
	params.Type = parser.GlobalMotionAffine
	testMotionExtras(&visit.Prediction).WarpedMotion = tile.WarpedMotionModel{Params: params}
	visit.Prediction.WarpedMotionValid = true
	filters := motion.InterpFilters{X: motion.InterpEightTapRegular, Y: motion.InterpEightTapSmooth}
	allocs := testing.AllocsPerRun(1000, func() {
		if err := ctx.PredictBlockLumaInterWithFilters(0, visit, filters); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("warp+scaled fallback allocated: %f", allocs)
	}
}

// TestFrameWorkBatchPredictBlockInterIntraScaledMatchesScaledTranslation
// anchors the libaom behavior described in av1_combine_interintra() and
// av1_make_inter_predictor() (av1/common/reconinter.c): inter-intra blocks
// stage the inter side into a block-sized scratch buffer, and when the
// reference is scaled the scaled 8-tap convolver runs there. The Q14
// reference scale factors come from xd->block_ref_scale_factors, which are
// derived per-frame from the *output frame size*, not from the staging
// buffer. The blend mask and intra side are independent of scaling.
//
// Concretely: the inter-intra luma output for a scaled-reference block must
// equal the same per-sample mask blend of (a) the scaled translational
// predictor on the block MV anchored to the output frame, and (b) the intra
// predictor for the configured mode. This test pins that equivalence
// sample-for-sample on a deterministic 32x32 -> 64x64 reference. If the
// staging path were to derive scale factors from the scratch buffer (16x16)
// instead of the output frame (64x64) the resulting samples would diverge
// from the libaom reference.
func TestFrameWorkBatchPredictBlockInterIntraScaledMatchesScaledTranslation(t *testing.T) {
	format := frame.Format{Width: 64, Height: 64, BitDepth: 8, MonoChrome: true, Align: 64}
	output := testBatchFrame(t, format)
	want := testBatchFrame(t, format)
	reference := testBatchFrame(t, frame.Format{Width: 32, Height: 32, BitDepth: 8, MonoChrome: true, Align: 64})
	// Deterministic pseudo-random reference so the scaled convolver does
	// real work and any drift between the geom-anchored and (buggy)
	// scratch-anchored scale factors surfaces sample-for-sample.
	for y := 0; y < reference.Y.Height; y++ {
		for x := 0; x < reference.Y.Width; x++ {
			setFrameWorkTestSample(reference.Y, reference.Layout.BytesPerSample, x, y, uint16((y*29+x*19+11)&0xff))
		}
	}
	// Seed the intra-side neighbors at the block boundary so the smooth
	// intra predictor produces a non-trivial signal that the mask must
	// actually blend (otherwise inter and intra branches could coincide
	// and hide the scaled-vs-scratch dimension bug).
	for x := 16; x < 32; x++ {
		setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, x, 15, 200)
		setFrameWorkTestSample(want.Y, want.Layout.BytesPerSample, x, 15, 200)
	}
	for y := 16; y < 32; y++ {
		setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, 15, y, 40)
		setFrameWorkTestSample(want.Y, want.Layout.BytesPerSample, 15, y, 40)
	}
	setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, 15, 15, 120)
	setFrameWorkTestSample(want.Y, want.Layout.BytesPerSample, 15, 15, 120)

	mv := motion.Vector{Row: 4, Col: -6}
	filters := motion.InterpFilters{X: motion.InterpEightTapRegular, Y: motion.InterpEightTapSmooth}

	ctx := testInterPredictionBatch(output, reference)
	visit := testInterIntraPredictionVisit(tile.InterIntraModeSmooth, false)
	visit.Prediction.InterMotion.MV[0] = mv
	var scratch FrameWorkInterPredictionScratch
	if err := ctx.PredictBlockInterWithFilters(0, visit, &scratch, filters); err != nil {
		t.Fatalf("PredictBlockInterWithFilters err=%v", err)
	}

	// Build the libaom reference: scaled translational predictor anchored
	// to the output frame size, blended with the configured intra mode via
	// the same per-sample mask. This drives the same internal helpers as
	// the inter-intra dispatch except for the (geom-anchored) scale factor
	// derivation, which is the new path under test.
	wantCtx := testInterPredictionBatch(want, reference)
	geom, ok, err := wantCtx.blockPredictionPlaneGeometry(0, visit.Block, FrameWorkPlaneY)
	if err != nil || !ok {
		t.Fatalf("plane geometry err=%v ok=%v", err, ok)
	}
	var interBuf, intraBuf FrameWorkInterPredictionScratch
	inter, err := frameWorkInterScratchPlane(interBuf.First[:], geom.bytesPerSample(), geom.width(), geom.height())
	if err != nil {
		t.Fatalf("inter scratch err=%v", err)
	}
	intra, err := frameWorkInterScratchPlane(intraBuf.Second[:], geom.bytesPerSample(), geom.width(), geom.height())
	if err != nil {
		t.Fatalf("intra scratch err=%v", err)
	}
	refPlane := frame.Plane{Pix: reference.Y.Pix, Stride: reference.Y.Stride, Width: reference.Y.Width, Height: reference.Y.Height}
	if err := frameWorkPredictScaledReferencePlaneToBuffer(inter, refPlane, geom, want.Format.BitDepth,
		0, 0, geom.X, geom.Y, mv, filters); err != nil {
		t.Fatalf("scaled-ref to buffer err=%v", err)
	}
	edgeBlock := frameWorkPredictionPlaneEdgeBlock(visit.Block, geom)
	var intraScratch FrameWorkIntraPredictionScratch
	edges, err := frameWorkIntraPredictionEdgesWithExtent(geom.Output, geom.bytesPerSample(), want.Format.BitDepth, geom.X, geom.Y, geom.width(), geom.height(), geom.width(), geom.height(), 0, 0, edgeBlock, &intraScratch, true)
	if err != nil {
		t.Fatalf("intra edges err=%v", err)
	}
	if err := prediction.PredictIntraPlaneBlock(intra, geom.bytesPerSample(), want.Format.BitDepth, 0, 0, geom.width(), geom.height(), prediction.IntraModeSmooth, edges); err != nil {
		t.Fatalf("intra predict err=%v", err)
	}
	var mask [frameWorkInterPredictionMaxMaskSamples]byte
	if err := frameWorkBuildInterIntraMask(mask[:], geom.width(), geom.width(), geom.height(), tile.InterIntraModeSmooth); err != nil {
		t.Fatalf("build mask err=%v", err)
	}
	if err := frameWorkBlendInterIntraBlock(geom.Output, inter, intra, geom.bytesPerSample(), want.Format.BitDepth, geom.X, geom.Y, geom.width(), geom.height(), mask[:], geom.width(), false, false); err != nil {
		t.Fatalf("blend err=%v", err)
	}

	assertFrameWorkPlaneBlockEqual(t, output.Y, want.Y, output.Layout.BytesPerSample, geom.X, geom.Y, geom.width(), geom.height())
}

// TestFrameWorkBatchPredictBlockInterIntraScaledAllocs locks the steady-state
// allocation profile for the inter-intra+scaled fallback path so the SVC
// vectors stay zero-alloc per visit (the inter-intra branch stages into the
// caller-owned scratch and the scaled convolver writes directly into it).
func TestFrameWorkBatchPredictBlockInterIntraScaledAllocs(t *testing.T) {
	format := frame.Format{Width: 64, Height: 64, BitDepth: 8, MonoChrome: true, Align: 64}
	output := testBatchFrame(t, format)
	reference := testBatchFrame(t, frame.Format{Width: 32, Height: 32, BitDepth: 8, MonoChrome: true, Align: 64})
	fillFrameWorkInterReference(reference, 0xff)
	ctx := testInterPredictionBatch(output, reference)
	visit := testInterIntraPredictionVisit(tile.InterIntraModeSmooth, false)
	visit.Prediction.InterMotion.MV[0] = motion.Vector{Row: 3, Col: -2}
	filters := motion.InterpFilters{X: motion.InterpEightTapRegular, Y: motion.InterpEightTapSmooth}
	var scratch FrameWorkInterPredictionScratch
	allocs := testing.AllocsPerRun(1000, func() {
		if err := ctx.PredictBlockInterWithFilters(0, visit, &scratch, filters); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("inter-intra+scaled fallback allocated: %f", allocs)
	}
}

func TestFrameWorkBatchPredictBlockInterCompoundScaledReferenceUsesConvBuf(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, Align: 64})
	last := testBatchFrame(t, output.Format)
	bwd := testBatchFrame(t, frame.Format{Width: 32, Height: 64, BitDepth: 8, Align: 64})
	fillFrameWorkInterReferenceVariant(last, 0xff, 23)
	fillFrameWorkInterReferenceVariant(bwd, 0xff, 197)

	ctx := testCompoundInterPredictionBatch(output, last, bwd)
	mv0 := motion.Vector{Col: 3, Row: 5}
	mv1 := motion.Vector{Col: 7, Row: -3}
	filters := motion.InterpFilters{X: motion.InterpEightTapSmooth, Y: motion.InterpMultiTapSharp}
	visit := testCompoundInterPredictionVisit(mv0, mv1, tile.CompoundTypeAverage)
	var scratch FrameWorkInterPredictionScratch
	if err := ctx.PredictBlockLumaInterCompoundWithFilters(0, visit, &scratch, filters); err != nil {
		t.Fatal(err)
	}

	first := testFrameWorkCompoundConvBuf(t, last.Y, output.Layout.BytesPerSample, output.Format.BitDepth, 16, 16, 16, 16, mv0, false, false, filters)
	second := testFrameWorkScaledCompoundConvBuf(t, bwd.Y, output.Format.Width, output.Format.Height, output.Layout.BytesPerSample, output.Format.BitDepth, 16, 16, 16, 16, mv1, false, false, filters)
	fwdOffset, bckOffset, err := ctx.frameWorkCompoundOffsets(visit.Prediction.InterMotion.References, visit.Prediction.CompoundBlend)
	if err != nil {
		t.Fatal(err)
	}
	assertFrameWorkCompoundBlendEqual(t, output.Y, first, second, output.Layout.BytesPerSample, output.Format.BitDepth, 16, 16, 16, 16, fwdOffset, bckOffset)
}

func TestFrameWorkBatchPredictBlockLumaInterAllocs(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 10, Align: 128})
	reference := testBatchFrame(t, output.Format)
	fillFrameWorkInterReference(reference, 0x3ff)
	ctx := testInterPredictionBatch(output, reference)
	visit := testInterPredictionVisit(motion.Vector{Col: 4, Row: 6})
	filters := motion.InterpFilters{X: motion.InterpEightTapRegular, Y: motion.InterpEightTapSmooth}
	allocs := testing.AllocsPerRun(1000, func() {
		if err := ctx.PredictBlockLumaInterWithFilters(0, visit, filters); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("PredictBlockLumaInter allocated: %f", allocs)
	}
}

func TestFrameWorkBatchPredictBlockLumaDispatchesIntraAndInter(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, Align: 64})
	reference := testBatchFrame(t, output.Format)
	fillFrameWorkInterReference(reference, 0xff)
	ctx := testInterPredictionBatch(output, reference)

	for x := 16; x < 32; x++ {
		setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, x, 15, 10)
	}
	for y := 16; y < 32; y++ {
		setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, 15, y, 50)
	}

	var scratch FrameWorkPredictionScratch
	if err := ctx.PredictBlockLuma(0, testIntraPredictionVisit(tile.IntraModeDC), &scratch); err != nil {
		t.Fatal(err)
	}
	if got := frameWorkTestSample(output.Y, output.Layout.BytesPerSample, 16, 16); got != 30 {
		t.Fatalf("intra dispatch sample=%d want 30", got)
	}

	inter := testInterPredictionVisit(motion.Vector{Col: 8, Row: 0})
	if err := ctx.PredictBlockLuma(0, inter, nil); err != nil {
		t.Fatal(err)
	}
	got := frameWorkTestSample(output.Y, output.Layout.BytesPerSample, 16, 16)
	want := frameWorkTestSample(reference.Y, reference.Layout.BytesPerSample, 17, 16)
	if got != want {
		t.Fatalf("inter dispatch sample=%d want %d", got, want)
	}
}

func TestFrameWorkBatchPredictBlockLumaRejectsInvalidDispatch(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, Align: 64})
	reference := testBatchFrame(t, output.Format)
	ctx := testInterPredictionBatch(output, reference)
	if err := ctx.PredictBlockLuma(0, tile.BlockLoopVisit{}, nil); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("missing prediction err=%v want %v", err, ErrInvalidBatch)
	}
	if err := ctx.PredictBlockLuma(0, testIntraPredictionVisit(tile.IntraModeDC), nil); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("nil intra scratch err=%v want %v", err, ErrInvalidBatch)
	}
	if err := ctx.PredictBlockLuma(0, func() tile.BlockLoopVisit {
		visit := testCompoundInterPredictionVisit(motion.Vector{}, motion.Vector{}, tile.CompoundTypeAverage)
		return visit
	}(), nil); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("nil compound scratch err=%v want %v", err, ErrInvalidBatch)
	}
}

func TestFrameWorkBatchPredictBlockLumaAllocs(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, Align: 64})
	reference := testBatchFrame(t, output.Format)
	fillFrameWorkInterReference(reference, 0xff)
	ctx := testInterPredictionBatch(output, reference)
	for x := 16; x < 32; x++ {
		setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, x, 15, 90)
	}
	for y := 16; y < 32; y++ {
		setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, 15, y, 92)
	}
	var scratch FrameWorkPredictionScratch
	intra := testIntraPredictionVisit(tile.IntraModeDC)
	inter := testInterPredictionVisit(motion.Vector{Col: 8, Row: 0})
	allocs := testing.AllocsPerRun(1000, func() {
		if err := ctx.PredictBlockLuma(0, intra, &scratch); err != nil {
			t.Fatal(err)
		}
		if err := ctx.PredictBlockLuma(0, inter, &scratch); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("PredictBlockLuma allocated: %f", allocs)
	}
}

func TestFrameWorkBatchPredictBlockLumaIntraRejectsInvalidInputs(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, Align: 64})
	ctx := testIntraPredictionBatch(output)
	valid := testIntraPredictionVisit(tile.IntraModeDC)
	var scratch FrameWorkIntraPredictionScratch

	tests := []struct {
		name    string
		ctx     FrameWorkBatch
		visit   tile.BlockLoopVisit
		scratch *FrameWorkIntraPredictionScratch
	}{
		{name: "nil scratch", ctx: ctx, visit: valid},
		{name: "missing prediction", ctx: ctx, visit: func() tile.BlockLoopVisit {
			visit := valid
			visit.Prediction.Valid = false
			return visit
		}(), scratch: &scratch},
		{name: "inter", ctx: ctx, visit: func() tile.BlockLoopVisit {
			visit := valid
			visit.Prediction.Intra = false
			return visit
		}(), scratch: &scratch},
		{name: "outside job", ctx: ctx, visit: func() tile.BlockLoopVisit {
			visit := valid
			visit.Block.MICol = 16
			visit.Block.MIColEnd = 20
			return visit
		}(), scratch: &scratch},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := tt.ctx.PredictBlockLumaIntra(0, tt.visit, tt.scratch); !errors.Is(err, ErrInvalidBatch) {
				t.Fatalf("err=%v want %v", err, ErrInvalidBatch)
			}
		})
	}
}

func TestFrameWorkBatchPredictBlockLumaIntraAllocs(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, Align: 64})
	ctx := testIntraPredictionBatch(output)
	for x := 16; x < 32; x++ {
		setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, x, 15, 90)
	}
	for y := 16; y < 32; y++ {
		setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, 15, y, 92)
	}
	setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, 15, 15, 91)

	var scratch FrameWorkIntraPredictionScratch
	visit := testIntraPredictionVisit(tile.IntraModeDC)
	directional := testIntraPredictionVisit(tile.IntraModeD135)
	allocs := testing.AllocsPerRun(1000, func() {
		if err := ctx.PredictBlockLumaIntra(0, visit, &scratch); err != nil {
			t.Fatal(err)
		}
		if err := ctx.PredictBlockLumaIntra(0, directional, &scratch); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("PredictBlockLumaIntra allocated: %f", allocs)
	}
}

func FuzzFrameWorkBatchPredictBlockLumaIntra(f *testing.F) {
	f.Add(uint8(0), uint16(90), uint16(92), uint16(88))
	f.Add(uint8(1), uint16(12), uint16(200), uint16(40))
	f.Add(uint8(6), uint16(511), uint16(700), uint16(512))

	modes := [...]tile.IntraMode{
		tile.IntraModeDC,
		tile.IntraModeVertical,
		tile.IntraModeHorizontal,
		tile.IntraModeSmooth,
		tile.IntraModeSmoothVertical,
		tile.IntraModeSmoothHorizontal,
		tile.IntraModePaeth,
		tile.IntraModeD45,
		tile.IntraModeD67,
		tile.IntraModeD113,
		tile.IntraModeD135,
		tile.IntraModeD157,
		tile.IntraModeD203,
	}
	f.Fuzz(func(t *testing.T, rawMode uint8, above uint16, left uint16, aboveLeft uint16) {
		output := testBatchFrame(t, frame.Format{Width: 96, Height: 96, BitDepth: 10, Align: 128})
		ctx := testIntraPredictionBatch(output)
		above &= 0x3ff
		left &= 0x3ff
		aboveLeft &= 0x3ff
		seedFrameWorkDirectionalEdges(output, 0x3ff, above, left, aboveLeft)

		var scratch FrameWorkIntraPredictionScratch
		visit := testIntraPredictionVisit(modes[int(rawMode)%len(modes)])
		if err := ctx.PredictBlockLumaIntra(0, visit, &scratch); err != nil {
			t.Fatalf("PredictBlockLumaIntra err=%v", err)
		}
		for y := 16; y < 32; y++ {
			for x := 16; x < 32; x++ {
				if got := frameWorkTestSample(output.Y, output.Layout.BytesPerSample, x, y); got > 0x3ff {
					t.Fatalf("sample(%d,%d)=%d exceeds 10-bit max", x, y, got)
				}
			}
		}
	})
}

func testCFLPredictionVisit() tile.BlockLoopVisit {
	visit := testIntraPredictionVisit(tile.IntraModeDC)
	visit.Prediction.ChromaMode = tile.ChromaIntraModeCFL
	visit.Prediction.ChromaModeValid = true
	visit.Prediction.CFLAlpha = tile.CFLAlphaResult{Index: 0x21, JointSign: 7}
	visit.Prediction.CFLAlphaValid = true
	return visit
}

func seedFrameWorkCFLPredictionFrame(output *frame.Frame) {
	for y := 16; y < 32; y++ {
		for x := 16; x < 32; x++ {
			value := uint16(35 + ((x-16)*7+(y-16)*11)%180)
			setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, x, y, value)
		}
	}
	for x := 8; x < 16; x++ {
		setFrameWorkTestSample(output.U, output.Layout.BytesPerSample, x, 7, uint16(50+x))
		setFrameWorkTestSample(output.V, output.Layout.BytesPerSample, x, 7, uint16(70+x))
	}
	for y := 8; y < 16; y++ {
		setFrameWorkTestSample(output.U, output.Layout.BytesPerSample, 7, y, uint16(90+y))
		setFrameWorkTestSample(output.V, output.Layout.BytesPerSample, 7, y, uint16(110+y))
	}
	setFrameWorkTestSample(output.U, output.Layout.BytesPerSample, 7, 7, 80)
	setFrameWorkTestSample(output.V, output.Layout.BytesPerSample, 7, 7, 100)
}

func testPredictFrameWorkCFLWant(t *testing.T, output *frame.Frame, visit tile.BlockLoopVisit, plane FrameWorkPlane) {
	t.Helper()
	var dst frame.Plane
	var predType prediction.CFLPredType
	switch plane {
	case FrameWorkPlaneU:
		dst = output.U
		predType = prediction.CFLPredU
	case FrameWorkPlaneV:
		dst = output.V
		predType = prediction.CFLPredV
	default:
		t.Fatalf("bad plane=%d", plane)
	}
	recon := make([]uint16, prediction.CFLBufSquare)
	ac := make([]int16, prediction.CFLBufSquare)
	offset := 16*output.Y.Stride + 16
	if err := prediction.SubsampleLuma8ToQ3(recon, output.Y.Pix[offset:], output.Y.Stride, 16, 16, true, true); err != nil {
		t.Fatal(err)
	}
	if err := prediction.SubtractCFLAverage(recon, ac, 8, 8); err != nil {
		t.Fatal(err)
	}
	edges := testFrameWorkIntraEdges(output, dst, 8, 8, 8, 8)
	if err := prediction.PredictIntraPlaneBlock(dst, output.Layout.BytesPerSample, output.Format.BitDepth, 8, 8, 8, 8, prediction.IntraModeDC, edges); err != nil {
		t.Fatal(err)
	}
	alphaQ3, err := prediction.CFLAlphaQ3(visit.Prediction.CFLAlpha.Index, visit.Prediction.CFLAlpha.JointSign, predType)
	if err != nil {
		t.Fatal(err)
	}
	if err := prediction.PredictCFLPlaneBlock(dst, output.Layout.BytesPerSample, output.Format.BitDepth, 8, 8, 8, 8, ac, alphaQ3); err != nil {
		t.Fatal(err)
	}
}

func testFrameWorkIntraEdges(output *frame.Frame, plane frame.Plane, x int, y int, width int, height int) prediction.IntraEdges {
	above := make([]uint16, width)
	left := make([]uint16, height)
	for col := range width {
		above[col] = frameWorkTestSample(plane, output.Layout.BytesPerSample, x+col, y-1)
	}
	for row := range height {
		left[row] = frameWorkTestSample(plane, output.Layout.BytesPerSample, x-1, y+row)
	}
	return prediction.IntraEdges{
		Above:              above,
		Left:               left,
		AboveAvailable:     true,
		LeftAvailable:      true,
		AboveLeft:          frameWorkTestSample(plane, output.Layout.BytesPerSample, x-1, y-1),
		AboveLeftAvailable: true,
	}
}

func testIntraPredictionBatch(output *frame.Frame) FrameWorkBatch {
	return FrameWorkBatch{
		Output: output,
		FrameWorkFrameContext: FrameWorkFrameContext{
			Sequence: FrameWorkSequenceContextFromHeader(parser.SequenceHeader{
				ColorConfig: parser.ColorConfig{
					BitDepth:     output.Format.BitDepth,
					MonoChrome:   output.Format.MonoChrome,
					SubsamplingX: output.Format.SubsamplingX,
					SubsamplingY: output.Format.SubsamplingY,
				},
			}),
			FrameSize: parser.FrameSize{CodedWidth: uint32(output.Format.Width), Height: uint32(output.Format.Height)},
		},
		Jobs:     []tile.Job{{SBCols: 1, SBRows: 1}},
		payloads: &testPayloads,
	}
}

func testInterPredictionBatch(output *frame.Frame, reference *frame.Frame) FrameWorkBatch {
	ctx := FrameWorkBatch{
		Output:     output,
		References: []*frame.Frame{reference},
		FrameWorkFrameContext: FrameWorkFrameContext{
			Sequence: FrameWorkSequenceContextFromHeader(parser.SequenceHeader{
				ColorConfig: parser.ColorConfig{
					BitDepth:     output.Format.BitDepth,
					MonoChrome:   output.Format.MonoChrome,
					SubsamplingX: output.Format.SubsamplingX,
					SubsamplingY: output.Format.SubsamplingY,
				},
			}),
			FrameSize: parser.FrameSize{CodedWidth: uint32(output.Format.Width), Height: uint32(output.Format.Height)},
			TileInfo:  parser.TileInfo{InterpolationFilter: parser.InterpolationEightTap},
		},
		Jobs:     []tile.Job{{SBCols: 1, SBRows: 1}},
		payloads: &testPayloads,
	}
	return ctx
}

func testCompoundInterPredictionBatch(output *frame.Frame, last *frame.Frame, bwd *frame.Frame) FrameWorkBatch {
	ctx := testInterPredictionBatch(output, last)
	refs := make([]*frame.Frame, int(FrameWorkReferenceBwd)+1)
	refs[int(FrameWorkReferenceLast)] = last
	refs[int(FrameWorkReferenceBwd)] = bwd
	ctx.References = refs
	ctx.Sequence.EnableOrderHint = true
	ctx.Sequence.OrderHintBits = 4
	ctx.FrameHeader.OrderHint = 8
	ctx.ReferenceOrderHints[tile.ReferenceFrameLast] = 4
	ctx.ReferenceOrderHints[tile.ReferenceFrameBWD] = 12
	return ctx
}

func testIntraPredictionVisit(mode tile.IntraMode) tile.BlockLoopVisit {
	return tile.BlockLoopVisit{
		Block: tile.BlockVisit{
			MICol: 4, MIRow: 4, MIColEnd: 8, MIRowEnd: 8,
			X4: 4, Y4: 4, Size: tile.BlockSize16x16, VisibleW4: 4, VisibleH4: 4,
			HaveTop: true, HaveLeft: true,
		},
		Prediction: tile.BlockPredictionModeResult{
			Valid:    true,
			Intra:    true,
			LumaMode: mode,
		},
	}
}

func testInterPredictionVisit(mv motion.Vector) tile.BlockLoopVisit {
	refs := tile.InterReferencesResult{Ref0: tile.ReferenceFrameLast, Ref1: tile.ReferenceFrameNone}
	return tile.BlockLoopVisit{
		Block: tile.BlockVisit{
			MICol: 4, MIRow: 4, MIColEnd: 8, MIRowEnd: 8,
			X4: 4, Y4: 4, Size: tile.BlockSize16x16, VisibleW4: 4, VisibleH4: 4,
			HaveTop: true, HaveLeft: true,
		},
		Prediction: tile.BlockPredictionModeResult{
			Valid:                true,
			Intra:                false,
			InterReferences:      refs,
			InterReferencesValid: true,
			InterMotion: tile.InterMotionResult{
				References: refs,
				MV:         [2]motion.Vector{mv},
			},
			InterMotionValid: true,
		},
	}
}

func testInterIntraPredictionVisit(mode tile.InterIntraMode, wedge bool) tile.BlockLoopVisit {
	visit := testInterPredictionVisit(motion.Vector{})
	visit.Prediction.InterIntra = tile.InterIntraResult{
		Enabled:  true,
		Mode:     mode,
		UseWedge: wedge,
	}
	visit.Prediction.InterIntraValid = true
	visit.Prediction.InterMotion.InterIntra = true
	visit.Prediction.MotionMode = tile.MotionModeTranslation
	visit.Prediction.MotionModeValid = true
	return visit
}

func testOBMCInterPredictionVisit(baseMV motion.Vector, aboveMV motion.Vector, leftMV motion.Vector) tile.BlockLoopVisit {
	visit := testInterPredictionVisit(baseMV)
	visit.Prediction.MotionMode = tile.MotionModeOBMC
	visit.Prediction.MotionModeValid = true
	visit.Prediction.OverlappableNeighborsValid = true
	testMotionExtras(&visit.Prediction).OverlappableNeighbors.AboveCount = 1
	testMotionExtras(&visit.Prediction).OverlappableNeighbors.Above[0] = tile.OverlappableNeighbor{
		RelX4: 0,
		Span4: 4,
		Size:  tile.BlockSize16x16,
		Motion: tile.InterMotionResult{
			References: visit.Prediction.InterMotion.References,
			MV:         [2]motion.Vector{aboveMV},
		},
		InterpFilters: motion.RegularFilters,
	}
	testMotionExtras(&visit.Prediction).OverlappableNeighbors.LeftCount = 1
	testMotionExtras(&visit.Prediction).OverlappableNeighbors.Left[0] = tile.OverlappableNeighbor{
		RelY4: 0,
		Span4: 4,
		Size:  tile.BlockSize16x16,
		Motion: tile.InterMotionResult{
			References: visit.Prediction.InterMotion.References,
			MV:         [2]motion.Vector{leftMV},
		},
		InterpFilters: motion.RegularFilters,
	}
	return visit
}

func testCompoundInterPredictionVisit(mv0 motion.Vector, mv1 motion.Vector, compoundType tile.CompoundType) tile.BlockLoopVisit {
	refs := tile.InterReferencesResult{Ref0: tile.ReferenceFrameLast, Ref1: tile.ReferenceFrameBWD, Compound: true}
	compoundIndex := uint8(0)
	if compoundType == tile.CompoundTypeAverage {
		compoundIndex = 1
	}
	return tile.BlockLoopVisit{
		Block: tile.BlockVisit{
			MICol: 4, MIRow: 4, MIColEnd: 8, MIRowEnd: 8,
			X4: 4, Y4: 4, Size: tile.BlockSize16x16, VisibleW4: 4, VisibleH4: 4,
			HaveTop: true, HaveLeft: true,
		},
		Prediction: tile.BlockPredictionModeResult{
			Valid:                true,
			Intra:                false,
			InterReferences:      refs,
			InterReferencesValid: true,
			InterMode: tile.InterModeResult{
				Compound:     true,
				CompoundMode: tile.CompoundInterModeNearestNearest,
			},
			InterModeValid: true,
			InterMotion: tile.InterMotionResult{
				References: refs,
				Mode: tile.InterModeResult{
					Compound:     true,
					CompoundMode: tile.CompoundInterModeNearestNearest,
				},
				MV: [2]motion.Vector{mv0, mv1},
			},
			InterMotionValid: true,
			MotionMode:       tile.MotionModeTranslation,
			MotionModeValid:  true,
			CompoundBlend: tile.CompoundBlendResult{
				Type:          compoundType,
				CompoundIndex: compoundIndex,
			},
			CompoundBlendValid: true,
		},
	}
}

func testIntraPrediction4x4Visit(mode tile.IntraMode) tile.BlockLoopVisit {
	visit := testIntraPredictionVisit(mode)
	visit.Block.MIColEnd = 5
	visit.Block.MIRowEnd = 5
	visit.Block.Size = tile.BlockSize4x4
	visit.Block.VisibleW4 = 1
	visit.Block.VisibleH4 = 1
	return visit
}

func frameWorkTestSample(plane frame.Plane, bytesPerSample int, x int, y int) uint16 {
	sample, ok := frameWorkLoadSample(&plane, bytesPerSample, x, y)
	if !ok {
		panic("bad test sample")
	}
	return sample
}

func frameWorkTestSampleClamped(plane frame.Plane, bytesPerSample int, x int, y int) uint16 {
	if x < 0 {
		x = 0
	} else if x >= plane.Width {
		x = plane.Width - 1
	}
	if y < 0 {
		y = 0
	} else if y >= plane.Height {
		y = plane.Height - 1
	}
	return frameWorkTestSample(plane, bytesPerSample, x, y)
}

func assertFrameWorkPlaneBlockEqual(t *testing.T, got frame.Plane, want frame.Plane, bytesPerSample int, x int, y int, width int, height int) {
	t.Helper()
	assertFrameWorkPlaneBlockEqualAt(t, got, x, y, want, x, y, bytesPerSample, width, height)
}

func assertFrameWorkPlaneBlockEqualAt(t *testing.T, got frame.Plane, gotX int, gotY int, want frame.Plane, wantX int, wantY int, bytesPerSample int, width int, height int) {
	t.Helper()
	for row := range height {
		for col := range width {
			g := frameWorkTestSample(got, bytesPerSample, gotX+col, gotY+row)
			w := frameWorkTestSample(want, bytesPerSample, wantX+col, wantY+row)
			if g != w {
				t.Fatalf("sample(%d,%d)=%d want %d", gotX+col, gotY+row, g, w)
			}
		}
	}
}

// testFrameWorkCompoundConvBuf produces the libaom-faithful un-rounded 16-bit
// CONV_BUF predictor (av1_dist_wtd_convolve_*) for one reference, matching the
// production compound prediction path. The reference helpers below blend two of
// these buffers and round once, exactly like av1_dist_wtd_convolve_* /
// aom_*_blend_a64_d16_mask_c.
func testFrameWorkCompoundConvBuf(t *testing.T, reference frame.Plane, bytesPerSample int, bitDepth uint8, dstX int, dstY int, width int, height int, mv motion.Vector, subsamplingX bool, subsamplingY bool, filters motion.InterpFilters) *motion.CompoundConvBuf {
	t.Helper()
	refX, refY, subX, subY, err := motion.ReferenceOriginSubsampled(dstX, dstY, mv, subsamplingX, subsamplingY)
	if err != nil {
		t.Fatal(err)
	}
	buf := &motion.CompoundConvBuf{}
	if err := motion.PredictInterCompoundRefToConvBuf(buf, reference, bytesPerSample, bitDepth, refX, refY, width, height, width, height, subX, subY, filters, nil); err != nil {
		t.Fatal(err)
	}
	return buf
}

func testFrameWorkScaledCompoundConvBuf(t *testing.T, reference frame.Plane, curWidth int, curHeight int, bytesPerSample int, bitDepth uint8, dstX int, dstY int, width int, height int, mv motion.Vector, subsamplingX bool, subsamplingY bool, filters motion.InterpFilters) *motion.CompoundConvBuf {
	t.Helper()
	sf, err := motion.NewScaleFactors(reference.Width, reference.Height, curWidth, curHeight)
	if err != nil {
		t.Fatal(err)
	}
	startX, startY, xStep, yStep, err := sf.ScaledBlockOrigin(dstX, dstY, mv, subsamplingX, subsamplingY)
	if err != nil {
		t.Fatal(err)
	}
	xTable, err := motion.SubpelKernelTableFor(filters.X, width)
	if err != nil {
		t.Fatal(err)
	}
	yTable, err := motion.SubpelKernelTableFor(filters.Y, height)
	if err != nil {
		t.Fatal(err)
	}
	buf := &motion.CompoundConvBuf{}
	if err := motion.PredictScaledCompoundRefToConvBufWithScratch(buf, reference, bytesPerSample, bitDepth, width, height, startX, xStep, startY, yStep, xTable, yTable, nil); err != nil {
		t.Fatal(err)
	}
	return buf
}

func assertFrameWorkCompoundBlendEqual(t *testing.T, got frame.Plane, buf0 *motion.CompoundConvBuf, buf1 *motion.CompoundConvBuf, bytesPerSample int, bitDepth uint8, x int, y int, width int, height int, fwdOffset int, bckOffset int) {
	t.Helper()
	stride := width * bytesPerSample
	want := frame.Plane{Pix: make([]byte, stride*height), Stride: stride, Width: width, Height: height}
	if err := motion.BlendCompoundAvg(want, buf0, buf1, bytesPerSample, bitDepth, 0, 0, width, height, fwdOffset, bckOffset); err != nil {
		t.Fatal(err)
	}
	for row := range height {
		for col := range width {
			w := frameWorkTestSample(want, bytesPerSample, col, row)
			g := frameWorkTestSample(got, bytesPerSample, x+col, y+row)
			if g != w {
				t.Fatalf("sample(%d,%d)=%d want %d", x+col, y+row, g, w)
			}
		}
	}
}

func assertFrameWorkMaskedCompoundEqual(t *testing.T, got frame.Plane, buf0 *motion.CompoundConvBuf, buf1 *motion.CompoundConvBuf, mask []byte, maskStride int, subX bool, subY bool, bytesPerSample int, bitDepth uint8, x int, y int, width int, height int) {
	t.Helper()
	stride := width * bytesPerSample
	want := frame.Plane{Pix: make([]byte, stride*height), Stride: stride, Width: width, Height: height}
	if err := motion.BlendCompoundMaskD16(want, buf0, buf1, bytesPerSample, bitDepth, 0, 0, width, height, mask, maskStride, subX, subY); err != nil {
		t.Fatal(err)
	}
	for row := range height {
		for col := range width {
			w := frameWorkTestSample(want, bytesPerSample, col, row)
			g := frameWorkTestSample(got, bytesPerSample, x+col, y+row)
			if g != w {
				t.Fatalf("sample(%d,%d)=%d want %d", x+col, y+row, g, w)
			}
		}
	}
}

func setFrameWorkTestSample(plane frame.Plane, bytesPerSample int, x int, y int, value uint16) {
	offset := y*plane.Stride + x*bytesPerSample
	if bytesPerSample == 1 {
		plane.Pix[offset] = byte(value)
		return
	}
	plane.Pix[offset] = byte(value)
	plane.Pix[offset+1] = byte(value >> 8)
}

func testFrameWorkMotionPredictionPlane(t *testing.T, reference frame.Plane, bytesPerSample int, bitDepth uint8, dstX int, dstY int, width int, height int, mv motion.Vector, filters motion.InterpFilters) frame.Plane {
	t.Helper()
	stride := width * bytesPerSample
	dst := frame.Plane{
		Pix:    make([]byte, stride*height),
		Stride: stride,
		Width:  width,
		Height: height,
	}
	refX, refY, subX, subY, err := motion.ReferenceOriginSubsampled(dstX, dstY, mv, false, false)
	if err != nil {
		t.Fatal(err)
	}
	if err := motion.PredictInterPlaneBlockFromOriginWithFilterBitDepth(dst, reference, bytesPerSample, bitDepth, 0, 0, refX, refY, width, height, subX, subY, filters); err != nil {
		t.Fatal(err)
	}
	return dst
}

func testFrameWorkMotionPredictionPlaneSubsampled(t *testing.T, reference frame.Plane, bytesPerSample int, bitDepth uint8, dstX int, dstY int, width int, height int, mv motion.Vector, subsamplingX bool, subsamplingY bool, filters motion.InterpFilters) frame.Plane {
	t.Helper()
	stride := width * bytesPerSample
	dst := frame.Plane{
		Pix:    make([]byte, stride*height),
		Stride: stride,
		Width:  width,
		Height: height,
	}
	refX, refY, subX, subY, err := motion.ReferenceOriginSubsampled(dstX, dstY, mv, subsamplingX, subsamplingY)
	if err != nil {
		t.Fatal(err)
	}
	if err := motion.PredictInterPlaneBlockFromOriginWithFilterBitDepth(dst, reference, bytesPerSample, bitDepth, 0, 0, refX, refY, width, height, subX, subY, filters); err != nil {
		t.Fatal(err)
	}
	return dst
}

func testFrameWorkDiffWtdMask(t *testing.T, buf0 *motion.CompoundConvBuf, buf1 *motion.CompoundConvBuf, bitDepth uint8, width int, height int, maskType tile.DiffWtdMaskType) []byte {
	t.Helper()
	mask := make([]byte, width*height)
	if err := motion.BuildDiffWtdMaskD16(mask, width, buf0, buf1, bitDepth, width, height, maskType == tile.DiffWtdMaskType38Inv); err != nil {
		t.Fatal(err)
	}
	return mask
}

func fillFrameWorkInterReference(reference *frame.Frame, max uint16) {
	for y := 0; y < reference.Y.Height; y++ {
		for x := 0; x < reference.Y.Width; x++ {
			setFrameWorkTestSample(reference.Y, reference.Layout.BytesPerSample, x, y, uint16((x*x+3*y*y+17*x+11*y)&int(max)))
		}
	}
}

func fillFrameWorkInterReferenceAllPlanes(reference *frame.Frame, max uint16) {
	fillFrameWorkInterReference(reference, max)
	for y := 0; y < reference.U.Height; y++ {
		for x := 0; x < reference.U.Width; x++ {
			setFrameWorkTestSample(reference.U, reference.Layout.BytesPerSample, x, y, uint16((5*x*x+7*y*y+13*x+19*y+23)&int(max)))
			setFrameWorkTestSample(reference.V, reference.Layout.BytesPerSample, x, y, uint16((11*x*x+3*y*y+29*x+31*y+37)&int(max)))
		}
	}
}

func fillFrameWorkInterReferenceVariant(reference *frame.Frame, max uint16, seed uint16) {
	for y := 0; y < reference.Y.Height; y++ {
		for x := 0; x < reference.Y.Width; x++ {
			setFrameWorkTestSample(reference.Y, reference.Layout.BytesPerSample, x, y, uint16((29*x+31*y+int(seed))&int(max)))
		}
	}
	for y := 0; y < reference.U.Height; y++ {
		for x := 0; x < reference.U.Width; x++ {
			setFrameWorkTestSample(reference.U, reference.Layout.BytesPerSample, x, y, uint16((17*x+23*y+int(seed)*3)&int(max)))
			setFrameWorkTestSample(reference.V, reference.Layout.BytesPerSample, x, y, uint16((41*x+43*y+int(seed)*5)&int(max)))
		}
	}
}

func seedFrameWorkDirectionalEdges(output *frame.Frame, max uint16, above uint16, left uint16, aboveLeft uint16) {
	for x := 0; x < output.Y.Width; x++ {
		setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, x, 15, uint16((int(above)+x)&int(max)))
	}
	for y := 0; y < output.Y.Height; y++ {
		setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, 15, y, uint16((int(left)+y)&int(max)))
	}
	setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, 15, 15, aboveLeft&max)
}

// TestFrameWorkApplyDirectionalIntraEdgeFilterCornerBlock32x32D157 pins the
// directional intra-edge-filter dispatch for the 32x32 D157 luma block at the
// frame's top-left corner. The 34x34 fast vector exercises this exact
// configuration (mi=0,0, EnableIntraEdgeFilter=true, smoothNeighbor=false,
// HaveTop=false, HaveLeft=false, angle=151 from luma angle delta -2) as its
// first decoded block, so any future change to the dispatch in
// frameWorkApplyDirectionalIntraEdgeFilter that drops the corner-filter call,
// flips the strength derivation, or re-enables the (top, left) edge filters
// when the corresponding neighbour counts are zero would silently regress the
// vector's reconstruction. The fixture mirrors libaom's
// build_directional_and_filter_intra_predictors fallback when both
// n_top_px=0 and n_left_px=0: above defaults to 127, left defaults to 129,
// and the above-left default 128 is preserved by the corner filter
// (128*5 + 127*6 + 129*5 = 2048; (2048+8)>>4 = 128).
func TestFrameWorkApplyDirectionalIntraEdgeFilterCornerBlock32x32D157(t *testing.T) {
	const (
		width  = 32
		height = 32
		angle  = 151 // libaom p_angle for D157 with luma_angle_delta=-2
	)
	var scratch FrameWorkIntraPredictionScratch
	origin := frameWorkDirectionalEdgeOrigin
	for i := range scratch.Above {
		scratch.Above[i] = 127
	}
	for i := range scratch.Left {
		scratch.Left[i] = 129
	}
	scratch.Above[origin-1] = 128
	scratch.Left[origin-1] = 128
	block := tile.BlockVisit{
		MICol: 0, MIRow: 0, MIColEnd: 8, MIRowEnd: 8,
		Size: tile.BlockSize32x32, VisibleW4: 8, VisibleH4: 8,
		HaveTop: false, HaveLeft: false,
	}
	edges := prediction.DirectionalEdges{
		Above:       scratch.Above[:],
		Left:        scratch.Left[:],
		AboveOrigin: origin,
		LeftOrigin:  origin,
	}
	if err := frameWorkApplyDirectionalIntraEdgeFilterBounded(8, width, height, angle, block, false, &edges, &scratch, false); err != nil {
		t.Fatalf("apply edge filter: %v", err)
	}
	// The corner filter rewrites above[-1] and left[-1] to
	// (left[0]*5 + above[-1]*6 + above[0]*5 + 8) >> 4. For the all-default
	// fallback (127/128/129) the result is again 128.
	if got := scratch.Above[origin-1]; got != 128 {
		t.Fatalf("above[-1]=%d want 128 (corner filter no-op)", got)
	}
	if got := scratch.Left[origin-1]; got != 128 {
		t.Fatalf("left[-1]=%d want 128 (corner filter no-op)", got)
	}
	// The above and left edge filters must be skipped because nTop=0 and
	// nLeft=0 — same as libaom's `if (n_top_px > 0)` guard. Confirm the
	// default above samples remain 127 and the default left samples remain
	// 129 (i.e. FilterIntraEdge was not run on them).
	for i := range width + height {
		if got := scratch.Above[origin+i]; got != 127 {
			t.Fatalf("above[%d]=%d want 127 (filter must be skipped)", i, got)
		}
		if got := scratch.Left[origin+i]; got != 129 {
			t.Fatalf("left[%d]=%d want 129 (filter must be skipped)", i, got)
		}
	}
	// UseIntraEdgeUpsample returns false for blockWH=64 with d=61>=40, so
	// the upsample step must NOT be marked active.
	if edges.UpsampleAbove {
		t.Fatalf("UpsampleAbove=true want false (d=61>=40 disables upsample for blockWH=64)")
	}
	if edges.UpsampleLeft {
		t.Fatalf("UpsampleLeft=true want false (d=-29, |d|>=8 + blockWH=64 disables upsample)")
	}
}

// TestFrameWorkDirectionalPredictionEdgesCornerBlock32x32D157 drives the
// full edge-fill + edge-filter dispatch via frameWorkDirectionalPredictionEdges
// for the 34x34 vector's first decoded block. The test seeds an empty plane,
// requests the corner-block edges, and asserts the returned DirectionalEdges
// match libaom's fallback for n_top_px=n_left_px=0 — bit-exactly equivalent
// to what TestPredictDirectionalIntraPlaneBlockCornerFrame32x32D157 pins at
// the prediction-API level.
func TestFrameWorkDirectionalPredictionEdgesCornerBlock32x32D157(t *testing.T) {
	const (
		width  = 32
		height = 32
		angle  = 151
	)
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, Align: 64})
	testFillFrame(output, 0)
	dst := frame.Plane{
		Pix:    output.Y.Pix,
		Stride: output.Y.Stride,
		Width:  output.Y.Width,
		Height: output.Y.Height,
	}
	var scratch FrameWorkIntraPredictionScratch
	block := tile.BlockVisit{
		MICol: 0, MIRow: 0, MIColEnd: 8, MIRowEnd: 8,
		Size: tile.BlockSize32x32, VisibleW4: 8, VisibleH4: 8,
		HaveTop: false, HaveLeft: false,
	}
	edges, err := frameWorkDirectionalPredictionEdges(dst, 1, 8, 0, 0, width, height, angle, block, &scratch, true, false, false, false, 0, 0)
	if err != nil {
		t.Fatalf("directional edges: %v", err)
	}
	origin := frameWorkDirectionalEdgeOrigin
	if edges.AboveOrigin != origin || edges.LeftOrigin != origin {
		t.Fatalf("edge origins: got above=%d left=%d want %d", edges.AboveOrigin, edges.LeftOrigin, origin)
	}
	// Corner-filter output preserves 128 for the all-default fallback.
	if got := edges.Above[origin-1]; got != 128 {
		t.Fatalf("above[-1]=%d want 128", got)
	}
	if got := edges.Left[origin-1]; got != 128 {
		t.Fatalf("left[-1]=%d want 128", got)
	}
	// Above samples remain at the missing-above default (127). Left samples
	// remain at the missing-left default (129). The edge filter is skipped
	// for both edges when no real neighbours exist.
	for i := range width + height - 1 {
		if got := edges.Above[origin+i]; got != 127 {
			t.Fatalf("above[%d]=%d want 127", i, got)
		}
		if got := edges.Left[origin+i]; got != 129 {
			t.Fatalf("left[%d]=%d want 129", i, got)
		}
	}
	// Upsample is disabled for blockWH=64 with d=|angle-90|=61>=40 (above)
	// and d=|angle-180|=29 in zone 2 (left).
	if edges.UpsampleAbove || edges.UpsampleLeft {
		t.Fatalf("upsample flags: above=%v left=%v want both false", edges.UpsampleAbove, edges.UpsampleLeft)
	}
}

func predictTestInterPlaneBlock(t *testing.T, dst frame.Plane, ref frame.Plane, bytesPerSample int, bitDepth uint8, dstX int, dstY int, width int, height int, mv motion.Vector, filters motion.InterpFilters) {
	t.Helper()
	refX, refY, subX, subY, err := motion.ReferenceOriginSubsampled(dstX, dstY, mv, false, false)
	if err != nil {
		t.Fatal(err)
	}
	if err := motion.PredictInterPlaneBlockFromOriginWithFilterBitDepth(dst, ref, bytesPerSample, bitDepth, dstX, dstY, refX, refY, width, height, subX, subY, filters); err != nil {
		t.Fatal(err)
	}
}

func TestFrameWorkBatchSetupTemporalMotionField(t *testing.T) {
	ctx := FrameWorkBatch{
		FrameWorkFrameContext: FrameWorkFrameContext{
			Sequence: FrameWorkSequenceContextFromHeader(parser.SequenceHeader{
				EnableOrderHint: true,
				OrderHintBits:   5,
				ColorConfig:     parser.ColorConfig{BitDepth: 8},
			}),
			FrameHeader: parser.FrameHeaderPrefix{OrderHint: 8},
			FrameSize:   parser.FrameSize{CodedWidth: 64, Height: 64},
		},
	}
	_, _, length, err := ctx.ReferenceMVFrameShape()
	if err != nil {
		t.Fatal(err)
	}
	temporal, err := ctx.BindTemporalMotionField(make([]tile.TemporalMotionEntry, length))
	if err != nil {
		t.Fatal(err)
	}
	start, err := ctx.BindReferenceMVFrame(make([]tile.ReferenceMVEntry, length))
	if err != nil {
		t.Fatal(err)
	}
	start.Entries[2*start.Stride+2] = tile.ReferenceMVEntry{
		Ref:   tile.ReferenceFrameLast,
		MV:    motion.Vector{Row: 64, Col: 64},
		Valid: true,
	}
	ctx.TemporalMVs = &temporal
	var refOrderHints [parser.InterRefsPerFrame]uint8
	refOrderHints[tile.ReferenceFrameLast] = 0
	refOrderHints[tile.ReferenceFrameAltref] = 1
	ctx.ReferenceMVs[tile.ReferenceFrameLast] = tile.TemporalMotionReferenceFrame{
		Frame:         &start,
		OrderHint:     4,
		RefOrderHints: refOrderHints,
	}

	stats, err := ctx.SetupTemporalMotionField()
	if err != nil {
		t.Fatal(err)
	}
	if stats.Projections != 1 || stats.RefStamp != 1 {
		t.Fatalf("stats=%+v", stats)
	}
	if got := temporal.Entries[1*temporal.Stride+1]; got != (tile.TemporalMotionEntry{
		MV:             motion.Vector{Row: 64, Col: 64},
		RefFrameOffset: 4,
		Valid:          true,
	}) {
		t.Fatalf("projected entry=%+v", got)
	}
}

func TestFrameWorkBatchSetupTemporalMotionFieldRejectsMissingField(t *testing.T) {
	emptyBatch := FrameWorkBatch{}
	if _, err := emptyBatch.SetupTemporalMotionField(); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("err=%v want %v", err, ErrInvalidBatch)
	}
}

func TestFrameWorkBatchBindRestorationFrameBuffers(t *testing.T) {
	ctx := testFrameWorkRestorationBufferBatch([3]parser.RestorationType{
		parser.RestorationWiener,
		parser.RestorationSGRProj,
		parser.RestorationNone,
	}, false)
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
	if buffers.Plan != plan {
		t.Fatalf("plan=%+v want %+v", buffers.Plan, plan)
	}
	for plane := 0; plane < int(plan.Planes); plane++ {
		if len(buffers.Records[plane]) != int(plan.UnitRecords[plane]) {
			t.Fatalf("plane %d records len=%d want %d", plane, len(buffers.Records[plane]), plan.UnitRecords[plane])
		}
		if len(buffers.Boundaries[plane].Above) != int(plan.Boundaries[plane].Len) ||
			len(buffers.Boundaries[plane].Below) != int(plan.Boundaries[plane].Len) ||
			buffers.Boundaries[plane].Stride != int(plan.Boundaries[plane].Stride) {
			t.Fatalf("plane %d boundaries=%+v plan=%+v", plane, buffers.Boundaries[plane], plan.Boundaries[plane])
		}
	}

	if len(buffers.Records[0]) == 0 || len(buffers.Boundaries[0].Above) == 0 {
		t.Fatalf("expected active Y restoration storage")
	}
	buffers.Records[0][0].Index = 99
	if recordBacking[0].Index != 99 {
		t.Fatalf("records do not alias caller backing")
	}
	buffers.Boundaries[0].Above[0] = 77
	if above[0] != 77 {
		t.Fatalf("boundaries do not alias caller backing")
	}
}

func TestFrameWorkRestorationFrameBuffersResetRecords(t *testing.T) {
	ctx := testFrameWorkRestorationBufferBatch([3]parser.RestorationType{
		parser.RestorationWiener,
		parser.RestorationSGRProj,
		parser.RestorationNone,
	}, false)
	plan, err := ctx.RestorationFramePlan()
	if err != nil {
		t.Fatal(err)
	}
	buffers, err := ctx.BindRestorationFrameBuffers(
		make([]tile.RestorationUnitRecord, plan.UnitRecordLen()),
		make([]uint16, plan.BoundaryBufferLen()),
		make([]uint16, plan.BoundaryBufferLen()),
	)
	if err != nil {
		t.Fatal(err)
	}
	for plane := range buffers.Records {
		for i := range buffers.Records[plane] {
			buffers.Records[plane][i] = tile.RestorationUnitRecord{Index: ^uint32(0)}
		}
	}

	if err := buffers.ResetRecords(); err != nil {
		t.Fatal(err)
	}
	for plane := 0; plane < int(plan.Planes); plane++ {
		grid := plan.Grids[plane]
		records := buffers.Records[plane]
		if plan.UnitRecords[plane] == 0 {
			if len(records) != 0 {
				t.Fatalf("plane %d inactive records len=%d", plane, len(records))
			}
			continue
		}
		first := records[0]
		if first.Index != 0 || first.Col != 0 || first.Row != 0 ||
			first.Unit.Type != parser.RestorationNone || first.StripeCount == 0 {
			t.Fatalf("plane %d first record=%+v", plane, first)
		}
		last := records[len(records)-1]
		if last.Index != uint32(len(records)-1) ||
			last.Col != grid.HorzUnits-1 ||
			last.Row != grid.VertUnits-1 ||
			last.Unit.Type != parser.RestorationNone ||
			last.StripeCount == 0 {
			t.Fatalf("plane %d last record=%+v grid=%+v", plane, last, grid)
		}
	}
}

func TestFrameWorkBatchBindRestorationFrameBuffersAllNone(t *testing.T) {
	ctx := testFrameWorkRestorationBufferBatch([3]parser.RestorationType{}, false)
	buffers, err := ctx.BindRestorationFrameBuffers(nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if buffers.Plan.Active || buffers.Plan.UnitRecordLen() != 0 || buffers.Plan.BoundaryBufferLen() != 0 {
		t.Fatalf("all-none buffers=%+v", buffers.Plan)
	}
	if err := buffers.ResetRecords(); err != nil {
		t.Fatal(err)
	}
}

func TestFrameWorkBatchBindRestorationFrameBuffersRejectsInvalidInputs(t *testing.T) {
	ctx := testFrameWorkRestorationBufferBatch([3]parser.RestorationType{
		parser.RestorationWiener,
		parser.RestorationSGRProj,
		parser.RestorationNone,
	}, false)
	plan, err := ctx.RestorationFramePlan()
	if err != nil {
		t.Fatal(err)
	}
	recordLen := plan.UnitRecordLen()
	boundaryLen := plan.BoundaryBufferLen()
	if _, err := ctx.BindRestorationFrameBuffers(
		make([]tile.RestorationUnitRecord, recordLen-1),
		make([]uint16, boundaryLen),
		make([]uint16, boundaryLen),
	); !errors.Is(err, tile.ErrJobBufferTooSmall) {
		t.Fatalf("short records err=%v want %v", err, tile.ErrJobBufferTooSmall)
	}
	if _, err := ctx.BindRestorationFrameBuffers(
		make([]tile.RestorationUnitRecord, recordLen+1),
		make([]uint16, boundaryLen),
		make([]uint16, boundaryLen),
	); !errors.Is(err, tile.ErrInvalidPlan) {
		t.Fatalf("long records err=%v want %v", err, tile.ErrInvalidPlan)
	}
	if _, err := ctx.BindRestorationFrameBuffers(
		make([]tile.RestorationUnitRecord, recordLen),
		make([]uint16, boundaryLen-1),
		make([]uint16, boundaryLen),
	); !errors.Is(err, tile.ErrJobBufferTooSmall) {
		t.Fatalf("short boundaries err=%v want %v", err, tile.ErrJobBufferTooSmall)
	}
	emptyBatch := FrameWorkBatch{}
	if _, err := emptyBatch.BindRestorationFrameBuffers(nil, nil, nil); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("invalid batch err=%v want %v", err, ErrInvalidBatch)
	}

	buffers, err := ctx.BindRestorationFrameBuffers(
		make([]tile.RestorationUnitRecord, recordLen),
		make([]uint16, boundaryLen),
		make([]uint16, boundaryLen),
	)
	if err != nil {
		t.Fatal(err)
	}
	buffers.Records[0] = append(buffers.Records[0], tile.RestorationUnitRecord{})
	if err := buffers.ResetRecords(); !errors.Is(err, tile.ErrInvalidPlan) {
		t.Fatalf("bad reset err=%v want %v", err, tile.ErrInvalidPlan)
	}
}

func TestFrameWorkRestorationFrameBuffersAllocs(t *testing.T) {
	ctx := testFrameWorkRestorationBufferBatch([3]parser.RestorationType{
		parser.RestorationWiener,
		parser.RestorationSGRProj,
		parser.RestorationNone,
	}, false)
	plan, err := ctx.RestorationFramePlan()
	if err != nil {
		t.Fatal(err)
	}
	recordBacking := make([]tile.RestorationUnitRecord, plan.UnitRecordLen())
	above := make([]uint16, plan.BoundaryBufferLen())
	below := make([]uint16, plan.BoundaryBufferLen())

	allocs := testing.AllocsPerRun(1000, func() {
		buffers, err := ctx.BindRestorationFrameBuffers(recordBacking, above, below)
		if err != nil {
			t.Fatal(err)
		}
		if err := buffers.ResetRecords(); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("FrameWorkRestorationFrameBuffers allocated: %f", allocs)
	}
}

func testFrameWorkRestorationBufferBatch(types [3]parser.RestorationType, mono bool) FrameWorkBatch {
	return FrameWorkBatch{
		FrameWorkFrameContext: FrameWorkFrameContext{
			Sequence: FrameWorkSequenceContextFromHeader(parser.SequenceHeader{
				Use128x128Superblock: true,
				ColorConfig: parser.ColorConfig{
					BitDepth:     8,
					MonoChrome:   mono,
					SubsamplingX: true,
					SubsamplingY: true,
				},
			}),
			FrameSize: parser.FrameSize{
				UpscaledWidth:       300,
				CodedWidth:          280,
				Height:              260,
				SuperResDenominator: 8,
			},
			Restoration: parser.RestorationParams{
				Type:       types,
				UnitSizeY:  128,
				UnitSizeUV: 64,
			},
		},
	}
}

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
			ReferenceOrderHints: [parser.InterRefsPerFrame]uint8{1, 9, 10, 4, 5, 6, 7},
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
	if int(got.Cols) != cols || int(got.Rows) != rows || int(got.Stride) != cols || len(got.Entries) != length {
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
	if int(temporal.Cols) != cols || int(temporal.Rows) != rows || int(temporal.Stride) != cols || len(temporal.Entries) != length {
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
	emptyBatch := FrameWorkBatch{}
	if _, _, _, err := emptyBatch.ReferenceMVFrameShape(); !errors.Is(err, ErrInvalidBatch) {
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
	refs := [parser.InterRefsPerFrame]uint8{7, 8, 9, 24, 4, 5, 6}
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

func TestFrameWorkTileResidualRunnerRun(t *testing.T) {
	ctx, _, _, _, req := testFrameWorkResidualDriver(t)
	var retained FrameWorkTileResidualCDFStorage
	retainedValid := false
	ctx.RetainedTileResidualCDFs = &retained
	ctx.RetainedTileResidualCDFsValid = &retainedValid
	ctx.Jobs[0].UpdatesFrameContext = true
	var runner FrameWorkTileResidualRunner
	runner.Workers = make([]FrameWorkTileResidualRunnerWorker, 1)
	runner.Workers[0].Int32Scratch = req.Int32Scratch
	runner.Workers[0].ResidualScratch = req.ResidualScratch
	afterBlocks := 0
	runner.Request.AfterBlock = func(visit tile.BlockLoopVisit) error {
		afterBlocks++
		if !visit.CoefficientsValid {
			t.Fatalf("visit missing coefficients: %+v", visit)
		}
		return nil
	}

	if err := runner.Run(ctx); err != nil {
		t.Fatal(err)
	}
	if stats := runner.Workers[0].Stats; stats.Loop.Blocks != 1 || stats.CoefficientBlocks != 1 || stats.TXBs != 1 || stats.Residuals != 1 {
		t.Fatalf("stats=%+v", stats)
	}
	if afterBlocks != 1 {
		t.Fatalf("afterBlocks=%d want 1", afterBlocks)
	}
	if !retainedValid {
		t.Fatal("retained CDF storage not marked valid")
	}
}

func TestFrameWorkTileResidualRunnerExecuteFrameWork(t *testing.T) {
	ctx, _, _, _, req := testFrameWorkResidualDriver(t)
	var batches [1]Batch
	n, err := BuildBatches(batches[:], ctx.Jobs, 1)
	if err != nil {
		t.Fatal(err)
	}
	pool, err := NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer pool.Close()

	var runner FrameWorkTileResidualRunner
	runner.Workers = make([]FrameWorkTileResidualRunnerWorker, 1)
	runner.Workers[0].Int32Scratch = req.Int32Scratch
	runner.Workers[0].ResidualScratch = req.ResidualScratch
	if err := pool.ExecuteFrameWorkRunner(batches[:n], ctx.Jobs, ctx, &runner); err != nil {
		t.Fatal(err)
	}
	if stats := runner.Workers[0].Stats; stats.Loop.Blocks != 1 || stats.CoefficientBlocks != 1 || stats.Residuals != 1 {
		t.Fatalf("stats=%+v", stats)
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
	shortCarrier := req
	scratch.LoopContext.Above = []tile.BlockLoopRootAboveContext{}
	if _, err := ctx.DecodeAndReconstructJobResiduals(0, state, cdfs, &scratch, shortCarrier); !errors.Is(err, tile.ErrInvalidDecodeState) {
		t.Fatalf("short scratch context carrier err=%v want %v", err, tile.ErrInvalidDecodeState)
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
	for i := range tile.FrameLoopFilterCount {
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

func TestFrameWorkTileResidualCDFStorageResetCDFCounts(t *testing.T) {
	var storage FrameWorkTileResidualCDFStorage
	if err := storage.InitDefault(64); err != nil {
		t.Fatal(err)
	}
	if err := storage.Partition.Partition[tile.BlockLevel32x32][0].Update(3); err != nil {
		t.Fatal(err)
	}
	if err := storage.Mode.Skip[0].Update(0); err != nil {
		t.Fatal(err)
	}
	if err := storage.Intra.YMode[0].Update(0); err != nil {
		t.Fatal(err)
	}
	if err := storage.Coeff.TXBSkip[0][0].Update(0); err != nil {
		t.Fatal(err)
	}

	partition := append([]uint16(nil), storage.Partition.Partition[tile.BlockLevel32x32][0].Values()...)
	if partition[len(partition)-1] == 0 {
		t.Fatal("partition CDF count did not update")
	}
	if err := storage.ResetCDFCounts(); err != nil {
		t.Fatal(err)
	}
	got := storage.Partition.Partition[tile.BlockLevel32x32][0].Values()
	for i := 0; i < len(got)-1; i++ {
		if got[i] != partition[i] {
			t.Fatalf("partition probability %d=%d want %d", i, got[i], partition[i])
		}
	}
	if got[len(got)-1] != 0 || storage.Mode.Skip[0].Values()[2] != 0 ||
		storage.Intra.YMode[0].Values()[13] != 0 || storage.Coeff.TXBSkip[0][0].Values()[2] != 0 {
		t.Fatalf("counts not reset: partition=%v skip=%v ymode=%v txbskip=%v",
			got, storage.Mode.Skip[0].Values(), storage.Intra.YMode[0].Values(),
			storage.Coeff.TXBSkip[0][0].Values())
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
	req.AfterBlock = func(visit tile.BlockLoopVisit) error {
		order = append(order, "after")
		if !visit.CoefficientsValid || scratch.stats.TXBs == 0 {
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
	if len(order) != 1 {
		t.Fatalf("order=%v want [after]", order)
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
	var scratch FrameWorkTileResidualScratch
	var stats FrameWorkTileResidualStats
	controller := frameWorkTileResidualLoopController{
		batch:   ctx,
		index:   0,
		state:   &tile.DecodeState{},
		scratch: &scratch,
		req: FrameWorkTileResidualRequest{
			PredictionScratch: &predictionScratch,
		},
		stats: &stats,
	}
	visit := testIntraPredictionVisit(tile.IntraModeDC)
	visit.Prefix.SkipTransform = true
	if err := controller.reconBBegin(&visit); err != nil {
		t.Fatal(err)
	}
	// Skip intra blocks are predicted per transform block from the all-zero
	// TXB visits the coefficient walk emits (dav1d recon_b_intra), not at
	// block begin.
	if stats.Predictions != 0 {
		t.Fatalf("predicted before the first txb: stats=%+v", stats)
	}
	skipBlock := testSkipIntraTXB(0, 4, 4, tile.TransformSize16x16, 4)
	if err := controller.fusedReconState().reconstructTXBWithDequantized(&visit, &skipBlock, nil, 0, 0, false); err != nil {
		t.Fatal(err)
	}
	if stats.Predictions != 1 || stats.SkippedBlocks != 1 || stats.CoefficientBlocks != 0 {
		t.Fatalf("stats=%+v", stats)
	}
	if got := frameWorkTestSample(output.Y, output.Layout.BytesPerSample, 16, 16); got != 30 {
		t.Fatalf("predicted sample=%d want 30", got)
	}
}

// testSkipIntraTXB is one all-zero transform-block visit as the coefficient
// walk emits it for a skip intra block.
func testSkipIntraTXB(plane uint8, x4, y4 uint8, size tile.TransformSize, vis4 uint8) tile.BlockCoeffBlock {
	return tile.BlockCoeffBlock{
		Plane:     plane,
		Block:     tile.TransformBlock{X4: x4, Y4: y4, Size: size, VisibleW4: vis4, VisibleH4: vis4},
		Transform: transform.TypeDCTDCT,
		Result:    tile.TXBDecodeResult{AllZero: true},
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
	if err := controller.reconBBegin(&visit); err != nil {
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
		state:   &tile.DecodeState{},
		scratch: &scratch,
		req: FrameWorkTileResidualRequest{
			PredictionScratch: &predictionScratch,
		},
		stats: &stats,
	}
	visit := testCFLPredictionVisit()
	visit.Prefix.SkipTransform = true
	if err := controller.reconBBegin(&visit); err != nil {
		t.Fatal(err)
	}
	for _, block := range []tile.BlockCoeffBlock{
		testSkipIntraTXB(0, 4, 4, tile.TransformSize16x16, 4),
		testSkipIntraTXB(1, 2, 2, tile.TransformSize8x8, 2),
		testSkipIntraTXB(2, 2, 2, tile.TransformSize8x8, 2),
	} {
		if err := controller.fusedReconState().reconstructTXBWithDequantized(&visit, &block, nil, 0, 0, false); err != nil {
			t.Fatal(err)
		}
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
	if err := controller.reconBBegin(&visit); err != nil {
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
	}
	if err := controller.fusedReconState().reconstructTXBWithDequantized(&visit, &chroma, nil, 0, 0, false); err != nil {
		t.Fatal(err)
	}
	testPredictFrameWorkCFLWant(t, want, visit, FrameWorkPlaneU)
	assertFrameWorkPlaneBlockEqual(t, output.U, want.U, output.Layout.BytesPerSample, 8, 8, 8, 8)
	if !controller.recon.cflPredictionDone || stats.Residuals != 1 {
		t.Fatalf("controller cflDone=%v stats=%+v", controller.recon.cflPredictionDone, stats)
	}
}

func TestFrameWorkDeferredPaletteArenaCopiesAfterGrowth(t *testing.T) {
	var scratch FrameWorkTileResidualScratch
	scratch.paletteArena = make([]tile.PaletteModeScratch, 0, 1)
	src := make([]tile.PaletteModeScratch, 1)
	scratch.fused = tile.BlockPayloads{Palette: src}
	visit := tile.BlockLoopVisit{
		Prediction: tile.BlockPredictionModeResult{
			Valid: true,
			Intra: true,
			Palette: tile.PaletteModeResult{
				YSize: 2, UVSize: 2, YMapSlot: 1, UVMapSlot: 1,
			},
		},
	}
	src[0].Y[0], src[0].UV[0] = 7, 9
	scratch.bufferDeferredVisit(&visit)
	src[0].Y[0], src[0].UV[0] = 11, 13
	scratch.bufferDeferredVisit(&visit)
	src[0].Y[0], src[0].UV[0] = 99, 99

	pl := scratch.deferredPayloads()
	want := [][2]uint8{{7, 9}, {11, 13}}
	for i := range scratch.reconVisits {
		pal := &scratch.reconVisits[i].Prediction.Palette
		y, uv := pl.PaletteYMap(pal), pl.PaletteUVMap(pal)
		if y == nil || uv == nil {
			t.Fatalf("visit %d: maps not copied", i)
		}
		if y[0] != want[i][0] || uv[0] != want[i][1] {
			t.Fatalf("visit %d: maps %d/%d, want %v", i, y[0], uv[0], want[i])
		}
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

func TestFrameWorkTileResidualRunnerRunAllocs(t *testing.T) {
	ctx, _, _, _, req := testFrameWorkResidualDriver(t)
	var runner FrameWorkTileResidualRunner
	runner.Workers = make([]FrameWorkTileResidualRunnerWorker, 1)
	runner.Workers[0].Int32Scratch = req.Int32Scratch
	runner.Workers[0].ResidualScratch = req.ResidualScratch
	allocs := testing.AllocsPerRun(1000, func() {
		if err := runner.Run(ctx); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("FrameWorkTileResidualRunner.Run allocated: %f", allocs)
	}
}

func TestFrameWorkTileResidualRunnerPoolRunnerAllocs(t *testing.T) {
	ctx, _, _, _, req := testFrameWorkResidualDriver(t)
	var batches [1]Batch
	n, err := BuildBatches(batches[:], ctx.Jobs, 1)
	if err != nil {
		t.Fatal(err)
	}
	pool, err := NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer pool.Close()

	var runner FrameWorkTileResidualRunner
	runner.Workers = make([]FrameWorkTileResidualRunnerWorker, 1)
	runner.Workers[0].Int32Scratch = req.Int32Scratch
	runner.Workers[0].ResidualScratch = req.ResidualScratch
	allocs := testing.AllocsPerRun(1000, func() {
		if err := pool.ExecuteFrameWorkRunner(batches[:n], ctx.Jobs, ctx, &runner); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("FrameWorkTileResidualRunner Pool.ExecuteFrameWorkRunner allocated: %f", allocs)
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
			Jobs: []tile.Job{{SBCols: 1, SBRows: 1, Offset: 0, Size: uint32(len(payload))}},
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
			Loop:            loopReq,
			TransformMode:   ctx.TransformRef.TransformMode,
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
		Loop:            loopReq,
		TransformMode:   ctx.TransformRef.TransformMode,
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

func testFrameWorkResidualScratch(t *testing.T, ctx FrameWorkBatch, size transform.Size, typ transform.Type) ([]int32, []int16) {
	t.Helper()
	_, lossless, err := ctx.BlockQuantizer(ctx.Quantization.BaseQIdx, 0, FrameWorkPlaneY)
	if err != nil {
		t.Fatal(err)
	}
	int32Len, int16Len, err := reconstruct.ScratchLen(reconstruct.Block{
		Size:      size,
		Transform: typ,
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

// TestFrameWorkDeferredPaletteViewIncremental covers the streamed
// wavefront's use of the palette arena: views are published a prefix at a
// time while later blocks keep appending (and the arena keeps moving), and
// every published map must keep reading its own block's colours.
func TestFrameWorkDeferredPaletteViewIncremental(t *testing.T) {
	var scratch FrameWorkTileResidualScratch
	scratch.paletteArena = make([]tile.PaletteModeScratch, 0, 1)
	src := make([]tile.PaletteModeScratch, 1)
	scratch.fused = tile.BlockPayloads{Palette: src}
	appendVisit := func(v uint8) {
		src[0].Y[0], src[0].UV[0] = v, v+1
		scratch.bufferDeferredVisit(&tile.BlockLoopVisit{
			Prediction: tile.BlockPredictionModeResult{
				Valid: true,
				Intra: true,
				Palette: tile.PaletteModeResult{
					YSize: 2, UVSize: 2, YMapSlot: 1, UVMapSlot: 1,
				},
			},
		})
	}
	type view struct {
		visits []tile.BlockLoopVisit
		pl     tile.BlockPayloads
	}
	var views []view
	for round := 0; round < 6; round++ {
		for k := 0; k < round+1; k++ {
			appendVisit(uint8(2 * len(scratch.reconVisits)))
		}
		views = append(views, view{scratch.reconVisits, scratch.deferredPayloads()})
	}
	src[0].Y[0], src[0].UV[0] = 255, 255
	for _, v := range views {
		for i := range v.visits {
			pal := &v.visits[i].Prediction.Palette
			y, uv := v.pl.PaletteYMap(pal), v.pl.PaletteUVMap(pal)
			if y == nil || uv == nil {
				t.Fatalf("visit %d: maps not copied", i)
			}
			if got, want := y[0], uint8(2*i); got != want {
				t.Fatalf("visit %d: Y map = %d, want %d", i, got, want)
			}
			if got, want := uv[0], uint8(2*i+1); got != want {
				t.Fatalf("visit %d: UV map = %d, want %d", i, got, want)
			}
		}
	}
}

// TestFrameWorkCFArenaZeroAfterFailedJob checks that a job that failed with
// coefficients still in the arena, including a decode_coefs slot written past
// the committed length, leaves the next job a zero arena.
func TestFrameWorkCFArenaZeroAfterFailedJob(t *testing.T) {
	var s FrameWorkTileResidualScratch
	full := make([]int32, 4096)
	for i := range full[:100+1024] {
		full[i] = int32(i + 1)
	}
	s.cfArena = full[:100]
	s.resetDeferredReconBuffers()
	s.zeroCFArena(false)
	for i, v := range full {
		if v != 0 {
			t.Fatalf("cf arena[%d] = %d after a failed job, want 0", i, v)
		}
	}
	if s.cfDirty != 0 {
		t.Fatalf("cfDirty = %d after clearing", s.cfDirty)
	}
}

var wavefrontEnsureStatesBenchSink int

// buildWavefrontReconController builds a controller whose deferred buffers are
// pre-filled with one 64x64 intra-DC skip-transform block per superblock of an
// sbCols x sbRows single-tile frame, in SB-raster order. Skip-transform intra
// blocks predict luma+chroma directly from neighbors with no coefficients, so
// they exercise the wavefront neighbor dependency without needing a real
// entropy decode. output receives the reconstructed pixels.
func buildWavefrontReconController(t testing.TB, output *frame.Frame, sbCols int, sbRows int, scratch *FrameWorkTileResidualScratch) *frameWorkTileResidualLoopController {
	t.Helper()
	batch := FrameWorkBatch{
		Output: output,
		FrameWorkFrameContext: FrameWorkFrameContext{
			Sequence: FrameWorkSequenceContextFromHeader(parser.SequenceHeader{
				ColorConfig: parser.ColorConfig{
					BitDepth:     output.Format.BitDepth,
					MonoChrome:   output.Format.MonoChrome,
					SubsamplingX: output.Format.SubsamplingX,
					SubsamplingY: output.Format.SubsamplingY,
				},
			}),
			FrameSize: parser.FrameSize{CodedWidth: uint32(output.Format.Width), Height: uint32(output.Format.Height)},
		},
		Jobs: []tile.Job{{SBCols: uint16(sbCols), SBRows: uint16(sbRows)}},
	}
	scratch.geomCache.reset()
	scratch.resetDeferredReconBuffers()
	batch.geomCache = &scratch.geomCache

	var stats FrameWorkTileResidualStats
	c := &scratch.controller
	*c = frameWorkTileResidualLoopController{
		batch:   batch,
		index:   0,
		scratch: scratch,
		stats:   &stats,
	}

	const sbMIB = 16 // 64x64 superblock = 16 4x4 MI units
	for r := range sbRows {
		for col := range sbCols {
			miCol := uint16(col * sbMIB)
			miRow := uint16(r * sbMIB)
			visit := tile.BlockLoopVisit{
				Block: tile.BlockVisit{
					MICol: miCol, MIRow: miRow,
					MIColEnd: miCol + sbMIB, MIRowEnd: miRow + sbMIB,
					X4: uint8(miCol), Y4: uint8(miRow),
					Size:      tile.BlockSize64x64,
					VisibleW4: sbMIB, VisibleH4: sbMIB,
					HaveTop:  r > 0,
					HaveLeft: col > 0,
				},
				Prediction: tile.BlockPredictionModeResult{
					Valid:           true,
					Intra:           true,
					LumaMode:        tile.IntraModeDC,
					ChromaMode:      tile.ChromaIntraModeDC,
					ChromaModeValid: true,
				},
			}
			visit.Prefix.SkipTransform = true
			// Skip intra blocks are predicted transform block by transform
			// block over the block's transform layout (dav1d recon_b_intra):
			// a 16x16 luma / 8x8 chroma grid gives each superblock 16 luma
			// and 2x16 chroma predictions, with no cbi entries.
			visit.Coefficients.Tree = tile.TransformTreeResult{Y: tile.TransformSize16x16, UV: tile.TransformSize8x8, HasUV: true}
			scratch.reconVisits = append(scratch.reconVisits, visit)
		}
	}
	return c
}

func TestReconWavefrontByteIdenticalToSerial(t *testing.T) {
	const sbCols, sbRows = 8, 8 // 512x512, 8 SB rows
	format := frame.Format{Width: sbCols * 64, Height: sbRows * 64, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 64}

	serialOut := testBatchFrame(t, format)
	wavefrontOut := testBatchFrame(t, format)
	// Seed the (past-visible) borders deterministically so neighbor reads are
	// well-defined and identical for both runs.
	testFillFrame(serialOut, 0)
	testFillFrame(wavefrontOut, 0)

	var predict, wfPredict FrameWorkPredictionScratch
	var serialScratch, wavefrontScratch FrameWorkTileResidualScratch

	cSerial := buildWavefrontReconController(t, serialOut, sbCols, sbRows, &serialScratch)
	cSerial.recon = frameWorkReconState{
		batch:             cSerial.batch,
		index:             0,
		stats:             cSerial.stats,
		predictionScratch: &predict,
		cflScratch:        &serialScratch.CFL,
	}
	if err := cSerial.testReplayDeferredSerial(); err != nil {
		t.Fatalf("serial replay: %v", err)
	}

	for _, workers := range []int{2, 3, 4} {
		testFillFrame(wavefrontOut, 0)
		cWave := buildWavefrontReconController(t, wavefrontOut, sbCols, sbRows, &wavefrontScratch)
		cWave.wavefrontWorkers = uint16(workers)
		cWave.recon = frameWorkReconState{
			batch:             cWave.batch,
			index:             0,
			stats:             cWave.stats,
			predictionScratch: &wfPredict,
			cflScratch:        &wavefrontScratch.CFL,
		}
		if err := cWave.testReplayDeferredWavefront(); err != nil {
			t.Fatalf("wavefront replay workers=%d: %v", workers, err)
		}
		if !equalFramePlanes(serialOut, wavefrontOut) {
			t.Fatalf("workers=%d wavefront output differs from serial replay", workers)
		}
	}
}

func TestReconWavefrontScratchUsesFlatArenas(t *testing.T) {
	var wf frameWorkReconWavefront
	c := frameWorkTileResidualLoopController{
		req: FrameWorkTileResidualRequest{
			Int32Scratch:    make([]int32, 7),
			ResidualScratch: make([]int16, 11),
		},
	}
	if err := wf.ensureStates(3, &c); err != nil {
		t.Fatal(err)
	}
	if got, want := len(wf.int32Arena), 21; got != want {
		t.Fatalf("int32 arena len=%d want %d", got, want)
	}
	if got, want := len(wf.residualArena), 33; got != want {
		t.Fatalf("residual arena len=%d want %d", got, want)
	}
	for w := range 3 {
		st := &wf.states[w]
		if got, want := len(st.int32Scratch), 7; got != want {
			t.Fatalf("worker %d int32 len=%d want %d", w, got, want)
		}
		if got, want := len(st.residualScratch), 11; got != want {
			t.Fatalf("worker %d residual len=%d want %d", w, got, want)
		}
		if &st.int32Scratch[0] != &wf.int32Arena[w*7] {
			t.Fatalf("worker %d int32 scratch is not arena-backed", w)
		}
		if &st.residualScratch[0] != &wf.residualArena[w*11] {
			t.Fatalf("worker %d residual scratch is not arena-backed", w)
		}
		if w > 0 && &wf.states[w-1].int32Scratch[6] == &st.int32Scratch[0] {
			t.Fatalf("worker %d int32 scratch overlaps previous worker", w)
		}
		if w > 0 && &wf.states[w-1].residualScratch[10] == &st.residualScratch[0] {
			t.Fatalf("worker %d residual scratch overlaps previous worker", w)
		}
	}
}

func testReconWavefrontTwoRows() *frameWorkReconWavefront {
	wf := &frameWorkReconWavefront{
		rowStart: []int32{0, 1, 2},
		sbSpans: []frameWorkReconSB{
			{start: 0, end: 0, col: 0},
			{start: 0, end: 0, col: 0},
		},
	}
	wf.ensureDone(2)
	wf.resetProgress(2)
	return wf
}

func TestReconWavefrontProgressBlocksUntilPublished(t *testing.T) {
	wf := testReconWavefrontTwoRows()
	done := make(chan error, 1)
	go func() {
		var st frameWorkReconState
		view := frameWorkReconView{rowStart: wf.rowStart, sbs: wf.sbSpans}
		done <- wf.reconstructRow(&st, &view, 1, 2)
	}()

	select {
	case err := <-done:
		t.Fatalf("row reconstructed before upper-row progress was published: %v", err)
	case <-time.After(10 * time.Millisecond):
	}

	wf.publishRowProgress(&wf.done[0], 1)
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("reconstruct row after progress: %v", err)
		}
	case <-time.After(time.Second):
		t.Fatal("row did not reconstruct after upper-row progress was published")
	}
}

func TestReconWavefrontAbortWakesWaiter(t *testing.T) {
	wf := testReconWavefrontTwoRows()
	done := make(chan error, 1)
	go func() {
		var st frameWorkReconState
		view := frameWorkReconView{rowStart: wf.rowStart, sbs: wf.sbSpans}
		done <- wf.reconstructRow(&st, &view, 1, 2)
	}()

	select {
	case err := <-done:
		t.Fatalf("row reconstructed before abort: %v", err)
	case <-time.After(10 * time.Millisecond):
	}

	wf.abortProgress()
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("reconstruct row after abort: %v", err)
		}
	case <-time.After(time.Second):
		t.Fatal("row waiter did not wake after abort")
	}
}

func BenchmarkReconWavefrontEnsureStates(b *testing.B) {
	c := frameWorkTileResidualLoopController{
		req: FrameWorkTileResidualRequest{
			Int32Scratch:    make([]int32, 4096),
			ResidualScratch: make([]int16, 4096),
		},
	}
	b.ReportAllocs()
	for b.Loop() {
		var wf frameWorkReconWavefront
		if err := wf.ensureStates(8, &c); err != nil {
			b.Fatal(err)
		}
		wavefrontEnsureStatesBenchSink += len(wf.states) + len(wf.int32Arena) + len(wf.residualArena)
	}
}

// testReplayDeferredSerial reconstructs every buffered block in decode order on
// the controller's own reconstruction state.
func (c *frameWorkTileResidualLoopController) testReplayDeferredSerial() error {
	c.recon.payloads = c.scratch.deferredPayloads()
	c.recon.batch.payloads = &c.recon.payloads
	cur := frameWorkReconCursor{cbi: c.scratch.reconCBI, cf: c.scratch.cfArena}
	for i := range c.scratch.reconVisits {
		if err := c.recon.reconBDeferred(&c.scratch.reconVisits[i], &cur); err != nil {
			return err
		}
	}
	return nil
}

// testReplayDeferredWavefront reconstructs a fully buffered tile of one 64x64
// skip block (no cbi entries or coefficients) per superblock through the
// SB-row wavefront lanes.
func (c *frameWorkTileResidualLoopController) testReplayDeferredWavefront() error {
	region, err := c.batch.JobRegion(c.index)
	if err != nil {
		return err
	}
	sbSizeMIB := int(c.batch.Sequence.SBSizeMIB)
	rowCount := int(region.SBRows)
	if sbSizeMIB == 0 || rowCount <= 0 {
		return ErrInvalidBatch
	}
	workers := min(int(c.wavefrontWorkers), rowCount)
	if c.scratch.wavefront == nil {
		c.scratch.wavefront = &frameWorkReconWavefront{}
	}
	wf := c.scratch.wavefront
	wf.beginSpans(region, sbSizeMIB)
	for i := range c.scratch.reconVisits {
		v := &c.scratch.reconVisits[i]
		if !v.Prefix.SkipTransform || v.Block.Size != tile.BlockSize64x64 {
			return ErrInvalidBatch
		}
		if _, err := wf.noteSuperblock(i, 0, 0, int(v.Block.MIRow), int(v.Block.MICol)); err != nil {
			return err
		}
	}
	if err := wf.closeSpans(len(c.scratch.reconVisits)); err != nil {
		return err
	}
	wf.ensureDone(rowCount)
	wf.resetProgress(rowCount)
	if err := wf.ensureStates(workers, c); err != nil {
		return err
	}
	wf.startLanes(c.batch.wavefrontLanes, workers, rowCount)
	wf.publishRows(rowCount, frameWorkReconView{
		rowStart: wf.rowStart,
		sbs:      wf.sbSpans,
		visits:   c.scratch.reconVisits,
		cbi:      c.scratch.reconCBI,
		cfArena:  c.scratch.cfArena,
	})
	return wf.finishLanes(c.stats)
}

func equalFramePlanes(a *frame.Frame, b *frame.Frame) bool {
	return string(a.Y.Pix) == string(b.Y.Pix) &&
		string(a.U.Pix) == string(b.U.Pix) &&
		string(a.V.Pix) == string(b.V.Pix)
}

// BenchmarkReconWavefrontReplay isolates the parallelized reconstruction phase:
// the serial deferred replay against the four-lane wavefront on a large
// single-tile workload (2048x2048, 32 superblock rows).
func BenchmarkReconWavefrontReplay(b *testing.B) {
	const sbCols, sbRows = 32, 32
	format := frame.Format{Width: sbCols * 64, Height: sbRows * 64, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 64}
	out := testBatchFrame(b, format)

	var predict FrameWorkPredictionScratch
	var scratch FrameWorkTileResidualScratch
	for _, workers := range []int{1, 4} {
		name := "serial"
		if workers > 1 {
			name = "wavefront4"
		}
		b.Run(name, func(b *testing.B) {
			for b.Loop() {
				b.StopTimer()
				testFillFrame(out, 0)
				c := buildWavefrontReconController(b, out, sbCols, sbRows, &scratch)
				c.recon = frameWorkReconState{
					batch:             c.batch,
					index:             0,
					stats:             c.stats,
					predictionScratch: &predict,
					cflScratch:        &scratch.CFL,
				}
				b.StartTimer()
				var err error
				if workers > 1 {
					c.wavefrontWorkers = uint16(workers)
					err = c.testReplayDeferredWavefront()
				} else {
					err = c.testReplayDeferredSerial()
				}
				if err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// TestFrameWorkWavefrontLanesCapsToRows checks that a pool wider than the tile
// has wavefront rows for runs on fewer lanes instead of falling back to the
// fused serial path.
func TestFrameWorkWavefrontLanesCapsToRows(t *testing.T) {
	format := frame.Format{Width: 1280, Height: 720, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 64}
	for _, tc := range []struct {
		sb128   bool
		workers uint16
		want    uint16
	}{
		{sb128: true, workers: 1, want: 0},  // no idle lanes
		{sb128: true, workers: 2, want: 2},  // 6 SB rows
		{sb128: true, workers: 3, want: 3},  // 6 rows / 2 per lane
		{sb128: true, workers: 4, want: 3},  // capped, not disabled
		{sb128: true, workers: 16, want: 3}, // capped, not disabled
		{sb128: false, workers: 8, want: 6}, // 12 SB rows of 64x64
	} {
		b := wavefrontLanesTestBatch(t, format, tc.sb128)
		b.WavefrontWorkers = tc.workers
		if got := frameWorkWavefrontLanes(b, 0); got != tc.want {
			t.Errorf("sb128=%v workers=%d: lanes=%d, want %d", tc.sb128, tc.workers, got, tc.want)
		}
		b.FrameSize.AllowIntrabc = true
		if got := frameWorkWavefrontLanes(b, 0); got != 0 {
			t.Errorf("sb128=%v workers=%d intrabc: lanes=%d, want 0", tc.sb128, tc.workers, got)
		}
	}
}

func wavefrontLanesTestBatch(t *testing.T, format frame.Format, sb128 bool) FrameWorkBatch {
	t.Helper()
	sb := 64
	if sb128 {
		sb = 128
	}
	b := FrameWorkBatch{
		FrameWorkFrameContext: FrameWorkFrameContext{
			Sequence: FrameWorkSequenceContextFromHeader(parser.SequenceHeader{
				Use128x128Superblock: sb128,
				ColorConfig: parser.ColorConfig{
					BitDepth:     format.BitDepth,
					SubsamplingX: format.SubsamplingX,
					SubsamplingY: format.SubsamplingY,
				},
			}),
			FrameSize: parser.FrameSize{CodedWidth: uint32(format.Width), Height: uint32(format.Height)},
		},
		Jobs: []tile.Job{{SBCols: uint16((format.Width + sb - 1) / sb), SBRows: uint16((format.Height + sb - 1) / sb)}},
	}
	return b
}
