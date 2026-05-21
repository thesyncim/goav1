package threading

import (
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/tile"
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
	})
	if allocs != 0 {
		t.Fatalf("FrameWorkBatch.JobRegion allocated: %f", allocs)
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
			Output: output,
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
