package threading

import (
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/loopfilter"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/tile"
)

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
	if got.Stride != cols || got.Rows != rows || len(got.Records) != length {
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
	if _, _, _, err := (FrameWorkBatch{}).LoopFilterMapShape(); !errors.Is(err, ErrInvalidBatch) {
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
				Ref: [2]tile.ReferenceFrame{tile.ReferenceFrameGolden, tile.ReferenceFrameNone},
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

	for miRow := uint32(2); miRow < 5; miRow++ {
		for miCol := uint32(4); miCol < 7; miCol++ {
			got := filterMap.Records[int(miRow)*filterMap.Stride+int(miCol)]
			if !got.Valid || got.Block != visit.Block || got.TransformTree != visit.Coefficients.Tree {
				t.Fatalf("record[%d,%d]=%+v", miCol, miRow, got)
			}
			if got.SegmentID != 5 || !got.SkipTransform || got.Intra {
				t.Fatalf("record[%d,%d] flags=%+v", miCol, miRow, got)
			}
			if got.RefFrame != int(tile.ReferenceFrameGolden)+1 || got.Mode != loopfilter.ModeDeltaClassMotion {
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
				Ref:      [2]tile.ReferenceFrame{tile.ReferenceFrameLast, tile.ReferenceFrameAltref},
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
	if got := filterMap.Records[2]; !got.Valid || got.Intra || got.RefFrame != int(tile.ReferenceFrameLast)+1 || got.Mode != loopfilter.ModeDeltaClassMotion {
		t.Fatalf("compound record=%+v", got)
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
				Ref: [2]tile.ReferenceFrame{tile.ReferenceFrameLast, tile.ReferenceFrameNone},
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
	badRef.Prediction.InterReferences.Ref[0] = tile.ReferenceFrameNone
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
				Ref: [2]tile.ReferenceFrame{tile.ReferenceFrameLast, tile.ReferenceFrameNone},
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
		if err := filterMap.Reset(); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("FrameWorkLoopFilterMap allocated: %f", allocs)
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
