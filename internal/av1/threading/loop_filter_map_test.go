package threading

import (
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/loopfilter"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/tile"
)

func TestFrameWorkBatchLoopFilterMapShapeAndBind(t *testing.T) {
	ctx := testFrameWorkLoopFilterMapBatch(130, 65)
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
		t.Fatal("bind did not reset in-map records")
	}
	if !records[length].Valid {
		t.Fatal("bind reset caller storage past map")
	}

	if _, err := ctx.BindLoopFilterMap(records[:length-1]); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("short bind err=%v want %v", err, ErrInvalidBatch)
	}
	if _, _, _, err := (FrameWorkBatch{}).LoopFilterMapShape(); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("invalid shape err=%v want %v", err, ErrInvalidBatch)
	}
}

func TestFrameWorkLoopFilterMapMarkBlock(t *testing.T) {
	ctx := testFrameWorkLoopFilterMapBatch(128, 64)
	_, _, length, err := ctx.LoopFilterMapShape()
	if err != nil {
		t.Fatal(err)
	}
	lfMap, err := ctx.BindLoopFilterMap(make([]FrameWorkLoopFilterBlockRecord, length))
	if err != nil {
		t.Fatal(err)
	}
	state := tile.DecodeState{
		DeltaLFFromBase: -3,
		DeltaLF:         [tile.FrameLoopFilterCount]int8{1, 2, -2, 4},
	}
	visit := tile.BlockLoopVisit{
		Block: tile.BlockVisit{
			MICol: 15, MIRow: 2, MIColEnd: 18, MIRowEnd: 4,
			Size: tile.BlockSize16x8, VisibleW4: 3, VisibleH4: 2,
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
			Tree: tile.TransformTreeResult{
				Y:        tile.TransformSize8x8,
				Variable: true,
				Split:    [2]uint16{3, 1},
			},
		},
	}
	if err := lfMap.MarkBlock(visit, &state); err != nil {
		t.Fatal(err)
	}

	for miRow := uint32(2); miRow < 4; miRow++ {
		for miCol := uint32(15); miCol < 18; miCol++ {
			record := lfMap.Records[int(miRow)*lfMap.Stride+int(miCol)]
			if !record.Valid || record.Block != visit.Block ||
				record.TransformTree != visit.Coefficients.Tree ||
				record.SegmentID != 5 || !record.SkipTransform || record.Intra ||
				record.RefFrame != int(tile.ReferenceFrameGolden)+1 ||
				record.Mode != loopfilter.ModeDeltaClassMotion ||
				record.DeltaLFFromBase != -3 || record.DeltaLF != state.DeltaLF {
				t.Fatalf("record (%d,%d)=%+v", miCol, miRow, record)
			}
		}
	}
	if lfMap.Records[0].Valid {
		t.Fatal("mark touched unrelated MI cell")
	}
	if err := lfMap.Reset(); err != nil {
		t.Fatal(err)
	}
	for i, record := range lfMap.Records {
		if record.Valid {
			t.Fatalf("reset record %d still valid: %+v", i, record)
		}
	}
}

func TestFrameWorkLoopFilterMapMarkBlockRejectsInvalidInputs(t *testing.T) {
	ctx := testFrameWorkLoopFilterMapBatch(64, 64)
	lfMap, err := ctx.BindLoopFilterMap(make([]FrameWorkLoopFilterBlockRecord, 16*16))
	if err != nil {
		t.Fatal(err)
	}
	valid := tile.BlockLoopVisit{
		Block: tile.BlockVisit{MICol: 0, MIRow: 0, MIColEnd: 4, MIRowEnd: 4},
	}
	var state tile.DecodeState
	if err := lfMap.MarkBlock(valid, nil); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("nil state err=%v want %v", err, ErrInvalidBatch)
	}
	outside := valid
	outside.Block.MICol = 16
	outside.Block.MIColEnd = 20
	if err := lfMap.MarkBlock(outside, &state); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("outside err=%v want %v", err, ErrInvalidBatch)
	}
	empty := valid
	empty.Block.MIColEnd = empty.Block.MICol
	if err := lfMap.MarkBlock(empty, &state); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("empty err=%v want %v", err, ErrInvalidBatch)
	}
	badRef := valid
	badRef.Prediction = tile.BlockPredictionModeResult{
		Valid:                true,
		InterReferencesValid: true,
		InterReferences:      tile.InterReferencesResult{Ref: [2]tile.ReferenceFrame{tile.ReferenceFrameNone, tile.ReferenceFrameNone}},
	}
	if err := lfMap.MarkBlock(badRef, &state); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("bad ref err=%v want %v", err, ErrInvalidBatch)
	}

	shortMap := FrameWorkLoopFilterMap{Records: nil, Stride: 1, Rows: 1}
	if err := shortMap.MarkBlock(valid, &state); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("short mark err=%v want %v", err, ErrInvalidBatch)
	}
	if err := shortMap.Reset(); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("short reset err=%v want %v", err, ErrInvalidBatch)
	}
}

func TestFrameWorkLoopFilterMapAllocs(t *testing.T) {
	ctx := testFrameWorkLoopFilterMapBatch(64, 64)
	records := make([]FrameWorkLoopFilterBlockRecord, 16*16)
	visit := tile.BlockLoopVisit{
		Block: tile.BlockVisit{MICol: 0, MIRow: 0, MIColEnd: 4, MIRowEnd: 4},
		Prediction: tile.BlockPredictionModeResult{
			Valid: true,
			Intra: true,
		},
	}
	var state tile.DecodeState
	allocs := testing.AllocsPerRun(1000, func() {
		lfMap, err := ctx.BindLoopFilterMap(records)
		if err != nil {
			t.Fatal(err)
		}
		if err := lfMap.MarkBlock(visit, &state); err != nil {
			t.Fatal(err)
		}
		if err := lfMap.Reset(); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("FrameWorkLoopFilterMap allocated: %f", allocs)
	}
}

func testFrameWorkLoopFilterMapBatch(width uint32, height uint32) FrameWorkBatch {
	return FrameWorkBatch{
		FrameWorkFrameContext: FrameWorkFrameContext{
			Sequence: FrameWorkSequenceContextFromHeader(parser.SequenceHeader{
				ColorConfig: parser.ColorConfig{BitDepth: 8},
			}),
			FrameSize: parser.FrameSize{CodedWidth: width, Height: height},
		},
	}
}
