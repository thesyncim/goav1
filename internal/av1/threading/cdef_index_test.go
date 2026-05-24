package threading

import (
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/tile"
)

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
	if got.Stride != cols || got.Rows != rows || len(got.Index) != length || len(got.Read) != length {
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

	if _, _, _, err := (FrameWorkBatch{}).CDEFIndexMapShape(); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("invalid batch shape err=%v want %v", err, ErrInvalidBatch)
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
	if cdefMap.Read[0] || cdefMap.Index[0] != 0 {
		t.Fatalf("single strength marked read=%v index=%d want false,0", cdefMap.Read[0], cdefMap.Index[0])
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
	if err := controller.BeforeBlockCoefficients(first); err != nil {
		t.Fatal(err)
	}
	if !cdefMap.Read[0] || cdefMap.Index[0] != 2 || stats.CoefficientBlocks != 1 {
		t.Fatalf("first map read=%v index=%d stats=%+v", cdefMap.Read[0], cdefMap.Index[0], stats)
	}

	skip := tile.BlockLoopVisit{
		Block:  tile.BlockVisit{MICol: 16, MIRow: 0, MIColEnd: 32, MIRowEnd: 16},
		Prefix: tile.BlockModeResult{SkipTransform: true, CDEFIndex: 3},
	}
	if err := controller.BeforeBlockCoefficients(skip); err != nil {
		t.Fatal(err)
	}
	if cdefMap.Read[1] || cdefMap.Index[1] != 0 || stats.SkippedBlocks != 1 {
		t.Fatalf("skip map read=%v index=%d stats=%+v", cdefMap.Read[1], cdefMap.Index[1], stats)
	}

	second := skip
	second.Prefix.SkipTransform = false
	if err := controller.BeforeBlockCoefficients(second); err != nil {
		t.Fatal(err)
	}
	if !cdefMap.Read[1] || cdefMap.Index[1] != 3 || stats.CoefficientBlocks != 2 {
		t.Fatalf("second map read=%v index=%d stats=%+v", cdefMap.Read[1], cdefMap.Index[1], stats)
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
