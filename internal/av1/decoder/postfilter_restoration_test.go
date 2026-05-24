package decoder

import (
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/tile"
)

func TestFrameWorkPostFilterContextApplyLoopRestorationPostFilterAllNoneRecords(t *testing.T) {
	const width = 64
	const height = 64

	pool := testFramePoolForSize(t, width, height, 1)
	_, output, err := pool.Acquire()
	if err != nil {
		t.Fatal(err)
	}
	output.Y.Pix[0] = 0x5a

	event := Event{
		SequenceHeader: testSequence(),
		FrameSize: parser.FrameSize{
			CodedWidth:          width,
			UpscaledWidth:       width,
			Height:              height,
			SuperResDenominator: 8,
		},
		Restoration: parser.RestorationParams{
			Type:      [3]parser.RestorationType{parser.RestorationWiener, parser.RestorationNone, parser.RestorationNone},
			UnitSizeY: 64,
		},
	}
	ctx := FrameWorkPostFilterContext{Event: event, Output: output}
	if got := ctx.ActivePostFilters(); got != FrameWorkPostFilterLoopRestoration {
		t.Fatalf("ActivePostFilters=%b want %b", got, FrameWorkPostFilterLoopRestoration)
	}

	plan, err := ctx.LoopRestorationPostFilterPlan()
	if err != nil {
		t.Fatal(err)
	}
	if !plan.Active || plan.UnitRecordLen() != 1 {
		t.Fatalf("plan active=%v records=%d", plan.Active, plan.UnitRecordLen())
	}
	records, err := tile.BindRestorationFrameRecordBuffers(plan, make([]tile.RestorationUnitRecord, plan.UnitRecordLen()))
	if err != nil {
		t.Fatal(err)
	}
	for plane := 0; plane < int(plan.Planes); plane++ {
		if plan.Grids[plane].Type == parser.RestorationNone {
			continue
		}
		if err := tile.ResetRestorationPlaneRecords(plan.Grids[plane], records[plane]); err != nil {
			t.Fatal(err)
		}
	}

	size, err := ctx.LoopRestorationPostFilterScratchLen(records, false)
	if err != nil {
		t.Fatal(err)
	}
	result, err := ctx.ApplyLoopRestorationPostFilter(FrameWorkRestorationPostFilterRequest{
		Records:     records,
		DataScratch: make([]uint16, size.Samples.DataLen),
		DstScratch:  make([]uint16, size.Samples.DstLen),
		Scratch:     testFrameWorkRestorationPostFilterScratch(size.Apply),
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Planes != 1 || result.Records != 1 || result.FilteredRecords != 0 {
		t.Fatalf("result=%+v", result)
	}
	if output.Y.Pix[0] != 0x5a {
		t.Fatalf("output sample=%d want 0x5a", output.Y.Pix[0])
	}
}

func TestFrameWorkPostFilterContextApplyLoopRestorationPostFilterRejectsEarlierActiveStages(t *testing.T) {
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			FrameSize: parser.FrameSize{
				CodedWidth:          64,
				UpscaledWidth:       64,
				Height:              64,
				SuperResDenominator: 8,
			},
			CDEF: parser.CDEFParams{
				StrengthCount: 1,
				YStrength:     [parser.MaxCDEFStrengths]uint8{1},
			},
			Restoration: parser.RestorationParams{
				Type:      [3]parser.RestorationType{parser.RestorationWiener},
				UnitSizeY: 64,
			},
		},
	}
	_, err := ctx.ApplyLoopRestorationPostFilter(FrameWorkRestorationPostFilterRequest{})
	if !errors.Is(err, ErrUnsupportedPostFilter) {
		t.Fatalf("ApplyLoopRestorationPostFilter err=%v want %v", err, ErrUnsupportedPostFilter)
	}
}

func testFrameWorkRestorationPostFilterScratch(size tile.RestorationUnitRecordBoundaryScratchSize) tile.RestorationUnitRecordBoundaryScratch {
	return tile.RestorationUnitRecordBoundaryScratch{
		Unit: tile.RestorationUnitScratch{
			Wiener:  make([]uint16, size.Unit.Wiener),
			SGRProj: make([]int32, size.Unit.SGRProj),
		},
		Boundary: tile.RestorationStripeBoundaryScratch{
			Above: make([]uint16, size.Boundary.Above),
			Below: make([]uint16, size.Boundary.Below),
		},
	}
}
