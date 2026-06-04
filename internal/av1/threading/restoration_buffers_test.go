package threading

import (
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/tile"
)

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
