package goav1

import (
	"errors"
	"testing"

	internalparser "github.com/thesyncim/goav1/internal/av1/parser"
	internalrestoration "github.com/thesyncim/goav1/internal/av1/restoration"
	internaltile "github.com/thesyncim/goav1/internal/av1/tile"
)

func TestPublicTileRestorationFramePlanAndBuffers(t *testing.T) {
	params := internalparser.RestorationParams{
		Type:       [3]internalparser.RestorationType{internalparser.RestorationWiener, internalparser.RestorationSGRProj, internalparser.RestorationNone},
		UnitSizeY:  128,
		UnitSizeUV: 64,
	}
	size := FrameSize{UpscaledWidth: 300, Height: 260, SuperResDenominator: 8}
	color := internalparser.ColorConfig{SubsamplingX: true, SubsamplingY: true}

	y, err := internaltile.BuildRestorationPlaneGrid(params, size, color, 0)
	if err != nil {
		t.Fatal(err)
	}
	if y.Type != internalparser.RestorationWiener || y.UnitSize != 128 || y.HorzUnits != 2 || y.VertUnits != 2 {
		t.Fatalf("y grid=%+v", y)
	}
	u, err := internaltile.BuildRestorationPlaneGrid(params, size, color, 1)
	if err != nil {
		t.Fatal(err)
	}
	if u.Type != internalparser.RestorationSGRProj || u.UnitSize != 64 || !u.SubsamplingX || !u.SubsamplingY ||
		u.HorzUnits != 2 || u.VertUnits != 2 {
		t.Fatalf("u grid=%+v", u)
	}

	plan, err := internaltile.BuildRestorationFramePlan(params, size, color)
	if err != nil {
		t.Fatal(err)
	}
	if !plan.Active || plan.Planes != 3 || plan.UnitRecords != ([3]uint32{4, 4, 0}) || plan.UnitRecordLen() != 8 {
		t.Fatalf("plan=%+v records=%d", plan, plan.UnitRecordLen())
	}
	if plan.Boundaries[0] != (internaltile.RestorationStripeBoundaryBufferSize{Stride: 320, Rows: 10, Len: 3200}) ||
		plan.Boundaries[1] != (internaltile.RestorationStripeBoundaryBufferSize{Stride: 160, Rows: 10, Len: 1600}) ||
		plan.BoundaryBufferLen() != 4800 {
		t.Fatalf("boundaries=%+v total=%d", plan.Boundaries, plan.BoundaryBufferLen())
	}

	recordBacking := make([]internaltile.RestorationUnitRecord, plan.UnitRecordLen())
	records, err := internaltile.BindRestorationFrameRecordBuffers(plan, recordBacking)
	if err != nil {
		t.Fatal(err)
	}
	if len(records[0]) != 4 || len(records[1]) != 4 || records[2] != nil {
		t.Fatalf("record lens=%d,%d,%d", len(records[0]), len(records[1]), len(records[2]))
	}
	if err := internaltile.ResetRestorationPlaneRecords(plan.Grids[0], records[0]); err != nil {
		t.Fatal(err)
	}
	if records[0][0].Unit.Type != internalparser.RestorationNone || records[0][0].StripeCount == 0 {
		t.Fatalf("record[0]=%+v", records[0][0])
	}
	updated := records[0][0]
	updated.Unit = internaltile.RestorationUnit{Type: internalparser.RestorationWiener, Wiener: internalrestoration.DefaultWienerInfo()}
	records[0][updated.Index] = updated
	if records[0][0].Unit.Type != internalparser.RestorationWiener {
		t.Fatalf("stored record=%+v", records[0][0])
	}

	above := make([]uint16, plan.BoundaryBufferLen())
	below := make([]uint16, plan.BoundaryBufferLen())
	boundaries, err := internaltile.BindRestorationFrameBoundaryBuffers(plan, above, below)
	if err != nil {
		t.Fatal(err)
	}
	if boundaries[0].Stride != 320 || len(boundaries[0].Above) != 3200 ||
		boundaries[1].Stride != 160 || len(boundaries[1].Below) != 1600 {
		t.Fatalf("boundaries=%+v", boundaries)
	}
}

func TestPublicTileRestorationGeometryBoundaryAndApply(t *testing.T) {
	grid, err := internaltile.BuildRestorationPlaneGrid(
		internalparser.RestorationParams{Type: [3]internalparser.RestorationType{internalparser.RestorationWiener}, UnitSizeY: 128, UnitSizeUV: 64},
		FrameSize{UpscaledWidth: 300, Height: 260, SuperResDenominator: 8},
		internalparser.ColorConfig{SubsamplingX: true, SubsamplingY: true},
		0,
	)
	if err != nil {
		t.Fatal(err)
	}
	rect, err := grid.UnitRect(0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if rect.Width() != 128 || rect.Height() != 120 {
		t.Fatalf("rect=%+v width=%d height=%d", rect, rect.Width(), rect.Height())
	}
	stripe, ok, err := grid.ProcessingStripe(rect, 0)
	if err != nil || !ok {
		t.Fatalf("stripe=%+v ok=%v err=%v", stripe, ok, err)
	}
	if stripe.ProcUnitWidth != internalrestoration.ProcUnitSize || stripe.TileStripe != 0 {
		t.Fatalf("stripe=%+v", stripe)
	}
	if scratch, err := internaltile.RestorationStripeBoundaryScratchLen(stripe, false); err != nil || scratch.Above != 0 || scratch.Below == 0 {
		t.Fatalf("boundary scratch=%+v err=%v", scratch, err)
	}

	size, err := internaltile.RestorationUnitScratchLen(64, 32)
	if err != nil {
		t.Fatal(err)
	}
	wantWiener, err := internalrestoration.WienerScratchLen(64, 32)
	if err != nil {
		t.Fatal(err)
	}
	wantSGR, err := internalrestoration.SelfguidedScratchLen(64, 32)
	if err != nil {
		t.Fatal(err)
	}
	if size != (internaltile.RestorationUnitScratchSize{Wiener: wantWiener, SGRProj: wantSGR}) {
		t.Fatalf("unit scratch=%+v want %d/%d", size, wantWiener, wantSGR)
	}

	const width, height = 5, 4
	const srcStride, dstStride = 11, 8
	const origin = 13
	src := make([]uint16, origin+(height-1)*srcStride+width)
	dst := make([]uint16, dstStride*height)
	for i := range dst {
		dst[i] = 0xffff
	}
	for row := range height {
		for col := range width {
			src[origin+row*srcStride+col] = uint16(20 + row*7 + col)
		}
	}
	result, err := internaltile.ApplyRestorationUnit(src, srcStride, origin, dst, dstStride, width, height, internaltile.RestorationUnit{Type: internalparser.RestorationNone}, 8, internaltile.RestorationUnitScratch{})
	if err != nil {
		t.Fatal(err)
	}
	if result != (internaltile.RestorationUnitApplyResult{Type: internalparser.RestorationNone}) {
		t.Fatalf("result=%+v", result)
	}
	for row := range height {
		for col := range width {
			if got, want := dst[row*dstStride+col], src[origin+row*srcStride+col]; got != want {
				t.Fatalf("dst[%d,%d]=%d want %d", row, col, got, want)
			}
		}
	}

	bordered, stride, borderOrigin := publicTileRestorationBordered(3, 2, 2, 1)
	bordered[borderOrigin+0*stride+0] = 10
	bordered[borderOrigin+0*stride+1] = 20
	bordered[borderOrigin+0*stride+2] = 30
	bordered[borderOrigin+1*stride+0] = 40
	bordered[borderOrigin+1*stride+1] = 50
	bordered[borderOrigin+1*stride+2] = 60
	if err := internaltile.ExtendRestorationFrame(bordered, stride, borderOrigin, 3, 2, 2, 1); err != nil {
		t.Fatal(err)
	}
	if bordered[borderOrigin-2] != 10 || bordered[borderOrigin+3] != 30 ||
		bordered[borderOrigin-stride] != 10 || bordered[borderOrigin+stride+4] != 60 {
		t.Fatalf("extended frame border not replicated")
	}
}

func TestPublicTileRestorationRejectsInvalid(t *testing.T) {
	params := internalparser.RestorationParams{Type: [3]internalparser.RestorationType{internalparser.RestorationWiener}, UnitSizeY: 128, UnitSizeUV: 64}
	if _, err := internaltile.BuildRestorationPlaneGrid(params, FrameSize{}, internalparser.ColorConfig{}, 0); !errors.Is(err, internaltile.ErrInvalidPlan) {
		t.Fatalf("BuildTileRestorationPlaneGrid err=%v want %v", err, internaltile.ErrInvalidPlan)
	}
	plan, err := internaltile.BuildRestorationFramePlan(params, FrameSize{UpscaledWidth: 128, Height: 64, SuperResDenominator: 8}, internalparser.ColorConfig{})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := internaltile.BindRestorationFrameRecordBuffers(plan, make([]internaltile.RestorationUnitRecord, plan.UnitRecordLen()-1)); !errors.Is(err, internaltile.ErrJobBufferTooSmall) {
		t.Fatalf("BindTileRestorationFrameRecordBuffers err=%v want %v", err, internaltile.ErrJobBufferTooSmall)
	}
	if _, err := internaltile.RestorationUnitScratchLen(0, 8); !errors.Is(err, internalrestoration.ErrInvalidRestoration) {
		t.Fatalf("TileRestorationUnitScratchLen err=%v want %v", err, internalrestoration.ErrInvalidRestoration)
	}
	if _, err := internaltile.ApplyRestorationUnit(nil, 0, 0, nil, 0, 1, 1, internaltile.RestorationUnit{Type: internalparser.RestorationType(99)}, 8, internaltile.RestorationUnitScratch{}); !errors.Is(err, internaltile.ErrInvalidPlan) {
		t.Fatalf("ApplyTileRestorationUnit err=%v want %v", err, internaltile.ErrInvalidPlan)
	}
}

func TestPublicTileRestorationAllocs(t *testing.T) {
	params := internalparser.RestorationParams{Type: [3]internalparser.RestorationType{internalparser.RestorationWiener, internalparser.RestorationSGRProj, internalparser.RestorationNone}, UnitSizeY: 128, UnitSizeUV: 64}
	size := FrameSize{UpscaledWidth: 300, Height: 260, SuperResDenominator: 8}
	color := internalparser.ColorConfig{SubsamplingX: true, SubsamplingY: true}
	plan, err := internaltile.BuildRestorationFramePlan(params, size, color)
	if err != nil {
		t.Fatal(err)
	}
	recordsBacking := make([]internaltile.RestorationUnitRecord, plan.UnitRecordLen())
	above := make([]uint16, plan.BoundaryBufferLen())
	below := make([]uint16, plan.BoundaryBufferLen())
	src := make([]uint16, 13+3*11+5)
	dst := make([]uint16, 8*4)
	allocs := testing.AllocsPerRun(1000, func() {
		records, err := internaltile.BindRestorationFrameRecordBuffers(plan, recordsBacking)
		if err != nil {
			t.Fatalf("BindTileRestorationFrameRecordBuffers err=%v", err)
		}
		if err := internaltile.ResetRestorationPlaneRecords(plan.Grids[0], records[0]); err != nil {
			t.Fatalf("ResetTileRestorationPlaneRecords err=%v", err)
		}
		if _, err := internaltile.BindRestorationFrameBoundaryBuffers(plan, above, below); err != nil {
			t.Fatalf("BindTileRestorationFrameBoundaryBuffers err=%v", err)
		}
		if _, err := internaltile.RestorationUnitScratchLen(64, 32); err != nil {
			t.Fatalf("TileRestorationUnitScratchLen err=%v", err)
		}
		if _, err := internaltile.ApplyRestorationUnit(src, 11, 13, dst, 8, 5, 4, internaltile.RestorationUnit{Type: internalparser.RestorationNone}, 8, internaltile.RestorationUnitScratch{}); err != nil {
			t.Fatalf("ApplyTileRestorationUnit err=%v", err)
		}
	})
	if allocs != 0 {
		t.Fatalf("allocs=%v want 0", allocs)
	}
}

func publicTileRestorationBordered(width int, height int, borderHorz int, borderVert int) ([]uint16, int, int) {
	stride := width + 2*borderHorz
	origin := borderVert*stride + borderHorz
	return make([]uint16, stride*(height+2*borderVert)), stride, origin
}
