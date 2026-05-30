package tile

import (
	"errors"
	"testing"

	av1frame "github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/parser"
	av1restoration "github.com/thesyncim/goav1/internal/av1/restoration"
)

const restorationApplyFrameHorzBorder = restorationExtraHorz + 16

func TestRestorationUnitScratchLenMatchesPrimitives(t *testing.T) {
	got, err := RestorationUnitScratchLen(64, 32)
	if err != nil {
		t.Fatal(err)
	}
	wantWiener, err := av1restoration.WienerScratchLen(64, 32)
	if err != nil {
		t.Fatal(err)
	}
	wantSGR, err := av1restoration.SelfguidedScratchLen(64, 32)
	if err != nil {
		t.Fatal(err)
	}
	if got != (RestorationUnitScratchSize{Wiener: wantWiener, SGRProj: wantSGR}) {
		t.Fatalf("scratch=%+v want wiener=%d sgr=%d", got, wantWiener, wantSGR)
	}
}

func TestApplyRestorationUnitNoneCopies(t *testing.T) {
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

	result, err := ApplyRestorationUnit(src, srcStride, origin, dst, dstStride, width, height, RestorationUnit{Type: parser.RestorationNone}, 8, RestorationUnitScratch{})
	if err != nil {
		t.Fatal(err)
	}
	if result != (RestorationUnitApplyResult{Type: parser.RestorationNone}) {
		t.Fatalf("result=%+v", result)
	}
	for row := range height {
		for col := range width {
			if got, want := dst[row*dstStride+col], src[origin+row*srcStride+col]; got != want {
				t.Fatalf("dst[%d,%d]=%d want %d", row, col, got, want)
			}
		}
		for col := width; col < dstStride; col++ {
			if dst[row*dstStride+col] != 0xffff {
				t.Fatalf("padding row=%d col=%d overwritten", row, col)
			}
		}
	}
}

func TestApplyRestorationUnitWienerMatchesPrimitive(t *testing.T) {
	const width, height = 16, 12
	const bitDepth = 10
	stride := width + 2*av1restoration.WienerHalfwin
	origin := av1restoration.WienerHalfwin*stride + av1restoration.WienerHalfwin
	src := makeRestorationApplySource(stride, height+2*av1restoration.WienerHalfwin, bitDepth)
	unit := RestorationUnit{
		Type: parser.RestorationWiener,
		Wiener: av1restoration.WienerInfo{
			VFilter: av1restoration.NewWienerFilter(-3, -11, 22),
			HFilter: av1restoration.NewWienerFilter(6, -9, 18),
		},
	}
	sizes, err := RestorationUnitScratchLen(width, height)
	if err != nil {
		t.Fatal(err)
	}
	got := make([]uint16, width*height)
	want := make([]uint16, width*height)
	result, err := ApplyRestorationUnit(src, stride, origin, got, width, width, height, unit, bitDepth, RestorationUnitScratch{Wiener: make([]uint16, sizes.Wiener)})
	if err != nil {
		t.Fatal(err)
	}
	if result != (RestorationUnitApplyResult{Type: parser.RestorationWiener, Filtered: true}) {
		t.Fatalf("result=%+v", result)
	}
	if err := av1restoration.ApplyWienerRestoration(src, stride, origin, want, width, width, height, unit.Wiener, bitDepth, make([]uint16, sizes.Wiener)); err != nil {
		t.Fatal(err)
	}
	assertSamplesEqual(t, got, want)
}

func TestApplyRestorationUnitSGRProjMatchesPrimitive(t *testing.T) {
	const width, height = 13, 9
	const bitDepth = 8
	stride := width + 2*av1restoration.SGRProjBorderHorz
	origin := av1restoration.SGRProjBorderVert*stride + av1restoration.SGRProjBorderHorz
	src := makeRestorationApplySource(stride, height+2*av1restoration.SGRProjBorderVert, bitDepth)
	unit := RestorationUnit{
		Type:    parser.RestorationSGRProj,
		SGRProj: SGRProjInfo{ParamsIndex: 1, XQD: [2]int{13, -9}},
	}
	sizes, err := RestorationUnitScratchLen(width, height)
	if err != nil {
		t.Fatal(err)
	}
	got := make([]uint16, width*height)
	want := make([]uint16, width*height)
	result, err := ApplyRestorationUnit(src, stride, origin, got, width, width, height, unit, bitDepth, RestorationUnitScratch{SGRProj: make([]int32, sizes.SGRProj)})
	if err != nil {
		t.Fatal(err)
	}
	if result != (RestorationUnitApplyResult{Type: parser.RestorationSGRProj, Filtered: true}) {
		t.Fatalf("result=%+v", result)
	}
	if err := av1restoration.ApplySelfguidedRestoration(src, stride, origin, want, width, width, height, int(unit.SGRProj.ParamsIndex), unit.SGRProj.XQD, bitDepth, make([]int32, sizes.SGRProj)); err != nil {
		t.Fatal(err)
	}
	assertSamplesEqual(t, got, want)
}

func TestRestorationUnitRecordScratchLen(t *testing.T) {
	grid := testRestorationApplyGrid(t, [3]parser.RestorationType{parser.RestorationWiener}, 0)
	none := testRestorationApplyRecord(t, grid, 1, 1, RestorationUnit{Type: parser.RestorationNone})
	sizes, err := RestorationUnitRecordScratchLen(grid, none)
	if err != nil {
		t.Fatal(err)
	}
	if sizes != (RestorationUnitScratchSize{}) {
		t.Fatalf("none scratch=%+v", sizes)
	}

	wiener := none
	wiener.Unit = RestorationUnit{Type: parser.RestorationWiener, Wiener: av1restoration.DefaultWienerInfo()}
	sizes, err = RestorationUnitRecordScratchLen(grid, wiener)
	if err != nil {
		t.Fatal(err)
	}
	wantWiener, err := av1restoration.WienerScratchLen(64, 64)
	if err != nil {
		t.Fatal(err)
	}
	if sizes.Wiener != wantWiener || sizes.SGRProj != 0 {
		t.Fatalf("wiener scratch=%+v want wiener=%d sgr=0", sizes, wantWiener)
	}

	uv := testRestorationApplyGrid(t, [3]parser.RestorationType{parser.RestorationNone, parser.RestorationSGRProj}, 1)
	sgr := testRestorationApplyRecord(t, uv, 1, 1, RestorationUnit{Type: parser.RestorationSGRProj, SGRProj: SGRProjInfo{ParamsIndex: 1, XQD: [2]int{13, -9}}})
	sizes, err = RestorationUnitRecordScratchLen(uv, sgr)
	if err != nil {
		t.Fatal(err)
	}
	wantSGR, err := av1restoration.SelfguidedScratchLen(32, 32)
	if err != nil {
		t.Fatal(err)
	}
	if sizes.Wiener != 0 || sizes.SGRProj != wantSGR {
		t.Fatalf("sgr scratch=%+v want wiener=0 sgr=%d", sizes, wantSGR)
	}
}

func TestRestorationUnitRecordBoundaryScratchLen(t *testing.T) {
	grid := testRestorationApplyGrid(t, [3]parser.RestorationType{parser.RestorationWiener}, 0)
	none := testRestorationApplyRecord(t, grid, 1, 1, RestorationUnit{Type: parser.RestorationNone})
	sizes, err := RestorationUnitRecordBoundaryScratchLen(grid, none, false)
	if err != nil {
		t.Fatal(err)
	}
	if sizes != (RestorationUnitRecordBoundaryScratchSize{}) {
		t.Fatalf("none boundary scratch=%+v", sizes)
	}

	wiener := none
	wiener.Unit = RestorationUnit{Type: parser.RestorationWiener, Wiener: av1restoration.DefaultWienerInfo()}
	sizes, err = RestorationUnitRecordBoundaryScratchLen(grid, wiener, false)
	if err != nil {
		t.Fatal(err)
	}
	lineWidth := int(wiener.Rect.Width()) + 2*restorationExtraHorz
	if sizes.Boundary.Above != lineWidth*restorationBorder || sizes.Boundary.Below != lineWidth*restorationBorder {
		t.Fatalf("boundary scratch=%+v lineWidth=%d", sizes.Boundary, lineWidth)
	}
	opt, err := RestorationUnitRecordBoundaryScratchLen(grid, wiener, true)
	if err != nil {
		t.Fatal(err)
	}
	if opt.Unit != sizes.Unit || opt.Boundary.Above != lineWidth || opt.Boundary.Below != lineWidth {
		t.Fatalf("optimized scratch=%+v want unit=%+v above/below=%d", opt, sizes.Unit, lineWidth)
	}
}

func TestRestorationPlaneApplyScratchLen(t *testing.T) {
	grid := testRestorationApplyGrid(t, [3]parser.RestorationType{parser.RestorationWiener}, 0)
	records := makeRestorationPlaneRecords(t, grid, func(i int) RestorationUnit {
		if i%2 == 0 {
			return RestorationUnit{Type: parser.RestorationNone}
		}
		return RestorationUnit{Type: parser.RestorationWiener, Wiener: av1restoration.DefaultWienerInfo()}
	})
	size, err := RestorationPlaneApplyScratchLen(grid, records, false)
	if err != nil {
		t.Fatal(err)
	}
	if size.Unit.Wiener == 0 || size.Unit.SGRProj != 0 || size.Boundary.Above == 0 || size.Boundary.Below == 0 {
		t.Fatalf("plane scratch=%+v", size)
	}
	opt, err := RestorationPlaneApplyScratchLen(grid, records, true)
	if err != nil {
		t.Fatal(err)
	}
	if opt.Unit != size.Unit || opt.Boundary.Above >= size.Boundary.Above || opt.Boundary.Below >= size.Boundary.Below {
		t.Fatalf("optimized scratch=%+v normal=%+v", opt, size)
	}
}

func TestRestorationFramePlaneScratchLen(t *testing.T) {
	grid := testRestorationApplyGrid(t, [3]parser.RestorationType{parser.RestorationWiener}, 0)
	records := makeRestorationPlaneRecords(t, grid, func(i int) RestorationUnit {
		if i%2 == 0 {
			return RestorationUnit{Type: parser.RestorationNone}
		}
		return RestorationUnit{Type: parser.RestorationWiener, Wiener: av1restoration.DefaultWienerInfo()}
	})
	got, err := RestorationFramePlaneScratchLen(grid, records, false)
	if err != nil {
		t.Fatal(err)
	}
	want, err := RestorationPlaneApplyScratchLen(grid, records, false)
	if err != nil {
		t.Fatal(err)
	}
	if got != want {
		t.Fatalf("frame scratch=%+v want %+v", got, want)
	}

	none, err := RestorationFramePlaneScratchLen(RestorationPlaneGrid{Plane: 2, Type: parser.RestorationNone}, nil, false)
	if err != nil {
		t.Fatal(err)
	}
	if none != (RestorationUnitRecordBoundaryScratchSize{}) {
		t.Fatalf("disabled scratch=%+v", none)
	}
	if _, err := RestorationFramePlaneScratchLen(RestorationPlaneGrid{Plane: 3, Type: parser.RestorationNone}, nil, false); !errors.Is(err, ErrInvalidPlan) {
		t.Fatalf("bad disabled plane err=%v want %v", err, ErrInvalidPlan)
	}
}

func TestRestorationFrameScratchLen(t *testing.T) {
	planes := makeRestorationFramePlanes(t, [3]parser.RestorationType{parser.RestorationWiener, parser.RestorationSGRProj, parser.RestorationNone}, 10, false)
	got, err := RestorationFrameScratchLen(planes, false)
	if err != nil {
		t.Fatal(err)
	}
	ySize, err := RestorationFramePlaneScratchLen(planes[0].Grid, planes[0].Records, false)
	if err != nil {
		t.Fatal(err)
	}
	uSize, err := RestorationFramePlaneScratchLen(planes[1].Grid, planes[1].Records, false)
	if err != nil {
		t.Fatal(err)
	}
	want := maxRestorationFrameScratch(ySize, uSize)
	if got != want {
		t.Fatalf("frame scratch=%+v want %+v", got, want)
	}

	mono, err := RestorationFrameScratchLen(planes[:1], false)
	if err != nil {
		t.Fatal(err)
	}
	if mono != ySize {
		t.Fatalf("mono scratch=%+v want %+v", mono, ySize)
	}
	if _, err := RestorationFrameScratchLen(planes[:2], false); !errors.Is(err, ErrInvalidPlan) {
		t.Fatalf("two-plane scratch err=%v want %v", err, ErrInvalidPlan)
	}
	bad := append([]RestorationFramePlane(nil), planes...)
	bad[1].Grid.Plane = 2
	if _, err := RestorationFrameScratchLen(bad, false); !errors.Is(err, ErrInvalidPlan) {
		t.Fatalf("bad plane order err=%v want %v", err, ErrInvalidPlan)
	}
}

func TestRestorationFrameSampleScratchLenMatchesBorderedLayouts(t *testing.T) {
	const bitDepth = 10
	types := [3]parser.RestorationType{parser.RestorationWiener, parser.RestorationSGRProj, parser.RestorationNone}
	plan := testRestorationFramePlan(t, types, false)
	frm := makeRestorationApplyFrame(t, bitDepth, false)

	got, err := RestorationFrameSampleScratchLen(plan, frm)
	if err != nil {
		t.Fatal(err)
	}
	yLayout, err := av1frame.BorderedSamplePlaneLen(frm.Y, 2, RestorationFrameBorder, RestorationFrameBorder, 32)
	if err != nil {
		t.Fatal(err)
	}
	uLayout, err := av1frame.BorderedSamplePlaneLen(frm.U, 2, RestorationFrameBorder, RestorationFrameBorder, 32)
	if err != nil {
		t.Fatal(err)
	}
	if got.Data[0] != yLayout || got.Dst[0] != yLayout {
		t.Fatalf("Y layouts=%+v/%+v want %+v", got.Data[0], got.Dst[0], yLayout)
	}
	if got.Data[1] != uLayout || got.Dst[1] != uLayout {
		t.Fatalf("U layouts=%+v/%+v want %+v", got.Data[1], got.Dst[1], uLayout)
	}
	if got.Data[2] != (av1frame.BorderedSamplePlaneLayout{}) || got.Dst[2] != (av1frame.BorderedSamplePlaneLayout{}) {
		t.Fatalf("disabled V layouts=%+v/%+v", got.Data[2], got.Dst[2])
	}
	if got.DataLen != yLayout.Len+uLayout.Len || got.DstLen != yLayout.Len+uLayout.Len {
		t.Fatalf("scratch=%+v want data/dst len %d", got, yLayout.Len+uLayout.Len)
	}

	allNone := testRestorationFramePlan(t, [3]parser.RestorationType{}, false)
	zero, err := RestorationFrameSampleScratchLen(allNone, frm)
	if err != nil {
		t.Fatal(err)
	}
	if zero != (RestorationFrameSampleScratchSize{}) {
		t.Fatalf("all-none scratch=%+v", zero)
	}
}

func TestApplyRestorationFrameToFrameMatchesSampleFlow(t *testing.T) {
	const bitDepth = 10
	types := [3]parser.RestorationType{parser.RestorationWiener, parser.RestorationSGRProj, parser.RestorationNone}
	plan := testRestorationFramePlan(t, types, false)
	planes := makeRestorationFramePlanes(t, types, bitDepth, false)
	manualPlanes := cloneRestorationFramePlanes(planes)
	frm := makeRestorationApplyFrame(t, bitDepth, false)
	fillFrameFromRestorationPlanes(t, frm, planes, bitDepth)

	var records [3][]RestorationUnitRecord
	var boundaries [3]RestorationStripeBoundaries
	for plane := range planes {
		records[plane] = planes[plane].Records
		boundaries[plane] = planes[plane].Boundaries
	}
	sampleSize, err := RestorationFrameSampleScratchLen(plan, frm)
	if err != nil {
		t.Fatal(err)
	}
	applySize, err := RestorationFrameScratchLen(planes, false)
	if err != nil {
		t.Fatal(err)
	}
	want, err := ApplyRestorationFrame(manualPlanes, bitDepth, makeRestorationBoundaryApplyScratch(applySize), false)
	if err != nil {
		t.Fatal(err)
	}
	got, err := ApplyRestorationFrameToFrame(plan, frm, records, boundaries, make([]uint16, sampleSize.DataLen), make([]uint16, sampleSize.DstLen), makeRestorationBoundaryApplyScratch(applySize), false)
	if err != nil {
		t.Fatal(err)
	}
	if got != want {
		t.Fatalf("result=%+v want %+v", got, want)
	}
	assertFrameMatchesRestorationPlanes(t, frm, manualPlanes, bitDepth)
}

func TestApplyRestorationFrameToFrameRejectsInvalidInputs(t *testing.T) {
	const bitDepth = 8
	types := [3]parser.RestorationType{parser.RestorationWiener, parser.RestorationNone, parser.RestorationNone}
	plan := testRestorationFramePlan(t, types, false)
	frm := makeRestorationApplyFrame(t, bitDepth, false)
	records := makeRestorationFrameNoneRecords(t, plan)
	size, err := RestorationFrameSampleScratchLen(plan, frm)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := ApplyRestorationFrameToFrame(plan, frm, records, [3]RestorationStripeBoundaries{}, make([]uint16, size.DataLen-1), make([]uint16, size.DstLen), RestorationUnitRecordBoundaryScratch{}, false); !errors.Is(err, ErrJobBufferTooSmall) {
		t.Fatalf("short data err=%v want %v", err, ErrJobBufferTooSmall)
	}
	if _, err := ApplyRestorationFrameToFrame(plan, frm, records, [3]RestorationStripeBoundaries{}, make([]uint16, size.DataLen+1), make([]uint16, size.DstLen), RestorationUnitRecordBoundaryScratch{}, false); !errors.Is(err, ErrInvalidPlan) {
		t.Fatalf("oversized data err=%v want %v", err, ErrInvalidPlan)
	}
	badFrame := frm
	badFrame.Y.Width--
	if _, err := RestorationFrameSampleScratchLen(plan, badFrame); !errors.Is(err, ErrInvalidPlan) {
		t.Fatalf("bad frame err=%v want %v", err, ErrInvalidPlan)
	}
}

func TestApplyRestorationUnitRecordNoneCopiesRect(t *testing.T) {
	grid := testRestorationApplyGrid(t, [3]parser.RestorationType{parser.RestorationWiener}, 0)
	record := testRestorationApplyRecord(t, grid, 1, 1, RestorationUnit{Type: parser.RestorationNone})
	const bitDepth = 8
	stride := int(grid.PlaneWidth) + 8
	origin := stride + 3
	rows := int(grid.PlaneHeight) + 4
	src := make([]uint16, stride*rows)
	dst := make([]uint16, stride*rows)
	fillRestorationRecordSource(src, stride, bitDepth)
	fillUint16(dst, 0xeeee)

	result, err := ApplyRestorationUnitRecord(grid, record, src, stride, origin, dst, stride, origin, bitDepth, RestorationUnitScratch{})
	if err != nil {
		t.Fatal(err)
	}
	if result != (RestorationUnitRecordApplyResult{Type: parser.RestorationNone}) {
		t.Fatalf("result=%+v", result)
	}
	for y := uint32(0); y < grid.PlaneHeight; y++ {
		for x := uint32(0); x < grid.PlaneWidth; x++ {
			dstOff, ok := restorationPlaneOffset(origin, stride, x, y)
			if !ok {
				t.Fatal("bad dst offset")
			}
			srcOff, ok := restorationPlaneOffset(origin, stride, x, y)
			if !ok {
				t.Fatal("bad src offset")
			}
			inside := x >= record.Rect.X0 && x < record.Rect.X1 && y >= record.Rect.Y0 && y < record.Rect.Y1
			switch {
			case inside && dst[dstOff] != src[srcOff]:
				t.Fatalf("copied sample x=%d y=%d got=%d want %d", x, y, dst[dstOff], src[srcOff])
			case !inside && dst[dstOff] != 0xeeee:
				t.Fatalf("outside sample x=%d y=%d overwritten with %d", x, y, dst[dstOff])
			}
		}
	}
}

func TestApplyRestorationUnitRecordWienerMatchesProcessingUnitPrimitives(t *testing.T) {
	grid := testRestorationApplyGrid(t, [3]parser.RestorationType{parser.RestorationWiener}, 0)
	record := testRestorationApplyRecord(t, grid, 1, 1, RestorationUnit{Type: parser.RestorationWiener, Wiener: av1restoration.DefaultWienerInfo()})
	const bitDepth = 10
	const stride = 320
	const border = av1restoration.WienerHalfwin
	origin := border*stride + border
	rows := int(grid.PlaneHeight) + 2*border
	src := make([]uint16, stride*rows)
	got := make([]uint16, stride*rows)
	want := make([]uint16, stride*rows)
	fillRestorationRecordSource(src, stride, bitDepth)
	fillUint16(got, 0xeeee)
	fillUint16(want, 0xeeee)
	sizes, err := RestorationUnitRecordScratchLen(grid, record)
	if err != nil {
		t.Fatal(err)
	}
	result, err := ApplyRestorationUnitRecord(grid, record, src, stride, origin, got, stride, origin, bitDepth, RestorationUnitScratch{Wiener: make([]uint16, sizes.Wiener)})
	if err != nil {
		t.Fatal(err)
	}
	if result != (RestorationUnitRecordApplyResult{Type: parser.RestorationWiener, Stripes: 3, ProcessingUnits: 9, Filtered: true}) {
		t.Fatalf("result=%+v", result)
	}
	applyRestorationRecordByProcessingUnits(t, grid, record, src, stride, origin, want, stride, origin, bitDepth, RestorationUnitScratch{Wiener: make([]uint16, sizes.Wiener)})
	assertSamplesEqual(t, got, want)
}

func TestApplyRestorationUnitRecordSGRProjChromaMatchesProcessingUnitPrimitives(t *testing.T) {
	grid := testRestorationApplyGrid(t, [3]parser.RestorationType{parser.RestorationNone, parser.RestorationSGRProj}, 1)
	record := testRestorationApplyRecord(t, grid, 1, 1, RestorationUnit{Type: parser.RestorationSGRProj, SGRProj: SGRProjInfo{ParamsIndex: 1, XQD: [2]int{13, -9}}})
	const bitDepth = 8
	const stride = 176
	const border = av1restoration.SGRProjBorderHorz
	origin := border*stride + border
	rows := int(grid.PlaneHeight) + 2*border
	src := make([]uint16, stride*rows)
	got := make([]uint16, stride*rows)
	want := make([]uint16, stride*rows)
	fillRestorationRecordSource(src, stride, bitDepth)
	fillUint16(got, 0xeeee)
	fillUint16(want, 0xeeee)
	sizes, err := RestorationUnitRecordScratchLen(grid, record)
	if err != nil {
		t.Fatal(err)
	}
	result, err := ApplyRestorationUnitRecord(grid, record, src, stride, origin, got, stride, origin, bitDepth, RestorationUnitScratch{SGRProj: make([]int32, sizes.SGRProj)})
	if err != nil {
		t.Fatal(err)
	}
	if result != (RestorationUnitRecordApplyResult{Type: parser.RestorationSGRProj, Stripes: 3, ProcessingUnits: 9, Filtered: true}) {
		t.Fatalf("result=%+v", result)
	}
	applyRestorationRecordByProcessingUnits(t, grid, record, src, stride, origin, want, stride, origin, bitDepth, RestorationUnitScratch{SGRProj: make([]int32, sizes.SGRProj)})
	assertSamplesEqual(t, got, want)
}

func TestApplyRestorationUnitRecordWithBoundariesMatchesManualStripeFlow(t *testing.T) {
	grid := testRestorationApplyGrid(t, [3]parser.RestorationType{parser.RestorationWiener}, 0)
	record := testRestorationApplyRecord(t, grid, 1, 1, RestorationUnit{Type: parser.RestorationWiener, Wiener: av1restoration.DefaultWienerInfo()})
	const bitDepth = 10
	dataStride := int(grid.PlaneWidth) + 2*restorationApplyFrameHorzBorder + 8
	dataOrigin := restorationBorder*dataStride + restorationApplyFrameHorzBorder
	dataRows := int(grid.PlaneHeight) + 2*restorationBorder
	data := makeRestorationApplySource(dataStride, dataRows, bitDepth)
	manualData := append([]uint16(nil), data...)
	originalData := append([]uint16(nil), data...)
	got := make([]uint16, dataStride*dataRows)
	want := make([]uint16, dataStride*dataRows)
	fillUint16(got, 0xeeee)
	fillUint16(want, 0xeeee)
	boundaries := makeRestorationApplyBoundaries(t, grid, bitDepth, 5)

	sizes, err := RestorationUnitRecordBoundaryScratchLen(grid, record, false)
	if err != nil {
		t.Fatal(err)
	}
	result, err := ApplyRestorationUnitRecordWithBoundaries(grid, record, boundaries, data, dataStride, dataOrigin, got, dataStride, dataOrigin, bitDepth, makeRestorationBoundaryApplyScratch(sizes), false)
	if err != nil {
		t.Fatal(err)
	}
	if result != (RestorationUnitRecordApplyResult{Type: parser.RestorationWiener, Stripes: 3, ProcessingUnits: 9, Filtered: true}) {
		t.Fatalf("result=%+v", result)
	}
	applyRestorationRecordWithBoundariesManually(t, grid, record, boundaries, manualData, dataStride, dataOrigin, want, dataStride, dataOrigin, bitDepth, makeRestorationBoundaryApplyScratch(sizes), false)
	assertSamplesEqual(t, got, want)
	assertSamplesEqual(t, data, originalData)
	assertSamplesEqual(t, manualData, originalData)
}

func TestApplyRestorationUnitRecordWithBoundariesUsesSavedRows(t *testing.T) {
	grid := testRestorationApplyGrid(t, [3]parser.RestorationType{parser.RestorationWiener}, 0)
	record := testRestorationApplyRecord(t, grid, 1, 1, RestorationUnit{Type: parser.RestorationWiener, Wiener: av1restoration.DefaultWienerInfo()})
	const bitDepth = 10
	dataStride := int(grid.PlaneWidth) + 2*restorationApplyFrameHorzBorder + 8
	dataOrigin := restorationBorder*dataStride + restorationApplyFrameHorzBorder
	dataRows := int(grid.PlaneHeight) + 2*restorationBorder
	dataA := makeRestorationApplySource(dataStride, dataRows, bitDepth)
	dataB := append([]uint16(nil), dataA...)
	dstA := make([]uint16, dataStride*dataRows)
	dstB := make([]uint16, dataStride*dataRows)
	boundariesA := makeRestorationApplyBoundaries(t, grid, bitDepth, 3)
	boundariesB := makeRestorationApplyBoundaries(t, grid, bitDepth, 97)

	sizes, err := RestorationUnitRecordBoundaryScratchLen(grid, record, false)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := ApplyRestorationUnitRecordWithBoundaries(grid, record, boundariesA, dataA, dataStride, dataOrigin, dstA, dataStride, dataOrigin, bitDepth, makeRestorationBoundaryApplyScratch(sizes), false); err != nil {
		t.Fatal(err)
	}
	if _, err := ApplyRestorationUnitRecordWithBoundaries(grid, record, boundariesB, dataB, dataStride, dataOrigin, dstB, dataStride, dataOrigin, bitDepth, makeRestorationBoundaryApplyScratch(sizes), false); err != nil {
		t.Fatal(err)
	}
	if samplesEqual(dstA, dstB) {
		t.Fatal("boundary variants produced identical output")
	}
}

func TestApplyRestorationUnitRecordWithBoundariesOptimizedIgnoresSavedRows(t *testing.T) {
	grid := testRestorationApplyGrid(t, [3]parser.RestorationType{parser.RestorationWiener}, 0)
	record := testRestorationApplyRecord(t, grid, 1, 1, RestorationUnit{Type: parser.RestorationWiener, Wiener: av1restoration.DefaultWienerInfo()})
	const bitDepth = 8
	dataStride := int(grid.PlaneWidth) + 2*restorationApplyFrameHorzBorder + 8
	dataOrigin := restorationBorder*dataStride + restorationApplyFrameHorzBorder
	dataRows := int(grid.PlaneHeight) + 2*restorationBorder
	dataA := makeRestorationApplySource(dataStride, dataRows, bitDepth)
	dataB := append([]uint16(nil), dataA...)
	dstA := make([]uint16, dataStride*dataRows)
	dstB := make([]uint16, dataStride*dataRows)
	size, err := RestorationStripeBoundaryBufferLen(grid)
	if err != nil {
		t.Fatal(err)
	}
	boundaries := makeSentinelRestorationBoundaries(size, 0xffff)
	sizes, err := RestorationUnitRecordBoundaryScratchLen(grid, record, true)
	if err != nil {
		t.Fatal(err)
	}

	if _, err := ApplyRestorationUnitRecordWithBoundaries(grid, record, RestorationStripeBoundaries{}, dataA, dataStride, dataOrigin, dstA, dataStride, dataOrigin, bitDepth, makeRestorationBoundaryApplyScratch(sizes), true); err != nil {
		t.Fatal(err)
	}
	if _, err := ApplyRestorationUnitRecordWithBoundaries(grid, record, boundaries, dataB, dataStride, dataOrigin, dstB, dataStride, dataOrigin, bitDepth, makeRestorationBoundaryApplyScratch(sizes), true); err != nil {
		t.Fatal(err)
	}
	assertSamplesEqual(t, dstA, dstB)
}

func TestApplyRestorationPlaneRecordsMatchesManualRecords(t *testing.T) {
	grid := testRestorationApplyGrid(t, [3]parser.RestorationType{parser.RestorationWiener}, 0)
	records := makeRestorationPlaneRecords(t, grid, func(i int) RestorationUnit {
		if i%2 == 0 {
			return RestorationUnit{Type: parser.RestorationNone}
		}
		return RestorationUnit{Type: parser.RestorationWiener, Wiener: av1restoration.DefaultWienerInfo()}
	})
	const bitDepth = 10
	dataStride := int(grid.PlaneWidth) + 2*restorationApplyFrameHorzBorder + 8
	dataOrigin := restorationBorder*dataStride + restorationApplyFrameHorzBorder
	dataRows := int(grid.PlaneHeight) + 2*restorationBorder
	data := makeRestorationApplySource(dataStride, dataRows, bitDepth)
	manualData := append([]uint16(nil), data...)
	originalData := append([]uint16(nil), data...)
	got := make([]uint16, dataStride*dataRows)
	want := make([]uint16, dataStride*dataRows)
	fillUint16(got, 0xeeee)
	fillUint16(want, 0xeeee)
	boundaries := makeRestorationApplyBoundaries(t, grid, bitDepth, 11)
	sizes, err := RestorationPlaneApplyScratchLen(grid, records, false)
	if err != nil {
		t.Fatal(err)
	}
	scratch := makeRestorationBoundaryApplyScratch(sizes)

	result, err := ApplyRestorationPlaneRecords(grid, records, boundaries, data, dataStride, dataOrigin, got, dataStride, dataOrigin, bitDepth, scratch, false)
	if err != nil {
		t.Fatal(err)
	}
	wantResult := applyRestorationPlaneRecordsManually(t, grid, records, boundaries, manualData, dataStride, dataOrigin, want, dataStride, dataOrigin, bitDepth, makeRestorationBoundaryApplyScratch(sizes), false)
	if result != wantResult {
		t.Fatalf("result=%+v want %+v", result, wantResult)
	}
	if result.Records != uint32(len(records)) || result.FilteredRecords != 2 {
		t.Fatalf("unexpected result=%+v len=%d", result, len(records))
	}
	assertSamplesEqual(t, got, want)
	assertSamplesEqual(t, data, originalData)
	assertSamplesEqual(t, manualData, originalData)
}

func TestApplyRestorationFrameMatchesPlaneLoop(t *testing.T) {
	const bitDepth = 10
	planes := makeRestorationFramePlanes(t, [3]parser.RestorationType{parser.RestorationWiener, parser.RestorationSGRProj, parser.RestorationNone}, bitDepth, false)
	manualPlanes := cloneRestorationFramePlanes(planes)
	sizes, err := RestorationFrameScratchLen(planes, false)
	if err != nil {
		t.Fatal(err)
	}

	result, err := ApplyRestorationFrame(planes, bitDepth, makeRestorationBoundaryApplyScratch(sizes), false)
	if err != nil {
		t.Fatal(err)
	}
	want := applyRestorationFrameManually(t, manualPlanes, bitDepth, makeRestorationBoundaryApplyScratch(sizes), false)
	if result != want {
		t.Fatalf("result=%+v want %+v", result, want)
	}
	if result.Planes != 2 || result.PlaneResults[2] != (RestorationPlaneApplyResult{}) {
		t.Fatalf("unexpected result=%+v", result)
	}
	assertRestorationFramePlanesEqual(t, planes, manualPlanes)
}

func TestApplyRestorationFrameMonochrome(t *testing.T) {
	const bitDepth = 8
	planes := makeRestorationFramePlanes(t, [3]parser.RestorationType{parser.RestorationWiener, parser.RestorationNone, parser.RestorationNone}, bitDepth, true)
	manualPlanes := cloneRestorationFramePlanes(planes)
	sizes, err := RestorationFrameScratchLen(planes, false)
	if err != nil {
		t.Fatal(err)
	}

	result, err := ApplyRestorationFrame(planes, bitDepth, makeRestorationBoundaryApplyScratch(sizes), false)
	if err != nil {
		t.Fatal(err)
	}
	want := applyRestorationFrameManually(t, manualPlanes, bitDepth, makeRestorationBoundaryApplyScratch(sizes), false)
	if result != want {
		t.Fatalf("result=%+v want %+v", result, want)
	}
	if result.Planes != 1 || result.Records != uint32(len(planes[0].Records)) {
		t.Fatalf("mono result=%+v records=%d", result, len(planes[0].Records))
	}
	assertRestorationFramePlanesEqual(t, planes, manualPlanes)
}

func TestApplyRestorationFramePlaneMatchesLibaomFlow(t *testing.T) {
	grid := testRestorationApplyGrid(t, [3]parser.RestorationType{parser.RestorationWiener}, 0)
	records := makeRestorationPlaneRecords(t, grid, func(i int) RestorationUnit {
		if i == 0 {
			return RestorationUnit{Type: parser.RestorationNone}
		}
		return RestorationUnit{Type: parser.RestorationWiener, Wiener: av1restoration.DefaultWienerInfo()}
	})
	const bitDepth = 10
	dataStride := int(grid.PlaneWidth) + 2*restorationApplyFrameHorzBorder + 8
	dataOrigin := restorationBorder*dataStride + restorationApplyFrameHorzBorder
	dataRows := int(grid.PlaneHeight) + 2*restorationBorder
	data := makeRestorationApplySource(dataStride, dataRows, bitDepth)
	manualData := append([]uint16(nil), data...)
	gotDst := make([]uint16, dataStride*dataRows)
	wantDst := make([]uint16, dataStride*dataRows)
	fillUint16(gotDst, 0xeeee)
	fillUint16(wantDst, 0xeeee)
	boundaries := makeRestorationApplyBoundaries(t, grid, bitDepth, 13)
	sizes, err := RestorationFramePlaneScratchLen(grid, records, false)
	if err != nil {
		t.Fatal(err)
	}

	result, err := ApplyRestorationFramePlane(grid, records, boundaries, data, dataStride, dataOrigin, gotDst, dataStride, dataOrigin, bitDepth, makeRestorationBoundaryApplyScratch(sizes), false)
	if err != nil {
		t.Fatal(err)
	}
	wantResult := applyRestorationFramePlaneManually(t, grid, records, boundaries, manualData, dataStride, dataOrigin, wantDst, dataStride, dataOrigin, bitDepth, makeRestorationBoundaryApplyScratch(sizes), false)
	if result != wantResult {
		t.Fatalf("result=%+v want %+v", result, wantResult)
	}
	if result.Records != uint32(len(records)) || result.FilteredRecords != uint32(len(records)-1) {
		t.Fatalf("unexpected result=%+v len=%d", result, len(records))
	}
	assertSamplesEqual(t, gotDst, wantDst)
	assertSamplesEqual(t, data, manualData)
	for y := uint32(0); y < grid.PlaneHeight; y++ {
		for x := uint32(0); x < grid.PlaneWidth; x++ {
			offset, ok := restorationPlaneOffset(dataOrigin, dataStride, x, y)
			if !ok {
				t.Fatal("bad plane offset")
			}
			if data[offset] != gotDst[offset] {
				t.Fatalf("visible x=%d y=%d data=%d dst=%d", x, y, data[offset], gotDst[offset])
			}
		}
	}
}

func TestApplyRestorationFramePlaneDisabledIsNoop(t *testing.T) {
	data := []uint16{1, 2, 3, 4}
	dst := []uint16{5, 6, 7, 8}
	originalData := append([]uint16(nil), data...)
	originalDst := append([]uint16(nil), dst...)
	result, err := ApplyRestorationFramePlane(RestorationPlaneGrid{Plane: 2, Type: parser.RestorationNone}, []RestorationUnitRecord{{Index: 99}}, RestorationStripeBoundaries{}, data, 0, -1, dst, 0, -1, 7, RestorationUnitRecordBoundaryScratch{}, false)
	if err != nil {
		t.Fatal(err)
	}
	if result != (RestorationPlaneApplyResult{}) {
		t.Fatalf("disabled result=%+v", result)
	}
	assertSamplesEqual(t, data, originalData)
	assertSamplesEqual(t, dst, originalDst)
}

func TestApplyRestorationUnitRejectsInvalidInputs(t *testing.T) {
	const width, height = 8, 8
	stride := width + 2*av1restoration.WienerHalfwin
	origin := av1restoration.WienerHalfwin*stride + av1restoration.WienerHalfwin
	src := makeRestorationApplySource(stride, height+2*av1restoration.WienerHalfwin, 8)
	dst := make([]uint16, width*height)
	sizes, err := RestorationUnitScratchLen(width, height)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := ApplyRestorationUnit(src, stride, origin, dst, width, width, height, RestorationUnit{Type: parser.RestorationSwitchable}, 8, RestorationUnitScratch{}); !errors.Is(err, ErrInvalidPlan) {
		t.Fatalf("invalid type err=%v want %v", err, ErrInvalidPlan)
	}
	bad := append([]uint16(nil), src...)
	bad[origin] = 256
	if _, err := ApplyRestorationUnit(bad, stride, origin, dst, width, width, height, RestorationUnit{Type: parser.RestorationNone}, 8, RestorationUnitScratch{}); !errors.Is(err, ErrInvalidPlan) {
		t.Fatalf("bad none sample err=%v want %v", err, ErrInvalidPlan)
	}
	unit := RestorationUnit{Type: parser.RestorationWiener, Wiener: av1restoration.DefaultWienerInfo()}
	if _, err := ApplyRestorationUnit(src, stride, origin, dst, width, width, height, unit, 8, RestorationUnitScratch{Wiener: make([]uint16, sizes.Wiener-1)}); !errors.Is(err, av1restoration.ErrInvalidRestoration) {
		t.Fatalf("short scratch err=%v want %v", err, av1restoration.ErrInvalidRestoration)
	}
}

func TestApplyRestorationUnitRecordRejectsInvalidInputs(t *testing.T) {
	grid := testRestorationApplyGrid(t, [3]parser.RestorationType{parser.RestorationWiener}, 0)
	record := testRestorationApplyRecord(t, grid, 1, 1, RestorationUnit{Type: parser.RestorationWiener, Wiener: av1restoration.DefaultWienerInfo()})
	const stride = 320
	const border = av1restoration.WienerHalfwin
	origin := border*stride + border
	rows := int(grid.PlaneHeight) + 2*border
	src := makeRestorationApplySource(stride, rows, 8)
	dst := make([]uint16, stride*rows)
	sizes, err := RestorationUnitRecordScratchLen(grid, record)
	if err != nil {
		t.Fatal(err)
	}

	bad := record
	bad.StripeCount--
	if _, err := ApplyRestorationUnitRecord(grid, bad, src, stride, origin, dst, stride, origin, 8, RestorationUnitScratch{Wiener: make([]uint16, sizes.Wiener)}); !errors.Is(err, ErrInvalidPlan) {
		t.Fatalf("bad stripe count err=%v want %v", err, ErrInvalidPlan)
	}
	bad = record
	bad.Unit.Type = parser.RestorationSGRProj
	if _, err := RestorationUnitRecordScratchLen(grid, bad); !errors.Is(err, ErrInvalidPlan) {
		t.Fatalf("bad unit type err=%v want %v", err, ErrInvalidPlan)
	}
	if _, err := ApplyRestorationUnitRecord(grid, record, src, stride, -1, dst, stride, origin, 8, RestorationUnitScratch{Wiener: make([]uint16, sizes.Wiener)}); !errors.Is(err, ErrInvalidPlan) {
		t.Fatalf("bad origin err=%v want %v", err, ErrInvalidPlan)
	}
}

func TestApplyRestorationUnitRecordWithBoundariesRejectsInvalidInputs(t *testing.T) {
	grid := testRestorationApplyGrid(t, [3]parser.RestorationType{parser.RestorationWiener}, 0)
	record := testRestorationApplyRecord(t, grid, 1, 1, RestorationUnit{Type: parser.RestorationWiener, Wiener: av1restoration.DefaultWienerInfo()})
	const bitDepth = 8
	dataStride := int(grid.PlaneWidth) + 2*restorationApplyFrameHorzBorder + 8
	dataOrigin := restorationBorder*dataStride + restorationApplyFrameHorzBorder
	dataRows := int(grid.PlaneHeight) + 2*restorationBorder
	data := makeRestorationApplySource(dataStride, dataRows, bitDepth)
	dst := make([]uint16, dataStride*dataRows)
	boundaries := makeRestorationApplyBoundaries(t, grid, bitDepth, 5)
	sizes, err := RestorationUnitRecordBoundaryScratchLen(grid, record, false)
	if err != nil {
		t.Fatal(err)
	}
	scratch := makeRestorationBoundaryApplyScratch(sizes)

	short := scratch
	short.Boundary.Above = short.Boundary.Above[:len(short.Boundary.Above)-1]
	if _, err := ApplyRestorationUnitRecordWithBoundaries(grid, record, boundaries, data, dataStride, dataOrigin, dst, dataStride, dataOrigin, bitDepth, short, false); !errors.Is(err, ErrInvalidPlan) {
		t.Fatalf("short boundary scratch err=%v want %v", err, ErrInvalidPlan)
	}
	badBoundaries := boundaries
	badBoundaries.Above = badBoundaries.Above[:len(badBoundaries.Above)-1]
	if _, err := ApplyRestorationUnitRecordWithBoundaries(grid, record, badBoundaries, data, dataStride, dataOrigin, dst, dataStride, dataOrigin, bitDepth, scratch, false); !errors.Is(err, ErrInvalidPlan) {
		t.Fatalf("short boundaries err=%v want %v", err, ErrInvalidPlan)
	}
	if _, err := ApplyRestorationUnitRecordWithBoundaries(grid, record, badBoundaries, data, dataStride, dataOrigin, dst, dataStride, dataOrigin, bitDepth, scratch, true); err != nil {
		t.Fatalf("optimized short boundaries err=%v", err)
	}
}

func TestApplyRestorationPlaneRecordsRejectsInvalidInputs(t *testing.T) {
	grid := testRestorationApplyGrid(t, [3]parser.RestorationType{parser.RestorationWiener}, 0)
	records := makeRestorationPlaneRecords(t, grid, func(i int) RestorationUnit {
		return RestorationUnit{Type: parser.RestorationWiener, Wiener: av1restoration.DefaultWienerInfo()}
	})
	const bitDepth = 8
	dataStride := int(grid.PlaneWidth) + 2*restorationApplyFrameHorzBorder + 8
	dataOrigin := restorationBorder*dataStride + restorationApplyFrameHorzBorder
	dataRows := int(grid.PlaneHeight) + 2*restorationBorder
	data := makeRestorationApplySource(dataStride, dataRows, bitDepth)
	dst := make([]uint16, dataStride*dataRows)
	boundaries := makeRestorationApplyBoundaries(t, grid, bitDepth, 5)
	sizes, err := RestorationPlaneApplyScratchLen(grid, records, false)
	if err != nil {
		t.Fatal(err)
	}
	scratch := makeRestorationBoundaryApplyScratch(sizes)

	if _, err := RestorationPlaneApplyScratchLen(grid, records[:len(records)-1], false); !errors.Is(err, ErrInvalidPlan) {
		t.Fatalf("short records scratch err=%v want %v", err, ErrInvalidPlan)
	}
	swapped := append([]RestorationUnitRecord(nil), records...)
	swapped[0], swapped[1] = swapped[1], swapped[0]
	if _, err := ApplyRestorationPlaneRecords(grid, swapped, boundaries, data, dataStride, dataOrigin, dst, dataStride, dataOrigin, bitDepth, scratch, false); !errors.Is(err, ErrInvalidPlan) {
		t.Fatalf("swapped records err=%v want %v", err, ErrInvalidPlan)
	}
	shortScratch := scratch
	shortScratch.Unit.Wiener = shortScratch.Unit.Wiener[:len(shortScratch.Unit.Wiener)-1]
	if _, err := ApplyRestorationPlaneRecords(grid, records, boundaries, data, dataStride, dataOrigin, dst, dataStride, dataOrigin, bitDepth, shortScratch, false); !errors.Is(err, av1restoration.ErrInvalidRestoration) {
		t.Fatalf("short unit scratch err=%v want %v", err, av1restoration.ErrInvalidRestoration)
	}
}

func TestApplyRestorationFrameRejectsInvalidInputs(t *testing.T) {
	const bitDepth = 8
	planes := makeRestorationFramePlanes(t, [3]parser.RestorationType{parser.RestorationWiener, parser.RestorationSGRProj, parser.RestorationNone}, bitDepth, false)
	sizes, err := RestorationFrameScratchLen(planes, false)
	if err != nil {
		t.Fatal(err)
	}
	scratch := makeRestorationBoundaryApplyScratch(sizes)

	if _, err := ApplyRestorationFrame(planes[:2], bitDepth, scratch, false); !errors.Is(err, ErrInvalidPlan) {
		t.Fatalf("two-plane frame err=%v want %v", err, ErrInvalidPlan)
	}
	bad := append([]RestorationFramePlane(nil), planes...)
	bad[2].Grid.Plane = 1
	if _, err := ApplyRestorationFrame(bad, bitDepth, scratch, false); !errors.Is(err, ErrInvalidPlan) {
		t.Fatalf("bad plane order err=%v want %v", err, ErrInvalidPlan)
	}
	shortScratch := scratch
	shortScratch.Unit.Wiener = shortScratch.Unit.Wiener[:len(shortScratch.Unit.Wiener)-1]
	if _, err := ApplyRestorationFrame(planes, bitDepth, shortScratch, false); !errors.Is(err, av1restoration.ErrInvalidRestoration) {
		t.Fatalf("short scratch err=%v want %v", err, av1restoration.ErrInvalidRestoration)
	}
}

func TestApplyRestorationFramePlaneRejectsInvalidInputs(t *testing.T) {
	grid := testRestorationApplyGrid(t, [3]parser.RestorationType{parser.RestorationWiener}, 0)
	records := makeRestorationPlaneRecords(t, grid, func(i int) RestorationUnit {
		return RestorationUnit{Type: parser.RestorationWiener, Wiener: av1restoration.DefaultWienerInfo()}
	})
	const bitDepth = 8
	dataStride := int(grid.PlaneWidth) + 2*restorationApplyFrameHorzBorder + 8
	dataOrigin := restorationBorder*dataStride + restorationApplyFrameHorzBorder
	dataRows := int(grid.PlaneHeight) + 2*restorationBorder
	data := makeRestorationApplySource(dataStride, dataRows, bitDepth)
	dst := make([]uint16, dataStride*dataRows)
	boundaries := makeRestorationApplyBoundaries(t, grid, bitDepth, 5)
	sizes, err := RestorationFramePlaneScratchLen(grid, records, false)
	if err != nil {
		t.Fatal(err)
	}
	scratch := makeRestorationBoundaryApplyScratch(sizes)

	if _, err := ApplyRestorationFramePlane(grid, records, boundaries, data[:dataOrigin], dataStride, dataOrigin, dst, dataStride, dataOrigin, bitDepth, scratch, false); !errors.Is(err, ErrInvalidPlan) {
		t.Fatalf("short data err=%v want %v", err, ErrInvalidPlan)
	}
	shortDst := dst[:dataOrigin+int(grid.PlaneWidth)]
	if _, err := ApplyRestorationFramePlane(grid, records, boundaries, data, dataStride, dataOrigin, shortDst, dataStride, dataOrigin, bitDepth, scratch, false); !errors.Is(err, ErrInvalidPlan) && !errors.Is(err, av1restoration.ErrInvalidRestoration) {
		t.Fatalf("short dst err=%v want %v", err, ErrInvalidPlan)
	}
	badGrid := RestorationPlaneGrid{Plane: 3, Type: parser.RestorationNone}
	if _, err := ApplyRestorationFramePlane(badGrid, nil, RestorationStripeBoundaries{}, data, dataStride, dataOrigin, dst, dataStride, dataOrigin, bitDepth, RestorationUnitRecordBoundaryScratch{}, false); !errors.Is(err, ErrInvalidPlan) {
		t.Fatalf("bad disabled plane err=%v want %v", err, ErrInvalidPlan)
	}
}

func TestApplyRestorationUnitAllocs(t *testing.T) {
	const width, height = 8, 8
	const stride = width + 2*av1restoration.WienerHalfwin
	const origin = av1restoration.WienerHalfwin*stride + av1restoration.WienerHalfwin
	src := makeRestorationApplySource(stride, height+2*av1restoration.WienerHalfwin, 8)
	dst := make([]uint16, width*height)
	sizes, err := RestorationUnitScratchLen(width, height)
	if err != nil {
		t.Fatal(err)
	}
	unit := RestorationUnit{Type: parser.RestorationWiener, Wiener: av1restoration.DefaultWienerInfo()}
	scratch := RestorationUnitScratch{Wiener: make([]uint16, sizes.Wiener)}
	allocs := testing.AllocsPerRun(1000, func() {
		if _, err := ApplyRestorationUnit(src, stride, origin, dst, width, width, height, unit, 8, scratch); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("ApplyRestorationUnit allocated: %f", allocs)
	}
}

func TestApplyRestorationUnitRecordAllocs(t *testing.T) {
	grid := testRestorationApplyGrid(t, [3]parser.RestorationType{parser.RestorationWiener}, 0)
	record := testRestorationApplyRecord(t, grid, 1, 1, RestorationUnit{Type: parser.RestorationWiener, Wiener: av1restoration.DefaultWienerInfo()})
	const bitDepth = 8
	const stride = 320
	const border = av1restoration.WienerHalfwin
	origin := border*stride + border
	rows := int(grid.PlaneHeight) + 2*border
	src := makeRestorationApplySource(stride, rows, bitDepth)
	dst := make([]uint16, stride*rows)
	sizes, err := RestorationUnitRecordScratchLen(grid, record)
	if err != nil {
		t.Fatal(err)
	}
	scratch := RestorationUnitScratch{Wiener: make([]uint16, sizes.Wiener)}
	allocs := testing.AllocsPerRun(1000, func() {
		for i := range dst {
			dst[i] = 0
		}
		if _, err := ApplyRestorationUnitRecord(grid, record, src, stride, origin, dst, stride, origin, bitDepth, scratch); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("ApplyRestorationUnitRecord allocated: %f", allocs)
	}
}

func TestApplyRestorationUnitRecordWithBoundariesAllocs(t *testing.T) {
	grid := testRestorationApplyGrid(t, [3]parser.RestorationType{parser.RestorationWiener}, 0)
	record := testRestorationApplyRecord(t, grid, 1, 1, RestorationUnit{Type: parser.RestorationWiener, Wiener: av1restoration.DefaultWienerInfo()})
	const bitDepth = 8
	dataStride := int(grid.PlaneWidth) + 2*restorationApplyFrameHorzBorder + 8
	dataOrigin := restorationBorder*dataStride + restorationApplyFrameHorzBorder
	dataRows := int(grid.PlaneHeight) + 2*restorationBorder
	data := makeRestorationApplySource(dataStride, dataRows, bitDepth)
	dst := make([]uint16, dataStride*dataRows)
	boundaries := makeRestorationApplyBoundaries(t, grid, bitDepth, 5)
	sizes, err := RestorationUnitRecordBoundaryScratchLen(grid, record, false)
	if err != nil {
		t.Fatal(err)
	}
	scratch := makeRestorationBoundaryApplyScratch(sizes)

	allocs := testing.AllocsPerRun(1000, func() {
		if _, err := ApplyRestorationUnitRecordWithBoundaries(grid, record, boundaries, data, dataStride, dataOrigin, dst, dataStride, dataOrigin, bitDepth, scratch, false); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("ApplyRestorationUnitRecordWithBoundaries allocated: %f", allocs)
	}
}

func TestApplyRestorationPlaneRecordsAllocs(t *testing.T) {
	grid := testRestorationApplyGrid(t, [3]parser.RestorationType{parser.RestorationWiener}, 0)
	records := makeRestorationPlaneRecords(t, grid, func(i int) RestorationUnit {
		if i%2 == 0 {
			return RestorationUnit{Type: parser.RestorationNone}
		}
		return RestorationUnit{Type: parser.RestorationWiener, Wiener: av1restoration.DefaultWienerInfo()}
	})
	const bitDepth = 8
	dataStride := int(grid.PlaneWidth) + 2*restorationApplyFrameHorzBorder + 8
	dataOrigin := restorationBorder*dataStride + restorationApplyFrameHorzBorder
	dataRows := int(grid.PlaneHeight) + 2*restorationBorder
	data := makeRestorationApplySource(dataStride, dataRows, bitDepth)
	dst := make([]uint16, dataStride*dataRows)
	boundaries := makeRestorationApplyBoundaries(t, grid, bitDepth, 5)
	sizes, err := RestorationPlaneApplyScratchLen(grid, records, false)
	if err != nil {
		t.Fatal(err)
	}
	scratch := makeRestorationBoundaryApplyScratch(sizes)

	allocs := testing.AllocsPerRun(1000, func() {
		if _, err := ApplyRestorationPlaneRecords(grid, records, boundaries, data, dataStride, dataOrigin, dst, dataStride, dataOrigin, bitDepth, scratch, false); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("ApplyRestorationPlaneRecords allocated: %f", allocs)
	}
}

func TestApplyRestorationFramePlaneAllocs(t *testing.T) {
	grid := testRestorationApplyGrid(t, [3]parser.RestorationType{parser.RestorationWiener}, 0)
	records := makeRestorationPlaneRecords(t, grid, func(i int) RestorationUnit {
		if i%2 == 0 {
			return RestorationUnit{Type: parser.RestorationNone}
		}
		return RestorationUnit{Type: parser.RestorationWiener, Wiener: av1restoration.DefaultWienerInfo()}
	})
	const bitDepth = 8
	dataStride := int(grid.PlaneWidth) + 2*restorationApplyFrameHorzBorder + 8
	dataOrigin := restorationBorder*dataStride + restorationApplyFrameHorzBorder
	dataRows := int(grid.PlaneHeight) + 2*restorationBorder
	data := makeRestorationApplySource(dataStride, dataRows, bitDepth)
	dst := make([]uint16, dataStride*dataRows)
	boundaries := makeRestorationApplyBoundaries(t, grid, bitDepth, 5)
	sizes, err := RestorationFramePlaneScratchLen(grid, records, false)
	if err != nil {
		t.Fatal(err)
	}
	scratch := makeRestorationBoundaryApplyScratch(sizes)

	allocs := testing.AllocsPerRun(1000, func() {
		if _, err := ApplyRestorationFramePlane(grid, records, boundaries, data, dataStride, dataOrigin, dst, dataStride, dataOrigin, bitDepth, scratch, false); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("ApplyRestorationFramePlane allocated: %f", allocs)
	}
}

func TestApplyRestorationFrameAllocs(t *testing.T) {
	const bitDepth = 8
	planes := makeRestorationFramePlanes(t, [3]parser.RestorationType{parser.RestorationWiener, parser.RestorationSGRProj, parser.RestorationNone}, bitDepth, false)
	sizes, err := RestorationFrameScratchLen(planes, false)
	if err != nil {
		t.Fatal(err)
	}
	scratch := makeRestorationBoundaryApplyScratch(sizes)

	allocs := testing.AllocsPerRun(1000, func() {
		if _, err := ApplyRestorationFrame(planes, bitDepth, scratch, false); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("ApplyRestorationFrame allocated: %f", allocs)
	}
}

func TestApplyRestorationFrameToFrameAllocs(t *testing.T) {
	const bitDepth = 8
	types := [3]parser.RestorationType{parser.RestorationWiener, parser.RestorationNone, parser.RestorationNone}
	plan := testRestorationFramePlan(t, types, false)
	frm := makeRestorationApplyFrame(t, bitDepth, false)
	fillRestorationTestFrame(frm, bitDepth, 0x12345678)
	records := makeRestorationFrameNoneRecords(t, plan)
	size, err := RestorationFrameSampleScratchLen(plan, frm)
	if err != nil {
		t.Fatal(err)
	}
	dataScratch := make([]uint16, size.DataLen)
	dstScratch := make([]uint16, size.DstLen)

	allocs := testing.AllocsPerRun(1000, func() {
		if _, err := ApplyRestorationFrameToFrame(plan, frm, records, [3]RestorationStripeBoundaries{}, dataScratch, dstScratch, RestorationUnitRecordBoundaryScratch{}, false); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("ApplyRestorationFrameToFrame allocated: %f", allocs)
	}
}

func FuzzApplyRestorationUnitNone(f *testing.F) {
	f.Add(uint8(5), uint8(4), uint8(0), []byte{0, 1, 2, 3, 255})
	f.Add(uint8(31), uint8(17), uint8(2), []byte{255, 1, 2, 3, 4})
	f.Fuzz(func(t *testing.T, rawW uint8, rawH uint8, rawDepth uint8, data []byte) {
		width := int(rawW%32) + 1
		height := int(rawH%32) + 1
		bitDepths := [...]uint8{8, 10, 12}
		bitDepth := bitDepths[rawDepth%uint8(len(bitDepths))]
		max := uint16((1 << bitDepth) - 1)
		stride := width + 3
		origin := stride + 1
		src := make([]uint16, origin+(height-1)*stride+width)
		for i := range src {
			src[i] = fuzzRestorationApplySample(data, i, max)
		}
		dst := make([]uint16, width*height)
		if _, err := ApplyRestorationUnit(src, stride, origin, dst, width, width, height, RestorationUnit{Type: parser.RestorationNone}, bitDepth, RestorationUnitScratch{}); err != nil {
			t.Fatalf("ApplyRestorationUnit none err=%v", err)
		}
		for row := range height {
			for col := range width {
				if got, want := dst[row*width+col], src[origin+row*stride+col]; got != want {
					t.Fatalf("row=%d col=%d got=%d want %d", row, col, got, want)
				}
			}
		}
	})
}

func FuzzApplyRestorationUnitRecordNone(f *testing.F) {
	f.Add(uint16(300), uint16(260), uint8(1), uint8(1), uint8(1), uint8(0), []byte{0, 1, 2, 3, 255})
	f.Add(uint16(127), uint16(95), uint8(0), uint8(0), uint8(1), uint8(2), []byte{255, 7, 11})
	f.Fuzz(func(t *testing.T, rawW uint16, rawH uint16, rawUnit uint8, rawCol uint8, rawRow uint8, rawDepth uint8, data []byte) {
		unitSizes := [...]uint16{64, 128, 256}
		unitSize := unitSizes[rawUnit%uint8(len(unitSizes))]
		bitDepths := [...]uint8{8, 10, 12}
		bitDepth := bitDepths[rawDepth%uint8(len(bitDepths))]
		grid, err := BuildRestorationPlaneGrid(parser.RestorationParams{
			Type:      [3]parser.RestorationType{parser.RestorationWiener},
			UnitSizeY: unitSize,
		}, parser.FrameSize{
			UpscaledWidth:       uint32(rawW%512) + 1,
			Height:              uint32(rawH%512) + 1,
			SuperResDenominator: 8,
		}, parser.ColorConfig{}, 0)
		if err != nil {
			t.Fatalf("BuildRestorationPlaneGrid err=%v", err)
		}
		record := testRestorationApplyRecord(t, grid, uint16(rawCol)%grid.HorzUnits, uint16(rawRow)%grid.VertUnits, RestorationUnit{Type: parser.RestorationNone})
		stride := int(grid.PlaneWidth) + 8
		origin := stride + 3
		rows := int(grid.PlaneHeight) + 4
		src := make([]uint16, stride*rows)
		dst := make([]uint16, stride*rows)
		max := uint16((1 << bitDepth) - 1)
		for i := range src {
			src[i] = fuzzRestorationApplySample(data, i, max)
		}
		fillUint16(dst, 0xffff)
		if _, err := ApplyRestorationUnitRecord(grid, record, src, stride, origin, dst, stride, origin, bitDepth, RestorationUnitScratch{}); err != nil {
			t.Fatalf("ApplyRestorationUnitRecord none err=%v", err)
		}
		for y := record.Rect.Y0; y < record.Rect.Y1; y++ {
			for x := record.Rect.X0; x < record.Rect.X1; x++ {
				offset, ok := restorationPlaneOffset(origin, stride, x, y)
				if !ok {
					t.Fatal("bad offset")
				}
				if got, want := dst[offset], src[offset]; got != want {
					t.Fatalf("x=%d y=%d got=%d want %d", x, y, got, want)
				}
			}
		}
	})
}

func FuzzApplyRestorationUnitRecordWithBoundaries(f *testing.F) {
	f.Add(uint16(300), uint16(260), uint8(1), uint8(1), uint8(1), uint8(0), uint8(0), false, false, []byte{0, 1, 2, 3, 255})
	f.Add(uint16(127), uint16(95), uint8(0), uint8(0), uint8(1), uint8(1), uint8(1), true, true, []byte{255, 7, 11})
	f.Fuzz(func(t *testing.T, rawW uint16, rawH uint16, rawUnit uint8, rawCol uint8, rawRow uint8, rawType uint8, rawDepth uint8, ssX bool, ssY bool, dataBytes []byte) {
		unitSizes := [...]uint16{64, 128, 256}
		unitSize := unitSizes[rawUnit%uint8(len(unitSizes))]
		types := [...]parser.RestorationType{parser.RestorationWiener, parser.RestorationSGRProj}
		unitType := types[rawType%uint8(len(types))]
		bitDepths := [...]uint8{8, 10, 12}
		bitDepth := bitDepths[rawDepth%uint8(len(bitDepths))]
		grid, err := BuildRestorationPlaneGrid(parser.RestorationParams{
			Type:       [3]parser.RestorationType{unitType, unitType, unitType},
			UnitSizeY:  unitSize,
			UnitSizeUV: unitSize,
		}, parser.FrameSize{
			UpscaledWidth:       uint32(rawW%256) + 1,
			Height:              uint32(rawH%256) + 1,
			SuperResDenominator: 8,
		}, parser.ColorConfig{SubsamplingX: ssX, SubsamplingY: ssY}, int(rawType%3))
		if err != nil {
			t.Fatalf("BuildRestorationPlaneGrid err=%v", err)
		}
		unit := RestorationUnit{Type: unitType, Wiener: av1restoration.DefaultWienerInfo(), SGRProj: SGRProjInfo{ParamsIndex: 1, XQD: [2]int{13, -9}}}
		record := testRestorationApplyRecord(t, grid, uint16(rawCol)%grid.HorzUnits, uint16(rawRow)%grid.VertUnits, unit)
		dataStride := int(grid.PlaneWidth) + 2*restorationApplyFrameHorzBorder + int(rawUnit%5)
		dataOrigin := restorationBorder*dataStride + restorationApplyFrameHorzBorder
		dataRows := int(grid.PlaneHeight) + 2*restorationBorder
		data := make([]uint16, dataStride*dataRows)
		max := uint16((1 << bitDepth) - 1)
		for i := range data {
			data[i] = fuzzRestorationApplySample(dataBytes, i, max)
		}
		original := append([]uint16(nil), data...)
		dst := make([]uint16, dataStride*dataRows)
		boundaries := makeRestorationApplyBoundaries(t, grid, bitDepth, int(rawUnit)+3)
		sizes, err := RestorationUnitRecordBoundaryScratchLen(grid, record, false)
		if err != nil {
			t.Fatalf("RestorationUnitRecordBoundaryScratchLen err=%v", err)
		}
		result, err := ApplyRestorationUnitRecordWithBoundaries(grid, record, boundaries, data, dataStride, dataOrigin, dst, dataStride, dataOrigin, bitDepth, makeRestorationBoundaryApplyScratch(sizes), false)
		if err != nil {
			t.Fatalf("ApplyRestorationUnitRecordWithBoundaries err=%v", err)
		}
		if !result.Filtered || result.Stripes != record.StripeCount || result.ProcessingUnits == 0 {
			t.Fatalf("result=%+v record=%+v", result, record)
		}
		assertSamplesEqual(t, data, original)
	})
}

func FuzzApplyRestorationPlaneRecords(f *testing.F) {
	f.Add(uint16(300), uint16(260), uint8(1), uint8(0), uint8(0), false, false, []byte{0, 1, 2, 3, 255})
	f.Add(uint16(127), uint16(95), uint8(0), uint8(1), uint8(1), true, true, []byte{255, 7, 11})
	f.Fuzz(func(t *testing.T, rawW uint16, rawH uint16, rawUnit uint8, rawType uint8, rawDepth uint8, ssX bool, ssY bool, dataBytes []byte) {
		unitSizes := [...]uint16{64, 128, 256}
		unitSize := unitSizes[rawUnit%uint8(len(unitSizes))]
		types := [...]parser.RestorationType{parser.RestorationWiener, parser.RestorationSGRProj}
		unitType := types[rawType%uint8(len(types))]
		bitDepths := [...]uint8{8, 10, 12}
		bitDepth := bitDepths[rawDepth%uint8(len(bitDepths))]
		grid, err := BuildRestorationPlaneGrid(parser.RestorationParams{
			Type:       [3]parser.RestorationType{unitType, unitType, unitType},
			UnitSizeY:  unitSize,
			UnitSizeUV: unitSize,
		}, parser.FrameSize{
			UpscaledWidth:       uint32(rawW%256) + 1,
			Height:              uint32(rawH%256) + 1,
			SuperResDenominator: 8,
		}, parser.ColorConfig{SubsamplingX: ssX, SubsamplingY: ssY}, int(rawType%3))
		if err != nil {
			t.Fatalf("BuildRestorationPlaneGrid err=%v", err)
		}
		filterUnit := RestorationUnit{Type: unitType, Wiener: av1restoration.DefaultWienerInfo(), SGRProj: SGRProjInfo{ParamsIndex: 1, XQD: [2]int{13, -9}}}
		records := makeRestorationPlaneRecords(t, grid, func(i int) RestorationUnit {
			if (i+int(rawUnit))%3 == 0 {
				return RestorationUnit{Type: parser.RestorationNone}
			}
			return filterUnit
		})
		dataStride := int(grid.PlaneWidth) + 2*restorationApplyFrameHorzBorder + int(rawUnit%5)
		dataOrigin := restorationBorder*dataStride + restorationApplyFrameHorzBorder
		dataRows := int(grid.PlaneHeight) + 2*restorationBorder
		data := make([]uint16, dataStride*dataRows)
		max := uint16((1 << bitDepth) - 1)
		for i := range data {
			data[i] = fuzzRestorationApplySample(dataBytes, i, max)
		}
		original := append([]uint16(nil), data...)
		dst := make([]uint16, dataStride*dataRows)
		boundaries := makeRestorationApplyBoundaries(t, grid, bitDepth, int(rawUnit)+7)
		sizes, err := RestorationPlaneApplyScratchLen(grid, records, false)
		if err != nil {
			t.Fatalf("RestorationPlaneApplyScratchLen err=%v", err)
		}
		result, err := ApplyRestorationPlaneRecords(grid, records, boundaries, data, dataStride, dataOrigin, dst, dataStride, dataOrigin, bitDepth, makeRestorationBoundaryApplyScratch(sizes), false)
		if err != nil {
			t.Fatalf("ApplyRestorationPlaneRecords err=%v", err)
		}
		if result.Records != uint32(len(records)) {
			t.Fatalf("records=%d want %d result=%+v", result.Records, len(records), result)
		}
		assertSamplesEqual(t, data, original)
	})
}

func FuzzApplyRestorationFrame(f *testing.F) {
	f.Add(uint16(160), uint16(112), uint8(0), uint8(0), uint8(0), false, false, false, []byte{0, 1, 2, 3, 255})
	f.Add(uint16(127), uint16(95), uint8(1), uint8(1), uint8(2), true, true, true, []byte{255, 7, 11})
	f.Fuzz(func(t *testing.T, rawW uint16, rawH uint16, rawUnit uint8, rawType uint8, rawDepth uint8, ssX bool, ssY bool, mono bool, dataBytes []byte) {
		unitSizes := [...]uint16{64, 128, 256}
		unitSize := unitSizes[rawUnit%uint8(len(unitSizes))]
		types := [...]parser.RestorationType{parser.RestorationWiener, parser.RestorationSGRProj}
		unitType := types[rawType%uint8(len(types))]
		bitDepths := [...]uint8{8, 10, 12}
		bitDepth := bitDepths[rawDepth%uint8(len(bitDepths))]
		params := parser.RestorationParams{
			Type:       [3]parser.RestorationType{unitType, unitType, parser.RestorationNone},
			UnitSizeY:  unitSize,
			UnitSizeUV: unitSize,
		}
		color := parser.ColorConfig{MonoChrome: mono, SubsamplingX: ssX, SubsamplingY: ssY}
		size := parser.FrameSize{
			UpscaledWidth:       uint32(rawW%160) + 1,
			Height:              uint32(rawH%160) + 1,
			SuperResDenominator: 8,
		}
		numPlanes := 3
		if mono {
			numPlanes = 1
		}
		planes := make([]RestorationFramePlane, numPlanes)
		max := uint16((1 << bitDepth) - 1)
		for plane := 0; plane < numPlanes; plane++ {
			grid, err := BuildRestorationPlaneGrid(params, size, color, plane)
			if err != nil {
				t.Fatalf("BuildRestorationPlaneGrid plane=%d err=%v", plane, err)
			}
			var records []RestorationUnitRecord
			if grid.Type != parser.RestorationNone {
				filterUnit := RestorationUnit{Type: unitType, Wiener: av1restoration.DefaultWienerInfo(), SGRProj: SGRProjInfo{ParamsIndex: 1, XQD: [2]int{13, -9}}}
				records = makeRestorationPlaneRecords(t, grid, func(i int) RestorationUnit {
					if (i+int(rawUnit)+plane)%4 == 0 {
						return RestorationUnit{Type: parser.RestorationNone}
					}
					return filterUnit
				})
			}
			dataStride := int(grid.PlaneWidth) + 2*restorationApplyFrameHorzBorder + int(rawUnit%5)
			dataOrigin := restorationBorder*dataStride + restorationApplyFrameHorzBorder
			dataRows := int(grid.PlaneHeight) + 2*restorationBorder
			data := make([]uint16, dataStride*dataRows)
			for i := range data {
				data[i] = fuzzRestorationApplySample(dataBytes, i+plane*13, max)
			}
			boundaries := RestorationStripeBoundaries{}
			if grid.Type != parser.RestorationNone {
				boundaries = makeRestorationApplyBoundaries(t, grid, bitDepth, int(rawUnit)+plane+17)
			}
			planes[plane] = RestorationFramePlane{
				Grid:       grid,
				Records:    records,
				Boundaries: boundaries,
				Data:       data,
				DataStride: dataStride,
				DataOrigin: dataOrigin,
				Dst:        make([]uint16, dataStride*dataRows),
				DstStride:  dataStride,
				DstOrigin:  dataOrigin,
			}
		}
		manualPlanes := cloneRestorationFramePlanes(planes)
		sizes, err := RestorationFrameScratchLen(planes, false)
		if err != nil {
			t.Fatalf("RestorationFrameScratchLen err=%v", err)
		}
		result, err := ApplyRestorationFrame(planes, bitDepth, makeRestorationBoundaryApplyScratch(sizes), false)
		if err != nil {
			t.Fatalf("ApplyRestorationFrame err=%v", err)
		}
		want := applyRestorationFrameManually(t, manualPlanes, bitDepth, makeRestorationBoundaryApplyScratch(sizes), false)
		if result != want {
			t.Fatalf("result=%+v want %+v", result, want)
		}
		assertRestorationFramePlanesEqual(t, planes, manualPlanes)
	})
}

func FuzzApplyRestorationFrameToFrame(f *testing.F) {
	f.Add(uint16(64), uint16(48), uint8(0), false, false, false, uint32(0x12345678))
	f.Add(uint16(127), uint16(95), uint8(2), true, true, false, uint32(0x90abcdef))
	f.Fuzz(func(t *testing.T, rawW uint16, rawH uint16, rawDepth uint8, ssX bool, ssY bool, mono bool, seed uint32) {
		bitDepths := [...]uint8{8, 10, 12}
		bitDepth := bitDepths[rawDepth%uint8(len(bitDepths))]
		width := int(rawW%128) + 1
		height := int(rawH%128) + 1
		format := av1frame.Format{
			Width:        width,
			Height:       height,
			BitDepth:     bitDepth,
			MonoChrome:   mono,
			SubsamplingX: ssX,
			SubsamplingY: ssY,
			Align:        64,
		}
		layout, err := av1frame.RequiredSize(format)
		if err != nil {
			t.Fatalf("RequiredSize err=%v", err)
		}
		buffer := make([]byte, layout.Size)
		frm, err := av1frame.Bind(buffer, format)
		if err != nil {
			t.Fatalf("Bind err=%v", err)
		}
		fillRestorationTestFrame(frm, bitDepth, seed)
		original := append([]byte(nil), buffer...)

		params := parser.RestorationParams{
			Type:       [3]parser.RestorationType{parser.RestorationWiener, parser.RestorationNone, parser.RestorationNone},
			UnitSizeY:  64,
			UnitSizeUV: 64,
		}
		size := parser.FrameSize{UpscaledWidth: uint32(width), Height: uint32(height), SuperResDenominator: 8}
		plan, err := BuildRestorationFramePlan(params, size, parser.ColorConfig{MonoChrome: mono, SubsamplingX: ssX, SubsamplingY: ssY})
		if err != nil {
			t.Fatalf("BuildRestorationFramePlan err=%v", err)
		}
		records := makeRestorationFrameNoneRecords(t, plan)
		sampleSize, err := RestorationFrameSampleScratchLen(plan, frm)
		if err != nil {
			t.Fatalf("RestorationFrameSampleScratchLen err=%v", err)
		}
		result, err := ApplyRestorationFrameToFrame(plan, frm, records, [3]RestorationStripeBoundaries{}, make([]uint16, sampleSize.DataLen), make([]uint16, sampleSize.DstLen), RestorationUnitRecordBoundaryScratch{}, false)
		if err != nil {
			t.Fatalf("ApplyRestorationFrameToFrame err=%v", err)
		}
		if result.Planes != 1 || result.Records != uint32(plan.UnitRecords[0]) || result.FilteredRecords != 0 {
			t.Fatalf("result=%+v plan=%+v", result, plan)
		}
		if string(buffer) != string(original) {
			t.Fatalf("all-none frame restore changed visible bytes")
		}
	})
}

func FuzzApplyRestorationFramePlane(f *testing.F) {
	f.Add(uint16(160), uint16(112), uint8(0), uint8(0), uint8(0), uint8(0), false, false, []byte{0, 1, 2, 3, 255})
	f.Add(uint16(127), uint16(95), uint8(1), uint8(1), uint8(1), uint8(2), true, true, []byte{255, 7, 11})
	f.Fuzz(func(t *testing.T, rawW uint16, rawH uint16, rawUnit uint8, rawType uint8, rawPlane uint8, rawDepth uint8, ssX bool, ssY bool, dataBytes []byte) {
		unitSizes := [...]uint16{64, 128, 256}
		unitSize := unitSizes[rawUnit%uint8(len(unitSizes))]
		types := [...]parser.RestorationType{parser.RestorationWiener, parser.RestorationSGRProj}
		unitType := types[rawType%uint8(len(types))]
		bitDepths := [...]uint8{8, 10, 12}
		bitDepth := bitDepths[rawDepth%uint8(len(bitDepths))]
		grid, err := BuildRestorationPlaneGrid(parser.RestorationParams{
			Type:       [3]parser.RestorationType{unitType, unitType, unitType},
			UnitSizeY:  unitSize,
			UnitSizeUV: unitSize,
		}, parser.FrameSize{
			UpscaledWidth:       uint32(rawW%192) + 1,
			Height:              uint32(rawH%192) + 1,
			SuperResDenominator: 8,
		}, parser.ColorConfig{SubsamplingX: ssX, SubsamplingY: ssY}, int(rawPlane%3))
		if err != nil {
			t.Fatalf("BuildRestorationPlaneGrid err=%v", err)
		}
		filterUnit := RestorationUnit{Type: unitType, Wiener: av1restoration.DefaultWienerInfo(), SGRProj: SGRProjInfo{ParamsIndex: 1, XQD: [2]int{13, -9}}}
		records := makeRestorationPlaneRecords(t, grid, func(i int) RestorationUnit {
			if (i+int(rawUnit))%4 == 0 {
				return RestorationUnit{Type: parser.RestorationNone}
			}
			return filterUnit
		})
		dataStride := int(grid.PlaneWidth) + 2*restorationApplyFrameHorzBorder + int(rawUnit%5)
		dataOrigin := restorationBorder*dataStride + restorationApplyFrameHorzBorder
		dataRows := int(grid.PlaneHeight) + 2*restorationBorder
		data := make([]uint16, dataStride*dataRows)
		max := uint16((1 << bitDepth) - 1)
		for i := range data {
			data[i] = fuzzRestorationApplySample(dataBytes, i, max)
		}
		manualData := append([]uint16(nil), data...)
		gotDst := make([]uint16, dataStride*dataRows)
		wantDst := make([]uint16, dataStride*dataRows)
		boundaries := makeRestorationApplyBoundaries(t, grid, bitDepth, int(rawUnit)+11)
		sizes, err := RestorationFramePlaneScratchLen(grid, records, false)
		if err != nil {
			t.Fatalf("RestorationFramePlaneScratchLen err=%v", err)
		}
		result, err := ApplyRestorationFramePlane(grid, records, boundaries, data, dataStride, dataOrigin, gotDst, dataStride, dataOrigin, bitDepth, makeRestorationBoundaryApplyScratch(sizes), false)
		if err != nil {
			t.Fatalf("ApplyRestorationFramePlane err=%v", err)
		}
		wantResult := applyRestorationFramePlaneManually(t, grid, records, boundaries, manualData, dataStride, dataOrigin, wantDst, dataStride, dataOrigin, bitDepth, makeRestorationBoundaryApplyScratch(sizes), false)
		if result != wantResult {
			t.Fatalf("result=%+v want %+v", result, wantResult)
		}
		assertSamplesEqual(t, gotDst, wantDst)
		assertSamplesEqual(t, data, manualData)
	})
}

func BenchmarkApplyRestorationUnitWiener(b *testing.B) {
	const width, height = 64, 64
	const bitDepth = 12
	stride := width + 2*av1restoration.WienerHalfwin
	origin := av1restoration.WienerHalfwin*stride + av1restoration.WienerHalfwin
	src := makeRestorationApplySource(stride, height+2*av1restoration.WienerHalfwin, bitDepth)
	dst := make([]uint16, width*height)
	sizes, err := RestorationUnitScratchLen(width, height)
	if err != nil {
		b.Fatal(err)
	}
	unit := RestorationUnit{Type: parser.RestorationWiener, Wiener: av1restoration.DefaultWienerInfo()}
	scratch := RestorationUnitScratch{Wiener: make([]uint16, sizes.Wiener)}
	b.ReportAllocs()
	b.SetBytes(width * height * 2)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := ApplyRestorationUnit(src, stride, origin, dst, width, width, height, unit, bitDepth, scratch); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkApplyRestorationUnitRecordWiener(b *testing.B) {
	grid := testRestorationApplyGrid(b, [3]parser.RestorationType{parser.RestorationWiener}, 0)
	record := testRestorationApplyRecord(b, grid, 1, 1, RestorationUnit{Type: parser.RestorationWiener, Wiener: av1restoration.DefaultWienerInfo()})
	const bitDepth = 12
	const stride = 320
	const border = av1restoration.WienerHalfwin
	origin := border*stride + border
	rows := int(grid.PlaneHeight) + 2*border
	src := makeRestorationApplySource(stride, rows, bitDepth)
	dst := make([]uint16, stride*rows)
	sizes, err := RestorationUnitRecordScratchLen(grid, record)
	if err != nil {
		b.Fatal(err)
	}
	scratch := RestorationUnitScratch{Wiener: make([]uint16, sizes.Wiener)}
	b.ReportAllocs()
	b.SetBytes(int64(record.Rect.Width() * record.Rect.Height() * 2))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := ApplyRestorationUnitRecord(grid, record, src, stride, origin, dst, stride, origin, bitDepth, scratch); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkApplyRestorationUnitRecordWithBoundariesWiener(b *testing.B) {
	grid := testRestorationApplyGrid(b, [3]parser.RestorationType{parser.RestorationWiener}, 0)
	record := testRestorationApplyRecord(b, grid, 1, 1, RestorationUnit{Type: parser.RestorationWiener, Wiener: av1restoration.DefaultWienerInfo()})
	const bitDepth = 12
	dataStride := int(grid.PlaneWidth) + 2*restorationApplyFrameHorzBorder + 8
	dataOrigin := restorationBorder*dataStride + restorationApplyFrameHorzBorder
	dataRows := int(grid.PlaneHeight) + 2*restorationBorder
	data := makeRestorationApplySource(dataStride, dataRows, bitDepth)
	dst := make([]uint16, dataStride*dataRows)
	boundaries := makeRestorationApplyBoundaries(b, grid, bitDepth, 5)
	sizes, err := RestorationUnitRecordBoundaryScratchLen(grid, record, false)
	if err != nil {
		b.Fatal(err)
	}
	scratch := makeRestorationBoundaryApplyScratch(sizes)
	b.ReportAllocs()
	b.SetBytes(int64(record.Rect.Width() * record.Rect.Height() * 2))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := ApplyRestorationUnitRecordWithBoundaries(grid, record, boundaries, data, dataStride, dataOrigin, dst, dataStride, dataOrigin, bitDepth, scratch, false); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkApplyRestorationPlaneRecordsWiener(b *testing.B) {
	grid := testRestorationApplyGrid(b, [3]parser.RestorationType{parser.RestorationWiener}, 0)
	records := makeRestorationPlaneRecords(b, grid, func(i int) RestorationUnit {
		return RestorationUnit{Type: parser.RestorationWiener, Wiener: av1restoration.DefaultWienerInfo()}
	})
	const bitDepth = 12
	dataStride := int(grid.PlaneWidth) + 2*restorationApplyFrameHorzBorder + 8
	dataOrigin := restorationBorder*dataStride + restorationApplyFrameHorzBorder
	dataRows := int(grid.PlaneHeight) + 2*restorationBorder
	data := makeRestorationApplySource(dataStride, dataRows, bitDepth)
	dst := make([]uint16, dataStride*dataRows)
	boundaries := makeRestorationApplyBoundaries(b, grid, bitDepth, 5)
	sizes, err := RestorationPlaneApplyScratchLen(grid, records, false)
	if err != nil {
		b.Fatal(err)
	}
	scratch := makeRestorationBoundaryApplyScratch(sizes)
	b.ReportAllocs()
	b.SetBytes(int64(grid.PlaneWidth * grid.PlaneHeight * 2))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := ApplyRestorationPlaneRecords(grid, records, boundaries, data, dataStride, dataOrigin, dst, dataStride, dataOrigin, bitDepth, scratch, false); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkApplyRestorationFramePlaneWiener(b *testing.B) {
	grid := testRestorationApplyGrid(b, [3]parser.RestorationType{parser.RestorationWiener}, 0)
	records := makeRestorationPlaneRecords(b, grid, func(i int) RestorationUnit {
		return RestorationUnit{Type: parser.RestorationWiener, Wiener: av1restoration.DefaultWienerInfo()}
	})
	const bitDepth = 12
	dataStride := int(grid.PlaneWidth) + 2*restorationApplyFrameHorzBorder + 8
	dataOrigin := restorationBorder*dataStride + restorationApplyFrameHorzBorder
	dataRows := int(grid.PlaneHeight) + 2*restorationBorder
	data := makeRestorationApplySource(dataStride, dataRows, bitDepth)
	dst := make([]uint16, dataStride*dataRows)
	boundaries := makeRestorationApplyBoundaries(b, grid, bitDepth, 5)
	sizes, err := RestorationFramePlaneScratchLen(grid, records, false)
	if err != nil {
		b.Fatal(err)
	}
	scratch := makeRestorationBoundaryApplyScratch(sizes)
	b.ReportAllocs()
	b.SetBytes(int64(grid.PlaneWidth * grid.PlaneHeight * 2))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := ApplyRestorationFramePlane(grid, records, boundaries, data, dataStride, dataOrigin, dst, dataStride, dataOrigin, bitDepth, scratch, false); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkApplyRestorationFrameWienerSGR(b *testing.B) {
	const bitDepth = 12
	planes := makeRestorationFramePlanes(b, [3]parser.RestorationType{parser.RestorationWiener, parser.RestorationSGRProj, parser.RestorationNone}, bitDepth, false)
	sizes, err := RestorationFrameScratchLen(planes, false)
	if err != nil {
		b.Fatal(err)
	}
	scratch := makeRestorationBoundaryApplyScratch(sizes)
	var bytes int64
	for i := range planes {
		bytes += int64(planes[i].Grid.PlaneWidth * planes[i].Grid.PlaneHeight * 2)
	}
	b.ReportAllocs()
	b.SetBytes(bytes)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := ApplyRestorationFrame(planes, bitDepth, scratch, false); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkApplyRestorationFrameToFrameWienerSGR(b *testing.B) {
	const bitDepth = 12
	types := [3]parser.RestorationType{parser.RestorationWiener, parser.RestorationSGRProj, parser.RestorationNone}
	plan := testRestorationFramePlan(b, types, false)
	planes := makeRestorationFramePlanes(b, types, bitDepth, false)
	frm := makeRestorationApplyFrame(b, bitDepth, false)
	fillFrameFromRestorationPlanes(b, frm, planes, bitDepth)
	var records [3][]RestorationUnitRecord
	var boundaries [3]RestorationStripeBoundaries
	for i := range planes {
		records[i] = planes[i].Records
		boundaries[i] = planes[i].Boundaries
	}
	sampleSize, err := RestorationFrameSampleScratchLen(plan, frm)
	if err != nil {
		b.Fatal(err)
	}
	applySize, err := RestorationFrameScratchLen(planes, false)
	if err != nil {
		b.Fatal(err)
	}
	dataScratch := make([]uint16, sampleSize.DataLen)
	dstScratch := make([]uint16, sampleSize.DstLen)
	scratch := makeRestorationBoundaryApplyScratch(applySize)
	var bytes int64
	for i := range planes {
		if planes[i].Grid.Type != parser.RestorationNone {
			bytes += int64(planes[i].Grid.PlaneWidth * planes[i].Grid.PlaneHeight * 2)
		}
	}
	b.ReportAllocs()
	b.SetBytes(bytes)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := ApplyRestorationFrameToFrame(plan, frm, records, boundaries, dataScratch, dstScratch, scratch, false); err != nil {
			b.Fatal(err)
		}
	}
}

func makeRestorationApplySource(stride int, height int, bitDepth uint8) []uint16 {
	max := uint16((1 << bitDepth) - 1)
	src := make([]uint16, stride*height)
	for row := range height {
		for col := range stride {
			src[row*stride+col] = uint16((row*37 + col*19 + row*col) & int(max))
		}
	}
	return src
}

func makeRestorationApplyBoundaries(tb testing.TB, grid RestorationPlaneGrid, bitDepth uint8, salt int) RestorationStripeBoundaries {
	tb.Helper()
	size, err := RestorationStripeBoundaryBufferLen(grid)
	if err != nil {
		tb.Fatal(err)
	}
	boundaries := makeSentinelRestorationBoundaries(size, 0)
	srcStride := int(grid.PlaneWidth) + 3
	deblock := makeRestorationBoundaryPlaneWithSalt(grid, srcStride, bitDepth, salt)
	cdef := makeRestorationBoundaryPlaneWithSalt(grid, srcStride, bitDepth, salt+17)
	if err := SaveRestorationBoundaryLines(grid, deblock, srcStride, 0, boundaries, false); err != nil {
		tb.Fatal(err)
	}
	if err := SaveRestorationBoundaryLines(grid, cdef, srcStride, 0, boundaries, true); err != nil {
		tb.Fatal(err)
	}
	return boundaries
}

func makeRestorationFramePlanes(tb testing.TB, types [3]parser.RestorationType, bitDepth uint8, mono bool) []RestorationFramePlane {
	tb.Helper()
	params := parser.RestorationParams{
		Type:       types,
		UnitSizeY:  128,
		UnitSizeUV: 64,
	}
	color := parser.ColorConfig{MonoChrome: mono, SubsamplingX: true, SubsamplingY: true}
	size := parser.FrameSize{UpscaledWidth: 300, Height: 260, SuperResDenominator: 8}
	numPlanes := 3
	if mono {
		numPlanes = 1
	}
	planes := make([]RestorationFramePlane, numPlanes)
	for plane := 0; plane < numPlanes; plane++ {
		grid, err := BuildRestorationPlaneGrid(params, size, color, plane)
		if err != nil {
			tb.Fatal(err)
		}
		var records []RestorationUnitRecord
		var boundaries RestorationStripeBoundaries
		if grid.Type != parser.RestorationNone {
			unit := RestorationUnit{Type: grid.Type, Wiener: av1restoration.DefaultWienerInfo(), SGRProj: SGRProjInfo{ParamsIndex: 1, XQD: [2]int{13, -9}}}
			if grid.Type == parser.RestorationSwitchable {
				unit.Type = parser.RestorationWiener
			}
			records = makeRestorationPlaneRecords(tb, grid, func(i int) RestorationUnit {
				if (i+plane)%4 == 0 {
					return RestorationUnit{Type: parser.RestorationNone}
				}
				return unit
			})
			boundaries = makeRestorationApplyBoundaries(tb, grid, bitDepth, 19+plane)
		}
		dataStride := int(grid.PlaneWidth) + 2*restorationApplyFrameHorzBorder + 8
		dataOrigin := restorationBorder*dataStride + restorationApplyFrameHorzBorder
		dataRows := int(grid.PlaneHeight) + 2*restorationBorder
		data := makeRestorationApplySource(dataStride, dataRows, bitDepth)
		dst := make([]uint16, dataStride*dataRows)
		fillUint16(dst, 0xeeee)
		planes[plane] = RestorationFramePlane{
			Grid:       grid,
			Records:    records,
			Boundaries: boundaries,
			Data:       data,
			DataStride: dataStride,
			DataOrigin: dataOrigin,
			Dst:        dst,
			DstStride:  dataStride,
			DstOrigin:  dataOrigin,
		}
	}
	return planes
}

func makeRestorationApplyFrame(tb testing.TB, bitDepth uint8, mono bool) av1frame.Frame {
	tb.Helper()
	format := av1frame.Format{
		Width:        300,
		Height:       260,
		BitDepth:     bitDepth,
		MonoChrome:   mono,
		SubsamplingX: true,
		SubsamplingY: true,
		Align:        64,
	}
	layout, err := av1frame.RequiredSize(format)
	if err != nil {
		tb.Fatal(err)
	}
	frm, err := av1frame.Bind(make([]byte, layout.Size), format)
	if err != nil {
		tb.Fatal(err)
	}
	return frm
}

func makeRestorationFrameNoneRecords(tb testing.TB, plan RestorationFramePlan) [3][]RestorationUnitRecord {
	tb.Helper()
	var records [3][]RestorationUnitRecord
	for plane := 0; plane < int(plan.Planes); plane++ {
		grid := plan.Grids[plane]
		if grid.Type == parser.RestorationNone {
			continue
		}
		records[plane] = make([]RestorationUnitRecord, plan.UnitRecords[plane])
		if err := ResetRestorationPlaneRecords(grid, records[plane]); err != nil {
			tb.Fatal(err)
		}
	}
	return records
}

func fillFrameFromRestorationPlanes(tb testing.TB, frm av1frame.Frame, planes []RestorationFramePlane, bitDepth uint8) {
	tb.Helper()
	bytesPerSample := restorationTestBytesPerSample(bitDepth)
	for plane := range planes {
		dst := restorationTestFramePlane(frm, plane)
		src := planes[plane]
		if dst.Width != int(src.Grid.PlaneWidth) || dst.Height != int(src.Grid.PlaneHeight) {
			tb.Fatalf("plane %d frame=%+v grid=%+v", plane, dst, src.Grid)
		}
		for y := 0; y < dst.Height; y++ {
			for x := 0; x < dst.Width; x++ {
				sample := src.Data[src.DataOrigin+y*src.DataStride+x]
				setRestorationFrameByteSample(dst, bytesPerSample, x, y, sample)
			}
		}
	}
}

func assertFrameMatchesRestorationPlanes(tb testing.TB, frm av1frame.Frame, planes []RestorationFramePlane, bitDepth uint8) {
	tb.Helper()
	bytesPerSample := restorationTestBytesPerSample(bitDepth)
	for plane := range planes {
		gotPlane := restorationTestFramePlane(frm, plane)
		wantPlane := planes[plane]
		for y := 0; y < gotPlane.Height; y++ {
			for x := 0; x < gotPlane.Width; x++ {
				got := getRestorationFrameByteSample(gotPlane, bytesPerSample, x, y)
				want := wantPlane.Data[wantPlane.DataOrigin+y*wantPlane.DataStride+x]
				if got != want {
					tb.Fatalf("plane=%d x=%d y=%d got=%d want %d", plane, x, y, got, want)
				}
			}
		}
	}
}

func fillRestorationTestFrame(frm av1frame.Frame, bitDepth uint8, seed uint32) {
	bytesPerSample := restorationTestBytesPerSample(bitDepth)
	max := uint16((1 << bitDepth) - 1)
	planes := [3]av1frame.Plane{frm.Y, frm.U, frm.V}
	count := 3
	if frm.Format.MonoChrome {
		count = 1
	}
	state := seed
	for plane := 0; plane < count; plane++ {
		for y := 0; y < planes[plane].Height; y++ {
			for x := 0; x < planes[plane].Width; x++ {
				state = state*1664525 + 1013904223 + uint32(plane+1)
				setRestorationFrameByteSample(planes[plane], bytesPerSample, x, y, uint16(state)&max)
			}
		}
	}
}

func restorationTestFramePlane(frm av1frame.Frame, plane int) av1frame.Plane {
	switch plane {
	case 0:
		return frm.Y
	case 1:
		return frm.U
	case 2:
		return frm.V
	default:
		return av1frame.Plane{}
	}
}

func restorationTestBytesPerSample(bitDepth uint8) int {
	if bitDepth > 8 {
		return 2
	}
	return 1
}

func getRestorationFrameByteSample(plane av1frame.Plane, bytesPerSample int, x int, y int) uint16 {
	offset := y*plane.Stride + x*bytesPerSample
	if bytesPerSample == 1 {
		return uint16(plane.Pix[offset])
	}
	return uint16(plane.Pix[offset]) | uint16(plane.Pix[offset+1])<<8
}

func setRestorationFrameByteSample(plane av1frame.Plane, bytesPerSample int, x int, y int, value uint16) {
	offset := y*plane.Stride + x*bytesPerSample
	if bytesPerSample == 1 {
		plane.Pix[offset] = byte(value)
		return
	}
	plane.Pix[offset] = byte(value)
	plane.Pix[offset+1] = byte(value >> 8)
}

func makeRestorationBoundaryPlaneWithSalt(grid RestorationPlaneGrid, stride int, bitDepth uint8, salt int) []uint16 {
	max := uint16((1 << bitDepth) - 1)
	src := make([]uint16, stride*int(grid.PlaneHeight))
	for y := uint32(0); y < grid.PlaneHeight; y++ {
		for x := uint32(0); x < grid.PlaneWidth; x++ {
			src[int(y)*stride+int(x)] = uint16((int(y)*97 + int(x)*31 + salt*53 + int(x*y)) & int(max))
		}
	}
	return src
}

func makeRestorationBoundaryApplyScratch(size RestorationUnitRecordBoundaryScratchSize) RestorationUnitRecordBoundaryScratch {
	return RestorationUnitRecordBoundaryScratch{
		Unit: RestorationUnitScratch{
			Wiener:  make([]uint16, size.Unit.Wiener),
			SGRProj: make([]int32, size.Unit.SGRProj),
		},
		Boundary: RestorationStripeBoundaryScratch{
			Above: make([]uint16, size.Boundary.Above),
			Below: make([]uint16, size.Boundary.Below),
		},
	}
}

func testRestorationApplyGrid(tb testing.TB, types [3]parser.RestorationType, plane int) RestorationPlaneGrid {
	tb.Helper()
	params := parser.RestorationParams{
		Type:       types,
		UnitSizeY:  128,
		UnitSizeUV: 64,
	}
	grid, err := BuildRestorationPlaneGrid(params, parser.FrameSize{UpscaledWidth: 300, Height: 260, SuperResDenominator: 8}, parser.ColorConfig{SubsamplingX: true, SubsamplingY: true}, plane)
	if err != nil {
		tb.Fatal(err)
	}
	return grid
}

func testRestorationApplyRecord(tb testing.TB, grid RestorationPlaneGrid, col uint16, row uint16, unit RestorationUnit) RestorationUnitRecord {
	tb.Helper()
	rect, err := grid.UnitRect(col, row)
	if err != nil {
		tb.Fatal(err)
	}
	stripeCount, err := grid.ProcessingStripeCount(rect)
	if err != nil {
		tb.Fatal(err)
	}
	return RestorationUnitRecord{
		Index:       uint32(row)*uint32(grid.HorzUnits) + uint32(col),
		Col:         col,
		Row:         row,
		Rect:        rect,
		StripeCount: uint8(stripeCount),
		Unit:        unit,
	}
}

func makeRestorationPlaneRecords(tb testing.TB, grid RestorationPlaneGrid, unitForIndex func(int) RestorationUnit) []RestorationUnitRecord {
	tb.Helper()
	count, err := restorationPlaneRecordCount(grid)
	if err != nil {
		tb.Fatal(err)
	}
	records := make([]RestorationUnitRecord, 0, count)
	for row := uint16(0); row < grid.VertUnits; row++ {
		for col := uint16(0); col < grid.HorzUnits; col++ {
			records = append(records, testRestorationApplyRecord(tb, grid, col, row, unitForIndex(len(records))))
		}
	}
	return records
}

func applyRestorationRecordByProcessingUnits(tb testing.TB, grid RestorationPlaneGrid, record RestorationUnitRecord, src []uint16, srcStride int, srcOrigin int, dst []uint16, dstStride int, dstOrigin int, bitDepth uint8, scratch RestorationUnitScratch) {
	tb.Helper()
	for stripeIndex := 0; stripeIndex < int(record.StripeCount); stripeIndex++ {
		stripe, ok, err := grid.ProcessingStripe(record.Rect, stripeIndex)
		if err != nil || !ok {
			tb.Fatalf("stripe %d ok=%v err=%v", stripeIndex, ok, err)
		}
		unitCount, err := grid.ProcessingUnitCount(stripe, record.Unit.Type)
		if err != nil {
			tb.Fatal(err)
		}
		for unitIndex := range unitCount {
			unit, ok, err := grid.ProcessingUnit(stripe, record.Unit.Type, unitIndex)
			if err != nil || !ok {
				tb.Fatalf("unit %d ok=%v err=%v", unitIndex, ok, err)
			}
			srcBlock, ok := restorationPlaneOffset(srcOrigin, srcStride, unit.FilterRect.X0, unit.FilterRect.Y0)
			if !ok {
				tb.Fatal("bad src block")
			}
			dstBlock, ok := restorationPlaneOffset(dstOrigin, dstStride, unit.FilterRect.X0, unit.FilterRect.Y0)
			if !ok || dstBlock > len(dst) {
				tb.Fatal("bad dst block")
			}
			if _, err := ApplyRestorationUnit(src, srcStride, srcBlock, dst[dstBlock:], dstStride, int(unit.FilterRect.Width()), int(unit.FilterRect.Height()), record.Unit, bitDepth, scratch); err != nil {
				tb.Fatal(err)
			}
		}
	}
}

func applyRestorationPlaneRecordsManually(tb testing.TB, grid RestorationPlaneGrid, records []RestorationUnitRecord, boundaries RestorationStripeBoundaries, data []uint16, dataStride int, dataOrigin int, dst []uint16, dstStride int, dstOrigin int, bitDepth uint8, scratch RestorationUnitRecordBoundaryScratch, optimized bool) RestorationPlaneApplyResult {
	tb.Helper()
	var result RestorationPlaneApplyResult
	for i := range records {
		recordResult, err := ApplyRestorationUnitRecordWithBoundaries(grid, records[i], boundaries, data, dataStride, dataOrigin, dst, dstStride, dstOrigin, bitDepth, scratch, optimized)
		if err != nil {
			tb.Fatal(err)
		}
		result.Records++
		if recordResult.Filtered {
			result.FilteredRecords++
		}
		result.Stripes += uint32(recordResult.Stripes)
		result.ProcessingUnits += uint32(recordResult.ProcessingUnits)
	}
	return result
}

func applyRestorationFrameManually(tb testing.TB, planes []RestorationFramePlane, bitDepth uint8, scratch RestorationUnitRecordBoundaryScratch, optimized bool) RestorationFrameApplyResult {
	tb.Helper()
	var result RestorationFrameApplyResult
	for i := range planes {
		planeResult, err := ApplyRestorationFramePlane(planes[i].Grid, planes[i].Records, planes[i].Boundaries, planes[i].Data, planes[i].DataStride, planes[i].DataOrigin, planes[i].Dst, planes[i].DstStride, planes[i].DstOrigin, bitDepth, scratch, optimized)
		if err != nil {
			tb.Fatal(err)
		}
		result.PlaneResults[i] = planeResult
		if planes[i].Grid.Type == parser.RestorationNone {
			continue
		}
		result.Planes++
		if err := accumulateRestorationFrameResult(&result, planeResult); err != nil {
			tb.Fatal(err)
		}
	}
	return result
}

func applyRestorationFramePlaneManually(tb testing.TB, grid RestorationPlaneGrid, records []RestorationUnitRecord, boundaries RestorationStripeBoundaries, data []uint16, dataStride int, dataOrigin int, dst []uint16, dstStride int, dstOrigin int, bitDepth uint8, scratch RestorationUnitRecordBoundaryScratch, optimized bool) RestorationPlaneApplyResult {
	tb.Helper()
	if err := ExtendRestorationFrame(data, dataStride, dataOrigin, int(grid.PlaneWidth), int(grid.PlaneHeight), restorationBorder, restorationBorder); err != nil {
		tb.Fatal(err)
	}
	result, err := ApplyRestorationPlaneRecords(grid, records, boundaries, data, dataStride, dataOrigin, dst, dstStride, dstOrigin, bitDepth, scratch, optimized)
	if err != nil {
		tb.Fatal(err)
	}
	if err := copyRestoredPlane(dst, dstStride, dstOrigin, data, dataStride, dataOrigin, int(grid.PlaneWidth), int(grid.PlaneHeight)); err != nil {
		tb.Fatal(err)
	}
	return result
}

func applyRestorationRecordWithBoundariesManually(tb testing.TB, grid RestorationPlaneGrid, record RestorationUnitRecord, boundaries RestorationStripeBoundaries, data []uint16, dataStride int, dataOrigin int, dst []uint16, dstStride int, dstOrigin int, bitDepth uint8, scratch RestorationUnitRecordBoundaryScratch, optimized bool) {
	tb.Helper()
	for stripeIndex := 0; stripeIndex < int(record.StripeCount); stripeIndex++ {
		stripe, ok, err := grid.ProcessingStripe(record.Rect, stripeIndex)
		if err != nil || !ok {
			tb.Fatalf("stripe %d ok=%v err=%v", stripeIndex, ok, err)
		}
		if err := SetupRestorationStripeBoundary(record.Rect, stripe, boundaries, data, dataStride, dataOrigin, scratch.Boundary, optimized); err != nil {
			tb.Fatal(err)
		}
		if _, err := applyRestorationStripeUnits(grid, record, stripe, data, dataStride, dataOrigin, dst, dstStride, dstOrigin, bitDepth, scratch.Unit); err != nil {
			tb.Fatal(err)
		}
		if err := RestoreRestorationStripeBoundary(record.Rect, stripe, data, dataStride, dataOrigin, scratch.Boundary, optimized); err != nil {
			tb.Fatal(err)
		}
	}
}

func fillRestorationRecordSource(dst []uint16, stride int, bitDepth uint8) {
	max := uint16((1 << bitDepth) - 1)
	for i := range dst {
		row := i / stride
		col := i - row*stride
		dst[i] = uint16((row*37 + col*19 + row*col + 11) & int(max))
	}
}

func fillUint16(dst []uint16, value uint16) {
	for i := range dst {
		dst[i] = value
	}
}

func assertSamplesEqual(t *testing.T, got []uint16, want []uint16) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("len=%d want %d", len(got), len(want))
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("sample[%d]=%d want %d", i, got[i], want[i])
		}
	}
}

func samplesEqual(a []uint16, b []uint16) bool {
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

func maxRestorationFrameScratch(a RestorationUnitRecordBoundaryScratchSize, b RestorationUnitRecordBoundaryScratchSize) RestorationUnitRecordBoundaryScratchSize {
	if b.Unit.Wiener > a.Unit.Wiener {
		a.Unit.Wiener = b.Unit.Wiener
	}
	if b.Unit.SGRProj > a.Unit.SGRProj {
		a.Unit.SGRProj = b.Unit.SGRProj
	}
	if b.Boundary.Above > a.Boundary.Above {
		a.Boundary.Above = b.Boundary.Above
	}
	if b.Boundary.Below > a.Boundary.Below {
		a.Boundary.Below = b.Boundary.Below
	}
	return a
}

func cloneRestorationFramePlanes(src []RestorationFramePlane) []RestorationFramePlane {
	dst := make([]RestorationFramePlane, len(src))
	for i := range src {
		dst[i] = src[i]
		dst[i].Data = append([]uint16(nil), src[i].Data...)
		dst[i].Dst = append([]uint16(nil), src[i].Dst...)
	}
	return dst
}

func assertRestorationFramePlanesEqual(t *testing.T, got []RestorationFramePlane, want []RestorationFramePlane) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("planes len=%d want %d", len(got), len(want))
	}
	for i := range got {
		assertSamplesEqual(t, got[i].Data, want[i].Data)
		assertSamplesEqual(t, got[i].Dst, want[i].Dst)
	}
}

func fuzzRestorationApplySample(data []byte, index int, max uint16) uint16 {
	if len(data) == 0 {
		return 0
	}
	lo := uint16(data[index%len(data)])
	hi := uint16(data[(index+1)%len(data)])
	return (lo | hi<<8) & max
}
