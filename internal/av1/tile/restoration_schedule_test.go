package tile

import (
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/entropy"
	"github.com/thesyncim/goav1/internal/av1/parser"
)

func TestBuildRestorationPlaneGridMatchesLibaomCounts(t *testing.T) {
	params := parser.RestorationParams{
		Type:       [3]parser.RestorationType{parser.RestorationWiener, parser.RestorationSGRProj, parser.RestorationNone},
		UnitSizeY:  128,
		UnitSizeUV: 64,
	}
	size := parser.FrameSize{UpscaledWidth: 300, Height: 260, SuperResDenominator: 8}
	color := parser.ColorConfig{SubsamplingX: true, SubsamplingY: true}

	y, err := BuildRestorationPlaneGrid(params, size, color, 0)
	if err != nil {
		t.Fatal(err)
	}
	if y.Type != parser.RestorationWiener || y.UnitSize != 128 || y.HorzUnits != 2 || y.VertUnits != 2 {
		t.Fatalf("y grid=%+v", y)
	}
	u, err := BuildRestorationPlaneGrid(params, size, color, 1)
	if err != nil {
		t.Fatal(err)
	}
	if u.Type != parser.RestorationSGRProj || u.UnitSize != 64 || !u.SubsamplingX || !u.SubsamplingY ||
		u.HorzUnits != 2 || u.VertUnits != 2 {
		t.Fatalf("u grid=%+v", u)
	}
	v, err := BuildRestorationPlaneGrid(params, size, color, 2)
	if err != nil {
		t.Fatal(err)
	}
	if v.Type != parser.RestorationNone || v.HorzUnits != 0 || v.VertUnits != 0 {
		t.Fatalf("v grid=%+v", v)
	}
	invalid := params
	invalid.Type[0] = parser.RestorationType(99)
	if _, err := BuildRestorationPlaneGrid(invalid, size, color, 0); !errors.Is(err, ErrInvalidPlan) {
		t.Fatalf("invalid type err=%v want %v", err, ErrInvalidPlan)
	}
}

func TestBuildRestorationFramePlanMatchesLibaomAllocationShape(t *testing.T) {
	params := parser.RestorationParams{
		Type:       [3]parser.RestorationType{parser.RestorationWiener, parser.RestorationSGRProj, parser.RestorationNone},
		UnitSizeY:  128,
		UnitSizeUV: 64,
	}
	size := parser.FrameSize{UpscaledWidth: 300, Height: 260, SuperResDenominator: 8}
	color := parser.ColorConfig{SubsamplingX: true, SubsamplingY: true}

	plan, err := BuildRestorationFramePlan(params, size, color)
	if err != nil {
		t.Fatal(err)
	}
	if !plan.Active || plan.Planes != 3 {
		t.Fatalf("plan active=%v planes=%d", plan.Active, plan.Planes)
	}
	if plan.UnitRecords != ([3]int{4, 4, 0}) || plan.UnitRecordLen() != 8 {
		t.Fatalf("unit records=%+v total=%d", plan.UnitRecords, plan.UnitRecordLen())
	}
	wantY := RestorationStripeBoundaryBufferSize{Stride: 320, Rows: 10, Len: 3200}
	wantU := RestorationStripeBoundaryBufferSize{Stride: 160, Rows: 10, Len: 1600}
	if plan.Boundaries[0] != wantY || plan.Boundaries[1] != wantU || plan.Boundaries[2] != (RestorationStripeBoundaryBufferSize{}) {
		t.Fatalf("boundaries=%+v", plan.Boundaries)
	}
	if plan.BoundaryBufferLen() != wantY.Len+wantU.Len {
		t.Fatalf("boundary total=%d", plan.BoundaryBufferLen())
	}
}

func TestBuildRestorationFramePlanAllNoneIsZeroSize(t *testing.T) {
	params := parser.RestorationParams{
		UnitSizeY:  256,
		UnitSizeUV: 256,
	}
	plan, err := BuildRestorationFramePlan(params, parser.FrameSize{UpscaledWidth: 128, Height: 64, SuperResDenominator: 8}, parser.ColorConfig{})
	if err != nil {
		t.Fatal(err)
	}
	if plan.Active || plan.Planes != 3 || plan.UnitRecordLen() != 0 || plan.BoundaryBufferLen() != 0 {
		t.Fatalf("plan=%+v records=%d boundaries=%d", plan, plan.UnitRecordLen(), plan.BoundaryBufferLen())
	}
	for plane := 0; plane < int(plan.Planes); plane++ {
		if plan.Grids[plane].Type != parser.RestorationNone ||
			plan.UnitRecords[plane] != 0 ||
			plan.Boundaries[plane] != (RestorationStripeBoundaryBufferSize{}) {
			t.Fatalf("plane %d grid=%+v records=%d boundaries=%+v", plane, plan.Grids[plane], plan.UnitRecords[plane], plan.Boundaries[plane])
		}
	}
}

func TestBuildRestorationFramePlanMonochromeSkipsChroma(t *testing.T) {
	params := parser.RestorationParams{
		Type:       [3]parser.RestorationType{parser.RestorationWiener, parser.RestorationSGRProj, parser.RestorationSGRProj},
		UnitSizeY:  64,
		UnitSizeUV: 64,
	}
	plan, err := BuildRestorationFramePlan(params, parser.FrameSize{UpscaledWidth: 128, Height: 128, SuperResDenominator: 8}, parser.ColorConfig{MonoChrome: true})
	if err != nil {
		t.Fatal(err)
	}
	if !plan.Active || plan.Planes != 1 || plan.UnitRecords != ([3]int{4, 0, 0}) || plan.Grids[1] != (RestorationPlaneGrid{}) {
		t.Fatalf("plan=%+v", plan)
	}
}

func TestBindRestorationFramePlanBuffers(t *testing.T) {
	plan := testRestorationFramePlan(t, [3]parser.RestorationType{parser.RestorationWiener, parser.RestorationSGRProj, parser.RestorationNone}, false)
	recordBacking := make([]RestorationUnitRecord, plan.UnitRecordLen())
	records, err := BindRestorationFrameRecordBuffers(plan, recordBacking)
	if err != nil {
		t.Fatal(err)
	}
	if len(records[0]) != 4 || len(records[1]) != 4 || records[2] != nil {
		t.Fatalf("records lens=%d,%d,%d", len(records[0]), len(records[1]), len(records[2]))
	}
	if err := ResetRestorationPlaneRecords(plan.Grids[0], records[0]); err != nil {
		t.Fatal(err)
	}
	if err := ResetRestorationPlaneRecords(plan.Grids[1], records[1]); err != nil {
		t.Fatal(err)
	}

	above := make([]uint16, plan.BoundaryBufferLen())
	below := make([]uint16, plan.BoundaryBufferLen())
	boundaries, err := BindRestorationFrameBoundaryBuffers(plan, above, below)
	if err != nil {
		t.Fatal(err)
	}
	if boundaries[0].Stride != 320 || len(boundaries[0].Above) != 3200 ||
		boundaries[1].Stride != 160 || len(boundaries[1].Below) != 1600 ||
		!restorationStripeBoundariesEmpty(boundaries[2]) {
		t.Fatalf("boundaries=%+v", boundaries)
	}
	boundaries[0].Above[0] = 11
	boundaries[1].Above[0] = 22
	if above[0] != 11 || above[3200] != 22 {
		t.Fatalf("partitioned above=%d,%d", above[0], above[3200])
	}
}

func TestBindRestorationFramePlanBuffersAllNoneIsZeroSize(t *testing.T) {
	plan := testRestorationFramePlan(t, [3]parser.RestorationType{}, false)
	records, err := BindRestorationFrameRecordBuffers(plan, nil)
	if err != nil {
		t.Fatal(err)
	}
	boundaries, err := BindRestorationFrameBoundaryBuffers(plan, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !restorationRecordBuffersEmpty(records) || !restorationFrameBoundariesEmpty(boundaries) {
		t.Fatalf("records=%+v boundaries=%+v", records, boundaries)
	}
	if _, err := BindRestorationFrameRecordBuffers(plan, make([]RestorationUnitRecord, 1)); !errors.Is(err, ErrInvalidPlan) {
		t.Fatalf("extra record backing err=%v want %v", err, ErrInvalidPlan)
	}
	if _, err := BindRestorationFrameBoundaryBuffers(plan, make([]uint16, 1), nil); !errors.Is(err, ErrInvalidPlan) {
		t.Fatalf("extra boundary backing err=%v want %v", err, ErrInvalidPlan)
	}
}

func TestBindRestorationFramePlanBuffersRejectInvalidInputs(t *testing.T) {
	plan := testRestorationFramePlan(t, [3]parser.RestorationType{parser.RestorationWiener, parser.RestorationSGRProj, parser.RestorationNone}, false)
	if _, err := BindRestorationFrameRecordBuffers(plan, make([]RestorationUnitRecord, plan.UnitRecordLen()-1)); !errors.Is(err, ErrJobBufferTooSmall) {
		t.Fatalf("short records err=%v want %v", err, ErrJobBufferTooSmall)
	}
	if _, err := BindRestorationFrameRecordBuffers(plan, make([]RestorationUnitRecord, plan.UnitRecordLen()+1)); !errors.Is(err, ErrInvalidPlan) {
		t.Fatalf("long records err=%v want %v", err, ErrInvalidPlan)
	}
	if _, err := BindRestorationFrameBoundaryBuffers(plan, make([]uint16, plan.BoundaryBufferLen()-1), make([]uint16, plan.BoundaryBufferLen())); !errors.Is(err, ErrJobBufferTooSmall) {
		t.Fatalf("short boundaries err=%v want %v", err, ErrJobBufferTooSmall)
	}
	bad := plan
	bad.UnitRecords[0]++
	if _, err := BindRestorationFrameRecordBuffers(bad, make([]RestorationUnitRecord, bad.UnitRecordLen())); !errors.Is(err, ErrInvalidPlan) {
		t.Fatalf("bad plan err=%v want %v", err, ErrInvalidPlan)
	}
}

func TestRestorationUnitsInSuperblockMatchesLibaomCorners(t *testing.T) {
	params := parser.RestorationParams{
		Type:      [3]parser.RestorationType{parser.RestorationWiener},
		UnitSizeY: 128,
	}
	size := parser.FrameSize{UpscaledWidth: 384, Height: 192, SuperResDenominator: 8}
	grid, err := BuildRestorationPlaneGrid(params, size, parser.ColorConfig{}, 0)
	if err != nil {
		t.Fatal(err)
	}
	tests := []struct {
		name string
		sbX  uint32
		sbY  uint32
		want RestorationUnitRange
		ok   bool
	}{
		{name: "origin", sbX: 0, sbY: 0, want: RestorationUnitRange{Col0: 0, Col1: 1, Row0: 0, Row1: 1}, ok: true},
		{name: "middle-no-corner", sbX: 1, sbY: 0, ok: false},
		{name: "second-unit", sbX: 2, sbY: 0, want: RestorationUnitRange{Col0: 1, Col1: 2, Row0: 0, Row1: 1}, ok: true},
		{name: "bottom-right", sbX: 4, sbY: 2, want: RestorationUnitRange{Col0: 2, Col1: 3, Row0: 1, Row1: 2}, ok: true},
	}
	for _, tt := range tests {
		got, ok, err := grid.UnitsInSuperblock(tt.sbX*16, tt.sbY*16, 16)
		if err != nil {
			t.Fatalf("%s err=%v", tt.name, err)
		}
		if ok != tt.ok || got != tt.want {
			t.Fatalf("%s range=%+v ok=%v want %+v ok=%v", tt.name, got, ok, tt.want, tt.ok)
		}
	}
}

func TestRestorationUnitsInSuperblockSuperresAndChroma(t *testing.T) {
	params := parser.RestorationParams{
		Type:       [3]parser.RestorationType{parser.RestorationNone, parser.RestorationWiener},
		UnitSizeUV: 64,
	}
	size := parser.FrameSize{
		UpscaledWidth:       257,
		Height:              129,
		SuperResEnabled:     true,
		SuperResDenominator: 16,
	}
	grid, err := BuildRestorationPlaneGrid(params, size, parser.ColorConfig{SubsamplingX: true, SubsamplingY: true}, 1)
	if err != nil {
		t.Fatal(err)
	}
	if grid.HorzUnits != 2 || grid.VertUnits != 1 {
		t.Fatalf("grid=%+v", grid)
	}
	got, ok, err := grid.UnitsInSuperblock(0, 0, 16)
	if err != nil {
		t.Fatal(err)
	}
	if !ok || got != (RestorationUnitRange{Col0: 0, Col1: 1, Row0: 0, Row1: 1}) {
		t.Fatalf("range=%+v ok=%v", got, ok)
	}
	got, ok, err = grid.UnitsInSuperblock(16, 0, 16)
	if err != nil {
		t.Fatal(err)
	}
	if !ok || got != (RestorationUnitRange{Col0: 1, Col1: 2, Row0: 0, Row1: 1}) {
		t.Fatalf("second range=%+v ok=%v", got, ok)
	}
}

func TestReadRestorationUnitsForSuperblock(t *testing.T) {
	params := parser.RestorationParams{
		Type:      [3]parser.RestorationType{parser.RestorationWiener},
		UnitSizeY: 64,
	}
	size := parser.FrameSize{UpscaledWidth: 128, Height: 128, SuperResDenominator: 8}
	grid, err := BuildRestorationPlaneGrid(params, size, parser.ColorConfig{}, 0)
	if err != nil {
		t.Fatal(err)
	}
	var state DecodeState
	if err := state.Reset([]byte{0x00}, Job{Offset: 0, Size: 1}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	cdfs := initRestorationCDFs(t)
	refs := DefaultRestorationReferences()
	var records [4]RestorationUnitRecord

	n, err := state.ReadRestorationUnitsForSuperblock(grid, 0, 0, 32, records[:], &refs, cdfs)
	if err != nil {
		t.Fatal(err)
	}
	if n != 4 {
		t.Fatalf("n=%d", n)
	}
	want := [4]RestorationUnitRecord{
		{Index: 0, Col: 0, Row: 0, Rect: RestorationUnitRect{X0: 0, Y0: 0, X1: 64, Y1: 56}, StripeCount: 1, Unit: RestorationUnit{Type: parser.RestorationNone}},
		{Index: 1, Col: 1, Row: 0, Rect: RestorationUnitRect{X0: 64, Y0: 0, X1: 128, Y1: 56}, StripeCount: 1, Unit: RestorationUnit{Type: parser.RestorationNone}},
		{Index: 2, Col: 0, Row: 1, Rect: RestorationUnitRect{X0: 0, Y0: 56, X1: 64, Y1: 128}, StripeCount: 2, Unit: RestorationUnit{Type: parser.RestorationNone}},
		{Index: 3, Col: 1, Row: 1, Rect: RestorationUnitRect{X0: 64, Y0: 56, X1: 128, Y1: 128}, StripeCount: 2, Unit: RestorationUnit{Type: parser.RestorationNone}},
	}
	for i := 0; i < n; i++ {
		if records[i] != want[i] {
			t.Fatalf("record[%d]=%+v want %+v", i, records[i], want[i])
		}
	}
}

func TestReadRestorationUnitsForSuperblockRejectsShortBuffer(t *testing.T) {
	grid := RestorationPlaneGrid{
		Plane:     0,
		Type:      parser.RestorationWiener,
		UnitSize:  64,
		HorzUnits: 2,
		VertUnits: 2,
	}
	var state DecodeState
	if err := state.Reset([]byte{0x00}, Job{Offset: 0, Size: 1}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	_, err := state.ReadRestorationUnitsForSuperblock(grid, 0, 0, 32, nil, &RestorationReferences{}, RestorationCDFs{})
	if !errors.Is(err, ErrJobBufferTooSmall) {
		t.Fatalf("err=%v want %v", err, ErrJobBufferTooSmall)
	}
}

func TestResetRestorationPlaneRecordsMatchesUnitInfoSlots(t *testing.T) {
	params := parser.RestorationParams{
		Type:      [3]parser.RestorationType{parser.RestorationWiener},
		UnitSizeY: 64,
	}
	size := parser.FrameSize{UpscaledWidth: 128, Height: 128, SuperResDenominator: 8}
	grid, err := BuildRestorationPlaneGrid(params, size, parser.ColorConfig{}, 0)
	if err != nil {
		t.Fatal(err)
	}
	records := make([]RestorationUnitRecord, 4)
	if err := ResetRestorationPlaneRecords(grid, records); err != nil {
		t.Fatal(err)
	}
	if err := validateRestorationPlaneRecords(grid, records); err != nil {
		t.Fatal(err)
	}
	want := [4]RestorationUnitRecord{
		{Index: 0, Col: 0, Row: 0, Rect: RestorationUnitRect{X0: 0, Y0: 0, X1: 64, Y1: 56}, StripeCount: 1, Unit: RestorationUnit{Type: parser.RestorationNone}},
		{Index: 1, Col: 1, Row: 0, Rect: RestorationUnitRect{X0: 64, Y0: 0, X1: 128, Y1: 56}, StripeCount: 1, Unit: RestorationUnit{Type: parser.RestorationNone}},
		{Index: 2, Col: 0, Row: 1, Rect: RestorationUnitRect{X0: 0, Y0: 56, X1: 64, Y1: 128}, StripeCount: 2, Unit: RestorationUnit{Type: parser.RestorationNone}},
		{Index: 3, Col: 1, Row: 1, Rect: RestorationUnitRect{X0: 64, Y0: 56, X1: 128, Y1: 128}, StripeCount: 2, Unit: RestorationUnit{Type: parser.RestorationNone}},
	}
	for i := range records {
		if records[i] != want[i] {
			t.Fatalf("record[%d]=%+v want %+v", i, records[i], want[i])
		}
	}
}

func TestStoreRestorationUnitRecordsWritesByIndex(t *testing.T) {
	grid := testRestorationScheduleGrid(t, parser.RestorationWiener, 64, 128, 128)
	records := make([]RestorationUnitRecord, 4)
	if err := ResetRestorationPlaneRecords(grid, records); err != nil {
		t.Fatal(err)
	}
	updates := []RestorationUnitRecord{records[3], records[1]}
	updates[0].Unit.Type = parser.RestorationWiener
	updates[1].Unit.Type = parser.RestorationWiener
	if err := StoreRestorationUnitRecords(grid, records, updates); err != nil {
		t.Fatal(err)
	}
	if records[0].Unit.Type != parser.RestorationNone ||
		records[1].Unit.Type != parser.RestorationWiener ||
		records[2].Unit.Type != parser.RestorationNone ||
		records[3].Unit.Type != parser.RestorationWiener {
		t.Fatalf("records=%+v", records)
	}
	if err := validateRestorationPlaneRecords(grid, records); err != nil {
		t.Fatal(err)
	}
}

func TestReadRestorationUnitsForSuperblockIntoWritesFrameSlots(t *testing.T) {
	grid := testRestorationScheduleGrid(t, parser.RestorationWiener, 128, 384, 192)
	records := make([]RestorationUnitRecord, 6)
	if err := ResetRestorationPlaneRecords(grid, records); err != nil {
		t.Fatal(err)
	}
	records[1].Unit.Type = parser.RestorationWiener
	var state DecodeState
	if err := state.Reset([]byte{0x00}, Job{Offset: 0, Size: 1}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	cdfs := initRestorationCDFs(t)
	refs := DefaultRestorationReferences()

	n, err := state.ReadRestorationUnitsForSuperblockInto(grid, 0, 0, 16, records, &refs, cdfs)
	if err != nil {
		t.Fatal(err)
	}
	if n != 1 {
		t.Fatalf("n=%d", n)
	}
	if records[0].Unit.Type != parser.RestorationNone || records[1].Unit.Type != parser.RestorationWiener {
		t.Fatalf("records[0]=%+v records[1]=%+v", records[0], records[1])
	}
}

func TestRestorationPlaneRecordBuffersRejectInvalidInputs(t *testing.T) {
	grid := testRestorationScheduleGrid(t, parser.RestorationWiener, 64, 128, 128)
	records := make([]RestorationUnitRecord, 4)
	if err := ResetRestorationPlaneRecords(grid, records[:3]); !errors.Is(err, ErrJobBufferTooSmall) {
		t.Fatalf("short reset err=%v want %v", err, ErrJobBufferTooSmall)
	}
	if err := ResetRestorationPlaneRecords(grid, append(records, RestorationUnitRecord{})); !errors.Is(err, ErrInvalidPlan) {
		t.Fatalf("long reset err=%v want %v", err, ErrInvalidPlan)
	}
	if err := ResetRestorationPlaneRecords(grid, records); err != nil {
		t.Fatal(err)
	}
	bad := records[0]
	bad.Index = 99
	if err := StoreRestorationUnitRecords(grid, records, []RestorationUnitRecord{bad}); !errors.Is(err, ErrInvalidPlan) {
		t.Fatalf("bad store err=%v want %v", err, ErrInvalidPlan)
	}
	var state DecodeState
	if err := state.Reset([]byte{0x00}, Job{Offset: 0, Size: 1}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	cdfs := initRestorationCDFs(t)
	refs := DefaultRestorationReferences()
	_, err := state.ReadRestorationUnitsForSuperblockInto(grid, 0, 0, 32, records[:3], &refs, cdfs)
	if !errors.Is(err, ErrJobBufferTooSmall) {
		t.Fatalf("short read-into err=%v want %v", err, ErrJobBufferTooSmall)
	}
}

func TestRestorationScheduleAllocs(t *testing.T) {
	params := parser.RestorationParams{
		Type:      [3]parser.RestorationType{parser.RestorationWiener},
		UnitSizeY: 64,
	}
	size := parser.FrameSize{UpscaledWidth: 128, Height: 128, SuperResDenominator: 8}
	grid, err := BuildRestorationPlaneGrid(params, size, parser.ColorConfig{}, 0)
	if err != nil {
		t.Fatal(err)
	}
	payload := []byte{0x00}
	var state DecodeState
	var switchable entropy.CDF
	var wiener entropy.CDF
	var sgr entropy.CDF
	cdfs := RestorationCDFs{Switchable: &switchable, Wiener: &wiener, SGRProj: &sgr}
	var records [4]RestorationUnitRecord

	allocs := testing.AllocsPerRun(1000, func() {
		if err := state.Reset(payload, Job{Offset: 0, Size: 1}, DecodeOptions{}); err != nil {
			t.Fatal(err)
		}
		if err := InitDefaultRestorationCDFs(cdfs); err != nil {
			t.Fatal(err)
		}
		refs := DefaultRestorationReferences()
		n, err := state.ReadRestorationUnitsForSuperblock(grid, 0, 0, 32, records[:], &refs, cdfs)
		if err != nil {
			t.Fatal(err)
		}
		if n != 4 {
			t.Fatalf("n=%d", n)
		}
	})
	if allocs != 0 {
		t.Fatalf("restoration schedule allocated: %f", allocs)
	}
}

func TestRestorationPlaneRecordBufferAllocs(t *testing.T) {
	grid := testRestorationScheduleGrid(t, parser.RestorationWiener, 64, 128, 128)
	var records [4]RestorationUnitRecord
	payload := []byte{0x00}
	var state DecodeState
	var switchable entropy.CDF
	var wiener entropy.CDF
	var sgr entropy.CDF
	cdfs := RestorationCDFs{Switchable: &switchable, Wiener: &wiener, SGRProj: &sgr}

	allocs := testing.AllocsPerRun(1000, func() {
		if err := ResetRestorationPlaneRecords(grid, records[:]); err != nil {
			t.Fatal(err)
		}
		updates := records[1:3]
		updates[0].Unit.Type = parser.RestorationWiener
		if err := StoreRestorationUnitRecords(grid, records[:], updates); err != nil {
			t.Fatal(err)
		}
		if err := state.Reset(payload, Job{Offset: 0, Size: 1}, DecodeOptions{}); err != nil {
			t.Fatal(err)
		}
		if err := InitDefaultRestorationCDFs(cdfs); err != nil {
			t.Fatal(err)
		}
		refs := DefaultRestorationReferences()
		if _, err := state.ReadRestorationUnitsForSuperblockInto(grid, 0, 0, 32, records[:], &refs, cdfs); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("restoration record buffer allocated: %f", allocs)
	}
}

func TestRestorationFramePlanAllocs(t *testing.T) {
	params := parser.RestorationParams{
		Type:       [3]parser.RestorationType{parser.RestorationWiener, parser.RestorationSGRProj, parser.RestorationNone},
		UnitSizeY:  128,
		UnitSizeUV: 64,
	}
	size := parser.FrameSize{UpscaledWidth: 300, Height: 260, SuperResDenominator: 8}
	color := parser.ColorConfig{SubsamplingX: true, SubsamplingY: true}

	allocs := testing.AllocsPerRun(1000, func() {
		plan, err := BuildRestorationFramePlan(params, size, color)
		if err != nil {
			t.Fatal(err)
		}
		if plan.UnitRecordLen() != 8 || plan.BoundaryBufferLen() != 4800 {
			t.Fatalf("plan=%+v", plan)
		}
	})
	if allocs != 0 {
		t.Fatalf("restoration frame plan allocated: %f", allocs)
	}
}

func TestBindRestorationFramePlanBuffersAllocs(t *testing.T) {
	plan := testRestorationFramePlan(t, [3]parser.RestorationType{parser.RestorationWiener, parser.RestorationSGRProj, parser.RestorationNone}, false)
	recordsBacking := make([]RestorationUnitRecord, plan.UnitRecordLen())
	above := make([]uint16, plan.BoundaryBufferLen())
	below := make([]uint16, plan.BoundaryBufferLen())

	allocs := testing.AllocsPerRun(1000, func() {
		records, err := BindRestorationFrameRecordBuffers(plan, recordsBacking)
		if err != nil {
			t.Fatal(err)
		}
		boundaries, err := BindRestorationFrameBoundaryBuffers(plan, above, below)
		if err != nil {
			t.Fatal(err)
		}
		if len(records[0]) != 4 || len(records[1]) != 4 || boundaries[0].Stride != 320 || boundaries[1].Stride != 160 {
			t.Fatalf("records=%+v boundaries=%+v", records, boundaries)
		}
	})
	if allocs != 0 {
		t.Fatalf("restoration frame plan bind allocated: %f", allocs)
	}
}

func FuzzRestorationUnitSchedule(f *testing.F) {
	f.Add(uint16(256), uint16(128), uint8(64), uint8(0), uint8(0), uint8(16), false, false, false)
	f.Add(uint16(257), uint16(129), uint8(128), uint8(1), uint8(2), uint8(16), true, true, true)
	f.Fuzz(func(t *testing.T, rawW uint16, rawH uint16, rawUnit uint8, rawPlane uint8, rawSBX uint8, rawSBMIB uint8, ssX bool, ssY bool, superres bool) {
		width := uint32(rawW) + 1
		height := uint32(rawH) + 1
		unitSizes := [...]uint16{64, 128, 256}
		unitSize := unitSizes[rawUnit%uint8(len(unitSizes))]
		plane := int(rawPlane % 3)
		params := parser.RestorationParams{Type: [3]parser.RestorationType{parser.RestorationWiener, parser.RestorationWiener, parser.RestorationWiener}}
		params.UnitSizeY = unitSize
		params.UnitSizeUV = unitSize
		size := parser.FrameSize{UpscaledWidth: width, Height: height, SuperResDenominator: 8}
		if superres {
			size.SuperResEnabled = true
			size.SuperResDenominator = 16
		}
		grid, err := BuildRestorationPlaneGrid(params, size, parser.ColorConfig{SubsamplingX: ssX, SubsamplingY: ssY}, plane)
		if err != nil {
			t.Fatalf("BuildRestorationPlaneGrid err=%v", err)
		}
		sbMIBs := [...]uint8{16, 32}
		sbSizeMIB := sbMIBs[rawSBMIB&1]
		maxSBX := uint8(1)
		if grid.HorzUnits > 1 {
			maxSBX = rawSBX
		}
		r, ok, err := grid.UnitsInSuperblock(uint32(maxSBX)*uint32(sbSizeMIB), uint32(rawSBX>>3)*uint32(sbSizeMIB), sbSizeMIB)
		if err != nil {
			t.Fatalf("UnitsInSuperblock err=%v grid=%+v", err, grid)
		}
		if ok {
			if r.Col0 >= r.Col1 || r.Row0 >= r.Row1 || r.Col1 > grid.HorzUnits || r.Row1 > grid.VertUnits {
				t.Fatalf("bad range=%+v grid=%+v", r, grid)
			}
		}
	})
}

func FuzzBuildRestorationFramePlan(f *testing.F) {
	f.Add(uint16(300), uint16(260), uint8(0), uint8(1), uint8(0), false, true, true, false)
	f.Add(uint16(128), uint16(64), uint8(3), uint8(3), uint8(3), false, false, false, false)
	f.Add(uint16(257), uint16(129), uint8(2), uint8(2), uint8(1), true, true, true, true)
	f.Fuzz(func(t *testing.T, rawW uint16, rawH uint16, rawY uint8, rawU uint8, rawV uint8, mono bool, ssX bool, ssY bool, superres bool) {
		types := [...]parser.RestorationType{
			parser.RestorationNone,
			parser.RestorationSwitchable,
			parser.RestorationWiener,
			parser.RestorationSGRProj,
		}
		unitSizes := [...]uint16{64, 128, 256}
		width := uint32(rawW) + 1
		height := uint32(rawH) + 1
		params := parser.RestorationParams{
			Type:       [3]parser.RestorationType{types[rawY&3], types[rawU&3], types[rawV&3]},
			UnitSizeY:  unitSizes[rawY%uint8(len(unitSizes))],
			UnitSizeUV: unitSizes[(rawU^rawV)%uint8(len(unitSizes))],
		}
		size := parser.FrameSize{UpscaledWidth: width, Height: height, SuperResDenominator: 8}
		if superres {
			size.SuperResEnabled = true
			size.SuperResDenominator = 16
		}
		color := parser.ColorConfig{MonoChrome: mono, SubsamplingX: ssX, SubsamplingY: ssY}

		plan, err := BuildRestorationFramePlan(params, size, color)
		if err != nil {
			t.Fatalf("BuildRestorationFramePlan err=%v", err)
		}
		wantPlanes := uint8(3)
		if mono {
			wantPlanes = 1
		}
		if plan.Planes != wantPlanes {
			t.Fatalf("planes=%d want %d", plan.Planes, wantPlanes)
		}
		active := false
		totalRecords := 0
		totalBoundaries := 0
		for plane := 0; plane < int(plan.Planes); plane++ {
			grid := plan.Grids[plane]
			if grid.Plane != uint8(plane) {
				t.Fatalf("grid plane=%d want %d", grid.Plane, plane)
			}
			if grid.Type == parser.RestorationNone {
				if plan.UnitRecords[plane] != 0 || plan.Boundaries[plane] != (RestorationStripeBoundaryBufferSize{}) {
					t.Fatalf("inactive plane=%d records=%d boundaries=%+v", plane, plan.UnitRecords[plane], plan.Boundaries[plane])
				}
				continue
			}
			active = true
			wantRecords := int(grid.HorzUnits) * int(grid.VertUnits)
			if wantRecords == 0 || plan.UnitRecords[plane] != wantRecords {
				t.Fatalf("plane=%d records=%d want %d grid=%+v", plane, plan.UnitRecords[plane], wantRecords, grid)
			}
			if plan.Boundaries[plane].Stride%32 != 0 || plan.Boundaries[plane].Rows == 0 || plan.Boundaries[plane].Len == 0 {
				t.Fatalf("plane=%d boundary=%+v", plane, plan.Boundaries[plane])
			}
			totalRecords += plan.UnitRecords[plane]
			totalBoundaries += plan.Boundaries[plane].Len
		}
		if plan.Active != active || plan.UnitRecordLen() != totalRecords || plan.BoundaryBufferLen() != totalBoundaries {
			t.Fatalf("plan=%+v active=%v records=%d boundaries=%d", plan, active, totalRecords, totalBoundaries)
		}
	})
}

func FuzzBindRestorationFramePlanBuffers(f *testing.F) {
	f.Add(uint16(300), uint16(260), uint8(2), uint8(3), uint8(0), false, true, true, false)
	f.Add(uint16(128), uint16(64), uint8(0), uint8(0), uint8(0), false, false, false, false)
	f.Add(uint16(257), uint16(129), uint8(1), uint8(2), uint8(1), true, true, true, true)
	f.Fuzz(func(t *testing.T, rawW uint16, rawH uint16, rawY uint8, rawU uint8, rawV uint8, mono bool, ssX bool, ssY bool, superres bool) {
		types := [...]parser.RestorationType{
			parser.RestorationNone,
			parser.RestorationSwitchable,
			parser.RestorationWiener,
			parser.RestorationSGRProj,
		}
		unitSizes := [...]uint16{64, 128, 256}
		params := parser.RestorationParams{
			Type:       [3]parser.RestorationType{types[rawY&3], types[rawU&3], types[rawV&3]},
			UnitSizeY:  unitSizes[rawY%uint8(len(unitSizes))],
			UnitSizeUV: unitSizes[(rawU^rawV)%uint8(len(unitSizes))],
		}
		size := parser.FrameSize{UpscaledWidth: uint32(rawW%512) + 1, Height: uint32(rawH%512) + 1, SuperResDenominator: 8}
		if superres {
			size.SuperResEnabled = true
			size.SuperResDenominator = 16
		}
		plan, err := BuildRestorationFramePlan(params, size, parser.ColorConfig{MonoChrome: mono, SubsamplingX: ssX, SubsamplingY: ssY})
		if err != nil {
			t.Fatalf("BuildRestorationFramePlan err=%v", err)
		}
		recordBacking := make([]RestorationUnitRecord, plan.UnitRecordLen())
		records, err := BindRestorationFrameRecordBuffers(plan, recordBacking)
		if err != nil {
			t.Fatalf("BindRestorationFrameRecordBuffers err=%v", err)
		}
		above := make([]uint16, plan.BoundaryBufferLen())
		below := make([]uint16, plan.BoundaryBufferLen())
		boundaries, err := BindRestorationFrameBoundaryBuffers(plan, above, below)
		if err != nil {
			t.Fatalf("BindRestorationFrameBoundaryBuffers err=%v", err)
		}
		totalRecords := 0
		totalBoundaries := 0
		for plane := 0; plane < int(plan.Planes); plane++ {
			if len(records[plane]) != plan.UnitRecords[plane] {
				t.Fatalf("plane=%d records=%d want %d", plane, len(records[plane]), plan.UnitRecords[plane])
			}
			if plan.Boundaries[plane].Len == 0 {
				if !restorationStripeBoundariesEmpty(boundaries[plane]) {
					t.Fatalf("plane=%d boundaries=%+v", plane, boundaries[plane])
				}
			} else if len(boundaries[plane].Above) != plan.Boundaries[plane].Len ||
				len(boundaries[plane].Below) != plan.Boundaries[plane].Len ||
				boundaries[plane].Stride != plan.Boundaries[plane].Stride {
				t.Fatalf("plane=%d boundaries=%+v want %+v", plane, boundaries[plane], plan.Boundaries[plane])
			}
			totalRecords += len(records[plane])
			totalBoundaries += len(boundaries[plane].Above)
		}
		if totalRecords != len(recordBacking) || totalBoundaries != len(above) || totalBoundaries != len(below) {
			t.Fatalf("totals records=%d/%d boundaries=%d/%d/%d", totalRecords, len(recordBacking), totalBoundaries, len(above), len(below))
		}
	})
}

func FuzzRestorationPlaneRecordBuffer(f *testing.F) {
	f.Add(uint16(128), uint16(128), uint8(0), uint8(0), uint8(1), uint8(4))
	f.Add(uint16(384), uint16(192), uint8(1), uint8(2), uint8(3), uint8(2))
	f.Fuzz(func(t *testing.T, rawW uint16, rawH uint16, rawType uint8, rawUnit uint8, rawIndex uint8, rawCount uint8) {
		types := [...]parser.RestorationType{
			parser.RestorationSwitchable,
			parser.RestorationWiener,
			parser.RestorationSGRProj,
		}
		unitSizes := [...]uint16{64, 128, 256}
		grid := testRestorationScheduleGrid(t, types[rawType%uint8(len(types))], unitSizes[rawUnit%uint8(len(unitSizes))], uint32(rawW%512)+1, uint32(rawH%512)+1)
		need, err := grid.UnitRecordLen()
		if err != nil {
			t.Fatalf("UnitRecordLen err=%v", err)
		}
		records := make([]RestorationUnitRecord, need)
		if err := ResetRestorationPlaneRecords(grid, records); err != nil {
			t.Fatalf("ResetRestorationPlaneRecords err=%v", err)
		}
		updates := make([]RestorationUnitRecord, 0, int(rawCount%5))
		for i := 0; i < int(rawCount%5); i++ {
			index := (int(rawIndex) + i*7) % len(records)
			record := records[index]
			record.Unit.Type = fuzzRestorationUnitType(grid.Type, rawType+uint8(i))
			updates = append(updates, record)
		}
		if err := StoreRestorationUnitRecords(grid, records, updates); err != nil {
			t.Fatalf("StoreRestorationUnitRecords err=%v", err)
		}
		if err := validateRestorationPlaneRecords(grid, records); err != nil {
			t.Fatalf("validateRestorationPlaneRecords err=%v", err)
		}
	})
}

func BenchmarkReadRestorationUnitsForSuperblock(b *testing.B) {
	params := parser.RestorationParams{
		Type:      [3]parser.RestorationType{parser.RestorationWiener},
		UnitSizeY: 64,
	}
	size := parser.FrameSize{UpscaledWidth: 128, Height: 128, SuperResDenominator: 8}
	grid, err := BuildRestorationPlaneGrid(params, size, parser.ColorConfig{}, 0)
	if err != nil {
		b.Fatal(err)
	}
	payload := []byte{0x00}
	var state DecodeState
	var switchable entropy.CDF
	var wiener entropy.CDF
	var sgr entropy.CDF
	cdfs := RestorationCDFs{Switchable: &switchable, Wiener: &wiener, SGRProj: &sgr}
	var records [4]RestorationUnitRecord

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if err := state.Reset(payload, Job{Offset: 0, Size: 1}, DecodeOptions{}); err != nil {
			b.Fatal(err)
		}
		if err := InitDefaultRestorationCDFs(cdfs); err != nil {
			b.Fatal(err)
		}
		refs := DefaultRestorationReferences()
		_, _ = state.ReadRestorationUnitsForSuperblock(grid, 0, 0, 32, records[:], &refs, cdfs)
	}
}

func testRestorationScheduleGrid(tb testing.TB, typ parser.RestorationType, unitSize uint16, width uint32, height uint32) RestorationPlaneGrid {
	tb.Helper()
	params := parser.RestorationParams{
		Type:      [3]parser.RestorationType{typ},
		UnitSizeY: unitSize,
	}
	grid, err := BuildRestorationPlaneGrid(params, parser.FrameSize{UpscaledWidth: width, Height: height, SuperResDenominator: 8}, parser.ColorConfig{}, 0)
	if err != nil {
		tb.Fatal(err)
	}
	return grid
}

func testRestorationFramePlan(tb testing.TB, types [3]parser.RestorationType, mono bool) RestorationFramePlan {
	tb.Helper()
	params := parser.RestorationParams{
		Type:       types,
		UnitSizeY:  128,
		UnitSizeUV: 64,
	}
	plan, err := BuildRestorationFramePlan(params, parser.FrameSize{UpscaledWidth: 300, Height: 260, SuperResDenominator: 8}, parser.ColorConfig{MonoChrome: mono, SubsamplingX: true, SubsamplingY: true})
	if err != nil {
		tb.Fatal(err)
	}
	return plan
}

func restorationRecordBuffersEmpty(records [3][]RestorationUnitRecord) bool {
	for i := range records {
		if records[i] != nil {
			return false
		}
	}
	return true
}

func restorationFrameBoundariesEmpty(boundaries [3]RestorationStripeBoundaries) bool {
	for i := range boundaries {
		if !restorationStripeBoundariesEmpty(boundaries[i]) {
			return false
		}
	}
	return true
}

func restorationStripeBoundariesEmpty(boundaries RestorationStripeBoundaries) bool {
	return boundaries.Above == nil && boundaries.Below == nil && boundaries.Stride == 0
}

func fuzzRestorationUnitType(frameType parser.RestorationType, raw uint8) parser.RestorationType {
	switch frameType {
	case parser.RestorationSwitchable:
		types := [...]parser.RestorationType{parser.RestorationNone, parser.RestorationWiener, parser.RestorationSGRProj}
		return types[raw%uint8(len(types))]
	case parser.RestorationWiener:
		if raw&1 == 0 {
			return parser.RestorationNone
		}
		return parser.RestorationWiener
	case parser.RestorationSGRProj:
		if raw&1 == 0 {
			return parser.RestorationNone
		}
		return parser.RestorationSGRProj
	default:
		return parser.RestorationNone
	}
}
