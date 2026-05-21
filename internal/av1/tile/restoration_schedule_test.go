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
		{Index: 0, Col: 0, Row: 0, Unit: RestorationUnit{Type: parser.RestorationNone}},
		{Index: 1, Col: 1, Row: 0, Unit: RestorationUnit{Type: parser.RestorationNone}},
		{Index: 2, Col: 0, Row: 1, Unit: RestorationUnit{Type: parser.RestorationNone}},
		{Index: 3, Col: 1, Row: 1, Unit: RestorationUnit{Type: parser.RestorationNone}},
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
