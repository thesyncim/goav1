package tile

import (
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/parser"
)

func TestRestorationUnitRectMatchesLibaomLimits(t *testing.T) {
	params := parser.RestorationParams{
		Type:       [3]parser.RestorationType{parser.RestorationWiener, parser.RestorationSGRProj},
		UnitSizeY:  128,
		UnitSizeUV: 64,
	}
	size := parser.FrameSize{UpscaledWidth: 300, Height: 260, SuperResDenominator: 8}
	y, err := BuildRestorationPlaneGrid(params, size, parser.ColorConfig{}, 0)
	if err != nil {
		t.Fatal(err)
	}
	if y.PlaneWidth != 300 || y.PlaneHeight != 260 {
		t.Fatalf("y grid dimensions=%dx%d", y.PlaneWidth, y.PlaneHeight)
	}
	yTests := []struct {
		col  uint16
		row  uint16
		want RestorationUnitRect
	}{
		{col: 0, row: 0, want: RestorationUnitRect{X0: 0, Y0: 0, X1: 128, Y1: 120}},
		{col: 1, row: 0, want: RestorationUnitRect{X0: 128, Y0: 0, X1: 300, Y1: 120}},
		{col: 0, row: 1, want: RestorationUnitRect{X0: 0, Y0: 120, X1: 128, Y1: 260}},
		{col: 1, row: 1, want: RestorationUnitRect{X0: 128, Y0: 120, X1: 300, Y1: 260}},
	}
	for _, tt := range yTests {
		got, err := y.UnitRect(tt.col, tt.row)
		if err != nil {
			t.Fatalf("y col=%d row=%d err=%v", tt.col, tt.row, err)
		}
		if got != tt.want {
			t.Fatalf("y col=%d row=%d rect=%+v want %+v", tt.col, tt.row, got, tt.want)
		}
	}

	uv, err := BuildRestorationPlaneGrid(params, size, parser.ColorConfig{SubsamplingX: true, SubsamplingY: true}, 1)
	if err != nil {
		t.Fatal(err)
	}
	if uv.PlaneWidth != 150 || uv.PlaneHeight != 130 {
		t.Fatalf("uv grid dimensions=%dx%d", uv.PlaneWidth, uv.PlaneHeight)
	}
	uvTests := []struct {
		col  uint16
		row  uint16
		want RestorationUnitRect
	}{
		{col: 0, row: 0, want: RestorationUnitRect{X0: 0, Y0: 0, X1: 64, Y1: 60}},
		{col: 1, row: 0, want: RestorationUnitRect{X0: 64, Y0: 0, X1: 150, Y1: 60}},
		{col: 0, row: 1, want: RestorationUnitRect{X0: 0, Y0: 60, X1: 64, Y1: 130}},
		{col: 1, row: 1, want: RestorationUnitRect{X0: 64, Y0: 60, X1: 150, Y1: 130}},
	}
	for _, tt := range uvTests {
		got, err := uv.UnitRect(tt.col, tt.row)
		if err != nil {
			t.Fatalf("uv col=%d row=%d err=%v", tt.col, tt.row, err)
		}
		if got != tt.want {
			t.Fatalf("uv col=%d row=%d rect=%+v want %+v", tt.col, tt.row, got, tt.want)
		}
	}
}

func TestRestorationProcessingStripesMatchLibaom(t *testing.T) {
	params := parser.RestorationParams{
		Type:      [3]parser.RestorationType{parser.RestorationWiener},
		UnitSizeY: 128,
	}
	grid, err := BuildRestorationPlaneGrid(params, parser.FrameSize{UpscaledWidth: 300, Height: 260, SuperResDenominator: 8}, parser.ColorConfig{}, 0)
	if err != nil {
		t.Fatal(err)
	}

	top, err := grid.UnitRect(0, 0)
	if err != nil {
		t.Fatal(err)
	}
	topWant := []RestorationProcessingStripe{
		{Rect: RestorationUnitRect{X0: 0, Y0: 0, X1: 128, Y1: 56}, ProcUnitWidth: 64, TileStripe: 0, CopyAbove: false, CopyBelow: true},
		{Rect: RestorationUnitRect{X0: 0, Y0: 56, X1: 128, Y1: 120}, ProcUnitWidth: 64, TileStripe: 1, CopyAbove: true, CopyBelow: true},
	}
	assertRestorationStripes(t, grid, top, topWant)

	bottom, err := grid.UnitRect(1, 1)
	if err != nil {
		t.Fatal(err)
	}
	bottomWant := []RestorationProcessingStripe{
		{Rect: RestorationUnitRect{X0: 128, Y0: 120, X1: 300, Y1: 184}, ProcUnitWidth: 64, TileStripe: 2, CopyAbove: true, CopyBelow: true},
		{Rect: RestorationUnitRect{X0: 128, Y0: 184, X1: 300, Y1: 248}, ProcUnitWidth: 64, TileStripe: 3, CopyAbove: true, CopyBelow: true},
		{Rect: RestorationUnitRect{X0: 128, Y0: 248, X1: 300, Y1: 260}, ProcUnitWidth: 64, TileStripe: 4, CopyAbove: true, CopyBelow: false},
	}
	assertRestorationStripes(t, grid, bottom, bottomWant)

	uvParams := parser.RestorationParams{
		Type:       [3]parser.RestorationType{parser.RestorationNone, parser.RestorationWiener},
		UnitSizeUV: 64,
	}
	uv, err := BuildRestorationPlaneGrid(uvParams, parser.FrameSize{UpscaledWidth: 300, Height: 260, SuperResDenominator: 8}, parser.ColorConfig{SubsamplingX: true, SubsamplingY: true}, 1)
	if err != nil {
		t.Fatal(err)
	}
	uvRect, err := uv.UnitRect(0, 0)
	if err != nil {
		t.Fatal(err)
	}
	uvWant := []RestorationProcessingStripe{
		{Rect: RestorationUnitRect{X0: 0, Y0: 0, X1: 64, Y1: 28}, ProcUnitWidth: 32, TileStripe: 0, CopyAbove: false, CopyBelow: true},
		{Rect: RestorationUnitRect{X0: 0, Y0: 28, X1: 64, Y1: 60}, ProcUnitWidth: 32, TileStripe: 1, CopyAbove: true, CopyBelow: true},
	}
	assertRestorationStripes(t, uv, uvRect, uvWant)
}

func TestRestorationGeometryRejectsInvalidInputs(t *testing.T) {
	params := parser.RestorationParams{
		Type:      [3]parser.RestorationType{parser.RestorationWiener},
		UnitSizeY: 128,
	}
	grid, err := BuildRestorationPlaneGrid(params, parser.FrameSize{UpscaledWidth: 300, Height: 260, SuperResDenominator: 8}, parser.ColorConfig{}, 0)
	if err != nil {
		t.Fatal(err)
	}
	rect, err := grid.UnitRect(0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := grid.UnitRect(grid.HorzUnits, 0); !errors.Is(err, ErrInvalidPlan) {
		t.Fatalf("bad col err=%v want %v", err, ErrInvalidPlan)
	}
	if _, _, err := grid.ProcessingStripe(rect, -1); !errors.Is(err, ErrInvalidPlan) {
		t.Fatalf("negative stripe err=%v want %v", err, ErrInvalidPlan)
	}
	if _, _, err := grid.ProcessingStripe(RestorationUnitRect{X0: 0, Y0: 0, X1: grid.PlaneWidth + 1, Y1: 1}, 0); !errors.Is(err, ErrInvalidPlan) {
		t.Fatalf("bad rect err=%v want %v", err, ErrInvalidPlan)
	}

	none := grid
	none.Type = parser.RestorationNone
	if _, err := none.UnitRect(0, 0); !errors.Is(err, ErrInvalidPlan) {
		t.Fatalf("none rect err=%v want %v", err, ErrInvalidPlan)
	}
}

func TestRestorationGeometryFuzzRegressionOffsetExtendsLastUnit(t *testing.T) {
	params := parser.RestorationParams{
		Type:       [3]parser.RestorationType{parser.RestorationWiener, parser.RestorationWiener},
		UnitSizeY:  64,
		UnitSizeUV: 64,
	}
	grid, err := BuildRestorationPlaneGrid(params, parser.FrameSize{
		UpscaledWidth:       261,
		Height:              317,
		SuperResDenominator: 8,
	}, parser.ColorConfig{SubsamplingX: true, SubsamplingY: true}, 1)
	if err != nil {
		t.Fatal(err)
	}
	rect, err := grid.UnitRect(1, 1)
	if err != nil {
		t.Fatal(err)
	}
	want := RestorationUnitRect{X0: 64, Y0: 60, X1: 131, Y1: 159}
	if rect != want {
		t.Fatalf("rect=%+v want %+v", rect, want)
	}
	if rect.Height() != uint32(64+64/2+restorationUnitOffset/2-1) {
		t.Fatalf("height=%d", rect.Height())
	}
}

func TestRestorationGeometryAllocs(t *testing.T) {
	params := parser.RestorationParams{
		Type:      [3]parser.RestorationType{parser.RestorationWiener},
		UnitSizeY: 128,
	}
	grid, err := BuildRestorationPlaneGrid(params, parser.FrameSize{UpscaledWidth: 300, Height: 260, SuperResDenominator: 8}, parser.ColorConfig{}, 0)
	if err != nil {
		t.Fatal(err)
	}
	allocs := testing.AllocsPerRun(1000, func() {
		rect, err := grid.UnitRect(1, 1)
		if err != nil {
			t.Fatal(err)
		}
		count, err := grid.ProcessingStripeCount(rect)
		if err != nil {
			t.Fatal(err)
		}
		if count != 3 {
			t.Fatalf("count=%d", count)
		}
	})
	if allocs != 0 {
		t.Fatalf("restoration geometry allocated: %f", allocs)
	}
}

func FuzzRestorationUnitGeometry(f *testing.F) {
	f.Add(uint16(300), uint16(260), uint8(1), uint8(0), uint8(0), uint8(0), false, false)
	f.Add(uint16(300), uint16(260), uint8(0), uint8(1), uint8(1), uint8(1), true, true)
	f.Add(uint16(260), uint16(316), uint8(0), uint8(1), uint8(1), uint8(1), true, true)
	f.Fuzz(func(t *testing.T, rawW uint16, rawH uint16, rawUnit uint8, rawPlane uint8, rawCol uint8, rawRow uint8, ssX bool, ssY bool) {
		unitSizes := [...]uint16{64, 128, 256}
		unitSize := unitSizes[rawUnit%uint8(len(unitSizes))]
		plane := int(rawPlane % 3)
		params := parser.RestorationParams{
			Type:       [3]parser.RestorationType{parser.RestorationWiener, parser.RestorationWiener, parser.RestorationWiener},
			UnitSizeY:  unitSize,
			UnitSizeUV: unitSize,
		}
		grid, err := BuildRestorationPlaneGrid(params, parser.FrameSize{
			UpscaledWidth:       uint32(rawW) + 1,
			Height:              uint32(rawH) + 1,
			SuperResDenominator: 8,
		}, parser.ColorConfig{SubsamplingX: ssX, SubsamplingY: ssY}, plane)
		if err != nil {
			t.Fatalf("BuildRestorationPlaneGrid err=%v", err)
		}
		col := uint16(rawCol) % grid.HorzUnits
		row := uint16(rawRow) % grid.VertUnits
		rect, err := grid.UnitRect(col, row)
		if err != nil {
			t.Fatalf("UnitRect err=%v grid=%+v col=%d row=%d", err, grid, col, row)
		}
		maxHeight := uint32(unitSize)*3/2 + uint32(restorationUnitOffset>>boolToShift(grid.SubsamplingY))
		if !rect.valid() || rect.X1 > grid.PlaneWidth || rect.Y1 > grid.PlaneHeight ||
			rect.Width() > uint32(unitSize)*3/2 || rect.Height() > maxHeight {
			t.Fatalf("bad rect=%+v grid=%+v", rect, grid)
		}
		count, err := grid.ProcessingStripeCount(rect)
		if err != nil {
			t.Fatalf("ProcessingStripeCount err=%v rect=%+v grid=%+v", err, rect, grid)
		}
		prevY := rect.Y0
		for i := 0; i < count; i++ {
			stripe, ok, err := grid.ProcessingStripe(rect, i)
			if err != nil || !ok {
				t.Fatalf("stripe %d err=%v ok=%v", i, err, ok)
			}
			if stripe.Rect.Y0 != prevY || stripe.Rect.X0 != rect.X0 || stripe.Rect.X1 != rect.X1 ||
				stripe.Rect.Y1 > rect.Y1 || !stripe.Rect.valid() {
				t.Fatalf("bad stripe=%+v rect=%+v", stripe, rect)
			}
			prevY = stripe.Rect.Y1
		}
		if prevY != rect.Y1 {
			t.Fatalf("stripes ended at %d want %d", prevY, rect.Y1)
		}
	})
}

func BenchmarkRestorationUnitGeometry(b *testing.B) {
	params := parser.RestorationParams{
		Type:      [3]parser.RestorationType{parser.RestorationWiener},
		UnitSizeY: 128,
	}
	grid, err := BuildRestorationPlaneGrid(params, parser.FrameSize{UpscaledWidth: 300, Height: 260, SuperResDenominator: 8}, parser.ColorConfig{}, 0)
	if err != nil {
		b.Fatal(err)
	}
	rect, err := grid.UnitRect(1, 1)
	if err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, _, _ = grid.ProcessingStripe(rect, i&3)
	}
}

func assertRestorationStripes(t *testing.T, grid RestorationPlaneGrid, rect RestorationUnitRect, want []RestorationProcessingStripe) {
	t.Helper()
	count, err := grid.ProcessingStripeCount(rect)
	if err != nil {
		t.Fatal(err)
	}
	if count != len(want) {
		t.Fatalf("count=%d want %d", count, len(want))
	}
	for i, w := range want {
		got, ok, err := grid.ProcessingStripe(rect, i)
		if err != nil {
			t.Fatalf("stripe %d err=%v", i, err)
		}
		if !ok || got != w {
			t.Fatalf("stripe %d=%+v ok=%v want %+v", i, got, ok, w)
		}
	}
	got, ok, err := grid.ProcessingStripe(rect, len(want))
	if err != nil {
		t.Fatalf("past-end err=%v", err)
	}
	if ok || got != (RestorationProcessingStripe{}) {
		t.Fatalf("past-end stripe=%+v ok=%v", got, ok)
	}
}
