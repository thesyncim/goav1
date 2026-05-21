package tile

import (
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/parser"
	av1restoration "github.com/thesyncim/goav1/internal/av1/restoration"
)

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
	for row := 0; row < height; row++ {
		for col := 0; col < width; col++ {
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
	for row := 0; row < height; row++ {
		for col := 0; col < width; col++ {
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
		for row := 0; row < height; row++ {
			for col := 0; col < width; col++ {
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

func makeRestorationApplySource(stride int, height int, bitDepth uint8) []uint16 {
	max := uint16((1 << bitDepth) - 1)
	src := make([]uint16, stride*height)
	for row := 0; row < height; row++ {
		for col := 0; col < stride; col++ {
			src[row*stride+col] = uint16((row*37 + col*19 + row*col) & int(max))
		}
	}
	return src
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
		for unitIndex := 0; unitIndex < unitCount; unitIndex++ {
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

func fuzzRestorationApplySample(data []byte, index int, max uint16) uint16 {
	if len(data) == 0 {
		return 0
	}
	lo := uint16(data[index%len(data)])
	hi := uint16(data[(index+1)%len(data)])
	return (lo | hi<<8) & max
}
