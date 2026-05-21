package tile

import (
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/parser"
)

func TestRestorationStripeBoundarySetupRestoreFromSavedRows(t *testing.T) {
	grid := testRestorationBoundaryGrid(t, false, false)
	rect, err := grid.UnitRect(1, 1)
	if err != nil {
		t.Fatal(err)
	}
	stripe, ok, err := grid.ProcessingStripe(rect, 0)
	if err != nil || !ok {
		t.Fatalf("stripe ok=%v err=%v", ok, err)
	}
	if !stripe.CopyAbove || !stripe.CopyBelow || stripe.TileStripe != 2 {
		t.Fatalf("stripe flags=%+v", stripe)
	}

	lineWidth, err := restorationStripeBoundaryLineWidth(stripe)
	if err != nil {
		t.Fatal(err)
	}
	stride := int(grid.PlaneWidth) + 32
	origin := restorationBorder*stride + restorationExtraHorz
	rows := int(grid.PlaneHeight) + 2*restorationBorder
	data := makeRestorationBoundarySamples(stride * rows)
	original := append([]uint16(nil), data...)
	boundaries := makeRestorationStripeBoundaries(stride+32, 16)
	sizes, err := RestorationStripeBoundaryScratchLen(stripe, false)
	if err != nil {
		t.Fatal(err)
	}
	scratch := RestorationStripeBoundaryScratch{
		Above: make([]uint16, sizes.Above),
		Below: make([]uint16, sizes.Below),
	}

	if err := SetupRestorationStripeBoundary(rect, stripe, boundaries, data, stride, origin, scratch, false); err != nil {
		t.Fatal(err)
	}
	rsbRow := int(stripe.TileStripe) * restorationCtxVert
	assertBoundaryRows(t, data, stride, origin, stripe, scratch.Above, boundaries.Above, boundaries.Stride, rsbRow, lineWidth, true)
	assertBoundaryRows(t, data, stride, origin, stripe, scratch.Below, boundaries.Below, boundaries.Stride, rsbRow, lineWidth, false)
	if err := RestoreRestorationStripeBoundary(rect, stripe, data, stride, origin, scratch, false); err != nil {
		t.Fatal(err)
	}
	assertSamplesEqual(t, data, original)
}

func TestRestorationStripeBoundaryOptimizedSetupRestore(t *testing.T) {
	grid := testRestorationBoundaryGrid(t, false, false)
	rect, err := grid.UnitRect(1, 1)
	if err != nil {
		t.Fatal(err)
	}
	stripe, ok, err := grid.ProcessingStripe(rect, 0)
	if err != nil || !ok {
		t.Fatalf("stripe ok=%v err=%v", ok, err)
	}
	lineWidth, err := restorationStripeBoundaryLineWidth(stripe)
	if err != nil {
		t.Fatal(err)
	}
	stride := int(grid.PlaneWidth) + 32
	origin := restorationBorder*stride + restorationExtraHorz
	rows := int(grid.PlaneHeight) + 2*restorationBorder
	data := makeRestorationBoundarySamples(stride * rows)
	original := append([]uint16(nil), data...)
	sizes, err := RestorationStripeBoundaryScratchLen(stripe, true)
	if err != nil {
		t.Fatal(err)
	}
	scratch := RestorationStripeBoundaryScratch{
		Above: make([]uint16, sizes.Above),
		Below: make([]uint16, sizes.Below),
	}

	if err := SetupRestorationStripeBoundary(rect, stripe, RestorationStripeBoundaries{}, data, stride, origin, scratch, true); err != nil {
		t.Fatal(err)
	}
	top, ok := restorationStripeDataLine(data, stride, origin, stripe, -restorationBorder, lineWidth)
	if !ok {
		t.Fatal("top line")
	}
	topSource, ok := restorationStripeDataLine(original, stride, origin, stripe, -restorationBorder+1, lineWidth)
	if !ok {
		t.Fatal("top source")
	}
	assertSamplesEqual(t, top, topSource)
	bottom, ok := restorationStripeDataLine(data, stride, origin, stripe, int(stripe.Rect.Height())+2, lineWidth)
	if !ok {
		t.Fatal("bottom line")
	}
	bottomSource, ok := restorationStripeDataLine(original, stride, origin, stripe, int(stripe.Rect.Height())+1, lineWidth)
	if !ok {
		t.Fatal("bottom source")
	}
	assertSamplesEqual(t, bottom, bottomSource)

	if err := RestoreRestorationStripeBoundary(rect, stripe, data, stride, origin, scratch, true); err != nil {
		t.Fatal(err)
	}
	assertSamplesEqual(t, data, original)
}

func TestRestorationStripeBoundaryScratchLenSkipsUnusedSides(t *testing.T) {
	grid := testRestorationBoundaryGrid(t, false, false)
	rect, err := grid.UnitRect(0, 0)
	if err != nil {
		t.Fatal(err)
	}
	stripe, ok, err := grid.ProcessingStripe(rect, 0)
	if err != nil || !ok {
		t.Fatalf("stripe ok=%v err=%v", ok, err)
	}
	if stripe.CopyAbove || !stripe.CopyBelow {
		t.Fatalf("stripe flags=%+v", stripe)
	}
	lineWidth, err := restorationStripeBoundaryLineWidth(stripe)
	if err != nil {
		t.Fatal(err)
	}
	sizes, err := RestorationStripeBoundaryScratchLen(stripe, false)
	if err != nil {
		t.Fatal(err)
	}
	if sizes.Above != 0 || sizes.Below != lineWidth*restorationBorder {
		t.Fatalf("sizes=%+v lineWidth=%d", sizes, lineWidth)
	}
	opt, err := RestorationStripeBoundaryScratchLen(stripe, true)
	if err != nil {
		t.Fatal(err)
	}
	if opt.Above != 0 || opt.Below != lineWidth {
		t.Fatalf("optimized sizes=%+v lineWidth=%d", opt, lineWidth)
	}
}

func TestRestorationStripeBoundaryRejectsInvalidInputs(t *testing.T) {
	grid := testRestorationBoundaryGrid(t, false, false)
	rect, err := grid.UnitRect(1, 1)
	if err != nil {
		t.Fatal(err)
	}
	stripe, ok, err := grid.ProcessingStripe(rect, 0)
	if err != nil || !ok {
		t.Fatalf("stripe ok=%v err=%v", ok, err)
	}
	stride := int(grid.PlaneWidth) + 32
	origin := restorationBorder*stride + restorationExtraHorz
	rows := int(grid.PlaneHeight) + 2*restorationBorder
	data := makeRestorationBoundarySamples(stride * rows)
	boundaries := makeRestorationStripeBoundaries(stride+32, 16)
	sizes, err := RestorationStripeBoundaryScratchLen(stripe, false)
	if err != nil {
		t.Fatal(err)
	}
	scratch := RestorationStripeBoundaryScratch{
		Above: make([]uint16, sizes.Above),
		Below: make([]uint16, sizes.Below),
	}

	badRect := rect
	badRect.X1--
	if err := SetupRestorationStripeBoundary(badRect, stripe, boundaries, data, stride, origin, scratch, false); !errors.Is(err, ErrInvalidPlan) {
		t.Fatalf("bad rect err=%v want %v", err, ErrInvalidPlan)
	}
	shortScratch := scratch
	shortScratch.Above = shortScratch.Above[:len(shortScratch.Above)-1]
	if err := SetupRestorationStripeBoundary(rect, stripe, boundaries, data, stride, origin, shortScratch, false); !errors.Is(err, ErrInvalidPlan) {
		t.Fatalf("short scratch err=%v want %v", err, ErrInvalidPlan)
	}
	shortBoundaries := boundaries
	shortBoundaries.Above = shortBoundaries.Above[:int(stripe.TileStripe)*restorationCtxVert*boundaries.Stride]
	if err := SetupRestorationStripeBoundary(rect, stripe, shortBoundaries, data, stride, origin, scratch, false); !errors.Is(err, ErrInvalidPlan) {
		t.Fatalf("short boundaries err=%v want %v", err, ErrInvalidPlan)
	}
}

func TestRestorationStripeBoundaryAllocs(t *testing.T) {
	grid := testRestorationBoundaryGrid(t, false, false)
	rect, err := grid.UnitRect(1, 1)
	if err != nil {
		t.Fatal(err)
	}
	stripe, ok, err := grid.ProcessingStripe(rect, 0)
	if err != nil || !ok {
		t.Fatalf("stripe ok=%v err=%v", ok, err)
	}
	stride := int(grid.PlaneWidth) + 32
	origin := restorationBorder*stride + restorationExtraHorz
	rows := int(grid.PlaneHeight) + 2*restorationBorder
	data := makeRestorationBoundarySamples(stride * rows)
	boundaries := makeRestorationStripeBoundaries(stride+32, 16)
	sizes, err := RestorationStripeBoundaryScratchLen(stripe, false)
	if err != nil {
		t.Fatal(err)
	}
	scratch := RestorationStripeBoundaryScratch{
		Above: make([]uint16, sizes.Above),
		Below: make([]uint16, sizes.Below),
	}

	allocs := testing.AllocsPerRun(1000, func() {
		if err := SetupRestorationStripeBoundary(rect, stripe, boundaries, data, stride, origin, scratch, false); err != nil {
			t.Fatal(err)
		}
		if err := RestoreRestorationStripeBoundary(rect, stripe, data, stride, origin, scratch, false); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("boundary setup/restore allocated: %f", allocs)
	}
}

func FuzzRestorationStripeBoundary(f *testing.F) {
	f.Add(uint16(300), uint16(260), uint8(1), uint8(0), uint8(1), uint8(1), uint8(0), false, false, false)
	f.Add(uint16(300), uint16(260), uint8(0), uint8(1), uint8(1), uint8(1), uint8(2), true, true, true)
	f.Fuzz(func(t *testing.T, rawW uint16, rawH uint16, rawUnit uint8, rawPlane uint8, rawCol uint8, rawRow uint8, rawStripe uint8, ssX bool, ssY bool, optimized bool) {
		unitSizes := [...]uint16{64, 128, 256}
		unitSize := unitSizes[rawUnit%uint8(len(unitSizes))]
		plane := int(rawPlane % 3)
		params := parser.RestorationParams{
			Type:       [3]parser.RestorationType{parser.RestorationWiener, parser.RestorationWiener, parser.RestorationWiener},
			UnitSizeY:  unitSize,
			UnitSizeUV: unitSize,
		}
		grid, err := BuildRestorationPlaneGrid(params, parser.FrameSize{
			UpscaledWidth:       uint32(rawW%512) + 1,
			Height:              uint32(rawH%512) + 1,
			SuperResDenominator: 8,
		}, parser.ColorConfig{SubsamplingX: ssX, SubsamplingY: ssY}, plane)
		if err != nil {
			t.Fatalf("BuildRestorationPlaneGrid err=%v", err)
		}
		rect, err := grid.UnitRect(uint16(rawCol)%grid.HorzUnits, uint16(rawRow)%grid.VertUnits)
		if err != nil {
			t.Fatalf("UnitRect err=%v", err)
		}
		stripe, ok, err := grid.ProcessingStripe(rect, int(rawStripe%4))
		if err != nil {
			t.Fatalf("ProcessingStripe err=%v", err)
		}
		if !ok {
			return
		}
		stride := int(grid.PlaneWidth) + 2*restorationExtraHorz + 16
		origin := restorationBorder*stride + restorationExtraHorz
		rows := int(grid.PlaneHeight) + 2*restorationBorder + 4
		data := makeRestorationBoundarySamples(stride * rows)
		original := append([]uint16(nil), data...)
		boundaries := makeRestorationStripeBoundaries(stride+32, 64)
		sizes, err := RestorationStripeBoundaryScratchLen(stripe, optimized)
		if err != nil {
			t.Fatalf("ScratchLen err=%v", err)
		}
		scratch := RestorationStripeBoundaryScratch{
			Above: make([]uint16, sizes.Above),
			Below: make([]uint16, sizes.Below),
		}
		if err := SetupRestorationStripeBoundary(rect, stripe, boundaries, data, stride, origin, scratch, optimized); err != nil {
			t.Fatalf("SetupRestorationStripeBoundary err=%v rect=%+v stripe=%+v", err, rect, stripe)
		}
		if err := RestoreRestorationStripeBoundary(rect, stripe, data, stride, origin, scratch, optimized); err != nil {
			t.Fatalf("RestoreRestorationStripeBoundary err=%v rect=%+v stripe=%+v", err, rect, stripe)
		}
		if len(data) != len(original) {
			t.Fatal("length changed")
		}
		for i := range data {
			if data[i] != original[i] {
				t.Fatalf("data[%d]=%d want %d", i, data[i], original[i])
			}
		}
	})
}

func BenchmarkRestorationStripeBoundarySetupRestore(b *testing.B) {
	grid := testRestorationBoundaryGrid(b, false, false)
	rect, err := grid.UnitRect(1, 1)
	if err != nil {
		b.Fatal(err)
	}
	stripe, ok, err := grid.ProcessingStripe(rect, 0)
	if err != nil || !ok {
		b.Fatalf("stripe ok=%v err=%v", ok, err)
	}
	stride := int(grid.PlaneWidth) + 32
	origin := restorationBorder*stride + restorationExtraHorz
	rows := int(grid.PlaneHeight) + 2*restorationBorder
	data := makeRestorationBoundarySamples(stride * rows)
	boundaries := makeRestorationStripeBoundaries(stride+32, 16)
	sizes, err := RestorationStripeBoundaryScratchLen(stripe, false)
	if err != nil {
		b.Fatal(err)
	}
	scratch := RestorationStripeBoundaryScratch{
		Above: make([]uint16, sizes.Above),
		Below: make([]uint16, sizes.Below),
	}
	b.ReportAllocs()
	b.SetBytes(int64((sizes.Above + sizes.Below) * 2))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := SetupRestorationStripeBoundary(rect, stripe, boundaries, data, stride, origin, scratch, false); err != nil {
			b.Fatal(err)
		}
		if err := RestoreRestorationStripeBoundary(rect, stripe, data, stride, origin, scratch, false); err != nil {
			b.Fatal(err)
		}
	}
}

func testRestorationBoundaryGrid(tb testing.TB, ssX bool, ssY bool) RestorationPlaneGrid {
	tb.Helper()
	grid, err := BuildRestorationPlaneGrid(parser.RestorationParams{
		Type:       [3]parser.RestorationType{parser.RestorationWiener, parser.RestorationWiener, parser.RestorationWiener},
		UnitSizeY:  128,
		UnitSizeUV: 64,
	}, parser.FrameSize{UpscaledWidth: 300, Height: 260, SuperResDenominator: 8}, parser.ColorConfig{SubsamplingX: ssX, SubsamplingY: ssY}, boolToPlane(ssX || ssY))
	if err != nil {
		tb.Fatal(err)
	}
	return grid
}

func boolToPlane(chroma bool) int {
	if chroma {
		return 1
	}
	return 0
}

func makeRestorationBoundarySamples(n int) []uint16 {
	samples := make([]uint16, n)
	for i := range samples {
		samples[i] = uint16(7 + i*13)
	}
	return samples
}

func makeRestorationStripeBoundaries(stride int, rows int) RestorationStripeBoundaries {
	above := make([]uint16, stride*rows)
	below := make([]uint16, stride*rows)
	for i := range above {
		above[i] = uint16(0x1000 + i*5)
		below[i] = uint16(0x6000 + i*7)
	}
	return RestorationStripeBoundaries{Above: above, Below: below, Stride: stride}
}

func assertBoundaryRows(t *testing.T, data []uint16, stride int, origin int, stripe RestorationProcessingStripe, scratch []uint16, boundary []uint16, boundaryStride int, rsbRow int, lineWidth int, above bool) {
	t.Helper()
	for i := 0; i < restorationBorder; i++ {
		rowOffset := int(stripe.Rect.Height()) + i
		bufRow := rsbRow + minInt(i, restorationCtxVert-1)
		if above {
			rowOffset = i - restorationBorder
			bufRow = rsbRow + maxInt(rowOffset+restorationCtxVert, 0)
		}
		got, ok := restorationStripeDataLine(data, stride, origin, stripe, rowOffset, lineWidth)
		if !ok {
			t.Fatalf("data line above=%v i=%d", above, i)
		}
		want, ok := restorationBoundaryLine(boundary, boundaryStride, bufRow, int(stripe.Rect.X0), lineWidth)
		if !ok {
			t.Fatalf("boundary line above=%v i=%d", above, i)
		}
		assertSamplesEqual(t, got, want)
		saved := scratch[i*lineWidth : (i+1)*lineWidth]
		if len(saved) != lineWidth {
			t.Fatalf("saved len=%d want %d", len(saved), lineWidth)
		}
	}
}
