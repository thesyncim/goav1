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

func TestRestorationStripeBoundaryBufferLen(t *testing.T) {
	grid := testRestorationBoundaryGrid(t, false, false)
	size, err := RestorationStripeBoundaryBufferLen(grid)
	if err != nil {
		t.Fatal(err)
	}
	if size != (RestorationStripeBoundaryBufferSize{Stride: 320, Rows: 10, Len: 3200}) {
		t.Fatalf("size=%+v", size)
	}
	uv := testRestorationBoundaryGrid(t, true, true)
	size, err = RestorationStripeBoundaryBufferLen(uv)
	if err != nil {
		t.Fatal(err)
	}
	if size != (RestorationStripeBoundaryBufferSize{Stride: 160, Rows: 10, Len: 1600}) {
		t.Fatalf("uv size=%+v", size)
	}
}

func TestExtendRestorationFrame(t *testing.T) {
	const width, height = 5, 4
	const borderHorz, borderVert = 3, 2
	const stride = width + 2*borderHorz + 5
	origin := borderVert*stride + borderHorz
	data := make([]uint16, stride*(height+2*borderVert))
	fillUint16(data, 0xeeee)
	for y := range height {
		for x := range width {
			data[origin+y*stride+x] = uint16(20 + y*11 + x)
		}
	}

	if err := ExtendRestorationFrame(data, stride, origin, width, height, borderHorz, borderVert); err != nil {
		t.Fatal(err)
	}
	for y := -borderVert; y < height+borderVert; y++ {
		clampedY := minInt(maxInt(y, 0), height-1)
		for x := -borderHorz; x < width+borderHorz; x++ {
			clampedX := minInt(maxInt(x, 0), width-1)
			offset, ok := restorationSignedPlaneOffset(origin, stride, x, y)
			if !ok {
				t.Fatalf("offset x=%d y=%d", x, y)
			}
			want := uint16(20 + clampedY*11 + clampedX)
			if data[offset] != want {
				t.Fatalf("sample x=%d y=%d got=%d want %d", x, y, data[offset], want)
			}
		}
	}
}

func TestExtendRestorationFrameRejectsInvalidInputs(t *testing.T) {
	const width, height = 5, 4
	const borderHorz, borderVert = 3, 2
	const stride = width + 2*borderHorz
	origin := borderVert*stride + borderHorz
	data := make([]uint16, stride*(height+2*borderVert))
	if err := ExtendRestorationFrame(data[:len(data)-1], stride, origin, width, height, borderHorz, borderVert); !errors.Is(err, ErrInvalidPlan) {
		t.Fatalf("short data err=%v want %v", err, ErrInvalidPlan)
	}
	if err := ExtendRestorationFrame(data, stride, -1, width, height, borderHorz, borderVert); !errors.Is(err, ErrInvalidPlan) {
		t.Fatalf("bad origin err=%v want %v", err, ErrInvalidPlan)
	}
	if err := ExtendRestorationFrame(data, stride, origin, width, height, -1, borderVert); !errors.Is(err, ErrInvalidPlan) {
		t.Fatalf("bad border err=%v want %v", err, ErrInvalidPlan)
	}
}

func TestSaveRestorationBoundaryLinesDeblock(t *testing.T) {
	grid := testRestorationBoundaryGrid(t, false, false)
	srcStride := int(grid.PlaneWidth) + 7
	src := makeRestorationBoundaryPlane(grid, srcStride)
	size, err := RestorationStripeBoundaryBufferLen(grid)
	if err != nil {
		t.Fatal(err)
	}
	boundaries := makeSentinelRestorationBoundaries(size, 0xeeee)

	if err := SaveRestorationBoundaryLines(grid, src, srcStride, 0, boundaries, false); err != nil {
		t.Fatal(err)
	}
	assertSavedBoundaryRow(t, grid, src, srcStride, boundaries.Below, boundaries.Stride, 0, 0, 56)
	assertSavedBoundaryRow(t, grid, src, srcStride, boundaries.Below, boundaries.Stride, 0, 1, 57)
	assertSavedBoundaryRow(t, grid, src, srcStride, boundaries.Above, boundaries.Stride, 1, 0, 54)
	assertSavedBoundaryRow(t, grid, src, srcStride, boundaries.Above, boundaries.Stride, 1, 1, 55)
	assertSavedBoundaryRow(t, grid, src, srcStride, boundaries.Above, boundaries.Stride, 4, 0, 246)
	assertSavedBoundaryRow(t, grid, src, srcStride, boundaries.Above, boundaries.Stride, 4, 1, 247)
	assertBoundaryUntouched(t, boundaries.Above, boundaries.Stride, 0, 0, int(grid.PlaneWidth), 0xeeee)
	assertBoundaryUntouched(t, boundaries.Below, boundaries.Stride, 4, 0, int(grid.PlaneWidth), 0xeeee)
}

func TestSaveRestorationBoundaryLinesCDEF(t *testing.T) {
	grid := testRestorationBoundaryGrid(t, false, false)
	srcStride := int(grid.PlaneWidth) + 7
	src := makeRestorationBoundaryPlane(grid, srcStride)
	size, err := RestorationStripeBoundaryBufferLen(grid)
	if err != nil {
		t.Fatal(err)
	}
	boundaries := makeSentinelRestorationBoundaries(size, 0xeeee)

	if err := SaveRestorationBoundaryLines(grid, src, srcStride, 0, boundaries, true); err != nil {
		t.Fatal(err)
	}
	assertSavedBoundaryRow(t, grid, src, srcStride, boundaries.Above, boundaries.Stride, 0, 0, 0)
	assertSavedBoundaryRow(t, grid, src, srcStride, boundaries.Above, boundaries.Stride, 0, 1, 0)
	assertSavedBoundaryRow(t, grid, src, srcStride, boundaries.Below, boundaries.Stride, 4, 0, 259)
	assertSavedBoundaryRow(t, grid, src, srcStride, boundaries.Below, boundaries.Stride, 4, 1, 259)
	assertBoundaryUntouched(t, boundaries.Above, boundaries.Stride, 1, 0, int(grid.PlaneWidth), 0xeeee)
	assertBoundaryUntouched(t, boundaries.Below, boundaries.Stride, 0, 0, int(grid.PlaneWidth), 0xeeee)
}

func TestSaveRestorationBoundaryLinesDuplicatesSingleDeblockRow(t *testing.T) {
	grid, err := BuildRestorationPlaneGrid(parser.RestorationParams{
		Type:      [3]parser.RestorationType{parser.RestorationWiener},
		UnitSizeY: 64,
	}, parser.FrameSize{UpscaledWidth: 64, Height: 57, SuperResDenominator: 8}, parser.ColorConfig{}, 0)
	if err != nil {
		t.Fatal(err)
	}
	srcStride := int(grid.PlaneWidth)
	src := makeRestorationBoundaryPlane(grid, srcStride)
	size, err := RestorationStripeBoundaryBufferLen(grid)
	if err != nil {
		t.Fatal(err)
	}
	boundaries := makeSentinelRestorationBoundaries(size, 0xeeee)

	if err := SaveRestorationBoundaryLines(grid, src, srcStride, 0, boundaries, false); err != nil {
		t.Fatal(err)
	}
	assertSavedBoundaryRow(t, grid, src, srcStride, boundaries.Below, boundaries.Stride, 0, 0, 56)
	assertSavedBoundaryRow(t, grid, src, srcStride, boundaries.Below, boundaries.Stride, 0, 1, 56)
}

func TestSaveRestorationFrameBoundaryLinesMatchesPlaneLoop(t *testing.T) {
	planes := makeRestorationFrameBoundaryPlanes(t, [3]parser.RestorationType{parser.RestorationWiener, parser.RestorationSGRProj, parser.RestorationNone}, 10, false)
	manual := cloneRestorationFrameBoundaryPlanes(planes)

	if err := SaveRestorationFrameBoundaryLines(planes, false); err != nil {
		t.Fatal(err)
	}
	saveRestorationFrameBoundaryLinesManually(t, manual, false)
	assertRestorationFrameBoundaryPlanesEqual(t, planes, manual)

	if err := SaveRestorationFrameBoundaryLines(planes, true); err != nil {
		t.Fatal(err)
	}
	saveRestorationFrameBoundaryLinesManually(t, manual, true)
	assertRestorationFrameBoundaryPlanesEqual(t, planes, manual)
}

func TestSaveRestorationFrameBoundaryLinesMonochrome(t *testing.T) {
	planes := makeRestorationFrameBoundaryPlanes(t, [3]parser.RestorationType{parser.RestorationWiener, parser.RestorationNone, parser.RestorationNone}, 8, true)
	manual := cloneRestorationFrameBoundaryPlanes(planes)
	if err := SaveRestorationFrameBoundaryLines(planes, false); err != nil {
		t.Fatal(err)
	}
	saveRestorationFrameBoundaryLinesManually(t, manual, false)
	assertRestorationFrameBoundaryPlanesEqual(t, planes, manual)
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

func TestSaveRestorationBoundaryLinesRejectsInvalidInputs(t *testing.T) {
	grid := testRestorationBoundaryGrid(t, false, false)
	srcStride := int(grid.PlaneWidth)
	src := makeRestorationBoundaryPlane(grid, srcStride)
	size, err := RestorationStripeBoundaryBufferLen(grid)
	if err != nil {
		t.Fatal(err)
	}
	boundaries := makeSentinelRestorationBoundaries(size, 0xeeee)
	short := boundaries
	short.Above = short.Above[:len(short.Above)-1]
	if err := SaveRestorationBoundaryLines(grid, src, srcStride, 0, short, false); !errors.Is(err, ErrInvalidPlan) {
		t.Fatalf("short boundary err=%v want %v", err, ErrInvalidPlan)
	}
	if err := SaveRestorationBoundaryLines(grid, src[:len(src)-1], srcStride, 0, boundaries, true); !errors.Is(err, ErrInvalidPlan) {
		t.Fatalf("short src err=%v want %v", err, ErrInvalidPlan)
	}
	if err := SaveRestorationBoundaryLines(grid, src, srcStride, -1, boundaries, false); !errors.Is(err, ErrInvalidPlan) {
		t.Fatalf("bad origin err=%v want %v", err, ErrInvalidPlan)
	}
}

func TestSaveRestorationFrameBoundaryLinesRejectsInvalidInputs(t *testing.T) {
	planes := makeRestorationFrameBoundaryPlanes(t, [3]parser.RestorationType{parser.RestorationWiener, parser.RestorationSGRProj, parser.RestorationNone}, 8, false)
	if err := SaveRestorationFrameBoundaryLines(planes[:2], false); !errors.Is(err, ErrInvalidPlan) {
		t.Fatalf("two-plane frame err=%v want %v", err, ErrInvalidPlan)
	}
	bad := append([]RestorationFrameBoundaryPlane(nil), planes...)
	bad[1].Grid.Plane = 2
	if err := SaveRestorationFrameBoundaryLines(bad, false); !errors.Is(err, ErrInvalidPlan) {
		t.Fatalf("bad plane order err=%v want %v", err, ErrInvalidPlan)
	}
	short := append([]RestorationFrameBoundaryPlane(nil), planes...)
	short[0].Boundaries.Above = short[0].Boundaries.Above[:len(short[0].Boundaries.Above)-1]
	if err := SaveRestorationFrameBoundaryLines(short, false); !errors.Is(err, ErrInvalidPlan) {
		t.Fatalf("short boundary err=%v want %v", err, ErrInvalidPlan)
	}
	disabled := []RestorationFrameBoundaryPlane{{Grid: RestorationPlaneGrid{Plane: 0, Type: parser.RestorationNone}}}
	if err := SaveRestorationFrameBoundaryLines(disabled, false); err != nil {
		t.Fatalf("disabled frame boundary err=%v", err)
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

func TestSaveRestorationBoundaryLinesAllocs(t *testing.T) {
	grid := testRestorationBoundaryGrid(t, false, false)
	srcStride := int(grid.PlaneWidth)
	src := makeRestorationBoundaryPlane(grid, srcStride)
	size, err := RestorationStripeBoundaryBufferLen(grid)
	if err != nil {
		t.Fatal(err)
	}
	boundaries := makeSentinelRestorationBoundaries(size, 0)

	allocs := testing.AllocsPerRun(1000, func() {
		if err := SaveRestorationBoundaryLines(grid, src, srcStride, 0, boundaries, false); err != nil {
			t.Fatal(err)
		}
		if err := SaveRestorationBoundaryLines(grid, src, srcStride, 0, boundaries, true); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("SaveRestorationBoundaryLines allocated: %f", allocs)
	}
}

func TestSaveRestorationFrameBoundaryLinesAllocs(t *testing.T) {
	planes := makeRestorationFrameBoundaryPlanes(t, [3]parser.RestorationType{parser.RestorationWiener, parser.RestorationSGRProj, parser.RestorationNone}, 8, false)
	allocs := testing.AllocsPerRun(1000, func() {
		if err := SaveRestorationFrameBoundaryLines(planes, false); err != nil {
			t.Fatal(err)
		}
		if err := SaveRestorationFrameBoundaryLines(planes, true); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("SaveRestorationFrameBoundaryLines allocated: %f", allocs)
	}
}

func TestExtendRestorationFrameAllocs(t *testing.T) {
	const width, height = 64, 57
	const borderHorz, borderVert = 6, 3
	const stride = width + 2*borderHorz
	origin := borderVert*stride + borderHorz
	data := makeRestorationBoundarySamples(stride * (height + 2*borderVert))

	allocs := testing.AllocsPerRun(1000, func() {
		if err := ExtendRestorationFrame(data, stride, origin, width, height, borderHorz, borderVert); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("ExtendRestorationFrame allocated: %f", allocs)
	}
}

func FuzzExtendRestorationFrame(f *testing.F) {
	f.Add(uint16(5), uint16(4), uint8(3), uint8(2), uint8(0), []byte{0, 1, 2, 3, 255})
	f.Add(uint16(64), uint16(57), uint8(6), uint8(3), uint8(2), []byte{255, 7, 11})
	f.Fuzz(func(t *testing.T, rawW uint16, rawH uint16, rawBH uint8, rawBV uint8, rawDepth uint8, dataBytes []byte) {
		width := int(rawW%256) + 1
		height := int(rawH%256) + 1
		borderHorz := int(rawBH % 16)
		borderVert := int(rawBV % 8)
		bitDepths := [...]uint8{8, 10, 12}
		bitDepth := bitDepths[rawDepth%uint8(len(bitDepths))]
		max := uint16((1 << bitDepth) - 1)
		stride := width + 2*borderHorz + int(rawBH%5)
		origin := borderVert*stride + borderHorz
		data := make([]uint16, stride*(height+2*borderVert))
		for y := range height {
			for x := range width {
				data[origin+y*stride+x] = fuzzRestorationApplySample(dataBytes, y*width+x, max)
			}
		}
		if err := ExtendRestorationFrame(data, stride, origin, width, height, borderHorz, borderVert); err != nil {
			t.Fatalf("ExtendRestorationFrame err=%v", err)
		}
		for y := -borderVert; y < height+borderVert; y++ {
			clampedY := minInt(maxInt(y, 0), height-1)
			for x := -borderHorz; x < width+borderHorz; x++ {
				clampedX := minInt(maxInt(x, 0), width-1)
				offset, ok := restorationSignedPlaneOffset(origin, stride, x, y)
				if !ok {
					t.Fatalf("offset x=%d y=%d", x, y)
				}
				want := fuzzRestorationApplySample(dataBytes, clampedY*width+clampedX, max)
				if data[offset] != want {
					t.Fatalf("sample x=%d y=%d got=%d want %d", x, y, data[offset], want)
				}
			}
		}
	})
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

func FuzzSaveRestorationBoundaryLines(f *testing.F) {
	f.Add(uint16(300), uint16(260), uint8(1), uint8(0), false, false, false)
	f.Add(uint16(64), uint16(57), uint8(0), uint8(1), true, true, true)
	f.Fuzz(func(t *testing.T, rawW uint16, rawH uint16, rawUnit uint8, rawPlane uint8, ssX bool, ssY bool, afterCDEF bool) {
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
		srcStride := int(grid.PlaneWidth) + int(rawUnit%7)
		src := makeRestorationBoundaryPlane(grid, srcStride)
		size, err := RestorationStripeBoundaryBufferLen(grid)
		if err != nil {
			t.Fatalf("RestorationStripeBoundaryBufferLen err=%v", err)
		}
		boundaries := makeSentinelRestorationBoundaries(size, 0xeeee)
		if err := SaveRestorationBoundaryLines(grid, src, srcStride, 0, boundaries, afterCDEF); err != nil {
			t.Fatalf("SaveRestorationBoundaryLines err=%v", err)
		}
		if len(boundaries.Above) < size.Len || len(boundaries.Below) < size.Len {
			t.Fatalf("bad boundary lengths above=%d below=%d size=%+v", len(boundaries.Above), len(boundaries.Below), size)
		}
	})
}

func FuzzSaveRestorationFrameBoundaryLines(f *testing.F) {
	f.Add(uint16(300), uint16(260), uint8(1), uint8(0), false, false, false, false)
	f.Add(uint16(64), uint16(57), uint8(0), uint8(1), true, true, true, true)
	f.Fuzz(func(t *testing.T, rawW uint16, rawH uint16, rawUnit uint8, rawType uint8, ssX bool, ssY bool, mono bool, afterCDEF bool) {
		unitSizes := [...]uint16{64, 128, 256}
		unitSize := unitSizes[rawUnit%uint8(len(unitSizes))]
		types := [...]parser.RestorationType{parser.RestorationWiener, parser.RestorationSGRProj}
		unitType := types[rawType%uint8(len(types))]
		params := parser.RestorationParams{
			Type:       [3]parser.RestorationType{unitType, unitType, parser.RestorationNone},
			UnitSizeY:  unitSize,
			UnitSizeUV: unitSize,
		}
		color := parser.ColorConfig{MonoChrome: mono, SubsamplingX: ssX, SubsamplingY: ssY}
		size := parser.FrameSize{
			UpscaledWidth:       uint32(rawW%256) + 1,
			Height:              uint32(rawH%256) + 1,
			SuperResDenominator: 8,
		}
		numPlanes := 3
		if mono {
			numPlanes = 1
		}
		planes := make([]RestorationFrameBoundaryPlane, numPlanes)
		for plane := 0; plane < numPlanes; plane++ {
			grid, err := BuildRestorationPlaneGrid(params, size, color, plane)
			if err != nil {
				t.Fatalf("BuildRestorationPlaneGrid plane=%d err=%v", plane, err)
			}
			srcStride := int(grid.PlaneWidth) + int(rawUnit%7)
			src := makeRestorationBoundaryPlane(grid, srcStride)
			var boundaries RestorationStripeBoundaries
			if grid.Type != parser.RestorationNone {
				boundarySize, err := RestorationStripeBoundaryBufferLen(grid)
				if err != nil {
					t.Fatalf("RestorationStripeBoundaryBufferLen plane=%d err=%v", plane, err)
				}
				boundaries = makeSentinelRestorationBoundaries(boundarySize, 0xeeee)
			}
			planes[plane] = RestorationFrameBoundaryPlane{
				Grid:       grid,
				Src:        src,
				SrcStride:  srcStride,
				Boundaries: boundaries,
			}
		}
		manual := cloneRestorationFrameBoundaryPlanes(planes)
		if err := SaveRestorationFrameBoundaryLines(planes, afterCDEF); err != nil {
			t.Fatalf("SaveRestorationFrameBoundaryLines err=%v", err)
		}
		saveRestorationFrameBoundaryLinesManually(t, manual, afterCDEF)
		assertRestorationFrameBoundaryPlanesEqual(t, planes, manual)
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

func BenchmarkSaveRestorationBoundaryLines(b *testing.B) {
	grid := testRestorationBoundaryGrid(b, false, false)
	srcStride := int(grid.PlaneWidth)
	src := makeRestorationBoundaryPlane(grid, srcStride)
	size, err := RestorationStripeBoundaryBufferLen(grid)
	if err != nil {
		b.Fatal(err)
	}
	boundaries := makeSentinelRestorationBoundaries(size, 0)
	b.ReportAllocs()
	b.SetBytes(int64(size.Len * 2))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := SaveRestorationBoundaryLines(grid, src, srcStride, 0, boundaries, i&1 == 0); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkSaveRestorationFrameBoundaryLines(b *testing.B) {
	planes := makeRestorationFrameBoundaryPlanes(b, [3]parser.RestorationType{parser.RestorationWiener, parser.RestorationSGRProj, parser.RestorationNone}, 12, false)
	var bytes int64
	for i := range planes {
		bytes += int64(planes[i].Grid.PlaneWidth * planes[i].Grid.PlaneHeight * 2)
	}
	b.ReportAllocs()
	b.SetBytes(bytes)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := SaveRestorationFrameBoundaryLines(planes, i&1 == 0); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkExtendRestorationFrame(b *testing.B) {
	const width, height = 300, 260
	const borderHorz, borderVert = 6, 3
	const stride = width + 2*borderHorz
	origin := borderVert*stride + borderHorz
	data := makeRestorationBoundarySamples(stride * (height + 2*borderVert))
	b.ReportAllocs()
	b.SetBytes(int64((width + 2*borderHorz) * (height + 2*borderVert) * 2))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := ExtendRestorationFrame(data, stride, origin, width, height, borderHorz, borderVert); err != nil {
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

func makeRestorationFrameBoundaryPlanes(tb testing.TB, types [3]parser.RestorationType, bitDepth uint8, mono bool) []RestorationFrameBoundaryPlane {
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
	planes := make([]RestorationFrameBoundaryPlane, numPlanes)
	for plane := 0; plane < numPlanes; plane++ {
		grid, err := BuildRestorationPlaneGrid(params, size, color, plane)
		if err != nil {
			tb.Fatal(err)
		}
		srcStride := int(grid.PlaneWidth) + 3 + plane
		src := makeRestorationBoundaryPlaneWithBitDepth(grid, srcStride, bitDepth, plane)
		var boundaries RestorationStripeBoundaries
		if grid.Type != parser.RestorationNone {
			boundarySize, err := RestorationStripeBoundaryBufferLen(grid)
			if err != nil {
				tb.Fatal(err)
			}
			boundaries = makeSentinelRestorationBoundaries(boundarySize, 0xeeee)
		} else {
			boundaries = makeRestorationStripeBoundaries(1, 1)
		}
		planes[plane] = RestorationFrameBoundaryPlane{
			Grid:       grid,
			Src:        src,
			SrcStride:  srcStride,
			Boundaries: boundaries,
		}
	}
	return planes
}

func saveRestorationFrameBoundaryLinesManually(tb testing.TB, planes []RestorationFrameBoundaryPlane, afterCDEF bool) {
	tb.Helper()
	for i := range planes {
		if planes[i].Grid.Type == parser.RestorationNone {
			continue
		}
		if err := SaveRestorationBoundaryLines(planes[i].Grid, planes[i].Src, planes[i].SrcStride, planes[i].SrcOrigin, planes[i].Boundaries, afterCDEF); err != nil {
			tb.Fatal(err)
		}
	}
}

func cloneRestorationFrameBoundaryPlanes(src []RestorationFrameBoundaryPlane) []RestorationFrameBoundaryPlane {
	dst := make([]RestorationFrameBoundaryPlane, len(src))
	for i := range src {
		dst[i] = src[i]
		dst[i].Src = append([]uint16(nil), src[i].Src...)
		dst[i].Boundaries.Above = append([]uint16(nil), src[i].Boundaries.Above...)
		dst[i].Boundaries.Below = append([]uint16(nil), src[i].Boundaries.Below...)
	}
	return dst
}

func assertRestorationFrameBoundaryPlanesEqual(t *testing.T, got []RestorationFrameBoundaryPlane, want []RestorationFrameBoundaryPlane) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("planes len=%d want %d", len(got), len(want))
	}
	for i := range got {
		assertSamplesEqual(t, got[i].Boundaries.Above, want[i].Boundaries.Above)
		assertSamplesEqual(t, got[i].Boundaries.Below, want[i].Boundaries.Below)
	}
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

func makeSentinelRestorationBoundaries(size RestorationStripeBoundaryBufferSize, sentinel uint16) RestorationStripeBoundaries {
	above := make([]uint16, size.Len)
	below := make([]uint16, size.Len)
	for i := range above {
		above[i] = sentinel
		below[i] = sentinel
	}
	return RestorationStripeBoundaries{Above: above, Below: below, Stride: int(size.Stride)}
}

func makeRestorationBoundaryPlane(grid RestorationPlaneGrid, stride int) []uint16 {
	src := make([]uint16, stride*int(grid.PlaneHeight))
	for y := uint32(0); y < grid.PlaneHeight; y++ {
		for x := uint32(0); x < grid.PlaneWidth; x++ {
			src[int(y)*stride+int(x)] = uint16(17 + y*97 + x*3)
		}
	}
	return src
}

func makeRestorationBoundaryPlaneWithBitDepth(grid RestorationPlaneGrid, stride int, bitDepth uint8, salt int) []uint16 {
	max := uint16((1 << bitDepth) - 1)
	src := make([]uint16, stride*int(grid.PlaneHeight))
	for y := uint32(0); y < grid.PlaneHeight; y++ {
		for x := uint32(0); x < grid.PlaneWidth; x++ {
			src[int(y)*stride+int(x)] = uint16((int(y)*97 + int(x)*3 + salt*53 + 17) & int(max))
		}
	}
	return src
}

func assertSavedBoundaryRow(t *testing.T, grid RestorationPlaneGrid, src []uint16, srcStride int, boundary []uint16, boundaryStride int, stripe int, ctxRow int, srcRow int) {
	t.Helper()
	width := int(grid.PlaneWidth)
	got, ok := restorationBoundaryPlaneLine(boundary, boundaryStride, stripe*restorationCtxVert+ctxRow, width)
	if !ok {
		t.Fatalf("boundary line stripe=%d row=%d", stripe, ctxRow)
	}
	srcOff := srcRow * srcStride
	want := src[srcOff : srcOff+width]
	for i := range restorationExtraHorz {
		if got[i] != want[0] {
			t.Fatalf("left extension[%d]=%d want %d", i, got[i], want[0])
		}
	}
	for i := range width {
		if got[restorationExtraHorz+i] != want[i] {
			t.Fatalf("sample[%d]=%d want %d", i, got[restorationExtraHorz+i], want[i])
		}
	}
	for i := range restorationExtraHorz {
		if got[restorationExtraHorz+width+i] != want[width-1] {
			t.Fatalf("right extension[%d]=%d want %d", i, got[restorationExtraHorz+width+i], want[width-1])
		}
	}
}

func assertBoundaryUntouched(t *testing.T, boundary []uint16, boundaryStride int, stripe int, ctxRow int, planeWidth int, sentinel uint16) {
	t.Helper()
	got, ok := restorationBoundaryPlaneLine(boundary, boundaryStride, stripe*restorationCtxVert+ctxRow, planeWidth)
	if !ok {
		t.Fatalf("boundary line stripe=%d row=%d", stripe, ctxRow)
	}
	for i, sample := range got {
		if sample != sentinel {
			t.Fatalf("sample[%d]=%d want sentinel %d", i, sample, sentinel)
		}
	}
}

func assertBoundaryRows(t *testing.T, data []uint16, stride int, origin int, stripe RestorationProcessingStripe, scratch []uint16, boundary []uint16, boundaryStride int, rsbRow int, lineWidth int, above bool) {
	t.Helper()
	for i := range restorationBorder {
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
