package tile

import (
	"bytes"
	"testing"

	av1frame "github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/parser"
)

// TestApplyRestorationFrameToFrameU8MatchesWidePath is the direct old-vs-new
// differential for the 8-bit pixel-domain restoration walk: the uint8
// snapshot + direct-frame-write path (ApplyRestorationFrameToFrame at
// bitDepth 8) must leave every frame byte identical to the widened
// uint16 round-trip (applyRestorationFrameToFrameWide) on the same input,
// across record-type mixes, monochrome, and both boundary modes.
func TestApplyRestorationFrameToFrameU8MatchesWidePath(t *testing.T) {
	const bitDepth = 8
	typeMixes := [][3]parser.RestorationType{
		{parser.RestorationWiener, parser.RestorationSGRProj, parser.RestorationNone},
		{parser.RestorationSwitchable, parser.RestorationSGRProj, parser.RestorationWiener},
		{parser.RestorationSGRProj, parser.RestorationWiener, parser.RestorationSwitchable},
	}
	for _, optimized := range []bool{false, true} {
		for _, mono := range []bool{false, true} {
			for mi, types := range typeMixes {
				plan := testRestorationFramePlan(t, types, mono)
				planes := makeRestorationFramePlanes(t, types, bitDepth, mono)
				frmWide := makeRestorationApplyFrame(t, bitDepth, mono)
				fillFrameFromRestorationPlanes(t, frmWide, planes, bitDepth)
				frmU8 := makeRestorationApplyFrame(t, bitDepth, mono)
				fillFrameFromRestorationPlanes(t, frmU8, planes, bitDepth)

				var records [3][]RestorationUnitRecord
				var boundaries [3]RestorationStripeBoundaries
				for plane := range planes {
					records[plane] = planes[plane].Records
					boundaries[plane] = planes[plane].Boundaries
				}
				sampleSize, err := RestorationFrameSampleScratchLen(plan, frmWide)
				if err != nil {
					t.Fatal(err)
				}
				applySize, err := RestorationFrameScratchLen(planes, optimized)
				if err != nil {
					t.Fatal(err)
				}
				align, err := restorationFrameSampleAlign(frmWide, 1)
				if err != nil {
					t.Fatal(err)
				}

				want, err := applyRestorationFrameToFrameWide(plan, frmWide, records, boundaries, make([]uint16, sampleSize.DataLen), make([]uint16, sampleSize.DstLen), makeRestorationBoundaryApplyScratch(applySize), optimized, sampleSize, 1, bitDepth, align)
				if err != nil {
					t.Fatal(err)
				}
				dataScratch := make([]uint16, sampleSize.DataLen)
				dstScratch := make([]uint16, sampleSize.DstLen)
				fillUint16(dataScratch, 0xbeef)
				fillUint16(dstScratch, 0xdead)
				got, err := ApplyRestorationFrameToFrame(plan, frmU8, records, boundaries, dataScratch, dstScratch, makeRestorationBoundaryApplyScratch(applySize), optimized)
				if err != nil {
					t.Fatal(err)
				}
				if got != want {
					t.Fatalf("optimized=%v mono=%v mix=%d result=%+v want %+v", optimized, mono, mi, got, want)
				}
				planeCount := 3
				if mono {
					planeCount = 1
				}
				for plane := 0; plane < planeCount; plane++ {
					gotPlane := restorationTestFramePlane(frmU8, plane)
					wantPlane := restorationTestFramePlane(frmWide, plane)
					if !bytes.Equal(gotPlane.Pix, wantPlane.Pix) {
						for i := range wantPlane.Pix {
							if gotPlane.Pix[i] != wantPlane.Pix[i] {
								t.Fatalf("optimized=%v mono=%v mix=%d plane=%d byte %d (x=%d y=%d): got %d want %d", optimized, mono, mi, plane, i, i%gotPlane.Stride, i/gotPlane.Stride, gotPlane.Pix[i], wantPlane.Pix[i])
							}
						}
					}
				}
			}
		}
	}
}

// TestApplyRestorationFrameToFrameU8IsZeroAlloc protects the decoder's
// steady-state budget: the 8-bit pixel-domain walk must not allocate per
// frame (the uint8 snapshot reuses the uint16 data arena via a byte view).
func TestApplyRestorationFrameToFrameU8IsZeroAlloc(t *testing.T) {
	const bitDepth = 8
	types := [3]parser.RestorationType{parser.RestorationWiener, parser.RestorationSGRProj, parser.RestorationSwitchable}
	plan := testRestorationFramePlan(t, types, false)
	planes := makeRestorationFramePlanes(t, types, bitDepth, false)
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
	dataScratch := make([]uint16, sampleSize.DataLen)
	dstScratch := make([]uint16, sampleSize.DstLen)
	applyScratch := makeRestorationBoundaryApplyScratch(applySize)
	allocs := testing.AllocsPerRun(20, func() {
		if _, err := ApplyRestorationFrameToFrame(plan, frm, records, boundaries, dataScratch, dstScratch, applyScratch, false); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("ApplyRestorationFrameToFrame (8-bit) allocated: %f", allocs)
	}
}

// TestLoadBorderedBytePlaneMatchesWidenedLoad proves the uint8 bordered
// snapshot has the exact layout and visible content of the widened uint16
// load, and leaves border bytes untouched.
func TestLoadBorderedBytePlaneMatchesWidenedLoad(t *testing.T) {
	plane := av1frame.Plane{Pix: make([]byte, 23*9), Stride: 23, Width: 17, Height: 9}
	for i := range plane.Pix {
		plane.Pix[i] = byte(i*31 + 7)
	}
	layout, err := av1frame.BorderedSamplePlaneLen(plane, 1, 3, 2, 8)
	if err != nil {
		t.Fatal(err)
	}
	wide, err := av1frame.LoadBorderedSamplePlane(make([]uint16, layout.Len), plane, 1, 3, 2, 8)
	if err != nil {
		t.Fatal(err)
	}
	buf := make([]uint8, layout.Len)
	for i := range buf {
		buf[i] = 0xa5
	}
	got, err := av1frame.LoadBorderedBytePlane(buf, plane, 3, 2, 8)
	if err != nil {
		t.Fatal(err)
	}
	if got.Stride != wide.Stride || got.Origin != wide.Origin || got.Width != wide.Width || got.Height != wide.Height ||
		got.BorderHorz != wide.BorderHorz || got.BorderVert != wide.BorderVert || len(got.Pix) != len(wide.Pix) {
		t.Fatalf("layout got=%+v want %+v", got, wide)
	}
	for y := 0; y < plane.Height; y++ {
		for x := 0; x < plane.Width; x++ {
			idx := got.Origin + y*got.Stride + x
			if uint16(got.Pix[idx]) != wide.Pix[idx] {
				t.Fatalf("visible (%d,%d) got=%d want %d", x, y, got.Pix[idx], wide.Pix[idx])
			}
		}
	}
	visible := make(map[int]bool, plane.Width*plane.Height)
	for y := 0; y < plane.Height; y++ {
		for x := 0; x < plane.Width; x++ {
			visible[got.Origin+y*got.Stride+x] = true
		}
	}
	for i := range got.Pix {
		if !visible[i] && got.Pix[i] != 0xa5 {
			t.Fatalf("border byte %d overwritten with %d", i, got.Pix[i])
		}
	}
}
