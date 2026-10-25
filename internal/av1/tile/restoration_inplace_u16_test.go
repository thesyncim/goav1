package tile

import (
	"bytes"
	"testing"

	av1frame "github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/parser"
	av1restoration "github.com/thesyncim/goav1/internal/av1/restoration"
)

// TestApplyRestorationFramePlaneInPlaceU16StripesMatchesFrame proves the
// 10/12-bit in-place stripe walk, run one stripe at a time in reverse order,
// restores every plane exactly as the whole-frame ApplyRestorationFrameToFrame.
func TestApplyRestorationFramePlaneInPlaceU16StripesMatchesFrame(t *testing.T) {
	wiener := RestorationUnit{Type: parser.RestorationWiener, Wiener: av1restoration.DefaultWienerInfo()}
	sgr := RestorationUnit{Type: parser.RestorationSGRProj, SGRProj: SGRProjInfo{ParamsIndex: 4, XQD: [2]int8{-32, 31}}}
	cases := []struct {
		name          string
		width, height int
		unitY, unitUV uint16
		types         [3]parser.RestorationType
	}{
		{"switchable", 300, 400, 64, 32, [3]parser.RestorationType{parser.RestorationSwitchable, parser.RestorationSwitchable, parser.RestorationSwitchable}},
		{"large_units", 355, 420, 128, 64, [3]parser.RestorationType{parser.RestorationWiener, parser.RestorationSGRProj, parser.RestorationWiener}},
		{"odd_size", 131, 77, 64, 32, [3]parser.RestorationType{parser.RestorationSGRProj, parser.RestorationNone, parser.RestorationWiener}},
	}
	for _, bitDepth := range []uint8{10, 12} {
		for _, tc := range cases {
			params := parser.RestorationParams{Type: tc.types, UnitSizeY: tc.unitY, UnitSizeUV: tc.unitUV}
			color := parser.ColorConfig{BitDepth: bitDepth, SubsamplingX: true, SubsamplingY: true}
			size := parser.FrameSize{UpscaledWidth: uint32(tc.width), Height: uint32(tc.height), SuperResDenominator: 8}
			plan, err := BuildRestorationFramePlan(params, size, color)
			if err != nil {
				t.Fatal(err)
			}
			var records [3][]RestorationUnitRecord
			var boundaries [3]RestorationStripeBoundaries
			var unitSize RestorationUnitRecordBoundaryScratchSize
			for plane := 0; plane < int(plan.Planes); plane++ {
				grid := plan.Grids[plane]
				if grid.Type == parser.RestorationNone {
					continue
				}
				records[plane] = makeRestorationPlaneRecords(t, grid, func(i int) RestorationUnit {
					typ := grid.Type
					if typ == parser.RestorationSwitchable {
						typ = [...]parser.RestorationType{parser.RestorationNone, parser.RestorationWiener, parser.RestorationSGRProj}[(i+plane)%3]
					}
					switch typ {
					case parser.RestorationWiener:
						return wiener
					case parser.RestorationSGRProj:
						u := sgr
						u.SGRProj.ParamsIndex = uint8((i + plane) % 16)
						return u
					}
					return RestorationUnit{Type: parser.RestorationNone}
				})
				boundaries[plane] = makeRestorationApplyBoundaries(t, grid, bitDepth, 11+plane*13)
				s, err := RestorationFramePlaneMaxScratchLen(grid, false)
				if err != nil {
					t.Fatal(err)
				}
				unitSize.Unit.Wiener = max(unitSize.Unit.Wiener, s.Unit.Wiener)
				unitSize.Unit.SGRProj = max(unitSize.Unit.SGRProj, s.Unit.SGRProj)
				unitSize.Boundary.Above = max(unitSize.Boundary.Above, s.Boundary.Above)
				unitSize.Boundary.Below = max(unitSize.Boundary.Below, s.Boundary.Below)
			}
			want := makeRestorationU16TestFrame(t, tc.width, tc.height, bitDepth)
			fillRestorationTestFrame(want, bitDepth, 0xbeef+uint32(len(tc.name)))
			got := makeRestorationU16TestFrame(t, tc.width, tc.height, bitDepth)
			copyRestorationTestFrame(t, got, want)
			orig := makeRestorationU16TestFrame(t, tc.width, tc.height, bitDepth)
			copyRestorationTestFrame(t, orig, want)

			sample, err := RestorationFrameSampleScratchLen(plan, want, false)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := ApplyRestorationFrameToFrame(plan, want, records, boundaries, make([]uint16, sample.DataLen), make([]uint16, sample.DstLen), makeRestorationBoundaryApplyScratch(unitSize), false); err != nil {
				t.Fatal(err)
			}
			for plane := 0; plane < int(plan.Planes); plane++ {
				grid := plan.Grids[plane]
				if grid.Type == parser.RestorationNone {
					continue
				}
				segLen, err := RestorationFramePlaneInPlaceU16ScratchLen(grid)
				if err != nil {
					t.Fatal(err)
				}
				buf := restorationTestFramePlane(got, plane)
				for s := RestorationTileStripeCount(grid) - 1; s >= 0; s-- {
					seg := make([]uint16, segLen)
					for i := range seg {
						seg[i] = 0xdead
					}
					if _, err := ApplyRestorationFramePlaneInPlaceU16StripesFollowing(grid, records[plane], boundaries[plane], buf, bitDepth, seg, makeRestorationBoundaryApplyScratch(unitSize), s, s+1); err != nil {
						t.Fatalf("%s %d-bit plane %d stripe %d: %v", tc.name, bitDepth, plane, s, err)
					}
				}
				if bytes.Equal(restorationTestFramePlane(orig, plane).Pix, restorationTestFramePlane(want, plane).Pix) {
					t.Fatalf("%s %d-bit plane %d: restoration changed nothing", tc.name, bitDepth, plane)
				}
				if !bytes.Equal(buf.Pix, restorationTestFramePlane(want, plane).Pix) {
					t.Fatalf("%s %d-bit plane %d differs from the whole-frame restoration", tc.name, bitDepth, plane)
				}
			}
		}
	}
}

func makeRestorationU16TestFrame(tb testing.TB, width, height int, bitDepth uint8) av1frame.Frame {
	tb.Helper()
	format := av1frame.Format{Width: width, Height: height, BitDepth: bitDepth, SubsamplingX: true, SubsamplingY: true, Align: 64}
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
