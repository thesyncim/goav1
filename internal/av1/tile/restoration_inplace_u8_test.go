package tile

import (
	"bytes"
	"testing"

	av1frame "github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/parser"
	av1restoration "github.com/thesyncim/goav1/internal/av1/restoration"
)

// restorationInPlaceU8TestCase describes one synthetic frame geometry for the
// in-place-vs-snapshot differential.
type restorationInPlaceU8TestCase struct {
	name       string
	width      int
	height     int
	unitSizeY  uint16
	unitSizeUV uint16
	mono       bool
	types      [3]parser.RestorationType
	// pattern selects each record's unit type from its row-major index; nil
	// means "always the plane's frame type".
	pattern func(plane int, grid RestorationPlaneGrid, i int) parser.RestorationType
}

// TestApplyRestorationU8InPlaceMatchesSnapshot is the direct differential for
// the in-place 8-bit walk: on identical frames, ApplyRestorationFrameToFrame
// (in-place walk at optimized=false) must produce byte-identical planes and
// identical result accounting to the whole-plane pre-restoration snapshot walk
// (applyRestorationPlaneSnapshotU8). The patterns exercise every horizontal
// halo case the in-place walk has to get right: filtered runs (left-neighbour
// backup chains), filtered/none alternation (direct pre-filter reads), single
// columns, mixed Wiener/SGR neighbours, and multi-stripe multi-row units.
func TestApplyRestorationU8InPlaceMatchesSnapshot(t *testing.T) {
	const bitDepth = 8
	wiener := RestorationUnit{Type: parser.RestorationWiener, Wiener: av1restoration.DefaultWienerInfo()}
	sgr := RestorationUnit{Type: parser.RestorationSGRProj, SGRProj: SGRProjInfo{ParamsIndex: 4, XQD: [2]int8{-32, 31}}}

	cases := []restorationInPlaceU8TestCase{
		{
			name: "all_filtered_wiener_5x4", width: 300, height: 260, unitSizeY: 64, unitSizeUV: 32,
			types: [3]parser.RestorationType{parser.RestorationWiener, parser.RestorationWiener, parser.RestorationWiener},
		},
		{
			name: "all_filtered_sgr_5x4", width: 300, height: 260, unitSizeY: 64, unitSizeUV: 32,
			types: [3]parser.RestorationType{parser.RestorationSGRProj, parser.RestorationSGRProj, parser.RestorationSGRProj},
		},
		{
			name: "checkerboard_none", width: 300, height: 260, unitSizeY: 64, unitSizeUV: 32,
			types: [3]parser.RestorationType{parser.RestorationSwitchable, parser.RestorationSwitchable, parser.RestorationSwitchable},
			pattern: func(plane int, grid RestorationPlaneGrid, i int) parser.RestorationType {
				row := i / int(grid.HorzUnits)
				col := i % int(grid.HorzUnits)
				if (row+col+plane)%2 == 0 {
					return parser.RestorationNone
				}
				if col%2 == 0 {
					return parser.RestorationWiener
				}
				return parser.RestorationSGRProj
			},
		},
		{
			name: "wiener_sgr_alternating_columns", width: 300, height: 260, unitSizeY: 64, unitSizeUV: 32,
			types: [3]parser.RestorationType{parser.RestorationSwitchable, parser.RestorationSwitchable, parser.RestorationSwitchable},
			pattern: func(plane int, grid RestorationPlaneGrid, i int) parser.RestorationType {
				if (i%int(grid.HorzUnits))%2 == 0 {
					return parser.RestorationWiener
				}
				return parser.RestorationSGRProj
			},
		},
		{
			name: "none_column_gaps", width: 300, height: 260, unitSizeY: 64, unitSizeUV: 32,
			types: [3]parser.RestorationType{parser.RestorationSwitchable, parser.RestorationSwitchable, parser.RestorationSwitchable},
			pattern: func(plane int, grid RestorationPlaneGrid, i int) parser.RestorationType {
				col := i % int(grid.HorzUnits)
				if col == 1 || col == 3 {
					return parser.RestorationNone
				}
				return parser.RestorationSGRProj
			},
		},
		{
			name: "large_units_multi_stripe", width: 300, height: 260, unitSizeY: 128, unitSizeUV: 64,
			types: [3]parser.RestorationType{parser.RestorationWiener, parser.RestorationSGRProj, parser.RestorationWiener},
		},
		{
			name: "wide_last_column", width: 355, height: 132, unitSizeY: 128, unitSizeUV: 64,
			types: [3]parser.RestorationType{parser.RestorationSwitchable, parser.RestorationSwitchable, parser.RestorationSwitchable},
			pattern: func(plane int, grid RestorationPlaneGrid, i int) parser.RestorationType {
				if (i+plane)%3 == 2 {
					return parser.RestorationNone
				}
				return parser.RestorationWiener
			},
		},
		{
			name: "single_unit_plane", width: 90, height: 70, unitSizeY: 64, unitSizeUV: 32,
			types: [3]parser.RestorationType{parser.RestorationWiener, parser.RestorationSGRProj, parser.RestorationNone},
		},
		{
			name: "mono_filtered_runs", width: 300, height: 260, unitSizeY: 64, mono: true,
			types: [3]parser.RestorationType{parser.RestorationSwitchable, parser.RestorationNone, parser.RestorationNone},
			pattern: func(plane int, grid RestorationPlaneGrid, i int) parser.RestorationType {
				row := i / int(grid.HorzUnits)
				if row%2 == 0 {
					return parser.RestorationWiener
				}
				return parser.RestorationNone
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			params := parser.RestorationParams{Type: tc.types, UnitSizeY: tc.unitSizeY, UnitSizeUV: tc.unitSizeUV}
			color := parser.ColorConfig{MonoChrome: tc.mono, SubsamplingX: true, SubsamplingY: true}
			size := parser.FrameSize{UpscaledWidth: uint32(tc.width), Height: uint32(tc.height), SuperResDenominator: 8}
			plan, err := BuildRestorationFramePlan(params, size, color)
			if err != nil {
				t.Fatal(err)
			}
			if !plan.Active {
				t.Fatal("plan inactive")
			}

			planeCount := int(plan.Planes)
			var records [3][]RestorationUnitRecord
			var boundaries [3]RestorationStripeBoundaries
			var planes []RestorationFramePlane
			for plane := 0; plane < planeCount; plane++ {
				grid := plan.Grids[plane]
				if grid.Type == parser.RestorationNone {
					planes = append(planes, RestorationFramePlane{Grid: grid})
					continue
				}
				wienerUnit := wiener
				sgrUnit := sgr
				records[plane] = makeRestorationPlaneRecords(t, grid, func(i int) RestorationUnit {
					typ := grid.Type
					if tc.pattern != nil {
						typ = tc.pattern(plane, grid, i)
					} else if typ == parser.RestorationSwitchable {
						typ = parser.RestorationWiener
					}
					switch typ {
					case parser.RestorationWiener:
						return wienerUnit
					case parser.RestorationSGRProj:
						u := sgrUnit
						u.SGRProj.ParamsIndex = uint8((i + plane) % 16)
						return u
					default:
						return RestorationUnit{Type: parser.RestorationNone}
					}
				})
				boundaries[plane] = makeRestorationApplyBoundaries(t, grid, bitDepth, 7+plane*13)
				planes = append(planes, RestorationFramePlane{Grid: grid, Records: records[plane], Boundaries: boundaries[plane]})
			}

			frmInPlace := makeRestorationInPlaceTestFrame(t, tc.width, tc.height, tc.mono)
			fillRestorationTestFrame(frmInPlace, bitDepth, 0x5eed+uint32(len(tc.name)))
			frmSnapshot := makeRestorationInPlaceTestFrame(t, tc.width, tc.height, tc.mono)
			copyRestorationTestFrame(t, frmSnapshot, frmInPlace)

			refSampleSize, err := RestorationFrameSampleScratchLen(plan, frmSnapshot, true)
			if err != nil {
				t.Fatal(err)
			}
			sampleSize, err := RestorationFrameSampleScratchLen(plan, frmInPlace, false)
			if err != nil {
				t.Fatal(err)
			}
			applySize, err := RestorationFrameScratchLen(planes, false)
			if err != nil {
				t.Fatal(err)
			}
			align, err := restorationFrameSampleAlign(frmInPlace, 1)
			if err != nil {
				t.Fatal(err)
			}

			// Reference: the whole-plane pre-restoration snapshot walk.
			var want RestorationFrameApplyResult
			refScratch := make([]uint16, refSampleSize.DataLen)
			refApply := makeRestorationBoundaryApplyScratch(applySize)
			dataOffset := 0
			for plane := 0; plane < planeCount; plane++ {
				grid := plan.Grids[plane]
				if grid.Type == parser.RestorationNone {
					continue
				}
				buffer, err := restorationFramePlaneBuffer(frmSnapshot, plane)
				if err != nil {
					t.Fatal(err)
				}
				dataLayout := refSampleSize.Data[plane]
				planeResult, err := applyRestorationPlaneSnapshotU8(grid, records[plane], boundaries[plane], buffer, refScratch[dataOffset:dataOffset+dataLayout.Len], dataLayout.Len, refApply, false, align)
				if err != nil {
					t.Fatal(err)
				}
				dataOffset += dataLayout.Len
				want.PlaneResults[plane] = planeResult
				want.Planes++
				if err := accumulateRestorationFrameResult(&want, planeResult); err != nil {
					t.Fatal(err)
				}
			}

			// Under test: the in-place walk behind ApplyRestorationFrameToFrame.
			dataScratch := make([]uint16, sampleSize.DataLen)
			dstScratch := make([]uint16, sampleSize.DstLen)
			fillUint16(dataScratch, 0xbeef)
			fillUint16(dstScratch, 0xdead)
			got, err := ApplyRestorationFrameToFrame(plan, frmInPlace, records, boundaries, dataScratch, dstScratch, makeRestorationBoundaryApplyScratch(applySize), false)
			if err != nil {
				t.Fatal(err)
			}
			if got != want {
				t.Fatalf("result mismatch: got %+v want %+v", got, want)
			}
			for plane := 0; plane < planeCount; plane++ {
				gotPlane := restorationTestFramePlane(frmInPlace, plane)
				wantPlane := restorationTestFramePlane(frmSnapshot, plane)
				if !bytes.Equal(gotPlane.Pix, wantPlane.Pix) {
					for i := range wantPlane.Pix {
						if gotPlane.Pix[i] != wantPlane.Pix[i] {
							t.Fatalf("plane=%d byte %d (x=%d y=%d): got %d want %d", plane, i, i%gotPlane.Stride, i/gotPlane.Stride, gotPlane.Pix[i], wantPlane.Pix[i])
						}
					}
				}
			}
		})
	}
}

func makeRestorationInPlaceTestFrame(tb testing.TB, width int, height int, mono bool) av1frame.Frame {
	tb.Helper()
	format := av1frame.Format{
		Width:        width,
		Height:       height,
		BitDepth:     8,
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

func copyRestorationTestFrame(tb testing.TB, dst av1frame.Frame, src av1frame.Frame) {
	tb.Helper()
	planeCount := 3
	if src.Format.MonoChrome {
		planeCount = 1
	}
	for plane := 0; plane < planeCount; plane++ {
		d := restorationTestFramePlane(dst, plane)
		s := restorationTestFramePlane(src, plane)
		if len(d.Pix) != len(s.Pix) {
			tb.Fatalf("plane %d size mismatch: %d vs %d", plane, len(d.Pix), len(s.Pix))
		}
		copy(d.Pix, s.Pix)
	}
}
