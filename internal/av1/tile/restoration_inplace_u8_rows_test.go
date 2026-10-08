package tile

import (
	"bytes"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/parser"
	av1restoration "github.com/thesyncim/goav1/internal/av1/restoration"
)

// TestApplyRestorationFramePlaneInPlaceU8RowsMatchesWholePlane proves the RU-row
// banded in-place walk (ApplyRestorationFramePlaneInPlaceU8Rows over disjoint
// row ranges, each with its own band-private scratch) is byte-identical to the
// whole-plane in-place walk. This is the byte-exactness contract the pooled
// frame-level restoration apply relies on: disjoint RU-row bands filter to the
// same pixels as the single top-to-bottom walk, so the banded result is
// worker-count invariant.
func TestApplyRestorationFramePlaneInPlaceU8RowsMatchesWholePlane(t *testing.T) {
	const bitDepth = 8
	wiener := RestorationUnit{Type: parser.RestorationWiener, Wiener: av1restoration.DefaultWienerInfo()}
	sgr := RestorationUnit{Type: parser.RestorationSGRProj, SGRProj: SGRProjInfo{ParamsIndex: 4, XQD: [2]int8{-32, 31}}}

	cases := []restorationInPlaceU8TestCase{
		{
			name: "all_wiener_multi_row", width: 300, height: 400, unitSizeY: 64, unitSizeUV: 32,
			types: [3]parser.RestorationType{parser.RestorationWiener, parser.RestorationWiener, parser.RestorationWiener},
		},
		{
			name: "switchable_pattern", width: 300, height: 400, unitSizeY: 64, unitSizeUV: 32,
			types: [3]parser.RestorationType{parser.RestorationSwitchable, parser.RestorationSwitchable, parser.RestorationSwitchable},
			pattern: func(plane int, grid RestorationPlaneGrid, i int) parser.RestorationType {
				row := i / int(grid.HorzUnits)
				col := i % int(grid.HorzUnits)
				switch (row + col + plane) % 3 {
				case 0:
					return parser.RestorationNone
				case 1:
					return parser.RestorationWiener
				default:
					return parser.RestorationSGRProj
				}
			},
		},
		{
			name: "large_units", width: 355, height: 420, unitSizeY: 128, unitSizeUV: 64,
			types: [3]parser.RestorationType{parser.RestorationWiener, parser.RestorationSGRProj, parser.RestorationWiener},
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
			for plane := 0; plane < planeCount; plane++ {
				grid := plan.Grids[plane]
				if grid.Type == parser.RestorationNone {
					continue
				}
				records[plane] = makeRestorationPlaneRecords(t, grid, func(i int) RestorationUnit {
					typ := grid.Type
					if tc.pattern != nil {
						typ = tc.pattern(plane, grid, i)
					} else if typ == parser.RestorationSwitchable {
						typ = parser.RestorationWiener
					}
					switch typ {
					case parser.RestorationWiener:
						return wiener
					case parser.RestorationSGRProj:
						u := sgr
						u.SGRProj.ParamsIndex = uint8((i + plane) % 16)
						return u
					default:
						return RestorationUnit{Type: parser.RestorationNone}
					}
				})
				boundaries[plane] = makeRestorationApplyBoundaries(t, grid, bitDepth, 11+plane*13)
			}

			// bandCounts: 2, 3, and one-band-per-RU-row are the seams the
			// pooled apply may land on for various worker counts.
			for _, bandCount := range []int{2, 3, 0} {
				whole := makeRestorationInPlaceTestFrame(t, tc.width, tc.height, tc.mono)
				fillRestorationTestFrame(whole, bitDepth, 0xa11ce+uint32(len(tc.name)))
				banded := makeRestorationInPlaceTestFrame(t, tc.width, tc.height, tc.mono)
				copyRestorationTestFrame(t, banded, whole)

				for plane := 0; plane < planeCount; plane++ {
					grid := plan.Grids[plane]
					if grid.Type == parser.RestorationNone {
						continue
					}
					segLen, err := restorationInPlaceU8SampleScratchLen(grid)
					if err != nil {
						t.Fatal(err)
					}
					unitSize, err := RestorationPlaneApplyScratchLen(grid, records[plane], false)
					if err != nil {
						t.Fatal(err)
					}

					// Whole-plane serial walk.
					wholeBuf := restorationTestFramePlane(whole, plane)
					wholeData := make([]uint16, segLen)
					wholeScratch := makeRestorationBoundaryApplyScratch(unitSize)
					wantRes, err := applyRestorationPlaneRecordsInPlaceU8(grid, records[plane], boundaries[plane], wholeBuf, wholeData, wholeScratch)
					if err != nil {
						t.Fatal(err)
					}

					// RU-row banded walk with band-private scratch.
					vertUnits := int(grid.VertUnits)
					nb := bandCount
					if nb <= 0 || nb > vertUnits {
						nb = vertUnits
					}
					bandedBuf := restorationTestFramePlane(banded, plane)
					var gotRes RestorationPlaneApplyResult
					base := vertUnits / nb
					extra := vertUnits % nb
					r0 := 0
					for b := 0; b < nb; b++ {
						count := base
						if b < extra {
							count++
						}
						r1 := r0 + count
						if r0 >= r1 {
							continue
						}
						bandData := make([]uint16, segLen)
						bandScratch := makeRestorationBoundaryApplyScratch(unitSize)
						bandRes, err := ApplyRestorationFramePlaneInPlaceU8Rows(grid, records[plane], boundaries[plane], bandedBuf, bandData, bandScratch, r0, r1)
						if err != nil {
							t.Fatalf("band [%d,%d): %v", r0, r1, err)
						}
						gotRes.Records += bandRes.Records
						gotRes.FilteredRecords += bandRes.FilteredRecords
						gotRes.Stripes += bandRes.Stripes
						gotRes.ProcessingUnits += bandRes.ProcessingUnits
						r0 = r1
					}
					if gotRes != wantRes {
						t.Fatalf("bandCount=%d plane=%d result mismatch: got %+v want %+v", bandCount, plane, gotRes, wantRes)
					}
					gp := restorationTestFramePlane(banded, plane)
					wp := restorationTestFramePlane(whole, plane)
					if !bytes.Equal(gp.Pix, wp.Pix) {
						for i := range wp.Pix {
							if gp.Pix[i] != wp.Pix[i] {
								t.Fatalf("bandCount=%d plane=%d byte %d (x=%d y=%d): banded=%d whole=%d",
									bandCount, plane, i, i%gp.Stride, i/gp.Stride, gp.Pix[i], wp.Pix[i])
							}
						}
					}
				}
			}
		})
	}
}
