// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package prediction

import (
	"slices"
	"testing"
)

// withPureGoDirKernels temporarily forces the dispatched directional-zone
// kernels (the zone-2 "above" run and the zone-3 column fill) back to their
// pure-Go references, runs fn, then restores the resolved slots (NEON on arm64,
// AVX2 on amd64 when active). It lets a differential test capture the SIMD
// output and the reference output through the identical predictor entry point.
func withPureGoDirKernels(fn func()) {
	savedAbove := dirAboveRun8Impl
	savedLeft := dirLeftCol8Impl
	dirAboveRun8Impl = dirAboveRun8PureGo
	dirLeftCol8Impl = dirLeftCol8PureGo
	defer func() {
		dirAboveRun8Impl = savedAbove
		dirLeftCol8Impl = savedLeft
	}()
	fn()
}

// dirBlockDims are the square AV1 block dimensions; the test pairs every width
// with every height to also cover the non-square shapes.
var dirBlockDims = []int{4, 8, 16, 32, 64}

// TestDirectionalKernelsDispatchMatchesPureGo guards the dispatched zone-2 and
// zone-3 directional-interpolation kernels in isolation: the resolved slots
// (NEON on arm64) must reproduce the pure-Go references sample for sample. It
// runs on every build; where no tuned variant exists the comparison is trivial.
func TestDirectionalKernelsDispatchMatchesPureGo(t *testing.T) {
	rnd := newLibaomIntraEdgeRandom(0x5151)

	// zone-2 above run: contiguous forward source, no clamp.
	for _, count := range []int{8, 16, 24, 32, 56, 64} {
		ref := make([]uint16, count+1)
		for i := range ref {
			ref[i] = uint16(rnd.pseudoUniform(256))
		}
		for _, shift := range []int{0, 1, 5, 10, 16, 23, 31} {
			got := make([]byte, count)
			want := make([]byte, count)
			dirAboveRun8Impl(got, ref, shift, count)
			dirAboveRun8PureGo(want, ref, shift, count)
			if !slices.Equal(got, want) {
				t.Fatalf("dirAboveRun8 count=%d shift=%d got=%v want=%v", count, shift, got, want)
			}
		}
	}

	// zone-3 column fill: contiguous forward source, strided byte stores.
	for _, count := range []int{8, 16, 24, 32, 56, 64} {
		ref := make([]uint16, count+1)
		for i := range ref {
			ref[i] = uint16(rnd.pseudoUniform(256))
		}
		for _, stride := range []int{1, 8, 17, 32, 71} {
			for _, shift := range []int{0, 1, 5, 10, 16, 23, 31} {
				gotBuf := make([]byte, count*stride+1)
				wantBuf := make([]byte, count*stride+1)
				dirLeftCol8Impl(gotBuf, stride, ref, shift, count)
				dirLeftCol8PureGo(wantBuf, stride, ref, shift, count)
				if !slices.Equal(gotBuf, wantBuf) {
					t.Fatalf("dirLeftCol8 count=%d stride=%d shift=%d differs", count, stride, shift)
				}
			}
		}
	}
}

// TestDirectionalZ2Z3DispatchMatchesPureGo is the end-to-end byte-exactness
// guard for the zone-2 and zone-3 NEON kernels. For every block shape
// (4x4..64x64, square and non-square), every zone-2 angle (91..179) and zone-3
// angle (181..269), upsample on/off, and a couple of partially-visible extents,
// it runs PredictDirectionalIntraPlaneBlock with the resolved dispatch slots
// (NEON on arm64) and with the slots forced to pure-Go, and asserts the two
// planes are byte-identical. It also cross-checks both against the independent
// predictDirectionalLibaomReference re-derivation.
func TestDirectionalZ2Z3DispatchMatchesPureGo(t *testing.T) {
	rnd := newLibaomIntraEdgeRandom(0xC0FFEE)

	type extent struct{ pw, ph int } // 0,0 means full

	for _, w := range dirBlockDims {
		for _, h := range dirBlockDims {
			// Zone 2 = (90,180), Zone 3 = (180,270). Sweep every integer angle;
			// angles 90/180/270 are vertical/horizontal and out of scope here.
			var angles []int
			for a := 91; a <= 179; a++ {
				angles = append(angles, a)
			}
			for a := 181; a <= 269; a++ {
				angles = append(angles, a)
			}

			for _, angle := range angles {
				for _, up := range []bool{false, true} {
					// Partial extents only make sense up to the block size and
					// only exercise extra reference rows/cols, not the kernels'
					// shape, so keep them small and cheap.
					extents := []extent{{0, 0}}
					if w >= 8 && h >= 8 {
						extents = append(extents, extent{w + 4, h + 4})
					}
					for _, ext := range extents {
						edges := randomDirectionalEdges(w, h, angle, up, 0xff, rnd)

						pw, ph := w, h
						if ext.pw != 0 {
							pw, ph = ext.pw, ext.ph
						}

						// Reference re-derivation (full block only).
						var refOut []uint16
						var refErr error
						if ext.pw == 0 {
							refOut, refErr = predictDirectionalLibaomReference(w, h, angle, edges)
						}

						// Resolved dispatch (NEON on arm64).
						neonPlane, _ := testPlane(w, h, 1, w+3)
						neonErr := PredictDirectionalIntraPlaneBlockWithExtent(neonPlane, 1, 8, 0, 0, w, h, pw, ph, angle, edges)

						// Forced pure-Go.
						var pureErr error
						purePlane, _ := testPlane(w, h, 1, w+3)
						withPureGoDirKernels(func() {
							pureErr = PredictDirectionalIntraPlaneBlockWithExtent(purePlane, 1, 8, 0, 0, w, h, pw, ph, angle, edges)
						})

						if (neonErr == nil) != (pureErr == nil) {
							t.Fatalf("w=%d h=%d angle=%d up=%v ext=%v err mismatch: neon=%v pure=%v",
								w, h, angle, up, ext, neonErr, pureErr)
						}
						if neonErr != nil {
							continue
						}

						neon := collectPlaneSamples(neonPlane, 1, w, h)
						pure := collectPlaneSamples(purePlane, 1, w, h)
						if !slices.Equal(neon, pure) {
							diffDirectional(t, "neon-vs-pure", w, h, angle, up, ext.pw, ext.ph, neon, pure)
						}
						if refErr == nil && refOut != nil {
							if !slices.Equal(neon, refOut) {
								diffDirectional(t, "neon-vs-ref", w, h, angle, up, ext.pw, ext.ph, neon, refOut)
							}
						}
					}
				}
			}
		}
	}
}

func diffDirectional(t *testing.T, tag string, w, h, angle int, up bool, pw, ph int, got, want []uint16) {
	t.Helper()
	for row := 0; row < h; row++ {
		for col := 0; col < w; col++ {
			i := row*w + col
			if got[i] != want[i] {
				t.Fatalf("%s w=%d h=%d angle=%d up=%v ext=%dx%d row=%d col=%d got=%d want=%d",
					tag, w, h, angle, up, pw, ph, row, col, got[i], want[i])
			}
		}
	}
}
