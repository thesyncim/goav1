// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build unix && goexperiment.simd && (arm64 || amd64) && !purego

package tile

import (
	"fmt"
	"slices"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/transform"
	"github.com/thesyncim/goav1/internal/guardpage"
)

// TestCoeffKernelsStayInsideGuardedBuffers runs the level-init and nz-map
// context kernels on coefficient, level and context buffers exactly as long
// as each transform size needs (maxEOB coefficients and contexts, scratchLen
// levels), ending at or starting right after an inaccessible page.
func TestCoeffKernelsStayInsideGuardedBuffers(t *testing.T) {
	rnd := newCoeffContextRandom(0x67756172)
	classes := [...]transform.Class{transform.Class2D, transform.ClassHoriz, transform.ClassVert}
	initRuns, nzRuns := 0, 0
	for size := range transformSizeCount {
		geo := coeffGeometryTable[size]
		if !geo.valid {
			continue
		}
		maxEOB, scratchLen := int(geo.maxEOB), int(geo.scratchLen)
		for _, before := range []bool{false, true} {
			var coeffs []int16
			var levels []uint8
			if before {
				coeffs, levels = guardpage.Before[int16](t, maxEOB), guardpage.Before[uint8](t, scratchLen)
			} else {
				coeffs, levels = guardpage.After[int16](t, maxEOB), guardpage.After[uint8](t, scratchLen)
			}
			for i := range coeffs {
				coeffs[i] = int16(rnd.u8())<<4 - 2048
			}
			name := fmt.Sprintf("size=%d before=%v", size, before)
			want := make([]uint8, scratchLen)
			var ran bool
			guardpage.Check(t, name+" init levels", func() {
				ran = coeffInitLevelsArch(coeffs, int(geo.scanWidth), int(geo.scanHeight), levels, scratchLen)
			})
			if ran {
				initRuns++
				coeffInitLevelsArchReference(slices.Clone(coeffs), int(geo.scanWidth), int(geo.scanHeight), want)
				if !slices.Equal(levels, want) {
					t.Fatalf("%s: guarded level init differs", name)
				}
			} else {
				for i := range levels {
					levels[i] = rnd.u8() & 0x7f
				}
			}
			for _, class := range classes {
				scan := coeffScanTable[size][class]
				if len(scan) < maxEOB {
					continue
				}
				for _, eob := range coeffNZMapEOBCases(maxEOB) {
					if eob <= 1 {
						continue
					}
					var got []int8
					if before {
						got = guardpage.Before[int8](t, maxEOB)
					} else {
						got = guardpage.After[int8](t, maxEOB)
					}
					ref := make([]int8, maxEOB)
					var ok bool
					guardpage.Check(t, fmt.Sprintf("%s class=%d eob=%d nz-map", name, class, eob), func() {
						ok = coeffNZMapContextsArch(levels, TransformSize(size), class, scan, eob, got, maxEOB)
					})
					if !ok {
						continue
					}
					nzRuns++
					if err := coeffNZMapContextsScalar(slices.Clone(levels), TransformSize(size), class, scan, eob, ref, maxEOB); err != nil {
						t.Fatal(err)
					}
					for _, pos := range scan[:eob] {
						if got[pos] != ref[pos] {
							t.Fatalf("%s class=%d eob=%d: guarded nz-map context %d = %d, want %d", name, class, eob, pos, got[pos], ref[pos])
						}
					}
				}
			}
		}
	}
	if coeffNZMapSIMDAvailable() && (initRuns == 0 || nzRuns == 0) {
		t.Fatalf("SIMD kernels never ran: init %d, nz-map %d", initRuns, nzRuns)
	}
}

// coeffInitLevelsArchReference is the scalar level init into a cleared
// scratch window.
func coeffInitLevelsArchReference(coeffs []int16, scanWidth, scanHeight int, levels []uint8) {
	clear(levels)
	stride := scanHeight + txPadHorizontal
	for col := range scanWidth {
		for row := range scanHeight {
			levels[col*stride+row] = coeffAbsClamp127(coeffs[col*scanHeight+row])
		}
	}
}
