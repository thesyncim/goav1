// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build unix && goexperiment.simd && (arm64 || amd64) && !purego

package tile

import (
	"fmt"
	"slices"
	"testing"

	"github.com/thesyncim/goav1/internal/guardpage"
)

// TestCoeffKernelsStayInsideGuardedBuffers runs the level-init kernel on
// coefficient and level buffers exactly as long as each transform size needs
// (maxEOB coefficients, scratchLen levels), ending at or starting right after
// an inaccessible page.
func TestCoeffKernelsStayInsideGuardedBuffers(t *testing.T) {
	rnd := newCoeffContextRandom(0x67756172)
	initRuns := 0
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
			if !ran {
				continue
			}
			initRuns++
			coeffInitLevelsArchReference(slices.Clone(coeffs), int(geo.scanWidth), int(geo.scanHeight), want)
			if !slices.Equal(levels, want) {
				t.Fatalf("%s: guarded level init differs", name)
			}
		}
	}
	if coeffNZMapSIMDAvailable() && initRuns == 0 {
		t.Fatal("SIMD level-init kernel never ran")
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
