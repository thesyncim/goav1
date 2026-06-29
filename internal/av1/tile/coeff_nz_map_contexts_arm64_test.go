// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

//go:build arm64 && !purego

package tile

import (
	"testing"

	"github.com/thesyncim/goav1/internal/av1/transform"
)

func TestCoeffNZMapContextsNEONMatchesScalar(t *testing.T) {
	rnd := newCoeffContextRandom(0x4e5a4d41)
	classes := [...]transform.Class{transform.Class2D, transform.ClassHoriz, transform.ClassVert}
	for size := range transformSizeCount {
		geo := coeffGeometryTable[size]
		if !geo.valid || (geo.scanHeight != 4 && geo.scanHeight != 8 && geo.scanHeight != 16 && geo.scanHeight != 32) {
			continue
		}
		maxEOB := int(geo.maxEOB)
		levels := make([]uint8, int(geo.scratchLen))
		for _, class := range classes {
			scan := coeffScanTable[size][class]
			if len(scan) < maxEOB {
				t.Fatalf("size=%d class=%d missing scan", size, class)
			}
			for iter := 0; iter < 16; iter++ {
				for i := range levels {
					levels[i] = rnd.u8() & 0x7f
				}
				for _, eob := range coeffNZMapEOBCases(maxEOB) {
					if eob <= 1 {
						continue
					}
					got := make([]int8, maxEOB)
					want := make([]int8, maxEOB)
					for i := range got {
						v := int8(rnd.u8() & 0x3f)
						got[i] = v
						want[i] = v
					}
					if err := coeffNZMapContextsScalar(levels, size, class, scan, eob, want, maxEOB); err != nil {
						t.Fatal(err)
					}
					if !coeffNZMapContextsArch(levels, size, class, scan, eob, got, maxEOB) {
						t.Fatalf("size=%d class=%d eob=%d did not use arch path", size, class, eob)
					}
					for i := range got {
						if got[i] != want[i] {
							t.Fatalf("size=%d class=%d eob=%d iter=%d context[%d]=%d want %d", size, class, eob, iter, i, got[i], want[i])
						}
					}
				}
			}
		}
	}
}
