// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

//go:build amd64 && !purego

package tile

import (
	"testing"

	"github.com/thesyncim/goav1/internal/av1/transform"
)

// coeffNZMapContextsAVX2ArchForTest reproduces coeffNZMapContextsArch but drives
// the unconditional AVX2 full-map kernel, so the assembly is exercised on every
// amd64 host including Rosetta (where CPUID hides AVX2 from the production gate).
func coeffNZMapContextsAVX2ArchForTest(levels []uint8, size TransformSize, class transform.Class, scan []int16, eob int, contexts []int8, maxEOB int) bool {
	if !class.Valid() || eob <= 1 {
		return false
	}
	geo := coeffGeometryTable[size]
	if !geo.valid || len(levels) < int(geo.scratchLen) || len(contexts) < maxEOB {
		return false
	}
	switch geo.scanHeight {
	case 4, 8, 16, 32:
		if eob == maxEOB {
			if !coeffNZMapContextsAVX2Full(levels, size, class, contexts) {
				return false
			}
			coeffNZMapFinalizeFullClass(contexts, scan, eob, maxEOB, class)
			return true
		}
		var full [maxCoeffScanLen]int8
		if !coeffNZMapContextsAVX2Full(levels, size, class, full[:]) {
			return false
		}
		coeffNZMapCopyPartialClass(full[:], contexts, scan, eob, maxEOB, class)
		return true
	default:
		return false
	}
}

func TestCoeffNZMapContextsAVX2MatchesScalar(t *testing.T) {
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
					if !coeffNZMapContextsAVX2ArchForTest(levels, size, class, scan, eob, got, maxEOB) {
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

func TestCoeffNZMapContextsAVX2FullZeroAlloc(t *testing.T) {
	geo := coeffGeometryTable[TransformSize32x32]
	levels := make([]uint8, int(geo.scratchLen))
	for i := range levels {
		levels[i] = uint8(i*37) & 0x7f
	}
	contexts := make([]int8, int(geo.maxEOB))
	allocs := testing.AllocsPerRun(200, func() {
		coeffNZMapContextsAVX2Full(levels, TransformSize32x32, transform.Class2D, contexts)
	})
	if allocs != 0 {
		t.Fatalf("coeffNZMapContextsAVX2Full allocs = %v, want 0", allocs)
	}
}
