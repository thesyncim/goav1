// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

//go:build goexperiment.simd && (arm64 || amd64) && !purego

package tile

import (
	"testing"

	"github.com/thesyncim/goav1/internal/av1/transform"
)

// TestCoeffNZMapContextsSIMDFullMatchesScalar drives the Go SIMD full-map kernel
// directly (no CPU gate, so it runs on every host the architecture build
// supports) and compares every dense context with the scalar reference.
func TestCoeffNZMapContextsSIMDFullMatchesScalar(t *testing.T) {
	rnd := newCoeffContextRandom(0x4e5a4d42)
	classes := [...]transform.Class{transform.Class2D, transform.ClassHoriz, transform.ClassVert}
	for size := range transformSizeCount {
		geo := coeffGeometryTable[size]
		if !geo.valid || (geo.scanHeight != 4 && geo.scanHeight != 8 && geo.scanHeight != 16 && geo.scanHeight != 32) {
			continue
		}
		maxEOB := int(geo.maxEOB)
		levels := make([]uint8, int(geo.scratchLen))
		for _, class := range classes {
			for iter := 0; iter < 32; iter++ {
				for i := range levels {
					switch iter & 3 {
					case 0:
						levels[i] = rnd.u8()
					case 1:
						levels[i] = rnd.u8() & 0x7f
					default:
						levels[i] = rnd.u8() & 3
					}
				}
				got := make([]int8, maxEOB)
				want := make([]int8, maxEOB)
				for i := range got {
					got[i] = int8(rnd.u8())
				}
				// The per-position reference: the eob-tail rule in the scalar
				// scan loop does not apply to the full map, so query each
				// raster position directly.
				for pos := range want {
					ctx, err := CoeffLowerLevelsContext(levels, size, class, pos)
					if err != nil {
						t.Fatal(err)
					}
					want[pos] = int8(ctx)
				}
				if !coeffNZMapContextsSIMDFull(levels, size, class, got) {
					t.Fatalf("size=%d class=%d did not use SIMD full map", size, class)
				}
				for i := range got {
					if got[i] != want[i] {
						t.Fatalf("size=%d class=%d iter=%d context[%d]=%d want %d", size, class, iter, i, got[i], want[i])
					}
				}
			}
		}
	}
}

// TestCoeffNZMapContextsSIMDMatchesScalar checks the dispatched arch path,
// including partial eob copies, against the scalar reference. It requires the
// SIMD path whenever the architecture gate reports it available.
func TestCoeffNZMapContextsSIMDMatchesScalar(t *testing.T) {
	if !coeffNZMapSIMDAvailable() {
		t.Skip("Go SIMD nz-map unavailable on this CPU")
	}
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

// TestCoeffNZMapContextsSIMDFullZeroAlloc covers the plain shape and the
// windowed tail groups of the 8-row and 4-row paths (Horiz reaches the furthest
// neighbour, so its last group is staged through the window).
func TestCoeffNZMapContextsSIMDFullZeroAlloc(t *testing.T) {
	shapes := [...]struct {
		size  TransformSize
		class transform.Class
	}{
		{TransformSize32x32, transform.Class2D},
		{TransformSize8x8, transform.ClassHoriz},
		{TransformSize4x4, transform.ClassHoriz},
		{TransformSize16x8, transform.ClassHoriz},
	}
	for _, s := range shapes {
		geo := coeffGeometryTable[s.size]
		levels := make([]uint8, int(geo.scratchLen))
		for i := range levels {
			levels[i] = uint8(i*37) & 0x7f
		}
		contexts := make([]int8, int(geo.maxEOB))
		allocs := testing.AllocsPerRun(200, func() {
			coeffNZMapContextsSIMDFull(levels, s.size, s.class, contexts)
		})
		if allocs != 0 {
			t.Fatalf("coeffNZMapContextsSIMDFull size=%d class=%d allocs = %v, want 0", s.size, s.class, allocs)
		}
	}
}
