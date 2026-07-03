// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

//go:build amd64 && !purego

package tile

import (
	"bytes"
	"testing"
)

// These tests call the AVX2 kernel directly rather than through the dispatch
// gate. The gate only binds AVX2 when CPUID reports OS-enabled AVX2; under
// Apple Rosetta 2 the guest CPUID hides AVX so the gate falls back to pure-Go
// even though Rosetta can execute the AVX2 instructions. Calling
// coeffInitLevelsAVX2 directly exercises the real assembly on every amd64 host.

func TestCoeffInitLevelsAVX2MatchesPureGo(t *testing.T) {
	for size := range transformSizeCount {
		geo := coeffGeometryTable[size]
		if !geo.valid {
			continue
		}
		switch geo.scanHeight {
		case 4, 8, 16, 32:
		default:
			continue
		}
		maxEOB := int(geo.maxEOB)
		scratchLen := int(geo.scratchLen)
		coeffs := make([]int16, maxEOB)
		for i := range coeffs {
			switch i % 13 {
			case 0:
				coeffs[i] = -32768
			case 1:
				coeffs[i] = -129
			case 2:
				coeffs[i] = -128
			case 3:
				coeffs[i] = -127
			case 4:
				coeffs[i] = -1
			case 5:
				coeffs[i] = 0
			case 6:
				coeffs[i] = 127
			case 7:
				coeffs[i] = 128
			default:
				coeffs[i] = int16((i*73)%2047 - 1023)
			}
		}

		got := bytes.Repeat([]byte{0x5a}, scratchLen)
		want := bytes.Repeat([]byte{0xa5}, scratchLen)
		coeffInitLevelsAVX2(coeffs, int(geo.scanWidth), int(geo.scanHeight), got, scratchLen)
		coeffInitLevelsPureGo(coeffs, int(geo.scanWidth), int(geo.scanHeight), want, scratchLen)
		if !bytes.Equal(got, want) {
			for i := range got {
				if got[i] != want[i] {
					t.Fatalf("size=%d levels[%d]=%d want %d", size, i, got[i], want[i])
				}
			}
		}
	}
}

func TestCoeffInitLevelsAVX2DoesNotAllocate(t *testing.T) {
	var coeffs [1024]int16
	var levels [maxCoeffScratchLen]uint8
	for i := range coeffs {
		coeffs[i] = int16((i*41)%8191 - 4095)
	}
	geo := coeffGeometryTable[TransformSize64x64]
	allocs := testing.AllocsPerRun(1000, func() {
		coeffInitLevelsAVX2(coeffs[:int(geo.maxEOB)], int(geo.scanWidth), int(geo.scanHeight), levels[:int(geo.scratchLen)], int(geo.scratchLen))
	})
	if allocs != 0 {
		t.Fatalf("CoeffInitLevels AVX2 allocated: %f", allocs)
	}
}
