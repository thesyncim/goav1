// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

//go:build arm64 && !purego

package tile

import (
	"bytes"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/dsp/cpu"
)

func TestCoeffInitLevelsNEONMatchesPureGo(t *testing.T) {
	if !cpu.Detected.NEON {
		t.Skip("NEON unavailable")
	}
	for size := range transformSizeCount {
		geo := coeffGeometryTable[size]
		if !geo.valid {
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
		if ok := coeffInitLevelsArch(coeffs, int(geo.scanWidth), int(geo.scanHeight), got, scratchLen); !ok {
			t.Fatalf("size=%d NEON hook not used", size)
		}
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

func TestCoeffInitLevelsNEONDoesNotAllocate(t *testing.T) {
	if !cpu.Detected.NEON {
		t.Skip("NEON unavailable")
	}
	var coeffs [1024]int16
	var levels [maxCoeffScratchLen]uint8
	for i := range coeffs {
		coeffs[i] = int16((i*41)%8191 - 4095)
	}
	geo := coeffGeometryTable[TransformSize64x64]
	allocs := testing.AllocsPerRun(1000, func() {
		if ok := coeffInitLevelsArch(coeffs[:int(geo.maxEOB)], int(geo.scanWidth), int(geo.scanHeight), levels[:int(geo.scratchLen)], int(geo.scratchLen)); !ok {
			t.Fatal("NEON hook not used")
		}
	})
	if allocs != 0 {
		t.Fatalf("CoeffInitLevels NEON allocated: %f", allocs)
	}
}
