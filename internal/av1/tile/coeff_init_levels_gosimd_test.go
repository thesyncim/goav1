// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

//go:build goexperiment.simd && (arm64 || amd64) && !purego

package tile

import (
	"bytes"
	"testing"
)

// TestCoeffInitLevelsSIMDRandomMatchesPureGo drives the architecture's Go SIMD
// level-init kernel directly over randomized full-range int16 coefficients and
// saturation edges, for every valid geometry with a SIMD-supported height, and
// compares it byte-for-byte with the pure-Go reference.
func TestCoeffInitLevelsSIMDRandomMatchesPureGo(t *testing.T) {
	rnd := newCoeffContextRandom(0x494e4c31)
	edges := [...]int16{-32768, -32767, -16384, -129, -128, -127, -1, 0, 1, 126, 127, 128, 129, 255, 256, 32767}
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
		got := make([]uint8, scratchLen)
		want := make([]uint8, scratchLen)
		for iter := 0; iter < 64; iter++ {
			for i := range coeffs {
				switch rnd.u8() & 3 {
				case 0:
					coeffs[i] = edges[int(rnd.u8())%len(edges)]
				case 1:
					coeffs[i] = int16(rnd.u16()%512) - 256
				default:
					coeffs[i] = int16(rnd.u16())
				}
			}
			for i := range got {
				got[i] = rnd.u8()
				want[i] = got[i]
			}
			clear(got)
			coeffInitLevelsSIMD(coeffs, int(geo.scanWidth), int(geo.scanHeight), got)
			coeffInitLevelsPureGo(coeffs, int(geo.scanWidth), int(geo.scanHeight), want, scratchLen)
			if !bytes.Equal(got, want) {
				for i := range got {
					if got[i] != want[i] {
						t.Fatalf("size=%d iter=%d levels[%d]=%d want %d", size, iter, i, got[i], want[i])
					}
				}
			}
		}
	}
}
