// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

//go:build arm64 && !purego

package tile

import "testing"

func TestCoeffCulLevelNEONPathMatchesScalar(t *testing.T) {
	geo := coeffGeometryTable[TransformSize32x32]
	maxEOB := int(geo.maxEOB)
	coeffs := make([]int16, maxEOB)
	for i := range coeffs {
		switch i & 5 {
		case 0:
			coeffs[i] = 0
		case 1:
			coeffs[i] = int16(i%17 - 8)
		default:
			coeffs[i] = int16((i*97)%4096 - 2048)
		}
	}
	scan := coeffScanTable[TransformSize32x32][0]
	for _, eob := range coeffCulLevelEOBCases(maxEOB) {
		got, ok := coeffCulLevelArch(coeffs, scan, eob)
		if eob == 0 {
			if ok {
				t.Fatalf("eob=0 unexpectedly used arch path")
			}
			continue
		}
		if !ok {
			t.Fatalf("eob=%d did not use arch path", eob)
		}
		want := coeffCulLevelScalar(coeffs, scan, eob)
		if got != want {
			t.Fatalf("eob=%d cul=%d want %d", eob, got, want)
		}
	}
}
