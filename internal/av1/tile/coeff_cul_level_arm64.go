// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

//go:build arm64 && !purego

package tile

import "github.com/thesyncim/goav1/internal/av1/dsp/cpu"

//go:noescape
func coeffCulLevelNEONAsm(coeffs *int16, scan *int16, eob uintptr) uint8

func coeffCulLevelArch(coeffs []int16, scan []int16, eob int) (uint8, bool) {
	if !cpu.Detected.NEON || eob <= 0 || len(scan) < eob || len(coeffs) == 0 {
		return 0, false
	}
	return coeffCulLevelNEONAsm(&coeffs[0], &scan[0], uintptr(eob)), true
}
