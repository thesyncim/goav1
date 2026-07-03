// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

//go:build amd64 && !purego

package tile

import "github.com/thesyncim/goav1/internal/av1/dsp/cpu"

//go:noescape
func coeffInitLevelsAVX2Asm(coeffs *int16, levels *uint8, columns uintptr, height uintptr, stride uintptr)

// coeffInitLevelsAVX2 clears the padded level scratch and fills the live rows
// with the AVX2 kernel. It is bit-identical to coeffInitLevelsPureGo and is
// exercised directly by the differential test on every amd64 host (including
// Rosetta, where CPUID hides AVX2 from the dispatch gate below).
func coeffInitLevelsAVX2(coeffs []int16, scanWidth int, scanHeight int, levels []uint8, scratchLen int) {
	clear(levels[:scratchLen])
	coeffInitLevelsAVX2Asm(&coeffs[0], &levels[0], uintptr(scanWidth), uintptr(scanHeight), uintptr(scanHeight+txPadHorizontal))
}

// coeffInitLevelsArch is the amd64 AVX2 slice of libaom's av1_txb_init_levels_c
// adapted to goav1's column-major levels[col*(height+TX_PAD_HOR)+row] scratch.
func coeffInitLevelsArch(coeffs []int16, scanWidth int, scanHeight int, levels []uint8, scratchLen int) bool {
	if !cpu.Detected.AVX2 {
		return false
	}
	switch scanHeight {
	case 4, 8, 16, 32:
	default:
		return false
	}
	coeffInitLevelsAVX2(coeffs, scanWidth, scanHeight, levels, scratchLen)
	return true
}
