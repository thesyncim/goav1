// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

//go:build arm64 && !purego

package tile

import "github.com/thesyncim/goav1/internal/av1/dsp/cpu"

//go:noescape
func coeffInitLevels4NEONAsm(coeffs *int16, levels *uint8, columns uintptr)

//go:noescape
func coeffInitLevels8NEONAsm(coeffs *int16, levels *uint8, columns uintptr)

//go:noescape
func coeffInitLevels16NEONAsm(coeffs *int16, levels *uint8, columns uintptr)

//go:noescape
func coeffInitLevels32NEONAsm(coeffs *int16, levels *uint8, columns uintptr)

// coeffInitLevelsArch is the arm64 slice of SVT-AV1's
// svt_av1_txb_init_levels_neon adapted to goav1's column-major level scratch
// ABI: levels[col*(height+TX_PAD_HOR)+row].
func coeffInitLevelsArch(coeffs []int16, scanWidth int, scanHeight int, levels []uint8, scratchLen int) bool {
	if !cpu.Detected.NEON {
		return false
	}
	clear(levels[:scratchLen])
	switch scanHeight {
	case 4:
		coeffInitLevels4NEONAsm(&coeffs[0], &levels[0], uintptr(scanWidth))
	case 8:
		coeffInitLevels8NEONAsm(&coeffs[0], &levels[0], uintptr(scanWidth))
	case 16:
		coeffInitLevels16NEONAsm(&coeffs[0], &levels[0], uintptr(scanWidth))
	case 32:
		coeffInitLevels32NEONAsm(&coeffs[0], &levels[0], uintptr(scanWidth))
	default:
		return false
	}
	return true
}
