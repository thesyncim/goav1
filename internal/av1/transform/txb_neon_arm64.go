// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

//go:build arm64 && !purego

package transform

//go:noescape
func txbInitLevels8x8NEONAsm(coeffs *int16, levels *uint8)

func txbPrep8x8Levels2DNEON(coeffs *[64]int16, levels *[256]uint8, absLevels *[64]uint16, eob int) TXB8x8PrepResult {
	txbInitLevels8x8NEONAsm(&coeffs[0], &levels[0])
	return txbPrep8x8Summary(coeffs, absLevels, eob)
}
