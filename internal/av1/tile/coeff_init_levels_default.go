// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

//go:build (!arm64 && !amd64) || purego

package tile

func coeffInitLevelsArch(coeffs []int16, scanWidth int, scanHeight int, levels []uint8, scratchLen int) bool {
	return false
}
