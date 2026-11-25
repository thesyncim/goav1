// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build !(goexperiment.simd && (arm64 || amd64) && !purego)

package tile

func coeffInitLevelsArch(coeffs []int16, scanWidth int, scanHeight int, levels []uint8, scratchLen int) bool {
	return false
}

func coeffNZMapContexts2DFullArch(levels []uint8, size TransformSize, contexts []int8) bool {
	return false
}
