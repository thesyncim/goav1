// SPDX-License-Identifier: BSD-2-Clause
// See LICENSE for the BSD-2-Clause grant.

//go:build purego || !goexperiment.simd || !(arm64 || amd64)

package filmgrain

// fgConstsU8 carries no state for the scalar block path.
type fgConstsU8 struct{}

func (k *fgConstsU8) init(scalingShift int, minValue, maxValue int, sx int, sy int) {}

func (k *fgConstsU8) initChroma(params *ChromaRowParams) {}

func fgLumaBlockU8(dst, src []uint8, grain []int16, scaling []uint8, params LumaRowParams, offsets [2][2]uint8, blockX, blockWidth, xStart, yStart int, k *fgConstsU8) {
	applyLumaBlockU8(dst, src, grain, scaling, params, offsets, blockX, blockWidth, xStart, yStart)
}

func fgChromaBlockU8(dst, src, luma []uint8, grain []int16, scaling []uint8, params ChromaRowParams, shiftX, shiftY int, offsets [2][2]uint8, blockX, blockWidth, xStart, yStart int, k *fgConstsU8) {
	applyChromaBlockU8(dst, src, luma, grain, scaling, params, shiftX, shiftY, offsets, blockX, blockWidth, xStart, yStart)
}
