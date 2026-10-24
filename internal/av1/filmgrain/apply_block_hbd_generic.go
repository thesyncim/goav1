// SPDX-License-Identifier: BSD-2-Clause
// See LICENSE for the BSD-2-Clause grant.

//go:build purego || !goexperiment.simd || !arm64

package filmgrain

// fgConstsHBD carries no state for the scalar block path.
type fgConstsHBD struct{}

func (k *fgConstsHBD) init(bitDepth uint8, scalingShift, minValue, maxValue, sx, sy int) {}

func (k *fgConstsHBD) initChroma(params *ChromaRowParams) {}

func fgLumaBlockHBD(dst, src []uint16, grain []int16, scaling []uint8, table *[ScalingTableSize]uint8, params LumaRowParams, offsets [2][2]uint8, blockX, blockWidth, xStart, yStart int, k *fgConstsHBD) {
	applyLumaBlock(dst, src, grain, scaling, params, offsets, blockX, blockWidth, xStart, yStart)
}

func fgChromaBlockHBD(dst, src, luma []uint16, grain []int16, scaling []uint8, table *[ScalingTableSize]uint8, params ChromaRowParams, shiftX, shiftY int, offsets [2][2]uint8, blockX, blockWidth, xStart, yStart int, k *fgConstsHBD) {
	applyChromaBlock(dst, src, luma, grain, scaling, params, shiftX, shiftY, offsets, blockX, blockWidth, xStart, yStart)
}
