// SPDX-License-Identifier: BSD-2-Clause
// See LICENSE for the BSD-2-Clause grant.

//go:build purego || !goexperiment.simd || !arm64

package filmgrain

func lumaGrainARLag3(grain []int16, params *LumaGrainParams, roundingOffset, coeffShift, grainMin, grainMax int) {
	generateLumaGrainARLag3(grain, params, roundingOffset, coeffShift, grainMin, grainMax)
}

func chromaGrainARLag3(grain []int16, luma []int16, params *ChromaGrainParams, width, height, roundingOffset, coeffShift, grainMin, grainMax int) {
	generateChromaGrainARLag3(grain, luma, params, width, height, roundingOffset, coeffShift, grainMin, grainMax)
}
