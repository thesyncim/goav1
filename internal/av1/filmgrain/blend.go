// Ported from libaom: av1/decoder/grain_synthesis.c
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

package filmgrain

const (
	LumaLegalMin      = 16
	LumaLegalMax      = 235
	ChromaLegalMax    = 240
	ChromaIdentityMax = 235
)

func blendLumaOverlap(previous int16, current int16, offset int, bitDepth uint8) int16 {
	previousWeight, currentWeight := 27, 17
	if offset == 1 {
		previousWeight, currentWeight = 17, 27
	}
	v := roundPowerOfTwo(int(previous)*previousWeight+int(current)*currentWeight, 5)
	grainMin := -(1 << (bitDepth - 1))
	grainMax := (1 << (bitDepth - 1)) - 1
	return int16(clipInt(v, grainMin, grainMax))
}

func applyLumaSample(orig uint16, grain int16, lut []uint8, bitDepth uint8, scalingShift uint8, restrictedRange bool) uint16 {
	scale := scaleLUT(lut, int(orig), bitDepth)
	noise := roundPowerOfTwo(int(scale)*int(grain), int(scalingShift))
	minValue := 0
	maxValue := (256 << (bitDepth - 8)) - 1
	if restrictedRange {
		minValue = LumaLegalMin << (bitDepth - 8)
		maxValue = LumaLegalMax << (bitDepth - 8)
	}
	return uint16(clipInt(int(orig)+noise, minValue, maxValue))
}
