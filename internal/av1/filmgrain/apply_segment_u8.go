// SPDX-License-Identifier: BSD-2-Clause
// See LICENSE for the BSD-2-Clause grant.

package filmgrain

// applyGrainSegmentU8 adds scaled grain to one 8-bit run for the scalar block
// path (partial right-edge blocks, and every block without arm64 SIMD).
func applyGrainSegmentU8(dst, src []uint8, scale []uint16, grain []int16, scalingShift, minValue, maxValue int) {
	for i := range dst {
		noise := roundPowerOfTwo(int(scale[i])*int(grain[i]), scalingShift)
		dst[i] = uint8(clipInt(int(src[i])+noise, minValue, maxValue))
	}
}
