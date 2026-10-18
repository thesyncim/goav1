// SPDX-License-Identifier: BSD-2-Clause
// See LICENSE for the BSD-2-Clause grant.

package filmgrain

func applyGrainSegmentU8PureGo(dst, src []uint8, scale []uint16, grain []int16, scalingShift, minValue, maxValue int) {
	for i := range dst {
		noise := roundPowerOfTwo(int(scale[i])*int(grain[i]), scalingShift)
		dst[i] = uint8(clipInt(int(src[i])+noise, minValue, maxValue))
	}
}
