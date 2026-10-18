// SPDX-License-Identifier: BSD-2-Clause
// See LICENSE for the BSD-2-Clause grant.

//go:build purego || !goexperiment.simd || !arm64

package filmgrain

func applyGrainSegmentU8(dst, src []uint8, scale []uint16, grain []int16, scalingShift, minValue, maxValue int) {
	applyGrainSegmentU8PureGo(dst, src, scale, grain, scalingShift, minValue, maxValue)
}
