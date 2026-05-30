// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build (!amd64 && !arm64) || (arm64 && purego)

package filmgrain

// applyGrainSegment is the grain apply entry point on architectures the
// dispatcher does not special-case, and on the arm64 purego build where the
// NEON asm is excluded. It calls the pure-Go reference directly; keeping it a
// concrete call preserves the zero-alloc property of the apply path.
func applyGrainSegment(dst []uint16, src []uint16, scale []uint16, grain []int16, scalingShift int, minValue int, maxValue int) {
	applyGrainSegmentPureGo(dst, src, scale, grain, scalingShift, minValue, maxValue)
}
