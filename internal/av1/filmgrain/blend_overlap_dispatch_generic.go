// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build !arm64 || purego

package filmgrain

// blendGrainRow is the overlap two-tap blend entry point on architectures
// without a dedicated kernel (and on the arm64 purego build where the NEON asm
// is excluded). It calls the pure-Go reference directly; keeping it a concrete
// call preserves the zero-alloc property of the apply path (the caller's blend
// scratch buffer stays on the stack).
func blendGrainRow(dst []int16, prev []int16, cur []int16, prevWeight int, curWeight int, grainMin int, grainMax int) {
	blendGrainRowPureGo(dst, prev, cur, prevWeight, curWeight, grainMin, grainMax)
}
