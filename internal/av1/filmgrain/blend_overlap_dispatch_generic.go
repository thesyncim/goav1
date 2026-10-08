// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build purego || !goexperiment.simd || !arm64

package filmgrain

// blendGrainRow is the overlap two-tap blend entry point on builds without the
// arm64 Go SIMD kernel (the default build, purego, and every other arch). It
// calls the pure-Go reference directly; keeping it a concrete call preserves the
// zero-alloc property of the apply path (the caller's blend scratch buffer stays
// on the stack).
func blendGrainRow(dst []int16, prev []int16, cur []int16, prevWeight int, curWeight int, grainMin int, grainMax int) {
	blendGrainRowPureGo(dst, prev, cur, prevWeight, curWeight, grainMin, grainMax)
}
