// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

package filmgrain

// NEON-accelerated film-grain overlap two-tap blend. The .s file implements the
// eight-wide inner loop: it sign-widens the prev/cur grain lanes to 32 bits,
// forms prev*prevWeight+cur*curWeight, applies the rounding-shift-right that
// reproduces roundPowerOfTwo(v, 5) exactly, clamps to [grainMin, grainMax] and
// narrows back to 16 bits.
//
// Bit-exactness with blendGrainRowPureGo:
//   - v = int32(prev)*prevWeight + int32(cur)*curWeight is computed in 32-bit
//     lanes. grain fits int16 and prevWeight+curWeight is 44 or 45, so v never
//     overflows int32.
//   - roundPowerOfTwo(v, 5) == (v + (1<<4)) >> 5 with an arithmetic right shift;
//     SRSHR #5 adds the same rounding constant then arithmetically shifts, which
//     matches Go's >> on a signed int (same identity the build_scale kernel
//     relies on for SRSHR #2).
//   - clipInt(v, grainMin, grainMax) is reproduced with SMAX(grainMin) /
//     SMIN(grainMax) in 32-bit lanes before the narrow; the clamped result lies
//     in the grain range and fits int16, so the final XTN narrow is exact.
//
// The Go wrapper handles the (<8)-lane tail with the scalar reference, so the
// asm only runs for full eight-sample groups.

// blendGrainRowNEONCtx is the asm calling context. Field order and sizes are
// part of the ABI shared with blend_overlap_neon_arm64.s; do not reorder.
type blendGrainRowNEONCtx struct {
	dst  *int16
	prev *int16
	cur  *int16

	groups uintptr // number of full 8-sample groups

	prevWeight int32 // broadcast to 4 dwords
	curWeight  int32 // broadcast to 4 dwords
	grainMin   int32 // broadcast to 4 dwords
	grainMax   int32 // broadcast to 4 dwords
}

//go:noescape
func blendGrainRowNEONAsm(ctx *blendGrainRowNEONCtx)

func blendGrainRowNEON(dst []int16, prev []int16, cur []int16, prevWeight int, curWeight int, grainMin int, grainMax int) {
	n := len(dst)
	groups := n / 8
	if groups > 0 {
		ctx := blendGrainRowNEONCtx{
			dst:        &dst[0],
			prev:       &prev[0],
			cur:        &cur[0],
			groups:     uintptr(groups),
			prevWeight: int32(prevWeight),
			curWeight:  int32(curWeight),
			grainMin:   int32(grainMin),
			grainMax:   int32(grainMax),
		}
		blendGrainRowNEONAsm(&ctx)
	}
	for i := groups * 8; i < n; i++ {
		v := roundPowerOfTwo(int(prev[i])*prevWeight+int(cur[i])*curWeight, 5)
		dst[i] = int16(clipInt(v, grainMin, grainMax))
	}
}
