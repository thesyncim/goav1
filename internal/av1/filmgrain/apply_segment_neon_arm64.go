// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

package filmgrain

// NEON-accelerated film-grain apply kernel. The .s file implements the
// eight-wide inner loop: it widens the gathered scale (unsigned) and grain
// (signed) lanes to 32 bits, forms the product, applies the rounding-shift that
// reproduces roundPowerOfTwo exactly, adds the widened source sample, clamps to
// [minValue, maxValue], and narrows back to 16 bits.
//
// Bit-exactness with applyGrainSegmentPureGo:
//   - prod = int32(scale)*int32(grain) is computed in 32-bit lanes. scale is in
//     [0,255] and grain fits int16, so the product never overflows int32.
//   - roundPowerOfTwo(prod, shift) for shift in [8,11] is (prod + (1<<(shift-1)))
//     >> shift with an arithmetic right shift; the asm adds the rounding bias
//     then uses SSHL by a negative count, matching Go's >> on a signed int.
//   - clipInt(int(src)+noise, min, max) is reproduced with SMAX(min)/SMIN(max)
//     in 32-bit lanes before the narrow, so no intermediate value is truncated.
//   - the result lies in [min,max] within [0,4095], so the final XTN narrow is
//     exact.
//
// The Go wrapper handles the (<8)-lane tail with the scalar reference, so the
// asm only runs for full eight-sample groups.

// applyGrainSegmentNEONCtx is the asm calling context. Field order and sizes
// are part of the ABI shared with apply_segment_neon_arm64.s; do not reorder.
type applyGrainSegmentNEONCtx struct {
	dst   *uint16
	src   *uint16
	scale *uint16
	grain *int16

	groups uintptr // number of full 8-sample groups

	roundBias int64 // 1 << (scalingShift-1)
	negShift  int64 // -scalingShift (for the arithmetic right shift)
	minValue  int64
	maxValue  int64
}

//go:noescape
func applyGrainSegmentNEONAsm(ctx *applyGrainSegmentNEONCtx)

func applyGrainSegmentNEON(dst []uint16, src []uint16, scale []uint16, grain []int16, scalingShift int, minValue int, maxValue int) {
	n := len(dst)
	groups := n / 8
	if groups > 0 {
		ctx := applyGrainSegmentNEONCtx{
			dst:       &dst[0],
			src:       &src[0],
			scale:     &scale[0],
			grain:     &grain[0],
			groups:    uintptr(groups),
			roundBias: int64(1) << (scalingShift - 1),
			negShift:  int64(-scalingShift),
			minValue:  int64(minValue),
			maxValue:  int64(maxValue),
		}
		applyGrainSegmentNEONAsm(&ctx)
	}
	for i := groups * 8; i < n; i++ {
		noise := roundPowerOfTwo(int(scale[i])*int(grain[i]), scalingShift)
		dst[i] = uint16(clipInt(int(src[i])+noise, minValue, maxValue))
	}
}
