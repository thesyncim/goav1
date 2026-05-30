// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64 && !purego

package filmgrain

// AVX2-accelerated film-grain apply kernel. The .s file implements the
// eight-wide inner loop: it widens the gathered scale (unsigned) and grain
// (signed) lanes to 32 bits, forms the product, applies the rounding-shift that
// reproduces roundPowerOfTwo exactly, adds the widened source sample, clamps to
// [minValue, maxValue], and narrows back to 16 bits.
//
// Bit-exactness with applyGrainSegmentPureGo:
//   - prod = int32(scale)*int32(grain) is computed in 32-bit lanes via VPMULLD.
//     scale is in [0,255] and grain fits int16, so the product never overflows
//     int32.
//   - roundPowerOfTwo(prod, shift) for shift in [8,11] is (prod + (1<<(shift-1)))
//     >> shift with an arithmetic right shift; the asm adds the rounding bias
//     then uses VPSRAD by the scalar shift count, matching Go's >> on a signed
//     int.
//   - clipInt(int(src)+noise, min, max) is reproduced with VPMAXSD(min)/
//     VPMINSD(max) in 32-bit lanes before the narrow, so no intermediate value
//     is truncated.
//   - the result lies in [min,max] within [0,4095], so the final VPACKSSDW
//     (signed) narrow is exact: every lane is non-negative and below 0x8000, so
//     signed saturation is a no-op.
//
// The Go wrapper handles the (<8)-lane tail with the scalar reference, so the
// asm only runs for full eight-sample groups.

// applyGrainSegmentAVX2Ctx is the asm calling context. Field order and sizes
// are part of the ABI shared with apply_segment_avx2_amd64.s; do not reorder.
type applyGrainSegmentAVX2Ctx struct {
	dst   *uint16
	src   *uint16
	scale *uint16
	grain *int16

	groups uintptr // number of full 8-sample groups

	roundBias int32 // 1 << (scalingShift-1), broadcast to 8 dwords
	shift     int32 // scalingShift (arithmetic right shift count)
	minValue  int32 // broadcast to 8 dwords
	maxValue  int32 // broadcast to 8 dwords
}

//go:noescape
func applyGrainSegmentAVX2Asm(ctx *applyGrainSegmentAVX2Ctx)

// cpuid is the CPUID instruction wrapper (see cpuid_amd64.s). The cpu package
// reports an SSE2-only baseline and does not run CPUID itself, so this package
// probes AVX2 directly.
//
//go:noescape
func cpuid(eaxArg, ecxArg uint32) (eax, ebx, ecx, edx uint32)

// detectAVX2 reports whether the host CPU advertises AVX2 and OSXSAVE-backed
// YMM state. It checks: max-leaf >= 7, CPUID.1:ECX.OSXSAVE (bit 27), XCR0 SSE
// and AVX state bits (1 and 2), and CPUID.7:EBX.AVX2 (bit 5). XGETBV is read via
// the xgetbv0 asm helper.
func detectAVX2() bool {
	maxLeaf, _, _, _ := cpuid(0, 0)
	if maxLeaf < 7 {
		return false
	}
	_, _, ecx1, _ := cpuid(1, 0)
	const osxsave = 1 << 27
	if ecx1&osxsave == 0 {
		return false
	}
	xcr0 := xgetbv0()
	const xcr0SSE = 1 << 1
	const xcr0AVX = 1 << 2
	if xcr0&(xcr0SSE|xcr0AVX) != (xcr0SSE | xcr0AVX) {
		return false
	}
	_, ebx7, _, _ := cpuid(7, 0)
	const avx2Bit = 1 << 5
	return ebx7&avx2Bit != 0
}

//go:noescape
func xgetbv0() uint32

func applyGrainSegmentAVX2(dst []uint16, src []uint16, scale []uint16, grain []int16, scalingShift int, minValue int, maxValue int) {
	n := len(dst)
	groups := n / 8
	if groups > 0 {
		ctx := applyGrainSegmentAVX2Ctx{
			dst:       &dst[0],
			src:       &src[0],
			scale:     &scale[0],
			grain:     &grain[0],
			groups:    uintptr(groups),
			roundBias: int32(1) << (scalingShift - 1),
			shift:     int32(scalingShift),
			minValue:  int32(minValue),
			maxValue:  int32(maxValue),
		}
		applyGrainSegmentAVX2Asm(&ctx)
	}
	for i := groups * 8; i < n; i++ {
		noise := roundPowerOfTwo(int(scale[i])*int(grain[i]), scalingShift)
		dst[i] = uint16(clipInt(int(src[i])+noise, minValue, maxValue))
	}
}
