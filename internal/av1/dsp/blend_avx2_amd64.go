// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64 && !purego

package dsp

// AVX2-accelerated BlendA64Mask inner loop. The .s file implements the two
// dominant mask layouts (no subsampling and 2x2 subX&&subY) for widths that are
// a multiple of eight; every other shape (width < 8 and the single-axis
// subsampling layouts) routes to the pure-Go reference. Strides arrive in
// samples and are converted to bytes for the assembly.
//
// Bit-exactness with blendA64MaskPureGo (see blend.go):
//   - the per-pixel weight m is gathered as 8 u16 lanes (the mask byte for
//     no-sub; for subX&&subY the round2 of the four 2x2 neighbours, summed with
//     VPMADDUBSW pairwise then VPADDW across rows and rounded with
//     VPADDW(2)/VPSRLW #2), matching blendMaskSample exactly.
//   - dst = (m*s0 + (64-m)*s1 + 32) >> 6 is computed in u32 lanes (VPMOVZXWD,
//     VPMULLD, VPADDD, VPSRLD #6), matching blendA64 / roundPowerOfTwoPositive.
//   - the valid/invalid verdict is computed with the range-safe unsigned
//     greater-than test VPSUBUSW(x, limit) != 0 for s0>max, s1>max and m>64,
//     OR-accumulated across every lane. Unlike a signed compare this is correct
//     for the whole uint16 range, so out-of-range samples at or above 0x8000
//     are still rejected. The reference stops at the first bad lane; the kernel
//     finishes the block before reporting, but callers consult only the
//     returned error, so the observable behaviour is identical.

//go:noescape
func blendA64NoSubAVX2Asm(dst *uint16, dstStride uintptr, src0 *uint16, src0Stride uintptr, src1 *uint16, src1Stride uintptr, mask *uint8, maskStride uintptr, maxv uint32, groups uintptr, height uintptr) uint32

//go:noescape
func blendA64SubXYAVX2Asm(dst *uint16, dstStride uintptr, src0 *uint16, src0Stride uintptr, src1 *uint16, src1Stride uintptr, mask *uint8, maskStride uintptr, maxv uint32, groups uintptr, height uintptr) uint32

func blendA64MaskAVX2(a blendA64MaskArgs) bool {
	groups := a.width >> 3
	// Single-axis subsampling and sub-vector widths fall back to the portable
	// path; they are rare and the reference is already correct there.
	if groups == 0 || (a.subX != a.subY) {
		return blendA64MaskPureGo(a)
	}

	var violation uint32
	if a.subX && a.subY {
		violation = blendA64SubXYAVX2Asm(
			&a.dst[0], uintptr(a.dstStride*2),
			&a.src0[0], uintptr(a.src0Stride*2),
			&a.src1[0], uintptr(a.src1Stride*2),
			&a.mask[0], uintptr(a.maskStride),
			uint32(a.max), uintptr(groups), uintptr(a.height),
		)
	} else {
		violation = blendA64NoSubAVX2Asm(
			&a.dst[0], uintptr(a.dstStride*2),
			&a.src0[0], uintptr(a.src0Stride*2),
			&a.src1[0], uintptr(a.src1Stride*2),
			&a.mask[0], uintptr(a.maskStride),
			uint32(a.max), uintptr(groups), uintptr(a.height),
		)
	}
	return violation == 0
}
