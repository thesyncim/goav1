// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

package dsp

// NEON-accelerated BlendA64Mask inner loop. The .s file implements the two
// dominant mask layouts (no subsampling and 2x2 subX&&subY) for widths that
// are a multiple of eight; every other shape (width < 8 and the single-axis
// subsampling layouts) routes to the pure-Go reference. Strides arrive in
// samples and are converted to bytes for the assembly.
//
// Bit-exactness with blendA64MaskPureGo: the kernel computes the same rounded
// weighted average and reports the same valid/invalid verdict. On invalid
// input the kernel may write part of dst before reporting (the reference stops
// at the first bad lane); callers only consult the returned error, so the
// observable behaviour is identical.

//go:noescape
func blendA64NoSubNEONAsm(dst *uint16, dstStride uintptr, src0 *uint16, src0Stride uintptr, src1 *uint16, src1Stride uintptr, mask *uint8, maskStride uintptr, maxv uint32, groups uintptr, height uintptr) uint32

//go:noescape
func blendA64SubXYNEONAsm(dst *uint16, dstStride uintptr, src0 *uint16, src0Stride uintptr, src1 *uint16, src1Stride uintptr, mask *uint8, maskStride uintptr, maxv uint32, groups uintptr, height uintptr) uint32

func blendA64MaskNEON(a blendA64MaskArgs) bool {
	groups := a.width >> 3
	// Single-axis subsampling and sub-vector widths fall back to the portable
	// path; they are rare and the reference is already correct there.
	if groups == 0 || (a.subX != a.subY) {
		return blendA64MaskPureGo(a)
	}

	var violation uint32
	if a.subX && a.subY {
		violation = blendA64SubXYNEONAsm(
			&a.dst[0], uintptr(a.dstStride*2),
			&a.src0[0], uintptr(a.src0Stride*2),
			&a.src1[0], uintptr(a.src1Stride*2),
			&a.mask[0], uintptr(a.maskStride),
			uint32(a.max), uintptr(groups), uintptr(a.height),
		)
	} else {
		violation = blendA64NoSubNEONAsm(
			&a.dst[0], uintptr(a.dstStride*2),
			&a.src0[0], uintptr(a.src0Stride*2),
			&a.src1[0], uintptr(a.src1Stride*2),
			&a.mask[0], uintptr(a.maskStride),
			uint32(a.max), uintptr(groups), uintptr(a.height),
		)
	}
	return violation == 0
}
