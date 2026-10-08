// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && (arm64 || amd64) && !purego

package dsp

// Go-native SIMD (simd/archsimd) BlendA64Mask kernels shared by arm64 and amd64.
// They cover the two vectorised mask layouts (no subsampling and 2x2
// subsampling) for widths that are a multiple of eight, exactly the shapes the
// retired assembly served; every other shape routes to blendA64MaskPureGo. The
// per-architecture files provide the row loops (blend_gosimd_{arm64,amd64}.go).
//
// Bit-exactness with blendA64MaskPureGo:
//   - the per-sample weight m is widened to u16 lanes; for subX&&subY the four
//     2x2 neighbours are summed and rounded as (sum+2)>>2, matching
//     blendMaskSample.
//   - the weighted average (m*s0 + (64-m)*s1 + 32) >> 6 is computed in u16 lanes
//     when max <= 1023: then the sum is at most 64*max + 32 <= 65504 and cannot
//     wrap. Wider samples (12-bit) are computed in u32 lanes.
//   - the valid/invalid verdict is the running unsigned maximum of the samples
//     (must be <= max) and of the weights (must be <= 64). The reference stops at
//     the first bad lane; the kernel finishes the block before reporting, and
//     callers consult only the returned error.

// blendA64MaskSIMD is the Go SIMD implementation selected by the SIMD
// dispatchers on arm64 and amd64.
func blendA64MaskSIMD(a blendA64MaskArgs) bool {
	groups := a.width >> 3
	// Single-axis subsampling and sub-vector widths fall back to the portable
	// path; they are rare and the reference is already correct there.
	if groups == 0 || (a.subX != a.subY) {
		return blendA64MaskPureGo(a)
	}
	if a.subX {
		return blendA64SubXYSIMD(a, groups)
	}
	return blendA64NoSubSIMD(a, groups)
}

