// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package motion

// The interior warp passes (resident 8-tap horizontal, full 8x8 vertical) are
// the hottest pure-Go inter-prediction kernels. warp.go routes them through the
// warp*Dispatch functions, which are selected at build time (see
// warp_dispatch_arm64.go / warp_dispatch_other.go). Build-time (static)
// dispatch — rather than a func-pointer var — is deliberate: the warp scratch
// (warpAffine8Offset's tmp array) is passed down into these calls, and an
// indirect call would defeat escape analysis and heap-allocate that scratch on
// every prediction. The pure-Go references in warp.go remain the canonical
// bit-exact oracle every SIMD variant must reproduce sample for sample.

type warpTmp = [warpedIntermediateRows * warpedIntermediateColumns]int32

// warpHorizResidentOffsInRange reports whether every warpedFilter index the
// resident horizontal pass would select stays inside [0, len(warpedFilter)),
// i.e. whether the scalar path's index clamp (warp.go warpHorizontal8Resident)
// is a no-op for this block. The selected index is
// offs(k,l) = 64 + roundPowerOfTwo(sx4 + beta*(k+4) + alpha*(l+4), 10), which is
// affine (hence monotone) in the sample offsets k+4 in [-3,11] and l+4 in [0,7];
// its extremes therefore sit at the corners. When this returns true the NEON
// kernel can skip the clamp and still match the scalar output bit-for-bit;
// otherwise the caller must fall back to scalar.
func warpHorizResidentOffsInRange(sx4, alpha, beta int) bool {
	loA, hiA := mulRange(beta, -3, 11)
	loB, hiB := mulRange(alpha, 0, 7)
	sxMin := sx4 + loA + loB
	sxMax := sx4 + hiA + hiB
	offsMin := warpedPixelPrecShifts + roundPowerOfTwo(sxMin, warpedDiffPrecBits)
	offsMax := warpedPixelPrecShifts + roundPowerOfTwo(sxMax, warpedDiffPrecBits)
	return offsMin >= 0 && offsMax < len(warpedFilter)
}

// warpVertFullOffsInRange is the vertical twin of warpHorizResidentOffsInRange.
// The full vertical pass selects offs(k,l) = 64 + roundPowerOfTwo(baseSY +
// delta*(k+4) + gamma*(l+4), 10) with k+4 and l+4 both in [0,7]. Unlike the
// horizontal pass the scalar vertical *skips* (leaves the destination pixel
// untouched) when an index falls out of range, so the NEON kernel — which has no
// per-lane skip — must defer to scalar whenever this returns false.
func warpVertFullOffsInRange(baseSY, gamma, delta int) bool {
	loD, hiD := mulRange(delta, 0, 7)
	loG, hiG := mulRange(gamma, 0, 7)
	syMin := baseSY + loD + loG
	syMax := baseSY + hiD + hiG
	offsMin := warpedPixelPrecShifts + roundPowerOfTwo(syMin, warpedDiffPrecBits)
	offsMax := warpedPixelPrecShifts + roundPowerOfTwo(syMax, warpedDiffPrecBits)
	return offsMin >= 0 && offsMax < len(warpedFilter)
}

// mulRange returns the minimum and maximum of step*n for n in [nLo, nHi].
func mulRange(step, nLo, nHi int) (int, int) {
	a := step * nLo
	b := step * nHi
	if a < b {
		return a, b
	}
	return b, a
}
