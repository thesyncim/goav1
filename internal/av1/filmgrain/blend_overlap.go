// Ported from libaom: av1/decoder/grain_synthesis.c
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

package filmgrain

// blendGrainRowPureGo is the canonical bit-exact reference for the two-tap
// overlap blend that apply.go performs along a contiguous run of grain-template
// samples at a horizontal (top) grain-block seam. Every tuned variant MUST match
// it sample for sample, including the rounding term in roundPowerOfTwo and the
// clamp in clipInt. It mirrors blendLumaOverlap / blendChromaOverlap (blend.go,
// apply.go) for a fixed weight pair:
//
//	for i in [0,n):
//	    v      = prev[i]*prevWeight + cur[i]*curWeight
//	    dst[i] = clip(roundPowerOfTwo(v, 5), grainMin, grainMax)
//
// The overlap-row position (blend offset) is constant across a top seam, so the
// weight pair is loop-invariant and only the horizontal (x-contiguous) top-seam
// blend is routed here — exactly as dav1d fuses the fgy/fguv top-overlap blend
// into its grain apply (third_party/upstream/dav1d/src/arm/64/filmgrain.S
// fgy_32x32xn, filmgrain16.S fguv). The narrow left-seam (<=2 wide) and 2x2
// corner blends stay scalar in apply.go.
//
// prevWeight+curWeight is 44 (luma / vertically non-subsampled chroma) or 45
// (vertically subsampled chroma); grain samples fit int16 so prev*w+cur*w never
// overflows int32. grainMin/grainMax are the bit-depth grain clamp bounds
// -(1<<(bd-1)) .. (1<<(bd-1))-1. The caller passes the two source grain runs and
// a separate scratch as dst.
//
// The architecture-specific entry point blendGrainRow (see
// blend_overlap_dispatch_*.go) is a concrete function — not a func-pointer
// indirection — so escape analysis can see through it and the caller's blend
// scratch buffer stays on the stack (the apply path is zero-alloc).
func blendGrainRowPureGo(dst []int16, prev []int16, cur []int16, prevWeight int, curWeight int, grainMin int, grainMax int) {
	n := len(dst)
	for i := 0; i < n; i++ {
		v := roundPowerOfTwo(int(prev[i])*prevWeight+int(cur[i])*curWeight, 5)
		dst[i] = int16(clipInt(v, grainMin, grainMax))
	}
}
