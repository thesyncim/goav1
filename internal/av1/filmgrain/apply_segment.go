// Ported from libaom: av1/decoder/grain_synthesis.c
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

package filmgrain

// applyGrainSegmentPureGo is the canonical bit-exact reference for the per-pixel
// grain "apply" kernel. Every tuned variant MUST match it sample for sample,
// including the rounding term in roundPowerOfTwo and the clamp in clipInt. The
// kernel only handles the common non-overlap interior of a grain block, where
// the destination, source, scale and grain runs are all contiguous; the overlap
// edges stay scalar in apply.go.
//
// For lane i in [0,n):
//
//	prod  = int32(scale[i]) * int32(grain[i])
//	noise = (prod + (1<<(scalingShift-1))) >> scalingShift   // arithmetic
//	dst[i] = clip(int(src[i]) + noise, minValue, maxValue)
//
// scale holds the per-sample scaling-LUT values (0..255) already gathered by
// the caller; keeping the LUT gather in the caller lets one kernel serve every
// bit depth byte-exactly. scalingShift is in [8,11]; minValue/maxValue are the
// bit-depth- and range-dependent clamp bounds. dst and src may alias.
//
// The architecture-specific entry point applyGrainSegment (see
// apply_segment_*.go) is a concrete function — not a func-pointer indirection —
// so escape analysis can see through it and the caller's scale scratch buffer
// stays on the stack (the apply path is zero-alloc).
func applyGrainSegmentPureGo(dst []uint16, src []uint16, scale []uint16, grain []int16, scalingShift int, minValue int, maxValue int) {
	n := len(dst)
	for i := 0; i < n; i++ {
		noise := roundPowerOfTwo(int(scale[i])*int(grain[i]), scalingShift)
		dst[i] = uint16(clipInt(int(src[i])+noise, minValue, maxValue))
	}
}
