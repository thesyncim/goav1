// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

package filmgrain

import "github.com/thesyncim/goav1/internal/av1/dsp/cpu"

// applyGrainSegmentUseNEON selects the NEON kernel. It is resolved once, at
// package init, before any decoder goroutine starts. It must not be mutated
// concurrently with live decoding.
//
// The dispatch is a plain bool rather than a func pointer so applyGrainSegment
// below stays a concrete (non-indirect) call: escape analysis can then see that
// neither implementation retains its slice arguments, which keeps the caller's
// scale scratch buffer on the stack and the apply path zero-alloc.
var applyGrainSegmentUseNEON bool

// init binds the architecture-best grain apply kernel on arm64. NEON is
// mandatory on every arm64 chip Go runs on; when present the kernel routes
// through the hand-written NEON asm, otherwise it keeps the pure-Go reference.
// All bit depths share one code path because the math is identical once the
// caller has gathered the scaling-LUT values and supplied the clamp bounds.
func init() {
	_ = cpu.Detected // ensure cpu package init runs before this point
	applyGrainSegmentUseNEON = cpu.Detected.NEON
}

func applyGrainSegment(dst []uint16, src []uint16, scale []uint16, grain []int16, scalingShift int, minValue int, maxValue int) {
	if applyGrainSegmentUseNEON {
		applyGrainSegmentNEON(dst, src, scale, grain, scalingShift, minValue, maxValue)
		return
	}
	applyGrainSegmentPureGo(dst, src, scale, grain, scalingShift, minValue, maxValue)
}
