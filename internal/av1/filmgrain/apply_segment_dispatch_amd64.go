// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64

package filmgrain

// applyGrainSegment is the grain apply entry point on amd64. No tuned amd64
// variant ships yet, so it calls the pure-Go reference directly. Keeping it a
// concrete call (not a func pointer) lets escape analysis prove the caller's
// scale scratch buffer does not escape, so the apply path stays zero-alloc.
//
// TODO: add SSE4.1/AVX2 variants here, gated on the matching cpu.Detected flag.
func applyGrainSegment(dst []uint16, src []uint16, scale []uint16, grain []int16, scalingShift int, minValue int, maxValue int) {
	applyGrainSegmentPureGo(dst, src, scale, grain, scalingShift, minValue, maxValue)
}
