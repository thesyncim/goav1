// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64 && !purego

package filmgrain

import "github.com/thesyncim/goav1/internal/av1/dsp/cpu"

// applyGrainSegmentUseAVX2 selects the AVX2 kernel. It is resolved once, at
// package init, before any decoder goroutine starts. It must not be mutated
// concurrently with live decoding.
//
// As on arm64 (see apply_segment_dispatch_arm64.go) the dispatch is a plain
// bool rather than a func pointer so applyGrainSegment below stays a concrete
// (non-indirect) call: escape analysis can then see that neither implementation
// retains its slice arguments, which keeps the caller's scale scratch buffer on
// the stack and the apply path zero-alloc.
var applyGrainSegmentUseAVX2 bool

// init binds the architecture-best grain apply kernel on amd64. AVX2 is not
// part of the GOAMD64=v1 baseline the Go toolchain assumes, so it is probed at
// runtime via CPUID (see cpuid_amd64.s); when present the kernel routes through
// the hand-written AVX2 asm, otherwise it keeps the pure-Go reference. All bit
// depths share one code path because the math is identical once the caller has
// gathered the scaling-LUT values and supplied the clamp bounds.
//
// cpu.Detected.AVX2 records the same result for diagnostics. The cpu package's
// own init reports a conservative SSE2-only baseline (it does not yet run
// CPUID), so this package performs the CPUID probe itself and publishes the
// AVX2 bit back into cpu.Detected without disturbing the other fields.
func init() {
	avx2 := detectAVX2()
	applyGrainSegmentUseAVX2 = avx2
	cpu.Detected.AVX2 = avx2
}

func applyGrainSegment(dst []uint16, src []uint16, scale []uint16, grain []int16, scalingShift int, minValue int, maxValue int) {
	if applyGrainSegmentUseAVX2 {
		applyGrainSegmentAVX2(dst, src, scale, grain, scalingShift, minValue, maxValue)
		return
	}
	applyGrainSegmentPureGo(dst, src, scale, grain, scalingShift, minValue, maxValue)
}
