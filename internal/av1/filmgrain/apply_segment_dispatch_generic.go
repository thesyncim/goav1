// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build purego || !goexperiment.simd || (!amd64 && !arm64)

package filmgrain

// applyGrainSegment is the grain apply entry point on builds without the Go
// SIMD kernels: the default (non-GOEXPERIMENT=simd) build, purego builds, and
// architectures the SIMD path does not cover. It calls the pure-Go reference
// directly; keeping it a concrete call preserves the zero-alloc property of the
// apply path.
func applyGrainSegment(dst []uint16, src []uint16, scale []uint16, grain []int16, scalingShift int, minValue int, maxValue int) {
	applyGrainSegmentPureGo(dst, src, scale, grain, scalingShift, minValue, maxValue)
}
