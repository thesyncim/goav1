// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64 && !purego

package prediction

import "github.com/thesyncim/goav1/internal/av1/dsp/cpu"

// init binds the intra static predictor, CfL and DC dispatch slots on amd64.
// The AVX2 variants are bound only when cpu.Detected.AVX2 is set. The current
// cpu package ships a conservative stub that reports SSE2 but not AVX2 (no
// CPUID detection yet), so on stock builds these slots keep the pure-Go
// reference; the AVX2 kernels still compile and are differentially validated
// against the reference by the amd64 tests. Once CPUID detection lands in the
// cpu package this binding activates with no change here.
func init() {
	_ = cpu.Detected // ensure cpu package init runs before this point
	if cpu.Detected.AVX2 {
		predictPaethImpl = predictPaethAVX2
		predictSmoothImpl = predictSmoothAVX2
		predictSmoothVerticalImpl = predictSmoothVerticalAVX2
		predictSmoothHorizontalImpl = predictSmoothHorizontalAVX2
		applyCFLImpl = applyCFLAVX2
		sumSamplesImpl = sumSamplesAVX2
		subsampleLuma8Impl = subsampleLuma8AVX2
		dirRowInterp8Impl = dirRowInterp8AVX2
		dirAboveRun8Impl = dirAboveRun8AVX2
		dirLeftCol8Impl = dirLeftCol8AVX2
		return
	}
	predictPaethImpl = predictPaethPureGo
	predictSmoothImpl = predictSmoothPureGo
	predictSmoothVerticalImpl = predictSmoothVerticalPureGo
	predictSmoothHorizontalImpl = predictSmoothHorizontalPureGo
	applyCFLImpl = applyCFLPureGo
	sumSamplesImpl = sumSamplesPureGo
}
