// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package prediction

import "github.com/thesyncim/goav1/internal/av1/dsp/cpu"

// init binds the Go-native SIMD PAETH and SMOOTH predictors as the dispatch
// kernels under the goexperiment.simd build (the NEON asm binding in
// intra_dispatch_arm64.go is excluded there via !goexperiment.simd). Every other
// intra kernel in the dispatch keeps its hand-written NEON asm, which is still
// compiled in this build (intra_neon_arm64.go / intra_kernels_neon_arm64.go are
// tagged arm64 && !purego); this init re-binds them so nothing regresses to the
// scalar reference. When NEON is somehow unavailable the pure-Go references stay.
// Every binding is byte-identical to the *PureGo references.
func init() {
	_ = cpu.Detected // ensure cpu package init runs before this point
	if !cpu.Detected.NEON {
		predictPaethImpl = predictPaethPureGo
		predictSmoothImpl = predictSmoothPureGo
		predictSmoothVerticalImpl = predictSmoothVerticalPureGo
		predictSmoothHorizontalImpl = predictSmoothHorizontalPureGo
		sumSamplesImpl = sumSamplesPureGo
		applyCFLImpl = applyCFLPureGo
		subsampleLuma8Impl = subsampleLuma8PureGo
		dirRowInterp8Impl = dirRowInterp8PureGo
		dirAboveRun8Impl = dirAboveRun8PureGo
		dirLeftCol8Impl = dirLeftCol8PureGo
		return
	}
	// PAETH and the three SMOOTH variants: Go-native SIMD.
	predictPaethImpl = predictPaethSIMD
	predictSmoothImpl = predictSmoothSIMD
	predictSmoothVerticalImpl = predictSmoothVerticalSIMD
	predictSmoothHorizontalImpl = predictSmoothHorizontalSIMD
	// Every other dispatched kernel: keep the NEON asm (no regression).
	sumSamplesImpl = sumSamplesNEON
	applyCFLImpl = applyCFLNEON
	subsampleLuma8Impl = subsampleLuma8NEON
	dirRowInterp8Impl = dirRowInterp8NEON
	dirAboveRun8Impl = dirAboveRun8NEON
	dirLeftCol8Impl = dirLeftCol8NEON
	predictFilterIntra8Impl = predictFilterIntraBlockDirect8NEON
	predictFilterIntra16Impl = predictFilterIntraBlockDirect16NEON
}
