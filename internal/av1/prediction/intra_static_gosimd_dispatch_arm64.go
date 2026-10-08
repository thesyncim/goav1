// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package prediction

import "github.com/thesyncim/goav1/internal/av1/dsp/cpu"

// init uses Go-native SIMD for measured CfL subsampling and average subtraction
// under the goexperiment.simd build. Paeth, Smooth, and CfL apply retain their
// NEON kernels because the current Go-SIMD candidates are slower on representative
// blocks. Every binding remains byte-identical to its pure-Go reference.
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
		subsampleLuma16Impl = subsampleLuma16PureGo
		subtractCFLAverageImpl = subtractCFLAveragePureGo
		dirRowInterp8Impl = dirRowInterp8PureGo
		dirAboveRun8Impl = dirAboveRun8PureGo
		dirLeftCol8Impl = dirLeftCol8PureGo
		return
	}
	// Keep the measured-losing static predictors and CfL apply on NEON.
	predictPaethImpl = predictPaethNEON
	predictSmoothImpl = predictSmoothNEON
	predictSmoothVerticalImpl = predictSmoothVerticalNEON
	predictSmoothHorizontalImpl = predictSmoothHorizontalNEON
	applyCFLImpl = applyCFLNEON
	// CfL subsampling and average subtraction use Go-native SIMD.
	subsampleLuma8Impl = subsampleLuma8SIMD
	subsampleLuma16Impl = subsampleLuma16SIMD
	subtractCFLAverageImpl = subtractCFLAverageSIMD
	// Every other dispatched kernel: keep the NEON asm (no regression).
	sumSamplesImpl = sumSamplesNEON
	dirRowInterp8Impl = dirRowInterp8NEON
	dirAboveRun8Impl = dirAboveRun8NEON
	dirLeftCol8Impl = dirLeftCol8NEON
	predictFilterIntra8Impl = predictFilterIntraBlockDirect8SIMD
	predictFilterIntra16Impl = predictFilterIntraBlockDirect16SIMD
}
