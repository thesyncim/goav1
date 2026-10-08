// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package prediction

import "github.com/thesyncim/goav1/internal/av1/dsp/cpu"

// init binds the Go-native SIMD intra kernels under the goexperiment.simd build:
// PAETH and the three SMOOTH predictors (intra_static_gosimd_arm64.go), the DC
// sum, CfL apply and subsampling, the directional interpolation and filter intra.
// Every binding remains byte-identical to its pure-Go reference.
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
	predictPaethImpl = predictPaethSIMD
	predictSmoothImpl = predictSmoothSIMD
	predictSmoothVerticalImpl = predictSmoothVerticalSIMD
	predictSmoothHorizontalImpl = predictSmoothHorizontalSIMD
	sumSamplesImpl = sumSamplesSIMD
	applyCFLImpl = applyCFLSIMD
	subsampleLuma8Impl = subsampleLuma8SIMD
	subsampleLuma16Impl = subsampleLuma16SIMD
	subtractCFLAverageImpl = subtractCFLAverageSIMD
	dirRowInterp8Impl = dirRowInterp8SIMD
	dirAboveRun8Impl = dirAboveRun8SIMD
	dirLeftCol8Impl = dirLeftCol8SIMD
	predictFilterIntra8Impl = predictFilterIntraBlockDirect8SIMD
	predictFilterIntra16Impl = predictFilterIntraBlockDirect16SIMD
}
