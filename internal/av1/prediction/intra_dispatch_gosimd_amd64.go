// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && amd64 && !purego

package prediction

import "simd/archsimd"

// init binds the Go-native SIMD intra kernels on amd64. The gate is the
// archsimd feature probe (not cpu.Detected) so the SIMD path activates under
// GOEXPERIMENT=simd on any AVX2 machine, while every other kernel keeps the
// pure-Go reference.
func init() {
	if !archsimd.X86.AVX2() {
		return
	}
	predictPaethImpl = predictPaethSIMD
	predictSmoothImpl = predictSmoothSIMD
	predictSmoothVerticalImpl = predictSmoothVerticalSIMD
	predictSmoothHorizontalImpl = predictSmoothHorizontalSIMD
	sumSamplesImpl = sumSamplesSIMD
	applyCFLImpl = applyCFLSIMD
	subsampleLuma8Impl = subsampleLuma8SIMD
	dirRowInterp8Impl = dirRowInterp8SIMD
	dirAboveRun8Impl = dirAboveRun8SIMD
	dirLeftCol8Impl = dirLeftCol8SIMD
	predictFilterIntra8Impl = predictFilterIntraBlockDirect8SIMD
}
