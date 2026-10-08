// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego && goexperiment.simd

package dsp

import "github.com/thesyncim/goav1/internal/av1/dsp/cpu"

// init binds the Go-native SIMD residual-add kernel for 8-bit blocks while
// leaving measured-loser kernels on their NEON implementations.
func init() {
	if cpu.Detected.NEON {
		addResidualPlaneBlockImpl = addResidualPlaneBlockSIMDDispatch
		addRawTransformPlaneBlockImpl = addRawTransformPlaneBlockNEON
		return
	}
	addResidualPlaneBlockImpl = addResidualPlaneBlockPureGo
	addRawTransformPlaneBlockImpl = addRawTransformPlaneBlockPureGo
}

// addResidualPlaneBlockSIMDDispatch selects the promoted 8-bit kernel and
// keeps high-bit-depth blocks on the existing NEON implementation.
func addResidualPlaneBlockSIMDDispatch(block planeBlock, bytesPerSample int, max uint16, width int, residual []int16, residualStride int) {
	if bytesPerSample == 1 && width >= 16 && width%16 == 0 {
		addResidualPlaneBlockSIMD(block, bytesPerSample, max, width, residual, residualStride)
		return
	}
	addResidualPlaneBlockNEON(block, bytesPerSample, max, width, residual, residualStride)
}
