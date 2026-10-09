// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego && goexperiment.simd

package dsp

import "github.com/thesyncim/goav1/internal/av1/dsp/cpu"

// init binds the Go-native SIMD AddResidualPlaneBlock and AddRawTransformPlaneBlock
// inner loops on arm64 SIMD builds. NEON is mandatory on every arm64 target Go
// supports; the pure-Go references are the fallback.
func init() {
	if cpu.Detected.NEON {
		addResidualPlaneBlockImpl = addResidualPlaneBlockSIMD
		addRawTransformPlaneBlockImpl = addRawTransformPlaneBlockSIMD
		return
	}
	addResidualPlaneBlockImpl = addResidualPlaneBlockPureGo
	addRawTransformPlaneBlockImpl = addRawTransformPlaneBlockPureGo
}
