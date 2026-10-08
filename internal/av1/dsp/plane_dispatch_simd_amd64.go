// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64 && !purego && goexperiment.simd

package dsp

import "simd/archsimd"

// init binds the Go-native SIMD AddResidualPlaneBlock and AddRawTransformPlaneBlock
// inner loops on amd64 SIMD builds. AVX2 is not part of the GOAMD64=v1 baseline,
// so the SIMD variants are selected only when archsimd reports AVX2; the pure-Go
// references are the fallback.
func init() {
	if archsimd.X86.AVX2() {
		addResidualPlaneBlockImpl = addResidualPlaneBlockSIMD
		addRawTransformPlaneBlockImpl = addRawTransformPlaneBlockSIMD
		return
	}
	addResidualPlaneBlockImpl = addResidualPlaneBlockPureGo
	addRawTransformPlaneBlockImpl = addRawTransformPlaneBlockPureGo
}
