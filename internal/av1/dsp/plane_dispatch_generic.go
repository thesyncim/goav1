// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build !goexperiment.simd || purego

package dsp

// init binds the pure-Go AddResidualPlaneBlock and AddRawTransformPlaneBlock
// inner loops wherever no Go SIMD variant is selected: the default build (no
// GOEXPERIMENT=simd), purego builds, and architectures without archsimd support.
func init() {
	addResidualPlaneBlockImpl = addResidualPlaneBlockPureGo
	addRawTransformPlaneBlockImpl = addRawTransformPlaneBlockPureGo
}
