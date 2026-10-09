// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64 && !purego && goexperiment.simd

package dsp

import "simd/archsimd"

// init binds the Go-native SIMD BlendA64Mask inner loop on amd64 SIMD builds.
// AVX2 is not part of the GOAMD64=v1 baseline, so the SIMD loop is selected only
// when archsimd reports AVX2; the pure-Go reference is the fallback.
func init() {
	if archsimd.X86.AVX2() {
		blendA64MaskImpl = blendA64MaskSIMD
		return
	}
	blendA64MaskImpl = blendA64MaskPureGo
}
