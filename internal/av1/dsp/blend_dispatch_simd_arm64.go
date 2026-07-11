// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego && goexperiment.simd

package dsp

import "github.com/thesyncim/goav1/internal/av1/dsp/cpu"

// init binds the Go-native SIMD BlendA64Mask kernel under GOEXPERIMENT=simd,
// replacing the NEON asm binding (blend_dispatch_arm64.go is excluded by
// !goexperiment.simd). The SIMD kernel handles the 8-bit non-subsampled fast
// path and defers everything else to the scalar reference. See SIMD_PORT.md.
func init() {
	if cpu.Detected.NEON {
		blendA64MaskImpl = blendA64MaskSIMD
		return
	}
	blendA64MaskImpl = blendA64MaskPureGo
}
