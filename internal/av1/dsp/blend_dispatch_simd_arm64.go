// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego && goexperiment.simd

package dsp

import "github.com/thesyncim/goav1/internal/av1/dsp/cpu"

// init keeps BlendA64Mask on the measured NEON implementation in SIMD builds.
func init() {
	if cpu.Detected.NEON {
		blendA64MaskImpl = blendA64MaskNEON
		return
	}
	blendA64MaskImpl = blendA64MaskPureGo
}
