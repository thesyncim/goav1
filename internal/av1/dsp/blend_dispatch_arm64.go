// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego && !goexperiment.simd

package dsp

import "github.com/thesyncim/goav1/internal/av1/dsp/cpu"

// init binds the NEON BlendA64Mask inner loop on arm64 builds that include
// hand-written assembly. NEON is mandatory on every arm64 target Go supports;
// the pure-Go reference is the fallback.
func init() {
	if cpu.Detected.NEON {
		blendA64MaskImpl = blendA64MaskNEON
		return
	}
	blendA64MaskImpl = blendA64MaskPureGo
}
