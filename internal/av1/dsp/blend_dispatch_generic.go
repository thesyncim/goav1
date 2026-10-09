// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build !goexperiment.simd || purego

package dsp

// init binds the pure-Go BlendA64Mask inner loop wherever no Go SIMD variant is
// selected: the default build (no GOEXPERIMENT=simd), purego builds, and
// architectures without archsimd support.
func init() {
	blendA64MaskImpl = blendA64MaskPureGo
}
