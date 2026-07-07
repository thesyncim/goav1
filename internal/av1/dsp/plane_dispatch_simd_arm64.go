// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego && goexperiment.simd

package dsp

import "github.com/thesyncim/goav1/internal/av1/dsp/cpu"

// init binds the Go-native SIMD kernels under GOEXPERIMENT=simd (Go 1.27+),
// replacing the hand-written NEON asm bindings from plane_dispatch_arm64.go
// (which is excluded by !goexperiment.simd). Kernels not yet ported to
// Go-native SIMD fall back to the NEON asm binding, which still compiles under
// tip. See SIMD_PORT.md.
func init() {
	if cpu.Detected.NEON {
		addResidualPlaneBlockImpl = addResidualPlaneBlockSIMD
		addRawTransformPlaneBlockImpl = addRawTransformPlaneBlockNEON // not yet ported
		return
	}
	addResidualPlaneBlockImpl = addResidualPlaneBlockPureGo
	addRawTransformPlaneBlockImpl = addRawTransformPlaneBlockPureGo
}
