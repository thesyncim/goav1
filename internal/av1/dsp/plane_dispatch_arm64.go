// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

package dsp

import "github.com/thesyncim/goav1/internal/av1/dsp/cpu"

// init binds the NEON plane-copy and residual inner loops on arm64 builds that
// include hand-written assembly. The assignments happen exactly once, before
// any decoder goroutine starts, so steady state has no feature-detection
// branches. NEON is mandatory on every arm64 target Go supports; the pure-Go
// references remain the fallback.
func init() {
	if cpu.Detected.NEON {
		copyPlaneBlockDisjointTrustedImpl = copyPlaneBlockDisjointTrustedNEON
		addResidualPlaneBlockImpl = addResidualPlaneBlockNEON
		addRawTransformPlaneBlockImpl = addRawTransformPlaneBlockNEON
		return
	}
	copyPlaneBlockDisjointTrustedImpl = copyPlaneBlockDisjointTrustedPureGo
	addResidualPlaneBlockImpl = addResidualPlaneBlockPureGo
	addRawTransformPlaneBlockImpl = addRawTransformPlaneBlockPureGo
}
