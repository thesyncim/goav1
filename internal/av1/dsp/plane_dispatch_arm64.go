// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego && !goexperiment.simd

package dsp

import "github.com/thesyncim/goav1/internal/av1/dsp/cpu"

// init binds the NEON AddResidualPlaneBlock inner loop on arm64 builds that
// include hand-written assembly. The assignment happens exactly once, before
// any decoder goroutine starts, so the steady-state cost is a single indirect
// call. NEON is mandatory on every arm64 target Go supports; the pure-Go
// reference is the fallback.
func init() {
	if cpu.Detected.NEON {
		addResidualPlaneBlockImpl = addResidualPlaneBlockNEON
		addRawTransformPlaneBlockImpl = addRawTransformPlaneBlockNEON
		return
	}
	addResidualPlaneBlockImpl = addResidualPlaneBlockPureGo
	addRawTransformPlaneBlockImpl = addRawTransformPlaneBlockPureGo
}
