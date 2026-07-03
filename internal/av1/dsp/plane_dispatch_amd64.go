// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64 && !purego

package dsp

import "github.com/thesyncim/goav1/internal/av1/dsp/cpu"

// init binds the AVX2 AddResidualPlaneBlock and AddRawTransformPlaneBlock inner
// loops on amd64 builds that include hand-written assembly. The assignment
// happens exactly once, before any decoder goroutine starts, so the steady-state
// cost is a single indirect call. The AVX2 variants are selected when the CPU
// advertises AVX2; the pure-Go references are the fallback.
func init() {
	if cpu.Detected.AVX2 {
		addResidualPlaneBlockImpl = addResidualPlaneBlockAVX2
		addRawTransformPlaneBlockImpl = addRawTransformPlaneBlockAVX2
		return
	}
	addResidualPlaneBlockImpl = addResidualPlaneBlockPureGo
	addRawTransformPlaneBlockImpl = addRawTransformPlaneBlockPureGo
}
