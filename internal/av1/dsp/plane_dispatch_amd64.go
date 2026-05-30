// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64 && !purego

package dsp

import "github.com/thesyncim/goav1/internal/av1/dsp/cpu"

// init binds the AVX2 AddResidualPlaneBlock inner loop on amd64 builds that
// include hand-written assembly. The assignment happens exactly once, before
// any decoder goroutine starts, so the steady-state cost is a single indirect
// call. The AVX2 variant is selected when the CPU advertises AVX2; the pure-Go
// reference is the fallback.
func init() {
	if cpu.Detected.AVX2 {
		addResidualPlaneBlockImpl = addResidualPlaneBlockAVX2
		return
	}
	addResidualPlaneBlockImpl = addResidualPlaneBlockPureGo
}
