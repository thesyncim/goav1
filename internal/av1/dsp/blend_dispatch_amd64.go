// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64 && !purego

package dsp

import "github.com/thesyncim/goav1/internal/av1/dsp/cpu"

// init binds the AVX2 BlendA64Mask inner loop on amd64 builds that include
// hand-written assembly. AVX2 is not part of the GOAMD64=v1 baseline, so it is
// selected only when the shared CPUID detector (see internal/av1/dsp/cpu)
// reports an OS-enabled AVX2 unit; the pure-Go reference is the fallback.
//
// The assignment happens exactly once, before any decoder goroutine starts, so
// the steady-state cost is a single indirect call.
func init() {
	if cpu.Detected.AVX2 {
		blendA64MaskImpl = blendA64MaskAVX2
		return
	}
	blendA64MaskImpl = blendA64MaskPureGo
}
