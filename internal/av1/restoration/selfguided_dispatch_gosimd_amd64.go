// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && amd64 && !purego

package restoration

import "simd/archsimd"

// init binds the Go-native AVX2 self-guided kernels on amd64 when AVX2 is
// available: the box sums and the per-pixel blend run eight lanes at a time. Without
// AVX2 the dispatch keeps the pure-Go reference. The data-dependent LUT gather
// inside calculateIntermediate stays scalar in both paths. The assignment happens
// once, before any decoder goroutine starts, so the steady-state cost is a single
// indirect call.
func init() {
	if archsimd.X86.AVX2() {
		boxsumImpl = boxsumSIMD
		selfguidedImpl = selfguidedSIMD
		selfguidedFastImpl = selfguidedFastSIMD
		return
	}
	boxsumImpl = boxsum
	selfguidedImpl = selfguided
	selfguidedFastImpl = selfguidedFast
}

// sgrSIMDBoxsumKernel is the box-sum kernel bound alongside the SIMD self-guided
// drivers.
var sgrSIMDBoxsumKernel = boxsumSIMD

// sgrSIMDDetected reports whether the SIMD self-guided kernels are bound on this
// host (AVX2 is required by the amd64 kernels).
func sgrSIMDDetected() bool { return archsimd.X86.AVX2() }
