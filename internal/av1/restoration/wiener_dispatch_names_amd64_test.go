// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && amd64 && !purego

package restoration

import "simd/archsimd"

// wienerSIMDDetected and sgrSIMDDetected report whether init bound the SIMD
// Wiener and self-guided kernels (AVX2 is required by the amd64 kernels).
func wienerSIMDDetected() bool { return archsimd.X86.AVX2() }

func sgrSIMDDetected() bool { return archsimd.X86.AVX2() }

// sgrSIMDBoxsumKernel is the box-sum kernel bound alongside the SIMD
// self-guided drivers.
var sgrSIMDBoxsumKernel = boxsumSIMD

var (
	wantWienerHorizontalU8Impl any = wienerHorizontalU8SIMDChecked
	wantWienerVerticalU8Impl   any = wienerVerticalU8SIMD
)
