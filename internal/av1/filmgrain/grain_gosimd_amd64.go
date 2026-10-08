// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && amd64 && !purego

package filmgrain

import "simd/archsimd"

// grainSIMDAvailable reports whether the Go SIMD grain kernels may run. They
// use only AVX2-level encodings, so they need AVX2 at run time.
func grainSIMDAvailable() bool {
	return archsimd.X86.AVX2()
}

// grainPack8 narrows eight int32 values (lo holds lanes 0..3, hi holds lanes
// 4..7; every value is already clamped into the int16 range) to uint16 lanes in
// order with one VPACKSSDW. The AVX-512 SaturateToInt16 form is avoided: it
// needs AVX-512 and these kernels must run on AVX2-only CPUs.
func grainPack8(lo, hi archsimd.Int32x4) archsimd.Uint16x8 {
	return lo.SaturateToInt16Concat(hi).ConvertToUint16()
}
