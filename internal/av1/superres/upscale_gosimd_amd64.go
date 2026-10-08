// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && amd64 && !purego

package superres

import "simd/archsimd"

// init binds the Go SIMD super-res row kernel on amd64 when the CPU advertises
// AVX2. Otherwise the pure-Go reference stays bound (the default value of
// upscaleRowImpl).
func init() {
	if archsimd.X86.AVX2() {
		upscaleRowImpl = upscaleRowSIMD
	}
}

// superresPack8 narrows eight int32 pixel sums (lo holds pixels 0..3, hi holds
// pixels 4..7; every value is already in [0, 4095]) to uint16 lanes in pixel
// order with one VPACKSSDW. The AVX-512 SaturateToInt16 form is avoided: it
// needs AVX-512 and this kernel must run on AVX2-only CPUs.
func superresPack8(lo, hi archsimd.Int32x4) archsimd.Uint16x8 {
	return lo.SaturateToInt16Concat(hi).ConvertToUint16()
}
