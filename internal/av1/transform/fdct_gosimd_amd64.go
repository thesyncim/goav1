// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && amd64 && !purego

package transform

import "simd/archsimd"

// The 8x8, 16x16 and 32x32 forward DCTs are Go SIMD on AVX2 hosts. Without
// AVX2 the portable bindings from fdct_neon_off.go stay in place.
func init() {
	if archsimd.X86.AVX2() {
		forwardDCT8x8Impl = forwardDCT8x8SIMDGuarded
		forwardDCT8x8Trusted8BitImpl = forwardDCT8x8SIMD
		forwardDCT16x16Impl = forwardDCT16x16SIMDGuarded
		forwardDCT16x16Trusted8BitImpl = forwardDCT16x16SIMD
		forwardDCT32x32Impl = forwardDCT32x32SIMDGuarded
		forwardDCT32x32Trusted8BitImpl = forwardDCT32x32SIMD
	}
}
