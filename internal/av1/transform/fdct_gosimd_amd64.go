// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && amd64 && !purego

package transform

import "simd/archsimd"

// The 8x8 forward DCT is Go SIMD on AVX2 hosts. Without AVX2 the portable
// bindings from fdct_neon_off.go stay in place.
func init() {
	if archsimd.X86.AVX2() {
		forwardDCT8x8Impl = forwardDCT8x8SIMDGuarded
		forwardDCT8x8Trusted8BitImpl = forwardDCT8x8SIMD
	}
}
