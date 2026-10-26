// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && amd64 && !purego

package transform

import "simd/archsimd"

// The 4x4, 8x8, 16x16 and 32x32 forward DCTs are Go SIMD on AVX2 hosts. Without
// AVX2 the portable bindings from fdct_neon_off.go stay in place.
func init() {
	if archsimd.X86.AVX2() {
		forwardDCT4x4Impl = forwardDCT4x4SIMDVZ
		forwardDCT4x4Trusted8BitImpl = forwardDCT4x4SIMDCoreVZ
		forwardDCT8x8Impl = forwardDCT8x8SIMDGuardedVZ
		forwardDCT8x8Trusted8BitImpl = forwardDCT8x8SIMDVZ
		forwardDCT16x16Impl = forwardDCT16x16SIMDGuardedVZ
		forwardDCT16x16Trusted8BitImpl = forwardDCT16x16SIMDVZ
		forwardDCT32x32Impl = forwardDCT32x32SIMDGuardedVZ
		forwardDCT32x32Trusted8BitImpl = forwardDCT32x32SIMDVZ
		forwardDCT64x64Trusted8BitImpl = forwardDCT64x64SIMDVZ
		forwardDCT8x8N4Impl = forwardDCT8x8N4SIMDVZ
	}
}

// The VZ wrappers return with clean upper ymm state: the amd64 kernels run
// 256-bit lanes and the Go compiler emits no vzeroupper, so the legacy-SSE
// code that follows them would stall on Zen 3.

func forwardDCT4x4SIMDVZ(coeff []int32, coeffStride int, residual []int16, residualStride int) {
	forwardDCT4x4SIMD(coeff, coeffStride, residual, residualStride)
	archsimd.ClearAVXUpperBits()
}

func forwardDCT4x4SIMDCoreVZ(coeff []int32, coeffStride int, residual []int16, residualStride int) {
	forwardDCT4x4SIMDCore(coeff, coeffStride, residual, residualStride)
	archsimd.ClearAVXUpperBits()
}

func forwardDCT8x8SIMDGuardedVZ(coeff []int32, coeffStride int, residual []int16, residualStride int) {
	forwardDCT8x8SIMDGuarded(coeff, coeffStride, residual, residualStride)
	archsimd.ClearAVXUpperBits()
}

func forwardDCT8x8SIMDVZ(coeff []int32, coeffStride int, residual []int16, residualStride int) {
	forwardDCT8x8SIMD(coeff, coeffStride, residual, residualStride)
	archsimd.ClearAVXUpperBits()
}

func forwardDCT16x16SIMDGuardedVZ(coeff []int32, coeffStride int, residual []int16, residualStride int) {
	forwardDCT16x16SIMDGuarded(coeff, coeffStride, residual, residualStride)
	archsimd.ClearAVXUpperBits()
}

func forwardDCT16x16SIMDVZ(coeff []int32, coeffStride int, residual []int16, residualStride int) {
	forwardDCT16x16SIMD(coeff, coeffStride, residual, residualStride)
	archsimd.ClearAVXUpperBits()
}

func forwardDCT32x32SIMDGuardedVZ(coeff []int32, coeffStride int, residual []int16, residualStride int) {
	forwardDCT32x32SIMDGuarded(coeff, coeffStride, residual, residualStride)
	archsimd.ClearAVXUpperBits()
}

func forwardDCT32x32SIMDVZ(coeff []int32, coeffStride int, residual []int16, residualStride int) {
	forwardDCT32x32SIMD(coeff, coeffStride, residual, residualStride)
	archsimd.ClearAVXUpperBits()
}

func forwardDCT64x64SIMDVZ(coeff []int32, coeffStride int, residual []int16, residualStride int) {
	forwardDCT64x64SIMD(coeff, coeffStride, residual, residualStride)
	archsimd.ClearAVXUpperBits()
}

func forwardDCT8x8N4SIMDVZ(coeff *[64]int32, residual *[64]int16) {
	forwardDCT8x8N4SIMD(coeff, residual)
	archsimd.ClearAVXUpperBits()
}
