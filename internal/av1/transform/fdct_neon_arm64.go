//go:build arm64 && !purego && !goexperiment.simd

package transform

import "unsafe"

// fdct8x8NEONCtx carries the kernel arguments; offsets are mirrored by
// #define in fdct_neon_arm64.s. Strides are in elements.
type fdct8x8NEONCtx struct {
	In        unsafe.Pointer
	InStride  int64
	Out       unsafe.Pointer
	OutStride int64
}

//go:noescape
func fdct8x8NEONAsm(ctx *fdct8x8NEONCtx)

// The NEON 8x8 kernel uses narrow intermediates. Keep the public int16 input
// contract by routing wider residuals to the int64 scalar reference.
var forwardDCT8x8Impl = forwardDCT8x8NEONGuarded
var forwardDCT8x8Trusted8BitImpl = forwardDCT8x8NEON

func forwardDCT8x8NEONGuarded(coeff []int32, coeffStride int, residual []int16, residualStride int) {
	if !residualFitsMagnitude(residual, residualStride, 8, 8, 255) {
		forwardDCT8x8PureGo(coeff, coeffStride, residual, residualStride)
		return
	}
	forwardDCT8x8NEON(coeff, coeffStride, residual, residualStride)
}

func forwardDCT8x8NEON(coeff []int32, coeffStride int, residual []int16, residualStride int) {
	ctx := fdct8x8NEONCtx{
		In:        unsafe.Pointer(&residual[0]),
		InStride:  int64(residualStride),
		Out:       unsafe.Pointer(&coeff[0]),
		OutStride: int64(coeffStride),
	}
	fdct8x8NEONAsm(&ctx)
}

// The default build has no Go SIMD backend; use the scalar reference here.
var forwardDCT4x4Impl = forwardDCT4x4PureGo
var forwardDCT4x4Trusted8BitImpl = forwardDCT4x4PureGo
