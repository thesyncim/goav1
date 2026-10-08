//go:build arm64 && !purego && !goexperiment.simd

package transform

import "unsafe"

// fdct16NEONCtx carries the kernel arguments; offsets are mirrored by
// #define in fdct16_neon_arm64.s. Buf points at caller-owned 16x16 int32
// scratch for the column-pass output.
type fdct16NEONCtx struct {
	In        unsafe.Pointer
	InStride  int64
	Out       unsafe.Pointer
	OutStride int64
	Buf       unsafe.Pointer
}

//go:noescape
func fdct16x16NEONAsm(ctx *fdct16NEONCtx)

// The NEON 16x16 kernel uses narrow intermediates. Keep the public int16 input
// contract by routing wider residuals to the int64 scalar reference.
var forwardDCT16x16Impl = forwardDCT16x16NEONGuarded

func forwardDCT16x16NEONGuarded(coeff []int32, coeffStride int, residual []int16, residualStride int) {
	if !residualFitsMagnitude(residual, residualStride, 16, 16, 255) {
		forwardDCT16x16PureGo(coeff, coeffStride, residual, residualStride)
		return
	}
	forwardDCT16x16NEON(coeff, coeffStride, residual, residualStride)
}

func forwardDCT16x16NEON(coeff []int32, coeffStride int, residual []int16, residualStride int) {
	var buf [256]int32
	ctx := fdct16NEONCtx{
		In:        unsafe.Pointer(&residual[0]),
		InStride:  int64(residualStride),
		Out:       unsafe.Pointer(&coeff[0]),
		OutStride: int64(coeffStride),
		Buf:       unsafe.Pointer(&buf[0]),
	}
	fdct16x16NEONAsm(&ctx)
}
