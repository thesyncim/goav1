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

// forwardDCT16x16Impl dispatches the 16x16 forward DCT; the NEON kernel is
// bit-exact with the portable reference for 8-bit residual ranges.
var forwardDCT16x16Impl = forwardDCT16x16NEON

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
