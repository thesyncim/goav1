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

// forwardDCT8x8Impl dispatches the 8x8 forward DCT; the NEON kernel is
// bit-exact with the portable reference for 8-bit residual ranges.
var forwardDCT8x8Impl = forwardDCT8x8NEON

func forwardDCT8x8NEON(coeff []int32, coeffStride int, residual []int16, residualStride int) {
	ctx := fdct8x8NEONCtx{
		In:        unsafe.Pointer(&residual[0]),
		InStride:  int64(residualStride),
		Out:       unsafe.Pointer(&coeff[0]),
		OutStride: int64(coeffStride),
	}
	fdct8x8NEONAsm(&ctx)
}

//go:noescape
func fdct4x4NEONAsm(ctx *fdct8x8NEONCtx)

// forwardDCT4x4Impl dispatches the 4x4 forward DCT; the NEON kernel is
// bit-exact with the portable reference for 8-bit residual ranges.
var forwardDCT4x4Impl = forwardDCT4x4NEON

func forwardDCT4x4NEON(coeff []int32, coeffStride int, residual []int16, residualStride int) {
	ctx := fdct8x8NEONCtx{
		In:        unsafe.Pointer(&residual[0]),
		InStride:  int64(residualStride),
		Out:       unsafe.Pointer(&coeff[0]),
		OutStride: int64(coeffStride),
	}
	fdct4x4NEONAsm(&ctx)
}
