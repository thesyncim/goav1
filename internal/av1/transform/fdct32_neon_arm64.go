//go:build arm64 && !purego

package transform

import "unsafe"

// fdct32NEONCtx carries the kernel arguments; offsets are mirrored by
// #define in fdct32_neon_arm64.s. Buf points at caller-owned 32x32 int32
// scratch for the column-pass output and Spill at sixteen vectors of
// stage-1 difference staging.
type fdct32NEONCtx struct {
	In        unsafe.Pointer
	InStride  int64
	Out       unsafe.Pointer
	OutStride int64
	Buf       unsafe.Pointer
	Spill     unsafe.Pointer
}

//go:noescape
func fdct32x32NEONAsm(ctx *fdct32NEONCtx)

// forwardDCT32x32Impl dispatches the 32x32 forward DCT; the NEON kernel is
// bit-exact with the portable reference for 8-bit residual ranges.
var forwardDCT32x32Impl = forwardDCT32x32NEON

func forwardDCT32x32NEON(coeff []int32, coeffStride int, residual []int16, residualStride int) {
	var buf [1024]int32
	var spill [64]int32
	ctx := fdct32NEONCtx{
		In:        unsafe.Pointer(&residual[0]),
		InStride:  int64(residualStride),
		Out:       unsafe.Pointer(&coeff[0]),
		OutStride: int64(coeffStride),
		Buf:       unsafe.Pointer(&buf[0]),
		Spill:     unsafe.Pointer(&spill[0]),
	}
	fdct32x32NEONAsm(&ctx)
}
