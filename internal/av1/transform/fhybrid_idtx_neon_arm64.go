//go:build arm64 && !purego && !goexperiment.simd

package transform

import "unsafe"

// forwardBlock8x8*Impl are the 8x8 hybrid dispatch slots for the plain arm64
// build: the hand-written NEON asm kernels. The GOEXPERIMENT=simd build binds
// these slots to Go-native SIMD kernels in fhybrid_gosimd_arm64.go instead.
var forwardBlock8x8ADSTDCTImpl = forwardBlock8x8ADSTDCTNEON
var forwardBlock8x8DCTADSTImpl = forwardBlock8x8DCTADSTNEON
var forwardBlock8x8ADSTADSTImpl = forwardBlock8x8ADSTADSTNEON
var forwardBlock8x8IDTXImpl = forwardBlock8x8IDTXNEON

//go:noescape
func fadstDCT8x8NEONAsm(ctx *fdct8x8NEONCtx)

//go:noescape
func fdctADST8x8NEONAsm(ctx *fdct8x8NEONCtx)

//go:noescape
func fadstADST8x8NEONAsm(ctx *fdct8x8NEONCtx)

//go:noescape
func fidtx8x8NEONAsm(ctx *fdct8x8NEONCtx)

func forwardBlock8x8ADSTDCTNEON(coeff []int32, coeffStride int, residual []int16, residualStride int, scratch []int32) {
	_ = scratch[63]
	ctx := fdct8x8NEONCtx{
		In:        unsafe.Pointer(&residual[0]),
		InStride:  int64(residualStride),
		Out:       unsafe.Pointer(&coeff[0]),
		OutStride: int64(coeffStride),
	}
	fadstDCT8x8NEONAsm(&ctx)
}

func forwardBlock8x8DCTADSTNEON(coeff []int32, coeffStride int, residual []int16, residualStride int, scratch []int32) {
	_ = scratch[63]
	ctx := fdct8x8NEONCtx{
		In:        unsafe.Pointer(&residual[0]),
		InStride:  int64(residualStride),
		Out:       unsafe.Pointer(&coeff[0]),
		OutStride: int64(coeffStride),
	}
	fdctADST8x8NEONAsm(&ctx)
}

func forwardBlock8x8ADSTADSTNEON(coeff []int32, coeffStride int, residual []int16, residualStride int, scratch []int32) {
	_ = scratch[63]
	ctx := fdct8x8NEONCtx{
		In:        unsafe.Pointer(&residual[0]),
		InStride:  int64(residualStride),
		Out:       unsafe.Pointer(&coeff[0]),
		OutStride: int64(coeffStride),
	}
	fadstADST8x8NEONAsm(&ctx)
}

func forwardBlock8x8IDTXNEON(coeff []int32, coeffStride int, residual []int16, residualStride int, scratch []int32) {
	_ = scratch[63]
	ctx := fdct8x8NEONCtx{
		In:        unsafe.Pointer(&residual[0]),
		InStride:  int64(residualStride),
		Out:       unsafe.Pointer(&coeff[0]),
		OutStride: int64(coeffStride),
	}
	fidtx8x8NEONAsm(&ctx)
}
