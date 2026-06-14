//go:build arm64 && !purego

package transform

import "unsafe"

//go:noescape
func fadstDCT8x8NEONAsm(ctx *fdct8x8NEONCtx)

//go:noescape
func fdctADST8x8NEONAsm(ctx *fdct8x8NEONCtx)

//go:noescape
func fadstADST8x8NEONAsm(ctx *fdct8x8NEONCtx)

//go:noescape
func fidtx8x8NEONAsm(ctx *fdct8x8NEONCtx)

func forwardBlock8x8ADSTDCTImpl(coeff []int32, coeffStride int, residual []int16, residualStride int, scratch []int32) {
	forwardBlock8x8ADSTDCTNEON(coeff, coeffStride, residual, residualStride, scratch)
}

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

func forwardBlock8x8DCTADSTImpl(coeff []int32, coeffStride int, residual []int16, residualStride int, scratch []int32) {
	forwardBlock8x8DCTADSTNEON(coeff, coeffStride, residual, residualStride, scratch)
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

func forwardBlock8x8ADSTADSTImpl(coeff []int32, coeffStride int, residual []int16, residualStride int, scratch []int32) {
	forwardBlock8x8ADSTADSTNEON(coeff, coeffStride, residual, residualStride, scratch)
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

func forwardBlock8x8IDTXImpl(coeff []int32, coeffStride int, residual []int16, residualStride int, scratch []int32) {
	forwardBlock8x8IDTXNEON(coeff, coeffStride, residual, residualStride, scratch)
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
