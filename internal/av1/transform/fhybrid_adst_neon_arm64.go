//go:build arm64 && !purego

package transform

import "unsafe"

//go:noescape
func fadstDCT8x8NEONAsm(ctx *fdct8x8NEONCtx)

//go:noescape
func fdctADST8x8NEONAsm(ctx *fdct8x8NEONCtx)

//go:noescape
func fadstADST8x8NEONAsm(ctx *fdct8x8NEONCtx)

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

func forwardBlock8x8ADSTDCTNEONGuarded(coeff []int32, coeffStride int, residual []int16, residualStride int, scratch []int32) {
	if !residualFitsMagnitude(residual, residualStride, 8, 8, 255) {
		forwardBlock8x8ADSTDCTPureGo(coeff, coeffStride, residual, residualStride, scratch)
		return
	}
	forwardBlock8x8ADSTDCTNEON(coeff, coeffStride, residual, residualStride, scratch)
}

func forwardBlock8x8DCTADSTNEONGuarded(coeff []int32, coeffStride int, residual []int16, residualStride int, scratch []int32) {
	if !residualFitsMagnitude(residual, residualStride, 8, 8, 255) {
		forwardBlock8x8DCTADSTPureGo(coeff, coeffStride, residual, residualStride, scratch)
		return
	}
	forwardBlock8x8DCTADSTNEON(coeff, coeffStride, residual, residualStride, scratch)
}

func forwardBlock8x8ADSTADSTNEONGuarded(coeff []int32, coeffStride int, residual []int16, residualStride int, scratch []int32) {
	if !residualFitsMagnitude(residual, residualStride, 8, 8, 255) {
		forwardBlock8x8ADSTADSTPureGo(coeff, coeffStride, residual, residualStride, scratch)
		return
	}
	forwardBlock8x8ADSTADSTNEON(coeff, coeffStride, residual, residualStride, scratch)
}
