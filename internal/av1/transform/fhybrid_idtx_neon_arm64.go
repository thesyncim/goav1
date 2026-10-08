//go:build arm64 && !purego && !goexperiment.simd

package transform

import "unsafe"

// forwardBlock8x8*Impl are the 8x8 hybrid dispatch slots for the plain arm64
// build: the hand-written NEON asm kernels. The GOEXPERIMENT=simd build binds
// these slots to Go-native SIMD kernels in fhybrid_gosimd_arm64.go instead.
var forwardBlock8x8ADSTDCTImpl = forwardBlock8x8ADSTDCTNEONGuarded
var forwardBlock8x8DCTADSTImpl = forwardBlock8x8DCTADSTNEONGuarded
var forwardBlock8x8ADSTADSTImpl = forwardBlock8x8ADSTADSTNEONGuarded
var forwardBlock8x8IDTXImpl = forwardBlock8x8IDTXNEONGuarded

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

func forwardBlock8x8IDTXNEONGuarded(coeff []int32, coeffStride int, residual []int16, residualStride int, scratch []int32) {
	if !residualFitsMagnitude(residual, residualStride, 8, 8, 255) {
		forwardBlock8x8IDTXPureGo(coeff, coeffStride, residual, residualStride, scratch)
		return
	}
	forwardBlock8x8IDTXNEON(coeff, coeffStride, residual, residualStride, scratch)
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
