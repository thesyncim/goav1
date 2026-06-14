//go:build !arm64 || purego

package transform

func forwardBlock8x8ADSTDCTImpl(coeff []int32, coeffStride int, residual []int16, residualStride int, scratch []int32) {
	forwardBlock8x8ADSTDCTPureGo(coeff, coeffStride, residual, residualStride, scratch)
}

func forwardBlock8x8DCTADSTImpl(coeff []int32, coeffStride int, residual []int16, residualStride int, scratch []int32) {
	forwardBlock8x8DCTADSTPureGo(coeff, coeffStride, residual, residualStride, scratch)
}

func forwardBlock8x8ADSTADSTImpl(coeff []int32, coeffStride int, residual []int16, residualStride int, scratch []int32) {
	forwardBlock8x8ADSTADSTPureGo(coeff, coeffStride, residual, residualStride, scratch)
}

func forwardBlock8x8IDTXImpl(coeff []int32, coeffStride int, residual []int16, residualStride int, scratch []int32) {
	forwardBlock8x8IDTXPureGo(coeff, coeffStride, residual, residualStride, scratch)
}
