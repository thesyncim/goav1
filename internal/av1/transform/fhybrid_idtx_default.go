//go:build !arm64 || purego

package transform

func forwardBlock8x8IDTXImpl(coeff []int32, coeffStride int, residual []int16, residualStride int, scratch []int32) {
	forwardBlock8x8IDTXPureGo(coeff, coeffStride, residual, residualStride, scratch)
}
