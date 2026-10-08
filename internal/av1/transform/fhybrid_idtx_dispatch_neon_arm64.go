// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego && !goexperiment.simd

package transform

// The default arm64 build uses the retained hybrid NEON kernels. IDTX has no
// assembly implementation after its measured Go SIMD replacement, so the
// default build uses the scalar reference.
var forwardBlock8x8ADSTDCTImpl = forwardBlock8x8ADSTDCTNEONGuarded
var forwardBlock8x8DCTADSTImpl = forwardBlock8x8DCTADSTNEONGuarded
var forwardBlock8x8ADSTADSTImpl = forwardBlock8x8ADSTADSTNEONGuarded
var forwardBlock8x8IDTXImpl = forwardBlock8x8IDTXPureGo

func forwardBlock8x8Hybrid8BitResidualTrusted(coeff []int32, coeffStride int, residual []int16, residualStride int, scratch []int32, typ Type) bool {
	switch typ {
	case TypeADSTDCT:
		forwardBlock8x8ADSTDCTNEON(coeff, coeffStride, residual, residualStride, scratch)
	case TypeDCTADST:
		forwardBlock8x8DCTADSTNEON(coeff, coeffStride, residual, residualStride, scratch)
	case TypeADSTADST:
		forwardBlock8x8ADSTADSTNEON(coeff, coeffStride, residual, residualStride, scratch)
	case TypeIDTX:
		forwardBlock8x8IDTXPureGo(coeff, coeffStride, residual, residualStride, scratch)
	default:
		return false
	}
	return true
}
