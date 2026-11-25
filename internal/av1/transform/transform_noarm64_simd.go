// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build !(goexperiment.simd && arm64 && !purego)

package transform

func forwardDCT16x16N4(coeff *[256]int32, residual *[256]int16) {
	var buf [4 * 16]int32
	forwardDCT16x16N4Col(&buf, residual)
	fwd16x16N4Rows(coeff, &buf, 0xf)
}

func forwardDCT32x32N4(coeff *[1024]int32, residual *[1024]int16) {
	var buf [8 * 32]int32
	forwardDCT32x32N4Col(&buf, residual)
	fwd32x32N4Rows(coeff, &buf, 0xff)
}

func forwardDCT16x16N4RowSkip(coeff *[256]int32, residual *[256]int16, thr0, thr int32) uint8 {
	var buf [4 * 16]int32
	forwardDCT16x16N4Col(&buf, residual)
	mask := fwdRowSkipMaskRows(buf[:], 16, 4, thr0, thr, fwdRowSkipSlack16)
	if mask == 0 {
		return 0
	}
	fwd16x16N4Rows(coeff, &buf, mask)
	return mask
}

func forwardDCT32x32N4RowSkip(coeff *[1024]int32, residual *[1024]int16, thr0, thr int32) uint8 {
	var buf [8 * 32]int32
	forwardDCT32x32N4Col(&buf, residual)
	mask := fwdRowSkipMaskRows(buf[:], 32, 8, thr0, thr, fwdRowSkipSlack32)
	if mask == 0 {
		return 0
	}
	fwd32x32N4Rows(coeff, &buf, mask)
	return mask
}

func fwd16x16N4Rows(coeff *[256]int32, buf *[4 * 16]int32, mask uint8) {
	fwd16x16N4RowsPureGo(coeff, buf, mask)
}

func fwd32x32N4Rows(coeff *[1024]int32, buf *[8 * 32]int32, mask uint8) {
	fwd32x32N4RowsPureGo(coeff, buf, mask)
}

// forwardDCTRect8BitTrusted has no vector kernel in this build; the scalar
// rectangles run instead.
func forwardDCTRect8BitTrusted(coeff []int32, coeffStride int, residual []int16, residualStride int, size Size) bool {
	return false
}

func forwardBlock8x8Hybrid8BitResidualTrusted(coeff []int32, coeffStride int, residual []int16, residualStride int, scratch []int32, typ Type) bool {
	return forwardBlock8x8HybridPureGo(coeff, coeffStride, residual, residualStride, scratch, typ)
}

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

// stageTransposeClamp stages the rows x cols rectangle of the column-major
// coefficient block into row-major scratch lines, clamping to [lo, hi] (rect2
// applies the sqrt2 scale first). This build binds the scalar reference; the
// goexperiment.simd arm64 build overrides it with a 4x4-block SIMD transpose
// (hybrid_stage_gosimd_arm64.go).
func stageTransposeClamp(scratch []int32, width int, coeff []int32, coeffStride int, rows int, cols int, rect2 bool, lo int32, hi int32) {
	stageTransposeClampScalar(scratch, width, coeff, coeffStride, rows, cols, rect2, lo, hi)
}
