// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build !goexperiment.simd || !arm64 || purego

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
