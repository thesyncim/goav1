// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build !(goexperiment.simd && arm64 && !purego)

package transform

func fwd16x16N4Rows(coeff *[256]int32, buf *[4 * 16]int32, mask uint8) {
	fwd16x16N4RowsPureGo(coeff, buf, mask)
}

func fwd32x32N4Rows(coeff *[1024]int32, buf *[8 * 32]int32, mask uint8) {
	fwd32x32N4RowsPureGo(coeff, buf, mask)
}
