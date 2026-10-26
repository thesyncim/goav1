// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build !(goexperiment.simd && (arm64 || amd64) && !purego)

package transform

func forwardDCT16x16N4Col(buf *[4 * 16]int32, residual *[256]int16) {
	forwardDCT16x16N4ColPureGo(buf, residual)
}

func forwardDCT32x32N4Col(buf *[8 * 32]int32, residual *[1024]int16) {
	forwardDCT32x32N4ColPureGo(buf, residual)
}
