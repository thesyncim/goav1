// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && amd64 && !purego

package transform

// The amd64 cores are the eight-lane scratch-transpose network.
func fwd8DCTCore(coeff *[64]int32, residual *[64]int16) {
	fwd8x8NetPacked(coeff, residual, false, false)
}

func fwd8ADSTDCTCore(coeff *[64]int32, residual *[64]int16) {
	fwd8x8NetPacked(coeff, residual, true, false)
}
