// Ported from SVT-AV1 v4.1.0:
//   Source/Lib/ASM_NEON/highbd_fwd_txfm_neon.c (highbd_fdct16_N4_xn_neon,
//   highbd_fdct32_N4_xn_neon: the N4 column passes across columns)
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && amd64 && !purego

package transform

func forwardDCT16x16N4Col(buf *[4 * 16]int32, residual *[256]int16) {
	forwardDCT16x16N4ColSIMD(buf, residual)
	fwdVecDone()
}

func forwardDCT32x32N4Col(buf *[8 * 32]int32, residual *[1024]int16) {
	forwardDCT32x32N4ColSIMD(buf, residual)
	fwdVecDone()
}

// forwardDCT16x16N4ColSIMD is forwardDCT16x16N4ColPureGo over fwdLanes
// columns per vector (fdct_n4_gen_amd64.go).
func forwardDCT16x16N4ColSIMD(buf *[4 * 16]int32, residual *[256]int16) {
	for g := 0; g < 16; g += fwdLanes {
		fwdColDCT16N4(buf[:], g, residual[:], 16)
	}
}

// forwardDCT32x32N4ColSIMD is forwardDCT32x32N4ColPureGo over fwdLanes
// columns per vector.
func forwardDCT32x32N4ColSIMD(buf *[8 * 32]int32, residual *[1024]int16) {
	for g := 0; g < 32; g += fwdLanes {
		fwdColDCT32N4(buf[:], g, residual[:], 32)
	}
}
