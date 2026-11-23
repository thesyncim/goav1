// Ported from SVT-AV1 v4.1.0:
//   Source/Lib/ASM_NEON/highbd_fwd_txfm_neon.c (highbd_fdct64_N2_x4_neon,
//   svt_av1_fwd_txfm2d_64x64_neon)
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package transform

// The 64x64 forward DCT_DCT for arm64 runs the straight-line N2 passes that
// fwdgen derives from the scalar fwdDCT64 (fdct_net64_gen_arm64.go; the amd64
// kernel keeps highbd_fdct64_N2_x4_neon's staged form). The column pass loads
// rows 0..62 as the eight samples from column g (those reads end inside the
// next row; the last row takes the in-row load), rounds by shift[1] = 2 and
// stores its 32 kept outputs transposed (bufT[c*32+k] is output k of column
// c); the row pass reads four rows of bufT, rounds by shift[2] = 2 and writes
// coeff[k*32+r].

// forwardDCT64x64SIMD is svt_av1_fwd_txfm2d_64x64's DCT_DCT case for 8-bit
// residuals, writing the coded 32x32 coefficients in coeff[k*coeffStride+r]
// layout (the frequency column k, row r) like ForwardDCT64x64.
func forwardDCT64x64SIMD(coeff []int32, coeffStride int, residual []int16, residualStride int) {
	var bufT [64 * 32]int32
	for g := 0; g < 64; g += fwdLanes {
		fwdColDCT64N2(bufT[:], g, residual, residualStride)
	}
	if coeffStride == 32 {
		for h := 0; h < 32; h += fwdLanes {
			fwdRowDCT64N2(coeff, bufT[:], h)
		}
		return
	}
	var buf [32 * 32]int32
	for h := 0; h < 32; h += fwdLanes {
		fwdRowDCT64N2(buf[:], bufT[:], h)
	}
	for k := range 32 {
		copy(coeff[k*coeffStride:k*coeffStride+32], buf[k*32:])
	}
}
