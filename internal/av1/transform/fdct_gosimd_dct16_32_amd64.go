// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && amd64 && !purego

package transform

// The amd64 drivers run the column pass (fdct_net16_32_gen_amd64.go) into a
// row-major buffer, transpose it and run the row pass.

// forwardDCT16x16SIMD is the 16x16 DCT_DCT kernel (forwardDCT16x16PureGo).
func forwardDCT16x16SIMD(coeff []int32, coeffStride int, residual []int16, residualStride int) {
	var buf, bufT [256]int32
	for g := 0; g < 16; g += fwdLanes {
		fwdColDCT16(buf[:], g, residual, residualStride)
	}
	fwdTranspose(bufT[:], buf[:], 16)
	for h := 0; h < 16; h += fwdLanes {
		fwdRowDCT16(coeff, coeffStride, bufT[:], h)
	}
}

// forwardDCT32x32SIMD is the 32x32 DCT_DCT kernel (forwardDCT32x32PureGo).
func forwardDCT32x32SIMD(coeff []int32, coeffStride int, residual []int16, residualStride int) {
	var buf, bufT [1024]int32
	for g := 0; g < 32; g += fwdLanes {
		fwdColDCT32(buf[:], g, residual, residualStride)
	}
	fwdTranspose(bufT[:], buf[:], 32)
	for h := 0; h < 32; h += fwdLanes {
		fwdRowDCT32(coeff, coeffStride, bufT[:], h)
	}
}
