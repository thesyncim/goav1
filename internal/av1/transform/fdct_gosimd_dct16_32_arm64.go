// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package transform

// The arm64 networks (fdct_net16_32_gen_arm64.go) load their twiddles off
// one table pointer and spell each butterfly out as intrinsics, so every
// operand is a single load at a constant offset and no inline-mark NOPs
// separate the operations. Their column passes load rows 0..n-2 as the eight
// samples from column g and keep the low four (those reads end inside the
// next row; only the last row takes the in-row load of fwdLoadResAt) and
// store their outputs transposed, so the row passes run in place.

// fwdTw12 is the cos_bit 12 twiddle table of the 16- and 32-point networks.
var fwdTw12 = newFwd64Twiddles(&fwdCospi12, 12)

// forwardDCT16x16SIMD is the 16x16 DCT_DCT kernel (forwardDCT16x16PureGo).
// The column pass stores its rows transposed, so coeff[c*16+k] holds
// column-pass row k of column c, and the row pass then transforms each group
// of four rows in place.
func forwardDCT16x16SIMD(coeff []int32, coeffStride int, residual []int16, residualStride int) {
	if coeffStride == 16 {
		forwardDCT16x16InPlace((*[256]int32)(coeff), residual, residualStride)
		return
	}
	var buf [256]int32
	forwardDCT16x16InPlace(&buf, residual, residualStride)
	for k := range 16 {
		copy(coeff[k*coeffStride:k*coeffStride+16], buf[k*16:])
	}
}

func forwardDCT16x16InPlace(c *[256]int32, residual []int16, residualStride int) {
	for g := 0; g < 16; g += fwdLanes {
		fwdColDCT16(c[:], g, residual, residualStride)
	}
	for h := 0; h < 16; h += fwdLanes {
		fwdRowDCT16(c[:], h)
	}
}

// forwardDCT32x32SIMD is the 32x32 DCT_DCT kernel (forwardDCT32x32PureGo),
// laid out like forwardDCT16x16SIMD.
func forwardDCT32x32SIMD(coeff []int32, coeffStride int, residual []int16, residualStride int) {
	if coeffStride == 32 {
		forwardDCT32x32InPlace((*[1024]int32)(coeff), residual, residualStride)
		return
	}
	var buf [1024]int32
	forwardDCT32x32InPlace(&buf, residual, residualStride)
	for k := range 32 {
		copy(coeff[k*coeffStride:k*coeffStride+32], buf[k*32:])
	}
}

func forwardDCT32x32InPlace(c *[1024]int32, residual []int16, residualStride int) {
	for g := 0; g < 32; g += fwdLanes {
		fwdColDCT32(c[:], g, residual, residualStride)
	}
	for h := 0; h < 32; h += fwdLanes {
		fwdRowDCT32(c[:], h)
	}
}
