// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package transform

// forwardDCTRect8BitTrusted runs the 2:1 DCT_DCT rectangles 16x8, 8x16, 32x16
// and 16x32 (svt_av1_fwd_txfm2d_{16x8,8x16,32x16,16x32}_neon) for 8-bit
// residuals whose regions the caller validated, reporting false for other
// sizes. Like the square kernels, the column pass (fdct_rect_gen_arm64.go,
// fdct_net16_32_gen_arm64.go) stores transposed so the row pass runs in place
// over coeff[c*h+r]; other coefficient strides compute into a local block.
func forwardDCTRect8BitTrusted(coeff []int32, coeffStride int, residual []int16, residualStride int, size Size) bool {
	switch size {
	case Size{Width: 16, Height: 8}:
		if coeffStride == 8 {
			fwdRect16x8((*[128]int32)(coeff), residual, residualStride)
			return true
		}
		var buf [128]int32
		fwdRect16x8(&buf, residual, residualStride)
		fwdRectCopyOut(coeff, coeffStride, buf[:], 16, 8)
	case Size{Width: 8, Height: 16}:
		if coeffStride == 16 {
			fwdRect8x16((*[128]int32)(coeff), residual, residualStride)
			return true
		}
		var buf [128]int32
		fwdRect8x16(&buf, residual, residualStride)
		fwdRectCopyOut(coeff, coeffStride, buf[:], 8, 16)
	case Size{Width: 32, Height: 16}:
		if coeffStride == 16 {
			fwdRect32x16((*[512]int32)(coeff), residual, residualStride)
			return true
		}
		var buf [512]int32
		fwdRect32x16(&buf, residual, residualStride)
		fwdRectCopyOut(coeff, coeffStride, buf[:], 32, 16)
	case Size{Width: 16, Height: 32}:
		if coeffStride == 32 {
			fwdRect16x32((*[512]int32)(coeff), residual, residualStride)
			return true
		}
		var buf [512]int32
		fwdRect16x32(&buf, residual, residualStride)
		fwdRectCopyOut(coeff, coeffStride, buf[:], 16, 32)
	default:
		return false
	}
	return true
}

func fwdRectCopyOut(coeff []int32, coeffStride int, buf []int32, w, h int) {
	for k := range w {
		copy(coeff[k*coeffStride:k*coeffStride+h], buf[k*h:])
	}
}

func fwdRect16x8(c *[128]int32, residual []int16, rs int) {
	for g := 0; g < 16; g += fwdLanes {
		fwdColDCT8T(c[:], g, residual, rs)
	}
	for h := 0; h < 8; h += fwdLanes {
		fwdRowDCT16x8(c[:], h)
	}
}

func fwdRect8x16(c *[128]int32, residual []int16, rs int) {
	for g := 0; g < 8; g += fwdLanes {
		fwdColDCT16(c[:], g, residual, rs)
	}
	for h := 0; h < 16; h += fwdLanes {
		fwdRowDCT8x16(c[:], h)
	}
}

func fwdRect32x16(c *[512]int32, residual []int16, rs int) {
	for g := 0; g < 32; g += fwdLanes {
		fwdColDCT16R4(c[:], g, residual, rs)
	}
	for h := 0; h < 16; h += fwdLanes {
		fwdRowDCT32x16(c[:], h)
	}
}

func fwdRect16x32(c *[512]int32, residual []int16, rs int) {
	for g := 0; g < 16; g += fwdLanes {
		fwdColDCT32(c[:], g, residual, rs)
	}
	for h := 0; h < 32; h += fwdLanes {
		fwdRowDCT16x32(c[:], h)
	}
}
