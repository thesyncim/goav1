// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package transform

import (
	"simd/archsimd"
	"unsafe"
)

// 10- and 12-bit forms of the fused two-pass DCT_DCT inverse transform plus
// add. dav1d's 16bpc NEON (src/arm/64/itx16.S) runs the row pass in 32-bit
// lanes and then reuses the 8bpc int16 column functions
// (inv_dct_8h_x16_neon, inv_dct32_odd_8h_x16_neon) for 10 bpc, because the
// column input is clamped to max(bd + 6, 16) = 16 bits. Here both passes are
// the generated 8-bit int16 kernels (itx_fused_gen_arm64.go, itxGenHBD*):
// the 8-bit row strips under their per-lane L1 guard, and column passes
// whose final stage adds the 16bpc prediction row:
//
//   - a coefficient row whose L1 norm fits the guard (itx16ColumnL1Bound)
//     has no coefficient outside int16 (the narrowing load saturates any that
//     is, which then fails the guard) and, by the certified network gain, no
//     butterfly value outside int16. The staged reference clamps the row input
//     to bd + 8 bits and the column input to max(bd + 6, 16) bits, so under
//     the guard none of its clamps bite at 10 or 12 bits either, and both
//     compute the unclamped transform;
//   - the column pass is the same int16 network with its L1 guard (the column
//     bound is 16 bits at 8 and 10 bits and wider at 12), and the final stage
//     rounds by 4 and adds the 16bpc prediction row with a saturating int16
//     add clipped to [0, 2^bd - 1], which equals the scalar
//     clip(pred + residual) for any int16 residual and pred <= 4095.
//
// A declined block is left untouched; the caller runs the staged path, which
// produces the same samples.

// itxAddRow16 stores clip(pred + res, 0, maxV) for the eight 16bpc samples
// at pred into dst.
func itxAddRow16(dst, pred unsafe.Pointer, res, zero, maxV archsimd.Int16x8) {
	archsimd.LoadInt16x8Array((*[8]int16)(pred)).AddSaturated(res).Max(zero).Min(maxV).StoreArray((*[8]int16)(dst))
}

// itxHBDMax returns the broadcast pixel maximum for a 10- or 12-bit block,
// false for any other depth.
func itxHBDMax(bitDepth uint8) (archsimd.Int16x8, bool) {
	if bitDepth != 10 && bitDepth != 12 {
		return archsimd.Int16x8{}, false
	}
	return archsimd.BroadcastInt16x8(int16(1)<<bitDepth - 1), true
}

// itxHBDFits reports whether an n x n block of 2-byte samples at stride
// (bytes) fits buf.
func itxHBDFits(buf []byte, stride, n int) bool {
	return stride >= 2*n && len(buf) >= (n-1)*stride+2*n
}

// itxHBDSquareOK checks the contract of the n x n 10/12-bit kernels (coded
// extent, the column-major coefficient stride cs and the 2-byte sample
// rows of dst and pred) and returns the pixel maximum.
func itxHBDSquareOK(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, activeRows, activeCols int, bitDepth uint8, n, cs int) (int16, bool) {
	if (bitDepth != 10 && bitDepth != 12) || activeRows <= 0 || activeRows > cs || activeCols <= 0 || activeCols > cs ||
		coeffStride != cs || len(coeff) < cs*cs || !itxHBDFits(dst, dstStride, n) || !itxHBDFits(pred, predStride, n) {
		return 0, false
	}
	return int16(1)<<bitDepth - 1, true
}

// InverseAddDCT8x8HBD is InverseAddDCT8x8 for a 10- or 12-bit block (2-byte
// samples, strides in bytes).
func InverseAddDCT8x8HBD(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, activeRows, activeCols int, bitDepth uint8) bool {
	mx, ok := itxHBDSquareOK(dst, dstStride, pred, predStride, coeff, coeffStride, activeRows, activeCols, bitDepth, 8, 8)
	if !ok {
		return false
	}
	// Strided rows inside the extents checked above; same shape as the residual-add rows (checked rows +9-86%).
	return itxGenHBDDCT8x8((*[64]int32)(coeff[:64]), unsafe.Pointer(unsafe.SliceData(dst)), dstStride,
		unsafe.Pointer(unsafe.SliceData(pred)), predStride, activeCols, activeRows, mx)
}

// InverseAddDCT16x16HBD is InverseAddDCT16x16 for a 10- or 12-bit block.
func InverseAddDCT16x16HBD(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, activeRows, activeCols int, bitDepth uint8) bool {
	mx, ok := itxHBDSquareOK(dst, dstStride, pred, predStride, coeff, coeffStride, activeRows, activeCols, bitDepth, 16, 16)
	if !ok {
		return false
	}
	// Strided rows inside the extents checked above; same shape as the residual-add rows (checked rows +9-86%).
	return itxGenHBDDCT16x16((*[256]int32)(coeff[:256]), unsafe.Pointer(unsafe.SliceData(dst)), dstStride,
		unsafe.Pointer(unsafe.SliceData(pred)), predStride, activeCols, activeRows, mx)
}

// InverseAddDCT32x32HBD is the 10- or 12-bit 32x32 DCT_DCT block with coded
// coefficients in rows 0..activeRows-1 and columns 0..activeCols-1.
func InverseAddDCT32x32HBD(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, activeRows, activeCols int, bitDepth uint8) bool {
	mx, ok := itxHBDSquareOK(dst, dstStride, pred, predStride, coeff, coeffStride, activeRows, activeCols, bitDepth, 32, 32)
	if !ok {
		return false
	}
	// Strided rows inside the extents checked above; same shape as the residual-add rows (checked rows +9-86%).
	return itxGenHBDDCT32x32((*[1024]int32)(coeff[:1024]), unsafe.Pointer(unsafe.SliceData(dst)), dstStride,
		unsafe.Pointer(unsafe.SliceData(pred)), predStride, activeCols, activeRows, mx)
}

// InverseAddDCT64x64HBD is InverseAddDCT64x64 for a 10- or 12-bit block
// (2-byte samples, strides in bytes): coded coefficients in the 32x32 corner
// (dav1d's itx16 runs the 64-point column pass of 10 bpc blocks with its
// 8bpc int16 functions too).
func InverseAddDCT64x64HBD(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, activeRows, activeCols int, bitDepth uint8) bool {
	mx, ok := itxHBDSquareOK(dst, dstStride, pred, predStride, coeff, coeffStride, activeRows, activeCols, bitDepth, 64, 32)
	if !ok {
		return false
	}
	// Strided rows inside the extents checked above; same shape as the residual-add rows (checked rows +9-86%).
	return itxGenHBDDCT64x64((*[1024]int32)(coeff[:1024]), unsafe.Pointer(unsafe.SliceData(dst)), dstStride,
		unsafe.Pointer(unsafe.SliceData(pred)), predStride, activeCols, activeRows, mx)
}
