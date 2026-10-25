// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package transform

import (
	"simd/archsimd"
	"unsafe"
)

// 10- and 12-bit forms of the fused 2:1 rectangular DCT_DCT blocks 16x32
// and 32x16 (itx_fused_rect). dav1d's itx16 runs their 10 bpc column pass
// with the 8bpc int16 functions, as for the square sizes
// (itx_fused_hbd_gosimd_arm64.go). The row strips are the 8-bit ones: the
// load scales each coefficient by 1/sqrt(2) in int32 and only then narrows
// with saturation, so a scaled value outside int16 saturates to +-32767 and
// fails the per-lane L1 guard (itx16ColumnL1Bound < 32767). Under the guard
// every scaled row input lies inside int16, so the staged path's clamp of
// the scaled input to bd + 8 bits does not bite, nor does its
// max(bd + 6, 16) column clamp; the column output is added to the 16bpc
// prediction with the saturating add clipped to the bit depth. A declined
// block is left untouched for the staged path.

// itx16x32ColStripAddHBD is itx16x32ColStripAdd onto 16bpc samples.
func itx16x32ColStripAddHBD(tmp *[32 * 16]int16, c0 int, inRows int, dst unsafe.Pointer, dstStride int, pred unsafe.Pointer, predStride int, maxV archsimd.Int16x8) {
	var in [32]archsimd.Int16x8
	for r := 0; r < inRows; r++ {
		in[r] = archsimd.LoadInt16x8Array((*[8]int16)(tmp[r*16+c0:]))
	}
	var out [32]archsimd.Int16x8
	itx32Net(inRows, &in, &out)
	one := archsimd.BroadcastInt16x8(1)
	sh1 := archsimd.BroadcastInt16x8(-1)
	sh3 := archsimd.BroadcastInt16x8(-3)
	zero := archsimd.BroadcastInt16x8(0)
	for r := 0; r < 32; r++ {
		itxAddRow16(unsafe.Add(dst, r*dstStride+2*c0), unsafe.Add(pred, r*predStride+2*c0), itx32Round2(out[r], sh3, sh1, one), zero, maxV)
	}
}

// InverseAddDCT16x32HBD is InverseAddDCT16x32 for a 10- or 12-bit block
// (2-byte samples, strides in bytes).
func InverseAddDCT16x32HBD(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, activeRows, activeCols int, bitDepth uint8) bool {
	maxV, ok := itxHBDMax(bitDepth)
	if !ok || activeRows <= 0 || activeRows > 32 || activeCols <= 0 || activeCols > 16 ||
		coeffStride < 32 || len(coeff) < 15*coeffStride+32 || !itxHBDRectFits(dst, dstStride, 16, 32) || !itxHBDRectFits(pred, predStride, 16, 32) {
		return false
	}
	inCols := itx16Form(activeCols)
	inRows := itx64Form(activeRows)
	var tmp [32 * 16]int16
	var colL1 [2]archsimd.Uint16x8
	written := 0
	for ; written < activeRows; written += 8 {
		if !itx16x32RowStrip(coeff, coeffStride, written, inCols, &tmp, &colL1) {
			return false
		}
	}
	// The column form may read tmp rows past the strips that ran; those rows
	// are the transform of zero coefficient rows, so zero.
	if written < inRows {
		clear(tmp[written*16 : inRows*16])
	}
	if colL1[0].Max(colL1[1]).ReduceMax() > itx16ColumnL1Bound {
		return false
	}
	dp := unsafe.Pointer(unsafe.SliceData(dst))
	pp := unsafe.Pointer(unsafe.SliceData(pred))
	itx16x32ColStripAddHBD(&tmp, 0, inRows, dp, dstStride, pp, predStride, maxV)
	itx16x32ColStripAddHBD(&tmp, 8, inRows, dp, dstStride, pp, predStride, maxV)
	return true
}

// itx32x16ColStripAddHBD is itx32x16ColStripAdd onto 16bpc samples.
func itx32x16ColStripAddHBD(tmp *[16 * 32]int16, c0 int, inRows int, dst unsafe.Pointer, dstStride int, pred unsafe.Pointer, predStride int, maxV archsimd.Int16x8) {
	var in [16]archsimd.Int16x8
	for r := 0; r < inRows; r++ {
		in[r] = archsimd.LoadInt16x8Array((*[8]int16)(tmp[r*32+c0:]))
	}
	var out [16]archsimd.Int16x8
	itx16Net(inRows, &in, &out)
	one := archsimd.BroadcastInt16x8(1)
	sh1 := archsimd.BroadcastInt16x8(-1)
	sh3 := archsimd.BroadcastInt16x8(-3)
	zero := archsimd.BroadcastInt16x8(0)
	for r := 0; r < 16; r++ {
		itxAddRow16(unsafe.Add(dst, r*dstStride+2*c0), unsafe.Add(pred, r*predStride+2*c0), itx32Round2(out[r], sh3, sh1, one), zero, maxV)
	}
}

// InverseAddDCT32x16HBD is InverseAddDCT32x16 for a 10- or 12-bit block.
func InverseAddDCT32x16HBD(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, activeRows, activeCols int, bitDepth uint8) bool {
	maxV, ok := itxHBDMax(bitDepth)
	if !ok || activeRows <= 0 || activeRows > 16 || activeCols <= 0 || activeCols > 32 ||
		coeffStride < 16 || len(coeff) < 31*coeffStride+16 || !itxHBDRectFits(dst, dstStride, 32, 16) || !itxHBDRectFits(pred, predStride, 32, 16) {
		return false
	}
	inCols := itx64Form(activeCols)
	inRows := itx16Form(activeRows)
	var tmp [16 * 32]int16
	var colL1 [4]archsimd.Uint16x8
	// A row strip writes all eight of its rows, so rows past the coded ones
	// come out zero and the column form reads them as such; the second strip
	// runs only when coded rows reach it.
	if !itx32x16RowStrip(coeff, coeffStride, 0, inCols, &tmp, &colL1) {
		return false
	}
	if activeRows > 8 && !itx32x16RowStrip(coeff, coeffStride, 8, inCols, &tmp, &colL1) {
		return false
	}
	if colL1[0].Max(colL1[1]).Max(colL1[2].Max(colL1[3])).ReduceMax() > itx16ColumnL1Bound {
		return false
	}
	dp := unsafe.Pointer(unsafe.SliceData(dst))
	pp := unsafe.Pointer(unsafe.SliceData(pred))
	for c0 := 0; c0 < 32; c0 += 8 {
		itx32x16ColStripAddHBD(&tmp, c0, inRows, dp, dstStride, pp, predStride, maxV)
	}
	return true
}

// itxHBDRectFits reports whether a w x h block of 2-byte samples at stride
// (bytes) fits buf.
func itxHBDRectFits(buf []byte, stride, w, h int) bool {
	return stride >= 2*w && len(buf) >= (h-1)*stride+2*w
}
