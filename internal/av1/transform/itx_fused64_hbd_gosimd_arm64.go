// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package transform

import (
	"simd/archsimd"
	"unsafe"
)

// The 10- and 12-bit form of the fused 64x64 DCT_DCT inverse transform plus
// add (itx_fused64). dav1d's itx16 runs the 64-point column pass of 10 bpc
// blocks with its 8bpc int16 functions (inv_dct64_step* on 8h vectors), as
// for the smaller sizes (itx_fused_hbd_gosimd_arm64.go). The row strips and
// column networks are the 8-bit ones under the same per-lane L1 guards
// (certified for the 64-point network by itx64_gain_test.go): under them no
// coefficient or butterfly leaves int16, so the staged path's bd + 8 row and
// max(bd + 6, 16) column clamps never bite, and the column output is added
// to the 16bpc prediction with the saturating add clipped to the bit depth.
// A declined block is left untouched for the staged path.

// itx64ColStripAddHBD is itx64ColStripAdd onto 16bpc samples.
func itx64ColStripAddHBD(tmp *[32 * 64]int16, c0 int, inRows int, dst unsafe.Pointer, dstStride int, pred unsafe.Pointer, predStride int, maxV archsimd.Int16x8) {
	var in [32]archsimd.Int16x8
	for r := 0; r < inRows; r++ {
		in[r] = archsimd.LoadInt16x8Array((*[8]int16)(tmp[r*64+c0:]))
	}
	var out [64]archsimd.Int16x8
	itx64Net(inRows, &in, &out)
	one := archsimd.BroadcastInt16x8(1)
	sh1 := archsimd.BroadcastInt16x8(-1)
	sh3 := archsimd.BroadcastInt16x8(-3)
	zero := archsimd.BroadcastInt16x8(0)
	for r := 0; r < 64; r++ {
		itxAddRow16(unsafe.Add(dst, r*dstStride+2*c0), unsafe.Add(pred, r*predStride+2*c0), itx32Round2(out[r], sh3, sh1, one), zero, maxV)
	}
}

// InverseAddDCT64x64HBD is InverseAddDCT64x64 for a 10- or 12-bit block
// (2-byte samples, strides in bytes).
func InverseAddDCT64x64HBD(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, activeRows, activeCols int, bitDepth uint8) bool {
	maxV, ok := itxHBDMax(bitDepth)
	if !ok || activeRows <= 0 || activeRows > 32 || activeCols <= 0 || activeCols > 32 ||
		coeffStride < 32 || len(coeff) < 31*coeffStride+32 || !itxHBDFits(dst, dstStride, 64) || !itxHBDFits(pred, predStride, 64) {
		return false
	}
	inCols := itx64Form(activeCols)
	inRows := itx64Form(activeRows)
	var tmp [32 * 64]int16
	var colL1 [8]archsimd.Uint16x8
	written := 0
	for ; written < activeRows; written += 8 {
		if !itx64RowStrip(coeff, coeffStride, written, inCols, &tmp, &colL1) {
			return false
		}
	}
	// The column form may read tmp rows past the strips that ran; those
	// rows are the transform of zero coefficient rows, so zero.
	if written < inRows {
		clear(tmp[written*64 : inRows*64])
	}
	m := colL1[0]
	for g := 1; g < 8; g++ {
		m = m.Max(colL1[g])
	}
	if m.ReduceMax() > itx16ColumnL1Bound {
		return false
	}
	dp := unsafe.Pointer(unsafe.SliceData(dst))
	pp := unsafe.Pointer(unsafe.SliceData(pred))
	for c0 := 0; c0 < 64; c0 += 8 {
		itx64ColStripAddHBD(&tmp, c0, inRows, dp, dstStride, pp, predStride, maxV)
	}
	return true
}
