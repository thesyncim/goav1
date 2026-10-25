// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package transform

import (
	"simd/archsimd"
	"unsafe"
)

// 10- and 12-bit forms of the fused 16x16 and 8x8 DCT/ADST/flipped-ADST
// pairings (itx_fused_hybrid). As for the DCT_DCT blocks
// (itx_fused_hbd_gosimd_arm64.go), dav1d's itx16 runs the 10 bpc column pass
// with its 8bpc int16 functions (inv_adst_8h_x16_neon, inv_flipadst_8h_x16_neon,
// ...), and the int16 row strips and column networks here are the 8-bit ones
// under the same per-lane L1 guards, which itx_hybrid_gain_test.go certifies
// for the ADST networks too: under them no coefficient or butterfly leaves
// int16, so the staged path's bd + 8 row and max(bd + 6, 16) column clamps
// never bite, and the column output is added to the 16bpc prediction with
// the saturating add clipped to the bit depth. A declined block is left
// untouched for the staged path.

// itx16ColStripAddTHBD is itx16ColStripAddT onto 16bpc samples.
func itx16ColStripAddTHBD(tmp *[16 * 16]int16, c0 int, inRows int, t tx1DType, dst unsafe.Pointer, dstStride int, pred unsafe.Pointer, predStride int, maxV archsimd.Int16x8) {
	var in [16]archsimd.Int16x8
	for r := 0; r < inRows; r++ {
		in[r] = archsimd.LoadInt16x8Array((*[8]int16)(tmp[r*16+c0:]))
	}
	var out [16]archsimd.Int16x8
	itx16Net1D(t, inRows, &in, &out)
	one := archsimd.BroadcastInt16x8(1)
	sh1 := archsimd.BroadcastInt16x8(-1)
	sh3 := archsimd.BroadcastInt16x8(-3)
	zero := archsimd.BroadcastInt16x8(0)
	for r := 0; r < 16; r++ {
		itxAddRow16(unsafe.Add(dst, r*dstStride+2*c0), unsafe.Add(pred, r*predStride+2*c0), itx32Round2(out[r], sh3, sh1, one), zero, maxV)
	}
}

// InverseAdd16x16HybridHBD is InverseAdd16x16Hybrid for a 10- or 12-bit
// block (2-byte samples, strides in bytes).
func InverseAdd16x16HybridHBD(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, txType Type, activeRows, activeCols int, bitDepth uint8) bool {
	vertical, horizontal, ok := itxHybridTypes(txType)
	if !ok {
		return false
	}
	maxV, ok := itxHBDMax(bitDepth)
	if !ok || activeRows <= 0 || activeRows > 16 || activeCols <= 0 || activeCols > 16 ||
		coeffStride < 16 || len(coeff) < 15*coeffStride+16 || !itxHBDFits(dst, dstStride, 16) || !itxHBDFits(pred, predStride, 16) {
		return false
	}
	inCols := itx16Form(activeCols)
	inRows := itx16Form(activeRows)
	var tmp [16 * 16]int16
	var colL1 [2]archsimd.Uint16x8
	if !itx16RowStripT(coeff, coeffStride, 0, inCols, horizontal, &tmp, &colL1) {
		return false
	}
	if activeRows > 8 && !itx16RowStripT(coeff, coeffStride, 8, inCols, horizontal, &tmp, &colL1) {
		return false
	}
	if colL1[0].Max(colL1[1]).ReduceMax() > itx16ColumnL1Bound {
		return false
	}
	dp := unsafe.Pointer(unsafe.SliceData(dst))
	pp := unsafe.Pointer(unsafe.SliceData(pred))
	itx16ColStripAddTHBD(&tmp, 0, inRows, vertical, dp, dstStride, pp, predStride, maxV)
	itx16ColStripAddTHBD(&tmp, 8, inRows, vertical, dp, dstStride, pp, predStride, maxV)
	return true
}

// InverseAdd8x8HybridHBD is InverseAdd8x8Hybrid for a 10- or 12-bit block.
func InverseAdd8x8HybridHBD(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, txType Type, activeRows, activeCols int, bitDepth uint8) bool {
	vertical, horizontal, ok := itxHybridTypes(txType)
	if !ok {
		return false
	}
	maxV, ok := itxHBDMax(bitDepth)
	if !ok || activeRows <= 0 || activeRows > 8 || activeCols <= 0 || activeCols > 8 ||
		coeffStride < 8 || len(coeff) < 7*coeffStride+8 || !itxHBDFits(dst, dstStride, 8) || !itxHBDFits(pred, predStride, 8) {
		return false
	}
	var in [8]archsimd.Int16x8
	var l1 archsimd.Uint16x8
	for k := 0; k < activeCols; k++ {
		x := itx16LoadNarrow(coeff[k*coeffStride : k*coeffStride+8])
		l1 = l1.AddSaturated(x.Abs().ToBits())
		in[k] = x
	}
	if l1.ReduceMax() > itx16ColumnL1Bound {
		return false
	}
	var mid [8]archsimd.Int16x8
	itx8Net1D(horizontal, activeCols, &in, &mid)
	one := archsimd.BroadcastInt16x8(1)
	sh1 := archsimd.BroadcastInt16x8(-1)
	var colL1 archsimd.Uint16x8
	for j := range mid {
		mid[j] = mid[j].Shift(sh1).Add(mid[j].And(one))
	}
	itx16Transpose8x8(&mid)
	for r := range mid {
		colL1 = colL1.AddSaturated(mid[r].Abs().ToBits())
	}
	if colL1.ReduceMax() > itx16ColumnL1Bound {
		return false
	}
	var out [8]archsimd.Int16x8
	itx8Net1D(vertical, activeRows, &mid, &out)
	sh3 := archsimd.BroadcastInt16x8(-3)
	zero := archsimd.BroadcastInt16x8(0)
	dp := unsafe.Pointer(unsafe.SliceData(dst))
	pp := unsafe.Pointer(unsafe.SliceData(pred))
	for r := 0; r < 8; r++ {
		itxAddRow16(unsafe.Add(dp, r*dstStride), unsafe.Add(pp, r*predStride), itx32Round2(out[r], sh3, sh1, one), zero, maxV)
	}
	return true
}
