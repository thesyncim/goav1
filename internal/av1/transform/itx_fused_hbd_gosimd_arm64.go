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
// add (itx_fused8/16/32*). dav1d's 16bpc NEON (src/arm/64/itx16.S) runs the
// row pass in 32-bit lanes and then reuses the 8bpc int16 column functions
// (inv_dct_8h_x16_neon, inv_dct32_odd_8h_x16_neon) for 10 bpc, because the
// column input is clamped to max(bd + 6, 16) = 16 bits. Here the row pass is
// the 8-bit int16 row strip itself, under its per-lane L1 guard:
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
//
//go:nocheckptr
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

// InverseAddDCT8x8HBD is InverseAddDCT8x8 for a 10- or 12-bit block (2-byte
// samples, strides in bytes).
func InverseAddDCT8x8HBD(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, activeRows, activeCols int, bitDepth uint8) bool {
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
	if activeCols <= 4 {
		itx8NetLow4(&in, &mid)
	} else {
		itx8Net(&in, &mid)
	}
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
	if activeRows <= 4 {
		itx8NetLow4(&mid, &out)
	} else {
		itx8Net(&mid, &out)
	}
	sh3 := archsimd.BroadcastInt16x8(-3)
	zero := archsimd.BroadcastInt16x8(0)
	dp := unsafe.Pointer(unsafe.SliceData(dst))
	pp := unsafe.Pointer(unsafe.SliceData(pred))
	for r := 0; r < 8; r++ {
		itxAddRow16(unsafe.Add(dp, r*dstStride), unsafe.Add(pp, r*predStride), itx32Round2(out[r], sh3, sh1, one), zero, maxV)
	}
	return true
}

// itx16ColStripAddHBD is itx16ColStripAdd onto 16bpc samples.
func itx16ColStripAddHBD(tmp *[16 * 16]int16, c0 int, inRows int, dst unsafe.Pointer, dstStride int, pred unsafe.Pointer, predStride int, maxV archsimd.Int16x8) {
	var in [16]archsimd.Int16x8
	for r := 0; r < inRows; r++ {
		in[r] = archsimd.LoadInt16x8Array((*[8]int16)(tmp[r*16+c0:]))
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

// InverseAddDCT16x16HBD is InverseAddDCT16x16 for a 10- or 12-bit block.
func InverseAddDCT16x16HBD(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, activeRows, activeCols int, bitDepth uint8) bool {
	maxV, ok := itxHBDMax(bitDepth)
	if !ok || activeRows <= 0 || activeRows > 16 || activeCols <= 0 || activeCols > 16 ||
		coeffStride < 16 || len(coeff) < 15*coeffStride+16 || !itxHBDFits(dst, dstStride, 16) || !itxHBDFits(pred, predStride, 16) {
		return false
	}
	inCols := itx16Form(activeCols)
	inRows := itx16Form(activeRows)
	var tmp [16 * 16]int16
	var colL1 [2]archsimd.Uint16x8
	if !itx16RowStrip(coeff, coeffStride, 0, inCols, &tmp, &colL1) {
		return false
	}
	if activeRows > 8 && !itx16RowStrip(coeff, coeffStride, 8, inCols, &tmp, &colL1) {
		return false
	}
	if colL1[0].Max(colL1[1]).ReduceMax() > itx16ColumnL1Bound {
		return false
	}
	dp := unsafe.Pointer(unsafe.SliceData(dst))
	pp := unsafe.Pointer(unsafe.SliceData(pred))
	itx16ColStripAddHBD(&tmp, 0, inRows, dp, dstStride, pp, predStride, maxV)
	itx16ColStripAddHBD(&tmp, 8, inRows, dp, dstStride, pp, predStride, maxV)
	return true
}

// itx32ColStripAddHBD runs the 32x32 column pass for columns c0..c0+7 from
// the first inRows rows (8, 16 or 32) of a 32-wide int16 tmp and adds the
// result to 16bpc samples.
func itx32ColStripAddHBD(tmp []int16, c0 int, inRows int, dst unsafe.Pointer, dstStride int, pred unsafe.Pointer, predStride int, maxV archsimd.Int16x8) {
	_ = tmp[(inRows-1)*32+c0+7]
	var in [32]archsimd.Int16x8
	for r := 0; r < inRows; r++ {
		in[r] = archsimd.LoadInt16x8Array((*[8]int16)(tmp[r*32+c0:]))
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

// InverseAddDCT32x32HBD is the 10- or 12-bit 32x32 DCT_DCT block with coded
// coefficients in rows 0..activeRows-1 and columns 0..activeCols-1, taking
// the 8-bit kernels' row strips (Low8, Low16 or full network forms) and the
// matching column form.
func InverseAddDCT32x32HBD(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, activeRows, activeCols int, bitDepth uint8) bool {
	maxV, ok := itxHBDMax(bitDepth)
	if !ok || activeRows <= 0 || activeRows > 32 || activeCols <= 0 || activeCols > 32 ||
		!itxHBDFits(dst, dstStride, 32) || !itxHBDFits(pred, predStride, 32) {
		return false
	}
	dp := unsafe.Pointer(unsafe.SliceData(dst))
	pp := unsafe.Pointer(unsafe.SliceData(pred))
	switch {
	case activeRows <= 8 && activeCols <= 8:
		if coeffStride < 8 || len(coeff) < 7*coeffStride+8 {
			return false
		}
		var tmp [8 * 32]int16
		if !itx32RowStripLow8(coeff, coeffStride, &tmp) {
			return false
		}
		for c0 := 0; c0 < 32; c0 += 8 {
			itx32ColStripAddHBD(tmp[:], c0, 8, dp, dstStride, pp, predStride, maxV)
		}
	case activeRows <= 16 && activeCols <= 16:
		if coeffStride < 16 || len(coeff) < 15*coeffStride+16 {
			return false
		}
		var tmp [16 * 32]int16
		var colL1 [4]archsimd.Uint16x8
		if !itx32RowStripLow16(coeff, coeffStride, 0, &tmp, &colL1) {
			return false
		}
		if activeRows > 8 {
			if !itx32RowStripLow16(coeff, coeffStride, 8, &tmp, &colL1) {
				return false
			}
		} else {
			clear(tmp[8*32:])
		}
		if colL1[0].Max(colL1[1]).Max(colL1[2].Max(colL1[3])).ReduceMax() > itx16ColumnL1Bound {
			return false
		}
		for c0 := 0; c0 < 32; c0 += 8 {
			itx32ColStripAddHBD(tmp[:], c0, 16, dp, dstStride, pp, predStride, maxV)
		}
	default:
		if coeffStride < 32 || len(coeff) < 31*coeffStride+32 {
			return false
		}
		inCols := itx64Form(activeCols)
		inRows := itx64Form(activeRows)
		var tmp [32 * 32]int16
		var colL1 [4]archsimd.Uint16x8
		written := 0
		for ; written < activeRows; written += 8 {
			if !itx32RowStrip(coeff, coeffStride, written, inCols, &tmp, &colL1) {
				return false
			}
		}
		if written < inRows {
			clear(tmp[written*32 : inRows*32])
		}
		if colL1[0].Max(colL1[1]).Max(colL1[2].Max(colL1[3])).ReduceMax() > itx16ColumnL1Bound {
			return false
		}
		for c0 := 0; c0 < 32; c0 += 8 {
			itx32ColStripAddHBD(tmp[:], c0, inRows, dp, dstStride, pp, predStride, maxV)
		}
	}
	return true
}
