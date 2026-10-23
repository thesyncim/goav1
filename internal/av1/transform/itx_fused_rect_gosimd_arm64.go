// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package transform

import (
	"simd/archsimd"
	"unsafe"
)

// itx_fused_rect: the fused two-pass inverse transform (itx_fused16,
// itx_fused32) for the 2:1 rectangular 8-bit DCT_DCT blocks 16x32 and
// 32x16. The pass structure is the square kernels' with two differences the
// scalar path has for the rect2 sizes:
//
//   - the coefficients are scaled by 1/sqrt(2) before the row transform,
//     Round2(c*2896, 12) = (c*181 + 128) >> 8, and only then clamped to the
//     row range, which the narrowing load does (itx16LoadNarrowRect2);
//   - the inter-pass shift is 1, not 2.
//
// The row network is the DCT of the block's width (DCT16 for 16x32, DCT32
// for 32x16) and the column network the DCT of its height, each in the form
// its coded extent allows. Both are the certified networks, so the L1 guards
// and their gain model carry over unchanged: they bound the values the
// networks see (the scaled row inputs, the stored inter-pass values), not how
// those values were produced.

// itxRect2Pre is the pre-clamp that keeps c*181+128 inside int32 without
// changing the result: scale(+-1<<22) already lies far outside the int16 row
// clamp, so clamping the input to +-1<<22 first is exact (see
// stageTransposeClamp).
const itxRect2Pre = 1 << 22

// itxRect2Consts holds the broadcast constants of the rect2 narrowing load.
type itxRect2Consts struct {
	preLo, preHi, k181, r128 archsimd.Int32x4
}

func newItxRect2Consts() itxRect2Consts {
	return itxRect2Consts{
		preLo: archsimd.BroadcastInt32x4(-itxRect2Pre),
		preHi: archsimd.BroadcastInt32x4(itxRect2Pre),
		k181:  archsimd.BroadcastInt32x4(181),
		r128:  archsimd.BroadcastInt32x4(128),
	}
}

// itx16LoadNarrowRect2 loads eight int32 coefficients, applies the rect2
// scale and narrows to int16 with saturation (the 8-bit row clamp): lane i
// is clip16(Round2(p[i]*2896, 12)).
func itx16LoadNarrowRect2(p []int32, k *itxRect2Consts) archsimd.Int16x8 {
	lo := archsimd.LoadInt32x4Array((*[4]int32)(p))
	hi := archsimd.LoadInt32x4Array((*[4]int32)(p[4:]))
	lo = lo.Max(k.preLo).Min(k.preHi).MulAdd(k.k181, k.r128).ShiftAllRight(8)
	hi = hi.Max(k.preLo).Min(k.preHi).MulAdd(k.k181, k.r128).ShiftAllRight(8)
	v := lo.SaturateToInt16().ToBits().ReshapeToUint64s().BitsToFloat64()
	v = v.SetElem(1, hi.SaturateToInt16().ToBits().ReshapeToUint64s().BitsToFloat64().GetElem(0))
	return v.ToBits().ReshapeToUint16s().BitsToInt16()
}

// itxRound2By1 is Round2(x, 1) on int16 lanes, (x >> 1) + (x & 1): exact and
// overflow-free.
func itxRound2By1(x, sh1, one archsimd.Int16x8) archsimd.Int16x8 {
	return x.Shift(sh1).Add(x.And(one))
}

// itx16x32RowStrip runs the 16x32 row pass (DCT16 rows) for coefficient rows
// r0..r0+7 into tmp rows r0..r0+7 (16 wide).
func itx16x32RowStrip(coeff []int32, stride int, r0 int, inCols int, tmp *[32 * 16]int16, colL1 *[2]archsimd.Uint16x8) bool {
	_ = coeff[(inCols-1)*stride+r0+7]
	k := newItxRect2Consts()
	var in [16]archsimd.Int16x8
	var l1 archsimd.Uint16x8
	for c := 0; c < inCols; c++ {
		x := itx16LoadNarrowRect2(coeff[c*stride+r0:c*stride+r0+8], &k)
		l1 = l1.AddSaturated(x.Abs().ToBits())
		in[c] = x
	}
	if l1.ReduceMax() > itx16ColumnL1Bound {
		return false
	}
	var out [16]archsimd.Int16x8
	itx16Net(inCols, &in, &out)
	one := archsimd.BroadcastInt16x8(1)
	sh1 := archsimd.BroadcastInt16x8(-1)
	for g := 0; g < 2; g++ {
		var v [8]archsimd.Int16x8
		for j := 0; j < 8; j++ {
			v[j] = itxRound2By1(out[8*g+j], sh1, one)
		}
		itx16Transpose8x8(&v)
		gl1 := colL1[g]
		for r := 0; r < 8; r++ {
			v[r].StoreArray((*[8]int16)(tmp[(r0+r)*16+8*g:]))
			gl1 = gl1.AddSaturated(v[r].Abs().ToBits())
		}
		colL1[g] = gl1
	}
	return true
}

// itx16x32ColStripAdd runs the 16x32 column pass (DCT32 columns) for columns
// c0..c0+7 from inRows coded tmp rows (8, 16 or 32) and adds the result.
func itx16x32ColStripAdd(tmp *[32 * 16]int16, c0 int, inRows int, dst unsafe.Pointer, dstStride int, pred unsafe.Pointer, predStride int) {
	var in [32]archsimd.Int16x8
	for r := 0; r < inRows; r++ {
		in[r] = archsimd.LoadInt16x8Array((*[8]int16)(tmp[r*16+c0:]))
	}
	var out [32]archsimd.Int16x8
	itx32Net(inRows, &in, &out)
	one := archsimd.BroadcastInt16x8(1)
	sh1 := archsimd.BroadcastInt16x8(-1)
	sh3 := archsimd.BroadcastInt16x8(-3)
	for r := 0; r < 32; r++ {
		res := itx32Round2(out[r], sh3, sh1, one)
		rows := [2]uint64{*(*uint64)(unsafe.Add(pred, r*predStride+c0)), 0}
		p := archsimd.LoadUint64x2Array(&rows).ReshapeToUint8s().ExtendLo8ToUint16().BitsToInt16()
		*(*uint64)(unsafe.Add(dst, r*dstStride+c0)) = p.Add(res).SaturateToUint8().ReshapeToUint64s().GetElem(0)
	}
}

// InverseAddDCT16x32 reconstructs one 8-bit 16x32 (16 wide, 32 high)
// DCT_DCT block with coded coefficients in rows 0..activeRows-1 and columns
// 0..activeCols-1: dst = clip(pred + inverse(coeff)), with the dequantized
// coefficients column-major at coeffStride. pred may be dst. It reports
// false, with dst untouched, when the block is outside the kernel's contract
// or a guard declines; the caller then runs the staged path, which produces
// the same bytes.
func InverseAddDCT16x32(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, activeRows, activeCols int) bool {
	if activeRows <= 0 || activeRows > 32 || activeCols <= 0 || activeCols > 16 ||
		coeffStride < 32 || len(coeff) < 15*coeffStride+32 || dstStride < 16 || predStride < 16 ||
		len(dst) < 31*dstStride+16 || len(pred) < 31*predStride+16 {
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
	itx16x32ColStripAdd(&tmp, 0, inRows, dp, dstStride, pp, predStride)
	itx16x32ColStripAdd(&tmp, 8, inRows, dp, dstStride, pp, predStride)
	return true
}

// itx32x16RowStrip runs the 32x16 row pass (DCT32 rows) for coefficient rows
// r0..r0+7 into tmp rows r0..r0+7 (32 wide).
func itx32x16RowStrip(coeff []int32, stride int, r0 int, inCols int, tmp *[16 * 32]int16, colL1 *[4]archsimd.Uint16x8) bool {
	_ = coeff[(inCols-1)*stride+r0+7]
	k := newItxRect2Consts()
	var in [32]archsimd.Int16x8
	var l1 archsimd.Uint16x8
	for c := 0; c < inCols; c++ {
		x := itx16LoadNarrowRect2(coeff[c*stride+r0:c*stride+r0+8], &k)
		l1 = l1.AddSaturated(x.Abs().ToBits())
		in[c] = x
	}
	if l1.ReduceMax() > itx16ColumnL1Bound {
		return false
	}
	var out [32]archsimd.Int16x8
	itx32Net(inCols, &in, &out)
	one := archsimd.BroadcastInt16x8(1)
	sh1 := archsimd.BroadcastInt16x8(-1)
	for g := 0; g < 4; g++ {
		var v [8]archsimd.Int16x8
		for j := 0; j < 8; j++ {
			v[j] = itxRound2By1(out[8*g+j], sh1, one)
		}
		itx16Transpose8x8(&v)
		gl1 := colL1[g]
		for r := 0; r < 8; r++ {
			v[r].StoreArray((*[8]int16)(tmp[(r0+r)*32+8*g:]))
			gl1 = gl1.AddSaturated(v[r].Abs().ToBits())
		}
		colL1[g] = gl1
	}
	return true
}

// itx32x16ColStripAdd runs the 32x16 column pass (DCT16 columns) for columns
// c0..c0+7 from inRows coded tmp rows (4, 8 or 16) and adds the result.
func itx32x16ColStripAdd(tmp *[16 * 32]int16, c0 int, inRows int, dst unsafe.Pointer, dstStride int, pred unsafe.Pointer, predStride int) {
	var in [16]archsimd.Int16x8
	for r := 0; r < inRows; r++ {
		in[r] = archsimd.LoadInt16x8Array((*[8]int16)(tmp[r*32+c0:]))
	}
	var out [16]archsimd.Int16x8
	itx16Net(inRows, &in, &out)
	one := archsimd.BroadcastInt16x8(1)
	sh1 := archsimd.BroadcastInt16x8(-1)
	sh3 := archsimd.BroadcastInt16x8(-3)
	for r := 0; r < 16; r++ {
		res := itx32Round2(out[r], sh3, sh1, one)
		rows := [2]uint64{*(*uint64)(unsafe.Add(pred, r*predStride+c0)), 0}
		p := archsimd.LoadUint64x2Array(&rows).ReshapeToUint8s().ExtendLo8ToUint16().BitsToInt16()
		*(*uint64)(unsafe.Add(dst, r*dstStride+c0)) = p.Add(res).SaturateToUint8().ReshapeToUint64s().GetElem(0)
	}
}

// InverseAddDCT32x16 is InverseAddDCT16x32 for the 32 wide, 16 high block.
func InverseAddDCT32x16(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, activeRows, activeCols int) bool {
	if activeRows <= 0 || activeRows > 16 || activeCols <= 0 || activeCols > 32 ||
		coeffStride < 16 || len(coeff) < 31*coeffStride+16 || dstStride < 32 || predStride < 32 ||
		len(dst) < 15*dstStride+32 || len(pred) < 15*predStride+32 {
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
		itx32x16ColStripAdd(&tmp, c0, inRows, dp, dstStride, pp, predStride)
	}
	return true
}
