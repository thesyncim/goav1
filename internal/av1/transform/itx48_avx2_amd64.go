// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && amd64 && !purego

package transform

import (
	"encoding/binary"
	"simd/archsimd"
)

// The fused 8-bit 4x8 and 8x4 inverse transform + add for amd64, in the
// form of dav1d's inv_txfm_add_4x8 / inv_txfm_add_8x4 (itx_avx2.asm): the
// rect2-scaled coefficients are narrowed to int16 once, the 4-point pass is
// the exact vpmaddwd network of the 4x4 kernel (itx4Net), the 8-point pass
// is the int16 network of the 8x8 kernel (itx8Net1D) under its per-lane L1
// guard, and the prediction add and clip happen in the same call. A block
// the guard declines is left untouched for the staged path.

// itx48RowMask[n] keeps int16 lanes 0..n-1 of a four-lane column vector.
var itx48RowMask = [5][8]int16{
	{},
	{-1},
	{-1, -1},
	{-1, -1, -1},
	{-1, -1, -1, -1},
}

// itx48Rect2x4 loads four int32 coefficients with the rect2 scale applied
// (see itxLoadNarrowRect2x16; the int16 narrow happens at the caller).
func itx48Rect2x4(p []int32) archsimd.Int32x4 {
	x := archsimd.LoadInt32x4(p[:4])
	x = x.Max(archsimd.BroadcastInt32x4(-itxRect2Pre)).Min(archsimd.BroadcastInt32x4(itxRect2Pre))
	return x.Mul(archsimd.BroadcastInt32x4(181)).Add(archsimd.BroadcastInt32x4(128)).ShiftAllRight(8)
}

// itx48Net8 is itx8Net1D extended with identity8 (x * 2; the saturation
// matches the staged clamps, see InverseAdd4x8).
func itx48Net8(t tx1DType, n int, in *[8]archsimd.Int16x8, out *[8]archsimd.Int16x8) {
	if t == tx1DIdentity {
		for i := range in {
			out[i] = in[i].AddSaturated(in[i])
		}
		return
	}
	itx8Net1D(t, n, in, out)
}

// itx48AddStore4x2 adds the residual rows z = [row r | row r+1] (four
// int16 lanes each) to the prediction rows r and r+1 and stores the
// clipped bytes.
func itx48AddStore4x2(dst []byte, dstStride int, pred []byte, predStride int, z archsimd.Int16x8) {
	var p archsimd.Uint32x4
	p = p.SetElem(0, binary.LittleEndian.Uint32(pred[0:4]))
	p = p.SetElem(1, binary.LittleEndian.Uint32(pred[predStride:predStride+4]))
	z = p.AsUint8x16().ExtendLo8ToUint16().AsInt16x8().AddSaturated(z).Max(archsimd.Int16x8{}).Min(archsimd.BroadcastInt16x8(255))
	q := itx4Shuffle(z, &itx4LoBytes).AsUint32x4()
	binary.LittleEndian.PutUint32(dst[0:4], q.GetElem(0))
	binary.LittleEndian.PutUint32(dst[dstStride:dstStride+4], q.GetElem(1))
}

// itx48Rows4 transposes the 4-point row outputs r0..r3 (lane i = row i,
// vector j = column j) into in[0..3] (lanes 0..3 = columns 0..3 of rows
// 0..3, upper lanes zero), narrowing with saturation (the staged mid-pass
// clamp), and adds the magnitudes to l1: lanes 0..3 take rows 0 and 2,
// lanes 4..7 rows 1 and 3, of columns 0..3.
func itx48Rows4(r0, r1, r2, r3 archsimd.Int32x4, in []archsimd.Int16x8, l1 *archsimd.Uint16x8) {
	_ = in[3]
	a := r0.SaturateToInt16Concat(r1)
	b := r2.SaturateToInt16Concat(r3)
	lo, hi := a.InterleaveLo(b), a.InterleaveHi(b)
	x01 := lo.InterleaveLo(hi).AsInt64x2()
	x23 := lo.InterleaveHi(hi).AsInt64x2()
	var zero archsimd.Int64x2
	in[0] = x01.InterleaveLo(zero).AsInt16x8()
	in[1] = x01.InterleaveHi(zero).AsInt16x8()
	in[2] = x23.InterleaveLo(zero).AsInt16x8()
	in[3] = x23.InterleaveHi(zero).AsInt16x8()
	*l1 = l1.AddSaturated(x01.AsInt16x8().Abs().AsUint16x8()).AddSaturated(x23.AsInt16x8().Abs().AsUint16x8())
}

// InverseAdd4x8 reconstructs one 8-bit 4x8 block (4 wide, 8 tall) of any
// transform type but IDTX: dst = clip(pred + inverse(coeff)), with the
// dequantized coefficients column-major at coeffStride, coded in rows
// 0..activeRows-1 and columns 0..activeCols-1. pred may be dst. It reports
// false, with dst untouched, when the block is outside the kernel's
// contract or the column guard declines; the caller then runs the staged
// path, which produces the same bytes.
//
// The 4-point row pass is exact (see itx4_avx2_amd64.go). The 8-point
// column pass runs under the L1 guard; identity8 is a saturating doubling,
// which equals the staged unclamped doubling wherever the final clip can
// tell them apart (|Round2(x, 4)| >= 2048 clips to 0 or 255 either way).
func InverseAdd4x8(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, txType Type, activeRows, activeCols int) bool {
	if !itxAVX2 || txType == TypeIDTX || uint(txType) >= uint(TypeCount) ||
		uint(activeRows-1) > 7 || uint(activeCols-1) > 3 || coeffStride < 8 ||
		len(coeff) < (activeCols-1)*coeffStride+8 || dstStride < 4 || predStride < 4 ||
		len(dst) < 7*dstStride+4 || len(pred) < 7*predStride+4 {
		return false
	}
	types := itx4Types[txType]

	// Row pass, rows 0..3 and 4..7: lane i of a column vector is row i.
	var lo, hi [4]archsimd.Int32x4
	for k := 0; k < activeCols; k++ {
		lo[k] = itx48Rect2x4(coeff[k*coeffStride : k*coeffStride+4])
		hi[k] = itx48Rect2x4(coeff[k*coeffStride+4 : k*coeffStride+8])
	}
	mLo := archsimd.LoadInt16x8Array(&itx4RowMask[min(activeRows, 4)])
	mHi := archsimd.LoadInt16x8Array(&itx4RowMask[max(activeRows-4, 0)])
	e := itx4Shuffle(lo[0].SaturateToInt16Concat(lo[2]), &itx4PairRows).And(mLo)
	o := itx4Shuffle(lo[1].SaturateToInt16Concat(lo[3]), &itx4PairRows).And(mLo)
	r0, r1, r2, r3 := itx4Net(types[1], e, o)
	e = itx4Shuffle(hi[0].SaturateToInt16Concat(hi[2]), &itx4PairRows).And(mHi)
	o = itx4Shuffle(hi[1].SaturateToInt16Concat(hi[3]), &itx4PairRows).And(mHi)
	s0, s1, s2, s3 := itx4Net(types[1], e, o)

	// Column pass: in[r] is row r across the four columns.
	var in, out [8]archsimd.Int16x8
	var l1 archsimd.Uint16x8
	itx48Rows4(r0, r1, r2, r3, in[0:4], &l1)
	itx48Rows4(s0, s1, s2, s3, in[4:8], &l1)
	// Lanes 0..3 summed the even rows and lanes 4..7 the odd rows of each
	// column; fold them into the column totals.
	l1 = l1.AddSaturated(l1.AsUint64x2().InterleaveHi(l1.AsUint64x2()).AsUint16x8())
	if !itxL1Ok8(l1) {
		return false
	}
	itx48Net8(types[0], activeRows, &in, &out)
	for r := 0; r < 8; r += 2 {
		z := itxRound2x8(out[r], 4).AsInt64x2().InterleaveLo(itxRound2x8(out[r+1], 4).AsInt64x2()).AsInt16x8()
		itx48AddStore4x2(dst[r*dstStride:], dstStride, pred[r*predStride:], predStride, z)
	}
	return true
}

// itx48Cols4 runs the 4-point column network t over four columns whose
// 8-point row outputs are v0..v3 (lanes 0..3 = rows 0..3) and returns the
// output rows 0..3 (lane i = column i).
func itx48Cols4(t tx1DType, v0, v1, v2, v3 archsimd.Int16x8) (y0, y1, y2, y3 archsimd.Int32x4) {
	a := itx4Shuffle(v0.AsInt64x2().InterleaveLo(v1.AsInt64x2()).AsInt16x8(), &itx4PairCols).AsInt64x2()
	b := itx4Shuffle(v2.AsInt64x2().InterleaveLo(v3.AsInt64x2()).AsInt16x8(), &itx4PairCols).AsInt64x2()
	return itx4Net(t, a.InterleaveLo(b).AsInt16x8(), a.InterleaveHi(b).AsInt16x8())
}

// InverseAdd8x4 is InverseAdd4x8 for the 8x4 block (8 wide, 4 tall): the
// 8-point row pass runs under the L1 guard and the 4-point column pass is
// exact.
func InverseAdd8x4(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, txType Type, activeRows, activeCols int) bool {
	if !itxAVX2 || txType == TypeIDTX || uint(txType) >= uint(TypeCount) ||
		uint(activeRows-1) > 3 || uint(activeCols-1) > 7 || coeffStride < 4 ||
		len(coeff) < (activeCols-1)*coeffStride+4 || dstStride < 8 || predStride < 8 ||
		len(dst) < 3*dstStride+8 || len(pred) < 3*predStride+8 {
		return false
	}
	types := itx4Types[txType]

	// Row pass: in[k] is coefficient column k, lane i = row i.
	var in, out [8]archsimd.Int16x8
	var l1 archsimd.Uint16x8
	var zero archsimd.Int32x4
	m := archsimd.LoadInt16x8Array(&itx48RowMask[activeRows])
	for k := 0; k < activeCols; k++ {
		x := itx48Rect2x4(coeff[k*coeffStride:k*coeffStride+4]).SaturateToInt16Concat(zero).And(m)
		l1 = l1.AddSaturated(x.Abs().AsUint16x8())
		in[k] = x
	}
	if !itxL1Ok8(l1) {
		return false
	}
	itx48Net8(types[1], activeCols, &in, &out)

	// Column pass, columns 0..3 and 4..7.
	y0, y1, y2, y3 := itx48Cols4(types[0], out[0], out[1], out[2], out[3])
	w0, w1, w2, w3 := itx48Cols4(types[0], out[4], out[5], out[6], out[7])
	b4 := archsimd.LoadInt32x4Array(&itx4Bias4)
	px := itxPix{dst, dstStride, pred, predStride, 0}
	itxAddStore8(px.dst, px.pred, y0.Add(b4).ShiftAllRight(4).SaturateToInt16Concat(w0.Add(b4).ShiftAllRight(4)))
	itxAddStore8(px.dst[dstStride:], px.pred[predStride:], y1.Add(b4).ShiftAllRight(4).SaturateToInt16Concat(w1.Add(b4).ShiftAllRight(4)))
	itxAddStore8(px.dst[2*dstStride:], px.pred[2*predStride:], y2.Add(b4).ShiftAllRight(4).SaturateToInt16Concat(w2.Add(b4).ShiftAllRight(4)))
	itxAddStore8(px.dst[3*dstStride:], px.pred[3*predStride:], y3.Add(b4).ShiftAllRight(4).SaturateToInt16Concat(w3.Add(b4).ShiftAllRight(4)))
	return true
}
