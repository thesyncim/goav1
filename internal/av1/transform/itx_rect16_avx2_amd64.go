// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && amd64 && !purego

package transform

import "simd/archsimd"

// The fused 8-bit 4x16, 16x4, 8x16 and 16x8 inverse transform + add for
// amd64, in the form of dav1d's inv_txfm_add_{4x16,16x4,8x16,16x8}
// (itx_avx2.asm). They are built from the pieces of the 4x4, 4x8/8x4 and
// 16x16 kernels: the exact 4-point vpmaddwd network (itx4Net), the int16
// 8- and 16-point networks (itx8Net1D, itx16Net1D) under their per-lane L1
// guards, and one transpose between the passes. A block a guard declines is
// left untouched for the staged path.
//
// These shapes round the row outputs by one bit before the column pass, so
// the staged clamps are reproduced with care:
//   - DCT4 clamps its outputs to the row range (int16) before the rounding;
//     ADST4 and identity4 do not clamp at all, so they round in int32 and
//     the narrow saturates (the staged mid-pass clamp).
//   - Identity8 rows give Round2(2x, 1) == x.
//   - Identity16 rows round Round2(x*5793, 11) by one more bit in int32.
//   - Identity8 and identity16 columns saturate in int16; where that differs
//     from the unclamped staged value, |Round2(x, 4)| >= 2048 and both clip
//     the sample to 0 or 255.

var (
	itxK5793x16 = [16]int16{5793, 5793, 5793, 5793, 5793, 5793, 5793, 5793, 5793, 5793, 5793, 5793, 5793, 5793, 5793, 5793}
	itxOne16    = [8]int32{1, 1, 1, 1, 1, 1, 1, 1}
)

// itxRect16RowMask[n] keeps the lanes of rows 0..n-1 of a sixteen-row
// column loaded with itxLoadNarrow16 (itxRowPerm lane order).
var itxRect16RowMask = func() (m [17][16]int16) {
	for n := range m {
		for lane, row := range itxRowPerm {
			if row < n {
				m[n][lane] = -1
			}
		}
	}
	return m
}()

// itx8LaneMask[n] keeps int16 lanes 0..n-1.
var itx8LaneMask = func() (m [9][8]int16) {
	for n := range m {
		for i := 0; i < n; i++ {
			m[n][i] = -1
		}
	}
	return m
}()

// itxIdent16Row is the identity16 row output rounded by one bit,
// Round2(Round2(x*5793, 11), 1), narrowed with saturation.
func itxIdent16Row(x archsimd.Int16x16) archsimd.Int16x16 {
	k := archsimd.BroadcastInt16x16(5793).InterleaveLoGrouped(archsimd.BroadcastInt16x16(1 << 10))
	one16 := archsimd.BroadcastInt16x16(1)
	one := archsimd.LoadInt32x8Array(&itxOne16)
	lo := x.InterleaveLoGrouped(one16).DotProductPairs(k).ShiftAllRight(11).Add(one).ShiftAllRight(1)
	hi := x.InterleaveHiGrouped(one16).DotProductPairs(k).ShiftAllRight(11).Add(one).ShiftAllRight(1)
	return lo.SaturateToInt16ConcatGrouped(hi)
}

// itxRect16RowNet runs the 16-point row network t over in[0..n-1] and
// rounds the outputs by one bit.
func itxRect16RowNet(t tx1DType, n int, in, out *[16]archsimd.Int16x16) {
	if t == tx1DIdentity {
		for i := range in {
			out[i] = itxIdent16Row(in[i])
		}
		return
	}
	itx16Net1D(t, n, in, out)
	for i := range out {
		out[i] = itxRound2x16(out[i], 1)
	}
}

// itxRect16ColNet runs the 16-point column network t over in[0..n-1].
func itxRect16ColNet(t tx1DType, n int, in, out *[16]archsimd.Int16x16) {
	if t == tx1DIdentity {
		k := archsimd.LoadInt16x16Array(&itxK5793x16)
		for i := range in {
			out[i] = itx16Rot1(in[i], k, 11)
		}
		return
	}
	itx16Net1D(t, n, in, out)
}

// itxRect8RowNet runs the 8-point row network t over in[0..n-1] and rounds
// the outputs by one bit.
func itxRect8RowNet(t tx1DType, n int, in, out *[8]archsimd.Int16x8) {
	if t == tx1DIdentity {
		*out = *in
		return
	}
	itx8Net1D(t, n, in, out)
	for i := range out {
		out[i] = itxRound2x8(out[i], 1)
	}
}

// itx4RowsRound1 is the 4-point row pass of four rows from the input pairs
// e and o, rounded by one bit (DCT4 clamps first, see above).
func itx4RowsRound1(t tx1DType, e, o archsimd.Int16x8) (r0, r1, r2, r3 archsimd.Int32x4) {
	r0, r1, r2, r3 = itx4Net(t, e, o)
	if t == tx1DDCT {
		lo, hi := archsimd.BroadcastInt32x4(-32768), archsimd.BroadcastInt32x4(32767)
		r0, r1, r2, r3 = r0.Max(lo).Min(hi), r1.Max(lo).Min(hi), r2.Max(lo).Min(hi), r3.Max(lo).Min(hi)
	}
	one := archsimd.BroadcastInt32x4(1)
	return r0.Add(one).ShiftAllRight(1), r1.Add(one).ShiftAllRight(1), r2.Add(one).ShiftAllRight(1), r3.Add(one).ShiftAllRight(1)
}

// itxFoldL1 folds the per-row-parity halves of an itx48Rows4 guard sum into
// the column totals in lanes 0..3.
func itxFoldL1(l1 archsimd.Uint16x8) archsimd.Uint16x8 {
	return l1.AddSaturated(l1.AsUint64x2().InterleaveHi(l1.AsUint64x2()).AsUint16x8())
}

// InverseAdd4x16 reconstructs one 8-bit 4x16 block (4 wide, 16 tall) of any
// transform type but IDTX, under the contract of InverseAdd4x8.
func InverseAdd4x16(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, txType Type, activeRows, activeCols int) bool {
	if !itxAVX2 || txType == TypeIDTX || uint(txType) >= uint(TypeCount) ||
		uint(activeRows-1) > 15 || uint(activeCols-1) > 3 || coeffStride < 16 ||
		len(coeff) < (activeCols-1)*coeffStride+16 || dstStride < 4 || predStride < 4 ||
		len(dst) < 15*dstStride+4 || len(pred) < 15*predStride+4 {
		return false
	}
	types := itx4Types[txType]
	var in, out [16]archsimd.Int16x16
	var rows [4]archsimd.Int16x8
	var l1 archsimd.Uint16x8
	for g := 0; g < 16 && g < activeRows; g += 4 {
		var c [4]archsimd.Int32x4
		for k := 0; k < activeCols; k++ {
			c[k] = archsimd.LoadInt32x4(coeff[k*coeffStride+g : k*coeffStride+g+4])
		}
		m := archsimd.LoadInt16x8Array(&itx4RowMask[min(activeRows-g, 4)])
		e := itx4Shuffle(c[0].SaturateToInt16Concat(c[2]), &itx4PairRows).And(m)
		o := itx4Shuffle(c[1].SaturateToInt16Concat(c[3]), &itx4PairRows).And(m)
		r0, r1, r2, r3 := itx4RowsRound1(types[1], e, o)
		itx48Rows4(r0, r1, r2, r3, rows[:], &l1)
		for i := range rows {
			in[g+i] = in[g+i].SetLo(rows[i])
		}
	}
	if !itxL1Ok8(itxFoldL1(l1)) {
		archsimd.ClearAVXUpperBits()
		return false
	}
	itxRect16ColNet(types[0], activeRows, &in, &out)
	for r := 0; r < 16; r += 2 {
		z := itxRound2x8(out[r].GetLo(), 4).AsInt64x2().InterleaveLo(itxRound2x8(out[r+1].GetLo(), 4).AsInt64x2()).AsInt16x8()
		itx48AddStore4x2(dst[r*dstStride:], dstStride, pred[r*predStride:], predStride, z)
	}
	archsimd.ClearAVXUpperBits()
	return true
}

// itxRoundPack4 rounds two 4-column output groups by four bits and packs
// them into one row of eight int16 lanes.
func itxRoundPack4(a, b archsimd.Int32x4) archsimd.Int16x8 {
	b4 := archsimd.LoadInt32x4Array(&itx4Bias4)
	return a.Add(b4).ShiftAllRight(4).SaturateToInt16Concat(b.Add(b4).ShiftAllRight(4))
}

// InverseAdd16x4 is InverseAdd4x16 for the 16x4 block (16 wide, 4 tall).
func InverseAdd16x4(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, txType Type, activeRows, activeCols int) bool {
	if !itxAVX2 || txType == TypeIDTX || uint(txType) >= uint(TypeCount) ||
		uint(activeRows-1) > 3 || uint(activeCols-1) > 15 || coeffStride < 4 ||
		len(coeff) < (activeCols-1)*coeffStride+4 || dstStride < 16 || predStride < 16 ||
		len(dst) < 3*dstStride+16 || len(pred) < 3*predStride+16 {
		return false
	}
	types := itx4Types[txType]
	var in, out [16]archsimd.Int16x16
	var l1 archsimd.Uint16x8
	var zero archsimd.Int32x4
	m := archsimd.LoadInt16x8Array(&itx48RowMask[activeRows])
	for k := 0; k < activeCols; k++ {
		x := archsimd.LoadInt32x4(coeff[k*coeffStride : k*coeffStride+4]).SaturateToInt16Concat(zero).And(m)
		l1 = l1.AddSaturated(x.Abs().AsUint16x8())
		in[k] = in[k].SetLo(x)
	}
	if !itxL1Ok8(l1) {
		archsimd.ClearAVXUpperBits()
		return false
	}
	itxRect16RowNet(types[1], activeCols, &in, &out)
	var y [4][4]archsimd.Int32x4
	for g := 0; g < 4; g++ {
		y[g][0], y[g][1], y[g][2], y[g][3] = itx48Cols4(types[0], out[4*g].GetLo(), out[4*g+1].GetLo(), out[4*g+2].GetLo(), out[4*g+3].GetLo())
	}
	for r := 0; r < 4; r++ {
		itxAddStore8(dst[r*dstStride:], pred[r*predStride:], itxRoundPack4(y[0][r], y[1][r]))
		itxAddStore8(dst[r*dstStride+8:], pred[r*predStride+8:], itxRoundPack4(y[2][r], y[3][r]))
	}
	archsimd.ClearAVXUpperBits()
	return true
}

// InverseAdd8x16 is InverseAdd4x16 for the 8x16 block (8 wide, 16 tall,
// rect2 scale).
func InverseAdd8x16(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, txType Type, activeRows, activeCols int) bool {
	if !itxAVX2 || txType == TypeIDTX || uint(txType) >= uint(TypeCount) ||
		uint(activeRows-1) > 15 || uint(activeCols-1) > 7 || coeffStride < 16 ||
		len(coeff) < (activeCols-1)*coeffStride+16 || dstStride < 8 || predStride < 8 ||
		len(dst) < 15*dstStride+8 || len(pred) < 15*predStride+8 {
		return false
	}
	types := itx4Types[txType]
	// Row pass: column k loads in itxRowPerm lane order; the low half holds
	// rows 0..3 and 8..11, the high half rows 4..7 and 12..15.
	var a, b, oa, ob [8]archsimd.Int16x8
	var l1 archsimd.Uint16x16
	m := archsimd.LoadInt16x16Array(&itxRect16RowMask[activeRows])
	for k := 0; k < activeCols; k++ {
		x := itxLoadNarrowRect2x16(coeff[k*coeffStride : k*coeffStride+16]).And(m)
		l1 = l1.AddSaturated(x.Abs().AsUint16x16())
		a[k], b[k] = x.GetLo(), x.GetHi()
	}
	if !itxL1Ok16(l1) {
		archsimd.ClearAVXUpperBits()
		return false
	}
	itxRect8RowNet(types[1], activeCols, &a, &oa)
	itxRect8RowNet(types[1], activeCols, &b, &ob)
	itxTranspose8x8(&oa)
	itxTranspose8x8(&ob)
	var in, out [16]archsimd.Int16x16
	var cl1 archsimd.Uint16x8
	for e := 0; e < 8; e++ {
		in[itxRowPerm[e]] = in[itxRowPerm[e]].SetLo(oa[e])
		in[itxRowPerm[8+e]] = in[itxRowPerm[8+e]].SetLo(ob[e])
		cl1 = cl1.AddSaturated(oa[e].Abs().AsUint16x8()).AddSaturated(ob[e].Abs().AsUint16x8())
	}
	if !itxL1Ok8(cl1) {
		archsimd.ClearAVXUpperBits()
		return false
	}
	itxRect16ColNet(types[0], activeRows, &in, &out)
	for r := 0; r < 16; r++ {
		itxAddStore8(dst[r*dstStride:], pred[r*predStride:], itxRound2x8(out[r].GetLo(), 4))
	}
	archsimd.ClearAVXUpperBits()
	return true
}

// InverseAdd16x8 is InverseAdd4x16 for the 16x8 block (16 wide, 8 tall,
// rect2 scale).
func InverseAdd16x8(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, txType Type, activeRows, activeCols int) bool {
	if !itxAVX2 || txType == TypeIDTX || uint(txType) >= uint(TypeCount) ||
		uint(activeRows-1) > 7 || uint(activeCols-1) > 15 || coeffStride < 8 ||
		len(coeff) < (activeCols-1)*coeffStride+8 || dstStride < 16 || predStride < 16 ||
		len(dst) < 7*dstStride+16 || len(pred) < 7*predStride+16 {
		return false
	}
	types := itx4Types[txType]
	var in, out [16]archsimd.Int16x16
	var l1 archsimd.Uint16x8
	m := archsimd.LoadInt16x8Array(&itx8LaneMask[activeRows])
	for k := 0; k < activeCols; k++ {
		x := itx48Rect2x4(coeff[k*coeffStride : k*coeffStride+4]).SaturateToInt16Concat(itx48Rect2x4(coeff[k*coeffStride+4 : k*coeffStride+8])).And(m)
		l1 = l1.AddSaturated(x.Abs().AsUint16x8())
		in[k] = in[k].SetLo(x)
	}
	if !itxL1Ok8(l1) {
		archsimd.ClearAVXUpperBits()
		return false
	}
	itxRect16RowNet(types[1], activeCols, &in, &out)
	// Column pass, columns 0..7 and 8..15: c[r] is row r across the columns.
	var ca, cb, oa, ob [8]archsimd.Int16x8
	for j := 0; j < 8; j++ {
		ca[j], cb[j] = out[j].GetLo(), out[8+j].GetLo()
	}
	itxTranspose8x8(&ca)
	itxTranspose8x8(&cb)
	var la, lb archsimd.Uint16x8
	for r := 0; r < 8; r++ {
		la = la.AddSaturated(ca[r].Abs().AsUint16x8())
		lb = lb.AddSaturated(cb[r].Abs().AsUint16x8())
	}
	if !itxL1Ok8(la) || !itxL1Ok8(lb) {
		archsimd.ClearAVXUpperBits()
		return false
	}
	itx48Net8(types[0], activeRows, &ca, &oa)
	itx48Net8(types[0], activeRows, &cb, &ob)
	for r := 0; r < 8; r++ {
		itxAddStore8(dst[r*dstStride:], pred[r*predStride:], itxRound2x8(oa[r], 4))
		itxAddStore8(dst[r*dstStride+8:], pred[r*predStride+8:], itxRound2x8(ob[r], 4))
	}
	archsimd.ClearAVXUpperBits()
	return true
}
