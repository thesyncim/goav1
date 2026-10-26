// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && amd64 && !purego

package transform

import (
	"encoding/binary"
	"simd/archsimd"
)

// The fused 8-bit 4x4 inverse transform + add for amd64, in the form of
// dav1d's inv_txfm_add_4x4 (itx_avx2.asm): the coefficients are narrowed to
// int16 once, every 4-point network is vpmaddwd on interleaved input pairs
// (dav1d's ITX_MUL2X_PACK / the ADST4 pmaddwd sums), and the block is
// transposed once between the passes.
//
// Unlike the larger fused kernels it needs no guard. At 8 bits the staged
// path clamps the row inputs, the row outputs and the column inputs to
// int16 (rowBits = colBits = 16), which is exactly what the saturating
// vpackssdw narrows do; the 4-point networks have no other intermediate
// clamp, and the int32 dot products and sums are exact for int16 inputs.
// The DCT4 column outputs are clamped to int16 in the staged path but not
// here: a lane where the clamp would bite has |Round2(x, 4)| >= 2048, so
// both forms clip the reconstructed sample to the same 0 or 255.

// itx4Pair* are the vpmaddwd constant pairs (a, b) applied to the input
// pairs (in0, in2) or (in1, in3) of each lane.
var (
	itx4PairDCTEvenSum  = [8]int16{181, 181, 181, 181, 181, 181, 181, 181}
	itx4PairDCTEvenDiff = [8]int16{181, -181, 181, -181, 181, -181, 181, -181}
	itx4PairDCTOdd2     = [8]int16{1567, -3784, 1567, -3784, 1567, -3784, 1567, -3784}
	itx4PairDCTOdd3     = [8]int16{3784, 1567, 3784, 1567, 3784, 1567, 3784, 1567}

	// ADST4 (dav1d's inv_adst4_1d_internal with the sinpi constants
	// 1321, 2482, 3344, 3803 folded back: Round2(v - 4096k, 12) + k ==
	// Round2(v, 12)). Even pairs are (in0, in2), odd pairs (in1, in3).
	itx4PairADST0E = [8]int16{1321, 3803, 1321, 3803, 1321, 3803, 1321, 3803}
	itx4PairADST0O = [8]int16{3344, 2482, 3344, 2482, 3344, 2482, 3344, 2482}
	itx4PairADST1E = [8]int16{2482, -1321, 2482, -1321, 2482, -1321, 2482, -1321}
	itx4PairADST1O = [8]int16{3344, -3803, 3344, -3803, 3344, -3803, 3344, -3803}
	itx4PairADST2E = [8]int16{209, -209, 209, -209, 209, -209, 209, -209}
	itx4PairADST2O = [8]int16{0, 209, 0, 209, 0, 209, 0, 209}
	itx4PairADST3E = [8]int16{3803, 2482, 3803, 2482, 3803, 2482, 3803, 2482}
	itx4PairADST3O = [8]int16{-3344, -1321, -3344, -1321, -3344, -1321, -3344, -1321}

	// Identity4: x + Round2(x*1697, 12) == Round2(x*5793, 12).
	itx4PairIdentLo = [8]int16{5793, 0, 5793, 0, 5793, 0, 5793, 0}
	itx4PairIdentHi = [8]int16{0, 5793, 0, 5793, 0, 5793, 0, 5793}

	itx4Bias8  = [4]int32{128, 128, 128, 128}
	itx4Bias12 = [4]int32{2048, 2048, 2048, 2048}
	itx4Bias4  = [4]int32{8, 8, 8, 8}

	// itx4PairRows interleaves the int16 lanes [a0 a1 a2 a3 b0 b1 b2 b3] of
	// one vpackssdw into the pairs (a0 b0) (a1 b1) (a2 b2) (a3 b3).
	itx4PairRows = [16]int8{0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15}
	// itx4PairCols regroups [a0 a1 a2 a3 b0 b1 b2 b3] into (a0 a2) (b0 b2)
	// (a1 a3) (b1 b3): the even and odd input pairs of columns a and b.
	itx4PairCols = [16]int8{0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15}
	// itx4RowMask[n] keeps the pairs of rows 0..n-1 (the staged path reads
	// no coefficient row at or past activeRows).
	itx4RowMask = [5][8]int16{
		{},
		{-1, -1},
		{-1, -1, -1, -1},
		{-1, -1, -1, -1, -1, -1},
		{-1, -1, -1, -1, -1, -1, -1, -1},
	}
	// itx4LoBytes gathers the low byte of every int16 lane.
	itx4LoBytes = [16]int8{0, 2, 4, 6, 8, 10, 12, 14, -1, -1, -1, -1, -1, -1, -1, -1}
)

// itx4Types holds the (vertical, horizontal) 4-point networks of every
// transform type; IDTX is left to its own path.
var itx4Types = [TypeCount][2]tx1DType{
	TypeDCTDCT:           {tx1DDCT, tx1DDCT},
	TypeADSTDCT:          {tx1DADST, tx1DDCT},
	TypeDCTADST:          {tx1DDCT, tx1DADST},
	TypeADSTADST:         {tx1DADST, tx1DADST},
	TypeFlipADSTDCT:      {tx1DFlipADST, tx1DDCT},
	TypeDCTFlipADST:      {tx1DDCT, tx1DFlipADST},
	TypeFlipADSTFlipADST: {tx1DFlipADST, tx1DFlipADST},
	TypeADSTFlipADST:     {tx1DADST, tx1DFlipADST},
	TypeFlipADSTADST:     {tx1DFlipADST, tx1DADST},
	TypeIDTX:             {tx1DIdentity, tx1DIdentity},
	TypeVDCT:             {tx1DDCT, tx1DIdentity},
	TypeHDCT:             {tx1DIdentity, tx1DDCT},
	TypeVADST:            {tx1DADST, tx1DIdentity},
	TypeHADST:            {tx1DIdentity, tx1DADST},
	TypeVFlipADST:        {tx1DFlipADST, tx1DIdentity},
	TypeHFlipADST:        {tx1DIdentity, tx1DFlipADST},
}

func itx4Dot(p archsimd.Int16x8, k *[8]int16) archsimd.Int32x4 {
	return p.DotProductPairs(archsimd.LoadInt16x8Array(k))
}

// itx4Net runs the 4-point network t on four lanes, from the input pairs
// e = (in0, in2) and o = (in1, in3), and returns outputs 0..3.
func itx4Net(t tx1DType, e, o archsimd.Int16x8) (y0, y1, y2, y3 archsimd.Int32x4) {
	b12 := archsimd.LoadInt32x4Array(&itx4Bias12)
	switch t {
	case tx1DDCT:
		b8 := archsimd.LoadInt32x4Array(&itx4Bias8)
		t0 := itx4Dot(e, &itx4PairDCTEvenSum).Add(b8).ShiftAllRight(8)
		t1 := itx4Dot(e, &itx4PairDCTEvenDiff).Add(b8).ShiftAllRight(8)
		t2 := itx4Dot(o, &itx4PairDCTOdd2).Add(b12).ShiftAllRight(12)
		t3 := itx4Dot(o, &itx4PairDCTOdd3).Add(b12).ShiftAllRight(12)
		return t0.Add(t3), t1.Add(t2), t1.Sub(t2), t0.Sub(t3)
	case tx1DIdentity:
		y0 = itx4Dot(e, &itx4PairIdentLo).Add(b12).ShiftAllRight(12)
		y1 = itx4Dot(o, &itx4PairIdentLo).Add(b12).ShiftAllRight(12)
		y2 = itx4Dot(e, &itx4PairIdentHi).Add(b12).ShiftAllRight(12)
		y3 = itx4Dot(o, &itx4PairIdentHi).Add(b12).ShiftAllRight(12)
		return y0, y1, y2, y3
	}
	b8 := archsimd.LoadInt32x4Array(&itx4Bias8)
	y0 = itx4Dot(e, &itx4PairADST0E).Add(itx4Dot(o, &itx4PairADST0O)).Add(b12).ShiftAllRight(12)
	y1 = itx4Dot(e, &itx4PairADST1E).Add(itx4Dot(o, &itx4PairADST1O)).Add(b12).ShiftAllRight(12)
	y2 = itx4Dot(e, &itx4PairADST2E).Add(itx4Dot(o, &itx4PairADST2O)).Add(b8).ShiftAllRight(8)
	y3 = itx4Dot(e, &itx4PairADST3E).Add(itx4Dot(o, &itx4PairADST3O)).Add(b12).ShiftAllRight(12)
	if t == tx1DFlipADST {
		return y3, y2, y1, y0
	}
	return y0, y1, y2, y3
}

// itx4Shuffle applies the byte shuffle tbl to the int16 lanes of x.
func itx4Shuffle(x archsimd.Int16x8, tbl *[16]int8) archsimd.Int16x8 {
	return x.AsUint8x16().PermuteOrZero(archsimd.LoadInt8x16Array(tbl)).AsInt16x8()
}

// InverseAdd4x4 reconstructs one 8-bit 4x4 block of any transform type but
// IDTX: dst = clip(pred + inverse(coeff)), with the dequantized
// coefficients column-major at coeffStride and coded in rows
// 0..activeRows-1. pred may be dst. It reports
// false, with dst untouched, when the block is outside the kernel's
// contract; the caller then runs the staged path, which produces the same
// bytes.
func InverseAdd4x4(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, txType Type, activeRows int) bool {
	if !itxAVX2 || txType == TypeIDTX || uint(txType) >= uint(TypeCount) || uint(activeRows-1) > 3 || coeffStride < 4 ||
		len(coeff) < 3*coeffStride+4 || dstStride < 4 || predStride < 4 ||
		len(dst) < 3*dstStride+4 || len(pred) < 3*predStride+4 {
		return false
	}
	types := itx4Types[txType]

	// Row pass: lane r of column vector c_k is row r, so the networks run
	// on the columns as loaded; the narrows are the staged input clamp.
	c0 := archsimd.LoadInt32x4(coeff[0:4])
	c1 := archsimd.LoadInt32x4(coeff[coeffStride : coeffStride+4])
	c2 := archsimd.LoadInt32x4(coeff[2*coeffStride : 2*coeffStride+4])
	c3 := archsimd.LoadInt32x4(coeff[3*coeffStride : 3*coeffStride+4])
	m := archsimd.LoadInt16x8Array(&itx4RowMask[activeRows])
	e := itx4Shuffle(c0.SaturateToInt16Concat(c2), &itx4PairRows).And(m)
	o := itx4Shuffle(c1.SaturateToInt16Concat(c3), &itx4PairRows).And(m)
	r0, r1, r2, r3 := itx4Net(types[1], e, o)

	// Transpose: r_j holds column j of rows 0..3; the narrows are the
	// staged mid-pass clamp (shift 0 at 4x4).
	a := itx4Shuffle(r0.SaturateToInt16Concat(r1), &itx4PairCols).AsInt64x2()
	b := itx4Shuffle(r2.SaturateToInt16Concat(r3), &itx4PairCols).AsInt64x2()
	e = a.InterleaveLo(b).AsInt16x8()
	o = a.InterleaveHi(b).AsInt16x8()
	y0, y1, y2, y3 := itx4Net(types[0], e, o)

	// Round2(y, 4), add the prediction and clip.
	b4 := archsimd.LoadInt32x4Array(&itx4Bias4)
	z01 := y0.Add(b4).ShiftAllRight(4).SaturateToInt16Concat(y1.Add(b4).ShiftAllRight(4))
	z23 := y2.Add(b4).ShiftAllRight(4).SaturateToInt16Concat(y3.Add(b4).ShiftAllRight(4))
	var p01, p23 archsimd.Uint32x4
	p01 = p01.SetElem(0, binary.LittleEndian.Uint32(pred[0:4]))
	p01 = p01.SetElem(1, binary.LittleEndian.Uint32(pred[predStride:predStride+4]))
	p23 = p23.SetElem(0, binary.LittleEndian.Uint32(pred[2*predStride:2*predStride+4]))
	p23 = p23.SetElem(1, binary.LittleEndian.Uint32(pred[3*predStride:3*predStride+4]))
	zero, max := archsimd.Int16x8{}, archsimd.BroadcastInt16x8(255)
	z01 = p01.AsUint8x16().ExtendLo8ToUint16().AsInt16x8().AddSaturated(z01).Max(zero).Min(max)
	z23 = p23.AsUint8x16().ExtendLo8ToUint16().AsInt16x8().AddSaturated(z23).Max(zero).Min(max)
	q01 := itx4Shuffle(z01, &itx4LoBytes).AsUint32x4()
	q23 := itx4Shuffle(z23, &itx4LoBytes).AsUint32x4()
	binary.LittleEndian.PutUint32(dst[0:4], q01.GetElem(0))
	binary.LittleEndian.PutUint32(dst[dstStride:dstStride+4], q01.GetElem(1))
	binary.LittleEndian.PutUint32(dst[2*dstStride:2*dstStride+4], q23.GetElem(0))
	binary.LittleEndian.PutUint32(dst[3*dstStride:3*dstStride+4], q23.GetElem(1))
	return true
}
