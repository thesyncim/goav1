// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && amd64 && !purego

package transform

import (
	"encoding/binary"
	"simd/archsimd"
)

// These index vectors duplicate each qword's low and high dwords. That lets
// one AVX2 widening multiply handle all four paired row/column lanes.
var (
	dctPairLowWords  = [8]uint32{0, 0, 2, 2, 4, 4, 6, 6}
	dctPairHighWords = [8]uint32{1, 1, 3, 3, 5, 5, 7, 7}
)

func dctPairLoad(p0, p1 []int32, stride int, index int) archsimd.Int64x4 {
	o := index * stride
	a, b := p0[o], p1[o]
	return archsimd.BroadcastInt64x4(int64(a)).InterleaveLoGrouped(archsimd.BroadcastInt64x4(int64(b)))
}

func dctPairStore(p0, p1 []int32, stride int, index int, v archsimd.Int64x4) {
	var pair [4]int64
	v.StoreArray(&pair)
	o := index * stride
	p0[o], p1[o] = int32(pair[0]), int32(pair[1])
}

// dctPairMul multiplies the signed qword lanes by a small signed coefficient.
// AVX2 has no qword multiply; split each value into signed low/high dwords,
// correct the high word when the low word's sign bit is set, and use VPMULDQ.
func dctPairMul(v archsimd.Int64x4, coeff int32, lowIndex, highIndex archsimd.Uint32x8) archsimd.Int64x4 {
	words := v.ToBits().ReshapeToUint32s()
	lo := words.Permute(lowIndex).BitsToInt32()
	hi := words.Permute(highIndex).BitsToInt32()
	hi = hi.Add(lo.ToBits().ShiftAllRight(31).BitsToInt32())
	c := archsimd.BroadcastInt32x8(coeff)
	loProduct := lo.MulWidenEven(c)
	hiProduct := hi.MulWidenEven(c)
	return loProduct.Add(hiProduct.ShiftAllLeft(32))
}

func dctPairRoundShift8(v archsimd.Int64x4) archsimd.Int64x4 {
	bias := v.Add(archsimd.BroadcastInt64x4((1 << 7) + (1 << 62)))
	return bias.ToBits().ShiftAllRight(8).BitsToInt64().Sub(archsimd.BroadcastInt64x4(1 << 54))
}

func dctPairRoundShift11(v archsimd.Int64x4) archsimd.Int64x4 {
	bias := v.Add(archsimd.BroadcastInt64x4((1 << 10) + (1 << 62)))
	return bias.ToBits().ShiftAllRight(11).BitsToInt64().Sub(archsimd.BroadcastInt64x4(1 << 51))
}

func dctPairRoundShift12(v archsimd.Int64x4) archsimd.Int64x4 {
	bias := v.Add(archsimd.BroadcastInt64x4((1 << 11) + (1 << 62)))
	return bias.ToBits().ShiftAllRight(12).BitsToInt64().Sub(archsimd.BroadcastInt64x4(1 << 50))
}

func dctPairClip(v, minv, maxv archsimd.Int64x4) archsimd.Int64x4 {
	v = minv.IfElse(v.Less(minv), v)
	return maxv.IfElse(v.Greater(maxv), v)
}

// inverseDCT8PairSIMD applies the scalar inverseDCT8 butterfly to two values
// per coefficient. Its inputs and outputs p0 and p1 may describe two rows or
// two adjacent columns; sample i of each is at i*stride.
func inverseDCT8PairSIMD(p0, p1 []int32, stride int, min, max int32) {
	lowIndex := archsimd.LoadUint32x8Array(&dctPairLowWords)
	highIndex := archsimd.LoadUint32x8Array(&dctPairHighWords)
	minv := archsimd.BroadcastInt64x4(int64(min))
	maxv := archsimd.BroadcastInt64x4(int64(max))

	// The even-index DCT4 pass.
	in0 := dctPairLoad(p0, p1, stride, 0)
	in2 := dctPairLoad(p0, p1, stride, 2)
	in4 := dctPairLoad(p0, p1, stride, 4)
	in6 := dctPairLoad(p0, p1, stride, 6)
	t0 := dctPairRoundShift8(dctPairMul(in0.Add(in4), 181, lowIndex, highIndex))
	t1 := dctPairRoundShift8(dctPairMul(in0.Sub(in4), 181, lowIndex, highIndex))
	t2 := dctPairRoundShift12(dctPairMul(in2, 1567, lowIndex, highIndex).
		Sub(dctPairMul(in6, 3784-4096, lowIndex, highIndex))).Sub(in6)
	t3 := dctPairRoundShift12(dctPairMul(in2, 3784-4096, lowIndex, highIndex).
		Add(dctPairMul(in6, 1567, lowIndex, highIndex))).Add(in2)
	e0 := dctPairClip(t0.Add(t3), minv, maxv)
	e1 := dctPairClip(t1.Add(t2), minv, maxv)
	e2 := dctPairClip(t1.Sub(t2), minv, maxv)
	e3 := dctPairClip(t0.Sub(t3), minv, maxv)

	// Odd-index rotations and their first clamp stage.
	in1 := dctPairLoad(p0, p1, stride, 1)
	in3 := dctPairLoad(p0, p1, stride, 3)
	in5 := dctPairLoad(p0, p1, stride, 5)
	in7 := dctPairLoad(p0, p1, stride, 7)
	t4a := dctPairRoundShift12(dctPairMul(in1, 799, lowIndex, highIndex).
		Sub(dctPairMul(in7, 4017-4096, lowIndex, highIndex))).Sub(in7)
	t5a := dctPairRoundShift11(dctPairMul(in5, 1703, lowIndex, highIndex).
		Sub(dctPairMul(in3, 1138, lowIndex, highIndex)))
	t6a := dctPairRoundShift11(dctPairMul(in5, 1138, lowIndex, highIndex).
		Add(dctPairMul(in3, 1703, lowIndex, highIndex)))
	t7a := dctPairRoundShift12(dctPairMul(in1, 4017-4096, lowIndex, highIndex).
		Add(dctPairMul(in7, 799, lowIndex, highIndex))).Add(in1)
	t4 := dctPairClip(t4a.Add(t5a), minv, maxv)
	t5a = dctPairClip(t4a.Sub(t5a), minv, maxv)
	t7 := dctPairClip(t7a.Add(t6a), minv, maxv)
	t6a = dctPairClip(t7a.Sub(t6a), minv, maxv)
	t5 := dctPairRoundShift8(dctPairMul(t6a.Sub(t5a), 181, lowIndex, highIndex))
	t6 := dctPairRoundShift8(dctPairMul(t6a.Add(t5a), 181, lowIndex, highIndex))

	// Final butterfly, with the same output clamp as inverseDCT8.
	dctPairStore(p0, p1, stride, 0, dctPairClip(e0.Add(t7), minv, maxv))
	dctPairStore(p0, p1, stride, 1, dctPairClip(e1.Add(t6), minv, maxv))
	dctPairStore(p0, p1, stride, 2, dctPairClip(e2.Add(t5), minv, maxv))
	dctPairStore(p0, p1, stride, 3, dctPairClip(e3.Add(t4), minv, maxv))
	dctPairStore(p0, p1, stride, 4, dctPairClip(e3.Sub(t4), minv, maxv))
	dctPairStore(p0, p1, stride, 5, dctPairClip(e2.Sub(t5), minv, maxv))
	dctPairStore(p0, p1, stride, 6, dctPairClip(e1.Sub(t6), minv, maxv))
	dctPairStore(p0, p1, stride, 7, dctPairClip(e0.Sub(t7), minv, maxv))
	archsimd.ClearAVXUpperBits()
}

func inverseDCT8Row2SIMDAdapter(r0, r1 []int32, min, max int32) {
	if len(r0) < dct8Size || len(r1) < dct8Size {
		inverseDCT8Row2PureGo(r0, r1, min, max)
		return
	}
	inverseDCT8PairSIMD(r0, r1, 1, min, max)
}

func inverseDCT8Col2SIMDAdapter(buf []int32, rowStride int, min, max int32) {
	if rowStride < 2 || len(buf) < (dct8Size-1)*rowStride+2 {
		inverseDCT8Col2PureGo(buf, rowStride, min, max)
		return
	}
	inverseDCT8PairSIMD(buf, buf[1:], rowStride, min, max)
}

// inverseDCT16PairSIMD applies inverseDCT16 to two rows or columns. The
// inverseDCT8 even pass is written inline so vectors stay live across stages.
func inverseDCT16PairSIMD(p0, p1 []int32, stride int, min, max int32) {
	lowIndex := archsimd.LoadUint32x8Array(&dctPairLowWords)
	highIndex := archsimd.LoadUint32x8Array(&dctPairHighWords)
	minv := archsimd.BroadcastInt64x4(int64(min))
	maxv := archsimd.BroadcastInt64x4(int64(max))
	evenStride := stride * 2

	// Inline inverseDCT8 on the even coefficients.
	in0 := dctPairLoad(p0, p1, evenStride, 0)
	in2 := dctPairLoad(p0, p1, evenStride, 2)
	in4 := dctPairLoad(p0, p1, evenStride, 4)
	in6 := dctPairLoad(p0, p1, evenStride, 6)
	t0 := dctPairRoundShift8(dctPairMul(in0.Add(in4), 181, lowIndex, highIndex))
	t1 := dctPairRoundShift8(dctPairMul(in0.Sub(in4), 181, lowIndex, highIndex))
	t2 := dctPairRoundShift12(dctPairMul(in2, 1567, lowIndex, highIndex).
		Sub(dctPairMul(in6, 3784-4096, lowIndex, highIndex))).Sub(in6)
	t3 := dctPairRoundShift12(dctPairMul(in2, 3784-4096, lowIndex, highIndex).
		Add(dctPairMul(in6, 1567, lowIndex, highIndex))).Add(in2)
	e0 := dctPairClip(t0.Add(t3), minv, maxv)
	e1 := dctPairClip(t1.Add(t2), minv, maxv)
	e2 := dctPairClip(t1.Sub(t2), minv, maxv)
	e3 := dctPairClip(t0.Sub(t3), minv, maxv)

	in1 := dctPairLoad(p0, p1, evenStride, 1)
	in3 := dctPairLoad(p0, p1, evenStride, 3)
	in5 := dctPairLoad(p0, p1, evenStride, 5)
	in7 := dctPairLoad(p0, p1, evenStride, 7)
	t4a := dctPairRoundShift12(dctPairMul(in1, 799, lowIndex, highIndex).
		Sub(dctPairMul(in7, 4017-4096, lowIndex, highIndex))).Sub(in7)
	t5a := dctPairRoundShift11(dctPairMul(in5, 1703, lowIndex, highIndex).
		Sub(dctPairMul(in3, 1138, lowIndex, highIndex)))
	t6a := dctPairRoundShift11(dctPairMul(in5, 1138, lowIndex, highIndex).
		Add(dctPairMul(in3, 1703, lowIndex, highIndex)))
	t7a := dctPairRoundShift12(dctPairMul(in1, 4017-4096, lowIndex, highIndex).
		Add(dctPairMul(in7, 799, lowIndex, highIndex))).Add(in1)
	t4 := dctPairClip(t4a.Add(t5a), minv, maxv)
	t5a = dctPairClip(t4a.Sub(t5a), minv, maxv)
	t7 := dctPairClip(t7a.Add(t6a), minv, maxv)
	t6a = dctPairClip(t7a.Sub(t6a), minv, maxv)
	t5 := dctPairRoundShift8(dctPairMul(t6a.Sub(t5a), 181, lowIndex, highIndex))
	t6 := dctPairRoundShift8(dctPairMul(t6a.Add(t5a), 181, lowIndex, highIndex))
	e4 := dctPairClip(e0.Add(t7), minv, maxv)
	e5 := dctPairClip(e1.Add(t6), minv, maxv)
	e6 := dctPairClip(e2.Add(t5), minv, maxv)
	e7 := dctPairClip(e3.Add(t4), minv, maxv)
	e4n := dctPairClip(e3.Sub(t4), minv, maxv)
	e5n := dctPairClip(e2.Sub(t5), minv, maxv)
	e6n := dctPairClip(e1.Sub(t6), minv, maxv)
	e7n := dctPairClip(e0.Sub(t7), minv, maxv)

	// Odd coefficients: first rotations and clamp stage.
	in1 = dctPairLoad(p0, p1, stride, 1)
	in3 = dctPairLoad(p0, p1, stride, 3)
	in5 = dctPairLoad(p0, p1, stride, 5)
	in7 = dctPairLoad(p0, p1, stride, 7)
	in9 := dctPairLoad(p0, p1, stride, 9)
	in11 := dctPairLoad(p0, p1, stride, 11)
	in13 := dctPairLoad(p0, p1, stride, 13)
	in15 := dctPairLoad(p0, p1, stride, 15)
	t8a := dctPairRoundShift12(dctPairMul(in1, 401, lowIndex, highIndex).
		Sub(dctPairMul(in15, 4076-4096, lowIndex, highIndex))).Sub(in15)
	t9a := dctPairRoundShift11(dctPairMul(in9, 1583, lowIndex, highIndex).
		Sub(dctPairMul(in7, 1299, lowIndex, highIndex)))
	t10a := dctPairRoundShift12(dctPairMul(in5, 1931, lowIndex, highIndex).
		Sub(dctPairMul(in11, 3612-4096, lowIndex, highIndex))).Sub(in11)
	t11a := dctPairRoundShift12(dctPairMul(in13, 3920-4096, lowIndex, highIndex).
		Sub(dctPairMul(in3, 1189, lowIndex, highIndex))).Add(in13)
	t12a := dctPairRoundShift12(dctPairMul(in13, 1189, lowIndex, highIndex).
		Add(dctPairMul(in3, 3920-4096, lowIndex, highIndex))).Add(in3)
	t13a := dctPairRoundShift12(dctPairMul(in5, 3612-4096, lowIndex, highIndex).
		Add(dctPairMul(in11, 1931, lowIndex, highIndex))).Add(in5)
	t14a := dctPairRoundShift11(dctPairMul(in9, 1299, lowIndex, highIndex).
		Add(dctPairMul(in7, 1583, lowIndex, highIndex)))
	t15a := dctPairRoundShift12(dctPairMul(in1, 4076-4096, lowIndex, highIndex).
		Add(dctPairMul(in15, 401, lowIndex, highIndex))).Add(in1)
	t8 := dctPairClip(t8a.Add(t9a), minv, maxv)
	t9 := dctPairClip(t8a.Sub(t9a), minv, maxv)
	t10 := dctPairClip(t11a.Sub(t10a), minv, maxv)
	t11 := dctPairClip(t11a.Add(t10a), minv, maxv)
	t12 := dctPairClip(t12a.Add(t13a), minv, maxv)
	t13 := dctPairClip(t12a.Sub(t13a), minv, maxv)
	t14 := dctPairClip(t15a.Sub(t14a), minv, maxv)
	t15 := dctPairClip(t15a.Add(t14a), minv, maxv)

	// Two odd butterfly stages and the final 16-point output permutation.
	t9a = dctPairRoundShift12(dctPairMul(t14, 1567, lowIndex, highIndex).
		Sub(dctPairMul(t9, 3784-4096, lowIndex, highIndex))).Sub(t9)
	t14a = dctPairRoundShift12(dctPairMul(t14, 3784-4096, lowIndex, highIndex).
		Add(dctPairMul(t9, 1567, lowIndex, highIndex))).Add(t14)
	t10a = dctPairRoundShift12(dctPairMul(t13, 4096-3784, lowIndex, highIndex).
		Sub(dctPairMul(t10, 1567, lowIndex, highIndex))).Sub(t13)
	t13a = dctPairRoundShift12(dctPairMul(t13, 1567, lowIndex, highIndex).
		Sub(dctPairMul(t10, 3784-4096, lowIndex, highIndex))).Sub(t10)
	t8a = dctPairClip(t8.Add(t11), minv, maxv)
	t9 = dctPairClip(t9a.Add(t10a), minv, maxv)
	t10 = dctPairClip(t9a.Sub(t10a), minv, maxv)
	t11a = dctPairClip(t8.Sub(t11), minv, maxv)
	t12a = dctPairClip(t15.Sub(t12), minv, maxv)
	t13 = dctPairClip(t14a.Sub(t13a), minv, maxv)
	t14 = dctPairClip(t14a.Add(t13a), minv, maxv)
	t15a = dctPairClip(t15.Add(t12), minv, maxv)
	t10a = dctPairRoundShift8(dctPairMul(t13.Sub(t10), 181, lowIndex, highIndex))
	t13a = dctPairRoundShift8(dctPairMul(t13.Add(t10), 181, lowIndex, highIndex))
	t11 = dctPairRoundShift8(dctPairMul(t12a.Sub(t11a), 181, lowIndex, highIndex))
	t12 = dctPairRoundShift8(dctPairMul(t12a.Add(t11a), 181, lowIndex, highIndex))

	dctPairStore(p0, p1, stride, 0, dctPairClip(e4.Add(t15a), minv, maxv))
	dctPairStore(p0, p1, stride, 1, dctPairClip(e5.Add(t14), minv, maxv))
	dctPairStore(p0, p1, stride, 2, dctPairClip(e6.Add(t13a), minv, maxv))
	dctPairStore(p0, p1, stride, 3, dctPairClip(e7.Add(t12), minv, maxv))
	dctPairStore(p0, p1, stride, 4, dctPairClip(e4n.Add(t11), minv, maxv))
	dctPairStore(p0, p1, stride, 5, dctPairClip(e5n.Add(t10a), minv, maxv))
	dctPairStore(p0, p1, stride, 6, dctPairClip(e6n.Add(t9), minv, maxv))
	dctPairStore(p0, p1, stride, 7, dctPairClip(e7n.Add(t8a), minv, maxv))
	dctPairStore(p0, p1, stride, 8, dctPairClip(e7n.Sub(t8a), minv, maxv))
	dctPairStore(p0, p1, stride, 9, dctPairClip(e6n.Sub(t9), minv, maxv))
	dctPairStore(p0, p1, stride, 10, dctPairClip(e5n.Sub(t10a), minv, maxv))
	dctPairStore(p0, p1, stride, 11, dctPairClip(e4n.Sub(t11), minv, maxv))
	dctPairStore(p0, p1, stride, 12, dctPairClip(e7.Sub(t12), minv, maxv))
	dctPairStore(p0, p1, stride, 13, dctPairClip(e6.Sub(t13a), minv, maxv))
	dctPairStore(p0, p1, stride, 14, dctPairClip(e5.Sub(t14), minv, maxv))
	dctPairStore(p0, p1, stride, 15, dctPairClip(e4.Sub(t15a), minv, maxv))
	archsimd.ClearAVXUpperBits()
}

func inverseDCT16Row2SIMDAdapter(r0, r1 []int32, min, max int32) {
	if len(r0) < dct16Size || len(r1) < dct16Size {
		inverseDCT16Row2PureGo(r0, r1, min, max)
		return
	}
	inverseDCT16PairSIMD(r0, r1, 1, min, max)
}

func inverseDCT16Col2SIMDAdapter(buf []int32, rowStride int, min, max int32) {
	if rowStride < 2 || len(buf) < (dct16Size-1)*rowStride+2 {
		inverseDCT16Col2PureGo(buf, rowStride, min, max)
		return
	}
	inverseDCT16PairSIMD(buf, buf[1:], rowStride, min, max)
}

func init() {
	if archsimd.X86.AVX2() {
		inverseDCT8Row2Impl = inverseDCT8Row2SIMDAdapter
		inverseDCT8Col2Impl = inverseDCT8Col2SIMDAdapter
		inverseDCT16Row2Impl = inverseDCT16Row2SIMDAdapter
		inverseDCT16Col2Impl = inverseDCT16Col2SIMDAdapter
	}
}

func init() {
	if archsimd.X86.AVX2() {
		inverseDCT32Col4Impl = inverseDCT32Lanes4ColSIMD
		inverseDCT32Row4Impl = inverseDCT32Lanes4RowSIMD
		inverseDCT64Col4Impl = inverseDCT64Col4SIMD
		inverseDCT64Row4Impl = inverseDCT64Row4SIMD
		inverseDCT64LowCol4Impl = inverseDCT64LowCol4SIMD
		inverseDCT64Row4LowImpl = inverseDCT64Row4LowSIMD
	}
}

// Pre-broadcast fdct4 twiddles, kept as package-level rodata so each loads with
// one instruction (the same tables the arm64 4x4 kernel uses).
var (
	fdct4W32   = [4]int32{fwdCospi13[32], fwdCospi13[32], fwdCospi13[32], fwdCospi13[32]}
	fdct4Wn32  = [4]int32{-fwdCospi13[32], -fwdCospi13[32], -fwdCospi13[32], -fwdCospi13[32]}
	fdct4W48   = [4]int32{fwdCospi13[48], fwdCospi13[48], fwdCospi13[48], fwdCospi13[48]}
	fdct4W16   = [4]int32{fwdCospi13[16], fwdCospi13[16], fwdCospi13[16], fwdCospi13[16]}
	fdct4Wn16  = [4]int32{-fwdCospi13[16], -fwdCospi13[16], -fwdCospi13[16], -fwdCospi13[16]}
	fdct4Round = [4]int32{1 << 12, 1 << 12, 1 << 12, 1 << 12}
)

// forwardDCT4x4SIMD is the 4x4 DCT_DCT kernel. The first-pass output magnitude is
// at most 46341 and the largest second-pass accumulator at most 2147446036 for
// |residual| <= 2048; larger public int16 inputs use the scalar path.
func forwardDCT4x4SIMD(coeff []int32, coeffStride int, residual []int16, residualStride int) {
	if !residualFitsMagnitude(residual, residualStride, 4, 4, 2048) {
		forwardDCT4x4PureGo(coeff, coeffStride, residual, residualStride)
		return
	}
	forwardDCT4x4SIMDCore(coeff, coeffStride, residual, residualStride)
}

// forwardDCT4x4SIMDCore runs one straight-line pass: widen four residual rows,
// the fdct4 column butterfly, an in-register 4x4 transpose, the fdct4 row
// butterfly, and the row stores. shift[1]=shift[2]=0 for 4x4.
func forwardDCT4x4SIMDCore(coeff []int32, coeffStride int, residual []int16, residualStride int) {
	if coeffStride == 4 && residualStride == 4 {
		forwardDCT4x4Packed((*[16]int32)(coeff[:16]), (*[16]int16)(residual[:16]))
		return
	}
	var r [16]int16
	var c [16]int32
	for i := range 4 {
		copy(r[4*i:4*i+4], residual[i*residualStride:i*residualStride+4])
	}
	forwardDCT4x4Packed(&c, &r)
	for i := range 4 {
		copy(coeff[i*coeffStride:i*coeffStride+4], c[4*i:4*i+4])
	}
}

// forwardDCT4x4Packed is forwardDCT4x4SIMDCore on a packed 4x4 block (both
// strides 4, the encoder's layout): two eight-sample loads cover the four
// residual rows, and the rows are stored at constant offsets.
func forwardDCT4x4Packed(coeff *[16]int32, residual *[16]int16) {
	// 128-bit only: the high row of each pair moves down with PUNPCKHQDQ.
	v01 := archsimd.LoadInt16x8Array((*[8]int16)(residual[0:8]))
	v23 := archsimd.LoadInt16x8Array((*[8]int16)(residual[8:16]))
	v11 := v01.AsInt64x2().InterleaveHi(v01.AsInt64x2()).AsInt16x8()
	v33 := v23.AsInt64x2().InterleaveHi(v23.AsInt64x2()).AsInt16x8()
	r0 := v01.ExtendLo4ToInt32().ShiftAllLeft(2)
	r1 := v11.ExtendLo4ToInt32().ShiftAllLeft(2)
	r2 := v23.ExtendLo4ToInt32().ShiftAllLeft(2)
	r3 := v33.ExtendLo4ToInt32().ShiftAllLeft(2)

	w32 := archsimd.LoadInt32x4Array(&fdct4W32)
	wn32 := archsimd.LoadInt32x4Array(&fdct4Wn32)
	w48 := archsimd.LoadInt32x4Array(&fdct4W48)
	w16 := archsimd.LoadInt32x4Array(&fdct4W16)
	wn16 := archsimd.LoadInt32x4Array(&fdct4Wn16)
	round := archsimd.LoadInt32x4Array(&fdct4Round)

	// Column pass: each half-butterfly is (a*w0 + b*w1 + round) >> 13.
	var c0, c1, c2, c3 archsimd.Int32x4
	{
		s0 := r0.Add(r3)
		s1 := r1.Add(r2)
		s2 := r1.Sub(r2)
		s3 := r0.Sub(r3)
		c0 = s1.Mul(w32).Add(s0.Mul(w32).Add(round)).ShiftAllRight(13)
		c2 = s0.Mul(w32).Add(s1.Mul(wn32).Add(round)).ShiftAllRight(13)
		c1 = s3.Mul(w16).Add(s2.Mul(w48).Add(round)).ShiftAllRight(13)
		c3 = s2.Mul(wn16).Add(s3.Mul(w48).Add(round)).ShiftAllRight(13)
	}

	// In-register 4x4 int32 transpose.
	e0 := c0.InterleaveLo(c1)
	e1 := c0.InterleaveHi(c1)
	e2 := c2.InterleaveLo(c3)
	e3 := c2.InterleaveHi(c3)
	t0 := fdctInt64AsInt32Amd(fdctInt32AsInt64Amd(e0).InterleaveLo(fdctInt32AsInt64Amd(e2)))
	t1 := fdctInt64AsInt32Amd(fdctInt32AsInt64Amd(e0).InterleaveHi(fdctInt32AsInt64Amd(e2)))
	t2 := fdctInt64AsInt32Amd(fdctInt32AsInt64Amd(e1).InterleaveLo(fdctInt32AsInt64Amd(e3)))
	t3 := fdctInt64AsInt32Amd(fdctInt32AsInt64Amd(e1).InterleaveHi(fdctInt32AsInt64Amd(e3)))

	// Row pass.
	var o0, o1, o2, o3 archsimd.Int32x4
	{
		s0 := t0.Add(t3)
		s1 := t1.Add(t2)
		s2 := t1.Sub(t2)
		s3 := t0.Sub(t3)
		o0 = s1.Mul(w32).Add(s0.Mul(w32).Add(round)).ShiftAllRight(13)
		o2 = s0.Mul(w32).Add(s1.Mul(wn32).Add(round)).ShiftAllRight(13)
		o1 = s3.Mul(w16).Add(s2.Mul(w48).Add(round)).ShiftAllRight(13)
		o3 = s2.Mul(wn16).Add(s3.Mul(w48).Add(round)).ShiftAllRight(13)
	}

	o0.StoreArray((*[4]int32)(coeff[0:4]))
	o1.StoreArray((*[4]int32)(coeff[4:8]))
	o2.StoreArray((*[4]int32)(coeff[8:12]))
	o3.StoreArray((*[4]int32)(coeff[12:16]))
}

func fdctInt32AsInt64Amd(v archsimd.Int32x4) archsimd.Int64x2 {
	return v.AsInt64x2()
}

func fdctInt64AsInt32Amd(v archsimd.Int64x2) archsimd.Int32x4 {
	return v.AsInt32x4()
}

// The amd64 cores are the eight-lane scratch-transpose network.
func fwd8DCTCore(coeff *[64]int32, residual *[64]int16) {
	fwd8x8NetPacked(coeff, residual, false, false)
}

func fwd8ADSTDCTCore(coeff *[64]int32, residual *[64]int16) {
	fwd8x8NetPacked(coeff, residual, true, false)
}

// The 4x4, 8x8, 16x16 and 32x32 forward DCTs are Go SIMD on AVX2 hosts. Without
// AVX2 the portable bindings from fdct_neon_off.go stay in place.
func init() {
	if archsimd.X86.AVX2() {
		forwardDCT4x4Impl = forwardDCT4x4SIMDVZ
		forwardDCT4x4Trusted8BitImpl = forwardDCT4x4SIMDCoreVZ
		forwardDCT8x8Impl = forwardDCT8x8SIMDGuardedVZ
		forwardDCT8x8Trusted8BitImpl = forwardDCT8x8SIMDVZ
		forwardDCT16x16Impl = forwardDCT16x16SIMDGuardedVZ
		forwardDCT16x16Trusted8BitImpl = forwardDCT16x16SIMDVZ
		forwardDCT32x32Impl = forwardDCT32x32SIMDGuardedVZ
		forwardDCT32x32Trusted8BitImpl = forwardDCT32x32SIMDVZ
		forwardDCT64x64Trusted8BitImpl = forwardDCT64x64SIMDVZ
		forwardDCT8x8N4Impl = forwardDCT8x8N4SIMDVZ
	}
}

// The VZ wrappers return with clean upper ymm state: the amd64 kernels run
// 256-bit lanes and the Go compiler emits no vzeroupper, so the legacy-SSE
// code that follows them would stall on Zen 3.

func forwardDCT4x4SIMDVZ(coeff []int32, coeffStride int, residual []int16, residualStride int) {
	forwardDCT4x4SIMD(coeff, coeffStride, residual, residualStride)
	archsimd.ClearAVXUpperBits()
}

func forwardDCT4x4SIMDCoreVZ(coeff []int32, coeffStride int, residual []int16, residualStride int) {
	forwardDCT4x4SIMDCore(coeff, coeffStride, residual, residualStride)
	archsimd.ClearAVXUpperBits()
}

func forwardDCT8x8SIMDGuardedVZ(coeff []int32, coeffStride int, residual []int16, residualStride int) {
	forwardDCT8x8SIMDGuarded(coeff, coeffStride, residual, residualStride)
	archsimd.ClearAVXUpperBits()
}

func forwardDCT8x8SIMDVZ(coeff []int32, coeffStride int, residual []int16, residualStride int) {
	forwardDCT8x8SIMD(coeff, coeffStride, residual, residualStride)
	archsimd.ClearAVXUpperBits()
}

func forwardDCT16x16SIMDGuardedVZ(coeff []int32, coeffStride int, residual []int16, residualStride int) {
	forwardDCT16x16SIMDGuarded(coeff, coeffStride, residual, residualStride)
	archsimd.ClearAVXUpperBits()
}

func forwardDCT16x16SIMDVZ(coeff []int32, coeffStride int, residual []int16, residualStride int) {
	forwardDCT16x16SIMD(coeff, coeffStride, residual, residualStride)
	archsimd.ClearAVXUpperBits()
}

func forwardDCT32x32SIMDGuardedVZ(coeff []int32, coeffStride int, residual []int16, residualStride int) {
	forwardDCT32x32SIMDGuarded(coeff, coeffStride, residual, residualStride)
	archsimd.ClearAVXUpperBits()
}

func forwardDCT32x32SIMDVZ(coeff []int32, coeffStride int, residual []int16, residualStride int) {
	forwardDCT32x32SIMD(coeff, coeffStride, residual, residualStride)
	archsimd.ClearAVXUpperBits()
}

func forwardDCT64x64SIMDVZ(coeff []int32, coeffStride int, residual []int16, residualStride int) {
	forwardDCT64x64SIMD(coeff, coeffStride, residual, residualStride)
	archsimd.ClearAVXUpperBits()
}

func forwardDCT8x8N4SIMDVZ(coeff *[64]int32, residual *[64]int16) {
	forwardDCT8x8N4SIMD(coeff, residual)
	archsimd.ClearAVXUpperBits()
}

// The amd64 drivers run the column pass (fdct_net16_32_gen_amd64.go) into a
// row-major buffer, transpose it and run the row pass.

// forwardDCT16x16SIMD is the 16x16 DCT_DCT kernel (forwardDCT16x16PureGo).
func forwardDCT16x16SIMD(coeff []int32, coeffStride int, residual []int16, residualStride int) {
	var buf, bufT [256]int32
	for g := 0; g < 16; g += fwdLanes {
		fwdColDCT16(buf[:], g, residual, residualStride)
	}
	fwdTranspose(bufT[:], buf[:], 16)
	for h := 0; h < 16; h += fwdLanes {
		fwdRowDCT16(coeff, coeffStride, bufT[:], h)
	}
}

// forwardDCT32x32SIMD is the 32x32 DCT_DCT kernel (forwardDCT32x32PureGo).
func forwardDCT32x32SIMD(coeff []int32, coeffStride int, residual []int16, residualStride int) {
	var buf, bufT [1024]int32
	for g := 0; g < 32; g += fwdLanes {
		fwdColDCT32(buf[:], g, residual, residualStride)
	}
	fwdTranspose(bufT[:], buf[:], 32)
	for h := 0; h < 32; h += fwdLanes {
		fwdRowDCT32(coeff, coeffStride, bufT[:], h)
	}
}

// fwdVec is one 256-bit lane group of the forward-transform networks: eight
// independent 1-D lines per vector (fdct_gosimd_net.go). Only AVX2 operations
// are used; the dispatcher checks archsimd.X86.AVX2 before binding them.
type fwdVec = archsimd.Int32x8

// fwdLanes is the number of 1-D lines a fwdVec carries.
const fwdLanes = 8

func fwdBcast(v int32) fwdVec {
	return archsimd.BroadcastInt32x8(v)
}

// fwdShr is an arithmetic right shift by the constant n in every lane (the
// AVX2 immediate form).
func fwdShr(v fwdVec, n uint64) fwdVec {
	return v.ShiftAllRight(n)
}

// fwdShl is a left shift by the constant n in every lane.
func fwdShl(v fwdVec, n uint64) fwdVec {
	return v.ShiftAllLeft(n)
}

func fwdStoreI32(s []int32, v fwdVec) {
	v.Store(s)
}

// fwdTranspose writes the transpose of the n x n row-major matrix src into dst
// (dst[c*n+r] = src[r*n+c]); n is a multiple of eight.
func fwdTranspose(dst, src []int32, n int) {
	for bi := 0; bi < n; bi += 8 {
		for bj := 0; bj < n; bj += 8 {
			fwdTransposeBlock8(dst[bj*n+bi:], n, src[bi*n+bj:], n)
		}
	}
}

// fwdTransposeBlock8 transposes the 8x8 block at src (row stride sn) into dst
// (row stride dn). Three AVX2 unpack levels (32-bit, 64-bit, then 128-bit
// lane permutes) put each output column in one register.
func fwdTransposeBlock8(dst []int32, dn int, src []int32, sn int) {
	r0 := archsimd.LoadInt32x8(src[0:])
	r1 := archsimd.LoadInt32x8(src[sn:])
	r2 := archsimd.LoadInt32x8(src[2*sn:])
	r3 := archsimd.LoadInt32x8(src[3*sn:])
	r4 := archsimd.LoadInt32x8(src[4*sn:])
	r5 := archsimd.LoadInt32x8(src[5*sn:])
	r6 := archsimd.LoadInt32x8(src[6*sn:])
	r7 := archsimd.LoadInt32x8(src[7*sn:])

	a0 := r0.InterleaveLoGrouped(r1)
	a1 := r0.InterleaveHiGrouped(r1)
	a2 := r2.InterleaveLoGrouped(r3)
	a3 := r2.InterleaveHiGrouped(r3)
	a4 := r4.InterleaveLoGrouped(r5)
	a5 := r4.InterleaveHiGrouped(r5)
	a6 := r6.InterleaveLoGrouped(r7)
	a7 := r6.InterleaveHiGrouped(r7)

	b0 := fwdI64AsI32x8(fwdI32x8AsI64(a0).InterleaveLoGrouped(fwdI32x8AsI64(a2)))
	b1 := fwdI64AsI32x8(fwdI32x8AsI64(a0).InterleaveHiGrouped(fwdI32x8AsI64(a2)))
	b2 := fwdI64AsI32x8(fwdI32x8AsI64(a1).InterleaveLoGrouped(fwdI32x8AsI64(a3)))
	b3 := fwdI64AsI32x8(fwdI32x8AsI64(a1).InterleaveHiGrouped(fwdI32x8AsI64(a3)))
	b4 := fwdI64AsI32x8(fwdI32x8AsI64(a4).InterleaveLoGrouped(fwdI32x8AsI64(a6)))
	b5 := fwdI64AsI32x8(fwdI32x8AsI64(a4).InterleaveHiGrouped(fwdI32x8AsI64(a6)))
	b6 := fwdI64AsI32x8(fwdI32x8AsI64(a5).InterleaveLoGrouped(fwdI32x8AsI64(a7)))
	b7 := fwdI64AsI32x8(fwdI32x8AsI64(a5).InterleaveHiGrouped(fwdI32x8AsI64(a7)))

	// Output column k is the low 128-bit half of b_k joined with the low half
	// of b_(k+4) (column k, rows 0-7), and likewise for the high halves.
	fwdStoreI32(dst[0*dn:], b0.ConcatPermute128Scalars(0, 2, b4))
	fwdStoreI32(dst[1*dn:], b1.ConcatPermute128Scalars(0, 2, b5))
	fwdStoreI32(dst[2*dn:], b2.ConcatPermute128Scalars(0, 2, b6))
	fwdStoreI32(dst[3*dn:], b3.ConcatPermute128Scalars(0, 2, b7))
	fwdStoreI32(dst[4*dn:], b0.ConcatPermute128Scalars(1, 3, b4))
	fwdStoreI32(dst[5*dn:], b1.ConcatPermute128Scalars(1, 3, b5))
	fwdStoreI32(dst[6*dn:], b2.ConcatPermute128Scalars(1, 3, b6))
	fwdStoreI32(dst[7*dn:], b3.ConcatPermute128Scalars(1, 3, b7))
}

func fwdI32x8AsI64(v fwdVec) archsimd.Int64x4 {
	return v.AsInt64x4()
}

func fwdI64AsI32x8(v archsimd.Int64x4) fwdVec {
	return v.AsInt32x8()
}

// The network drivers have already validated their buffers; each access is
// one checked fixed-length window.
func fwdLoadI32At(s []int32, i int) fwdVec {
	return archsimd.LoadInt32x8Array((*[8]int32)(s[i : i+8]))
}

func fwdStoreI32At(s []int32, i int, v fwdVec) {
	v.StoreArray((*[8]int32)(s[i : i+8]))
}

func fwdLoadResAt(s []int16, i, g int) fwdVec {
	w := archsimd.LoadInt16x8Array((*[8]int16)(s[i : i+8]))
	return fwdShl(w.ExtendToInt32(), 2)
}

// fwdMulAdd computes a*w + acc. The arm64 intrinsic maps to MLA; AVX2
// has no integer multiply-add instruction, so its port uses two operations.
func fwdMulAdd(a, w, acc fwdVec) fwdVec {
	return a.Mul(w).Add(acc)
}

// fwdHalfBtf13V computes the rounded Q13 forward butterfly.
func fwdHalfBtf13V(w0, a, w1, b fwdVec) fwdVec {
	return fwdShr(a.Mul(w0).Add(b.Mul(w1)).Add(fwdRound13), 13)
}

// fwdHalfBtf12V computes the rounded Q12 forward butterfly.
func fwdHalfBtf12V(w0, a, w1, b fwdVec) fwdVec {
	return fwdShr(a.Mul(w0).Add(b.Mul(w1)).Add(fwdRound12), 12)
}

// fwdLoadResRawAt is fwdLoadResAt without the shift[0] = 2 pre-scale (the
// 64-point transforms take shift[0] = 0).
func fwdLoadResRawAt(s []int16, i, g int) fwdVec {
	return archsimd.LoadInt16x8Array((*[8]int16)(s[i : i+8])).ExtendToInt32()
}

// fwdTransposeRect writes the transpose of the rows x cols row-major matrix
// src into dst (dst[c*rows+r] = src[r*cols+c]); both are multiples of eight.
func fwdTransposeRect(dst, src []int32, rows, cols int) {
	for bi := 0; bi < rows; bi += 8 {
		for bj := 0; bj < cols; bj += 8 {
			fwdTransposeBlock8(dst[bj*rows+bi:], rows, src[bi*cols+bj:], cols)
		}
	}
}

// fwdShift is a runtime arithmetic right-shift amount for fwdShrBy, held as a
// per-lane count vector (VPSRAVD). A scalar count would be moved into an XMM
// register with a legacy-SSE movq at every use, which stalls on Zen 3 while
// the upper ymm state is dirty.
type fwdShift = archsimd.Uint32x8

func fwdMakeShift(n uint64) fwdShift { return archsimd.BroadcastUint32x8(uint32(n)) }

func fwdShrBy(v fwdVec, s fwdShift) fwdVec { return v.ShiftRight(s) }

// fwdVecDone returns the upper ymm state to clean after a run of fwdVec code
// (the compiler emits no vzeroupper), so the scalar and legacy-SSE code that
// follows does not stall on Zen 3.
func fwdVecDone() { archsimd.ClearAVXUpperBits() }

// init binds the AVX2 Go SIMD inverse-transform glue kernels when the CPU
// advertises AVX2. Every operation used here is AVX2 (VPSRAD, VPADDD, VPMINSD,
// VPMAXSD, VPACKSSDW); none lowers to AVX-512.
func init() {
	if invGlueSIMDAvailable() {
		clampRoundImpl = clampRoundSIMD
		narrowStoreImpl = narrowStoreSIMD
	}
}

// invGlueSIMDAvailable reports whether the AVX2 glue kernels may be bound.
func invGlueSIMDAvailable() bool { return archsimd.X86.AVX2() }

// clampRoundSIMD rounds-and-shifts scratch in place (shift > 0) and clamps it to
// [min, max], eight int32 lanes at a time. The rounding uses
// (x>>s) + ((x>>(s-1))&1), which equals roundShift(x, s) for every int32 x and
// cannot overflow. The staged transforms shift by 0, 1 or 2, each with
// immediate shift counts (a variable ymm shift count is rematerialized with a
// legacy-SSE movq, which stalls on dirty upper state on Zen 3), and the upper
// state is cleared on return. Tails and other shifts use the pure-Go
// reference.
func clampRoundSIMD(scratch []int32, shift int, min, max int32) {
	if shift < 0 || shift > 2 {
		clampRoundPureGo(scratch, shift, min, max)
		return
	}
	minV := archsimd.BroadcastInt32x8(min)
	maxV := archsimd.BroadcastInt32x8(max)
	one := archsimd.BroadcastInt32x8(1)
	i := 0
	switch shift {
	case 0:
		for ; i+8 <= len(scratch); i += 8 {
			archsimd.LoadInt32x8Array((*[8]int32)(scratch[i:])).
				Max(minV).Min(maxV).
				StoreArray((*[8]int32)(scratch[i:]))
		}
	case 1:
		for ; i+8 <= len(scratch); i += 8 {
			v := archsimd.LoadInt32x8Array((*[8]int32)(scratch[i:]))
			r := v.ShiftAllRight(1).Add(v.And(one))
			r.Max(minV).Min(maxV).StoreArray((*[8]int32)(scratch[i:]))
		}
	case 2:
		for ; i+8 <= len(scratch); i += 8 {
			v := archsimd.LoadInt32x8Array((*[8]int32)(scratch[i:]))
			r := v.ShiftAllRight(2).Add(v.ShiftAllRight(1).And(one))
			r.Max(minV).Min(maxV).StoreArray((*[8]int32)(scratch[i:]))
		}
	}
	archsimd.ClearAVXUpperBits()
	if i < len(scratch) {
		clampRoundPureGo(scratch[i:], shift, min, max)
	}
}

// roundShiftNarrowInt32x4PairToInt16x8 rounding-shifts two int32x4 halves right
// by shift and saturates them to one int16x8 in lane order (lo lanes 0-3, hi
// lanes 4-7). SaturateToInt16Concat is VPACKSSDW (AVX, not AVX-512).
func roundShiftNarrowInt32x4PairToInt16x8(lo, hi archsimd.Int32x4, shift uint64) archsimd.Int16x8 {
	one := archsimd.BroadcastInt32x4(1)
	lo16 := lo.ShiftAllRight(shift).Add(lo.ShiftAllRight(shift - 1).And(one))
	hi16 := hi.ShiftAllRight(shift).Add(hi.ShiftAllRight(shift - 1).And(one))
	return lo16.SaturateToInt16Concat(hi16)
}

// narrowStoreSIMD writes the final residual rows: every int32 lane is
// rounding-shifted right by four and saturated to int16. Width is four or a
// multiple of eight; other widths use the pure-Go reference.
func narrowStoreSIMD(dst []int16, dstStride int, scratch []int32, width, height int) {
	if width != 4 && width%8 != 0 {
		narrowStorePureGo(dst, dstStride, scratch, width, height)
		return
	}
	for row := range height {
		dstLine := dst[row*dstStride : row*dstStride+width : row*dstStride+width]
		srcLine := scratch[row*width : row*width+width : row*width+width]
		if width == 4 {
			v := archsimd.LoadInt32x4Array((*[4]int32)(srcLine))
			var out [8]int16
			roundShiftNarrowInt32x4PairToInt16x8(v, v, 4).StoreArray(&out)
			copy(dstLine, out[:4])
			continue
		}
		for col := 0; col < width; col += 8 {
			lo := archsimd.LoadInt32x4Array((*[4]int32)(srcLine[col:]))
			hi := archsimd.LoadInt32x4Array((*[4]int32)(srcLine[col+4:]))
			roundShiftNarrowInt32x4PairToInt16x8(lo, hi, 4).StoreArray((*[8]int16)(dstLine[col:]))
		}
	}
}

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
		x := itx48Rect2x4(coeff[k*coeffStride : k*coeffStride+4]).SaturateToInt16Concat(zero).And(m)
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

// The fused 8-bit inverse transform + add kernels for amd64 (AVX2), in
// dav1d's itx_avx2 form: the int16 butterfly networks of the arm64 fused
// kernels (itx_avx2_nets*_amd64.go, copied from itx_fused*_gosimd_arm64.go)
// run sixteen lanes per ymm register, and every rotation is dav1d's
// ITX_MULSUB_2W: punpck{l,h}wd of the two inputs, vpmaddwd against the
// interleaved constant pair, the rounding bias, vpsrad and vpackssdw. The
// int32 sum is exact and the pack saturates, so under the shared per-lane
// L1 guard (itx16ColumnL1Bound) every rotation equals the scalar
// Round2(a*ca + b*cb, n), as the arm64 kernels' wrapping narrow does.
//
// Blocks wider than eight run sixteen rows (row pass) or sixteen columns
// (column pass) per strip, with one 16x16 transpose between the passes; the
// 8x8 blocks run eight lanes in xmm registers end to end.

// itxAVX2 gates every fused kernel; without AVX2 they all decline.
var itxAVX2 = archsimd.X86.AVX2()

// itx16ColumnL1Bound is the per-lane guard of the int16 networks: a lane
// whose inputs have L1 norm at most the bound has every rotation output in
// int16 (itx16_gain_test.go and its DCT32/DCT64/ADST companions).
const itx16ColumnL1Bound = 32767 - 64

// itxBroadcast16 broadcasts each constant to sixteen int16 lanes.
func itxBroadcast16(v []int16) [][16]int16 {
	out := make([][16]int16, len(v))
	for i, c := range v {
		for l := range out[i] {
			out[i][l] = c
		}
	}
	return out
}

// itx64K[i] broadcasts cospi(i) and itx64K[64+i] broadcasts -cospi(i).
var itx64K = func() (k [128][16]int16) {
	for i, c := range itx64Cos {
		for l := 0; l < 16; l++ {
			k[i][l] = c
			k[64+i][l] = -c
		}
	}
	return k
}()

// itx16Rot is Round2(a*ca + b*cb, sh) on sixteen lanes (ca, cb broadcast).
func itx16Rot(a, b, ca, cb archsimd.Int16x16, sh uint8) archsimd.Int16x16 {
	k := ca.InterleaveLoGrouped(cb)
	bias := archsimd.BroadcastInt32x8(1 << (sh - 1))
	lo := a.InterleaveLoGrouped(b).DotProductPairs(k).Add(bias).ShiftAllRight(uint64(sh))
	hi := a.InterleaveHiGrouped(b).DotProductPairs(k).Add(bias).ShiftAllRight(uint64(sh))
	return lo.SaturateToInt16ConcatGrouped(hi)
}

// itx16Rot1 is Round2(a*ca, sh) on sixteen lanes; the rounding bias rides
// in the second product of the vpmaddwd pair.
func itx16Rot1(a, ca archsimd.Int16x16, sh uint8) archsimd.Int16x16 {
	k := ca.InterleaveLoGrouped(archsimd.BroadcastInt16x16(1 << (sh - 1)))
	one := archsimd.BroadcastInt16x16(1)
	lo := a.InterleaveLoGrouped(one).DotProductPairs(k).ShiftAllRight(uint64(sh))
	hi := a.InterleaveHiGrouped(one).DotProductPairs(k).ShiftAllRight(uint64(sh))
	return lo.SaturateToInt16ConcatGrouped(hi)
}

// itx8Rot is itx16Rot on eight lanes.
func itx8Rot(a, b, ca, cb archsimd.Int16x8, sh uint8) archsimd.Int16x8 {
	k := ca.InterleaveLo(cb)
	bias := archsimd.BroadcastInt32x4(1 << (sh - 1))
	lo := a.InterleaveLo(b).DotProductPairs(k).Add(bias).ShiftAllRight(uint64(sh))
	hi := a.InterleaveHi(b).DotProductPairs(k).Add(bias).ShiftAllRight(uint64(sh))
	return lo.SaturateToInt16Concat(hi)
}

// itx8Rot1 is itx16Rot1 on eight lanes.
func itx8Rot1(a, ca archsimd.Int16x8, sh uint8) archsimd.Int16x8 {
	k := ca.InterleaveLo(archsimd.BroadcastInt16x8(1 << (sh - 1)))
	one := archsimd.BroadcastInt16x8(1)
	lo := a.InterleaveLo(one).DotProductPairs(k).ShiftAllRight(uint64(sh))
	hi := a.InterleaveHi(one).DotProductPairs(k).ShiftAllRight(uint64(sh))
	return lo.SaturateToInt16Concat(hi)
}

// itxLoad8 loads the first eight lanes of a sixteen-lane constant.
func itxLoad8(p *[16]int16) archsimd.Int16x8 {
	return archsimd.LoadInt16x8Array((*[8]int16)(p[:8]))
}

// itxRowPerm is the row of each lane after itxLoadNarrow16: vpackssdw packs
// within 128-bit lanes, so lanes 4..7 hold rows 8..11 and lanes 8..11 rows
// 4..7. The row pass is lane-wise, so the order is undone only where the
// transposed rows are stored.
var itxRowPerm = [16]int{0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 12, 13, 14, 15}

// itxLoadNarrow16 loads sixteen int32 values and narrows them to int16 with
// saturation, in itxRowPerm lane order.
func itxLoadNarrow16(p []int32) archsimd.Int16x16 {
	lo := archsimd.LoadInt32x8(p[:8])
	hi := archsimd.LoadInt32x8(p[8:16])
	return lo.SaturateToInt16ConcatGrouped(hi)
}

// itxRect2Pre is the pre-clamp that keeps c*181+128 inside int32 without
// changing the result (see stageTransposeClamp).
const itxRect2Pre = 1 << 22

// itxLoadNarrowRect2x16 is itxLoadNarrow16 with the rect2 scale applied
// first: lane i is clip16(Round2(p[i]*2896, 12)) = clip16((p*181+128)>>8).
func itxLoadNarrowRect2x16(p []int32) archsimd.Int16x16 {
	lo := archsimd.LoadInt32x8(p[:8])
	hi := archsimd.LoadInt32x8(p[8:16])
	preLo, preHi := archsimd.BroadcastInt32x8(-itxRect2Pre), archsimd.BroadcastInt32x8(itxRect2Pre)
	k181, r128 := archsimd.BroadcastInt32x8(181), archsimd.BroadcastInt32x8(128)
	lo = lo.Max(preLo).Min(preHi).Mul(k181).Add(r128).ShiftAllRight(8)
	hi = hi.Max(preLo).Min(preHi).Mul(k181).Add(r128).ShiftAllRight(8)
	return lo.SaturateToInt16ConcatGrouped(hi)
}

// itxLoadNarrow8 loads eight int32 values narrowed to int16 with saturation.
func itxLoadNarrow8(p []int32) archsimd.Int16x8 {
	return archsimd.LoadInt32x4(p[:4]).SaturateToInt16Concat(archsimd.LoadInt32x4(p[4:8]))
}

// itxL1Ok16 reports whether every lane of l1 is within the guard.
func itxL1Ok16(l1 archsimd.Uint16x16) bool {
	return l1.SubSaturated(archsimd.BroadcastUint16x16(itx16ColumnL1Bound)).IsZero()
}

// itxL1Ok8 is itxL1Ok16 on eight lanes.
func itxL1Ok8(l1 archsimd.Uint16x8) bool {
	return l1.SubSaturated(archsimd.BroadcastUint16x8(itx16ColumnL1Bound)).IsZero()
}

// itxRound2x16 is Round2(x, n) for n in {1, 2, 4} on int16 lanes, formed
// without overflow: (x >> 1) + (x & 1) for n = 1, ((x >> (n-1)) + 1) >> 1
// otherwise.
func itxRound2x16(x archsimd.Int16x16, n uint8) archsimd.Int16x16 {
	one := archsimd.BroadcastInt16x16(1)
	if n == 1 {
		return x.ShiftAllRight(1).Add(x.And(one))
	}
	return x.ShiftAllRight(uint64(n - 1)).Add(one).ShiftAllRight(1)
}

// itxRound2x8 is itxRound2x16 on eight lanes.
func itxRound2x8(x archsimd.Int16x8, n uint8) archsimd.Int16x8 {
	one := archsimd.BroadcastInt16x8(1)
	if n == 1 {
		return x.ShiftAllRight(1).Add(x.And(one))
	}
	return x.ShiftAllRight(uint64(n - 1)).Add(one).ShiftAllRight(1)
}

// itxTranspose8x8Lanes transposes the 8x8 int16 matrices held in each
// 128-bit lane of v[0..7] (v[r] = row r): three interleave stages at 16, 32
// and 64 bits, all within lanes.
func itxTranspose8x8Lanes(v []archsimd.Int16x16) {
	_ = v[7]
	a0, a1 := v[0].InterleaveLoGrouped(v[1]).AsInt32x8(), v[0].InterleaveHiGrouped(v[1]).AsInt32x8()
	a2, a3 := v[2].InterleaveLoGrouped(v[3]).AsInt32x8(), v[2].InterleaveHiGrouped(v[3]).AsInt32x8()
	a4, a5 := v[4].InterleaveLoGrouped(v[5]).AsInt32x8(), v[4].InterleaveHiGrouped(v[5]).AsInt32x8()
	a6, a7 := v[6].InterleaveLoGrouped(v[7]).AsInt32x8(), v[6].InterleaveHiGrouped(v[7]).AsInt32x8()
	b0, b1 := a0.InterleaveLoGrouped(a2).AsInt64x4(), a0.InterleaveHiGrouped(a2).AsInt64x4()
	b2, b3 := a1.InterleaveLoGrouped(a3).AsInt64x4(), a1.InterleaveHiGrouped(a3).AsInt64x4()
	b4, b5 := a4.InterleaveLoGrouped(a6).AsInt64x4(), a4.InterleaveHiGrouped(a6).AsInt64x4()
	b6, b7 := a5.InterleaveLoGrouped(a7).AsInt64x4(), a5.InterleaveHiGrouped(a7).AsInt64x4()
	v[0] = b0.InterleaveLoGrouped(b4).AsInt16x16()
	v[1] = b0.InterleaveHiGrouped(b4).AsInt16x16()
	v[2] = b1.InterleaveLoGrouped(b5).AsInt16x16()
	v[3] = b1.InterleaveHiGrouped(b5).AsInt16x16()
	v[4] = b2.InterleaveLoGrouped(b6).AsInt16x16()
	v[5] = b2.InterleaveHiGrouped(b6).AsInt16x16()
	v[6] = b3.InterleaveLoGrouped(b7).AsInt16x16()
	v[7] = b3.InterleaveHiGrouped(b7).AsInt16x16()
}

// itxTranspose16x16 transposes the 16x16 int16 matrix v (v[i] = row i): the
// two 8x8 quadrant pairs transpose within lanes, then vperm2i128 swaps the
// off-diagonal quadrants.
func itxTranspose16x16(v *[16]archsimd.Int16x16) {
	itxTranspose8x8Lanes(v[:8])
	itxTranspose8x8Lanes(v[8:])
	for i := 0; i < 8; i++ {
		w, u := v[i], v[8+i]
		v[i] = w.ConcatPermute128Scalars(0, 2, u)
		v[8+i] = w.ConcatPermute128Scalars(1, 3, u)
	}
}

// itxTranspose8x8 transposes the 8x8 int16 matrix v (v[r] = row r).
func itxTranspose8x8(v *[8]archsimd.Int16x8) {
	a0, a1 := v[0].InterleaveLo(v[1]).AsInt32x4(), v[0].InterleaveHi(v[1]).AsInt32x4()
	a2, a3 := v[2].InterleaveLo(v[3]).AsInt32x4(), v[2].InterleaveHi(v[3]).AsInt32x4()
	a4, a5 := v[4].InterleaveLo(v[5]).AsInt32x4(), v[4].InterleaveHi(v[5]).AsInt32x4()
	a6, a7 := v[6].InterleaveLo(v[7]).AsInt32x4(), v[6].InterleaveHi(v[7]).AsInt32x4()
	b0, b1 := a0.InterleaveLo(a2).AsInt64x2(), a0.InterleaveHi(a2).AsInt64x2()
	b2, b3 := a1.InterleaveLo(a3).AsInt64x2(), a1.InterleaveHi(a3).AsInt64x2()
	b4, b5 := a4.InterleaveLo(a6).AsInt64x2(), a4.InterleaveHi(a6).AsInt64x2()
	b6, b7 := a5.InterleaveLo(a7).AsInt64x2(), a5.InterleaveHi(a7).AsInt64x2()
	v[0] = b0.InterleaveLo(b4).AsInt16x8()
	v[1] = b0.InterleaveHi(b4).AsInt16x8()
	v[2] = b1.InterleaveLo(b5).AsInt16x8()
	v[3] = b1.InterleaveHi(b5).AsInt16x8()
	v[4] = b2.InterleaveLo(b6).AsInt16x8()
	v[5] = b2.InterleaveHi(b6).AsInt16x8()
	v[6] = b3.InterleaveLo(b7).AsInt16x8()
	v[7] = b3.InterleaveHi(b7).AsInt16x8()
}

// itxPackLoBytes gathers the low byte of every int16 lane into the low
// eight bytes of its 128-bit lane (vpshufb; index -1 zeroes).
var itxPackLoBytes = [32]int8{
	0, 2, 4, 6, 8, 10, 12, 14, -1, -1, -1, -1, -1, -1, -1, -1,
	0, 2, 4, 6, 8, 10, 12, 14, -1, -1, -1, -1, -1, -1, -1, -1,
}

// itxAddStore16 stores clip(pred[0..15] + res) to dst[0..15]. vpackuswb is
// not in archsimd, so the clamp is explicit and vpshufb plus a 64-bit
// interleave pack the bytes.
func itxAddStore16(dst, pred []byte, res archsimd.Int16x16) {
	p := archsimd.LoadUint8x16(pred[:16]).ExtendToUint16().AsInt16x16()
	v := p.Add(res).Max(archsimd.Int16x16{}).Min(archsimd.BroadcastInt16x16(255))
	b := v.AsUint8x32().PermuteOrZeroGrouped(archsimd.LoadInt8x32Array(&itxPackLoBytes))
	b.GetLo().AsUint64x2().InterleaveLo(b.GetHi().AsUint64x2()).AsUint8x16().Store(dst[:16])
}

// itxAddStore8 stores clip(pred[0..7] + res) to dst[0..7].
func itxAddStore8(dst, pred []byte, res archsimd.Int16x8) {
	var pv archsimd.Uint64x2
	pv = pv.SetElem(0, binary.LittleEndian.Uint64(pred[:8]))
	p := pv.AsUint8x16().ExtendLo8ToUint16().AsInt16x8()
	v := p.Add(res).Max(archsimd.Int16x8{}).Min(archsimd.BroadcastInt16x8(255))
	b := v.AsUint8x16().PermuteOrZero(archsimd.LoadInt8x16Array((*[16]int8)(itxPackLoBytes[:16])))
	binary.LittleEndian.PutUint64(dst[:8], b.AsUint64x2().GetElem(0))
}

// itxPix is the reconstruction target of a fused kernel: dst = clip(pred +
// residual) with 1-byte samples when maxV is 0, else 2-byte samples clipped
// to [0, maxV] (strides in bytes).
type itxPix struct {
	dst        []byte
	dstStride  int
	pred       []byte
	predStride int
	maxV       int16
}

// itxAddStore16HBD stores clip(pred[0..15] + res, 0, maxV) for sixteen
// 2-byte samples. The saturating add equals the scalar clip for any int16
// residual and pred <= 4095.
func itxAddStore16HBD(dst, pred []byte, res archsimd.Int16x16, maxV int16) {
	p := archsimd.LoadUint8x32Array((*[32]byte)(pred)).AsInt16x16()
	v := p.AddSaturated(res).Max(archsimd.Int16x16{}).Min(archsimd.BroadcastInt16x16(maxV))
	v.AsUint8x32().StoreArray((*[32]byte)(dst))
}

// itxAddStore8HBD is itxAddStore16HBD for eight samples.
func itxAddStore8HBD(dst, pred []byte, res archsimd.Int16x8, maxV int16) {
	p := archsimd.LoadUint8x16Array((*[16]byte)(pred)).AsInt16x8()
	v := p.AddSaturated(res).Max(archsimd.Int16x8{}).Min(archsimd.BroadcastInt16x8(maxV))
	v.AsUint8x16().StoreArray((*[16]byte)(dst))
}

// itxStoreRowGroup rounds the sixteen row-pass outputs v (v[j] = output j
// of every lane, lanes in itxRowPerm order of rows r0..r0+15) by n,
// transposes them and stores row r0+itxRowPerm[e] of tmp (width w) at
// columns c0..c0+15, adding the stored magnitudes to the column guard.
func itxStoreRowGroup(v *[16]archsimd.Int16x16, n uint8, tmp []int16, w, r0, c0 int, colL1 *archsimd.Uint16x16) {
	// Immediate shift counts only: a variable ymm shift count is
	// rematerialized with a legacy-SSE movq, which stalls on dirty upper
	// state (Zen 3).
	if n == 1 {
		for j := range v {
			v[j] = itxRound2x16(v[j], 1)
		}
	} else {
		for j := range v {
			v[j] = itxRound2x16(v[j], 2)
		}
	}
	itxTranspose16x16(v)
	l1 := *colL1
	for e := 0; e < 16; e++ {
		v[e].Store(tmp[(r0+itxRowPerm[e])*w+c0:])
		l1 = l1.AddSaturated(v[e].Abs().AsUint16x16())
	}
	*colL1 = l1
	archsimd.ClearAVXUpperBits()
}

// itxRowLoad loads inCols coefficient columns of rows r0..r0+15 (16 lanes)
// into in, with the rect2 scale when rect2 is set, and reports whether
// every lane passes the L1 guard.
func itxRowLoad(coeff []int32, stride, r0, inCols int, rect2 bool, in []archsimd.Int16x16) bool {
	_ = coeff[(inCols-1)*stride+r0+15]
	var l1 archsimd.Uint16x16
	for k := 0; k < inCols; k++ {
		var x archsimd.Int16x16
		if rect2 {
			x = itxLoadNarrowRect2x16(coeff[k*stride+r0 : k*stride+r0+16])
		} else {
			x = itxLoadNarrow16(coeff[k*stride+r0 : k*stride+r0+16])
		}
		l1 = l1.AddSaturated(x.Abs().AsUint16x16())
		in[k] = x
	}
	return itxL1Ok16(l1)
}

// itxRowStrip16 runs the 16-point row pass (network t) for rows r0..r0+15
// into tmp (width 16), rounding by n.
func itxRowStrip16(coeff []int32, stride, r0, inCols int, t tx1DType, rect2 bool, n uint8, tmp []int16, colL1 *archsimd.Uint16x16) bool {
	var in [16]archsimd.Int16x16
	if !itxRowLoad(coeff, stride, r0, inCols, rect2, in[:]) {
		archsimd.ClearAVXUpperBits()
		return false
	}
	var out [16]archsimd.Int16x16
	itx16Net1D(t, inCols, &in, &out)
	itxStoreRowGroup(&out, n, tmp, 16, r0, 0, colL1)
	archsimd.ClearAVXUpperBits()
	return true
}

// itxRowStrip32 runs the 32-point DCT row pass for rows r0..r0+15 into tmp
// (width 32), rounding by n.
func itxRowStrip32(coeff []int32, stride, r0, inCols int, rect2 bool, n uint8, tmp []int16, colL1 *[2]archsimd.Uint16x16) bool {
	var in [32]archsimd.Int16x16
	if !itxRowLoad(coeff, stride, r0, inCols, rect2, in[:]) {
		archsimd.ClearAVXUpperBits()
		return false
	}
	var out [32]archsimd.Int16x16
	itx32Net(inCols, &in, &out)
	for g := 0; g < 2; g++ {
		itxStoreRowGroup((*[16]archsimd.Int16x16)(out[16*g:]), n, tmp, 32, r0, 16*g, &colL1[g])
	}
	archsimd.ClearAVXUpperBits()
	return true
}

// itxRowStrip64 runs the 64-point DCT row pass for rows r0..r0+15 into tmp
// (width 64), rounding by 2.
func itxRowStrip64(coeff []int32, stride, r0, inCols int, tmp []int16, colL1 *[4]archsimd.Uint16x16) bool {
	var in [32]archsimd.Int16x16
	if !itxRowLoad(coeff, stride, r0, inCols, false, in[:]) {
		archsimd.ClearAVXUpperBits()
		return false
	}
	var out [64]archsimd.Int16x16
	itx64Net(inCols, &in, &out)
	for g := 0; g < 4; g++ {
		itxStoreRowGroup((*[16]archsimd.Int16x16)(out[16*g:]), 2, tmp, 64, r0, 16*g, &colL1[g])
	}
	archsimd.ClearAVXUpperBits()
	return true
}

// itxColAdd adds Round2(out[r], 4) to rows 0..len(out)-1 of the target at
// columns c0..c0+15.
func itxColAdd(out []archsimd.Int16x16, c0 int, px *itxPix) {
	if px.maxV == 0 {
		for r := range out {
			itxAddStore16(px.dst[r*px.dstStride+c0:], px.pred[r*px.predStride+c0:], itxRound2x16(out[r], 4))
		}
		archsimd.ClearAVXUpperBits()
		return
	}
	for r := range out {
		itxAddStore16HBD(px.dst[r*px.dstStride+2*c0:], px.pred[r*px.predStride+2*c0:], itxRound2x16(out[r], 4), px.maxV)
	}
	archsimd.ClearAVXUpperBits()
}

// itxColStripAdd16 runs the 16-point column pass (network t) for columns
// c0..c0+15 from inRows tmp rows (width w) and adds the result.
func itxColStripAdd16(tmp []int16, w, c0, inRows int, t tx1DType, px *itxPix) {
	var in [16]archsimd.Int16x16
	for r := 0; r < inRows; r++ {
		in[r] = archsimd.LoadInt16x16(tmp[r*w+c0:])
	}
	var out [16]archsimd.Int16x16
	itx16Net1D(t, inRows, &in, &out)
	itxColAdd(out[:], c0, px)
	archsimd.ClearAVXUpperBits()
}

// itxColStripAdd32 runs the 32-point DCT column pass for columns c0..c0+15.
func itxColStripAdd32(tmp []int16, w, c0, inRows int, px *itxPix) {
	var in [32]archsimd.Int16x16
	for r := 0; r < inRows; r++ {
		in[r] = archsimd.LoadInt16x16(tmp[r*w+c0:])
	}
	var out [32]archsimd.Int16x16
	itx32Net(inRows, &in, &out)
	itxColAdd(out[:], c0, px)
	archsimd.ClearAVXUpperBits()
}

// itxColStripAdd64 runs the 64-point DCT column pass for columns c0..c0+15.
func itxColStripAdd64(tmp []int16, w, c0, inRows int, px *itxPix) {
	var in [32]archsimd.Int16x16
	for r := 0; r < inRows; r++ {
		in[r] = archsimd.LoadInt16x16(tmp[r*w+c0:])
	}
	var out [64]archsimd.Int16x16
	itx64Net(inRows, &in, &out)
	itxColAdd(out[:], c0, px)
	archsimd.ClearAVXUpperBits()
}

// itx16Form rounds a coded extent up to the 16-point network form covering
// it (4, 8 or 16).
func itx16Form(n int) int {
	switch {
	case n <= 4:
		return 4
	case n <= 8:
		return 8
	default:
		return 16
	}
}

// itx64Form rounds a coded extent up to the 32/64-point network form
// covering it (8, 16 or 32).
func itx64Form(n int) int {
	switch {
	case n <= 8:
		return 8
	case n <= 16:
		return 16
	default:
		return 32
	}
}

// itxBlockOK checks the common contract of the fused kernels for a w x h
// block of bytesPerSample-byte samples whose coefficients are maxCols
// columns of maxRows at coeffStride, activeRows x activeCols of them coded.
func itxBlockOK(w, h, bytesPerSample, maxRows, maxCols int, dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, activeRows, activeCols int) bool {
	rowBytes := w * bytesPerSample
	return itxAVX2 && activeRows > 0 && activeRows <= maxRows && activeCols > 0 && activeCols <= maxCols &&
		coeffStride >= maxRows && len(coeff) >= (maxCols-1)*coeffStride+maxRows &&
		dstStride >= rowBytes && predStride >= rowBytes &&
		len(dst) >= (h-1)*dstStride+rowBytes && len(pred) >= (h-1)*predStride+rowBytes
}

// itxHBDMax returns the pixel maximum of a 10- or 12-bit block, false for
// any other depth.
func itxHBDMax(bitDepth uint8) (int16, bool) {
	if bitDepth != 10 && bitDepth != 12 {
		return 0, false
	}
	return int16(1)<<bitDepth - 1, true
}

// itxSquare16 is the 16x16 kernel for a horizontal (row) network th and a
// vertical (column) network tv.
func itxSquare16(px *itxPix, coeff []int32, coeffStride int, tv, th tx1DType, activeRows, activeCols int) bool {
	inCols := itx16Form(activeCols)
	inRows := itx16Form(activeRows)
	var tmp [16 * 16]int16
	var colL1 archsimd.Uint16x16
	if !itxRowStrip16(coeff, coeffStride, 0, inCols, th, false, 2, tmp[:], &colL1) || !itxL1Ok16(colL1) {
		archsimd.ClearAVXUpperBits()
		return false
	}
	itxColStripAdd16(tmp[:], 16, 0, inRows, tv, px)
	archsimd.ClearAVXUpperBits()
	return true
}

// itxSquare8 is the 8x8 kernel: the whole block stays in xmm registers.
func itxSquare8(px *itxPix, coeff []int32, coeffStride int, tv, th tx1DType, activeRows, activeCols int) bool {
	var in [8]archsimd.Int16x8
	var l1 archsimd.Uint16x8
	for k := 0; k < activeCols; k++ {
		x := itxLoadNarrow8(coeff[k*coeffStride : k*coeffStride+8])
		l1 = l1.AddSaturated(x.Abs().AsUint16x8())
		in[k] = x
	}
	if !itxL1Ok8(l1) {
		return false
	}
	var mid [8]archsimd.Int16x8
	itxRect8RowNet(th, activeCols, &in, &mid)
	itxTranspose8x8(&mid)
	var colL1 archsimd.Uint16x8
	for r := range mid {
		colL1 = colL1.AddSaturated(mid[r].Abs().AsUint16x8())
	}
	if !itxL1Ok8(colL1) {
		return false
	}
	var out [8]archsimd.Int16x8
	itx48Net8(tv, activeRows, &mid, &out)
	if px.maxV == 0 {
		for r := 0; r < 8; r++ {
			itxAddStore8(px.dst[r*px.dstStride:], px.pred[r*px.predStride:], itxRound2x8(out[r], 4))
		}
		return true
	}
	for r := 0; r < 8; r++ {
		itxAddStore8HBD(px.dst[r*px.dstStride:], px.pred[r*px.predStride:], itxRound2x8(out[r], 4), px.maxV)
	}
	return true
}

// itxDCT32 is the 32x32 DCT_DCT kernel for any coded extent.
func itxDCT32(px *itxPix, coeff []int32, coeffStride int, activeRows, activeCols int) bool {
	inCols := itx64Form(activeCols)
	inRows := itx64Form(activeRows)
	var tmp [32 * 32]int16
	var colL1 [2]archsimd.Uint16x16
	written := 0
	for ; written < activeRows; written += 16 {
		if !itxRowStrip32(coeff, coeffStride, written, inCols, false, 2, tmp[:], &colL1) {
			archsimd.ClearAVXUpperBits()
			return false
		}
	}
	// The column form may read tmp rows past the strips that ran; those
	// rows are the transform of zero coefficient rows, so zero.
	if written < inRows {
		clear(tmp[written*32 : inRows*32])
	}
	if !itxL1Ok16(colL1[0].Max(colL1[1])) {
		archsimd.ClearAVXUpperBits()
		return false
	}
	itxColStripAdd32(tmp[:], 32, 0, inRows, px)
	itxColStripAdd32(tmp[:], 32, 16, inRows, px)
	archsimd.ClearAVXUpperBits()
	return true
}

// itxDCT64 is the 64x64 DCT_DCT kernel (coded coefficients in the 32x32
// corner).
func itxDCT64(px *itxPix, coeff []int32, coeffStride int, activeRows, activeCols int) bool {
	inCols := itx64Form(activeCols)
	inRows := itx64Form(activeRows)
	var tmp [32 * 64]int16
	var colL1 [4]archsimd.Uint16x16
	written := 0
	for ; written < activeRows; written += 16 {
		if !itxRowStrip64(coeff, coeffStride, written, inCols, tmp[:], &colL1) {
			archsimd.ClearAVXUpperBits()
			return false
		}
	}
	if written < inRows {
		clear(tmp[written*64 : inRows*64])
	}
	if !itxL1Ok16(colL1[0].Max(colL1[1]).Max(colL1[2].Max(colL1[3]))) {
		archsimd.ClearAVXUpperBits()
		return false
	}
	for c0 := 0; c0 < 64; c0 += 16 {
		itxColStripAdd64(tmp[:], 64, c0, inRows, px)
	}
	archsimd.ClearAVXUpperBits()
	return true
}

// itxDCT16x32 is the 16 wide, 32 high DCT_DCT kernel: rect2-scaled DCT16
// rows with the inter-pass Round2(., 1), then DCT32 columns.
func itxDCT16x32(px *itxPix, coeff []int32, coeffStride int, activeRows, activeCols int) bool {
	inCols := itx16Form(activeCols)
	inRows := itx64Form(activeRows)
	var tmp [32 * 16]int16
	var colL1 archsimd.Uint16x16
	written := 0
	for ; written < activeRows; written += 16 {
		if !itxRowStrip16(coeff, coeffStride, written, inCols, tx1DDCT, true, 1, tmp[:], &colL1) {
			archsimd.ClearAVXUpperBits()
			return false
		}
	}
	if written < inRows {
		clear(tmp[written*16 : inRows*16])
	}
	if !itxL1Ok16(colL1) {
		archsimd.ClearAVXUpperBits()
		return false
	}
	itxColStripAdd32(tmp[:], 16, 0, inRows, px)
	archsimd.ClearAVXUpperBits()
	return true
}

// itxDCT32x16 is itxDCT16x32 for the 32 wide, 16 high block.
func itxDCT32x16(px *itxPix, coeff []int32, coeffStride int, activeRows, activeCols int) bool {
	inCols := itx64Form(activeCols)
	inRows := itx16Form(activeRows)
	var tmp [16 * 32]int16
	var colL1 [2]archsimd.Uint16x16
	if !itxRowStrip32(coeff, coeffStride, 0, inCols, true, 1, tmp[:], &colL1) || !itxL1Ok16(colL1[0].Max(colL1[1])) {
		archsimd.ClearAVXUpperBits()
		return false
	}
	itxColStripAdd16(tmp[:], 32, 0, inRows, tx1DDCT, px)
	itxColStripAdd16(tmp[:], 32, 16, inRows, tx1DDCT, px)
	archsimd.ClearAVXUpperBits()
	return true
}

// itxHybridTypes returns the per-pass networks of the DCT/ADST/flipped-ADST
// pairings the fused hybrid kernels cover; ok is false for every other
// transform type (including DCT_DCT, which has its own kernels).
func itxHybridTypes(txType Type) (vertical, horizontal tx1DType, ok bool) {
	switch txType {
	case TypeADSTDCT, TypeDCTADST, TypeADSTADST, TypeFlipADSTDCT, TypeDCTFlipADST,
		TypeFlipADSTFlipADST, TypeADSTFlipADST, TypeFlipADSTADST:
		return txType.tx1DTypes()
	}
	return 0, 0, false
}

// InverseAddDCT8x8 reconstructs one 8-bit 8x8 DCT_DCT block with coded
// coefficients in rows 0..activeRows-1 and columns 0..activeCols-1:
// dst = clip(pred + inverse(coeff)), with the dequantized coefficients
// column-major at coeffStride. pred may be dst. It reports false, with dst
// untouched, when the block is outside the kernel's contract or a guard
// declines; the caller then runs the staged path, which produces the same
// bytes.
func InverseAddDCT8x8(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, activeRows, activeCols int) bool {
	if !itxBlockOK(8, 8, 1, 8, 8, dst, dstStride, pred, predStride, coeff, coeffStride, activeRows, activeCols) {
		return false
	}
	px := itxPix{dst, dstStride, pred, predStride, 0}
	return itxSquare8(&px, coeff, coeffStride, tx1DDCT, tx1DDCT, activeRows, activeCols)
}

// InverseAdd8x8Hybrid is InverseAddDCT8x8 for the DCT/ADST/flipped-ADST
// pairings (see itxHybridTypes).
//
// At 8 bits it also covers the identity pairings but IDTX (V_DCT, H_DCT,
// V_ADST, H_ADST, V_FLIPADST, H_FLIPADST): identity8 rows give
// Round2(2x, 1) == x, and identity8 columns saturate in int16 where only the
// clipped sample could tell the difference (see itx_rect16_avx2_amd64.go).
func InverseAdd8x8Hybrid(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, txType Type, activeRows, activeCols int) bool {
	vertical, horizontal, ok := itxHybridTypes(txType)
	if !ok && txType > TypeIDTX && txType < TypeCount {
		vertical, horizontal, ok = itx4Types[txType][0], itx4Types[txType][1], true
	}
	if !ok || !itxBlockOK(8, 8, 1, 8, 8, dst, dstStride, pred, predStride, coeff, coeffStride, activeRows, activeCols) {
		return false
	}
	px := itxPix{dst, dstStride, pred, predStride, 0}
	return itxSquare8(&px, coeff, coeffStride, vertical, horizontal, activeRows, activeCols)
}

// InverseAddDCT16x16 is InverseAddDCT8x8 for the 16x16 block.
func InverseAddDCT16x16(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, activeRows, activeCols int) bool {
	if !itxBlockOK(16, 16, 1, 16, 16, dst, dstStride, pred, predStride, coeff, coeffStride, activeRows, activeCols) {
		return false
	}
	px := itxPix{dst, dstStride, pred, predStride, 0}
	return itxSquare16(&px, coeff, coeffStride, tx1DDCT, tx1DDCT, activeRows, activeCols)
}

// InverseAdd16x16Hybrid is InverseAdd8x8Hybrid for the 16x16 block.
func InverseAdd16x16Hybrid(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, txType Type, activeRows, activeCols int) bool {
	vertical, horizontal, ok := itxHybridTypes(txType)
	if !ok || !itxBlockOK(16, 16, 1, 16, 16, dst, dstStride, pred, predStride, coeff, coeffStride, activeRows, activeCols) {
		return false
	}
	px := itxPix{dst, dstStride, pred, predStride, 0}
	return itxSquare16(&px, coeff, coeffStride, vertical, horizontal, activeRows, activeCols)
}

// InverseAddDCT32x32Low8 is InverseAddDCT8x8 for the 32x32 block whose coded
// coefficients lie in rows and columns 0..7.
func InverseAddDCT32x32Low8(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int) bool {
	if !itxBlockOK(32, 32, 1, 32, 8, dst, dstStride, pred, predStride, coeff, coeffStride, 8, 8) {
		return false
	}
	px := itxPix{dst, dstStride, pred, predStride, 0}
	return itxDCT32(&px, coeff, coeffStride, 8, 8)
}

// InverseAddDCT32x32Low16 is InverseAddDCT32x32Low8 for the 16x16 corner.
func InverseAddDCT32x32Low16(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, activeRows int) bool {
	if !itxBlockOK(32, 32, 1, 32, 16, dst, dstStride, pred, predStride, coeff, coeffStride, 16, 16) {
		return false
	}
	px := itxPix{dst, dstStride, pred, predStride, 0}
	return itxDCT32(&px, coeff, coeffStride, 16, 16)
}

// InverseAddDCT32x32Full is InverseAddDCT8x8 for the 32x32 block with any
// coded extent.
func InverseAddDCT32x32Full(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, activeRows, activeCols int) bool {
	if !itxBlockOK(32, 32, 1, 32, 32, dst, dstStride, pred, predStride, coeff, coeffStride, activeRows, activeCols) {
		return false
	}
	px := itxPix{dst, dstStride, pred, predStride, 0}
	return itxDCT32(&px, coeff, coeffStride, activeRows, activeCols)
}

// InverseAddDCT64x64 is InverseAddDCT8x8 for the 64x64 block, whose coded
// coefficients lie in the 32x32 corner (the 32x32 scan at coeffStride).
func InverseAddDCT64x64(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, activeRows, activeCols int) bool {
	if !itxBlockOK(64, 64, 1, 32, 32, dst, dstStride, pred, predStride, coeff, coeffStride, activeRows, activeCols) {
		return false
	}
	px := itxPix{dst, dstStride, pred, predStride, 0}
	return itxDCT64(&px, coeff, coeffStride, activeRows, activeCols)
}

// InverseAddDCT16x32 is InverseAddDCT8x8 for the 16 wide, 32 high block.
func InverseAddDCT16x32(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, activeRows, activeCols int) bool {
	if !itxBlockOK(16, 32, 1, 32, 16, dst, dstStride, pred, predStride, coeff, coeffStride, activeRows, activeCols) {
		return false
	}
	px := itxPix{dst, dstStride, pred, predStride, 0}
	return itxDCT16x32(&px, coeff, coeffStride, activeRows, activeCols)
}

// InverseAddDCT32x16 is InverseAddDCT8x8 for the 32 wide, 16 high block.
func InverseAddDCT32x16(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, activeRows, activeCols int) bool {
	if !itxBlockOK(32, 16, 1, 16, 32, dst, dstStride, pred, predStride, coeff, coeffStride, activeRows, activeCols) {
		return false
	}
	px := itxPix{dst, dstStride, pred, predStride, 0}
	return itxDCT32x16(&px, coeff, coeffStride, activeRows, activeCols)
}

// The 10- and 12-bit forms (2-byte samples, strides in bytes). dav1d's
// itx16 reuses its 8bpc int16 column functions at 10 bits; here both passes
// are the 8-bit int16 networks under the same per-lane L1 guards, so none
// of the staged path's bd + 8 row or max(bd + 6, 16) column clamps can bite
// (see itx_fused_hbd_gosimd_arm64.go), and the add saturates and clips to
// the bit depth.

// InverseAddDCT8x8HBD is InverseAddDCT8x8 for a 10- or 12-bit block.
func InverseAddDCT8x8HBD(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, activeRows, activeCols int, bitDepth uint8) bool {
	maxV, ok := itxHBDMax(bitDepth)
	if !ok || !itxBlockOK(8, 8, 2, 8, 8, dst, dstStride, pred, predStride, coeff, coeffStride, activeRows, activeCols) {
		return false
	}
	px := itxPix{dst, dstStride, pred, predStride, maxV}
	return itxSquare8(&px, coeff, coeffStride, tx1DDCT, tx1DDCT, activeRows, activeCols)
}

// InverseAdd8x8HybridHBD is InverseAdd8x8Hybrid for a 10- or 12-bit block.
func InverseAdd8x8HybridHBD(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, txType Type, activeRows, activeCols int, bitDepth uint8) bool {
	vertical, horizontal, ok := itxHybridTypes(txType)
	maxV, okBD := itxHBDMax(bitDepth)
	if !ok || !okBD || !itxBlockOK(8, 8, 2, 8, 8, dst, dstStride, pred, predStride, coeff, coeffStride, activeRows, activeCols) {
		return false
	}
	px := itxPix{dst, dstStride, pred, predStride, maxV}
	return itxSquare8(&px, coeff, coeffStride, vertical, horizontal, activeRows, activeCols)
}

// InverseAddDCT16x16HBD is InverseAddDCT16x16 for a 10- or 12-bit block.
func InverseAddDCT16x16HBD(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, activeRows, activeCols int, bitDepth uint8) bool {
	maxV, ok := itxHBDMax(bitDepth)
	if !ok || !itxBlockOK(16, 16, 2, 16, 16, dst, dstStride, pred, predStride, coeff, coeffStride, activeRows, activeCols) {
		return false
	}
	px := itxPix{dst, dstStride, pred, predStride, maxV}
	return itxSquare16(&px, coeff, coeffStride, tx1DDCT, tx1DDCT, activeRows, activeCols)
}

// InverseAdd16x16HybridHBD is InverseAdd16x16Hybrid for a 10- or 12-bit
// block.
func InverseAdd16x16HybridHBD(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, txType Type, activeRows, activeCols int, bitDepth uint8) bool {
	vertical, horizontal, ok := itxHybridTypes(txType)
	maxV, okBD := itxHBDMax(bitDepth)
	if !ok || !okBD || !itxBlockOK(16, 16, 2, 16, 16, dst, dstStride, pred, predStride, coeff, coeffStride, activeRows, activeCols) {
		return false
	}
	px := itxPix{dst, dstStride, pred, predStride, maxV}
	return itxSquare16(&px, coeff, coeffStride, vertical, horizontal, activeRows, activeCols)
}

// InverseAddDCT32x32HBD is InverseAddDCT32x32Full for a 10- or 12-bit block.
func InverseAddDCT32x32HBD(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, activeRows, activeCols int, bitDepth uint8) bool {
	maxV, ok := itxHBDMax(bitDepth)
	if !ok || !itxBlockOK(32, 32, 2, 32, 32, dst, dstStride, pred, predStride, coeff, coeffStride, activeRows, activeCols) {
		return false
	}
	px := itxPix{dst, dstStride, pred, predStride, maxV}
	return itxDCT32(&px, coeff, coeffStride, activeRows, activeCols)
}

// InverseAddDCT64x64HBD is InverseAddDCT64x64 for a 10- or 12-bit block.
func InverseAddDCT64x64HBD(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, activeRows, activeCols int, bitDepth uint8) bool {
	maxV, ok := itxHBDMax(bitDepth)
	if !ok || !itxBlockOK(64, 64, 2, 32, 32, dst, dstStride, pred, predStride, coeff, coeffStride, activeRows, activeCols) {
		return false
	}
	px := itxPix{dst, dstStride, pred, predStride, maxV}
	return itxDCT64(&px, coeff, coeffStride, activeRows, activeCols)
}

// InverseAddDCT16x32HBD is InverseAddDCT16x32 for a 10- or 12-bit block.
func InverseAddDCT16x32HBD(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, activeRows, activeCols int, bitDepth uint8) bool {
	maxV, ok := itxHBDMax(bitDepth)
	if !ok || !itxBlockOK(16, 32, 2, 32, 16, dst, dstStride, pred, predStride, coeff, coeffStride, activeRows, activeCols) {
		return false
	}
	px := itxPix{dst, dstStride, pred, predStride, maxV}
	return itxDCT16x32(&px, coeff, coeffStride, activeRows, activeCols)
}

// InverseAddDCT32x16HBD is InverseAddDCT32x16 for a 10- or 12-bit block.
func InverseAddDCT32x16HBD(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, activeRows, activeCols int, bitDepth uint8) bool {
	maxV, ok := itxHBDMax(bitDepth)
	if !ok || !itxBlockOK(32, 16, 2, 16, 32, dst, dstStride, pred, predStride, coeff, coeffStride, activeRows, activeCols) {
		return false
	}
	px := itxPix{dst, dstStride, pred, predStride, maxV}
	return itxDCT32x16(&px, coeff, coeffStride, activeRows, activeCols)
}

func itxRoundBias(amount uint8) archsimd.Int32x4 {
	return archsimd.BroadcastInt32x4(1 << (amount - 1))
}

func itxShiftAmount(amount int32) archsimd.Int32x4 {
	return archsimd.BroadcastInt32x4(-amount)
}

func itxMulAcc(accumulator, value, coefficient archsimd.Int32x4) archsimd.Int32x4 {
	return accumulator.Add(value.Mul(coefficient))
}

func itxShr12(value, amount archsimd.Int32x4) archsimd.Int32x4 {
	return value.ShiftAllRight(12)
}

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
