// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

//go:build goexperiment.simd && amd64 && !purego

package transform

import (
	"simd/archsimd"
	"unsafe"
)

// These index vectors duplicate each qword's low and high dwords. That lets
// one AVX2 widening multiply handle all four paired row/column lanes.
var (
	dctPairLowWords  = [8]uint32{0, 0, 2, 2, 4, 4, 6, 6}
	dctPairHighWords = [8]uint32{1, 1, 3, 3, 5, 5, 7, 7}
)

func dctPairLoad(p0, p1 unsafe.Pointer, stride uintptr, index uintptr) archsimd.Int64x4 {
	a := *(*int32)(unsafe.Add(p0, index*stride))
	b := *(*int32)(unsafe.Add(p1, index*stride))
	return archsimd.BroadcastInt64x4(int64(a)).InterleaveLoGrouped(archsimd.BroadcastInt64x4(int64(b)))
}

func dctPairStore(p0, p1 unsafe.Pointer, stride uintptr, index uintptr, v archsimd.Int64x4) {
	var pair [4]int64
	v.StoreArray(&pair)
	*(*int32)(unsafe.Add(p0, index*stride)) = int32(pair[0])
	*(*int32)(unsafe.Add(p1, index*stride)) = int32(pair[1])
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
// per coefficient. Its input and output pointers may describe two rows or two
// adjacent columns; stride is the byte distance between successive samples.
func inverseDCT8PairSIMD(p0, p1 unsafe.Pointer, stride uintptr, min, max int32) {
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
}

func inverseDCT8Row2SIMDAdapter(r0, r1 []int32, min, max int32) {
	if len(r0) < dct8Size || len(r1) < dct8Size {
		inverseDCT8Row2PureGo(r0, r1, min, max)
		return
	}
	inverseDCT8PairSIMD(unsafe.Pointer(&r0[0]), unsafe.Pointer(&r1[0]), 4, min, max)
}

func inverseDCT8Col2SIMDAdapter(buf []int32, rowStride int, min, max int32) {
	if rowStride < 2 || len(buf) < (dct8Size-1)*rowStride+2 {
		inverseDCT8Col2PureGo(buf, rowStride, min, max)
		return
	}
	base := unsafe.Pointer(&buf[0])
	inverseDCT8PairSIMD(base, unsafe.Add(base, 4), uintptr(rowStride)*4, min, max)
}

// inverseDCT16PairSIMD applies inverseDCT16 to two rows or columns. The
// inverseDCT8 even pass is written inline so vectors stay live across stages.
func inverseDCT16PairSIMD(p0, p1 unsafe.Pointer, stride uintptr, min, max int32) {
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
}

func inverseDCT16Row2SIMDAdapter(r0, r1 []int32, min, max int32) {
	if len(r0) < dct16Size || len(r1) < dct16Size {
		inverseDCT16Row2PureGo(r0, r1, min, max)
		return
	}
	inverseDCT16PairSIMD(unsafe.Pointer(&r0[0]), unsafe.Pointer(&r1[0]), 4, min, max)
}

func inverseDCT16Col2SIMDAdapter(buf []int32, rowStride int, min, max int32) {
	if rowStride < 2 || len(buf) < (dct16Size-1)*rowStride+2 {
		inverseDCT16Col2PureGo(buf, rowStride, min, max)
		return
	}
	base := unsafe.Pointer(&buf[0])
	inverseDCT16PairSIMD(base, unsafe.Add(base, 4), uintptr(rowStride)*4, min, max)
}

func init() {
	if archsimd.X86.AVX2() {
		inverseDCT8Row2Impl = inverseDCT8Row2SIMDAdapter
		inverseDCT8Col2Impl = inverseDCT8Col2SIMDAdapter
		inverseDCT16Row2Impl = inverseDCT16Row2SIMDAdapter
		inverseDCT16Col2Impl = inverseDCT16Col2SIMDAdapter
	}
}
