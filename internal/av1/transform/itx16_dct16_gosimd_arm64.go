// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package transform

import "simd/archsimd"

// int16 8-wide inverse DCT column kernels in dav1d's 8bpc form, on the stock
// archsimd API:
//
//   - every rotation Round2(a*ca + b*cb, n) widens both halves with
//     SMULL/SMULL2 (the compiler fuses x.HiToLo().MulWidenLo(k.HiToLo()) into
//     SMULL2), sums in int32, then rounds and narrows in two ops per four
//     lanes: a 32-bit MLA computes 2^(16-n)*p + 2^15, and UZP2 (ConcatOdd on
//     the 16-bit view) keeps the high halves, which are floor((p + 2^(n-1)) /
//     2^n) modulo 2^16, so the narrow is exact whenever the true rounded value
//     fits int16;
//   - the cos(pi/4) stages use dav1d's 2896 form, Round2(2896*a - 2896*b, 12)
//     == Round2((a-b)*181, 8), so no intermediate difference is formed in
//     int16;
//   - the butterfly sums are SQADD/SQSUB, which on the 8-bit path equal the
//     scalar clip to [-32768, 32767].
//
// A kernel is therefore byte-identical to the scalar inverseDCTn[int16] for
// every input whose rotation outputs all fit int16 (itx16_gain_test.go bounds
// those by the per-column L1 norm of the inputs). Callers guard with
// itx16ColumnInputSafe and fall back to the int32 path otherwise.

// itx16K holds the broadcast int16 rotation constants.
var itx16K = [...][8]int16{
	{2896, 2896, 2896, 2896, 2896, 2896, 2896, 2896},
	{-2896, -2896, -2896, -2896, -2896, -2896, -2896, -2896},
	{1567, 1567, 1567, 1567, 1567, 1567, 1567, 1567},
	{3784, 3784, 3784, 3784, 3784, 3784, 3784, 3784},
	{-3784, -3784, -3784, -3784, -3784, -3784, -3784, -3784},
	{799, 799, 799, 799, 799, 799, 799, 799},
	{4017, 4017, 4017, 4017, 4017, 4017, 4017, 4017},
	{-4017, -4017, -4017, -4017, -4017, -4017, -4017, -4017},
	{1703, 1703, 1703, 1703, 1703, 1703, 1703, 1703},
	{1138, 1138, 1138, 1138, 1138, 1138, 1138, 1138},
	{-1138, -1138, -1138, -1138, -1138, -1138, -1138, -1138},
	{401, 401, 401, 401, 401, 401, 401, 401},
	{4076, 4076, 4076, 4076, 4076, 4076, 4076, 4076},
	{-4076, -4076, -4076, -4076, -4076, -4076, -4076, -4076},
	{1583, 1583, 1583, 1583, 1583, 1583, 1583, 1583},
	{1299, 1299, 1299, 1299, 1299, 1299, 1299, 1299},
	{-1299, -1299, -1299, -1299, -1299, -1299, -1299, -1299},
	{1931, 1931, 1931, 1931, 1931, 1931, 1931, 1931},
	{3612, 3612, 3612, 3612, 3612, 3612, 3612, 3612},
	{-3612, -3612, -3612, -3612, -3612, -3612, -3612, -3612},
	{3920, 3920, 3920, 3920, 3920, 3920, 3920, 3920},
	{1189, 1189, 1189, 1189, 1189, 1189, 1189, 1189},
	{-1189, -1189, -1189, -1189, -1189, -1189, -1189, -1189},
	{-1567, -1567, -1567, -1567, -1567, -1567, -1567, -1567},
}

const (
	itx16Kp2896 = iota
	itx16Km2896
	itx16Kp1567
	itx16Kp3784
	itx16Km3784
	itx16Kp799
	itx16Kp4017
	itx16Km4017
	itx16Kp1703
	itx16Kp1138
	itx16Km1138
	itx16Kp401
	itx16Kp4076
	itx16Km4076
	itx16Kp1583
	itx16Kp1299
	itx16Km1299
	itx16Kp1931
	itx16Kp3612
	itx16Km3612
	itx16Kp3920
	itx16Kp1189
	itx16Km1189
	itx16Km1567
)

// itx16Scale[n] is the MLA multiplier 2^(16-n) for Round2(., n); the bias is
// always 2^15.
var itx16Scale12 = [4]int32{16, 16, 16, 16}
var itx16Scale11 = [4]int32{32, 32, 32, 32}
var itx16Bias = [4]int32{32768, 32768, 32768, 32768}

// itx16Rot is Round2(a*ca + b*cb, 12) on eight lanes.
func itx16Rot(a, b, ca, cb archsimd.Int16x8, scale archsimd.Int32x4) archsimd.Int16x8 {
	bias := archsimd.LoadInt32x4Array(&itx16Bias)
	lo := a.MulWidenLo(ca).Add(b.MulWidenLo(cb)).MulAdd(scale, bias)
	bias = archsimd.LoadInt32x4Array(&itx16Bias)
	hi := a.HiToLo().MulWidenLo(ca.HiToLo()).Add(b.HiToLo().MulWidenLo(cb.HiToLo())).MulAdd(scale, bias)
	return lo.ToBits().ReshapeToUint16s().ConcatOdd(hi.ToBits().ReshapeToUint16s()).BitsToInt16()
}

// inverseDCT16Col8SIMD16v2 applies inverseDCT16 to the eight int16 columns
// buf[k*stride+0..7]. Exact under the itx16 column guard; the clamp range is
// the int16 range (the 8-bit column bounds), which the saturating adds apply,
// so min and max are only accepted for the inverseDCT16Col8Impl16 signature.
func inverseDCT16Col8SIMD16v2(buf []int16, stride int, min int32, max int32) {
	_ = buf[15*stride+7]
	ld := func(k int) archsimd.Int16x8 { return archsimd.LoadInt16x8Array((*[8]int16)(buf[k*stride:])) }
	k := func(i int) archsimd.Int16x8 { return archsimd.LoadInt16x8Array(&itx16K[i]) }
	s12 := archsimd.LoadInt32x4Array(&itx16Scale12)
	s11 := archsimd.LoadInt32x4Array(&itx16Scale11)
	p2896, m2896 := k(itx16Kp2896), k(itx16Km2896)
	p1567, p3784, m3784 := k(itx16Kp1567), k(itx16Kp3784), k(itx16Km3784)

	// Even half: inverseDCT8 on rows 0,2,...,14 (inverseDCT4 on 0,4,8,12).
	in0, in4, in8, in12 := ld(0), ld(4), ld(8), ld(12)
	t0 := itx16Rot(in0, in8, p2896, p2896, s12)
	t1 := itx16Rot(in0, in8, p2896, m2896, s12)
	t2 := itx16Rot(in4, in12, p1567, m3784, s12)
	t3 := itx16Rot(in4, in12, p3784, p1567, s12)
	d0 := t0.AddSaturated(t3)
	d1 := t1.AddSaturated(t2)
	d2 := t1.SubSaturated(t2)
	d3 := t0.SubSaturated(t3)

	in2, in6, in10, in14 := ld(2), ld(6), ld(10), ld(14)
	t4a := itx16Rot(in2, in14, k(itx16Kp799), k(itx16Km4017), s12)
	t5a := itx16Rot(in10, in6, k(itx16Kp1703), k(itx16Km1138), s11)
	t6a := itx16Rot(in10, in6, k(itx16Kp1138), k(itx16Kp1703), s11)
	t7a := itx16Rot(in2, in14, k(itx16Kp4017), k(itx16Kp799), s12)
	t4 := t4a.AddSaturated(t5a)
	t5a = t4a.SubSaturated(t5a)
	t7 := t7a.AddSaturated(t6a)
	t6a = t7a.SubSaturated(t6a)
	t5 := itx16Rot(t6a, t5a, p2896, m2896, s12)
	t6 := itx16Rot(t6a, t5a, p2896, p2896, s12)
	e0 := d0.AddSaturated(t7)
	e1 := d1.AddSaturated(t6)
	e2 := d2.AddSaturated(t5)
	e3 := d3.AddSaturated(t4)
	e4 := d3.SubSaturated(t4)
	e5 := d2.SubSaturated(t5)
	e6 := d1.SubSaturated(t6)
	e7 := d0.SubSaturated(t7)

	// Odd half.
	in1, in15 := ld(1), ld(15)
	t8a := itx16Rot(in1, in15, k(itx16Kp401), k(itx16Km4076), s12)
	t15a := itx16Rot(in1, in15, k(itx16Kp4076), k(itx16Kp401), s12)
	in9, in7 := ld(9), ld(7)
	t9a := itx16Rot(in9, in7, k(itx16Kp1583), k(itx16Km1299), s11)
	t14a := itx16Rot(in9, in7, k(itx16Kp1299), k(itx16Kp1583), s11)
	in5, in11 := ld(5), ld(11)
	t10a := itx16Rot(in5, in11, k(itx16Kp1931), k(itx16Km3612), s12)
	t13a := itx16Rot(in5, in11, k(itx16Kp3612), k(itx16Kp1931), s12)
	in13, in3 := ld(13), ld(3)
	t11a := itx16Rot(in13, in3, k(itx16Kp3920), k(itx16Km1189), s12)
	t12a := itx16Rot(in13, in3, k(itx16Kp1189), k(itx16Kp3920), s12)

	t8 := t8a.AddSaturated(t9a)
	t9 := t8a.SubSaturated(t9a)
	t10 := t11a.SubSaturated(t10a)
	t11 := t11a.AddSaturated(t10a)
	t12 := t12a.AddSaturated(t13a)
	t13 := t12a.SubSaturated(t13a)
	t14 := t15a.SubSaturated(t14a)
	t15 := t15a.AddSaturated(t14a)

	t9a = itx16Rot(t14, t9, p1567, m3784, s12)
	t14a = itx16Rot(t14, t9, p3784, p1567, s12)
	t10a = itx16Rot(t13, t10, m3784, k(itx16Km1567), s12)
	t13a = itx16Rot(t13, t10, p1567, m3784, s12)

	t8a = t8.AddSaturated(t11)
	t9 = t9a.AddSaturated(t10a)
	t10 = t9a.SubSaturated(t10a)
	t11a = t8.SubSaturated(t11)
	t12a = t15.SubSaturated(t12)
	t13 = t14a.SubSaturated(t13a)
	t14 = t14a.AddSaturated(t13a)
	t15a = t15.AddSaturated(t12)

	t10a = itx16Rot(t13, t10, p2896, m2896, s12)
	t13a = itx16Rot(t13, t10, p2896, p2896, s12)
	t11 = itx16Rot(t12a, t11a, p2896, m2896, s12)
	t12 = itx16Rot(t12a, t11a, p2896, p2896, s12)

	st := func(k int, v archsimd.Int16x8) { v.StoreArray((*[8]int16)(buf[k*stride:])) }
	st(0, e0.AddSaturated(t15a))
	st(15, e0.SubSaturated(t15a))
	st(1, e1.AddSaturated(t14))
	st(14, e1.SubSaturated(t14))
	st(2, e2.AddSaturated(t13a))
	st(13, e2.SubSaturated(t13a))
	st(3, e3.AddSaturated(t12))
	st(12, e3.SubSaturated(t12))
	st(4, e4.AddSaturated(t11))
	st(11, e4.SubSaturated(t11))
	st(5, e5.AddSaturated(t10a))
	st(10, e5.SubSaturated(t10a))
	st(6, e6.AddSaturated(t9))
	st(9, e6.SubSaturated(t9))
	st(7, e7.AddSaturated(t8a))
	st(8, e7.SubSaturated(t8a))
}

// inverseDCT16Col8RawSIMD16 is inverseDCT16Col8SIMD16v2 over eight int32
// columns buf[k*stride+0..7] whose values are already in the int16 range: the
// loads narrow (SQXTN/SQXTN2), the per-column L1 guard accumulates alongside,
// and the stores widen back (SXTL/SXTL2). When a column fails the guard
// nothing is stored and it reports false, so the caller can run the int32
// kernels on the untouched columns.
func inverseDCT16Col8RawSIMD16(buf []int32, stride int) bool {
	_ = buf[15*stride+7]
	var l1 archsimd.Uint16x8
	ld := func(k int) archsimd.Int16x8 {
		lo := archsimd.LoadInt32x4Array((*[4]int32)(buf[k*stride:]))
		hi := archsimd.LoadInt32x4Array((*[4]int32)(buf[k*stride+4:]))
		v := lo.SaturateToInt16().ToBits().ReshapeToUint64s().BitsToFloat64()
		v = v.SetElem(1, hi.SaturateToInt16().ToBits().ReshapeToUint64s().BitsToFloat64().GetElem(0))
		x := v.ToBits().ReshapeToUint16s().BitsToInt16()
		l1 = l1.AddSaturated(x.Abs().ToBits())
		return x
	}
	k := func(i int) archsimd.Int16x8 { return archsimd.LoadInt16x8Array(&itx16K[i]) }
	s12 := archsimd.LoadInt32x4Array(&itx16Scale12)
	s11 := archsimd.LoadInt32x4Array(&itx16Scale11)
	p2896, m2896 := k(itx16Kp2896), k(itx16Km2896)
	p1567, p3784, m3784 := k(itx16Kp1567), k(itx16Kp3784), k(itx16Km3784)

	in0, in4, in8, in12 := ld(0), ld(4), ld(8), ld(12)
	t0 := itx16Rot(in0, in8, p2896, p2896, s12)
	t1 := itx16Rot(in0, in8, p2896, m2896, s12)
	t2 := itx16Rot(in4, in12, p1567, m3784, s12)
	t3 := itx16Rot(in4, in12, p3784, p1567, s12)
	d0 := t0.AddSaturated(t3)
	d1 := t1.AddSaturated(t2)
	d2 := t1.SubSaturated(t2)
	d3 := t0.SubSaturated(t3)

	in2, in6, in10, in14 := ld(2), ld(6), ld(10), ld(14)
	t4a := itx16Rot(in2, in14, k(itx16Kp799), k(itx16Km4017), s12)
	t5a := itx16Rot(in10, in6, k(itx16Kp1703), k(itx16Km1138), s11)
	t6a := itx16Rot(in10, in6, k(itx16Kp1138), k(itx16Kp1703), s11)
	t7a := itx16Rot(in2, in14, k(itx16Kp4017), k(itx16Kp799), s12)
	t4 := t4a.AddSaturated(t5a)
	t5a = t4a.SubSaturated(t5a)
	t7 := t7a.AddSaturated(t6a)
	t6a = t7a.SubSaturated(t6a)
	t5 := itx16Rot(t6a, t5a, p2896, m2896, s12)
	t6 := itx16Rot(t6a, t5a, p2896, p2896, s12)
	e0 := d0.AddSaturated(t7)
	e1 := d1.AddSaturated(t6)
	e2 := d2.AddSaturated(t5)
	e3 := d3.AddSaturated(t4)
	e4 := d3.SubSaturated(t4)
	e5 := d2.SubSaturated(t5)
	e6 := d1.SubSaturated(t6)
	e7 := d0.SubSaturated(t7)

	in1, in15 := ld(1), ld(15)
	t8a := itx16Rot(in1, in15, k(itx16Kp401), k(itx16Km4076), s12)
	t15a := itx16Rot(in1, in15, k(itx16Kp4076), k(itx16Kp401), s12)
	in9, in7 := ld(9), ld(7)
	t9a := itx16Rot(in9, in7, k(itx16Kp1583), k(itx16Km1299), s11)
	t14a := itx16Rot(in9, in7, k(itx16Kp1299), k(itx16Kp1583), s11)
	in5, in11 := ld(5), ld(11)
	t10a := itx16Rot(in5, in11, k(itx16Kp1931), k(itx16Km3612), s12)
	t13a := itx16Rot(in5, in11, k(itx16Kp3612), k(itx16Kp1931), s12)
	in13, in3 := ld(13), ld(3)
	t11a := itx16Rot(in13, in3, k(itx16Kp3920), k(itx16Km1189), s12)
	t12a := itx16Rot(in13, in3, k(itx16Kp1189), k(itx16Kp3920), s12)

	t8 := t8a.AddSaturated(t9a)
	t9 := t8a.SubSaturated(t9a)
	t10 := t11a.SubSaturated(t10a)
	t11 := t11a.AddSaturated(t10a)
	t12 := t12a.AddSaturated(t13a)
	t13 := t12a.SubSaturated(t13a)
	t14 := t15a.SubSaturated(t14a)
	t15 := t15a.AddSaturated(t14a)

	t9a = itx16Rot(t14, t9, p1567, m3784, s12)
	t14a = itx16Rot(t14, t9, p3784, p1567, s12)
	t10a = itx16Rot(t13, t10, m3784, k(itx16Km1567), s12)
	t13a = itx16Rot(t13, t10, p1567, m3784, s12)

	t8a = t8.AddSaturated(t11)
	t9 = t9a.AddSaturated(t10a)
	t10 = t9a.SubSaturated(t10a)
	t11a = t8.SubSaturated(t11)
	t12a = t15.SubSaturated(t12)
	t13 = t14a.SubSaturated(t13a)
	t14 = t14a.AddSaturated(t13a)
	t15a = t15.AddSaturated(t12)

	t10a = itx16Rot(t13, t10, p2896, m2896, s12)
	t13a = itx16Rot(t13, t10, p2896, p2896, s12)
	t11 = itx16Rot(t12a, t11a, p2896, m2896, s12)
	t12 = itx16Rot(t12a, t11a, p2896, p2896, s12)

	if l1.ReduceMax() > itx16ColumnL1Bound {
		return false
	}
	st := func(k int, v archsimd.Int16x8) {
		v.HiToLo().ExtendLo4ToInt32().StoreArray((*[4]int32)(buf[k*stride+4:]))
		v.ExtendLo4ToInt32().StoreArray((*[4]int32)(buf[k*stride:]))
	}
	st(0, e0.AddSaturated(t15a))
	st(15, e0.SubSaturated(t15a))
	st(1, e1.AddSaturated(t14))
	st(14, e1.SubSaturated(t14))
	st(2, e2.AddSaturated(t13a))
	st(13, e2.SubSaturated(t13a))
	st(3, e3.AddSaturated(t12))
	st(12, e3.SubSaturated(t12))
	st(4, e4.AddSaturated(t11))
	st(11, e4.SubSaturated(t11))
	st(5, e5.AddSaturated(t10a))
	st(10, e5.SubSaturated(t10a))
	st(6, e6.AddSaturated(t9))
	st(9, e6.SubSaturated(t9))
	st(7, e7.AddSaturated(t8a))
	st(8, e7.SubSaturated(t8a))
	return true
}
