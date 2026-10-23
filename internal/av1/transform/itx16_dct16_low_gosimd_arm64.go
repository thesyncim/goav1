// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package transform

import "simd/archsimd"

// Sparse-input variants of inverseDCT16Col8RawSIMD16 (dav1d's eob-bounded
// itx): when only the first 4 or 8 input rows can be nonzero, the rotations
// whose other operand is zero are single products, and the zero rows are
// neither loaded nor guarded. The dense stages after the first butterfly are
// unchanged, so the exactness contract (every rotation output fits int16,
// certified by the per-column L1 guard over the nonzero rows) is the full
// kernel's.

// itx16Rot1 is Round2(a*ca, 12) (or 11 with scale 32) on eight lanes.
func itx16Rot1(a, ca archsimd.Int16x8, scale archsimd.Int32x4) archsimd.Int16x8 {
	bias := archsimd.LoadInt32x4Array(&itx16Bias)
	lo := a.MulWidenLo(ca).MulAdd(scale, bias)
	bias = archsimd.LoadInt32x4Array(&itx16Bias)
	hi := a.HiToLo().MulWidenLo(ca.HiToLo()).MulAdd(scale, bias)
	return lo.ToBits().ReshapeToUint16s().ConcatOdd(hi.ToBits().ReshapeToUint16s()).BitsToInt16()
}

// itx16LoadNarrow loads eight int32 values in the int16 range as Int16x8.
func itx16LoadNarrow(p []int32) archsimd.Int16x8 {
	lo := archsimd.LoadInt32x4Array((*[4]int32)(p))
	hi := archsimd.LoadInt32x4Array((*[4]int32)(p[4:]))
	v := lo.SaturateToInt16().ToBits().ReshapeToUint64s().BitsToFloat64()
	v = v.SetElem(1, hi.SaturateToInt16().ToBits().ReshapeToUint64s().BitsToFloat64().GetElem(0))
	return v.ToBits().ReshapeToUint16s().BitsToInt16()
}

// itx16StoreWiden stores an Int16x8 as eight int32 values.
func itx16StoreWiden(p []int32, v archsimd.Int16x8) {
	v.HiToLo().ExtendLo4ToInt32().StoreArray((*[4]int32)(p[4:]))
	v.ExtendLo4ToInt32().StoreArray((*[4]int32)(p))
}

// inverseDCT16Col8RawLow4SIMD16 is inverseDCT16Col8RawSIMD16 for inputs whose
// rows 4..15 are zero.
func inverseDCT16Col8RawLow4SIMD16(buf []int32, stride int) bool {
	_ = buf[15*stride+7]
	k := func(i int) archsimd.Int16x8 { return archsimd.LoadInt16x8Array(&itx16K[i]) }
	s12 := archsimd.LoadInt32x4Array(&itx16Scale12)
	p2896, m2896 := k(itx16Kp2896), k(itx16Km2896)
	p1567, p3784, m3784 := k(itx16Kp1567), k(itx16Kp3784), k(itx16Km3784)

	in0 := itx16LoadNarrow(buf)
	in1 := itx16LoadNarrow(buf[stride:])
	in2 := itx16LoadNarrow(buf[2*stride:])
	in3 := itx16LoadNarrow(buf[3*stride:])
	l1 := in0.Abs().ToBits().AddSaturated(in1.Abs().ToBits()).AddSaturated(in2.Abs().ToBits()).AddSaturated(in3.Abs().ToBits())

	// Even half: inverseDCT4 on (in0, 0, 0, 0) gives t0 = t1, t2 = t3 = 0.
	t0 := itx16Rot1(in0, p2896, s12)
	t4a := itx16Rot1(in2, k(itx16Kp799), s12)
	t7a := itx16Rot1(in2, k(itx16Kp4017), s12)
	// t5a = t6a = 0: t4 = t4a, t5a' = t4a, t7 = t7a, t6a' = t7a.
	t5 := itx16Rot(t7a, t4a, p2896, m2896, s12)
	t6 := itx16Rot(t7a, t4a, p2896, p2896, s12)
	e0 := t0.AddSaturated(t7a)
	e1 := t0.AddSaturated(t6)
	e2 := t0.AddSaturated(t5)
	e3 := t0.AddSaturated(t4a)
	e4 := t0.SubSaturated(t4a)
	e5 := t0.SubSaturated(t5)
	e6 := t0.SubSaturated(t6)
	e7 := t0.SubSaturated(t7a)

	// Odd half: t9a = t14a = t10a = t13a = 0.
	t8a := itx16Rot1(in1, k(itx16Kp401), s12)
	t15a := itx16Rot1(in1, k(itx16Kp4076), s12)
	t11a := itx16Rot1(in3, k(itx16Km1189), s12)
	t12a := itx16Rot1(in3, k(itx16Kp3920), s12)
	// t8 = t9 = t8a; t10 = t11 = t11a; t12 = t13 = t12a; t14 = t15 = t15a.
	t9a := itx16Rot(t15a, t8a, p1567, m3784, s12)
	t14a := itx16Rot(t15a, t8a, p3784, p1567, s12)
	t10a := itx16Rot(t12a, t11a, m3784, k(itx16Km1567), s12)
	t13a := itx16Rot(t12a, t11a, p1567, m3784, s12)

	t8 := t8a.AddSaturated(t11a)
	t9 := t9a.AddSaturated(t10a)
	t10 := t9a.SubSaturated(t10a)
	t11 := t8a.SubSaturated(t11a)
	t12 := t15a.SubSaturated(t12a)
	t13 := t14a.SubSaturated(t13a)
	t14 := t14a.AddSaturated(t13a)
	t15 := t15a.AddSaturated(t12a)

	t10a = itx16Rot(t13, t10, p2896, m2896, s12)
	t13a = itx16Rot(t13, t10, p2896, p2896, s12)
	t11a = itx16Rot(t12, t11, p2896, m2896, s12)
	t12a = itx16Rot(t12, t11, p2896, p2896, s12)

	if l1.ReduceMax() > itx16ColumnL1Bound {
		return false
	}
	itx16StoreWiden(buf, e0.AddSaturated(t15))
	itx16StoreWiden(buf[15*stride:], e0.SubSaturated(t15))
	itx16StoreWiden(buf[stride:], e1.AddSaturated(t14))
	itx16StoreWiden(buf[14*stride:], e1.SubSaturated(t14))
	itx16StoreWiden(buf[2*stride:], e2.AddSaturated(t13a))
	itx16StoreWiden(buf[13*stride:], e2.SubSaturated(t13a))
	itx16StoreWiden(buf[3*stride:], e3.AddSaturated(t12a))
	itx16StoreWiden(buf[12*stride:], e3.SubSaturated(t12a))
	itx16StoreWiden(buf[4*stride:], e4.AddSaturated(t11a))
	itx16StoreWiden(buf[11*stride:], e4.SubSaturated(t11a))
	itx16StoreWiden(buf[5*stride:], e5.AddSaturated(t10a))
	itx16StoreWiden(buf[10*stride:], e5.SubSaturated(t10a))
	itx16StoreWiden(buf[6*stride:], e6.AddSaturated(t9))
	itx16StoreWiden(buf[9*stride:], e6.SubSaturated(t9))
	itx16StoreWiden(buf[7*stride:], e7.AddSaturated(t8))
	itx16StoreWiden(buf[8*stride:], e7.SubSaturated(t8))
	return true
}

// inverseDCT16Col8RawLow8SIMD16 is inverseDCT16Col8RawSIMD16 for inputs whose
// rows 8..15 are zero.
func inverseDCT16Col8RawLow8SIMD16(buf []int32, stride int) bool {
	_ = buf[15*stride+7]
	k := func(i int) archsimd.Int16x8 { return archsimd.LoadInt16x8Array(&itx16K[i]) }
	s12 := archsimd.LoadInt32x4Array(&itx16Scale12)
	s11 := archsimd.LoadInt32x4Array(&itx16Scale11)
	p2896, m2896 := k(itx16Kp2896), k(itx16Km2896)
	p1567, p3784, m3784 := k(itx16Kp1567), k(itx16Kp3784), k(itx16Km3784)

	in0 := itx16LoadNarrow(buf)
	in1 := itx16LoadNarrow(buf[stride:])
	in2 := itx16LoadNarrow(buf[2*stride:])
	in3 := itx16LoadNarrow(buf[3*stride:])
	in4 := itx16LoadNarrow(buf[4*stride:])
	in5 := itx16LoadNarrow(buf[5*stride:])
	in6 := itx16LoadNarrow(buf[6*stride:])
	in7 := itx16LoadNarrow(buf[7*stride:])
	l1 := in0.Abs().ToBits().AddSaturated(in1.Abs().ToBits()).AddSaturated(in2.Abs().ToBits()).AddSaturated(in3.Abs().ToBits())
	l1 = l1.AddSaturated(in4.Abs().ToBits()).AddSaturated(in5.Abs().ToBits()).AddSaturated(in6.Abs().ToBits()).AddSaturated(in7.Abs().ToBits())

	// Even half: inverseDCT8 on (in0, in2, in4, in6, 0, 0, 0, 0).
	t0 := itx16Rot1(in0, p2896, s12)
	t2 := itx16Rot1(in4, p1567, s12)
	t3 := itx16Rot1(in4, p3784, s12)
	d0 := t0.AddSaturated(t3)
	d1 := t0.AddSaturated(t2)
	d2 := t0.SubSaturated(t2)
	d3 := t0.SubSaturated(t3)
	t4a := itx16Rot1(in2, k(itx16Kp799), s12)
	t5a := itx16Rot1(in6, k(itx16Km1138), s11)
	t6a := itx16Rot1(in6, k(itx16Kp1703), s11)
	t7a := itx16Rot1(in2, k(itx16Kp4017), s12)
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

	// Odd half with in9 = in11 = in13 = in15 = 0.
	t8a := itx16Rot1(in1, k(itx16Kp401), s12)
	t15a := itx16Rot1(in1, k(itx16Kp4076), s12)
	t9a := itx16Rot1(in7, k(itx16Km1299), s11)
	t14a := itx16Rot1(in7, k(itx16Kp1583), s11)
	t10a := itx16Rot1(in5, k(itx16Kp1931), s12)
	t13a := itx16Rot1(in5, k(itx16Kp3612), s12)
	t11a := itx16Rot1(in3, k(itx16Km1189), s12)
	t12a := itx16Rot1(in3, k(itx16Kp3920), s12)

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
	itx16StoreWiden(buf, e0.AddSaturated(t15a))
	itx16StoreWiden(buf[15*stride:], e0.SubSaturated(t15a))
	itx16StoreWiden(buf[stride:], e1.AddSaturated(t14))
	itx16StoreWiden(buf[14*stride:], e1.SubSaturated(t14))
	itx16StoreWiden(buf[2*stride:], e2.AddSaturated(t13a))
	itx16StoreWiden(buf[13*stride:], e2.SubSaturated(t13a))
	itx16StoreWiden(buf[3*stride:], e3.AddSaturated(t12))
	itx16StoreWiden(buf[12*stride:], e3.SubSaturated(t12))
	itx16StoreWiden(buf[4*stride:], e4.AddSaturated(t11))
	itx16StoreWiden(buf[11*stride:], e4.SubSaturated(t11))
	itx16StoreWiden(buf[5*stride:], e5.AddSaturated(t10a))
	itx16StoreWiden(buf[10*stride:], e5.SubSaturated(t10a))
	itx16StoreWiden(buf[6*stride:], e6.AddSaturated(t9))
	itx16StoreWiden(buf[9*stride:], e6.SubSaturated(t9))
	itx16StoreWiden(buf[7*stride:], e7.AddSaturated(t8a))
	itx16StoreWiden(buf[8*stride:], e7.SubSaturated(t8a))
	return true
}
