// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package transform

import "simd/archsimd"

// int16 8-wide inverse DCT32 column kernel for the eob-bounded blocks whose
// input rows 8..31 are zero (the common 32-high shape), in the form of the
// DCT16 kernels (itx16_dct16_gosimd_arm64.go): the even half is the DCT16
// Low4 network on rows 0, 2, 4 and 6 kept in registers, the odd half follows
// inverseDCT32 in dct.go stage for stage from rows 1, 3, 5 and 7 (the
// first-stage rotations are single products, the second-stage butterflies
// pass the rotation outputs through), and every output is a saturating sum
// of an even and an odd value, widened on store. Nothing takes a round trip
// through the buffer between the halves.
//
// Exactness: itx16_gain_test.go's DCT32 float model bounds every rotation
// output by 0.9989 times the column's L1 norm plus 9.15 of rounding slack
// (DCT16: 0.9951 and 4.03), so the per-column guard itx16ColumnL1Bound over
// the eight nonzero rows certifies that every narrowed value fits int16. A
// failing column group is left untouched and reports false.

// itx32K holds the broadcast int16 rotation constants of the odd half that
// the DCT16 table lacks.
var itx32K = [...][8]int16{
	{201, 201, 201, 201, 201, 201, 201, 201},
	{4091, 4091, 4091, 4091, 4091, 4091, 4091, 4091},
	{3857, 3857, 3857, 3857, 3857, 3857, 3857, 3857},
	{-1380, -1380, -1380, -1380, -1380, -1380, -1380, -1380},
	{995, 995, 995, 995, 995, 995, 995, 995},
	{3973, 3973, 3973, 3973, 3973, 3973, 3973, 3973},
	{4052, 4052, 4052, 4052, 4052, 4052, 4052, 4052},
	{-601, -601, -601, -601, -601, -601, -601, -601},
	{-799, -799, -799, -799, -799, -799, -799, -799},
	{-1703, -1703, -1703, -1703, -1703, -1703, -1703, -1703},
}

const (
	itx32Kp201 = iota
	itx32Kp4091
	itx32Kp3857
	itx32Km1380
	itx32Kp995
	itx32Kp3973
	itx32Kp4052
	itx32Km601
	itx32Km799
	itx32Km1703
)

// inverseDCT32Col8RawLow8SIMD16 applies inverseDCT32 to the eight int32
// columns buf[k*stride+0..7], whose rows 8..31 are zero and whose values are
// already in the int16 range. It reports false, leaving the columns
// untouched, when a column's L1 norm exceeds the guard.
func inverseDCT32Col8RawLow8SIMD16(buf []int32, stride int) bool {
	_ = buf[31*stride+7]
	var l1 archsimd.Uint16x8
	ld := func(k int) archsimd.Int16x8 {
		x := itx16LoadNarrow(buf[k*stride:])
		l1 = l1.AddSaturated(x.Abs().ToBits())
		return x
	}
	k := func(i int) archsimd.Int16x8 { return archsimd.LoadInt16x8Array(&itx16K[i]) }
	k32 := func(i int) archsimd.Int16x8 { return archsimd.LoadInt16x8Array(&itx32K[i]) }
	s12 := archsimd.LoadInt32x4Array(&itx16Scale12)
	s11 := archsimd.LoadInt32x4Array(&itx16Scale11)
	p2896, m2896 := k(itx16Kp2896), k(itx16Km2896)
	p1567, m1567, p3784, m3784 := k(itx16Kp1567), k(itx16Km1567), k(itx16Kp3784), k(itx16Km3784)

	// Even half: inverseDCT16 on rows 0, 2, 4, 6 (inverseDCT16Col8RawLow4SIMD16).
	in0, in2, in4, in6 := ld(0), ld(2), ld(4), ld(6)
	t0 := itx16Rot1(in0, p2896, s12)
	t4a := itx16Rot1(in4, k(itx16Kp799), s12)
	t7a := itx16Rot1(in4, k(itx16Kp4017), s12)
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
	t8a := itx16Rot1(in2, k(itx16Kp401), s12)
	t15a := itx16Rot1(in2, k(itx16Kp4076), s12)
	t11a := itx16Rot1(in6, k(itx16Km1189), s12)
	t12a := itx16Rot1(in6, k(itx16Kp3920), s12)
	t9a := itx16Rot(t15a, t8a, p1567, m3784, s12)
	t14a := itx16Rot(t15a, t8a, p3784, p1567, s12)
	t10a := itx16Rot(t12a, t11a, m3784, m1567, s12)
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
	E0, E15 := e0.AddSaturated(t15), e0.SubSaturated(t15)
	E1, E14 := e1.AddSaturated(t14), e1.SubSaturated(t14)
	E2, E13 := e2.AddSaturated(t13a), e2.SubSaturated(t13a)
	E3, E12 := e3.AddSaturated(t12a), e3.SubSaturated(t12a)
	E4, E11 := e4.AddSaturated(t11a), e4.SubSaturated(t11a)
	E5, E10 := e5.AddSaturated(t10a), e5.SubSaturated(t10a)
	E6, E9 := e6.AddSaturated(t9), e6.SubSaturated(t9)
	E7, E8 := e7.AddSaturated(t8), e7.SubSaturated(t8)

	// Odd half, first stage: rows 9..31 are zero, so t17a, t18a, t21a, t22a,
	// t25a, t26a, t29a and t30a are zero and the second-stage butterflies
	// pass t16a, t19a, t20a, t23a, t24a, t27a, t28a and t31a through.
	in1, in3, in5, in7 := ld(1), ld(3), ld(5), ld(7)
	t16 := itx16Rot1(in1, k32(itx32Kp201), s12)
	t31 := itx16Rot1(in1, k32(itx32Kp4091), s12)
	t19 := itx16Rot1(in7, k32(itx32Km1380), s12)
	t28 := itx16Rot1(in7, k32(itx32Kp3857), s12)
	t20 := itx16Rot1(in5, k32(itx32Kp995), s12)
	t27 := itx16Rot1(in5, k32(itx32Kp3973), s12)
	t23 := itx16Rot1(in3, k32(itx32Km601), s12)
	t24 := itx16Rot1(in3, k32(itx32Kp4052), s12)
	t17, t18, t21, t22, t25, t26, t29, t30 := t16, t19, t20, t23, t24, t27, t28, t31

	if l1.ReduceMax() > itx16ColumnL1Bound {
		return false
	}

	p799, m799, p4017, m4017 := k(itx16Kp799), k32(itx32Km799), k(itx16Kp4017), k(itx16Km4017)
	t17a := itx16Rot(t30, t17, p799, m4017, s12)
	t30a := itx16Rot(t30, t17, p4017, p799, s12)
	t18a := itx16Rot(t29, t18, m4017, m799, s12)
	t29a := itx16Rot(t29, t18, p799, m4017, s12)
	p1703, m1703, p1138, m1138 := k(itx16Kp1703), k32(itx32Km1703), k(itx16Kp1138), k(itx16Km1138)
	t21a := itx16Rot(t26, t21, p1703, m1138, s11)
	t26a := itx16Rot(t26, t21, p1138, p1703, s11)
	t22a := itx16Rot(t25, t22, m1138, m1703, s11)
	t25a := itx16Rot(t25, t22, p1703, m1138, s11)

	t16a := t16.AddSaturated(t19)
	t17 = t17a.AddSaturated(t18a)
	t18 = t17a.SubSaturated(t18a)
	t19a := t16.SubSaturated(t19)
	t20a := t23.SubSaturated(t20)
	t21 = t22a.SubSaturated(t21a)
	t22 = t22a.AddSaturated(t21a)
	t23a := t23.AddSaturated(t20)
	t24a := t24.AddSaturated(t27)
	t25 = t25a.AddSaturated(t26a)
	t26 = t25a.SubSaturated(t26a)
	t27a := t24.SubSaturated(t27)
	t28a := t31.SubSaturated(t28)
	t29 = t30a.SubSaturated(t29a)
	t30 = t30a.AddSaturated(t29a)
	t31a := t31.AddSaturated(t28)

	t18a = itx16Rot(t29, t18, p1567, m3784, s12)
	t29a = itx16Rot(t29, t18, p3784, p1567, s12)
	t19 = itx16Rot(t28a, t19a, p1567, m3784, s12)
	t28 = itx16Rot(t28a, t19a, p3784, p1567, s12)
	t20 = itx16Rot(t27a, t20a, m3784, m1567, s12)
	t27 = itx16Rot(t27a, t20a, p1567, m3784, s12)
	t21a = itx16Rot(t26, t21, m3784, m1567, s12)
	t26a = itx16Rot(t26, t21, p1567, m3784, s12)

	t16 = t16a.AddSaturated(t23a)
	t17a = t17.AddSaturated(t22)
	t18 = t18a.AddSaturated(t21a)
	t19a = t19.AddSaturated(t20)
	t20a = t19.SubSaturated(t20)
	t21 = t18a.SubSaturated(t21a)
	t22a = t17.SubSaturated(t22)
	t23 = t16a.SubSaturated(t23a)
	t24 = t31a.SubSaturated(t24a)
	t25a = t30.SubSaturated(t25)
	t26 = t29a.SubSaturated(t26a)
	t27a = t28.SubSaturated(t27)
	t28a = t28.AddSaturated(t27)
	t29 = t29a.AddSaturated(t26a)
	t30a = t30.AddSaturated(t25)
	t31 = t31a.AddSaturated(t24a)

	t20 = itx16Rot(t27a, t20a, p2896, m2896, s12)
	t27 = itx16Rot(t27a, t20a, p2896, p2896, s12)
	t21a = itx16Rot(t26, t21, p2896, m2896, s12)
	t26a = itx16Rot(t26, t21, p2896, p2896, s12)
	t22 = itx16Rot(t25a, t22a, p2896, m2896, s12)
	t25 = itx16Rot(t25a, t22a, p2896, p2896, s12)
	t23a = itx16Rot(t24, t23, p2896, m2896, s12)
	t24a = itx16Rot(t24, t23, p2896, p2896, s12)

	st := func(row int, v archsimd.Int16x8) { itx16StoreWiden(buf[row*stride:], v) }
	st(0, E0.AddSaturated(t31))
	st(31, E0.SubSaturated(t31))
	st(1, E1.AddSaturated(t30a))
	st(30, E1.SubSaturated(t30a))
	st(2, E2.AddSaturated(t29))
	st(29, E2.SubSaturated(t29))
	st(3, E3.AddSaturated(t28a))
	st(28, E3.SubSaturated(t28a))
	st(4, E4.AddSaturated(t27))
	st(27, E4.SubSaturated(t27))
	st(5, E5.AddSaturated(t26a))
	st(26, E5.SubSaturated(t26a))
	st(6, E6.AddSaturated(t25))
	st(25, E6.SubSaturated(t25))
	st(7, E7.AddSaturated(t24a))
	st(24, E7.SubSaturated(t24a))
	st(8, E8.AddSaturated(t23a))
	st(23, E8.SubSaturated(t23a))
	st(9, E9.AddSaturated(t22))
	st(22, E9.SubSaturated(t22))
	st(10, E10.AddSaturated(t21a))
	st(21, E10.SubSaturated(t21a))
	st(11, E11.AddSaturated(t20))
	st(20, E11.SubSaturated(t20))
	st(12, E12.AddSaturated(t19a))
	st(19, E12.SubSaturated(t19a))
	st(13, E13.AddSaturated(t18))
	st(18, E13.SubSaturated(t18))
	st(14, E14.AddSaturated(t17a))
	st(17, E14.SubSaturated(t17a))
	st(15, E15.AddSaturated(t16))
	st(16, E15.SubSaturated(t16))
	return true
}

// itx16RawColumnPass32Low8SIMD runs the DCT32 column pass over the int32 raw
// scratch of a width x 32 block whose rows 8..31 are zero, eight columns at a
// time with the int16 kernel; a column group that fails its guard is
// untouched and takes the int32 low-row kernels.
func itx16RawColumnPass32Low8SIMD(scratch []int32, width int, min, max int32) {
	for c := 0; c < width; c += 8 {
		if inverseDCT32Col8RawLow8SIMD16(scratch[c:], width) {
			continue
		}
		if inverseDCT32Low8Col4Impl != nil {
			inverseDCT32Low8Col4Impl(scratch[c:], width, min, max)
			inverseDCT32Low8Col4Impl(scratch[c+4:], width, min, max)
			continue
		}
		inverse1DCol4(scratch[c:], width, dct32Size, tx1DDCT, min, max)
		inverse1DCol4(scratch[c+4:], width, dct32Size, tx1DDCT, min, max)
	}
}
