// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package transform

import "simd/archsimd"

// itx32Low4Net is inverseDCT32 on eight lanes for inputs whose indices 4..31
// are zero: in[k] is coefficient k, out[r] is output r. It is itx32Low8Net
// with the zero operands folded (a zero input makes every rotation and sum
// that touches only zeros zero, and x +- 0 is x exactly), so it produces
// itx32Low8Net's values on those inputs; TestItx32Low4NetMatchesLow8 pins that
// over random and saturating lanes. The even half is the DCT16 of coefficients
// 0 and 2, whose first-stage sums are all t0.
func itx32Low4Net(in *[4]archsimd.Int16x8, out *[32]archsimd.Int16x8) {
	k := func(i int) archsimd.Int16x8 { return archsimd.LoadInt16x8Array(&itx16K[i]) }
	k32 := func(i int) archsimd.Int16x8 { return archsimd.LoadInt16x8Array(&itx32K[i]) }
	s12 := archsimd.LoadInt32x4Array(&itx16Scale12)
	s11 := archsimd.LoadInt32x4Array(&itx16Scale11)
	p2896, m2896 := k(itx16Kp2896), k(itx16Km2896)
	p1567, m1567, p3784, m3784 := k(itx16Kp1567), k(itx16Km1567), k(itx16Kp3784), k(itx16Km3784)

	// Even half.
	t0 := itx16Rot1(in[0], p2896, s12)
	t8 := itx16Rot1(in[2], k(itx16Kp401), s12)
	t15 := itx16Rot1(in[2], k(itx16Kp4076), s12)
	t9 := itx16Rot(t15, t8, p1567, m3784, s12)
	t14 := itx16Rot(t15, t8, p3784, p1567, s12)
	t10a := itx16Rot(t14, t9, p2896, m2896, s12)
	t13a := itx16Rot(t14, t9, p2896, p2896, s12)
	t11a := itx16Rot(t15, t8, p2896, m2896, s12)
	t12a := itx16Rot(t15, t8, p2896, p2896, s12)
	E0, E15 := t0.AddSaturated(t15), t0.SubSaturated(t15)
	E1, E14 := t0.AddSaturated(t14), t0.SubSaturated(t14)
	E2, E13 := t0.AddSaturated(t13a), t0.SubSaturated(t13a)
	E3, E12 := t0.AddSaturated(t12a), t0.SubSaturated(t12a)
	E4, E11 := t0.AddSaturated(t11a), t0.SubSaturated(t11a)
	E5, E10 := t0.AddSaturated(t10a), t0.SubSaturated(t10a)
	E6, E9 := t0.AddSaturated(t9), t0.SubSaturated(t9)
	E7, E8 := t0.AddSaturated(t8), t0.SubSaturated(t8)

	// Odd half from inputs 1 and 3: the first-stage rotations are single
	// products and the rotations of the zero terms vanish.
	t16 := itx16Rot1(in[1], k32(itx32Kp201), s12)
	t31 := itx16Rot1(in[1], k32(itx32Kp4091), s12)
	t23 := itx16Rot1(in[3], k32(itx32Km601), s12)
	t24 := itx16Rot1(in[3], k32(itx32Kp4052), s12)

	p799, m4017, p4017 := k(itx16Kp799), k(itx16Km4017), k(itx16Kp4017)
	t17a := itx16Rot(t31, t16, p799, m4017, s12)
	t30a := itx16Rot(t31, t16, p4017, p799, s12)
	p1703, m1703, m1138 := k(itx16Kp1703), k32(itx32Km1703), k(itx16Km1138)
	t22a := itx16Rot(t24, t23, m1138, m1703, s11)
	t25a := itx16Rot(t24, t23, p1703, m1138, s11)

	t18a := itx16Rot(t30a, t17a, p1567, m3784, s12)
	t29a := itx16Rot(t30a, t17a, p3784, p1567, s12)
	t19 := itx16Rot(t31, t16, p1567, m3784, s12)
	t28 := itx16Rot(t31, t16, p3784, p1567, s12)
	t20 := itx16Rot(t24, t23, m3784, m1567, s12)
	t27 := itx16Rot(t24, t23, p1567, m3784, s12)
	t21a := itx16Rot(t25a, t22a, m3784, m1567, s12)
	t26a := itx16Rot(t25a, t22a, p1567, m3784, s12)

	c16 := t16.AddSaturated(t23)
	c17 := t17a.AddSaturated(t22a)
	c18 := t18a.AddSaturated(t21a)
	c19 := t19.AddSaturated(t20)
	c20 := t19.SubSaturated(t20)
	c21 := t18a.SubSaturated(t21a)
	c22 := t17a.SubSaturated(t22a)
	c23 := t16.SubSaturated(t23)
	c24 := t31.SubSaturated(t24)
	c25 := t30a.SubSaturated(t25a)
	c26 := t29a.SubSaturated(t26a)
	c27 := t28.SubSaturated(t27)
	c28 := t28.AddSaturated(t27)
	c29 := t29a.AddSaturated(t26a)
	c30 := t30a.AddSaturated(t25a)
	c31 := t31.AddSaturated(t24)

	d20 := itx16Rot(c27, c20, p2896, m2896, s12)
	d27 := itx16Rot(c27, c20, p2896, p2896, s12)
	d21 := itx16Rot(c26, c21, p2896, m2896, s12)
	d26 := itx16Rot(c26, c21, p2896, p2896, s12)
	d22 := itx16Rot(c25, c22, p2896, m2896, s12)
	d25 := itx16Rot(c25, c22, p2896, p2896, s12)
	d23 := itx16Rot(c24, c23, p2896, m2896, s12)
	d24 := itx16Rot(c24, c23, p2896, p2896, s12)

	out[0], out[31] = E0.AddSaturated(c31), E0.SubSaturated(c31)
	out[1], out[30] = E1.AddSaturated(c30), E1.SubSaturated(c30)
	out[2], out[29] = E2.AddSaturated(c29), E2.SubSaturated(c29)
	out[3], out[28] = E3.AddSaturated(c28), E3.SubSaturated(c28)
	out[4], out[27] = E4.AddSaturated(d27), E4.SubSaturated(d27)
	out[5], out[26] = E5.AddSaturated(d26), E5.SubSaturated(d26)
	out[6], out[25] = E6.AddSaturated(d25), E6.SubSaturated(d25)
	out[7], out[24] = E7.AddSaturated(d24), E7.SubSaturated(d24)
	out[8], out[23] = E8.AddSaturated(d23), E8.SubSaturated(d23)
	out[9], out[22] = E9.AddSaturated(d22), E9.SubSaturated(d22)
	out[10], out[21] = E10.AddSaturated(d21), E10.SubSaturated(d21)
	out[11], out[20] = E11.AddSaturated(d20), E11.SubSaturated(d20)
	out[12], out[19] = E12.AddSaturated(c19), E12.SubSaturated(c19)
	out[13], out[18] = E13.AddSaturated(c18), E13.SubSaturated(c18)
	out[14], out[17] = E14.AddSaturated(c17), E14.SubSaturated(c17)
	out[15], out[16] = E15.AddSaturated(c16), E15.SubSaturated(c16)
}
