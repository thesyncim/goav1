// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && (arm64 || amd64) && !purego

package transform

import (
	"simd/archsimd"
)

func inverseDCT32Lanes4(p []int32, stride int, min, max int32) bool {
	switch {
	case min >= -itxNarrowBound && max < itxNarrowBound:
		inverseDCT32Lanes4Narrow(p, stride, min, max)
	case min >= -itxWideBound && max < itxWideBound:
		inverseDCT32Lanes4Wide(p, stride, min, max)
	default:
		return false
	}
	return true
}

func inverseDCT32Lanes4ColSIMD(buffer []int32, stride int, min, max int32) {
	if stride < 4 || len(buffer) < 4 || stride > (len(buffer)-4)/(dct32Size-1) ||
		!inverseDCT32Lanes4(buffer, stride, min, max) {
		inverseDCT32Col4PureGo(buffer, stride, min, max)
	}
}

func inverseDCT32Lanes4RowSIMD(row0, row1, row2, row3 []int32, min, max int32) {
	if len(row0) < dct32Size || len(row1) < dct32Size || len(row2) < dct32Size || len(row3) < dct32Size ||
		min < -itxWideBound || max >= itxWideBound {
		inverseDCT32Row4PureGo(row0, row1, row2, row3, min, max)
		return
	}
	var staging [dct32Size * 4]int32
	for index := 0; index < dct32Size; index += 4 {
		w := (*[16]int32)(staging[4*index : 4*index+16])
		value0, value1, value2, value3 := itxTranspose4(
			archsimd.LoadInt32x4Array((*[4]int32)(row0[index:])),
			archsimd.LoadInt32x4Array((*[4]int32)(row1[index:])),
			archsimd.LoadInt32x4Array((*[4]int32)(row2[index:])),
			archsimd.LoadInt32x4Array((*[4]int32)(row3[index:])))
		value0.StoreArray((*[4]int32)(w[0:4]))
		value1.StoreArray((*[4]int32)(w[4:8]))
		value2.StoreArray((*[4]int32)(w[8:12]))
		value3.StoreArray((*[4]int32)(w[12:16]))
	}
	inverseDCT32Lanes4(staging[:], 4, min, max)
	for index := 0; index < dct32Size; index += 4 {
		w := (*[16]int32)(staging[4*index : 4*index+16])
		value0, value1, value2, value3 := itxTranspose4(
			archsimd.LoadInt32x4Array((*[4]int32)(w[0:4])),
			archsimd.LoadInt32x4Array((*[4]int32)(w[4:8])),
			archsimd.LoadInt32x4Array((*[4]int32)(w[8:12])),
			archsimd.LoadInt32x4Array((*[4]int32)(w[12:16])))
		value0.StoreArray((*[4]int32)(row0[index:]))
		value1.StoreArray((*[4]int32)(row1[index:]))
		value2.StoreArray((*[4]int32)(row2[index:]))
		value3.StoreArray((*[4]int32)(row3[index:]))
	}
}

func inverseDCT32Lanes4Narrow(p []int32, stride int, min, max int32) {
	k := itxCoefsPtr
	mn, mx := archsimd.BroadcastInt32x4(min), archsimd.BroadcastInt32x4(max)
	r12, n12 := itxRoundBias(12), itxShiftAmount(12)
	ld := func(i int) archsimd.Int32x4 {
		o := i * stride
		return archsimd.LoadInt32x4Array((*[4]int32)(p[o : o+4]))
	}
	st := func(i int, v archsimd.Int32x4) {
		o := i * stride
		v.StoreArray((*[4]int32)(p[o : o+4]))
	}
	clamp := func(v archsimd.Int32x4) archsimd.Int32x4 { return v.Max(mn).Min(mx) }
	mac := func(acc, x archsimd.Int32x4, c int) archsimd.Int32x4 {
		return itxMulAcc(acc, x, archsimd.LoadInt32x4Array(&k[c]))
	}
	shr12 := func(v archsimd.Int32x4) archsimd.Int32x4 { return itxShr12(v, n12) }

	in4 := ld(2)
	in60 := ld(30)
	t8_4 := shr12(mac(mac(r12, in4, kc401), in60, kcN4076))
	in36 := ld(18)
	in28 := ld(14)
	t9_4 := shr12(mac(mac(r12, in36, kc3166), in28, kcN2598))
	in20 := ld(10)
	in44 := ld(22)
	t10_4 := shr12(mac(mac(r12, in20, kc1931), in44, kcN3612))
	in52 := ld(26)
	in12 := ld(6)
	t11_4 := shr12(mac(mac(r12, in52, kc3920), in12, kcN1189))
	t12_4 := shr12(mac(mac(r12, in52, kc1189), in12, kc3920))
	t13_4 := shr12(mac(mac(r12, in20, kc3612), in44, kc1931))
	t14_4 := shr12(mac(mac(r12, in36, kc2598), in28, kc3166))
	t15_4 := shr12(mac(mac(r12, in4, kc4076), in60, kc401))
	in8 := ld(4)
	in56 := ld(28)
	t4_5 := shr12(mac(mac(r12, in8, kc799), in56, kcN4017))
	in40 := ld(20)
	in24 := ld(12)
	t5_5 := shr12(mac(mac(r12, in40, kc3406), in24, kcN2276))
	t6_5 := shr12(mac(mac(r12, in40, kc2276), in24, kc3406))
	t7_5 := shr12(mac(mac(r12, in8, kc4017), in56, kc799))
	t8_5 := clamp(t8_4.Add(t9_4))
	t9_5 := clamp(t8_4.Sub(t9_4))
	t10_5 := clamp(t11_4.Sub(t10_4))
	t11_5 := clamp(t10_4.Add(t11_4))
	t12_5 := clamp(t12_4.Add(t13_4))
	t13_5 := clamp(t12_4.Sub(t13_4))
	t14_5 := clamp(t15_4.Sub(t14_4))
	t15_5 := clamp(t14_4.Add(t15_4))
	in0 := ld(0)
	in32 := ld(16)
	t0_6 := shr12(mac(r12, in0.Add(in32), kc2896))
	t1_6 := shr12(mac(r12, in0.Sub(in32), kc2896))
	in16 := ld(8)
	in48 := ld(24)
	t2_6 := shr12(mac(mac(r12, in16, kc1567), in48, kcN3784))
	t3_6 := shr12(mac(mac(r12, in16, kc3784), in48, kc1567))
	t4_6 := clamp(t4_5.Add(t5_5))
	t5_6 := clamp(t4_5.Sub(t5_5))
	t6_6 := clamp(t7_5.Sub(t6_5))
	t7_6 := clamp(t6_5.Add(t7_5))
	t9_6 := shr12(mac(mac(r12, t9_5, kcN3784), t14_5, kc1567))
	t10_6 := shr12(mac(mac(r12, t10_5, kcN1567), t13_5, kcN3784))
	t13_6 := shr12(mac(mac(r12, t10_5, kcN3784), t13_5, kc1567))
	t14_6 := shr12(mac(mac(r12, t9_5, kc1567), t14_5, kc3784))
	t0_7 := clamp(t0_6.Add(t3_6))
	t1_7 := clamp(t1_6.Add(t2_6))
	t2_7 := clamp(t1_6.Sub(t2_6))
	t3_7 := clamp(t0_6.Sub(t3_6))
	t5_7 := shr12(mac(r12, t6_6.Sub(t5_6), kc2896))
	t6_7 := shr12(mac(r12, t5_6.Add(t6_6), kc2896))
	t8_7 := clamp(t8_5.Add(t11_5))
	t9_7 := clamp(t9_6.Add(t10_6))
	t10_7 := clamp(t9_6.Sub(t10_6))
	t11_7 := clamp(t8_5.Sub(t11_5))
	t12_7 := clamp(t15_5.Sub(t12_5))
	t13_7 := clamp(t14_6.Sub(t13_6))
	t14_7 := clamp(t13_6.Add(t14_6))
	t15_7 := clamp(t12_5.Add(t15_5))
	t0_8 := clamp(t0_7.Add(t7_6))
	t1_8 := clamp(t1_7.Add(t6_7))
	t2_8 := clamp(t2_7.Add(t5_7))
	t3_8 := clamp(t3_7.Add(t4_6))
	t4_8 := clamp(t3_7.Sub(t4_6))
	t5_8 := clamp(t2_7.Sub(t5_7))
	t6_8 := clamp(t1_7.Sub(t6_7))
	t7_8 := clamp(t0_7.Sub(t7_6))
	t10_8 := shr12(mac(r12, t13_7.Sub(t10_7), kc2896))
	t11_8 := shr12(mac(r12, t12_7.Sub(t11_7), kc2896))
	t12_8 := shr12(mac(r12, t11_7.Add(t12_7), kc2896))
	t13_8 := shr12(mac(r12, t10_7.Add(t13_7), kc2896))
	t0_9 := clamp(t0_8.Add(t15_7))
	t1_9 := clamp(t1_8.Add(t14_7))
	t2_9 := clamp(t2_8.Add(t13_8))
	t3_9 := clamp(t3_8.Add(t12_8))
	t4_9 := clamp(t4_8.Add(t11_8))
	t5_9 := clamp(t5_8.Add(t10_8))
	t6_9 := clamp(t6_8.Add(t9_7))
	t7_9 := clamp(t7_8.Add(t8_7))
	t8_9 := clamp(t7_8.Sub(t8_7))
	t9_9 := clamp(t6_8.Sub(t9_7))
	t10_9 := clamp(t5_8.Sub(t10_8))
	t11_9 := clamp(t4_8.Sub(t11_8))
	t12_9 := clamp(t3_8.Sub(t12_8))
	t13_9 := clamp(t2_8.Sub(t13_8))
	t14_9 := clamp(t1_8.Sub(t14_7))
	t15_9 := clamp(t0_8.Sub(t15_7))

	in2 := ld(1)
	in62 := ld(31)
	t16_3 := shr12(mac(mac(r12, in2, kc201), in62, kcN4091))
	in34 := ld(17)
	in30 := ld(15)
	t17_3 := shr12(mac(mac(r12, in34, kc3035), in30, kcN2751))
	in18 := ld(9)
	in46 := ld(23)
	t18_3 := shr12(mac(mac(r12, in18, kc1751), in46, kcN3703))
	in50 := ld(25)
	in14 := ld(7)
	t19_3 := shr12(mac(mac(r12, in50, kc3857), in14, kcN1380))
	in10 := ld(5)
	in54 := ld(27)
	t20_3 := shr12(mac(mac(r12, in10, kc995), in54, kcN3973))
	in42 := ld(21)
	in22 := ld(11)
	t21_3 := shr12(mac(mac(r12, in42, kc3513), in22, kcN2106))
	in26 := ld(13)
	in38 := ld(19)
	t22_3 := shr12(mac(mac(r12, in26, kc2440), in38, kcN3290))
	in58 := ld(29)
	in6 := ld(3)
	t23_3 := shr12(mac(mac(r12, in58, kc4052), in6, kcN601))
	t24_3 := shr12(mac(mac(r12, in58, kc601), in6, kc4052))
	t25_3 := shr12(mac(mac(r12, in26, kc3290), in38, kc2440))
	t26_3 := shr12(mac(mac(r12, in42, kc2106), in22, kc3513))
	t27_3 := shr12(mac(mac(r12, in10, kc3973), in54, kc995))
	t28_3 := shr12(mac(mac(r12, in50, kc1380), in14, kc3857))
	t29_3 := shr12(mac(mac(r12, in18, kc3703), in46, kc1751))
	t30_3 := shr12(mac(mac(r12, in34, kc2751), in30, kc3035))
	t31_3 := shr12(mac(mac(r12, in2, kc4091), in62, kc201))
	t16_4 := clamp(t16_3.Add(t17_3))
	t17_4 := clamp(t16_3.Sub(t17_3))
	t18_4 := clamp(t19_3.Sub(t18_3))
	t19_4 := clamp(t18_3.Add(t19_3))
	t20_4 := clamp(t20_3.Add(t21_3))
	t21_4 := clamp(t20_3.Sub(t21_3))
	t22_4 := clamp(t23_3.Sub(t22_3))
	t23_4 := clamp(t22_3.Add(t23_3))
	t24_4 := clamp(t24_3.Add(t25_3))
	t25_4 := clamp(t24_3.Sub(t25_3))
	t26_4 := clamp(t27_3.Sub(t26_3))
	t27_4 := clamp(t26_3.Add(t27_3))
	t28_4 := clamp(t28_3.Add(t29_3))
	t29_4 := clamp(t28_3.Sub(t29_3))
	t30_4 := clamp(t31_3.Sub(t30_3))
	t31_4 := clamp(t30_3.Add(t31_3))
	t17_5 := shr12(mac(mac(r12, t17_4, kcN4017), t30_4, kc799))
	t18_5 := shr12(mac(mac(r12, t18_4, kcN799), t29_4, kcN4017))
	t21_5 := shr12(mac(mac(r12, t21_4, kcN2276), t26_4, kc3406))
	t22_5 := shr12(mac(mac(r12, t22_4, kcN3406), t25_4, kcN2276))
	t25_5 := shr12(mac(mac(r12, t22_4, kcN2276), t25_4, kc3406))
	t26_5 := shr12(mac(mac(r12, t21_4, kc3406), t26_4, kc2276))
	t29_5 := shr12(mac(mac(r12, t18_4, kcN4017), t29_4, kc799))
	t30_5 := shr12(mac(mac(r12, t17_4, kc799), t30_4, kc4017))
	t16_6 := clamp(t16_4.Add(t19_4))
	t17_6 := clamp(t17_5.Add(t18_5))
	t18_6 := clamp(t17_5.Sub(t18_5))
	t19_6 := clamp(t16_4.Sub(t19_4))
	t20_6 := clamp(t23_4.Sub(t20_4))
	t21_6 := clamp(t22_5.Sub(t21_5))
	t22_6 := clamp(t21_5.Add(t22_5))
	t23_6 := clamp(t20_4.Add(t23_4))
	t24_6 := clamp(t24_4.Add(t27_4))
	t25_6 := clamp(t25_5.Add(t26_5))
	t26_6 := clamp(t25_5.Sub(t26_5))
	t27_6 := clamp(t24_4.Sub(t27_4))
	t28_6 := clamp(t31_4.Sub(t28_4))
	t29_6 := clamp(t30_5.Sub(t29_5))
	t30_6 := clamp(t29_5.Add(t30_5))
	t31_6 := clamp(t28_4.Add(t31_4))
	t18_7 := shr12(mac(mac(r12, t18_6, kcN3784), t29_6, kc1567))
	t19_7 := shr12(mac(mac(r12, t19_6, kcN3784), t28_6, kc1567))
	t20_7 := shr12(mac(mac(r12, t20_6, kcN1567), t27_6, kcN3784))
	t21_7 := shr12(mac(mac(r12, t21_6, kcN1567), t26_6, kcN3784))
	t26_7 := shr12(mac(mac(r12, t21_6, kcN3784), t26_6, kc1567))
	t27_7 := shr12(mac(mac(r12, t20_6, kcN3784), t27_6, kc1567))
	t28_7 := shr12(mac(mac(r12, t19_6, kc1567), t28_6, kc3784))
	t29_7 := shr12(mac(mac(r12, t18_6, kc1567), t29_6, kc3784))
	t16_8 := clamp(t16_6.Add(t23_6))
	t17_8 := clamp(t17_6.Add(t22_6))
	t18_8 := clamp(t18_7.Add(t21_7))
	t19_8 := clamp(t19_7.Add(t20_7))
	t20_8 := clamp(t19_7.Sub(t20_7))
	t21_8 := clamp(t18_7.Sub(t21_7))
	t22_8 := clamp(t17_6.Sub(t22_6))
	t23_8 := clamp(t16_6.Sub(t23_6))
	t24_8 := clamp(t31_6.Sub(t24_6))
	t25_8 := clamp(t30_6.Sub(t25_6))
	t26_8 := clamp(t29_7.Sub(t26_7))
	t27_8 := clamp(t28_7.Sub(t27_7))
	t28_8 := clamp(t27_7.Add(t28_7))
	t29_8 := clamp(t26_7.Add(t29_7))
	t30_8 := clamp(t25_6.Add(t30_6))
	t31_8 := clamp(t24_6.Add(t31_6))
	t20_9 := shr12(mac(r12, t27_8.Sub(t20_8), kc2896))
	t21_9 := shr12(mac(r12, t26_8.Sub(t21_8), kc2896))
	t22_9 := shr12(mac(r12, t25_8.Sub(t22_8), kc2896))
	t23_9 := shr12(mac(r12, t24_8.Sub(t23_8), kc2896))
	t24_9 := shr12(mac(r12, t23_8.Add(t24_8), kc2896))
	t25_9 := shr12(mac(r12, t22_8.Add(t25_8), kc2896))
	t26_9 := shr12(mac(r12, t21_8.Add(t26_8), kc2896))
	t27_9 := shr12(mac(r12, t20_8.Add(t27_8), kc2896))

	t0_10 := clamp(t0_9.Add(t31_8))
	t1_10 := clamp(t1_9.Add(t30_8))
	t2_10 := clamp(t2_9.Add(t29_8))
	t3_10 := clamp(t3_9.Add(t28_8))
	t4_10 := clamp(t4_9.Add(t27_9))
	t5_10 := clamp(t5_9.Add(t26_9))
	t6_10 := clamp(t6_9.Add(t25_9))
	t7_10 := clamp(t7_9.Add(t24_9))
	t8_10 := clamp(t8_9.Add(t23_9))
	t9_10 := clamp(t9_9.Add(t22_9))
	t10_10 := clamp(t10_9.Add(t21_9))
	t11_10 := clamp(t11_9.Add(t20_9))
	t12_10 := clamp(t12_9.Add(t19_8))
	t13_10 := clamp(t13_9.Add(t18_8))
	t14_10 := clamp(t14_9.Add(t17_8))
	t15_10 := clamp(t15_9.Add(t16_8))
	t16_10 := clamp(t15_9.Sub(t16_8))
	t17_10 := clamp(t14_9.Sub(t17_8))
	t18_10 := clamp(t13_9.Sub(t18_8))
	t19_10 := clamp(t12_9.Sub(t19_8))
	t20_10 := clamp(t11_9.Sub(t20_9))
	t21_10 := clamp(t10_9.Sub(t21_9))
	t22_10 := clamp(t9_9.Sub(t22_9))
	t23_10 := clamp(t8_9.Sub(t23_9))
	t24_10 := clamp(t7_9.Sub(t24_9))
	t25_10 := clamp(t6_9.Sub(t25_9))
	t26_10 := clamp(t5_9.Sub(t26_9))
	t27_10 := clamp(t4_9.Sub(t27_9))
	t28_10 := clamp(t3_9.Sub(t28_8))
	t29_10 := clamp(t2_9.Sub(t29_8))
	t30_10 := clamp(t1_9.Sub(t30_8))
	t31_10 := clamp(t0_9.Sub(t31_8))
	st(0, t0_10)
	st(1, t1_10)
	st(2, t2_10)
	st(3, t3_10)
	st(4, t4_10)
	st(5, t5_10)
	st(6, t6_10)
	st(7, t7_10)
	st(8, t8_10)
	st(9, t9_10)
	st(10, t10_10)
	st(11, t11_10)
	st(12, t12_10)
	st(13, t13_10)
	st(14, t14_10)
	st(15, t15_10)
	st(16, t16_10)
	st(17, t17_10)
	st(18, t18_10)
	st(19, t19_10)
	st(20, t20_10)
	st(21, t21_10)
	st(22, t22_10)
	st(23, t23_10)
	st(24, t24_10)
	st(25, t25_10)
	st(26, t26_10)
	st(27, t27_10)
	st(28, t28_10)
	st(29, t29_10)
	st(30, t30_10)
	st(31, t31_10)
}

func inverseDCT32Lanes4Wide(p []int32, stride int, min, max int32) {
	k := itxCoefsPtr
	mn, mx := archsimd.BroadcastInt32x4(min), archsimd.BroadcastInt32x4(max)
	r12, n12 := itxRoundBias(12), itxShiftAmount(12)
	ld := func(i int) archsimd.Int32x4 {
		o := i * stride
		return archsimd.LoadInt32x4Array((*[4]int32)(p[o : o+4]))
	}
	st := func(i int, v archsimd.Int32x4) {
		o := i * stride
		v.StoreArray((*[4]int32)(p[o : o+4]))
	}
	clamp := func(v archsimd.Int32x4) archsimd.Int32x4 { return v.Max(mn).Min(mx) }
	mac := func(acc, x archsimd.Int32x4, c int) archsimd.Int32x4 {
		return itxMulAcc(acc, x, archsimd.LoadInt32x4Array(&k[c]))
	}
	shr12 := func(v archsimd.Int32x4) archsimd.Int32x4 { return itxShr12(v, n12) }

	in4 := ld(2)
	in60 := ld(30)
	t8_4 := shr12(mac(mac(r12, in4, kc401), in60, kc20)).Sub(in60)
	in36 := ld(18)
	in28 := ld(14)
	t9_4 := shr12(mac(mac(r12, in36, kcN930), in28, kc1498)).Add(in36).Sub(in28)
	in20 := ld(10)
	in44 := ld(22)
	t10_4 := shr12(mac(mac(r12, in20, kc1931), in44, kc484)).Sub(in44)
	in52 := ld(26)
	in12 := ld(6)
	t11_4 := shr12(mac(mac(r12, in52, kcN176), in12, kcN1189)).Add(in52)
	t12_4 := shr12(mac(mac(r12, in52, kc1189), in12, kcN176)).Add(in12)
	t13_4 := shr12(mac(mac(r12, in20, kcN484), in44, kc1931)).Add(in20)
	t14_4 := shr12(mac(mac(r12, in36, kcN1498), in28, kcN930)).Add(in36).Add(in28)
	t15_4 := shr12(mac(mac(r12, in4, kcN20), in60, kc401)).Add(in4)
	in8 := ld(4)
	in56 := ld(28)
	t4_5 := shr12(mac(mac(r12, in8, kc799), in56, kc79)).Sub(in56)
	in40 := ld(20)
	in24 := ld(12)
	t5_5 := shr12(mac(mac(r12, in40, kcN690), in24, kc1820)).Add(in40).Sub(in24)
	t6_5 := shr12(mac(mac(r12, in40, kcN1820), in24, kcN690)).Add(in40).Add(in24)
	t7_5 := shr12(mac(mac(r12, in8, kcN79), in56, kc799)).Add(in8)
	t8_5 := clamp(t8_4.Add(t9_4))
	t9_5 := clamp(t8_4.Sub(t9_4))
	t10_5 := clamp(t11_4.Sub(t10_4))
	t11_5 := clamp(t10_4.Add(t11_4))
	t12_5 := clamp(t12_4.Add(t13_4))
	t13_5 := clamp(t12_4.Sub(t13_4))
	t14_5 := clamp(t15_4.Sub(t14_4))
	t15_5 := clamp(t14_4.Add(t15_4))
	in0 := ld(0)
	in32 := ld(16)
	d236 := in0.Add(in32)
	t0_6 := shr12(mac(r12, d236, kcN1200)).Add(d236)
	d237 := in0.Sub(in32)
	t1_6 := shr12(mac(r12, d237, kcN1200)).Add(d237)
	in16 := ld(8)
	in48 := ld(24)
	t2_6 := shr12(mac(mac(r12, in16, kc1567), in48, kc312)).Sub(in48)
	t3_6 := shr12(mac(mac(r12, in16, kcN312), in48, kc1567)).Add(in16)
	t4_6 := clamp(t4_5.Add(t5_5))
	t5_6 := clamp(t4_5.Sub(t5_5))
	t6_6 := clamp(t7_5.Sub(t6_5))
	t7_6 := clamp(t6_5.Add(t7_5))
	t9_6 := shr12(mac(mac(r12, t9_5, kc312), t14_5, kc1567)).Sub(t9_5)
	t10_6 := shr12(mac(mac(r12, t10_5, kcN1567), t13_5, kc312)).Sub(t13_5)
	t13_6 := shr12(mac(mac(r12, t10_5, kc312), t13_5, kc1567)).Sub(t10_5)
	t14_6 := shr12(mac(mac(r12, t9_5, kc1567), t14_5, kcN312)).Add(t14_5)
	t0_7 := clamp(t0_6.Add(t3_6))
	t1_7 := clamp(t1_6.Add(t2_6))
	t2_7 := clamp(t1_6.Sub(t2_6))
	t3_7 := clamp(t0_6.Sub(t3_6))
	d284 := t6_6.Sub(t5_6)
	t5_7 := shr12(mac(r12, d284, kcN1200)).Add(d284)
	d285 := t5_6.Add(t6_6)
	t6_7 := shr12(mac(r12, d285, kcN1200)).Add(d285)
	t8_7 := clamp(t8_5.Add(t11_5))
	t9_7 := clamp(t9_6.Add(t10_6))
	t10_7 := clamp(t9_6.Sub(t10_6))
	t11_7 := clamp(t8_5.Sub(t11_5))
	t12_7 := clamp(t15_5.Sub(t12_5))
	t13_7 := clamp(t14_6.Sub(t13_6))
	t14_7 := clamp(t13_6.Add(t14_6))
	t15_7 := clamp(t12_5.Add(t15_5))
	t0_8 := clamp(t0_7.Add(t7_6))
	t1_8 := clamp(t1_7.Add(t6_7))
	t2_8 := clamp(t2_7.Add(t5_7))
	t3_8 := clamp(t3_7.Add(t4_6))
	t4_8 := clamp(t3_7.Sub(t4_6))
	t5_8 := clamp(t2_7.Sub(t5_7))
	t6_8 := clamp(t1_7.Sub(t6_7))
	t7_8 := clamp(t0_7.Sub(t7_6))
	d342 := t13_7.Sub(t10_7)
	t10_8 := shr12(mac(r12, d342, kcN1200)).Add(d342)
	d343 := t12_7.Sub(t11_7)
	t11_8 := shr12(mac(r12, d343, kcN1200)).Add(d343)
	d344 := t11_7.Add(t12_7)
	t12_8 := shr12(mac(r12, d344, kcN1200)).Add(d344)
	d345 := t10_7.Add(t13_7)
	t13_8 := shr12(mac(r12, d345, kcN1200)).Add(d345)
	t0_9 := clamp(t0_8.Add(t15_7))
	t1_9 := clamp(t1_8.Add(t14_7))
	t2_9 := clamp(t2_8.Add(t13_8))
	t3_9 := clamp(t3_8.Add(t12_8))
	t4_9 := clamp(t4_8.Add(t11_8))
	t5_9 := clamp(t5_8.Add(t10_8))
	t6_9 := clamp(t6_8.Add(t9_7))
	t7_9 := clamp(t7_8.Add(t8_7))
	t8_9 := clamp(t7_8.Sub(t8_7))
	t9_9 := clamp(t6_8.Sub(t9_7))
	t10_9 := clamp(t5_8.Sub(t10_8))
	t11_9 := clamp(t4_8.Sub(t11_8))
	t12_9 := clamp(t3_8.Sub(t12_8))
	t13_9 := clamp(t2_8.Sub(t13_8))
	t14_9 := clamp(t1_8.Sub(t14_7))
	t15_9 := clamp(t0_8.Sub(t15_7))

	in2 := ld(1)
	in62 := ld(31)
	t16_3 := shr12(mac(mac(r12, in2, kc201), in62, kc5)).Sub(in62)
	in34 := ld(17)
	in30 := ld(15)
	t17_3 := shr12(mac(mac(r12, in34, kcN1061), in30, kc1345)).Add(in34).Sub(in30)
	in18 := ld(9)
	in46 := ld(23)
	t18_3 := shr12(mac(mac(r12, in18, kc1751), in46, kc393)).Sub(in46)
	in50 := ld(25)
	in14 := ld(7)
	t19_3 := shr12(mac(mac(r12, in50, kcN239), in14, kcN1380)).Add(in50)
	in10 := ld(5)
	in54 := ld(27)
	t20_3 := shr12(mac(mac(r12, in10, kc995), in54, kc123)).Sub(in54)
	in42 := ld(21)
	in22 := ld(11)
	t21_3 := shr12(mac(mac(r12, in42, kcN583), in22, kc1990)).Add(in42).Sub(in22)
	in26 := ld(13)
	in38 := ld(19)
	t22_3 := shr12(mac(mac(r12, in26, kcN1656), in38, kc806)).Add(in26).Sub(in38)
	in58 := ld(29)
	in6 := ld(3)
	t23_3 := shr12(mac(mac(r12, in58, kcN44), in6, kcN601)).Add(in58)
	t24_3 := shr12(mac(mac(r12, in58, kc601), in6, kcN44)).Add(in6)
	t25_3 := shr12(mac(mac(r12, in26, kcN806), in38, kcN1656)).Add(in26).Add(in38)
	t26_3 := shr12(mac(mac(r12, in42, kcN1990), in22, kcN583)).Add(in42).Add(in22)
	t27_3 := shr12(mac(mac(r12, in10, kcN123), in54, kc995)).Add(in10)
	t28_3 := shr12(mac(mac(r12, in50, kc1380), in14, kcN239)).Add(in14)
	t29_3 := shr12(mac(mac(r12, in18, kcN393), in46, kc1751)).Add(in18)
	t30_3 := shr12(mac(mac(r12, in34, kcN1345), in30, kcN1061)).Add(in34).Add(in30)
	t31_3 := shr12(mac(mac(r12, in2, kcN5), in62, kc201)).Add(in2)
	t16_4 := clamp(t16_3.Add(t17_3))
	t17_4 := clamp(t16_3.Sub(t17_3))
	t18_4 := clamp(t19_3.Sub(t18_3))
	t19_4 := clamp(t18_3.Add(t19_3))
	t20_4 := clamp(t20_3.Add(t21_3))
	t21_4 := clamp(t20_3.Sub(t21_3))
	t22_4 := clamp(t23_3.Sub(t22_3))
	t23_4 := clamp(t22_3.Add(t23_3))
	t24_4 := clamp(t24_3.Add(t25_3))
	t25_4 := clamp(t24_3.Sub(t25_3))
	t26_4 := clamp(t27_3.Sub(t26_3))
	t27_4 := clamp(t26_3.Add(t27_3))
	t28_4 := clamp(t28_3.Add(t29_3))
	t29_4 := clamp(t28_3.Sub(t29_3))
	t30_4 := clamp(t31_3.Sub(t30_3))
	t31_4 := clamp(t30_3.Add(t31_3))
	t17_5 := shr12(mac(mac(r12, t17_4, kc79), t30_4, kc799)).Sub(t17_4)
	t18_5 := shr12(mac(mac(r12, t18_4, kcN799), t29_4, kc79)).Sub(t29_4)
	t21_5 := shr12(mac(mac(r12, t21_4, kc1820), t26_4, kcN690)).Sub(t21_4).Add(t26_4)
	t22_5 := shr12(mac(mac(r12, t22_4, kc690), t25_4, kc1820)).Sub(t22_4).Sub(t25_4)
	t25_5 := shr12(mac(mac(r12, t22_4, kc1820), t25_4, kcN690)).Sub(t22_4).Add(t25_4)
	t26_5 := shr12(mac(mac(r12, t21_4, kcN690), t26_4, kcN1820)).Add(t21_4).Add(t26_4)
	t29_5 := shr12(mac(mac(r12, t18_4, kc79), t29_4, kc799)).Sub(t18_4)
	t30_5 := shr12(mac(mac(r12, t17_4, kc799), t30_4, kcN79)).Add(t30_4)
	t16_6 := clamp(t16_4.Add(t19_4))
	t17_6 := clamp(t17_5.Add(t18_5))
	t18_6 := clamp(t17_5.Sub(t18_5))
	t19_6 := clamp(t16_4.Sub(t19_4))
	t20_6 := clamp(t23_4.Sub(t20_4))
	t21_6 := clamp(t22_5.Sub(t21_5))
	t22_6 := clamp(t21_5.Add(t22_5))
	t23_6 := clamp(t20_4.Add(t23_4))
	t24_6 := clamp(t24_4.Add(t27_4))
	t25_6 := clamp(t25_5.Add(t26_5))
	t26_6 := clamp(t25_5.Sub(t26_5))
	t27_6 := clamp(t24_4.Sub(t27_4))
	t28_6 := clamp(t31_4.Sub(t28_4))
	t29_6 := clamp(t30_5.Sub(t29_5))
	t30_6 := clamp(t29_5.Add(t30_5))
	t31_6 := clamp(t28_4.Add(t31_4))
	t18_7 := shr12(mac(mac(r12, t18_6, kc312), t29_6, kc1567)).Sub(t18_6)
	t19_7 := shr12(mac(mac(r12, t19_6, kc312), t28_6, kc1567)).Sub(t19_6)
	t20_7 := shr12(mac(mac(r12, t20_6, kcN1567), t27_6, kc312)).Sub(t27_6)
	t21_7 := shr12(mac(mac(r12, t21_6, kcN1567), t26_6, kc312)).Sub(t26_6)
	t26_7 := shr12(mac(mac(r12, t21_6, kc312), t26_6, kc1567)).Sub(t21_6)
	t27_7 := shr12(mac(mac(r12, t20_6, kc312), t27_6, kc1567)).Sub(t20_6)
	t28_7 := shr12(mac(mac(r12, t19_6, kc1567), t28_6, kcN312)).Add(t28_6)
	t29_7 := shr12(mac(mac(r12, t18_6, kc1567), t29_6, kcN312)).Add(t29_6)
	t16_8 := clamp(t16_6.Add(t23_6))
	t17_8 := clamp(t17_6.Add(t22_6))
	t18_8 := clamp(t18_7.Add(t21_7))
	t19_8 := clamp(t19_7.Add(t20_7))
	t20_8 := clamp(t19_7.Sub(t20_7))
	t21_8 := clamp(t18_7.Sub(t21_7))
	t22_8 := clamp(t17_6.Sub(t22_6))
	t23_8 := clamp(t16_6.Sub(t23_6))
	t24_8 := clamp(t31_6.Sub(t24_6))
	t25_8 := clamp(t30_6.Sub(t25_6))
	t26_8 := clamp(t29_7.Sub(t26_7))
	t27_8 := clamp(t28_7.Sub(t27_7))
	t28_8 := clamp(t27_7.Add(t28_7))
	t29_8 := clamp(t26_7.Add(t29_7))
	t30_8 := clamp(t25_6.Add(t30_6))
	t31_8 := clamp(t24_6.Add(t31_6))
	d394 := t27_8.Sub(t20_8)
	t20_9 := shr12(mac(r12, d394, kcN1200)).Add(d394)
	d395 := t26_8.Sub(t21_8)
	t21_9 := shr12(mac(r12, d395, kcN1200)).Add(d395)
	d396 := t25_8.Sub(t22_8)
	t22_9 := shr12(mac(r12, d396, kcN1200)).Add(d396)
	d397 := t24_8.Sub(t23_8)
	t23_9 := shr12(mac(r12, d397, kcN1200)).Add(d397)
	d398 := t23_8.Add(t24_8)
	t24_9 := shr12(mac(r12, d398, kcN1200)).Add(d398)
	d399 := t22_8.Add(t25_8)
	t25_9 := shr12(mac(r12, d399, kcN1200)).Add(d399)
	d400 := t21_8.Add(t26_8)
	t26_9 := shr12(mac(r12, d400, kcN1200)).Add(d400)
	d401 := t20_8.Add(t27_8)
	t27_9 := shr12(mac(r12, d401, kcN1200)).Add(d401)

	t0_10 := clamp(t0_9.Add(t31_8))
	t1_10 := clamp(t1_9.Add(t30_8))
	t2_10 := clamp(t2_9.Add(t29_8))
	t3_10 := clamp(t3_9.Add(t28_8))
	t4_10 := clamp(t4_9.Add(t27_9))
	t5_10 := clamp(t5_9.Add(t26_9))
	t6_10 := clamp(t6_9.Add(t25_9))
	t7_10 := clamp(t7_9.Add(t24_9))
	t8_10 := clamp(t8_9.Add(t23_9))
	t9_10 := clamp(t9_9.Add(t22_9))
	t10_10 := clamp(t10_9.Add(t21_9))
	t11_10 := clamp(t11_9.Add(t20_9))
	t12_10 := clamp(t12_9.Add(t19_8))
	t13_10 := clamp(t13_9.Add(t18_8))
	t14_10 := clamp(t14_9.Add(t17_8))
	t15_10 := clamp(t15_9.Add(t16_8))
	t16_10 := clamp(t15_9.Sub(t16_8))
	t17_10 := clamp(t14_9.Sub(t17_8))
	t18_10 := clamp(t13_9.Sub(t18_8))
	t19_10 := clamp(t12_9.Sub(t19_8))
	t20_10 := clamp(t11_9.Sub(t20_9))
	t21_10 := clamp(t10_9.Sub(t21_9))
	t22_10 := clamp(t9_9.Sub(t22_9))
	t23_10 := clamp(t8_9.Sub(t23_9))
	t24_10 := clamp(t7_9.Sub(t24_9))
	t25_10 := clamp(t6_9.Sub(t25_9))
	t26_10 := clamp(t5_9.Sub(t26_9))
	t27_10 := clamp(t4_9.Sub(t27_9))
	t28_10 := clamp(t3_9.Sub(t28_8))
	t29_10 := clamp(t2_9.Sub(t29_8))
	t30_10 := clamp(t1_9.Sub(t30_8))
	t31_10 := clamp(t0_9.Sub(t31_8))
	st(0, t0_10)
	st(1, t1_10)
	st(2, t2_10)
	st(3, t3_10)
	st(4, t4_10)
	st(5, t5_10)
	st(6, t6_10)
	st(7, t7_10)
	st(8, t8_10)
	st(9, t9_10)
	st(10, t10_10)
	st(11, t11_10)
	st(12, t12_10)
	st(13, t13_10)
	st(14, t14_10)
	st(15, t15_10)
	st(16, t16_10)
	st(17, t17_10)
	st(18, t18_10)
	st(19, t19_10)
	st(20, t20_10)
	st(21, t21_10)
	st(22, t22_10)
	st(23, t23_10)
	st(24, t24_10)
	st(25, t25_10)
	st(26, t26_10)
	st(27, t27_10)
	st(28, t28_10)
	st(29, t29_10)
	st(30, t30_10)
	st(31, t31_10)
}

// Prebuilt vectors avoid rebuilding a broadcast for each butterfly. Keep the
// signed values explicit: the networks use both signs of several twiddles.
var (
	fwdConstN8153 = fwdBcast(-8153)
	fwdConstN8035 = fwdBcast(-8035)
	fwdConstN7568 = fwdBcast(-7568)
	fwdConstN7225 = fwdBcast(-7225)
	fwdConstN5793 = fwdBcast(-5793)
	fwdConstN5197 = fwdBcast(-5197)
	fwdConstN4551 = fwdBcast(-4551)
	fwdConstN4091 = fwdBcast(-4091)
	fwdConstN4076 = fwdBcast(-4076)
	fwdConstN4017 = fwdBcast(-4017)
	fwdConstN3973 = fwdBcast(-3973)
	fwdConstN3784 = fwdBcast(-3784)
	fwdConstN3703 = fwdBcast(-3703)
	fwdConstN3612 = fwdBcast(-3612)
	fwdConstN3406 = fwdBcast(-3406)
	fwdConstN3290 = fwdBcast(-3290)
	fwdConstN3135 = fwdBcast(-3135)
	fwdConstN2896 = fwdBcast(-2896)
	fwdConstN2751 = fwdBcast(-2751)
	fwdConstN2598 = fwdBcast(-2598)
	fwdConstN2378 = fwdBcast(-2378)
	fwdConstN2276 = fwdBcast(-2276)
	fwdConstN2106 = fwdBcast(-2106)
	fwdConstN1567 = fwdBcast(-1567)
	fwdConstN1380 = fwdBcast(-1380)
	fwdConstN1189 = fwdBcast(-1189)
	fwdConstN799  = fwdBcast(-799)
	fwdConstN601  = fwdBcast(-601)
	fwdConst1     = fwdBcast(1)
	fwdConst2     = fwdBcast(2)
	fwdConst8     = fwdBcast(8)
	fwdConst201   = fwdBcast(201)
	fwdConst401   = fwdBcast(401)
	fwdConst601   = fwdBcast(601)
	fwdConst799   = fwdBcast(799)
	fwdConst803   = fwdBcast(803)
	fwdConst995   = fwdBcast(995)
	fwdConst1189  = fwdBcast(1189)
	fwdConst1380  = fwdBcast(1380)
	fwdConst1567  = fwdBcast(1567)
	fwdConst1598  = fwdBcast(1598)
	fwdConst1751  = fwdBcast(1751)
	fwdConst1931  = fwdBcast(1931)
	fwdConst2106  = fwdBcast(2106)
	fwdConst2276  = fwdBcast(2276)
	fwdConst2378  = fwdBcast(2378)
	fwdConst2440  = fwdBcast(2440)
	fwdConst2598  = fwdBcast(2598)
	fwdConst2751  = fwdBcast(2751)
	fwdConst2896  = fwdBcast(2896)
	fwdConst3035  = fwdBcast(3035)
	fwdConst3135  = fwdBcast(3135)
	fwdConst3166  = fwdBcast(3166)
	fwdConst3290  = fwdBcast(3290)
	fwdConst3406  = fwdBcast(3406)
	fwdConst3513  = fwdBcast(3513)
	fwdConst3612  = fwdBcast(3612)
	fwdConst3703  = fwdBcast(3703)
	fwdConst3784  = fwdBcast(3784)
	fwdConst3857  = fwdBcast(3857)
	fwdConst3862  = fwdBcast(3862)
	fwdConst3920  = fwdBcast(3920)
	fwdConst3973  = fwdBcast(3973)
	fwdConst4017  = fwdBcast(4017)
	fwdConst4052  = fwdBcast(4052)
	fwdConst4076  = fwdBcast(4076)
	fwdConst4091  = fwdBcast(4091)
	fwdConst4551  = fwdBcast(4551)
	fwdConst5197  = fwdBcast(5197)
	fwdConst5793  = fwdBcast(5793)
	fwdConst6333  = fwdBcast(6333)
	fwdConst6811  = fwdBcast(6811)
	fwdConst7225  = fwdBcast(7225)
	fwdConst7568  = fwdBcast(7568)
	fwdConst7839  = fwdBcast(7839)
	fwdConst8035  = fwdBcast(8035)
	fwdConst8153  = fwdBcast(8153)
)

// The 16- and 32-point forward DCT networks are generated by fwdgen from the
// scalar fwdDCT16 and fwdDCT32 (fdct.go): one statement per node of the
// scalar network, so they match the scalar oracle exactly. The int32 lanes
// are exact over the 8-bit residual guard: every half-butterfly pre-shift sum
// is bounded below 2^31 there.

// fwdRoundShift2V is fwdRoundShift2 per lane: (v + 2) >> 2 (16x16 shift[1]).
func fwdRoundShift2V(v fwdVec) fwdVec {
	return fwdShr(v.Add(fwdConst2), 2)
}

// fwdRoundShift4V is fwdRoundShift4 per lane: (v + 8) >> 4 (32x32 shift[1]).
func fwdRoundShift4V(v fwdVec) fwdVec {
	return fwdShr(v.Add(fwdConst8), 4)
}

// forwardDCT16x16SIMDGuarded routes residuals outside the 8-bit range to the
// int64 scalar reference.
func forwardDCT16x16SIMDGuarded(coeff []int32, coeffStride int, residual []int16, residualStride int) {
	if !residualFitsMagnitude(residual, residualStride, 16, 16, 255) {
		forwardDCT16x16PureGo(coeff, coeffStride, residual, residualStride)
		return
	}
	forwardDCT16x16SIMD(coeff, coeffStride, residual, residualStride)
}

// forwardDCT32x32SIMDGuarded routes residuals outside the 8-bit range to the
// int64 scalar reference.
func forwardDCT32x32SIMDGuarded(coeff []int32, coeffStride int, residual []int16, residualStride int) {
	if !residualFitsMagnitude(residual, residualStride, 32, 32, 255) {
		forwardDCT32x32PureGo(coeff, coeffStride, residual, residualStride)
		return
	}
	forwardDCT32x32SIMD(coeff, coeffStride, residual, residualStride)
}

// Shared Go SIMD forward-transform network. Each 1-D pass is vectorized across
// independent lines: a fwdVec holds one sample index for fwdLanes lines (the
// columns on the column pass, the rows on the row pass), so every butterfly is
// one vector operation. The butterfly bodies mirror fwdDCT8Values and
// fwdADST8Values, and the drivers mirror the PureGo two-pass references, so the
// output is bit-identical to the scalar oracle for every residual the callers'
// guards admit. Intermediates stay in int32 lanes; the guards bound them.

// fwdRound13 and fwdRound12 are the half_btf rounding terms, pre-broadcast so
// each use is one load instead of a broadcast.
var (
	fwdRound13 = fwdBcast(1 << 12)
	fwdRound12 = fwdBcast(1 << 11)
)

// fwdRoundShift1V is fwdRoundShift1Value per lane: (v + 1 + (v>>31)) >> 1.
func fwdRoundShift1V(v fwdVec) fwdVec {
	return fwdShr(v.Add(fwdConst1).Add(fwdShr(v, 31)), 1)
}

// fwdColDCT8 runs the DCT8 column pass for the fwdLanes columns starting at g
// of the packed 8x8 residual and stores the shift[1]=-1 rounded outputs into
// buf (row-major, stride 8).
func fwdColDCT8(buf *[64]int32, g int, residual *[64]int16) {
	g &= 8 - fwdLanes
	x0 := fwdLoadResAt(residual[:], 0*8, g)
	x1 := fwdLoadResAt(residual[:], 1*8, g)
	x2 := fwdLoadResAt(residual[:], 2*8, g)
	x3 := fwdLoadResAt(residual[:], 3*8, g)
	x4 := fwdLoadResAt(residual[:], 4*8, g)
	x5 := fwdLoadResAt(residual[:], 5*8, g)
	x6 := fwdLoadResAt(residual[:], 6*8, g)
	x7 := fwdLoadResAt(residual[:], 7*8, g)
	c8 := fwdConst8035
	c16 := fwdConst7568
	c24 := fwdConst6811
	c32 := fwdConst5793
	nc32 := fwdConstN5793
	c40 := fwdConst4551
	c48 := fwdConst3135
	c56 := fwdConst1598
	nc8 := fwdConstN8035
	nc16 := fwdConstN7568
	nc40 := fwdConstN4551

	b0 := x0.Add(x7)
	b1 := x1.Add(x6)
	b2 := x2.Add(x5)
	b3 := x3.Add(x4)
	b4 := x3.Sub(x4)
	b5 := x2.Sub(x5)
	b6 := x1.Sub(x6)
	b7 := x0.Sub(x7)

	s0 := b0.Add(b3)
	s1 := b1.Add(b2)
	s2 := b1.Sub(b2)
	s3 := b0.Sub(b3)
	s4 := b4
	s5 := fwdHalfBtf13V(nc32, b5, c32, b6)
	s6 := fwdHalfBtf13V(c32, b6, c32, b5)
	s7 := b7

	b0 = fwdHalfBtf13V(c32, s0, c32, s1)
	b1 = fwdHalfBtf13V(nc32, s1, c32, s0)
	b2 = fwdHalfBtf13V(c48, s2, c16, s3)
	b3 = fwdHalfBtf13V(c48, s3, nc16, s2)
	b4 = s4.Add(s5)
	b5 = s4.Sub(s5)
	b6 = s7.Sub(s6)
	b7 = s7.Add(s6)

	s0 = b0
	s1 = b1
	s2 = b2
	s3 = b3
	s4 = fwdHalfBtf13V(c56, b4, c8, b7)
	s5 = fwdHalfBtf13V(c24, b5, c40, b6)
	s6 = fwdHalfBtf13V(c24, b6, nc40, b5)
	s7 = fwdHalfBtf13V(c56, b7, nc8, b4)
	fwdStoreI32At(buf[:], 0*8+g, fwdRoundShift1V(s0))
	fwdStoreI32At(buf[:], 1*8+g, fwdRoundShift1V(s4))
	fwdStoreI32At(buf[:], 2*8+g, fwdRoundShift1V(s2))
	fwdStoreI32At(buf[:], 3*8+g, fwdRoundShift1V(s6))
	fwdStoreI32At(buf[:], 4*8+g, fwdRoundShift1V(s1))
	fwdStoreI32At(buf[:], 5*8+g, fwdRoundShift1V(s5))
	fwdStoreI32At(buf[:], 6*8+g, fwdRoundShift1V(s3))
	fwdStoreI32At(buf[:], 7*8+g, fwdRoundShift1V(s7))
}

// fwdColADST8 is fwdColDCT8 with the ADST8 column transform.
func fwdColADST8(buf *[64]int32, g int, residual *[64]int16) {
	g &= 8 - fwdLanes
	x0 := fwdLoadResAt(residual[:], 0*8, g)
	x1 := fwdLoadResAt(residual[:], 1*8, g)
	x2 := fwdLoadResAt(residual[:], 2*8, g)
	x3 := fwdLoadResAt(residual[:], 3*8, g)
	x4 := fwdLoadResAt(residual[:], 4*8, g)
	x5 := fwdLoadResAt(residual[:], 5*8, g)
	x6 := fwdLoadResAt(residual[:], 6*8, g)
	x7 := fwdLoadResAt(residual[:], 7*8, g)
	c4 := fwdConst8153
	c12 := fwdConst7839
	c16 := fwdConst7568
	c20 := fwdConst7225
	c28 := fwdConst6333
	c32 := fwdConst5793
	nc32 := fwdConstN5793
	c36 := fwdConst5197
	c44 := fwdConst3862
	c48 := fwdConst3135
	nc48 := fwdConstN3135
	c52 := fwdConst2378
	c60 := fwdConst803
	nc4 := fwdConstN8153
	nc20 := fwdConstN7225
	nc16 := fwdConstN7568
	nc36 := fwdConstN5197
	nc52 := fwdConstN2378

	nx3 := x3.Neg()
	nx7 := x7.Neg()
	nx5 := x5.Neg()
	nx1 := x1.Neg()

	s0 := x0
	s1 := nx7
	s2 := fwdHalfBtf13V(c32, nx3, c32, x4)
	s3 := fwdHalfBtf13V(c32, nx3, nc32, x4)
	s4 := nx1
	s5 := x6
	s6 := fwdHalfBtf13V(c32, x2, c32, nx5)
	s7 := fwdHalfBtf13V(c32, x2, nc32, nx5)

	t0 := s0.Add(s2)
	t1 := s1.Add(s3)
	t2 := s0.Sub(s2)
	t3 := s1.Sub(s3)
	t4 := s4.Add(s6)
	t5 := s5.Add(s7)
	t6 := s4.Sub(s6)
	t7 := s5.Sub(s7)

	s4 = fwdHalfBtf13V(c16, t4, c48, t5)
	s5 = fwdHalfBtf13V(c48, t4, nc16, t5)
	s6 = fwdHalfBtf13V(nc48, t6, c16, t7)
	s7 = fwdHalfBtf13V(c16, t6, c48, t7)

	t4 = t0.Sub(s4)
	t5 = t1.Sub(s5)
	t6 = t2.Sub(s6)
	t7 = t3.Sub(s7)
	t0 = t0.Add(s4)
	t1 = t1.Add(s5)
	t2 = t2.Add(s6)
	t3 = t3.Add(s7)

	s0 = fwdHalfBtf13V(c4, t0, c60, t1)
	s1 = fwdHalfBtf13V(c60, t0, nc4, t1)
	s2 = fwdHalfBtf13V(c20, t2, c44, t3)
	s3 = fwdHalfBtf13V(c44, t2, nc20, t3)
	s4 = fwdHalfBtf13V(c36, t4, c28, t5)
	s5 = fwdHalfBtf13V(c28, t4, nc36, t5)
	s6 = fwdHalfBtf13V(c52, t6, c12, t7)
	s7 = fwdHalfBtf13V(c12, t6, nc52, t7)
	fwdStoreI32At(buf[:], 0*8+g, fwdRoundShift1V(s1))
	fwdStoreI32At(buf[:], 1*8+g, fwdRoundShift1V(s6))
	fwdStoreI32At(buf[:], 2*8+g, fwdRoundShift1V(s3))
	fwdStoreI32At(buf[:], 3*8+g, fwdRoundShift1V(s4))
	fwdStoreI32At(buf[:], 4*8+g, fwdRoundShift1V(s5))
	fwdStoreI32At(buf[:], 5*8+g, fwdRoundShift1V(s2))
	fwdStoreI32At(buf[:], 6*8+g, fwdRoundShift1V(s7))
	fwdStoreI32At(buf[:], 7*8+g, fwdRoundShift1V(s0))
}

// fwdRowDCT8 runs the DCT8 row pass for the fwdLanes rows starting at h into
// the packed 8x8 coeff. bufT is the transposed column-pass output: bufT[c*8+k]
// is the value for row k.
func fwdRowDCT8(coeff *[64]int32, bufT *[64]int32, h int) {
	h &= 8 - fwdLanes
	x0 := fwdLoadI32At(bufT[:], 0*8+h)
	x1 := fwdLoadI32At(bufT[:], 1*8+h)
	x2 := fwdLoadI32At(bufT[:], 2*8+h)
	x3 := fwdLoadI32At(bufT[:], 3*8+h)
	x4 := fwdLoadI32At(bufT[:], 4*8+h)
	x5 := fwdLoadI32At(bufT[:], 5*8+h)
	x6 := fwdLoadI32At(bufT[:], 6*8+h)
	x7 := fwdLoadI32At(bufT[:], 7*8+h)
	c8 := fwdConst8035
	c16 := fwdConst7568
	c24 := fwdConst6811
	c32 := fwdConst5793
	nc32 := fwdConstN5793
	c40 := fwdConst4551
	c48 := fwdConst3135
	c56 := fwdConst1598
	nc8 := fwdConstN8035
	nc16 := fwdConstN7568
	nc40 := fwdConstN4551

	b0 := x0.Add(x7)
	b1 := x1.Add(x6)
	b2 := x2.Add(x5)
	b3 := x3.Add(x4)
	b4 := x3.Sub(x4)
	b5 := x2.Sub(x5)
	b6 := x1.Sub(x6)
	b7 := x0.Sub(x7)

	s0 := b0.Add(b3)
	s1 := b1.Add(b2)
	s2 := b1.Sub(b2)
	s3 := b0.Sub(b3)
	s4 := b4
	s5 := fwdHalfBtf13V(nc32, b5, c32, b6)
	s6 := fwdHalfBtf13V(c32, b6, c32, b5)
	s7 := b7

	b0 = fwdHalfBtf13V(c32, s0, c32, s1)
	b1 = fwdHalfBtf13V(nc32, s1, c32, s0)
	b2 = fwdHalfBtf13V(c48, s2, c16, s3)
	b3 = fwdHalfBtf13V(c48, s3, nc16, s2)
	b4 = s4.Add(s5)
	b5 = s4.Sub(s5)
	b6 = s7.Sub(s6)
	b7 = s7.Add(s6)

	s0 = b0
	s1 = b1
	s2 = b2
	s3 = b3
	s4 = fwdHalfBtf13V(c56, b4, c8, b7)
	s5 = fwdHalfBtf13V(c24, b5, c40, b6)
	s6 = fwdHalfBtf13V(c24, b6, nc40, b5)
	s7 = fwdHalfBtf13V(c56, b7, nc8, b4)
	fwdStoreI32At(coeff[:], 0*8+h, s0)
	fwdStoreI32At(coeff[:], 1*8+h, s4)
	fwdStoreI32At(coeff[:], 2*8+h, s2)
	fwdStoreI32At(coeff[:], 3*8+h, s6)
	fwdStoreI32At(coeff[:], 4*8+h, s1)
	fwdStoreI32At(coeff[:], 5*8+h, s5)
	fwdStoreI32At(coeff[:], 6*8+h, s3)
	fwdStoreI32At(coeff[:], 7*8+h, s7)
}

// fwdRowADST8 is fwdRowDCT8 with the ADST8 row transform.
func fwdRowADST8(coeff *[64]int32, bufT *[64]int32, h int) {
	h &= 8 - fwdLanes
	x0 := fwdLoadI32At(bufT[:], 0*8+h)
	x1 := fwdLoadI32At(bufT[:], 1*8+h)
	x2 := fwdLoadI32At(bufT[:], 2*8+h)
	x3 := fwdLoadI32At(bufT[:], 3*8+h)
	x4 := fwdLoadI32At(bufT[:], 4*8+h)
	x5 := fwdLoadI32At(bufT[:], 5*8+h)
	x6 := fwdLoadI32At(bufT[:], 6*8+h)
	x7 := fwdLoadI32At(bufT[:], 7*8+h)
	c4 := fwdConst8153
	c12 := fwdConst7839
	c16 := fwdConst7568
	c20 := fwdConst7225
	c28 := fwdConst6333
	c32 := fwdConst5793
	nc32 := fwdConstN5793
	c36 := fwdConst5197
	c44 := fwdConst3862
	c48 := fwdConst3135
	nc48 := fwdConstN3135
	c52 := fwdConst2378
	c60 := fwdConst803
	nc4 := fwdConstN8153
	nc20 := fwdConstN7225
	nc16 := fwdConstN7568
	nc36 := fwdConstN5197
	nc52 := fwdConstN2378

	nx3 := x3.Neg()
	nx7 := x7.Neg()
	nx5 := x5.Neg()
	nx1 := x1.Neg()

	s0 := x0
	s1 := nx7
	s2 := fwdHalfBtf13V(c32, nx3, c32, x4)
	s3 := fwdHalfBtf13V(c32, nx3, nc32, x4)
	s4 := nx1
	s5 := x6
	s6 := fwdHalfBtf13V(c32, x2, c32, nx5)
	s7 := fwdHalfBtf13V(c32, x2, nc32, nx5)

	t0 := s0.Add(s2)
	t1 := s1.Add(s3)
	t2 := s0.Sub(s2)
	t3 := s1.Sub(s3)
	t4 := s4.Add(s6)
	t5 := s5.Add(s7)
	t6 := s4.Sub(s6)
	t7 := s5.Sub(s7)

	s4 = fwdHalfBtf13V(c16, t4, c48, t5)
	s5 = fwdHalfBtf13V(c48, t4, nc16, t5)
	s6 = fwdHalfBtf13V(nc48, t6, c16, t7)
	s7 = fwdHalfBtf13V(c16, t6, c48, t7)

	t4 = t0.Sub(s4)
	t5 = t1.Sub(s5)
	t6 = t2.Sub(s6)
	t7 = t3.Sub(s7)
	t0 = t0.Add(s4)
	t1 = t1.Add(s5)
	t2 = t2.Add(s6)
	t3 = t3.Add(s7)

	s0 = fwdHalfBtf13V(c4, t0, c60, t1)
	s1 = fwdHalfBtf13V(c60, t0, nc4, t1)
	s2 = fwdHalfBtf13V(c20, t2, c44, t3)
	s3 = fwdHalfBtf13V(c44, t2, nc20, t3)
	s4 = fwdHalfBtf13V(c36, t4, c28, t5)
	s5 = fwdHalfBtf13V(c28, t4, nc36, t5)
	s6 = fwdHalfBtf13V(c52, t6, c12, t7)
	s7 = fwdHalfBtf13V(c12, t6, nc52, t7)
	fwdStoreI32At(coeff[:], 0*8+h, s1)
	fwdStoreI32At(coeff[:], 1*8+h, s6)
	fwdStoreI32At(coeff[:], 2*8+h, s3)
	fwdStoreI32At(coeff[:], 3*8+h, s4)
	fwdStoreI32At(coeff[:], 4*8+h, s5)
	fwdStoreI32At(coeff[:], 5*8+h, s2)
	fwdStoreI32At(coeff[:], 6*8+h, s7)
	fwdStoreI32At(coeff[:], 7*8+h, s0)
}

// forwardDCT8x8SIMD is the 8x8 DCT_DCT kernel (forwardDCT8x8PureGo). It
// assumes the residual is within the 8-bit range checked by the guarded binding.
func forwardDCT8x8SIMD(coeff []int32, coeffStride int, residual []int16, residualStride int) {
	if coeffStride == 8 && residualStride == 8 {
		fwd8DCTCore((*[64]int32)(coeff[:64]), (*[64]int16)(residual[:64]))
		return
	}
	fwd8x8Staged(coeff, coeffStride, residual, residualStride, false, false)
}

// forwardDCT8x8SIMDGuarded routes residuals outside the 8-bit range to the
// int64 scalar reference.
func forwardDCT8x8SIMDGuarded(coeff []int32, coeffStride int, residual []int16, residualStride int) {
	if !residualFitsMagnitude(residual, residualStride, 8, 8, 255) {
		forwardDCT8x8PureGo(coeff, coeffStride, residual, residualStride)
		return
	}
	forwardDCT8x8SIMD(coeff, coeffStride, residual, residualStride)
}

// forwardBlock8x8ADSTDCTSIMD is the ADST_DCT kernel: ADST on the columns, DCT
// on the rows (forwardBlock8x8ADSTDCTPureGo).
func forwardBlock8x8ADSTDCTSIMD(coeff []int32, coeffStride int, residual []int16, residualStride int, scratch []int32) {
	_ = scratch[63]
	if coeffStride == 8 && residualStride == 8 {
		fwd8ADSTDCTCore((*[64]int32)(coeff[:64]), (*[64]int16)(residual[:64]))
		return
	}
	fwd8x8Staged(coeff, coeffStride, residual, residualStride, true, false)
}

// forwardBlock8x8DCTADSTSIMD is the DCT_ADST kernel: DCT on the columns, ADST
// on the rows (forwardBlock8x8DCTADSTPureGo).
func forwardBlock8x8DCTADSTSIMD(coeff []int32, coeffStride int, residual []int16, residualStride int, scratch []int32) {
	_ = scratch[63]
	if coeffStride == 8 && residualStride == 8 {
		fwd8DCTADSTPacked((*[64]int32)(coeff[:64]), (*[64]int16)(residual[:64]))
		return
	}
	fwd8x8Staged(coeff, coeffStride, residual, residualStride, false, true)
}

// forwardBlock8x8ADSTADSTSIMD is the ADST_ADST kernel (forwardBlock8x8ADSTADSTPureGo).
func forwardBlock8x8ADSTADSTSIMD(coeff []int32, coeffStride int, residual []int16, residualStride int, scratch []int32) {
	_ = scratch[63]
	if coeffStride == 8 && residualStride == 8 {
		fwd8ADSTADSTPacked((*[64]int32)(coeff[:64]), (*[64]int16)(residual[:64]))
		return
	}
	fwd8x8Staged(coeff, coeffStride, residual, residualStride, true, true)
}

// The encoder's 8x8 blocks are packed (both strides 8): the drivers above run
// them in place through fixed 64-entry views, so no row access needs a check.
// fwd8x8Staged takes any other stride through packed copies.
func fwd8x8Staged(coeff []int32, coeffStride int, residual []int16, residualStride int, colADST, rowADST bool) {
	var r [64]int16
	var c [64]int32
	for i := range 8 {
		copy(r[8*i:8*i+8], residual[i*residualStride:i*residualStride+8])
	}
	switch {
	case rowADST:
		fwd8x8NetPacked(&c, &r, colADST, rowADST)
	case colADST:
		fwd8ADSTDCTCore(&c, &r)
	default:
		fwd8DCTCore(&c, &r)
	}
	for i := range 8 {
		copy(coeff[i*coeffStride:i*coeffStride+8], c[8*i:8*i+8])
	}
}

func fwd8DCTADSTPacked(coeff *[64]int32, residual *[64]int16) {
	fwd8x8NetPacked(coeff, residual, false, true)
}

func fwd8ADSTADSTPacked(coeff *[64]int32, residual *[64]int16) {
	fwd8x8NetPacked(coeff, residual, true, true)
}

// fwd8x8NetPacked is the scratch-transpose network on packed 8x8 blocks (the
// arm64 DCT_DCT and ADST_DCT cores are register-resident instead).
func fwd8x8NetPacked(coeff *[64]int32, residual *[64]int16, colADST, rowADST bool) {
	var buf, bufT [64]int32
	for g := 0; g < 8; g += fwdLanes {
		if colADST {
			fwdColADST8(&buf, g, residual)
		} else {
			fwdColDCT8(&buf, g, residual)
		}
	}
	fwdTranspose(bufT[:], buf[:], 8)
	for h := 0; h < 8; h += fwdLanes {
		if rowADST {
			fwdRowADST8(coeff, &bufT, h)
		} else {
			fwdRowDCT8(coeff, &bufT, h)
		}
	}
}

// forwardBlock8x8ADSTDCTSIMDGuarded and its siblings route residuals outside
// the 8-bit range to the int64 scalar reference.
func forwardBlock8x8ADSTDCTSIMDGuarded(coeff []int32, coeffStride int, residual []int16, residualStride int, scratch []int32) {
	if !residualFitsMagnitude(residual, residualStride, 8, 8, 255) {
		forwardBlock8x8ADSTDCTPureGo(coeff, coeffStride, residual, residualStride, scratch)
		return
	}
	forwardBlock8x8ADSTDCTSIMD(coeff, coeffStride, residual, residualStride, scratch)
}

func forwardBlock8x8DCTADSTSIMDGuarded(coeff []int32, coeffStride int, residual []int16, residualStride int, scratch []int32) {
	if !residualFitsMagnitude(residual, residualStride, 8, 8, 255) {
		forwardBlock8x8DCTADSTPureGo(coeff, coeffStride, residual, residualStride, scratch)
		return
	}
	forwardBlock8x8DCTADSTSIMD(coeff, coeffStride, residual, residualStride, scratch)
}

func forwardBlock8x8ADSTADSTSIMDGuarded(coeff []int32, coeffStride int, residual []int16, residualStride int, scratch []int32) {
	if !residualFitsMagnitude(residual, residualStride, 8, 8, 255) {
		forwardBlock8x8ADSTADSTPureGo(coeff, coeffStride, residual, residualStride, scratch)
		return
	}
	forwardBlock8x8ADSTADSTSIMD(coeff, coeffStride, residual, residualStride, scratch)
}

// Go-native SIMD four-lane inverse DCT64 (dav1d's high-bitdepth itx shape,
// src/arm/64/itx16.S): four adjacent columns, or four transposed rows, ride
// the four int32 lanes of every vector. The butterfly kernels are generated
// (dct64_lanes4_gosimd.go) in two flavours: Narrow for clamp bounds within
// +/-2^17 (8- and 10-bit streams, and 12-bit column passes) and Wide for the
// +/-2^19 envelope of every supported bit depth. Both are bit-identical to
// inverseDCT64 on each lane when the inputs are pre-clamped to [min, max],
// which the row staging and the mid-pass clamp (hybrid.go) guarantee.

const (
	itxNarrowBound = 1 << 17
	itxWideBound   = 1 << 19
)

// inverseDCT64Lanes4 picks the kernel for the clamp bounds; it reports false
// when the bounds exceed the int32-exact envelope.
func inverseDCT64Lanes4(p []int32, stride int, min, max int32) bool {
	switch {
	case min >= -itxNarrowBound && max < itxNarrowBound:
		inverseDCT64Lanes4Narrow(p, stride, min, max)
	case min >= -itxWideBound && max < itxWideBound:
		inverseDCT64Lanes4Wide(p, stride, min, max)
	default:
		return false
	}
	return true
}

func inverseDCT64Col4SIMD(buf []int32, rowStride int, min, max int32) {
	if rowStride < 4 || len(buf) < 4 || rowStride > (len(buf)-4)/(dct64Size-1) ||
		!inverseDCT64Lanes4(buf, rowStride, min, max) {
		inverseDCT64Col4PureGo(buf, rowStride, min, max)
	}
}

// inverseDCT64Row4SIMD transposes the four rows into 64 four-lane vectors,
// runs the column kernel on them and transposes back.
func inverseDCT64Row4SIMD(r0, r1, r2, r3 []int32, min, max int32) {
	if len(r0) < dct64Size || len(r1) < dct64Size || len(r2) < dct64Size || len(r3) < dct64Size ||
		min < -itxWideBound || max >= itxWideBound {
		inverseDCT64Row4PureGo(r0, r1, r2, r3, min, max)
		return
	}
	var t [dct64Size * 4]int32
	a, b, c, d := (*[dct64Size]int32)(r0), (*[dct64Size]int32)(r1), (*[dct64Size]int32)(r2), (*[dct64Size]int32)(r3)
	for i := 0; i < dct64Size; i += 4 {
		w := (*[16]int32)(t[4*i : 4*i+16])
		t0, t1, t2, t3 := itxTranspose4(
			archsimd.LoadInt32x4Array((*[4]int32)(a[i:])), archsimd.LoadInt32x4Array((*[4]int32)(b[i:])),
			archsimd.LoadInt32x4Array((*[4]int32)(c[i:])), archsimd.LoadInt32x4Array((*[4]int32)(d[i:])))
		t0.StoreArray((*[4]int32)(w[0:4]))
		t1.StoreArray((*[4]int32)(w[4:8]))
		t2.StoreArray((*[4]int32)(w[8:12]))
		t3.StoreArray((*[4]int32)(w[12:16]))
	}
	inverseDCT64Lanes4(t[:], 4, min, max)
	for i := 0; i < dct64Size; i += 4 {
		w := (*[16]int32)(t[4*i : 4*i+16])
		o0, o1, o2, o3 := itxTranspose4(
			archsimd.LoadInt32x4Array((*[4]int32)(w[0:4])), archsimd.LoadInt32x4Array((*[4]int32)(w[4:8])),
			archsimd.LoadInt32x4Array((*[4]int32)(w[8:12])), archsimd.LoadInt32x4Array((*[4]int32)(w[12:16])))
		o0.StoreArray((*[4]int32)(a[i:]))
		o1.StoreArray((*[4]int32)(b[i:]))
		o2.StoreArray((*[4]int32)(c[i:]))
		o3.StoreArray((*[4]int32)(d[i:]))
	}
}

// itxTranspose4 transposes the 4x4 int32 matrix whose rows are v0..v3.
func itxTranspose4(v0, v1, v2, v3 archsimd.Int32x4) (archsimd.Int32x4, archsimd.Int32x4, archsimd.Int32x4, archsimd.Int32x4) {
	e0 := v0.InterleaveLo(v1).ToBits().ReshapeToUint64s()
	e1 := v0.InterleaveHi(v1).ToBits().ReshapeToUint64s()
	e2 := v2.InterleaveLo(v3).ToBits().ReshapeToUint64s()
	e3 := v2.InterleaveHi(v3).ToBits().ReshapeToUint64s()
	return e0.InterleaveLo(e2).ReshapeToUint32s().BitsToInt32(), e0.InterleaveHi(e2).ReshapeToUint32s().BitsToInt32(),
		e1.InterleaveLo(e3).ReshapeToUint32s().BitsToInt32(), e1.InterleaveHi(e3).ReshapeToUint32s().BitsToInt32()
}

// inverseDCT64Lanes4Low is inverseDCT64Lanes4 for inputs whose entries at
// lowN.. (lowN = 8, 16 or 32) are exactly zero. AV1 never codes DCT64
// coefficients past 32, so lowN = 32 holds for every 64-point transform, as
// in dav1d's dct64 kernels.
func inverseDCT64Lanes4Low(p []int32, stride int, lowN int, min, max int32) bool {
	if min > 0 || max < 0 {
		return inverseDCT64Lanes4(p, stride, min, max)
	}
	switch {
	case min >= -itxNarrowBound && max < itxNarrowBound:
		switch {
		case lowN <= 8:
			inverseDCT64Lanes4NarrowLow8(p, stride, min, max)
		case lowN <= 16:
			inverseDCT64Lanes4NarrowLow16(p, stride, min, max)
		default:
			inverseDCT64Lanes4NarrowLow32(p, stride, min, max)
		}
	case min >= -itxWideBound && max < itxWideBound:
		switch {
		case lowN <= 8:
			inverseDCT64Lanes4WideLow8(p, stride, min, max)
		case lowN <= 16:
			inverseDCT64Lanes4WideLow16(p, stride, min, max)
		default:
			inverseDCT64Lanes4WideLow32(p, stride, min, max)
		}
	default:
		return false
	}
	return true
}

func inverseDCT64LowCol4SIMD(buf []int32, rowStride int, lowN int, min, max int32) {
	if rowStride < 4 || len(buf) < 4 || rowStride > (len(buf)-4)/(dct64Size-1) ||
		!inverseDCT64Lanes4Low(buf, rowStride, lowN, min, max) {
		inverseDCT64Col4PureGo(buf, rowStride, min, max)
	}
}

// inverseDCT64Row4LowSIMD is inverseDCT64Row4SIMD for rows whose inputs at
// lowN.. are zero: only the first lowN inputs are transposed in.
func inverseDCT64Row4LowSIMD(r0, r1, r2, r3 []int32, lowN int, lo, hi int32) {
	if len(r0) < dct64Size || len(r1) < dct64Size || len(r2) < dct64Size || len(r3) < dct64Size ||
		lo < -itxWideBound || hi >= itxWideBound || lo > 0 || hi < 0 {
		inverseDCT64Row4PureGo(r0, r1, r2, r3, lo, hi)
		return
	}
	lowN = min(max(lowN, 8), 32)
	lowN = (lowN + 7) &^ 7
	var t [dct64Size * 4]int32
	a, b, c, d := (*[dct64Size]int32)(r0), (*[dct64Size]int32)(r1), (*[dct64Size]int32)(r2), (*[dct64Size]int32)(r3)
	for i := 0; i < lowN; i += 4 {
		w := (*[16]int32)(t[4*i : 4*i+16])
		t0, t1, t2, t3 := itxTranspose4(
			archsimd.LoadInt32x4Array((*[4]int32)(a[i:])), archsimd.LoadInt32x4Array((*[4]int32)(b[i:])),
			archsimd.LoadInt32x4Array((*[4]int32)(c[i:])), archsimd.LoadInt32x4Array((*[4]int32)(d[i:])))
		t0.StoreArray((*[4]int32)(w[0:4]))
		t1.StoreArray((*[4]int32)(w[4:8]))
		t2.StoreArray((*[4]int32)(w[8:12]))
		t3.StoreArray((*[4]int32)(w[12:16]))
	}
	inverseDCT64Lanes4Low(t[:], 4, lowN, lo, hi)
	for i := 0; i < dct64Size; i += 4 {
		w := (*[16]int32)(t[4*i : 4*i+16])
		o0, o1, o2, o3 := itxTranspose4(
			archsimd.LoadInt32x4Array((*[4]int32)(w[0:4])), archsimd.LoadInt32x4Array((*[4]int32)(w[4:8])),
			archsimd.LoadInt32x4Array((*[4]int32)(w[8:12])), archsimd.LoadInt32x4Array((*[4]int32)(w[12:16])))
		o0.StoreArray((*[4]int32)(a[i:]))
		o1.StoreArray((*[4]int32)(b[i:]))
		o2.StoreArray((*[4]int32)(c[i:]))
		o3.StoreArray((*[4]int32)(d[i:]))
	}
}
