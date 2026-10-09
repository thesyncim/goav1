// Code generated from the scalar inverseDCT64 butterfly (dct64row.go) by an
// out-of-tree generator; the stage order follows dav1d's src/arm/64/itx16.S
// (inv_dct_4s_x16 + inv_dct32_odd for the even half, inv_dct64_step1 and
// inv_dct64_step2 for the odd half). DO NOT EDIT by hand without re-running
// the dct64 differential tests.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

//go:build goexperiment.simd && (amd64 || arm64) && !purego

package transform

import (
	"simd/archsimd"
	"unsafe"
)

// itxCoefs holds every butterfly multiplier of the four-lane kernels,
// pre-broadcast so each use is a single vector load.
const (
	kcN1 = iota
	kc1
	kcN5
	kc5
	kcN11
	kcN20
	kc20
	kcN31
	kc31
	kcN44
	kcN60
	kcN79
	kc79
	kcN100
	kc100
	kc101
	kcN123
	kc123
	kcN148
	kcN176
	kc176
	kc201
	kcN207
	kc207
	kcN239
	kcN274
	kcN301
	kc301
	kcN312
	kc312
	kcN351
	kc351
	kcN393
	kc393
	kcN401
	kc401
	kcN437
	kcN484
	kc484
	kc501
	kcN532
	kc532
	kcN583
	kcN601
	kc601
	kcN635
	kcN690
	kc690
	kcN700
	kc700
	kcN747
	kc747
	kcN799
	kc799
	kcN806
	kc806
	kcN867
	kc897
	kcN930
	kc930
	kcN994
	kc994
	kc995
	kcN1061
	kcN1092
	kc1092
	kcN1129
	kcN1189
	kc1189
	kcN1200
	kcN1272
	kc1272
	kc1285
	kcN1345
	kc1345
	kcN1380
	kc1380
	kcN1421
	kcN1474
	kc1474
	kcN1498
	kc1498
	kcN1567
	kc1567
	kcN1576
	kc1576
	kcN1656
	kc1660
	kcN1737
	kc1751
	kcN1820
	kc1820
	kcN1842
	kc1842
	kcN1905
	kc1905
	kcN1931
	kc1931
	kcN1990
	kc1990
	kc2019
	kcN2106
	kc2106
	kcN2191
	kc2191
	kcN2276
	kc2276
	kc2359
	kc2440
	kcN2520
	kc2520
	kcN2598
	kc2598
	kc2675
	kcN2751
	kc2751
	kcN2824
	kc2824
	kc2896
	kc2967
	kc3035
	kcN3102
	kc3102
	kcN3166
	kc3166
	kc3229
	kcN3290
	kc3290
	kcN3349
	kc3349
	kcN3406
	kc3406
	kc3461
	kc3513
	kcN3564
	kc3564
	kcN3612
	kc3612
	kc3659
	kcN3703
	kc3703
	kcN3745
	kc3745
	kcN3784
	kc3784
	kc3822
	kc3857
	kcN3889
	kc3889
	kcN3920
	kc3920
	kc3948
	kcN3973
	kc3973
	kcN3996
	kc3996
	kcN4017
	kc4017
	kc4036
	kc4052
	kcN4065
	kc4065
	kcN4076
	kc4076
	kc4085
	kcN4091
	kc4091
	kcN4095
	kc4095
)

var itxCoefs = [...][4]int32{
	kcN1:    {-1, -1, -1, -1},
	kc1:     {1, 1, 1, 1},
	kcN5:    {-5, -5, -5, -5},
	kc5:     {5, 5, 5, 5},
	kcN11:   {-11, -11, -11, -11},
	kcN20:   {-20, -20, -20, -20},
	kc20:    {20, 20, 20, 20},
	kcN31:   {-31, -31, -31, -31},
	kc31:    {31, 31, 31, 31},
	kcN44:   {-44, -44, -44, -44},
	kcN60:   {-60, -60, -60, -60},
	kcN79:   {-79, -79, -79, -79},
	kc79:    {79, 79, 79, 79},
	kcN100:  {-100, -100, -100, -100},
	kc100:   {100, 100, 100, 100},
	kc101:   {101, 101, 101, 101},
	kcN123:  {-123, -123, -123, -123},
	kc123:   {123, 123, 123, 123},
	kcN148:  {-148, -148, -148, -148},
	kcN176:  {-176, -176, -176, -176},
	kc176:   {176, 176, 176, 176},
	kc201:   {201, 201, 201, 201},
	kcN207:  {-207, -207, -207, -207},
	kc207:   {207, 207, 207, 207},
	kcN239:  {-239, -239, -239, -239},
	kcN274:  {-274, -274, -274, -274},
	kcN301:  {-301, -301, -301, -301},
	kc301:   {301, 301, 301, 301},
	kcN312:  {-312, -312, -312, -312},
	kc312:   {312, 312, 312, 312},
	kcN351:  {-351, -351, -351, -351},
	kc351:   {351, 351, 351, 351},
	kcN393:  {-393, -393, -393, -393},
	kc393:   {393, 393, 393, 393},
	kcN401:  {-401, -401, -401, -401},
	kc401:   {401, 401, 401, 401},
	kcN437:  {-437, -437, -437, -437},
	kcN484:  {-484, -484, -484, -484},
	kc484:   {484, 484, 484, 484},
	kc501:   {501, 501, 501, 501},
	kcN532:  {-532, -532, -532, -532},
	kc532:   {532, 532, 532, 532},
	kcN583:  {-583, -583, -583, -583},
	kcN601:  {-601, -601, -601, -601},
	kc601:   {601, 601, 601, 601},
	kcN635:  {-635, -635, -635, -635},
	kcN690:  {-690, -690, -690, -690},
	kc690:   {690, 690, 690, 690},
	kcN700:  {-700, -700, -700, -700},
	kc700:   {700, 700, 700, 700},
	kcN747:  {-747, -747, -747, -747},
	kc747:   {747, 747, 747, 747},
	kcN799:  {-799, -799, -799, -799},
	kc799:   {799, 799, 799, 799},
	kcN806:  {-806, -806, -806, -806},
	kc806:   {806, 806, 806, 806},
	kcN867:  {-867, -867, -867, -867},
	kc897:   {897, 897, 897, 897},
	kcN930:  {-930, -930, -930, -930},
	kc930:   {930, 930, 930, 930},
	kcN994:  {-994, -994, -994, -994},
	kc994:   {994, 994, 994, 994},
	kc995:   {995, 995, 995, 995},
	kcN1061: {-1061, -1061, -1061, -1061},
	kcN1092: {-1092, -1092, -1092, -1092},
	kc1092:  {1092, 1092, 1092, 1092},
	kcN1129: {-1129, -1129, -1129, -1129},
	kcN1189: {-1189, -1189, -1189, -1189},
	kc1189:  {1189, 1189, 1189, 1189},
	kcN1200: {-1200, -1200, -1200, -1200},
	kcN1272: {-1272, -1272, -1272, -1272},
	kc1272:  {1272, 1272, 1272, 1272},
	kc1285:  {1285, 1285, 1285, 1285},
	kcN1345: {-1345, -1345, -1345, -1345},
	kc1345:  {1345, 1345, 1345, 1345},
	kcN1380: {-1380, -1380, -1380, -1380},
	kc1380:  {1380, 1380, 1380, 1380},
	kcN1421: {-1421, -1421, -1421, -1421},
	kcN1474: {-1474, -1474, -1474, -1474},
	kc1474:  {1474, 1474, 1474, 1474},
	kcN1498: {-1498, -1498, -1498, -1498},
	kc1498:  {1498, 1498, 1498, 1498},
	kcN1567: {-1567, -1567, -1567, -1567},
	kc1567:  {1567, 1567, 1567, 1567},
	kcN1576: {-1576, -1576, -1576, -1576},
	kc1576:  {1576, 1576, 1576, 1576},
	kcN1656: {-1656, -1656, -1656, -1656},
	kc1660:  {1660, 1660, 1660, 1660},
	kcN1737: {-1737, -1737, -1737, -1737},
	kc1751:  {1751, 1751, 1751, 1751},
	kcN1820: {-1820, -1820, -1820, -1820},
	kc1820:  {1820, 1820, 1820, 1820},
	kcN1842: {-1842, -1842, -1842, -1842},
	kc1842:  {1842, 1842, 1842, 1842},
	kcN1905: {-1905, -1905, -1905, -1905},
	kc1905:  {1905, 1905, 1905, 1905},
	kcN1931: {-1931, -1931, -1931, -1931},
	kc1931:  {1931, 1931, 1931, 1931},
	kcN1990: {-1990, -1990, -1990, -1990},
	kc1990:  {1990, 1990, 1990, 1990},
	kc2019:  {2019, 2019, 2019, 2019},
	kcN2106: {-2106, -2106, -2106, -2106},
	kc2106:  {2106, 2106, 2106, 2106},
	kcN2191: {-2191, -2191, -2191, -2191},
	kc2191:  {2191, 2191, 2191, 2191},
	kcN2276: {-2276, -2276, -2276, -2276},
	kc2276:  {2276, 2276, 2276, 2276},
	kc2359:  {2359, 2359, 2359, 2359},
	kc2440:  {2440, 2440, 2440, 2440},
	kcN2520: {-2520, -2520, -2520, -2520},
	kc2520:  {2520, 2520, 2520, 2520},
	kcN2598: {-2598, -2598, -2598, -2598},
	kc2598:  {2598, 2598, 2598, 2598},
	kc2675:  {2675, 2675, 2675, 2675},
	kcN2751: {-2751, -2751, -2751, -2751},
	kc2751:  {2751, 2751, 2751, 2751},
	kcN2824: {-2824, -2824, -2824, -2824},
	kc2824:  {2824, 2824, 2824, 2824},
	kc2896:  {2896, 2896, 2896, 2896},
	kc2967:  {2967, 2967, 2967, 2967},
	kc3035:  {3035, 3035, 3035, 3035},
	kcN3102: {-3102, -3102, -3102, -3102},
	kc3102:  {3102, 3102, 3102, 3102},
	kcN3166: {-3166, -3166, -3166, -3166},
	kc3166:  {3166, 3166, 3166, 3166},
	kc3229:  {3229, 3229, 3229, 3229},
	kcN3290: {-3290, -3290, -3290, -3290},
	kc3290:  {3290, 3290, 3290, 3290},
	kcN3349: {-3349, -3349, -3349, -3349},
	kc3349:  {3349, 3349, 3349, 3349},
	kcN3406: {-3406, -3406, -3406, -3406},
	kc3406:  {3406, 3406, 3406, 3406},
	kc3461:  {3461, 3461, 3461, 3461},
	kc3513:  {3513, 3513, 3513, 3513},
	kcN3564: {-3564, -3564, -3564, -3564},
	kc3564:  {3564, 3564, 3564, 3564},
	kcN3612: {-3612, -3612, -3612, -3612},
	kc3612:  {3612, 3612, 3612, 3612},
	kc3659:  {3659, 3659, 3659, 3659},
	kcN3703: {-3703, -3703, -3703, -3703},
	kc3703:  {3703, 3703, 3703, 3703},
	kcN3745: {-3745, -3745, -3745, -3745},
	kc3745:  {3745, 3745, 3745, 3745},
	kcN3784: {-3784, -3784, -3784, -3784},
	kc3784:  {3784, 3784, 3784, 3784},
	kc3822:  {3822, 3822, 3822, 3822},
	kc3857:  {3857, 3857, 3857, 3857},
	kcN3889: {-3889, -3889, -3889, -3889},
	kc3889:  {3889, 3889, 3889, 3889},
	kcN3920: {-3920, -3920, -3920, -3920},
	kc3920:  {3920, 3920, 3920, 3920},
	kc3948:  {3948, 3948, 3948, 3948},
	kcN3973: {-3973, -3973, -3973, -3973},
	kc3973:  {3973, 3973, 3973, 3973},
	kcN3996: {-3996, -3996, -3996, -3996},
	kc3996:  {3996, 3996, 3996, 3996},
	kcN4017: {-4017, -4017, -4017, -4017},
	kc4017:  {4017, 4017, 4017, 4017},
	kc4036:  {4036, 4036, 4036, 4036},
	kc4052:  {4052, 4052, 4052, 4052},
	kcN4065: {-4065, -4065, -4065, -4065},
	kc4065:  {4065, 4065, 4065, 4065},
	kcN4076: {-4076, -4076, -4076, -4076},
	kc4076:  {4076, 4076, 4076, 4076},
	kc4085:  {4085, 4085, 4085, 4085},
	kcN4091: {-4091, -4091, -4091, -4091},
	kc4091:  {4091, 4091, 4091, 4091},
	kcN4095: {-4095, -4095, -4095, -4095},
	kc4095:  {4095, 4095, 4095, 4095},
}

// itxCoefsPtr keeps the table base in a register: a direct reference to
// itxCoefs would re-materialize the symbol address for every load.
var itxCoefsPtr = &itxCoefs

// inverseDCT64Lanes4Narrow applies inverseDCT64 to four adjacent columns of int32 lanes: lane j
// of row i is p+i*stride+4*j. Every input must lie in [min, max] and
// -1<<17 <= min, max < 1<<17; under that bound every product sum below fits in
// int32 (checked when this file was generated), so the result is
// bit-identical to the scalar kernel run on each lane.
func inverseDCT64Lanes4Narrow(p unsafe.Pointer, stride uintptr, min, max int32) {
	k := itxCoefsPtr
	mn, mx := archsimd.BroadcastInt32x4(min), archsimd.BroadcastInt32x4(max)
	r12, n12 := itxRoundBias(12), itxShiftAmount(12)
	ld := func(i uintptr) archsimd.Int32x4 {
		return archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Add(p, i*stride)))
	}
	st := func(i uintptr, v archsimd.Int32x4) { v.StoreArray((*[4]int32)(unsafe.Add(p, i*stride))) }
	clamp := func(v archsimd.Int32x4) archsimd.Int32x4 { return v.Max(mn).Min(mx) }
	mac := func(acc, x archsimd.Int32x4, c int) archsimd.Int32x4 {
		return itxMulAcc(acc, x, archsimd.LoadInt32x4Array(&k[c]))
	}
	shr12 := func(v archsimd.Int32x4) archsimd.Int32x4 { return itxShr12(v, n12) }

	// dct16 (even inputs)
	in4 := ld(4)
	in60 := ld(60)
	t8_4 := shr12(mac(mac(r12, in4, kc401), in60, kcN4076))
	in36 := ld(36)
	in28 := ld(28)
	t9_4 := shr12(mac(mac(r12, in36, kc3166), in28, kcN2598))
	in20 := ld(20)
	in44 := ld(44)
	t10_4 := shr12(mac(mac(r12, in20, kc1931), in44, kcN3612))
	in52 := ld(52)
	in12 := ld(12)
	t11_4 := shr12(mac(mac(r12, in52, kc3920), in12, kcN1189))
	t12_4 := shr12(mac(mac(r12, in52, kc1189), in12, kc3920))
	t13_4 := shr12(mac(mac(r12, in20, kc3612), in44, kc1931))
	t14_4 := shr12(mac(mac(r12, in36, kc2598), in28, kc3166))
	t15_4 := shr12(mac(mac(r12, in4, kc4076), in60, kc401))
	in8 := ld(8)
	in56 := ld(56)
	t4_5 := shr12(mac(mac(r12, in8, kc799), in56, kcN4017))
	in40 := ld(40)
	in24 := ld(24)
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
	in32 := ld(32)
	t0_6 := shr12(mac(r12, in0.Add(in32), kc2896))
	t1_6 := shr12(mac(r12, in0.Sub(in32), kc2896))
	in16 := ld(16)
	in48 := ld(48)
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

	// dct32 odd half
	in2 := ld(2)
	in62 := ld(62)
	t16_3 := shr12(mac(mac(r12, in2, kc201), in62, kcN4091))
	in34 := ld(34)
	in30 := ld(30)
	t17_3 := shr12(mac(mac(r12, in34, kc3035), in30, kcN2751))
	in18 := ld(18)
	in46 := ld(46)
	t18_3 := shr12(mac(mac(r12, in18, kc1751), in46, kcN3703))
	in50 := ld(50)
	in14 := ld(14)
	t19_3 := shr12(mac(mac(r12, in50, kc3857), in14, kcN1380))
	in10 := ld(10)
	in54 := ld(54)
	t20_3 := shr12(mac(mac(r12, in10, kc995), in54, kcN3973))
	in42 := ld(42)
	in22 := ld(22)
	t21_3 := shr12(mac(mac(r12, in42, kc3513), in22, kcN2106))
	in26 := ld(26)
	in38 := ld(38)
	t22_3 := shr12(mac(mac(r12, in26, kc2440), in38, kcN3290))
	in58 := ld(58)
	in6 := ld(6)
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

	// dct32 outputs (kept for the final butterfly)
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

	// step1: in1/31/17/15 -> t32-35, t60-63
	in1 := ld(1)
	in63 := ld(63)
	t32_2 := shr12(mac(mac(r12, in1, kc101), in63, kcN4095))
	in33 := ld(33)
	in31 := ld(31)
	t33_2 := shr12(mac(mac(r12, in33, kc2967), in31, kcN2824))
	in17 := ld(17)
	in47 := ld(47)
	t34_2 := shr12(mac(mac(r12, in17, kc1660), in47, kcN3745))
	in49 := ld(49)
	in15 := ld(15)
	t35_2 := shr12(mac(mac(r12, in49, kc3822), in15, kcN1474))
	t60_2 := shr12(mac(mac(r12, in49, kc1474), in15, kc3822))
	t61_2 := shr12(mac(mac(r12, in17, kc3745), in47, kc1660))
	t62_2 := shr12(mac(mac(r12, in33, kc2824), in31, kc2967))
	t63_2 := shr12(mac(mac(r12, in1, kc4095), in63, kc101))
	t32_3 := clamp(t32_2.Add(t33_2))
	t33_3 := clamp(t32_2.Sub(t33_2))
	t34_3 := clamp(t35_2.Sub(t34_2))
	t35_3 := clamp(t34_2.Add(t35_2))
	t60_3 := clamp(t60_2.Add(t61_2))
	t61_3 := clamp(t60_2.Sub(t61_2))
	t62_3 := clamp(t63_2.Sub(t62_2))
	t63_3 := clamp(t62_2.Add(t63_2))
	t33_4 := shr12(mac(mac(r12, t33_3, kcN4076), t62_3, kc401))
	t34_4 := shr12(mac(mac(r12, t34_3, kcN401), t61_3, kcN4076))
	t61_4 := shr12(mac(mac(r12, t34_3, kcN4076), t61_3, kc401))
	t62_4 := shr12(mac(mac(r12, t33_3, kc401), t62_3, kc4076))
	t32_5 := clamp(t32_3.Add(t35_3))
	t33_5 := clamp(t33_4.Add(t34_4))
	t34_5 := clamp(t33_4.Sub(t34_4))
	t35_5 := clamp(t32_3.Sub(t35_3))
	t60_5 := clamp(t63_3.Sub(t60_3))
	t61_5 := clamp(t62_4.Sub(t61_4))
	t62_5 := clamp(t61_4.Add(t62_4))
	t63_5 := clamp(t60_3.Add(t63_3))
	t34_6 := shr12(mac(mac(r12, t34_5, kcN4017), t61_5, kc799))
	t35_6 := shr12(mac(mac(r12, t35_5, kcN4017), t60_5, kc799))
	t60_6 := shr12(mac(mac(r12, t35_5, kc799), t60_5, kc4017))
	t61_6 := shr12(mac(mac(r12, t34_5, kc799), t61_5, kc4017))

	// step1: in7/25/23/9 -> t56-59, t36-39
	in9 := ld(9)
	in55 := ld(55)
	t36_2 := shr12(mac(mac(r12, in9, kc897), in55, kcN3996))
	in41 := ld(41)
	in23 := ld(23)
	t37_2 := shr12(mac(mac(r12, in41, kc3461), in23, kcN2191))
	in25 := ld(25)
	in39 := ld(39)
	t38_2 := shr12(mac(mac(r12, in25, kc2359), in39, kcN3349))
	in57 := ld(57)
	in7 := ld(7)
	t39_2 := shr12(mac(mac(r12, in57, kc4036), in7, kcN700))
	t56_2 := shr12(mac(mac(r12, in57, kc700), in7, kc4036))
	t57_2 := shr12(mac(mac(r12, in25, kc3349), in39, kc2359))
	t58_2 := shr12(mac(mac(r12, in41, kc2191), in23, kc3461))
	t59_2 := shr12(mac(mac(r12, in9, kc3996), in55, kc897))
	t36_3 := clamp(t36_2.Add(t37_2))
	t37_3 := clamp(t36_2.Sub(t37_2))
	t38_3 := clamp(t39_2.Sub(t38_2))
	t39_3 := clamp(t38_2.Add(t39_2))
	t56_3 := clamp(t56_2.Add(t57_2))
	t57_3 := clamp(t56_2.Sub(t57_2))
	t58_3 := clamp(t59_2.Sub(t58_2))
	t59_3 := clamp(t58_2.Add(t59_2))
	t37_4 := shr12(mac(mac(r12, t37_3, kcN2598), t58_3, kc3166))
	t38_4 := shr12(mac(mac(r12, t38_3, kcN3166), t57_3, kcN2598))
	t57_4 := shr12(mac(mac(r12, t38_3, kcN2598), t57_3, kc3166))
	t58_4 := shr12(mac(mac(r12, t37_3, kc3166), t58_3, kc2598))
	t36_5 := clamp(t39_3.Sub(t36_3))
	t37_5 := clamp(t38_4.Sub(t37_4))
	t38_5 := clamp(t37_4.Add(t38_4))
	t39_5 := clamp(t36_3.Add(t39_3))
	t56_5 := clamp(t56_3.Add(t59_3))
	t57_5 := clamp(t57_4.Add(t58_4))
	t58_5 := clamp(t57_4.Sub(t58_4))
	t59_5 := clamp(t56_3.Sub(t59_3))
	t36_6 := shr12(mac(mac(r12, t36_5, kcN799), t59_5, kcN4017))
	t37_6 := shr12(mac(mac(r12, t37_5, kcN799), t58_5, kcN4017))
	t58_6 := shr12(mac(mac(r12, t37_5, kcN4017), t58_5, kc799))
	t59_6 := shr12(mac(mac(r12, t36_5, kcN4017), t59_5, kc799))

	// step1: in5/27/21/11 -> t40-43, t52-55
	in5 := ld(5)
	in59 := ld(59)
	t40_2 := shr12(mac(mac(r12, in5, kc501), in59, kcN4065))
	in37 := ld(37)
	in27 := ld(27)
	t41_2 := shr12(mac(mac(r12, in37, kc3229), in27, kcN2520))
	in21 := ld(21)
	in43 := ld(43)
	t42_2 := shr12(mac(mac(r12, in21, kc2019), in43, kcN3564))
	in53 := ld(53)
	in11 := ld(11)
	t43_2 := shr12(mac(mac(r12, in53, kc3948), in11, kcN1092))
	t52_2 := shr12(mac(mac(r12, in53, kc1092), in11, kc3948))
	t53_2 := shr12(mac(mac(r12, in21, kc3564), in43, kc2019))
	t54_2 := shr12(mac(mac(r12, in37, kc2520), in27, kc3229))
	t55_2 := shr12(mac(mac(r12, in5, kc4065), in59, kc501))
	t40_3 := clamp(t40_2.Add(t41_2))
	t41_3 := clamp(t40_2.Sub(t41_2))
	t42_3 := clamp(t43_2.Sub(t42_2))
	t43_3 := clamp(t42_2.Add(t43_2))
	t52_3 := clamp(t52_2.Add(t53_2))
	t53_3 := clamp(t52_2.Sub(t53_2))
	t54_3 := clamp(t55_2.Sub(t54_2))
	t55_3 := clamp(t54_2.Add(t55_2))
	t41_4 := shr12(mac(mac(r12, t41_3, kcN3612), t54_3, kc1931))
	t42_4 := shr12(mac(mac(r12, t42_3, kcN1931), t53_3, kcN3612))
	t53_4 := shr12(mac(mac(r12, t42_3, kcN3612), t53_3, kc1931))
	t54_4 := shr12(mac(mac(r12, t41_3, kc1931), t54_3, kc3612))
	t40_5 := clamp(t40_3.Add(t43_3))
	t41_5 := clamp(t41_4.Add(t42_4))
	t42_5 := clamp(t41_4.Sub(t42_4))
	t43_5 := clamp(t40_3.Sub(t43_3))
	t52_5 := clamp(t55_3.Sub(t52_3))
	t53_5 := clamp(t54_4.Sub(t53_4))
	t54_5 := clamp(t53_4.Add(t54_4))
	t55_5 := clamp(t52_3.Add(t55_3))
	t42_6 := shr12(mac(mac(r12, t42_5, kcN2276), t53_5, kc3406))
	t43_6 := shr12(mac(mac(r12, t43_5, kcN2276), t52_5, kc3406))
	t52_6 := shr12(mac(mac(r12, t43_5, kc3406), t52_5, kc2276))
	t53_6 := shr12(mac(mac(r12, t42_5, kc3406), t53_5, kc2276))

	// step1: in3/29/19/13 -> t48-51, t44-47
	in13 := ld(13)
	in51 := ld(51)
	t44_2 := shr12(mac(mac(r12, in13, kc1285), in51, kcN3889))
	in45 := ld(45)
	in19 := ld(19)
	t45_2 := shr12(mac(mac(r12, in45, kc3659), in19, kcN1842))
	in29 := ld(29)
	in35 := ld(35)
	t46_2 := shr12(mac(mac(r12, in29, kc2675), in35, kcN3102))
	in61 := ld(61)
	in3 := ld(3)
	t47_2 := shr12(mac(mac(r12, in61, kc4085), in3, kcN301))
	t48_2 := shr12(mac(mac(r12, in61, kc301), in3, kc4085))
	t49_2 := shr12(mac(mac(r12, in29, kc3102), in35, kc2675))
	t50_2 := shr12(mac(mac(r12, in45, kc1842), in19, kc3659))
	t51_2 := shr12(mac(mac(r12, in13, kc3889), in51, kc1285))
	t44_3 := clamp(t44_2.Add(t45_2))
	t45_3 := clamp(t44_2.Sub(t45_2))
	t46_3 := clamp(t47_2.Sub(t46_2))
	t47_3 := clamp(t46_2.Add(t47_2))
	t48_3 := clamp(t48_2.Add(t49_2))
	t49_3 := clamp(t48_2.Sub(t49_2))
	t50_3 := clamp(t51_2.Sub(t50_2))
	t51_3 := clamp(t50_2.Add(t51_2))
	t45_4 := shr12(mac(mac(r12, t45_3, kcN1189), t50_3, kc3920))
	t46_4 := shr12(mac(mac(r12, t46_3, kcN3920), t49_3, kcN1189))
	t49_4 := shr12(mac(mac(r12, t46_3, kcN1189), t49_3, kc3920))
	t50_4 := shr12(mac(mac(r12, t45_3, kc3920), t50_3, kc1189))
	t44_5 := clamp(t47_3.Sub(t44_3))
	t45_5 := clamp(t46_4.Sub(t45_4))
	t46_5 := clamp(t45_4.Add(t46_4))
	t47_5 := clamp(t44_3.Add(t47_3))
	t48_5 := clamp(t48_3.Add(t51_3))
	t49_5 := clamp(t49_4.Add(t50_4))
	t50_5 := clamp(t49_4.Sub(t50_4))
	t51_5 := clamp(t48_3.Sub(t51_3))
	t44_6 := shr12(mac(mac(r12, t44_5, kcN3406), t51_5, kcN2276))
	t45_6 := shr12(mac(mac(r12, t45_5, kcN3406), t50_5, kcN2276))
	t50_6 := shr12(mac(mac(r12, t45_5, kcN2276), t50_5, kc3406))
	t51_6 := shr12(mac(mac(r12, t44_5, kcN2276), t51_5, kc3406))

	// step2: t32/39/40/47/48/55/56/63, then outputs
	t32_7 := clamp(t32_5.Add(t39_5))
	t39_7 := clamp(t32_5.Sub(t39_5))
	t40_7 := clamp(t47_5.Sub(t40_5))
	t47_7 := clamp(t40_5.Add(t47_5))
	t48_7 := clamp(t48_5.Add(t55_5))
	t55_7 := clamp(t48_5.Sub(t55_5))
	t56_7 := clamp(t63_5.Sub(t56_5))
	t63_7 := clamp(t56_5.Add(t63_5))
	t39_8 := shr12(mac(mac(r12, t39_7, kcN3784), t56_7, kc1567))
	t40_8 := shr12(mac(mac(r12, t40_7, kcN1567), t55_7, kcN3784))
	t55_8 := shr12(mac(mac(r12, t40_7, kcN3784), t55_7, kc1567))
	t56_8 := shr12(mac(mac(r12, t39_7, kc1567), t56_7, kc3784))
	t32_9 := clamp(t32_7.Add(t47_7))
	t39_9 := clamp(t39_8.Add(t40_8))
	t40_9 := clamp(t39_8.Sub(t40_8))
	t47_9 := clamp(t32_7.Sub(t47_7))
	t48_9 := clamp(t63_7.Sub(t48_7))
	t55_9 := clamp(t56_8.Sub(t55_8))
	t56_9 := clamp(t55_8.Add(t56_8))
	t63_9 := clamp(t48_7.Add(t63_7))
	t40_10 := shr12(mac(r12, t55_9.Sub(t40_9), kc2896))
	t47_10 := shr12(mac(r12, t48_9.Sub(t47_9), kc2896))
	t48_10 := shr12(mac(r12, t47_9.Add(t48_9), kc2896))
	t55_10 := shr12(mac(r12, t40_9.Add(t55_9), kc2896))
	st(0, clamp(t0_10.Add(t63_9)))
	st(7, clamp(t7_10.Add(t56_9)))
	st(8, clamp(t8_10.Add(t55_10)))
	st(15, clamp(t15_10.Add(t48_10)))
	st(16, clamp(t16_10.Add(t47_10)))
	st(23, clamp(t23_10.Add(t40_10)))
	st(24, clamp(t24_10.Add(t39_9)))
	st(31, clamp(t31_10.Add(t32_9)))
	st(32, clamp(t31_10.Sub(t32_9)))
	st(39, clamp(t24_10.Sub(t39_9)))
	st(40, clamp(t23_10.Sub(t40_10)))
	st(47, clamp(t16_10.Sub(t47_10)))
	st(48, clamp(t15_10.Sub(t48_10)))
	st(55, clamp(t8_10.Sub(t55_10)))
	st(56, clamp(t7_10.Sub(t56_9)))
	st(63, clamp(t0_10.Sub(t63_9)))

	// step2: t33/38/41/46/49/54/57/62, then outputs
	t33_7 := clamp(t33_5.Add(t38_5))
	t38_7 := clamp(t33_5.Sub(t38_5))
	t41_7 := clamp(t46_5.Sub(t41_5))
	t46_7 := clamp(t41_5.Add(t46_5))
	t49_7 := clamp(t49_5.Add(t54_5))
	t54_7 := clamp(t49_5.Sub(t54_5))
	t57_7 := clamp(t62_5.Sub(t57_5))
	t62_7 := clamp(t57_5.Add(t62_5))
	t38_8 := shr12(mac(mac(r12, t38_7, kcN3784), t57_7, kc1567))
	t41_8 := shr12(mac(mac(r12, t41_7, kcN1567), t54_7, kcN3784))
	t54_8 := shr12(mac(mac(r12, t41_7, kcN3784), t54_7, kc1567))
	t57_8 := shr12(mac(mac(r12, t38_7, kc1567), t57_7, kc3784))
	t33_9 := clamp(t33_7.Add(t46_7))
	t38_9 := clamp(t38_8.Add(t41_8))
	t41_9 := clamp(t38_8.Sub(t41_8))
	t46_9 := clamp(t33_7.Sub(t46_7))
	t49_9 := clamp(t62_7.Sub(t49_7))
	t54_9 := clamp(t57_8.Sub(t54_8))
	t57_9 := clamp(t54_8.Add(t57_8))
	t62_9 := clamp(t49_7.Add(t62_7))
	t41_10 := shr12(mac(r12, t54_9.Sub(t41_9), kc2896))
	t46_10 := shr12(mac(r12, t49_9.Sub(t46_9), kc2896))
	t49_10 := shr12(mac(r12, t46_9.Add(t49_9), kc2896))
	t54_10 := shr12(mac(r12, t41_9.Add(t54_9), kc2896))
	st(1, clamp(t1_10.Add(t62_9)))
	st(6, clamp(t6_10.Add(t57_9)))
	st(9, clamp(t9_10.Add(t54_10)))
	st(14, clamp(t14_10.Add(t49_10)))
	st(17, clamp(t17_10.Add(t46_10)))
	st(22, clamp(t22_10.Add(t41_10)))
	st(25, clamp(t25_10.Add(t38_9)))
	st(30, clamp(t30_10.Add(t33_9)))
	st(33, clamp(t30_10.Sub(t33_9)))
	st(38, clamp(t25_10.Sub(t38_9)))
	st(41, clamp(t22_10.Sub(t41_10)))
	st(46, clamp(t17_10.Sub(t46_10)))
	st(49, clamp(t14_10.Sub(t49_10)))
	st(54, clamp(t9_10.Sub(t54_10)))
	st(57, clamp(t6_10.Sub(t57_9)))
	st(62, clamp(t1_10.Sub(t62_9)))

	// step2: t34/37/42/45/50/53/58/61, then outputs
	t34_7 := clamp(t34_6.Add(t37_6))
	t37_7 := clamp(t34_6.Sub(t37_6))
	t42_7 := clamp(t45_6.Sub(t42_6))
	t45_7 := clamp(t42_6.Add(t45_6))
	t50_7 := clamp(t50_6.Add(t53_6))
	t53_7 := clamp(t50_6.Sub(t53_6))
	t58_7 := clamp(t61_6.Sub(t58_6))
	t61_7 := clamp(t58_6.Add(t61_6))
	t37_8 := shr12(mac(mac(r12, t37_7, kcN3784), t58_7, kc1567))
	t42_8 := shr12(mac(mac(r12, t42_7, kcN1567), t53_7, kcN3784))
	t53_8 := shr12(mac(mac(r12, t42_7, kcN3784), t53_7, kc1567))
	t58_8 := shr12(mac(mac(r12, t37_7, kc1567), t58_7, kc3784))
	t34_9 := clamp(t34_7.Add(t45_7))
	t37_9 := clamp(t37_8.Add(t42_8))
	t42_9 := clamp(t37_8.Sub(t42_8))
	t45_9 := clamp(t34_7.Sub(t45_7))
	t50_9 := clamp(t61_7.Sub(t50_7))
	t53_9 := clamp(t58_8.Sub(t53_8))
	t58_9 := clamp(t53_8.Add(t58_8))
	t61_9 := clamp(t50_7.Add(t61_7))
	t42_10 := shr12(mac(r12, t53_9.Sub(t42_9), kc2896))
	t45_10 := shr12(mac(r12, t50_9.Sub(t45_9), kc2896))
	t50_10 := shr12(mac(r12, t45_9.Add(t50_9), kc2896))
	t53_10 := shr12(mac(r12, t42_9.Add(t53_9), kc2896))
	st(2, clamp(t2_10.Add(t61_9)))
	st(5, clamp(t5_10.Add(t58_9)))
	st(10, clamp(t10_10.Add(t53_10)))
	st(13, clamp(t13_10.Add(t50_10)))
	st(18, clamp(t18_10.Add(t45_10)))
	st(21, clamp(t21_10.Add(t42_10)))
	st(26, clamp(t26_10.Add(t37_9)))
	st(29, clamp(t29_10.Add(t34_9)))
	st(34, clamp(t29_10.Sub(t34_9)))
	st(37, clamp(t26_10.Sub(t37_9)))
	st(42, clamp(t21_10.Sub(t42_10)))
	st(45, clamp(t18_10.Sub(t45_10)))
	st(50, clamp(t13_10.Sub(t50_10)))
	st(53, clamp(t10_10.Sub(t53_10)))
	st(58, clamp(t5_10.Sub(t58_9)))
	st(61, clamp(t2_10.Sub(t61_9)))

	// step2: t35/36/43/44/51/52/59/60, then outputs
	t35_7 := clamp(t35_6.Add(t36_6))
	t36_7 := clamp(t35_6.Sub(t36_6))
	t43_7 := clamp(t44_6.Sub(t43_6))
	t44_7 := clamp(t43_6.Add(t44_6))
	t51_7 := clamp(t51_6.Add(t52_6))
	t52_7 := clamp(t51_6.Sub(t52_6))
	t59_7 := clamp(t60_6.Sub(t59_6))
	t60_7 := clamp(t59_6.Add(t60_6))
	t36_8 := shr12(mac(mac(r12, t36_7, kcN3784), t59_7, kc1567))
	t43_8 := shr12(mac(mac(r12, t43_7, kcN1567), t52_7, kcN3784))
	t52_8 := shr12(mac(mac(r12, t43_7, kcN3784), t52_7, kc1567))
	t59_8 := shr12(mac(mac(r12, t36_7, kc1567), t59_7, kc3784))
	t35_9 := clamp(t35_7.Add(t44_7))
	t36_9 := clamp(t36_8.Add(t43_8))
	t43_9 := clamp(t36_8.Sub(t43_8))
	t44_9 := clamp(t35_7.Sub(t44_7))
	t51_9 := clamp(t60_7.Sub(t51_7))
	t52_9 := clamp(t59_8.Sub(t52_8))
	t59_9 := clamp(t52_8.Add(t59_8))
	t60_9 := clamp(t51_7.Add(t60_7))
	t43_10 := shr12(mac(r12, t52_9.Sub(t43_9), kc2896))
	t44_10 := shr12(mac(r12, t51_9.Sub(t44_9), kc2896))
	t51_10 := shr12(mac(r12, t44_9.Add(t51_9), kc2896))
	t52_10 := shr12(mac(r12, t43_9.Add(t52_9), kc2896))
	st(3, clamp(t3_10.Add(t60_9)))
	st(4, clamp(t4_10.Add(t59_9)))
	st(11, clamp(t11_10.Add(t52_10)))
	st(12, clamp(t12_10.Add(t51_10)))
	st(19, clamp(t19_10.Add(t44_10)))
	st(20, clamp(t20_10.Add(t43_10)))
	st(27, clamp(t27_10.Add(t36_9)))
	st(28, clamp(t28_10.Add(t35_9)))
	st(35, clamp(t28_10.Sub(t35_9)))
	st(36, clamp(t27_10.Sub(t36_9)))
	st(43, clamp(t20_10.Sub(t43_10)))
	st(44, clamp(t19_10.Sub(t44_10)))
	st(51, clamp(t12_10.Sub(t51_10)))
	st(52, clamp(t11_10.Sub(t52_10)))
	st(59, clamp(t4_10.Sub(t59_9)))
	st(60, clamp(t3_10.Sub(t60_9)))
}

// inverseDCT64Lanes4Wide applies inverseDCT64 to four adjacent columns of int32 lanes: lane j
// of row i is p+i*stride+4*j. Every input must lie in [min, max] and
// -1<<19 <= min, max < 1<<19; under that bound every product sum below fits in
// int32 (checked when this file was generated), so the result is
// bit-identical to the scalar kernel run on each lane.
// Multipliers above 2048 are applied as (w-4096)*x with x added back after
// the shift, an exact identity that keeps the accumulator within int32.
func inverseDCT64Lanes4Wide(p unsafe.Pointer, stride uintptr, min, max int32) {
	k := itxCoefsPtr
	mn, mx := archsimd.BroadcastInt32x4(min), archsimd.BroadcastInt32x4(max)
	r12, n12 := itxRoundBias(12), itxShiftAmount(12)
	ld := func(i uintptr) archsimd.Int32x4 {
		return archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Add(p, i*stride)))
	}
	st := func(i uintptr, v archsimd.Int32x4) { v.StoreArray((*[4]int32)(unsafe.Add(p, i*stride))) }
	clamp := func(v archsimd.Int32x4) archsimd.Int32x4 { return v.Max(mn).Min(mx) }
	mac := func(acc, x archsimd.Int32x4, c int) archsimd.Int32x4 {
		return itxMulAcc(acc, x, archsimd.LoadInt32x4Array(&k[c]))
	}
	shr12 := func(v archsimd.Int32x4) archsimd.Int32x4 { return itxShr12(v, n12) }

	// dct16 (even inputs)
	in4 := ld(4)
	in60 := ld(60)
	t8_4 := shr12(mac(mac(r12, in4, kc401), in60, kc20)).Sub(in60)
	in36 := ld(36)
	in28 := ld(28)
	t9_4 := shr12(mac(mac(r12, in36, kcN930), in28, kc1498)).Add(in36).Sub(in28)
	in20 := ld(20)
	in44 := ld(44)
	t10_4 := shr12(mac(mac(r12, in20, kc1931), in44, kc484)).Sub(in44)
	in52 := ld(52)
	in12 := ld(12)
	t11_4 := shr12(mac(mac(r12, in52, kcN176), in12, kcN1189)).Add(in52)
	t12_4 := shr12(mac(mac(r12, in52, kc1189), in12, kcN176)).Add(in12)
	t13_4 := shr12(mac(mac(r12, in20, kcN484), in44, kc1931)).Add(in20)
	t14_4 := shr12(mac(mac(r12, in36, kcN1498), in28, kcN930)).Add(in36).Add(in28)
	t15_4 := shr12(mac(mac(r12, in4, kcN20), in60, kc401)).Add(in4)
	in8 := ld(8)
	in56 := ld(56)
	t4_5 := shr12(mac(mac(r12, in8, kc799), in56, kc79)).Sub(in56)
	in40 := ld(40)
	in24 := ld(24)
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
	in32 := ld(32)
	d236 := in0.Add(in32)
	t0_6 := shr12(mac(r12, d236, kcN1200)).Add(d236)
	d237 := in0.Sub(in32)
	t1_6 := shr12(mac(r12, d237, kcN1200)).Add(d237)
	in16 := ld(16)
	in48 := ld(48)
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

	// dct32 odd half
	in2 := ld(2)
	in62 := ld(62)
	t16_3 := shr12(mac(mac(r12, in2, kc201), in62, kc5)).Sub(in62)
	in34 := ld(34)
	in30 := ld(30)
	t17_3 := shr12(mac(mac(r12, in34, kcN1061), in30, kc1345)).Add(in34).Sub(in30)
	in18 := ld(18)
	in46 := ld(46)
	t18_3 := shr12(mac(mac(r12, in18, kc1751), in46, kc393)).Sub(in46)
	in50 := ld(50)
	in14 := ld(14)
	t19_3 := shr12(mac(mac(r12, in50, kcN239), in14, kcN1380)).Add(in50)
	in10 := ld(10)
	in54 := ld(54)
	t20_3 := shr12(mac(mac(r12, in10, kc995), in54, kc123)).Sub(in54)
	in42 := ld(42)
	in22 := ld(22)
	t21_3 := shr12(mac(mac(r12, in42, kcN583), in22, kc1990)).Add(in42).Sub(in22)
	in26 := ld(26)
	in38 := ld(38)
	t22_3 := shr12(mac(mac(r12, in26, kcN1656), in38, kc806)).Add(in26).Sub(in38)
	in58 := ld(58)
	in6 := ld(6)
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

	// dct32 outputs (kept for the final butterfly)
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

	// step1: in1/31/17/15 -> t32-35, t60-63
	in1 := ld(1)
	in63 := ld(63)
	t32_2 := shr12(mac(mac(r12, in1, kc101), in63, kc1)).Sub(in63)
	in33 := ld(33)
	in31 := ld(31)
	t33_2 := shr12(mac(mac(r12, in33, kcN1129), in31, kc1272)).Add(in33).Sub(in31)
	in17 := ld(17)
	in47 := ld(47)
	t34_2 := shr12(mac(mac(r12, in17, kc1660), in47, kc351)).Sub(in47)
	in49 := ld(49)
	in15 := ld(15)
	t35_2 := shr12(mac(mac(r12, in49, kcN274), in15, kcN1474)).Add(in49)
	t60_2 := shr12(mac(mac(r12, in49, kc1474), in15, kcN274)).Add(in15)
	t61_2 := shr12(mac(mac(r12, in17, kcN351), in47, kc1660)).Add(in17)
	t62_2 := shr12(mac(mac(r12, in33, kcN1272), in31, kcN1129)).Add(in33).Add(in31)
	t63_2 := shr12(mac(mac(r12, in1, kcN1), in63, kc101)).Add(in1)
	t32_3 := clamp(t32_2.Add(t33_2))
	t33_3 := clamp(t32_2.Sub(t33_2))
	t34_3 := clamp(t35_2.Sub(t34_2))
	t35_3 := clamp(t34_2.Add(t35_2))
	t60_3 := clamp(t60_2.Add(t61_2))
	t61_3 := clamp(t60_2.Sub(t61_2))
	t62_3 := clamp(t63_2.Sub(t62_2))
	t63_3 := clamp(t62_2.Add(t63_2))
	t33_4 := shr12(mac(mac(r12, t33_3, kc20), t62_3, kc401)).Sub(t33_3)
	t34_4 := shr12(mac(mac(r12, t34_3, kcN401), t61_3, kc20)).Sub(t61_3)
	t61_4 := shr12(mac(mac(r12, t34_3, kc20), t61_3, kc401)).Sub(t34_3)
	t62_4 := shr12(mac(mac(r12, t33_3, kc401), t62_3, kcN20)).Add(t62_3)
	t32_5 := clamp(t32_3.Add(t35_3))
	t33_5 := clamp(t33_4.Add(t34_4))
	t34_5 := clamp(t33_4.Sub(t34_4))
	t35_5 := clamp(t32_3.Sub(t35_3))
	t60_5 := clamp(t63_3.Sub(t60_3))
	t61_5 := clamp(t62_4.Sub(t61_4))
	t62_5 := clamp(t61_4.Add(t62_4))
	t63_5 := clamp(t60_3.Add(t63_3))
	t34_6 := shr12(mac(mac(r12, t34_5, kc79), t61_5, kc799)).Sub(t34_5)
	t35_6 := shr12(mac(mac(r12, t35_5, kc79), t60_5, kc799)).Sub(t35_5)
	t60_6 := shr12(mac(mac(r12, t35_5, kc799), t60_5, kcN79)).Add(t60_5)
	t61_6 := shr12(mac(mac(r12, t34_5, kc799), t61_5, kcN79)).Add(t61_5)

	// step1: in7/25/23/9 -> t56-59, t36-39
	in9 := ld(9)
	in55 := ld(55)
	t36_2 := shr12(mac(mac(r12, in9, kc897), in55, kc100)).Sub(in55)
	in41 := ld(41)
	in23 := ld(23)
	t37_2 := shr12(mac(mac(r12, in41, kcN635), in23, kc1905)).Add(in41).Sub(in23)
	in25 := ld(25)
	in39 := ld(39)
	t38_2 := shr12(mac(mac(r12, in25, kcN1737), in39, kc747)).Add(in25).Sub(in39)
	in57 := ld(57)
	in7 := ld(7)
	t39_2 := shr12(mac(mac(r12, in57, kcN60), in7, kcN700)).Add(in57)
	t56_2 := shr12(mac(mac(r12, in57, kc700), in7, kcN60)).Add(in7)
	t57_2 := shr12(mac(mac(r12, in25, kcN747), in39, kcN1737)).Add(in25).Add(in39)
	t58_2 := shr12(mac(mac(r12, in41, kcN1905), in23, kcN635)).Add(in41).Add(in23)
	t59_2 := shr12(mac(mac(r12, in9, kcN100), in55, kc897)).Add(in9)
	t36_3 := clamp(t36_2.Add(t37_2))
	t37_3 := clamp(t36_2.Sub(t37_2))
	t38_3 := clamp(t39_2.Sub(t38_2))
	t39_3 := clamp(t38_2.Add(t39_2))
	t56_3 := clamp(t56_2.Add(t57_2))
	t57_3 := clamp(t56_2.Sub(t57_2))
	t58_3 := clamp(t59_2.Sub(t58_2))
	t59_3 := clamp(t58_2.Add(t59_2))
	t37_4 := shr12(mac(mac(r12, t37_3, kc1498), t58_3, kcN930)).Sub(t37_3).Add(t58_3)
	t38_4 := shr12(mac(mac(r12, t38_3, kc930), t57_3, kc1498)).Sub(t38_3).Sub(t57_3)
	t57_4 := shr12(mac(mac(r12, t38_3, kc1498), t57_3, kcN930)).Sub(t38_3).Add(t57_3)
	t58_4 := shr12(mac(mac(r12, t37_3, kcN930), t58_3, kcN1498)).Add(t37_3).Add(t58_3)
	t36_5 := clamp(t39_3.Sub(t36_3))
	t37_5 := clamp(t38_4.Sub(t37_4))
	t38_5 := clamp(t37_4.Add(t38_4))
	t39_5 := clamp(t36_3.Add(t39_3))
	t56_5 := clamp(t56_3.Add(t59_3))
	t57_5 := clamp(t57_4.Add(t58_4))
	t58_5 := clamp(t57_4.Sub(t58_4))
	t59_5 := clamp(t56_3.Sub(t59_3))
	t36_6 := shr12(mac(mac(r12, t36_5, kcN799), t59_5, kc79)).Sub(t59_5)
	t37_6 := shr12(mac(mac(r12, t37_5, kcN799), t58_5, kc79)).Sub(t58_5)
	t58_6 := shr12(mac(mac(r12, t37_5, kc79), t58_5, kc799)).Sub(t37_5)
	t59_6 := shr12(mac(mac(r12, t36_5, kc79), t59_5, kc799)).Sub(t36_5)

	// step1: in5/27/21/11 -> t40-43, t52-55
	in5 := ld(5)
	in59 := ld(59)
	t40_2 := shr12(mac(mac(r12, in5, kc501), in59, kc31)).Sub(in59)
	in37 := ld(37)
	in27 := ld(27)
	t41_2 := shr12(mac(mac(r12, in37, kcN867), in27, kc1576)).Add(in37).Sub(in27)
	in21 := ld(21)
	in43 := ld(43)
	t42_2 := shr12(mac(mac(r12, in21, kc2019), in43, kc532)).Sub(in43)
	in53 := ld(53)
	in11 := ld(11)
	t43_2 := shr12(mac(mac(r12, in53, kcN148), in11, kcN1092)).Add(in53)
	t52_2 := shr12(mac(mac(r12, in53, kc1092), in11, kcN148)).Add(in11)
	t53_2 := shr12(mac(mac(r12, in21, kcN532), in43, kc2019)).Add(in21)
	t54_2 := shr12(mac(mac(r12, in37, kcN1576), in27, kcN867)).Add(in37).Add(in27)
	t55_2 := shr12(mac(mac(r12, in5, kcN31), in59, kc501)).Add(in5)
	t40_3 := clamp(t40_2.Add(t41_2))
	t41_3 := clamp(t40_2.Sub(t41_2))
	t42_3 := clamp(t43_2.Sub(t42_2))
	t43_3 := clamp(t42_2.Add(t43_2))
	t52_3 := clamp(t52_2.Add(t53_2))
	t53_3 := clamp(t52_2.Sub(t53_2))
	t54_3 := clamp(t55_2.Sub(t54_2))
	t55_3 := clamp(t54_2.Add(t55_2))
	t41_4 := shr12(mac(mac(r12, t41_3, kc484), t54_3, kc1931)).Sub(t41_3)
	t42_4 := shr12(mac(mac(r12, t42_3, kcN1931), t53_3, kc484)).Sub(t53_3)
	t53_4 := shr12(mac(mac(r12, t42_3, kc484), t53_3, kc1931)).Sub(t42_3)
	t54_4 := shr12(mac(mac(r12, t41_3, kc1931), t54_3, kcN484)).Add(t54_3)
	t40_5 := clamp(t40_3.Add(t43_3))
	t41_5 := clamp(t41_4.Add(t42_4))
	t42_5 := clamp(t41_4.Sub(t42_4))
	t43_5 := clamp(t40_3.Sub(t43_3))
	t52_5 := clamp(t55_3.Sub(t52_3))
	t53_5 := clamp(t54_4.Sub(t53_4))
	t54_5 := clamp(t53_4.Add(t54_4))
	t55_5 := clamp(t52_3.Add(t55_3))
	t42_6 := shr12(mac(mac(r12, t42_5, kc1820), t53_5, kcN690)).Sub(t42_5).Add(t53_5)
	t43_6 := shr12(mac(mac(r12, t43_5, kc1820), t52_5, kcN690)).Sub(t43_5).Add(t52_5)
	t52_6 := shr12(mac(mac(r12, t43_5, kcN690), t52_5, kcN1820)).Add(t43_5).Add(t52_5)
	t53_6 := shr12(mac(mac(r12, t42_5, kcN690), t53_5, kcN1820)).Add(t42_5).Add(t53_5)

	// step1: in3/29/19/13 -> t48-51, t44-47
	in13 := ld(13)
	in51 := ld(51)
	t44_2 := shr12(mac(mac(r12, in13, kc1285), in51, kc207)).Sub(in51)
	in45 := ld(45)
	in19 := ld(19)
	t45_2 := shr12(mac(mac(r12, in45, kcN437), in19, kcN1842)).Add(in45)
	in29 := ld(29)
	in35 := ld(35)
	t46_2 := shr12(mac(mac(r12, in29, kcN1421), in35, kc994)).Add(in29).Sub(in35)
	in61 := ld(61)
	in3 := ld(3)
	t47_2 := shr12(mac(mac(r12, in61, kcN11), in3, kcN301)).Add(in61)
	t48_2 := shr12(mac(mac(r12, in61, kc301), in3, kcN11)).Add(in3)
	t49_2 := shr12(mac(mac(r12, in29, kcN994), in35, kcN1421)).Add(in29).Add(in35)
	t50_2 := shr12(mac(mac(r12, in45, kc1842), in19, kcN437)).Add(in19)
	t51_2 := shr12(mac(mac(r12, in13, kcN207), in51, kc1285)).Add(in13)
	t44_3 := clamp(t44_2.Add(t45_2))
	t45_3 := clamp(t44_2.Sub(t45_2))
	t46_3 := clamp(t47_2.Sub(t46_2))
	t47_3 := clamp(t46_2.Add(t47_2))
	t48_3 := clamp(t48_2.Add(t49_2))
	t49_3 := clamp(t48_2.Sub(t49_2))
	t50_3 := clamp(t51_2.Sub(t50_2))
	t51_3 := clamp(t50_2.Add(t51_2))
	t45_4 := shr12(mac(mac(r12, t45_3, kcN1189), t50_3, kcN176)).Add(t50_3)
	t46_4 := shr12(mac(mac(r12, t46_3, kc176), t49_3, kcN1189)).Sub(t46_3)
	t49_4 := shr12(mac(mac(r12, t46_3, kcN1189), t49_3, kcN176)).Add(t49_3)
	t50_4 := shr12(mac(mac(r12, t45_3, kcN176), t50_3, kc1189)).Add(t45_3)
	t44_5 := clamp(t47_3.Sub(t44_3))
	t45_5 := clamp(t46_4.Sub(t45_4))
	t46_5 := clamp(t45_4.Add(t46_4))
	t47_5 := clamp(t44_3.Add(t47_3))
	t48_5 := clamp(t48_3.Add(t51_3))
	t49_5 := clamp(t49_4.Add(t50_4))
	t50_5 := clamp(t49_4.Sub(t50_4))
	t51_5 := clamp(t48_3.Sub(t51_3))
	t44_6 := shr12(mac(mac(r12, t44_5, kc690), t51_5, kc1820)).Sub(t44_5).Sub(t51_5)
	t45_6 := shr12(mac(mac(r12, t45_5, kc690), t50_5, kc1820)).Sub(t45_5).Sub(t50_5)
	t50_6 := shr12(mac(mac(r12, t45_5, kc1820), t50_5, kcN690)).Sub(t45_5).Add(t50_5)
	t51_6 := shr12(mac(mac(r12, t44_5, kc1820), t51_5, kcN690)).Sub(t44_5).Add(t51_5)

	// step2: t32/39/40/47/48/55/56/63, then outputs
	t32_7 := clamp(t32_5.Add(t39_5))
	t39_7 := clamp(t32_5.Sub(t39_5))
	t40_7 := clamp(t47_5.Sub(t40_5))
	t47_7 := clamp(t40_5.Add(t47_5))
	t48_7 := clamp(t48_5.Add(t55_5))
	t55_7 := clamp(t48_5.Sub(t55_5))
	t56_7 := clamp(t63_5.Sub(t56_5))
	t63_7 := clamp(t56_5.Add(t63_5))
	t39_8 := shr12(mac(mac(r12, t39_7, kc312), t56_7, kc1567)).Sub(t39_7)
	t40_8 := shr12(mac(mac(r12, t40_7, kcN1567), t55_7, kc312)).Sub(t55_7)
	t55_8 := shr12(mac(mac(r12, t40_7, kc312), t55_7, kc1567)).Sub(t40_7)
	t56_8 := shr12(mac(mac(r12, t39_7, kc1567), t56_7, kcN312)).Add(t56_7)
	t32_9 := clamp(t32_7.Add(t47_7))
	t39_9 := clamp(t39_8.Add(t40_8))
	t40_9 := clamp(t39_8.Sub(t40_8))
	t47_9 := clamp(t32_7.Sub(t47_7))
	t48_9 := clamp(t63_7.Sub(t48_7))
	t55_9 := clamp(t56_8.Sub(t55_8))
	t56_9 := clamp(t55_8.Add(t56_8))
	t63_9 := clamp(t48_7.Add(t63_7))
	d466 := t55_9.Sub(t40_9)
	t40_10 := shr12(mac(r12, d466, kcN1200)).Add(d466)
	d473 := t48_9.Sub(t47_9)
	t47_10 := shr12(mac(r12, d473, kcN1200)).Add(d473)
	d474 := t47_9.Add(t48_9)
	t48_10 := shr12(mac(r12, d474, kcN1200)).Add(d474)
	d481 := t40_9.Add(t55_9)
	t55_10 := shr12(mac(r12, d481, kcN1200)).Add(d481)
	st(0, clamp(t0_10.Add(t63_9)))
	st(7, clamp(t7_10.Add(t56_9)))
	st(8, clamp(t8_10.Add(t55_10)))
	st(15, clamp(t15_10.Add(t48_10)))
	st(16, clamp(t16_10.Add(t47_10)))
	st(23, clamp(t23_10.Add(t40_10)))
	st(24, clamp(t24_10.Add(t39_9)))
	st(31, clamp(t31_10.Add(t32_9)))
	st(32, clamp(t31_10.Sub(t32_9)))
	st(39, clamp(t24_10.Sub(t39_9)))
	st(40, clamp(t23_10.Sub(t40_10)))
	st(47, clamp(t16_10.Sub(t47_10)))
	st(48, clamp(t15_10.Sub(t48_10)))
	st(55, clamp(t8_10.Sub(t55_10)))
	st(56, clamp(t7_10.Sub(t56_9)))
	st(63, clamp(t0_10.Sub(t63_9)))

	// step2: t33/38/41/46/49/54/57/62, then outputs
	t33_7 := clamp(t33_5.Add(t38_5))
	t38_7 := clamp(t33_5.Sub(t38_5))
	t41_7 := clamp(t46_5.Sub(t41_5))
	t46_7 := clamp(t41_5.Add(t46_5))
	t49_7 := clamp(t49_5.Add(t54_5))
	t54_7 := clamp(t49_5.Sub(t54_5))
	t57_7 := clamp(t62_5.Sub(t57_5))
	t62_7 := clamp(t57_5.Add(t62_5))
	t38_8 := shr12(mac(mac(r12, t38_7, kc312), t57_7, kc1567)).Sub(t38_7)
	t41_8 := shr12(mac(mac(r12, t41_7, kcN1567), t54_7, kc312)).Sub(t54_7)
	t54_8 := shr12(mac(mac(r12, t41_7, kc312), t54_7, kc1567)).Sub(t41_7)
	t57_8 := shr12(mac(mac(r12, t38_7, kc1567), t57_7, kcN312)).Add(t57_7)
	t33_9 := clamp(t33_7.Add(t46_7))
	t38_9 := clamp(t38_8.Add(t41_8))
	t41_9 := clamp(t38_8.Sub(t41_8))
	t46_9 := clamp(t33_7.Sub(t46_7))
	t49_9 := clamp(t62_7.Sub(t49_7))
	t54_9 := clamp(t57_8.Sub(t54_8))
	t57_9 := clamp(t54_8.Add(t57_8))
	t62_9 := clamp(t49_7.Add(t62_7))
	d467 := t54_9.Sub(t41_9)
	t41_10 := shr12(mac(r12, d467, kcN1200)).Add(d467)
	d472 := t49_9.Sub(t46_9)
	t46_10 := shr12(mac(r12, d472, kcN1200)).Add(d472)
	d475 := t46_9.Add(t49_9)
	t49_10 := shr12(mac(r12, d475, kcN1200)).Add(d475)
	d480 := t41_9.Add(t54_9)
	t54_10 := shr12(mac(r12, d480, kcN1200)).Add(d480)
	st(1, clamp(t1_10.Add(t62_9)))
	st(6, clamp(t6_10.Add(t57_9)))
	st(9, clamp(t9_10.Add(t54_10)))
	st(14, clamp(t14_10.Add(t49_10)))
	st(17, clamp(t17_10.Add(t46_10)))
	st(22, clamp(t22_10.Add(t41_10)))
	st(25, clamp(t25_10.Add(t38_9)))
	st(30, clamp(t30_10.Add(t33_9)))
	st(33, clamp(t30_10.Sub(t33_9)))
	st(38, clamp(t25_10.Sub(t38_9)))
	st(41, clamp(t22_10.Sub(t41_10)))
	st(46, clamp(t17_10.Sub(t46_10)))
	st(49, clamp(t14_10.Sub(t49_10)))
	st(54, clamp(t9_10.Sub(t54_10)))
	st(57, clamp(t6_10.Sub(t57_9)))
	st(62, clamp(t1_10.Sub(t62_9)))

	// step2: t34/37/42/45/50/53/58/61, then outputs
	t34_7 := clamp(t34_6.Add(t37_6))
	t37_7 := clamp(t34_6.Sub(t37_6))
	t42_7 := clamp(t45_6.Sub(t42_6))
	t45_7 := clamp(t42_6.Add(t45_6))
	t50_7 := clamp(t50_6.Add(t53_6))
	t53_7 := clamp(t50_6.Sub(t53_6))
	t58_7 := clamp(t61_6.Sub(t58_6))
	t61_7 := clamp(t58_6.Add(t61_6))
	t37_8 := shr12(mac(mac(r12, t37_7, kc312), t58_7, kc1567)).Sub(t37_7)
	t42_8 := shr12(mac(mac(r12, t42_7, kcN1567), t53_7, kc312)).Sub(t53_7)
	t53_8 := shr12(mac(mac(r12, t42_7, kc312), t53_7, kc1567)).Sub(t42_7)
	t58_8 := shr12(mac(mac(r12, t37_7, kc1567), t58_7, kcN312)).Add(t58_7)
	t34_9 := clamp(t34_7.Add(t45_7))
	t37_9 := clamp(t37_8.Add(t42_8))
	t42_9 := clamp(t37_8.Sub(t42_8))
	t45_9 := clamp(t34_7.Sub(t45_7))
	t50_9 := clamp(t61_7.Sub(t50_7))
	t53_9 := clamp(t58_8.Sub(t53_8))
	t58_9 := clamp(t53_8.Add(t58_8))
	t61_9 := clamp(t50_7.Add(t61_7))
	d468 := t53_9.Sub(t42_9)
	t42_10 := shr12(mac(r12, d468, kcN1200)).Add(d468)
	d471 := t50_9.Sub(t45_9)
	t45_10 := shr12(mac(r12, d471, kcN1200)).Add(d471)
	d476 := t45_9.Add(t50_9)
	t50_10 := shr12(mac(r12, d476, kcN1200)).Add(d476)
	d479 := t42_9.Add(t53_9)
	t53_10 := shr12(mac(r12, d479, kcN1200)).Add(d479)
	st(2, clamp(t2_10.Add(t61_9)))
	st(5, clamp(t5_10.Add(t58_9)))
	st(10, clamp(t10_10.Add(t53_10)))
	st(13, clamp(t13_10.Add(t50_10)))
	st(18, clamp(t18_10.Add(t45_10)))
	st(21, clamp(t21_10.Add(t42_10)))
	st(26, clamp(t26_10.Add(t37_9)))
	st(29, clamp(t29_10.Add(t34_9)))
	st(34, clamp(t29_10.Sub(t34_9)))
	st(37, clamp(t26_10.Sub(t37_9)))
	st(42, clamp(t21_10.Sub(t42_10)))
	st(45, clamp(t18_10.Sub(t45_10)))
	st(50, clamp(t13_10.Sub(t50_10)))
	st(53, clamp(t10_10.Sub(t53_10)))
	st(58, clamp(t5_10.Sub(t58_9)))
	st(61, clamp(t2_10.Sub(t61_9)))

	// step2: t35/36/43/44/51/52/59/60, then outputs
	t35_7 := clamp(t35_6.Add(t36_6))
	t36_7 := clamp(t35_6.Sub(t36_6))
	t43_7 := clamp(t44_6.Sub(t43_6))
	t44_7 := clamp(t43_6.Add(t44_6))
	t51_7 := clamp(t51_6.Add(t52_6))
	t52_7 := clamp(t51_6.Sub(t52_6))
	t59_7 := clamp(t60_6.Sub(t59_6))
	t60_7 := clamp(t59_6.Add(t60_6))
	t36_8 := shr12(mac(mac(r12, t36_7, kc312), t59_7, kc1567)).Sub(t36_7)
	t43_8 := shr12(mac(mac(r12, t43_7, kcN1567), t52_7, kc312)).Sub(t52_7)
	t52_8 := shr12(mac(mac(r12, t43_7, kc312), t52_7, kc1567)).Sub(t43_7)
	t59_8 := shr12(mac(mac(r12, t36_7, kc1567), t59_7, kcN312)).Add(t59_7)
	t35_9 := clamp(t35_7.Add(t44_7))
	t36_9 := clamp(t36_8.Add(t43_8))
	t43_9 := clamp(t36_8.Sub(t43_8))
	t44_9 := clamp(t35_7.Sub(t44_7))
	t51_9 := clamp(t60_7.Sub(t51_7))
	t52_9 := clamp(t59_8.Sub(t52_8))
	t59_9 := clamp(t52_8.Add(t59_8))
	t60_9 := clamp(t51_7.Add(t60_7))
	d469 := t52_9.Sub(t43_9)
	t43_10 := shr12(mac(r12, d469, kcN1200)).Add(d469)
	d470 := t51_9.Sub(t44_9)
	t44_10 := shr12(mac(r12, d470, kcN1200)).Add(d470)
	d477 := t44_9.Add(t51_9)
	t51_10 := shr12(mac(r12, d477, kcN1200)).Add(d477)
	d478 := t43_9.Add(t52_9)
	t52_10 := shr12(mac(r12, d478, kcN1200)).Add(d478)
	st(3, clamp(t3_10.Add(t60_9)))
	st(4, clamp(t4_10.Add(t59_9)))
	st(11, clamp(t11_10.Add(t52_10)))
	st(12, clamp(t12_10.Add(t51_10)))
	st(19, clamp(t19_10.Add(t44_10)))
	st(20, clamp(t20_10.Add(t43_10)))
	st(27, clamp(t27_10.Add(t36_9)))
	st(28, clamp(t28_10.Add(t35_9)))
	st(35, clamp(t28_10.Sub(t35_9)))
	st(36, clamp(t27_10.Sub(t36_9)))
	st(43, clamp(t20_10.Sub(t43_10)))
	st(44, clamp(t19_10.Sub(t44_10)))
	st(51, clamp(t12_10.Sub(t51_10)))
	st(52, clamp(t11_10.Sub(t52_10)))
	st(59, clamp(t4_10.Sub(t59_9)))
	st(60, clamp(t3_10.Sub(t60_9)))
}
