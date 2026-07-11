// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

// 4-wide Go-native SIMD inverse DCT32 column pass in the int32 domain, the
// DCT8/DCT16 pattern (colpass_gosimd4_arm64.go) extended to the 32-point odd
// butterfly. Four columns per Int32x4; every rotation rounds its combined
// product-sum exactly once, byte-identical to scalar inverseDCT32. The
// reduced-constant folds (w-4096 with a compensating +/-input term) are the
// scalar reference's own algebra, reproduced verbatim.
//
// Range guard: for inputs within +/-2^15 (the 8- and 10-bit column clamp
// envelope, colBits = max(bd+6,16) = 16) every two-term product-sum is below
// 2^15 * 3812 < 2^27.9, safely inside int32. Wider bounds (12-bit, colBits
// 18) route to the NEON adapter, which owns the +/-2^19 envelope.

package transform

import (
	"simd/archsimd"
	"unsafe"
)

// inverseDCT32Col4SIMD applies inverseDCT32 to four adjacent columns
// buf[k*stride+0..3], byte-for-byte with running inverseDCT32 on each column.
func inverseDCT32Col4SIMD(buf []int32, stride int, min int32, max int32) {
	if min < -(1<<15) || max >= (1<<15) || stride < 4 || len(buf) < (dct32Size-1)*stride+4 {
		inverseDCT32Col4NEONAdapter(buf, stride, min, max)
		return
	}
	// Even part inlined textually (a call would spill the live vector set and
	// rebuild constants; see the skill notes in filter4_gosimd_arm64.go):
	// DCT8 on rows 0,4,..,28, then the DCT16 odd butterfly on rows 2,6,..,30.

	minV := archsimd.BroadcastInt32x4(min)
	maxV := archsimd.BroadcastInt32x4(max)
	// Constants load from rodata at (or near) their use sites -- one LDR each
	// -- instead of 23 MOVD+VDUP broadcasts pinned in registers for the whole
	// body, which fought the ~28 live butterfly values and spilled them.
	// Load the table base through a pointer variable so the address is not a
	// foldable constant: the compiler then keeps the base register-resident
	// and each kt() is one LDR [base, #imm] instead of ADRP+ADD+LDR per use.
	ktb := unsafe.Pointer(dct32KTblPtr)
	kt := func(i int) archsimd.Int32x4 {
		return archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Add(ktb, i*16)))
	}
	r8 := kt(ktR8)
	r11 := kt(ktR11)
	r12 := kt(ktR12)

	// The adapter guard above validated the full 32-row extent, so the hot
	// loads/stores walk a raw base pointer with no per-access bounds checks
	// (62 panicBounds branches otherwise -- the DCT32 body has ~48 row
	// accesses, and the checked form measured ~1.4x slower than the asm).
	base := unsafe.Pointer(&buf[0])
	rowB := stride * 4
	ld := func(k int) archsimd.Int32x4 { return archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Add(base, k*rowB))) }
	st := func(k int, v archsimd.Int32x4) { v.StoreArray((*[4]int32)(unsafe.Add(base, k*rowB))) }
	clip := func(v archsimd.Int32x4) archsimd.Int32x4 { return v.Max(minV).Min(maxV) }

	// ---- DCT8 even core on rows 0,4,..,28 (inverseDCT8Col4SIMD inlined) ----
	{
		c0, c1, c2, c3 := ld(0), ld(4), ld(8), ld(12)
		c4, c5, c6, c7 := ld(16), ld(20), ld(24), ld(28)
		t0 := c0.Add(c4).MulAdd(kt(kt_k181), r8).ShiftAllRightConst(8)
		t1 := c0.Sub(c4).MulAdd(kt(kt_k181), r8).ShiftAllRightConst(8)
		t2 := c6.MulAdd(kt(kt_k312), c2.MulAdd(kt(kt_k1567), r12)).ShiftAllRightConst(12).Sub(c6)
		t3 := c6.MulAdd(kt(kt_k1567), c2.MulAdd(kt(kt_km312), r12)).ShiftAllRightConst(12).Add(c2)
		d0 := clip(t0.Add(t3))
		d2 := clip(t1.Add(t2))
		d4 := clip(t1.Sub(t2))
		d6 := clip(t0.Sub(t3))
		t4a := c7.MulAdd(kt(kt_k79), c1.MulAdd(kt(kt_k799), r12)).ShiftAllRightConst(12).Sub(c7)
		t5a := c3.MulAdd(kt(kt_km1138), c5.MulAdd(kt(kt_k1703), r11)).ShiftAllRightConst(11)
		t6a := c3.MulAdd(kt(kt_k1703), c5.MulAdd(kt(kt_k1138), r11)).ShiftAllRightConst(11)
		t7a := c7.MulAdd(kt(kt_k799), c1.MulAdd(kt(kt_km79), r12)).ShiftAllRightConst(12).Add(c1)
		t4 := clip(t4a.Add(t5a))
		t5c := clip(t4a.Sub(t5a))
		t7 := clip(t7a.Add(t6a))
		t6c := clip(t7a.Sub(t6a))
		t5 := t6c.Sub(t5c).MulAdd(kt(kt_k181), r8).ShiftAllRightConst(8)
		t6 := t6c.Add(t5c).MulAdd(kt(kt_k181), r8).ShiftAllRightConst(8)
		st(0, clip(d0.Add(t7)))
		st(4, clip(d2.Add(t6)))
		st(8, clip(d4.Add(t5)))
		st(12, clip(d6.Add(t4)))
		st(16, clip(d6.Sub(t4)))
		st(20, clip(d4.Sub(t5)))
		st(24, clip(d2.Sub(t6)))
		st(28, clip(d0.Sub(t7)))
	}
	// ---- DCT16 odd butterfly on rows 2,6,..,30 (inverseDCT16Col4SIMD odd
	// part inlined; DCT16-frame row k maps to DCT32 row 2k) ----
	{
		in1, in3, in5, in7 := ld(2), ld(6), ld(10), ld(14)
		in9, in11, in13, in15 := ld(18), ld(22), ld(26), ld(30)
		t8a := in15.MulAdd(kt(kt_k20), in1.MulAdd(kt(kt_k401), r12)).ShiftAllRightConst(12).Sub(in15)
		t9a := in7.MulAdd(kt(kt_km1299), in9.MulAdd(kt(kt_k1583), r11)).ShiftAllRightConst(11)
		t10a := in11.MulAdd(kt(kt_k484), in5.MulAdd(kt(kt_k1931), r12)).ShiftAllRightConst(12).Sub(in11)
		t11a := in3.MulAdd(kt(kt_km1189), in13.MulAdd(kt(kt_km176), r12)).ShiftAllRightConst(12).Add(in13)
		t12a := in3.MulAdd(kt(kt_km176), in13.MulAdd(kt(kt_k1189), r12)).ShiftAllRightConst(12).Add(in3)
		t13a := in11.MulAdd(kt(kt_k1931), in5.MulAdd(kt(kt_km484), r12)).ShiftAllRightConst(12).Add(in5)
		t14a := in7.MulAdd(kt(kt_k1583), in9.MulAdd(kt(kt_k1299), r11)).ShiftAllRightConst(11)
		t15a := in15.MulAdd(kt(kt_k401), in1.MulAdd(kt(kt_km20), r12)).ShiftAllRightConst(12).Add(in1)
		t8 := clip(t8a.Add(t9a))
		t9 := clip(t8a.Sub(t9a))
		t10 := clip(t11a.Sub(t10a))
		t11 := clip(t11a.Add(t10a))
		t12 := clip(t12a.Add(t13a))
		t13 := clip(t12a.Sub(t13a))
		t14 := clip(t15a.Sub(t14a))
		t15 := clip(t15a.Add(t14a))
		t9a = t9.MulAdd(kt(kt_k312), t14.MulAdd(kt(kt_k1567), r12)).ShiftAllRightConst(12).Sub(t9)
		t14a = t9.MulAdd(kt(kt_k1567), t14.MulAdd(kt(kt_km312), r12)).ShiftAllRightConst(12).Add(t14)
		t10a = t10.MulAdd(kt(kt_km1567), t13.MulAdd(kt(kt_k312), r12)).ShiftAllRightConst(12).Sub(t13)
		t13a = t10.MulAdd(kt(kt_k312), t13.MulAdd(kt(kt_k1567), r12)).ShiftAllRightConst(12).Sub(t10)
		t8a2 := clip(t8.Add(t11))
		t9b := clip(t9a.Add(t10a))
		t10b := clip(t9a.Sub(t10a))
		t11a2 := clip(t8.Sub(t11))
		t12a2 := clip(t15.Sub(t12))
		t13b := clip(t14a.Sub(t13a))
		t14b := clip(t14a.Add(t13a))
		t15a2 := clip(t15.Add(t12))
		t10c := t13b.Sub(t10b).MulAdd(kt(kt_k181), r8).ShiftAllRightConst(8)
		t13c := t13b.Add(t10b).MulAdd(kt(kt_k181), r8).ShiftAllRightConst(8)
		t11b := t12a2.Sub(t11a2).MulAdd(kt(kt_k181), r8).ShiftAllRightConst(8)
		t12b := t12a2.Add(t11a2).MulAdd(kt(kt_k181), r8).ShiftAllRightConst(8)
		e0, e1, e2, e3 := ld(0), ld(4), ld(8), ld(12)
		e4, e5, e6, e7 := ld(16), ld(20), ld(24), ld(28)
		st(0, clip(e0.Add(t15a2)))
		st(2, clip(e1.Add(t14b)))
		st(4, clip(e2.Add(t13c)))
		st(6, clip(e3.Add(t12b)))
		st(8, clip(e4.Add(t11b)))
		st(10, clip(e5.Add(t10c)))
		st(12, clip(e6.Add(t9b)))
		st(14, clip(e7.Add(t8a2)))
		st(16, clip(e7.Sub(t8a2)))
		st(18, clip(e6.Sub(t9b)))
		st(20, clip(e5.Sub(t10c)))
		st(22, clip(e4.Sub(t11b)))
		st(24, clip(e3.Sub(t12b)))
		st(26, clip(e2.Sub(t13c)))
		st(28, clip(e1.Sub(t14b)))
		st(30, clip(e0.Sub(t15a2)))
	}

	in1, in3, in5, in7 := ld(1), ld(3), ld(5), ld(7)
	in9, in11, in13, in15 := ld(9), ld(11), ld(13), ld(15)
	in17, in19, in21, in23 := ld(17), ld(19), ld(21), ld(23)
	in25, in27, in29, in31 := ld(25), ld(27), ld(29), ld(31)

	t16a := in31.MulAdd(kt(kt_k5), in1.MulAdd(kt(kt_k201), r12)).ShiftAllRightConst(12).Sub(in31)
	t17a := in15.MulAdd(kt(kt_km2751), in17.MulAdd(kt(kt_k1092), r12)).ShiftAllRightConst(12).Add(in17)
	t18a := in23.MulAdd(kt(kt_k393), in9.MulAdd(kt(kt_k1751), r12)).ShiftAllRightConst(12).Sub(in23)
	t19a := in7.MulAdd(kt(kt_km1380), in25.MulAdd(kt(kt_km239), r12)).ShiftAllRightConst(12).Add(in25)
	t20a := in27.MulAdd(kt(kt_k123), in5.MulAdd(kt(kt_k995), r12)).ShiftAllRightConst(12).Sub(in27)
	t21a := in11.MulAdd(kt(kt_km2106), in21.MulAdd(kt(kt_km583), r12)).ShiftAllRightConst(12).Add(in21)
	t22a := in19.MulAdd(kt(kt_km1645), in13.MulAdd(kt(kt_k1220), r11)).ShiftAllRightConst(11)
	t23a := in3.MulAdd(kt(kt_km601), in29.MulAdd(kt(kt_km44), r12)).ShiftAllRightConst(12).Add(in29)
	t24a := in3.MulAdd(kt(kt_km44), in29.MulAdd(kt(kt_k601), r12)).ShiftAllRightConst(12).Add(in3)
	t25a := in19.MulAdd(kt(kt_k1220), in13.MulAdd(kt(kt_k1645), r11)).ShiftAllRightConst(11)
	t26a := in11.MulAdd(kt(kt_km583), in21.MulAdd(kt(kt_k2106), r12)).ShiftAllRightConst(12).Add(in11)
	t27a := in27.MulAdd(kt(kt_k995), in5.MulAdd(kt(kt_km123), r12)).ShiftAllRightConst(12).Add(in5)
	t28a := in7.MulAdd(kt(kt_km239), in25.MulAdd(kt(kt_k1380), r12)).ShiftAllRightConst(12).Add(in7)
	t29a := in23.MulAdd(kt(kt_k1751), in9.MulAdd(kt(kt_km393), r12)).ShiftAllRightConst(12).Add(in9)
	t30a := in15.MulAdd(kt(kt_k1092), in17.MulAdd(kt(kt_k2751), r12)).ShiftAllRightConst(12).Add(in15)
	t31a := in31.MulAdd(kt(kt_k201), in1.MulAdd(kt(kt_km5), r12)).ShiftAllRightConst(12).Add(in1)

	t16 := clip(t16a.Add(t17a))
	t17 := clip(t16a.Sub(t17a))
	t18 := clip(t19a.Sub(t18a))
	t19 := clip(t19a.Add(t18a))
	t20 := clip(t20a.Add(t21a))
	t21 := clip(t20a.Sub(t21a))
	t22 := clip(t23a.Sub(t22a))
	t23 := clip(t23a.Add(t22a))
	t24 := clip(t24a.Add(t25a))
	t25 := clip(t24a.Sub(t25a))
	t26 := clip(t27a.Sub(t26a))
	t27 := clip(t27a.Add(t26a))
	t28 := clip(t28a.Add(t29a))
	t29 := clip(t28a.Sub(t29a))
	t30 := clip(t31a.Sub(t30a))
	t31 := clip(t31a.Add(t30a))

	t17a = t17.MulAdd(kt(kt_k79), t30.MulAdd(kt(kt_k799), r12)).ShiftAllRightConst(12).Sub(t17)
	t30a = t17.MulAdd(kt(kt_k799), t30.MulAdd(kt(kt_km79), r12)).ShiftAllRightConst(12).Add(t30)
	t18a = t18.MulAdd(kt(kt_km799), t29.MulAdd(kt(kt_k79), r12)).ShiftAllRightConst(12).Sub(t29)
	t29a = t18.MulAdd(kt(kt_k79), t29.MulAdd(kt(kt_k799), r12)).ShiftAllRightConst(12).Sub(t18)
	t21a = t21.MulAdd(kt(kt_km1138), t26.MulAdd(kt(kt_k1703), r11)).ShiftAllRightConst(11)
	t26a = t21.MulAdd(kt(kt_k1703), t26.MulAdd(kt(kt_k1138), r11)).ShiftAllRightConst(11)
	t22a = t22.MulAdd(kt(kt_km1703), t25.MulAdd(kt(kt_km1138), r11)).ShiftAllRightConst(11)
	t25a = t22.MulAdd(kt(kt_km1138), t25.MulAdd(kt(kt_k1703), r11)).ShiftAllRightConst(11)

	t16a = clip(t16.Add(t19))
	t17 = clip(t17a.Add(t18a))
	t18 = clip(t17a.Sub(t18a))
	t19a = clip(t16.Sub(t19))
	t20a = clip(t23.Sub(t20))
	t21 = clip(t22a.Sub(t21a))
	t22 = clip(t22a.Add(t21a))
	t23a = clip(t23.Add(t20))
	t24a = clip(t24.Add(t27))
	t25 = clip(t25a.Add(t26a))
	t26 = clip(t25a.Sub(t26a))
	t27a = clip(t24.Sub(t27))
	t28a = clip(t31.Sub(t28))
	t29 = clip(t30a.Sub(t29a))
	t30 = clip(t30a.Add(t29a))
	t31a = clip(t31.Add(t28))

	t18a = t18.MulAdd(kt(kt_k312), t29.MulAdd(kt(kt_k1567), r12)).ShiftAllRightConst(12).Sub(t18)
	t29a = t18.MulAdd(kt(kt_k1567), t29.MulAdd(kt(kt_km312), r12)).ShiftAllRightConst(12).Add(t29)
	t19 = t19a.MulAdd(kt(kt_k312), t28a.MulAdd(kt(kt_k1567), r12)).ShiftAllRightConst(12).Sub(t19a)
	t28 = t19a.MulAdd(kt(kt_k1567), t28a.MulAdd(kt(kt_km312), r12)).ShiftAllRightConst(12).Add(t28a)
	t20 = t20a.MulAdd(kt(kt_km1567), t27a.MulAdd(kt(kt_k312), r12)).ShiftAllRightConst(12).Sub(t27a)
	t27 = t20a.MulAdd(kt(kt_k312), t27a.MulAdd(kt(kt_k1567), r12)).ShiftAllRightConst(12).Sub(t20a)
	t21a = t21.MulAdd(kt(kt_km1567), t26.MulAdd(kt(kt_k312), r12)).ShiftAllRightConst(12).Sub(t26)
	t26a = t21.MulAdd(kt(kt_k312), t26.MulAdd(kt(kt_k1567), r12)).ShiftAllRightConst(12).Sub(t21)

	t16 = clip(t16a.Add(t23a))
	t17a = clip(t17.Add(t22))
	t18 = clip(t18a.Add(t21a))
	t19a = clip(t19.Add(t20))
	t20a = clip(t19.Sub(t20))
	t21 = clip(t18a.Sub(t21a))
	t22a = clip(t17.Sub(t22))
	t23 = clip(t16a.Sub(t23a))
	t24 = clip(t31a.Sub(t24a))
	t25a = clip(t30.Sub(t25))
	t26 = clip(t29a.Sub(t26a))
	t27a = clip(t28.Sub(t27))
	t28a = clip(t28.Add(t27))
	t29 = clip(t29a.Add(t26a))
	t30a = clip(t30.Add(t25))
	t31 = clip(t31a.Add(t24a))

	t20 = t27a.Sub(t20a).MulAdd(kt(kt_k181), r8).ShiftAllRightConst(8)
	t27 = t27a.Add(t20a).MulAdd(kt(kt_k181), r8).ShiftAllRightConst(8)
	t21a = t26.Sub(t21).MulAdd(kt(kt_k181), r8).ShiftAllRightConst(8)
	t26a = t26.Add(t21).MulAdd(kt(kt_k181), r8).ShiftAllRightConst(8)
	t22 = t25a.Sub(t22a).MulAdd(kt(kt_k181), r8).ShiftAllRightConst(8)
	t25 = t25a.Add(t22a).MulAdd(kt(kt_k181), r8).ShiftAllRightConst(8)
	t23a = t24.Sub(t23).MulAdd(kt(kt_k181), r8).ShiftAllRightConst(8)
	t24a = t24.Add(t23).MulAdd(kt(kt_k181), r8).ShiftAllRightConst(8)

	e0, e1, e2, e3 := ld(0), ld(2), ld(4), ld(6)
	e4, e5, e6, e7 := ld(8), ld(10), ld(12), ld(14)
	st(0, clip(e0.Add(t31)))
	st(1, clip(e1.Add(t30a)))
	st(2, clip(e2.Add(t29)))
	st(3, clip(e3.Add(t28a)))
	st(4, clip(e4.Add(t27)))
	st(5, clip(e5.Add(t26a)))
	st(6, clip(e6.Add(t25)))
	st(7, clip(e7.Add(t24a)))
	e8, e9, e10, e11 := ld(16), ld(18), ld(20), ld(22)
	e12, e13, e14, e15 := ld(24), ld(26), ld(28), ld(30)
	st(8, clip(e8.Add(t23a)))
	st(9, clip(e9.Add(t22)))
	st(10, clip(e10.Add(t21a)))
	st(11, clip(e11.Add(t20)))
	st(12, clip(e12.Add(t19a)))
	st(13, clip(e13.Add(t18)))
	st(14, clip(e14.Add(t17a)))
	st(15, clip(e15.Add(t16)))
	st(16, clip(e15.Sub(t16)))
	st(17, clip(e14.Sub(t17a)))
	st(18, clip(e13.Sub(t18)))
	st(19, clip(e12.Sub(t19a)))
	st(20, clip(e11.Sub(t20)))
	st(21, clip(e10.Sub(t21a)))
	st(22, clip(e9.Sub(t22)))
	st(23, clip(e8.Sub(t23a)))
	st(24, clip(e7.Sub(t24a)))
	st(25, clip(e6.Sub(t25)))
	st(26, clip(e5.Sub(t26a)))
	st(27, clip(e4.Sub(t27)))
	st(28, clip(e3.Sub(t28a)))
	st(29, clip(e2.Sub(t29)))
	st(30, clip(e1.Sub(t30a)))
	st(31, clip(e0.Sub(t31)))
}

// dct32KTbl holds the pre-broadcast rotation constants (reduced w-4096
// forms) and rounding biases; one vector load each at the use site.
var dct32KTbl = [52][4]int32{
	{201, 201, 201, 201},                 // kt_k201
	{-5, -5, -5, -5},                     // kt_km5
	{-1061, -1061, -1061, -1061},         // kt_k1092
	{2751, 2751, 2751, 2751},             // kt_k2751
	{1751, 1751, 1751, 1751},             // kt_k1751
	{-393, -393, -393, -393},             // kt_km393
	{-239, -239, -239, -239},             // kt_km239
	{1380, 1380, 1380, 1380},             // kt_k1380
	{995, 995, 995, 995},                 // kt_k995
	{-123, -123, -123, -123},             // kt_km123
	{-583, -583, -583, -583},             // kt_km583
	{2106, 2106, 2106, 2106},             // kt_k2106
	{1220, 1220, 1220, 1220},             // kt_k1220
	{1645, 1645, 1645, 1645},             // kt_k1645
	{-44, -44, -44, -44},                 // kt_km44
	{601, 601, 601, 601},                 // kt_k601
	{799, 799, 799, 799},                 // kt_k799
	{-79, -79, -79, -79},                 // kt_km79
	{1703, 1703, 1703, 1703},             // kt_k1703
	{1138, 1138, 1138, 1138},             // kt_k1138
	{1567, 1567, 1567, 1567},             // kt_k1567
	{-312, -312, -312, -312},             // kt_km312
	{181, 181, 181, 181},                 // kt_k181
	{401, 401, 401, 401},                 // kt_k401
	{-20, -20, -20, -20},                 // kt_km20
	{1583, 1583, 1583, 1583},             // kt_k1583
	{1299, 1299, 1299, 1299},             // kt_k1299
	{1931, 1931, 1931, 1931},             // kt_k1931
	{-484, -484, -484, -484},             // kt_km484
	{-176, -176, -176, -176},             // kt_km176
	{1189, 1189, 1189, 1189},             // kt_k1189
	{1 << 7, 1 << 7, 1 << 7, 1 << 7},     // ktR8
	{1 << 10, 1 << 10, 1 << 10, 1 << 10}, // ktR11
	{1 << 11, 1 << 11, 1 << 11, 1 << 11}, // ktR12
	{123, 123, 123, 123},                 // kt_k123
	{20, 20, 20, 20},                     // kt_k20
	{312, 312, 312, 312},                 // kt_k312
	{393, 393, 393, 393},                 // kt_k393
	{484, 484, 484, 484},                 // kt_k484
	{5, 5, 5, 5},                         // kt_k5
	{79, 79, 79, 79},                     // kt_k79
	{-1138, -1138, -1138, -1138},         // kt_km1138
	{-1189, -1189, -1189, -1189},         // kt_km1189
	{-1299, -1299, -1299, -1299},         // kt_km1299
	{-1380, -1380, -1380, -1380},         // kt_km1380
	{-1567, -1567, -1567, -1567},         // kt_km1567
	{-1645, -1645, -1645, -1645},         // kt_km1645
	{-1703, -1703, -1703, -1703},         // kt_km1703
	{-2106, -2106, -2106, -2106},         // kt_km2106
	{-2751, -2751, -2751, -2751},         // kt_km2751
	{-601, -601, -601, -601},             // kt_km601
	{-799, -799, -799, -799},             // kt_km799
}

const (
	kt_k201   = 0
	kt_km5    = 1
	kt_k1092  = 2
	kt_k2751  = 3
	kt_k1751  = 4
	kt_km393  = 5
	kt_km239  = 6
	kt_k1380  = 7
	kt_k995   = 8
	kt_km123  = 9
	kt_km583  = 10
	kt_k2106  = 11
	kt_k1220  = 12
	kt_k1645  = 13
	kt_km44   = 14
	kt_k601   = 15
	kt_k799   = 16
	kt_km79   = 17
	kt_k1703  = 18
	kt_k1138  = 19
	kt_k1567  = 20
	kt_km312  = 21
	kt_k181   = 22
	kt_k401   = 23
	kt_km20   = 24
	kt_k1583  = 25
	kt_k1299  = 26
	kt_k1931  = 27
	kt_km484  = 28
	kt_km176  = 29
	kt_k1189  = 30
	ktR8      = 31
	ktR11     = 32
	ktR12     = 33
	kt_k123   = 34
	kt_k20    = 35
	kt_k312   = 36
	kt_k393   = 37
	kt_k484   = 38
	kt_k5     = 39
	kt_k79    = 40
	kt_km1138 = 41
	kt_km1189 = 42
	kt_km1299 = 43
	kt_km1380 = 44
	kt_km1567 = 45
	kt_km1645 = 46
	kt_km1703 = 47
	kt_km2106 = 48
	kt_km2751 = 49
	kt_km601  = 50
	kt_km799  = 51
)

// dct32KTblPtr is the opaque table base for inverseDCT32Col4SIMD (see the
// kt() comment).
var dct32KTblPtr = &dct32KTbl
