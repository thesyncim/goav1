// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

// 4-wide Go-native SIMD inverse DCT8 column pass in the int32 domain. Four
// columns are processed per Int32x4. Every rotation rounds its combined
// product-sum EXACTLY ONCE (matching the scalar roundShift), so it is
// byte-identical; the int64 fallback (colpass_gosimd_arm64.go) covers ranges
// where the int32 products could overflow.
//
// For inputs within int16, each product coeff*cospi (cospi < 4096) is < 2^27
// and a two-term sum < 2^28, so int32 never overflows — hence the range guard.

package transform

import "simd/archsimd"

// inverseDCT8Col4SIMD applies inverseDCT8 to four adjacent columns
// buf[k*stride+0..3], byte-for-byte with running inverseDCT8 on each column.
func inverseDCT8Col4SIMD(buf []int32, stride int, min int32, max int32) {
	if min < -(1<<15) || max >= (1 << 15) {
		for col := 0; col < 4; col++ {
			inverseDCT8(buf[col:], stride, min, max)
		}
		return
	}
	minV := archsimd.BroadcastInt32x4(min)
	maxV := archsimd.BroadcastInt32x4(max)
	r8 := archsimd.BroadcastInt32x4(1 << 7)
	r11 := archsimd.BroadcastInt32x4(1 << 10)
	r12 := archsimd.BroadcastInt32x4(1 << 11)
	k181 := archsimd.BroadcastInt32x4(181)
	k1567 := archsimd.BroadcastInt32x4(1567)
	km312 := archsimd.BroadcastInt32x4(3784 - 4096)
	k799 := archsimd.BroadcastInt32x4(799)
	km79 := archsimd.BroadcastInt32x4(4017 - 4096)
	k1703 := archsimd.BroadcastInt32x4(1703)
	k1138 := archsimd.BroadcastInt32x4(1138)

	ld := func(k int) archsimd.Int32x4 { return archsimd.LoadInt32x4Array((*[4]int32)(buf[k*stride:])) }
	st := func(k int, v archsimd.Int32x4) { v.StoreArray((*[4]int32)(buf[k*stride:])) }
	clip := func(v archsimd.Int32x4) archsimd.Int32x4 { return v.Max(minV).Min(maxV) }
	rs8 := func(v archsimd.Int32x4) archsimd.Int32x4 { return v.Add(r8).ShiftAllRight(8) }
	rs11 := func(v archsimd.Int32x4) archsimd.Int32x4 { return v.Add(r11).ShiftAllRight(11) }
	rs12 := func(v archsimd.Int32x4) archsimd.Int32x4 { return v.Add(r12).ShiftAllRight(12) }

	c0, c1, c2, c3 := ld(0), ld(1), ld(2), ld(3)
	c4, c5, c6, c7 := ld(4), ld(5), ld(6), ld(7)

	// inverseDCT4 on even rows (c0,c2,c4,c6) — each roundShift rounds once.
	t0 := rs8(c0.Add(c4).Mul(k181))
	t1 := rs8(c0.Sub(c4).Mul(k181))
	t2 := rs12(c2.Mul(k1567).Sub(c6.Mul(km312))).Sub(c6)
	t3 := rs12(c2.Mul(km312).Add(c6.Mul(k1567))).Add(c2)
	d0 := clip(t0.Add(t3))
	d2 := clip(t1.Add(t2))
	d4 := clip(t1.Sub(t2))
	d6 := clip(t0.Sub(t3))

	// inverseDCT8 odd part (c1,c3,c5,c7).
	t4a := rs12(c1.Mul(k799).Sub(c7.Mul(km79))).Sub(c7)
	t5a := rs11(c5.Mul(k1703).Sub(c3.Mul(k1138)))
	t6a := rs11(c5.Mul(k1138).Add(c3.Mul(k1703)))
	t7a := rs12(c1.Mul(km79).Add(c7.Mul(k799))).Add(c1)
	t4 := clip(t4a.Add(t5a))
	t5c := clip(t4a.Sub(t5a))
	t7 := clip(t7a.Add(t6a))
	t6c := clip(t7a.Sub(t6a))
	t5 := rs8(t6c.Sub(t5c).Mul(k181))
	t6 := rs8(t6c.Add(t5c).Mul(k181))

	st(0, clip(d0.Add(t7)))
	st(1, clip(d2.Add(t6)))
	st(2, clip(d4.Add(t5)))
	st(3, clip(d6.Add(t4)))
	st(4, clip(d6.Sub(t4)))
	st(5, clip(d4.Sub(t5)))
	st(6, clip(d2.Sub(t6)))
	st(7, clip(d0.Sub(t7)))
}

// inverseDCT16Col4SIMD applies inverseDCT8 (even rows) then the 16-point odd
// butterfly to four adjacent columns, byte-for-byte with scalar inverseDCT16,
// in the int32 round-once domain. Falls back to scalar outside int16 range.
func inverseDCT16Col4SIMD(buf []int32, stride int, min int32, max int32) {
	if min < -(1<<15) || max >= (1 << 15) {
		for col := 0; col < 4; col++ {
			inverseDCT16(buf[col:], stride, min, max)
		}
		return
	}
	// Even part: 8-point DCT on rows 0,2,..,14 (stride*2).
	inverseDCT8Col4SIMD(buf, stride<<1, min, max)

	minV := archsimd.BroadcastInt32x4(min)
	maxV := archsimd.BroadcastInt32x4(max)
	r8 := archsimd.BroadcastInt32x4(1 << 7)
	r11 := archsimd.BroadcastInt32x4(1 << 10)
	r12 := archsimd.BroadcastInt32x4(1 << 11)
	kc := func(v int32) archsimd.Int32x4 { return archsimd.BroadcastInt32x4(v) }
	k401, k20, k1583, k1299 := kc(401), kc(4076-4096), kc(1583), kc(1299)
	k1931, k484, k176, k1189 := kc(1931), kc(3612-4096), kc(3920-4096), kc(1189)
	k1567, k312, k181 := kc(1567), kc(3784-4096), kc(181)

	ld := func(k int) archsimd.Int32x4 { return archsimd.LoadInt32x4Array((*[4]int32)(buf[k*stride:])) }
	st := func(k int, v archsimd.Int32x4) { v.StoreArray((*[4]int32)(buf[k*stride:])) }
	clip := func(v archsimd.Int32x4) archsimd.Int32x4 { return v.Max(minV).Min(maxV) }
	rs8 := func(v archsimd.Int32x4) archsimd.Int32x4 { return v.Add(r8).ShiftAllRight(8) }
	rs11 := func(v archsimd.Int32x4) archsimd.Int32x4 { return v.Add(r11).ShiftAllRight(11) }
	rs12 := func(v archsimd.Int32x4) archsimd.Int32x4 { return v.Add(r12).ShiftAllRight(12) }

	in1, in3, in5, in7 := ld(1), ld(3), ld(5), ld(7)
	in9, in11, in13, in15 := ld(9), ld(11), ld(13), ld(15)

	t8a := rs12(in1.Mul(k401).Sub(in15.Mul(k20))).Sub(in15)
	t9a := rs11(in9.Mul(k1583).Sub(in7.Mul(k1299)))
	t10a := rs12(in5.Mul(k1931).Sub(in11.Mul(k484))).Sub(in11)
	t11a := rs12(in13.Mul(k176).Sub(in3.Mul(k1189))).Add(in13)
	t12a := rs12(in13.Mul(k1189).Add(in3.Mul(k176))).Add(in3)
	t13a := rs12(in5.Mul(k484).Add(in11.Mul(k1931))).Add(in5)
	t14a := rs11(in9.Mul(k1299).Add(in7.Mul(k1583)))
	t15a := rs12(in1.Mul(k20).Add(in15.Mul(k401))).Add(in1)

	t8 := clip(t8a.Add(t9a))
	t9 := clip(t8a.Sub(t9a))
	t10 := clip(t11a.Sub(t10a))
	t11 := clip(t11a.Add(t10a))
	t12 := clip(t12a.Add(t13a))
	t13 := clip(t12a.Sub(t13a))
	t14 := clip(t15a.Sub(t14a))
	t15 := clip(t15a.Add(t14a))

	t9a = rs12(t14.Mul(k1567).Sub(t9.Mul(k312))).Sub(t9)
	t14a = rs12(t14.Mul(k312).Add(t9.Mul(k1567))).Add(t14)
	t10a = rs12(t13.Mul(k312).Add(t10.Mul(k1567)).Neg()).Sub(t13)
	t13a = rs12(t13.Mul(k1567).Sub(t10.Mul(k312))).Sub(t10)

	t8a2 := clip(t8.Add(t11))
	t9b := clip(t9a.Add(t10a))
	t10b := clip(t9a.Sub(t10a))
	t11a2 := clip(t8.Sub(t11))
	t12a2 := clip(t15.Sub(t12))
	t13b := clip(t14a.Sub(t13a))
	t14b := clip(t14a.Add(t13a))
	t15a2 := clip(t15.Add(t12))

	t10c := rs8(t13b.Sub(t10b).Mul(k181))
	t13c := rs8(t13b.Add(t10b).Mul(k181))
	t11b := rs8(t12a2.Sub(t11a2).Mul(k181))
	t12b := rs8(t12a2.Add(t11a2).Mul(k181))

	e0, e1, e2, e3 := ld(0), ld(2), ld(4), ld(6)
	e4, e5, e6, e7 := ld(8), ld(10), ld(12), ld(14)

	st(0, clip(e0.Add(t15a2)))
	st(1, clip(e1.Add(t14b)))
	st(2, clip(e2.Add(t13c)))
	st(3, clip(e3.Add(t12b)))
	st(4, clip(e4.Add(t11b)))
	st(5, clip(e5.Add(t10c)))
	st(6, clip(e6.Add(t9b)))
	st(7, clip(e7.Add(t8a2)))
	st(8, clip(e7.Sub(t8a2)))
	st(9, clip(e6.Sub(t9b)))
	st(10, clip(e5.Sub(t10c)))
	st(11, clip(e4.Sub(t11b)))
	st(12, clip(e3.Sub(t12b)))
	st(13, clip(e2.Sub(t13c)))
	st(14, clip(e1.Sub(t14b)))
	st(15, clip(e0.Sub(t15a2)))
}
