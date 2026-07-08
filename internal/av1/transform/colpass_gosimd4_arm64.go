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
	st := func(k int, v archsimd.Int32x4) { v.StorePart(buf[k*stride : k*stride+4]) }
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
