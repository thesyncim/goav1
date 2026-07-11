// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

// int16 8-wide Go-native SIMD inverse DCT8 column pass (dav1d technique).
// Eight columns are processed per Int16x8. Two-term rotations accumulate both
// products in int32 via MulWidenLo/MulWidenHi (SMULL/SMULL2), then round ONCE
// and narrow back to int16 with SQRSHRN/SQRSHRN2 (ShiftRightRoundNarrow[Hi]).
// This is byte-identical to the scalar roundShift and processes twice the
// columns of the int32 4-wide path.

package transform

import "simd/archsimd"

// inverseDCT8Col8SIMD applies inverseDCT8 to eight adjacent columns
// buf[k*stride+0..7], byte-for-byte with scalar inverseDCT8, for the int16
// (8-bit) clamp range. Falls back to scalar per column otherwise.
func inverseDCT8Col8SIMD(buf []int32, stride int, min int32, max int32) {
	if min < -(1<<15) || max >= (1 << 15) {
		for col := 0; col < 8; col++ {
			inverseDCT8(buf[col:], stride, min, max)
		}
		return
	}
	minV := archsimd.BroadcastInt16x8(int16(min))
	maxV := archsimd.BroadcastInt16x8(int16(max))

	ld := func(k int) archsimd.Int16x8 {
		lo := archsimd.LoadInt32x4Array((*[4]int32)(buf[k*stride:]))
		hi := archsimd.LoadInt32x4Array((*[4]int32)(buf[k*stride+4:]))
		a := lo.SaturateToInt16().ConvertToUint16().ReshapeToUint64s()
		b := hi.SaturateToInt16().ConvertToUint16().ReshapeToUint64s()
		return a.InterleaveLo(b).ReshapeToUint16s().ConvertToInt16()
	}
	st := func(k int, v archsimd.Int16x8) {
		var arr [8]int16
		v.StoreArray(&arr)
		row := buf[k*stride : k*stride+8]
		for i := 0; i < 8; i++ {
			row[i] = int32(arr[i])
		}
	}
	clip := func(v archsimd.Int16x8) archsimd.Int16x8 { return v.Max(minV).Min(maxV) }
	// mw widens a*c into (lo,hi) int32 accumulators (8 lanes total).
	mw := func(a archsimd.Int16x8, c int16) (archsimd.Int32x4, archsimd.Int32x4) {
		kc := archsimd.BroadcastInt16x8(c)
		return a.MulWidenLo(kc), a.MulWidenHi(kc)
	}
	// nr rounds two int32 accumulators once by n and narrows to int16 (8 lanes).
	nr := func(lo, hi archsimd.Int32x4, n uint8) archsimd.Int16x8 {
		return lo.ShiftRightRoundNarrow(n).ShiftRightRoundNarrowHi(hi, n)
	}

	c0, c1, c2, c3 := ld(0), ld(1), ld(2), ld(3)
	c4, c5, c6, c7 := ld(4), ld(5), ld(6), ld(7)

	// inverseDCT4 on even rows (c0,c2,c4,c6).
	a0l, a0h := mw(c0.Add(c4), 181)
	t0 := nr(a0l, a0h, 8)
	a1l, a1h := mw(c0.Sub(c4), 181)
	t1 := nr(a1l, a1h, 8)
	p2al, p2ah := mw(c2, 1567)
	p2bl, p2bh := mw(c6, 3784-4096)
	t2 := nr(p2al.Sub(p2bl), p2ah.Sub(p2bh), 12).Sub(c6)
	p3al, p3ah := mw(c2, 3784-4096)
	p3bl, p3bh := mw(c6, 1567)
	t3 := nr(p3al.Add(p3bl), p3ah.Add(p3bh), 12).Add(c2)
	d0 := clip(t0.Add(t3))
	d2 := clip(t1.Add(t2))
	d4 := clip(t1.Sub(t2))
	d6 := clip(t0.Sub(t3))

	// inverseDCT8 odd part (c1,c3,c5,c7).
	q4al, q4ah := mw(c1, 799)
	q4bl, q4bh := mw(c7, 4017-4096)
	t4a := nr(q4al.Sub(q4bl), q4ah.Sub(q4bh), 12).Sub(c7)
	q5al, q5ah := mw(c5, 1703)
	q5bl, q5bh := mw(c3, 1138)
	t5a := nr(q5al.Sub(q5bl), q5ah.Sub(q5bh), 11)
	q6al, q6ah := mw(c5, 1138)
	q6bl, q6bh := mw(c3, 1703)
	t6a := nr(q6al.Add(q6bl), q6ah.Add(q6bh), 11)
	q7al, q7ah := mw(c1, 4017-4096)
	q7bl, q7bh := mw(c7, 799)
	t7a := nr(q7al.Add(q7bl), q7ah.Add(q7bh), 12).Add(c1)
	t4 := clip(t4a.Add(t5a))
	t5c := clip(t4a.Sub(t5a))
	t7 := clip(t7a.Add(t6a))
	t6c := clip(t7a.Sub(t6a))
	s5l, s5h := mw(t6c.Sub(t5c), 181)
	t5 := nr(s5l, s5h, 8)
	s6l, s6h := mw(t6c.Add(t5c), 181)
	t6 := nr(s6l, s6h, 8)

	st(0, clip(d0.Add(t7)))
	st(1, clip(d2.Add(t6)))
	st(2, clip(d4.Add(t5)))
	st(3, clip(d6.Add(t4)))
	st(4, clip(d6.Sub(t4)))
	st(5, clip(d4.Sub(t5)))
	st(6, clip(d2.Sub(t6)))
	st(7, clip(d0.Sub(t7)))
}

// inverseDCT8Col8SIMD16 is inverseDCT8Col8SIMD operating directly on an int16
// scratch buffer — no load-narrow or store-widen. This is the form used by the
// dav1d-style int16 column pipeline (8/10-bit); the whole column pass runs in
// int16 so the 8-wide kernel pays off with zero boundary conversion.
func inverseDCT8Col8SIMD16(buf []int16, stride int, min int32, max int32) {
	minV := archsimd.BroadcastInt16x8(int16(min))
	maxV := archsimd.BroadcastInt16x8(int16(max))
	ld := func(k int) archsimd.Int16x8 { return archsimd.LoadInt16x8Array((*[8]int16)(buf[k*stride:])) }
	st := func(k int, v archsimd.Int16x8) { v.StoreArray((*[8]int16)(buf[k*stride:])) }
	clip := func(v archsimd.Int16x8) archsimd.Int16x8 { return v.Max(minV).Min(maxV) }
	mw := func(a archsimd.Int16x8, c int16) (archsimd.Int32x4, archsimd.Int32x4) {
		kc := archsimd.BroadcastInt16x8(c)
		return a.MulWidenLo(kc), a.MulWidenHi(kc)
	}
	nr := func(lo, hi archsimd.Int32x4, n uint8) archsimd.Int16x8 {
		return lo.ShiftRightRoundNarrow(n).ShiftRightRoundNarrowHi(hi, n)
	}
	// int16 butterfly adds saturate, matching dav1d's sqadd/sqsub (byte-exact
	// with the non-saturating scalar when no overflow occurs, i.e. valid decode).
	sadd := func(a, b archsimd.Int16x8) archsimd.Int16x8 { return a.AddSaturated(b) }
	ssub := func(a, b archsimd.Int16x8) archsimd.Int16x8 { return a.SubSaturated(b) }

	c0, c1, c2, c3 := ld(0), ld(1), ld(2), ld(3)
	c4, c5, c6, c7 := ld(4), ld(5), ld(6), ld(7)

	a0l, a0h := mw(sadd(c0, c4), 181)
	t0 := nr(a0l, a0h, 8)
	a1l, a1h := mw(ssub(c0, c4), 181)
	t1 := nr(a1l, a1h, 8)
	p2al, p2ah := mw(c2, 1567)
	p2bl, p2bh := mw(c6, 3784-4096)
	t2 := ssub(nr(p2al.Sub(p2bl), p2ah.Sub(p2bh), 12), c6)
	p3al, p3ah := mw(c2, 3784-4096)
	p3bl, p3bh := mw(c6, 1567)
	t3 := sadd(nr(p3al.Add(p3bl), p3ah.Add(p3bh), 12), c2)
	d0 := clip(sadd(t0, t3))
	d2 := clip(sadd(t1, t2))
	d4 := clip(ssub(t1, t2))
	d6 := clip(ssub(t0, t3))

	q4al, q4ah := mw(c1, 799)
	q4bl, q4bh := mw(c7, 4017-4096)
	t4a := ssub(nr(q4al.Sub(q4bl), q4ah.Sub(q4bh), 12), c7)
	q5al, q5ah := mw(c5, 1703)
	q5bl, q5bh := mw(c3, 1138)
	t5a := nr(q5al.Sub(q5bl), q5ah.Sub(q5bh), 11)
	q6al, q6ah := mw(c5, 1138)
	q6bl, q6bh := mw(c3, 1703)
	t6a := nr(q6al.Add(q6bl), q6ah.Add(q6bh), 11)
	q7al, q7ah := mw(c1, 4017-4096)
	q7bl, q7bh := mw(c7, 799)
	t7a := sadd(nr(q7al.Add(q7bl), q7ah.Add(q7bh), 12), c1)
	t4 := clip(sadd(t4a, t5a))
	t5c := clip(ssub(t4a, t5a))
	t7 := clip(sadd(t7a, t6a))
	t6c := clip(ssub(t7a, t6a))
	s5l, s5h := mw(ssub(t6c, t5c), 181)
	t5 := nr(s5l, s5h, 8)
	s6l, s6h := mw(sadd(t6c, t5c), 181)
	t6 := nr(s6l, s6h, 8)

	st(0, clip(sadd(d0, t7)))
	st(1, clip(sadd(d2, t6)))
	st(2, clip(sadd(d4, t5)))
	st(3, clip(sadd(d6, t4)))
	st(4, clip(ssub(d6, t4)))
	st(5, clip(ssub(d4, t5)))
	st(6, clip(ssub(d2, t6)))
	st(7, clip(ssub(d0, t7)))
}

// inverseDCT16Col8SIMD16 is the int16 8-wide DCT16 column pass (dav1d 8bpc):
// inverseDCT8 on the even rows, then the 16-point odd butterfly, saturating
// int16 adds throughout. Byte-exact with scalar inverseDCT16[int16].
func inverseDCT16Col8SIMD16(buf []int16, stride int, min int32, max int32) {
	inverseDCT8Col8SIMD16(buf, stride<<1, min, max)

	minV := archsimd.BroadcastInt16x8(int16(min))
	maxV := archsimd.BroadcastInt16x8(int16(max))
	ld := func(k int) archsimd.Int16x8 { return archsimd.LoadInt16x8Array((*[8]int16)(buf[k*stride:])) }
	st := func(k int, v archsimd.Int16x8) { v.StoreArray((*[8]int16)(buf[k*stride:])) }
	clip := func(v archsimd.Int16x8) archsimd.Int16x8 { return v.Max(minV).Min(maxV) }
	sadd := func(a, b archsimd.Int16x8) archsimd.Int16x8 { return a.AddSaturated(b) }
	ssub := func(a, b archsimd.Int16x8) archsimd.Int16x8 { return a.SubSaturated(b) }
	mw := func(a archsimd.Int16x8, c int16) (archsimd.Int32x4, archsimd.Int32x4) {
		kc := archsimd.BroadcastInt16x8(c)
		return a.MulWidenLo(kc), a.MulWidenHi(kc)
	}
	nr := func(lo, hi archsimd.Int32x4, n uint8) archsimd.Int16x8 {
		return lo.ShiftRightRoundNarrow(n).ShiftRightRoundNarrowHi(hi, n)
	}
	// roundShift(a*ca - b*cb, n) and roundShift(a*ca + b*cb, n) and the negated
	// two-term; single-term roundShift(a*c, n).
	rSub := func(a archsimd.Int16x8, ca int16, b archsimd.Int16x8, cb int16, n uint8) archsimd.Int16x8 {
		al, ah := mw(a, ca)
		bl, bh := mw(b, cb)
		return nr(al.Sub(bl), ah.Sub(bh), n)
	}
	rAdd := func(a archsimd.Int16x8, ca int16, b archsimd.Int16x8, cb int16, n uint8) archsimd.Int16x8 {
		al, ah := mw(a, ca)
		bl, bh := mw(b, cb)
		return nr(al.Add(bl), ah.Add(bh), n)
	}
	rAddNeg := func(a archsimd.Int16x8, ca int16, b archsimd.Int16x8, cb int16, n uint8) archsimd.Int16x8 {
		al, ah := mw(a, ca)
		bl, bh := mw(b, cb)
		return nr(al.Add(bl).Neg(), ah.Add(bh).Neg(), n)
	}
	r1 := func(a archsimd.Int16x8, c int16, n uint8) archsimd.Int16x8 {
		l, h := mw(a, c)
		return nr(l, h, n)
	}

	in1, in3, in5, in7 := ld(1), ld(3), ld(5), ld(7)
	in9, in11, in13, in15 := ld(9), ld(11), ld(13), ld(15)

	t8a := ssub(rSub(in1, 401, in15, 4076-4096, 12), in15)
	t9a := rSub(in9, 1583, in7, 1299, 11)
	t10a := ssub(rSub(in5, 1931, in11, 3612-4096, 12), in11)
	t11a := sadd(rSub(in13, 3920-4096, in3, 1189, 12), in13)
	t12a := sadd(rAdd(in13, 1189, in3, 3920-4096, 12), in3)
	t13a := sadd(rAdd(in5, 3612-4096, in11, 1931, 12), in5)
	t14a := rAdd(in9, 1299, in7, 1583, 11)
	t15a := sadd(rAdd(in1, 4076-4096, in15, 401, 12), in1)

	t8 := clip(sadd(t8a, t9a))
	t9 := clip(ssub(t8a, t9a))
	t10 := clip(ssub(t11a, t10a))
	t11 := clip(sadd(t11a, t10a))
	t12 := clip(sadd(t12a, t13a))
	t13 := clip(ssub(t12a, t13a))
	t14 := clip(ssub(t15a, t14a))
	t15 := clip(sadd(t15a, t14a))

	t9a = ssub(rSub(t14, 1567, t9, 3784-4096, 12), t9)
	t14a = sadd(rAdd(t14, 3784-4096, t9, 1567, 12), t14)
	t10a = ssub(rAddNeg(t13, 3784-4096, t10, 1567, 12), t13)
	t13a = ssub(rSub(t13, 1567, t10, 3784-4096, 12), t10)

	t8a2 := clip(sadd(t8, t11))
	t9b := clip(sadd(t9a, t10a))
	t10b := clip(ssub(t9a, t10a))
	t11a2 := clip(ssub(t8, t11))
	t12a2 := clip(ssub(t15, t12))
	t13b := clip(ssub(t14a, t13a))
	t14b := clip(sadd(t14a, t13a))
	t15a2 := clip(sadd(t15, t12))

	t10c := r1(ssub(t13b, t10b), 181, 8)
	t13c := r1(sadd(t13b, t10b), 181, 8)
	t11b := r1(ssub(t12a2, t11a2), 181, 8)
	t12b := r1(sadd(t12a2, t11a2), 181, 8)

	e0, e1, e2, e3 := ld(0), ld(2), ld(4), ld(6)
	e4, e5, e6, e7 := ld(8), ld(10), ld(12), ld(14)

	st(0, clip(sadd(e0, t15a2)))
	st(1, clip(sadd(e1, t14b)))
	st(2, clip(sadd(e2, t13c)))
	st(3, clip(sadd(e3, t12b)))
	st(4, clip(sadd(e4, t11b)))
	st(5, clip(sadd(e5, t10c)))
	st(6, clip(sadd(e6, t9b)))
	st(7, clip(sadd(e7, t8a2)))
	st(8, clip(ssub(e7, t8a2)))
	st(9, clip(ssub(e6, t9b)))
	st(10, clip(ssub(e5, t10c)))
	st(11, clip(ssub(e4, t11b)))
	st(12, clip(ssub(e3, t12b)))
	st(13, clip(ssub(e2, t13c)))
	st(14, clip(ssub(e1, t14b)))
	st(15, clip(ssub(e0, t15a2)))
}

func inverseDCT32Col8SIMD16(buf []int16, stride int, min int32, max int32) {
	inverseDCT16Col8SIMD16(buf, stride<<1, min, max)

	minV := archsimd.BroadcastInt16x8(int16(min))
	maxV := archsimd.BroadcastInt16x8(int16(max))
	ld := func(k int) archsimd.Int16x8 { return archsimd.LoadInt16x8Array((*[8]int16)(buf[k*stride:])) }
	st := func(k int, v archsimd.Int16x8) { v.StoreArray((*[8]int16)(buf[k*stride:])) }
	clip := func(v archsimd.Int16x8) archsimd.Int16x8 { return v.Max(minV).Min(maxV) }
	sadd := func(a, b archsimd.Int16x8) archsimd.Int16x8 { return a.AddSaturated(b) }
	ssub := func(a, b archsimd.Int16x8) archsimd.Int16x8 { return a.SubSaturated(b) }
	mw := func(a archsimd.Int16x8, c int16) (archsimd.Int32x4, archsimd.Int32x4) {
		kc := archsimd.BroadcastInt16x8(c)
		return a.MulWidenLo(kc), a.MulWidenHi(kc)
	}
	nr := func(lo, hi archsimd.Int32x4, n uint8) archsimd.Int16x8 {
		return lo.ShiftRightRoundNarrow(n).ShiftRightRoundNarrowHi(hi, n)
	}
	rSub := func(a archsimd.Int16x8, ca int16, b archsimd.Int16x8, cb int16, n uint8) archsimd.Int16x8 {
		al, ah := mw(a, ca)
		bl, bh := mw(b, cb)
		return nr(al.Sub(bl), ah.Sub(bh), n)
	}
	rAdd := func(a archsimd.Int16x8, ca int16, b archsimd.Int16x8, cb int16, n uint8) archsimd.Int16x8 {
		al, ah := mw(a, ca)
		bl, bh := mw(b, cb)
		return nr(al.Add(bl), ah.Add(bh), n)
	}
	rAddNeg := func(a archsimd.Int16x8, ca int16, b archsimd.Int16x8, cb int16, n uint8) archsimd.Int16x8 {
		al, ah := mw(a, ca)
		bl, bh := mw(b, cb)
		return nr(al.Add(bl).Neg(), ah.Add(bh).Neg(), n)
	}
	r1 := func(a archsimd.Int16x8, c int16, n uint8) archsimd.Int16x8 {
		l, h := mw(a, c)
		return nr(l, h, n)
	}

	in1, in3, in5, in7 := ld(1), ld(3), ld(5), ld(7)
	in9, in11, in13, in15 := ld(9), ld(11), ld(13), ld(15)
	in17, in19, in21, in23 := ld(17), ld(19), ld(21), ld(23)
	in25, in27, in29, in31 := ld(25), ld(27), ld(29), ld(31)

	t16a := ssub(rSub(in1, 201, in31, 4091-4096, 12), in31)
	t17a := sadd(rSub(in17, 3035-4096, in15, 2751, 12), in17)
	t18a := ssub(rSub(in9, 1751, in23, 3703-4096, 12), in23)
	t19a := sadd(rSub(in25, 3857-4096, in7, 1380, 12), in25)
	t20a := ssub(rSub(in5, 995, in27, 3973-4096, 12), in27)
	t21a := sadd(rSub(in21, 3513-4096, in11, 2106, 12), in21)
	t22a := rSub(in13, 1220, in19, 1645, 11)
	t23a := sadd(rSub(in29, 4052-4096, in3, 601, 12), in29)
	t24a := sadd(rAdd(in29, 601, in3, 4052-4096, 12), in3)
	t25a := rAdd(in13, 1645, in19, 1220, 11)
	t26a := sadd(rAdd(in21, 2106, in11, 3513-4096, 12), in11)
	t27a := sadd(rAdd(in5, 3973-4096, in27, 995, 12), in5)
	t28a := sadd(rAdd(in25, 1380, in7, 3857-4096, 12), in7)
	t29a := sadd(rAdd(in9, 3703-4096, in23, 1751, 12), in9)
	t30a := sadd(rAdd(in17, 2751, in15, 3035-4096, 12), in15)
	t31a := sadd(rAdd(in1, 4091-4096, in31, 201, 12), in1)

	t16 := clip(sadd(t16a, t17a))
	t17 := clip(ssub(t16a, t17a))
	t18 := clip(ssub(t19a, t18a))
	t19 := clip(sadd(t19a, t18a))
	t20 := clip(sadd(t20a, t21a))
	t21 := clip(ssub(t20a, t21a))
	t22 := clip(ssub(t23a, t22a))
	t23 := clip(sadd(t23a, t22a))
	t24 := clip(sadd(t24a, t25a))
	t25 := clip(ssub(t24a, t25a))
	t26 := clip(ssub(t27a, t26a))
	t27 := clip(sadd(t27a, t26a))
	t28 := clip(sadd(t28a, t29a))
	t29 := clip(ssub(t28a, t29a))
	t30 := clip(ssub(t31a, t30a))
	t31 := clip(sadd(t31a, t30a))

	t17a = ssub(rSub(t30, 799, t17, 4017-4096, 12), t17)
	t30a = sadd(rAdd(t30, 4017-4096, t17, 799, 12), t30)
	t18a = ssub(rAddNeg(t29, 4017-4096, t18, 799, 12), t29)
	t29a = ssub(rSub(t29, 799, t18, 4017-4096, 12), t18)
	t21a = rSub(t26, 1703, t21, 1138, 11)
	t26a = rAdd(t26, 1138, t21, 1703, 11)
	t22a = rAddNeg(t25, 1138, t22, 1703, 11)
	t25a = rSub(t25, 1703, t22, 1138, 11)

	t16a = clip(sadd(t16, t19))
	t17 = clip(sadd(t17a, t18a))
	t18 = clip(ssub(t17a, t18a))
	t19a = clip(ssub(t16, t19))
	t20a = clip(ssub(t23, t20))
	t21 = clip(ssub(t22a, t21a))
	t22 = clip(sadd(t22a, t21a))
	t23a = clip(sadd(t23, t20))
	t24a = clip(sadd(t24, t27))
	t25 = clip(sadd(t25a, t26a))
	t26 = clip(ssub(t25a, t26a))
	t27a = clip(ssub(t24, t27))
	t28a = clip(ssub(t31, t28))
	t29 = clip(ssub(t30a, t29a))
	t30 = clip(sadd(t30a, t29a))
	t31a = clip(sadd(t31, t28))

	t18a = ssub(rSub(t29, 1567, t18, 3784-4096, 12), t18)
	t29a = sadd(rAdd(t29, 3784-4096, t18, 1567, 12), t29)
	t19 = ssub(rSub(t28a, 1567, t19a, 3784-4096, 12), t19a)
	t28 = sadd(rAdd(t28a, 3784-4096, t19a, 1567, 12), t28a)
	t20 = ssub(rAddNeg(t27a, 3784-4096, t20a, 1567, 12), t27a)
	t27 = ssub(rSub(t27a, 1567, t20a, 3784-4096, 12), t20a)
	t21a = ssub(rAddNeg(t26, 3784-4096, t21, 1567, 12), t26)
	t26a = ssub(rSub(t26, 1567, t21, 3784-4096, 12), t21)

	t16 = clip(sadd(t16a, t23a))
	t17a = clip(sadd(t17, t22))
	t18 = clip(sadd(t18a, t21a))
	t19a = clip(sadd(t19, t20))
	t20a = clip(ssub(t19, t20))
	t21 = clip(ssub(t18a, t21a))
	t22a = clip(ssub(t17, t22))
	t23 = clip(ssub(t16a, t23a))
	t24 = clip(ssub(t31a, t24a))
	t25a = clip(ssub(t30, t25))
	t26 = clip(ssub(t29a, t26a))
	t27a = clip(ssub(t28, t27))
	t28a = clip(sadd(t28, t27))
	t29 = clip(sadd(t29a, t26a))
	t30a = clip(sadd(t30, t25))
	t31 = clip(sadd(t31a, t24a))

	t20 = r1(ssub(t27a, t20a), 181, 8)
	t27 = r1(sadd(t27a, t20a), 181, 8)
	t21a = r1(ssub(t26, t21), 181, 8)
	t26a = r1(sadd(t26, t21), 181, 8)
	t22 = r1(ssub(t25a, t22a), 181, 8)
	t25 = r1(sadd(t25a, t22a), 181, 8)
	t23a = r1(ssub(t24, t23), 181, 8)
	t24a = r1(sadd(t24, t23), 181, 8)

	t0, t1, t2, t3 := ld(0), ld(2), ld(4), ld(6)
	t4, t5, t6, t7 := ld(8), ld(10), ld(12), ld(14)
	t8, t9, t10, t11 := ld(16), ld(18), ld(20), ld(22)
	t12, t13, t14, t15 := ld(24), ld(26), ld(28), ld(30)

	st(0, clip(sadd(t0, t31)))
	st(1, clip(sadd(t1, t30a)))
	st(2, clip(sadd(t2, t29)))
	st(3, clip(sadd(t3, t28a)))
	st(4, clip(sadd(t4, t27)))
	st(5, clip(sadd(t5, t26a)))
	st(6, clip(sadd(t6, t25)))
	st(7, clip(sadd(t7, t24a)))
	st(8, clip(sadd(t8, t23a)))
	st(9, clip(sadd(t9, t22)))
	st(10, clip(sadd(t10, t21a)))
	st(11, clip(sadd(t11, t20)))
	st(12, clip(sadd(t12, t19a)))
	st(13, clip(sadd(t13, t18)))
	st(14, clip(sadd(t14, t17a)))
	st(15, clip(sadd(t15, t16)))
	st(16, clip(ssub(t15, t16)))
	st(17, clip(ssub(t14, t17a)))
	st(18, clip(ssub(t13, t18)))
	st(19, clip(ssub(t12, t19a)))
	st(20, clip(ssub(t11, t20)))
	st(21, clip(ssub(t10, t21a)))
	st(22, clip(ssub(t9, t22)))
	st(23, clip(ssub(t8, t23a)))
	st(24, clip(ssub(t7, t24a)))
	st(25, clip(ssub(t6, t25)))
	st(26, clip(ssub(t5, t26a)))
	st(27, clip(ssub(t4, t27)))
	st(28, clip(ssub(t3, t28a)))
	st(29, clip(ssub(t2, t29)))
	st(30, clip(ssub(t1, t30a)))
	st(31, clip(ssub(t0, t31)))
}

func inverseDCT64Col8SIMD16(buf []int16, stride int, min int32, max int32) {
	inverseDCT32Col8SIMD16(buf, stride<<1, min, max)

	minV := archsimd.BroadcastInt16x8(int16(min))
	maxV := archsimd.BroadcastInt16x8(int16(max))
	ld := func(k int) archsimd.Int16x8 { return archsimd.LoadInt16x8Array((*[8]int16)(buf[k*stride:])) }
	st := func(k int, v archsimd.Int16x8) { v.StoreArray((*[8]int16)(buf[k*stride:])) }
	clip := func(v archsimd.Int16x8) archsimd.Int16x8 { return v.Max(minV).Min(maxV) }
	sadd := func(a, b archsimd.Int16x8) archsimd.Int16x8 { return a.AddSaturated(b) }
	ssub := func(a, b archsimd.Int16x8) archsimd.Int16x8 { return a.SubSaturated(b) }
	mw := func(a archsimd.Int16x8, c int16) (archsimd.Int32x4, archsimd.Int32x4) {
		kc := archsimd.BroadcastInt16x8(c)
		return a.MulWidenLo(kc), a.MulWidenHi(kc)
	}
	nr := func(lo, hi archsimd.Int32x4, n uint8) archsimd.Int16x8 {
		return lo.ShiftRightRoundNarrow(n).ShiftRightRoundNarrowHi(hi, n)
	}
	rSub := func(a archsimd.Int16x8, ca int16, b archsimd.Int16x8, cb int16, n uint8) archsimd.Int16x8 {
		al, ah := mw(a, ca)
		bl, bh := mw(b, cb)
		return nr(al.Sub(bl), ah.Sub(bh), n)
	}
	rAdd := func(a archsimd.Int16x8, ca int16, b archsimd.Int16x8, cb int16, n uint8) archsimd.Int16x8 {
		al, ah := mw(a, ca)
		bl, bh := mw(b, cb)
		return nr(al.Add(bl), ah.Add(bh), n)
	}
	rAddNeg := func(a archsimd.Int16x8, ca int16, b archsimd.Int16x8, cb int16, n uint8) archsimd.Int16x8 {
		al, ah := mw(a, ca)
		bl, bh := mw(b, cb)
		return nr(al.Add(bl).Neg(), ah.Add(bh).Neg(), n)
	}
	r1 := func(a archsimd.Int16x8, c int16, n uint8) archsimd.Int16x8 {
		l, h := mw(a, c)
		return nr(l, h, n)
	}
	_, _, _ = rSub, rAddNeg, r1
	hbtf := func(ca int16, a archsimd.Int16x8, cb int16, b archsimd.Int16x8) archsimd.Int16x8 {
		return rAdd(a, ca, b, cb, 12)
	}
	add := func(a, b archsimd.Int16x8) archsimd.Int16x8 { return clip(sadd(a, b)) }
	sub := func(a, b archsimd.Int16x8) archsimd.Int16x8 { return clip(ssub(a, b)) }
	nadd := func(a, b archsimd.Int16x8) archsimd.Int16x8 { return clip(ssub(b, a)) }

	const (
		cospi1  = 4095
		cospi3  = 4085
		cospi4  = 4076
		cospi5  = 4065
		cospi7  = 4036
		cospi8  = 4017
		cospi9  = 3996
		cospi11 = 3948
		cospi12 = 3920
		cospi13 = 3889
		cospi15 = 3822
		cospi16 = 3784
		cospi17 = 3745
		cospi19 = 3659
		cospi20 = 3612
		cospi21 = 3564
		cospi23 = 3461
		cospi24 = 3406
		cospi25 = 3349
		cospi27 = 3229
		cospi28 = 3166
		cospi29 = 3102
		cospi31 = 2967
		cospi32 = 2896
		cospi33 = 2824
		cospi35 = 2675
		cospi36 = 2598
		cospi37 = 2520
		cospi39 = 2359
		cospi40 = 2276
		cospi41 = 2191
		cospi43 = 2019
		cospi44 = 1931
		cospi45 = 1842
		cospi47 = 1660
		cospi48 = 1567
		cospi49 = 1474
		cospi51 = 1285
		cospi52 = 1189
		cospi53 = 1092
		cospi55 = 897
		cospi56 = 799
		cospi57 = 700
		cospi59 = 501
		cospi60 = 401
		cospi61 = 301
		cospi63 = 101
	)

	// Stage 1: bit-reversed odd inputs.
	x32, x33, x34, x35 := ld(1), ld(33), ld(17), ld(49)
	x36, x37, x38, x39 := ld(9), ld(41), ld(25), ld(57)
	x40, x41, x42, x43 := ld(5), ld(37), ld(21), ld(53)
	x44, x45, x46, x47 := ld(13), ld(45), ld(29), ld(61)
	x48, x49, x50, x51 := ld(3), ld(35), ld(19), ld(51)
	x52, x53, x54, x55 := ld(11), ld(43), ld(27), ld(59)
	x56, x57, x58, x59 := ld(7), ld(39), ld(23), ld(55)
	x60, x61, x62, x63 := ld(15), ld(47), ld(31), ld(63)

	// Stage 2.
	y32 := hbtf(cospi63, x32, -cospi1, x63)
	y33 := hbtf(cospi31, x33, -cospi33, x62)
	y34 := hbtf(cospi47, x34, -cospi17, x61)
	y35 := hbtf(cospi15, x35, -cospi49, x60)
	y36 := hbtf(cospi55, x36, -cospi9, x59)
	y37 := hbtf(cospi23, x37, -cospi41, x58)
	y38 := hbtf(cospi39, x38, -cospi25, x57)
	y39 := hbtf(cospi7, x39, -cospi57, x56)
	y40 := hbtf(cospi59, x40, -cospi5, x55)
	y41 := hbtf(cospi27, x41, -cospi37, x54)
	y42 := hbtf(cospi43, x42, -cospi21, x53)
	y43 := hbtf(cospi11, x43, -cospi53, x52)
	y44 := hbtf(cospi51, x44, -cospi13, x51)
	y45 := hbtf(cospi19, x45, -cospi45, x50)
	y46 := hbtf(cospi35, x46, -cospi29, x49)
	y47 := hbtf(cospi3, x47, -cospi61, x48)
	y48 := hbtf(cospi61, x47, cospi3, x48)
	y49 := hbtf(cospi29, x46, cospi35, x49)
	y50 := hbtf(cospi45, x45, cospi19, x50)
	y51 := hbtf(cospi13, x44, cospi51, x51)
	y52 := hbtf(cospi53, x43, cospi11, x52)
	y53 := hbtf(cospi21, x42, cospi43, x53)
	y54 := hbtf(cospi37, x41, cospi27, x54)
	y55 := hbtf(cospi5, x40, cospi59, x55)
	y56 := hbtf(cospi57, x39, cospi7, x56)
	y57 := hbtf(cospi25, x38, cospi39, x57)
	y58 := hbtf(cospi41, x37, cospi23, x58)
	y59 := hbtf(cospi9, x36, cospi55, x59)
	y60 := hbtf(cospi49, x35, cospi15, x60)
	y61 := hbtf(cospi17, x34, cospi47, x61)
	y62 := hbtf(cospi33, x33, cospi31, x62)
	y63 := hbtf(cospi1, x32, cospi63, x63)
	x32, x33, x34, x35 = y32, y33, y34, y35
	x36, x37, x38, x39 = y36, y37, y38, y39
	x40, x41, x42, x43 = y40, y41, y42, y43
	x44, x45, x46, x47 = y44, y45, y46, y47
	x48, x49, x50, x51 = y48, y49, y50, y51
	x52, x53, x54, x55 = y52, y53, y54, y55
	x56, x57, x58, x59 = y56, y57, y58, y59
	x60, x61, x62, x63 = y60, y61, y62, y63

	// Stage 3.
	y32 = add(x32, x33)
	y33 = sub(x32, x33)
	y34 = nadd(x34, x35)
	y35 = add(x34, x35)
	y36 = add(x36, x37)
	y37 = sub(x36, x37)
	y38 = nadd(x38, x39)
	y39 = add(x38, x39)
	y40 = add(x40, x41)
	y41 = sub(x40, x41)
	y42 = nadd(x42, x43)
	y43 = add(x42, x43)
	y44 = add(x44, x45)
	y45 = sub(x44, x45)
	y46 = nadd(x46, x47)
	y47 = add(x46, x47)
	y48 = add(x48, x49)
	y49 = sub(x48, x49)
	y50 = nadd(x50, x51)
	y51 = add(x50, x51)
	y52 = add(x52, x53)
	y53 = sub(x52, x53)
	y54 = nadd(x54, x55)
	y55 = add(x54, x55)
	y56 = add(x56, x57)
	y57 = sub(x56, x57)
	y58 = nadd(x58, x59)
	y59 = add(x58, x59)
	y60 = add(x60, x61)
	y61 = sub(x60, x61)
	y62 = nadd(x62, x63)
	y63 = add(x62, x63)
	x32, x33, x34, x35 = y32, y33, y34, y35
	x36, x37, x38, x39 = y36, y37, y38, y39
	x40, x41, x42, x43 = y40, y41, y42, y43
	x44, x45, x46, x47 = y44, y45, y46, y47
	x48, x49, x50, x51 = y48, y49, y50, y51
	x52, x53, x54, x55 = y52, y53, y54, y55
	x56, x57, x58, x59 = y56, y57, y58, y59
	x60, x61, x62, x63 = y60, y61, y62, y63

	// Stage 4.
	y33 = hbtf(-cospi4, x33, cospi60, x62)
	y34 = hbtf(-cospi60, x34, -cospi4, x61)
	y37 = hbtf(-cospi36, x37, cospi28, x58)
	y38 = hbtf(-cospi28, x38, -cospi36, x57)
	y41 = hbtf(-cospi20, x41, cospi44, x54)
	y42 = hbtf(-cospi44, x42, -cospi20, x53)
	y45 = hbtf(-cospi52, x45, cospi12, x50)
	y46 = hbtf(-cospi12, x46, -cospi52, x49)
	y49 = hbtf(-cospi52, x46, cospi12, x49)
	y50 = hbtf(cospi12, x45, cospi52, x50)
	y53 = hbtf(-cospi20, x42, cospi44, x53)
	y54 = hbtf(cospi44, x41, cospi20, x54)
	y57 = hbtf(-cospi36, x38, cospi28, x57)
	y58 = hbtf(cospi28, x37, cospi36, x58)
	y61 = hbtf(-cospi4, x34, cospi60, x61)
	y62 = hbtf(cospi60, x33, cospi4, x62)
	x33, x34, x37, x38 = y33, y34, y37, y38
	x41, x42, x45, x46 = y41, y42, y45, y46
	x49, x50, x53, x54 = y49, y50, y53, y54
	x57, x58, x61, x62 = y57, y58, y61, y62

	// Stage 5.
	y32 = add(x32, x35)
	y33 = add(x33, x34)
	y34 = sub(x33, x34)
	y35 = sub(x32, x35)
	y36 = nadd(x36, x39)
	y37 = nadd(x37, x38)
	y38 = add(x37, x38)
	y39 = add(x36, x39)
	y40 = add(x40, x43)
	y41 = add(x41, x42)
	y42 = sub(x41, x42)
	y43 = sub(x40, x43)
	y44 = nadd(x44, x47)
	y45 = nadd(x45, x46)
	y46 = add(x45, x46)
	y47 = add(x44, x47)
	y48 = add(x48, x51)
	y49 = add(x49, x50)
	y50 = sub(x49, x50)
	y51 = sub(x48, x51)
	y52 = nadd(x52, x55)
	y53 = nadd(x53, x54)
	y54 = add(x53, x54)
	y55 = add(x52, x55)
	y56 = add(x56, x59)
	y57 = add(x57, x58)
	y58 = sub(x57, x58)
	y59 = sub(x56, x59)
	y60 = nadd(x60, x63)
	y61 = nadd(x61, x62)
	y62 = add(x61, x62)
	y63 = add(x60, x63)
	x32, x33, x34, x35 = y32, y33, y34, y35
	x36, x37, x38, x39 = y36, y37, y38, y39
	x40, x41, x42, x43 = y40, y41, y42, y43
	x44, x45, x46, x47 = y44, y45, y46, y47
	x48, x49, x50, x51 = y48, y49, y50, y51
	x52, x53, x54, x55 = y52, y53, y54, y55
	x56, x57, x58, x59 = y56, y57, y58, y59
	x60, x61, x62, x63 = y60, y61, y62, y63

	// Stage 6.
	y34 = hbtf(-cospi8, x34, cospi56, x61)
	y35 = hbtf(-cospi8, x35, cospi56, x60)
	y36 = hbtf(-cospi56, x36, -cospi8, x59)
	y37 = hbtf(-cospi56, x37, -cospi8, x58)
	y42 = hbtf(-cospi40, x42, cospi24, x53)
	y43 = hbtf(-cospi40, x43, cospi24, x52)
	y44 = hbtf(-cospi24, x44, -cospi40, x51)
	y45 = hbtf(-cospi24, x45, -cospi40, x50)
	y50 = hbtf(-cospi40, x45, cospi24, x50)
	y51 = hbtf(-cospi40, x44, cospi24, x51)
	y52 = hbtf(cospi24, x43, cospi40, x52)
	y53 = hbtf(cospi24, x42, cospi40, x53)
	y58 = hbtf(-cospi8, x37, cospi56, x58)
	y59 = hbtf(-cospi8, x36, cospi56, x59)
	y60 = hbtf(cospi56, x35, cospi8, x60)
	y61 = hbtf(cospi56, x34, cospi8, x61)
	x34, x35, x36, x37 = y34, y35, y36, y37
	x42, x43, x44, x45 = y42, y43, y44, y45
	x50, x51, x52, x53 = y50, y51, y52, y53
	x58, x59, x60, x61 = y58, y59, y60, y61

	// Stage 7.
	y32 = add(x32, x39)
	y33 = add(x33, x38)
	y34 = add(x34, x37)
	y35 = add(x35, x36)
	y36 = sub(x35, x36)
	y37 = sub(x34, x37)
	y38 = sub(x33, x38)
	y39 = sub(x32, x39)
	y40 = nadd(x40, x47)
	y41 = nadd(x41, x46)
	y42 = nadd(x42, x45)
	y43 = nadd(x43, x44)
	y44 = add(x43, x44)
	y45 = add(x42, x45)
	y46 = add(x41, x46)
	y47 = add(x40, x47)
	y48 = add(x48, x55)
	y49 = add(x49, x54)
	y50 = add(x50, x53)
	y51 = add(x51, x52)
	y52 = sub(x51, x52)
	y53 = sub(x50, x53)
	y54 = sub(x49, x54)
	y55 = sub(x48, x55)
	y56 = nadd(x56, x63)
	y57 = nadd(x57, x62)
	y58 = nadd(x58, x61)
	y59 = nadd(x59, x60)
	y60 = add(x59, x60)
	y61 = add(x58, x61)
	y62 = add(x57, x62)
	y63 = add(x56, x63)
	x32, x33, x34, x35 = y32, y33, y34, y35
	x36, x37, x38, x39 = y36, y37, y38, y39
	x40, x41, x42, x43 = y40, y41, y42, y43
	x44, x45, x46, x47 = y44, y45, y46, y47
	x48, x49, x50, x51 = y48, y49, y50, y51
	x52, x53, x54, x55 = y52, y53, y54, y55
	x56, x57, x58, x59 = y56, y57, y58, y59
	x60, x61, x62, x63 = y60, y61, y62, y63

	// Stage 8.
	y36 = hbtf(-cospi16, x36, cospi48, x59)
	y37 = hbtf(-cospi16, x37, cospi48, x58)
	y38 = hbtf(-cospi16, x38, cospi48, x57)
	y39 = hbtf(-cospi16, x39, cospi48, x56)
	y40 = hbtf(-cospi48, x40, -cospi16, x55)
	y41 = hbtf(-cospi48, x41, -cospi16, x54)
	y42 = hbtf(-cospi48, x42, -cospi16, x53)
	y43 = hbtf(-cospi48, x43, -cospi16, x52)
	y52 = hbtf(-cospi16, x43, cospi48, x52)
	y53 = hbtf(-cospi16, x42, cospi48, x53)
	y54 = hbtf(-cospi16, x41, cospi48, x54)
	y55 = hbtf(-cospi16, x40, cospi48, x55)
	y56 = hbtf(cospi48, x39, cospi16, x56)
	y57 = hbtf(cospi48, x38, cospi16, x57)
	y58 = hbtf(cospi48, x37, cospi16, x58)
	y59 = hbtf(cospi48, x36, cospi16, x59)
	x36, x37, x38, x39 = y36, y37, y38, y39
	x40, x41, x42, x43 = y40, y41, y42, y43
	x52, x53, x54, x55 = y52, y53, y54, y55
	x56, x57, x58, x59 = y56, y57, y58, y59

	// Stage 9.
	y32 = add(x32, x47)
	y33 = add(x33, x46)
	y34 = add(x34, x45)
	y35 = add(x35, x44)
	y36 = add(x36, x43)
	y37 = add(x37, x42)
	y38 = add(x38, x41)
	y39 = add(x39, x40)
	y40 = sub(x39, x40)
	y41 = sub(x38, x41)
	y42 = sub(x37, x42)
	y43 = sub(x36, x43)
	y44 = sub(x35, x44)
	y45 = sub(x34, x45)
	y46 = sub(x33, x46)
	y47 = sub(x32, x47)
	y48 = nadd(x48, x63)
	y49 = nadd(x49, x62)
	y50 = nadd(x50, x61)
	y51 = nadd(x51, x60)
	y52 = nadd(x52, x59)
	y53 = nadd(x53, x58)
	y54 = nadd(x54, x57)
	y55 = nadd(x55, x56)
	y56 = add(x55, x56)
	y57 = add(x54, x57)
	y58 = add(x53, x58)
	y59 = add(x52, x59)
	y60 = add(x51, x60)
	y61 = add(x50, x61)
	y62 = add(x49, x62)
	y63 = add(x48, x63)
	x32, x33, x34, x35 = y32, y33, y34, y35
	x36, x37, x38, x39 = y36, y37, y38, y39
	x40, x41, x42, x43 = y40, y41, y42, y43
	x44, x45, x46, x47 = y44, y45, y46, y47
	x48, x49, x50, x51 = y48, y49, y50, y51
	x52, x53, x54, x55 = y52, y53, y54, y55
	x56, x57, x58, x59 = y56, y57, y58, y59
	x60, x61, x62, x63 = y60, y61, y62, y63

	// Stage 10.
	y40 = hbtf(-cospi32, x40, cospi32, x55)
	y41 = hbtf(-cospi32, x41, cospi32, x54)
	y42 = hbtf(-cospi32, x42, cospi32, x53)
	y43 = hbtf(-cospi32, x43, cospi32, x52)
	y44 = hbtf(-cospi32, x44, cospi32, x51)
	y45 = hbtf(-cospi32, x45, cospi32, x50)
	y46 = hbtf(-cospi32, x46, cospi32, x49)
	y47 = hbtf(-cospi32, x47, cospi32, x48)
	y48 = hbtf(cospi32, x47, cospi32, x48)
	y49 = hbtf(cospi32, x46, cospi32, x49)
	y50 = hbtf(cospi32, x45, cospi32, x50)
	y51 = hbtf(cospi32, x44, cospi32, x51)
	y52 = hbtf(cospi32, x43, cospi32, x52)
	y53 = hbtf(cospi32, x42, cospi32, x53)
	y54 = hbtf(cospi32, x41, cospi32, x54)
	y55 = hbtf(cospi32, x40, cospi32, x55)
	x40, x41, x42, x43 = y40, y41, y42, y43
	x44, x45, x46, x47 = y44, y45, y46, y47
	x48, x49, x50, x51 = y48, y49, y50, y51
	x52, x53, x54, x55 = y52, y53, y54, y55

	e0, e1, e2, e3 := ld(0), ld(2), ld(4), ld(6)
	e4, e5, e6, e7 := ld(8), ld(10), ld(12), ld(14)
	e8, e9, e10, e11 := ld(16), ld(18), ld(20), ld(22)
	e12, e13, e14, e15 := ld(24), ld(26), ld(28), ld(30)
	e16, e17, e18, e19 := ld(32), ld(34), ld(36), ld(38)
	e20, e21, e22, e23 := ld(40), ld(42), ld(44), ld(46)
	e24, e25, e26, e27 := ld(48), ld(50), ld(52), ld(54)
	e28, e29, e30, e31 := ld(56), ld(58), ld(60), ld(62)

	st(0, add(e0, x63))
	st(1, add(e1, x62))
	st(2, add(e2, x61))
	st(3, add(e3, x60))
	st(4, add(e4, x59))
	st(5, add(e5, x58))
	st(6, add(e6, x57))
	st(7, add(e7, x56))
	st(8, add(e8, x55))
	st(9, add(e9, x54))
	st(10, add(e10, x53))
	st(11, add(e11, x52))
	st(12, add(e12, x51))
	st(13, add(e13, x50))
	st(14, add(e14, x49))
	st(15, add(e15, x48))
	st(16, add(e16, x47))
	st(17, add(e17, x46))
	st(18, add(e18, x45))
	st(19, add(e19, x44))
	st(20, add(e20, x43))
	st(21, add(e21, x42))
	st(22, add(e22, x41))
	st(23, add(e23, x40))
	st(24, add(e24, x39))
	st(25, add(e25, x38))
	st(26, add(e26, x37))
	st(27, add(e27, x36))
	st(28, add(e28, x35))
	st(29, add(e29, x34))
	st(30, add(e30, x33))
	st(31, add(e31, x32))
	st(32, sub(e31, x32))
	st(33, sub(e30, x33))
	st(34, sub(e29, x34))
	st(35, sub(e28, x35))
	st(36, sub(e27, x36))
	st(37, sub(e26, x37))
	st(38, sub(e25, x38))
	st(39, sub(e24, x39))
	st(40, sub(e23, x40))
	st(41, sub(e22, x41))
	st(42, sub(e21, x42))
	st(43, sub(e20, x43))
	st(44, sub(e19, x44))
	st(45, sub(e18, x45))
	st(46, sub(e17, x46))
	st(47, sub(e16, x47))
	st(48, sub(e15, x48))
	st(49, sub(e14, x49))
	st(50, sub(e13, x50))
	st(51, sub(e12, x51))
	st(52, sub(e11, x52))
	st(53, sub(e10, x53))
	st(54, sub(e9, x54))
	st(55, sub(e8, x55))
	st(56, sub(e7, x56))
	st(57, sub(e6, x57))
	st(58, sub(e5, x58))
	st(59, sub(e4, x59))
	st(60, sub(e3, x60))
	st(61, sub(e2, x61))
	st(62, sub(e1, x62))
	st(63, sub(e0, x63))
}
