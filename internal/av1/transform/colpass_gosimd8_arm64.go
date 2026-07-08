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
