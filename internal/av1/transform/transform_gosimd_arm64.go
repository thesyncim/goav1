// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package transform

import (
	"encoding/binary"
	"github.com/thesyncim/goav1/internal/av1/dsp/cpu"
	"simd/archsimd"
)

// roundShiftNarrowInt32x4ToInt16x8 matches signed round-to-nearest with ties
// toward positive infinity, followed by signed saturation. shift must be
// positive. Splitting the rounding bit from the shifted value avoids overflow
// in x + (1 << (n-1)).
func roundShiftNarrowInt32x4ToInt16x8(lo, hi archsimd.Int32x4, shift uint8) archsimd.Int16x8 {
	shift64 := uint64(shift)
	one := archsimd.BroadcastInt32x4(1)
	lo16 := lo.ShiftAllRight(shift64).Add(lo.ShiftAllRight(shift64 - 1).And(one)).SaturateToInt16()
	hi16 := hi.ShiftAllRight(shift64).Add(hi.ShiftAllRight(shift64 - 1).And(one)).SaturateToInt16()
	return lo16.ToBits().ReshapeToUint64s().InterleaveLo(hi16.ToBits().ReshapeToUint64s()).ReshapeToUint16s().BitsToInt16()
}

type int32x8Pair struct {
	lo archsimd.Int32x4
	hi archsimd.Int32x4
}

func widenInt16x8(x archsimd.Int16x8) int32x8Pair {
	return int32x8Pair{lo: x.ExtendLo4ToInt32(), hi: x.HiToLo().ExtendLo4ToInt32()}
}

func mulWidenInt16x8(x archsimd.Int16x8, c int16) int32x8Pair {
	k := archsimd.BroadcastInt16x8(c)
	return int32x8Pair{lo: x.MulWidenLo(k), hi: x.HiToLo().MulWidenLo(k)}
}

func addInt32x8Pair(x, y int32x8Pair) int32x8Pair {
	return int32x8Pair{lo: x.lo.Add(y.lo), hi: x.hi.Add(y.hi)}
}

func subInt32x8Pair(x, y int32x8Pair) int32x8Pair {
	return int32x8Pair{lo: x.lo.Sub(y.lo), hi: x.hi.Sub(y.hi)}
}

func roundShiftInt32x8Pair(x int32x8Pair, shift uint8) int32x8Pair {
	n := uint64(shift)
	one := archsimd.BroadcastInt32x4(1)
	return int32x8Pair{
		lo: x.lo.ShiftAllRight(n).Add(x.lo.ShiftAllRight(n - 1).And(one)),
		hi: x.hi.ShiftAllRight(n).Add(x.hi.ShiftAllRight(n - 1).And(one)),
	}
}

func clipInt32x8PairToInt16(x int32x8Pair, minV, maxV archsimd.Int16x8) archsimd.Int16x8 {
	lo := x.lo.SaturateToInt16()
	hi := x.hi.SaturateToInt16()
	packed := lo.ToBits().ReshapeToUint64s().InterleaveLo(hi.ToBits().ReshapeToUint64s()).ReshapeToUint16s().BitsToInt16()
	return packed.Max(minV).Min(maxV)
}

// inverseDCT8Col8SIMD16 evaluates every DCT8 rotation in int32 and applies
// int16 saturation only at the scalar transform's actual clip stages. The
// whole int16 input range is therefore exact, including sums that exceed int16
// before a later rotation or stage clip.
func inverseDCT8Col8SIMD16(buf []int16, stride int, min int32, max int32) {
	min16V, max16V := archsimd.BroadcastInt16x8(int16(min)), archsimd.BroadcastInt16x8(int16(max))
	ld := func(k int) archsimd.Int16x8 { return archsimd.LoadInt16x8Array((*[8]int16)(buf[k*stride:])) }
	st := func(k int, v archsimd.Int16x8) { v.StoreArray((*[8]int16)(buf[k*stride:])) }
	clip := func(v archsimd.Int16x8) archsimd.Int16x8 { return v.Max(min16V).Min(max16V) }
	clipWide := func(v int32x8Pair) archsimd.Int16x8 { return clipInt32x8PairToInt16(v, min16V, max16V) }
	clippedAdd := func(a, b archsimd.Int16x8) archsimd.Int16x8 {
		return clip(a.AddSaturated(b))
	}
	clippedSub := func(a, b archsimd.Int16x8) archsimd.Int16x8 {
		return clip(a.SubSaturated(b))
	}
	rotate := func(a, b archsimd.Int16x8, aCoeff, bCoeff int16, shift uint8, subtract bool) int32x8Pair {
		left, right := mulWidenInt16x8(a, aCoeff), mulWidenInt16x8(b, bCoeff)
		var sum int32x8Pair
		if subtract {
			sum = subInt32x8Pair(left, right)
		} else {
			sum = addInt32x8Pair(left, right)
		}
		return roundShiftInt32x8Pair(sum, shift)
	}
	wide16 := widenInt16x8

	c0, c1, c2, c3 := ld(0), ld(1), ld(2), ld(3)
	c4, c5, c6, c7 := ld(4), ld(5), ld(6), ld(7)

	// Even DCT4. Keep rounded rotations wide until the four scalar clip points.
	t0 := rotate(c0, c4, 181, 181, 8, false)
	t1 := rotate(c0, c4, 181, 181, 8, true)
	t2 := rotate(c2, c6, 1567, -3784, 12, false)
	t3 := rotate(c2, c6, 3784, 1567, 12, false)
	d0 := clipWide(addInt32x8Pair(t0, t3))
	d2 := clipWide(addInt32x8Pair(t1, t2))
	d4 := clipWide(subInt32x8Pair(t1, t2))
	d6 := clipWide(subInt32x8Pair(t0, t3))

	// Odd rotations and their actual scalar clip stages.
	t4a := rotate(c1, c7, 799, -4017, 12, false)
	t5a := rotate(c5, c3, 1703, 1138, 11, true)
	t6a := rotate(c5, c3, 1138, 1703, 11, false)
	t7a := rotate(c1, c7, 4017, 799, 12, false)
	t4 := clipWide(addInt32x8Pair(t4a, t5a))
	t5c := clipWide(subInt32x8Pair(t4a, t5a))
	t7 := clipWide(addInt32x8Pair(t7a, t6a))
	t6c := clipWide(subInt32x8Pair(t7a, t6a))

	// These final 181 rotations may exceed int16 and are not clipped until the
	// final output butterflies. Form the product from widened terms directly.
	t5 := roundShiftInt32x8Pair(addInt32x8Pair(mulWidenInt16x8(t6c, 181), mulWidenInt16x8(t5c, -181)), 8)
	t6 := roundShiftInt32x8Pair(addInt32x8Pair(mulWidenInt16x8(t6c, 181), mulWidenInt16x8(t5c, 181)), 8)

	st(0, clippedAdd(d0, t7))
	st(1, clipWide(addInt32x8Pair(wide16(d2), t6)))
	st(2, clipWide(addInt32x8Pair(wide16(d4), t5)))
	st(3, clippedAdd(d6, t4))
	st(4, clippedSub(d6, t4))
	st(5, clipWide(subInt32x8Pair(wide16(d4), t5)))
	st(6, clipWide(subInt32x8Pair(wide16(d2), t6)))
	st(7, clippedSub(d0, t7))
}

// inverseDCT16Col8SIMD16 is the int16 8-wide DCT16 column pass (dav1d 8bpc):
// inverseDCT8 on the even rows, then the 16-point odd butterfly, saturating
// int16 adds throughout. It is byte-exact with scalar inverseDCT16[int16] for
// values accepted by the DCT16 input guard.
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
		return a.MulWidenLo(kc), a.HiToLo().MulWidenLo(kc)
	}
	nr := func(lo, hi archsimd.Int32x4, n uint8) archsimd.Int16x8 {
		return roundShiftNarrowInt32x4ToInt16x8(lo, hi, n)
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
		return a.MulWidenLo(kc), a.HiToLo().MulWidenLo(kc)
	}
	nr := func(lo, hi archsimd.Int32x4, n uint8) archsimd.Int16x8 {
		return roundShiftNarrowInt32x4ToInt16x8(lo, hi, n)
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
		return a.MulWidenLo(kc), a.HiToLo().MulWidenLo(kc)
	}
	nr := func(lo, hi archsimd.Int32x4, n uint8) archsimd.Int16x8 {
		return roundShiftNarrowInt32x4ToInt16x8(lo, hi, n)
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

// Bind the int16 8-wide SIMD DCT column kernels under GOEXPERIMENT=simd.
// The public path keeps the int16 buffer between passes and uses these kernels
// for the vertical DCT when the block height is supported.
func init() {
	inverseDCT8Col8Impl16 = inverseDCT8Col8SIMD16
	inverseDCT16Col8Impl16 = inverseDCT16Col8SIMD16
	inverseDCT32Col8Impl16 = inverseDCT32Col8SIMD16
	inverseDCT64Col8Impl16 = inverseDCT64Col8SIMD16
	clampRoundNarrowInt16Impl = clampRoundNarrowInt16SIMD
	int16ColumnSIMDInputEligible = int16ColumnSIMDInputEligibleSIMD
	int16ColumnFast = true
}

// int16ColumnSIMDInputSafeSIMD checks the certified absolute-value bound with
// one unsigned max reduction for the full block. Abs(MinInt16) retains the
// 0x8000 bit pattern, which becomes unsigned 32768 through ToBits and therefore
// correctly fails every DCT16/32/64 bound.
func int16ColumnSIMDInputSafeSIMD(buf []int16, width int, height int, min int32, max int32) bool {
	if width < 8 {
		return true
	}
	if width <= 0 || height <= 0 || len(buf) < width*height || min > max || min < minInt16 || max > maxInt16 {
		return false
	}
	if height == dct8Size {
		return true
	}
	if min > 0 || max < 0 {
		return false
	}
	limit := int16ColumnSIMDInputBound(height)
	if limit == 0 {
		return false
	}
	return int16ColumnSIMDAbsWithinBound(buf, width, height, uint16(limit))
}

// int16ColumnSIMDInputEligibleSIMD separates exactness from workload policy.
// DCT8 is exact over the full int16 range, but the int32/NEON fallback wins
// above the measured profitability threshold. Other DCT sizes use only their
// certified exactness bounds.
func int16ColumnSIMDInputEligibleSIMD(buf []int16, width int, height int, min int32, max int32) bool {
	if !int16ColumnSIMDInputSafeSIMD(buf, width, height, min, max) {
		return false
	}
	if height == dct8Size {
		return int16ColumnSIMDAbsWithinBound(buf, width, height, uint16(dct8SIMDProfitabilityBound))
	}
	return true
}

// int16ColumnSIMDAbsWithinBound uses an unsigned absolute-value reduction.
// Abs(MinInt16) has bit pattern 0x8000, which remains unsigned 32768 and thus
// fails every narrower bound. Checking the first vector immediately rejects
// common high-range inputs before scanning the rest of the block.
func int16ColumnSIMDAbsWithinBound(buf []int16, width int, height int, limit uint16) bool {
	if width < 8 {
		return true
	}
	if width <= 0 || height <= 0 || len(buf) < width*height {
		return false
	}
	total := width * height
	maxAbs := archsimd.LoadInt16x8Array((*[8]int16)(buf)).Abs().ToBits()
	if maxAbs.ReduceMax() > limit {
		return false
	}
	i := 8
	for ; i+8 <= total; i += 8 {
		v := archsimd.LoadInt16x8Array((*[8]int16)(buf[i:]))
		maxAbs = maxAbs.Max(v.Abs().ToBits())
	}
	if maxAbs.ReduceMax() > limit {
		return false
	}
	for ; i < total; i++ {
		v := int32(buf[i])
		if v < 0 {
			v = -v
		}
		if uint16(v) > limit {
			return false
		}
	}
	return true
}

// clampRoundNarrowInt16SIMD is the fused mid-pass round+clamp+narrow: at
// bitDepth 8 the column clamp bounds are exactly the int16 range, so
// roundShift + clip + narrow is one saturating rounding narrow (SQRSHRN) per
// four lanes. Shifts other than 1/2 (and non-int16 bounds, which cannot occur
// on the 8-bit path) fall back to the scalar sweep.
func clampRoundNarrowInt16SIMD(src []int32, dst []int16, shift int, lo int32, hi int32) {
	if lo != -32768 || hi != 32767 || (shift != 1 && shift != 2) {
		clampRoundNarrowInt16Scalar(src, dst, shift, lo, hi)
		return
	}
	n := len(src)
	dst = dst[:n]
	i := 0
	if shift == 1 {
		for ; i <= len(src)-8 && i <= len(dst)-8; i += 8 {
			s := (*[8]int32)(src[i:])
			v0 := archsimd.LoadInt32x4Array((*[4]int32)(s[0:4]))
			v1 := archsimd.LoadInt32x4Array((*[4]int32)(s[4:8]))
			roundShiftNarrowInt32x4ToInt16x8(v0, v1, 1).StoreArray((*[8]int16)(dst[i:]))
		}
	} else {
		for ; i <= len(src)-8 && i <= len(dst)-8; i += 8 {
			s := (*[8]int32)(src[i:])
			v0 := archsimd.LoadInt32x4Array((*[4]int32)(s[0:4]))
			v1 := archsimd.LoadInt32x4Array((*[4]int32)(s[4:8]))
			roundShiftNarrowInt32x4ToInt16x8(v0, v1, 2).StoreArray((*[8]int16)(dst[i:]))
		}
	}
	if i < n {
		clampRoundNarrowInt16Scalar(src[i:], dst[i:], shift, lo, hi)
	}
}

// init binds the Go-native SIMD four-lane DCT64 column and row kernels. Every
// arm64 chip has NEON; the gate lets tests force the pure-Go path.
func init() {
	if cpu.Detected.NEON {
		inverseDCT64Col4Impl = inverseDCT64Col4SIMD
		inverseDCT64Row4Impl = inverseDCT64Row4SIMD
		inverseDCT64LowCol4Impl = inverseDCT64LowCol4SIMD
		inverseDCT64Row4LowImpl = inverseDCT64Row4LowSIMD
	}
}

// fwd8DCTCore keeps the block register-resident through the column pass, a
// transpose and the row pass. Each cNr vector holds four adjacent spatial columns at
// transformed row N; the interleaves gather four rows for the row pass.
func fwd8DCTCore(coeffArr *[64]int32, residualArr *[64]int16) {
	coeff, coeffStride, residual, residualStride := coeffArr[:], 8, residualArr[:], 8
	var c0l, c1l, c2l, c3l, c4l, c5l, c6l, c7l, c0h, c1h, c2h, c3h, c4h, c5h, c6h, c7h fwdVec
	{
		g := 0
		rs := residualStride

		x0 := fwdLoadResAt(residual, 0*rs+g&^7, g)
		x1 := fwdLoadResAt(residual, 1*rs+g&^7, g)
		x2 := fwdLoadResAt(residual, 2*rs+g&^7, g)
		x3 := fwdLoadResAt(residual, 3*rs+g&^7, g)
		x4 := fwdLoadResAt(residual, 4*rs+g&^7, g)
		x5 := fwdLoadResAt(residual, 5*rs+g&^7, g)
		x6 := fwdLoadResAt(residual, 6*rs+g&^7, g)
		x7 := fwdLoadResAt(residual, 7*rs+g&^7, g)
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
		c0l = fwdRoundShift1V(s0)
		c1l = fwdRoundShift1V(s4)
		c2l = fwdRoundShift1V(s2)
		c3l = fwdRoundShift1V(s6)
		c4l = fwdRoundShift1V(s1)
		c5l = fwdRoundShift1V(s5)
		c6l = fwdRoundShift1V(s3)
		c7l = fwdRoundShift1V(s7)
	}
	{
		g := 4
		rs := residualStride

		x0 := fwdLoadResAt(residual, 0*rs+g&^7, g)
		x1 := fwdLoadResAt(residual, 1*rs+g&^7, g)
		x2 := fwdLoadResAt(residual, 2*rs+g&^7, g)
		x3 := fwdLoadResAt(residual, 3*rs+g&^7, g)
		x4 := fwdLoadResAt(residual, 4*rs+g&^7, g)
		x5 := fwdLoadResAt(residual, 5*rs+g&^7, g)
		x6 := fwdLoadResAt(residual, 6*rs+g&^7, g)
		x7 := fwdLoadResAt(residual, 7*rs+g&^7, g)
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
		c0h = fwdRoundShift1V(s0)
		c1h = fwdRoundShift1V(s4)
		c2h = fwdRoundShift1V(s2)
		c3h = fwdRoundShift1V(s6)
		c4h = fwdRoundShift1V(s1)
		c5h = fwdRoundShift1V(s5)
		c6h = fwdRoundShift1V(s3)
		c7h = fwdRoundShift1V(s7)
	}
	{
		h := 0

		le0 := c0l.InterleaveLo(c1l)
		le1 := c0l.InterleaveHi(c1l)
		le2 := c2l.InterleaveLo(c3l)
		le3 := c2l.InterleaveHi(c3l)
		x0 := fwdI64AsI32(fwdI32AsI64(le0).InterleaveLo(fwdI32AsI64(le2)))
		x1 := fwdI64AsI32(fwdI32AsI64(le0).InterleaveHi(fwdI32AsI64(le2)))
		x2 := fwdI64AsI32(fwdI32AsI64(le1).InterleaveLo(fwdI32AsI64(le3)))
		x3 := fwdI64AsI32(fwdI32AsI64(le1).InterleaveHi(fwdI32AsI64(le3)))

		he0 := c0h.InterleaveLo(c1h)
		he1 := c0h.InterleaveHi(c1h)
		he2 := c2h.InterleaveLo(c3h)
		he3 := c2h.InterleaveHi(c3h)
		x4 := fwdI64AsI32(fwdI32AsI64(he0).InterleaveLo(fwdI32AsI64(he2)))
		x5 := fwdI64AsI32(fwdI32AsI64(he0).InterleaveHi(fwdI32AsI64(he2)))
		x6 := fwdI64AsI32(fwdI32AsI64(he1).InterleaveLo(fwdI32AsI64(he3)))
		x7 := fwdI64AsI32(fwdI32AsI64(he1).InterleaveHi(fwdI32AsI64(he3)))
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
		fwdStoreI32At(coeff, 0*coeffStride+h, s0)
		fwdStoreI32At(coeff, 1*coeffStride+h, s4)
		fwdStoreI32At(coeff, 2*coeffStride+h, s2)
		fwdStoreI32At(coeff, 3*coeffStride+h, s6)
		fwdStoreI32At(coeff, 4*coeffStride+h, s1)
		fwdStoreI32At(coeff, 5*coeffStride+h, s5)
		fwdStoreI32At(coeff, 6*coeffStride+h, s3)
		fwdStoreI32At(coeff, 7*coeffStride+h, s7)
	}
	{
		h := 4

		le0 := c4l.InterleaveLo(c5l)
		le1 := c4l.InterleaveHi(c5l)
		le2 := c6l.InterleaveLo(c7l)
		le3 := c6l.InterleaveHi(c7l)
		x0 := fwdI64AsI32(fwdI32AsI64(le0).InterleaveLo(fwdI32AsI64(le2)))
		x1 := fwdI64AsI32(fwdI32AsI64(le0).InterleaveHi(fwdI32AsI64(le2)))
		x2 := fwdI64AsI32(fwdI32AsI64(le1).InterleaveLo(fwdI32AsI64(le3)))
		x3 := fwdI64AsI32(fwdI32AsI64(le1).InterleaveHi(fwdI32AsI64(le3)))

		he0 := c4h.InterleaveLo(c5h)
		he1 := c4h.InterleaveHi(c5h)
		he2 := c6h.InterleaveLo(c7h)
		he3 := c6h.InterleaveHi(c7h)
		x4 := fwdI64AsI32(fwdI32AsI64(he0).InterleaveLo(fwdI32AsI64(he2)))
		x5 := fwdI64AsI32(fwdI32AsI64(he0).InterleaveHi(fwdI32AsI64(he2)))
		x6 := fwdI64AsI32(fwdI32AsI64(he1).InterleaveLo(fwdI32AsI64(he3)))
		x7 := fwdI64AsI32(fwdI32AsI64(he1).InterleaveHi(fwdI32AsI64(he3)))
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
		fwdStoreI32At(coeff, 0*coeffStride+h, s0)
		fwdStoreI32At(coeff, 1*coeffStride+h, s4)
		fwdStoreI32At(coeff, 2*coeffStride+h, s2)
		fwdStoreI32At(coeff, 3*coeffStride+h, s6)
		fwdStoreI32At(coeff, 4*coeffStride+h, s1)
		fwdStoreI32At(coeff, 5*coeffStride+h, s5)
		fwdStoreI32At(coeff, 6*coeffStride+h, s3)
		fwdStoreI32At(coeff, 7*coeffStride+h, s7)
	}
}

// forwardDCT8x8RowSkipSIMD is fwd8DCTCore with the row-skip bound of
// forwardDCT8x8RowSkipPureGo between its register-resident column and row
// passes: each cNl/cNh pair is column-pass row N, so |cNl|+|cNh| carries that
// row's magnitude sum in four partial lanes.
func forwardDCT8x8RowSkipSIMD(coeffArr *[64]int32, residualArr *[64]int16, thr0, thr int32) uint8 {
	coeff, coeffStride, residual, residualStride := coeffArr[:], 8, residualArr[:], 8
	var c0l, c1l, c2l, c3l, c4l, c5l, c6l, c7l, c0h, c1h, c2h, c3h, c4h, c5h, c6h, c7h fwdVec
	{
		g := 0
		rs := residualStride

		x0 := fwdLoadResAt(residual, 0*rs+g&^7, g)
		x1 := fwdLoadResAt(residual, 1*rs+g&^7, g)
		x2 := fwdLoadResAt(residual, 2*rs+g&^7, g)
		x3 := fwdLoadResAt(residual, 3*rs+g&^7, g)
		x4 := fwdLoadResAt(residual, 4*rs+g&^7, g)
		x5 := fwdLoadResAt(residual, 5*rs+g&^7, g)
		x6 := fwdLoadResAt(residual, 6*rs+g&^7, g)
		x7 := fwdLoadResAt(residual, 7*rs+g&^7, g)
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
		c0l = fwdRoundShift1V(s0)
		c1l = fwdRoundShift1V(s4)
		c2l = fwdRoundShift1V(s2)
		c3l = fwdRoundShift1V(s6)
		c4l = fwdRoundShift1V(s1)
		c5l = fwdRoundShift1V(s5)
		c6l = fwdRoundShift1V(s3)
		c7l = fwdRoundShift1V(s7)
	}
	{
		g := 4
		rs := residualStride

		x0 := fwdLoadResAt(residual, 0*rs+g&^7, g)
		x1 := fwdLoadResAt(residual, 1*rs+g&^7, g)
		x2 := fwdLoadResAt(residual, 2*rs+g&^7, g)
		x3 := fwdLoadResAt(residual, 3*rs+g&^7, g)
		x4 := fwdLoadResAt(residual, 4*rs+g&^7, g)
		x5 := fwdLoadResAt(residual, 5*rs+g&^7, g)
		x6 := fwdLoadResAt(residual, 6*rs+g&^7, g)
		x7 := fwdLoadResAt(residual, 7*rs+g&^7, g)
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
		c0h = fwdRoundShift1V(s0)
		c1h = fwdRoundShift1V(s4)
		c2h = fwdRoundShift1V(s2)
		c3h = fwdRoundShift1V(s6)
		c4h = fwdRoundShift1V(s1)
		c5h = fwdRoundShift1V(s5)
		c6h = fwdRoundShift1V(s3)
		c7h = fwdRoundShift1V(s7)
	}
	thrRest := archsimd.BroadcastInt32x4(thr)
	var mask uint8
	if fwdRowSkipGroupLive(c0l.Abs().Add(c0h.Abs()), c1l.Abs().Add(c1h.Abs()),
		c2l.Abs().Add(c2h.Abs()), c3l.Abs().Add(c3h.Abs()), thrRest.SetElem(0, thr0), fwdRowSkipSlack8V) {
		mask |= 1
	}
	if fwdRowSkipGroupLive(c4l.Abs().Add(c4h.Abs()), c5l.Abs().Add(c5h.Abs()),
		c6l.Abs().Add(c6h.Abs()), c7l.Abs().Add(c7h.Abs()), thrRest, fwdRowSkipSlack8V) {
		mask |= 2
	}
	if mask == 0 {
		return 0
	}
	zero := archsimd.BroadcastInt32x4(0)
	if mask&1 == 0 {
		for l := range 8 {
			fwdStoreI32At(coeff, l*coeffStride, zero)
		}
	} else {
		h := 0

		le0 := c0l.InterleaveLo(c1l)
		le1 := c0l.InterleaveHi(c1l)
		le2 := c2l.InterleaveLo(c3l)
		le3 := c2l.InterleaveHi(c3l)
		x0 := fwdI64AsI32(fwdI32AsI64(le0).InterleaveLo(fwdI32AsI64(le2)))
		x1 := fwdI64AsI32(fwdI32AsI64(le0).InterleaveHi(fwdI32AsI64(le2)))
		x2 := fwdI64AsI32(fwdI32AsI64(le1).InterleaveLo(fwdI32AsI64(le3)))
		x3 := fwdI64AsI32(fwdI32AsI64(le1).InterleaveHi(fwdI32AsI64(le3)))

		he0 := c0h.InterleaveLo(c1h)
		he1 := c0h.InterleaveHi(c1h)
		he2 := c2h.InterleaveLo(c3h)
		he3 := c2h.InterleaveHi(c3h)
		x4 := fwdI64AsI32(fwdI32AsI64(he0).InterleaveLo(fwdI32AsI64(he2)))
		x5 := fwdI64AsI32(fwdI32AsI64(he0).InterleaveHi(fwdI32AsI64(he2)))
		x6 := fwdI64AsI32(fwdI32AsI64(he1).InterleaveLo(fwdI32AsI64(he3)))
		x7 := fwdI64AsI32(fwdI32AsI64(he1).InterleaveHi(fwdI32AsI64(he3)))
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
		fwdStoreI32At(coeff, 0*coeffStride+h, s0)
		fwdStoreI32At(coeff, 1*coeffStride+h, s4)
		fwdStoreI32At(coeff, 2*coeffStride+h, s2)
		fwdStoreI32At(coeff, 3*coeffStride+h, s6)
		fwdStoreI32At(coeff, 4*coeffStride+h, s1)
		fwdStoreI32At(coeff, 5*coeffStride+h, s5)
		fwdStoreI32At(coeff, 6*coeffStride+h, s3)
		fwdStoreI32At(coeff, 7*coeffStride+h, s7)
	}
	if mask&2 == 0 {
		// Unrolled: the loop form leaves one bounds check (l*8+8 <= 64 is not proven).
		fwdStoreI32At(coeff, 0*coeffStride+4, zero)
		fwdStoreI32At(coeff, 1*coeffStride+4, zero)
		fwdStoreI32At(coeff, 2*coeffStride+4, zero)
		fwdStoreI32At(coeff, 3*coeffStride+4, zero)
		fwdStoreI32At(coeff, 4*coeffStride+4, zero)
		fwdStoreI32At(coeff, 5*coeffStride+4, zero)
		fwdStoreI32At(coeff, 6*coeffStride+4, zero)
		fwdStoreI32At(coeff, 7*coeffStride+4, zero)
	} else {
		h := 4

		le0 := c4l.InterleaveLo(c5l)
		le1 := c4l.InterleaveHi(c5l)
		le2 := c6l.InterleaveLo(c7l)
		le3 := c6l.InterleaveHi(c7l)
		x0 := fwdI64AsI32(fwdI32AsI64(le0).InterleaveLo(fwdI32AsI64(le2)))
		x1 := fwdI64AsI32(fwdI32AsI64(le0).InterleaveHi(fwdI32AsI64(le2)))
		x2 := fwdI64AsI32(fwdI32AsI64(le1).InterleaveLo(fwdI32AsI64(le3)))
		x3 := fwdI64AsI32(fwdI32AsI64(le1).InterleaveHi(fwdI32AsI64(le3)))

		he0 := c4h.InterleaveLo(c5h)
		he1 := c4h.InterleaveHi(c5h)
		he2 := c6h.InterleaveLo(c7h)
		he3 := c6h.InterleaveHi(c7h)
		x4 := fwdI64AsI32(fwdI32AsI64(he0).InterleaveLo(fwdI32AsI64(he2)))
		x5 := fwdI64AsI32(fwdI32AsI64(he0).InterleaveHi(fwdI32AsI64(he2)))
		x6 := fwdI64AsI32(fwdI32AsI64(he1).InterleaveLo(fwdI32AsI64(he3)))
		x7 := fwdI64AsI32(fwdI32AsI64(he1).InterleaveHi(fwdI32AsI64(he3)))
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
		fwdStoreI32At(coeff, 0*coeffStride+h, s0)
		fwdStoreI32At(coeff, 1*coeffStride+h, s4)
		fwdStoreI32At(coeff, 2*coeffStride+h, s2)
		fwdStoreI32At(coeff, 3*coeffStride+h, s6)
		fwdStoreI32At(coeff, 4*coeffStride+h, s1)
		fwdStoreI32At(coeff, 5*coeffStride+h, s5)
		fwdStoreI32At(coeff, 6*coeffStride+h, s3)
		fwdStoreI32At(coeff, 7*coeffStride+h, s7)
	}
	return mask
}

// Every forward DCT dispatch slot is Go SIMD on arm64.
var forwardDCT4x4Impl = forwardDCT4x4SIMD
var forwardDCT8x8Impl = forwardDCT8x8SIMDGuarded
var forwardDCT16x16Impl = forwardDCT16x16SIMDGuarded
var forwardDCT32x32Impl = forwardDCT32x32SIMDGuarded
var forwardDCT4x4Trusted8BitImpl = forwardDCT4x4SIMDCore
var forwardDCT8x8Trusted8BitImpl = forwardDCT8x8SIMD
var forwardDCT16x16Trusted8BitImpl = forwardDCT16x16SIMD
var forwardDCT32x32Trusted8BitImpl = forwardDCT32x32SIMD
var forwardDCT64x64Trusted8BitImpl = forwardDCT64x64SIMD
var forwardDCT8x8N4Impl = forwardDCT8x8N4SIMD

// Pre-broadcast fdct4 twiddle vectors, kept as package-level (rodata) arrays so
// each loads with a single instruction instead of a per-call stack fill. This
// mirrors how the hand-written NEON kernels keep their coefficients in a loaded
// constant vector rather than re-broadcasting a scalar every use.
var (
	fdct4W32   = [4]int32{fwdCospi13[32], fwdCospi13[32], fwdCospi13[32], fwdCospi13[32]}
	fdct4Wn32  = [4]int32{-fwdCospi13[32], -fwdCospi13[32], -fwdCospi13[32], -fwdCospi13[32]}
	fdct4W48   = [4]int32{fwdCospi13[48], fwdCospi13[48], fwdCospi13[48], fwdCospi13[48]}
	fdct4W16   = [4]int32{fwdCospi13[16], fwdCospi13[16], fwdCospi13[16], fwdCospi13[16]}
	fdct4Wn16  = [4]int32{-fwdCospi13[16], -fwdCospi13[16], -fwdCospi13[16], -fwdCospi13[16]}
	fdct4Round = [4]int32{1 << 12, 1 << 12, 1 << 12, 1 << 12}
)

func forwardDCT4x4SIMD(coeff []int32, coeffStride int, residual []int16, residualStride int) {
	// One straight-line pass: load 4 rows of int16 residual widened to int32
	// (<<2), fdct4 column butterfly, in-register 4x4 int32 transpose, fdct4 row
	// butterfly, row store. shift[1]=shift[2]=0 for 4x4 so there is no
	// inter-pass round-shift; everything stays in Int32x4 registers.
	// The first-pass output magnitude is at most 46341 and the largest
	// second-pass accumulator at most 2147446036 for |residual| <= 2048. Larger
	// public int16 inputs use the scalar path, whose multiply-accumulates are
	// int64.
	if !residualFitsMagnitude(residual, residualStride, 4, 4, 2048) {
		forwardDCT4x4PureGo(coeff, coeffStride, residual, residualStride)
		return
	}
	forwardDCT4x4SIMDCore(coeff, coeffStride, residual, residualStride)
}

func forwardDCT4x4SIMDCore(coeff []int32, coeffStride int, residual []int16, residualStride int) {
	if coeffStride == 4 && residualStride == 4 {
		forwardDCT4x4Packed((*[16]int32)(coeff[:16]), (*[16]int16)(residual[:16]))
		return
	}
	var r [16]int16
	var c [16]int32
	for i := range 4 {
		copy(r[4*i:4*i+4], residual[i*residualStride:i*residualStride+4])
	}
	forwardDCT4x4Packed(&c, &r)
	for i := range 4 {
		copy(coeff[i*coeffStride:i*coeffStride+4], c[4*i:4*i+4])
	}
}

// forwardDCT4x4Packed is forwardDCT4x4SIMDCore on a packed 4x4 block (both
// strides 4, the encoder's layout): two eight-sample loads cover the four
// residual rows, and the rows are stored at constant offsets.
func forwardDCT4x4Packed(coeff *[16]int32, residual *[16]int16) {
	shL2 := archsimd.BroadcastInt32x4(2)
	shR13 := archsimd.BroadcastInt32x4(-13)
	v01 := archsimd.LoadInt16x8Array((*[8]int16)(residual[0:8]))
	v23 := archsimd.LoadInt16x8Array((*[8]int16)(residual[8:16]))
	r0 := v01.ExtendLo4ToInt32().Shift(shL2)
	r1 := v01.HiToLo().ExtendLo4ToInt32().Shift(shL2)
	r2 := v23.ExtendLo4ToInt32().Shift(shL2)
	r3 := v23.HiToLo().ExtendLo4ToInt32().Shift(shL2)

	// Load the pre-broadcast fdct4 twiddles (single VLD1/FMOVQ each, straight
	// from rodata; shared by both passes). This is far cheaper than
	// BroadcastInt32x4 of a scalar, which materializes the value in a GPR and
	// does VMOV+VDUP (three instructions plus a preserving move) per twiddle.
	w32 := archsimd.LoadInt32x4Array(&fdct4W32)
	wn32 := archsimd.LoadInt32x4Array(&fdct4Wn32)
	w48 := archsimd.LoadInt32x4Array(&fdct4W48)
	w16 := archsimd.LoadInt32x4Array(&fdct4W16)
	wn16 := archsimd.LoadInt32x4Array(&fdct4Wn16)
	round := archsimd.LoadInt32x4Array(&fdct4Round)

	// --- Column pass (inlined fdct4; output in (t0,t2,t1,t3) order) ---
	// Each half-butterfly is (a*w0 + b*w1 + round) >> 13, built with fused
	// multiply-adds (VMLA) so there is one VMUL + one VMLA per lane instead of
	// two VMUL + two VADD.
	var c0, c1, c2, c3 archsimd.Int32x4
	{
		s0 := r0.Add(r3)
		s1 := r1.Add(r2)
		s2 := r1.Sub(r2)
		s3 := r0.Sub(r3)
		c0 = s1.MulAdd(w32, s0.MulAdd(w32, round)).Shift(shR13)
		c2 = s0.MulAdd(w32, s1.MulAdd(wn32, round)).Shift(shR13)
		c1 = s3.MulAdd(w16, s2.MulAdd(w48, round)).Shift(shR13)
		c3 = s2.MulAdd(wn16, s3.MulAdd(w48, round)).Shift(shR13)
	}

	// In-register 4x4 int32 transpose.
	e0 := c0.InterleaveLo(c1)
	e1 := c0.InterleaveHi(c1)
	e2 := c2.InterleaveLo(c3)
	e3 := c2.InterleaveHi(c3)
	t0 := fdctInt64AsInt32(fdctInt32AsInt64(e0).InterleaveLo(fdctInt32AsInt64(e2)))
	t1 := fdctInt64AsInt32(fdctInt32AsInt64(e0).InterleaveHi(fdctInt32AsInt64(e2)))
	t2 := fdctInt64AsInt32(fdctInt32AsInt64(e1).InterleaveLo(fdctInt32AsInt64(e3)))
	t3 := fdctInt64AsInt32(fdctInt32AsInt64(e1).InterleaveHi(fdctInt32AsInt64(e3)))

	// --- Row pass (inlined fdct4) ---
	var o0, o1, o2, o3 archsimd.Int32x4
	{
		s0 := t0.Add(t3)
		s1 := t1.Add(t2)
		s2 := t1.Sub(t2)
		s3 := t0.Sub(t3)
		o0 = s1.MulAdd(w32, s0.MulAdd(w32, round)).Shift(shR13)
		o2 = s0.MulAdd(w32, s1.MulAdd(wn32, round)).Shift(shR13)
		o1 = s3.MulAdd(w16, s2.MulAdd(w48, round)).Shift(shR13)
		o3 = s2.MulAdd(wn16, s3.MulAdd(w48, round)).Shift(shR13)
	}

	o0.StoreArray((*[4]int32)(coeff[0:4]))
	o1.StoreArray((*[4]int32)(coeff[4:8]))
	o2.StoreArray((*[4]int32)(coeff[8:12]))
	o3.StoreArray((*[4]int32)(coeff[12:16]))
}

func fdctInt32AsInt64(v archsimd.Int32x4) archsimd.Int64x2 {
	return v.ToBits().ReshapeToUint64s().BitsToInt64()
}

func fdctInt64AsInt32(v archsimd.Int64x2) archsimd.Int32x4 {
	return v.ToBits().ReshapeToUint32s().BitsToInt32()
}

// The arm64 networks (fdct_net16_32_gen_arm64.go) load their twiddles off
// one table pointer and spell each butterfly out as intrinsics, so every
// operand is a single load at a constant offset and no inline-mark NOPs
// separate the operations. Their column passes load rows 0..n-2 as the eight
// samples from column g and keep the low four (those reads end inside the
// next row; only the last row takes the in-row load of fwdLoadResAt) and
// store their outputs transposed, so the row passes run in place.

// fwdTw12 is the cos_bit 12 twiddle table of the 16- and 32-point networks.
var fwdTw12 = newFwd64Twiddles(&fwdCospi12, 12)

// forwardDCT16x16SIMD is the 16x16 DCT_DCT kernel (forwardDCT16x16PureGo).
// The column pass stores its rows transposed, so coeff[c*16+k] holds
// column-pass row k of column c, and the row pass then transforms each group
// of four rows in place.
func forwardDCT16x16SIMD(coeff []int32, coeffStride int, residual []int16, residualStride int) {
	if coeffStride == 16 {
		forwardDCT16x16InPlace((*[256]int32)(coeff), residual, residualStride)
		return
	}
	var buf [256]int32
	forwardDCT16x16InPlace(&buf, residual, residualStride)
	for k := range 16 {
		copy(coeff[k*coeffStride:k*coeffStride+16], buf[k*16:])
	}
}

func forwardDCT16x16InPlace(c *[256]int32, residual []int16, residualStride int) {
	for g := 0; g < 16; g += fwdLanes {
		fwdColDCT16(c[:], g, residual, residualStride)
	}
	for h := 0; h < 16; h += fwdLanes {
		fwdRowDCT16(c[:], h)
	}
}

// forwardDCT32x32SIMD is the 32x32 DCT_DCT kernel (forwardDCT32x32PureGo),
// laid out like forwardDCT16x16SIMD.
func forwardDCT32x32SIMD(coeff []int32, coeffStride int, residual []int16, residualStride int) {
	if coeffStride == 32 {
		forwardDCT32x32InPlace((*[1024]int32)(coeff), residual, residualStride)
		return
	}
	var buf [1024]int32
	forwardDCT32x32InPlace(&buf, residual, residualStride)
	for k := range 32 {
		copy(coeff[k*coeffStride:k*coeffStride+32], buf[k*32:])
	}
}

func forwardDCT32x32InPlace(c *[1024]int32, residual []int16, residualStride int) {
	for g := 0; g < 32; g += fwdLanes {
		fwdColDCT32(c[:], g, residual, residualStride)
	}
	for h := 0; h < 32; h += fwdLanes {
		fwdRowDCT32(c[:], h)
	}
}

// fwdVec is one 128-bit lane group of the forward-transform networks: four
// independent 1-D lines per vector (fdct_gosimd_net.go).
type fwdVec = archsimd.Int32x4

// fwdLanes is the number of 1-D lines a fwdVec carries.
const fwdLanes = 4

func fwdBcast(v int32) fwdVec {
	return archsimd.BroadcastInt32x4(v)
}

// Per-lane shift amounts. archsimd's immediate ShiftAllRight/ShiftAllLeft
// lower to VSSHL with the amount rebuilt on every use (a move, a duplicate and
// a negate each time); a register-amount Shift with a package-level vector
// keeps one load per use instead.
var (
	fwdShrNeg1  = archsimd.BroadcastInt32x4(-1)
	fwdShrNeg2  = archsimd.BroadcastInt32x4(-2)
	fwdShrNeg4  = archsimd.BroadcastInt32x4(-4)
	fwdShrNeg10 = archsimd.BroadcastInt32x4(-10)
	fwdShrNeg12 = archsimd.BroadcastInt32x4(-12)
	fwdShrNeg13 = archsimd.BroadcastInt32x4(-13)
	fwdShrNeg31 = archsimd.BroadcastInt32x4(-31)
	fwdShlPos2  = archsimd.BroadcastInt32x4(2)
)

// fwdShr is an arithmetic right shift by the constant n in every lane; the
// networks call it only with the constants listed here.
func fwdShr(v fwdVec, n uint64) fwdVec {
	switch n {
	case 1:
		return v.Shift(fwdShrNeg1)
	case 2:
		return v.Shift(fwdShrNeg2)
	case 4:
		return v.Shift(fwdShrNeg4)
	case 10:
		return v.Shift(fwdShrNeg10)
	case 12:
		return v.Shift(fwdShrNeg12)
	case 13:
		return v.Shift(fwdShrNeg13)
	case 31:
		return v.Shift(fwdShrNeg31)
	}
	return v.ShiftAllRight(n)
}

// fwdShl is a left shift by the constant n in every lane.
func fwdShl(v fwdVec, n uint64) fwdVec {
	if n == 2 {
		return v.Shift(fwdShlPos2)
	}
	return v.ShiftAllLeft(n)
}

// fwdTranspose writes the transpose of the n x n row-major matrix src into dst
// (dst[c*n+r] = src[r*n+c]); n is a multiple of four.
func fwdTranspose(dst, src []int32, n int) {
	for bi := 0; bi < n; bi += 4 {
		for bj := 0; bj < n; bj += 4 {
			fwdTransposeBlock4(dst[bj*n+bi:], n, src[bi*n+bj:], n)
		}
	}
}

// fwdTransposeBlock4 transposes the 4x4 block at src (row stride sn) into dst
// (row stride dn) in registers: interleave the row pairs at 32 bits, then
// combine the pairs at 64 bits. The guards bound every row it touches (a
// stride indexes inside its slice, so 3*stride cannot overflow), so the rows
// are addressed from the slice bases.
func fwdTransposeBlock4(dst []int32, dn int, src []int32, sn int) {
	_, _ = src[sn], src[3*sn+3]
	_, _ = dst[dn], dst[3*dn+3]
	v0 := fwdLoadI32At(src, 0)
	v1 := fwdLoadI32At(src, sn)
	v2 := fwdLoadI32At(src, 2*sn)
	v3 := fwdLoadI32At(src, 3*sn)
	e0 := v0.InterleaveLo(v1)
	e1 := v0.InterleaveHi(v1)
	e2 := v2.InterleaveLo(v3)
	e3 := v2.InterleaveHi(v3)
	fwdStoreI32At(dst, 0, fwdI64AsI32(fwdI32AsI64(e0).InterleaveLo(fwdI32AsI64(e2))))
	fwdStoreI32At(dst, dn, fwdI64AsI32(fwdI32AsI64(e0).InterleaveHi(fwdI32AsI64(e2))))
	fwdStoreI32At(dst, 2*dn, fwdI64AsI32(fwdI32AsI64(e1).InterleaveLo(fwdI32AsI64(e3))))
	fwdStoreI32At(dst, 3*dn, fwdI64AsI32(fwdI32AsI64(e1).InterleaveHi(fwdI32AsI64(e3))))
}

func fwdI32AsI64(v fwdVec) archsimd.Int64x2 {
	return v.ToBits().ReshapeToUint64s().BitsToInt64()
}

func fwdI64AsI32(v archsimd.Int64x2) fwdVec {
	return v.ToBits().ReshapeToUint32s().BitsToInt32()
}

// The network drivers have already validated their buffers; each access is
// one checked fixed-length window.
func fwdLoadI32At(s []int32, i int) fwdVec {
	return archsimd.LoadInt32x4Array((*[4]int32)(s[i : i+4]))
}

func fwdStoreI32At(s []int32, i int, v fwdVec) {
	v.StoreArray((*[4]int32)(s[i : i+4]))
}

func fwdLoadResAt(s []int16, i, g int) fwdVec {
	w := archsimd.LoadInt16x8Array((*[8]int16)(s[i : i+8]))
	if g&4 == 0 {
		return fwdShl(w.ExtendLo4ToInt32(), 2)
	}
	return fwdShl(w.HiToLo().ExtendLo4ToInt32(), 2)
}

func fwdLoadI16At(s []int16, i int) archsimd.Int16x8 {
	return archsimd.LoadInt16x8Array((*[8]int16)(s[i : i+8]))
}

// fwdLoadI32Arr and fwdStoreI32Arr address a fixed-length view at a constant
// offset, which folds into the vector load or store.
func fwdLoadI32Arr(s []int32, i int) fwdVec {
	return archsimd.LoadInt32x4Array((*[4]int32)(s[i : i+4]))
}

func fwdStoreI32Arr(s []int32, i int, v fwdVec) {
	v.StoreArray((*[4]int32)(s[i : i+4]))
}

// fwdMulAdd computes a*w + acc. The arm64 intrinsic maps to MLA; AVX2
// has no integer multiply-add instruction, so its port uses two operations.
func fwdMulAdd(a, w, acc fwdVec) fwdVec {
	return a.MulAdd(w, acc)
}

// fwdHalfBtf13V computes the rounded Q13 forward butterfly.
func fwdHalfBtf13V(w0, a, w1, b fwdVec) fwdVec {
	return fwdShr(b.MulAdd(w1, a.MulAdd(w0, fwdRound13)), 13)
}

// fwdHalfBtf12V computes the rounded Q12 forward butterfly.
func fwdHalfBtf12V(w0, a, w1, b fwdVec) fwdVec {
	return fwdShr(b.MulAdd(w1, a.MulAdd(w0, fwdRound12)), 12)
}

// fwdLoadResRawAt is fwdLoadResAt without the shift[0] = 2 pre-scale (the
// 64-point transforms take shift[0] = 0): i addresses the 8-sample chunk
// containing column group g.
func fwdLoadResRawAt(s []int16, i, g int) fwdVec {
	w := archsimd.LoadInt16x8Array((*[8]int16)(s[i : i+8]))
	if g&4 == 0 {
		return w.ExtendLo4ToInt32()
	}
	return w.HiToLo().ExtendLo4ToInt32()
}

// fwdShift is a runtime arithmetic right-shift amount for fwdShrBy: the
// negated count broadcast for the register-amount SSHL.
type fwdShift = fwdVec

func fwdMakeShift(n uint64) fwdShift { return archsimd.BroadcastInt32x4(-int32(n)) }

func fwdShrBy(v fwdVec, s fwdShift) fwdVec { return v.Shift(s) }

// fwdVecDone is a no-op on arm64 (see the amd64 counterpart).
func fwdVecDone() {}

// forwardDCTRect8BitTrusted runs the 2:1 DCT_DCT rectangles 16x8, 8x16, 32x16
// and 16x32 (svt_av1_fwd_txfm2d_{16x8,8x16,32x16,16x32}_neon) for 8-bit
// residuals whose regions the caller validated, reporting false for other
// sizes. Like the square kernels, the column pass (fdct_rect_gen_arm64.go,
// fdct_net16_32_gen_arm64.go) stores transposed so the row pass runs in place
// over coeff[c*h+r]; other coefficient strides compute into a local block.
func forwardDCTRect8BitTrusted(coeff []int32, coeffStride int, residual []int16, residualStride int, size Size) bool {
	switch size {
	case Size{Width: 16, Height: 8}:
		if coeffStride == 8 {
			fwdRect16x8((*[128]int32)(coeff), residual, residualStride)
			return true
		}
		var buf [128]int32
		fwdRect16x8(&buf, residual, residualStride)
		fwdRectCopyOut(coeff, coeffStride, buf[:], 16, 8)
	case Size{Width: 8, Height: 16}:
		if coeffStride == 16 {
			fwdRect8x16((*[128]int32)(coeff), residual, residualStride)
			return true
		}
		var buf [128]int32
		fwdRect8x16(&buf, residual, residualStride)
		fwdRectCopyOut(coeff, coeffStride, buf[:], 8, 16)
	case Size{Width: 32, Height: 16}:
		if coeffStride == 16 {
			fwdRect32x16((*[512]int32)(coeff), residual, residualStride)
			return true
		}
		var buf [512]int32
		fwdRect32x16(&buf, residual, residualStride)
		fwdRectCopyOut(coeff, coeffStride, buf[:], 32, 16)
	case Size{Width: 16, Height: 32}:
		if coeffStride == 32 {
			fwdRect16x32((*[512]int32)(coeff), residual, residualStride)
			return true
		}
		var buf [512]int32
		fwdRect16x32(&buf, residual, residualStride)
		fwdRectCopyOut(coeff, coeffStride, buf[:], 16, 32)
	default:
		return false
	}
	return true
}

func fwdRectCopyOut(coeff []int32, coeffStride int, buf []int32, w, h int) {
	for k := range w {
		copy(coeff[k*coeffStride:k*coeffStride+h], buf[k*h:])
	}
}

func fwdRect16x8(c *[128]int32, residual []int16, rs int) {
	for g := 0; g < 16; g += fwdLanes {
		fwdColDCT8T(c[:], g, residual, rs)
	}
	for h := 0; h < 8; h += fwdLanes {
		fwdRowDCT16x8(c[:], h)
	}
}

func fwdRect8x16(c *[128]int32, residual []int16, rs int) {
	for g := 0; g < 8; g += fwdLanes {
		fwdColDCT16(c[:], g, residual, rs)
	}
	for h := 0; h < 16; h += fwdLanes {
		fwdRowDCT8x16(c[:], h)
	}
}

func fwdRect32x16(c *[512]int32, residual []int16, rs int) {
	for g := 0; g < 32; g += fwdLanes {
		fwdColDCT16R4(c[:], g, residual, rs)
	}
	for h := 0; h < 16; h += fwdLanes {
		fwdRowDCT32x16(c[:], h)
	}
}

func fwdRect16x32(c *[512]int32, residual []int16, rs int) {
	for g := 0; g < 16; g += fwdLanes {
		fwdColDCT32(c[:], g, residual, rs)
	}
	for h := 0; h < 32; h += fwdLanes {
		fwdRowDCT16x32(c[:], h)
	}
}

func init() {
	forwardDCT8x8RowSkipImpl = forwardDCT8x8RowSkipSIMD
	forwardDCT16x16RowSkipImpl = forwardDCT16x16RowSkipSIMD
	forwardDCT32x32RowSkipImpl = forwardDCT32x32RowSkipSIMD
}

var (
	fwdRowSkipShrNeg6  = archsimd.BroadcastInt32x4(-6)
	fwdRowSkipSlack8V  = archsimd.BroadcastInt32x4(fwdRowSkipSlack8)
	fwdRowSkipSlack16V = archsimd.BroadcastInt32x4(fwdRowSkipSlack16)
	fwdRowSkipSlack32V = archsimd.BroadcastInt32x4(fwdRowSkipSlack32)
)

// fwdRowSkipGroupLive takes the per-row partial magnitude sums r0..r3 (lane
// sums of |T| for four consecutive column-pass rows) and reports whether any
// row's S + S>>6 + slack reaches its threshold lane.
func fwdRowSkipGroupLive(r0, r1, r2, r3, thr, slack archsimd.Int32x4) bool {
	s := r0.ConcatAddPairs(r1).ConcatAddPairs(r2.ConcatAddPairs(r3))
	m := s.Add(s.Shift(fwdRowSkipShrNeg6)).Add(slack).Sub(thr)
	return m.ReduceMax() >= 0
}

// fwdRowSkipSumsLive takes the magnitude sums s of four column-pass rows (one
// row per lane) and reports whether any row's S + S>>6 + slack reaches its
// threshold lane.
func fwdRowSkipSumsLive(s, thr, slack archsimd.Int32x4) bool {
	m := s.Add(s.Shift(fwdRowSkipShrNeg6)).Add(slack).Sub(thr)
	return m.ReduceMax() >= 0
}

// forwardDCT16x16RowSkipSIMD is forwardDCT16x16RowSkipPureGo over the
// forwardDCT16x16SIMD passes: the column pass runs in full into coeff
// (transposed), then each group of four column-pass rows is bounded from it
// and only live groups are row-transformed in place.
func forwardDCT16x16RowSkipSIMD(coeff *[256]int32, residual *[256]int16, thr0, thr int32) uint8 {
	for g := 0; g < 16; g += fwdLanes {
		fwdColDCT16(coeff[:], g, residual[:], 16)
	}
	var sums [4]archsimd.Int32x4
	for h := 0; h < 16; h += 4 {
		r := (*[15*16 + 4]int32)(coeff[h:])
		var s0, s1 archsimd.Int32x4
		for c := 0; c < 16; c += 2 {
			s0 = s0.Add(fwdLoadI32Arr(r[:], c*16).Abs())
			s1 = s1.Add(fwdLoadI32Arr(r[:], c*16+16).Abs())
		}
		sums[h>>2] = s0.Add(s1)
	}
	return fwdRowSkipRows(coeff[:], 16, sums[:], thr0, thr, fwdRowSkipSlack16V, fwdRowDCT16)
}

// forwardDCT32x32RowSkipSIMD is forwardDCT32x32RowSkipPureGo over the
// forwardDCT32x32SIMD passes (see forwardDCT16x16RowSkipSIMD).
func forwardDCT32x32RowSkipSIMD(coeff *[1024]int32, residual *[1024]int16, thr0, thr int32) uint8 {
	for g := 0; g < 32; g += fwdLanes {
		fwdColDCT32(coeff[:], g, residual[:], 32)
	}
	var sums [8]archsimd.Int32x4
	for h := 0; h < 32; h += 4 {
		r := (*[31*32 + 4]int32)(coeff[h:])
		var s0, s1 archsimd.Int32x4
		for c := 0; c < 32; c += 2 {
			s0 = s0.Add(fwdLoadI32Arr(r[:], c*32).Abs())
			s1 = s1.Add(fwdLoadI32Arr(r[:], c*32+32).Abs())
		}
		sums[h>>2] = s0.Add(s1)
	}
	return fwdRowSkipRows(coeff[:], 32, sums[:], thr0, thr, fwdRowSkipSlack32V, fwdRowDCT32)
}

// fwdRowSkipRows takes the row magnitude sums of each group of four rows of
// the transposed n x n column-pass output c (sums[g] covers rows 4g..4g+3),
// row-transforms the live groups in place and zeroes the others, returning
// the live-group mask (c is left as is when none is).
func fwdRowSkipRows(c []int32, n int, sums []archsimd.Int32x4, thr0, thr int32, slack archsimd.Int32x4, row func([]int32, int)) uint8 {
	thrRest := archsimd.BroadcastInt32x4(thr)
	t := thrRest.SetElem(0, thr0)
	var mask uint8
	for g, s := range sums {
		if fwdRowSkipSumsLive(s, t, slack) {
			mask |= 1 << g
		}
		t = thrRest
	}
	if mask == 0 {
		return 0
	}
	zero := archsimd.BroadcastInt32x4(0)
	c = c[:n*n]
	for h := 0; h < n; h += 4 {
		if mask&(1<<(h>>2)) == 0 {
			for l := 0; l < n; l++ {
				zero.StoreArray((*[4]int32)(c[l*n+h : l*n+h+4]))
			}
			continue
		}
		row(c, h)
	}
	return mask
}

// The hybrid cores keep both passes and the 8x8 transpose in vector locals.
// The guarded public bindings preserve the scalar path for wider residuals.
func fwd8ADSTDCTCore(coeffArr *[64]int32, residualArr *[64]int16) {
	coeff, coeffStride, residual, residualStride := coeffArr[:], 8, residualArr[:], 8
	var c0l, c1l, c2l, c3l, c4l, c5l, c6l, c7l, c0h, c1h, c2h, c3h, c4h, c5h, c6h, c7h fwdVec
	{
		g := 0
		rs := residualStride

		x0 := fwdLoadResAt(residual, 0*rs+g&^7, g)
		x1 := fwdLoadResAt(residual, 1*rs+g&^7, g)
		x2 := fwdLoadResAt(residual, 2*rs+g&^7, g)
		x3 := fwdLoadResAt(residual, 3*rs+g&^7, g)
		x4 := fwdLoadResAt(residual, 4*rs+g&^7, g)
		x5 := fwdLoadResAt(residual, 5*rs+g&^7, g)
		x6 := fwdLoadResAt(residual, 6*rs+g&^7, g)
		x7 := fwdLoadResAt(residual, 7*rs+g&^7, g)
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
		c0l = fwdRoundShift1V(s1)
		c1l = fwdRoundShift1V(s6)
		c2l = fwdRoundShift1V(s3)
		c3l = fwdRoundShift1V(s4)
		c4l = fwdRoundShift1V(s5)
		c5l = fwdRoundShift1V(s2)
		c6l = fwdRoundShift1V(s7)
		c7l = fwdRoundShift1V(s0)
	}
	{
		g := 4
		rs := residualStride

		x0 := fwdLoadResAt(residual, 0*rs+g&^7, g)
		x1 := fwdLoadResAt(residual, 1*rs+g&^7, g)
		x2 := fwdLoadResAt(residual, 2*rs+g&^7, g)
		x3 := fwdLoadResAt(residual, 3*rs+g&^7, g)
		x4 := fwdLoadResAt(residual, 4*rs+g&^7, g)
		x5 := fwdLoadResAt(residual, 5*rs+g&^7, g)
		x6 := fwdLoadResAt(residual, 6*rs+g&^7, g)
		x7 := fwdLoadResAt(residual, 7*rs+g&^7, g)
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
		c0h = fwdRoundShift1V(s1)
		c1h = fwdRoundShift1V(s6)
		c2h = fwdRoundShift1V(s3)
		c3h = fwdRoundShift1V(s4)
		c4h = fwdRoundShift1V(s5)
		c5h = fwdRoundShift1V(s2)
		c6h = fwdRoundShift1V(s7)
		c7h = fwdRoundShift1V(s0)
	}
	{
		h := 0

		le0 := c0l.InterleaveLo(c1l)
		le1 := c0l.InterleaveHi(c1l)
		le2 := c2l.InterleaveLo(c3l)
		le3 := c2l.InterleaveHi(c3l)
		x0 := fwdI64AsI32(fwdI32AsI64(le0).InterleaveLo(fwdI32AsI64(le2)))
		x1 := fwdI64AsI32(fwdI32AsI64(le0).InterleaveHi(fwdI32AsI64(le2)))
		x2 := fwdI64AsI32(fwdI32AsI64(le1).InterleaveLo(fwdI32AsI64(le3)))
		x3 := fwdI64AsI32(fwdI32AsI64(le1).InterleaveHi(fwdI32AsI64(le3)))

		he0 := c0h.InterleaveLo(c1h)
		he1 := c0h.InterleaveHi(c1h)
		he2 := c2h.InterleaveLo(c3h)
		he3 := c2h.InterleaveHi(c3h)
		x4 := fwdI64AsI32(fwdI32AsI64(he0).InterleaveLo(fwdI32AsI64(he2)))
		x5 := fwdI64AsI32(fwdI32AsI64(he0).InterleaveHi(fwdI32AsI64(he2)))
		x6 := fwdI64AsI32(fwdI32AsI64(he1).InterleaveLo(fwdI32AsI64(he3)))
		x7 := fwdI64AsI32(fwdI32AsI64(he1).InterleaveHi(fwdI32AsI64(he3)))
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
		fwdStoreI32At(coeff, 0*coeffStride+h, s0)
		fwdStoreI32At(coeff, 1*coeffStride+h, s4)
		fwdStoreI32At(coeff, 2*coeffStride+h, s2)
		fwdStoreI32At(coeff, 3*coeffStride+h, s6)
		fwdStoreI32At(coeff, 4*coeffStride+h, s1)
		fwdStoreI32At(coeff, 5*coeffStride+h, s5)
		fwdStoreI32At(coeff, 6*coeffStride+h, s3)
		fwdStoreI32At(coeff, 7*coeffStride+h, s7)
	}
	{
		h := 4

		le0 := c4l.InterleaveLo(c5l)
		le1 := c4l.InterleaveHi(c5l)
		le2 := c6l.InterleaveLo(c7l)
		le3 := c6l.InterleaveHi(c7l)
		x0 := fwdI64AsI32(fwdI32AsI64(le0).InterleaveLo(fwdI32AsI64(le2)))
		x1 := fwdI64AsI32(fwdI32AsI64(le0).InterleaveHi(fwdI32AsI64(le2)))
		x2 := fwdI64AsI32(fwdI32AsI64(le1).InterleaveLo(fwdI32AsI64(le3)))
		x3 := fwdI64AsI32(fwdI32AsI64(le1).InterleaveHi(fwdI32AsI64(le3)))

		he0 := c4h.InterleaveLo(c5h)
		he1 := c4h.InterleaveHi(c5h)
		he2 := c6h.InterleaveLo(c7h)
		he3 := c6h.InterleaveHi(c7h)
		x4 := fwdI64AsI32(fwdI32AsI64(he0).InterleaveLo(fwdI32AsI64(he2)))
		x5 := fwdI64AsI32(fwdI32AsI64(he0).InterleaveHi(fwdI32AsI64(he2)))
		x6 := fwdI64AsI32(fwdI32AsI64(he1).InterleaveLo(fwdI32AsI64(he3)))
		x7 := fwdI64AsI32(fwdI32AsI64(he1).InterleaveHi(fwdI32AsI64(he3)))
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
		fwdStoreI32At(coeff, 0*coeffStride+h, s0)
		fwdStoreI32At(coeff, 1*coeffStride+h, s4)
		fwdStoreI32At(coeff, 2*coeffStride+h, s2)
		fwdStoreI32At(coeff, 3*coeffStride+h, s6)
		fwdStoreI32At(coeff, 4*coeffStride+h, s1)
		fwdStoreI32At(coeff, 5*coeffStride+h, s5)
		fwdStoreI32At(coeff, 6*coeffStride+h, s3)
		fwdStoreI32At(coeff, 7*coeffStride+h, s7)
	}
}

// The 8x8 hybrid kernels and the IDTX kernel are Go SIMD (fhybrid_idtx and
// fdct_gosimd_net.go).
var forwardBlock8x8ADSTDCTImpl = forwardBlock8x8ADSTDCTSIMDGuarded
var forwardBlock8x8DCTADSTImpl = forwardBlock8x8DCTADSTSIMDGuarded
var forwardBlock8x8ADSTADSTImpl = forwardBlock8x8ADSTADSTSIMDGuarded
var forwardBlock8x8IDTXImpl = forwardBlock8x8IDTXSIMD

func forwardBlock8x8Hybrid8BitResidualTrusted(coeff []int32, coeffStride int, residual []int16, residualStride int, scratch []int32, typ Type) bool {
	switch typ {
	case TypeADSTDCT:
		forwardBlock8x8ADSTDCTSIMD(coeff, coeffStride, residual, residualStride, scratch)
	case TypeDCTADST:
		forwardBlock8x8DCTADSTSIMD(coeff, coeffStride, residual, residualStride, scratch)
	case TypeADSTADST:
		forwardBlock8x8ADSTADSTSIMD(coeff, coeffStride, residual, residualStride, scratch)
	case TypeIDTX:
		forwardBlock8x8IDTXSIMD(coeff, coeffStride, residual, residualStride, scratch)
	default:
		return false
	}
	return true
}

func adstInt16AsInt32(v archsimd.Int16x8) archsimd.Int32x4 {
	return v.ToBits().ReshapeToUint32s().BitsToInt32()
}
func adstInt32AsInt16(v archsimd.Int32x4) archsimd.Int16x8 {
	return v.ToBits().ReshapeToUint16s().BitsToInt16()
}
func adstInt16AsInt64(v archsimd.Int16x8) archsimd.Int64x2 {
	return v.ToBits().ReshapeToUint64s().BitsToInt64()
}
func adstInt64AsInt16(v archsimd.Int64x2) archsimd.Int16x8 {
	return v.ToBits().ReshapeToUint16s().BitsToInt16()
}

func idtxStoreInt16x8ToInt32(dst *[8]int32, v archsimd.Int16x8) {
	// Scale after sign extension so every int16 residual preserves its full
	// public value range before being promoted to the coefficient type.
	lo := v.ExtendLo4ToInt32().ShiftAllLeft(3)
	hi := v.HiToLo().ExtendLo4ToInt32().ShiftAllLeft(3)
	lo.StoreArray((*[4]int32)(dst[0:4]))
	hi.StoreArray((*[4]int32)(dst[4:8]))
}

// forwardBlock8x8IDTXSIMD is the IDTX identity scale (coeff[c*stride+r]=res[r][c]<<3).
// The encoder's 8x8 blocks are packed (both strides 8), so that shape reads
// and writes one fixed 64-entry view of each buffer with no row checks; any
// other stride stages its rows through a packed copy.
func forwardBlock8x8IDTXSIMD(coeff []int32, coeffStride int, residual []int16, residualStride int, scratch []int32) {
	_ = scratch[63]
	if residualStride == 8 && coeffStride == 8 {
		forwardBlock8x8IDTXPacked((*[64]int32)(coeff[:64]), (*[64]int16)(residual[:64]))
		return
	}
	var r [64]int16
	var c [64]int32
	for i := range 8 {
		copy(r[i*8:i*8+8], residual[i*residualStride:i*residualStride+8])
	}
	forwardBlock8x8IDTXPacked(&c, &r)
	for i := range 8 {
		copy(coeff[i*coeffStride:i*coeffStride+8], c[i*8:i*8+8])
	}
}

// forwardBlock8x8IDTXPacked is forwardBlock8x8IDTXSIMD on packed 8x8 blocks.
func forwardBlock8x8IDTXPacked(coeff *[64]int32, residual *[64]int16) {
	l0 := archsimd.LoadInt16x8Array((*[8]int16)(residual[0:8]))
	l1 := archsimd.LoadInt16x8Array((*[8]int16)(residual[8:16]))
	l2 := archsimd.LoadInt16x8Array((*[8]int16)(residual[16:24]))
	l3 := archsimd.LoadInt16x8Array((*[8]int16)(residual[24:32]))
	l4 := archsimd.LoadInt16x8Array((*[8]int16)(residual[32:40]))
	l5 := archsimd.LoadInt16x8Array((*[8]int16)(residual[40:48]))
	l6 := archsimd.LoadInt16x8Array((*[8]int16)(residual[48:56]))
	l7 := archsimd.LoadInt16x8Array((*[8]int16)(residual[56:64]))

	// Transpose the residual and widen the result to coeff layout on store.
	a0 := l0.InterleaveLo(l1)
	a1 := l0.InterleaveHi(l1)
	a2 := l2.InterleaveLo(l3)
	a3 := l2.InterleaveHi(l3)
	a4 := l4.InterleaveLo(l5)
	a5 := l4.InterleaveHi(l5)
	a6 := l6.InterleaveLo(l7)
	a7 := l6.InterleaveHi(l7)
	b0 := adstInt32AsInt16(adstInt16AsInt32(a0).InterleaveLo(adstInt16AsInt32(a2)))
	b1 := adstInt32AsInt16(adstInt16AsInt32(a0).InterleaveHi(adstInt16AsInt32(a2)))
	b2 := adstInt32AsInt16(adstInt16AsInt32(a1).InterleaveLo(adstInt16AsInt32(a3)))
	b3 := adstInt32AsInt16(adstInt16AsInt32(a1).InterleaveHi(adstInt16AsInt32(a3)))
	b4 := adstInt32AsInt16(adstInt16AsInt32(a4).InterleaveLo(adstInt16AsInt32(a6)))
	b5 := adstInt32AsInt16(adstInt16AsInt32(a4).InterleaveHi(adstInt16AsInt32(a6)))
	b6 := adstInt32AsInt16(adstInt16AsInt32(a5).InterleaveLo(adstInt16AsInt32(a7)))
	b7 := adstInt32AsInt16(adstInt16AsInt32(a5).InterleaveHi(adstInt16AsInt32(a7)))
	o0 := adstInt64AsInt16(adstInt16AsInt64(b0).InterleaveLo(adstInt16AsInt64(b4)))
	o1 := adstInt64AsInt16(adstInt16AsInt64(b0).InterleaveHi(adstInt16AsInt64(b4)))
	o2 := adstInt64AsInt16(adstInt16AsInt64(b1).InterleaveLo(adstInt16AsInt64(b5)))
	o3 := adstInt64AsInt16(adstInt16AsInt64(b1).InterleaveHi(adstInt16AsInt64(b5)))
	o4 := adstInt64AsInt16(adstInt16AsInt64(b2).InterleaveLo(adstInt16AsInt64(b6)))
	o5 := adstInt64AsInt16(adstInt16AsInt64(b2).InterleaveHi(adstInt16AsInt64(b6)))
	o6 := adstInt64AsInt16(adstInt16AsInt64(b3).InterleaveLo(adstInt16AsInt64(b7)))
	o7 := adstInt64AsInt16(adstInt16AsInt64(b3).InterleaveHi(adstInt16AsInt64(b7)))

	idtxStoreInt16x8ToInt32((*[8]int32)(coeff[0:8]), o0)
	idtxStoreInt16x8ToInt32((*[8]int32)(coeff[8:16]), o1)
	idtxStoreInt16x8ToInt32((*[8]int32)(coeff[16:24]), o2)
	idtxStoreInt16x8ToInt32((*[8]int32)(coeff[24:32]), o3)
	idtxStoreInt16x8ToInt32((*[8]int32)(coeff[32:40]), o4)
	idtxStoreInt16x8ToInt32((*[8]int32)(coeff[40:48]), o5)
	idtxStoreInt16x8ToInt32((*[8]int32)(coeff[48:56]), o6)
	idtxStoreInt16x8ToInt32((*[8]int32)(coeff[56:64]), o7)
}

// stageRect2SafeBound is the row-clamp magnitude bound under which the rect2
// pre-clamp argument above holds: rect2Scale(1<<22) = (1<<22*181+128)>>8 =
// 2966528+, and every |lo|,|hi| below this bound clamps saturated inputs
// identically to the scalar clipInt32 path.
const stageRect2SafeBound = 2966528

// stageTransposeClamp is the SIMD staging kernel. It works in 4x4 tiles: a
// shape with fewer than four rows still loads full tiles (the column-major
// coefficient block is at least four rows tall) and stores only the rows it
// stages, and a row count that is not a multiple of four finishes with a tile
// overlapping the previous one, rewriting those rows with identical values.
// Single-row stagings (a strided gather either way; the tile would load four
// rows to keep one), blocks narrower than one tile, and rect2 calls with
// out-of-contract clamp bounds (which production never produces) use the
// scalar reference.
func stageTransposeClamp(scratch []int32, width int, coeff []int32, coeffStride int, rows int, cols int, rect2 bool, lo int32, hi int32) {
	colsBlk := cols &^ 3
	if rows <= 1 || cols < 4 || coeffStride < 4 || len(coeff) < (colsBlk-1)*coeffStride+max(rows, 4) ||
		(rect2 && (hi >= stageRect2SafeBound || lo <= -stageRect2SafeBound)) {
		stageTransposeClampScalar(scratch, width, coeff, coeffStride, rows, cols, rect2, lo, hi)
		return
	}
	k := stageClampKernel{
		lo: archsimd.BroadcastInt32x4(lo),
		hi: archsimd.BroadcastInt32x4(hi),
	}
	if rect2 {
		k.preLo = archsimd.BroadcastInt32x4(-(1 << 22))
		k.preHi = archsimd.BroadcastInt32x4(1 << 22)
		k.k181 = archsimd.BroadcastInt32x4(181)
		k.r128 = archsimd.BroadcastInt32x4(128)
	}
	if rows < 4 {
		for c := 0; c < colsBlk; c += 4 {
			t0, t1, t2, _ := k.tile(coeff, c*coeffStride, coeffStride, rect2)
			t0.StoreArray((*[4]int32)(scratch[c:]))
			if rows > 1 {
				t1.StoreArray((*[4]int32)(scratch[width+c:]))
			}
			if rows > 2 {
				t2.StoreArray((*[4]int32)(scratch[2*width+c:]))
			}
		}
	} else {
		for r := 0; r < rows; r += 4 {
			if r+4 > rows {
				r = rows - 4
			}
			for c := 0; c < colsBlk; c += 4 {
				t0, t1, t2, t3 := k.tile(coeff, c*coeffStride+r, coeffStride, rect2)
				t0.StoreArray((*[4]int32)(scratch[(r+0)*width+c:]))
				t1.StoreArray((*[4]int32)(scratch[(r+1)*width+c:]))
				t2.StoreArray((*[4]int32)(scratch[(r+2)*width+c:]))
				t3.StoreArray((*[4]int32)(scratch[(r+3)*width+c:]))
			}
		}
	}
	// Column tail: cols not covered by a 4-wide tile.
	if colsBlk < cols {
		for row := 0; row < rows; row++ {
			line := scratch[row*width : row*width+cols : row*width+cols]
			if rect2 {
				for col := colsBlk; col < cols; col++ {
					line[col] = clipRange(int64(rect2Scale(coeff[col*coeffStride+row])), lo, hi)
				}
			} else {
				for col := colsBlk; col < cols; col++ {
					line[col] = clipRange(int64(coeff[col*coeffStride+row]), lo, hi)
				}
			}
		}
	}
}

// stageClampKernel holds the broadcast clamp (and rect2 scale) constants.
type stageClampKernel struct {
	lo, hi, preLo, preHi, k181, r128 archsimd.Int32x4
}

// tile loads the 4x4 tile whose first column starts at coeff[base] (columns
// coeffStride apart), clamps it (rect2-scaling first when set) and returns it
// transposed: tN holds row N of the tile.
func (k *stageClampKernel) tile(coeff []int32, base int, coeffStride int, rect2 bool) (t0, t1, t2, t3 archsimd.Int32x4) {
	v0 := archsimd.LoadInt32x4Array((*[4]int32)(coeff[base:]))
	v1 := archsimd.LoadInt32x4Array((*[4]int32)(coeff[base+coeffStride:]))
	v2 := archsimd.LoadInt32x4Array((*[4]int32)(coeff[base+2*coeffStride:]))
	v3 := archsimd.LoadInt32x4Array((*[4]int32)(coeff[base+3*coeffStride:]))
	if rect2 {
		v0 = v0.Max(k.preLo).Min(k.preHi).MulAdd(k.k181, k.r128).ShiftAllRight(8)
		v1 = v1.Max(k.preLo).Min(k.preHi).MulAdd(k.k181, k.r128).ShiftAllRight(8)
		v2 = v2.Max(k.preLo).Min(k.preHi).MulAdd(k.k181, k.r128).ShiftAllRight(8)
		v3 = v3.Max(k.preLo).Min(k.preHi).MulAdd(k.k181, k.r128).ShiftAllRight(8)
	}
	v0 = v0.Max(k.lo).Min(k.hi)
	v1 = v1.Max(k.lo).Min(k.hi)
	v2 = v2.Max(k.lo).Min(k.hi)
	v3 = v3.Max(k.lo).Min(k.hi)
	// In-register 4x4 int32 transpose: vN holds column N rows 0..3; tN holds
	// row N columns 0..3.
	e0 := v0.InterleaveLo(v1)
	e1 := v0.InterleaveHi(v1)
	e2 := v2.InterleaveLo(v3)
	e3 := v2.InterleaveHi(v3)
	t0 = fdctInt64AsInt32(fdctInt32AsInt64(e0).InterleaveLo(fdctInt32AsInt64(e2)))
	t1 = fdctInt64AsInt32(fdctInt32AsInt64(e0).InterleaveHi(fdctInt32AsInt64(e2)))
	t2 = fdctInt64AsInt32(fdctInt32AsInt64(e1).InterleaveLo(fdctInt32AsInt64(e3)))
	t3 = fdctInt64AsInt32(fdctInt32AsInt64(e1).InterleaveHi(fdctInt32AsInt64(e3)))
	return
}

// init binds the NEON-width Go SIMD inverse-transform glue kernels. They are
// bit-exact with clampRoundPureGo and narrowStorePureGo (see hybrid.go); the
// shared differential tests in invglue_gosimd_test.go pin that equivalence.
func init() {
	clampRoundImpl = clampRoundSIMD
	narrowStoreImpl = narrowStoreSIMD
}

// invGlueNegShift[s] holds -s in every lane: the variable-shift count that
// shifts right by s. Loading it from a table costs one vector load, where a
// broadcast per call is a lane insert plus a dup.
var invGlueNegShift = func() (t [32][4]int32) {
	for s := range t {
		for l := range t[s] {
			t[s][l] = -int32(s)
		}
	}
	return t
}()

// invGlueRoundVec returns roundShift(x, s) for each lane of v, for s >= 1.
// With y = x>>(s-1) (negS1 = -(s-1) broadcast), roundShift(x, s) == (y+1)>>1:
// the rounded halving add of y and zero (SRHADD) computes that without the
// overflow a plain y+1 would risk at MaxInt32.
func invGlueRoundVec(v, negS1, zero archsimd.Int32x4) archsimd.Int32x4 {
	return v.Shift(negS1).Average(zero)
}

// clampRoundSIMD rounds-and-shifts scratch in place (shift > 0) and clamps it to
// [min, max], eight int32 lanes per iteration. Tails and out-of-range shifts
// use the pure-Go reference.
func clampRoundSIMD(scratch []int32, shift int, min, max int32) {
	if shift < 0 || shift > 31 {
		clampRoundPureGo(scratch, shift, min, max)
		return
	}
	minV := archsimd.BroadcastInt32x4(min)
	maxV := archsimd.BroadcastInt32x4(max)
	n := len(scratch) &^ 7
	if shift == 0 {
		for i := 0; i < n; i += 8 {
			blk := scratch[i : i+8 : i+8]
			a := archsimd.LoadInt32x4Array((*[4]int32)(blk[0:4])).Max(minV).Min(maxV)
			b := archsimd.LoadInt32x4Array((*[4]int32)(blk[4:8])).Max(minV).Min(maxV)
			a.StoreArray((*[4]int32)(blk[0:4]))
			b.StoreArray((*[4]int32)(blk[4:8]))
		}
	} else {
		negS1 := archsimd.LoadInt32x4Array(&invGlueNegShift[shift-1])
		var zero archsimd.Int32x4
		for i := 0; i < n; i += 8 {
			blk := scratch[i : i+8 : i+8]
			a := invGlueRoundVec(archsimd.LoadInt32x4Array((*[4]int32)(blk[0:4])), negS1, zero)
			b := invGlueRoundVec(archsimd.LoadInt32x4Array((*[4]int32)(blk[4:8])), negS1, zero)
			a.Max(minV).Min(maxV).StoreArray((*[4]int32)(blk[0:4]))
			b.Max(minV).Min(maxV).StoreArray((*[4]int32)(blk[4:8]))
		}
	}
	if n < len(scratch) {
		clampRoundPureGo(scratch[n:], shift, min, max)
	}
}

// narrowStoreSIMD writes the final residual rows: every int32 lane is
// rounding-shifted right by four and saturated to int16. Width is four or a
// multiple of eight; other widths use the pure-Go reference.
func narrowStoreSIMD(dst []int16, dstStride int, scratch []int32, width, height int) {
	if width != 4 && width%8 != 0 {
		narrowStorePureGo(dst, dstStride, scratch, width, height)
		return
	}
	neg3 := archsimd.LoadInt32x4Array(&invGlueNegShift[3])
	var zero archsimd.Int32x4
	for row := range height {
		dstLine := dst[row*dstStride : row*dstStride+width : row*dstStride+width]
		srcLine := scratch[row*width : row*width+width : row*width+width]
		if width == 4 {
			v := archsimd.LoadInt32x4Array((*[4]int32)(srcLine))
			var out [8]int16
			narrowRound4ToInt16x8(v, v, neg3, zero).StoreArray(&out)
			copy(dstLine, out[:4])
			continue
		}
		for col := 0; col < width; col += 8 {
			lo := archsimd.LoadInt32x4Array((*[4]int32)(srcLine[col:]))
			hi := archsimd.LoadInt32x4Array((*[4]int32)(srcLine[col+4:]))
			narrowRound4ToInt16x8(lo, hi, neg3, zero).StoreArray((*[8]int16)(dstLine[col:]))
		}
	}
}

// narrowRound4ToInt16x8 rounding-shifts two int32x4 halves right by four and
// saturates them to one int16x8 in lane order (lo lanes 0-3, hi lanes 4-7). It
// is the hoisted-count form of roundShiftNarrowInt32x4ToInt16x8.
func narrowRound4ToInt16x8(lo, hi, neg3, zero archsimd.Int32x4) archsimd.Int16x8 {
	lo16 := invGlueRoundVec(lo, neg3, zero).SaturateToInt16()
	hi16 := invGlueRoundVec(hi, neg3, zero).SaturateToInt16()
	return lo16.ToBits().ReshapeToUint64s().InterleaveLo(hi16.ToBits().ReshapeToUint64s()).ReshapeToUint16s().BitsToInt16()
}

// int16 8-wide inverse DCT column kernels in dav1d's 8bpc form, on the stock
// archsimd API:
//
//   - every rotation Round2(a*ca + b*cb, n) widens both halves with
//     SMULL/SMULL2 (the compiler fuses x.HiToLo().MulWidenLo(k.HiToLo()) into
//     SMULL2), sums in int32, then rounds and narrows in two ops per four
//     lanes: a 32-bit MLA computes 2^(16-n)*p + 2^15, and UZP2 (ConcatOdd on
//     the 16-bit view) keeps the high halves, which are floor((p + 2^(n-1)) /
//     2^n) modulo 2^16, so the narrow is exact whenever the true rounded value
//     fits int16;
//   - the cos(pi/4) stages use dav1d's 2896 form, Round2(2896*a - 2896*b, 12)
//     == Round2((a-b)*181, 8), so no intermediate difference is formed in
//     int16;
//   - the butterfly sums are SQADD/SQSUB, which on the 8-bit path equal the
//     scalar clip to [-32768, 32767].
//
// A kernel is therefore byte-identical to the scalar inverseDCTn[int16] for
// every input whose rotation outputs all fit int16 (itx16_gain_test.go bounds
// those by the per-column L1 norm of the inputs). Callers guard with
// itx16ColumnInputSafe and fall back to the int32 path otherwise.

// itx16K holds the broadcast int16 rotation constants.
var itx16K = [...][8]int16{
	{2896, 2896, 2896, 2896, 2896, 2896, 2896, 2896},
	{-2896, -2896, -2896, -2896, -2896, -2896, -2896, -2896},
	{1567, 1567, 1567, 1567, 1567, 1567, 1567, 1567},
	{3784, 3784, 3784, 3784, 3784, 3784, 3784, 3784},
	{-3784, -3784, -3784, -3784, -3784, -3784, -3784, -3784},
	{799, 799, 799, 799, 799, 799, 799, 799},
	{4017, 4017, 4017, 4017, 4017, 4017, 4017, 4017},
	{-4017, -4017, -4017, -4017, -4017, -4017, -4017, -4017},
	{1703, 1703, 1703, 1703, 1703, 1703, 1703, 1703},
	{1138, 1138, 1138, 1138, 1138, 1138, 1138, 1138},
	{-1138, -1138, -1138, -1138, -1138, -1138, -1138, -1138},
	{401, 401, 401, 401, 401, 401, 401, 401},
	{4076, 4076, 4076, 4076, 4076, 4076, 4076, 4076},
	{-4076, -4076, -4076, -4076, -4076, -4076, -4076, -4076},
	{1583, 1583, 1583, 1583, 1583, 1583, 1583, 1583},
	{1299, 1299, 1299, 1299, 1299, 1299, 1299, 1299},
	{-1299, -1299, -1299, -1299, -1299, -1299, -1299, -1299},
	{1931, 1931, 1931, 1931, 1931, 1931, 1931, 1931},
	{3612, 3612, 3612, 3612, 3612, 3612, 3612, 3612},
	{-3612, -3612, -3612, -3612, -3612, -3612, -3612, -3612},
	{3920, 3920, 3920, 3920, 3920, 3920, 3920, 3920},
	{1189, 1189, 1189, 1189, 1189, 1189, 1189, 1189},
	{-1189, -1189, -1189, -1189, -1189, -1189, -1189, -1189},
	{-1567, -1567, -1567, -1567, -1567, -1567, -1567, -1567},
}

const (
	itx16Kp2896 = iota
	itx16Km2896
	itx16Kp1567
	itx16Kp3784
	itx16Km3784
	itx16Kp799
	itx16Kp4017
	itx16Km4017
	itx16Kp1703
	itx16Kp1138
	itx16Km1138
	itx16Kp401
	itx16Kp4076
	itx16Km4076
	itx16Kp1583
	itx16Kp1299
	itx16Km1299
	itx16Kp1931
	itx16Kp3612
	itx16Km3612
	itx16Kp3920
	itx16Kp1189
	itx16Km1189
	itx16Km1567
)

// itx16Scale[n] is the MLA multiplier 2^(16-n) for Round2(., n); the bias is
// always 2^15.
var itx16Scale12 = [4]int32{16, 16, 16, 16}
var itx16Scale11 = [4]int32{32, 32, 32, 32}
var itx16Bias = [4]int32{32768, 32768, 32768, 32768}

// itx16Rot is Round2(a*ca + b*cb, 12) on eight lanes.
func itx16Rot(a, b, ca, cb archsimd.Int16x8, scale archsimd.Int32x4) archsimd.Int16x8 {
	bias := archsimd.LoadInt32x4Array(&itx16Bias)
	lo := a.MulWidenLo(ca).Add(b.MulWidenLo(cb)).MulAdd(scale, bias)
	bias = archsimd.LoadInt32x4Array(&itx16Bias)
	hi := a.HiToLo().MulWidenLo(ca.HiToLo()).Add(b.HiToLo().MulWidenLo(cb.HiToLo())).MulAdd(scale, bias)
	return lo.ToBits().ReshapeToUint16s().ConcatOdd(hi.ToBits().ReshapeToUint16s()).BitsToInt16()
}

// inverseDCT16Col8SIMD16v2 applies inverseDCT16 to the eight int16 columns
// buf[k*stride+0..7]. Exact under the itx16 column guard; the clamp range is
// the int16 range (the 8-bit column bounds), which the saturating adds apply,
// so min and max are only accepted for the inverseDCT16Col8Impl16 signature.
func inverseDCT16Col8SIMD16v2(buf []int16, stride int, min int32, max int32) {
	_ = buf[15*stride+7]
	ld := func(k int) archsimd.Int16x8 { return archsimd.LoadInt16x8Array((*[8]int16)(buf[k*stride:])) }
	k := func(i int) archsimd.Int16x8 { return archsimd.LoadInt16x8Array(&itx16K[i]) }
	s12 := archsimd.LoadInt32x4Array(&itx16Scale12)
	s11 := archsimd.LoadInt32x4Array(&itx16Scale11)
	p2896, m2896 := k(itx16Kp2896), k(itx16Km2896)
	p1567, p3784, m3784 := k(itx16Kp1567), k(itx16Kp3784), k(itx16Km3784)

	// Even half: inverseDCT8 on rows 0,2,...,14 (inverseDCT4 on 0,4,8,12).
	in0, in4, in8, in12 := ld(0), ld(4), ld(8), ld(12)
	t0 := itx16Rot(in0, in8, p2896, p2896, s12)
	t1 := itx16Rot(in0, in8, p2896, m2896, s12)
	t2 := itx16Rot(in4, in12, p1567, m3784, s12)
	t3 := itx16Rot(in4, in12, p3784, p1567, s12)
	d0 := t0.AddSaturated(t3)
	d1 := t1.AddSaturated(t2)
	d2 := t1.SubSaturated(t2)
	d3 := t0.SubSaturated(t3)

	in2, in6, in10, in14 := ld(2), ld(6), ld(10), ld(14)
	t4a := itx16Rot(in2, in14, k(itx16Kp799), k(itx16Km4017), s12)
	t5a := itx16Rot(in10, in6, k(itx16Kp1703), k(itx16Km1138), s11)
	t6a := itx16Rot(in10, in6, k(itx16Kp1138), k(itx16Kp1703), s11)
	t7a := itx16Rot(in2, in14, k(itx16Kp4017), k(itx16Kp799), s12)
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

	// Odd half.
	in1, in15 := ld(1), ld(15)
	t8a := itx16Rot(in1, in15, k(itx16Kp401), k(itx16Km4076), s12)
	t15a := itx16Rot(in1, in15, k(itx16Kp4076), k(itx16Kp401), s12)
	in9, in7 := ld(9), ld(7)
	t9a := itx16Rot(in9, in7, k(itx16Kp1583), k(itx16Km1299), s11)
	t14a := itx16Rot(in9, in7, k(itx16Kp1299), k(itx16Kp1583), s11)
	in5, in11 := ld(5), ld(11)
	t10a := itx16Rot(in5, in11, k(itx16Kp1931), k(itx16Km3612), s12)
	t13a := itx16Rot(in5, in11, k(itx16Kp3612), k(itx16Kp1931), s12)
	in13, in3 := ld(13), ld(3)
	t11a := itx16Rot(in13, in3, k(itx16Kp3920), k(itx16Km1189), s12)
	t12a := itx16Rot(in13, in3, k(itx16Kp1189), k(itx16Kp3920), s12)

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

	st := func(k int, v archsimd.Int16x8) { v.StoreArray((*[8]int16)(buf[k*stride:])) }
	st(0, e0.AddSaturated(t15a))
	st(15, e0.SubSaturated(t15a))
	st(1, e1.AddSaturated(t14))
	st(14, e1.SubSaturated(t14))
	st(2, e2.AddSaturated(t13a))
	st(13, e2.SubSaturated(t13a))
	st(3, e3.AddSaturated(t12))
	st(12, e3.SubSaturated(t12))
	st(4, e4.AddSaturated(t11))
	st(11, e4.SubSaturated(t11))
	st(5, e5.AddSaturated(t10a))
	st(10, e5.SubSaturated(t10a))
	st(6, e6.AddSaturated(t9))
	st(9, e6.SubSaturated(t9))
	st(7, e7.AddSaturated(t8a))
	st(8, e7.SubSaturated(t8a))
}

// inverseDCT16Col8RawSIMD16 is inverseDCT16Col8SIMD16v2 over eight int32
// columns buf[k*stride+0..7] whose values are already in the int16 range: the
// loads narrow (SQXTN/SQXTN2), the per-column L1 guard accumulates alongside,
// and the stores widen back (SXTL/SXTL2). When a column fails the guard
// nothing is stored and it reports false, so the caller can run the int32
// kernels on the untouched columns.
func inverseDCT16Col8RawSIMD16(buf []int32, stride int) bool {
	_ = buf[15*stride+7]
	var l1 archsimd.Uint16x8
	ld := func(k int) archsimd.Int16x8 {
		lo := archsimd.LoadInt32x4Array((*[4]int32)(buf[k*stride:]))
		hi := archsimd.LoadInt32x4Array((*[4]int32)(buf[k*stride+4:]))
		v := lo.SaturateToInt16().ToBits().ReshapeToUint64s().BitsToFloat64()
		v = v.SetElem(1, hi.SaturateToInt16().ToBits().ReshapeToUint64s().BitsToFloat64().GetElem(0))
		x := v.ToBits().ReshapeToUint16s().BitsToInt16()
		l1 = l1.AddSaturated(x.Abs().ToBits())
		return x
	}
	k := func(i int) archsimd.Int16x8 { return archsimd.LoadInt16x8Array(&itx16K[i]) }
	s12 := archsimd.LoadInt32x4Array(&itx16Scale12)
	s11 := archsimd.LoadInt32x4Array(&itx16Scale11)
	p2896, m2896 := k(itx16Kp2896), k(itx16Km2896)
	p1567, p3784, m3784 := k(itx16Kp1567), k(itx16Kp3784), k(itx16Km3784)

	in0, in4, in8, in12 := ld(0), ld(4), ld(8), ld(12)
	t0 := itx16Rot(in0, in8, p2896, p2896, s12)
	t1 := itx16Rot(in0, in8, p2896, m2896, s12)
	t2 := itx16Rot(in4, in12, p1567, m3784, s12)
	t3 := itx16Rot(in4, in12, p3784, p1567, s12)
	d0 := t0.AddSaturated(t3)
	d1 := t1.AddSaturated(t2)
	d2 := t1.SubSaturated(t2)
	d3 := t0.SubSaturated(t3)

	in2, in6, in10, in14 := ld(2), ld(6), ld(10), ld(14)
	t4a := itx16Rot(in2, in14, k(itx16Kp799), k(itx16Km4017), s12)
	t5a := itx16Rot(in10, in6, k(itx16Kp1703), k(itx16Km1138), s11)
	t6a := itx16Rot(in10, in6, k(itx16Kp1138), k(itx16Kp1703), s11)
	t7a := itx16Rot(in2, in14, k(itx16Kp4017), k(itx16Kp799), s12)
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

	in1, in15 := ld(1), ld(15)
	t8a := itx16Rot(in1, in15, k(itx16Kp401), k(itx16Km4076), s12)
	t15a := itx16Rot(in1, in15, k(itx16Kp4076), k(itx16Kp401), s12)
	in9, in7 := ld(9), ld(7)
	t9a := itx16Rot(in9, in7, k(itx16Kp1583), k(itx16Km1299), s11)
	t14a := itx16Rot(in9, in7, k(itx16Kp1299), k(itx16Kp1583), s11)
	in5, in11 := ld(5), ld(11)
	t10a := itx16Rot(in5, in11, k(itx16Kp1931), k(itx16Km3612), s12)
	t13a := itx16Rot(in5, in11, k(itx16Kp3612), k(itx16Kp1931), s12)
	in13, in3 := ld(13), ld(3)
	t11a := itx16Rot(in13, in3, k(itx16Kp3920), k(itx16Km1189), s12)
	t12a := itx16Rot(in13, in3, k(itx16Kp1189), k(itx16Kp3920), s12)

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
	st := func(k int, v archsimd.Int16x8) {
		v.HiToLo().ExtendLo4ToInt32().StoreArray((*[4]int32)(buf[k*stride+4:]))
		v.ExtendLo4ToInt32().StoreArray((*[4]int32)(buf[k*stride:]))
	}
	st(0, e0.AddSaturated(t15a))
	st(15, e0.SubSaturated(t15a))
	st(1, e1.AddSaturated(t14))
	st(14, e1.SubSaturated(t14))
	st(2, e2.AddSaturated(t13a))
	st(13, e2.SubSaturated(t13a))
	st(3, e3.AddSaturated(t12))
	st(12, e3.SubSaturated(t12))
	st(4, e4.AddSaturated(t11))
	st(11, e4.SubSaturated(t11))
	st(5, e5.AddSaturated(t10a))
	st(10, e5.SubSaturated(t10a))
	st(6, e6.AddSaturated(t9))
	st(9, e6.SubSaturated(t9))
	st(7, e7.AddSaturated(t8a))
	st(8, e7.SubSaturated(t8a))
	return true
}

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

func init() {
	itx16RawColumnPass = itx16RawColumnPassSIMD
	itx16RawColumnPass32Low8 = itx16RawColumnPass32Low8SIMD
	inverseDCT16Col8Impl16 = inverseDCT16Col8SIMD16v2
}

// itx16ColumnL1Bound is the per-column guard of the int16 DCT16 column kernel:
// sum_j |x_j| <= bound certifies that every rotation output fits int16, since
// the largest input coefficient of any rotation output is 0.9951 and the
// accumulated rounding slack is at most 4.03 (itx16_gain_test.go), with
// margin.
const itx16ColumnL1Bound = 32767 - 64

// itx16RawColumnPassSIMD runs the DCT16 column pass over the int32 raw
// scratch (width x 16, already rounded and clamped to the int16 range) eight
// columns at a time with the int16 kernel; a column group that fails the
// kernel's guard is untouched and takes the int32 kernels (the low-row ones
// when only rowLimit input rows are nonzero).
func itx16RawColumnPassSIMD(scratch []int32, width, rowLimit int, min, max int32) {
	for c := 0; c < width; c += 8 {
		var ok bool
		switch {
		case rowLimit > 0 && rowLimit <= 4:
			ok = inverseDCT16Col8RawLow4SIMD16(scratch[c:], width)
		case rowLimit > 0 && rowLimit <= 8:
			ok = inverseDCT16Col8RawLow8SIMD16(scratch[c:], width)
		default:
			ok = inverseDCT16Col8RawSIMD16(scratch[c:], width)
		}
		if ok {
			continue
		}
		if rowLimit > 0 && rowLimit <= 8 && inverseDCT16LowCol4Impl != nil {
			lowN := 8
			if rowLimit <= 4 {
				lowN = 4
			}
			inverseDCT16LowCol4Impl(scratch[c:], width, lowN, min, max)
			inverseDCT16LowCol4Impl(scratch[c+4:], width, lowN, min, max)
			continue
		}
		inverse1DCol4(scratch[c:], width, dct16Size, tx1DDCT, min, max)
		inverse1DCol4(scratch[c+4:], width, dct16Size, tx1DDCT, min, max)
	}
}

// colClampBoundNEON is the stage-range envelope the four-column int32-lane
// kernels are proven overflow-free for: every supported bit depth's row and
// column stage bounds satisfy |bound| <= 1<<19 (stageRangeBounds caps at
// bitDepth 12: rowBits 20). Wider bounds fall back to pure Go.
const colClampBoundNEON = 1 << 19

// init binds the Go SIMD batched inverse kernels. The column kernels run four
// (or two) columns in int32 lanes and the row kernels run rows through a 4x4
// transpose; the adapters gate on the clamp envelope the int32 lanes are exact
// for and fall back to the pure-Go reference outside it, so output bytes do not
// depend on the path taken.
func init() {
	inverseDCT4Col4Impl = inverseDCT4Col4SIMDAdapter
	inverseDCT8Col2Impl = inverseDCT8Col2SIMDAdapter
	inverseDCT8Col4Impl = inverseDCT8Col4SIMDAdapter
	inverseDCT16Col2Impl = inverseDCT16Col2SIMDAdapter
	inverseDCT16Col4Impl = inverseDCT16Col4SIMDAdapter
	inverseADST16Col4Impl = inverseADST16Col4SIMDAdapter
	inverseADST16Col4FlipImpl = inverseADST16Col4FlipSIMDAdapter
	inverseDCT8Row2Impl = inverseDCT8Row2SIMDAdapter
	inverseDCT16Row2Impl = inverseDCT16Row2SIMDAdapter
	inverseADST16Row4Impl = inverseADST16Row4SIMDAdapter
	inverseADST16Row4FlipImpl = inverseADST16Row4FlipSIMDAdapter
	inverseDCT32Col2Impl = inverseDCT32Col2SIMDAdapter
	inverseDCT32Col4Impl = inverseDCT32Col4SIMDAdapter
	inverseDCT32Low8Col4Impl = inverseDCT32Low8Col4SIMDAdapter
	inverseDCT32Row2Impl = inverseDCT32Row2SIMDAdapter
	inverseDCT32Row4Impl = inverseDCT32Row4SIMDAdapter
	inverseDCT32Low16Col4Impl = inverseDCT32Low16Col4SIMDAdapter
	inverseDCT32Row4LowImpl = inverseDCT32Row4LowSIMDAdapter
	inverseDCT32Row2LowImpl = inverseDCT32Row2LowSIMDAdapter
	inverseDCT16Row2LowImpl = inverseDCT16Row2LowSIMDAdapter
	inverseDCT16LowCol4Impl = inverseDCT16LowCol4SIMDAdapter
}

// simdColEnvelope reports whether the [min, max] stage clamp lies inside the
// +/-2^19 envelope the int32-lane kernels are exact for. Column kernels rely on
// the staged pre-clamp; row kernels read
// rows that inverseSeparableBlockClampedRowsToScratch staged through
// stageTransposeClamp into the same [min, max] range, so checking the bounds
// covers every input without scanning the rows.
func simdColEnvelope(min, max int32) bool {
	return min >= -colClampBoundNEON && max < colClampBoundNEON
}

func inverseDCT4Col4SIMDAdapter(buf []int32, rowStride int, min, max int32) {
	if rowStride < 4 || len(buf) < (dct4Size-1)*rowStride+4 || !simdColEnvelope(min, max) {
		inverseDCT4Col4PureGo(buf, rowStride, min, max)
		return
	}
	inverseDCT4Col4SIMD(buf, rowStride, min, max)
}

func inverseDCT8Col2SIMDAdapter(buf []int32, rowStride int, min, max int32) {
	if rowStride < 2 || len(buf) < (dct8Size-1)*rowStride+2 || !simdColEnvelope(min, max) {
		inverseDCT8Col2PureGo(buf, rowStride, min, max)
		return
	}
	inverseDCT8Col2SIMD(buf, rowStride, min, max)
}

func inverseDCT16Col2SIMDAdapter(buf []int32, rowStride int, min, max int32) {
	if rowStride < 2 || len(buf) < (dct16Size-1)*rowStride+2 || !simdColEnvelope(min, max) {
		inverseDCT16Col2PureGo(buf, rowStride, min, max)
		return
	}
	inverseDCT16Col2SIMD(buf, rowStride, min, max)
}

func inverseDCT8Col4SIMDAdapter(buf []int32, rowStride int, min, max int32) {
	if rowStride < 4 || len(buf) < (dct8Size-1)*rowStride+4 || !simdColEnvelope(min, max) {
		inverseDCT8Col2SIMDAdapter(buf, rowStride, min, max)
		inverseDCT8Col2SIMDAdapter(buf[2:], rowStride, min, max)
		return
	}
	inverseDCT8Col4SIMD(buf, rowStride, min, max)
}

func inverseDCT16Col4SIMDAdapter(buf []int32, rowStride int, min, max int32) {
	if rowStride < 4 || len(buf) < (dct16Size-1)*rowStride+4 || !simdColEnvelope(min, max) {
		inverseDCT16Col2SIMDAdapter(buf, rowStride, min, max)
		inverseDCT16Col2SIMDAdapter(buf[2:], rowStride, min, max)
		return
	}
	inverseDCT16Col4SIMD(buf, rowStride, min, max)
}

func inverseADST16Col4SIMDAdapter(buf []int32, rowStride int, min, max int32) {
	if rowStride < 4 || len(buf) < (adst16Size-1)*rowStride+4 || !simdColEnvelope(min, max) {
		inverseADST16Col4PureGo(buf, rowStride, min, max)
		return
	}
	inverseADST16Col4SIMD(buf, rowStride, min, max)
}

func inverseADST16Col4FlipSIMDAdapter(buf []int32, rowStride int, min, max int32) {
	if rowStride < 4 || len(buf) < (adst16Size-1)*rowStride+4 || !simdColEnvelope(min, max) {
		inverseADST16Col4FlipPureGo(buf, rowStride, min, max)
		return
	}
	inverseADST16Col4FlipSIMD(buf, rowStride, min, max)
}

func inverseDCT8Row2SIMDAdapter(r0, r1 []int32, min, max int32) {
	if len(r0) < dct8Size || len(r1) < dct8Size || !simdColEnvelope(min, max) {
		inverseDCT8Row2PureGo(r0, r1, min, max)
		return
	}
	inverseDCT8Row2SIMD(r0[:dct8Size], r1[:dct8Size], min, max)
}

func inverseDCT16Row2SIMDAdapter(r0, r1 []int32, min, max int32) {
	if len(r0) < dct16Size || len(r1) < dct16Size || !simdColEnvelope(min, max) {
		inverseDCT16Row2PureGo(r0, r1, min, max)
		return
	}
	inverseDCT16Row2SIMD(r0[:dct16Size], r1[:dct16Size], min, max)
}

func inverseADST16Row4SIMDAdapter(r0, r1, r2, r3 []int32, min, max int32) {
	if len(r0) < adst16Size || len(r1) < adst16Size || len(r2) < adst16Size || len(r3) < adst16Size ||
		!simdColEnvelope(min, max) {
		inverseADST16Row4PureGo(r0, r1, r2, r3, min, max)
		return
	}
	inverseADST16Row4SIMD(r0[:adst16Size], r1[:adst16Size], r2[:adst16Size], r3[:adst16Size], min, max)
}

func inverseADST16Row4FlipSIMDAdapter(r0, r1, r2, r3 []int32, min, max int32) {
	if len(r0) < adst16Size || len(r1) < adst16Size || len(r2) < adst16Size || len(r3) < adst16Size ||
		!simdColEnvelope(min, max) {
		inverseADST16Row4FlipPureGo(r0, r1, r2, r3, min, max)
		return
	}
	inverseADST16Row4FlipSIMD(r0[:adst16Size], r1[:adst16Size], r2[:adst16Size], r3[:adst16Size], min, max)
}

func inverseDCT32Col2SIMDAdapter(buf []int32, rowStride int, min, max int32) {
	if rowStride < 2 || len(buf) < (dct32Size-1)*rowStride+2 || !simdColEnvelope(min, max) {
		inverseDCT32Col2PureGo(buf, rowStride, min, max)
		return
	}
	inverseDCT32Col2SIMD(buf, rowStride, min, max)
}

func inverseDCT32Col4SIMDAdapter(buf []int32, rowStride int, min, max int32) {
	if rowStride < 4 || len(buf) < (dct32Size-1)*rowStride+4 || !simdColEnvelope(min, max) {
		inverseDCT32Col2SIMDAdapter(buf, rowStride, min, max)
		inverseDCT32Col2SIMDAdapter(buf[2:], rowStride, min, max)
		return
	}
	inverseDCT32Col4SIMD(buf, rowStride, min, max)
}

func inverseDCT32Low8Col4SIMDAdapter(buf []int32, rowStride int, min, max int32) {
	// Replacing omitted input rows with zero also requires the column clamp to
	// preserve zero; use the full kernel for intervals wholly above or below it.
	if rowStride < 4 || len(buf) < (dct32Size-1)*rowStride+4 || !simdColEnvelope(min, max) || min > 0 || max < 0 {
		inverseDCT32Col4SIMDAdapter(buf, rowStride, min, max)
		return
	}
	inverseDCT32Col4Low8SIMD(buf, rowStride, min, max)
}

func inverseDCT32Row2SIMDAdapter(r0, r1 []int32, min, max int32) {
	if len(r0) < dct32Size || len(r1) < dct32Size || !simdColEnvelope(min, max) {
		inverseDCT32Row2PureGo(r0, r1, min, max)
		return
	}
	inverseDCT32Row2SIMD(r0[:dct32Size], r1[:dct32Size], min, max)
}

func inverseDCT32Row4SIMDAdapter(r0, r1, r2, r3 []int32, min, max int32) {
	if len(r0) < dct32Size || len(r1) < dct32Size || len(r2) < dct32Size || len(r3) < dct32Size ||
		!simdColEnvelope(min, max) {
		inverseDCT32Row4PureGo(r0, r1, r2, r3, min, max)
		return
	}
	inverseDCT32Row4SIMD(r0[:dct32Size], r1[:dct32Size], r2[:dct32Size], r3[:dct32Size], min, max)
}

func inverseDCT32Low16Col4SIMDAdapter(buf []int32, rowStride int, min, max int32) {
	if rowStride < 4 || len(buf) < (dct32Size-1)*rowStride+4 || !simdColEnvelope(min, max) || min > 0 || max < 0 {
		inverseDCT32Col4SIMDAdapter(buf, rowStride, min, max)
		return
	}
	inverseDCT32Col4Low16SIMD(buf, rowStride, min, max)
}

func inverseDCT32Row4LowSIMDAdapter(r0, r1, r2, r3 []int32, lowN int, min, max int32) {
	if len(r0) < dct32Size || len(r1) < dct32Size || len(r2) < dct32Size || len(r3) < dct32Size ||
		!simdColEnvelope(min, max) || min > 0 || max < 0 {
		inverseDCT32Row4PureGo(r0, r1, r2, r3, min, max)
		return
	}
	if lowN <= 8 {
		inverseDCT32Row4Low8SIMD(r0[:dct32Size], r1[:dct32Size], r2[:dct32Size], r3[:dct32Size], min, max)
		return
	}
	inverseDCT32Row4Low16SIMD(r0[:dct32Size], r1[:dct32Size], r2[:dct32Size], r3[:dct32Size], min, max)
}

func inverseDCT32Row2LowSIMDAdapter(r0, r1 []int32, lowN int, min, max int32) {
	if len(r0) < dct32Size || len(r1) < dct32Size || !simdColEnvelope(min, max) || min > 0 || max < 0 {
		inverseDCT32Row2PureGo(r0, r1, min, max)
		return
	}
	if lowN <= 8 {
		inverseDCT32Row2Low8SIMD(r0[:dct32Size], r1[:dct32Size], min, max)
		return
	}
	inverseDCT32Row2Low16SIMD(r0[:dct32Size], r1[:dct32Size], min, max)
}

func inverseDCT16Row2LowSIMDAdapter(r0, r1 []int32, lowN int, min, max int32) {
	if len(r0) < dct16Size || len(r1) < dct16Size || !simdColEnvelope(min, max) || min > 0 || max < 0 {
		inverseDCT16Row2SIMDAdapter(r0, r1, min, max)
		return
	}
	if lowN <= 4 {
		inverseDCT16Row2Low4SIMD(r0[:dct16Size], r1[:dct16Size], min, max)
		return
	}
	inverseDCT16Row2Low8SIMD(r0[:dct16Size], r1[:dct16Size], min, max)
}

func inverseDCT16LowCol4SIMDAdapter(buf []int32, rowStride int, lowN int, min, max int32) {
	if rowStride < 4 || len(buf) < (dct16Size-1)*rowStride+4 || !simdColEnvelope(min, max) || min > 0 || max < 0 {
		inverseDCT16Col4SIMDAdapter(buf, rowStride, min, max)
		return
	}
	if lowN <= 4 {
		inverseDCT16Col4Low4SIMD(buf, rowStride, min, max)
		return
	}
	inverseDCT16Col4Low8SIMD(buf, rowStride, min, max)
}

// itx_fused16: dav1d's pass structure (itx_fused32) for the 8-bit 16x16
// DCT_DCT block. The row pass runs one or two eight-row strips straight
// from the column-major coefficients with the DCT16 network across vectors
// (its full form, or the Low8 / Low4 forms when the coded columns stop at
// 8 or 4), rounds by 2 in registers and transposes 8x8 into a 16x16 int16
// tmp; the two column strips run the network form that matches the coded
// rows and fuse Round2(., 4), the prediction add and the byte saturation.

// itx16NetFull is inverseDCT16 on eight lanes: in[k] is coefficient k,
// out[r] output r (the network of inverseDCT16Col8SIMD16v2).
func itx16NetFull(in *[16]archsimd.Int16x8, out *[16]archsimd.Int16x8) {
	k := func(i int) archsimd.Int16x8 { return archsimd.LoadInt16x8Array(&itx16K[i]) }
	s12 := archsimd.LoadInt32x4Array(&itx16Scale12)
	s11 := archsimd.LoadInt32x4Array(&itx16Scale11)
	p2896, m2896 := k(itx16Kp2896), k(itx16Km2896)
	p1567, p3784, m3784 := k(itx16Kp1567), k(itx16Kp3784), k(itx16Km3784)

	in0, in4, in8, in12 := in[0], in[4], in[8], in[12]
	t0 := itx16Rot(in0, in8, p2896, p2896, s12)
	t1 := itx16Rot(in0, in8, p2896, m2896, s12)
	t2 := itx16Rot(in4, in12, p1567, m3784, s12)
	t3 := itx16Rot(in4, in12, p3784, p1567, s12)
	d0 := t0.AddSaturated(t3)
	d1 := t1.AddSaturated(t2)
	d2 := t1.SubSaturated(t2)
	d3 := t0.SubSaturated(t3)

	in2, in6, in10, in14 := in[2], in[6], in[10], in[14]
	t4a := itx16Rot(in2, in14, k(itx16Kp799), k(itx16Km4017), s12)
	t5a := itx16Rot(in10, in6, k(itx16Kp1703), k(itx16Km1138), s11)
	t6a := itx16Rot(in10, in6, k(itx16Kp1138), k(itx16Kp1703), s11)
	t7a := itx16Rot(in2, in14, k(itx16Kp4017), k(itx16Kp799), s12)
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

	in1, in15 := in[1], in[15]
	t8a := itx16Rot(in1, in15, k(itx16Kp401), k(itx16Km4076), s12)
	t15a := itx16Rot(in1, in15, k(itx16Kp4076), k(itx16Kp401), s12)
	in9, in7 := in[9], in[7]
	t9a := itx16Rot(in9, in7, k(itx16Kp1583), k(itx16Km1299), s11)
	t14a := itx16Rot(in9, in7, k(itx16Kp1299), k(itx16Kp1583), s11)
	in5, in11 := in[5], in[11]
	t10a := itx16Rot(in5, in11, k(itx16Kp1931), k(itx16Km3612), s12)
	t13a := itx16Rot(in5, in11, k(itx16Kp3612), k(itx16Kp1931), s12)
	in13, in3 := in[13], in[3]
	t11a := itx16Rot(in13, in3, k(itx16Kp3920), k(itx16Km1189), s12)
	t12a := itx16Rot(in13, in3, k(itx16Kp1189), k(itx16Kp3920), s12)

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

	out[0], out[15] = e0.AddSaturated(t15a), e0.SubSaturated(t15a)
	out[1], out[14] = e1.AddSaturated(t14), e1.SubSaturated(t14)
	out[2], out[13] = e2.AddSaturated(t13a), e2.SubSaturated(t13a)
	out[3], out[12] = e3.AddSaturated(t12), e3.SubSaturated(t12)
	out[4], out[11] = e4.AddSaturated(t11), e4.SubSaturated(t11)
	out[5], out[10] = e5.AddSaturated(t10a), e5.SubSaturated(t10a)
	out[6], out[9] = e6.AddSaturated(t9), e6.SubSaturated(t9)
	out[7], out[8] = e7.AddSaturated(t8a), e7.SubSaturated(t8a)
}

// itx16NetLow8 is itx16NetFull for inputs 8..15 zero (the network of
// inverseDCT16Col8RawLow8SIMD16).
func itx16NetLow8(in *[16]archsimd.Int16x8, out *[16]archsimd.Int16x8) {
	k := func(i int) archsimd.Int16x8 { return archsimd.LoadInt16x8Array(&itx16K[i]) }
	s12 := archsimd.LoadInt32x4Array(&itx16Scale12)
	s11 := archsimd.LoadInt32x4Array(&itx16Scale11)
	p2896, m2896 := k(itx16Kp2896), k(itx16Km2896)
	p1567, p3784, m3784 := k(itx16Kp1567), k(itx16Kp3784), k(itx16Km3784)
	in0, in1, in2, in3, in4, in5, in6, in7 := in[0], in[1], in[2], in[3], in[4], in[5], in[6], in[7]

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

	out[0], out[15] = e0.AddSaturated(t15a), e0.SubSaturated(t15a)
	out[1], out[14] = e1.AddSaturated(t14), e1.SubSaturated(t14)
	out[2], out[13] = e2.AddSaturated(t13a), e2.SubSaturated(t13a)
	out[3], out[12] = e3.AddSaturated(t12), e3.SubSaturated(t12)
	out[4], out[11] = e4.AddSaturated(t11), e4.SubSaturated(t11)
	out[5], out[10] = e5.AddSaturated(t10a), e5.SubSaturated(t10a)
	out[6], out[9] = e6.AddSaturated(t9), e6.SubSaturated(t9)
	out[7], out[8] = e7.AddSaturated(t8a), e7.SubSaturated(t8a)
}

// itx16NetLow4 is itx16NetFull for inputs 4..15 zero (the network of
// inverseDCT16Col8RawLow4SIMD16).
func itx16NetLow4(in *[16]archsimd.Int16x8, out *[16]archsimd.Int16x8) {
	k := func(i int) archsimd.Int16x8 { return archsimd.LoadInt16x8Array(&itx16K[i]) }
	s12 := archsimd.LoadInt32x4Array(&itx16Scale12)
	p2896, m2896 := k(itx16Kp2896), k(itx16Km2896)
	p1567, p3784, m3784 := k(itx16Kp1567), k(itx16Kp3784), k(itx16Km3784)
	in0, in1, in2, in3 := in[0], in[1], in[2], in[3]

	t0 := itx16Rot1(in0, p2896, s12)
	t4a := itx16Rot1(in2, k(itx16Kp799), s12)
	t7a := itx16Rot1(in2, k(itx16Kp4017), s12)
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

	t8a := itx16Rot1(in1, k(itx16Kp401), s12)
	t15a := itx16Rot1(in1, k(itx16Kp4076), s12)
	t11a := itx16Rot1(in3, k(itx16Km1189), s12)
	t12a := itx16Rot1(in3, k(itx16Kp3920), s12)
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

	out[0], out[15] = e0.AddSaturated(t15), e0.SubSaturated(t15)
	out[1], out[14] = e1.AddSaturated(t14), e1.SubSaturated(t14)
	out[2], out[13] = e2.AddSaturated(t13a), e2.SubSaturated(t13a)
	out[3], out[12] = e3.AddSaturated(t12a), e3.SubSaturated(t12a)
	out[4], out[11] = e4.AddSaturated(t11a), e4.SubSaturated(t11a)
	out[5], out[10] = e5.AddSaturated(t10a), e5.SubSaturated(t10a)
	out[6], out[9] = e6.AddSaturated(t9), e6.SubSaturated(t9)
	out[7], out[8] = e7.AddSaturated(t8), e7.SubSaturated(t8)
}

// itx16Net runs the DCT16 network form for n nonzero inputs (4, 8 or 16).
func itx16Net(n int, in *[16]archsimd.Int16x8, out *[16]archsimd.Int16x8) {
	switch {
	case n <= 4:
		itx16NetLow4(in, out)
	case n <= 8:
		itx16NetLow8(in, out)
	default:
		itx16NetFull(in, out)
	}
}

// itx16RowStrip runs the 16x16 row pass for rows r0..r0+7 with inCols
// coded coefficient columns (4, 8 or 16) into tmp rows r0..r0+7 and adds
// the stored values' magnitudes to the two column guards. It reports false,
// with tmp untouched, when a row fails the L1 guard.
func itx16RowStrip(coeff []int32, stride int, r0 int, inCols int, tmp *[16 * 16]int16, colL1 *[2]archsimd.Uint16x8) bool {
	_ = coeff[(inCols-1)*stride+r0+7]
	var in [16]archsimd.Int16x8
	var l1 archsimd.Uint16x8
	for k := 0; k < inCols; k++ {
		x := itx16LoadNarrow(coeff[k*stride+r0 : k*stride+r0+8])
		l1 = l1.AddSaturated(x.Abs().ToBits())
		in[k] = x
	}
	if l1.ReduceMax() > itx16ColumnL1Bound {
		return false
	}
	var out [16]archsimd.Int16x8
	itx16Net(inCols, &in, &out)
	one := archsimd.BroadcastInt16x8(1)
	sh1 := archsimd.BroadcastInt16x8(-1)
	for g := 0; g < 2; g++ {
		var v [8]archsimd.Int16x8
		for j := 0; j < 8; j++ {
			v[j] = itx32Round2(out[8*g+j], sh1, sh1, one)
		}
		itx16Transpose8x8(&v)
		gl1 := colL1[g]
		for r := 0; r < 8; r++ {
			v[r].StoreArray((*[8]int16)(tmp[(r0+r)*16+8*g:]))
			gl1 = gl1.AddSaturated(v[r].Abs().ToBits())
		}
		colL1[g] = gl1
	}
	return true
}

// itx16Form rounds a coded extent up to the network form that covers it.
func itx16Form(n int) int {
	switch {
	case n <= 4:
		return 4
	case n <= 8:
		return 8
	default:
		return 16
	}
}

// InverseAddDCT16x16 reconstructs one 8-bit 16x16 DCT_DCT block with coded
// coefficients in rows 0..activeRows-1 and columns 0..activeCols-1:
// dst = clip(pred + inverse(coeff)), with the dequantized coefficients
// column-major at coeffStride (16). pred may be dst. It reports false, with
// dst untouched, when the block is outside the kernel's contract or a guard
// declines; the caller then runs the staged path, which produces the same
// bytes. The kernels are generated (itx_fused_gen_arm64.go) per network
// form of the coded columns and rows.
func InverseAddDCT16x16(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, activeRows, activeCols int) bool {
	if activeRows <= 0 || activeRows > 16 || activeCols <= 0 || activeCols > 16 ||
		coeffStride != 16 || len(coeff) < 256 || dstStride < 16 || predStride < 16 ||
		len(dst) < 15*dstStride+16 || len(pred) < 15*predStride+16 {
		return false
	}
	// The kernel switch is written here rather than in a dispatch function:
	// one call level less.
	c := (*[256]int32)(coeff[:256])
	switch {
	case activeCols <= 4:
		switch {
		case activeRows <= 4:
			return itxGenDCT16x16C4R4(c, dst, dstStride, pred, predStride)
		case activeRows <= 8:
			return itxGenDCT16x16C4R8(c, dst, dstStride, pred, predStride)
		default:
			return itxGenDCT16x16C4R16(c, dst, dstStride, pred, predStride)
		}
	case activeCols <= 8:
		switch {
		case activeRows <= 4:
			return itxGenDCT16x16C8R4(c, dst, dstStride, pred, predStride)
		case activeRows <= 8:
			return itxGenDCT16x16C8R8(c, dst, dstStride, pred, predStride)
		default:
			return itxGenDCT16x16C8R16(c, dst, dstStride, pred, predStride)
		}
	default:
		switch {
		case activeRows <= 4:
			return itxGenDCT16x16C16R4(c, dst, dstStride, pred, predStride)
		case activeRows <= 8:
			return itxGenDCT16x16C16R8(c, dst, dstStride, pred, predStride)
		default:
			return itxGenDCT16x16C16R16(c, dst, dstStride, pred, predStride)
		}
	}
}

// itx_fused32_full: the fused 32x32 path (itx_fused32) for blocks whose
// coded coefficients reach past the 16x16 corner. The row pass runs up to
// four eight-row strips with the DCT32 network form matching the coded
// columns (Low8, Low16 or the full form below) into a 32x32 int16 tmp; the
// four column strips run the form matching the coded rows with the add
// fused.

// itx32FullNet is inverseDCT32 on eight lanes for all 32 inputs: in[k] is
// coefficient k, out[r] is output r. It is itx32Low16Net with the DCT16 even
// half in its full form and the odd first stage as rotation pairs.
func itx32FullNet(in *[32]archsimd.Int16x8, out *[32]archsimd.Int16x8) {
	k := func(i int) archsimd.Int16x8 { return archsimd.LoadInt16x8Array(&itx16K[i]) }
	k32 := func(i int) archsimd.Int16x8 { return archsimd.LoadInt16x8Array(&itx32K[i]) }
	c := func(i int) archsimd.Int16x8 { return archsimd.LoadInt16x8Array(&itx64K[i]) }
	cn := func(i int) archsimd.Int16x8 { return archsimd.LoadInt16x8Array(&itx64K[64+i]) }
	s12 := archsimd.LoadInt32x4Array(&itx16Scale12)
	s11 := archsimd.LoadInt32x4Array(&itx16Scale11)
	p2896, m2896 := k(itx16Kp2896), k(itx16Km2896)
	p1567, m1567, p3784, m3784 := k(itx16Kp1567), k(itx16Km1567), k(itx16Kp3784), k(itx16Km3784)

	// Even half: inverseDCT16 on inputs 0, 2, ..., 30.
	var e [16]archsimd.Int16x8
	for i := 0; i < 16; i++ {
		e[i] = in[2*i]
	}
	var E [16]archsimd.Int16x8
	itx16NetFull(&e, &E)

	// Odd half: rotation pairs (cospi index i is cospi(i) = itx64Cos[i]).
	in1, in3, in5, in7 := in[1], in[3], in[5], in[7]
	in9, in11, in13, in15 := in[9], in[11], in[13], in[15]
	in17, in19, in21, in23 := in[17], in[19], in[21], in[23]
	in25, in27, in29, in31 := in[25], in[27], in[29], in[31]
	t16a := itx16Rot(in1, in31, c(62), cn(2), s12)
	t31a := itx16Rot(in1, in31, c(2), c(62), s12)
	t17a := itx16Rot(in17, in15, c(30), cn(34), s12)
	t30a := itx16Rot(in17, in15, c(34), c(30), s12)
	t18a := itx16Rot(in9, in23, c(46), cn(18), s12)
	t29a := itx16Rot(in9, in23, c(18), c(46), s12)
	t19a := itx16Rot(in25, in7, c(14), cn(50), s12)
	t28a := itx16Rot(in25, in7, c(50), c(14), s12)
	t20a := itx16Rot(in5, in27, c(54), cn(10), s12)
	t27a := itx16Rot(in5, in27, c(10), c(54), s12)
	t21a := itx16Rot(in21, in11, c(22), cn(42), s12)
	t26a := itx16Rot(in21, in11, c(42), c(22), s12)
	t22a := itx16Rot(in13, in19, c(38), cn(26), s12)
	t25a := itx16Rot(in13, in19, c(26), c(38), s12)
	t23a := itx16Rot(in29, in3, c(6), cn(58), s12)
	t24a := itx16Rot(in29, in3, c(58), c(6), s12)

	t16 := t16a.AddSaturated(t17a)
	t17 := t16a.SubSaturated(t17a)
	t18 := t19a.SubSaturated(t18a)
	t19 := t19a.AddSaturated(t18a)
	t20 := t20a.AddSaturated(t21a)
	t21 := t20a.SubSaturated(t21a)
	t22 := t23a.SubSaturated(t22a)
	t23 := t23a.AddSaturated(t22a)
	t24 := t24a.AddSaturated(t25a)
	t25 := t24a.SubSaturated(t25a)
	t26 := t27a.SubSaturated(t26a)
	t27 := t27a.AddSaturated(t26a)
	t28 := t28a.AddSaturated(t29a)
	t29 := t28a.SubSaturated(t29a)
	t30 := t31a.SubSaturated(t30a)
	t31 := t31a.AddSaturated(t30a)

	p799, m799, p4017, m4017 := k(itx16Kp799), k32(itx32Km799), k(itx16Kp4017), k(itx16Km4017)
	t17a = itx16Rot(t30, t17, p799, m4017, s12)
	t30a = itx16Rot(t30, t17, p4017, p799, s12)
	t18a = itx16Rot(t29, t18, m4017, m799, s12)
	t29a = itx16Rot(t29, t18, p799, m4017, s12)
	p1703, m1703, p1138, m1138 := k(itx16Kp1703), k32(itx32Km1703), k(itx16Kp1138), k(itx16Km1138)
	t21a = itx16Rot(t26, t21, p1703, m1138, s11)
	t26a = itx16Rot(t26, t21, p1138, p1703, s11)
	t22a = itx16Rot(t25, t22, m1138, m1703, s11)
	t25a = itx16Rot(t25, t22, p1703, m1138, s11)

	t16a = t16.AddSaturated(t19)
	t17 = t17a.AddSaturated(t18a)
	t18 = t17a.SubSaturated(t18a)
	t19a = t16.SubSaturated(t19)
	t20a = t23.SubSaturated(t20)
	t21 = t22a.SubSaturated(t21a)
	t22 = t22a.AddSaturated(t21a)
	t23a = t23.AddSaturated(t20)
	t24a = t24.AddSaturated(t27)
	t25 = t25a.AddSaturated(t26a)
	t26 = t25a.SubSaturated(t26a)
	t27a = t24.SubSaturated(t27)
	t28a = t31.SubSaturated(t28)
	t29 = t30a.SubSaturated(t29a)
	t30 = t30a.AddSaturated(t29a)
	t31a = t31.AddSaturated(t28)

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

	out[0], out[31] = E[0].AddSaturated(t31), E[0].SubSaturated(t31)
	out[1], out[30] = E[1].AddSaturated(t30a), E[1].SubSaturated(t30a)
	out[2], out[29] = E[2].AddSaturated(t29), E[2].SubSaturated(t29)
	out[3], out[28] = E[3].AddSaturated(t28a), E[3].SubSaturated(t28a)
	out[4], out[27] = E[4].AddSaturated(t27), E[4].SubSaturated(t27)
	out[5], out[26] = E[5].AddSaturated(t26a), E[5].SubSaturated(t26a)
	out[6], out[25] = E[6].AddSaturated(t25), E[6].SubSaturated(t25)
	out[7], out[24] = E[7].AddSaturated(t24a), E[7].SubSaturated(t24a)
	out[8], out[23] = E[8].AddSaturated(t23a), E[8].SubSaturated(t23a)
	out[9], out[22] = E[9].AddSaturated(t22), E[9].SubSaturated(t22)
	out[10], out[21] = E[10].AddSaturated(t21a), E[10].SubSaturated(t21a)
	out[11], out[20] = E[11].AddSaturated(t20), E[11].SubSaturated(t20)
	out[12], out[19] = E[12].AddSaturated(t19a), E[12].SubSaturated(t19a)
	out[13], out[18] = E[13].AddSaturated(t18), E[13].SubSaturated(t18)
	out[14], out[17] = E[14].AddSaturated(t17a), E[14].SubSaturated(t17a)
	out[15], out[16] = E[15].AddSaturated(t16), E[15].SubSaturated(t16)
}

// itx32Net runs the DCT32 network form for n nonzero inputs (8, 16 or 32).
func itx32Net(n int, in *[32]archsimd.Int16x8, out *[32]archsimd.Int16x8) {
	switch {
	case n <= 8:
		itx32Low8Net((*[8]archsimd.Int16x8)(in[:8]), out)
	case n <= 16:
		itx32Low16Net((*[16]archsimd.Int16x8)(in[:16]), out)
	default:
		itx32FullNet(in, out)
	}
}

// itx32RowStrip runs the 32x32 row pass for rows r0..r0+7 with inCols coded
// coefficient columns (8, 16 or 32) into tmp rows r0..r0+7 and adds the
// stored values' magnitudes to the four column guards. It reports false,
// with tmp untouched, when a row fails the L1 guard.
func itx32RowStrip(coeff []int32, stride int, r0 int, inCols int, tmp *[32 * 32]int16, colL1 *[4]archsimd.Uint16x8) bool {
	_ = coeff[(inCols-1)*stride+r0+7]
	var in [32]archsimd.Int16x8
	var l1 archsimd.Uint16x8
	for k := 0; k < inCols; k++ {
		x := itx16LoadNarrow(coeff[k*stride+r0 : k*stride+r0+8])
		l1 = l1.AddSaturated(x.Abs().ToBits())
		in[k] = x
	}
	if l1.ReduceMax() > itx16ColumnL1Bound {
		return false
	}
	var out [32]archsimd.Int16x8
	itx32Net(inCols, &in, &out)
	one := archsimd.BroadcastInt16x8(1)
	sh1 := archsimd.BroadcastInt16x8(-1)
	for g := 0; g < 4; g++ {
		var v [8]archsimd.Int16x8
		for j := 0; j < 8; j++ {
			v[j] = itx32Round2(out[8*g+j], sh1, sh1, one)
		}
		itx16Transpose8x8(&v)
		gl1 := colL1[g]
		for r := 0; r < 8; r++ {
			v[r].StoreArray((*[8]int16)(tmp[(r0+r)*32+8*g:]))
			gl1 = gl1.AddSaturated(v[r].Abs().ToBits())
		}
		colL1[g] = gl1
	}
	return true
}

// InverseAddDCT32x32Full reconstructs one 8-bit 32x32 DCT_DCT block with
// coded coefficients in rows 0..activeRows-1 and columns 0..activeCols-1,
// whichever network forms that takes: dst = clip(pred + inverse(coeff)),
// with the dequantized coefficients column-major at coeffStride. pred may
// be dst. It reports false, with dst untouched, when the block is outside
// the kernel's contract or a guard declines; the caller then runs the
// staged path, which produces the same bytes. Blocks within the 8x8 or
// 16x16 corner are served faster by InverseAddDCT32x32Low8 and Low16.
func InverseAddDCT32x32Full(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, activeRows, activeCols int) bool {
	if activeRows <= 0 || activeRows > 32 || activeCols <= 0 || activeCols > 32 ||
		coeffStride != 32 || len(coeff) < 1024 || dstStride < 32 || predStride < 32 ||
		len(dst) < 31*dstStride+32 || len(pred) < 31*predStride+32 {
		return false
	}
	return itxGenDCT32x32((*[1024]int32)(coeff[:1024]), dst, dstStride, pred, predStride, activeCols, activeRows)
}

// itx_fused32: dav1d's inverse-transform pass structure (src/arm/64/itx.S
// inv_txfm_add_dct_dct_32x32_8bpc_neon) for the 8-bit 32x32 DCT_DCT block
// whose coded coefficients lie in rows and columns 0..7 (the eob-bounded
// shape the decoder meets most):
//
//   - the row pass reads the dequantized coefficients directly. They are
//     stored column-major, so coefficient k of rows 0..7 is one contiguous
//     load; narrowing it to int16 lanes (SQXTN) is the 8-bit row clamp. The
//     DCT32 runs across vectors with the rows in the lanes (the exact int16
//     network of the DCT32 Low8 column kernel, with its per-lane L1 guard),
//     the inter-pass Round2(., 2) is applied in registers, and the outputs
//     are transposed 8x8 at a time (ZIP) into one row-major int16 tmp, the
//     only intermediate dav1d keeps;
//   - each column strip loads its eight columns of tmp (contiguous), runs
//     the same network with the columns in the lanes, and its final stage
//     rounds by 4, adds the prediction row, saturates to bytes and stores:
//     the residual add is fused, with no int32 scratch or separate add pass.
//
// Exactness: both passes are the certified int16 network (itx16_gain_test.go,
// DCT32 bound 0.9989 plus 9.15 slack under the per-lane L1 guard
// itx16ColumnL1Bound), the butterflies saturate exactly as the scalar clip
// to the 8-bit row and column bounds (the int16 range), Round2(x, n) is
// formed as ((x >> (n-1)) + 1) >> 1 so no lane overflows, and the add
// saturates to [0, 255] as the raw add does. The row strip checks its guard
// before writing tmp and gathers the column strips' guards from the
// transposed values, so a declined block is left untouched and the caller
// falls back to the staged path.

// itx16Transpose8x8 transposes the 8x8 int16 matrix v (v[r] = row r) in
// registers: three interleave stages at 16, 32 and 64 bits.
func itx16Transpose8x8(v *[8]archsimd.Int16x8) {
	a0, a1 := v[0].InterleaveLo(v[1]).ToBits().ReshapeToUint32s(), v[0].InterleaveHi(v[1]).ToBits().ReshapeToUint32s()
	a2, a3 := v[2].InterleaveLo(v[3]).ToBits().ReshapeToUint32s(), v[2].InterleaveHi(v[3]).ToBits().ReshapeToUint32s()
	a4, a5 := v[4].InterleaveLo(v[5]).ToBits().ReshapeToUint32s(), v[4].InterleaveHi(v[5]).ToBits().ReshapeToUint32s()
	a6, a7 := v[6].InterleaveLo(v[7]).ToBits().ReshapeToUint32s(), v[6].InterleaveHi(v[7]).ToBits().ReshapeToUint32s()
	b0, b1 := a0.InterleaveLo(a2).ReshapeToUint64s(), a0.InterleaveHi(a2).ReshapeToUint64s()
	b2, b3 := a1.InterleaveLo(a3).ReshapeToUint64s(), a1.InterleaveHi(a3).ReshapeToUint64s()
	b4, b5 := a4.InterleaveLo(a6).ReshapeToUint64s(), a4.InterleaveHi(a6).ReshapeToUint64s()
	b6, b7 := a5.InterleaveLo(a7).ReshapeToUint64s(), a5.InterleaveHi(a7).ReshapeToUint64s()
	v[0] = b0.InterleaveLo(b4).ReshapeToUint16s().BitsToInt16()
	v[1] = b0.InterleaveHi(b4).ReshapeToUint16s().BitsToInt16()
	v[2] = b1.InterleaveLo(b5).ReshapeToUint16s().BitsToInt16()
	v[3] = b1.InterleaveHi(b5).ReshapeToUint16s().BitsToInt16()
	v[4] = b2.InterleaveLo(b6).ReshapeToUint16s().BitsToInt16()
	v[5] = b2.InterleaveHi(b6).ReshapeToUint16s().BitsToInt16()
	v[6] = b3.InterleaveLo(b7).ReshapeToUint16s().BitsToInt16()
	v[7] = b3.InterleaveHi(b7).ReshapeToUint16s().BitsToInt16()
}

// itx32Low8Net is inverseDCT32 on eight lanes for inputs whose indices 8..31
// are zero: in[k] is coefficient k, out[r] is output r. It is the network of
// inverseDCT32Col8RawLow8SIMD16 with the loads and stores lifted out.
func itx32Low8Net(in *[8]archsimd.Int16x8, out *[32]archsimd.Int16x8) {
	k := func(i int) archsimd.Int16x8 { return archsimd.LoadInt16x8Array(&itx16K[i]) }
	k32 := func(i int) archsimd.Int16x8 { return archsimd.LoadInt16x8Array(&itx32K[i]) }
	s12 := archsimd.LoadInt32x4Array(&itx16Scale12)
	s11 := archsimd.LoadInt32x4Array(&itx16Scale11)
	p2896, m2896 := k(itx16Kp2896), k(itx16Km2896)
	p1567, m1567, p3784, m3784 := k(itx16Kp1567), k(itx16Km1567), k(itx16Kp3784), k(itx16Km3784)

	// Even half: inverseDCT16 on inputs 0, 2, 4, 6 (its Low4 form).
	in0, in2, in4, in6 := in[0], in[2], in[4], in[6]
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

	// Odd half from inputs 1, 3, 5, 7: the first-stage rotations are single
	// products and the second-stage butterflies pass them through.
	in1, in3, in5, in7 := in[1], in[3], in[5], in[7]
	t16 := itx16Rot1(in1, k32(itx32Kp201), s12)
	t31 := itx16Rot1(in1, k32(itx32Kp4091), s12)
	t19 := itx16Rot1(in7, k32(itx32Km1380), s12)
	t28 := itx16Rot1(in7, k32(itx32Kp3857), s12)
	t20 := itx16Rot1(in5, k32(itx32Kp995), s12)
	t27 := itx16Rot1(in5, k32(itx32Kp3973), s12)
	t23 := itx16Rot1(in3, k32(itx32Km601), s12)
	t24 := itx16Rot1(in3, k32(itx32Kp4052), s12)
	t17, t18, t21, t22, t25, t26, t29, t30 := t16, t19, t20, t23, t24, t27, t28, t31

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

	out[0], out[31] = E0.AddSaturated(t31), E0.SubSaturated(t31)
	out[1], out[30] = E1.AddSaturated(t30a), E1.SubSaturated(t30a)
	out[2], out[29] = E2.AddSaturated(t29), E2.SubSaturated(t29)
	out[3], out[28] = E3.AddSaturated(t28a), E3.SubSaturated(t28a)
	out[4], out[27] = E4.AddSaturated(t27), E4.SubSaturated(t27)
	out[5], out[26] = E5.AddSaturated(t26a), E5.SubSaturated(t26a)
	out[6], out[25] = E6.AddSaturated(t25), E6.SubSaturated(t25)
	out[7], out[24] = E7.AddSaturated(t24a), E7.SubSaturated(t24a)
	out[8], out[23] = E8.AddSaturated(t23a), E8.SubSaturated(t23a)
	out[9], out[22] = E9.AddSaturated(t22), E9.SubSaturated(t22)
	out[10], out[21] = E10.AddSaturated(t21a), E10.SubSaturated(t21a)
	out[11], out[20] = E11.AddSaturated(t20), E11.SubSaturated(t20)
	out[12], out[19] = E12.AddSaturated(t19a), E12.SubSaturated(t19a)
	out[13], out[18] = E13.AddSaturated(t18), E13.SubSaturated(t18)
	out[14], out[17] = E14.AddSaturated(t17a), E14.SubSaturated(t17a)
	out[15], out[16] = E15.AddSaturated(t16), E15.SubSaturated(t16)
}

// itx32Round2 is Round2(x, n) on int16 lanes as ((x >> (n-1)) + 1) >> 1,
// which equals (x + 2^(n-1)) >> n for every lane without overflowing.
func itx32Round2(x archsimd.Int16x8, shiftNm1, shift1 archsimd.Int16x8, one archsimd.Int16x8) archsimd.Int16x8 {
	return x.Shift(shiftNm1).Add(one).Shift(shift1)
}

// itx32RowStripLow8 runs the row pass of the 8-bit 32x32 DCT_DCT block
// whose coefficients lie in rows and columns 0..7: rows 0..7 of the
// column-major coefficients, narrowed to int16, transformed across vectors,
// rounded by 2, transposed and stored into tmp rows 0..7 (32 wide). It
// reports false, with tmp untouched, when a row fails the L1 guard, and
// otherwise also checks the column strips' guards on the stored values.
func itx32RowStripLow8(coeff []int32, stride int, tmp *[8 * 32]int16) bool {
	_ = coeff[7*stride+7]
	var in [8]archsimd.Int16x8
	var l1 archsimd.Uint16x8
	for k := 0; k < 8; k++ {
		x := itx16LoadNarrow(coeff[k*stride : k*stride+8])
		l1 = l1.AddSaturated(x.Abs().ToBits())
		in[k] = x
	}
	if l1.ReduceMax() > itx16ColumnL1Bound {
		return false
	}
	var out [32]archsimd.Int16x8
	itx32Low8Net(&in, &out)
	one := archsimd.BroadcastInt16x8(1)
	sh1 := archsimd.BroadcastInt16x8(-1)
	var colL1 archsimd.Uint16x8
	for g := 0; g < 4; g++ {
		var v [8]archsimd.Int16x8
		for j := 0; j < 8; j++ {
			v[j] = itx32Round2(out[8*g+j], sh1, sh1, one)
		}
		itx16Transpose8x8(&v)
		var gl1 archsimd.Uint16x8
		for r := 0; r < 8; r++ {
			v[r].StoreArray((*[8]int16)(tmp[r*32+8*g:]))
			gl1 = gl1.AddSaturated(v[r].Abs().ToBits())
		}
		colL1 = colL1.Max(gl1)
	}
	return colL1.ReduceMax() <= itx16ColumnL1Bound
}

// InverseAddDCT32x32Low8 reconstructs one 8-bit 32x32 DCT_DCT block whose
// coded coefficients lie in rows and columns 0..7 (the caller's activeRows
// and activeCols are at most 8): dst = clip(pred + inverse(coeff)), with
// the dequantized coefficients column-major at coeffStride. pred may be dst.
// It reports false, with dst untouched, when the block is outside the
// kernel's contract or a guard declines; the caller then runs the staged
// path, which produces the same bytes.
func InverseAddDCT32x32Low8(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int) bool {
	if coeffStride != 32 || len(coeff) < 1024 || dstStride < 32 || predStride < 32 ||
		len(dst) < 31*dstStride+32 || len(pred) < 31*predStride+32 {
		return false
	}
	return itxGenDCT32x32C8R8((*[1024]int32)(coeff[:1024]), dst, dstStride, pred, predStride, 8)
}

// The Low16 form of itx_fused32: the 8-bit 32x32 DCT_DCT block whose coded
// coefficients lie in rows and columns 0..15. Two row strips of eight rows
// each read coefficients 0..15, the column strips read sixteen tmp rows, and
// the network's first stage has single-product rotations for every odd
// input (inputs 17..31 are zero); everything else is the Low8 form's.

// itx32K16 holds the odd-half rotation constants the Low16 first stage needs
// beyond itx32K.
var itx32K16 = [...][8]int16{
	{-2751, -2751, -2751, -2751, -2751, -2751, -2751, -2751},
	{3035, 3035, 3035, 3035, 3035, 3035, 3035, 3035},
	{1751, 1751, 1751, 1751, 1751, 1751, 1751, 1751},
	{3703, 3703, 3703, 3703, 3703, 3703, 3703, 3703},
	{-2106, -2106, -2106, -2106, -2106, -2106, -2106, -2106},
	{3513, 3513, 3513, 3513, 3513, 3513, 3513, 3513},
	{1220, 1220, 1220, 1220, 1220, 1220, 1220, 1220},
	{1645, 1645, 1645, 1645, 1645, 1645, 1645, 1645},
}

const (
	itx32K16m2751 = iota
	itx32K16p3035
	itx32K16p1751
	itx32K16p3703
	itx32K16m2106
	itx32K16p3513
	itx32K16p1220
	itx32K16p1645
)

// itx32Low16Net is inverseDCT32 on eight lanes for inputs whose indices
// 16..31 are zero: in[k] is coefficient k, out[r] is output r.
func itx32Low16Net(in *[16]archsimd.Int16x8, out *[32]archsimd.Int16x8) {
	k := func(i int) archsimd.Int16x8 { return archsimd.LoadInt16x8Array(&itx16K[i]) }
	k32 := func(i int) archsimd.Int16x8 { return archsimd.LoadInt16x8Array(&itx32K[i]) }
	k16 := func(i int) archsimd.Int16x8 { return archsimd.LoadInt16x8Array(&itx32K16[i]) }
	s12 := archsimd.LoadInt32x4Array(&itx16Scale12)
	s11 := archsimd.LoadInt32x4Array(&itx16Scale11)
	p2896, m2896 := k(itx16Kp2896), k(itx16Km2896)
	p1567, m1567, p3784, m3784 := k(itx16Kp1567), k(itx16Km1567), k(itx16Kp3784), k(itx16Km3784)

	// Even half: inverseDCT16 on inputs 0, 2, ..., 14 (its Low8 form with
	// DCT16 inputs 0..7 = DCT32 inputs 0, 2, ..., 14).
	e0i, e1i, e2i, e3i := in[0], in[2], in[4], in[6]
	e4i, e5i, e6i, e7i := in[8], in[10], in[12], in[14]
	t0 := itx16Rot1(e0i, p2896, s12)
	t2 := itx16Rot1(e4i, p1567, s12)
	t3 := itx16Rot1(e4i, p3784, s12)
	d0 := t0.AddSaturated(t3)
	d1 := t0.AddSaturated(t2)
	d2 := t0.SubSaturated(t2)
	d3 := t0.SubSaturated(t3)
	t4a := itx16Rot1(e2i, k(itx16Kp799), s12)
	t5a := itx16Rot1(e6i, k(itx16Km1138), s11)
	t6a := itx16Rot1(e6i, k(itx16Kp1703), s11)
	t7a := itx16Rot1(e2i, k(itx16Kp4017), s12)
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
	t8a := itx16Rot1(e1i, k(itx16Kp401), s12)
	t15a := itx16Rot1(e1i, k(itx16Kp4076), s12)
	t9a := itx16Rot1(e7i, k(itx16Km1299), s11)
	t14a := itx16Rot1(e7i, k(itx16Kp1583), s11)
	t10a := itx16Rot1(e5i, k(itx16Kp1931), s12)
	t13a := itx16Rot1(e5i, k(itx16Kp3612), s12)
	t11a := itx16Rot1(e3i, k(itx16Km1189), s12)
	t12a := itx16Rot1(e3i, k(itx16Kp3920), s12)
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
	t10a = itx16Rot(t13, t10, m3784, m1567, s12)
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
	E0, E15 := e0.AddSaturated(t15a), e0.SubSaturated(t15a)
	E1, E14 := e1.AddSaturated(t14), e1.SubSaturated(t14)
	E2, E13 := e2.AddSaturated(t13a), e2.SubSaturated(t13a)
	E3, E12 := e3.AddSaturated(t12), e3.SubSaturated(t12)
	E4, E11 := e4.AddSaturated(t11), e4.SubSaturated(t11)
	E5, E10 := e5.AddSaturated(t10a), e5.SubSaturated(t10a)
	E6, E9 := e6.AddSaturated(t9), e6.SubSaturated(t9)
	E7, E8 := e7.AddSaturated(t8a), e7.SubSaturated(t8a)

	// Odd half from inputs 1, 3, ..., 15: single-product first stage.
	in1, in3, in5, in7 := in[1], in[3], in[5], in[7]
	in9, in11, in13, in15 := in[9], in[11], in[13], in[15]
	t16a := itx16Rot1(in1, k32(itx32Kp201), s12)
	t31a := itx16Rot1(in1, k32(itx32Kp4091), s12)
	t17a := itx16Rot1(in15, k16(itx32K16m2751), s12)
	t30a := itx16Rot1(in15, k16(itx32K16p3035), s12)
	t18a := itx16Rot1(in9, k16(itx32K16p1751), s12)
	t29a := itx16Rot1(in9, k16(itx32K16p3703), s12)
	t19a := itx16Rot1(in7, k32(itx32Km1380), s12)
	t28a := itx16Rot1(in7, k32(itx32Kp3857), s12)
	t20a := itx16Rot1(in5, k32(itx32Kp995), s12)
	t27a := itx16Rot1(in5, k32(itx32Kp3973), s12)
	t21a := itx16Rot1(in11, k16(itx32K16m2106), s12)
	t26a := itx16Rot1(in11, k16(itx32K16p3513), s12)
	t22a := itx16Rot1(in13, k16(itx32K16p1220), s11)
	t25a := itx16Rot1(in13, k16(itx32K16p1645), s11)
	t23a := itx16Rot1(in3, k32(itx32Km601), s12)
	t24a := itx16Rot1(in3, k32(itx32Kp4052), s12)

	t16 := t16a.AddSaturated(t17a)
	t17 := t16a.SubSaturated(t17a)
	t18 := t19a.SubSaturated(t18a)
	t19 := t19a.AddSaturated(t18a)
	t20 := t20a.AddSaturated(t21a)
	t21 := t20a.SubSaturated(t21a)
	t22 := t23a.SubSaturated(t22a)
	t23 := t23a.AddSaturated(t22a)
	t24 := t24a.AddSaturated(t25a)
	t25 := t24a.SubSaturated(t25a)
	t26 := t27a.SubSaturated(t26a)
	t27 := t27a.AddSaturated(t26a)
	t28 := t28a.AddSaturated(t29a)
	t29 := t28a.SubSaturated(t29a)
	t30 := t31a.SubSaturated(t30a)
	t31 := t31a.AddSaturated(t30a)

	p799, m799, p4017, m4017 := k(itx16Kp799), k32(itx32Km799), k(itx16Kp4017), k(itx16Km4017)
	t17a = itx16Rot(t30, t17, p799, m4017, s12)
	t30a = itx16Rot(t30, t17, p4017, p799, s12)
	t18a = itx16Rot(t29, t18, m4017, m799, s12)
	t29a = itx16Rot(t29, t18, p799, m4017, s12)
	p1703, m1703, p1138, m1138 := k(itx16Kp1703), k32(itx32Km1703), k(itx16Kp1138), k(itx16Km1138)
	t21a = itx16Rot(t26, t21, p1703, m1138, s11)
	t26a = itx16Rot(t26, t21, p1138, p1703, s11)
	t22a = itx16Rot(t25, t22, m1138, m1703, s11)
	t25a = itx16Rot(t25, t22, p1703, m1138, s11)

	t16a = t16.AddSaturated(t19)
	t17 = t17a.AddSaturated(t18a)
	t18 = t17a.SubSaturated(t18a)
	t19a = t16.SubSaturated(t19)
	t20a = t23.SubSaturated(t20)
	t21 = t22a.SubSaturated(t21a)
	t22 = t22a.AddSaturated(t21a)
	t23a = t23.AddSaturated(t20)
	t24a = t24.AddSaturated(t27)
	t25 = t25a.AddSaturated(t26a)
	t26 = t25a.SubSaturated(t26a)
	t27a = t24.SubSaturated(t27)
	t28a = t31.SubSaturated(t28)
	t29 = t30a.SubSaturated(t29a)
	t30 = t30a.AddSaturated(t29a)
	t31a = t31.AddSaturated(t28)

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

	out[0], out[31] = E0.AddSaturated(t31), E0.SubSaturated(t31)
	out[1], out[30] = E1.AddSaturated(t30a), E1.SubSaturated(t30a)
	out[2], out[29] = E2.AddSaturated(t29), E2.SubSaturated(t29)
	out[3], out[28] = E3.AddSaturated(t28a), E3.SubSaturated(t28a)
	out[4], out[27] = E4.AddSaturated(t27), E4.SubSaturated(t27)
	out[5], out[26] = E5.AddSaturated(t26a), E5.SubSaturated(t26a)
	out[6], out[25] = E6.AddSaturated(t25), E6.SubSaturated(t25)
	out[7], out[24] = E7.AddSaturated(t24a), E7.SubSaturated(t24a)
	out[8], out[23] = E8.AddSaturated(t23a), E8.SubSaturated(t23a)
	out[9], out[22] = E9.AddSaturated(t22), E9.SubSaturated(t22)
	out[10], out[21] = E10.AddSaturated(t21a), E10.SubSaturated(t21a)
	out[11], out[20] = E11.AddSaturated(t20), E11.SubSaturated(t20)
	out[12], out[19] = E12.AddSaturated(t19a), E12.SubSaturated(t19a)
	out[13], out[18] = E13.AddSaturated(t18), E13.SubSaturated(t18)
	out[14], out[17] = E14.AddSaturated(t17a), E14.SubSaturated(t17a)
	out[15], out[16] = E15.AddSaturated(t16), E15.SubSaturated(t16)
}

// itx32RowStripLow16 runs the row pass for rows r0..r0+7 of a block whose
// coefficients lie in rows and columns 0..15, into tmp rows r0..r0+7, and
// adds the stored values' magnitudes to the column guards colL1[g] (strip g
// of eight columns). It reports false, with tmp untouched, when a row
// fails the L1 guard.
func itx32RowStripLow16(coeff []int32, stride int, r0 int, tmp *[16 * 32]int16, colL1 *[4]archsimd.Uint16x8) bool {
	_ = coeff[15*stride+r0+7]
	var in [16]archsimd.Int16x8
	var l1 archsimd.Uint16x8
	for k := 0; k < 16; k++ {
		x := itx16LoadNarrow(coeff[k*stride+r0 : k*stride+r0+8])
		l1 = l1.AddSaturated(x.Abs().ToBits())
		in[k] = x
	}
	if l1.ReduceMax() > itx16ColumnL1Bound {
		return false
	}
	var out [32]archsimd.Int16x8
	itx32Low16Net(&in, &out)
	one := archsimd.BroadcastInt16x8(1)
	sh1 := archsimd.BroadcastInt16x8(-1)
	for g := 0; g < 4; g++ {
		var v [8]archsimd.Int16x8
		for j := 0; j < 8; j++ {
			v[j] = itx32Round2(out[8*g+j], sh1, sh1, one)
		}
		itx16Transpose8x8(&v)
		gl1 := colL1[g]
		for r := 0; r < 8; r++ {
			v[r].StoreArray((*[8]int16)(tmp[(r0+r)*32+8*g:]))
			gl1 = gl1.AddSaturated(v[r].Abs().ToBits())
		}
		colL1[g] = gl1
	}
	return true
}

// InverseAddDCT32x32Low16 is InverseAddDCT32x32Low8 for a block whose coded
// coefficients lie in rows and columns 0..15 (activeRows and activeCols at
// most 16); activeRows tells it whether coefficient rows 8..15 hold
// anything, so a block with eight coded rows but up to sixteen coded
// columns skips its second row strip.
func InverseAddDCT32x32Low16(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, activeRows int) bool {
	if coeffStride != 32 || len(coeff) < 1024 || activeRows <= 0 || activeRows > 16 || dstStride < 32 || predStride < 32 ||
		len(dst) < 31*dstStride+32 || len(pred) < 31*predStride+32 {
		return false
	}
	return itxGenDCT32x32((*[1024]int32)(coeff[:1024]), dst, dstStride, pred, predStride, 16, activeRows)
}

// itx_fused64: dav1d's pass structure (itx_fused32) for the 8-bit 64x64
// DCT_DCT block, whose coded coefficients lie in the 32x32 corner. The row
// pass runs up to four eight-row strips straight from the column-major
// coefficients with the DCT64 network form that matches the coded columns
// (8, 16 or 32 inputs), rounds by 2 in registers and transposes 8x8 into a
// 32x64 int16 tmp; the eight column strips run the form matching the coded
// rows with Round2(., 4), the prediction add and the byte saturation fused
// into their final stage. The even half of the network is the DCT32 of
// itx_fused32 on the even inputs; the odd half (itx64Odd*) is generated
// from the stage structure of inverseDCT64 with the zero operands of each
// form folded. itx64_gain_test.go certifies the per-lane L1 guard for it.

// itx64K[i] broadcasts cospi(i) and itx64K[64+i] broadcasts -cospi(i).
var itx64K = func() (k [128][8]int16) {
	for i, c := range itx64Cos {
		for l := 0; l < 8; l++ {
			k[i][l] = c
			k[64+i][l] = -c
		}
	}
	return k
}()

// itx64Odd32 is the odd half of inverseDCT64 on eight lanes for odd inputs
// below 32 (the rest zero): in[k] is coefficient k and o[j] is stage-10
// value 32+j. Generated from the stage structure of inverseDCT64 with the
// zero operands folded.
func itx64Odd32(in *[32]archsimd.Int16x8, o *[32]archsimd.Int16x8) {
	k := func(i int) archsimd.Int16x8 { return archsimd.LoadInt16x8Array(&itx64K[i]) }
	kn := func(i int) archsimd.Int16x8 { return archsimd.LoadInt16x8Array(&itx64K[64+i]) }
	s12 := archsimd.LoadInt32x4Array(&itx16Scale12)
	a1 := itx16Rot1(in[1], k(63), s12)
	a2 := itx16Rot1(in[1], k(1), s12)
	a3 := itx16Rot1(in[31], kn(33), s12)
	a4 := itx16Rot1(in[31], k(31), s12)
	a5 := itx16Rot1(in[17], k(47), s12)
	a6 := itx16Rot1(in[17], k(17), s12)
	a7 := itx16Rot1(in[15], kn(49), s12)
	a8 := itx16Rot1(in[15], k(15), s12)
	a9 := itx16Rot1(in[9], k(55), s12)
	a10 := itx16Rot1(in[9], k(9), s12)
	a11 := itx16Rot1(in[23], kn(41), s12)
	a12 := itx16Rot1(in[23], k(23), s12)
	a13 := itx16Rot1(in[25], k(39), s12)
	a14 := itx16Rot1(in[25], k(25), s12)
	a15 := itx16Rot1(in[7], kn(57), s12)
	a16 := itx16Rot1(in[7], k(7), s12)
	a17 := itx16Rot1(in[5], k(59), s12)
	a18 := itx16Rot1(in[5], k(5), s12)
	a19 := itx16Rot1(in[27], kn(37), s12)
	a20 := itx16Rot1(in[27], k(27), s12)
	a21 := itx16Rot1(in[21], k(43), s12)
	a22 := itx16Rot1(in[21], k(21), s12)
	a23 := itx16Rot1(in[11], kn(53), s12)
	a24 := itx16Rot1(in[11], k(11), s12)
	a25 := itx16Rot1(in[13], k(51), s12)
	a26 := itx16Rot1(in[13], k(13), s12)
	a27 := itx16Rot1(in[19], kn(45), s12)
	a28 := itx16Rot1(in[19], k(19), s12)
	a29 := itx16Rot1(in[29], k(35), s12)
	a30 := itx16Rot1(in[29], k(29), s12)
	a31 := itx16Rot1(in[3], kn(61), s12)
	a32 := itx16Rot1(in[3], k(3), s12)
	b33 := a1.AddSaturated(a3)
	b34 := a1.SubSaturated(a3)
	b35 := a7.SubSaturated(a5)
	b36 := a5.AddSaturated(a7)
	b37 := a9.AddSaturated(a11)
	b38 := a9.SubSaturated(a11)
	b39 := a15.SubSaturated(a13)
	b40 := a13.AddSaturated(a15)
	b41 := a17.AddSaturated(a19)
	b42 := a17.SubSaturated(a19)
	b43 := a23.SubSaturated(a21)
	b44 := a21.AddSaturated(a23)
	b45 := a25.AddSaturated(a27)
	b46 := a25.SubSaturated(a27)
	b47 := a31.SubSaturated(a29)
	b48 := a29.AddSaturated(a31)
	b49 := a32.AddSaturated(a30)
	b50 := a32.SubSaturated(a30)
	b51 := a26.SubSaturated(a28)
	b52 := a28.AddSaturated(a26)
	b53 := a24.AddSaturated(a22)
	b54 := a24.SubSaturated(a22)
	b55 := a18.SubSaturated(a20)
	b56 := a20.AddSaturated(a18)
	b57 := a16.AddSaturated(a14)
	b58 := a16.SubSaturated(a14)
	b59 := a10.SubSaturated(a12)
	b60 := a12.AddSaturated(a10)
	b61 := a8.AddSaturated(a6)
	b62 := a8.SubSaturated(a6)
	b63 := a2.SubSaturated(a4)
	b64 := a4.AddSaturated(a2)
	c65 := itx16Rot(b34, b63, kn(4), k(60), s12)
	c66 := itx16Rot(b34, b63, k(60), k(4), s12)
	c67 := itx16Rot(b35, b62, kn(60), kn(4), s12)
	c68 := itx16Rot(b35, b62, kn(4), k(60), s12)
	c69 := itx16Rot(b38, b59, kn(36), k(28), s12)
	c70 := itx16Rot(b38, b59, k(28), k(36), s12)
	c71 := itx16Rot(b39, b58, kn(28), kn(36), s12)
	c72 := itx16Rot(b39, b58, kn(36), k(28), s12)
	c73 := itx16Rot(b42, b55, kn(20), k(44), s12)
	c74 := itx16Rot(b42, b55, k(44), k(20), s12)
	c75 := itx16Rot(b43, b54, kn(44), kn(20), s12)
	c76 := itx16Rot(b43, b54, kn(20), k(44), s12)
	c77 := itx16Rot(b46, b51, kn(52), k(12), s12)
	c78 := itx16Rot(b46, b51, k(12), k(52), s12)
	c79 := itx16Rot(b47, b50, kn(12), kn(52), s12)
	c80 := itx16Rot(b47, b50, kn(52), k(12), s12)
	d81 := b33.AddSaturated(b36)
	d82 := c65.AddSaturated(c67)
	d83 := c65.SubSaturated(c67)
	d84 := b33.SubSaturated(b36)
	d85 := b40.SubSaturated(b37)
	d86 := c71.SubSaturated(c69)
	d87 := c69.AddSaturated(c71)
	d88 := b37.AddSaturated(b40)
	d89 := b41.AddSaturated(b44)
	d90 := c73.AddSaturated(c75)
	d91 := c73.SubSaturated(c75)
	d92 := b41.SubSaturated(b44)
	d93 := b48.SubSaturated(b45)
	d94 := c79.SubSaturated(c77)
	d95 := c77.AddSaturated(c79)
	d96 := b45.AddSaturated(b48)
	d97 := b49.AddSaturated(b52)
	d98 := c80.AddSaturated(c78)
	d99 := c80.SubSaturated(c78)
	d100 := b49.SubSaturated(b52)
	d101 := b56.SubSaturated(b53)
	d102 := c74.SubSaturated(c76)
	d103 := c76.AddSaturated(c74)
	d104 := b53.AddSaturated(b56)
	d105 := b57.AddSaturated(b60)
	d106 := c72.AddSaturated(c70)
	d107 := c72.SubSaturated(c70)
	d108 := b57.SubSaturated(b60)
	d109 := b64.SubSaturated(b61)
	d110 := c66.SubSaturated(c68)
	d111 := c68.AddSaturated(c66)
	d112 := b61.AddSaturated(b64)
	e113 := itx16Rot(d83, d110, kn(8), k(56), s12)
	e114 := itx16Rot(d83, d110, k(56), k(8), s12)
	e115 := itx16Rot(d84, d109, kn(8), k(56), s12)
	e116 := itx16Rot(d84, d109, k(56), k(8), s12)
	e117 := itx16Rot(d85, d108, kn(56), kn(8), s12)
	e118 := itx16Rot(d85, d108, kn(8), k(56), s12)
	e119 := itx16Rot(d86, d107, kn(56), kn(8), s12)
	e120 := itx16Rot(d86, d107, kn(8), k(56), s12)
	e121 := itx16Rot(d91, d102, kn(40), k(24), s12)
	e122 := itx16Rot(d91, d102, k(24), k(40), s12)
	e123 := itx16Rot(d92, d101, kn(40), k(24), s12)
	e124 := itx16Rot(d92, d101, k(24), k(40), s12)
	e125 := itx16Rot(d93, d100, kn(24), kn(40), s12)
	e126 := itx16Rot(d93, d100, kn(40), k(24), s12)
	e127 := itx16Rot(d94, d99, kn(24), kn(40), s12)
	e128 := itx16Rot(d94, d99, kn(40), k(24), s12)
	f129 := d81.AddSaturated(d88)
	f130 := d81.SubSaturated(d88)
	f131 := d96.SubSaturated(d89)
	f132 := d89.AddSaturated(d96)
	f133 := d82.AddSaturated(d87)
	f134 := d82.SubSaturated(d87)
	f135 := d95.SubSaturated(d90)
	f136 := d90.AddSaturated(d95)
	f137 := e113.AddSaturated(e119)
	f138 := e113.SubSaturated(e119)
	f139 := e127.SubSaturated(e121)
	f140 := e121.AddSaturated(e127)
	f141 := e115.AddSaturated(e117)
	f142 := e115.SubSaturated(e117)
	f143 := e125.SubSaturated(e123)
	f144 := e123.AddSaturated(e125)
	f145 := d97.AddSaturated(d104)
	f146 := d97.SubSaturated(d104)
	f147 := d112.SubSaturated(d105)
	f148 := d105.AddSaturated(d112)
	f149 := d98.AddSaturated(d103)
	f150 := d98.SubSaturated(d103)
	f151 := d111.SubSaturated(d106)
	f152 := d106.AddSaturated(d111)
	f153 := e128.AddSaturated(e122)
	f154 := e128.SubSaturated(e122)
	f155 := e114.SubSaturated(e120)
	f156 := e120.AddSaturated(e114)
	f157 := e126.AddSaturated(e124)
	f158 := e126.SubSaturated(e124)
	f159 := e116.SubSaturated(e118)
	f160 := e118.AddSaturated(e116)
	g161 := itx16Rot(f142, f159, kn(16), k(48), s12)
	g162 := itx16Rot(f142, f159, k(48), k(16), s12)
	g163 := itx16Rot(f131, f146, kn(48), kn(16), s12)
	g164 := itx16Rot(f131, f146, kn(16), k(48), s12)
	g165 := itx16Rot(f138, f155, kn(16), k(48), s12)
	g166 := itx16Rot(f138, f155, k(48), k(16), s12)
	g167 := itx16Rot(f135, f150, kn(48), kn(16), s12)
	g168 := itx16Rot(f135, f150, kn(16), k(48), s12)
	g169 := itx16Rot(f134, f151, kn(16), k(48), s12)
	g170 := itx16Rot(f134, f151, k(48), k(16), s12)
	g171 := itx16Rot(f139, f154, kn(48), kn(16), s12)
	g172 := itx16Rot(f139, f154, kn(16), k(48), s12)
	g173 := itx16Rot(f130, f147, kn(16), k(48), s12)
	g174 := itx16Rot(f130, f147, k(48), k(16), s12)
	g175 := itx16Rot(f143, f158, kn(48), kn(16), s12)
	g176 := itx16Rot(f143, f158, kn(16), k(48), s12)
	h177 := f129.AddSaturated(f132)
	h178 := f129.SubSaturated(f132)
	h179 := f148.SubSaturated(f145)
	h180 := f145.AddSaturated(f148)
	h181 := f133.AddSaturated(f136)
	h182 := f133.SubSaturated(f136)
	h183 := f152.SubSaturated(f149)
	h184 := f149.AddSaturated(f152)
	h185 := f137.AddSaturated(f140)
	h186 := f137.SubSaturated(f140)
	h187 := f156.SubSaturated(f153)
	h188 := f153.AddSaturated(f156)
	h189 := f141.AddSaturated(f144)
	h190 := f141.SubSaturated(f144)
	h191 := f160.SubSaturated(f157)
	h192 := f157.AddSaturated(f160)
	h193 := g161.AddSaturated(g175)
	h194 := g161.SubSaturated(g175)
	h195 := g162.SubSaturated(g176)
	h196 := g176.AddSaturated(g162)
	h197 := g165.AddSaturated(g171)
	h198 := g165.SubSaturated(g171)
	h199 := g166.SubSaturated(g172)
	h200 := g172.AddSaturated(g166)
	h201 := g169.AddSaturated(g167)
	h202 := g169.SubSaturated(g167)
	h203 := g170.SubSaturated(g168)
	h204 := g168.AddSaturated(g170)
	h205 := g173.AddSaturated(g163)
	h206 := g173.SubSaturated(g163)
	h207 := g174.SubSaturated(g164)
	h208 := g164.AddSaturated(g174)
	i209 := itx16Rot(h206, h207, kn(32), k(32), s12)
	i210 := itx16Rot(h206, h207, k(32), k(32), s12)
	i211 := itx16Rot(h202, h203, kn(32), k(32), s12)
	i212 := itx16Rot(h202, h203, k(32), k(32), s12)
	i213 := itx16Rot(h198, h199, kn(32), k(32), s12)
	i214 := itx16Rot(h198, h199, k(32), k(32), s12)
	i215 := itx16Rot(h194, h195, kn(32), k(32), s12)
	i216 := itx16Rot(h194, h195, k(32), k(32), s12)
	i217 := itx16Rot(h190, h191, kn(32), k(32), s12)
	i218 := itx16Rot(h190, h191, k(32), k(32), s12)
	i219 := itx16Rot(h186, h187, kn(32), k(32), s12)
	i220 := itx16Rot(h186, h187, k(32), k(32), s12)
	i221 := itx16Rot(h182, h183, kn(32), k(32), s12)
	i222 := itx16Rot(h182, h183, k(32), k(32), s12)
	i223 := itx16Rot(h178, h179, kn(32), k(32), s12)
	i224 := itx16Rot(h178, h179, k(32), k(32), s12)
	o[0] = h177
	o[1] = h181
	o[2] = h185
	o[3] = h189
	o[4] = h193
	o[5] = h197
	o[6] = h201
	o[7] = h205
	o[8] = i209
	o[9] = i211
	o[10] = i213
	o[11] = i215
	o[12] = i217
	o[13] = i219
	o[14] = i221
	o[15] = i223
	o[16] = i224
	o[17] = i222
	o[18] = i220
	o[19] = i218
	o[20] = i216
	o[21] = i214
	o[22] = i212
	o[23] = i210
	o[24] = h208
	o[25] = h204
	o[26] = h200
	o[27] = h196
	o[28] = h192
	o[29] = h188
	o[30] = h184
	o[31] = h180
}

// itx64Odd16 is the odd half of inverseDCT64 on eight lanes for odd inputs
// below 16 (the rest zero): in[k] is coefficient k and o[j] is stage-10
// value 32+j. Generated from the stage structure of inverseDCT64 with the
// zero operands folded.
func itx64Odd16(in *[32]archsimd.Int16x8, o *[32]archsimd.Int16x8) {
	k := func(i int) archsimd.Int16x8 { return archsimd.LoadInt16x8Array(&itx64K[i]) }
	kn := func(i int) archsimd.Int16x8 { return archsimd.LoadInt16x8Array(&itx64K[64+i]) }
	s12 := archsimd.LoadInt32x4Array(&itx16Scale12)
	a1 := itx16Rot1(in[1], k(63), s12)
	a2 := itx16Rot1(in[1], k(1), s12)
	a3 := itx16Rot1(in[15], kn(49), s12)
	a4 := itx16Rot1(in[15], k(15), s12)
	a5 := itx16Rot1(in[9], k(55), s12)
	a6 := itx16Rot1(in[9], k(9), s12)
	a7 := itx16Rot1(in[7], kn(57), s12)
	a8 := itx16Rot1(in[7], k(7), s12)
	a9 := itx16Rot1(in[5], k(59), s12)
	a10 := itx16Rot1(in[5], k(5), s12)
	a11 := itx16Rot1(in[11], kn(53), s12)
	a12 := itx16Rot1(in[11], k(11), s12)
	a13 := itx16Rot1(in[13], k(51), s12)
	a14 := itx16Rot1(in[13], k(13), s12)
	a15 := itx16Rot1(in[3], kn(61), s12)
	a16 := itx16Rot1(in[3], k(3), s12)
	c17 := itx16Rot(a1, a2, kn(4), k(60), s12)
	c18 := itx16Rot(a1, a2, k(60), k(4), s12)
	c19 := itx16Rot(a3, a4, kn(60), kn(4), s12)
	c20 := itx16Rot(a3, a4, kn(4), k(60), s12)
	c21 := itx16Rot(a5, a6, kn(36), k(28), s12)
	c22 := itx16Rot(a5, a6, k(28), k(36), s12)
	c23 := itx16Rot(a7, a8, kn(28), kn(36), s12)
	c24 := itx16Rot(a7, a8, kn(36), k(28), s12)
	c25 := itx16Rot(a9, a10, kn(20), k(44), s12)
	c26 := itx16Rot(a9, a10, k(44), k(20), s12)
	c27 := itx16Rot(a11, a12, kn(44), kn(20), s12)
	c28 := itx16Rot(a11, a12, kn(20), k(44), s12)
	c29 := itx16Rot(a13, a14, kn(52), k(12), s12)
	c30 := itx16Rot(a13, a14, k(12), k(52), s12)
	c31 := itx16Rot(a15, a16, kn(12), kn(52), s12)
	c32 := itx16Rot(a15, a16, kn(52), k(12), s12)
	d33 := a1.AddSaturated(a3)
	d34 := c17.AddSaturated(c19)
	d35 := c17.SubSaturated(c19)
	d36 := a1.SubSaturated(a3)
	d37 := a7.SubSaturated(a5)
	d38 := c23.SubSaturated(c21)
	d39 := c21.AddSaturated(c23)
	d40 := a5.AddSaturated(a7)
	d41 := a9.AddSaturated(a11)
	d42 := c25.AddSaturated(c27)
	d43 := c25.SubSaturated(c27)
	d44 := a9.SubSaturated(a11)
	d45 := a15.SubSaturated(a13)
	d46 := c31.SubSaturated(c29)
	d47 := c29.AddSaturated(c31)
	d48 := a13.AddSaturated(a15)
	d49 := a16.AddSaturated(a14)
	d50 := c32.AddSaturated(c30)
	d51 := c32.SubSaturated(c30)
	d52 := a16.SubSaturated(a14)
	d53 := a10.SubSaturated(a12)
	d54 := c26.SubSaturated(c28)
	d55 := c28.AddSaturated(c26)
	d56 := a12.AddSaturated(a10)
	d57 := a8.AddSaturated(a6)
	d58 := c24.AddSaturated(c22)
	d59 := c24.SubSaturated(c22)
	d60 := a8.SubSaturated(a6)
	d61 := a2.SubSaturated(a4)
	d62 := c18.SubSaturated(c20)
	d63 := c20.AddSaturated(c18)
	d64 := a4.AddSaturated(a2)
	e65 := itx16Rot(d35, d62, kn(8), k(56), s12)
	e66 := itx16Rot(d35, d62, k(56), k(8), s12)
	e67 := itx16Rot(d36, d61, kn(8), k(56), s12)
	e68 := itx16Rot(d36, d61, k(56), k(8), s12)
	e69 := itx16Rot(d37, d60, kn(56), kn(8), s12)
	e70 := itx16Rot(d37, d60, kn(8), k(56), s12)
	e71 := itx16Rot(d38, d59, kn(56), kn(8), s12)
	e72 := itx16Rot(d38, d59, kn(8), k(56), s12)
	e73 := itx16Rot(d43, d54, kn(40), k(24), s12)
	e74 := itx16Rot(d43, d54, k(24), k(40), s12)
	e75 := itx16Rot(d44, d53, kn(40), k(24), s12)
	e76 := itx16Rot(d44, d53, k(24), k(40), s12)
	e77 := itx16Rot(d45, d52, kn(24), kn(40), s12)
	e78 := itx16Rot(d45, d52, kn(40), k(24), s12)
	e79 := itx16Rot(d46, d51, kn(24), kn(40), s12)
	e80 := itx16Rot(d46, d51, kn(40), k(24), s12)
	f81 := d33.AddSaturated(d40)
	f82 := d33.SubSaturated(d40)
	f83 := d48.SubSaturated(d41)
	f84 := d41.AddSaturated(d48)
	f85 := d34.AddSaturated(d39)
	f86 := d34.SubSaturated(d39)
	f87 := d47.SubSaturated(d42)
	f88 := d42.AddSaturated(d47)
	f89 := e65.AddSaturated(e71)
	f90 := e65.SubSaturated(e71)
	f91 := e79.SubSaturated(e73)
	f92 := e73.AddSaturated(e79)
	f93 := e67.AddSaturated(e69)
	f94 := e67.SubSaturated(e69)
	f95 := e77.SubSaturated(e75)
	f96 := e75.AddSaturated(e77)
	f97 := d49.AddSaturated(d56)
	f98 := d49.SubSaturated(d56)
	f99 := d64.SubSaturated(d57)
	f100 := d57.AddSaturated(d64)
	f101 := d50.AddSaturated(d55)
	f102 := d50.SubSaturated(d55)
	f103 := d63.SubSaturated(d58)
	f104 := d58.AddSaturated(d63)
	f105 := e80.AddSaturated(e74)
	f106 := e80.SubSaturated(e74)
	f107 := e66.SubSaturated(e72)
	f108 := e72.AddSaturated(e66)
	f109 := e78.AddSaturated(e76)
	f110 := e78.SubSaturated(e76)
	f111 := e68.SubSaturated(e70)
	f112 := e70.AddSaturated(e68)
	g113 := itx16Rot(f94, f111, kn(16), k(48), s12)
	g114 := itx16Rot(f94, f111, k(48), k(16), s12)
	g115 := itx16Rot(f83, f98, kn(48), kn(16), s12)
	g116 := itx16Rot(f83, f98, kn(16), k(48), s12)
	g117 := itx16Rot(f90, f107, kn(16), k(48), s12)
	g118 := itx16Rot(f90, f107, k(48), k(16), s12)
	g119 := itx16Rot(f87, f102, kn(48), kn(16), s12)
	g120 := itx16Rot(f87, f102, kn(16), k(48), s12)
	g121 := itx16Rot(f86, f103, kn(16), k(48), s12)
	g122 := itx16Rot(f86, f103, k(48), k(16), s12)
	g123 := itx16Rot(f91, f106, kn(48), kn(16), s12)
	g124 := itx16Rot(f91, f106, kn(16), k(48), s12)
	g125 := itx16Rot(f82, f99, kn(16), k(48), s12)
	g126 := itx16Rot(f82, f99, k(48), k(16), s12)
	g127 := itx16Rot(f95, f110, kn(48), kn(16), s12)
	g128 := itx16Rot(f95, f110, kn(16), k(48), s12)
	h129 := f81.AddSaturated(f84)
	h130 := f81.SubSaturated(f84)
	h131 := f100.SubSaturated(f97)
	h132 := f97.AddSaturated(f100)
	h133 := f85.AddSaturated(f88)
	h134 := f85.SubSaturated(f88)
	h135 := f104.SubSaturated(f101)
	h136 := f101.AddSaturated(f104)
	h137 := f89.AddSaturated(f92)
	h138 := f89.SubSaturated(f92)
	h139 := f108.SubSaturated(f105)
	h140 := f105.AddSaturated(f108)
	h141 := f93.AddSaturated(f96)
	h142 := f93.SubSaturated(f96)
	h143 := f112.SubSaturated(f109)
	h144 := f109.AddSaturated(f112)
	h145 := g113.AddSaturated(g127)
	h146 := g113.SubSaturated(g127)
	h147 := g114.SubSaturated(g128)
	h148 := g128.AddSaturated(g114)
	h149 := g117.AddSaturated(g123)
	h150 := g117.SubSaturated(g123)
	h151 := g118.SubSaturated(g124)
	h152 := g124.AddSaturated(g118)
	h153 := g121.AddSaturated(g119)
	h154 := g121.SubSaturated(g119)
	h155 := g122.SubSaturated(g120)
	h156 := g120.AddSaturated(g122)
	h157 := g125.AddSaturated(g115)
	h158 := g125.SubSaturated(g115)
	h159 := g126.SubSaturated(g116)
	h160 := g116.AddSaturated(g126)
	i161 := itx16Rot(h158, h159, kn(32), k(32), s12)
	i162 := itx16Rot(h158, h159, k(32), k(32), s12)
	i163 := itx16Rot(h154, h155, kn(32), k(32), s12)
	i164 := itx16Rot(h154, h155, k(32), k(32), s12)
	i165 := itx16Rot(h150, h151, kn(32), k(32), s12)
	i166 := itx16Rot(h150, h151, k(32), k(32), s12)
	i167 := itx16Rot(h146, h147, kn(32), k(32), s12)
	i168 := itx16Rot(h146, h147, k(32), k(32), s12)
	i169 := itx16Rot(h142, h143, kn(32), k(32), s12)
	i170 := itx16Rot(h142, h143, k(32), k(32), s12)
	i171 := itx16Rot(h138, h139, kn(32), k(32), s12)
	i172 := itx16Rot(h138, h139, k(32), k(32), s12)
	i173 := itx16Rot(h134, h135, kn(32), k(32), s12)
	i174 := itx16Rot(h134, h135, k(32), k(32), s12)
	i175 := itx16Rot(h130, h131, kn(32), k(32), s12)
	i176 := itx16Rot(h130, h131, k(32), k(32), s12)
	o[0] = h129
	o[1] = h133
	o[2] = h137
	o[3] = h141
	o[4] = h145
	o[5] = h149
	o[6] = h153
	o[7] = h157
	o[8] = i161
	o[9] = i163
	o[10] = i165
	o[11] = i167
	o[12] = i169
	o[13] = i171
	o[14] = i173
	o[15] = i175
	o[16] = i176
	o[17] = i174
	o[18] = i172
	o[19] = i170
	o[20] = i168
	o[21] = i166
	o[22] = i164
	o[23] = i162
	o[24] = h160
	o[25] = h156
	o[26] = h152
	o[27] = h148
	o[28] = h144
	o[29] = h140
	o[30] = h136
	o[31] = h132
}

// itx64Odd8 is the odd half of inverseDCT64 on eight lanes for odd inputs
// below 8 (the rest zero): in[k] is coefficient k and o[j] is stage-10
// value 32+j. Generated from the stage structure of inverseDCT64 with the
// zero operands folded.
func itx64Odd8(in *[32]archsimd.Int16x8, o *[32]archsimd.Int16x8) {
	k := func(i int) archsimd.Int16x8 { return archsimd.LoadInt16x8Array(&itx64K[i]) }
	kn := func(i int) archsimd.Int16x8 { return archsimd.LoadInt16x8Array(&itx64K[64+i]) }
	s12 := archsimd.LoadInt32x4Array(&itx16Scale12)
	a1 := itx16Rot1(in[1], k(63), s12)
	a2 := itx16Rot1(in[1], k(1), s12)
	a3 := itx16Rot1(in[7], kn(57), s12)
	a4 := itx16Rot1(in[7], k(7), s12)
	a5 := itx16Rot1(in[5], k(59), s12)
	a6 := itx16Rot1(in[5], k(5), s12)
	a7 := itx16Rot1(in[3], kn(61), s12)
	a8 := itx16Rot1(in[3], k(3), s12)
	c9 := itx16Rot(a1, a2, kn(4), k(60), s12)
	c10 := itx16Rot(a1, a2, k(60), k(4), s12)
	c11 := itx16Rot(a3, a4, kn(28), kn(36), s12)
	c12 := itx16Rot(a3, a4, kn(36), k(28), s12)
	c13 := itx16Rot(a5, a6, kn(20), k(44), s12)
	c14 := itx16Rot(a5, a6, k(44), k(20), s12)
	c15 := itx16Rot(a7, a8, kn(12), kn(52), s12)
	c16 := itx16Rot(a7, a8, kn(52), k(12), s12)
	e17 := itx16Rot(c9, c10, kn(8), k(56), s12)
	e18 := itx16Rot(c9, c10, k(56), k(8), s12)
	e19 := itx16Rot(a1, a2, kn(8), k(56), s12)
	e20 := itx16Rot(a1, a2, k(56), k(8), s12)
	e21 := itx16Rot(a3, a4, kn(56), kn(8), s12)
	e22 := itx16Rot(a3, a4, kn(8), k(56), s12)
	e23 := itx16Rot(c11, c12, kn(56), kn(8), s12)
	e24 := itx16Rot(c11, c12, kn(8), k(56), s12)
	e25 := itx16Rot(c13, c14, kn(40), k(24), s12)
	e26 := itx16Rot(c13, c14, k(24), k(40), s12)
	e27 := itx16Rot(a5, a6, kn(40), k(24), s12)
	e28 := itx16Rot(a5, a6, k(24), k(40), s12)
	e29 := itx16Rot(a7, a8, kn(24), kn(40), s12)
	e30 := itx16Rot(a7, a8, kn(40), k(24), s12)
	e31 := itx16Rot(c15, c16, kn(24), kn(40), s12)
	e32 := itx16Rot(c15, c16, kn(40), k(24), s12)
	f33 := a1.AddSaturated(a3)
	f34 := a1.SubSaturated(a3)
	f35 := a7.SubSaturated(a5)
	f36 := a5.AddSaturated(a7)
	f37 := c9.AddSaturated(c11)
	f38 := c9.SubSaturated(c11)
	f39 := c15.SubSaturated(c13)
	f40 := c13.AddSaturated(c15)
	f41 := e17.AddSaturated(e23)
	f42 := e17.SubSaturated(e23)
	f43 := e31.SubSaturated(e25)
	f44 := e25.AddSaturated(e31)
	f45 := e19.AddSaturated(e21)
	f46 := e19.SubSaturated(e21)
	f47 := e29.SubSaturated(e27)
	f48 := e27.AddSaturated(e29)
	f49 := a8.AddSaturated(a6)
	f50 := a8.SubSaturated(a6)
	f51 := a2.SubSaturated(a4)
	f52 := a4.AddSaturated(a2)
	f53 := c16.AddSaturated(c14)
	f54 := c16.SubSaturated(c14)
	f55 := c10.SubSaturated(c12)
	f56 := c12.AddSaturated(c10)
	f57 := e32.AddSaturated(e26)
	f58 := e32.SubSaturated(e26)
	f59 := e18.SubSaturated(e24)
	f60 := e24.AddSaturated(e18)
	f61 := e30.AddSaturated(e28)
	f62 := e30.SubSaturated(e28)
	f63 := e20.SubSaturated(e22)
	f64 := e22.AddSaturated(e20)
	g65 := itx16Rot(f46, f63, kn(16), k(48), s12)
	g66 := itx16Rot(f46, f63, k(48), k(16), s12)
	g67 := itx16Rot(f35, f50, kn(48), kn(16), s12)
	g68 := itx16Rot(f35, f50, kn(16), k(48), s12)
	g69 := itx16Rot(f42, f59, kn(16), k(48), s12)
	g70 := itx16Rot(f42, f59, k(48), k(16), s12)
	g71 := itx16Rot(f39, f54, kn(48), kn(16), s12)
	g72 := itx16Rot(f39, f54, kn(16), k(48), s12)
	g73 := itx16Rot(f38, f55, kn(16), k(48), s12)
	g74 := itx16Rot(f38, f55, k(48), k(16), s12)
	g75 := itx16Rot(f43, f58, kn(48), kn(16), s12)
	g76 := itx16Rot(f43, f58, kn(16), k(48), s12)
	g77 := itx16Rot(f34, f51, kn(16), k(48), s12)
	g78 := itx16Rot(f34, f51, k(48), k(16), s12)
	g79 := itx16Rot(f47, f62, kn(48), kn(16), s12)
	g80 := itx16Rot(f47, f62, kn(16), k(48), s12)
	h81 := f33.AddSaturated(f36)
	h82 := f33.SubSaturated(f36)
	h83 := f52.SubSaturated(f49)
	h84 := f49.AddSaturated(f52)
	h85 := f37.AddSaturated(f40)
	h86 := f37.SubSaturated(f40)
	h87 := f56.SubSaturated(f53)
	h88 := f53.AddSaturated(f56)
	h89 := f41.AddSaturated(f44)
	h90 := f41.SubSaturated(f44)
	h91 := f60.SubSaturated(f57)
	h92 := f57.AddSaturated(f60)
	h93 := f45.AddSaturated(f48)
	h94 := f45.SubSaturated(f48)
	h95 := f64.SubSaturated(f61)
	h96 := f61.AddSaturated(f64)
	h97 := g65.AddSaturated(g79)
	h98 := g65.SubSaturated(g79)
	h99 := g66.SubSaturated(g80)
	h100 := g80.AddSaturated(g66)
	h101 := g69.AddSaturated(g75)
	h102 := g69.SubSaturated(g75)
	h103 := g70.SubSaturated(g76)
	h104 := g76.AddSaturated(g70)
	h105 := g73.AddSaturated(g71)
	h106 := g73.SubSaturated(g71)
	h107 := g74.SubSaturated(g72)
	h108 := g72.AddSaturated(g74)
	h109 := g77.AddSaturated(g67)
	h110 := g77.SubSaturated(g67)
	h111 := g78.SubSaturated(g68)
	h112 := g68.AddSaturated(g78)
	i113 := itx16Rot(h110, h111, kn(32), k(32), s12)
	i114 := itx16Rot(h110, h111, k(32), k(32), s12)
	i115 := itx16Rot(h106, h107, kn(32), k(32), s12)
	i116 := itx16Rot(h106, h107, k(32), k(32), s12)
	i117 := itx16Rot(h102, h103, kn(32), k(32), s12)
	i118 := itx16Rot(h102, h103, k(32), k(32), s12)
	i119 := itx16Rot(h98, h99, kn(32), k(32), s12)
	i120 := itx16Rot(h98, h99, k(32), k(32), s12)
	i121 := itx16Rot(h94, h95, kn(32), k(32), s12)
	i122 := itx16Rot(h94, h95, k(32), k(32), s12)
	i123 := itx16Rot(h90, h91, kn(32), k(32), s12)
	i124 := itx16Rot(h90, h91, k(32), k(32), s12)
	i125 := itx16Rot(h86, h87, kn(32), k(32), s12)
	i126 := itx16Rot(h86, h87, k(32), k(32), s12)
	i127 := itx16Rot(h82, h83, kn(32), k(32), s12)
	i128 := itx16Rot(h82, h83, k(32), k(32), s12)
	o[0] = h81
	o[1] = h85
	o[2] = h89
	o[3] = h93
	o[4] = h97
	o[5] = h101
	o[6] = h105
	o[7] = h109
	o[8] = i113
	o[9] = i115
	o[10] = i117
	o[11] = i119
	o[12] = i121
	o[13] = i123
	o[14] = i125
	o[15] = i127
	o[16] = i128
	o[17] = i126
	o[18] = i124
	o[19] = i122
	o[20] = i120
	o[21] = i118
	o[22] = i116
	o[23] = i114
	o[24] = h112
	o[25] = h108
	o[26] = h104
	o[27] = h100
	o[28] = h96
	o[29] = h92
	o[30] = h88
	o[31] = h84
}

// itx64Net is inverseDCT64 on eight lanes for n nonzero inputs (8, 16 or
// 32): in[k] is coefficient k, out[r] is output r.
func itx64Net(n int, in *[32]archsimd.Int16x8, out *[64]archsimd.Int16x8) {
	var ev [32]archsimd.Int16x8
	var o [32]archsimd.Int16x8
	switch {
	case n <= 8:
		var e [8]archsimd.Int16x8
		e[0], e[1], e[2], e[3] = in[0], in[2], in[4], in[6]
		itx32Low8Net(&e, &ev)
		itx64Odd8(in, &o)
	case n <= 16:
		var e [8]archsimd.Int16x8
		for k := 0; k < 8; k++ {
			e[k] = in[2*k]
		}
		itx32Low8Net(&e, &ev)
		itx64Odd16(in, &o)
	default:
		var e [16]archsimd.Int16x8
		for k := 0; k < 16; k++ {
			e[k] = in[2*k]
		}
		itx32Low16Net(&e, &ev)
		itx64Odd32(in, &o)
	}
	for i := 0; i < 32; i++ {
		out[i] = ev[i].AddSaturated(o[31-i])
		out[63-i] = ev[i].SubSaturated(o[31-i])
	}
}

// itx64Form rounds a coded extent up to the network form that covers it.
func itx64Form(n int) int {
	switch {
	case n <= 8:
		return 8
	case n <= 16:
		return 16
	default:
		return 32
	}
}

// itx64RowStrip runs the 64x64 row pass for coefficient rows r0..r0+7 with
// inCols coded coefficient columns (8, 16 or 32) into tmp rows r0..r0+7 and
// adds the stored values' magnitudes to the eight column guards. It reports
// false, with tmp untouched, when a row fails the L1 guard.
func itx64RowStrip(coeff []int32, stride int, r0 int, inCols int, tmp *[32 * 64]int16, colL1 *[8]archsimd.Uint16x8) bool {
	_ = coeff[(inCols-1)*stride+r0+7]
	var in [32]archsimd.Int16x8
	var l1 archsimd.Uint16x8
	for k := 0; k < inCols; k++ {
		x := itx16LoadNarrow(coeff[k*stride+r0 : k*stride+r0+8])
		l1 = l1.AddSaturated(x.Abs().ToBits())
		in[k] = x
	}
	if l1.ReduceMax() > itx16ColumnL1Bound {
		return false
	}
	var out [64]archsimd.Int16x8
	itx64Net(inCols, &in, &out)
	one := archsimd.BroadcastInt16x8(1)
	sh1 := archsimd.BroadcastInt16x8(-1)
	for g := 0; g < 8; g++ {
		var v [8]archsimd.Int16x8
		for j := 0; j < 8; j++ {
			v[j] = itx32Round2(out[8*g+j], sh1, sh1, one)
		}
		itx16Transpose8x8(&v)
		gl1 := colL1[g]
		for r := 0; r < 8; r++ {
			v[r].StoreArray((*[8]int16)(tmp[(r0+r)*64+8*g:]))
			gl1 = gl1.AddSaturated(v[r].Abs().ToBits())
		}
		colL1[g] = gl1
	}
	return true
}

// InverseAddDCT64x64 reconstructs one 8-bit 64x64 DCT_DCT block whose coded
// coefficients lie in rows 0..activeRows-1 and columns 0..activeCols-1 of
// the 32x32 corner: dst = clip(pred + inverse(coeff)), with the dequantized
// coefficients column-major at coeffStride (the 32x32 scan). pred may be
// dst. It reports false, with dst untouched, when the block is outside the
// kernel's contract or a guard declines; the caller then runs the staged
// path, which produces the same bytes.
func InverseAddDCT64x64(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, activeRows, activeCols int) bool {
	if activeRows <= 0 || activeRows > 32 || activeCols <= 0 || activeCols > 32 ||
		coeffStride != 32 || len(coeff) < 1024 || dstStride < 64 || predStride < 64 ||
		len(dst) < 63*dstStride+64 || len(pred) < 63*predStride+64 {
		return false
	}
	return itxGenDCT64x64((*[1024]int32)(coeff[:1024]), dst, dstStride, pred, predStride, activeCols, activeRows)
}

// itx_fused8: dav1d's pass structure (itx_fused32) for the 8-bit 8x8
// DCT_DCT block: one eight-row strip straight from the column-major
// coefficients with the DCT8 network across vectors, the inter-pass
// Round2(., 1) in registers, one 8x8 transpose, then the column network and
// the fused Round2(., 4) + add + saturate. The whole block stays in
// registers; nothing is stored between the passes.

// itx8Net is inverseDCT8 on eight lanes: in[k] is coefficient k, out[r]
// output r (the even half of inverseDCT16Col8SIMD16v2 with inputs 0..7).
func itx8Net(in *[8]archsimd.Int16x8, out *[8]archsimd.Int16x8) {
	k := func(i int) archsimd.Int16x8 { return archsimd.LoadInt16x8Array(&itx16K[i]) }
	s12 := archsimd.LoadInt32x4Array(&itx16Scale12)
	s11 := archsimd.LoadInt32x4Array(&itx16Scale11)
	p2896, m2896 := k(itx16Kp2896), k(itx16Km2896)
	p1567, p3784, m3784 := k(itx16Kp1567), k(itx16Kp3784), k(itx16Km3784)

	in0, in2, in4, in6 := in[0], in[2], in[4], in[6]
	t0 := itx16Rot(in0, in4, p2896, p2896, s12)
	t1 := itx16Rot(in0, in4, p2896, m2896, s12)
	t2 := itx16Rot(in2, in6, p1567, m3784, s12)
	t3 := itx16Rot(in2, in6, p3784, p1567, s12)
	d0 := t0.AddSaturated(t3)
	d1 := t1.AddSaturated(t2)
	d2 := t1.SubSaturated(t2)
	d3 := t0.SubSaturated(t3)

	in1, in3, in5, in7 := in[1], in[3], in[5], in[7]
	t4a := itx16Rot(in1, in7, k(itx16Kp799), k(itx16Km4017), s12)
	t5a := itx16Rot(in5, in3, k(itx16Kp1703), k(itx16Km1138), s11)
	t6a := itx16Rot(in5, in3, k(itx16Kp1138), k(itx16Kp1703), s11)
	t7a := itx16Rot(in1, in7, k(itx16Kp4017), k(itx16Kp799), s12)
	t4 := t4a.AddSaturated(t5a)
	t5a = t4a.SubSaturated(t5a)
	t7 := t7a.AddSaturated(t6a)
	t6a = t7a.SubSaturated(t6a)
	t5 := itx16Rot(t6a, t5a, p2896, m2896, s12)
	t6 := itx16Rot(t6a, t5a, p2896, p2896, s12)
	out[0], out[7] = d0.AddSaturated(t7), d0.SubSaturated(t7)
	out[1], out[6] = d1.AddSaturated(t6), d1.SubSaturated(t6)
	out[2], out[5] = d2.AddSaturated(t5), d2.SubSaturated(t5)
	out[3], out[4] = d3.AddSaturated(t4), d3.SubSaturated(t4)
}

// InverseAddDCT8x8 reconstructs one 8-bit 8x8 DCT_DCT block with coded
// coefficients in rows 0..activeRows-1 and columns 0..activeCols-1:
// dst = clip(pred + inverse(coeff)), with the dequantized coefficients
// column-major at coeffStride (8). pred may be dst. It reports false, with
// dst untouched, when the block is outside the kernel's contract or a guard
// declines; the caller then runs the staged path, which produces the same
// bytes. The kernels are generated (itx_fused_gen_arm64.go) per network
// form of the coded columns and rows.
func InverseAddDCT8x8(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, activeRows, activeCols int) bool {
	if activeRows <= 0 || activeRows > 8 || activeCols <= 0 || activeCols > 8 ||
		coeffStride != 8 || len(coeff) < 64 || dstStride < 8 || predStride < 8 ||
		len(dst) < 7*dstStride+8 || len(pred) < 7*predStride+8 {
		return false
	}
	// The kernel switch is written here rather than in a dispatch function:
	// one call level less for the most frequent block.
	c := (*[64]int32)(coeff[:64])
	switch {
	case activeCols <= 4 && activeRows <= 4:
		return itxGenDCT8x8C4R4(c, dst, dstStride, pred, predStride)
	case activeCols <= 4:
		return itxGenDCT8x8C4R8(c, dst, dstStride, pred, predStride)
	case activeRows <= 4:
		return itxGenDCT8x8C8R4(c, dst, dstStride, pred, predStride)
	default:
		return itxGenDCT8x8C8R8(c, dst, dstStride, pred, predStride)
	}
}

// itx8NetLow4 is inverseDCT8 on eight lanes for inputs whose indices 4..7
// are zero: in[k] is coefficient k, out[r] output r. Generated from the
// network of itx8Net with the zero operands folded.
func itx8NetLow4(in *[8]archsimd.Int16x8, out *[8]archsimd.Int16x8) {
	k := func(i int) archsimd.Int16x8 { return archsimd.LoadInt16x8Array(&itx64K[i]) }
	kn := func(i int) archsimd.Int16x8 { return archsimd.LoadInt16x8Array(&itx64K[64+i]) }
	s12 := archsimd.LoadInt32x4Array(&itx16Scale12)
	a1 := itx16Rot1(in[0], k(32), s12)
	a2 := itx16Rot1(in[2], k(48), s12)
	a3 := itx16Rot1(in[2], k(16), s12)
	b4 := a1.AddSaturated(a3)
	b5 := a1.AddSaturated(a2)
	b6 := a1.SubSaturated(a2)
	b7 := a1.SubSaturated(a3)
	c8 := itx16Rot1(in[1], k(56), s12)
	c9 := itx16Rot1(in[3], kn(40), s12)
	c10 := itx16Rot1(in[3], k(24), s12)
	c11 := itx16Rot1(in[1], k(8), s12)
	d12 := c8.AddSaturated(c9)
	d13 := c8.SubSaturated(c9)
	d14 := c11.AddSaturated(c10)
	d15 := c11.SubSaturated(c10)
	e16 := itx16Rot(d15, d13, k(32), kn(32), s12)
	e17 := itx16Rot(d15, d13, k(32), k(32), s12)
	f18 := b4.AddSaturated(d14)
	f19 := b5.AddSaturated(e17)
	f20 := b6.AddSaturated(e16)
	f21 := b7.AddSaturated(d12)
	f22 := b7.SubSaturated(d12)
	f23 := b6.SubSaturated(e16)
	f24 := b5.SubSaturated(e17)
	f25 := b4.SubSaturated(d14)
	out[0] = f18
	out[1] = f19
	out[2] = f20
	out[3] = f21
	out[4] = f22
	out[5] = f23
	out[6] = f24
	out[7] = f25
}

// 10- and 12-bit forms of the fused two-pass DCT_DCT inverse transform plus
// add. dav1d's 16bpc NEON (src/arm/64/itx16.S) runs the row pass in 32-bit
// lanes and then reuses the 8bpc int16 column functions
// (inv_dct_8h_x16_neon, inv_dct32_odd_8h_x16_neon) for 10 bpc, because the
// column input is clamped to max(bd + 6, 16) = 16 bits. Here both passes are
// the generated 8-bit int16 kernels (itx_fused_gen_arm64.go, itxGenHBD*):
// the 8-bit row strips under their per-lane L1 guard, and column passes
// whose final stage adds the 16bpc prediction row:
//
//   - a coefficient row whose L1 norm fits the guard (itx16ColumnL1Bound)
//     has no coefficient outside int16 (the narrowing load saturates any that
//     is, which then fails the guard) and, by the certified network gain, no
//     butterfly value outside int16. The staged reference clamps the row input
//     to bd + 8 bits and the column input to max(bd + 6, 16) bits, so under
//     the guard none of its clamps bite at 10 or 12 bits either, and both
//     compute the unclamped transform;
//   - the column pass is the same int16 network with its L1 guard (the column
//     bound is 16 bits at 8 and 10 bits and wider at 12), and the final stage
//     rounds by 4 and adds the 16bpc prediction row with a saturating int16
//     add clipped to [0, 2^bd - 1], which equals the scalar
//     clip(pred + residual) for any int16 residual and pred <= 4095.
//
// A declined block is left untouched; the caller runs the staged path, which
// produces the same samples.

// itxAddRow16 stores clip(pred + res, 0, maxV) for the eight 16bpc samples
// at pred[po:] into dst[do:].
func itxAddRow16(dst []byte, do int, pred []byte, po int, res, zero, maxV archsimd.Int16x8) {
	d, p := (*[16]byte)(dst[do:do+16]), (*[16]byte)(pred[po:po+16])
	archsimd.LoadUint8x16Array(p).ReshapeToUint16s().BitsToInt16().AddSaturated(res).Max(zero).Min(maxV).ToBits().ReshapeToUint8s().StoreArray(d)
}

// itxHBDMax returns the broadcast pixel maximum for a 10- or 12-bit block,
// false for any other depth.
func itxHBDMax(bitDepth uint8) (archsimd.Int16x8, bool) {
	if bitDepth != 10 && bitDepth != 12 {
		return archsimd.Int16x8{}, false
	}
	return archsimd.BroadcastInt16x8(int16(1)<<bitDepth - 1), true
}

// itxHBDFits reports whether an n x n block of 2-byte samples at stride
// (bytes, at most itxGenMaxStride) fits buf.
func itxHBDFits(buf []byte, stride, n int) bool {
	return stride >= 2*n && stride <= itxGenMaxStride && len(buf) >= (n-1)*stride+2*n
}

// itxHBDSquareOK checks the contract of the n x n 10/12-bit kernels (coded
// extent, the column-major coefficient stride cs and the 2-byte sample
// rows of dst and pred) and returns the pixel maximum.
func itxHBDSquareOK(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, activeRows, activeCols int, bitDepth uint8, n, cs int) (int16, bool) {
	if (bitDepth != 10 && bitDepth != 12) || activeRows <= 0 || activeRows > cs || activeCols <= 0 || activeCols > cs ||
		coeffStride != cs || len(coeff) < cs*cs || !itxHBDFits(dst, dstStride, n) || !itxHBDFits(pred, predStride, n) {
		return 0, false
	}
	return int16(1)<<bitDepth - 1, true
}

// InverseAddDCT8x8HBD is InverseAddDCT8x8 for a 10- or 12-bit block (2-byte
// samples, strides in bytes).
func InverseAddDCT8x8HBD(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, activeRows, activeCols int, bitDepth uint8) bool {
	mx, ok := itxHBDSquareOK(dst, dstStride, pred, predStride, coeff, coeffStride, activeRows, activeCols, bitDepth, 8, 8)
	if !ok {
		return false
	}
	return itxGenHBDDCT8x8((*[64]int32)(coeff[:64]), dst, dstStride, pred, predStride, activeCols, activeRows, mx)
}

// InverseAddDCT16x16HBD is InverseAddDCT16x16 for a 10- or 12-bit block.
func InverseAddDCT16x16HBD(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, activeRows, activeCols int, bitDepth uint8) bool {
	mx, ok := itxHBDSquareOK(dst, dstStride, pred, predStride, coeff, coeffStride, activeRows, activeCols, bitDepth, 16, 16)
	if !ok {
		return false
	}
	return itxGenHBDDCT16x16((*[256]int32)(coeff[:256]), dst, dstStride, pred, predStride, activeCols, activeRows, mx)
}

// InverseAddDCT32x32HBD is the 10- or 12-bit 32x32 DCT_DCT block with coded
// coefficients in rows 0..activeRows-1 and columns 0..activeCols-1.
func InverseAddDCT32x32HBD(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, activeRows, activeCols int, bitDepth uint8) bool {
	mx, ok := itxHBDSquareOK(dst, dstStride, pred, predStride, coeff, coeffStride, activeRows, activeCols, bitDepth, 32, 32)
	if !ok {
		return false
	}
	return itxGenHBDDCT32x32((*[1024]int32)(coeff[:1024]), dst, dstStride, pred, predStride, activeCols, activeRows, mx)
}

// InverseAddDCT64x64HBD is InverseAddDCT64x64 for a 10- or 12-bit block
// (2-byte samples, strides in bytes): coded coefficients in the 32x32 corner
// (dav1d's itx16 runs the 64-point column pass of 10 bpc blocks with its
// 8bpc int16 functions too).
func InverseAddDCT64x64HBD(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, activeRows, activeCols int, bitDepth uint8) bool {
	mx, ok := itxHBDSquareOK(dst, dstStride, pred, predStride, coeff, coeffStride, activeRows, activeCols, bitDepth, 64, 32)
	if !ok {
		return false
	}
	return itxGenHBDDCT64x64((*[1024]int32)(coeff[:1024]), dst, dstStride, pred, predStride, activeCols, activeRows, mx)
}

// 10- and 12-bit forms of the fused 16x16 and 8x8 DCT/ADST/flipped-ADST
// pairings (itx_fused_hybrid). As for the DCT_DCT blocks
// (itx_fused_hbd_gosimd_arm64.go), dav1d's itx16 runs the 10 bpc column pass
// with its 8bpc int16 functions (inv_adst_8h_x16_neon, inv_flipadst_8h_x16_neon,
// ...), and the int16 row strips and column networks here are the 8-bit ones
// under the same per-lane L1 guards, which itx_hybrid_gain_test.go certifies
// for the ADST networks too: under them no coefficient or butterfly leaves
// int16, so the staged path's bd + 8 row and max(bd + 6, 16) column clamps
// never bite, and the column output is added to the 16bpc prediction with
// the saturating add clipped to the bit depth. A declined block is left
// untouched for the staged path.

// itx16ColStripAddTHBD runs the 16x16 column network t for columns
// c0..c0+7 from tmp rows 0..inRows-1 and adds the result onto 16bpc samples.
func itx16ColStripAddTHBD(tmp *[16 * 16]int16, c0 int, inRows int, t tx1DType, dst []byte, dstStride int, pred []byte, predStride int, maxV archsimd.Int16x8) {
	dst, pred = dst[2*c0:], pred[2*c0:]
	// itxHBDFits checked the strides against itxGenMaxStride; the conversion
	// carries that range, so each row window is one bounds check.
	ds, ps := int(uint32(dstStride)), int(uint32(predStride))
	var in [16]archsimd.Int16x8
	for r := 0; r < inRows; r++ {
		in[r] = archsimd.LoadInt16x8Array((*[8]int16)(tmp[r*16+c0:]))
	}
	var out [16]archsimd.Int16x8
	itx16Net1D(t, inRows, &in, &out)
	one := archsimd.BroadcastInt16x8(1)
	sh1 := archsimd.BroadcastInt16x8(-1)
	sh3 := archsimd.BroadcastInt16x8(-3)
	zero := archsimd.BroadcastInt16x8(0)
	for r := 0; r < 16; r++ {
		itxAddRow16(dst, r*ds, pred, r*ps, itx32Round2(out[r], sh3, sh1, one), zero, maxV)
	}
}

// InverseAdd16x16HybridHBD is InverseAdd16x16Hybrid for a 10- or 12-bit
// block (2-byte samples, strides in bytes).
func InverseAdd16x16HybridHBD(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, txType Type, activeRows, activeCols int, bitDepth uint8) bool {
	vertical, horizontal, ok := itxHybridTypes(txType)
	if !ok {
		return false
	}
	maxV, ok := itxHBDMax(bitDepth)
	if !ok || activeRows <= 0 || activeRows > 16 || activeCols <= 0 || activeCols > 16 ||
		coeffStride < 16 || len(coeff) < 15*coeffStride+16 || !itxHBDFits(dst, dstStride, 16) || !itxHBDFits(pred, predStride, 16) {
		return false
	}
	inCols := itx16Form(activeCols)
	inRows := itx16Form(activeRows)
	var tmp [16 * 16]int16
	var colL1 [2]archsimd.Uint16x8
	if !itx16RowStripT(coeff, coeffStride, 0, inCols, horizontal, &tmp, &colL1) {
		return false
	}
	if activeRows > 8 && !itx16RowStripT(coeff, coeffStride, 8, inCols, horizontal, &tmp, &colL1) {
		return false
	}
	if colL1[0].Max(colL1[1]).ReduceMax() > itx16ColumnL1Bound {
		return false
	}
	itx16ColStripAddTHBD(&tmp, 0, inRows, vertical, dst, dstStride, pred, predStride, maxV)
	itx16ColStripAddTHBD(&tmp, 8, inRows, vertical, dst, dstStride, pred, predStride, maxV)
	return true
}

// InverseAdd8x8HybridHBD is InverseAdd8x8Hybrid for a 10- or 12-bit block.
func InverseAdd8x8HybridHBD(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, txType Type, activeRows, activeCols int, bitDepth uint8) bool {
	vertical, horizontal, ok := itxHybridTypes(txType)
	if !ok {
		return false
	}
	maxV, ok := itxHBDMax(bitDepth)
	if !ok || activeRows <= 0 || activeRows > 8 || activeCols <= 0 || activeCols > 8 ||
		coeffStride < 8 || len(coeff) < 7*coeffStride+8 || !itxHBDFits(dst, dstStride, 8) || !itxHBDFits(pred, predStride, 8) {
		return false
	}
	var in [8]archsimd.Int16x8
	var l1 archsimd.Uint16x8
	for k := 0; k < activeCols; k++ {
		x := itx16LoadNarrow(coeff[k*coeffStride : k*coeffStride+8])
		l1 = l1.AddSaturated(x.Abs().ToBits())
		in[k] = x
	}
	if l1.ReduceMax() > itx16ColumnL1Bound {
		return false
	}
	var mid [8]archsimd.Int16x8
	itx8Net1D(horizontal, activeCols, &in, &mid)
	one := archsimd.BroadcastInt16x8(1)
	sh1 := archsimd.BroadcastInt16x8(-1)
	var colL1 archsimd.Uint16x8
	for j := range mid {
		mid[j] = mid[j].Shift(sh1).Add(mid[j].And(one))
	}
	itx16Transpose8x8(&mid)
	for r := range mid {
		colL1 = colL1.AddSaturated(mid[r].Abs().ToBits())
	}
	if colL1.ReduceMax() > itx16ColumnL1Bound {
		return false
	}
	var out [8]archsimd.Int16x8
	itx8Net1D(vertical, activeRows, &mid, &out)
	sh3 := archsimd.BroadcastInt16x8(-3)
	zero := archsimd.BroadcastInt16x8(0)
	// itxHBDFits checked the strides against itxGenMaxStride; the conversion
	// carries that range, so each row window is one bounds check.
	ds, ps := int(uint32(dstStride)), int(uint32(predStride))
	for r := 0; r < 8; r++ {
		itxAddRow16(dst, r*ds, pred, r*ps, itx32Round2(out[r], sh3, sh1, one), zero, maxV)
	}
	return true
}

// itx_fused_hybrid: the fused two-pass structure of itx_fused16 and
// itx_fused8 for the 8-bit 16x16 and 8x8 blocks whose transform pairs the
// DCT with the ADST or the flipped ADST in either direction (the intra
// types). The strips are those of the DCT kernels with the one-dimensional
// network chosen per pass; a flipped ADST reverses its outputs before the
// transpose (rows) or before the add (columns), as inverseFlipADST1D does.
// The ADST networks below are generated from the stage structure of
// inverseADST16 and inverseADST8To with the zero operands of each form
// folded; itx_hybrid_gain_test.go certifies the shared per-lane L1 guard
// for them.

// itx16ADSTNet16 is inverseADST16 on eight lanes for inputs below 16 (the rest
// zero): in[k] is coefficient k, out[r] output r. Generated from the stage
// structure of inverseADST16 with the zero operands folded.
func itx16ADSTNet16(in *[16]archsimd.Int16x8, out *[16]archsimd.Int16x8) {
	k := func(i int) archsimd.Int16x8 { return archsimd.LoadInt16x8Array(&itx64K[i]) }
	kn := func(i int) archsimd.Int16x8 { return archsimd.LoadInt16x8Array(&itx64K[64+i]) }
	s12 := archsimd.LoadInt32x4Array(&itx16Scale12)
	var zero archsimd.Int16x8
	a1 := itx16Rot(in[15], in[0], k(2), k(62), s12)
	a2 := itx16Rot(in[15], in[0], k(62), kn(2), s12)
	a3 := itx16Rot(in[13], in[2], k(10), k(54), s12)
	a4 := itx16Rot(in[13], in[2], k(54), kn(10), s12)
	a5 := itx16Rot(in[11], in[4], k(18), k(46), s12)
	a6 := itx16Rot(in[11], in[4], k(46), kn(18), s12)
	a7 := itx16Rot(in[9], in[6], k(26), k(38), s12)
	a8 := itx16Rot(in[9], in[6], k(38), kn(26), s12)
	a9 := itx16Rot(in[7], in[8], k(34), k(30), s12)
	a10 := itx16Rot(in[7], in[8], k(30), kn(34), s12)
	a11 := itx16Rot(in[5], in[10], k(42), k(22), s12)
	a12 := itx16Rot(in[5], in[10], k(22), kn(42), s12)
	a13 := itx16Rot(in[3], in[12], k(50), k(14), s12)
	a14 := itx16Rot(in[3], in[12], k(14), kn(50), s12)
	a15 := itx16Rot(in[1], in[14], k(58), k(6), s12)
	a16 := itx16Rot(in[1], in[14], k(6), kn(58), s12)
	b17 := a1.AddSaturated(a9)
	b18 := a1.SubSaturated(a9)
	b19 := a2.AddSaturated(a10)
	b20 := a2.SubSaturated(a10)
	b21 := a3.AddSaturated(a11)
	b22 := a3.SubSaturated(a11)
	b23 := a4.AddSaturated(a12)
	b24 := a4.SubSaturated(a12)
	b25 := a5.AddSaturated(a13)
	b26 := a5.SubSaturated(a13)
	b27 := a6.AddSaturated(a14)
	b28 := a6.SubSaturated(a14)
	b29 := a7.AddSaturated(a15)
	b30 := a7.SubSaturated(a15)
	b31 := a8.AddSaturated(a16)
	b32 := a8.SubSaturated(a16)
	c33 := itx16Rot(b18, b20, k(8), k(56), s12)
	c34 := itx16Rot(b18, b20, k(56), kn(8), s12)
	c35 := itx16Rot(b22, b24, k(40), k(24), s12)
	c36 := itx16Rot(b22, b24, k(24), kn(40), s12)
	c37 := itx16Rot(b28, b26, k(8), kn(56), s12)
	c38 := itx16Rot(b28, b26, k(56), k(8), s12)
	c39 := itx16Rot(b32, b30, k(40), kn(24), s12)
	c40 := itx16Rot(b32, b30, k(24), k(40), s12)
	d41 := b17.AddSaturated(b25)
	d42 := b17.SubSaturated(b25)
	d43 := b19.AddSaturated(b27)
	d44 := b19.SubSaturated(b27)
	d45 := b21.AddSaturated(b29)
	d46 := b21.SubSaturated(b29)
	d47 := b23.AddSaturated(b31)
	d48 := b23.SubSaturated(b31)
	d49 := c33.AddSaturated(c37)
	d50 := c33.SubSaturated(c37)
	d51 := c34.AddSaturated(c38)
	d52 := c34.SubSaturated(c38)
	d53 := c35.AddSaturated(c39)
	d54 := c35.SubSaturated(c39)
	d55 := c36.AddSaturated(c40)
	d56 := c36.SubSaturated(c40)
	e57 := itx16Rot(d42, d44, k(16), k(48), s12)
	e58 := itx16Rot(d42, d44, k(48), kn(16), s12)
	e59 := itx16Rot(d48, d46, k(16), kn(48), s12)
	e60 := itx16Rot(d48, d46, k(48), k(16), s12)
	e61 := itx16Rot(d50, d52, k(16), k(48), s12)
	e62 := itx16Rot(d50, d52, k(48), kn(16), s12)
	e63 := itx16Rot(d56, d54, k(16), kn(48), s12)
	e64 := itx16Rot(d56, d54, k(48), k(16), s12)
	f65 := d41.AddSaturated(d45)
	f66 := d43.AddSaturated(d47)
	f67 := zero.SubSaturated(f66)
	f68 := d41.SubSaturated(d45)
	f69 := d43.SubSaturated(d47)
	f70 := e57.AddSaturated(e59)
	f71 := zero.SubSaturated(f70)
	f72 := e58.AddSaturated(e60)
	f73 := e57.SubSaturated(e59)
	f74 := e58.SubSaturated(e60)
	f75 := d49.AddSaturated(d53)
	f76 := zero.SubSaturated(f75)
	f77 := d51.AddSaturated(d55)
	f78 := d49.SubSaturated(d53)
	f79 := d51.SubSaturated(d55)
	f80 := e61.AddSaturated(e63)
	f81 := e62.AddSaturated(e64)
	f82 := zero.SubSaturated(f81)
	f83 := e61.SubSaturated(e63)
	f84 := e62.SubSaturated(e64)
	g85 := itx16Rot(f68, f69, k(32), k(32), s12)
	g86 := zero.SubSaturated(g85)
	g87 := itx16Rot(f68, f69, k(32), kn(32), s12)
	g88 := itx16Rot(f73, f74, k(32), k(32), s12)
	g89 := itx16Rot(f73, f74, k(32), kn(32), s12)
	g90 := zero.SubSaturated(g89)
	g91 := itx16Rot(f78, f79, k(32), k(32), s12)
	g92 := itx16Rot(f78, f79, k(32), kn(32), s12)
	g93 := zero.SubSaturated(g92)
	g94 := itx16Rot(f83, f84, k(32), k(32), s12)
	g95 := zero.SubSaturated(g94)
	g96 := itx16Rot(f83, f84, k(32), kn(32), s12)
	out[0] = f65
	out[1] = f76
	out[2] = f80
	out[3] = f71
	out[4] = g88
	out[5] = g95
	out[6] = g91
	out[7] = g86
	out[8] = g87
	out[9] = g93
	out[10] = g96
	out[11] = g90
	out[12] = f72
	out[13] = f82
	out[14] = f77
	out[15] = f67
}

// itx16ADSTNet8 is inverseADST16 on eight lanes for inputs below 8 (the rest
// zero): in[k] is coefficient k, out[r] output r. Generated from the stage
// structure of inverseADST16 with the zero operands folded.
func itx16ADSTNet8(in *[16]archsimd.Int16x8, out *[16]archsimd.Int16x8) {
	k := func(i int) archsimd.Int16x8 { return archsimd.LoadInt16x8Array(&itx64K[i]) }
	kn := func(i int) archsimd.Int16x8 { return archsimd.LoadInt16x8Array(&itx64K[64+i]) }
	s12 := archsimd.LoadInt32x4Array(&itx16Scale12)
	var zero archsimd.Int16x8
	a1 := itx16Rot1(in[0], k(62), s12)
	a2 := itx16Rot1(in[0], kn(2), s12)
	a3 := itx16Rot1(in[2], k(54), s12)
	a4 := itx16Rot1(in[2], kn(10), s12)
	a5 := itx16Rot1(in[4], k(46), s12)
	a6 := itx16Rot1(in[4], kn(18), s12)
	a7 := itx16Rot1(in[6], k(38), s12)
	a8 := itx16Rot1(in[6], kn(26), s12)
	a9 := itx16Rot1(in[7], k(34), s12)
	a10 := itx16Rot1(in[7], k(30), s12)
	a11 := itx16Rot1(in[5], k(42), s12)
	a12 := itx16Rot1(in[5], k(22), s12)
	a13 := itx16Rot1(in[3], k(50), s12)
	a14 := itx16Rot1(in[3], k(14), s12)
	a15 := itx16Rot1(in[1], k(58), s12)
	a16 := itx16Rot1(in[1], k(6), s12)
	b17 := a1.AddSaturated(a9)
	b18 := a1.SubSaturated(a9)
	b19 := a2.AddSaturated(a10)
	b20 := a2.SubSaturated(a10)
	b21 := a3.AddSaturated(a11)
	b22 := a3.SubSaturated(a11)
	b23 := a4.AddSaturated(a12)
	b24 := a4.SubSaturated(a12)
	b25 := a5.AddSaturated(a13)
	b26 := a5.SubSaturated(a13)
	b27 := a6.AddSaturated(a14)
	b28 := a6.SubSaturated(a14)
	b29 := a7.AddSaturated(a15)
	b30 := a7.SubSaturated(a15)
	b31 := a8.AddSaturated(a16)
	b32 := a8.SubSaturated(a16)
	c33 := itx16Rot(b18, b20, k(8), k(56), s12)
	c34 := itx16Rot(b18, b20, k(56), kn(8), s12)
	c35 := itx16Rot(b22, b24, k(40), k(24), s12)
	c36 := itx16Rot(b22, b24, k(24), kn(40), s12)
	c37 := itx16Rot(b28, b26, k(8), kn(56), s12)
	c38 := itx16Rot(b28, b26, k(56), k(8), s12)
	c39 := itx16Rot(b32, b30, k(40), kn(24), s12)
	c40 := itx16Rot(b32, b30, k(24), k(40), s12)
	d41 := b17.AddSaturated(b25)
	d42 := b17.SubSaturated(b25)
	d43 := b19.AddSaturated(b27)
	d44 := b19.SubSaturated(b27)
	d45 := b21.AddSaturated(b29)
	d46 := b21.SubSaturated(b29)
	d47 := b23.AddSaturated(b31)
	d48 := b23.SubSaturated(b31)
	d49 := c33.AddSaturated(c37)
	d50 := c33.SubSaturated(c37)
	d51 := c34.AddSaturated(c38)
	d52 := c34.SubSaturated(c38)
	d53 := c35.AddSaturated(c39)
	d54 := c35.SubSaturated(c39)
	d55 := c36.AddSaturated(c40)
	d56 := c36.SubSaturated(c40)
	e57 := itx16Rot(d42, d44, k(16), k(48), s12)
	e58 := itx16Rot(d42, d44, k(48), kn(16), s12)
	e59 := itx16Rot(d48, d46, k(16), kn(48), s12)
	e60 := itx16Rot(d48, d46, k(48), k(16), s12)
	e61 := itx16Rot(d50, d52, k(16), k(48), s12)
	e62 := itx16Rot(d50, d52, k(48), kn(16), s12)
	e63 := itx16Rot(d56, d54, k(16), kn(48), s12)
	e64 := itx16Rot(d56, d54, k(48), k(16), s12)
	f65 := d41.AddSaturated(d45)
	f66 := d43.AddSaturated(d47)
	f67 := zero.SubSaturated(f66)
	f68 := d41.SubSaturated(d45)
	f69 := d43.SubSaturated(d47)
	f70 := e57.AddSaturated(e59)
	f71 := zero.SubSaturated(f70)
	f72 := e58.AddSaturated(e60)
	f73 := e57.SubSaturated(e59)
	f74 := e58.SubSaturated(e60)
	f75 := d49.AddSaturated(d53)
	f76 := zero.SubSaturated(f75)
	f77 := d51.AddSaturated(d55)
	f78 := d49.SubSaturated(d53)
	f79 := d51.SubSaturated(d55)
	f80 := e61.AddSaturated(e63)
	f81 := e62.AddSaturated(e64)
	f82 := zero.SubSaturated(f81)
	f83 := e61.SubSaturated(e63)
	f84 := e62.SubSaturated(e64)
	g85 := itx16Rot(f68, f69, k(32), k(32), s12)
	g86 := zero.SubSaturated(g85)
	g87 := itx16Rot(f68, f69, k(32), kn(32), s12)
	g88 := itx16Rot(f73, f74, k(32), k(32), s12)
	g89 := itx16Rot(f73, f74, k(32), kn(32), s12)
	g90 := zero.SubSaturated(g89)
	g91 := itx16Rot(f78, f79, k(32), k(32), s12)
	g92 := itx16Rot(f78, f79, k(32), kn(32), s12)
	g93 := zero.SubSaturated(g92)
	g94 := itx16Rot(f83, f84, k(32), k(32), s12)
	g95 := zero.SubSaturated(g94)
	g96 := itx16Rot(f83, f84, k(32), kn(32), s12)
	out[0] = f65
	out[1] = f76
	out[2] = f80
	out[3] = f71
	out[4] = g88
	out[5] = g95
	out[6] = g91
	out[7] = g86
	out[8] = g87
	out[9] = g93
	out[10] = g96
	out[11] = g90
	out[12] = f72
	out[13] = f82
	out[14] = f77
	out[15] = f67
}

// itx16ADSTNet4 is inverseADST16 on eight lanes for inputs below 4 (the rest
// zero): in[k] is coefficient k, out[r] output r. Generated from the stage
// structure of inverseADST16 with the zero operands folded.
func itx16ADSTNet4(in *[16]archsimd.Int16x8, out *[16]archsimd.Int16x8) {
	k := func(i int) archsimd.Int16x8 { return archsimd.LoadInt16x8Array(&itx64K[i]) }
	kn := func(i int) archsimd.Int16x8 { return archsimd.LoadInt16x8Array(&itx64K[64+i]) }
	s12 := archsimd.LoadInt32x4Array(&itx16Scale12)
	var zero archsimd.Int16x8
	a1 := itx16Rot1(in[0], k(62), s12)
	a2 := itx16Rot1(in[0], kn(2), s12)
	a3 := itx16Rot1(in[2], k(54), s12)
	a4 := itx16Rot1(in[2], kn(10), s12)
	a5 := itx16Rot1(in[3], k(50), s12)
	a6 := itx16Rot1(in[3], k(14), s12)
	a7 := itx16Rot1(in[1], k(58), s12)
	a8 := itx16Rot1(in[1], k(6), s12)
	b9 := zero.SubSaturated(a5)
	b10 := zero.SubSaturated(a6)
	b11 := zero.SubSaturated(a7)
	b12 := zero.SubSaturated(a8)
	c13 := itx16Rot(a1, a2, k(8), k(56), s12)
	c14 := itx16Rot(a1, a2, k(56), kn(8), s12)
	c15 := itx16Rot(a3, a4, k(40), k(24), s12)
	c16 := itx16Rot(a3, a4, k(24), kn(40), s12)
	c17 := itx16Rot(b10, b9, k(8), kn(56), s12)
	c18 := itx16Rot(b10, b9, k(56), k(8), s12)
	c19 := itx16Rot(b12, b11, k(40), kn(24), s12)
	c20 := itx16Rot(b12, b11, k(24), k(40), s12)
	d21 := a1.AddSaturated(a5)
	d22 := a1.SubSaturated(a5)
	d23 := a2.AddSaturated(a6)
	d24 := a2.SubSaturated(a6)
	d25 := a3.AddSaturated(a7)
	d26 := a3.SubSaturated(a7)
	d27 := a4.AddSaturated(a8)
	d28 := a4.SubSaturated(a8)
	d29 := c13.AddSaturated(c17)
	d30 := c13.SubSaturated(c17)
	d31 := c14.AddSaturated(c18)
	d32 := c14.SubSaturated(c18)
	d33 := c15.AddSaturated(c19)
	d34 := c15.SubSaturated(c19)
	d35 := c16.AddSaturated(c20)
	d36 := c16.SubSaturated(c20)
	e37 := itx16Rot(d22, d24, k(16), k(48), s12)
	e38 := itx16Rot(d22, d24, k(48), kn(16), s12)
	e39 := itx16Rot(d28, d26, k(16), kn(48), s12)
	e40 := itx16Rot(d28, d26, k(48), k(16), s12)
	e41 := itx16Rot(d30, d32, k(16), k(48), s12)
	e42 := itx16Rot(d30, d32, k(48), kn(16), s12)
	e43 := itx16Rot(d36, d34, k(16), kn(48), s12)
	e44 := itx16Rot(d36, d34, k(48), k(16), s12)
	f45 := d21.AddSaturated(d25)
	f46 := d23.AddSaturated(d27)
	f47 := zero.SubSaturated(f46)
	f48 := d21.SubSaturated(d25)
	f49 := d23.SubSaturated(d27)
	f50 := e37.AddSaturated(e39)
	f51 := zero.SubSaturated(f50)
	f52 := e38.AddSaturated(e40)
	f53 := e37.SubSaturated(e39)
	f54 := e38.SubSaturated(e40)
	f55 := d29.AddSaturated(d33)
	f56 := zero.SubSaturated(f55)
	f57 := d31.AddSaturated(d35)
	f58 := d29.SubSaturated(d33)
	f59 := d31.SubSaturated(d35)
	f60 := e41.AddSaturated(e43)
	f61 := e42.AddSaturated(e44)
	f62 := zero.SubSaturated(f61)
	f63 := e41.SubSaturated(e43)
	f64 := e42.SubSaturated(e44)
	g65 := itx16Rot(f48, f49, k(32), k(32), s12)
	g66 := zero.SubSaturated(g65)
	g67 := itx16Rot(f48, f49, k(32), kn(32), s12)
	g68 := itx16Rot(f53, f54, k(32), k(32), s12)
	g69 := itx16Rot(f53, f54, k(32), kn(32), s12)
	g70 := zero.SubSaturated(g69)
	g71 := itx16Rot(f58, f59, k(32), k(32), s12)
	g72 := itx16Rot(f58, f59, k(32), kn(32), s12)
	g73 := zero.SubSaturated(g72)
	g74 := itx16Rot(f63, f64, k(32), k(32), s12)
	g75 := zero.SubSaturated(g74)
	g76 := itx16Rot(f63, f64, k(32), kn(32), s12)
	out[0] = f45
	out[1] = f56
	out[2] = f60
	out[3] = f51
	out[4] = g68
	out[5] = g75
	out[6] = g71
	out[7] = g66
	out[8] = g67
	out[9] = g73
	out[10] = g76
	out[11] = g70
	out[12] = f52
	out[13] = f62
	out[14] = f57
	out[15] = f47
}

// itx8ADSTNet8 is inverseADST8 on eight lanes for inputs below 8 (the rest
// zero): in[k] is coefficient k, out[r] output r. Generated from the stage
// structure of inverseADST8To with the zero operands folded.
func itx8ADSTNet8(in *[8]archsimd.Int16x8, out *[8]archsimd.Int16x8) {
	k := func(i int) archsimd.Int16x8 { return archsimd.LoadInt16x8Array(&itx64K[i]) }
	kn := func(i int) archsimd.Int16x8 { return archsimd.LoadInt16x8Array(&itx64K[64+i]) }
	s12 := archsimd.LoadInt32x4Array(&itx16Scale12)
	var zero archsimd.Int16x8
	a1 := itx16Rot(in[7], in[0], k(4), k(60), s12)
	a2 := itx16Rot(in[7], in[0], k(60), kn(4), s12)
	a3 := itx16Rot(in[5], in[2], k(20), k(44), s12)
	a4 := itx16Rot(in[5], in[2], k(44), kn(20), s12)
	a5 := itx16Rot(in[3], in[4], k(36), k(28), s12)
	a6 := itx16Rot(in[3], in[4], k(28), kn(36), s12)
	a7 := itx16Rot(in[1], in[6], k(52), k(12), s12)
	a8 := itx16Rot(in[1], in[6], k(12), kn(52), s12)
	b9 := a1.AddSaturated(a5)
	b10 := a1.SubSaturated(a5)
	b11 := a2.AddSaturated(a6)
	b12 := a2.SubSaturated(a6)
	b13 := a3.AddSaturated(a7)
	b14 := a3.SubSaturated(a7)
	b15 := a4.AddSaturated(a8)
	b16 := a4.SubSaturated(a8)
	c17 := itx16Rot(b10, b12, k(16), k(48), s12)
	c18 := itx16Rot(b10, b12, k(48), kn(16), s12)
	c19 := itx16Rot(b16, b14, k(16), kn(48), s12)
	c20 := itx16Rot(b16, b14, k(48), k(16), s12)
	d21 := b9.AddSaturated(b13)
	d22 := b11.AddSaturated(b15)
	d23 := zero.SubSaturated(d22)
	d24 := b9.SubSaturated(b13)
	d25 := b11.SubSaturated(b15)
	d26 := c17.AddSaturated(c19)
	d27 := zero.SubSaturated(d26)
	d28 := c18.AddSaturated(c20)
	d29 := c17.SubSaturated(c19)
	d30 := c18.SubSaturated(c20)
	e31 := itx16Rot(d24, d25, k(32), k(32), s12)
	e32 := zero.SubSaturated(e31)
	e33 := itx16Rot(d24, d25, k(32), kn(32), s12)
	e34 := itx16Rot(d29, d30, k(32), k(32), s12)
	e35 := itx16Rot(d29, d30, k(32), kn(32), s12)
	e36 := zero.SubSaturated(e35)
	out[0] = d21
	out[1] = d27
	out[2] = e34
	out[3] = e32
	out[4] = e33
	out[5] = e36
	out[6] = d28
	out[7] = d23
}

// itx8ADSTNet4 is inverseADST8 on eight lanes for inputs below 4 (the rest
// zero): in[k] is coefficient k, out[r] output r. Generated from the stage
// structure of inverseADST8To with the zero operands folded.
func itx8ADSTNet4(in *[8]archsimd.Int16x8, out *[8]archsimd.Int16x8) {
	k := func(i int) archsimd.Int16x8 { return archsimd.LoadInt16x8Array(&itx64K[i]) }
	kn := func(i int) archsimd.Int16x8 { return archsimd.LoadInt16x8Array(&itx64K[64+i]) }
	s12 := archsimd.LoadInt32x4Array(&itx16Scale12)
	var zero archsimd.Int16x8
	a1 := itx16Rot1(in[0], k(60), s12)
	a2 := itx16Rot1(in[0], kn(4), s12)
	a3 := itx16Rot1(in[2], k(44), s12)
	a4 := itx16Rot1(in[2], kn(20), s12)
	a5 := itx16Rot1(in[3], k(36), s12)
	a6 := itx16Rot1(in[3], k(28), s12)
	a7 := itx16Rot1(in[1], k(52), s12)
	a8 := itx16Rot1(in[1], k(12), s12)
	b9 := a1.AddSaturated(a5)
	b10 := a1.SubSaturated(a5)
	b11 := a2.AddSaturated(a6)
	b12 := a2.SubSaturated(a6)
	b13 := a3.AddSaturated(a7)
	b14 := a3.SubSaturated(a7)
	b15 := a4.AddSaturated(a8)
	b16 := a4.SubSaturated(a8)
	c17 := itx16Rot(b10, b12, k(16), k(48), s12)
	c18 := itx16Rot(b10, b12, k(48), kn(16), s12)
	c19 := itx16Rot(b16, b14, k(16), kn(48), s12)
	c20 := itx16Rot(b16, b14, k(48), k(16), s12)
	d21 := b9.AddSaturated(b13)
	d22 := b11.AddSaturated(b15)
	d23 := zero.SubSaturated(d22)
	d24 := b9.SubSaturated(b13)
	d25 := b11.SubSaturated(b15)
	d26 := c17.AddSaturated(c19)
	d27 := zero.SubSaturated(d26)
	d28 := c18.AddSaturated(c20)
	d29 := c17.SubSaturated(c19)
	d30 := c18.SubSaturated(c20)
	e31 := itx16Rot(d24, d25, k(32), k(32), s12)
	e32 := zero.SubSaturated(e31)
	e33 := itx16Rot(d24, d25, k(32), kn(32), s12)
	e34 := itx16Rot(d29, d30, k(32), k(32), s12)
	e35 := itx16Rot(d29, d30, k(32), kn(32), s12)
	e36 := zero.SubSaturated(e35)
	out[0] = d21
	out[1] = d27
	out[2] = e34
	out[3] = e32
	out[4] = e33
	out[5] = e36
	out[6] = d28
	out[7] = d23
}

// itx16Net1D runs the 16-point network of t for n nonzero inputs (4, 8 or
// 16), reversing the outputs for the flipped ADST.
func itx16Net1D(t tx1DType, n int, in *[16]archsimd.Int16x8, out *[16]archsimd.Int16x8) {
	if t == tx1DDCT {
		itx16Net(n, in, out)
		return
	}
	switch {
	case n <= 4:
		itx16ADSTNet4(in, out)
	case n <= 8:
		itx16ADSTNet8(in, out)
	default:
		itx16ADSTNet16(in, out)
	}
	if t == tx1DFlipADST {
		for i := 0; i < 8; i++ {
			out[i], out[15-i] = out[15-i], out[i]
		}
	}
}

// itx8Net1D is itx16Net1D for the 8-point networks (n is 4 or 8).
func itx8Net1D(t tx1DType, n int, in *[8]archsimd.Int16x8, out *[8]archsimd.Int16x8) {
	if t == tx1DDCT {
		itx8Net(in, out)
		return
	}
	if n <= 4 {
		itx8ADSTNet4(in, out)
	} else {
		itx8ADSTNet8(in, out)
	}
	if t == tx1DFlipADST {
		for i := 0; i < 4; i++ {
			out[i], out[7-i] = out[7-i], out[i]
		}
	}
}

// itx16RowStripT is itx16RowStrip with the row network t.
func itx16RowStripT(coeff []int32, stride int, r0 int, inCols int, t tx1DType, tmp *[16 * 16]int16, colL1 *[2]archsimd.Uint16x8) bool {
	_ = coeff[(inCols-1)*stride+r0+7]
	var in [16]archsimd.Int16x8
	var l1 archsimd.Uint16x8
	for k := 0; k < inCols; k++ {
		x := itx16LoadNarrow(coeff[k*stride+r0 : k*stride+r0+8])
		l1 = l1.AddSaturated(x.Abs().ToBits())
		in[k] = x
	}
	if l1.ReduceMax() > itx16ColumnL1Bound {
		return false
	}
	var out [16]archsimd.Int16x8
	itx16Net1D(t, inCols, &in, &out)
	one := archsimd.BroadcastInt16x8(1)
	sh1 := archsimd.BroadcastInt16x8(-1)
	for g := 0; g < 2; g++ {
		var v [8]archsimd.Int16x8
		for j := 0; j < 8; j++ {
			v[j] = itx32Round2(out[8*g+j], sh1, sh1, one)
		}
		itx16Transpose8x8(&v)
		gl1 := colL1[g]
		for r := 0; r < 8; r++ {
			v[r].StoreArray((*[8]int16)(tmp[(r0+r)*16+8*g:]))
			gl1 = gl1.AddSaturated(v[r].Abs().ToBits())
		}
		colL1[g] = gl1
	}
	return true
}

// itxHybridTypes returns the per-pass networks of the DCT/ADST/flipped-ADST
// pairings the fused hybrid kernels cover; ok is false for every other
// transform type (including DCT_DCT, which has its own kernels).
func itxHybridTypes(txType Type) (vertical, horizontal tx1DType, ok bool) {
	switch txType {
	case TypeADSTDCT, TypeDCTADST, TypeADSTADST, TypeFlipADSTDCT, TypeDCTFlipADST,
		TypeFlipADSTFlipADST, TypeADSTFlipADST, TypeFlipADSTADST:
		return txType.tx1DTypes()
	}
	return 0, 0, false
}

// InverseAdd16x16Hybrid reconstructs one 8-bit 16x16 block of a
// DCT/ADST/flipped-ADST/identity transform pairing other than DCT_DCT and
// IDTX with coded coefficients in rows 0..activeRows-1 and columns
// 0..activeCols-1: dst = clip(pred + inverse(coeff)), with the dequantized
// coefficients column-major at coeffStride (16). pred may be dst. It
// reports false, with dst untouched, when the block is outside the
// kernel's contract or a guard declines; the caller then runs the staged
// path, which produces the same bytes. The kernels are generated
// (itx_fused_gen_arm64.go) per type and network form.
func InverseAdd16x16Hybrid(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, txType Type, activeRows, activeCols int) bool {
	if activeRows <= 0 || activeRows > 16 || activeCols <= 0 || activeCols > 16 ||
		coeffStride != 16 || len(coeff) < 256 || dstStride < 16 || predStride < 16 ||
		len(dst) < 15*dstStride+16 || len(pred) < 15*predStride+16 {
		return false
	}
	return itxGen16x16Hybrid((*[256]int32)(coeff[:256]), dst, dstStride, pred, predStride, txType, activeCols, activeRows)
}

// InverseAdd8x8Hybrid is InverseAdd16x16Hybrid for the 8x8 block (every
// pairing but DCT_DCT and IDTX; coeffStride 8).
func InverseAdd8x8Hybrid(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, txType Type, activeRows, activeCols int) bool {
	if activeRows <= 0 || activeRows > 8 || activeCols <= 0 || activeCols > 8 ||
		coeffStride != 8 || len(coeff) < 64 || dstStride < 8 || predStride < 8 ||
		len(dst) < 7*dstStride+8 || len(pred) < 7*predStride+8 {
		return false
	}
	return itxGen8x8Hybrid((*[64]int32)(coeff[:64]), dst, dstStride, pred, predStride, txType, activeCols, activeRows)
}

// The 4x4 and the 4- and 8/16-point rectangular 8-bit blocks of every type
// but IDTX: the generated kernels (itx_fused_gen_arm64.go), DCT_DCT per
// network form of the coded columns and rows, every other type in the
// full form.

// itxSmallFits reports whether a w x h block's buffers satisfy the
// generated kernels' contract: coefficients column-major at stride h, dst
// and pred rows of w bytes at their strides.
func itxSmallFits(w, h int, dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride, activeRows, activeCols int) bool {
	return activeRows > 0 && activeRows <= h && activeCols > 0 && activeCols <= w &&
		coeffStride == h && len(coeff) >= w*h && dstStride >= w && predStride >= w &&
		len(dst) >= (h-1)*dstStride+w && len(pred) >= (h-1)*predStride+w
}

// InverseAdd4x4 reconstructs one 8-bit 4x4 block of any type but IDTX:
// dst = clip(pred + inverse(coeff)), with the dequantized coefficients
// column-major at coeffStride (4). pred may be dst. It reports false, with
// dst untouched, when the block is outside the kernels' contract or a
// guard declines; the caller then runs the staged path, which produces the
// same bytes.
func InverseAdd4x4(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, txType Type, activeRows int) bool {
	if !itxSmallFits(4, 4, dst, dstStride, pred, predStride, coeff, coeffStride, activeRows, 4) {
		return false
	}
	return itxGen4x4((*[16]int32)(coeff[:16]), dst, dstStride, pred, predStride, txType, 4, activeRows)
}

// InverseAdd4x8 is InverseAdd4x4 for the 4 wide, 8 high block.
func InverseAdd4x8(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, txType Type, activeRows, activeCols int) bool {
	if !itxSmallFits(4, 8, dst, dstStride, pred, predStride, coeff, coeffStride, activeRows, activeCols) {
		return false
	}
	return itxGen4x8((*[32]int32)(coeff[:32]), dst, dstStride, pred, predStride, txType, activeCols, activeRows)
}

// InverseAdd8x4 is InverseAdd4x4 for the 8 wide, 4 high block.
func InverseAdd8x4(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, txType Type, activeRows, activeCols int) bool {
	if !itxSmallFits(8, 4, dst, dstStride, pred, predStride, coeff, coeffStride, activeRows, activeCols) {
		return false
	}
	return itxGen8x4((*[32]int32)(coeff[:32]), dst, dstStride, pred, predStride, txType, activeCols, activeRows)
}

// InverseAdd4x16 is InverseAdd4x4 for the 4 wide, 16 high block.
func InverseAdd4x16(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, txType Type, activeRows, activeCols int) bool {
	if !itxSmallFits(4, 16, dst, dstStride, pred, predStride, coeff, coeffStride, activeRows, activeCols) {
		return false
	}
	return itxGen4x16((*[64]int32)(coeff[:64]), dst, dstStride, pred, predStride, txType, activeCols, activeRows)
}

// InverseAdd16x4 is InverseAdd4x4 for the 16 wide, 4 high block.
func InverseAdd16x4(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, txType Type, activeRows, activeCols int) bool {
	if !itxSmallFits(16, 4, dst, dstStride, pred, predStride, coeff, coeffStride, activeRows, activeCols) {
		return false
	}
	return itxGen16x4((*[64]int32)(coeff[:64]), dst, dstStride, pred, predStride, txType, activeCols, activeRows)
}

// InverseAdd8x16 is InverseAdd4x4 for the 8 wide, 16 high block.
func InverseAdd8x16(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, txType Type, activeRows, activeCols int) bool {
	if !itxSmallFits(8, 16, dst, dstStride, pred, predStride, coeff, coeffStride, activeRows, activeCols) {
		return false
	}
	return itxGen8x16((*[128]int32)(coeff[:128]), dst, dstStride, pred, predStride, txType, activeCols, activeRows)
}

// InverseAdd16x8 is InverseAdd4x4 for the 16 wide, 8 high block.
func InverseAdd16x8(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, txType Type, activeRows, activeCols int) bool {
	if !itxSmallFits(16, 8, dst, dstStride, pred, predStride, coeff, coeffStride, activeRows, activeCols) {
		return false
	}
	return itxGen16x8((*[128]int32)(coeff[:128]), dst, dstStride, pred, predStride, txType, activeCols, activeRows)
}

// itx_fused_rect: the fused two-pass inverse transform (itx_fused16,
// itx_fused32) for the 2:1 rectangular 8-bit DCT_DCT blocks 16x32 and
// 32x16. The pass structure is the square kernels' with two differences the
// scalar path has for the rect2 sizes:
//
//   - the coefficients are scaled by 1/sqrt(2) before the row transform,
//     Round2(c*2896, 12) = (c*181 + 128) >> 8, and only then clamped to the
//     row range, which the narrowing load does (itx16LoadNarrowRect2);
//   - the inter-pass shift is 1, not 2.
//
// The row network is the DCT of the block's width (DCT16 for 16x32, DCT32
// for 32x16) and the column network the DCT of its height, each in the form
// its coded extent allows. Both are the certified networks, so the L1 guards
// and their gain model carry over unchanged: they bound the values the
// networks see (the scaled row inputs, the stored inter-pass values), not how
// those values were produced.

// itxRect2Pre is the pre-clamp that keeps c*181+128 inside int32 without
// changing the result: scale(+-1<<22) already lies far outside the int16 row
// clamp, so clamping the input to +-1<<22 first is exact (see
// stageTransposeClamp).
const itxRect2Pre = 1 << 22

// itxRect2Consts holds the broadcast constants of the rect2 narrowing load.
type itxRect2Consts struct {
	preLo, preHi, k181, r128 archsimd.Int32x4
}

func newItxRect2Consts() itxRect2Consts {
	return itxRect2Consts{
		preLo: archsimd.BroadcastInt32x4(-itxRect2Pre),
		preHi: archsimd.BroadcastInt32x4(itxRect2Pre),
		k181:  archsimd.BroadcastInt32x4(181),
		r128:  archsimd.BroadcastInt32x4(128),
	}
}

// itx16LoadNarrowRect2 loads eight int32 coefficients, applies the rect2
// scale and narrows to int16 with saturation (the 8-bit row clamp): lane i
// is clip16(Round2(p[i]*2896, 12)).
func itx16LoadNarrowRect2(p []int32, k *itxRect2Consts) archsimd.Int16x8 {
	lo := archsimd.LoadInt32x4Array((*[4]int32)(p))
	hi := archsimd.LoadInt32x4Array((*[4]int32)(p[4:]))
	lo = lo.Max(k.preLo).Min(k.preHi).MulAdd(k.k181, k.r128).ShiftAllRight(8)
	hi = hi.Max(k.preLo).Min(k.preHi).MulAdd(k.k181, k.r128).ShiftAllRight(8)
	v := lo.SaturateToInt16().ToBits().ReshapeToUint64s().BitsToFloat64()
	v = v.SetElem(1, hi.SaturateToInt16().ToBits().ReshapeToUint64s().BitsToFloat64().GetElem(0))
	return v.ToBits().ReshapeToUint16s().BitsToInt16()
}

// itxRound2By1 is Round2(x, 1) on int16 lanes, (x >> 1) + (x & 1): exact and
// overflow-free.
func itxRound2By1(x, sh1, one archsimd.Int16x8) archsimd.Int16x8 {
	return x.Shift(sh1).Add(x.And(one))
}

// itx16x32RowStrip runs the 16x32 row pass (DCT16 rows) for coefficient rows
// r0..r0+7 into tmp rows r0..r0+7 (16 wide).
func itx16x32RowStrip(coeff []int32, stride int, r0 int, inCols int, tmp *[32 * 16]int16, colL1 *[2]archsimd.Uint16x8) bool {
	_ = coeff[(inCols-1)*stride+r0+7]
	k := newItxRect2Consts()
	var in [16]archsimd.Int16x8
	var l1 archsimd.Uint16x8
	for c := 0; c < inCols; c++ {
		x := itx16LoadNarrowRect2(coeff[c*stride+r0:c*stride+r0+8], &k)
		l1 = l1.AddSaturated(x.Abs().ToBits())
		in[c] = x
	}
	if l1.ReduceMax() > itx16ColumnL1Bound {
		return false
	}
	var out [16]archsimd.Int16x8
	itx16Net(inCols, &in, &out)
	one := archsimd.BroadcastInt16x8(1)
	sh1 := archsimd.BroadcastInt16x8(-1)
	for g := 0; g < 2; g++ {
		var v [8]archsimd.Int16x8
		for j := 0; j < 8; j++ {
			v[j] = itxRound2By1(out[8*g+j], sh1, one)
		}
		itx16Transpose8x8(&v)
		gl1 := colL1[g]
		for r := 0; r < 8; r++ {
			v[r].StoreArray((*[8]int16)(tmp[(r0+r)*16+8*g:]))
			gl1 = gl1.AddSaturated(v[r].Abs().ToBits())
		}
		colL1[g] = gl1
	}
	return true
}

// itx16x32ColStripAdd runs the 16x32 column pass (DCT32 columns) for columns
// c0..c0+7 from inRows coded tmp rows (8, 16 or 32) and adds the result.
func itx16x32ColStripAdd(tmp *[32 * 16]int16, c0 int, inRows int, dst []byte, dstStride int, pred []byte, predStride int) {
	dst, pred = dst[c0:], pred[c0:]
	// The entry point checked the strides against itxGenMaxStride; the
	// conversion carries that range, so each row window is one bounds check.
	ds, ps := int(uint32(dstStride)), int(uint32(predStride))
	// One compare per row window (see addRows in itxgen).
	dl, pl := len(dst)-8, len(pred)-8
	var in [32]archsimd.Int16x8
	for r := 0; r < inRows; r++ {
		in[r] = archsimd.LoadInt16x8Array((*[8]int16)(tmp[r*16+c0:]))
	}
	var out [32]archsimd.Int16x8
	itx32Net(inRows, &in, &out)
	one := archsimd.BroadcastInt16x8(1)
	sh1 := archsimd.BroadcastInt16x8(-1)
	sh3 := archsimd.BroadcastInt16x8(-3)
	for r := 0; r < 32; r++ {
		res := itx32Round2(out[r], sh3, sh1, one)
		o, po := r*ds, r*ps
		if o > dl || po > pl {
			panic(errItxGenWindow)
		}
		d, q := (*[8]byte)(dst[o:o+8]), (*[8]byte)(pred[po:po+8])
		rows := [2]uint64{binary.LittleEndian.Uint64(q[:]), 0}
		p := archsimd.LoadUint64x2Array(&rows).ReshapeToUint8s().ExtendLo8ToUint16().BitsToInt16()
		binary.LittleEndian.PutUint64(d[:], p.Add(res).SaturateToUint8().ReshapeToUint64s().GetElem(0))
	}
}

// InverseAddDCT16x32 reconstructs one 8-bit 16x32 (16 wide, 32 high)
// DCT_DCT block with coded coefficients in rows 0..activeRows-1 and columns
// 0..activeCols-1: dst = clip(pred + inverse(coeff)), with the dequantized
// coefficients column-major at coeffStride. pred may be dst. It reports
// false, with dst untouched, when the block is outside the kernel's contract
// or a guard declines; the caller then runs the staged path, which produces
// the same bytes.
func InverseAddDCT16x32(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, activeRows, activeCols int) bool {
	if activeRows <= 0 || activeRows > 32 || activeCols <= 0 || activeCols > 16 ||
		coeffStride < 32 || len(coeff) < 15*coeffStride+32 || dstStride < 16 || predStride < 16 || dstStride > itxGenMaxStride || predStride > itxGenMaxStride ||
		len(dst) < 31*dstStride+16 || len(pred) < 31*predStride+16 {
		return false
	}
	inCols := itx16Form(activeCols)
	inRows := itx64Form(activeRows)
	var tmp [32 * 16]int16
	var colL1 [2]archsimd.Uint16x8
	written := 0
	for ; written < activeRows; written += 8 {
		if !itx16x32RowStrip(coeff, coeffStride, written, inCols, &tmp, &colL1) {
			return false
		}
	}
	// The column form may read tmp rows past the strips that ran; those rows
	// are the transform of zero coefficient rows, so zero.
	if written < inRows {
		clear(tmp[written*16 : inRows*16])
	}
	if colL1[0].Max(colL1[1]).ReduceMax() > itx16ColumnL1Bound {
		return false
	}
	itx16x32ColStripAdd(&tmp, 0, inRows, dst, dstStride, pred, predStride)
	itx16x32ColStripAdd(&tmp, 8, inRows, dst, dstStride, pred, predStride)
	return true
}

// itx32x16RowStrip runs the 32x16 row pass (DCT32 rows) for coefficient rows
// r0..r0+7 into tmp rows r0..r0+7 (32 wide).
func itx32x16RowStrip(coeff []int32, stride int, r0 int, inCols int, tmp *[16 * 32]int16, colL1 *[4]archsimd.Uint16x8) bool {
	_ = coeff[(inCols-1)*stride+r0+7]
	k := newItxRect2Consts()
	var in [32]archsimd.Int16x8
	var l1 archsimd.Uint16x8
	for c := 0; c < inCols; c++ {
		x := itx16LoadNarrowRect2(coeff[c*stride+r0:c*stride+r0+8], &k)
		l1 = l1.AddSaturated(x.Abs().ToBits())
		in[c] = x
	}
	if l1.ReduceMax() > itx16ColumnL1Bound {
		return false
	}
	var out [32]archsimd.Int16x8
	itx32Net(inCols, &in, &out)
	one := archsimd.BroadcastInt16x8(1)
	sh1 := archsimd.BroadcastInt16x8(-1)
	for g := 0; g < 4; g++ {
		var v [8]archsimd.Int16x8
		for j := 0; j < 8; j++ {
			v[j] = itxRound2By1(out[8*g+j], sh1, one)
		}
		itx16Transpose8x8(&v)
		gl1 := colL1[g]
		for r := 0; r < 8; r++ {
			v[r].StoreArray((*[8]int16)(tmp[(r0+r)*32+8*g:]))
			gl1 = gl1.AddSaturated(v[r].Abs().ToBits())
		}
		colL1[g] = gl1
	}
	return true
}

// itx32x16ColStripAdd runs the 32x16 column pass (DCT16 columns) for columns
// c0..c0+7 from inRows coded tmp rows (4, 8 or 16) and adds the result.
func itx32x16ColStripAdd(tmp *[16 * 32]int16, c0 int, inRows int, dst []byte, dstStride int, pred []byte, predStride int) {
	dst, pred = dst[c0:], pred[c0:]
	// The entry point checked the strides against itxGenMaxStride; the
	// conversion carries that range, so each row window is one bounds check.
	ds, ps := int(uint32(dstStride)), int(uint32(predStride))
	// One compare per row window (see addRows in itxgen).
	dl, pl := len(dst)-8, len(pred)-8
	var in [16]archsimd.Int16x8
	for r := 0; r < inRows; r++ {
		in[r] = archsimd.LoadInt16x8Array((*[8]int16)(tmp[r*32+c0:]))
	}
	var out [16]archsimd.Int16x8
	itx16Net(inRows, &in, &out)
	one := archsimd.BroadcastInt16x8(1)
	sh1 := archsimd.BroadcastInt16x8(-1)
	sh3 := archsimd.BroadcastInt16x8(-3)
	for r := 0; r < 16; r++ {
		res := itx32Round2(out[r], sh3, sh1, one)
		o, po := r*ds, r*ps
		if o > dl || po > pl {
			panic(errItxGenWindow)
		}
		d, q := (*[8]byte)(dst[o:o+8]), (*[8]byte)(pred[po:po+8])
		rows := [2]uint64{binary.LittleEndian.Uint64(q[:]), 0}
		p := archsimd.LoadUint64x2Array(&rows).ReshapeToUint8s().ExtendLo8ToUint16().BitsToInt16()
		binary.LittleEndian.PutUint64(d[:], p.Add(res).SaturateToUint8().ReshapeToUint64s().GetElem(0))
	}
}

// InverseAddDCT32x16 is InverseAddDCT16x32 for the 32 wide, 16 high block.
func InverseAddDCT32x16(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, activeRows, activeCols int) bool {
	if activeRows <= 0 || activeRows > 16 || activeCols <= 0 || activeCols > 32 ||
		coeffStride < 16 || len(coeff) < 31*coeffStride+16 || dstStride < 32 || predStride < 32 || dstStride > itxGenMaxStride || predStride > itxGenMaxStride ||
		len(dst) < 15*dstStride+32 || len(pred) < 15*predStride+32 {
		return false
	}
	inCols := itx64Form(activeCols)
	inRows := itx16Form(activeRows)
	var tmp [16 * 32]int16
	var colL1 [4]archsimd.Uint16x8
	// A row strip writes all eight of its rows, so rows past the coded ones
	// come out zero and the column form reads them as such; the second strip
	// runs only when coded rows reach it.
	if !itx32x16RowStrip(coeff, coeffStride, 0, inCols, &tmp, &colL1) {
		return false
	}
	if activeRows > 8 && !itx32x16RowStrip(coeff, coeffStride, 8, inCols, &tmp, &colL1) {
		return false
	}
	if colL1[0].Max(colL1[1]).Max(colL1[2].Max(colL1[3])).ReduceMax() > itx16ColumnL1Bound {
		return false
	}
	for c0 := 0; c0 < 32; c0 += 8 {
		itx32x16ColStripAdd(&tmp, c0, inRows, dst, dstStride, pred, predStride)
	}
	return true
}

// 10- and 12-bit forms of the fused 2:1 rectangular DCT_DCT blocks 16x32
// and 32x16 (itx_fused_rect). dav1d's itx16 runs their 10 bpc column pass
// with the 8bpc int16 functions, as for the square sizes
// (itx_fused_hbd_gosimd_arm64.go). The row strips are the 8-bit ones: the
// load scales each coefficient by 1/sqrt(2) in int32 and only then narrows
// with saturation, so a scaled value outside int16 saturates to +-32767 and
// fails the per-lane L1 guard (itx16ColumnL1Bound < 32767). Under the guard
// every scaled row input lies inside int16, so the staged path's clamp of
// the scaled input to bd + 8 bits does not bite, nor does its
// max(bd + 6, 16) column clamp; the column output is added to the 16bpc
// prediction with the saturating add clipped to the bit depth. A declined
// block is left untouched for the staged path.

// itx16x32ColStripAddHBD is itx16x32ColStripAdd onto 16bpc samples.
func itx16x32ColStripAddHBD(tmp *[32 * 16]int16, c0 int, inRows int, dst []byte, dstStride int, pred []byte, predStride int, maxV archsimd.Int16x8) {
	dst, pred = dst[2*c0:], pred[2*c0:]
	// itxHBDRectFits checked the strides against itxGenMaxStride; the conversion
	// carries that range, so each row window is one bounds check.
	ds, ps := int(uint32(dstStride)), int(uint32(predStride))
	var in [32]archsimd.Int16x8
	for r := 0; r < inRows; r++ {
		in[r] = archsimd.LoadInt16x8Array((*[8]int16)(tmp[r*16+c0:]))
	}
	var out [32]archsimd.Int16x8
	itx32Net(inRows, &in, &out)
	one := archsimd.BroadcastInt16x8(1)
	sh1 := archsimd.BroadcastInt16x8(-1)
	sh3 := archsimd.BroadcastInt16x8(-3)
	zero := archsimd.BroadcastInt16x8(0)
	for r := 0; r < 32; r++ {
		itxAddRow16(dst, r*ds, pred, r*ps, itx32Round2(out[r], sh3, sh1, one), zero, maxV)
	}
}

// InverseAddDCT16x32HBD is InverseAddDCT16x32 for a 10- or 12-bit block
// (2-byte samples, strides in bytes).
func InverseAddDCT16x32HBD(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, activeRows, activeCols int, bitDepth uint8) bool {
	maxV, ok := itxHBDMax(bitDepth)
	if !ok || activeRows <= 0 || activeRows > 32 || activeCols <= 0 || activeCols > 16 ||
		coeffStride < 32 || len(coeff) < 15*coeffStride+32 || !itxHBDRectFits(dst, dstStride, 16, 32) || !itxHBDRectFits(pred, predStride, 16, 32) {
		return false
	}
	inCols := itx16Form(activeCols)
	inRows := itx64Form(activeRows)
	var tmp [32 * 16]int16
	var colL1 [2]archsimd.Uint16x8
	written := 0
	for ; written < activeRows; written += 8 {
		if !itx16x32RowStrip(coeff, coeffStride, written, inCols, &tmp, &colL1) {
			return false
		}
	}
	// The column form may read tmp rows past the strips that ran; those rows
	// are the transform of zero coefficient rows, so zero.
	if written < inRows {
		clear(tmp[written*16 : inRows*16])
	}
	if colL1[0].Max(colL1[1]).ReduceMax() > itx16ColumnL1Bound {
		return false
	}
	itx16x32ColStripAddHBD(&tmp, 0, inRows, dst, dstStride, pred, predStride, maxV)
	itx16x32ColStripAddHBD(&tmp, 8, inRows, dst, dstStride, pred, predStride, maxV)
	return true
}

// itx32x16ColStripAddHBD is itx32x16ColStripAdd onto 16bpc samples.
func itx32x16ColStripAddHBD(tmp *[16 * 32]int16, c0 int, inRows int, dst []byte, dstStride int, pred []byte, predStride int, maxV archsimd.Int16x8) {
	dst, pred = dst[2*c0:], pred[2*c0:]
	// itxHBDRectFits checked the strides against itxGenMaxStride; the conversion
	// carries that range, so each row window is one bounds check.
	ds, ps := int(uint32(dstStride)), int(uint32(predStride))
	var in [16]archsimd.Int16x8
	for r := 0; r < inRows; r++ {
		in[r] = archsimd.LoadInt16x8Array((*[8]int16)(tmp[r*32+c0:]))
	}
	var out [16]archsimd.Int16x8
	itx16Net(inRows, &in, &out)
	one := archsimd.BroadcastInt16x8(1)
	sh1 := archsimd.BroadcastInt16x8(-1)
	sh3 := archsimd.BroadcastInt16x8(-3)
	zero := archsimd.BroadcastInt16x8(0)
	for r := 0; r < 16; r++ {
		itxAddRow16(dst, r*ds, pred, r*ps, itx32Round2(out[r], sh3, sh1, one), zero, maxV)
	}
}

// InverseAddDCT32x16HBD is InverseAddDCT32x16 for a 10- or 12-bit block.
func InverseAddDCT32x16HBD(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, activeRows, activeCols int, bitDepth uint8) bool {
	maxV, ok := itxHBDMax(bitDepth)
	if !ok || activeRows <= 0 || activeRows > 16 || activeCols <= 0 || activeCols > 32 ||
		coeffStride < 16 || len(coeff) < 31*coeffStride+16 || !itxHBDRectFits(dst, dstStride, 32, 16) || !itxHBDRectFits(pred, predStride, 32, 16) {
		return false
	}
	inCols := itx64Form(activeCols)
	inRows := itx16Form(activeRows)
	var tmp [16 * 32]int16
	var colL1 [4]archsimd.Uint16x8
	// A row strip writes all eight of its rows, so rows past the coded ones
	// come out zero and the column form reads them as such; the second strip
	// runs only when coded rows reach it.
	if !itx32x16RowStrip(coeff, coeffStride, 0, inCols, &tmp, &colL1) {
		return false
	}
	if activeRows > 8 && !itx32x16RowStrip(coeff, coeffStride, 8, inCols, &tmp, &colL1) {
		return false
	}
	if colL1[0].Max(colL1[1]).Max(colL1[2].Max(colL1[3])).ReduceMax() > itx16ColumnL1Bound {
		return false
	}
	for c0 := 0; c0 < 32; c0 += 8 {
		itx32x16ColStripAddHBD(&tmp, c0, inRows, dst, dstStride, pred, predStride, maxV)
	}
	return true
}

// itxHBDRectFits reports whether a w x h block of 2-byte samples at stride
// (bytes) fits buf.
func itxHBDRectFits(buf []byte, stride, w, h int) bool {
	return stride >= 2*w && stride <= itxGenMaxStride && len(buf) >= (h-1)*stride+2*w
}

// inverseDCT32Col4Low8SIMD is mechanically derived from
// inverseDCT32Col4SIMD. The caller selects it only after staging and the row
// pass prove rows 8..31 are exactly zero. Those input vectors are replaced by
// zero; all remaining graph operations and all 32 output stores retain the
// full kernel's order and clamp points.
func inverseDCT32Col4Low8SIMD(buf []int32, rowStride int, min, max int32) {
	minV := archsimd.BroadcastInt32x4(min)
	maxV := archsimd.BroadcastInt32x4(max)
	// Shift counts built once: a constant ShiftAllRight rebuilds its count
	// vector (MOVD + INS + DUP) at every use.
	shr7 := archsimd.BroadcastInt32x4(-7)
	shr10 := archsimd.BroadcastInt32x4(-10)
	shr11 := archsimd.BroadcastInt32x4(-11)
	var zero archsimd.Int32x4
	km1380 := archsimd.LoadInt32x4Array(&itxCm1380)
	km1189 := archsimd.LoadInt32x4Array(&itxCm1189)
	km1138 := archsimd.LoadInt32x4Array(&itxCm1138)
	km601 := archsimd.LoadInt32x4Array(&itxCm601)
	km312 := archsimd.LoadInt32x4Array(&itxCm312)
	km239 := archsimd.LoadInt32x4Array(&itxCm239)
	km176 := archsimd.LoadInt32x4Array(&itxCm176)
	km123 := archsimd.LoadInt32x4Array(&itxCm123)
	km79 := archsimd.LoadInt32x4Array(&itxCm79)
	km44 := archsimd.LoadInt32x4Array(&itxCm44)
	km20 := archsimd.LoadInt32x4Array(&itxCm20)
	km5 := archsimd.LoadInt32x4Array(&itxCm5)
	kp79 := archsimd.LoadInt32x4Array(&itxCp79)
	kp181 := archsimd.LoadInt32x4Array(&itxCp181)
	kp201 := archsimd.LoadInt32x4Array(&itxCp201)
	kp312 := archsimd.LoadInt32x4Array(&itxCp312)
	kp401 := archsimd.LoadInt32x4Array(&itxCp401)
	kp799 := archsimd.LoadInt32x4Array(&itxCp799)
	kp995 := archsimd.LoadInt32x4Array(&itxCp995)
	kp1138 := archsimd.LoadInt32x4Array(&itxCp1138)
	kp1567 := archsimd.LoadInt32x4Array(&itxCp1567)
	kp1703 := archsimd.LoadInt32x4Array(&itxCp1703)
	x0 := archsimd.LoadInt32x4Array((*[4]int32)(buf[0*rowStride:]))
	x1 := archsimd.LoadInt32x4Array((*[4]int32)(buf[1*rowStride:]))
	x2 := archsimd.LoadInt32x4Array((*[4]int32)(buf[2*rowStride:]))
	x3 := archsimd.LoadInt32x4Array((*[4]int32)(buf[3*rowStride:]))
	x4 := archsimd.LoadInt32x4Array((*[4]int32)(buf[4*rowStride:]))
	x5 := archsimd.LoadInt32x4Array((*[4]int32)(buf[5*rowStride:]))
	x6 := archsimd.LoadInt32x4Array((*[4]int32)(buf[6*rowStride:]))
	x7 := archsimd.LoadInt32x4Array((*[4]int32)(buf[7*rowStride:]))
	v1 := x0.Mul(kp181).Shift(shr7).Average(zero)
	v3 := zero
	v5 := v1.Max(minV).Min(maxV)
	v9 := x4.Mul(kp799).Shift(shr11).Average(v3)
	v12 := x4.Mul(km79).Shift(shr11).Average(v3).Add(x4)
	v13 := v9.Max(minV).Min(maxV)
	v15 := v12.Max(minV).Min(maxV)
	v17 := v15.Sub(v13).Mul(kp181).Shift(shr7).Average(v3)
	v18 := v15.Add(v13).Mul(kp181).Shift(shr7).Average(v3)
	v19 := v5.Add(v15).Max(minV).Min(maxV)
	v20 := v5.Add(v18).Max(minV).Min(maxV)
	v21 := v5.Add(v17).Max(minV).Min(maxV)
	v22 := v5.Add(v13).Max(minV).Min(maxV)
	v23 := v5.Sub(v13).Max(minV).Min(maxV)
	v24 := v5.Sub(v17).Max(minV).Min(maxV)
	v25 := v5.Sub(v18).Max(minV).Min(maxV)
	v26 := v5.Sub(v15).Max(minV).Min(maxV)
	v27 := x2.Mul(kp401).Shift(shr11).Average(v3)
	v30 := x6.Mul(km1189).Shift(shr11).Average(v3)
	v31 := x6.Mul(km176).Shift(shr11).Average(v3).Add(x6)
	v34 := x2.Mul(km20).Shift(shr11).Average(v3).Add(x2)
	v35 := v27.Max(minV).Min(maxV)
	v37 := v30.Max(minV).Min(maxV)
	v39 := v31.Max(minV).Min(maxV)
	v41 := v34.Max(minV).Min(maxV)
	v43 := v35.MulAdd(kp312, v41.Mul(kp1567)).Shift(shr11).Average(v3).Sub(v35)
	v44 := v35.MulAdd(kp1567, v41.Mul(km312)).Shift(shr11).Average(v3).Add(v41)
	v45 := v3.Sub(v37.MulAdd(kp1567, v39.Mul(km312))).Shift(shr11).Average(v3).Sub(v39)
	v46 := v37.MulAdd(kp312, v39.Mul(kp1567)).Shift(shr11).Average(v3).Sub(v37)
	v47 := v35.Add(v37).Max(minV).Min(maxV)
	v48 := v43.Add(v45).Max(minV).Min(maxV)
	v49 := v43.Sub(v45).Max(minV).Min(maxV)
	v50 := v35.Sub(v37).Max(minV).Min(maxV)
	v51 := v41.Sub(v39).Max(minV).Min(maxV)
	v52 := v44.Sub(v46).Max(minV).Min(maxV)
	v53 := v44.Add(v46).Max(minV).Min(maxV)
	v54 := v41.Add(v39).Max(minV).Min(maxV)
	v55 := v52.Sub(v49).Mul(kp181).Shift(shr7).Average(v3)
	v56 := v52.Add(v49).Mul(kp181).Shift(shr7).Average(v3)
	v57 := v51.Sub(v50).Mul(kp181).Shift(shr7).Average(v3)
	v58 := v51.Add(v50).Mul(kp181).Shift(shr7).Average(v3)
	v59 := v19.Add(v54).Max(minV).Min(maxV)
	v60 := v20.Add(v53).Max(minV).Min(maxV)
	v61 := v21.Add(v56).Max(minV).Min(maxV)
	v62 := v22.Add(v58).Max(minV).Min(maxV)
	v63 := v23.Add(v57).Max(minV).Min(maxV)
	v64 := v24.Add(v55).Max(minV).Min(maxV)
	v65 := v25.Add(v48).Max(minV).Min(maxV)
	v66 := v26.Add(v47).Max(minV).Min(maxV)
	v67 := v26.Sub(v47).Max(minV).Min(maxV)
	v68 := v25.Sub(v48).Max(minV).Min(maxV)
	v69 := v24.Sub(v55).Max(minV).Min(maxV)
	v70 := v23.Sub(v57).Max(minV).Min(maxV)
	v71 := v22.Sub(v58).Max(minV).Min(maxV)
	v72 := v21.Sub(v56).Max(minV).Min(maxV)
	v73 := v20.Sub(v53).Max(minV).Min(maxV)
	v74 := v19.Sub(v54).Max(minV).Min(maxV)
	v75 := x1.Mul(kp201).Shift(shr11).Average(v3)
	v78 := x7.Mul(km1380).Shift(shr11).Average(v3)
	v79 := x5.Mul(kp995).Shift(shr11).Average(v3)
	v82 := x3.Mul(km601).Shift(shr11).Average(v3)
	v83 := x3.Mul(km44).Shift(shr11).Average(v3).Add(x3)
	v86 := x5.Mul(km123).Shift(shr11).Average(v3).Add(x5)
	v87 := x7.Mul(km239).Shift(shr11).Average(v3).Add(x7)
	v90 := x1.Mul(km5).Shift(shr11).Average(v3).Add(x1)
	v91 := v75.Max(minV).Min(maxV)
	v93 := v78.Max(minV).Min(maxV)
	v95 := v79.Max(minV).Min(maxV)
	v97 := v82.Max(minV).Min(maxV)
	v99 := v83.Max(minV).Min(maxV)
	v101 := v86.Max(minV).Min(maxV)
	v103 := v87.Max(minV).Min(maxV)
	v105 := v90.Max(minV).Min(maxV)
	v107 := v91.MulAdd(kp79, v105.Mul(kp799)).Shift(shr11).Average(v3).Sub(v91)
	v108 := v91.MulAdd(kp799, v105.Mul(km79)).Shift(shr11).Average(v3).Add(v105)
	v109 := v3.Sub(v93.MulAdd(kp799, v103.Mul(km79))).Shift(shr11).Average(v3).Sub(v103)
	v110 := v93.MulAdd(kp79, v103.Mul(kp799)).Shift(shr11).Average(v3).Sub(v93)
	v111 := v95.MulAdd(km1138, v101.Mul(kp1703)).Shift(shr10).Average(v3)
	v112 := v95.MulAdd(kp1703, v101.Mul(kp1138)).Shift(shr10).Average(v3)
	v113 := v3.Sub(v97.MulAdd(kp1703, v99.Mul(kp1138))).Shift(shr10).Average(v3)
	v114 := v97.MulAdd(km1138, v99.Mul(kp1703)).Shift(shr10).Average(v3)
	v115 := v91.Add(v93).Max(minV).Min(maxV)
	v116 := v107.Add(v109).Max(minV).Min(maxV)
	v117 := v107.Sub(v109).Max(minV).Min(maxV)
	v118 := v91.Sub(v93).Max(minV).Min(maxV)
	v119 := v97.Sub(v95).Max(minV).Min(maxV)
	v120 := v113.Sub(v111).Max(minV).Min(maxV)
	v121 := v113.Add(v111).Max(minV).Min(maxV)
	v122 := v97.Add(v95).Max(minV).Min(maxV)
	v123 := v99.Add(v101).Max(minV).Min(maxV)
	v124 := v114.Add(v112).Max(minV).Min(maxV)
	v125 := v114.Sub(v112).Max(minV).Min(maxV)
	v126 := v99.Sub(v101).Max(minV).Min(maxV)
	v127 := v105.Sub(v103).Max(minV).Min(maxV)
	v128 := v108.Sub(v110).Max(minV).Min(maxV)
	v129 := v108.Add(v110).Max(minV).Min(maxV)
	v130 := v105.Add(v103).Max(minV).Min(maxV)
	v131 := v117.MulAdd(kp312, v128.Mul(kp1567)).Shift(shr11).Average(v3).Sub(v117)
	v132 := v117.MulAdd(kp1567, v128.Mul(km312)).Shift(shr11).Average(v3).Add(v128)
	v133 := v118.MulAdd(kp312, v127.Mul(kp1567)).Shift(shr11).Average(v3).Sub(v118)
	v134 := v118.MulAdd(kp1567, v127.Mul(km312)).Shift(shr11).Average(v3).Add(v127)
	v135 := v3.Sub(v119.MulAdd(kp1567, v126.Mul(km312))).Shift(shr11).Average(v3).Sub(v126)
	v136 := v119.MulAdd(kp312, v126.Mul(kp1567)).Shift(shr11).Average(v3).Sub(v119)
	v137 := v3.Sub(v120.MulAdd(kp1567, v125.Mul(km312))).Shift(shr11).Average(v3).Sub(v125)
	v138 := v120.MulAdd(kp312, v125.Mul(kp1567)).Shift(shr11).Average(v3).Sub(v120)
	v139 := v115.Add(v122).Max(minV).Min(maxV)
	v140 := v116.Add(v121).Max(minV).Min(maxV)
	v141 := v131.Add(v137).Max(minV).Min(maxV)
	v142 := v133.Add(v135).Max(minV).Min(maxV)
	v143 := v133.Sub(v135).Max(minV).Min(maxV)
	v144 := v131.Sub(v137).Max(minV).Min(maxV)
	v145 := v116.Sub(v121).Max(minV).Min(maxV)
	v146 := v115.Sub(v122).Max(minV).Min(maxV)
	v147 := v130.Sub(v123).Max(minV).Min(maxV)
	v148 := v129.Sub(v124).Max(minV).Min(maxV)
	v149 := v132.Sub(v138).Max(minV).Min(maxV)
	v150 := v134.Sub(v136).Max(minV).Min(maxV)
	v151 := v134.Add(v136).Max(minV).Min(maxV)
	v152 := v132.Add(v138).Max(minV).Min(maxV)
	v153 := v129.Add(v124).Max(minV).Min(maxV)
	v154 := v130.Add(v123).Max(minV).Min(maxV)
	v155 := v150.Sub(v143).Mul(kp181).Shift(shr7).Average(v3)
	v156 := v150.Add(v143).Mul(kp181).Shift(shr7).Average(v3)
	v157 := v149.Sub(v144).Mul(kp181).Shift(shr7).Average(v3)
	v158 := v149.Add(v144).Mul(kp181).Shift(shr7).Average(v3)
	v159 := v148.Sub(v145).Mul(kp181).Shift(shr7).Average(v3)
	v160 := v148.Add(v145).Mul(kp181).Shift(shr7).Average(v3)
	v161 := v147.Sub(v146).Mul(kp181).Shift(shr7).Average(v3)
	v162 := v147.Add(v146).Mul(kp181).Shift(shr7).Average(v3)
	v163 := v59.Add(v154).Max(minV).Min(maxV)
	v164 := v60.Add(v153).Max(minV).Min(maxV)
	v165 := v61.Add(v152).Max(minV).Min(maxV)
	v166 := v62.Add(v151).Max(minV).Min(maxV)
	v167 := v63.Add(v156).Max(minV).Min(maxV)
	v168 := v64.Add(v158).Max(minV).Min(maxV)
	v169 := v65.Add(v160).Max(minV).Min(maxV)
	v170 := v66.Add(v162).Max(minV).Min(maxV)
	v171 := v67.Add(v161).Max(minV).Min(maxV)
	v172 := v68.Add(v159).Max(minV).Min(maxV)
	v173 := v69.Add(v157).Max(minV).Min(maxV)
	v174 := v70.Add(v155).Max(minV).Min(maxV)
	v175 := v71.Add(v142).Max(minV).Min(maxV)
	v176 := v72.Add(v141).Max(minV).Min(maxV)
	v177 := v73.Add(v140).Max(minV).Min(maxV)
	v178 := v74.Add(v139).Max(minV).Min(maxV)
	v179 := v74.Sub(v139).Max(minV).Min(maxV)
	v180 := v73.Sub(v140).Max(minV).Min(maxV)
	v181 := v72.Sub(v141).Max(minV).Min(maxV)
	v182 := v71.Sub(v142).Max(minV).Min(maxV)
	v183 := v70.Sub(v155).Max(minV).Min(maxV)
	v184 := v69.Sub(v157).Max(minV).Min(maxV)
	v185 := v68.Sub(v159).Max(minV).Min(maxV)
	v186 := v67.Sub(v161).Max(minV).Min(maxV)
	v187 := v66.Sub(v162).Max(minV).Min(maxV)
	v188 := v65.Sub(v160).Max(minV).Min(maxV)
	v189 := v64.Sub(v158).Max(minV).Min(maxV)
	v190 := v63.Sub(v156).Max(minV).Min(maxV)
	v191 := v62.Sub(v151).Max(minV).Min(maxV)
	v192 := v61.Sub(v152).Max(minV).Min(maxV)
	v193 := v60.Sub(v153).Max(minV).Min(maxV)
	v194 := v59.Sub(v154).Max(minV).Min(maxV)
	v163.StoreArray((*[4]int32)(buf[0*rowStride:]))
	v164.StoreArray((*[4]int32)(buf[1*rowStride:]))
	v165.StoreArray((*[4]int32)(buf[2*rowStride:]))
	v166.StoreArray((*[4]int32)(buf[3*rowStride:]))
	v167.StoreArray((*[4]int32)(buf[4*rowStride:]))
	v168.StoreArray((*[4]int32)(buf[5*rowStride:]))
	v169.StoreArray((*[4]int32)(buf[6*rowStride:]))
	v170.StoreArray((*[4]int32)(buf[7*rowStride:]))
	v171.StoreArray((*[4]int32)(buf[8*rowStride:]))
	v172.StoreArray((*[4]int32)(buf[9*rowStride:]))
	v173.StoreArray((*[4]int32)(buf[10*rowStride:]))
	v174.StoreArray((*[4]int32)(buf[11*rowStride:]))
	v175.StoreArray((*[4]int32)(buf[12*rowStride:]))
	v176.StoreArray((*[4]int32)(buf[13*rowStride:]))
	v177.StoreArray((*[4]int32)(buf[14*rowStride:]))
	v178.StoreArray((*[4]int32)(buf[15*rowStride:]))
	v179.StoreArray((*[4]int32)(buf[16*rowStride:]))
	v180.StoreArray((*[4]int32)(buf[17*rowStride:]))
	v181.StoreArray((*[4]int32)(buf[18*rowStride:]))
	v182.StoreArray((*[4]int32)(buf[19*rowStride:]))
	v183.StoreArray((*[4]int32)(buf[20*rowStride:]))
	v184.StoreArray((*[4]int32)(buf[21*rowStride:]))
	v185.StoreArray((*[4]int32)(buf[22*rowStride:]))
	v186.StoreArray((*[4]int32)(buf[23*rowStride:]))
	v187.StoreArray((*[4]int32)(buf[24*rowStride:]))
	v188.StoreArray((*[4]int32)(buf[25*rowStride:]))
	v189.StoreArray((*[4]int32)(buf[26*rowStride:]))
	v190.StoreArray((*[4]int32)(buf[27*rowStride:]))
	v191.StoreArray((*[4]int32)(buf[28*rowStride:]))
	v192.StoreArray((*[4]int32)(buf[29*rowStride:]))
	v193.StoreArray((*[4]int32)(buf[30*rowStride:]))
	v194.StoreArray((*[4]int32)(buf[31*rowStride:]))
}

// inverseDCT4Col4SIMD applies inverseDCT4 to 4 adjacent columns of buf in int32 lanes.
func inverseDCT4Col4SIMD(buf []int32, rowStride int, min, max int32) {
	minV := archsimd.BroadcastInt32x4(min)
	maxV := archsimd.BroadcastInt32x4(max)
	var zero archsimd.Int32x4
	km312 := archsimd.LoadInt32x4Array(&itxCm312)
	kp181 := archsimd.LoadInt32x4Array(&itxCp181)
	kp312 := archsimd.LoadInt32x4Array(&itxCp312)
	kp1567 := archsimd.LoadInt32x4Array(&itxCp1567)
	x0 := archsimd.LoadInt32x4Array((*[4]int32)(buf[0*rowStride:]))
	x1 := archsimd.LoadInt32x4Array((*[4]int32)(buf[1*rowStride:]))
	x2 := archsimd.LoadInt32x4Array((*[4]int32)(buf[2*rowStride:]))
	x3 := archsimd.LoadInt32x4Array((*[4]int32)(buf[3*rowStride:]))
	v1 := x0.Add(x2).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v2 := x0.Sub(x2).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v3 := x3.MulAdd(kp312, x1.Mul(kp1567)).Shift(itxShr(11)).Average(zero).Sub(x3)
	v4 := x3.MulAdd(kp1567, x1.Mul(km312)).Shift(itxShr(11)).Average(zero).Add(x1)
	v5 := v1.Add(v4).Max(minV).Min(maxV)
	v6 := v2.Add(v3).Max(minV).Min(maxV)
	v7 := v2.Sub(v3).Max(minV).Min(maxV)
	v8 := v1.Sub(v4).Max(minV).Min(maxV)
	v5.StoreArray((*[4]int32)(buf[0*rowStride:]))
	v6.StoreArray((*[4]int32)(buf[1*rowStride:]))
	v7.StoreArray((*[4]int32)(buf[2*rowStride:]))
	v8.StoreArray((*[4]int32)(buf[3*rowStride:]))
}

// inverseDCT8Col4SIMD applies inverseDCT8 to 4 adjacent columns of buf in int32 lanes.
func inverseDCT8Col4SIMD(buf []int32, rowStride int, min, max int32) {
	minV := archsimd.BroadcastInt32x4(min)
	maxV := archsimd.BroadcastInt32x4(max)
	var zero archsimd.Int32x4
	km1138 := archsimd.LoadInt32x4Array(&itxCm1138)
	km312 := archsimd.LoadInt32x4Array(&itxCm312)
	km79 := archsimd.LoadInt32x4Array(&itxCm79)
	kp79 := archsimd.LoadInt32x4Array(&itxCp79)
	kp181 := archsimd.LoadInt32x4Array(&itxCp181)
	kp312 := archsimd.LoadInt32x4Array(&itxCp312)
	kp799 := archsimd.LoadInt32x4Array(&itxCp799)
	kp1138 := archsimd.LoadInt32x4Array(&itxCp1138)
	kp1567 := archsimd.LoadInt32x4Array(&itxCp1567)
	kp1703 := archsimd.LoadInt32x4Array(&itxCp1703)
	x0 := archsimd.LoadInt32x4Array((*[4]int32)(buf[0*rowStride:]))
	x1 := archsimd.LoadInt32x4Array((*[4]int32)(buf[1*rowStride:]))
	x2 := archsimd.LoadInt32x4Array((*[4]int32)(buf[2*rowStride:]))
	x3 := archsimd.LoadInt32x4Array((*[4]int32)(buf[3*rowStride:]))
	x4 := archsimd.LoadInt32x4Array((*[4]int32)(buf[4*rowStride:]))
	x5 := archsimd.LoadInt32x4Array((*[4]int32)(buf[5*rowStride:]))
	x6 := archsimd.LoadInt32x4Array((*[4]int32)(buf[6*rowStride:]))
	x7 := archsimd.LoadInt32x4Array((*[4]int32)(buf[7*rowStride:]))
	v1 := x0.Add(x4).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v2 := x0.Sub(x4).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v3 := x6.MulAdd(kp312, x2.Mul(kp1567)).Shift(itxShr(11)).Average(zero).Sub(x6)
	v4 := x6.MulAdd(kp1567, x2.Mul(km312)).Shift(itxShr(11)).Average(zero).Add(x2)
	v5 := v1.Add(v4).Max(minV).Min(maxV)
	v6 := v2.Add(v3).Max(minV).Min(maxV)
	v7 := v2.Sub(v3).Max(minV).Min(maxV)
	v8 := v1.Sub(v4).Max(minV).Min(maxV)
	v9 := x7.MulAdd(kp79, x1.Mul(kp799)).Shift(itxShr(11)).Average(zero).Sub(x7)
	v10 := x3.MulAdd(km1138, x5.Mul(kp1703)).Shift(itxShr(10)).Average(zero)
	v11 := x3.MulAdd(kp1703, x5.Mul(kp1138)).Shift(itxShr(10)).Average(zero)
	v12 := x7.MulAdd(kp799, x1.Mul(km79)).Shift(itxShr(11)).Average(zero).Add(x1)
	v13 := v9.Add(v10).Max(minV).Min(maxV)
	v14 := v9.Sub(v10).Max(minV).Min(maxV)
	v15 := v12.Add(v11).Max(minV).Min(maxV)
	v16 := v12.Sub(v11).Max(minV).Min(maxV)
	v17 := v16.Sub(v14).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v18 := v16.Add(v14).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v19 := v5.Add(v15).Max(minV).Min(maxV)
	v20 := v6.Add(v18).Max(minV).Min(maxV)
	v21 := v7.Add(v17).Max(minV).Min(maxV)
	v22 := v8.Add(v13).Max(minV).Min(maxV)
	v23 := v8.Sub(v13).Max(minV).Min(maxV)
	v24 := v7.Sub(v17).Max(minV).Min(maxV)
	v25 := v6.Sub(v18).Max(minV).Min(maxV)
	v26 := v5.Sub(v15).Max(minV).Min(maxV)
	v19.StoreArray((*[4]int32)(buf[0*rowStride:]))
	v20.StoreArray((*[4]int32)(buf[1*rowStride:]))
	v21.StoreArray((*[4]int32)(buf[2*rowStride:]))
	v22.StoreArray((*[4]int32)(buf[3*rowStride:]))
	v23.StoreArray((*[4]int32)(buf[4*rowStride:]))
	v24.StoreArray((*[4]int32)(buf[5*rowStride:]))
	v25.StoreArray((*[4]int32)(buf[6*rowStride:]))
	v26.StoreArray((*[4]int32)(buf[7*rowStride:]))
}

// inverseDCT16Col4SIMD applies inverseDCT16 to 4 adjacent columns of buf in int32 lanes.
func inverseDCT16Col4SIMD(buf []int32, rowStride int, min, max int32) {
	minV := archsimd.BroadcastInt32x4(min)
	maxV := archsimd.BroadcastInt32x4(max)
	var zero archsimd.Int32x4
	km1299 := archsimd.LoadInt32x4Array(&itxCm1299)
	km1189 := archsimd.LoadInt32x4Array(&itxCm1189)
	km1138 := archsimd.LoadInt32x4Array(&itxCm1138)
	km484 := archsimd.LoadInt32x4Array(&itxCm484)
	km312 := archsimd.LoadInt32x4Array(&itxCm312)
	km176 := archsimd.LoadInt32x4Array(&itxCm176)
	km79 := archsimd.LoadInt32x4Array(&itxCm79)
	km20 := archsimd.LoadInt32x4Array(&itxCm20)
	kp20 := archsimd.LoadInt32x4Array(&itxCp20)
	kp79 := archsimd.LoadInt32x4Array(&itxCp79)
	kp181 := archsimd.LoadInt32x4Array(&itxCp181)
	kp312 := archsimd.LoadInt32x4Array(&itxCp312)
	kp401 := archsimd.LoadInt32x4Array(&itxCp401)
	kp484 := archsimd.LoadInt32x4Array(&itxCp484)
	kp799 := archsimd.LoadInt32x4Array(&itxCp799)
	kp1138 := archsimd.LoadInt32x4Array(&itxCp1138)
	kp1189 := archsimd.LoadInt32x4Array(&itxCp1189)
	kp1299 := archsimd.LoadInt32x4Array(&itxCp1299)
	kp1567 := archsimd.LoadInt32x4Array(&itxCp1567)
	kp1583 := archsimd.LoadInt32x4Array(&itxCp1583)
	kp1703 := archsimd.LoadInt32x4Array(&itxCp1703)
	kp1931 := archsimd.LoadInt32x4Array(&itxCp1931)
	x0 := archsimd.LoadInt32x4Array((*[4]int32)(buf[0*rowStride:]))
	x1 := archsimd.LoadInt32x4Array((*[4]int32)(buf[1*rowStride:]))
	x2 := archsimd.LoadInt32x4Array((*[4]int32)(buf[2*rowStride:]))
	x3 := archsimd.LoadInt32x4Array((*[4]int32)(buf[3*rowStride:]))
	x4 := archsimd.LoadInt32x4Array((*[4]int32)(buf[4*rowStride:]))
	x5 := archsimd.LoadInt32x4Array((*[4]int32)(buf[5*rowStride:]))
	x6 := archsimd.LoadInt32x4Array((*[4]int32)(buf[6*rowStride:]))
	x7 := archsimd.LoadInt32x4Array((*[4]int32)(buf[7*rowStride:]))
	x8 := archsimd.LoadInt32x4Array((*[4]int32)(buf[8*rowStride:]))
	x9 := archsimd.LoadInt32x4Array((*[4]int32)(buf[9*rowStride:]))
	x10 := archsimd.LoadInt32x4Array((*[4]int32)(buf[10*rowStride:]))
	x11 := archsimd.LoadInt32x4Array((*[4]int32)(buf[11*rowStride:]))
	x12 := archsimd.LoadInt32x4Array((*[4]int32)(buf[12*rowStride:]))
	x13 := archsimd.LoadInt32x4Array((*[4]int32)(buf[13*rowStride:]))
	x14 := archsimd.LoadInt32x4Array((*[4]int32)(buf[14*rowStride:]))
	x15 := archsimd.LoadInt32x4Array((*[4]int32)(buf[15*rowStride:]))
	v1 := x0.Add(x8).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v2 := x0.Sub(x8).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v3 := x12.MulAdd(kp312, x4.Mul(kp1567)).Shift(itxShr(11)).Average(zero).Sub(x12)
	v4 := x12.MulAdd(kp1567, x4.Mul(km312)).Shift(itxShr(11)).Average(zero).Add(x4)
	v5 := v1.Add(v4).Max(minV).Min(maxV)
	v6 := v2.Add(v3).Max(minV).Min(maxV)
	v7 := v2.Sub(v3).Max(minV).Min(maxV)
	v8 := v1.Sub(v4).Max(minV).Min(maxV)
	v9 := x14.MulAdd(kp79, x2.Mul(kp799)).Shift(itxShr(11)).Average(zero).Sub(x14)
	v10 := x6.MulAdd(km1138, x10.Mul(kp1703)).Shift(itxShr(10)).Average(zero)
	v11 := x6.MulAdd(kp1703, x10.Mul(kp1138)).Shift(itxShr(10)).Average(zero)
	v12 := x14.MulAdd(kp799, x2.Mul(km79)).Shift(itxShr(11)).Average(zero).Add(x2)
	v13 := v9.Add(v10).Max(minV).Min(maxV)
	v14 := v9.Sub(v10).Max(minV).Min(maxV)
	v15 := v12.Add(v11).Max(minV).Min(maxV)
	v16 := v12.Sub(v11).Max(minV).Min(maxV)
	v17 := v16.Sub(v14).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v18 := v16.Add(v14).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v19 := v5.Add(v15).Max(minV).Min(maxV)
	v20 := v6.Add(v18).Max(minV).Min(maxV)
	v21 := v7.Add(v17).Max(minV).Min(maxV)
	v22 := v8.Add(v13).Max(minV).Min(maxV)
	v23 := v8.Sub(v13).Max(minV).Min(maxV)
	v24 := v7.Sub(v17).Max(minV).Min(maxV)
	v25 := v6.Sub(v18).Max(minV).Min(maxV)
	v26 := v5.Sub(v15).Max(minV).Min(maxV)
	v27 := x15.MulAdd(kp20, x1.Mul(kp401)).Shift(itxShr(11)).Average(zero).Sub(x15)
	v28 := x7.MulAdd(km1299, x9.Mul(kp1583)).Shift(itxShr(10)).Average(zero)
	v29 := x11.MulAdd(kp484, x5.Mul(kp1931)).Shift(itxShr(11)).Average(zero).Sub(x11)
	v30 := x3.MulAdd(km1189, x13.Mul(km176)).Shift(itxShr(11)).Average(zero).Add(x13)
	v31 := x3.MulAdd(km176, x13.Mul(kp1189)).Shift(itxShr(11)).Average(zero).Add(x3)
	v32 := x11.MulAdd(kp1931, x5.Mul(km484)).Shift(itxShr(11)).Average(zero).Add(x5)
	v33 := x7.MulAdd(kp1583, x9.Mul(kp1299)).Shift(itxShr(10)).Average(zero)
	v34 := x15.MulAdd(kp401, x1.Mul(km20)).Shift(itxShr(11)).Average(zero).Add(x1)
	v35 := v27.Add(v28).Max(minV).Min(maxV)
	v36 := v27.Sub(v28).Max(minV).Min(maxV)
	v37 := v30.Sub(v29).Max(minV).Min(maxV)
	v38 := v30.Add(v29).Max(minV).Min(maxV)
	v39 := v31.Add(v32).Max(minV).Min(maxV)
	v40 := v31.Sub(v32).Max(minV).Min(maxV)
	v41 := v34.Sub(v33).Max(minV).Min(maxV)
	v42 := v34.Add(v33).Max(minV).Min(maxV)
	v43 := v36.MulAdd(kp312, v41.Mul(kp1567)).Shift(itxShr(11)).Average(zero).Sub(v36)
	v44 := v36.MulAdd(kp1567, v41.Mul(km312)).Shift(itxShr(11)).Average(zero).Add(v41)
	v45 := zero.Sub(v37.MulAdd(kp1567, v40.Mul(km312))).Shift(itxShr(11)).Average(zero).Sub(v40)
	v46 := v37.MulAdd(kp312, v40.Mul(kp1567)).Shift(itxShr(11)).Average(zero).Sub(v37)
	v47 := v35.Add(v38).Max(minV).Min(maxV)
	v48 := v43.Add(v45).Max(minV).Min(maxV)
	v49 := v43.Sub(v45).Max(minV).Min(maxV)
	v50 := v35.Sub(v38).Max(minV).Min(maxV)
	v51 := v42.Sub(v39).Max(minV).Min(maxV)
	v52 := v44.Sub(v46).Max(minV).Min(maxV)
	v53 := v44.Add(v46).Max(minV).Min(maxV)
	v54 := v42.Add(v39).Max(minV).Min(maxV)
	v55 := v52.Sub(v49).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v56 := v52.Add(v49).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v57 := v51.Sub(v50).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v58 := v51.Add(v50).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v59 := v19.Add(v54).Max(minV).Min(maxV)
	v60 := v20.Add(v53).Max(minV).Min(maxV)
	v61 := v21.Add(v56).Max(minV).Min(maxV)
	v62 := v22.Add(v58).Max(minV).Min(maxV)
	v63 := v23.Add(v57).Max(minV).Min(maxV)
	v64 := v24.Add(v55).Max(minV).Min(maxV)
	v65 := v25.Add(v48).Max(minV).Min(maxV)
	v66 := v26.Add(v47).Max(minV).Min(maxV)
	v67 := v26.Sub(v47).Max(minV).Min(maxV)
	v68 := v25.Sub(v48).Max(minV).Min(maxV)
	v69 := v24.Sub(v55).Max(minV).Min(maxV)
	v70 := v23.Sub(v57).Max(minV).Min(maxV)
	v71 := v22.Sub(v58).Max(minV).Min(maxV)
	v72 := v21.Sub(v56).Max(minV).Min(maxV)
	v73 := v20.Sub(v53).Max(minV).Min(maxV)
	v74 := v19.Sub(v54).Max(minV).Min(maxV)
	v59.StoreArray((*[4]int32)(buf[0*rowStride:]))
	v60.StoreArray((*[4]int32)(buf[1*rowStride:]))
	v61.StoreArray((*[4]int32)(buf[2*rowStride:]))
	v62.StoreArray((*[4]int32)(buf[3*rowStride:]))
	v63.StoreArray((*[4]int32)(buf[4*rowStride:]))
	v64.StoreArray((*[4]int32)(buf[5*rowStride:]))
	v65.StoreArray((*[4]int32)(buf[6*rowStride:]))
	v66.StoreArray((*[4]int32)(buf[7*rowStride:]))
	v67.StoreArray((*[4]int32)(buf[8*rowStride:]))
	v68.StoreArray((*[4]int32)(buf[9*rowStride:]))
	v69.StoreArray((*[4]int32)(buf[10*rowStride:]))
	v70.StoreArray((*[4]int32)(buf[11*rowStride:]))
	v71.StoreArray((*[4]int32)(buf[12*rowStride:]))
	v72.StoreArray((*[4]int32)(buf[13*rowStride:]))
	v73.StoreArray((*[4]int32)(buf[14*rowStride:]))
	v74.StoreArray((*[4]int32)(buf[15*rowStride:]))
}

// inverseDCT32Col4SIMD applies inverseDCT32 to 4 adjacent columns of buf in int32 lanes.
func inverseDCT32Col4SIMD(buf []int32, rowStride int, min, max int32) {
	minV := archsimd.BroadcastInt32x4(min)
	maxV := archsimd.BroadcastInt32x4(max)
	var zero archsimd.Int32x4
	km2751 := archsimd.LoadInt32x4Array(&itxCm2751)
	km2106 := archsimd.LoadInt32x4Array(&itxCm2106)
	km1645 := archsimd.LoadInt32x4Array(&itxCm1645)
	km1380 := archsimd.LoadInt32x4Array(&itxCm1380)
	km1299 := archsimd.LoadInt32x4Array(&itxCm1299)
	km1189 := archsimd.LoadInt32x4Array(&itxCm1189)
	km1138 := archsimd.LoadInt32x4Array(&itxCm1138)
	km1061 := archsimd.LoadInt32x4Array(&itxCm1061)
	km601 := archsimd.LoadInt32x4Array(&itxCm601)
	km583 := archsimd.LoadInt32x4Array(&itxCm583)
	km484 := archsimd.LoadInt32x4Array(&itxCm484)
	km393 := archsimd.LoadInt32x4Array(&itxCm393)
	km312 := archsimd.LoadInt32x4Array(&itxCm312)
	km239 := archsimd.LoadInt32x4Array(&itxCm239)
	km176 := archsimd.LoadInt32x4Array(&itxCm176)
	km123 := archsimd.LoadInt32x4Array(&itxCm123)
	km79 := archsimd.LoadInt32x4Array(&itxCm79)
	km44 := archsimd.LoadInt32x4Array(&itxCm44)
	km20 := archsimd.LoadInt32x4Array(&itxCm20)
	km5 := archsimd.LoadInt32x4Array(&itxCm5)
	kp5 := archsimd.LoadInt32x4Array(&itxCp5)
	kp20 := archsimd.LoadInt32x4Array(&itxCp20)
	kp79 := archsimd.LoadInt32x4Array(&itxCp79)
	kp123 := archsimd.LoadInt32x4Array(&itxCp123)
	kp181 := archsimd.LoadInt32x4Array(&itxCp181)
	kp201 := archsimd.LoadInt32x4Array(&itxCp201)
	kp312 := archsimd.LoadInt32x4Array(&itxCp312)
	kp393 := archsimd.LoadInt32x4Array(&itxCp393)
	kp401 := archsimd.LoadInt32x4Array(&itxCp401)
	kp484 := archsimd.LoadInt32x4Array(&itxCp484)
	kp601 := archsimd.LoadInt32x4Array(&itxCp601)
	kp799 := archsimd.LoadInt32x4Array(&itxCp799)
	kp995 := archsimd.LoadInt32x4Array(&itxCp995)
	kp1138 := archsimd.LoadInt32x4Array(&itxCp1138)
	kp1189 := archsimd.LoadInt32x4Array(&itxCp1189)
	kp1220 := archsimd.LoadInt32x4Array(&itxCp1220)
	kp1299 := archsimd.LoadInt32x4Array(&itxCp1299)
	kp1380 := archsimd.LoadInt32x4Array(&itxCp1380)
	kp1567 := archsimd.LoadInt32x4Array(&itxCp1567)
	kp1583 := archsimd.LoadInt32x4Array(&itxCp1583)
	kp1645 := archsimd.LoadInt32x4Array(&itxCp1645)
	kp1703 := archsimd.LoadInt32x4Array(&itxCp1703)
	kp1751 := archsimd.LoadInt32x4Array(&itxCp1751)
	kp1931 := archsimd.LoadInt32x4Array(&itxCp1931)
	kp2106 := archsimd.LoadInt32x4Array(&itxCp2106)
	kp2751 := archsimd.LoadInt32x4Array(&itxCp2751)
	x0 := archsimd.LoadInt32x4Array((*[4]int32)(buf[0*rowStride:]))
	x1 := archsimd.LoadInt32x4Array((*[4]int32)(buf[1*rowStride:]))
	x2 := archsimd.LoadInt32x4Array((*[4]int32)(buf[2*rowStride:]))
	x3 := archsimd.LoadInt32x4Array((*[4]int32)(buf[3*rowStride:]))
	x4 := archsimd.LoadInt32x4Array((*[4]int32)(buf[4*rowStride:]))
	x5 := archsimd.LoadInt32x4Array((*[4]int32)(buf[5*rowStride:]))
	x6 := archsimd.LoadInt32x4Array((*[4]int32)(buf[6*rowStride:]))
	x7 := archsimd.LoadInt32x4Array((*[4]int32)(buf[7*rowStride:]))
	x8 := archsimd.LoadInt32x4Array((*[4]int32)(buf[8*rowStride:]))
	x9 := archsimd.LoadInt32x4Array((*[4]int32)(buf[9*rowStride:]))
	x10 := archsimd.LoadInt32x4Array((*[4]int32)(buf[10*rowStride:]))
	x11 := archsimd.LoadInt32x4Array((*[4]int32)(buf[11*rowStride:]))
	x12 := archsimd.LoadInt32x4Array((*[4]int32)(buf[12*rowStride:]))
	x13 := archsimd.LoadInt32x4Array((*[4]int32)(buf[13*rowStride:]))
	x14 := archsimd.LoadInt32x4Array((*[4]int32)(buf[14*rowStride:]))
	x15 := archsimd.LoadInt32x4Array((*[4]int32)(buf[15*rowStride:]))
	x16 := archsimd.LoadInt32x4Array((*[4]int32)(buf[16*rowStride:]))
	x17 := archsimd.LoadInt32x4Array((*[4]int32)(buf[17*rowStride:]))
	x18 := archsimd.LoadInt32x4Array((*[4]int32)(buf[18*rowStride:]))
	x19 := archsimd.LoadInt32x4Array((*[4]int32)(buf[19*rowStride:]))
	x20 := archsimd.LoadInt32x4Array((*[4]int32)(buf[20*rowStride:]))
	x21 := archsimd.LoadInt32x4Array((*[4]int32)(buf[21*rowStride:]))
	x22 := archsimd.LoadInt32x4Array((*[4]int32)(buf[22*rowStride:]))
	x23 := archsimd.LoadInt32x4Array((*[4]int32)(buf[23*rowStride:]))
	x24 := archsimd.LoadInt32x4Array((*[4]int32)(buf[24*rowStride:]))
	x25 := archsimd.LoadInt32x4Array((*[4]int32)(buf[25*rowStride:]))
	x26 := archsimd.LoadInt32x4Array((*[4]int32)(buf[26*rowStride:]))
	x27 := archsimd.LoadInt32x4Array((*[4]int32)(buf[27*rowStride:]))
	x28 := archsimd.LoadInt32x4Array((*[4]int32)(buf[28*rowStride:]))
	x29 := archsimd.LoadInt32x4Array((*[4]int32)(buf[29*rowStride:]))
	x30 := archsimd.LoadInt32x4Array((*[4]int32)(buf[30*rowStride:]))
	x31 := archsimd.LoadInt32x4Array((*[4]int32)(buf[31*rowStride:]))
	v1 := x0.Add(x16).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v2 := x0.Sub(x16).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v3 := x24.MulAdd(kp312, x8.Mul(kp1567)).Shift(itxShr(11)).Average(zero).Sub(x24)
	v4 := x24.MulAdd(kp1567, x8.Mul(km312)).Shift(itxShr(11)).Average(zero).Add(x8)
	v5 := v1.Add(v4).Max(minV).Min(maxV)
	v6 := v2.Add(v3).Max(minV).Min(maxV)
	v7 := v2.Sub(v3).Max(minV).Min(maxV)
	v8 := v1.Sub(v4).Max(minV).Min(maxV)
	v9 := x28.MulAdd(kp79, x4.Mul(kp799)).Shift(itxShr(11)).Average(zero).Sub(x28)
	v10 := x12.MulAdd(km1138, x20.Mul(kp1703)).Shift(itxShr(10)).Average(zero)
	v11 := x12.MulAdd(kp1703, x20.Mul(kp1138)).Shift(itxShr(10)).Average(zero)
	v12 := x28.MulAdd(kp799, x4.Mul(km79)).Shift(itxShr(11)).Average(zero).Add(x4)
	v13 := v9.Add(v10).Max(minV).Min(maxV)
	v14 := v9.Sub(v10).Max(minV).Min(maxV)
	v15 := v12.Add(v11).Max(minV).Min(maxV)
	v16 := v12.Sub(v11).Max(minV).Min(maxV)
	v17 := v16.Sub(v14).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v18 := v16.Add(v14).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v19 := v5.Add(v15).Max(minV).Min(maxV)
	v20 := v6.Add(v18).Max(minV).Min(maxV)
	v21 := v7.Add(v17).Max(minV).Min(maxV)
	v22 := v8.Add(v13).Max(minV).Min(maxV)
	v23 := v8.Sub(v13).Max(minV).Min(maxV)
	v24 := v7.Sub(v17).Max(minV).Min(maxV)
	v25 := v6.Sub(v18).Max(minV).Min(maxV)
	v26 := v5.Sub(v15).Max(minV).Min(maxV)
	v27 := x30.MulAdd(kp20, x2.Mul(kp401)).Shift(itxShr(11)).Average(zero).Sub(x30)
	v28 := x14.MulAdd(km1299, x18.Mul(kp1583)).Shift(itxShr(10)).Average(zero)
	v29 := x22.MulAdd(kp484, x10.Mul(kp1931)).Shift(itxShr(11)).Average(zero).Sub(x22)
	v30 := x6.MulAdd(km1189, x26.Mul(km176)).Shift(itxShr(11)).Average(zero).Add(x26)
	v31 := x6.MulAdd(km176, x26.Mul(kp1189)).Shift(itxShr(11)).Average(zero).Add(x6)
	v32 := x22.MulAdd(kp1931, x10.Mul(km484)).Shift(itxShr(11)).Average(zero).Add(x10)
	v33 := x14.MulAdd(kp1583, x18.Mul(kp1299)).Shift(itxShr(10)).Average(zero)
	v34 := x30.MulAdd(kp401, x2.Mul(km20)).Shift(itxShr(11)).Average(zero).Add(x2)
	v35 := v27.Add(v28).Max(minV).Min(maxV)
	v36 := v27.Sub(v28).Max(minV).Min(maxV)
	v37 := v30.Sub(v29).Max(minV).Min(maxV)
	v38 := v30.Add(v29).Max(minV).Min(maxV)
	v39 := v31.Add(v32).Max(minV).Min(maxV)
	v40 := v31.Sub(v32).Max(minV).Min(maxV)
	v41 := v34.Sub(v33).Max(minV).Min(maxV)
	v42 := v34.Add(v33).Max(minV).Min(maxV)
	v43 := v36.MulAdd(kp312, v41.Mul(kp1567)).Shift(itxShr(11)).Average(zero).Sub(v36)
	v44 := v36.MulAdd(kp1567, v41.Mul(km312)).Shift(itxShr(11)).Average(zero).Add(v41)
	v45 := zero.Sub(v37.MulAdd(kp1567, v40.Mul(km312))).Shift(itxShr(11)).Average(zero).Sub(v40)
	v46 := v37.MulAdd(kp312, v40.Mul(kp1567)).Shift(itxShr(11)).Average(zero).Sub(v37)
	v47 := v35.Add(v38).Max(minV).Min(maxV)
	v48 := v43.Add(v45).Max(minV).Min(maxV)
	v49 := v43.Sub(v45).Max(minV).Min(maxV)
	v50 := v35.Sub(v38).Max(minV).Min(maxV)
	v51 := v42.Sub(v39).Max(minV).Min(maxV)
	v52 := v44.Sub(v46).Max(minV).Min(maxV)
	v53 := v44.Add(v46).Max(minV).Min(maxV)
	v54 := v42.Add(v39).Max(minV).Min(maxV)
	v55 := v52.Sub(v49).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v56 := v52.Add(v49).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v57 := v51.Sub(v50).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v58 := v51.Add(v50).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v59 := v19.Add(v54).Max(minV).Min(maxV)
	v60 := v20.Add(v53).Max(minV).Min(maxV)
	v61 := v21.Add(v56).Max(minV).Min(maxV)
	v62 := v22.Add(v58).Max(minV).Min(maxV)
	v63 := v23.Add(v57).Max(minV).Min(maxV)
	v64 := v24.Add(v55).Max(minV).Min(maxV)
	v65 := v25.Add(v48).Max(minV).Min(maxV)
	v66 := v26.Add(v47).Max(minV).Min(maxV)
	v67 := v26.Sub(v47).Max(minV).Min(maxV)
	v68 := v25.Sub(v48).Max(minV).Min(maxV)
	v69 := v24.Sub(v55).Max(minV).Min(maxV)
	v70 := v23.Sub(v57).Max(minV).Min(maxV)
	v71 := v22.Sub(v58).Max(minV).Min(maxV)
	v72 := v21.Sub(v56).Max(minV).Min(maxV)
	v73 := v20.Sub(v53).Max(minV).Min(maxV)
	v74 := v19.Sub(v54).Max(minV).Min(maxV)
	v75 := x31.MulAdd(kp5, x1.Mul(kp201)).Shift(itxShr(11)).Average(zero).Sub(x31)
	v76 := x15.MulAdd(km2751, x17.Mul(km1061)).Shift(itxShr(11)).Average(zero).Add(x17)
	v77 := x23.MulAdd(kp393, x9.Mul(kp1751)).Shift(itxShr(11)).Average(zero).Sub(x23)
	v78 := x7.MulAdd(km1380, x25.Mul(km239)).Shift(itxShr(11)).Average(zero).Add(x25)
	v79 := x27.MulAdd(kp123, x5.Mul(kp995)).Shift(itxShr(11)).Average(zero).Sub(x27)
	v80 := x11.MulAdd(km2106, x21.Mul(km583)).Shift(itxShr(11)).Average(zero).Add(x21)
	v81 := x19.MulAdd(km1645, x13.Mul(kp1220)).Shift(itxShr(10)).Average(zero)
	v82 := x3.MulAdd(km601, x29.Mul(km44)).Shift(itxShr(11)).Average(zero).Add(x29)
	v83 := x3.MulAdd(km44, x29.Mul(kp601)).Shift(itxShr(11)).Average(zero).Add(x3)
	v84 := x19.MulAdd(kp1220, x13.Mul(kp1645)).Shift(itxShr(10)).Average(zero)
	v85 := x11.MulAdd(km583, x21.Mul(kp2106)).Shift(itxShr(11)).Average(zero).Add(x11)
	v86 := x27.MulAdd(kp995, x5.Mul(km123)).Shift(itxShr(11)).Average(zero).Add(x5)
	v87 := x7.MulAdd(km239, x25.Mul(kp1380)).Shift(itxShr(11)).Average(zero).Add(x7)
	v88 := x23.MulAdd(kp1751, x9.Mul(km393)).Shift(itxShr(11)).Average(zero).Add(x9)
	v89 := x15.MulAdd(km1061, x17.Mul(kp2751)).Shift(itxShr(11)).Average(zero).Add(x15)
	v90 := x31.MulAdd(kp201, x1.Mul(km5)).Shift(itxShr(11)).Average(zero).Add(x1)
	v91 := v75.Add(v76).Max(minV).Min(maxV)
	v92 := v75.Sub(v76).Max(minV).Min(maxV)
	v93 := v78.Sub(v77).Max(minV).Min(maxV)
	v94 := v78.Add(v77).Max(minV).Min(maxV)
	v95 := v79.Add(v80).Max(minV).Min(maxV)
	v96 := v79.Sub(v80).Max(minV).Min(maxV)
	v97 := v82.Sub(v81).Max(minV).Min(maxV)
	v98 := v82.Add(v81).Max(minV).Min(maxV)
	v99 := v83.Add(v84).Max(minV).Min(maxV)
	v100 := v83.Sub(v84).Max(minV).Min(maxV)
	v101 := v86.Sub(v85).Max(minV).Min(maxV)
	v102 := v86.Add(v85).Max(minV).Min(maxV)
	v103 := v87.Add(v88).Max(minV).Min(maxV)
	v104 := v87.Sub(v88).Max(minV).Min(maxV)
	v105 := v90.Sub(v89).Max(minV).Min(maxV)
	v106 := v90.Add(v89).Max(minV).Min(maxV)
	v107 := v92.MulAdd(kp79, v105.Mul(kp799)).Shift(itxShr(11)).Average(zero).Sub(v92)
	v108 := v92.MulAdd(kp799, v105.Mul(km79)).Shift(itxShr(11)).Average(zero).Add(v105)
	v109 := zero.Sub(v93.MulAdd(kp799, v104.Mul(km79))).Shift(itxShr(11)).Average(zero).Sub(v104)
	v110 := v93.MulAdd(kp79, v104.Mul(kp799)).Shift(itxShr(11)).Average(zero).Sub(v93)
	v111 := v96.MulAdd(km1138, v101.Mul(kp1703)).Shift(itxShr(10)).Average(zero)
	v112 := v96.MulAdd(kp1703, v101.Mul(kp1138)).Shift(itxShr(10)).Average(zero)
	v113 := zero.Sub(v97.MulAdd(kp1703, v100.Mul(kp1138))).Shift(itxShr(10)).Average(zero)
	v114 := v97.MulAdd(km1138, v100.Mul(kp1703)).Shift(itxShr(10)).Average(zero)
	v115 := v91.Add(v94).Max(minV).Min(maxV)
	v116 := v107.Add(v109).Max(minV).Min(maxV)
	v117 := v107.Sub(v109).Max(minV).Min(maxV)
	v118 := v91.Sub(v94).Max(minV).Min(maxV)
	v119 := v98.Sub(v95).Max(minV).Min(maxV)
	v120 := v113.Sub(v111).Max(minV).Min(maxV)
	v121 := v113.Add(v111).Max(minV).Min(maxV)
	v122 := v98.Add(v95).Max(minV).Min(maxV)
	v123 := v99.Add(v102).Max(minV).Min(maxV)
	v124 := v114.Add(v112).Max(minV).Min(maxV)
	v125 := v114.Sub(v112).Max(minV).Min(maxV)
	v126 := v99.Sub(v102).Max(minV).Min(maxV)
	v127 := v106.Sub(v103).Max(minV).Min(maxV)
	v128 := v108.Sub(v110).Max(minV).Min(maxV)
	v129 := v108.Add(v110).Max(minV).Min(maxV)
	v130 := v106.Add(v103).Max(minV).Min(maxV)
	v131 := v117.MulAdd(kp312, v128.Mul(kp1567)).Shift(itxShr(11)).Average(zero).Sub(v117)
	v132 := v117.MulAdd(kp1567, v128.Mul(km312)).Shift(itxShr(11)).Average(zero).Add(v128)
	v133 := v118.MulAdd(kp312, v127.Mul(kp1567)).Shift(itxShr(11)).Average(zero).Sub(v118)
	v134 := v118.MulAdd(kp1567, v127.Mul(km312)).Shift(itxShr(11)).Average(zero).Add(v127)
	v135 := zero.Sub(v119.MulAdd(kp1567, v126.Mul(km312))).Shift(itxShr(11)).Average(zero).Sub(v126)
	v136 := v119.MulAdd(kp312, v126.Mul(kp1567)).Shift(itxShr(11)).Average(zero).Sub(v119)
	v137 := zero.Sub(v120.MulAdd(kp1567, v125.Mul(km312))).Shift(itxShr(11)).Average(zero).Sub(v125)
	v138 := v120.MulAdd(kp312, v125.Mul(kp1567)).Shift(itxShr(11)).Average(zero).Sub(v120)
	v139 := v115.Add(v122).Max(minV).Min(maxV)
	v140 := v116.Add(v121).Max(minV).Min(maxV)
	v141 := v131.Add(v137).Max(minV).Min(maxV)
	v142 := v133.Add(v135).Max(minV).Min(maxV)
	v143 := v133.Sub(v135).Max(minV).Min(maxV)
	v144 := v131.Sub(v137).Max(minV).Min(maxV)
	v145 := v116.Sub(v121).Max(minV).Min(maxV)
	v146 := v115.Sub(v122).Max(minV).Min(maxV)
	v147 := v130.Sub(v123).Max(minV).Min(maxV)
	v148 := v129.Sub(v124).Max(minV).Min(maxV)
	v149 := v132.Sub(v138).Max(minV).Min(maxV)
	v150 := v134.Sub(v136).Max(minV).Min(maxV)
	v151 := v134.Add(v136).Max(minV).Min(maxV)
	v152 := v132.Add(v138).Max(minV).Min(maxV)
	v153 := v129.Add(v124).Max(minV).Min(maxV)
	v154 := v130.Add(v123).Max(minV).Min(maxV)
	v155 := v150.Sub(v143).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v156 := v150.Add(v143).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v157 := v149.Sub(v144).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v158 := v149.Add(v144).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v159 := v148.Sub(v145).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v160 := v148.Add(v145).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v161 := v147.Sub(v146).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v162 := v147.Add(v146).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v163 := v59.Add(v154).Max(minV).Min(maxV)
	v164 := v60.Add(v153).Max(minV).Min(maxV)
	v165 := v61.Add(v152).Max(minV).Min(maxV)
	v166 := v62.Add(v151).Max(minV).Min(maxV)
	v167 := v63.Add(v156).Max(minV).Min(maxV)
	v168 := v64.Add(v158).Max(minV).Min(maxV)
	v169 := v65.Add(v160).Max(minV).Min(maxV)
	v170 := v66.Add(v162).Max(minV).Min(maxV)
	v171 := v67.Add(v161).Max(minV).Min(maxV)
	v172 := v68.Add(v159).Max(minV).Min(maxV)
	v173 := v69.Add(v157).Max(minV).Min(maxV)
	v174 := v70.Add(v155).Max(minV).Min(maxV)
	v175 := v71.Add(v142).Max(minV).Min(maxV)
	v176 := v72.Add(v141).Max(minV).Min(maxV)
	v177 := v73.Add(v140).Max(minV).Min(maxV)
	v178 := v74.Add(v139).Max(minV).Min(maxV)
	v179 := v74.Sub(v139).Max(minV).Min(maxV)
	v180 := v73.Sub(v140).Max(minV).Min(maxV)
	v181 := v72.Sub(v141).Max(minV).Min(maxV)
	v182 := v71.Sub(v142).Max(minV).Min(maxV)
	v183 := v70.Sub(v155).Max(minV).Min(maxV)
	v184 := v69.Sub(v157).Max(minV).Min(maxV)
	v185 := v68.Sub(v159).Max(minV).Min(maxV)
	v186 := v67.Sub(v161).Max(minV).Min(maxV)
	v187 := v66.Sub(v162).Max(minV).Min(maxV)
	v188 := v65.Sub(v160).Max(minV).Min(maxV)
	v189 := v64.Sub(v158).Max(minV).Min(maxV)
	v190 := v63.Sub(v156).Max(minV).Min(maxV)
	v191 := v62.Sub(v151).Max(minV).Min(maxV)
	v192 := v61.Sub(v152).Max(minV).Min(maxV)
	v193 := v60.Sub(v153).Max(minV).Min(maxV)
	v194 := v59.Sub(v154).Max(minV).Min(maxV)
	v163.StoreArray((*[4]int32)(buf[0*rowStride:]))
	v164.StoreArray((*[4]int32)(buf[1*rowStride:]))
	v165.StoreArray((*[4]int32)(buf[2*rowStride:]))
	v166.StoreArray((*[4]int32)(buf[3*rowStride:]))
	v167.StoreArray((*[4]int32)(buf[4*rowStride:]))
	v168.StoreArray((*[4]int32)(buf[5*rowStride:]))
	v169.StoreArray((*[4]int32)(buf[6*rowStride:]))
	v170.StoreArray((*[4]int32)(buf[7*rowStride:]))
	v171.StoreArray((*[4]int32)(buf[8*rowStride:]))
	v172.StoreArray((*[4]int32)(buf[9*rowStride:]))
	v173.StoreArray((*[4]int32)(buf[10*rowStride:]))
	v174.StoreArray((*[4]int32)(buf[11*rowStride:]))
	v175.StoreArray((*[4]int32)(buf[12*rowStride:]))
	v176.StoreArray((*[4]int32)(buf[13*rowStride:]))
	v177.StoreArray((*[4]int32)(buf[14*rowStride:]))
	v178.StoreArray((*[4]int32)(buf[15*rowStride:]))
	v179.StoreArray((*[4]int32)(buf[16*rowStride:]))
	v180.StoreArray((*[4]int32)(buf[17*rowStride:]))
	v181.StoreArray((*[4]int32)(buf[18*rowStride:]))
	v182.StoreArray((*[4]int32)(buf[19*rowStride:]))
	v183.StoreArray((*[4]int32)(buf[20*rowStride:]))
	v184.StoreArray((*[4]int32)(buf[21*rowStride:]))
	v185.StoreArray((*[4]int32)(buf[22*rowStride:]))
	v186.StoreArray((*[4]int32)(buf[23*rowStride:]))
	v187.StoreArray((*[4]int32)(buf[24*rowStride:]))
	v188.StoreArray((*[4]int32)(buf[25*rowStride:]))
	v189.StoreArray((*[4]int32)(buf[26*rowStride:]))
	v190.StoreArray((*[4]int32)(buf[27*rowStride:]))
	v191.StoreArray((*[4]int32)(buf[28*rowStride:]))
	v192.StoreArray((*[4]int32)(buf[29*rowStride:]))
	v193.StoreArray((*[4]int32)(buf[30*rowStride:]))
	v194.StoreArray((*[4]int32)(buf[31*rowStride:]))
}

// inverseADST16Col4SIMD applies inverseADST16 to 4 adjacent columns of buf in int32 lanes.
func inverseADST16Col4SIMD(buf []int32, rowStride int, min, max int32) {
	minV := archsimd.BroadcastInt32x4(min)
	maxV := archsimd.BroadcastInt32x4(max)
	var zero archsimd.Int32x4
	km2751 := archsimd.LoadInt32x4Array(&itxCm2751)
	km2276 := archsimd.LoadInt32x4Array(&itxCm2276)
	km2106 := archsimd.LoadInt32x4Array(&itxCm2106)
	km1645 := archsimd.LoadInt32x4Array(&itxCm1645)
	km1567 := archsimd.LoadInt32x4Array(&itxCm1567)
	km1380 := archsimd.LoadInt32x4Array(&itxCm1380)
	km1061 := archsimd.LoadInt32x4Array(&itxCm1061)
	km799 := archsimd.LoadInt32x4Array(&itxCm799)
	km690 := archsimd.LoadInt32x4Array(&itxCm690)
	km601 := archsimd.LoadInt32x4Array(&itxCm601)
	km583 := archsimd.LoadInt32x4Array(&itxCm583)
	km393 := archsimd.LoadInt32x4Array(&itxCm393)
	km312 := archsimd.LoadInt32x4Array(&itxCm312)
	km239 := archsimd.LoadInt32x4Array(&itxCm239)
	km123 := archsimd.LoadInt32x4Array(&itxCm123)
	km79 := archsimd.LoadInt32x4Array(&itxCm79)
	km44 := archsimd.LoadInt32x4Array(&itxCm44)
	km5 := archsimd.LoadInt32x4Array(&itxCm5)
	kp5 := archsimd.LoadInt32x4Array(&itxCp5)
	kp79 := archsimd.LoadInt32x4Array(&itxCp79)
	kp123 := archsimd.LoadInt32x4Array(&itxCp123)
	kp181 := archsimd.LoadInt32x4Array(&itxCp181)
	kp201 := archsimd.LoadInt32x4Array(&itxCp201)
	kp312 := archsimd.LoadInt32x4Array(&itxCp312)
	kp393 := archsimd.LoadInt32x4Array(&itxCp393)
	kp601 := archsimd.LoadInt32x4Array(&itxCp601)
	kp690 := archsimd.LoadInt32x4Array(&itxCp690)
	kp799 := archsimd.LoadInt32x4Array(&itxCp799)
	kp995 := archsimd.LoadInt32x4Array(&itxCp995)
	kp1220 := archsimd.LoadInt32x4Array(&itxCp1220)
	kp1380 := archsimd.LoadInt32x4Array(&itxCp1380)
	kp1567 := archsimd.LoadInt32x4Array(&itxCp1567)
	kp1645 := archsimd.LoadInt32x4Array(&itxCp1645)
	kp1751 := archsimd.LoadInt32x4Array(&itxCp1751)
	kp2106 := archsimd.LoadInt32x4Array(&itxCp2106)
	kp2276 := archsimd.LoadInt32x4Array(&itxCp2276)
	kp2751 := archsimd.LoadInt32x4Array(&itxCp2751)
	x0 := archsimd.LoadInt32x4Array((*[4]int32)(buf[0*rowStride:]))
	x1 := archsimd.LoadInt32x4Array((*[4]int32)(buf[1*rowStride:]))
	x2 := archsimd.LoadInt32x4Array((*[4]int32)(buf[2*rowStride:]))
	x3 := archsimd.LoadInt32x4Array((*[4]int32)(buf[3*rowStride:]))
	x4 := archsimd.LoadInt32x4Array((*[4]int32)(buf[4*rowStride:]))
	x5 := archsimd.LoadInt32x4Array((*[4]int32)(buf[5*rowStride:]))
	x6 := archsimd.LoadInt32x4Array((*[4]int32)(buf[6*rowStride:]))
	x7 := archsimd.LoadInt32x4Array((*[4]int32)(buf[7*rowStride:]))
	x8 := archsimd.LoadInt32x4Array((*[4]int32)(buf[8*rowStride:]))
	x9 := archsimd.LoadInt32x4Array((*[4]int32)(buf[9*rowStride:]))
	x10 := archsimd.LoadInt32x4Array((*[4]int32)(buf[10*rowStride:]))
	x11 := archsimd.LoadInt32x4Array((*[4]int32)(buf[11*rowStride:]))
	x12 := archsimd.LoadInt32x4Array((*[4]int32)(buf[12*rowStride:]))
	x13 := archsimd.LoadInt32x4Array((*[4]int32)(buf[13*rowStride:]))
	x14 := archsimd.LoadInt32x4Array((*[4]int32)(buf[14*rowStride:]))
	x15 := archsimd.LoadInt32x4Array((*[4]int32)(buf[15*rowStride:]))
	v1 := x0.MulAdd(kp201, x15.Mul(km5)).Shift(itxShr(11)).Average(zero).Add(x15)
	v2 := x0.MulAdd(kp5, x15.Mul(kp201)).Shift(itxShr(11)).Average(zero).Sub(x0)
	v3 := x2.MulAdd(kp995, x13.Mul(km123)).Shift(itxShr(11)).Average(zero).Add(x13)
	v4 := x2.MulAdd(kp123, x13.Mul(kp995)).Shift(itxShr(11)).Average(zero).Sub(x2)
	v5 := x4.MulAdd(kp1751, x11.Mul(km393)).Shift(itxShr(11)).Average(zero).Add(x11)
	v6 := x4.MulAdd(kp393, x11.Mul(kp1751)).Shift(itxShr(11)).Average(zero).Sub(x4)
	v7 := x6.MulAdd(kp1220, x9.Mul(kp1645)).Shift(itxShr(10)).Average(zero)
	v8 := x6.MulAdd(km1645, x9.Mul(kp1220)).Shift(itxShr(10)).Average(zero)
	v9 := x8.MulAdd(km1061, x7.Mul(kp2751)).Shift(itxShr(11)).Average(zero).Add(x8)
	v10 := x8.MulAdd(km2751, x7.Mul(km1061)).Shift(itxShr(11)).Average(zero).Add(x7)
	v11 := x10.MulAdd(km583, x5.Mul(kp2106)).Shift(itxShr(11)).Average(zero).Add(x10)
	v12 := x10.MulAdd(km2106, x5.Mul(km583)).Shift(itxShr(11)).Average(zero).Add(x5)
	v13 := x12.MulAdd(km239, x3.Mul(kp1380)).Shift(itxShr(11)).Average(zero).Add(x12)
	v14 := x12.MulAdd(km1380, x3.Mul(km239)).Shift(itxShr(11)).Average(zero).Add(x3)
	v15 := x14.MulAdd(km44, x1.Mul(kp601)).Shift(itxShr(11)).Average(zero).Add(x14)
	v16 := x14.MulAdd(km601, x1.Mul(km44)).Shift(itxShr(11)).Average(zero).Add(x1)
	v17 := v1.Add(v9).Max(minV).Min(maxV)
	v18 := v2.Add(v10).Max(minV).Min(maxV)
	v19 := v3.Add(v11).Max(minV).Min(maxV)
	v20 := v4.Add(v12).Max(minV).Min(maxV)
	v21 := v5.Add(v13).Max(minV).Min(maxV)
	v22 := v6.Add(v14).Max(minV).Min(maxV)
	v23 := v7.Add(v15).Max(minV).Min(maxV)
	v24 := v8.Add(v16).Max(minV).Min(maxV)
	v25 := v1.Sub(v9).Max(minV).Min(maxV)
	v26 := v2.Sub(v10).Max(minV).Min(maxV)
	v27 := v3.Sub(v11).Max(minV).Min(maxV)
	v28 := v4.Sub(v12).Max(minV).Min(maxV)
	v29 := v5.Sub(v13).Max(minV).Min(maxV)
	v30 := v6.Sub(v14).Max(minV).Min(maxV)
	v31 := v7.Sub(v15).Max(minV).Min(maxV)
	v32 := v8.Sub(v16).Max(minV).Min(maxV)
	v33 := v26.MulAdd(kp799, v25.Mul(km79)).Shift(itxShr(11)).Average(zero).Add(v25)
	v34 := v26.MulAdd(kp79, v25.Mul(kp799)).Shift(itxShr(11)).Average(zero).Sub(v26)
	v35 := v28.MulAdd(km690, v27.Mul(kp2276)).Shift(itxShr(11)).Average(zero).Add(v28)
	v36 := v28.MulAdd(km2276, v27.Mul(km690)).Shift(itxShr(11)).Average(zero).Add(v27)
	v37 := v29.MulAdd(km799, v30.Mul(km79)).Shift(itxShr(11)).Average(zero).Add(v30)
	v38 := v29.MulAdd(km79, v30.Mul(kp799)).Shift(itxShr(11)).Average(zero).Add(v29)
	v39 := v31.MulAdd(kp690, v32.Mul(kp2276)).Shift(itxShr(11)).Average(zero).Sub(v31)
	v40 := v31.MulAdd(kp2276, v32.Mul(km690)).Shift(itxShr(11)).Average(zero).Add(v32)
	v41 := v17.Add(v21).Max(minV).Min(maxV)
	v42 := v18.Add(v22).Max(minV).Min(maxV)
	v43 := v19.Add(v23).Max(minV).Min(maxV)
	v44 := v20.Add(v24).Max(minV).Min(maxV)
	v45 := v17.Sub(v21).Max(minV).Min(maxV)
	v46 := v18.Sub(v22).Max(minV).Min(maxV)
	v47 := v19.Sub(v23).Max(minV).Min(maxV)
	v48 := v20.Sub(v24).Max(minV).Min(maxV)
	v49 := v33.Add(v37).Max(minV).Min(maxV)
	v50 := v34.Add(v38).Max(minV).Min(maxV)
	v51 := v35.Add(v39).Max(minV).Min(maxV)
	v52 := v36.Add(v40).Max(minV).Min(maxV)
	v53 := v33.Sub(v37).Max(minV).Min(maxV)
	v54 := v34.Sub(v38).Max(minV).Min(maxV)
	v55 := v35.Sub(v39).Max(minV).Min(maxV)
	v56 := v36.Sub(v40).Max(minV).Min(maxV)
	v57 := v46.MulAdd(kp1567, v45.Mul(km312)).Shift(itxShr(11)).Average(zero).Add(v45)
	v58 := v46.MulAdd(kp312, v45.Mul(kp1567)).Shift(itxShr(11)).Average(zero).Sub(v46)
	v59 := v47.MulAdd(km1567, v48.Mul(km312)).Shift(itxShr(11)).Average(zero).Add(v48)
	v60 := v47.MulAdd(km312, v48.Mul(kp1567)).Shift(itxShr(11)).Average(zero).Add(v47)
	v61 := v54.MulAdd(kp1567, v53.Mul(km312)).Shift(itxShr(11)).Average(zero).Add(v53)
	v62 := v54.MulAdd(kp312, v53.Mul(kp1567)).Shift(itxShr(11)).Average(zero).Sub(v54)
	v63 := v55.MulAdd(km1567, v56.Mul(km312)).Shift(itxShr(11)).Average(zero).Add(v56)
	v64 := v55.MulAdd(km312, v56.Mul(kp1567)).Shift(itxShr(11)).Average(zero).Add(v55)
	v65 := v41.Add(v43).Max(minV).Min(maxV)
	v66 := zero.Sub(v42.Add(v44).Max(minV).Min(maxV))
	v67 := v41.Sub(v43).Max(minV).Min(maxV)
	v68 := v42.Sub(v44).Max(minV).Min(maxV)
	v69 := zero.Sub(v57.Add(v59).Max(minV).Min(maxV))
	v70 := v58.Add(v60).Max(minV).Min(maxV)
	v71 := v57.Sub(v59).Max(minV).Min(maxV)
	v72 := v58.Sub(v60).Max(minV).Min(maxV)
	v73 := zero.Sub(v49.Add(v51).Max(minV).Min(maxV))
	v74 := v50.Add(v52).Max(minV).Min(maxV)
	v75 := v49.Sub(v51).Max(minV).Min(maxV)
	v76 := v50.Sub(v52).Max(minV).Min(maxV)
	v77 := v61.Add(v63).Max(minV).Min(maxV)
	v78 := zero.Sub(v62.Add(v64).Max(minV).Min(maxV))
	v79 := v61.Sub(v63).Max(minV).Min(maxV)
	v80 := v62.Sub(v64).Max(minV).Min(maxV)
	v81 := zero.Sub(v67.Add(v68).Mul(kp181).Shift(itxShr(7)).Average(zero))
	v82 := v67.Sub(v68).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v83 := v71.Add(v72).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v84 := zero.Sub(v71.Sub(v72).Mul(kp181).Shift(itxShr(7)).Average(zero))
	v85 := v75.Add(v76).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v86 := zero.Sub(v75.Sub(v76).Mul(kp181).Shift(itxShr(7)).Average(zero))
	v87 := zero.Sub(v79.Add(v80).Mul(kp181).Shift(itxShr(7)).Average(zero))
	v88 := v79.Sub(v80).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v65.StoreArray((*[4]int32)(buf[0*rowStride:]))
	v73.StoreArray((*[4]int32)(buf[1*rowStride:]))
	v77.StoreArray((*[4]int32)(buf[2*rowStride:]))
	v69.StoreArray((*[4]int32)(buf[3*rowStride:]))
	v83.StoreArray((*[4]int32)(buf[4*rowStride:]))
	v87.StoreArray((*[4]int32)(buf[5*rowStride:]))
	v85.StoreArray((*[4]int32)(buf[6*rowStride:]))
	v81.StoreArray((*[4]int32)(buf[7*rowStride:]))
	v82.StoreArray((*[4]int32)(buf[8*rowStride:]))
	v86.StoreArray((*[4]int32)(buf[9*rowStride:]))
	v88.StoreArray((*[4]int32)(buf[10*rowStride:]))
	v84.StoreArray((*[4]int32)(buf[11*rowStride:]))
	v70.StoreArray((*[4]int32)(buf[12*rowStride:]))
	v78.StoreArray((*[4]int32)(buf[13*rowStride:]))
	v74.StoreArray((*[4]int32)(buf[14*rowStride:]))
	v66.StoreArray((*[4]int32)(buf[15*rowStride:]))
}

// inverseADST16Col4FlipSIMD applies inverseADST16 to 4 adjacent columns of buf in int32 lanes.
func inverseADST16Col4FlipSIMD(buf []int32, rowStride int, min, max int32) {
	minV := archsimd.BroadcastInt32x4(min)
	maxV := archsimd.BroadcastInt32x4(max)
	var zero archsimd.Int32x4
	km2751 := archsimd.LoadInt32x4Array(&itxCm2751)
	km2276 := archsimd.LoadInt32x4Array(&itxCm2276)
	km2106 := archsimd.LoadInt32x4Array(&itxCm2106)
	km1645 := archsimd.LoadInt32x4Array(&itxCm1645)
	km1567 := archsimd.LoadInt32x4Array(&itxCm1567)
	km1380 := archsimd.LoadInt32x4Array(&itxCm1380)
	km1061 := archsimd.LoadInt32x4Array(&itxCm1061)
	km799 := archsimd.LoadInt32x4Array(&itxCm799)
	km690 := archsimd.LoadInt32x4Array(&itxCm690)
	km601 := archsimd.LoadInt32x4Array(&itxCm601)
	km583 := archsimd.LoadInt32x4Array(&itxCm583)
	km393 := archsimd.LoadInt32x4Array(&itxCm393)
	km312 := archsimd.LoadInt32x4Array(&itxCm312)
	km239 := archsimd.LoadInt32x4Array(&itxCm239)
	km123 := archsimd.LoadInt32x4Array(&itxCm123)
	km79 := archsimd.LoadInt32x4Array(&itxCm79)
	km44 := archsimd.LoadInt32x4Array(&itxCm44)
	km5 := archsimd.LoadInt32x4Array(&itxCm5)
	kp5 := archsimd.LoadInt32x4Array(&itxCp5)
	kp79 := archsimd.LoadInt32x4Array(&itxCp79)
	kp123 := archsimd.LoadInt32x4Array(&itxCp123)
	kp181 := archsimd.LoadInt32x4Array(&itxCp181)
	kp201 := archsimd.LoadInt32x4Array(&itxCp201)
	kp312 := archsimd.LoadInt32x4Array(&itxCp312)
	kp393 := archsimd.LoadInt32x4Array(&itxCp393)
	kp601 := archsimd.LoadInt32x4Array(&itxCp601)
	kp690 := archsimd.LoadInt32x4Array(&itxCp690)
	kp799 := archsimd.LoadInt32x4Array(&itxCp799)
	kp995 := archsimd.LoadInt32x4Array(&itxCp995)
	kp1220 := archsimd.LoadInt32x4Array(&itxCp1220)
	kp1380 := archsimd.LoadInt32x4Array(&itxCp1380)
	kp1567 := archsimd.LoadInt32x4Array(&itxCp1567)
	kp1645 := archsimd.LoadInt32x4Array(&itxCp1645)
	kp1751 := archsimd.LoadInt32x4Array(&itxCp1751)
	kp2106 := archsimd.LoadInt32x4Array(&itxCp2106)
	kp2276 := archsimd.LoadInt32x4Array(&itxCp2276)
	kp2751 := archsimd.LoadInt32x4Array(&itxCp2751)
	x0 := archsimd.LoadInt32x4Array((*[4]int32)(buf[0*rowStride:]))
	x1 := archsimd.LoadInt32x4Array((*[4]int32)(buf[1*rowStride:]))
	x2 := archsimd.LoadInt32x4Array((*[4]int32)(buf[2*rowStride:]))
	x3 := archsimd.LoadInt32x4Array((*[4]int32)(buf[3*rowStride:]))
	x4 := archsimd.LoadInt32x4Array((*[4]int32)(buf[4*rowStride:]))
	x5 := archsimd.LoadInt32x4Array((*[4]int32)(buf[5*rowStride:]))
	x6 := archsimd.LoadInt32x4Array((*[4]int32)(buf[6*rowStride:]))
	x7 := archsimd.LoadInt32x4Array((*[4]int32)(buf[7*rowStride:]))
	x8 := archsimd.LoadInt32x4Array((*[4]int32)(buf[8*rowStride:]))
	x9 := archsimd.LoadInt32x4Array((*[4]int32)(buf[9*rowStride:]))
	x10 := archsimd.LoadInt32x4Array((*[4]int32)(buf[10*rowStride:]))
	x11 := archsimd.LoadInt32x4Array((*[4]int32)(buf[11*rowStride:]))
	x12 := archsimd.LoadInt32x4Array((*[4]int32)(buf[12*rowStride:]))
	x13 := archsimd.LoadInt32x4Array((*[4]int32)(buf[13*rowStride:]))
	x14 := archsimd.LoadInt32x4Array((*[4]int32)(buf[14*rowStride:]))
	x15 := archsimd.LoadInt32x4Array((*[4]int32)(buf[15*rowStride:]))
	v1 := x0.MulAdd(kp201, x15.Mul(km5)).Shift(itxShr(11)).Average(zero).Add(x15)
	v2 := x0.MulAdd(kp5, x15.Mul(kp201)).Shift(itxShr(11)).Average(zero).Sub(x0)
	v3 := x2.MulAdd(kp995, x13.Mul(km123)).Shift(itxShr(11)).Average(zero).Add(x13)
	v4 := x2.MulAdd(kp123, x13.Mul(kp995)).Shift(itxShr(11)).Average(zero).Sub(x2)
	v5 := x4.MulAdd(kp1751, x11.Mul(km393)).Shift(itxShr(11)).Average(zero).Add(x11)
	v6 := x4.MulAdd(kp393, x11.Mul(kp1751)).Shift(itxShr(11)).Average(zero).Sub(x4)
	v7 := x6.MulAdd(kp1220, x9.Mul(kp1645)).Shift(itxShr(10)).Average(zero)
	v8 := x6.MulAdd(km1645, x9.Mul(kp1220)).Shift(itxShr(10)).Average(zero)
	v9 := x8.MulAdd(km1061, x7.Mul(kp2751)).Shift(itxShr(11)).Average(zero).Add(x8)
	v10 := x8.MulAdd(km2751, x7.Mul(km1061)).Shift(itxShr(11)).Average(zero).Add(x7)
	v11 := x10.MulAdd(km583, x5.Mul(kp2106)).Shift(itxShr(11)).Average(zero).Add(x10)
	v12 := x10.MulAdd(km2106, x5.Mul(km583)).Shift(itxShr(11)).Average(zero).Add(x5)
	v13 := x12.MulAdd(km239, x3.Mul(kp1380)).Shift(itxShr(11)).Average(zero).Add(x12)
	v14 := x12.MulAdd(km1380, x3.Mul(km239)).Shift(itxShr(11)).Average(zero).Add(x3)
	v15 := x14.MulAdd(km44, x1.Mul(kp601)).Shift(itxShr(11)).Average(zero).Add(x14)
	v16 := x14.MulAdd(km601, x1.Mul(km44)).Shift(itxShr(11)).Average(zero).Add(x1)
	v17 := v1.Add(v9).Max(minV).Min(maxV)
	v18 := v2.Add(v10).Max(minV).Min(maxV)
	v19 := v3.Add(v11).Max(minV).Min(maxV)
	v20 := v4.Add(v12).Max(minV).Min(maxV)
	v21 := v5.Add(v13).Max(minV).Min(maxV)
	v22 := v6.Add(v14).Max(minV).Min(maxV)
	v23 := v7.Add(v15).Max(minV).Min(maxV)
	v24 := v8.Add(v16).Max(minV).Min(maxV)
	v25 := v1.Sub(v9).Max(minV).Min(maxV)
	v26 := v2.Sub(v10).Max(minV).Min(maxV)
	v27 := v3.Sub(v11).Max(minV).Min(maxV)
	v28 := v4.Sub(v12).Max(minV).Min(maxV)
	v29 := v5.Sub(v13).Max(minV).Min(maxV)
	v30 := v6.Sub(v14).Max(minV).Min(maxV)
	v31 := v7.Sub(v15).Max(minV).Min(maxV)
	v32 := v8.Sub(v16).Max(minV).Min(maxV)
	v33 := v26.MulAdd(kp799, v25.Mul(km79)).Shift(itxShr(11)).Average(zero).Add(v25)
	v34 := v26.MulAdd(kp79, v25.Mul(kp799)).Shift(itxShr(11)).Average(zero).Sub(v26)
	v35 := v28.MulAdd(km690, v27.Mul(kp2276)).Shift(itxShr(11)).Average(zero).Add(v28)
	v36 := v28.MulAdd(km2276, v27.Mul(km690)).Shift(itxShr(11)).Average(zero).Add(v27)
	v37 := v29.MulAdd(km799, v30.Mul(km79)).Shift(itxShr(11)).Average(zero).Add(v30)
	v38 := v29.MulAdd(km79, v30.Mul(kp799)).Shift(itxShr(11)).Average(zero).Add(v29)
	v39 := v31.MulAdd(kp690, v32.Mul(kp2276)).Shift(itxShr(11)).Average(zero).Sub(v31)
	v40 := v31.MulAdd(kp2276, v32.Mul(km690)).Shift(itxShr(11)).Average(zero).Add(v32)
	v41 := v17.Add(v21).Max(minV).Min(maxV)
	v42 := v18.Add(v22).Max(minV).Min(maxV)
	v43 := v19.Add(v23).Max(minV).Min(maxV)
	v44 := v20.Add(v24).Max(minV).Min(maxV)
	v45 := v17.Sub(v21).Max(minV).Min(maxV)
	v46 := v18.Sub(v22).Max(minV).Min(maxV)
	v47 := v19.Sub(v23).Max(minV).Min(maxV)
	v48 := v20.Sub(v24).Max(minV).Min(maxV)
	v49 := v33.Add(v37).Max(minV).Min(maxV)
	v50 := v34.Add(v38).Max(minV).Min(maxV)
	v51 := v35.Add(v39).Max(minV).Min(maxV)
	v52 := v36.Add(v40).Max(minV).Min(maxV)
	v53 := v33.Sub(v37).Max(minV).Min(maxV)
	v54 := v34.Sub(v38).Max(minV).Min(maxV)
	v55 := v35.Sub(v39).Max(minV).Min(maxV)
	v56 := v36.Sub(v40).Max(minV).Min(maxV)
	v57 := v46.MulAdd(kp1567, v45.Mul(km312)).Shift(itxShr(11)).Average(zero).Add(v45)
	v58 := v46.MulAdd(kp312, v45.Mul(kp1567)).Shift(itxShr(11)).Average(zero).Sub(v46)
	v59 := v47.MulAdd(km1567, v48.Mul(km312)).Shift(itxShr(11)).Average(zero).Add(v48)
	v60 := v47.MulAdd(km312, v48.Mul(kp1567)).Shift(itxShr(11)).Average(zero).Add(v47)
	v61 := v54.MulAdd(kp1567, v53.Mul(km312)).Shift(itxShr(11)).Average(zero).Add(v53)
	v62 := v54.MulAdd(kp312, v53.Mul(kp1567)).Shift(itxShr(11)).Average(zero).Sub(v54)
	v63 := v55.MulAdd(km1567, v56.Mul(km312)).Shift(itxShr(11)).Average(zero).Add(v56)
	v64 := v55.MulAdd(km312, v56.Mul(kp1567)).Shift(itxShr(11)).Average(zero).Add(v55)
	v65 := v41.Add(v43).Max(minV).Min(maxV)
	v66 := zero.Sub(v42.Add(v44).Max(minV).Min(maxV))
	v67 := v41.Sub(v43).Max(minV).Min(maxV)
	v68 := v42.Sub(v44).Max(minV).Min(maxV)
	v69 := zero.Sub(v57.Add(v59).Max(minV).Min(maxV))
	v70 := v58.Add(v60).Max(minV).Min(maxV)
	v71 := v57.Sub(v59).Max(minV).Min(maxV)
	v72 := v58.Sub(v60).Max(minV).Min(maxV)
	v73 := zero.Sub(v49.Add(v51).Max(minV).Min(maxV))
	v74 := v50.Add(v52).Max(minV).Min(maxV)
	v75 := v49.Sub(v51).Max(minV).Min(maxV)
	v76 := v50.Sub(v52).Max(minV).Min(maxV)
	v77 := v61.Add(v63).Max(minV).Min(maxV)
	v78 := zero.Sub(v62.Add(v64).Max(minV).Min(maxV))
	v79 := v61.Sub(v63).Max(minV).Min(maxV)
	v80 := v62.Sub(v64).Max(minV).Min(maxV)
	v81 := zero.Sub(v67.Add(v68).Mul(kp181).Shift(itxShr(7)).Average(zero))
	v82 := v67.Sub(v68).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v83 := v71.Add(v72).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v84 := zero.Sub(v71.Sub(v72).Mul(kp181).Shift(itxShr(7)).Average(zero))
	v85 := v75.Add(v76).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v86 := zero.Sub(v75.Sub(v76).Mul(kp181).Shift(itxShr(7)).Average(zero))
	v87 := zero.Sub(v79.Add(v80).Mul(kp181).Shift(itxShr(7)).Average(zero))
	v88 := v79.Sub(v80).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v66.StoreArray((*[4]int32)(buf[0*rowStride:]))
	v74.StoreArray((*[4]int32)(buf[1*rowStride:]))
	v78.StoreArray((*[4]int32)(buf[2*rowStride:]))
	v70.StoreArray((*[4]int32)(buf[3*rowStride:]))
	v84.StoreArray((*[4]int32)(buf[4*rowStride:]))
	v88.StoreArray((*[4]int32)(buf[5*rowStride:]))
	v86.StoreArray((*[4]int32)(buf[6*rowStride:]))
	v82.StoreArray((*[4]int32)(buf[7*rowStride:]))
	v81.StoreArray((*[4]int32)(buf[8*rowStride:]))
	v85.StoreArray((*[4]int32)(buf[9*rowStride:]))
	v87.StoreArray((*[4]int32)(buf[10*rowStride:]))
	v83.StoreArray((*[4]int32)(buf[11*rowStride:]))
	v69.StoreArray((*[4]int32)(buf[12*rowStride:]))
	v77.StoreArray((*[4]int32)(buf[13*rowStride:]))
	v73.StoreArray((*[4]int32)(buf[14*rowStride:]))
	v65.StoreArray((*[4]int32)(buf[15*rowStride:]))
}

// inverseDCT8Col2SIMD applies inverseDCT8 to 2 adjacent columns of buf in int32 lanes.
func inverseDCT8Col2SIMD(buf []int32, rowStride int, min, max int32) {
	minV := archsimd.BroadcastInt32x4(min)
	maxV := archsimd.BroadcastInt32x4(max)
	var zero archsimd.Int32x4
	km1138 := archsimd.LoadInt32x4Array(&itxCm1138)
	km312 := archsimd.LoadInt32x4Array(&itxCm312)
	km79 := archsimd.LoadInt32x4Array(&itxCm79)
	kp79 := archsimd.LoadInt32x4Array(&itxCp79)
	kp181 := archsimd.LoadInt32x4Array(&itxCp181)
	kp312 := archsimd.LoadInt32x4Array(&itxCp312)
	kp799 := archsimd.LoadInt32x4Array(&itxCp799)
	kp1138 := archsimd.LoadInt32x4Array(&itxCp1138)
	kp1567 := archsimd.LoadInt32x4Array(&itxCp1567)
	kp1703 := archsimd.LoadInt32x4Array(&itxCp1703)
	x0, _ := archsimd.LoadInt32x4Part(buf[0*rowStride : 0*rowStride+2])
	x1, _ := archsimd.LoadInt32x4Part(buf[1*rowStride : 1*rowStride+2])
	x2, _ := archsimd.LoadInt32x4Part(buf[2*rowStride : 2*rowStride+2])
	x3, _ := archsimd.LoadInt32x4Part(buf[3*rowStride : 3*rowStride+2])
	x4, _ := archsimd.LoadInt32x4Part(buf[4*rowStride : 4*rowStride+2])
	x5, _ := archsimd.LoadInt32x4Part(buf[5*rowStride : 5*rowStride+2])
	x6, _ := archsimd.LoadInt32x4Part(buf[6*rowStride : 6*rowStride+2])
	x7, _ := archsimd.LoadInt32x4Part(buf[7*rowStride : 7*rowStride+2])
	v1 := x0.Add(x4).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v2 := x0.Sub(x4).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v3 := x6.MulAdd(kp312, x2.Mul(kp1567)).Shift(itxShr(11)).Average(zero).Sub(x6)
	v4 := x6.MulAdd(kp1567, x2.Mul(km312)).Shift(itxShr(11)).Average(zero).Add(x2)
	v5 := v1.Add(v4).Max(minV).Min(maxV)
	v6 := v2.Add(v3).Max(minV).Min(maxV)
	v7 := v2.Sub(v3).Max(minV).Min(maxV)
	v8 := v1.Sub(v4).Max(minV).Min(maxV)
	v9 := x7.MulAdd(kp79, x1.Mul(kp799)).Shift(itxShr(11)).Average(zero).Sub(x7)
	v10 := x3.MulAdd(km1138, x5.Mul(kp1703)).Shift(itxShr(10)).Average(zero)
	v11 := x3.MulAdd(kp1703, x5.Mul(kp1138)).Shift(itxShr(10)).Average(zero)
	v12 := x7.MulAdd(kp799, x1.Mul(km79)).Shift(itxShr(11)).Average(zero).Add(x1)
	v13 := v9.Add(v10).Max(minV).Min(maxV)
	v14 := v9.Sub(v10).Max(minV).Min(maxV)
	v15 := v12.Add(v11).Max(minV).Min(maxV)
	v16 := v12.Sub(v11).Max(minV).Min(maxV)
	v17 := v16.Sub(v14).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v18 := v16.Add(v14).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v19 := v5.Add(v15).Max(minV).Min(maxV)
	v20 := v6.Add(v18).Max(minV).Min(maxV)
	v21 := v7.Add(v17).Max(minV).Min(maxV)
	v22 := v8.Add(v13).Max(minV).Min(maxV)
	v23 := v8.Sub(v13).Max(minV).Min(maxV)
	v24 := v7.Sub(v17).Max(minV).Min(maxV)
	v25 := v6.Sub(v18).Max(minV).Min(maxV)
	v26 := v5.Sub(v15).Max(minV).Min(maxV)
	v19.StorePart(buf[0*rowStride : 0*rowStride+2])
	v20.StorePart(buf[1*rowStride : 1*rowStride+2])
	v21.StorePart(buf[2*rowStride : 2*rowStride+2])
	v22.StorePart(buf[3*rowStride : 3*rowStride+2])
	v23.StorePart(buf[4*rowStride : 4*rowStride+2])
	v24.StorePart(buf[5*rowStride : 5*rowStride+2])
	v25.StorePart(buf[6*rowStride : 6*rowStride+2])
	v26.StorePart(buf[7*rowStride : 7*rowStride+2])
}

// inverseDCT16Col2SIMD applies inverseDCT16 to 2 adjacent columns of buf in int32 lanes.
func inverseDCT16Col2SIMD(buf []int32, rowStride int, min, max int32) {
	minV := archsimd.BroadcastInt32x4(min)
	maxV := archsimd.BroadcastInt32x4(max)
	var zero archsimd.Int32x4
	km1299 := archsimd.LoadInt32x4Array(&itxCm1299)
	km1189 := archsimd.LoadInt32x4Array(&itxCm1189)
	km1138 := archsimd.LoadInt32x4Array(&itxCm1138)
	km484 := archsimd.LoadInt32x4Array(&itxCm484)
	km312 := archsimd.LoadInt32x4Array(&itxCm312)
	km176 := archsimd.LoadInt32x4Array(&itxCm176)
	km79 := archsimd.LoadInt32x4Array(&itxCm79)
	km20 := archsimd.LoadInt32x4Array(&itxCm20)
	kp20 := archsimd.LoadInt32x4Array(&itxCp20)
	kp79 := archsimd.LoadInt32x4Array(&itxCp79)
	kp181 := archsimd.LoadInt32x4Array(&itxCp181)
	kp312 := archsimd.LoadInt32x4Array(&itxCp312)
	kp401 := archsimd.LoadInt32x4Array(&itxCp401)
	kp484 := archsimd.LoadInt32x4Array(&itxCp484)
	kp799 := archsimd.LoadInt32x4Array(&itxCp799)
	kp1138 := archsimd.LoadInt32x4Array(&itxCp1138)
	kp1189 := archsimd.LoadInt32x4Array(&itxCp1189)
	kp1299 := archsimd.LoadInt32x4Array(&itxCp1299)
	kp1567 := archsimd.LoadInt32x4Array(&itxCp1567)
	kp1583 := archsimd.LoadInt32x4Array(&itxCp1583)
	kp1703 := archsimd.LoadInt32x4Array(&itxCp1703)
	kp1931 := archsimd.LoadInt32x4Array(&itxCp1931)
	x0, _ := archsimd.LoadInt32x4Part(buf[0*rowStride : 0*rowStride+2])
	x1, _ := archsimd.LoadInt32x4Part(buf[1*rowStride : 1*rowStride+2])
	x2, _ := archsimd.LoadInt32x4Part(buf[2*rowStride : 2*rowStride+2])
	x3, _ := archsimd.LoadInt32x4Part(buf[3*rowStride : 3*rowStride+2])
	x4, _ := archsimd.LoadInt32x4Part(buf[4*rowStride : 4*rowStride+2])
	x5, _ := archsimd.LoadInt32x4Part(buf[5*rowStride : 5*rowStride+2])
	x6, _ := archsimd.LoadInt32x4Part(buf[6*rowStride : 6*rowStride+2])
	x7, _ := archsimd.LoadInt32x4Part(buf[7*rowStride : 7*rowStride+2])
	x8, _ := archsimd.LoadInt32x4Part(buf[8*rowStride : 8*rowStride+2])
	x9, _ := archsimd.LoadInt32x4Part(buf[9*rowStride : 9*rowStride+2])
	x10, _ := archsimd.LoadInt32x4Part(buf[10*rowStride : 10*rowStride+2])
	x11, _ := archsimd.LoadInt32x4Part(buf[11*rowStride : 11*rowStride+2])
	x12, _ := archsimd.LoadInt32x4Part(buf[12*rowStride : 12*rowStride+2])
	x13, _ := archsimd.LoadInt32x4Part(buf[13*rowStride : 13*rowStride+2])
	x14, _ := archsimd.LoadInt32x4Part(buf[14*rowStride : 14*rowStride+2])
	x15, _ := archsimd.LoadInt32x4Part(buf[15*rowStride : 15*rowStride+2])
	v1 := x0.Add(x8).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v2 := x0.Sub(x8).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v3 := x12.MulAdd(kp312, x4.Mul(kp1567)).Shift(itxShr(11)).Average(zero).Sub(x12)
	v4 := x12.MulAdd(kp1567, x4.Mul(km312)).Shift(itxShr(11)).Average(zero).Add(x4)
	v5 := v1.Add(v4).Max(minV).Min(maxV)
	v6 := v2.Add(v3).Max(minV).Min(maxV)
	v7 := v2.Sub(v3).Max(minV).Min(maxV)
	v8 := v1.Sub(v4).Max(minV).Min(maxV)
	v9 := x14.MulAdd(kp79, x2.Mul(kp799)).Shift(itxShr(11)).Average(zero).Sub(x14)
	v10 := x6.MulAdd(km1138, x10.Mul(kp1703)).Shift(itxShr(10)).Average(zero)
	v11 := x6.MulAdd(kp1703, x10.Mul(kp1138)).Shift(itxShr(10)).Average(zero)
	v12 := x14.MulAdd(kp799, x2.Mul(km79)).Shift(itxShr(11)).Average(zero).Add(x2)
	v13 := v9.Add(v10).Max(minV).Min(maxV)
	v14 := v9.Sub(v10).Max(minV).Min(maxV)
	v15 := v12.Add(v11).Max(minV).Min(maxV)
	v16 := v12.Sub(v11).Max(minV).Min(maxV)
	v17 := v16.Sub(v14).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v18 := v16.Add(v14).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v19 := v5.Add(v15).Max(minV).Min(maxV)
	v20 := v6.Add(v18).Max(minV).Min(maxV)
	v21 := v7.Add(v17).Max(minV).Min(maxV)
	v22 := v8.Add(v13).Max(minV).Min(maxV)
	v23 := v8.Sub(v13).Max(minV).Min(maxV)
	v24 := v7.Sub(v17).Max(minV).Min(maxV)
	v25 := v6.Sub(v18).Max(minV).Min(maxV)
	v26 := v5.Sub(v15).Max(minV).Min(maxV)
	v27 := x15.MulAdd(kp20, x1.Mul(kp401)).Shift(itxShr(11)).Average(zero).Sub(x15)
	v28 := x7.MulAdd(km1299, x9.Mul(kp1583)).Shift(itxShr(10)).Average(zero)
	v29 := x11.MulAdd(kp484, x5.Mul(kp1931)).Shift(itxShr(11)).Average(zero).Sub(x11)
	v30 := x3.MulAdd(km1189, x13.Mul(km176)).Shift(itxShr(11)).Average(zero).Add(x13)
	v31 := x3.MulAdd(km176, x13.Mul(kp1189)).Shift(itxShr(11)).Average(zero).Add(x3)
	v32 := x11.MulAdd(kp1931, x5.Mul(km484)).Shift(itxShr(11)).Average(zero).Add(x5)
	v33 := x7.MulAdd(kp1583, x9.Mul(kp1299)).Shift(itxShr(10)).Average(zero)
	v34 := x15.MulAdd(kp401, x1.Mul(km20)).Shift(itxShr(11)).Average(zero).Add(x1)
	v35 := v27.Add(v28).Max(minV).Min(maxV)
	v36 := v27.Sub(v28).Max(minV).Min(maxV)
	v37 := v30.Sub(v29).Max(minV).Min(maxV)
	v38 := v30.Add(v29).Max(minV).Min(maxV)
	v39 := v31.Add(v32).Max(minV).Min(maxV)
	v40 := v31.Sub(v32).Max(minV).Min(maxV)
	v41 := v34.Sub(v33).Max(minV).Min(maxV)
	v42 := v34.Add(v33).Max(minV).Min(maxV)
	v43 := v36.MulAdd(kp312, v41.Mul(kp1567)).Shift(itxShr(11)).Average(zero).Sub(v36)
	v44 := v36.MulAdd(kp1567, v41.Mul(km312)).Shift(itxShr(11)).Average(zero).Add(v41)
	v45 := zero.Sub(v37.MulAdd(kp1567, v40.Mul(km312))).Shift(itxShr(11)).Average(zero).Sub(v40)
	v46 := v37.MulAdd(kp312, v40.Mul(kp1567)).Shift(itxShr(11)).Average(zero).Sub(v37)
	v47 := v35.Add(v38).Max(minV).Min(maxV)
	v48 := v43.Add(v45).Max(minV).Min(maxV)
	v49 := v43.Sub(v45).Max(minV).Min(maxV)
	v50 := v35.Sub(v38).Max(minV).Min(maxV)
	v51 := v42.Sub(v39).Max(minV).Min(maxV)
	v52 := v44.Sub(v46).Max(minV).Min(maxV)
	v53 := v44.Add(v46).Max(minV).Min(maxV)
	v54 := v42.Add(v39).Max(minV).Min(maxV)
	v55 := v52.Sub(v49).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v56 := v52.Add(v49).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v57 := v51.Sub(v50).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v58 := v51.Add(v50).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v59 := v19.Add(v54).Max(minV).Min(maxV)
	v60 := v20.Add(v53).Max(minV).Min(maxV)
	v61 := v21.Add(v56).Max(minV).Min(maxV)
	v62 := v22.Add(v58).Max(minV).Min(maxV)
	v63 := v23.Add(v57).Max(minV).Min(maxV)
	v64 := v24.Add(v55).Max(minV).Min(maxV)
	v65 := v25.Add(v48).Max(minV).Min(maxV)
	v66 := v26.Add(v47).Max(minV).Min(maxV)
	v67 := v26.Sub(v47).Max(minV).Min(maxV)
	v68 := v25.Sub(v48).Max(minV).Min(maxV)
	v69 := v24.Sub(v55).Max(minV).Min(maxV)
	v70 := v23.Sub(v57).Max(minV).Min(maxV)
	v71 := v22.Sub(v58).Max(minV).Min(maxV)
	v72 := v21.Sub(v56).Max(minV).Min(maxV)
	v73 := v20.Sub(v53).Max(minV).Min(maxV)
	v74 := v19.Sub(v54).Max(minV).Min(maxV)
	v59.StorePart(buf[0*rowStride : 0*rowStride+2])
	v60.StorePart(buf[1*rowStride : 1*rowStride+2])
	v61.StorePart(buf[2*rowStride : 2*rowStride+2])
	v62.StorePart(buf[3*rowStride : 3*rowStride+2])
	v63.StorePart(buf[4*rowStride : 4*rowStride+2])
	v64.StorePart(buf[5*rowStride : 5*rowStride+2])
	v65.StorePart(buf[6*rowStride : 6*rowStride+2])
	v66.StorePart(buf[7*rowStride : 7*rowStride+2])
	v67.StorePart(buf[8*rowStride : 8*rowStride+2])
	v68.StorePart(buf[9*rowStride : 9*rowStride+2])
	v69.StorePart(buf[10*rowStride : 10*rowStride+2])
	v70.StorePart(buf[11*rowStride : 11*rowStride+2])
	v71.StorePart(buf[12*rowStride : 12*rowStride+2])
	v72.StorePart(buf[13*rowStride : 13*rowStride+2])
	v73.StorePart(buf[14*rowStride : 14*rowStride+2])
	v74.StorePart(buf[15*rowStride : 15*rowStride+2])
}

// inverseDCT32Col2SIMD applies inverseDCT32 to 2 adjacent columns of buf in int32 lanes.
func inverseDCT32Col2SIMD(buf []int32, rowStride int, min, max int32) {
	minV := archsimd.BroadcastInt32x4(min)
	maxV := archsimd.BroadcastInt32x4(max)
	var zero archsimd.Int32x4
	km2751 := archsimd.LoadInt32x4Array(&itxCm2751)
	km2106 := archsimd.LoadInt32x4Array(&itxCm2106)
	km1645 := archsimd.LoadInt32x4Array(&itxCm1645)
	km1380 := archsimd.LoadInt32x4Array(&itxCm1380)
	km1299 := archsimd.LoadInt32x4Array(&itxCm1299)
	km1189 := archsimd.LoadInt32x4Array(&itxCm1189)
	km1138 := archsimd.LoadInt32x4Array(&itxCm1138)
	km1061 := archsimd.LoadInt32x4Array(&itxCm1061)
	km601 := archsimd.LoadInt32x4Array(&itxCm601)
	km583 := archsimd.LoadInt32x4Array(&itxCm583)
	km484 := archsimd.LoadInt32x4Array(&itxCm484)
	km393 := archsimd.LoadInt32x4Array(&itxCm393)
	km312 := archsimd.LoadInt32x4Array(&itxCm312)
	km239 := archsimd.LoadInt32x4Array(&itxCm239)
	km176 := archsimd.LoadInt32x4Array(&itxCm176)
	km123 := archsimd.LoadInt32x4Array(&itxCm123)
	km79 := archsimd.LoadInt32x4Array(&itxCm79)
	km44 := archsimd.LoadInt32x4Array(&itxCm44)
	km20 := archsimd.LoadInt32x4Array(&itxCm20)
	km5 := archsimd.LoadInt32x4Array(&itxCm5)
	kp5 := archsimd.LoadInt32x4Array(&itxCp5)
	kp20 := archsimd.LoadInt32x4Array(&itxCp20)
	kp79 := archsimd.LoadInt32x4Array(&itxCp79)
	kp123 := archsimd.LoadInt32x4Array(&itxCp123)
	kp181 := archsimd.LoadInt32x4Array(&itxCp181)
	kp201 := archsimd.LoadInt32x4Array(&itxCp201)
	kp312 := archsimd.LoadInt32x4Array(&itxCp312)
	kp393 := archsimd.LoadInt32x4Array(&itxCp393)
	kp401 := archsimd.LoadInt32x4Array(&itxCp401)
	kp484 := archsimd.LoadInt32x4Array(&itxCp484)
	kp601 := archsimd.LoadInt32x4Array(&itxCp601)
	kp799 := archsimd.LoadInt32x4Array(&itxCp799)
	kp995 := archsimd.LoadInt32x4Array(&itxCp995)
	kp1138 := archsimd.LoadInt32x4Array(&itxCp1138)
	kp1189 := archsimd.LoadInt32x4Array(&itxCp1189)
	kp1220 := archsimd.LoadInt32x4Array(&itxCp1220)
	kp1299 := archsimd.LoadInt32x4Array(&itxCp1299)
	kp1380 := archsimd.LoadInt32x4Array(&itxCp1380)
	kp1567 := archsimd.LoadInt32x4Array(&itxCp1567)
	kp1583 := archsimd.LoadInt32x4Array(&itxCp1583)
	kp1645 := archsimd.LoadInt32x4Array(&itxCp1645)
	kp1703 := archsimd.LoadInt32x4Array(&itxCp1703)
	kp1751 := archsimd.LoadInt32x4Array(&itxCp1751)
	kp1931 := archsimd.LoadInt32x4Array(&itxCp1931)
	kp2106 := archsimd.LoadInt32x4Array(&itxCp2106)
	kp2751 := archsimd.LoadInt32x4Array(&itxCp2751)
	x0, _ := archsimd.LoadInt32x4Part(buf[0*rowStride : 0*rowStride+2])
	x1, _ := archsimd.LoadInt32x4Part(buf[1*rowStride : 1*rowStride+2])
	x2, _ := archsimd.LoadInt32x4Part(buf[2*rowStride : 2*rowStride+2])
	x3, _ := archsimd.LoadInt32x4Part(buf[3*rowStride : 3*rowStride+2])
	x4, _ := archsimd.LoadInt32x4Part(buf[4*rowStride : 4*rowStride+2])
	x5, _ := archsimd.LoadInt32x4Part(buf[5*rowStride : 5*rowStride+2])
	x6, _ := archsimd.LoadInt32x4Part(buf[6*rowStride : 6*rowStride+2])
	x7, _ := archsimd.LoadInt32x4Part(buf[7*rowStride : 7*rowStride+2])
	x8, _ := archsimd.LoadInt32x4Part(buf[8*rowStride : 8*rowStride+2])
	x9, _ := archsimd.LoadInt32x4Part(buf[9*rowStride : 9*rowStride+2])
	x10, _ := archsimd.LoadInt32x4Part(buf[10*rowStride : 10*rowStride+2])
	x11, _ := archsimd.LoadInt32x4Part(buf[11*rowStride : 11*rowStride+2])
	x12, _ := archsimd.LoadInt32x4Part(buf[12*rowStride : 12*rowStride+2])
	x13, _ := archsimd.LoadInt32x4Part(buf[13*rowStride : 13*rowStride+2])
	x14, _ := archsimd.LoadInt32x4Part(buf[14*rowStride : 14*rowStride+2])
	x15, _ := archsimd.LoadInt32x4Part(buf[15*rowStride : 15*rowStride+2])
	x16, _ := archsimd.LoadInt32x4Part(buf[16*rowStride : 16*rowStride+2])
	x17, _ := archsimd.LoadInt32x4Part(buf[17*rowStride : 17*rowStride+2])
	x18, _ := archsimd.LoadInt32x4Part(buf[18*rowStride : 18*rowStride+2])
	x19, _ := archsimd.LoadInt32x4Part(buf[19*rowStride : 19*rowStride+2])
	x20, _ := archsimd.LoadInt32x4Part(buf[20*rowStride : 20*rowStride+2])
	x21, _ := archsimd.LoadInt32x4Part(buf[21*rowStride : 21*rowStride+2])
	x22, _ := archsimd.LoadInt32x4Part(buf[22*rowStride : 22*rowStride+2])
	x23, _ := archsimd.LoadInt32x4Part(buf[23*rowStride : 23*rowStride+2])
	x24, _ := archsimd.LoadInt32x4Part(buf[24*rowStride : 24*rowStride+2])
	x25, _ := archsimd.LoadInt32x4Part(buf[25*rowStride : 25*rowStride+2])
	x26, _ := archsimd.LoadInt32x4Part(buf[26*rowStride : 26*rowStride+2])
	x27, _ := archsimd.LoadInt32x4Part(buf[27*rowStride : 27*rowStride+2])
	x28, _ := archsimd.LoadInt32x4Part(buf[28*rowStride : 28*rowStride+2])
	x29, _ := archsimd.LoadInt32x4Part(buf[29*rowStride : 29*rowStride+2])
	x30, _ := archsimd.LoadInt32x4Part(buf[30*rowStride : 30*rowStride+2])
	x31, _ := archsimd.LoadInt32x4Part(buf[31*rowStride : 31*rowStride+2])
	v1 := x0.Add(x16).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v2 := x0.Sub(x16).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v3 := x24.MulAdd(kp312, x8.Mul(kp1567)).Shift(itxShr(11)).Average(zero).Sub(x24)
	v4 := x24.MulAdd(kp1567, x8.Mul(km312)).Shift(itxShr(11)).Average(zero).Add(x8)
	v5 := v1.Add(v4).Max(minV).Min(maxV)
	v6 := v2.Add(v3).Max(minV).Min(maxV)
	v7 := v2.Sub(v3).Max(minV).Min(maxV)
	v8 := v1.Sub(v4).Max(minV).Min(maxV)
	v9 := x28.MulAdd(kp79, x4.Mul(kp799)).Shift(itxShr(11)).Average(zero).Sub(x28)
	v10 := x12.MulAdd(km1138, x20.Mul(kp1703)).Shift(itxShr(10)).Average(zero)
	v11 := x12.MulAdd(kp1703, x20.Mul(kp1138)).Shift(itxShr(10)).Average(zero)
	v12 := x28.MulAdd(kp799, x4.Mul(km79)).Shift(itxShr(11)).Average(zero).Add(x4)
	v13 := v9.Add(v10).Max(minV).Min(maxV)
	v14 := v9.Sub(v10).Max(minV).Min(maxV)
	v15 := v12.Add(v11).Max(minV).Min(maxV)
	v16 := v12.Sub(v11).Max(minV).Min(maxV)
	v17 := v16.Sub(v14).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v18 := v16.Add(v14).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v19 := v5.Add(v15).Max(minV).Min(maxV)
	v20 := v6.Add(v18).Max(minV).Min(maxV)
	v21 := v7.Add(v17).Max(minV).Min(maxV)
	v22 := v8.Add(v13).Max(minV).Min(maxV)
	v23 := v8.Sub(v13).Max(minV).Min(maxV)
	v24 := v7.Sub(v17).Max(minV).Min(maxV)
	v25 := v6.Sub(v18).Max(minV).Min(maxV)
	v26 := v5.Sub(v15).Max(minV).Min(maxV)
	v27 := x30.MulAdd(kp20, x2.Mul(kp401)).Shift(itxShr(11)).Average(zero).Sub(x30)
	v28 := x14.MulAdd(km1299, x18.Mul(kp1583)).Shift(itxShr(10)).Average(zero)
	v29 := x22.MulAdd(kp484, x10.Mul(kp1931)).Shift(itxShr(11)).Average(zero).Sub(x22)
	v30 := x6.MulAdd(km1189, x26.Mul(km176)).Shift(itxShr(11)).Average(zero).Add(x26)
	v31 := x6.MulAdd(km176, x26.Mul(kp1189)).Shift(itxShr(11)).Average(zero).Add(x6)
	v32 := x22.MulAdd(kp1931, x10.Mul(km484)).Shift(itxShr(11)).Average(zero).Add(x10)
	v33 := x14.MulAdd(kp1583, x18.Mul(kp1299)).Shift(itxShr(10)).Average(zero)
	v34 := x30.MulAdd(kp401, x2.Mul(km20)).Shift(itxShr(11)).Average(zero).Add(x2)
	v35 := v27.Add(v28).Max(minV).Min(maxV)
	v36 := v27.Sub(v28).Max(minV).Min(maxV)
	v37 := v30.Sub(v29).Max(minV).Min(maxV)
	v38 := v30.Add(v29).Max(minV).Min(maxV)
	v39 := v31.Add(v32).Max(minV).Min(maxV)
	v40 := v31.Sub(v32).Max(minV).Min(maxV)
	v41 := v34.Sub(v33).Max(minV).Min(maxV)
	v42 := v34.Add(v33).Max(minV).Min(maxV)
	v43 := v36.MulAdd(kp312, v41.Mul(kp1567)).Shift(itxShr(11)).Average(zero).Sub(v36)
	v44 := v36.MulAdd(kp1567, v41.Mul(km312)).Shift(itxShr(11)).Average(zero).Add(v41)
	v45 := zero.Sub(v37.MulAdd(kp1567, v40.Mul(km312))).Shift(itxShr(11)).Average(zero).Sub(v40)
	v46 := v37.MulAdd(kp312, v40.Mul(kp1567)).Shift(itxShr(11)).Average(zero).Sub(v37)
	v47 := v35.Add(v38).Max(minV).Min(maxV)
	v48 := v43.Add(v45).Max(minV).Min(maxV)
	v49 := v43.Sub(v45).Max(minV).Min(maxV)
	v50 := v35.Sub(v38).Max(minV).Min(maxV)
	v51 := v42.Sub(v39).Max(minV).Min(maxV)
	v52 := v44.Sub(v46).Max(minV).Min(maxV)
	v53 := v44.Add(v46).Max(minV).Min(maxV)
	v54 := v42.Add(v39).Max(minV).Min(maxV)
	v55 := v52.Sub(v49).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v56 := v52.Add(v49).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v57 := v51.Sub(v50).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v58 := v51.Add(v50).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v59 := v19.Add(v54).Max(minV).Min(maxV)
	v60 := v20.Add(v53).Max(minV).Min(maxV)
	v61 := v21.Add(v56).Max(minV).Min(maxV)
	v62 := v22.Add(v58).Max(minV).Min(maxV)
	v63 := v23.Add(v57).Max(minV).Min(maxV)
	v64 := v24.Add(v55).Max(minV).Min(maxV)
	v65 := v25.Add(v48).Max(minV).Min(maxV)
	v66 := v26.Add(v47).Max(minV).Min(maxV)
	v67 := v26.Sub(v47).Max(minV).Min(maxV)
	v68 := v25.Sub(v48).Max(minV).Min(maxV)
	v69 := v24.Sub(v55).Max(minV).Min(maxV)
	v70 := v23.Sub(v57).Max(minV).Min(maxV)
	v71 := v22.Sub(v58).Max(minV).Min(maxV)
	v72 := v21.Sub(v56).Max(minV).Min(maxV)
	v73 := v20.Sub(v53).Max(minV).Min(maxV)
	v74 := v19.Sub(v54).Max(minV).Min(maxV)
	v75 := x31.MulAdd(kp5, x1.Mul(kp201)).Shift(itxShr(11)).Average(zero).Sub(x31)
	v76 := x15.MulAdd(km2751, x17.Mul(km1061)).Shift(itxShr(11)).Average(zero).Add(x17)
	v77 := x23.MulAdd(kp393, x9.Mul(kp1751)).Shift(itxShr(11)).Average(zero).Sub(x23)
	v78 := x7.MulAdd(km1380, x25.Mul(km239)).Shift(itxShr(11)).Average(zero).Add(x25)
	v79 := x27.MulAdd(kp123, x5.Mul(kp995)).Shift(itxShr(11)).Average(zero).Sub(x27)
	v80 := x11.MulAdd(km2106, x21.Mul(km583)).Shift(itxShr(11)).Average(zero).Add(x21)
	v81 := x19.MulAdd(km1645, x13.Mul(kp1220)).Shift(itxShr(10)).Average(zero)
	v82 := x3.MulAdd(km601, x29.Mul(km44)).Shift(itxShr(11)).Average(zero).Add(x29)
	v83 := x3.MulAdd(km44, x29.Mul(kp601)).Shift(itxShr(11)).Average(zero).Add(x3)
	v84 := x19.MulAdd(kp1220, x13.Mul(kp1645)).Shift(itxShr(10)).Average(zero)
	v85 := x11.MulAdd(km583, x21.Mul(kp2106)).Shift(itxShr(11)).Average(zero).Add(x11)
	v86 := x27.MulAdd(kp995, x5.Mul(km123)).Shift(itxShr(11)).Average(zero).Add(x5)
	v87 := x7.MulAdd(km239, x25.Mul(kp1380)).Shift(itxShr(11)).Average(zero).Add(x7)
	v88 := x23.MulAdd(kp1751, x9.Mul(km393)).Shift(itxShr(11)).Average(zero).Add(x9)
	v89 := x15.MulAdd(km1061, x17.Mul(kp2751)).Shift(itxShr(11)).Average(zero).Add(x15)
	v90 := x31.MulAdd(kp201, x1.Mul(km5)).Shift(itxShr(11)).Average(zero).Add(x1)
	v91 := v75.Add(v76).Max(minV).Min(maxV)
	v92 := v75.Sub(v76).Max(minV).Min(maxV)
	v93 := v78.Sub(v77).Max(minV).Min(maxV)
	v94 := v78.Add(v77).Max(minV).Min(maxV)
	v95 := v79.Add(v80).Max(minV).Min(maxV)
	v96 := v79.Sub(v80).Max(minV).Min(maxV)
	v97 := v82.Sub(v81).Max(minV).Min(maxV)
	v98 := v82.Add(v81).Max(minV).Min(maxV)
	v99 := v83.Add(v84).Max(minV).Min(maxV)
	v100 := v83.Sub(v84).Max(minV).Min(maxV)
	v101 := v86.Sub(v85).Max(minV).Min(maxV)
	v102 := v86.Add(v85).Max(minV).Min(maxV)
	v103 := v87.Add(v88).Max(minV).Min(maxV)
	v104 := v87.Sub(v88).Max(minV).Min(maxV)
	v105 := v90.Sub(v89).Max(minV).Min(maxV)
	v106 := v90.Add(v89).Max(minV).Min(maxV)
	v107 := v92.MulAdd(kp79, v105.Mul(kp799)).Shift(itxShr(11)).Average(zero).Sub(v92)
	v108 := v92.MulAdd(kp799, v105.Mul(km79)).Shift(itxShr(11)).Average(zero).Add(v105)
	v109 := zero.Sub(v93.MulAdd(kp799, v104.Mul(km79))).Shift(itxShr(11)).Average(zero).Sub(v104)
	v110 := v93.MulAdd(kp79, v104.Mul(kp799)).Shift(itxShr(11)).Average(zero).Sub(v93)
	v111 := v96.MulAdd(km1138, v101.Mul(kp1703)).Shift(itxShr(10)).Average(zero)
	v112 := v96.MulAdd(kp1703, v101.Mul(kp1138)).Shift(itxShr(10)).Average(zero)
	v113 := zero.Sub(v97.MulAdd(kp1703, v100.Mul(kp1138))).Shift(itxShr(10)).Average(zero)
	v114 := v97.MulAdd(km1138, v100.Mul(kp1703)).Shift(itxShr(10)).Average(zero)
	v115 := v91.Add(v94).Max(minV).Min(maxV)
	v116 := v107.Add(v109).Max(minV).Min(maxV)
	v117 := v107.Sub(v109).Max(minV).Min(maxV)
	v118 := v91.Sub(v94).Max(minV).Min(maxV)
	v119 := v98.Sub(v95).Max(minV).Min(maxV)
	v120 := v113.Sub(v111).Max(minV).Min(maxV)
	v121 := v113.Add(v111).Max(minV).Min(maxV)
	v122 := v98.Add(v95).Max(minV).Min(maxV)
	v123 := v99.Add(v102).Max(minV).Min(maxV)
	v124 := v114.Add(v112).Max(minV).Min(maxV)
	v125 := v114.Sub(v112).Max(minV).Min(maxV)
	v126 := v99.Sub(v102).Max(minV).Min(maxV)
	v127 := v106.Sub(v103).Max(minV).Min(maxV)
	v128 := v108.Sub(v110).Max(minV).Min(maxV)
	v129 := v108.Add(v110).Max(minV).Min(maxV)
	v130 := v106.Add(v103).Max(minV).Min(maxV)
	v131 := v117.MulAdd(kp312, v128.Mul(kp1567)).Shift(itxShr(11)).Average(zero).Sub(v117)
	v132 := v117.MulAdd(kp1567, v128.Mul(km312)).Shift(itxShr(11)).Average(zero).Add(v128)
	v133 := v118.MulAdd(kp312, v127.Mul(kp1567)).Shift(itxShr(11)).Average(zero).Sub(v118)
	v134 := v118.MulAdd(kp1567, v127.Mul(km312)).Shift(itxShr(11)).Average(zero).Add(v127)
	v135 := zero.Sub(v119.MulAdd(kp1567, v126.Mul(km312))).Shift(itxShr(11)).Average(zero).Sub(v126)
	v136 := v119.MulAdd(kp312, v126.Mul(kp1567)).Shift(itxShr(11)).Average(zero).Sub(v119)
	v137 := zero.Sub(v120.MulAdd(kp1567, v125.Mul(km312))).Shift(itxShr(11)).Average(zero).Sub(v125)
	v138 := v120.MulAdd(kp312, v125.Mul(kp1567)).Shift(itxShr(11)).Average(zero).Sub(v120)
	v139 := v115.Add(v122).Max(minV).Min(maxV)
	v140 := v116.Add(v121).Max(minV).Min(maxV)
	v141 := v131.Add(v137).Max(minV).Min(maxV)
	v142 := v133.Add(v135).Max(minV).Min(maxV)
	v143 := v133.Sub(v135).Max(minV).Min(maxV)
	v144 := v131.Sub(v137).Max(minV).Min(maxV)
	v145 := v116.Sub(v121).Max(minV).Min(maxV)
	v146 := v115.Sub(v122).Max(minV).Min(maxV)
	v147 := v130.Sub(v123).Max(minV).Min(maxV)
	v148 := v129.Sub(v124).Max(minV).Min(maxV)
	v149 := v132.Sub(v138).Max(minV).Min(maxV)
	v150 := v134.Sub(v136).Max(minV).Min(maxV)
	v151 := v134.Add(v136).Max(minV).Min(maxV)
	v152 := v132.Add(v138).Max(minV).Min(maxV)
	v153 := v129.Add(v124).Max(minV).Min(maxV)
	v154 := v130.Add(v123).Max(minV).Min(maxV)
	v155 := v150.Sub(v143).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v156 := v150.Add(v143).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v157 := v149.Sub(v144).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v158 := v149.Add(v144).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v159 := v148.Sub(v145).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v160 := v148.Add(v145).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v161 := v147.Sub(v146).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v162 := v147.Add(v146).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v163 := v59.Add(v154).Max(minV).Min(maxV)
	v164 := v60.Add(v153).Max(minV).Min(maxV)
	v165 := v61.Add(v152).Max(minV).Min(maxV)
	v166 := v62.Add(v151).Max(minV).Min(maxV)
	v167 := v63.Add(v156).Max(minV).Min(maxV)
	v168 := v64.Add(v158).Max(minV).Min(maxV)
	v169 := v65.Add(v160).Max(minV).Min(maxV)
	v170 := v66.Add(v162).Max(minV).Min(maxV)
	v171 := v67.Add(v161).Max(minV).Min(maxV)
	v172 := v68.Add(v159).Max(minV).Min(maxV)
	v173 := v69.Add(v157).Max(minV).Min(maxV)
	v174 := v70.Add(v155).Max(minV).Min(maxV)
	v175 := v71.Add(v142).Max(minV).Min(maxV)
	v176 := v72.Add(v141).Max(minV).Min(maxV)
	v177 := v73.Add(v140).Max(minV).Min(maxV)
	v178 := v74.Add(v139).Max(minV).Min(maxV)
	v179 := v74.Sub(v139).Max(minV).Min(maxV)
	v180 := v73.Sub(v140).Max(minV).Min(maxV)
	v181 := v72.Sub(v141).Max(minV).Min(maxV)
	v182 := v71.Sub(v142).Max(minV).Min(maxV)
	v183 := v70.Sub(v155).Max(minV).Min(maxV)
	v184 := v69.Sub(v157).Max(minV).Min(maxV)
	v185 := v68.Sub(v159).Max(minV).Min(maxV)
	v186 := v67.Sub(v161).Max(minV).Min(maxV)
	v187 := v66.Sub(v162).Max(minV).Min(maxV)
	v188 := v65.Sub(v160).Max(minV).Min(maxV)
	v189 := v64.Sub(v158).Max(minV).Min(maxV)
	v190 := v63.Sub(v156).Max(minV).Min(maxV)
	v191 := v62.Sub(v151).Max(minV).Min(maxV)
	v192 := v61.Sub(v152).Max(minV).Min(maxV)
	v193 := v60.Sub(v153).Max(minV).Min(maxV)
	v194 := v59.Sub(v154).Max(minV).Min(maxV)
	v163.StorePart(buf[0*rowStride : 0*rowStride+2])
	v164.StorePart(buf[1*rowStride : 1*rowStride+2])
	v165.StorePart(buf[2*rowStride : 2*rowStride+2])
	v166.StorePart(buf[3*rowStride : 3*rowStride+2])
	v167.StorePart(buf[4*rowStride : 4*rowStride+2])
	v168.StorePart(buf[5*rowStride : 5*rowStride+2])
	v169.StorePart(buf[6*rowStride : 6*rowStride+2])
	v170.StorePart(buf[7*rowStride : 7*rowStride+2])
	v171.StorePart(buf[8*rowStride : 8*rowStride+2])
	v172.StorePart(buf[9*rowStride : 9*rowStride+2])
	v173.StorePart(buf[10*rowStride : 10*rowStride+2])
	v174.StorePart(buf[11*rowStride : 11*rowStride+2])
	v175.StorePart(buf[12*rowStride : 12*rowStride+2])
	v176.StorePart(buf[13*rowStride : 13*rowStride+2])
	v177.StorePart(buf[14*rowStride : 14*rowStride+2])
	v178.StorePart(buf[15*rowStride : 15*rowStride+2])
	v179.StorePart(buf[16*rowStride : 16*rowStride+2])
	v180.StorePart(buf[17*rowStride : 17*rowStride+2])
	v181.StorePart(buf[18*rowStride : 18*rowStride+2])
	v182.StorePart(buf[19*rowStride : 19*rowStride+2])
	v183.StorePart(buf[20*rowStride : 20*rowStride+2])
	v184.StorePart(buf[21*rowStride : 21*rowStride+2])
	v185.StorePart(buf[22*rowStride : 22*rowStride+2])
	v186.StorePart(buf[23*rowStride : 23*rowStride+2])
	v187.StorePart(buf[24*rowStride : 24*rowStride+2])
	v188.StorePart(buf[25*rowStride : 25*rowStride+2])
	v189.StorePart(buf[26*rowStride : 26*rowStride+2])
	v190.StorePart(buf[27*rowStride : 27*rowStride+2])
	v191.StorePart(buf[28*rowStride : 28*rowStride+2])
	v192.StorePart(buf[29*rowStride : 29*rowStride+2])
	v193.StorePart(buf[30*rowStride : 30*rowStride+2])
	v194.StorePart(buf[31*rowStride : 31*rowStride+2])
}

// inverseDCT8Row2SIMD applies inverseDCT8 to 2 rows held as lanes of transposed vectors.
func inverseDCT8Row2SIMD(r0, r1 []int32, min, max int32) {
	minV := archsimd.BroadcastInt32x4(min)
	maxV := archsimd.BroadcastInt32x4(max)
	var zero archsimd.Int32x4
	km1138 := archsimd.LoadInt32x4Array(&itxCm1138)
	km312 := archsimd.LoadInt32x4Array(&itxCm312)
	km79 := archsimd.LoadInt32x4Array(&itxCm79)
	kp79 := archsimd.LoadInt32x4Array(&itxCp79)
	kp181 := archsimd.LoadInt32x4Array(&itxCp181)
	kp312 := archsimd.LoadInt32x4Array(&itxCp312)
	kp799 := archsimd.LoadInt32x4Array(&itxCp799)
	kp1138 := archsimd.LoadInt32x4Array(&itxCp1138)
	kp1567 := archsimd.LoadInt32x4Array(&itxCp1567)
	kp1703 := archsimd.LoadInt32x4Array(&itxCp1703)
	x0, x1, x2, x3 := transpose4x4(archsimd.LoadInt32x4Array((*[4]int32)(r0[0:])), archsimd.LoadInt32x4Array((*[4]int32)(r1[0:])), zero, zero)
	x4, x5, x6, x7 := transpose4x4(archsimd.LoadInt32x4Array((*[4]int32)(r0[4:])), archsimd.LoadInt32x4Array((*[4]int32)(r1[4:])), zero, zero)
	v1 := x0.Add(x4).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v2 := x0.Sub(x4).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v3 := x6.MulAdd(kp312, x2.Mul(kp1567)).Shift(itxShr(11)).Average(zero).Sub(x6)
	v4 := x6.MulAdd(kp1567, x2.Mul(km312)).Shift(itxShr(11)).Average(zero).Add(x2)
	v5 := v1.Add(v4).Max(minV).Min(maxV)
	v6 := v2.Add(v3).Max(minV).Min(maxV)
	v7 := v2.Sub(v3).Max(minV).Min(maxV)
	v8 := v1.Sub(v4).Max(minV).Min(maxV)
	v9 := x7.MulAdd(kp79, x1.Mul(kp799)).Shift(itxShr(11)).Average(zero).Sub(x7)
	v10 := x3.MulAdd(km1138, x5.Mul(kp1703)).Shift(itxShr(10)).Average(zero)
	v11 := x3.MulAdd(kp1703, x5.Mul(kp1138)).Shift(itxShr(10)).Average(zero)
	v12 := x7.MulAdd(kp799, x1.Mul(km79)).Shift(itxShr(11)).Average(zero).Add(x1)
	v13 := v9.Add(v10).Max(minV).Min(maxV)
	v14 := v9.Sub(v10).Max(minV).Min(maxV)
	v15 := v12.Add(v11).Max(minV).Min(maxV)
	v16 := v12.Sub(v11).Max(minV).Min(maxV)
	v17 := v16.Sub(v14).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v18 := v16.Add(v14).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v19 := v5.Add(v15).Max(minV).Min(maxV)
	v20 := v6.Add(v18).Max(minV).Min(maxV)
	v21 := v7.Add(v17).Max(minV).Min(maxV)
	v22 := v8.Add(v13).Max(minV).Min(maxV)
	v23 := v8.Sub(v13).Max(minV).Min(maxV)
	v24 := v7.Sub(v17).Max(minV).Min(maxV)
	v25 := v6.Sub(v18).Max(minV).Min(maxV)
	v26 := v5.Sub(v15).Max(minV).Min(maxV)
	y0_0, y0_1, _, _ := transpose4x4(v19, v20, v21, v22)
	y0_0.StoreArray((*[4]int32)(r0[0:]))
	y0_1.StoreArray((*[4]int32)(r1[0:]))
	y1_0, y1_1, _, _ := transpose4x4(v23, v24, v25, v26)
	y1_0.StoreArray((*[4]int32)(r0[4:]))
	y1_1.StoreArray((*[4]int32)(r1[4:]))
}

// inverseDCT16Row2SIMD applies inverseDCT16 to 2 rows held as lanes of transposed vectors.
func inverseDCT16Row2SIMD(r0, r1 []int32, min, max int32) {
	minV := archsimd.BroadcastInt32x4(min)
	maxV := archsimd.BroadcastInt32x4(max)
	var zero archsimd.Int32x4
	km1299 := archsimd.LoadInt32x4Array(&itxCm1299)
	km1189 := archsimd.LoadInt32x4Array(&itxCm1189)
	km1138 := archsimd.LoadInt32x4Array(&itxCm1138)
	km484 := archsimd.LoadInt32x4Array(&itxCm484)
	km312 := archsimd.LoadInt32x4Array(&itxCm312)
	km176 := archsimd.LoadInt32x4Array(&itxCm176)
	km79 := archsimd.LoadInt32x4Array(&itxCm79)
	km20 := archsimd.LoadInt32x4Array(&itxCm20)
	kp20 := archsimd.LoadInt32x4Array(&itxCp20)
	kp79 := archsimd.LoadInt32x4Array(&itxCp79)
	kp181 := archsimd.LoadInt32x4Array(&itxCp181)
	kp312 := archsimd.LoadInt32x4Array(&itxCp312)
	kp401 := archsimd.LoadInt32x4Array(&itxCp401)
	kp484 := archsimd.LoadInt32x4Array(&itxCp484)
	kp799 := archsimd.LoadInt32x4Array(&itxCp799)
	kp1138 := archsimd.LoadInt32x4Array(&itxCp1138)
	kp1189 := archsimd.LoadInt32x4Array(&itxCp1189)
	kp1299 := archsimd.LoadInt32x4Array(&itxCp1299)
	kp1567 := archsimd.LoadInt32x4Array(&itxCp1567)
	kp1583 := archsimd.LoadInt32x4Array(&itxCp1583)
	kp1703 := archsimd.LoadInt32x4Array(&itxCp1703)
	kp1931 := archsimd.LoadInt32x4Array(&itxCp1931)
	x0, x1, x2, x3 := transpose4x4(archsimd.LoadInt32x4Array((*[4]int32)(r0[0:])), archsimd.LoadInt32x4Array((*[4]int32)(r1[0:])), zero, zero)
	x4, x5, x6, x7 := transpose4x4(archsimd.LoadInt32x4Array((*[4]int32)(r0[4:])), archsimd.LoadInt32x4Array((*[4]int32)(r1[4:])), zero, zero)
	x8, x9, x10, x11 := transpose4x4(archsimd.LoadInt32x4Array((*[4]int32)(r0[8:])), archsimd.LoadInt32x4Array((*[4]int32)(r1[8:])), zero, zero)
	x12, x13, x14, x15 := transpose4x4(archsimd.LoadInt32x4Array((*[4]int32)(r0[12:])), archsimd.LoadInt32x4Array((*[4]int32)(r1[12:])), zero, zero)
	v1 := x0.Add(x8).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v2 := x0.Sub(x8).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v3 := x12.MulAdd(kp312, x4.Mul(kp1567)).Shift(itxShr(11)).Average(zero).Sub(x12)
	v4 := x12.MulAdd(kp1567, x4.Mul(km312)).Shift(itxShr(11)).Average(zero).Add(x4)
	v5 := v1.Add(v4).Max(minV).Min(maxV)
	v6 := v2.Add(v3).Max(minV).Min(maxV)
	v7 := v2.Sub(v3).Max(minV).Min(maxV)
	v8 := v1.Sub(v4).Max(minV).Min(maxV)
	v9 := x14.MulAdd(kp79, x2.Mul(kp799)).Shift(itxShr(11)).Average(zero).Sub(x14)
	v10 := x6.MulAdd(km1138, x10.Mul(kp1703)).Shift(itxShr(10)).Average(zero)
	v11 := x6.MulAdd(kp1703, x10.Mul(kp1138)).Shift(itxShr(10)).Average(zero)
	v12 := x14.MulAdd(kp799, x2.Mul(km79)).Shift(itxShr(11)).Average(zero).Add(x2)
	v13 := v9.Add(v10).Max(minV).Min(maxV)
	v14 := v9.Sub(v10).Max(minV).Min(maxV)
	v15 := v12.Add(v11).Max(minV).Min(maxV)
	v16 := v12.Sub(v11).Max(minV).Min(maxV)
	v17 := v16.Sub(v14).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v18 := v16.Add(v14).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v19 := v5.Add(v15).Max(minV).Min(maxV)
	v20 := v6.Add(v18).Max(minV).Min(maxV)
	v21 := v7.Add(v17).Max(minV).Min(maxV)
	v22 := v8.Add(v13).Max(minV).Min(maxV)
	v23 := v8.Sub(v13).Max(minV).Min(maxV)
	v24 := v7.Sub(v17).Max(minV).Min(maxV)
	v25 := v6.Sub(v18).Max(minV).Min(maxV)
	v26 := v5.Sub(v15).Max(minV).Min(maxV)
	v27 := x15.MulAdd(kp20, x1.Mul(kp401)).Shift(itxShr(11)).Average(zero).Sub(x15)
	v28 := x7.MulAdd(km1299, x9.Mul(kp1583)).Shift(itxShr(10)).Average(zero)
	v29 := x11.MulAdd(kp484, x5.Mul(kp1931)).Shift(itxShr(11)).Average(zero).Sub(x11)
	v30 := x3.MulAdd(km1189, x13.Mul(km176)).Shift(itxShr(11)).Average(zero).Add(x13)
	v31 := x3.MulAdd(km176, x13.Mul(kp1189)).Shift(itxShr(11)).Average(zero).Add(x3)
	v32 := x11.MulAdd(kp1931, x5.Mul(km484)).Shift(itxShr(11)).Average(zero).Add(x5)
	v33 := x7.MulAdd(kp1583, x9.Mul(kp1299)).Shift(itxShr(10)).Average(zero)
	v34 := x15.MulAdd(kp401, x1.Mul(km20)).Shift(itxShr(11)).Average(zero).Add(x1)
	v35 := v27.Add(v28).Max(minV).Min(maxV)
	v36 := v27.Sub(v28).Max(minV).Min(maxV)
	v37 := v30.Sub(v29).Max(minV).Min(maxV)
	v38 := v30.Add(v29).Max(minV).Min(maxV)
	v39 := v31.Add(v32).Max(minV).Min(maxV)
	v40 := v31.Sub(v32).Max(minV).Min(maxV)
	v41 := v34.Sub(v33).Max(minV).Min(maxV)
	v42 := v34.Add(v33).Max(minV).Min(maxV)
	v43 := v36.MulAdd(kp312, v41.Mul(kp1567)).Shift(itxShr(11)).Average(zero).Sub(v36)
	v44 := v36.MulAdd(kp1567, v41.Mul(km312)).Shift(itxShr(11)).Average(zero).Add(v41)
	v45 := zero.Sub(v37.MulAdd(kp1567, v40.Mul(km312))).Shift(itxShr(11)).Average(zero).Sub(v40)
	v46 := v37.MulAdd(kp312, v40.Mul(kp1567)).Shift(itxShr(11)).Average(zero).Sub(v37)
	v47 := v35.Add(v38).Max(minV).Min(maxV)
	v48 := v43.Add(v45).Max(minV).Min(maxV)
	v49 := v43.Sub(v45).Max(minV).Min(maxV)
	v50 := v35.Sub(v38).Max(minV).Min(maxV)
	v51 := v42.Sub(v39).Max(minV).Min(maxV)
	v52 := v44.Sub(v46).Max(minV).Min(maxV)
	v53 := v44.Add(v46).Max(minV).Min(maxV)
	v54 := v42.Add(v39).Max(minV).Min(maxV)
	v55 := v52.Sub(v49).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v56 := v52.Add(v49).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v57 := v51.Sub(v50).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v58 := v51.Add(v50).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v59 := v19.Add(v54).Max(minV).Min(maxV)
	v60 := v20.Add(v53).Max(minV).Min(maxV)
	v61 := v21.Add(v56).Max(minV).Min(maxV)
	v62 := v22.Add(v58).Max(minV).Min(maxV)
	v63 := v23.Add(v57).Max(minV).Min(maxV)
	v64 := v24.Add(v55).Max(minV).Min(maxV)
	v65 := v25.Add(v48).Max(minV).Min(maxV)
	v66 := v26.Add(v47).Max(minV).Min(maxV)
	v67 := v26.Sub(v47).Max(minV).Min(maxV)
	v68 := v25.Sub(v48).Max(minV).Min(maxV)
	v69 := v24.Sub(v55).Max(minV).Min(maxV)
	v70 := v23.Sub(v57).Max(minV).Min(maxV)
	v71 := v22.Sub(v58).Max(minV).Min(maxV)
	v72 := v21.Sub(v56).Max(minV).Min(maxV)
	v73 := v20.Sub(v53).Max(minV).Min(maxV)
	v74 := v19.Sub(v54).Max(minV).Min(maxV)
	y0_0, y0_1, _, _ := transpose4x4(v59, v60, v61, v62)
	y0_0.StoreArray((*[4]int32)(r0[0:]))
	y0_1.StoreArray((*[4]int32)(r1[0:]))
	y1_0, y1_1, _, _ := transpose4x4(v63, v64, v65, v66)
	y1_0.StoreArray((*[4]int32)(r0[4:]))
	y1_1.StoreArray((*[4]int32)(r1[4:]))
	y2_0, y2_1, _, _ := transpose4x4(v67, v68, v69, v70)
	y2_0.StoreArray((*[4]int32)(r0[8:]))
	y2_1.StoreArray((*[4]int32)(r1[8:]))
	y3_0, y3_1, _, _ := transpose4x4(v71, v72, v73, v74)
	y3_0.StoreArray((*[4]int32)(r0[12:]))
	y3_1.StoreArray((*[4]int32)(r1[12:]))
}

// inverseDCT32Row2SIMD applies inverseDCT32 to 2 rows held as lanes of transposed vectors.
func inverseDCT32Row2SIMD(r0, r1 []int32, min, max int32) {
	minV := archsimd.BroadcastInt32x4(min)
	maxV := archsimd.BroadcastInt32x4(max)
	var zero archsimd.Int32x4
	km2751 := archsimd.LoadInt32x4Array(&itxCm2751)
	km2106 := archsimd.LoadInt32x4Array(&itxCm2106)
	km1645 := archsimd.LoadInt32x4Array(&itxCm1645)
	km1380 := archsimd.LoadInt32x4Array(&itxCm1380)
	km1299 := archsimd.LoadInt32x4Array(&itxCm1299)
	km1189 := archsimd.LoadInt32x4Array(&itxCm1189)
	km1138 := archsimd.LoadInt32x4Array(&itxCm1138)
	km1061 := archsimd.LoadInt32x4Array(&itxCm1061)
	km601 := archsimd.LoadInt32x4Array(&itxCm601)
	km583 := archsimd.LoadInt32x4Array(&itxCm583)
	km484 := archsimd.LoadInt32x4Array(&itxCm484)
	km393 := archsimd.LoadInt32x4Array(&itxCm393)
	km312 := archsimd.LoadInt32x4Array(&itxCm312)
	km239 := archsimd.LoadInt32x4Array(&itxCm239)
	km176 := archsimd.LoadInt32x4Array(&itxCm176)
	km123 := archsimd.LoadInt32x4Array(&itxCm123)
	km79 := archsimd.LoadInt32x4Array(&itxCm79)
	km44 := archsimd.LoadInt32x4Array(&itxCm44)
	km20 := archsimd.LoadInt32x4Array(&itxCm20)
	km5 := archsimd.LoadInt32x4Array(&itxCm5)
	kp5 := archsimd.LoadInt32x4Array(&itxCp5)
	kp20 := archsimd.LoadInt32x4Array(&itxCp20)
	kp79 := archsimd.LoadInt32x4Array(&itxCp79)
	kp123 := archsimd.LoadInt32x4Array(&itxCp123)
	kp181 := archsimd.LoadInt32x4Array(&itxCp181)
	kp201 := archsimd.LoadInt32x4Array(&itxCp201)
	kp312 := archsimd.LoadInt32x4Array(&itxCp312)
	kp393 := archsimd.LoadInt32x4Array(&itxCp393)
	kp401 := archsimd.LoadInt32x4Array(&itxCp401)
	kp484 := archsimd.LoadInt32x4Array(&itxCp484)
	kp601 := archsimd.LoadInt32x4Array(&itxCp601)
	kp799 := archsimd.LoadInt32x4Array(&itxCp799)
	kp995 := archsimd.LoadInt32x4Array(&itxCp995)
	kp1138 := archsimd.LoadInt32x4Array(&itxCp1138)
	kp1189 := archsimd.LoadInt32x4Array(&itxCp1189)
	kp1220 := archsimd.LoadInt32x4Array(&itxCp1220)
	kp1299 := archsimd.LoadInt32x4Array(&itxCp1299)
	kp1380 := archsimd.LoadInt32x4Array(&itxCp1380)
	kp1567 := archsimd.LoadInt32x4Array(&itxCp1567)
	kp1583 := archsimd.LoadInt32x4Array(&itxCp1583)
	kp1645 := archsimd.LoadInt32x4Array(&itxCp1645)
	kp1703 := archsimd.LoadInt32x4Array(&itxCp1703)
	kp1751 := archsimd.LoadInt32x4Array(&itxCp1751)
	kp1931 := archsimd.LoadInt32x4Array(&itxCp1931)
	kp2106 := archsimd.LoadInt32x4Array(&itxCp2106)
	kp2751 := archsimd.LoadInt32x4Array(&itxCp2751)
	x0, x1, x2, x3 := transpose4x4(archsimd.LoadInt32x4Array((*[4]int32)(r0[0:])), archsimd.LoadInt32x4Array((*[4]int32)(r1[0:])), zero, zero)
	x4, x5, x6, x7 := transpose4x4(archsimd.LoadInt32x4Array((*[4]int32)(r0[4:])), archsimd.LoadInt32x4Array((*[4]int32)(r1[4:])), zero, zero)
	x8, x9, x10, x11 := transpose4x4(archsimd.LoadInt32x4Array((*[4]int32)(r0[8:])), archsimd.LoadInt32x4Array((*[4]int32)(r1[8:])), zero, zero)
	x12, x13, x14, x15 := transpose4x4(archsimd.LoadInt32x4Array((*[4]int32)(r0[12:])), archsimd.LoadInt32x4Array((*[4]int32)(r1[12:])), zero, zero)
	x16, x17, x18, x19 := transpose4x4(archsimd.LoadInt32x4Array((*[4]int32)(r0[16:])), archsimd.LoadInt32x4Array((*[4]int32)(r1[16:])), zero, zero)
	x20, x21, x22, x23 := transpose4x4(archsimd.LoadInt32x4Array((*[4]int32)(r0[20:])), archsimd.LoadInt32x4Array((*[4]int32)(r1[20:])), zero, zero)
	x24, x25, x26, x27 := transpose4x4(archsimd.LoadInt32x4Array((*[4]int32)(r0[24:])), archsimd.LoadInt32x4Array((*[4]int32)(r1[24:])), zero, zero)
	x28, x29, x30, x31 := transpose4x4(archsimd.LoadInt32x4Array((*[4]int32)(r0[28:])), archsimd.LoadInt32x4Array((*[4]int32)(r1[28:])), zero, zero)
	v1 := x0.Add(x16).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v2 := x0.Sub(x16).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v3 := x24.MulAdd(kp312, x8.Mul(kp1567)).Shift(itxShr(11)).Average(zero).Sub(x24)
	v4 := x24.MulAdd(kp1567, x8.Mul(km312)).Shift(itxShr(11)).Average(zero).Add(x8)
	v5 := v1.Add(v4).Max(minV).Min(maxV)
	v6 := v2.Add(v3).Max(minV).Min(maxV)
	v7 := v2.Sub(v3).Max(minV).Min(maxV)
	v8 := v1.Sub(v4).Max(minV).Min(maxV)
	v9 := x28.MulAdd(kp79, x4.Mul(kp799)).Shift(itxShr(11)).Average(zero).Sub(x28)
	v10 := x12.MulAdd(km1138, x20.Mul(kp1703)).Shift(itxShr(10)).Average(zero)
	v11 := x12.MulAdd(kp1703, x20.Mul(kp1138)).Shift(itxShr(10)).Average(zero)
	v12 := x28.MulAdd(kp799, x4.Mul(km79)).Shift(itxShr(11)).Average(zero).Add(x4)
	v13 := v9.Add(v10).Max(minV).Min(maxV)
	v14 := v9.Sub(v10).Max(minV).Min(maxV)
	v15 := v12.Add(v11).Max(minV).Min(maxV)
	v16 := v12.Sub(v11).Max(minV).Min(maxV)
	v17 := v16.Sub(v14).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v18 := v16.Add(v14).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v19 := v5.Add(v15).Max(minV).Min(maxV)
	v20 := v6.Add(v18).Max(minV).Min(maxV)
	v21 := v7.Add(v17).Max(minV).Min(maxV)
	v22 := v8.Add(v13).Max(minV).Min(maxV)
	v23 := v8.Sub(v13).Max(minV).Min(maxV)
	v24 := v7.Sub(v17).Max(minV).Min(maxV)
	v25 := v6.Sub(v18).Max(minV).Min(maxV)
	v26 := v5.Sub(v15).Max(minV).Min(maxV)
	v27 := x30.MulAdd(kp20, x2.Mul(kp401)).Shift(itxShr(11)).Average(zero).Sub(x30)
	v28 := x14.MulAdd(km1299, x18.Mul(kp1583)).Shift(itxShr(10)).Average(zero)
	v29 := x22.MulAdd(kp484, x10.Mul(kp1931)).Shift(itxShr(11)).Average(zero).Sub(x22)
	v30 := x6.MulAdd(km1189, x26.Mul(km176)).Shift(itxShr(11)).Average(zero).Add(x26)
	v31 := x6.MulAdd(km176, x26.Mul(kp1189)).Shift(itxShr(11)).Average(zero).Add(x6)
	v32 := x22.MulAdd(kp1931, x10.Mul(km484)).Shift(itxShr(11)).Average(zero).Add(x10)
	v33 := x14.MulAdd(kp1583, x18.Mul(kp1299)).Shift(itxShr(10)).Average(zero)
	v34 := x30.MulAdd(kp401, x2.Mul(km20)).Shift(itxShr(11)).Average(zero).Add(x2)
	v35 := v27.Add(v28).Max(minV).Min(maxV)
	v36 := v27.Sub(v28).Max(minV).Min(maxV)
	v37 := v30.Sub(v29).Max(minV).Min(maxV)
	v38 := v30.Add(v29).Max(minV).Min(maxV)
	v39 := v31.Add(v32).Max(minV).Min(maxV)
	v40 := v31.Sub(v32).Max(minV).Min(maxV)
	v41 := v34.Sub(v33).Max(minV).Min(maxV)
	v42 := v34.Add(v33).Max(minV).Min(maxV)
	v43 := v36.MulAdd(kp312, v41.Mul(kp1567)).Shift(itxShr(11)).Average(zero).Sub(v36)
	v44 := v36.MulAdd(kp1567, v41.Mul(km312)).Shift(itxShr(11)).Average(zero).Add(v41)
	v45 := zero.Sub(v37.MulAdd(kp1567, v40.Mul(km312))).Shift(itxShr(11)).Average(zero).Sub(v40)
	v46 := v37.MulAdd(kp312, v40.Mul(kp1567)).Shift(itxShr(11)).Average(zero).Sub(v37)
	v47 := v35.Add(v38).Max(minV).Min(maxV)
	v48 := v43.Add(v45).Max(minV).Min(maxV)
	v49 := v43.Sub(v45).Max(minV).Min(maxV)
	v50 := v35.Sub(v38).Max(minV).Min(maxV)
	v51 := v42.Sub(v39).Max(minV).Min(maxV)
	v52 := v44.Sub(v46).Max(minV).Min(maxV)
	v53 := v44.Add(v46).Max(minV).Min(maxV)
	v54 := v42.Add(v39).Max(minV).Min(maxV)
	v55 := v52.Sub(v49).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v56 := v52.Add(v49).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v57 := v51.Sub(v50).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v58 := v51.Add(v50).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v59 := v19.Add(v54).Max(minV).Min(maxV)
	v60 := v20.Add(v53).Max(minV).Min(maxV)
	v61 := v21.Add(v56).Max(minV).Min(maxV)
	v62 := v22.Add(v58).Max(minV).Min(maxV)
	v63 := v23.Add(v57).Max(minV).Min(maxV)
	v64 := v24.Add(v55).Max(minV).Min(maxV)
	v65 := v25.Add(v48).Max(minV).Min(maxV)
	v66 := v26.Add(v47).Max(minV).Min(maxV)
	v67 := v26.Sub(v47).Max(minV).Min(maxV)
	v68 := v25.Sub(v48).Max(minV).Min(maxV)
	v69 := v24.Sub(v55).Max(minV).Min(maxV)
	v70 := v23.Sub(v57).Max(minV).Min(maxV)
	v71 := v22.Sub(v58).Max(minV).Min(maxV)
	v72 := v21.Sub(v56).Max(minV).Min(maxV)
	v73 := v20.Sub(v53).Max(minV).Min(maxV)
	v74 := v19.Sub(v54).Max(minV).Min(maxV)
	v75 := x31.MulAdd(kp5, x1.Mul(kp201)).Shift(itxShr(11)).Average(zero).Sub(x31)
	v76 := x15.MulAdd(km2751, x17.Mul(km1061)).Shift(itxShr(11)).Average(zero).Add(x17)
	v77 := x23.MulAdd(kp393, x9.Mul(kp1751)).Shift(itxShr(11)).Average(zero).Sub(x23)
	v78 := x7.MulAdd(km1380, x25.Mul(km239)).Shift(itxShr(11)).Average(zero).Add(x25)
	v79 := x27.MulAdd(kp123, x5.Mul(kp995)).Shift(itxShr(11)).Average(zero).Sub(x27)
	v80 := x11.MulAdd(km2106, x21.Mul(km583)).Shift(itxShr(11)).Average(zero).Add(x21)
	v81 := x19.MulAdd(km1645, x13.Mul(kp1220)).Shift(itxShr(10)).Average(zero)
	v82 := x3.MulAdd(km601, x29.Mul(km44)).Shift(itxShr(11)).Average(zero).Add(x29)
	v83 := x3.MulAdd(km44, x29.Mul(kp601)).Shift(itxShr(11)).Average(zero).Add(x3)
	v84 := x19.MulAdd(kp1220, x13.Mul(kp1645)).Shift(itxShr(10)).Average(zero)
	v85 := x11.MulAdd(km583, x21.Mul(kp2106)).Shift(itxShr(11)).Average(zero).Add(x11)
	v86 := x27.MulAdd(kp995, x5.Mul(km123)).Shift(itxShr(11)).Average(zero).Add(x5)
	v87 := x7.MulAdd(km239, x25.Mul(kp1380)).Shift(itxShr(11)).Average(zero).Add(x7)
	v88 := x23.MulAdd(kp1751, x9.Mul(km393)).Shift(itxShr(11)).Average(zero).Add(x9)
	v89 := x15.MulAdd(km1061, x17.Mul(kp2751)).Shift(itxShr(11)).Average(zero).Add(x15)
	v90 := x31.MulAdd(kp201, x1.Mul(km5)).Shift(itxShr(11)).Average(zero).Add(x1)
	v91 := v75.Add(v76).Max(minV).Min(maxV)
	v92 := v75.Sub(v76).Max(minV).Min(maxV)
	v93 := v78.Sub(v77).Max(minV).Min(maxV)
	v94 := v78.Add(v77).Max(minV).Min(maxV)
	v95 := v79.Add(v80).Max(minV).Min(maxV)
	v96 := v79.Sub(v80).Max(minV).Min(maxV)
	v97 := v82.Sub(v81).Max(minV).Min(maxV)
	v98 := v82.Add(v81).Max(minV).Min(maxV)
	v99 := v83.Add(v84).Max(minV).Min(maxV)
	v100 := v83.Sub(v84).Max(minV).Min(maxV)
	v101 := v86.Sub(v85).Max(minV).Min(maxV)
	v102 := v86.Add(v85).Max(minV).Min(maxV)
	v103 := v87.Add(v88).Max(minV).Min(maxV)
	v104 := v87.Sub(v88).Max(minV).Min(maxV)
	v105 := v90.Sub(v89).Max(minV).Min(maxV)
	v106 := v90.Add(v89).Max(minV).Min(maxV)
	v107 := v92.MulAdd(kp79, v105.Mul(kp799)).Shift(itxShr(11)).Average(zero).Sub(v92)
	v108 := v92.MulAdd(kp799, v105.Mul(km79)).Shift(itxShr(11)).Average(zero).Add(v105)
	v109 := zero.Sub(v93.MulAdd(kp799, v104.Mul(km79))).Shift(itxShr(11)).Average(zero).Sub(v104)
	v110 := v93.MulAdd(kp79, v104.Mul(kp799)).Shift(itxShr(11)).Average(zero).Sub(v93)
	v111 := v96.MulAdd(km1138, v101.Mul(kp1703)).Shift(itxShr(10)).Average(zero)
	v112 := v96.MulAdd(kp1703, v101.Mul(kp1138)).Shift(itxShr(10)).Average(zero)
	v113 := zero.Sub(v97.MulAdd(kp1703, v100.Mul(kp1138))).Shift(itxShr(10)).Average(zero)
	v114 := v97.MulAdd(km1138, v100.Mul(kp1703)).Shift(itxShr(10)).Average(zero)
	v115 := v91.Add(v94).Max(minV).Min(maxV)
	v116 := v107.Add(v109).Max(minV).Min(maxV)
	v117 := v107.Sub(v109).Max(minV).Min(maxV)
	v118 := v91.Sub(v94).Max(minV).Min(maxV)
	v119 := v98.Sub(v95).Max(minV).Min(maxV)
	v120 := v113.Sub(v111).Max(minV).Min(maxV)
	v121 := v113.Add(v111).Max(minV).Min(maxV)
	v122 := v98.Add(v95).Max(minV).Min(maxV)
	v123 := v99.Add(v102).Max(minV).Min(maxV)
	v124 := v114.Add(v112).Max(minV).Min(maxV)
	v125 := v114.Sub(v112).Max(minV).Min(maxV)
	v126 := v99.Sub(v102).Max(minV).Min(maxV)
	v127 := v106.Sub(v103).Max(minV).Min(maxV)
	v128 := v108.Sub(v110).Max(minV).Min(maxV)
	v129 := v108.Add(v110).Max(minV).Min(maxV)
	v130 := v106.Add(v103).Max(minV).Min(maxV)
	v131 := v117.MulAdd(kp312, v128.Mul(kp1567)).Shift(itxShr(11)).Average(zero).Sub(v117)
	v132 := v117.MulAdd(kp1567, v128.Mul(km312)).Shift(itxShr(11)).Average(zero).Add(v128)
	v133 := v118.MulAdd(kp312, v127.Mul(kp1567)).Shift(itxShr(11)).Average(zero).Sub(v118)
	v134 := v118.MulAdd(kp1567, v127.Mul(km312)).Shift(itxShr(11)).Average(zero).Add(v127)
	v135 := zero.Sub(v119.MulAdd(kp1567, v126.Mul(km312))).Shift(itxShr(11)).Average(zero).Sub(v126)
	v136 := v119.MulAdd(kp312, v126.Mul(kp1567)).Shift(itxShr(11)).Average(zero).Sub(v119)
	v137 := zero.Sub(v120.MulAdd(kp1567, v125.Mul(km312))).Shift(itxShr(11)).Average(zero).Sub(v125)
	v138 := v120.MulAdd(kp312, v125.Mul(kp1567)).Shift(itxShr(11)).Average(zero).Sub(v120)
	v139 := v115.Add(v122).Max(minV).Min(maxV)
	v140 := v116.Add(v121).Max(minV).Min(maxV)
	v141 := v131.Add(v137).Max(minV).Min(maxV)
	v142 := v133.Add(v135).Max(minV).Min(maxV)
	v143 := v133.Sub(v135).Max(minV).Min(maxV)
	v144 := v131.Sub(v137).Max(minV).Min(maxV)
	v145 := v116.Sub(v121).Max(minV).Min(maxV)
	v146 := v115.Sub(v122).Max(minV).Min(maxV)
	v147 := v130.Sub(v123).Max(minV).Min(maxV)
	v148 := v129.Sub(v124).Max(minV).Min(maxV)
	v149 := v132.Sub(v138).Max(minV).Min(maxV)
	v150 := v134.Sub(v136).Max(minV).Min(maxV)
	v151 := v134.Add(v136).Max(minV).Min(maxV)
	v152 := v132.Add(v138).Max(minV).Min(maxV)
	v153 := v129.Add(v124).Max(minV).Min(maxV)
	v154 := v130.Add(v123).Max(minV).Min(maxV)
	v155 := v150.Sub(v143).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v156 := v150.Add(v143).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v157 := v149.Sub(v144).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v158 := v149.Add(v144).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v159 := v148.Sub(v145).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v160 := v148.Add(v145).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v161 := v147.Sub(v146).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v162 := v147.Add(v146).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v163 := v59.Add(v154).Max(minV).Min(maxV)
	v164 := v60.Add(v153).Max(minV).Min(maxV)
	v165 := v61.Add(v152).Max(minV).Min(maxV)
	v166 := v62.Add(v151).Max(minV).Min(maxV)
	v167 := v63.Add(v156).Max(minV).Min(maxV)
	v168 := v64.Add(v158).Max(minV).Min(maxV)
	v169 := v65.Add(v160).Max(minV).Min(maxV)
	v170 := v66.Add(v162).Max(minV).Min(maxV)
	v171 := v67.Add(v161).Max(minV).Min(maxV)
	v172 := v68.Add(v159).Max(minV).Min(maxV)
	v173 := v69.Add(v157).Max(minV).Min(maxV)
	v174 := v70.Add(v155).Max(minV).Min(maxV)
	v175 := v71.Add(v142).Max(minV).Min(maxV)
	v176 := v72.Add(v141).Max(minV).Min(maxV)
	v177 := v73.Add(v140).Max(minV).Min(maxV)
	v178 := v74.Add(v139).Max(minV).Min(maxV)
	v179 := v74.Sub(v139).Max(minV).Min(maxV)
	v180 := v73.Sub(v140).Max(minV).Min(maxV)
	v181 := v72.Sub(v141).Max(minV).Min(maxV)
	v182 := v71.Sub(v142).Max(minV).Min(maxV)
	v183 := v70.Sub(v155).Max(minV).Min(maxV)
	v184 := v69.Sub(v157).Max(minV).Min(maxV)
	v185 := v68.Sub(v159).Max(minV).Min(maxV)
	v186 := v67.Sub(v161).Max(minV).Min(maxV)
	v187 := v66.Sub(v162).Max(minV).Min(maxV)
	v188 := v65.Sub(v160).Max(minV).Min(maxV)
	v189 := v64.Sub(v158).Max(minV).Min(maxV)
	v190 := v63.Sub(v156).Max(minV).Min(maxV)
	v191 := v62.Sub(v151).Max(minV).Min(maxV)
	v192 := v61.Sub(v152).Max(minV).Min(maxV)
	v193 := v60.Sub(v153).Max(minV).Min(maxV)
	v194 := v59.Sub(v154).Max(minV).Min(maxV)
	y0_0, y0_1, _, _ := transpose4x4(v163, v164, v165, v166)
	y0_0.StoreArray((*[4]int32)(r0[0:]))
	y0_1.StoreArray((*[4]int32)(r1[0:]))
	y1_0, y1_1, _, _ := transpose4x4(v167, v168, v169, v170)
	y1_0.StoreArray((*[4]int32)(r0[4:]))
	y1_1.StoreArray((*[4]int32)(r1[4:]))
	y2_0, y2_1, _, _ := transpose4x4(v171, v172, v173, v174)
	y2_0.StoreArray((*[4]int32)(r0[8:]))
	y2_1.StoreArray((*[4]int32)(r1[8:]))
	y3_0, y3_1, _, _ := transpose4x4(v175, v176, v177, v178)
	y3_0.StoreArray((*[4]int32)(r0[12:]))
	y3_1.StoreArray((*[4]int32)(r1[12:]))
	y4_0, y4_1, _, _ := transpose4x4(v179, v180, v181, v182)
	y4_0.StoreArray((*[4]int32)(r0[16:]))
	y4_1.StoreArray((*[4]int32)(r1[16:]))
	y5_0, y5_1, _, _ := transpose4x4(v183, v184, v185, v186)
	y5_0.StoreArray((*[4]int32)(r0[20:]))
	y5_1.StoreArray((*[4]int32)(r1[20:]))
	y6_0, y6_1, _, _ := transpose4x4(v187, v188, v189, v190)
	y6_0.StoreArray((*[4]int32)(r0[24:]))
	y6_1.StoreArray((*[4]int32)(r1[24:]))
	y7_0, y7_1, _, _ := transpose4x4(v191, v192, v193, v194)
	y7_0.StoreArray((*[4]int32)(r0[28:]))
	y7_1.StoreArray((*[4]int32)(r1[28:]))
}

// inverseDCT32Row4SIMD applies inverseDCT32 to 4 rows held as lanes of transposed vectors.
func inverseDCT32Row4SIMD(r0, r1, r2, r3 []int32, min, max int32) {
	minV := archsimd.BroadcastInt32x4(min)
	maxV := archsimd.BroadcastInt32x4(max)
	var zero archsimd.Int32x4
	km2751 := archsimd.LoadInt32x4Array(&itxCm2751)
	km2106 := archsimd.LoadInt32x4Array(&itxCm2106)
	km1645 := archsimd.LoadInt32x4Array(&itxCm1645)
	km1380 := archsimd.LoadInt32x4Array(&itxCm1380)
	km1299 := archsimd.LoadInt32x4Array(&itxCm1299)
	km1189 := archsimd.LoadInt32x4Array(&itxCm1189)
	km1138 := archsimd.LoadInt32x4Array(&itxCm1138)
	km1061 := archsimd.LoadInt32x4Array(&itxCm1061)
	km601 := archsimd.LoadInt32x4Array(&itxCm601)
	km583 := archsimd.LoadInt32x4Array(&itxCm583)
	km484 := archsimd.LoadInt32x4Array(&itxCm484)
	km393 := archsimd.LoadInt32x4Array(&itxCm393)
	km312 := archsimd.LoadInt32x4Array(&itxCm312)
	km239 := archsimd.LoadInt32x4Array(&itxCm239)
	km176 := archsimd.LoadInt32x4Array(&itxCm176)
	km123 := archsimd.LoadInt32x4Array(&itxCm123)
	km79 := archsimd.LoadInt32x4Array(&itxCm79)
	km44 := archsimd.LoadInt32x4Array(&itxCm44)
	km20 := archsimd.LoadInt32x4Array(&itxCm20)
	km5 := archsimd.LoadInt32x4Array(&itxCm5)
	kp5 := archsimd.LoadInt32x4Array(&itxCp5)
	kp20 := archsimd.LoadInt32x4Array(&itxCp20)
	kp79 := archsimd.LoadInt32x4Array(&itxCp79)
	kp123 := archsimd.LoadInt32x4Array(&itxCp123)
	kp181 := archsimd.LoadInt32x4Array(&itxCp181)
	kp201 := archsimd.LoadInt32x4Array(&itxCp201)
	kp312 := archsimd.LoadInt32x4Array(&itxCp312)
	kp393 := archsimd.LoadInt32x4Array(&itxCp393)
	kp401 := archsimd.LoadInt32x4Array(&itxCp401)
	kp484 := archsimd.LoadInt32x4Array(&itxCp484)
	kp601 := archsimd.LoadInt32x4Array(&itxCp601)
	kp799 := archsimd.LoadInt32x4Array(&itxCp799)
	kp995 := archsimd.LoadInt32x4Array(&itxCp995)
	kp1138 := archsimd.LoadInt32x4Array(&itxCp1138)
	kp1189 := archsimd.LoadInt32x4Array(&itxCp1189)
	kp1220 := archsimd.LoadInt32x4Array(&itxCp1220)
	kp1299 := archsimd.LoadInt32x4Array(&itxCp1299)
	kp1380 := archsimd.LoadInt32x4Array(&itxCp1380)
	kp1567 := archsimd.LoadInt32x4Array(&itxCp1567)
	kp1583 := archsimd.LoadInt32x4Array(&itxCp1583)
	kp1645 := archsimd.LoadInt32x4Array(&itxCp1645)
	kp1703 := archsimd.LoadInt32x4Array(&itxCp1703)
	kp1751 := archsimd.LoadInt32x4Array(&itxCp1751)
	kp1931 := archsimd.LoadInt32x4Array(&itxCp1931)
	kp2106 := archsimd.LoadInt32x4Array(&itxCp2106)
	kp2751 := archsimd.LoadInt32x4Array(&itxCp2751)
	x0, x1, x2, x3 := transpose4x4(archsimd.LoadInt32x4Array((*[4]int32)(r0[0:])), archsimd.LoadInt32x4Array((*[4]int32)(r1[0:])), archsimd.LoadInt32x4Array((*[4]int32)(r2[0:])), archsimd.LoadInt32x4Array((*[4]int32)(r3[0:])))
	x4, x5, x6, x7 := transpose4x4(archsimd.LoadInt32x4Array((*[4]int32)(r0[4:])), archsimd.LoadInt32x4Array((*[4]int32)(r1[4:])), archsimd.LoadInt32x4Array((*[4]int32)(r2[4:])), archsimd.LoadInt32x4Array((*[4]int32)(r3[4:])))
	x8, x9, x10, x11 := transpose4x4(archsimd.LoadInt32x4Array((*[4]int32)(r0[8:])), archsimd.LoadInt32x4Array((*[4]int32)(r1[8:])), archsimd.LoadInt32x4Array((*[4]int32)(r2[8:])), archsimd.LoadInt32x4Array((*[4]int32)(r3[8:])))
	x12, x13, x14, x15 := transpose4x4(archsimd.LoadInt32x4Array((*[4]int32)(r0[12:])), archsimd.LoadInt32x4Array((*[4]int32)(r1[12:])), archsimd.LoadInt32x4Array((*[4]int32)(r2[12:])), archsimd.LoadInt32x4Array((*[4]int32)(r3[12:])))
	x16, x17, x18, x19 := transpose4x4(archsimd.LoadInt32x4Array((*[4]int32)(r0[16:])), archsimd.LoadInt32x4Array((*[4]int32)(r1[16:])), archsimd.LoadInt32x4Array((*[4]int32)(r2[16:])), archsimd.LoadInt32x4Array((*[4]int32)(r3[16:])))
	x20, x21, x22, x23 := transpose4x4(archsimd.LoadInt32x4Array((*[4]int32)(r0[20:])), archsimd.LoadInt32x4Array((*[4]int32)(r1[20:])), archsimd.LoadInt32x4Array((*[4]int32)(r2[20:])), archsimd.LoadInt32x4Array((*[4]int32)(r3[20:])))
	x24, x25, x26, x27 := transpose4x4(archsimd.LoadInt32x4Array((*[4]int32)(r0[24:])), archsimd.LoadInt32x4Array((*[4]int32)(r1[24:])), archsimd.LoadInt32x4Array((*[4]int32)(r2[24:])), archsimd.LoadInt32x4Array((*[4]int32)(r3[24:])))
	x28, x29, x30, x31 := transpose4x4(archsimd.LoadInt32x4Array((*[4]int32)(r0[28:])), archsimd.LoadInt32x4Array((*[4]int32)(r1[28:])), archsimd.LoadInt32x4Array((*[4]int32)(r2[28:])), archsimd.LoadInt32x4Array((*[4]int32)(r3[28:])))
	v1 := x0.Add(x16).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v2 := x0.Sub(x16).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v3 := x24.MulAdd(kp312, x8.Mul(kp1567)).Shift(itxShr(11)).Average(zero).Sub(x24)
	v4 := x24.MulAdd(kp1567, x8.Mul(km312)).Shift(itxShr(11)).Average(zero).Add(x8)
	v5 := v1.Add(v4).Max(minV).Min(maxV)
	v6 := v2.Add(v3).Max(minV).Min(maxV)
	v7 := v2.Sub(v3).Max(minV).Min(maxV)
	v8 := v1.Sub(v4).Max(minV).Min(maxV)
	v9 := x28.MulAdd(kp79, x4.Mul(kp799)).Shift(itxShr(11)).Average(zero).Sub(x28)
	v10 := x12.MulAdd(km1138, x20.Mul(kp1703)).Shift(itxShr(10)).Average(zero)
	v11 := x12.MulAdd(kp1703, x20.Mul(kp1138)).Shift(itxShr(10)).Average(zero)
	v12 := x28.MulAdd(kp799, x4.Mul(km79)).Shift(itxShr(11)).Average(zero).Add(x4)
	v13 := v9.Add(v10).Max(minV).Min(maxV)
	v14 := v9.Sub(v10).Max(minV).Min(maxV)
	v15 := v12.Add(v11).Max(minV).Min(maxV)
	v16 := v12.Sub(v11).Max(minV).Min(maxV)
	v17 := v16.Sub(v14).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v18 := v16.Add(v14).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v19 := v5.Add(v15).Max(minV).Min(maxV)
	v20 := v6.Add(v18).Max(minV).Min(maxV)
	v21 := v7.Add(v17).Max(minV).Min(maxV)
	v22 := v8.Add(v13).Max(minV).Min(maxV)
	v23 := v8.Sub(v13).Max(minV).Min(maxV)
	v24 := v7.Sub(v17).Max(minV).Min(maxV)
	v25 := v6.Sub(v18).Max(minV).Min(maxV)
	v26 := v5.Sub(v15).Max(minV).Min(maxV)
	v27 := x30.MulAdd(kp20, x2.Mul(kp401)).Shift(itxShr(11)).Average(zero).Sub(x30)
	v28 := x14.MulAdd(km1299, x18.Mul(kp1583)).Shift(itxShr(10)).Average(zero)
	v29 := x22.MulAdd(kp484, x10.Mul(kp1931)).Shift(itxShr(11)).Average(zero).Sub(x22)
	v30 := x6.MulAdd(km1189, x26.Mul(km176)).Shift(itxShr(11)).Average(zero).Add(x26)
	v31 := x6.MulAdd(km176, x26.Mul(kp1189)).Shift(itxShr(11)).Average(zero).Add(x6)
	v32 := x22.MulAdd(kp1931, x10.Mul(km484)).Shift(itxShr(11)).Average(zero).Add(x10)
	v33 := x14.MulAdd(kp1583, x18.Mul(kp1299)).Shift(itxShr(10)).Average(zero)
	v34 := x30.MulAdd(kp401, x2.Mul(km20)).Shift(itxShr(11)).Average(zero).Add(x2)
	v35 := v27.Add(v28).Max(minV).Min(maxV)
	v36 := v27.Sub(v28).Max(minV).Min(maxV)
	v37 := v30.Sub(v29).Max(minV).Min(maxV)
	v38 := v30.Add(v29).Max(minV).Min(maxV)
	v39 := v31.Add(v32).Max(minV).Min(maxV)
	v40 := v31.Sub(v32).Max(minV).Min(maxV)
	v41 := v34.Sub(v33).Max(minV).Min(maxV)
	v42 := v34.Add(v33).Max(minV).Min(maxV)
	v43 := v36.MulAdd(kp312, v41.Mul(kp1567)).Shift(itxShr(11)).Average(zero).Sub(v36)
	v44 := v36.MulAdd(kp1567, v41.Mul(km312)).Shift(itxShr(11)).Average(zero).Add(v41)
	v45 := zero.Sub(v37.MulAdd(kp1567, v40.Mul(km312))).Shift(itxShr(11)).Average(zero).Sub(v40)
	v46 := v37.MulAdd(kp312, v40.Mul(kp1567)).Shift(itxShr(11)).Average(zero).Sub(v37)
	v47 := v35.Add(v38).Max(minV).Min(maxV)
	v48 := v43.Add(v45).Max(minV).Min(maxV)
	v49 := v43.Sub(v45).Max(minV).Min(maxV)
	v50 := v35.Sub(v38).Max(minV).Min(maxV)
	v51 := v42.Sub(v39).Max(minV).Min(maxV)
	v52 := v44.Sub(v46).Max(minV).Min(maxV)
	v53 := v44.Add(v46).Max(minV).Min(maxV)
	v54 := v42.Add(v39).Max(minV).Min(maxV)
	v55 := v52.Sub(v49).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v56 := v52.Add(v49).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v57 := v51.Sub(v50).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v58 := v51.Add(v50).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v59 := v19.Add(v54).Max(minV).Min(maxV)
	v60 := v20.Add(v53).Max(minV).Min(maxV)
	v61 := v21.Add(v56).Max(minV).Min(maxV)
	v62 := v22.Add(v58).Max(minV).Min(maxV)
	v63 := v23.Add(v57).Max(minV).Min(maxV)
	v64 := v24.Add(v55).Max(minV).Min(maxV)
	v65 := v25.Add(v48).Max(minV).Min(maxV)
	v66 := v26.Add(v47).Max(minV).Min(maxV)
	v67 := v26.Sub(v47).Max(minV).Min(maxV)
	v68 := v25.Sub(v48).Max(minV).Min(maxV)
	v69 := v24.Sub(v55).Max(minV).Min(maxV)
	v70 := v23.Sub(v57).Max(minV).Min(maxV)
	v71 := v22.Sub(v58).Max(minV).Min(maxV)
	v72 := v21.Sub(v56).Max(minV).Min(maxV)
	v73 := v20.Sub(v53).Max(minV).Min(maxV)
	v74 := v19.Sub(v54).Max(minV).Min(maxV)
	v75 := x31.MulAdd(kp5, x1.Mul(kp201)).Shift(itxShr(11)).Average(zero).Sub(x31)
	v76 := x15.MulAdd(km2751, x17.Mul(km1061)).Shift(itxShr(11)).Average(zero).Add(x17)
	v77 := x23.MulAdd(kp393, x9.Mul(kp1751)).Shift(itxShr(11)).Average(zero).Sub(x23)
	v78 := x7.MulAdd(km1380, x25.Mul(km239)).Shift(itxShr(11)).Average(zero).Add(x25)
	v79 := x27.MulAdd(kp123, x5.Mul(kp995)).Shift(itxShr(11)).Average(zero).Sub(x27)
	v80 := x11.MulAdd(km2106, x21.Mul(km583)).Shift(itxShr(11)).Average(zero).Add(x21)
	v81 := x19.MulAdd(km1645, x13.Mul(kp1220)).Shift(itxShr(10)).Average(zero)
	v82 := x3.MulAdd(km601, x29.Mul(km44)).Shift(itxShr(11)).Average(zero).Add(x29)
	v83 := x3.MulAdd(km44, x29.Mul(kp601)).Shift(itxShr(11)).Average(zero).Add(x3)
	v84 := x19.MulAdd(kp1220, x13.Mul(kp1645)).Shift(itxShr(10)).Average(zero)
	v85 := x11.MulAdd(km583, x21.Mul(kp2106)).Shift(itxShr(11)).Average(zero).Add(x11)
	v86 := x27.MulAdd(kp995, x5.Mul(km123)).Shift(itxShr(11)).Average(zero).Add(x5)
	v87 := x7.MulAdd(km239, x25.Mul(kp1380)).Shift(itxShr(11)).Average(zero).Add(x7)
	v88 := x23.MulAdd(kp1751, x9.Mul(km393)).Shift(itxShr(11)).Average(zero).Add(x9)
	v89 := x15.MulAdd(km1061, x17.Mul(kp2751)).Shift(itxShr(11)).Average(zero).Add(x15)
	v90 := x31.MulAdd(kp201, x1.Mul(km5)).Shift(itxShr(11)).Average(zero).Add(x1)
	v91 := v75.Add(v76).Max(minV).Min(maxV)
	v92 := v75.Sub(v76).Max(minV).Min(maxV)
	v93 := v78.Sub(v77).Max(minV).Min(maxV)
	v94 := v78.Add(v77).Max(minV).Min(maxV)
	v95 := v79.Add(v80).Max(minV).Min(maxV)
	v96 := v79.Sub(v80).Max(minV).Min(maxV)
	v97 := v82.Sub(v81).Max(minV).Min(maxV)
	v98 := v82.Add(v81).Max(minV).Min(maxV)
	v99 := v83.Add(v84).Max(minV).Min(maxV)
	v100 := v83.Sub(v84).Max(minV).Min(maxV)
	v101 := v86.Sub(v85).Max(minV).Min(maxV)
	v102 := v86.Add(v85).Max(minV).Min(maxV)
	v103 := v87.Add(v88).Max(minV).Min(maxV)
	v104 := v87.Sub(v88).Max(minV).Min(maxV)
	v105 := v90.Sub(v89).Max(minV).Min(maxV)
	v106 := v90.Add(v89).Max(minV).Min(maxV)
	v107 := v92.MulAdd(kp79, v105.Mul(kp799)).Shift(itxShr(11)).Average(zero).Sub(v92)
	v108 := v92.MulAdd(kp799, v105.Mul(km79)).Shift(itxShr(11)).Average(zero).Add(v105)
	v109 := zero.Sub(v93.MulAdd(kp799, v104.Mul(km79))).Shift(itxShr(11)).Average(zero).Sub(v104)
	v110 := v93.MulAdd(kp79, v104.Mul(kp799)).Shift(itxShr(11)).Average(zero).Sub(v93)
	v111 := v96.MulAdd(km1138, v101.Mul(kp1703)).Shift(itxShr(10)).Average(zero)
	v112 := v96.MulAdd(kp1703, v101.Mul(kp1138)).Shift(itxShr(10)).Average(zero)
	v113 := zero.Sub(v97.MulAdd(kp1703, v100.Mul(kp1138))).Shift(itxShr(10)).Average(zero)
	v114 := v97.MulAdd(km1138, v100.Mul(kp1703)).Shift(itxShr(10)).Average(zero)
	v115 := v91.Add(v94).Max(minV).Min(maxV)
	v116 := v107.Add(v109).Max(minV).Min(maxV)
	v117 := v107.Sub(v109).Max(minV).Min(maxV)
	v118 := v91.Sub(v94).Max(minV).Min(maxV)
	v119 := v98.Sub(v95).Max(minV).Min(maxV)
	v120 := v113.Sub(v111).Max(minV).Min(maxV)
	v121 := v113.Add(v111).Max(minV).Min(maxV)
	v122 := v98.Add(v95).Max(minV).Min(maxV)
	v123 := v99.Add(v102).Max(minV).Min(maxV)
	v124 := v114.Add(v112).Max(minV).Min(maxV)
	v125 := v114.Sub(v112).Max(minV).Min(maxV)
	v126 := v99.Sub(v102).Max(minV).Min(maxV)
	v127 := v106.Sub(v103).Max(minV).Min(maxV)
	v128 := v108.Sub(v110).Max(minV).Min(maxV)
	v129 := v108.Add(v110).Max(minV).Min(maxV)
	v130 := v106.Add(v103).Max(minV).Min(maxV)
	v131 := v117.MulAdd(kp312, v128.Mul(kp1567)).Shift(itxShr(11)).Average(zero).Sub(v117)
	v132 := v117.MulAdd(kp1567, v128.Mul(km312)).Shift(itxShr(11)).Average(zero).Add(v128)
	v133 := v118.MulAdd(kp312, v127.Mul(kp1567)).Shift(itxShr(11)).Average(zero).Sub(v118)
	v134 := v118.MulAdd(kp1567, v127.Mul(km312)).Shift(itxShr(11)).Average(zero).Add(v127)
	v135 := zero.Sub(v119.MulAdd(kp1567, v126.Mul(km312))).Shift(itxShr(11)).Average(zero).Sub(v126)
	v136 := v119.MulAdd(kp312, v126.Mul(kp1567)).Shift(itxShr(11)).Average(zero).Sub(v119)
	v137 := zero.Sub(v120.MulAdd(kp1567, v125.Mul(km312))).Shift(itxShr(11)).Average(zero).Sub(v125)
	v138 := v120.MulAdd(kp312, v125.Mul(kp1567)).Shift(itxShr(11)).Average(zero).Sub(v120)
	v139 := v115.Add(v122).Max(minV).Min(maxV)
	v140 := v116.Add(v121).Max(minV).Min(maxV)
	v141 := v131.Add(v137).Max(minV).Min(maxV)
	v142 := v133.Add(v135).Max(minV).Min(maxV)
	v143 := v133.Sub(v135).Max(minV).Min(maxV)
	v144 := v131.Sub(v137).Max(minV).Min(maxV)
	v145 := v116.Sub(v121).Max(minV).Min(maxV)
	v146 := v115.Sub(v122).Max(minV).Min(maxV)
	v147 := v130.Sub(v123).Max(minV).Min(maxV)
	v148 := v129.Sub(v124).Max(minV).Min(maxV)
	v149 := v132.Sub(v138).Max(minV).Min(maxV)
	v150 := v134.Sub(v136).Max(minV).Min(maxV)
	v151 := v134.Add(v136).Max(minV).Min(maxV)
	v152 := v132.Add(v138).Max(minV).Min(maxV)
	v153 := v129.Add(v124).Max(minV).Min(maxV)
	v154 := v130.Add(v123).Max(minV).Min(maxV)
	v155 := v150.Sub(v143).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v156 := v150.Add(v143).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v157 := v149.Sub(v144).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v158 := v149.Add(v144).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v159 := v148.Sub(v145).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v160 := v148.Add(v145).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v161 := v147.Sub(v146).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v162 := v147.Add(v146).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v163 := v59.Add(v154).Max(minV).Min(maxV)
	v164 := v60.Add(v153).Max(minV).Min(maxV)
	v165 := v61.Add(v152).Max(minV).Min(maxV)
	v166 := v62.Add(v151).Max(minV).Min(maxV)
	v167 := v63.Add(v156).Max(minV).Min(maxV)
	v168 := v64.Add(v158).Max(minV).Min(maxV)
	v169 := v65.Add(v160).Max(minV).Min(maxV)
	v170 := v66.Add(v162).Max(minV).Min(maxV)
	v171 := v67.Add(v161).Max(minV).Min(maxV)
	v172 := v68.Add(v159).Max(minV).Min(maxV)
	v173 := v69.Add(v157).Max(minV).Min(maxV)
	v174 := v70.Add(v155).Max(minV).Min(maxV)
	v175 := v71.Add(v142).Max(minV).Min(maxV)
	v176 := v72.Add(v141).Max(minV).Min(maxV)
	v177 := v73.Add(v140).Max(minV).Min(maxV)
	v178 := v74.Add(v139).Max(minV).Min(maxV)
	v179 := v74.Sub(v139).Max(minV).Min(maxV)
	v180 := v73.Sub(v140).Max(minV).Min(maxV)
	v181 := v72.Sub(v141).Max(minV).Min(maxV)
	v182 := v71.Sub(v142).Max(minV).Min(maxV)
	v183 := v70.Sub(v155).Max(minV).Min(maxV)
	v184 := v69.Sub(v157).Max(minV).Min(maxV)
	v185 := v68.Sub(v159).Max(minV).Min(maxV)
	v186 := v67.Sub(v161).Max(minV).Min(maxV)
	v187 := v66.Sub(v162).Max(minV).Min(maxV)
	v188 := v65.Sub(v160).Max(minV).Min(maxV)
	v189 := v64.Sub(v158).Max(minV).Min(maxV)
	v190 := v63.Sub(v156).Max(minV).Min(maxV)
	v191 := v62.Sub(v151).Max(minV).Min(maxV)
	v192 := v61.Sub(v152).Max(minV).Min(maxV)
	v193 := v60.Sub(v153).Max(minV).Min(maxV)
	v194 := v59.Sub(v154).Max(minV).Min(maxV)
	y0_0, y0_1, y0_2, y0_3 := transpose4x4(v163, v164, v165, v166)
	y0_0.StoreArray((*[4]int32)(r0[0:]))
	y0_1.StoreArray((*[4]int32)(r1[0:]))
	y0_2.StoreArray((*[4]int32)(r2[0:]))
	y0_3.StoreArray((*[4]int32)(r3[0:]))
	y1_0, y1_1, y1_2, y1_3 := transpose4x4(v167, v168, v169, v170)
	y1_0.StoreArray((*[4]int32)(r0[4:]))
	y1_1.StoreArray((*[4]int32)(r1[4:]))
	y1_2.StoreArray((*[4]int32)(r2[4:]))
	y1_3.StoreArray((*[4]int32)(r3[4:]))
	y2_0, y2_1, y2_2, y2_3 := transpose4x4(v171, v172, v173, v174)
	y2_0.StoreArray((*[4]int32)(r0[8:]))
	y2_1.StoreArray((*[4]int32)(r1[8:]))
	y2_2.StoreArray((*[4]int32)(r2[8:]))
	y2_3.StoreArray((*[4]int32)(r3[8:]))
	y3_0, y3_1, y3_2, y3_3 := transpose4x4(v175, v176, v177, v178)
	y3_0.StoreArray((*[4]int32)(r0[12:]))
	y3_1.StoreArray((*[4]int32)(r1[12:]))
	y3_2.StoreArray((*[4]int32)(r2[12:]))
	y3_3.StoreArray((*[4]int32)(r3[12:]))
	y4_0, y4_1, y4_2, y4_3 := transpose4x4(v179, v180, v181, v182)
	y4_0.StoreArray((*[4]int32)(r0[16:]))
	y4_1.StoreArray((*[4]int32)(r1[16:]))
	y4_2.StoreArray((*[4]int32)(r2[16:]))
	y4_3.StoreArray((*[4]int32)(r3[16:]))
	y5_0, y5_1, y5_2, y5_3 := transpose4x4(v183, v184, v185, v186)
	y5_0.StoreArray((*[4]int32)(r0[20:]))
	y5_1.StoreArray((*[4]int32)(r1[20:]))
	y5_2.StoreArray((*[4]int32)(r2[20:]))
	y5_3.StoreArray((*[4]int32)(r3[20:]))
	y6_0, y6_1, y6_2, y6_3 := transpose4x4(v187, v188, v189, v190)
	y6_0.StoreArray((*[4]int32)(r0[24:]))
	y6_1.StoreArray((*[4]int32)(r1[24:]))
	y6_2.StoreArray((*[4]int32)(r2[24:]))
	y6_3.StoreArray((*[4]int32)(r3[24:]))
	y7_0, y7_1, y7_2, y7_3 := transpose4x4(v191, v192, v193, v194)
	y7_0.StoreArray((*[4]int32)(r0[28:]))
	y7_1.StoreArray((*[4]int32)(r1[28:]))
	y7_2.StoreArray((*[4]int32)(r2[28:]))
	y7_3.StoreArray((*[4]int32)(r3[28:]))
}

// inverseADST16Row4SIMD applies inverseADST16 to 4 rows held as lanes of transposed vectors.
func inverseADST16Row4SIMD(r0, r1, r2, r3 []int32, min, max int32) {
	minV := archsimd.BroadcastInt32x4(min)
	maxV := archsimd.BroadcastInt32x4(max)
	var zero archsimd.Int32x4
	km2751 := archsimd.LoadInt32x4Array(&itxCm2751)
	km2276 := archsimd.LoadInt32x4Array(&itxCm2276)
	km2106 := archsimd.LoadInt32x4Array(&itxCm2106)
	km1645 := archsimd.LoadInt32x4Array(&itxCm1645)
	km1567 := archsimd.LoadInt32x4Array(&itxCm1567)
	km1380 := archsimd.LoadInt32x4Array(&itxCm1380)
	km1061 := archsimd.LoadInt32x4Array(&itxCm1061)
	km799 := archsimd.LoadInt32x4Array(&itxCm799)
	km690 := archsimd.LoadInt32x4Array(&itxCm690)
	km601 := archsimd.LoadInt32x4Array(&itxCm601)
	km583 := archsimd.LoadInt32x4Array(&itxCm583)
	km393 := archsimd.LoadInt32x4Array(&itxCm393)
	km312 := archsimd.LoadInt32x4Array(&itxCm312)
	km239 := archsimd.LoadInt32x4Array(&itxCm239)
	km123 := archsimd.LoadInt32x4Array(&itxCm123)
	km79 := archsimd.LoadInt32x4Array(&itxCm79)
	km44 := archsimd.LoadInt32x4Array(&itxCm44)
	km5 := archsimd.LoadInt32x4Array(&itxCm5)
	kp5 := archsimd.LoadInt32x4Array(&itxCp5)
	kp79 := archsimd.LoadInt32x4Array(&itxCp79)
	kp123 := archsimd.LoadInt32x4Array(&itxCp123)
	kp181 := archsimd.LoadInt32x4Array(&itxCp181)
	kp201 := archsimd.LoadInt32x4Array(&itxCp201)
	kp312 := archsimd.LoadInt32x4Array(&itxCp312)
	kp393 := archsimd.LoadInt32x4Array(&itxCp393)
	kp601 := archsimd.LoadInt32x4Array(&itxCp601)
	kp690 := archsimd.LoadInt32x4Array(&itxCp690)
	kp799 := archsimd.LoadInt32x4Array(&itxCp799)
	kp995 := archsimd.LoadInt32x4Array(&itxCp995)
	kp1220 := archsimd.LoadInt32x4Array(&itxCp1220)
	kp1380 := archsimd.LoadInt32x4Array(&itxCp1380)
	kp1567 := archsimd.LoadInt32x4Array(&itxCp1567)
	kp1645 := archsimd.LoadInt32x4Array(&itxCp1645)
	kp1751 := archsimd.LoadInt32x4Array(&itxCp1751)
	kp2106 := archsimd.LoadInt32x4Array(&itxCp2106)
	kp2276 := archsimd.LoadInt32x4Array(&itxCp2276)
	kp2751 := archsimd.LoadInt32x4Array(&itxCp2751)
	x0, x1, x2, x3 := transpose4x4(archsimd.LoadInt32x4Array((*[4]int32)(r0[0:])), archsimd.LoadInt32x4Array((*[4]int32)(r1[0:])), archsimd.LoadInt32x4Array((*[4]int32)(r2[0:])), archsimd.LoadInt32x4Array((*[4]int32)(r3[0:])))
	x4, x5, x6, x7 := transpose4x4(archsimd.LoadInt32x4Array((*[4]int32)(r0[4:])), archsimd.LoadInt32x4Array((*[4]int32)(r1[4:])), archsimd.LoadInt32x4Array((*[4]int32)(r2[4:])), archsimd.LoadInt32x4Array((*[4]int32)(r3[4:])))
	x8, x9, x10, x11 := transpose4x4(archsimd.LoadInt32x4Array((*[4]int32)(r0[8:])), archsimd.LoadInt32x4Array((*[4]int32)(r1[8:])), archsimd.LoadInt32x4Array((*[4]int32)(r2[8:])), archsimd.LoadInt32x4Array((*[4]int32)(r3[8:])))
	x12, x13, x14, x15 := transpose4x4(archsimd.LoadInt32x4Array((*[4]int32)(r0[12:])), archsimd.LoadInt32x4Array((*[4]int32)(r1[12:])), archsimd.LoadInt32x4Array((*[4]int32)(r2[12:])), archsimd.LoadInt32x4Array((*[4]int32)(r3[12:])))
	v1 := x0.MulAdd(kp201, x15.Mul(km5)).Shift(itxShr(11)).Average(zero).Add(x15)
	v2 := x0.MulAdd(kp5, x15.Mul(kp201)).Shift(itxShr(11)).Average(zero).Sub(x0)
	v3 := x2.MulAdd(kp995, x13.Mul(km123)).Shift(itxShr(11)).Average(zero).Add(x13)
	v4 := x2.MulAdd(kp123, x13.Mul(kp995)).Shift(itxShr(11)).Average(zero).Sub(x2)
	v5 := x4.MulAdd(kp1751, x11.Mul(km393)).Shift(itxShr(11)).Average(zero).Add(x11)
	v6 := x4.MulAdd(kp393, x11.Mul(kp1751)).Shift(itxShr(11)).Average(zero).Sub(x4)
	v7 := x6.MulAdd(kp1220, x9.Mul(kp1645)).Shift(itxShr(10)).Average(zero)
	v8 := x6.MulAdd(km1645, x9.Mul(kp1220)).Shift(itxShr(10)).Average(zero)
	v9 := x8.MulAdd(km1061, x7.Mul(kp2751)).Shift(itxShr(11)).Average(zero).Add(x8)
	v10 := x8.MulAdd(km2751, x7.Mul(km1061)).Shift(itxShr(11)).Average(zero).Add(x7)
	v11 := x10.MulAdd(km583, x5.Mul(kp2106)).Shift(itxShr(11)).Average(zero).Add(x10)
	v12 := x10.MulAdd(km2106, x5.Mul(km583)).Shift(itxShr(11)).Average(zero).Add(x5)
	v13 := x12.MulAdd(km239, x3.Mul(kp1380)).Shift(itxShr(11)).Average(zero).Add(x12)
	v14 := x12.MulAdd(km1380, x3.Mul(km239)).Shift(itxShr(11)).Average(zero).Add(x3)
	v15 := x14.MulAdd(km44, x1.Mul(kp601)).Shift(itxShr(11)).Average(zero).Add(x14)
	v16 := x14.MulAdd(km601, x1.Mul(km44)).Shift(itxShr(11)).Average(zero).Add(x1)
	v17 := v1.Add(v9).Max(minV).Min(maxV)
	v18 := v2.Add(v10).Max(minV).Min(maxV)
	v19 := v3.Add(v11).Max(minV).Min(maxV)
	v20 := v4.Add(v12).Max(minV).Min(maxV)
	v21 := v5.Add(v13).Max(minV).Min(maxV)
	v22 := v6.Add(v14).Max(minV).Min(maxV)
	v23 := v7.Add(v15).Max(minV).Min(maxV)
	v24 := v8.Add(v16).Max(minV).Min(maxV)
	v25 := v1.Sub(v9).Max(minV).Min(maxV)
	v26 := v2.Sub(v10).Max(minV).Min(maxV)
	v27 := v3.Sub(v11).Max(minV).Min(maxV)
	v28 := v4.Sub(v12).Max(minV).Min(maxV)
	v29 := v5.Sub(v13).Max(minV).Min(maxV)
	v30 := v6.Sub(v14).Max(minV).Min(maxV)
	v31 := v7.Sub(v15).Max(minV).Min(maxV)
	v32 := v8.Sub(v16).Max(minV).Min(maxV)
	v33 := v26.MulAdd(kp799, v25.Mul(km79)).Shift(itxShr(11)).Average(zero).Add(v25)
	v34 := v26.MulAdd(kp79, v25.Mul(kp799)).Shift(itxShr(11)).Average(zero).Sub(v26)
	v35 := v28.MulAdd(km690, v27.Mul(kp2276)).Shift(itxShr(11)).Average(zero).Add(v28)
	v36 := v28.MulAdd(km2276, v27.Mul(km690)).Shift(itxShr(11)).Average(zero).Add(v27)
	v37 := v29.MulAdd(km799, v30.Mul(km79)).Shift(itxShr(11)).Average(zero).Add(v30)
	v38 := v29.MulAdd(km79, v30.Mul(kp799)).Shift(itxShr(11)).Average(zero).Add(v29)
	v39 := v31.MulAdd(kp690, v32.Mul(kp2276)).Shift(itxShr(11)).Average(zero).Sub(v31)
	v40 := v31.MulAdd(kp2276, v32.Mul(km690)).Shift(itxShr(11)).Average(zero).Add(v32)
	v41 := v17.Add(v21).Max(minV).Min(maxV)
	v42 := v18.Add(v22).Max(minV).Min(maxV)
	v43 := v19.Add(v23).Max(minV).Min(maxV)
	v44 := v20.Add(v24).Max(minV).Min(maxV)
	v45 := v17.Sub(v21).Max(minV).Min(maxV)
	v46 := v18.Sub(v22).Max(minV).Min(maxV)
	v47 := v19.Sub(v23).Max(minV).Min(maxV)
	v48 := v20.Sub(v24).Max(minV).Min(maxV)
	v49 := v33.Add(v37).Max(minV).Min(maxV)
	v50 := v34.Add(v38).Max(minV).Min(maxV)
	v51 := v35.Add(v39).Max(minV).Min(maxV)
	v52 := v36.Add(v40).Max(minV).Min(maxV)
	v53 := v33.Sub(v37).Max(minV).Min(maxV)
	v54 := v34.Sub(v38).Max(minV).Min(maxV)
	v55 := v35.Sub(v39).Max(minV).Min(maxV)
	v56 := v36.Sub(v40).Max(minV).Min(maxV)
	v57 := v46.MulAdd(kp1567, v45.Mul(km312)).Shift(itxShr(11)).Average(zero).Add(v45)
	v58 := v46.MulAdd(kp312, v45.Mul(kp1567)).Shift(itxShr(11)).Average(zero).Sub(v46)
	v59 := v47.MulAdd(km1567, v48.Mul(km312)).Shift(itxShr(11)).Average(zero).Add(v48)
	v60 := v47.MulAdd(km312, v48.Mul(kp1567)).Shift(itxShr(11)).Average(zero).Add(v47)
	v61 := v54.MulAdd(kp1567, v53.Mul(km312)).Shift(itxShr(11)).Average(zero).Add(v53)
	v62 := v54.MulAdd(kp312, v53.Mul(kp1567)).Shift(itxShr(11)).Average(zero).Sub(v54)
	v63 := v55.MulAdd(km1567, v56.Mul(km312)).Shift(itxShr(11)).Average(zero).Add(v56)
	v64 := v55.MulAdd(km312, v56.Mul(kp1567)).Shift(itxShr(11)).Average(zero).Add(v55)
	v65 := v41.Add(v43).Max(minV).Min(maxV)
	v66 := zero.Sub(v42.Add(v44).Max(minV).Min(maxV))
	v67 := v41.Sub(v43).Max(minV).Min(maxV)
	v68 := v42.Sub(v44).Max(minV).Min(maxV)
	v69 := zero.Sub(v57.Add(v59).Max(minV).Min(maxV))
	v70 := v58.Add(v60).Max(minV).Min(maxV)
	v71 := v57.Sub(v59).Max(minV).Min(maxV)
	v72 := v58.Sub(v60).Max(minV).Min(maxV)
	v73 := zero.Sub(v49.Add(v51).Max(minV).Min(maxV))
	v74 := v50.Add(v52).Max(minV).Min(maxV)
	v75 := v49.Sub(v51).Max(minV).Min(maxV)
	v76 := v50.Sub(v52).Max(minV).Min(maxV)
	v77 := v61.Add(v63).Max(minV).Min(maxV)
	v78 := zero.Sub(v62.Add(v64).Max(minV).Min(maxV))
	v79 := v61.Sub(v63).Max(minV).Min(maxV)
	v80 := v62.Sub(v64).Max(minV).Min(maxV)
	v81 := zero.Sub(v67.Add(v68).Mul(kp181).Shift(itxShr(7)).Average(zero))
	v82 := v67.Sub(v68).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v83 := v71.Add(v72).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v84 := zero.Sub(v71.Sub(v72).Mul(kp181).Shift(itxShr(7)).Average(zero))
	v85 := v75.Add(v76).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v86 := zero.Sub(v75.Sub(v76).Mul(kp181).Shift(itxShr(7)).Average(zero))
	v87 := zero.Sub(v79.Add(v80).Mul(kp181).Shift(itxShr(7)).Average(zero))
	v88 := v79.Sub(v80).Mul(kp181).Shift(itxShr(7)).Average(zero)
	y0_0, y0_1, y0_2, y0_3 := transpose4x4(v65, v73, v77, v69)
	y0_0.StoreArray((*[4]int32)(r0[0:]))
	y0_1.StoreArray((*[4]int32)(r1[0:]))
	y0_2.StoreArray((*[4]int32)(r2[0:]))
	y0_3.StoreArray((*[4]int32)(r3[0:]))
	y1_0, y1_1, y1_2, y1_3 := transpose4x4(v83, v87, v85, v81)
	y1_0.StoreArray((*[4]int32)(r0[4:]))
	y1_1.StoreArray((*[4]int32)(r1[4:]))
	y1_2.StoreArray((*[4]int32)(r2[4:]))
	y1_3.StoreArray((*[4]int32)(r3[4:]))
	y2_0, y2_1, y2_2, y2_3 := transpose4x4(v82, v86, v88, v84)
	y2_0.StoreArray((*[4]int32)(r0[8:]))
	y2_1.StoreArray((*[4]int32)(r1[8:]))
	y2_2.StoreArray((*[4]int32)(r2[8:]))
	y2_3.StoreArray((*[4]int32)(r3[8:]))
	y3_0, y3_1, y3_2, y3_3 := transpose4x4(v70, v78, v74, v66)
	y3_0.StoreArray((*[4]int32)(r0[12:]))
	y3_1.StoreArray((*[4]int32)(r1[12:]))
	y3_2.StoreArray((*[4]int32)(r2[12:]))
	y3_3.StoreArray((*[4]int32)(r3[12:]))
}

// inverseADST16Row4FlipSIMD applies inverseADST16 to 4 rows held as lanes of transposed vectors.
func inverseADST16Row4FlipSIMD(r0, r1, r2, r3 []int32, min, max int32) {
	minV := archsimd.BroadcastInt32x4(min)
	maxV := archsimd.BroadcastInt32x4(max)
	var zero archsimd.Int32x4
	km2751 := archsimd.LoadInt32x4Array(&itxCm2751)
	km2276 := archsimd.LoadInt32x4Array(&itxCm2276)
	km2106 := archsimd.LoadInt32x4Array(&itxCm2106)
	km1645 := archsimd.LoadInt32x4Array(&itxCm1645)
	km1567 := archsimd.LoadInt32x4Array(&itxCm1567)
	km1380 := archsimd.LoadInt32x4Array(&itxCm1380)
	km1061 := archsimd.LoadInt32x4Array(&itxCm1061)
	km799 := archsimd.LoadInt32x4Array(&itxCm799)
	km690 := archsimd.LoadInt32x4Array(&itxCm690)
	km601 := archsimd.LoadInt32x4Array(&itxCm601)
	km583 := archsimd.LoadInt32x4Array(&itxCm583)
	km393 := archsimd.LoadInt32x4Array(&itxCm393)
	km312 := archsimd.LoadInt32x4Array(&itxCm312)
	km239 := archsimd.LoadInt32x4Array(&itxCm239)
	km123 := archsimd.LoadInt32x4Array(&itxCm123)
	km79 := archsimd.LoadInt32x4Array(&itxCm79)
	km44 := archsimd.LoadInt32x4Array(&itxCm44)
	km5 := archsimd.LoadInt32x4Array(&itxCm5)
	kp5 := archsimd.LoadInt32x4Array(&itxCp5)
	kp79 := archsimd.LoadInt32x4Array(&itxCp79)
	kp123 := archsimd.LoadInt32x4Array(&itxCp123)
	kp181 := archsimd.LoadInt32x4Array(&itxCp181)
	kp201 := archsimd.LoadInt32x4Array(&itxCp201)
	kp312 := archsimd.LoadInt32x4Array(&itxCp312)
	kp393 := archsimd.LoadInt32x4Array(&itxCp393)
	kp601 := archsimd.LoadInt32x4Array(&itxCp601)
	kp690 := archsimd.LoadInt32x4Array(&itxCp690)
	kp799 := archsimd.LoadInt32x4Array(&itxCp799)
	kp995 := archsimd.LoadInt32x4Array(&itxCp995)
	kp1220 := archsimd.LoadInt32x4Array(&itxCp1220)
	kp1380 := archsimd.LoadInt32x4Array(&itxCp1380)
	kp1567 := archsimd.LoadInt32x4Array(&itxCp1567)
	kp1645 := archsimd.LoadInt32x4Array(&itxCp1645)
	kp1751 := archsimd.LoadInt32x4Array(&itxCp1751)
	kp2106 := archsimd.LoadInt32x4Array(&itxCp2106)
	kp2276 := archsimd.LoadInt32x4Array(&itxCp2276)
	kp2751 := archsimd.LoadInt32x4Array(&itxCp2751)
	x0, x1, x2, x3 := transpose4x4(archsimd.LoadInt32x4Array((*[4]int32)(r0[0:])), archsimd.LoadInt32x4Array((*[4]int32)(r1[0:])), archsimd.LoadInt32x4Array((*[4]int32)(r2[0:])), archsimd.LoadInt32x4Array((*[4]int32)(r3[0:])))
	x4, x5, x6, x7 := transpose4x4(archsimd.LoadInt32x4Array((*[4]int32)(r0[4:])), archsimd.LoadInt32x4Array((*[4]int32)(r1[4:])), archsimd.LoadInt32x4Array((*[4]int32)(r2[4:])), archsimd.LoadInt32x4Array((*[4]int32)(r3[4:])))
	x8, x9, x10, x11 := transpose4x4(archsimd.LoadInt32x4Array((*[4]int32)(r0[8:])), archsimd.LoadInt32x4Array((*[4]int32)(r1[8:])), archsimd.LoadInt32x4Array((*[4]int32)(r2[8:])), archsimd.LoadInt32x4Array((*[4]int32)(r3[8:])))
	x12, x13, x14, x15 := transpose4x4(archsimd.LoadInt32x4Array((*[4]int32)(r0[12:])), archsimd.LoadInt32x4Array((*[4]int32)(r1[12:])), archsimd.LoadInt32x4Array((*[4]int32)(r2[12:])), archsimd.LoadInt32x4Array((*[4]int32)(r3[12:])))
	v1 := x0.MulAdd(kp201, x15.Mul(km5)).Shift(itxShr(11)).Average(zero).Add(x15)
	v2 := x0.MulAdd(kp5, x15.Mul(kp201)).Shift(itxShr(11)).Average(zero).Sub(x0)
	v3 := x2.MulAdd(kp995, x13.Mul(km123)).Shift(itxShr(11)).Average(zero).Add(x13)
	v4 := x2.MulAdd(kp123, x13.Mul(kp995)).Shift(itxShr(11)).Average(zero).Sub(x2)
	v5 := x4.MulAdd(kp1751, x11.Mul(km393)).Shift(itxShr(11)).Average(zero).Add(x11)
	v6 := x4.MulAdd(kp393, x11.Mul(kp1751)).Shift(itxShr(11)).Average(zero).Sub(x4)
	v7 := x6.MulAdd(kp1220, x9.Mul(kp1645)).Shift(itxShr(10)).Average(zero)
	v8 := x6.MulAdd(km1645, x9.Mul(kp1220)).Shift(itxShr(10)).Average(zero)
	v9 := x8.MulAdd(km1061, x7.Mul(kp2751)).Shift(itxShr(11)).Average(zero).Add(x8)
	v10 := x8.MulAdd(km2751, x7.Mul(km1061)).Shift(itxShr(11)).Average(zero).Add(x7)
	v11 := x10.MulAdd(km583, x5.Mul(kp2106)).Shift(itxShr(11)).Average(zero).Add(x10)
	v12 := x10.MulAdd(km2106, x5.Mul(km583)).Shift(itxShr(11)).Average(zero).Add(x5)
	v13 := x12.MulAdd(km239, x3.Mul(kp1380)).Shift(itxShr(11)).Average(zero).Add(x12)
	v14 := x12.MulAdd(km1380, x3.Mul(km239)).Shift(itxShr(11)).Average(zero).Add(x3)
	v15 := x14.MulAdd(km44, x1.Mul(kp601)).Shift(itxShr(11)).Average(zero).Add(x14)
	v16 := x14.MulAdd(km601, x1.Mul(km44)).Shift(itxShr(11)).Average(zero).Add(x1)
	v17 := v1.Add(v9).Max(minV).Min(maxV)
	v18 := v2.Add(v10).Max(minV).Min(maxV)
	v19 := v3.Add(v11).Max(minV).Min(maxV)
	v20 := v4.Add(v12).Max(minV).Min(maxV)
	v21 := v5.Add(v13).Max(minV).Min(maxV)
	v22 := v6.Add(v14).Max(minV).Min(maxV)
	v23 := v7.Add(v15).Max(minV).Min(maxV)
	v24 := v8.Add(v16).Max(minV).Min(maxV)
	v25 := v1.Sub(v9).Max(minV).Min(maxV)
	v26 := v2.Sub(v10).Max(minV).Min(maxV)
	v27 := v3.Sub(v11).Max(minV).Min(maxV)
	v28 := v4.Sub(v12).Max(minV).Min(maxV)
	v29 := v5.Sub(v13).Max(minV).Min(maxV)
	v30 := v6.Sub(v14).Max(minV).Min(maxV)
	v31 := v7.Sub(v15).Max(minV).Min(maxV)
	v32 := v8.Sub(v16).Max(minV).Min(maxV)
	v33 := v26.MulAdd(kp799, v25.Mul(km79)).Shift(itxShr(11)).Average(zero).Add(v25)
	v34 := v26.MulAdd(kp79, v25.Mul(kp799)).Shift(itxShr(11)).Average(zero).Sub(v26)
	v35 := v28.MulAdd(km690, v27.Mul(kp2276)).Shift(itxShr(11)).Average(zero).Add(v28)
	v36 := v28.MulAdd(km2276, v27.Mul(km690)).Shift(itxShr(11)).Average(zero).Add(v27)
	v37 := v29.MulAdd(km799, v30.Mul(km79)).Shift(itxShr(11)).Average(zero).Add(v30)
	v38 := v29.MulAdd(km79, v30.Mul(kp799)).Shift(itxShr(11)).Average(zero).Add(v29)
	v39 := v31.MulAdd(kp690, v32.Mul(kp2276)).Shift(itxShr(11)).Average(zero).Sub(v31)
	v40 := v31.MulAdd(kp2276, v32.Mul(km690)).Shift(itxShr(11)).Average(zero).Add(v32)
	v41 := v17.Add(v21).Max(minV).Min(maxV)
	v42 := v18.Add(v22).Max(minV).Min(maxV)
	v43 := v19.Add(v23).Max(minV).Min(maxV)
	v44 := v20.Add(v24).Max(minV).Min(maxV)
	v45 := v17.Sub(v21).Max(minV).Min(maxV)
	v46 := v18.Sub(v22).Max(minV).Min(maxV)
	v47 := v19.Sub(v23).Max(minV).Min(maxV)
	v48 := v20.Sub(v24).Max(minV).Min(maxV)
	v49 := v33.Add(v37).Max(minV).Min(maxV)
	v50 := v34.Add(v38).Max(minV).Min(maxV)
	v51 := v35.Add(v39).Max(minV).Min(maxV)
	v52 := v36.Add(v40).Max(minV).Min(maxV)
	v53 := v33.Sub(v37).Max(minV).Min(maxV)
	v54 := v34.Sub(v38).Max(minV).Min(maxV)
	v55 := v35.Sub(v39).Max(minV).Min(maxV)
	v56 := v36.Sub(v40).Max(minV).Min(maxV)
	v57 := v46.MulAdd(kp1567, v45.Mul(km312)).Shift(itxShr(11)).Average(zero).Add(v45)
	v58 := v46.MulAdd(kp312, v45.Mul(kp1567)).Shift(itxShr(11)).Average(zero).Sub(v46)
	v59 := v47.MulAdd(km1567, v48.Mul(km312)).Shift(itxShr(11)).Average(zero).Add(v48)
	v60 := v47.MulAdd(km312, v48.Mul(kp1567)).Shift(itxShr(11)).Average(zero).Add(v47)
	v61 := v54.MulAdd(kp1567, v53.Mul(km312)).Shift(itxShr(11)).Average(zero).Add(v53)
	v62 := v54.MulAdd(kp312, v53.Mul(kp1567)).Shift(itxShr(11)).Average(zero).Sub(v54)
	v63 := v55.MulAdd(km1567, v56.Mul(km312)).Shift(itxShr(11)).Average(zero).Add(v56)
	v64 := v55.MulAdd(km312, v56.Mul(kp1567)).Shift(itxShr(11)).Average(zero).Add(v55)
	v65 := v41.Add(v43).Max(minV).Min(maxV)
	v66 := zero.Sub(v42.Add(v44).Max(minV).Min(maxV))
	v67 := v41.Sub(v43).Max(minV).Min(maxV)
	v68 := v42.Sub(v44).Max(minV).Min(maxV)
	v69 := zero.Sub(v57.Add(v59).Max(minV).Min(maxV))
	v70 := v58.Add(v60).Max(minV).Min(maxV)
	v71 := v57.Sub(v59).Max(minV).Min(maxV)
	v72 := v58.Sub(v60).Max(minV).Min(maxV)
	v73 := zero.Sub(v49.Add(v51).Max(minV).Min(maxV))
	v74 := v50.Add(v52).Max(minV).Min(maxV)
	v75 := v49.Sub(v51).Max(minV).Min(maxV)
	v76 := v50.Sub(v52).Max(minV).Min(maxV)
	v77 := v61.Add(v63).Max(minV).Min(maxV)
	v78 := zero.Sub(v62.Add(v64).Max(minV).Min(maxV))
	v79 := v61.Sub(v63).Max(minV).Min(maxV)
	v80 := v62.Sub(v64).Max(minV).Min(maxV)
	v81 := zero.Sub(v67.Add(v68).Mul(kp181).Shift(itxShr(7)).Average(zero))
	v82 := v67.Sub(v68).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v83 := v71.Add(v72).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v84 := zero.Sub(v71.Sub(v72).Mul(kp181).Shift(itxShr(7)).Average(zero))
	v85 := v75.Add(v76).Mul(kp181).Shift(itxShr(7)).Average(zero)
	v86 := zero.Sub(v75.Sub(v76).Mul(kp181).Shift(itxShr(7)).Average(zero))
	v87 := zero.Sub(v79.Add(v80).Mul(kp181).Shift(itxShr(7)).Average(zero))
	v88 := v79.Sub(v80).Mul(kp181).Shift(itxShr(7)).Average(zero)
	y0_0, y0_1, y0_2, y0_3 := transpose4x4(v66, v74, v78, v70)
	y0_0.StoreArray((*[4]int32)(r0[0:]))
	y0_1.StoreArray((*[4]int32)(r1[0:]))
	y0_2.StoreArray((*[4]int32)(r2[0:]))
	y0_3.StoreArray((*[4]int32)(r3[0:]))
	y1_0, y1_1, y1_2, y1_3 := transpose4x4(v84, v88, v86, v82)
	y1_0.StoreArray((*[4]int32)(r0[4:]))
	y1_1.StoreArray((*[4]int32)(r1[4:]))
	y1_2.StoreArray((*[4]int32)(r2[4:]))
	y1_3.StoreArray((*[4]int32)(r3[4:]))
	y2_0, y2_1, y2_2, y2_3 := transpose4x4(v81, v85, v87, v83)
	y2_0.StoreArray((*[4]int32)(r0[8:]))
	y2_1.StoreArray((*[4]int32)(r1[8:]))
	y2_2.StoreArray((*[4]int32)(r2[8:]))
	y2_3.StoreArray((*[4]int32)(r3[8:]))
	y3_0, y3_1, y3_2, y3_3 := transpose4x4(v69, v77, v73, v65)
	y3_0.StoreArray((*[4]int32)(r0[12:]))
	y3_1.StoreArray((*[4]int32)(r1[12:]))
	y3_2.StoreArray((*[4]int32)(r2[12:]))
	y3_3.StoreArray((*[4]int32)(r3[12:]))
}

// Lane tables for the butterfly constants used by the kernels above; loading
// them is cheaper per call than broadcasting each constant.
var (
	itxCm2751 = [4]int32{-2751, -2751, -2751, -2751}
	itxCm2276 = [4]int32{-2276, -2276, -2276, -2276}
	itxCm2106 = [4]int32{-2106, -2106, -2106, -2106}
	itxCm1645 = [4]int32{-1645, -1645, -1645, -1645}
	itxCm1567 = [4]int32{-1567, -1567, -1567, -1567}
	itxCm1380 = [4]int32{-1380, -1380, -1380, -1380}
	itxCm1299 = [4]int32{-1299, -1299, -1299, -1299}
	itxCm1189 = [4]int32{-1189, -1189, -1189, -1189}
	itxCm1138 = [4]int32{-1138, -1138, -1138, -1138}
	itxCm1061 = [4]int32{-1061, -1061, -1061, -1061}
	itxCm799  = [4]int32{-799, -799, -799, -799}
	itxCm690  = [4]int32{-690, -690, -690, -690}
	itxCm601  = [4]int32{-601, -601, -601, -601}
	itxCm583  = [4]int32{-583, -583, -583, -583}
	itxCm484  = [4]int32{-484, -484, -484, -484}
	itxCm393  = [4]int32{-393, -393, -393, -393}
	itxCm312  = [4]int32{-312, -312, -312, -312}
	itxCm239  = [4]int32{-239, -239, -239, -239}
	itxCm176  = [4]int32{-176, -176, -176, -176}
	itxCm123  = [4]int32{-123, -123, -123, -123}
	itxCm79   = [4]int32{-79, -79, -79, -79}
	itxCm44   = [4]int32{-44, -44, -44, -44}
	itxCm20   = [4]int32{-20, -20, -20, -20}
	itxCm5    = [4]int32{-5, -5, -5, -5}
	itxCp5    = [4]int32{5, 5, 5, 5}
	itxCp20   = [4]int32{20, 20, 20, 20}
	itxCp79   = [4]int32{79, 79, 79, 79}
	itxCp123  = [4]int32{123, 123, 123, 123}
	itxCp181  = [4]int32{181, 181, 181, 181}
	itxCp201  = [4]int32{201, 201, 201, 201}
	itxCp312  = [4]int32{312, 312, 312, 312}
	itxCp393  = [4]int32{393, 393, 393, 393}
	itxCp401  = [4]int32{401, 401, 401, 401}
	itxCp484  = [4]int32{484, 484, 484, 484}
	itxCp601  = [4]int32{601, 601, 601, 601}
	itxCp690  = [4]int32{690, 690, 690, 690}
	itxCp799  = [4]int32{799, 799, 799, 799}
	itxCp995  = [4]int32{995, 995, 995, 995}
	itxCp1138 = [4]int32{1138, 1138, 1138, 1138}
	itxCp1189 = [4]int32{1189, 1189, 1189, 1189}
	itxCp1220 = [4]int32{1220, 1220, 1220, 1220}
	itxCp1299 = [4]int32{1299, 1299, 1299, 1299}
	itxCp1380 = [4]int32{1380, 1380, 1380, 1380}
	itxCp1567 = [4]int32{1567, 1567, 1567, 1567}
	itxCp1583 = [4]int32{1583, 1583, 1583, 1583}
	itxCp1645 = [4]int32{1645, 1645, 1645, 1645}
	itxCp1703 = [4]int32{1703, 1703, 1703, 1703}
	itxCp1751 = [4]int32{1751, 1751, 1751, 1751}
	itxCp1931 = [4]int32{1931, 1931, 1931, 1931}
	itxCp2106 = [4]int32{2106, 2106, 2106, 2106}
	itxCp2276 = [4]int32{2276, 2276, 2276, 2276}
	itxCp2751 = [4]int32{2751, 2751, 2751, 2751}
)

// arm64 lowering of the four-lane inverse-transform primitives used by the
// generated kernels (dct64_lanes4_gosimd.go). Each is small enough to inline.
//
// NEON has no rounding shift in archsimd, so a rounding shift is a bias added
// into the multiply-accumulate (itxRoundBias) followed by an arithmetic shift.
// The shift goes through SSHL by a register-resident negative amount: a
// constant ShiftAllRight is lowered as an insert+dup+sshl on every use.

func itxRoundBias(n uint8) archsimd.Int32x4 { return archsimd.BroadcastInt32x4(1 << (n - 1)) }

func itxShiftAmount(n int32) archsimd.Int32x4 { return archsimd.BroadcastInt32x4(-n) }

// itxShr is the count vector of an arithmetic right shift by n, for
// v.Shift(itxShr(n)) in the generated kernels. Broadcasting into a fresh
// vector lets the compiler share one count per function, where a constant
// ShiftAllRight inserts its count into the operand and so rebuilds it
// (MOVD + copy + INS + DUP) at every use.
func itxShr(n int32) archsimd.Int32x4 { return archsimd.BroadcastInt32x4(-n) }

// itxMulAcc returns acc + x*k (MLA).
func itxMulAcc(acc, x, k archsimd.Int32x4) archsimd.Int32x4 { return x.MulAdd(k, acc) }

func itxShr12(v, n12 archsimd.Int32x4) archsimd.Int32x4 { return v.Shift(n12) }

// transpose4x4 transposes a 4x4 block of int32 lanes held as four row vectors.
// It is its own inverse, so it both gathers rows into per-element lane vectors
// (rows become lanes) and scatters them back.
func transpose4x4(a, b, c, d archsimd.Int32x4) (archsimd.Int32x4, archsimd.Int32x4, archsimd.Int32x4, archsimd.Int32x4) {
	t0 := a.InterleaveLo(b) // a0 b0 a1 b1
	t1 := a.InterleaveHi(b) // a2 b2 a3 b3
	t2 := c.InterleaveLo(d) // c0 d0 c1 d1
	t3 := c.InterleaveHi(d) // c2 d2 c3 d3
	e0 := fdctInt32AsInt64(t0)
	e2 := fdctInt32AsInt64(t2)
	f1 := fdctInt32AsInt64(t1)
	f3 := fdctInt32AsInt64(t3)
	return fdctInt64AsInt32(e0.InterleaveLo(e2)), fdctInt64AsInt32(e0.InterleaveHi(e2)),
		fdctInt64AsInt32(f1.InterleaveLo(f3)), fdctInt64AsInt32(f1.InterleaveHi(f3))
}

// txbInitLevels8x8SIMD is the Go SIMD form of the 8x8 TXB level init:
// levels[col*12+row] = min(abs(coeff[col*8+row]), 127) for row 0..7, with the
// four pad bytes at col*12+8..11 cleared. abs wraps MinInt16 to the unsigned
// pattern 0x8000 (32768), which the unsigned min clips to 127 exactly as the
// saturating abs did.
func txbInitLevels8x8SIMD(coeffs *[64]int16, levels *[256]uint8) {
	lim := archsimd.BroadcastUint16x8(127)
	for col := range 8 {
		v := archsimd.LoadInt16x8Array((*[8]int16)(coeffs[col*8:]))
		lv := v.Abs().ToBits().Min(lim).TruncToUint8()
		base := col * 12
		binary.LittleEndian.PutUint64(levels[base:], lv.ReshapeToUint64s().GetElem(0))
		binary.LittleEndian.PutUint32(levels[base+8:], 0)
	}
}

func txbPrep8x8Levels2DSIMD(coeffs *[64]int16, levels *[256]uint8, absLevels *[64]uint16, eob int) TXB8x8PrepResult {
	txbInitLevels8x8SIMD(coeffs, levels)
	return txbPrep8x8Summary(coeffs, absLevels, eob)
}

// PrepTXB8x8Levels2D prepares padded levels and scan-ordered absolute levels
// for a default 8x8/Class2D TXB. levels must be zeroed by the caller; this
// mirrors the trusted TXB writer stack scratch. The arm64 SIMD path initialises
// the padded levels with Go SIMD (txb_gosimd_arm64.go), then computes the
// scan-order summary source-shaped to compute_cul_level.
func PrepTXB8x8Levels2D(coeffs *[64]int16, levels *[256]uint8, absLevels *[64]uint16, eob int) TXB8x8PrepResult {
	return txbPrep8x8Levels2DSIMD(coeffs, levels, absLevels, eob)
}
