// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package motion

import (
	"encoding/binary"
	"simd/archsimd"

	"github.com/thesyncim/goav1/internal/unsafeutil"
)

// The 16bpc six-tap vertical kernels (dav1d mc16 put_6tap/prep_6tap v) over
// Grid views. Each source row is widened to int32 once, when it is loaded,
// and the halved taps run as int32 multiply-accumulates (mc8Phase.w): a
// halved-tap sum of 10- or 12-bit samples needs 32-bit lanes, and widening
// the rows instead of the products drops the per-tap widening multiplies,
// their adds and the high-half moves (the sums are the same integers). Each
// eight-sample column strip keeps five rows resident and filters eight
// output rows per chunk (two, then one, for the last rows); the chunks'
// Fits tests are the only checks the rows pay, and the entries make them
// unreachable, so the kernels stay leaf functions.

// mc16V6I is the six-tap int32 vertical sum of rows r0..r5, plus rnd.
func mc16V6I(r0, r1, r2, r3, r4, r5, c0, c1, c2, c3, c4, c5, rnd archsimd.Int32x4) archsimd.Int32x4 {
	a := r0.MulAdd(c0, rnd)
	a = r1.MulAdd(c1, a)
	a = r2.MulAdd(c2, a)
	a = r3.MulAdd(c3, a)
	a = r4.MulAdd(c4, a)
	return r5.MulAdd(c5, a)
}

// mc16PutV6G is the 16bpc put vertical-only pass, clip((hs + 32) >> 6),
// for w = og.Width()/2 a multiple of 8: sg holds the h+5 source rows
// (h = og.Height()), each readable for 16 bytes from every eight-sample
// strip, and og the h destination rows.
func mc16PutV6G(sg unsafeutil.Grid[uint8], ph *mc8Phase, maxPix int16, og unsafeutil.Grid[uint8]) {
	c0 := archsimd.LoadInt32x4Array(&ph.w[0])
	c1 := archsimd.LoadInt32x4Array(&ph.w[1])
	c2 := archsimd.LoadInt32x4Array(&ph.w[2])
	c3 := archsimd.LoadInt32x4Array(&ph.w[3])
	c4 := archsimd.LoadInt32x4Array(&ph.w[4])
	c5 := archsimd.LoadInt32x4Array(&ph.w[5])
	rnd := archsimd.LoadInt32x4Array(&mc8Vec.w32)
	shv := archsimd.LoadInt32x4Array(&mc8Vec.wsh6)
	zero := archsimd.BroadcastInt16x8(0)
	maxV := archsimd.BroadcastInt16x8(maxPix)
	h, w := og.Height(), og.Width()/2
	for c := 0; c+8 <= w; c += 8 {
		if !sg.Fits(0, 2*c, 16, h+5) || !og.Fits(0, 2*c, 16, h) {
			return
		}
		s, o := sg.Sub(0, 2*c, 16, h+5), og.Sub(0, 2*c, 16, h)
		if !s.Fits(0, 0, 16, 5) {
			return
		}
		hd := s.Sub(0, 0, 16, 5)
		var x archsimd.Int16x8
		x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](hd, 0, 0)).ReshapeToUint16s().BitsToInt16()
		l0, h0 := x.ExtendLo4ToInt32(), x.HiToLo().ExtendLo4ToInt32()
		x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](hd, 1, 0)).ReshapeToUint16s().BitsToInt16()
		l1, h1 := x.ExtendLo4ToInt32(), x.HiToLo().ExtendLo4ToInt32()
		x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](hd, 2, 0)).ReshapeToUint16s().BitsToInt16()
		l2, h2 := x.ExtendLo4ToInt32(), x.HiToLo().ExtendLo4ToInt32()
		x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](hd, 3, 0)).ReshapeToUint16s().BitsToInt16()
		l3, h3 := x.ExtendLo4ToInt32(), x.HiToLo().ExtendLo4ToInt32()
		x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](hd, 4, 0)).ReshapeToUint16s().BitsToInt16()
		l4, h4 := x.ExtendLo4ToInt32(), x.HiToLo().ExtendLo4ToInt32()
		r := 0
		if h >= 8 {
			for ; r+8 <= h; r += 8 {
				if !s.Fits(r+5, 0, 16, 8) || !o.Fits(r, 0, 16, 8) {
					return
				}
				sw, ow := s.Sub(r+5, 0, 16, 8), o.Sub(r, 0, 16, 8)
				x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 0, 0)).ReshapeToUint16s().BitsToInt16()
				l5, h5 := x.ExtendLo4ToInt32(), x.HiToLo().ExtendLo4ToInt32()
				mc16Clip(mc16V6I(l0, l1, l2, l3, l4, l5, c0, c1, c2, c3, c4, c5, rnd).Shift(shv), mc16V6I(h0, h1, h2, h3, h4, h5, c0, c1, c2, c3, c4, c5, rnd).Shift(shv), zero, maxV).ToBits().ReshapeToUint8s().StoreArray(unsafeutil.Row[[16]byte](ow, 0, 0))
				x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 1, 0)).ReshapeToUint16s().BitsToInt16()
				l6, h6 := x.ExtendLo4ToInt32(), x.HiToLo().ExtendLo4ToInt32()
				mc16Clip(mc16V6I(l1, l2, l3, l4, l5, l6, c0, c1, c2, c3, c4, c5, rnd).Shift(shv), mc16V6I(h1, h2, h3, h4, h5, h6, c0, c1, c2, c3, c4, c5, rnd).Shift(shv), zero, maxV).ToBits().ReshapeToUint8s().StoreArray(unsafeutil.Row[[16]byte](ow, 1, 0))
				x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 2, 0)).ReshapeToUint16s().BitsToInt16()
				l7, h7 := x.ExtendLo4ToInt32(), x.HiToLo().ExtendLo4ToInt32()
				mc16Clip(mc16V6I(l2, l3, l4, l5, l6, l7, c0, c1, c2, c3, c4, c5, rnd).Shift(shv), mc16V6I(h2, h3, h4, h5, h6, h7, c0, c1, c2, c3, c4, c5, rnd).Shift(shv), zero, maxV).ToBits().ReshapeToUint8s().StoreArray(unsafeutil.Row[[16]byte](ow, 2, 0))
				x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 3, 0)).ReshapeToUint16s().BitsToInt16()
				l8, h8 := x.ExtendLo4ToInt32(), x.HiToLo().ExtendLo4ToInt32()
				mc16Clip(mc16V6I(l3, l4, l5, l6, l7, l8, c0, c1, c2, c3, c4, c5, rnd).Shift(shv), mc16V6I(h3, h4, h5, h6, h7, h8, c0, c1, c2, c3, c4, c5, rnd).Shift(shv), zero, maxV).ToBits().ReshapeToUint8s().StoreArray(unsafeutil.Row[[16]byte](ow, 3, 0))
				x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 4, 0)).ReshapeToUint16s().BitsToInt16()
				l9, h9 := x.ExtendLo4ToInt32(), x.HiToLo().ExtendLo4ToInt32()
				mc16Clip(mc16V6I(l4, l5, l6, l7, l8, l9, c0, c1, c2, c3, c4, c5, rnd).Shift(shv), mc16V6I(h4, h5, h6, h7, h8, h9, c0, c1, c2, c3, c4, c5, rnd).Shift(shv), zero, maxV).ToBits().ReshapeToUint8s().StoreArray(unsafeutil.Row[[16]byte](ow, 4, 0))
				x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 5, 0)).ReshapeToUint16s().BitsToInt16()
				l10, h10 := x.ExtendLo4ToInt32(), x.HiToLo().ExtendLo4ToInt32()
				mc16Clip(mc16V6I(l5, l6, l7, l8, l9, l10, c0, c1, c2, c3, c4, c5, rnd).Shift(shv), mc16V6I(h5, h6, h7, h8, h9, h10, c0, c1, c2, c3, c4, c5, rnd).Shift(shv), zero, maxV).ToBits().ReshapeToUint8s().StoreArray(unsafeutil.Row[[16]byte](ow, 5, 0))
				x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 6, 0)).ReshapeToUint16s().BitsToInt16()
				l11, h11 := x.ExtendLo4ToInt32(), x.HiToLo().ExtendLo4ToInt32()
				mc16Clip(mc16V6I(l6, l7, l8, l9, l10, l11, c0, c1, c2, c3, c4, c5, rnd).Shift(shv), mc16V6I(h6, h7, h8, h9, h10, h11, c0, c1, c2, c3, c4, c5, rnd).Shift(shv), zero, maxV).ToBits().ReshapeToUint8s().StoreArray(unsafeutil.Row[[16]byte](ow, 6, 0))
				x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 7, 0)).ReshapeToUint16s().BitsToInt16()
				l12, h12 := x.ExtendLo4ToInt32(), x.HiToLo().ExtendLo4ToInt32()
				mc16Clip(mc16V6I(l7, l8, l9, l10, l11, l12, c0, c1, c2, c3, c4, c5, rnd).Shift(shv), mc16V6I(h7, h8, h9, h10, h11, h12, c0, c1, c2, c3, c4, c5, rnd).Shift(shv), zero, maxV).ToBits().ReshapeToUint8s().StoreArray(unsafeutil.Row[[16]byte](ow, 7, 0))
				l0, h0, l1, h1, l2, h2, l3, h3, l4, h4 = l8, h8, l9, h9, l10, h10, l11, h11, l12, h12
			}
		}
		if r < h {
			if r+4 <= h {
				if !s.Fits(r+5, 0, 16, 4) || !o.Fits(r, 0, 16, 4) {
					return
				}
				sw, ow := s.Sub(r+5, 0, 16, 4), o.Sub(r, 0, 16, 4)
				x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 0, 0)).ReshapeToUint16s().BitsToInt16()
				l5, h5 := x.ExtendLo4ToInt32(), x.HiToLo().ExtendLo4ToInt32()
				mc16Clip(mc16V6I(l0, l1, l2, l3, l4, l5, c0, c1, c2, c3, c4, c5, rnd).Shift(shv), mc16V6I(h0, h1, h2, h3, h4, h5, c0, c1, c2, c3, c4, c5, rnd).Shift(shv), zero, maxV).ToBits().ReshapeToUint8s().StoreArray(unsafeutil.Row[[16]byte](ow, 0, 0))
				x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 1, 0)).ReshapeToUint16s().BitsToInt16()
				l6, h6 := x.ExtendLo4ToInt32(), x.HiToLo().ExtendLo4ToInt32()
				mc16Clip(mc16V6I(l1, l2, l3, l4, l5, l6, c0, c1, c2, c3, c4, c5, rnd).Shift(shv), mc16V6I(h1, h2, h3, h4, h5, h6, c0, c1, c2, c3, c4, c5, rnd).Shift(shv), zero, maxV).ToBits().ReshapeToUint8s().StoreArray(unsafeutil.Row[[16]byte](ow, 1, 0))
				x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 2, 0)).ReshapeToUint16s().BitsToInt16()
				l7, h7 := x.ExtendLo4ToInt32(), x.HiToLo().ExtendLo4ToInt32()
				mc16Clip(mc16V6I(l2, l3, l4, l5, l6, l7, c0, c1, c2, c3, c4, c5, rnd).Shift(shv), mc16V6I(h2, h3, h4, h5, h6, h7, c0, c1, c2, c3, c4, c5, rnd).Shift(shv), zero, maxV).ToBits().ReshapeToUint8s().StoreArray(unsafeutil.Row[[16]byte](ow, 2, 0))
				x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 3, 0)).ReshapeToUint16s().BitsToInt16()
				l8, h8 := x.ExtendLo4ToInt32(), x.HiToLo().ExtendLo4ToInt32()
				mc16Clip(mc16V6I(l3, l4, l5, l6, l7, l8, c0, c1, c2, c3, c4, c5, rnd).Shift(shv), mc16V6I(h3, h4, h5, h6, h7, h8, c0, c1, c2, c3, c4, c5, rnd).Shift(shv), zero, maxV).ToBits().ReshapeToUint8s().StoreArray(unsafeutil.Row[[16]byte](ow, 3, 0))
				l0, h0, l1, h1, l2, h2, l3, h3, l4, h4 = l4, h4, l5, h5, l6, h6, l7, h7, l8, h8
				r += 4
			}
			if r+2 <= h {
				if !s.Fits(r+5, 0, 16, 2) || !o.Fits(r, 0, 16, 2) {
					return
				}
				sw, ow := s.Sub(r+5, 0, 16, 2), o.Sub(r, 0, 16, 2)
				x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 0, 0)).ReshapeToUint16s().BitsToInt16()
				l5, h5 := x.ExtendLo4ToInt32(), x.HiToLo().ExtendLo4ToInt32()
				mc16Clip(mc16V6I(l0, l1, l2, l3, l4, l5, c0, c1, c2, c3, c4, c5, rnd).Shift(shv), mc16V6I(h0, h1, h2, h3, h4, h5, c0, c1, c2, c3, c4, c5, rnd).Shift(shv), zero, maxV).ToBits().ReshapeToUint8s().StoreArray(unsafeutil.Row[[16]byte](ow, 0, 0))
				x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 1, 0)).ReshapeToUint16s().BitsToInt16()
				l6, h6 := x.ExtendLo4ToInt32(), x.HiToLo().ExtendLo4ToInt32()
				mc16Clip(mc16V6I(l1, l2, l3, l4, l5, l6, c0, c1, c2, c3, c4, c5, rnd).Shift(shv), mc16V6I(h1, h2, h3, h4, h5, h6, c0, c1, c2, c3, c4, c5, rnd).Shift(shv), zero, maxV).ToBits().ReshapeToUint8s().StoreArray(unsafeutil.Row[[16]byte](ow, 1, 0))
				l0, h0, l1, h1, l2, h2, l3, h3, l4, h4 = l2, h2, l3, h3, l4, h4, l5, h5, l6, h6
				r += 2
			}
			if r < h {
				if !s.Fits(r+5, 0, 16, 1) || !o.Fits(r, 0, 16, 1) {
					return
				}
				sw, ow := s.Sub(r+5, 0, 16, 1), o.Sub(r, 0, 16, 1)
				x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 0, 0)).ReshapeToUint16s().BitsToInt16()
				l5, h5 := x.ExtendLo4ToInt32(), x.HiToLo().ExtendLo4ToInt32()
				mc16Clip(mc16V6I(l0, l1, l2, l3, l4, l5, c0, c1, c2, c3, c4, c5, rnd).Shift(shv), mc16V6I(h0, h1, h2, h3, h4, h5, c0, c1, c2, c3, c4, c5, rnd).Shift(shv), zero, maxV).ToBits().ReshapeToUint8s().StoreArray(unsafeutil.Row[[16]byte](ow, 0, 0))
			}
		}
	}
}

// mc16PutV6GW4 is mc16PutV6G for a four-sample-wide block.
func mc16PutV6GW4(s unsafeutil.Grid[uint8], ph *mc8Phase, maxPix int16, o unsafeutil.Grid[uint8]) {
	c0 := archsimd.LoadInt32x4Array(&ph.w[0])
	c1 := archsimd.LoadInt32x4Array(&ph.w[1])
	c2 := archsimd.LoadInt32x4Array(&ph.w[2])
	c3 := archsimd.LoadInt32x4Array(&ph.w[3])
	c4 := archsimd.LoadInt32x4Array(&ph.w[4])
	c5 := archsimd.LoadInt32x4Array(&ph.w[5])
	rnd := archsimd.LoadInt32x4Array(&mc8Vec.w32)
	shv := archsimd.LoadInt32x4Array(&mc8Vec.wsh6)
	zero := archsimd.BroadcastInt16x8(0)
	maxV := archsimd.BroadcastInt16x8(maxPix)
	var zero32 archsimd.Int32x4
	h := o.Height()
	if !s.Fits(0, 0, 16, 5) {
		return
	}
	hd := s.Sub(0, 0, 16, 5)
	var x archsimd.Int16x8
	x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](hd, 0, 0)).ReshapeToUint16s().BitsToInt16()
	l0 := x.ExtendLo4ToInt32()
	x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](hd, 1, 0)).ReshapeToUint16s().BitsToInt16()
	l1 := x.ExtendLo4ToInt32()
	x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](hd, 2, 0)).ReshapeToUint16s().BitsToInt16()
	l2 := x.ExtendLo4ToInt32()
	x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](hd, 3, 0)).ReshapeToUint16s().BitsToInt16()
	l3 := x.ExtendLo4ToInt32()
	x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](hd, 4, 0)).ReshapeToUint16s().BitsToInt16()
	l4 := x.ExtendLo4ToInt32()
	r := 0
	if h >= 8 {
		for ; r+8 <= h; r += 8 {
			if !s.Fits(r+5, 0, 16, 8) || !o.Fits(r, 0, 8, 8) {
				return
			}
			sw, ow := s.Sub(r+5, 0, 16, 8), o.Sub(r, 0, 8, 8)
			x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 0, 0)).ReshapeToUint16s().BitsToInt16()
			l5 := x.ExtendLo4ToInt32()
			binary.LittleEndian.PutUint64(unsafeutil.Row[[8]byte](ow, 0, 0)[:], mc16Clip(mc16V6I(l0, l1, l2, l3, l4, l5, c0, c1, c2, c3, c4, c5, rnd).Shift(shv), zero32, zero, maxV).ToBits().ReshapeToUint64s().GetElem(0))
			x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 1, 0)).ReshapeToUint16s().BitsToInt16()
			l6 := x.ExtendLo4ToInt32()
			binary.LittleEndian.PutUint64(unsafeutil.Row[[8]byte](ow, 1, 0)[:], mc16Clip(mc16V6I(l1, l2, l3, l4, l5, l6, c0, c1, c2, c3, c4, c5, rnd).Shift(shv), zero32, zero, maxV).ToBits().ReshapeToUint64s().GetElem(0))
			x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 2, 0)).ReshapeToUint16s().BitsToInt16()
			l7 := x.ExtendLo4ToInt32()
			binary.LittleEndian.PutUint64(unsafeutil.Row[[8]byte](ow, 2, 0)[:], mc16Clip(mc16V6I(l2, l3, l4, l5, l6, l7, c0, c1, c2, c3, c4, c5, rnd).Shift(shv), zero32, zero, maxV).ToBits().ReshapeToUint64s().GetElem(0))
			x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 3, 0)).ReshapeToUint16s().BitsToInt16()
			l8 := x.ExtendLo4ToInt32()
			binary.LittleEndian.PutUint64(unsafeutil.Row[[8]byte](ow, 3, 0)[:], mc16Clip(mc16V6I(l3, l4, l5, l6, l7, l8, c0, c1, c2, c3, c4, c5, rnd).Shift(shv), zero32, zero, maxV).ToBits().ReshapeToUint64s().GetElem(0))
			x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 4, 0)).ReshapeToUint16s().BitsToInt16()
			l9 := x.ExtendLo4ToInt32()
			binary.LittleEndian.PutUint64(unsafeutil.Row[[8]byte](ow, 4, 0)[:], mc16Clip(mc16V6I(l4, l5, l6, l7, l8, l9, c0, c1, c2, c3, c4, c5, rnd).Shift(shv), zero32, zero, maxV).ToBits().ReshapeToUint64s().GetElem(0))
			x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 5, 0)).ReshapeToUint16s().BitsToInt16()
			l10 := x.ExtendLo4ToInt32()
			binary.LittleEndian.PutUint64(unsafeutil.Row[[8]byte](ow, 5, 0)[:], mc16Clip(mc16V6I(l5, l6, l7, l8, l9, l10, c0, c1, c2, c3, c4, c5, rnd).Shift(shv), zero32, zero, maxV).ToBits().ReshapeToUint64s().GetElem(0))
			x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 6, 0)).ReshapeToUint16s().BitsToInt16()
			l11 := x.ExtendLo4ToInt32()
			binary.LittleEndian.PutUint64(unsafeutil.Row[[8]byte](ow, 6, 0)[:], mc16Clip(mc16V6I(l6, l7, l8, l9, l10, l11, c0, c1, c2, c3, c4, c5, rnd).Shift(shv), zero32, zero, maxV).ToBits().ReshapeToUint64s().GetElem(0))
			x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 7, 0)).ReshapeToUint16s().BitsToInt16()
			l12 := x.ExtendLo4ToInt32()
			binary.LittleEndian.PutUint64(unsafeutil.Row[[8]byte](ow, 7, 0)[:], mc16Clip(mc16V6I(l7, l8, l9, l10, l11, l12, c0, c1, c2, c3, c4, c5, rnd).Shift(shv), zero32, zero, maxV).ToBits().ReshapeToUint64s().GetElem(0))
			l0, l1, l2, l3, l4 = l8, l9, l10, l11, l12
		}
	}
	if r < h {
		if r+4 <= h {
			if !s.Fits(r+5, 0, 16, 4) || !o.Fits(r, 0, 8, 4) {
				return
			}
			sw, ow := s.Sub(r+5, 0, 16, 4), o.Sub(r, 0, 8, 4)
			x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 0, 0)).ReshapeToUint16s().BitsToInt16()
			l5 := x.ExtendLo4ToInt32()
			binary.LittleEndian.PutUint64(unsafeutil.Row[[8]byte](ow, 0, 0)[:], mc16Clip(mc16V6I(l0, l1, l2, l3, l4, l5, c0, c1, c2, c3, c4, c5, rnd).Shift(shv), zero32, zero, maxV).ToBits().ReshapeToUint64s().GetElem(0))
			x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 1, 0)).ReshapeToUint16s().BitsToInt16()
			l6 := x.ExtendLo4ToInt32()
			binary.LittleEndian.PutUint64(unsafeutil.Row[[8]byte](ow, 1, 0)[:], mc16Clip(mc16V6I(l1, l2, l3, l4, l5, l6, c0, c1, c2, c3, c4, c5, rnd).Shift(shv), zero32, zero, maxV).ToBits().ReshapeToUint64s().GetElem(0))
			x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 2, 0)).ReshapeToUint16s().BitsToInt16()
			l7 := x.ExtendLo4ToInt32()
			binary.LittleEndian.PutUint64(unsafeutil.Row[[8]byte](ow, 2, 0)[:], mc16Clip(mc16V6I(l2, l3, l4, l5, l6, l7, c0, c1, c2, c3, c4, c5, rnd).Shift(shv), zero32, zero, maxV).ToBits().ReshapeToUint64s().GetElem(0))
			x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 3, 0)).ReshapeToUint16s().BitsToInt16()
			l8 := x.ExtendLo4ToInt32()
			binary.LittleEndian.PutUint64(unsafeutil.Row[[8]byte](ow, 3, 0)[:], mc16Clip(mc16V6I(l3, l4, l5, l6, l7, l8, c0, c1, c2, c3, c4, c5, rnd).Shift(shv), zero32, zero, maxV).ToBits().ReshapeToUint64s().GetElem(0))
			l0, l1, l2, l3, l4 = l4, l5, l6, l7, l8
			r += 4
		}
		if r+2 <= h {
			if !s.Fits(r+5, 0, 16, 2) || !o.Fits(r, 0, 8, 2) {
				return
			}
			sw, ow := s.Sub(r+5, 0, 16, 2), o.Sub(r, 0, 8, 2)
			x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 0, 0)).ReshapeToUint16s().BitsToInt16()
			l5 := x.ExtendLo4ToInt32()
			binary.LittleEndian.PutUint64(unsafeutil.Row[[8]byte](ow, 0, 0)[:], mc16Clip(mc16V6I(l0, l1, l2, l3, l4, l5, c0, c1, c2, c3, c4, c5, rnd).Shift(shv), zero32, zero, maxV).ToBits().ReshapeToUint64s().GetElem(0))
			x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 1, 0)).ReshapeToUint16s().BitsToInt16()
			l6 := x.ExtendLo4ToInt32()
			binary.LittleEndian.PutUint64(unsafeutil.Row[[8]byte](ow, 1, 0)[:], mc16Clip(mc16V6I(l1, l2, l3, l4, l5, l6, c0, c1, c2, c3, c4, c5, rnd).Shift(shv), zero32, zero, maxV).ToBits().ReshapeToUint64s().GetElem(0))
			l0, l1, l2, l3, l4 = l2, l3, l4, l5, l6
			r += 2
		}
		if r < h {
			if !s.Fits(r+5, 0, 16, 1) || !o.Fits(r, 0, 8, 1) {
				return
			}
			sw, ow := s.Sub(r+5, 0, 16, 1), o.Sub(r, 0, 8, 1)
			x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 0, 0)).ReshapeToUint16s().BitsToInt16()
			l5 := x.ExtendLo4ToInt32()
			binary.LittleEndian.PutUint64(unsafeutil.Row[[8]byte](ow, 0, 0)[:], mc16Clip(mc16V6I(l0, l1, l2, l3, l4, l5, c0, c1, c2, c3, c4, c5, rnd).Shift(shv), zero32, zero, maxV).ToBits().ReshapeToUint64s().GetElem(0))
		}
	}
}

// mc16PrepV6G is the 16bpc prep vertical-only pass, ((hs + 2^(sh-1)) >> sh) + roundOffset,
// for w = og.Width() a multiple of 8: sg holds the h+5 source rows
// (h = og.Height()), each readable for 16 bytes from every eight-sample
// strip, and og the h CONV_BUF rows.
func mc16PrepV6G(sg unsafeutil.Grid[uint8], ph *mc8Phase, sh int, og unsafeutil.Grid[uint16]) {
	c0 := archsimd.LoadInt32x4Array(&ph.w[0])
	c1 := archsimd.LoadInt32x4Array(&ph.w[1])
	c2 := archsimd.LoadInt32x4Array(&ph.w[2])
	c3 := archsimd.LoadInt32x4Array(&ph.w[3])
	c4 := archsimd.LoadInt32x4Array(&ph.w[4])
	c5 := archsimd.LoadInt32x4Array(&ph.w[5])
	rnd := archsimd.BroadcastInt32x4(int32(1)<<(sh-1) + int32(mc16PrepRoundOffset)<<sh)
	shv := archsimd.BroadcastInt32x4(int32(-sh))
	h, w := og.Height(), og.Width()
	for c := 0; c+8 <= w; c += 8 {
		if !sg.Fits(0, 2*c, 16, h+5) || !og.Fits(0, c, 8, h) {
			return
		}
		s, o := sg.Sub(0, 2*c, 16, h+5), og.Sub(0, c, 8, h)
		if !s.Fits(0, 0, 16, 5) {
			return
		}
		hd := s.Sub(0, 0, 16, 5)
		var x archsimd.Int16x8
		x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](hd, 0, 0)).ReshapeToUint16s().BitsToInt16()
		l0, h0 := x.ExtendLo4ToInt32(), x.HiToLo().ExtendLo4ToInt32()
		x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](hd, 1, 0)).ReshapeToUint16s().BitsToInt16()
		l1, h1 := x.ExtendLo4ToInt32(), x.HiToLo().ExtendLo4ToInt32()
		x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](hd, 2, 0)).ReshapeToUint16s().BitsToInt16()
		l2, h2 := x.ExtendLo4ToInt32(), x.HiToLo().ExtendLo4ToInt32()
		x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](hd, 3, 0)).ReshapeToUint16s().BitsToInt16()
		l3, h3 := x.ExtendLo4ToInt32(), x.HiToLo().ExtendLo4ToInt32()
		x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](hd, 4, 0)).ReshapeToUint16s().BitsToInt16()
		l4, h4 := x.ExtendLo4ToInt32(), x.HiToLo().ExtendLo4ToInt32()
		r := 0
		if h >= 8 {
			for ; r+8 <= h; r += 8 {
				if !s.Fits(r+5, 0, 16, 8) || !o.Fits(r, 0, 8, 8) {
					return
				}
				sw, ow := s.Sub(r+5, 0, 16, 8), o.Sub(r, 0, 8, 8)
				x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 0, 0)).ReshapeToUint16s().BitsToInt16()
				l5, h5 := x.ExtendLo4ToInt32(), x.HiToLo().ExtendLo4ToInt32()
				mc16PackU16(mc16V6I(l0, l1, l2, l3, l4, l5, c0, c1, c2, c3, c4, c5, rnd).Shift(shv), mc16V6I(h0, h1, h2, h3, h4, h5, c0, c1, c2, c3, c4, c5, rnd).Shift(shv)).StoreArray(unsafeutil.Row[[8]uint16](ow, 0, 0))
				x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 1, 0)).ReshapeToUint16s().BitsToInt16()
				l6, h6 := x.ExtendLo4ToInt32(), x.HiToLo().ExtendLo4ToInt32()
				mc16PackU16(mc16V6I(l1, l2, l3, l4, l5, l6, c0, c1, c2, c3, c4, c5, rnd).Shift(shv), mc16V6I(h1, h2, h3, h4, h5, h6, c0, c1, c2, c3, c4, c5, rnd).Shift(shv)).StoreArray(unsafeutil.Row[[8]uint16](ow, 1, 0))
				x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 2, 0)).ReshapeToUint16s().BitsToInt16()
				l7, h7 := x.ExtendLo4ToInt32(), x.HiToLo().ExtendLo4ToInt32()
				mc16PackU16(mc16V6I(l2, l3, l4, l5, l6, l7, c0, c1, c2, c3, c4, c5, rnd).Shift(shv), mc16V6I(h2, h3, h4, h5, h6, h7, c0, c1, c2, c3, c4, c5, rnd).Shift(shv)).StoreArray(unsafeutil.Row[[8]uint16](ow, 2, 0))
				x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 3, 0)).ReshapeToUint16s().BitsToInt16()
				l8, h8 := x.ExtendLo4ToInt32(), x.HiToLo().ExtendLo4ToInt32()
				mc16PackU16(mc16V6I(l3, l4, l5, l6, l7, l8, c0, c1, c2, c3, c4, c5, rnd).Shift(shv), mc16V6I(h3, h4, h5, h6, h7, h8, c0, c1, c2, c3, c4, c5, rnd).Shift(shv)).StoreArray(unsafeutil.Row[[8]uint16](ow, 3, 0))
				x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 4, 0)).ReshapeToUint16s().BitsToInt16()
				l9, h9 := x.ExtendLo4ToInt32(), x.HiToLo().ExtendLo4ToInt32()
				mc16PackU16(mc16V6I(l4, l5, l6, l7, l8, l9, c0, c1, c2, c3, c4, c5, rnd).Shift(shv), mc16V6I(h4, h5, h6, h7, h8, h9, c0, c1, c2, c3, c4, c5, rnd).Shift(shv)).StoreArray(unsafeutil.Row[[8]uint16](ow, 4, 0))
				x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 5, 0)).ReshapeToUint16s().BitsToInt16()
				l10, h10 := x.ExtendLo4ToInt32(), x.HiToLo().ExtendLo4ToInt32()
				mc16PackU16(mc16V6I(l5, l6, l7, l8, l9, l10, c0, c1, c2, c3, c4, c5, rnd).Shift(shv), mc16V6I(h5, h6, h7, h8, h9, h10, c0, c1, c2, c3, c4, c5, rnd).Shift(shv)).StoreArray(unsafeutil.Row[[8]uint16](ow, 5, 0))
				x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 6, 0)).ReshapeToUint16s().BitsToInt16()
				l11, h11 := x.ExtendLo4ToInt32(), x.HiToLo().ExtendLo4ToInt32()
				mc16PackU16(mc16V6I(l6, l7, l8, l9, l10, l11, c0, c1, c2, c3, c4, c5, rnd).Shift(shv), mc16V6I(h6, h7, h8, h9, h10, h11, c0, c1, c2, c3, c4, c5, rnd).Shift(shv)).StoreArray(unsafeutil.Row[[8]uint16](ow, 6, 0))
				x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 7, 0)).ReshapeToUint16s().BitsToInt16()
				l12, h12 := x.ExtendLo4ToInt32(), x.HiToLo().ExtendLo4ToInt32()
				mc16PackU16(mc16V6I(l7, l8, l9, l10, l11, l12, c0, c1, c2, c3, c4, c5, rnd).Shift(shv), mc16V6I(h7, h8, h9, h10, h11, h12, c0, c1, c2, c3, c4, c5, rnd).Shift(shv)).StoreArray(unsafeutil.Row[[8]uint16](ow, 7, 0))
				l0, h0, l1, h1, l2, h2, l3, h3, l4, h4 = l8, h8, l9, h9, l10, h10, l11, h11, l12, h12
			}
		}
		if r < h {
			if r+4 <= h {
				if !s.Fits(r+5, 0, 16, 4) || !o.Fits(r, 0, 8, 4) {
					return
				}
				sw, ow := s.Sub(r+5, 0, 16, 4), o.Sub(r, 0, 8, 4)
				x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 0, 0)).ReshapeToUint16s().BitsToInt16()
				l5, h5 := x.ExtendLo4ToInt32(), x.HiToLo().ExtendLo4ToInt32()
				mc16PackU16(mc16V6I(l0, l1, l2, l3, l4, l5, c0, c1, c2, c3, c4, c5, rnd).Shift(shv), mc16V6I(h0, h1, h2, h3, h4, h5, c0, c1, c2, c3, c4, c5, rnd).Shift(shv)).StoreArray(unsafeutil.Row[[8]uint16](ow, 0, 0))
				x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 1, 0)).ReshapeToUint16s().BitsToInt16()
				l6, h6 := x.ExtendLo4ToInt32(), x.HiToLo().ExtendLo4ToInt32()
				mc16PackU16(mc16V6I(l1, l2, l3, l4, l5, l6, c0, c1, c2, c3, c4, c5, rnd).Shift(shv), mc16V6I(h1, h2, h3, h4, h5, h6, c0, c1, c2, c3, c4, c5, rnd).Shift(shv)).StoreArray(unsafeutil.Row[[8]uint16](ow, 1, 0))
				x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 2, 0)).ReshapeToUint16s().BitsToInt16()
				l7, h7 := x.ExtendLo4ToInt32(), x.HiToLo().ExtendLo4ToInt32()
				mc16PackU16(mc16V6I(l2, l3, l4, l5, l6, l7, c0, c1, c2, c3, c4, c5, rnd).Shift(shv), mc16V6I(h2, h3, h4, h5, h6, h7, c0, c1, c2, c3, c4, c5, rnd).Shift(shv)).StoreArray(unsafeutil.Row[[8]uint16](ow, 2, 0))
				x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 3, 0)).ReshapeToUint16s().BitsToInt16()
				l8, h8 := x.ExtendLo4ToInt32(), x.HiToLo().ExtendLo4ToInt32()
				mc16PackU16(mc16V6I(l3, l4, l5, l6, l7, l8, c0, c1, c2, c3, c4, c5, rnd).Shift(shv), mc16V6I(h3, h4, h5, h6, h7, h8, c0, c1, c2, c3, c4, c5, rnd).Shift(shv)).StoreArray(unsafeutil.Row[[8]uint16](ow, 3, 0))
				l0, h0, l1, h1, l2, h2, l3, h3, l4, h4 = l4, h4, l5, h5, l6, h6, l7, h7, l8, h8
				r += 4
			}
			if r+2 <= h {
				if !s.Fits(r+5, 0, 16, 2) || !o.Fits(r, 0, 8, 2) {
					return
				}
				sw, ow := s.Sub(r+5, 0, 16, 2), o.Sub(r, 0, 8, 2)
				x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 0, 0)).ReshapeToUint16s().BitsToInt16()
				l5, h5 := x.ExtendLo4ToInt32(), x.HiToLo().ExtendLo4ToInt32()
				mc16PackU16(mc16V6I(l0, l1, l2, l3, l4, l5, c0, c1, c2, c3, c4, c5, rnd).Shift(shv), mc16V6I(h0, h1, h2, h3, h4, h5, c0, c1, c2, c3, c4, c5, rnd).Shift(shv)).StoreArray(unsafeutil.Row[[8]uint16](ow, 0, 0))
				x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 1, 0)).ReshapeToUint16s().BitsToInt16()
				l6, h6 := x.ExtendLo4ToInt32(), x.HiToLo().ExtendLo4ToInt32()
				mc16PackU16(mc16V6I(l1, l2, l3, l4, l5, l6, c0, c1, c2, c3, c4, c5, rnd).Shift(shv), mc16V6I(h1, h2, h3, h4, h5, h6, c0, c1, c2, c3, c4, c5, rnd).Shift(shv)).StoreArray(unsafeutil.Row[[8]uint16](ow, 1, 0))
				l0, h0, l1, h1, l2, h2, l3, h3, l4, h4 = l2, h2, l3, h3, l4, h4, l5, h5, l6, h6
				r += 2
			}
			if r < h {
				if !s.Fits(r+5, 0, 16, 1) || !o.Fits(r, 0, 8, 1) {
					return
				}
				sw, ow := s.Sub(r+5, 0, 16, 1), o.Sub(r, 0, 8, 1)
				x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 0, 0)).ReshapeToUint16s().BitsToInt16()
				l5, h5 := x.ExtendLo4ToInt32(), x.HiToLo().ExtendLo4ToInt32()
				mc16PackU16(mc16V6I(l0, l1, l2, l3, l4, l5, c0, c1, c2, c3, c4, c5, rnd).Shift(shv), mc16V6I(h0, h1, h2, h3, h4, h5, c0, c1, c2, c3, c4, c5, rnd).Shift(shv)).StoreArray(unsafeutil.Row[[8]uint16](ow, 0, 0))
			}
		}
	}
}

// mc16PrepV6GW4 is mc16PrepV6G for a four-sample-wide block.
func mc16PrepV6GW4(s unsafeutil.Grid[uint8], ph *mc8Phase, sh int, o unsafeutil.Grid[uint16]) {
	c0 := archsimd.LoadInt32x4Array(&ph.w[0])
	c1 := archsimd.LoadInt32x4Array(&ph.w[1])
	c2 := archsimd.LoadInt32x4Array(&ph.w[2])
	c3 := archsimd.LoadInt32x4Array(&ph.w[3])
	c4 := archsimd.LoadInt32x4Array(&ph.w[4])
	c5 := archsimd.LoadInt32x4Array(&ph.w[5])
	rnd := archsimd.BroadcastInt32x4(int32(1)<<(sh-1) + int32(mc16PrepRoundOffset)<<sh)
	shv := archsimd.BroadcastInt32x4(int32(-sh))
	var zero32 archsimd.Int32x4
	h := o.Height()
	if !s.Fits(0, 0, 16, 5) {
		return
	}
	hd := s.Sub(0, 0, 16, 5)
	var x archsimd.Int16x8
	x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](hd, 0, 0)).ReshapeToUint16s().BitsToInt16()
	l0 := x.ExtendLo4ToInt32()
	x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](hd, 1, 0)).ReshapeToUint16s().BitsToInt16()
	l1 := x.ExtendLo4ToInt32()
	x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](hd, 2, 0)).ReshapeToUint16s().BitsToInt16()
	l2 := x.ExtendLo4ToInt32()
	x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](hd, 3, 0)).ReshapeToUint16s().BitsToInt16()
	l3 := x.ExtendLo4ToInt32()
	x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](hd, 4, 0)).ReshapeToUint16s().BitsToInt16()
	l4 := x.ExtendLo4ToInt32()
	r := 0
	if h >= 8 {
		for ; r+8 <= h; r += 8 {
			if !s.Fits(r+5, 0, 16, 8) || !o.Fits(r, 0, 4, 8) {
				return
			}
			sw, ow := s.Sub(r+5, 0, 16, 8), o.Sub(r, 0, 4, 8)
			x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 0, 0)).ReshapeToUint16s().BitsToInt16()
			l5 := x.ExtendLo4ToInt32()
			mc8Store4U16(unsafeutil.Row[[4]uint16](ow, 0, 0), mc16PackU16(mc16V6I(l0, l1, l2, l3, l4, l5, c0, c1, c2, c3, c4, c5, rnd).Shift(shv), zero32).ReshapeToUint64s().GetElem(0))
			x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 1, 0)).ReshapeToUint16s().BitsToInt16()
			l6 := x.ExtendLo4ToInt32()
			mc8Store4U16(unsafeutil.Row[[4]uint16](ow, 1, 0), mc16PackU16(mc16V6I(l1, l2, l3, l4, l5, l6, c0, c1, c2, c3, c4, c5, rnd).Shift(shv), zero32).ReshapeToUint64s().GetElem(0))
			x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 2, 0)).ReshapeToUint16s().BitsToInt16()
			l7 := x.ExtendLo4ToInt32()
			mc8Store4U16(unsafeutil.Row[[4]uint16](ow, 2, 0), mc16PackU16(mc16V6I(l2, l3, l4, l5, l6, l7, c0, c1, c2, c3, c4, c5, rnd).Shift(shv), zero32).ReshapeToUint64s().GetElem(0))
			x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 3, 0)).ReshapeToUint16s().BitsToInt16()
			l8 := x.ExtendLo4ToInt32()
			mc8Store4U16(unsafeutil.Row[[4]uint16](ow, 3, 0), mc16PackU16(mc16V6I(l3, l4, l5, l6, l7, l8, c0, c1, c2, c3, c4, c5, rnd).Shift(shv), zero32).ReshapeToUint64s().GetElem(0))
			x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 4, 0)).ReshapeToUint16s().BitsToInt16()
			l9 := x.ExtendLo4ToInt32()
			mc8Store4U16(unsafeutil.Row[[4]uint16](ow, 4, 0), mc16PackU16(mc16V6I(l4, l5, l6, l7, l8, l9, c0, c1, c2, c3, c4, c5, rnd).Shift(shv), zero32).ReshapeToUint64s().GetElem(0))
			x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 5, 0)).ReshapeToUint16s().BitsToInt16()
			l10 := x.ExtendLo4ToInt32()
			mc8Store4U16(unsafeutil.Row[[4]uint16](ow, 5, 0), mc16PackU16(mc16V6I(l5, l6, l7, l8, l9, l10, c0, c1, c2, c3, c4, c5, rnd).Shift(shv), zero32).ReshapeToUint64s().GetElem(0))
			x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 6, 0)).ReshapeToUint16s().BitsToInt16()
			l11 := x.ExtendLo4ToInt32()
			mc8Store4U16(unsafeutil.Row[[4]uint16](ow, 6, 0), mc16PackU16(mc16V6I(l6, l7, l8, l9, l10, l11, c0, c1, c2, c3, c4, c5, rnd).Shift(shv), zero32).ReshapeToUint64s().GetElem(0))
			x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 7, 0)).ReshapeToUint16s().BitsToInt16()
			l12 := x.ExtendLo4ToInt32()
			mc8Store4U16(unsafeutil.Row[[4]uint16](ow, 7, 0), mc16PackU16(mc16V6I(l7, l8, l9, l10, l11, l12, c0, c1, c2, c3, c4, c5, rnd).Shift(shv), zero32).ReshapeToUint64s().GetElem(0))
			l0, l1, l2, l3, l4 = l8, l9, l10, l11, l12
		}
	}
	if r < h {
		if r+4 <= h {
			if !s.Fits(r+5, 0, 16, 4) || !o.Fits(r, 0, 4, 4) {
				return
			}
			sw, ow := s.Sub(r+5, 0, 16, 4), o.Sub(r, 0, 4, 4)
			x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 0, 0)).ReshapeToUint16s().BitsToInt16()
			l5 := x.ExtendLo4ToInt32()
			mc8Store4U16(unsafeutil.Row[[4]uint16](ow, 0, 0), mc16PackU16(mc16V6I(l0, l1, l2, l3, l4, l5, c0, c1, c2, c3, c4, c5, rnd).Shift(shv), zero32).ReshapeToUint64s().GetElem(0))
			x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 1, 0)).ReshapeToUint16s().BitsToInt16()
			l6 := x.ExtendLo4ToInt32()
			mc8Store4U16(unsafeutil.Row[[4]uint16](ow, 1, 0), mc16PackU16(mc16V6I(l1, l2, l3, l4, l5, l6, c0, c1, c2, c3, c4, c5, rnd).Shift(shv), zero32).ReshapeToUint64s().GetElem(0))
			x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 2, 0)).ReshapeToUint16s().BitsToInt16()
			l7 := x.ExtendLo4ToInt32()
			mc8Store4U16(unsafeutil.Row[[4]uint16](ow, 2, 0), mc16PackU16(mc16V6I(l2, l3, l4, l5, l6, l7, c0, c1, c2, c3, c4, c5, rnd).Shift(shv), zero32).ReshapeToUint64s().GetElem(0))
			x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 3, 0)).ReshapeToUint16s().BitsToInt16()
			l8 := x.ExtendLo4ToInt32()
			mc8Store4U16(unsafeutil.Row[[4]uint16](ow, 3, 0), mc16PackU16(mc16V6I(l3, l4, l5, l6, l7, l8, c0, c1, c2, c3, c4, c5, rnd).Shift(shv), zero32).ReshapeToUint64s().GetElem(0))
			l0, l1, l2, l3, l4 = l4, l5, l6, l7, l8
			r += 4
		}
		if r+2 <= h {
			if !s.Fits(r+5, 0, 16, 2) || !o.Fits(r, 0, 4, 2) {
				return
			}
			sw, ow := s.Sub(r+5, 0, 16, 2), o.Sub(r, 0, 4, 2)
			x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 0, 0)).ReshapeToUint16s().BitsToInt16()
			l5 := x.ExtendLo4ToInt32()
			mc8Store4U16(unsafeutil.Row[[4]uint16](ow, 0, 0), mc16PackU16(mc16V6I(l0, l1, l2, l3, l4, l5, c0, c1, c2, c3, c4, c5, rnd).Shift(shv), zero32).ReshapeToUint64s().GetElem(0))
			x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 1, 0)).ReshapeToUint16s().BitsToInt16()
			l6 := x.ExtendLo4ToInt32()
			mc8Store4U16(unsafeutil.Row[[4]uint16](ow, 1, 0), mc16PackU16(mc16V6I(l1, l2, l3, l4, l5, l6, c0, c1, c2, c3, c4, c5, rnd).Shift(shv), zero32).ReshapeToUint64s().GetElem(0))
			l0, l1, l2, l3, l4 = l2, l3, l4, l5, l6
			r += 2
		}
		if r < h {
			if !s.Fits(r+5, 0, 16, 1) || !o.Fits(r, 0, 4, 1) {
				return
			}
			sw, ow := s.Sub(r+5, 0, 16, 1), o.Sub(r, 0, 4, 1)
			x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 0, 0)).ReshapeToUint16s().BitsToInt16()
			l5 := x.ExtendLo4ToInt32()
			mc8Store4U16(unsafeutil.Row[[4]uint16](ow, 0, 0), mc16PackU16(mc16V6I(l0, l1, l2, l3, l4, l5, c0, c1, c2, c3, c4, c5, rnd).Shift(shv), zero32).ReshapeToUint64s().GetElem(0))
		}
	}
}
