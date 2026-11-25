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

// The 8-bit six-tap 2D (hv) kernels over Grid views: the horizontal pass
// writes the strip-major int32 intermediate (strips of eight columns, each
// strip's rows contiguous) through one Grid per strip, and the vertical pass
// writes eight output rows per chunk of the destination view. As in the
// single-pass kernels, every Fits test is unreachable (the entries sized the
// views and the intermediate), so the kernels stay leaf functions.

// mc8H4S is the four-tap halved horizontal sum (taps 1..4 of the six-tap
// window, where taps 0 and 5 are zero) plus rnd.
func mc8H4S(p *[16]uint8, c1, c2, c3, c4, rnd archsimd.Int16x8) archsimd.Int16x8 {
	a := archsimd.LoadUint8x16Array(p)
	lo, hi := a.ExtendLo8ToUint16().ReshapeToUint8s(), a.HiToLo().ExtendLo8ToUint16().ReshapeToUint8s()
	s := hi.ConcatShiftBytesRight(lo, 2).ReshapeToUint16s().BitsToInt16().MulAdd(c1, rnd)
	s = hi.ConcatShiftBytesRight(lo, 4).ReshapeToUint16s().BitsToInt16().MulAdd(c2, s)
	s = hi.ConcatShiftBytesRight(lo, 6).ReshapeToUint16s().BitsToInt16().MulAdd(c3, s)
	return hi.ConcatShiftBytesRight(lo, 8).ReshapeToUint16s().BitsToInt16().MulAdd(c4, s)
}

// mc8PutH4S4 is mc8PutH6S4 when the horizontal filter has at most 4 non-zero
// taps (taps 1..4: regular4, smooth4, bilinear — the 99.9% case for w == 4).
func mc8PutH4S4(s unsafeutil.Grid[uint8], ph *mc8Phase, o unsafeutil.Grid[uint8]) {
	c1 := archsimd.LoadInt16x8Array(&ph.h[1])
	c2 := archsimd.LoadInt16x8Array(&ph.h[2])
	c3 := archsimd.LoadInt16x8Array(&ph.h[3])
	c4 := archsimd.LoadInt16x8Array(&ph.h[4])
	rnd := archsimd.LoadInt16x8Array(&mc8Vec.rnd34)
	sh := archsimd.LoadInt16x8Array(&mc8Vec.sh6)
	h := o.Height()
	r := 0
	if h >= 8 {
		for ; r+8 <= h; r += 8 {
			if !s.Fits(r, 0, 16, 8) || !o.Fits(r, 0, 4, 8) {
				return
			}
			sw, ow := s.Sub(r, 0, 16, 8), o.Sub(r, 0, 4, 8)
			binary.LittleEndian.PutUint32(ow.Row4(0, 0)[:], uint32(mc8H4S(sw.Row16(0, 0), c1, c2, c3, c4, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0)))
			binary.LittleEndian.PutUint32(ow.Row4(1, 0)[:], uint32(mc8H4S(sw.Row16(1, 0), c1, c2, c3, c4, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0)))
			binary.LittleEndian.PutUint32(ow.Row4(2, 0)[:], uint32(mc8H4S(sw.Row16(2, 0), c1, c2, c3, c4, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0)))
			binary.LittleEndian.PutUint32(ow.Row4(3, 0)[:], uint32(mc8H4S(sw.Row16(3, 0), c1, c2, c3, c4, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0)))
			binary.LittleEndian.PutUint32(ow.Row4(4, 0)[:], uint32(mc8H4S(sw.Row16(4, 0), c1, c2, c3, c4, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0)))
			binary.LittleEndian.PutUint32(ow.Row4(5, 0)[:], uint32(mc8H4S(sw.Row16(5, 0), c1, c2, c3, c4, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0)))
			binary.LittleEndian.PutUint32(ow.Row4(6, 0)[:], uint32(mc8H4S(sw.Row16(6, 0), c1, c2, c3, c4, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0)))
			binary.LittleEndian.PutUint32(ow.Row4(7, 0)[:], uint32(mc8H4S(sw.Row16(7, 0), c1, c2, c3, c4, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0)))
		}
	}
	if r < h {
		if r+4 <= h {
			if !s.Fits(r, 0, 16, 4) || !o.Fits(r, 0, 4, 4) {
				return
			}
			sw, ow := s.Sub(r, 0, 16, 4), o.Sub(r, 0, 4, 4)
			binary.LittleEndian.PutUint32(ow.Row4(0, 0)[:], uint32(mc8H4S(sw.Row16(0, 0), c1, c2, c3, c4, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0)))
			binary.LittleEndian.PutUint32(ow.Row4(1, 0)[:], uint32(mc8H4S(sw.Row16(1, 0), c1, c2, c3, c4, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0)))
			binary.LittleEndian.PutUint32(ow.Row4(2, 0)[:], uint32(mc8H4S(sw.Row16(2, 0), c1, c2, c3, c4, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0)))
			binary.LittleEndian.PutUint32(ow.Row4(3, 0)[:], uint32(mc8H4S(sw.Row16(3, 0), c1, c2, c3, c4, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0)))
			r += 4
		}
		if r+2 <= h {
			if !s.Fits(r, 0, 16, 2) || !o.Fits(r, 0, 4, 2) {
				return
			}
			sw, ow := s.Sub(r, 0, 16, 2), o.Sub(r, 0, 4, 2)
			binary.LittleEndian.PutUint32(ow.Row4(0, 0)[:], uint32(mc8H4S(sw.Row16(0, 0), c1, c2, c3, c4, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0)))
			binary.LittleEndian.PutUint32(ow.Row4(1, 0)[:], uint32(mc8H4S(sw.Row16(1, 0), c1, c2, c3, c4, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0)))
			r += 2
		}
		if r < h {
			if !s.Fits(r, 0, 16, 1) || !o.Fits(r, 0, 4, 1) {
				return
			}
			sw, ow := s.Sub(r, 0, 16, 1), o.Sub(r, 0, 4, 1)
			binary.LittleEndian.PutUint32(ow.Row4(0, 0)[:], uint32(mc8H4S(sw.Row16(0, 0), c1, c2, c3, c4, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0)))
		}
	}
}

// mc8PutV4 is the 1D vertical 4-tap put pass (taps 1..4: regular4, smooth4,
// bilinear) for w == 4 or w multiple of 8: skips rows 0 and h+4 and drops the
// zero outer taps.
func mc8PutV4(sg unsafeutil.Grid[uint8], w, h int, ph *mc8Phase, dg unsafeutil.Grid[uint8]) {
	c1 := archsimd.LoadInt16x8Array(&ph.h[1])
	c2 := archsimd.LoadInt16x8Array(&ph.h[2])
	c3 := archsimd.LoadInt16x8Array(&ph.h[3])
	c4 := archsimd.LoadInt16x8Array(&ph.h[4])
	rnd := archsimd.LoadInt16x8Array(&mc8Vec.rnd32)
	sh := archsimd.LoadInt16x8Array(&mc8Vec.sh6)
	rows := h + 4
	if w == 4 {
		if !sg.Fits(0, 0, 16, rows) || !dg.Fits(0, 0, 4, h) {
			return
		}
		s, d := sg.Sub(0, 0, 16, rows), dg.Sub(0, 0, 4, h)
		if !s.Fits(1, 0, 16, 3) {
			return
		}
		hd := s.Sub(1, 0, 16, 3)
		r1 := archsimd.LoadUint8x16Array(hd.Row16(0, 0)).ExtendLo8ToUint16().BitsToInt16()
		r2 := archsimd.LoadUint8x16Array(hd.Row16(1, 0)).ExtendLo8ToUint16().BitsToInt16()
		r3 := archsimd.LoadUint8x16Array(hd.Row16(2, 0)).ExtendLo8ToUint16().BitsToInt16()
		r := 0
		for ; r+4 <= h; r += 4 {
			if !s.Fits(r+4, 0, 16, 4) || !d.Fits(r, 0, 4, 4) {
				return
			}
			sw, dw := s.Sub(r+4, 0, 16, 4), d.Sub(r, 0, 4, 4)
			r4 := archsimd.LoadUint8x16Array(sw.Row16(0, 0)).ExtendLo8ToUint16().BitsToInt16()
			r5 := archsimd.LoadUint8x16Array(sw.Row16(1, 0)).ExtendLo8ToUint16().BitsToInt16()
			r6 := archsimd.LoadUint8x16Array(sw.Row16(2, 0)).ExtendLo8ToUint16().BitsToInt16()
			r7 := archsimd.LoadUint8x16Array(sw.Row16(3, 0)).ExtendLo8ToUint16().BitsToInt16()
			binary.LittleEndian.PutUint32(dw.Row4(0, 0)[:], uint32(r4.MulAdd(c4, r3.MulAdd(c3, r2.MulAdd(c2, r1.MulAdd(c1, rnd)))).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0)))
			binary.LittleEndian.PutUint32(dw.Row4(1, 0)[:], uint32(r5.MulAdd(c4, r4.MulAdd(c3, r3.MulAdd(c2, r2.MulAdd(c1, rnd)))).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0)))
			binary.LittleEndian.PutUint32(dw.Row4(2, 0)[:], uint32(r6.MulAdd(c4, r5.MulAdd(c3, r4.MulAdd(c2, r3.MulAdd(c1, rnd)))).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0)))
			binary.LittleEndian.PutUint32(dw.Row4(3, 0)[:], uint32(r7.MulAdd(c4, r6.MulAdd(c3, r5.MulAdd(c2, r4.MulAdd(c1, rnd)))).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0)))
			r1, r2, r3 = r5, r6, r7
		}
		for ; r < h; r++ {
			r4 := archsimd.LoadUint8x16Array(s.Row16(r+4, 0)).ExtendLo8ToUint16().BitsToInt16()
			v := r4.MulAdd(c4, r3.MulAdd(c3, r2.MulAdd(c2, r1.MulAdd(c1, rnd)))).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0)
			binary.LittleEndian.PutUint32(d.Row4(r, 0)[:], uint32(v))
			r1, r2, r3 = r2, r3, r4
		}
		return
	}
	for c := 0; c < w; c += 8 {
		if !sg.Fits(0, c, 16, rows) || !dg.Fits(0, c, 8, h) {
			return
		}
		s, d := sg.Sub(0, c, 16, rows), dg.Sub(0, c, 8, h)
		if !s.Fits(1, 0, 16, 3) {
			return
		}
		hd := s.Sub(1, 0, 16, 3)
		r1 := archsimd.LoadUint8x16Array(hd.Row16(0, 0)).ExtendLo8ToUint16().BitsToInt16()
		r2 := archsimd.LoadUint8x16Array(hd.Row16(1, 0)).ExtendLo8ToUint16().BitsToInt16()
		r3 := archsimd.LoadUint8x16Array(hd.Row16(2, 0)).ExtendLo8ToUint16().BitsToInt16()
		r := 0
		for ; r+4 <= h; r += 4 {
			if !s.Fits(r+4, 0, 16, 4) || !d.Fits(r, 0, 8, 4) {
				return
			}
			sw, dw := s.Sub(r+4, 0, 16, 4), d.Sub(r, 0, 8, 4)
			r4 := archsimd.LoadUint8x16Array(sw.Row16(0, 0)).ExtendLo8ToUint16().BitsToInt16()
			r5 := archsimd.LoadUint8x16Array(sw.Row16(1, 0)).ExtendLo8ToUint16().BitsToInt16()
			r6 := archsimd.LoadUint8x16Array(sw.Row16(2, 0)).ExtendLo8ToUint16().BitsToInt16()
			r7 := archsimd.LoadUint8x16Array(sw.Row16(3, 0)).ExtendLo8ToUint16().BitsToInt16()
			binary.LittleEndian.PutUint64(dw.Row8(0, 0)[:], r4.MulAdd(c4, r3.MulAdd(c3, r2.MulAdd(c2, r1.MulAdd(c1, rnd)))).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0))
			binary.LittleEndian.PutUint64(dw.Row8(1, 0)[:], r5.MulAdd(c4, r4.MulAdd(c3, r3.MulAdd(c2, r2.MulAdd(c1, rnd)))).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0))
			binary.LittleEndian.PutUint64(dw.Row8(2, 0)[:], r6.MulAdd(c4, r5.MulAdd(c3, r4.MulAdd(c2, r3.MulAdd(c1, rnd)))).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0))
			binary.LittleEndian.PutUint64(dw.Row8(3, 0)[:], r7.MulAdd(c4, r6.MulAdd(c3, r5.MulAdd(c2, r4.MulAdd(c1, rnd)))).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0))
			r1, r2, r3 = r5, r6, r7
		}
		for ; r < h; r++ {
			r4 := archsimd.LoadUint8x16Array(s.Row16(r+4, 0)).ExtendLo8ToUint16().BitsToInt16()
			v := r4.MulAdd(c4, r3.MulAdd(c3, r2.MulAdd(c2, r1.MulAdd(c1, rnd)))).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0)
			binary.LittleEndian.PutUint64(d.Row8(r, 0)[:], v)
			r1, r2, r3 = r2, r3, r4
		}
	}
}

// mc8PrepH4S4 is mc8PrepH6S4 when the horizontal filter has at most 4 non-zero
// taps (taps 1..4: regular4, smooth4, bilinear — the 99.9% case for w == 4).
func mc8PrepH4S4(s unsafeutil.Grid[uint8], ph *mc8Phase, o unsafeutil.Grid[uint16]) {
	c1 := archsimd.LoadInt16x8Array(&ph.h[1])
	c2 := archsimd.LoadInt16x8Array(&ph.h[2])
	c3 := archsimd.LoadInt16x8Array(&ph.h[3])
	c4 := archsimd.LoadInt16x8Array(&ph.h[4])
	rnd := archsimd.LoadInt16x8Array(&mc8Vec.prepRnd)
	sh := archsimd.LoadInt16x8Array(&mc8Vec.sh2)
	h := o.Height()
	r := 0
	for ; r+4 <= h; r += 4 {
		if !s.Fits(r, 0, 16, 4) || !o.Fits(r, 0, 4, 4) {
			return
		}
		sw, ow := s.Sub(r, 0, 16, 4), o.Sub(r, 0, 4, 4)
		mc8Store4U16(ow.Row4(0, 0), mc8H4S(sw.Row16(0, 0), c1, c2, c3, c4, rnd).ToBits().Shift(sh).ReshapeToUint64s().GetElem(0))
		mc8Store4U16(ow.Row4(1, 0), mc8H4S(sw.Row16(1, 0), c1, c2, c3, c4, rnd).ToBits().Shift(sh).ReshapeToUint64s().GetElem(0))
		mc8Store4U16(ow.Row4(2, 0), mc8H4S(sw.Row16(2, 0), c1, c2, c3, c4, rnd).ToBits().Shift(sh).ReshapeToUint64s().GetElem(0))
		mc8Store4U16(ow.Row4(3, 0), mc8H4S(sw.Row16(3, 0), c1, c2, c3, c4, rnd).ToBits().Shift(sh).ReshapeToUint64s().GetElem(0))
	}
	if r < h {
		if !s.Fits(r, 0, 16, h-r) || !o.Fits(r, 0, 4, h-r) {
			return
		}
		sw, ow := s.Sub(r, 0, 16, h-r), o.Sub(r, 0, 4, h-r)
		for k := 0; k < h-r; k++ {
			mc8Store4U16(ow.Row4(k, 0), mc8H4S(sw.Row16(k, 0), c1, c2, c3, c4, rnd).ToBits().Shift(sh).ReshapeToUint64s().GetElem(0))
		}
	}
}

// mc8PrepV4 is the 1D vertical 4-tap prep pass (taps 1..4) for w == 4 or w
// multiple of 8: skips rows 0 and h+4 and drops the zero outer taps.
func mc8PrepV4(sg unsafeutil.Grid[uint8], w, h int, ph *mc8Phase, og unsafeutil.Grid[uint16]) {
	c1 := archsimd.LoadInt16x8Array(&ph.h[1])
	c2 := archsimd.LoadInt16x8Array(&ph.h[2])
	c3 := archsimd.LoadInt16x8Array(&ph.h[3])
	c4 := archsimd.LoadInt16x8Array(&ph.h[4])
	rnd := archsimd.LoadInt16x8Array(&mc8Vec.prepRnd)
	sh := archsimd.LoadInt16x8Array(&mc8Vec.sh2)
	rows := h + 4
	if w == 4 {
		if !sg.Fits(0, 0, 16, rows) || !og.Fits(0, 0, 4, h) {
			return
		}
		s, o := sg.Sub(0, 0, 16, rows), og.Sub(0, 0, 4, h)
		if !s.Fits(1, 0, 16, 3) {
			return
		}
		hd := s.Sub(1, 0, 16, 3)
		r1 := archsimd.LoadUint8x16Array(hd.Row16(0, 0)).ExtendLo8ToUint16().BitsToInt16()
		r2 := archsimd.LoadUint8x16Array(hd.Row16(1, 0)).ExtendLo8ToUint16().BitsToInt16()
		r3 := archsimd.LoadUint8x16Array(hd.Row16(2, 0)).ExtendLo8ToUint16().BitsToInt16()
		r := 0
		for ; r+4 <= h; r += 4 {
			if !s.Fits(r+4, 0, 16, 4) || !o.Fits(r, 0, 4, 4) {
				return
			}
			sw, ow := s.Sub(r+4, 0, 16, 4), o.Sub(r, 0, 4, 4)
			r4 := archsimd.LoadUint8x16Array(sw.Row16(0, 0)).ExtendLo8ToUint16().BitsToInt16()
			r5 := archsimd.LoadUint8x16Array(sw.Row16(1, 0)).ExtendLo8ToUint16().BitsToInt16()
			r6 := archsimd.LoadUint8x16Array(sw.Row16(2, 0)).ExtendLo8ToUint16().BitsToInt16()
			r7 := archsimd.LoadUint8x16Array(sw.Row16(3, 0)).ExtendLo8ToUint16().BitsToInt16()
			mc8Store4U16(ow.Row4(0, 0), r4.MulAdd(c4, r3.MulAdd(c3, r2.MulAdd(c2, r1.MulAdd(c1, rnd)))).ToBits().Shift(sh).ReshapeToUint64s().GetElem(0))
			mc8Store4U16(ow.Row4(1, 0), r5.MulAdd(c4, r4.MulAdd(c3, r3.MulAdd(c2, r2.MulAdd(c1, rnd)))).ToBits().Shift(sh).ReshapeToUint64s().GetElem(0))
			mc8Store4U16(ow.Row4(2, 0), r6.MulAdd(c4, r5.MulAdd(c3, r4.MulAdd(c2, r3.MulAdd(c1, rnd)))).ToBits().Shift(sh).ReshapeToUint64s().GetElem(0))
			mc8Store4U16(ow.Row4(3, 0), r7.MulAdd(c4, r6.MulAdd(c3, r5.MulAdd(c2, r4.MulAdd(c1, rnd)))).ToBits().Shift(sh).ReshapeToUint64s().GetElem(0))
			r1, r2, r3 = r5, r6, r7
		}
		for ; r < h; r++ {
			r4 := archsimd.LoadUint8x16Array(s.Row16(r+4, 0)).ExtendLo8ToUint16().BitsToInt16()
			mc8Store4U16(o.Row4(r, 0), r4.MulAdd(c4, r3.MulAdd(c3, r2.MulAdd(c2, r1.MulAdd(c1, rnd)))).ToBits().Shift(sh).ReshapeToUint64s().GetElem(0))
			r1, r2, r3 = r2, r3, r4
		}
		return
	}
	for c := 0; c < w; c += 8 {
		if !sg.Fits(0, c, 16, rows) || !og.Fits(0, c, 8, h) {
			return
		}
		s, o := sg.Sub(0, c, 16, rows), og.Sub(0, c, 8, h)
		if !s.Fits(1, 0, 16, 3) {
			return
		}
		hd := s.Sub(1, 0, 16, 3)
		r1 := archsimd.LoadUint8x16Array(hd.Row16(0, 0)).ExtendLo8ToUint16().BitsToInt16()
		r2 := archsimd.LoadUint8x16Array(hd.Row16(1, 0)).ExtendLo8ToUint16().BitsToInt16()
		r3 := archsimd.LoadUint8x16Array(hd.Row16(2, 0)).ExtendLo8ToUint16().BitsToInt16()
		r := 0
		for ; r+4 <= h; r += 4 {
			if !s.Fits(r+4, 0, 16, 4) || !o.Fits(r, 0, 8, 4) {
				return
			}
			sw, ow := s.Sub(r+4, 0, 16, 4), o.Sub(r, 0, 8, 4)
			r4 := archsimd.LoadUint8x16Array(sw.Row16(0, 0)).ExtendLo8ToUint16().BitsToInt16()
			r5 := archsimd.LoadUint8x16Array(sw.Row16(1, 0)).ExtendLo8ToUint16().BitsToInt16()
			r6 := archsimd.LoadUint8x16Array(sw.Row16(2, 0)).ExtendLo8ToUint16().BitsToInt16()
			r7 := archsimd.LoadUint8x16Array(sw.Row16(3, 0)).ExtendLo8ToUint16().BitsToInt16()
			r4.MulAdd(c4, r3.MulAdd(c3, r2.MulAdd(c2, r1.MulAdd(c1, rnd)))).ToBits().Shift(sh).StoreArray(ow.Row8(0, 0))
			r5.MulAdd(c4, r4.MulAdd(c3, r3.MulAdd(c2, r2.MulAdd(c1, rnd)))).ToBits().Shift(sh).StoreArray(ow.Row8(1, 0))
			r6.MulAdd(c4, r5.MulAdd(c3, r4.MulAdd(c2, r3.MulAdd(c1, rnd)))).ToBits().Shift(sh).StoreArray(ow.Row8(2, 0))
			r7.MulAdd(c4, r6.MulAdd(c3, r5.MulAdd(c2, r4.MulAdd(c1, rnd)))).ToBits().Shift(sh).StoreArray(ow.Row8(3, 0))
			r1, r2, r3 = r5, r6, r7
		}
		for ; r < h; r++ {
			r4 := archsimd.LoadUint8x16Array(s.Row16(r+4, 0)).ExtendLo8ToUint16().BitsToInt16()
			r4.MulAdd(c4, r3.MulAdd(c3, r2.MulAdd(c2, r1.MulAdd(c1, rnd)))).ToBits().Shift(sh).StoreArray(o.Row8(r, 0))
			r1, r2, r3 = r2, r3, r4
		}
	}
}

// mc8PrepHV4x4W4 is the fused width-4 compound 2D pass when both horizontal
// and vertical filters have at most 4 non-zero taps (taps 1..4).
func mc8PrepHV4x4W4(sg unsafeutil.Grid[uint8], h int, phx, phy *mc8Phase, out []uint16) {
	hc1 := archsimd.LoadInt16x8Array(&phx.h[1])
	hc2 := archsimd.LoadInt16x8Array(&phx.h[2])
	hc3 := archsimd.LoadInt16x8Array(&phx.h[3])
	hc4 := archsimd.LoadInt16x8Array(&phx.h[4])
	rnd := archsimd.BroadcastInt16x8(2)
	sh := archsimd.LoadInt16x8Array(&mc8Vec.sh2)
	vc1 := archsimd.LoadInt32x4Array(&phy.p[1])
	vc2 := archsimd.LoadInt32x4Array(&phy.p[2])
	vc3 := archsimd.LoadInt32x4Array(&phy.p[3])
	vc4 := archsimd.LoadInt32x4Array(&phy.p[4])
	vrnd := archsimd.BroadcastInt32x4(mc8PrepV6Rnd)
	rows := h + 4
	if !sg.Fits(0, 0, 16, rows) || len(out) < 4*h {
		return
	}
	s := sg.Sub(0, 0, 16, rows)
	ot := out[: 4*h : 4*h]
	if !s.Fits(1, 0, 16, 3) {
		return
	}
	hd := s.Sub(1, 0, 16, 3)
	x1 := mc8H4S(hd.Row16(0, 0), hc1, hc2, hc3, hc4, rnd).Shift(sh).ExtendLo4ToInt32()
	x2 := mc8H4S(hd.Row16(1, 0), hc1, hc2, hc3, hc4, rnd).Shift(sh).ExtendLo4ToInt32()
	x3 := mc8H4S(hd.Row16(2, 0), hc1, hc2, hc3, hc4, rnd).Shift(sh).ExtendLo4ToInt32()
	r := 0
	for ; r+4 <= h; r += 4 {
		if !s.Fits(r+4, 0, 16, 4) {
			return
		}
		sw := s.Sub(r+4, 0, 16, 4)
		x4 := mc8H4S(sw.Row16(0, 0), hc1, hc2, hc3, hc4, rnd).Shift(sh).ExtendLo4ToInt32()
		x5 := mc8H4S(sw.Row16(1, 0), hc1, hc2, hc3, hc4, rnd).Shift(sh).ExtendLo4ToInt32()
		a0 := x4.MulAdd(vc4, x3.MulAdd(vc3, x2.MulAdd(vc2, x1.MulAdd(vc1, vrnd))))
		a1 := x5.MulAdd(vc4, x4.MulAdd(vc3, x3.MulAdd(vc2, x2.MulAdd(vc1, vrnd))))
		a0.ToBits().ReshapeToUint16s().ConcatOdd(a1.ToBits().ReshapeToUint16s()).StoreArray((*[8]uint16)(ot[4*r : 4*r+8]))
		x6 := mc8H4S(sw.Row16(2, 0), hc1, hc2, hc3, hc4, rnd).Shift(sh).ExtendLo4ToInt32()
		x7 := mc8H4S(sw.Row16(3, 0), hc1, hc2, hc3, hc4, rnd).Shift(sh).ExtendLo4ToInt32()
		a2 := x6.MulAdd(vc4, x5.MulAdd(vc3, x4.MulAdd(vc2, x3.MulAdd(vc1, vrnd))))
		a3 := x7.MulAdd(vc4, x6.MulAdd(vc3, x5.MulAdd(vc2, x4.MulAdd(vc1, vrnd))))
		a2.ToBits().ReshapeToUint16s().ConcatOdd(a3.ToBits().ReshapeToUint16s()).StoreArray((*[8]uint16)(ot[4*r+8 : 4*r+16]))
		x1, x2, x3 = x5, x6, x7
	}
	for ; r+2 <= h; r += 2 {
		x4 := mc8H4S(s.Row16(r+4, 0), hc1, hc2, hc3, hc4, rnd).Shift(sh).ExtendLo4ToInt32()
		x5 := mc8H4S(s.Row16(r+5, 0), hc1, hc2, hc3, hc4, rnd).Shift(sh).ExtendLo4ToInt32()
		a0 := x4.MulAdd(vc4, x3.MulAdd(vc3, x2.MulAdd(vc2, x1.MulAdd(vc1, vrnd))))
		a1 := x5.MulAdd(vc4, x4.MulAdd(vc3, x3.MulAdd(vc2, x2.MulAdd(vc1, vrnd))))
		a0.ToBits().ReshapeToUint16s().ConcatOdd(a1.ToBits().ReshapeToUint16s()).StoreArray((*[8]uint16)(ot[4*r : 4*r+8]))
		x1, x2, x3 = x3, x4, x5
	}
	if r < h {
		x4 := mc8H4S(s.Row16(r+4, 0), hc1, hc2, hc3, hc4, rnd).Shift(sh).ExtendLo4ToInt32()
		a0 := x4.MulAdd(vc4, x3.MulAdd(vc3, x2.MulAdd(vc2, x1.MulAdd(vc1, vrnd))))
		v := a0.ToBits().ReshapeToUint16s().ConcatOdd(a0.ToBits().ReshapeToUint16s()).ReshapeToUint64s().GetElem(0)
		mc8Store4U16((*[4]uint16)(ot[4*r : 4*r+4]), v)
	}
}

// mc8PrepHV4W4 is the fused width-4 compound 2D pass when the horizontal filter
// has at most 4 non-zero taps (taps 1..4) and the vertical filter has 6 taps.
func mc8PrepHV4W4(sg unsafeutil.Grid[uint8], h int, phx, phy *mc8Phase, out []uint16) {
	hc1 := archsimd.LoadInt16x8Array(&phx.h[1])
	hc2 := archsimd.LoadInt16x8Array(&phx.h[2])
	hc3 := archsimd.LoadInt16x8Array(&phx.h[3])
	hc4 := archsimd.LoadInt16x8Array(&phx.h[4])
	rnd := archsimd.BroadcastInt16x8(2)
	sh := archsimd.LoadInt16x8Array(&mc8Vec.sh2)
	vc0 := archsimd.LoadInt32x4Array(&phy.p[0])
	vc1 := archsimd.LoadInt32x4Array(&phy.p[1])
	vc2 := archsimd.LoadInt32x4Array(&phy.p[2])
	vc3 := archsimd.LoadInt32x4Array(&phy.p[3])
	vc4 := archsimd.LoadInt32x4Array(&phy.p[4])
	vc5 := archsimd.LoadInt32x4Array(&phy.p[5])
	vrnd := archsimd.BroadcastInt32x4(mc8PrepV6Rnd)
	rows := h + 5
	if !sg.Fits(0, 0, 16, rows) || len(out) < 4*h {
		return
	}
	s := sg.Sub(0, 0, 16, rows)
	ot := out[: 4*h : 4*h]
	if !s.Fits(0, 0, 16, 5) {
		return
	}
	hd := s.Sub(0, 0, 16, 5)
	x0 := mc8H4S(hd.Row16(0, 0), hc1, hc2, hc3, hc4, rnd).Shift(sh).ExtendLo4ToInt32()
	x1 := mc8H4S(hd.Row16(1, 0), hc1, hc2, hc3, hc4, rnd).Shift(sh).ExtendLo4ToInt32()
	x2 := mc8H4S(hd.Row16(2, 0), hc1, hc2, hc3, hc4, rnd).Shift(sh).ExtendLo4ToInt32()
	x3 := mc8H4S(hd.Row16(3, 0), hc1, hc2, hc3, hc4, rnd).Shift(sh).ExtendLo4ToInt32()
	x4 := mc8H4S(hd.Row16(4, 0), hc1, hc2, hc3, hc4, rnd).Shift(sh).ExtendLo4ToInt32()
	r := 0
	for ; r+4 <= h; r += 4 {
		if !s.Fits(r+5, 0, 16, 4) {
			return
		}
		sw := s.Sub(r+5, 0, 16, 4)
		x5 := mc8H4S(sw.Row16(0, 0), hc1, hc2, hc3, hc4, rnd).Shift(sh).ExtendLo4ToInt32()
		x6 := mc8H4S(sw.Row16(1, 0), hc1, hc2, hc3, hc4, rnd).Shift(sh).ExtendLo4ToInt32()
		a0 := x5.MulAdd(vc5, x4.MulAdd(vc4, x3.MulAdd(vc3, x2.MulAdd(vc2, x1.MulAdd(vc1, x0.MulAdd(vc0, vrnd))))))
		a1 := x6.MulAdd(vc5, x5.MulAdd(vc4, x4.MulAdd(vc3, x3.MulAdd(vc2, x2.MulAdd(vc1, x1.MulAdd(vc0, vrnd))))))
		a0.ToBits().ReshapeToUint16s().ConcatOdd(a1.ToBits().ReshapeToUint16s()).StoreArray((*[8]uint16)(ot[4*r : 4*r+8]))
		x7 := mc8H4S(sw.Row16(2, 0), hc1, hc2, hc3, hc4, rnd).Shift(sh).ExtendLo4ToInt32()
		x8 := mc8H4S(sw.Row16(3, 0), hc1, hc2, hc3, hc4, rnd).Shift(sh).ExtendLo4ToInt32()
		a2 := x7.MulAdd(vc5, x6.MulAdd(vc4, x5.MulAdd(vc3, x4.MulAdd(vc2, x3.MulAdd(vc1, x2.MulAdd(vc0, vrnd))))))
		a3 := x8.MulAdd(vc5, x7.MulAdd(vc4, x6.MulAdd(vc3, x5.MulAdd(vc2, x4.MulAdd(vc1, x3.MulAdd(vc0, vrnd))))))
		a2.ToBits().ReshapeToUint16s().ConcatOdd(a3.ToBits().ReshapeToUint16s()).StoreArray((*[8]uint16)(ot[4*r+8 : 4*r+16]))
		x0, x1, x2, x3, x4 = x4, x5, x6, x7, x8
	}
	for ; r+2 <= h; r += 2 {
		x5 := mc8H4S(s.Row16(r+5, 0), hc1, hc2, hc3, hc4, rnd).Shift(sh).ExtendLo4ToInt32()
		x6 := mc8H4S(s.Row16(r+6, 0), hc1, hc2, hc3, hc4, rnd).Shift(sh).ExtendLo4ToInt32()
		a0 := x5.MulAdd(vc5, x4.MulAdd(vc4, x3.MulAdd(vc3, x2.MulAdd(vc2, x1.MulAdd(vc1, x0.MulAdd(vc0, vrnd))))))
		a1 := x6.MulAdd(vc5, x5.MulAdd(vc4, x4.MulAdd(vc3, x3.MulAdd(vc2, x2.MulAdd(vc1, x1.MulAdd(vc0, vrnd))))))
		a0.ToBits().ReshapeToUint16s().ConcatOdd(a1.ToBits().ReshapeToUint16s()).StoreArray((*[8]uint16)(ot[4*r : 4*r+8]))
		x0, x1, x2, x3, x4 = x2, x3, x4, x5, x6
	}
	if r < h {
		x5 := mc8H4S(s.Row16(r+5, 0), hc1, hc2, hc3, hc4, rnd).Shift(sh).ExtendLo4ToInt32()
		a0 := x5.MulAdd(vc5, x4.MulAdd(vc4, x3.MulAdd(vc3, x2.MulAdd(vc2, x1.MulAdd(vc1, x0.MulAdd(vc0, vrnd))))))
		v := a0.ToBits().ReshapeToUint16s().ConcatOdd(a0.ToBits().ReshapeToUint16s()).ReshapeToUint64s().GetElem(0)
		mc8Store4U16((*[4]uint16)(ot[4*r : 4*r+4]), v)
	}
}

// mc8PrepHV2 is the fused 2-tap bilinear compound 2D pass over 8-column strips
// (taps 2 and 3 of the 6-tap window on both axes), with no intermediate buffer.
func mc8PrepHV2(sg unsafeutil.Grid[uint8], w, h int, phx, phy *mc8Phase, og unsafeutil.Grid[uint16]) {
	hc2 := archsimd.LoadInt16x8Array(&phx.h[2])
	hc3 := archsimd.LoadInt16x8Array(&phx.h[3])
	rnd := archsimd.BroadcastInt16x8(2)
	sh := archsimd.LoadInt16x8Array(&mc8Vec.sh2)
	vc2 := archsimd.LoadInt32x4Array(&phy.p[2])
	vc3 := archsimd.LoadInt32x4Array(&phy.p[3])
	vrnd := archsimd.BroadcastInt32x4(mc8PrepV6Rnd)
	rows := h + 4
	for c := 0; c < w; c += 8 {
		if !sg.Fits(0, c, 16, rows) || !og.Fits(0, c, 8, h) {
			return
		}
		s, o := sg.Sub(0, c, 16, rows), og.Sub(0, c, 8, h)
		a := archsimd.LoadUint8x16Array(s.Row16(2, 0))
		lo, hi := a.ExtendLo8ToUint16().ReshapeToUint8s(), a.HiToLo().ExtendLo8ToUint16().ReshapeToUint8s()
		h0 := hi.ConcatShiftBytesRight(lo, 6).ReshapeToUint16s().BitsToInt16().MulAdd(hc3, hi.ConcatShiftBytesRight(lo, 4).ReshapeToUint16s().BitsToInt16().MulAdd(hc2, rnd)).Shift(sh)
		x0, y0 := h0.ExtendLo4ToInt32(), h0.HiToLo().ExtendLo4ToInt32()
		for r := 0; r < h; r++ {
			a = archsimd.LoadUint8x16Array(s.Row16(r+3, 0))
			lo, hi = a.ExtendLo8ToUint16().ReshapeToUint8s(), a.HiToLo().ExtendLo8ToUint16().ReshapeToUint8s()
			h1 := hi.ConcatShiftBytesRight(lo, 6).ReshapeToUint16s().BitsToInt16().MulAdd(hc3, hi.ConcatShiftBytesRight(lo, 4).ReshapeToUint16s().BitsToInt16().MulAdd(hc2, rnd)).Shift(sh)
			x1, y1 := h1.ExtendLo4ToInt32(), h1.HiToLo().ExtendLo4ToInt32()
			a0 := x1.MulAdd(vc3, x0.MulAdd(vc2, vrnd))
			b0 := y1.MulAdd(vc3, y0.MulAdd(vc2, vrnd))
			a0.ToBits().ReshapeToUint16s().ConcatOdd(b0.ToBits().ReshapeToUint16s()).StoreArray(o.Row8(r, 0))
			x0, y0 = x1, y1
		}
	}
}

// mc8PrepHV6x4 is the fused compound 2D 6-tap H x 4-tap V pass over 8-column
// strips directly in registers (23 NEON regs, zero spills, no intermediate buffer).
func mc8PrepHV6x4(sg unsafeutil.Grid[uint8], w, h int, phx, phy *mc8Phase, og unsafeutil.Grid[uint16]) {
	hc0 := archsimd.LoadInt16x8Array(&phx.h[0])
	hc1 := archsimd.LoadInt16x8Array(&phx.h[1])
	hc2 := archsimd.LoadInt16x8Array(&phx.h[2])
	hc3 := archsimd.LoadInt16x8Array(&phx.h[3])
	hc4 := archsimd.LoadInt16x8Array(&phx.h[4])
	hc5 := archsimd.LoadInt16x8Array(&phx.h[5])
	rnd := archsimd.BroadcastInt16x8(2)
	sh := archsimd.LoadInt16x8Array(&mc8Vec.sh2)
	vc1 := archsimd.LoadInt32x4Array(&phy.p[1])
	vc2 := archsimd.LoadInt32x4Array(&phy.p[2])
	vc3 := archsimd.LoadInt32x4Array(&phy.p[3])
	vc4 := archsimd.LoadInt32x4Array(&phy.p[4])
	vrnd := archsimd.BroadcastInt32x4(mc8PrepV6Rnd)
	rows := h + 4
	for c := 0; c < w; c += 8 {
		if !sg.Fits(0, c, 16, rows) || !og.Fits(0, c, 8, h) {
			return
		}
		s, o := sg.Sub(0, c, 16, rows), og.Sub(0, c, 8, h)
		if !s.Fits(1, 0, 16, 3) {
			return
		}
		hd := s.Sub(1, 0, 16, 3)
		a := mc8H6S(hd.Row16(0, 0), hc0, hc1, hc2, hc3, hc4, hc5, rnd).Shift(sh)
		x1, y1 := a.ExtendLo4ToInt32(), a.HiToLo().ExtendLo4ToInt32()
		a = mc8H6S(hd.Row16(1, 0), hc0, hc1, hc2, hc3, hc4, hc5, rnd).Shift(sh)
		x2, y2 := a.ExtendLo4ToInt32(), a.HiToLo().ExtendLo4ToInt32()
		a = mc8H6S(hd.Row16(2, 0), hc0, hc1, hc2, hc3, hc4, hc5, rnd).Shift(sh)
		x3, y3 := a.ExtendLo4ToInt32(), a.HiToLo().ExtendLo4ToInt32()
		r := 0
		for ; r+4 <= h; r += 4 {
			if !s.Fits(r+4, 0, 16, 4) || !o.Fits(r, 0, 8, 4) {
				return
			}
			sw, ow := s.Sub(r+4, 0, 16, 4), o.Sub(r, 0, 8, 4)
			a4 := mc8H6S(sw.Row16(0, 0), hc0, hc1, hc2, hc3, hc4, hc5, rnd).Shift(sh)
			x4, y4 := a4.ExtendLo4ToInt32(), a4.HiToLo().ExtendLo4ToInt32()
			a5 := mc8H6S(sw.Row16(1, 0), hc0, hc1, hc2, hc3, hc4, hc5, rnd).Shift(sh)
			x5, y5 := a5.ExtendLo4ToInt32(), a5.HiToLo().ExtendLo4ToInt32()
			p0 := x4.MulAdd(vc4, x3.MulAdd(vc3, x2.MulAdd(vc2, x1.MulAdd(vc1, vrnd))))
			q0 := y4.MulAdd(vc4, y3.MulAdd(vc3, y2.MulAdd(vc2, y1.MulAdd(vc1, vrnd))))
			p1 := x5.MulAdd(vc4, x4.MulAdd(vc3, x3.MulAdd(vc2, x2.MulAdd(vc1, vrnd))))
			q1 := y5.MulAdd(vc4, y4.MulAdd(vc3, y3.MulAdd(vc2, y2.MulAdd(vc1, vrnd))))
			p0.ToBits().ReshapeToUint16s().ConcatOdd(q0.ToBits().ReshapeToUint16s()).StoreArray(ow.Row8(0, 0))
			p1.ToBits().ReshapeToUint16s().ConcatOdd(q1.ToBits().ReshapeToUint16s()).StoreArray(ow.Row8(1, 0))
			a6 := mc8H6S(sw.Row16(2, 0), hc0, hc1, hc2, hc3, hc4, hc5, rnd).Shift(sh)
			x6, y6 := a6.ExtendLo4ToInt32(), a6.HiToLo().ExtendLo4ToInt32()
			a7 := mc8H6S(sw.Row16(3, 0), hc0, hc1, hc2, hc3, hc4, hc5, rnd).Shift(sh)
			x7, y7 := a7.ExtendLo4ToInt32(), a7.HiToLo().ExtendLo4ToInt32()
			p2 := x6.MulAdd(vc4, x5.MulAdd(vc3, x4.MulAdd(vc2, x3.MulAdd(vc1, vrnd))))
			q2 := y6.MulAdd(vc4, y5.MulAdd(vc3, y4.MulAdd(vc2, y3.MulAdd(vc1, vrnd))))
			p3 := x7.MulAdd(vc4, x6.MulAdd(vc3, x5.MulAdd(vc2, x4.MulAdd(vc1, vrnd))))
			q3 := y7.MulAdd(vc4, y6.MulAdd(vc3, y5.MulAdd(vc2, y4.MulAdd(vc1, vrnd))))
			p2.ToBits().ReshapeToUint16s().ConcatOdd(q2.ToBits().ReshapeToUint16s()).StoreArray(ow.Row8(2, 0))
			p3.ToBits().ReshapeToUint16s().ConcatOdd(q3.ToBits().ReshapeToUint16s()).StoreArray(ow.Row8(3, 0))
			x1, y1, x2, y2, x3, y3 = x5, y5, x6, y6, x7, y7
		}
		for ; r < h; r++ {
			a4 := mc8H6S(s.Row16(r+4, 0), hc0, hc1, hc2, hc3, hc4, hc5, rnd).Shift(sh)
			x4, y4 := a4.ExtendLo4ToInt32(), a4.HiToLo().ExtendLo4ToInt32()
			p0 := x4.MulAdd(vc4, x3.MulAdd(vc3, x2.MulAdd(vc2, x1.MulAdd(vc1, vrnd))))
			q0 := y4.MulAdd(vc4, y3.MulAdd(vc3, y2.MulAdd(vc2, y1.MulAdd(vc1, vrnd))))
			p0.ToBits().ReshapeToUint16s().ConcatOdd(q0.ToBits().ReshapeToUint16s()).StoreArray(o.Row8(r, 0))
			x1, y1, x2, y2, x3, y3 = x2, y2, x3, y3, x4, y4
		}
	}
}

// mc8PutHV4x4W4 is mc8PutHV4W4 when both horizontal and vertical filters have
// at most 4 non-zero taps (taps 1..4): skips rows 0 and h+4 and drops the zero
// outer vertical taps.
func mc8PutHV4x4W4(sg unsafeutil.Grid[uint8], h int, phx, phy *mc8Phase, dg unsafeutil.Grid[uint8]) {
	hc1 := archsimd.LoadInt16x8Array(&phx.h[1])
	hc2 := archsimd.LoadInt16x8Array(&phx.h[2])
	hc3 := archsimd.LoadInt16x8Array(&phx.h[3])
	hc4 := archsimd.LoadInt16x8Array(&phx.h[4])
	rnd := archsimd.BroadcastInt16x8(mc8PutH6ToI32Rnd)
	sh := archsimd.LoadInt16x8Array(&mc8Vec.sh2)
	vc1 := archsimd.LoadInt32x4Array(&phy.v[1])
	vc2 := archsimd.LoadInt32x4Array(&phy.v[2])
	vc3 := archsimd.LoadInt32x4Array(&phy.v[3])
	vc4 := archsimd.LoadInt32x4Array(&phy.v[4])
	rows := h + 4
	if !sg.Fits(0, 0, 16, rows) || !dg.Fits(0, 0, 4, h) {
		return
	}
	s, d := sg.Sub(0, 0, 16, rows), dg.Sub(0, 0, 4, h)
	if !s.Fits(1, 0, 16, 3) {
		return
	}
	hd := s.Sub(1, 0, 16, 3)
	x1 := mc8H4S(hd.Row16(0, 0), hc1, hc2, hc3, hc4, rnd).Shift(sh).ExtendLo4ToInt32()
	x2 := mc8H4S(hd.Row16(1, 0), hc1, hc2, hc3, hc4, rnd).Shift(sh).ExtendLo4ToInt32()
	x3 := mc8H4S(hd.Row16(2, 0), hc1, hc2, hc3, hc4, rnd).Shift(sh).ExtendLo4ToInt32()
	r := 0
	for ; r+4 <= h; r += 4 {
		if !s.Fits(r+4, 0, 16, 4) || !d.Fits(r, 0, 4, 4) {
			return
		}
		sw, dw := s.Sub(r+4, 0, 16, 4), d.Sub(r, 0, 4, 4)
		x4 := mc8H4S(sw.Row16(0, 0), hc1, hc2, hc3, hc4, rnd).Shift(sh).ExtendLo4ToInt32()
		x5 := mc8H4S(sw.Row16(1, 0), hc1, hc2, hc3, hc4, rnd).Shift(sh).ExtendLo4ToInt32()
		a0 := x4.MulAdd(vc4, x3.MulAdd(vc3, x2.MulAdd(vc2, x1.Mul(vc1))))
		a1 := x5.MulAdd(vc4, x4.MulAdd(vc3, x3.MulAdd(vc2, x2.Mul(vc1))))
		v0 := a0.ToBits().ReshapeToUint16s().ConcatOdd(a1.ToBits().ReshapeToUint16s()).BitsToInt16().SaturateToUint8().ReshapeToUint32s()
		binary.LittleEndian.PutUint32(dw.Row4(0, 0)[:], v0.GetElem(0))
		binary.LittleEndian.PutUint32(dw.Row4(1, 0)[:], v0.GetElem(1))
		x6 := mc8H4S(sw.Row16(2, 0), hc1, hc2, hc3, hc4, rnd).Shift(sh).ExtendLo4ToInt32()
		x7 := mc8H4S(sw.Row16(3, 0), hc1, hc2, hc3, hc4, rnd).Shift(sh).ExtendLo4ToInt32()
		a2 := x6.MulAdd(vc4, x5.MulAdd(vc3, x4.MulAdd(vc2, x3.Mul(vc1))))
		a3 := x7.MulAdd(vc4, x6.MulAdd(vc3, x5.MulAdd(vc2, x4.Mul(vc1))))
		v1 := a2.ToBits().ReshapeToUint16s().ConcatOdd(a3.ToBits().ReshapeToUint16s()).BitsToInt16().SaturateToUint8().ReshapeToUint32s()
		binary.LittleEndian.PutUint32(dw.Row4(2, 0)[:], v1.GetElem(0))
		binary.LittleEndian.PutUint32(dw.Row4(3, 0)[:], v1.GetElem(1))
		x1, x2, x3 = x5, x6, x7
	}
	for ; r+2 <= h; r += 2 {
		x4 := mc8H4S(s.Row16(r+4, 0), hc1, hc2, hc3, hc4, rnd).Shift(sh).ExtendLo4ToInt32()
		x5 := mc8H4S(s.Row16(r+5, 0), hc1, hc2, hc3, hc4, rnd).Shift(sh).ExtendLo4ToInt32()
		a0 := x4.MulAdd(vc4, x3.MulAdd(vc3, x2.MulAdd(vc2, x1.Mul(vc1))))
		a1 := x5.MulAdd(vc4, x4.MulAdd(vc3, x3.MulAdd(vc2, x2.Mul(vc1))))
		v := a0.ToBits().ReshapeToUint16s().ConcatOdd(a1.ToBits().ReshapeToUint16s()).BitsToInt16().SaturateToUint8().ReshapeToUint32s()
		binary.LittleEndian.PutUint32(d.Row4(r, 0)[:], v.GetElem(0))
		binary.LittleEndian.PutUint32(d.Row4(r+1, 0)[:], v.GetElem(1))
		x1, x2, x3 = x3, x4, x5
	}
	if r < h {
		x4 := mc8H4S(s.Row16(r+4, 0), hc1, hc2, hc3, hc4, rnd).Shift(sh).ExtendLo4ToInt32()
		a0 := x4.MulAdd(vc4, x3.MulAdd(vc3, x2.MulAdd(vc2, x1.Mul(vc1))))
		v := a0.ToBits().ReshapeToUint16s().ConcatOdd(a0.ToBits().ReshapeToUint16s()).BitsToInt16().SaturateToUint8().ReshapeToUint32s()
		binary.LittleEndian.PutUint32(d.Row4(r, 0)[:], v.GetElem(0))
	}
}

// mc8PutHV4W4 is the fused width-4 put 2D pass when the horizontal filter has
// at most 4 non-zero taps (regular4, smooth4, bilinear). All state stays in
// NEON registers with no intermediate buffer.
func mc8PutHV4W4(sg unsafeutil.Grid[uint8], h int, phx, phy *mc8Phase, dg unsafeutil.Grid[uint8]) {
	hc1 := archsimd.LoadInt16x8Array(&phx.h[1])
	hc2 := archsimd.LoadInt16x8Array(&phx.h[2])
	hc3 := archsimd.LoadInt16x8Array(&phx.h[3])
	hc4 := archsimd.LoadInt16x8Array(&phx.h[4])
	rnd := archsimd.BroadcastInt16x8(mc8PutH6ToI32Rnd)
	sh := archsimd.LoadInt16x8Array(&mc8Vec.sh2)
	vc0 := archsimd.LoadInt32x4Array(&phy.v[0])
	vc1 := archsimd.LoadInt32x4Array(&phy.v[1])
	vc2 := archsimd.LoadInt32x4Array(&phy.v[2])
	vc3 := archsimd.LoadInt32x4Array(&phy.v[3])
	vc4 := archsimd.LoadInt32x4Array(&phy.v[4])
	vc5 := archsimd.LoadInt32x4Array(&phy.v[5])
	rows := h + 5
	if !sg.Fits(0, 0, 16, rows) || !dg.Fits(0, 0, 4, h) {
		return
	}
	s, d := sg.Sub(0, 0, 16, rows), dg.Sub(0, 0, 4, h)
	if !s.Fits(0, 0, 16, 5) {
		return
	}
	hd := s.Sub(0, 0, 16, 5)
	x0 := mc8H4S(hd.Row16(0, 0), hc1, hc2, hc3, hc4, rnd).Shift(sh).ExtendLo4ToInt32()
	x1 := mc8H4S(hd.Row16(1, 0), hc1, hc2, hc3, hc4, rnd).Shift(sh).ExtendLo4ToInt32()
	x2 := mc8H4S(hd.Row16(2, 0), hc1, hc2, hc3, hc4, rnd).Shift(sh).ExtendLo4ToInt32()
	x3 := mc8H4S(hd.Row16(3, 0), hc1, hc2, hc3, hc4, rnd).Shift(sh).ExtendLo4ToInt32()
	x4 := mc8H4S(hd.Row16(4, 0), hc1, hc2, hc3, hc4, rnd).Shift(sh).ExtendLo4ToInt32()
	r := 0
	for ; r+4 <= h; r += 4 {
		if !s.Fits(r+5, 0, 16, 4) || !d.Fits(r, 0, 4, 4) {
			return
		}
		sw, dw := s.Sub(r+5, 0, 16, 4), d.Sub(r, 0, 4, 4)
		x5 := mc8H4S(sw.Row16(0, 0), hc1, hc2, hc3, hc4, rnd).Shift(sh).ExtendLo4ToInt32()
		x6 := mc8H4S(sw.Row16(1, 0), hc1, hc2, hc3, hc4, rnd).Shift(sh).ExtendLo4ToInt32()
		a0 := x5.MulAdd(vc5, x4.MulAdd(vc4, x3.MulAdd(vc3, x2.MulAdd(vc2, x1.MulAdd(vc1, x0.Mul(vc0))))))
		a1 := x6.MulAdd(vc5, x5.MulAdd(vc4, x4.MulAdd(vc3, x3.MulAdd(vc2, x2.MulAdd(vc1, x1.Mul(vc0))))))
		v0 := a0.ToBits().ReshapeToUint16s().ConcatOdd(a1.ToBits().ReshapeToUint16s()).BitsToInt16().SaturateToUint8().ReshapeToUint32s()
		binary.LittleEndian.PutUint32(dw.Row4(0, 0)[:], v0.GetElem(0))
		binary.LittleEndian.PutUint32(dw.Row4(1, 0)[:], v0.GetElem(1))
		x7 := mc8H4S(sw.Row16(2, 0), hc1, hc2, hc3, hc4, rnd).Shift(sh).ExtendLo4ToInt32()
		x8 := mc8H4S(sw.Row16(3, 0), hc1, hc2, hc3, hc4, rnd).Shift(sh).ExtendLo4ToInt32()
		a2 := x7.MulAdd(vc5, x6.MulAdd(vc4, x5.MulAdd(vc3, x4.MulAdd(vc2, x3.MulAdd(vc1, x2.Mul(vc0))))))
		a3 := x8.MulAdd(vc5, x7.MulAdd(vc4, x6.MulAdd(vc3, x5.MulAdd(vc2, x4.MulAdd(vc1, x3.Mul(vc0))))))
		v1 := a2.ToBits().ReshapeToUint16s().ConcatOdd(a3.ToBits().ReshapeToUint16s()).BitsToInt16().SaturateToUint8().ReshapeToUint32s()
		binary.LittleEndian.PutUint32(dw.Row4(2, 0)[:], v1.GetElem(0))
		binary.LittleEndian.PutUint32(dw.Row4(3, 0)[:], v1.GetElem(1))
		x0, x1, x2, x3, x4 = x4, x5, x6, x7, x8
	}
	for ; r+2 <= h; r += 2 {
		x5 := mc8H4S(s.Row16(r+5, 0), hc1, hc2, hc3, hc4, rnd).Shift(sh).ExtendLo4ToInt32()
		x6 := mc8H4S(s.Row16(r+6, 0), hc1, hc2, hc3, hc4, rnd).Shift(sh).ExtendLo4ToInt32()
		a0 := x5.MulAdd(vc5, x4.MulAdd(vc4, x3.MulAdd(vc3, x2.MulAdd(vc2, x1.MulAdd(vc1, x0.Mul(vc0))))))
		a1 := x6.MulAdd(vc5, x5.MulAdd(vc4, x4.MulAdd(vc3, x3.MulAdd(vc2, x2.MulAdd(vc1, x1.Mul(vc0))))))
		v := a0.ToBits().ReshapeToUint16s().ConcatOdd(a1.ToBits().ReshapeToUint16s()).BitsToInt16().SaturateToUint8().ReshapeToUint32s()
		binary.LittleEndian.PutUint32(d.Row4(r, 0)[:], v.GetElem(0))
		binary.LittleEndian.PutUint32(d.Row4(r+1, 0)[:], v.GetElem(1))
		x0, x1, x2, x3, x4 = x2, x3, x4, x5, x6
	}
	if r < h {
		x5 := mc8H4S(s.Row16(r+5, 0), hc1, hc2, hc3, hc4, rnd).Shift(sh).ExtendLo4ToInt32()
		a0 := x5.MulAdd(vc5, x4.MulAdd(vc4, x3.MulAdd(vc3, x2.MulAdd(vc2, x1.MulAdd(vc1, x0.Mul(vc0))))))
		v := a0.ToBits().ReshapeToUint16s().ConcatOdd(a0.ToBits().ReshapeToUint16s()).BitsToInt16().SaturateToUint8().ReshapeToUint32s()
		binary.LittleEndian.PutUint32(d.Row4(r, 0)[:], v.GetElem(0))
	}
}

// mc8PutHV2 is the fused 2-tap bilinear put 2D pass over 8-column strips
// (taps 2 and 3 of the 6-tap window on both axes).
func mc8PutHV2(sg unsafeutil.Grid[uint8], w, h int, phx, phy *mc8Phase, dg unsafeutil.Grid[uint8]) {
	hc2 := archsimd.LoadInt16x8Array(&phx.h[2])
	hc3 := archsimd.LoadInt16x8Array(&phx.h[3])
	rnd := archsimd.BroadcastInt16x8(mc8PutH6ToI32Rnd)
	sh := archsimd.LoadInt16x8Array(&mc8Vec.sh2)
	vc2 := archsimd.LoadInt32x4Array(&phy.v[2])
	vc3 := archsimd.LoadInt32x4Array(&phy.v[3])
	rows := h + 5
	for c := 0; c < w; c += 8 {
		if !sg.Fits(0, c, 16, rows) || !dg.Fits(0, c, 8, h) {
			return
		}
		s, d := sg.Sub(0, c, 16, rows), dg.Sub(0, c, 8, h)
		a := archsimd.LoadUint8x16Array(s.Row16(2, 0))
		lo, hi := a.ExtendLo8ToUint16().ReshapeToUint8s(), a.HiToLo().ExtendLo8ToUint16().ReshapeToUint8s()
		h0 := hi.ConcatShiftBytesRight(lo, 6).ReshapeToUint16s().BitsToInt16().MulAdd(hc3, hi.ConcatShiftBytesRight(lo, 4).ReshapeToUint16s().BitsToInt16().MulAdd(hc2, rnd)).Shift(sh)
		x0, y0 := h0.ExtendLo4ToInt32(), h0.HiToLo().ExtendLo4ToInt32()
		for r := 0; r < h; r++ {
			a = archsimd.LoadUint8x16Array(s.Row16(r+3, 0))
			lo, hi = a.ExtendLo8ToUint16().ReshapeToUint8s(), a.HiToLo().ExtendLo8ToUint16().ReshapeToUint8s()
			h1 := hi.ConcatShiftBytesRight(lo, 6).ReshapeToUint16s().BitsToInt16().MulAdd(hc3, hi.ConcatShiftBytesRight(lo, 4).ReshapeToUint16s().BitsToInt16().MulAdd(hc2, rnd)).Shift(sh)
			x1, y1 := h1.ExtendLo4ToInt32(), h1.HiToLo().ExtendLo4ToInt32()
			a0 := x1.MulAdd(vc3, x0.Mul(vc2))
			b0 := y1.MulAdd(vc3, y0.Mul(vc2))
			v0 := a0.ToBits().ReshapeToUint16s().ConcatOdd(b0.ToBits().ReshapeToUint16s()).BitsToInt16()
			binary.LittleEndian.PutUint64(d.Row8(r, 0)[:], mc8Lo64(v0.SaturateToUint8()))
			x0, y0 = x1, y1
		}
	}
}

// mc8PutHV6x4 is mc8PutHV6 when the vertical filter has at most 4 non-zero
// taps (taps 1..4, e.g. h == 4 regular4/smooth4/bilinear): skips rows 0 and
// h+4 and drops the zero outer vertical taps.
func mc8PutHV6x4(sg unsafeutil.Grid[uint8], w, h int, phx, phy *mc8Phase, dg unsafeutil.Grid[uint8]) {
	hc0 := archsimd.LoadInt16x8Array(&phx.h[0])
	hc1 := archsimd.LoadInt16x8Array(&phx.h[1])
	hc2 := archsimd.LoadInt16x8Array(&phx.h[2])
	hc3 := archsimd.LoadInt16x8Array(&phx.h[3])
	hc4 := archsimd.LoadInt16x8Array(&phx.h[4])
	hc5 := archsimd.LoadInt16x8Array(&phx.h[5])
	rnd := archsimd.BroadcastInt16x8(mc8PutH6ToI32Rnd)
	sh := archsimd.LoadInt16x8Array(&mc8Vec.sh2)
	vc1 := archsimd.LoadInt32x4Array(&phy.v[1])
	vc2 := archsimd.LoadInt32x4Array(&phy.v[2])
	vc3 := archsimd.LoadInt32x4Array(&phy.v[3])
	vc4 := archsimd.LoadInt32x4Array(&phy.v[4])
	rows := h + 4
	for c := 0; c < w; c += 8 {
		if !sg.Fits(0, c, 16, rows) || !dg.Fits(0, c, 8, h) {
			return
		}
		s, d := sg.Sub(0, c, 16, rows), dg.Sub(0, c, 8, h)
		if !s.Fits(1, 0, 16, 3) {
			return
		}
		hd := s.Sub(1, 0, 16, 3)
		a := mc8H6S(hd.Row16(0, 0), hc0, hc1, hc2, hc3, hc4, hc5, rnd).Shift(sh)
		x1, y1 := a.ExtendLo4ToInt32(), a.HiToLo().ExtendLo4ToInt32()
		a = mc8H6S(hd.Row16(1, 0), hc0, hc1, hc2, hc3, hc4, hc5, rnd).Shift(sh)
		x2, y2 := a.ExtendLo4ToInt32(), a.HiToLo().ExtendLo4ToInt32()
		a = mc8H6S(hd.Row16(2, 0), hc0, hc1, hc2, hc3, hc4, hc5, rnd).Shift(sh)
		x3, y3 := a.ExtendLo4ToInt32(), a.HiToLo().ExtendLo4ToInt32()
		r := 0
		for ; r+4 <= h; r += 4 {
			if !s.Fits(r+4, 0, 16, 4) || !d.Fits(r, 0, 8, 4) {
				return
			}
			sw, dw := s.Sub(r+4, 0, 16, 4), d.Sub(r, 0, 8, 4)
			a4 := mc8H6S(sw.Row16(0, 0), hc0, hc1, hc2, hc3, hc4, hc5, rnd).Shift(sh)
			x4, y4 := a4.ExtendLo4ToInt32(), a4.HiToLo().ExtendLo4ToInt32()
			a5 := mc8H6S(sw.Row16(1, 0), hc0, hc1, hc2, hc3, hc4, hc5, rnd).Shift(sh)
			x5, y5 := a5.ExtendLo4ToInt32(), a5.HiToLo().ExtendLo4ToInt32()
			p0 := x4.MulAdd(vc4, x3.MulAdd(vc3, x2.MulAdd(vc2, x1.Mul(vc1))))
			q0 := y4.MulAdd(vc4, y3.MulAdd(vc3, y2.MulAdd(vc2, y1.Mul(vc1))))
			p1 := x5.MulAdd(vc4, x4.MulAdd(vc3, x3.MulAdd(vc2, x2.Mul(vc1))))
			q1 := y5.MulAdd(vc4, y4.MulAdd(vc3, y3.MulAdd(vc2, y2.Mul(vc1))))
			v0 := p0.ToBits().ReshapeToUint16s().ConcatOdd(q0.ToBits().ReshapeToUint16s()).BitsToInt16()
			v1 := p1.ToBits().ReshapeToUint16s().ConcatOdd(q1.ToBits().ReshapeToUint16s()).BitsToInt16()
			binary.LittleEndian.PutUint64(dw.Row8(0, 0)[:], mc8Lo64(v0.SaturateToUint8()))
			binary.LittleEndian.PutUint64(dw.Row8(1, 0)[:], mc8Lo64(v1.SaturateToUint8()))
			a6 := mc8H6S(sw.Row16(2, 0), hc0, hc1, hc2, hc3, hc4, hc5, rnd).Shift(sh)
			x6, y6 := a6.ExtendLo4ToInt32(), a6.HiToLo().ExtendLo4ToInt32()
			a7 := mc8H6S(sw.Row16(3, 0), hc0, hc1, hc2, hc3, hc4, hc5, rnd).Shift(sh)
			x7, y7 := a7.ExtendLo4ToInt32(), a7.HiToLo().ExtendLo4ToInt32()
			p2 := x6.MulAdd(vc4, x5.MulAdd(vc3, x4.MulAdd(vc2, x3.Mul(vc1))))
			q2 := y6.MulAdd(vc4, y5.MulAdd(vc3, y4.MulAdd(vc2, y3.Mul(vc1))))
			p3 := x7.MulAdd(vc4, x6.MulAdd(vc3, x5.MulAdd(vc2, x4.Mul(vc1))))
			q3 := y7.MulAdd(vc4, y6.MulAdd(vc3, y5.MulAdd(vc2, y4.Mul(vc1))))
			v2 := p2.ToBits().ReshapeToUint16s().ConcatOdd(q2.ToBits().ReshapeToUint16s()).BitsToInt16()
			v3 := p3.ToBits().ReshapeToUint16s().ConcatOdd(q3.ToBits().ReshapeToUint16s()).BitsToInt16()
			binary.LittleEndian.PutUint64(dw.Row8(2, 0)[:], mc8Lo64(v2.SaturateToUint8()))
			binary.LittleEndian.PutUint64(dw.Row8(3, 0)[:], mc8Lo64(v3.SaturateToUint8()))
			x1, y1, x2, y2, x3, y3 = x5, y5, x6, y6, x7, y7
		}
		for ; r+2 <= h; r += 2 {
			a4 := mc8H6S(s.Row16(r+4, 0), hc0, hc1, hc2, hc3, hc4, hc5, rnd).Shift(sh)
			x4, y4 := a4.ExtendLo4ToInt32(), a4.HiToLo().ExtendLo4ToInt32()
			a5 := mc8H6S(s.Row16(r+5, 0), hc0, hc1, hc2, hc3, hc4, hc5, rnd).Shift(sh)
			x5, y5 := a5.ExtendLo4ToInt32(), a5.HiToLo().ExtendLo4ToInt32()
			p0 := x4.MulAdd(vc4, x3.MulAdd(vc3, x2.MulAdd(vc2, x1.Mul(vc1))))
			q0 := y4.MulAdd(vc4, y3.MulAdd(vc3, y2.MulAdd(vc2, y1.Mul(vc1))))
			p1 := x5.MulAdd(vc4, x4.MulAdd(vc3, x3.MulAdd(vc2, x2.Mul(vc1))))
			q1 := y5.MulAdd(vc4, y4.MulAdd(vc3, y3.MulAdd(vc2, y2.Mul(vc1))))
			v0 := p0.ToBits().ReshapeToUint16s().ConcatOdd(q0.ToBits().ReshapeToUint16s()).BitsToInt16()
			v1 := p1.ToBits().ReshapeToUint16s().ConcatOdd(q1.ToBits().ReshapeToUint16s()).BitsToInt16()
			binary.LittleEndian.PutUint64(d.Row8(r, 0)[:], mc8Lo64(v0.SaturateToUint8()))
			binary.LittleEndian.PutUint64(d.Row8(r+1, 0)[:], mc8Lo64(v1.SaturateToUint8()))
			x1, y1, x2, y2, x3, y3 = x3, y3, x4, y4, x5, y5
		}
		if r < h {
			a4 := mc8H6S(s.Row16(r+4, 0), hc0, hc1, hc2, hc3, hc4, hc5, rnd).Shift(sh)
			x4, y4 := a4.ExtendLo4ToInt32(), a4.HiToLo().ExtendLo4ToInt32()
			p0 := x4.MulAdd(vc4, x3.MulAdd(vc3, x2.MulAdd(vc2, x1.Mul(vc1))))
			q0 := y4.MulAdd(vc4, y3.MulAdd(vc3, y2.MulAdd(vc2, y1.Mul(vc1))))
			v0 := p0.ToBits().ReshapeToUint16s().ConcatOdd(q0.ToBits().ReshapeToUint16s()).BitsToInt16()
			binary.LittleEndian.PutUint64(d.Row8(r, 0)[:], mc8Lo64(v0.SaturateToUint8()))
		}
	}
}

// mc8PutHV6 is the fused put 2D six-tap horizontal+vertical pass over 8-column
// strips directly in registers with no intermediate buffer.
func mc8PutHV6(sg unsafeutil.Grid[uint8], w, h int, phx, phy *mc8Phase, dg unsafeutil.Grid[uint8]) {
	hc0 := archsimd.LoadInt16x8Array(&phx.h[0])
	hc1 := archsimd.LoadInt16x8Array(&phx.h[1])
	hc2 := archsimd.LoadInt16x8Array(&phx.h[2])
	hc3 := archsimd.LoadInt16x8Array(&phx.h[3])
	hc4 := archsimd.LoadInt16x8Array(&phx.h[4])
	hc5 := archsimd.LoadInt16x8Array(&phx.h[5])
	rnd := archsimd.BroadcastInt16x8(mc8PutH6ToI32Rnd)
	sh := archsimd.LoadInt16x8Array(&mc8Vec.sh2)
	vc0 := archsimd.LoadInt32x4Array(&phy.v[0])
	vc1 := archsimd.LoadInt32x4Array(&phy.v[1])
	vc2 := archsimd.LoadInt32x4Array(&phy.v[2])
	vc3 := archsimd.LoadInt32x4Array(&phy.v[3])
	vc4 := archsimd.LoadInt32x4Array(&phy.v[4])
	vc5 := archsimd.LoadInt32x4Array(&phy.v[5])
	rows := h + 5
	for c := 0; c < w; c += 8 {
		if !sg.Fits(0, c, 16, rows) || !dg.Fits(0, c, 8, h) {
			return
		}
		s, d := sg.Sub(0, c, 16, rows), dg.Sub(0, c, 8, h)
		if !s.Fits(0, 0, 16, 5) {
			return
		}
		hd := s.Sub(0, 0, 16, 5)
		a := mc8H6S(hd.Row16(0, 0), hc0, hc1, hc2, hc3, hc4, hc5, rnd).Shift(sh)
		x0, y0 := a.ExtendLo4ToInt32(), a.HiToLo().ExtendLo4ToInt32()
		a = mc8H6S(hd.Row16(1, 0), hc0, hc1, hc2, hc3, hc4, hc5, rnd).Shift(sh)
		x1, y1 := a.ExtendLo4ToInt32(), a.HiToLo().ExtendLo4ToInt32()
		a = mc8H6S(hd.Row16(2, 0), hc0, hc1, hc2, hc3, hc4, hc5, rnd).Shift(sh)
		x2, y2 := a.ExtendLo4ToInt32(), a.HiToLo().ExtendLo4ToInt32()
		a = mc8H6S(hd.Row16(3, 0), hc0, hc1, hc2, hc3, hc4, hc5, rnd).Shift(sh)
		x3, y3 := a.ExtendLo4ToInt32(), a.HiToLo().ExtendLo4ToInt32()
		a = mc8H6S(hd.Row16(4, 0), hc0, hc1, hc2, hc3, hc4, hc5, rnd).Shift(sh)
		x4, y4 := a.ExtendLo4ToInt32(), a.HiToLo().ExtendLo4ToInt32()
		r := 0
		for ; r+4 <= h; r += 4 {
			if !s.Fits(r+5, 0, 16, 4) || !d.Fits(r, 0, 8, 4) {
				return
			}
			sw, dw := s.Sub(r+5, 0, 16, 4), d.Sub(r, 0, 8, 4)
			a5 := mc8H6S(sw.Row16(0, 0), hc0, hc1, hc2, hc3, hc4, hc5, rnd).Shift(sh)
			x5, y5 := a5.ExtendLo4ToInt32(), a5.HiToLo().ExtendLo4ToInt32()
			a6 := mc8H6S(sw.Row16(1, 0), hc0, hc1, hc2, hc3, hc4, hc5, rnd).Shift(sh)
			x6, y6 := a6.ExtendLo4ToInt32(), a6.HiToLo().ExtendLo4ToInt32()
			u0, u1 := mc8V6PutPair(x0, y0, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, vc0, vc1, vc2, vc3, vc4, vc5)
			binary.LittleEndian.PutUint64(dw.Row8(0, 0)[:], u0)
			binary.LittleEndian.PutUint64(dw.Row8(1, 0)[:], u1)
			a7 := mc8H6S(sw.Row16(2, 0), hc0, hc1, hc2, hc3, hc4, hc5, rnd).Shift(sh)
			x7, y7 := a7.ExtendLo4ToInt32(), a7.HiToLo().ExtendLo4ToInt32()
			a8 := mc8H6S(sw.Row16(3, 0), hc0, hc1, hc2, hc3, hc4, hc5, rnd).Shift(sh)
			x8, y8 := a8.ExtendLo4ToInt32(), a8.HiToLo().ExtendLo4ToInt32()
			u2, u3 := mc8V6PutPair(x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8, vc0, vc1, vc2, vc3, vc4, vc5)
			binary.LittleEndian.PutUint64(dw.Row8(2, 0)[:], u2)
			binary.LittleEndian.PutUint64(dw.Row8(3, 0)[:], u3)
			x0, y0, x1, y1, x2, y2, x3, y3, x4, y4 = x4, y4, x5, y5, x6, y6, x7, y7, x8, y8
		}
		for ; r+2 <= h; r += 2 {
			a5 := mc8H6S(s.Row16(r+5, 0), hc0, hc1, hc2, hc3, hc4, hc5, rnd).Shift(sh)
			x5, y5 := a5.ExtendLo4ToInt32(), a5.HiToLo().ExtendLo4ToInt32()
			a6 := mc8H6S(s.Row16(r+6, 0), hc0, hc1, hc2, hc3, hc4, hc5, rnd).Shift(sh)
			x6, y6 := a6.ExtendLo4ToInt32(), a6.HiToLo().ExtendLo4ToInt32()
			u0, u1 := mc8V6PutPair(x0, y0, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, vc0, vc1, vc2, vc3, vc4, vc5)
			binary.LittleEndian.PutUint64(d.Row8(r, 0)[:], u0)
			binary.LittleEndian.PutUint64(d.Row8(r+1, 0)[:], u1)
			x0, y0, x1, y1, x2, y2, x3, y3, x4, y4 = x2, y2, x3, y3, x4, y4, x5, y5, x6, y6
		}
		if r < h {
			a5 := mc8H6S(s.Row16(r+5, 0), hc0, hc1, hc2, hc3, hc4, hc5, rnd).Shift(sh)
			x5, y5 := a5.ExtendLo4ToInt32(), a5.HiToLo().ExtendLo4ToInt32()
			u0, _ := mc8V6PutPair(x0, y0, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x5, y5, vc0, vc1, vc2, vc3, vc4, vc5)
			binary.LittleEndian.PutUint64(d.Row8(r, 0)[:], u0)
		}
	}
}

// mc8H6ToI32 writes the rows (h+5) x w rounded horizontal sums
// ((s + round) >> 2) of the source view sg (rows of max(w+8, 16) bytes) into
// the strip-major intermediate im, as int32.
func mc8H6ToI32(sg unsafeutil.Grid[uint8], w, rows int, ph *mc8Phase, im []int32, round int16) {
	c0 := archsimd.LoadInt16x8Array(&ph.h[0])
	c1 := archsimd.LoadInt16x8Array(&ph.h[1])
	c2 := archsimd.LoadInt16x8Array(&ph.h[2])
	c3 := archsimd.LoadInt16x8Array(&ph.h[3])
	c4 := archsimd.LoadInt16x8Array(&ph.h[4])
	c5 := archsimd.LoadInt16x8Array(&ph.h[5])
	rnd := archsimd.BroadcastInt16x8(round)
	sh := archsimd.LoadInt16x8Array(&mc8Vec.sh2)
	c := 0
	for ; c+16 <= w; c += 16 {
		// Two strips per pass: one 24-byte window per row, whose bytes 0..15
		// and 8..23 are the two strips' tap windows.
		if !sg.Fits(0, c, 24, rows) || !unsafeutil.GridFit(len(im), c*rows, 8, 8, 2*rows) {
			return
		}
		s, m := sg.Sub(0, c, 24, rows), unsafeutil.NewGrid(im, c*rows, 8, 8, 2*rows)
		if !m.Fits(rows, 0, 8, rows) {
			return
		}
		m0, m1 := m.Sub(0, 0, 8, rows), m.Sub(rows, 0, 8, rows)
		for r := 0; r < rows; r++ {
			win := s.Row24(r, 0)
			a := mc8H6S((*[16]uint8)(win[0:16]), c0, c1, c2, c3, c4, c5, rnd).Shift(sh)
			b := mc8H6S((*[16]uint8)(win[8:24]), c0, c1, c2, c3, c4, c5, rnd).Shift(sh)
			p, q := m0.Row8(r, 0), m1.Row8(r, 0)
			a.ExtendLo4ToInt32().StoreArray((*[4]int32)(p[0:4]))
			a.HiToLo().ExtendLo4ToInt32().StoreArray((*[4]int32)(p[4:8]))
			b.ExtendLo4ToInt32().StoreArray((*[4]int32)(q[0:4]))
			b.HiToLo().ExtendLo4ToInt32().StoreArray((*[4]int32)(q[4:8]))
		}
	}
	if c < w {
		if !sg.Fits(0, c, 16, rows) || !unsafeutil.GridFit(len(im), c*rows, 8, 8, rows) {
			return
		}
		s, m := sg.Sub(0, c, 16, rows), unsafeutil.NewGrid(im, c*rows, 8, 8, rows)
		for r := 0; r < rows; r++ {
			a := mc8H6S(s.Row16(r, 0), c0, c1, c2, c3, c4, c5, rnd).Shift(sh)
			p := m.Row8(r, 0)
			a.ExtendLo4ToInt32().StoreArray((*[4]int32)(p[0:4]))
			a.HiToLo().ExtendLo4ToInt32().StoreArray((*[4]int32)(p[4:8]))
		}
	}
}

// mc8H6ToI32W4 writes rows x 4 rounded horizontal sums ((s + 2) >> 2) of the
// source view sg (rows of 16 readable bytes) as int32, row stride 4.
func mc8H6ToI32W4(sg unsafeutil.Grid[uint8], rows int, ph *mc8Phase, im []int32) {
	c0 := archsimd.LoadInt16x8Array(&ph.h[0])
	c1 := archsimd.LoadInt16x8Array(&ph.h[1])
	c2 := archsimd.LoadInt16x8Array(&ph.h[2])
	c3 := archsimd.LoadInt16x8Array(&ph.h[3])
	c4 := archsimd.LoadInt16x8Array(&ph.h[4])
	c5 := archsimd.LoadInt16x8Array(&ph.h[5])
	rnd := archsimd.LoadInt16x8Array(&mc8Vec.rnd2)
	sh := archsimd.LoadInt16x8Array(&mc8Vec.sh2)
	if !sg.Fits(0, 0, 16, rows) || !unsafeutil.GridFit(len(im), 0, 4, 4, rows) {
		return
	}
	s, m := sg.Sub(0, 0, 16, rows), unsafeutil.NewGrid(im, 0, 4, 4, rows)
	for r := 0; r < rows; r++ {
		mc8H6S(s.Row16(r, 0), c0, c1, c2, c3, c4, c5, rnd).Shift(sh).ExtendLo4ToInt32().StoreArray(m.Row4(r, 0))
	}
}

// mc8PutV6FromI32 is the put 2D vertical pass over the strip-major
// intermediate (h+5 rows per strip) into the w x h destination view dg.
func mc8PutV6FromI32(im []int32, w, h int, ph *mc8Phase, dg unsafeutil.Grid[uint8]) {
	c0 := archsimd.LoadInt32x4Array(&ph.v[0])
	c1 := archsimd.LoadInt32x4Array(&ph.v[1])
	c2 := archsimd.LoadInt32x4Array(&ph.v[2])
	c3 := archsimd.LoadInt32x4Array(&ph.v[3])
	c4 := archsimd.LoadInt32x4Array(&ph.v[4])
	c5 := archsimd.LoadInt32x4Array(&ph.v[5])
	rows := h + 5
	n := rows * 8
	for c := 0; c < w; c += 8 {
		if !dg.Fits(0, c, 8, h) {
			return
		}
		dc := dg.Sub(0, c, 8, h)
		// Strip c/8 of the strip-major intermediate (rows of eight
		// contiguous samples): every window below is proven in bounds.
		st := im[c*rows : c*rows+n : c*rows+n]
		head := (*[40]int32)(st[0:40])
		x0, y0 := archsimd.LoadInt32x4Array((*[4]int32)(head[0:4])), archsimd.LoadInt32x4Array((*[4]int32)(head[4:8]))
		x1, y1 := archsimd.LoadInt32x4Array((*[4]int32)(head[8:12])), archsimd.LoadInt32x4Array((*[4]int32)(head[12:16]))
		x2, y2 := archsimd.LoadInt32x4Array((*[4]int32)(head[16:20])), archsimd.LoadInt32x4Array((*[4]int32)(head[20:24]))
		x3, y3 := archsimd.LoadInt32x4Array((*[4]int32)(head[24:28])), archsimd.LoadInt32x4Array((*[4]int32)(head[28:32]))
		x4, y4 := archsimd.LoadInt32x4Array((*[4]int32)(head[32:36])), archsimd.LoadInt32x4Array((*[4]int32)(head[36:40]))
		r, q := 0, 40
		for ; q <= len(st)-64; q += 64 {
			if !dc.Fits(r, 0, 8, 8) {
				return
			}
			d := dc.Sub(r, 0, 8, 8)
			win := (*[64]int32)(st[q : q+64])
			x5, y5 := archsimd.LoadInt32x4Array((*[4]int32)(win[0:4])), archsimd.LoadInt32x4Array((*[4]int32)(win[4:8]))
			x6, y6 := archsimd.LoadInt32x4Array((*[4]int32)(win[8:12])), archsimd.LoadInt32x4Array((*[4]int32)(win[12:16]))
			x7, y7 := archsimd.LoadInt32x4Array((*[4]int32)(win[16:20])), archsimd.LoadInt32x4Array((*[4]int32)(win[20:24]))
			x8, y8 := archsimd.LoadInt32x4Array((*[4]int32)(win[24:28])), archsimd.LoadInt32x4Array((*[4]int32)(win[28:32]))
			x9, y9 := archsimd.LoadInt32x4Array((*[4]int32)(win[32:36])), archsimd.LoadInt32x4Array((*[4]int32)(win[36:40]))
			x10, y10 := archsimd.LoadInt32x4Array((*[4]int32)(win[40:44])), archsimd.LoadInt32x4Array((*[4]int32)(win[44:48]))
			x11, y11 := archsimd.LoadInt32x4Array((*[4]int32)(win[48:52])), archsimd.LoadInt32x4Array((*[4]int32)(win[52:56]))
			x12, y12 := archsimd.LoadInt32x4Array((*[4]int32)(win[56:60])), archsimd.LoadInt32x4Array((*[4]int32)(win[60:64]))
			u0, u1 := mc8V6PutPair(x0, y0, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, c0, c1, c2, c3, c4, c5)
			binary.LittleEndian.PutUint64(d.Row8(0, 0)[:], u0)
			binary.LittleEndian.PutUint64(d.Row8(1, 0)[:], u1)
			u0, u1 = mc8V6PutPair(x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8, c0, c1, c2, c3, c4, c5)
			binary.LittleEndian.PutUint64(d.Row8(2, 0)[:], u0)
			binary.LittleEndian.PutUint64(d.Row8(3, 0)[:], u1)
			u0, u1 = mc8V6PutPair(x4, y4, x5, y5, x6, y6, x7, y7, x8, y8, x9, y9, x10, y10, c0, c1, c2, c3, c4, c5)
			binary.LittleEndian.PutUint64(d.Row8(4, 0)[:], u0)
			binary.LittleEndian.PutUint64(d.Row8(5, 0)[:], u1)
			u0, u1 = mc8V6PutPair(x6, y6, x7, y7, x8, y8, x9, y9, x10, y10, x11, y11, x12, y12, c0, c1, c2, c3, c4, c5)
			binary.LittleEndian.PutUint64(d.Row8(6, 0)[:], u0)
			binary.LittleEndian.PutUint64(d.Row8(7, 0)[:], u1)
			r += 8
			x0, y0, x1, y1, x2, y2, x3, y3, x4, y4 = x8, y8, x9, y9, x10, y10, x11, y11, x12, y12
		}
		for ; q <= len(st)-16; q += 16 {
			if !dc.Fits(r, 0, 8, 2) {
				return
			}
			d := dc.Sub(r, 0, 8, 2)
			win := (*[16]int32)(st[q : q+16])
			x5, y5 := archsimd.LoadInt32x4Array((*[4]int32)(win[0:4])), archsimd.LoadInt32x4Array((*[4]int32)(win[4:8]))
			x6, y6 := archsimd.LoadInt32x4Array((*[4]int32)(win[8:12])), archsimd.LoadInt32x4Array((*[4]int32)(win[12:16]))
			u0, u1 := mc8V6PutPair(x0, y0, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, c0, c1, c2, c3, c4, c5)
			binary.LittleEndian.PutUint64(d.Row8(0, 0)[:], u0)
			binary.LittleEndian.PutUint64(d.Row8(1, 0)[:], u1)
			r += 2
			x0, y0, x1, y1, x2, y2, x3, y3, x4, y4 = x2, y2, x3, y3, x4, y4, x5, y5, x6, y6
		}
		if q <= len(st)-8 {
			if !dc.Fits(r, 0, 8, 1) {
				return
			}
			x5, y5 := mc8LoadI32x8((*[8]int32)(st[q : q+8]))
			u0, _ := mc8V6PutPair(x0, y0, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x5, y5, c0, c1, c2, c3, c4, c5)
			binary.LittleEndian.PutUint64(dc.Sub(r, 0, 8, 1).Row8(0, 0)[:], u0)
		}
	}
}

// mc8PrepV6FromI32S is the prep 2D vertical pass over the strip-major
// intermediate (h+5 rows per strip) into the w x h CONV_BUF view og.
func mc8PrepV6FromI32S(im []int32, w, h int, ph *mc8Phase, og unsafeutil.Grid[uint16], rndConst int32) {
	c0 := archsimd.LoadInt32x4Array(&ph.p[0])
	c1 := archsimd.LoadInt32x4Array(&ph.p[1])
	c2 := archsimd.LoadInt32x4Array(&ph.p[2])
	c3 := archsimd.LoadInt32x4Array(&ph.p[3])
	c4 := archsimd.LoadInt32x4Array(&ph.p[4])
	c5 := archsimd.LoadInt32x4Array(&ph.p[5])
	rnd := archsimd.BroadcastInt32x4(rndConst)
	rows := h + 5
	n := rows * 8
	for c := 0; c < w; c += 8 {
		if !og.Fits(0, c, 8, h) {
			return
		}
		oc := og.Sub(0, c, 8, h)
		st := im[c*rows : c*rows+n : c*rows+n]
		head := (*[40]int32)(st[0:40])
		x0, y0 := archsimd.LoadInt32x4Array((*[4]int32)(head[0:4])), archsimd.LoadInt32x4Array((*[4]int32)(head[4:8]))
		x1, y1 := archsimd.LoadInt32x4Array((*[4]int32)(head[8:12])), archsimd.LoadInt32x4Array((*[4]int32)(head[12:16]))
		x2, y2 := archsimd.LoadInt32x4Array((*[4]int32)(head[16:20])), archsimd.LoadInt32x4Array((*[4]int32)(head[20:24]))
		x3, y3 := archsimd.LoadInt32x4Array((*[4]int32)(head[24:28])), archsimd.LoadInt32x4Array((*[4]int32)(head[28:32]))
		x4, y4 := archsimd.LoadInt32x4Array((*[4]int32)(head[32:36])), archsimd.LoadInt32x4Array((*[4]int32)(head[36:40]))
		r, q := 0, 40
		for ; q <= len(st)-64; q += 64 {
			if !oc.Fits(r, 0, 8, 8) {
				return
			}
			o := oc.Sub(r, 0, 8, 8)
			win := (*[64]int32)(st[q : q+64])
			x5, y5 := archsimd.LoadInt32x4Array((*[4]int32)(win[0:4])), archsimd.LoadInt32x4Array((*[4]int32)(win[4:8]))
			x6, y6 := archsimd.LoadInt32x4Array((*[4]int32)(win[8:12])), archsimd.LoadInt32x4Array((*[4]int32)(win[12:16]))
			x7, y7 := archsimd.LoadInt32x4Array((*[4]int32)(win[16:20])), archsimd.LoadInt32x4Array((*[4]int32)(win[20:24]))
			x8, y8 := archsimd.LoadInt32x4Array((*[4]int32)(win[24:28])), archsimd.LoadInt32x4Array((*[4]int32)(win[28:32]))
			x9, y9 := archsimd.LoadInt32x4Array((*[4]int32)(win[32:36])), archsimd.LoadInt32x4Array((*[4]int32)(win[36:40]))
			x10, y10 := archsimd.LoadInt32x4Array((*[4]int32)(win[40:44])), archsimd.LoadInt32x4Array((*[4]int32)(win[44:48]))
			x11, y11 := archsimd.LoadInt32x4Array((*[4]int32)(win[48:52])), archsimd.LoadInt32x4Array((*[4]int32)(win[52:56]))
			x12, y12 := archsimd.LoadInt32x4Array((*[4]int32)(win[56:60])), archsimd.LoadInt32x4Array((*[4]int32)(win[60:64]))
			u0, u1 := mc8V6PrepPair(x0, y0, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, c0, c1, c2, c3, c4, c5, rnd)
			u0.StoreArray(o.Row8(0, 0))
			u1.StoreArray(o.Row8(1, 0))
			u0, u1 = mc8V6PrepPair(x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8, c0, c1, c2, c3, c4, c5, rnd)
			u0.StoreArray(o.Row8(2, 0))
			u1.StoreArray(o.Row8(3, 0))
			u0, u1 = mc8V6PrepPair(x4, y4, x5, y5, x6, y6, x7, y7, x8, y8, x9, y9, x10, y10, c0, c1, c2, c3, c4, c5, rnd)
			u0.StoreArray(o.Row8(4, 0))
			u1.StoreArray(o.Row8(5, 0))
			u0, u1 = mc8V6PrepPair(x6, y6, x7, y7, x8, y8, x9, y9, x10, y10, x11, y11, x12, y12, c0, c1, c2, c3, c4, c5, rnd)
			u0.StoreArray(o.Row8(6, 0))
			u1.StoreArray(o.Row8(7, 0))
			r += 8
			x0, y0, x1, y1, x2, y2, x3, y3, x4, y4 = x8, y8, x9, y9, x10, y10, x11, y11, x12, y12
		}
		for ; q <= len(st)-16; q += 16 {
			if !oc.Fits(r, 0, 8, 2) {
				return
			}
			o := oc.Sub(r, 0, 8, 2)
			win := (*[16]int32)(st[q : q+16])
			x5, y5 := archsimd.LoadInt32x4Array((*[4]int32)(win[0:4])), archsimd.LoadInt32x4Array((*[4]int32)(win[4:8]))
			x6, y6 := archsimd.LoadInt32x4Array((*[4]int32)(win[8:12])), archsimd.LoadInt32x4Array((*[4]int32)(win[12:16]))
			u0, u1 := mc8V6PrepPair(x0, y0, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, c0, c1, c2, c3, c4, c5, rnd)
			u0.StoreArray(o.Row8(0, 0))
			u1.StoreArray(o.Row8(1, 0))
			r += 2
			x0, y0, x1, y1, x2, y2, x3, y3, x4, y4 = x2, y2, x3, y3, x4, y4, x5, y5, x6, y6
		}
		if q <= len(st)-8 {
			if !oc.Fits(r, 0, 8, 1) {
				return
			}
			x5, y5 := mc8LoadI32x8((*[8]int32)(st[q : q+8]))
			u0, _ := mc8V6PrepPair(x0, y0, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x5, y5, c0, c1, c2, c3, c4, c5, rnd)
			u0.StoreArray(oc.Sub(r, 0, 8, 1).Row8(0, 0))
		}
	}
}

// mc8PutV6FromI32W4 is the width-4 put 2D vertical pass from the h+5 rows of
// four int32 sums at the start of im (row stride 4) into the 4 x h
// destination view dg: four output rows per chunk (then at most one chunk of
// two and one of one row). Two output rows' high halves pack into one vector
// (UZP2) that saturates to both rows. The intermediate is a Grid view checked
// once, so, as the other kernels, the pass is a leaf.
func mc8PutV6FromI32W4(im []int32, h int, ph *mc8Phase, dg unsafeutil.Grid[uint8]) {
	c0 := archsimd.LoadInt32x4Array(&ph.v[0])
	c1 := archsimd.LoadInt32x4Array(&ph.v[1])
	c2 := archsimd.LoadInt32x4Array(&ph.v[2])
	c3 := archsimd.LoadInt32x4Array(&ph.v[3])
	c4 := archsimd.LoadInt32x4Array(&ph.v[4])
	c5 := archsimd.LoadInt32x4Array(&ph.v[5])
	rnd := archsimd.LoadInt32x4Array(&mc8Vec.w15)
	if !unsafeutil.GridFit(len(im), 0, 4, 4, h+5) {
		return
	}
	m := unsafeutil.NewGrid(im, 0, 4, 4, h+5)
	if !m.Fits(0, 0, 4, 5) {
		return
	}
	hd := m.Sub(0, 0, 4, 5)
	x0 := archsimd.LoadInt32x4Array(hd.Row4(0, 0))
	x1 := archsimd.LoadInt32x4Array(hd.Row4(1, 0))
	x2 := archsimd.LoadInt32x4Array(hd.Row4(2, 0))
	x3 := archsimd.LoadInt32x4Array(hd.Row4(3, 0))
	x4 := archsimd.LoadInt32x4Array(hd.Row4(4, 0))
	r := 0
	for ; r+4 <= h; r += 4 {
		if !m.Fits(r+5, 0, 4, 4) || !dg.Fits(r, 0, 4, 4) {
			return
		}
		mw, d := m.Sub(r+5, 0, 4, 4), dg.Sub(r, 0, 4, 4)
		x5 := archsimd.LoadInt32x4Array(mw.Row4(0, 0))
		x6 := archsimd.LoadInt32x4Array(mw.Row4(1, 0))
		a, b := mc8V6I32W4(x0, x1, x2, x3, x4, x5, x6, c0, c1, c2, c3, c4, c5, rnd)
		u := a.ToBits().ReshapeToUint16s().ConcatOdd(b.ToBits().ReshapeToUint16s()).BitsToInt16().SaturateToUint8().ReshapeToUint64s().GetElem(0)
		binary.LittleEndian.PutUint32(d.Row4(0, 0)[:], uint32(u))
		binary.LittleEndian.PutUint32(d.Row4(1, 0)[:], uint32(u>>32))
		x7 := archsimd.LoadInt32x4Array(mw.Row4(2, 0))
		x8 := archsimd.LoadInt32x4Array(mw.Row4(3, 0))
		a, b = mc8V6I32W4(x2, x3, x4, x5, x6, x7, x8, c0, c1, c2, c3, c4, c5, rnd)
		u = a.ToBits().ReshapeToUint16s().ConcatOdd(b.ToBits().ReshapeToUint16s()).BitsToInt16().SaturateToUint8().ReshapeToUint64s().GetElem(0)
		binary.LittleEndian.PutUint32(d.Row4(2, 0)[:], uint32(u))
		binary.LittleEndian.PutUint32(d.Row4(3, 0)[:], uint32(u>>32))
		x0, x1, x2, x3, x4 = x4, x5, x6, x7, x8
	}
	if r+2 <= h {
		if !m.Fits(r+5, 0, 4, 2) || !dg.Fits(r, 0, 4, 2) {
			return
		}
		mw, d := m.Sub(r+5, 0, 4, 2), dg.Sub(r, 0, 4, 2)
		x5 := archsimd.LoadInt32x4Array(mw.Row4(0, 0))
		x6 := archsimd.LoadInt32x4Array(mw.Row4(1, 0))
		a, b := mc8V6I32W4(x0, x1, x2, x3, x4, x5, x6, c0, c1, c2, c3, c4, c5, rnd)
		u := a.ToBits().ReshapeToUint16s().ConcatOdd(b.ToBits().ReshapeToUint16s()).BitsToInt16().SaturateToUint8().ReshapeToUint64s().GetElem(0)
		binary.LittleEndian.PutUint32(d.Row4(0, 0)[:], uint32(u))
		binary.LittleEndian.PutUint32(d.Row4(1, 0)[:], uint32(u>>32))
		r += 2
		x0, x1, x2, x3, x4 = x2, x3, x4, x5, x6
	}
	if r < h {
		if !m.Fits(r+5, 0, 4, 1) || !dg.Fits(r, 0, 4, 1) {
			return
		}
		mw, d := m.Sub(r+5, 0, 4, 1), dg.Sub(r, 0, 4, 1)
		x5 := archsimd.LoadInt32x4Array(mw.Row4(0, 0))
		a, _ := mc8V6I32W4(x0, x1, x2, x3, x4, x5, x5, c0, c1, c2, c3, c4, c5, rnd)
		v := a.ToBits().ReshapeToUint16s().ConcatOdd(a.ToBits().ReshapeToUint16s()).BitsToInt16()
		binary.LittleEndian.PutUint32(d.Row4(0, 0)[:], uint32(v.SaturateToUint8().ReshapeToUint64s().GetElem(0)))
	}
}
