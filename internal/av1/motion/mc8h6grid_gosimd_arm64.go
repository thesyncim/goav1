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

// The 8-bit six-tap horizontal kernels (dav1d put_6tap/prep_6tap h) over
// Grid views, eight rows per chunk (then at most one chunk each of four, two and one rows) of each
// column strip. Sixteen-column strips read one 24-byte window per row (the
// two eight-output tap windows are its bytes 0..15 and 8..23); every chunk is
// a fixed-shape sub-view whose Fits test is the only check its rows pay, a
// test the entries make unreachable, so the kernels stay leaf functions.

// mc8H6S is the six-tap halved horizontal sum (plus rnd) of the eight output
// columns whose tap window starts at p (taps read lanes 0..12 of the widened
// bytes). It calls no helper, so a call on the line of the vector store that
// uses it leaves no inline-mark NOP.
func mc8H6S(p *[16]uint8, c0, c1, c2, c3, c4, c5, rnd archsimd.Int16x8) archsimd.Int16x8 {
	a := archsimd.LoadUint8x16Array(p)
	lo, hi := a.ExtendLo8ToUint16().ReshapeToUint8s(), a.HiToLo().ExtendLo8ToUint16().ReshapeToUint8s()
	s := lo.ReshapeToUint16s().BitsToInt16().MulAdd(c0, rnd)
	s = hi.ConcatShiftBytesRight(lo, 2).ReshapeToUint16s().BitsToInt16().MulAdd(c1, s)
	s = hi.ConcatShiftBytesRight(lo, 4).ReshapeToUint16s().BitsToInt16().MulAdd(c2, s)
	s = hi.ConcatShiftBytesRight(lo, 6).ReshapeToUint16s().BitsToInt16().MulAdd(c3, s)
	s = hi.ConcatShiftBytesRight(lo, 8).ReshapeToUint16s().BitsToInt16().MulAdd(c4, s)
	return hi.ConcatShiftBytesRight(lo, 10).ReshapeToUint16s().BitsToInt16().MulAdd(c5, s)
}

// mc8PutH6W16 is the put horizontal-only pass, clip((s + 34) >> 6),
// over sixteen-column strips: sg holds the h = og.Height() source rows, each
// readable for 24 bytes from every strip (og.Width() a multiple of 16), and
// og the h destination rows.
func mc8PutH6W16(sg unsafeutil.Grid[uint8], ph *mc8Phase, og unsafeutil.Grid[uint8]) {
	c0 := archsimd.LoadInt16x8Array(&ph.h[0])
	c1 := archsimd.LoadInt16x8Array(&ph.h[1])
	c2 := archsimd.LoadInt16x8Array(&ph.h[2])
	c3 := archsimd.LoadInt16x8Array(&ph.h[3])
	c4 := archsimd.LoadInt16x8Array(&ph.h[4])
	c5 := archsimd.LoadInt16x8Array(&ph.h[5])
	rnd := archsimd.LoadInt16x8Array(&mc8Vec.rnd34)
	sh := archsimd.LoadInt16x8Array(&mc8Vec.sh6)
	h, w := og.Height(), og.Width()
	for c := 0; c+16 <= w; c += 16 {
		if !sg.Fits(0, c, 24, h) || !og.Fits(0, c, 16, h) {
			return
		}
		s, o := sg.Sub(0, c, 24, h), og.Sub(0, c, 16, h)
		var win *[24]byte
		r := 0
		if h >= 8 {
			for ; r+8 <= h; r += 8 {
				if !s.Fits(r, 0, 24, 8) || !o.Fits(r, 0, 16, 8) {
					return
				}
				sw, ow := s.Sub(r, 0, 24, 8), o.Sub(r, 0, 16, 8)
				win = unsafeutil.Row[[24]byte](sw, 0, 0)
				mc8H6S((*[16]uint8)(win[0:16]), c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().InterleaveLo(mc8H6S((*[16]uint8)(win[8:24]), c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s()).ReshapeToUint8s().StoreArray(unsafeutil.Row[[16]byte](ow, 0, 0))
				win = unsafeutil.Row[[24]byte](sw, 1, 0)
				mc8H6S((*[16]uint8)(win[0:16]), c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().InterleaveLo(mc8H6S((*[16]uint8)(win[8:24]), c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s()).ReshapeToUint8s().StoreArray(unsafeutil.Row[[16]byte](ow, 1, 0))
				win = unsafeutil.Row[[24]byte](sw, 2, 0)
				mc8H6S((*[16]uint8)(win[0:16]), c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().InterleaveLo(mc8H6S((*[16]uint8)(win[8:24]), c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s()).ReshapeToUint8s().StoreArray(unsafeutil.Row[[16]byte](ow, 2, 0))
				win = unsafeutil.Row[[24]byte](sw, 3, 0)
				mc8H6S((*[16]uint8)(win[0:16]), c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().InterleaveLo(mc8H6S((*[16]uint8)(win[8:24]), c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s()).ReshapeToUint8s().StoreArray(unsafeutil.Row[[16]byte](ow, 3, 0))
				win = unsafeutil.Row[[24]byte](sw, 4, 0)
				mc8H6S((*[16]uint8)(win[0:16]), c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().InterleaveLo(mc8H6S((*[16]uint8)(win[8:24]), c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s()).ReshapeToUint8s().StoreArray(unsafeutil.Row[[16]byte](ow, 4, 0))
				win = unsafeutil.Row[[24]byte](sw, 5, 0)
				mc8H6S((*[16]uint8)(win[0:16]), c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().InterleaveLo(mc8H6S((*[16]uint8)(win[8:24]), c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s()).ReshapeToUint8s().StoreArray(unsafeutil.Row[[16]byte](ow, 5, 0))
				win = unsafeutil.Row[[24]byte](sw, 6, 0)
				mc8H6S((*[16]uint8)(win[0:16]), c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().InterleaveLo(mc8H6S((*[16]uint8)(win[8:24]), c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s()).ReshapeToUint8s().StoreArray(unsafeutil.Row[[16]byte](ow, 6, 0))
				win = unsafeutil.Row[[24]byte](sw, 7, 0)
				mc8H6S((*[16]uint8)(win[0:16]), c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().InterleaveLo(mc8H6S((*[16]uint8)(win[8:24]), c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s()).ReshapeToUint8s().StoreArray(unsafeutil.Row[[16]byte](ow, 7, 0))
			}
		}
		if r < h {
			if r+4 <= h {
				if !s.Fits(r, 0, 24, 4) || !o.Fits(r, 0, 16, 4) {
					return
				}
				sw, ow := s.Sub(r, 0, 24, 4), o.Sub(r, 0, 16, 4)
				win = unsafeutil.Row[[24]byte](sw, 0, 0)
				mc8H6S((*[16]uint8)(win[0:16]), c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().InterleaveLo(mc8H6S((*[16]uint8)(win[8:24]), c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s()).ReshapeToUint8s().StoreArray(unsafeutil.Row[[16]byte](ow, 0, 0))
				win = unsafeutil.Row[[24]byte](sw, 1, 0)
				mc8H6S((*[16]uint8)(win[0:16]), c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().InterleaveLo(mc8H6S((*[16]uint8)(win[8:24]), c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s()).ReshapeToUint8s().StoreArray(unsafeutil.Row[[16]byte](ow, 1, 0))
				win = unsafeutil.Row[[24]byte](sw, 2, 0)
				mc8H6S((*[16]uint8)(win[0:16]), c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().InterleaveLo(mc8H6S((*[16]uint8)(win[8:24]), c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s()).ReshapeToUint8s().StoreArray(unsafeutil.Row[[16]byte](ow, 2, 0))
				win = unsafeutil.Row[[24]byte](sw, 3, 0)
				mc8H6S((*[16]uint8)(win[0:16]), c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().InterleaveLo(mc8H6S((*[16]uint8)(win[8:24]), c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s()).ReshapeToUint8s().StoreArray(unsafeutil.Row[[16]byte](ow, 3, 0))
				r += 4
			}
			if r+2 <= h {
				if !s.Fits(r, 0, 24, 2) || !o.Fits(r, 0, 16, 2) {
					return
				}
				sw, ow := s.Sub(r, 0, 24, 2), o.Sub(r, 0, 16, 2)
				win = unsafeutil.Row[[24]byte](sw, 0, 0)
				mc8H6S((*[16]uint8)(win[0:16]), c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().InterleaveLo(mc8H6S((*[16]uint8)(win[8:24]), c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s()).ReshapeToUint8s().StoreArray(unsafeutil.Row[[16]byte](ow, 0, 0))
				win = unsafeutil.Row[[24]byte](sw, 1, 0)
				mc8H6S((*[16]uint8)(win[0:16]), c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().InterleaveLo(mc8H6S((*[16]uint8)(win[8:24]), c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s()).ReshapeToUint8s().StoreArray(unsafeutil.Row[[16]byte](ow, 1, 0))
				r += 2
			}
			if r < h {
				if !s.Fits(r, 0, 24, 1) || !o.Fits(r, 0, 16, 1) {
					return
				}
				sw, ow := s.Sub(r, 0, 24, 1), o.Sub(r, 0, 16, 1)
				win = unsafeutil.Row[[24]byte](sw, 0, 0)
				mc8H6S((*[16]uint8)(win[0:16]), c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().InterleaveLo(mc8H6S((*[16]uint8)(win[8:24]), c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s()).ReshapeToUint8s().StoreArray(unsafeutil.Row[[16]byte](ow, 0, 0))
			}
		}
	}
}

// mc8PutH6S8 is mc8PutH6W16 over one eight-column strip: s holds its
// h source rows of sixteen readable bytes, o its h destination rows.
func mc8PutH6S8(s unsafeutil.Grid[uint8], ph *mc8Phase, o unsafeutil.Grid[uint8]) {
	c0 := archsimd.LoadInt16x8Array(&ph.h[0])
	c1 := archsimd.LoadInt16x8Array(&ph.h[1])
	c2 := archsimd.LoadInt16x8Array(&ph.h[2])
	c3 := archsimd.LoadInt16x8Array(&ph.h[3])
	c4 := archsimd.LoadInt16x8Array(&ph.h[4])
	c5 := archsimd.LoadInt16x8Array(&ph.h[5])
	rnd := archsimd.LoadInt16x8Array(&mc8Vec.rnd34)
	sh := archsimd.LoadInt16x8Array(&mc8Vec.sh6)
	h := o.Height()
	r := 0
	if h >= 8 {
		for ; r+8 <= h; r += 8 {
			if !s.Fits(r, 0, 16, 8) || !o.Fits(r, 0, 8, 8) {
				return
			}
			sw, ow := s.Sub(r, 0, 16, 8), o.Sub(r, 0, 8, 8)
			binary.LittleEndian.PutUint64(unsafeutil.Row[[8]byte](ow, 0, 0)[:], mc8H6S(unsafeutil.Row[[16]byte](sw, 0, 0), c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0))
			binary.LittleEndian.PutUint64(unsafeutil.Row[[8]byte](ow, 1, 0)[:], mc8H6S(unsafeutil.Row[[16]byte](sw, 1, 0), c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0))
			binary.LittleEndian.PutUint64(unsafeutil.Row[[8]byte](ow, 2, 0)[:], mc8H6S(unsafeutil.Row[[16]byte](sw, 2, 0), c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0))
			binary.LittleEndian.PutUint64(unsafeutil.Row[[8]byte](ow, 3, 0)[:], mc8H6S(unsafeutil.Row[[16]byte](sw, 3, 0), c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0))
			binary.LittleEndian.PutUint64(unsafeutil.Row[[8]byte](ow, 4, 0)[:], mc8H6S(unsafeutil.Row[[16]byte](sw, 4, 0), c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0))
			binary.LittleEndian.PutUint64(unsafeutil.Row[[8]byte](ow, 5, 0)[:], mc8H6S(unsafeutil.Row[[16]byte](sw, 5, 0), c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0))
			binary.LittleEndian.PutUint64(unsafeutil.Row[[8]byte](ow, 6, 0)[:], mc8H6S(unsafeutil.Row[[16]byte](sw, 6, 0), c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0))
			binary.LittleEndian.PutUint64(unsafeutil.Row[[8]byte](ow, 7, 0)[:], mc8H6S(unsafeutil.Row[[16]byte](sw, 7, 0), c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0))
		}
	}
	if r < h {
		if r+4 <= h {
			if !s.Fits(r, 0, 16, 4) || !o.Fits(r, 0, 8, 4) {
				return
			}
			sw, ow := s.Sub(r, 0, 16, 4), o.Sub(r, 0, 8, 4)
			binary.LittleEndian.PutUint64(unsafeutil.Row[[8]byte](ow, 0, 0)[:], mc8H6S(unsafeutil.Row[[16]byte](sw, 0, 0), c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0))
			binary.LittleEndian.PutUint64(unsafeutil.Row[[8]byte](ow, 1, 0)[:], mc8H6S(unsafeutil.Row[[16]byte](sw, 1, 0), c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0))
			binary.LittleEndian.PutUint64(unsafeutil.Row[[8]byte](ow, 2, 0)[:], mc8H6S(unsafeutil.Row[[16]byte](sw, 2, 0), c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0))
			binary.LittleEndian.PutUint64(unsafeutil.Row[[8]byte](ow, 3, 0)[:], mc8H6S(unsafeutil.Row[[16]byte](sw, 3, 0), c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0))
			r += 4
		}
		if r+2 <= h {
			if !s.Fits(r, 0, 16, 2) || !o.Fits(r, 0, 8, 2) {
				return
			}
			sw, ow := s.Sub(r, 0, 16, 2), o.Sub(r, 0, 8, 2)
			binary.LittleEndian.PutUint64(unsafeutil.Row[[8]byte](ow, 0, 0)[:], mc8H6S(unsafeutil.Row[[16]byte](sw, 0, 0), c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0))
			binary.LittleEndian.PutUint64(unsafeutil.Row[[8]byte](ow, 1, 0)[:], mc8H6S(unsafeutil.Row[[16]byte](sw, 1, 0), c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0))
			r += 2
		}
		if r < h {
			if !s.Fits(r, 0, 16, 1) || !o.Fits(r, 0, 8, 1) {
				return
			}
			sw, ow := s.Sub(r, 0, 16, 1), o.Sub(r, 0, 8, 1)
			binary.LittleEndian.PutUint64(unsafeutil.Row[[8]byte](ow, 0, 0)[:], mc8H6S(unsafeutil.Row[[16]byte](sw, 0, 0), c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0))
		}
	}
}

// mc8PutH6S4 is mc8PutH6S8 for a four-column block: s holds its h
// source rows of sixteen readable bytes, o its h destination rows of four samples.
func mc8PutH6S4(s unsafeutil.Grid[uint8], ph *mc8Phase, o unsafeutil.Grid[uint8]) {
	c0 := archsimd.LoadInt16x8Array(&ph.h[0])
	c1 := archsimd.LoadInt16x8Array(&ph.h[1])
	c2 := archsimd.LoadInt16x8Array(&ph.h[2])
	c3 := archsimd.LoadInt16x8Array(&ph.h[3])
	c4 := archsimd.LoadInt16x8Array(&ph.h[4])
	c5 := archsimd.LoadInt16x8Array(&ph.h[5])
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
			binary.LittleEndian.PutUint32(unsafeutil.Row[[4]byte](ow, 0, 0)[:], uint32(mc8H6S(unsafeutil.Row[[16]byte](sw, 0, 0), c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0)))
			binary.LittleEndian.PutUint32(unsafeutil.Row[[4]byte](ow, 1, 0)[:], uint32(mc8H6S(unsafeutil.Row[[16]byte](sw, 1, 0), c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0)))
			binary.LittleEndian.PutUint32(unsafeutil.Row[[4]byte](ow, 2, 0)[:], uint32(mc8H6S(unsafeutil.Row[[16]byte](sw, 2, 0), c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0)))
			binary.LittleEndian.PutUint32(unsafeutil.Row[[4]byte](ow, 3, 0)[:], uint32(mc8H6S(unsafeutil.Row[[16]byte](sw, 3, 0), c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0)))
			binary.LittleEndian.PutUint32(unsafeutil.Row[[4]byte](ow, 4, 0)[:], uint32(mc8H6S(unsafeutil.Row[[16]byte](sw, 4, 0), c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0)))
			binary.LittleEndian.PutUint32(unsafeutil.Row[[4]byte](ow, 5, 0)[:], uint32(mc8H6S(unsafeutil.Row[[16]byte](sw, 5, 0), c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0)))
			binary.LittleEndian.PutUint32(unsafeutil.Row[[4]byte](ow, 6, 0)[:], uint32(mc8H6S(unsafeutil.Row[[16]byte](sw, 6, 0), c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0)))
			binary.LittleEndian.PutUint32(unsafeutil.Row[[4]byte](ow, 7, 0)[:], uint32(mc8H6S(unsafeutil.Row[[16]byte](sw, 7, 0), c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0)))
		}
	}
	if r < h {
		if r+4 <= h {
			if !s.Fits(r, 0, 16, 4) || !o.Fits(r, 0, 4, 4) {
				return
			}
			sw, ow := s.Sub(r, 0, 16, 4), o.Sub(r, 0, 4, 4)
			binary.LittleEndian.PutUint32(unsafeutil.Row[[4]byte](ow, 0, 0)[:], uint32(mc8H6S(unsafeutil.Row[[16]byte](sw, 0, 0), c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0)))
			binary.LittleEndian.PutUint32(unsafeutil.Row[[4]byte](ow, 1, 0)[:], uint32(mc8H6S(unsafeutil.Row[[16]byte](sw, 1, 0), c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0)))
			binary.LittleEndian.PutUint32(unsafeutil.Row[[4]byte](ow, 2, 0)[:], uint32(mc8H6S(unsafeutil.Row[[16]byte](sw, 2, 0), c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0)))
			binary.LittleEndian.PutUint32(unsafeutil.Row[[4]byte](ow, 3, 0)[:], uint32(mc8H6S(unsafeutil.Row[[16]byte](sw, 3, 0), c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0)))
			r += 4
		}
		if r+2 <= h {
			if !s.Fits(r, 0, 16, 2) || !o.Fits(r, 0, 4, 2) {
				return
			}
			sw, ow := s.Sub(r, 0, 16, 2), o.Sub(r, 0, 4, 2)
			binary.LittleEndian.PutUint32(unsafeutil.Row[[4]byte](ow, 0, 0)[:], uint32(mc8H6S(unsafeutil.Row[[16]byte](sw, 0, 0), c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0)))
			binary.LittleEndian.PutUint32(unsafeutil.Row[[4]byte](ow, 1, 0)[:], uint32(mc8H6S(unsafeutil.Row[[16]byte](sw, 1, 0), c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0)))
			r += 2
		}
		if r < h {
			if !s.Fits(r, 0, 16, 1) || !o.Fits(r, 0, 4, 1) {
				return
			}
			sw, ow := s.Sub(r, 0, 16, 1), o.Sub(r, 0, 4, 1)
			binary.LittleEndian.PutUint32(unsafeutil.Row[[4]byte](ow, 0, 0)[:], uint32(mc8H6S(unsafeutil.Row[[16]byte](sw, 0, 0), c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0)))
		}
	}
}

// mc8PrepH6W16 is the prep horizontal-only pass, ((s + 2) >> 2) + 6144 (the offset sums lie in [0, 2^16) and shift as unsigned),
// over sixteen-column strips: sg holds the h = og.Height() source rows, each
// readable for 24 bytes from every strip (og.Width() a multiple of 16), and
// og the h CONV_BUF rows.
func mc8PrepH6W16(sg unsafeutil.Grid[uint8], ph *mc8Phase, og unsafeutil.Grid[uint16]) {
	c0 := archsimd.LoadInt16x8Array(&ph.h[0])
	c1 := archsimd.LoadInt16x8Array(&ph.h[1])
	c2 := archsimd.LoadInt16x8Array(&ph.h[2])
	c3 := archsimd.LoadInt16x8Array(&ph.h[3])
	c4 := archsimd.LoadInt16x8Array(&ph.h[4])
	c5 := archsimd.LoadInt16x8Array(&ph.h[5])
	rnd := archsimd.LoadInt16x8Array(&mc8Vec.prepRnd)
	sh := archsimd.LoadInt16x8Array(&mc8Vec.sh2)
	h, w := og.Height(), og.Width()
	for c := 0; c+16 <= w; c += 16 {
		if !sg.Fits(0, c, 24, h) || !og.Fits(0, c, 16, h) {
			return
		}
		s, o := sg.Sub(0, c, 24, h), og.Sub(0, c, 16, h)
		var win *[24]byte
		r := 0
		if h >= 8 {
			for ; r+8 <= h; r += 8 {
				if !s.Fits(r, 0, 24, 8) || !o.Fits(r, 0, 16, 8) {
					return
				}
				sw, ow := s.Sub(r, 0, 24, 8), o.Sub(r, 0, 16, 8)
				win = unsafeutil.Row[[24]byte](sw, 0, 0)
				mc8H6S((*[16]uint8)(win[0:16]), c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 0, 0))
				mc8H6S((*[16]uint8)(win[8:24]), c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 0, 8))
				win = unsafeutil.Row[[24]byte](sw, 1, 0)
				mc8H6S((*[16]uint8)(win[0:16]), c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 1, 0))
				mc8H6S((*[16]uint8)(win[8:24]), c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 1, 8))
				win = unsafeutil.Row[[24]byte](sw, 2, 0)
				mc8H6S((*[16]uint8)(win[0:16]), c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 2, 0))
				mc8H6S((*[16]uint8)(win[8:24]), c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 2, 8))
				win = unsafeutil.Row[[24]byte](sw, 3, 0)
				mc8H6S((*[16]uint8)(win[0:16]), c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 3, 0))
				mc8H6S((*[16]uint8)(win[8:24]), c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 3, 8))
				win = unsafeutil.Row[[24]byte](sw, 4, 0)
				mc8H6S((*[16]uint8)(win[0:16]), c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 4, 0))
				mc8H6S((*[16]uint8)(win[8:24]), c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 4, 8))
				win = unsafeutil.Row[[24]byte](sw, 5, 0)
				mc8H6S((*[16]uint8)(win[0:16]), c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 5, 0))
				mc8H6S((*[16]uint8)(win[8:24]), c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 5, 8))
				win = unsafeutil.Row[[24]byte](sw, 6, 0)
				mc8H6S((*[16]uint8)(win[0:16]), c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 6, 0))
				mc8H6S((*[16]uint8)(win[8:24]), c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 6, 8))
				win = unsafeutil.Row[[24]byte](sw, 7, 0)
				mc8H6S((*[16]uint8)(win[0:16]), c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 7, 0))
				mc8H6S((*[16]uint8)(win[8:24]), c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 7, 8))
			}
		}
		if r < h {
			if r+4 <= h {
				if !s.Fits(r, 0, 24, 4) || !o.Fits(r, 0, 16, 4) {
					return
				}
				sw, ow := s.Sub(r, 0, 24, 4), o.Sub(r, 0, 16, 4)
				win = unsafeutil.Row[[24]byte](sw, 0, 0)
				mc8H6S((*[16]uint8)(win[0:16]), c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 0, 0))
				mc8H6S((*[16]uint8)(win[8:24]), c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 0, 8))
				win = unsafeutil.Row[[24]byte](sw, 1, 0)
				mc8H6S((*[16]uint8)(win[0:16]), c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 1, 0))
				mc8H6S((*[16]uint8)(win[8:24]), c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 1, 8))
				win = unsafeutil.Row[[24]byte](sw, 2, 0)
				mc8H6S((*[16]uint8)(win[0:16]), c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 2, 0))
				mc8H6S((*[16]uint8)(win[8:24]), c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 2, 8))
				win = unsafeutil.Row[[24]byte](sw, 3, 0)
				mc8H6S((*[16]uint8)(win[0:16]), c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 3, 0))
				mc8H6S((*[16]uint8)(win[8:24]), c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 3, 8))
				r += 4
			}
			if r+2 <= h {
				if !s.Fits(r, 0, 24, 2) || !o.Fits(r, 0, 16, 2) {
					return
				}
				sw, ow := s.Sub(r, 0, 24, 2), o.Sub(r, 0, 16, 2)
				win = unsafeutil.Row[[24]byte](sw, 0, 0)
				mc8H6S((*[16]uint8)(win[0:16]), c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 0, 0))
				mc8H6S((*[16]uint8)(win[8:24]), c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 0, 8))
				win = unsafeutil.Row[[24]byte](sw, 1, 0)
				mc8H6S((*[16]uint8)(win[0:16]), c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 1, 0))
				mc8H6S((*[16]uint8)(win[8:24]), c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 1, 8))
				r += 2
			}
			if r < h {
				if !s.Fits(r, 0, 24, 1) || !o.Fits(r, 0, 16, 1) {
					return
				}
				sw, ow := s.Sub(r, 0, 24, 1), o.Sub(r, 0, 16, 1)
				win = unsafeutil.Row[[24]byte](sw, 0, 0)
				mc8H6S((*[16]uint8)(win[0:16]), c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 0, 0))
				mc8H6S((*[16]uint8)(win[8:24]), c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 0, 8))
			}
		}
	}
}

// mc8PrepH6S8 is mc8PrepH6W16 over one eight-column strip: s holds its
// h source rows of sixteen readable bytes, o its h CONV_BUF rows.
func mc8PrepH6S8(s unsafeutil.Grid[uint8], ph *mc8Phase, o unsafeutil.Grid[uint16]) {
	c0 := archsimd.LoadInt16x8Array(&ph.h[0])
	c1 := archsimd.LoadInt16x8Array(&ph.h[1])
	c2 := archsimd.LoadInt16x8Array(&ph.h[2])
	c3 := archsimd.LoadInt16x8Array(&ph.h[3])
	c4 := archsimd.LoadInt16x8Array(&ph.h[4])
	c5 := archsimd.LoadInt16x8Array(&ph.h[5])
	rnd := archsimd.LoadInt16x8Array(&mc8Vec.prepRnd)
	sh := archsimd.LoadInt16x8Array(&mc8Vec.sh2)
	h := o.Height()
	r := 0
	if h >= 8 {
		for ; r+8 <= h; r += 8 {
			if !s.Fits(r, 0, 16, 8) || !o.Fits(r, 0, 8, 8) {
				return
			}
			sw, ow := s.Sub(r, 0, 16, 8), o.Sub(r, 0, 8, 8)
			mc8H6S(unsafeutil.Row[[16]byte](sw, 0, 0), c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 0, 0))
			mc8H6S(unsafeutil.Row[[16]byte](sw, 1, 0), c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 1, 0))
			mc8H6S(unsafeutil.Row[[16]byte](sw, 2, 0), c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 2, 0))
			mc8H6S(unsafeutil.Row[[16]byte](sw, 3, 0), c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 3, 0))
			mc8H6S(unsafeutil.Row[[16]byte](sw, 4, 0), c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 4, 0))
			mc8H6S(unsafeutil.Row[[16]byte](sw, 5, 0), c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 5, 0))
			mc8H6S(unsafeutil.Row[[16]byte](sw, 6, 0), c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 6, 0))
			mc8H6S(unsafeutil.Row[[16]byte](sw, 7, 0), c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 7, 0))
		}
	}
	if r < h {
		if r+4 <= h {
			if !s.Fits(r, 0, 16, 4) || !o.Fits(r, 0, 8, 4) {
				return
			}
			sw, ow := s.Sub(r, 0, 16, 4), o.Sub(r, 0, 8, 4)
			mc8H6S(unsafeutil.Row[[16]byte](sw, 0, 0), c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 0, 0))
			mc8H6S(unsafeutil.Row[[16]byte](sw, 1, 0), c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 1, 0))
			mc8H6S(unsafeutil.Row[[16]byte](sw, 2, 0), c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 2, 0))
			mc8H6S(unsafeutil.Row[[16]byte](sw, 3, 0), c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 3, 0))
			r += 4
		}
		if r+2 <= h {
			if !s.Fits(r, 0, 16, 2) || !o.Fits(r, 0, 8, 2) {
				return
			}
			sw, ow := s.Sub(r, 0, 16, 2), o.Sub(r, 0, 8, 2)
			mc8H6S(unsafeutil.Row[[16]byte](sw, 0, 0), c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 0, 0))
			mc8H6S(unsafeutil.Row[[16]byte](sw, 1, 0), c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 1, 0))
			r += 2
		}
		if r < h {
			if !s.Fits(r, 0, 16, 1) || !o.Fits(r, 0, 8, 1) {
				return
			}
			sw, ow := s.Sub(r, 0, 16, 1), o.Sub(r, 0, 8, 1)
			mc8H6S(unsafeutil.Row[[16]byte](sw, 0, 0), c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 0, 0))
		}
	}
}

// mc8PrepH6S4 is mc8PrepH6S8 for a four-column block: s holds its h
// source rows of sixteen readable bytes, o its h CONV_BUF rows of four samples.
func mc8PrepH6S4(s unsafeutil.Grid[uint8], ph *mc8Phase, o unsafeutil.Grid[uint16]) {
	c0 := archsimd.LoadInt16x8Array(&ph.h[0])
	c1 := archsimd.LoadInt16x8Array(&ph.h[1])
	c2 := archsimd.LoadInt16x8Array(&ph.h[2])
	c3 := archsimd.LoadInt16x8Array(&ph.h[3])
	c4 := archsimd.LoadInt16x8Array(&ph.h[4])
	c5 := archsimd.LoadInt16x8Array(&ph.h[5])
	rnd := archsimd.LoadInt16x8Array(&mc8Vec.prepRnd)
	sh := archsimd.LoadInt16x8Array(&mc8Vec.sh2)
	h := o.Height()
	r := 0
	if h >= 8 {
		for ; r+8 <= h; r += 8 {
			if !s.Fits(r, 0, 16, 8) || !o.Fits(r, 0, 4, 8) {
				return
			}
			sw, ow := s.Sub(r, 0, 16, 8), o.Sub(r, 0, 4, 8)
			mc8Store4U16(unsafeutil.Row[[4]uint16](ow, 0, 0), mc8H6S(unsafeutil.Row[[16]byte](sw, 0, 0), c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).ReshapeToUint64s().GetElem(0))
			mc8Store4U16(unsafeutil.Row[[4]uint16](ow, 1, 0), mc8H6S(unsafeutil.Row[[16]byte](sw, 1, 0), c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).ReshapeToUint64s().GetElem(0))
			mc8Store4U16(unsafeutil.Row[[4]uint16](ow, 2, 0), mc8H6S(unsafeutil.Row[[16]byte](sw, 2, 0), c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).ReshapeToUint64s().GetElem(0))
			mc8Store4U16(unsafeutil.Row[[4]uint16](ow, 3, 0), mc8H6S(unsafeutil.Row[[16]byte](sw, 3, 0), c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).ReshapeToUint64s().GetElem(0))
			mc8Store4U16(unsafeutil.Row[[4]uint16](ow, 4, 0), mc8H6S(unsafeutil.Row[[16]byte](sw, 4, 0), c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).ReshapeToUint64s().GetElem(0))
			mc8Store4U16(unsafeutil.Row[[4]uint16](ow, 5, 0), mc8H6S(unsafeutil.Row[[16]byte](sw, 5, 0), c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).ReshapeToUint64s().GetElem(0))
			mc8Store4U16(unsafeutil.Row[[4]uint16](ow, 6, 0), mc8H6S(unsafeutil.Row[[16]byte](sw, 6, 0), c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).ReshapeToUint64s().GetElem(0))
			mc8Store4U16(unsafeutil.Row[[4]uint16](ow, 7, 0), mc8H6S(unsafeutil.Row[[16]byte](sw, 7, 0), c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).ReshapeToUint64s().GetElem(0))
		}
	}
	if r < h {
		if r+4 <= h {
			if !s.Fits(r, 0, 16, 4) || !o.Fits(r, 0, 4, 4) {
				return
			}
			sw, ow := s.Sub(r, 0, 16, 4), o.Sub(r, 0, 4, 4)
			mc8Store4U16(unsafeutil.Row[[4]uint16](ow, 0, 0), mc8H6S(unsafeutil.Row[[16]byte](sw, 0, 0), c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).ReshapeToUint64s().GetElem(0))
			mc8Store4U16(unsafeutil.Row[[4]uint16](ow, 1, 0), mc8H6S(unsafeutil.Row[[16]byte](sw, 1, 0), c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).ReshapeToUint64s().GetElem(0))
			mc8Store4U16(unsafeutil.Row[[4]uint16](ow, 2, 0), mc8H6S(unsafeutil.Row[[16]byte](sw, 2, 0), c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).ReshapeToUint64s().GetElem(0))
			mc8Store4U16(unsafeutil.Row[[4]uint16](ow, 3, 0), mc8H6S(unsafeutil.Row[[16]byte](sw, 3, 0), c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).ReshapeToUint64s().GetElem(0))
			r += 4
		}
		if r+2 <= h {
			if !s.Fits(r, 0, 16, 2) || !o.Fits(r, 0, 4, 2) {
				return
			}
			sw, ow := s.Sub(r, 0, 16, 2), o.Sub(r, 0, 4, 2)
			mc8Store4U16(unsafeutil.Row[[4]uint16](ow, 0, 0), mc8H6S(unsafeutil.Row[[16]byte](sw, 0, 0), c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).ReshapeToUint64s().GetElem(0))
			mc8Store4U16(unsafeutil.Row[[4]uint16](ow, 1, 0), mc8H6S(unsafeutil.Row[[16]byte](sw, 1, 0), c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).ReshapeToUint64s().GetElem(0))
			r += 2
		}
		if r < h {
			if !s.Fits(r, 0, 16, 1) || !o.Fits(r, 0, 4, 1) {
				return
			}
			sw, ow := s.Sub(r, 0, 16, 1), o.Sub(r, 0, 4, 1)
			mc8Store4U16(unsafeutil.Row[[4]uint16](ow, 0, 0), mc8H6S(unsafeutil.Row[[16]byte](sw, 0, 0), c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).ReshapeToUint64s().GetElem(0))
		}
	}
}
