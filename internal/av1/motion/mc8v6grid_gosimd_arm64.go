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

// The 8-bit six-tap vertical kernels (dav1d put_6tap/prep_6tap v) over Grid
// views. Each column strip keeps five source rows in registers and filters
// eight output rows per chunk (two, then one, for the last rows), so the
// resident rows rotate once per eight output rows. Every chunk is a
// fixed-shape sub-view whose Fits test is the only check its rows pay; the
// tests cannot fail (the entries sized the views), and with no panic behind
// them the kernels stay leaf functions.

// mc8PutV6W16 is the put vertical-only pass, clip((s + 32) >> 6),
// over sixteen-column strips: sg holds the h+5 source rows (h = og.Height(),
// og.Width() a multiple of 16) and og the h destination rows. Each source
// row is one load for both eight-sample halves.
func mc8PutV6W16(sg unsafeutil.Grid[uint8], ph *mc8Phase, og unsafeutil.Grid[uint8]) {
	c0 := archsimd.LoadInt16x8Array(&ph.h[0])
	c1 := archsimd.LoadInt16x8Array(&ph.h[1])
	c2 := archsimd.LoadInt16x8Array(&ph.h[2])
	c3 := archsimd.LoadInt16x8Array(&ph.h[3])
	c4 := archsimd.LoadInt16x8Array(&ph.h[4])
	c5 := archsimd.LoadInt16x8Array(&ph.h[5])
	rnd := archsimd.BroadcastInt16x8(32)
	sh := archsimd.BroadcastInt16x8(-6)
	h, w := og.Height(), og.Width()
	for c := 0; c+16 <= w; c += 16 {
		if !sg.Fits(0, c, 16, h+5) || !og.Fits(0, c, 16, h) {
			return
		}
		s, o := sg.Sub(0, c, 16, h+5), og.Sub(0, c, 16, h)
		if !s.Fits(0, 0, 16, 5) {
			return
		}
		hd := s.Sub(0, 0, 16, 5)
		x := archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](hd, 0, 0))
		l0, h0 := x.ExtendLo8ToUint16().BitsToInt16(), x.HiToLo().ExtendLo8ToUint16().BitsToInt16()
		x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](hd, 1, 0))
		l1, h1 := x.ExtendLo8ToUint16().BitsToInt16(), x.HiToLo().ExtendLo8ToUint16().BitsToInt16()
		x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](hd, 2, 0))
		l2, h2 := x.ExtendLo8ToUint16().BitsToInt16(), x.HiToLo().ExtendLo8ToUint16().BitsToInt16()
		x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](hd, 3, 0))
		l3, h3 := x.ExtendLo8ToUint16().BitsToInt16(), x.HiToLo().ExtendLo8ToUint16().BitsToInt16()
		x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](hd, 4, 0))
		l4, h4 := x.ExtendLo8ToUint16().BitsToInt16(), x.HiToLo().ExtendLo8ToUint16().BitsToInt16()
		r := 0
		for ; r+8 <= h; r += 8 {
			if !s.Fits(r+5, 0, 16, 8) || !o.Fits(r, 0, 16, 8) {
				return
			}
			sw, ow := s.Sub(r+5, 0, 16, 8), o.Sub(r, 0, 16, 8)
			x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 0, 0))
			l5, h5 := x.ExtendLo8ToUint16().BitsToInt16(), x.HiToLo().ExtendLo8ToUint16().BitsToInt16()
			mc8V6(l0, l1, l2, l3, l4, l5, c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().InterleaveLo(mc8V6(h0, h1, h2, h3, h4, h5, c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s()).ReshapeToUint8s().StoreArray(unsafeutil.Row[[16]byte](ow, 0, 0))
			x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 1, 0))
			l6, h6 := x.ExtendLo8ToUint16().BitsToInt16(), x.HiToLo().ExtendLo8ToUint16().BitsToInt16()
			mc8V6(l1, l2, l3, l4, l5, l6, c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().InterleaveLo(mc8V6(h1, h2, h3, h4, h5, h6, c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s()).ReshapeToUint8s().StoreArray(unsafeutil.Row[[16]byte](ow, 1, 0))
			x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 2, 0))
			l7, h7 := x.ExtendLo8ToUint16().BitsToInt16(), x.HiToLo().ExtendLo8ToUint16().BitsToInt16()
			mc8V6(l2, l3, l4, l5, l6, l7, c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().InterleaveLo(mc8V6(h2, h3, h4, h5, h6, h7, c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s()).ReshapeToUint8s().StoreArray(unsafeutil.Row[[16]byte](ow, 2, 0))
			x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 3, 0))
			l8, h8 := x.ExtendLo8ToUint16().BitsToInt16(), x.HiToLo().ExtendLo8ToUint16().BitsToInt16()
			mc8V6(l3, l4, l5, l6, l7, l8, c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().InterleaveLo(mc8V6(h3, h4, h5, h6, h7, h8, c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s()).ReshapeToUint8s().StoreArray(unsafeutil.Row[[16]byte](ow, 3, 0))
			x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 4, 0))
			l9, h9 := x.ExtendLo8ToUint16().BitsToInt16(), x.HiToLo().ExtendLo8ToUint16().BitsToInt16()
			mc8V6(l4, l5, l6, l7, l8, l9, c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().InterleaveLo(mc8V6(h4, h5, h6, h7, h8, h9, c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s()).ReshapeToUint8s().StoreArray(unsafeutil.Row[[16]byte](ow, 4, 0))
			x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 5, 0))
			l10, h10 := x.ExtendLo8ToUint16().BitsToInt16(), x.HiToLo().ExtendLo8ToUint16().BitsToInt16()
			mc8V6(l5, l6, l7, l8, l9, l10, c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().InterleaveLo(mc8V6(h5, h6, h7, h8, h9, h10, c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s()).ReshapeToUint8s().StoreArray(unsafeutil.Row[[16]byte](ow, 5, 0))
			x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 6, 0))
			l11, h11 := x.ExtendLo8ToUint16().BitsToInt16(), x.HiToLo().ExtendLo8ToUint16().BitsToInt16()
			mc8V6(l6, l7, l8, l9, l10, l11, c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().InterleaveLo(mc8V6(h6, h7, h8, h9, h10, h11, c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s()).ReshapeToUint8s().StoreArray(unsafeutil.Row[[16]byte](ow, 6, 0))
			x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 7, 0))
			l12, h12 := x.ExtendLo8ToUint16().BitsToInt16(), x.HiToLo().ExtendLo8ToUint16().BitsToInt16()
			mc8V6(l7, l8, l9, l10, l11, l12, c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().InterleaveLo(mc8V6(h7, h8, h9, h10, h11, h12, c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s()).ReshapeToUint8s().StoreArray(unsafeutil.Row[[16]byte](ow, 7, 0))
			l0, h0, l1, h1, l2, h2, l3, h3, l4, h4 = l8, h8, l9, h9, l10, h10, l11, h11, l12, h12
		}
		if r+4 <= h {
			if !s.Fits(r+5, 0, 16, 4) || !o.Fits(r, 0, 16, 4) {
				return
			}
			sw, ow := s.Sub(r+5, 0, 16, 4), o.Sub(r, 0, 16, 4)
			x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 0, 0))
			l5, h5 := x.ExtendLo8ToUint16().BitsToInt16(), x.HiToLo().ExtendLo8ToUint16().BitsToInt16()
			mc8V6(l0, l1, l2, l3, l4, l5, c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().InterleaveLo(mc8V6(h0, h1, h2, h3, h4, h5, c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s()).ReshapeToUint8s().StoreArray(unsafeutil.Row[[16]byte](ow, 0, 0))
			x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 1, 0))
			l6, h6 := x.ExtendLo8ToUint16().BitsToInt16(), x.HiToLo().ExtendLo8ToUint16().BitsToInt16()
			mc8V6(l1, l2, l3, l4, l5, l6, c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().InterleaveLo(mc8V6(h1, h2, h3, h4, h5, h6, c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s()).ReshapeToUint8s().StoreArray(unsafeutil.Row[[16]byte](ow, 1, 0))
			x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 2, 0))
			l7, h7 := x.ExtendLo8ToUint16().BitsToInt16(), x.HiToLo().ExtendLo8ToUint16().BitsToInt16()
			mc8V6(l2, l3, l4, l5, l6, l7, c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().InterleaveLo(mc8V6(h2, h3, h4, h5, h6, h7, c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s()).ReshapeToUint8s().StoreArray(unsafeutil.Row[[16]byte](ow, 2, 0))
			x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 3, 0))
			l8, h8 := x.ExtendLo8ToUint16().BitsToInt16(), x.HiToLo().ExtendLo8ToUint16().BitsToInt16()
			mc8V6(l3, l4, l5, l6, l7, l8, c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().InterleaveLo(mc8V6(h3, h4, h5, h6, h7, h8, c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s()).ReshapeToUint8s().StoreArray(unsafeutil.Row[[16]byte](ow, 3, 0))
			l0, h0, l1, h1, l2, h2, l3, h3, l4, h4 = l4, h4, l5, h5, l6, h6, l7, h7, l8, h8
			r += 4
		}
		if r+2 <= h {
			if !s.Fits(r+5, 0, 16, 2) || !o.Fits(r, 0, 16, 2) {
				return
			}
			sw, ow := s.Sub(r+5, 0, 16, 2), o.Sub(r, 0, 16, 2)
			x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 0, 0))
			l5, h5 := x.ExtendLo8ToUint16().BitsToInt16(), x.HiToLo().ExtendLo8ToUint16().BitsToInt16()
			mc8V6(l0, l1, l2, l3, l4, l5, c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().InterleaveLo(mc8V6(h0, h1, h2, h3, h4, h5, c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s()).ReshapeToUint8s().StoreArray(unsafeutil.Row[[16]byte](ow, 0, 0))
			x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 1, 0))
			l6, h6 := x.ExtendLo8ToUint16().BitsToInt16(), x.HiToLo().ExtendLo8ToUint16().BitsToInt16()
			mc8V6(l1, l2, l3, l4, l5, l6, c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().InterleaveLo(mc8V6(h1, h2, h3, h4, h5, h6, c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s()).ReshapeToUint8s().StoreArray(unsafeutil.Row[[16]byte](ow, 1, 0))
			l0, h0, l1, h1, l2, h2, l3, h3, l4, h4 = l2, h2, l3, h3, l4, h4, l5, h5, l6, h6
			r += 2
		}
		if r < h {
			if !s.Fits(r+5, 0, 16, 1) || !o.Fits(r, 0, 16, 1) {
				return
			}
			sw, ow := s.Sub(r+5, 0, 16, 1), o.Sub(r, 0, 16, 1)
			x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 0, 0))
			l5, h5 := x.ExtendLo8ToUint16().BitsToInt16(), x.HiToLo().ExtendLo8ToUint16().BitsToInt16()
			mc8V6(l0, l1, l2, l3, l4, l5, c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().InterleaveLo(mc8V6(h0, h1, h2, h3, h4, h5, c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s()).ReshapeToUint8s().StoreArray(unsafeutil.Row[[16]byte](ow, 0, 0))
		}
	}
}

// mc8PutV6S8 is mc8PutV6W16 over one eight-column strip: s holds its
// h+5 source rows of sixteen readable bytes, o its h destination rows.
func mc8PutV6S8(s unsafeutil.Grid[uint8], ph *mc8Phase, o unsafeutil.Grid[uint8]) {
	c0 := archsimd.LoadInt16x8Array(&ph.h[0])
	c1 := archsimd.LoadInt16x8Array(&ph.h[1])
	c2 := archsimd.LoadInt16x8Array(&ph.h[2])
	c3 := archsimd.LoadInt16x8Array(&ph.h[3])
	c4 := archsimd.LoadInt16x8Array(&ph.h[4])
	c5 := archsimd.LoadInt16x8Array(&ph.h[5])
	rnd := archsimd.BroadcastInt16x8(32)
	sh := archsimd.BroadcastInt16x8(-6)
	h := o.Height()
	if !s.Fits(0, 0, 16, 5) {
		return
	}
	hd := s.Sub(0, 0, 16, 5)
	r0 := archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](hd, 0, 0)).ExtendLo8ToUint16().BitsToInt16()
	r1 := archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](hd, 1, 0)).ExtendLo8ToUint16().BitsToInt16()
	r2 := archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](hd, 2, 0)).ExtendLo8ToUint16().BitsToInt16()
	r3 := archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](hd, 3, 0)).ExtendLo8ToUint16().BitsToInt16()
	r4 := archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](hd, 4, 0)).ExtendLo8ToUint16().BitsToInt16()
	r := 0
	for ; r+8 <= h; r += 8 {
		if !s.Fits(r+5, 0, 16, 8) || !o.Fits(r, 0, 8, 8) {
			return
		}
		sw, ow := s.Sub(r+5, 0, 16, 8), o.Sub(r, 0, 8, 8)
		r5 := archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 0, 0)).ExtendLo8ToUint16().BitsToInt16()
		binary.LittleEndian.PutUint64(unsafeutil.Row[[8]byte](ow, 0, 0)[:], mc8V6(r0, r1, r2, r3, r4, r5, c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0))
		r6 := archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 1, 0)).ExtendLo8ToUint16().BitsToInt16()
		binary.LittleEndian.PutUint64(unsafeutil.Row[[8]byte](ow, 1, 0)[:], mc8V6(r1, r2, r3, r4, r5, r6, c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0))
		r7 := archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 2, 0)).ExtendLo8ToUint16().BitsToInt16()
		binary.LittleEndian.PutUint64(unsafeutil.Row[[8]byte](ow, 2, 0)[:], mc8V6(r2, r3, r4, r5, r6, r7, c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0))
		r8 := archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 3, 0)).ExtendLo8ToUint16().BitsToInt16()
		binary.LittleEndian.PutUint64(unsafeutil.Row[[8]byte](ow, 3, 0)[:], mc8V6(r3, r4, r5, r6, r7, r8, c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0))
		r9 := archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 4, 0)).ExtendLo8ToUint16().BitsToInt16()
		binary.LittleEndian.PutUint64(unsafeutil.Row[[8]byte](ow, 4, 0)[:], mc8V6(r4, r5, r6, r7, r8, r9, c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0))
		r10 := archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 5, 0)).ExtendLo8ToUint16().BitsToInt16()
		binary.LittleEndian.PutUint64(unsafeutil.Row[[8]byte](ow, 5, 0)[:], mc8V6(r5, r6, r7, r8, r9, r10, c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0))
		r11 := archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 6, 0)).ExtendLo8ToUint16().BitsToInt16()
		binary.LittleEndian.PutUint64(unsafeutil.Row[[8]byte](ow, 6, 0)[:], mc8V6(r6, r7, r8, r9, r10, r11, c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0))
		r12 := archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 7, 0)).ExtendLo8ToUint16().BitsToInt16()
		binary.LittleEndian.PutUint64(unsafeutil.Row[[8]byte](ow, 7, 0)[:], mc8V6(r7, r8, r9, r10, r11, r12, c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0))
		r0, r1, r2, r3, r4 = r8, r9, r10, r11, r12
	}
	if r+4 <= h {
		if !s.Fits(r+5, 0, 16, 4) || !o.Fits(r, 0, 8, 4) {
			return
		}
		sw, ow := s.Sub(r+5, 0, 16, 4), o.Sub(r, 0, 8, 4)
		r5 := archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 0, 0)).ExtendLo8ToUint16().BitsToInt16()
		binary.LittleEndian.PutUint64(unsafeutil.Row[[8]byte](ow, 0, 0)[:], mc8V6(r0, r1, r2, r3, r4, r5, c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0))
		r6 := archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 1, 0)).ExtendLo8ToUint16().BitsToInt16()
		binary.LittleEndian.PutUint64(unsafeutil.Row[[8]byte](ow, 1, 0)[:], mc8V6(r1, r2, r3, r4, r5, r6, c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0))
		r7 := archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 2, 0)).ExtendLo8ToUint16().BitsToInt16()
		binary.LittleEndian.PutUint64(unsafeutil.Row[[8]byte](ow, 2, 0)[:], mc8V6(r2, r3, r4, r5, r6, r7, c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0))
		r8 := archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 3, 0)).ExtendLo8ToUint16().BitsToInt16()
		binary.LittleEndian.PutUint64(unsafeutil.Row[[8]byte](ow, 3, 0)[:], mc8V6(r3, r4, r5, r6, r7, r8, c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0))
		r0, r1, r2, r3, r4 = r4, r5, r6, r7, r8
		r += 4
	}
	if r+2 <= h {
		if !s.Fits(r+5, 0, 16, 2) || !o.Fits(r, 0, 8, 2) {
			return
		}
		sw, ow := s.Sub(r+5, 0, 16, 2), o.Sub(r, 0, 8, 2)
		r5 := archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 0, 0)).ExtendLo8ToUint16().BitsToInt16()
		binary.LittleEndian.PutUint64(unsafeutil.Row[[8]byte](ow, 0, 0)[:], mc8V6(r0, r1, r2, r3, r4, r5, c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0))
		r6 := archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 1, 0)).ExtendLo8ToUint16().BitsToInt16()
		binary.LittleEndian.PutUint64(unsafeutil.Row[[8]byte](ow, 1, 0)[:], mc8V6(r1, r2, r3, r4, r5, r6, c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0))
		r0, r1, r2, r3, r4 = r2, r3, r4, r5, r6
		r += 2
	}
	if r < h {
		if !s.Fits(r+5, 0, 16, 1) || !o.Fits(r, 0, 8, 1) {
			return
		}
		sw, ow := s.Sub(r+5, 0, 16, 1), o.Sub(r, 0, 8, 1)
		r5 := archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 0, 0)).ExtendLo8ToUint16().BitsToInt16()
		binary.LittleEndian.PutUint64(unsafeutil.Row[[8]byte](ow, 0, 0)[:], mc8V6(r0, r1, r2, r3, r4, r5, c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0))
	}
}

// mc8PutV6S4 is mc8PutV6S8 for a four-column block (dav1d's w4 v path):
// s holds its h+5 source rows of sixteen readable bytes, o its h destination rows
// of four samples.
func mc8PutV6S4(s unsafeutil.Grid[uint8], ph *mc8Phase, o unsafeutil.Grid[uint8]) {
	c0 := archsimd.LoadInt16x8Array(&ph.h[0])
	c1 := archsimd.LoadInt16x8Array(&ph.h[1])
	c2 := archsimd.LoadInt16x8Array(&ph.h[2])
	c3 := archsimd.LoadInt16x8Array(&ph.h[3])
	c4 := archsimd.LoadInt16x8Array(&ph.h[4])
	c5 := archsimd.LoadInt16x8Array(&ph.h[5])
	rnd := archsimd.BroadcastInt16x8(32)
	sh := archsimd.BroadcastInt16x8(-6)
	h := o.Height()
	if !s.Fits(0, 0, 16, 5) {
		return
	}
	hd := s.Sub(0, 0, 16, 5)
	r0 := archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](hd, 0, 0)).ExtendLo8ToUint16().BitsToInt16()
	r1 := archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](hd, 1, 0)).ExtendLo8ToUint16().BitsToInt16()
	r2 := archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](hd, 2, 0)).ExtendLo8ToUint16().BitsToInt16()
	r3 := archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](hd, 3, 0)).ExtendLo8ToUint16().BitsToInt16()
	r4 := archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](hd, 4, 0)).ExtendLo8ToUint16().BitsToInt16()
	r := 0
	for ; r+8 <= h; r += 8 {
		if !s.Fits(r+5, 0, 16, 8) || !o.Fits(r, 0, 4, 8) {
			return
		}
		sw, ow := s.Sub(r+5, 0, 16, 8), o.Sub(r, 0, 4, 8)
		r5 := archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 0, 0)).ExtendLo8ToUint16().BitsToInt16()
		binary.LittleEndian.PutUint32(unsafeutil.Row[[4]byte](ow, 0, 0)[:], uint32(mc8V6(r0, r1, r2, r3, r4, r5, c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0)))
		r6 := archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 1, 0)).ExtendLo8ToUint16().BitsToInt16()
		binary.LittleEndian.PutUint32(unsafeutil.Row[[4]byte](ow, 1, 0)[:], uint32(mc8V6(r1, r2, r3, r4, r5, r6, c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0)))
		r7 := archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 2, 0)).ExtendLo8ToUint16().BitsToInt16()
		binary.LittleEndian.PutUint32(unsafeutil.Row[[4]byte](ow, 2, 0)[:], uint32(mc8V6(r2, r3, r4, r5, r6, r7, c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0)))
		r8 := archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 3, 0)).ExtendLo8ToUint16().BitsToInt16()
		binary.LittleEndian.PutUint32(unsafeutil.Row[[4]byte](ow, 3, 0)[:], uint32(mc8V6(r3, r4, r5, r6, r7, r8, c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0)))
		r9 := archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 4, 0)).ExtendLo8ToUint16().BitsToInt16()
		binary.LittleEndian.PutUint32(unsafeutil.Row[[4]byte](ow, 4, 0)[:], uint32(mc8V6(r4, r5, r6, r7, r8, r9, c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0)))
		r10 := archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 5, 0)).ExtendLo8ToUint16().BitsToInt16()
		binary.LittleEndian.PutUint32(unsafeutil.Row[[4]byte](ow, 5, 0)[:], uint32(mc8V6(r5, r6, r7, r8, r9, r10, c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0)))
		r11 := archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 6, 0)).ExtendLo8ToUint16().BitsToInt16()
		binary.LittleEndian.PutUint32(unsafeutil.Row[[4]byte](ow, 6, 0)[:], uint32(mc8V6(r6, r7, r8, r9, r10, r11, c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0)))
		r12 := archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 7, 0)).ExtendLo8ToUint16().BitsToInt16()
		binary.LittleEndian.PutUint32(unsafeutil.Row[[4]byte](ow, 7, 0)[:], uint32(mc8V6(r7, r8, r9, r10, r11, r12, c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0)))
		r0, r1, r2, r3, r4 = r8, r9, r10, r11, r12
	}
	if r+4 <= h {
		if !s.Fits(r+5, 0, 16, 4) || !o.Fits(r, 0, 4, 4) {
			return
		}
		sw, ow := s.Sub(r+5, 0, 16, 4), o.Sub(r, 0, 4, 4)
		r5 := archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 0, 0)).ExtendLo8ToUint16().BitsToInt16()
		binary.LittleEndian.PutUint32(unsafeutil.Row[[4]byte](ow, 0, 0)[:], uint32(mc8V6(r0, r1, r2, r3, r4, r5, c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0)))
		r6 := archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 1, 0)).ExtendLo8ToUint16().BitsToInt16()
		binary.LittleEndian.PutUint32(unsafeutil.Row[[4]byte](ow, 1, 0)[:], uint32(mc8V6(r1, r2, r3, r4, r5, r6, c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0)))
		r7 := archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 2, 0)).ExtendLo8ToUint16().BitsToInt16()
		binary.LittleEndian.PutUint32(unsafeutil.Row[[4]byte](ow, 2, 0)[:], uint32(mc8V6(r2, r3, r4, r5, r6, r7, c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0)))
		r8 := archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 3, 0)).ExtendLo8ToUint16().BitsToInt16()
		binary.LittleEndian.PutUint32(unsafeutil.Row[[4]byte](ow, 3, 0)[:], uint32(mc8V6(r3, r4, r5, r6, r7, r8, c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0)))
		r0, r1, r2, r3, r4 = r4, r5, r6, r7, r8
		r += 4
	}
	if r+2 <= h {
		if !s.Fits(r+5, 0, 16, 2) || !o.Fits(r, 0, 4, 2) {
			return
		}
		sw, ow := s.Sub(r+5, 0, 16, 2), o.Sub(r, 0, 4, 2)
		r5 := archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 0, 0)).ExtendLo8ToUint16().BitsToInt16()
		binary.LittleEndian.PutUint32(unsafeutil.Row[[4]byte](ow, 0, 0)[:], uint32(mc8V6(r0, r1, r2, r3, r4, r5, c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0)))
		r6 := archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 1, 0)).ExtendLo8ToUint16().BitsToInt16()
		binary.LittleEndian.PutUint32(unsafeutil.Row[[4]byte](ow, 1, 0)[:], uint32(mc8V6(r1, r2, r3, r4, r5, r6, c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0)))
		r0, r1, r2, r3, r4 = r2, r3, r4, r5, r6
		r += 2
	}
	if r < h {
		if !s.Fits(r+5, 0, 16, 1) || !o.Fits(r, 0, 4, 1) {
			return
		}
		sw, ow := s.Sub(r+5, 0, 16, 1), o.Sub(r, 0, 4, 1)
		r5 := archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 0, 0)).ExtendLo8ToUint16().BitsToInt16()
		binary.LittleEndian.PutUint32(unsafeutil.Row[[4]byte](ow, 0, 0)[:], uint32(mc8V6(r0, r1, r2, r3, r4, r5, c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0)))
	}
}

// mc8PrepV6W16 is the prep vertical-only pass, ((s + 2) >> 2) + 6144 (the offset sums lie in [0, 2^16) and shift as unsigned),
// over sixteen-column strips: sg holds the h+5 source rows (h = og.Height(),
// og.Width() a multiple of 16) and og the h CONV_BUF rows. Each source
// row is one load for both eight-sample halves.
func mc8PrepV6W16(sg unsafeutil.Grid[uint8], ph *mc8Phase, og unsafeutil.Grid[uint16]) {
	c0 := archsimd.LoadInt16x8Array(&ph.h[0])
	c1 := archsimd.LoadInt16x8Array(&ph.h[1])
	c2 := archsimd.LoadInt16x8Array(&ph.h[2])
	c3 := archsimd.LoadInt16x8Array(&ph.h[3])
	c4 := archsimd.LoadInt16x8Array(&ph.h[4])
	c5 := archsimd.LoadInt16x8Array(&ph.h[5])
	rnd := archsimd.BroadcastInt16x8(2 + 6144<<2)
	sh := archsimd.BroadcastInt16x8(-2)
	h, w := og.Height(), og.Width()
	for c := 0; c+16 <= w; c += 16 {
		if !sg.Fits(0, c, 16, h+5) || !og.Fits(0, c, 16, h) {
			return
		}
		s, o := sg.Sub(0, c, 16, h+5), og.Sub(0, c, 16, h)
		if !s.Fits(0, 0, 16, 5) {
			return
		}
		hd := s.Sub(0, 0, 16, 5)
		x := archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](hd, 0, 0))
		l0, h0 := x.ExtendLo8ToUint16().BitsToInt16(), x.HiToLo().ExtendLo8ToUint16().BitsToInt16()
		x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](hd, 1, 0))
		l1, h1 := x.ExtendLo8ToUint16().BitsToInt16(), x.HiToLo().ExtendLo8ToUint16().BitsToInt16()
		x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](hd, 2, 0))
		l2, h2 := x.ExtendLo8ToUint16().BitsToInt16(), x.HiToLo().ExtendLo8ToUint16().BitsToInt16()
		x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](hd, 3, 0))
		l3, h3 := x.ExtendLo8ToUint16().BitsToInt16(), x.HiToLo().ExtendLo8ToUint16().BitsToInt16()
		x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](hd, 4, 0))
		l4, h4 := x.ExtendLo8ToUint16().BitsToInt16(), x.HiToLo().ExtendLo8ToUint16().BitsToInt16()
		r := 0
		for ; r+8 <= h; r += 8 {
			if !s.Fits(r+5, 0, 16, 8) || !o.Fits(r, 0, 16, 8) {
				return
			}
			sw, ow := s.Sub(r+5, 0, 16, 8), o.Sub(r, 0, 16, 8)
			x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 0, 0))
			l5, h5 := x.ExtendLo8ToUint16().BitsToInt16(), x.HiToLo().ExtendLo8ToUint16().BitsToInt16()
			mc8V6(l0, l1, l2, l3, l4, l5, c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 0, 0))
			mc8V6(h0, h1, h2, h3, h4, h5, c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 0, 8))
			x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 1, 0))
			l6, h6 := x.ExtendLo8ToUint16().BitsToInt16(), x.HiToLo().ExtendLo8ToUint16().BitsToInt16()
			mc8V6(l1, l2, l3, l4, l5, l6, c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 1, 0))
			mc8V6(h1, h2, h3, h4, h5, h6, c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 1, 8))
			x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 2, 0))
			l7, h7 := x.ExtendLo8ToUint16().BitsToInt16(), x.HiToLo().ExtendLo8ToUint16().BitsToInt16()
			mc8V6(l2, l3, l4, l5, l6, l7, c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 2, 0))
			mc8V6(h2, h3, h4, h5, h6, h7, c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 2, 8))
			x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 3, 0))
			l8, h8 := x.ExtendLo8ToUint16().BitsToInt16(), x.HiToLo().ExtendLo8ToUint16().BitsToInt16()
			mc8V6(l3, l4, l5, l6, l7, l8, c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 3, 0))
			mc8V6(h3, h4, h5, h6, h7, h8, c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 3, 8))
			x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 4, 0))
			l9, h9 := x.ExtendLo8ToUint16().BitsToInt16(), x.HiToLo().ExtendLo8ToUint16().BitsToInt16()
			mc8V6(l4, l5, l6, l7, l8, l9, c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 4, 0))
			mc8V6(h4, h5, h6, h7, h8, h9, c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 4, 8))
			x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 5, 0))
			l10, h10 := x.ExtendLo8ToUint16().BitsToInt16(), x.HiToLo().ExtendLo8ToUint16().BitsToInt16()
			mc8V6(l5, l6, l7, l8, l9, l10, c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 5, 0))
			mc8V6(h5, h6, h7, h8, h9, h10, c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 5, 8))
			x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 6, 0))
			l11, h11 := x.ExtendLo8ToUint16().BitsToInt16(), x.HiToLo().ExtendLo8ToUint16().BitsToInt16()
			mc8V6(l6, l7, l8, l9, l10, l11, c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 6, 0))
			mc8V6(h6, h7, h8, h9, h10, h11, c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 6, 8))
			x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 7, 0))
			l12, h12 := x.ExtendLo8ToUint16().BitsToInt16(), x.HiToLo().ExtendLo8ToUint16().BitsToInt16()
			mc8V6(l7, l8, l9, l10, l11, l12, c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 7, 0))
			mc8V6(h7, h8, h9, h10, h11, h12, c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 7, 8))
			l0, h0, l1, h1, l2, h2, l3, h3, l4, h4 = l8, h8, l9, h9, l10, h10, l11, h11, l12, h12
		}
		if r+4 <= h {
			if !s.Fits(r+5, 0, 16, 4) || !o.Fits(r, 0, 16, 4) {
				return
			}
			sw, ow := s.Sub(r+5, 0, 16, 4), o.Sub(r, 0, 16, 4)
			x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 0, 0))
			l5, h5 := x.ExtendLo8ToUint16().BitsToInt16(), x.HiToLo().ExtendLo8ToUint16().BitsToInt16()
			mc8V6(l0, l1, l2, l3, l4, l5, c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 0, 0))
			mc8V6(h0, h1, h2, h3, h4, h5, c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 0, 8))
			x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 1, 0))
			l6, h6 := x.ExtendLo8ToUint16().BitsToInt16(), x.HiToLo().ExtendLo8ToUint16().BitsToInt16()
			mc8V6(l1, l2, l3, l4, l5, l6, c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 1, 0))
			mc8V6(h1, h2, h3, h4, h5, h6, c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 1, 8))
			x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 2, 0))
			l7, h7 := x.ExtendLo8ToUint16().BitsToInt16(), x.HiToLo().ExtendLo8ToUint16().BitsToInt16()
			mc8V6(l2, l3, l4, l5, l6, l7, c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 2, 0))
			mc8V6(h2, h3, h4, h5, h6, h7, c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 2, 8))
			x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 3, 0))
			l8, h8 := x.ExtendLo8ToUint16().BitsToInt16(), x.HiToLo().ExtendLo8ToUint16().BitsToInt16()
			mc8V6(l3, l4, l5, l6, l7, l8, c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 3, 0))
			mc8V6(h3, h4, h5, h6, h7, h8, c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 3, 8))
			l0, h0, l1, h1, l2, h2, l3, h3, l4, h4 = l4, h4, l5, h5, l6, h6, l7, h7, l8, h8
			r += 4
		}
		if r+2 <= h {
			if !s.Fits(r+5, 0, 16, 2) || !o.Fits(r, 0, 16, 2) {
				return
			}
			sw, ow := s.Sub(r+5, 0, 16, 2), o.Sub(r, 0, 16, 2)
			x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 0, 0))
			l5, h5 := x.ExtendLo8ToUint16().BitsToInt16(), x.HiToLo().ExtendLo8ToUint16().BitsToInt16()
			mc8V6(l0, l1, l2, l3, l4, l5, c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 0, 0))
			mc8V6(h0, h1, h2, h3, h4, h5, c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 0, 8))
			x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 1, 0))
			l6, h6 := x.ExtendLo8ToUint16().BitsToInt16(), x.HiToLo().ExtendLo8ToUint16().BitsToInt16()
			mc8V6(l1, l2, l3, l4, l5, l6, c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 1, 0))
			mc8V6(h1, h2, h3, h4, h5, h6, c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 1, 8))
			l0, h0, l1, h1, l2, h2, l3, h3, l4, h4 = l2, h2, l3, h3, l4, h4, l5, h5, l6, h6
			r += 2
		}
		if r < h {
			if !s.Fits(r+5, 0, 16, 1) || !o.Fits(r, 0, 16, 1) {
				return
			}
			sw, ow := s.Sub(r+5, 0, 16, 1), o.Sub(r, 0, 16, 1)
			x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 0, 0))
			l5, h5 := x.ExtendLo8ToUint16().BitsToInt16(), x.HiToLo().ExtendLo8ToUint16().BitsToInt16()
			mc8V6(l0, l1, l2, l3, l4, l5, c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 0, 0))
			mc8V6(h0, h1, h2, h3, h4, h5, c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 0, 8))
		}
	}
}

// mc8PrepV6S8 is mc8PrepV6W16 over one eight-column strip: s holds its
// h+5 source rows of sixteen readable bytes, o its h CONV_BUF rows.
func mc8PrepV6S8(s unsafeutil.Grid[uint8], ph *mc8Phase, o unsafeutil.Grid[uint16]) {
	c0 := archsimd.LoadInt16x8Array(&ph.h[0])
	c1 := archsimd.LoadInt16x8Array(&ph.h[1])
	c2 := archsimd.LoadInt16x8Array(&ph.h[2])
	c3 := archsimd.LoadInt16x8Array(&ph.h[3])
	c4 := archsimd.LoadInt16x8Array(&ph.h[4])
	c5 := archsimd.LoadInt16x8Array(&ph.h[5])
	rnd := archsimd.BroadcastInt16x8(2 + 6144<<2)
	sh := archsimd.BroadcastInt16x8(-2)
	h := o.Height()
	if !s.Fits(0, 0, 16, 5) {
		return
	}
	hd := s.Sub(0, 0, 16, 5)
	r0 := archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](hd, 0, 0)).ExtendLo8ToUint16().BitsToInt16()
	r1 := archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](hd, 1, 0)).ExtendLo8ToUint16().BitsToInt16()
	r2 := archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](hd, 2, 0)).ExtendLo8ToUint16().BitsToInt16()
	r3 := archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](hd, 3, 0)).ExtendLo8ToUint16().BitsToInt16()
	r4 := archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](hd, 4, 0)).ExtendLo8ToUint16().BitsToInt16()
	r := 0
	for ; r+8 <= h; r += 8 {
		if !s.Fits(r+5, 0, 16, 8) || !o.Fits(r, 0, 8, 8) {
			return
		}
		sw, ow := s.Sub(r+5, 0, 16, 8), o.Sub(r, 0, 8, 8)
		r5 := archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 0, 0)).ExtendLo8ToUint16().BitsToInt16()
		mc8V6(r0, r1, r2, r3, r4, r5, c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 0, 0))
		r6 := archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 1, 0)).ExtendLo8ToUint16().BitsToInt16()
		mc8V6(r1, r2, r3, r4, r5, r6, c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 1, 0))
		r7 := archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 2, 0)).ExtendLo8ToUint16().BitsToInt16()
		mc8V6(r2, r3, r4, r5, r6, r7, c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 2, 0))
		r8 := archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 3, 0)).ExtendLo8ToUint16().BitsToInt16()
		mc8V6(r3, r4, r5, r6, r7, r8, c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 3, 0))
		r9 := archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 4, 0)).ExtendLo8ToUint16().BitsToInt16()
		mc8V6(r4, r5, r6, r7, r8, r9, c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 4, 0))
		r10 := archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 5, 0)).ExtendLo8ToUint16().BitsToInt16()
		mc8V6(r5, r6, r7, r8, r9, r10, c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 5, 0))
		r11 := archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 6, 0)).ExtendLo8ToUint16().BitsToInt16()
		mc8V6(r6, r7, r8, r9, r10, r11, c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 6, 0))
		r12 := archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 7, 0)).ExtendLo8ToUint16().BitsToInt16()
		mc8V6(r7, r8, r9, r10, r11, r12, c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 7, 0))
		r0, r1, r2, r3, r4 = r8, r9, r10, r11, r12
	}
	if r+4 <= h {
		if !s.Fits(r+5, 0, 16, 4) || !o.Fits(r, 0, 8, 4) {
			return
		}
		sw, ow := s.Sub(r+5, 0, 16, 4), o.Sub(r, 0, 8, 4)
		r5 := archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 0, 0)).ExtendLo8ToUint16().BitsToInt16()
		mc8V6(r0, r1, r2, r3, r4, r5, c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 0, 0))
		r6 := archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 1, 0)).ExtendLo8ToUint16().BitsToInt16()
		mc8V6(r1, r2, r3, r4, r5, r6, c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 1, 0))
		r7 := archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 2, 0)).ExtendLo8ToUint16().BitsToInt16()
		mc8V6(r2, r3, r4, r5, r6, r7, c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 2, 0))
		r8 := archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 3, 0)).ExtendLo8ToUint16().BitsToInt16()
		mc8V6(r3, r4, r5, r6, r7, r8, c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 3, 0))
		r0, r1, r2, r3, r4 = r4, r5, r6, r7, r8
		r += 4
	}
	if r+2 <= h {
		if !s.Fits(r+5, 0, 16, 2) || !o.Fits(r, 0, 8, 2) {
			return
		}
		sw, ow := s.Sub(r+5, 0, 16, 2), o.Sub(r, 0, 8, 2)
		r5 := archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 0, 0)).ExtendLo8ToUint16().BitsToInt16()
		mc8V6(r0, r1, r2, r3, r4, r5, c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 0, 0))
		r6 := archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 1, 0)).ExtendLo8ToUint16().BitsToInt16()
		mc8V6(r1, r2, r3, r4, r5, r6, c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 1, 0))
		r0, r1, r2, r3, r4 = r2, r3, r4, r5, r6
		r += 2
	}
	if r < h {
		if !s.Fits(r+5, 0, 16, 1) || !o.Fits(r, 0, 8, 1) {
			return
		}
		sw, ow := s.Sub(r+5, 0, 16, 1), o.Sub(r, 0, 8, 1)
		r5 := archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 0, 0)).ExtendLo8ToUint16().BitsToInt16()
		mc8V6(r0, r1, r2, r3, r4, r5, c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 0, 0))
	}
}

// mc8PrepV6S4 is mc8PrepV6S8 for a four-column block (dav1d's w4 v path):
// s holds its h+5 source rows of sixteen readable bytes, o its h CONV_BUF rows
// of four samples.
func mc8PrepV6S4(s unsafeutil.Grid[uint8], ph *mc8Phase, o unsafeutil.Grid[uint16]) {
	c0 := archsimd.LoadInt16x8Array(&ph.h[0])
	c1 := archsimd.LoadInt16x8Array(&ph.h[1])
	c2 := archsimd.LoadInt16x8Array(&ph.h[2])
	c3 := archsimd.LoadInt16x8Array(&ph.h[3])
	c4 := archsimd.LoadInt16x8Array(&ph.h[4])
	c5 := archsimd.LoadInt16x8Array(&ph.h[5])
	rnd := archsimd.BroadcastInt16x8(2 + 6144<<2)
	sh := archsimd.BroadcastInt16x8(-2)
	h := o.Height()
	if !s.Fits(0, 0, 16, 5) {
		return
	}
	hd := s.Sub(0, 0, 16, 5)
	r0 := archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](hd, 0, 0)).ExtendLo8ToUint16().BitsToInt16()
	r1 := archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](hd, 1, 0)).ExtendLo8ToUint16().BitsToInt16()
	r2 := archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](hd, 2, 0)).ExtendLo8ToUint16().BitsToInt16()
	r3 := archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](hd, 3, 0)).ExtendLo8ToUint16().BitsToInt16()
	r4 := archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](hd, 4, 0)).ExtendLo8ToUint16().BitsToInt16()
	r := 0
	for ; r+8 <= h; r += 8 {
		if !s.Fits(r+5, 0, 16, 8) || !o.Fits(r, 0, 4, 8) {
			return
		}
		sw, ow := s.Sub(r+5, 0, 16, 8), o.Sub(r, 0, 4, 8)
		r5 := archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 0, 0)).ExtendLo8ToUint16().BitsToInt16()
		mc8Store4U16(unsafeutil.Row[[4]uint16](ow, 0, 0), mc8V6(r0, r1, r2, r3, r4, r5, c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).ReshapeToUint64s().GetElem(0))
		r6 := archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 1, 0)).ExtendLo8ToUint16().BitsToInt16()
		mc8Store4U16(unsafeutil.Row[[4]uint16](ow, 1, 0), mc8V6(r1, r2, r3, r4, r5, r6, c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).ReshapeToUint64s().GetElem(0))
		r7 := archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 2, 0)).ExtendLo8ToUint16().BitsToInt16()
		mc8Store4U16(unsafeutil.Row[[4]uint16](ow, 2, 0), mc8V6(r2, r3, r4, r5, r6, r7, c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).ReshapeToUint64s().GetElem(0))
		r8 := archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 3, 0)).ExtendLo8ToUint16().BitsToInt16()
		mc8Store4U16(unsafeutil.Row[[4]uint16](ow, 3, 0), mc8V6(r3, r4, r5, r6, r7, r8, c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).ReshapeToUint64s().GetElem(0))
		r9 := archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 4, 0)).ExtendLo8ToUint16().BitsToInt16()
		mc8Store4U16(unsafeutil.Row[[4]uint16](ow, 4, 0), mc8V6(r4, r5, r6, r7, r8, r9, c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).ReshapeToUint64s().GetElem(0))
		r10 := archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 5, 0)).ExtendLo8ToUint16().BitsToInt16()
		mc8Store4U16(unsafeutil.Row[[4]uint16](ow, 5, 0), mc8V6(r5, r6, r7, r8, r9, r10, c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).ReshapeToUint64s().GetElem(0))
		r11 := archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 6, 0)).ExtendLo8ToUint16().BitsToInt16()
		mc8Store4U16(unsafeutil.Row[[4]uint16](ow, 6, 0), mc8V6(r6, r7, r8, r9, r10, r11, c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).ReshapeToUint64s().GetElem(0))
		r12 := archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 7, 0)).ExtendLo8ToUint16().BitsToInt16()
		mc8Store4U16(unsafeutil.Row[[4]uint16](ow, 7, 0), mc8V6(r7, r8, r9, r10, r11, r12, c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).ReshapeToUint64s().GetElem(0))
		r0, r1, r2, r3, r4 = r8, r9, r10, r11, r12
	}
	if r+4 <= h {
		if !s.Fits(r+5, 0, 16, 4) || !o.Fits(r, 0, 4, 4) {
			return
		}
		sw, ow := s.Sub(r+5, 0, 16, 4), o.Sub(r, 0, 4, 4)
		r5 := archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 0, 0)).ExtendLo8ToUint16().BitsToInt16()
		mc8Store4U16(unsafeutil.Row[[4]uint16](ow, 0, 0), mc8V6(r0, r1, r2, r3, r4, r5, c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).ReshapeToUint64s().GetElem(0))
		r6 := archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 1, 0)).ExtendLo8ToUint16().BitsToInt16()
		mc8Store4U16(unsafeutil.Row[[4]uint16](ow, 1, 0), mc8V6(r1, r2, r3, r4, r5, r6, c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).ReshapeToUint64s().GetElem(0))
		r7 := archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 2, 0)).ExtendLo8ToUint16().BitsToInt16()
		mc8Store4U16(unsafeutil.Row[[4]uint16](ow, 2, 0), mc8V6(r2, r3, r4, r5, r6, r7, c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).ReshapeToUint64s().GetElem(0))
		r8 := archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 3, 0)).ExtendLo8ToUint16().BitsToInt16()
		mc8Store4U16(unsafeutil.Row[[4]uint16](ow, 3, 0), mc8V6(r3, r4, r5, r6, r7, r8, c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).ReshapeToUint64s().GetElem(0))
		r0, r1, r2, r3, r4 = r4, r5, r6, r7, r8
		r += 4
	}
	if r+2 <= h {
		if !s.Fits(r+5, 0, 16, 2) || !o.Fits(r, 0, 4, 2) {
			return
		}
		sw, ow := s.Sub(r+5, 0, 16, 2), o.Sub(r, 0, 4, 2)
		r5 := archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 0, 0)).ExtendLo8ToUint16().BitsToInt16()
		mc8Store4U16(unsafeutil.Row[[4]uint16](ow, 0, 0), mc8V6(r0, r1, r2, r3, r4, r5, c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).ReshapeToUint64s().GetElem(0))
		r6 := archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 1, 0)).ExtendLo8ToUint16().BitsToInt16()
		mc8Store4U16(unsafeutil.Row[[4]uint16](ow, 1, 0), mc8V6(r1, r2, r3, r4, r5, r6, c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).ReshapeToUint64s().GetElem(0))
		r0, r1, r2, r3, r4 = r2, r3, r4, r5, r6
		r += 2
	}
	if r < h {
		if !s.Fits(r+5, 0, 16, 1) || !o.Fits(r, 0, 4, 1) {
			return
		}
		sw, ow := s.Sub(r+5, 0, 16, 1), o.Sub(r, 0, 4, 1)
		r5 := archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 0, 0)).ExtendLo8ToUint16().BitsToInt16()
		mc8Store4U16(unsafeutil.Row[[4]uint16](ow, 0, 0), mc8V6(r0, r1, r2, r3, r4, r5, c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).ReshapeToUint64s().GetElem(0))
	}
}
