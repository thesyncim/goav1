// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package motion

import (
	"simd/archsimd"

	"github.com/thesyncim/goav1/internal/unsafeutil"
)

// The single-pass and copy kernels index footprint-checked unsafeutil.Grid
// views. Each kernel takes fixed-shape chunks of its views (Sub with
// constant widths and row counts: two compares per chunk) and names the
// chunk's rows by constant indices, which the compiler proves in range; the
// vertical kernels run eight output rows per chunk, so the five resident
// source rows rotate once per eight rows instead of once per two.
//
// Every row access is written on the line of the vector load or store that
// uses it (and every inlined helper call on the line of a vector operation),
// so the inlining marks attach to those instructions instead of costing a
// NOP each.
//
// The entries validate every view (unsafeutil.GridFit, falling back when it
// fails). The kernels restate what they rely on with Grid.Fits and return
// early when it does not hold, which the entries make unreachable: a test
// with no panic behind it leaves the kernels leaf functions (no frame), and
// the compiler drops the repeated test inside each Sub.

// mc8CopyStrips is the put copy (dav1d put_neon) for the widths
// unsafeutil.CopyBlock leaves to its caller (no
// AV1 block has one).
func mc8CopyStrips(dg, sg unsafeutil.Grid[uint8]) bool {
	h, n := dg.Height(), dg.Width()
	if !sg.Fits(0, 0, n, h) {
		return false
	}
	for c := 0; c+4 <= n; c += 4 {
		d, s := dg.Sub(0, c, 4, h), sg.Sub(0, c, 4, h)
		for r := 0; r < h; r++ {
			*unsafeutil.Row[[4]byte](d, r, 0) = *unsafeutil.Row[[4]byte](s, r, 0)
		}
	}
	return true
}

// mc8V6 is the six-tap vertical sum of one output row from rows r0..r5, plus
// rnd.
func mc8V6(r0, r1, r2, r3, r4, r5, c0, c1, c2, c3, c4, c5, rnd archsimd.Int16x8) archsimd.Int16x8 {
	a := r0.MulAdd(c0, rnd)
	a = r1.MulAdd(c1, a)
	a = r2.MulAdd(c2, a)
	a = r3.MulAdd(c3, a)
	a = r4.MulAdd(c4, a)
	return r5.MulAdd(c5, a)
}

// mc8U8 widens the low eight of sixteen loaded bytes to int16 lanes.
func mc8U8(a archsimd.Uint8x16) archsimd.Int16x8 {
	return a.ExtendLo8ToUint16().BitsToInt16()
}

// mc8U8Hi widens the high eight of sixteen loaded bytes to int16 lanes.
func mc8U8Hi(a archsimd.Uint8x16) archsimd.Int16x8 {
	return a.HiToLo().ExtendLo8ToUint16().BitsToInt16()
}

// mc8PrepV6 is the prep vertical-only pass, ((s + 2) >> 2) + 6144, over the
// source window sg (h+5 rows, each readable for 16 bytes from every
// eight-column strip) into the h-row CONV_BUF og (w a multiple of 8).
// Sixteen-column strips take one load per source row for both halves.
func mc8PrepV6(sg unsafeutil.Grid[uint8], ph *mc8Phase, og unsafeutil.Grid[uint16]) {
	h, w := og.Height(), og.Width()
	c := 0
	if w >= 16 {
		c0 := archsimd.LoadInt16x8Array(&ph.h[0])
		c1 := archsimd.LoadInt16x8Array(&ph.h[1])
		c2 := archsimd.LoadInt16x8Array(&ph.h[2])
		c3 := archsimd.LoadInt16x8Array(&ph.h[3])
		c4 := archsimd.LoadInt16x8Array(&ph.h[4])
		c5 := archsimd.LoadInt16x8Array(&ph.h[5])
		rnd := archsimd.BroadcastInt16x8(2 + 6144<<2)
		sh := archsimd.BroadcastInt16x8(-2)
		for ; c+16 <= w; c += 16 {
			s, o := sg.Sub(0, c, 16, h+5), og.Sub(0, c, 16, h)
			x := archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](s, 0, 0))
			l0, h0 := mc8U8(x), mc8U8Hi(x)
			x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](s, 1, 0))
			l1, h1 := mc8U8(x), mc8U8Hi(x)
			x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](s, 2, 0))
			l2, h2 := mc8U8(x), mc8U8Hi(x)
			x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](s, 3, 0))
			l3, h3 := mc8U8(x), mc8U8Hi(x)
			x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](s, 4, 0))
			l4, h4 := mc8U8(x), mc8U8Hi(x)
			r := 0
			for ; r+8 <= h; r += 8 {
				sw, ow := s.Sub(r+5, 0, 16, 8), o.Sub(r, 0, 16, 8)
				x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 0, 0))
				l5, h5 := mc8U8(x), mc8U8Hi(x)
				mc8V6(l0, l1, l2, l3, l4, l5, c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 0, 0))
				mc8V6(h0, h1, h2, h3, h4, h5, c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 0, 8))
				x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 1, 0))
				l6, h6 := mc8U8(x), mc8U8Hi(x)
				mc8V6(l1, l2, l3, l4, l5, l6, c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 1, 0))
				mc8V6(h1, h2, h3, h4, h5, h6, c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 1, 8))
				x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 2, 0))
				l7, h7 := mc8U8(x), mc8U8Hi(x)
				mc8V6(l2, l3, l4, l5, l6, l7, c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 2, 0))
				mc8V6(h2, h3, h4, h5, h6, h7, c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 2, 8))
				x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 3, 0))
				l8, h8 := mc8U8(x), mc8U8Hi(x)
				mc8V6(l3, l4, l5, l6, l7, l8, c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 3, 0))
				mc8V6(h3, h4, h5, h6, h7, h8, c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 3, 8))
				x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 4, 0))
				l9, h9 := mc8U8(x), mc8U8Hi(x)
				mc8V6(l4, l5, l6, l7, l8, l9, c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 4, 0))
				mc8V6(h4, h5, h6, h7, h8, h9, c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 4, 8))
				x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 5, 0))
				l10, h10 := mc8U8(x), mc8U8Hi(x)
				mc8V6(l5, l6, l7, l8, l9, l10, c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 5, 0))
				mc8V6(h5, h6, h7, h8, h9, h10, c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 5, 8))
				x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 6, 0))
				l11, h11 := mc8U8(x), mc8U8Hi(x)
				mc8V6(l6, l7, l8, l9, l10, l11, c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 6, 0))
				mc8V6(h6, h7, h8, h9, h10, h11, c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 6, 8))
				x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 7, 0))
				l12, h12 := mc8U8(x), mc8U8Hi(x)
				mc8V6(l7, l8, l9, l10, l11, l12, c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 7, 0))
				mc8V6(h7, h8, h9, h10, h11, h12, c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 7, 8))
				l0, h0, l1, h1, l2, h2, l3, h3, l4, h4 = l8, h8, l9, h9, l10, h10, l11, h11, l12, h12
			}
			for ; r < h; r++ {
				sw, ow := s.Sub(r+5, 0, 16, 1), o.Sub(r, 0, 16, 1)
				x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 0, 0))
				l5, h5 := mc8U8(x), mc8U8Hi(x)
				mc8V6(l0, l1, l2, l3, l4, l5, c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 0, 0))
				mc8V6(h0, h1, h2, h3, h4, h5, c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 0, 8))
				l0, h0, l1, h1, l2, h2, l3, h3, l4, h4 = l1, h1, l2, h2, l3, h3, l4, h4, l5, h5
			}
		}
		if c == w {
			return
		}
	}
	mc8PrepV6S8(sg.Sub(0, c, 16, h+5), ph, og.Sub(0, c, 8, h))
}

// mc8PrepV6S8 is mc8PrepV6 over one eight-column strip: s holds its h+5
// source rows of sixteen readable bytes, o its h CONV_BUF rows.
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
	hd := s.Sub(0, 0, 16, 5)
	r0 := mc8U8(archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](hd, 0, 0)))
	r1 := mc8U8(archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](hd, 1, 0)))
	r2 := mc8U8(archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](hd, 2, 0)))
	r3 := mc8U8(archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](hd, 3, 0)))
	r4 := mc8U8(archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](hd, 4, 0)))
	r := 0
	for ; r+8 <= h; r += 8 {
		sw, ow := s.Sub(r+5, 0, 16, 8), o.Sub(r, 0, 8, 8)
		r5 := mc8U8(archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 0, 0)))
		mc8V6(r0, r1, r2, r3, r4, r5, c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 0, 0))
		r6 := mc8U8(archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 1, 0)))
		mc8V6(r1, r2, r3, r4, r5, r6, c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 1, 0))
		r7 := mc8U8(archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 2, 0)))
		mc8V6(r2, r3, r4, r5, r6, r7, c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 2, 0))
		r8 := mc8U8(archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 3, 0)))
		mc8V6(r3, r4, r5, r6, r7, r8, c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 3, 0))
		r9 := mc8U8(archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 4, 0)))
		mc8V6(r4, r5, r6, r7, r8, r9, c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 4, 0))
		r10 := mc8U8(archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 5, 0)))
		mc8V6(r5, r6, r7, r8, r9, r10, c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 5, 0))
		r11 := mc8U8(archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 6, 0)))
		mc8V6(r6, r7, r8, r9, r10, r11, c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 6, 0))
		r12 := mc8U8(archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 7, 0)))
		mc8V6(r7, r8, r9, r10, r11, r12, c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 7, 0))
		r0, r1, r2, r3, r4 = r8, r9, r10, r11, r12
	}
	for ; r < h; r++ {
		sw, ow := s.Sub(r+5, 0, 16, 1), o.Sub(r, 0, 8, 1)
		r5 := mc8U8(archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sw, 0, 0)))
		mc8V6(r0, r1, r2, r3, r4, r5, c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, 0, 0))
		r0, r1, r2, r3, r4 = r1, r2, r3, r4, r5
	}
}
