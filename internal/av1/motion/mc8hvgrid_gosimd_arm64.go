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
			win := unsafeutil.Row[[24]byte](s, r, 0)
			a := mc8H6S((*[16]uint8)(win[0:16]), c0, c1, c2, c3, c4, c5, rnd).Shift(sh)
			b := mc8H6S((*[16]uint8)(win[8:24]), c0, c1, c2, c3, c4, c5, rnd).Shift(sh)
			p, q := unsafeutil.Row[[8]int32](m0, r, 0), unsafeutil.Row[[8]int32](m1, r, 0)
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
			a := mc8H6S(unsafeutil.Row[[16]byte](s, r, 0), c0, c1, c2, c3, c4, c5, rnd).Shift(sh)
			p := unsafeutil.Row[[8]int32](m, r, 0)
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
		mc8H6S(unsafeutil.Row[[16]byte](s, r, 0), c0, c1, c2, c3, c4, c5, rnd).Shift(sh).ExtendLo4ToInt32().StoreArray(unsafeutil.Row[[4]int32](m, r, 0))
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
			binary.LittleEndian.PutUint64(unsafeutil.Row[[8]byte](d, 0, 0)[:], u0)
			binary.LittleEndian.PutUint64(unsafeutil.Row[[8]byte](d, 1, 0)[:], u1)
			u0, u1 = mc8V6PutPair(x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8, c0, c1, c2, c3, c4, c5)
			binary.LittleEndian.PutUint64(unsafeutil.Row[[8]byte](d, 2, 0)[:], u0)
			binary.LittleEndian.PutUint64(unsafeutil.Row[[8]byte](d, 3, 0)[:], u1)
			u0, u1 = mc8V6PutPair(x4, y4, x5, y5, x6, y6, x7, y7, x8, y8, x9, y9, x10, y10, c0, c1, c2, c3, c4, c5)
			binary.LittleEndian.PutUint64(unsafeutil.Row[[8]byte](d, 4, 0)[:], u0)
			binary.LittleEndian.PutUint64(unsafeutil.Row[[8]byte](d, 5, 0)[:], u1)
			u0, u1 = mc8V6PutPair(x6, y6, x7, y7, x8, y8, x9, y9, x10, y10, x11, y11, x12, y12, c0, c1, c2, c3, c4, c5)
			binary.LittleEndian.PutUint64(unsafeutil.Row[[8]byte](d, 6, 0)[:], u0)
			binary.LittleEndian.PutUint64(unsafeutil.Row[[8]byte](d, 7, 0)[:], u1)
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
			binary.LittleEndian.PutUint64(unsafeutil.Row[[8]byte](d, 0, 0)[:], u0)
			binary.LittleEndian.PutUint64(unsafeutil.Row[[8]byte](d, 1, 0)[:], u1)
			r += 2
			x0, y0, x1, y1, x2, y2, x3, y3, x4, y4 = x2, y2, x3, y3, x4, y4, x5, y5, x6, y6
		}
		if q <= len(st)-8 {
			if !dc.Fits(r, 0, 8, 1) {
				return
			}
			x5, y5 := mc8LoadI32x8((*[8]int32)(st[q : q+8]))
			u0, _ := mc8V6PutPair(x0, y0, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x5, y5, c0, c1, c2, c3, c4, c5)
			binary.LittleEndian.PutUint64(unsafeutil.Row[[8]byte](dc.Sub(r, 0, 8, 1), 0, 0)[:], u0)
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
			u0.StoreArray(unsafeutil.Row[[8]uint16](o, 0, 0))
			u1.StoreArray(unsafeutil.Row[[8]uint16](o, 1, 0))
			u0, u1 = mc8V6PrepPair(x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8, c0, c1, c2, c3, c4, c5, rnd)
			u0.StoreArray(unsafeutil.Row[[8]uint16](o, 2, 0))
			u1.StoreArray(unsafeutil.Row[[8]uint16](o, 3, 0))
			u0, u1 = mc8V6PrepPair(x4, y4, x5, y5, x6, y6, x7, y7, x8, y8, x9, y9, x10, y10, c0, c1, c2, c3, c4, c5, rnd)
			u0.StoreArray(unsafeutil.Row[[8]uint16](o, 4, 0))
			u1.StoreArray(unsafeutil.Row[[8]uint16](o, 5, 0))
			u0, u1 = mc8V6PrepPair(x6, y6, x7, y7, x8, y8, x9, y9, x10, y10, x11, y11, x12, y12, c0, c1, c2, c3, c4, c5, rnd)
			u0.StoreArray(unsafeutil.Row[[8]uint16](o, 6, 0))
			u1.StoreArray(unsafeutil.Row[[8]uint16](o, 7, 0))
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
			u0.StoreArray(unsafeutil.Row[[8]uint16](o, 0, 0))
			u1.StoreArray(unsafeutil.Row[[8]uint16](o, 1, 0))
			r += 2
			x0, y0, x1, y1, x2, y2, x3, y3, x4, y4 = x2, y2, x3, y3, x4, y4, x5, y5, x6, y6
		}
		if q <= len(st)-8 {
			if !oc.Fits(r, 0, 8, 1) {
				return
			}
			x5, y5 := mc8LoadI32x8((*[8]int32)(st[q : q+8]))
			u0, _ := mc8V6PrepPair(x0, y0, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x5, y5, c0, c1, c2, c3, c4, c5, rnd)
			u0.StoreArray(unsafeutil.Row[[8]uint16](oc.Sub(r, 0, 8, 1), 0, 0))
		}
	}
}

// mc8PutV6FromI32W4 is the width-4 put 2D vertical pass into the 4 x h
// destination view dg. The two output rows' high halves pack into one vector
// (UZP2) that saturates to rows r and r+1.
func mc8PutV6FromI32W4(im []int32, h int, ph *mc8Phase, dg unsafeutil.Grid[uint8]) {
	c0 := archsimd.LoadInt32x4Array(&ph.v[0])
	c1 := archsimd.LoadInt32x4Array(&ph.v[1])
	c2 := archsimd.LoadInt32x4Array(&ph.v[2])
	c3 := archsimd.LoadInt32x4Array(&ph.v[3])
	c4 := archsimd.LoadInt32x4Array(&ph.v[4])
	c5 := archsimd.LoadInt32x4Array(&ph.v[5])
	rnd := archsimd.LoadInt32x4Array(&mc8Vec.w15)
	st := im[: (h+5)*4 : (h+5)*4]
	head := (*[20]int32)(st[0:20])
	x0 := archsimd.LoadInt32x4Array((*[4]int32)(head[0:4]))
	x1 := archsimd.LoadInt32x4Array((*[4]int32)(head[4:8]))
	x2 := archsimd.LoadInt32x4Array((*[4]int32)(head[8:12]))
	x3 := archsimd.LoadInt32x4Array((*[4]int32)(head[12:16]))
	x4 := archsimd.LoadInt32x4Array((*[4]int32)(head[16:20]))
	r, q := 0, 20
	for ; q <= len(st)-8; q += 8 {
		if !dg.Fits(r, 0, 4, 2) {
			return
		}
		d := dg.Sub(r, 0, 4, 2)
		win := (*[8]int32)(st[q : q+8])
		x5 := archsimd.LoadInt32x4Array((*[4]int32)(win[0:4]))
		x6 := archsimd.LoadInt32x4Array((*[4]int32)(win[4:8]))
		a, b := mc8V6I32W4(x0, x1, x2, x3, x4, x5, x6, c0, c1, c2, c3, c4, c5, rnd)
		u := a.ToBits().ReshapeToUint16s().ConcatOdd(b.ToBits().ReshapeToUint16s()).BitsToInt16().SaturateToUint8().ReshapeToUint64s().GetElem(0)
		binary.LittleEndian.PutUint32(unsafeutil.Row[[4]byte](d, 0, 0)[:], uint32(u))
		binary.LittleEndian.PutUint32(unsafeutil.Row[[4]byte](d, 1, 0)[:], uint32(u>>32))
		r += 2
		x0, x1, x2, x3, x4 = x2, x3, x4, x5, x6
	}
	if q <= len(st)-4 {
		if !dg.Fits(r, 0, 4, 1) {
			return
		}
		x5 := archsimd.LoadInt32x4Array((*[4]int32)(st[q : q+4]))
		a, _ := mc8V6I32W4(x0, x1, x2, x3, x4, x5, x5, c0, c1, c2, c3, c4, c5, rnd)
		v := a.ToBits().ReshapeToUint16s().ConcatOdd(a.ToBits().ReshapeToUint16s()).BitsToInt16()
		binary.LittleEndian.PutUint32(unsafeutil.Row[[4]byte](dg.Sub(r, 0, 4, 1), 0, 0)[:], uint32(v.SaturateToUint8().ReshapeToUint64s().GetElem(0)))
	}
}
