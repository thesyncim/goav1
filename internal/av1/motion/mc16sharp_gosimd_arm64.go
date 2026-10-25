// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package motion

import (
	"simd/archsimd"
	"unsafe"
)

// Eight-tap 16bpc passes for the sharp filter (dav1d src/arm/64/mc16.S
// filter_fn 8tap: put/prep_8tap_{sharp,regular_sharp,sharp_regular,...}). The
// sharp table is the only one with nonzero outer taps, so it is the only kind
// that needs window taps 0..7. Each axis picks its own pass: a sharp axis runs
// the eight-tap pass below from refX-3 / refY-3, the other axis keeps its
// six-tap pass from refX-2 / refY-2, and the 2D passes meet in the same int32
// intermediate as the six-tap kernels (so the rounding is unchanged: the
// formulas in mc16_gosimd_arm64.go and mc16put_gosimd_arm64.go hold for any
// tap count, sum(tap) = 128). dav1d's 8tap functions filter the non-sharp axis
// with eight taps too (its outer taps zero); doing it in six is the same sum.
//
// The halved sharp taps are at most 62, so a 12-bit horizontal sum is bounded
// by 4095 * 92 and the multiply-widen into int32 is exact; the prep 2D
// vertical pass is exact modulo 2^32 as before.

// mc16Phase8 is one sharp filter phase pre-broadcast for the eight-tap window.
type mc16Phase8 struct {
	h [8][8]int16 // halved taps
	v [8][4]int32 // put 2D vertical: 32 * tap
	p [8][4]int32 // prep 2D vertical: 512 * tap
}

var mc16SharpPhases [16]mc16Phase8

func init() {
	for s := range 16 {
		k := &subpelFilters8Sharp[s]
		ph := &mc16SharpPhases[s]
		for i, c := range k {
			for l := range 8 {
				ph.h[i][l] = c / 2
			}
			for l := range 4 {
				ph.v[i][l] = int32(c) * 32
				ph.p[i][l] = int32(c) * 512
			}
		}
	}
}

// mc16Taps is the pass length of one axis: 0 when the axis is not filtered,
// 8 for the sharp table, 6 for every other kind.
func mc16Taps(sub, kind int) int {
	switch {
	case sub == 0:
		return 0
	case mc8SixTap[kind]:
		return 6
	default:
		return 8
	}
}

// mc16Load8Taps loads the eight broadcast halved taps of a sharp phase.
func mc16Load8Taps(ph *mc16Phase8) (c0, c1, c2, c3, c4, c5, c6, c7 archsimd.Int16x8) {
	return archsimd.LoadInt16x8Array(&ph.h[0]), archsimd.LoadInt16x8Array(&ph.h[1]),
		archsimd.LoadInt16x8Array(&ph.h[2]), archsimd.LoadInt16x8Array(&ph.h[3]),
		archsimd.LoadInt16x8Array(&ph.h[4]), archsimd.LoadInt16x8Array(&ph.h[5]),
		archsimd.LoadInt16x8Array(&ph.h[6]), archsimd.LoadInt16x8Array(&ph.h[7])
}

// mc16H8ToI32 is mc16H6ToI32 over window taps 0..7 (w a multiple of 8).
//
//go:nocheckptr
func mc16H8ToI32(src unsafe.Pointer, srcStride, w, rows int, ph *mc16Phase8, sh int, im unsafe.Pointer) {
	c0, c1, c2, c3, c4, c5, c6, c7 := mc16Load8Taps(ph)
	h0, h1, h2, h3, h4, h5, h6, h7 := c0.HiToLo(), c1.HiToLo(), c2.HiToLo(), c3.HiToLo(), c4.HiToLo(), c5.HiToLo(), c6.HiToLo(), c7.HiToLo()
	rnd := archsimd.BroadcastInt32x4(int32(1) << (sh - 1))
	shv := archsimd.BroadcastInt32x4(int32(-sh))
	imStride := w * 4
	for r := 0; r < rows; r++ {
		s := unsafe.Add(src, r*srcStride)
		o := unsafe.Add(im, r*imStride)
		for col := 0; col < w; col += 8 {
			p := unsafe.Add(s, 2*col)
			x0 := mc16L16(p)
			x1 := mc16L16(unsafe.Add(p, 2))
			x2 := mc16L16(unsafe.Add(p, 4))
			x3 := mc16L16(unsafe.Add(p, 6))
			x4 := mc16L16(unsafe.Add(p, 8))
			x5 := mc16L16(unsafe.Add(p, 10))
			x6 := mc16L16(unsafe.Add(p, 12))
			x7 := mc16L16(unsafe.Add(p, 14))
			lo := rnd.Add(x0.MulWidenLo(c0))
			hi := rnd.Add(x0.HiToLo().MulWidenLo(h0))
			lo = lo.Add(x1.MulWidenLo(c1))
			hi = hi.Add(x1.HiToLo().MulWidenLo(h1))
			lo = lo.Add(x2.MulWidenLo(c2))
			hi = hi.Add(x2.HiToLo().MulWidenLo(h2))
			lo = lo.Add(x3.MulWidenLo(c3))
			hi = hi.Add(x3.HiToLo().MulWidenLo(h3))
			lo = lo.Add(x4.MulWidenLo(c4))
			hi = hi.Add(x4.HiToLo().MulWidenLo(h4))
			lo = lo.Add(x5.MulWidenLo(c5))
			hi = hi.Add(x5.HiToLo().MulWidenLo(h5))
			lo = lo.Add(x6.MulWidenLo(c6))
			hi = hi.Add(x6.HiToLo().MulWidenLo(h6))
			lo = lo.Add(x7.MulWidenLo(c7))
			hi = hi.Add(x7.HiToLo().MulWidenLo(h7))
			q := unsafe.Add(o, col*4)
			lo.Shift(shv).StoreArray((*[4]int32)(q))
			hi.Shift(shv).StoreArray((*[4]int32)(unsafe.Add(q, 16)))
		}
	}
}

// mc16PrepH8 is mc16PrepH6 over window taps 0..7 (w a multiple of 8).
//
//go:nocheckptr
func mc16PrepH8(src unsafe.Pointer, srcStride, w, h int, ph *mc16Phase8, sh int, out unsafe.Pointer) {
	c0, c1, c2, c3, c4, c5, c6, c7 := mc16Load8Taps(ph)
	h0, h1, h2, h3, h4, h5, h6, h7 := c0.HiToLo(), c1.HiToLo(), c2.HiToLo(), c3.HiToLo(), c4.HiToLo(), c5.HiToLo(), c6.HiToLo(), c7.HiToLo()
	rnd := archsimd.BroadcastInt32x4(int32(1)<<(sh-1) + int32(mc16PrepRoundOffset)<<sh)
	shv := archsimd.BroadcastInt32x4(int32(-sh))
	ro := w * 2
	for r := 0; r < h; r++ {
		s := unsafe.Add(src, r*srcStride)
		o := unsafe.Add(out, r*ro)
		for col := 0; col < w; col += 8 {
			p := unsafe.Add(s, 2*col)
			x0 := mc16L16(p)
			x1 := mc16L16(unsafe.Add(p, 2))
			x2 := mc16L16(unsafe.Add(p, 4))
			x3 := mc16L16(unsafe.Add(p, 6))
			x4 := mc16L16(unsafe.Add(p, 8))
			x5 := mc16L16(unsafe.Add(p, 10))
			x6 := mc16L16(unsafe.Add(p, 12))
			x7 := mc16L16(unsafe.Add(p, 14))
			lo := rnd.Add(x0.MulWidenLo(c0))
			hi := rnd.Add(x0.HiToLo().MulWidenLo(h0))
			lo = lo.Add(x1.MulWidenLo(c1))
			hi = hi.Add(x1.HiToLo().MulWidenLo(h1))
			lo = lo.Add(x2.MulWidenLo(c2))
			hi = hi.Add(x2.HiToLo().MulWidenLo(h2))
			lo = lo.Add(x3.MulWidenLo(c3))
			hi = hi.Add(x3.HiToLo().MulWidenLo(h3))
			lo = lo.Add(x4.MulWidenLo(c4))
			hi = hi.Add(x4.HiToLo().MulWidenLo(h4))
			lo = lo.Add(x5.MulWidenLo(c5))
			hi = hi.Add(x5.HiToLo().MulWidenLo(h5))
			lo = lo.Add(x6.MulWidenLo(c6))
			hi = hi.Add(x6.HiToLo().MulWidenLo(h6))
			lo = lo.Add(x7.MulWidenLo(c7))
			hi = hi.Add(x7.HiToLo().MulWidenLo(h7))
			mc16PackU16(lo.Shift(shv), hi.Shift(shv)).StoreArray((*[8]uint16)(unsafe.Add(o, 2*col)))
		}
	}
}

// mc16PutH8 is mc16PutH6 over window taps 0..7 (w a multiple of 8).
//
//go:nocheckptr
func mc16PutH8(src unsafe.Pointer, srcStride, w, h int, ph *mc16Phase8, rndC int32, maxPix int16, dst unsafe.Pointer, dstStride int) {
	c0, c1, c2, c3, c4, c5, c6, c7 := mc16Load8Taps(ph)
	h0, h1, h2, h3, h4, h5, h6, h7 := c0.HiToLo(), c1.HiToLo(), c2.HiToLo(), c3.HiToLo(), c4.HiToLo(), c5.HiToLo(), c6.HiToLo(), c7.HiToLo()
	rnd := archsimd.BroadcastInt32x4(rndC)
	shv := archsimd.BroadcastInt32x4(-6)
	zero := archsimd.BroadcastInt16x8(0)
	maxV := archsimd.BroadcastInt16x8(maxPix)
	for r := 0; r < h; r++ {
		s := unsafe.Add(src, r*srcStride)
		o := unsafe.Add(dst, r*dstStride)
		for col := 0; col < w; col += 8 {
			p := unsafe.Add(s, 2*col)
			x0 := mc16L16(p)
			x1 := mc16L16(unsafe.Add(p, 2))
			x2 := mc16L16(unsafe.Add(p, 4))
			x3 := mc16L16(unsafe.Add(p, 6))
			x4 := mc16L16(unsafe.Add(p, 8))
			x5 := mc16L16(unsafe.Add(p, 10))
			x6 := mc16L16(unsafe.Add(p, 12))
			x7 := mc16L16(unsafe.Add(p, 14))
			lo := rnd.Add(x0.MulWidenLo(c0))
			hi := rnd.Add(x0.HiToLo().MulWidenLo(h0))
			lo = lo.Add(x1.MulWidenLo(c1))
			hi = hi.Add(x1.HiToLo().MulWidenLo(h1))
			lo = lo.Add(x2.MulWidenLo(c2))
			hi = hi.Add(x2.HiToLo().MulWidenLo(h2))
			lo = lo.Add(x3.MulWidenLo(c3))
			hi = hi.Add(x3.HiToLo().MulWidenLo(h3))
			lo = lo.Add(x4.MulWidenLo(c4))
			hi = hi.Add(x4.HiToLo().MulWidenLo(h4))
			lo = lo.Add(x5.MulWidenLo(c5))
			hi = hi.Add(x5.HiToLo().MulWidenLo(h5))
			lo = lo.Add(x6.MulWidenLo(c6))
			hi = hi.Add(x6.HiToLo().MulWidenLo(h6))
			lo = lo.Add(x7.MulWidenLo(c7))
			hi = hi.Add(x7.HiToLo().MulWidenLo(h7))
			mc16Clip(lo.Shift(shv), hi.Shift(shv), zero, maxV).StoreArray((*[8]int16)(unsafe.Add(o, 2*col)))
		}
	}
}

// mc16V8Lo is the eight-tap vertical chain over the low four lanes of the
// rows x0..x7: rnd + sum(half tap * x) as int32.
func mc16V8Lo(x0, x1, x2, x3, x4, x5, x6, x7, c0, c1, c2, c3, c4, c5, c6, c7 archsimd.Int16x8, rnd archsimd.Int32x4) archsimd.Int32x4 {
	a := rnd.Add(x0.MulWidenLo(c0))
	a = a.Add(x1.MulWidenLo(c1))
	a = a.Add(x2.MulWidenLo(c2))
	a = a.Add(x3.MulWidenLo(c3))
	a = a.Add(x4.MulWidenLo(c4))
	a = a.Add(x5.MulWidenLo(c5))
	a = a.Add(x6.MulWidenLo(c6))
	return a.Add(x7.MulWidenLo(c7))
}

// mc16PrepV8 is mc16PrepV6 over window rows 0..7: each column strip slides
// an eight-row window down the block, two output rows per iteration.
//
//go:nocheckptr
func mc16PrepV8(src unsafe.Pointer, srcStride, w, h int, ph *mc16Phase8, sh int, out unsafe.Pointer) {
	c0, c1, c2, c3, c4, c5, c6, c7 := mc16Load8Taps(ph)
	h0, h1, h2, h3, h4, h5, h6, h7 := c0.HiToLo(), c1.HiToLo(), c2.HiToLo(), c3.HiToLo(), c4.HiToLo(), c5.HiToLo(), c6.HiToLo(), c7.HiToLo()
	rnd := archsimd.BroadcastInt32x4(int32(1)<<(sh-1) + int32(mc16PrepRoundOffset)<<sh)
	shv := archsimd.BroadcastInt32x4(int32(-sh))
	ro := w * 2
	if w == 4 {
		// Two output rows share one vector (the CONV_BUF rows are contiguous).
		s := src
		r0 := mc16L16(s)
		r1 := mc16L16(unsafe.Add(s, srcStride))
		r2 := mc16L16(unsafe.Add(s, 2*srcStride))
		r3 := mc16L16(unsafe.Add(s, 3*srcStride))
		r4 := mc16L16(unsafe.Add(s, 4*srcStride))
		r5 := mc16L16(unsafe.Add(s, 5*srcStride))
		r6 := mc16L16(unsafe.Add(s, 6*srcStride))
		s = unsafe.Add(s, 7*srcStride)
		o := out
		r := 0
		for ; r+2 <= h; r += 2 {
			r7 := mc16L16(s)
			r8 := mc16L16(unsafe.Add(s, srcStride))
			a := mc16V8Lo(r0, r1, r2, r3, r4, r5, r6, r7, c0, c1, c2, c3, c4, c5, c6, c7, rnd)
			b := mc16V8Lo(r1, r2, r3, r4, r5, r6, r7, r8, c0, c1, c2, c3, c4, c5, c6, c7, rnd)
			mc16PackU16(a.Shift(shv), b.Shift(shv)).StoreArray((*[8]uint16)(o))
			r0, r1, r2, r3, r4, r5, r6 = r2, r3, r4, r5, r6, r7, r8
			s = unsafe.Add(s, 2*srcStride)
			o = unsafe.Add(o, 2*ro)
		}
		if r < h {
			r7 := mc16L16(s)
			a := mc16V8Lo(r0, r1, r2, r3, r4, r5, r6, r7, c0, c1, c2, c3, c4, c5, c6, c7, rnd).Shift(shv)
			*(*uint64)(o) = mc16PackU16(a, a).ReshapeToUint64s().GetElem(0)
		}
		return
	}
	for col := 0; col < w; col += 8 {
		s := unsafe.Add(src, 2*col)
		o := unsafe.Add(out, 2*col)
		r0 := mc16L16(s)
		r1 := mc16L16(unsafe.Add(s, srcStride))
		r2 := mc16L16(unsafe.Add(s, 2*srcStride))
		r3 := mc16L16(unsafe.Add(s, 3*srcStride))
		r4 := mc16L16(unsafe.Add(s, 4*srcStride))
		r5 := mc16L16(unsafe.Add(s, 5*srcStride))
		r6 := mc16L16(unsafe.Add(s, 6*srcStride))
		s = unsafe.Add(s, 7*srcStride)
		r := 0
		for ; r+2 <= h; r += 2 {
			r7 := mc16L16(s)
			r8 := mc16L16(unsafe.Add(s, srcStride))
			alo := mc16V8Lo(r0, r1, r2, r3, r4, r5, r6, r7, c0, c1, c2, c3, c4, c5, c6, c7, rnd)
			ahi := rnd.Add(r0.HiToLo().MulWidenLo(h0))
			ahi = ahi.Add(r1.HiToLo().MulWidenLo(h1))
			ahi = ahi.Add(r2.HiToLo().MulWidenLo(h2))
			ahi = ahi.Add(r3.HiToLo().MulWidenLo(h3))
			ahi = ahi.Add(r4.HiToLo().MulWidenLo(h4))
			ahi = ahi.Add(r5.HiToLo().MulWidenLo(h5))
			ahi = ahi.Add(r6.HiToLo().MulWidenLo(h6))
			ahi = ahi.Add(r7.HiToLo().MulWidenLo(h7))
			mc16PackU16(alo.Shift(shv), ahi.Shift(shv)).StoreArray((*[8]uint16)(o))
			blo := mc16V8Lo(r1, r2, r3, r4, r5, r6, r7, r8, c0, c1, c2, c3, c4, c5, c6, c7, rnd)
			bhi := rnd.Add(r1.HiToLo().MulWidenLo(h0))
			bhi = bhi.Add(r2.HiToLo().MulWidenLo(h1))
			bhi = bhi.Add(r3.HiToLo().MulWidenLo(h2))
			bhi = bhi.Add(r4.HiToLo().MulWidenLo(h3))
			bhi = bhi.Add(r5.HiToLo().MulWidenLo(h4))
			bhi = bhi.Add(r6.HiToLo().MulWidenLo(h5))
			bhi = bhi.Add(r7.HiToLo().MulWidenLo(h6))
			bhi = bhi.Add(r8.HiToLo().MulWidenLo(h7))
			mc16PackU16(blo.Shift(shv), bhi.Shift(shv)).StoreArray((*[8]uint16)(unsafe.Add(o, ro)))
			r0, r1, r2, r3, r4, r5, r6 = r2, r3, r4, r5, r6, r7, r8
			s = unsafe.Add(s, 2*srcStride)
			o = unsafe.Add(o, 2*ro)
		}
		if r < h {
			r7 := mc16L16(s)
			alo := mc16V8Lo(r0, r1, r2, r3, r4, r5, r6, r7, c0, c1, c2, c3, c4, c5, c6, c7, rnd)
			ahi := rnd.Add(r0.HiToLo().MulWidenLo(h0))
			ahi = ahi.Add(r1.HiToLo().MulWidenLo(h1))
			ahi = ahi.Add(r2.HiToLo().MulWidenLo(h2))
			ahi = ahi.Add(r3.HiToLo().MulWidenLo(h3))
			ahi = ahi.Add(r4.HiToLo().MulWidenLo(h4))
			ahi = ahi.Add(r5.HiToLo().MulWidenLo(h5))
			ahi = ahi.Add(r6.HiToLo().MulWidenLo(h6))
			ahi = ahi.Add(r7.HiToLo().MulWidenLo(h7))
			mc16PackU16(alo.Shift(shv), ahi.Shift(shv)).StoreArray((*[8]uint16)(o))
		}
	}
}

// mc16PutV8 is mc16PutV6 over window rows 0..7.
//
//go:nocheckptr
func mc16PutV8(src unsafe.Pointer, srcStride, w, h int, ph *mc16Phase8, maxPix int16, dst unsafe.Pointer, dstStride int) {
	c0, c1, c2, c3, c4, c5, c6, c7 := mc16Load8Taps(ph)
	h0, h1, h2, h3, h4, h5, h6, h7 := c0.HiToLo(), c1.HiToLo(), c2.HiToLo(), c3.HiToLo(), c4.HiToLo(), c5.HiToLo(), c6.HiToLo(), c7.HiToLo()
	rnd := archsimd.BroadcastInt32x4(32)
	shv := archsimd.BroadcastInt32x4(-6)
	zero := archsimd.BroadcastInt16x8(0)
	maxV := archsimd.BroadcastInt16x8(maxPix)
	if w == 4 {
		s := src
		r0 := mc16L16(s)
		r1 := mc16L16(unsafe.Add(s, srcStride))
		r2 := mc16L16(unsafe.Add(s, 2*srcStride))
		r3 := mc16L16(unsafe.Add(s, 3*srcStride))
		r4 := mc16L16(unsafe.Add(s, 4*srcStride))
		r5 := mc16L16(unsafe.Add(s, 5*srcStride))
		r6 := mc16L16(unsafe.Add(s, 6*srcStride))
		s = unsafe.Add(s, 7*srcStride)
		d := dst
		r := 0
		for ; r+2 <= h; r += 2 {
			r7 := mc16L16(s)
			r8 := mc16L16(unsafe.Add(s, srcStride))
			a := mc16V8Lo(r0, r1, r2, r3, r4, r5, r6, r7, c0, c1, c2, c3, c4, c5, c6, c7, rnd)
			b := mc16V8Lo(r1, r2, r3, r4, r5, r6, r7, r8, c0, c1, c2, c3, c4, c5, c6, c7, rnd)
			v := mc16Clip(a.Shift(shv), b.Shift(shv), zero, maxV).ToBits().ReshapeToUint64s()
			*(*uint64)(d) = v.GetElem(0)
			*(*uint64)(unsafe.Add(d, dstStride)) = v.GetElem(1)
			r0, r1, r2, r3, r4, r5, r6 = r2, r3, r4, r5, r6, r7, r8
			s = unsafe.Add(s, 2*srcStride)
			d = unsafe.Add(d, 2*dstStride)
		}
		if r < h {
			r7 := mc16L16(s)
			a := mc16V8Lo(r0, r1, r2, r3, r4, r5, r6, r7, c0, c1, c2, c3, c4, c5, c6, c7, rnd).Shift(shv)
			*(*uint64)(d) = mc16Clip(a, a, zero, maxV).ToBits().ReshapeToUint64s().GetElem(0)
		}
		return
	}
	for col := 0; col < w; col += 8 {
		s := unsafe.Add(src, 2*col)
		d := unsafe.Add(dst, 2*col)
		r0 := mc16L16(s)
		r1 := mc16L16(unsafe.Add(s, srcStride))
		r2 := mc16L16(unsafe.Add(s, 2*srcStride))
		r3 := mc16L16(unsafe.Add(s, 3*srcStride))
		r4 := mc16L16(unsafe.Add(s, 4*srcStride))
		r5 := mc16L16(unsafe.Add(s, 5*srcStride))
		r6 := mc16L16(unsafe.Add(s, 6*srcStride))
		s = unsafe.Add(s, 7*srcStride)
		r := 0
		for ; r+2 <= h; r += 2 {
			r7 := mc16L16(s)
			r8 := mc16L16(unsafe.Add(s, srcStride))
			alo := mc16V8Lo(r0, r1, r2, r3, r4, r5, r6, r7, c0, c1, c2, c3, c4, c5, c6, c7, rnd)
			ahi := rnd.Add(r0.HiToLo().MulWidenLo(h0))
			ahi = ahi.Add(r1.HiToLo().MulWidenLo(h1))
			ahi = ahi.Add(r2.HiToLo().MulWidenLo(h2))
			ahi = ahi.Add(r3.HiToLo().MulWidenLo(h3))
			ahi = ahi.Add(r4.HiToLo().MulWidenLo(h4))
			ahi = ahi.Add(r5.HiToLo().MulWidenLo(h5))
			ahi = ahi.Add(r6.HiToLo().MulWidenLo(h6))
			ahi = ahi.Add(r7.HiToLo().MulWidenLo(h7))
			mc16Clip(alo.Shift(shv), ahi.Shift(shv), zero, maxV).StoreArray((*[8]int16)(d))
			blo := mc16V8Lo(r1, r2, r3, r4, r5, r6, r7, r8, c0, c1, c2, c3, c4, c5, c6, c7, rnd)
			bhi := rnd.Add(r1.HiToLo().MulWidenLo(h0))
			bhi = bhi.Add(r2.HiToLo().MulWidenLo(h1))
			bhi = bhi.Add(r3.HiToLo().MulWidenLo(h2))
			bhi = bhi.Add(r4.HiToLo().MulWidenLo(h3))
			bhi = bhi.Add(r5.HiToLo().MulWidenLo(h4))
			bhi = bhi.Add(r6.HiToLo().MulWidenLo(h5))
			bhi = bhi.Add(r7.HiToLo().MulWidenLo(h6))
			bhi = bhi.Add(r8.HiToLo().MulWidenLo(h7))
			mc16Clip(blo.Shift(shv), bhi.Shift(shv), zero, maxV).StoreArray((*[8]int16)(unsafe.Add(d, dstStride)))
			r0, r1, r2, r3, r4, r5, r6 = r2, r3, r4, r5, r6, r7, r8
			s = unsafe.Add(s, 2*srcStride)
			d = unsafe.Add(d, 2*dstStride)
		}
		if r < h {
			r7 := mc16L16(s)
			alo := mc16V8Lo(r0, r1, r2, r3, r4, r5, r6, r7, c0, c1, c2, c3, c4, c5, c6, c7, rnd)
			ahi := rnd.Add(r0.HiToLo().MulWidenLo(h0))
			ahi = ahi.Add(r1.HiToLo().MulWidenLo(h1))
			ahi = ahi.Add(r2.HiToLo().MulWidenLo(h2))
			ahi = ahi.Add(r3.HiToLo().MulWidenLo(h3))
			ahi = ahi.Add(r4.HiToLo().MulWidenLo(h4))
			ahi = ahi.Add(r5.HiToLo().MulWidenLo(h5))
			ahi = ahi.Add(r6.HiToLo().MulWidenLo(h6))
			ahi = ahi.Add(r7.HiToLo().MulWidenLo(h7))
			mc16Clip(alo.Shift(shv), ahi.Shift(shv), zero, maxV).StoreArray((*[8]int16)(d))
		}
	}
}

// mc16V8I32 is the eight-tap 2D vertical chain over four int32 intermediate
// lanes of the rows x0..x7: rnd + sum(c * x).
func mc16V8I32(x0, x1, x2, x3, x4, x5, x6, x7, c0, c1, c2, c3, c4, c5, c6, c7, rnd archsimd.Int32x4) archsimd.Int32x4 {
	a := x0.MulAdd(c0, rnd)
	a = x1.MulAdd(c1, a)
	a = x2.MulAdd(c2, a)
	a = x3.MulAdd(c3, a)
	a = x4.MulAdd(c4, a)
	a = x5.MulAdd(c5, a)
	a = x6.MulAdd(c6, a)
	return x7.MulAdd(c7, a)
}

// mc16V8FromI32 runs the eight-tap 2D vertical pass over the int32
// intermediate (row stride w*4 bytes) in strips of four columns, two output
// rows per iteration: the nine-row window stays in registers, as in the
// width-4 six-tap kernels. Each output is the high half of sum(c * m) + rnd;
// a pair of rows packs into one vector (UZP2). put clamps it to [0, maxPix]
// (clip), prep stores it as the CONV_BUF uint16. Output rows are outStride
// bytes apart.
//
//go:nocheckptr
func mc16V8FromI32(im unsafe.Pointer, w, h int, c *[8][4]int32, rndC int32, clip bool, maxPix int16, out unsafe.Pointer, outStride int) {
	c0 := archsimd.LoadInt32x4Array(&c[0])
	c1 := archsimd.LoadInt32x4Array(&c[1])
	c2 := archsimd.LoadInt32x4Array(&c[2])
	c3 := archsimd.LoadInt32x4Array(&c[3])
	c4 := archsimd.LoadInt32x4Array(&c[4])
	c5 := archsimd.LoadInt32x4Array(&c[5])
	c6 := archsimd.LoadInt32x4Array(&c[6])
	c7 := archsimd.LoadInt32x4Array(&c[7])
	rnd := archsimd.BroadcastInt32x4(rndC)
	zero := archsimd.BroadcastInt16x8(0)
	maxV := archsimd.BroadcastInt16x8(maxPix)
	rs := w * 4
	for col := 0; col < w; col += 4 {
		q := unsafe.Add(im, col*4)
		o := unsafe.Add(out, col*2)
		x0 := archsimd.LoadInt32x4Array((*[4]int32)(q))
		x1 := archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Add(q, rs)))
		x2 := archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Add(q, 2*rs)))
		x3 := archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Add(q, 3*rs)))
		x4 := archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Add(q, 4*rs)))
		x5 := archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Add(q, 5*rs)))
		x6 := archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Add(q, 6*rs)))
		q = unsafe.Add(q, 7*rs)
		r := 0
		for ; r+2 <= h; r += 2 {
			x7 := archsimd.LoadInt32x4Array((*[4]int32)(q))
			x8 := archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Add(q, rs)))
			q = unsafe.Add(q, 2*rs)
			a := mc16V8I32(x0, x1, x2, x3, x4, x5, x6, x7, c0, c1, c2, c3, c4, c5, c6, c7, rnd)
			b := mc16V8I32(x1, x2, x3, x4, x5, x6, x7, x8, c0, c1, c2, c3, c4, c5, c6, c7, rnd)
			v := a.ToBits().ReshapeToUint16s().ConcatOdd(b.ToBits().ReshapeToUint16s())
			if clip {
				v = v.BitsToInt16().Max(zero).Min(maxV).ToBits()
			}
			u := v.ReshapeToUint64s()
			*(*uint64)(o) = u.GetElem(0)
			*(*uint64)(unsafe.Add(o, outStride)) = u.GetElem(1)
			o = unsafe.Add(o, 2*outStride)
			x0, x1, x2, x3, x4, x5, x6 = x2, x3, x4, x5, x6, x7, x8
		}
		if r < h {
			x7 := archsimd.LoadInt32x4Array((*[4]int32)(q))
			a := mc16V8I32(x0, x1, x2, x3, x4, x5, x6, x7, c0, c1, c2, c3, c4, c5, c6, c7, rnd)
			v := a.ToBits().ReshapeToUint16s().ConcatOdd(a.ToBits().ReshapeToUint16s())
			if clip {
				v = v.BitsToInt16().Max(zero).Min(maxV).ToBits()
			}
			*(*uint64)(o) = v.ReshapeToUint64s().GetElem(0)
		}
	}
}
