// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package motion

import (
	"simd/archsimd"
	"unsafe"

	"github.com/thesyncim/goav1/internal/av1/frame"
)

// dav1d-style 16bpc single-reference put (src/mc_tmpl.c put_8tap,
// src/arm/64/mc16.S put_6tap / filter_fn), the put sibling of the 16bpc prep in
// mc16_gosimd_arm64.go. Same six-tap kinds, source window and SMULL/SMULL2
// widening; the arithmetic follows the libaom-exact convolve{X,Y,2D}HighBD
// formulas (highBDRoundBits: round0 = 3 and round1 = 11 at 10 bits, 5 and 9 at
// 12) on the halved taps hs:
//
//	put H:  clip((hs + 2^(round0-2) + 32) >> 6)   (round0 then the 7-round0 bits shift, nested)
//	put V:  clip((hs + 32) >> 6)
//	put 2D: m = (hs + 2^(round0-2)) >> (round0-1)  (as prep)
//	        clip((sum(32 * tap * m) + 2^15) >> 16)  (10-bit; libaom's offset terms cancel)
//
// The 2D vertical pass is mc8PutV6FromI32's register window with the 32x
// taps of mc8Phase.v, so it serves 10-bit only (12-bit's round1 = 9 would need
// 128x taps); 12-bit 2D blocks keep the general path. Clipping is to
// [0, 2^bd - 1] in int16 lanes.

// mc16PutFast predicts a w x h block of a 10- or 12-bit plane with the
// dav1d-style put kernels. It returns false for shapes it does not cover.
func mc16PutFast(dst *frame.Plane, ref *frame.Plane, bitDepth, dstX, dstY, refX, refY, w, h, filterW, filterH, subX, subY int, filters InterpFilters, scratch *ConvolveScratch) bool {
	if subX|subY == 0 || (bitDepth != 10 && bitDepth != 12) || (w&7 != 0 && w != 4) || w > maxBlockSize || h <= 0 || h > maxBlockSize || scratch == nil {
		return false
	}
	if subX != 0 && subY != 0 && bitDepth != 10 {
		return false
	}
	kx := mc8Kind(filters.X, filterW)
	ky := mc8Kind(filters.Y, filterH)
	tx, ty := mc16Taps(subX, kx), mc16Taps(subY, ky)
	if tx == 8 && w == 4 {
		return false
	}
	if dstX < 0 || dstY < 0 || dstX+w > dst.Width || dstY+h > dst.Height || dst.Stride < 2*dst.Width {
		return false
	}
	dOff := dstY*dst.Stride + 2*dstX
	if dOff+(h-1)*dst.Stride+2*w > len(dst.Pix) {
		return false
	}
	src, stride, ok := mc16Source(ref, refX, refY, w, h, tx, ty, &scratch.edge16)
	if !ok {
		return false
	}
	d := unsafe.Pointer(&dst.Pix[dOff])
	maxV := int16(1)<<bitDepth - 1
	switch {
	case tx != 0 && ty != 0:
		im := unsafe.Pointer(&scratch.imHBD[0])
		rows := h + ty - 1
		switch {
		case tx == 8:
			mc16H8ToI32(src, stride, w, rows, &mc16SharpPhases[subX], 2, im)
		case w == 4:
			mc16H6ToI32W4(src, stride, rows, &mc8Phases[kx][subX], 2, im)
		default:
			mc16H6ToI32(src, stride, w, rows, &mc8Phases[kx][subX], 2, im)
		}
		switch {
		case ty == 8:
			mc16V8FromI32(im, w, h, &mc16SharpPhases[subY].v, 1<<15, true, maxV, d, dst.Stride)
		case w == 4:
			mc16PutV6FromI32W4(im, h, &mc8Phases[ky][subY], maxV, d, dst.Stride)
		default:
			mc16PutV6FromI32(im, w, h, &mc8Phases[ky][subY], maxV, d, dst.Stride)
		}
	case tx != 0:
		round0, _ := highBDRoundBits(uint8(bitDepth))
		if tx == 8 {
			mc16PutH8(src, stride, w, h, &mc16SharpPhases[subX], int32(1)<<(round0-2)+32, maxV, d, dst.Stride)
		} else {
			mc16PutH6(src, stride, w, h, &mc8Phases[kx][subX], int32(1)<<(round0-2)+32, maxV, d, dst.Stride)
		}
	case ty == 8:
		mc16PutV8(src, stride, w, h, &mc16SharpPhases[subY], maxV, d, dst.Stride)
	default:
		mc16PutV6(src, stride, w, h, &mc8Phases[ky][subY], maxV, d, dst.Stride)
	}
	return true
}

// mc16Clip packs the int32 lanes of lo then hi (already shifted into int16
// range) and clamps them to [0, maxV].
func mc16Clip(lo, hi archsimd.Int32x4, zero, maxV archsimd.Int16x8) archsimd.Int16x8 {
	return mc16PackU16(lo, hi).BitsToInt16().Max(zero).Min(maxV)
}

// mc16PutH6 is the put horizontal-only pass: clip((hs + rnd) >> 6).
//
//go:nocheckptr
func mc16PutH6(src unsafe.Pointer, srcStride, w, h int, ph *mc8Phase, rndC int32, maxPix int16, dst unsafe.Pointer, dstStride int) {
	c0 := archsimd.LoadInt16x8Array(&ph.h[0])
	c1 := archsimd.LoadInt16x8Array(&ph.h[1])
	c2 := archsimd.LoadInt16x8Array(&ph.h[2])
	c3 := archsimd.LoadInt16x8Array(&ph.h[3])
	c4 := archsimd.LoadInt16x8Array(&ph.h[4])
	c5 := archsimd.LoadInt16x8Array(&ph.h[5])
	h0 := c0.HiToLo()
	h1 := c1.HiToLo()
	h2 := c2.HiToLo()
	h3 := c3.HiToLo()
	h4 := c4.HiToLo()
	h5 := c5.HiToLo()
	rnd := archsimd.BroadcastInt32x4(rndC)
	shv := archsimd.BroadcastInt32x4(-6)
	zero := archsimd.BroadcastInt16x8(0)
	maxV := archsimd.BroadcastInt16x8(maxPix)
	if w == 4 {
		for r := 0; r < h; r++ {
			p := unsafe.Add(src, r*srcStride)
			lo := rnd.Add(mc16L16(p).MulWidenLo(c0))
			lo = lo.Add(mc16L16(unsafe.Add(p, 2)).MulWidenLo(c1))
			lo = lo.Add(mc16L16(unsafe.Add(p, 4)).MulWidenLo(c2))
			lo = lo.Add(mc16L16(unsafe.Add(p, 6)).MulWidenLo(c3))
			lo = lo.Add(mc16L16(unsafe.Add(p, 8)).MulWidenLo(c4))
			lo = lo.Add(mc16L16(unsafe.Add(p, 10)).MulWidenLo(c5))
			lo = lo.Shift(shv)
			*(*uint64)(unsafe.Add(dst, r*dstStride)) = mc16Clip(lo, lo, zero, maxV).ToBits().ReshapeToUint64s().GetElem(0)
		}
		return
	}
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
			mc16Clip(lo.Shift(shv), hi.Shift(shv), zero, maxV).StoreArray((*[8]int16)(unsafe.Add(o, 2*col)))
		}
	}
}

// mc16PutV6 is the put vertical-only pass: clip((hs + 32) >> 6), sliding a
// six-row window down each column strip, two output rows per iteration.
//
//go:nocheckptr
func mc16PutV6(src unsafe.Pointer, srcStride, w, h int, ph *mc8Phase, maxPix int16, dst unsafe.Pointer, dstStride int) {
	c0 := archsimd.LoadInt16x8Array(&ph.h[0])
	c1 := archsimd.LoadInt16x8Array(&ph.h[1])
	c2 := archsimd.LoadInt16x8Array(&ph.h[2])
	c3 := archsimd.LoadInt16x8Array(&ph.h[3])
	c4 := archsimd.LoadInt16x8Array(&ph.h[4])
	c5 := archsimd.LoadInt16x8Array(&ph.h[5])
	h0 := c0.HiToLo()
	h1 := c1.HiToLo()
	h2 := c2.HiToLo()
	h3 := c3.HiToLo()
	h4 := c4.HiToLo()
	h5 := c5.HiToLo()
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
		s = unsafe.Add(s, 5*srcStride)
		d := dst
		r := 0
		for ; r+2 <= h; r += 2 {
			r5 := mc16L16(s)
			r6 := mc16L16(unsafe.Add(s, srcStride))
			a := rnd.Add(r0.MulWidenLo(c0))
			a = a.Add(r1.MulWidenLo(c1))
			a = a.Add(r2.MulWidenLo(c2))
			a = a.Add(r3.MulWidenLo(c3))
			a = a.Add(r4.MulWidenLo(c4))
			a = a.Add(r5.MulWidenLo(c5))
			b := rnd.Add(r1.MulWidenLo(c0))
			b = b.Add(r2.MulWidenLo(c1))
			b = b.Add(r3.MulWidenLo(c2))
			b = b.Add(r4.MulWidenLo(c3))
			b = b.Add(r5.MulWidenLo(c4))
			b = b.Add(r6.MulWidenLo(c5))
			v := mc16Clip(a.Shift(shv), b.Shift(shv), zero, maxV).ToBits().ReshapeToUint64s()
			*(*uint64)(d) = v.GetElem(0)
			*(*uint64)(unsafe.Add(d, dstStride)) = v.GetElem(1)
			r0, r1, r2, r3, r4 = r2, r3, r4, r5, r6
			s = unsafe.Add(s, 2*srcStride)
			d = unsafe.Add(d, 2*dstStride)
		}
		if r < h {
			r5 := mc16L16(s)
			a := rnd.Add(r0.MulWidenLo(c0))
			a = a.Add(r1.MulWidenLo(c1))
			a = a.Add(r2.MulWidenLo(c2))
			a = a.Add(r3.MulWidenLo(c3))
			a = a.Add(r4.MulWidenLo(c4))
			a = a.Add(r5.MulWidenLo(c5))
			a = a.Shift(shv)
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
		s = unsafe.Add(s, 5*srcStride)
		r := 0
		for ; r+2 <= h; r += 2 {
			r5 := mc16L16(s)
			r6 := mc16L16(unsafe.Add(s, srcStride))
			alo := rnd.Add(r0.MulWidenLo(c0))
			ahi := rnd.Add(r0.HiToLo().MulWidenLo(h0))
			alo = alo.Add(r1.MulWidenLo(c1))
			ahi = ahi.Add(r1.HiToLo().MulWidenLo(h1))
			alo = alo.Add(r2.MulWidenLo(c2))
			ahi = ahi.Add(r2.HiToLo().MulWidenLo(h2))
			alo = alo.Add(r3.MulWidenLo(c3))
			ahi = ahi.Add(r3.HiToLo().MulWidenLo(h3))
			alo = alo.Add(r4.MulWidenLo(c4))
			ahi = ahi.Add(r4.HiToLo().MulWidenLo(h4))
			alo = alo.Add(r5.MulWidenLo(c5))
			ahi = ahi.Add(r5.HiToLo().MulWidenLo(h5))
			blo := rnd.Add(r1.MulWidenLo(c0))
			bhi := rnd.Add(r1.HiToLo().MulWidenLo(h0))
			blo = blo.Add(r2.MulWidenLo(c1))
			bhi = bhi.Add(r2.HiToLo().MulWidenLo(h1))
			blo = blo.Add(r3.MulWidenLo(c2))
			bhi = bhi.Add(r3.HiToLo().MulWidenLo(h2))
			blo = blo.Add(r4.MulWidenLo(c3))
			bhi = bhi.Add(r4.HiToLo().MulWidenLo(h3))
			blo = blo.Add(r5.MulWidenLo(c4))
			bhi = bhi.Add(r5.HiToLo().MulWidenLo(h4))
			blo = blo.Add(r6.MulWidenLo(c5))
			bhi = bhi.Add(r6.HiToLo().MulWidenLo(h5))
			mc16Clip(alo.Shift(shv), ahi.Shift(shv), zero, maxV).StoreArray((*[8]int16)(d))
			mc16Clip(blo.Shift(shv), bhi.Shift(shv), zero, maxV).StoreArray((*[8]int16)(unsafe.Add(d, dstStride)))
			r0, r1, r2, r3, r4 = r2, r3, r4, r5, r6
			s = unsafe.Add(s, 2*srcStride)
			d = unsafe.Add(d, 2*dstStride)
		}
		if r < h {
			r5 := mc16L16(s)
			alo := rnd.Add(r0.MulWidenLo(c0))
			ahi := rnd.Add(r0.HiToLo().MulWidenLo(h0))
			alo = alo.Add(r1.MulWidenLo(c1))
			ahi = ahi.Add(r1.HiToLo().MulWidenLo(h1))
			alo = alo.Add(r2.MulWidenLo(c2))
			ahi = ahi.Add(r2.HiToLo().MulWidenLo(h2))
			alo = alo.Add(r3.MulWidenLo(c3))
			ahi = ahi.Add(r3.HiToLo().MulWidenLo(h3))
			alo = alo.Add(r4.MulWidenLo(c4))
			ahi = ahi.Add(r4.HiToLo().MulWidenLo(h4))
			alo = alo.Add(r5.MulWidenLo(c5))
			ahi = ahi.Add(r5.HiToLo().MulWidenLo(h5))
			mc16Clip(alo.Shift(shv), ahi.Shift(shv), zero, maxV).StoreArray((*[8]int16)(d))
		}
	}
}

// mc16PutV6FromI32 is the 10-bit put 2D vertical pass over the int32
// intermediate (mc8PutV6FromI32 with a 16bpc clip and store).
//
//go:nocheckptr
func mc16PutV6FromI32(im unsafe.Pointer, w, h int, ph *mc8Phase, maxPix int16, dst unsafe.Pointer, dstStride int) {
	c0 := archsimd.LoadInt32x4Array(&ph.v[0])
	c1 := archsimd.LoadInt32x4Array(&ph.v[1])
	c2 := archsimd.LoadInt32x4Array(&ph.v[2])
	c3 := archsimd.LoadInt32x4Array(&ph.v[3])
	c4 := archsimd.LoadInt32x4Array(&ph.v[4])
	c5 := archsimd.LoadInt32x4Array(&ph.v[5])
	rnd := archsimd.BroadcastInt32x4(1 << 15)
	zero := archsimd.BroadcastInt16x8(0)
	maxV := archsimd.BroadcastInt16x8(maxPix)
	rs := w * 4
	for c := 0; c < w; c += 8 {
		q := unsafe.Add(im, c*4)
		d := unsafe.Add(dst, 2*c)
		x0, y0 := archsimd.LoadInt32x4Array((*[4]int32)(q)), archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Add(q, 16)))
		q = unsafe.Add(q, rs)
		x1, y1 := archsimd.LoadInt32x4Array((*[4]int32)(q)), archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Add(q, 16)))
		q = unsafe.Add(q, rs)
		x2, y2 := archsimd.LoadInt32x4Array((*[4]int32)(q)), archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Add(q, 16)))
		q = unsafe.Add(q, rs)
		x3, y3 := archsimd.LoadInt32x4Array((*[4]int32)(q)), archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Add(q, 16)))
		q = unsafe.Add(q, rs)
		x4, y4 := archsimd.LoadInt32x4Array((*[4]int32)(q)), archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Add(q, 16)))
		q = unsafe.Add(q, rs)
		r := 0
		for ; r+2 <= h; r += 2 {
			x5, y5 := archsimd.LoadInt32x4Array((*[4]int32)(q)), archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Add(q, 16)))
			q = unsafe.Add(q, rs)
			x6, y6 := archsimd.LoadInt32x4Array((*[4]int32)(q)), archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Add(q, 16)))
			q = unsafe.Add(q, rs)
			a0 := x0.MulAdd(c0, rnd)
			b0 := y0.MulAdd(c0, rnd)
			a0 = x1.MulAdd(c1, a0)
			b0 = y1.MulAdd(c1, b0)
			a0 = x2.MulAdd(c2, a0)
			b0 = y2.MulAdd(c2, b0)
			a0 = x3.MulAdd(c3, a0)
			b0 = y3.MulAdd(c3, b0)
			a0 = x4.MulAdd(c4, a0)
			b0 = y4.MulAdd(c4, b0)
			a0 = x5.MulAdd(c5, a0)
			b0 = y5.MulAdd(c5, b0)
			a1 := x1.MulAdd(c0, rnd)
			b1 := y1.MulAdd(c0, rnd)
			a1 = x2.MulAdd(c1, a1)
			b1 = y2.MulAdd(c1, b1)
			a1 = x3.MulAdd(c2, a1)
			b1 = y3.MulAdd(c2, b1)
			a1 = x4.MulAdd(c3, a1)
			b1 = y4.MulAdd(c3, b1)
			a1 = x5.MulAdd(c4, a1)
			b1 = y5.MulAdd(c4, b1)
			a1 = x6.MulAdd(c5, a1)
			b1 = y6.MulAdd(c5, b1)
			a0.ToBits().ReshapeToUint16s().ConcatOdd(b0.ToBits().ReshapeToUint16s()).BitsToInt16().Max(zero).Min(maxV).StoreArray((*[8]int16)(d))
			a1.ToBits().ReshapeToUint16s().ConcatOdd(b1.ToBits().ReshapeToUint16s()).BitsToInt16().Max(zero).Min(maxV).StoreArray((*[8]int16)(unsafe.Add(d, dstStride)))
			d = unsafe.Add(d, 2*dstStride)
			x0, y0, x1, y1, x2, y2, x3, y3, x4, y4 = x2, y2, x3, y3, x4, y4, x5, y5, x6, y6
		}
		if r < h {
			x5, y5 := archsimd.LoadInt32x4Array((*[4]int32)(q)), archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Add(q, 16)))
			a0 := x0.MulAdd(c0, rnd)
			b0 := y0.MulAdd(c0, rnd)
			a0 = x1.MulAdd(c1, a0)
			b0 = y1.MulAdd(c1, b0)
			a0 = x2.MulAdd(c2, a0)
			b0 = y2.MulAdd(c2, b0)
			a0 = x3.MulAdd(c3, a0)
			b0 = y3.MulAdd(c3, b0)
			a0 = x4.MulAdd(c4, a0)
			b0 = y4.MulAdd(c4, b0)
			a0 = x5.MulAdd(c5, a0)
			b0 = y5.MulAdd(c5, b0)
			a0.ToBits().ReshapeToUint16s().ConcatOdd(b0.ToBits().ReshapeToUint16s()).BitsToInt16().Max(zero).Min(maxV).StoreArray((*[8]int16)(d))
		}
	}
}

// mc16PutV6FromI32W4 is the width-4 form: two output rows per vector (the
// intermediate rows of a width-4 block are contiguous).
//
//go:nocheckptr
func mc16PutV6FromI32W4(im unsafe.Pointer, h int, ph *mc8Phase, maxPix int16, dst unsafe.Pointer, dstStride int) {
	c0 := archsimd.LoadInt32x4Array(&ph.v[0])
	c1 := archsimd.LoadInt32x4Array(&ph.v[1])
	c2 := archsimd.LoadInt32x4Array(&ph.v[2])
	c3 := archsimd.LoadInt32x4Array(&ph.v[3])
	c4 := archsimd.LoadInt32x4Array(&ph.v[4])
	c5 := archsimd.LoadInt32x4Array(&ph.v[5])
	rnd := archsimd.BroadcastInt32x4(1 << 15)
	zero := archsimd.BroadcastInt16x8(0)
	maxV := archsimd.BroadcastInt16x8(maxPix)
	x0 := archsimd.LoadInt32x4Array((*[4]int32)(im))
	x1 := archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Add(im, 16)))
	x2 := archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Add(im, 32)))
	x3 := archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Add(im, 48)))
	x4 := archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Add(im, 64)))
	q := unsafe.Add(im, 80)
	d := dst
	r := 0
	for ; r+2 <= h; r += 2 {
		x5 := archsimd.LoadInt32x4Array((*[4]int32)(q))
		x6 := archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Add(q, 16)))
		q = unsafe.Add(q, 32)
		a, b := mc8V6I32W4(x0, x1, x2, x3, x4, x5, x6, c0, c1, c2, c3, c4, c5, rnd)
		v := a.ToBits().ReshapeToUint16s().ConcatOdd(b.ToBits().ReshapeToUint16s()).BitsToInt16().Max(zero).Min(maxV).ToBits().ReshapeToUint64s()
		*(*uint64)(d) = v.GetElem(0)
		*(*uint64)(unsafe.Add(d, dstStride)) = v.GetElem(1)
		d = unsafe.Add(d, 2*dstStride)
		x0, x1, x2, x3, x4 = x2, x3, x4, x5, x6
	}
	if r < h {
		x5 := archsimd.LoadInt32x4Array((*[4]int32)(q))
		a, _ := mc8V6I32W4(x0, x1, x2, x3, x4, x5, x5, c0, c1, c2, c3, c4, c5, rnd)
		v := a.ToBits().ReshapeToUint16s().ConcatOdd(a.ToBits().ReshapeToUint16s()).BitsToInt16().Max(zero).Min(maxV)
		*(*uint64)(d) = v.ToBits().ReshapeToUint64s().GetElem(0)
	}
}
