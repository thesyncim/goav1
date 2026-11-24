// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package motion

import (
	"encoding/binary"
	"simd/archsimd"

	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/unsafeutil"
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
	dExt := (h-1)*dst.Stride + 2*w
	if dOff+dExt > len(dst.Pix) || dExt > mc8WindowMax {
		return false
	}
	maxV := int16(1)<<bitDepth - 1
	if tx != 8 && ty != 8 {
		if !unsafeutil.GridFit(len(dst.Pix), dOff, dst.Stride, 2*w, h) {
			return false
		}
		sg, ok := mc16SourceGrid(ref, refX, refY, w, h, tx, ty, &scratch.edge16)
		if !ok {
			return false
		}
		dg := unsafeutil.NewGrid(dst.Pix, dOff, dst.Stride, 2*w, h)
		switch {
		case tx != 0 && ty != 0:
			im := scratch.imHBD[:]
			mc16H6ToI32G(sg, w, h+5, &mc8Phases[kx][subX], 2, im)
			if w == 4 {
				mc16PutV6FromI32GW4(im, h, &mc8Phases[ky][subY], maxV, dg)
			} else {
				mc16PutV6FromI32G(im, w, h, &mc8Phases[ky][subY], maxV, dg)
			}
		case tx != 0:
			round0, _ := highBDRoundBits(uint8(bitDepth))
			mc16PutH6G(sg, &mc8Phases[kx][subX], int32(1)<<(round0-2)+32, maxV, dg)
		case w == 4:
			mc16PutV6GW4(sg, &mc8Phases[ky][subY], maxV, dg)
		default:
			mc16PutV6G(sg, &mc8Phases[ky][subY], maxV, dg)
		}
		return true
	}
	return mc16PutSharp(dst, ref, bitDepth, dOff, dExt, refX, refY, w, h, kx, ky, tx, ty, subX, subY, maxV, scratch)
}

// mc16PutSharp is mc16PutFast for blocks with an eight-tap (sharp) axis,
// over the slice source window of mc16Source.
func mc16PutSharp(dst *frame.Plane, ref *frame.Plane, bitDepth, dOff, dExt, refX, refY, w, h, kx, ky, tx, ty, subX, subY int, maxV int16, scratch *ConvolveScratch) bool {
	src, stride, ok := mc16Source(ref, refX, refY, w, h, tx, ty, &scratch.edge16)
	if !ok {
		return false
	}
	d := dst.Pix[dOff : dOff+dExt]
	switch {
	case tx != 0 && ty != 0:
		im := scratch.imHBD[:]
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
			mc16V8FromI32(im, w, h, &mc16SharpPhases[subY].v, 1<<15, true, maxV, d, dst.Stride, nil)
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
func mc16PutH6(src []byte, srcStride, w, h int, ph *mc8Phase, rndC int32, maxPix int16, dst []byte, dstStride int) {
	c0, c1, c2, c3, c4, c5, h0, h1, h2, h3, h4, h5 := mc16LoadTaps6(ph)
	rnd := archsimd.BroadcastInt32x4(rndC)
	shv := archsimd.BroadcastInt32x4(-6)
	zero := archsimd.BroadcastInt16x8(0)
	maxV := archsimd.BroadcastInt16x8(maxPix)
	ss, ds := uint32(srcStride), uint32(dstStride)
	var so, do uint32
	if w == 4 {
		for r := 0; r < h; r++ {
			lo := mc16H6Lo(mc16W26(src, so), c0, c1, c2, c3, c4, c5, h0, h1, h2, h3, h4, h5, rnd).Shift(shv)
			binary.LittleEndian.PutUint64(mc8W8(dst, do)[:], mc16Clip(lo, lo, zero, maxV).ToBits().ReshapeToUint64s().GetElem(0))
			so, do = so+ss, do+ds
		}
		return
	}
	for r := 0; r < h; r++ {
		for col := uint32(0); col < uint32(2*w); col += 16 {
			hw := mc16W26(src, so+col)
			x0 := mc16LB((*[16]byte)(hw[0:16]))
			x1 := mc16LB((*[16]byte)(hw[2:18]))
			x2 := mc16LB((*[16]byte)(hw[4:20]))
			x3 := mc16LB((*[16]byte)(hw[6:22]))
			x4 := mc16LB((*[16]byte)(hw[8:24]))
			x5 := mc16LB((*[16]byte)(hw[10:26]))
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
			mc16St16(dst, do+col, mc16Clip(lo.Shift(shv), hi.Shift(shv), zero, maxV))
		}
		so, do = so+ss, do+ds
	}
}

// mc16PutV6 is the put vertical-only pass: clip((hs + 32) >> 6), sliding a
// six-row window down each column strip, two output rows per iteration.
func mc16PutV6(src []byte, srcStride, w, h int, ph *mc8Phase, maxPix int16, dst []byte, dstStride int) {
	c0, c1, c2, c3, c4, c5, h0, h1, h2, h3, h4, h5 := mc16LoadTaps6(ph)
	rnd := archsimd.BroadcastInt32x4(32)
	shv := archsimd.BroadcastInt32x4(-6)
	zero := archsimd.BroadcastInt16x8(0)
	maxV := archsimd.BroadcastInt16x8(maxPix)
	ss, ds := uint32(srcStride), uint32(dstStride)
	if w == 4 {
		r0 := mc16L16(src, 0)
		r1 := mc16L16(src, ss)
		r2 := mc16L16(src, 2*ss)
		r3 := mc16L16(src, 3*ss)
		so := 4 * ss
		r4 := mc16L16(src, so)
		var d uint32
		r := 0
		for ; r+2 <= h; r += 2 {
			r5 := mc16L16(src, so+ss)
			so += 2 * ss
			r6 := mc16L16(src, so)
			a := mc16V6Lo(r0, r1, r2, r3, r4, r5, c0, c1, c2, c3, c4, c5, h0, h1, h2, h3, h4, h5, rnd)
			b := mc16V6Lo(r1, r2, r3, r4, r5, r6, c0, c1, c2, c3, c4, c5, h0, h1, h2, h3, h4, h5, rnd)
			v := mc16Clip(a.Shift(shv), b.Shift(shv), zero, maxV).ToBits().ReshapeToUint64s()
			binary.LittleEndian.PutUint64(mc8W8(dst, d)[:], v.GetElem(0))
			binary.LittleEndian.PutUint64(mc8W8(dst, d+ds)[:], v.GetElem(1))
			d += 2 * ds
			r0, r1, r2, r3, r4 = r2, r3, r4, r5, r6
		}
		if r < h {
			r5 := mc16L16(src, so+ss)
			a := mc16V6Lo(r0, r1, r2, r3, r4, r5, c0, c1, c2, c3, c4, c5, h0, h1, h2, h3, h4, h5, rnd).Shift(shv)
			binary.LittleEndian.PutUint64(mc8W8(dst, d)[:], mc16Clip(a, a, zero, maxV).ToBits().ReshapeToUint64s().GetElem(0))
		}
		return
	}
	for col := uint32(0); col < uint32(2*w); col += 16 {
		so := col
		r0 := mc16L16(src, so)
		r1 := mc16L16(src, so+ss)
		r2 := mc16L16(src, so+2*ss)
		r3 := mc16L16(src, so+3*ss)
		so += 4 * ss
		r4 := mc16L16(src, so)
		d := col
		r := 0
		for ; r+2 <= h; r += 2 {
			r5 := mc16L16(src, so+ss)
			so += 2 * ss
			r6 := mc16L16(src, so)
			alo, ahi := mc16V6(r0, r1, r2, r3, r4, r5, c0, c1, c2, c3, c4, c5, h0, h1, h2, h3, h4, h5, rnd)
			blo, bhi := mc16V6(r1, r2, r3, r4, r5, r6, c0, c1, c2, c3, c4, c5, h0, h1, h2, h3, h4, h5, rnd)
			mc16St16(dst, d, mc16Clip(alo.Shift(shv), ahi.Shift(shv), zero, maxV))
			mc16St16(dst, d+ds, mc16Clip(blo.Shift(shv), bhi.Shift(shv), zero, maxV))
			d += 2 * ds
			r0, r1, r2, r3, r4 = r2, r3, r4, r5, r6
		}
		if r < h {
			r5 := mc16L16(src, so+ss)
			alo, ahi := mc16V6(r0, r1, r2, r3, r4, r5, c0, c1, c2, c3, c4, c5, h0, h1, h2, h3, h4, h5, rnd)
			mc16St16(dst, d, mc16Clip(alo.Shift(shv), ahi.Shift(shv), zero, maxV))
		}
	}
}

// mc16V6PutPair is the 10-bit put 2D vertical chain of output rows r and r+1
// from the seven strip-major intermediate rows r..r+6, clamped to
// [0, maxV].
func mc16V6PutPair(x0, y0, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, c0, c1, c2, c3, c4, c5, rnd archsimd.Int32x4, zero, maxV archsimd.Int16x8) (archsimd.Int16x8, archsimd.Int16x8) {
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
	v0 := a0.ToBits().ReshapeToUint16s().ConcatOdd(b0.ToBits().ReshapeToUint16s()).BitsToInt16().Max(zero).Min(maxV)
	v1 := a1.ToBits().ReshapeToUint16s().ConcatOdd(b1.ToBits().ReshapeToUint16s()).BitsToInt16().Max(zero).Min(maxV)
	return v0, v1
}

// mc16PutV6FromI32 is the 10-bit put 2D vertical pass over the strip-major
// int32 intermediate (mc8PutV6FromI32 with a 16bpc clip and store).
func mc16PutV6FromI32(im []int32, w, h int, ph *mc8Phase, maxPix int16, dst []byte, dstStride int) {
	c0 := archsimd.LoadInt32x4Array(&ph.v[0])
	c1 := archsimd.LoadInt32x4Array(&ph.v[1])
	c2 := archsimd.LoadInt32x4Array(&ph.v[2])
	c3 := archsimd.LoadInt32x4Array(&ph.v[3])
	c4 := archsimd.LoadInt32x4Array(&ph.v[4])
	c5 := archsimd.LoadInt32x4Array(&ph.v[5])
	rnd := archsimd.BroadcastInt32x4(1 << 15)
	zero := archsimd.BroadcastInt16x8(0)
	maxV := archsimd.BroadcastInt16x8(maxPix)
	ds := uint32(dstStride)
	rows := h + 5
	n := rows * 8
	for c := 0; c < w; c += 8 {
		st := im[c*rows : c*rows+n : c*rows+n]
		head := (*[40]int32)(st[0:40])
		x0, y0 := mc8LoadI32x8((*[8]int32)(head[0:8]))
		x1, y1 := mc8LoadI32x8((*[8]int32)(head[8:16]))
		x2, y2 := mc8LoadI32x8((*[8]int32)(head[16:24]))
		x3, y3 := mc8LoadI32x8((*[8]int32)(head[24:32]))
		x4, y4 := mc8LoadI32x8((*[8]int32)(head[32:40]))
		d := uint32(2 * c)
		q := 40
		for ; q <= len(st)-16; q += 16 {
			win := (*[16]int32)(st[q : q+16])
			x5, y5 := mc8LoadI32x8((*[8]int32)(win[0:8]))
			x6, y6 := mc8LoadI32x8((*[8]int32)(win[8:16]))
			v0, v1 := mc16V6PutPair(x0, y0, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, c0, c1, c2, c3, c4, c5, rnd, zero, maxV)
			mc16St16(dst, d, v0)
			mc16St16(dst, d+ds, v1)
			d += 2 * ds
			x0, y0, x1, y1, x2, y2, x3, y3, x4, y4 = x2, y2, x3, y3, x4, y4, x5, y5, x6, y6
		}
		if q <= len(st)-8 {
			x5, y5 := mc8LoadI32x8((*[8]int32)(st[q : q+8]))
			v0, _ := mc16V6PutPair(x0, y0, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x5, y5, c0, c1, c2, c3, c4, c5, rnd, zero, maxV)
			mc16St16(dst, d, v0)
		}
	}
}

// mc16PutV6FromI32W4 is the width-4 form: two output rows per vector (the
// intermediate rows of a width-4 block are contiguous).
func mc16PutV6FromI32W4(im []int32, h int, ph *mc8Phase, maxPix int16, dst []byte, dstStride int) {
	c0 := archsimd.LoadInt32x4Array(&ph.v[0])
	c1 := archsimd.LoadInt32x4Array(&ph.v[1])
	c2 := archsimd.LoadInt32x4Array(&ph.v[2])
	c3 := archsimd.LoadInt32x4Array(&ph.v[3])
	c4 := archsimd.LoadInt32x4Array(&ph.v[4])
	c5 := archsimd.LoadInt32x4Array(&ph.v[5])
	rnd := archsimd.BroadcastInt32x4(1 << 15)
	zero := archsimd.BroadcastInt16x8(0)
	maxV := archsimd.BroadcastInt16x8(maxPix)
	st := im[: (h+5)*4 : (h+5)*4]
	head := (*[20]int32)(st[0:20])
	x0 := archsimd.LoadInt32x4Array((*[4]int32)(head[0:4]))
	x1 := archsimd.LoadInt32x4Array((*[4]int32)(head[4:8]))
	x2 := archsimd.LoadInt32x4Array((*[4]int32)(head[8:12]))
	x3 := archsimd.LoadInt32x4Array((*[4]int32)(head[12:16]))
	x4 := archsimd.LoadInt32x4Array((*[4]int32)(head[16:20]))
	ds := uint32(dstStride)
	var d uint32
	q := 20
	for ; q <= len(st)-8; q += 8 {
		win := (*[8]int32)(st[q : q+8])
		x5 := archsimd.LoadInt32x4Array((*[4]int32)(win[0:4]))
		x6 := archsimd.LoadInt32x4Array((*[4]int32)(win[4:8]))
		a, b := mc8V6I32W4(x0, x1, x2, x3, x4, x5, x6, c0, c1, c2, c3, c4, c5, rnd)
		v := a.ToBits().ReshapeToUint16s().ConcatOdd(b.ToBits().ReshapeToUint16s()).BitsToInt16().Max(zero).Min(maxV).ToBits().ReshapeToUint64s()
		binary.LittleEndian.PutUint64(mc8W8(dst, d)[:], v.GetElem(0))
		binary.LittleEndian.PutUint64(mc8W8(dst, d+ds)[:], v.GetElem(1))
		d += 2 * ds
		x0, x1, x2, x3, x4 = x2, x3, x4, x5, x6
	}
	if q <= len(st)-4 {
		x5 := archsimd.LoadInt32x4Array((*[4]int32)(st[q : q+4]))
		a, _ := mc8V6I32W4(x0, x1, x2, x3, x4, x5, x5, c0, c1, c2, c3, c4, c5, rnd)
		v := a.ToBits().ReshapeToUint16s().ConcatOdd(a.ToBits().ReshapeToUint16s()).BitsToInt16().Max(zero).Min(maxV)
		binary.LittleEndian.PutUint64(mc8W8(dst, d)[:], v.ToBits().ReshapeToUint64s().GetElem(0))
	}
}
