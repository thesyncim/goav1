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
	maxV := int16(1)<<bitDepth - 1
	if tx == 8 || ty == 8 {
		if dstX < 0 || dstY < 0 || dstX+w > dst.Width || dstY+h > dst.Height || dst.Stride < 2*dst.Width {
			return false
		}
		dOff := dstY*dst.Stride + 2*dstX
		dExt := (h-1)*dst.Stride + 2*w
		if dOff+dExt > len(dst.Pix) || dExt > mc8WindowMax {
			return false
		}
		return mc16PutSharp(dst, ref, bitDepth, dOff, dExt, refX, refY, w, h, kx, ky, tx, ty, subX, subY, maxV, scratch)
	}
	// The six-tap source window (mc16SourceGrid), in place when it lies
	// inside the plane. As in Put8, the in-plane case checks both views in
	// line and runs its kernel with no call before it; mc16PutEdge takes the
	// rest.
	hx, hy := mc8Halo(subX), mc8Halo(subY)
	x0, y0, bh := refX-hx, refY-hy, h+hy+hy+hy>>1
	dOff, sOff, read := dstY*dst.Stride+2*dstX, y0*ref.Stride+2*x0, 2*max(w+5, 13)
	if dstX < 0 || dstY < 0 || dstX+w > dst.Width || dstY+h > dst.Height || dst.Stride < 2*dst.Width ||
		x0 < 0 || y0 < 0 || x0+w+hx+hx+hx>>1 > ref.Width || y0+bh > ref.Height || ref.Stride < 2*ref.Width ||
		!unsafeutil.GridFit(len(dst.Pix), dOff, dst.Stride, 2*w, h) || !unsafeutil.GridFit(len(ref.Pix), sOff, ref.Stride, read, bh) {
		return mc16PutEdge(dst, ref, bitDepth, dstX, dstY, refX, refY, w, h, kx, ky, tx, ty, subX, subY, maxV, scratch)
	}
	dg, sg := unsafeutil.NewGrid(dst.Pix, dOff, dst.Stride, 2*w, h), unsafeutil.NewGrid(ref.Pix, sOff, ref.Stride, read, bh)
	// mc16PutGrid written out, as in Put8.
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

// mc16PutEdge is mc16PutFast's six-tap path for a destination or source
// window that is not inside its plane (or views that do not fit): the
// checked destination view and the source window from mc16SourceGrid, which
// clamps it.
func mc16PutEdge(dst *frame.Plane, ref *frame.Plane, bitDepth, dstX, dstY, refX, refY, w, h, kx, ky, tx, ty, subX, subY int, maxV int16, scratch *ConvolveScratch) bool {
	dOff := dstY*dst.Stride + 2*dstX
	if dstX < 0 || dstY < 0 || dstX+w > dst.Width || dstY+h > dst.Height || dst.Stride < 2*dst.Width ||
		!unsafeutil.GridFit(len(dst.Pix), dOff, dst.Stride, 2*w, h) {
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
	case tx == 8:
		round0, _ := highBDRoundBits(uint8(bitDepth))
		mc16PutH8(src, stride, w, h, &mc16SharpPhases[subX], int32(1)<<(round0-2)+32, maxV, d, dst.Stride)
	default: // ty == 8
		mc16PutV8(src, stride, w, h, &mc16SharpPhases[subY], maxV, d, dst.Stride)
	}
	return true
}

// mc16Clip packs the int32 lanes of lo then hi (already shifted into int16
// range) and clamps them to [0, maxV].
func mc16Clip(lo, hi archsimd.Int32x4, zero, maxV archsimd.Int16x8) archsimd.Int16x8 {
	return mc16PackU16(lo, hi).BitsToInt16().Max(zero).Min(maxV)
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
