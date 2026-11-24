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

// The 16bpc six-tap horizontal, 2D and copy kernels over Grid views (the
// vertical ones are in mc16v6grid_gosimd_arm64.go). A horizontal sum of eight
// outputs reads samples 0..12 of its window as two 16-byte loads (samples
// 0..7 and 5..12), widens them once to four int32 vectors, and forms the six
// tap windows from those by lane shifts (EXT), where the slice-based kernels
// loaded six overlapping windows and widened every product: the same integer
// sums with eight fewer operations per eight outputs. Every view is sized by
// the entries; the Fits tests restate it and cannot fail, so the kernels stay
// leaf functions.

// mc16Ext32 is int32 lanes k..k+3 (n = 4k bytes) of the eight lanes a:b.
func mc16Ext32(a, b archsimd.Uint32x4, n uint64) archsimd.Int32x4 {
	return b.ReshapeToUint8s().ConcatShiftBytesRight(a.ReshapeToUint8s(), n).ReshapeToUint32s().BitsToInt32()
}

// mc16H6E is the six-tap halved horizontal sum, plus rnd, of the eight
// outputs whose 13-sample window is w (samples 0..7 at bytes 0..15, samples
// 5..12 at bytes 10..25), as int32 halves lo (outputs 0..3) and hi (4..7).
func mc16H6E(w *[26]byte, c0, c1, c2, c3, c4, c5, rnd archsimd.Int32x4) (lo, hi archsimd.Int32x4) {
	a := archsimd.LoadUint8x16Array((*[16]byte)(w[0:16])).ReshapeToUint16s()
	b := archsimd.LoadUint8x16Array((*[16]byte)(w[10:26])).ReshapeToUint16s()
	a0, a1 := a.ExtendLo4ToUint32(), a.HiToLo().ExtendLo4ToUint32()
	b0, b1 := b.ExtendLo4ToUint32(), b.HiToLo().ExtendLo4ToUint32()
	lo = a0.BitsToInt32().MulAdd(c0, rnd)
	hi = a1.BitsToInt32().MulAdd(c0, rnd)
	lo = mc16Ext32(a0, a1, 4).MulAdd(c1, lo)
	hi = b0.BitsToInt32().MulAdd(c1, hi)
	lo = mc16Ext32(a0, a1, 8).MulAdd(c2, lo)
	hi = mc16Ext32(b0, b1, 4).MulAdd(c2, hi)
	lo = mc16Ext32(a0, a1, 12).MulAdd(c3, lo)
	hi = mc16Ext32(b0, b1, 8).MulAdd(c3, hi)
	lo = a1.BitsToInt32().MulAdd(c4, lo)
	hi = mc16Ext32(b0, b1, 12).MulAdd(c4, hi)
	lo = b0.BitsToInt32().MulAdd(c5, lo)
	hi = b1.BitsToInt32().MulAdd(c5, hi)
	return lo, hi
}

// mc16H6ELo is mc16H6E's low half alone (four outputs).
func mc16H6ELo(w *[26]byte, c0, c1, c2, c3, c4, c5, rnd archsimd.Int32x4) archsimd.Int32x4 {
	a := archsimd.LoadUint8x16Array((*[16]byte)(w[0:16])).ReshapeToUint16s()
	b := archsimd.LoadUint8x16Array((*[16]byte)(w[10:26])).ReshapeToUint16s()
	a0, a1 := a.ExtendLo4ToUint32(), a.HiToLo().ExtendLo4ToUint32()
	lo := a0.BitsToInt32().MulAdd(c0, rnd)
	lo = mc16Ext32(a0, a1, 4).MulAdd(c1, lo)
	lo = mc16Ext32(a0, a1, 8).MulAdd(c2, lo)
	lo = mc16Ext32(a0, a1, 12).MulAdd(c3, lo)
	lo = a1.BitsToInt32().MulAdd(c4, lo)
	return b.ExtendLo4ToUint32().BitsToInt32().MulAdd(c5, lo)
}

// mc16LoadTaps32 loads the six broadcast halved taps as int32 lanes.
func mc16LoadTaps32(ph *mc8Phase) (c0, c1, c2, c3, c4, c5 archsimd.Int32x4) {
	return archsimd.LoadInt32x4Array(&ph.w[0]), archsimd.LoadInt32x4Array(&ph.w[1]), archsimd.LoadInt32x4Array(&ph.w[2]),
		archsimd.LoadInt32x4Array(&ph.w[3]), archsimd.LoadInt32x4Array(&ph.w[4]), archsimd.LoadInt32x4Array(&ph.w[5])
}

// mc16PutH6G is the 16bpc put horizontal-only pass, clip((hs + rndC) >> 6),
// from the source view sg (h rows of 2*max(w+5, 13) bytes) onto dg (h rows of
// 2*w bytes; w = dg.Width()/2 is 4 or a multiple of 8).
func mc16PutH6G(sg unsafeutil.Grid[uint8], ph *mc8Phase, rndC int32, maxPix int16, dg unsafeutil.Grid[uint8]) {
	c0, c1, c2, c3, c4, c5 := mc16LoadTaps32(ph)
	rnd := archsimd.BroadcastInt32x4(rndC)
	shv := archsimd.LoadInt32x4Array(&mc8Vec.wsh6)
	zero := archsimd.BroadcastInt16x8(0)
	maxV := archsimd.BroadcastInt16x8(maxPix)
	h, w := dg.Height(), dg.Width()/2
	if w == 4 {
		if !sg.Fits(0, 0, 26, h) || !dg.Fits(0, 0, 8, h) {
			return
		}
		s, o := sg.Sub(0, 0, 26, h), dg.Sub(0, 0, 8, h)
		for r := 0; r < h; r++ {
			lo := mc16H6ELo(unsafeutil.Row[[26]byte](s, r, 0), c0, c1, c2, c3, c4, c5, rnd).Shift(shv)
			binary.LittleEndian.PutUint64(unsafeutil.Row[[8]byte](o, r, 0)[:], mc16Clip(lo, lo, zero, maxV).ToBits().ReshapeToUint64s().GetElem(0))
		}
		return
	}
	for c := 0; c+8 <= w; c += 8 {
		if !sg.Fits(0, 2*c, 26, h) || !dg.Fits(0, 2*c, 16, h) {
			return
		}
		s, o := sg.Sub(0, 2*c, 26, h), dg.Sub(0, 2*c, 16, h)
		for r := 0; r < h; r++ {
			lo, hi := mc16H6E(unsafeutil.Row[[26]byte](s, r, 0), c0, c1, c2, c3, c4, c5, rnd)
			mc16Clip(lo.Shift(shv), hi.Shift(shv), zero, maxV).ToBits().ReshapeToUint8s().StoreArray(unsafeutil.Row[[16]byte](o, r, 0))
		}
	}
}

// mc16PrepH6G is the 16bpc prep horizontal-only pass, ((hs + 2^(sh-1)) >> sh)
// + roundOffset with the offset folded into the rounding constant (a multiple
// of 2^sh commutes with the floor), into the CONV_BUF view og.
func mc16PrepH6G(sg unsafeutil.Grid[uint8], ph *mc8Phase, sh int, og unsafeutil.Grid[uint16]) {
	c0, c1, c2, c3, c4, c5 := mc16LoadTaps32(ph)
	rnd := archsimd.BroadcastInt32x4(int32(1)<<(sh-1) + int32(mc16PrepRoundOffset)<<sh)
	shv := archsimd.BroadcastInt32x4(int32(-sh))
	h, w := og.Height(), og.Width()
	if w == 4 {
		if !sg.Fits(0, 0, 26, h) {
			return
		}
		s := sg.Sub(0, 0, 26, h)
		for r := 0; r < h; r++ {
			lo := mc16H6ELo(unsafeutil.Row[[26]byte](s, r, 0), c0, c1, c2, c3, c4, c5, rnd).Shift(shv)
			mc8Store4U16(unsafeutil.Row[[4]uint16](og, r, 0), mc16PackU16(lo, lo).ReshapeToUint64s().GetElem(0))
		}
		return
	}
	for c := 0; c+8 <= w; c += 8 {
		if !sg.Fits(0, 2*c, 26, h) || !og.Fits(0, c, 8, h) {
			return
		}
		s, o := sg.Sub(0, 2*c, 26, h), og.Sub(0, c, 8, h)
		for r := 0; r < h; r++ {
			lo, hi := mc16H6E(unsafeutil.Row[[26]byte](s, r, 0), c0, c1, c2, c3, c4, c5, rnd)
			mc16PackU16(lo.Shift(shv), hi.Shift(shv)).StoreArray(unsafeutil.Row[[8]uint16](o, r, 0))
		}
	}
}

// mc16H6ToI32G writes the rows x w rounded horizontal sums m =
// (hs + 2^(sh-1)) >> sh of the source view sg into the strip-major int32
// intermediate (a width-4 block is one strip of four).
func mc16H6ToI32G(sg unsafeutil.Grid[uint8], w, rows int, ph *mc8Phase, sh int, im []int32) {
	c0, c1, c2, c3, c4, c5 := mc16LoadTaps32(ph)
	rnd := archsimd.BroadcastInt32x4(int32(1) << (sh - 1))
	shv := archsimd.BroadcastInt32x4(int32(-sh))
	if w == 4 {
		if !sg.Fits(0, 0, 26, rows) || !unsafeutil.GridFit(len(im), 0, 4, 4, rows) {
			return
		}
		s, m := sg.Sub(0, 0, 26, rows), unsafeutil.NewGrid(im, 0, 4, 4, rows)
		for r := 0; r < rows; r++ {
			mc16H6ELo(unsafeutil.Row[[26]byte](s, r, 0), c0, c1, c2, c3, c4, c5, rnd).Shift(shv).StoreArray(unsafeutil.Row[[4]int32](m, r, 0))
		}
		return
	}
	for c := 0; c+8 <= w; c += 8 {
		if !sg.Fits(0, 2*c, 26, rows) || !unsafeutil.GridFit(len(im), c*rows, 8, 8, rows) {
			return
		}
		s, m := sg.Sub(0, 2*c, 26, rows), unsafeutil.NewGrid(im, c*rows, 8, 8, rows)
		for r := 0; r < rows; r++ {
			lo, hi := mc16H6E(unsafeutil.Row[[26]byte](s, r, 0), c0, c1, c2, c3, c4, c5, rnd)
			p := unsafeutil.Row[[8]int32](m, r, 0)
			lo.Shift(shv).StoreArray((*[4]int32)(p[0:4]))
			hi.Shift(shv).StoreArray((*[4]int32)(p[4:8]))
		}
	}
}

// mc16PrepCopyG is the 16bpc prep copy, (px << shl) + roundOffset in uint16
// lanes (below 2^16 at 10 and 12 bits), from sg (rows of at least 2*w
// readable bytes, 16 at width 4) into the CONV_BUF view og.
func mc16PrepCopyG(sg unsafeutil.Grid[uint8], shl int, og unsafeutil.Grid[uint16]) {
	off := archsimd.BroadcastUint16x8(mc16PrepRoundOffset)
	sv := archsimd.BroadcastInt16x8(int16(shl))
	h, w := og.Height(), og.Width()
	switch {
	case w == 4:
		if !sg.Fits(0, 0, 16, h) {
			return
		}
		s := sg.Sub(0, 0, 16, h)
		for r := 0; r < h; r++ {
			v := archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](s, r, 0)).ReshapeToUint16s().Shift(sv).Add(off)
			mc8Store4U16(unsafeutil.Row[[4]uint16](og, r, 0), v.ReshapeToUint64s().GetElem(0))
		}
	case w&15 == 0:
		for c := 0; c+16 <= w; c += 16 {
			if !sg.Fits(0, 2*c, 32, h) || !og.Fits(0, c, 16, h) {
				return
			}
			s, o := sg.Sub(0, 2*c, 32, h), og.Sub(0, c, 16, h)
			for r := 0; r < h; r++ {
				x, p := unsafeutil.Row[[32]byte](s, r, 0), unsafeutil.Row[[16]uint16](o, r, 0)
				archsimd.LoadUint8x16Array((*[16]byte)(x[0:16])).ReshapeToUint16s().Shift(sv).Add(off).StoreArray((*[8]uint16)(p[0:8]))
				archsimd.LoadUint8x16Array((*[16]byte)(x[16:32])).ReshapeToUint16s().Shift(sv).Add(off).StoreArray((*[8]uint16)(p[8:16]))
			}
		}
	default:
		for c := 0; c+8 <= w; c += 8 {
			if !sg.Fits(0, 2*c, 16, h) || !og.Fits(0, c, 8, h) {
				return
			}
			s, o := sg.Sub(0, 2*c, 16, h), og.Sub(0, c, 8, h)
			for r := 0; r < h; r++ {
				archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](s, r, 0)).ReshapeToUint16s().Shift(sv).Add(off).StoreArray(unsafeutil.Row[[8]uint16](o, r, 0))
			}
		}
	}
}

// mc16PutV6FromI32G is the 10-bit put 2D vertical pass over the strip-major
// int32 intermediate (h+5 rows per strip) onto the destination view dg (h
// rows of 2*w bytes), eight rows per chunk.
func mc16PutV6FromI32G(im []int32, w, h int, ph *mc8Phase, maxPix int16, dg unsafeutil.Grid[uint8]) {
	c0 := archsimd.LoadInt32x4Array(&ph.v[0])
	c1 := archsimd.LoadInt32x4Array(&ph.v[1])
	c2 := archsimd.LoadInt32x4Array(&ph.v[2])
	c3 := archsimd.LoadInt32x4Array(&ph.v[3])
	c4 := archsimd.LoadInt32x4Array(&ph.v[4])
	c5 := archsimd.LoadInt32x4Array(&ph.v[5])
	rnd := archsimd.LoadInt32x4Array(&mc8Vec.w15)
	zero := archsimd.BroadcastInt16x8(0)
	maxV := archsimd.BroadcastInt16x8(maxPix)
	rows := h + 5
	n := rows * 8
	for c := 0; c < w; c += 8 {
		if !dg.Fits(0, 2*c, 16, h) {
			return
		}
		dc := dg.Sub(0, 2*c, 16, h)
		st := im[c*rows : c*rows+n : c*rows+n]
		head := (*[40]int32)(st[0:40])
		x0, y0 := mc8LoadI32x8((*[8]int32)(head[0:8]))
		x1, y1 := mc8LoadI32x8((*[8]int32)(head[8:16]))
		x2, y2 := mc8LoadI32x8((*[8]int32)(head[16:24]))
		x3, y3 := mc8LoadI32x8((*[8]int32)(head[24:32]))
		x4, y4 := mc8LoadI32x8((*[8]int32)(head[32:40]))
		r, q := 0, 40
		for ; q <= len(st)-64; q += 64 {
			if !dc.Fits(r, 0, 16, 8) {
				return
			}
			d := dc.Sub(r, 0, 16, 8)
			win := (*[64]int32)(st[q : q+64])
			x5, y5 := mc8LoadI32x8((*[8]int32)(win[0:8]))
			x6, y6 := mc8LoadI32x8((*[8]int32)(win[8:16]))
			x7, y7 := mc8LoadI32x8((*[8]int32)(win[16:24]))
			x8, y8 := mc8LoadI32x8((*[8]int32)(win[24:32]))
			x9, y9 := mc8LoadI32x8((*[8]int32)(win[32:40]))
			x10, y10 := mc8LoadI32x8((*[8]int32)(win[40:48]))
			x11, y11 := mc8LoadI32x8((*[8]int32)(win[48:56]))
			x12, y12 := mc8LoadI32x8((*[8]int32)(win[56:64]))
			v0, v1 := mc16V6PutPair(x0, y0, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, c0, c1, c2, c3, c4, c5, rnd, zero, maxV)
			v0.ToBits().ReshapeToUint8s().StoreArray(unsafeutil.Row[[16]byte](d, 0, 0))
			v1.ToBits().ReshapeToUint8s().StoreArray(unsafeutil.Row[[16]byte](d, 1, 0))
			v0, v1 = mc16V6PutPair(x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8, c0, c1, c2, c3, c4, c5, rnd, zero, maxV)
			v0.ToBits().ReshapeToUint8s().StoreArray(unsafeutil.Row[[16]byte](d, 2, 0))
			v1.ToBits().ReshapeToUint8s().StoreArray(unsafeutil.Row[[16]byte](d, 3, 0))
			v0, v1 = mc16V6PutPair(x4, y4, x5, y5, x6, y6, x7, y7, x8, y8, x9, y9, x10, y10, c0, c1, c2, c3, c4, c5, rnd, zero, maxV)
			v0.ToBits().ReshapeToUint8s().StoreArray(unsafeutil.Row[[16]byte](d, 4, 0))
			v1.ToBits().ReshapeToUint8s().StoreArray(unsafeutil.Row[[16]byte](d, 5, 0))
			v0, v1 = mc16V6PutPair(x6, y6, x7, y7, x8, y8, x9, y9, x10, y10, x11, y11, x12, y12, c0, c1, c2, c3, c4, c5, rnd, zero, maxV)
			v0.ToBits().ReshapeToUint8s().StoreArray(unsafeutil.Row[[16]byte](d, 6, 0))
			v1.ToBits().ReshapeToUint8s().StoreArray(unsafeutil.Row[[16]byte](d, 7, 0))
			r += 8
			x0, y0, x1, y1, x2, y2, x3, y3, x4, y4 = x8, y8, x9, y9, x10, y10, x11, y11, x12, y12
		}
		for ; q <= len(st)-16; q += 16 {
			if !dc.Fits(r, 0, 16, 2) {
				return
			}
			d := dc.Sub(r, 0, 16, 2)
			win := (*[16]int32)(st[q : q+16])
			x5, y5 := mc8LoadI32x8((*[8]int32)(win[0:8]))
			x6, y6 := mc8LoadI32x8((*[8]int32)(win[8:16]))
			v0, v1 := mc16V6PutPair(x0, y0, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, c0, c1, c2, c3, c4, c5, rnd, zero, maxV)
			v0.ToBits().ReshapeToUint8s().StoreArray(unsafeutil.Row[[16]byte](d, 0, 0))
			v1.ToBits().ReshapeToUint8s().StoreArray(unsafeutil.Row[[16]byte](d, 1, 0))
			r += 2
			x0, y0, x1, y1, x2, y2, x3, y3, x4, y4 = x2, y2, x3, y3, x4, y4, x5, y5, x6, y6
		}
		if q <= len(st)-8 {
			if !dc.Fits(r, 0, 16, 1) {
				return
			}
			x5, y5 := mc8LoadI32x8((*[8]int32)(st[q : q+8]))
			v0, _ := mc16V6PutPair(x0, y0, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x5, y5, c0, c1, c2, c3, c4, c5, rnd, zero, maxV)
			v0.ToBits().ReshapeToUint8s().StoreArray(unsafeutil.Row[[16]byte](dc.Sub(r, 0, 16, 1), 0, 0))
		}
	}
}

// mc16PutV6FromI32GW4 is the width-4 form: two output rows per vector (the
// intermediate rows of a width-4 block are contiguous).
func mc16PutV6FromI32GW4(im []int32, h int, ph *mc8Phase, maxPix int16, dg unsafeutil.Grid[uint8]) {
	c0 := archsimd.LoadInt32x4Array(&ph.v[0])
	c1 := archsimd.LoadInt32x4Array(&ph.v[1])
	c2 := archsimd.LoadInt32x4Array(&ph.v[2])
	c3 := archsimd.LoadInt32x4Array(&ph.v[3])
	c4 := archsimd.LoadInt32x4Array(&ph.v[4])
	c5 := archsimd.LoadInt32x4Array(&ph.v[5])
	rnd := archsimd.LoadInt32x4Array(&mc8Vec.w15)
	zero := archsimd.BroadcastInt16x8(0)
	maxV := archsimd.BroadcastInt16x8(maxPix)
	st := im[: (h+5)*4 : (h+5)*4]
	head := (*[20]int32)(st[0:20])
	x0 := archsimd.LoadInt32x4Array((*[4]int32)(head[0:4]))
	x1 := archsimd.LoadInt32x4Array((*[4]int32)(head[4:8]))
	x2 := archsimd.LoadInt32x4Array((*[4]int32)(head[8:12]))
	x3 := archsimd.LoadInt32x4Array((*[4]int32)(head[12:16]))
	x4 := archsimd.LoadInt32x4Array((*[4]int32)(head[16:20]))
	r, q := 0, 20
	for ; q <= len(st)-8; q += 8 {
		if !dg.Fits(r, 0, 8, 2) {
			return
		}
		d := dg.Sub(r, 0, 8, 2)
		win := (*[8]int32)(st[q : q+8])
		x5 := archsimd.LoadInt32x4Array((*[4]int32)(win[0:4]))
		x6 := archsimd.LoadInt32x4Array((*[4]int32)(win[4:8]))
		a, b := mc8V6I32W4(x0, x1, x2, x3, x4, x5, x6, c0, c1, c2, c3, c4, c5, rnd)
		v := a.ToBits().ReshapeToUint16s().ConcatOdd(b.ToBits().ReshapeToUint16s()).BitsToInt16().Max(zero).Min(maxV).ToBits().ReshapeToUint64s()
		binary.LittleEndian.PutUint64(unsafeutil.Row[[8]byte](d, 0, 0)[:], v.GetElem(0))
		binary.LittleEndian.PutUint64(unsafeutil.Row[[8]byte](d, 1, 0)[:], v.GetElem(1))
		r += 2
		x0, x1, x2, x3, x4 = x2, x3, x4, x5, x6
	}
	if q <= len(st)-4 {
		if !dg.Fits(r, 0, 8, 1) {
			return
		}
		x5 := archsimd.LoadInt32x4Array((*[4]int32)(st[q : q+4]))
		a, _ := mc8V6I32W4(x0, x1, x2, x3, x4, x5, x5, c0, c1, c2, c3, c4, c5, rnd)
		v := a.ToBits().ReshapeToUint16s().ConcatOdd(a.ToBits().ReshapeToUint16s()).BitsToInt16().Max(zero).Min(maxV)
		binary.LittleEndian.PutUint64(unsafeutil.Row[[8]byte](dg.Sub(r, 0, 8, 1), 0, 0)[:], v.ToBits().ReshapeToUint64s().GetElem(0))
	}
}

// mc16PrepCopyW8 is mc16PrepCopyG for an eight-wide block (contiguous
// CONV_BUF rows): eight rows per chunk, as mc8PrepCopyW8.
func mc16PrepCopyW8(sg unsafeutil.Grid[uint8], shl int, out []uint16, h int) {
	off := archsimd.BroadcastUint16x8(mc16PrepRoundOffset)
	sv := archsimd.BroadcastInt16x8(int16(shl))
	r := 0
	for ; r+8 <= h; r += 8 {
		if !sg.Fits(r, 0, 16, 8) || 8*r+64 > len(out) {
			return
		}
		s, o := sg.Sub(r, 0, 16, 8), (*[64]uint16)(out[8*r:8*r+64])
		archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](s, 0, 0)).ReshapeToUint16s().Shift(sv).Add(off).StoreArray((*[8]uint16)(o[0:8]))
		archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](s, 1, 0)).ReshapeToUint16s().Shift(sv).Add(off).StoreArray((*[8]uint16)(o[8:16]))
		archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](s, 2, 0)).ReshapeToUint16s().Shift(sv).Add(off).StoreArray((*[8]uint16)(o[16:24]))
		archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](s, 3, 0)).ReshapeToUint16s().Shift(sv).Add(off).StoreArray((*[8]uint16)(o[24:32]))
		archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](s, 4, 0)).ReshapeToUint16s().Shift(sv).Add(off).StoreArray((*[8]uint16)(o[32:40]))
		archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](s, 5, 0)).ReshapeToUint16s().Shift(sv).Add(off).StoreArray((*[8]uint16)(o[40:48]))
		archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](s, 6, 0)).ReshapeToUint16s().Shift(sv).Add(off).StoreArray((*[8]uint16)(o[48:56]))
		archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](s, 7, 0)).ReshapeToUint16s().Shift(sv).Add(off).StoreArray((*[8]uint16)(o[56:64]))
	}
	for ; r < h; r++ {
		if !sg.Fits(r, 0, 16, 1) || 8*r+8 > len(out) {
			return
		}
		archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sg.Sub(r, 0, 16, 1), 0, 0)).ReshapeToUint16s().Shift(sv).Add(off).StoreArray((*[8]uint16)(out[8*r : 8*r+8]))
	}
}
