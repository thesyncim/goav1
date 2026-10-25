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

// dav1d-style 16bpc compound prep (src/mc_tmpl.c prep_8tap, src/arm/64/mc16.S
// prep_6tap / filter_fn) for 10- and 12-bit planes. It mirrors the 8-bit
// kernels of mc8_gosimd_arm64.go: regular and smooth filters (and the <=4 and
// bilinear tables) run as six taps (window taps 1..6), the source window is
// addressed through a pointer and stride, and blocks that overhang the plane
// first materialize their window with emuEdge16.
//
// Unlike 8 bpc, a halved-tap sum of 10-bit samples does not fit int16, so as
// in dav1d's mc16 the horizontal and 1D passes multiply-widen int16 samples by
// int16 taps into int32 lanes (SMULL / SMULL2). Arithmetic follows the
// libaom-exact CONV_BUF formulas of predictInterCompoundRefHighBDToConvBuf*
// with round0 = compoundRound0(bd) (3 at 10 bits, 5 at 12) and the halved taps
// of mc8Phases (sum = 2 * hs; all AV1 subpel taps are even):
//
//	prep H, V: ((hs + 2^(round0-2)) >> (round0-1)) + roundOffset
//	prep 2D:   m = (hs + 2^(round0-2)) >> (round0-1)
//	           (sum(tap * m) + 2^21 + 2^20 + 64) >> 7
//	prep copy: (px << (7 - round0)) + roundOffset
//
// roundOffset = 2^14 + 2^13 at both depths. The 2D horizontal pass drops
// libaom's 2^(bd+6) bias (a multiple of 2^round0, so the floor commutes);
// sum(tap) = 128 turns it into the 2^20 term of the vertical constant, which
// is the same at 10 and 12 bits. The vertical pass is the 8-bit
// mc8PrepV6FromI32 with that constant.

const (
	mc16PrepRoundOffset = 1<<14 + 1<<13
	mc16PrepV6Rnd       = (1<<21 + 1<<20 + 64) << 9
)

// mc16Source is mc8Source for 16bpc planes (2 bytes per sample, Stride in
// bytes). Every kernel loads eight samples per tap and output group, so the
// window must stay readable for max(w+5, 13) samples from its left edge on
// every row.
func mc16Source(ref *frame.Plane, refX, refY, w, h int, hx, hy bool, edge *emuEdge16Buf) (src unsafe.Pointer, stride int, ok bool) {
	x0, y0, bw, bh := refX, refY, w, h
	if hx {
		x0 -= 2
		bw += 5
	}
	if hy {
		y0 -= 2
		bh += 5
	}
	if ref.Width <= 0 || ref.Height <= 0 || ref.Stride < 2*ref.Width {
		return nil, 0, false
	}
	read := 2 * max(w+5, 13)
	if x0 >= 0 && y0 >= 0 && x0+bw <= ref.Width && y0+bh <= ref.Height {
		start := y0*ref.Stride + 2*x0
		if start+(bh-1)*ref.Stride+read <= len(ref.Pix) {
			return unsafe.Pointer(unsafe.SliceData(ref.Pix[start:])), ref.Stride, true
		}
	}
	if len(ref.Pix) < (ref.Height-1)*ref.Stride+2*ref.Width {
		return nil, 0, false
	}
	// The edge window holds emuEdge16Rows (136) rows of emuEdge16Stride (144)
	// samples; bh <= 133 and the kernels read at most 133 samples per row.
	emuEdge16(bw, bh, *ref, x0, y0, edge[:], 2*emuEdge16Stride)
	return unsafe.Pointer(&edge[0]), 2 * emuEdge16Stride, true
}

// mc16PrepFast fills a w x h CONV_BUF (row stride w) from a 10- or 12-bit
// reference with the dav1d-style prep kernels. It returns false for shapes it
// does not cover (sharp filters, widths other than 4 or a multiple of 8), in
// which case the caller runs the general path.
func mc16PrepFast(out []uint16, ref *frame.Plane, bitDepth, refX, refY, w, h, filterW, filterH, subX, subY int, filters InterpFilters, scratch *CompoundConvolveScratch) bool {
	if (bitDepth != 10 && bitDepth != 12) || (w&7 != 0 && w != 4) || w > maxBlockSize || h <= 0 || h > maxBlockSize || scratch == nil || len(out) < w*h {
		return false
	}
	kx := mc8Kind(filters.X, filterW)
	ky := mc8Kind(filters.Y, filterH)
	if (subX != 0 && !mc8SixTap[kx]) || (subY != 0 && !mc8SixTap[ky]) {
		return false
	}
	src, stride, ok := mc16Source(ref, refX, refY, w, h, subX != 0, subY != 0, &scratch.edge16)
	if !ok {
		return false
	}
	// Halved taps: the round0 shift drops by one.
	sh := compoundRound0(uint8(bitDepth)) - 1
	o := unsafe.Pointer(&out[0])
	switch {
	case subX != 0 && subY != 0:
		im := unsafe.Pointer(&scratch.im[0])
		if w == 4 {
			mc16H6ToI32W4(src, stride, h+5, &mc8Phases[kx][subX], sh, im)
			mc8PrepV6FromI32W4(im, h, &mc8Phases[ky][subY], o, mc16PrepV6Rnd)
		} else {
			mc16H6ToI32(src, stride, w, h+5, &mc8Phases[kx][subX], sh, im)
			mc8PrepV6FromI32(im, w, h, &mc8Phases[ky][subY], o, mc16PrepV6Rnd)
		}
	case subX != 0:
		mc16PrepH6(src, stride, w, h, &mc8Phases[kx][subX], sh, o)
	case subY != 0:
		mc16PrepV6(src, stride, w, h, &mc8Phases[ky][subY], sh, o)
	default:
		mc16PrepCopy(src, stride, w, h, 6-sh, o)
	}
	return true
}

// mc16L16 loads eight 16bpc samples at p as int16 lanes (samples are at most
// 4095, so the bit pattern is a non-negative int16).
//
//go:nocheckptr
func mc16L16(p unsafe.Pointer) archsimd.Int16x8 {
	return archsimd.LoadInt16x8Array((*[8]int16)(p))
}

// The kernels below spell out their six-tap chains inline so the taps stay in
// registers across the pixel loops. Each high-half product is
// HiToLo(x) * HiToLo(c), which the compiler fuses into one SMULL2.

// mc16PackU16 keeps the low 16 bits of each int32 lane of lo then hi (UZP1),
// libaom's uint16 cast.
func mc16PackU16(lo, hi archsimd.Int32x4) archsimd.Uint16x8 {
	return lo.ToBits().ReshapeToUint16s().ConcatEven(hi.ToBits().ReshapeToUint16s())
}

// mc16H6ToI32 writes rows x w rounded horizontal sums m as int32 with row
// stride w (the mc8PrepV6FromI32 layout).
//
//go:nocheckptr
func mc16H6ToI32(src unsafe.Pointer, srcStride, w, rows int, ph *mc8Phase, sh int, im unsafe.Pointer) {
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
			q := unsafe.Add(o, col*4)
			lo.Shift(shv).StoreArray((*[4]int32)(q))
			hi.Shift(shv).StoreArray((*[4]int32)(unsafe.Add(q, 16)))
		}
	}
}

// mc16H6ToI32W4 writes rows x 4 rounded horizontal sums as int32, row stride
// 4 (the mc8PrepV6FromI32W4 layout).
//
//go:nocheckptr
func mc16H6ToI32W4(src unsafe.Pointer, srcStride, rows int, ph *mc8Phase, sh int, im unsafe.Pointer) {
	c0 := archsimd.LoadInt16x8Array(&ph.h[0])
	c1 := archsimd.LoadInt16x8Array(&ph.h[1])
	c2 := archsimd.LoadInt16x8Array(&ph.h[2])
	c3 := archsimd.LoadInt16x8Array(&ph.h[3])
	c4 := archsimd.LoadInt16x8Array(&ph.h[4])
	c5 := archsimd.LoadInt16x8Array(&ph.h[5])
	rnd := archsimd.BroadcastInt32x4(int32(1) << (sh - 1))
	shv := archsimd.BroadcastInt32x4(int32(-sh))
	for r := 0; r < rows; r++ {
		p := unsafe.Add(src, r*srcStride)
		lo := rnd.Add(mc16L16(p).MulWidenLo(c0))
		lo = lo.Add(mc16L16(unsafe.Add(p, 2)).MulWidenLo(c1))
		lo = lo.Add(mc16L16(unsafe.Add(p, 4)).MulWidenLo(c2))
		lo = lo.Add(mc16L16(unsafe.Add(p, 6)).MulWidenLo(c3))
		lo = lo.Add(mc16L16(unsafe.Add(p, 8)).MulWidenLo(c4))
		lo = lo.Add(mc16L16(unsafe.Add(p, 10)).MulWidenLo(c5))
		lo.Shift(shv).StoreArray((*[4]int32)(unsafe.Add(im, r*16)))
	}
}

// mc16PrepH6 is the prep horizontal-only pass. roundOffset is folded into the
// rounding constant (a multiple of 2^sh commutes with the floor).
//
//go:nocheckptr
func mc16PrepH6(src unsafe.Pointer, srcStride, w, h int, ph *mc8Phase, sh int, out unsafe.Pointer) {
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
	rnd := archsimd.BroadcastInt32x4(int32(1)<<(sh-1) + int32(mc16PrepRoundOffset)<<sh)
	shv := archsimd.BroadcastInt32x4(int32(-sh))
	ro := w * 2
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
			*(*uint64)(unsafe.Add(out, r*ro)) = mc16PackU16(lo, lo).ReshapeToUint64s().GetElem(0)
		}
		return
	}
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
			mc16PackU16(lo.Shift(shv), hi.Shift(shv)).StoreArray((*[8]uint16)(unsafe.Add(o, 2*col)))
		}
	}
}

// mc16PrepV6 is the prep vertical-only pass; each column strip slides a
// six-row window down the block (dav1d's register-resident v path), two
// output rows per iteration.
//
//go:nocheckptr
func mc16PrepV6(src unsafe.Pointer, srcStride, w, h int, ph *mc8Phase, sh int, out unsafe.Pointer) {
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
	rnd := archsimd.BroadcastInt32x4(int32(1)<<(sh-1) + int32(mc16PrepRoundOffset)<<sh)
	shv := archsimd.BroadcastInt32x4(int32(-sh))
	ro := w * 2
	if w == 4 {
		// Two output rows share one vector: the low lanes of rows r and r+1
		// (the CONV_BUF rows are contiguous).
		s := src
		r0 := mc16L16(s)
		r1 := mc16L16(unsafe.Add(s, srcStride))
		r2 := mc16L16(unsafe.Add(s, 2*srcStride))
		r3 := mc16L16(unsafe.Add(s, 3*srcStride))
		r4 := mc16L16(unsafe.Add(s, 4*srcStride))
		s = unsafe.Add(s, 5*srcStride)
		o := out
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
			mc16PackU16(a.Shift(shv), b.Shift(shv)).StoreArray((*[8]uint16)(o))
			r0, r1, r2, r3, r4 = r2, r3, r4, r5, r6
			s = unsafe.Add(s, 2*srcStride)
			o = unsafe.Add(o, 2*ro)
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
			mc16PackU16(alo.Shift(shv), ahi.Shift(shv)).StoreArray((*[8]uint16)(o))
			mc16PackU16(blo.Shift(shv), bhi.Shift(shv)).StoreArray((*[8]uint16)(unsafe.Add(o, ro)))
			r0, r1, r2, r3, r4 = r2, r3, r4, r5, r6
			s = unsafe.Add(s, 2*srcStride)
			o = unsafe.Add(o, 2*ro)
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
			mc16PackU16(alo.Shift(shv), ahi.Shift(shv)).StoreArray((*[8]uint16)(o))
		}
	}
}

// mc16PrepCopy is the prep copy: (px << shl) + roundOffset, in uint16 lanes
// (the result stays below 2^16 at 10 and 12 bits).
//
//go:nocheckptr
func mc16PrepCopy(src unsafe.Pointer, srcStride, w, h, shl int, out unsafe.Pointer) {
	mul := archsimd.BroadcastUint16x8(uint16(1) << shl)
	off := archsimd.BroadcastUint16x8(mc16PrepRoundOffset)
	ro := w * 2
	if w == 4 {
		for r := 0; r < h; r++ {
			v := archsimd.BroadcastUint64x2(*(*uint64)(unsafe.Add(src, r*srcStride))).ReshapeToUint16s().MulAdd(mul, off)
			*(*uint64)(unsafe.Add(out, r*ro)) = v.ReshapeToUint64s().GetElem(0)
		}
		return
	}
	for r := 0; r < h; r++ {
		s := unsafe.Add(src, r*srcStride)
		o := unsafe.Add(out, r*ro)
		for col := 0; col < w; col += 8 {
			archsimd.LoadUint16x8Array((*[8]uint16)(unsafe.Add(s, 2*col))).MulAdd(mul, off).StoreArray((*[8]uint16)(unsafe.Add(o, 2*col)))
		}
	}
}
