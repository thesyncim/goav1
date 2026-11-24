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

// dav1d-style 16bpc compound prep (src/mc_tmpl.c prep_8tap, src/arm/64/mc16.S
// prep_6tap / filter_fn) for 10- and 12-bit planes. It mirrors the 8-bit
// kernels of mc8_gosimd_arm64.go: regular and smooth filters (and the <=4 and
// bilinear tables) run as six taps (window taps 1..6), the source window is
// a byte slice plus stride, and blocks that overhang the plane first
// materialize their window with emuEdge16.
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
// mc8PrepV6FromI32S with that constant.
//
// Every access is a bounds-checked window: the source and destination plane
// rows at uint32 offsets (one compare each, see mc8W16), the eight-tap
// horizontal windows as one 26- or 30-byte window per eight outputs, and the
// int32 intermediate in the strip-major layout of mc8H6ToI32 (strips of
// eight columns, rows contiguous; a width-4 block is one strip of four), whose
// windows the loop bounds prove.

const (
	mc16PrepRoundOffset = 1<<14 + 1<<13
	mc16PrepV6Rnd       = (1<<21 + 1<<20 + 64) << 9
)

// mc16Source is mc8Source for 16bpc planes (2 bytes per sample, Stride in
// bytes). tx and ty are the pass lengths of each axis (mc16Taps: 0, 6 or 8);
// a six-tap axis reads from two samples before the block, an eight-tap axis
// from three. Every kernel loads eight samples per tap and output group, so
// the window stays readable for max(w+5, 13) samples (max(w+7, 15) with
// eight horizontal taps) from its left edge on every row.
func mc16Source(ref *frame.Plane, refX, refY, w, h, tx, ty int, edge *emuEdge16Buf) (src []byte, stride int, ok bool) {
	x0, y0, bw, bh := refX, refY, w, h
	read := 2 * max(w+5, 13)
	if tx != 0 {
		x0 -= tx/2 - 1
		bw += tx - 1
		if tx == 8 {
			read = 2 * max(w+7, 15)
		}
	}
	if ty != 0 {
		y0 -= ty/2 - 1
		bh += ty - 1
	}
	if ref.Width <= 0 || ref.Height <= 0 || ref.Stride < 2*ref.Width {
		return nil, 0, false
	}
	if x0 >= 0 && y0 >= 0 && x0+bw <= ref.Width && y0+bh <= ref.Height {
		start := y0*ref.Stride + 2*x0
		ext := (bh-1)*ref.Stride + read
		if start+ext <= len(ref.Pix) && ext <= mc8WindowMax {
			return ref.Pix[start : start+ext], ref.Stride, true
		}
	}
	if len(ref.Pix) < (ref.Height-1)*ref.Stride+2*ref.Width {
		return nil, 0, false
	}
	// The edge window holds emuEdge16Rows (136) rows of emuEdge16Stride (144)
	// samples; bh <= 135 and the kernels read at most 135 samples per row.
	emuEdge16(bw, bh, *ref, x0, y0, edge[:], 2*emuEdge16Stride)
	return edge[:], 2 * emuEdge16Stride, true
}

// mc16SourceGrid is mc16Source for six-tap (or unfiltered) axes as a Grid
// view: bh rows of 2*max(w+5, 13) bytes from the window's top-left tap
// sample.
func mc16SourceGrid(ref *frame.Plane, refX, refY, w, h, tx, ty int, edge *emuEdge16Buf) (src unsafeutil.Grid[uint8], ok bool) {
	x0, y0, bw, bh := refX, refY, w, h
	read := 2 * max(w+5, 13)
	if tx != 0 {
		x0 -= 2
		bw += 5
	}
	if ty != 0 {
		y0 -= 2
		bh += 5
	}
	if ref.Width <= 0 || ref.Height <= 0 || ref.Stride < 2*ref.Width {
		return src, false
	}
	if x0 >= 0 && y0 >= 0 && x0+bw <= ref.Width && y0+bh <= ref.Height {
		start := y0*ref.Stride + 2*x0
		if unsafeutil.GridFit(len(ref.Pix), start, ref.Stride, read, bh) {
			return unsafeutil.NewGrid(ref.Pix, start, ref.Stride, read, bh), true
		}
	}
	if len(ref.Pix) < (ref.Height-1)*ref.Stride+2*ref.Width {
		return src, false
	}
	// The edge window holds emuEdge16Rows (136) rows of emuEdge16Stride (144)
	// samples; bh <= 133 and the kernels read at most 133 samples per row.
	emuEdge16(bw, bh, *ref, x0, y0, edge[:], 2*emuEdge16Stride)
	if !unsafeutil.GridFit(len(edge), 0, 2*emuEdge16Stride, read, bh) {
		return src, false
	}
	return unsafeutil.NewGrid(edge[:], 0, 2*emuEdge16Stride, read, bh), true
}

// mc16PrepFast fills a w x h CONV_BUF (row stride w) from a 10- or 12-bit
// reference with the dav1d-style prep kernels. It returns false for shapes it
// does not cover (widths other than 4 or a multiple of 8, or a width-4 block
// with a sharp horizontal filter), in which case the caller runs the general
// path.
func mc16PrepFast(out []uint16, ref *frame.Plane, bitDepth, refX, refY, w, h, filterW, filterH, subX, subY int, filters InterpFilters, scratch *CompoundConvolveScratch) bool {
	if (bitDepth != 10 && bitDepth != 12) || (w&7 != 0 && w != 4) || w > maxBlockSize || h <= 0 || h > maxBlockSize || scratch == nil || len(out) < w*h {
		return false
	}
	kx := mc8Kind(filters.X, filterW)
	ky := mc8Kind(filters.Y, filterH)
	tx, ty := mc16Taps(subX, kx), mc16Taps(subY, ky)
	if tx == 8 && w == 4 {
		return false
	}
	// Halved taps: the round0 shift drops by one.
	sh := compoundRound0(uint8(bitDepth)) - 1
	if tx != 8 && ty != 8 {
		sg, ok := mc16SourceGrid(ref, refX, refY, w, h, tx, ty, &scratch.edge16)
		if !ok || !unsafeutil.GridFit(len(out), 0, w, w, h) {
			return false
		}
		og := unsafeutil.NewGrid(out, 0, w, w, h)
		switch {
		case tx != 0 && ty != 0:
			im := scratch.im[:]
			mc16H6ToI32G(sg, w, h+5, &mc8Phases[kx][subX], sh, im)
			if w == 4 {
				mc8PrepV6FromI32W4(im, h, &mc8Phases[ky][subY], out[:4*h], mc16PrepV6Rnd)
			} else {
				mc8PrepV6FromI32S(im, w, h, &mc8Phases[ky][subY], og, mc16PrepV6Rnd)
			}
		case tx != 0:
			mc16PrepH6G(sg, &mc8Phases[kx][subX], sh, og)
		case ty != 0:
			if w == 4 {
				mc16PrepV6GW4(sg, &mc8Phases[ky][subY], sh, og)
			} else {
				mc16PrepV6G(sg, &mc8Phases[ky][subY], sh, og)
			}
		default:
			mc16PrepCopyG(sg, 6-sh, og)
		}
		return true
	}
	return mc16PrepSharp(out, ref, refX, refY, w, h, kx, ky, tx, ty, subX, subY, sh, scratch)
}

// mc16PrepSharp is mc16PrepFast for blocks with an eight-tap (sharp) axis,
// over the slice source window of mc16Source; kept out of mc16PrepFast so the
// six-tap path carries no sharp state.
func mc16PrepSharp(out []uint16, ref *frame.Plane, refX, refY, w, h, kx, ky, tx, ty, subX, subY, sh int, scratch *CompoundConvolveScratch) bool {
	src, stride, ok := mc16Source(ref, refX, refY, w, h, tx, ty, &scratch.edge16)
	if !ok {
		return false
	}
	o := out[: w*h : w*h]
	switch {
	case tx != 0 && ty != 0:
		im := scratch.im[:]
		rows := h + ty - 1
		switch {
		case tx == 8:
			mc16H8ToI32(src, stride, w, rows, &mc16SharpPhases[subX], sh, im)
		case w == 4:
			mc16H6ToI32W4(src, stride, rows, &mc8Phases[kx][subX], sh, im)
		default:
			mc16H6ToI32(src, stride, w, rows, &mc8Phases[kx][subX], sh, im)
		}
		switch {
		case ty == 8:
			mc16V8FromI32(im, w, h, &mc16SharpPhases[subY].p, mc16PrepV6Rnd, false, 0, nil, 0, o)
		case w == 4:
			mc8PrepV6FromI32W4(im, h, &mc8Phases[ky][subY], o, mc16PrepV6Rnd)
		default:
			if !unsafeutil.GridFit(len(o), 0, w, w, h) {
				return false
			}
			mc8PrepV6FromI32S(im, w, h, &mc8Phases[ky][subY], unsafeutil.NewGrid(o, 0, w, w, h), mc16PrepV6Rnd)
		}
	case tx == 8:
		mc16PrepH8(src, stride, w, h, &mc16SharpPhases[subX], sh, o)
	case tx != 0:
		mc16PrepH6(src, stride, w, h, &mc8Phases[kx][subX], sh, o)
	case ty == 8:
		mc16PrepV8(src, stride, w, h, &mc16SharpPhases[subY], sh, o)
	case ty != 0:
		mc16PrepV6(src, stride, w, h, &mc8Phases[ky][subY], sh, o)
	default:
		mc16PrepCopy(src, stride, w, h, 6-sh, o)
	}
	return true
}

// mc16LB loads the eight 16bpc samples of p as int16 lanes (samples are at
// most 4095, so the bit pattern is a non-negative int16).
func mc16LB(p *[16]byte) archsimd.Int16x8 {
	return archsimd.LoadUint8x16Array(p).ReshapeToUint16s().BitsToInt16()
}

// mc16L16 loads the eight samples at byte offset o of s.
func mc16L16(s []byte, o uint32) archsimd.Int16x8 { return mc16LB(mc8W16(s, o)) }

// mc16St16 stores eight int16 lanes at byte offset o of s.
func mc16St16(s []byte, o uint32, v archsimd.Int16x8) {
	v.ToBits().ReshapeToUint8s().StoreArray(mc8W16(s, o))
}

// mc16W26 and mc16W30 are the six- and eight-tap horizontal windows of eight
// outputs (13 and 15 samples).
func mc16W26(s []byte, o uint32) *[26]byte { return (*[26]byte)(s[uint(o) : uint(o)+26]) }
func mc16W30(s []byte, o uint32) *[30]byte { return (*[30]byte)(s[uint(o) : uint(o)+30]) }

// mc16PackU16 keeps the low 16 bits of each int32 lane of lo then hi (UZP1),
// libaom's uint16 cast.
func mc16PackU16(lo, hi archsimd.Int32x4) archsimd.Uint16x8 {
	return lo.ToBits().ReshapeToUint16s().ConcatEven(hi.ToBits().ReshapeToUint16s())
}

// mc16LoadTaps6 loads the six broadcast halved taps and their high halves.
// The taps travel as separate values (not a struct) so they stay in
// registers across the pixel loops.
func mc16LoadTaps6(ph *mc8Phase) (c0, c1, c2, c3, c4, c5, h0, h1, h2, h3, h4, h5 archsimd.Int16x8) {
	c0 = archsimd.LoadInt16x8Array(&ph.h[0])
	c1 = archsimd.LoadInt16x8Array(&ph.h[1])
	c2 = archsimd.LoadInt16x8Array(&ph.h[2])
	c3 = archsimd.LoadInt16x8Array(&ph.h[3])
	c4 = archsimd.LoadInt16x8Array(&ph.h[4])
	c5 = archsimd.LoadInt16x8Array(&ph.h[5])
	return c0, c1, c2, c3, c4, c5, c0.HiToLo(), c1.HiToLo(), c2.HiToLo(), c3.HiToLo(), c4.HiToLo(), c5.HiToLo()
}

// mc16H6Lo is mc16H6 over the low four outputs only (width 4).
func mc16H6Lo(win *[26]byte, c0, c1, c2, c3, c4, c5, h0, h1, h2, h3, h4, h5 archsimd.Int16x8, rnd archsimd.Int32x4) archsimd.Int32x4 {
	lo := rnd.Add(mc16LB((*[16]byte)(win[0:16])).MulWidenLo(c0))
	lo = lo.Add(mc16LB((*[16]byte)(win[2:18])).MulWidenLo(c1))
	lo = lo.Add(mc16LB((*[16]byte)(win[4:20])).MulWidenLo(c2))
	lo = lo.Add(mc16LB((*[16]byte)(win[6:22])).MulWidenLo(c3))
	lo = lo.Add(mc16LB((*[16]byte)(win[8:24])).MulWidenLo(c4))
	return lo.Add(mc16LB((*[16]byte)(win[10:26])).MulWidenLo(c5))
}

// mc16V6 is the six-tap vertical sum of one output row from the rows r0..r5.
func mc16V6(r0, r1, r2, r3, r4, r5 archsimd.Int16x8, c0, c1, c2, c3, c4, c5, h0, h1, h2, h3, h4, h5 archsimd.Int16x8, rnd archsimd.Int32x4) (lo, hi archsimd.Int32x4) {
	lo = rnd.Add(r0.MulWidenLo(c0))
	hi = rnd.Add(r0.HiToLo().MulWidenLo(h0))
	lo = lo.Add(r1.MulWidenLo(c1))
	hi = hi.Add(r1.HiToLo().MulWidenLo(h1))
	lo = lo.Add(r2.MulWidenLo(c2))
	hi = hi.Add(r2.HiToLo().MulWidenLo(h2))
	lo = lo.Add(r3.MulWidenLo(c3))
	hi = hi.Add(r3.HiToLo().MulWidenLo(h3))
	lo = lo.Add(r4.MulWidenLo(c4))
	hi = hi.Add(r4.HiToLo().MulWidenLo(h4))
	lo = lo.Add(r5.MulWidenLo(c5))
	hi = hi.Add(r5.HiToLo().MulWidenLo(h5))
	return lo, hi
}

// mc16V6Lo is mc16V6 over the low four lanes only (width 4).
func mc16V6Lo(r0, r1, r2, r3, r4, r5 archsimd.Int16x8, c0, c1, c2, c3, c4, c5, h0, h1, h2, h3, h4, h5 archsimd.Int16x8, rnd archsimd.Int32x4) archsimd.Int32x4 {
	a := rnd.Add(r0.MulWidenLo(c0))
	a = a.Add(r1.MulWidenLo(c1))
	a = a.Add(r2.MulWidenLo(c2))
	a = a.Add(r3.MulWidenLo(c3))
	a = a.Add(r4.MulWidenLo(c4))
	return a.Add(r5.MulWidenLo(c5))
}

// mc16H6ToI32 writes rows x w rounded horizontal sums m as int32 in the
// strip-major layout of mc8H6ToI32 (the mc8PrepV6FromI32S layout).
func mc16H6ToI32(src []byte, srcStride, w, rows int, ph *mc8Phase, sh int, im []int32) {
	c0, c1, c2, c3, c4, c5, h0, h1, h2, h3, h4, h5 := mc16LoadTaps6(ph)
	rnd := archsimd.BroadcastInt32x4(int32(1) << (sh - 1))
	shv := archsimd.BroadcastInt32x4(int32(-sh))
	ss := uint32(srcStride)
	n := rows * 8
	for c := 0; c < w; c += 8 {
		st := im[c*rows : c*rows+n : c*rows+n]
		so := uint32(2 * c)
		for q := 0; q <= len(st)-8; q += 8 {
			hw := mc16W26(src, so)
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
			win := (*[8]int32)(st[q : q+8])
			lo.Shift(shv).StoreArray((*[4]int32)(win[0:4]))
			hi.Shift(shv).StoreArray((*[4]int32)(win[4:8]))
			so += ss
		}
	}
}

// mc16H6ToI32W4 writes rows x 4 rounded horizontal sums as int32, row stride
// 4 (the mc8PrepV6FromI32W4 layout).
func mc16H6ToI32W4(src []byte, srcStride, rows int, ph *mc8Phase, sh int, im []int32) {
	c0, c1, c2, c3, c4, c5, h0, h1, h2, h3, h4, h5 := mc16LoadTaps6(ph)
	rnd := archsimd.BroadcastInt32x4(int32(1) << (sh - 1))
	shv := archsimd.BroadcastInt32x4(int32(-sh))
	ss := uint32(srcStride)
	st := im[: rows*4 : rows*4]
	var so uint32
	for q := 0; q <= len(st)-4; q += 4 {
		mc16H6Lo(mc16W26(src, so), c0, c1, c2, c3, c4, c5, h0, h1, h2, h3, h4, h5, rnd).Shift(shv).StoreArray((*[4]int32)(st[q : q+4]))
		so += ss
	}
}

// mc16PrepH6 is the prep horizontal-only pass. roundOffset is folded into the
// rounding constant (a multiple of 2^sh commutes with the floor). The
// CONV_BUF is walked as one stream whose loop bound proves each window.
func mc16PrepH6(src []byte, srcStride, w, h int, ph *mc8Phase, sh int, out []uint16) {
	c0, c1, c2, c3, c4, c5, h0, h1, h2, h3, h4, h5 := mc16LoadTaps6(ph)
	rnd := archsimd.BroadcastInt32x4(int32(1)<<(sh-1) + int32(mc16PrepRoundOffset)<<sh)
	shv := archsimd.BroadcastInt32x4(int32(-sh))
	ss := uint32(srcStride)
	out = out[: w*h : w*h]
	var so uint32
	if w == 4 {
		for q := 0; q <= len(out)-4; q += 4 {
			lo := mc16H6Lo(mc16W26(src, so), c0, c1, c2, c3, c4, c5, h0, h1, h2, h3, h4, h5, rnd).Shift(shv)
			mc8Store4U16((*[4]uint16)(out[q:q+4]), mc16PackU16(lo, lo).ReshapeToUint64s().GetElem(0))
			so += ss
		}
		return
	}
	c, ws := uint32(0), uint32(w)
	for q := 0; q <= len(out)-8; q += 8 {
		hw := mc16W26(src, so+2*c)
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
		mc16PackU16(lo.Shift(shv), hi.Shift(shv)).StoreArray((*[8]uint16)(out[q : q+8]))
		if c += 8; c == ws {
			c, so = 0, so+ss
		}
	}
}

// mc16PrepV6 is the prep vertical-only pass; each column strip slides a
// six-row window down the block (dav1d's register-resident v path), two
// output rows per iteration.
func mc16PrepV6(src []byte, srcStride, w, h int, ph *mc8Phase, sh int, out []uint16) {
	c0, c1, c2, c3, c4, c5, h0, h1, h2, h3, h4, h5 := mc16LoadTaps6(ph)
	rnd := archsimd.BroadcastInt32x4(int32(1)<<(sh-1) + int32(mc16PrepRoundOffset)<<sh)
	shv := archsimd.BroadcastInt32x4(int32(-sh))
	ss, ws := uint32(srcStride), uint32(w)
	if w == 4 {
		// Two output rows share one vector: the low lanes of rows r and r+1
		// (the CONV_BUF rows are contiguous, walked as one proven stream).
		out = out[: 4*h : 4*h]
		r0 := mc16L16(src, 0)
		r1 := mc16L16(src, ss)
		r2 := mc16L16(src, 2*ss)
		r3 := mc16L16(src, 3*ss)
		so := 4 * ss
		r4 := mc16L16(src, so)
		q := 0
		for ; q <= len(out)-8; q += 8 {
			r5 := mc16L16(src, so+ss)
			so += 2 * ss
			r6 := mc16L16(src, so)
			a := mc16V6Lo(r0, r1, r2, r3, r4, r5, c0, c1, c2, c3, c4, c5, h0, h1, h2, h3, h4, h5, rnd)
			b := mc16V6Lo(r1, r2, r3, r4, r5, r6, c0, c1, c2, c3, c4, c5, h0, h1, h2, h3, h4, h5, rnd)
			mc16PackU16(a.Shift(shv), b.Shift(shv)).StoreArray((*[8]uint16)(out[q : q+8]))
			r0, r1, r2, r3, r4 = r2, r3, r4, r5, r6
		}
		if q <= len(out)-4 {
			r5 := mc16L16(src, so+ss)
			a := mc16V6Lo(r0, r1, r2, r3, r4, r5, c0, c1, c2, c3, c4, c5, h0, h1, h2, h3, h4, h5, rnd).Shift(shv)
			mc8Store4U16((*[4]uint16)(out[q:q+4]), mc16PackU16(a, a).ReshapeToUint64s().GetElem(0))
		}
		return
	}
	for col := uint32(0); col < ws; col += 8 {
		so := 2 * col
		r0 := mc16L16(src, so)
		r1 := mc16L16(src, so+ss)
		r2 := mc16L16(src, so+2*ss)
		r3 := mc16L16(src, so+3*ss)
		so += 4 * ss
		r4 := mc16L16(src, so)
		o := col
		r := 0
		for ; r+2 <= h; r += 2 {
			r5 := mc16L16(src, so+ss)
			so += 2 * ss
			r6 := mc16L16(src, so)
			alo, ahi := mc16V6(r0, r1, r2, r3, r4, r5, c0, c1, c2, c3, c4, c5, h0, h1, h2, h3, h4, h5, rnd)
			blo, bhi := mc16V6(r1, r2, r3, r4, r5, r6, c0, c1, c2, c3, c4, c5, h0, h1, h2, h3, h4, h5, rnd)
			mc16PackU16(alo.Shift(shv), ahi.Shift(shv)).StoreArray(mc8WU8(out, o))
			mc16PackU16(blo.Shift(shv), bhi.Shift(shv)).StoreArray(mc8WU8(out, o+ws))
			o += 2 * ws
			r0, r1, r2, r3, r4 = r2, r3, r4, r5, r6
		}
		if r < h {
			r5 := mc16L16(src, so+ss)
			alo, ahi := mc16V6(r0, r1, r2, r3, r4, r5, c0, c1, c2, c3, c4, c5, h0, h1, h2, h3, h4, h5, rnd)
			mc16PackU16(alo.Shift(shv), ahi.Shift(shv)).StoreArray(mc8WU8(out, o))
		}
	}
}

// mc16PrepCopy is the prep copy: (px << shl) + roundOffset, in uint16 lanes
// (the result stays below 2^16 at 10 and 12 bits). The CONV_BUF is walked
// as one proven stream; each source row group is one window.
func mc16PrepCopy(src []byte, srcStride, w, h, shl int, out []uint16) {
	mul := archsimd.BroadcastUint16x8(uint16(1) << shl)
	off := archsimd.BroadcastUint16x8(mc16PrepRoundOffset)
	ss := srcStride
	out = out[: w*h : w*h]
	// The CONV_BUF is walked as one stream whose loop bound proves its
	// windows; the source rows are limit-checked windows (mc8R8/16/32).
	switch {
	case w == 4:
		sl := len(src) - 8
		q, so := 0, 0
		for ; q <= len(out)-8; q += 8 {
			a, so1 := binary.LittleEndian.Uint64(mc8R8(src, so, sl)[:]), so+ss
			b := binary.LittleEndian.Uint64(mc8R8(src, so1, sl)[:])
			v := archsimd.LoadUint64x2Array(&[2]uint64{a, b})
			v.ReshapeToUint16s().MulAdd(mul, off).StoreArray((*[8]uint16)(out[q : q+8]))
			so = so1 + ss
		}
		if q <= len(out)-4 {
			v := archsimd.BroadcastUint64x2(binary.LittleEndian.Uint64(mc8R8(src, so, sl)[:])).ReshapeToUint16s().MulAdd(mul, off)
			mc8Store4U16((*[4]uint16)(out[q:q+4]), v.ReshapeToUint64s().GetElem(0))
		}
	case w == 8:
		sl := len(src) - 16
		so := 0
		for q := 0; q <= len(out)-8; q += 8 {
			archsimd.LoadUint8x16Array(mc8R16(src, so, sl)).ReshapeToUint16s().MulAdd(mul, off).StoreArray((*[8]uint16)(out[q : q+8]))
			so += ss
		}
	case w&15 != 0:
		sl := len(src) - 16
		c, so, rb := 0, 0, 2*w
		for q := 0; q <= len(out)-8; q += 8 {
			archsimd.LoadUint8x16Array(mc8R16(src, so+c, sl)).ReshapeToUint16s().MulAdd(mul, off).StoreArray((*[8]uint16)(out[q : q+8]))
			if c += 16; c == rb {
				c, so = 0, so+ss
			}
		}
	default:
		// w a multiple of 16: sixteen samples (one 32-byte source window)
		// per step.
		sl := len(src) - 32
		c, so, rb := 0, 0, 2*w
		for q := 0; q <= len(out)-16; q += 16 {
			win, o := mc8R32(src, so+c, sl), (*[16]uint16)(out[q:q+16])
			archsimd.LoadUint8x16Array((*[16]byte)(win[0:16])).ReshapeToUint16s().MulAdd(mul, off).StoreArray((*[8]uint16)(o[0:8]))
			archsimd.LoadUint8x16Array((*[16]byte)(win[16:32])).ReshapeToUint16s().MulAdd(mul, off).StoreArray((*[8]uint16)(o[8:16]))
			if c += 32; c == rb {
				c, so = 0, so+ss
			}
		}
	}
}
