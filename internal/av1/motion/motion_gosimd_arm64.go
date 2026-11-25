// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package motion

import (
	"encoding/binary"
	"errors"
	"github.com/thesyncim/goav1/internal/av1/dsp/cpu"
	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/unsafeutil"
	"os"
	"simd/archsimd"
)

func u8HalfMAC(x, coefficient, sum archsimd.Int16x8) archsimd.Int16x8 {
	return x.MulAdd(coefficient, sum)
}

func u8HalfSaturate(x archsimd.Int16x8) archsimd.Uint8x16 {
	return x.SaturateToUint8()
}

func u8WideMAC8(lo, hi archsimd.Int32x4, samples, coefficient archsimd.Int16x8) (archsimd.Int32x4, archsimd.Int32x4) {
	return lo.Add(samples.MulWidenLo(coefficient)), hi.Add(samples.HiToLo().MulWidenLo(coefficient.HiToLo()))
}

func u8PackClipped32(lo, hi archsimd.Int32x4) archsimd.Uint8x16 {
	return lo.TruncToInt16().ToBits().ReshapeToUint64s().InterleaveLo(hi.TruncToInt16().ToBits().ReshapeToUint64s()).ReshapeToUint16s().BitsToInt16().SaturateToUint8()
}

func u8PackU16From32(lo, hi archsimd.Int32x4) archsimd.Uint16x8 {
	return lo.TruncToInt16().ToBits().ReshapeToUint64s().InterleaveLo(hi.TruncToInt16().ToBits().ReshapeToUint64s()).ReshapeToUint16s()
}

func u8MulAdd32(x, coefficient, sum archsimd.Int32x4) archsimd.Int32x4 {
	return x.MulAdd(coefficient, sum)
}

func u8WidenU16(x archsimd.Uint16x8) (archsimd.Int32x4, archsimd.Int32x4) {
	return x.ExtendLo4ToUint32().BitsToInt32(), x.HiToLo().ExtendLo4ToUint32().BitsToInt32()
}

func u8PackSaturate32(lo, hi archsimd.Int32x4) archsimd.Uint8x16 {
	return lo.SaturateToUint16().ReshapeToUint64s().InterleaveLo(hi.SaturateToUint16().ReshapeToUint64s()).ReshapeToUint16s().SaturateToUint8()
}

func u8Right16(x, amount archsimd.Int16x8, count uint64) archsimd.Int16x8 {
	return x.Shift(amount)
}

func u8Right32(x, amount archsimd.Int32x4, count uint64) archsimd.Int32x4 {
	return x.Shift(amount)
}

// scaledStep2HorizontalRow filters one source row into eight output columns
// at a time for the 2:1 scaled convolve. Even lanes read bytes t+j and odd
// lanes bytes t+j+odd of the load, so tap t interleaves the load shifted by t
// with the load shifted by t+odd (EXT + ZIP1).
func scaledStep2HorizontalRow(src []byte, w8 int, odd int, coef *[filterTaps]archsimd.Int16x8, row []int16) {
	xRound := archsimd.BroadcastInt16x8(8194)
	right2 := archsimd.BroadcastInt16x8(-2)
	for x := 0; x < w8; x += 8 {
		a := archsimd.LoadUint8x16Array((*[16]byte)(src[x/2 : x/2+16]))
		s1 := a.ConcatShiftBytesRight(a, 1)
		s2 := a.ConcatShiftBytesRight(a, 2)
		s3 := a.ConcatShiftBytesRight(a, 3)
		s4 := a.ConcatShiftBytesRight(a, 4)
		s5 := a.ConcatShiftBytesRight(a, 5)
		s6 := a.ConcatShiftBytesRight(a, 6)
		s7 := a.ConcatShiftBytesRight(a, 7)
		var t0, t1, t2, t3, t4, t5, t6, t7 archsimd.Uint8x16
		if odd == 0 {
			t0, t1, t2, t3 = a.InterleaveLo(a), s1.InterleaveLo(s1), s2.InterleaveLo(s2), s3.InterleaveLo(s3)
			t4, t5, t6, t7 = s4.InterleaveLo(s4), s5.InterleaveLo(s5), s6.InterleaveLo(s6), s7.InterleaveLo(s7)
		} else {
			s8 := a.ConcatShiftBytesRight(a, 8)
			t0, t1, t2, t3 = a.InterleaveLo(s1), s1.InterleaveLo(s2), s2.InterleaveLo(s3), s3.InterleaveLo(s4)
			t4, t5, t6, t7 = s4.InterleaveLo(s5), s5.InterleaveLo(s6), s6.InterleaveLo(s7), s7.InterleaveLo(s8)
		}
		sum := u8HalfMAC(t0.ExtendLo8ToUint16().BitsToInt16(), coef[0], xRound)
		sum = u8HalfMAC(t1.ExtendLo8ToUint16().BitsToInt16(), coef[1], sum)
		sum = u8HalfMAC(t2.ExtendLo8ToUint16().BitsToInt16(), coef[2], sum)
		sum = u8HalfMAC(t3.ExtendLo8ToUint16().BitsToInt16(), coef[3], sum)
		sum = u8HalfMAC(t4.ExtendLo8ToUint16().BitsToInt16(), coef[4], sum)
		sum = u8HalfMAC(t5.ExtendLo8ToUint16().BitsToInt16(), coef[5], sum)
		sum = u8HalfMAC(t6.ExtendLo8ToUint16().BitsToInt16(), coef[6], sum)
		sum = u8HalfMAC(t7.ExtendLo8ToUint16().BitsToInt16(), coef[7], sum)
		u8Right16(sum, right2, 2).Store(row[x : x+8])
	}
}

// hbdSIMDAvailable reports whether the Go SIMD high-bit-depth kernels may run.
// NEON is mandatory on every arm64 CPU Go supports.
func hbdSIMDAvailable() bool {
	return cpu.Detected.NEON
}

// hbdJoinU16 packs the four low lanes of lo and hi into eight uint16 lanes
// (lo first). The narrowing truncation keeps the scalar uint16 cast of the
// CONV_BUF; the clipped pixel values are already in range.
func hbdJoinU16(lo, hi archsimd.Int32x4) archsimd.Uint16x8 {
	lo16 := lo.TruncToInt16().ToBits().ReshapeToUint64s()
	hi16 := hi.TruncToInt16().ToBits().ReshapeToUint64s()
	return lo16.InterleaveLo(hi16).ReshapeToUint16s()
}

// hbdMAC8 accumulates the widening products of eight int16 samples s with the
// broadcast coefficient c into lo (lanes 0..3) and hi (lanes 4..7).
func hbdMAC8(lo, hi archsimd.Int32x4, s, c archsimd.Int16x8) (archsimd.Int32x4, archsimd.Int32x4) {
	return lo.Add(s.MulWidenLo(c)), hi.Add(s.HiToLo().MulWidenLo(c))
}

func hbdMAC4(sum archsimd.Int32x4, s, c archsimd.Int16x8) archsimd.Int32x4 {
	return sum.Add(s.MulWidenLo(c))
}

// hbdMulAdd32 returns x*y + z for int32 lanes (one fused multiply-accumulate).
func hbdMulAdd32(x, y, z archsimd.Int32x4) archsimd.Int32x4 {
	return x.MulAdd(y, z)
}

// hbdShiftRight uses arm64's signed per-lane shift with a hoisted amount.
func hbdShiftRight(v, amount archsimd.Int32x4, _ int) archsimd.Int32x4 {
	return v.Shift(amount)
}

func hbdShiftRightU16(v archsimd.Uint16x8, amount archsimd.Int16x8, _ int) archsimd.Uint16x8 {
	return v.Shift(amount)
}

// hbdWidenU16 zero-extends eight uint16 lanes to two int32 vectors (lanes 0..3
// and 4..7).
func hbdWidenU16(u archsimd.Uint16x8) (lo, hi archsimd.Int32x4) {
	return u.ExtendLo4ToUint32().BitsToInt32(), u.HiToLo().ExtendLo4ToUint32().BitsToInt32()
}

// narrowU8x8 packs the sixteen int32 lanes of lo and hi (already clipped to
// [0, 255]) into the low eight bytes of the result.
func narrowU8x8(lo, hi archsimd.Int32x4) archsimd.Uint8x16 {
	return hbdJoinU16(lo, hi).TruncToUint8()
}

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
		// The source window (mc16SourceGrid), in place when it lies inside
		// the plane: as in Put8, that case is checked in line and runs its
		// kernel with no call before it.
		hx, hy := mc8Halo(subX), mc8Halo(subY)
		x0, y0, bh := refX-hx, refY-hy, h+hy+hy+hy>>1
		sOff, read := y0*ref.Stride+2*x0, 2*max(w+5, 13)
		if x0 < 0 || y0 < 0 || x0+w+hx+hx+hx>>1 > ref.Width || y0+bh > ref.Height || ref.Stride < 2*ref.Width ||
			!unsafeutil.GridFit(len(out), 0, w, w, h) || !unsafeutil.GridFit(len(ref.Pix), sOff, ref.Stride, read, bh) {
			return mc16PrepEdge(out, ref, refX, refY, w, h, kx, ky, tx, ty, subX, subY, sh, scratch)
		}
		og, sg := unsafeutil.NewGrid(out, 0, w, w, h), unsafeutil.NewGrid(ref.Pix, sOff, ref.Stride, read, bh)
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
			if w == 8 {
				mc16PrepCopyW8(sg, 6-sh, out, h)
			} else {
				mc16PrepCopyG(sg, 6-sh, og)
			}
		}
		return true
	}
	return mc16PrepSharp(out, ref, refX, refY, w, h, kx, ky, tx, ty, subX, subY, sh, scratch)
}

// mc16PrepEdge is mc16PrepFast's six-tap path for a source window that is
// not inside the plane (or views that do not fit): the window from
// mc16SourceGrid, which clamps it.
func mc16PrepEdge(out []uint16, ref *frame.Plane, refX, refY, w, h, kx, ky, tx, ty, subX, subY, sh int, scratch *CompoundConvolveScratch) bool {
	if !unsafeutil.GridFit(len(out), 0, w, w, h) {
		return false
	}
	sg, ok := mc16SourceGrid(ref, refX, refY, w, h, tx, ty, &scratch.edge16)
	if !ok {
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
		if w == 8 {
			mc16PrepCopyW8(sg, 6-sh, out, h)
		} else {
			mc16PrepCopyG(sg, 6-sh, og)
		}
	}
	return true
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
	default: // ty == 8
		mc16PrepV8(src, stride, w, h, &mc16SharpPhases[subY], sh, o)
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

// mc16CopyFast is the 16bpc put copy (dav1d mc16.S put_neon): an in-plane
// w x h block of a 2-byte-sample plane, w a multiple of 4, moved as vector
// rows by unsafeutil.CopyRect over 2w bytes instead of one memmove per row. It returns
// false when the source leaves the plane (the clamped copy runs).
func mc16CopyFast(dst *frame.Plane, ref *frame.Plane, dstX, dstY, refX, refY, w, h int) bool {
	if w&3 != 0 || w > maxBlockSize || h <= 0 {
		return false
	}
	if dstX < 0 || dstY < 0 || dstX+w > dst.Width || dstY+h > dst.Height || dst.Stride < 2*dst.Width {
		return false
	}
	if refX < 0 || refY < 0 || refX+w > ref.Width || refY+h > ref.Height || ref.Stride < 2*ref.Width {
		return false
	}
	dOff, sOff := dstY*dst.Stride+2*dstX, refY*ref.Stride+2*refX
	if w&(w-1) == 0 { // every AV1 block width
		return unsafeutil.CopyRect(dst.Pix, dOff, dst.Stride, ref.Pix, sOff, ref.Stride, 2*w, h)
	}
	if !unsafeutil.GridFit(len(dst.Pix), dOff, dst.Stride, 2*w, h) || !unsafeutil.GridFit(len(ref.Pix), sOff, ref.Stride, 2*w, h) {
		return false
	}
	return mc8CopyStrips(unsafeutil.NewGrid(dst.Pix, dOff, dst.Stride, 2*w, h), unsafeutil.NewGrid(ref.Pix, sOff, ref.Stride, 2*w, h))
}

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
			lo := mc16H6ELo(s.Row26(r, 0), c0, c1, c2, c3, c4, c5, rnd).Shift(shv)
			binary.LittleEndian.PutUint64(o.Row8(r, 0)[:], mc16Clip(lo, lo, zero, maxV).ToBits().ReshapeToUint64s().GetElem(0))
		}
		return
	}
	for c := 0; c+8 <= w; c += 8 {
		if !sg.Fits(0, 2*c, 26, h) || !dg.Fits(0, 2*c, 16, h) {
			return
		}
		s, o := sg.Sub(0, 2*c, 26, h), dg.Sub(0, 2*c, 16, h)
		for r := 0; r < h; r++ {
			lo, hi := mc16H6E(s.Row26(r, 0), c0, c1, c2, c3, c4, c5, rnd)
			mc16Clip(lo.Shift(shv), hi.Shift(shv), zero, maxV).ToBits().ReshapeToUint8s().StoreArray(o.Row16(r, 0))
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
			lo := mc16H6ELo(s.Row26(r, 0), c0, c1, c2, c3, c4, c5, rnd).Shift(shv)
			mc8Store4U16(og.Row4(r, 0), mc16PackU16(lo, lo).ReshapeToUint64s().GetElem(0))
		}
		return
	}
	for c := 0; c+8 <= w; c += 8 {
		if !sg.Fits(0, 2*c, 26, h) || !og.Fits(0, c, 8, h) {
			return
		}
		s, o := sg.Sub(0, 2*c, 26, h), og.Sub(0, c, 8, h)
		for r := 0; r < h; r++ {
			lo, hi := mc16H6E(s.Row26(r, 0), c0, c1, c2, c3, c4, c5, rnd)
			mc16PackU16(lo.Shift(shv), hi.Shift(shv)).StoreArray(o.Row8(r, 0))
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
			mc16H6ELo(s.Row26(r, 0), c0, c1, c2, c3, c4, c5, rnd).Shift(shv).StoreArray(m.Row4(r, 0))
		}
		return
	}
	for c := 0; c+8 <= w; c += 8 {
		if !sg.Fits(0, 2*c, 26, rows) || !unsafeutil.GridFit(len(im), c*rows, 8, 8, rows) {
			return
		}
		s, m := sg.Sub(0, 2*c, 26, rows), unsafeutil.NewGrid(im, c*rows, 8, 8, rows)
		for r := 0; r < rows; r++ {
			lo, hi := mc16H6E(s.Row26(r, 0), c0, c1, c2, c3, c4, c5, rnd)
			p := m.Row8(r, 0)
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
			v := archsimd.LoadUint8x16Array(s.Row16(r, 0)).ReshapeToUint16s().Shift(sv).Add(off)
			mc8Store4U16(og.Row4(r, 0), v.ReshapeToUint64s().GetElem(0))
		}
	case w&15 == 0:
		for c := 0; c+16 <= w; c += 16 {
			if !sg.Fits(0, 2*c, 32, h) || !og.Fits(0, c, 16, h) {
				return
			}
			s, o := sg.Sub(0, 2*c, 32, h), og.Sub(0, c, 16, h)
			for r := 0; r < h; r++ {
				x, p := s.Row32(r, 0), o.Row16(r, 0)
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
				archsimd.LoadUint8x16Array(s.Row16(r, 0)).ReshapeToUint16s().Shift(sv).Add(off).StoreArray(o.Row8(r, 0))
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
			v0.ToBits().ReshapeToUint8s().StoreArray(d.Row16(0, 0))
			v1.ToBits().ReshapeToUint8s().StoreArray(d.Row16(1, 0))
			v0, v1 = mc16V6PutPair(x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8, c0, c1, c2, c3, c4, c5, rnd, zero, maxV)
			v0.ToBits().ReshapeToUint8s().StoreArray(d.Row16(2, 0))
			v1.ToBits().ReshapeToUint8s().StoreArray(d.Row16(3, 0))
			v0, v1 = mc16V6PutPair(x4, y4, x5, y5, x6, y6, x7, y7, x8, y8, x9, y9, x10, y10, c0, c1, c2, c3, c4, c5, rnd, zero, maxV)
			v0.ToBits().ReshapeToUint8s().StoreArray(d.Row16(4, 0))
			v1.ToBits().ReshapeToUint8s().StoreArray(d.Row16(5, 0))
			v0, v1 = mc16V6PutPair(x6, y6, x7, y7, x8, y8, x9, y9, x10, y10, x11, y11, x12, y12, c0, c1, c2, c3, c4, c5, rnd, zero, maxV)
			v0.ToBits().ReshapeToUint8s().StoreArray(d.Row16(6, 0))
			v1.ToBits().ReshapeToUint8s().StoreArray(d.Row16(7, 0))
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
			v0.ToBits().ReshapeToUint8s().StoreArray(d.Row16(0, 0))
			v1.ToBits().ReshapeToUint8s().StoreArray(d.Row16(1, 0))
			r += 2
			x0, y0, x1, y1, x2, y2, x3, y3, x4, y4 = x2, y2, x3, y3, x4, y4, x5, y5, x6, y6
		}
		if q <= len(st)-8 {
			if !dc.Fits(r, 0, 16, 1) {
				return
			}
			x5, y5 := mc8LoadI32x8((*[8]int32)(st[q : q+8]))
			v0, _ := mc16V6PutPair(x0, y0, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x5, y5, c0, c1, c2, c3, c4, c5, rnd, zero, maxV)
			v0.ToBits().ReshapeToUint8s().StoreArray(dc.Sub(r, 0, 16, 1).Row16(0, 0))
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
		binary.LittleEndian.PutUint64(d.Row8(0, 0)[:], v.GetElem(0))
		binary.LittleEndian.PutUint64(d.Row8(1, 0)[:], v.GetElem(1))
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
		binary.LittleEndian.PutUint64(dg.Sub(r, 0, 8, 1).Row8(0, 0)[:], v.ToBits().ReshapeToUint64s().GetElem(0))
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
		archsimd.LoadUint8x16Array(s.Row16(0, 0)).ReshapeToUint16s().Shift(sv).Add(off).StoreArray((*[8]uint16)(o[0:8]))
		archsimd.LoadUint8x16Array(s.Row16(1, 0)).ReshapeToUint16s().Shift(sv).Add(off).StoreArray((*[8]uint16)(o[8:16]))
		archsimd.LoadUint8x16Array(s.Row16(2, 0)).ReshapeToUint16s().Shift(sv).Add(off).StoreArray((*[8]uint16)(o[16:24]))
		archsimd.LoadUint8x16Array(s.Row16(3, 0)).ReshapeToUint16s().Shift(sv).Add(off).StoreArray((*[8]uint16)(o[24:32]))
		archsimd.LoadUint8x16Array(s.Row16(4, 0)).ReshapeToUint16s().Shift(sv).Add(off).StoreArray((*[8]uint16)(o[32:40]))
		archsimd.LoadUint8x16Array(s.Row16(5, 0)).ReshapeToUint16s().Shift(sv).Add(off).StoreArray((*[8]uint16)(o[40:48]))
		archsimd.LoadUint8x16Array(s.Row16(6, 0)).ReshapeToUint16s().Shift(sv).Add(off).StoreArray((*[8]uint16)(o[48:56]))
		archsimd.LoadUint8x16Array(s.Row16(7, 0)).ReshapeToUint16s().Shift(sv).Add(off).StoreArray((*[8]uint16)(o[56:64]))
	}
	for ; r < h; r++ {
		if !sg.Fits(r, 0, 16, 1) || 8*r+8 > len(out) {
			return
		}
		archsimd.LoadUint8x16Array(sg.Sub(r, 0, 16, 1).Row16(0, 0)).ReshapeToUint16s().Shift(sv).Add(off).StoreArray((*[8]uint16)(out[8*r : 8*r+8]))
	}
}

// 16bpc compound blends (dav1d src/arm/64/mc16.S avg / mask / w_mask), the
// 10- and 12-bit siblings of mc8_gosimd_arm64.go's mc8BlendAvg and
// mc8mask_gosimd_arm64.go. They compute libaom's CONV_BUF formulas exactly for
// every uint16 input. With rb = 2*7 - round0 - 7 (4 at 10 bits, 2 at 12) and
// roundOffset = 2^14 + 2^13:
//
//	avg:  clip(((a + b) >> 1) - roundOffset + 2^(rb-1)) >> rb)
//	      floor((a + b) / 2) is the rounding average minus (a ^ b) & 1 (no
//	      uint16 overflow), and the bias is a saturating subtract: a negative
//	      result clips to 0 either way.
//	mask: clip((X - 64*(roundOffset - 2^(rb-1))) >> (6 + rb)),
//	      X = m*a + (64-m)*b by nested floor division; the products widen to
//	      uint32 (UMULL/UMULL2) with m pre-scaled by 2^(10-rb), so the shift
//	      is the high half of each lane (UZP2).
//	diff-weighted mask (round = 6 at both depths):
//	      min(38 + (((|a-b| >> 5) + 1) >> 5), 64), i.e. 38 + round6(|a-b|) / 16.

// mc16BlendRoundBits returns rb and the avg bias roundOffset - 2^(rb-1).
func mc16BlendRoundBits(bitDepth int) (rb int, bias uint16) {
	rb = 2*filterBits - compoundRound0(uint8(bitDepth)) - compoundRound1Bits
	return rb, uint16(mc16PrepRoundOffsetBlend - 1<<(rb-1))
}

const mc16PrepRoundOffsetBlend = 1<<14 + 1<<13

// The blends walk the two CONV_BUF predictors as one stream under
// three-index slices, so the loop bounds prove their windows; the plane and
// mask rows are uint32-offset windows (one compare each, see mc8W16).

// mc16BlendAvgFast is the 16bpc plain compound average (fwd = bck = 8) over
// two CONV_BUF predictors with row stride w (w a multiple of 8, or 4 with an
// even height).
func mc16BlendAvgFast(dst *frame.Plane, a, b []uint16, bitDepth, dstX, dstY, w, h int) bool {
	if (bitDepth != 10 && bitDepth != 12) || w <= 0 || (w != 4 && w&7 != 0) || h <= 0 || len(a) < w*h || len(b) < w*h || (w == 4 && h&1 != 0) {
		return false
	}
	off := dstY*dst.Stride + 2*dstX
	ext := (h-1)*dst.Stride + 2*w
	if dstX < 0 || dstY < 0 || dstX+w > dst.Width || off+ext > len(dst.Pix) || ext > mc8WindowMax {
		return false
	}
	rb, bias := mc16BlendRoundBits(bitDepth)
	mc16BlendAvg(dst.Pix[off:off+ext], dst.Stride, a, b, w, h, rb, bias, uint16(1)<<bitDepth-1)
	return true
}

func mc16BlendAvg(dst []byte, dstStride int, a, b []uint16, w, h, rb int, biasC, maxC uint16) {
	one := archsimd.BroadcastUint16x8(1)
	bias := archsimd.BroadcastUint16x8(biasC)
	maxV := archsimd.BroadcastUint16x8(maxC)
	sh := archsimd.BroadcastInt16x8(int16(-rb))
	avg := func(pa, pb *[8]uint16) archsimd.Uint16x8 {
		va := archsimd.LoadUint16x8Array(pa)
		vb := archsimd.LoadUint16x8Array(pb)
		return va.Average(vb).Sub(va.Xor(vb).And(one)).SubSaturated(bias).Shift(sh).Min(maxV)
	}
	n := w * h
	a, b = a[:n:n], b[:n:n]
	ds := uint32(dstStride)
	var d uint32
	switch {
	case w == 4:
		for q := 0; q <= len(a)-8; q += 8 {
			v := avg((*[8]uint16)(a[q:q+8]), (*[8]uint16)(b[q:q+8])).ReshapeToUint64s()
			binary.LittleEndian.PutUint64(mc8W8(dst, d)[:], v.GetElem(0))
			binary.LittleEndian.PutUint64(mc8W8(dst, d+ds)[:], v.GetElem(1))
			d += 2 * ds
		}
	case w == 8:
		for q := 0; q <= len(a)-8; q += 8 {
			avg((*[8]uint16)(a[q:q+8]), (*[8]uint16)(b[q:q+8])).ReshapeToUint8s().StoreArray(mc8W16(dst, d))
			d += ds
		}
	case w&15 == 0:
		// Sixteen samples per step: one 32-byte destination window.
		c, rs := uint32(0), uint32(2*w)
		for q := 0; q <= len(a)-16; q += 16 {
			pa, pb := (*[16]uint16)(a[q:q+16]), (*[16]uint16)(b[q:q+16])
			dw := mc8W32(dst, d+c)
			avg((*[8]uint16)(pa[0:8]), (*[8]uint16)(pb[0:8])).ReshapeToUint8s().StoreArray((*[16]byte)(dw[0:16]))
			avg((*[8]uint16)(pa[8:16]), (*[8]uint16)(pb[8:16])).ReshapeToUint8s().StoreArray((*[16]byte)(dw[16:32]))
			if c += 32; c == rs {
				c, d = 0, d+ds
			}
		}
	default:
		c, rs := uint32(0), uint32(2*w)
		for q := 0; q <= len(a)-8; q += 8 {
			avg((*[8]uint16)(a[q:q+8]), (*[8]uint16)(b[q:q+8])).ReshapeToUint8s().StoreArray(mc8W16(dst, d+c))
			if c += 16; c == rs {
				c, d = 0, d+ds
			}
		}
	}
}

// mc16BlendMaskFast blends with a full-resolution (4:4:4 / luma) or 4:2:0
// subsampled mask; width must be a multiple of 8.
func mc16BlendMaskFast(dst frame.Plane, a, b []uint16, bitDepth, dstX, dstY, w, h int, mask []byte, maskStride int, subX, subY bool) bool {
	if (bitDepth != 10 && bitDepth != 12) || w&7 != 0 || w > maxBlockSize || h <= 0 || len(a) < w*h || len(b) < w*h || subX != subY {
		return false
	}
	off := dstY*dst.Stride + 2*dstX
	ext := (h-1)*dst.Stride + 2*w
	if dstX < 0 || dstY < 0 || dstX+w > dst.Width || off+ext > len(dst.Pix) || ext > mc8WindowMax {
		return false
	}
	mw, mh := w, h
	if subX {
		mw, mh = 2*w, 2*h
	}
	// The unsubsampled path loads 16 mask bytes per eight columns.
	mExt := (mh-1)*maskStride + mw + 8
	if maskStride < mw || mExt-8 > len(mask) || mExt > cap(mask) || mExt > mc8WindowMax {
		return false
	}
	rb, bias := mc16BlendRoundBits(bitDepth)
	mc16BlendMask(dst.Pix[off:off+ext], dst.Stride, a, b, w, h, mask[:mExt], maskStride, subX,
		uint32(bias)<<(16-rb), uint16(10-rb), int16(1)<<bitDepth-1)
	return true
}

func mc16BlendMask(dst []byte, dstStride int, a, b []uint16, w, h int, mask []byte, maskStride int, sub bool, biasC uint32, mshift uint16, maxPix int16) {
	bias := archsimd.BroadcastUint32x4(biasC)
	m64 := archsimd.BroadcastUint16x8(64)
	two := archsimd.BroadcastUint16x8(2)
	right2 := archsimd.BroadcastInt16x8(-2)
	msh := archsimd.BroadcastInt16x8(int16(mshift))
	zero := archsimd.BroadcastInt16x8(0)
	maxV := archsimd.BroadcastInt16x8(maxPix)
	n := w * h
	a, b = a[:n:n], b[:n:n]
	ds, ms, ws := uint32(dstStride), uint32(maskStride), uint32(w)
	mstep := ms
	if sub {
		mstep = 2 * ms
	}
	var c, d, mo uint32
	for q := 0; q <= len(a)-8; q += 8 {
		var m archsimd.Uint16x8
		if sub {
			v0 := archsimd.LoadUint8x16Array(mc8W16(mask, mo+2*c))
			v1 := archsimd.LoadUint8x16Array(mc8W16(mask, mo+ms+2*c))
			lo := v0.ExtendLo8ToUint16().Add(v1.ExtendLo8ToUint16())
			hi := v0.HiToLo().ExtendLo8ToUint16().Add(v1.HiToLo().ExtendLo8ToUint16())
			m = lo.ConcatAddPairs(hi).Add(two).Shift(right2)
		} else {
			m = archsimd.LoadUint8x16Array(mc8W16(mask, mo+c)).ExtendLo8ToUint16()
		}
		mshv := m.Shift(msh)
		mi := m64.Sub(m).Shift(msh)
		va := archsimd.LoadUint16x8Array((*[8]uint16)(a[q : q+8]))
		vb := archsimd.LoadUint16x8Array((*[8]uint16)(b[q : q+8]))
		lo := va.MulWidenLo(mshv).Add(vb.MulWidenLo(mi)).Sub(bias)
		hi := va.HiToLo().MulWidenLo(mshv.HiToLo()).Add(vb.HiToLo().MulWidenLo(mi.HiToLo())).Sub(bias)
		v := lo.ReshapeToUint16s().ConcatOdd(hi.ReshapeToUint16s()).BitsToInt16()
		mc16St16(dst, d+2*c, v.Max(zero).Min(maxV))
		if c += 8; c == ws {
			c, d, mo = 0, d+ds, mo+mstep
		}
	}
}

// mc16DiffWtdMaskFast builds the 10- or 12-bit difference-weighted mask; width
// must be a multiple of 8.
func mc16DiffWtdMaskFast(mask []byte, maskStride int, a, b []uint16, bitDepth, w, h int, invert bool) bool {
	if (bitDepth != 10 && bitDepth != 12) || w&7 != 0 || w > maxBlockSize || h <= 0 || len(a) < w*h || len(b) < w*h ||
		maskStride < w || (h-1)*maskStride+w > len(mask) || (h-1)*maskStride+w > mc8WindowMax {
		return false
	}
	mc16DiffWtdMask(mask[:(h-1)*maskStride+w], maskStride, a, b, w, h, invert)
	return true
}

func mc16DiffWtdMask(mask []byte, maskStride int, a, b []uint16, w, h int, invert bool) {
	right5 := archsimd.BroadcastInt16x8(-5)
	one := archsimd.BroadcastUint16x8(1)
	base := archsimd.BroadcastUint16x8(38)
	maxM := archsimd.BroadcastUint16x8(64)
	n := w * h
	a, b = a[:n:n], b[:n:n]
	ms, ws := uint32(maskStride), uint32(w)
	var c, mo uint32
	mk := func(pa, pb *[8]uint16) archsimd.Uint16x8 {
		va := archsimd.LoadUint16x8Array(pa)
		vb := archsimd.LoadUint16x8Array(pb)
		diff := va.Max(vb).Sub(va.Min(vb))
		m := diff.Shift(right5).Add(one).Shift(right5).Add(base).Min(maxM)
		if invert {
			m = maxM.Sub(m)
		}
		return m
	}
	if w&15 == 0 {
		// Sixteen samples per step: one 16-byte mask window.
		for q := 0; q <= len(a)-16; q += 16 {
			pa, pb := (*[16]uint16)(a[q:q+16]), (*[16]uint16)(b[q:q+16])
			m0 := mk((*[8]uint16)(pa[0:8]), (*[8]uint16)(pb[0:8])).TruncToUint8().ReshapeToUint64s()
			m1 := mk((*[8]uint16)(pa[8:16]), (*[8]uint16)(pb[8:16])).TruncToUint8().ReshapeToUint64s()
			m0.InterleaveLo(m1).ReshapeToUint8s().StoreArray(mc8W16(mask, mo+c))
			if c += 16; c == ws {
				c, mo = 0, mo+ms
			}
		}
		return
	}
	for q := 0; q <= len(a)-8; q += 8 {
		m := mk((*[8]uint16)(a[q:q+8]), (*[8]uint16)(b[q:q+8]))
		binary.LittleEndian.PutUint64(mc8W8(mask, mo+c)[:], m.TruncToUint8().ReshapeToUint64s().GetElem(0))
		if c += 8; c == ws {
			c, mo = 0, mo+ms
		}
	}
}

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

// mc16LoadTaps8 loads the eight broadcast halved sharp taps and their high
// halves, as separate values so they stay in registers.
func mc16LoadTaps8(ph *mc16Phase8) (c0, c1, c2, c3, c4, c5, c6, c7, h0, h1, h2, h3, h4, h5, h6, h7 archsimd.Int16x8) {
	c0, c1, c2, c3, c4, c5, c6, c7 = mc16Load8Taps(ph)
	return c0, c1, c2, c3, c4, c5, c6, c7, c0.HiToLo(), c1.HiToLo(), c2.HiToLo(), c3.HiToLo(), c4.HiToLo(), c5.HiToLo(), c6.HiToLo(), c7.HiToLo()
}

// mc16H8ToI32 is mc16H6ToI32 over window taps 0..7 (w a multiple of 8),
// strip-major.
func mc16H8ToI32(src []byte, srcStride, w, rows int, ph *mc16Phase8, sh int, im []int32) {
	c0, c1, c2, c3, c4, c5, c6, c7, h0, h1, h2, h3, h4, h5, h6, h7 := mc16LoadTaps8(ph)
	rnd := archsimd.BroadcastInt32x4(int32(1) << (sh - 1))
	shv := archsimd.BroadcastInt32x4(int32(-sh))
	ss := uint32(srcStride)
	n := rows * 8
	for c := 0; c < w; c += 8 {
		st := im[c*rows : c*rows+n : c*rows+n]
		so := uint32(2 * c)
		for q := 0; q <= len(st)-8; q += 8 {
			hw := mc16W30(src, so)
			x0 := mc16LB((*[16]byte)(hw[0:16]))
			x1 := mc16LB((*[16]byte)(hw[2:18]))
			x2 := mc16LB((*[16]byte)(hw[4:20]))
			x3 := mc16LB((*[16]byte)(hw[6:22]))
			x4 := mc16LB((*[16]byte)(hw[8:24]))
			x5 := mc16LB((*[16]byte)(hw[10:26]))
			x6 := mc16LB((*[16]byte)(hw[12:28]))
			x7 := mc16LB((*[16]byte)(hw[14:30]))
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
			win := (*[8]int32)(st[q : q+8])
			lo.Shift(shv).StoreArray((*[4]int32)(win[0:4]))
			hi.Shift(shv).StoreArray((*[4]int32)(win[4:8]))
			so += ss
		}
	}
}

// mc16PrepH8 is mc16PrepH6 over window taps 0..7 (w a multiple of 8).
func mc16PrepH8(src []byte, srcStride, w, h int, ph *mc16Phase8, sh int, out []uint16) {
	c0, c1, c2, c3, c4, c5, c6, c7, h0, h1, h2, h3, h4, h5, h6, h7 := mc16LoadTaps8(ph)
	rnd := archsimd.BroadcastInt32x4(int32(1)<<(sh-1) + int32(mc16PrepRoundOffset)<<sh)
	shv := archsimd.BroadcastInt32x4(int32(-sh))
	ss := uint32(srcStride)
	out = out[: w*h : w*h]
	var so uint32
	c, ws := uint32(0), uint32(w)
	for q := 0; q <= len(out)-8; q += 8 {
		hw := mc16W30(src, so+2*c)
		x0 := mc16LB((*[16]byte)(hw[0:16]))
		x1 := mc16LB((*[16]byte)(hw[2:18]))
		x2 := mc16LB((*[16]byte)(hw[4:20]))
		x3 := mc16LB((*[16]byte)(hw[6:22]))
		x4 := mc16LB((*[16]byte)(hw[8:24]))
		x5 := mc16LB((*[16]byte)(hw[10:26]))
		x6 := mc16LB((*[16]byte)(hw[12:28]))
		x7 := mc16LB((*[16]byte)(hw[14:30]))
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
		mc16PackU16(lo.Shift(shv), hi.Shift(shv)).StoreArray((*[8]uint16)(out[q : q+8]))
		if c += 8; c == ws {
			c, so = 0, so+ss
		}
	}
}

// mc16PutH8 is mc16PutH6 over window taps 0..7 (w a multiple of 8).
func mc16PutH8(src []byte, srcStride, w, h int, ph *mc16Phase8, rndC int32, maxPix int16, dst []byte, dstStride int) {
	c0, c1, c2, c3, c4, c5, c6, c7, h0, h1, h2, h3, h4, h5, h6, h7 := mc16LoadTaps8(ph)
	rnd := archsimd.BroadcastInt32x4(rndC)
	shv := archsimd.BroadcastInt32x4(-6)
	zero := archsimd.BroadcastInt16x8(0)
	maxV := archsimd.BroadcastInt16x8(maxPix)
	ss, ds := uint32(srcStride), uint32(dstStride)
	var so, do uint32
	for r := 0; r < h; r++ {
		for col := uint32(0); col < uint32(2*w); col += 16 {
			hw := mc16W30(src, so+col)
			x0 := mc16LB((*[16]byte)(hw[0:16]))
			x1 := mc16LB((*[16]byte)(hw[2:18]))
			x2 := mc16LB((*[16]byte)(hw[4:20]))
			x3 := mc16LB((*[16]byte)(hw[6:22]))
			x4 := mc16LB((*[16]byte)(hw[8:24]))
			x5 := mc16LB((*[16]byte)(hw[10:26]))
			x6 := mc16LB((*[16]byte)(hw[12:28]))
			x7 := mc16LB((*[16]byte)(hw[14:30]))
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
			mc16St16(dst, do+col, mc16Clip(lo.Shift(shv), hi.Shift(shv), zero, maxV))
		}
		so, do = so+ss, do+ds
	}
}

// mc16V8Lo is the eight-tap vertical chain over the low four lanes of the
// rows x0..x7: rnd + sum(half tap * x) as int32.
func mc16V8Lo(x0, x1, x2, x3, x4, x5, x6, x7 archsimd.Int16x8, c0, c1, c2, c3, c4, c5, c6, c7, h0, h1, h2, h3, h4, h5, h6, h7 archsimd.Int16x8, rnd archsimd.Int32x4) archsimd.Int32x4 {
	a := rnd.Add(x0.MulWidenLo(c0))
	a = a.Add(x1.MulWidenLo(c1))
	a = a.Add(x2.MulWidenLo(c2))
	a = a.Add(x3.MulWidenLo(c3))
	a = a.Add(x4.MulWidenLo(c4))
	a = a.Add(x5.MulWidenLo(c5))
	a = a.Add(x6.MulWidenLo(c6))
	return a.Add(x7.MulWidenLo(c7))
}

// mc16V8Hi is mc16V8Lo over the high four lanes.
func mc16V8Hi(x0, x1, x2, x3, x4, x5, x6, x7 archsimd.Int16x8, c0, c1, c2, c3, c4, c5, c6, c7, h0, h1, h2, h3, h4, h5, h6, h7 archsimd.Int16x8, rnd archsimd.Int32x4) archsimd.Int32x4 {
	a := rnd.Add(x0.HiToLo().MulWidenLo(h0))
	a = a.Add(x1.HiToLo().MulWidenLo(h1))
	a = a.Add(x2.HiToLo().MulWidenLo(h2))
	a = a.Add(x3.HiToLo().MulWidenLo(h3))
	a = a.Add(x4.HiToLo().MulWidenLo(h4))
	a = a.Add(x5.HiToLo().MulWidenLo(h5))
	a = a.Add(x6.HiToLo().MulWidenLo(h6))
	return a.Add(x7.HiToLo().MulWidenLo(h7))
}

// mc16PrepV8 is mc16PrepV6 over window rows 0..7: each column strip slides
// an eight-row window down the block, two output rows per iteration.
func mc16PrepV8(src []byte, srcStride, w, h int, ph *mc16Phase8, sh int, out []uint16) {
	c0, c1, c2, c3, c4, c5, c6, c7, h0, h1, h2, h3, h4, h5, h6, h7 := mc16LoadTaps8(ph)
	rnd := archsimd.BroadcastInt32x4(int32(1)<<(sh-1) + int32(mc16PrepRoundOffset)<<sh)
	shv := archsimd.BroadcastInt32x4(int32(-sh))
	ss, ws := uint32(srcStride), uint32(w)
	if w == 4 {
		// Two output rows share one vector (the CONV_BUF rows are contiguous).
		out = out[: 4*h : 4*h]
		r0 := mc16L16(src, 0)
		r1 := mc16L16(src, ss)
		r2 := mc16L16(src, 2*ss)
		r3 := mc16L16(src, 3*ss)
		r4 := mc16L16(src, 4*ss)
		r5 := mc16L16(src, 5*ss)
		so := 6 * ss
		r6 := mc16L16(src, so)
		q := 0
		for ; q <= len(out)-8; q += 8 {
			r7 := mc16L16(src, so+ss)
			so += 2 * ss
			r8 := mc16L16(src, so)
			a := mc16V8Lo(r0, r1, r2, r3, r4, r5, r6, r7, c0, c1, c2, c3, c4, c5, c6, c7, h0, h1, h2, h3, h4, h5, h6, h7, rnd)
			b := mc16V8Lo(r1, r2, r3, r4, r5, r6, r7, r8, c0, c1, c2, c3, c4, c5, c6, c7, h0, h1, h2, h3, h4, h5, h6, h7, rnd)
			mc16PackU16(a.Shift(shv), b.Shift(shv)).StoreArray((*[8]uint16)(out[q : q+8]))
			r0, r1, r2, r3, r4, r5, r6 = r2, r3, r4, r5, r6, r7, r8
		}
		if q <= len(out)-4 {
			r7 := mc16L16(src, so+ss)
			a := mc16V8Lo(r0, r1, r2, r3, r4, r5, r6, r7, c0, c1, c2, c3, c4, c5, c6, c7, h0, h1, h2, h3, h4, h5, h6, h7, rnd).Shift(shv)
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
		r4 := mc16L16(src, so+4*ss)
		r5 := mc16L16(src, so+5*ss)
		so += 6 * ss
		r6 := mc16L16(src, so)
		o := col
		r := 0
		for ; r+2 <= h; r += 2 {
			r7 := mc16L16(src, so+ss)
			so += 2 * ss
			r8 := mc16L16(src, so)
			alo := mc16V8Lo(r0, r1, r2, r3, r4, r5, r6, r7, c0, c1, c2, c3, c4, c5, c6, c7, h0, h1, h2, h3, h4, h5, h6, h7, rnd)
			ahi := mc16V8Hi(r0, r1, r2, r3, r4, r5, r6, r7, c0, c1, c2, c3, c4, c5, c6, c7, h0, h1, h2, h3, h4, h5, h6, h7, rnd)
			mc16PackU16(alo.Shift(shv), ahi.Shift(shv)).StoreArray(mc8WU8(out, o))
			blo := mc16V8Lo(r1, r2, r3, r4, r5, r6, r7, r8, c0, c1, c2, c3, c4, c5, c6, c7, h0, h1, h2, h3, h4, h5, h6, h7, rnd)
			bhi := mc16V8Hi(r1, r2, r3, r4, r5, r6, r7, r8, c0, c1, c2, c3, c4, c5, c6, c7, h0, h1, h2, h3, h4, h5, h6, h7, rnd)
			mc16PackU16(blo.Shift(shv), bhi.Shift(shv)).StoreArray(mc8WU8(out, o+ws))
			o += 2 * ws
			r0, r1, r2, r3, r4, r5, r6 = r2, r3, r4, r5, r6, r7, r8
		}
		if r < h {
			r7 := mc16L16(src, so+ss)
			alo := mc16V8Lo(r0, r1, r2, r3, r4, r5, r6, r7, c0, c1, c2, c3, c4, c5, c6, c7, h0, h1, h2, h3, h4, h5, h6, h7, rnd)
			ahi := mc16V8Hi(r0, r1, r2, r3, r4, r5, r6, r7, c0, c1, c2, c3, c4, c5, c6, c7, h0, h1, h2, h3, h4, h5, h6, h7, rnd)
			mc16PackU16(alo.Shift(shv), ahi.Shift(shv)).StoreArray(mc8WU8(out, o))
		}
	}
}

// mc16PutV8 is mc16PutV6 over window rows 0..7.
func mc16PutV8(src []byte, srcStride, w, h int, ph *mc16Phase8, maxPix int16, dst []byte, dstStride int) {
	c0, c1, c2, c3, c4, c5, c6, c7, h0, h1, h2, h3, h4, h5, h6, h7 := mc16LoadTaps8(ph)
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
		r4 := mc16L16(src, 4*ss)
		r5 := mc16L16(src, 5*ss)
		so := 6 * ss
		r6 := mc16L16(src, so)
		var d uint32
		r := 0
		for ; r+2 <= h; r += 2 {
			r7 := mc16L16(src, so+ss)
			so += 2 * ss
			r8 := mc16L16(src, so)
			a := mc16V8Lo(r0, r1, r2, r3, r4, r5, r6, r7, c0, c1, c2, c3, c4, c5, c6, c7, h0, h1, h2, h3, h4, h5, h6, h7, rnd)
			b := mc16V8Lo(r1, r2, r3, r4, r5, r6, r7, r8, c0, c1, c2, c3, c4, c5, c6, c7, h0, h1, h2, h3, h4, h5, h6, h7, rnd)
			v := mc16Clip(a.Shift(shv), b.Shift(shv), zero, maxV).ToBits().ReshapeToUint64s()
			binary.LittleEndian.PutUint64(mc8W8(dst, d)[:], v.GetElem(0))
			binary.LittleEndian.PutUint64(mc8W8(dst, d+ds)[:], v.GetElem(1))
			d += 2 * ds
			r0, r1, r2, r3, r4, r5, r6 = r2, r3, r4, r5, r6, r7, r8
		}
		if r < h {
			r7 := mc16L16(src, so+ss)
			a := mc16V8Lo(r0, r1, r2, r3, r4, r5, r6, r7, c0, c1, c2, c3, c4, c5, c6, c7, h0, h1, h2, h3, h4, h5, h6, h7, rnd).Shift(shv)
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
		r4 := mc16L16(src, so+4*ss)
		r5 := mc16L16(src, so+5*ss)
		so += 6 * ss
		r6 := mc16L16(src, so)
		d := col
		r := 0
		for ; r+2 <= h; r += 2 {
			r7 := mc16L16(src, so+ss)
			so += 2 * ss
			r8 := mc16L16(src, so)
			alo := mc16V8Lo(r0, r1, r2, r3, r4, r5, r6, r7, c0, c1, c2, c3, c4, c5, c6, c7, h0, h1, h2, h3, h4, h5, h6, h7, rnd)
			ahi := mc16V8Hi(r0, r1, r2, r3, r4, r5, r6, r7, c0, c1, c2, c3, c4, c5, c6, c7, h0, h1, h2, h3, h4, h5, h6, h7, rnd)
			mc16St16(dst, d, mc16Clip(alo.Shift(shv), ahi.Shift(shv), zero, maxV))
			blo := mc16V8Lo(r1, r2, r3, r4, r5, r6, r7, r8, c0, c1, c2, c3, c4, c5, c6, c7, h0, h1, h2, h3, h4, h5, h6, h7, rnd)
			bhi := mc16V8Hi(r1, r2, r3, r4, r5, r6, r7, r8, c0, c1, c2, c3, c4, c5, c6, c7, h0, h1, h2, h3, h4, h5, h6, h7, rnd)
			mc16St16(dst, d+ds, mc16Clip(blo.Shift(shv), bhi.Shift(shv), zero, maxV))
			d += 2 * ds
			r0, r1, r2, r3, r4, r5, r6 = r2, r3, r4, r5, r6, r7, r8
		}
		if r < h {
			r7 := mc16L16(src, so+ss)
			alo := mc16V8Lo(r0, r1, r2, r3, r4, r5, r6, r7, c0, c1, c2, c3, c4, c5, c6, c7, h0, h1, h2, h3, h4, h5, h6, h7, rnd)
			ahi := mc16V8Hi(r0, r1, r2, r3, r4, r5, r6, r7, c0, c1, c2, c3, c4, c5, c6, c7, h0, h1, h2, h3, h4, h5, h6, h7, rnd)
			mc16St16(dst, d, mc16Clip(alo.Shift(shv), ahi.Shift(shv), zero, maxV))
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

// mc16V8FromI32 runs the eight-tap 2D vertical pass over the strip-major
// int32 intermediate (strips of eight columns, or one strip of four at width
// 4) in strips of four columns, two output rows per iteration: the nine-row
// window stays in registers, as in the width-4 six-tap kernels. Each output
// is the high half of sum(c * m) + rnd; a pair of rows packs into one vector
// (UZP2). With clip (put) it is clamped to [0, maxPix] and stored to the
// dst plane rows (dstStride bytes apart); otherwise (prep) it is the
// CONV_BUF out (row stride w).
func mc16V8FromI32(im []int32, w, h int, c *[8][4]int32, rndC int32, clip bool, maxPix int16, dst []byte, dstStride int, out []uint16) {
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
	rows := h + 7
	sw := min(w, 8) // strip width
	n := rows * sw
	ds, ws := uint32(dstStride), uint32(w)
	for col := 0; col < w; col += 4 {
		// Strip col/8, lanes col%8 .. col%8+3 of each of its rows.
		base := (col&^7)*rows + col&7
		st := im[base : base+n-(col&7) : base+n-(col&7)]
		x0 := archsimd.LoadInt32x4Array((*[4]int32)(st[0:4]))
		q := sw
		x1 := archsimd.LoadInt32x4Array((*[4]int32)(st[q : q+4]))
		x2 := archsimd.LoadInt32x4Array((*[4]int32)(st[2*sw : 2*sw+4]))
		x3 := archsimd.LoadInt32x4Array((*[4]int32)(st[3*sw : 3*sw+4]))
		x4 := archsimd.LoadInt32x4Array((*[4]int32)(st[4*sw : 4*sw+4]))
		x5 := archsimd.LoadInt32x4Array((*[4]int32)(st[5*sw : 5*sw+4]))
		x6 := archsimd.LoadInt32x4Array((*[4]int32)(st[6*sw : 6*sw+4]))
		q = 7 * sw
		dO, oO := uint32(2*col), uint32(col)
		r := 0
		for ; r+2 <= h; r += 2 {
			x7 := archsimd.LoadInt32x4Array((*[4]int32)(st[q : q+4]))
			x8 := archsimd.LoadInt32x4Array((*[4]int32)(st[q+sw : q+sw+4]))
			q += 2 * sw
			a := mc16V8I32(x0, x1, x2, x3, x4, x5, x6, x7, c0, c1, c2, c3, c4, c5, c6, c7, rnd)
			b := mc16V8I32(x1, x2, x3, x4, x5, x6, x7, x8, c0, c1, c2, c3, c4, c5, c6, c7, rnd)
			v := a.ToBits().ReshapeToUint16s().ConcatOdd(b.ToBits().ReshapeToUint16s())
			if clip {
				u := v.BitsToInt16().Max(zero).Min(maxV).ToBits().ReshapeToUint64s()
				binary.LittleEndian.PutUint64(mc8W8(dst, dO)[:], u.GetElem(0))
				binary.LittleEndian.PutUint64(mc8W8(dst, dO+ds)[:], u.GetElem(1))
				dO += 2 * ds
			} else {
				u := v.ReshapeToUint64s()
				mc8Store4U16(mc8WU4(out, oO), u.GetElem(0))
				mc8Store4U16(mc8WU4(out, oO+ws), u.GetElem(1))
				oO += 2 * ws
			}
			x0, x1, x2, x3, x4, x5, x6 = x2, x3, x4, x5, x6, x7, x8
		}
		if r < h {
			x7 := archsimd.LoadInt32x4Array((*[4]int32)(st[q : q+4]))
			a := mc16V8I32(x0, x1, x2, x3, x4, x5, x6, x7, c0, c1, c2, c3, c4, c5, c6, c7, rnd)
			v := a.ToBits().ReshapeToUint16s().ConcatOdd(a.ToBits().ReshapeToUint16s())
			if clip {
				v = v.BitsToInt16().Max(zero).Min(maxV).ToBits()
				binary.LittleEndian.PutUint64(mc8W8(dst, dO)[:], v.ReshapeToUint64s().GetElem(0))
			} else {
				mc8Store4U16(mc8WU4(out, oO), v.ReshapeToUint64s().GetElem(0))
			}
		}
	}
}

// dav1d-style 8-bit translational motion compensation (src/mc_tmpl.c put_8tap
// and prep_8tap, src/arm/64/mc.S filter_fn). The decoder spends most of its
// inter-prediction time in the regular and smooth filters, whose outer taps are
// zero, so like dav1d these kernels run them as six taps (window taps 1..6).
// The <=4 filters and bilinear also fit that window. Sharp (eight taps), width
// 4 and the high-bit-depth paths keep the general kernels.
//
// Arithmetic follows the libaom-exact single-prediction and CONV_BUF formulas
// of convolve{X,Y,2D}8PureGo and predictInterCompoundRef8ToConvBuf*PureGo. All
// AV1 subpel taps are even, so the horizontal and 1D passes use halved taps in
// int16 lanes (the sums are exact modulo 2^16 and the final sums fit). The 2D
// intermediate is the rounded horizontal sum widened to int32 so the vertical
// pass is one multiply-accumulate per tap and lane half:
//
//	put H:   clip((s + 34) >> 6)               s = sum(half tap * px)
//	put V:   clip((s + 32) >> 6)
//	put 2D:  m = (s + 2) >> 2;  clip((sum(32 * tap * m) + 2^15) >> 16)
//	prep H:  ((s + 2) >> 2) + 6144              (as (s + 24578) >> 2, unsigned)
//	prep V:  ((s + 2) >> 2) + 6144
//	prep 2D: (sum(tap * m) + 64 + 6144 << 7) >> 7
//
// put H folds libaom's round3-then-round4 into one shift (nested floor
// division), put 2D reads the >>16 straight from the high half of each int32
// lane, and prep H/V fold the CONV_BUF offset into the rounding constant.
//
// Every access is a bounds-checked fixed-size window of a row slice: the
// source window from mc8Source, the destination rows Put8 checked, and the
// intermediate and CONV_BUF rows of their scratch arrays.

const (
	mc8KindRegular = iota
	mc8KindSmooth
	mc8KindSharp
	mc8KindBilinear
	mc8KindRegular4
	mc8KindSmooth4
	mc8KindCount
)

// mc8Phase is one filter phase pre-broadcast for the six-tap window.
type mc8Phase struct {
	h [6][8]int16 // halved taps
	v [6][4]int32 // put 2D vertical: 32 * tap
	p [6][4]int32 // prep 2D vertical: 512 * tap
	w [6][4]int32 // 16bpc one-axis passes: halved taps on int32 lanes
}

var mc8Phases [mc8KindCount][16]mc8Phase

// mc8Vec holds the kernels' constant rounding and shift vectors. A kernel
// loads them (one table address, one vector load each) rather than
// broadcasting: a broadcast builds the value in a general register, inserts
// it into a lane and duplicates it.
var mc8Vec = struct {
	rnd32, rnd34, prepRnd, rnd2, sh6, sh2 [8]int16
	w32, wsh6, w15                        [4]int32
	mul16                                 [16]uint8
	off6144                               [8]uint16
}{
	rnd2:    [8]int16{2, 2, 2, 2, 2, 2, 2, 2},
	w15:     [4]int32{1 << 15, 1 << 15, 1 << 15, 1 << 15},
	mul16:   [16]uint8{16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16},
	off6144: [8]uint16{6144, 6144, 6144, 6144, 6144, 6144, 6144, 6144},
	rnd32:   [8]int16{32, 32, 32, 32, 32, 32, 32, 32},
	rnd34:   [8]int16{34, 34, 34, 34, 34, 34, 34, 34},
	prepRnd: [8]int16{2 + 6144<<2, 2 + 6144<<2, 2 + 6144<<2, 2 + 6144<<2, 2 + 6144<<2, 2 + 6144<<2, 2 + 6144<<2, 2 + 6144<<2},
	sh6:     [8]int16{-6, -6, -6, -6, -6, -6, -6, -6},
	sh2:     [8]int16{-2, -2, -2, -2, -2, -2, -2, -2},
	w32:     [4]int32{32, 32, 32, 32},
	wsh6:    [4]int32{-6, -6, -6, -6},
}

// mc8SixTap reports whether every phase of kind lies inside window taps 1..6.
var mc8SixTap [mc8KindCount]bool

func init() {
	tables := [mc8KindCount]*[16][filterTaps]int16{
		mc8KindRegular:  &subpelFilters8,
		mc8KindSmooth:   &subpelFilters8Smooth,
		mc8KindSharp:    &subpelFilters8Sharp,
		mc8KindBilinear: &bilinearFilters,
		mc8KindRegular4: &subpelFilters4,
		mc8KindSmooth4:  &subpelFilters4Smooth,
	}
	for kind, tab := range tables {
		six := true
		for s := range 16 {
			k := &tab[s]
			if k[0] != 0 || k[7] != 0 {
				six = false
			}
			for _, c := range k {
				if c&1 != 0 {
					six = false
				}
			}
			ph := &mc8Phases[kind][s]
			for i := range 6 {
				c := k[1+i]
				for l := range 8 {
					ph.h[i][l] = c / 2
				}
				for l := range 4 {
					ph.v[i][l] = int32(c) * 32
					ph.p[i][l] = int32(c) * 512
					ph.w[i][l] = int32(c / 2)
				}
			}
		}
		mc8SixTap[kind] = six
	}
}

// mc8Kind mirrors interpKernel's table selection: one table load indexed by
// the filter (values past the last filter select bilinear, as interpKernel's
// default arm does) and whether the block is at most four samples wide.
func mc8Kind(f InterpFilter, blockSize int) int {
	small := 0
	if blockSize <= 4 {
		small = 1
	}
	return int(mc8KindTab[small][min(f, InterpBilinear)])
}

var mc8KindTab = [2][interpFilterCount]uint8{
	{mc8KindRegular, mc8KindSmooth, mc8KindSharp, mc8KindBilinear},
	{mc8KindRegular4, mc8KindSmooth4, mc8KindRegular4, mc8KindBilinear},
}

// mc8Source resolves the six-tap source window of a w x h block whose
// first output sample reads ref at (refX, refY). halo selects the filtered
// axes. The window is returned as a Grid view from its top-left tap sample:
// bh rows (h, plus 5 on a filtered vertical axis) of max(w+8, 16) bytes, the
// bytes every kernel may load (16 per eight output columns). When any tap
// falls outside the plane the clamped window is first materialized in edge
// (dav1d src/recon_tmpl.c mc() emu_edge). ok is false only when the reference
// plane itself is unusable.
func mc8Source(ref *frame.Plane, refX, refY, w, h int, hx, hy bool, edge *[emuEdgeStride * emuEdgeRows]byte) (src unsafeutil.Grid[uint8], ok bool) {
	x0, y0, bw, bh := refX, refY, w, h
	if hx {
		x0 -= 2
		bw += 5
	}
	if hy {
		y0 -= 2
		bh += 5
	}
	if ref.Width <= 0 || ref.Height <= 0 || ref.Stride < ref.Width {
		return src, false
	}
	sw := max(w+8, 16)
	if x0 >= 0 && y0 >= 0 && x0+bw <= ref.Width && y0+bh <= ref.Height {
		start := y0*ref.Stride + x0
		if unsafeutil.GridFit(len(ref.Pix), start, ref.Stride, sw, bh) {
			return unsafeutil.NewGrid(ref.Pix, start, ref.Stride, sw, bh), true
		}
	}
	if len(ref.Pix) < (ref.Height-1)*ref.Stride+ref.Width {
		return src, false
	}
	// The edge window holds 135 rows of 136 bytes; bh <= 133 and the
	// kernels read at most max(w+8, 16) <= 136 bytes per row.
	emuEdge8(bw, bh, *ref, x0, y0, edge[:], emuEdgeStride)
	if !unsafeutil.GridFit(len(edge), 0, emuEdgeStride, sw, bh) {
		return src, false
	}
	return unsafeutil.NewGrid(edge[:], 0, emuEdgeStride, sw, bh), true
}

// Put8 is dav1d's mc() put for an unscaled 8-bit single-reference block:
// the copy (integer vector) or the put kernels for the block's filters, from
// the reference origin (refX, refY) and Q4 subpel offsets (subX, subY). It
// returns false, with nothing written, for shapes it does not cover; the
// caller then runs the general path.
func Put8(dst *frame.Plane, ref *frame.Plane, dstX, dstY, refX, refY, w, h, filterW, filterH, subX, subY int, filters InterpFilters, scratch *ConvolveScratch) bool {
	if subX|subY == 0 {
		// mc8CopyFast written out for the power-of-two widths (every AV1
		// block width): one call, CopyRect's.
		if w&(w-1) != 0 || w&3 != 0 || w > maxBlockSize || h <= 0 ||
			dstX < 0 || dstY < 0 || dstX+w > dst.Width || dstY+h > dst.Height || dst.Stride < dst.Width ||
			refX < 0 || refY < 0 || refX+w > ref.Width || refY+h > ref.Height || ref.Stride < ref.Width {
			return mc8CopyFast(dst, ref, dstX, dstY, refX, refY, w, h)
		}
		return unsafeutil.CopyRect(dst.Pix, dstY*dst.Stride+dstX, dst.Stride, ref.Pix, refY*ref.Stride+refX, ref.Stride, w, h)
	}
	if w == 2 {
		return mc8PutW2(dst, ref, dstX, dstY, refX, refY, h, filterW, filterH, subX, subY, filters, scratch)
	}
	if (w&7 != 0 && w != 4) || w > maxBlockSize || h <= 0 || h > maxBlockSize || scratch == nil {
		return false
	}
	kx := mc8Kind(filters.X, filterW)
	ky := mc8Kind(filters.Y, filterH)
	if mc8NeedsEightTaps(subX, subY, kx, ky) {
		return w&7 == 0 && mc8Put8TapChecked(dst, ref, dstX, dstY, refX, refY, w, h, subX, subY, kx, ky, scratch)
	}
	// The six-tap source window (mc8Source), in place when it lies inside
	// the plane: halo 2 before and 3 after each sample on a filtered axis.
	// The common case, destination and source window both in their planes,
	// is checked in line with one footprint test for the pair and runs the
	// kernel with no call before it; everything else is mc8PutEdge.
	hx, hy := mc8Halo(subX), mc8Halo(subY)
	x0, y0, bh := refX-hx, refY-hy, h+hy+hy+hy>>1
	dOff, sOff, sw := dstY*dst.Stride+dstX, y0*ref.Stride+x0, max(w+8, 16)
	if dstX < 0 || dstY < 0 || dstX+w > dst.Width || dstY+h > dst.Height || dst.Stride < dst.Width ||
		x0 < 0 || y0 < 0 || x0+w+hx+hx+hx>>1 > ref.Width || y0+bh > ref.Height || ref.Stride < ref.Width ||
		!unsafeutil.GridFit(len(dst.Pix), dOff, dst.Stride, w, h) || !unsafeutil.GridFit(len(ref.Pix), sOff, ref.Stride, sw, bh) {
		return mc8PutEdge(dst, ref, dstX, dstY, refX, refY, w, h, subX, subY, kx, ky, scratch)
	}
	dg, sg := unsafeutil.NewGrid(dst.Pix, dOff, dst.Stride, w, h), unsafeutil.NewGrid(ref.Pix, sOff, ref.Stride, sw, bh)
	// mc8PutGrid written out: one call level less for the common kernels.
	phx, phy := &mc8Phases[kx][subX&15], &mc8Phases[ky][subY&15]
	switch {
	case subX != 0 && subY != 0:
		if w == 4 {
			if kx >= mc8KindBilinear {
				if ky >= mc8KindBilinear {
					mc8PutHV4x4W4(sg, h, phx, phy, dg)
				} else {
					mc8PutHV4W4(sg, h, phx, phy, dg)
				}
			} else {
				im := scratch.imHBD[:]
				mc8H6ToI32W4(sg, h+5, phx, im)
				mc8PutV6FromI32W4(im, h, phy, dg)
			}
		} else if kx == mc8KindBilinear && ky == mc8KindBilinear {
			mc8PutHV2(sg, w, h, phx, phy, dg)
		} else if ky >= mc8KindBilinear {
			mc8PutHV6x4(sg, w, h, phx, phy, dg)
		} else if phy.v[0][0]|phy.v[5][0] == 0 {
			mc8PutHV6x4(sg, w, h, phx, phy, dg)
		} else {
			mc8PutHV6(sg, w, h, phx, phy, dg)
		}
	case subX != 0:
		switch {
		case w == 4:
			if phx.h[0][0]|phx.h[5][0] == 0 {
				mc8PutH4S4(sg, phx, dg)
			} else {
				mc8PutH6S4(sg, phx, dg)
			}
		case w == 8:
			mc8PutH6S8(sg, phx, dg)
		case w&15 == 0:
			mc8PutH6W16(sg, phx, dg)
		default:
			mc8PutH6Split(sg, phx, dg)
		}
	default:
		if phy.h[0][0]|phy.h[5][0] == 0 {
			mc8PutV4(sg, w, h, phy, dg)
		} else {
			switch {
			case w == 4:
				mc8PutV6S4(sg, phy, dg)
			case w == 8:
				mc8PutV6S8(sg, phy, dg)
			case w&15 == 0:
				mc8PutV6W16(sg, phy, dg)
			default:
				mc8PutV6Split(sg, phy, dg)
			}
		}
	}
	return true
}

// mc8PutEdge is Put8's six-tap path for a block whose destination or source
// window is not inside its plane (or whose views do not fit): the checked
// destination view and the source window from mc8Source, which clamps it.
func mc8PutEdge(dst *frame.Plane, ref *frame.Plane, dstX, dstY, refX, refY, w, h, subX, subY, kx, ky int, scratch *ConvolveScratch) bool {
	dg, ok := mc8DstGrid(dst, dstX, dstY, w, h)
	if !ok {
		return false
	}
	sg, ok := mc8Source(ref, refX, refY, w, h, subX != 0, subY != 0, &scratch.edge)
	if !ok {
		return false
	}
	mc8PutGrid(dg, sg, w, h, subX, subY, &mc8Phases[kx][subX&15], &mc8Phases[ky][subY&15], scratch)
	return true
}

// mc8SameGeometry reports whether two planes have the same dimensions,
// stride and buffer length, so one footprint check covers both.
func mc8SameGeometry(a, b *frame.Plane) bool {
	return a.Width == b.Width && a.Height == b.Height && a.Stride == b.Stride && len(a.Pix) == len(b.Pix)
}

// Put8UV is Put8 for the U and V blocks of one chroma prediction: both planes
// share the block, the reference origin, the phase and the filters (dav1d
// recon_b_inter's two chroma mc() calls). When the two destination planes and
// the two reference planes have the same geometry, the shape, kernel and phase
// selection and every footprint check run once for the pair; the per-plane
// kernels then run back to back. It returns false when some plane was not
// written; the caller then predicts both planes plane by plane (rewriting a
// plane already written here yields the same samples).
func Put8UV(dstU, dstV, refU, refV *frame.Plane, dstX, dstY, refX, refY, w, h, filterW, filterH, subX, subY int, filters InterpFilters, scratch *ConvolveScratch) bool {
	if w == 2 || !mc8SameGeometry(dstU, dstV) || !mc8SameGeometry(refU, refV) {
		return Put8(dstU, refU, dstX, dstY, refX, refY, w, h, filterW, filterH, subX, subY, filters, scratch) &&
			Put8(dstV, refV, dstX, dstY, refX, refY, w, h, filterW, filterH, subX, subY, filters, scratch)
	}
	if subX|subY == 0 {
		if w&3 != 0 || w > maxBlockSize || h <= 0 {
			return false
		}
		if dstX < 0 || dstY < 0 || dstX+w > dstU.Width || dstY+h > dstU.Height || dstU.Stride < dstU.Width ||
			refX < 0 || refY < 0 || refX+w > refU.Width || refY+h > refU.Height || refU.Stride < refU.Width {
			return false
		}
		// The planes of each pair share their geometry, so the V views fit
		// exactly when the U views do.
		dOff, sOff := dstY*dstU.Stride+dstX, refY*refU.Stride+refX
		if w&(w-1) == 0 { // every AV1 block width
			return unsafeutil.CopyRect(dstU.Pix, dOff, dstU.Stride, refU.Pix, sOff, refU.Stride, w, h) &&
				unsafeutil.CopyRect(dstV.Pix, dOff, dstV.Stride, refV.Pix, sOff, refV.Stride, w, h)
		}
		if !unsafeutil.GridFit(len(dstU.Pix), dOff, dstU.Stride, w, h) || !unsafeutil.GridFit(len(refU.Pix), sOff, refU.Stride, w, h) {
			return false
		}
		du, su := unsafeutil.NewGrid(dstU.Pix, dOff, dstU.Stride, w, h), unsafeutil.NewGrid(refU.Pix, sOff, refU.Stride, w, h)
		dv, sv := unsafeutil.NewGrid(dstV.Pix, dOff, dstV.Stride, w, h), unsafeutil.NewGrid(refV.Pix, sOff, refV.Stride, w, h)
		return mc8CopyStrips(du, su) && mc8CopyStrips(dv, sv)
	}
	if (w&7 != 0 && w != 4) || w > maxBlockSize || h <= 0 || h > maxBlockSize || scratch == nil {
		return false
	}
	kx := mc8Kind(filters.X, filterW)
	ky := mc8Kind(filters.Y, filterH)
	if mc8NeedsEightTaps(subX, subY, kx, ky) {
		return w&7 == 0 && mc8Put8TapChecked(dstU, refU, dstX, dstY, refX, refY, w, h, subX, subY, kx, ky, scratch) &&
			mc8Put8TapChecked(dstV, refV, dstX, dstY, refX, refY, w, h, subX, subY, kx, ky, scratch)
	}
	// The source window, in-plane for both planes at once: that case checks
	// the U views in line (the V planes share their geometry, so the V views
	// fit exactly when they do) and runs the two planes' kernels back to back
	// with no call before them; mc8PutUVEdge takes the rest.
	hx, hy := mc8Halo(subX), mc8Halo(subY)
	x0, y0, bh := refX-hx, refY-hy, h+hy+hy+hy>>1
	dOff, sOff, sw := dstY*dstU.Stride+dstX, y0*refU.Stride+x0, max(w+8, 16)
	if dstX < 0 || dstY < 0 || dstX+w > dstU.Width || dstY+h > dstU.Height || dstU.Stride < dstU.Width ||
		x0 < 0 || y0 < 0 || x0+w+hx+hx+hx>>1 > refU.Width || y0+bh > refU.Height || refU.Stride < refU.Width ||
		!unsafeutil.GridFit(len(dstU.Pix), dOff, dstU.Stride, w, h) || !unsafeutil.GridFit(len(refU.Pix), sOff, refU.Stride, sw, bh) {
		return mc8PutUVEdge(dstU, dstV, refU, refV, dstX, dstY, refX, refY, w, h, subX, subY, kx, ky, scratch)
	}
	du, su := unsafeutil.NewGrid(dstU.Pix, dOff, dstU.Stride, w, h), unsafeutil.NewGrid(refU.Pix, sOff, refU.Stride, sw, bh)
	dv, sv := unsafeutil.NewGrid(dstV.Pix, dOff, dstV.Stride, w, h), unsafeutil.NewGrid(refV.Pix, sOff, refV.Stride, sw, bh)
	phx, phy := &mc8Phases[kx][subX&15], &mc8Phases[ky][subY&15]
	switch {
	case subX != 0 && subY != 0:
		if w == 4 {
			if kx >= mc8KindBilinear {
				if ky >= mc8KindBilinear {
					mc8PutHV4x4W4(su, h, phx, phy, du)
					mc8PutHV4x4W4(sv, h, phx, phy, dv)
				} else {
					mc8PutHV4W4(su, h, phx, phy, du)
					mc8PutHV4W4(sv, h, phx, phy, dv)
				}
			} else {
				im := scratch.imHBD[:]
				mc8H6ToI32W4(su, h+5, phx, im)
				mc8PutV6FromI32W4(im, h, phy, du)
				mc8H6ToI32W4(sv, h+5, phx, im)
				mc8PutV6FromI32W4(im, h, phy, dv)
			}
		} else if kx == mc8KindBilinear && ky == mc8KindBilinear {
			mc8PutHV2(su, w, h, phx, phy, du)
			mc8PutHV2(sv, w, h, phx, phy, dv)
		} else if ky >= mc8KindBilinear {
			mc8PutHV6x4(su, w, h, phx, phy, du)
			mc8PutHV6x4(sv, w, h, phx, phy, dv)
		} else {
			mc8PutHV6(su, w, h, phx, phy, du)
			mc8PutHV6(sv, w, h, phx, phy, dv)
		}
	case subX != 0:
		switch {
		case w == 4:
			if kx >= mc8KindBilinear {
				mc8PutH4S4(su, phx, du)
				mc8PutH4S4(sv, phx, dv)
			} else {
				mc8PutH6S4(su, phx, du)
				mc8PutH6S4(sv, phx, dv)
			}
		case w == 8:
			mc8PutH6S8(su, phx, du)
			mc8PutH6S8(sv, phx, dv)
		case w&15 == 0:
			mc8PutH6W16(su, phx, du)
			mc8PutH6W16(sv, phx, dv)
		default:
			mc8PutH6Split(su, phx, du)
			mc8PutH6Split(sv, phx, dv)
		}
	default:
		if ky >= mc8KindBilinear {
			mc8PutV4(su, w, h, phy, du)
			mc8PutV4(sv, w, h, phy, dv)
		} else {
			switch {
			case w == 4:
				mc8PutV6S4(su, phy, du)
				mc8PutV6S4(sv, phy, dv)
			case w == 8:
				mc8PutV6S8(su, phy, du)
				mc8PutV6S8(sv, phy, dv)
			case w&15 == 0:
				mc8PutV6W16(su, phy, du)
				mc8PutV6W16(sv, phy, dv)
			default:
				mc8PutV6Split(su, phy, du)
				mc8PutV6Split(sv, phy, dv)
			}
		}
	}
	return true
}

// mc8PutUVEdge is Put8UV's six-tap path when the destination or the shared
// source window is not inside its planes (or the views do not fit): the
// checked destination views, and each plane's clamped window materialized in
// turn in the shared edge buffer.
func mc8PutUVEdge(dstU, dstV, refU, refV *frame.Plane, dstX, dstY, refX, refY, w, h, subX, subY, kx, ky int, scratch *ConvolveScratch) bool {
	dOff := dstY*dstU.Stride + dstX
	if dstX < 0 || dstY < 0 || dstX+w > dstU.Width || dstY+h > dstU.Height || dstU.Stride < dstU.Width ||
		!unsafeutil.GridFit(len(dstU.Pix), dOff, dstU.Stride, w, h) {
		return false
	}
	phx, phy := &mc8Phases[kx][subX&15], &mc8Phases[ky][subY&15]
	for p := 0; p < 2; p++ {
		dst, ref := dstU, refU
		if p == 1 {
			dst, ref = dstV, refV
		}
		sg, ok := mc8Source(ref, refX, refY, w, h, subX != 0, subY != 0, &scratch.edge)
		if !ok {
			return false
		}
		mc8PutGrid(unsafeutil.NewGrid(dst.Pix, dOff, dst.Stride, w, h), sg, w, h, subX, subY, phx, phy, scratch)
	}
	return true
}

// mc8DstGrid is Put8's destination check: the w x h block at (x, y) of dst as
// a Grid view.
func mc8DstGrid(dst *frame.Plane, x, y, w, h int) (unsafeutil.Grid[uint8], bool) {
	off := y*dst.Stride + x
	if x < 0 || y < 0 || x+w > dst.Width || y+h > dst.Height || dst.Stride < dst.Width ||
		!unsafeutil.GridFit(len(dst.Pix), off, dst.Stride, w, h) {
		return unsafeutil.Grid[uint8]{}, false
	}
	return unsafeutil.NewGrid(dst.Pix, off, dst.Stride, w, h), true
}

// mc8Halo is the source window halo of an axis: 2 when its phase is nonzero.
func mc8Halo(sub int) int {
	if sub != 0 {
		return 2
	}
	return 0
}

// mc8PutGrid runs the six-tap put kernel of a validated block (w 4 or a
// multiple of 8, a nonzero phase on some axis): dg is the w x h destination,
// sg the source window (mc8Source).
func mc8PutGrid(dg, sg unsafeutil.Grid[uint8], w, h, subX, subY int, phx, phy *mc8Phase, scratch *ConvolveScratch) {
	switch {
	case subX != 0 && subY != 0:
		if w == 4 {
			if phx.h[0][0]|phx.h[5][0] == 0 {
				mc8PutHV4W4(sg, h, phx, phy, dg)
			} else {
				im := scratch.imHBD[:]
				mc8H6ToI32W4(sg, h+5, phx, im)
				mc8PutV6FromI32W4(im, h, phy, dg)
			}
		} else if phy.v[0][0]|phy.v[5][0] == 0 {
			mc8PutHV6x4(sg, w, h, phx, phy, dg)
		} else {
			mc8PutHV6(sg, w, h, phx, phy, dg)
		}
	case subX != 0:
		switch {
		case w == 4:
			if phx.h[0][0]|phx.h[5][0] == 0 {
				mc8PutH4S4(sg, phx, dg)
			} else {
				mc8PutH6S4(sg, phx, dg)
			}
		case w == 8:
			mc8PutH6S8(sg, phx, dg)
		case w&15 == 0:
			mc8PutH6W16(sg, phx, dg)
		default:
			mc8PutH6Split(sg, phx, dg)
		}
	default:
		if phy.h[0][0]|phy.h[5][0] == 0 {
			mc8PutV4(sg, w, h, phy, dg)
		} else {
			switch {
			case w == 4:
				mc8PutV6S4(sg, phy, dg)
			case w == 8:
				mc8PutV6S8(sg, phy, dg)
			case w&15 == 0:
				mc8PutV6W16(sg, phy, dg)
			default:
				mc8PutV6Split(sg, phy, dg)
			}
		}
	}
}

// mc8PutH6Split is the put horizontal pass of a block 16k+8 wide (no AV1
// block is; the kernels allow it): the sixteen-column strips, then the last
// eight columns.
func mc8PutH6Split(sg unsafeutil.Grid[uint8], ph *mc8Phase, dg unsafeutil.Grid[uint8]) {
	w, h := dg.Width(), dg.Height()
	w16 := w &^ 15
	mc8PutH6W16(sg, ph, dg.Sub(0, 0, w16, h))
	mc8PutH6S8(sg.Sub(0, w16, 16, h), ph, dg.Sub(0, w16, 8, h))
}

// mc8PutV6Split is mc8PutH6Split for the vertical pass.
func mc8PutV6Split(sg unsafeutil.Grid[uint8], ph *mc8Phase, dg unsafeutil.Grid[uint8]) {
	w, h := dg.Width(), dg.Height()
	w16 := w &^ 15
	mc8PutV6W16(sg, ph, dg.Sub(0, 0, w16, h))
	mc8PutV6S8(sg.Sub(0, w16, 16, h+5), ph, dg.Sub(0, w16, 8, h))
}

// mc8PrepH6Split is mc8PutH6Split for the prep horizontal pass.
func mc8PrepH6Split(sg unsafeutil.Grid[uint8], ph *mc8Phase, og unsafeutil.Grid[uint16]) {
	w, h := og.Width(), og.Height()
	w16 := w &^ 15
	mc8PrepH6W16(sg, ph, og.Sub(0, 0, w16, h))
	mc8PrepH6S8(sg.Sub(0, w16, 16, h), ph, og.Sub(0, w16, 8, h))
}

// mc8PrepV6Split is mc8PutH6Split for the prep vertical pass.
func mc8PrepV6Split(sg unsafeutil.Grid[uint8], ph *mc8Phase, og unsafeutil.Grid[uint16]) {
	w, h := og.Width(), og.Height()
	w16 := w &^ 15
	mc8PrepV6W16(sg, ph, og.Sub(0, 0, w16, h))
	mc8PrepV6S8(sg.Sub(0, w16, 16, h+5), ph, og.Sub(0, w16, 8, h))
}

// mc8PutW2 is Put8 at width 2, dav1d's put_8tap w2 for the sub-8x8 chroma
// blocks of recon_b_inter: the width-4 kernels filter into a local block of
// 4-byte rows whose first two columns are copied out, so nothing past the
// block is written. Output columns 0 and 1 read only the taps a 2-wide filter
// reads; mc8Source sizes the window for the 4-wide kernels, and any taps it
// edge-extends past the plane feed only the discarded columns.
func mc8PutW2(dst *frame.Plane, ref *frame.Plane, dstX, dstY, refX, refY, h, filterW, filterH, subX, subY int, filters InterpFilters, scratch *ConvolveScratch) bool {
	if h <= 0 || h > mc8W2MaxH || scratch == nil {
		return false
	}
	kx := mc8Kind(filters.X, filterW)
	ky := mc8Kind(filters.Y, filterH)
	if mc8NeedsEightTaps(subX, subY, kx, ky) {
		return false
	}
	if dstX < 0 || dstY < 0 || dstX+2 > dst.Width || dstY+h > dst.Height || dst.Stride < dst.Width {
		return false
	}
	dOff := dstY*dst.Stride + dstX
	if dOff+(h-1)*dst.Stride+2 > len(dst.Pix) {
		return false
	}
	sg, ok := mc8Source(ref, refX, refY, 4, h, subX != 0, subY != 0, &scratch.edge)
	if !ok {
		return false
	}
	var tmp [4 * mc8W2MaxH]byte
	tg := unsafeutil.NewGrid(tmp[:], 0, 4, 4, h)
	switch {
	case subX != 0 && subY != 0:
		im := scratch.imHBD[:]
		mc8H6ToI32W4(sg, h+5, &mc8Phases[kx][subX], im)
		mc8PutV6FromI32W4(im, h, &mc8Phases[ky][subY], tg)
	case subX != 0:
		mc8PutH6S4(sg, &mc8Phases[kx][subX], tg)
	default:
		mc8PutV6S4(sg, &mc8Phases[ky][subY], tg)
	}
	rows := dst.Pix[dOff : dOff+(h-1)*dst.Stride+2]
	for r := 0; r < h; r++ {
		row := rows[r*dst.Stride : r*dst.Stride+2]
		row[0], row[1] = tmp[4*r], tmp[4*r+1]
	}
	return true
}

// mc8W2MaxH bounds mc8PutW2's block height: a 2-wide chroma block comes from
// a 4-wide luma block, at most 4x16 (2x16 under 4:2:2).
const mc8W2MaxH = 16

// mc8CopyFast is the put copy (dav1d put_neon) for an in-plane w x h 8-bit
// block with w a multiple of 4: vector rows instead of one memmove per row.
// It returns false when the source leaves the plane (the clamped copy runs).
func mc8CopyFast(dst *frame.Plane, ref *frame.Plane, dstX, dstY, refX, refY, w, h int) bool {
	if w&3 != 0 || w > maxBlockSize || h <= 0 {
		return false
	}
	if dstX < 0 || dstY < 0 || dstX+w > dst.Width || dstY+h > dst.Height || dst.Stride < dst.Width {
		return false
	}
	if refX < 0 || refY < 0 || refX+w > ref.Width || refY+h > ref.Height || ref.Stride < ref.Width {
		return false
	}
	dOff, sOff := dstY*dst.Stride+dstX, refY*ref.Stride+refX
	if w&(w-1) == 0 { // every AV1 block width
		return unsafeutil.CopyRect(dst.Pix, dOff, dst.Stride, ref.Pix, sOff, ref.Stride, w, h)
	}
	if !unsafeutil.GridFit(len(dst.Pix), dOff, dst.Stride, w, h) || !unsafeutil.GridFit(len(ref.Pix), sOff, ref.Stride, w, h) {
		return false
	}
	return mc8CopyStrips(unsafeutil.NewGrid(dst.Pix, dOff, dst.Stride, w, h), unsafeutil.NewGrid(ref.Pix, sOff, ref.Stride, w, h))
}

// Window helpers. Offsets are uint32 so o+N cannot overflow and each window
// costs one compare; the entries keep every window slice below 1<<32
// elements (mc8WindowMax), so the offsets never wrap.
func mc8W4(s []byte, o uint32) *[4]byte      { return (*[4]byte)(s[uint(o) : uint(o)+4]) }
func mc8W8(s []byte, o uint32) *[8]byte      { return (*[8]byte)(s[uint(o) : uint(o)+8]) }
func mc8W16(s []byte, o uint32) *[16]byte    { return (*[16]byte)(s[uint(o) : uint(o)+16]) }
func mc8W32(s []byte, o uint32) *[32]byte    { return (*[32]byte)(s[uint(o) : uint(o)+32]) }
func mc8WU4(s []uint16, o uint32) *[4]uint16 { return (*[4]uint16)(s[uint(o) : uint(o)+4]) }
func mc8WU8(s []uint16, o uint32) *[8]uint16 { return (*[8]uint16)(s[uint(o) : uint(o)+8]) }

// mc8WindowMax bounds the window slices handed to the kernels (see mc8W16).
const mc8WindowMax = 1<<32 - 1

// Limit-checked windows. A kernel computes lim = len(s)-N once and walks a
// plain int offset; each window is then a sign test and one compare against
// lim, which together prove the slice expression (0 <= o, o+N <= len(s)), so
// no zero-extension or end-offset arithmetic is left per row.
func mc8R8(s []byte, o, lim int) *[8]byte {
	if o < 0 || o > lim {
		panic(errMC8Window)
	}
	return (*[8]byte)(s[o : o+8])
}

func mc8R16(s []byte, o, lim int) *[16]byte {
	if o < 0 || o > lim {
		panic(errMC8Window)
	}
	return (*[16]byte)(s[o : o+16])
}

func mc8R32(s []byte, o, lim int) *[32]byte {
	if o < 0 || o > lim {
		panic(errMC8Window)
	}
	return (*[32]byte)(s[o : o+32])
}

// errMC8Window is the panic value of a window past its limit (a caller bug:
// the entries validate every block footprint).
var errMC8Window = errors.New("motion: kernel window out of range")

// mc8PrepFast fills a w x h CONV_BUF (row stride w) with the dav1d-style prep
// kernels. It returns false for shapes it does not cover.
func mc8PrepFast(out []uint16, ref *frame.Plane, refX, refY, w, h, filterW, filterH, subX, subY int, filters InterpFilters, scratch *CompoundConvolveScratch) bool {
	if (w&7 != 0 && w != 4) || w > maxBlockSize || h <= 0 || h > maxBlockSize || scratch == nil {
		return false
	}
	kx := mc8Kind(filters.X, filterW)
	ky := mc8Kind(filters.Y, filterH)
	if mc8NeedsEightTaps(subX, subY, kx, ky) {
		return w&7 == 0 && len(out) >= w*h && mc8Prep8Tap(out[:w*h], ref, refX, refY, w, h, subX, subY, kx, ky, scratch)
	}
	// The six-tap source window (mc8Source), in place when it lies inside
	// the plane, as in Put8: that case is checked in line and runs its
	// kernel with no call before it; mc8PrepEdge takes the rest.
	hx, hy := mc8Halo(subX), mc8Halo(subY)
	x0, y0, bh := refX-hx, refY-hy, h+hy+hy+hy>>1
	sOff, sw := y0*ref.Stride+x0, max(w+8, 16)
	if x0 < 0 || y0 < 0 || x0+w+hx+hx+hx>>1 > ref.Width || y0+bh > ref.Height || ref.Stride < ref.Width ||
		!unsafeutil.GridFit(len(out), 0, w, w, h) || !unsafeutil.GridFit(len(ref.Pix), sOff, ref.Stride, sw, bh) {
		return mc8PrepEdge(out, ref, refX, refY, w, h, subX, subY, kx, ky, scratch)
	}
	og, sg := unsafeutil.NewGrid(out, 0, w, w, h), unsafeutil.NewGrid(ref.Pix, sOff, ref.Stride, sw, bh)
	// mc8PrepGrid written out, as in Put8.
	switch {
	case subX != 0 && subY != 0:
		if w == 4 {
			if kx >= mc8KindBilinear {
				if ky >= mc8KindBilinear {
					mc8PrepHV4x4W4(sg, h, &mc8Phases[kx][subX], &mc8Phases[ky][subY], out[:4*h])
				} else {
					mc8PrepHV4W4(sg, h, &mc8Phases[kx][subX], &mc8Phases[ky][subY], out[:4*h])
				}
			} else {
				im := scratch.im[:]
				mc8H6ToI32W4(sg, h+5, &mc8Phases[kx][subX], im)
				mc8PrepV6FromI32W4(im, h, &mc8Phases[ky][subY], out[:4*h], mc8PrepV6Rnd)
			}
		} else if kx == mc8KindBilinear && ky == mc8KindBilinear {
			mc8PrepHV2(sg, w, h, &mc8Phases[kx][subX], &mc8Phases[ky][subY], og)
		} else if ky >= mc8KindBilinear {
			mc8PrepHV6x4(sg, w, h, &mc8Phases[kx][subX], &mc8Phases[ky][subY], og)
		} else {
			im := scratch.im[:]
			mc8H6ToI32(sg, w, h+5, &mc8Phases[kx][subX], im, 2)
			mc8PrepV6FromI32S(im, w, h, &mc8Phases[ky][subY], og, mc8PrepV6Rnd)
		}
	case subX != 0:
		ph := &mc8Phases[kx][subX]
		switch {
		case w == 4:
			if kx >= mc8KindBilinear {
				mc8PrepH4S4(sg, ph, og)
			} else {
				mc8PrepH6S4(sg, ph, og)
			}
		case w == 8:
			mc8PrepH6S8(sg, ph, og)
		case w&15 == 0:
			mc8PrepH6W16(sg, ph, og)
		default:
			mc8PrepH6Split(sg, ph, og)
		}
	case subY != 0:
		ph := &mc8Phases[ky][subY]
		if ky >= mc8KindBilinear {
			mc8PrepV4(sg, w, h, ph, og)
		} else {
			switch {
			case w == 4:
				mc8PrepV6S4(sg, ph, og)
			case w == 8:
				mc8PrepV6S8(sg, ph, og)
			case w&15 == 0:
				mc8PrepV6W16(sg, ph, og)
			default:
				mc8PrepV6Split(sg, ph, og)
			}
		}
	default:
		if w == 8 {
			mc8PrepCopyW8(sg, out, h)
		} else {
			mc8PrepCopy(sg, og)
		}
	}
	return true
}

// mc8PrepEdge is mc8PrepFast's six-tap path for a source window that is not
// inside the plane (or views that do not fit): the window from mc8Source,
// which clamps it.
func mc8PrepEdge(out []uint16, ref *frame.Plane, refX, refY, w, h, subX, subY, kx, ky int, scratch *CompoundConvolveScratch) bool {
	if !unsafeutil.GridFit(len(out), 0, w, w, h) {
		return false
	}
	sg, ok := mc8Source(ref, refX, refY, w, h, subX != 0, subY != 0, &scratch.edge)
	if !ok {
		return false
	}
	mc8PrepGrid(out, unsafeutil.NewGrid(out, 0, w, w, h), sg, w, h, subX, subY, &mc8Phases[kx][subX&15], &mc8Phases[ky][subY&15], scratch)
	return true
}

// mc8PrepGrid runs the six-tap prep kernel of a validated block (w 4 or a
// multiple of 8): og is the w x h CONV_BUF view of out, sg the source window
// (mc8Source).
func mc8PrepGrid(out []uint16, og unsafeutil.Grid[uint16], sg unsafeutil.Grid[uint8], w, h, subX, subY int, phx, phy *mc8Phase, scratch *CompoundConvolveScratch) {
	switch {
	case subX != 0 && subY != 0:
		if w == 4 {
			if phx.h[0][0]|phx.h[5][0] == 0 {
				if phy.p[0][0]|phy.p[5][0] == 0 {
					mc8PrepHV4x4W4(sg, h, phx, phy, out[:4*h])
				} else {
					mc8PrepHV4W4(sg, h, phx, phy, out[:4*h])
				}
			} else {
				im := scratch.im[:]
				mc8H6ToI32W4(sg, h+5, phx, im)
				mc8PrepV6FromI32W4(im, h, phy, out[:4*h], mc8PrepV6Rnd)
			}
		} else if phy.p[0][0]|phy.p[5][0] == 0 {
			mc8PrepHV6x4(sg, w, h, phx, phy, og)
		} else {
			im := scratch.im[:]
			mc8H6ToI32(sg, w, h+5, phx, im, 2)
			mc8PrepV6FromI32S(im, w, h, phy, og, mc8PrepV6Rnd)
		}
	case subX != 0:
		switch {
		case w == 4:
			if phx.h[0][0]|phx.h[5][0] == 0 {
				mc8PrepH4S4(sg, phx, og)
			} else {
				mc8PrepH6S4(sg, phx, og)
			}
		case w == 8:
			mc8PrepH6S8(sg, phx, og)
		case w&15 == 0:
			mc8PrepH6W16(sg, phx, og)
		default:
			mc8PrepH6Split(sg, phx, og)
		}
	case subY != 0:
		if phy.h[0][0]|phy.h[5][0] == 0 {
			mc8PrepV4(sg, w, h, phy, og)
		} else {
			switch {
			case w == 4:
				mc8PrepV6S4(sg, phy, og)
			case w == 8:
				mc8PrepV6S8(sg, phy, og)
			case w&15 == 0:
				mc8PrepV6W16(sg, phy, og)
			default:
				mc8PrepV6Split(sg, phy, og)
			}
		}
	default:
		if w == 8 {
			mc8PrepCopyW8(sg, out, h)
		} else {
			mc8PrepCopy(sg, og)
		}
	}
}

// mc8LoadWiden widens the 16 bytes of p to two uint16 vectors, returned as
// bytes for the tap shifts; the six-tap horizontal sums read lanes 0..12.
func mc8LoadWiden(p *[16]uint8) (archsimd.Uint8x16, archsimd.Uint8x16) {
	a := archsimd.LoadUint8x16Array(p)
	return a.ExtendLo8ToUint16().ReshapeToUint8s(), a.HiToLo().ExtendLo8ToUint16().ReshapeToUint8s()
}

// mc8LoadWiden3 is mc8LoadWiden passing o through, so a call written as
// mc8LoadWiden3(mc8R16(s, o+st, lim), o+st) carries an instruction of its own
// line (the add) and leaves no inline-mark NOP.
func mc8LoadWiden3(p *[16]uint8, o int) (archsimd.Uint8x16, archsimd.Uint8x16, int) {
	lo, hi := mc8LoadWiden(p)
	return lo, hi, o
}

func mc8I16(x archsimd.Uint8x16) archsimd.Int16x8 {
	return x.ReshapeToUint16s().BitsToInt16()
}

// mc8LoadRow16 loads the 16 bytes at p widened to int16 lanes 0..7.
func mc8LoadRow16(p *[16]uint8) archsimd.Int16x8 {
	return archsimd.LoadUint8x16Array(p).ExtendLo8ToUint16().BitsToInt16()
}

// mc8Lo64 is the low 8 bytes of v (the eight saturated output pixels).
func mc8Lo64(v archsimd.Uint8x16) uint64 {
	return v.ReshapeToUint64s().GetElem(0)
}

// mc8LoadI32x8 loads the eight int32 intermediate samples at p as two halves.
func mc8LoadI32x8(p *[8]int32) (archsimd.Int32x4, archsimd.Int32x4) {
	return archsimd.LoadInt32x4Array((*[4]int32)(p[0:4])), archsimd.LoadInt32x4Array((*[4]int32)(p[4:8]))
}

// mc8PutH6ToI32Rnd is the put 2D horizontal rounding constant: (s + 2) >> 2
// plus a bias of 8 on every intermediate sample, ((s + 34) >> 2) exactly
// (floor division by 4 commutes with adding a multiple of 4). The six
// vertical taps of every six-tap phase sum to 128, scaled by 32 to 4096, so
// the bias adds 8 * 4096 = 2^15 to every vertical sum: the vertical pass's
// rounding, which mc8V6PutPair therefore no longer adds. The biased samples
// stay below 2^15 (|s| < 20000) and the vertical sums inside int32.
const mc8PutH6ToI32Rnd = 2 + 4*8

// mc8V6PutPair is the put 2D vertical chain of output rows r and r+1 from
// the seven intermediate rows (x, y halves) r..r+6, saturated to bytes. The
// intermediate carries the vertical rounding (mc8PutH6ToI32Rnd).
func mc8V6PutPair(x0, y0, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, c0, c1, c2, c3, c4, c5 archsimd.Int32x4) (uint64, uint64) {
	a0 := x0.Mul(c0)
	b0 := y0.Mul(c0)
	a1 := x1.Mul(c0)
	b1 := y1.Mul(c0)
	a0 = x1.MulAdd(c1, a0)
	b0 = y1.MulAdd(c1, b0)
	a1 = x2.MulAdd(c1, a1)
	b1 = y2.MulAdd(c1, b1)
	a0 = x2.MulAdd(c2, a0)
	b0 = y2.MulAdd(c2, b0)
	a1 = x3.MulAdd(c2, a1)
	b1 = y3.MulAdd(c2, b1)
	a0 = x3.MulAdd(c3, a0)
	b0 = y3.MulAdd(c3, b0)
	a1 = x4.MulAdd(c3, a1)
	b1 = y4.MulAdd(c3, b1)
	a0 = x4.MulAdd(c4, a0)
	b0 = y4.MulAdd(c4, b0)
	a1 = x5.MulAdd(c4, a1)
	b1 = y5.MulAdd(c4, b1)
	a0 = x5.MulAdd(c5, a0)
	b0 = y5.MulAdd(c5, b0)
	a1 = x6.MulAdd(c5, a1)
	b1 = y6.MulAdd(c5, b1)
	v0 := a0.ToBits().ReshapeToUint16s().ConcatOdd(b0.ToBits().ReshapeToUint16s()).BitsToInt16()
	v1 := a1.ToBits().ReshapeToUint16s().ConcatOdd(b1.ToBits().ReshapeToUint16s()).BitsToInt16()
	return mc8Lo64(v0.SaturateToUint8()), mc8Lo64(v1.SaturateToUint8())
}

// mc8PrepV6Rnd is the 8-bit prep 2D vertical rounding constant, pre-scaled
// by the 512x vertical taps: (64 + 6144 << 7) << 9.
const mc8PrepV6Rnd = (64 + 6144<<7) << 9

// mc8V6PrepPair is the prep 2D vertical chain of CONV_BUF rows r and r+1
// from the seven intermediate rows r..r+6.
func mc8V6PrepPair(x0, y0, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, c0, c1, c2, c3, c4, c5, rnd archsimd.Int32x4) (archsimd.Uint16x8, archsimd.Uint16x8) {
	// Multiply first and add rnd last, as in mc8V6PutPair.
	a0 := x0.Mul(c0)
	b0 := y0.Mul(c0)
	a1 := x1.Mul(c0)
	b1 := y1.Mul(c0)
	a0 = x1.MulAdd(c1, a0)
	b0 = y1.MulAdd(c1, b0)
	a1 = x2.MulAdd(c1, a1)
	b1 = y2.MulAdd(c1, b1)
	a0 = x2.MulAdd(c2, a0)
	b0 = y2.MulAdd(c2, b0)
	a1 = x3.MulAdd(c2, a1)
	b1 = y3.MulAdd(c2, b1)
	a0 = x3.MulAdd(c3, a0)
	b0 = y3.MulAdd(c3, b0)
	a1 = x4.MulAdd(c3, a1)
	b1 = y4.MulAdd(c3, b1)
	a0 = x4.MulAdd(c4, a0)
	b0 = y4.MulAdd(c4, b0)
	a1 = x5.MulAdd(c4, a1)
	b1 = y5.MulAdd(c4, b1)
	a0 = x5.MulAdd(c5, a0)
	b0 = y5.MulAdd(c5, b0)
	a1 = x6.MulAdd(c5, a1)
	b1 = y6.MulAdd(c5, b1)
	a0, b0, a1, b1 = a0.Add(rnd), b0.Add(rnd), a1.Add(rnd), b1.Add(rnd)
	return a0.ToBits().ReshapeToUint16s().ConcatOdd(b0.ToBits().ReshapeToUint16s()), a1.ToBits().ReshapeToUint16s().ConcatOdd(b1.ToBits().ReshapeToUint16s())
}

// mc8H6Sum is the six-tap halved horizontal sum (plus rnd) of the eight
// output columns whose window starts at p.
func mc8H6Sum(p *[16]uint8, c0, c1, c2, c3, c4, c5, rnd archsimd.Int16x8) archsimd.Int16x8 {
	lo, hi := mc8LoadWiden(p)
	sum := mc8I16(lo).MulAdd(c0, rnd)
	sum = mc8I16(hi.ConcatShiftBytesRight(lo, 2)).MulAdd(c1, sum)
	sum = mc8I16(hi.ConcatShiftBytesRight(lo, 4)).MulAdd(c2, sum)
	sum = mc8I16(hi.ConcatShiftBytesRight(lo, 6)).MulAdd(c3, sum)
	sum = mc8I16(hi.ConcatShiftBytesRight(lo, 8)).MulAdd(c4, sum)
	return mc8I16(hi.ConcatShiftBytesRight(lo, 10)).MulAdd(c5, sum)
}

// mc8BlendAvgFast is the 8-bit plain compound average (fwd = bck = 8) over
// two CONV_BUF predictors with row stride w: libaom's
// clip(round4(((a*8 + b*8) >> 4) - 6144)) is (a + b - 12272) >> 5 by nested
// floor division, evaluated in int16 lanes (8-bit CONV_BUF values stay below
// 2^14, so a + b cannot wrap) and saturated to bytes.
func mc8BlendAvgFast(dst *frame.Plane, a, b []uint16, dstX, dstY, w, h int) bool {
	if w <= 0 || (w != 4 && w&7 != 0) || h <= 0 || len(a) < w*h || len(b) < w*h || (w == 4 && h&1 != 0) {
		return false
	}
	off := dstY*dst.Stride + dstX
	ext := (h-1)*dst.Stride + w
	if dstX < 0 || dstY < 0 || dstX+w > dst.Width || !unsafeutil.GridFit(len(dst.Pix), off, dst.Stride, w, h) || ext > mc8WindowMax {
		return false
	}
	mc8BlendAvg(unsafeutil.NewGrid(dst.Pix, off, dst.Stride, w, h), a[:w*h], b[:w*h], w, h)
	return true
}

func mc8BlendAvgCopyFast(dst, ref0, ref1 *frame.Plane, dstX, dstY, refX0, refY0, refX1, refY1, w, h int) bool {
	if w != 4 && w&7 != 0 {
		return false
	}
	dOff := dstY*dst.Stride + dstX
	s0Off := refY0*ref0.Stride + refX0
	s1Off := refY1*ref1.Stride + refX1
	if !unsafeutil.GridFit(len(dst.Pix), dOff, dst.Stride, w, h) ||
		!unsafeutil.GridFit(len(ref0.Pix), s0Off, ref0.Stride, w, h) ||
		!unsafeutil.GridFit(len(ref1.Pix), s1Off, ref1.Stride, w, h) {
		return false
	}
	dg := unsafeutil.NewGrid(dst.Pix, dOff, dst.Stride, w, h)
	g0 := unsafeutil.NewGrid(ref0.Pix, s0Off, ref0.Stride, w, h)
	g1 := unsafeutil.NewGrid(ref1.Pix, s1Off, ref1.Stride, w, h)
	if !dg.Fits(0, 0, w, h) || !g0.Fits(0, 0, w, h) || !g1.Fits(0, 0, w, h) {
		return false
	}
	// Exact rounding average without widening: (a + b + 1) >> 1 == (a | b) - ((a ^ b) >> 1).
	switch {
	case w == 4:
		for r := range h {
			a := binary.LittleEndian.Uint32(g0.Row4(r, 0)[:])
			b := binary.LittleEndian.Uint32(g1.Row4(r, 0)[:])
			binary.LittleEndian.PutUint32(dg.Row4(r, 0)[:], (a|b)-(((a^b)&0xfefefefe)>>1))
		}
	case w == 8:
		for r := range h {
			a := binary.LittleEndian.Uint64(g0.Row8(r, 0)[:])
			b := binary.LittleEndian.Uint64(g1.Row8(r, 0)[:])
			binary.LittleEndian.PutUint64(dg.Row8(r, 0)[:], (a|b)-(((a^b)&0xfefefefefefefefe)>>1))
		}
	case w&15 == 0:
		for r := range h {
			for c := 0; c < w; c += 16 {
				a := archsimd.LoadUint8x16Array(g0.Row16(r, c))
				b := archsimd.LoadUint8x16Array(g1.Row16(r, c))
				a.Average(b).StoreArray(dg.Row16(r, c))
			}
		}
	default: // w = 16k + 8
		for r := range h {
			c := 0
			for ; c+16 <= w; c += 16 {
				a := archsimd.LoadUint8x16Array(g0.Row16(r, c))
				b := archsimd.LoadUint8x16Array(g1.Row16(r, c))
				a.Average(b).StoreArray(dg.Row16(r, c))
			}
			a := binary.LittleEndian.Uint64(g0.Row8(r, c)[:])
			b := binary.LittleEndian.Uint64(g1.Row8(r, c)[:])
			binary.LittleEndian.PutUint64(dg.Row8(r, c)[:], (a|b)-(((a^b)&0xfefefefefefefefe)>>1))
		}
	}
	return true
}

func mc8BlendDistWtdCopyFast(dst, ref0, ref1 *frame.Plane, dstX, dstY, refX0, refY0, refX1, refY1, w, h, fwd, bck int) bool {
	if w != 8 && w&15 != 0 {
		return false
	}
	dOff := dstY*dst.Stride + dstX
	s0Off := refY0*ref0.Stride + refX0
	s1Off := refY1*ref1.Stride + refX1
	if !unsafeutil.GridFit(len(dst.Pix), dOff, dst.Stride, w, h) ||
		!unsafeutil.GridFit(len(ref0.Pix), s0Off, ref0.Stride, w, h) ||
		!unsafeutil.GridFit(len(ref1.Pix), s1Off, ref1.Stride, w, h) {
		return false
	}
	dg := unsafeutil.NewGrid(dst.Pix, dOff, dst.Stride, w, h)
	g0 := unsafeutil.NewGrid(ref0.Pix, s0Off, ref0.Stride, w, h)
	g1 := unsafeutil.NewGrid(ref1.Pix, s1Off, ref1.Stride, w, h)
	if !dg.Fits(0, 0, w, h) || !g0.Fits(0, 0, w, h) || !g1.Fits(0, 0, w, h) {
		return false
	}
	fw := archsimd.BroadcastUint16x8(uint16(fwd))
	bw := archsimd.BroadcastUint16x8(uint16(bck))
	rnd := archsimd.BroadcastUint16x8(8)
	shR4 := archsimd.BroadcastInt16x8(-4)
	if w == 8 {
		for r := range h {
			a := archsimd.BroadcastUint64x2(binary.LittleEndian.Uint64(g0.Row8(r, 0)[:])).ReshapeToUint8s()
			b := archsimd.BroadcastUint64x2(binary.LittleEndian.Uint64(g1.Row8(r, 0)[:])).ReshapeToUint8s()
			v := a.ExtendLo8ToUint16().MulAdd(fw, rnd).Add(b.ExtendLo8ToUint16().Mul(bw)).Shift(shR4).TruncToUint8()
			binary.LittleEndian.PutUint64(dg.Row8(r, 0)[:], v.ReshapeToUint64s().GetElem(0))
		}
		return true
	}
	for r := range h {
		for c := 0; c < w; c += 16 {
			a := archsimd.LoadUint8x16Array(g0.Row16(r, c))
			b := archsimd.LoadUint8x16Array(g1.Row16(r, c))
			lo := a.ExtendLo8ToUint16().MulAdd(fw, rnd).Add(b.ExtendLo8ToUint16().Mul(bw)).Shift(shR4).TruncToUint8()
			hi := a.HiToLo().ExtendLo8ToUint16().MulAdd(fw, rnd).Add(b.HiToLo().ExtendLo8ToUint16().Mul(bw)).Shift(shR4).TruncToUint8()
			lo.ReshapeToUint64s().InterleaveLo(hi.ReshapeToUint64s()).ReshapeToUint8s().StoreArray(dg.Row16(r, c))
		}
	}
	return true
}

func mc8BlendAvg(dg unsafeutil.Grid[uint8], a, b []uint16, w, h int) {
	// clamp((a + b - 12272) >> 5, 0, 255) in the shape of dav1d's add +
	// sqrshrun pair: a saturating subtract clamps below at 0, a saturating
	// << 3 clamps floor(u / 32) at 255 in each high byte, and UZP2 gathers
	// those bytes.
	bias := archsimd.BroadcastUint16x8(12272)
	sh := archsimd.BroadcastInt16x8(3)
	avg := func(pa, pb *[8]uint16) archsimd.Uint8x16 {
		return archsimd.LoadUint16x8Array(pa).Add(archsimd.LoadUint16x8Array(pb)).SubSaturated(bias).ShiftSaturated(sh).ReshapeToUint8s()
	}
	n := w * h
	a, b = a[:n:n], b[:n:n]
	if !dg.Fits(0, 0, w, h) {
		return
	}
	switch {
	case w == 4:
		i := 0
		for r := 0; r+2 <= h; r += 2 {
			pa, pb := (*[8]uint16)(a[i:i+8]), (*[8]uint16)(b[i:i+8])
			v := avg(pa, pb)
			x := v.ConcatOdd(v).ReshapeToUint64s().GetElem(0)
			binary.LittleEndian.PutUint32(dg.Row4(r, 0)[:], uint32(x))
			binary.LittleEndian.PutUint32(dg.Row4(r+1, 0)[:], uint32(x>>32))
			i += 8
		}
	case w == 8:
		i := 0
		for r := 0; r < h; r++ {
			pa, pb := (*[8]uint16)(a[i:i+8]), (*[8]uint16)(b[i:i+8])
			v := avg(pa, pb)
			binary.LittleEndian.PutUint64(dg.Row8(r, 0)[:], v.ConcatOdd(v).ReshapeToUint64s().GetElem(0))
			i += 8
		}
	case w&15 == 0:
		i := 0
		for r := 0; r < h; r++ {
			for c := 0; c < w; c += 16 {
				pa, pb := (*[16]uint16)(a[i:i+16]), (*[16]uint16)(b[i:i+16])
				v0 := avg((*[8]uint16)(pa[0:8]), (*[8]uint16)(pb[0:8]))
				v1 := avg((*[8]uint16)(pa[8:16]), (*[8]uint16)(pb[8:16]))
				v0.ConcatOdd(v1).StoreArray(dg.Row16(r, c))
				i += 16
			}
		}
	default: // w = 16k + 8
		i := 0
		for r := 0; r < h; r++ {
			c := 0
			for ; c+16 <= w; c += 16 {
				pa, pb := (*[16]uint16)(a[i:i+16]), (*[16]uint16)(b[i:i+16])
				v0 := avg((*[8]uint16)(pa[0:8]), (*[8]uint16)(pb[0:8]))
				v1 := avg((*[8]uint16)(pa[8:16]), (*[8]uint16)(pb[8:16]))
				v0.ConcatOdd(v1).StoreArray(dg.Row16(r, c))
				i += 16
			}
			pa, pb := (*[8]uint16)(a[i:i+8]), (*[8]uint16)(b[i:i+8])
			v := avg(pa, pb)
			binary.LittleEndian.PutUint64(dg.Row8(r, c)[:], v.ConcatOdd(v).ReshapeToUint64s().GetElem(0))
			i += 8
		}
	}
}

// mc8L16U loads eight uint16 lanes from p (len(p) >= 8).
func mc8L16U(p []uint16) archsimd.Uint16x8 {
	return archsimd.LoadUint16x8Array((*[8]uint16)(p))
}

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
// unsafeutil.CopyRect leaves to its caller (no
// AV1 block has one).
func mc8CopyStrips(dg, sg unsafeutil.Grid[uint8]) bool {
	h, n := dg.Height(), dg.Width()
	if !sg.Fits(0, 0, n, h) {
		return false
	}
	for c := 0; c+4 <= n; c += 4 {
		d, s := dg.Sub(0, c, 4, h), sg.Sub(0, c, 4, h)
		for r := 0; r < h; r++ {
			*d.Row4(r, 0) = *s.Row4(r, 0)
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

// mc8PrepCopy is the prep copy, 16 * px + 6144 (UMULL by 16, then the
// offset), from the source view sg (rows of max(w+8, 16) readable bytes) into
// the CONV_BUF view og (w = og.Width() a multiple of 8, or 4).
func mc8PrepCopy(sg unsafeutil.Grid[uint8], og unsafeutil.Grid[uint16]) {
	mul := archsimd.LoadUint8x16Array(&mc8Vec.mul16)
	mulHi := mul.HiToLo()
	off := archsimd.LoadUint16x8Array(&mc8Vec.off6144)
	h, w := og.Height(), og.Width()
	if w == 4 {
		if !sg.Fits(0, 0, 16, h) {
			return
		}
		s := sg.Sub(0, 0, 16, h)
		for r := 0; r < h; r++ {
			v := archsimd.LoadUint8x16Array(s.Row16(r, 0)).MulWidenLo(mul).Add(off)
			mc8Store4U16(og.Row4(r, 0), v.ReshapeToUint64s().GetElem(0))
		}
		return
	}
	c := 0
	for ; c+16 <= w; c += 16 {
		if !sg.Fits(0, c, 16, h) || !og.Fits(0, c, 16, h) {
			return
		}
		s, o := sg.Sub(0, c, 16, h), og.Sub(0, c, 16, h)
		for r := 0; r < h; r++ {
			a := archsimd.LoadUint8x16Array(s.Row16(r, 0))
			p := o.Row16(r, 0)
			a.MulWidenLo(mul).Add(off).StoreArray((*[8]uint16)(p[0:8]))
			a.HiToLo().MulWidenLo(mulHi).Add(off).StoreArray((*[8]uint16)(p[8:16]))
		}
	}
	if c < w {
		if !sg.Fits(0, c, 16, h) || !og.Fits(0, c, 8, h) {
			return
		}
		s, o := sg.Sub(0, c, 16, h), og.Sub(0, c, 8, h)
		for r := 0; r < h; r++ {
			archsimd.LoadUint8x16Array(s.Row16(r, 0)).MulWidenLo(mul).Add(off).StoreArray(o.Row8(r, 0))
		}
	}
}

// mc8PrepCopyW8 is mc8PrepCopy for an eight-wide block, whose CONV_BUF rows
// are contiguous: eight rows per chunk are one 64-sample window of out
// (constant store offsets) and one fixed-shape sub-view of sg.
func mc8PrepCopyW8(sg unsafeutil.Grid[uint8], out []uint16, h int) {
	mul := archsimd.LoadUint8x16Array(&mc8Vec.mul16)
	off := archsimd.LoadUint16x8Array(&mc8Vec.off6144)
	r := 0
	for ; r+8 <= h; r += 8 {
		if !sg.Fits(r, 0, 16, 8) || 8*r+64 > len(out) {
			return
		}
		s, o := sg.Sub(r, 0, 16, 8), (*[64]uint16)(out[8*r:8*r+64])
		archsimd.LoadUint8x16Array(s.Row16(0, 0)).MulWidenLo(mul).Add(off).StoreArray((*[8]uint16)(o[0:8]))
		archsimd.LoadUint8x16Array(s.Row16(1, 0)).MulWidenLo(mul).Add(off).StoreArray((*[8]uint16)(o[8:16]))
		archsimd.LoadUint8x16Array(s.Row16(2, 0)).MulWidenLo(mul).Add(off).StoreArray((*[8]uint16)(o[16:24]))
		archsimd.LoadUint8x16Array(s.Row16(3, 0)).MulWidenLo(mul).Add(off).StoreArray((*[8]uint16)(o[24:32]))
		archsimd.LoadUint8x16Array(s.Row16(4, 0)).MulWidenLo(mul).Add(off).StoreArray((*[8]uint16)(o[32:40]))
		archsimd.LoadUint8x16Array(s.Row16(5, 0)).MulWidenLo(mul).Add(off).StoreArray((*[8]uint16)(o[40:48]))
		archsimd.LoadUint8x16Array(s.Row16(6, 0)).MulWidenLo(mul).Add(off).StoreArray((*[8]uint16)(o[48:56]))
		archsimd.LoadUint8x16Array(s.Row16(7, 0)).MulWidenLo(mul).Add(off).StoreArray((*[8]uint16)(o[56:64]))
	}
	for ; r < h; r++ {
		if !sg.Fits(r, 0, 16, 1) || 8*r+8 > len(out) {
			return
		}
		archsimd.LoadUint8x16Array(sg.Sub(r, 0, 16, 1).Row16(0, 0)).MulWidenLo(mul).Add(off).StoreArray((*[8]uint16)(out[8*r : 8*r+8]))
	}
}

// The 8-bit six-tap 2D (hv) kernels over Grid views: the horizontal pass
// writes the strip-major int32 intermediate (strips of eight columns, each
// strip's rows contiguous) through one Grid per strip, and the vertical pass
// writes eight output rows per chunk of the destination view. As in the
// single-pass kernels, every Fits test is unreachable (the entries sized the
// views and the intermediate), so the kernels stay leaf functions.

// mc8H4S is the four-tap halved horizontal sum (taps 1..4 of the six-tap
// window, where taps 0 and 5 are zero) plus rnd.
func mc8H4S(p *[16]uint8, c1, c2, c3, c4, rnd archsimd.Int16x8) archsimd.Int16x8 {
	a := archsimd.LoadUint8x16Array(p)
	lo, hi := a.ExtendLo8ToUint16().ReshapeToUint8s(), a.HiToLo().ExtendLo8ToUint16().ReshapeToUint8s()
	s := hi.ConcatShiftBytesRight(lo, 2).ReshapeToUint16s().BitsToInt16().MulAdd(c1, rnd)
	s = hi.ConcatShiftBytesRight(lo, 4).ReshapeToUint16s().BitsToInt16().MulAdd(c2, s)
	s = hi.ConcatShiftBytesRight(lo, 6).ReshapeToUint16s().BitsToInt16().MulAdd(c3, s)
	return hi.ConcatShiftBytesRight(lo, 8).ReshapeToUint16s().BitsToInt16().MulAdd(c4, s)
}

// mc8PutH4S4 is mc8PutH6S4 when the horizontal filter has at most 4 non-zero
// taps (taps 1..4: regular4, smooth4, bilinear — the 99.9% case for w == 4).
func mc8PutH4S4(s unsafeutil.Grid[uint8], ph *mc8Phase, o unsafeutil.Grid[uint8]) {
	c1 := archsimd.LoadInt16x8Array(&ph.h[1])
	c2 := archsimd.LoadInt16x8Array(&ph.h[2])
	c3 := archsimd.LoadInt16x8Array(&ph.h[3])
	c4 := archsimd.LoadInt16x8Array(&ph.h[4])
	rnd := archsimd.LoadInt16x8Array(&mc8Vec.rnd34)
	sh := archsimd.LoadInt16x8Array(&mc8Vec.sh6)
	h := o.Height()
	r := 0
	if h >= 8 {
		for ; r+8 <= h; r += 8 {
			if !s.Fits(r, 0, 16, 8) || !o.Fits(r, 0, 4, 8) {
				return
			}
			sw, ow := s.Sub(r, 0, 16, 8), o.Sub(r, 0, 4, 8)
			binary.LittleEndian.PutUint32(ow.Row4(0, 0)[:], uint32(mc8H4S(sw.Row16(0, 0), c1, c2, c3, c4, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0)))
			binary.LittleEndian.PutUint32(ow.Row4(1, 0)[:], uint32(mc8H4S(sw.Row16(1, 0), c1, c2, c3, c4, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0)))
			binary.LittleEndian.PutUint32(ow.Row4(2, 0)[:], uint32(mc8H4S(sw.Row16(2, 0), c1, c2, c3, c4, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0)))
			binary.LittleEndian.PutUint32(ow.Row4(3, 0)[:], uint32(mc8H4S(sw.Row16(3, 0), c1, c2, c3, c4, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0)))
			binary.LittleEndian.PutUint32(ow.Row4(4, 0)[:], uint32(mc8H4S(sw.Row16(4, 0), c1, c2, c3, c4, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0)))
			binary.LittleEndian.PutUint32(ow.Row4(5, 0)[:], uint32(mc8H4S(sw.Row16(5, 0), c1, c2, c3, c4, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0)))
			binary.LittleEndian.PutUint32(ow.Row4(6, 0)[:], uint32(mc8H4S(sw.Row16(6, 0), c1, c2, c3, c4, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0)))
			binary.LittleEndian.PutUint32(ow.Row4(7, 0)[:], uint32(mc8H4S(sw.Row16(7, 0), c1, c2, c3, c4, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0)))
		}
	}
	if r < h {
		if r+4 <= h {
			if !s.Fits(r, 0, 16, 4) || !o.Fits(r, 0, 4, 4) {
				return
			}
			sw, ow := s.Sub(r, 0, 16, 4), o.Sub(r, 0, 4, 4)
			binary.LittleEndian.PutUint32(ow.Row4(0, 0)[:], uint32(mc8H4S(sw.Row16(0, 0), c1, c2, c3, c4, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0)))
			binary.LittleEndian.PutUint32(ow.Row4(1, 0)[:], uint32(mc8H4S(sw.Row16(1, 0), c1, c2, c3, c4, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0)))
			binary.LittleEndian.PutUint32(ow.Row4(2, 0)[:], uint32(mc8H4S(sw.Row16(2, 0), c1, c2, c3, c4, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0)))
			binary.LittleEndian.PutUint32(ow.Row4(3, 0)[:], uint32(mc8H4S(sw.Row16(3, 0), c1, c2, c3, c4, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0)))
			r += 4
		}
		if r+2 <= h {
			if !s.Fits(r, 0, 16, 2) || !o.Fits(r, 0, 4, 2) {
				return
			}
			sw, ow := s.Sub(r, 0, 16, 2), o.Sub(r, 0, 4, 2)
			binary.LittleEndian.PutUint32(ow.Row4(0, 0)[:], uint32(mc8H4S(sw.Row16(0, 0), c1, c2, c3, c4, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0)))
			binary.LittleEndian.PutUint32(ow.Row4(1, 0)[:], uint32(mc8H4S(sw.Row16(1, 0), c1, c2, c3, c4, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0)))
			r += 2
		}
		if r < h {
			if !s.Fits(r, 0, 16, 1) || !o.Fits(r, 0, 4, 1) {
				return
			}
			sw, ow := s.Sub(r, 0, 16, 1), o.Sub(r, 0, 4, 1)
			binary.LittleEndian.PutUint32(ow.Row4(0, 0)[:], uint32(mc8H4S(sw.Row16(0, 0), c1, c2, c3, c4, rnd).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0)))
		}
	}
}

// mc8PutV4 is the 1D vertical 4-tap put pass (taps 1..4: regular4, smooth4,
// bilinear) for w == 4 or w multiple of 8: skips rows 0 and h+4 and drops the
// zero outer taps.
func mc8PutV4(sg unsafeutil.Grid[uint8], w, h int, ph *mc8Phase, dg unsafeutil.Grid[uint8]) {
	c1 := archsimd.LoadInt16x8Array(&ph.h[1])
	c2 := archsimd.LoadInt16x8Array(&ph.h[2])
	c3 := archsimd.LoadInt16x8Array(&ph.h[3])
	c4 := archsimd.LoadInt16x8Array(&ph.h[4])
	rnd := archsimd.LoadInt16x8Array(&mc8Vec.rnd32)
	sh := archsimd.LoadInt16x8Array(&mc8Vec.sh6)
	rows := h + 4
	if w == 4 {
		if !sg.Fits(0, 0, 16, rows) || !dg.Fits(0, 0, 4, h) {
			return
		}
		s, d := sg.Sub(0, 0, 16, rows), dg.Sub(0, 0, 4, h)
		if !s.Fits(1, 0, 16, 3) {
			return
		}
		hd := s.Sub(1, 0, 16, 3)
		r1 := archsimd.LoadUint8x16Array(hd.Row16(0, 0)).ExtendLo8ToUint16().BitsToInt16()
		r2 := archsimd.LoadUint8x16Array(hd.Row16(1, 0)).ExtendLo8ToUint16().BitsToInt16()
		r3 := archsimd.LoadUint8x16Array(hd.Row16(2, 0)).ExtendLo8ToUint16().BitsToInt16()
		r := 0
		for ; r+4 <= h; r += 4 {
			if !s.Fits(r+4, 0, 16, 4) || !d.Fits(r, 0, 4, 4) {
				return
			}
			sw, dw := s.Sub(r+4, 0, 16, 4), d.Sub(r, 0, 4, 4)
			r4 := archsimd.LoadUint8x16Array(sw.Row16(0, 0)).ExtendLo8ToUint16().BitsToInt16()
			r5 := archsimd.LoadUint8x16Array(sw.Row16(1, 0)).ExtendLo8ToUint16().BitsToInt16()
			r6 := archsimd.LoadUint8x16Array(sw.Row16(2, 0)).ExtendLo8ToUint16().BitsToInt16()
			r7 := archsimd.LoadUint8x16Array(sw.Row16(3, 0)).ExtendLo8ToUint16().BitsToInt16()
			binary.LittleEndian.PutUint32(dw.Row4(0, 0)[:], uint32(r4.MulAdd(c4, r3.MulAdd(c3, r2.MulAdd(c2, r1.MulAdd(c1, rnd)))).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0)))
			binary.LittleEndian.PutUint32(dw.Row4(1, 0)[:], uint32(r5.MulAdd(c4, r4.MulAdd(c3, r3.MulAdd(c2, r2.MulAdd(c1, rnd)))).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0)))
			binary.LittleEndian.PutUint32(dw.Row4(2, 0)[:], uint32(r6.MulAdd(c4, r5.MulAdd(c3, r4.MulAdd(c2, r3.MulAdd(c1, rnd)))).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0)))
			binary.LittleEndian.PutUint32(dw.Row4(3, 0)[:], uint32(r7.MulAdd(c4, r6.MulAdd(c3, r5.MulAdd(c2, r4.MulAdd(c1, rnd)))).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0)))
			r1, r2, r3 = r5, r6, r7
		}
		for ; r < h; r++ {
			r4 := archsimd.LoadUint8x16Array(s.Row16(r+4, 0)).ExtendLo8ToUint16().BitsToInt16()
			v := r4.MulAdd(c4, r3.MulAdd(c3, r2.MulAdd(c2, r1.MulAdd(c1, rnd)))).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0)
			binary.LittleEndian.PutUint32(d.Row4(r, 0)[:], uint32(v))
			r1, r2, r3 = r2, r3, r4
		}
		return
	}
	for c := 0; c < w; c += 8 {
		if !sg.Fits(0, c, 16, rows) || !dg.Fits(0, c, 8, h) {
			return
		}
		s, d := sg.Sub(0, c, 16, rows), dg.Sub(0, c, 8, h)
		if !s.Fits(1, 0, 16, 3) {
			return
		}
		hd := s.Sub(1, 0, 16, 3)
		r1 := archsimd.LoadUint8x16Array(hd.Row16(0, 0)).ExtendLo8ToUint16().BitsToInt16()
		r2 := archsimd.LoadUint8x16Array(hd.Row16(1, 0)).ExtendLo8ToUint16().BitsToInt16()
		r3 := archsimd.LoadUint8x16Array(hd.Row16(2, 0)).ExtendLo8ToUint16().BitsToInt16()
		r := 0
		for ; r+4 <= h; r += 4 {
			if !s.Fits(r+4, 0, 16, 4) || !d.Fits(r, 0, 8, 4) {
				return
			}
			sw, dw := s.Sub(r+4, 0, 16, 4), d.Sub(r, 0, 8, 4)
			r4 := archsimd.LoadUint8x16Array(sw.Row16(0, 0)).ExtendLo8ToUint16().BitsToInt16()
			r5 := archsimd.LoadUint8x16Array(sw.Row16(1, 0)).ExtendLo8ToUint16().BitsToInt16()
			r6 := archsimd.LoadUint8x16Array(sw.Row16(2, 0)).ExtendLo8ToUint16().BitsToInt16()
			r7 := archsimd.LoadUint8x16Array(sw.Row16(3, 0)).ExtendLo8ToUint16().BitsToInt16()
			binary.LittleEndian.PutUint64(dw.Row8(0, 0)[:], r4.MulAdd(c4, r3.MulAdd(c3, r2.MulAdd(c2, r1.MulAdd(c1, rnd)))).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0))
			binary.LittleEndian.PutUint64(dw.Row8(1, 0)[:], r5.MulAdd(c4, r4.MulAdd(c3, r3.MulAdd(c2, r2.MulAdd(c1, rnd)))).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0))
			binary.LittleEndian.PutUint64(dw.Row8(2, 0)[:], r6.MulAdd(c4, r5.MulAdd(c3, r4.MulAdd(c2, r3.MulAdd(c1, rnd)))).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0))
			binary.LittleEndian.PutUint64(dw.Row8(3, 0)[:], r7.MulAdd(c4, r6.MulAdd(c3, r5.MulAdd(c2, r4.MulAdd(c1, rnd)))).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0))
			r1, r2, r3 = r5, r6, r7
		}
		for ; r < h; r++ {
			r4 := archsimd.LoadUint8x16Array(s.Row16(r+4, 0)).ExtendLo8ToUint16().BitsToInt16()
			v := r4.MulAdd(c4, r3.MulAdd(c3, r2.MulAdd(c2, r1.MulAdd(c1, rnd)))).Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0)
			binary.LittleEndian.PutUint64(d.Row8(r, 0)[:], v)
			r1, r2, r3 = r2, r3, r4
		}
	}
}

// mc8PrepH4S4 is mc8PrepH6S4 when the horizontal filter has at most 4 non-zero
// taps (taps 1..4: regular4, smooth4, bilinear — the 99.9% case for w == 4).
func mc8PrepH4S4(s unsafeutil.Grid[uint8], ph *mc8Phase, o unsafeutil.Grid[uint16]) {
	c1 := archsimd.LoadInt16x8Array(&ph.h[1])
	c2 := archsimd.LoadInt16x8Array(&ph.h[2])
	c3 := archsimd.LoadInt16x8Array(&ph.h[3])
	c4 := archsimd.LoadInt16x8Array(&ph.h[4])
	rnd := archsimd.LoadInt16x8Array(&mc8Vec.prepRnd)
	sh := archsimd.LoadInt16x8Array(&mc8Vec.sh2)
	h := o.Height()
	r := 0
	for ; r+4 <= h; r += 4 {
		if !s.Fits(r, 0, 16, 4) || !o.Fits(r, 0, 4, 4) {
			return
		}
		sw, ow := s.Sub(r, 0, 16, 4), o.Sub(r, 0, 4, 4)
		mc8Store4U16(ow.Row4(0, 0), mc8H4S(sw.Row16(0, 0), c1, c2, c3, c4, rnd).ToBits().Shift(sh).ReshapeToUint64s().GetElem(0))
		mc8Store4U16(ow.Row4(1, 0), mc8H4S(sw.Row16(1, 0), c1, c2, c3, c4, rnd).ToBits().Shift(sh).ReshapeToUint64s().GetElem(0))
		mc8Store4U16(ow.Row4(2, 0), mc8H4S(sw.Row16(2, 0), c1, c2, c3, c4, rnd).ToBits().Shift(sh).ReshapeToUint64s().GetElem(0))
		mc8Store4U16(ow.Row4(3, 0), mc8H4S(sw.Row16(3, 0), c1, c2, c3, c4, rnd).ToBits().Shift(sh).ReshapeToUint64s().GetElem(0))
	}
	if r < h {
		if !s.Fits(r, 0, 16, h-r) || !o.Fits(r, 0, 4, h-r) {
			return
		}
		sw, ow := s.Sub(r, 0, 16, h-r), o.Sub(r, 0, 4, h-r)
		for k := 0; k < h-r; k++ {
			mc8Store4U16(ow.Row4(k, 0), mc8H4S(sw.Row16(k, 0), c1, c2, c3, c4, rnd).ToBits().Shift(sh).ReshapeToUint64s().GetElem(0))
		}
	}
}

// mc8PrepV4 is the 1D vertical 4-tap prep pass (taps 1..4) for w == 4 or w
// multiple of 8: skips rows 0 and h+4 and drops the zero outer taps.
func mc8PrepV4(sg unsafeutil.Grid[uint8], w, h int, ph *mc8Phase, og unsafeutil.Grid[uint16]) {
	c1 := archsimd.LoadInt16x8Array(&ph.h[1])
	c2 := archsimd.LoadInt16x8Array(&ph.h[2])
	c3 := archsimd.LoadInt16x8Array(&ph.h[3])
	c4 := archsimd.LoadInt16x8Array(&ph.h[4])
	rnd := archsimd.LoadInt16x8Array(&mc8Vec.prepRnd)
	sh := archsimd.LoadInt16x8Array(&mc8Vec.sh2)
	rows := h + 4
	if w == 4 {
		if !sg.Fits(0, 0, 16, rows) || !og.Fits(0, 0, 4, h) {
			return
		}
		s, o := sg.Sub(0, 0, 16, rows), og.Sub(0, 0, 4, h)
		if !s.Fits(1, 0, 16, 3) {
			return
		}
		hd := s.Sub(1, 0, 16, 3)
		r1 := archsimd.LoadUint8x16Array(hd.Row16(0, 0)).ExtendLo8ToUint16().BitsToInt16()
		r2 := archsimd.LoadUint8x16Array(hd.Row16(1, 0)).ExtendLo8ToUint16().BitsToInt16()
		r3 := archsimd.LoadUint8x16Array(hd.Row16(2, 0)).ExtendLo8ToUint16().BitsToInt16()
		r := 0
		for ; r+4 <= h; r += 4 {
			if !s.Fits(r+4, 0, 16, 4) || !o.Fits(r, 0, 4, 4) {
				return
			}
			sw, ow := s.Sub(r+4, 0, 16, 4), o.Sub(r, 0, 4, 4)
			r4 := archsimd.LoadUint8x16Array(sw.Row16(0, 0)).ExtendLo8ToUint16().BitsToInt16()
			r5 := archsimd.LoadUint8x16Array(sw.Row16(1, 0)).ExtendLo8ToUint16().BitsToInt16()
			r6 := archsimd.LoadUint8x16Array(sw.Row16(2, 0)).ExtendLo8ToUint16().BitsToInt16()
			r7 := archsimd.LoadUint8x16Array(sw.Row16(3, 0)).ExtendLo8ToUint16().BitsToInt16()
			mc8Store4U16(ow.Row4(0, 0), r4.MulAdd(c4, r3.MulAdd(c3, r2.MulAdd(c2, r1.MulAdd(c1, rnd)))).ToBits().Shift(sh).ReshapeToUint64s().GetElem(0))
			mc8Store4U16(ow.Row4(1, 0), r5.MulAdd(c4, r4.MulAdd(c3, r3.MulAdd(c2, r2.MulAdd(c1, rnd)))).ToBits().Shift(sh).ReshapeToUint64s().GetElem(0))
			mc8Store4U16(ow.Row4(2, 0), r6.MulAdd(c4, r5.MulAdd(c3, r4.MulAdd(c2, r3.MulAdd(c1, rnd)))).ToBits().Shift(sh).ReshapeToUint64s().GetElem(0))
			mc8Store4U16(ow.Row4(3, 0), r7.MulAdd(c4, r6.MulAdd(c3, r5.MulAdd(c2, r4.MulAdd(c1, rnd)))).ToBits().Shift(sh).ReshapeToUint64s().GetElem(0))
			r1, r2, r3 = r5, r6, r7
		}
		for ; r < h; r++ {
			r4 := archsimd.LoadUint8x16Array(s.Row16(r+4, 0)).ExtendLo8ToUint16().BitsToInt16()
			mc8Store4U16(o.Row4(r, 0), r4.MulAdd(c4, r3.MulAdd(c3, r2.MulAdd(c2, r1.MulAdd(c1, rnd)))).ToBits().Shift(sh).ReshapeToUint64s().GetElem(0))
			r1, r2, r3 = r2, r3, r4
		}
		return
	}
	for c := 0; c < w; c += 8 {
		if !sg.Fits(0, c, 16, rows) || !og.Fits(0, c, 8, h) {
			return
		}
		s, o := sg.Sub(0, c, 16, rows), og.Sub(0, c, 8, h)
		if !s.Fits(1, 0, 16, 3) {
			return
		}
		hd := s.Sub(1, 0, 16, 3)
		r1 := archsimd.LoadUint8x16Array(hd.Row16(0, 0)).ExtendLo8ToUint16().BitsToInt16()
		r2 := archsimd.LoadUint8x16Array(hd.Row16(1, 0)).ExtendLo8ToUint16().BitsToInt16()
		r3 := archsimd.LoadUint8x16Array(hd.Row16(2, 0)).ExtendLo8ToUint16().BitsToInt16()
		r := 0
		for ; r+4 <= h; r += 4 {
			if !s.Fits(r+4, 0, 16, 4) || !o.Fits(r, 0, 8, 4) {
				return
			}
			sw, ow := s.Sub(r+4, 0, 16, 4), o.Sub(r, 0, 8, 4)
			r4 := archsimd.LoadUint8x16Array(sw.Row16(0, 0)).ExtendLo8ToUint16().BitsToInt16()
			r5 := archsimd.LoadUint8x16Array(sw.Row16(1, 0)).ExtendLo8ToUint16().BitsToInt16()
			r6 := archsimd.LoadUint8x16Array(sw.Row16(2, 0)).ExtendLo8ToUint16().BitsToInt16()
			r7 := archsimd.LoadUint8x16Array(sw.Row16(3, 0)).ExtendLo8ToUint16().BitsToInt16()
			r4.MulAdd(c4, r3.MulAdd(c3, r2.MulAdd(c2, r1.MulAdd(c1, rnd)))).ToBits().Shift(sh).StoreArray(ow.Row8(0, 0))
			r5.MulAdd(c4, r4.MulAdd(c3, r3.MulAdd(c2, r2.MulAdd(c1, rnd)))).ToBits().Shift(sh).StoreArray(ow.Row8(1, 0))
			r6.MulAdd(c4, r5.MulAdd(c3, r4.MulAdd(c2, r3.MulAdd(c1, rnd)))).ToBits().Shift(sh).StoreArray(ow.Row8(2, 0))
			r7.MulAdd(c4, r6.MulAdd(c3, r5.MulAdd(c2, r4.MulAdd(c1, rnd)))).ToBits().Shift(sh).StoreArray(ow.Row8(3, 0))
			r1, r2, r3 = r5, r6, r7
		}
		for ; r < h; r++ {
			r4 := archsimd.LoadUint8x16Array(s.Row16(r+4, 0)).ExtendLo8ToUint16().BitsToInt16()
			r4.MulAdd(c4, r3.MulAdd(c3, r2.MulAdd(c2, r1.MulAdd(c1, rnd)))).ToBits().Shift(sh).StoreArray(o.Row8(r, 0))
			r1, r2, r3 = r2, r3, r4
		}
	}
}

// mc8PrepHV4x4W4 is the fused width-4 compound 2D pass when both horizontal
// and vertical filters have at most 4 non-zero taps (taps 1..4).
func mc8PrepHV4x4W4(sg unsafeutil.Grid[uint8], h int, phx, phy *mc8Phase, out []uint16) {
	hc1 := archsimd.LoadInt16x8Array(&phx.h[1])
	hc2 := archsimd.LoadInt16x8Array(&phx.h[2])
	hc3 := archsimd.LoadInt16x8Array(&phx.h[3])
	hc4 := archsimd.LoadInt16x8Array(&phx.h[4])
	rnd := archsimd.BroadcastInt16x8(2)
	sh := archsimd.LoadInt16x8Array(&mc8Vec.sh2)
	vc1 := archsimd.LoadInt32x4Array(&phy.p[1])
	vc2 := archsimd.LoadInt32x4Array(&phy.p[2])
	vc3 := archsimd.LoadInt32x4Array(&phy.p[3])
	vc4 := archsimd.LoadInt32x4Array(&phy.p[4])
	vrnd := archsimd.BroadcastInt32x4(mc8PrepV6Rnd)
	rows := h + 4
	if !sg.Fits(0, 0, 16, rows) || len(out) < 4*h {
		return
	}
	s := sg.Sub(0, 0, 16, rows)
	ot := out[: 4*h : 4*h]
	if !s.Fits(1, 0, 16, 3) {
		return
	}
	hd := s.Sub(1, 0, 16, 3)
	x1 := mc8H4S(hd.Row16(0, 0), hc1, hc2, hc3, hc4, rnd).Shift(sh).ExtendLo4ToInt32()
	x2 := mc8H4S(hd.Row16(1, 0), hc1, hc2, hc3, hc4, rnd).Shift(sh).ExtendLo4ToInt32()
	x3 := mc8H4S(hd.Row16(2, 0), hc1, hc2, hc3, hc4, rnd).Shift(sh).ExtendLo4ToInt32()
	r := 0
	for ; r+4 <= h; r += 4 {
		if !s.Fits(r+4, 0, 16, 4) {
			return
		}
		sw := s.Sub(r+4, 0, 16, 4)
		x4 := mc8H4S(sw.Row16(0, 0), hc1, hc2, hc3, hc4, rnd).Shift(sh).ExtendLo4ToInt32()
		x5 := mc8H4S(sw.Row16(1, 0), hc1, hc2, hc3, hc4, rnd).Shift(sh).ExtendLo4ToInt32()
		a0 := x4.MulAdd(vc4, x3.MulAdd(vc3, x2.MulAdd(vc2, x1.MulAdd(vc1, vrnd))))
		a1 := x5.MulAdd(vc4, x4.MulAdd(vc3, x3.MulAdd(vc2, x2.MulAdd(vc1, vrnd))))
		a0.ToBits().ReshapeToUint16s().ConcatOdd(a1.ToBits().ReshapeToUint16s()).StoreArray((*[8]uint16)(ot[4*r : 4*r+8]))
		x6 := mc8H4S(sw.Row16(2, 0), hc1, hc2, hc3, hc4, rnd).Shift(sh).ExtendLo4ToInt32()
		x7 := mc8H4S(sw.Row16(3, 0), hc1, hc2, hc3, hc4, rnd).Shift(sh).ExtendLo4ToInt32()
		a2 := x6.MulAdd(vc4, x5.MulAdd(vc3, x4.MulAdd(vc2, x3.MulAdd(vc1, vrnd))))
		a3 := x7.MulAdd(vc4, x6.MulAdd(vc3, x5.MulAdd(vc2, x4.MulAdd(vc1, vrnd))))
		a2.ToBits().ReshapeToUint16s().ConcatOdd(a3.ToBits().ReshapeToUint16s()).StoreArray((*[8]uint16)(ot[4*r+8 : 4*r+16]))
		x1, x2, x3 = x5, x6, x7
	}
	for ; r+2 <= h; r += 2 {
		x4 := mc8H4S(s.Row16(r+4, 0), hc1, hc2, hc3, hc4, rnd).Shift(sh).ExtendLo4ToInt32()
		x5 := mc8H4S(s.Row16(r+5, 0), hc1, hc2, hc3, hc4, rnd).Shift(sh).ExtendLo4ToInt32()
		a0 := x4.MulAdd(vc4, x3.MulAdd(vc3, x2.MulAdd(vc2, x1.MulAdd(vc1, vrnd))))
		a1 := x5.MulAdd(vc4, x4.MulAdd(vc3, x3.MulAdd(vc2, x2.MulAdd(vc1, vrnd))))
		a0.ToBits().ReshapeToUint16s().ConcatOdd(a1.ToBits().ReshapeToUint16s()).StoreArray((*[8]uint16)(ot[4*r : 4*r+8]))
		x1, x2, x3 = x3, x4, x5
	}
	if r < h {
		x4 := mc8H4S(s.Row16(r+4, 0), hc1, hc2, hc3, hc4, rnd).Shift(sh).ExtendLo4ToInt32()
		a0 := x4.MulAdd(vc4, x3.MulAdd(vc3, x2.MulAdd(vc2, x1.MulAdd(vc1, vrnd))))
		v := a0.ToBits().ReshapeToUint16s().ConcatOdd(a0.ToBits().ReshapeToUint16s()).ReshapeToUint64s().GetElem(0)
		mc8Store4U16((*[4]uint16)(ot[4*r:4*r+4]), v)
	}
}

// mc8PrepHV4W4 is the fused width-4 compound 2D pass when the horizontal filter
// has at most 4 non-zero taps (taps 1..4) and the vertical filter has 6 taps.
func mc8PrepHV4W4(sg unsafeutil.Grid[uint8], h int, phx, phy *mc8Phase, out []uint16) {
	hc1 := archsimd.LoadInt16x8Array(&phx.h[1])
	hc2 := archsimd.LoadInt16x8Array(&phx.h[2])
	hc3 := archsimd.LoadInt16x8Array(&phx.h[3])
	hc4 := archsimd.LoadInt16x8Array(&phx.h[4])
	rnd := archsimd.BroadcastInt16x8(2)
	sh := archsimd.LoadInt16x8Array(&mc8Vec.sh2)
	vc0 := archsimd.LoadInt32x4Array(&phy.p[0])
	vc1 := archsimd.LoadInt32x4Array(&phy.p[1])
	vc2 := archsimd.LoadInt32x4Array(&phy.p[2])
	vc3 := archsimd.LoadInt32x4Array(&phy.p[3])
	vc4 := archsimd.LoadInt32x4Array(&phy.p[4])
	vc5 := archsimd.LoadInt32x4Array(&phy.p[5])
	vrnd := archsimd.BroadcastInt32x4(mc8PrepV6Rnd)
	rows := h + 5
	if !sg.Fits(0, 0, 16, rows) || len(out) < 4*h {
		return
	}
	s := sg.Sub(0, 0, 16, rows)
	ot := out[: 4*h : 4*h]
	if !s.Fits(0, 0, 16, 5) {
		return
	}
	hd := s.Sub(0, 0, 16, 5)
	x0 := mc8H4S(hd.Row16(0, 0), hc1, hc2, hc3, hc4, rnd).Shift(sh).ExtendLo4ToInt32()
	x1 := mc8H4S(hd.Row16(1, 0), hc1, hc2, hc3, hc4, rnd).Shift(sh).ExtendLo4ToInt32()
	x2 := mc8H4S(hd.Row16(2, 0), hc1, hc2, hc3, hc4, rnd).Shift(sh).ExtendLo4ToInt32()
	x3 := mc8H4S(hd.Row16(3, 0), hc1, hc2, hc3, hc4, rnd).Shift(sh).ExtendLo4ToInt32()
	x4 := mc8H4S(hd.Row16(4, 0), hc1, hc2, hc3, hc4, rnd).Shift(sh).ExtendLo4ToInt32()
	r := 0
	for ; r+4 <= h; r += 4 {
		if !s.Fits(r+5, 0, 16, 4) {
			return
		}
		sw := s.Sub(r+5, 0, 16, 4)
		x5 := mc8H4S(sw.Row16(0, 0), hc1, hc2, hc3, hc4, rnd).Shift(sh).ExtendLo4ToInt32()
		x6 := mc8H4S(sw.Row16(1, 0), hc1, hc2, hc3, hc4, rnd).Shift(sh).ExtendLo4ToInt32()
		a0 := x5.MulAdd(vc5, x4.MulAdd(vc4, x3.MulAdd(vc3, x2.MulAdd(vc2, x1.MulAdd(vc1, x0.MulAdd(vc0, vrnd))))))
		a1 := x6.MulAdd(vc5, x5.MulAdd(vc4, x4.MulAdd(vc3, x3.MulAdd(vc2, x2.MulAdd(vc1, x1.MulAdd(vc0, vrnd))))))
		a0.ToBits().ReshapeToUint16s().ConcatOdd(a1.ToBits().ReshapeToUint16s()).StoreArray((*[8]uint16)(ot[4*r : 4*r+8]))
		x7 := mc8H4S(sw.Row16(2, 0), hc1, hc2, hc3, hc4, rnd).Shift(sh).ExtendLo4ToInt32()
		x8 := mc8H4S(sw.Row16(3, 0), hc1, hc2, hc3, hc4, rnd).Shift(sh).ExtendLo4ToInt32()
		a2 := x7.MulAdd(vc5, x6.MulAdd(vc4, x5.MulAdd(vc3, x4.MulAdd(vc2, x3.MulAdd(vc1, x2.MulAdd(vc0, vrnd))))))
		a3 := x8.MulAdd(vc5, x7.MulAdd(vc4, x6.MulAdd(vc3, x5.MulAdd(vc2, x4.MulAdd(vc1, x3.MulAdd(vc0, vrnd))))))
		a2.ToBits().ReshapeToUint16s().ConcatOdd(a3.ToBits().ReshapeToUint16s()).StoreArray((*[8]uint16)(ot[4*r+8 : 4*r+16]))
		x0, x1, x2, x3, x4 = x4, x5, x6, x7, x8
	}
	for ; r+2 <= h; r += 2 {
		x5 := mc8H4S(s.Row16(r+5, 0), hc1, hc2, hc3, hc4, rnd).Shift(sh).ExtendLo4ToInt32()
		x6 := mc8H4S(s.Row16(r+6, 0), hc1, hc2, hc3, hc4, rnd).Shift(sh).ExtendLo4ToInt32()
		a0 := x5.MulAdd(vc5, x4.MulAdd(vc4, x3.MulAdd(vc3, x2.MulAdd(vc2, x1.MulAdd(vc1, x0.MulAdd(vc0, vrnd))))))
		a1 := x6.MulAdd(vc5, x5.MulAdd(vc4, x4.MulAdd(vc3, x3.MulAdd(vc2, x2.MulAdd(vc1, x1.MulAdd(vc0, vrnd))))))
		a0.ToBits().ReshapeToUint16s().ConcatOdd(a1.ToBits().ReshapeToUint16s()).StoreArray((*[8]uint16)(ot[4*r : 4*r+8]))
		x0, x1, x2, x3, x4 = x2, x3, x4, x5, x6
	}
	if r < h {
		x5 := mc8H4S(s.Row16(r+5, 0), hc1, hc2, hc3, hc4, rnd).Shift(sh).ExtendLo4ToInt32()
		a0 := x5.MulAdd(vc5, x4.MulAdd(vc4, x3.MulAdd(vc3, x2.MulAdd(vc2, x1.MulAdd(vc1, x0.MulAdd(vc0, vrnd))))))
		v := a0.ToBits().ReshapeToUint16s().ConcatOdd(a0.ToBits().ReshapeToUint16s()).ReshapeToUint64s().GetElem(0)
		mc8Store4U16((*[4]uint16)(ot[4*r:4*r+4]), v)
	}
}

// mc8PrepHV2 is the fused 2-tap bilinear compound 2D pass over 8-column strips
// (taps 2 and 3 of the 6-tap window on both axes), with no intermediate buffer.
func mc8PrepHV2(sg unsafeutil.Grid[uint8], w, h int, phx, phy *mc8Phase, og unsafeutil.Grid[uint16]) {
	hc2 := archsimd.LoadInt16x8Array(&phx.h[2])
	hc3 := archsimd.LoadInt16x8Array(&phx.h[3])
	rnd := archsimd.BroadcastInt16x8(2)
	sh := archsimd.LoadInt16x8Array(&mc8Vec.sh2)
	vc2 := archsimd.LoadInt32x4Array(&phy.p[2])
	vc3 := archsimd.LoadInt32x4Array(&phy.p[3])
	vrnd := archsimd.BroadcastInt32x4(mc8PrepV6Rnd)
	rows := h + 4
	for c := 0; c < w; c += 8 {
		if !sg.Fits(0, c, 16, rows) || !og.Fits(0, c, 8, h) {
			return
		}
		s, o := sg.Sub(0, c, 16, rows), og.Sub(0, c, 8, h)
		a := archsimd.LoadUint8x16Array(s.Row16(2, 0))
		lo, hi := a.ExtendLo8ToUint16().ReshapeToUint8s(), a.HiToLo().ExtendLo8ToUint16().ReshapeToUint8s()
		h0 := hi.ConcatShiftBytesRight(lo, 6).ReshapeToUint16s().BitsToInt16().MulAdd(hc3, hi.ConcatShiftBytesRight(lo, 4).ReshapeToUint16s().BitsToInt16().MulAdd(hc2, rnd)).Shift(sh)
		x0, y0 := h0.ExtendLo4ToInt32(), h0.HiToLo().ExtendLo4ToInt32()
		for r := 0; r < h; r++ {
			a = archsimd.LoadUint8x16Array(s.Row16(r+3, 0))
			lo, hi = a.ExtendLo8ToUint16().ReshapeToUint8s(), a.HiToLo().ExtendLo8ToUint16().ReshapeToUint8s()
			h1 := hi.ConcatShiftBytesRight(lo, 6).ReshapeToUint16s().BitsToInt16().MulAdd(hc3, hi.ConcatShiftBytesRight(lo, 4).ReshapeToUint16s().BitsToInt16().MulAdd(hc2, rnd)).Shift(sh)
			x1, y1 := h1.ExtendLo4ToInt32(), h1.HiToLo().ExtendLo4ToInt32()
			a0 := x1.MulAdd(vc3, x0.MulAdd(vc2, vrnd))
			b0 := y1.MulAdd(vc3, y0.MulAdd(vc2, vrnd))
			a0.ToBits().ReshapeToUint16s().ConcatOdd(b0.ToBits().ReshapeToUint16s()).StoreArray(o.Row8(r, 0))
			x0, y0 = x1, y1
		}
	}
}

// mc8PrepHV6x4 is the fused compound 2D 6-tap H x 4-tap V pass over 8-column
// strips directly in registers (23 NEON regs, zero spills, no intermediate buffer).
func mc8PrepHV6x4(sg unsafeutil.Grid[uint8], w, h int, phx, phy *mc8Phase, og unsafeutil.Grid[uint16]) {
	hc0 := archsimd.LoadInt16x8Array(&phx.h[0])
	hc1 := archsimd.LoadInt16x8Array(&phx.h[1])
	hc2 := archsimd.LoadInt16x8Array(&phx.h[2])
	hc3 := archsimd.LoadInt16x8Array(&phx.h[3])
	hc4 := archsimd.LoadInt16x8Array(&phx.h[4])
	hc5 := archsimd.LoadInt16x8Array(&phx.h[5])
	rnd := archsimd.BroadcastInt16x8(2)
	sh := archsimd.LoadInt16x8Array(&mc8Vec.sh2)
	vc1 := archsimd.LoadInt32x4Array(&phy.p[1])
	vc2 := archsimd.LoadInt32x4Array(&phy.p[2])
	vc3 := archsimd.LoadInt32x4Array(&phy.p[3])
	vc4 := archsimd.LoadInt32x4Array(&phy.p[4])
	vrnd := archsimd.BroadcastInt32x4(mc8PrepV6Rnd)
	rows := h + 4
	for c := 0; c < w; c += 8 {
		if !sg.Fits(0, c, 16, rows) || !og.Fits(0, c, 8, h) {
			return
		}
		s, o := sg.Sub(0, c, 16, rows), og.Sub(0, c, 8, h)
		if !s.Fits(1, 0, 16, 3) {
			return
		}
		hd := s.Sub(1, 0, 16, 3)
		a := mc8H6S(hd.Row16(0, 0), hc0, hc1, hc2, hc3, hc4, hc5, rnd).Shift(sh)
		x1, y1 := a.ExtendLo4ToInt32(), a.HiToLo().ExtendLo4ToInt32()
		a = mc8H6S(hd.Row16(1, 0), hc0, hc1, hc2, hc3, hc4, hc5, rnd).Shift(sh)
		x2, y2 := a.ExtendLo4ToInt32(), a.HiToLo().ExtendLo4ToInt32()
		a = mc8H6S(hd.Row16(2, 0), hc0, hc1, hc2, hc3, hc4, hc5, rnd).Shift(sh)
		x3, y3 := a.ExtendLo4ToInt32(), a.HiToLo().ExtendLo4ToInt32()
		r := 0
		for ; r+4 <= h; r += 4 {
			if !s.Fits(r+4, 0, 16, 4) || !o.Fits(r, 0, 8, 4) {
				return
			}
			sw, ow := s.Sub(r+4, 0, 16, 4), o.Sub(r, 0, 8, 4)
			a4 := mc8H6S(sw.Row16(0, 0), hc0, hc1, hc2, hc3, hc4, hc5, rnd).Shift(sh)
			x4, y4 := a4.ExtendLo4ToInt32(), a4.HiToLo().ExtendLo4ToInt32()
			a5 := mc8H6S(sw.Row16(1, 0), hc0, hc1, hc2, hc3, hc4, hc5, rnd).Shift(sh)
			x5, y5 := a5.ExtendLo4ToInt32(), a5.HiToLo().ExtendLo4ToInt32()
			p0 := x4.MulAdd(vc4, x3.MulAdd(vc3, x2.MulAdd(vc2, x1.MulAdd(vc1, vrnd))))
			q0 := y4.MulAdd(vc4, y3.MulAdd(vc3, y2.MulAdd(vc2, y1.MulAdd(vc1, vrnd))))
			p1 := x5.MulAdd(vc4, x4.MulAdd(vc3, x3.MulAdd(vc2, x2.MulAdd(vc1, vrnd))))
			q1 := y5.MulAdd(vc4, y4.MulAdd(vc3, y3.MulAdd(vc2, y2.MulAdd(vc1, vrnd))))
			p0.ToBits().ReshapeToUint16s().ConcatOdd(q0.ToBits().ReshapeToUint16s()).StoreArray(ow.Row8(0, 0))
			p1.ToBits().ReshapeToUint16s().ConcatOdd(q1.ToBits().ReshapeToUint16s()).StoreArray(ow.Row8(1, 0))
			a6 := mc8H6S(sw.Row16(2, 0), hc0, hc1, hc2, hc3, hc4, hc5, rnd).Shift(sh)
			x6, y6 := a6.ExtendLo4ToInt32(), a6.HiToLo().ExtendLo4ToInt32()
			a7 := mc8H6S(sw.Row16(3, 0), hc0, hc1, hc2, hc3, hc4, hc5, rnd).Shift(sh)
			x7, y7 := a7.ExtendLo4ToInt32(), a7.HiToLo().ExtendLo4ToInt32()
			p2 := x6.MulAdd(vc4, x5.MulAdd(vc3, x4.MulAdd(vc2, x3.MulAdd(vc1, vrnd))))
			q2 := y6.MulAdd(vc4, y5.MulAdd(vc3, y4.MulAdd(vc2, y3.MulAdd(vc1, vrnd))))
			p3 := x7.MulAdd(vc4, x6.MulAdd(vc3, x5.MulAdd(vc2, x4.MulAdd(vc1, vrnd))))
			q3 := y7.MulAdd(vc4, y6.MulAdd(vc3, y5.MulAdd(vc2, y4.MulAdd(vc1, vrnd))))
			p2.ToBits().ReshapeToUint16s().ConcatOdd(q2.ToBits().ReshapeToUint16s()).StoreArray(ow.Row8(2, 0))
			p3.ToBits().ReshapeToUint16s().ConcatOdd(q3.ToBits().ReshapeToUint16s()).StoreArray(ow.Row8(3, 0))
			x1, y1, x2, y2, x3, y3 = x5, y5, x6, y6, x7, y7
		}
		for ; r < h; r++ {
			a4 := mc8H6S(s.Row16(r+4, 0), hc0, hc1, hc2, hc3, hc4, hc5, rnd).Shift(sh)
			x4, y4 := a4.ExtendLo4ToInt32(), a4.HiToLo().ExtendLo4ToInt32()
			p0 := x4.MulAdd(vc4, x3.MulAdd(vc3, x2.MulAdd(vc2, x1.MulAdd(vc1, vrnd))))
			q0 := y4.MulAdd(vc4, y3.MulAdd(vc3, y2.MulAdd(vc2, y1.MulAdd(vc1, vrnd))))
			p0.ToBits().ReshapeToUint16s().ConcatOdd(q0.ToBits().ReshapeToUint16s()).StoreArray(o.Row8(r, 0))
			x1, y1, x2, y2, x3, y3 = x2, y2, x3, y3, x4, y4
		}
	}
}

// mc8PutHV4x4W4 is mc8PutHV4W4 when both horizontal and vertical filters have
// at most 4 non-zero taps (taps 1..4): skips rows 0 and h+4 and drops the zero
// outer vertical taps.
func mc8PutHV4x4W4(sg unsafeutil.Grid[uint8], h int, phx, phy *mc8Phase, dg unsafeutil.Grid[uint8]) {
	hc1 := archsimd.LoadInt16x8Array(&phx.h[1])
	hc2 := archsimd.LoadInt16x8Array(&phx.h[2])
	hc3 := archsimd.LoadInt16x8Array(&phx.h[3])
	hc4 := archsimd.LoadInt16x8Array(&phx.h[4])
	rnd := archsimd.BroadcastInt16x8(mc8PutH6ToI32Rnd)
	sh := archsimd.LoadInt16x8Array(&mc8Vec.sh2)
	vc1 := archsimd.LoadInt32x4Array(&phy.v[1])
	vc2 := archsimd.LoadInt32x4Array(&phy.v[2])
	vc3 := archsimd.LoadInt32x4Array(&phy.v[3])
	vc4 := archsimd.LoadInt32x4Array(&phy.v[4])
	rows := h + 4
	if !sg.Fits(0, 0, 16, rows) || !dg.Fits(0, 0, 4, h) {
		return
	}
	s, d := sg.Sub(0, 0, 16, rows), dg.Sub(0, 0, 4, h)
	if !s.Fits(1, 0, 16, 3) {
		return
	}
	hd := s.Sub(1, 0, 16, 3)
	x1 := mc8H4S(hd.Row16(0, 0), hc1, hc2, hc3, hc4, rnd).Shift(sh).ExtendLo4ToInt32()
	x2 := mc8H4S(hd.Row16(1, 0), hc1, hc2, hc3, hc4, rnd).Shift(sh).ExtendLo4ToInt32()
	x3 := mc8H4S(hd.Row16(2, 0), hc1, hc2, hc3, hc4, rnd).Shift(sh).ExtendLo4ToInt32()
	r := 0
	for ; r+4 <= h; r += 4 {
		if !s.Fits(r+4, 0, 16, 4) || !d.Fits(r, 0, 4, 4) {
			return
		}
		sw, dw := s.Sub(r+4, 0, 16, 4), d.Sub(r, 0, 4, 4)
		x4 := mc8H4S(sw.Row16(0, 0), hc1, hc2, hc3, hc4, rnd).Shift(sh).ExtendLo4ToInt32()
		x5 := mc8H4S(sw.Row16(1, 0), hc1, hc2, hc3, hc4, rnd).Shift(sh).ExtendLo4ToInt32()
		a0 := x4.MulAdd(vc4, x3.MulAdd(vc3, x2.MulAdd(vc2, x1.Mul(vc1))))
		a1 := x5.MulAdd(vc4, x4.MulAdd(vc3, x3.MulAdd(vc2, x2.Mul(vc1))))
		v0 := a0.ToBits().ReshapeToUint16s().ConcatOdd(a1.ToBits().ReshapeToUint16s()).BitsToInt16().SaturateToUint8().ReshapeToUint32s()
		binary.LittleEndian.PutUint32(dw.Row4(0, 0)[:], v0.GetElem(0))
		binary.LittleEndian.PutUint32(dw.Row4(1, 0)[:], v0.GetElem(1))
		x6 := mc8H4S(sw.Row16(2, 0), hc1, hc2, hc3, hc4, rnd).Shift(sh).ExtendLo4ToInt32()
		x7 := mc8H4S(sw.Row16(3, 0), hc1, hc2, hc3, hc4, rnd).Shift(sh).ExtendLo4ToInt32()
		a2 := x6.MulAdd(vc4, x5.MulAdd(vc3, x4.MulAdd(vc2, x3.Mul(vc1))))
		a3 := x7.MulAdd(vc4, x6.MulAdd(vc3, x5.MulAdd(vc2, x4.Mul(vc1))))
		v1 := a2.ToBits().ReshapeToUint16s().ConcatOdd(a3.ToBits().ReshapeToUint16s()).BitsToInt16().SaturateToUint8().ReshapeToUint32s()
		binary.LittleEndian.PutUint32(dw.Row4(2, 0)[:], v1.GetElem(0))
		binary.LittleEndian.PutUint32(dw.Row4(3, 0)[:], v1.GetElem(1))
		x1, x2, x3 = x5, x6, x7
	}
	for ; r+2 <= h; r += 2 {
		x4 := mc8H4S(s.Row16(r+4, 0), hc1, hc2, hc3, hc4, rnd).Shift(sh).ExtendLo4ToInt32()
		x5 := mc8H4S(s.Row16(r+5, 0), hc1, hc2, hc3, hc4, rnd).Shift(sh).ExtendLo4ToInt32()
		a0 := x4.MulAdd(vc4, x3.MulAdd(vc3, x2.MulAdd(vc2, x1.Mul(vc1))))
		a1 := x5.MulAdd(vc4, x4.MulAdd(vc3, x3.MulAdd(vc2, x2.Mul(vc1))))
		v := a0.ToBits().ReshapeToUint16s().ConcatOdd(a1.ToBits().ReshapeToUint16s()).BitsToInt16().SaturateToUint8().ReshapeToUint32s()
		binary.LittleEndian.PutUint32(d.Row4(r, 0)[:], v.GetElem(0))
		binary.LittleEndian.PutUint32(d.Row4(r+1, 0)[:], v.GetElem(1))
		x1, x2, x3 = x3, x4, x5
	}
	if r < h {
		x4 := mc8H4S(s.Row16(r+4, 0), hc1, hc2, hc3, hc4, rnd).Shift(sh).ExtendLo4ToInt32()
		a0 := x4.MulAdd(vc4, x3.MulAdd(vc3, x2.MulAdd(vc2, x1.Mul(vc1))))
		v := a0.ToBits().ReshapeToUint16s().ConcatOdd(a0.ToBits().ReshapeToUint16s()).BitsToInt16().SaturateToUint8().ReshapeToUint32s()
		binary.LittleEndian.PutUint32(d.Row4(r, 0)[:], v.GetElem(0))
	}
}

// mc8PutHV4W4 is the fused width-4 put 2D pass when the horizontal filter has
// at most 4 non-zero taps (regular4, smooth4, bilinear). All state stays in
// NEON registers with no intermediate buffer.
func mc8PutHV4W4(sg unsafeutil.Grid[uint8], h int, phx, phy *mc8Phase, dg unsafeutil.Grid[uint8]) {
	hc1 := archsimd.LoadInt16x8Array(&phx.h[1])
	hc2 := archsimd.LoadInt16x8Array(&phx.h[2])
	hc3 := archsimd.LoadInt16x8Array(&phx.h[3])
	hc4 := archsimd.LoadInt16x8Array(&phx.h[4])
	rnd := archsimd.BroadcastInt16x8(mc8PutH6ToI32Rnd)
	sh := archsimd.LoadInt16x8Array(&mc8Vec.sh2)
	vc0 := archsimd.LoadInt32x4Array(&phy.v[0])
	vc1 := archsimd.LoadInt32x4Array(&phy.v[1])
	vc2 := archsimd.LoadInt32x4Array(&phy.v[2])
	vc3 := archsimd.LoadInt32x4Array(&phy.v[3])
	vc4 := archsimd.LoadInt32x4Array(&phy.v[4])
	vc5 := archsimd.LoadInt32x4Array(&phy.v[5])
	rows := h + 5
	if !sg.Fits(0, 0, 16, rows) || !dg.Fits(0, 0, 4, h) {
		return
	}
	s, d := sg.Sub(0, 0, 16, rows), dg.Sub(0, 0, 4, h)
	if !s.Fits(0, 0, 16, 5) {
		return
	}
	hd := s.Sub(0, 0, 16, 5)
	x0 := mc8H4S(hd.Row16(0, 0), hc1, hc2, hc3, hc4, rnd).Shift(sh).ExtendLo4ToInt32()
	x1 := mc8H4S(hd.Row16(1, 0), hc1, hc2, hc3, hc4, rnd).Shift(sh).ExtendLo4ToInt32()
	x2 := mc8H4S(hd.Row16(2, 0), hc1, hc2, hc3, hc4, rnd).Shift(sh).ExtendLo4ToInt32()
	x3 := mc8H4S(hd.Row16(3, 0), hc1, hc2, hc3, hc4, rnd).Shift(sh).ExtendLo4ToInt32()
	x4 := mc8H4S(hd.Row16(4, 0), hc1, hc2, hc3, hc4, rnd).Shift(sh).ExtendLo4ToInt32()
	r := 0
	for ; r+4 <= h; r += 4 {
		if !s.Fits(r+5, 0, 16, 4) || !d.Fits(r, 0, 4, 4) {
			return
		}
		sw, dw := s.Sub(r+5, 0, 16, 4), d.Sub(r, 0, 4, 4)
		x5 := mc8H4S(sw.Row16(0, 0), hc1, hc2, hc3, hc4, rnd).Shift(sh).ExtendLo4ToInt32()
		x6 := mc8H4S(sw.Row16(1, 0), hc1, hc2, hc3, hc4, rnd).Shift(sh).ExtendLo4ToInt32()
		a0 := x5.MulAdd(vc5, x4.MulAdd(vc4, x3.MulAdd(vc3, x2.MulAdd(vc2, x1.MulAdd(vc1, x0.Mul(vc0))))))
		a1 := x6.MulAdd(vc5, x5.MulAdd(vc4, x4.MulAdd(vc3, x3.MulAdd(vc2, x2.MulAdd(vc1, x1.Mul(vc0))))))
		v0 := a0.ToBits().ReshapeToUint16s().ConcatOdd(a1.ToBits().ReshapeToUint16s()).BitsToInt16().SaturateToUint8().ReshapeToUint32s()
		binary.LittleEndian.PutUint32(dw.Row4(0, 0)[:], v0.GetElem(0))
		binary.LittleEndian.PutUint32(dw.Row4(1, 0)[:], v0.GetElem(1))
		x7 := mc8H4S(sw.Row16(2, 0), hc1, hc2, hc3, hc4, rnd).Shift(sh).ExtendLo4ToInt32()
		x8 := mc8H4S(sw.Row16(3, 0), hc1, hc2, hc3, hc4, rnd).Shift(sh).ExtendLo4ToInt32()
		a2 := x7.MulAdd(vc5, x6.MulAdd(vc4, x5.MulAdd(vc3, x4.MulAdd(vc2, x3.MulAdd(vc1, x2.Mul(vc0))))))
		a3 := x8.MulAdd(vc5, x7.MulAdd(vc4, x6.MulAdd(vc3, x5.MulAdd(vc2, x4.MulAdd(vc1, x3.Mul(vc0))))))
		v1 := a2.ToBits().ReshapeToUint16s().ConcatOdd(a3.ToBits().ReshapeToUint16s()).BitsToInt16().SaturateToUint8().ReshapeToUint32s()
		binary.LittleEndian.PutUint32(dw.Row4(2, 0)[:], v1.GetElem(0))
		binary.LittleEndian.PutUint32(dw.Row4(3, 0)[:], v1.GetElem(1))
		x0, x1, x2, x3, x4 = x4, x5, x6, x7, x8
	}
	for ; r+2 <= h; r += 2 {
		x5 := mc8H4S(s.Row16(r+5, 0), hc1, hc2, hc3, hc4, rnd).Shift(sh).ExtendLo4ToInt32()
		x6 := mc8H4S(s.Row16(r+6, 0), hc1, hc2, hc3, hc4, rnd).Shift(sh).ExtendLo4ToInt32()
		a0 := x5.MulAdd(vc5, x4.MulAdd(vc4, x3.MulAdd(vc3, x2.MulAdd(vc2, x1.MulAdd(vc1, x0.Mul(vc0))))))
		a1 := x6.MulAdd(vc5, x5.MulAdd(vc4, x4.MulAdd(vc3, x3.MulAdd(vc2, x2.MulAdd(vc1, x1.Mul(vc0))))))
		v := a0.ToBits().ReshapeToUint16s().ConcatOdd(a1.ToBits().ReshapeToUint16s()).BitsToInt16().SaturateToUint8().ReshapeToUint32s()
		binary.LittleEndian.PutUint32(d.Row4(r, 0)[:], v.GetElem(0))
		binary.LittleEndian.PutUint32(d.Row4(r+1, 0)[:], v.GetElem(1))
		x0, x1, x2, x3, x4 = x2, x3, x4, x5, x6
	}
	if r < h {
		x5 := mc8H4S(s.Row16(r+5, 0), hc1, hc2, hc3, hc4, rnd).Shift(sh).ExtendLo4ToInt32()
		a0 := x5.MulAdd(vc5, x4.MulAdd(vc4, x3.MulAdd(vc3, x2.MulAdd(vc2, x1.MulAdd(vc1, x0.Mul(vc0))))))
		v := a0.ToBits().ReshapeToUint16s().ConcatOdd(a0.ToBits().ReshapeToUint16s()).BitsToInt16().SaturateToUint8().ReshapeToUint32s()
		binary.LittleEndian.PutUint32(d.Row4(r, 0)[:], v.GetElem(0))
	}
}

// mc8PutHV2 is the fused 2-tap bilinear put 2D pass over 8-column strips
// (taps 2 and 3 of the 6-tap window on both axes).
func mc8PutHV2(sg unsafeutil.Grid[uint8], w, h int, phx, phy *mc8Phase, dg unsafeutil.Grid[uint8]) {
	hc2 := archsimd.LoadInt16x8Array(&phx.h[2])
	hc3 := archsimd.LoadInt16x8Array(&phx.h[3])
	rnd := archsimd.BroadcastInt16x8(mc8PutH6ToI32Rnd)
	sh := archsimd.LoadInt16x8Array(&mc8Vec.sh2)
	vc2 := archsimd.LoadInt32x4Array(&phy.v[2])
	vc3 := archsimd.LoadInt32x4Array(&phy.v[3])
	rows := h + 5
	for c := 0; c < w; c += 8 {
		if !sg.Fits(0, c, 16, rows) || !dg.Fits(0, c, 8, h) {
			return
		}
		s, d := sg.Sub(0, c, 16, rows), dg.Sub(0, c, 8, h)
		a := archsimd.LoadUint8x16Array(s.Row16(2, 0))
		lo, hi := a.ExtendLo8ToUint16().ReshapeToUint8s(), a.HiToLo().ExtendLo8ToUint16().ReshapeToUint8s()
		h0 := hi.ConcatShiftBytesRight(lo, 6).ReshapeToUint16s().BitsToInt16().MulAdd(hc3, hi.ConcatShiftBytesRight(lo, 4).ReshapeToUint16s().BitsToInt16().MulAdd(hc2, rnd)).Shift(sh)
		x0, y0 := h0.ExtendLo4ToInt32(), h0.HiToLo().ExtendLo4ToInt32()
		for r := 0; r < h; r++ {
			a = archsimd.LoadUint8x16Array(s.Row16(r+3, 0))
			lo, hi = a.ExtendLo8ToUint16().ReshapeToUint8s(), a.HiToLo().ExtendLo8ToUint16().ReshapeToUint8s()
			h1 := hi.ConcatShiftBytesRight(lo, 6).ReshapeToUint16s().BitsToInt16().MulAdd(hc3, hi.ConcatShiftBytesRight(lo, 4).ReshapeToUint16s().BitsToInt16().MulAdd(hc2, rnd)).Shift(sh)
			x1, y1 := h1.ExtendLo4ToInt32(), h1.HiToLo().ExtendLo4ToInt32()
			a0 := x1.MulAdd(vc3, x0.Mul(vc2))
			b0 := y1.MulAdd(vc3, y0.Mul(vc2))
			v0 := a0.ToBits().ReshapeToUint16s().ConcatOdd(b0.ToBits().ReshapeToUint16s()).BitsToInt16()
			binary.LittleEndian.PutUint64(d.Row8(r, 0)[:], mc8Lo64(v0.SaturateToUint8()))
			x0, y0 = x1, y1
		}
	}
}

// mc8PutHV6x4 is mc8PutHV6 when the vertical filter has at most 4 non-zero
// taps (taps 1..4, e.g. h == 4 regular4/smooth4/bilinear): skips rows 0 and
// h+4 and drops the zero outer vertical taps.
func mc8PutHV6x4(sg unsafeutil.Grid[uint8], w, h int, phx, phy *mc8Phase, dg unsafeutil.Grid[uint8]) {
	hc0 := archsimd.LoadInt16x8Array(&phx.h[0])
	hc1 := archsimd.LoadInt16x8Array(&phx.h[1])
	hc2 := archsimd.LoadInt16x8Array(&phx.h[2])
	hc3 := archsimd.LoadInt16x8Array(&phx.h[3])
	hc4 := archsimd.LoadInt16x8Array(&phx.h[4])
	hc5 := archsimd.LoadInt16x8Array(&phx.h[5])
	rnd := archsimd.BroadcastInt16x8(mc8PutH6ToI32Rnd)
	sh := archsimd.LoadInt16x8Array(&mc8Vec.sh2)
	vc1 := archsimd.LoadInt32x4Array(&phy.v[1])
	vc2 := archsimd.LoadInt32x4Array(&phy.v[2])
	vc3 := archsimd.LoadInt32x4Array(&phy.v[3])
	vc4 := archsimd.LoadInt32x4Array(&phy.v[4])
	rows := h + 4
	for c := 0; c < w; c += 8 {
		if !sg.Fits(0, c, 16, rows) || !dg.Fits(0, c, 8, h) {
			return
		}
		s, d := sg.Sub(0, c, 16, rows), dg.Sub(0, c, 8, h)
		if !s.Fits(1, 0, 16, 3) {
			return
		}
		hd := s.Sub(1, 0, 16, 3)
		a := mc8H6S(hd.Row16(0, 0), hc0, hc1, hc2, hc3, hc4, hc5, rnd).Shift(sh)
		x1, y1 := a.ExtendLo4ToInt32(), a.HiToLo().ExtendLo4ToInt32()
		a = mc8H6S(hd.Row16(1, 0), hc0, hc1, hc2, hc3, hc4, hc5, rnd).Shift(sh)
		x2, y2 := a.ExtendLo4ToInt32(), a.HiToLo().ExtendLo4ToInt32()
		a = mc8H6S(hd.Row16(2, 0), hc0, hc1, hc2, hc3, hc4, hc5, rnd).Shift(sh)
		x3, y3 := a.ExtendLo4ToInt32(), a.HiToLo().ExtendLo4ToInt32()
		r := 0
		for ; r+4 <= h; r += 4 {
			if !s.Fits(r+4, 0, 16, 4) || !d.Fits(r, 0, 8, 4) {
				return
			}
			sw, dw := s.Sub(r+4, 0, 16, 4), d.Sub(r, 0, 8, 4)
			a4 := mc8H6S(sw.Row16(0, 0), hc0, hc1, hc2, hc3, hc4, hc5, rnd).Shift(sh)
			x4, y4 := a4.ExtendLo4ToInt32(), a4.HiToLo().ExtendLo4ToInt32()
			a5 := mc8H6S(sw.Row16(1, 0), hc0, hc1, hc2, hc3, hc4, hc5, rnd).Shift(sh)
			x5, y5 := a5.ExtendLo4ToInt32(), a5.HiToLo().ExtendLo4ToInt32()
			p0 := x4.MulAdd(vc4, x3.MulAdd(vc3, x2.MulAdd(vc2, x1.Mul(vc1))))
			q0 := y4.MulAdd(vc4, y3.MulAdd(vc3, y2.MulAdd(vc2, y1.Mul(vc1))))
			p1 := x5.MulAdd(vc4, x4.MulAdd(vc3, x3.MulAdd(vc2, x2.Mul(vc1))))
			q1 := y5.MulAdd(vc4, y4.MulAdd(vc3, y3.MulAdd(vc2, y2.Mul(vc1))))
			v0 := p0.ToBits().ReshapeToUint16s().ConcatOdd(q0.ToBits().ReshapeToUint16s()).BitsToInt16()
			v1 := p1.ToBits().ReshapeToUint16s().ConcatOdd(q1.ToBits().ReshapeToUint16s()).BitsToInt16()
			binary.LittleEndian.PutUint64(dw.Row8(0, 0)[:], mc8Lo64(v0.SaturateToUint8()))
			binary.LittleEndian.PutUint64(dw.Row8(1, 0)[:], mc8Lo64(v1.SaturateToUint8()))
			a6 := mc8H6S(sw.Row16(2, 0), hc0, hc1, hc2, hc3, hc4, hc5, rnd).Shift(sh)
			x6, y6 := a6.ExtendLo4ToInt32(), a6.HiToLo().ExtendLo4ToInt32()
			a7 := mc8H6S(sw.Row16(3, 0), hc0, hc1, hc2, hc3, hc4, hc5, rnd).Shift(sh)
			x7, y7 := a7.ExtendLo4ToInt32(), a7.HiToLo().ExtendLo4ToInt32()
			p2 := x6.MulAdd(vc4, x5.MulAdd(vc3, x4.MulAdd(vc2, x3.Mul(vc1))))
			q2 := y6.MulAdd(vc4, y5.MulAdd(vc3, y4.MulAdd(vc2, y3.Mul(vc1))))
			p3 := x7.MulAdd(vc4, x6.MulAdd(vc3, x5.MulAdd(vc2, x4.Mul(vc1))))
			q3 := y7.MulAdd(vc4, y6.MulAdd(vc3, y5.MulAdd(vc2, y4.Mul(vc1))))
			v2 := p2.ToBits().ReshapeToUint16s().ConcatOdd(q2.ToBits().ReshapeToUint16s()).BitsToInt16()
			v3 := p3.ToBits().ReshapeToUint16s().ConcatOdd(q3.ToBits().ReshapeToUint16s()).BitsToInt16()
			binary.LittleEndian.PutUint64(dw.Row8(2, 0)[:], mc8Lo64(v2.SaturateToUint8()))
			binary.LittleEndian.PutUint64(dw.Row8(3, 0)[:], mc8Lo64(v3.SaturateToUint8()))
			x1, y1, x2, y2, x3, y3 = x5, y5, x6, y6, x7, y7
		}
		for ; r+2 <= h; r += 2 {
			a4 := mc8H6S(s.Row16(r+4, 0), hc0, hc1, hc2, hc3, hc4, hc5, rnd).Shift(sh)
			x4, y4 := a4.ExtendLo4ToInt32(), a4.HiToLo().ExtendLo4ToInt32()
			a5 := mc8H6S(s.Row16(r+5, 0), hc0, hc1, hc2, hc3, hc4, hc5, rnd).Shift(sh)
			x5, y5 := a5.ExtendLo4ToInt32(), a5.HiToLo().ExtendLo4ToInt32()
			p0 := x4.MulAdd(vc4, x3.MulAdd(vc3, x2.MulAdd(vc2, x1.Mul(vc1))))
			q0 := y4.MulAdd(vc4, y3.MulAdd(vc3, y2.MulAdd(vc2, y1.Mul(vc1))))
			p1 := x5.MulAdd(vc4, x4.MulAdd(vc3, x3.MulAdd(vc2, x2.Mul(vc1))))
			q1 := y5.MulAdd(vc4, y4.MulAdd(vc3, y3.MulAdd(vc2, y2.Mul(vc1))))
			v0 := p0.ToBits().ReshapeToUint16s().ConcatOdd(q0.ToBits().ReshapeToUint16s()).BitsToInt16()
			v1 := p1.ToBits().ReshapeToUint16s().ConcatOdd(q1.ToBits().ReshapeToUint16s()).BitsToInt16()
			binary.LittleEndian.PutUint64(d.Row8(r, 0)[:], mc8Lo64(v0.SaturateToUint8()))
			binary.LittleEndian.PutUint64(d.Row8(r+1, 0)[:], mc8Lo64(v1.SaturateToUint8()))
			x1, y1, x2, y2, x3, y3 = x3, y3, x4, y4, x5, y5
		}
		if r < h {
			a4 := mc8H6S(s.Row16(r+4, 0), hc0, hc1, hc2, hc3, hc4, hc5, rnd).Shift(sh)
			x4, y4 := a4.ExtendLo4ToInt32(), a4.HiToLo().ExtendLo4ToInt32()
			p0 := x4.MulAdd(vc4, x3.MulAdd(vc3, x2.MulAdd(vc2, x1.Mul(vc1))))
			q0 := y4.MulAdd(vc4, y3.MulAdd(vc3, y2.MulAdd(vc2, y1.Mul(vc1))))
			v0 := p0.ToBits().ReshapeToUint16s().ConcatOdd(q0.ToBits().ReshapeToUint16s()).BitsToInt16()
			binary.LittleEndian.PutUint64(d.Row8(r, 0)[:], mc8Lo64(v0.SaturateToUint8()))
		}
	}
}

// mc8PutHV6 is the fused put 2D six-tap horizontal+vertical pass over 8-column
// strips directly in registers with no intermediate buffer.
func mc8PutHV6(sg unsafeutil.Grid[uint8], w, h int, phx, phy *mc8Phase, dg unsafeutil.Grid[uint8]) {
	hc0 := archsimd.LoadInt16x8Array(&phx.h[0])
	hc1 := archsimd.LoadInt16x8Array(&phx.h[1])
	hc2 := archsimd.LoadInt16x8Array(&phx.h[2])
	hc3 := archsimd.LoadInt16x8Array(&phx.h[3])
	hc4 := archsimd.LoadInt16x8Array(&phx.h[4])
	hc5 := archsimd.LoadInt16x8Array(&phx.h[5])
	rnd := archsimd.BroadcastInt16x8(mc8PutH6ToI32Rnd)
	sh := archsimd.LoadInt16x8Array(&mc8Vec.sh2)
	vc0 := archsimd.LoadInt32x4Array(&phy.v[0])
	vc1 := archsimd.LoadInt32x4Array(&phy.v[1])
	vc2 := archsimd.LoadInt32x4Array(&phy.v[2])
	vc3 := archsimd.LoadInt32x4Array(&phy.v[3])
	vc4 := archsimd.LoadInt32x4Array(&phy.v[4])
	vc5 := archsimd.LoadInt32x4Array(&phy.v[5])
	rows := h + 5
	for c := 0; c < w; c += 8 {
		if !sg.Fits(0, c, 16, rows) || !dg.Fits(0, c, 8, h) {
			return
		}
		s, d := sg.Sub(0, c, 16, rows), dg.Sub(0, c, 8, h)
		if !s.Fits(0, 0, 16, 5) {
			return
		}
		hd := s.Sub(0, 0, 16, 5)
		a := mc8H6S(hd.Row16(0, 0), hc0, hc1, hc2, hc3, hc4, hc5, rnd).Shift(sh)
		x0, y0 := a.ExtendLo4ToInt32(), a.HiToLo().ExtendLo4ToInt32()
		a = mc8H6S(hd.Row16(1, 0), hc0, hc1, hc2, hc3, hc4, hc5, rnd).Shift(sh)
		x1, y1 := a.ExtendLo4ToInt32(), a.HiToLo().ExtendLo4ToInt32()
		a = mc8H6S(hd.Row16(2, 0), hc0, hc1, hc2, hc3, hc4, hc5, rnd).Shift(sh)
		x2, y2 := a.ExtendLo4ToInt32(), a.HiToLo().ExtendLo4ToInt32()
		a = mc8H6S(hd.Row16(3, 0), hc0, hc1, hc2, hc3, hc4, hc5, rnd).Shift(sh)
		x3, y3 := a.ExtendLo4ToInt32(), a.HiToLo().ExtendLo4ToInt32()
		a = mc8H6S(hd.Row16(4, 0), hc0, hc1, hc2, hc3, hc4, hc5, rnd).Shift(sh)
		x4, y4 := a.ExtendLo4ToInt32(), a.HiToLo().ExtendLo4ToInt32()
		r := 0
		for ; r+4 <= h; r += 4 {
			if !s.Fits(r+5, 0, 16, 4) || !d.Fits(r, 0, 8, 4) {
				return
			}
			sw, dw := s.Sub(r+5, 0, 16, 4), d.Sub(r, 0, 8, 4)
			a5 := mc8H6S(sw.Row16(0, 0), hc0, hc1, hc2, hc3, hc4, hc5, rnd).Shift(sh)
			x5, y5 := a5.ExtendLo4ToInt32(), a5.HiToLo().ExtendLo4ToInt32()
			a6 := mc8H6S(sw.Row16(1, 0), hc0, hc1, hc2, hc3, hc4, hc5, rnd).Shift(sh)
			x6, y6 := a6.ExtendLo4ToInt32(), a6.HiToLo().ExtendLo4ToInt32()
			u0, u1 := mc8V6PutPair(x0, y0, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, vc0, vc1, vc2, vc3, vc4, vc5)
			binary.LittleEndian.PutUint64(dw.Row8(0, 0)[:], u0)
			binary.LittleEndian.PutUint64(dw.Row8(1, 0)[:], u1)
			a7 := mc8H6S(sw.Row16(2, 0), hc0, hc1, hc2, hc3, hc4, hc5, rnd).Shift(sh)
			x7, y7 := a7.ExtendLo4ToInt32(), a7.HiToLo().ExtendLo4ToInt32()
			a8 := mc8H6S(sw.Row16(3, 0), hc0, hc1, hc2, hc3, hc4, hc5, rnd).Shift(sh)
			x8, y8 := a8.ExtendLo4ToInt32(), a8.HiToLo().ExtendLo4ToInt32()
			u2, u3 := mc8V6PutPair(x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8, vc0, vc1, vc2, vc3, vc4, vc5)
			binary.LittleEndian.PutUint64(dw.Row8(2, 0)[:], u2)
			binary.LittleEndian.PutUint64(dw.Row8(3, 0)[:], u3)
			x0, y0, x1, y1, x2, y2, x3, y3, x4, y4 = x4, y4, x5, y5, x6, y6, x7, y7, x8, y8
		}
		for ; r+2 <= h; r += 2 {
			a5 := mc8H6S(s.Row16(r+5, 0), hc0, hc1, hc2, hc3, hc4, hc5, rnd).Shift(sh)
			x5, y5 := a5.ExtendLo4ToInt32(), a5.HiToLo().ExtendLo4ToInt32()
			a6 := mc8H6S(s.Row16(r+6, 0), hc0, hc1, hc2, hc3, hc4, hc5, rnd).Shift(sh)
			x6, y6 := a6.ExtendLo4ToInt32(), a6.HiToLo().ExtendLo4ToInt32()
			u0, u1 := mc8V6PutPair(x0, y0, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, vc0, vc1, vc2, vc3, vc4, vc5)
			binary.LittleEndian.PutUint64(d.Row8(r, 0)[:], u0)
			binary.LittleEndian.PutUint64(d.Row8(r+1, 0)[:], u1)
			x0, y0, x1, y1, x2, y2, x3, y3, x4, y4 = x2, y2, x3, y3, x4, y4, x5, y5, x6, y6
		}
		if r < h {
			a5 := mc8H6S(s.Row16(r+5, 0), hc0, hc1, hc2, hc3, hc4, hc5, rnd).Shift(sh)
			x5, y5 := a5.ExtendLo4ToInt32(), a5.HiToLo().ExtendLo4ToInt32()
			u0, _ := mc8V6PutPair(x0, y0, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x5, y5, vc0, vc1, vc2, vc3, vc4, vc5)
			binary.LittleEndian.PutUint64(d.Row8(r, 0)[:], u0)
		}
	}
}

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
			win := s.Row24(r, 0)
			a := mc8H6S((*[16]uint8)(win[0:16]), c0, c1, c2, c3, c4, c5, rnd).Shift(sh)
			b := mc8H6S((*[16]uint8)(win[8:24]), c0, c1, c2, c3, c4, c5, rnd).Shift(sh)
			p, q := m0.Row8(r, 0), m1.Row8(r, 0)
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
			a := mc8H6S(s.Row16(r, 0), c0, c1, c2, c3, c4, c5, rnd).Shift(sh)
			p := m.Row8(r, 0)
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
		mc8H6S(s.Row16(r, 0), c0, c1, c2, c3, c4, c5, rnd).Shift(sh).ExtendLo4ToInt32().StoreArray(m.Row4(r, 0))
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
			binary.LittleEndian.PutUint64(d.Row8(0, 0)[:], u0)
			binary.LittleEndian.PutUint64(d.Row8(1, 0)[:], u1)
			u0, u1 = mc8V6PutPair(x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8, c0, c1, c2, c3, c4, c5)
			binary.LittleEndian.PutUint64(d.Row8(2, 0)[:], u0)
			binary.LittleEndian.PutUint64(d.Row8(3, 0)[:], u1)
			u0, u1 = mc8V6PutPair(x4, y4, x5, y5, x6, y6, x7, y7, x8, y8, x9, y9, x10, y10, c0, c1, c2, c3, c4, c5)
			binary.LittleEndian.PutUint64(d.Row8(4, 0)[:], u0)
			binary.LittleEndian.PutUint64(d.Row8(5, 0)[:], u1)
			u0, u1 = mc8V6PutPair(x6, y6, x7, y7, x8, y8, x9, y9, x10, y10, x11, y11, x12, y12, c0, c1, c2, c3, c4, c5)
			binary.LittleEndian.PutUint64(d.Row8(6, 0)[:], u0)
			binary.LittleEndian.PutUint64(d.Row8(7, 0)[:], u1)
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
			binary.LittleEndian.PutUint64(d.Row8(0, 0)[:], u0)
			binary.LittleEndian.PutUint64(d.Row8(1, 0)[:], u1)
			r += 2
			x0, y0, x1, y1, x2, y2, x3, y3, x4, y4 = x2, y2, x3, y3, x4, y4, x5, y5, x6, y6
		}
		if q <= len(st)-8 {
			if !dc.Fits(r, 0, 8, 1) {
				return
			}
			x5, y5 := mc8LoadI32x8((*[8]int32)(st[q : q+8]))
			u0, _ := mc8V6PutPair(x0, y0, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x5, y5, c0, c1, c2, c3, c4, c5)
			binary.LittleEndian.PutUint64(dc.Sub(r, 0, 8, 1).Row8(0, 0)[:], u0)
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
			u0.StoreArray(o.Row8(0, 0))
			u1.StoreArray(o.Row8(1, 0))
			u0, u1 = mc8V6PrepPair(x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8, c0, c1, c2, c3, c4, c5, rnd)
			u0.StoreArray(o.Row8(2, 0))
			u1.StoreArray(o.Row8(3, 0))
			u0, u1 = mc8V6PrepPair(x4, y4, x5, y5, x6, y6, x7, y7, x8, y8, x9, y9, x10, y10, c0, c1, c2, c3, c4, c5, rnd)
			u0.StoreArray(o.Row8(4, 0))
			u1.StoreArray(o.Row8(5, 0))
			u0, u1 = mc8V6PrepPair(x6, y6, x7, y7, x8, y8, x9, y9, x10, y10, x11, y11, x12, y12, c0, c1, c2, c3, c4, c5, rnd)
			u0.StoreArray(o.Row8(6, 0))
			u1.StoreArray(o.Row8(7, 0))
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
			u0.StoreArray(o.Row8(0, 0))
			u1.StoreArray(o.Row8(1, 0))
			r += 2
			x0, y0, x1, y1, x2, y2, x3, y3, x4, y4 = x2, y2, x3, y3, x4, y4, x5, y5, x6, y6
		}
		if q <= len(st)-8 {
			if !oc.Fits(r, 0, 8, 1) {
				return
			}
			x5, y5 := mc8LoadI32x8((*[8]int32)(st[q : q+8]))
			u0, _ := mc8V6PrepPair(x0, y0, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x5, y5, c0, c1, c2, c3, c4, c5, rnd)
			u0.StoreArray(oc.Sub(r, 0, 8, 1).Row8(0, 0))
		}
	}
}

// mc8PutV6FromI32W4 is the width-4 put 2D vertical pass from the h+5 rows of
// four int32 sums at the start of im (row stride 4) into the 4 x h
// destination view dg: four output rows per chunk (then at most one chunk of
// two and one of one row). Two output rows' high halves pack into one vector
// (UZP2) that saturates to both rows. The intermediate is a Grid view checked
// once, so, as the other kernels, the pass is a leaf.
func mc8PutV6FromI32W4(im []int32, h int, ph *mc8Phase, dg unsafeutil.Grid[uint8]) {
	c0 := archsimd.LoadInt32x4Array(&ph.v[0])
	c1 := archsimd.LoadInt32x4Array(&ph.v[1])
	c2 := archsimd.LoadInt32x4Array(&ph.v[2])
	c3 := archsimd.LoadInt32x4Array(&ph.v[3])
	c4 := archsimd.LoadInt32x4Array(&ph.v[4])
	c5 := archsimd.LoadInt32x4Array(&ph.v[5])
	rnd := archsimd.LoadInt32x4Array(&mc8Vec.w15)
	if !unsafeutil.GridFit(len(im), 0, 4, 4, h+5) {
		return
	}
	m := unsafeutil.NewGrid(im, 0, 4, 4, h+5)
	if !m.Fits(0, 0, 4, 5) {
		return
	}
	hd := m.Sub(0, 0, 4, 5)
	x0 := archsimd.LoadInt32x4Array(hd.Row4(0, 0))
	x1 := archsimd.LoadInt32x4Array(hd.Row4(1, 0))
	x2 := archsimd.LoadInt32x4Array(hd.Row4(2, 0))
	x3 := archsimd.LoadInt32x4Array(hd.Row4(3, 0))
	x4 := archsimd.LoadInt32x4Array(hd.Row4(4, 0))
	r := 0
	for ; r+4 <= h; r += 4 {
		if !m.Fits(r+5, 0, 4, 4) || !dg.Fits(r, 0, 4, 4) {
			return
		}
		mw, d := m.Sub(r+5, 0, 4, 4), dg.Sub(r, 0, 4, 4)
		x5 := archsimd.LoadInt32x4Array(mw.Row4(0, 0))
		x6 := archsimd.LoadInt32x4Array(mw.Row4(1, 0))
		a, b := mc8V6I32W4(x0, x1, x2, x3, x4, x5, x6, c0, c1, c2, c3, c4, c5, rnd)
		u := a.ToBits().ReshapeToUint16s().ConcatOdd(b.ToBits().ReshapeToUint16s()).BitsToInt16().SaturateToUint8().ReshapeToUint64s().GetElem(0)
		binary.LittleEndian.PutUint32(d.Row4(0, 0)[:], uint32(u))
		binary.LittleEndian.PutUint32(d.Row4(1, 0)[:], uint32(u>>32))
		x7 := archsimd.LoadInt32x4Array(mw.Row4(2, 0))
		x8 := archsimd.LoadInt32x4Array(mw.Row4(3, 0))
		a, b = mc8V6I32W4(x2, x3, x4, x5, x6, x7, x8, c0, c1, c2, c3, c4, c5, rnd)
		u = a.ToBits().ReshapeToUint16s().ConcatOdd(b.ToBits().ReshapeToUint16s()).BitsToInt16().SaturateToUint8().ReshapeToUint64s().GetElem(0)
		binary.LittleEndian.PutUint32(d.Row4(2, 0)[:], uint32(u))
		binary.LittleEndian.PutUint32(d.Row4(3, 0)[:], uint32(u>>32))
		x0, x1, x2, x3, x4 = x4, x5, x6, x7, x8
	}
	if r+2 <= h {
		if !m.Fits(r+5, 0, 4, 2) || !dg.Fits(r, 0, 4, 2) {
			return
		}
		mw, d := m.Sub(r+5, 0, 4, 2), dg.Sub(r, 0, 4, 2)
		x5 := archsimd.LoadInt32x4Array(mw.Row4(0, 0))
		x6 := archsimd.LoadInt32x4Array(mw.Row4(1, 0))
		a, b := mc8V6I32W4(x0, x1, x2, x3, x4, x5, x6, c0, c1, c2, c3, c4, c5, rnd)
		u := a.ToBits().ReshapeToUint16s().ConcatOdd(b.ToBits().ReshapeToUint16s()).BitsToInt16().SaturateToUint8().ReshapeToUint64s().GetElem(0)
		binary.LittleEndian.PutUint32(d.Row4(0, 0)[:], uint32(u))
		binary.LittleEndian.PutUint32(d.Row4(1, 0)[:], uint32(u>>32))
		r += 2
		x0, x1, x2, x3, x4 = x2, x3, x4, x5, x6
	}
	if r < h {
		if !m.Fits(r+5, 0, 4, 1) || !dg.Fits(r, 0, 4, 1) {
			return
		}
		mw, d := m.Sub(r+5, 0, 4, 1), dg.Sub(r, 0, 4, 1)
		x5 := archsimd.LoadInt32x4Array(mw.Row4(0, 0))
		a, _ := mc8V6I32W4(x0, x1, x2, x3, x4, x5, x5, c0, c1, c2, c3, c4, c5, rnd)
		v := a.ToBits().ReshapeToUint16s().ConcatOdd(a.ToBits().ReshapeToUint16s()).BitsToInt16()
		binary.LittleEndian.PutUint32(d.Row4(0, 0)[:], uint32(v.SaturateToUint8().ReshapeToUint64s().GetElem(0)))
	}
}

// 8-bit A64 soft-mask compound blend and difference-weighted mask builder
// (dav1d mc mask / w_mask), the vector forms of BlendCompoundMaskD16 and
// BuildDiffWtdMaskD16 for 8-bit CONV_BUFs.
//
// Blend: libaom's clip(round4(((m*a + (64-m)*b) >> 6) - 6144)) equals
// (X - 392704) >> 10 with X = m*a + (64-m)*b (nested floor division). The
// products widen to uint32 (UMULL/UMULL2) with the mask pre-scaled by 64, so
// the >> 10 is read from the high half of each lane (UZP2).
//
// Mask: 38 + (round4(|a - b|) >> 4), capped at 64, is
// 38 + ((|a - b| >> 3) + 1) >> 5 by nested floor division.

// mc8BlendMaskFast blends with a full-resolution (4:4:4 / luma) or 4:2:0
// subsampled mask; width must be a multiple of 8. It returns false for shapes
// it does not cover.
func mc8BlendMaskFast(dst frame.Plane, a, b []uint16, dstX, dstY, w, h int, mask []byte, maskStride int, subX, subY bool) bool {
	if w&7 != 0 || w > maxBlockSize || h <= 0 || len(a) < w*h || len(b) < w*h || subX != subY {
		return false
	}
	off := dstY*dst.Stride + dstX
	if dstX < 0 || dstY < 0 || dstX+w > dst.Width || off+(h-1)*dst.Stride+w > len(dst.Pix) {
		return false
	}
	mw, mh := w, h
	if subX {
		mw, mh = 2*w, 2*h
	}
	// The unsubsampled path loads 16 mask bytes per eight columns.
	if maskStride < mw || (mh-1)*maskStride+mw > len(mask) || (mh-1)*maskStride+mw+8 > cap(mask) {
		return false
	}
	if (h-1)*dst.Stride+w > mc8WindowMax || (mh-1)*maskStride+mw+8 > mc8WindowMax {
		return false
	}
	mc8BlendMask(dst.Pix[off:off+(h-1)*dst.Stride+w], dst.Stride, a[:w*h], b[:w*h], w, h, mask[:(mh-1)*maskStride+mw+8], maskStride, subX)
	return true
}

func mc8BlendMask(dst []byte, dstStride int, a, b []uint16, w, h int, mask []byte, maskStride int, sub bool) {
	bias := archsimd.BroadcastUint32x4(392704 << 6)
	m64 := archsimd.BroadcastUint16x8(64)
	two := archsimd.BroadcastUint16x8(2)
	right2 := archsimd.BroadcastInt16x8(-2)
	six := archsimd.BroadcastInt16x8(6)
	ws, ds, ms := uint32(w), uint32(dstStride), uint32(maskStride)
	var ao, do, mo uint32
	for r := 0; r < h; r++ {
		for c := uint32(0); c < ws; c += 8 {
			var m archsimd.Uint16x8
			if sub {
				v0 := archsimd.LoadUint8x16Array(mc8W16(mask, mo+2*c))
				v1 := archsimd.LoadUint8x16Array(mc8W16(mask, mo+ms+2*c))
				lo := v0.ExtendLo8ToUint16().Add(v1.ExtendLo8ToUint16())
				hi := v0.HiToLo().ExtendLo8ToUint16().Add(v1.HiToLo().ExtendLo8ToUint16())
				m = lo.ConcatAddPairs(hi).Add(two).Shift(right2)
			} else {
				m = archsimd.LoadUint8x16Array(mc8W16(mask, mo+c)).ExtendLo8ToUint16()
			}
			msh := m.Shift(six)
			mi := m64.Sub(m).Shift(six)
			va := archsimd.LoadUint16x8Array(mc8WU8(a, ao+c))
			vb := archsimd.LoadUint16x8Array(mc8WU8(b, ao+c))
			lo := va.MulWidenLo(msh).Add(vb.MulWidenLo(mi)).Sub(bias)
			hi := va.HiToLo().MulWidenLo(msh.HiToLo()).Add(vb.HiToLo().MulWidenLo(mi.HiToLo())).Sub(bias)
			v := lo.ReshapeToUint16s().ConcatOdd(hi.ReshapeToUint16s()).BitsToInt16()
			binary.LittleEndian.PutUint64(mc8W8(dst, do+c)[:], v.SaturateToUint8().ReshapeToUint64s().GetElem(0))
		}
		ao, do = ao+ws, do+ds
		if sub {
			mo += 2 * ms
		} else {
			mo += ms
		}
	}
}

// mc8DiffWtdMaskFast builds the 8-bit difference-weighted mask; width must be a
// multiple of 8.
func mc8DiffWtdMaskFast(mask []byte, maskStride int, a, b []uint16, w, h int, invert bool) bool {
	if w&7 != 0 || w > maxBlockSize || h <= 0 || len(a) < w*h || len(b) < w*h ||
		maskStride < w || (h-1)*maskStride+w > len(mask) {
		return false
	}
	if (h-1)*maskStride+w > mc8WindowMax {
		return false
	}
	mc8DiffWtdMask(mask[:(h-1)*maskStride+w], maskStride, a[:w*h], b[:w*h], w, h, invert)
	return true
}

func mc8DiffWtdMask(mask []byte, maskStride int, a, b []uint16, w, h int, invert bool) {
	right3 := archsimd.BroadcastInt16x8(-3)
	right5 := archsimd.BroadcastInt16x8(-5)
	one := archsimd.BroadcastUint16x8(1)
	base := archsimd.BroadcastUint16x8(38)
	maxM := archsimd.BroadcastUint16x8(64)
	ws, ms := uint32(w), uint32(maskStride)
	var ao, mo uint32
	for r := 0; r < h; r++ {
		for c := uint32(0); c < ws; c += 8 {
			va := archsimd.LoadUint16x8Array(mc8WU8(a, ao+c))
			vb := archsimd.LoadUint16x8Array(mc8WU8(b, ao+c))
			diff := va.Max(vb).Sub(va.Min(vb))
			m := diff.Shift(right3).Add(one).Shift(right5).Add(base).Min(maxM)
			if invert {
				m = maxM.Sub(m)
			}
			binary.LittleEndian.PutUint64(mc8W8(mask, mo+c)[:], m.TruncToUint8().ReshapeToUint64s().GetElem(0))
		}
		ao, mo = ao+ws, mo+ms
	}
}

// Eight-tap 8bpc put/prep (dav1d src/arm/64/mc.S filter_fn 8tap:
// put/prep_8tap_{sharp,regular_sharp,sharp_regular,...}). The sharp table is
// the only one with nonzero outer taps; when either filtered axis is sharp both
// axes run eight taps from refX-3 / refY-3 (a six-tap kind's outer taps are
// zero, so its sum is unchanged), as dav1d's 8tap functions do. The formulas
// are the six-tap kernels' (mc8_gosimd_arm64.go) with two more taps: halved
// taps (every AV1 tap is even) in int16 for the one-pass forms, and dav1d's
// int16 2D intermediate (s + 2) >> 2 with full taps widened into int32 for the
// vertical pass. The halved sharp sums stay inside int16 for 8-bit samples
// (positive taps sum to at most 78 * 255), and the prep offset sums inside
// [0, 2^16), so the unsigned shifts read them exactly.
//
// Every access is a bounds-checked fixed-size window of a row slice.

// mc8Phase8 is one filter phase for the eight-tap window.
type mc8Phase8 struct {
	h [8][8]int16 // halved taps, broadcast
	f [8][8]int16 // full taps, broadcast
}

var mc8Phases8 [mc8KindCount][16]mc8Phase8

func init() {
	tables := [mc8KindCount]*[16][filterTaps]int16{
		mc8KindRegular:  &subpelFilters8,
		mc8KindSmooth:   &subpelFilters8Smooth,
		mc8KindSharp:    &subpelFilters8Sharp,
		mc8KindBilinear: &bilinearFilters,
		mc8KindRegular4: &subpelFilters4,
		mc8KindSmooth4:  &subpelFilters4Smooth,
	}
	for kind, tab := range tables {
		for s := range 16 {
			for i, c := range tab[s] {
				for l := range 8 {
					mc8Phases8[kind][s].h[i][l] = c / 2
					mc8Phases8[kind][s].f[i][l] = c
				}
			}
		}
	}
}

// mc8Source8 is mc8Source for the eight-tap window: three samples left/above
// and four right/below of each filtered axis, as a slice from the window's
// top-left tap plus its stride. Rows stay readable for max(w+8, 16) bytes.
func mc8Source8(ref *frame.Plane, refX, refY, w, h int, hx, hy bool, edge *[emuEdgeStride * emuEdgeRows]byte) ([]byte, int, bool) {
	x0, y0, bw, bh := refX, refY, w, h
	if hx {
		x0 -= 3
		bw += 7
	}
	if hy {
		y0 -= 3
		bh += 7
	}
	if ref.Width <= 0 || ref.Height <= 0 || ref.Stride < ref.Width {
		return nil, 0, false
	}
	if x0 >= 0 && y0 >= 0 && x0+bw <= ref.Width && y0+bh <= ref.Height {
		start := y0*ref.Stride + x0
		if start+(bh-1)*ref.Stride+max(w+8, 16) <= len(ref.Pix) {
			return ref.Pix[start:], ref.Stride, true
		}
	}
	if len(ref.Pix) < (ref.Height-1)*ref.Stride+ref.Width {
		return nil, 0, false
	}
	emuEdge8(bw, bh, *ref, x0, y0, edge[:], emuEdgeStride)
	return edge[:], emuEdgeStride, true
}

// mc8NeedsEightTaps reports whether a filtered axis uses a kind with nonzero
// outer taps (the sharp table).
func mc8NeedsEightTaps(subX, subY, kx, ky int) bool {
	return (subX != 0 && !mc8SixTap[kx]) || (subY != 0 && !mc8SixTap[ky])
}

// mc8Put8TapChecked is Put8 for blocks with an eight-tap axis (w a multiple
// of 8): Put8's destination checks, then mc8Put8Tap. Keeping it out of line
// leaves Put8's six-tap body (and its PGO call-site offsets) unchanged.
func mc8Put8TapChecked(dst *frame.Plane, ref *frame.Plane, dstX, dstY, refX, refY, w, h, subX, subY, kx, ky int, scratch *ConvolveScratch) bool {
	if dstX < 0 || dstY < 0 || dstX+w > dst.Width || dstY+h > dst.Height || dst.Stride < dst.Width {
		return false
	}
	dOff := dstY*dst.Stride + dstX
	if dOff+(h-1)*dst.Stride+w > len(dst.Pix) {
		return false
	}
	return mc8Put8Tap(dst.Pix[dOff:], dst.Stride, ref, refX, refY, w, h, subX, subY, kx, ky, scratch)
}

// mc8Put8Tap is Put8 for blocks with an eight-tap axis (w a multiple of 8).
func mc8Put8Tap(dst []byte, dstStride int, ref *frame.Plane, refX, refY, w, h, subX, subY, kx, ky int, scratch *ConvolveScratch) bool {
	src, stride, ok := mc8Source8(ref, refX, refY, w, h, subX != 0, subY != 0, &scratch.edge)
	if !ok {
		return false
	}
	switch {
	case subX != 0 && subY != 0:
		im := scratch.im[:(h+7)*w]
		mc8H8ToI16(src, stride, w, h+7, &mc8Phases8[kx][subX], im)
		mc8PutV8FromI16(im, w, h, &mc8Phases8[ky][subY], dst, dstStride)
	case subX != 0:
		mc8PutH8(src, stride, w, h, &mc8Phases8[kx][subX], dst, dstStride)
	default:
		mc8PutV8(src, stride, w, h, &mc8Phases8[ky][subY], dst, dstStride)
	}
	return true
}

// mc8Prep8Tap is mc8PrepFast for blocks with an eight-tap axis (w a multiple
// of 8).
func mc8Prep8Tap(out []uint16, ref *frame.Plane, refX, refY, w, h, subX, subY, kx, ky int, scratch *CompoundConvolveScratch) bool {
	src, stride, ok := mc8Source8(ref, refX, refY, w, h, subX != 0, subY != 0, &scratch.edge)
	if !ok {
		return false
	}
	switch {
	case subX != 0 && subY != 0:
		im := scratch.im8[:(h+7)*w]
		mc8H8ToI16(src, stride, w, h+7, &mc8Phases8[kx][subX], im)
		mc8PrepV8FromI16(im, w, h, &mc8Phases8[ky][subY], out)
	case subX != 0:
		mc8PrepH8(src, stride, w, h, &mc8Phases8[kx][subX], out)
	default:
		mc8PrepV8(src, stride, w, h, &mc8Phases8[ky][subY], out)
	}
	return true
}

// mc8H8ToI16 writes rows x w rounded horizontal sums ((s + 2) >> 2) as int16 with row stride w.
func mc8H8ToI16(src []byte, stride, w, rows int, ph *mc8Phase8, im []int16) {
	c0 := archsimd.LoadInt16x8Array(&ph.h[0])
	c1 := archsimd.LoadInt16x8Array(&ph.h[1])
	c2 := archsimd.LoadInt16x8Array(&ph.h[2])
	c3 := archsimd.LoadInt16x8Array(&ph.h[3])
	c4 := archsimd.LoadInt16x8Array(&ph.h[4])
	c5 := archsimd.LoadInt16x8Array(&ph.h[5])
	c6 := archsimd.LoadInt16x8Array(&ph.h[6])
	c7 := archsimd.LoadInt16x8Array(&ph.h[7])
	rnd := archsimd.BroadcastInt16x8(2)
	sh := archsimd.BroadcastInt16x8(-2)
	r := 0
	for ; r+2 <= rows; r += 2 {
		row0 := src[r*stride : r*stride+w+8]
		row1 := src[(r+1)*stride : (r+1)*stride+w+8]
		for c := 0; c+8 <= w; c += 8 {
			a0 := archsimd.LoadUint8x16Array((*[16]uint8)(row0[c : c+16]))
			lo0, hi0 := a0.ExtendLo8ToUint16().ReshapeToUint8s(), a0.HiToLo().ExtendLo8ToUint16().ReshapeToUint8s()
			s0 := mc8I16(lo0).MulAdd(c0, rnd)
			s0 = mc8I16(hi0.ConcatShiftBytesRight(lo0, 2)).MulAdd(c1, s0)
			s0 = mc8I16(hi0.ConcatShiftBytesRight(lo0, 4)).MulAdd(c2, s0)
			s0 = mc8I16(hi0.ConcatShiftBytesRight(lo0, 6)).MulAdd(c3, s0)
			s0 = mc8I16(hi0.ConcatShiftBytesRight(lo0, 8)).MulAdd(c4, s0)
			s0 = mc8I16(hi0.ConcatShiftBytesRight(lo0, 10)).MulAdd(c5, s0)
			s0 = mc8I16(hi0.ConcatShiftBytesRight(lo0, 12)).MulAdd(c6, s0)
			s0 = mc8I16(hi0.ConcatShiftBytesRight(lo0, 14)).MulAdd(c7, s0)
			a1 := archsimd.LoadUint8x16Array((*[16]uint8)(row1[c : c+16]))
			lo1, hi1 := a1.ExtendLo8ToUint16().ReshapeToUint8s(), a1.HiToLo().ExtendLo8ToUint16().ReshapeToUint8s()
			s1 := mc8I16(lo1).MulAdd(c0, rnd)
			s1 = mc8I16(hi1.ConcatShiftBytesRight(lo1, 2)).MulAdd(c1, s1)
			s1 = mc8I16(hi1.ConcatShiftBytesRight(lo1, 4)).MulAdd(c2, s1)
			s1 = mc8I16(hi1.ConcatShiftBytesRight(lo1, 6)).MulAdd(c3, s1)
			s1 = mc8I16(hi1.ConcatShiftBytesRight(lo1, 8)).MulAdd(c4, s1)
			s1 = mc8I16(hi1.ConcatShiftBytesRight(lo1, 10)).MulAdd(c5, s1)
			s1 = mc8I16(hi1.ConcatShiftBytesRight(lo1, 12)).MulAdd(c6, s1)
			s1 = mc8I16(hi1.ConcatShiftBytesRight(lo1, 14)).MulAdd(c7, s1)
			s0.Shift(sh).StoreArray((*[8]int16)(im[(r)*w+c : (r)*w+c+8]))
			s1.Shift(sh).StoreArray((*[8]int16)(im[(r+1)*w+c : (r+1)*w+c+8]))
		}
	}
	if r < rows {
		row0 := src[r*stride : r*stride+w+8]
		for c := 0; c+8 <= w; c += 8 {
			a0 := archsimd.LoadUint8x16Array((*[16]uint8)(row0[c : c+16]))
			lo0, hi0 := a0.ExtendLo8ToUint16().ReshapeToUint8s(), a0.HiToLo().ExtendLo8ToUint16().ReshapeToUint8s()
			s0 := mc8I16(lo0).MulAdd(c0, rnd)
			s0 = mc8I16(hi0.ConcatShiftBytesRight(lo0, 2)).MulAdd(c1, s0)
			s0 = mc8I16(hi0.ConcatShiftBytesRight(lo0, 4)).MulAdd(c2, s0)
			s0 = mc8I16(hi0.ConcatShiftBytesRight(lo0, 6)).MulAdd(c3, s0)
			s0 = mc8I16(hi0.ConcatShiftBytesRight(lo0, 8)).MulAdd(c4, s0)
			s0 = mc8I16(hi0.ConcatShiftBytesRight(lo0, 10)).MulAdd(c5, s0)
			s0 = mc8I16(hi0.ConcatShiftBytesRight(lo0, 12)).MulAdd(c6, s0)
			s0 = mc8I16(hi0.ConcatShiftBytesRight(lo0, 14)).MulAdd(c7, s0)
			s0.Shift(sh).StoreArray((*[8]int16)(im[(r)*w+c : (r)*w+c+8]))
		}
	}
}

// mc8PutH8 is the put horizontal-only pass: clip((s + 34) >> 6).
func mc8PutH8(src []byte, stride, w, rows int, ph *mc8Phase8, dst []byte, dstStride int) {
	c0 := archsimd.LoadInt16x8Array(&ph.h[0])
	c1 := archsimd.LoadInt16x8Array(&ph.h[1])
	c2 := archsimd.LoadInt16x8Array(&ph.h[2])
	c3 := archsimd.LoadInt16x8Array(&ph.h[3])
	c4 := archsimd.LoadInt16x8Array(&ph.h[4])
	c5 := archsimd.LoadInt16x8Array(&ph.h[5])
	c6 := archsimd.LoadInt16x8Array(&ph.h[6])
	c7 := archsimd.LoadInt16x8Array(&ph.h[7])
	rnd := archsimd.BroadcastInt16x8(34)
	sh6 := archsimd.BroadcastInt16x8(-6)
	r := 0
	for ; r+2 <= rows; r += 2 {
		row0 := src[r*stride : r*stride+w+8]
		row1 := src[(r+1)*stride : (r+1)*stride+w+8]
		for c := 0; c+8 <= w; c += 8 {
			a0 := archsimd.LoadUint8x16Array((*[16]uint8)(row0[c : c+16]))
			lo0, hi0 := a0.ExtendLo8ToUint16().ReshapeToUint8s(), a0.HiToLo().ExtendLo8ToUint16().ReshapeToUint8s()
			s0 := mc8I16(lo0).MulAdd(c0, rnd)
			s0 = mc8I16(hi0.ConcatShiftBytesRight(lo0, 2)).MulAdd(c1, s0)
			s0 = mc8I16(hi0.ConcatShiftBytesRight(lo0, 4)).MulAdd(c2, s0)
			s0 = mc8I16(hi0.ConcatShiftBytesRight(lo0, 6)).MulAdd(c3, s0)
			s0 = mc8I16(hi0.ConcatShiftBytesRight(lo0, 8)).MulAdd(c4, s0)
			s0 = mc8I16(hi0.ConcatShiftBytesRight(lo0, 10)).MulAdd(c5, s0)
			s0 = mc8I16(hi0.ConcatShiftBytesRight(lo0, 12)).MulAdd(c6, s0)
			s0 = mc8I16(hi0.ConcatShiftBytesRight(lo0, 14)).MulAdd(c7, s0)
			a1 := archsimd.LoadUint8x16Array((*[16]uint8)(row1[c : c+16]))
			lo1, hi1 := a1.ExtendLo8ToUint16().ReshapeToUint8s(), a1.HiToLo().ExtendLo8ToUint16().ReshapeToUint8s()
			s1 := mc8I16(lo1).MulAdd(c0, rnd)
			s1 = mc8I16(hi1.ConcatShiftBytesRight(lo1, 2)).MulAdd(c1, s1)
			s1 = mc8I16(hi1.ConcatShiftBytesRight(lo1, 4)).MulAdd(c2, s1)
			s1 = mc8I16(hi1.ConcatShiftBytesRight(lo1, 6)).MulAdd(c3, s1)
			s1 = mc8I16(hi1.ConcatShiftBytesRight(lo1, 8)).MulAdd(c4, s1)
			s1 = mc8I16(hi1.ConcatShiftBytesRight(lo1, 10)).MulAdd(c5, s1)
			s1 = mc8I16(hi1.ConcatShiftBytesRight(lo1, 12)).MulAdd(c6, s1)
			s1 = mc8I16(hi1.ConcatShiftBytesRight(lo1, 14)).MulAdd(c7, s1)
			s0.Shift(sh6).SaturateToUint8().StorePart(dst[(r)*dstStride+c : (r)*dstStride+c+8])
			s1.Shift(sh6).SaturateToUint8().StorePart(dst[(r+1)*dstStride+c : (r+1)*dstStride+c+8])
		}
	}
	if r < rows {
		row0 := src[r*stride : r*stride+w+8]
		for c := 0; c+8 <= w; c += 8 {
			a0 := archsimd.LoadUint8x16Array((*[16]uint8)(row0[c : c+16]))
			lo0, hi0 := a0.ExtendLo8ToUint16().ReshapeToUint8s(), a0.HiToLo().ExtendLo8ToUint16().ReshapeToUint8s()
			s0 := mc8I16(lo0).MulAdd(c0, rnd)
			s0 = mc8I16(hi0.ConcatShiftBytesRight(lo0, 2)).MulAdd(c1, s0)
			s0 = mc8I16(hi0.ConcatShiftBytesRight(lo0, 4)).MulAdd(c2, s0)
			s0 = mc8I16(hi0.ConcatShiftBytesRight(lo0, 6)).MulAdd(c3, s0)
			s0 = mc8I16(hi0.ConcatShiftBytesRight(lo0, 8)).MulAdd(c4, s0)
			s0 = mc8I16(hi0.ConcatShiftBytesRight(lo0, 10)).MulAdd(c5, s0)
			s0 = mc8I16(hi0.ConcatShiftBytesRight(lo0, 12)).MulAdd(c6, s0)
			s0 = mc8I16(hi0.ConcatShiftBytesRight(lo0, 14)).MulAdd(c7, s0)
			s0.Shift(sh6).SaturateToUint8().StorePart(dst[(r)*dstStride+c : (r)*dstStride+c+8])
		}
	}
}

// mc8PrepH8 is the prep horizontal-only pass: ((s + 2) >> 2) + 6144, the offset sums shifted as unsigned.
func mc8PrepH8(src []byte, stride, w, rows int, ph *mc8Phase8, out []uint16) {
	c0 := archsimd.LoadInt16x8Array(&ph.h[0])
	c1 := archsimd.LoadInt16x8Array(&ph.h[1])
	c2 := archsimd.LoadInt16x8Array(&ph.h[2])
	c3 := archsimd.LoadInt16x8Array(&ph.h[3])
	c4 := archsimd.LoadInt16x8Array(&ph.h[4])
	c5 := archsimd.LoadInt16x8Array(&ph.h[5])
	c6 := archsimd.LoadInt16x8Array(&ph.h[6])
	c7 := archsimd.LoadInt16x8Array(&ph.h[7])
	rnd := archsimd.BroadcastInt16x8(2 + 6144<<2)
	sh := archsimd.BroadcastInt16x8(-2)
	r := 0
	for ; r+2 <= rows; r += 2 {
		row0 := src[r*stride : r*stride+w+8]
		row1 := src[(r+1)*stride : (r+1)*stride+w+8]
		for c := 0; c+8 <= w; c += 8 {
			a0 := archsimd.LoadUint8x16Array((*[16]uint8)(row0[c : c+16]))
			lo0, hi0 := a0.ExtendLo8ToUint16().ReshapeToUint8s(), a0.HiToLo().ExtendLo8ToUint16().ReshapeToUint8s()
			s0 := mc8I16(lo0).MulAdd(c0, rnd)
			s0 = mc8I16(hi0.ConcatShiftBytesRight(lo0, 2)).MulAdd(c1, s0)
			s0 = mc8I16(hi0.ConcatShiftBytesRight(lo0, 4)).MulAdd(c2, s0)
			s0 = mc8I16(hi0.ConcatShiftBytesRight(lo0, 6)).MulAdd(c3, s0)
			s0 = mc8I16(hi0.ConcatShiftBytesRight(lo0, 8)).MulAdd(c4, s0)
			s0 = mc8I16(hi0.ConcatShiftBytesRight(lo0, 10)).MulAdd(c5, s0)
			s0 = mc8I16(hi0.ConcatShiftBytesRight(lo0, 12)).MulAdd(c6, s0)
			s0 = mc8I16(hi0.ConcatShiftBytesRight(lo0, 14)).MulAdd(c7, s0)
			a1 := archsimd.LoadUint8x16Array((*[16]uint8)(row1[c : c+16]))
			lo1, hi1 := a1.ExtendLo8ToUint16().ReshapeToUint8s(), a1.HiToLo().ExtendLo8ToUint16().ReshapeToUint8s()
			s1 := mc8I16(lo1).MulAdd(c0, rnd)
			s1 = mc8I16(hi1.ConcatShiftBytesRight(lo1, 2)).MulAdd(c1, s1)
			s1 = mc8I16(hi1.ConcatShiftBytesRight(lo1, 4)).MulAdd(c2, s1)
			s1 = mc8I16(hi1.ConcatShiftBytesRight(lo1, 6)).MulAdd(c3, s1)
			s1 = mc8I16(hi1.ConcatShiftBytesRight(lo1, 8)).MulAdd(c4, s1)
			s1 = mc8I16(hi1.ConcatShiftBytesRight(lo1, 10)).MulAdd(c5, s1)
			s1 = mc8I16(hi1.ConcatShiftBytesRight(lo1, 12)).MulAdd(c6, s1)
			s1 = mc8I16(hi1.ConcatShiftBytesRight(lo1, 14)).MulAdd(c7, s1)
			s0.ToBits().Shift(sh).StoreArray((*[8]uint16)(out[(r)*w+c : (r)*w+c+8]))
			s1.ToBits().Shift(sh).StoreArray((*[8]uint16)(out[(r+1)*w+c : (r+1)*w+c+8]))
		}
	}
	if r < rows {
		row0 := src[r*stride : r*stride+w+8]
		for c := 0; c+8 <= w; c += 8 {
			a0 := archsimd.LoadUint8x16Array((*[16]uint8)(row0[c : c+16]))
			lo0, hi0 := a0.ExtendLo8ToUint16().ReshapeToUint8s(), a0.HiToLo().ExtendLo8ToUint16().ReshapeToUint8s()
			s0 := mc8I16(lo0).MulAdd(c0, rnd)
			s0 = mc8I16(hi0.ConcatShiftBytesRight(lo0, 2)).MulAdd(c1, s0)
			s0 = mc8I16(hi0.ConcatShiftBytesRight(lo0, 4)).MulAdd(c2, s0)
			s0 = mc8I16(hi0.ConcatShiftBytesRight(lo0, 6)).MulAdd(c3, s0)
			s0 = mc8I16(hi0.ConcatShiftBytesRight(lo0, 8)).MulAdd(c4, s0)
			s0 = mc8I16(hi0.ConcatShiftBytesRight(lo0, 10)).MulAdd(c5, s0)
			s0 = mc8I16(hi0.ConcatShiftBytesRight(lo0, 12)).MulAdd(c6, s0)
			s0 = mc8I16(hi0.ConcatShiftBytesRight(lo0, 14)).MulAdd(c7, s0)
			s0.ToBits().Shift(sh).StoreArray((*[8]uint16)(out[(r)*w+c : (r)*w+c+8]))
		}
	}
}

// mc8PutV8 is the put vertical-only pass: clip((s + 32) >> 6).
func mc8PutV8(src []byte, stride, w, h int, ph *mc8Phase8, dst []byte, dstStride int) {
	c0 := archsimd.LoadInt16x8Array(&ph.h[0])
	c1 := archsimd.LoadInt16x8Array(&ph.h[1])
	c2 := archsimd.LoadInt16x8Array(&ph.h[2])
	c3 := archsimd.LoadInt16x8Array(&ph.h[3])
	c4 := archsimd.LoadInt16x8Array(&ph.h[4])
	c5 := archsimd.LoadInt16x8Array(&ph.h[5])
	c6 := archsimd.LoadInt16x8Array(&ph.h[6])
	c7 := archsimd.LoadInt16x8Array(&ph.h[7])
	rnd := archsimd.BroadcastInt16x8(32)
	sh := archsimd.BroadcastInt16x8(-6)
	for c := 0; c+8 <= w; c += 8 {
		load := func(r int) archsimd.Int16x8 {
			return archsimd.LoadUint8x16Array((*[16]uint8)(src[r*stride+c : r*stride+c+16])).ExtendLo8ToUint16().BitsToInt16()
		}
		r0, r1, r2, r3, r4, r5, r6 := load(0), load(1), load(2), load(3), load(4), load(5), load(6)
		r := 0
		for ; r+2 <= h; r += 2 {
			r7, r8 := load(r+7), load(r+8)
			a := r0.MulAdd(c0, rnd)
			b := r1.MulAdd(c0, rnd)
			a = r1.MulAdd(c1, a)
			b = r2.MulAdd(c1, b)
			a = r2.MulAdd(c2, a)
			b = r3.MulAdd(c2, b)
			a = r3.MulAdd(c3, a)
			b = r4.MulAdd(c3, b)
			a = r4.MulAdd(c4, a)
			b = r5.MulAdd(c4, b)
			a = r5.MulAdd(c5, a)
			b = r6.MulAdd(c5, b)
			a = r6.MulAdd(c6, a)
			b = r7.MulAdd(c6, b)
			a = r7.MulAdd(c7, a)
			b = r8.MulAdd(c7, b)
			a.Shift(sh).SaturateToUint8().StorePart(dst[(r)*dstStride+c : (r)*dstStride+c+8])
			b.Shift(sh).SaturateToUint8().StorePart(dst[(r+1)*dstStride+c : (r+1)*dstStride+c+8])
			r0, r1, r2, r3, r4, r5, r6 = r2, r3, r4, r5, r6, r7, r8
		}
		if r < h {
			r7 := load(r + 7)
			a := r0.MulAdd(c0, rnd)
			a = r1.MulAdd(c1, a)
			a = r2.MulAdd(c2, a)
			a = r3.MulAdd(c3, a)
			a = r4.MulAdd(c4, a)
			a = r5.MulAdd(c5, a)
			a = r6.MulAdd(c6, a)
			a = r7.MulAdd(c7, a)
			a.Shift(sh).SaturateToUint8().StorePart(dst[(r)*dstStride+c : (r)*dstStride+c+8])
		}
	}
}

// mc8PrepV8 is the prep vertical-only pass: ((s + 2) >> 2) + 6144.
func mc8PrepV8(src []byte, stride, w, h int, ph *mc8Phase8, out []uint16) {
	c0 := archsimd.LoadInt16x8Array(&ph.h[0])
	c1 := archsimd.LoadInt16x8Array(&ph.h[1])
	c2 := archsimd.LoadInt16x8Array(&ph.h[2])
	c3 := archsimd.LoadInt16x8Array(&ph.h[3])
	c4 := archsimd.LoadInt16x8Array(&ph.h[4])
	c5 := archsimd.LoadInt16x8Array(&ph.h[5])
	c6 := archsimd.LoadInt16x8Array(&ph.h[6])
	c7 := archsimd.LoadInt16x8Array(&ph.h[7])
	rnd := archsimd.BroadcastInt16x8(2 + 6144<<2)
	sh := archsimd.BroadcastInt16x8(-2)
	for c := 0; c+8 <= w; c += 8 {
		load := func(r int) archsimd.Int16x8 {
			return archsimd.LoadUint8x16Array((*[16]uint8)(src[r*stride+c : r*stride+c+16])).ExtendLo8ToUint16().BitsToInt16()
		}
		r0, r1, r2, r3, r4, r5, r6 := load(0), load(1), load(2), load(3), load(4), load(5), load(6)
		r := 0
		for ; r+2 <= h; r += 2 {
			r7, r8 := load(r+7), load(r+8)
			a := r0.MulAdd(c0, rnd)
			b := r1.MulAdd(c0, rnd)
			a = r1.MulAdd(c1, a)
			b = r2.MulAdd(c1, b)
			a = r2.MulAdd(c2, a)
			b = r3.MulAdd(c2, b)
			a = r3.MulAdd(c3, a)
			b = r4.MulAdd(c3, b)
			a = r4.MulAdd(c4, a)
			b = r5.MulAdd(c4, b)
			a = r5.MulAdd(c5, a)
			b = r6.MulAdd(c5, b)
			a = r6.MulAdd(c6, a)
			b = r7.MulAdd(c6, b)
			a = r7.MulAdd(c7, a)
			b = r8.MulAdd(c7, b)
			a.ToBits().Shift(sh).StoreArray((*[8]uint16)(out[(r)*w+c : (r)*w+c+8]))
			b.ToBits().Shift(sh).StoreArray((*[8]uint16)(out[(r+1)*w+c : (r+1)*w+c+8]))
			r0, r1, r2, r3, r4, r5, r6 = r2, r3, r4, r5, r6, r7, r8
		}
		if r < h {
			r7 := load(r + 7)
			a := r0.MulAdd(c0, rnd)
			a = r1.MulAdd(c1, a)
			a = r2.MulAdd(c2, a)
			a = r3.MulAdd(c3, a)
			a = r4.MulAdd(c4, a)
			a = r5.MulAdd(c5, a)
			a = r6.MulAdd(c6, a)
			a = r7.MulAdd(c7, a)
			a.ToBits().Shift(sh).StoreArray((*[8]uint16)(out[(r)*w+c : (r)*w+c+8]))
		}
	}
}

// mc8PutV8FromI16 is the put 2D vertical pass: clip((sum(tap * m) + 2^10) >> 11).
func mc8PutV8FromI16(im []int16, w, h int, ph *mc8Phase8, dst []byte, dstStride int) {
	f0 := archsimd.LoadInt16x8Array(&ph.f[0])
	f1 := archsimd.LoadInt16x8Array(&ph.f[1])
	f2 := archsimd.LoadInt16x8Array(&ph.f[2])
	f3 := archsimd.LoadInt16x8Array(&ph.f[3])
	f4 := archsimd.LoadInt16x8Array(&ph.f[4])
	f5 := archsimd.LoadInt16x8Array(&ph.f[5])
	f6 := archsimd.LoadInt16x8Array(&ph.f[6])
	f7 := archsimd.LoadInt16x8Array(&ph.f[7])
	bias := archsimd.BroadcastInt32x4(1 << 10)
	shv := archsimd.BroadcastInt32x4(-11)
	for c := 0; c+8 <= w; c += 8 {
		load := func(r int) archsimd.Int16x8 {
			return archsimd.LoadInt16x8Array((*[8]int16)(im[r*w+c : r*w+c+8]))
		}
		m0, m1, m2, m3, m4, m5, m6 := load(0), load(1), load(2), load(3), load(4), load(5), load(6)
		for r := 0; r < h; r++ {
			m7 := load(r + 7)
			lo := bias.Add(m0.MulWidenLo(f0)).Add(m1.MulWidenLo(f1)).Add(m2.MulWidenLo(f2)).Add(m3.MulWidenLo(f3)).Add(m4.MulWidenLo(f4)).Add(m5.MulWidenLo(f5)).Add(m6.MulWidenLo(f6)).Add(m7.MulWidenLo(f7))
			hi := bias.Add(m0.HiToLo().MulWidenLo(f0.HiToLo())).Add(m1.HiToLo().MulWidenLo(f1.HiToLo())).Add(m2.HiToLo().MulWidenLo(f2.HiToLo())).Add(m3.HiToLo().MulWidenLo(f3.HiToLo())).Add(m4.HiToLo().MulWidenLo(f4.HiToLo())).Add(m5.HiToLo().MulWidenLo(f5.HiToLo())).Add(m6.HiToLo().MulWidenLo(f6.HiToLo())).Add(m7.HiToLo().MulWidenLo(f7.HiToLo()))
			u8PackSaturate32(lo.Shift(shv), hi.Shift(shv)).StorePart(dst[r*dstStride+c : r*dstStride+c+8])
			m0, m1, m2, m3, m4, m5, m6 = m1, m2, m3, m4, m5, m6, m7
		}
	}
}

// mc8PrepV8FromI16 is the prep 2D vertical pass: ((sum(tap * m) + 64) >> 7) + 6144.
func mc8PrepV8FromI16(im []int16, w, h int, ph *mc8Phase8, out []uint16) {
	f0 := archsimd.LoadInt16x8Array(&ph.f[0])
	f1 := archsimd.LoadInt16x8Array(&ph.f[1])
	f2 := archsimd.LoadInt16x8Array(&ph.f[2])
	f3 := archsimd.LoadInt16x8Array(&ph.f[3])
	f4 := archsimd.LoadInt16x8Array(&ph.f[4])
	f5 := archsimd.LoadInt16x8Array(&ph.f[5])
	f6 := archsimd.LoadInt16x8Array(&ph.f[6])
	f7 := archsimd.LoadInt16x8Array(&ph.f[7])
	bias := archsimd.BroadcastInt32x4(64 + 6144<<7)
	shv := archsimd.BroadcastInt32x4(-7)
	for c := 0; c+8 <= w; c += 8 {
		load := func(r int) archsimd.Int16x8 {
			return archsimd.LoadInt16x8Array((*[8]int16)(im[r*w+c : r*w+c+8]))
		}
		m0, m1, m2, m3, m4, m5, m6 := load(0), load(1), load(2), load(3), load(4), load(5), load(6)
		for r := 0; r < h; r++ {
			m7 := load(r + 7)
			lo := bias.Add(m0.MulWidenLo(f0)).Add(m1.MulWidenLo(f1)).Add(m2.MulWidenLo(f2)).Add(m3.MulWidenLo(f3)).Add(m4.MulWidenLo(f4)).Add(m5.MulWidenLo(f5)).Add(m6.MulWidenLo(f6)).Add(m7.MulWidenLo(f7))
			hi := bias.Add(m0.HiToLo().MulWidenLo(f0.HiToLo())).Add(m1.HiToLo().MulWidenLo(f1.HiToLo())).Add(m2.HiToLo().MulWidenLo(f2.HiToLo())).Add(m3.HiToLo().MulWidenLo(f3.HiToLo())).Add(m4.HiToLo().MulWidenLo(f4.HiToLo())).Add(m5.HiToLo().MulWidenLo(f5.HiToLo())).Add(m6.HiToLo().MulWidenLo(f6.HiToLo())).Add(m7.HiToLo().MulWidenLo(f7.HiToLo()))
			u8PackU16From32(lo.Shift(shv), hi.Shift(shv)).StoreArray((*[8]uint16)(out[r*w+c : r*w+c+8]))
			m0, m1, m2, m3, m4, m5, m6 = m1, m2, m3, m4, m5, m6, m7
		}
	}
}

// Width-4 variants of the dav1d-style 8-bit put/prep kernels (dav1d mc.S
// "40:" paths). Width 4 selects the 4-tap (or bilinear) tables, which sit
// inside the same six-tap window, so the arithmetic is identical to the
// eight-wide kernels: the horizontal sums are formed on eight lanes of which
// the low four are kept, and the vertical passes pack two output rows into one
// vector (rows of a width-4 block are contiguous in the int32 intermediate and
// in the CONV_BUF). Source rows are 16-byte windows (mc8Source keeps 16 bytes
// readable per row).

// mc8V6I32W4 is the shared vertical tap chain of the width-4 2D passes: rows
// r (a) and r+1 (b) from the six-row window x0..x6.
func mc8V6I32W4(x0, x1, x2, x3, x4, x5, x6, c0, c1, c2, c3, c4, c5, rnd archsimd.Int32x4) (archsimd.Int32x4, archsimd.Int32x4) {
	a := x0.MulAdd(c0, rnd)
	b := x1.MulAdd(c0, rnd)
	a = x1.MulAdd(c1, a)
	b = x2.MulAdd(c1, b)
	a = x2.MulAdd(c2, a)
	b = x3.MulAdd(c2, b)
	a = x3.MulAdd(c3, a)
	b = x4.MulAdd(c3, b)
	a = x4.MulAdd(c4, a)
	b = x5.MulAdd(c4, b)
	a = x5.MulAdd(c5, a)
	b = x6.MulAdd(c5, b)
	return a, b
}

// mc8Store4U16 stores the four uint16 lanes packed in x.
func mc8Store4U16(o *[4]uint16, x uint64) {
	o[0], o[1], o[2], o[3] = uint16(x), uint16(x>>16), uint16(x>>32), uint16(x>>48)
}

// mc8PrepV6FromI32W4 is the width-4 prep 2D vertical pass; two CONV_BUF rows
// are one 16-byte store.
func mc8PrepV6FromI32W4(im []int32, h int, ph *mc8Phase, out []uint16, rndConst int32) {
	c0 := archsimd.LoadInt32x4Array(&ph.p[0])
	c1 := archsimd.LoadInt32x4Array(&ph.p[1])
	c2 := archsimd.LoadInt32x4Array(&ph.p[2])
	c3 := archsimd.LoadInt32x4Array(&ph.p[3])
	c4 := archsimd.LoadInt32x4Array(&ph.p[4])
	c5 := archsimd.LoadInt32x4Array(&ph.p[5])
	rnd := archsimd.BroadcastInt32x4(rndConst)
	st := im[: (h+5)*4 : (h+5)*4]
	head := (*[20]int32)(st[0:20])
	x0 := archsimd.LoadInt32x4Array((*[4]int32)(head[0:4]))
	x1 := archsimd.LoadInt32x4Array((*[4]int32)(head[4:8]))
	x2 := archsimd.LoadInt32x4Array((*[4]int32)(head[8:12]))
	x3 := archsimd.LoadInt32x4Array((*[4]int32)(head[12:16]))
	x4 := archsimd.LoadInt32x4Array((*[4]int32)(head[16:20]))
	ot := out[: h*4 : h*4]
	q := 20
	for ; q <= len(st)-8; q += 8 {
		win := (*[8]int32)(st[q : q+8])
		x5 := archsimd.LoadInt32x4Array((*[4]int32)(win[0:4]))
		x6 := archsimd.LoadInt32x4Array((*[4]int32)(win[4:8]))
		a, b := mc8V6I32W4(x0, x1, x2, x3, x4, x5, x6, c0, c1, c2, c3, c4, c5, rnd)
		a.ToBits().ReshapeToUint16s().ConcatOdd(b.ToBits().ReshapeToUint16s()).StoreArray((*[8]uint16)(ot[q-20 : q-12]))
		x0, x1, x2, x3, x4 = x2, x3, x4, x5, x6
	}
	if q <= len(st)-4 {
		x5 := archsimd.LoadInt32x4Array((*[4]int32)(st[q : q+4]))
		a, _ := mc8V6I32W4(x0, x1, x2, x3, x4, x5, x5, c0, c1, c2, c3, c4, c5, rnd)
		v := a.ToBits().ReshapeToUint16s().ConcatOdd(a.ToBits().ReshapeToUint16s())
		mc8Store4U16((*[4]uint16)(ot[q-20:q-16]), v.ReshapeToUint64s().GetElem(0))
	}
}

// 8-bit compound warp (dav1d warp_affine_8x8t, src/arm/64/mc.S). The
// horizontal pass writes the 15x8 intermediate as int16 rows, and the vertical
// pass reads those rows straight into its widening multiplies, so the int32
// warpTmp round trip of the non-compound path is gone. The results equal
// PredictWarpedCompoundToConvBuf's scalar formulas sample for sample.

// warpCompoundMid is the 15x8 int16 intermediate without the +4096 offset of
// the scalar warpTmp (see warpCompoundV8 for where it goes).
type warpCompoundMid = [15 * 8]int16

// warpCompound8x8Dispatch predicts one full 8x8 compound warp block into out
// (row r at out[r*stride:]). It reports false when the SIMD kernel does not
// apply; the caller then runs the scalar block.
func warpCompound8x8Dispatch(out []uint16, stride int, ref frame.Plane, ix4, sx4, iy4, sy4, alpha, beta, gamma, delta int) bool {
	if !warpGoSIMDEnabled || !warpHorizResidentOffsInRange(sx4, alpha, beta) || !warpVertFullOffsInRange(sy4, gamma, delta) ||
		!unsafeutil.GridFit(len(out), 0, stride, 8, 8) {
		return false
	}
	var mid warpCompoundMid
	// The resident window also needs its sixteenth byte on the last row
	// inside ref.Pix (it is loaded, never used); otherwise go through the edge
	// copy, which reads the same samples.
	base := (iy4-7)*ref.Stride + ix4 - 7
	if iy4 >= 7 && iy4+7 < ref.Height && ix4 >= 7 && ix4+8 <= ref.Width && unsafeutil.GridFit(len(ref.Pix), base, ref.Stride, 16, 15) {
		warpCompoundH8(&mid, &warp8FilterPadded, unsafeutil.NewGrid(ref.Pix, base, ref.Stride, 16, 15), sx4, alpha, beta)
	} else {
		var edge [15 * 16]byte
		warpCompoundEmuEdge(&edge, ref, ix4, iy4)
		warpCompoundH8(&mid, &warp8FilterPadded, unsafeutil.NewGrid(edge[:], 0, 16, 16, 15), sx4, alpha, beta)
	}
	warpCompoundV8(unsafeutil.NewGrid(out, 0, stride, 8, 8), &mid, &warp8FilterPadded, sy4, gamma, delta)
	return true
}

// warpCompoundEmuEdge copies the 15x16 source window at (ix4-7, iy4-7) with
// the coordinates clamped to the plane (dav1d emu_edge), which is exactly the
// sample the scalar edge path reads for every tap.
func warpCompoundEmuEdge(edge *[15 * 16]byte, ref frame.Plane, ix4, iy4 int) {
	x0 := ix4 - 7
	inside := x0 >= 0 && x0+16 <= ref.Width
	for r := range 15 {
		row := ref.Pix[clampInt(iy4-7+r, 0, ref.Height-1)*ref.Stride:]
		dst := edge[r*16 : r*16+16]
		if inside {
			copy(dst, row[x0:x0+16])
			continue
		}
		for x := range dst {
			dst[x] = row[clampInt(x0+x, 0, ref.Width-1)]
		}
	}
}

// warpCompoundH8 computes mid[k] = (sum(s*c) + 4) >> 3 for the 15 source rows
// in sg with the samples centred by -128. Every warp filter row sums to 128, so
// the centring removes 2^14 from the sum and the scalar's 2^14 offset cancels
// it; the scalar's +4096 after the shift is added back in the vertical
// accumulator.
func warpCompoundH8(mid *warpCompoundMid, filters *[256][16]int8, sg unsafeutil.Grid[uint8], sx4, alpha, beta int) {
	if !sg.Fits(0, 0, 16, 15) {
		return
	}
	centre := archsimd.BroadcastUint8x16(128)
	round := archsimd.BroadcastInt16x8(4)
	right := archsimd.BroadcastInt16x8(-3)
	// The filter row is 64 + round2(sx, 10); the 64 rides in the phase.
	sx := sx4 - 3*beta + 64<<10 + 512
	if alpha == 0 {
		for k := range 15 {
			a := archsimd.LoadUint8x16Array(sg.Row16(k, 0)).Xor(centre)
			f := archsimd.LoadInt8x16Array(&filters[(sx>>10)&255])
			p0 := a.BitsToInt8().MulWidenLo(f)
			p1 := a.ConcatShiftBytesRight(a, 1).BitsToInt8().MulWidenLo(f)
			p2 := a.ConcatShiftBytesRight(a, 2).BitsToInt8().MulWidenLo(f)
			p3 := a.ConcatShiftBytesRight(a, 3).BitsToInt8().MulWidenLo(f)
			p4 := a.ConcatShiftBytesRight(a, 4).BitsToInt8().MulWidenLo(f)
			p5 := a.ConcatShiftBytesRight(a, 5).BitsToInt8().MulWidenLo(f)
			p6 := a.ConcatShiftBytesRight(a, 6).BitsToInt8().MulWidenLo(f)
			p7 := a.ConcatShiftBytesRight(a, 7).BitsToInt8().MulWidenLo(f)
			low := p0.ConcatAddPairs(p1).ConcatAddPairs(p2.ConcatAddPairs(p3))
			high := p4.ConcatAddPairs(p5).ConcatAddPairs(p6.ConcatAddPairs(p7))
			low.ConcatAddPairs(high).Add(round).Shift(right).StoreArray((*[8]int16)(mid[8*k : 8*k+8]))
			sx += beta
		}
		return
	}
	for k := range 15 {
		a := archsimd.LoadUint8x16Array(sg.Row16(k, 0)).Xor(centre)
		f0 := archsimd.LoadInt8x16Array(&filters[(sx>>10)&255])
		f1 := archsimd.LoadInt8x16Array(&filters[((sx+alpha)>>10)&255])
		f2 := archsimd.LoadInt8x16Array(&filters[((sx+2*alpha)>>10)&255])
		f3 := archsimd.LoadInt8x16Array(&filters[((sx+3*alpha)>>10)&255])
		f4 := archsimd.LoadInt8x16Array(&filters[((sx+4*alpha)>>10)&255])
		f5 := archsimd.LoadInt8x16Array(&filters[((sx+5*alpha)>>10)&255])
		f6 := archsimd.LoadInt8x16Array(&filters[((sx+6*alpha)>>10)&255])
		f7 := archsimd.LoadInt8x16Array(&filters[((sx+7*alpha)>>10)&255])
		p0 := a.BitsToInt8().MulWidenLo(f0)
		p1 := a.ConcatShiftBytesRight(a, 1).BitsToInt8().MulWidenLo(f1)
		p2 := a.ConcatShiftBytesRight(a, 2).BitsToInt8().MulWidenLo(f2)
		p3 := a.ConcatShiftBytesRight(a, 3).BitsToInt8().MulWidenLo(f3)
		p4 := a.ConcatShiftBytesRight(a, 4).BitsToInt8().MulWidenLo(f4)
		p5 := a.ConcatShiftBytesRight(a, 5).BitsToInt8().MulWidenLo(f5)
		p6 := a.ConcatShiftBytesRight(a, 6).BitsToInt8().MulWidenLo(f6)
		p7 := a.ConcatShiftBytesRight(a, 7).BitsToInt8().MulWidenLo(f7)
		low := p0.ConcatAddPairs(p1).ConcatAddPairs(p2.ConcatAddPairs(p3))
		high := p4.ConcatAddPairs(p5).ConcatAddPairs(p6.ConcatAddPairs(p7))
		low.ConcatAddPairs(high).Add(round).Shift(right).StoreArray((*[8]int16)(mid[8*k : 8*k+8]))
		sx += beta
	}
}

// warpCompoundV8 computes the 8x8 CONV_BUF block:
// (2^19 + sum(c*(mid+4096)) + 2^6) >> 7 = (2^20 + 2^6 + sum(c*mid)) >> 7, as
// each filter row sums to 128. The eight column filters of a row are
// transposed in registers into eight tap vectors. The filter table comes in
// as a parameter so its address stays in a register across the row loop.
func warpCompoundV8(og unsafeutil.Grid[uint16], mid *warpCompoundMid, filters *[256][16]int8, sy4, gamma, delta int) {
	if !og.Fits(0, 0, 8, 8) {
		return
	}
	const seed = 1<<(8+2*filterBits-compoundRound0Bits) + 4096*128 + 1<<(compoundRound1Bits-1)
	bias := archsimd.BroadcastInt32x4(seed)
	right := archsimd.BroadcastInt32x4(-compoundRound1Bits)
	sy := sy4 + 64<<10 + 512
	// Keep the 8-row sliding window of mid in registers across output rows
	// (15 loads total instead of 64).
	m0 := archsimd.LoadInt16x8Array((*[8]int16)(mid[0:8]))
	m1 := archsimd.LoadInt16x8Array((*[8]int16)(mid[8:16]))
	m2 := archsimd.LoadInt16x8Array((*[8]int16)(mid[16:24]))
	m3 := archsimd.LoadInt16x8Array((*[8]int16)(mid[24:32]))
	m4 := archsimd.LoadInt16x8Array((*[8]int16)(mid[32:40]))
	m5 := archsimd.LoadInt16x8Array((*[8]int16)(mid[40:48]))
	m6 := archsimd.LoadInt16x8Array((*[8]int16)(mid[48:56]))
	for row := range 8 {
		m7 := archsimd.LoadInt16x8Array((*[8]int16)(mid[8*row+56 : 8*row+64]))
		var t0, t1, t2, t3, t4, t5, t6, t7 archsimd.Int16x8
		if gamma == 0 {
			f := &filters[(sy>>10)&255]
			t0 = archsimd.BroadcastInt16x8(int16(f[0]))
			t1 = archsimd.BroadcastInt16x8(int16(f[1]))
			t2 = archsimd.BroadcastInt16x8(int16(f[2]))
			t3 = archsimd.BroadcastInt16x8(int16(f[3]))
			t4 = archsimd.BroadcastInt16x8(int16(f[4]))
			t5 = archsimd.BroadcastInt16x8(int16(f[5]))
			t6 = archsimd.BroadcastInt16x8(int16(f[6]))
			t7 = archsimd.BroadcastInt16x8(int16(f[7]))
		} else {
			f0 := archsimd.LoadInt8x16Array(&filters[(sy>>10)&255])
			f1 := archsimd.LoadInt8x16Array(&filters[((sy+gamma)>>10)&255])
			f2 := archsimd.LoadInt8x16Array(&filters[((sy+2*gamma)>>10)&255])
			f3 := archsimd.LoadInt8x16Array(&filters[((sy+3*gamma)>>10)&255])
			f4 := archsimd.LoadInt8x16Array(&filters[((sy+4*gamma)>>10)&255])
			f5 := archsimd.LoadInt8x16Array(&filters[((sy+5*gamma)>>10)&255])
			f6 := archsimd.LoadInt8x16Array(&filters[((sy+6*gamma)>>10)&255])
			f7 := archsimd.LoadInt8x16Array(&filters[((sy+7*gamma)>>10)&255])
			a01 := f0.InterleaveLo(f1).ToBits().ReshapeToUint16s()
			a23 := f2.InterleaveLo(f3).ToBits().ReshapeToUint16s()
			a45 := f4.InterleaveLo(f5).ToBits().ReshapeToUint16s()
			a67 := f6.InterleaveLo(f7).ToBits().ReshapeToUint16s()
			b03lo := a01.InterleaveLo(a23).ReshapeToUint32s()
			b03hi := a01.InterleaveHi(a23).ReshapeToUint32s()
			b47lo := a45.InterleaveLo(a67).ReshapeToUint32s()
			b47hi := a45.InterleaveHi(a67).ReshapeToUint32s()
			c01 := b03lo.InterleaveLo(b47lo).ReshapeToUint8s().BitsToInt8()
			c23 := b03lo.InterleaveHi(b47lo).ReshapeToUint8s().BitsToInt8()
			c45 := b03hi.InterleaveLo(b47hi).ReshapeToUint8s().BitsToInt8()
			c67 := b03hi.InterleaveHi(b47hi).ReshapeToUint8s().BitsToInt8()
			t0 = c01.ExtendLo8ToInt16()
			t1 = c01.HiToLo().ExtendLo8ToInt16()
			t2 = c23.ExtendLo8ToInt16()
			t3 = c23.HiToLo().ExtendLo8ToInt16()
			t4 = c45.ExtendLo8ToInt16()
			t5 = c45.HiToLo().ExtendLo8ToInt16()
			t6 = c67.ExtendLo8ToInt16()
			t7 = c67.HiToLo().ExtendLo8ToInt16()
		}
		lo := bias.Add(m0.MulWidenLo(t0).Add(m1.MulWidenLo(t1)).Add(m2.MulWidenLo(t2).Add(m3.MulWidenLo(t3)))).
			Add(m4.MulWidenLo(t4).Add(m5.MulWidenLo(t5)).Add(m6.MulWidenLo(t6).Add(m7.MulWidenLo(t7))))
		hi := bias.Add(m0.HiToLo().MulWidenLo(t0.HiToLo()).Add(m1.HiToLo().MulWidenLo(t1.HiToLo())).
			Add(m2.HiToLo().MulWidenLo(t2.HiToLo()).Add(m3.HiToLo().MulWidenLo(t3.HiToLo())))).
			Add(m4.HiToLo().MulWidenLo(t4.HiToLo()).Add(m5.HiToLo().MulWidenLo(t5.HiToLo())).
				Add(m6.HiToLo().MulWidenLo(t6.HiToLo()).Add(m7.HiToLo().MulWidenLo(t7.HiToLo()))))
		lo.Shift(right).TruncToInt16().ToBits().ReshapeToUint64s().
			InterleaveLo(hi.Shift(right).TruncToInt16().ToBits().ReshapeToUint64s()).
			ReshapeToUint16s().StoreArray(og.Row8(row, 0))
		m0, m1, m2, m3, m4, m5, m6 = m1, m2, m3, m4, m5, m6, m7
		sy += delta
	}
}

// Go-native SIMD 8-bit warped-motion kernels. The horizontal pass multiplies
// centred byte samples by byte coefficients in int16 lanes. The vertical pass
// forms each output column's eight-tap dot product in int32 lanes. Both compute
// the scalar formulas of warpHorizontal8Resident and warpVertical8Full exactly.

// Each row is 16 bytes so every filter row can be loaded as a full vector.
// Only the first eight lanes are used. The source table itself has eight-byte
// rows, which makes a full vector load at its last row unsafe.
//
// The table has 256 rows, of which the first warpedPixelPrecShifts*3 + 1 are
// filters, so the kernels index it as [i&255] without a bounds check. Their
// routing preconditions (warpHorizResidentOffsInRange, warpVertFullOffsInRange)
// keep every index inside the filter rows, so the mask never changes a row.
var warp8FilterPadded [256][16]int8

func init() {
	for i := range warpedFilterI8 {
		copy(warp8FilterPadded[i][:8], warpedFilterI8[i][:])
	}
}

// warpGoSIMDEnabled selects the Go SIMD warp kernels on arm64. The
// GOAV1_DISABLE_WARP_ASM variable keeps the scalar path for A/B timing; it is
// measurement scaffolding, not a shipped switch.
var warpGoSIMDEnabled = cpu.Detected.NEON && os.Getenv("GOAV1_DISABLE_WARP_ASM") == ""

func warpHorizontal8ResidentDispatch(tmp *warpTmp, ref frame.Plane, ix4, sx4, iy4, sy4, alpha, beta, reduceBitsHoriz, offsetBitsHoriz int) int {
	if warpGoSIMDEnabled {
		return warpHorizontal8ResidentGoSIMD(tmp, ref, ix4, sx4, iy4, sy4, alpha, beta, reduceBitsHoriz, offsetBitsHoriz)
	}
	return warpHorizontal8Resident(tmp, ref, ix4, sx4, iy4, sy4, alpha, beta, reduceBitsHoriz, offsetBitsHoriz)
}

func warpVertical8FullDispatch(dst frame.Plane, tmp *warpTmp, i, j, rowShift, colShift, baseSY, gamma, delta, reduceBitsVert, offsetBitsVert int) {
	if warpGoSIMDEnabled {
		warpVertical8FullGoSIMD(dst, tmp, i, j, rowShift, colShift, baseSY, gamma, delta, reduceBitsVert, offsetBitsVert)
		return
	}
	warpVertical8Full(dst, tmp, i, j, rowShift, colShift, baseSY, gamma, delta, reduceBitsVert, offsetBitsVert)
}

func warpVertical8FullGamma0Dispatch(dst frame.Plane, tmp *warpTmp, i, j, rowShift, colShift, baseSY, delta, reduceBitsVert, offsetBitsVert int) {
	if warpGoSIMDEnabled {
		warpVertical8FullGamma0GoSIMD(dst, tmp, i, j, rowShift, colShift, baseSY, delta, reduceBitsVert, offsetBitsVert)
		return
	}
	warpVertical8FullGamma0(dst, tmp, i, j, rowShift, colShift, baseSY, delta, reduceBitsVert, offsetBitsVert)
}

// warp8FilterIndex is the warpedFilter row for a filter phase: the rounded
// phase plus the precision shift, clamped to the table (as the scalar path).
func warp8FilterIndex(sx int) int {
	offs := roundPowerOfTwo(sx, warpedDiffPrecBits) + warpedPixelPrecShifts
	if offs < 0 {
		return 0
	}
	if offs >= len(warpedFilter) {
		return len(warpedFilter) - 1
	}
	return offs
}

// warpHorizontal8ResidentGoSIMD computes the resident 15x8 intermediate.
// Its routing matches the scalar warpHorizontal8Resident preconditions.
func warpHorizontal8ResidentGoSIMD(tmp *warpTmp, ref frame.Plane, ix4, sx4, iy4, sy4, alpha, beta, reduceBitsHoriz, offsetBitsHoriz int) int {
	if reduceBitsHoriz != round0Bits || offsetBitsHoriz != 8+filterBits-1 ||
		!warpHorizResidentOffsInRange(sx4, alpha, beta) {
		return warpHorizontal8Resident(tmp, ref, ix4, sx4, iy4, sy4, alpha, beta, reduceBitsHoriz, offsetBitsHoriz)
	}
	centre := archsimd.BroadcastUint8x16(128)
	round := archsimd.BroadcastInt16x8(4)
	right := archsimd.BroadcastInt16x8(-3)
	restore := archsimd.BroadcastInt16x8(4096)
	for k := -7; k < 8; k++ {
		rowByte := (iy4+k)*ref.Stride + ix4 - 7
		sx := sx4 + beta*(k+4)
		// The resident 15-sample window fits; u8Bytes16 also handles the
		// exact last-row case where the sixteenth byte is outside ref.Pix.
		a := u8Bytes16(ref.Pix[rowByte:]).Xor(centre)
		f0 := archsimd.LoadInt8x16Array(&warp8FilterPadded[(64+((sx+512)>>10))&255])
		f1 := archsimd.LoadInt8x16Array(&warp8FilterPadded[(64+((sx+alpha+512)>>10))&255])
		f2 := archsimd.LoadInt8x16Array(&warp8FilterPadded[(64+((sx+2*alpha+512)>>10))&255])
		f3 := archsimd.LoadInt8x16Array(&warp8FilterPadded[(64+((sx+3*alpha+512)>>10))&255])
		f4 := archsimd.LoadInt8x16Array(&warp8FilterPadded[(64+((sx+4*alpha+512)>>10))&255])
		f5 := archsimd.LoadInt8x16Array(&warp8FilterPadded[(64+((sx+5*alpha+512)>>10))&255])
		f6 := archsimd.LoadInt8x16Array(&warp8FilterPadded[(64+((sx+6*alpha+512)>>10))&255])
		f7 := archsimd.LoadInt8x16Array(&warp8FilterPadded[(64+((sx+7*alpha+512)>>10))&255])
		p0 := a.BitsToInt8().MulWidenLo(f0)
		p1 := a.ConcatShiftBytesRight(a, 1).BitsToInt8().MulWidenLo(f1)
		p2 := a.ConcatShiftBytesRight(a, 2).BitsToInt8().MulWidenLo(f2)
		p3 := a.ConcatShiftBytesRight(a, 3).BitsToInt8().MulWidenLo(f3)
		p4 := a.ConcatShiftBytesRight(a, 4).BitsToInt8().MulWidenLo(f4)
		p5 := a.ConcatShiftBytesRight(a, 5).BitsToInt8().MulWidenLo(f5)
		p6 := a.ConcatShiftBytesRight(a, 6).BitsToInt8().MulWidenLo(f6)
		p7 := a.ConcatShiftBytesRight(a, 7).BitsToInt8().MulWidenLo(f7)
		low := p0.ConcatAddPairs(p1).ConcatAddPairs(p2.ConcatAddPairs(p3))
		high := p4.ConcatAddPairs(p5).ConcatAddPairs(p6.ConcatAddPairs(p7))
		low = low.ConcatAddPairs(low)
		high = high.ConcatAddPairs(high)
		out := (k + 7) * warpedIntermediateColumns
		low.Add(round).Shift(right).Add(restore).ExtendLo4ToInt32().StoreArray((*[4]int32)(tmp[out : out+4]))
		high.Add(round).Shift(right).Add(restore).ExtendLo4ToInt32().StoreArray((*[4]int32)(tmp[out+4 : out+8]))
	}
	return sy4
}

// warpVertical8FullGoSIMD computes the vertical block with the same
// routing; the eight output columns of each row are two vectors of four.
func warpVertical8FullGoSIMD(dst frame.Plane, tmp *warpTmp, i, j, rowShift, colShift, baseSY, gamma, delta, reduceBitsVert, offsetBitsVert int) {
	if reduceBitsVert != round1Bits || offsetBitsVert != 8+2*filterBits-round0Bits ||
		!warpVertFullOffsInRange(baseSY, gamma, delta) {
		warpVertical8Full(dst, tmp, i, j, rowShift, colShift, baseSY, gamma, delta, reduceBitsVert, offsetBitsVert)
		return
	}
	warpVerticalGoSIMDCommon(dst, tmp, i, j, rowShift, colShift, baseSY, gamma, delta, reduceBitsVert, offsetBitsVert)
}

// warpVertical8FullGamma0GoSIMD is the gamma-zero vertical kernel: every column
// of a row shares one filter phase.
func warpVertical8FullGamma0GoSIMD(dst frame.Plane, tmp *warpTmp, i, j, rowShift, colShift, baseSY, delta, reduceBitsVert, offsetBitsVert int) {
	if reduceBitsVert != round1Bits || offsetBitsVert != 8+2*filterBits-round0Bits ||
		!warpVertFullOffsInRange(baseSY, 0, delta) {
		warpVertical8FullGamma0(dst, tmp, i, j, rowShift, colShift, baseSY, delta, reduceBitsVert, offsetBitsVert)
		return
	}
	warpVertical8Gamma0Kernel(dst, tmp, i, j, rowShift, colShift, baseSY, delta)
}

// warpVertical8Gamma0Kernel uses one filter for all eight columns. Folding the
// final subtraction into the accumulator seed is exact for the arithmetic
// right shift: (sum >> 11) - 384 == (sum - 384*2048) >> 11.
// The caller has proved that all eight destination columns fit.
func warpVertical8Gamma0Kernel(dst frame.Plane, tmp *warpTmp, i, j, rowShift, colShift, baseSY, delta int) {
	const seed = (1 << (8 + 2*filterBits - round0Bits)) + (1 << (round1Bits - 1)) - ((1<<7)+(1<<8))*(1<<round1Bits)
	bias := archsimd.BroadcastInt32x4(seed)
	right := archsimd.BroadcastInt32x4(-round1Bits)
	for row := 0; row < 8; row++ {
		// The eight taps of this output row read tmp rows row..row+7.
		win := (*[8 * warpedIntermediateColumns]int32)(tmp[row*warpedIntermediateColumns:])
		load := func(off int) archsimd.Int32x4 {
			return archsimd.LoadInt32x4Array((*[4]int32)(win[off : off+4]))
		}
		coeff := &warpedFilter[warp8FilterIndex(baseSY+delta*row)]
		c0 := archsimd.BroadcastInt32x4(int32(coeff[0]))
		c1 := archsimd.BroadcastInt32x4(int32(coeff[1]))
		c2 := archsimd.BroadcastInt32x4(int32(coeff[2]))
		c3 := archsimd.BroadcastInt32x4(int32(coeff[3]))
		c4 := archsimd.BroadcastInt32x4(int32(coeff[4]))
		c5 := archsimd.BroadcastInt32x4(int32(coeff[5]))
		c6 := archsimd.BroadcastInt32x4(int32(coeff[6]))
		c7 := archsimd.BroadcastInt32x4(int32(coeff[7]))
		const p = 0
		lo := load(p).MulAdd(c0, bias)
		hi := load(p+4).MulAdd(c0, bias)
		lo = load(p+8).MulAdd(c1, lo)
		hi = load(p+12).MulAdd(c1, hi)
		lo = load(p+16).MulAdd(c2, lo)
		hi = load(p+20).MulAdd(c2, hi)
		lo = load(p+24).MulAdd(c3, lo)
		hi = load(p+28).MulAdd(c3, hi)
		lo = load(p+32).MulAdd(c4, lo)
		hi = load(p+36).MulAdd(c4, hi)
		lo = load(p+40).MulAdd(c5, lo)
		hi = load(p+44).MulAdd(c5, hi)
		lo = load(p+48).MulAdd(c6, lo)
		hi = load(p+52).MulAdd(c6, hi)
		lo = load(p+56).MulAdd(c7, lo)
		hi = load(p+60).MulAdd(c7, hi)
		dstRow := (i+rowShift+row)*dst.Stride + j + colShift
		u8PackSaturate32(u8Right32(lo, right, round1Bits), u8Right32(hi, right, round1Bits)).StorePart(dst.Pix[dstRow : dstRow+8])
	}
}

// warpVerticalGoSIMDCommon computes the 8x8 vertical output block. Eight byte
// filter rows are transposed in registers so each tap vector contains the
// coefficients for eight columns. The int32 horizontal intermediate fits
// signed int16, so widening products preserve the scalar dot product.
func warpVerticalGoSIMDCommon(dst frame.Plane, tmp *warpTmp, i, j, rowShift, colShift, baseSY, gamma, delta, reduceBitsVert, offsetBitsVert int) {
	const seed = (1 << (8 + 2*filterBits - round0Bits)) + (1 << (round1Bits - 1)) - ((1<<7)+(1<<8))*(1<<round1Bits)
	bias := archsimd.BroadcastInt32x4(seed)
	right := archsimd.BroadcastInt32x4(-round1Bits)
	for row := 0; row < 8; row++ {
		// The eight taps of this output row read tmp rows row..row+7.
		win := (*[8 * warpedIntermediateColumns]int32)(tmp[row*warpedIntermediateColumns:])
		load := func(off int) archsimd.Int16x8 {
			lo := archsimd.LoadInt32x4Array((*[4]int32)(win[off : off+4]))
			hi := archsimd.LoadInt32x4Array((*[4]int32)(win[off+4 : off+8]))
			// u8PackU16From32 written out, with the second narrow as XTN2 into
			// the high half: through the helper each load left an inline-mark
			// NOP and joined two XTN results with a ZIP1.
			n := lo.TruncToInt16().ToBits().ReshapeToUint64s().BitsToFloat64()
			return n.SetElem(1, hi.TruncToInt16().ToBits().ReshapeToUint64s().BitsToFloat64().GetElem(0)).ToBits().ReshapeToUint16s().BitsToInt16()
		}
		sy := baseSY + delta*row + 512
		f0 := archsimd.LoadInt8x16Array(&warp8FilterPadded[(64+(sy>>10))&255])
		f1 := archsimd.LoadInt8x16Array(&warp8FilterPadded[(64+((sy+gamma)>>10))&255])
		f2 := archsimd.LoadInt8x16Array(&warp8FilterPadded[(64+((sy+2*gamma)>>10))&255])
		f3 := archsimd.LoadInt8x16Array(&warp8FilterPadded[(64+((sy+3*gamma)>>10))&255])
		f4 := archsimd.LoadInt8x16Array(&warp8FilterPadded[(64+((sy+4*gamma)>>10))&255])
		f5 := archsimd.LoadInt8x16Array(&warp8FilterPadded[(64+((sy+5*gamma)>>10))&255])
		f6 := archsimd.LoadInt8x16Array(&warp8FilterPadded[(64+((sy+6*gamma)>>10))&255])
		f7 := archsimd.LoadInt8x16Array(&warp8FilterPadded[(64+((sy+7*gamma)>>10))&255])
		a01 := f0.InterleaveLo(f1).ToBits().ReshapeToUint16s()
		a23 := f2.InterleaveLo(f3).ToBits().ReshapeToUint16s()
		a45 := f4.InterleaveLo(f5).ToBits().ReshapeToUint16s()
		a67 := f6.InterleaveLo(f7).ToBits().ReshapeToUint16s()
		b03lo := a01.InterleaveLo(a23).ReshapeToUint32s()
		b03hi := a01.InterleaveHi(a23).ReshapeToUint32s()
		b47lo := a45.InterleaveLo(a67).ReshapeToUint32s()
		b47hi := a45.InterleaveHi(a67).ReshapeToUint32s()
		c01 := b03lo.InterleaveLo(b47lo).ReshapeToUint8s()
		c23 := b03lo.InterleaveHi(b47lo).ReshapeToUint8s()
		c45 := b03hi.InterleaveLo(b47hi).ReshapeToUint8s()
		c67 := b03hi.InterleaveHi(b47hi).ReshapeToUint8s()
		t0 := c01.BitsToInt8().ExtendLo8ToInt16()
		t1 := c01.BitsToInt8().HiToLo().ExtendLo8ToInt16()
		t2 := c23.BitsToInt8().ExtendLo8ToInt16()
		t3 := c23.BitsToInt8().HiToLo().ExtendLo8ToInt16()
		t4 := c45.BitsToInt8().ExtendLo8ToInt16()
		t5 := c45.BitsToInt8().HiToLo().ExtendLo8ToInt16()
		t6 := c67.BitsToInt8().ExtendLo8ToInt16()
		t7 := c67.BitsToInt8().HiToLo().ExtendLo8ToInt16()
		const p = 0
		lo, hi := u8WideMAC8(bias, bias, load(p), t0)
		lo, hi = u8WideMAC8(lo, hi, load(p+8), t1)
		lo, hi = u8WideMAC8(lo, hi, load(p+16), t2)
		lo, hi = u8WideMAC8(lo, hi, load(p+24), t3)
		lo, hi = u8WideMAC8(lo, hi, load(p+32), t4)
		lo, hi = u8WideMAC8(lo, hi, load(p+40), t5)
		lo, hi = u8WideMAC8(lo, hi, load(p+48), t6)
		lo, hi = u8WideMAC8(lo, hi, load(p+56), t7)
		dstRow := (i+rowShift+row)*dst.Stride + j + colShift
		u8PackSaturate32(u8Right32(lo, right, round1Bits), u8Right32(hi, right, round1Bits)).StorePart(dst.Pix[dstRow : dstRow+8])
	}
}

// 8-bit single-reference warp (dav1d warp_affine_8x8, src/arm/64/mc.S): the
// same int16 15x8 intermediate as the compound block (warpCompoundH8), so the
// vertical pass multiplies the int16 rows directly instead of narrowing the
// int32 warpTmp rows for every output row. The results equal
// warpAffine8Offset's scalar formulas sample for sample.

// warpPut8x8Dispatch predicts one full 8x8 warp block into dst (row r at
// dst[r*stride:]). It reports false when the SIMD kernel does not apply; the
// caller then runs the int32 path.
func warpPut8x8Dispatch(dst []byte, stride int, ref frame.Plane, ix4, sx4, iy4, sy4, alpha, beta, gamma, delta int) bool {
	if !warpGoSIMDEnabled || !warpHorizResidentOffsInRange(sx4, alpha, beta) || !warpVertFullOffsInRange(sy4, gamma, delta) {
		return false
	}
	var mid warpCompoundMid
	base := (iy4-7)*ref.Stride + ix4 - 7
	if iy4 >= 7 && iy4+7 < ref.Height && ix4 >= 7 && ix4+8 <= ref.Width && unsafeutil.GridFit(len(ref.Pix), base, ref.Stride, 16, 15) {
		warpCompoundH8(&mid, &warp8FilterPadded, unsafeutil.NewGrid(ref.Pix, base, ref.Stride, 16, 15), sx4, alpha, beta)
	} else {
		var edge [15 * 16]byte
		warpCompoundEmuEdge(&edge, ref, ix4, iy4)
		warpCompoundH8(&mid, &warp8FilterPadded, unsafeutil.NewGrid(edge[:], 0, 16, 16, 15), sx4, alpha, beta)
	}
	if gamma == 0 {
		warpPutV8Gamma0(dst, stride, &mid, &warp8FilterPadded, sy4, delta)
	} else {
		warpPutV8(dst, stride, &mid, &warp8FilterPadded, sy4, gamma, delta)
	}
	return true
}

// warpPutSeed folds the scalar vertical constants into the accumulator:
// round2(2^19 + sum(c*(mid+4096)), 11) - 384 == (seed + sum(c*mid)) >> 11 with
// seed = 2^19 + 4096*128 + 2^10 - 384*2^11 (each filter row sums to 128, and
// subtracting a multiple of 2^11 commutes with the arithmetic shift).
const warpPutSeed = 1<<(8+2*filterBits-round0Bits) + 4096*128 + 1<<(round1Bits-1) - ((1<<7)+(1<<8))<<round1Bits

// warpPutV8 is the vertical pass with per-column filters (gamma != 0): the
// eight column filters of a row are transposed in registers into eight tap
// vectors, as dav1d's transpose_8x8b_xtl.
func warpPutV8(dst []byte, stride int, mid *warpCompoundMid, filters *[256][16]int8, sy4, gamma, delta int) {
	bias := archsimd.BroadcastInt32x4(warpPutSeed)
	right := archsimd.BroadcastInt32x4(-round1Bits)
	sy := sy4 + 64<<10 + 512
	for row := range 8 {
		f0 := archsimd.LoadInt8x16Array(&filters[(sy>>10)&255])
		f1 := archsimd.LoadInt8x16Array(&filters[((sy+gamma)>>10)&255])
		f2 := archsimd.LoadInt8x16Array(&filters[((sy+2*gamma)>>10)&255])
		f3 := archsimd.LoadInt8x16Array(&filters[((sy+3*gamma)>>10)&255])
		f4 := archsimd.LoadInt8x16Array(&filters[((sy+4*gamma)>>10)&255])
		f5 := archsimd.LoadInt8x16Array(&filters[((sy+5*gamma)>>10)&255])
		f6 := archsimd.LoadInt8x16Array(&filters[((sy+6*gamma)>>10)&255])
		f7 := archsimd.LoadInt8x16Array(&filters[((sy+7*gamma)>>10)&255])
		a01 := f0.InterleaveLo(f1).ToBits().ReshapeToUint16s()
		a23 := f2.InterleaveLo(f3).ToBits().ReshapeToUint16s()
		a45 := f4.InterleaveLo(f5).ToBits().ReshapeToUint16s()
		a67 := f6.InterleaveLo(f7).ToBits().ReshapeToUint16s()
		b03lo := a01.InterleaveLo(a23).ReshapeToUint32s()
		b03hi := a01.InterleaveHi(a23).ReshapeToUint32s()
		b47lo := a45.InterleaveLo(a67).ReshapeToUint32s()
		b47hi := a45.InterleaveHi(a67).ReshapeToUint32s()
		c01 := b03lo.InterleaveLo(b47lo).ReshapeToUint8s().BitsToInt8()
		c23 := b03lo.InterleaveHi(b47lo).ReshapeToUint8s().BitsToInt8()
		c45 := b03hi.InterleaveLo(b47hi).ReshapeToUint8s().BitsToInt8()
		c67 := b03hi.InterleaveHi(b47hi).ReshapeToUint8s().BitsToInt8()
		t0 := c01.ExtendLo8ToInt16()
		t1 := c01.HiToLo().ExtendLo8ToInt16()
		t2 := c23.ExtendLo8ToInt16()
		t3 := c23.HiToLo().ExtendLo8ToInt16()
		t4 := c45.ExtendLo8ToInt16()
		t5 := c45.HiToLo().ExtendLo8ToInt16()
		t6 := c67.ExtendLo8ToInt16()
		t7 := c67.HiToLo().ExtendLo8ToInt16()
		win := (*[64]int16)(mid[8*row : 8*row+64])
		m0 := archsimd.LoadInt16x8Array((*[8]int16)(win[0:8]))
		m1 := archsimd.LoadInt16x8Array((*[8]int16)(win[8:16]))
		m2 := archsimd.LoadInt16x8Array((*[8]int16)(win[16:24]))
		m3 := archsimd.LoadInt16x8Array((*[8]int16)(win[24:32]))
		m4 := archsimd.LoadInt16x8Array((*[8]int16)(win[32:40]))
		m5 := archsimd.LoadInt16x8Array((*[8]int16)(win[40:48]))
		m6 := archsimd.LoadInt16x8Array((*[8]int16)(win[48:56]))
		m7 := archsimd.LoadInt16x8Array((*[8]int16)(win[56:64]))
		lo := bias.Add(m0.MulWidenLo(t0)).Add(m1.MulWidenLo(t1)).Add(m2.MulWidenLo(t2)).Add(m3.MulWidenLo(t3)).
			Add(m4.MulWidenLo(t4)).Add(m5.MulWidenLo(t5)).Add(m6.MulWidenLo(t6)).Add(m7.MulWidenLo(t7))
		hi := bias.Add(m0.HiToLo().MulWidenLo(t0.HiToLo())).Add(m1.HiToLo().MulWidenLo(t1.HiToLo())).
			Add(m2.HiToLo().MulWidenLo(t2.HiToLo())).Add(m3.HiToLo().MulWidenLo(t3.HiToLo())).
			Add(m4.HiToLo().MulWidenLo(t4.HiToLo())).Add(m5.HiToLo().MulWidenLo(t5.HiToLo())).
			Add(m6.HiToLo().MulWidenLo(t6.HiToLo())).Add(m7.HiToLo().MulWidenLo(t7.HiToLo()))
		o := row * stride
		u8PackSaturate32(lo.Shift(right), hi.Shift(right)).StorePart(dst[o : o+8])
		sy += delta
	}
}

// warpLane[k] is the TBL index vector that broadcasts int16 lane k.
var warpLane = func() (t [8][16]uint8) {
	for k := range t {
		for l := 0; l < 8; l++ {
			t[k][2*l] = uint8(2 * k)
			t[k][2*l+1] = uint8(2*k + 1)
		}
	}
	return t
}()

// warpPutV8Gamma0 is warpPutV8 when every column of a row shares one filter
// (gamma == 0): the eight taps are lane broadcasts of the single filter row.
func warpPutV8Gamma0(dst []byte, stride int, mid *warpCompoundMid, filters *[256][16]int8, sy4, delta int) {
	bias := archsimd.BroadcastInt32x4(warpPutSeed)
	right := archsimd.BroadcastInt32x4(-round1Bits)
	l0, l1 := archsimd.LoadUint8x16Array(&warpLane[0]), archsimd.LoadUint8x16Array(&warpLane[1])
	l2, l3 := archsimd.LoadUint8x16Array(&warpLane[2]), archsimd.LoadUint8x16Array(&warpLane[3])
	l4, l5 := archsimd.LoadUint8x16Array(&warpLane[4]), archsimd.LoadUint8x16Array(&warpLane[5])
	l6, l7 := archsimd.LoadUint8x16Array(&warpLane[6]), archsimd.LoadUint8x16Array(&warpLane[7])
	sy := sy4 + 64<<10 + 512
	for row := range 8 {
		fb := archsimd.LoadInt8x16Array(&filters[(sy>>10)&255]).ExtendLo8ToInt16().ToBits().ReshapeToUint8s()
		t0 := fb.LookupOrZero(l0).ReshapeToUint16s().BitsToInt16()
		t1 := fb.LookupOrZero(l1).ReshapeToUint16s().BitsToInt16()
		t2 := fb.LookupOrZero(l2).ReshapeToUint16s().BitsToInt16()
		t3 := fb.LookupOrZero(l3).ReshapeToUint16s().BitsToInt16()
		t4 := fb.LookupOrZero(l4).ReshapeToUint16s().BitsToInt16()
		t5 := fb.LookupOrZero(l5).ReshapeToUint16s().BitsToInt16()
		t6 := fb.LookupOrZero(l6).ReshapeToUint16s().BitsToInt16()
		t7 := fb.LookupOrZero(l7).ReshapeToUint16s().BitsToInt16()
		win := (*[64]int16)(mid[8*row : 8*row+64])
		m0 := archsimd.LoadInt16x8Array((*[8]int16)(win[0:8]))
		m1 := archsimd.LoadInt16x8Array((*[8]int16)(win[8:16]))
		m2 := archsimd.LoadInt16x8Array((*[8]int16)(win[16:24]))
		m3 := archsimd.LoadInt16x8Array((*[8]int16)(win[24:32]))
		m4 := archsimd.LoadInt16x8Array((*[8]int16)(win[32:40]))
		m5 := archsimd.LoadInt16x8Array((*[8]int16)(win[40:48]))
		m6 := archsimd.LoadInt16x8Array((*[8]int16)(win[48:56]))
		m7 := archsimd.LoadInt16x8Array((*[8]int16)(win[56:64]))
		lo := bias.Add(m0.MulWidenLo(t0)).Add(m1.MulWidenLo(t1)).Add(m2.MulWidenLo(t2)).Add(m3.MulWidenLo(t3)).
			Add(m4.MulWidenLo(t4)).Add(m5.MulWidenLo(t5)).Add(m6.MulWidenLo(t6)).Add(m7.MulWidenLo(t7))
		hi := bias.Add(m0.HiToLo().MulWidenLo(t0.HiToLo())).Add(m1.HiToLo().MulWidenLo(t1.HiToLo())).
			Add(m2.HiToLo().MulWidenLo(t2.HiToLo())).Add(m3.HiToLo().MulWidenLo(t3.HiToLo())).
			Add(m4.HiToLo().MulWidenLo(t4.HiToLo())).Add(m5.HiToLo().MulWidenLo(t5.HiToLo())).
			Add(m6.HiToLo().MulWidenLo(t6.HiToLo())).Add(m7.HiToLo().MulWidenLo(t7.HiToLo()))
		o := row * stride
		u8PackSaturate32(lo.Shift(right), hi.Shift(right)).StorePart(dst[o : o+8])
		sy += delta
	}
}

// SIMD dot products center uint16 samples before signed widening. Every row in
// warpedFilter sums to 128, so this restores the removed 32768*128 from each
// dot while preserving the full uint16 input domain.
const warpHighBDSampleCenterBias = 1 << 22

// warpHorizontalHighBDResidentGoSIMD computes the resident 15x8 tile using
// pairwise vector reductions to produce four horizontal outputs at a time.
func warpHorizontalHighBDResidentGoSIMD(tmp *[warpedIntermediateRows * warpedIntermediateColumns]int32, ref frame.Plane, ix4 int, sx4 int, iy4 int, sy4 int, alpha int, beta int, reduceBitsHoriz int, offsetBitsHoriz int) int {
	// The samples load as bytes, so no alignment is required of ref.
	center := archsimd.BroadcastInt16x8(-1 << 15)
	// All 193 filter rows sum to 128 and have L1 <= 222 (checked in tests).
	// Centered dots are bounded by 7,274,496; the compensation and rounding
	// bias therefore remain safely within int32.
	roundBias := archsimd.BroadcastInt32x4(int32(warpHighBDSampleCenterBias + (1 << offsetBitsHoriz) + (1 << (reduceBitsHoriz - 1))))
	roundShift := archsimd.BroadcastInt32x4(-int32(reduceBitsHoriz))
	for k := -7; k < 8; k++ {
		// The row's 15 taps, columns ix4-7 .. ix4+7 (the caller keeps them
		// inside the plane).
		row := (*[30]byte)(ref.Pix[(iy4+k)*ref.Stride+(ix4-7)*2:])
		sx := sx4 + beta*(k+4)
		for group := 0; group < warpedIntermediateColumns; group += 4 {
			phase0 := sx
			phase1 := phase0 + alpha
			phase2 := phase1 + alpha
			phase3 := phase2 + alpha
			offs0 := warpHorizontalHighBDFilterIndex(phase0)
			offs1 := warpHorizontalHighBDFilterIndex(phase1)
			offs2 := warpHorizontalHighBDFilterIndex(phase2)
			offs3 := warpHorizontalHighBDFilterIndex(phase3)
			sx = phase3 + alpha

			g := (*[22]byte)(row[2*group:])
			s0 := archsimd.LoadUint8x16Array((*[16]byte)(g[0:16])).ReshapeToUint16s().BitsToInt16().Xor(center)
			s1 := archsimd.LoadUint8x16Array((*[16]byte)(g[2:18])).ReshapeToUint16s().BitsToInt16().Xor(center)
			s2 := archsimd.LoadUint8x16Array((*[16]byte)(g[4:20])).ReshapeToUint16s().BitsToInt16().Xor(center)
			s3 := archsimd.LoadUint8x16Array((*[16]byte)(g[6:22])).ReshapeToUint16s().BitsToInt16().Xor(center)
			c0 := archsimd.LoadInt16x8Array(&warpedFilter[offs0])
			c1 := archsimd.LoadInt16x8Array(&warpedFilter[offs1])
			c2 := archsimd.LoadInt16x8Array(&warpedFilter[offs2])
			c3 := archsimd.LoadInt16x8Array(&warpedFilter[offs3])

			p0 := s0.MulWidenLo(c0)
			p1 := s1.MulWidenLo(c1)
			p2 := s2.MulWidenLo(c2)
			p3 := s3.MulWidenLo(c3)
			q0 := s0.HiToLo().MulWidenLo(c0.HiToLo())
			q1 := s1.HiToLo().MulWidenLo(c1.HiToLo())
			q2 := s2.HiToLo().MulWidenLo(c2.HiToLo())
			q3 := s3.HiToLo().MulWidenLo(c3.HiToLo())
			// Each pairwise tree gathers four terms from each output lane.
			lo := p0.ConcatAddPairs(p1).ConcatAddPairs(p2.ConcatAddPairs(p3))
			hi := q0.ConcatAddPairs(q1).ConcatAddPairs(q2.ConcatAddPairs(q3))
			dots := lo.Add(hi)

			out := (k+7)*warpedIntermediateColumns + group
			dots.Add(roundBias).Shift(roundShift).StoreArray((*[4]int32)(tmp[out:]))
		}
	}
	return sy4
}

func warpHorizontalHighBDResidentDispatch(tmp *[warpedIntermediateRows * warpedIntermediateColumns]int32, ref frame.Plane, ix4 int, sx4 int, iy4 int, sy4 int, alpha int, beta int, reduceBitsHoriz int, offsetBitsHoriz int) int {
	return warpHorizontalHighBDResidentGoSIMD(tmp, ref, ix4, sx4, iy4, sy4, alpha, beta, reduceBitsHoriz, offsetBitsHoriz)
}

func warpHorizontalHighBDFilterIndex(sx int) int {
	offs := roundPowerOfTwo(sx, warpedDiffPrecBits) + warpedPixelPrecShifts
	if offs < 0 {
		return 0
	}
	if offs >= len(warpedFilter) {
		return len(warpedFilter) - 1
	}
	return offs
}

// The 16bpc vertical warp pass (dav1d warp_affine_8x8_16bpc_neon, the
// vertical half of src/arm/64/mc16.S warp): eight output rows of eight
// columns, each column's eight-tap filter taken from its own phase. As in
// dav1d the eight filter rows of an output row are transposed in registers,
// so tap m is one vector over the eight columns; the int32 intermediate rows
// multiply-accumulate against those taps widened to int32 lanes (MLA), which
// is exact for any int32 intermediate the horizontal pass produces (the sums
// stay far inside int32). The scalar rounding and the final
// - 2^(bd-1) - 2^bd fold into the accumulator seed: for the arithmetic right
// shift, (s >> r) - k == (s - k*2^r) >> r.

func warpVerticalHighBDFullDispatch(dst frame.Plane, bitDepth uint8, max uint16, tmp *warpTmp, i, j, rowShift, colShift, baseSY, gamma, delta, reduceBitsVert, offsetBitsVert int) {
	if !warpGoSIMDEnabled || !warpVertFullOffsInRange(baseSY, gamma, delta) {
		warpVerticalHighBDFull(dst, bitDepth, max, tmp, i, j, rowShift, colShift, baseSY, gamma, delta, reduceBitsVert, offsetBitsVert)
		return
	}
	warpVerticalHighBDGoSIMD(dst, bitDepth, max, tmp, i+rowShift, j+colShift, baseSY, gamma, delta, reduceBitsVert, offsetBitsVert)
}

func warpVerticalHighBDFullGamma0Dispatch(dst frame.Plane, bitDepth uint8, max uint16, tmp *warpTmp, i, j, rowShift, colShift, baseSY, delta, reduceBitsVert, offsetBitsVert int) {
	if !warpGoSIMDEnabled || !warpVertFullOffsInRange(baseSY, 0, delta) {
		warpVerticalHighBDFullGamma0(dst, bitDepth, max, tmp, i, j, rowShift, colShift, baseSY, delta, reduceBitsVert, offsetBitsVert)
		return
	}
	warpVerticalHighBDGamma0GoSIMD(dst, bitDepth, max, tmp, i+rowShift, j+colShift, baseSY, delta, reduceBitsVert, offsetBitsVert)
}

// warpHBDVertSeed is the accumulator seed of the 16bpc vertical pass:
// 2^offset + 2^(reduce-1) - (2^(bd-1) + 2^bd) * 2^reduce.
func warpHBDVertSeed(bitDepth uint8, reduce, offset int) int32 {
	return int32(1)<<offset + int32(1)<<(reduce-1) - (int32(1)<<(bitDepth-1)+int32(1)<<bitDepth)<<reduce
}

// warpHBDStoreRow clips the eight int32 sums lo|hi (already shifted) to
// [0, max] and stores them as eight 16-bit samples at dst row y, column x.
func warpHBDStoreRow(dst frame.Plane, y, x int, lo, hi, zero, maxV archsimd.Int32x4) {
	v := lo.Max(zero).Min(maxV).ToBits().ReshapeToUint16s().ConcatEven(hi.Max(zero).Min(maxV).ToBits().ReshapeToUint16s())
	off := y*dst.Stride + 2*x
	v.ReshapeToUint8s().StoreArray((*[16]byte)(dst.Pix[off : off+16]))
}

// warpVerticalHighBDGoSIMD computes the 8x8 output block at (x0, y0); the
// caller has checked every filter phase is in range.
func warpVerticalHighBDGoSIMD(dst frame.Plane, bitDepth uint8, max uint16, tmp *warpTmp, y0, x0, baseSY, gamma, delta, reduce, offset int) {
	seed := archsimd.BroadcastInt32x4(warpHBDVertSeed(bitDepth, reduce, offset))
	right := archsimd.BroadcastInt32x4(-int32(reduce))
	zero := archsimd.Int32x4{}
	maxV := archsimd.BroadcastInt32x4(int32(max))
	for row := 0; row < 8; row++ {
		// The eight taps of this output row read tmp rows row..row+7.
		win := (*[8 * warpedIntermediateColumns]int32)(tmp[row*warpedIntermediateColumns:])
		sy := baseSY + delta*row + 512
		f0 := archsimd.LoadInt8x16Array(&warp8FilterPadded[(64+(sy>>10))&255])
		f1 := archsimd.LoadInt8x16Array(&warp8FilterPadded[(64+((sy+gamma)>>10))&255])
		f2 := archsimd.LoadInt8x16Array(&warp8FilterPadded[(64+((sy+2*gamma)>>10))&255])
		f3 := archsimd.LoadInt8x16Array(&warp8FilterPadded[(64+((sy+3*gamma)>>10))&255])
		f4 := archsimd.LoadInt8x16Array(&warp8FilterPadded[(64+((sy+4*gamma)>>10))&255])
		f5 := archsimd.LoadInt8x16Array(&warp8FilterPadded[(64+((sy+5*gamma)>>10))&255])
		f6 := archsimd.LoadInt8x16Array(&warp8FilterPadded[(64+((sy+6*gamma)>>10))&255])
		f7 := archsimd.LoadInt8x16Array(&warp8FilterPadded[(64+((sy+7*gamma)>>10))&255])
		a01 := f0.InterleaveLo(f1).ToBits().ReshapeToUint16s()
		a23 := f2.InterleaveLo(f3).ToBits().ReshapeToUint16s()
		a45 := f4.InterleaveLo(f5).ToBits().ReshapeToUint16s()
		a67 := f6.InterleaveLo(f7).ToBits().ReshapeToUint16s()
		b03lo := a01.InterleaveLo(a23).ReshapeToUint32s()
		b03hi := a01.InterleaveHi(a23).ReshapeToUint32s()
		b47lo := a45.InterleaveLo(a67).ReshapeToUint32s()
		b47hi := a45.InterleaveHi(a67).ReshapeToUint32s()
		// c01 holds tap 0 of columns 0..7 in bytes 0..7 and tap 1 in bytes
		// 8..15, and so on.
		c01 := b03lo.InterleaveLo(b47lo).ReshapeToUint8s().BitsToInt8()
		c23 := b03lo.InterleaveHi(b47lo).ReshapeToUint8s().BitsToInt8()
		c45 := b03hi.InterleaveLo(b47hi).ReshapeToUint8s().BitsToInt8()
		c67 := b03hi.InterleaveHi(b47hi).ReshapeToUint8s().BitsToInt8()
		lo, hi := seed, seed
		mac := func(m int, t archsimd.Int16x8) {
			lo = archsimd.LoadInt32x4Array((*[4]int32)(win[8*m:8*m+4])).MulAdd(t.ExtendLo4ToInt32(), lo)
			hi = archsimd.LoadInt32x4Array((*[4]int32)(win[8*m+4:8*m+8])).MulAdd(t.HiToLo().ExtendLo4ToInt32(), hi)
		}
		mac(0, c01.ExtendLo8ToInt16())
		mac(1, c01.HiToLo().ExtendLo8ToInt16())
		mac(2, c23.ExtendLo8ToInt16())
		mac(3, c23.HiToLo().ExtendLo8ToInt16())
		mac(4, c45.ExtendLo8ToInt16())
		mac(5, c45.HiToLo().ExtendLo8ToInt16())
		mac(6, c67.ExtendLo8ToInt16())
		mac(7, c67.HiToLo().ExtendLo8ToInt16())
		warpHBDStoreRow(dst, y0+row, x0, lo.Shift(right), hi.Shift(right), zero, maxV)
	}
}

// warpVerticalHighBDGamma0GoSIMD is the gamma-zero vertical pass: one filter
// for all eight columns of a row.
func warpVerticalHighBDGamma0GoSIMD(dst frame.Plane, bitDepth uint8, max uint16, tmp *warpTmp, y0, x0, baseSY, delta, reduce, offset int) {
	seed := archsimd.BroadcastInt32x4(warpHBDVertSeed(bitDepth, reduce, offset))
	right := archsimd.BroadcastInt32x4(-int32(reduce))
	zero := archsimd.Int32x4{}
	maxV := archsimd.BroadcastInt32x4(int32(max))
	for row := 0; row < 8; row++ {
		win := (*[8 * warpedIntermediateColumns]int32)(tmp[row*warpedIntermediateColumns:])
		coeff := &warpedFilter[64+((baseSY+delta*row+512)>>10)]
		lo, hi := seed, seed
		mac := func(m int) {
			c := archsimd.BroadcastInt32x4(int32(coeff[m]))
			lo = archsimd.LoadInt32x4Array((*[4]int32)(win[8*m:8*m+4])).MulAdd(c, lo)
			hi = archsimd.LoadInt32x4Array((*[4]int32)(win[8*m+4:8*m+8])).MulAdd(c, hi)
		}
		mac(0)
		mac(1)
		mac(2)
		mac(3)
		mac(4)
		mac(5)
		mac(6)
		mac(7)
		warpHBDStoreRow(dst, y0+row, x0, lo.Shift(right), hi.Shift(right), zero, maxV)
	}
}
