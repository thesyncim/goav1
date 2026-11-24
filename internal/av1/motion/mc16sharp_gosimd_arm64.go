// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package motion

import (
	"encoding/binary"
	"simd/archsimd"
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
