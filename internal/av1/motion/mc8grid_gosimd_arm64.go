// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package motion

import (
	"simd/archsimd"

	"github.com/thesyncim/goav1/internal/unsafeutil"
)

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
			*unsafeutil.Row[[4]byte](d, r, 0) = *unsafeutil.Row[[4]byte](s, r, 0)
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
	mul := archsimd.BroadcastUint8x16(16)
	mulHi := mul.HiToLo()
	off := archsimd.BroadcastUint16x8(6144)
	h, w := og.Height(), og.Width()
	if w == 4 {
		if !sg.Fits(0, 0, 16, h) {
			return
		}
		s := sg.Sub(0, 0, 16, h)
		for r := 0; r < h; r++ {
			v := archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](s, r, 0)).MulWidenLo(mul).Add(off)
			mc8Store4U16(unsafeutil.Row[[4]uint16](og, r, 0), v.ReshapeToUint64s().GetElem(0))
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
			a := archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](s, r, 0))
			p := unsafeutil.Row[[16]uint16](o, r, 0)
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
			archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](s, r, 0)).MulWidenLo(mul).Add(off).StoreArray(unsafeutil.Row[[8]uint16](o, r, 0))
		}
	}
}

// mc8PrepCopyW8 is mc8PrepCopy for an eight-wide block, whose CONV_BUF rows
// are contiguous: eight rows per chunk are one 64-sample window of out
// (constant store offsets) and one fixed-shape sub-view of sg.
func mc8PrepCopyW8(sg unsafeutil.Grid[uint8], out []uint16, h int) {
	mul := archsimd.BroadcastUint8x16(16)
	off := archsimd.BroadcastUint16x8(6144)
	r := 0
	for ; r+8 <= h; r += 8 {
		if !sg.Fits(r, 0, 16, 8) || 8*r+64 > len(out) {
			return
		}
		s, o := sg.Sub(r, 0, 16, 8), (*[64]uint16)(out[8*r:8*r+64])
		archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](s, 0, 0)).MulWidenLo(mul).Add(off).StoreArray((*[8]uint16)(o[0:8]))
		archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](s, 1, 0)).MulWidenLo(mul).Add(off).StoreArray((*[8]uint16)(o[8:16]))
		archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](s, 2, 0)).MulWidenLo(mul).Add(off).StoreArray((*[8]uint16)(o[16:24]))
		archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](s, 3, 0)).MulWidenLo(mul).Add(off).StoreArray((*[8]uint16)(o[24:32]))
		archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](s, 4, 0)).MulWidenLo(mul).Add(off).StoreArray((*[8]uint16)(o[32:40]))
		archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](s, 5, 0)).MulWidenLo(mul).Add(off).StoreArray((*[8]uint16)(o[40:48]))
		archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](s, 6, 0)).MulWidenLo(mul).Add(off).StoreArray((*[8]uint16)(o[48:56]))
		archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](s, 7, 0)).MulWidenLo(mul).Add(off).StoreArray((*[8]uint16)(o[56:64]))
	}
	for ; r < h; r++ {
		if !sg.Fits(r, 0, 16, 1) || 8*r+8 > len(out) {
			return
		}
		archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](sg.Sub(r, 0, 16, 1), 0, 0)).MulWidenLo(mul).Add(off).StoreArray((*[8]uint16)(out[8*r : 8*r+8]))
	}
}
