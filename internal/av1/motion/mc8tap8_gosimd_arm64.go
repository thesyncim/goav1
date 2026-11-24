// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package motion

import (
	"simd/archsimd"

	"github.com/thesyncim/goav1/internal/av1/frame"
)

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
