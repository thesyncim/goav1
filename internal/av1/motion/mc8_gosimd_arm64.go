// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package motion

import (
	"encoding/binary"
	"simd/archsimd"

	"github.com/thesyncim/goav1/internal/av1/frame"
)

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
}

var mc8Phases [mc8KindCount][16]mc8Phase

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
				}
			}
		}
		mc8SixTap[kind] = six
	}
}

// mc8Kind mirrors interpKernel's table selection.
func mc8Kind(f InterpFilter, blockSize int) int {
	if blockSize <= 4 && f != InterpBilinear {
		if f == InterpEightTapSmooth {
			return mc8KindSmooth4
		}
		return mc8KindRegular4
	}
	switch f {
	case InterpEightTapRegular:
		return mc8KindRegular
	case InterpEightTapSmooth:
		return mc8KindSmooth
	case InterpMultiTapSharp:
		return mc8KindSharp
	default:
		return mc8KindBilinear
	}
}

// mc8Source resolves the six-tap source window of a w x h block whose
// first output sample reads ref at (refX, refY). halo selects the filtered
// axes. The window is returned as a slice from its top-left tap sample plus a
// stride; when any tap falls outside the plane the clamped window is first
// materialized in edge (dav1d src/recon_tmpl.c mc() emu_edge). Every kernel
// loads 16 bytes per eight output columns, so the window stays readable
// for max(w+8, 16) bytes from its left edge on every row. ok is false only when the
// reference plane itself is unusable.
func mc8Source(ref *frame.Plane, refX, refY, w, h int, hx, hy bool, edge *[emuEdgeStride * emuEdgeRows]byte) (src []byte, stride int, ok bool) {
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
		return nil, 0, false
	}
	if x0 >= 0 && y0 >= 0 && x0+bw <= ref.Width && y0+bh <= ref.Height {
		start := y0*ref.Stride + x0
		ext := (bh-1)*ref.Stride + max(w+8, 16)
		if start+ext <= len(ref.Pix) && ext <= mc8WindowMax {
			return ref.Pix[start : start+ext], ref.Stride, true
		}
	}
	if len(ref.Pix) < (ref.Height-1)*ref.Stride+ref.Width {
		return nil, 0, false
	}
	// The edge window holds 135 rows of 136 bytes; bh <= 133 and the
	// kernels read at most max(w+8, 16) <= 136 bytes per row.
	emuEdge8(bw, bh, *ref, x0, y0, edge[:], emuEdgeStride)
	return edge[:], emuEdgeStride, true
}

// Put8 is dav1d's mc() put for an unscaled 8-bit single-reference block:
// the copy (integer vector) or the put kernels for the block's filters, from
// the reference origin (refX, refY) and Q4 subpel offsets (subX, subY). It
// returns false, with nothing written, for shapes it does not cover; the
// caller then runs the general path.
func Put8(dst *frame.Plane, ref *frame.Plane, dstX, dstY, refX, refY, w, h, filterW, filterH, subX, subY int, filters InterpFilters, scratch *ConvolveScratch) bool {
	if subX|subY == 0 {
		return mc8CopyFast(dst, ref, dstX, dstY, refX, refY, w, h)
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
	if dstX < 0 || dstY < 0 || dstX+w > dst.Width || dstY+h > dst.Height || dst.Stride < dst.Width {
		return false
	}
	dOff := dstY*dst.Stride + dstX
	dExt := (h-1)*dst.Stride + w
	if dOff+dExt > len(dst.Pix) || dExt > mc8WindowMax {
		return false
	}
	src, stride, ok := mc8Source(ref, refX, refY, w, h, subX != 0, subY != 0, &scratch.edge)
	if !ok {
		return false
	}
	d := dst.Pix[dOff : dOff+dExt]
	if w == 4 {
		switch {
		case subX != 0 && subY != 0:
			im := scratch.imHBD[:]
			mc8H6ToI32W4(src, stride, h+5, &mc8Phases[kx][subX], im)
			mc8PutV6FromI32W4(im, h, &mc8Phases[ky][subY], d, dst.Stride)
		case subX != 0:
			mc8PutH6W4(src, stride, h, &mc8Phases[kx][subX], d, dst.Stride)
		default:
			mc8PutV6W4(src, stride, h, &mc8Phases[ky][subY], d, dst.Stride)
		}
		return true
	}
	switch {
	case subX != 0 && subY != 0:
		im := scratch.imHBD[:]
		mc8H6ToI32(src, stride, w, h+5, &mc8Phases[kx][subX], im)
		mc8PutV6FromI32(im, w, h, &mc8Phases[ky][subY], d, dst.Stride)
	case subX != 0:
		mc8PutH6(src, stride, w, h, &mc8Phases[kx][subX], d, dst.Stride)
	default:
		mc8PutV6(src, stride, w, h, &mc8Phases[ky][subY], d, dst.Stride)
	}
	return true
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
	src, stride, ok := mc8Source(ref, refX, refY, 4, h, subX != 0, subY != 0, &scratch.edge)
	if !ok {
		return false
	}
	var tmp [4 * mc8W2MaxH]byte
	switch {
	case subX != 0 && subY != 0:
		im := scratch.imHBD[:]
		mc8H6ToI32W4(src, stride, h+5, &mc8Phases[kx][subX], im)
		mc8PutV6FromI32W4(im, h, &mc8Phases[ky][subY], tmp[:], 4)
	case subX != 0:
		mc8PutH6W4(src, stride, h, &mc8Phases[kx][subX], tmp[:], 4)
	default:
		mc8PutV6W4(src, stride, h, &mc8Phases[ky][subY], tmp[:], 4)
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
	dOff := dstY*dst.Stride + dstX
	sOff := refY*ref.Stride + refX
	dExt, sExt := (h-1)*dst.Stride+w, (h-1)*ref.Stride+w
	if dOff+dExt > len(dst.Pix) || sOff+sExt > len(ref.Pix) || max(dExt, sExt) > mc8WindowMax {
		return false
	}
	mc8Copy(dst.Pix[dOff:dOff+dExt], dst.Stride, ref.Pix[sOff:sOff+sExt], ref.Stride, w, h)
	return true
}

// mc8Copy copies h rows of w bytes (w a multiple of 4); each row is one
// window of each slice.
func mc8Copy(dst []byte, dstStride int, src []byte, srcStride int, w, h int) {
	ss, ds := uint32(srcStride), uint32(dstStride)
	var s, d uint32
	switch {
	case w == 16:
		for r := 0; r < h; r++ {
			archsimd.LoadUint8x16Array(mc8W16(src, s)).StoreArray(mc8W16(dst, d))
			s, d = s+ss, d+ds
		}
	case w == 32:
		for r := 0; r < h; r++ {
			*mc8W32(dst, d) = *mc8W32(src, s)
			s, d = s+ss, d+ds
		}
	case w == 64:
		for r := 0; r < h; r++ {
			*mc8W64(dst, d) = *mc8W64(src, s)
			s, d = s+ss, d+ds
		}
	case w == 128:
		for r := 0; r < h; r++ {
			*mc8W128(dst, d) = *mc8W128(src, s)
			s, d = s+ss, d+ds
		}
	case w&31 == 0:
		for r := 0; r < h; r++ {
			for c := uint32(0); c < uint32(w); c += 32 {
				sw, dw := mc8W32(src, s+c), mc8W32(dst, d+c)
				v0 := archsimd.LoadUint8x16Array((*[16]uint8)(sw[0:16]))
				v1 := archsimd.LoadUint8x16Array((*[16]uint8)(sw[16:32]))
				v0.StoreArray((*[16]uint8)(dw[0:16]))
				v1.StoreArray((*[16]uint8)(dw[16:32]))
			}
			s, d = s+ss, d+ds
		}
	case w&15 == 0:
		for r := 0; r < h; r++ {
			for c := uint32(0); c < uint32(w); c += 16 {
				archsimd.LoadUint8x16Array(mc8W16(src, s+c)).StoreArray(mc8W16(dst, d+c))
			}
			s, d = s+ss, d+ds
		}
	case w == 8:
		for r := 0; r < h; r++ {
			*mc8W8(dst, d) = *mc8W8(src, s)
			s, d = s+ss, d+ds
		}
	default:
		for r := 0; r < h; r++ {
			for c := uint32(0); c < uint32(w); c += 4 {
				*mc8W4(dst, d+c) = *mc8W4(src, s+c)
			}
			s, d = s+ss, d+ds
		}
	}
}

// Window helpers. Offsets are uint32 so o+N cannot overflow and each window
// costs one compare; the entries keep every window slice below 1<<32
// elements (mc8WindowMax), so the offsets never wrap.
func mc8W4(s []byte, o uint32) *[4]byte        { return (*[4]byte)(s[uint(o) : uint(o)+4]) }
func mc8W8(s []byte, o uint32) *[8]byte        { return (*[8]byte)(s[uint(o) : uint(o)+8]) }
func mc8W16(s []byte, o uint32) *[16]byte      { return (*[16]byte)(s[uint(o) : uint(o)+16]) }
func mc8W24(s []byte, o uint32) *[24]byte      { return (*[24]byte)(s[uint(o) : uint(o)+24]) }
func mc8W128(s []byte, o uint32) *[128]byte    { return (*[128]byte)(s[uint(o) : uint(o)+128]) }
func mc8W64(s []byte, o uint32) *[64]byte      { return (*[64]byte)(s[uint(o) : uint(o)+64]) }
func mc8W32(s []byte, o uint32) *[32]byte      { return (*[32]byte)(s[uint(o) : uint(o)+32]) }
func mc8WU4(s []uint16, o uint32) *[4]uint16   { return (*[4]uint16)(s[uint(o) : uint(o)+4]) }
func mc8WU8(s []uint16, o uint32) *[8]uint16   { return (*[8]uint16)(s[uint(o) : uint(o)+8]) }
func mc8WU16(s []uint16, o uint32) *[16]uint16 { return (*[16]uint16)(s[uint(o) : uint(o)+16]) }
func mc8WI4(s []int32, o uint32) *[4]int32     { return (*[4]int32)(s[uint(o) : uint(o)+4]) }
func mc8WI8(s []int32, o uint32) *[8]int32     { return (*[8]int32)(s[uint(o) : uint(o)+8]) }

// mc8WindowMax bounds the window slices handed to the kernels (see mc8W16).
const mc8WindowMax = 1<<32 - 1

// mc8PrepFast fills a w x h CONV_BUF (row stride w) with the dav1d-style prep
// kernels. It returns false for shapes it does not cover.
func mc8PrepFast(out []uint16, ref *frame.Plane, refX, refY, w, h, filterW, filterH, subX, subY int, filters InterpFilters, scratch *CompoundConvolveScratch) bool {
	if (w&7 != 0 && w != 4) || w > maxBlockSize || h <= 0 || h > maxBlockSize || scratch == nil || len(out) < w*h {
		return false
	}
	kx := mc8Kind(filters.X, filterW)
	ky := mc8Kind(filters.Y, filterH)
	if mc8NeedsEightTaps(subX, subY, kx, ky) {
		return w&7 == 0 && mc8Prep8Tap(out[:w*h], ref, refX, refY, w, h, subX, subY, kx, ky, scratch)
	}
	src, stride, ok := mc8Source(ref, refX, refY, w, h, subX != 0, subY != 0, &scratch.edge)
	if !ok {
		return false
	}
	o := out[:w*h]
	if w == 4 {
		switch {
		case subX != 0 && subY != 0:
			im := scratch.im[:]
			mc8H6ToI32W4(src, stride, h+5, &mc8Phases[kx][subX], im)
			mc8PrepV6FromI32W4(im, h, &mc8Phases[ky][subY], o, mc8PrepV6Rnd)
		case subX != 0:
			mc8PrepH6W4(src, stride, h, &mc8Phases[kx][subX], o)
		case subY != 0:
			mc8PrepV6W4(src, stride, h, &mc8Phases[ky][subY], o)
		default:
			mc8PrepCopyW4(src, stride, h, o)
		}
		return true
	}
	switch {
	case subX != 0 && subY != 0:
		im := scratch.im[:]
		mc8H6ToI32(src, stride, w, h+5, &mc8Phases[kx][subX], im)
		mc8PrepV6FromI32S(im, w, h, &mc8Phases[ky][subY], o, mc8PrepV6Rnd)
	case subX != 0:
		mc8PrepH6(src, stride, w, h, &mc8Phases[kx][subX], o)
	case subY != 0:
		mc8PrepV6(src, stride, w, h, &mc8Phases[ky][subY], o)
	default:
		mc8PrepCopy(src, stride, w, h, o)
	}
	return true
}

// mc8LoadWiden widens the 16 bytes of p to two uint16 vectors, returned as
// bytes for the tap shifts; the six-tap horizontal sums read lanes 0..12.
func mc8LoadWiden(p *[16]uint8) (archsimd.Uint8x16, archsimd.Uint8x16) {
	a := archsimd.LoadUint8x16Array(p)
	return a.ExtendLo8ToUint16().ReshapeToUint8s(), a.HiToLo().ExtendLo8ToUint16().ReshapeToUint8s()
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

// mc8H6ToI32 writes rows x w rounded horizontal sums ((s + 2) >> 2) as int32
// with row stride w.
func mc8H6ToI32(src []byte, srcStride, w, rows int, ph *mc8Phase, im []int32) {
	c0 := archsimd.LoadInt16x8Array(&ph.h[0])
	c1 := archsimd.LoadInt16x8Array(&ph.h[1])
	c2 := archsimd.LoadInt16x8Array(&ph.h[2])
	c3 := archsimd.LoadInt16x8Array(&ph.h[3])
	c4 := archsimd.LoadInt16x8Array(&ph.h[4])
	c5 := archsimd.LoadInt16x8Array(&ph.h[5])
	rnd := archsimd.BroadcastInt16x8(2)
	sh := archsimd.BroadcastInt16x8(-2)
	ss := uint32(srcStride)
	n := rows * 8
	sum6 := func(lo, hi archsimd.Uint8x16) archsimd.Int16x8 {
		sum := mc8I16(lo).MulAdd(c0, rnd)
		sum = mc8I16(hi.ConcatShiftBytesRight(lo, 2)).MulAdd(c1, sum)
		sum = mc8I16(hi.ConcatShiftBytesRight(lo, 4)).MulAdd(c2, sum)
		sum = mc8I16(hi.ConcatShiftBytesRight(lo, 6)).MulAdd(c3, sum)
		sum = mc8I16(hi.ConcatShiftBytesRight(lo, 8)).MulAdd(c4, sum)
		sum = mc8I16(hi.ConcatShiftBytesRight(lo, 10)).MulAdd(c5, sum)
		return sum.Shift(sh)
	}
	for c := 0; c < w; c += 8 {
		// Strip c/8 of the strip-major intermediate: its rows are contiguous
		// eight-sample groups, so the stores below are proven in bounds.
		st := im[c*rows : c*rows+n : c*rows+n]
		so := uint32(c)
		q := 0
		for ; q <= len(st)-16; q += 16 {
			lo, hi := mc8LoadWiden(mc8W16(src, so))
			lo1, hi1 := mc8LoadWiden(mc8W16(src, so+ss))
			sum, sum1 := sum6(lo, hi), sum6(lo1, hi1)
			win := (*[16]int32)(st[q : q+16])
			sum.ExtendLo4ToInt32().StoreArray((*[4]int32)(win[0:4]))
			sum.HiToLo().ExtendLo4ToInt32().StoreArray((*[4]int32)(win[4:8]))
			sum1.ExtendLo4ToInt32().StoreArray((*[4]int32)(win[8:12]))
			sum1.HiToLo().ExtendLo4ToInt32().StoreArray((*[4]int32)(win[12:16]))
			so += 2 * ss
		}
		if q <= len(st)-8 {
			lo, hi := mc8LoadWiden(mc8W16(src, so))
			sum := sum6(lo, hi)
			win := (*[8]int32)(st[q : q+8])
			sum.ExtendLo4ToInt32().StoreArray((*[4]int32)(win[0:4]))
			sum.HiToLo().ExtendLo4ToInt32().StoreArray((*[4]int32)(win[4:8]))
		}
	}
}

// mc8LoadI32x8 loads the eight int32 intermediate samples at p as two halves.
func mc8LoadI32x8(p *[8]int32) (archsimd.Int32x4, archsimd.Int32x4) {
	return archsimd.LoadInt32x4Array((*[4]int32)(p[0:4])), archsimd.LoadInt32x4Array((*[4]int32)(p[4:8]))
}

// mc8PutV6FromI32 is the put 2D vertical pass over the int32 intermediate.
func mc8PutV6FromI32(im []int32, w, h int, ph *mc8Phase, dst []byte, dstStride int) {
	c0 := archsimd.LoadInt32x4Array(&ph.v[0])
	c1 := archsimd.LoadInt32x4Array(&ph.v[1])
	c2 := archsimd.LoadInt32x4Array(&ph.v[2])
	c3 := archsimd.LoadInt32x4Array(&ph.v[3])
	c4 := archsimd.LoadInt32x4Array(&ph.v[4])
	c5 := archsimd.LoadInt32x4Array(&ph.v[5])
	rnd := archsimd.BroadcastInt32x4(1 << 15)
	ds := uint32(dstStride)
	rows := h + 5
	n := rows * 8
	for c := 0; c < w; c += 8 {
		// Strip c/8 of the strip-major intermediate (rows of eight
		// contiguous samples): every window below is proven in bounds.
		st := im[c*rows : c*rows+n : c*rows+n]
		head := (*[40]int32)(st[0:40])
		x0, y0 := mc8LoadI32x8((*[8]int32)(head[0:8]))
		x1, y1 := mc8LoadI32x8((*[8]int32)(head[8:16]))
		x2, y2 := mc8LoadI32x8((*[8]int32)(head[16:24]))
		x3, y3 := mc8LoadI32x8((*[8]int32)(head[24:32]))
		x4, y4 := mc8LoadI32x8((*[8]int32)(head[32:40]))
		o := uint32(c)
		q := 40
		for ; q <= len(st)-64; q += 64 {
			win := (*[64]int32)(st[q : q+64])
			x5, y5 := mc8LoadI32x8((*[8]int32)(win[0:8]))
			x6, y6 := mc8LoadI32x8((*[8]int32)(win[8:16]))
			x7, y7 := mc8LoadI32x8((*[8]int32)(win[16:24]))
			x8, y8 := mc8LoadI32x8((*[8]int32)(win[24:32]))
			x9, y9 := mc8LoadI32x8((*[8]int32)(win[32:40]))
			x10, y10 := mc8LoadI32x8((*[8]int32)(win[40:48]))
			x11, y11 := mc8LoadI32x8((*[8]int32)(win[48:56]))
			x12, y12 := mc8LoadI32x8((*[8]int32)(win[56:64]))
			u0, u1 := mc8V6PutPair(x0, y0, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, c0, c1, c2, c3, c4, c5, rnd)
			binary.LittleEndian.PutUint64(mc8W8(dst, o)[:], u0)
			binary.LittleEndian.PutUint64(mc8W8(dst, o+1*ds)[:], u1)
			u0, u1 = mc8V6PutPair(x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8, c0, c1, c2, c3, c4, c5, rnd)
			binary.LittleEndian.PutUint64(mc8W8(dst, o+2*ds)[:], u0)
			binary.LittleEndian.PutUint64(mc8W8(dst, o+3*ds)[:], u1)
			u0, u1 = mc8V6PutPair(x4, y4, x5, y5, x6, y6, x7, y7, x8, y8, x9, y9, x10, y10, c0, c1, c2, c3, c4, c5, rnd)
			binary.LittleEndian.PutUint64(mc8W8(dst, o+4*ds)[:], u0)
			binary.LittleEndian.PutUint64(mc8W8(dst, o+5*ds)[:], u1)
			u0, u1 = mc8V6PutPair(x6, y6, x7, y7, x8, y8, x9, y9, x10, y10, x11, y11, x12, y12, c0, c1, c2, c3, c4, c5, rnd)
			binary.LittleEndian.PutUint64(mc8W8(dst, o+6*ds)[:], u0)
			binary.LittleEndian.PutUint64(mc8W8(dst, o+7*ds)[:], u1)
			o += 8 * ds
			x0, y0, x1, y1, x2, y2, x3, y3, x4, y4 = x8, y8, x9, y9, x10, y10, x11, y11, x12, y12
		}
		for ; q <= len(st)-16; q += 16 {
			win := (*[16]int32)(st[q : q+16])
			x5, y5 := mc8LoadI32x8((*[8]int32)(win[0:8]))
			x6, y6 := mc8LoadI32x8((*[8]int32)(win[8:16]))
			u0, u1 := mc8V6PutPair(x0, y0, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, c0, c1, c2, c3, c4, c5, rnd)
			binary.LittleEndian.PutUint64(mc8W8(dst, o)[:], u0)
			binary.LittleEndian.PutUint64(mc8W8(dst, o+ds)[:], u1)
			o += 2 * ds
			x0, y0, x1, y1, x2, y2, x3, y3, x4, y4 = x2, y2, x3, y3, x4, y4, x5, y5, x6, y6
		}
		if q <= len(st)-8 {
			x5, y5 := mc8LoadI32x8((*[8]int32)(st[q : q+8]))
			u0, _ := mc8V6PutPair(x0, y0, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x5, y5, c0, c1, c2, c3, c4, c5, rnd)
			binary.LittleEndian.PutUint64(mc8W8(dst, o)[:], u0)
		}
	}
}

// mc8V6PutPair is the put 2D vertical chain of output rows r and r+1 from
// the seven intermediate rows (x, y halves) r..r+6, saturated to bytes.
func mc8V6PutPair(x0, y0, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, c0, c1, c2, c3, c4, c5, rnd archsimd.Int32x4) (uint64, uint64) {
	a0 := x0.MulAdd(c0, rnd)
	b0 := y0.MulAdd(c0, rnd)
	a1 := x1.MulAdd(c0, rnd)
	b1 := y1.MulAdd(c0, rnd)
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

// mc8PrepV6FromI32S is mc8PrepV6FromI32 over the strip-major intermediate
// of mc8H6ToI32.
func mc8PrepV6FromI32S(im []int32, w, h int, ph *mc8Phase, out []uint16, rndConst int32) {
	c0 := archsimd.LoadInt32x4Array(&ph.p[0])
	c1 := archsimd.LoadInt32x4Array(&ph.p[1])
	c2 := archsimd.LoadInt32x4Array(&ph.p[2])
	c3 := archsimd.LoadInt32x4Array(&ph.p[3])
	c4 := archsimd.LoadInt32x4Array(&ph.p[4])
	c5 := archsimd.LoadInt32x4Array(&ph.p[5])
	rnd := archsimd.BroadcastInt32x4(rndConst)
	ds := uint32(w)
	rows := h + 5
	n := rows * 8
	for c := 0; c < w; c += 8 {
		// Strip c/8 of the strip-major intermediate (rows of eight
		// contiguous samples): every window below is proven in bounds.
		st := im[c*rows : c*rows+n : c*rows+n]
		head := (*[40]int32)(st[0:40])
		x0, y0 := mc8LoadI32x8((*[8]int32)(head[0:8]))
		x1, y1 := mc8LoadI32x8((*[8]int32)(head[8:16]))
		x2, y2 := mc8LoadI32x8((*[8]int32)(head[16:24]))
		x3, y3 := mc8LoadI32x8((*[8]int32)(head[24:32]))
		x4, y4 := mc8LoadI32x8((*[8]int32)(head[32:40]))
		o := uint32(c)
		q := 40
		for ; q <= len(st)-64; q += 64 {
			win := (*[64]int32)(st[q : q+64])
			x5, y5 := mc8LoadI32x8((*[8]int32)(win[0:8]))
			x6, y6 := mc8LoadI32x8((*[8]int32)(win[8:16]))
			x7, y7 := mc8LoadI32x8((*[8]int32)(win[16:24]))
			x8, y8 := mc8LoadI32x8((*[8]int32)(win[24:32]))
			x9, y9 := mc8LoadI32x8((*[8]int32)(win[32:40]))
			x10, y10 := mc8LoadI32x8((*[8]int32)(win[40:48]))
			x11, y11 := mc8LoadI32x8((*[8]int32)(win[48:56]))
			x12, y12 := mc8LoadI32x8((*[8]int32)(win[56:64]))
			u0, u1 := mc8V6PrepPair(x0, y0, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, c0, c1, c2, c3, c4, c5, rnd)
			u0.StoreArray(mc8WU8(out, o))
			u1.StoreArray(mc8WU8(out, o+1*ds))
			u0, u1 = mc8V6PrepPair(x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8, c0, c1, c2, c3, c4, c5, rnd)
			u0.StoreArray(mc8WU8(out, o+2*ds))
			u1.StoreArray(mc8WU8(out, o+3*ds))
			u0, u1 = mc8V6PrepPair(x4, y4, x5, y5, x6, y6, x7, y7, x8, y8, x9, y9, x10, y10, c0, c1, c2, c3, c4, c5, rnd)
			u0.StoreArray(mc8WU8(out, o+4*ds))
			u1.StoreArray(mc8WU8(out, o+5*ds))
			u0, u1 = mc8V6PrepPair(x6, y6, x7, y7, x8, y8, x9, y9, x10, y10, x11, y11, x12, y12, c0, c1, c2, c3, c4, c5, rnd)
			u0.StoreArray(mc8WU8(out, o+6*ds))
			u1.StoreArray(mc8WU8(out, o+7*ds))
			o += 8 * ds
			x0, y0, x1, y1, x2, y2, x3, y3, x4, y4 = x8, y8, x9, y9, x10, y10, x11, y11, x12, y12
		}
		for ; q <= len(st)-16; q += 16 {
			win := (*[16]int32)(st[q : q+16])
			x5, y5 := mc8LoadI32x8((*[8]int32)(win[0:8]))
			x6, y6 := mc8LoadI32x8((*[8]int32)(win[8:16]))
			u0, u1 := mc8V6PrepPair(x0, y0, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, c0, c1, c2, c3, c4, c5, rnd)
			u0.StoreArray(mc8WU8(out, o))
			u1.StoreArray(mc8WU8(out, o+ds))
			o += 2 * ds
			x0, y0, x1, y1, x2, y2, x3, y3, x4, y4 = x2, y2, x3, y3, x4, y4, x5, y5, x6, y6
		}
		if q <= len(st)-8 {
			x5, y5 := mc8LoadI32x8((*[8]int32)(st[q : q+8]))
			u0, _ := mc8V6PrepPair(x0, y0, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x5, y5, c0, c1, c2, c3, c4, c5, rnd)
			u0.StoreArray(mc8WU8(out, o))
		}
	}
}

// mc8V6PrepPair is the prep 2D vertical chain of CONV_BUF rows r and r+1
// from the seven intermediate rows r..r+6.
func mc8V6PrepPair(x0, y0, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, c0, c1, c2, c3, c4, c5, rnd archsimd.Int32x4) (archsimd.Uint16x8, archsimd.Uint16x8) {
	a0 := x0.MulAdd(c0, rnd)
	b0 := y0.MulAdd(c0, rnd)
	a1 := x1.MulAdd(c0, rnd)
	b1 := y1.MulAdd(c0, rnd)
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
	return a0.ToBits().ReshapeToUint16s().ConcatOdd(b0.ToBits().ReshapeToUint16s()), a1.ToBits().ReshapeToUint16s().ConcatOdd(b1.ToBits().ReshapeToUint16s())
}

// mc8PutH6 is the put horizontal-only pass: clip((s + 34) >> 6).
func mc8PutH6(src []byte, srcStride, w, h int, ph *mc8Phase, dst []byte, dstStride int) {
	c0 := archsimd.LoadInt16x8Array(&ph.h[0])
	c1 := archsimd.LoadInt16x8Array(&ph.h[1])
	c2 := archsimd.LoadInt16x8Array(&ph.h[2])
	c3 := archsimd.LoadInt16x8Array(&ph.h[3])
	c4 := archsimd.LoadInt16x8Array(&ph.h[4])
	c5 := archsimd.LoadInt16x8Array(&ph.h[5])
	rnd := archsimd.BroadcastInt16x8(34)
	sh := archsimd.BroadcastInt16x8(-6)
	ss, ds := uint32(srcStride), uint32(dstStride)
	if w&15 == 0 {
		// Sixteen columns per window: one 24-byte source window per row and
		// one 16-byte store.
		for r := 0; r < h; r++ {
			so, do := uint32(r)*ss, uint32(r)*ds
			for c := uint32(0); c < uint32(w); c += 16 {
				win := mc8W24(src, so+c)
				a := mc8H6Sum((*[16]uint8)(win[0:16]), c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8()
				b := mc8H6Sum((*[16]uint8)(win[8:24]), c0, c1, c2, c3, c4, c5, rnd).Shift(sh).SaturateToUint8()
				a.ReshapeToUint64s().InterleaveLo(b.ReshapeToUint64s()).ReshapeToUint8s().StoreArray(mc8W16(dst, do+c))
			}
		}
		return
	}
	var so, do uint32
	r := 0
	for ; r+2 <= h; r += 2 {
		for c := uint32(0); c < uint32(w); c += 8 {
			lo, hi := mc8LoadWiden(mc8W16(src, so+c))
			lo1, hi1 := mc8LoadWiden(mc8W16(src, so+ss+c))
			sum := mc8I16(lo).MulAdd(c0, rnd)
			sum1 := mc8I16(lo1).MulAdd(c0, rnd)
			sum = mc8I16(hi.ConcatShiftBytesRight(lo, 2)).MulAdd(c1, sum)
			sum1 = mc8I16(hi1.ConcatShiftBytesRight(lo1, 2)).MulAdd(c1, sum1)
			sum = mc8I16(hi.ConcatShiftBytesRight(lo, 4)).MulAdd(c2, sum)
			sum1 = mc8I16(hi1.ConcatShiftBytesRight(lo1, 4)).MulAdd(c2, sum1)
			sum = mc8I16(hi.ConcatShiftBytesRight(lo, 6)).MulAdd(c3, sum)
			sum1 = mc8I16(hi1.ConcatShiftBytesRight(lo1, 6)).MulAdd(c3, sum1)
			sum = mc8I16(hi.ConcatShiftBytesRight(lo, 8)).MulAdd(c4, sum)
			sum1 = mc8I16(hi1.ConcatShiftBytesRight(lo1, 8)).MulAdd(c4, sum1)
			sum = mc8I16(hi.ConcatShiftBytesRight(lo, 10)).MulAdd(c5, sum)
			sum1 = mc8I16(hi1.ConcatShiftBytesRight(lo1, 10)).MulAdd(c5, sum1)
			binary.LittleEndian.PutUint64(mc8W8(dst, do+c)[:], mc8Lo64(sum.Shift(sh).SaturateToUint8()))
			binary.LittleEndian.PutUint64(mc8W8(dst, do+ds+c)[:], mc8Lo64(sum1.Shift(sh).SaturateToUint8()))
		}
		so, do = so+2*ss, do+2*ds
	}
	if r < h {
		for c := uint32(0); c < uint32(w); c += 8 {
			lo, hi := mc8LoadWiden(mc8W16(src, so+c))
			sum := mc8I16(lo).MulAdd(c0, rnd)
			sum = mc8I16(hi.ConcatShiftBytesRight(lo, 2)).MulAdd(c1, sum)
			sum = mc8I16(hi.ConcatShiftBytesRight(lo, 4)).MulAdd(c2, sum)
			sum = mc8I16(hi.ConcatShiftBytesRight(lo, 6)).MulAdd(c3, sum)
			sum = mc8I16(hi.ConcatShiftBytesRight(lo, 8)).MulAdd(c4, sum)
			sum = mc8I16(hi.ConcatShiftBytesRight(lo, 10)).MulAdd(c5, sum)
			binary.LittleEndian.PutUint64(mc8W8(dst, do+c)[:], mc8Lo64(sum.Shift(sh).SaturateToUint8()))
		}
	}
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

// mc8PrepH6 is the prep horizontal-only pass: ((s + 2) >> 2) + 6144.
func mc8PrepH6(src []byte, srcStride, w, h int, ph *mc8Phase, out []uint16) {
	c0 := archsimd.LoadInt16x8Array(&ph.h[0])
	c1 := archsimd.LoadInt16x8Array(&ph.h[1])
	c2 := archsimd.LoadInt16x8Array(&ph.h[2])
	c3 := archsimd.LoadInt16x8Array(&ph.h[3])
	c4 := archsimd.LoadInt16x8Array(&ph.h[4])
	c5 := archsimd.LoadInt16x8Array(&ph.h[5])
	rnd := archsimd.BroadcastInt16x8(2 + 6144<<2)
	sh := archsimd.BroadcastInt16x8(-2)
	ss, ws := uint32(srcStride), uint32(w)
	if w&15 == 0 {
		// Sixteen columns per window: one 24-byte source window per row and
		// one 16-sample CONV_BUF window.
		var so, oo uint32
		for r := 0; r < h; r++ {
			for c := uint32(0); c < ws; c += 16 {
				win := mc8W24(src, so+c)
				o := mc8WU16(out, oo+c)
				// The offset sums lie in [0, 2^16): shift them as unsigned.
				mc8H6Sum((*[16]uint8)(win[0:16]), c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray((*[8]uint16)(o[0:8]))
				mc8H6Sum((*[16]uint8)(win[8:24]), c0, c1, c2, c3, c4, c5, rnd).ToBits().Shift(sh).StoreArray((*[8]uint16)(o[8:16]))
			}
			so, oo = so+ss, oo+ws
		}
		return
	}
	var so, oo uint32
	r := 0
	for ; r+2 <= h; r += 2 {
		for c := uint32(0); c < ws; c += 8 {
			lo, hi := mc8LoadWiden(mc8W16(src, so+c))
			lo1, hi1 := mc8LoadWiden(mc8W16(src, so+ss+c))
			sum := mc8I16(lo).MulAdd(c0, rnd)
			sum1 := mc8I16(lo1).MulAdd(c0, rnd)
			sum = mc8I16(hi.ConcatShiftBytesRight(lo, 2)).MulAdd(c1, sum)
			sum1 = mc8I16(hi1.ConcatShiftBytesRight(lo1, 2)).MulAdd(c1, sum1)
			sum = mc8I16(hi.ConcatShiftBytesRight(lo, 4)).MulAdd(c2, sum)
			sum1 = mc8I16(hi1.ConcatShiftBytesRight(lo1, 4)).MulAdd(c2, sum1)
			sum = mc8I16(hi.ConcatShiftBytesRight(lo, 6)).MulAdd(c3, sum)
			sum1 = mc8I16(hi1.ConcatShiftBytesRight(lo1, 6)).MulAdd(c3, sum1)
			sum = mc8I16(hi.ConcatShiftBytesRight(lo, 8)).MulAdd(c4, sum)
			sum1 = mc8I16(hi1.ConcatShiftBytesRight(lo1, 8)).MulAdd(c4, sum1)
			sum = mc8I16(hi.ConcatShiftBytesRight(lo, 10)).MulAdd(c5, sum)
			sum1 = mc8I16(hi1.ConcatShiftBytesRight(lo1, 10)).MulAdd(c5, sum1)
			// The offset sums lie in [0, 2^16): shift them as unsigned.
			sum.ToBits().Shift(sh).StoreArray(mc8WU8(out, oo+c))
			sum1.ToBits().Shift(sh).StoreArray(mc8WU8(out, oo+ws+c))
		}
		so, oo = so+2*ss, oo+2*ws
	}
	if r < h {
		for c := uint32(0); c < ws; c += 8 {
			lo, hi := mc8LoadWiden(mc8W16(src, so+c))
			sum := mc8I16(lo).MulAdd(c0, rnd)
			sum = mc8I16(hi.ConcatShiftBytesRight(lo, 2)).MulAdd(c1, sum)
			sum = mc8I16(hi.ConcatShiftBytesRight(lo, 4)).MulAdd(c2, sum)
			sum = mc8I16(hi.ConcatShiftBytesRight(lo, 6)).MulAdd(c3, sum)
			sum = mc8I16(hi.ConcatShiftBytesRight(lo, 8)).MulAdd(c4, sum)
			sum = mc8I16(hi.ConcatShiftBytesRight(lo, 10)).MulAdd(c5, sum)
			sum.ToBits().Shift(sh).StoreArray(mc8WU8(out, oo+c))
		}
	}
}

// mc8PutV6 is the put vertical-only pass: clip((s + 32) >> 6). Each eight
// column strip keeps its source rows resident, loads two new rows per two
// output rows and runs the two rows' tap chains interleaved.
func mc8PutV6(src []byte, srcStride, w, h int, ph *mc8Phase, dst []byte, dstStride int) {
	c0 := archsimd.LoadInt16x8Array(&ph.h[0])
	c1 := archsimd.LoadInt16x8Array(&ph.h[1])
	c2 := archsimd.LoadInt16x8Array(&ph.h[2])
	c3 := archsimd.LoadInt16x8Array(&ph.h[3])
	c4 := archsimd.LoadInt16x8Array(&ph.h[4])
	c5 := archsimd.LoadInt16x8Array(&ph.h[5])
	rnd := archsimd.BroadcastInt16x8(32)
	sh := archsimd.BroadcastInt16x8(-6)
	ss, ds := uint32(srcStride), uint32(dstStride)
	if w&15 == 0 {
		mc8PutV6W16(src, ss, w, h, c0, c1, c2, c3, c4, c5, dst, ds)
		return
	}
	for c := uint32(0); c < uint32(w); c += 8 {
		o := c
		r0 := mc8LoadRow16(mc8W16(src, o))
		r1 := mc8LoadRow16(mc8W16(src, o+ss))
		r2 := mc8LoadRow16(mc8W16(src, o+2*ss))
		r3 := mc8LoadRow16(mc8W16(src, o+3*ss))
		o += 4 * ss
		r4 := mc8LoadRow16(mc8W16(src, o))
		d := c
		r := 0
		for ; r+2 <= h; r += 2 {
			o += ss
			r5 := mc8LoadRow16(mc8W16(src, o))
			o += ss
			r6 := mc8LoadRow16(mc8W16(src, o))
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
			binary.LittleEndian.PutUint64(mc8W8(dst, d)[:], mc8Lo64(a.Shift(sh).SaturateToUint8()))
			binary.LittleEndian.PutUint64(mc8W8(dst, d+ds)[:], mc8Lo64(b.Shift(sh).SaturateToUint8()))
			d += 2 * ds
			r0, r1, r2, r3, r4 = r2, r3, r4, r5, r6
		}
		if r < h {
			r5 := mc8LoadRow16(mc8W16(src, o+ss))
			a := r0.MulAdd(c0, rnd)
			a = r1.MulAdd(c1, a)
			a = r2.MulAdd(c2, a)
			a = r3.MulAdd(c3, a)
			a = r4.MulAdd(c4, a)
			a = r5.MulAdd(c5, a)
			binary.LittleEndian.PutUint64(mc8W8(dst, d)[:], mc8Lo64(a.Shift(sh).SaturateToUint8()))
		}
	}
}

// mc8PutV6W16 is mc8PutV6 over 16-column strips (dav1d's w16 v path): each
// source row is one 16-byte window widened to two halves, and each output
// row one 16-byte store.
func mc8PutV6W16(src []byte, ss uint32, w, h int, c0, c1, c2, c3, c4, c5 archsimd.Int16x8, dst []byte, ds uint32) {
	rnd := archsimd.BroadcastInt16x8(32)
	sh := archsimd.BroadcastInt16x8(-6)
	ld := func(o uint32) (archsimd.Int16x8, archsimd.Int16x8) {
		a := archsimd.LoadUint8x16Array(mc8W16(src, o))
		return a.ExtendLo8ToUint16().BitsToInt16(), a.HiToLo().ExtendLo8ToUint16().BitsToInt16()
	}
	pack := func(lo, hi archsimd.Int16x8) archsimd.Uint8x16 {
		return lo.Shift(sh).SaturateToUint8().ReshapeToUint64s().InterleaveLo(hi.Shift(sh).SaturateToUint8().ReshapeToUint64s()).ReshapeToUint8s()
	}
	for c := uint32(0); c < uint32(w); c += 16 {
		o := c
		l0, h0 := ld(o)
		l1, h1 := ld(o + ss)
		l2, h2 := ld(o + 2*ss)
		l3, h3 := ld(o + 3*ss)
		o += 4 * ss
		l4, h4 := ld(o)
		d := c
		for r := 0; r < h; r++ {
			o += ss
			l5, h5 := ld(o)
			a := l0.MulAdd(c0, rnd)
			b := h0.MulAdd(c0, rnd)
			a = l1.MulAdd(c1, a)
			b = h1.MulAdd(c1, b)
			a = l2.MulAdd(c2, a)
			b = h2.MulAdd(c2, b)
			a = l3.MulAdd(c3, a)
			b = h3.MulAdd(c3, b)
			a = l4.MulAdd(c4, a)
			b = h4.MulAdd(c4, b)
			a = l5.MulAdd(c5, a)
			b = h5.MulAdd(c5, b)
			pack(a, b).StoreArray(mc8W16(dst, d))
			d += ds
			l0, h0, l1, h1, l2, h2, l3, h3, l4, h4 = l1, h1, l2, h2, l3, h3, l4, h4, l5, h5
		}
	}
}

// mc8PrepV6 is the prep vertical-only pass: ((s + 2) >> 2) + 6144.
func mc8PrepV6(src []byte, srcStride, w, h int, ph *mc8Phase, out []uint16) {
	c0 := archsimd.LoadInt16x8Array(&ph.h[0])
	c1 := archsimd.LoadInt16x8Array(&ph.h[1])
	c2 := archsimd.LoadInt16x8Array(&ph.h[2])
	c3 := archsimd.LoadInt16x8Array(&ph.h[3])
	c4 := archsimd.LoadInt16x8Array(&ph.h[4])
	c5 := archsimd.LoadInt16x8Array(&ph.h[5])
	rnd := archsimd.BroadcastInt16x8(2 + 6144<<2)
	sh := archsimd.BroadcastInt16x8(-2)
	ss, ws := uint32(srcStride), uint32(w)
	for c := uint32(0); c < ws; c += 8 {
		so := c
		r0 := mc8LoadRow16(mc8W16(src, so))
		r1 := mc8LoadRow16(mc8W16(src, so+ss))
		r2 := mc8LoadRow16(mc8W16(src, so+2*ss))
		r3 := mc8LoadRow16(mc8W16(src, so+3*ss))
		so += 4 * ss
		r4 := mc8LoadRow16(mc8W16(src, so))
		o := c
		r := 0
		for ; r+2 <= h; r += 2 {
			so += ss
			r5 := mc8LoadRow16(mc8W16(src, so))
			so += ss
			r6 := mc8LoadRow16(mc8W16(src, so))
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
			a.ToBits().Shift(sh).StoreArray(mc8WU8(out, o))
			b.ToBits().Shift(sh).StoreArray(mc8WU8(out, o+ws))
			o += 2 * ws
			r0, r1, r2, r3, r4 = r2, r3, r4, r5, r6
		}
		if r < h {
			r5 := mc8LoadRow16(mc8W16(src, so+ss))
			a := r0.MulAdd(c0, rnd)
			a = r1.MulAdd(c1, a)
			a = r2.MulAdd(c2, a)
			a = r3.MulAdd(c3, a)
			a = r4.MulAdd(c4, a)
			a = r5.MulAdd(c5, a)
			a.ToBits().Shift(sh).StoreArray(mc8WU8(out, o))
		}
	}
}

// mc8PrepCopy is the prep copy: 16 * px + 6144 (UMULL by 16, then the
// offset), one source and one CONV_BUF row window per row.
func mc8PrepCopy(src []byte, srcStride, w, h int, out []uint16) {
	mul := archsimd.BroadcastUint8x16(16)
	mulHi := mul.HiToLo()
	off := archsimd.BroadcastUint16x8(6144)
	ss, ws := uint32(srcStride), uint32(w)
	var so, oo uint32
	if w == 8 {
		for r := 0; r < h; r++ {
			a := archsimd.LoadUint8x16Array(mc8W16(src, so))
			a.MulWidenLo(mul).Add(off).StoreArray(mc8WU8(out, oo))
			so, oo = so+ss, oo+8
		}
		return
	}
	for r := 0; r < h; r++ {
		for c := uint32(0); c < ws; c += 16 {
			a := archsimd.LoadUint8x16Array(mc8W16(src, so+c))
			a.MulWidenLo(mul).Add(off).StoreArray(mc8WU8(out, oo+c))
			if ws-c == 8 {
				break
			}
			a.HiToLo().MulWidenLo(mulHi).Add(off).StoreArray(mc8WU8(out, oo+c+8))
		}
		so, oo = so+ss, oo+ws
	}
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
	if dstX < 0 || dstY < 0 || dstX+w > dst.Width || off+ext > len(dst.Pix) || ext > mc8WindowMax {
		return false
	}
	mc8BlendAvg(dst.Pix[off:off+ext], dst.Stride, a[:w*h], b[:w*h], w, h)
	return true
}

func mc8BlendAvg(dst []byte, dstStride int, a, b []uint16, w, h int) {
	// clamp((a + b - 12272) >> 5, 0, 255) in the shape of dav1d's add +
	// sqrshrun pair: a saturating subtract clamps below at 0, a saturating
	// << 3 clamps floor(u / 32) at 255 in each high byte, and UZP2 gathers
	// those bytes.
	bias := archsimd.BroadcastUint16x8(12272)
	sh := archsimd.BroadcastInt16x8(3)
	avg := func(pa, pb *[8]uint16) archsimd.Uint8x16 {
		return archsimd.LoadUint16x8Array(pa).Add(archsimd.LoadUint16x8Array(pb)).SubSaturated(bias).ShiftSaturated(sh).ReshapeToUint8s()
	}
	// The predictors are contiguous: they are walked as one stream whose
	// loop bound proves each window; only the destination rows are checked.
	n := w * h
	a, b = a[:n:n], b[:n:n]
	ds := uint32(dstStride)
	var d uint32
	switch {
	case w == 4:
		for i := 0; i <= len(a)-8; i += 8 {
			pa, pb := (*[8]uint16)(a[i:i+8]), (*[8]uint16)(b[i:i+8])
			v := avg(pa, pb)
			x := v.ConcatOdd(v).ReshapeToUint64s().GetElem(0)
			binary.LittleEndian.PutUint32(mc8W4(dst, d)[:], uint32(x))
			binary.LittleEndian.PutUint32(mc8W4(dst, d+ds)[:], uint32(x>>32))
			d += 2 * ds
		}
	case w == 8:
		for i := 0; i <= len(a)-8; i += 8 {
			pa, pb := (*[8]uint16)(a[i:i+8]), (*[8]uint16)(b[i:i+8])
			v := avg(pa, pb)
			binary.LittleEndian.PutUint64(mc8W8(dst, d)[:], v.ConcatOdd(v).ReshapeToUint64s().GetElem(0))
			d += ds
		}
	case w&15 == 0:
		ws := uint32(w)
		var c uint32
		for i := 0; i <= len(a)-16; i += 16 {
			pa, pb := (*[16]uint16)(a[i:i+16]), (*[16]uint16)(b[i:i+16])
			v0 := avg((*[8]uint16)(pa[0:8]), (*[8]uint16)(pb[0:8]))
			v1 := avg((*[8]uint16)(pa[8:16]), (*[8]uint16)(pb[8:16]))
			v0.ConcatOdd(v1).StoreArray(mc8W16(dst, d+c))
			if c += 16; c == ws {
				c, d = 0, d+ds
			}
		}
	default: // w = 16k + 8
		ws := uint32(w)
		var c uint32
		for i := 0; i <= len(a)-8; {
			if ws-c == 8 {
				pa, pb := (*[8]uint16)(a[i:i+8]), (*[8]uint16)(b[i:i+8])
				v := avg(pa, pb)
				binary.LittleEndian.PutUint64(mc8W8(dst, d+c)[:], v.ConcatOdd(v).ReshapeToUint64s().GetElem(0))
				i += 8
				c, d = 0, d+ds
				continue
			}
			pa, pb := (*[16]uint16)(a[i:i+16]), (*[16]uint16)(b[i:i+16])
			v0 := avg((*[8]uint16)(pa[0:8]), (*[8]uint16)(pb[0:8]))
			v1 := avg((*[8]uint16)(pa[8:16]), (*[8]uint16)(pb[8:16]))
			v0.ConcatOdd(v1).StoreArray(mc8W16(dst, d+c))
			i += 16
			c += 16
		}
	}
}

// mc8L16U loads eight uint16 lanes from p (len(p) >= 8).
func mc8L16U(p []uint16) archsimd.Uint16x8 {
	return archsimd.LoadUint16x8Array((*[8]uint16)(p))
}
