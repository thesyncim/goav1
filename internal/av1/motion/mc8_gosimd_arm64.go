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
// axes. The window is returned as a pointer to its top-left tap sample plus a
// stride; when any tap falls outside the plane the clamped window is first
// materialized in edge (dav1d src/recon_tmpl.c mc() emu_edge). Every kernel
// loads 16 bytes per eight output columns, so the window must stay readable
// for max(w+8, 16) bytes from its left edge on every row. ok is false only when the
// reference plane itself is unusable.
func mc8Source(ref *frame.Plane, refX, refY, w, h int, hx, hy bool, edge *[emuEdgeStride * emuEdgeRows]byte) (src unsafe.Pointer, stride int, ok bool) {
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
		if start+(bh-1)*ref.Stride+max(w+8, 16) <= len(ref.Pix) {
			return unsafe.Pointer(unsafe.SliceData(ref.Pix[start:])), ref.Stride, true
		}
	}
	if len(ref.Pix) < (ref.Height-1)*ref.Stride+ref.Width {
		return nil, 0, false
	}
	// The edge window holds 135 rows of 136 bytes; bh <= 133 and the
	// kernels read at most max(w+8, 16) <= 136 bytes per row.
	emuEdge8(bw, bh, *ref, x0, y0, edge[:], emuEdgeStride)
	return unsafe.Pointer(&edge[0]), emuEdgeStride, true
}

// mc8PutFast predicts a w x h 8-bit block (w a multiple of 8) with the
// dav1d-style put kernels. It returns false for shapes it does not cover, in
// which case the caller runs the general path.
func mc8PutFast(dst *frame.Plane, ref *frame.Plane, dstX, dstY, refX, refY, w, h, filterW, filterH, subX, subY int, filters InterpFilters, scratch *ConvolveScratch) bool {
	if subX|subY == 0 || (w&7 != 0 && w != 4) || w > maxBlockSize || h <= 0 || h > maxBlockSize || scratch == nil {
		return false
	}
	kx := mc8Kind(filters.X, filterW)
	ky := mc8Kind(filters.Y, filterH)
	if (subX != 0 && !mc8SixTap[kx]) || (subY != 0 && !mc8SixTap[ky]) {
		return false
	}
	if dstX < 0 || dstY < 0 || dstX+w > dst.Width || dstY+h > dst.Height || dst.Stride < dst.Width {
		return false
	}
	dOff := dstY*dst.Stride + dstX
	if dOff+(h-1)*dst.Stride+w > len(dst.Pix) {
		return false
	}
	src, stride, ok := mc8Source(ref, refX, refY, w, h, subX != 0, subY != 0, &scratch.edge)
	if !ok {
		return false
	}
	d := unsafe.Pointer(&dst.Pix[dOff])
	if w == 4 {
		switch {
		case subX != 0 && subY != 0:
			im := unsafe.Pointer(&scratch.imHBD[0])
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
		im := unsafe.Pointer(&scratch.imHBD[0])
		mc8H6ToI32(src, stride, w, h+5, &mc8Phases[kx][subX], im)
		mc8PutV6FromI32(im, w, h, &mc8Phases[ky][subY], d, dst.Stride)
	case subX != 0:
		mc8PutH6(src, stride, w, h, &mc8Phases[kx][subX], d, dst.Stride)
	default:
		mc8PutV6(src, stride, w, h, &mc8Phases[ky][subY], d, dst.Stride)
	}
	return true
}

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
	if dOff+(h-1)*dst.Stride+w > len(dst.Pix) || sOff+(h-1)*ref.Stride+w > len(ref.Pix) {
		return false
	}
	mc8Copy(unsafe.Pointer(&dst.Pix[dOff]), dst.Stride, unsafe.Pointer(&ref.Pix[sOff]), ref.Stride, w, h)
	return true
}

//go:nocheckptr
func mc8Copy(dst unsafe.Pointer, dstStride int, src unsafe.Pointer, srcStride int, w, h int) {
	switch {
	case w&15 == 0:
		for r := 0; r < h; r++ {
			s := unsafe.Add(src, r*srcStride)
			d := unsafe.Add(dst, r*dstStride)
			for c := 0; c < w; c += 16 {
				archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(s, c))).StoreArray((*[16]uint8)(unsafe.Add(d, c)))
			}
		}
	case w == 8:
		for r := 0; r < h; r++ {
			*(*uint64)(unsafe.Add(dst, r*dstStride)) = *(*uint64)(unsafe.Add(src, r*srcStride))
		}
	default:
		for r := 0; r < h; r++ {
			s := unsafe.Add(src, r*srcStride)
			d := unsafe.Add(dst, r*dstStride)
			for c := 0; c < w; c += 4 {
				*(*uint32)(unsafe.Add(d, c)) = *(*uint32)(unsafe.Add(s, c))
			}
		}
	}
}

// mc8PrepFast fills a w x h CONV_BUF (row stride w) with the dav1d-style prep
// kernels. It returns false for shapes it does not cover.
func mc8PrepFast(out []uint16, ref *frame.Plane, refX, refY, w, h, subX, subY int, filters InterpFilters, scratch *CompoundConvolveScratch) bool {
	if (w&7 != 0 && w != 4) || w > maxBlockSize || h <= 0 || h > maxBlockSize || scratch == nil || len(out) < w*h {
		return false
	}
	kx := mc8Kind(filters.X, w)
	ky := mc8Kind(filters.Y, h)
	if (subX != 0 && !mc8SixTap[kx]) || (subY != 0 && !mc8SixTap[ky]) {
		return false
	}
	src, stride, ok := mc8Source(ref, refX, refY, w, h, subX != 0, subY != 0, &scratch.edge)
	if !ok {
		return false
	}
	o := unsafe.Pointer(&out[0])
	if w == 4 {
		switch {
		case subX != 0 && subY != 0:
			im := unsafe.Pointer(&scratch.im[0])
			mc8H6ToI32W4(src, stride, h+5, &mc8Phases[kx][subX], im)
			mc8PrepV6FromI32W4(im, h, &mc8Phases[ky][subY], o)
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
		im := unsafe.Pointer(&scratch.im[0])
		mc8H6ToI32(src, stride, w, h+5, &mc8Phases[kx][subX], im)
		mc8PrepV6FromI32(im, w, h, &mc8Phases[ky][subY], o)
	case subX != 0:
		mc8PrepH6(src, stride, w, h, &mc8Phases[kx][subX], o)
	case subY != 0:
		mc8PrepV6(src, stride, w, h, &mc8Phases[ky][subY], o)
	default:
		mc8PrepCopy(src, stride, w, h, o)
	}
	return true
}

// mc8H6Sum is the six-tap halved horizontal sum of the eight output columns
// starting at p, plus rnd. lo/hi widen the 16 loaded bytes; the taps read
// lanes 0..12.
//
//go:nocheckptr
func mc8LoadWiden(p unsafe.Pointer) (archsimd.Uint8x16, archsimd.Uint8x16) {
	a := archsimd.LoadUint8x16Array((*[16]uint8)(p))
	return a.ExtendLo8ToUint16().ReshapeToUint8s(), a.HiToLo().ExtendLo8ToUint16().ReshapeToUint8s()
}

func mc8I16(x archsimd.Uint8x16) archsimd.Int16x8 {
	return x.ReshapeToUint16s().BitsToInt16()
}

// mc8H6ToI32 writes rows x w rounded horizontal sums ((s + 2) >> 2) as int32
// with row stride w.
//
//go:nocheckptr
func mc8H6ToI32(src unsafe.Pointer, srcStride, w, rows int, ph *mc8Phase, im unsafe.Pointer) {
	c0 := archsimd.LoadInt16x8Array(&ph.h[0])
	c1 := archsimd.LoadInt16x8Array(&ph.h[1])
	c2 := archsimd.LoadInt16x8Array(&ph.h[2])
	c3 := archsimd.LoadInt16x8Array(&ph.h[3])
	c4 := archsimd.LoadInt16x8Array(&ph.h[4])
	c5 := archsimd.LoadInt16x8Array(&ph.h[5])
	rnd := archsimd.BroadcastInt16x8(2)
	sh := archsimd.BroadcastInt16x8(-2)
	imStride := w * 4
	r := 0
	for ; r+2 <= rows; r += 2 {
		s := unsafe.Add(src, r*srcStride)
		o := unsafe.Add(im, r*imStride)
		for c := 0; c < w; c += 8 {
			lo, hi := mc8LoadWiden(unsafe.Add(s, c))
			lo1, hi1 := mc8LoadWiden(unsafe.Add(s, c+srcStride))
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
			sum = sum.Shift(sh)
			sum1 = sum1.Shift(sh)
			q := unsafe.Add(o, c*4)
			sum.ExtendLo4ToInt32().StoreArray((*[4]int32)(q))
			sum.HiToLo().ExtendLo4ToInt32().StoreArray((*[4]int32)(unsafe.Add(q, 16)))
			q = unsafe.Add(q, imStride)
			sum1.ExtendLo4ToInt32().StoreArray((*[4]int32)(q))
			sum1.HiToLo().ExtendLo4ToInt32().StoreArray((*[4]int32)(unsafe.Add(q, 16)))
		}
	}
	if r < rows {
		s := unsafe.Add(src, r*srcStride)
		o := unsafe.Add(im, r*imStride)
		for c := 0; c < w; c += 8 {
			lo, hi := mc8LoadWiden(unsafe.Add(s, c))
			sum := mc8I16(lo).MulAdd(c0, rnd)
			sum = mc8I16(hi.ConcatShiftBytesRight(lo, 2)).MulAdd(c1, sum)
			sum = mc8I16(hi.ConcatShiftBytesRight(lo, 4)).MulAdd(c2, sum)
			sum = mc8I16(hi.ConcatShiftBytesRight(lo, 6)).MulAdd(c3, sum)
			sum = mc8I16(hi.ConcatShiftBytesRight(lo, 8)).MulAdd(c4, sum)
			sum = mc8I16(hi.ConcatShiftBytesRight(lo, 10)).MulAdd(c5, sum)
			sum = sum.Shift(sh)
			q := unsafe.Add(o, c*4)
			sum.ExtendLo4ToInt32().StoreArray((*[4]int32)(q))
			sum.HiToLo().ExtendLo4ToInt32().StoreArray((*[4]int32)(unsafe.Add(q, 16)))
		}
	}
}

// mc8PutV6FromI32 is the put 2D vertical pass over the int32 intermediate.
//
//go:nocheckptr
func mc8PutV6FromI32(im unsafe.Pointer, w, h int, ph *mc8Phase, dst unsafe.Pointer, dstStride int) {
	c0 := archsimd.LoadInt32x4Array(&ph.v[0])
	c1 := archsimd.LoadInt32x4Array(&ph.v[1])
	c2 := archsimd.LoadInt32x4Array(&ph.v[2])
	c3 := archsimd.LoadInt32x4Array(&ph.v[3])
	c4 := archsimd.LoadInt32x4Array(&ph.v[4])
	c5 := archsimd.LoadInt32x4Array(&ph.v[5])
	rnd := archsimd.BroadcastInt32x4(1 << 15)
	rs := w * 4
	for c := 0; c < w; c += 8 {
		q := unsafe.Add(im, c*4)
		d := unsafe.Add(dst, c)
		x0, y0 := mc8L32(q), mc8L32(unsafe.Add(q, 16))
		q = unsafe.Add(q, rs)
		x1, y1 := mc8L32(q), mc8L32(unsafe.Add(q, 16))
		q = unsafe.Add(q, rs)
		x2, y2 := mc8L32(q), mc8L32(unsafe.Add(q, 16))
		q = unsafe.Add(q, rs)
		x3, y3 := mc8L32(q), mc8L32(unsafe.Add(q, 16))
		q = unsafe.Add(q, rs)
		x4, y4 := mc8L32(q), mc8L32(unsafe.Add(q, 16))
		q = unsafe.Add(q, rs)
		r := 0
		for ; r+2 <= h; r += 2 {
			x5, y5 := mc8L32(q), mc8L32(unsafe.Add(q, 16))
			q = unsafe.Add(q, rs)
			x6, y6 := mc8L32(q), mc8L32(unsafe.Add(q, 16))
			q = unsafe.Add(q, rs)
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
			*(*uint64)(d) = v0.SaturateToUint8().ReshapeToUint64s().GetElem(0)
			*(*uint64)(unsafe.Add(d, dstStride)) = v1.SaturateToUint8().ReshapeToUint64s().GetElem(0)
			d = unsafe.Add(d, 2*dstStride)
			x0, y0, x1, y1, x2, y2, x3, y3, x4, y4 = x2, y2, x3, y3, x4, y4, x5, y5, x6, y6
		}
		if r < h {
			x5, y5 := mc8L32(q), mc8L32(unsafe.Add(q, 16))
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
			v0 := a0.ToBits().ReshapeToUint16s().ConcatOdd(b0.ToBits().ReshapeToUint16s()).BitsToInt16()
			*(*uint64)(d) = v0.SaturateToUint8().ReshapeToUint64s().GetElem(0)
		}
	}
}

// mc8L32 loads four int32 lanes at p.
//
//go:nocheckptr
func mc8L32(p unsafe.Pointer) archsimd.Int32x4 {
	return archsimd.LoadInt32x4Array((*[4]int32)(p))
}

// mc8PrepV6FromI32 is the prep 2D vertical pass over the int32 intermediate.
//
//go:nocheckptr
func mc8PrepV6FromI32(im unsafe.Pointer, w, h int, ph *mc8Phase, out unsafe.Pointer) {
	c0 := archsimd.LoadInt32x4Array(&ph.p[0])
	c1 := archsimd.LoadInt32x4Array(&ph.p[1])
	c2 := archsimd.LoadInt32x4Array(&ph.p[2])
	c3 := archsimd.LoadInt32x4Array(&ph.p[3])
	c4 := archsimd.LoadInt32x4Array(&ph.p[4])
	c5 := archsimd.LoadInt32x4Array(&ph.p[5])
	rnd := archsimd.BroadcastInt32x4((64 + 6144<<7) << 9)
	rs := w * 4
	ro := w * 2
	for c := 0; c < w; c += 8 {
		q := unsafe.Add(im, c*4)
		o := unsafe.Add(out, c*2)
		x0, y0 := mc8L32(q), mc8L32(unsafe.Add(q, 16))
		q = unsafe.Add(q, rs)
		x1, y1 := mc8L32(q), mc8L32(unsafe.Add(q, 16))
		q = unsafe.Add(q, rs)
		x2, y2 := mc8L32(q), mc8L32(unsafe.Add(q, 16))
		q = unsafe.Add(q, rs)
		x3, y3 := mc8L32(q), mc8L32(unsafe.Add(q, 16))
		q = unsafe.Add(q, rs)
		x4, y4 := mc8L32(q), mc8L32(unsafe.Add(q, 16))
		q = unsafe.Add(q, rs)
		r := 0
		for ; r+2 <= h; r += 2 {
			x5, y5 := mc8L32(q), mc8L32(unsafe.Add(q, 16))
			q = unsafe.Add(q, rs)
			x6, y6 := mc8L32(q), mc8L32(unsafe.Add(q, 16))
			q = unsafe.Add(q, rs)
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
			a0.ToBits().ReshapeToUint16s().ConcatOdd(b0.ToBits().ReshapeToUint16s()).StoreArray((*[8]uint16)(o))
			a1.ToBits().ReshapeToUint16s().ConcatOdd(b1.ToBits().ReshapeToUint16s()).StoreArray((*[8]uint16)(unsafe.Add(o, ro)))
			o = unsafe.Add(o, 2*ro)
			x0, y0, x1, y1, x2, y2, x3, y3, x4, y4 = x2, y2, x3, y3, x4, y4, x5, y5, x6, y6
		}
		if r < h {
			x5, y5 := mc8L32(q), mc8L32(unsafe.Add(q, 16))
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
			a0.ToBits().ReshapeToUint16s().ConcatOdd(b0.ToBits().ReshapeToUint16s()).StoreArray((*[8]uint16)(o))
		}
	}
}

// mc8PutH6 is the put horizontal-only pass: clip((s + 34) >> 6).
//
//go:nocheckptr
func mc8PutH6(src unsafe.Pointer, srcStride, w, h int, ph *mc8Phase, dst unsafe.Pointer, dstStride int) {
	c0 := archsimd.LoadInt16x8Array(&ph.h[0])
	c1 := archsimd.LoadInt16x8Array(&ph.h[1])
	c2 := archsimd.LoadInt16x8Array(&ph.h[2])
	c3 := archsimd.LoadInt16x8Array(&ph.h[3])
	c4 := archsimd.LoadInt16x8Array(&ph.h[4])
	c5 := archsimd.LoadInt16x8Array(&ph.h[5])
	rnd := archsimd.BroadcastInt16x8(34)
	sh := archsimd.BroadcastInt16x8(-6)
	r := 0
	for ; r+2 <= h; r += 2 {
		s := unsafe.Add(src, r*srcStride)
		d := unsafe.Add(dst, r*dstStride)
		for c := 0; c < w; c += 8 {
			lo, hi := mc8LoadWiden(unsafe.Add(s, c))
			lo1, hi1 := mc8LoadWiden(unsafe.Add(s, c+srcStride))
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
			*(*uint64)(unsafe.Add(d, c)) = sum.Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0)
			*(*uint64)(unsafe.Add(d, c+dstStride)) = sum1.Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0)
		}
	}
	if r < h {
		s := unsafe.Add(src, r*srcStride)
		d := unsafe.Add(dst, r*dstStride)
		for c := 0; c < w; c += 8 {
			lo, hi := mc8LoadWiden(unsafe.Add(s, c))
			sum := mc8I16(lo).MulAdd(c0, rnd)
			sum = mc8I16(hi.ConcatShiftBytesRight(lo, 2)).MulAdd(c1, sum)
			sum = mc8I16(hi.ConcatShiftBytesRight(lo, 4)).MulAdd(c2, sum)
			sum = mc8I16(hi.ConcatShiftBytesRight(lo, 6)).MulAdd(c3, sum)
			sum = mc8I16(hi.ConcatShiftBytesRight(lo, 8)).MulAdd(c4, sum)
			sum = mc8I16(hi.ConcatShiftBytesRight(lo, 10)).MulAdd(c5, sum)
			*(*uint64)(unsafe.Add(d, c)) = sum.Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0)
		}
	}
}

// mc8PrepH6 is the prep horizontal-only pass: ((s + 2) >> 2) + 6144.
//
//go:nocheckptr
func mc8PrepH6(src unsafe.Pointer, srcStride, w, h int, ph *mc8Phase, out unsafe.Pointer) {
	c0 := archsimd.LoadInt16x8Array(&ph.h[0])
	c1 := archsimd.LoadInt16x8Array(&ph.h[1])
	c2 := archsimd.LoadInt16x8Array(&ph.h[2])
	c3 := archsimd.LoadInt16x8Array(&ph.h[3])
	c4 := archsimd.LoadInt16x8Array(&ph.h[4])
	c5 := archsimd.LoadInt16x8Array(&ph.h[5])
	rnd := archsimd.BroadcastInt16x8(2 + 6144<<2)
	sh := archsimd.BroadcastInt16x8(-2)
	ro := w * 2
	r := 0
	for ; r+2 <= h; r += 2 {
		s := unsafe.Add(src, r*srcStride)
		o := unsafe.Add(out, r*ro)
		for c := 0; c < w; c += 8 {
			lo, hi := mc8LoadWiden(unsafe.Add(s, c))
			lo1, hi1 := mc8LoadWiden(unsafe.Add(s, c+srcStride))
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
			sum.ToBits().Shift(sh).StoreArray((*[8]uint16)(unsafe.Add(o, c*2)))
			sum1.ToBits().Shift(sh).StoreArray((*[8]uint16)(unsafe.Add(o, c*2+ro)))
		}
	}
	if r < h {
		s := unsafe.Add(src, r*srcStride)
		o := unsafe.Add(out, r*ro)
		for c := 0; c < w; c += 8 {
			lo, hi := mc8LoadWiden(unsafe.Add(s, c))
			sum := mc8I16(lo).MulAdd(c0, rnd)
			sum = mc8I16(hi.ConcatShiftBytesRight(lo, 2)).MulAdd(c1, sum)
			sum = mc8I16(hi.ConcatShiftBytesRight(lo, 4)).MulAdd(c2, sum)
			sum = mc8I16(hi.ConcatShiftBytesRight(lo, 6)).MulAdd(c3, sum)
			sum = mc8I16(hi.ConcatShiftBytesRight(lo, 8)).MulAdd(c4, sum)
			sum = mc8I16(hi.ConcatShiftBytesRight(lo, 10)).MulAdd(c5, sum)
			sum.ToBits().Shift(sh).StoreArray((*[8]uint16)(unsafe.Add(o, c*2)))
		}
	}
}

// mc8PutV6 is the put vertical-only pass: clip((s + 32) >> 6). Each eight
// column strip keeps its source rows resident, loads two new rows per two
// output rows and runs the two rows' tap chains interleaved.
//
//go:nocheckptr
func mc8PutV6(src unsafe.Pointer, srcStride, w, h int, ph *mc8Phase, dst unsafe.Pointer, dstStride int) {
	c0 := archsimd.LoadInt16x8Array(&ph.h[0])
	c1 := archsimd.LoadInt16x8Array(&ph.h[1])
	c2 := archsimd.LoadInt16x8Array(&ph.h[2])
	c3 := archsimd.LoadInt16x8Array(&ph.h[3])
	c4 := archsimd.LoadInt16x8Array(&ph.h[4])
	c5 := archsimd.LoadInt16x8Array(&ph.h[5])
	rnd := archsimd.BroadcastInt16x8(32)
	sh := archsimd.BroadcastInt16x8(-6)
	for c := 0; c < w; c += 8 {
		s := unsafe.Add(src, c)
		d := unsafe.Add(dst, c)
		r0 := mc8LoadRow(s)
		r1 := mc8LoadRow(unsafe.Add(s, srcStride))
		r2 := mc8LoadRow(unsafe.Add(s, 2*srcStride))
		r3 := mc8LoadRow(unsafe.Add(s, 3*srcStride))
		r4 := mc8LoadRow(unsafe.Add(s, 4*srcStride))
		s = unsafe.Add(s, 5*srcStride)
		r := 0
		for ; r+2 <= h; r += 2 {
			r5 := mc8LoadRow(s)
			r6 := mc8LoadRow(unsafe.Add(s, srcStride))
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
			*(*uint64)(d) = a.Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0)
			*(*uint64)(unsafe.Add(d, dstStride)) = b.Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0)
			r0, r1, r2, r3, r4 = r2, r3, r4, r5, r6
			s = unsafe.Add(s, 2*srcStride)
			d = unsafe.Add(d, 2*dstStride)
		}
		if r < h {
			r5 := mc8LoadRow(s)
			a := r0.MulAdd(c0, rnd)
			a = r1.MulAdd(c1, a)
			a = r2.MulAdd(c2, a)
			a = r3.MulAdd(c3, a)
			a = r4.MulAdd(c4, a)
			a = r5.MulAdd(c5, a)
			*(*uint64)(d) = a.Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0)
		}
	}
}

// mc8PrepV6 is the prep vertical-only pass: ((s + 2) >> 2) + 6144.
//
//go:nocheckptr
func mc8PrepV6(src unsafe.Pointer, srcStride, w, h int, ph *mc8Phase, out unsafe.Pointer) {
	c0 := archsimd.LoadInt16x8Array(&ph.h[0])
	c1 := archsimd.LoadInt16x8Array(&ph.h[1])
	c2 := archsimd.LoadInt16x8Array(&ph.h[2])
	c3 := archsimd.LoadInt16x8Array(&ph.h[3])
	c4 := archsimd.LoadInt16x8Array(&ph.h[4])
	c5 := archsimd.LoadInt16x8Array(&ph.h[5])
	rnd := archsimd.BroadcastInt16x8(2 + 6144<<2)
	sh := archsimd.BroadcastInt16x8(-2)
	ro := w * 2
	for c := 0; c < w; c += 8 {
		s := unsafe.Add(src, c)
		o := unsafe.Add(out, c*2)
		r0 := mc8LoadRow(s)
		r1 := mc8LoadRow(unsafe.Add(s, srcStride))
		r2 := mc8LoadRow(unsafe.Add(s, 2*srcStride))
		r3 := mc8LoadRow(unsafe.Add(s, 3*srcStride))
		r4 := mc8LoadRow(unsafe.Add(s, 4*srcStride))
		s = unsafe.Add(s, 5*srcStride)
		r := 0
		for ; r+2 <= h; r += 2 {
			r5 := mc8LoadRow(s)
			r6 := mc8LoadRow(unsafe.Add(s, srcStride))
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
			a.ToBits().Shift(sh).StoreArray((*[8]uint16)(o))
			b.ToBits().Shift(sh).StoreArray((*[8]uint16)(unsafe.Add(o, ro)))
			r0, r1, r2, r3, r4 = r2, r3, r4, r5, r6
			s = unsafe.Add(s, 2*srcStride)
			o = unsafe.Add(o, 2*ro)
		}
		if r < h {
			r5 := mc8LoadRow(s)
			a := r0.MulAdd(c0, rnd)
			a = r1.MulAdd(c1, a)
			a = r2.MulAdd(c2, a)
			a = r3.MulAdd(c3, a)
			a = r4.MulAdd(c4, a)
			a = r5.MulAdd(c5, a)
			a.ToBits().Shift(sh).StoreArray((*[8]uint16)(o))
		}
	}
}

// mc8PrepCopy is the prep copy: 16 * px + 6144.
//
//go:nocheckptr
func mc8PrepCopy(src unsafe.Pointer, srcStride, w, h int, out unsafe.Pointer) {
	mul := archsimd.BroadcastUint16x8(16)
	off := archsimd.BroadcastUint16x8(6144)
	ro := w * 2
	for r := 0; r < h; r++ {
		s := unsafe.Add(src, r*srcStride)
		o := unsafe.Add(out, r*ro)
		c := 0
		for ; c+16 <= w; c += 16 {
			a := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(s, c)))
			a.ExtendLo8ToUint16().MulAdd(mul, off).StoreArray((*[8]uint16)(unsafe.Add(o, c*2)))
			a.HiToLo().ExtendLo8ToUint16().MulAdd(mul, off).StoreArray((*[8]uint16)(unsafe.Add(o, c*2+16)))
		}
		if c < w {
			a := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(s, c)))
			a.ExtendLo8ToUint16().MulAdd(mul, off).StoreArray((*[8]uint16)(unsafe.Add(o, c*2)))
		}
	}
}

// mc8LoadRow widens the eight bytes at p (16 bytes are read).
//
//go:nocheckptr
func mc8LoadRow(p unsafe.Pointer) archsimd.Int16x8 {
	return archsimd.LoadUint8x16Array((*[16]uint8)(p)).ExtendLo8ToUint16().BitsToInt16()
}

// mc8PackU8 saturates two int16x8 vectors to one uint8x16 (SQXTUN +
// SQXTUN2).
func mc8PackU8(v0, v1 archsimd.Int16x8) archsimd.Uint8x16 {
	lo := v0.SaturateToUint8().ReshapeToUint64s().BitsToFloat64()
	return lo.SetElem(1, v1.SaturateToUint8().ReshapeToUint64s().BitsToFloat64().GetElem(0)).ToBits().ReshapeToUint8s()
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
	if dstX < 0 || dstY < 0 || dstX+w > dst.Width || off+(h-1)*dst.Stride+w > len(dst.Pix) {
		return false
	}
	mc8BlendAvg(unsafe.Pointer(&dst.Pix[off]), dst.Stride, unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), w, h)
	return true
}

//go:nocheckptr
func mc8BlendAvg(dst unsafe.Pointer, dstStride int, a, b unsafe.Pointer, w, h int) {
	bias := archsimd.BroadcastInt16x8(-12272)
	sh := archsimd.BroadcastInt16x8(-5)
	if w == 4 {
		for r := 0; r < h; r += 2 {
			o := r * 8
			v := mc8L16(unsafe.Add(a, o)).Add(mc8L16(unsafe.Add(b, o))).Add(bias).Shift(sh)
			x := v.SaturateToUint8().ReshapeToUint64s().GetElem(0)
			d := unsafe.Add(dst, r*dstStride)
			*(*uint32)(d) = uint32(x)
			*(*uint32)(unsafe.Add(d, dstStride)) = uint32(x >> 32)
		}
		return
	}
	ro := w * 2
	for r := 0; r < h; r++ {
		pa := unsafe.Add(a, r*ro)
		pb := unsafe.Add(b, r*ro)
		d := unsafe.Add(dst, r*dstStride)
		c := 0
		for ; c+16 <= w; c += 16 {
			v0 := mc8L16(unsafe.Add(pa, 2*c)).Add(mc8L16(unsafe.Add(pb, 2*c))).Add(bias).Shift(sh)
			v1 := mc8L16(unsafe.Add(pa, 2*c+16)).Add(mc8L16(unsafe.Add(pb, 2*c+16))).Add(bias).Shift(sh)
			mc8PackU8(v0, v1).StoreArray((*[16]uint8)(unsafe.Add(d, c)))
		}
		if c < w {
			v := mc8L16(unsafe.Add(pa, 2*c)).Add(mc8L16(unsafe.Add(pb, 2*c))).Add(bias).Shift(sh)
			*(*uint64)(unsafe.Add(d, c)) = v.SaturateToUint8().ReshapeToUint64s().GetElem(0)
		}
	}
}

// mc8L16 loads eight int16 lanes at p.
//
//go:nocheckptr
func mc8L16(p unsafe.Pointer) archsimd.Int16x8 {
	return archsimd.LoadInt16x8Array((*[8]int16)(p))
}
