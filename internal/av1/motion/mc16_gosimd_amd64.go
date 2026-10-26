// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && amd64 && !purego

package motion

import (
	"simd/archsimd"
	"unsafe"

	"github.com/thesyncim/goav1/internal/av1/frame"
)

// AVX2 16bpc (10- and 12-bit) translational motion compensation, ported from
// dav1d's src/x86/mc16_avx2.asm put_8tap_16bpc / prep_8tap_16bpc: every
// filter runs as eight taps whose pairs feed vpmaddwd against the
// interleaved (s[i], s[i+1]) sample pairs, exact in int32 (samples are at
// most 4095, taps fit int16). Sixteen outputs per ymm pair; the in-lane
// unpack order is undone by the in-lane pack. The arithmetic is the
// libaom-exact one of the scalar paths on the full-tap sum S (round0/round1
// from highBDRoundBits, rc = compoundRound0):
//
//	put H:   clip((S + 2^(round0-1) + 64) >> 7)       (nested round0, 7-round0)
//	put V:   clip((S + 64) >> 7)
//	put 2D:  m = (S + 2^(round0-1)) >> round0
//	         clip((sum(f * m) + 2^(round1-1)) >> round1)  (libaom's offsets cancel)
//	prep H, V: ((S + 2^(rc-1)) >> rc) + 2^14 + 2^13, as uint16
//	prep 2D: m = (S + 2^(rc-1)) >> rc
//	         (sum(f * m) + 2^21 + 2^20 + 64) >> 7, as uint16
//
// Every shift count is an immediate or a hoisted vpsravd vector, and every
// kernel returns through vzeroupper: a legacy-SSE instruction with dirty
// upper ymm state stalls on Zen 3, and the Go compiler emits neither.

// mc16Pairs[kind][phase][i] is the vpmaddwd tap pair (f[2i], f[2i+1]) of the
// full eight-tap filter.
var mc16Pairs [mc8KindCount][16][4]int32

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
			for i := range 4 {
				mc16Pairs[kind][s][i] = int32(uint32(uint16(tab[s][2*i])) | uint32(uint16(tab[s][2*i+1]))<<16)
			}
		}
	}
}

const mc16PrepOffset = 1<<14 + 1<<13

// mc16SourceAVX2 resolves the eight-tap source window of a w x h block: an
// axis with a filter reads from three samples before the block. The kernels
// load sixteen (or eight, for the tails and w == 4) samples per tap, so the
// window must stay readable for max(w, 8) + 7 samples per row when filtered
// horizontally (max(w, 8) otherwise). Overhanging blocks materialize the
// window with emuEdge16.
func mc16SourceAVX2(ref *frame.Plane, refX, refY, w, h int, hx, hy bool, edge *emuEdge16Buf) (src unsafe.Pointer, stride int, ok bool) {
	x0, y0, bw, bh := refX, refY, w, h
	read := max(w, 8)
	if hx {
		x0 -= 3
		bw += 7
		read += 7
	}
	if hy {
		y0 -= 3
		bh += 7
	}
	read *= 2
	if ref.Width <= 0 || ref.Height <= 0 || ref.Stride < 2*ref.Width {
		return nil, 0, false
	}
	if x0 >= 0 && y0 >= 0 && x0+bw <= ref.Width && y0+bh <= ref.Height {
		start := y0*ref.Stride + 2*x0
		if start+(bh-1)*ref.Stride+read <= len(ref.Pix) {
			return unsafe.Pointer(unsafe.SliceData(ref.Pix[start:])), ref.Stride, true
		}
	}
	if len(ref.Pix) < (ref.Height-1)*ref.Stride+2*ref.Width {
		return nil, 0, false
	}
	// emuEdge16Rows (136) rows of emuEdge16Stride (144) samples hold the
	// window (bw <= 135, bh <= 135) and the kernels' over-read.
	emuEdge16(bw, bh, *ref, x0, y0, edge[:], 2*emuEdge16Stride)
	return unsafe.Pointer(&edge[0]), 2 * emuEdge16Stride, true
}

// mc16Shape reports whether the fast kernels cover a w x h block.
func mc16Shape(bitDepth, w, h int) bool {
	return mc8AVX2 && (bitDepth == 10 || bitDepth == 12) && (w&7 == 0 || w == 4) && w <= maxBlockSize && h > 0 && h <= maxBlockSize
}

// mc16PutFast predicts a w x h block of a 10- or 12-bit plane. It returns
// false for shapes it does not cover (the caller runs the general path).
func mc16PutFast(dst *frame.Plane, ref *frame.Plane, bitDepth, dstX, dstY, refX, refY, w, h, filterW, filterH, subX, subY int, filters InterpFilters, scratch *ConvolveScratch) bool {
	if subX|subY == 0 || !mc16Shape(bitDepth, w, h) || scratch == nil {
		return false
	}
	if dstX < 0 || dstY < 0 || dstX+w > dst.Width || dstY+h > dst.Height || dst.Stride < 2*dst.Width {
		return false
	}
	dOff := dstY*dst.Stride + 2*dstX
	if dOff+(h-1)*dst.Stride+2*w > len(dst.Pix) {
		return false
	}
	src, stride, ok := mc16SourceAVX2(ref, refX, refY, w, h, subX != 0, subY != 0, &scratch.edge16)
	if !ok {
		return false
	}
	d := unsafe.Pointer(&dst.Pix[dOff])
	maxV := int16(1)<<bitDepth - 1
	round0, round1 := highBDRoundBits(uint8(bitDepth))
	px := &mc16Pairs[mc8Kind(filters.X, filterW)][subX]
	py := &mc16Pairs[mc8Kind(filters.Y, filterH)][subY]
	switch {
	case subX != 0 && subY != 0:
		im := unsafe.Pointer(&scratch.imHBD[0])
		mc16HToI32(src, stride, w, h+7, px, round0, im)
		mc16VFromI32(im, w, h, py, true, round1, maxV, d, dst.Stride)
	case subX != 0:
		mc16H(src, stride, w, h, px, true, round0, maxV, d, dst.Stride)
	default:
		mc16V(src, stride, w, h, py, true, 0, maxV, d, dst.Stride)
	}
	return true
}

// mc16PrepFast fills a w x h CONV_BUF (row stride w) from a 10- or 12-bit
// reference. It returns false for shapes it does not cover.
func mc16PrepFast(out []uint16, ref *frame.Plane, bitDepth, refX, refY, w, h, filterW, filterH, subX, subY int, filters InterpFilters, scratch *CompoundConvolveScratch) bool {
	if !mc16Shape(bitDepth, w, h) || scratch == nil || len(out) < w*h {
		return false
	}
	src, stride, ok := mc16SourceAVX2(ref, refX, refY, w, h, subX != 0, subY != 0, &scratch.edge16)
	if !ok {
		return false
	}
	rc := compoundRound0(uint8(bitDepth))
	o := unsafe.Pointer(&out[0])
	px := &mc16Pairs[mc8Kind(filters.X, filterW)][subX]
	py := &mc16Pairs[mc8Kind(filters.Y, filterH)][subY]
	switch {
	case subX != 0 && subY != 0:
		im := unsafe.Pointer(&scratch.im[0])
		mc16HToI32(src, stride, w, h+7, px, rc, im)
		mc16VFromI32(im, w, h, py, false, 0, 0, o, 2*w)
	case subX != 0:
		mc16H(src, stride, w, h, px, false, rc, 0, o, 2*w)
	case subY != 0:
		mc16V(src, stride, w, h, py, false, rc, 0, o, 2*w)
	default:
		mc16PrepCopy(src, stride, w, h, 7-rc, o)
	}
	return true
}

// mc16TapPairs16 broadcasts the four tap pairs.
func mc16TapPairs16(p *[4]int32) (k0, k1, k2, k3 archsimd.Int16x16) {
	return archsimd.BroadcastInt32x8(p[0]).AsInt16x16(), archsimd.BroadcastInt32x8(p[1]).AsInt16x16(),
		archsimd.BroadcastInt32x8(p[2]).AsInt16x16(), archsimd.BroadcastInt32x8(p[3]).AsInt16x16()
}

// mc16Ld16 loads sixteen samples at p.
//
//go:nocheckptr
func mc16Ld16(p unsafe.Pointer) archsimd.Int16x16 {
	return archsimd.LoadInt16x16Array((*[16]int16)(p))
}

// mc16Ld8 loads eight samples at p into the low lane.
//
//go:nocheckptr
func mc16Ld8(p unsafe.Pointer) archsimd.Int16x16 {
	return archsimd.Int16x16{}.SetLo(archsimd.LoadInt16x8Array((*[8]int16)(p)))
}

// mc16Dot8 is the eight-tap sum of the sample rows (or shifted sample
// windows) a0..a7: lo holds outputs 0-3 and 8-11, hi 4-7 and 12-15 (the
// in-lane unpack order).
func mc16Dot8(a0, a1, a2, a3, a4, a5, a6, a7, k0, k1, k2, k3 archsimd.Int16x16) (lo, hi archsimd.Int32x8) {
	lo = a0.InterleaveLoGrouped(a1).DotProductPairs(k0).Add(a2.InterleaveLoGrouped(a3).DotProductPairs(k1)).
		Add(a4.InterleaveLoGrouped(a5).DotProductPairs(k2)).Add(a6.InterleaveLoGrouped(a7).DotProductPairs(k3))
	hi = a0.InterleaveHiGrouped(a1).DotProductPairs(k0).Add(a2.InterleaveHiGrouped(a3).DotProductPairs(k1)).
		Add(a4.InterleaveHiGrouped(a5).DotProductPairs(k2)).Add(a6.InterleaveHiGrouped(a7).DotProductPairs(k3))
	return lo, hi
}

// mc16HDot loads the eight shifted windows of the sixteen (or, with n8, the
// eight) outputs at p and returns their tap sums. The loads are spelled out
// (not through mc16Ld16) so none is left as an outlined call.
//
//go:nocheckptr
func mc16HDot(p unsafe.Pointer, n8 bool, k0, k1, k2, k3 archsimd.Int16x16) (lo, hi archsimd.Int32x8) {
	if n8 {
		var z archsimd.Int16x16
		l := func(o uintptr) archsimd.Int16x16 {
			return z.SetLo(archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(p, o))))
		}
		return mc16Dot8(l(0), l(2), l(4), l(6), l(8), l(10), l(12), l(14), k0, k1, k2, k3)
	}
	return mc16Dot8(archsimd.LoadInt16x16Array((*[16]int16)(p)), archsimd.LoadInt16x16Array((*[16]int16)(unsafe.Add(p, 2))),
		archsimd.LoadInt16x16Array((*[16]int16)(unsafe.Add(p, 4))), archsimd.LoadInt16x16Array((*[16]int16)(unsafe.Add(p, 6))),
		archsimd.LoadInt16x16Array((*[16]int16)(unsafe.Add(p, 8))), archsimd.LoadInt16x16Array((*[16]int16)(unsafe.Add(p, 10))),
		archsimd.LoadInt16x16Array((*[16]int16)(unsafe.Add(p, 12))), archsimd.LoadInt16x16Array((*[16]int16)(unsafe.Add(p, 14))), k0, k1, k2, k3)
}

// mc16Out finishes sixteen (n = 16), eight or four sums in the in-lane order
// and stores them at d: put clips (lo, hi) >> sh to [0, maxV]; prep adds the
// CONV_BUF offset and keeps the low 16 bits (libaom's uint16 cast).
//
//go:nocheckptr
func mc16Out(lo, hi archsimd.Int32x8, n int, put bool, shV archsimd.Uint32x8, maxV archsimd.Int16x16, d unsafe.Pointer) {
	lo, hi = lo.ShiftRight(shV), hi.ShiftRight(shV)
	var v archsimd.Int16x16
	if put {
		v = lo.SaturateToInt16ConcatGrouped(hi).Max(archsimd.Int16x16{}).Min(maxV)
	} else {
		off := archsimd.BroadcastInt32x8(mc16PrepOffset)
		mask := archsimd.BroadcastInt32x8(0xffff)
		v = lo.Add(off).And(mask).SaturateToUint16ConcatGrouped(hi.Add(off).And(mask)).AsInt16x16()
	}
	switch n {
	case 16:
		v.StoreArray((*[16]int16)(d))
	case 8:
		v.GetLo().StoreArray((*[8]int16)(d))
	default:
		*(*uint64)(d) = v.GetLo().AsUint64x2().GetElem(0)
	}
}

// mc16H is the horizontal-only pass: put (clip((S + rnd) >> 7)) or prep
// (((S + rnd) >> rc) + offset). r is round0 (put) or rc (prep).
//
//go:nocheckptr
func mc16H(src unsafe.Pointer, ss, w, h int, pairs *[4]int32, put bool, r int, maxPix int16, dst unsafe.Pointer, ds int) {
	k0, k1, k2, k3 := mc16TapPairs16(pairs)
	var rnd int32
	var shV archsimd.Uint32x8
	if put {
		rnd = int32(1)<<(r-1) + 64
		shV = archsimd.BroadcastUint32x8(7)
	} else {
		rnd = int32(1) << (r - 1)
		shV = archsimd.BroadcastUint32x8(uint32(r))
	}
	rndV := archsimd.BroadcastInt32x8(rnd)
	maxV := archsimd.BroadcastInt16x16(maxPix)
	for y := 0; y < h; y++ {
		s := unsafe.Add(src, y*ss)
		d := unsafe.Add(dst, y*ds)
		c := 0
		for ; c+16 <= w; c += 16 {
			lo, hi := mc16HDot(unsafe.Add(s, 2*c), false, k0, k1, k2, k3)
			mc16Out(lo.Add(rndV), hi.Add(rndV), 16, put, shV, maxV, unsafe.Add(d, 2*c))
		}
		if c < w {
			lo, hi := mc16HDot(unsafe.Add(s, 2*c), true, k0, k1, k2, k3)
			mc16Out(lo.Add(rndV), hi.Add(rndV), w-c, put, shV, maxV, unsafe.Add(d, 2*c))
		}
	}
	archsimd.ClearAVXUpperBits()
}

// mc16V is the vertical-only pass (put: clip((S + 64) >> 7); prep: as
// mc16H), sliding an eight-row window down each column strip.
//
//go:nocheckptr
func mc16V(src unsafe.Pointer, ss, w, h int, pairs *[4]int32, put bool, rc int, maxPix int16, dst unsafe.Pointer, ds int) {
	k0, k1, k2, k3 := mc16TapPairs16(pairs)
	var rnd int32
	var shV archsimd.Uint32x8
	if put {
		rnd = 64
		shV = archsimd.BroadcastUint32x8(7)
	} else {
		rnd = int32(1) << (rc - 1)
		shV = archsimd.BroadcastUint32x8(uint32(rc))
	}
	rndV := archsimd.BroadcastInt32x8(rnd)
	maxV := archsimd.BroadcastInt16x16(maxPix)
	for c := 0; c < w; c += 16 {
		n := min(16, w-c)
		ld := func(p unsafe.Pointer) archsimd.Int16x16 {
			if n == 16 {
				return archsimd.LoadInt16x16Array((*[16]int16)(p))
			}
			return archsimd.Int16x16{}.SetLo(archsimd.LoadInt16x8Array((*[8]int16)(p)))
		}
		s := unsafe.Add(src, 2*c)
		d := unsafe.Add(dst, 2*c)
		r0, r1, r2, r3 := ld(s), ld(unsafe.Add(s, ss)), ld(unsafe.Add(s, 2*ss)), ld(unsafe.Add(s, 3*ss))
		r4, r5, r6 := ld(unsafe.Add(s, 4*ss)), ld(unsafe.Add(s, 5*ss)), ld(unsafe.Add(s, 6*ss))
		s = unsafe.Add(s, 7*ss)
		for y := 0; y < h; y++ {
			r7 := ld(s)
			lo, hi := mc16Dot8(r0, r1, r2, r3, r4, r5, r6, r7, k0, k1, k2, k3)
			mc16Out(lo.Add(rndV), hi.Add(rndV), n, put, shV, maxV, d)
			r0, r1, r2, r3, r4, r5, r6 = r1, r2, r3, r4, r5, r6, r7
			s = unsafe.Add(s, ss)
			d = unsafe.Add(d, ds)
		}
	}
	archsimd.ClearAVXUpperBits()
}

// mc16HToI32 writes rows x w rounded horizontal sums m = (S + 2^(r-1)) >> r
// as int32 with row stride w (r is round0 for put, rc for prep).
//
//go:nocheckptr
func mc16HToI32(src unsafe.Pointer, ss, w, rows int, pairs *[4]int32, r int, im unsafe.Pointer) {
	k0, k1, k2, k3 := mc16TapPairs16(pairs)
	rndV := archsimd.BroadcastInt32x8(int32(1) << (r - 1))
	shV := archsimd.BroadcastUint32x8(uint32(r))
	for y := 0; y < rows; y++ {
		s := unsafe.Add(src, y*ss)
		o := unsafe.Add(im, 4*y*w)
		c := 0
		for ; c+16 <= w; c += 16 {
			lo, hi := mc16HDot(unsafe.Add(s, 2*c), false, k0, k1, k2, k3)
			lo, hi = lo.Add(rndV).ShiftRight(shV), hi.Add(rndV).ShiftRight(shV)
			// Undo the in-lane order: outputs 0-7, then 8-15.
			lo.ConcatPermute128Scalars(0, 2, hi).StoreArray((*[8]int32)(unsafe.Add(o, 4*c)))
			lo.ConcatPermute128Scalars(1, 3, hi).StoreArray((*[8]int32)(unsafe.Add(o, 4*c+32)))
		}
		if c < w {
			lo, hi := mc16HDot(unsafe.Add(s, 2*c), true, k0, k1, k2, k3)
			lo, hi = lo.Add(rndV).ShiftRight(shV), hi.Add(rndV).ShiftRight(shV)
			v := lo.ConcatPermute128Scalars(0, 2, hi)
			if w-c == 8 {
				v.StoreArray((*[8]int32)(unsafe.Add(o, 4*c)))
			} else {
				v.GetLo().StoreArray((*[4]int32)(unsafe.Add(o, 4*c)))
			}
		}
	}
	archsimd.ClearAVXUpperBits()
}

// mc16PairOrder is the vpermd order that restores sample order after a
// vpackssdw/vpackusdw of two eight-column int32 vectors.
var mc16PairOrder = [8]uint32{0, 1, 4, 5, 2, 3, 6, 7}

// mc16VFromI32 is the 2D vertical pass over the int32 intermediate (row
// stride w): sum(f * m) in int32 (vpmulld), then put clips
// (sum + 2^(round1-1)) >> round1 and prep stores (sum + 2^21 + 2^20 + 64) >> 7
// as uint16.
//
//go:nocheckptr
func mc16VFromI32(im unsafe.Pointer, w, h int, pairs *[4]int32, put bool, round1 int, maxPix int16, dst unsafe.Pointer, ds int) {
	var f [8]archsimd.Int32x8
	for i := range 4 {
		f[2*i] = archsimd.BroadcastInt32x8(int32(int16(pairs[i])))
		f[2*i+1] = archsimd.BroadcastInt32x8(pairs[i] >> 16)
	}
	var rnd int32
	var shV archsimd.Uint32x8
	if put {
		rnd = int32(1) << (round1 - 1)
		shV = archsimd.BroadcastUint32x8(uint32(round1))
	} else {
		rnd = 1<<21 + 1<<20 + 64
		shV = archsimd.BroadcastUint32x8(7)
	}
	rndV := archsimd.BroadcastInt32x8(rnd)
	maxV := archsimd.BroadcastInt16x16(maxPix)
	mask := archsimd.BroadcastInt32x8(0xffff)
	perm := archsimd.LoadUint32x8Array(&mc16PairOrder)
	rs := 4 * w
	ld := func(p unsafe.Pointer, n int) archsimd.Int32x8 {
		if n == 4 {
			return archsimd.Int32x8{}.SetLo(archsimd.LoadInt32x4Array((*[4]int32)(p)))
		}
		return archsimd.LoadInt32x8Array((*[8]int32)(p))
	}
	for c := 0; c < w; c += 8 {
		n := min(8, w-c)
		q := unsafe.Add(im, 4*c)
		d := unsafe.Add(dst, 2*c)
		for y := 0; y < h; y++ {
			p := unsafe.Add(q, y*rs)
			acc := rndV.Add(ld(p, n).Mul(f[0]))
			acc = acc.Add(ld(unsafe.Add(p, rs), n).Mul(f[1]))
			acc = acc.Add(ld(unsafe.Add(p, 2*rs), n).Mul(f[2]))
			acc = acc.Add(ld(unsafe.Add(p, 3*rs), n).Mul(f[3]))
			acc = acc.Add(ld(unsafe.Add(p, 4*rs), n).Mul(f[4]))
			acc = acc.Add(ld(unsafe.Add(p, 5*rs), n).Mul(f[5]))
			acc = acc.Add(ld(unsafe.Add(p, 6*rs), n).Mul(f[6]))
			acc = acc.Add(ld(unsafe.Add(p, 7*rs), n).Mul(f[7]))
			acc = acc.ShiftRight(shV)
			var v archsimd.Int16x16
			if put {
				v = acc.SaturateToInt16ConcatGrouped(acc).AsInt32x8().Permute(perm).AsInt16x16().Max(archsimd.Int16x16{}).Min(maxV)
			} else {
				a := acc.And(mask)
				v = a.SaturateToUint16ConcatGrouped(a).AsInt32x8().Permute(perm).AsInt16x16()
			}
			o := unsafe.Add(d, y*ds)
			if n == 8 {
				v.GetLo().StoreArray((*[8]int16)(o))
			} else {
				*(*uint64)(o) = v.GetLo().AsUint64x2().GetElem(0)
			}
		}
	}
	archsimd.ClearAVXUpperBits()
}

// mc16PrepCopy is the prep copy: (px << shl) + offset, in uint16 lanes (the
// result stays below 2^16 at 10 and 12 bits).
//
//go:nocheckptr
func mc16PrepCopy(src unsafe.Pointer, ss, w, h, shl int, out unsafe.Pointer) {
	mul := archsimd.BroadcastInt16x16(int16(1) << shl)
	off := archsimd.BroadcastInt16x16(mc16PrepOffset) // wraps mod 2^16 like the uint16 sum
	for y := 0; y < h; y++ {
		s := unsafe.Add(src, y*ss)
		o := unsafe.Add(out, 2*y*w)
		c := 0
		for ; c+16 <= w; c += 16 {
			mc16Ld16(unsafe.Add(s, 2*c)).Mul(mul).Add(off).StoreArray((*[16]int16)(unsafe.Add(o, 2*c)))
		}
		if c < w {
			v := mc16Ld8(unsafe.Add(s, 2*c)).Mul(mul).Add(off)
			if w-c == 8 {
				v.GetLo().StoreArray((*[8]int16)(unsafe.Add(o, 2*c)))
			} else {
				*(*uint64)(unsafe.Add(o, 2*c)) = v.GetLo().AsUint64x2().GetElem(0)
			}
		}
	}
	archsimd.ClearAVXUpperBits()
}
