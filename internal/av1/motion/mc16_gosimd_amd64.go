// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && amd64 && !purego

package motion

import (
	"encoding/binary"
	"simd/archsimd"

	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/unsafeutil"
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

// mc16IM is the int32 2D intermediate: ConvolveScratch.imHBD and
// CompoundConvolveScratch.im share this shape.
type mc16IM = [compoundIMMaxSamples]int32

// mc16SourceAVX2 resolves the eight-tap source window of a w x h block: an
// axis with a filter reads from three samples before the block. The kernels
// load sixteen (or eight, for the tails and w == 4) samples per tap, so the
// window must stay readable for max(w, 8) + 7 samples per row when filtered
// horizontally (max(w, 8) otherwise). Overhanging blocks materialize the
// window with emuEdge16. The returned window is exactly bh rows of that
// footprint, so the kernels' checked row loads stay inside it.
func mc16SourceAVX2(ref *frame.Plane, refX, refY, w, h int, hx, hy bool, edge *emuEdge16Buf) (src []byte, stride int, ok bool) {
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
	if ref.Width <= 0 || ref.Height <= 0 || ref.Stride < 2*ref.Width || ref.Stride > mc8MaxStride {
		return nil, 0, false
	}
	if x0 >= 0 && y0 >= 0 && x0+bw <= ref.Width && y0+bh <= ref.Height {
		start := y0*ref.Stride + 2*x0
		if end := start + (bh-1)*ref.Stride + read; end <= len(ref.Pix) {
			return ref.Pix[start:end], ref.Stride, true
		}
	}
	if len(ref.Pix) < (ref.Height-1)*ref.Stride+2*ref.Width {
		return nil, 0, false
	}
	// emuEdge16Rows (136) rows of emuEdge16Stride (144) samples hold the
	// window (bw <= 135, bh <= 135) and the kernels' over-read.
	end := (bh-1)*2*emuEdge16Stride + read
	if end > len(edge) {
		return nil, 0, false
	}
	emuEdge16(bw, bh, *ref, x0, y0, edge[:], 2*emuEdge16Stride)
	return edge[:end], 2 * emuEdge16Stride, true
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
	dOff := dstY*dst.Stride + 2*dstX
	if dstX < 0 || dstY < 0 || dstX+w > dst.Width || dstY+h > dst.Height || dst.Stride < 2*dst.Width || dst.Stride > mc8MaxStride ||
		!unsafeutil.GridFit(len(dst.Pix), dOff, dst.Stride, 2*w, h) {
		return false
	}
	src, stride, ok := mc16SourceAVX2(ref, refX, refY, w, h, subX != 0, subY != 0, &scratch.edge16)
	if !ok {
		return false
	}
	dg := unsafeutil.NewGrid(dst.Pix, dOff, dst.Stride, 2*w, h)
	maxV := int16(1)<<bitDepth - 1
	round0, round1 := highBDRoundBits(uint8(bitDepth))
	px := &mc16Pairs[mc8Kind(filters.X, filterW)][subX]
	py := &mc16Pairs[mc8Kind(filters.Y, filterH)][subY]
	switch {
	case subX != 0 && subY != 0:
		mc16HToI32(src, stride, w, h+7, px, round0, &scratch.imHBD)
		mc16VFromI32(&scratch.imHBD, w, h, py, true, round1, maxV, dg, nil)
	case subX != 0:
		mc16H(src, stride, w, h, px, true, round0, maxV, dst.Pix[dOff:dOff+(h-1)*dst.Stride+2*w], nil, dst.Stride)
	default:
		if !unsafeutil.GridFit(len(src), 0, stride, 2*max(w, 8), h+7) {
			return false
		}
		mc16V(unsafeutil.NewGrid(src, 0, stride, 2*max(w, 8), h+7), w, h, py, true, 0, maxV, dg, nil)
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
	o := out[:w*h]
	px := &mc16Pairs[mc8Kind(filters.X, filterW)][subX]
	py := &mc16Pairs[mc8Kind(filters.Y, filterH)][subY]
	switch {
	case subX != 0 && subY != 0:
		im := (*mc16IM)(&scratch.im)
		mc16HToI32(src, stride, w, h+7, px, rc, im)
		mc16VFromI32(im, w, h, py, false, 0, 0, unsafeutil.Grid[uint8]{}, o)
	case subX != 0:
		mc16H(src, stride, w, h, px, false, rc, 0, nil, o, w)
	case subY != 0:
		if !unsafeutil.GridFit(len(src), 0, stride, 2*max(w, 8), h+7) {
			return false
		}
		mc16V(unsafeutil.NewGrid(src, 0, stride, 2*max(w, 8), h+7), w, h, py, false, rc, 0, unsafeutil.Grid[uint8]{}, o)
	default:
		if !unsafeutil.GridFit(len(src), 0, stride, 2*max(w, 8), h) {
			return false
		}
		mc16PrepCopy(unsafeutil.NewGrid(src, 0, stride, 2*max(w, 8), h), w, h, 7-rc, o)
	}
	return true
}

// mc16TapPairs16 broadcasts the four tap pairs.
func mc16TapPairs16(p *[4]int32) (k0, k1, k2, k3 archsimd.Int16x16) {
	return archsimd.BroadcastInt32x8(p[0]).AsInt16x16(), archsimd.BroadcastInt32x8(p[1]).AsInt16x16(),
		archsimd.BroadcastInt32x8(p[2]).AsInt16x16(), archsimd.BroadcastInt32x8(p[3]).AsInt16x16()
}

// Sample windows: offsets are in bytes of the 16-bit plane rows (or in
// elements of the uint16 / int32 buffers); the kernels bound their dimensions
// first (mc8Dims), so each checked window is one compare.

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

// mc16HDot loads the eight shifted windows of the sixteen outputs at s[o:]
// (one checked 46-byte window) and returns their tap sums.
func mc16HDot(s []byte, o int, k0, k1, k2, k3 archsimd.Int16x16) (lo, hi archsimd.Int32x8) {
	x := (*[46]byte)(s[uint(o) : uint(o)+46])
	l := func(k int) archsimd.Int16x16 {
		return archsimd.LoadUint8x32Array((*[32]byte)(x[k : k+32])).AsInt16x16()
	}
	return mc16Dot8(l(0), l(2), l(4), l(6), l(8), l(10), l(12), l(14), k0, k1, k2, k3)
}

// mc16HDot8 is mc16HDot for eight outputs (one checked 30-byte window).
func mc16HDot8(s []byte, o int, k0, k1, k2, k3 archsimd.Int16x16) (lo, hi archsimd.Int32x8) {
	x := (*[30]byte)(s[uint(o) : uint(o)+30])
	var z archsimd.Int16x16
	l := func(k int) archsimd.Int16x16 {
		return z.SetLo(archsimd.LoadUint8x16Array((*[16]byte)(x[k : k+16])).AsInt16x8())
	}
	return mc16Dot8(l(0), l(2), l(4), l(6), l(8), l(10), l(12), l(14), k0, k1, k2, k3)
}

// mc16Out finishes sixteen (n = 16), eight or four sums in the in-lane order
// and stores them: put clips (lo, hi) >> sh to [0, maxV] into the plane bytes
// d at byte offset do; prep adds the CONV_BUF offset, keeps the low 16 bits
// (libaom's uint16 cast) and stores into out at element offset oo.
func mc16Out(lo, hi archsimd.Int32x8, n int, put bool, shV archsimd.Uint32x8, maxV archsimd.Int16x16, d []byte, do int, out []uint16, oo int) {
	lo, hi = lo.ShiftRight(shV), hi.ShiftRight(shV)
	var v archsimd.Int16x16
	if put {
		v = lo.SaturateToInt16ConcatGrouped(hi).Max(archsimd.Int16x16{}).Min(maxV)
		switch n {
		case 16:
			v.AsUint8x32().StoreArray((*[32]byte)(d[uint(do) : uint(do)+32]))
		case 8:
			v.GetLo().AsUint8x16().StoreArray((*[16]byte)(d[uint(do) : uint(do)+16]))
		default:
			binary.LittleEndian.PutUint64(d[uint(do):uint(do)+8], v.GetLo().AsUint64x2().GetElem(0))
		}
		return
	}
	off := archsimd.BroadcastInt32x8(mc16PrepOffset)
	mask := archsimd.BroadcastInt32x8(0xffff)
	u := lo.Add(off).And(mask).SaturateToUint16ConcatGrouped(hi.Add(off).And(mask))
	switch n {
	case 16:
		u.StoreArray((*[16]uint16)(out[uint(oo) : uint(oo)+16]))
	case 8:
		u.GetLo().StoreArray((*[8]uint16)(out[uint(oo) : uint(oo)+8]))
	default:
		mc8StU16x4(out, oo, u.GetLo().ReshapeToUint64s().GetElem(0))
	}
}

// mc16H is the horizontal-only pass: put (clip((S + rnd) >> 7)) into dst
// (row stride ds bytes) or prep (((S + rnd) >> rc) + offset) into out (row
// stride ds elements). r is round0 (put) or rc (prep).
func mc16H(src []byte, ss, w, h int, pairs *[4]int32, put bool, r int, maxPix int16, dst []byte, out []uint16, ds int) {
	mc8Dims(w, h, ss, ds)
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
		s, d := y*ss, y*ds
		for uc := uint(0); uc < uint(w&^15); uc += 16 {
			c := int(uc)
			lo, hi := mc16HDot(src, s+2*c, k0, k1, k2, k3)
			mc16Out(lo.Add(rndV), hi.Add(rndV), 16, put, shV, maxV, dst, d+2*c, out, d+c)
		}
		if c := w &^ 15; c < w {
			lo, hi := mc16HDot8(src, s+2*c, k0, k1, k2, k3)
			mc16Out(lo.Add(rndV), hi.Add(rndV), w-c, put, shV, maxV, dst, d+2*c, out, d+c)
		}
	}
	archsimd.ClearAVXUpperBits()
}

// mc16V is the vertical-only pass (put: clip((S + 64) >> 7) into dg; prep:
// as mc16H into out), sliding an eight-row window down each column strip of
// sg, the h+7 source rows. The strips are Grid views (sixteen samples, or
// eight for the tail), their rows named by the output row.
func mc16V(sg unsafeutil.Grid[uint8], w, h int, pairs *[4]int32, put bool, rc int, maxPix int16, dg unsafeutil.Grid[uint8], out []uint16) {
	mc16VCols(sg, w, h, pairs, put, rc, maxPix, dg, out)
	archsimd.ClearAVXUpperBits()
}

func mc16VCols(sg unsafeutil.Grid[uint8], w, h int, pairs *[4]int32, put bool, rc int, maxPix int16, dg unsafeutil.Grid[uint8], out []uint16) {
	if uint(h-1) >= mc8MaxRows {
		return
	}
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
	off := archsimd.BroadcastInt32x8(mc16PrepOffset)
	mask := archsimd.BroadcastInt32x8(0xffff)
	var zero archsimd.Int16x16
	c := 0
	for ; c+16 <= w; c += 16 {
		if !sg.Fits(0, 2*c, 32, h+7) || (put && !dg.Fits(0, 2*c, 32, h)) {
			return
		}
		s := sg.Sub(0, 2*c, 32, h+7)
		r0 := archsimd.LoadUint8x32Array(s.Row32(0, 0)).AsInt16x16()
		r1 := archsimd.LoadUint8x32Array(s.Row32(1, 0)).AsInt16x16()
		r2 := archsimd.LoadUint8x32Array(s.Row32(2, 0)).AsInt16x16()
		r3 := archsimd.LoadUint8x32Array(s.Row32(3, 0)).AsInt16x16()
		r4 := archsimd.LoadUint8x32Array(s.Row32(4, 0)).AsInt16x16()
		r5 := archsimd.LoadUint8x32Array(s.Row32(5, 0)).AsInt16x16()
		r6 := archsimd.LoadUint8x32Array(s.Row32(6, 0)).AsInt16x16()
		t := s.Sub(7, 0, 32, h)
		var d unsafeutil.Grid[uint8]
		if put {
			d = dg.Sub(0, 2*c, 32, h)
		}
		for y := 0; y < h; y++ {
			r7 := archsimd.LoadUint8x32Array(t.Row32(y, 0)).AsInt16x16()
			lo, hi := mc16Dot8(r0, r1, r2, r3, r4, r5, r6, r7, k0, k1, k2, k3)
			lo, hi = lo.Add(rndV).ShiftRight(shV), hi.Add(rndV).ShiftRight(shV)
			if put {
				lo.SaturateToInt16ConcatGrouped(hi).Max(zero).Min(maxV).AsUint8x32().StoreArray(d.Row32(y, 0))
			} else {
				o := y*w + c
				lo.Add(off).And(mask).SaturateToUint16ConcatGrouped(hi.Add(off).And(mask)).StoreArray((*[16]uint16)(out[uint(o) : uint(o)+16]))
			}
			r0, r1, r2, r3, r4, r5, r6 = r1, r2, r3, r4, r5, r6, r7
		}
	}
	if n := w - c; n > 0 {
		// Eight (or, at w == 4, four) columns: sixteen-byte rows.
		if !sg.Fits(0, 2*c, 16, h+7) || (put && !dg.Fits(0, 2*c, 2*n, h)) {
			return
		}
		s := sg.Sub(0, 2*c, 16, h+7)
		r0 := zero.SetLo(archsimd.LoadUint8x16Array(s.Row16(0, 0)).AsInt16x8())
		r1 := zero.SetLo(archsimd.LoadUint8x16Array(s.Row16(1, 0)).AsInt16x8())
		r2 := zero.SetLo(archsimd.LoadUint8x16Array(s.Row16(2, 0)).AsInt16x8())
		r3 := zero.SetLo(archsimd.LoadUint8x16Array(s.Row16(3, 0)).AsInt16x8())
		r4 := zero.SetLo(archsimd.LoadUint8x16Array(s.Row16(4, 0)).AsInt16x8())
		r5 := zero.SetLo(archsimd.LoadUint8x16Array(s.Row16(5, 0)).AsInt16x8())
		r6 := zero.SetLo(archsimd.LoadUint8x16Array(s.Row16(6, 0)).AsInt16x8())
		t := s.Sub(7, 0, 16, h)
		var d unsafeutil.Grid[uint8]
		if put {
			d = dg.Sub(0, 2*c, 2*n, h)
		}
		for y := 0; y < h; y++ {
			r7 := zero.SetLo(archsimd.LoadUint8x16Array(t.Row16(y, 0)).AsInt16x8())
			lo, hi := mc16Dot8(r0, r1, r2, r3, r4, r5, r6, r7, k0, k1, k2, k3)
			lo, hi = lo.Add(rndV).ShiftRight(shV), hi.Add(rndV).ShiftRight(shV)
			o := y*w + c
			switch {
			case put && n == 8:
				lo.SaturateToInt16ConcatGrouped(hi).Max(zero).Min(maxV).GetLo().AsUint8x16().StoreArray(d.Row16(y, 0))
			case put:
				binary.LittleEndian.PutUint64(d.Row8(y, 0)[:], lo.SaturateToInt16ConcatGrouped(hi).Max(zero).Min(maxV).GetLo().AsUint64x2().GetElem(0))
			case n == 8:
				lo.Add(off).And(mask).SaturateToUint16ConcatGrouped(hi.Add(off).And(mask)).GetLo().StoreArray((*[8]uint16)(out[uint(o) : uint(o)+8]))
			default:
				mc8StU16x4(out, o, lo.Add(off).And(mask).SaturateToUint16ConcatGrouped(hi.Add(off).And(mask)).GetLo().ReshapeToUint64s().GetElem(0))
			}
			r0, r1, r2, r3, r4, r5, r6 = r1, r2, r3, r4, r5, r6, r7
		}
	}
}

// mc16HToI32 writes rows x w rounded horizontal sums m = (S + 2^(r-1)) >> r
// as int32 with row stride w (r is round0 for put, rc for prep).
func mc16HToI32(src []byte, ss, w, rows int, pairs *[4]int32, r int, im *mc16IM) {
	mc8Dims(w, rows, ss, 0)
	_ = im[0] // one nil check, ahead of the loops
	k0, k1, k2, k3 := mc16TapPairs16(pairs)
	rndV := archsimd.BroadcastInt32x8(int32(1) << (r - 1))
	shV := archsimd.BroadcastUint32x8(uint32(r))
	for y := 0; y < rows; y++ {
		s, o := y*ss, y*w
		for uc := uint(0); uc < uint(w&^15); uc += 16 {
			c := int(uc)
			lo, hi := mc16HDot(src, s+2*c, k0, k1, k2, k3)
			lo, hi = lo.Add(rndV).ShiftRight(shV), hi.Add(rndV).ShiftRight(shV)
			// Undo the in-lane order: outputs 0-7, then 8-15.
			p := (*[16]int32)(im[uint(o+c) : uint(o+c)+16])
			lo.ConcatPermute128Scalars(0, 2, hi).StoreArray((*[8]int32)(p[:8]))
			lo.ConcatPermute128Scalars(1, 3, hi).StoreArray((*[8]int32)(p[8:]))
		}
		if c := w &^ 15; c < w {
			lo, hi := mc16HDot8(src, s+2*c, k0, k1, k2, k3)
			lo, hi = lo.Add(rndV).ShiftRight(shV), hi.Add(rndV).ShiftRight(shV)
			v := lo.ConcatPermute128Scalars(0, 2, hi)
			if w-c == 8 {
				v.StoreArray((*[8]int32)(im[uint(o+c) : uint(o+c)+8]))
			} else {
				v.GetLo().StoreArray((*[4]int32)(im[uint(o+c) : uint(o+c)+4]))
			}
		}
	}
	archsimd.ClearAVXUpperBits()
}

// mc16PairOrder is the vpermd order that restores sample order after a
// vpackssdw/vpackusdw of two eight-column int32 vectors.
var mc16PairOrder = [8]uint32{0, 1, 4, 5, 2, 3, 6, 7}

// mc16VFromI32 is the 2D vertical pass over the int32 intermediate (row
// stride w, h+7 rows, a Grid view whose eight tap rows of output row y are
// row y of eight h-row sub-views, one per tap): sum(f * m) in int32 (vpmulld), then put
// clips (sum + 2^(round1-1)) >> round1 into dg and prep stores
// (sum + 2^21 + 2^20 + 64) >> 7 as uint16 into out (row stride w).
func mc16VFromI32(im *mc16IM, w, h int, pairs *[4]int32, put bool, round1 int, maxPix int16, dg unsafeutil.Grid[uint8], out []uint16) {
	mc16VFromI32Cols(im, w, h, pairs, put, round1, maxPix, dg, out)
	archsimd.ClearAVXUpperBits()
}

func mc16VFromI32Cols(im *mc16IM, w, h int, pairs *[4]int32, put bool, round1 int, maxPix int16, dg unsafeutil.Grid[uint8], out []uint16) {
	if uint(h-1) >= mc8MaxRows || uint(w-1) >= maxBlockSize || !unsafeutil.GridFit(len(im), 0, w, w, h+7) {
		return
	}
	g := unsafeutil.NewGrid(im[:], 0, w, w, h+7)
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
	c := 0
	for ; c+8 <= w; c += 8 {
		if !g.Fits(0, c, 8, h+7) || (put && !dg.Fits(0, 2*c, 16, h)) {
			return
		}
		s0 := g.Sub(0, c, 8, h)
		s1 := g.Sub(1, c, 8, h)
		s2 := g.Sub(2, c, 8, h)
		s3 := g.Sub(3, c, 8, h)
		s4 := g.Sub(4, c, 8, h)
		s5 := g.Sub(5, c, 8, h)
		s6 := g.Sub(6, c, 8, h)
		s7 := g.Sub(7, c, 8, h)
		var d unsafeutil.Grid[uint8]
		if put {
			d = dg.Sub(0, 2*c, 16, h)
		}
		for y := 0; y < h; y++ {
			acc := rndV.Add(archsimd.LoadInt32x8Array(s0.Row8(y, 0)).Mul(f[0]))
			acc = acc.Add(archsimd.LoadInt32x8Array(s1.Row8(y, 0)).Mul(f[1]))
			acc = acc.Add(archsimd.LoadInt32x8Array(s2.Row8(y, 0)).Mul(f[2]))
			acc = acc.Add(archsimd.LoadInt32x8Array(s3.Row8(y, 0)).Mul(f[3]))
			acc = acc.Add(archsimd.LoadInt32x8Array(s4.Row8(y, 0)).Mul(f[4]))
			acc = acc.Add(archsimd.LoadInt32x8Array(s5.Row8(y, 0)).Mul(f[5]))
			acc = acc.Add(archsimd.LoadInt32x8Array(s6.Row8(y, 0)).Mul(f[6]))
			acc = acc.Add(archsimd.LoadInt32x8Array(s7.Row8(y, 0)).Mul(f[7]))
			acc = acc.ShiftRight(shV)
			if put {
				acc.SaturateToInt16ConcatGrouped(acc).AsInt32x8().Permute(perm).AsInt16x16().Max(archsimd.Int16x16{}).Min(maxV).GetLo().AsUint8x16().StoreArray(d.Row16(y, 0))
			} else {
				a := acc.And(mask)
				o := y*w + c
				a.SaturateToUint16ConcatGrouped(a).AsInt32x8().Permute(perm).AsUint16x16().GetLo().StoreArray((*[8]uint16)(out[uint(o) : uint(o)+8]))
			}
		}
	}
	if c < w {
		// Four columns (w == 4).
		if !g.Fits(0, c, 4, h+7) || (put && !dg.Fits(0, 2*c, 8, h)) {
			return
		}
		s0 := g.Sub(0, c, 4, h)
		s1 := g.Sub(1, c, 4, h)
		s2 := g.Sub(2, c, 4, h)
		s3 := g.Sub(3, c, 4, h)
		s4 := g.Sub(4, c, 4, h)
		s5 := g.Sub(5, c, 4, h)
		s6 := g.Sub(6, c, 4, h)
		s7 := g.Sub(7, c, 4, h)
		var d unsafeutil.Grid[uint8]
		if put {
			d = dg.Sub(0, 2*c, 8, h)
		}
		var z archsimd.Int32x8
		for y := 0; y < h; y++ {
			acc := rndV.Add(z.SetLo(archsimd.LoadInt32x4Array(s0.Row4(y, 0))).Mul(f[0]))
			acc = acc.Add(z.SetLo(archsimd.LoadInt32x4Array(s1.Row4(y, 0))).Mul(f[1]))
			acc = acc.Add(z.SetLo(archsimd.LoadInt32x4Array(s2.Row4(y, 0))).Mul(f[2]))
			acc = acc.Add(z.SetLo(archsimd.LoadInt32x4Array(s3.Row4(y, 0))).Mul(f[3]))
			acc = acc.Add(z.SetLo(archsimd.LoadInt32x4Array(s4.Row4(y, 0))).Mul(f[4]))
			acc = acc.Add(z.SetLo(archsimd.LoadInt32x4Array(s5.Row4(y, 0))).Mul(f[5]))
			acc = acc.Add(z.SetLo(archsimd.LoadInt32x4Array(s6.Row4(y, 0))).Mul(f[6]))
			acc = acc.Add(z.SetLo(archsimd.LoadInt32x4Array(s7.Row4(y, 0))).Mul(f[7]))
			acc = acc.ShiftRight(shV)
			if put {
				v := acc.SaturateToInt16ConcatGrouped(acc).AsInt32x8().Permute(perm).AsInt16x16().Max(archsimd.Int16x16{}).Min(maxV)
				binary.LittleEndian.PutUint64(d.Row8(y, 0)[:], v.GetLo().AsUint64x2().GetElem(0))
			} else {
				a := acc.And(mask)
				v := a.SaturateToUint16ConcatGrouped(a).AsInt32x8().Permute(perm).AsUint16x16()
				mc8StU16x4(out, y*w+c, v.GetLo().ReshapeToUint64s().GetElem(0))
			}
		}
	}
}

// mc16PrepCopy is the prep copy: (px << shl) + offset, in uint16 lanes (the
// result stays below 2^16 at 10 and 12 bits), from the h rows of sg.
func mc16PrepCopy(sg unsafeutil.Grid[uint8], w, h, shl int, out []uint16) {
	mc16PrepCopyCols(sg, w, h, shl, out)
	archsimd.ClearAVXUpperBits()
}

func mc16PrepCopyCols(sg unsafeutil.Grid[uint8], w, h, shl int, out []uint16) {
	if uint(h-1) >= mc8MaxRows {
		return
	}
	mul := archsimd.BroadcastInt16x16(int16(1) << shl)
	off := archsimd.BroadcastInt16x16(mc16PrepOffset) // wraps mod 2^16 like the uint16 sum
	c := 0
	for ; c+16 <= w; c += 16 {
		if !sg.Fits(0, 2*c, 32, h) {
			return
		}
		s := sg.Sub(0, 2*c, 32, h)
		for y := 0; y < h; y++ {
			o := y*w + c
			archsimd.LoadUint8x32Array(s.Row32(y, 0)).AsInt16x16().Mul(mul).Add(off).AsUint16x16().StoreArray((*[16]uint16)(out[uint(o) : uint(o)+16]))
		}
	}
	if n := w - c; n > 0 {
		if !sg.Fits(0, 2*c, 16, h) {
			return
		}
		s := sg.Sub(0, 2*c, 16, h)
		for y := 0; y < h; y++ {
			o := y*w + c
			v := archsimd.Int16x16{}.SetLo(archsimd.LoadUint8x16Array(s.Row16(y, 0)).AsInt16x8()).Mul(mul).Add(off).GetLo().AsUint16x8()
			if n == 8 {
				v.StoreArray((*[8]uint16)(out[uint(o) : uint(o)+8]))
			} else {
				mc8StU16x4(out, o, v.ReshapeToUint64s().GetElem(0))
			}
		}
	}
}
