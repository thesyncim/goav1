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
