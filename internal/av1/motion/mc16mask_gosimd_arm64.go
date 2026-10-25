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

// mc16BlendAvgFast is the 16bpc plain compound average (fwd = bck = 8) over
// two CONV_BUF predictors with row stride w (w a multiple of 8, or 4 with an
// even height).
func mc16BlendAvgFast(dst *frame.Plane, a, b []uint16, bitDepth, dstX, dstY, w, h int) bool {
	if (bitDepth != 10 && bitDepth != 12) || w <= 0 || (w != 4 && w&7 != 0) || h <= 0 || len(a) < w*h || len(b) < w*h || (w == 4 && h&1 != 0) {
		return false
	}
	off := dstY*dst.Stride + 2*dstX
	if dstX < 0 || dstY < 0 || dstX+w > dst.Width || off+(h-1)*dst.Stride+2*w > len(dst.Pix) {
		return false
	}
	rb, bias := mc16BlendRoundBits(bitDepth)
	mc16BlendAvg(unsafe.Pointer(&dst.Pix[off]), dst.Stride, unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), w, h, rb, bias, uint16(1)<<bitDepth-1)
	return true
}

//go:nocheckptr
func mc16BlendAvg(dst unsafe.Pointer, dstStride int, a, b unsafe.Pointer, w, h, rb int, biasC, maxC uint16) {
	one := archsimd.BroadcastUint16x8(1)
	bias := archsimd.BroadcastUint16x8(biasC)
	maxV := archsimd.BroadcastUint16x8(maxC)
	sh := archsimd.BroadcastInt16x8(int16(-rb))
	if w == 4 {
		for r := 0; r < h; r += 2 {
			o := r * 8
			va := archsimd.LoadUint16x8Array((*[8]uint16)(unsafe.Add(a, o)))
			vb := archsimd.LoadUint16x8Array((*[8]uint16)(unsafe.Add(b, o)))
			v := va.Average(vb).Sub(va.Xor(vb).And(one)).SubSaturated(bias).Shift(sh).Min(maxV).ReshapeToUint64s()
			d := unsafe.Add(dst, r*dstStride)
			*(*uint64)(d) = v.GetElem(0)
			*(*uint64)(unsafe.Add(d, dstStride)) = v.GetElem(1)
		}
		return
	}
	ro := w * 2
	for r := 0; r < h; r++ {
		pa := unsafe.Add(a, r*ro)
		pb := unsafe.Add(b, r*ro)
		d := unsafe.Add(dst, r*dstStride)
		for c := 0; c < ro; c += 16 {
			va := archsimd.LoadUint16x8Array((*[8]uint16)(unsafe.Add(pa, c)))
			vb := archsimd.LoadUint16x8Array((*[8]uint16)(unsafe.Add(pb, c)))
			va.Average(vb).Sub(va.Xor(vb).And(one)).SubSaturated(bias).Shift(sh).Min(maxV).StoreArray((*[8]uint16)(unsafe.Add(d, c)))
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
	if dstX < 0 || dstY < 0 || dstX+w > dst.Width || off+(h-1)*dst.Stride+2*w > len(dst.Pix) {
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
	rb, bias := mc16BlendRoundBits(bitDepth)
	mc16BlendMask(unsafe.Pointer(&dst.Pix[off]), dst.Stride, unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), w, h, unsafe.Pointer(&mask[0]), maskStride, subX,
		uint32(bias)<<(16-rb), uint16(10-rb), int16(1)<<bitDepth-1)
	return true
}

//go:nocheckptr
func mc16BlendMask(dst unsafe.Pointer, dstStride int, a, b unsafe.Pointer, w, h int, mask unsafe.Pointer, maskStride int, sub bool, biasC uint32, mshift uint16, maxPix int16) {
	bias := archsimd.BroadcastUint32x4(biasC)
	m64 := archsimd.BroadcastUint16x8(64)
	two := archsimd.BroadcastUint16x8(2)
	right2 := archsimd.BroadcastInt16x8(-2)
	msh := archsimd.BroadcastInt16x8(int16(mshift))
	zero := archsimd.BroadcastInt16x8(0)
	maxV := archsimd.BroadcastInt16x8(maxPix)
	ro := w * 2
	for r := 0; r < h; r++ {
		pa := unsafe.Add(a, r*ro)
		pb := unsafe.Add(b, r*ro)
		d := unsafe.Add(dst, r*dstStride)
		var mrow, mrow1 unsafe.Pointer
		if sub {
			mrow = unsafe.Add(mask, 2*r*maskStride)
			mrow1 = unsafe.Add(mrow, maskStride)
		} else {
			mrow = unsafe.Add(mask, r*maskStride)
		}
		for c := 0; c < w; c += 8 {
			var m archsimd.Uint16x8
			if sub {
				v0 := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(mrow, 2*c)))
				v1 := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(mrow1, 2*c)))
				lo := v0.ExtendLo8ToUint16().Add(v1.ExtendLo8ToUint16())
				hi := v0.HiToLo().ExtendLo8ToUint16().Add(v1.HiToLo().ExtendLo8ToUint16())
				m = lo.ConcatAddPairs(hi).Add(two).Shift(right2)
			} else {
				m = archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(mrow, c))).ExtendLo8ToUint16()
			}
			ms := m.Shift(msh)
			mi := m64.Sub(m).Shift(msh)
			va := archsimd.LoadUint16x8Array((*[8]uint16)(unsafe.Add(pa, 2*c)))
			vb := archsimd.LoadUint16x8Array((*[8]uint16)(unsafe.Add(pb, 2*c)))
			lo := va.MulWidenLo(ms).Add(vb.MulWidenLo(mi)).Sub(bias)
			hi := va.HiToLo().MulWidenLo(ms.HiToLo()).Add(vb.HiToLo().MulWidenLo(mi.HiToLo())).Sub(bias)
			v := lo.ReshapeToUint16s().ConcatOdd(hi.ReshapeToUint16s()).BitsToInt16()
			v.Max(zero).Min(maxV).StoreArray((*[8]int16)(unsafe.Add(d, 2*c)))
		}
	}
}

// mc16DiffWtdMaskFast builds the 10- or 12-bit difference-weighted mask; width
// must be a multiple of 8.
func mc16DiffWtdMaskFast(mask []byte, maskStride int, a, b []uint16, bitDepth, w, h int, invert bool) bool {
	if (bitDepth != 10 && bitDepth != 12) || w&7 != 0 || w > maxBlockSize || h <= 0 || len(a) < w*h || len(b) < w*h ||
		maskStride < w || (h-1)*maskStride+w > len(mask) {
		return false
	}
	mc16DiffWtdMask(unsafe.Pointer(&mask[0]), maskStride, unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), w, h, invert)
	return true
}

//go:nocheckptr
func mc16DiffWtdMask(mask unsafe.Pointer, maskStride int, a, b unsafe.Pointer, w, h int, invert bool) {
	right5 := archsimd.BroadcastInt16x8(-5)
	one := archsimd.BroadcastUint16x8(1)
	base := archsimd.BroadcastUint16x8(38)
	maxM := archsimd.BroadcastUint16x8(64)
	ro := w * 2
	for r := 0; r < h; r++ {
		pa := unsafe.Add(a, r*ro)
		pb := unsafe.Add(b, r*ro)
		mrow := unsafe.Add(mask, r*maskStride)
		for c := 0; c < w; c += 8 {
			va := archsimd.LoadUint16x8Array((*[8]uint16)(unsafe.Add(pa, 2*c)))
			vb := archsimd.LoadUint16x8Array((*[8]uint16)(unsafe.Add(pb, 2*c)))
			diff := va.Max(vb).Sub(va.Min(vb))
			m := diff.Shift(right5).Add(one).Shift(right5).Add(base).Min(maxM)
			if invert {
				m = maxM.Sub(m)
			}
			*(*uint64)(unsafe.Add(mrow, c)) = m.TruncToUint8().ReshapeToUint64s().GetElem(0)
		}
	}
}
