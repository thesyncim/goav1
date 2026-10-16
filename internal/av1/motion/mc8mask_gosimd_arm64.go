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
	mc8BlendMask(unsafe.Pointer(&dst.Pix[off]), dst.Stride, unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), w, h, unsafe.Pointer(&mask[0]), maskStride, subX)
	return true
}

//go:nocheckptr
func mc8BlendMask(dst unsafe.Pointer, dstStride int, a, b unsafe.Pointer, w, h int, mask unsafe.Pointer, maskStride int, sub bool) {
	bias := archsimd.BroadcastUint32x4(392704 << 6)
	m64 := archsimd.BroadcastUint16x8(64)
	two := archsimd.BroadcastUint16x8(2)
	right2 := archsimd.BroadcastInt16x8(-2)
	six := archsimd.BroadcastInt16x8(6)
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
			ms := m.Shift(six)
			mi := m64.Sub(m).Shift(six)
			va := archsimd.LoadUint16x8Array((*[8]uint16)(unsafe.Add(pa, 2*c)))
			vb := archsimd.LoadUint16x8Array((*[8]uint16)(unsafe.Add(pb, 2*c)))
			lo := va.MulWidenLo(ms).Add(vb.MulWidenLo(mi)).Sub(bias)
			hi := va.HiToLo().MulWidenLo(ms.HiToLo()).Add(vb.HiToLo().MulWidenLo(mi.HiToLo())).Sub(bias)
			v := lo.ReshapeToUint16s().ConcatOdd(hi.ReshapeToUint16s()).BitsToInt16()
			*(*uint64)(unsafe.Add(d, c)) = v.SaturateToUint8().ReshapeToUint64s().GetElem(0)
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
	mc8DiffWtdMask(unsafe.Pointer(&mask[0]), maskStride, unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), w, h, invert)
	return true
}

//go:nocheckptr
func mc8DiffWtdMask(mask unsafe.Pointer, maskStride int, a, b unsafe.Pointer, w, h int, invert bool) {
	right3 := archsimd.BroadcastInt16x8(-3)
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
			m := diff.Shift(right3).Add(one).Shift(right5).Add(base).Min(maxM)
			if invert {
				m = maxM.Sub(m)
			}
			*(*uint64)(unsafe.Add(mrow, c)) = m.TruncToUint8().ReshapeToUint64s().GetElem(0)
		}
	}
}
