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

// AVX2 16bpc compound blends (dav1d src/x86/mc16_avx2.asm avg / mask /
// w_mask), the 10- and 12-bit siblings of mc8mask_gosimd_amd64.go. They
// compute libaom's CONV_BUF formulas exactly for every uint16 input, with
// rb = 2*7 - round0 - 7 (4 at 10 bits, 2 at 12) and roundOffset = 2^14 + 2^13:
//
//	avg:  clip((((a + b) >> 1) - roundOffset + 2^(rb-1)) >> rb)
//	      floor((a + b) / 2) is the rounding vpavgw minus (a ^ b) & 1, and the
//	      bias is a saturating subtract (a negative result clips to 0 either
//	      way); >> rb is vpmulhuw by 2^(16-rb).
//	mask: clip((m*a + (64-m)*b - 64*(roundOffset - 2^(rb-1))) >> (6 + rb)),
//	      vpmaddwd on the samples biased by -2^15 (exact in int32), the bias
//	      folded into the constant.
//	diff-weighted mask: min(38 + (((|a-b| >> 5) + 1) >> 5), 64).

// mc16BlendRoundBits returns rb and the avg bias roundOffset - 2^(rb-1).
func mc16BlendRoundBits(bitDepth int) (rb int, bias uint16) {
	rb = 2*filterBits - compoundRound0(uint8(bitDepth)) - compoundRound1Bits
	return rb, uint16(mc16PrepOffset - 1<<(rb-1))
}

// mc16BlendDst checks the destination window of a w x h 16bpc blend.
func mc16BlendDst(dst *frame.Plane, dstX, dstY, w, h int) (int, bool) {
	off := dstY*dst.Stride + 2*dstX
	return off, dstX >= 0 && dstY >= 0 && dstX+w <= dst.Width && off+(h-1)*dst.Stride+2*w <= len(dst.Pix)
}

// mc16BlendAvgFast is the 16bpc plain compound average (fwd = bck = 8) over
// two CONV_BUF predictors with row stride w (w a multiple of 8, or 4 with an
// even height).
func mc16BlendAvgFast(dst *frame.Plane, a, b []uint16, bitDepth, dstX, dstY, w, h int) bool {
	if !mc8AVX2 || (bitDepth != 10 && bitDepth != 12) || w <= 0 || (w != 4 && w&7 != 0) || h <= 0 || len(a) < w*h || len(b) < w*h || (w == 4 && h&1 != 0) {
		return false
	}
	off, ok := mc16BlendDst(dst, dstX, dstY, w, h)
	if !ok {
		return false
	}
	rb, bias := mc16BlendRoundBits(bitDepth)
	mc16BlendAvgAVX2(unsafe.Pointer(&dst.Pix[off]), dst.Stride, unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), w, h, rb, bias, uint16(1)<<bitDepth-1)
	return true
}

// mc16Avg16 is the avg formula on sixteen lanes.
func mc16Avg16(va, vb, one, bias, mul, maxV archsimd.Uint16x16) archsimd.Uint16x16 {
	return va.Average(vb).Sub(va.Xor(vb).And(one)).SubSaturated(bias).MulHigh(mul).Min(maxV)
}

//go:nocheckptr
func mc16BlendAvgAVX2(dst unsafe.Pointer, ds int, a, b unsafe.Pointer, w, h, rb int, biasC, maxC uint16) {
	one := archsimd.BroadcastUint16x16(1)
	bias := archsimd.BroadcastUint16x16(biasC)
	mul := archsimd.BroadcastUint16x16(uint16(1) << (16 - rb))
	maxV := archsimd.BroadcastUint16x16(maxC)
	ld := func(p unsafe.Pointer) archsimd.Uint16x16 { return archsimd.LoadUint16x16Array((*[16]uint16)(p)) }
	switch {
	case w == 4:
		// Four rows of four per vector (the CONV_BUF rows are contiguous);
		// a height of 2 mod 4 finishes with eight lanes.
		r := 0
		for ; r+4 <= h; r += 4 {
			v := mc16Avg16(ld(unsafe.Add(a, 8*r)), ld(unsafe.Add(b, 8*r)), one, bias, mul, maxV).AsUint64x4()
			d := unsafe.Add(dst, r*ds)
			*(*uint64)(d) = v.GetLo().GetElem(0)
			*(*uint64)(unsafe.Add(d, ds)) = v.GetLo().GetElem(1)
			*(*uint64)(unsafe.Add(d, 2*ds)) = v.GetHi().GetElem(0)
			*(*uint64)(unsafe.Add(d, 3*ds)) = v.GetHi().GetElem(1)
		}
		if r < h {
			va := archsimd.Uint16x16{}.SetLo(archsimd.LoadUint16x8Array((*[8]uint16)(unsafe.Add(a, 8*r))))
			vb := archsimd.Uint16x16{}.SetLo(archsimd.LoadUint16x8Array((*[8]uint16)(unsafe.Add(b, 8*r))))
			v := mc16Avg16(va, vb, one, bias, mul, maxV).GetLo().AsUint64x2()
			d := unsafe.Add(dst, r*ds)
			*(*uint64)(d) = v.GetElem(0)
			*(*uint64)(unsafe.Add(d, ds)) = v.GetElem(1)
		}
	default:
		ro := 2 * w
		for r := 0; r < h; r++ {
			pa := unsafe.Add(a, r*ro)
			pb := unsafe.Add(b, r*ro)
			d := unsafe.Add(dst, r*ds)
			c := 0
			for ; c+32 <= ro; c += 32 {
				mc16Avg16(ld(unsafe.Add(pa, c)), ld(unsafe.Add(pb, c)), one, bias, mul, maxV).StoreArray((*[16]uint16)(unsafe.Add(d, c)))
			}
			if c < ro {
				va := archsimd.Uint16x16{}.SetLo(archsimd.LoadUint16x8Array((*[8]uint16)(unsafe.Add(pa, c))))
				vb := archsimd.Uint16x16{}.SetLo(archsimd.LoadUint16x8Array((*[8]uint16)(unsafe.Add(pb, c))))
				mc16Avg16(va, vb, one, bias, mul, maxV).GetLo().StoreArray((*[8]uint16)(unsafe.Add(d, c)))
			}
		}
	}
	archsimd.ClearAVXUpperBits()
}

// mc16BlendMaskFast blends with a full-resolution (4:4:4 / luma) or 4:2:0
// subsampled mask; width must be a multiple of 8.
func mc16BlendMaskFast(dst frame.Plane, a, b []uint16, bitDepth, dstX, dstY, w, h int, mask []byte, maskStride int, subX, subY bool) bool {
	if !mc8AVX2 || (bitDepth != 10 && bitDepth != 12) || w&7 != 0 || w > maxBlockSize || h <= 0 || len(a) < w*h || len(b) < w*h || subX != subY {
		return false
	}
	off, ok := mc16BlendDst(&dst, dstX, dstY, w, h)
	if !ok {
		return false
	}
	mw, mh := w, h
	if subX {
		mw, mh = 2*w, 2*h
	}
	// The kernels load sixteen mask bytes per eight (unsubsampled) or
	// sixteen (subsampled) output columns.
	if maskStride < mw || (mh-1)*maskStride+mw > len(mask) || (mh-1)*maskStride+mw+8 > cap(mask) {
		return false
	}
	mask = mask[:cap(mask)]
	rb, bias := mc16BlendRoundBits(bitDepth)
	mc16BlendMaskAVX2(unsafe.Pointer(&dst.Pix[off]), dst.Stride, unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), w, h, unsafe.Pointer(&mask[0]), maskStride, subX,
		64<<15-64*int32(bias), uint32(6+rb), int16(1)<<bitDepth-1)
	return true
}

// mc16MaskBlend16 blends sixteen samples with the int16 weights m (0..64).
func mc16MaskBlend16(va, vb, m, m64, bias16 archsimd.Int16x16, c, z archsimd.Int32x8, shV archsimd.Uint32x8, maxV archsimd.Int16x16) archsimd.Int16x16 {
	// Biased samples a - 2^15 fit int16: m*a + (64-m)*b
	// = m*a' + (64-m)*b' + 64*2^15.
	sa, sb := va.Sub(bias16), vb.Sub(bias16)
	mi := m64.Sub(m)
	lo := sa.InterleaveLoGrouped(sb).DotProductPairs(m.InterleaveLoGrouped(mi)).Add(c).ShiftRight(shV)
	hi := sa.InterleaveHiGrouped(sb).DotProductPairs(m.InterleaveHiGrouped(mi)).Add(c).ShiftRight(shV)
	return lo.SaturateToInt16ConcatGrouped(hi).Max(archsimd.Int16x16{}).Min(maxV)
}

//go:nocheckptr
func mc16BlendMaskAVX2(dst unsafe.Pointer, ds int, a, b unsafe.Pointer, w, h int, mask unsafe.Pointer, maskStride int, sub bool, cC int32, sh uint32, maxPix int16) {
	m64 := archsimd.BroadcastInt16x16(64)
	bias16 := archsimd.BroadcastInt16x16(-1 << 15)
	c := archsimd.BroadcastInt32x8(cC)
	shV := archsimd.BroadcastUint32x8(sh)
	maxV := archsimd.BroadcastInt16x16(maxPix)
	ones := archsimd.BroadcastInt8x32(1)
	two := archsimd.BroadcastInt16x16(2)
	var z archsimd.Int32x8
	ro := 2 * w
	for r := 0; r < h; r++ {
		pa := unsafe.Add(a, r*ro)
		pb := unsafe.Add(b, r*ro)
		d := unsafe.Add(dst, r*ds)
		var mrow, mrow1 unsafe.Pointer
		if sub {
			mrow = unsafe.Add(mask, 2*r*maskStride)
			mrow1 = unsafe.Add(mrow, maskStride)
		} else {
			mrow = unsafe.Add(mask, r*maskStride)
		}
		for col := 0; col < w; col += 16 {
			var m archsimd.Int16x16
			if sub {
				// 2x2 sums: vpmaddubsw against ones adds the column pairs of
				// each row in place.
				var u0, u1 archsimd.Uint8x32
				if w-col >= 16 {
					u0 = archsimd.LoadUint8x32Array((*[32]uint8)(unsafe.Add(mrow, 2*col)))
					u1 = archsimd.LoadUint8x32Array((*[32]uint8)(unsafe.Add(mrow1, 2*col)))
				} else {
					u0 = archsimd.Uint8x32{}.SetLo(archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(mrow, 2*col))))
					u1 = archsimd.Uint8x32{}.SetLo(archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(mrow1, 2*col))))
				}
				v0 := u0.DotProductPairsSaturated(ones)
				v1 := u1.DotProductPairsSaturated(ones)
				m = v0.Add(v1).Add(two).ShiftAllRight(2)
			} else {
				m = archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(mrow, col))).ExtendToUint16().AsInt16x16()
			}
			if w-col >= 16 {
				va := archsimd.LoadInt16x16Array((*[16]int16)(unsafe.Add(pa, 2*col)))
				vb := archsimd.LoadInt16x16Array((*[16]int16)(unsafe.Add(pb, 2*col)))
				mc16MaskBlend16(va, vb, m, m64, bias16, c, z, shV, maxV).StoreArray((*[16]int16)(unsafe.Add(d, 2*col)))
			} else {
				va := archsimd.Int16x16{}.SetLo(archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(pa, 2*col))))
				vb := archsimd.Int16x16{}.SetLo(archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(pb, 2*col))))
				mc16MaskBlend16(va, vb, m, m64, bias16, c, z, shV, maxV).GetLo().StoreArray((*[8]int16)(unsafe.Add(d, 2*col)))
			}
		}
	}
	archsimd.ClearAVXUpperBits()
}

// mc16DiffWtdMaskFast builds the 10- or 12-bit difference-weighted mask; width
// must be a multiple of 8.
func mc16DiffWtdMaskFast(mask []byte, maskStride int, a, b []uint16, bitDepth, w, h int, invert bool) bool {
	if !mc8AVX2 || (bitDepth != 10 && bitDepth != 12) || w&7 != 0 || w > maxBlockSize || h <= 0 || len(a) < w*h || len(b) < w*h ||
		maskStride < w || (h-1)*maskStride+w > len(mask) {
		return false
	}
	mc16DiffWtdMaskAVX2(unsafe.Pointer(&mask[0]), maskStride, unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), w, h, invert)
	return true
}

// mc16LowBytes gathers the low byte of every int16 lane into the low half of
// its 128-bit lane.
var mc16LowBytes = [32]int8{
	0, 2, 4, 6, 8, 10, 12, 14, -1, -1, -1, -1, -1, -1, -1, -1,
	0, 2, 4, 6, 8, 10, 12, 14, -1, -1, -1, -1, -1, -1, -1, -1,
}

//go:nocheckptr
func mc16DiffWtdMaskAVX2(mask unsafe.Pointer, maskStride int, a, b unsafe.Pointer, w, h int, invert bool) {
	one := archsimd.BroadcastUint16x16(1)
	base := archsimd.BroadcastUint16x16(38)
	maxM := archsimd.BroadcastUint16x16(64)
	lowBytes := archsimd.LoadInt8x32Array(&mc16LowBytes)
	ro := 2 * w
	for r := 0; r < h; r++ {
		pa := unsafe.Add(a, r*ro)
		pb := unsafe.Add(b, r*ro)
		mrow := unsafe.Add(mask, r*maskStride)
		for col := 0; col < w; col += 16 {
			n := min(16, w-col)
			var va, vb archsimd.Uint16x16
			if n == 16 {
				va = archsimd.LoadUint16x16Array((*[16]uint16)(unsafe.Add(pa, 2*col)))
				vb = archsimd.LoadUint16x16Array((*[16]uint16)(unsafe.Add(pb, 2*col)))
			} else {
				va = archsimd.Uint16x16{}.SetLo(archsimd.LoadUint16x8Array((*[8]uint16)(unsafe.Add(pa, 2*col))))
				vb = archsimd.Uint16x16{}.SetLo(archsimd.LoadUint16x8Array((*[8]uint16)(unsafe.Add(pb, 2*col))))
			}
			diff := va.Max(vb).Sub(va.Min(vb))
			m := diff.ShiftAllRight(5).Add(one).ShiftAllRight(5).Add(base).Min(maxM)
			if invert {
				m = maxM.Sub(m)
			}
			v := m.AsUint8x32().PermuteOrZeroGrouped(lowBytes)
			*(*uint64)(unsafe.Add(mrow, col)) = v.GetLo().AsUint64x2().GetElem(0)
			if n == 16 {
				*(*uint64)(unsafe.Add(mrow, col+8)) = v.GetHi().AsUint64x2().GetElem(0)
			}
		}
	}
	archsimd.ClearAVXUpperBits()
}
