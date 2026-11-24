// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && amd64 && !purego

package motion

import (
	"encoding/binary"
	"simd/archsimd"

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
//
// The kernels take the block's rows as slices (the destination window, the
// w*h CONV_BUFs, the mask window) and load and store through one checked
// window per vector, with the dimensions bounded first (mc8Dims).

// mc16BlendRoundBits returns rb and the avg bias roundOffset - 2^(rb-1).
func mc16BlendRoundBits(bitDepth int) (rb int, bias uint16) {
	rb = 2*filterBits - compoundRound0(uint8(bitDepth)) - compoundRound1Bits
	return rb, uint16(mc16PrepOffset - 1<<(rb-1))
}

// mc16BlendDst checks the destination window of a w x h 16bpc blend and
// returns it.
func mc16BlendDst(dst *frame.Plane, dstX, dstY, w, h int) ([]byte, bool) {
	off := dstY*dst.Stride + 2*dstX
	end := off + (h-1)*dst.Stride + 2*w
	if dstX < 0 || dstY < 0 || dstX+w > dst.Width || end > len(dst.Pix) || dst.Stride > mc8MaxStride || w > maxBlockSize || h > maxBlockSize {
		return nil, false
	}
	return dst.Pix[off:end], true
}

// mc16BlendAvgFast is the 16bpc plain compound average (fwd = bck = 8) over
// two CONV_BUF predictors with row stride w (w a multiple of 8, or 4 with an
// even height).
func mc16BlendAvgFast(dst *frame.Plane, a, b []uint16, bitDepth, dstX, dstY, w, h int) bool {
	if !mc8AVX2 || (bitDepth != 10 && bitDepth != 12) || w <= 0 || (w != 4 && w&7 != 0) || h <= 0 || len(a) < w*h || len(b) < w*h || (w == 4 && h&1 != 0) {
		return false
	}
	d, ok := mc16BlendDst(dst, dstX, dstY, w, h)
	if !ok {
		return false
	}
	rb, bias := mc16BlendRoundBits(bitDepth)
	mc16BlendAvgAVX2(d, dst.Stride, a[:w*h], b[:w*h], w, h, rb, bias, uint16(1)<<bitDepth-1)
	return true
}

// mc16Avg16 is the avg formula on sixteen lanes.
func mc16Avg16(va, vb, one, bias, mul, maxV archsimd.Uint16x16) archsimd.Uint16x16 {
	return va.Average(vb).Sub(va.Xor(vb).And(one)).SubSaturated(bias).MulHigh(mul).Min(maxV)
}

func mc16LdU16(s []uint16, o int) archsimd.Uint16x16 {
	return archsimd.LoadUint16x16Array((*[16]uint16)(s[uint(o) : uint(o)+16]))
}

func mc16LdU16x8(s []uint16, o int) archsimd.Uint16x16 {
	return archsimd.Uint16x16{}.SetLo(archsimd.LoadUint16x8Array((*[8]uint16)(s[uint(o) : uint(o)+8])))
}

func mc16BlendAvgAVX2(dst []byte, ds int, a, b []uint16, w, h, rb int, biasC, maxC uint16) {
	mc8Dims(w, h, 0, ds)
	one := archsimd.BroadcastUint16x16(1)
	bias := archsimd.BroadcastUint16x16(biasC)
	mul := archsimd.BroadcastUint16x16(uint16(1) << (16 - rb))
	maxV := archsimd.BroadcastUint16x16(maxC)
	switch {
	case w == 4:
		// Four rows of four per vector (the CONV_BUF rows are contiguous);
		// a height of 2 mod 4 finishes with eight lanes.
		for ur := uint(0); ur < uint(h&^3); ur += 4 {
			r := int(ur)
			v := mc16Avg16(mc16LdU16(a, 4*r), mc16LdU16(b, 4*r), one, bias, mul, maxV).AsUint64x4()
			d := r * ds
			mc8St64(dst, d, v.GetLo().GetElem(0))
			mc8St64(dst, d+ds, v.GetLo().GetElem(1))
			mc8St64(dst, d+2*ds, v.GetHi().GetElem(0))
			mc8St64(dst, d+3*ds, v.GetHi().GetElem(1))
		}
		if r := h &^ 3; r < h {
			v := mc16Avg16(mc16LdU16x8(a, 4*r), mc16LdU16x8(b, 4*r), one, bias, mul, maxV).GetLo().AsUint64x2()
			d := r * ds
			mc8St64(dst, d, v.GetElem(0))
			mc8St64(dst, d+ds, v.GetElem(1))
		}
	default:
		for r := 0; r < h; r++ {
			o, d := r*w, r*ds
			for uc := uint(0); uc < uint(w&^15); uc += 16 {
				c := int(uc)
				v := mc16Avg16(mc16LdU16(a, o+c), mc16LdU16(b, o+c), one, bias, mul, maxV)
				v.AsUint8x32().StoreArray((*[32]byte)(dst[uint(d+2*c) : uint(d+2*c)+32]))
			}
			if c := w &^ 15; c < w {
				v := mc16Avg16(mc16LdU16x8(a, o+c), mc16LdU16x8(b, o+c), one, bias, mul, maxV)
				v.GetLo().AsUint8x16().StoreArray((*[16]byte)(dst[uint(d+2*c) : uint(d+2*c)+16]))
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
	d, ok := mc16BlendDst(&dst, dstX, dstY, w, h)
	if !ok {
		return false
	}
	mw, mh := w, h
	if subX {
		mw, mh = 2*w, 2*h
	}
	// The kernels load sixteen mask bytes per eight (unsubsampled) or
	// sixteen (subsampled) output columns.
	mEnd := (mh-1)*maskStride + mw + 8
	if maskStride < mw || maskStride > mc8MaxStride || mEnd-8 > len(mask) || mEnd > cap(mask) {
		return false
	}
	rb, bias := mc16BlendRoundBits(bitDepth)
	mc16BlendMaskAVX2(d, dst.Stride, a[:w*h], b[:w*h], w, h, mask[:mEnd], maskStride, subX,
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

func mc16BlendMaskAVX2(dst []byte, ds int, a, b []uint16, w, h int, mask []byte, maskStride int, sub bool, cC int32, sh uint32, maxPix int16) {
	mc8Dims(w, h, maskStride, ds)
	m64 := archsimd.BroadcastInt16x16(64)
	bias16 := archsimd.BroadcastInt16x16(-1 << 15)
	c := archsimd.BroadcastInt32x8(cC)
	shV := archsimd.BroadcastUint32x8(sh)
	maxV := archsimd.BroadcastInt16x16(maxPix)
	ones := archsimd.BroadcastInt8x32(1)
	two := archsimd.BroadcastInt16x16(2)
	var z archsimd.Int32x8
	for r := 0; r < h; r++ {
		o, d := r*w, r*ds
		mrow, mrow1 := r*maskStride, 0
		if sub {
			mrow = 2 * r * maskStride
			mrow1 = mrow + maskStride
		}
		for ucol := uint(0); ucol < uint(w); ucol += 16 {
			col := int(ucol)
			full := w-col >= 16
			var m archsimd.Int16x16
			if sub {
				// 2x2 sums: vpmaddubsw against ones adds the column pairs of
				// each row in place.
				var u0, u1 archsimd.Uint8x32
				if full {
					u0 = mc8Ld32B(mask, mrow+2*col)
					u1 = mc8Ld32B(mask, mrow1+2*col)
				} else {
					u0 = archsimd.Uint8x32{}.SetLo(archsimd.LoadUint8x16Array(mc8W16(mask, mrow+2*col)))
					u1 = archsimd.Uint8x32{}.SetLo(archsimd.LoadUint8x16Array(mc8W16(mask, mrow1+2*col)))
				}
				v0 := u0.DotProductPairsSaturated(ones)
				v1 := u1.DotProductPairsSaturated(ones)
				m = v0.Add(v1).Add(two).ShiftAllRight(2)
			} else {
				m = mc8Zx(mask, mrow+col).AsInt16x16()
			}
			if full {
				va := mc16LdU16(a, o+col).AsInt16x16()
				vb := mc16LdU16(b, o+col).AsInt16x16()
				v := mc16MaskBlend16(va, vb, m, m64, bias16, c, z, shV, maxV)
				v.AsUint8x32().StoreArray((*[32]byte)(dst[uint(d+2*col) : uint(d+2*col)+32]))
			} else {
				va := mc16LdU16x8(a, o+col).AsInt16x16()
				vb := mc16LdU16x8(b, o+col).AsInt16x16()
				v := mc16MaskBlend16(va, vb, m, m64, bias16, c, z, shV, maxV)
				v.GetLo().AsUint8x16().StoreArray((*[16]byte)(dst[uint(d+2*col) : uint(d+2*col)+16]))
			}
		}
	}
	archsimd.ClearAVXUpperBits()
}

// mc16DiffWtdMaskFast builds the 10- or 12-bit difference-weighted mask; width
// must be a multiple of 8.
func mc16DiffWtdMaskFast(mask []byte, maskStride int, a, b []uint16, bitDepth, w, h int, invert bool) bool {
	if !mc8AVX2 || (bitDepth != 10 && bitDepth != 12) || w&7 != 0 || w > maxBlockSize || h <= 0 || h > maxBlockSize || len(a) < w*h || len(b) < w*h ||
		maskStride < w || maskStride > mc8MaxStride || (h-1)*maskStride+w > len(mask) {
		return false
	}
	mc16DiffWtdMaskAVX2(mask[:(h-1)*maskStride+w], maskStride, a[:w*h], b[:w*h], w, h, invert)
	return true
}

// mc16LowBytes gathers the low byte of every int16 lane into the low half of
// its 128-bit lane.
var mc16LowBytes = [32]int8{
	0, 2, 4, 6, 8, 10, 12, 14, -1, -1, -1, -1, -1, -1, -1, -1,
	0, 2, 4, 6, 8, 10, 12, 14, -1, -1, -1, -1, -1, -1, -1, -1,
}

func mc16DiffWtdMaskAVX2(mask []byte, maskStride int, a, b []uint16, w, h int, invert bool) {
	mc8Dims(w, h, 0, maskStride)
	one := archsimd.BroadcastUint16x16(1)
	base := archsimd.BroadcastUint16x16(38)
	maxM := archsimd.BroadcastUint16x16(64)
	lowBytes := archsimd.LoadInt8x32Array(&mc16LowBytes)
	for r := 0; r < h; r++ {
		o, mrow := r*w, r*maskStride
		for ucol := uint(0); ucol < uint(w); ucol += 16 {
			col := int(ucol)
			full := w-col >= 16
			var va, vb archsimd.Uint16x16
			if full {
				va, vb = mc16LdU16(a, o+col), mc16LdU16(b, o+col)
			} else {
				va, vb = mc16LdU16x8(a, o+col), mc16LdU16x8(b, o+col)
			}
			diff := va.Max(vb).Sub(va.Min(vb))
			m := diff.ShiftAllRight(5).Add(one).ShiftAllRight(5).Add(base).Min(maxM)
			if invert {
				m = maxM.Sub(m)
			}
			v := m.AsUint8x32().PermuteOrZeroGrouped(lowBytes)
			if full {
				binary.LittleEndian.PutUint64(mask[uint(mrow+col):uint(mrow+col)+8], v.GetLo().AsUint64x2().GetElem(0))
				binary.LittleEndian.PutUint64(mask[uint(mrow+col+8):uint(mrow+col+8)+8], v.GetHi().AsUint64x2().GetElem(0))
			} else {
				binary.LittleEndian.PutUint64(mask[uint(mrow+col):uint(mrow+col)+8], v.GetLo().AsUint64x2().GetElem(0))
			}
		}
	}
	archsimd.ClearAVXUpperBits()
}
