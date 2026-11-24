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

// AVX2 8-bit compound blends over two CONV_BUFs (dav1d mc_avx2.asm mask_8bpc,
// w_avg_8bpc and the w_mask difference mask), the vector forms of
// BlendCompoundMaskD16, BlendCompoundAvg's distance-weighted branch and
// BuildDiffWtdMaskD16 at 8 bits.
//
// Mask blend: libaom's clip(round4(((m*a + (64-m)*b) >> 6) - 6144)) equals
// (X - 392704) >> 10 with X = m*a + (64-m)*b (nested floor division). One
// VPMADDWD per four samples forms X from the word-interleaved (a, b) and
// (m, 64-m) pairs; a and b are biased by -32768 so any uint16 sample is exact
// as int16, which subtracts 64 * 32768 from X.
//
// Distance-weighted average: clip(round4(((a*fwd + b*bck) >> 4) - 6144)) is
// (S - 98176) >> 8 with S = a*fwd + b*bck, formed the same way.
//
// Difference mask: 38 + (round4(|a - b|) >> 4), capped at 64, is
// 38 + ((|a - b| >> 3) + 1) >> 5 by nested floor division.
//
// Each step works on 16 CONV_BUF samples: one row of 16 columns, two rows of
// 8 or four rows of 4.

const (
	mc8OpMask = iota
	mc8OpWAvg
)

// mc8BlendShapeOK reports whether the AVX2 blends cover a w x h block: rows of
// 16 or 32k columns, or whole groups of rows of 8 (pairs) or 4 (quads).
func mc8BlendShapeOK(w, h int) bool {
	switch {
	case w <= 0 || h <= 0 || w > maxBlockSize || h > maxBlockSize:
		return false
	case w == 4:
		return h&3 == 0
	case w == 8:
		return h&1 == 0
	default:
		return w == 16 || w&31 == 0
	}
}

// mc8BlendMaskFast blends with a full-resolution or 4:2:0 subsampled mask.
// It returns false for shapes it does not cover.
func mc8BlendMaskFast(dst frame.Plane, a, b []uint16, dstX, dstY, w, h int, mask []byte, maskStride int, subX, subY bool) bool {
	if !mc8AVX2 || !mc8BlendShapeOK(w, h) || len(a) < w*h || len(b) < w*h || subX != subY {
		return false
	}
	off := dstY*dst.Stride + dstX
	end := off + (h-1)*dst.Stride + w
	if dstX < 0 || dstY < 0 || dstX+w > dst.Width || end > len(dst.Pix) || dst.Stride > mc8MaxStride {
		return false
	}
	mw, mh := w, h
	if subX {
		mw, mh = 2*w, 2*h
	}
	mEnd := (mh-1)*maskStride + mw
	if maskStride < mw || maskStride > mc8MaxStride || mEnd > len(mask) {
		return false
	}
	mc8AVX2Blend(dst.Pix[off:end], dst.Stride, a[:w*h], b[:w*h], w, h, mask[:mEnd], maskStride, subX, mc8OpMask, 0)
	archsimd.ClearAVXUpperBits()
	return true
}

// blendCompoundAvg8AVX2 is the blendCompoundAvg8Impl slot on AVX2 machines:
// the distance-weighted 8-bit average, with the general kernel for anything
// it does not cover.
func blendCompoundAvg8AVX2(dst frame.Plane, src0 []uint16, src1 []uint16, dstX int, dstY int, width int, height int, fwdOffset int, bckOffset int, roundOffset int, roundBits int) {
	off := dstY*dst.Stride + dstX
	end := off + (height-1)*dst.Stride + width
	if !mc8AVX2 || roundOffset != 6144 || roundBits != 4 || fwdOffset < 0 || fwdOffset > 16 || bckOffset < 0 || bckOffset > 16 ||
		!mc8BlendShapeOK(width, height) || len(src0) < width*height || len(src1) < width*height ||
		dstX < 0 || dstY < 0 || dstX+width > dst.Width || end > len(dst.Pix) || dst.Stride > mc8MaxStride {
		blendCompoundAvg8GoSIMD(dst, src0, src1, dstX, dstY, width, height, fwdOffset, bckOffset, roundOffset, roundBits)
		return
	}
	wts := int32(uint32(uint16(fwdOffset)) | uint32(uint16(bckOffset))<<16)
	mc8AVX2Blend(dst.Pix[off:end], dst.Stride, src0[:width*height], src1[:width*height], width, height, nil, 0, false, mc8OpWAvg, wts)
	archsimd.ClearAVXUpperBits()
}

func init() {
	if mc8AVX2 {
		blendCompoundAvg8Impl = blendCompoundAvg8AVX2
	}
}

// mc8Ld16U16 loads the 16 samples at s[o:].
func mc8Ld16U16(s []uint16, o int) archsimd.Uint16x16 {
	return archsimd.LoadUint16x16Array((*[16]uint16)(s[uint(o) : uint(o)+16]))
}

// The mask getters return the 16 mask values of one step, laid out like the
// samples. The 4:2:0 forms average 2x2 mask blocks ((sum + 2) >> 2) with
// VPMADDUBSW against ones, as dav1d's mask_420.

func mc8MaskW4(mask []byte, o, ms int) archsimd.Uint16x16 {
	var z archsimd.Uint32x4
	v := z.SetElem(0, mc8Ld32(mask, o)).SetElem(1, mc8Ld32(mask, o+ms)).SetElem(2, mc8Ld32(mask, o+2*ms)).SetElem(3, mc8Ld32(mask, o+3*ms))
	return v.ReshapeToUint8s().ExtendToUint16()
}

func mc8MaskW8(mask []byte, o, ms int) archsimd.Uint16x16 {
	var z archsimd.Uint64x2
	return z.SetElem(0, mc8Ld64(mask, o)).SetElem(1, mc8Ld64(mask, o+ms)).ReshapeToUint8s().ExtendToUint16()
}

// mc8Ld4x8 gathers the 8 bytes at o, o+2*ms, o+4*ms and o+6*ms.
func mc8Ld4x8(mask []byte, o, ms int) archsimd.Uint8x32 {
	var z2 archsimd.Uint64x2
	var z4 archsimd.Uint64x4
	lo := z2.SetElem(0, mc8Ld64(mask, o)).SetElem(1, mc8Ld64(mask, o+2*ms))
	hi := z2.SetElem(0, mc8Ld64(mask, o+4*ms)).SetElem(1, mc8Ld64(mask, o+6*ms))
	return z4.SetLo(lo).SetHi(hi).ReshapeToUint8s()
}

// mc8Mask16 returns the mask of the 16 columns at row r, column c.
func mc8Mask16(mask []byte, ms, r, c int, sub bool, ones archsimd.Int8x32, two archsimd.Int16x16) archsimd.Uint16x16 {
	if !sub {
		return mc8Zx(mask, r*ms+c)
	}
	p := 2 * (r*ms + c)
	return mc8Mask420(mc8Ld32B(mask, p), mc8Ld32B(mask, p+ms), ones, two)
}

func mc8Mask420(x, y archsimd.Uint8x32, ones archsimd.Int8x32, two archsimd.Int16x16) archsimd.Uint16x16 {
	s := x.DotProductPairsSaturated(ones).Add(y.DotProductPairsSaturated(ones))
	return s.Add(two).ShiftAllRight(2).ToBits()
}

func mc8Ld32B(s []byte, o int) archsimd.Uint8x32 {
	return archsimd.LoadUint8x32Array((*[32]uint8)(s[uint(o) : uint(o)+32]))
}

// mc8BlendStep blends the 16 samples at a[o:] and b[o:]: with mask m
// (mc8OpMask) or with the broadcast (fwd, bck) word pair wts (mc8OpWAvg).
func mc8BlendStep(a, b []uint16, o int, m archsimd.Uint16x16, op int, wts archsimd.Int16x16, sign, m64 archsimd.Uint16x16, kMask, kWAvg archsimd.Int32x8) archsimd.Int16x16 {
	x := mc8Ld16U16(a, o).Xor(sign).BitsToInt16()
	y := mc8Ld16U16(b, o).Xor(sign).BitsToInt16()
	lo, hi := x.InterleaveLoGrouped(y), x.InterleaveHiGrouped(y)
	if op == mc8OpWAvg {
		return lo.DotProductPairs(wts).Add(kWAvg).ShiftAllRight(8).SaturateToInt16ConcatGrouped(hi.DotProductPairs(wts).Add(kWAvg).ShiftAllRight(8))
	}
	mi := m64.Sub(m)
	ml := m.InterleaveLoGrouped(mi).BitsToInt16()
	mh := m.InterleaveHiGrouped(mi).BitsToInt16()
	return lo.DotProductPairs(ml).Add(kMask).ShiftAllRight(10).SaturateToInt16ConcatGrouped(hi.DotProductPairs(mh).Add(kMask).ShiftAllRight(10))
}

// mc8AVX2Blend runs the mask blend or the distance-weighted average over a
// block mc8BlendShapeOK accepts; a and b hold w*h samples and dst and mask
// the block's rows.
func mc8AVX2Blend(dst []byte, ds int, a, b []uint16, w, h int, mask []byte, ms int, sub bool, op int, wts int32) {
	mc8Dims(w, h, ms, ds)
	ones := archsimd.BroadcastInt8x32(1)
	two := archsimd.BroadcastInt16x16(2)
	sign := archsimd.BroadcastUint16x16(0x8000)
	m64 := archsimd.BroadcastUint16x16(64)
	kMask := archsimd.BroadcastInt32x8(64*32768 - 392704)
	kWAvg := archsimd.BroadcastInt32x8(32768*(int32(uint16(wts))+wts>>16) - 98176)
	wv := mc8TapsD(wts)
	zero := archsimd.Int16x16{}
	c255 := archsimd.BroadcastInt16x16(255)
	deint := archsimd.LoadInt8x32Array(&mc8Deint)
	var m archsimd.Uint16x16
	switch w {
	case 4, 8:
		rows := 2
		if w == 4 {
			rows = 4
		}
		for r := 0; r < h; r += rows {
			if op == mc8OpMask {
				switch {
				case !sub && w == 4:
					m = mc8MaskW4(mask, r*ms, ms)
				case !sub:
					m = mc8MaskW8(mask, r*ms, ms)
				case w == 4:
					p := 2 * r * ms
					m = mc8Mask420(mc8Ld4x8(mask, p, ms), mc8Ld4x8(mask, p+ms, ms), ones, two)
				default:
					p := 2 * r * ms
					m = mc8Mask420(mc8Ld2At(mask, p, p+2*ms), mc8Ld2At(mask, p+ms, p+3*ms), ones, two)
				}
			}
			u := mc8BlendStep(a, b, r*w, m, op, wv, sign, m64, kMask, kWAvg)
			v := mc8Pack(u, u, zero, c255, deint)
			if w == 4 {
				mc8StPutW4(dst, ds, r, h, v)
			} else {
				y0, y1 := mc8Rows2x8(dst, r*ds, ds)
				u := v.ReshapeToUint64s()
				binary.LittleEndian.PutUint64(y0[:], u.GetLo().GetElem(0))
				binary.LittleEndian.PutUint64(y1[:], u.GetHi().GetElem(0))
			}
		}
	case 16:
		q3120 := archsimd.LoadUint32x8Array(&mc8Q3120)
		for ur := uint(0); ur < uint(h); ur += 2 {
			r := int(ur)
			r1 := min(r+1, h-1)
			if op == mc8OpMask {
				m = mc8Mask16(mask, ms, r, 0, sub, ones, two)
			}
			u0 := mc8BlendStep(a, b, 16*r, m, op, wv, sign, m64, kMask, kWAvg)
			if op == mc8OpMask {
				m = mc8Mask16(mask, ms, r1, 0, sub, ones, two)
			}
			u1 := mc8BlendStep(a, b, 16*r1, m, op, wv, sign, m64, kMask, kWAvg)
			v := mc8Pack(u0, u1, zero, c255, deint).ReshapeToUint32s().Permute(q3120).ReshapeToUint8s()
			d := r * ds
			mc8St16(dst, d, v.GetLo())
			if r+1 < h {
				mc8St16(dst, d+ds, v.GetHi())
			}
		}
	default:
		q3120 := archsimd.LoadUint32x8Array(&mc8Q3120)
		for r := 0; r < h; r++ {
			d := r * ds
			for uc := uint(0); uc < uint(w); uc += 32 {
				c := int(uc)
				o := r*w + c
				if op == mc8OpMask {
					m = mc8Mask16(mask, ms, r, c, sub, ones, two)
				}
				u0 := mc8BlendStep(a, b, o, m, op, wv, sign, m64, kMask, kWAvg)
				if op == mc8OpMask {
					m = mc8Mask16(mask, ms, r, c+16, sub, ones, two)
				}
				u1 := mc8BlendStep(a, b, o+16, m, op, wv, sign, m64, kMask, kWAvg)
				v := mc8Pack(u0, u1, zero, c255, deint).ReshapeToUint32s().Permute(q3120).ReshapeToUint8s()
				v.StoreArray((*[32]uint8)(dst[uint(d+c) : uint(d+c)+32]))
			}
		}
	}
}

// mc8DiffWtdMaskFast builds the 8-bit difference-weighted mask. It returns
// false for shapes it does not cover.
func mc8DiffWtdMaskFast(mask []byte, maskStride int, a, b []uint16, w, h int, invert bool) bool {
	if !mc8AVX2 || !mc8BlendShapeOK(w, h) || len(a) < w*h || len(b) < w*h ||
		maskStride < w || maskStride > mc8MaxStride || (h-1)*maskStride+w > len(mask) {
		return false
	}
	mc8AVX2DiffWtd(mask[:(h-1)*maskStride+w], maskStride, a[:w*h], b[:w*h], w, h, invert)
	archsimd.ClearAVXUpperBits()
	return true
}

func mc8DiffStep(a, b []uint16, o int, invert bool, one, base, m64 archsimd.Uint16x16) archsimd.Uint16x16 {
	x, y := mc8Ld16U16(a, o), mc8Ld16U16(b, o)
	m := x.Max(y).Sub(x.Min(y)).ShiftAllRight(3).Add(one).ShiftAllRight(5).Add(base).Min(m64)
	if invert {
		m = m64.Sub(m)
	}
	return m
}

func mc8AVX2DiffWtd(mask []byte, ms int, a, b []uint16, w, h int, invert bool) {
	mc8Dims(w, h, 0, ms)
	one := archsimd.BroadcastUint16x16(1)
	base := archsimd.BroadcastUint16x16(38)
	m64 := archsimd.BroadcastUint16x16(64)
	deint := archsimd.LoadInt8x32Array(&mc8Deint)
	q3120 := archsimd.LoadUint32x8Array(&mc8Q3120)
	switch w {
	case 4, 8:
		rows := 2
		if w == 4 {
			rows = 4
		}
		for r := 0; r < h; r += rows {
			v := mc8DiffStep(a, b, r*w, invert, one, base, m64).ReshapeToUint8s().PermuteOrZeroGrouped(deint)
			if w == 4 {
				mc8StPutW4(mask, ms, r, h, v)
			} else {
				y0, y1 := mc8Rows2x8(mask, r*ms, ms)
				u := v.ReshapeToUint64s()
				binary.LittleEndian.PutUint64(y0[:], u.GetLo().GetElem(0))
				binary.LittleEndian.PutUint64(y1[:], u.GetHi().GetElem(0))
			}
		}
	case 16:
		for ur := uint(0); ur < uint(h); ur += 2 {
			r := int(ur)
			r1 := min(r+1, h-1)
			x := mc8DiffStep(a, b, 16*r, invert, one, base, m64)
			y := mc8DiffStep(a, b, 16*r1, invert, one, base, m64)
			v := x.Or(y.ShiftAllLeft(8)).ReshapeToUint8s().PermuteOrZeroGrouped(deint).ReshapeToUint32s().Permute(q3120).ReshapeToUint8s()
			d := r * ms
			mc8St16(mask, d, v.GetLo())
			if r+1 < h {
				mc8St16(mask, d+ms, v.GetHi())
			}
		}
	default:
		for r := 0; r < h; r++ {
			d := r * ms
			for uc := uint(0); uc < uint(w); uc += 32 {
				c := int(uc)
				o := r*w + c
				x := mc8DiffStep(a, b, o, invert, one, base, m64)
				y := mc8DiffStep(a, b, o+16, invert, one, base, m64)
				v := x.Or(y.ShiftAllLeft(8)).ReshapeToUint8s().PermuteOrZeroGrouped(deint).ReshapeToUint32s().Permute(q3120).ReshapeToUint8s()
				v.StoreArray((*[32]uint8)(mask[uint(d+c) : uint(d+c)+32]))
			}
		}
	}
}
