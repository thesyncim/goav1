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
	if dstX < 0 || dstY < 0 || dstX+w > dst.Width || off+(h-1)*dst.Stride+w > len(dst.Pix) {
		return false
	}
	mw, mh := w, h
	if subX {
		mw, mh = 2*w, 2*h
	}
	if maskStride < mw || (mh-1)*maskStride+mw > len(mask) {
		return false
	}
	mc8AVX2Blend(unsafe.Pointer(&dst.Pix[off]), dst.Stride, a[:w*h], b[:w*h], w, h, unsafe.Pointer(&mask[0]), maskStride, subX, mc8OpMask, 0)
	archsimd.ClearAVXUpperBits()
	return true
}

// blendCompoundAvg8AVX2 is the blendCompoundAvg8Impl slot on AVX2 machines:
// the distance-weighted 8-bit average, with the general kernel for anything
// it does not cover.
func blendCompoundAvg8AVX2(dst frame.Plane, src0 []uint16, src1 []uint16, dstX int, dstY int, width int, height int, fwdOffset int, bckOffset int, roundOffset int, roundBits int) {
	off := dstY*dst.Stride + dstX
	if !mc8AVX2 || roundOffset != 6144 || roundBits != 4 || fwdOffset < 0 || fwdOffset > 16 || bckOffset < 0 || bckOffset > 16 ||
		!mc8BlendShapeOK(width, height) || len(src0) < width*height || len(src1) < width*height ||
		dstX < 0 || dstY < 0 || dstX+width > dst.Width || off+(height-1)*dst.Stride+width > len(dst.Pix) {
		blendCompoundAvg8GoSIMD(dst, src0, src1, dstX, dstY, width, height, fwdOffset, bckOffset, roundOffset, roundBits)
		return
	}
	wts := int32(uint32(uint16(fwdOffset)) | uint32(uint16(bckOffset))<<16)
	mc8AVX2Blend(unsafe.Pointer(&dst.Pix[off]), dst.Stride, src0[:width*height], src1[:width*height], width, height, nil, 0, false, mc8OpWAvg, wts)
	archsimd.ClearAVXUpperBits()
}

func init() {
	if mc8AVX2 {
		blendCompoundAvg8Impl = blendCompoundAvg8AVX2
	}
}

// mc8Ld16U16 loads 16 samples at p.
//
//go:nocheckptr
func mc8Ld16U16(p unsafe.Pointer) archsimd.Uint16x16 {
	return archsimd.LoadUint16x16Array((*[16]uint16)(p))
}

//go:nocheckptr
func mc8Ld32(p unsafe.Pointer) uint32 {
	return *(*uint32)(p)
}

// The mask getters return the 16 mask values of one step, laid out like the
// samples. The 4:2:0 forms average 2x2 mask blocks ((sum + 2) >> 2) with
// VPMADDUBSW against ones, as dav1d's mask_420.

//go:nocheckptr
func mc8MaskW4(p unsafe.Pointer, ms int) archsimd.Uint16x16 {
	var z archsimd.Uint32x4
	v := z.SetElem(0, mc8Ld32(p)).SetElem(1, mc8Ld32(unsafe.Add(p, ms))).SetElem(2, mc8Ld32(unsafe.Add(p, 2*ms))).SetElem(3, mc8Ld32(unsafe.Add(p, 3*ms)))
	return v.ReshapeToUint8s().ExtendToUint16()
}

//go:nocheckptr
func mc8MaskW8(p unsafe.Pointer, ms int) archsimd.Uint16x16 {
	var z archsimd.Uint64x2
	return z.SetElem(0, mc8Ld64(p)).SetElem(1, mc8Ld64(unsafe.Add(p, ms))).ReshapeToUint8s().ExtendToUint16()
}

// mc8Ld4x8 gathers the 8 bytes at p, p+2*ms, p+4*ms and p+6*ms.
//
//go:nocheckptr
func mc8Ld4x8(p unsafe.Pointer, ms int) archsimd.Uint8x32 {
	var z2 archsimd.Uint64x2
	var z4 archsimd.Uint64x4
	lo := z2.SetElem(0, mc8Ld64(p)).SetElem(1, mc8Ld64(unsafe.Add(p, 2*ms)))
	hi := z2.SetElem(0, mc8Ld64(unsafe.Add(p, 4*ms))).SetElem(1, mc8Ld64(unsafe.Add(p, 6*ms)))
	return z4.SetLo(lo).SetHi(hi).ReshapeToUint8s()
}

// mc8Mask16 returns the mask of the 16 columns at row r, column c.
//
//go:nocheckptr
func mc8Mask16(mask unsafe.Pointer, ms, r, c int, sub bool, ones archsimd.Int8x32, two archsimd.Int16x16) archsimd.Uint16x16 {
	if !sub {
		return mc8Zx(unsafe.Add(mask, r*ms+c))
	}
	p := unsafe.Add(mask, 2*(r*ms+c))
	return mc8Mask420(mc8Ld32B(p), mc8Ld32B(unsafe.Add(p, ms)), ones, two)
}

func mc8Mask420(x, y archsimd.Uint8x32, ones archsimd.Int8x32, two archsimd.Int16x16) archsimd.Uint16x16 {
	s := x.DotProductPairsSaturated(ones).Add(y.DotProductPairsSaturated(ones))
	return s.Add(two).ShiftAllRight(2).ToBits()
}

//go:nocheckptr
func mc8Ld32B(p unsafe.Pointer) archsimd.Uint8x32 {
	return archsimd.LoadUint8x32Array((*[32]uint8)(p))
}

// mc8BlendStep blends the 16 samples at a and b: with mask m (mc8OpMask) or
// with the broadcast (fwd, bck) word pair wts (mc8OpWAvg).
//
//go:nocheckptr
func mc8BlendStep(a, b unsafe.Pointer, m archsimd.Uint16x16, op int, wts archsimd.Int16x16, sign, m64 archsimd.Uint16x16, kMask, kWAvg archsimd.Int32x8) archsimd.Int16x16 {
	x := mc8Ld16U16(a).Xor(sign).BitsToInt16()
	y := mc8Ld16U16(b).Xor(sign).BitsToInt16()
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
// block mc8BlendShapeOK accepts; a and b hold w*h samples.
//
//go:nocheckptr
func mc8AVX2Blend(dst unsafe.Pointer, ds int, a, b []uint16, w, h int, mask unsafe.Pointer, ms int, sub bool, op int, wts int32) {
	pa := unsafe.Pointer(&a[0])
	pb := unsafe.Pointer(&b[0])
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
					m = mc8MaskW4(unsafe.Add(mask, r*ms), ms)
				case !sub:
					m = mc8MaskW8(unsafe.Add(mask, r*ms), ms)
				case w == 4:
					p := unsafe.Add(mask, 2*r*ms)
					m = mc8Mask420(mc8Ld4x8(p, ms), mc8Ld4x8(unsafe.Add(p, ms), ms), ones, two)
				default:
					p := unsafe.Add(mask, 2*r*ms)
					m = mc8Mask420(mc8Ld2(p, unsafe.Add(p, 2*ms)), mc8Ld2(unsafe.Add(p, ms), unsafe.Add(p, 3*ms)), ones, two)
				}
			}
			o := 2 * r * w
			u := mc8BlendStep(unsafe.Add(pa, o), unsafe.Add(pb, o), m, op, wv, sign, m64, kMask, kWAvg)
			v := mc8Pack(u, u, zero, c255, deint)
			if w == 4 {
				mc8StPutW4(dst, ds, r, h, v)
			} else {
				d := unsafe.Add(dst, r*ds)
				mc8St8(d, v.GetLo())
				mc8St8(unsafe.Add(d, ds), v.GetHi())
			}
		}
	case 16:
		q3120 := archsimd.LoadUint32x8Array(&mc8Q3120)
		for r := 0; r < h; r += 2 {
			r1 := min(r+1, h-1)
			if op == mc8OpMask {
				m = mc8Mask16(mask, ms, r, 0, sub, ones, two)
			}
			u0 := mc8BlendStep(unsafe.Add(pa, 32*r), unsafe.Add(pb, 32*r), m, op, wv, sign, m64, kMask, kWAvg)
			if op == mc8OpMask {
				m = mc8Mask16(mask, ms, r1, 0, sub, ones, two)
			}
			u1 := mc8BlendStep(unsafe.Add(pa, 32*r1), unsafe.Add(pb, 32*r1), m, op, wv, sign, m64, kMask, kWAvg)
			v := mc8Pack(u0, u1, zero, c255, deint).ReshapeToUint32s().Permute(q3120).ReshapeToUint8s()
			d := unsafe.Add(dst, r*ds)
			mc8St16(d, v.GetLo())
			if r+1 < h {
				mc8St16(unsafe.Add(d, ds), v.GetHi())
			}
		}
	default:
		q3120 := archsimd.LoadUint32x8Array(&mc8Q3120)
		for r := 0; r < h; r++ {
			d := unsafe.Add(dst, r*ds)
			for c := 0; c < w; c += 32 {
				o := 2 * (r*w + c)
				if op == mc8OpMask {
					m = mc8Mask16(mask, ms, r, c, sub, ones, two)
				}
				u0 := mc8BlendStep(unsafe.Add(pa, o), unsafe.Add(pb, o), m, op, wv, sign, m64, kMask, kWAvg)
				if op == mc8OpMask {
					m = mc8Mask16(mask, ms, r, c+16, sub, ones, two)
				}
				u1 := mc8BlendStep(unsafe.Add(pa, o+32), unsafe.Add(pb, o+32), m, op, wv, sign, m64, kMask, kWAvg)
				v := mc8Pack(u0, u1, zero, c255, deint).ReshapeToUint32s().Permute(q3120).ReshapeToUint8s()
				v.StoreArray((*[32]uint8)(unsafe.Add(d, c)))
			}
		}
	}
}

// mc8DiffWtdMaskFast builds the 8-bit difference-weighted mask. It returns
// false for shapes it does not cover.
func mc8DiffWtdMaskFast(mask []byte, maskStride int, a, b []uint16, w, h int, invert bool) bool {
	if !mc8AVX2 || !mc8BlendShapeOK(w, h) || len(a) < w*h || len(b) < w*h ||
		maskStride < w || (h-1)*maskStride+w > len(mask) {
		return false
	}
	mc8AVX2DiffWtd(unsafe.Pointer(&mask[0]), maskStride, a[:w*h], b[:w*h], w, h, invert)
	archsimd.ClearAVXUpperBits()
	return true
}

func mc8DiffStep(a, b unsafe.Pointer, invert bool, one, base, m64 archsimd.Uint16x16) archsimd.Uint16x16 {
	x, y := mc8Ld16U16(a), mc8Ld16U16(b)
	m := x.Max(y).Sub(x.Min(y)).ShiftAllRight(3).Add(one).ShiftAllRight(5).Add(base).Min(m64)
	if invert {
		m = m64.Sub(m)
	}
	return m
}

//go:nocheckptr
func mc8AVX2DiffWtd(mask unsafe.Pointer, ms int, a, b []uint16, w, h int, invert bool) {
	pa := unsafe.Pointer(&a[0])
	pb := unsafe.Pointer(&b[0])
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
			o := 2 * r * w
			v := mc8DiffStep(unsafe.Add(pa, o), unsafe.Add(pb, o), invert, one, base, m64).ReshapeToUint8s().PermuteOrZeroGrouped(deint)
			if w == 4 {
				mc8StPutW4(mask, ms, r, h, v)
			} else {
				d := unsafe.Add(mask, r*ms)
				mc8St8(d, v.GetLo())
				mc8St8(unsafe.Add(d, ms), v.GetHi())
			}
		}
	case 16:
		for r := 0; r < h; r += 2 {
			r1 := min(r+1, h-1)
			x := mc8DiffStep(unsafe.Add(pa, 32*r), unsafe.Add(pb, 32*r), invert, one, base, m64)
			y := mc8DiffStep(unsafe.Add(pa, 32*r1), unsafe.Add(pb, 32*r1), invert, one, base, m64)
			v := x.Or(y.ShiftAllLeft(8)).ReshapeToUint8s().PermuteOrZeroGrouped(deint).ReshapeToUint32s().Permute(q3120).ReshapeToUint8s()
			d := unsafe.Add(mask, r*ms)
			mc8St16(d, v.GetLo())
			if r+1 < h {
				mc8St16(unsafe.Add(d, ms), v.GetHi())
			}
		}
	default:
		for r := 0; r < h; r++ {
			d := unsafe.Add(mask, r*ms)
			for c := 0; c < w; c += 32 {
				o := 2 * (r*w + c)
				x := mc8DiffStep(unsafe.Add(pa, o), unsafe.Add(pb, o), invert, one, base, m64)
				y := mc8DiffStep(unsafe.Add(pa, o+32), unsafe.Add(pb, o+32), invert, one, base, m64)
				v := x.Or(y.ShiftAllLeft(8)).ReshapeToUint8s().PermuteOrZeroGrouped(deint).ReshapeToUint32s().Permute(q3120).ReshapeToUint8s()
				v.StoreArray((*[32]uint8)(unsafe.Add(d, c)))
			}
		}
	}
}
