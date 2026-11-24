// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package dsp

// Go-native SIMD (simd/archsimd) AddResidualPlaneBlock and AddRawTransform-
// PlaneBlock kernels for arm64.
//
// Shapes: widths that are a multiple of eight run eight samples per vector
// (sixteen for 8-bit widths that are a multiple of sixteen), width four runs
// two rows per vector, and every 8-bit kernel needs max == 0xff (the only 8-bit
// maximum AV1 produces). Anything else routes to addResidualPlaneBlockPureGo /
// addRawTransformPlaneBlockPureGo.
//
// Bit-exactness with the scalar references, for the full input domain:
//   - 8-bit: the reference output is clamp(d + r, 0, 255) for a byte d and an
//     int16 r. d + r is at least -32768 and saturates only above 32767, which
//     still clamps to 255, so the signed saturating add (SQADD) followed by the
//     signed-to-unsigned saturating narrow (SQXTUN) is exact.
//   - 16-bit: the reference output is clamp(d + r, 0, max) for any u16 d. With
//     the bias b = d ^ 0x8000 = d - 32768 as int16, the saturating add b + r
//     saturates exactly where d + r leaves [0, 65535], so
//     (sat16(b + r) ^ 0x8000) = clamp(d + r, 0, 65535); UMIN with max finishes
//     the clamp.
//   - raw: rawTransformResidual(v) = sat16(floor((v + 8) / 16)). The saturating
//     add v + 8 differs from the exact sum only for v > MaxInt32 - 8, where
//     both sums shifted right by four saturate to 32767 in the narrow (SQXTN).
//     arm64 archsimd has no immediate-shift form, so the shift is SSHL by a
//     hoisted vector of -4.
//
// The 8- and 4-sample rows of 8-bit blocks, and the 4-sample rows of 16-bit
// blocks, are narrower than a vector register. They are read and written with
// exact-width scalar accesses, so no byte outside the block is touched, and are
// paired two rows to a vector.

import "simd/archsimd"

// The kernels take each row as a bounds-checked window: the entry points bound
// the strides (planeStrideMax) and height (planeRowsMax), so a row window
// s[y*stride : y*stride+n] is one compare, and the column loads inside a row
// index a slice whose length is the row width (no further checks).
const (
	planeStrideMax = 1 << 30
	planeRowsMax   = 1 << 16
)

// addResidualPlaneBlockSIMD is the Go SIMD residual add for 8-bit and 16-bit
// samples.
func addResidualPlaneBlockSIMD(block planeBlock, bytesPerSample int, max uint16, width int, residual []int16, residualStride int) {
	h := block.height
	if h <= 0 || h > planeRowsMax || uint(block.stride) > planeStrideMax || uint(residualStride) > planeStrideMax ||
		(width%8 != 0 && width != 4) || (bytesPerSample == 1 && max != 0xff) || (bytesPerSample != 1 && bytesPerSample != 2) {
		addResidualPlaneBlockPureGo(block, bytesPerSample, max, width, residual, residualStride)
		return
	}
	d, ds, r, rs := block.pix, block.stride, residual, residualStride
	if bytesPerSample == 1 {
		switch width {
		case 4:
			planeResidual8x4(d, ds, r, rs, h)
		case 8:
			planeResidual8x8(d, ds, r, rs, h)
		case 16:
			planeResidual8W16(d, ds, r, rs, h)
		case 32:
			planeResidual8W32(d, ds, r, rs, h)
		case 64:
			planeResidual8W64(d, ds, r, rs, h)
		default:
			planeResidual8Wide(d, ds, r, rs, width, h)
		}
		return
	}
	switch width {
	case 4:
		planeResidual16x4(d, ds, r, rs, max, h)
		return
	case 8:
		planeResidual16x8(d, ds, r, rs, max, h)
		return
	case 16:
		planeResidual16W16(d, ds, r, rs, max, h)
		return
	case 32:
		planeResidual16W32(d, ds, r, rs, max, h)
		return
	case 64:
		planeResidual16W64(d, ds, r, rs, max, h)
		return
	}
	planeResidual16Wide(d, ds, r, rs, max, width, h)
}

// addRawTransformPlaneBlockSIMD is the Go SIMD fused raw-transform add for 8-bit
// and 16-bit samples.
func addRawTransformPlaneBlockSIMD(block planeBlock, bytesPerSample int, max uint16, width int, raw []int32, rawStride int) {
	h := block.height
	if h <= 0 || h > planeRowsMax || uint(block.stride) > planeStrideMax || uint(rawStride) > planeStrideMax ||
		(width%8 != 0 && width != 4) || (bytesPerSample == 1 && max != 0xff) || (bytesPerSample != 1 && bytesPerSample != 2) {
		addRawTransformPlaneBlockPureGo(block, bytesPerSample, max, width, raw, rawStride)
		return
	}
	d, ds, s, ss := block.pix, block.stride, raw, rawStride
	if bytesPerSample == 1 {
		switch width {
		case 4:
			planeRaw8x4(d, ds, s, ss, h)
		case 8:
			planeRaw8x8(d, ds, s, ss, h)
		case 16:
			planeRaw8W16(d, ds, s, ss, h)
		case 32:
			planeRaw8W32(d, ds, s, ss, h)
		case 64:
			planeRaw8W64(d, ds, s, ss, h)
		default:
			planeRaw8Wide(d, ds, s, ss, width, h)
		}
		return
	}
	switch width {
	case 4:
		planeRaw16x4(d, ds, s, ss, max, h)
		return
	case 8:
		planeRaw16x8(d, ds, s, ss, max, h)
		return
	case 16:
		planeRaw16W16(d, ds, s, ss, max, h)
		return
	case 32:
		planeRaw16W32(d, ds, s, ss, max, h)
		return
	case 64:
		planeRaw16W64(d, ds, s, ss, max, h)
		return
	}
	planeRaw16Wide(d, ds, s, ss, max, width, h)
}

// planeI16x4 packs the four int16 residuals of r as the low 64 bits.
func planeI16x4(r *[4]int16) uint64 {
	return uint64(uint16(r[0])) | uint64(uint16(r[1]))<<16 | uint64(uint16(r[2]))<<32 | uint64(uint16(r[3]))<<48
}

// planeU16x4 packs four uint16 samples as the low 64 bits.
func planeU16x4(r *[4]uint16) uint64 {
	return uint64(r[0]) | uint64(r[1])<<16 | uint64(r[2])<<32 | uint64(r[3])<<48
}

// planeRawRound returns sat16(floor((v + 8) / 16)) of the eight int32 samples
// of p, in lane order.
func planeRawRound(p *[8]int32, c8, sh4 archsimd.Int32x4) archsimd.Int16x8 {
	// planeRawRound4Pair written out: a nested inlined call leaves an
	// inline-mark NOP at every use.
	lo := archsimd.LoadInt32x4Array((*[4]int32)(p[0:4])).AddSaturated(c8).Shift(sh4).SaturateToInt16().ToBits().ReshapeToUint64s().BitsToFloat64()
	hi := archsimd.LoadInt32x4Array((*[4]int32)(p[4:8])).AddSaturated(c8).Shift(sh4).SaturateToInt16().ToBits().ReshapeToUint64s().BitsToFloat64()
	return lo.SetElem(1, hi.GetElem(0)).ToBits().ReshapeToUint16s().BitsToInt16()
}

// planeRawRound4Pair returns sat16(floor((v + 8) / 16)) of four int32 samples of
// a in lanes 0-3 and four of b in lanes 4-7.
//
// The high half narrows as SQXTN2 (SetElem of the low 64 bits) rather than a
// second SQXTN joined by ZIP1.
func planeRawRound4Pair(a, b *[4]int32, c8, sh4 archsimd.Int32x4) archsimd.Int16x8 {
	lo := archsimd.LoadInt32x4Array(a).AddSaturated(c8).Shift(sh4).SaturateToInt16().ToBits().ReshapeToUint64s().BitsToFloat64()
	hi := archsimd.LoadInt32x4Array(b).AddSaturated(c8).Shift(sh4).SaturateToInt16().ToBits().ReshapeToUint64s().BitsToFloat64()
	return lo.SetElem(1, hi.GetElem(0)).ToBits().ReshapeToUint16s().BitsToInt16()
}

// planeAdd8 returns clamp(d + r, 0, 255) as eight bytes in the low 64 bits of
// the result. d holds eight bytes zero-extended to u16 lanes (interleaved with
// a zero vector).
func planeAdd8(d archsimd.Uint8x16, r archsimd.Int16x8) archsimd.Uint64x2 {
	return d.ReshapeToUint16s().BitsToInt16().AddSaturated(r).SaturateToUint8().ReshapeToUint64s()
}

// planeAdd16 returns clamp(d + r, 0, max) lane by lane for u16 samples d,
// int16 residuals r and bias = 0x8000 in every lane.
func planeAdd16(d archsimd.Uint16x8, r archsimd.Int16x8, bias archsimd.Uint16x8, maxv archsimd.Uint16x8) archsimd.Uint16x8 {
	return d.Xor(bias).BitsToInt16().AddSaturated(r).ToBits().Xor(bias).Min(maxv)
}

// The kernels below take the destination d (row stride ds bytes) and the
// residual or raw source (stride rs/ss elements); h is at least one. Row
// pairs are counted by pair (a step-2 row counter costs a second check per
// window).

func planeResidual8x4(d []byte, ds int, r []int16, rs int, h int) {
	if uint(ds) > planeStrideMax || uint(rs) > planeStrideMax || uint(h) > planeRowsMax {
		panic("dsp: plane kernel arguments out of range")
	}
	zero := archsimd.BroadcastUint8x16(0)
	for i := 0; i < h>>1; i++ {
		y := 2 * i
		pa, pb := (*[4]byte)(d[y*ds:y*ds+4]), (*[4]byte)(d[(y+1)*ds:(y+1)*ds+4])
		ra, rb := (*[4]int16)(r[y*rs:y*rs+4]), (*[4]int16)(r[(y+1)*rs:(y+1)*rs+4])
		dv := archsimd.BroadcastUint64x2(uint64(planeLE32(pa)) | uint64(planeLE32(pb))<<32).ReshapeToUint8s()
		rv := archsimd.LoadUint64x2Array(&[2]uint64{planeI16x4(ra), planeI16x4(rb)}).ReshapeToUint16s().BitsToInt16()
		w := planeAdd8(dv.InterleaveLo(zero), rv).GetElem(0)
		planePut32(pa, uint32(w))
		planePut32(pb, uint32(w>>32))
	}
	if y := h &^ 1; y < h {
		pa := (*[4]byte)(d[y*ds : y*ds+4])
		ra := (*[4]int16)(r[y*rs : y*rs+4])
		dv := archsimd.BroadcastUint64x2(uint64(planeLE32(pa))).ReshapeToUint8s()
		rv := archsimd.BroadcastUint64x2(planeI16x4(ra)).ReshapeToUint16s().BitsToInt16()
		planePut32(pa, uint32(planeAdd8(dv.InterleaveLo(zero), rv).GetElem(0)))
	}
}

func planeResidual8x8(d []byte, ds int, r []int16, rs int, h int) {
	if uint(ds) > planeStrideMax || uint(rs) > planeStrideMax || uint(h) > planeRowsMax {
		panic("dsp: plane kernel arguments out of range")
	}
	zero := archsimd.BroadcastUint8x16(0)
	for i := 0; i < h>>1; i++ {
		y := 2 * i
		pa, pb := (*[8]byte)(d[y*ds:y*ds+8]), (*[8]byte)(d[(y+1)*ds:(y+1)*ds+8])
		ra, rb := (*[8]int16)(r[y*rs:y*rs+8]), (*[8]int16)(r[(y+1)*rs:(y+1)*rs+8])
		dv := archsimd.LoadUint64x2Array(&[2]uint64{planeLE64(pa), planeLE64(pb)}).ReshapeToUint8s()
		oa := planeAdd8(dv.InterleaveLo(zero), archsimd.LoadInt16x8Array(ra))
		ob := planeAdd8(dv.InterleaveHi(zero), archsimd.LoadInt16x8Array(rb))
		planePut64(pa, oa.GetElem(0))
		planePut64(pb, ob.GetElem(0))
	}
	if y := h &^ 1; y < h {
		pa := (*[8]byte)(d[y*ds : y*ds+8])
		ra := (*[8]int16)(r[y*rs : y*rs+8])
		dv := archsimd.BroadcastUint64x2(planeLE64(pa)).ReshapeToUint8s()
		planePut64(pa, planeAdd8(dv.InterleaveLo(zero), archsimd.LoadInt16x8Array(ra)).GetElem(0))
	}
}

func planeResidual8Wide(d []byte, ds int, r []int16, rs int, width int, h int) {
	if uint(ds) > planeStrideMax || uint(rs) > planeStrideMax || uint(h) > planeRowsMax {
		panic("dsp: plane kernel arguments out of range")
	}
	zero := archsimd.BroadcastUint8x16(0)
	for y := 0; y < h; y++ {
		pa := d[y*ds : y*ds+width]
		ra := r[y*rs : y*rs+width]
		x := 0
		for ; x+16 <= len(pa); x += 16 {
			p := (*[16]uint8)(pa[x : x+16])
			rr := (*[16]int16)(ra[x : x+16])
			dv := archsimd.LoadUint8x16Array(p)
			lo := planeAdd8(dv.InterleaveLo(zero), archsimd.LoadInt16x8Array((*[8]int16)(rr[0:8])))
			hi := planeAdd8(dv.InterleaveHi(zero), archsimd.LoadInt16x8Array((*[8]int16)(rr[8:16])))
			lo.InterleaveLo(hi).ReshapeToUint8s().StoreArray(p)
		}
		if x+8 <= len(pa) {
			p := (*[8]byte)(pa[x : x+8])
			dv := archsimd.BroadcastUint64x2(planeLE64(p)).ReshapeToUint8s()
			planePut64(p, planeAdd8(dv.InterleaveLo(zero), archsimd.LoadInt16x8Array((*[8]int16)(ra[x:x+8]))).GetElem(0))
		}
	}
}

func planeResidual16x4(d []byte, ds int, r []int16, rs int, max uint16, h int) {
	if uint(ds) > planeStrideMax || uint(rs) > planeStrideMax || uint(h) > planeRowsMax {
		panic("dsp: plane kernel arguments out of range")
	}
	bias := archsimd.BroadcastUint16x8(0x8000)
	maxv := archsimd.BroadcastUint16x8(max)
	for i := 0; i < h>>1; i++ {
		y := 2 * i
		pa, pb := (*[8]byte)(d[y*ds:y*ds+8]), (*[8]byte)(d[(y+1)*ds:(y+1)*ds+8])
		ra, rb := (*[4]int16)(r[y*rs:y*rs+4]), (*[4]int16)(r[(y+1)*rs:(y+1)*rs+4])
		dv := archsimd.LoadUint64x2Array(&[2]uint64{planeLE64(pa), planeLE64(pb)}).ReshapeToUint16s()
		rv := archsimd.LoadUint64x2Array(&[2]uint64{planeI16x4(ra), planeI16x4(rb)}).ReshapeToUint16s().BitsToInt16()
		o := planeAdd16(dv, rv, bias, maxv).ReshapeToUint64s()
		planePut64(pa, o.GetElem(0))
		planePut64(pb, o.GetElem(1))
	}
	if y := h &^ 1; y < h {
		pa := (*[8]byte)(d[y*ds : y*ds+8])
		ra := (*[4]int16)(r[y*rs : y*rs+4])
		dv := archsimd.BroadcastUint64x2(planeLE64(pa)).ReshapeToUint16s()
		rv := archsimd.BroadcastUint64x2(planeI16x4(ra)).ReshapeToUint16s().BitsToInt16()
		planePut64(pa, planeAdd16(dv, rv, bias, maxv).ReshapeToUint64s().GetElem(0))
	}
}

func planeResidual16Wide(d []byte, ds int, r []int16, rs int, max uint16, width int, h int) {
	if uint(ds) > planeStrideMax || uint(rs) > planeStrideMax || uint(h) > planeRowsMax {
		panic("dsp: plane kernel arguments out of range")
	}
	bias := archsimd.BroadcastUint16x8(0x8000)
	maxv := archsimd.BroadcastUint16x8(max)
	for y := 0; y < h; y++ {
		pa := d[y*ds : y*ds+2*width]
		ra := r[y*rs : y*rs+width]
		for x := 0; x+8 <= len(ra); x += 8 {
			p := (*[16]uint8)(pa[2*x : 2*x+16])
			rv := archsimd.LoadInt16x8Array((*[8]int16)(ra[x : x+8]))
			planeAdd16(archsimd.LoadUint8x16Array(p).ReshapeToUint16s(), rv, bias, maxv).ReshapeToUint8s().StoreArray(p)
		}
	}
}

func planeResidual16x8(d []byte, ds int, r []int16, rs int, max uint16, h int) {
	if uint(ds) > planeStrideMax || uint(rs) > planeStrideMax || uint(h) > planeRowsMax {
		panic("dsp: plane kernel arguments out of range")
	}
	bias := archsimd.BroadcastUint16x8(0x8000)
	maxv := archsimd.BroadcastUint16x8(max)
	for i := 0; i < h>>1; i++ {
		y := 2 * i
		pa, pb := (*[16]uint8)(d[y*ds:y*ds+16]), (*[16]uint8)(d[(y+1)*ds:(y+1)*ds+16])
		ra, rb := (*[8]int16)(r[y*rs:y*rs+8]), (*[8]int16)(r[(y+1)*rs:(y+1)*rs+8])
		oa := planeAdd16(archsimd.LoadUint8x16Array(pa).ReshapeToUint16s(), archsimd.LoadInt16x8Array(ra), bias, maxv)
		ob := planeAdd16(archsimd.LoadUint8x16Array(pb).ReshapeToUint16s(), archsimd.LoadInt16x8Array(rb), bias, maxv)
		oa.ReshapeToUint8s().StoreArray(pa)
		ob.ReshapeToUint8s().StoreArray(pb)
	}
	if y := h &^ 1; y < h {
		pa := (*[16]uint8)(d[y*ds : y*ds+16])
		rv := archsimd.LoadInt16x8Array((*[8]int16)(r[y*rs : y*rs+8]))
		planeAdd16(archsimd.LoadUint8x16Array(pa).ReshapeToUint16s(), rv, bias, maxv).ReshapeToUint8s().StoreArray(pa)
	}
}

func planeRaw8x4(d []byte, ds int, s []int32, ss int, h int) {
	if uint(ds) > planeStrideMax || uint(ss) > planeStrideMax || uint(h) > planeRowsMax {
		panic("dsp: plane kernel arguments out of range")
	}
	zero := archsimd.BroadcastUint8x16(0)
	c8 := archsimd.BroadcastInt32x4(8)
	sh4 := archsimd.BroadcastInt32x4(-4)
	for i := 0; i < h>>1; i++ {
		y := 2 * i
		pa, pb := (*[4]byte)(d[y*ds:y*ds+4]), (*[4]byte)(d[(y+1)*ds:(y+1)*ds+4])
		sa, sb := (*[4]int32)(s[y*ss:y*ss+4]), (*[4]int32)(s[(y+1)*ss:(y+1)*ss+4])
		dv := archsimd.BroadcastUint64x2(uint64(planeLE32(pa)) | uint64(planeLE32(pb))<<32).ReshapeToUint8s()
		w := planeAdd8(dv.InterleaveLo(zero), planeRawRound4Pair(sa, sb, c8, sh4)).GetElem(0)
		planePut32(pa, uint32(w))
		planePut32(pb, uint32(w>>32))
	}
	if y := h &^ 1; y < h {
		pa := (*[4]byte)(d[y*ds : y*ds+4])
		sa := (*[4]int32)(s[y*ss : y*ss+4])
		rv := planeRawRound4Pair(sa, sa, c8, sh4)
		dv := archsimd.BroadcastUint64x2(uint64(planeLE32(pa))).ReshapeToUint8s()
		planePut32(pa, uint32(planeAdd8(dv.InterleaveLo(zero), rv).GetElem(0)))
	}
}

func planeRaw8x8(d []byte, ds int, s []int32, ss int, h int) {
	if uint(ds) > planeStrideMax || uint(ss) > planeStrideMax || uint(h) > planeRowsMax {
		panic("dsp: plane kernel arguments out of range")
	}
	zero := archsimd.BroadcastUint8x16(0)
	c8 := archsimd.BroadcastInt32x4(8)
	sh4 := archsimd.BroadcastInt32x4(-4)
	for i := 0; i < h>>1; i++ {
		y := 2 * i
		pa, pb := (*[8]byte)(d[y*ds:y*ds+8]), (*[8]byte)(d[(y+1)*ds:(y+1)*ds+8])
		sa, sb := (*[8]int32)(s[y*ss:y*ss+8]), (*[8]int32)(s[(y+1)*ss:(y+1)*ss+8])
		// Each round feeds its add on the same line: alone on a line, the
		// inlined call would leave an inline-mark NOP.
		dv := archsimd.LoadUint64x2Array(&[2]uint64{planeLE64(pa), planeLE64(pb)}).ReshapeToUint8s()
		planePut64(pa, planeAdd8(dv.InterleaveLo(zero), planeRawRound(sa, c8, sh4)).GetElem(0))
		planePut64(pb, planeAdd8(dv.InterleaveHi(zero), planeRawRound(sb, c8, sh4)).GetElem(0))
	}
	if y := h &^ 1; y < h {
		pa := (*[8]byte)(d[y*ds : y*ds+8])
		rv := planeRawRound((*[8]int32)(s[y*ss:y*ss+8]), c8, sh4)
		dv := archsimd.BroadcastUint64x2(planeLE64(pa)).ReshapeToUint8s()
		planePut64(pa, planeAdd8(dv.InterleaveLo(zero), rv).GetElem(0))
	}
}

func planeRaw8Wide(d []byte, ds int, s []int32, ss int, width int, h int) {
	if uint(ds) > planeStrideMax || uint(ss) > planeStrideMax || uint(h) > planeRowsMax {
		panic("dsp: plane kernel arguments out of range")
	}
	zero := archsimd.BroadcastUint8x16(0)
	c8 := archsimd.BroadcastInt32x4(8)
	sh4 := archsimd.BroadcastInt32x4(-4)
	for y := 0; y < h; y++ {
		pa := d[y*ds : y*ds+width]
		sa := s[y*ss : y*ss+width]
		x := 0
		for ; x+16 <= len(pa); x += 16 {
			p := (*[16]uint8)(pa[x : x+16])
			sr := (*[16]int32)(sa[x : x+16])
			// The rounds feed the adds on their own lines (no inline-mark NOP), and
			// the high half saturates as SQXTUN2 instead of SQXTUN + ZIP1.
			dv := archsimd.LoadUint8x16Array(p)
			lo := planeAdd8(dv.InterleaveLo(zero), planeRawRound((*[8]int32)(sr[0:8]), c8, sh4)).BitsToFloat64()
			hi := planeAdd8(dv.InterleaveHi(zero), planeRawRound((*[8]int32)(sr[8:16]), c8, sh4)).BitsToFloat64()
			lo.SetElem(1, hi.GetElem(0)).ToBits().ReshapeToUint8s().StoreArray(p)
		}
		if x+8 <= len(pa) {
			p := (*[8]byte)(pa[x : x+8])
			rv := planeRawRound((*[8]int32)(sa[x:x+8]), c8, sh4)
			dv := archsimd.BroadcastUint64x2(planeLE64(p)).ReshapeToUint8s()
			planePut64(p, planeAdd8(dv.InterleaveLo(zero), rv).GetElem(0))
		}
	}
}

func planeRaw16x4(d []byte, ds int, s []int32, ss int, max uint16, h int) {
	if uint(ds) > planeStrideMax || uint(ss) > planeStrideMax || uint(h) > planeRowsMax {
		panic("dsp: plane kernel arguments out of range")
	}
	bias := archsimd.BroadcastUint16x8(0x8000)
	maxv := archsimd.BroadcastUint16x8(max)
	c8 := archsimd.BroadcastInt32x4(8)
	sh4 := archsimd.BroadcastInt32x4(-4)
	for i := 0; i < h>>1; i++ {
		y := 2 * i
		pa, pb := (*[8]byte)(d[y*ds:y*ds+8]), (*[8]byte)(d[(y+1)*ds:(y+1)*ds+8])
		sa, sb := (*[4]int32)(s[y*ss:y*ss+4]), (*[4]int32)(s[(y+1)*ss:(y+1)*ss+4])
		rv := planeRawRound4Pair(sa, sb, c8, sh4)
		dv := archsimd.LoadUint64x2Array(&[2]uint64{planeLE64(pa), planeLE64(pb)}).ReshapeToUint16s()
		o := planeAdd16(dv, rv, bias, maxv).ReshapeToUint64s()
		planePut64(pa, o.GetElem(0))
		planePut64(pb, o.GetElem(1))
	}
	if y := h &^ 1; y < h {
		pa := (*[8]byte)(d[y*ds : y*ds+8])
		sa := (*[4]int32)(s[y*ss : y*ss+4])
		rv := planeRawRound4Pair(sa, sa, c8, sh4)
		dv := archsimd.BroadcastUint64x2(planeLE64(pa)).ReshapeToUint16s()
		planePut64(pa, planeAdd16(dv, rv, bias, maxv).ReshapeToUint64s().GetElem(0))
	}
}

func planeRaw16Wide(d []byte, ds int, s []int32, ss int, max uint16, width int, h int) {
	if uint(ds) > planeStrideMax || uint(ss) > planeStrideMax || uint(h) > planeRowsMax {
		panic("dsp: plane kernel arguments out of range")
	}
	bias := archsimd.BroadcastUint16x8(0x8000)
	maxv := archsimd.BroadcastUint16x8(max)
	c8 := archsimd.BroadcastInt32x4(8)
	sh4 := archsimd.BroadcastInt32x4(-4)
	for y := 0; y < h; y++ {
		pa := d[y*ds : y*ds+2*width]
		sa := s[y*ss : y*ss+width]
		for x := 0; x+8 <= len(sa); x += 8 {
			p := (*[16]uint8)(pa[2*x : 2*x+16])
			rv := planeRawRound((*[8]int32)(sa[x:x+8]), c8, sh4)
			planeAdd16(archsimd.LoadUint8x16Array(p).ReshapeToUint16s(), rv, bias, maxv).ReshapeToUint8s().StoreArray(p)
		}
	}
}

func planeRaw16x8(d []byte, ds int, s []int32, ss int, max uint16, h int) {
	if uint(ds) > planeStrideMax || uint(ss) > planeStrideMax || uint(h) > planeRowsMax {
		panic("dsp: plane kernel arguments out of range")
	}
	bias := archsimd.BroadcastUint16x8(0x8000)
	maxv := archsimd.BroadcastUint16x8(max)
	c8 := archsimd.BroadcastInt32x4(8)
	sh4 := archsimd.BroadcastInt32x4(-4)
	for i := 0; i < h>>1; i++ {
		y := 2 * i
		pa, pb := (*[16]uint8)(d[y*ds:y*ds+16]), (*[16]uint8)(d[(y+1)*ds:(y+1)*ds+16])
		sa, sb := (*[8]int32)(s[y*ss:y*ss+8]), (*[8]int32)(s[(y+1)*ss:(y+1)*ss+8])
		oa := planeAdd16(archsimd.LoadUint8x16Array(pa).ReshapeToUint16s(), planeRawRound(sa, c8, sh4), bias, maxv)
		ob := planeAdd16(archsimd.LoadUint8x16Array(pb).ReshapeToUint16s(), planeRawRound(sb, c8, sh4), bias, maxv)
		oa.ReshapeToUint8s().StoreArray(pa)
		ob.ReshapeToUint8s().StoreArray(pb)
	}
	if y := h &^ 1; y < h {
		pa := (*[16]uint8)(d[y*ds : y*ds+16])
		rv := planeRawRound((*[8]int32)(s[y*ss:y*ss+8]), c8, sh4)
		planeAdd16(archsimd.LoadUint8x16Array(pa).ReshapeToUint16s(), rv, bias, maxv).ReshapeToUint8s().StoreArray(pa)
	}
}

// Width-specialized row loops for the transform widths 16, 32 and 64: each
// row is one fixed-size array window of the destination and of the source,
// and the column chunks index them with constants, so a row costs one
// compare per window and no per-chunk checks.

func planeResidual8W16(d []byte, ds int, r []int16, rs int, h int) {
	if uint(ds) > planeStrideMax || uint(rs) > planeStrideMax || uint(h) > planeRowsMax {
		panic("dsp: plane kernel arguments out of range")
	}
	zero := archsimd.BroadcastUint8x16(0)
	for y := 0; y < h; y++ {
		pa := (*[16]byte)(d[y*ds : y*ds+16])
		ra := (*[16]int16)(r[y*rs : y*rs+16])
		for k := 0; k < 1; k++ {
			p := (*[16]uint8)(pa[16*k : 16*k+16])
			dv := archsimd.LoadUint8x16Array(p)
			lo := planeAdd8(dv.InterleaveLo(zero), archsimd.LoadInt16x8Array((*[8]int16)(ra[16*k:16*k+8])))
			hi := planeAdd8(dv.InterleaveHi(zero), archsimd.LoadInt16x8Array((*[8]int16)(ra[16*k+8:16*k+16])))
			lo.InterleaveLo(hi).ReshapeToUint8s().StoreArray(p)
		}
	}
}

func planeResidual16W16(d []byte, ds int, r []int16, rs int, max uint16, h int) {
	if uint(ds) > planeStrideMax || uint(rs) > planeStrideMax || uint(h) > planeRowsMax {
		panic("dsp: plane kernel arguments out of range")
	}
	bias := archsimd.BroadcastUint16x8(0x8000)
	maxv := archsimd.BroadcastUint16x8(max)
	for y := 0; y < h; y++ {
		pa := (*[32]byte)(d[y*ds : y*ds+32])
		ra := (*[16]int16)(r[y*rs : y*rs+16])
		for k := 0; k < 2; k++ {
			p := (*[16]uint8)(pa[16*k : 16*k+16])
			rv := archsimd.LoadInt16x8Array((*[8]int16)(ra[8*k : 8*k+8]))
			planeAdd16(archsimd.LoadUint8x16Array(p).ReshapeToUint16s(), rv, bias, maxv).ReshapeToUint8s().StoreArray(p)
		}
	}
}

func planeRaw8W16(d []byte, ds int, s []int32, ss int, h int) {
	if uint(ds) > planeStrideMax || uint(ss) > planeStrideMax || uint(h) > planeRowsMax {
		panic("dsp: plane kernel arguments out of range")
	}
	zero := archsimd.BroadcastUint8x16(0)
	c8 := archsimd.BroadcastInt32x4(8)
	sh4 := archsimd.BroadcastInt32x4(-4)
	for y := 0; y < h; y++ {
		pa := (*[16]byte)(d[y*ds : y*ds+16])
		sa := (*[16]int32)(s[y*ss : y*ss+16])
		for k := 0; k < 1; k++ {
			p := (*[16]uint8)(pa[16*k : 16*k+16])
			dv := archsimd.LoadUint8x16Array(p)
			lo := planeAdd8(dv.InterleaveLo(zero), planeRawRound((*[8]int32)(sa[16*k:16*k+8]), c8, sh4)).BitsToFloat64()
			hi := planeAdd8(dv.InterleaveHi(zero), planeRawRound((*[8]int32)(sa[16*k+8:16*k+16]), c8, sh4)).BitsToFloat64()
			lo.SetElem(1, hi.GetElem(0)).ToBits().ReshapeToUint8s().StoreArray(p)
		}
	}
}

func planeRaw16W16(d []byte, ds int, s []int32, ss int, max uint16, h int) {
	if uint(ds) > planeStrideMax || uint(ss) > planeStrideMax || uint(h) > planeRowsMax {
		panic("dsp: plane kernel arguments out of range")
	}
	bias := archsimd.BroadcastUint16x8(0x8000)
	maxv := archsimd.BroadcastUint16x8(max)
	c8 := archsimd.BroadcastInt32x4(8)
	sh4 := archsimd.BroadcastInt32x4(-4)
	for y := 0; y < h; y++ {
		pa := (*[32]byte)(d[y*ds : y*ds+32])
		sa := (*[16]int32)(s[y*ss : y*ss+16])
		for k := 0; k < 2; k++ {
			p := (*[16]uint8)(pa[16*k : 16*k+16])
			rv := planeRawRound((*[8]int32)(sa[8*k:8*k+8]), c8, sh4)
			planeAdd16(archsimd.LoadUint8x16Array(p).ReshapeToUint16s(), rv, bias, maxv).ReshapeToUint8s().StoreArray(p)
		}
	}
}

func planeResidual8W32(d []byte, ds int, r []int16, rs int, h int) {
	if uint(ds) > planeStrideMax || uint(rs) > planeStrideMax || uint(h) > planeRowsMax {
		panic("dsp: plane kernel arguments out of range")
	}
	zero := archsimd.BroadcastUint8x16(0)
	for y := 0; y < h; y++ {
		pa := (*[32]byte)(d[y*ds : y*ds+32])
		ra := (*[32]int16)(r[y*rs : y*rs+32])
		for k := 0; k < 2; k++ {
			p := (*[16]uint8)(pa[16*k : 16*k+16])
			dv := archsimd.LoadUint8x16Array(p)
			lo := planeAdd8(dv.InterleaveLo(zero), archsimd.LoadInt16x8Array((*[8]int16)(ra[16*k:16*k+8])))
			hi := planeAdd8(dv.InterleaveHi(zero), archsimd.LoadInt16x8Array((*[8]int16)(ra[16*k+8:16*k+16])))
			lo.InterleaveLo(hi).ReshapeToUint8s().StoreArray(p)
		}
	}
}

func planeResidual16W32(d []byte, ds int, r []int16, rs int, max uint16, h int) {
	if uint(ds) > planeStrideMax || uint(rs) > planeStrideMax || uint(h) > planeRowsMax {
		panic("dsp: plane kernel arguments out of range")
	}
	bias := archsimd.BroadcastUint16x8(0x8000)
	maxv := archsimd.BroadcastUint16x8(max)
	for y := 0; y < h; y++ {
		pa := (*[64]byte)(d[y*ds : y*ds+64])
		ra := (*[32]int16)(r[y*rs : y*rs+32])
		for k := 0; k < 4; k++ {
			p := (*[16]uint8)(pa[16*k : 16*k+16])
			rv := archsimd.LoadInt16x8Array((*[8]int16)(ra[8*k : 8*k+8]))
			planeAdd16(archsimd.LoadUint8x16Array(p).ReshapeToUint16s(), rv, bias, maxv).ReshapeToUint8s().StoreArray(p)
		}
	}
}

func planeRaw8W32(d []byte, ds int, s []int32, ss int, h int) {
	if uint(ds) > planeStrideMax || uint(ss) > planeStrideMax || uint(h) > planeRowsMax {
		panic("dsp: plane kernel arguments out of range")
	}
	zero := archsimd.BroadcastUint8x16(0)
	c8 := archsimd.BroadcastInt32x4(8)
	sh4 := archsimd.BroadcastInt32x4(-4)
	for y := 0; y < h; y++ {
		pa := (*[32]byte)(d[y*ds : y*ds+32])
		sa := (*[32]int32)(s[y*ss : y*ss+32])
		for k := 0; k < 2; k++ {
			p := (*[16]uint8)(pa[16*k : 16*k+16])
			dv := archsimd.LoadUint8x16Array(p)
			lo := planeAdd8(dv.InterleaveLo(zero), planeRawRound((*[8]int32)(sa[16*k:16*k+8]), c8, sh4)).BitsToFloat64()
			hi := planeAdd8(dv.InterleaveHi(zero), planeRawRound((*[8]int32)(sa[16*k+8:16*k+16]), c8, sh4)).BitsToFloat64()
			lo.SetElem(1, hi.GetElem(0)).ToBits().ReshapeToUint8s().StoreArray(p)
		}
	}
}

func planeRaw16W32(d []byte, ds int, s []int32, ss int, max uint16, h int) {
	if uint(ds) > planeStrideMax || uint(ss) > planeStrideMax || uint(h) > planeRowsMax {
		panic("dsp: plane kernel arguments out of range")
	}
	bias := archsimd.BroadcastUint16x8(0x8000)
	maxv := archsimd.BroadcastUint16x8(max)
	c8 := archsimd.BroadcastInt32x4(8)
	sh4 := archsimd.BroadcastInt32x4(-4)
	for y := 0; y < h; y++ {
		pa := (*[64]byte)(d[y*ds : y*ds+64])
		sa := (*[32]int32)(s[y*ss : y*ss+32])
		for k := 0; k < 4; k++ {
			p := (*[16]uint8)(pa[16*k : 16*k+16])
			rv := planeRawRound((*[8]int32)(sa[8*k:8*k+8]), c8, sh4)
			planeAdd16(archsimd.LoadUint8x16Array(p).ReshapeToUint16s(), rv, bias, maxv).ReshapeToUint8s().StoreArray(p)
		}
	}
}

func planeResidual8W64(d []byte, ds int, r []int16, rs int, h int) {
	if uint(ds) > planeStrideMax || uint(rs) > planeStrideMax || uint(h) > planeRowsMax {
		panic("dsp: plane kernel arguments out of range")
	}
	zero := archsimd.BroadcastUint8x16(0)
	for y := 0; y < h; y++ {
		pa := (*[64]byte)(d[y*ds : y*ds+64])
		ra := (*[64]int16)(r[y*rs : y*rs+64])
		for k := 0; k < 4; k++ {
			p := (*[16]uint8)(pa[16*k : 16*k+16])
			dv := archsimd.LoadUint8x16Array(p)
			lo := planeAdd8(dv.InterleaveLo(zero), archsimd.LoadInt16x8Array((*[8]int16)(ra[16*k:16*k+8])))
			hi := planeAdd8(dv.InterleaveHi(zero), archsimd.LoadInt16x8Array((*[8]int16)(ra[16*k+8:16*k+16])))
			lo.InterleaveLo(hi).ReshapeToUint8s().StoreArray(p)
		}
	}
}

func planeResidual16W64(d []byte, ds int, r []int16, rs int, max uint16, h int) {
	if uint(ds) > planeStrideMax || uint(rs) > planeStrideMax || uint(h) > planeRowsMax {
		panic("dsp: plane kernel arguments out of range")
	}
	bias := archsimd.BroadcastUint16x8(0x8000)
	maxv := archsimd.BroadcastUint16x8(max)
	for y := 0; y < h; y++ {
		pa := (*[128]byte)(d[y*ds : y*ds+128])
		ra := (*[64]int16)(r[y*rs : y*rs+64])
		for k := 0; k < 8; k++ {
			p := (*[16]uint8)(pa[16*k : 16*k+16])
			rv := archsimd.LoadInt16x8Array((*[8]int16)(ra[8*k : 8*k+8]))
			planeAdd16(archsimd.LoadUint8x16Array(p).ReshapeToUint16s(), rv, bias, maxv).ReshapeToUint8s().StoreArray(p)
		}
	}
}

func planeRaw8W64(d []byte, ds int, s []int32, ss int, h int) {
	if uint(ds) > planeStrideMax || uint(ss) > planeStrideMax || uint(h) > planeRowsMax {
		panic("dsp: plane kernel arguments out of range")
	}
	zero := archsimd.BroadcastUint8x16(0)
	c8 := archsimd.BroadcastInt32x4(8)
	sh4 := archsimd.BroadcastInt32x4(-4)
	for y := 0; y < h; y++ {
		pa := (*[64]byte)(d[y*ds : y*ds+64])
		sa := (*[64]int32)(s[y*ss : y*ss+64])
		for k := 0; k < 4; k++ {
			p := (*[16]uint8)(pa[16*k : 16*k+16])
			dv := archsimd.LoadUint8x16Array(p)
			lo := planeAdd8(dv.InterleaveLo(zero), planeRawRound((*[8]int32)(sa[16*k:16*k+8]), c8, sh4)).BitsToFloat64()
			hi := planeAdd8(dv.InterleaveHi(zero), planeRawRound((*[8]int32)(sa[16*k+8:16*k+16]), c8, sh4)).BitsToFloat64()
			lo.SetElem(1, hi.GetElem(0)).ToBits().ReshapeToUint8s().StoreArray(p)
		}
	}
}

func planeRaw16W64(d []byte, ds int, s []int32, ss int, max uint16, h int) {
	if uint(ds) > planeStrideMax || uint(ss) > planeStrideMax || uint(h) > planeRowsMax {
		panic("dsp: plane kernel arguments out of range")
	}
	bias := archsimd.BroadcastUint16x8(0x8000)
	maxv := archsimd.BroadcastUint16x8(max)
	c8 := archsimd.BroadcastInt32x4(8)
	sh4 := archsimd.BroadcastInt32x4(-4)
	for y := 0; y < h; y++ {
		pa := (*[128]byte)(d[y*ds : y*ds+128])
		sa := (*[64]int32)(s[y*ss : y*ss+64])
		for k := 0; k < 8; k++ {
			p := (*[16]uint8)(pa[16*k : 16*k+16])
			rv := planeRawRound((*[8]int32)(sa[8*k:8*k+8]), c8, sh4)
			planeAdd16(archsimd.LoadUint8x16Array(p).ReshapeToUint16s(), rv, bias, maxv).ReshapeToUint8s().StoreArray(p)
		}
	}
}
