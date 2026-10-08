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
//     both sums shifted right by four saturate to 32767 in the narrow (SQXTN);
//     this is the sequence the retired assembly used. arm64 archsimd has no
//     immediate-shift form, so the shift is SSHL by a hoisted vector of -4.
//
// The 8- and 4-sample rows of 8-bit blocks, and the 4-sample rows of 16-bit
// blocks, are narrower than a vector register. They are read and written with
// exact-width scalar accesses, so no byte outside the block is touched, and are
// paired two rows to a vector.

import (
	"simd/archsimd"
	"unsafe"
)

// addResidualPlaneBlockSIMD is the Go SIMD residual add for 8-bit and 16-bit
// samples.
func addResidualPlaneBlockSIMD(block planeBlock, bytesPerSample int, max uint16, width int, residual []int16, residualStride int) {
	h := block.height
	if h <= 0 || (width%8 != 0 && width != 4) || (bytesPerSample == 1 && max != 0xff) || (bytesPerSample != 1 && bytesPerSample != 2) {
		addResidualPlaneBlockPureGo(block, bytesPerSample, max, width, residual, residualStride)
		return
	}
	// The caller validated the extents; these two checks keep the raw-pointer
	// loops below inside the slices even if that contract is broken.
	_ = block.pix[(h-1)*block.stride+width*bytesPerSample-1]
	_ = residual[(h-1)*residualStride+width-1]
	dp := unsafe.Pointer(unsafe.SliceData(block.pix))
	rp := unsafe.Pointer(unsafe.SliceData(residual))
	if bytesPerSample == 1 {
		switch width {
		case 4:
			planeResidual8x4(dp, block.stride, rp, 2*residualStride, h)
		case 8:
			planeResidual8x8(dp, block.stride, rp, 2*residualStride, h)
		default:
			planeResidual8Wide(dp, block.stride, rp, 2*residualStride, width, h)
		}
		return
	}
	switch width {
	case 4:
		planeResidual16x4(dp, block.stride, rp, 2*residualStride, max, h)
		return
	case 8:
		planeResidual16x8(dp, block.stride, rp, 2*residualStride, max, h)
		return
	}
	planeResidual16Wide(dp, block.stride, rp, 2*residualStride, max, width, h)
}

// addRawTransformPlaneBlockSIMD is the Go SIMD fused raw-transform add for 8-bit
// and 16-bit samples.
func addRawTransformPlaneBlockSIMD(block planeBlock, bytesPerSample int, max uint16, width int, raw []int32, rawStride int) {
	h := block.height
	if h <= 0 || (width%8 != 0 && width != 4) || (bytesPerSample == 1 && max != 0xff) || (bytesPerSample != 1 && bytesPerSample != 2) {
		addRawTransformPlaneBlockPureGo(block, bytesPerSample, max, width, raw, rawStride)
		return
	}
	_ = block.pix[(h-1)*block.stride+width*bytesPerSample-1]
	_ = raw[(h-1)*rawStride+width-1]
	dp := unsafe.Pointer(unsafe.SliceData(block.pix))
	sp := unsafe.Pointer(unsafe.SliceData(raw))
	if bytesPerSample == 1 {
		switch width {
		case 4:
			planeRaw8x4(dp, block.stride, sp, 4*rawStride, h)
		case 8:
			planeRaw8x8(dp, block.stride, sp, 4*rawStride, h)
		default:
			planeRaw8Wide(dp, block.stride, sp, 4*rawStride, width, h)
		}
		return
	}
	switch width {
	case 4:
		planeRaw16x4(dp, block.stride, sp, 4*rawStride, max, h)
		return
	case 8:
		planeRaw16x8(dp, block.stride, sp, 4*rawStride, max, h)
		return
	}
	planeRaw16Wide(dp, block.stride, sp, 4*rawStride, max, width, h)
}

// planeLoadU64 and planeStoreU64 are exact-width unaligned 8-byte accesses.
func planeLoadU64(p unsafe.Pointer) uint64     { return *(*uint64)(p) }
func planeStoreU64(p unsafe.Pointer, v uint64) { *(*uint64)(p) = v }
func planeLoadU32(p unsafe.Pointer) uint32     { return *(*uint32)(p) }
func planeStoreU32(p unsafe.Pointer, v uint32) { *(*uint32)(p) = v }

// planePairU64 returns the vector whose low 64 bits are a and high 64 bits b.
func planePairU64(a, b uint64) archsimd.Uint64x2 {
	t := [2]uint64{a, b}
	return archsimd.LoadUint64x2Array(&t)
}

// planeRawRound returns sat16(floor((v + 8) / 16)) of the eight int32 samples
// at p, in lane order.
func planeRawRound(p unsafe.Pointer, c8, sh4 archsimd.Int32x4) archsimd.Int16x8 {
	return planeRawRound4Pair(p, unsafe.Add(p, 16), c8, sh4)
}

// planeRawRound4Pair returns sat16(floor((v + 8) / 16)) of four int32 samples at
// a in lanes 0-3 and four at b in lanes 4-7.
func planeRawRound4Pair(a, b unsafe.Pointer, c8, sh4 archsimd.Int32x4) archsimd.Int16x8 {
	lo := archsimd.LoadInt32x4Array((*[4]int32)(a)).AddSaturated(c8).Shift(sh4).SaturateToInt16()
	hi := archsimd.LoadInt32x4Array((*[4]int32)(b)).AddSaturated(c8).Shift(sh4).SaturateToInt16()
	return lo.ToBits().ReshapeToUint64s().InterleaveLo(hi.ToBits().ReshapeToUint64s()).ReshapeToUint16s().BitsToInt16()
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

// The kernels below take raw row pointers and byte strides for the destination
// (dp, ds) and the residual or raw source (rp/sp, rs/ss); h is at least one.

func planeResidual8x4(dp unsafe.Pointer, ds int, rp unsafe.Pointer, rs int, h int) {
	zero := archsimd.BroadcastUint8x16(0)
	y := 0
	for ; y+1 < h; y += 2 {
		pa := unsafe.Add(dp, y*ds)
		pb := unsafe.Add(pa, ds)
		ra := unsafe.Add(rp, y*rs)
		rb := unsafe.Add(ra, rs)
		d := archsimd.BroadcastUint64x2(uint64(planeLoadU32(pa)) | uint64(planeLoadU32(pb))<<32).ReshapeToUint8s()
		r := planePairU64(planeLoadU64(ra), planeLoadU64(rb)).ReshapeToUint16s().BitsToInt16()
		w := planeAdd8(d.InterleaveLo(zero), r).GetElem(0)
		planeStoreU32(pa, uint32(w))
		planeStoreU32(pb, uint32(w>>32))
	}
	if y < h {
		pa := unsafe.Add(dp, y*ds)
		ra := unsafe.Add(rp, y*rs)
		d := archsimd.BroadcastUint64x2(uint64(planeLoadU32(pa))).ReshapeToUint8s()
		r := archsimd.BroadcastUint64x2(planeLoadU64(ra)).ReshapeToUint16s().BitsToInt16()
		planeStoreU32(pa, uint32(planeAdd8(d.InterleaveLo(zero), r).GetElem(0)))
	}
}

func planeResidual8x8(dp unsafe.Pointer, ds int, rp unsafe.Pointer, rs int, h int) {
	zero := archsimd.BroadcastUint8x16(0)
	y := 0
	for ; y+1 < h; y += 2 {
		pa := unsafe.Add(dp, y*ds)
		pb := unsafe.Add(pa, ds)
		ra := unsafe.Add(rp, y*rs)
		rb := unsafe.Add(ra, rs)
		d := planePairU64(planeLoadU64(pa), planeLoadU64(pb)).ReshapeToUint8s()
		oa := planeAdd8(d.InterleaveLo(zero), archsimd.LoadInt16x8Array((*[8]int16)(ra)))
		ob := planeAdd8(d.InterleaveHi(zero), archsimd.LoadInt16x8Array((*[8]int16)(rb)))
		planeStoreU64(pa, oa.GetElem(0))
		planeStoreU64(pb, ob.GetElem(0))
	}
	if y < h {
		pa := unsafe.Add(dp, y*ds)
		ra := unsafe.Add(rp, y*rs)
		d := archsimd.BroadcastUint64x2(planeLoadU64(pa)).ReshapeToUint8s()
		planeStoreU64(pa, planeAdd8(d.InterleaveLo(zero), archsimd.LoadInt16x8Array((*[8]int16)(ra))).GetElem(0))
	}
}

func planeResidual8Wide(dp unsafe.Pointer, ds int, rp unsafe.Pointer, rs int, width int, h int) {
	zero := archsimd.BroadcastUint8x16(0)
	for y := 0; y < h; y++ {
		pa := unsafe.Add(dp, y*ds)
		ra := unsafe.Add(rp, y*rs)
		x := 0
		for ; x+16 <= width; x += 16 {
			p := unsafe.Add(pa, x)
			r := unsafe.Add(ra, 2*x)
			d := archsimd.LoadUint8x16Array((*[16]uint8)(p))
			lo := planeAdd8(d.InterleaveLo(zero), archsimd.LoadInt16x8Array((*[8]int16)(r)))
			hi := planeAdd8(d.InterleaveHi(zero), archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(r, 16))))
			lo.InterleaveLo(hi).ReshapeToUint8s().StoreArray((*[16]uint8)(p))
		}
		if x < width {
			p := unsafe.Add(pa, x)
			d := archsimd.BroadcastUint64x2(planeLoadU64(p)).ReshapeToUint8s()
			planeStoreU64(p, planeAdd8(d.InterleaveLo(zero), archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(ra, 2*x)))).GetElem(0))
		}
	}
}

func planeResidual16x4(dp unsafe.Pointer, ds int, rp unsafe.Pointer, rs int, max uint16, h int) {
	bias := archsimd.BroadcastUint16x8(0x8000)
	maxv := archsimd.BroadcastUint16x8(max)
	y := 0
	for ; y+1 < h; y += 2 {
		pa := unsafe.Add(dp, y*ds)
		pb := unsafe.Add(pa, ds)
		ra := unsafe.Add(rp, y*rs)
		rb := unsafe.Add(ra, rs)
		d := planePairU64(planeLoadU64(pa), planeLoadU64(pb)).ReshapeToUint16s()
		r := planePairU64(planeLoadU64(ra), planeLoadU64(rb)).ReshapeToUint16s().BitsToInt16()
		o := planeAdd16(d, r, bias, maxv).ReshapeToUint64s()
		planeStoreU64(pa, o.GetElem(0))
		planeStoreU64(pb, o.GetElem(1))
	}
	if y < h {
		pa := unsafe.Add(dp, y*ds)
		ra := unsafe.Add(rp, y*rs)
		d := archsimd.BroadcastUint64x2(planeLoadU64(pa)).ReshapeToUint16s()
		r := archsimd.BroadcastUint64x2(planeLoadU64(ra)).ReshapeToUint16s().BitsToInt16()
		planeStoreU64(pa, planeAdd16(d, r, bias, maxv).ReshapeToUint64s().GetElem(0))
	}
}

func planeResidual16Wide(dp unsafe.Pointer, ds int, rp unsafe.Pointer, rs int, max uint16, width int, h int) {
	bias := archsimd.BroadcastUint16x8(0x8000)
	maxv := archsimd.BroadcastUint16x8(max)
	for y := 0; y < h; y++ {
		pa := unsafe.Add(dp, y*ds)
		ra := unsafe.Add(rp, y*rs)
		for x := 0; x < width; x += 8 {
			p := (*[16]uint8)(unsafe.Add(pa, 2*x))
			r := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(ra, 2*x)))
			planeAdd16(archsimd.LoadUint8x16Array(p).ReshapeToUint16s(), r, bias, maxv).ReshapeToUint8s().StoreArray(p)
		}
	}
}

func planeRaw8x4(dp unsafe.Pointer, ds int, sp unsafe.Pointer, ss int, h int) {
	zero := archsimd.BroadcastUint8x16(0)
	c8 := archsimd.BroadcastInt32x4(8)
	sh4 := archsimd.BroadcastInt32x4(-4)
	y := 0
	for ; y+1 < h; y += 2 {
		pa := unsafe.Add(dp, y*ds)
		pb := unsafe.Add(pa, ds)
		sa := unsafe.Add(sp, y*ss)
		r := planeRawRound4Pair(sa, unsafe.Add(sa, ss), c8, sh4)
		d := archsimd.BroadcastUint64x2(uint64(planeLoadU32(pa)) | uint64(planeLoadU32(pb))<<32).ReshapeToUint8s()
		w := planeAdd8(d.InterleaveLo(zero), r).GetElem(0)
		planeStoreU32(pa, uint32(w))
		planeStoreU32(pb, uint32(w>>32))
	}
	if y < h {
		pa := unsafe.Add(dp, y*ds)
		sa := unsafe.Add(sp, y*ss)
		r := planeRawRound4Pair(sa, sa, c8, sh4)
		d := archsimd.BroadcastUint64x2(uint64(planeLoadU32(pa))).ReshapeToUint8s()
		planeStoreU32(pa, uint32(planeAdd8(d.InterleaveLo(zero), r).GetElem(0)))
	}
}

func planeRaw8x8(dp unsafe.Pointer, ds int, sp unsafe.Pointer, ss int, h int) {
	zero := archsimd.BroadcastUint8x16(0)
	c8 := archsimd.BroadcastInt32x4(8)
	sh4 := archsimd.BroadcastInt32x4(-4)
	y := 0
	for ; y+1 < h; y += 2 {
		pa := unsafe.Add(dp, y*ds)
		pb := unsafe.Add(pa, ds)
		sa := unsafe.Add(sp, y*ss)
		ra := planeRawRound(sa, c8, sh4)
		rb := planeRawRound(unsafe.Add(sa, ss), c8, sh4)
		d := planePairU64(planeLoadU64(pa), planeLoadU64(pb)).ReshapeToUint8s()
		planeStoreU64(pa, planeAdd8(d.InterleaveLo(zero), ra).GetElem(0))
		planeStoreU64(pb, planeAdd8(d.InterleaveHi(zero), rb).GetElem(0))
	}
	if y < h {
		pa := unsafe.Add(dp, y*ds)
		r := planeRawRound(unsafe.Add(sp, y*ss), c8, sh4)
		d := archsimd.BroadcastUint64x2(planeLoadU64(pa)).ReshapeToUint8s()
		planeStoreU64(pa, planeAdd8(d.InterleaveLo(zero), r).GetElem(0))
	}
}

func planeRaw8Wide(dp unsafe.Pointer, ds int, sp unsafe.Pointer, ss int, width int, h int) {
	zero := archsimd.BroadcastUint8x16(0)
	c8 := archsimd.BroadcastInt32x4(8)
	sh4 := archsimd.BroadcastInt32x4(-4)
	for y := 0; y < h; y++ {
		pa := unsafe.Add(dp, y*ds)
		sa := unsafe.Add(sp, y*ss)
		x := 0
		for ; x+16 <= width; x += 16 {
			p := unsafe.Add(pa, x)
			s := unsafe.Add(sa, 4*x)
			rlo := planeRawRound(s, c8, sh4)
			rhi := planeRawRound(unsafe.Add(s, 32), c8, sh4)
			d := archsimd.LoadUint8x16Array((*[16]uint8)(p))
			lo := planeAdd8(d.InterleaveLo(zero), rlo)
			hi := planeAdd8(d.InterleaveHi(zero), rhi)
			lo.InterleaveLo(hi).ReshapeToUint8s().StoreArray((*[16]uint8)(p))
		}
		if x < width {
			p := unsafe.Add(pa, x)
			r := planeRawRound(unsafe.Add(sa, 4*x), c8, sh4)
			d := archsimd.BroadcastUint64x2(planeLoadU64(p)).ReshapeToUint8s()
			planeStoreU64(p, planeAdd8(d.InterleaveLo(zero), r).GetElem(0))
		}
	}
}

func planeRaw16x4(dp unsafe.Pointer, ds int, sp unsafe.Pointer, ss int, max uint16, h int) {
	bias := archsimd.BroadcastUint16x8(0x8000)
	maxv := archsimd.BroadcastUint16x8(max)
	c8 := archsimd.BroadcastInt32x4(8)
	sh4 := archsimd.BroadcastInt32x4(-4)
	y := 0
	for ; y+1 < h; y += 2 {
		pa := unsafe.Add(dp, y*ds)
		pb := unsafe.Add(pa, ds)
		sa := unsafe.Add(sp, y*ss)
		r := planeRawRound4Pair(sa, unsafe.Add(sa, ss), c8, sh4)
		d := planePairU64(planeLoadU64(pa), planeLoadU64(pb)).ReshapeToUint16s()
		o := planeAdd16(d, r, bias, maxv).ReshapeToUint64s()
		planeStoreU64(pa, o.GetElem(0))
		planeStoreU64(pb, o.GetElem(1))
	}
	if y < h {
		pa := unsafe.Add(dp, y*ds)
		sa := unsafe.Add(sp, y*ss)
		r := planeRawRound4Pair(sa, sa, c8, sh4)
		d := archsimd.BroadcastUint64x2(planeLoadU64(pa)).ReshapeToUint16s()
		planeStoreU64(pa, planeAdd16(d, r, bias, maxv).ReshapeToUint64s().GetElem(0))
	}
}

func planeRaw16Wide(dp unsafe.Pointer, ds int, sp unsafe.Pointer, ss int, max uint16, width int, h int) {
	bias := archsimd.BroadcastUint16x8(0x8000)
	maxv := archsimd.BroadcastUint16x8(max)
	c8 := archsimd.BroadcastInt32x4(8)
	sh4 := archsimd.BroadcastInt32x4(-4)
	for y := 0; y < h; y++ {
		pa := unsafe.Add(dp, y*ds)
		sa := unsafe.Add(sp, y*ss)
		for x := 0; x < width; x += 8 {
			p := (*[16]uint8)(unsafe.Add(pa, 2*x))
			r := planeRawRound(unsafe.Add(sa, 4*x), c8, sh4)
			planeAdd16(archsimd.LoadUint8x16Array(p).ReshapeToUint16s(), r, bias, maxv).ReshapeToUint8s().StoreArray(p)
		}
	}
}

func planeResidual16x8(dp unsafe.Pointer, ds int, rp unsafe.Pointer, rs int, max uint16, h int) {
	bias := archsimd.BroadcastUint16x8(0x8000)
	maxv := archsimd.BroadcastUint16x8(max)
	y := 0
	for ; y+1 < h; y += 2 {
		pa := (*[16]uint8)(unsafe.Add(dp, y*ds))
		pb := (*[16]uint8)(unsafe.Add(unsafe.Pointer(pa), ds))
		ra := unsafe.Add(rp, y*rs)
		oa := planeAdd16(archsimd.LoadUint8x16Array(pa).ReshapeToUint16s(), archsimd.LoadInt16x8Array((*[8]int16)(ra)), bias, maxv)
		ob := planeAdd16(archsimd.LoadUint8x16Array(pb).ReshapeToUint16s(), archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(ra, rs))), bias, maxv)
		oa.ReshapeToUint8s().StoreArray(pa)
		ob.ReshapeToUint8s().StoreArray(pb)
	}
	if y < h {
		pa := (*[16]uint8)(unsafe.Add(dp, y*ds))
		r := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(rp, y*rs)))
		planeAdd16(archsimd.LoadUint8x16Array(pa).ReshapeToUint16s(), r, bias, maxv).ReshapeToUint8s().StoreArray(pa)
	}
}

func planeRaw16x8(dp unsafe.Pointer, ds int, sp unsafe.Pointer, ss int, max uint16, h int) {
	bias := archsimd.BroadcastUint16x8(0x8000)
	maxv := archsimd.BroadcastUint16x8(max)
	c8 := archsimd.BroadcastInt32x4(8)
	sh4 := archsimd.BroadcastInt32x4(-4)
	y := 0
	for ; y+1 < h; y += 2 {
		pa := (*[16]uint8)(unsafe.Add(dp, y*ds))
		pb := (*[16]uint8)(unsafe.Add(unsafe.Pointer(pa), ds))
		sa := unsafe.Add(sp, y*ss)
		oa := planeAdd16(archsimd.LoadUint8x16Array(pa).ReshapeToUint16s(), planeRawRound(sa, c8, sh4), bias, maxv)
		ob := planeAdd16(archsimd.LoadUint8x16Array(pb).ReshapeToUint16s(), planeRawRound(unsafe.Add(sa, ss), c8, sh4), bias, maxv)
		oa.ReshapeToUint8s().StoreArray(pa)
		ob.ReshapeToUint8s().StoreArray(pb)
	}
	if y < h {
		pa := (*[16]uint8)(unsafe.Add(dp, y*ds))
		r := planeRawRound(unsafe.Add(sp, y*ss), c8, sh4)
		planeAdd16(archsimd.LoadUint8x16Array(pa).ReshapeToUint16s(), r, bias, maxv).ReshapeToUint8s().StoreArray(pa)
	}
}
