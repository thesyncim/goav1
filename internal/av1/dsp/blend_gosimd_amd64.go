// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && amd64 && !purego

package dsp

import "simd/archsimd"

// AVX2-width row loops for the Go SIMD BlendA64Mask (see blend_gosimd_common.go
// for the shared contract). Right shifts by constants lower to immediate
// VPSRLW/VPSRLD, so the shift counts need no loop-invariant vectors. The 12-bit
// weighted sum widens the eight u16 lanes to one 256-bit vector of u32 lanes.

// blendA64NoSubSIMD blends width/8 groups of eight samples per row with
// one-to-one weights.
func blendA64NoSubSIMD(a blendA64MaskArgs, groups int) bool {
	sampleMax := archsimd.BroadcastUint16x8(0)
	weightMax := archsimd.BroadcastUint16x8(0)
	for row := range a.height {
		dstRow := a.dst[row*a.dstStride : row*a.dstStride+a.width]
		s0Row := a.src0[row*a.src0Stride : row*a.src0Stride+a.width]
		s1Row := a.src1[row*a.src1Stride : row*a.src1Stride+a.width]
		mRow := a.mask[row*a.maskStride : row*a.maskStride+a.width]
		for g := range groups {
			x := g << 3
			s0 := archsimd.LoadUint16x8Array((*[8]uint16)(s0Row[x:]))
			s1 := archsimd.LoadUint16x8Array((*[8]uint16)(s1Row[x:]))
			m := blendLoadWeights8(mRow[x:])
			sampleMax = sampleMax.Max(s0).Max(s1)
			weightMax = weightMax.Max(m)
			blendStore8(a.max, m, s0, s1, dstRow[x:])
		}
	}
	return blendMaxLane(sampleMax) <= a.max && blendMaxLane(weightMax) <= blendA64MaxAlpha
}

// blendA64SubXYSIMD blends width/8 groups of eight samples per row with 2x2
// subsampled weights. Each group reads sixteen mask bytes from two mask rows.
func blendA64SubXYSIMD(a blendA64MaskArgs, groups int) bool {
	sampleMax := archsimd.BroadcastUint16x8(0)
	weightMax := archsimd.BroadcastUint16x8(0)
	for row := range a.height {
		dstRow := a.dst[row*a.dstStride : row*a.dstStride+a.width]
		s0Row := a.src0[row*a.src0Stride : row*a.src0Stride+a.width]
		s1Row := a.src1[row*a.src1Stride : row*a.src1Stride+a.width]
		mRowA := a.mask[(2*row)*a.maskStride : (2*row)*a.maskStride+2*a.width]
		mRowB := a.mask[(2*row+1)*a.maskStride : (2*row+1)*a.maskStride+2*a.width]
		for g := range groups {
			x := g << 3
			s0 := archsimd.LoadUint16x8Array((*[8]uint16)(s0Row[x:]))
			s1 := archsimd.LoadUint16x8Array((*[8]uint16)(s1Row[x:]))
			m := blendPairSums16(mRowA[2*x:]).Add(blendPairSums16(mRowB[2*x:])).
				Add(archsimd.BroadcastUint16x8(2)).ShiftAllRight(2)
			sampleMax = sampleMax.Max(s0).Max(s1)
			weightMax = weightMax.Max(m)
			blendStore8(a.max, m, s0, s1, dstRow[x:])
		}
	}
	return blendMaxLane(sampleMax) <= a.max && blendMaxLane(weightMax) <= blendA64MaxAlpha
}

// blendStore8 blends eight lanes and stores them to dst[:8]. Samples up to 1023
// use the u16 form; 12-bit samples use the u32 form.
func blendStore8(maxv uint16, m, s0, s1 archsimd.Uint16x8, dst []uint16) {
	inv := archsimd.BroadcastUint16x8(blendA64MaxAlpha).Sub(m)
	if maxv <= 1023 {
		m.Mul(s0).Add(inv.Mul(s1)).Add(archsimd.BroadcastUint16x8(1 << 5)).ShiftAllRight(6).StoreArray((*[8]uint16)(dst))
		return
	}
	mLo, mHi := blendWiden32(m)
	sLo, sHi := blendWiden32(s0)
	iLo, iHi := blendWiden32(inv)
	tLo, tHi := blendWiden32(s1)
	round := archsimd.BroadcastUint32x4(1 << 5)
	lo := mLo.Mul(sLo).Add(iLo.Mul(tLo)).Add(round).ShiftAllRight(6)
	hi := mHi.Mul(sHi).Add(iHi.Mul(tHi)).Add(round).ShiftAllRight(6)
	// Narrow the eight results (each at most 4095) to u16 with the AVX2 pack. The
	// 128-bit-wide SaturateToUint16 forms are AVX-512 only, so they are not used.
	// VPACKUSDW packs each 128-bit half of x as [x.lo4, x.lo4]; the low halves of
	// the two 128-bit lanes then hold the eight results in order.
	packed := archsimd.BroadcastUint32x8(0).SetLo(lo).SetHi(hi).ConvertToInt32().
		SaturateToUint16ConcatGrouped(archsimd.BroadcastInt32x8(0))
	packed.GetLo().ReshapeToUint64s().
		InterleaveLo(packed.GetHi().ReshapeToUint64s()).
		ReshapeToUint16s().StoreArray((*[8]uint16)(dst))
}

// blendZeroU16 is the zero vector the u32 widening interleaves with. It is read
// from memory rather than broadcast: an interleave with a constant zero is
// rewritten to VPMOVZXWD, which this archsimd encodes with EVEX and which
// faults on AVX2 hosts without AVX-512.
var blendZeroU16 [8]uint16

// blendWiden32 zero-extends the four low and the four high u16 lanes of v to two
// vectors of u32 lanes (VPUNPCKLWD/VPUNPCKHWD against zero).
func blendWiden32(v archsimd.Uint16x8) (lo, hi archsimd.Uint32x4) {
	zero := archsimd.LoadUint16x8Array(&blendZeroU16)
	return v.InterleaveLo(zero).ReshapeToUint32s(), v.InterleaveHi(zero).ReshapeToUint32s()
}

// blendPairSums16 returns the 2x2 horizontal pair sums of sixteen mask bytes as
// eight u16 lanes: lane c holds p[2c] + p[2c+1]. Each u16 lane of the reshaped
// vector holds its byte pair little-endian, so the low byte and the high byte
// are the two horizontal neighbours.
func blendPairSums16(p []uint8) archsimd.Uint16x8 {
	w := archsimd.LoadUint8x16Array((*[16]uint8)(p[:16])).ReshapeToUint16s()
	return w.And(archsimd.BroadcastUint16x8(0x00ff)).Add(w.ShiftAllRight(8))
}

// blendMaxLane returns the largest of the eight u16 lanes. archsimd exposes no
// horizontal maximum on AVX2 for 128-bit vectors, so the lanes are stored once
// per block.
func blendMaxLane(v archsimd.Uint16x8) uint16 {
	var lanes [8]uint16
	v.StoreArray(&lanes)
	m := lanes[0]
	for _, x := range lanes[1:] {
		if x > m {
			m = x
		}
	}
	return m
}

// blendLoadWeights8 loads eight mask bytes as u16 lanes. A full sixteen-byte
// vector load is used whenever the row has that many bytes left; the final
// group of a row copies into a local array instead.
func blendLoadWeights8(p []uint8) archsimd.Uint16x8 {
	if len(p) >= 16 {
		return archsimd.LoadUint8x16Array((*[16]uint8)(p)).ExtendLo8ToUint16()
	}
	var t [16]uint8
	*(*[8]uint8)(t[:8]) = [8]uint8(p[:8])
	return archsimd.LoadUint8x16Array(&t).ExtendLo8ToUint16()
}
