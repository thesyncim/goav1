// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package dsp

import (
	"encoding/binary"
	"simd/archsimd"
	"unsafe"
)

// NEON-width row loops for the Go SIMD BlendA64Mask (see blend_gosimd_common.go
// for the shared contract).
//
// Each row is addressed through a base pointer and the group loop advances a
// byte offset. The row slices were validated by BlendA64Mask, and every access
// below stays inside its row, so the per-group slice bounds checks are not
// needed. On arm64 a right shift by a constant lowers to a per-use shift-amount
// vector, so the shift counts and rounding constants are built once per call as
// loop-invariant locals and applied with the variable Shift.

// blendA64NoSubSIMD blends width/8 groups of eight samples per row with
// one-to-one weights.
func blendA64NoSubSIMD(a blendA64MaskArgs, groups int) bool {
	if a.max > 1023 {
		return blendA64NoSubWideSIMD(a, groups)
	}
	sampleMax := archsimd.BroadcastUint16x8(0)
	weightMax := archsimd.BroadcastUint16x8(0)
	c64 := archsimd.BroadcastUint16x8(blendA64MaxAlpha)
	round := archsimd.BroadcastUint16x8(1 << 5)
	sh6 := archsimd.BroadcastInt16x8(-6)
	for row := range a.height {
		dstP := unsafe.Pointer(&a.dst[row*a.dstStride])
		s0P := unsafe.Pointer(&a.src0[row*a.src0Stride])
		s1P := unsafe.Pointer(&a.src1[row*a.src1Stride])
		mP := unsafe.Pointer(&a.mask[row*a.maskStride])
		for g := range groups {
			off := g << 4 // byte offset of sample group g (eight u16 lanes)
			s0 := blendLoadU16x8(s0P, off)
			s1 := blendLoadU16x8(s1P, off)
			m := blendLoadWeightsNoSub(mP, g, groups)
			sampleMax = sampleMax.Max(s0).Max(s1)
			weightMax = weightMax.Max(m)
			m.Mul(s0).Add(c64.Sub(m).Mul(s1)).Add(round).Shift(sh6).StoreArray((*[8]uint16)(unsafe.Add(dstP, off)))
		}
	}
	return sampleMax.ReduceMax() <= a.max && weightMax.ReduceMax() <= blendA64MaxAlpha
}

// blendA64NoSubWideSIMD is blendA64NoSubSIMD for 12-bit samples, where the
// weighted sum exceeds u16 and is formed in u32 lane pairs.
func blendA64NoSubWideSIMD(a blendA64MaskArgs, groups int) bool {
	sampleMax := archsimd.BroadcastUint16x8(0)
	weightMax := archsimd.BroadcastUint16x8(0)
	c64 := archsimd.BroadcastUint16x8(blendA64MaxAlpha)
	round := archsimd.BroadcastUint32x4(1 << 5)
	sh6 := archsimd.BroadcastInt32x4(-6)
	for row := range a.height {
		dstP := unsafe.Pointer(&a.dst[row*a.dstStride])
		s0P := unsafe.Pointer(&a.src0[row*a.src0Stride])
		s1P := unsafe.Pointer(&a.src1[row*a.src1Stride])
		mP := unsafe.Pointer(&a.mask[row*a.maskStride])
		for g := range groups {
			off := g << 4
			s0 := blendLoadU16x8(s0P, off)
			s1 := blendLoadU16x8(s1P, off)
			m := blendLoadWeightsNoSub(mP, g, groups)
			sampleMax = sampleMax.Max(s0).Max(s1)
			weightMax = weightMax.Max(m)
			blendStoreWide8(m, s0, s1, c64, round, sh6, unsafe.Add(dstP, off))
		}
	}
	return sampleMax.ReduceMax() <= a.max && weightMax.ReduceMax() <= blendA64MaxAlpha
}

// blendA64SubXYSIMD blends width/8 groups of eight samples per row with 2x2
// subsampled weights. Each group reads sixteen mask bytes from two mask rows.
func blendA64SubXYSIMD(a blendA64MaskArgs, groups int) bool {
	if a.max > 1023 {
		return blendA64SubXYWideSIMD(a, groups)
	}
	sampleMax := archsimd.BroadcastUint16x8(0)
	weightMax := archsimd.BroadcastUint16x8(0)
	c64 := archsimd.BroadcastUint16x8(blendA64MaxAlpha)
	round2 := archsimd.BroadcastUint16x8(2)
	round := archsimd.BroadcastUint16x8(1 << 5)
	sh6 := archsimd.BroadcastInt16x8(-6)
	sh2 := archsimd.BroadcastInt16x8(-2)
	sh8 := archsimd.BroadcastInt16x8(-8)
	lowByte := archsimd.BroadcastUint16x8(0x00ff)
	for row := range a.height {
		dstP := unsafe.Pointer(&a.dst[row*a.dstStride])
		s0P := unsafe.Pointer(&a.src0[row*a.src0Stride])
		s1P := unsafe.Pointer(&a.src1[row*a.src1Stride])
		mAP := unsafe.Pointer(&a.mask[(2*row)*a.maskStride])
		mBP := unsafe.Pointer(&a.mask[(2*row+1)*a.maskStride])
		for g := range groups {
			off := g << 4
			s0 := blendLoadU16x8(s0P, off)
			s1 := blendLoadU16x8(s1P, off)
			m := blendPairSums16(mAP, off, lowByte, sh8).Add(blendPairSums16(mBP, off, lowByte, sh8)).Add(round2).Shift(sh2)
			sampleMax = sampleMax.Max(s0).Max(s1)
			weightMax = weightMax.Max(m)
			m.Mul(s0).Add(c64.Sub(m).Mul(s1)).Add(round).Shift(sh6).StoreArray((*[8]uint16)(unsafe.Add(dstP, off)))
		}
	}
	return sampleMax.ReduceMax() <= a.max && weightMax.ReduceMax() <= blendA64MaxAlpha
}

// blendA64SubXYWideSIMD is blendA64SubXYSIMD for 12-bit samples.
func blendA64SubXYWideSIMD(a blendA64MaskArgs, groups int) bool {
	sampleMax := archsimd.BroadcastUint16x8(0)
	weightMax := archsimd.BroadcastUint16x8(0)
	c64 := archsimd.BroadcastUint16x8(blendA64MaxAlpha)
	round2 := archsimd.BroadcastUint16x8(2)
	round := archsimd.BroadcastUint32x4(1 << 5)
	sh6 := archsimd.BroadcastInt32x4(-6)
	sh2 := archsimd.BroadcastInt16x8(-2)
	sh8 := archsimd.BroadcastInt16x8(-8)
	lowByte := archsimd.BroadcastUint16x8(0x00ff)
	for row := range a.height {
		dstP := unsafe.Pointer(&a.dst[row*a.dstStride])
		s0P := unsafe.Pointer(&a.src0[row*a.src0Stride])
		s1P := unsafe.Pointer(&a.src1[row*a.src1Stride])
		mAP := unsafe.Pointer(&a.mask[(2*row)*a.maskStride])
		mBP := unsafe.Pointer(&a.mask[(2*row+1)*a.maskStride])
		for g := range groups {
			off := g << 4
			s0 := blendLoadU16x8(s0P, off)
			s1 := blendLoadU16x8(s1P, off)
			m := blendPairSums16(mAP, off, lowByte, sh8).Add(blendPairSums16(mBP, off, lowByte, sh8)).Add(round2).Shift(sh2)
			sampleMax = sampleMax.Max(s0).Max(s1)
			weightMax = weightMax.Max(m)
			blendStoreWide8(m, s0, s1, c64, round, sh6, unsafe.Add(dstP, off))
		}
	}
	return sampleMax.ReduceMax() <= a.max && weightMax.ReduceMax() <= blendA64MaxAlpha
}

// blendLoadU16x8 loads eight u16 samples at byte offset off from p.
func blendLoadU16x8(p unsafe.Pointer, off int) archsimd.Uint16x8 {
	return archsimd.LoadUint16x8Array((*[8]uint16)(unsafe.Add(p, off)))
}

// blendLoadWeightsNoSub loads the eight weights of group g from the mask row at
// p. Groups before the last read a full sixteen-byte vector, which stays inside
// the row; the last group loads its eight bytes as one 64-bit scalar so the read
// cannot run past the row.
func blendLoadWeightsNoSub(p unsafe.Pointer, g int, groups int) archsimd.Uint16x8 {
	if g+1 < groups {
		return archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(p, g<<3))).ExtendLo8ToUint16()
	}
	w := binary.LittleEndian.Uint64(unsafe.Slice((*byte)(unsafe.Add(p, g<<3)), 8))
	return archsimd.BroadcastUint64x2(w).ReshapeToUint8s().ExtendLo8ToUint16()
}

// blendStoreWide8 computes (m*s0 + (64-m)*s1 + 32) >> 6 for eight lanes in u32
// lane pairs (the low four and high four lanes, since the products exceed u16
// for 12-bit samples) and stores the eight results at dst.
func blendStoreWide8(m, s0, s1, c64 archsimd.Uint16x8, round archsimd.Uint32x4, sh6 archsimd.Int32x4, dst unsafe.Pointer) {
	inv := c64.Sub(m)
	lo := m.MulWidenLo(s0).Add(inv.MulWidenLo(s1)).Add(round).Shift(sh6)
	hi := blendHiHalf16(m).MulWidenLo(blendHiHalf16(s0)).
		Add(blendHiHalf16(inv).MulWidenLo(blendHiHalf16(s1))).
		Add(round).Shift(sh6)
	lo.SaturateToUint16().ReshapeToUint64s().
		InterleaveLo(hi.SaturateToUint16().ReshapeToUint64s()).
		ReshapeToUint16s().StoreArray((*[8]uint16)(dst))
}

// blendHiHalf16 moves lanes four to seven of v into lanes zero to three.
func blendHiHalf16(v archsimd.Uint16x8) archsimd.Uint16x8 {
	b := v.ReshapeToUint8s()
	return b.ConcatShiftBytesRight(b, 8).ReshapeToUint16s()
}

// blendPairSums16 returns the 2x2 horizontal pair sums of the sixteen mask bytes
// at byte offset off from p: lane c holds p[2c] + p[2c+1]. Each u16 lane of the
// reshaped vector holds its byte pair little-endian, so the low byte and the
// high byte are the two horizontal neighbours. The caller's row holds 2*width
// bytes, so the sixteen bytes of every group are inside the row.
func blendPairSums16(p unsafe.Pointer, off int, lowByte archsimd.Uint16x8, sh8 archsimd.Int16x8) archsimd.Uint16x8 {
	w := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(p, off))).ReshapeToUint16s()
	return w.And(lowByte).Add(w.Shift(sh8))
}
