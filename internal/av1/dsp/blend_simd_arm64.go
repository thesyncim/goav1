// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package dsp

import "simd/archsimd"

// blendA64MaskSIMD is the Go-native-SIMD analogue of blendA64MaskPureGo for the
// common 8-bit, non-subsampled mask path (subX==subY==false), width a multiple
// of 16. Every other case (subsampled mask, high bit depth) falls back to the
// scalar reference, which also performs the range validation. On the fast path
// src0/src1 are 8-bit predictions (<= max == 255) and the mask is a valid
// decoder-generated alpha (0..64), so the scalar's per-pixel range checks do
// not fire for valid input. Each chunk is checked before it is written;
// malformed chunks finish in scalar lane order before returning false.
//
// blendA64(m, s0, s1) = (m*s0 + (64-m)*s1 + 32) >> 6. For 8-bit inputs every
// intermediate fits in int16 (max m*s0 = 64*255 = 16320), so the whole blend is
// exact in the int16 lane domain — byte-identical to the scalar reference.
func blendA64MaskSIMD(a blendA64MaskArgs) bool {
	if a.max != 255 || a.subX || a.subY || a.width < 16 || a.width%16 != 0 {
		return blendA64MaskPureGo(a)
	}
	c64 := archsimd.BroadcastInt16x8(64)
	c32 := archsimd.BroadcastInt16x8(32)
	maxAlpha := archsimd.BroadcastUint8x16(blendA64MaxAlpha)
	maxSample := archsimd.BroadcastUint16x8(255)
	blendHalf := func(m, s0, s1 archsimd.Int16x8) archsimd.Uint16x8 {
		inv := c64.Sub(m)
		return m.Mul(s0).Add(inv.Mul(s1)).Add(c32).ShiftAllRight(6).ConvertToUint16()
	}
	for row := 0; row < a.height; row++ {
		maskRow := a.mask[row*a.maskStride:]
		s0Row := a.src0[row*a.src0Stride:]
		s1Row := a.src1[row*a.src1Stride:]
		dstRow := a.dst[row*a.dstStride:]
		for col := 0; col < a.width; col += 16 {
			mv := archsimd.LoadUint8x16Array((*[16]uint8)(maskRow[col:]))
			s0Lo16 := archsimd.LoadUint16x8Array((*[8]uint16)(s0Row[col:]))
			s0Hi16 := archsimd.LoadUint16x8Array((*[8]uint16)(s0Row[col+8:]))
			s1Lo16 := archsimd.LoadUint16x8Array((*[8]uint16)(s1Row[col:]))
			s1Hi16 := archsimd.LoadUint16x8Array((*[8]uint16)(s1Row[col+8:]))
			if mv.Greater(maxAlpha).ToInt8x16().ToBits().ReduceSum() != 0 ||
				s0Lo16.Greater(maxSample).ToInt16x8().ToBits().ReduceSum() != 0 ||
				s0Hi16.Greater(maxSample).ToInt16x8().ToBits().ReduceSum() != 0 ||
				s1Lo16.Greater(maxSample).ToInt16x8().ToBits().ReduceSum() != 0 ||
				s1Hi16.Greater(maxSample).ToInt16x8().ToBits().ReduceSum() != 0 {
				// Complete this chunk in scalar lane order. Earlier chunks have
				// already been written exactly once, which matters when dst aliases
				// either source and when invalid input follows valid samples.
				for lane := 0; lane < 16; lane++ {
					i := col + lane
					s0, s1 := s0Row[i], s1Row[i]
					if s0 > a.max || s1 > a.max || maskRow[i] > blendA64MaxAlpha {
						return false
					}
					dstRow[i] = blendA64(maskRow[i], s0, s1)
				}
				// The vector check found an invalid lane, so the scalar loop above
				// must have returned false.
				return false
			}
			mLo := mv.ExtendLo8ToUint16().ConvertToInt16()
			mHi := mv.ConcatShiftBytesRight(mv, 8).ExtendLo8ToUint16().ConvertToInt16()
			s0Lo := s0Lo16.ConvertToInt16()
			s0Hi := s0Hi16.ConvertToInt16()
			s1Lo := s1Lo16.ConvertToInt16()
			s1Hi := s1Hi16.ConvertToInt16()
			blendHalf(mLo, s0Lo, s1Lo).StoreArray((*[8]uint16)(dstRow[col:]))
			blendHalf(mHi, s0Hi, s1Hi).StoreArray((*[8]uint16)(dstRow[col+8:]))
		}
	}
	return true
}
