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
// decoder-generated alpha (0..64), so the scalar's per-pixel range checks never
// fire — the SIMD path assumes those preconditions and returns true.
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
			mLo := mv.ExtendLo8ToUint16().ConvertToInt16()
			mHi := mv.ConcatShiftBytesRight(mv, 8).ExtendLo8ToUint16().ConvertToInt16()
			s0Lo := archsimd.LoadUint16x8Array((*[8]uint16)(s0Row[col:])).ConvertToInt16()
			s0Hi := archsimd.LoadUint16x8Array((*[8]uint16)(s0Row[col+8:])).ConvertToInt16()
			s1Lo := archsimd.LoadUint16x8Array((*[8]uint16)(s1Row[col:])).ConvertToInt16()
			s1Hi := archsimd.LoadUint16x8Array((*[8]uint16)(s1Row[col+8:])).ConvertToInt16()
			blendHalf(mLo, s0Lo, s1Lo).StoreArray((*[8]uint16)(dstRow[col:]))
			blendHalf(mHi, s0Hi, s1Hi).StoreArray((*[8]uint16)(dstRow[col+8:]))
		}
	}
	return true
}
