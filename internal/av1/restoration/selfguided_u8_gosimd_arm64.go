// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package restoration

import "simd/archsimd"

// Go-native SIMD final self-guided projection for 8-bit pixels (the dav1d
// sgr_weighted2_8bpc role with libaom's exact arithmetic). Per eight pixels:
//
//	u  = s << SGRProjRstBits                          (widen + VSSHL #4)
//	r  = roundPowerOfTwo(u*(128-xq0-xq1) + xq0*f0 + xq1*f1, 11)   (sgrProjectHalf)
//	w  = int32(int16(r))                              (VXTN: wrap to 16 bits)
//	d  = uint8(clampInt32(w, 0, 255))                 (SQXTUN: saturate to [0,255])
//
// The identity u<<7 + xq0*(f0-u) + xq1*(f1-u) == u*(128-xq0-xq1) + xq0*f0 +
// xq1*f1 holds in wrapping int32 arithmetic, so the projection is the same one the
// u16 kernel uses. The int16 wrap and the unsigned saturating narrow together equal
// clampInt32(int32(int16(r)), 0, 255) for every r.

// sgrWeightedRowU8SIMD is the Go-native SIMD form of sgrWeightedRowU8. Sixteen
// columns per iteration: one 16-byte load, two 8-lane projections, one 16-byte
// store. The trailing width%16 columns use the scalar reference.
//
// The constants are struct literals built in this function, not returned from a
// helper, so they stay register-resident across the loop; the flt rows are viewed
// as fixed-size array pointers so every lane load has a constant, in-range index.
func sgrWeightedRowU8SIMD(dst []uint8, src []uint8, f0 []int32, f1 []int32, xq0 int32, xq1 int32) {
	width := len(dst)
	src = src[:width]
	f0 = f0[:width]
	f1 = f1[:width]
	c := sgrConsts{
		xq0V:  archsimd.BroadcastInt32x4(xq0),
		xq1V:  archsimd.BroadcastInt32x4(xq1),
		bias:  archsimd.BroadcastInt32x4(1 << (SGRProjPrjBits + SGRProjRstBits - 1)),
		cuV:   archsimd.BroadcastInt32x4((1 << SGRProjPrjBits) - xq0 - xq1),
		shRst: archsimd.BroadcastInt32x4(SGRProjRstBits),
		shRnd: archsimd.BroadcastInt32x4(-(SGRProjPrjBits + SGRProjRstBits)),
	}
	col := 0
	for ; col+16 <= width; col += 16 {
		sv := archsimd.LoadUint8x16Array((*[16]uint8)(src[col:]))
		fa := (*[16]int32)(f0[col:])
		fb := (*[16]int32)(f1[col:])
		lo := sv.ExtendLo8ToUint16()
		hi := sv.HiToLo().ExtendLo8ToUint16()
		// Strip A: samples 0..7 of this group.
		a0 := sgrProjectHalf(lo.ExtendLo4ToUint32().ConvertToInt32().Shift(c.shRst),
			archsimd.LoadInt32x4Array((*[4]int32)(fa[0:4])), archsimd.LoadInt32x4Array((*[4]int32)(fb[0:4])), c)
		a1 := sgrProjectHalf(lo.HiToLo().ExtendLo4ToUint32().ConvertToInt32().Shift(c.shRst),
			archsimd.LoadInt32x4Array((*[4]int32)(fa[4:8])), archsimd.LoadInt32x4Array((*[4]int32)(fb[4:8])), c)
		outA := restorationTruncateInt32PairToInt16(a0, a1).SaturateToUint8()
		// Strip B: samples 8..15 of this group.
		b0 := sgrProjectHalf(hi.ExtendLo4ToUint32().ConvertToInt32().Shift(c.shRst),
			archsimd.LoadInt32x4Array((*[4]int32)(fa[8:12])), archsimd.LoadInt32x4Array((*[4]int32)(fb[8:12])), c)
		b1 := sgrProjectHalf(hi.HiToLo().ExtendLo4ToUint32().ConvertToInt32().Shift(c.shRst),
			archsimd.LoadInt32x4Array((*[4]int32)(fa[12:16])), archsimd.LoadInt32x4Array((*[4]int32)(fb[12:16])), c)
		outB := restorationTruncateInt32PairToInt16(b0, b1).SaturateToUint8()
		// Each strip's eight bytes sit in the low 64 bits; join them in order.
		joined := outA.ReshapeToUint64s().InterleaveLo(outB.ReshapeToUint64s()).ReshapeToUint8s()
		joined.StoreArray((*[16]uint8)(dst[col:]))
	}
	if col < width {
		sgrWeightedRowU8(dst[col:], src[col:], f0[col:], f1[col:], xq0, xq1)
	}
}
