// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && amd64 && !purego

package prediction

import "simd/archsimd"

// AVX2 implementations of the operations the shared intra kernels need. Every
// operation here lowers to a VEX (AVX/AVX2) instruction: the narrowing VPMOV*
// and VPERMB forms are EVEX-only and would SIGILL on AVX2-only CPUs, so bytes
// are gathered with VPSHUFB and 32-bit values are packed with VPACKSSDW.

// evenBytesIdx selects the even bytes of a vector into its low eight lanes; the
// high lanes are zeroed (index MSB set).
var evenBytesIdx = archsimd.LoadInt8x16Array(&[16]int8{0, 2, 4, 6, 8, 10, 12, 14, -128, -128, -128, -128, -128, -128, -128, -128})

// narrowU16ToU8 narrows eight uint16 lanes already in [0,255] to the low eight
// bytes of the result.
func narrowU16ToU8(v archsimd.Uint16x8) archsimd.Uint8x16 {
	return v.AsUint8x16().PermuteOrZero(evenBytesIdx)
}

// hiToLoU16 moves the upper four lanes of v into the lower four (VPALIGNR).
func hiToLoU16(v archsimd.Uint16x8) archsimd.Uint16x8 {
	b := v.AsUint8x16()
	return b.ConcatShiftBytesRight(b, 8).AsUint16x8()
}

// hiToLoI16 moves the upper four lanes of v into the lower four.
func hiToLoI16(v archsimd.Int16x8) archsimd.Int16x8 {
	return hiToLoU16(v.ConvertToUint16()).ConvertToInt16()
}

// mulWidenLo16 multiplies the low four lanes of a and b into int32 lanes.
func mulWidenLo16(a, b archsimd.Int16x8) archsimd.Int32x4 {
	return a.ExtendLo4ToInt32().Mul(b.ExtendLo4ToInt32())
}

// packInt32PairToInt16 packs eight int32 values (lo's four lanes, then hi's) into
// int16 lanes with the AVX2 signed pack; callers guarantee the values fit.
func packInt32PairToInt16(lo, hi archsimd.Int32x4) archsimd.Int16x8 {
	return lo.SaturateToInt16Concat(hi)
}

// reduceSumU32 returns the sum of the four uint32 lanes.
func reduceSumU32(v archsimd.Uint32x4) uint32 {
	return v.GetElem(0) + v.GetElem(1) + v.GetElem(2) + v.GetElem(3)
}

// shrU16_1, shrU16_5 and shrU16_8 are logical right shifts of uint16 lanes (VPSRLW
// with an immediate).
func shrU16_1(v archsimd.Uint16x8) archsimd.Uint16x8 { return v.ShiftAllRight(1) }
func shrU16_5(v archsimd.Uint16x8) archsimd.Uint16x8 { return v.ShiftAllRight(5) }
func shrU16_8(v archsimd.Uint16x8) archsimd.Uint16x8 { return v.ShiftAllRight(8) }

// shrI32_6 and shrI32_31 are arithmetic right shifts of int32 lanes (VPSRAD).
func shrI32_6(v archsimd.Int32x4) archsimd.Int32x4  { return v.ShiftAllRight(6) }
func shrI32_31(v archsimd.Int32x4) archsimd.Int32x4 { return v.ShiftAllRight(31) }
