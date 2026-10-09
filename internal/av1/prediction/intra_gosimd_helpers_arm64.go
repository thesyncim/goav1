// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package prediction

import "simd/archsimd"

// Arm64 implementations of the operations the shared intra kernels need. NEON
// has single instructions for each of these, so the shared kernels map onto the
// native forms directly.

// Right-shift amounts as per-lane shift vectors. The arm64 toolchain lowers
// ShiftAllRight(const) to a per-use DUP of the negated amount, whereas a per-lane
// Shift against a hoisted vector is a single VUSHL/VSSHL.
var (
	shr1U16  = archsimd.BroadcastInt16x8(-1)
	shr5U16  = archsimd.BroadcastInt16x8(-5)
	shr8U16  = archsimd.BroadcastInt16x8(-smoothShift1D)
	shr6I32  = archsimd.BroadcastInt32x4(-6)
	shr31I32 = archsimd.BroadcastInt32x4(-31)
	shr8I32  = archsimd.BroadcastInt32x4(-smoothShift1D)
	shr9I32  = archsimd.BroadcastInt32x4(-smoothShiftFull)
)

// narrowU16ToU8 narrows eight uint16 lanes already in [0,255] to bytes (UQXTN).
func narrowU16ToU8(v archsimd.Uint16x8) archsimd.Uint8x16 {
	return v.SaturateToUint8()
}

// hiToLoU16 moves the upper four lanes of v into the lower four.
func hiToLoU16(v archsimd.Uint16x8) archsimd.Uint16x8 {
	return v.HiToLo()
}

// hiToLoI16 moves the upper four lanes of v into the lower four.
func hiToLoI16(v archsimd.Int16x8) archsimd.Int16x8 {
	return v.HiToLo()
}

// mulWidenLo16 multiplies the low four lanes of a and b into int32 lanes.
func mulWidenLo16(a, b archsimd.Int16x8) archsimd.Int32x4 {
	return a.MulWidenLo(b)
}

// packInt32PairToInt16 packs eight int32 values (lo's four lanes, then hi's) into
// int16 lanes, truncating; callers guarantee the values fit.
func packInt32PairToInt16(lo, hi archsimd.Int32x4) archsimd.Int16x8 {
	return cflTruncateInt32PairToInt16(lo, hi)
}

// reduceSumU32 returns the sum of the four uint32 lanes.
func reduceSumU32(v archsimd.Uint32x4) uint32 {
	return v.ReduceSum()
}

// shrU16_1, shrU16_5 and shrU16_8 are logical right shifts of uint16 lanes.
func shrU16_1(v archsimd.Uint16x8) archsimd.Uint16x8 { return v.Shift(shr1U16) }
func shrU16_5(v archsimd.Uint16x8) archsimd.Uint16x8 { return v.Shift(shr5U16) }
func shrU16_8(v archsimd.Uint16x8) archsimd.Uint16x8 { return v.Shift(shr8U16) }

// shrI32_6 and shrI32_31 are arithmetic right shifts of int32 lanes.
func shrI32_6(v archsimd.Int32x4) archsimd.Int32x4  { return v.Shift(shr6I32) }
func shrI32_31(v archsimd.Int32x4) archsimd.Int32x4 { return v.Shift(shr31I32) }
