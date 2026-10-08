// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package restoration

import "simd/archsimd"

// restorationTruncateInt32PairToInt16 packs the low four lanes from each
// int32 vector into one int16 vector, matching the former XTN/XTN2 pair.
func restorationTruncateInt32PairToInt16(lo, hi archsimd.Int32x4) archsimd.Int16x8 {
	lo64 := lo.TruncToInt16().ToBits().ReshapeToUint64s()
	hi64 := hi.TruncToInt16().ToBits().ReshapeToUint64s()
	return lo64.InterleaveLo(hi64).ReshapeToUint16s().BitsToInt16()
}

// restorationSaturateInt32PairToUint16 packs two groups of four signed values
// after unsigned saturation, matching the former SQXTUN/SQXTUN2 pair.
func restorationSaturateInt32PairToUint16(lo, hi archsimd.Int32x4) archsimd.Uint16x8 {
	lo64 := lo.SaturateToUint16().ReshapeToUint64s()
	hi64 := hi.SaturateToUint16().ReshapeToUint64s()
	return lo64.InterleaveLo(hi64).ReshapeToUint16s()
}

// restorationShiftRightSaturateInt32PairToUint16 arithmetic-shifts two int32
// halves, clamps negatives to zero, then packs with unsigned saturation. It
// replaces SQSHRUN/SQSHRUN2 with the equivalent official vector operations.
func restorationShiftRightSaturateInt32PairToUint16(lo, hi archsimd.Int32x4, shift uint8) archsimd.Uint16x8 {
	zero := archsimd.BroadcastInt32x4(0)
	lo = lo.ShiftAllRight(uint64(shift)).Max(zero)
	hi = hi.ShiftAllRight(uint64(shift)).Max(zero)
	return restorationSaturateInt32PairToUint16(lo, hi)
}
