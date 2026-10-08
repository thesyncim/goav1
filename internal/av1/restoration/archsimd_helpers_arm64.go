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

// restorationRoundShiftInt32 applies AV1's rounded arithmetic right shift
// without adding a bias to the accumulator, which could overflow int32.
func restorationRoundShiftInt32(v archsimd.Int32x4, shift uint8) archsimd.Int32x4 {
	if shift == 0 {
		return v
	}
	s := uint64(shift)
	roundBit := v.ShiftAllRight(s - 1).And(archsimd.BroadcastInt32x4(1))
	return v.ShiftAllRight(s).Add(roundBit)
}

// restorationRoundShiftNarrowInt32Pair rounds and signed-saturates two int32
// halves to int16, matching the former SQRSHRN/SQRSHRN2 pair.
func restorationRoundShiftNarrowInt32Pair(lo, hi archsimd.Int32x4, shift uint8) archsimd.Int16x8 {
	lo16 := restorationRoundShiftInt32(lo, shift).SaturateToInt16()
	hi16 := restorationRoundShiftInt32(hi, shift).SaturateToInt16()
	lo64 := lo16.ToBits().ReshapeToUint64s()
	hi64 := hi16.ToBits().ReshapeToUint64s()
	return lo64.InterleaveLo(hi64).ReshapeToUint16s().BitsToInt16()
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

// restorationSaturateInt16PairToUint8 packs two groups of eight signed values
// after unsigned saturation, matching the former SQXTUN/SQXTUN2 pair.
func restorationSaturateInt16PairToUint8(lo, hi archsimd.Int16x8) archsimd.Uint8x16 {
	lo64 := lo.SaturateToUint8().ReshapeToUint64s()
	hi64 := hi.SaturateToUint8().ReshapeToUint64s()
	return lo64.InterleaveLo(hi64).ReshapeToUint8s()
}
