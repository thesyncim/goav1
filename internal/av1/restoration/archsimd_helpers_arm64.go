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

// restorationRoundShiftNarrowInt32Pair rounds and signed-saturates two int32
// halves to int16. The two shift vectors and one mask are broadcast once by the
// caller, outside the row loop.
func restorationRoundShiftNarrowInt32Pair(lo, hi, shiftV, roundBitShiftV, oneV archsimd.Int32x4) archsimd.Int16x8 {
	lo16 := lo.Shift(shiftV).Add(lo.Shift(roundBitShiftV).And(oneV)).SaturateToInt16()
	hi16 := hi.Shift(shiftV).Add(hi.Shift(roundBitShiftV).And(oneV)).SaturateToInt16()
	lo64 := lo16.ToBits().ReshapeToUint64s()
	hi64 := hi16.ToBits().ReshapeToUint64s()
	return lo64.InterleaveLo(hi64).ReshapeToUint16s().BitsToInt16()
}

// restorationSaturateInt16PairToUint8 packs two groups of eight signed values
// after unsigned saturation.
func restorationSaturateInt16PairToUint8(lo, hi archsimd.Int16x8) archsimd.Uint8x16 {
	lo64 := lo.SaturateToUint8().ReshapeToUint64s()
	hi64 := hi.SaturateToUint8().ReshapeToUint64s()
	return lo64.InterleaveLo(hi64).ReshapeToUint8s()
}
