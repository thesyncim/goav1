// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package motion

import "simd/archsimd"

var motionConvXTapPermutes = func() [filterTaps][16]uint8 {
	var out [filterTaps][16]uint8
	for tap := range filterTaps {
		for lane := range 8 {
			out[tap][lane] = uint8(tap + lane)
		}
	}
	return out
}()

// simdRoundShiftInt16 mirrors signed rounding shift right (SQRSHR) without
// adding a positive bias to x. The arithmetic shift plus the discarded
// half-bit also rounds negative ties toward +infinity, matching the NEON
// rounding shift while avoiding overflow at MaxInt16.
func simdRoundShiftInt16(x archsimd.Int16x8, shift uint64) archsimd.Int16x8 {
	if shift == 0 {
		return x
	}
	return x.ShiftAllRight(shift).Add(
		x.ShiftAllRight(shift - 1).And(archsimd.BroadcastInt16x8(1)),
	)
}

func simdRoundShiftInt32(x archsimd.Int32x4, shift uint64) archsimd.Int32x4 {
	if shift == 0 {
		return x
	}
	return x.ShiftAllRight(shift).Add(
		x.ShiftAllRight(shift - 1).And(archsimd.BroadcastInt32x4(1)),
	)
}

// simdConcatInt16x8 joins the low four int16 values from each narrowed int32
// vector. Narrowing instructions zero their unused upper half, so VEXT #8
// supplies the second vector's four values without a lane spill.
func simdConcatInt16x8(lo, hi archsimd.Int16x8) archsimd.Int16x8 {
	loBytes := lo.ToBits().ReshapeToUint8s()
	hiBytes := hi.ToBits().ReshapeToUint8s()
	return loBytes.Or(hiBytes.ConcatShiftBytesRight(loBytes, 8)).ReshapeToUint16s().BitsToInt16()
}

func simdConcatUint16x8(lo, hi archsimd.Uint16x8) archsimd.Uint16x8 {
	loBytes := lo.ReshapeToUint8s()
	hiBytes := hi.ReshapeToUint8s()
	return loBytes.Or(hiBytes.ConcatShiftBytesRight(loBytes, 8)).ReshapeToUint16s()
}

// simdConcatUint8x16 joins the low eight bytes from each narrowing result.
func simdConcatUint8x16(lo, hi archsimd.Uint8x16) archsimd.Uint8x16 {
	return lo.Or(hi.ConcatShiftBytesRight(lo, 8))
}

func simdRoundShiftNarrowInt32x4(x archsimd.Int32x4, shift uint64) archsimd.Int16x8 {
	return simdRoundShiftInt32(x, shift).SaturateToInt16()
}

func simdRoundShiftNarrowInt32Pair(lo, hi archsimd.Int32x4, shift uint64) archsimd.Int16x8 {
	return simdConcatInt16x8(simdRoundShiftNarrowInt32x4(lo, shift), simdRoundShiftNarrowInt32x4(hi, shift))
}

func simdRoundShiftNarrowUint8(x archsimd.Int16x8, shift uint64) archsimd.Uint8x16 {
	return simdRoundShiftInt16(x, shift).SaturateToUint8()
}

func simdShiftSaturateNarrowUint16(x archsimd.Int32x4, shift uint64) archsimd.Uint16x8 {
	return x.ShiftAllRight(shift).SaturateToUint16()
}

func simdShiftSaturateNarrowUint16Pair(lo, hi archsimd.Int32x4, shift uint64) archsimd.Uint16x8 {
	return simdConcatUint16x8(simdShiftSaturateNarrowUint16(lo, shift), simdShiftSaturateNarrowUint16(hi, shift))
}

func simdMulWidenAddInt16Lo(acc archsimd.Int32x4, x, y archsimd.Int16x8) archsimd.Int32x4 {
	return acc.Add(x.MulWidenLo(y))
}

func simdMulWidenAddInt16Hi(acc archsimd.Int32x4, x, y archsimd.Int16x8) archsimd.Int32x4 {
	return acc.Add(x.HiToLo().MulWidenLo(y.HiToLo()))
}

func simdMulWidenAddUint16Lo(acc archsimd.Uint32x4, x, y archsimd.Uint16x8) archsimd.Uint32x4 {
	return acc.Add(x.MulWidenLo(y))
}

func simdMulWidenAddUint16Hi(acc archsimd.Uint32x4, x, y archsimd.Uint16x8) archsimd.Uint32x4 {
	return acc.Add(x.HiToLo().MulWidenLo(y.HiToLo()))
}

// simdHorizontalConvAcc computes eight adjacent unsigned-byte by signed-int16
// FIR outputs, keeping each four-lane half in int32 until the caller applies
// the codec's specified rounding stage.
func simdHorizontalConvAcc(raw archsimd.Uint8x16, kernel [filterTaps]int16, bias int32) (archsimd.Int32x4, archsimd.Int32x4) {
	lo := archsimd.BroadcastInt32x4(bias)
	hi := archsimd.BroadcastInt32x4(bias)
	for tap := range filterTaps {
		samples := raw.LookupOrZero(archsimd.LoadUint8x16Array(&motionConvXTapPermutes[tap])).
			ExtendLo8ToUint16().ConvertToInt16()
		coeff := archsimd.BroadcastInt16x8(kernel[tap])
		lo = lo.Add(samples.MulWidenLo(coeff))
		hi = hi.Add(samples.HiToLo().MulWidenLo(coeff.HiToLo()))
	}
	return lo, hi
}

func simdKernelFromI8MMFilter(filter [16]byte, f0 uint8) [filterTaps]int16 {
	var kernel [filterTaps]int16
	kernel[0] = -2 * int16(f0)
	for tap := 1; tap < filterTaps; tap++ {
		kernel[tap] = 2 * int16(int8(filter[tap-1]))
	}
	return kernel
}

// simdDotProdUS composes a four-lane unsigned-byte by signed-byte dot product
// using NEON widening and pairwise-add operations. Each output lane sums one
// consecutive group of four bytes, exactly the USDOT lane contract.
func simdDotProdUS(acc archsimd.Int32x4, samples archsimd.Uint8x16, taps archsimd.Int8x16) archsimd.Int32x4 {
	low := simdDotProdUS8(samples, taps)
	high := simdDotProdUS8(samples.HiToLo(), taps.HiToLo())
	return acc.Add(low.ConcatEven(high))
}

// simdDotProdUS8 returns two four-byte dot products duplicated into adjacent
// lanes; callers use the low/even lanes when joining the low and high halves.
func simdDotProdUS8(samples archsimd.Uint8x16, taps archsimd.Int8x16) archsimd.Int32x4 {
	s := samples.ExtendLo8ToUint16().ConvertToInt16()
	t := taps.ExtendLo8ToInt16()
	products := s.Mul(t)
	prodLo := products.ExtendLo4ToInt32()
	prodHi := products.HiToLo().ExtendLo4ToInt32()
	pairLo := prodLo.ConcatAddPairs(prodLo)
	pairHi := prodHi.ConcatAddPairs(prodHi)
	return pairLo.ConcatAddPairs(pairHi)
}
