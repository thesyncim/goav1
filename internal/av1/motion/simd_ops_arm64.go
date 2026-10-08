// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package motion

import (
	"simd/archsimd"
	"unsafe"
)

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

// simdHorizontalConvMAC adds one tap's unsigned-byte by signed-int16 products
// to the low and high four output lanes. Coefficients are broadcast once per
// kernel outside the pixel loops, and callers use literal VEXT shifts for the
// sample windows so the inner loop contains no tap loop or lookup table loads.
func simdHorizontalConvMAC(lo, hi archsimd.Int32x4, samples, coeff archsimd.Int16x8) (archsimd.Int32x4, archsimd.Int32x4) {
	return lo.Add(samples.MulWidenLo(coeff)), hi.Add(samples.HiToLo().MulWidenLo(coeff))
}

// simdNarrowHorizontalKernel allows a faster 16-bit multiply-accumulate for
// filters whose half-coefficients have an absolute sum no greater than 128.
// For byte samples, every partial sum is then bounded by 255*128 = 32640, so
// Int16x8.MulAdd cannot wrap. Filters outside this bound keep the wide path.
func simdNarrowHorizontalKernel(kernel [filterTaps]int16) (half [filterTaps]int16, ok bool) {
	var sumAbs int32
	for i, coeff := range kernel {
		if coeff%2 != 0 {
			return [filterTaps]int16{}, false
		}
		h := coeff / 2
		half[i] = h
		if h < 0 {
			sumAbs -= int32(h)
		} else {
			sumAbs += int32(h)
		}
	}
	return half, sumAbs <= 128
}

// simdHorizontalNarrowToIM computes the horizontal byte-to-int16 pass for a
// filter accepted by simdNarrowHorizontalKernel. The int16 MAC bound is checked
// once per kernel. Since full coefficients are twice the half coefficients,
// roundPowerOfTwo(2*h+xBias,3) is roundPowerOfTwo(h,2)+xBias/8. The int16
// bound proves both the round and post-round bias fit without saturation.
func simdHorizontalNarrowToIM(src, dst unsafe.Pointer, width, height, srcStride, dstStride int, kernel [filterTaps]int16, xBias int32) {
	k0 := archsimd.BroadcastInt16x8(kernel[0])
	k1 := archsimd.BroadcastInt16x8(kernel[1])
	k2 := archsimd.BroadcastInt16x8(kernel[2])
	k3 := archsimd.BroadcastInt16x8(kernel[3])
	k4 := archsimd.BroadcastInt16x8(kernel[4])
	k5 := archsimd.BroadcastInt16x8(kernel[5])
	k6 := archsimd.BroadcastInt16x8(kernel[6])
	k7 := archsimd.BroadcastInt16x8(kernel[7])
	zero := archsimd.BroadcastInt16x8(0)
	xBiasHalf := archsimd.BroadcastInt16x8(int16(xBias >> round0Bits))
	roundBias := archsimd.BroadcastInt16x8(2)
	roundShift := archsimd.BroadcastInt16x8(-2)

	for y := 0; y < height; y++ {
		sp := unsafe.Add(src, y*srcStride)
		ip := unsafe.Add(dst, y*dstStride*2)
		for col := 0; col < width; col += 8 {
			raw := archsimd.LoadUint8x16Array((*[16]uint8)(sp))
			even := raw.ExtendLo8ToUint16().ConvertToInt16().MulAdd(k0, zero)
			odd := raw.ConcatShiftBytesRight(raw, 1).ExtendLo8ToUint16().ConvertToInt16().MulAdd(k1, zero)
			even = raw.ConcatShiftBytesRight(raw, 2).ExtendLo8ToUint16().ConvertToInt16().MulAdd(k2, even)
			odd = raw.ConcatShiftBytesRight(raw, 3).ExtendLo8ToUint16().ConvertToInt16().MulAdd(k3, odd)
			even = raw.ConcatShiftBytesRight(raw, 4).ExtendLo8ToUint16().ConvertToInt16().MulAdd(k4, even)
			odd = raw.ConcatShiftBytesRight(raw, 5).ExtendLo8ToUint16().ConvertToInt16().MulAdd(k5, odd)
			even = raw.ConcatShiftBytesRight(raw, 6).ExtendLo8ToUint16().ConvertToInt16().MulAdd(k6, even)
			odd = raw.ConcatShiftBytesRight(raw, 7).ExtendLo8ToUint16().ConvertToInt16().MulAdd(k7, odd)
			// fullSum=2*halfSum, so roundPowerOfTwo(fullSum+xBias, 3)
			// equals roundPowerOfTwo(halfSum, 2)+xBias/8. The coefficient
			// bound keeps that intermediate within int16, so saturation is inert.
			// The guarded half-sum has two spare units of int16 headroom, so
			// add the rounding bias before one precomputed arithmetic shift.
			outIM := even.Add(odd).Add(roundBias).Shift(roundShift).Add(xBiasHalf)
			outIM.StoreArray((*[8]int16)(ip))

			if col+8 < width {
				sp = unsafe.Add(sp, 8)
				ip = unsafe.Add(ip, 8*2)
			}
		}
	}
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
