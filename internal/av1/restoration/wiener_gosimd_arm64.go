// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package restoration

import (
	"simd/archsimd"
	"unsafe"
)

// Go-native SIMD uint16 Wiener vertical pass for arm64. The seven vertical taps
// are a plain 7-tap widening multiply-accumulate (VSMULL/VSMULL2 + VADD) into two
// int32 lane groups (columns 0..3 and 4..7), then the accumulator is round-shifted,
// clamped to [0,max] and narrowed back to uint16.
//
// Two folds make the inner loop a plain 7-tap MAC: the libaom center reapplication
// s3<<WienerFilterBits is folded into the center tap (tap3 += 1<<WienerFilterBits),
// and the rounding bias 1<<(round1-1) together with -offset is folded into the
// accumulator seed so the trailing arithmetic right shift reproduces
// roundPowerOfTwo(sum, round1) exactly.
//
// Every temp sample is the horizontal pass's output, clamped to [0,maxClamp] with
// maxClamp <= 32767 for all bit depths. A value in that range reinterprets as
// non-negative int16, so int16 widening multiplication reproduces the reference's
// product exactly, and the 7-tap int32 sum never overflows.
//
// Each output row reloads its seven temp rows, as the hand asm does. Memory is
// addressed per load with a bounds-checked slice index so no pointer ever leaves
// its allocation. Widths that are not a multiple of 8 use the scalar reference.
func wienerVerticalSIMD(temp []uint16, tempStride int, dst []uint16, dstStride int, width int, height int, filter WienerFilter, bitDepth int, round1 int, max uint16) {
	if width < 8 || width%8 != 0 {
		wienerVertical(temp, tempStride, dst, dstStride, width, height, filter, bitDepth, round1, max)
		return
	}
	offset := int32(1 << (bitDepth + round1 - 1))
	// seed folds -offset and the rounding bias 1<<(round1-1) so the trailing
	// arithmetic shift reproduces roundPowerOfTwo(sum, round1) exactly.
	seedV := archsimd.BroadcastInt32x4(-offset + roundBias(round1))
	maxV := archsimd.BroadcastUint16x8(max)
	// Per-lane arithmetic right shift by round1 as a broadcast count vector
	// (VSSHL, negative => shift right); round1 is a runtime value.
	negShiftV := archsimd.BroadcastInt32x4(int32(-round1))
	f0V := archsimd.BroadcastInt16x8(filter[0])
	f1V := archsimd.BroadcastInt16x8(filter[1])
	f2V := archsimd.BroadcastInt16x8(filter[2])
	// The center tap absorbs the s3<<WienerFilterBits center reapplication.
	f3V := archsimd.BroadcastInt16x8(filter[3] + (1 << WienerFilterBits))
	f4V := archsimd.BroadcastInt16x8(filter[4])
	f5V := archsimd.BroadcastInt16x8(filter[5])
	f6V := archsimd.BroadcastInt16x8(filter[6])
	hf0V, hf1V := f0V.HiToLo(), f1V.HiToLo()
	hf2V, hf3V := f2V.HiToLo(), f3V.HiToLo()
	hf4V, hf5V := f4V.HiToLo(), f5V.HiToLo()
	hf6V := f6V.HiToLo()

	for row := 0; row < height; row++ {
		for col := 0; col < width; col += 8 {
			t := row*tempStride + col
			r0 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Pointer(&temp[t])))
			r1 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Pointer(&temp[t+1*tempStride])))
			r2 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Pointer(&temp[t+2*tempStride])))
			r3 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Pointer(&temp[t+3*tempStride])))
			r4 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Pointer(&temp[t+4*tempStride])))
			r5 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Pointer(&temp[t+5*tempStride])))
			r6 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Pointer(&temp[t+6*tempStride])))
			lo := seedV.Add(r0.MulWidenLo(f0V)).Add(r1.MulWidenLo(f1V)).
				Add(r2.MulWidenLo(f2V)).Add(r3.MulWidenLo(f3V)).
				Add(r4.MulWidenLo(f4V)).Add(r5.MulWidenLo(f5V)).
				Add(r6.MulWidenLo(f6V))
			hi := seedV.Add(r0.HiToLo().MulWidenLo(hf0V)).Add(r1.HiToLo().MulWidenLo(hf1V)).
				Add(r2.HiToLo().MulWidenLo(hf2V)).Add(r3.HiToLo().MulWidenLo(hf3V)).
				Add(r4.HiToLo().MulWidenLo(hf4V)).Add(r5.HiToLo().MulWidenLo(hf5V)).
				Add(r6.HiToLo().MulWidenLo(hf6V))
			lo = lo.Shift(negShiftV)
			hi = hi.Shift(negShiftV)
			out := restorationSaturateInt32PairToUint16(lo, hi).Min(maxV)
			out.StoreArray((*[8]uint16)(unsafe.Pointer(&dst[row*dstStride+col])))
		}
	}
}
