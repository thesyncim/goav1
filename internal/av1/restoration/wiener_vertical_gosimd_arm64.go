// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package restoration

import (
	"encoding/binary"
	"simd/archsimd"
	"unsafe"
)

// wienerVerticalU8SIMD is the Go-native SIMD form of the 8-bit Wiener vertical
// pass over the horizontal-pass int16 intermediate, rounded by round1 and clamped
// to uint8. Symmetric taps combine the three outer row pairs in int16, reducing
// the column MAC to four widening products per half. The center-tap DC boost folds
// into f3' = f3 + (1<<WienerFilterBits).
//
// temp is bounded to [0,8191] (< 2^15), so its uint16 cells reinterpret losslessly
// as int16 for widening multiplication and each pair sum is at most 16382. The seed
// folds -offset and the rounding bias, so one arithmetic right shift reproduces
// roundPowerOfTwo, and int16 then uint8 saturation is exactly clampInt32(x, 0, 255).
//
// Sixteen columns per row: two 8-lane strips, one 16-byte store, as the hand asm
// does; a trailing 8-column strip handles width%16 == 8. Each output row reloads
// its seven temp rows, and every access is a bounds-checked slice index.
//
// Symmetric filters only: asymmetric filters take the scalar reference.
func wienerVerticalU8SIMD(temp []uint16, tempStride int, dst []uint8, dstStride int, width int, height int, filter WienerFilter, round1 int) {
	if width < 8 || width%8 != 0 || round1 < 1 || round1 > 16 || !wienerFilterSymmetric(filter) {
		wienerVerticalU8(temp, tempStride, dst, dstStride, width, height, filter, round1)
		return
	}
	offset := int32(1) << (8 + uint(round1) - 1)
	biasV := archsimd.BroadcastInt32x4(-offset + roundBias(round1))
	negShiftV := archsimd.BroadcastInt32x4(-int32(round1))
	f0 := archsimd.BroadcastInt16x8(filter[0])
	f1 := archsimd.BroadcastInt16x8(filter[1])
	f2 := archsimd.BroadcastInt16x8(filter[2])
	f3 := archsimd.BroadcastInt16x8(filter[3] + (1 << WienerFilterBits))
	hf0, hf1 := f0.HiToLo(), f1.HiToLo()
	hf2, hf3 := f2.HiToLo(), f3.HiToLo()

	col := 0
	for ; col+16 <= width; col += 16 {
		for row := 0; row < height; row++ {
			ta := row*tempStride + col
			a0 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Pointer(&temp[ta])))
			a1 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Pointer(&temp[ta+1*tempStride])))
			a2 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Pointer(&temp[ta+2*tempStride])))
			a3 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Pointer(&temp[ta+3*tempStride])))
			a4 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Pointer(&temp[ta+4*tempStride])))
			a5 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Pointer(&temp[ta+5*tempStride])))
			a6 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Pointer(&temp[ta+6*tempStride])))
			pa0, pa1, pa2 := a0.Add(a6), a1.Add(a5), a2.Add(a4)
			loA := biasV.Add(pa0.MulWidenLo(f0)).Add(pa1.MulWidenLo(f1)).
				Add(pa2.MulWidenLo(f2)).Add(a3.MulWidenLo(f3))
			hiA := biasV.Add(pa0.HiToLo().MulWidenLo(hf0)).Add(pa1.HiToLo().MulWidenLo(hf1)).
				Add(pa2.HiToLo().MulWidenLo(hf2)).Add(a3.HiToLo().MulWidenLo(hf3))
			outA := restorationSaturateInt32PairToUint16(loA.Shift(negShiftV), hiA.Shift(negShiftV)).SaturateToUint8()

			tb := ta + 8
			b0 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Pointer(&temp[tb])))
			b1 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Pointer(&temp[tb+1*tempStride])))
			b2 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Pointer(&temp[tb+2*tempStride])))
			b3 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Pointer(&temp[tb+3*tempStride])))
			b4 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Pointer(&temp[tb+4*tempStride])))
			b5 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Pointer(&temp[tb+5*tempStride])))
			b6 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Pointer(&temp[tb+6*tempStride])))
			pb0, pb1, pb2 := b0.Add(b6), b1.Add(b5), b2.Add(b4)
			loB := biasV.Add(pb0.MulWidenLo(f0)).Add(pb1.MulWidenLo(f1)).
				Add(pb2.MulWidenLo(f2)).Add(b3.MulWidenLo(f3))
			hiB := biasV.Add(pb0.HiToLo().MulWidenLo(hf0)).Add(pb1.HiToLo().MulWidenLo(hf1)).
				Add(pb2.HiToLo().MulWidenLo(hf2)).Add(b3.HiToLo().MulWidenLo(hf3))
			outB := restorationSaturateInt32PairToUint16(loB.Shift(negShiftV), hiB.Shift(negShiftV)).SaturateToUint8()

			// Each strip's eight bytes sit in the low 64 bits; join them in order.
			joined := outA.ReshapeToUint64s().InterleaveLo(outB.ReshapeToUint64s()).ReshapeToUint8s()
			joined.StoreArray((*[16]uint8)(unsafe.Pointer(&dst[row*dstStride+col])))
		}
	}
	// Trailing 8-column strip when width%16 == 8.
	for ; col < width; col += 8 {
		for row := 0; row < height; row++ {
			t := row*tempStride + col
			r0 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Pointer(&temp[t])))
			r1 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Pointer(&temp[t+1*tempStride])))
			r2 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Pointer(&temp[t+2*tempStride])))
			r3 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Pointer(&temp[t+3*tempStride])))
			r4 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Pointer(&temp[t+4*tempStride])))
			r5 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Pointer(&temp[t+5*tempStride])))
			r6 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Pointer(&temp[t+6*tempStride])))
			pair0, pair1, pair2 := r0.Add(r6), r1.Add(r5), r2.Add(r4)
			lo := biasV.Add(pair0.MulWidenLo(f0)).Add(pair1.MulWidenLo(f1)).
				Add(pair2.MulWidenLo(f2)).Add(r3.MulWidenLo(f3))
			hi := biasV.Add(pair0.HiToLo().MulWidenLo(hf0)).Add(pair1.HiToLo().MulWidenLo(hf1)).
				Add(pair2.HiToLo().MulWidenLo(hf2)).Add(r3.HiToLo().MulWidenLo(hf3))
			out := restorationSaturateInt32PairToUint16(lo.Shift(negShiftV), hi.Shift(negShiftV)).SaturateToUint8()
			o := row*dstStride + col
			binary.LittleEndian.PutUint64(dst[o:o+8], out.ReshapeToUint64s().GetElem(0))
		}
	}
}
