// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package restoration

import (
	"simd/archsimd"
	"unsafe"
)

// wienerVerticalU8SIMD is the Go-native SIMD form of the 8-bit Wiener vertical
// pass over the horizontal-pass int16 intermediate, rounded by round1 and
// clamped to uint8. Symmetric taps combine the three outer row pairs in int16,
// reducing the column MAC to four widening products. The center-tap DC boost
// folds into f3' = f3 + (1<<WienerFilterBits).
//
// Column-strip walk with a SLIDING 7-row register window: each temp row is loaded
// once and reused across the 7 output rows that read it (the hand asm's trick),
// instead of reloading all 7 rows per output row. temp is bounded to [0,8191]
// (< 2^15), so its uint16 cells reinterpret losslessly as int16 for widening
// multiplication and each pair sum is at most 16382. The int32 accumulator's
// overflow-safe rounded shift and uint8 saturation reproduce the scalar result.
func wienerVerticalU8SIMD(temp []uint16, tempStride int, dst []uint8, dstStride int, width int, height int, filter WienerFilter, round1 int) {
	if width < 8 || round1 < 1 || round1 > 16 ||
		filter[0] != filter[6] || filter[1] != filter[5] || filter[2] != filter[4] {
		wienerVerticalU8(temp, tempStride, dst, dstStride, width, height, filter, round1)
		return
	}
	offset := int32(1) << (8 + uint(round1) - 1)
	biasV := archsimd.BroadcastInt32x4(-offset)
	f0 := archsimd.BroadcastInt16x8(filter[0])
	f1 := archsimd.BroadcastInt16x8(filter[1])
	f2 := archsimd.BroadcastInt16x8(filter[2])
	f3 := archsimd.BroadcastInt16x8(filter[3] + (1 << WienerFilterBits))
	hf0, hf1 := f0.HiToLo(), f1.HiToLo()
	hf2, hf3 := f2.HiToLo(), f3.HiToLo()
	roundShiftV := archsimd.BroadcastInt32x4(-int32(round1))
	roundBitShiftV := archsimd.BroadcastInt32x4(-int32(round1 - 1))
	roundOneV := archsimd.BroadcastInt32x4(1)

	const u16 = 2
	w16 := width &^ 15
	w8 := width &^ 7
	tp := unsafe.Pointer(&temp[0])
	dp := unsafe.Pointer(&dst[0])
	// Main 16-wide sliding-window path: one 16-byte store per (row, strip), each
	// temp row loaded once and reused across its 7 output rows.
	for col := 0; col < w16; col += 16 {
		rowa := unsafe.Add(tp, col*u16)
		rowb := unsafe.Add(tp, (col+8)*u16)
		r0a := archsimd.LoadInt16x8Array((*[8]int16)(rowa))
		r1a := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(rowa, 1*tempStride*u16)))
		r2a := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(rowa, 2*tempStride*u16)))
		r3a := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(rowa, 3*tempStride*u16)))
		r4a := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(rowa, 4*tempStride*u16)))
		r5a := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(rowa, 5*tempStride*u16)))
		r0b := archsimd.LoadInt16x8Array((*[8]int16)(rowb))
		r1b := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(rowb, 1*tempStride*u16)))
		r2b := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(rowb, 2*tempStride*u16)))
		r3b := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(rowb, 3*tempStride*u16)))
		r4b := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(rowb, 4*tempStride*u16)))
		r5b := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(rowb, 5*tempStride*u16)))
		nexta := unsafe.Add(rowa, 6*tempStride*u16)
		nextb := unsafe.Add(rowb, 6*tempStride*u16)
		drow := unsafe.Add(dp, col)
		for row := 0; row < height; row++ {
			r6a := archsimd.LoadInt16x8Array((*[8]int16)(nexta))
			r6b := archsimd.LoadInt16x8Array((*[8]int16)(nextb))
			pair0a, pair1a, pair2a := r0a.Add(r6a), r1a.Add(r5a), r2a.Add(r4a)
			loA := biasV.Add(pair0a.MulWidenLo(f0)).Add(pair1a.MulWidenLo(f1)).
				Add(pair2a.MulWidenLo(f2)).Add(r3a.MulWidenLo(f3))
			hiA := biasV.Add(pair0a.HiToLo().MulWidenLo(hf0)).Add(pair1a.HiToLo().MulWidenLo(hf1)).
				Add(pair2a.HiToLo().MulWidenLo(hf2)).Add(r3a.HiToLo().MulWidenLo(hf3))
			pa := restorationRoundShiftNarrowInt32Pair(loA, hiA, roundShiftV, roundBitShiftV, roundOneV)
			pair0b, pair1b, pair2b := r0b.Add(r6b), r1b.Add(r5b), r2b.Add(r4b)
			loB := biasV.Add(pair0b.MulWidenLo(f0)).Add(pair1b.MulWidenLo(f1)).
				Add(pair2b.MulWidenLo(f2)).Add(r3b.MulWidenLo(f3))
			hiB := biasV.Add(pair0b.HiToLo().MulWidenLo(hf0)).Add(pair1b.HiToLo().MulWidenLo(hf1)).
				Add(pair2b.HiToLo().MulWidenLo(hf2)).Add(r3b.HiToLo().MulWidenLo(hf3))
			pb := restorationRoundShiftNarrowInt32Pair(loB, hiB, roundShiftV, roundBitShiftV, roundOneV)
			out := restorationSaturateInt16PairToUint8(pa, pb)
			out.StoreArray((*[16]uint8)(unsafe.Add(drow, row*dstStride)))
			if row+1 < height {
				r0a, r1a, r2a, r3a, r4a, r5a = r1a, r2a, r3a, r4a, r5a, r6a
				r0b, r1b, r2b, r3b, r4b, r5b = r1b, r2b, r3b, r4b, r5b, r6b
				nexta = unsafe.Add(nexta, tempStride*u16)
				nextb = unsafe.Add(nextb, tempStride*u16)
			}
		}
	}
	// 8-wide sliding tail for a trailing width%16 == 8 strip.
	for col := w16; col < w8; col += 8 {
		off := col * u16
		rowp := unsafe.Add(tp, off) // &temp[0*stride+col]
		r0 := archsimd.LoadInt16x8Array((*[8]int16)(rowp))
		r1 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(rowp, 1*tempStride*u16)))
		r2 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(rowp, 2*tempStride*u16)))
		r3 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(rowp, 3*tempStride*u16)))
		r4 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(rowp, 4*tempStride*u16)))
		r5 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(rowp, 5*tempStride*u16)))
		nextp := unsafe.Add(rowp, 6*tempStride*u16) // &temp[(row+6)*stride+col]
		drow := unsafe.Add(dp, col)
		for row := 0; row < height; row++ {
			r6 := archsimd.LoadInt16x8Array((*[8]int16)(nextp))
			pair0, pair1, pair2 := r0.Add(r6), r1.Add(r5), r2.Add(r4)
			lo := pair0.MulWidenLo(f0).Add(pair1.MulWidenLo(f1)).
				Add(pair2.MulWidenLo(f2)).Add(r3.MulWidenLo(f3)).Add(biasV)
			hi := pair0.HiToLo().MulWidenLo(hf0).Add(pair1.HiToLo().MulWidenLo(hf1)).
				Add(pair2.HiToLo().MulWidenLo(hf2)).Add(r3.HiToLo().MulWidenLo(hf3)).Add(biasV)
			out := restorationRoundShiftNarrowInt32Pair(lo, hi, roundShiftV, roundBitShiftV, roundOneV).SaturateToUint8()
			*(*float64)(unsafe.Add(drow, row*dstStride)) = out.ReshapeToUint64s().BitsToFloat64().GetElem(0)
			// Slide the window: drop r0, shift up, r6 becomes the new r5 side.
			if row+1 < height {
				r0, r1, r2, r3, r4, r5 = r1, r2, r3, r4, r5, r6
				nextp = unsafe.Add(nextp, tempStride*u16)
			}
		}
	}
	// scalar remainder for the trailing width%8 columns.
	for col := w8; col < width; col++ {
		for row := 0; row < height; row++ {
			c3 := int32(temp[(row+3)*tempStride+col])
			sum := c3<<WienerFilterBits - offset
			sum += int32(temp[(row+0)*tempStride+col])*int32(filter[0]) +
				int32(temp[(row+1)*tempStride+col])*int32(filter[1]) +
				int32(temp[(row+2)*tempStride+col])*int32(filter[2]) +
				c3*int32(filter[3]) +
				int32(temp[(row+4)*tempStride+col])*int32(filter[4]) +
				int32(temp[(row+5)*tempStride+col])*int32(filter[5]) +
				int32(temp[(row+6)*tempStride+col])*int32(filter[6])
			dst[row*dstStride+col] = uint8(clampInt32(roundPowerOfTwo(sum, round1), 0, 255))
		}
	}
}
