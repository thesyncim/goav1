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
// pass: a 7-tap column MAC over the horizontal-pass int16 intermediate, rounded
// by round1 and clamped to uint8. The center-tap DC boost c3<<WienerFilterBits
// folds into f3' = f3 + (1<<WienerFilterBits), so the body is a pure 7-tap MAC:
// sum = -offset + Σ r_i*f_i', then roundPowerOfTwo(sum, round1) clamped [0,255].
//
// Column-strip walk with a SLIDING 7-row register window: each temp row is loaded
// once and reused across the 7 output rows that read it (the hand asm's trick),
// instead of reloading all 7 rows per output row. temp is bounded to [0,8191]
// (< 2^15), so its uint16 cells reinterpret losslessly as int16 for SMLAL; the
// int32 accumulator's SQRSHRN (rounding narrow) then SQXTUN (uint8 saturate)
// reproduce clamp(roundPowerOfTwo(sum,round1),0,255) byte-for-byte.
func wienerVerticalU8SIMD(temp []uint16, tempStride int, dst []uint8, dstStride int, width int, height int, filter WienerFilter, round1 int) {
	if width < 8 || round1 < 1 || round1 > 16 {
		wienerVerticalU8(temp, tempStride, dst, dstStride, width, height, filter, round1)
		return
	}
	offset := int32(1) << (8 + uint(round1) - 1)
	biasV := archsimd.BroadcastInt32x4(-offset)
	f0 := archsimd.BroadcastInt16x8(filter[0])
	f1 := archsimd.BroadcastInt16x8(filter[1])
	f2 := archsimd.BroadcastInt16x8(filter[2])
	f3 := archsimd.BroadcastInt16x8(filter[3] + (1 << WienerFilterBits))
	f4 := archsimd.BroadcastInt16x8(filter[4])
	f5 := archsimd.BroadcastInt16x8(filter[5])
	f6 := archsimd.BroadcastInt16x8(filter[6])
	rb := uint8(round1)

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
			loA := biasV.MulWidenLoAdd(r0a, f0).MulWidenLoAdd(r1a, f1).MulWidenLoAdd(r2a, f2).
				MulWidenLoAdd(r3a, f3).MulWidenLoAdd(r4a, f4).MulWidenLoAdd(r5a, f5).MulWidenLoAdd(r6a, f6)
			hiA := biasV.MulWidenHiAdd(r0a, f0).MulWidenHiAdd(r1a, f1).MulWidenHiAdd(r2a, f2).
				MulWidenHiAdd(r3a, f3).MulWidenHiAdd(r4a, f4).MulWidenHiAdd(r5a, f5).MulWidenHiAdd(r6a, f6)
			pa := loA.ShiftRightRoundNarrow(rb).ShiftRightRoundNarrowHi(hiA, rb)
			loB := biasV.MulWidenLoAdd(r0b, f0).MulWidenLoAdd(r1b, f1).MulWidenLoAdd(r2b, f2).
				MulWidenLoAdd(r3b, f3).MulWidenLoAdd(r4b, f4).MulWidenLoAdd(r5b, f5).MulWidenLoAdd(r6b, f6)
			hiB := biasV.MulWidenHiAdd(r0b, f0).MulWidenHiAdd(r1b, f1).MulWidenHiAdd(r2b, f2).
				MulWidenHiAdd(r3b, f3).MulWidenHiAdd(r4b, f4).MulWidenHiAdd(r5b, f5).MulWidenHiAdd(r6b, f6)
			pb := loB.ShiftRightRoundNarrow(rb).ShiftRightRoundNarrowHi(hiB, rb)
			out := pa.SaturateToUint8().SaturateToUint8Hi(pb)
			out.StoreArray((*[16]uint8)(unsafe.Add(drow, row*dstStride)))
			r0a, r1a, r2a, r3a, r4a, r5a = r1a, r2a, r3a, r4a, r5a, r6a
			r0b, r1b, r2b, r3b, r4b, r5b = r1b, r2b, r3b, r4b, r5b, r6b
			nexta = unsafe.Add(nexta, tempStride*u16)
			nextb = unsafe.Add(nextb, tempStride*u16)
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
			lo := r0.MulWidenLo(f0).MulWidenLoAdd(r1, f1).MulWidenLoAdd(r2, f2).
				MulWidenLoAdd(r3, f3).MulWidenLoAdd(r4, f4).MulWidenLoAdd(r5, f5).MulWidenLoAdd(r6, f6).Add(biasV)
			hi := r0.MulWidenHi(f0).MulWidenHiAdd(r1, f1).MulWidenHiAdd(r2, f2).
				MulWidenHiAdd(r3, f3).MulWidenHiAdd(r4, f4).MulWidenHiAdd(r5, f5).MulWidenHiAdd(r6, f6).Add(biasV)
			out := lo.ShiftRightRoundNarrow(rb).ShiftRightRoundNarrowHi(hi, rb).SaturateToUint8()
			*(*float64)(unsafe.Add(drow, row*dstStride)) = out.ReshapeToFloat64x2().GetElem(0)
			// Slide the window: drop r0, shift up, r6 becomes the new r5 side.
			r0, r1, r2, r3, r4, r5 = r1, r2, r3, r4, r5, r6
			nextp = unsafe.Add(nextp, tempStride*u16)
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
