// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package motion

import (
	"simd/archsimd"
	"unsafe"
)

// scaledStep2HorizontalRow filters one source row into eight output columns
// at a time for the 2:1 scaled convolve. Even lanes read bytes t+j and odd
// lanes bytes t+j+odd of the load, so tap t interleaves the load shifted by t
// with the load shifted by t+odd (EXT + ZIP1).
//
//go:nocheckptr
func scaledStep2HorizontalRow(src unsafe.Pointer, w8 int, odd int, coef *[filterTaps]archsimd.Int16x8, row []int16) {
	xRound := archsimd.BroadcastInt16x8(8194)
	right2 := archsimd.BroadcastInt16x8(-2)
	for x := 0; x < w8; x += 8 {
		a := archsimd.LoadUint8x16((*[16]byte)(unsafe.Add(src, x/2))[:])
		s1 := a.ConcatShiftBytesRight(a, 1)
		s2 := a.ConcatShiftBytesRight(a, 2)
		s3 := a.ConcatShiftBytesRight(a, 3)
		s4 := a.ConcatShiftBytesRight(a, 4)
		s5 := a.ConcatShiftBytesRight(a, 5)
		s6 := a.ConcatShiftBytesRight(a, 6)
		s7 := a.ConcatShiftBytesRight(a, 7)
		var t0, t1, t2, t3, t4, t5, t6, t7 archsimd.Uint8x16
		if odd == 0 {
			t0, t1, t2, t3 = a.InterleaveLo(a), s1.InterleaveLo(s1), s2.InterleaveLo(s2), s3.InterleaveLo(s3)
			t4, t5, t6, t7 = s4.InterleaveLo(s4), s5.InterleaveLo(s5), s6.InterleaveLo(s6), s7.InterleaveLo(s7)
		} else {
			s8 := a.ConcatShiftBytesRight(a, 8)
			t0, t1, t2, t3 = a.InterleaveLo(s1), s1.InterleaveLo(s2), s2.InterleaveLo(s3), s3.InterleaveLo(s4)
			t4, t5, t6, t7 = s4.InterleaveLo(s5), s5.InterleaveLo(s6), s6.InterleaveLo(s7), s7.InterleaveLo(s8)
		}
		sum := u8HalfMAC(t0.ExtendLo8ToUint16().BitsToInt16(), coef[0], xRound)
		sum = u8HalfMAC(t1.ExtendLo8ToUint16().BitsToInt16(), coef[1], sum)
		sum = u8HalfMAC(t2.ExtendLo8ToUint16().BitsToInt16(), coef[2], sum)
		sum = u8HalfMAC(t3.ExtendLo8ToUint16().BitsToInt16(), coef[3], sum)
		sum = u8HalfMAC(t4.ExtendLo8ToUint16().BitsToInt16(), coef[4], sum)
		sum = u8HalfMAC(t5.ExtendLo8ToUint16().BitsToInt16(), coef[5], sum)
		sum = u8HalfMAC(t6.ExtendLo8ToUint16().BitsToInt16(), coef[6], sum)
		sum = u8HalfMAC(t7.ExtendLo8ToUint16().BitsToInt16(), coef[7], sum)
		u8Right16(sum, right2, 2).Store(row[x : x+8])
	}
}
