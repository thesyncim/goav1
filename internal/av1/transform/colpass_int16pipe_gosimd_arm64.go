// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package transform

import (
	"simd/archsimd"
	"unsafe"
)

// Bind the int16 8-wide SIMD DCT8 column kernel under GOEXPERIMENT=simd. It is
// byte-exact and ~3.8x faster than the NEON asm (no boundary narrowing).
func init() {
	inverseDCT8Col8Impl16 = inverseDCT8Col8SIMD16
	inverseDCT16Col8Impl16 = inverseDCT16Col8SIMD16
	inverseDCT32Col8Impl16 = inverseDCT32Col8SIMD16
	inverseDCT64Col8Impl16 = inverseDCT64Col8SIMD16
	clampRoundNarrowInt16Impl = clampRoundNarrowInt16SIMD
	int16ColumnFast = true
}


// clampRoundNarrowInt16SIMD is the fused mid-pass round+clamp+narrow: at
// bitDepth 8 the column clamp bounds are exactly the int16 range, so
// roundShift + clip + narrow is one saturating rounding narrow (SQRSHRN) per
// four lanes. Shifts other than 1/2 (and non-int16 bounds, which cannot occur
// on the 8-bit path) fall back to the scalar sweep.
func clampRoundNarrowInt16SIMD(src []int32, dst []int16, shift int, lo int32, hi int32) {
	if lo != -32768 || hi != 32767 || (shift != 1 && shift != 2) {
		clampRoundNarrowInt16Scalar(src, dst, shift, lo, hi)
		return
	}
	n := len(src)
	i := 0
	if shift == 1 {
		for ; i+8 <= n; i += 8 {
			v0 := archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Pointer(&src[i])))
			v1 := archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Pointer(&src[i+4])))
			v0.ShiftRightRoundNarrow(1).ShiftRightRoundNarrowHi(v1, 1).
				StoreArray((*[8]int16)(unsafe.Pointer(&dst[i])))
		}
	} else {
		for ; i+8 <= n; i += 8 {
			v0 := archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Pointer(&src[i])))
			v1 := archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Pointer(&src[i+4])))
			v0.ShiftRightRoundNarrow(2).ShiftRightRoundNarrowHi(v1, 2).
				StoreArray((*[8]int16)(unsafe.Pointer(&dst[i])))
		}
	}
	if i < n {
		clampRoundNarrowInt16Scalar(src[i:], dst[i:], shift, lo, hi)
	}
}
