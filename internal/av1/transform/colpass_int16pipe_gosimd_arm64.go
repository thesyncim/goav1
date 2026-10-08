// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package transform

import (
	"simd/archsimd"
	"unsafe"
)

// Bind the int16 8-wide SIMD DCT column kernels under GOEXPERIMENT=simd.
// The public path keeps the int16 buffer between passes and uses these kernels
// for the vertical DCT when the block height is supported.
func init() {
	inverseDCT8Col8Impl16 = inverseDCT8Col8SIMD16
	inverseDCT16Col8Impl16 = inverseDCT16Col8SIMD16
	inverseDCT32Col8Impl16 = inverseDCT32Col8SIMD16
	inverseDCT64Col8Impl16 = inverseDCT64Col8SIMD16
	clampRoundNarrowInt16Impl = clampRoundNarrowInt16SIMD
	int16ColumnSIMDInputSafe = int16ColumnSIMDInputSafeSIMD
	int16ColumnSIMDInputEligible = int16ColumnSIMDInputEligibleSIMD
	int16ColumnFast = true
}

// int16ColumnSIMDInputSafeSIMD checks the certified absolute-value bound with
// one unsigned max reduction for the full block. Abs(MinInt16) retains the
// 0x8000 bit pattern, which becomes unsigned 32768 through ToBits and therefore
// correctly fails every DCT16/32/64 bound.
func int16ColumnSIMDInputSafeSIMD(buf []int16, width int, height int, min int32, max int32) bool {
	if width < 8 {
		return true
	}
	if width <= 0 || height <= 0 || len(buf) < width*height || min > max || min < minInt16 || max > maxInt16 {
		return false
	}
	if height == dct8Size {
		return true
	}
	if min > 0 || max < 0 {
		return false
	}
	limit := int16ColumnSIMDInputBound(height)
	if limit == 0 {
		return false
	}
	return int16ColumnSIMDAbsWithinBound(buf, width, height, uint16(limit))
}

// int16ColumnSIMDInputEligibleSIMD separates exactness from workload policy.
// DCT8 is exact over the full int16 range, but the int32/NEON fallback wins
// above the measured profitability threshold. Other DCT sizes use only their
// certified exactness bounds.
func int16ColumnSIMDInputEligibleSIMD(buf []int16, width int, height int, min int32, max int32) bool {
	if !int16ColumnSIMDInputSafeSIMD(buf, width, height, min, max) {
		return false
	}
	if height == dct8Size {
		return int16ColumnSIMDAbsWithinBound(buf, width, height, uint16(dct8SIMDProfitabilityBound))
	}
	return true
}

// int16ColumnSIMDAbsWithinBound uses an unsigned absolute-value reduction.
// Abs(MinInt16) has bit pattern 0x8000, which remains unsigned 32768 and thus
// fails every narrower bound. Checking the first vector immediately rejects
// common high-range inputs before scanning the rest of the block.
func int16ColumnSIMDAbsWithinBound(buf []int16, width int, height int, limit uint16) bool {
	if width < 8 {
		return true
	}
	if width <= 0 || height <= 0 || len(buf) < width*height {
		return false
	}
	total := width * height
	maxAbs := archsimd.LoadInt16x8Array((*[8]int16)(buf)).Abs().ToBits()
	if maxAbs.ReduceMax() > limit {
		return false
	}
	i := 8
	for ; i+8 <= total; i += 8 {
		v := archsimd.LoadInt16x8Array((*[8]int16)(buf[i:]))
		maxAbs = maxAbs.Max(v.Abs().ToBits())
	}
	if maxAbs.ReduceMax() > limit {
		return false
	}
	for ; i < total; i++ {
		v := int32(buf[i])
		if v < 0 {
			v = -v
		}
		if uint16(v) > limit {
			return false
		}
	}
	return true
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
			roundShiftNarrowInt32x4ToInt16x8(v0, v1, 1).
				StoreArray((*[8]int16)(unsafe.Pointer(&dst[i])))
		}
	} else {
		for ; i+8 <= n; i += 8 {
			v0 := archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Pointer(&src[i])))
			v1 := archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Pointer(&src[i+4])))
			roundShiftNarrowInt32x4ToInt16x8(v0, v1, 2).
				StoreArray((*[8]int16)(unsafe.Pointer(&dst[i])))
		}
	}
	if i < n {
		clampRoundNarrowInt16Scalar(src[i:], dst[i:], shift, lo, hi)
	}
}
