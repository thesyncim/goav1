// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && amd64 && !purego

package dsp

import "simd/archsimd"

// Horizontal reductions for minmax_gosimd.go. AVX2 archsimd has no reduction
// intrinsic for 128-bit vectors, so each reduction folds the lanes with a
// rotate-and-min ladder and reads lane zero.

// minmaxReduceMinU8 folds the sixteen byte lanes to their minimum with a
// rotate-and-min ladder (rotations of 8, 4, 2 and 1 bytes), so lane zero holds
// the minimum of every lane.
func minmaxReduceMinU8(v archsimd.Uint8x16) uint8 {
	v = v.Min(v.ConcatShiftBytesRight(v, 8))
	v = v.Min(v.ConcatShiftBytesRight(v, 4))
	v = v.Min(v.ConcatShiftBytesRight(v, 2))
	v = v.Min(v.ConcatShiftBytesRight(v, 1))
	return v.GetElem(0)
}

// minmaxReduceMaxU8 is minmaxReduceMinU8 with unsigned maximum.
func minmaxReduceMaxU8(v archsimd.Uint8x16) uint8 {
	v = v.Max(v.ConcatShiftBytesRight(v, 8))
	v = v.Max(v.ConcatShiftBytesRight(v, 4))
	v = v.Max(v.ConcatShiftBytesRight(v, 2))
	v = v.Max(v.ConcatShiftBytesRight(v, 1))
	return v.GetElem(0)
}

// minmaxReduceMinU16 folds the eight u16 lanes to their minimum with rotations
// of 8, 4 and 2 bytes (four, two and one lanes).
func minmaxReduceMinU16(v archsimd.Uint16x8) uint16 {
	v = v.Min(minmaxRotateU16(v, 8))
	v = v.Min(minmaxRotateU16(v, 4))
	v = v.Min(minmaxRotateU16(v, 2))
	return v.GetElem(0)
}

// minmaxReduceMaxU16 is minmaxReduceMinU16 with unsigned maximum.
func minmaxReduceMaxU16(v archsimd.Uint16x8) uint16 {
	v = v.Max(minmaxRotateU16(v, 8))
	v = v.Max(minmaxRotateU16(v, 4))
	v = v.Max(minmaxRotateU16(v, 2))
	return v.GetElem(0)
}

// minmaxRotateU16 rotates the bytes of v right by shift bytes.
func minmaxRotateU16(v archsimd.Uint16x8, shift uint64) archsimd.Uint16x8 {
	b := v.ReshapeToUint8s()
	return b.ConcatShiftBytesRight(b, shift).ReshapeToUint16s()
}
