// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && (arm64 || amd64) && !purego

package loopfilter

import (
	"simd/archsimd"
	"unsafe"
)

// lfSample is the sample width a Go-native SIMD deblocking kernel is
// instantiated for: one byte per sample at 8-bit, two bytes at 10/12-bit.
type lfSample interface {
	uint8 | uint16
}

// lfSize returns the byte width of one sample of type S.
func lfSize[S lfSample]() int {
	var s S
	return int(unsafe.Sizeof(s))
}

// lfLoad loads eight consecutive samples starting at byte offset off as signed
// 16-bit lanes. Loopfilter samples are at most 4095, so the sign bit stays clear.
// The 8-bit form reads exactly eight bytes, so it never touches memory past the
// tap window.
func lfLoad[S lfSample](pix []byte, off int) archsimd.Int16x8 {
	if lfSize[S]() == 1 {
		return lfLoad8(pix, off)
	}
	return archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Pointer(&pix[off])))
}

// lfStore writes eight filtered samples back starting at byte offset off. Lane
// values are always within the sample range, so the 8-bit narrowing is exact.
func lfStore[S lfSample](pix []byte, off int, v archsimd.Int16x8) {
	if lfSize[S]() == 1 {
		lfStore8(pix, off, v)
		return
	}
	v.StoreArray((*[8]int16)(unsafe.Pointer(&pix[off])))
}

// lf16LoadP loads eight contiguous uint16 samples as signed 16-bit lanes.
// Loopfilter samples are at most 4095, so the sign bit stays clear.
func lf16LoadP(p unsafe.Pointer) archsimd.Int16x8 {
	return archsimd.LoadInt16x8Array((*[8]int16)(p))
}

// lf16StoreP writes eight filtered 16-bit samples back to a contiguous row.
func lf16StoreP(p unsafe.Pointer, v archsimd.Int16x8) {
	v.StoreArray((*[8]int16)(p))
}

// lfAbsDiffInt16x8 computes signed absolute difference in each 16-bit lane,
// preserving the low 16 bits when the full-range difference wraps.
func lfAbsDiffInt16x8(a, b archsimd.Int16x8) archsimd.Int16x8 {
	return a.Max(b).Sub(a.Min(b))
}
