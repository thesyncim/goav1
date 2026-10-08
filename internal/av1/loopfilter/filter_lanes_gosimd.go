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
// tap window. The 16-bit form reads exactly sixteen bytes.
func lfLoad[S lfSample](pix []byte, off int) archsimd.Int16x8 {
	if lfSize[S]() == 1 {
		return lfLoad8(pix, off)
	}
	return lfLoad16(pix, off)
}

// lfStore writes eight filtered samples back starting at byte offset off. Lane
// values are always within the sample range, so the 8-bit narrowing is exact.
func lfStore[S lfSample](pix []byte, off int, v archsimd.Int16x8) {
	if lfSize[S]() == 1 {
		lfStore8(pix, off, v)
		return
	}
	lfStore16(pix, off, v)
}

// lfLoad16 loads eight 16-bit samples directly from the validated window.
//
//go:nocheckptr
func lfLoad16(pix []byte, off int) archsimd.Int16x8 {
	p := (*[16]uint8)(unsafe.Add(unsafe.Pointer(unsafe.SliceData(pix)), off))
	return archsimd.LoadUint8x16Array(p).ReshapeToUint16s().BitsToInt16()
}

// lfStore16 writes eight 16-bit lanes to the validated window.
//
//go:nocheckptr
func lfStore16(pix []byte, off int, v archsimd.Int16x8) {
	p := (*[16]uint8)(unsafe.Add(unsafe.Pointer(unsafe.SliceData(pix)), off))
	v.ToBits().ReshapeToUint8s().StoreArray(p)
}

// lfAbsDiffInt16x8 computes signed absolute difference in each 16-bit lane,
// preserving the low 16 bits when the full-range difference wraps.
func lfAbsDiffInt16x8(a, b archsimd.Int16x8) archsimd.Int16x8 {
	return a.Max(b).Sub(a.Min(b))
}

// Eight-bit samples have a difference in [-255, 255], so subtraction followed
// by absolute value is exact. The wide path retains the wrapped lane behavior
// tested for the full int16 helper domain.
func lfAbsDiff[S lfSample](a, b archsimd.Int16x8) archsimd.Int16x8 {
	if lfSize[S]() == 1 {
		return a.Sub(b).Abs()
	}
	return lfAbsDiffInt16x8(a, b)
}
