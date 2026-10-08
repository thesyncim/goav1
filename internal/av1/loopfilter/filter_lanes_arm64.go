// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package loopfilter

import (
	"simd/archsimd"
	"unsafe"
)

// lfStore8 narrows eight in-range lanes and writes exactly eight bytes. The
// narrow is a NEON SQXTUN and the low 64-bit lane becomes one store.
//
//go:nocheckptr
func lfStore8(pix []byte, off int, v archsimd.Int16x8) {
	*(*uint64)(unsafe.Add(unsafe.Pointer(unsafe.SliceData(pix)), off)) = v.ToBits().SaturateToUint8().ReshapeToUint64s().GetElem(0)
}

// lfLoad8 reads exactly eight bytes and zero-extends them to signed lanes.
//
//go:nocheckptr
func lfLoad8(pix []byte, off int) archsimd.Int16x8 {
	p := unsafe.Add(unsafe.Pointer(unsafe.SliceData(pix)), off)
	return archsimd.BroadcastUint64x2(*(*uint64)(p)).ReshapeToUint8s().ExtendLo8ToUint16().BitsToInt16()
}

// lfAny reports whether any lane of the mask is set.
func lfAny(m archsimd.Mask16x8) bool {
	return m.ToInt16x8().ToBits().ReduceSum() != 0
}

func lfAll(m archsimd.Mask16x8) bool {
	return m.ToInt16x8().ToBits().ReduceSum() == 65528
}

// lfShift1 shifts by a reusable vector on arm64.
func lfShift1(v, by archsimd.Int16x8) archsimd.Int16x8 {
	return v.Shift(by)
}

// lfShift3 shifts by a reusable vector on arm64.
func lfShift3(v, by archsimd.Int16x8) archsimd.Int16x8 {
	return v.Shift(by)
}

// lfShift4 shifts by a reusable vector on arm64.
func lfShift4(v, by archsimd.Int16x8) archsimd.Int16x8 {
	return v.Shift(by)
}
