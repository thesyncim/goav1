// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package loopfilter

import (
	"encoding/binary"
	"simd/archsimd"
	"unsafe"
)

// lfTranspose8x8U8 transposes eight rows of eight bytes. Both source and
// destination strides may differ, so the same register ladder serves gather
// and scatter. Every load and store covers exactly one eight-byte row.
//
//go:nocheckptr
func lfTranspose8x8U8(src []byte, srcOff, srcStride int, dst []byte, dstOff, dstStride int) {
	sp := unsafe.Add(unsafe.Pointer(unsafe.SliceData(src)), srcOff)
	dp := unsafe.Add(unsafe.Pointer(unsafe.SliceData(dst)), dstOff)
	a := archsimd.BroadcastUint64x2(*(*uint64)(sp)).ReshapeToUint8s()
	b := archsimd.BroadcastUint64x2(*(*uint64)(unsafe.Add(sp, srcStride))).ReshapeToUint8s()
	c := archsimd.BroadcastUint64x2(*(*uint64)(unsafe.Add(sp, 2*srcStride))).ReshapeToUint8s()
	d := archsimd.BroadcastUint64x2(*(*uint64)(unsafe.Add(sp, 3*srcStride))).ReshapeToUint8s()
	e := archsimd.BroadcastUint64x2(*(*uint64)(unsafe.Add(sp, 4*srcStride))).ReshapeToUint8s()
	f := archsimd.BroadcastUint64x2(*(*uint64)(unsafe.Add(sp, 5*srcStride))).ReshapeToUint8s()
	g := archsimd.BroadcastUint64x2(*(*uint64)(unsafe.Add(sp, 6*srcStride))).ReshapeToUint8s()
	h := archsimd.BroadcastUint64x2(*(*uint64)(unsafe.Add(sp, 7*srcStride))).ReshapeToUint8s()
	ab0, ab1 := a.InterleaveEven(b), a.InterleaveOdd(b)
	cd0, cd1 := c.InterleaveEven(d), c.InterleaveOdd(d)
	ef0, ef1 := e.InterleaveEven(f), e.InterleaveOdd(f)
	gh0, gh1 := g.InterleaveEven(h), g.InterleaveOdd(h)
	ac0, ac2 := ab0.ReshapeToUint16s().InterleaveEven(cd0.ReshapeToUint16s()), ab0.ReshapeToUint16s().InterleaveOdd(cd0.ReshapeToUint16s())
	ac1, ac3 := ab1.ReshapeToUint16s().InterleaveEven(cd1.ReshapeToUint16s()), ab1.ReshapeToUint16s().InterleaveOdd(cd1.ReshapeToUint16s())
	eg0, eg2 := ef0.ReshapeToUint16s().InterleaveEven(gh0.ReshapeToUint16s()), ef0.ReshapeToUint16s().InterleaveOdd(gh0.ReshapeToUint16s())
	eg1, eg3 := ef1.ReshapeToUint16s().InterleaveEven(gh1.ReshapeToUint16s()), ef1.ReshapeToUint16s().InterleaveOdd(gh1.ReshapeToUint16s())
	o0 := ac0.ReshapeToUint32s().InterleaveEven(eg0.ReshapeToUint32s()).ReshapeToUint64s().GetElem(0)
	o1 := ac1.ReshapeToUint32s().InterleaveEven(eg1.ReshapeToUint32s()).ReshapeToUint64s().GetElem(0)
	o2 := ac2.ReshapeToUint32s().InterleaveEven(eg2.ReshapeToUint32s()).ReshapeToUint64s().GetElem(0)
	o3 := ac3.ReshapeToUint32s().InterleaveEven(eg3.ReshapeToUint32s()).ReshapeToUint64s().GetElem(0)
	o4 := ac0.ReshapeToUint32s().InterleaveOdd(eg0.ReshapeToUint32s()).ReshapeToUint64s().GetElem(0)
	o5 := ac1.ReshapeToUint32s().InterleaveOdd(eg1.ReshapeToUint32s()).ReshapeToUint64s().GetElem(0)
	o6 := ac2.ReshapeToUint32s().InterleaveOdd(eg2.ReshapeToUint32s()).ReshapeToUint64s().GetElem(0)
	o7 := ac3.ReshapeToUint32s().InterleaveOdd(eg3.ReshapeToUint32s()).ReshapeToUint64s().GetElem(0)
	*(*uint64)(dp) = o0
	*(*uint64)(unsafe.Add(dp, dstStride)) = o1
	*(*uint64)(unsafe.Add(dp, 2*dstStride)) = o2
	*(*uint64)(unsafe.Add(dp, 3*dstStride)) = o3
	*(*uint64)(unsafe.Add(dp, 4*dstStride)) = o4
	*(*uint64)(unsafe.Add(dp, 5*dstStride)) = o5
	*(*uint64)(unsafe.Add(dp, 6*dstStride)) = o6
	*(*uint64)(unsafe.Add(dp, 7*dstStride)) = o7
}

// lfLoadShortU8 reads exactly four or six bytes; the unused upper lanes are
// zero. The scatter uses eight-byte reads from scratch, including unused tap
// rows, whose transposed output is discarded by lfStoreShortU8.
//
//go:nocheckptr
func lfLoadShortU8(pp unsafe.Pointer, width int) archsimd.Uint8x16 {
	if width == 8 {
		return archsimd.BroadcastUint64x2(*(*uint64)(pp)).ReshapeToUint8s()
	}
	v := uint64(*(*uint32)(pp))
	if width == 6 {
		v |= uint64(*(*uint16)(unsafe.Add(pp, 4))) << 32
	}
	return archsimd.BroadcastUint64x2(v).ReshapeToUint8s()
}

//go:nocheckptr
func lfStoreShortU8(pp unsafe.Pointer, width int, v uint64) {
	if width == 8 {
		*(*uint64)(pp) = v
	} else {
		*(*uint32)(pp) = uint32(v)
		if width == 6 {
			*(*uint16)(unsafe.Add(pp, 4)) = uint16(v >> 32)
		}
	}
}

//go:nocheckptr
func lfTranspose8x8U8Short(src []byte, srcOff, srcStride int, dst []byte, dstOff, dstStride int, readBytes, writeBytes int) {
	sp := unsafe.Add(unsafe.Pointer(unsafe.SliceData(src)), srcOff)
	dp := unsafe.Add(unsafe.Pointer(unsafe.SliceData(dst)), dstOff)
	a := lfLoadShortU8(sp, readBytes)
	b := lfLoadShortU8(unsafe.Add(sp, srcStride), readBytes)
	c := lfLoadShortU8(unsafe.Add(sp, 2*srcStride), readBytes)
	d := lfLoadShortU8(unsafe.Add(sp, 3*srcStride), readBytes)
	e := lfLoadShortU8(unsafe.Add(sp, 4*srcStride), readBytes)
	f := lfLoadShortU8(unsafe.Add(sp, 5*srcStride), readBytes)
	g := lfLoadShortU8(unsafe.Add(sp, 6*srcStride), readBytes)
	h := lfLoadShortU8(unsafe.Add(sp, 7*srcStride), readBytes)
	ab0, ab1 := a.InterleaveEven(b), a.InterleaveOdd(b)
	cd0, cd1 := c.InterleaveEven(d), c.InterleaveOdd(d)
	ef0, ef1 := e.InterleaveEven(f), e.InterleaveOdd(f)
	gh0, gh1 := g.InterleaveEven(h), g.InterleaveOdd(h)
	ac0, ac2 := ab0.ReshapeToUint16s().InterleaveEven(cd0.ReshapeToUint16s()), ab0.ReshapeToUint16s().InterleaveOdd(cd0.ReshapeToUint16s())
	ac1, ac3 := ab1.ReshapeToUint16s().InterleaveEven(cd1.ReshapeToUint16s()), ab1.ReshapeToUint16s().InterleaveOdd(cd1.ReshapeToUint16s())
	eg0, eg2 := ef0.ReshapeToUint16s().InterleaveEven(gh0.ReshapeToUint16s()), ef0.ReshapeToUint16s().InterleaveOdd(gh0.ReshapeToUint16s())
	eg1, eg3 := ef1.ReshapeToUint16s().InterleaveEven(gh1.ReshapeToUint16s()), ef1.ReshapeToUint16s().InterleaveOdd(gh1.ReshapeToUint16s())
	lfStoreShortU8(dp, writeBytes, ac0.ReshapeToUint32s().InterleaveEven(eg0.ReshapeToUint32s()).ReshapeToUint64s().GetElem(0))
	lfStoreShortU8(unsafe.Add(dp, dstStride), writeBytes, ac1.ReshapeToUint32s().InterleaveEven(eg1.ReshapeToUint32s()).ReshapeToUint64s().GetElem(0))
	lfStoreShortU8(unsafe.Add(dp, 2*dstStride), writeBytes, ac2.ReshapeToUint32s().InterleaveEven(eg2.ReshapeToUint32s()).ReshapeToUint64s().GetElem(0))
	lfStoreShortU8(unsafe.Add(dp, 3*dstStride), writeBytes, ac3.ReshapeToUint32s().InterleaveEven(eg3.ReshapeToUint32s()).ReshapeToUint64s().GetElem(0))
	lfStoreShortU8(unsafe.Add(dp, 4*dstStride), writeBytes, ac0.ReshapeToUint32s().InterleaveOdd(eg0.ReshapeToUint32s()).ReshapeToUint64s().GetElem(0))
	lfStoreShortU8(unsafe.Add(dp, 5*dstStride), writeBytes, ac1.ReshapeToUint32s().InterleaveOdd(eg1.ReshapeToUint32s()).ReshapeToUint64s().GetElem(0))
	lfStoreShortU8(unsafe.Add(dp, 6*dstStride), writeBytes, ac2.ReshapeToUint32s().InterleaveOdd(eg2.ReshapeToUint32s()).ReshapeToUint64s().GetElem(0))
	lfStoreShortU8(unsafe.Add(dp, 7*dstStride), writeBytes, ac3.ReshapeToUint32s().InterleaveOdd(eg3.ReshapeToUint32s()).ReshapeToUint64s().GetElem(0))
}

// lfTranspose8x8U16 is the sixteen-bit counterpart. Each access is one whole
// vector and stays inside the caller's validated eight-sample tile.
//
//go:nocheckptr
func lfTranspose8x8U16(src []byte, srcOff, srcStride int, dst []byte, dstOff, dstStride int) {
	sp := unsafe.Add(unsafe.Pointer(unsafe.SliceData(src)), srcOff)
	dp := unsafe.Add(unsafe.Pointer(unsafe.SliceData(dst)), dstOff)
	a := archsimd.LoadUint8x16Array((*[16]uint8)(sp)).ReshapeToUint16s()
	b := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(sp, srcStride))).ReshapeToUint16s()
	c := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(sp, 2*srcStride))).ReshapeToUint16s()
	d := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(sp, 3*srcStride))).ReshapeToUint16s()
	e := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(sp, 4*srcStride))).ReshapeToUint16s()
	f := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(sp, 5*srcStride))).ReshapeToUint16s()
	g := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(sp, 6*srcStride))).ReshapeToUint16s()
	h := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(sp, 7*srcStride))).ReshapeToUint16s()
	ab0, ab1 := a.InterleaveEven(b), a.InterleaveOdd(b)
	cd0, cd1 := c.InterleaveEven(d), c.InterleaveOdd(d)
	ef0, ef1 := e.InterleaveEven(f), e.InterleaveOdd(f)
	gh0, gh1 := g.InterleaveEven(h), g.InterleaveOdd(h)
	ac0, ac2 := ab0.ReshapeToUint32s().InterleaveEven(cd0.ReshapeToUint32s()), ab0.ReshapeToUint32s().InterleaveOdd(cd0.ReshapeToUint32s())
	ac1, ac3 := ab1.ReshapeToUint32s().InterleaveEven(cd1.ReshapeToUint32s()), ab1.ReshapeToUint32s().InterleaveOdd(cd1.ReshapeToUint32s())
	eg0, eg2 := ef0.ReshapeToUint32s().InterleaveEven(gh0.ReshapeToUint32s()), ef0.ReshapeToUint32s().InterleaveOdd(gh0.ReshapeToUint32s())
	eg1, eg3 := ef1.ReshapeToUint32s().InterleaveEven(gh1.ReshapeToUint32s()), ef1.ReshapeToUint32s().InterleaveOdd(gh1.ReshapeToUint32s())
	ac0.ReshapeToUint64s().InterleaveEven(eg0.ReshapeToUint64s()).ReshapeToUint8s().StoreArray((*[16]uint8)(dp))
	ac1.ReshapeToUint64s().InterleaveEven(eg1.ReshapeToUint64s()).ReshapeToUint8s().StoreArray((*[16]uint8)(unsafe.Add(dp, dstStride)))
	ac2.ReshapeToUint64s().InterleaveEven(eg2.ReshapeToUint64s()).ReshapeToUint8s().StoreArray((*[16]uint8)(unsafe.Add(dp, 2*dstStride)))
	ac3.ReshapeToUint64s().InterleaveEven(eg3.ReshapeToUint64s()).ReshapeToUint8s().StoreArray((*[16]uint8)(unsafe.Add(dp, 3*dstStride)))
	ac0.ReshapeToUint64s().InterleaveOdd(eg0.ReshapeToUint64s()).ReshapeToUint8s().StoreArray((*[16]uint8)(unsafe.Add(dp, 4*dstStride)))
	ac1.ReshapeToUint64s().InterleaveOdd(eg1.ReshapeToUint64s()).ReshapeToUint8s().StoreArray((*[16]uint8)(unsafe.Add(dp, 5*dstStride)))
	ac2.ReshapeToUint64s().InterleaveOdd(eg2.ReshapeToUint64s()).ReshapeToUint8s().StoreArray((*[16]uint8)(unsafe.Add(dp, 6*dstStride)))
	ac3.ReshapeToUint64s().InterleaveOdd(eg3.ReshapeToUint64s()).ReshapeToUint8s().StoreArray((*[16]uint8)(unsafe.Add(dp, 7*dstStride)))
}

//go:nocheckptr
func lfLoadShortU16(pp unsafe.Pointer, width int) archsimd.Uint16x8 {
	if width == 8 {
		return archsimd.LoadUint8x16Array((*[16]uint8)(pp)).ReshapeToUint16s()
	}
	var tmp [16]byte
	binary.LittleEndian.PutUint64(tmp[:], *(*uint64)(pp))
	if width == 6 {
		binary.LittleEndian.PutUint32(tmp[8:], *(*uint32)(unsafe.Add(pp, 8)))
	}
	return archsimd.LoadUint8x16Array(&tmp).ReshapeToUint16s()
}

//go:nocheckptr
func lfStoreShortU16(pp unsafe.Pointer, width int, v archsimd.Uint16x8) {
	if width == 8 {
		v.ReshapeToUint8s().StoreArray((*[16]uint8)(pp))
		return
	}
	var tmp [16]byte
	v.ReshapeToUint8s().StoreArray(&tmp)
	*(*uint64)(pp) = binary.LittleEndian.Uint64(tmp[:])
	if width == 6 {
		*(*uint32)(unsafe.Add(pp, 8)) = binary.LittleEndian.Uint32(tmp[8:])
	}
}

//go:nocheckptr
func lfTranspose8x8U16Short(src []byte, srcOff, srcStride int, dst []byte, dstOff, dstStride int, readWidth, writeWidth int) {
	sp := unsafe.Add(unsafe.Pointer(unsafe.SliceData(src)), srcOff)
	dp := unsafe.Add(unsafe.Pointer(unsafe.SliceData(dst)), dstOff)
	a := lfLoadShortU16(sp, readWidth)
	b := lfLoadShortU16(unsafe.Add(sp, srcStride), readWidth)
	c := lfLoadShortU16(unsafe.Add(sp, 2*srcStride), readWidth)
	d := lfLoadShortU16(unsafe.Add(sp, 3*srcStride), readWidth)
	e := lfLoadShortU16(unsafe.Add(sp, 4*srcStride), readWidth)
	f := lfLoadShortU16(unsafe.Add(sp, 5*srcStride), readWidth)
	g := lfLoadShortU16(unsafe.Add(sp, 6*srcStride), readWidth)
	h := lfLoadShortU16(unsafe.Add(sp, 7*srcStride), readWidth)
	ab0, ab1 := a.InterleaveEven(b), a.InterleaveOdd(b)
	cd0, cd1 := c.InterleaveEven(d), c.InterleaveOdd(d)
	ef0, ef1 := e.InterleaveEven(f), e.InterleaveOdd(f)
	gh0, gh1 := g.InterleaveEven(h), g.InterleaveOdd(h)
	ac0, ac2 := ab0.ReshapeToUint32s().InterleaveEven(cd0.ReshapeToUint32s()), ab0.ReshapeToUint32s().InterleaveOdd(cd0.ReshapeToUint32s())
	ac1, ac3 := ab1.ReshapeToUint32s().InterleaveEven(cd1.ReshapeToUint32s()), ab1.ReshapeToUint32s().InterleaveOdd(cd1.ReshapeToUint32s())
	eg0, eg2 := ef0.ReshapeToUint32s().InterleaveEven(gh0.ReshapeToUint32s()), ef0.ReshapeToUint32s().InterleaveOdd(gh0.ReshapeToUint32s())
	eg1, eg3 := ef1.ReshapeToUint32s().InterleaveEven(gh1.ReshapeToUint32s()), ef1.ReshapeToUint32s().InterleaveOdd(gh1.ReshapeToUint32s())
	lfStoreShortU16(dp, writeWidth, ac0.ReshapeToUint64s().InterleaveEven(eg0.ReshapeToUint64s()).ReshapeToUint16s())
	lfStoreShortU16(unsafe.Add(dp, dstStride), writeWidth, ac1.ReshapeToUint64s().InterleaveEven(eg1.ReshapeToUint64s()).ReshapeToUint16s())
	lfStoreShortU16(unsafe.Add(dp, 2*dstStride), writeWidth, ac2.ReshapeToUint64s().InterleaveEven(eg2.ReshapeToUint64s()).ReshapeToUint16s())
	lfStoreShortU16(unsafe.Add(dp, 3*dstStride), writeWidth, ac3.ReshapeToUint64s().InterleaveEven(eg3.ReshapeToUint64s()).ReshapeToUint16s())
	lfStoreShortU16(unsafe.Add(dp, 4*dstStride), writeWidth, ac0.ReshapeToUint64s().InterleaveOdd(eg0.ReshapeToUint64s()).ReshapeToUint16s())
	lfStoreShortU16(unsafe.Add(dp, 5*dstStride), writeWidth, ac1.ReshapeToUint64s().InterleaveOdd(eg1.ReshapeToUint64s()).ReshapeToUint16s())
	lfStoreShortU16(unsafe.Add(dp, 6*dstStride), writeWidth, ac2.ReshapeToUint64s().InterleaveOdd(eg2.ReshapeToUint64s()).ReshapeToUint16s())
	lfStoreShortU16(unsafe.Add(dp, 7*dstStride), writeWidth, ac3.ReshapeToUint64s().InterleaveOdd(eg3.ReshapeToUint64s()).ReshapeToUint16s())
}
