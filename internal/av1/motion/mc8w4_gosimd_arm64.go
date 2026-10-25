// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package motion

import (
	"simd/archsimd"
	"unsafe"
)

// Width-4 variants of the dav1d-style 8-bit put/prep kernels (dav1d mc.S
// "40:" paths). Width 4 selects the 4-tap (or bilinear) tables, which sit
// inside the same six-tap window, so the arithmetic is identical to the
// eight-wide kernels: the horizontal sums are formed on eight lanes of which
// the low four are kept, and the vertical passes pack two output rows into one
// vector (rows of a width-4 block are contiguous in the int32 intermediate and
// in the CONV_BUF).

// mc8H6ToI32W4 writes rows x 4 rounded horizontal sums as int32, row stride 4.
//
//go:nocheckptr
func mc8H6ToI32W4(src unsafe.Pointer, srcStride, rows int, ph *mc8Phase, im unsafe.Pointer) {
	c0 := archsimd.LoadInt16x8Array(&ph.h[0])
	c1 := archsimd.LoadInt16x8Array(&ph.h[1])
	c2 := archsimd.LoadInt16x8Array(&ph.h[2])
	c3 := archsimd.LoadInt16x8Array(&ph.h[3])
	c4 := archsimd.LoadInt16x8Array(&ph.h[4])
	c5 := archsimd.LoadInt16x8Array(&ph.h[5])
	rnd := archsimd.BroadcastInt16x8(2)
	sh := archsimd.BroadcastInt16x8(-2)
	r := 0
	for ; r+2 <= rows; r += 2 {
		s := unsafe.Add(src, r*srcStride)
		lo, hi := mc8LoadWiden(s)
		lo1, hi1 := mc8LoadWiden(unsafe.Add(s, srcStride))
		sum := mc8I16(lo).MulAdd(c0, rnd)
		sum1 := mc8I16(lo1).MulAdd(c0, rnd)
		sum = mc8I16(hi.ConcatShiftBytesRight(lo, 2)).MulAdd(c1, sum)
		sum1 = mc8I16(hi1.ConcatShiftBytesRight(lo1, 2)).MulAdd(c1, sum1)
		sum = mc8I16(hi.ConcatShiftBytesRight(lo, 4)).MulAdd(c2, sum)
		sum1 = mc8I16(hi1.ConcatShiftBytesRight(lo1, 4)).MulAdd(c2, sum1)
		sum = mc8I16(hi.ConcatShiftBytesRight(lo, 6)).MulAdd(c3, sum)
		sum1 = mc8I16(hi1.ConcatShiftBytesRight(lo1, 6)).MulAdd(c3, sum1)
		sum = mc8I16(hi.ConcatShiftBytesRight(lo, 8)).MulAdd(c4, sum)
		sum1 = mc8I16(hi1.ConcatShiftBytesRight(lo1, 8)).MulAdd(c4, sum1)
		sum = mc8I16(hi.ConcatShiftBytesRight(lo, 10)).MulAdd(c5, sum)
		sum1 = mc8I16(hi1.ConcatShiftBytesRight(lo1, 10)).MulAdd(c5, sum1)
		q := unsafe.Add(im, r*16)
		sum.Shift(sh).ExtendLo4ToInt32().StoreArray((*[4]int32)(q))
		sum1.Shift(sh).ExtendLo4ToInt32().StoreArray((*[4]int32)(unsafe.Add(q, 16)))
	}
	if r < rows {
		lo, hi := mc8LoadWiden(unsafe.Add(src, r*srcStride))
		sum := mc8I16(lo).MulAdd(c0, rnd)
		sum = mc8I16(hi.ConcatShiftBytesRight(lo, 2)).MulAdd(c1, sum)
		sum = mc8I16(hi.ConcatShiftBytesRight(lo, 4)).MulAdd(c2, sum)
		sum = mc8I16(hi.ConcatShiftBytesRight(lo, 6)).MulAdd(c3, sum)
		sum = mc8I16(hi.ConcatShiftBytesRight(lo, 8)).MulAdd(c4, sum)
		sum = mc8I16(hi.ConcatShiftBytesRight(lo, 10)).MulAdd(c5, sum)
		sum.Shift(sh).ExtendLo4ToInt32().StoreArray((*[4]int32)(unsafe.Add(im, r*16)))
	}
}

// mc8V6I32W4 is the shared vertical tap chain of the width-4 2D passes: rows
// r (a) and r+1 (b) from the six-row window x0..x6.
func mc8V6I32W4(x0, x1, x2, x3, x4, x5, x6, c0, c1, c2, c3, c4, c5, rnd archsimd.Int32x4) (archsimd.Int32x4, archsimd.Int32x4) {
	a := x0.MulAdd(c0, rnd)
	b := x1.MulAdd(c0, rnd)
	a = x1.MulAdd(c1, a)
	b = x2.MulAdd(c1, b)
	a = x2.MulAdd(c2, a)
	b = x3.MulAdd(c2, b)
	a = x3.MulAdd(c3, a)
	b = x4.MulAdd(c3, b)
	a = x4.MulAdd(c4, a)
	b = x5.MulAdd(c4, b)
	a = x5.MulAdd(c5, a)
	b = x6.MulAdd(c5, b)
	return a, b
}

// mc8PutV6FromI32W4 is the width-4 put 2D vertical pass. The two output rows'
// high halves pack into one vector (UZP2) that saturates to rows r and r+1.
//
//go:nocheckptr
func mc8PutV6FromI32W4(im unsafe.Pointer, h int, ph *mc8Phase, dst unsafe.Pointer, dstStride int) {
	c0 := archsimd.LoadInt32x4Array(&ph.v[0])
	c1 := archsimd.LoadInt32x4Array(&ph.v[1])
	c2 := archsimd.LoadInt32x4Array(&ph.v[2])
	c3 := archsimd.LoadInt32x4Array(&ph.v[3])
	c4 := archsimd.LoadInt32x4Array(&ph.v[4])
	c5 := archsimd.LoadInt32x4Array(&ph.v[5])
	rnd := archsimd.BroadcastInt32x4(1 << 15)
	x0 := archsimd.LoadInt32x4Array((*[4]int32)(im))
	x1 := archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Add(im, 16)))
	x2 := archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Add(im, 32)))
	x3 := archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Add(im, 48)))
	x4 := archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Add(im, 64)))
	q := unsafe.Add(im, 80)
	d := dst
	r := 0
	for ; r+2 <= h; r += 2 {
		x5 := archsimd.LoadInt32x4Array((*[4]int32)(q))
		x6 := archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Add(q, 16)))
		q = unsafe.Add(q, 32)
		a, b := mc8V6I32W4(x0, x1, x2, x3, x4, x5, x6, c0, c1, c2, c3, c4, c5, rnd)
		v := a.ToBits().ReshapeToUint16s().ConcatOdd(b.ToBits().ReshapeToUint16s()).BitsToInt16()
		u := v.SaturateToUint8().ReshapeToUint64s().GetElem(0)
		*(*uint32)(d) = uint32(u)
		*(*uint32)(unsafe.Add(d, dstStride)) = uint32(u >> 32)
		d = unsafe.Add(d, 2*dstStride)
		x0, x1, x2, x3, x4 = x2, x3, x4, x5, x6
	}
	if r < h {
		x5 := archsimd.LoadInt32x4Array((*[4]int32)(q))
		a, _ := mc8V6I32W4(x0, x1, x2, x3, x4, x5, x5, c0, c1, c2, c3, c4, c5, rnd)
		v := a.ToBits().ReshapeToUint16s().ConcatOdd(a.ToBits().ReshapeToUint16s()).BitsToInt16()
		*(*uint32)(d) = uint32(v.SaturateToUint8().ReshapeToUint64s().GetElem(0))
	}
}

// mc8PrepV6FromI32W4 is the width-4 prep 2D vertical pass; two CONV_BUF rows
// are one 16-byte store.
//
//go:nocheckptr
func mc8PrepV6FromI32W4(im unsafe.Pointer, h int, ph *mc8Phase, out unsafe.Pointer, rndConst int32) {
	c0 := archsimd.LoadInt32x4Array(&ph.p[0])
	c1 := archsimd.LoadInt32x4Array(&ph.p[1])
	c2 := archsimd.LoadInt32x4Array(&ph.p[2])
	c3 := archsimd.LoadInt32x4Array(&ph.p[3])
	c4 := archsimd.LoadInt32x4Array(&ph.p[4])
	c5 := archsimd.LoadInt32x4Array(&ph.p[5])
	rnd := archsimd.BroadcastInt32x4(rndConst)
	x0 := archsimd.LoadInt32x4Array((*[4]int32)(im))
	x1 := archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Add(im, 16)))
	x2 := archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Add(im, 32)))
	x3 := archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Add(im, 48)))
	x4 := archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Add(im, 64)))
	q := unsafe.Add(im, 80)
	o := out
	r := 0
	for ; r+2 <= h; r += 2 {
		x5 := archsimd.LoadInt32x4Array((*[4]int32)(q))
		x6 := archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Add(q, 16)))
		q = unsafe.Add(q, 32)
		a, b := mc8V6I32W4(x0, x1, x2, x3, x4, x5, x6, c0, c1, c2, c3, c4, c5, rnd)
		a.ToBits().ReshapeToUint16s().ConcatOdd(b.ToBits().ReshapeToUint16s()).StoreArray((*[8]uint16)(o))
		o = unsafe.Add(o, 16)
		x0, x1, x2, x3, x4 = x2, x3, x4, x5, x6
	}
	if r < h {
		x5 := archsimd.LoadInt32x4Array((*[4]int32)(q))
		a, _ := mc8V6I32W4(x0, x1, x2, x3, x4, x5, x5, c0, c1, c2, c3, c4, c5, rnd)
		v := a.ToBits().ReshapeToUint16s().ConcatOdd(a.ToBits().ReshapeToUint16s())
		*(*uint64)(o) = v.ReshapeToUint64s().GetElem(0)
	}
}

// mc8PutH6W4 is the width-4 put horizontal-only pass (two rows per
// iteration; an odd last row reruns the chain on itself).
//
//go:nocheckptr
func mc8PutH6W4(src unsafe.Pointer, srcStride, h int, ph *mc8Phase, dst unsafe.Pointer, dstStride int) {
	c0 := archsimd.LoadInt16x8Array(&ph.h[0])
	c1 := archsimd.LoadInt16x8Array(&ph.h[1])
	c2 := archsimd.LoadInt16x8Array(&ph.h[2])
	c3 := archsimd.LoadInt16x8Array(&ph.h[3])
	c4 := archsimd.LoadInt16x8Array(&ph.h[4])
	c5 := archsimd.LoadInt16x8Array(&ph.h[5])
	rnd := archsimd.BroadcastInt16x8(34)
	sh := archsimd.BroadcastInt16x8(-6)
	for r := 0; r < h; r += 2 {
		s := unsafe.Add(src, r*srcStride)
		d := unsafe.Add(dst, r*dstStride)
		step := srcStride
		if r+2 > h {
			step = 0
		}
		lo, hi := mc8LoadWiden(s)
		lo1, hi1 := mc8LoadWiden(unsafe.Add(s, step))
		sum := mc8I16(lo).MulAdd(c0, rnd)
		sum1 := mc8I16(lo1).MulAdd(c0, rnd)
		sum = mc8I16(hi.ConcatShiftBytesRight(lo, 2)).MulAdd(c1, sum)
		sum1 = mc8I16(hi1.ConcatShiftBytesRight(lo1, 2)).MulAdd(c1, sum1)
		sum = mc8I16(hi.ConcatShiftBytesRight(lo, 4)).MulAdd(c2, sum)
		sum1 = mc8I16(hi1.ConcatShiftBytesRight(lo1, 4)).MulAdd(c2, sum1)
		sum = mc8I16(hi.ConcatShiftBytesRight(lo, 6)).MulAdd(c3, sum)
		sum1 = mc8I16(hi1.ConcatShiftBytesRight(lo1, 6)).MulAdd(c3, sum1)
		sum = mc8I16(hi.ConcatShiftBytesRight(lo, 8)).MulAdd(c4, sum)
		sum1 = mc8I16(hi1.ConcatShiftBytesRight(lo1, 8)).MulAdd(c4, sum1)
		sum = mc8I16(hi.ConcatShiftBytesRight(lo, 10)).MulAdd(c5, sum)
		sum1 = mc8I16(hi1.ConcatShiftBytesRight(lo1, 10)).MulAdd(c5, sum1)
		*(*uint32)(d) = uint32(sum.Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0))
		if step != 0 {
			*(*uint32)(unsafe.Add(d, dstStride)) = uint32(sum1.Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0))
		}
	}
}

// mc8PrepH6W4 is the width-4 prep horizontal-only pass; two CONV_BUF rows are
// one 16-byte store.
//
//go:nocheckptr
func mc8PrepH6W4(src unsafe.Pointer, srcStride, h int, ph *mc8Phase, out unsafe.Pointer) {
	c0 := archsimd.LoadInt16x8Array(&ph.h[0])
	c1 := archsimd.LoadInt16x8Array(&ph.h[1])
	c2 := archsimd.LoadInt16x8Array(&ph.h[2])
	c3 := archsimd.LoadInt16x8Array(&ph.h[3])
	c4 := archsimd.LoadInt16x8Array(&ph.h[4])
	c5 := archsimd.LoadInt16x8Array(&ph.h[5])
	rnd := archsimd.BroadcastInt16x8(2 + 6144<<2)
	sh := archsimd.BroadcastInt16x8(-2)
	for r := 0; r < h; r += 2 {
		s := unsafe.Add(src, r*srcStride)
		o := unsafe.Add(out, r*8)
		step := srcStride
		if r+2 > h {
			step = 0
		}
		lo, hi := mc8LoadWiden(s)
		lo1, hi1 := mc8LoadWiden(unsafe.Add(s, step))
		sum := mc8I16(lo).MulAdd(c0, rnd)
		sum1 := mc8I16(lo1).MulAdd(c0, rnd)
		sum = mc8I16(hi.ConcatShiftBytesRight(lo, 2)).MulAdd(c1, sum)
		sum1 = mc8I16(hi1.ConcatShiftBytesRight(lo1, 2)).MulAdd(c1, sum1)
		sum = mc8I16(hi.ConcatShiftBytesRight(lo, 4)).MulAdd(c2, sum)
		sum1 = mc8I16(hi1.ConcatShiftBytesRight(lo1, 4)).MulAdd(c2, sum1)
		sum = mc8I16(hi.ConcatShiftBytesRight(lo, 6)).MulAdd(c3, sum)
		sum1 = mc8I16(hi1.ConcatShiftBytesRight(lo1, 6)).MulAdd(c3, sum1)
		sum = mc8I16(hi.ConcatShiftBytesRight(lo, 8)).MulAdd(c4, sum)
		sum1 = mc8I16(hi1.ConcatShiftBytesRight(lo1, 8)).MulAdd(c4, sum1)
		sum = mc8I16(hi.ConcatShiftBytesRight(lo, 10)).MulAdd(c5, sum)
		sum1 = mc8I16(hi1.ConcatShiftBytesRight(lo1, 10)).MulAdd(c5, sum1)
		// The offset sums lie in [0, 2^16): shift them as unsigned.
		a := sum.ToBits().Shift(sh).ReshapeToUint64s()
		if step == 0 {
			*(*uint64)(o) = a.GetElem(0)
			break
		}
		a.InterleaveLo(sum1.ToBits().Shift(sh).ReshapeToUint64s()).ReshapeToUint16s().StoreArray((*[8]uint16)(o))
	}
}

// mc8V6W4Sums is the six-tap vertical chain of two width-4 rows (eight lanes,
// low four kept) from the seven-row window r0..r6.
func mc8V6W4Sums(r0, r1, r2, r3, r4, r5, r6, c0, c1, c2, c3, c4, c5, rnd archsimd.Int16x8) (archsimd.Int16x8, archsimd.Int16x8) {
	a := r0.MulAdd(c0, rnd)
	b := r1.MulAdd(c0, rnd)
	a = r1.MulAdd(c1, a)
	b = r2.MulAdd(c1, b)
	a = r2.MulAdd(c2, a)
	b = r3.MulAdd(c2, b)
	a = r3.MulAdd(c3, a)
	b = r4.MulAdd(c3, b)
	a = r4.MulAdd(c4, a)
	b = r5.MulAdd(c4, b)
	a = r5.MulAdd(c5, a)
	b = r6.MulAdd(c5, b)
	return a, b
}

// mc8PutV6W4 is the width-4 put vertical-only pass.
//
//go:nocheckptr
func mc8PutV6W4(src unsafe.Pointer, srcStride, h int, ph *mc8Phase, dst unsafe.Pointer, dstStride int) {
	c0 := archsimd.LoadInt16x8Array(&ph.h[0])
	c1 := archsimd.LoadInt16x8Array(&ph.h[1])
	c2 := archsimd.LoadInt16x8Array(&ph.h[2])
	c3 := archsimd.LoadInt16x8Array(&ph.h[3])
	c4 := archsimd.LoadInt16x8Array(&ph.h[4])
	c5 := archsimd.LoadInt16x8Array(&ph.h[5])
	rnd := archsimd.BroadcastInt16x8(32)
	sh := archsimd.BroadcastInt16x8(-6)
	r0 := archsimd.LoadUint8x16Array((*[16]uint8)(src)).ExtendLo8ToUint16().BitsToInt16()
	r1 := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(src, srcStride))).ExtendLo8ToUint16().BitsToInt16()
	r2 := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(src, 2*srcStride))).ExtendLo8ToUint16().BitsToInt16()
	r3 := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(src, 3*srcStride))).ExtendLo8ToUint16().BitsToInt16()
	r4 := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(src, 4*srcStride))).ExtendLo8ToUint16().BitsToInt16()
	s := unsafe.Add(src, 5*srcStride)
	d := dst
	r := 0
	for ; r+2 <= h; r += 2 {
		r5 := archsimd.LoadUint8x16Array((*[16]uint8)(s)).ExtendLo8ToUint16().BitsToInt16()
		r6 := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(s, srcStride))).ExtendLo8ToUint16().BitsToInt16()
		a, b := mc8V6W4Sums(r0, r1, r2, r3, r4, r5, r6, c0, c1, c2, c3, c4, c5, rnd)
		*(*uint32)(d) = uint32(a.Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0))
		*(*uint32)(unsafe.Add(d, dstStride)) = uint32(b.Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0))
		r0, r1, r2, r3, r4 = r2, r3, r4, r5, r6
		s = unsafe.Add(s, 2*srcStride)
		d = unsafe.Add(d, 2*dstStride)
	}
	if r < h {
		r5 := archsimd.LoadUint8x16Array((*[16]uint8)(s)).ExtendLo8ToUint16().BitsToInt16()
		a, _ := mc8V6W4Sums(r0, r1, r2, r3, r4, r5, r5, c0, c1, c2, c3, c4, c5, rnd)
		*(*uint32)(d) = uint32(a.Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0))
	}
}

// mc8PrepV6W4 is the width-4 prep vertical-only pass; two CONV_BUF rows are one
// 16-byte store.
//
//go:nocheckptr
func mc8PrepV6W4(src unsafe.Pointer, srcStride, h int, ph *mc8Phase, out unsafe.Pointer) {
	c0 := archsimd.LoadInt16x8Array(&ph.h[0])
	c1 := archsimd.LoadInt16x8Array(&ph.h[1])
	c2 := archsimd.LoadInt16x8Array(&ph.h[2])
	c3 := archsimd.LoadInt16x8Array(&ph.h[3])
	c4 := archsimd.LoadInt16x8Array(&ph.h[4])
	c5 := archsimd.LoadInt16x8Array(&ph.h[5])
	rnd := archsimd.BroadcastInt16x8(2 + 6144<<2)
	sh := archsimd.BroadcastInt16x8(-2)
	r0 := archsimd.LoadUint8x16Array((*[16]uint8)(src)).ExtendLo8ToUint16().BitsToInt16()
	r1 := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(src, srcStride))).ExtendLo8ToUint16().BitsToInt16()
	r2 := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(src, 2*srcStride))).ExtendLo8ToUint16().BitsToInt16()
	r3 := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(src, 3*srcStride))).ExtendLo8ToUint16().BitsToInt16()
	r4 := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(src, 4*srcStride))).ExtendLo8ToUint16().BitsToInt16()
	s := unsafe.Add(src, 5*srcStride)
	o := out
	r := 0
	for ; r+2 <= h; r += 2 {
		r5 := archsimd.LoadUint8x16Array((*[16]uint8)(s)).ExtendLo8ToUint16().BitsToInt16()
		r6 := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(s, srcStride))).ExtendLo8ToUint16().BitsToInt16()
		a, b := mc8V6W4Sums(r0, r1, r2, r3, r4, r5, r6, c0, c1, c2, c3, c4, c5, rnd)
		a.ToBits().Shift(sh).ReshapeToUint64s().InterleaveLo(b.ToBits().Shift(sh).ReshapeToUint64s()).ReshapeToUint16s().StoreArray((*[8]uint16)(o))
		r0, r1, r2, r3, r4 = r2, r3, r4, r5, r6
		s = unsafe.Add(s, 2*srcStride)
		o = unsafe.Add(o, 16)
	}
	if r < h {
		r5 := archsimd.LoadUint8x16Array((*[16]uint8)(s)).ExtendLo8ToUint16().BitsToInt16()
		a, _ := mc8V6W4Sums(r0, r1, r2, r3, r4, r5, r5, c0, c1, c2, c3, c4, c5, rnd)
		*(*uint64)(o) = a.ToBits().Shift(sh).ReshapeToUint64s().GetElem(0)
	}
}

// mc8PrepCopyW4 is the width-4 prep copy: 16 * px + 6144, two rows per store.
//
//go:nocheckptr
func mc8PrepCopyW4(src unsafe.Pointer, srcStride, h int, out unsafe.Pointer) {
	mul := archsimd.BroadcastUint16x8(16)
	off := archsimd.BroadcastUint16x8(6144)
	r := 0
	for ; r+2 <= h; r += 2 {
		s := unsafe.Add(src, r*srcStride)
		a := uint64(*(*uint32)(s)) | uint64(*(*uint32)(unsafe.Add(s, srcStride)))<<32
		v := archsimd.BroadcastUint64x2(a).ReshapeToUint8s().ExtendLo8ToUint16().MulAdd(mul, off)
		v.StoreArray((*[8]uint16)(unsafe.Add(out, r*8)))
	}
	if r < h {
		v := archsimd.BroadcastUint64x2(uint64(*(*uint32)(unsafe.Add(src, r*srcStride)))).ReshapeToUint8s().ExtendLo8ToUint16().MulAdd(mul, off)
		*(*uint64)(unsafe.Add(out, r*8)) = v.ReshapeToUint64s().GetElem(0)
	}
}
