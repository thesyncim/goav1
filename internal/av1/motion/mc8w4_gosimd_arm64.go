// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package motion

import (
	"encoding/binary"
	"simd/archsimd"
)

// Width-4 variants of the dav1d-style 8-bit put/prep kernels (dav1d mc.S
// "40:" paths). Width 4 selects the 4-tap (or bilinear) tables, which sit
// inside the same six-tap window, so the arithmetic is identical to the
// eight-wide kernels: the horizontal sums are formed on eight lanes of which
// the low four are kept, and the vertical passes pack two output rows into one
// vector (rows of a width-4 block are contiguous in the int32 intermediate and
// in the CONV_BUF). Source rows are 16-byte windows (mc8Source keeps 16 bytes
// readable per row).

// mc8H6ToI32W4 writes rows x 4 rounded horizontal sums as int32, row stride 4.
func mc8H6ToI32W4(src []byte, srcStride, rows int, ph *mc8Phase, im []int32) {
	c0 := archsimd.LoadInt16x8Array(&ph.h[0])
	c1 := archsimd.LoadInt16x8Array(&ph.h[1])
	c2 := archsimd.LoadInt16x8Array(&ph.h[2])
	c3 := archsimd.LoadInt16x8Array(&ph.h[3])
	c4 := archsimd.LoadInt16x8Array(&ph.h[4])
	c5 := archsimd.LoadInt16x8Array(&ph.h[5])
	rnd := archsimd.BroadcastInt16x8(2)
	sh := archsimd.BroadcastInt16x8(-2)
	ss := uint32(srcStride)
	var so, io uint32
	r := 0
	for ; r+2 <= rows; r += 2 {
		lo, hi := mc8LoadWiden(mc8W16(src, so))
		lo1, hi1 := mc8LoadWiden(mc8W16(src, so+ss))
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
		q := mc8WI8(im, io)
		sum.Shift(sh).ExtendLo4ToInt32().StoreArray((*[4]int32)(q[0:4]))
		sum1.Shift(sh).ExtendLo4ToInt32().StoreArray((*[4]int32)(q[4:8]))
		so, io = so+2*ss, io+8
	}
	if r < rows {
		lo, hi := mc8LoadWiden(mc8W16(src, so))
		sum := mc8I16(lo).MulAdd(c0, rnd)
		sum = mc8I16(hi.ConcatShiftBytesRight(lo, 2)).MulAdd(c1, sum)
		sum = mc8I16(hi.ConcatShiftBytesRight(lo, 4)).MulAdd(c2, sum)
		sum = mc8I16(hi.ConcatShiftBytesRight(lo, 6)).MulAdd(c3, sum)
		sum = mc8I16(hi.ConcatShiftBytesRight(lo, 8)).MulAdd(c4, sum)
		sum = mc8I16(hi.ConcatShiftBytesRight(lo, 10)).MulAdd(c5, sum)
		sum.Shift(sh).ExtendLo4ToInt32().StoreArray(mc8WI4(im, io))
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
func mc8PutV6FromI32W4(im []int32, h int, ph *mc8Phase, dst []byte, dstStride int) {
	c0 := archsimd.LoadInt32x4Array(&ph.v[0])
	c1 := archsimd.LoadInt32x4Array(&ph.v[1])
	c2 := archsimd.LoadInt32x4Array(&ph.v[2])
	c3 := archsimd.LoadInt32x4Array(&ph.v[3])
	c4 := archsimd.LoadInt32x4Array(&ph.v[4])
	c5 := archsimd.LoadInt32x4Array(&ph.v[5])
	rnd := archsimd.BroadcastInt32x4(1 << 15)
	head := (*[20]int32)(im[0:20])
	x0 := archsimd.LoadInt32x4Array((*[4]int32)(head[0:4]))
	x1 := archsimd.LoadInt32x4Array((*[4]int32)(head[4:8]))
	x2 := archsimd.LoadInt32x4Array((*[4]int32)(head[8:12]))
	x3 := archsimd.LoadInt32x4Array((*[4]int32)(head[12:16]))
	x4 := archsimd.LoadInt32x4Array((*[4]int32)(head[16:20]))
	q0, d, ds := uint32(20), uint32(0), uint32(dstStride)
	r := 0
	for ; r+2 <= h; r += 2 {
		q := mc8WI8(im, q0)
		x5 := archsimd.LoadInt32x4Array((*[4]int32)(q[0:4]))
		x6 := archsimd.LoadInt32x4Array((*[4]int32)(q[4:8]))
		a, b := mc8V6I32W4(x0, x1, x2, x3, x4, x5, x6, c0, c1, c2, c3, c4, c5, rnd)
		v := a.ToBits().ReshapeToUint16s().ConcatOdd(b.ToBits().ReshapeToUint16s()).BitsToInt16()
		u := v.SaturateToUint8().ReshapeToUint64s().GetElem(0)
		binary.LittleEndian.PutUint32(mc8W4(dst, d)[:], uint32(u))
		binary.LittleEndian.PutUint32(mc8W4(dst, d+ds)[:], uint32(u>>32))
		q0, d = q0+8, d+2*ds
		x0, x1, x2, x3, x4 = x2, x3, x4, x5, x6
	}
	if r < h {
		x5 := archsimd.LoadInt32x4Array(mc8WI4(im, q0))
		a, _ := mc8V6I32W4(x0, x1, x2, x3, x4, x5, x5, c0, c1, c2, c3, c4, c5, rnd)
		v := a.ToBits().ReshapeToUint16s().ConcatOdd(a.ToBits().ReshapeToUint16s()).BitsToInt16()
		binary.LittleEndian.PutUint32(mc8W4(dst, d)[:], uint32(v.SaturateToUint8().ReshapeToUint64s().GetElem(0)))
	}
}

// mc8Store4U16 stores the four uint16 lanes packed in x.
func mc8Store4U16(o *[4]uint16, x uint64) {
	o[0], o[1], o[2], o[3] = uint16(x), uint16(x>>16), uint16(x>>32), uint16(x>>48)
}

// mc8PrepV6FromI32W4 is the width-4 prep 2D vertical pass; two CONV_BUF rows
// are one 16-byte store.
func mc8PrepV6FromI32W4(im []int32, h int, ph *mc8Phase, out []uint16, rndConst int32) {
	c0 := archsimd.LoadInt32x4Array(&ph.p[0])
	c1 := archsimd.LoadInt32x4Array(&ph.p[1])
	c2 := archsimd.LoadInt32x4Array(&ph.p[2])
	c3 := archsimd.LoadInt32x4Array(&ph.p[3])
	c4 := archsimd.LoadInt32x4Array(&ph.p[4])
	c5 := archsimd.LoadInt32x4Array(&ph.p[5])
	rnd := archsimd.BroadcastInt32x4(rndConst)
	head := (*[20]int32)(im[0:20])
	x0 := archsimd.LoadInt32x4Array((*[4]int32)(head[0:4]))
	x1 := archsimd.LoadInt32x4Array((*[4]int32)(head[4:8]))
	x2 := archsimd.LoadInt32x4Array((*[4]int32)(head[8:12]))
	x3 := archsimd.LoadInt32x4Array((*[4]int32)(head[12:16]))
	x4 := archsimd.LoadInt32x4Array((*[4]int32)(head[16:20]))
	q0 := uint32(20)
	r := 0
	for ; r+2 <= h; r += 2 {
		q := mc8WI8(im, q0)
		x5 := archsimd.LoadInt32x4Array((*[4]int32)(q[0:4]))
		x6 := archsimd.LoadInt32x4Array((*[4]int32)(q[4:8]))
		a, b := mc8V6I32W4(x0, x1, x2, x3, x4, x5, x6, c0, c1, c2, c3, c4, c5, rnd)
		a.ToBits().ReshapeToUint16s().ConcatOdd(b.ToBits().ReshapeToUint16s()).StoreArray(mc8WU8(out, q0-20))
		q0 += 8
		x0, x1, x2, x3, x4 = x2, x3, x4, x5, x6
	}
	if r < h {
		x5 := archsimd.LoadInt32x4Array(mc8WI4(im, q0))
		a, _ := mc8V6I32W4(x0, x1, x2, x3, x4, x5, x5, c0, c1, c2, c3, c4, c5, rnd)
		v := a.ToBits().ReshapeToUint16s().ConcatOdd(a.ToBits().ReshapeToUint16s())
		mc8Store4U16(mc8WU4(out, q0-20), v.ReshapeToUint64s().GetElem(0))
	}
}

// mc8PutH6W4 is the width-4 put horizontal-only pass (two rows per
// iteration; an odd last row reruns the chain on itself).
func mc8PutH6W4(src []byte, srcStride, h int, ph *mc8Phase, dst []byte, dstStride int) {
	c0 := archsimd.LoadInt16x8Array(&ph.h[0])
	c1 := archsimd.LoadInt16x8Array(&ph.h[1])
	c2 := archsimd.LoadInt16x8Array(&ph.h[2])
	c3 := archsimd.LoadInt16x8Array(&ph.h[3])
	c4 := archsimd.LoadInt16x8Array(&ph.h[4])
	c5 := archsimd.LoadInt16x8Array(&ph.h[5])
	rnd := archsimd.BroadcastInt16x8(34)
	sh := archsimd.BroadcastInt16x8(-6)
	ss := uint32(srcStride)
	var so, o uint32
	for r := 0; r < h; r += 2 {
		step := ss
		if r+2 > h {
			step = 0
		}
		lo, hi := mc8LoadWiden(mc8W16(src, so))
		lo1, hi1 := mc8LoadWiden(mc8W16(src, so+step))
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
		binary.LittleEndian.PutUint32(mc8W4(dst, o)[:], uint32(sum.Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0)))
		if step != 0 {
			binary.LittleEndian.PutUint32(mc8W4(dst, o+uint32(dstStride))[:], uint32(sum1.Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0)))
		}
		so, o = so+2*ss, o+2*uint32(dstStride)
	}
}

// mc8PrepH6W4 is the width-4 prep horizontal-only pass; two CONV_BUF rows are
// one 16-byte store.
func mc8PrepH6W4(src []byte, srcStride, h int, ph *mc8Phase, out []uint16) {
	c0 := archsimd.LoadInt16x8Array(&ph.h[0])
	c1 := archsimd.LoadInt16x8Array(&ph.h[1])
	c2 := archsimd.LoadInt16x8Array(&ph.h[2])
	c3 := archsimd.LoadInt16x8Array(&ph.h[3])
	c4 := archsimd.LoadInt16x8Array(&ph.h[4])
	c5 := archsimd.LoadInt16x8Array(&ph.h[5])
	rnd := archsimd.BroadcastInt16x8(2 + 6144<<2)
	sh := archsimd.BroadcastInt16x8(-2)
	ss := uint32(srcStride)
	var so, o uint32
	for r := 0; r < h; r += 2 {
		step := ss
		if r+2 > h {
			step = 0
		}
		lo, hi := mc8LoadWiden(mc8W16(src, so))
		lo1, hi1 := mc8LoadWiden(mc8W16(src, so+step))
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
			mc8Store4U16(mc8WU4(out, o), a.GetElem(0))
			break
		}
		a.InterleaveLo(sum1.ToBits().Shift(sh).ReshapeToUint64s()).ReshapeToUint16s().StoreArray(mc8WU8(out, o))
		so, o = so+2*ss, o+8
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
func mc8PutV6W4(src []byte, srcStride, h int, ph *mc8Phase, dst []byte, dstStride int) {
	c0 := archsimd.LoadInt16x8Array(&ph.h[0])
	c1 := archsimd.LoadInt16x8Array(&ph.h[1])
	c2 := archsimd.LoadInt16x8Array(&ph.h[2])
	c3 := archsimd.LoadInt16x8Array(&ph.h[3])
	c4 := archsimd.LoadInt16x8Array(&ph.h[4])
	c5 := archsimd.LoadInt16x8Array(&ph.h[5])
	rnd := archsimd.BroadcastInt16x8(32)
	sh := archsimd.BroadcastInt16x8(-6)
	ss := uint32(srcStride)
	r0 := mc8LoadRow16(mc8W16(src, 0))
	r1 := mc8LoadRow16(mc8W16(src, ss))
	r2 := mc8LoadRow16(mc8W16(src, 2*ss))
	r3 := mc8LoadRow16(mc8W16(src, 3*ss))
	so := 4 * ss
	r4 := mc8LoadRow16(mc8W16(src, so))
	var o uint32
	r := 0
	for ; r+2 <= h; r += 2 {
		so += ss
		r5 := mc8LoadRow16(mc8W16(src, so))
		so += ss
		r6 := mc8LoadRow16(mc8W16(src, so))
		a, b := mc8V6W4Sums(r0, r1, r2, r3, r4, r5, r6, c0, c1, c2, c3, c4, c5, rnd)
		binary.LittleEndian.PutUint32(mc8W4(dst, o)[:], uint32(a.Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0)))
		binary.LittleEndian.PutUint32(mc8W4(dst, o+uint32(dstStride))[:], uint32(b.Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0)))
		o += 2 * uint32(dstStride)
		r0, r1, r2, r3, r4 = r2, r3, r4, r5, r6
	}
	if r < h {
		r5 := mc8LoadRow16(mc8W16(src, so+ss))
		a, _ := mc8V6W4Sums(r0, r1, r2, r3, r4, r5, r5, c0, c1, c2, c3, c4, c5, rnd)
		binary.LittleEndian.PutUint32(mc8W4(dst, o)[:], uint32(a.Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0)))
	}
}

// mc8PrepV6W4 is the width-4 prep vertical-only pass; two CONV_BUF rows are one
// 16-byte store.
func mc8PrepV6W4(src []byte, srcStride, h int, ph *mc8Phase, out []uint16) {
	c0 := archsimd.LoadInt16x8Array(&ph.h[0])
	c1 := archsimd.LoadInt16x8Array(&ph.h[1])
	c2 := archsimd.LoadInt16x8Array(&ph.h[2])
	c3 := archsimd.LoadInt16x8Array(&ph.h[3])
	c4 := archsimd.LoadInt16x8Array(&ph.h[4])
	c5 := archsimd.LoadInt16x8Array(&ph.h[5])
	rnd := archsimd.BroadcastInt16x8(2 + 6144<<2)
	sh := archsimd.BroadcastInt16x8(-2)
	ss := uint32(srcStride)
	r0 := mc8LoadRow16(mc8W16(src, 0))
	r1 := mc8LoadRow16(mc8W16(src, ss))
	r2 := mc8LoadRow16(mc8W16(src, 2*ss))
	r3 := mc8LoadRow16(mc8W16(src, 3*ss))
	so := 4 * ss
	r4 := mc8LoadRow16(mc8W16(src, so))
	var o uint32
	r := 0
	for ; r+2 <= h; r += 2 {
		so += ss
		r5 := mc8LoadRow16(mc8W16(src, so))
		so += ss
		r6 := mc8LoadRow16(mc8W16(src, so))
		a, b := mc8V6W4Sums(r0, r1, r2, r3, r4, r5, r6, c0, c1, c2, c3, c4, c5, rnd)
		a.ToBits().Shift(sh).ReshapeToUint64s().InterleaveLo(b.ToBits().Shift(sh).ReshapeToUint64s()).ReshapeToUint16s().StoreArray(mc8WU8(out, o))
		o += 8
		r0, r1, r2, r3, r4 = r2, r3, r4, r5, r6
	}
	if r < h {
		r5 := mc8LoadRow16(mc8W16(src, so+ss))
		a, _ := mc8V6W4Sums(r0, r1, r2, r3, r4, r5, r5, c0, c1, c2, c3, c4, c5, rnd)
		mc8Store4U16(mc8WU4(out, o), a.ToBits().Shift(sh).ReshapeToUint64s().GetElem(0))
	}
}

// mc8PrepCopyW4 is the width-4 prep copy: 16 * px + 6144, two rows per store.
func mc8PrepCopyW4(src []byte, srcStride, h int, out []uint16) {
	mul := archsimd.BroadcastUint16x8(16)
	off := archsimd.BroadcastUint16x8(6144)
	ss := uint32(srcStride)
	var so, o uint32
	r := 0
	for ; r+2 <= h; r += 2 {
		a := uint64(binary.LittleEndian.Uint32(mc8W4(src, so)[:])) | uint64(binary.LittleEndian.Uint32(mc8W4(src, so+ss)[:]))<<32
		v := archsimd.BroadcastUint64x2(a).ReshapeToUint8s().ExtendLo8ToUint16().MulAdd(mul, off)
		v.StoreArray(mc8WU8(out, o))
		so, o = so+2*ss, o+8
	}
	if r < h {
		v := archsimd.BroadcastUint64x2(uint64(binary.LittleEndian.Uint32(mc8W4(src, so)[:]))).ReshapeToUint8s().ExtendLo8ToUint16().MulAdd(mul, off)
		mc8Store4U16(mc8WU4(out, o), v.ReshapeToUint64s().GetElem(0))
	}
}
