// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package motion

import (
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
	st := im[: (h+5)*4 : (h+5)*4]
	head := (*[20]int32)(st[0:20])
	x0 := archsimd.LoadInt32x4Array((*[4]int32)(head[0:4]))
	x1 := archsimd.LoadInt32x4Array((*[4]int32)(head[4:8]))
	x2 := archsimd.LoadInt32x4Array((*[4]int32)(head[8:12]))
	x3 := archsimd.LoadInt32x4Array((*[4]int32)(head[12:16]))
	x4 := archsimd.LoadInt32x4Array((*[4]int32)(head[16:20]))
	ot := out[: h*4 : h*4]
	q := 20
	for ; q <= len(st)-8; q += 8 {
		win := (*[8]int32)(st[q : q+8])
		x5 := archsimd.LoadInt32x4Array((*[4]int32)(win[0:4]))
		x6 := archsimd.LoadInt32x4Array((*[4]int32)(win[4:8]))
		a, b := mc8V6I32W4(x0, x1, x2, x3, x4, x5, x6, c0, c1, c2, c3, c4, c5, rnd)
		a.ToBits().ReshapeToUint16s().ConcatOdd(b.ToBits().ReshapeToUint16s()).StoreArray((*[8]uint16)(ot[q-20 : q-12]))
		x0, x1, x2, x3, x4 = x2, x3, x4, x5, x6
	}
	if q <= len(st)-4 {
		x5 := archsimd.LoadInt32x4Array((*[4]int32)(st[q : q+4]))
		a, _ := mc8V6I32W4(x0, x1, x2, x3, x4, x5, x5, c0, c1, c2, c3, c4, c5, rnd)
		v := a.ToBits().ReshapeToUint16s().ConcatOdd(a.ToBits().ReshapeToUint16s())
		mc8Store4U16((*[4]uint16)(ot[q-20:q-16]), v.ReshapeToUint64s().GetElem(0))
	}
}
