// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && amd64 && !purego

// AVX2-host primitives for the Go-native-SIMD SAD kernels. The shape
// composition (block widths, chunking, slice checks) is shared with arm64 in
// sad_simd.go. SumOf8AbsDiff lowers to VPSADBW, which sums absolute byte
// differences per 8-byte group into 64-bit lanes, so no lane can overflow.
//
// The kernels mirror sad_simd_arm64.go: one bounds-checked window per row
// pair and array, w := s[o : o+stride+n : o+stride+n], with row 2i at w[:n]
// and row 2i+1 at w[stride:].

package encoder

import (
	"simd/archsimd"
	"unsafe"
)

// step advances a raw byte pointer by n bytes.
func step(p unsafe.Pointer, n int) unsafe.Pointer { return unsafe.Add(p, n) }

// load16 loads the 16 bytes at p. Callers guarantee they are readable.
func load16(p unsafe.Pointer) archsimd.Uint8x16 {
	return archsimd.LoadUint8x16Array((*[16]uint8)(p))
}

// pack2Rows loads two 8-byte rows (at base p and base+stride) into one
// 16-byte vector: lanes 0..7 are the first row and lanes 8..15 are the second.
func pack2Rows(p unsafe.Pointer, stride int) archsimd.Uint8x16 {
	lo := *(*uint64)(p)
	hi := *(*uint64)(step(p, stride))
	return archsimd.BroadcastUint64x2(lo).SetElem(1, hi).ReshapeToUint8s()
}

// le64 is the little-endian uint64 of the eight bytes of b; the compiler
// combines the byte loads into one 8-byte load.
func le64(b *[8]byte) uint64 {
	return uint64(b[0]) | uint64(b[1])<<8 | uint64(b[2])<<16 | uint64(b[3])<<24 |
		uint64(b[4])<<32 | uint64(b[5])<<40 | uint64(b[6])<<48 | uint64(b[7])<<56
}

// pack2 packs the 8-byte rows a (lanes 0..7) and b (lanes 8..15) into one
// vector.
func pack2(a, b *[8]byte) archsimd.Uint8x16 {
	return archsimd.BroadcastUint64x2(le64(a)).SetElem(1, le64(b)).ReshapeToUint8s()
}

// sumU64 returns the exact sum of the two 64-bit lanes of acc.
func sumU64(acc archsimd.Uint64x2) int {
	return int(acc.GetElem(0) + acc.GetElem(1))
}

// sadAcc16 adds the SAD of the 16-byte rows s and r to acc.
func sadAcc16(acc archsimd.Uint64x2, s, r *[16]uint8) archsimd.Uint64x2 {
	return acc.Add(archsimd.LoadUint8x16Array(s).SumOf8AbsDiff(archsimd.LoadUint8x16Array(r)))
}

// sad8x8Rows is the 8x8 SAD with independent strides.
func sad8x8Rows(src []byte, srcStride int, ref []byte, refStride int) int {
	if uint(srcStride) > rowStrideMax || uint(refStride) > rowStrideMax {
		panic(errKernelArgs)
	}
	sn, rn := srcStride+8, refStride+8
	acc := archsimd.BroadcastUint64x2(0)
	for i := 0; i < 4; i++ {
		so, ro := 2*i*srcStride, 2*i*refStride
		sw := src[so : so+sn : so+sn]
		rw := ref[ro : ro+rn : ro+rn]
		s := pack2((*[8]byte)(sw), (*[8]byte)(sw[srcStride:]))
		r := pack2((*[8]byte)(rw), (*[8]byte)(rw[refStride:]))
		acc = acc.Add(s.SumOf8AbsDiff(r))
	}
	return sumU64(acc)
}

// sad8x8Same is the 8x8 SAD of two blocks sharing one stride.
func sad8x8Same(src, ref []byte, stride int) int {
	if uint(stride) > rowStrideMax {
		panic(errKernelArgs)
	}
	n := stride + 8
	acc := archsimd.BroadcastUint64x2(0)
	for i := 0; i < 4; i++ {
		o := 2 * i * stride
		sw := src[o : o+n : o+n]
		rw := ref[o : o+n : o+n]
		s := pack2((*[8]byte)(sw), (*[8]byte)(sw[stride:]))
		r := pack2((*[8]byte)(rw), (*[8]byte)(rw[stride:]))
		acc = acc.Add(s.SumOf8AbsDiff(r))
	}
	return sumU64(acc)
}

// sad8x8x4Rows computes four 8x8 SADs of one source block against four
// reference origins sharing one stride. Each source row pair is packed once
// and reused across all four references.
func sad8x8x4Rows(src, ref0, ref1, ref2, ref3 []byte, stride int) (int, int, int, int) {
	if uint(stride) > rowStrideMax {
		panic(errKernelArgs)
	}
	n := stride + 8
	c0 := archsimd.BroadcastUint64x2(0)
	c1 := archsimd.BroadcastUint64x2(0)
	c2 := archsimd.BroadcastUint64x2(0)
	c3 := archsimd.BroadcastUint64x2(0)
	for i := 0; i < 4; i++ {
		o := 2 * i * stride
		w := src[o : o+n : o+n]
		s := pack2((*[8]byte)(w), (*[8]byte)(w[stride:]))
		w = ref0[o : o+n : o+n]
		c0 = c0.Add(s.SumOf8AbsDiff(pack2((*[8]byte)(w), (*[8]byte)(w[stride:]))))
		w = ref1[o : o+n : o+n]
		c1 = c1.Add(s.SumOf8AbsDiff(pack2((*[8]byte)(w), (*[8]byte)(w[stride:]))))
		w = ref2[o : o+n : o+n]
		c2 = c2.Add(s.SumOf8AbsDiff(pack2((*[8]byte)(w), (*[8]byte)(w[stride:]))))
		w = ref3[o : o+n : o+n]
		c3 = c3.Add(s.SumOf8AbsDiff(pack2((*[8]byte)(w), (*[8]byte)(w[stride:]))))
	}
	return sumU64(c0), sumU64(c1), sumU64(c2), sumU64(c3)
}

// sad8x8CompoundRows computes SAD(src, round((ref0+ref1)/2)) over one 8x8
// block. PAVGB (Average) is exactly (a+b+1)>>1 per byte.
func sad8x8CompoundRows(src []byte, srcStride int, ref0 []byte, s0 int, ref1 []byte, s1 int) int {
	if uint(srcStride) > rowStrideMax || uint(s0) > rowStrideMax || uint(s1) > rowStrideMax {
		panic(errKernelArgs)
	}
	sn, n0, n1 := srcStride+8, s0+8, s1+8
	acc := archsimd.BroadcastUint64x2(0)
	for i := 0; i < 4; i++ {
		so, o0, o1 := 2*i*srcStride, 2*i*s0, 2*i*s1
		sw := src[so : so+sn : so+sn]
		w0 := ref0[o0 : o0+n0 : o0+n0]
		w1 := ref1[o1 : o1+n1 : o1+n1]
		s := pack2((*[8]byte)(sw), (*[8]byte)(sw[srcStride:]))
		pred := pack2((*[8]byte)(w0), (*[8]byte)(w0[s0:])).Average(pack2((*[8]byte)(w1), (*[8]byte)(w1[s1:])))
		acc = acc.Add(s.SumOf8AbsDiff(pred))
	}
	return sumU64(acc)
}

// sad16ColsDual sums the SAD of one 16-column strip over h rows with
// independent strides. h is even (16, 32, or 64 in every caller).
func sad16ColsDual(src []byte, srcStride int, ref []byte, refStride int, h int) int {
	if uint(srcStride) > rowStrideMax || uint(refStride) > rowStrideMax || uint(h) > rowsMax {
		panic(errKernelArgs)
	}
	sn, rn := srcStride+16, refStride+16
	even := archsimd.BroadcastUint64x2(0)
	odd := archsimd.BroadcastUint64x2(0)
	for i := 0; i < h/2; i++ {
		so, ro := 2*i*srcStride, 2*i*refStride
		sw := src[so : so+sn : so+sn]
		rw := ref[ro : ro+rn : ro+rn]
		even = sadAcc16(even, (*[16]uint8)(sw), (*[16]uint8)(rw))
		odd = sadAcc16(odd, (*[16]uint8)(sw[srcStride:]), (*[16]uint8)(rw[refStride:]))
	}
	return sumU64(even.Add(odd))
}

// sad16x16Same is the 16x16 SAD of two blocks sharing one stride.
func sad16x16Same(src, ref []byte, stride int) int {
	if uint(stride) > rowStrideMax {
		panic(errKernelArgs)
	}
	n := stride + 16
	even := archsimd.BroadcastUint64x2(0)
	odd := archsimd.BroadcastUint64x2(0)
	for i := 0; i < 8; i++ {
		o := 2 * i * stride
		sw := src[o : o+n : o+n]
		rw := ref[o : o+n : o+n]
		even = sadAcc16(even, (*[16]uint8)(sw), (*[16]uint8)(rw))
		odd = sadAcc16(odd, (*[16]uint8)(sw[stride:]), (*[16]uint8)(rw[stride:]))
	}
	return sumU64(even.Add(odd))
}

// sad16x16Packed is the 16x16 SAD against a packed 16-stride reference
// block: the reference rows sit at constant offsets in one array.
func sad16x16Packed(src []byte, srcStride int, ref *[256]uint8) int {
	if uint(srcStride) > rowStrideMax {
		panic(errKernelArgs)
	}
	n := srcStride + 16
	even := archsimd.BroadcastUint64x2(0)
	odd := archsimd.BroadcastUint64x2(0)
	for i := 0; i < 8; i++ {
		o := 2 * i * srcStride
		sw := src[o : o+n : o+n]
		rw := (*[32]uint8)(ref[32*i : 32*i+32])
		even = sadAcc16(even, (*[16]uint8)(sw), (*[16]uint8)(rw[0:16]))
		odd = sadAcc16(odd, (*[16]uint8)(sw[srcStride:]), (*[16]uint8)(rw[16:32]))
	}
	return sumU64(even.Add(odd))
}

// sad32Rows is the 32x32 SAD with independent strides, one 32-byte window
// per row pair and array.
func sad32Rows(src []byte, srcStride int, ref []byte, refStride int) int {
	if uint(srcStride) > rowStrideMax || uint(refStride) > rowStrideMax {
		panic(errKernelArgs)
	}
	sn, rn := srcStride+32, refStride+32
	var a0, a1 archsimd.Uint64x2
	for i := 0; i < 16; i++ {
		so, oo := 2*i*srcStride, 2*i*refStride
		sw := src[so : so+sn : so+sn]
		rw := ref[oo : oo+rn : oo+rn]
		s0, r0 := (*[32]uint8)(sw), (*[32]uint8)(rw)
		s1, r1 := (*[32]uint8)(sw[srcStride:]), (*[32]uint8)(rw[refStride:])
		a0 = sadAcc16(a0, (*[16]uint8)(s0[0:16]), (*[16]uint8)(r0[0:16]))
		a1 = sadAcc16(a1, (*[16]uint8)(s0[16:32]), (*[16]uint8)(r0[16:32]))
		a0 = sadAcc16(a0, (*[16]uint8)(s1[0:16]), (*[16]uint8)(r1[0:16]))
		a1 = sadAcc16(a1, (*[16]uint8)(s1[16:32]), (*[16]uint8)(r1[16:32]))
	}
	return sumU64(a0.Add(a1))
}

// sad64Rows is the 64x64 SAD with independent strides, one 64-byte window
// per row pair and array.
func sad64Rows(src []byte, srcStride int, ref []byte, refStride int) int {
	if uint(srcStride) > rowStrideMax || uint(refStride) > rowStrideMax {
		panic(errKernelArgs)
	}
	sn, rn := srcStride+64, refStride+64
	var a0, a1, a2, a3 archsimd.Uint64x2
	for i := 0; i < 32; i++ {
		so, oo := 2*i*srcStride, 2*i*refStride
		sw := src[so : so+sn : so+sn]
		rw := ref[oo : oo+rn : oo+rn]
		s0, r0 := (*[64]uint8)(sw), (*[64]uint8)(rw)
		s1, r1 := (*[64]uint8)(sw[srcStride:]), (*[64]uint8)(rw[refStride:])
		a0 = sadAcc16(a0, (*[16]uint8)(s0[0:16]), (*[16]uint8)(r0[0:16]))
		a1 = sadAcc16(a1, (*[16]uint8)(s0[16:32]), (*[16]uint8)(r0[16:32]))
		a2 = sadAcc16(a2, (*[16]uint8)(s0[32:48]), (*[16]uint8)(r0[32:48]))
		a3 = sadAcc16(a3, (*[16]uint8)(s0[48:64]), (*[16]uint8)(r0[48:64]))
		a0 = sadAcc16(a0, (*[16]uint8)(s1[0:16]), (*[16]uint8)(r1[0:16]))
		a1 = sadAcc16(a1, (*[16]uint8)(s1[16:32]), (*[16]uint8)(r1[16:32]))
		a2 = sadAcc16(a2, (*[16]uint8)(s1[32:48]), (*[16]uint8)(r1[32:48]))
		a3 = sadAcc16(a3, (*[16]uint8)(s1[48:64]), (*[16]uint8)(r1[48:64]))
	}
	return sumU64(a0.Add(a1).Add(a2.Add(a3)))
}

// sad16ColsX4 is sad16ColsDual for four references sharing one stride,
// as bounds-checked row windows. With stride and h checked against
// rowStrideMax and rowsMax, each row window is one bounds check.
func sad16ColsX4(src, ref0, ref1, ref2, ref3 []byte, stride int, h int) (int, int, int, int) {
	if uint(stride) > rowStrideMax || uint(h) > rowsMax {
		panic(errKernelArgs)
	}
	a0 := archsimd.BroadcastUint64x2(0)
	a1 := archsimd.BroadcastUint64x2(0)
	a2 := archsimd.BroadcastUint64x2(0)
	a3 := archsimd.BroadcastUint64x2(0)
	for row := 0; row < h; row++ {
		o := row * stride
		s := archsimd.LoadUint8x16Array((*[16]uint8)(src[o : o+16]))
		a0 = a0.Add(s.SumOf8AbsDiff(archsimd.LoadUint8x16Array((*[16]uint8)(ref0[o : o+16]))))
		a1 = a1.Add(s.SumOf8AbsDiff(archsimd.LoadUint8x16Array((*[16]uint8)(ref1[o : o+16]))))
		a2 = a2.Add(s.SumOf8AbsDiff(archsimd.LoadUint8x16Array((*[16]uint8)(ref2[o : o+16]))))
		a3 = a3.Add(s.SumOf8AbsDiff(archsimd.LoadUint8x16Array((*[16]uint8)(ref3[o : o+16]))))
	}
	return sumU64(a0), sumU64(a1), sumU64(a2), sumU64(a3)
}
