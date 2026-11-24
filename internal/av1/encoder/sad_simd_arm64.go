// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

// NEON primitives for the Go-native-SIMD SAD kernels. The shape composition
// (block widths, chunking, slice checks) is shared with amd64 in sad_simd.go;
// this file supplies the 16-byte row accumulate and the horizontal reduction.
// arm64 has no VPSADBW equivalent in archsimd, so absolute differences are
// widened to 16-bit lanes and accumulated there.

package encoder

import "simd/archsimd"

// absDiffU8x16 computes unsigned byte absolute differences using only the
// public archsimd API. The maximum is never below the minimum, so subtraction
// is exact without saturation.
func absDiffU8x16(a, b archsimd.Uint8x16) archsimd.Uint8x16 {
	return a.Max(b).Sub(a.Min(b))
}

// widen16 widens one 16-byte abs-diff vector to a Uint16x8 by adding its low
// and high halves lane-wise.
func widen16(absd archsimd.Uint8x16) archsimd.Uint16x8 {
	return absd.ExtendLo8ToUint16().Add(absd.HiToLo().ExtendLo8ToUint16())
}

// absAcc16 adds the absolute differences of one 16-byte row pair into acc. The
// row's widened sum is formed before it joins acc, so the loop-carried chain is
// one add per row. Each Uint16 lane gains at most 2*255 per call, so a lane is
// exact for up to 128 calls; every caller accumulates at most 64 rows.
func absAcc16(acc archsimd.Uint16x8, s, r archsimd.Uint8x16) archsimd.Uint16x8 {
	return acc.Add(widen16(absDiffU8x16(s, r)))
}

// sumU16 returns the exact sum of the eight lanes of acc. The lanes are widened
// to 32 bits after pairwise addition so the total cannot wrap. Each lane is at
// most 2*32*255 for the largest caller, so a pair fits in uint16.
func sumU16(acc archsimd.Uint16x8) int {
	return int(acc.ConcatAddPairs(acc).ExtendLo4ToUint32().ReduceSum())
}

// le64 is the little-endian uint64 of the eight bytes of b; the compiler
// combines the byte loads into one 8-byte load.
func le64(b *[8]byte) uint64 {
	return uint64(b[0]) | uint64(b[1])<<8 | uint64(b[2])<<16 | uint64(b[3])<<24 |
		uint64(b[4])<<32 | uint64(b[5])<<40 | uint64(b[6])<<48 | uint64(b[7])<<56
}

// pack2 packs the 8-byte rows a (lanes 0..7) and b (lanes 8..15) into one
// vector, through a stack store and a vector load (cheaper on Apple cores
// than two general-register lane inserts).
func pack2(a, b *[8]byte) archsimd.Uint8x16 {
	rows := [2]uint64{le64(a), le64(b)}
	return archsimd.LoadUint64x2Array(&rows).ReshapeToUint8s()
}

// The single-reference kernels take one bounds-checked window per row pair:
// w := s[o : o+stride+n : o+stride+n] holds row 2i at w[:n] and row 2i+1 at
// w[stride:], and the compiler proves both from the one window compare (its
// length folds to stride+n). Strides are bounded by rowStrideMax and row
// counts by rowsMax, so the offsets cannot overflow.

// sad8x8Rows is the 8x8 SAD with independent strides. A candidate's maximum
// total is 64*255 = 16,320, so the uint16 lane reduction is exact.
func sad8x8Rows(src []byte, srcStride int, ref []byte, refStride int) int {
	if uint(srcStride) > rowStrideMax || uint(refStride) > rowStrideMax {
		panic(errKernelArgs)
	}
	sn, rn := srcStride+8, refStride+8
	acc := archsimd.BroadcastUint16x8(0)
	for i := 0; i < 4; i++ {
		so, ro := 2*i*srcStride, 2*i*refStride
		sw := src[so : so+sn : so+sn]
		rw := ref[ro : ro+rn : ro+rn]
		s := pack2((*[8]byte)(sw), (*[8]byte)(sw[srcStride:]))
		r := pack2((*[8]byte)(rw), (*[8]byte)(rw[refStride:]))
		acc = acc.Add(widen16(absDiffU8x16(s, r)))
	}
	return int(acc.ReduceSum())
}

// sad8x8x4Rows computes four 8x8 SADs of one source block against four
// reference origins sharing one stride. Each source row pair is packed once
// and reused across all four references.
func sad8x8x4Rows(src, ref0, ref1, ref2, ref3 []byte, stride int) (int, int, int, int) {
	if uint(stride) > rowStrideMax {
		panic(errKernelArgs)
	}
	n := stride + 8
	c0 := archsimd.BroadcastUint16x8(0)
	c1 := archsimd.BroadcastUint16x8(0)
	c2 := archsimd.BroadcastUint16x8(0)
	c3 := archsimd.BroadcastUint16x8(0)
	for i := 0; i < 4; i++ {
		o := 2 * i * stride
		w := src[o : o+n : o+n]
		s := pack2((*[8]byte)(w), (*[8]byte)(w[stride:]))
		w = ref0[o : o+n : o+n]
		c0 = c0.Add(widen16(absDiffU8x16(s, pack2((*[8]byte)(w), (*[8]byte)(w[stride:])))))
		w = ref1[o : o+n : o+n]
		c1 = c1.Add(widen16(absDiffU8x16(s, pack2((*[8]byte)(w), (*[8]byte)(w[stride:])))))
		w = ref2[o : o+n : o+n]
		c2 = c2.Add(widen16(absDiffU8x16(s, pack2((*[8]byte)(w), (*[8]byte)(w[stride:])))))
		w = ref3[o : o+n : o+n]
		c3 = c3.Add(widen16(absDiffU8x16(s, pack2((*[8]byte)(w), (*[8]byte)(w[stride:])))))
	}
	return int(c0.ReduceSum()), int(c1.ReduceSum()), int(c2.ReduceSum()), int(c3.ReduceSum())
}

// sad8x8x4Step4Rows is sad8x8x4Rows for the reference origins ref+0, ref+4,
// ref+8 and ref+12: one 20-byte reference window per row (two per row-pair
// window) holds all four candidates at constant offsets.
func sad8x8x4Step4Rows(src, ref []byte, stride int) (int, int, int, int) {
	if uint(stride) > rowStrideMax {
		panic(errKernelArgs)
	}
	sn, rn := stride+8, stride+20
	c0 := archsimd.BroadcastUint16x8(0)
	c1 := archsimd.BroadcastUint16x8(0)
	c2 := archsimd.BroadcastUint16x8(0)
	c3 := archsimd.BroadcastUint16x8(0)
	for i := 0; i < 4; i++ {
		o := 2 * i * stride
		w := src[o : o+sn : o+sn]
		s := pack2((*[8]byte)(w), (*[8]byte)(w[stride:]))
		rw := ref[o : o+rn : o+rn]
		r0, r1 := (*[20]byte)(rw), (*[20]byte)(rw[stride:])
		c0 = c0.Add(widen16(absDiffU8x16(s, pack2((*[8]byte)(r0[0:8]), (*[8]byte)(r1[0:8])))))
		c1 = c1.Add(widen16(absDiffU8x16(s, pack2((*[8]byte)(r0[4:12]), (*[8]byte)(r1[4:12])))))
		c2 = c2.Add(widen16(absDiffU8x16(s, pack2((*[8]byte)(r0[8:16]), (*[8]byte)(r1[8:16])))))
		c3 = c3.Add(widen16(absDiffU8x16(s, pack2((*[8]byte)(r0[12:20]), (*[8]byte)(r1[12:20])))))
	}
	return int(c0.ReduceSum()), int(c1.ReduceSum()), int(c2.ReduceSum()), int(c3.ReduceSum())
}

// sad8x8CompoundRows computes SAD(src, round((ref0+ref1)/2)) over one 8x8
// block. URHADD (Average) is exactly (a+b+1)>>1 per byte.
func sad8x8CompoundRows(src []byte, srcStride int, ref0 []byte, s0 int, ref1 []byte, s1 int) int {
	if uint(srcStride) > rowStrideMax || uint(s0) > rowStrideMax || uint(s1) > rowStrideMax {
		panic(errKernelArgs)
	}
	sn, n0, n1 := srcStride+8, s0+8, s1+8
	acc := archsimd.BroadcastUint16x8(0)
	for i := 0; i < 4; i++ {
		so, o0, o1 := 2*i*srcStride, 2*i*s0, 2*i*s1
		sw := src[so : so+sn : so+sn]
		w0 := ref0[o0 : o0+n0 : o0+n0]
		w1 := ref1[o1 : o1+n1 : o1+n1]
		s := pack2((*[8]byte)(sw), (*[8]byte)(sw[srcStride:]))
		pred := pack2((*[8]byte)(w0), (*[8]byte)(w0[s0:])).Average(pack2((*[8]byte)(w1), (*[8]byte)(w1[s1:])))
		acc = acc.Add(widen16(absDiffU8x16(s, pred)))
	}
	return int(acc.ReduceSum())
}

// sad16ColsDual sums the SAD of one 16-column strip over h rows with
// independent strides. h is even (16, 32, or 64 in every caller). Even and
// odd rows feed separate accumulators so the two add chains overlap. Each
// lane gets at most 2*255 per row into its accumulator, so h <= 64 keeps the
// lanes exact.
func sad16ColsDual(src []byte, srcStride int, ref []byte, refStride int, h int) int {
	if uint(srcStride) > rowStrideMax || uint(refStride) > rowStrideMax || uint(h) > rowsMax {
		panic(errKernelArgs)
	}
	sn, rn := srcStride+16, refStride+16
	even := archsimd.BroadcastUint16x8(0)
	odd := archsimd.BroadcastUint16x8(0)
	for i := 0; i < h/2; i++ {
		so, ro := 2*i*srcStride, 2*i*refStride
		sw := src[so : so+sn : so+sn]
		rw := ref[ro : ro+rn : ro+rn]
		// absAcc16 written out: its nested inlined helpers left five
		// inline-mark NOPs per row pair.
		s := archsimd.LoadUint8x16Array((*[16]uint8)(sw))
		r := archsimd.LoadUint8x16Array((*[16]uint8)(rw))
		d := s.Max(r).Sub(s.Min(r))
		even = even.Add(d.ExtendLo8ToUint16().Add(d.HiToLo().ExtendLo8ToUint16()))
		s = archsimd.LoadUint8x16Array((*[16]uint8)(sw[srcStride:]))
		r = archsimd.LoadUint8x16Array((*[16]uint8)(rw[refStride:]))
		d = s.Max(r).Sub(s.Min(r))
		odd = odd.Add(d.ExtendLo8ToUint16().Add(d.HiToLo().ExtendLo8ToUint16()))
	}
	if h == 16 {
		// 16*16*255 = 65,280, so the horizontal uint16 sum is exact.
		return int(even.Add(odd).ReduceSum())
	}
	if h == 32 {
		// Each accumulator holds 16 rows and fits independently in uint16.
		return int(even.ReduceSum()) + int(odd.ReduceSum())
	}
	return sumU16(even) + sumU16(odd)
}

// sadAcc16 adds the absolute differences of the 16-byte rows s and r to acc,
// widened to uint16 lanes (each lane gains at most 2*255).
func sadAcc16(acc archsimd.Uint16x8, s, r *[16]uint8) archsimd.Uint16x8 {
	a, b := archsimd.LoadUint8x16Array(s), archsimd.LoadUint8x16Array(r)
	d := a.Max(b).Sub(a.Min(b))
	return acc.Add(d.ExtendLo8ToUint16().Add(d.HiToLo().ExtendLo8ToUint16()))
}

// sad32Rows is the 32x32 SAD with independent strides, one 32-byte window
// per row pair and array. Each of the four accumulators (column half by row
// parity) holds 16 rows, so its horizontal uint16 sum is exact.
func sad32Rows(src []byte, srcStride int, ref []byte, refStride int) int {
	if uint(srcStride) > rowStrideMax || uint(refStride) > rowStrideMax {
		panic(errKernelArgs)
	}
	sn, rn := srcStride+32, refStride+32
	var le, lo, re, ro archsimd.Uint16x8
	for i := 0; i < 16; i++ {
		so, oo := 2*i*srcStride, 2*i*refStride
		sw := src[so : so+sn : so+sn]
		rw := ref[oo : oo+rn : oo+rn]
		s0, r0 := (*[32]uint8)(sw), (*[32]uint8)(rw)
		s1, r1 := (*[32]uint8)(sw[srcStride:]), (*[32]uint8)(rw[refStride:])
		le = sadAcc16(le, (*[16]uint8)(s0[0:16]), (*[16]uint8)(r0[0:16]))
		re = sadAcc16(re, (*[16]uint8)(s0[16:32]), (*[16]uint8)(r0[16:32]))
		lo = sadAcc16(lo, (*[16]uint8)(s1[0:16]), (*[16]uint8)(r1[0:16]))
		ro = sadAcc16(ro, (*[16]uint8)(s1[16:32]), (*[16]uint8)(r1[16:32]))
	}
	return int(le.ReduceSum()) + int(lo.ReduceSum()) + int(re.ReduceSum()) + int(ro.ReduceSum())
}

// sad64Rows is the 64x64 SAD with independent strides, one 64-byte window
// per row pair and array. Each of the eight accumulators (16-column strip by
// row parity) holds 32 rows: a lane is at most 32*2*255 = 16,320, so the
// pairwise sumU16 reduction is exact.
func sad64Rows(src []byte, srcStride int, ref []byte, refStride int) int {
	if uint(srcStride) > rowStrideMax || uint(refStride) > rowStrideMax {
		panic(errKernelArgs)
	}
	sn, rn := srcStride+64, refStride+64
	var a0, a1, a2, a3, b0, b1, b2, b3 archsimd.Uint16x8
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
		b0 = sadAcc16(b0, (*[16]uint8)(s1[0:16]), (*[16]uint8)(r1[0:16]))
		b1 = sadAcc16(b1, (*[16]uint8)(s1[16:32]), (*[16]uint8)(r1[16:32]))
		b2 = sadAcc16(b2, (*[16]uint8)(s1[32:48]), (*[16]uint8)(r1[32:48]))
		b3 = sadAcc16(b3, (*[16]uint8)(s1[48:64]), (*[16]uint8)(r1[48:64]))
	}
	return sumU16(a0) + sumU16(a1) + sumU16(a2) + sumU16(a3) +
		sumU16(b0) + sumU16(b1) + sumU16(b2) + sumU16(b3)
}

// sad16x16Same is the 16x16 SAD of two blocks sharing one stride: one
// offset per row pair serves both windows.
func sad16x16Same(src, ref []byte, stride int) int {
	if uint(stride) > rowStrideMax {
		panic(errKernelArgs)
	}
	n := stride + 16
	even := archsimd.BroadcastUint16x8(0)
	odd := archsimd.BroadcastUint16x8(0)
	for i := 0; i < 8; i++ {
		o := 2 * i * stride
		sw := src[o : o+n : o+n]
		rw := ref[o : o+n : o+n]
		s := archsimd.LoadUint8x16Array((*[16]uint8)(sw))
		r := archsimd.LoadUint8x16Array((*[16]uint8)(rw))
		d := s.Max(r).Sub(s.Min(r))
		even = even.Add(d.ExtendLo8ToUint16().Add(d.HiToLo().ExtendLo8ToUint16()))
		s = archsimd.LoadUint8x16Array((*[16]uint8)(sw[stride:]))
		r = archsimd.LoadUint8x16Array((*[16]uint8)(rw[stride:]))
		d = s.Max(r).Sub(s.Min(r))
		odd = odd.Add(d.ExtendLo8ToUint16().Add(d.HiToLo().ExtendLo8ToUint16()))
	}
	// 16*16*255 = 65,280, so the horizontal uint16 sum is exact.
	return int(even.Add(odd).ReduceSum())
}

// sad16x16Packed is the 16x16 SAD against a packed 16-stride reference
// block (the prediction scratch of the near-MV and compound checks): the
// reference rows sit at constant offsets in one array, so only the source
// pays a window per row pair.
func sad16x16Packed(src []byte, srcStride int, ref *[256]uint8) int {
	if uint(srcStride) > rowStrideMax {
		panic(errKernelArgs)
	}
	n := srcStride + 16
	even := archsimd.BroadcastUint16x8(0)
	odd := archsimd.BroadcastUint16x8(0)
	for i := 0; i < 8; i++ {
		o := 2 * i * srcStride
		sw := src[o : o+n : o+n]
		rw := (*[32]uint8)(ref[32*i : 32*i+32])
		s := archsimd.LoadUint8x16Array((*[16]uint8)(sw))
		r := archsimd.LoadUint8x16Array((*[16]uint8)(rw[0:16]))
		d := s.Max(r).Sub(s.Min(r))
		even = even.Add(d.ExtendLo8ToUint16().Add(d.HiToLo().ExtendLo8ToUint16()))
		s = archsimd.LoadUint8x16Array((*[16]uint8)(sw[srcStride:]))
		r = archsimd.LoadUint8x16Array((*[16]uint8)(rw[16:32]))
		d = s.Max(r).Sub(s.Min(r))
		odd = odd.Add(d.ExtendLo8ToUint16().Add(d.HiToLo().ExtendLo8ToUint16()))
	}
	return int(even.Add(odd).ReduceSum())
}

// sad8x8Same is the 8x8 SAD of two blocks sharing one stride.
func sad8x8Same(src, ref []byte, stride int) int {
	if uint(stride) > rowStrideMax {
		panic(errKernelArgs)
	}
	n := stride + 8
	acc := archsimd.BroadcastUint16x8(0)
	for i := 0; i < 4; i++ {
		o := 2 * i * stride
		sw := src[o : o+n : o+n]
		rw := ref[o : o+n : o+n]
		s := pack2((*[8]byte)(sw), (*[8]byte)(sw[stride:]))
		r := pack2((*[8]byte)(rw), (*[8]byte)(rw[stride:]))
		acc = acc.Add(widen16(absDiffU8x16(s, r)))
	}
	return int(acc.ReduceSum())
}

// sad16ColsX4 is sad16ColsDual for four references sharing one stride, as
// bounds-checked row windows (this shape measured neutral against the
// pointer form). Each source row is loaded once and reused across the four
// candidates. With stride and h checked against rowStrideMax and rowsMax,
// each row window is one bounds check.
func sad16ColsX4(src, ref0, ref1, ref2, ref3 []byte, stride int, h int) (int, int, int, int) {
	if uint(stride) > rowStrideMax || uint(h) > rowsMax {
		panic(errKernelArgs)
	}
	a0 := archsimd.BroadcastUint16x8(0)
	a1 := archsimd.BroadcastUint16x8(0)
	a2 := archsimd.BroadcastUint16x8(0)
	a3 := archsimd.BroadcastUint16x8(0)
	for row := 0; row < h; row++ {
		o := row * stride
		s := archsimd.LoadUint8x16Array((*[16]uint8)(src[o : o+16]))
		a0 = absAcc16(a0, s, archsimd.LoadUint8x16Array((*[16]uint8)(ref0[o:o+16])))
		a1 = absAcc16(a1, s, archsimd.LoadUint8x16Array((*[16]uint8)(ref1[o:o+16])))
		a2 = absAcc16(a2, s, archsimd.LoadUint8x16Array((*[16]uint8)(ref2[o:o+16])))
		a3 = absAcc16(a3, s, archsimd.LoadUint8x16Array((*[16]uint8)(ref3[o:o+16])))
	}
	if h == 16 {
		return int(a0.ReduceSum()), int(a1.ReduceSum()), int(a2.ReduceSum()), int(a3.ReduceSum())
	}
	return sumU16(a0), sumU16(a1), sumU16(a2), sumU16(a3)
}
