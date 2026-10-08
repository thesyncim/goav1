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

// pack2Rows loads two 8-byte rows (at base p and base+stride) into one
// 16-byte vector: lanes 0..7 are the first row and lanes 8..15 are the second.
func pack2Rows(p unsafe.Pointer, stride int) archsimd.Uint8x16 {
	rows := [2]uint64{*(*uint64)(p), *(*uint64)(step(p, stride))}
	return archsimd.LoadUint64x2Array(&rows).ReshapeToUint8s()
}

// sad8x8Ptr is the 8x8 SAD with independent strides. A candidate's maximum
// total is 64*255 = 16,320, so the uint16 lane reduction is exact.
func sad8x8Ptr(sp unsafe.Pointer, srcStride int, rp unsafe.Pointer, refStride int) int {
	acc := archsimd.BroadcastUint16x8(0)
	for row := 0; row < 8; row += 2 {
		s := pack2Rows(step(sp, row*srcStride), srcStride)
		r := pack2Rows(step(rp, row*refStride), refStride)
		acc = acc.Add(widen16(absDiffU8x16(s, r)))
	}
	return int(acc.ReduceSum())
}

// sad8x8x4Ptr computes four 8x8 SADs of one source block against four
// reference origins sharing one stride. Each source row pair is packed once
// and reused across all four references.
func sad8x8x4Ptr(sp, p0, p1, p2, p3 unsafe.Pointer, stride int) (int, int, int, int) {
	c0 := archsimd.BroadcastUint16x8(0)
	c1 := archsimd.BroadcastUint16x8(0)
	c2 := archsimd.BroadcastUint16x8(0)
	c3 := archsimd.BroadcastUint16x8(0)
	for row := 0; row < 8; row += 2 {
		// Offsets come from the original bases each time. Advancing raw pointers
		// after the last pair would form an out-of-bounds pointer for an exact
		// 7*stride+8-byte input, even though it is unused.
		o := row * stride
		s := pack2Rows(step(sp, o), stride)
		c0 = c0.Add(widen16(absDiffU8x16(s, pack2Rows(step(p0, o), stride))))
		c1 = c1.Add(widen16(absDiffU8x16(s, pack2Rows(step(p1, o), stride))))
		c2 = c2.Add(widen16(absDiffU8x16(s, pack2Rows(step(p2, o), stride))))
		c3 = c3.Add(widen16(absDiffU8x16(s, pack2Rows(step(p3, o), stride))))
	}
	return int(c0.ReduceSum()), int(c1.ReduceSum()), int(c2.ReduceSum()), int(c3.ReduceSum())
}

// sad8x8CompoundPtr computes SAD(src, round((ref0+ref1)/2)) over one 8x8 block.
// URHADD (Average) is exactly (a+b+1)>>1 per byte.
func sad8x8CompoundPtr(sp unsafe.Pointer, srcStride int, p0 unsafe.Pointer, s0 int, p1 unsafe.Pointer, s1 int) int {
	acc := archsimd.BroadcastUint16x8(0)
	for row := 0; row < 8; row += 2 {
		s := pack2Rows(step(sp, row*srcStride), srcStride)
		pred := pack2Rows(step(p0, row*s0), s0).Average(pack2Rows(step(p1, row*s1), s1))
		acc = acc.Add(widen16(absDiffU8x16(s, pred)))
	}
	return int(acc.ReduceSum())
}

// sad16ColsDualPtr sums the SAD of one 16-column strip over h rows with
// independent strides. h is even (16, 32, or 64 in every caller). Even and odd
// rows feed separate accumulators so the two add chains overlap. Each lane gets
// at most 2*255 per row into its accumulator, so h <= 64 keeps the lanes exact.
func sad16ColsDualPtr(sp unsafe.Pointer, srcStride int, rp unsafe.Pointer, refStride int, h int) int {
	even := archsimd.BroadcastUint16x8(0)
	odd := archsimd.BroadcastUint16x8(0)
	for row := 0; row < h; row += 2 {
		even = absAcc16(even, load16(step(sp, row*srcStride)), load16(step(rp, row*refStride)))
		odd = absAcc16(odd, load16(step(sp, (row+1)*srcStride)), load16(step(rp, (row+1)*refStride)))
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

// sad16ColsX4Ptr is sad16ColsDualPtr for four references sharing one stride.
// Each source row is loaded once and reused across the four candidates.
func sad16ColsX4Ptr(sp, p0, p1, p2, p3 unsafe.Pointer, stride int, h int) (int, int, int, int) {
	a0 := archsimd.BroadcastUint16x8(0)
	a1 := archsimd.BroadcastUint16x8(0)
	a2 := archsimd.BroadcastUint16x8(0)
	a3 := archsimd.BroadcastUint16x8(0)
	for row := 0; row < h; row++ {
		o := row * stride
		s := load16(step(sp, o))
		a0 = absAcc16(a0, s, load16(step(p0, o)))
		a1 = absAcc16(a1, s, load16(step(p1, o)))
		a2 = absAcc16(a2, s, load16(step(p2, o)))
		a3 = absAcc16(a3, s, load16(step(p3, o)))
	}
	if h == 16 {
		return int(a0.ReduceSum()), int(a1.ReduceSum()), int(a2.ReduceSum()), int(a3.ReduceSum())
	}
	return sumU16(a0), sumU16(a1), sumU16(a2), sumU16(a3)
}
