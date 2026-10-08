// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && amd64 && !purego

// AVX2-host primitives for the Go-native-SIMD SAD kernels. The shape
// composition (block widths, chunking, slice checks) is shared with arm64 in
// sad_simd.go. SumOf8AbsDiff lowers to VPSADBW, which sums absolute byte
// differences per 8-byte group into 64-bit lanes, so no lane can overflow.

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

// sumU64 returns the exact sum of the two 64-bit lanes of acc.
func sumU64(acc archsimd.Uint64x2) int {
	return int(acc.GetElem(0) + acc.GetElem(1))
}

// sad8x8Ptr is the 8x8 SAD with independent strides.
func sad8x8Ptr(sp unsafe.Pointer, srcStride int, rp unsafe.Pointer, refStride int) int {
	acc := archsimd.BroadcastUint64x2(0)
	for row := 0; row < 8; row += 2 {
		s := pack2Rows(step(sp, row*srcStride), srcStride)
		r := pack2Rows(step(rp, row*refStride), refStride)
		acc = acc.Add(s.SumOf8AbsDiff(r))
	}
	return sumU64(acc)
}

// sad8x8x4Ptr computes four 8x8 SADs of one source block against four
// reference origins sharing one stride. Each source row pair is packed once
// and reused across all four references.
func sad8x8x4Ptr(sp, p0, p1, p2, p3 unsafe.Pointer, stride int) (int, int, int, int) {
	c0 := archsimd.BroadcastUint64x2(0)
	c1 := archsimd.BroadcastUint64x2(0)
	c2 := archsimd.BroadcastUint64x2(0)
	c3 := archsimd.BroadcastUint64x2(0)
	for row := 0; row < 8; row += 2 {
		o := row * stride
		s := pack2Rows(step(sp, o), stride)
		c0 = c0.Add(s.SumOf8AbsDiff(pack2Rows(step(p0, o), stride)))
		c1 = c1.Add(s.SumOf8AbsDiff(pack2Rows(step(p1, o), stride)))
		c2 = c2.Add(s.SumOf8AbsDiff(pack2Rows(step(p2, o), stride)))
		c3 = c3.Add(s.SumOf8AbsDiff(pack2Rows(step(p3, o), stride)))
	}
	return sumU64(c0), sumU64(c1), sumU64(c2), sumU64(c3)
}

// sad8x8CompoundPtr computes SAD(src, round((ref0+ref1)/2)) over one 8x8 block.
// PAVGB (Average) is exactly (a+b+1)>>1 per byte.
func sad8x8CompoundPtr(sp unsafe.Pointer, srcStride int, p0 unsafe.Pointer, s0 int, p1 unsafe.Pointer, s1 int) int {
	acc := archsimd.BroadcastUint64x2(0)
	for row := 0; row < 8; row += 2 {
		s := pack2Rows(step(sp, row*srcStride), srcStride)
		pred := pack2Rows(step(p0, row*s0), s0).Average(pack2Rows(step(p1, row*s1), s1))
		acc = acc.Add(s.SumOf8AbsDiff(pred))
	}
	return sumU64(acc)
}

// sad16ColsDualPtr sums the SAD of one 16-column strip over h rows with
// independent strides.
func sad16ColsDualPtr(sp unsafe.Pointer, srcStride int, rp unsafe.Pointer, refStride int, h int) int {
	acc := archsimd.BroadcastUint64x2(0)
	for row := 0; row < h; row++ {
		acc = acc.Add(load16(step(sp, row*srcStride)).SumOf8AbsDiff(load16(step(rp, row*refStride))))
	}
	return sumU64(acc)
}

// sad16ColsX4Ptr is sad16ColsDualPtr for four references sharing one stride.
// Each source row is loaded once and reused across the four candidates.
func sad16ColsX4Ptr(sp, p0, p1, p2, p3 unsafe.Pointer, stride int, h int) (int, int, int, int) {
	a0 := archsimd.BroadcastUint64x2(0)
	a1 := archsimd.BroadcastUint64x2(0)
	a2 := archsimd.BroadcastUint64x2(0)
	a3 := archsimd.BroadcastUint64x2(0)
	for row := 0; row < h; row++ {
		o := row * stride
		s := load16(step(sp, o))
		a0 = a0.Add(s.SumOf8AbsDiff(load16(step(p0, o))))
		a1 = a1.Add(s.SumOf8AbsDiff(load16(step(p1, o))))
		a2 = a2.Add(s.SumOf8AbsDiff(load16(step(p2, o))))
		a3 = a3.Add(s.SumOf8AbsDiff(load16(step(p3, o))))
	}
	return sumU64(a0), sumU64(a1), sumU64(a2), sumU64(a3)
}
