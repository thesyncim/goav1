// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

// Go-native-SIMD 8x8 four-reference SAD for motion estimation. This kernel
// reuses each packed source row pair across four candidates and returns exact
// integer sums matching sad8x8x4PureGo.

package encoder

import (
	"simd/archsimd"
	"unsafe"
)

// step advances a raw byte pointer by n bytes.
func step(p unsafe.Pointer, n int) unsafe.Pointer { return unsafe.Add(p, n) }

// absDiffU8x16 computes unsigned byte absolute differences using only the
// public archsimd API. The two saturating differences are mutually exclusive
// per lane, so OR combines them into the exact absolute difference.
func absDiffU8x16(a, b archsimd.Uint8x16) archsimd.Uint8x16 {
	return a.SubSaturated(b).Or(b.SubSaturated(a))
}

// widen16 widens one 16-byte abs-diff vector to a Uint16x8 by adding its low
// and high halves lane-wise.
func widen16(absd archsimd.Uint8x16) archsimd.Uint16x8 {
	return absd.ExtendLo8ToUint16().Add(absd.HiToLo().ExtendLo8ToUint16())
}

// pack2Rows loads two 8-byte rows (at base p and base+stride) into one
// 16-byte vector: lanes 0..7 are the first row and lanes 8..15 are the second.
func pack2Rows(p unsafe.Pointer, stride int) archsimd.Uint8x16 {
	lo := *(*uint64)(p)
	hi := *(*uint64)(step(p, stride))
	return archsimd.BroadcastUint64x2(lo).SetElem(1, hi).ReshapeToUint8s()
}

// sad8x8x4SIMD computes four 8x8 SADs of one source block against four
// independent reference origins. Each source row pair is packed once and
// reused across all four references. A candidate's maximum total is
// 64*255 = 16,320, so the final uint16 lane reduction is exact.
func sad8x8x4SIMD(src, ref0, ref1, ref2, ref3 []byte, stride int) (int, int, int, int) {
	sp := unsafe.Pointer(&src[0])
	p0 := unsafe.Pointer(&ref0[0])
	p1 := unsafe.Pointer(&ref1[0])
	p2 := unsafe.Pointer(&ref2[0])
	p3 := unsafe.Pointer(&ref3[0])
	c0 := archsimd.BroadcastUint16x8(0)
	c1 := archsimd.BroadcastUint16x8(0)
	c2 := archsimd.BroadcastUint16x8(0)
	c3 := archsimd.BroadcastUint16x8(0)
	stride2 := 2 * stride
	for row, rowOffset := 0, 0; row < 8; row, rowOffset = row+2, rowOffset+stride2 {
		// Derive each row-pair pointer from its original slice base. Advancing
		// the raw pointers after the last pair would form an out-of-bounds
		// pointer for an exact 7*stride+8-byte input, even though it is unused.
		s := pack2Rows(step(sp, rowOffset), stride)
		c0 = c0.Add(widen16(absDiffU8x16(s, pack2Rows(step(p0, rowOffset), stride))))
		c1 = c1.Add(widen16(absDiffU8x16(s, pack2Rows(step(p1, rowOffset), stride))))
		c2 = c2.Add(widen16(absDiffU8x16(s, pack2Rows(step(p2, rowOffset), stride))))
		c3 = c3.Add(widen16(absDiffU8x16(s, pack2Rows(step(p3, rowOffset), stride))))
	}
	return int(c0.ReduceSum()), int(c1.ReduceSum()), int(c2.ReduceSum()), int(c3.ReduceSum())
}
