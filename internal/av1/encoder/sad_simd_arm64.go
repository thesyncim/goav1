// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

// Go-native-SIMD motion-estimation SAD kernels (simd/archsimd, Go 1.27+ under
// GOEXPERIMENT=simd). These replace the hand-written NEON asm SAD bodies under
// the simd experiment; regular builds without GOEXPERIMENT=simd keep using the
// assembly dispatch.
//
// SAD = sum over the WxH block of |src[i]-ref[i]| for uint8 pixels. The result
// is an exact integer sum, so every kernel here is byte-identical to the
// scalar reference (sad*PureGo) — proven by the differential test in
// sad_simd_arm64_test.go.
//
// Difference and accumulation strategy. The standard Go 1.27 archsimd API
// does not expose unsigned byte absolute difference, so each byte difference
// is the OR of both saturated subtraction orders. One order is nonzero per
// lane. The result is widened to uint16 and accumulated in independent vectors.
// For 8x8 and 16x16 blocks, the complete result fits in uint16 (at most 16,320
// and 65,280). Each uint16 lane in a 32x32 accumulator is bounded by 32,640;
// the final reduction widens before summing those lanes. The SIMD path uses
// only baseline NEON operations available on arm64; it does not require the
// optional DOTPROD extension.
//
// Hot-loop pointer discipline: the row base is advanced with unsafe.Add(p,
// stride) each row and read through a *[16]uint8 array pointer. This keeps the
// inner loop free of per-row bounds checks and per-row row*stride multiplies
// (which a []byte reslice would emit), matching the asm's `ADD stride, ptr`
// stepping. The callers guarantee the block fits (block geometry + stride), the
// same contract the asm bodies rely on.

package encoder

import (
	"simd/archsimd"
	"unsafe"
)

// load16 reads a 16-byte vector from a raw base pointer (no bounds check).
func load16(p unsafe.Pointer) archsimd.Uint8x16 {
	return archsimd.LoadUint8x16Array((*[16]uint8)(p))
}

// step advances a raw byte pointer by n bytes.
func step(p unsafe.Pointer, n int) unsafe.Pointer { return unsafe.Add(p, n) }

// absDiffU8x16 computes unsigned byte absolute differences using only the
// public archsimd API. The two saturating differences are mutually exclusive
// per lane, so OR combines them into the exact absolute difference.
func absDiffU8x16(a, b archsimd.Uint8x16) archsimd.Uint8x16 {
	return a.SubSaturated(b).Or(b.SubSaturated(a))
}

// widen16 widens one 16-byte abs-diff vector to a Uint16x8 (lo+hi halves added).
func widen16(absd archsimd.Uint8x16) archsimd.Uint16x8 {
	return absd.ExtendLo8ToUint16().Add(absd.HiToLo().ExtendLo8ToUint16())
}

// reduceU16 widens a Uint16x8 accumulator to Uint32x4 (both halves) and reduces
// to a scalar. Used where the running uint16 lanes cannot have overflowed.
func reduceU16(v archsimd.Uint16x8) int {
	return int(v.ExtendLo4ToUint32().Add(v.HiToLo().ExtendLo4ToUint32()).ReduceSum())
}

// pack2Rows loads two 8-byte rows (at base p and base+stride) into one 16-byte
// vector by moving two raw uint64 loads into a Uint64x2 with SetElem (VMOV),
// reinterpreted to Uint8x16. Register moves only — no memory-staging round
// trip and no bounds check. lanes 0..7 = row at p, lanes 8..15 = row at
// p+stride. Packing through the vector lanes avoids staging the two loads in
// a [16]uint8 array before calculating the difference.
func pack2Rows(p unsafe.Pointer, stride int) archsimd.Uint8x16 {
	lo := *(*uint64)(p)
	hi := *(*uint64)(step(p, stride))
	return archsimd.BroadcastUint64x2(lo).SetElem(1, hi).ReshapeToUint8s()
}

// --- single-block full SAD ---------------------------------------------------

// sad8x8SIMD computes the full 8x8 SAD. The row is 8 bytes, so two rows (r and
// r+1) are packed into one 16-byte vector: lanes 0..7 = row r, lanes 8..15 =
// row r+1, identically for src and ref, so the difference covers 16 valid pixels.
// The limit hint is ignored (the full-block total is byte-identical; callers
// compare the total).
func sad8x8SIMD(src, ref []byte, stride int, _ int) int {
	sp := unsafe.Pointer(&src[0])
	rp := unsafe.Pointer(&ref[0])
	acc0 := archsimd.BroadcastUint16x8(0)
	acc1 := archsimd.BroadcastUint16x8(0)
	stride4 := 4 * stride
	for pair := 0; pair < 8; pair += 4 {
		abs0 := absDiffU8x16(pack2Rows(sp, stride), pack2Rows(rp, stride))
		sp2, rp2 := step(sp, 2*stride), step(rp, 2*stride)
		abs1 := absDiffU8x16(pack2Rows(sp2, stride), pack2Rows(rp2, stride))
		acc0 = acc0.Add(widen16(abs0))
		acc1 = acc1.Add(widen16(abs1))
		sp, rp = step(sp, stride4), step(rp, stride4)
	}
	// The complete 8x8 total is at most 64*255 = 16320, so the horizontal
	// uint16 reduction cannot wrap.
	return int(acc0.Add(acc1).ReduceSum())
}

// sad16x16SIMD computes the full 16x16 SAD: one 16-byte difference per row.
func sad16x16SIMD(src, ref []byte, stride int) int {
	sp := unsafe.Pointer(&src[0])
	rp := unsafe.Pointer(&ref[0])
	// Four independent accumulators shorten the row-sum dependency chain. Each
	// accumulator receives four rows, and their final lane-wise sum is at most
	// 16*2*255 = 8160.
	acc0 := archsimd.BroadcastUint16x8(0)
	acc1 := archsimd.BroadcastUint16x8(0)
	acc2 := archsimd.BroadcastUint16x8(0)
	acc3 := archsimd.BroadcastUint16x8(0)
	s0, s1 := sp, step(sp, stride)
	s2, s3 := step(sp, 2*stride), step(sp, 3*stride)
	r0, r1 := rp, step(rp, stride)
	r2, r3 := step(rp, 2*stride), step(rp, 3*stride)
	stride4 := 4 * stride
	for group := 0; group < 4; group++ {
		acc0 = acc0.Add(widen16(absDiffU8x16(load16(s0), load16(r0))))
		acc1 = acc1.Add(widen16(absDiffU8x16(load16(s1), load16(r1))))
		acc2 = acc2.Add(widen16(absDiffU8x16(load16(s2), load16(r2))))
		acc3 = acc3.Add(widen16(absDiffU8x16(load16(s3), load16(r3))))
		s0, s1 = step(s0, stride4), step(s1, stride4)
		s2, s3 = step(s2, stride4), step(s3, stride4)
		r0, r1 = step(r0, stride4), step(r1, stride4)
		r2, r3 = step(r2, stride4), step(r3, stride4)
	}
	// The complete 16x16 total is at most 256*255 = 65280, so reducing directly
	// from uint16 is exact.
	return int(acc0.Add(acc1).Add(acc2).Add(acc3).ReduceSum())
}

// sad32x32SIMD computes the full 32x32 SAD: two 16-byte differences per row.
func sad32x32SIMD(src, ref []byte, stride int) int {
	sp := unsafe.Pointer(&src[0])
	rp := unsafe.Pointer(&ref[0])
	// Each of four independent accumulators receives eight rows. A lane sums
	// four differences per row across the two chunks, so each partial is at most
	// 8*4*255 = 8160 and the final 32-row sum is at most 32640.
	acc0 := archsimd.BroadcastUint16x8(0)
	acc1 := archsimd.BroadcastUint16x8(0)
	acc2 := archsimd.BroadcastUint16x8(0)
	acc3 := archsimd.BroadcastUint16x8(0)
	s0, s1 := sp, step(sp, stride)
	s2, s3 := step(sp, 2*stride), step(sp, 3*stride)
	r0, r1 := rp, step(rp, stride)
	r2, r3 := step(rp, 2*stride), step(rp, 3*stride)
	stride4 := 4 * stride
	for group := 0; group < 8; group++ {
		d0 := widen16(absDiffU8x16(load16(s0), load16(r0))).Add(widen16(absDiffU8x16(load16(step(s0, 16)), load16(step(r0, 16)))))
		d1 := widen16(absDiffU8x16(load16(s1), load16(r1))).Add(widen16(absDiffU8x16(load16(step(s1, 16)), load16(step(r1, 16)))))
		d2 := widen16(absDiffU8x16(load16(s2), load16(r2))).Add(widen16(absDiffU8x16(load16(step(s2, 16)), load16(step(r2, 16)))))
		d3 := widen16(absDiffU8x16(load16(s3), load16(r3))).Add(widen16(absDiffU8x16(load16(step(s3, 16)), load16(step(r3, 16)))))
		acc0, acc1 = acc0.Add(d0), acc1.Add(d1)
		acc2, acc3 = acc2.Add(d2), acc3.Add(d3)
		s0, s1 = step(s0, stride4), step(s1, stride4)
		s2, s3 = step(s2, stride4), step(s3, stride4)
		r0, r1 = step(r0, stride4), step(r1, stride4)
		r2, r3 = step(r2, stride4), step(r3, stride4)
	}
	return reduceU16(acc0.Add(acc1).Add(acc2).Add(acc3))
}

// --- 4-reference search variants (the motion-search hot path) ----------------

// sad8x8x4SIMD computes four 8x8 SADs of one src block against four independent
// reference origins. The two-row-per-vector src pack is done ONCE per row-pair
// and reused across the four references, and the four accumulation chains are
// independent.
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
	s2 := 2 * stride
	for row := 0; row < 8; row += 2 {
		s := pack2Rows(sp, stride)
		c0 = c0.Add(widen16(absDiffU8x16(s, pack2Rows(p0, stride))))
		c1 = c1.Add(widen16(absDiffU8x16(s, pack2Rows(p1, stride))))
		c2 = c2.Add(widen16(absDiffU8x16(s, pack2Rows(p2, stride))))
		c3 = c3.Add(widen16(absDiffU8x16(s, pack2Rows(p3, stride))))
		sp = step(sp, s2)
		p0, p1, p2, p3 = step(p0, s2), step(p1, s2), step(p2, s2), step(p3, s2)
	}
	// Each complete 8x8 result is at most 64*255 = 16320, so direct uint16
	// reductions are exact.
	return int(c0.ReduceSum()), int(c1.ReduceSum()), int(c2.ReduceSum()), int(c3.ReduceSum())
}

// sad16x16x4SIMD computes four 16x16 SADs of one src block against four
// independent reference origins. The src row is loaded once per row and reused
// across the four references.
func sad16x16x4SIMD(src, ref0, ref1, ref2, ref3 []byte, stride int) (int, int, int, int) {
	sp := unsafe.Pointer(&src[0])
	p0 := unsafe.Pointer(&ref0[0])
	p1 := unsafe.Pointer(&ref1[0])
	p2 := unsafe.Pointer(&ref2[0])
	p3 := unsafe.Pointer(&ref3[0])
	// Even and odd rows use independent accumulators for each candidate. Each
	// lane sums at most 16*2*255 = 8160 across the two partial vectors.
	c00, c01 := archsimd.BroadcastUint16x8(0), archsimd.BroadcastUint16x8(0)
	c10, c11 := archsimd.BroadcastUint16x8(0), archsimd.BroadcastUint16x8(0)
	c20, c21 := archsimd.BroadcastUint16x8(0), archsimd.BroadcastUint16x8(0)
	c30, c31 := archsimd.BroadcastUint16x8(0), archsimd.BroadcastUint16x8(0)
	stride2 := 2 * stride
	for row := 0; row < 16; row += 2 {
		s0, s1 := load16(sp), load16(step(sp, stride))
		c00 = c00.Add(widen16(absDiffU8x16(s0, load16(p0))))
		c01 = c01.Add(widen16(absDiffU8x16(s1, load16(step(p0, stride)))))
		c10 = c10.Add(widen16(absDiffU8x16(s0, load16(p1))))
		c11 = c11.Add(widen16(absDiffU8x16(s1, load16(step(p1, stride)))))
		c20 = c20.Add(widen16(absDiffU8x16(s0, load16(p2))))
		c21 = c21.Add(widen16(absDiffU8x16(s1, load16(step(p2, stride)))))
		c30 = c30.Add(widen16(absDiffU8x16(s0, load16(p3))))
		c31 = c31.Add(widen16(absDiffU8x16(s1, load16(step(p3, stride)))))
		sp = step(sp, stride2)
		p0, p1 = step(p0, stride2), step(p1, stride2)
		p2, p3 = step(p2, stride2), step(p3, stride2)
	}
	v0, v1 := c00.Add(c01), c10.Add(c11)
	v2, v3 := c20.Add(c21), c30.Add(c31)
	// Each complete 16x16 result is at most 256*255 = 65280, so direct uint16
	// reductions are exact.
	return int(v0.ReduceSum()), int(v1.ReduceSum()), int(v2.ReduceSum()), int(v3.ReduceSum())
}

// sad32x32x4SIMD computes four 32x32 SADs of one src block against four
// independent reference origins.
func sad32x32x4SIMD(src, ref0, ref1, ref2, ref3 []byte, stride int) (int, int, int, int) {
	sp := unsafe.Pointer(&src[0])
	p0 := unsafe.Pointer(&ref0[0])
	p1 := unsafe.Pointer(&ref1[0])
	p2 := unsafe.Pointer(&ref2[0])
	p3 := unsafe.Pointer(&ref3[0])
	// Even and odd rows accumulate separately for each candidate. Each partial
	// sums 16 rows; the final 32-row lane sum is bounded by 32640.
	c00, c01 := archsimd.BroadcastUint16x8(0), archsimd.BroadcastUint16x8(0)
	c10, c11 := archsimd.BroadcastUint16x8(0), archsimd.BroadcastUint16x8(0)
	c20, c21 := archsimd.BroadcastUint16x8(0), archsimd.BroadcastUint16x8(0)
	c30, c31 := archsimd.BroadcastUint16x8(0), archsimd.BroadcastUint16x8(0)
	stride2 := 2 * stride
	for row := 0; row < 32; row += 2 {
		s0Lo, s0Hi := load16(sp), load16(step(sp, 16))
		s1 := step(sp, stride)
		s1Lo, s1Hi := load16(s1), load16(step(s1, 16))
		c00 = c00.Add(widen16(absDiffU8x16(s0Lo, load16(p0)))).Add(widen16(absDiffU8x16(s0Hi, load16(step(p0, 16)))))
		c01 = c01.Add(widen16(absDiffU8x16(s1Lo, load16(step(p0, stride))))).Add(widen16(absDiffU8x16(s1Hi, load16(step(step(p0, stride), 16)))))
		c10 = c10.Add(widen16(absDiffU8x16(s0Lo, load16(p1)))).Add(widen16(absDiffU8x16(s0Hi, load16(step(p1, 16)))))
		c11 = c11.Add(widen16(absDiffU8x16(s1Lo, load16(step(p1, stride))))).Add(widen16(absDiffU8x16(s1Hi, load16(step(step(p1, stride), 16)))))
		c20 = c20.Add(widen16(absDiffU8x16(s0Lo, load16(p2)))).Add(widen16(absDiffU8x16(s0Hi, load16(step(p2, 16)))))
		c21 = c21.Add(widen16(absDiffU8x16(s1Lo, load16(step(p2, stride))))).Add(widen16(absDiffU8x16(s1Hi, load16(step(step(p2, stride), 16)))))
		c30 = c30.Add(widen16(absDiffU8x16(s0Lo, load16(p3)))).Add(widen16(absDiffU8x16(s0Hi, load16(step(p3, 16)))))
		c31 = c31.Add(widen16(absDiffU8x16(s1Lo, load16(step(p3, stride))))).Add(widen16(absDiffU8x16(s1Hi, load16(step(step(p3, stride), 16)))))
		sp = step(sp, stride2)
		p0, p1 = step(p0, stride2), step(p1, stride2)
		p2, p3 = step(p2, stride2), step(p3, stride2)
	}
	v0, v1 := c00.Add(c01), c10.Add(c11)
	v2, v3 := c20.Add(c21), c30.Add(c31)
	return reduceU16(v0), reduceU16(v1), reduceU16(v2), reduceU16(v3)
}
