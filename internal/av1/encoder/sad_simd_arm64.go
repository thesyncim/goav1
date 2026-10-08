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
// Accumulation strategy. The standard Go 1.27 archsimd API does not expose
// unsigned byte absolute-difference or unsigned byte dot-product intrinsics,
// so compute the absolute difference as max(a,b)-min(a,b), widen to uint16,
// and accumulate.
// The largest uint16 lane total before a flush or final reduction is 32*255
// (=8160): four differences per row for eight rows in the 32x32 kernels, or
// two differences per row for all sixteen rows in the 16x16 kernels. The 32-row
// kernels periodically flush to uint32 before overflow. The SIMD path uses only
// baseline NEON operations available on arm64; it does not require the optional
// DOTPROD extension.
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
// public archsimd API. Unsigned max/min followed by wrapping subtraction is
// exact because max(a,b) is always at least min(a,b) in every lane.
func absDiffU8x16(a, b archsimd.Uint8x16) archsimd.Uint8x16 {
	return a.Max(b).Sub(a.Min(b))
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
	acc := archsimd.BroadcastUint16x8(0)
	for row := 0; row < 8; row += 2 {
		absd := absDiffU8x16(pack2Rows(sp, stride), pack2Rows(rp, stride))
		acc = acc.Add(widen16(absd))
		sp, rp = step(sp, 2*stride), step(rp, 2*stride)
	}
	// 8x8 max per uint16 lane: 8 rows * 255 = 2040 < 65535, safe.
	return reduceU16(acc)
}

// sad16x16SIMD computes the full 16x16 SAD: one 16-byte difference per row.
func sad16x16SIMD(src, ref []byte, stride int) int {
	sp := unsafe.Pointer(&src[0])
	rp := unsafe.Pointer(&ref[0])
	// Each lane sums two differences per row: 16*2*255 = 8160, safe.
	acc := archsimd.BroadcastUint16x8(0)
	for row := 0; row < 16; row++ {
		acc = acc.Add(widen16(absDiffU8x16(load16(sp), load16(rp))))
		sp, rp = step(sp, stride), step(rp, stride)
	}
	return reduceU16(acc)
}

// sad32x32SIMD computes the full 32x32 SAD: two 16-byte differences per row.
func sad32x32SIMD(src, ref []byte, stride int) int {
	sp := unsafe.Pointer(&src[0])
	rp := unsafe.Pointer(&ref[0])
	// Flush every 8 rows: each lane sums four differences per row across the two
	// chunks, so 8*4*255 = 8160 remains within uint16.
	total := archsimd.BroadcastUint32x4(0)
	acc := archsimd.BroadcastUint16x8(0)
	for row := 0; row < 32; row++ {
		acc = acc.Add(widen16(absDiffU8x16(load16(sp), load16(rp)))).
			Add(widen16(absDiffU8x16(load16(step(sp, 16)), load16(step(rp, 16)))))
		if row&7 == 7 {
			total = total.Add(acc.ExtendLo4ToUint32()).Add(acc.HiToLo().ExtendLo4ToUint32())
			acc = archsimd.BroadcastUint16x8(0)
		}
		sp, rp = step(sp, stride), step(rp, stride)
	}
	return int(total.ReduceSum())
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
	return reduceU16(c0), reduceU16(c1), reduceU16(c2), reduceU16(c3)
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
	c0 := archsimd.BroadcastUint16x8(0)
	c1 := archsimd.BroadcastUint16x8(0)
	c2 := archsimd.BroadcastUint16x8(0)
	c3 := archsimd.BroadcastUint16x8(0)
	for row := 0; row < 16; row++ {
		s := load16(sp)
		c0 = c0.Add(widen16(absDiffU8x16(s, load16(p0))))
		c1 = c1.Add(widen16(absDiffU8x16(s, load16(p1))))
		c2 = c2.Add(widen16(absDiffU8x16(s, load16(p2))))
		c3 = c3.Add(widen16(absDiffU8x16(s, load16(p3))))
		sp = step(sp, stride)
		p0, p1, p2, p3 = step(p0, stride), step(p1, stride), step(p2, stride), step(p3, stride)
	}
	return reduceU16(c0), reduceU16(c1), reduceU16(c2), reduceU16(c3)
}

// sad32x32x4SIMD computes four 32x32 SADs of one src block against four
// independent reference origins.
func sad32x32x4SIMD(src, ref0, ref1, ref2, ref3 []byte, stride int) (int, int, int, int) {
	sp := unsafe.Pointer(&src[0])
	p0 := unsafe.Pointer(&ref0[0])
	p1 := unsafe.Pointer(&ref1[0])
	p2 := unsafe.Pointer(&ref2[0])
	p3 := unsafe.Pointer(&ref3[0])
	t0 := archsimd.BroadcastUint32x4(0)
	t1 := archsimd.BroadcastUint32x4(0)
	t2 := archsimd.BroadcastUint32x4(0)
	t3 := archsimd.BroadcastUint32x4(0)
	c0 := archsimd.BroadcastUint16x8(0)
	c1 := archsimd.BroadcastUint16x8(0)
	c2 := archsimd.BroadcastUint16x8(0)
	c3 := archsimd.BroadcastUint16x8(0)
	flush := func(c archsimd.Uint16x8, t archsimd.Uint32x4) archsimd.Uint32x4 {
		return t.Add(c.ExtendLo4ToUint32()).Add(c.HiToLo().ExtendLo4ToUint32())
	}
	for row := 0; row < 32; row++ {
		sLo := load16(sp)
		sHi := load16(step(sp, 16))
		c0 = c0.Add(widen16(absDiffU8x16(sLo, load16(p0)))).Add(widen16(absDiffU8x16(sHi, load16(step(p0, 16)))))
		c1 = c1.Add(widen16(absDiffU8x16(sLo, load16(p1)))).Add(widen16(absDiffU8x16(sHi, load16(step(p1, 16)))))
		c2 = c2.Add(widen16(absDiffU8x16(sLo, load16(p2)))).Add(widen16(absDiffU8x16(sHi, load16(step(p2, 16)))))
		c3 = c3.Add(widen16(absDiffU8x16(sLo, load16(p3)))).Add(widen16(absDiffU8x16(sHi, load16(step(p3, 16)))))
		if row&7 == 7 {
			t0, t1, t2, t3 = flush(c0, t0), flush(c1, t1), flush(c2, t2), flush(c3, t3)
			c0 = archsimd.BroadcastUint16x8(0)
			c1 = archsimd.BroadcastUint16x8(0)
			c2 = archsimd.BroadcastUint16x8(0)
			c3 = archsimd.BroadcastUint16x8(0)
		}
		sp = step(sp, stride)
		p0, p1, p2, p3 = step(p0, stride), step(p1, stride), step(p2, stride), step(p3, stride)
	}
	return int(t0.ReduceSum()), int(t1.ReduceSum()), int(t2.ReduceSum()), int(t3.ReduceSum())
}
