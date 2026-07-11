// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

// Go-native-SIMD motion-estimation SAD kernels (simd/archsimd, Go 1.27+ under
// GOEXPERIMENT=simd). These replace the hand-written NEON asm SAD bodies under
// the simd experiment; the normal Go 1.26 production build never sees this file
// (it is gated by goexperiment.simd) and keeps using the asm dispatch.
//
// SAD = sum over the WxH block of |src[i]-ref[i]| for uint8 pixels. The result
// is an exact integer sum, so every kernel here is byte-identical to the
// scalar reference (sad*PureGo) — proven by the differential test in
// sad_simd_arm64_test.go.
//
// Accumulation strategy. Each 16-byte row chunk is one AbsDiff (VUABD) giving
// 16 unsigned abs-diffs, each <= 255. A uint16 accumulator would overflow (a
// 32x32 block sums 1024 bytes per column-lane), so we accumulate WITHOUT
// overflow into a Uint32x4:
//   - DOTPROD path (Apple Silicon, ARMv8.2): acc = acc.DotProd(absdiff, ones)
//     (VUDOT). Each UDOT lane sums four abs-diff bytes into a uint32 lane in one
//     instruction — the SVT-AV1 UABD+UDOT shape, no widening/overflow.
//   - base-NEON fallback: widen the 16 abs-diff bytes to two Uint16x8 halves
//     (VUXTL/VUXTL2), add them, accumulate in a Uint16x8; a uint16 lane holds up
//     to 16 rows of 255 (=4080) safely, so blocks >16 rows tall flush to uint32
//     (VUXTL/VUXTL2 + add) before overflow.
// A single Uint32x4.ReduceSum() / Uint16x8 widen+reduce (ADDV) folds the lanes
// to the scalar total at the end.
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

	"github.com/thesyncim/goav1/internal/av1/dsp/cpu"
)

// useDotProdSADSIMD selects the VUDOT accumulate when the CPU advertises the
// dot-product extension (all Apple Silicon does). Mirrors useDotProdSAD.
var useDotProdSADSIMD = cpu.Detected.DOTPROD

// load16 reads a 16-byte vector from a raw base pointer (no bounds check).
func load16(p unsafe.Pointer) archsimd.Uint8x16 {
	return archsimd.LoadUint8x16Array((*[16]uint8)(p))
}

// step advances a raw byte pointer by n bytes.
func step(p unsafe.Pointer, n int) unsafe.Pointer { return unsafe.Add(p, n) }

// widen16 widens one 16-byte abs-diff vector to a Uint16x8 (lo+hi halves added).
func widen16(absd archsimd.Uint8x16) archsimd.Uint16x8 {
	return absd.ExtendLo8ToUint16().Add(absd.ExtendHi8ToUint16())
}

// reduceU16 widens a Uint16x8 accumulator to Uint32x4 (both halves) and reduces
// to a scalar. Used where the running uint16 lanes cannot have overflowed.
func reduceU16(v archsimd.Uint16x8) int {
	return int(v.ExtendLo4ToUint32().Add(v.ExtendHi4ToUint32()).ReduceSum())
}

// pack2Rows loads two 8-byte rows (at base p and base+stride) into one 16-byte
// vector by moving two raw uint64 loads into a Uint64x2 with SetElem (VMOV),
// reinterpreted to Uint8x16. Register moves only — no memory-staging round
// trip and no bounds check. lanes 0..7 = row at p, lanes 8..15 = row at
// p+stride. This is what makes the 8-wide SAD competitive with the asm: the
// naive path packs through a [16]uint8 stack slot (LDR+STR per half), whereas
// this keeps the doublewords in registers straight into the vector lanes.
func pack2Rows(p unsafe.Pointer, stride int) archsimd.Uint8x16 {
	lo := *(*uint64)(p)
	hi := *(*uint64)(step(p, stride))
	return archsimd.BroadcastUint64x2(lo).SetElem(1, hi).ReshapeToUint8s()
}

// --- single-block full SAD ---------------------------------------------------

// sad8x8SIMD computes the full 8x8 SAD. The row is 8 bytes, so two rows (r and
// r+1) are packed into one 16-byte vector: lanes 0..7 = row r, lanes 8..15 =
// row r+1, identically for src and ref, so AbsDiff yields 16 valid abs-diffs.
// The limit hint is ignored (the full-block total is byte-identical; callers
// compare the total).
func sad8x8SIMD(src, ref []byte, stride int, _ int) int {
	sp := unsafe.Pointer(&src[0])
	rp := unsafe.Pointer(&ref[0])
	if useDotProdSADSIMD {
		ones := archsimd.BroadcastUint8x16(1)
		// Two independent UDOT accumulators over the four packed row-pair
		// vectors, breaking the serial UDOT chain. NOTE: sad8x8 single-block
		// stays on the NEON asm in the dispatch (the two-row register pack —
		// VDUP+VMOV per pair — costs more than the asm's single VLD1 .8b when
		// there is no 4x reference reuse to amortize it, see sad8x8x4SIMD).
		// This kernel is kept byte-exact and benched for completeness.
		a0 := archsimd.BroadcastUint32x4(0)
		a1 := archsimd.BroadcastUint32x4(0)
		s2 := 2 * stride
		for row := 0; row < 8; row += 4 {
			d0 := pack2Rows(sp, stride).AbsDiff(pack2Rows(rp, stride))
			d1 := pack2Rows(step(sp, s2), stride).AbsDiff(pack2Rows(step(rp, s2), stride))
			a0 = a0.DotProd(d0, ones)
			a1 = a1.DotProd(d1, ones)
			sp, rp = step(sp, 2*s2), step(rp, 2*s2)
		}
		return int(a0.Add(a1).ReduceSum())
	}
	acc := archsimd.BroadcastUint16x8(0)
	for row := 0; row < 8; row += 2 {
		absd := pack2Rows(sp, stride).AbsDiff(pack2Rows(rp, stride))
		acc = acc.Add(widen16(absd))
		sp, rp = step(sp, 2*stride), step(rp, 2*stride)
	}
	// 8x8 max per uint16 lane: 8 rows * 255 = 2040 < 65535, safe.
	return reduceU16(acc)
}

// sad16x16SIMD computes the full 16x16 SAD: one 16-byte AbsDiff per row.
func sad16x16SIMD(src, ref []byte, stride int) int {
	sp := unsafe.Pointer(&src[0])
	rp := unsafe.Pointer(&ref[0])
	if useDotProdSADSIMD {
		ones := archsimd.BroadcastUint8x16(1)
		// Two independent UDOT accumulators broken across even/odd rows: this
		// splits the per-row serial UDOT dependency chain so the two SIMD pipes
		// run in parallel (2 rows retire per issue window). Summed at the end.
		s2 := 2 * stride
		acc0 := archsimd.BroadcastUint32x4(0)
		acc1 := archsimd.BroadcastUint32x4(0)
		for row := 0; row < 16; row += 2 {
			acc0 = acc0.DotProd(load16(sp).AbsDiff(load16(rp)), ones)
			acc1 = acc1.DotProd(load16(step(sp, stride)).AbsDiff(load16(step(rp, stride))), ones)
			sp, rp = step(sp, s2), step(rp, s2)
		}
		return int(acc0.Add(acc1).ReduceSum())
	}
	// 16 rows * 255 = 4080 per uint16 lane, safe for the whole block.
	acc := archsimd.BroadcastUint16x8(0)
	for row := 0; row < 16; row++ {
		acc = acc.Add(widen16(load16(sp).AbsDiff(load16(rp))))
		sp, rp = step(sp, stride), step(rp, stride)
	}
	return reduceU16(acc)
}

// sad32x32SIMD computes the full 32x32 SAD: two 16-byte AbsDiffs per row.
func sad32x32SIMD(src, ref []byte, stride int) int {
	sp := unsafe.Pointer(&src[0])
	rp := unsafe.Pointer(&ref[0])
	if useDotProdSADSIMD {
		ones := archsimd.BroadcastUint8x16(1)
		// Independent accumulators for the low and high 16-byte column halves:
		// the two UDOT chains are data-independent, so they issue in parallel.
		accLo := archsimd.BroadcastUint32x4(0)
		accHi := archsimd.BroadcastUint32x4(0)
		for row := 0; row < 32; row++ {
			accLo = accLo.DotProd(load16(sp).AbsDiff(load16(rp)), ones)
			accHi = accHi.DotProd(load16(step(sp, 16)).AbsDiff(load16(step(rp, 16))), ones)
			sp, rp = step(sp, stride), step(rp, stride)
		}
		return int(accLo.Add(accHi).ReduceSum())
	}
	// widen path: flush the uint16 accumulator to uint32 every 8 rows to stay
	// well inside uint16 range (8 rows * 2 chunks * 255 spread over 8 lanes).
	total := archsimd.BroadcastUint32x4(0)
	acc := archsimd.BroadcastUint16x8(0)
	for row := 0; row < 32; row++ {
		acc = acc.Add(widen16(load16(sp).AbsDiff(load16(rp)))).
			Add(widen16(load16(step(sp, 16)).AbsDiff(load16(step(rp, 16)))))
		if row&7 == 7 {
			total = total.Add(acc.ExtendLo4ToUint32()).Add(acc.ExtendHi4ToUint32())
			acc = archsimd.BroadcastUint16x8(0)
		}
		sp, rp = step(sp, stride), step(rp, stride)
	}
	return int(total.ReduceSum())
}

// --- 4-reference search variants (the motion-search hot path) ----------------

// sad8x8x4SIMD computes four 8x8 SADs of one src block against four independent
// reference origins. The two-row-per-vector src pack is done ONCE per row-pair
// and reused across the four references (its cost — the expensive part of the
// 8-wide shape — amortizes 4x), and the four UDOT chains are independent for
// full ILP.
func sad8x8x4SIMD(src, ref0, ref1, ref2, ref3 []byte, stride int) (int, int, int, int) {
	sp := unsafe.Pointer(&src[0])
	p0 := unsafe.Pointer(&ref0[0])
	p1 := unsafe.Pointer(&ref1[0])
	p2 := unsafe.Pointer(&ref2[0])
	p3 := unsafe.Pointer(&ref3[0])
	if useDotProdSADSIMD {
		ones := archsimd.BroadcastUint8x16(1)
		a0 := archsimd.BroadcastUint32x4(0)
		a1 := archsimd.BroadcastUint32x4(0)
		a2 := archsimd.BroadcastUint32x4(0)
		a3 := archsimd.BroadcastUint32x4(0)
		s2 := 2 * stride
		for row := 0; row < 8; row += 2 {
			s := pack2Rows(sp, stride)
			a0 = a0.DotProd(s.AbsDiff(pack2Rows(p0, stride)), ones)
			a1 = a1.DotProd(s.AbsDiff(pack2Rows(p1, stride)), ones)
			a2 = a2.DotProd(s.AbsDiff(pack2Rows(p2, stride)), ones)
			a3 = a3.DotProd(s.AbsDiff(pack2Rows(p3, stride)), ones)
			sp = step(sp, s2)
			p0, p1, p2, p3 = step(p0, s2), step(p1, s2), step(p2, s2), step(p3, s2)
		}
		return int(a0.ReduceSum()), int(a1.ReduceSum()), int(a2.ReduceSum()), int(a3.ReduceSum())
	}
	c0 := archsimd.BroadcastUint16x8(0)
	c1 := archsimd.BroadcastUint16x8(0)
	c2 := archsimd.BroadcastUint16x8(0)
	c3 := archsimd.BroadcastUint16x8(0)
	s2 := 2 * stride
	for row := 0; row < 8; row += 2 {
		s := pack2Rows(sp, stride)
		c0 = c0.Add(widen16(s.AbsDiff(pack2Rows(p0, stride))))
		c1 = c1.Add(widen16(s.AbsDiff(pack2Rows(p1, stride))))
		c2 = c2.Add(widen16(s.AbsDiff(pack2Rows(p2, stride))))
		c3 = c3.Add(widen16(s.AbsDiff(pack2Rows(p3, stride))))
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
	if useDotProdSADSIMD {
		ones := archsimd.BroadcastUint8x16(1)
		a0 := archsimd.BroadcastUint32x4(0)
		a1 := archsimd.BroadcastUint32x4(0)
		a2 := archsimd.BroadcastUint32x4(0)
		a3 := archsimd.BroadcastUint32x4(0)
		for row := 0; row < 16; row++ {
			s := load16(sp)
			a0 = a0.DotProd(s.AbsDiff(load16(p0)), ones)
			a1 = a1.DotProd(s.AbsDiff(load16(p1)), ones)
			a2 = a2.DotProd(s.AbsDiff(load16(p2)), ones)
			a3 = a3.DotProd(s.AbsDiff(load16(p3)), ones)
			sp = step(sp, stride)
			p0, p1, p2, p3 = step(p0, stride), step(p1, stride), step(p2, stride), step(p3, stride)
		}
		return int(a0.ReduceSum()), int(a1.ReduceSum()), int(a2.ReduceSum()), int(a3.ReduceSum())
	}
	c0 := archsimd.BroadcastUint16x8(0)
	c1 := archsimd.BroadcastUint16x8(0)
	c2 := archsimd.BroadcastUint16x8(0)
	c3 := archsimd.BroadcastUint16x8(0)
	for row := 0; row < 16; row++ {
		s := load16(sp)
		c0 = c0.Add(widen16(s.AbsDiff(load16(p0))))
		c1 = c1.Add(widen16(s.AbsDiff(load16(p1))))
		c2 = c2.Add(widen16(s.AbsDiff(load16(p2))))
		c3 = c3.Add(widen16(s.AbsDiff(load16(p3))))
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
	if useDotProdSADSIMD {
		ones := archsimd.BroadcastUint8x16(1)
		// Eight independent UDOT accumulators (lo/hi column half per reference):
		// every DotProd chain is data-independent, maximizing ILP across the
		// machine's SIMD issue ports. Folded lo+hi per ref at the end.
		a0l := archsimd.BroadcastUint32x4(0)
		a0h := archsimd.BroadcastUint32x4(0)
		a1l := archsimd.BroadcastUint32x4(0)
		a1h := archsimd.BroadcastUint32x4(0)
		a2l := archsimd.BroadcastUint32x4(0)
		a2h := archsimd.BroadcastUint32x4(0)
		a3l := archsimd.BroadcastUint32x4(0)
		a3h := archsimd.BroadcastUint32x4(0)
		for row := 0; row < 32; row++ {
			sLo := load16(sp)
			sHi := load16(step(sp, 16))
			a0l = a0l.DotProd(sLo.AbsDiff(load16(p0)), ones)
			a0h = a0h.DotProd(sHi.AbsDiff(load16(step(p0, 16))), ones)
			a1l = a1l.DotProd(sLo.AbsDiff(load16(p1)), ones)
			a1h = a1h.DotProd(sHi.AbsDiff(load16(step(p1, 16))), ones)
			a2l = a2l.DotProd(sLo.AbsDiff(load16(p2)), ones)
			a2h = a2h.DotProd(sHi.AbsDiff(load16(step(p2, 16))), ones)
			a3l = a3l.DotProd(sLo.AbsDiff(load16(p3)), ones)
			a3h = a3h.DotProd(sHi.AbsDiff(load16(step(p3, 16))), ones)
			sp = step(sp, stride)
			p0, p1, p2, p3 = step(p0, stride), step(p1, stride), step(p2, stride), step(p3, stride)
		}
		return int(a0l.Add(a0h).ReduceSum()), int(a1l.Add(a1h).ReduceSum()),
			int(a2l.Add(a2h).ReduceSum()), int(a3l.Add(a3h).ReduceSum())
	}
	t0 := archsimd.BroadcastUint32x4(0)
	t1 := archsimd.BroadcastUint32x4(0)
	t2 := archsimd.BroadcastUint32x4(0)
	t3 := archsimd.BroadcastUint32x4(0)
	c0 := archsimd.BroadcastUint16x8(0)
	c1 := archsimd.BroadcastUint16x8(0)
	c2 := archsimd.BroadcastUint16x8(0)
	c3 := archsimd.BroadcastUint16x8(0)
	flush := func(c archsimd.Uint16x8, t archsimd.Uint32x4) archsimd.Uint32x4 {
		return t.Add(c.ExtendLo4ToUint32()).Add(c.ExtendHi4ToUint32())
	}
	for row := 0; row < 32; row++ {
		sLo := load16(sp)
		sHi := load16(step(sp, 16))
		c0 = c0.Add(widen16(sLo.AbsDiff(load16(p0)))).Add(widen16(sHi.AbsDiff(load16(step(p0, 16)))))
		c1 = c1.Add(widen16(sLo.AbsDiff(load16(p1)))).Add(widen16(sHi.AbsDiff(load16(step(p1, 16)))))
		c2 = c2.Add(widen16(sLo.AbsDiff(load16(p2)))).Add(widen16(sHi.AbsDiff(load16(step(p2, 16)))))
		c3 = c3.Add(widen16(sLo.AbsDiff(load16(p3)))).Add(widen16(sHi.AbsDiff(load16(step(p3, 16)))))
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
