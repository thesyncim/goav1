// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

// Go-native SIMD 8-bit fourteen-sample (filter14) deblocking kernel, int16
// 8-wide (simd/archsimd). Eight edge positions per Int16x8, one lane per
// position; same contiguous-horizontal data layout as the narrow kernel (see
// filter4_gosimd_arm64.go). Vertical edges reuse the NEON trn-ladder
// gather/scatter (filter14_vtrn_neon_arm64.s) around this horizontal core;
// short tails route through the pure-Go reference.
//
// Structure (vs filter14EdgeNEONAsm, which is a straight-line two-pass kernel
// that recomputes the need/flat masks twice and re-sums each of the twelve
// wide averages from scratch, ~581 instructions per group):
//   - single pass: the need / flat8in / flat8out masks are computed once.
//   - dav1d-style branch ladder (third_party/upstream/dav1d/src/arm/64/
//     loopfilter.S): if no lane is need&&flat the group is narrow-only (four
//     gated stores; the six outer tap rows are not even loaded); if no lane is
//     also flat8out the group takes the filter8 path (six stores); only
//     groups with at least one wide lane run the fourteen-tap averages.
//   - the six/twelve flat averages are running box sums (dav1d's incremental
//     add/subtract of the sliding window): the accumulator starts at the first
//     weighted sum plus the rounding bias, and each next output costs two pair
//     adds and two accumulator ops instead of an 11..13-term re-summation.
//     With the bias folded into the accumulator, roundPowerOfTwo(sum, n) is a
//     single arithmetic ShiftAllRightConst(n) (sums are non-negative and at
//     most 255*16+8, inside int16).
//
// Byte-exactness with filter14EdgePureGo:
//   - needsFilter8, flatMask4(thr, p3..q3) and flatMask4(thr, p6,p5,p4,p0,q0,
//     q4,q5,q6) are reproduced lane-wise (AbsDiff / LessEqual against the
//     broadcast thresholds) into the need / flat / flat2 masks.
//   - the sliding-window sums are algebraically identical to the reference's
//     weighted averages (each step adds the incoming pair and subtracts the
//     outgoing pair of the 8/16-wide window).
//   - the not-flat lanes reuse the exact filter4 narrow update; the
//     flat-but-not-flat2 lanes take the filter8 six-output flat branch; every
//     blend is a per-output IfElse chain over flat2/flat/need, matching the
//     reference's nested conditional writes (outer taps p5..p3/q3..q5 keep
//     their original samples unless the lane is need&&flat&&flat2).

package loopfilter

import (
	"simd/archsimd"
	"unsafe"
)

// filter14EdgeSIMD is the Go-native SIMD form of filter14EdgePureGo for 8-bit
// horizontal edges. Vertical edges route through the NEON transpose around the
// same horizontal core (filter14VertSIMD); other layouts and short tails fall
// back to the NEON asm / pure-Go reference.
func filter14EdgeSIMD(pix []byte, q0Base int, step int, outer int, length int, scale int, params filter4Params) {
	groups := length / 8
	if outer != 1 || groups == 0 {
		if step == 1 && groups > 0 {
			filter14VertSIMD(pix, q0Base, step, outer, length, scale, params)
			return
		}
		filter14EdgeNEON(pix, q0Base, step, outer, length, scale, params)
		return
	}
	// Constants as flat locals so they stay register-resident (see
	// filter4_gosimd_arm64.go); the whole group body is pasted inline — a
	// helper call would push the fourteen tap vectors through the stack ABI.
	limit := archsimd.BroadcastInt16x8(params.limit)
	blimit := archsimd.BroadcastInt16x8(params.blimit)
	hevT := archsimd.BroadcastInt16x8(params.hev)
	center := archsimd.BroadcastInt16x8(params.center)
	minV := archsimd.BroadcastInt16x8(params.min)
	maxV := archsimd.BroadcastInt16x8(params.max)
	one := archsimd.BroadcastInt16x8(1)
	three16 := archsimd.BroadcastInt16x8(3)
	four := archsimd.BroadcastInt16x8(4)
	eight := archsimd.BroadcastInt16x8(8)
	flatThr := archsimd.BroadcastInt16x8(int16(scale))
	q0p := unsafe.Pointer(&pix[q0Base])
	for g := 0; g < groups; g++ {
		base := unsafe.Add(q0p, g*8)
		pP3 := unsafe.Add(base, -4*step)
		pP2 := unsafe.Add(base, -3*step)
		pP1 := unsafe.Add(base, -2*step)
		pP0 := unsafe.Add(base, -step)
		pQ1 := unsafe.Add(base, step)
		pQ2 := unsafe.Add(base, 2*step)
		pQ3 := unsafe.Add(base, 3*step)
		p3 := lf8LoadP(pP3)
		p2 := lf8LoadP(pP2)
		p1 := lf8LoadP(pP1)
		p0 := lf8LoadP(pP0)
		q0 := lf8LoadP(base)
		q1 := lf8LoadP(pQ1)
		q2 := lf8LoadP(pQ2)
		q3 := lf8LoadP(pQ3)

		// needsFilter8 (inlined)
		need := p3.AbsDiff(p2).LessEqual(limit).
			And(p2.AbsDiff(p1).LessEqual(limit)).
			And(p1.AbsDiff(p0).LessEqual(limit)).
			And(q1.AbsDiff(q0).LessEqual(limit)).
			And(q2.AbsDiff(q1).LessEqual(limit)).
			And(q3.AbsDiff(q2).LessEqual(limit)).
			And(p0.AbsDiff(q0).ShiftAllLeftConst(1).
				Add(p1.AbsDiff(q1).ShiftAllRightConst(1)).
				LessEqual(blimit))
		// flat8in mask (flatMask4 over p3..q3, inlined)
		flat := p1.AbsDiff(p0).LessEqual(flatThr).
			And(q1.AbsDiff(q0).LessEqual(flatThr)).
			And(p2.AbsDiff(p0).LessEqual(flatThr)).
			And(q2.AbsDiff(q0).LessEqual(flatThr)).
			And(p3.AbsDiff(p0).LessEqual(flatThr)).
			And(q3.AbsDiff(q0).LessEqual(flatThr))
		// narrow four-tap (inlined filter4TapNoGate), needed on every path
		hev := p1.AbsDiff(p0).Greater(hevT).Or(q1.AbsDiff(q0).Greater(hevT))
		ps1 := p1.Sub(center)
		ps0 := p0.Sub(center)
		qs0 := q0.Sub(center)
		qs1 := q1.Sub(center)
		f := ps1.Sub(qs1).Max(minV).Min(maxV).Masked(hev)
		f = f.Add(qs0.Sub(ps0).Mul(three16)).Max(minV).Min(maxV)
		filter1 := f.Add(four).Max(minV).Min(maxV).ShiftAllRightConst(3)
		filter2 := f.Add(three16).Max(minV).Min(maxV).ShiftAllRightConst(3)
		np0 := ps0.Add(filter2).Max(minV).Min(maxV).Add(center)
		nq0 := qs0.Sub(filter1).Max(minV).Min(maxV).Add(center)
		ov := filter1.Add(one).ShiftAllRightConst(1)
		np1 := p1.IfElse(hev, ps1.Add(ov).Max(minV).Min(maxV).Add(center))
		nq1 := q1.IfElse(hev, qs1.Sub(ov).Max(minV).Min(maxV).Add(center))

		gateNF := need.And(flat)
		if gateNF.ToInt16x8().ToBits().ReduceSum() == 0 {
			// No lane is flat: pure narrow update, p2/q2 and the outer taps
			// keep their originals (never loaded).
			lf8StoreP(pP1, np1.IfElse(need, p1))
			lf8StoreP(pP0, np0.IfElse(need, p0))
			lf8StoreP(base, nq0.IfElse(need, q0))
			lf8StoreP(pQ1, nq1.IfElse(need, q1))
			continue
		}

		// At least one flat lane: load the outer taps and decide flat8out.
		pP5 := unsafe.Add(base, -6*step)
		pP4 := unsafe.Add(base, -5*step)
		pQ4 := unsafe.Add(base, 4*step)
		pQ5 := unsafe.Add(base, 5*step)
		p6 := lf8LoadP(unsafe.Add(base, -7*step))
		p5 := lf8LoadP(pP5)
		p4 := lf8LoadP(pP4)
		q4 := lf8LoadP(pQ4)
		q5 := lf8LoadP(pQ5)
		q6 := lf8LoadP(unsafe.Add(base, 6*step))
		// flat8out mask (flatMask4 over p6,p5,p4,p0,q0,q4,q5,q6, inlined)
		flat2 := p4.AbsDiff(p0).LessEqual(flatThr).
			And(q4.AbsDiff(q0).LessEqual(flatThr)).
			And(p5.AbsDiff(p0).LessEqual(flatThr)).
			And(q5.AbsDiff(q0).LessEqual(flatThr)).
			And(p6.AbsDiff(p0).LessEqual(flatThr)).
			And(q6.AbsDiff(q0).LessEqual(flatThr))
		wideM := gateNF.And(flat2)

		// filter8 six-output flat branch as a running box sum:
		//   accB = 3*p3 + 2*p2 + p1 + p0 + q0 + 4, then per output
		//   accB += (incoming pair) - (outgoing pair); out = accB >> 3.
		accB := p3.ShiftAllLeftConst(1).Add(p3).
			Add(p2.ShiftAllLeftConst(1)).
			Add(p1).Add(p0).Add(q0).Add(four)

		if wideM.ToInt16x8().ToBits().ReduceSum() == 0 {
			// flat lanes but no wide lane: exact filter8 store pattern.
			f8p2 := accB.ShiftAllRightConst(3)
			lf8StoreP(pP2, f8p2.IfElse(flat, p2).IfElse(need, p2))
			accB = accB.Add(p1.Add(q1)).Sub(p3.Add(p2))
			f8p1 := accB.ShiftAllRightConst(3)
			lf8StoreP(pP1, f8p1.IfElse(flat, np1).IfElse(need, p1))
			accB = accB.Add(p0.Add(q2)).Sub(p3.Add(p1))
			f8p0 := accB.ShiftAllRightConst(3)
			lf8StoreP(pP0, f8p0.IfElse(flat, np0).IfElse(need, p0))
			accB = accB.Add(q0.Add(q3)).Sub(p3.Add(p0))
			f8q0 := accB.ShiftAllRightConst(3)
			lf8StoreP(base, f8q0.IfElse(flat, nq0).IfElse(need, q0))
			accB = accB.Add(q1.Add(q3)).Sub(p2.Add(q0))
			f8q1 := accB.ShiftAllRightConst(3)
			lf8StoreP(pQ1, f8q1.IfElse(flat, nq1).IfElse(need, q1))
			accB = accB.Add(q2.Add(q3)).Sub(p1.Add(q1))
			f8q2 := accB.ShiftAllRightConst(3)
			lf8StoreP(pQ2, f8q2.IfElse(flat, q2).IfElse(need, q2))
			continue
		}

		// Wide path: fourteen-tap running box sum,
		//   accW = 7*p6 + 2*p5 + 2*p4 + p3 + p2 + p1 + p0 + q0 + 8,
		// stepped by add(incoming pair)/sub(outgoing pair); out = accW >> 4.
		// The six inner outputs interleave the filter8 accumulator so each f8
		// value is blended and dead before the next is formed.
		accW := p6.ShiftAllLeftConst(3).Sub(p6).
			Add(p5.Add(p4).ShiftAllLeftConst(1)).
			Add(p3.Add(p2)).
			Add(p1.Add(p0)).
			Add(q0).Add(eight)
		lf8StoreP(pP5, accW.ShiftAllRightConst(4).IfElse(wideM, p5))
		accW = accW.Add(p3.Add(q1)).Sub(p6.ShiftAllLeftConst(1))
		lf8StoreP(pP4, accW.ShiftAllRightConst(4).IfElse(wideM, p4))
		accW = accW.Add(p2.Add(q2)).Sub(p6.Add(p5))
		lf8StoreP(pP3, accW.ShiftAllRightConst(4).IfElse(wideM, p3))
		accW = accW.Add(p1.Add(q3)).Sub(p6.Add(p4))
		f8p2 := accB.ShiftAllRightConst(3)
		lf8StoreP(pP2, accW.ShiftAllRightConst(4).IfElse(flat2, f8p2).IfElse(flat, p2).IfElse(need, p2))
		accW = accW.Add(p0.Add(q4)).Sub(p6.Add(p3))
		accB = accB.Add(p1.Add(q1)).Sub(p3.Add(p2))
		f8p1 := accB.ShiftAllRightConst(3)
		lf8StoreP(pP1, accW.ShiftAllRightConst(4).IfElse(flat2, f8p1).IfElse(flat, np1).IfElse(need, p1))
		accW = accW.Add(q0.Add(q5)).Sub(p6.Add(p2))
		accB = accB.Add(p0.Add(q2)).Sub(p3.Add(p1))
		f8p0 := accB.ShiftAllRightConst(3)
		lf8StoreP(pP0, accW.ShiftAllRightConst(4).IfElse(flat2, f8p0).IfElse(flat, np0).IfElse(need, p0))
		accW = accW.Add(q1.Add(q6)).Sub(p6.Add(p1))
		accB = accB.Add(q0.Add(q3)).Sub(p3.Add(p0))
		f8q0 := accB.ShiftAllRightConst(3)
		lf8StoreP(base, accW.ShiftAllRightConst(4).IfElse(flat2, f8q0).IfElse(flat, nq0).IfElse(need, q0))
		accW = accW.Add(q2.Add(q6)).Sub(p5.Add(p0))
		accB = accB.Add(q1.Add(q3)).Sub(p2.Add(q0))
		f8q1 := accB.ShiftAllRightConst(3)
		lf8StoreP(pQ1, accW.ShiftAllRightConst(4).IfElse(flat2, f8q1).IfElse(flat, nq1).IfElse(need, q1))
		accW = accW.Add(q3.Add(q6)).Sub(p4.Add(q0))
		accB = accB.Add(q2.Add(q3)).Sub(p1.Add(q1))
		f8q2 := accB.ShiftAllRightConst(3)
		lf8StoreP(pQ2, accW.ShiftAllRightConst(4).IfElse(flat2, f8q2).IfElse(flat, q2).IfElse(need, q2))
		accW = accW.Add(q4.Add(q6)).Sub(p3.Add(q1))
		lf8StoreP(pQ3, accW.ShiftAllRightConst(4).IfElse(wideM, q3))
		accW = accW.Add(q5.Add(q6)).Sub(p2.Add(q2))
		lf8StoreP(pQ4, accW.ShiftAllRightConst(4).IfElse(wideM, q4))
		accW = accW.Add(q6.ShiftAllLeftConst(1)).Sub(p1.Add(q3))
		lf8StoreP(pQ5, accW.ShiftAllRightConst(4).IfElse(wideM, q5))
	}
	if rem := length - groups*8; rem > 0 {
		filter14EdgePureGo(pix, q0Base+groups*8, step, outer, rem, scale, params)
	}
}

// filter14VertSIMD mirrors filter14VertNEON — vertical edges are transposed in
// batches through the NEON trn-ladder gather/scatter into a stack scratch laid
// out as a horizontal edge — but runs the Go-native SIMD horizontal core on the
// scratch instead of the asm. Byte-exactness follows by construction: the core
// consumes exactly the bytes the reference transpose feeds it.
func filter14VertSIMD(pix []byte, q0Base int, step int, outer int, length int, scale int, params filter4Params) {
	groups := length / 8
	if step != 1 || groups == 0 {
		filter14EdgePureGo(pix, q0Base, step, outer, length, scale, params)
		return
	}
	const scratchStride = 8 * filter14VertBatchGroups
	var scratch [14 * scratchStride]byte
	for g := 0; g < groups; g += filter14VertBatchGroups {
		n := groups - g
		if n > filter14VertBatchGroups {
			n = filter14VertBatchGroups
		}
		colBase := q0Base + g*8*outer
		ctx := wideVertTransposeCtx{
			src:     &pix[colBase-7], // p6 of the first position
			stride:  uintptr(outer),
			scratch: &scratch[0],
			count:   uintptr(n),
		}
		filter14VertGatherNEONAsm(&ctx)
		sq0 := 7 * scratchStride // q0 tap row
		filter14EdgeSIMD(scratch[:], sq0, scratchStride, 1, n*8, scale, params)
		filter14VertScatterNEONAsm(&ctx)
	}
	if rem := length - groups*8; rem > 0 {
		filter14EdgePureGo(pix, q0Base+groups*8*outer, step, outer, rem, scale, params)
	}
}
