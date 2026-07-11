// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

// Go-native SIMD 8-bit six-sample (filter6) deblocking kernel, int16 8-wide
// (simd/archsimd). Eight edge positions per Int16x8, one lane per position;
// contiguous-horizontal layout only (see filter4_gosimd_arm64.go). Vertical
// edges and short tails route through the hand-written NEON asm / pure-Go
// reference via filter6EdgeNEON's own fallbacks.
//
// Structure (vs filter6EdgeNEONAsm, a straight-line kernel that re-sums each
// of the four flat averages from scratch):
//   - the flat decision gates a branch: groups with no need&&flat lane store
//     only the four narrow-gated taps and skip the flat sums entirely.
//   - the four flat averages are a running box sum (dav1d-style incremental
//     add/subtract of the sliding 8-wide window) with the rounding bias folded
//     into the accumulator, so each output after the first costs two pair adds,
//     two accumulator ops and one arithmetic shift.
//
// Byte-exactness with filter6EdgePureGo:
//   - needsFilter6 and the four-term flat decision are reproduced lane-wise.
//   - the sliding sums are algebraically identical to the reference's weighted
//     averages (non-negative, at most 255*8+4, inside int16), so
//     (sum+4)>>3 == roundPowerOfTwo(sum, 3) exactly.
//   - the not-flat lanes reuse the exact filter4 narrow update; every blend is
//     an IfElse chain over flat/need matching the reference's conditional
//     writes.

package loopfilter

import (
	"simd/archsimd"
	"unsafe"
)

// filter6EdgeSIMD is the Go-native SIMD form of filter6EdgePureGo for 8-bit
// horizontal edges (outer == 1, contiguous taps).
func filter6EdgeSIMD(pix []byte, q0Base int, step int, outer int, length int, scale int, params filter4Params) {
	groups := length / 8
	if outer != 1 || groups == 0 {
		filter6EdgeNEON(pix, q0Base, step, outer, length, scale, params)
		return
	}
	limit := archsimd.BroadcastInt16x8(params.limit)
	blimit := archsimd.BroadcastInt16x8(params.blimit)
	hevT := archsimd.BroadcastInt16x8(params.hev)
	center := archsimd.BroadcastInt16x8(params.center)
	minV := archsimd.BroadcastInt16x8(params.min)
	maxV := archsimd.BroadcastInt16x8(params.max)
	one := archsimd.BroadcastInt16x8(1)
	three16 := archsimd.BroadcastInt16x8(3)
	four := archsimd.BroadcastInt16x8(4)
	flatThr := archsimd.BroadcastInt16x8(int16(scale))
	q0p := unsafe.Pointer(&pix[q0Base])
	for g := 0; g < groups; g++ {
		base := unsafe.Add(q0p, g*8)
		pP1 := unsafe.Add(base, -2*step)
		pP0 := unsafe.Add(base, -step)
		pQ1 := unsafe.Add(base, step)
		p2 := lf8LoadP(unsafe.Add(base, -3*step))
		p1 := lf8LoadP(pP1)
		p0 := lf8LoadP(pP0)
		q0 := lf8LoadP(base)
		q1 := lf8LoadP(pQ1)
		q2 := lf8LoadP(unsafe.Add(base, 2*step))

		// needsFilter6 (inlined)
		need := p2.AbsDiff(p1).LessEqual(limit).
			And(p1.AbsDiff(p0).LessEqual(limit)).
			And(q1.AbsDiff(q0).LessEqual(limit)).
			And(q2.AbsDiff(q1).LessEqual(limit)).
			And(p0.AbsDiff(q0).ShiftAllLeftConst(1).
				Add(p1.AbsDiff(q1).ShiftAllRightConst(1)).
				LessEqual(blimit))
		// four-term flat decision (inlined)
		flat := p1.AbsDiff(p0).LessEqual(flatThr).
			And(q1.AbsDiff(q0).LessEqual(flatThr)).
			And(p2.AbsDiff(p0).LessEqual(flatThr)).
			And(q2.AbsDiff(q0).LessEqual(flatThr))
		// narrow four-tap (inlined filter4TapNoGate), needed on both paths
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

		if need.And(flat).ToInt16x8().ToBits().ReduceSum() == 0 {
			// No flat lane: pure narrow update.
			lf8StoreP(pP1, np1.IfElse(need, p1))
			lf8StoreP(pP0, np0.IfElse(need, p0))
			lf8StoreP(base, nq0.IfElse(need, q0))
			lf8StoreP(pQ1, nq1.IfElse(need, q1))
			continue
		}

		// Sliding flat sums: acc = 3*p2 + 2*p1 + 2*p0 + q0 + 4, then
		// acc += (incoming pair) - (outgoing pair); out = acc >> 3.
		acc := p2.ShiftAllLeftConst(1).Add(p2).
			Add(p1.Add(p0).ShiftAllLeftConst(1)).
			Add(q0).Add(four)
		lf8StoreP(pP1, acc.ShiftAllRightConst(3).IfElse(flat, np1).IfElse(need, p1))
		acc = acc.Add(q0.Add(q1)).Sub(p2.ShiftAllLeftConst(1))
		lf8StoreP(pP0, acc.ShiftAllRightConst(3).IfElse(flat, np0).IfElse(need, p0))
		acc = acc.Add(q1.Add(q2)).Sub(p2.Add(p1))
		lf8StoreP(base, acc.ShiftAllRightConst(3).IfElse(flat, nq0).IfElse(need, q0))
		acc = acc.Add(q2.ShiftAllLeftConst(1)).Sub(p1.Add(p0))
		lf8StoreP(pQ1, acc.ShiftAllRightConst(3).IfElse(flat, nq1).IfElse(need, q1))
	}
	if rem := length - groups*8; rem > 0 {
		filter6EdgePureGo(pix, q0Base+groups*8, step, outer, rem, scale, params)
	}
}
