// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

// Go-native SIMD 8-bit and 10/12-bit wide eight-sample (filter8) deblocking
// kernels, int16 8-wide (simd/archsimd). Eight edge positions per Int16x8, one
// lane per position; same contiguous-horizontal data layout as the narrow
// kernel (see filter4_gosimd_arm64.go). Vertical edges and short tails route
// through the hand-written NEON asm, which owns the ld4/st4 transpose.
//
// Byte-exactness with filter8EdgePureGo / filter8Edge16PureGo:
//   - needsFilter8 and the flat decision are reproduced lane-wise (AbsDiff
//     compared LessEqual against broadcast limit/blimit and the flat threshold),
//     forming a need mask and a flat mask.
//   - the flat lanes use the six weighted-average outputs; every sum is a sum of
//     seven/eight non-negative samples, so at 8-bit the max is 255*8 = 2040 and
//     at 12-bit 4095*8 = 32760, both < 32767, and (sum+4)>>3 (Add(4) then
//     arithmetic ShiftAllRight(3)) reproduces roundPowerOfTwo(sum, 3) exactly
//     for the non-negative sums.
//   - the not-flat lanes reuse filter4TapSIMD, exactly as filter8Samples calls
//     filter4Samples unconditionally on the non-flat branch.
//   - the flat/not-flat choice is a per-output IfElse(flatMask); the whole result
//     is gated by IfElse(needMask, original), matching filter8EdgePureGo's
//     conditional writes (p2/q2 keep their original samples on the not-flat
//     branch, and every tap keeps its original where needsFilter8 is false).

package loopfilter

import (
	"simd/archsimd"
	"unsafe"
)

// filter8SIMDConst extends the narrow-kernel constants with the flat threshold
// (== scale; 1 at 8-bit) and the flat-average rounding bias.
type filter8SIMDConst struct {
	narrow filter4SIMDConst
	flat   archsimd.Int16x8 // flat threshold (scale)
	roundD archsimd.Int16x8 // rounding bias for (sum+4)>>3
}

func makeFilter8SIMDConst(scale int, params filter4Params) filter8SIMDConst {
	return filter8SIMDConst{
		narrow: makeFilter4SIMDConst(params),
		flat:   archsimd.BroadcastInt16x8(int16(scale)),
		roundD: archsimd.BroadcastInt16x8(4),
	}
}

// flatMask8SIMD reproduces the filter8Samples flat decision lane-wise.
func flatMask8SIMD(p3, p2, p1, p0, q0, q1, q2, q3 archsimd.Int16x8, cst filter8SIMDConst) archsimd.Mask16x8 {
	thr := cst.flat
	return p1.AbsDiff(p0).LessEqual(thr).
		And(q1.AbsDiff(q0).LessEqual(thr)).
		And(p2.AbsDiff(p0).LessEqual(thr)).
		And(q2.AbsDiff(q0).LessEqual(thr)).
		And(p3.AbsDiff(p0).LessEqual(thr)).
		And(q3.AbsDiff(q0).LessEqual(thr))
}

// needsFilter8Mask reproduces needsFilter8 lane-wise.
func needsFilter8Mask(p3, p2, p1, p0, q0, q1, q2, q3 archsimd.Int16x8, cst filter8SIMDConst) archsimd.Mask16x8 {
	lim := cst.narrow.limit
	return p3.AbsDiff(p2).LessEqual(lim).
		And(p2.AbsDiff(p1).LessEqual(lim)).
		And(p1.AbsDiff(p0).LessEqual(lim)).
		And(q1.AbsDiff(q0).LessEqual(lim)).
		And(q2.AbsDiff(q1).LessEqual(lim)).
		And(q3.AbsDiff(q2).LessEqual(lim)).
		And(p0.AbsDiff(q0).ShiftAllLeftConst(1).
			Add(p1.AbsDiff(q1).ShiftAllRightConst(1)).
			LessEqual(cst.narrow.blimit))
}

// round3SIMD computes roundPowerOfTwo(sum, 3) = (sum+4)>>3 for non-negative sum.
func round3SIMD(sum archsimd.Int16x8, cst filter8SIMDConst) archsimd.Int16x8 {
	return sum.Add(cst.roundD).ShiftAllRightConst(3)
}

// filter8FlatSIMD computes the six flat-branch weighted averages for one group.
// Each is roundPowerOfTwo(sum, 3) over seven/eight non-negative samples; see the
// file header for the non-overflow argument. Kept small so it inlines into the
// edge loops (the flat sums account for most of filter8's instruction count).
func filter8FlatSIMD(p3, p2, p1, p0, q0, q1, q2, q3 archsimd.Int16x8, cst filter8SIMDConst) (fp2, fp1, fp0, fq0, fq1, fq2 archsimd.Int16x8) {
	three := cst.narrow.three16
	fp2 = round3SIMD(p3.Mul(three).Add(p2.ShiftAllLeftConst(1)).Add(p1).Add(p0).Add(q0), cst)
	fp1 = round3SIMD(p3.ShiftAllLeftConst(1).Add(p2).Add(p1.ShiftAllLeftConst(1)).Add(p0).Add(q0).Add(q1), cst)
	fp0 = round3SIMD(p3.Add(p2).Add(p1).Add(p0.ShiftAllLeftConst(1)).Add(q0).Add(q1).Add(q2), cst)
	fq0 = round3SIMD(p2.Add(p1).Add(p0).Add(q0.ShiftAllLeftConst(1)).Add(q1).Add(q2).Add(q3), cst)
	fq1 = round3SIMD(p1.Add(p0).Add(q0).Add(q1.ShiftAllLeftConst(1)).Add(q2).Add(q3.ShiftAllLeftConst(1)), cst)
	fq2 = round3SIMD(p0.Add(q0).Add(q1).Add(q2.ShiftAllLeftConst(1)).Add(q3.Mul(three)), cst)
	return
}

// filter8KernelSIMD is the canonical (outlined) form of the eight-sample kernel,
// byte-identical to filter8Samples gated by needsFilter8. The hot edge loops
// paste this arithmetic inline instead of calling it, because an outlined call
// pushes the sixteen-plus Int16x8 args/results through the stack ABI, spilling
// the tap vectors (see filter4's file header). This function stays for reference
// and testing.
func filter8KernelSIMD(p3, p2, p1, p0, q0, q1, q2, q3 archsimd.Int16x8, cst filter8SIMDConst) (op2, op1, op0, oq0, oq1, oq2 archsimd.Int16x8) {
	need := needsFilter8Mask(p3, p2, p1, p0, q0, q1, q2, q3, cst)
	flat := flatMask8SIMD(p3, p2, p1, p0, q0, q1, q2, q3, cst)
	np1, np0, nq0, nq1 := filter4TapNoGate(p1, p0, q0, q1, cst.narrow)
	fp2, fp1, fp0, fq0, fq1, fq2 := filter8FlatSIMD(p3, p2, p1, p0, q0, q1, q2, q3, cst)
	op2 = fp2.IfElse(flat, p2).IfElse(need, p2)
	op1 = fp1.IfElse(flat, np1).IfElse(need, p1)
	op0 = fp0.IfElse(flat, np0).IfElse(need, p0)
	oq0 = fq0.IfElse(flat, nq0).IfElse(need, q0)
	oq1 = fq1.IfElse(flat, nq1).IfElse(need, q1)
	oq2 = fq2.IfElse(flat, q2).IfElse(need, q2)
	return
}

// filter8EdgeSIMD is the Go-native SIMD form of filter8EdgePureGo for 8-bit
// horizontal edges. Vertical edges and short tails route through the NEON asm
// (which handles the transpose) or the pure-Go reference. The per-group math is
// pasted inline so the tap vectors stay register-resident (an outlined call
// costs ~1.15x here; the body is large enough to spill somewhat even inlined,
// which is why filter8 stays further behind the NEON asm than filter4).
func filter8EdgeSIMD(pix []byte, q0Base int, step int, outer int, length int, scale int, params filter4Params) {
	groups := length / 8
	if outer != 1 || groups == 0 {
		filter8EdgeNEON(pix, q0Base, step, outer, length, scale, params)
		return
	}
	// Constants as flat locals (no returned/nested struct) so they stay
	// register-resident; the helper bodies are inlined below and each flat
	// output is computed-blended-stored in turn so the six flat results never
	// co-exist — the fixes that took filter4 to parity, applied to the 8-tap.
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
	roundD := archsimd.BroadcastInt16x8(4)
	q0p := unsafe.Pointer(&pix[q0Base])
	for g := 0; g < groups; g++ {
		base := unsafe.Add(q0p, g*8)
		pP2 := unsafe.Add(base, -3*step)
		pP1 := unsafe.Add(base, -2*step)
		pP0 := unsafe.Add(base, -step)
		pQ1 := unsafe.Add(base, step)
		pQ2 := unsafe.Add(base, 2*step)
		p3 := lf8LoadP(unsafe.Add(base, -4*step))
		p2 := lf8LoadP(pP2)
		p1 := lf8LoadP(pP1)
		p0 := lf8LoadP(pP0)
		q0 := lf8LoadP(base)
		q1 := lf8LoadP(pQ1)
		q2 := lf8LoadP(pQ2)
		q3 := lf8LoadP(unsafe.Add(base, 3*step))

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
		// flat mask (inlined)
		flat := p1.AbsDiff(p0).LessEqual(flatThr).
			And(q1.AbsDiff(q0).LessEqual(flatThr)).
			And(p2.AbsDiff(p0).LessEqual(flatThr)).
			And(q2.AbsDiff(q0).LessEqual(flatThr)).
			And(p3.AbsDiff(p0).LessEqual(flatThr)).
			And(q3.AbsDiff(q0).LessEqual(flatThr))
		// narrow four-tap (inlined filter4TapNoGate; clamp = Max(minV).Min(maxV))
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
		// six flat averages (inlined filter8FlatSIMD + round3), each computed,
		// blended and stored before the next so they never all live at once.
		fp2 := p3.Mul(three16).Add(p2.ShiftAllLeftConst(1)).Add(p1).Add(p0).Add(q0).Add(roundD).ShiftAllRightConst(3)
		lf8StoreP(pP2, fp2.IfElse(flat, p2).IfElse(need, p2))
		fp1 := p3.ShiftAllLeftConst(1).Add(p2).Add(p1.ShiftAllLeftConst(1)).Add(p0).Add(q0).Add(q1).Add(roundD).ShiftAllRightConst(3)
		lf8StoreP(pP1, fp1.IfElse(flat, np1).IfElse(need, p1))
		fp0 := p3.Add(p2).Add(p1).Add(p0.ShiftAllLeftConst(1)).Add(q0).Add(q1).Add(q2).Add(roundD).ShiftAllRightConst(3)
		lf8StoreP(pP0, fp0.IfElse(flat, np0).IfElse(need, p0))
		fq0 := p2.Add(p1).Add(p0).Add(q0.ShiftAllLeftConst(1)).Add(q1).Add(q2).Add(q3).Add(roundD).ShiftAllRightConst(3)
		lf8StoreP(base, fq0.IfElse(flat, nq0).IfElse(need, q0))
		fq1 := p1.Add(p0).Add(q0).Add(q1.ShiftAllLeftConst(1)).Add(q2).Add(q3.ShiftAllLeftConst(1)).Add(roundD).ShiftAllRightConst(3)
		lf8StoreP(pQ1, fq1.IfElse(flat, nq1).IfElse(need, q1))
		fq2 := p0.Add(q0).Add(q1).Add(q2.ShiftAllLeftConst(1)).Add(q3.Mul(three16)).Add(roundD).ShiftAllRightConst(3)
		lf8StoreP(pQ2, fq2.IfElse(flat, q2).IfElse(need, q2))
	}
	if rem := length - groups*8; rem > 0 {
		filter8EdgePureGo(pix, q0Base+groups*8, step, outer, rem, scale, params)
	}
}

// filter8Edge16SIMD is the Go-native SIMD form of filter8Edge16PureGo for
// 10/12-bit horizontal edges (contiguous two-byte taps).
func filter8Edge16SIMD(pix []byte, q0Base int, step int, outer int, length int, scale int, params filter4Params) {
	groups := length / 8
	if outer != 2 || groups == 0 {
		filter8Edge16NEON(pix, q0Base, step, outer, length, scale, params)
		return
	}
	// Constants as flat locals + inlined helper bodies + per-output compute-
	// blend-store (matches filter8EdgeSIMD; keeps constants register-resident
	// and avoids holding all six flat results at once).
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
	roundD := archsimd.BroadcastInt16x8(4)
	q0p := unsafe.Pointer(&pix[q0Base])
	for g := 0; g < groups; g++ {
		base := unsafe.Add(q0p, g*16) // 8 positions * 2 bytes
		pP2 := unsafe.Add(base, -3*step)
		pP1 := unsafe.Add(base, -2*step)
		pP0 := unsafe.Add(base, -step)
		pQ1 := unsafe.Add(base, step)
		pQ2 := unsafe.Add(base, 2*step)
		p3 := lf16LoadP(unsafe.Add(base, -4*step))
		p2 := lf16LoadP(pP2)
		p1 := lf16LoadP(pP1)
		p0 := lf16LoadP(pP0)
		q0 := lf16LoadP(base)
		q1 := lf16LoadP(pQ1)
		q2 := lf16LoadP(pQ2)
		q3 := lf16LoadP(unsafe.Add(base, 3*step))

		need := p3.AbsDiff(p2).LessEqual(limit).
			And(p2.AbsDiff(p1).LessEqual(limit)).
			And(p1.AbsDiff(p0).LessEqual(limit)).
			And(q1.AbsDiff(q0).LessEqual(limit)).
			And(q2.AbsDiff(q1).LessEqual(limit)).
			And(q3.AbsDiff(q2).LessEqual(limit)).
			And(p0.AbsDiff(q0).ShiftAllLeftConst(1).
				Add(p1.AbsDiff(q1).ShiftAllRightConst(1)).
				LessEqual(blimit))
		flat := p1.AbsDiff(p0).LessEqual(flatThr).
			And(q1.AbsDiff(q0).LessEqual(flatThr)).
			And(p2.AbsDiff(p0).LessEqual(flatThr)).
			And(q2.AbsDiff(q0).LessEqual(flatThr)).
			And(p3.AbsDiff(p0).LessEqual(flatThr)).
			And(q3.AbsDiff(q0).LessEqual(flatThr))
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
		fp2 := p3.Mul(three16).Add(p2.ShiftAllLeftConst(1)).Add(p1).Add(p0).Add(q0).Add(roundD).ShiftAllRightConst(3)
		lf16StoreP(pP2, fp2.IfElse(flat, p2).IfElse(need, p2))
		fp1 := p3.ShiftAllLeftConst(1).Add(p2).Add(p1.ShiftAllLeftConst(1)).Add(p0).Add(q0).Add(q1).Add(roundD).ShiftAllRightConst(3)
		lf16StoreP(pP1, fp1.IfElse(flat, np1).IfElse(need, p1))
		fp0 := p3.Add(p2).Add(p1).Add(p0.ShiftAllLeftConst(1)).Add(q0).Add(q1).Add(q2).Add(roundD).ShiftAllRightConst(3)
		lf16StoreP(pP0, fp0.IfElse(flat, np0).IfElse(need, p0))
		fq0 := p2.Add(p1).Add(p0).Add(q0.ShiftAllLeftConst(1)).Add(q1).Add(q2).Add(q3).Add(roundD).ShiftAllRightConst(3)
		lf16StoreP(base, fq0.IfElse(flat, nq0).IfElse(need, q0))
		fq1 := p1.Add(p0).Add(q0).Add(q1.ShiftAllLeftConst(1)).Add(q2).Add(q3.ShiftAllLeftConst(1)).Add(roundD).ShiftAllRightConst(3)
		lf16StoreP(pQ1, fq1.IfElse(flat, nq1).IfElse(need, q1))
		fq2 := p0.Add(q0).Add(q1).Add(q2.ShiftAllLeftConst(1)).Add(q3.Mul(three16)).Add(roundD).ShiftAllRightConst(3)
		lf16StoreP(pQ2, fq2.IfElse(flat, q2).IfElse(need, q2))
	}
	if rem := length - groups*8; rem > 0 {
		filter8Edge16PureGo(pix, q0Base+groups*8*outer, step, outer, rem, scale, params)
	}
}
