// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

// Go-native SIMD high-bit-depth fourteen-tap deblocking kernel.

package loopfilter

import (
	"simd/archsimd"
	"unsafe"
)

// filter14Edge16SIMD is the Go-native SIMD form of filter14Edge16PureGo for
// 10/12-bit horizontal edges (contiguous two-byte taps). Its structure uses
// masks once, the dav1d branch ladder, and running box sums, with
// one domain change: the fourteen-tap window sum reaches 4095*16 + 8 = 65528
// at 12-bit, which overflows int16. The wide accumulator carries a -32768
// offset (T = sum + 8 - 32768 is always
// in [-32760, 32760], exactly representable), so every output is one
// arithmetic ShiftAllRight(4) plus an Add(2048) — immediate shifts, pure
// int16 lanes, and one code path for both bit depths. Intermediate wrap-around
// during the add/subtract stepping is exact mod 2^16, and the true value fits
// int16 at every shift point. The six filter8 sums peak at 4095*8+4 = 32764
// and stay unbiased. Vertical edges reuse the NEON trn-ladder gather/scatter
// around this core; other layouts and short tails
// route through the pure-Go reference (matching the NEON wrapper's own
// fallbacks).
func filter14Edge16SIMD(pix []byte, q0Base int, step int, outer int, length int, scale int, params filter4Params) {
	groups := length / 8
	if outer != 2 || groups == 0 {
		if step == 2 && groups > 0 {
			filter14Vert16SIMD(pix, q0Base, step, outer, length, scale, params)
			return
		}
		filter14Edge16PureGo(pix, q0Base, step, outer, length, scale, params)
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
	biasW := archsimd.BroadcastInt16x8(8 - 32768) // wide rounding bias with the -32768 offset folded in
	half := archsimd.BroadcastInt16x8(2048)       // 32768 >> 4, restored after each wide shift
	flatThr := archsimd.BroadcastInt16x8(int16(scale))
	q0p := unsafe.Pointer(&pix[q0Base])
	for g := 0; g < groups; g++ {
		base := unsafe.Add(q0p, g*16) // 8 positions * 2 bytes
		pP3 := unsafe.Add(base, -4*step)
		pP2 := unsafe.Add(base, -3*step)
		pP1 := unsafe.Add(base, -2*step)
		pP0 := unsafe.Add(base, -step)
		pQ1 := unsafe.Add(base, step)
		pQ2 := unsafe.Add(base, 2*step)
		pQ3 := unsafe.Add(base, 3*step)
		p3 := lf16LoadP(pP3)
		p2 := lf16LoadP(pP2)
		p1 := lf16LoadP(pP1)
		p0 := lf16LoadP(pP0)
		q0 := lf16LoadP(base)
		q1 := lf16LoadP(pQ1)
		q2 := lf16LoadP(pQ2)
		q3 := lf16LoadP(pQ3)

		need := lfAbsDiffInt16x8(p3, p2).LessEqual(limit).
			And(lfAbsDiffInt16x8(p2, p1).LessEqual(limit)).
			And(lfAbsDiffInt16x8(p1, p0).LessEqual(limit)).
			And(lfAbsDiffInt16x8(q1, q0).LessEqual(limit)).
			And(lfAbsDiffInt16x8(q2, q1).LessEqual(limit)).
			And(lfAbsDiffInt16x8(q3, q2).LessEqual(limit)).
			And(lfAbsDiffInt16x8(p0, q0).ShiftAllLeft(1).
				Add(lfAbsDiffInt16x8(p1, q1).ShiftAllRight(1)).
				LessEqual(blimit))
		flat := lfAbsDiffInt16x8(p1, p0).LessEqual(flatThr).
			And(lfAbsDiffInt16x8(q1, q0).LessEqual(flatThr)).
			And(lfAbsDiffInt16x8(p2, p0).LessEqual(flatThr)).
			And(lfAbsDiffInt16x8(q2, q0).LessEqual(flatThr)).
			And(lfAbsDiffInt16x8(p3, p0).LessEqual(flatThr)).
			And(lfAbsDiffInt16x8(q3, q0).LessEqual(flatThr))
		hev := lfAbsDiffInt16x8(p1, p0).Greater(hevT).Or(lfAbsDiffInt16x8(q1, q0).Greater(hevT))
		ps1 := p1.Sub(center)
		ps0 := p0.Sub(center)
		qs0 := q0.Sub(center)
		qs1 := q1.Sub(center)
		f := ps1.Sub(qs1).Max(minV).Min(maxV).Masked(hev)
		f = f.Add(qs0.Sub(ps0).Mul(three16)).Max(minV).Min(maxV)
		filter1 := f.Add(four).Max(minV).Min(maxV).ShiftAllRight(3)
		filter2 := f.Add(three16).Max(minV).Min(maxV).ShiftAllRight(3)
		np0 := ps0.Add(filter2).Max(minV).Min(maxV).Add(center)
		nq0 := qs0.Sub(filter1).Max(minV).Min(maxV).Add(center)
		ov := filter1.Add(one).ShiftAllRight(1)
		np1 := p1.IfElse(hev, ps1.Add(ov).Max(minV).Min(maxV).Add(center))
		nq1 := q1.IfElse(hev, qs1.Sub(ov).Max(minV).Min(maxV).Add(center))

		gateNF := need.And(flat)
		if gateNF.ToInt16x8().ToBits().ReduceSum() == 0 {
			lf16StoreP(pP1, np1.IfElse(need, p1))
			lf16StoreP(pP0, np0.IfElse(need, p0))
			lf16StoreP(base, nq0.IfElse(need, q0))
			lf16StoreP(pQ1, nq1.IfElse(need, q1))
			continue
		}

		pP5 := unsafe.Add(base, -6*step)
		pP4 := unsafe.Add(base, -5*step)
		pQ4 := unsafe.Add(base, 4*step)
		pQ5 := unsafe.Add(base, 5*step)
		p6 := lf16LoadP(unsafe.Add(base, -7*step))
		p5 := lf16LoadP(pP5)
		p4 := lf16LoadP(pP4)
		q4 := lf16LoadP(pQ4)
		q5 := lf16LoadP(pQ5)
		q6 := lf16LoadP(unsafe.Add(base, 6*step))
		flat2 := lfAbsDiffInt16x8(p4, p0).LessEqual(flatThr).
			And(lfAbsDiffInt16x8(q4, q0).LessEqual(flatThr)).
			And(lfAbsDiffInt16x8(p5, p0).LessEqual(flatThr)).
			And(lfAbsDiffInt16x8(q5, q0).LessEqual(flatThr)).
			And(lfAbsDiffInt16x8(p6, p0).LessEqual(flatThr)).
			And(lfAbsDiffInt16x8(q6, q0).LessEqual(flatThr))
		wideM := gateNF.And(flat2)

		accB := p3.ShiftAllLeft(1).Add(p3).
			Add(p2.ShiftAllLeft(1)).
			Add(p1).Add(p0).Add(q0).Add(four)

		if wideM.ToInt16x8().ToBits().ReduceSum() == 0 {
			f8p2 := accB.ShiftAllRight(3)
			lf16StoreP(pP2, f8p2.IfElse(flat, p2).IfElse(need, p2))
			accB = accB.Add(p1.Add(q1)).Sub(p3.Add(p2))
			f8p1 := accB.ShiftAllRight(3)
			lf16StoreP(pP1, f8p1.IfElse(flat, np1).IfElse(need, p1))
			accB = accB.Add(p0.Add(q2)).Sub(p3.Add(p1))
			f8p0 := accB.ShiftAllRight(3)
			lf16StoreP(pP0, f8p0.IfElse(flat, np0).IfElse(need, p0))
			accB = accB.Add(q0.Add(q3)).Sub(p3.Add(p0))
			f8q0 := accB.ShiftAllRight(3)
			lf16StoreP(base, f8q0.IfElse(flat, nq0).IfElse(need, q0))
			accB = accB.Add(q1.Add(q3)).Sub(p2.Add(q0))
			f8q1 := accB.ShiftAllRight(3)
			lf16StoreP(pQ1, f8q1.IfElse(flat, nq1).IfElse(need, q1))
			accB = accB.Add(q2.Add(q3)).Sub(p1.Add(q1))
			f8q2 := accB.ShiftAllRight(3)
			lf16StoreP(pQ2, f8q2.IfElse(flat, q2).IfElse(need, q2))
			continue
		}

		accW := p6.ShiftAllLeft(3).Sub(p6).
			Add(p5.Add(p4).ShiftAllLeft(1)).
			Add(p3.Add(p2)).
			Add(p1.Add(p0)).
			Add(q0).Add(biasW)
		lf16StoreP(pP5, accW.ShiftAllRight(4).Add(half).IfElse(wideM, p5))
		accW = accW.Add(p3.Add(q1)).Sub(p6.Add(p6))
		lf16StoreP(pP4, accW.ShiftAllRight(4).Add(half).IfElse(wideM, p4))
		accW = accW.Add(p2.Add(q2)).Sub(p6.Add(p5))
		lf16StoreP(pP3, accW.ShiftAllRight(4).Add(half).IfElse(wideM, p3))
		accW = accW.Add(p1.Add(q3)).Sub(p6.Add(p4))
		f8p2 := accB.ShiftAllRight(3)
		lf16StoreP(pP2, accW.ShiftAllRight(4).Add(half).IfElse(flat2, f8p2).IfElse(flat, p2).IfElse(need, p2))
		accW = accW.Add(p0.Add(q4)).Sub(p6.Add(p3))
		accB = accB.Add(p1.Add(q1)).Sub(p3.Add(p2))
		f8p1 := accB.ShiftAllRight(3)
		lf16StoreP(pP1, accW.ShiftAllRight(4).Add(half).IfElse(flat2, f8p1).IfElse(flat, np1).IfElse(need, p1))
		accW = accW.Add(q0.Add(q5)).Sub(p6.Add(p2))
		accB = accB.Add(p0.Add(q2)).Sub(p3.Add(p1))
		f8p0 := accB.ShiftAllRight(3)
		lf16StoreP(pP0, accW.ShiftAllRight(4).Add(half).IfElse(flat2, f8p0).IfElse(flat, np0).IfElse(need, p0))
		accW = accW.Add(q1.Add(q6)).Sub(p6.Add(p1))
		accB = accB.Add(q0.Add(q3)).Sub(p3.Add(p0))
		f8q0 := accB.ShiftAllRight(3)
		lf16StoreP(base, accW.ShiftAllRight(4).Add(half).IfElse(flat2, f8q0).IfElse(flat, nq0).IfElse(need, q0))
		accW = accW.Add(q2.Add(q6)).Sub(p5.Add(p0))
		accB = accB.Add(q1.Add(q3)).Sub(p2.Add(q0))
		f8q1 := accB.ShiftAllRight(3)
		lf16StoreP(pQ1, accW.ShiftAllRight(4).Add(half).IfElse(flat2, f8q1).IfElse(flat, nq1).IfElse(need, q1))
		accW = accW.Add(q3.Add(q6)).Sub(p4.Add(q0))
		accB = accB.Add(q2.Add(q3)).Sub(p1.Add(q1))
		f8q2 := accB.ShiftAllRight(3)
		lf16StoreP(pQ2, accW.ShiftAllRight(4).Add(half).IfElse(flat2, f8q2).IfElse(flat, q2).IfElse(need, q2))
		accW = accW.Add(q4.Add(q6)).Sub(p3.Add(q1))
		lf16StoreP(pQ3, accW.ShiftAllRight(4).Add(half).IfElse(wideM, q3))
		accW = accW.Add(q5.Add(q6)).Sub(p2.Add(q2))
		lf16StoreP(pQ4, accW.ShiftAllRight(4).Add(half).IfElse(wideM, q4))
		accW = accW.Add(q6.Add(q6)).Sub(p1.Add(q3))
		lf16StoreP(pQ5, accW.ShiftAllRight(4).Add(half).IfElse(wideM, q5))
	}
	if rem := length - groups*8; rem > 0 {
		filter14Edge16PureGo(pix, q0Base+groups*8*outer, step, outer, rem, scale, params)
	}
}

// filter14Vert16SIMD mirrors filter14Vert16NEON — vertical 16-bit edges are
// transposed in batches through the NEON trn-ladder gather/scatter into a
// stack scratch laid out as a horizontal edge — but runs the Go-native SIMD
// horizontal core, which (unlike the asm) also covers 12-bit.
func filter14Vert16SIMD(pix []byte, q0Base int, step int, outer int, length int, scale int, params filter4Params) {
	groups := length / 8
	if step != 2 || groups == 0 {
		filter14Edge16PureGo(pix, q0Base, step, outer, length, scale, params)
		return
	}
	const scratchStride = 16 * filter14VertBatchGroups
	var scratch [14 * scratchStride]byte
	for g := 0; g < groups; g += filter14VertBatchGroups {
		n := groups - g
		if n > filter14VertBatchGroups {
			n = filter14VertBatchGroups
		}
		colBase := q0Base + g*8*outer
		ctx := wideVertTransposeCtx{
			src:     &pix[colBase-7*step], // p6 of the first position
			stride:  uintptr(outer),
			scratch: &scratch[0],
			count:   uintptr(n),
		}
		filter14Vert16GatherNEONAsm(&ctx)
		filter14Edge16SIMD(scratch[:], 7*scratchStride, scratchStride, 2, n*8, scale, params)
		filter14Vert16ScatterNEONAsm(&ctx)
	}
	if rem := length - groups*8; rem > 0 {
		filter14Edge16PureGo(pix, q0Base+groups*8*outer, step, outer, rem, scale, params)
	}
}
