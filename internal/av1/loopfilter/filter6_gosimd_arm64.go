// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

// Go-native SIMD high-bit-depth six-tap deblocking kernel.

package loopfilter

import (
	"simd/archsimd"
	"unsafe"
)

// filter6Edge16SIMD is the Go-native SIMD form of filter6Edge16PureGo for
// 10/12-bit edges. The horizontal sums stay inside int16 at both depths (max
// 4095*8 + 4 = 32764), so no accumulator offset is needed. The NEON horizontal
// core has been removed; vertical SIMD reuses its gather/scatter routines.
// Other layouts and short tails use the pure-Go reference.
func filter6Edge16SIMD(pix []byte, q0Base int, step int, outer int, length int, scale int, params filter4Params) {
	groups := length / 8
	if outer != 2 || groups == 0 {
		if step == 2 && groups > 0 {
			filter6Vert16SIMD(pix, q0Base, step, outer, length, scale, params)
			return
		}
		filter6Edge16PureGo(pix, q0Base, step, outer, length, scale, params)
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
		base := unsafe.Add(q0p, g*16) // 8 positions * 2 bytes
		pP1 := unsafe.Add(base, -2*step)
		pP0 := unsafe.Add(base, -step)
		pQ1 := unsafe.Add(base, step)
		p2 := lf16LoadP(unsafe.Add(base, -3*step))
		p1 := lf16LoadP(pP1)
		p0 := lf16LoadP(pP0)
		q0 := lf16LoadP(base)
		q1 := lf16LoadP(pQ1)
		q2 := lf16LoadP(unsafe.Add(base, 2*step))

		need := lfAbsDiffInt16x8(p2, p1).LessEqual(limit).
			And(lfAbsDiffInt16x8(p1, p0).LessEqual(limit)).
			And(lfAbsDiffInt16x8(q1, q0).LessEqual(limit)).
			And(lfAbsDiffInt16x8(q2, q1).LessEqual(limit)).
			And(lfAbsDiffInt16x8(p0, q0).ShiftAllLeft(1).
				Add(lfAbsDiffInt16x8(p1, q1).ShiftAllRight(1)).
				LessEqual(blimit))
		flat := lfAbsDiffInt16x8(p1, p0).LessEqual(flatThr).
			And(lfAbsDiffInt16x8(q1, q0).LessEqual(flatThr)).
			And(lfAbsDiffInt16x8(p2, p0).LessEqual(flatThr)).
			And(lfAbsDiffInt16x8(q2, q0).LessEqual(flatThr))
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

		if need.And(flat).ToInt16x8().ToBits().ReduceSum() == 0 {
			lf16StoreP(pP1, np1.IfElse(need, p1))
			lf16StoreP(pP0, np0.IfElse(need, p0))
			lf16StoreP(base, nq0.IfElse(need, q0))
			lf16StoreP(pQ1, nq1.IfElse(need, q1))
			continue
		}

		acc := p2.ShiftAllLeft(1).Add(p2).
			Add(p1.Add(p0).ShiftAllLeft(1)).
			Add(q0).Add(four)
		lf16StoreP(pP1, acc.ShiftAllRight(3).IfElse(flat, np1).IfElse(need, p1))
		acc = acc.Add(q0.Add(q1)).Sub(p2.ShiftAllLeft(1))
		lf16StoreP(pP0, acc.ShiftAllRight(3).IfElse(flat, np0).IfElse(need, p0))
		acc = acc.Add(q1.Add(q2)).Sub(p2.Add(p1))
		lf16StoreP(base, acc.ShiftAllRight(3).IfElse(flat, nq0).IfElse(need, q0))
		acc = acc.Add(q2.ShiftAllLeft(1)).Sub(p1.Add(p0))
		lf16StoreP(pQ1, acc.ShiftAllRight(3).IfElse(flat, nq1).IfElse(need, q1))
	}
	if rem := length - groups*8; rem > 0 {
		filter6Edge16PureGo(pix, q0Base+groups*8*outer, step, outer, rem, scale, params)
	}
}

// filter6Vert16SIMD mirrors filter6Vert16NEON with the Go-native SIMD
// horizontal core (which also covers 12-bit) between the NEON gather/scatter.
func filter6Vert16SIMD(pix []byte, q0Base int, step int, outer int, length int, scale int, params filter4Params) {
	groups := length / 8
	if step != 2 || groups == 0 {
		filter6Edge16PureGo(pix, q0Base, step, outer, length, scale, params)
		return
	}
	var scratch [6 * wide16VertScratchStride]byte
	for g := 0; g < groups; g += filter14VertBatchGroups {
		n := groups - g
		if n > filter14VertBatchGroups {
			n = filter14VertBatchGroups
		}
		colBase := q0Base + g*8*outer
		ctx := wideVertTransposeCtx{
			src:     &pix[colBase-3*step], // p2 of the first position
			stride:  uintptr(outer),
			scratch: &scratch[0],
			count:   uintptr(n),
		}
		filter6Vert16GatherNEONAsm(&ctx)
		filter6Edge16SIMD(scratch[:], 3*wide16VertScratchStride, wide16VertScratchStride, 2, n*8, scale, params)
		filter6Vert16ScatterNEONAsm(&ctx)
	}
	if rem := length - groups*8; rem > 0 {
		filter6Edge16PureGo(pix, q0Base+groups*8*outer, step, outer, rem, scale, params)
	}
}
