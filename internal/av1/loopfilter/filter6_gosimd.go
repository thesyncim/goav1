// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && (arm64 || amd64) && !purego

// Go-native SIMD six-tap deblocking kernel, shared by the 8-bit and 10/12-bit
// paths.

package loopfilter

import "simd/archsimd"

// filter6EdgeSIMD is the Go-native SIMD form of filter6EdgePureGo for 8-bit
// edges.
func filter6EdgeSIMD(pix []byte, q0Base int, step int, outer int, length int, scale int, params filter4Params) {
	lfEdge[uint8](lfKind6, pix, q0Base, step, outer, length, scale, params)
}

// filter6Edge16SIMD is the 10/12-bit form of filter6EdgeSIMD over two-byte
// samples. The horizontal sums stay inside int16 at both depths (max
// 4095*8 + 4 = 32764), so no accumulator offset is needed.
func filter6Edge16SIMD(pix []byte, q0Base int, step int, outer int, length int, scale int, params filter4Params) {
	lfEdge[uint16](lfKind6, pix, q0Base, step, outer, length, scale, params)
}

// lfFilter6Core runs the six-tap filter over length positions (a multiple of
// eight) of horizontal taps. The positions are contiguous samples of type S.
func lfFilter6Core[S lfSample](pix []byte, q0Base int, step int, length int, scale int, params filter4Params) uint {
	sz := lfSize[S]()
	limit := archsimd.BroadcastInt16x8(params.limit)
	blimit := archsimd.BroadcastInt16x8(params.blimit)
	hevT := archsimd.BroadcastInt16x8(params.hev)
	center := archsimd.BroadcastInt16x8(params.center)
	minV := archsimd.BroadcastInt16x8(params.min)
	maxV := archsimd.BroadcastInt16x8(params.max)
	one := archsimd.BroadcastInt16x8(1)
	three := archsimd.BroadcastInt16x8(3)
	four := archsimd.BroadcastInt16x8(4)
	flatThr := archsimd.BroadcastInt16x8(int16(scale))
	shift1 := archsimd.BroadcastInt16x8(-1)
	shift3 := archsimd.BroadcastInt16x8(-3)
	var changed uint
	for g := 0; g < length/8; g++ {
		base := q0Base + g*8*sz
		p2 := lfLoad[S](pix, base-3*step)
		p1 := lfLoad[S](pix, base-2*step)
		p0 := lfLoad[S](pix, base-step)
		q0 := lfLoad[S](pix, base)
		q1 := lfLoad[S](pix, base+step)
		q2 := lfLoad[S](pix, base+2*step)

		d0q0 := lfAbsDiff[S](p0, q0)
		need := lfAbsDiff[S](p2, p1).LessEqual(limit).
			And(lfAbsDiff[S](p1, p0).LessEqual(limit)).
			And(lfAbsDiff[S](q1, q0).LessEqual(limit)).
			And(lfAbsDiff[S](q2, q1).LessEqual(limit)).
			And(d0q0.Add(d0q0).
				Add(lfShift1(lfAbsDiff[S](p1, q1), shift1)).
				LessEqual(blimit))
		if !lfAny(need) {
			continue
		}
		changed |= 1 << uint(g)
		flat := lfAbsDiff[S](p1, p0).LessEqual(flatThr).
			And(lfAbsDiff[S](q1, q0).LessEqual(flatThr)).
			And(lfAbsDiff[S](p2, p0).LessEqual(flatThr)).
			And(lfAbsDiff[S](q2, q0).LessEqual(flatThr))
		if lfAll(need.And(flat)) {
			acc := p2.Add(p2).Add(p2).
				Add(p1.Add(p0).Add(p1.Add(p0))).
				Add(q0).Add(four)
			lfStore[S](pix, base-2*step, lfShift3(acc, shift3))
			acc = acc.Add(q0.Add(q1)).Sub(p2.Add(p2))
			lfStore[S](pix, base-step, lfShift3(acc, shift3))
			acc = acc.Add(q1.Add(q2)).Sub(p2.Add(p1))
			lfStore[S](pix, base, lfShift3(acc, shift3))
			acc = acc.Add(q2.Add(q2)).Sub(p1.Add(p0))
			lfStore[S](pix, base+step, lfShift3(acc, shift3))
			continue
		}
		hev := lfAbsDiff[S](p1, p0).Greater(hevT).Or(lfAbsDiff[S](q1, q0).Greater(hevT))
		ps1 := p1.Sub(center)
		ps0 := p0.Sub(center)
		qs0 := q0.Sub(center)
		qs1 := q1.Sub(center)
		f := ps1.Sub(qs1).Max(minV).Min(maxV).Masked(hev)
		f = f.Add(qs0.Sub(ps0).Mul(three)).Max(minV).Min(maxV)
		filter1 := lfShift3(f.Add(four).Max(minV).Min(maxV), shift3)
		filter2 := lfShift3(f.Add(three).Max(minV).Min(maxV), shift3)
		np0 := ps0.Add(filter2).Max(minV).Min(maxV).Add(center)
		nq0 := qs0.Sub(filter1).Max(minV).Min(maxV).Add(center)
		ov := lfShift1(filter1.Add(one), shift1)
		np1 := p1.IfElse(hev, ps1.Add(ov).Max(minV).Min(maxV).Add(center))
		nq1 := q1.IfElse(hev, qs1.Sub(ov).Max(minV).Min(maxV).Add(center))

		if !lfAny(need.And(flat)) {
			lfStore[S](pix, base-2*step, np1.IfElse(need, p1))
			lfStore[S](pix, base-step, np0.IfElse(need, p0))
			lfStore[S](pix, base, nq0.IfElse(need, q0))
			lfStore[S](pix, base+step, nq1.IfElse(need, q1))
			continue
		}

		acc := p2.Add(p2).Add(p2).
			Add(p1.Add(p0).Add(p1.Add(p0))).
			Add(q0).Add(four)
		lfStore[S](pix, base-2*step, lfShift3(acc, shift3).IfElse(flat, np1).IfElse(need, p1))
		acc = acc.Add(q0.Add(q1)).Sub(p2.Add(p2))
		lfStore[S](pix, base-step, lfShift3(acc, shift3).IfElse(flat, np0).IfElse(need, p0))
		acc = acc.Add(q1.Add(q2)).Sub(p2.Add(p1))
		lfStore[S](pix, base, lfShift3(acc, shift3).IfElse(flat, nq0).IfElse(need, q0))
		acc = acc.Add(q2.Add(q2)).Sub(p1.Add(p0))
		lfStore[S](pix, base+step, lfShift3(acc, shift3).IfElse(flat, nq1).IfElse(need, q1))
	}
	return changed
}
