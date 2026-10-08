// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && (arm64 || amd64) && !purego

// Go-native SIMD eight-tap deblocking kernel, shared by the 8-bit and 10/12-bit
// paths.

package loopfilter

import "simd/archsimd"

// filter8EdgeSIMD is the Go-native SIMD form of filter8EdgePureGo for 8-bit
// edges.
func filter8EdgeSIMD(pix []byte, q0Base int, step int, outer int, length int, scale int, params filter4Params) {
	lfEdge[uint8](lfKind8, pix, q0Base, step, outer, length, scale, params)
}

// filter8Edge16SIMD is the 10/12-bit form of filter8EdgeSIMD over two-byte
// samples. The six flat sums peak at 4095*8 + 4 = 32764 and stay in int16 with
// no offset.
func filter8Edge16SIMD(pix []byte, q0Base int, step int, outer int, length int, scale int, params filter4Params) {
	lfEdge[uint16](lfKind8, pix, q0Base, step, outer, length, scale, params)
}

// lfFilter8Core runs the eight-tap filter over length positions (a multiple of
// eight) of horizontal taps. The flat outputs follow the running window sum of
// filter8Samples: each output adds its incoming tap pair and drops the outgoing
// pair, so the six results cost five updates and one shift each.
func lfFilter8Core[S lfSample](pix []byte, q0Base int, step int, length int, scale int, params filter4Params) uint {
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
		p3 := lfLoad[S](pix, base-4*step)
		p2 := lfLoad[S](pix, base-3*step)
		p1 := lfLoad[S](pix, base-2*step)
		p0 := lfLoad[S](pix, base-step)
		q0 := lfLoad[S](pix, base)
		q1 := lfLoad[S](pix, base+step)
		q2 := lfLoad[S](pix, base+2*step)
		q3 := lfLoad[S](pix, base+3*step)

		d0q0 := lfAbsDiff[S](p0, q0)
		need := lfAbsDiff[S](p3, p2).LessEqual(limit).
			And(lfAbsDiff[S](p2, p1).LessEqual(limit)).
			And(lfAbsDiff[S](p1, p0).LessEqual(limit)).
			And(lfAbsDiff[S](q1, q0).LessEqual(limit)).
			And(lfAbsDiff[S](q2, q1).LessEqual(limit)).
			And(lfAbsDiff[S](q3, q2).LessEqual(limit)).
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
			And(lfAbsDiff[S](q2, q0).LessEqual(flatThr)).
			And(lfAbsDiff[S](p3, p0).LessEqual(flatThr)).
			And(lfAbsDiff[S](q3, q0).LessEqual(flatThr))
		if lfAll(need.And(flat)) {
			acc := p3.Add(p3).Add(p3).Add(p2.Add(p2)).Add(p1).Add(p0).Add(q0).Add(four)
			lfStore[S](pix, base-3*step, lfShift3(acc, shift3))
			acc = acc.Add(p1.Add(q1)).Sub(p3.Add(p2))
			lfStore[S](pix, base-2*step, lfShift3(acc, shift3))
			acc = acc.Add(p0.Add(q2)).Sub(p3.Add(p1))
			lfStore[S](pix, base-step, lfShift3(acc, shift3))
			acc = acc.Add(q0.Add(q3)).Sub(p3.Add(p0))
			lfStore[S](pix, base, lfShift3(acc, shift3))
			acc = acc.Add(q1.Add(q3)).Sub(p2.Add(q0))
			lfStore[S](pix, base+step, lfShift3(acc, shift3))
			acc = acc.Add(q2.Add(q3)).Sub(p1.Add(q1))
			lfStore[S](pix, base+2*step, lfShift3(acc, shift3))
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

		gate := need.And(flat)
		if !lfAny(gate) {
			lfStore[S](pix, base-2*step, np1.IfElse(need, p1))
			lfStore[S](pix, base-step, np0.IfElse(need, p0))
			lfStore[S](pix, base, nq0.IfElse(need, q0))
			lfStore[S](pix, base+step, nq1.IfElse(need, q1))
			continue
		}

		// Flat window sums: accB holds p3*3 + p2*2 + p1 + p0 + q0 + 4 and is
		// advanced one tap pair at a time through the six outputs.
		accB := p3.Add(p3).Add(p3).Add(p2.Add(p2)).Add(p1).Add(p0).Add(q0).Add(four)
		f8p2 := lfShift3(accB, shift3)
		lfStore[S](pix, base-3*step, f8p2.IfElse(gate, p2))
		accB = accB.Add(p1.Add(q1)).Sub(p3.Add(p2))
		f8p1 := lfShift3(accB, shift3)
		lfStore[S](pix, base-2*step, f8p1.IfElse(gate, np1.IfElse(need, p1)))
		accB = accB.Add(p0.Add(q2)).Sub(p3.Add(p1))
		f8p0 := lfShift3(accB, shift3)
		lfStore[S](pix, base-step, f8p0.IfElse(gate, np0.IfElse(need, p0)))
		accB = accB.Add(q0.Add(q3)).Sub(p3.Add(p0))
		f8q0 := lfShift3(accB, shift3)
		lfStore[S](pix, base, f8q0.IfElse(gate, nq0.IfElse(need, q0)))
		accB = accB.Add(q1.Add(q3)).Sub(p2.Add(q0))
		f8q1 := lfShift3(accB, shift3)
		lfStore[S](pix, base+step, f8q1.IfElse(gate, nq1.IfElse(need, q1)))
		accB = accB.Add(q2.Add(q3)).Sub(p1.Add(q1))
		f8q2 := lfShift3(accB, shift3)
		lfStore[S](pix, base+2*step, f8q2.IfElse(gate, q2))
	}
	return changed
}
