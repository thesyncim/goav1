// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && (arm64 || amd64) && !purego

// Go-native SIMD narrow (four-sample) deblocking kernel, shared by the 8-bit and
// 10/12-bit paths.

package loopfilter

import (
	"simd/archsimd"
)

// filter4EdgeSIMD is the Go-native SIMD form of filter4EdgePureGo for 8-bit
// edges. It is bit-exact with the reference over the full input domain: every
// lane runs in signed 16-bit arithmetic on centred samples (-128..127) and the
// clamps reproduce signedClamp exactly.
func filter4EdgeSIMD(pix []byte, q0Base int, step int, outer int, length int, params filter4Params) {
	if outer == 1 && length >= 16 && params.center == 128 && params.min == -128 && params.max == 127 && params.blimit >= 0 && params.blimit < 255 && params.limit >= 0 && params.limit <= 255 && params.hev >= 0 && params.hev <= 255 {
		done := length &^ 15
		lfFilter4U8Wide(pix, q0Base, step, done, params, nil)
		if done < length {
			lfEdge[uint8](lfKind4, pix, q0Base+done, step, outer, length-done, 1, params)
		}
		return
	}
	lfEdge[uint8](lfKind4, pix, q0Base, step, outer, length, 1, params)
}

// filter4Edge16SIMD is the 10/12-bit form of filter4EdgeSIMD over two-byte
// samples. The centred values reach +-4095, and the widest intermediate
// (3*(qs0-ps0)+filter) stays inside int16.
func filter4Edge16SIMD(pix []byte, q0Base int, step int, outer int, length int, params filter4Params) {
	lfEdge[uint16](lfKind4, pix, q0Base, step, outer, length, 1, params)
}

// lfFilter4Core runs the narrow filter over length positions (a multiple of
// eight) of horizontal taps. The positions are contiguous samples of type S.
func lfFilter4Core[S lfSample](pix []byte, q0Base int, step int, length int, _ int, params filter4Params) uint {
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
	shift1 := archsimd.BroadcastInt16x8(-1)
	shift3 := archsimd.BroadcastInt16x8(-3)
	var changed uint
	for g := 0; g < length/8; g++ {
		base := q0Base + g*8*sz
		p1 := lfLoad[S](pix, base-2*step)
		p0 := lfLoad[S](pix, base-step)
		q0 := lfLoad[S](pix, base)
		q1 := lfLoad[S](pix, base+step)

		d0q0 := lfAbsDiff[S](p0, q0)
		need := lfAbsDiff[S](p1, p0).LessEqual(limit).
			And(lfAbsDiff[S](q1, q0).LessEqual(limit)).
			And(d0q0.Add(d0q0).
				Add(lfShift1(lfAbsDiff[S](p1, q1), shift1)).
				LessEqual(blimit))
		if !lfAny(need) {
			continue
		}
		changed |= 1 << uint(g)
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

		lfStore[S](pix, base-2*step, np1.IfElse(need, p1))
		lfStore[S](pix, base-step, np0.IfElse(need, p0))
		lfStore[S](pix, base, nq0.IfElse(need, q0))
		lfStore[S](pix, base+step, nq1.IfElse(need, q1))
	}
	return changed
}
