// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && (arm64 || amd64) && !purego

package loopfilter

import (
	"simd/archsimd"
)

// filter14EdgeSIMD is the Go-native SIMD form of filter14EdgePureGo for 8-bit
// edges.
func filter14EdgeSIMD(pix []byte, q0Base int, step int, outer int, length int, scale int, params filter4Params) {
	lfEdge[uint8](lfKind14, pix, q0Base, step, outer, length, scale, params)
}

// filter14Edge16SIMD is the 10/12-bit form of filter14EdgeSIMD over two-byte
// samples. The fourteen-tap window sum reaches 4095*16 + 8 = 65528 at 12-bit,
// which overflows int16. The wide accumulator therefore carries a -32768 offset
// (T = sum + 8 - 32768 is always in [-32760, 32760], exactly representable), so
// every output is one arithmetic ShiftAllRight(4) plus an Add(2048). Intermediate
// wrap-around during the add/subtract stepping is exact mod 2^16, and the true
// value fits int16 at every shift point. The same offset form is exact at 8-bit,
// where the window sum is at most 4080 + 8.
func filter14Edge16SIMD(pix []byte, q0Base int, step int, outer int, length int, scale int, params filter4Params) {
	lfEdge[uint16](lfKind14, pix, q0Base, step, outer, length, scale, params)
}

// lfFilter14Core runs the fourteen-tap filter over length positions (a multiple
// of eight) of horizontal taps. The positions are contiguous samples of type S.
func lfFilter14Core[S lfSample](pix []byte, q0Base int, step int, length int, scale int, params filter4Params) uint {
	sz := lfSize[S]()
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
	shift1 := archsimd.BroadcastInt16x8(-1)
	shift3 := archsimd.BroadcastInt16x8(-3)
	shift4 := archsimd.BroadcastInt16x8(-4)
	var changed uint
	// Each tap row is one window of pix taken per call, all len(r0) long,
	// so the groups index them with no bounds checks.
	r0 := pix[q0Base:][: length*sz : length*sz]
	rm7 := pix[q0Base-7*step:][:len(r0):len(r0)]
	rm6 := pix[q0Base-6*step:][:len(r0):len(r0)]
	rm5 := pix[q0Base-5*step:][:len(r0):len(r0)]
	rm4 := pix[q0Base-4*step:][:len(r0):len(r0)]
	rm3 := pix[q0Base-3*step:][:len(r0):len(r0)]
	rm2 := pix[q0Base-2*step:][:len(r0):len(r0)]
	rm1 := pix[q0Base-step:][:len(r0):len(r0)]
	rp1 := pix[q0Base+step:][:len(r0):len(r0)]
	rp2 := pix[q0Base+2*step:][:len(r0):len(r0)]
	rp3 := pix[q0Base+3*step:][:len(r0):len(r0)]
	rp4 := pix[q0Base+4*step:][:len(r0):len(r0)]
	rp5 := pix[q0Base+5*step:][:len(r0):len(r0)]
	rp6 := pix[q0Base+6*step:][:len(r0):len(r0)]
	for g, o := 0, 0; o <= len(r0)-8*sz; g, o = g+1, o+8*sz {
		p3 := lfLoad[S](rm4, o)
		p2 := lfLoad[S](rm3, o)
		p1 := lfLoad[S](rm2, o)
		p0 := lfLoad[S](rm1, o)
		q0 := lfLoad[S](r0, o)
		q1 := lfLoad[S](rp1, o)
		q2 := lfLoad[S](rp2, o)
		q3 := lfLoad[S](rp3, o)

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
			p6w := lfLoad[S](rm7, o)
			p5w := lfLoad[S](rm6, o)
			p4w := lfLoad[S](rm5, o)
			q4w := lfLoad[S](rp4, o)
			q5w := lfLoad[S](rp5, o)
			q6w := lfLoad[S](rp6, o)
			flat2w := lfAbsDiff[S](p4w, p0).LessEqual(flatThr).
				And(lfAbsDiff[S](q4w, q0).LessEqual(flatThr)).
				And(lfAbsDiff[S](p5w, p0).LessEqual(flatThr)).
				And(lfAbsDiff[S](q5w, q0).LessEqual(flatThr)).
				And(lfAbsDiff[S](p6w, p0).LessEqual(flatThr)).
				And(lfAbsDiff[S](q6w, q0).LessEqual(flatThr))
			if lfAll(flat2w) {
				p6x2 := p6w.Add(p6w)
				p6x7 := p6x2.Add(p6x2).Add(p6x2).Add(p6w)
				acc := p6x7.Add(p5w.Add(p4w).Add(p5w.Add(p4w))).
					Add(p3.Add(p2)).Add(p1.Add(p0)).Add(q0).Add(biasW)
				lfStore[S](rm6, o, lfShift4(acc, shift4).Add(half))
				acc = acc.Add(p3.Add(q1)).Sub(p6x2)
				lfStore[S](rm5, o, lfShift4(acc, shift4).Add(half))
				acc = acc.Add(p2.Add(q2)).Sub(p6w.Add(p5w))
				lfStore[S](rm4, o, lfShift4(acc, shift4).Add(half))
				acc = acc.Add(p1.Add(q3)).Sub(p6w.Add(p4w))
				lfStore[S](rm3, o, lfShift4(acc, shift4).Add(half))
				acc = acc.Add(p0.Add(q4w)).Sub(p6w.Add(p3))
				lfStore[S](rm2, o, lfShift4(acc, shift4).Add(half))
				acc = acc.Add(q0.Add(q5w)).Sub(p6w.Add(p2))
				lfStore[S](rm1, o, lfShift4(acc, shift4).Add(half))
				acc = acc.Add(q1.Add(q6w)).Sub(p6w.Add(p1))
				lfStore[S](r0, o, lfShift4(acc, shift4).Add(half))
				acc = acc.Add(q2.Add(q6w)).Sub(p5w.Add(p0))
				lfStore[S](rp1, o, lfShift4(acc, shift4).Add(half))
				acc = acc.Add(q3.Add(q6w)).Sub(p4w.Add(q0))
				lfStore[S](rp2, o, lfShift4(acc, shift4).Add(half))
				acc = acc.Add(q4w.Add(q6w)).Sub(p3.Add(q1))
				lfStore[S](rp3, o, lfShift4(acc, shift4).Add(half))
				acc = acc.Add(q5w.Add(q6w)).Sub(p2.Add(q2))
				lfStore[S](rp4, o, lfShift4(acc, shift4).Add(half))
				acc = acc.Add(q6w.Add(q6w)).Sub(p1.Add(q3))
				lfStore[S](rp5, o, lfShift4(acc, shift4).Add(half))
				continue
			}
		}
		hev := lfAbsDiff[S](p1, p0).Greater(hevT).Or(lfAbsDiff[S](q1, q0).Greater(hevT))
		ps1 := p1.Sub(center)
		ps0 := p0.Sub(center)
		qs0 := q0.Sub(center)
		qs1 := q1.Sub(center)
		f := ps1.Sub(qs1).Max(minV).Min(maxV).Masked(hev)
		f = f.Add(qs0.Sub(ps0).Mul(three16)).Max(minV).Min(maxV)
		filter1 := lfShift3(f.Add(four).Max(minV).Min(maxV), shift3)
		filter2 := lfShift3(f.Add(three16).Max(minV).Min(maxV), shift3)
		np0 := ps0.Add(filter2).Max(minV).Min(maxV).Add(center)
		nq0 := qs0.Sub(filter1).Max(minV).Min(maxV).Add(center)
		ov := lfShift1(filter1.Add(one), shift1)
		np1 := p1.IfElse(hev, ps1.Add(ov).Max(minV).Min(maxV).Add(center))
		nq1 := q1.IfElse(hev, qs1.Sub(ov).Max(minV).Min(maxV).Add(center))

		gateNF := need.And(flat)
		if !lfAny(gateNF) {
			lfStore[S](rm2, o, np1.IfElse(need, p1))
			lfStore[S](rm1, o, np0.IfElse(need, p0))
			lfStore[S](r0, o, nq0.IfElse(need, q0))
			lfStore[S](rp1, o, nq1.IfElse(need, q1))
			continue
		}

		p6 := lfLoad[S](rm7, o)
		p5 := lfLoad[S](rm6, o)
		p4 := lfLoad[S](rm5, o)
		q4 := lfLoad[S](rp4, o)
		q5 := lfLoad[S](rp5, o)
		q6 := lfLoad[S](rp6, o)
		flat2 := lfAbsDiff[S](p4, p0).LessEqual(flatThr).
			And(lfAbsDiff[S](q4, q0).LessEqual(flatThr)).
			And(lfAbsDiff[S](p5, p0).LessEqual(flatThr)).
			And(lfAbsDiff[S](q5, q0).LessEqual(flatThr)).
			And(lfAbsDiff[S](p6, p0).LessEqual(flatThr)).
			And(lfAbsDiff[S](q6, q0).LessEqual(flatThr))
		wideM := gateNF.And(flat2)

		accB := p3.Add(p3).Add(p3).
			Add(p2.Add(p2)).
			Add(p1).Add(p0).Add(q0).Add(four)

		if !lfAny(wideM) {
			lfStore[S](rm3, o, lfShift3(accB, shift3).IfElse(flat, p2).IfElse(need, p2))
			accB = accB.Add(p1.Add(q1)).Sub(p3.Add(p2))
			lfStore[S](rm2, o, lfShift3(accB, shift3).IfElse(flat, np1).IfElse(need, p1))
			accB = accB.Add(p0.Add(q2)).Sub(p3.Add(p1))
			lfStore[S](rm1, o, lfShift3(accB, shift3).IfElse(flat, np0).IfElse(need, p0))
			accB = accB.Add(q0.Add(q3)).Sub(p3.Add(p0))
			lfStore[S](r0, o, lfShift3(accB, shift3).IfElse(flat, nq0).IfElse(need, q0))
			accB = accB.Add(q1.Add(q3)).Sub(p2.Add(q0))
			lfStore[S](rp1, o, lfShift3(accB, shift3).IfElse(flat, nq1).IfElse(need, q1))
			accB = accB.Add(q2.Add(q3)).Sub(p1.Add(q1))
			lfStore[S](rp2, o, lfShift3(accB, shift3).IfElse(flat, q2).IfElse(need, q2))
			continue
		}

		p6x2 := p6.Add(p6)
		p6x7 := p6x2.Add(p6x2).Add(p6x2).Add(p6)
		accW := p6x7.Add(p5.Add(p4).Add(p5.Add(p4))).
			Add(p3.Add(p2)).
			Add(p1.Add(p0)).
			Add(q0).Add(biasW)
		lfStore[S](rm6, o, lfShift4(accW, shift4).Add(half).IfElse(wideM, p5))
		accW = accW.Add(p3.Add(q1)).Sub(p6.Add(p6))
		lfStore[S](rm5, o, lfShift4(accW, shift4).Add(half).IfElse(wideM, p4))
		accW = accW.Add(p2.Add(q2)).Sub(p6.Add(p5))
		lfStore[S](rm4, o, lfShift4(accW, shift4).Add(half).IfElse(wideM, p3))
		accW = accW.Add(p1.Add(q3)).Sub(p6.Add(p4))
		f8p2 := lfShift3(accB, shift3)
		lfStore[S](rm3, o, lfShift4(accW, shift4).Add(half).IfElse(flat2, f8p2).IfElse(flat, p2).IfElse(need, p2))
		accW = accW.Add(p0.Add(q4)).Sub(p6.Add(p3))
		accB = accB.Add(p1.Add(q1)).Sub(p3.Add(p2))
		f8p1 := lfShift3(accB, shift3)
		lfStore[S](rm2, o, lfShift4(accW, shift4).Add(half).IfElse(flat2, f8p1).IfElse(flat, np1).IfElse(need, p1))
		accW = accW.Add(q0.Add(q5)).Sub(p6.Add(p2))
		accB = accB.Add(p0.Add(q2)).Sub(p3.Add(p1))
		f8p0 := lfShift3(accB, shift3)
		lfStore[S](rm1, o, lfShift4(accW, shift4).Add(half).IfElse(flat2, f8p0).IfElse(flat, np0).IfElse(need, p0))
		accW = accW.Add(q1.Add(q6)).Sub(p6.Add(p1))
		accB = accB.Add(q0.Add(q3)).Sub(p3.Add(p0))
		f8q0 := lfShift3(accB, shift3)
		lfStore[S](r0, o, lfShift4(accW, shift4).Add(half).IfElse(flat2, f8q0).IfElse(flat, nq0).IfElse(need, q0))
		accW = accW.Add(q2.Add(q6)).Sub(p5.Add(p0))
		accB = accB.Add(q1.Add(q3)).Sub(p2.Add(q0))
		f8q1 := lfShift3(accB, shift3)
		lfStore[S](rp1, o, lfShift4(accW, shift4).Add(half).IfElse(flat2, f8q1).IfElse(flat, nq1).IfElse(need, q1))
		accW = accW.Add(q3.Add(q6)).Sub(p4.Add(q0))
		accB = accB.Add(q2.Add(q3)).Sub(p1.Add(q1))
		f8q2 := lfShift3(accB, shift3)
		lfStore[S](rp2, o, lfShift4(accW, shift4).Add(half).IfElse(flat2, f8q2).IfElse(flat, q2).IfElse(need, q2))
		accW = accW.Add(q4.Add(q6)).Sub(p3.Add(q1))
		lfStore[S](rp3, o, lfShift4(accW, shift4).Add(half).IfElse(wideM, q3))
		accW = accW.Add(q5.Add(q6)).Sub(p2.Add(q2))
		lfStore[S](rp4, o, lfShift4(accW, shift4).Add(half).IfElse(wideM, q4))
		accW = accW.Add(q6.Add(q6)).Sub(p1.Add(q3))
		lfStore[S](rp5, o, lfShift4(accW, shift4).Add(half).IfElse(wideM, q5))
	}
	return changed
}

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
	// Each tap row is one window of pix taken per call, all len(r0) long,
	// so the groups index them with no bounds checks.
	r0 := pix[q0Base:][: length*sz : length*sz]
	rm2 := pix[q0Base-2*step:][:len(r0):len(r0)]
	rm1 := pix[q0Base-step:][:len(r0):len(r0)]
	rp1 := pix[q0Base+step:][:len(r0):len(r0)]
	for g, o := 0, 0; o <= len(r0)-8*sz; g, o = g+1, o+8*sz {
		p1 := lfLoad[S](rm2, o)
		p0 := lfLoad[S](rm1, o)
		q0 := lfLoad[S](r0, o)
		q1 := lfLoad[S](rp1, o)

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

		lfStore[S](rm2, o, np1.IfElse(need, p1))
		lfStore[S](rm1, o, np0.IfElse(need, p0))
		lfStore[S](r0, o, nq0.IfElse(need, q0))
		lfStore[S](rp1, o, nq1.IfElse(need, q1))
	}
	return changed
}

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
	// Each tap row is one window of pix taken per call, all len(r0) long,
	// so the groups index them with no bounds checks.
	r0 := pix[q0Base:][: length*sz : length*sz]
	rm3 := pix[q0Base-3*step:][:len(r0):len(r0)]
	rm2 := pix[q0Base-2*step:][:len(r0):len(r0)]
	rm1 := pix[q0Base-step:][:len(r0):len(r0)]
	rp1 := pix[q0Base+step:][:len(r0):len(r0)]
	rp2 := pix[q0Base+2*step:][:len(r0):len(r0)]
	for g, o := 0, 0; o <= len(r0)-8*sz; g, o = g+1, o+8*sz {
		p2 := lfLoad[S](rm3, o)
		p1 := lfLoad[S](rm2, o)
		p0 := lfLoad[S](rm1, o)
		q0 := lfLoad[S](r0, o)
		q1 := lfLoad[S](rp1, o)
		q2 := lfLoad[S](rp2, o)

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
			lfStore[S](rm2, o, lfShift3(acc, shift3))
			acc = acc.Add(q0.Add(q1)).Sub(p2.Add(p2))
			lfStore[S](rm1, o, lfShift3(acc, shift3))
			acc = acc.Add(q1.Add(q2)).Sub(p2.Add(p1))
			lfStore[S](r0, o, lfShift3(acc, shift3))
			acc = acc.Add(q2.Add(q2)).Sub(p1.Add(p0))
			lfStore[S](rp1, o, lfShift3(acc, shift3))
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
			lfStore[S](rm2, o, np1.IfElse(need, p1))
			lfStore[S](rm1, o, np0.IfElse(need, p0))
			lfStore[S](r0, o, nq0.IfElse(need, q0))
			lfStore[S](rp1, o, nq1.IfElse(need, q1))
			continue
		}

		acc := p2.Add(p2).Add(p2).
			Add(p1.Add(p0).Add(p1.Add(p0))).
			Add(q0).Add(four)
		lfStore[S](rm2, o, lfShift3(acc, shift3).IfElse(flat, np1).IfElse(need, p1))
		acc = acc.Add(q0.Add(q1)).Sub(p2.Add(p2))
		lfStore[S](rm1, o, lfShift3(acc, shift3).IfElse(flat, np0).IfElse(need, p0))
		acc = acc.Add(q1.Add(q2)).Sub(p2.Add(p1))
		lfStore[S](r0, o, lfShift3(acc, shift3).IfElse(flat, nq0).IfElse(need, q0))
		acc = acc.Add(q2.Add(q2)).Sub(p1.Add(p0))
		lfStore[S](rp1, o, lfShift3(acc, shift3).IfElse(flat, nq1).IfElse(need, q1))
	}
	return changed
}

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
	// Each tap row is one window of pix taken per call, all len(r0) long,
	// so the groups index them with no bounds checks.
	r0 := pix[q0Base:][: length*sz : length*sz]
	rm4 := pix[q0Base-4*step:][:len(r0):len(r0)]
	rm3 := pix[q0Base-3*step:][:len(r0):len(r0)]
	rm2 := pix[q0Base-2*step:][:len(r0):len(r0)]
	rm1 := pix[q0Base-step:][:len(r0):len(r0)]
	rp1 := pix[q0Base+step:][:len(r0):len(r0)]
	rp2 := pix[q0Base+2*step:][:len(r0):len(r0)]
	rp3 := pix[q0Base+3*step:][:len(r0):len(r0)]
	for g, o := 0, 0; o <= len(r0)-8*sz; g, o = g+1, o+8*sz {
		p3 := lfLoad[S](rm4, o)
		p2 := lfLoad[S](rm3, o)
		p1 := lfLoad[S](rm2, o)
		p0 := lfLoad[S](rm1, o)
		q0 := lfLoad[S](r0, o)
		q1 := lfLoad[S](rp1, o)
		q2 := lfLoad[S](rp2, o)
		q3 := lfLoad[S](rp3, o)

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
			lfStore[S](rm3, o, lfShift3(acc, shift3))
			acc = acc.Add(p1.Add(q1)).Sub(p3.Add(p2))
			lfStore[S](rm2, o, lfShift3(acc, shift3))
			acc = acc.Add(p0.Add(q2)).Sub(p3.Add(p1))
			lfStore[S](rm1, o, lfShift3(acc, shift3))
			acc = acc.Add(q0.Add(q3)).Sub(p3.Add(p0))
			lfStore[S](r0, o, lfShift3(acc, shift3))
			acc = acc.Add(q1.Add(q3)).Sub(p2.Add(q0))
			lfStore[S](rp1, o, lfShift3(acc, shift3))
			acc = acc.Add(q2.Add(q3)).Sub(p1.Add(q1))
			lfStore[S](rp2, o, lfShift3(acc, shift3))
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
			lfStore[S](rm2, o, np1.IfElse(need, p1))
			lfStore[S](rm1, o, np0.IfElse(need, p0))
			lfStore[S](r0, o, nq0.IfElse(need, q0))
			lfStore[S](rp1, o, nq1.IfElse(need, q1))
			continue
		}

		// Flat window sums: accB holds p3*3 + p2*2 + p1 + p0 + q0 + 4 and is
		// advanced one tap pair at a time through the six outputs.
		accB := p3.Add(p3).Add(p3).Add(p2.Add(p2)).Add(p1).Add(p0).Add(q0).Add(four)
		f8p2 := lfShift3(accB, shift3)
		lfStore[S](rm3, o, f8p2.IfElse(gate, p2))
		accB = accB.Add(p1.Add(q1)).Sub(p3.Add(p2))
		f8p1 := lfShift3(accB, shift3)
		lfStore[S](rm2, o, f8p1.IfElse(gate, np1.IfElse(need, p1)))
		accB = accB.Add(p0.Add(q2)).Sub(p3.Add(p1))
		f8p0 := lfShift3(accB, shift3)
		lfStore[S](rm1, o, f8p0.IfElse(gate, np0.IfElse(need, p0)))
		accB = accB.Add(q0.Add(q3)).Sub(p3.Add(p0))
		f8q0 := lfShift3(accB, shift3)
		lfStore[S](r0, o, f8q0.IfElse(gate, nq0.IfElse(need, q0)))
		accB = accB.Add(q1.Add(q3)).Sub(p2.Add(q0))
		f8q1 := lfShift3(accB, shift3)
		lfStore[S](rp1, o, f8q1.IfElse(gate, nq1.IfElse(need, q1)))
		accB = accB.Add(q2.Add(q3)).Sub(p1.Add(q1))
		f8q2 := lfShift3(accB, shift3)
		lfStore[S](rp2, o, f8q2.IfElse(gate, q2))
	}
	return changed
}

// lfBatchGroups is the number of eight-position groups a vertical edge gathers
// into scratch per pass. The scratch row stride is 8*lfBatchGroups samples.
const lfBatchGroups = 4

// lfMaxTaps is the largest tap window (filter14: p6..q6) a kernel gathers.
const lfMaxTaps = 14

// lfMaxPlane bounds the planes the vertical batches transpose: their row
// offsets then fit the 32-bit offsets the transposes index with. Larger planes
// (beyond any AV1 frame) take the scalar reference.
const lfMaxPlane = 1 << 32

// lfScratch is the vertical batches' transpose buffer: lfMaxTaps rows of
// lfBatchGroups eight-position groups of two-byte samples. lfScrRow8 and
// lfScrRow16 are its row strides.
type lfScratch [lfMaxTaps * 8 * lfBatchGroups * 2]byte

const (
	lfScrRow8  = 8 * lfBatchGroups
	lfScrRow16 = 16 * lfBatchGroups
)

// lfKind selects a deblocking filter family. The kernels are chosen by a switch
// over direct generic calls rather than by function values: a function value
// would make the vertical scratch escape to the heap.
type lfKind uint8

const (
	lfKind4 lfKind = iota
	lfKind6
	lfKind8
	lfKind14
)

// lfSpec returns the window geometry of a filter family: before is the number
// of taps preceding q0, taps is the window size, and rows [t0, t1) are the
// window rows a kernel may modify (written back after a vertical batch).
func lfSpec(k lfKind) (before, taps, t0, t1 int) {
	switch k {
	case lfKind4:
		return 2, 4, 0, 4
	case lfKind6:
		return 3, 6, 1, 5
	case lfKind8:
		return 4, 8, 1, 7
	}
	return 7, 14, 1, 13
}

// lfCore runs the horizontal Go-native SIMD kernel of family k over length
// positions (a multiple of eight). Positions are contiguous samples of type S
// and step is the byte distance between adjacent taps.
func lfCore[S lfSample](k lfKind, pix []byte, q0Base int, step int, length int, scale int, params filter4Params) uint {
	switch k {
	case lfKind4:
		return lfFilter4Core[S](pix, q0Base, step, length, scale, params)
	case lfKind6:
		return lfFilter6Core[S](pix, q0Base, step, length, scale, params)
	case lfKind8:
		return lfFilter8Core[S](pix, q0Base, step, length, scale, params)
	default:
		return lfFilter14Core[S](pix, q0Base, step, length, scale, params)
	}
}

// lfPure is the scalar reference of family k for the sub-group tail and for
// layouts the SIMD path does not cover. S selects the 8-bit or 10/12-bit form.
func lfPure[S lfSample](k lfKind, pix []byte, q0Base int, step int, outer int, length int, scale int, params filter4Params) {
	if lfSize[S]() == 1 {
		switch k {
		case lfKind4:
			filter4EdgePureGo(pix, q0Base, step, outer, length, params)
		case lfKind6:
			filter6EdgePureGo(pix, q0Base, step, outer, length, scale, params)
		case lfKind8:
			filter8EdgePureGo(pix, q0Base, step, outer, length, scale, params)
		default:
			filter14EdgePureGo(pix, q0Base, step, outer, length, scale, params)
		}
		return
	}
	switch k {
	case lfKind4:
		filter4Edge16PureGo(pix, q0Base, step, outer, length, params)
	case lfKind6:
		filter6Edge16PureGo(pix, q0Base, step, outer, length, scale, params)
	case lfKind8:
		filter8Edge16PureGo(pix, q0Base, step, outer, length, scale, params)
	default:
		filter14Edge16PureGo(pix, q0Base, step, outer, length, scale, params)
	}
}

// lfEdge runs one deblocking edge of family k through the Go-native SIMD kernel.
//
// A horizontal edge has contiguous positions (outer == sample size) and its taps
// are whole rows apart, so the kernel reads the window in place. A vertical edge
// has contiguous taps (step == sample size) and positions one row apart. Batches
// of up to lfBatchGroups groups are transposed into stack scratch, filtered, and
// transposed back only for groups with an active filter lane. Sub-group tails
// and other layouts take the scalar reference.
func lfEdge[S lfSample](k lfKind, pix []byte, q0Base int, step int, outer int, length int, scale int, params filter4Params) {
	sz := lfSize[S]()
	groups := length / 8
	if outer == sz && groups > 0 {
		_ = lfCore[S](k, pix, q0Base, step, groups*8, scale, params)
		if rem := length - groups*8; rem > 0 {
			lfPure[S](k, pix, q0Base+groups*8*outer, step, outer, rem, scale, params)
		}
		return
	}
	if step != sz || groups == 0 || len(pix) > lfMaxPlane {
		lfPure[S](k, pix, q0Base, step, outer, length, scale, params)
		return
	}
	before, taps, t0, t1 := lfSpec(k)
	var scratch lfScratch
	row := 8 * lfBatchGroups * sz
	for g := 0; g < groups; g += lfBatchGroups {
		n := groups - g
		if n > lfBatchGroups {
			n = lfBatchGroups
		}
		src := q0Base + g*8*outer - before*step
		lfVertGather[S](&scratch, pix, src, outer, taps, n*8, row)
		var changed uint
		if k == lfKind4 && sz == 1 && n >= 2 && params.center == 128 && params.min == -128 && params.max == 127 && params.blimit >= 0 && params.blimit < 255 && params.limit >= 0 && params.limit <= 255 && params.hev >= 0 && params.hev <= 255 {
			done := (n * 8) &^ 15
			lfFilter4U8Wide(scratch[:], before*row, row, done, params, &changed)
			if done < n*8 {
				changed |= lfFilter4Core[uint8](scratch[:], before*row+done, row, n*8-done, 1, params) << uint(done/8)
			}
		} else {
			changed = lfCore[S](k, scratch[:], before*row, row, n*8, scale, params)
		}
		if changed != 0 {
			lfVertScatter[S](pix, &scratch, src, outer, t0, t1, n*8, row, changed)
		}
	}
	if rem := length - groups*8; rem > 0 {
		lfPure[S](k, pix, q0Base+groups*8*outer, step, outer, rem, scale, params)
	}
}

// lfVertGather copies the taps of n positions from pix into scratch. Taps are
// contiguous within a position (step == sample size), so tap t of position k sits
// at src + k*outer + t*sz in pix and lands at t*row + k*sz in scratch, where the
// horizontal kernel reads it as a contiguous row.
func lfVertGather[S lfSample](scratch *lfScratch, pix []byte, src int, outer int, taps int, n int, row int) {
	if lfSize[S]() == 1 {
		if taps >= 8 {
			for g := 0; g < n/8; g++ {
				in := src + g*8*outer
				out := g * 8
				lfGatherU8(pix, in, outer, scratch, out, 8)
				if taps == 14 {
					lfGatherU8(pix, in+6, outer, scratch, out+6*row, 8)
				}
			}
			return
		}
		// When the backing row has two spare bytes, a whole-word load is
		// cheaper than assembling four or six bytes. The extra tap rows are
		// discarded; exact-width slices retain the short-load path.
		read := 8
		if src+(n-1)*outer+8 > len(pix) {
			read = taps
		}
		for g := 0; g < n/8; g++ {
			lfGatherU8(pix, src+g*8*outer, outer, scratch, g*8, read)
		}
		return
	}
	if taps >= 8 {
		for g := 0; g < n/8; g++ {
			in := src + g*8*outer
			out := g * 16
			lfGatherU16(pix, in, outer, scratch, out, 8)
			if taps == 14 {
				lfGatherU16(pix, in+12, outer, scratch, out+6*row, 8)
			}
		}
		return
	}
	read := 8
	if src+(n-1)*outer+16 > len(pix) {
		read = taps
	}
	for g := 0; g < n/8; g++ {
		lfGatherU16(pix, src+g*8*outer, outer, scratch, g*16, read)
	}
}

// lfVertScatter writes scratch tap rows [t0, t1) of n positions back to pix,
// the inverse of lfVertGather.
func lfVertScatter[S lfSample](pix []byte, scratch *lfScratch, src int, outer int, t0 int, t1 int, n int, row int, changed uint) {
	if lfSize[S]() == 1 {
		if t1-t0 >= 6 {
			for g := 0; g < n/8; g++ {
				if changed&(1<<uint(g)) == 0 {
					continue
				}
				in := g * 8
				out := src + g*8*outer
				if t1 == 7 { // eight-tap: unchanged outer taps are in scratch too
					lfScatterU8(scratch, in, pix, out, outer, 8)
				} else { // fourteen-tap: only rows 1..12 may change
					lfScatterU8(scratch, in+row, pix, out+1, outer, 8)
					lfScatterU8(scratch, in+5*row, pix, out+5, outer, 8)
				}
			}
			return
		}
		width := 6
		if t1 == 4 {
			width = 4
		}
		for g := 0; g < n/8; g++ {
			if changed&(1<<uint(g)) == 0 {
				continue
			}
			lfScatterU8(scratch, g*8, pix, src+g*8*outer, outer, width)
		}
		return
	}
	if t1-t0 >= 6 {
		for g := 0; g < n/8; g++ {
			if changed&(1<<uint(g)) == 0 {
				continue
			}
			in := g * 16
			out := src + g*8*outer
			if t1 == 7 {
				lfScatterU16(scratch, in, pix, out, outer, 8)
			} else {
				lfScatterU16(scratch, in+row, pix, out+2, outer, 8)
				lfScatterU16(scratch, in+5*row, pix, out+10, outer, 8)
			}
		}
		return
	}
	width := 6
	if t1 == 4 {
		width = 4
	}
	for g := 0; g < n/8; g++ {
		if changed&(1<<uint(g)) == 0 {
			continue
		}
		lfScatterU16(scratch, g*16, pix, src+g*8*outer, outer, width)
	}
}

// lfSample is the sample width a Go-native SIMD deblocking kernel is
// instantiated for: one byte per sample at 8-bit, two bytes at 10/12-bit.
type lfSample interface {
	uint8 | uint16
}

// lfSize returns the byte width of one sample of type S: the all-ones value
// has a bit at position 8 only at two bytes. Each instantiation folds it to a
// constant.
func lfSize[S lfSample]() int {
	return 1 + int(uint16(^S(0))>>8&1)
}

// lfLoad loads eight consecutive samples starting at byte offset off as signed
// 16-bit lanes. Loopfilter samples are at most 4095, so the sign bit stays clear.
// The 8-bit form reads exactly eight bytes, so it never touches memory past the
// tap window. The 16-bit form reads exactly sixteen bytes.
func lfLoad[S lfSample](pix []byte, off int) archsimd.Int16x8 {
	if lfSize[S]() == 1 {
		return lfLoad8(pix, off)
	}
	return lfLoad16(pix, off)
}

// lfStore writes eight filtered samples back starting at byte offset off. Lane
// values are always within the sample range, so the 8-bit narrowing is exact.
func lfStore[S lfSample](pix []byte, off int, v archsimd.Int16x8) {
	if lfSize[S]() == 1 {
		lfStore8(pix, off, v)
		return
	}
	lfStore16(pix, off, v)
}

// lfLoad16 loads eight 16-bit samples directly from the validated window.
func lfLoad16(pix []byte, off int) archsimd.Int16x8 {
	return archsimd.LoadUint8x16Array((*[16]uint8)(pix[off : off+16])).ReshapeToUint16s().BitsToInt16()
}

// lfStore16 writes eight 16-bit lanes to the validated window.
func lfStore16(pix []byte, off int, v archsimd.Int16x8) {
	v.ToBits().ReshapeToUint8s().StoreArray((*[16]uint8)(pix[off : off+16]))
}

// lfAbsDiffInt16x8 computes signed absolute difference in each 16-bit lane,
// preserving the low 16 bits when the full-range difference wraps.
func lfAbsDiffInt16x8(a, b archsimd.Int16x8) archsimd.Int16x8 {
	return a.Max(b).Sub(a.Min(b))
}

// Eight-bit samples have a difference in [-255, 255], so subtraction followed
// by absolute value is exact. The wide path retains the wrapped lane behavior
// tested for the full int16 helper domain.
func lfAbsDiff[S lfSample](a, b archsimd.Int16x8) archsimd.Int16x8 {
	if lfSize[S]() == 1 {
		return a.Sub(b).Abs()
	}
	return lfAbsDiffInt16x8(a, b)
}
