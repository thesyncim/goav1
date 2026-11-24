// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && (arm64 || amd64) && !purego

// Go-native SIMD fourteen-tap deblocking kernel, shared by the 8-bit and
// 10/12-bit paths.

package loopfilter

import "simd/archsimd"

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
