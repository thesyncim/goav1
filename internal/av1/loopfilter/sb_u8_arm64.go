// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package loopfilter

import (
	"simd/archsimd"
	"unsafe"
)

func lpfLoad(p unsafe.Pointer, off int) archsimd.Uint8x16 {
	return archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(p, off)))
}

func lpfStore(p unsafe.Pointer, off int, v archsimd.Uint8x16) {
	v.StoreArray((*[16]uint8)(unsafe.Add(p, off)))
}

func lpfAbsDiff(a, b archsimd.Uint8x16) archsimd.Uint8x16 {
	return a.SubSaturated(b).Or(b.SubSaturated(a))
}

func lpfAny(m archsimd.Mask8x16) bool {
	b := m.ToInt8x16().ToBits().ReshapeToUint64s()
	return b.GetElem(0)|b.GetElem(1) != 0
}

// The filter bodies load, store, widen (ExtendLo8ToUint16, and HiToLo +
// ExtendLo8ToUint16, which is UXTL2) and narrow through the intrinsics written
// out at each use: a helper call alone on its line leaves an inline-mark NOP,
// and ShiftAllRight with a constant rebuilds its count vector (MOVD + INS +
// DUP) every time, so the narrows shift by a count vector built once.

// lpf16U8 is dav1d's lpf_16_wd{4,6,8,16}_neon: it filters sixteen lanes whose
// taps are laid out as rows around q0 at p (tap row k below the edge is at
// p + k*stride, p_k above it at p - (k+1)*stride) and stores the changed rows
// back. wd is the widest filter any lane needs (4, 6, 8 or 16); the per-lane
// masks select each lane's own width exactly as the C reference does.
//
//go:nocheckptr
func lpf16U8W[W lpfWidth](p unsafe.Pointer, stride int, eV, iV, hV archsimd.Uint8x16, m4, mGt4, m16 archsimd.Mask8x16) bool {
	var wdT W
	wd := int(unsafe.Sizeof(wdT))
	p1 := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(p, -2*stride)))
	p0 := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(p, -stride)))
	q0 := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(p, 0)))
	q1 := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(p, stride)))
	d10 := lpfAbsDiff(p1, p0)
	dq := lpfAbsDiff(q1, q0)
	maxd := d10.Max(dq)
	t := maxd
	var p2, q2, p3, q3 archsimd.Uint8x16
	if wd >= 6 {
		p2 = archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(p, -3*stride)))
		q2 = archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(p, 2*stride)))
		tt := lpfAbsDiff(p2, p1).Max(lpfAbsDiff(q2, q1))
		if wd >= 8 {
			p3 = archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(p, -4*stride)))
			q3 = archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(p, 3*stride)))
			tt = tt.Max(lpfAbsDiff(p3, p2).Max(lpfAbsDiff(q3, q2)))
		}
		t = t.Max(tt.Masked(mGt4))
	}
	d00 := lpfAbsDiff(p0, q0)
	sum := d00.AddSaturated(d00).AddSaturated(lpfAbsDiff(p1, q1).ShiftAllRight(1))
	fm := iV.GreaterEqual(t).And(eV.GreaterEqual(sum)).And(m4)
	if !lpfAny(fm) {
		return false
	}

	one := archsimd.BroadcastUint8x16(1)
	mF4 := fm
	var mFlat, mFlat16 archsimd.Mask8x16
	var p6, p5, p4, q4, q5, q6 archsimd.Uint8x16
	if wd >= 6 {
		fl := lpfAbsDiff(p2, p0).Max(maxd).Max(lpfAbsDiff(q2, q0))
		if wd >= 8 {
			fl = fl.Max(lpfAbsDiff(p3, p0)).Max(lpfAbsDiff(q3, q0))
		}
		mFlat = one.GreaterEqual(fl).And(mGt4).And(fm)
		mF4 = fm.And(mFlat.Not())
		if wd == 16 {
			p6 = archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(p, -7*stride)))
			p5 = archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(p, -6*stride)))
			p4 = archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(p, -5*stride)))
			q4 = archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(p, 4*stride)))
			q5 = archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(p, 5*stride)))
			q6 = archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(p, 6*stride)))
			fo := lpfAbsDiff(p6, p0).Max(lpfAbsDiff(p5, p0)).Max(lpfAbsDiff(p4, p0)).
				Max(lpfAbsDiff(q4, q0)).Max(lpfAbsDiff(q5, q0)).Max(lpfAbsDiff(q6, q0))
			mFlat16 = one.GreaterEqual(fo).And(m16).And(mFlat)
			mFlat = mFlat.And(mFlat16.Not())
		}
	}

	// Outputs; every filter reads the original taps, lanes are disjoint.
	o1, o0, n0, n1 := p1, p0, q0, q1

	if lpfAny(mF4) {
		bias := archsimd.BroadcastUint8x16(128)
		zero := archsimd.Uint8x16{}
		hev := maxd.Greater(hV)
		p1c := p1.Xor(bias).BitsToInt8()
		p0c := p0.Xor(bias).BitsToInt8()
		q0c := q0.Xor(bias).BitsToInt8()
		q1c := q1.Xor(bias).BitsToInt8()
		outerF := p1c.SubSaturated(q1c).Masked(hev)
		three16 := archsimd.BroadcastInt16x8(3)
		dLo := q0.InterleaveLo(zero).ReshapeToUint16s().BitsToInt16().Sub(p0.InterleaveLo(zero).ReshapeToUint16s().BitsToInt16())
		dHi := q0.InterleaveHi(zero).ReshapeToUint16s().BitsToInt16().Sub(p0.InterleaveHi(zero).ReshapeToUint16s().BitsToInt16())
		fLo := dLo.Mul(three16).Add(outerF.ExtendLo8ToInt16()).SaturateToInt8()
		fHi := dHi.Mul(three16).Add(outerF.HiToLo().ExtendLo8ToInt16()).SaturateToInt8()
		f := fLo.ToBits().ReshapeToUint64s().InterleaveLo(fHi.ToBits().ReshapeToUint64s()).ReshapeToUint8s().BitsToInt8()
		shift3 := archsimd.BroadcastInt8x16(-3)
		shift1 := archsimd.BroadcastInt8x16(-1)
		filter1 := f.AddSaturated(archsimd.BroadcastInt8x16(4)).Shift(shift3)
		filter2 := f.AddSaturated(archsimd.BroadcastInt8x16(3)).Shift(shift3)
		outer := filter1.Add(archsimd.BroadcastInt8x16(1)).Shift(shift1)
		o0 = p0c.AddSaturated(filter2).ToBits().Xor(bias).IfElse(mF4, p0)
		n0 = q0c.SubSaturated(filter1).ToBits().Xor(bias).IfElse(mF4, q0)
		mOuter := mF4.And(hev.Not())
		o1 = p1c.AddSaturated(outer).ToBits().Xor(bias).IfElse(mOuter, p1)
		n1 = q1c.SubSaturated(outer).ToBits().Xor(bias).IfElse(mOuter, q1)
	}

	if wd == 6 {
		if lpfAny(mFlat) {
			sh3 := archsimd.BroadcastInt16x8(-3)
			four := archsimd.BroadcastUint16x8(4)
			// Low half.
			lp2, lp1, lp0 := p2.ExtendLo8ToUint16(), p1.ExtendLo8ToUint16(), p0.ExtendLo8ToUint16()
			lq0, lq1, lq2 := q0.ExtendLo8ToUint16(), q1.ExtendLo8ToUint16(), q2.ExtendLo8ToUint16()
			s := lp2.Add(lp2).Add(lp2).Add(lp1.Add(lp0).Add(lp1.Add(lp0))).Add(lq0).Add(four)
			ap1 := s
			s = s.Add(lq0.Add(lq1)).Sub(lp2.Add(lp2))
			ap0 := s
			s = s.Add(lq1.Add(lq2)).Sub(lp2.Add(lp1))
			aq0 := s
			s = s.Add(lq2.Add(lq2)).Sub(lp1.Add(lp0))
			aq1 := s
			// High half.
			hp2, hp1, hp0 := p2.HiToLo().ExtendLo8ToUint16(), p1.HiToLo().ExtendLo8ToUint16(), p0.HiToLo().ExtendLo8ToUint16()
			hq0, hq1, hq2 := q0.HiToLo().ExtendLo8ToUint16(), q1.HiToLo().ExtendLo8ToUint16(), q2.HiToLo().ExtendLo8ToUint16()
			s = hp2.Add(hp2).Add(hp2).Add(hp1.Add(hp0).Add(hp1.Add(hp0))).Add(hq0).Add(four)
			bp1 := s
			s = s.Add(hq0.Add(hq1)).Sub(hp2.Add(hp2))
			bp0 := s
			s = s.Add(hq1.Add(hq2)).Sub(hp2.Add(hp1))
			bq0 := s
			s = s.Add(hq2.Add(hq2)).Sub(hp1.Add(hp0))
			bq1 := s
			o1 = ap1.Shift(sh3).ReshapeToUint8s().ConcatEven(bp1.Shift(sh3).ReshapeToUint8s()).IfElse(mFlat, o1)
			o0 = ap0.Shift(sh3).ReshapeToUint8s().ConcatEven(bp0.Shift(sh3).ReshapeToUint8s()).IfElse(mFlat, o0)
			n0 = aq0.Shift(sh3).ReshapeToUint8s().ConcatEven(bq0.Shift(sh3).ReshapeToUint8s()).IfElse(mFlat, n0)
			n1 = aq1.Shift(sh3).ReshapeToUint8s().ConcatEven(bq1.Shift(sh3).ReshapeToUint8s()).IfElse(mFlat, n1)
		}
		o1.StoreArray((*[16]uint8)(unsafe.Add(p, -2*stride)))
		o0.StoreArray((*[16]uint8)(unsafe.Add(p, -stride)))
		n0.StoreArray((*[16]uint8)(unsafe.Add(p, 0)))
		n1.StoreArray((*[16]uint8)(unsafe.Add(p, stride)))
		return true
	}

	o2, n2 := p2, q2
	if wd >= 8 && lpfAny(mFlat) {
		sh3 := archsimd.BroadcastInt16x8(-3)
		four := archsimd.BroadcastUint16x8(4)
		lp3, lp2, lp1, lp0 := p3.ExtendLo8ToUint16(), p2.ExtendLo8ToUint16(), p1.ExtendLo8ToUint16(), p0.ExtendLo8ToUint16()
		lq0, lq1, lq2, lq3 := q0.ExtendLo8ToUint16(), q1.ExtendLo8ToUint16(), q2.ExtendLo8ToUint16(), q3.ExtendLo8ToUint16()
		s := lp3.Add(lp3).Add(lp3).Add(lp2.Add(lp2)).Add(lp1.Add(lp0)).Add(lq0.Add(four))
		ap2 := s
		s = s.Add(lp1.Add(lq1)).Sub(lp3.Add(lp2))
		ap1 := s
		s = s.Add(lp0.Add(lq2)).Sub(lp3.Add(lp1))
		ap0 := s
		s = s.Add(lq0.Add(lq3)).Sub(lp3.Add(lp0))
		aq0 := s
		s = s.Add(lq1.Add(lq3)).Sub(lp2.Add(lq0))
		aq1 := s
		s = s.Add(lq2.Add(lq3)).Sub(lp1.Add(lq1))
		aq2 := s
		hp3, hp2, hp1, hp0 := p3.HiToLo().ExtendLo8ToUint16(), p2.HiToLo().ExtendLo8ToUint16(), p1.HiToLo().ExtendLo8ToUint16(), p0.HiToLo().ExtendLo8ToUint16()
		hq0, hq1, hq2, hq3 := q0.HiToLo().ExtendLo8ToUint16(), q1.HiToLo().ExtendLo8ToUint16(), q2.HiToLo().ExtendLo8ToUint16(), q3.HiToLo().ExtendLo8ToUint16()
		s = hp3.Add(hp3).Add(hp3).Add(hp2.Add(hp2)).Add(hp1.Add(hp0)).Add(hq0.Add(four))
		bp2 := s
		s = s.Add(hp1.Add(hq1)).Sub(hp3.Add(hp2))
		bp1 := s
		s = s.Add(hp0.Add(hq2)).Sub(hp3.Add(hp1))
		bp0 := s
		s = s.Add(hq0.Add(hq3)).Sub(hp3.Add(hp0))
		bq0 := s
		s = s.Add(hq1.Add(hq3)).Sub(hp2.Add(hq0))
		bq1 := s
		s = s.Add(hq2.Add(hq3)).Sub(hp1.Add(hq1))
		bq2 := s
		o2 = ap2.Shift(sh3).ReshapeToUint8s().ConcatEven(bp2.Shift(sh3).ReshapeToUint8s()).IfElse(mFlat, o2)
		o1 = ap1.Shift(sh3).ReshapeToUint8s().ConcatEven(bp1.Shift(sh3).ReshapeToUint8s()).IfElse(mFlat, o1)
		o0 = ap0.Shift(sh3).ReshapeToUint8s().ConcatEven(bp0.Shift(sh3).ReshapeToUint8s()).IfElse(mFlat, o0)
		n0 = aq0.Shift(sh3).ReshapeToUint8s().ConcatEven(bq0.Shift(sh3).ReshapeToUint8s()).IfElse(mFlat, n0)
		n1 = aq1.Shift(sh3).ReshapeToUint8s().ConcatEven(bq1.Shift(sh3).ReshapeToUint8s()).IfElse(mFlat, n1)
		n2 = aq2.Shift(sh3).ReshapeToUint8s().ConcatEven(bq2.Shift(sh3).ReshapeToUint8s()).IfElse(mFlat, n2)
	}

	if wd == 16 && lpfAny(mFlat16) {
		o5, o4, o3, n3, n4, n5 := lpfFlat16U8(p6, p5, p4, p3, p2, p1, p0, q0, q1, q2, q3, q4, q5, q6, mFlat16, &o2, &o1, &o0, &n0, &n1, &n2)
		o5.StoreArray((*[16]uint8)(unsafe.Add(p, -6*stride)))
		o4.StoreArray((*[16]uint8)(unsafe.Add(p, -5*stride)))
		o3.StoreArray((*[16]uint8)(unsafe.Add(p, -4*stride)))
		n3.StoreArray((*[16]uint8)(unsafe.Add(p, 3*stride)))
		n4.StoreArray((*[16]uint8)(unsafe.Add(p, 4*stride)))
		n5.StoreArray((*[16]uint8)(unsafe.Add(p, 5*stride)))
	}
	if wd >= 8 {
		o2.StoreArray((*[16]uint8)(unsafe.Add(p, -3*stride)))
		n2.StoreArray((*[16]uint8)(unsafe.Add(p, 2*stride)))
	}
	o1.StoreArray((*[16]uint8)(unsafe.Add(p, -2*stride)))
	o0.StoreArray((*[16]uint8)(unsafe.Add(p, -stride)))
	n0.StoreArray((*[16]uint8)(unsafe.Add(p, 0)))
	n1.StoreArray((*[16]uint8)(unsafe.Add(p, stride)))
	return true
}

// lpfFlat16U8 computes the 13-tap flat filter (dav1d wd16, AV1 filter14) on
// the lanes in m and blends p2..q2 into the caller's outputs; it returns the
// outer p5..p3 / q3..q5 rows (unchanged outside m).
func lpfFlat16U8(p6, p5, p4, p3, p2, p1, p0, q0, q1, q2, q3, q4, q5, q6 archsimd.Uint8x16, m archsimd.Mask8x16, o2, o1, o0, n0, n1, n2 *archsimd.Uint8x16) (r5, r4, r3, s3, s4, s5 archsimd.Uint8x16) {
	sh4 := archsimd.BroadcastInt16x8(-4)
	var lo, hi [12]archsimd.Uint16x8
	lpfFlat16Half(p6.ExtendLo8ToUint16(), p5.ExtendLo8ToUint16(), p4.ExtendLo8ToUint16(), p3.ExtendLo8ToUint16(),
		p2.ExtendLo8ToUint16(), p1.ExtendLo8ToUint16(), p0.ExtendLo8ToUint16(), q0.ExtendLo8ToUint16(),
		q1.ExtendLo8ToUint16(), q2.ExtendLo8ToUint16(), q3.ExtendLo8ToUint16(), q4.ExtendLo8ToUint16(),
		q5.ExtendLo8ToUint16(), q6.ExtendLo8ToUint16(), &lo)
	lpfFlat16Half(p6.HiToLo().ExtendLo8ToUint16(), p5.HiToLo().ExtendLo8ToUint16(), p4.HiToLo().ExtendLo8ToUint16(), p3.HiToLo().ExtendLo8ToUint16(),
		p2.HiToLo().ExtendLo8ToUint16(), p1.HiToLo().ExtendLo8ToUint16(), p0.HiToLo().ExtendLo8ToUint16(), q0.HiToLo().ExtendLo8ToUint16(),
		q1.HiToLo().ExtendLo8ToUint16(), q2.HiToLo().ExtendLo8ToUint16(), q3.HiToLo().ExtendLo8ToUint16(), q4.HiToLo().ExtendLo8ToUint16(),
		q5.HiToLo().ExtendLo8ToUint16(), q6.HiToLo().ExtendLo8ToUint16(), &hi)
	r5 = lo[0].Shift(sh4).ReshapeToUint8s().ConcatEven(hi[0].Shift(sh4).ReshapeToUint8s()).IfElse(m, p5)
	r4 = lo[1].Shift(sh4).ReshapeToUint8s().ConcatEven(hi[1].Shift(sh4).ReshapeToUint8s()).IfElse(m, p4)
	r3 = lo[2].Shift(sh4).ReshapeToUint8s().ConcatEven(hi[2].Shift(sh4).ReshapeToUint8s()).IfElse(m, p3)
	*o2 = lo[3].Shift(sh4).ReshapeToUint8s().ConcatEven(hi[3].Shift(sh4).ReshapeToUint8s()).IfElse(m, *o2)
	*o1 = lo[4].Shift(sh4).ReshapeToUint8s().ConcatEven(hi[4].Shift(sh4).ReshapeToUint8s()).IfElse(m, *o1)
	*o0 = lo[5].Shift(sh4).ReshapeToUint8s().ConcatEven(hi[5].Shift(sh4).ReshapeToUint8s()).IfElse(m, *o0)
	*n0 = lo[6].Shift(sh4).ReshapeToUint8s().ConcatEven(hi[6].Shift(sh4).ReshapeToUint8s()).IfElse(m, *n0)
	*n1 = lo[7].Shift(sh4).ReshapeToUint8s().ConcatEven(hi[7].Shift(sh4).ReshapeToUint8s()).IfElse(m, *n1)
	*n2 = lo[8].Shift(sh4).ReshapeToUint8s().ConcatEven(hi[8].Shift(sh4).ReshapeToUint8s()).IfElse(m, *n2)
	s3 = lo[9].Shift(sh4).ReshapeToUint8s().ConcatEven(hi[9].Shift(sh4).ReshapeToUint8s()).IfElse(m, q3)
	s4 = lo[10].Shift(sh4).ReshapeToUint8s().ConcatEven(hi[10].Shift(sh4).ReshapeToUint8s()).IfElse(m, q4)
	s5 = lo[11].Shift(sh4).ReshapeToUint8s().ConcatEven(hi[11].Shift(sh4).ReshapeToUint8s()).IfElse(m, q5)
	return
}

// lpfFlat16Half runs the dav1d sliding-sum form of the 13-tap filter on eight
// widened lanes; out[k] holds the biased sum of output p5+k (p5..q5).
func lpfFlat16Half(p6, p5, p4, p3, p2, p1, p0, q0, q1, q2, q3, q4, q5, q6 archsimd.Uint16x8, out *[12]archsimd.Uint16x8) {
	p6x2 := p6.Add(p6)
	s := p6.Add(p6x2).Add(p6x2.Add(p6x2)) // 7*p6
	s = s.Add(p5.Add(p5)).Add(p4.Add(p4)).Add(p3.Add(p2)).Add(p1.Add(p0)).Add(q0).Add(archsimd.BroadcastUint16x8(8))
	out[0] = s
	s = s.Add(p3.Add(q1)).Sub(p6x2)
	out[1] = s
	s = s.Add(p2.Add(q2)).Sub(p6.Add(p5))
	out[2] = s
	s = s.Add(p1.Add(q3)).Sub(p6.Add(p4))
	out[3] = s
	s = s.Add(p0.Add(q4)).Sub(p6.Add(p3))
	out[4] = s
	s = s.Add(q0.Add(q5)).Sub(p6.Add(p2))
	out[5] = s
	s = s.Add(q1.Add(q6)).Sub(p6.Add(p1))
	out[6] = s
	s = s.Add(q2.Add(q6)).Sub(p5.Add(p0))
	out[7] = s
	s = s.Add(q3.Add(q6)).Sub(p4.Add(q0))
	out[8] = s
	s = s.Add(q4.Add(q6)).Sub(p3.Add(q1))
	out[9] = s
	s = s.Add(q5.Add(q6)).Sub(p2.Add(q2))
	out[10] = s
	s = s.Add(q6.Add(q6)).Sub(p1.Add(q3))
	out[11] = s
}

// sbNibbleLanes[b] selects the four byte lanes of every unit in the four-unit
// group whose bit is set in b (unit k owns lanes 4k..4k+3): dav1d lpf_func's
// dup + cmtst of a vmask nibble against word_1248, as one table load.
var sbNibbleLanes = func() (t [16][16]uint8) {
	for b := range t {
		for k := 0; k < 4; k++ {
			if b&(1<<k) != 0 {
				for i := 0; i < 4; i++ {
					t[b][4*k+i] = 0xff
				}
			}
		}
	}
	return t
}()

func sbNibbleMask(n uint32) archsimd.Mask8x16 {
	return archsimd.LoadUint8x16Array(&sbNibbleLanes[n&15]).BitsToInt8().ToMask()
}

// sbAcc carries goav1's apply counters through a superblock call as vector
// accumulators (one reduction per call instead of per group); dav1d keeps no
// such counters.
type sbAcc struct {
	cnt, max archsimd.Uint8x16
}

func (a *sbAcc) stats() SBStats {
	return SBStats{
		Applied:  uint32(a.cnt.ReduceSum()) >> 2,
		MaxLevel: uint8(a.max.BitsToInt8().ReduceMax()),
	}
}

// sbGroupLimits is the per-group prologue of dav1d's lpf_{v,h}_sb_{y,uv}
// drivers (src/arm/64/loopfilter.S lpf_func) across lanes, one unit per
// 32-bit lane: cur and prev hold the units' own and fallback (left or above)
// levels in the low byte; L = cur ? cur : prev is replicated over the unit's
// four bytes, I = imax(imin(L >> sharp[0], sharp[1]), 1), E = 2*(L+2)+I and
// H = L>>4 follow without a table, and the "wd >= 4" lane mask is the vmask
// nibble restricted to L != 0. ok is false when no selected unit has a level
// (dav1d "if (!L) continue"). The wider-width masks need no level test: the
// kernel already ANDs them with fm, which includes m4.
func sbGroupLimits(cur, prev archsimd.Uint32x4, nAll uint32, sharp0 uint64, sharp1 uint8, acc *sbAcc) (eV, iV, hV archsimd.Uint8x16, m4 archsimd.Mask8x16, ok bool) {
	zero32 := archsimd.Uint32x4{}
	L := prev.IfElse(cur.Equal(zero32), cur)
	lr := L.Mul(archsimd.BroadcastUint32x4(0x01010101)).ReshapeToUint8s()
	zero := archsimd.Uint8x16{}
	act := archsimd.LoadUint8x16Array(&sbNibbleLanes[nAll&15]).AndNot(lr.Equal(zero).ToInt8x16().ToBits())
	h := act.ReshapeToUint64s()
	if h.GetElem(0)|h.GetElem(1) == 0 {
		return
	}
	acc.cnt = acc.cnt.Sub(act)
	acc.max = acc.max.Max(lr.And(act))
	hV = lr.ShiftAllRight(4)
	iV = lr.ShiftAllRight(sharp0).Min(archsimd.BroadcastUint8x16(sharp1)).Max(archsimd.BroadcastUint8x16(1))
	e := lr.Add(archsimd.BroadcastUint8x16(2))
	eV = e.Add(e).Add(iV)
	m4 = act.BitsToInt8().ToMask()
	ok = true
	return
}

// sbWidth is dav1d lpf_func's kernel choice from the raw vmask nibbles: the
// per-lane masks narrow every lane to its own width inside the kernel.
func sbWidth(nGt4, n16 uint32, luma bool) int {
	switch {
	case n16 != 0:
		return 16
	case nGt4 != 0:
		if luma {
			return 8
		}
		return 6
	default:
		return 4
	}
}

// sbLevelsRow loads the levels of four consecutive units (horizontal edges:
// dav1d lpf_v_sb ld1 of l[0] and l[-b4_stride]) with component comp moved to
// the low byte of each lane.
//
//go:nocheckptr
func sbLevelsRow(lvl [][4]uint8, u0, lvlStride, comp int, nAll uint32) (cur, prev archsimd.Uint32x4) {
	ff := archsimd.BroadcastUint32x4(0xff)
	sh := uint64(8 * comp)
	if u0-lvlStride >= 0 && u0+4 <= len(lvl) {
		base := unsafe.Pointer(unsafe.SliceData(lvl))
		cur = archsimd.LoadUint32x4Array((*[4]uint32)(unsafe.Add(base, 4*u0))).ShiftAllRight(sh).And(ff)
		prev = archsimd.LoadUint32x4Array((*[4]uint32)(unsafe.Add(base, 4*(u0-lvlStride)))).ShiftAllRight(sh).And(ff)
		return
	}
	var c, p [4]uint32
	for k := 0; k < 4; k++ {
		if nAll&(1<<uint(k)) != 0 {
			c[k] = uint32(lvl[u0+k][comp])
			p[k] = uint32(lvl[u0+k-lvlStride][comp])
		}
	}
	return archsimd.LoadUint32x4Array(&c), archsimd.LoadUint32x4Array(&p)
}

// sbLevelsCol loads the levels of four units stacked lvlStride cells apart
// (vertical edges: dav1d lpf_h_sb ld2 of the l[-1], l[0] pairs) with
// component comp moved to the low byte of each lane.
//
//go:nocheckptr
func sbLevelsCol(lvl [][4]uint8, li, lvlStride, comp int, nAll uint32) (cur, prev archsimd.Uint32x4) {
	ff := archsimd.BroadcastUint32x4(0xff)
	sh := uint64(8 * comp)
	if li-1 >= 0 && li+3*lvlStride < len(lvl) {
		base := unsafe.Pointer(unsafe.SliceData(lvl))
		var pr [4]uint64
		pr[0] = *(*uint64)(unsafe.Add(base, 4*(li-1)))
		pr[1] = *(*uint64)(unsafe.Add(base, 4*(li+lvlStride-1)))
		pr[2] = *(*uint64)(unsafe.Add(base, 4*(li+2*lvlStride-1)))
		pr[3] = *(*uint64)(unsafe.Add(base, 4*(li+3*lvlStride-1)))
		a := archsimd.LoadUint64x2Array((*[2]uint64)(pr[0:2])).ReshapeToUint32s()
		b := archsimd.LoadUint64x2Array((*[2]uint64)(pr[2:4])).ReshapeToUint32s()
		prev = a.ConcatEven(b).ShiftAllRight(sh).And(ff)
		cur = a.ConcatOdd(b).ShiftAllRight(sh).And(ff)
		return
	}
	var c, p [4]uint32
	for k := 0; k < 4; k++ {
		if nAll&(1<<uint(k)) != 0 {
			c[k] = uint32(lvl[li+k*lvlStride][comp])
			p[k] = uint32(lvl[li+k*lvlStride-1][comp])
		}
	}
	return archsimd.LoadUint32x4Array(&c), archsimd.LoadUint32x4Array(&p)
}

//go:nocheckptr
func filterSBHorizontal8(pix []byte, off, stride, rowBytes int, vm [3]uint32, luma bool, lvl [][4]uint8, lvlOff, lvlStride, comp int, lut *SBLUT) SBStats {
	var st SBStats
	var acc sbAcc
	gt4All := vm[1] | vm[2]
	all := vm[0] | gt4All
	base := unsafe.Pointer(unsafe.SliceData(pix))
	sharp0, sharp1 := uint64(lut.Sharp[0]), lut.Sharp[1]
	for g := 0; g < 8 && all>>(4*uint(g)) != 0; g++ {
		sh := 4 * uint(g)
		nib := (all >> sh) & 0xf
		if nib == 0 {
			continue
		}
		nGt4 := (gt4All >> sh) & 0xf
		n16 := (vm[2] >> sh) & 0xf
		wd := sbWidth(nGt4, n16, luma)
		u0 := lvlOff + 4*g
		q0Off := off + 16*g
		reach := 4
		if wd == 16 {
			reach = 7
		}
		if 16*g+16 > rowBytes || q0Off-reach*stride < 0 || q0Off+(reach-1)*stride+16 > len(pix) {
			sub := vm
			for k := range sub {
				sub[k] = (sub[k] >> sh) & 0xf
			}
			st.Add(filterSBHorizontal8Scalar(pix, q0Off, stride, sub, luma, lvl, u0, lvlStride, comp, lut))
			continue
		}
		cur, prev := sbLevelsRow(lvl, u0, lvlStride, comp, nib)
		eV, iV, hV, m4, ok := sbGroupLimits(cur, prev, nib, sharp0, sharp1, &acc)
		if !ok {
			continue
		}
		q0 := unsafe.Add(base, q0Off)
		mGt4, m16 := sbNibbleMask(nGt4), sbNibbleMask(n16)
		switch wd {
		case 16:
			lpf16U8W[lpfW16](q0, stride, eV, iV, hV, m4, mGt4, m16)
		case 8:
			lpf16U8W[lpfW8](q0, stride, eV, iV, hV, m4, mGt4, m16)
		case 6:
			lpf16U8W[lpfW6](q0, stride, eV, iV, hV, m4, mGt4, m16)
		default:
			lpf16U8W[lpfW4](q0, stride, eV, iV, hV, m4, mGt4, m16)
		}
	}
	st.Add(acc.stats())
	return st
}

//go:nocheckptr
func filterSBVertical8(pix []byte, off, stride, rowBytes int, vm [3]uint32, luma bool, lvl [][4]uint8, lvlOff, lvlStride, comp int, lut *SBLUT) SBStats {
	var st SBStats
	var acc sbAcc
	gt4All := vm[1] | vm[2]
	all := vm[0] | gt4All
	base := unsafe.Pointer(unsafe.SliceData(pix))
	sharp0, sharp1 := uint64(lut.Sharp[0]), lut.Sharp[1]
	for g := 0; g < 8 && all>>(4*uint(g)) != 0; g++ {
		sh := 4 * uint(g)
		nib := (all >> sh) & 0xf
		if nib == 0 {
			continue
		}
		nGt4 := (gt4All >> sh) & 0xf
		n16 := (vm[2] >> sh) & 0xf
		wd := sbWidth(nGt4, n16, luma)
		u0 := lvlOff + 4*g*lvlStride
		q0Off := off + 16*g*stride
		lastRow := q0Off + 15*stride
		if wd == 16 {
			if q0Off-8 < 0 || lastRow+8 > len(pix) {
				st.Add(sbVerticalScalarGroup(pix, q0Off, stride, vm, sh, luma, lvl, u0, lvlStride, comp, lut))
				continue
			}
		} else if q0Off-4 < 0 || lastRow+4 > len(pix) {
			st.Add(sbVerticalScalarGroup(pix, q0Off, stride, vm, sh, luma, lvl, u0, lvlStride, comp, lut))
			continue
		}
		cur, prev := sbLevelsCol(lvl, u0, lvlStride, comp, nib)
		eV, iV, hV, m4, ok := sbGroupLimits(cur, prev, nib, sharp0, sharp1, &acc)
		if !ok {
			continue
		}
		if wd == 16 {
			lpfVertical16x16U8(unsafe.Add(base, q0Off-8), stride, eV, iV, hV, m4, sbNibbleMask(nGt4), sbNibbleMask(n16))
		} else if rowBytes >= 12 && lastRow+12 <= len(pix) {
			p3 := unsafe.Add(base, q0Off-4)
			mGt4 := sbNibbleMask(nGt4)
			switch wd {
			case 8:
				lpfVertical16x8U8WideW[lpfW8](p3, stride, eV, iV, hV, m4, mGt4)
			case 6:
				lpfVertical16x8U8WideW[lpfW6](p3, stride, eV, iV, hV, m4, mGt4)
			default:
				lpfVertical16x8U8WideW[lpfW4](p3, stride, eV, iV, hV, m4, mGt4)
			}
		} else {
			lpfVertical16x8U8(unsafe.Add(base, q0Off-4), stride, wd, eV, iV, hV, m4, sbNibbleMask(nGt4), sbNibbleMask(n16))
		}
	}
	st.Add(acc.stats())
	return st
}

func sbVerticalScalarGroup(pix []byte, q0Off, stride int, vm [3]uint32, sh uint, luma bool, lvl [][4]uint8, u0, lvlStride, comp int, lut *SBLUT) SBStats {
	sub := vm
	for k := range sub {
		sub[k] = (sub[k] >> sh) & 0xf
	}
	return filterSBVertical8Scalar(pix, q0Off, stride, sub, luma, lvl, u0, lvlStride, comp, lut)
}

// lpfVertical16x8U8 filters a vertical edge across sixteen rows using eight
// bytes per row (p3..q3): p points at p3 of the first row. The rows are
// transposed into tap vectors, filtered, and transposed back.
//
//go:nocheckptr
func lpfVertical16x8U8(p unsafe.Pointer, stride int, wd int, eV, iV, hV archsimd.Uint8x16, m4, mGt4, m16 archsimd.Mask8x16) {
	var buf [8][16]uint8
	// Pair row r (low half) with row r+8 (high half), then an 8x8 byte
	// transpose of both halves at once yields one vector per tap.
	var pr [8][2]uint64
	for r := 0; r < 8; r++ {
		pr[r][0] = *(*uint64)(unsafe.Add(p, r*stride))
		pr[r][1] = *(*uint64)(unsafe.Add(p, (r+8)*stride))
	}
	a := archsimd.LoadUint64x2Array(&pr[0]).ReshapeToUint8s()
	b := archsimd.LoadUint64x2Array(&pr[1]).ReshapeToUint8s()
	c := archsimd.LoadUint64x2Array(&pr[2]).ReshapeToUint8s()
	d := archsimd.LoadUint64x2Array(&pr[3]).ReshapeToUint8s()
	e := archsimd.LoadUint64x2Array(&pr[4]).ReshapeToUint8s()
	f := archsimd.LoadUint64x2Array(&pr[5]).ReshapeToUint8s()
	g := archsimd.LoadUint64x2Array(&pr[6]).ReshapeToUint8s()
	h := archsimd.LoadUint64x2Array(&pr[7]).ReshapeToUint8s()
	t0, t1, t2, t3, t4, t5, t6, t7 := lpfTranspose8x8Pairs(a, b, c, d, e, f, g, h)
	// t_k now holds tap k (p3..q3) for rows 0..7 (low) and 8..15 (high), but
	// the pairing interleaves the row order: lane i<8 is row i, lane 8+i row 8+i.
	t0.StoreArray(&buf[0])
	t1.StoreArray(&buf[1])
	t2.StoreArray(&buf[2])
	t3.StoreArray(&buf[3])
	t4.StoreArray(&buf[4])
	t5.StoreArray(&buf[5])
	t6.StoreArray(&buf[6])
	t7.StoreArray(&buf[7])
	if !lpf16U8(unsafe.Pointer(&buf[4]), 16, wd, eV, iV, hV, m4, mGt4, m16) {
		return
	}
	a = archsimd.LoadUint8x16Array(&buf[0])
	b = archsimd.LoadUint8x16Array(&buf[1])
	c = archsimd.LoadUint8x16Array(&buf[2])
	d = archsimd.LoadUint8x16Array(&buf[3])
	e = archsimd.LoadUint8x16Array(&buf[4])
	f = archsimd.LoadUint8x16Array(&buf[5])
	g = archsimd.LoadUint8x16Array(&buf[6])
	h = archsimd.LoadUint8x16Array(&buf[7])
	t0, t1, t2, t3, t4, t5, t6, t7 = lpfTranspose8x8Pairs(a, b, c, d, e, f, g, h)
	t0.ReshapeToUint64s().StoreArray(&pr[0])
	t1.ReshapeToUint64s().StoreArray(&pr[1])
	t2.ReshapeToUint64s().StoreArray(&pr[2])
	t3.ReshapeToUint64s().StoreArray(&pr[3])
	t4.ReshapeToUint64s().StoreArray(&pr[4])
	t5.ReshapeToUint64s().StoreArray(&pr[5])
	t6.ReshapeToUint64s().StoreArray(&pr[6])
	t7.ReshapeToUint64s().StoreArray(&pr[7])
	for r := 0; r < 8; r++ {
		*(*uint64)(unsafe.Add(p, r*stride)) = pr[r][0]
		*(*uint64)(unsafe.Add(p, (r+8)*stride)) = pr[r][1]
	}
}

// lpfTranspose8x8Pairs transposes the 8x8 byte matrix held in the low halves
// and, independently, the one in the high halves of eight vectors.
func lpfTranspose8x8Pairs(a, b, c, d, e, f, g, h archsimd.Uint8x16) (o0, o1, o2, o3, o4, o5, o6, o7 archsimd.Uint8x16) {
	ab0, ab1 := a.InterleaveEven(b), a.InterleaveOdd(b)
	cd0, cd1 := c.InterleaveEven(d), c.InterleaveOdd(d)
	ef0, ef1 := e.InterleaveEven(f), e.InterleaveOdd(f)
	gh0, gh1 := g.InterleaveEven(h), g.InterleaveOdd(h)
	ac0, ac2 := ab0.ReshapeToUint16s().InterleaveEven(cd0.ReshapeToUint16s()), ab0.ReshapeToUint16s().InterleaveOdd(cd0.ReshapeToUint16s())
	ac1, ac3 := ab1.ReshapeToUint16s().InterleaveEven(cd1.ReshapeToUint16s()), ab1.ReshapeToUint16s().InterleaveOdd(cd1.ReshapeToUint16s())
	eg0, eg2 := ef0.ReshapeToUint16s().InterleaveEven(gh0.ReshapeToUint16s()), ef0.ReshapeToUint16s().InterleaveOdd(gh0.ReshapeToUint16s())
	eg1, eg3 := ef1.ReshapeToUint16s().InterleaveEven(gh1.ReshapeToUint16s()), ef1.ReshapeToUint16s().InterleaveOdd(gh1.ReshapeToUint16s())
	o0 = ac0.ReshapeToUint32s().InterleaveEven(eg0.ReshapeToUint32s()).ReshapeToUint8s()
	o1 = ac1.ReshapeToUint32s().InterleaveEven(eg1.ReshapeToUint32s()).ReshapeToUint8s()
	o2 = ac2.ReshapeToUint32s().InterleaveEven(eg2.ReshapeToUint32s()).ReshapeToUint8s()
	o3 = ac3.ReshapeToUint32s().InterleaveEven(eg3.ReshapeToUint32s()).ReshapeToUint8s()
	o4 = ac0.ReshapeToUint32s().InterleaveOdd(eg0.ReshapeToUint32s()).ReshapeToUint8s()
	o5 = ac1.ReshapeToUint32s().InterleaveOdd(eg1.ReshapeToUint32s()).ReshapeToUint8s()
	o6 = ac2.ReshapeToUint32s().InterleaveOdd(eg2.ReshapeToUint32s()).ReshapeToUint8s()
	o7 = ac3.ReshapeToUint32s().InterleaveOdd(eg3.ReshapeToUint32s()).ReshapeToUint8s()
	return
}

// lpfVertical16x16U8 is the wd16 variant: sixteen bytes per row (p7..q7, p
// points at p7 of the first row), a full 16x16 transpose each way.
//
//go:nocheckptr
func lpfVertical16x16U8(p unsafe.Pointer, stride int, eV, iV, hV archsimd.Uint8x16, m4, mGt4, m16 archsimd.Mask8x16) {
	var buf [16][16]uint8
	lpfTranspose16x16Store(p, stride, unsafe.Pointer(&buf), 16)
	if !lpf16U8(unsafe.Pointer(&buf[8]), 16, 16, eV, iV, hV, m4, mGt4, m16) {
		return
	}
	lpfTranspose16x16Store(unsafe.Pointer(&buf), 16, p, stride)
}

func lpfTrn8(a, b archsimd.Uint8x16) (archsimd.Uint8x16, archsimd.Uint8x16) {
	return a.InterleaveEven(b), a.InterleaveOdd(b)
}

func lpfTrn16(a, b archsimd.Uint8x16) (archsimd.Uint8x16, archsimd.Uint8x16) {
	x, y := a.ReshapeToUint16s(), b.ReshapeToUint16s()
	return x.InterleaveEven(y).ReshapeToUint8s(), x.InterleaveOdd(y).ReshapeToUint8s()
}

func lpfTrn32(a, b archsimd.Uint8x16) (archsimd.Uint8x16, archsimd.Uint8x16) {
	x, y := a.ReshapeToUint32s(), b.ReshapeToUint32s()
	return x.InterleaveEven(y).ReshapeToUint8s(), x.InterleaveOdd(y).ReshapeToUint8s()
}

func lpfZip64(a, b archsimd.Uint8x16) (archsimd.Uint8x16, archsimd.Uint8x16) {
	x, y := a.ReshapeToUint64s(), b.ReshapeToUint64s()
	return x.InterleaveLo(y).ReshapeToUint8s(), x.InterleaveHi(y).ReshapeToUint8s()
}

// lpfTranspose16x16Store loads sixteen 16-byte rows from src, transposes the
// 16x16 byte matrix in registers and stores it as sixteen rows at dst.
//
//go:nocheckptr
func lpfTranspose16x16Store(src unsafe.Pointer, srcStride int, dst unsafe.Pointer, dstStride int) {
	r0, r1 := lpfTrn8(lpfLoad(src, 0), lpfLoad(src, srcStride))
	r2, r3 := lpfTrn8(lpfLoad(src, 2*srcStride), lpfLoad(src, 3*srcStride))
	r4, r5 := lpfTrn8(lpfLoad(src, 4*srcStride), lpfLoad(src, 5*srcStride))
	r6, r7 := lpfTrn8(lpfLoad(src, 6*srcStride), lpfLoad(src, 7*srcStride))
	r8, r9 := lpfTrn8(lpfLoad(src, 8*srcStride), lpfLoad(src, 9*srcStride))
	r10, r11 := lpfTrn8(lpfLoad(src, 10*srcStride), lpfLoad(src, 11*srcStride))
	r12, r13 := lpfTrn8(lpfLoad(src, 12*srcStride), lpfLoad(src, 13*srcStride))
	r14, r15 := lpfTrn8(lpfLoad(src, 14*srcStride), lpfLoad(src, 15*srcStride))

	r0, r2 = lpfTrn16(r0, r2)
	r1, r3 = lpfTrn16(r1, r3)
	r4, r6 = lpfTrn16(r4, r6)
	r5, r7 = lpfTrn16(r5, r7)
	r8, r10 = lpfTrn16(r8, r10)
	r9, r11 = lpfTrn16(r9, r11)
	r12, r14 = lpfTrn16(r12, r14)
	r13, r15 = lpfTrn16(r13, r15)

	r0, r4 = lpfTrn32(r0, r4)
	r1, r5 = lpfTrn32(r1, r5)
	r2, r6 = lpfTrn32(r2, r6)
	r3, r7 = lpfTrn32(r3, r7)
	r8, r12 = lpfTrn32(r8, r12)
	r9, r13 = lpfTrn32(r9, r13)
	r10, r14 = lpfTrn32(r10, r14)
	r11, r15 = lpfTrn32(r11, r15)

	r0, r8 = lpfZip64(r0, r8)
	r1, r9 = lpfZip64(r1, r9)
	r2, r10 = lpfZip64(r2, r10)
	r3, r11 = lpfZip64(r3, r11)
	r4, r12 = lpfZip64(r4, r12)
	r5, r13 = lpfZip64(r5, r13)
	r6, r14 = lpfZip64(r6, r14)
	r7, r15 = lpfZip64(r7, r15)

	lpfStore(dst, 0, r0)
	lpfStore(dst, dstStride, r1)
	lpfStore(dst, 2*dstStride, r2)
	lpfStore(dst, 3*dstStride, r3)
	lpfStore(dst, 4*dstStride, r4)
	lpfStore(dst, 5*dstStride, r5)
	lpfStore(dst, 6*dstStride, r6)
	lpfStore(dst, 7*dstStride, r7)
	lpfStore(dst, 8*dstStride, r8)
	lpfStore(dst, 9*dstStride, r9)
	lpfStore(dst, 10*dstStride, r10)
	lpfStore(dst, 11*dstStride, r11)
	lpfStore(dst, 12*dstStride, r12)
	lpfStore(dst, 13*dstStride, r13)
	lpfStore(dst, 14*dstStride, r14)
	lpfStore(dst, 15*dstStride, r15)
}

// lpfLowHalf selects the low eight bytes of a vector.
var lpfLowHalf = [16]uint8{0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff}

// lpfVertical16x8U8Wide is lpfVertical16x8U8 for groups whose rows can be read
// and rewritten sixteen bytes at a time from p (p3 of the edge): each row goes
// straight into a vector register, rows r and r+8 pair with one zip, and the
// filtered eight bytes are merged back into the untouched upper eight with one
// bit-select per row, avoiding the scalar gather/scatter through the stack.
//
//go:nocheckptr
func lpfVertical16x8U8WideW[W lpfWidth](p unsafe.Pointer, stride int, eV, iV, hV archsimd.Uint8x16, m4, mGt4 archsimd.Mask8x16) {
	r0, r8 := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(p, 0))), archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(p, 8*stride)))
	r1, r9 := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(p, stride))), archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(p, 9*stride)))
	r2, r10 := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(p, 2*stride))), archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(p, 10*stride)))
	r3, r11 := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(p, 3*stride))), archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(p, 11*stride)))
	r4, r12 := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(p, 4*stride))), archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(p, 12*stride)))
	r5, r13 := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(p, 5*stride))), archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(p, 13*stride)))
	r6, r14 := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(p, 6*stride))), archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(p, 14*stride)))
	r7, r15 := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(p, 7*stride))), archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(p, 15*stride)))
	t0, t1, t2, t3, t4, t5, t6, t7 := lpfTranspose8x8Pairs(
		r0.ReshapeToUint64s().InterleaveLo(r8.ReshapeToUint64s()).ReshapeToUint8s(), r1.ReshapeToUint64s().InterleaveLo(r9.ReshapeToUint64s()).ReshapeToUint8s(), r2.ReshapeToUint64s().InterleaveLo(r10.ReshapeToUint64s()).ReshapeToUint8s(), r3.ReshapeToUint64s().InterleaveLo(r11.ReshapeToUint64s()).ReshapeToUint8s(),
		r4.ReshapeToUint64s().InterleaveLo(r12.ReshapeToUint64s()).ReshapeToUint8s(), r5.ReshapeToUint64s().InterleaveLo(r13.ReshapeToUint64s()).ReshapeToUint8s(), r6.ReshapeToUint64s().InterleaveLo(r14.ReshapeToUint64s()).ReshapeToUint8s(), r7.ReshapeToUint64s().InterleaveLo(r15.ReshapeToUint64s()).ReshapeToUint8s())
	var ok bool
	t1, t2, t3, t4, t5, t6, ok = lpf16U8RegsW[W](t0, t1, t2, t3, t4, t5, t6, t7, eV, iV, hV, m4, mGt4)
	if !ok {
		// dav1d lpf_16_wd* x14 = 1: nothing filtered, nothing to store.
		return
	}
	t0, t1, t2, t3, t4, t5, t6, t7 = lpfTranspose8x8Pairs(t0, t1, t2, t3, t4, t5, t6, t7)
	lo := archsimd.LoadUint8x16Array(&lpfLowHalf).BitsToInt8().ToMask()
	t0.IfElse(lo, r0).StoreArray((*[16]uint8)(unsafe.Add(p, 0)))
	t1.IfElse(lo, r1).StoreArray((*[16]uint8)(unsafe.Add(p, stride)))
	t2.IfElse(lo, r2).StoreArray((*[16]uint8)(unsafe.Add(p, 2*stride)))
	t3.IfElse(lo, r3).StoreArray((*[16]uint8)(unsafe.Add(p, 3*stride)))
	t4.IfElse(lo, r4).StoreArray((*[16]uint8)(unsafe.Add(p, 4*stride)))
	t5.IfElse(lo, r5).StoreArray((*[16]uint8)(unsafe.Add(p, 5*stride)))
	t6.IfElse(lo, r6).StoreArray((*[16]uint8)(unsafe.Add(p, 6*stride)))
	t7.IfElse(lo, r7).StoreArray((*[16]uint8)(unsafe.Add(p, 7*stride)))
	t0.ReshapeToUint64s().InterleaveHi(r8.ReshapeToUint64s()).ReshapeToUint8s().StoreArray((*[16]uint8)(unsafe.Add(p, 8*stride)))
	t1.ReshapeToUint64s().InterleaveHi(r9.ReshapeToUint64s()).ReshapeToUint8s().StoreArray((*[16]uint8)(unsafe.Add(p, 9*stride)))
	t2.ReshapeToUint64s().InterleaveHi(r10.ReshapeToUint64s()).ReshapeToUint8s().StoreArray((*[16]uint8)(unsafe.Add(p, 10*stride)))
	t3.ReshapeToUint64s().InterleaveHi(r11.ReshapeToUint64s()).ReshapeToUint8s().StoreArray((*[16]uint8)(unsafe.Add(p, 11*stride)))
	t4.ReshapeToUint64s().InterleaveHi(r12.ReshapeToUint64s()).ReshapeToUint8s().StoreArray((*[16]uint8)(unsafe.Add(p, 12*stride)))
	t5.ReshapeToUint64s().InterleaveHi(r13.ReshapeToUint64s()).ReshapeToUint8s().StoreArray((*[16]uint8)(unsafe.Add(p, 13*stride)))
	t6.ReshapeToUint64s().InterleaveHi(r14.ReshapeToUint64s()).ReshapeToUint8s().StoreArray((*[16]uint8)(unsafe.Add(p, 14*stride)))
	t7.ReshapeToUint64s().InterleaveHi(r15.ReshapeToUint64s()).ReshapeToUint8s().StoreArray((*[16]uint8)(unsafe.Add(p, 15*stride)))
}

// lpf16U8Regs is lpf16U8 for widths up to 8 on taps held in registers:
// p3..q3 in, the filtered p2..q2 out (p3/q3 never change at these widths).
// Vector arguments and results travel in SIMD registers, so the transposed
// vertical path filters without a stack round trip.
func lpf16U8RegsW[W lpfWidth](p3, p2, p1, p0, q0, q1, q2, q3, eV, iV, hV archsimd.Uint8x16, m4, mGt4 archsimd.Mask8x16) (o2, o1, o0, n0, n1, n2 archsimd.Uint8x16, ok bool) {
	var wdT W
	wd := int(unsafe.Sizeof(wdT))
	o2, o1, o0, n0, n1, n2 = p2, p1, p0, q0, q1, q2
	d10 := lpfAbsDiff(p1, p0)
	dq := lpfAbsDiff(q1, q0)
	maxd := d10.Max(dq)
	t := maxd
	if wd >= 6 {
		tt := lpfAbsDiff(p2, p1).Max(lpfAbsDiff(q2, q1))
		if wd >= 8 {
			tt = tt.Max(lpfAbsDiff(p3, p2).Max(lpfAbsDiff(q3, q2)))
		}
		t = t.Max(tt.Masked(mGt4))
	}
	d00 := lpfAbsDiff(p0, q0)
	sum := d00.AddSaturated(d00).AddSaturated(lpfAbsDiff(p1, q1).ShiftAllRight(1))
	fm := iV.GreaterEqual(t).And(eV.GreaterEqual(sum)).And(m4)
	if !lpfAny(fm) {
		return
	}
	ok = true
	mF4 := fm
	var mFlat archsimd.Mask8x16
	if wd >= 6 {
		fl := lpfAbsDiff(p2, p0).Max(maxd).Max(lpfAbsDiff(q2, q0))
		if wd >= 8 {
			fl = fl.Max(lpfAbsDiff(p3, p0)).Max(lpfAbsDiff(q3, q0))
		}
		mFlat = archsimd.BroadcastUint8x16(1).GreaterEqual(fl).And(mGt4).And(fm)
		mF4 = fm.And(mFlat.Not())
	}
	if lpfAny(mF4) {
		bias := archsimd.BroadcastUint8x16(128)
		zero := archsimd.Uint8x16{}
		hev := maxd.Greater(hV)
		p1c := p1.Xor(bias).BitsToInt8()
		p0c := p0.Xor(bias).BitsToInt8()
		q0c := q0.Xor(bias).BitsToInt8()
		q1c := q1.Xor(bias).BitsToInt8()
		outerF := p1c.SubSaturated(q1c).Masked(hev)
		three16 := archsimd.BroadcastInt16x8(3)
		dLo := q0.InterleaveLo(zero).ReshapeToUint16s().BitsToInt16().Sub(p0.InterleaveLo(zero).ReshapeToUint16s().BitsToInt16())
		dHi := q0.InterleaveHi(zero).ReshapeToUint16s().BitsToInt16().Sub(p0.InterleaveHi(zero).ReshapeToUint16s().BitsToInt16())
		fLo := dLo.Mul(three16).Add(outerF.ExtendLo8ToInt16()).SaturateToInt8()
		fHi := dHi.Mul(three16).Add(outerF.HiToLo().ExtendLo8ToInt16()).SaturateToInt8()
		f := fLo.ToBits().ReshapeToUint64s().InterleaveLo(fHi.ToBits().ReshapeToUint64s()).ReshapeToUint8s().BitsToInt8()
		shift3 := archsimd.BroadcastInt8x16(-3)
		shift1 := archsimd.BroadcastInt8x16(-1)
		filter1 := f.AddSaturated(archsimd.BroadcastInt8x16(4)).Shift(shift3)
		filter2 := f.AddSaturated(archsimd.BroadcastInt8x16(3)).Shift(shift3)
		outer := filter1.Add(archsimd.BroadcastInt8x16(1)).Shift(shift1)
		o0 = p0c.AddSaturated(filter2).ToBits().Xor(bias).IfElse(mF4, p0)
		n0 = q0c.SubSaturated(filter1).ToBits().Xor(bias).IfElse(mF4, q0)
		mOuter := mF4.And(hev.Not())
		o1 = p1c.AddSaturated(outer).ToBits().Xor(bias).IfElse(mOuter, p1)
		n1 = q1c.SubSaturated(outer).ToBits().Xor(bias).IfElse(mOuter, q1)
	}
	if wd < 6 || !lpfAny(mFlat) {
		return
	}
	sh3 := archsimd.BroadcastInt16x8(-3)
	four := archsimd.BroadcastUint16x8(4)
	if wd == 6 {
		lp2, lp1, lp0 := p2.ExtendLo8ToUint16(), p1.ExtendLo8ToUint16(), p0.ExtendLo8ToUint16()
		lq0, lq1, lq2 := q0.ExtendLo8ToUint16(), q1.ExtendLo8ToUint16(), q2.ExtendLo8ToUint16()
		s := lp2.Add(lp2).Add(lp2).Add(lp1.Add(lp0).Add(lp1.Add(lp0))).Add(lq0).Add(four)
		ap1 := s
		s = s.Add(lq0.Add(lq1)).Sub(lp2.Add(lp2))
		ap0 := s
		s = s.Add(lq1.Add(lq2)).Sub(lp2.Add(lp1))
		aq0 := s
		s = s.Add(lq2.Add(lq2)).Sub(lp1.Add(lp0))
		aq1 := s
		hp2, hp1, hp0 := p2.HiToLo().ExtendLo8ToUint16(), p1.HiToLo().ExtendLo8ToUint16(), p0.HiToLo().ExtendLo8ToUint16()
		hq0, hq1, hq2 := q0.HiToLo().ExtendLo8ToUint16(), q1.HiToLo().ExtendLo8ToUint16(), q2.HiToLo().ExtendLo8ToUint16()
		s = hp2.Add(hp2).Add(hp2).Add(hp1.Add(hp0).Add(hp1.Add(hp0))).Add(hq0).Add(four)
		bp1 := s
		s = s.Add(hq0.Add(hq1)).Sub(hp2.Add(hp2))
		bp0 := s
		s = s.Add(hq1.Add(hq2)).Sub(hp2.Add(hp1))
		bq0 := s
		s = s.Add(hq2.Add(hq2)).Sub(hp1.Add(hp0))
		bq1 := s
		o1 = ap1.Shift(sh3).ReshapeToUint8s().ConcatEven(bp1.Shift(sh3).ReshapeToUint8s()).IfElse(mFlat, o1)
		o0 = ap0.Shift(sh3).ReshapeToUint8s().ConcatEven(bp0.Shift(sh3).ReshapeToUint8s()).IfElse(mFlat, o0)
		n0 = aq0.Shift(sh3).ReshapeToUint8s().ConcatEven(bq0.Shift(sh3).ReshapeToUint8s()).IfElse(mFlat, n0)
		n1 = aq1.Shift(sh3).ReshapeToUint8s().ConcatEven(bq1.Shift(sh3).ReshapeToUint8s()).IfElse(mFlat, n1)
		return
	}
	lp3, lp2, lp1, lp0 := p3.ExtendLo8ToUint16(), p2.ExtendLo8ToUint16(), p1.ExtendLo8ToUint16(), p0.ExtendLo8ToUint16()
	lq0, lq1, lq2, lq3 := q0.ExtendLo8ToUint16(), q1.ExtendLo8ToUint16(), q2.ExtendLo8ToUint16(), q3.ExtendLo8ToUint16()
	s := lp3.Add(lp3).Add(lp3).Add(lp2.Add(lp2)).Add(lp1.Add(lp0)).Add(lq0.Add(four))
	ap2 := s
	s = s.Add(lp1.Add(lq1)).Sub(lp3.Add(lp2))
	ap1 := s
	s = s.Add(lp0.Add(lq2)).Sub(lp3.Add(lp1))
	ap0 := s
	s = s.Add(lq0.Add(lq3)).Sub(lp3.Add(lp0))
	aq0 := s
	s = s.Add(lq1.Add(lq3)).Sub(lp2.Add(lq0))
	aq1 := s
	s = s.Add(lq2.Add(lq3)).Sub(lp1.Add(lq1))
	aq2 := s
	hp3, hp2, hp1, hp0 := p3.HiToLo().ExtendLo8ToUint16(), p2.HiToLo().ExtendLo8ToUint16(), p1.HiToLo().ExtendLo8ToUint16(), p0.HiToLo().ExtendLo8ToUint16()
	hq0, hq1, hq2, hq3 := q0.HiToLo().ExtendLo8ToUint16(), q1.HiToLo().ExtendLo8ToUint16(), q2.HiToLo().ExtendLo8ToUint16(), q3.HiToLo().ExtendLo8ToUint16()
	s = hp3.Add(hp3).Add(hp3).Add(hp2.Add(hp2)).Add(hp1.Add(hp0)).Add(hq0.Add(four))
	bp2 := s
	s = s.Add(hp1.Add(hq1)).Sub(hp3.Add(hp2))
	bp1 := s
	s = s.Add(hp0.Add(hq2)).Sub(hp3.Add(hp1))
	bp0 := s
	s = s.Add(hq0.Add(hq3)).Sub(hp3.Add(hp0))
	bq0 := s
	s = s.Add(hq1.Add(hq3)).Sub(hp2.Add(hq0))
	bq1 := s
	s = s.Add(hq2.Add(hq3)).Sub(hp1.Add(hq1))
	bq2 := s
	o2 = ap2.Shift(sh3).ReshapeToUint8s().ConcatEven(bp2.Shift(sh3).ReshapeToUint8s()).IfElse(mFlat, o2)
	o1 = ap1.Shift(sh3).ReshapeToUint8s().ConcatEven(bp1.Shift(sh3).ReshapeToUint8s()).IfElse(mFlat, o1)
	o0 = ap0.Shift(sh3).ReshapeToUint8s().ConcatEven(bp0.Shift(sh3).ReshapeToUint8s()).IfElse(mFlat, o0)
	n0 = aq0.Shift(sh3).ReshapeToUint8s().ConcatEven(bq0.Shift(sh3).ReshapeToUint8s()).IfElse(mFlat, n0)
	n1 = aq1.Shift(sh3).ReshapeToUint8s().ConcatEven(bq1.Shift(sh3).ReshapeToUint8s()).IfElse(mFlat, n1)
	n2 = aq2.Shift(sh3).ReshapeToUint8s().ConcatEven(bq2.Shift(sh3).ReshapeToUint8s()).IfElse(mFlat, n2)
	return
}

// lpfW4..lpfW16 select a width-specialized kernel instance, as dav1d's
// loop_filter wd macro emits lpf_16_wd{4,6,8,16}_neon from one source: each
// type's size is its filter width, so every stenciled body sees a constant
// width and drops the code of the other widths.
type (
	lpfW4  [4]byte
	lpfW6  [6]byte
	lpfW8  [8]byte
	lpfW16 [16]byte
)

type lpfWidth interface {
	lpfW4 | lpfW6 | lpfW8 | lpfW16
}

// lpf16U8 dispatches to the width-specialized lpf16U8W instance.
func lpf16U8(p unsafe.Pointer, stride int, wd int, eV, iV, hV archsimd.Uint8x16, m4, mGt4, m16 archsimd.Mask8x16) bool {
	switch wd {
	case 16:
		return lpf16U8W[lpfW16](p, stride, eV, iV, hV, m4, mGt4, m16)
	case 8:
		return lpf16U8W[lpfW8](p, stride, eV, iV, hV, m4, mGt4, m16)
	case 6:
		return lpf16U8W[lpfW6](p, stride, eV, iV, hV, m4, mGt4, m16)
	default:
		return lpf16U8W[lpfW4](p, stride, eV, iV, hV, m4, mGt4, m16)
	}
}
