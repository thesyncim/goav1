// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package loopfilter

import (
	"encoding/binary"
	"math/bits"
	"simd/archsimd"

	"github.com/thesyncim/goav1/internal/unsafeutil"
)

// The 8-bit drivers check each call's sample and level footprints once, as
// unsafeutil.Rows views, and hand every group's tap rows to the kernels as an
// unsafeutil.Block: the kernels name their rows by constant row numbers, so
// no tap access costs a check (dav1d trusts its caller the same way, with
// the check moved to the view). The transposes run through local arrays.
// The 16-bit drivers (sb_u16_arm64.go) work the same way.

// lpfMaxPlane bounds the planes the vector drivers take; larger planes
// (beyond any AV1 frame) take the scalar reference.
const lpfMaxPlane = 1 << 32

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

// lpf16U8Wd16 is dav1d's lpf_16_wd16_neon: it filters sixteen lanes whose
// taps are the fourteen rows p6..q6 of tv (q0 is row 7, the first row below
// the edge), and stores the changed rows back; the wd4, flat8in and flat8out
// lanes run in one body, each lane at its own width exactly as the C reference
// does. Widths up to 8 run lpf8U8W, which the transposed vertical path shares.
func lpf16U8Wd16(tv unsafeutil.Block[lpfR14], eV, iV, hV archsimd.Uint8x16, m4, mGt4, m16 archsimd.Mask8x16) bool {
	// Each row is named at its load and its store, on the line of the
	// access, so the accessor leaves no inline mark (CSE shares the address).
	p3 := archsimd.LoadUint8x16Array(tv.Row16(3))
	p2 := archsimd.LoadUint8x16Array(tv.Row16(4))
	p1 := archsimd.LoadUint8x16Array(tv.Row16(5))
	p0 := archsimd.LoadUint8x16Array(tv.Row16(6))
	q0 := archsimd.LoadUint8x16Array(tv.Row16(7))
	q1 := archsimd.LoadUint8x16Array(tv.Row16(8))
	q2 := archsimd.LoadUint8x16Array(tv.Row16(9))
	q3 := archsimd.LoadUint8x16Array(tv.Row16(10))
	maxd := lpfAbsDiff(p1, p0).Max(lpfAbsDiff(q1, q0))
	tt := lpfAbsDiff(p2, p1).Max(lpfAbsDiff(q2, q1)).Max(lpfAbsDiff(p3, p2).Max(lpfAbsDiff(q3, q2)))
	t := maxd.Max(tt.Masked(mGt4))
	d00 := lpfAbsDiff(p0, q0)
	sum := d00.AddSaturated(d00).AddSaturated(lpfAbsDiff(p1, q1).ShiftAllRight(1))
	fm := iV.GreaterEqual(t).And(eV.GreaterEqual(sum)).And(m4)
	if !lpfAny(fm) {
		return false
	}
	one := archsimd.BroadcastUint8x16(1)
	mFlat := one.GreaterEqual(lpfSpread(p2, p3, p0).Max(lpfSpread(q2, q3, q0)).Max(maxd)).And(mGt4).And(fm)
	mF4 := fm.And(mFlat.Not())
	p6 := archsimd.LoadUint8x16Array(tv.Row16(0))
	p5 := archsimd.LoadUint8x16Array(tv.Row16(1))
	p4 := archsimd.LoadUint8x16Array(tv.Row16(2))
	q4 := archsimd.LoadUint8x16Array(tv.Row16(11))
	q5 := archsimd.LoadUint8x16Array(tv.Row16(12))
	q6 := archsimd.LoadUint8x16Array(tv.Row16(13))
	fo := lpfSpread3(p4, p5, p6, p0).Max(lpfSpread3(q4, q5, q6, q0))
	mFlat16 := one.GreaterEqual(fo).And(m16).And(mFlat)
	mFlat = mFlat.And(mFlat16.Not())

	// Outputs; every filter reads the original taps, lanes are disjoint.
	o2, o1, o0, n0, n1, n2 := p2, p1, p0, q0, q1, q2
	if lpfAny(mF4) {
		// dav1d wd4 filter. archsimd has no USQADD, so the taps are biased to
		// signed as the C reference does; f = iclip_diff(3*(q0-p0) + (hev ?
		// iclip_diff(p1-q1) : 0)) is three saturating adds of the saturated
		// q0-p0 rather than dav1d's widened usubl/mul/saddw/sqxtn (exact: the
		// partial sums move monotonically toward the clamp, so saturating
		// early never changes the clamped total), and (f1+1)>>1 is a rounding
		// halving add with zero (srshr #1).
		bias := archsimd.BroadcastUint8x16(128)
		hev := maxd.Greater(hV)
		p1c := p1.Xor(bias).BitsToInt8()
		p0c := p0.Xor(bias).BitsToInt8()
		q0c := q0.Xor(bias).BitsToInt8()
		q1c := q1.Xor(bias).BitsToInt8()
		d := q0c.SubSaturated(p0c)
		f := p1c.SubSaturated(q1c).Masked(hev).AddSaturated(d).AddSaturated(d).AddSaturated(d)
		sh3 := archsimd.BroadcastInt8x16(-3)
		f1 := f.AddSaturated(archsimd.BroadcastInt8x16(4)).Shift(sh3)
		f2 := f.AddSaturated(archsimd.BroadcastInt8x16(3)).Shift(sh3)
		outer := f1.Average(archsimd.Int8x16{})
		o0 = p0c.AddSaturated(f2).ToBits().Xor(bias).IfElse(mF4, p0)
		n0 = q0c.SubSaturated(f1).ToBits().Xor(bias).IfElse(mF4, q0)
		mOuter := mF4.And(hev.Not())
		o1 = p1c.AddSaturated(outer).ToBits().Xor(bias).IfElse(mOuter, p1)
		n1 = q1c.SubSaturated(outer).ToBits().Xor(bias).IfElse(mOuter, q1)
	}
	if lpfAny(mFlat) {
		k, k2, k3 := archsimd.BroadcastUint8x16(32), archsimd.BroadcastUint8x16(64), archsimd.BroadcastUint8x16(96)
		bias8 := archsimd.BroadcastUint16x8(128)
		lP3, hP3 := p3.MulWidenLo(k), p3.HiToLo().MulWidenLo(k.HiToLo())
		lP2, hP2 := p2.MulWidenLo(k), p2.HiToLo().MulWidenLo(k.HiToLo())
		lP1, hP1 := p1.MulWidenLo(k), p1.HiToLo().MulWidenLo(k.HiToLo())
		lP0, hP0 := p0.MulWidenLo(k), p0.HiToLo().MulWidenLo(k.HiToLo())
		lQ0, hQ0 := q0.MulWidenLo(k), q0.HiToLo().MulWidenLo(k.HiToLo())
		lQ1, hQ1 := q1.MulWidenLo(k), q1.HiToLo().MulWidenLo(k.HiToLo())
		lQ2, hQ2 := q2.MulWidenLo(k), q2.HiToLo().MulWidenLo(k.HiToLo())
		lQ3, hQ3 := q3.MulWidenLo(k), q3.HiToLo().MulWidenLo(k.HiToLo())
		l3P3, h3P3 := p3.MulWidenLo(k3), p3.HiToLo().MulWidenLo(k3.HiToLo())
		l2P2, h2P2 := p2.MulWidenLo(k2), p2.HiToLo().MulWidenLo(k2.HiToLo())
		la2, la1, la0, lb0, lb1, lb2 := lpfFlat8Half(lP3, lP2, lP1, lP0, lQ0, lQ1, lQ2, lQ3, l3P3.Add(l2P2).Add(lP1.Add(lP0)).Add(lQ0.Add(bias8)))
		ha2, ha1, ha0, hb0, hb1, hb2 := lpfFlat8Half(hP3, hP2, hP1, hP0, hQ0, hQ1, hQ2, hQ3, h3P3.Add(h2P2).Add(hP1.Add(hP0)).Add(hQ0.Add(bias8)))
		o2 = la2.ReshapeToUint8s().ConcatOdd(ha2.ReshapeToUint8s()).IfElse(mFlat, o2)
		o1 = la1.ReshapeToUint8s().ConcatOdd(ha1.ReshapeToUint8s()).IfElse(mFlat, o1)
		o0 = la0.ReshapeToUint8s().ConcatOdd(ha0.ReshapeToUint8s()).IfElse(mFlat, o0)
		n0 = lb0.ReshapeToUint8s().ConcatOdd(hb0.ReshapeToUint8s()).IfElse(mFlat, n0)
		n1 = lb1.ReshapeToUint8s().ConcatOdd(hb1.ReshapeToUint8s()).IfElse(mFlat, n1)
		n2 = lb2.ReshapeToUint8s().ConcatOdd(hb2.ReshapeToUint8s()).IfElse(mFlat, n2)
	}
	if lpfAny(mFlat16) {
		// The flat16 lanes finish in their own frame, which reloads the
		// (still unmodified) taps and stores every output row, so nothing of
		// this frame stays live across the call.
		lpfFlat16U8Store(tv, mFlat16, o2, o1, o0, n0, n1, n2)
		return true
	}
	o2.StoreArray(tv.Row16(4))
	n2.StoreArray(tv.Row16(9))
	o1.StoreArray(tv.Row16(5))
	o0.StoreArray(tv.Row16(6))
	n0.StoreArray(tv.Row16(7))
	n1.StoreArray(tv.Row16(8))
	return true
}

// lpfFlat16U8Store is lpfFlat16U8 for lpf16U8Wd16: it loads the taps p6..q6
// from tv (rows 0..13, not yet written), blends the 13-tap outputs on the
// lanes in m into the caller's p2..q2 outputs (o2..n2) and stores the twelve
// rows p5..q5.
func lpfFlat16U8Store(tv unsafeutil.Block[lpfR14], m archsimd.Mask8x16, o2, o1, o0, n0, n1, n2 archsimd.Uint8x16) {
	p6 := archsimd.LoadUint8x16Array(tv.Row16(0))
	p5 := archsimd.LoadUint8x16Array(tv.Row16(1))
	p4 := archsimd.LoadUint8x16Array(tv.Row16(2))
	p3 := archsimd.LoadUint8x16Array(tv.Row16(3))
	p2 := archsimd.LoadUint8x16Array(tv.Row16(4))
	p1 := archsimd.LoadUint8x16Array(tv.Row16(5))
	p0 := archsimd.LoadUint8x16Array(tv.Row16(6))
	q0 := archsimd.LoadUint8x16Array(tv.Row16(7))
	q1 := archsimd.LoadUint8x16Array(tv.Row16(8))
	q2 := archsimd.LoadUint8x16Array(tv.Row16(9))
	q3 := archsimd.LoadUint8x16Array(tv.Row16(10))
	q4 := archsimd.LoadUint8x16Array(tv.Row16(11))
	q5 := archsimd.LoadUint8x16Array(tv.Row16(12))
	q6 := archsimd.LoadUint8x16Array(tv.Row16(13))
	r5, r4, r3, s3, s4, s5 := lpfFlat16U8(p6, p5, p4, p3, p2, p1, p0, q0, q1, q2, q3, q4, q5, q6, m, &o2, &o1, &o0, &n0, &n1, &n2)
	r5.StoreArray(tv.Row16(1))
	r4.StoreArray(tv.Row16(2))
	r3.StoreArray(tv.Row16(3))
	o2.StoreArray(tv.Row16(4))
	o1.StoreArray(tv.Row16(5))
	o0.StoreArray(tv.Row16(6))
	n0.StoreArray(tv.Row16(7))
	n1.StoreArray(tv.Row16(8))
	n2.StoreArray(tv.Row16(9))
	s3.StoreArray(tv.Row16(10))
	s4.StoreArray(tv.Row16(11))
	s5.StoreArray(tv.Row16(12))
}

// lpfFlat16U8 computes the 13-tap flat filter (dav1d wd16, AV1 filter14) on
// the lanes in m and blends p2..q2 into the caller's outputs; it returns the
// outer p5..p3 / q3..q5 rows (unchanged outside m). The halves run on lanes
// widened and scaled by 16, so each output byte is the high byte of its
// biased sum.
func lpfFlat16U8(p6, p5, p4, p3, p2, p1, p0, q0, q1, q2, q3, q4, q5, q6 archsimd.Uint8x16, m archsimd.Mask8x16, o2, o1, o0, n0, n1, n2 *archsimd.Uint8x16) (r5, r4, r3, s3, s4, s5 archsimd.Uint8x16) {
	k := archsimd.BroadcastUint8x16(16)
	k7 := archsimd.BroadcastUint8x16(112)
	var lo, hi [12]archsimd.Uint16x8
	lpfFlat16Half(p6.MulWidenLo(k7), p6.MulWidenLo(k), p5.MulWidenLo(k), p4.MulWidenLo(k), p3.MulWidenLo(k),
		p2.MulWidenLo(k), p1.MulWidenLo(k), p0.MulWidenLo(k), q0.MulWidenLo(k),
		q1.MulWidenLo(k), q2.MulWidenLo(k), q3.MulWidenLo(k), q4.MulWidenLo(k),
		q5.MulWidenLo(k), q6.MulWidenLo(k), &lo)
	kh, k7h := k.HiToLo(), k7.HiToLo()
	lpfFlat16Half(p6.HiToLo().MulWidenLo(k7h), p6.HiToLo().MulWidenLo(kh), p5.HiToLo().MulWidenLo(kh), p4.HiToLo().MulWidenLo(kh), p3.HiToLo().MulWidenLo(kh),
		p2.HiToLo().MulWidenLo(kh), p1.HiToLo().MulWidenLo(kh), p0.HiToLo().MulWidenLo(kh), q0.HiToLo().MulWidenLo(kh),
		q1.HiToLo().MulWidenLo(kh), q2.HiToLo().MulWidenLo(kh), q3.HiToLo().MulWidenLo(kh), q4.HiToLo().MulWidenLo(kh),
		q5.HiToLo().MulWidenLo(kh), q6.HiToLo().MulWidenLo(kh), &hi)
	r5 = lo[0].ReshapeToUint8s().ConcatOdd(hi[0].ReshapeToUint8s()).IfElse(m, p5)
	r4 = lo[1].ReshapeToUint8s().ConcatOdd(hi[1].ReshapeToUint8s()).IfElse(m, p4)
	r3 = lo[2].ReshapeToUint8s().ConcatOdd(hi[2].ReshapeToUint8s()).IfElse(m, p3)
	*o2 = lo[3].ReshapeToUint8s().ConcatOdd(hi[3].ReshapeToUint8s()).IfElse(m, *o2)
	*o1 = lo[4].ReshapeToUint8s().ConcatOdd(hi[4].ReshapeToUint8s()).IfElse(m, *o1)
	*o0 = lo[5].ReshapeToUint8s().ConcatOdd(hi[5].ReshapeToUint8s()).IfElse(m, *o0)
	*n0 = lo[6].ReshapeToUint8s().ConcatOdd(hi[6].ReshapeToUint8s()).IfElse(m, *n0)
	*n1 = lo[7].ReshapeToUint8s().ConcatOdd(hi[7].ReshapeToUint8s()).IfElse(m, *n1)
	*n2 = lo[8].ReshapeToUint8s().ConcatOdd(hi[8].ReshapeToUint8s()).IfElse(m, *n2)
	s3 = lo[9].ReshapeToUint8s().ConcatOdd(hi[9].ReshapeToUint8s()).IfElse(m, q3)
	s4 = lo[10].ReshapeToUint8s().ConcatOdd(hi[10].ReshapeToUint8s()).IfElse(m, q4)
	s5 = lo[11].ReshapeToUint8s().ConcatOdd(hi[11].ReshapeToUint8s()).IfElse(m, q5)
	return
}

// lpfFlat16Half runs the dav1d sliding-sum form of the 13-tap filter on eight
// widened lanes scaled by 16 (p6x7 is 7*p6 at that scale); out[k] holds the
// biased sum of output p5+k (p5..q5).
func lpfFlat16Half(p6x7, p6, p5, p4, p3, p2, p1, p0, q0, q1, q2, q3, q4, q5, q6 archsimd.Uint16x8, out *[12]archsimd.Uint16x8) {
	p6x2 := p6.Add(p6)
	s := p6x7.Add(p5.Add(p5)).Add(p4.Add(p4)).Add(p3.Add(p2)).Add(p1.Add(p0)).Add(q0.Add(archsimd.BroadcastUint16x8(128)))
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

// sbLowByteLanes keeps the low byte of each 32-bit lane: dav1d lpf_func's
// movi v2.4s, #0xff, as one table load instead of a GPR-to-vector broadcast.
var sbLowByteLanes = [4]uint32{0xff, 0xff, 0xff, 0xff}

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
	return archsimd.Int8x16{}.Greater(archsimd.LoadUint8x16Array(&sbNibbleLanes[n&15]).BitsToInt8())
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
func sbGroupLimits(cur, prev archsimd.Uint32x4, nAll uint32, sharp0 uint64, sharp1 uint8) (eV, iV, hV archsimd.Uint8x16, m4 archsimd.Mask8x16, ok bool) {
	zero32 := archsimd.Uint32x4{}
	L := prev.IfElse(cur.Equal(zero32), cur)
	lr := L.Mul(archsimd.BroadcastUint32x4(0x01010101)).ReshapeToUint8s()
	zero := archsimd.Uint8x16{}
	act := archsimd.LoadUint8x16Array(&sbNibbleLanes[nAll&15]).AndNot(lr.Equal(zero).ToInt8x16().ToBits())
	h := act.ReshapeToUint64s()
	if h.GetElem(0)|h.GetElem(1) == 0 {
		return
	}
	hV = lr.ShiftAllRight(4)
	iV = lr.ShiftAllRight(sharp0).Min(archsimd.BroadcastUint8x16(sharp1)).Max(archsimd.BroadcastUint8x16(1))
	e := lr.Add(archsimd.BroadcastUint8x16(2))
	eV = e.Add(e).Add(iV)
	m4 = archsimd.Int8x16{}.Greater(act.BitsToInt8())
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

// The level map holds four bytes per 4x4 cell (lfmask.LevelCache), so a
// byte cursor li = 4*cell + comp reads component comp of a cell; loads of
// whole cells keep comp in the low byte of each 32-bit lane after an AND with
// sbLowByteLanes, as dav1d's lpf_func does.

// sbLevels4 is component comp of the four cells from the byte cursor li, one
// per 32-bit lane (dav1d's ld1 of l[0]): one sixteen-byte load.
func sbLevels4(lvl []uint8, li int) archsimd.Uint32x4 {
	return archsimd.LoadUint8x16Array((*[16]uint8)(lvl[li : li+16])).ReshapeToUint32s().And(archsimd.LoadUint32x4Array(&sbLowByteLanes))
}

// sbLevelsPairs is component comp of the (l[-1], l[0]) cell pairs of four
// units (dav1d's ld2 lane loads): rows r0..r0+3 of lv, eight bytes each, a
// view whose row k starts at the cell before unit k's at the component offset.
// cur holds each unit's l[0], prev its l[-1]; a block outside lv reads zero
// levels (no unit filters).
func sbLevelsPairs(lv unsafeutil.Rows, r0 int) (cur, prev archsimd.Uint32x4) {
	tv := lv.Sub(r0, 0, 8, 4)
	return sbPairs(binary.LittleEndian.Uint64(tv.Row8(0, 0)[:]), binary.LittleEndian.Uint64(tv.Row8(1, 0)[:]),
		binary.LittleEndian.Uint64(tv.Row8(2, 0)[:]), binary.LittleEndian.Uint64(tv.Row8(3, 0)[:]))
}

// sbPairs splits four units' (l[-1], l[0]) pairs, one unit per uint64, into
// the l[0] lanes (cur) and the l[-1] lanes (prev), component in the low byte.
func sbPairs(x0, x1, y0, y1 uint64) (cur, prev archsimd.Uint32x4) {
	x := [2]uint64{x0, x1}
	y := [2]uint64{y0, y1}
	a := archsimd.LoadUint64x2Array(&x).ReshapeToUint32s()
	b := archsimd.LoadUint64x2Array(&y).ReshapeToUint32s()
	ff := archsimd.LoadUint32x4Array(&sbLowByteLanes)
	return a.ConcatOdd(b).And(ff), a.ConcatEven(b).And(ff)
}

// sbLevelsRow loads the levels of four consecutive units (horizontal edges:
// dav1d lpf_v_sb ld1 of l[0] and l[-b4_stride]) with component comp moved to
// the low byte of each lane.
func sbLevelsRow(lvl []uint8, u0, lvlStride, comp int, nAll uint32) (cur, prev archsimd.Uint32x4) {
	if li := 4*u0 + comp; u0-lvlStride >= 0 && li+16 <= len(lvl) {
		return sbLevels4(lvl, li), sbLevels4(lvl, li-4*lvlStride)
	}
	var c, p [4]uint32
	for k := 0; k < 4; k++ {
		if nAll&(1<<uint(k)) != 0 {
			c[k] = uint32(lvl[4*(u0+k)+comp])
			p[k] = uint32(lvl[4*(u0+k-lvlStride)+comp])
		}
	}
	return archsimd.LoadUint32x4Array(&c), archsimd.LoadUint32x4Array(&p)
}

// sbLevelsCol loads the levels of four units stacked lvlStride cells apart
// (vertical edges: dav1d lpf_h_sb ld2 of the l[-1], l[0] pairs) with
// component comp moved to the low byte of each lane.
func sbLevelsCol(lvl []uint8, li, lvlStride, comp int, nAll uint32) (cur, prev archsimd.Uint32x4) {
	if unsafeutil.RowsFit(len(lvl), 4*li+comp-4, 4*lvlStride, 8, 4) {
		return sbLevelsPairs(unsafeutil.NewRows(lvl, 4*li+comp-4, 4*lvlStride, 8, 4), 0)
	}
	var c, p [4]uint32
	for k := 0; k < 4; k++ {
		if nAll&(1<<uint(k)) != 0 {
			c[k] = uint32(lvl[4*(li+k*lvlStride)+comp])
			p[k] = uint32(lvl[4*(li+k*lvlStride-1)+comp])
		}
	}
	return archsimd.LoadUint32x4Array(&c), archsimd.LoadUint32x4Array(&p)
}

// filterSBHorizontal8Edge is the group-by-group driver for calls whose
// samples or levels reach the end of their buffers: each group checks its own
// reach and falls back to the unit-by-unit reference when it would leave pix.
func filterSBHorizontal8Edge(pix []byte, off, stride, rowBytes int, vm [3]uint32, luma bool, lvl []uint8, lvlOff, lvlStride, comp int, lut *SBLUT) {
	gt4All := vm[1] | vm[2]
	all := vm[0] | gt4All
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
		// The group's tap rows: the wd/2 rows each side of the edge (seven
		// at wd 16), sixteen bytes each.
		rh := wd / 2
		if wd == 16 {
			rh = 7
		}
		if 16*g+16 > rowBytes || !unsafeutil.RowsFit(len(pix), q0Off-rh*stride, stride, 16, 2*rh) {
			sub := vm
			for k := range sub {
				sub[k] = (sub[k] >> sh) & 0xf
			}
			filterSBHorizontal8Scalar(pix, q0Off, stride, sub, luma, lvl, u0, lvlStride, comp, lut)
			continue
		}
		cur, prev := sbLevelsRow(lvl, u0, lvlStride, comp, nib)
		eV, iV, hV, m4, ok := sbGroupLimits(cur, prev, nib, sharp0, sharp1)
		if !ok {
			continue
		}
		lpfRowsU8(unsafeutil.NewRows(pix, q0Off-rh*stride, stride, 16, 2*rh), wd, eV, iV, hV, m4, sbNibbleMask(nGt4), sbNibbleMask(n16))
	}
}

// filterSBVertical8Edge is filterSBHorizontal8Edge for vertical edges.
func filterSBVertical8Edge(pix []byte, off, stride, rowBytes int, vm [3]uint32, luma bool, lvl []uint8, lvlOff, lvlStride, comp int, lut *SBLUT) {
	gt4All := vm[1] | vm[2]
	all := vm[0] | gt4All
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
		// The group's sixteen rows: p7..q7 (sixteen bytes) at wd 16, else
		// p3..q3 read sixteen bytes at a time when the row has twelve bytes
		// right of the edge, or eight bytes at a time.
		var gv unsafeutil.Rows
		inPix, wide := false, false
		switch {
		case wd == 16:
			if inPix = unsafeutil.RowsFit(len(pix), q0Off-8, stride, 16, 16); inPix {
				gv = unsafeutil.NewRows(pix, q0Off-8, stride, 16, 16)
			}
		case rowBytes >= 12 && unsafeutil.RowsFit(len(pix), q0Off-4, stride, 16, 16):
			gv, inPix, wide = unsafeutil.NewRows(pix, q0Off-4, stride, 16, 16), true, true
		default:
			if inPix = unsafeutil.RowsFit(len(pix), q0Off-4, stride, 8, 16); inPix {
				gv = unsafeutil.NewRows(pix, q0Off-4, stride, 8, 16)
			}
		}
		if !inPix {
			sbVerticalScalarGroup(pix, q0Off, stride, vm, sh, luma, lvl, u0, lvlStride, comp, lut)
			continue
		}
		cur, prev := sbLevelsCol(lvl, u0, lvlStride, comp, nib)
		eV, iV, hV, m4, ok := sbGroupLimits(cur, prev, nib, sharp0, sharp1)
		if !ok {
			continue
		}
		if wd == 16 {
			lpfVertical16x16U8(unsafeutil.BlockAt[lpfR16](gv, 0, 0), eV, iV, hV, m4, sbNibbleMask(nGt4), sbNibbleMask(n16))
		} else if wide {
			mGt4 := sbNibbleMask(nGt4)
			gb := unsafeutil.BlockAt[lpfR16](gv, 0, 0)
			switch wd {
			case 8:
				lpfVertical16x8U8WideW[lpfW8](gb, eV, iV, hV, m4, mGt4)
			case 6:
				lpfVertical16x8U8WideW[lpfW6](gb, eV, iV, hV, m4, mGt4)
			default:
				lpfVertical16x8U8WideW[lpfW4](gb, eV, iV, hV, m4, mGt4)
			}
		} else {
			lpfVertical16x8U8(gv, 0, 0, wd, eV, iV, hV, m4, sbNibbleMask(nGt4))
		}
	}
}

// sbActive and sbLimitsOf are the per-group level prologue of dav1d lpf_func
// (src/arm/64/loopfilter.S), one unit per 32-bit lane: cur and prev hold each
// unit's own and fallback (left or above) level in the low byte of its lane
// with the other bytes zero. L = cur ? cur : prev is replicated over the
// unit's four bytes (lr) by a multiply with the bytes-of-one vector that also
// floors I; I = imax(imin(L >> sharp[0], sharp[1]), 1) and E = 2*(L+2)+I come
// from the LUT's replicated sharp vectors, H = L>>4.
//
// sbActive is the first half: lr and the active lanes act ("wd >= 4"
// restricted to L != 0), and whether any lane is active ("if (!L)
// continue"). The drivers test it before sbLimitsOf builds the limits, so a
// skipped group carries no zeroed limit vectors.
func sbActive(cur, prev archsimd.Uint32x4, nib uint32) (lr, act archsimd.Uint8x16, ok bool) {
	L := prev.IfElse(cur.Equal(archsimd.Uint32x4{}), cur)
	lr = L.Mul(archsimd.BroadcastUint8x16(1).ReshapeToUint32s()).ReshapeToUint8s()
	act = archsimd.LoadUint8x16Array(&sbNibbleLanes[nib&15]).AndNot(lr.Equal(archsimd.Uint8x16{}).ToInt8x16().ToBits())
	h := act.ReshapeToUint64s()
	return lr, act, h.GetElem(0)|h.GetElem(1) != 0
}

// sbLimitsOf is sbLimits' second half: E, I and H from lr, and the wd >= 4
// lane mask from act.
func sbLimitsOf(lr, act archsimd.Uint8x16, lut *SBLUT) (eV, iV, hV archsimd.Uint8x16, m4 archsimd.Mask8x16) {
	hV = lr.ShiftAllRight(4)
	iV = lr.Shift(archsimd.LoadInt8x16Array(&lut.SharpShift)).Min(archsimd.LoadUint8x16Array(&lut.SharpCap)).Max(archsimd.BroadcastUint8x16(1))
	e := lr.Add(archsimd.BroadcastUint8x16(2))
	eV = e.Add(e).Add(iV)
	m4 = archsimd.Int8x16{}.Greater(act.BitsToInt8())
	return
}

// filterSBHorizontal8 is dav1d lpf_v_sb_{y,uv}_8bpc_neon: the three width
// masks shift down one four-unit nibble per group while the sample and level
// cursors advance, each group's levels are one load of l[0] and one of
// l[-b4_stride] from a level cursor already pointing at component comp (the
// C side's &lvl[x][comp]), and the widest set nibble picks the kernel. The
// call's reach is checked once up front, as dav1d trusts its caller; calls
// that touch the end of pix, the row or the level map take the edge driver.
func filterSBHorizontal8(pix []byte, off, stride, rowBytes int, vm [3]uint32, luma bool, lvl []uint8, lvlOff, lvlStride, comp int, lut *SBLUT) {
	vm0, vm1, vm2 := vm[0]|vm[1]|vm[2], vm[1]|vm[2], vm[2]
	end := 16 * ((31-bits.LeadingZeros32(vm0))>>2 + 1)
	li, ls4 := 4*lvlOff+comp, 4*lvlStride
	if len(pix) > lpfMaxPlane {
		filterSBHorizontal8Scalar(pix, off, stride, vm, luma, lvl, lvlOff, lvlStride, comp, lut)
		return
	}
	if end > rowBytes {
		filterSBHorizontal8Edge(pix, off, stride, rowBytes, vm, luma, lvl, lvlOff, lvlStride, comp, lut)
		return
	}
	var ok bool
	switch {
	case vm2 != 0:
		ok = sbRowsU8[lpfW16](pix, off, stride, end, vm0, vm1, vm2, luma, lvl, li, ls4, lut)
	case vm1 != 0 && luma:
		ok = sbRowsU8[lpfW8](pix, off, stride, end, vm0, vm1, 0, luma, lvl, li, ls4, lut)
	case vm1 != 0:
		ok = sbRowsU8[lpfW6](pix, off, stride, end, vm0, vm1, 0, luma, lvl, li, ls4, lut)
	default:
		ok = sbRowsU8[lpfW4](pix, off, stride, end, vm0, 0, 0, luma, lvl, li, ls4, lut)
	}
	if !ok {
		filterSBHorizontal8Edge(pix, off, stride, rowBytes, vm, luma, lvl, lvlOff, lvlStride, comp, lut)
	}
}

// sbRowsU8 is filterSBHorizontal8's group loop for calls whose widest group
// is W: one view of the call's tap rows (the W/2 rows each side of the edge at
// off, seven at wd 16, n bytes each) and one of its two level rows (l[0] at
// li and l[-b4_stride] ls4 bytes before it) replace the call's reach checks;
// every group's sixteen-byte (four-cell) block is then a column of those
// views, and the narrower kernels of a wider call take their inner rows. It
// reports false, having filtered nothing, when either view leaves its slice.
func sbRowsU8[W lpfWidth](pix []byte, off, stride, n int, vm0, vm1, vm2 uint32, luma bool, lvl []uint8, li, ls4 int, lut *SBLUT) bool {
	var wdT W
	wd := len(wdT)
	rh := wd / 2
	if wd == 16 {
		rh = 7
	}
	if !unsafeutil.RowsFit(len(pix), off-rh*stride, stride, n, 2*rh) || !unsafeutil.RowsFit(len(lvl), li-ls4, ls4, n, 2) {
		return false
	}
	v := unsafeutil.NewRows(pix, off-rh*stride, stride, n, 2*rh)
	lv := unsafeutil.NewRows(lvl, li-ls4, ls4, n, 2)
	ff := archsimd.LoadUint32x4Array(&sbLowByteLanes)
	maxX := v.Width() - 16
	for x := 0; vm0 != 0 && x <= maxX; {
		if vm0&15 == 0 {
			skip := bits.TrailingZeros32(vm0) >> 2
			sh := uint(skip) << 2
			vm0, vm1, vm2, x = vm0>>sh, vm1>>sh, vm2>>sh, x+skip<<4
			if x > maxX {
				break
			}
		}
		// The level block is named at each load (no inline mark of its own).
		cur := archsimd.LoadUint8x16Array(unsafeutil.BlockAt[lpfR2](lv, 0, x).Row16(1)).ReshapeToUint32s().And(ff)
		prev := archsimd.LoadUint8x16Array(unsafeutil.BlockAt[lpfR2](lv, 0, x).Row16(0)).ReshapeToUint32s().And(ff)
		// The helpers share their lines with the tests, so they leave no
		// inline marks.
		var lr, act archsimd.Uint8x16
		var any bool
		if lr, act, any = sbActive(cur, prev, vm0); !any {
			vm0, vm1, vm2, x = vm0>>4, vm1>>4, vm2>>4, x+16
			continue
		}
		if eV, iV, hV, m4 := sbLimitsOf(lr, act, lut); wd == 16 && vm2&15 != 0 {
			lpf16U8Wd16(unsafeutil.BlockAt[lpfR14](v, 0, x), eV, iV, hV, m4, sbNibbleMask(vm1), sbNibbleMask(vm2))
		} else if wd >= 6 && vm1&15 != 0 {
			if luma {
				lpf8U8W[lpfW8, lpfRows](unsafeutil.BlockAt[lpfR8](v, rh-4, x), eV, iV, hV, m4, sbNibbleMask(vm1))
			} else {
				lpf8U8W[lpfW6, lpfRows](unsafeutil.BlockAt[lpfR6](v, rh-3, x), eV, iV, hV, m4, sbNibbleMask(vm1))
			}
		} else {
			lpf8U8W[lpfW4, lpfRows](unsafeutil.BlockAt[lpfR4](v, rh-2, x), eV, iV, hV, m4, archsimd.Mask8x16{})
		}
		vm0, vm1, vm2, x = vm0>>4, vm1>>4, vm2>>4, x+16
	}
	return true
}

// lpfRowsU8 is the per-group kernel choice of filterSBHorizontal8 for the
// edge driver: the width-wd row kernel on the tap rows of gv (the wd/2 rows
// each side of the edge, seven at wd 16).
func lpfRowsU8(gv unsafeutil.Rows, wd int, eV, iV, hV archsimd.Uint8x16, m4, mGt4, m16 archsimd.Mask8x16) {
	switch wd {
	case 16:
		lpf16U8Wd16(unsafeutil.BlockAt[lpfR14](gv, 0, 0), eV, iV, hV, m4, mGt4, m16)
	case 8:
		lpf8U8W[lpfW8, lpfRows](unsafeutil.BlockAt[lpfR8](gv, 0, 0), eV, iV, hV, m4, mGt4)
	case 6:
		lpf8U8W[lpfW6, lpfRows](unsafeutil.BlockAt[lpfR6](gv, 0, 0), eV, iV, hV, m4, mGt4)
	default:
		lpf8U8W[lpfW4, lpfRows](unsafeutil.BlockAt[lpfR4](gv, 0, 0), eV, iV, hV, m4, mGt4)
	}
}

// filterSBVertical8 is dav1d lpf_h_sb_{y,uv}_8bpc_neon in the same shape:
// each group reads the (l[-1], l[0]) level pairs of its four units (the ld2
// lane loads) from the component-offset cursor.
func filterSBVertical8(pix []byte, off, stride, rowBytes int, vm [3]uint32, luma bool, lvl []uint8, lvlOff, lvlStride, comp int, lut *SBLUT) {
	vm0, vm1, vm2 := vm[0]|vm[1]|vm[2], vm[1]|vm[2], vm[2]
	hiG := (31 - bits.LeadingZeros32(vm0)) >> 2
	reach := 4
	if vm2 != 0 {
		reach = 8
	}
	li, ls4 := 4*lvlOff+comp, 4*lvlStride
	if len(pix) > lpfMaxPlane {
		filterSBVertical8Scalar(pix, off, stride, vm, luma, lvl, lvlOff, lvlStride, comp, lut)
		return
	}
	// One view of the call's samples (its 16*(hiG+1) rows, reach bytes left
	// of the edge to twelve right of it) and one of its level pairs (the
	// (l[-1], l[0]) cell pairs of its 4*(hiG+1) units) replace the reach
	// checks; each group is then a sixteen-row block of the sample view and a
	// four-row block of the level view.
	if rowBytes < 12 || !unsafeutil.RowsFit(len(pix), off-reach, stride, reach+12, 16*hiG+16) || !unsafeutil.RowsFit(len(lvl), li-4, ls4, 8, 4*hiG+4) {
		filterSBVertical8Edge(pix, off, stride, rowBytes, vm, luma, lvl, lvlOff, lvlStride, comp, lut)
		return
	}
	v := unsafeutil.NewRows(pix, off-reach, stride, reach+12, 16*hiG+16)
	lv := unsafeutil.NewRows(lvl, li-4, ls4, 8, 4*hiG+4)
	c16, c8 := reach-8, reach-4 // p7 and p3 columns of the view
	for g := 0; vm0 != 0; {
		if vm0&15 == 0 {
			skip := bits.TrailingZeros32(vm0) >> 2
			sh := uint(skip) << 2
			vm0, vm1, vm2, g = vm0>>sh, vm1>>sh, vm2>>sh, g+skip
		}
		// sbLevelsPairs written out, each helper on the line of an
		// instruction of this function (no inline marks): the group's four
		// (l[-1], l[0]) pairs go to two vectors through the array stores.
		r4 := 4 * g
		l0 := [2]uint64{binary.LittleEndian.Uint64(lv.Row8(r4, 0)[:]), binary.LittleEndian.Uint64(lv.Row8(r4+1, 0)[:])}
		l1 := [2]uint64{binary.LittleEndian.Uint64(lv.Row8(r4+2, 0)[:]), binary.LittleEndian.Uint64(lv.Row8(r4+3, 0)[:])}
		la := archsimd.LoadUint64x2Array(&l0).ReshapeToUint32s()
		lb := archsimd.LoadUint64x2Array(&l1).ReshapeToUint32s()
		ff := archsimd.LoadUint32x4Array(&sbLowByteLanes)
		var lr, act archsimd.Uint8x16
		var any bool
		if lr, act, any = sbActive(la.ConcatOdd(lb).And(ff), la.ConcatEven(lb).And(ff), vm0); !any {
			vm0, vm1, vm2, g = vm0>>4, vm1>>4, vm2>>4, g+1
			continue
		}
		if eV, iV, hV, m4 := sbLimitsOf(lr, act, lut); vm2&15 != 0 {
			lpfVertical16x16U8(unsafeutil.BlockAt[lpfR16](v, 16*g, c16), eV, iV, hV, m4, sbNibbleMask(vm1), sbNibbleMask(vm2))
		} else if vm1&15 == 0 {
			lpfVertical16x8U8WideW[lpfW4](unsafeutil.BlockAt[lpfR16](v, 16*g, c8), eV, iV, hV, m4, archsimd.Mask8x16{})
		} else if luma {
			lpfVertical16x8U8WideW[lpfW8](unsafeutil.BlockAt[lpfR16](v, 16*g, c8), eV, iV, hV, m4, sbNibbleMask(vm1))
		} else {
			lpfVertical16x8U8WideW[lpfW6](unsafeutil.BlockAt[lpfR16](v, 16*g, c8), eV, iV, hV, m4, sbNibbleMask(vm1))
		}
		vm0, vm1, vm2, g = vm0>>4, vm1>>4, vm2>>4, g+1
	}
}

func sbVerticalScalarGroup(pix []byte, q0Off, stride int, vm [3]uint32, sh uint, luma bool, lvl []uint8, u0, lvlStride, comp int, lut *SBLUT) {
	sub := vm
	for k := range sub {
		sub[k] = (sub[k] >> sh) & 0xf
	}
	filterSBVertical8Scalar(pix, q0Off, stride, sub, luma, lvl, u0, lvlStride, comp, lut)
}

// lpfVertical16x8U8 filters a vertical edge across sixteen rows using eight
// bytes per row (p3..q3): the sixteen rows of v from row r0, column c0 (p3).
// The rows are transposed into tap vectors, filtered, and transposed back. A
// block outside v filters nothing.
func lpfVertical16x8U8(v unsafeutil.Rows, r0, c0 int, wd int, eV, iV, hV archsimd.Uint8x16, m4, mGt4 archsimd.Mask8x16) {
	tv := v.Sub(r0, c0, 8, 16)
	var buf [128]byte
	// Pair row r (low half) with row r+8 (high half), then an 8x8 byte
	// transpose of both halves at once yields one vector per tap.
	var pr [8][2]uint64
	for r := 0; r < 8; r++ {
		pr[r][0] = binary.LittleEndian.Uint64(tv.Row8(r, 0)[:])
		pr[r][1] = binary.LittleEndian.Uint64(tv.Row8(r+8, 0)[:])
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
	t0.StoreArray((*[16]byte)(buf[0:16]))
	t1.StoreArray((*[16]byte)(buf[16:32]))
	t2.StoreArray((*[16]byte)(buf[32:48]))
	t3.StoreArray((*[16]byte)(buf[48:64]))
	t4.StoreArray((*[16]byte)(buf[64:80]))
	t5.StoreArray((*[16]byte)(buf[80:96]))
	t6.StoreArray((*[16]byte)(buf[96:112]))
	t7.StoreArray((*[16]byte)(buf[112:128]))
	bv := unsafeutil.NewRows(buf[:], 0, 16, 16, 8)
	if !lpf8U8(wd, bv, 0, 0, eV, iV, hV, m4, mGt4) {
		return
	}
	a = archsimd.LoadUint8x16Array((*[16]byte)(buf[0:16]))
	b = archsimd.LoadUint8x16Array((*[16]byte)(buf[16:32]))
	c = archsimd.LoadUint8x16Array((*[16]byte)(buf[32:48]))
	d = archsimd.LoadUint8x16Array((*[16]byte)(buf[48:64]))
	e = archsimd.LoadUint8x16Array((*[16]byte)(buf[64:80]))
	f = archsimd.LoadUint8x16Array((*[16]byte)(buf[80:96]))
	g = archsimd.LoadUint8x16Array((*[16]byte)(buf[96:112]))
	h = archsimd.LoadUint8x16Array((*[16]byte)(buf[112:128]))
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
		binary.LittleEndian.PutUint64(tv.Row8(r, 0)[:], pr[r][0])
		binary.LittleEndian.PutUint64(tv.Row8(r+8, 0)[:], pr[r][1])
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

// lpfVertical16x16U8 is the wd16 variant: sixteen bytes per row (p7..q7),
// the sixteen rows of tv, a full 16x16 transpose each way through buf, where
// the row kernel sees taps p6..q6 as buffer rows 1..14.
func lpfVertical16x16U8(tv unsafeutil.Block[lpfR16], eV, iV, hV archsimd.Uint8x16, m4, mGt4, m16 archsimd.Mask8x16) {
	var buf [256]byte
	lpfTranspose16x16FromPix(tv, &buf)
	bv := unsafeutil.NewRows(buf[:], 0, 16, 16, 16)
	if !lpf16U8Wd16(unsafeutil.BlockAt[lpfR14](bv, 1, 0), eV, iV, hV, m4, mGt4, m16) {
		return
	}
	lpfTranspose16x16ToPix(&buf, tv)
}

// lpfR4..lpfR16 are the tap-row counts of the kernels' Blocks.
type (
	lpfR2  = [2]struct{}
	lpfR4  = [4]struct{}
	lpfR6  = [6]struct{}
	lpfR8  = [8]struct{}
	lpfR14 = [14]struct{}
	lpfR16 = [16]struct{}
)

// lpfTranspose16x16FromPix loads the sixteen 16-byte rows of tv, transposes
// the 16x16 byte matrix in registers and stores it as the rows of buf.
func lpfTranspose16x16FromPix(tv unsafeutil.Block[lpfR16], buf *[256]byte) {
	r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15 := lpfTranspose16x16(
		archsimd.LoadUint8x16Array(tv.Row16(0)), archsimd.LoadUint8x16Array(tv.Row16(1)), archsimd.LoadUint8x16Array(tv.Row16(2)), archsimd.LoadUint8x16Array(tv.Row16(3)), archsimd.LoadUint8x16Array(tv.Row16(4)), archsimd.LoadUint8x16Array(tv.Row16(5)), archsimd.LoadUint8x16Array(tv.Row16(6)), archsimd.LoadUint8x16Array(tv.Row16(7)), archsimd.LoadUint8x16Array(tv.Row16(8)), archsimd.LoadUint8x16Array(tv.Row16(9)), archsimd.LoadUint8x16Array(tv.Row16(10)), archsimd.LoadUint8x16Array(tv.Row16(11)), archsimd.LoadUint8x16Array(tv.Row16(12)), archsimd.LoadUint8x16Array(tv.Row16(13)), archsimd.LoadUint8x16Array(tv.Row16(14)), archsimd.LoadUint8x16Array(tv.Row16(15)))
	r0.StoreArray((*[16]byte)(buf[0:16]))
	r1.StoreArray((*[16]byte)(buf[16:32]))
	r2.StoreArray((*[16]byte)(buf[32:48]))
	r3.StoreArray((*[16]byte)(buf[48:64]))
	r4.StoreArray((*[16]byte)(buf[64:80]))
	r5.StoreArray((*[16]byte)(buf[80:96]))
	r6.StoreArray((*[16]byte)(buf[96:112]))
	r7.StoreArray((*[16]byte)(buf[112:128]))
	r8.StoreArray((*[16]byte)(buf[128:144]))
	r9.StoreArray((*[16]byte)(buf[144:160]))
	r10.StoreArray((*[16]byte)(buf[160:176]))
	r11.StoreArray((*[16]byte)(buf[176:192]))
	r12.StoreArray((*[16]byte)(buf[192:208]))
	r13.StoreArray((*[16]byte)(buf[208:224]))
	r14.StoreArray((*[16]byte)(buf[224:240]))
	r15.StoreArray((*[16]byte)(buf[240:256]))
}

// lpfTranspose16x16ToPix is lpfTranspose16x16FromPix back: the rows of buf,
// transposed, to the sixteen rows of tv.
func lpfTranspose16x16ToPix(buf *[256]byte, tv unsafeutil.Block[lpfR16]) {
	r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15 := lpfTranspose16x16(
		archsimd.LoadUint8x16Array((*[16]byte)(buf[0:16])), archsimd.LoadUint8x16Array((*[16]byte)(buf[16:32])), archsimd.LoadUint8x16Array((*[16]byte)(buf[32:48])), archsimd.LoadUint8x16Array((*[16]byte)(buf[48:64])), archsimd.LoadUint8x16Array((*[16]byte)(buf[64:80])), archsimd.LoadUint8x16Array((*[16]byte)(buf[80:96])), archsimd.LoadUint8x16Array((*[16]byte)(buf[96:112])), archsimd.LoadUint8x16Array((*[16]byte)(buf[112:128])), archsimd.LoadUint8x16Array((*[16]byte)(buf[128:144])), archsimd.LoadUint8x16Array((*[16]byte)(buf[144:160])), archsimd.LoadUint8x16Array((*[16]byte)(buf[160:176])), archsimd.LoadUint8x16Array((*[16]byte)(buf[176:192])), archsimd.LoadUint8x16Array((*[16]byte)(buf[192:208])), archsimd.LoadUint8x16Array((*[16]byte)(buf[208:224])), archsimd.LoadUint8x16Array((*[16]byte)(buf[224:240])), archsimd.LoadUint8x16Array((*[16]byte)(buf[240:256])))
	r0.StoreArray(tv.Row16(0))
	r1.StoreArray(tv.Row16(1))
	r2.StoreArray(tv.Row16(2))
	r3.StoreArray(tv.Row16(3))
	r4.StoreArray(tv.Row16(4))
	r5.StoreArray(tv.Row16(5))
	r6.StoreArray(tv.Row16(6))
	r7.StoreArray(tv.Row16(7))
	r8.StoreArray(tv.Row16(8))
	r9.StoreArray(tv.Row16(9))
	r10.StoreArray(tv.Row16(10))
	r11.StoreArray(tv.Row16(11))
	r12.StoreArray(tv.Row16(12))
	r13.StoreArray(tv.Row16(13))
	r14.StoreArray(tv.Row16(14))
	r15.StoreArray(tv.Row16(15))
}

// lpfTranspose16x16 transposes the 16x16 byte matrix whose rows are l0..l15.
func lpfTranspose16x16(l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14, l15 archsimd.Uint8x16) (r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15 archsimd.Uint8x16) {
	r0, r1 = l0.InterleaveEven(l1), l0.InterleaveOdd(l1)
	r2, r3 = l2.InterleaveEven(l3), l2.InterleaveOdd(l3)
	r4, r5 = l4.InterleaveEven(l5), l4.InterleaveOdd(l5)
	r6, r7 = l6.InterleaveEven(l7), l6.InterleaveOdd(l7)
	r8, r9 = l8.InterleaveEven(l9), l8.InterleaveOdd(l9)
	r10, r11 = l10.InterleaveEven(l11), l10.InterleaveOdd(l11)
	r12, r13 = l12.InterleaveEven(l13), l12.InterleaveOdd(l13)
	r14, r15 = l14.InterleaveEven(l15), l14.InterleaveOdd(l15)

	r0, r2 = r0.ReshapeToUint16s().InterleaveEven(r2.ReshapeToUint16s()).ReshapeToUint8s(), r0.ReshapeToUint16s().InterleaveOdd(r2.ReshapeToUint16s()).ReshapeToUint8s()
	r1, r3 = r1.ReshapeToUint16s().InterleaveEven(r3.ReshapeToUint16s()).ReshapeToUint8s(), r1.ReshapeToUint16s().InterleaveOdd(r3.ReshapeToUint16s()).ReshapeToUint8s()
	r4, r6 = r4.ReshapeToUint16s().InterleaveEven(r6.ReshapeToUint16s()).ReshapeToUint8s(), r4.ReshapeToUint16s().InterleaveOdd(r6.ReshapeToUint16s()).ReshapeToUint8s()
	r5, r7 = r5.ReshapeToUint16s().InterleaveEven(r7.ReshapeToUint16s()).ReshapeToUint8s(), r5.ReshapeToUint16s().InterleaveOdd(r7.ReshapeToUint16s()).ReshapeToUint8s()
	r8, r10 = r8.ReshapeToUint16s().InterleaveEven(r10.ReshapeToUint16s()).ReshapeToUint8s(), r8.ReshapeToUint16s().InterleaveOdd(r10.ReshapeToUint16s()).ReshapeToUint8s()
	r9, r11 = r9.ReshapeToUint16s().InterleaveEven(r11.ReshapeToUint16s()).ReshapeToUint8s(), r9.ReshapeToUint16s().InterleaveOdd(r11.ReshapeToUint16s()).ReshapeToUint8s()
	r12, r14 = r12.ReshapeToUint16s().InterleaveEven(r14.ReshapeToUint16s()).ReshapeToUint8s(), r12.ReshapeToUint16s().InterleaveOdd(r14.ReshapeToUint16s()).ReshapeToUint8s()
	r13, r15 = r13.ReshapeToUint16s().InterleaveEven(r15.ReshapeToUint16s()).ReshapeToUint8s(), r13.ReshapeToUint16s().InterleaveOdd(r15.ReshapeToUint16s()).ReshapeToUint8s()

	r0, r4 = r0.ReshapeToUint32s().InterleaveEven(r4.ReshapeToUint32s()).ReshapeToUint8s(), r0.ReshapeToUint32s().InterleaveOdd(r4.ReshapeToUint32s()).ReshapeToUint8s()
	r1, r5 = r1.ReshapeToUint32s().InterleaveEven(r5.ReshapeToUint32s()).ReshapeToUint8s(), r1.ReshapeToUint32s().InterleaveOdd(r5.ReshapeToUint32s()).ReshapeToUint8s()
	r2, r6 = r2.ReshapeToUint32s().InterleaveEven(r6.ReshapeToUint32s()).ReshapeToUint8s(), r2.ReshapeToUint32s().InterleaveOdd(r6.ReshapeToUint32s()).ReshapeToUint8s()
	r3, r7 = r3.ReshapeToUint32s().InterleaveEven(r7.ReshapeToUint32s()).ReshapeToUint8s(), r3.ReshapeToUint32s().InterleaveOdd(r7.ReshapeToUint32s()).ReshapeToUint8s()
	r8, r12 = r8.ReshapeToUint32s().InterleaveEven(r12.ReshapeToUint32s()).ReshapeToUint8s(), r8.ReshapeToUint32s().InterleaveOdd(r12.ReshapeToUint32s()).ReshapeToUint8s()
	r9, r13 = r9.ReshapeToUint32s().InterleaveEven(r13.ReshapeToUint32s()).ReshapeToUint8s(), r9.ReshapeToUint32s().InterleaveOdd(r13.ReshapeToUint32s()).ReshapeToUint8s()
	r10, r14 = r10.ReshapeToUint32s().InterleaveEven(r14.ReshapeToUint32s()).ReshapeToUint8s(), r10.ReshapeToUint32s().InterleaveOdd(r14.ReshapeToUint32s()).ReshapeToUint8s()
	r11, r15 = r11.ReshapeToUint32s().InterleaveEven(r15.ReshapeToUint32s()).ReshapeToUint8s(), r11.ReshapeToUint32s().InterleaveOdd(r15.ReshapeToUint32s()).ReshapeToUint8s()

	r0, r8 = r0.ReshapeToUint64s().InterleaveLo(r8.ReshapeToUint64s()).ReshapeToUint8s(), r0.ReshapeToUint64s().InterleaveHi(r8.ReshapeToUint64s()).ReshapeToUint8s()
	r1, r9 = r1.ReshapeToUint64s().InterleaveLo(r9.ReshapeToUint64s()).ReshapeToUint8s(), r1.ReshapeToUint64s().InterleaveHi(r9.ReshapeToUint64s()).ReshapeToUint8s()
	r2, r10 = r2.ReshapeToUint64s().InterleaveLo(r10.ReshapeToUint64s()).ReshapeToUint8s(), r2.ReshapeToUint64s().InterleaveHi(r10.ReshapeToUint64s()).ReshapeToUint8s()
	r3, r11 = r3.ReshapeToUint64s().InterleaveLo(r11.ReshapeToUint64s()).ReshapeToUint8s(), r3.ReshapeToUint64s().InterleaveHi(r11.ReshapeToUint64s()).ReshapeToUint8s()
	r4, r12 = r4.ReshapeToUint64s().InterleaveLo(r12.ReshapeToUint64s()).ReshapeToUint8s(), r4.ReshapeToUint64s().InterleaveHi(r12.ReshapeToUint64s()).ReshapeToUint8s()
	r5, r13 = r5.ReshapeToUint64s().InterleaveLo(r13.ReshapeToUint64s()).ReshapeToUint8s(), r5.ReshapeToUint64s().InterleaveHi(r13.ReshapeToUint64s()).ReshapeToUint8s()
	r6, r14 = r6.ReshapeToUint64s().InterleaveLo(r14.ReshapeToUint64s()).ReshapeToUint8s(), r6.ReshapeToUint64s().InterleaveHi(r14.ReshapeToUint64s()).ReshapeToUint8s()
	r7, r15 = r7.ReshapeToUint64s().InterleaveLo(r15.ReshapeToUint64s()).ReshapeToUint8s(), r7.ReshapeToUint64s().InterleaveHi(r15.ReshapeToUint64s()).ReshapeToUint8s()
	return
}

// lpfVertical16x8U8WideW is lpfVertical16x8U8 for groups whose rows can be
// read sixteen bytes at a time from p3 of the edge (the sixteen rows of tv):
// lpf8U8W in its column layout.
func lpfVertical16x8U8WideW[W lpfWidth](tv unsafeutil.Block[lpfR16], eV, iV, hV archsimd.Uint8x16, m4, mGt4 archsimd.Mask8x16) {
	lpf8U8W[W, lpfCols](tv, eV, iV, hV, m4, mGt4)
}

// lpfRows and lpfCols select lpf8U8W's tap layout (each type's size names
// it, as lpfW4..lpfW16 do the width).
type (
	// lpfRows: the taps are rows around q0 at p (dav1d lpf_v_16_*), p_k at
	// p - (k+1)*stride and q_k at p + k*stride.
	lpfRows [1]byte
	// lpfCols: the taps are columns across sixteen rows starting at p, p3 of
	// the first row (dav1d lpf_h_16_*). The rows load sixteen bytes at a time,
	// rows r and r+8 pair with one zip and an 8x8 transpose yields one vector
	// per tap; the filtered columns go back as dav1d stores them: wd 4/6
	// change only p1..q1, so a 4x16 transpose feeds one four-byte store per
	// row, and wd 8 runs the full 8x8 transpose back and stores eight bytes
	// per row. Lane stores leave the bytes past q3 untouched.
	lpfCols [2]byte
)

type lpfLayout interface {
	lpfRows | lpfCols
}

// lpf8U8W is dav1d lpf_16_wd{4,6,8}_neon with its lpf_{v,h}_16_* loads and
// stores: it filters sixteen lanes in layout L at width W and reports whether
// any lane passed the filter mask (nothing is stored otherwise). p3/q3 are
// only read at wd 8, p2/q2 from wd 6; p3/q3 never change at these widths.
//
// The taps are the rows of tv: in the row layout the wd/2 rows above the edge
// and the wd/2 below it (the 4+4 rows of wd 8, 3+3 of wd 6, 2+2 of wd 4; R
// holds 2*(wd/2) rows), sixteen bytes each; in the column layout sixteen rows
// of sixteen bytes from p3. Every tap row below is a constant row of the
// Block, so no access costs a check.
func lpf8U8W[W lpfWidth, L lpfLayout, R unsafeutil.BlockRows](tv unsafeutil.Block[R], eV, iV, hV archsimd.Uint8x16, m4, mGt4 archsimd.Mask8x16) bool {
	var wdT W
	var lT L
	wd := len(wdT)
	cols := len(lT) == 2
	rh := wd / 2 // rows per side in the row layout
	var p3, p2, p1, p0, q0, q1, q2, q3 archsimd.Uint8x16
	if cols {
		lo, hi := unsafeutil.BlockHalves(any(tv).(unsafeutil.Block[lpfR16]))
		r0, r8 := archsimd.LoadUint8x16Array(lo.Row16(0)), archsimd.LoadUint8x16Array(hi.Row16(0))
		r1, r9 := archsimd.LoadUint8x16Array(lo.Row16(1)), archsimd.LoadUint8x16Array(hi.Row16(1))
		r2, r10 := archsimd.LoadUint8x16Array(lo.Row16(2)), archsimd.LoadUint8x16Array(hi.Row16(2))
		r3, r11 := archsimd.LoadUint8x16Array(lo.Row16(3)), archsimd.LoadUint8x16Array(hi.Row16(3))
		r4, r12 := archsimd.LoadUint8x16Array(lo.Row16(4)), archsimd.LoadUint8x16Array(hi.Row16(4))
		r5, r13 := archsimd.LoadUint8x16Array(lo.Row16(5)), archsimd.LoadUint8x16Array(hi.Row16(5))
		r6, r14 := archsimd.LoadUint8x16Array(lo.Row16(6)), archsimd.LoadUint8x16Array(hi.Row16(6))
		r7, r15 := archsimd.LoadUint8x16Array(lo.Row16(7)), archsimd.LoadUint8x16Array(hi.Row16(7))
		p3, p2, p1, p0, q0, q1, q2, q3 = lpfTranspose8x8Pairs(
			r0.ReshapeToUint64s().InterleaveLo(r8.ReshapeToUint64s()).ReshapeToUint8s(), r1.ReshapeToUint64s().InterleaveLo(r9.ReshapeToUint64s()).ReshapeToUint8s(), r2.ReshapeToUint64s().InterleaveLo(r10.ReshapeToUint64s()).ReshapeToUint8s(), r3.ReshapeToUint64s().InterleaveLo(r11.ReshapeToUint64s()).ReshapeToUint8s(),
			r4.ReshapeToUint64s().InterleaveLo(r12.ReshapeToUint64s()).ReshapeToUint8s(), r5.ReshapeToUint64s().InterleaveLo(r13.ReshapeToUint64s()).ReshapeToUint8s(), r6.ReshapeToUint64s().InterleaveLo(r14.ReshapeToUint64s()).ReshapeToUint8s(), r7.ReshapeToUint64s().InterleaveLo(r15.ReshapeToUint64s()).ReshapeToUint8s())
	} else {
		p1 = archsimd.LoadUint8x16Array(tv.Row16(rh - 2))
		p0 = archsimd.LoadUint8x16Array(tv.Row16(rh - 1))
		q0 = archsimd.LoadUint8x16Array(tv.Row16(rh))
		q1 = archsimd.LoadUint8x16Array(tv.Row16(rh + 1))
		if wd >= 6 {
			p2 = archsimd.LoadUint8x16Array(tv.Row16(rh - 3))
			q2 = archsimd.LoadUint8x16Array(tv.Row16(rh + 2))
		}
		if wd >= 8 {
			p3 = archsimd.LoadUint8x16Array(tv.Row16(rh - 4))
			q3 = archsimd.LoadUint8x16Array(tv.Row16(rh + 3))
		}
	}
	o2, o1, o0, n0, n1, n2 := p2, p1, p0, q0, q1, q2
	maxd := lpfAbsDiff(p1, p0).Max(lpfAbsDiff(q1, q0))
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
		// dav1d lpf_16_wd* x14 = 1: nothing filtered, nothing to store.
		return false
	}
	mF4 := fm
	var mFlat archsimd.Mask8x16
	if wd >= 6 {
		var fl archsimd.Uint8x16
		if wd >= 8 {
			fl = lpfSpread(p2, p3, p0).Max(lpfSpread(q2, q3, q0))
		} else {
			fl = lpfAbsDiff(p2, p0).Max(lpfAbsDiff(q2, q0))
		}
		mFlat = archsimd.BroadcastUint8x16(1).GreaterEqual(fl.Max(maxd)).And(mGt4).And(fm)
		mF4 = fm.And(mFlat.Not())
	}
	if lpfAny(mF4) {
		// dav1d wd4 filter. archsimd has no USQADD, so the taps are biased to
		// signed as the C reference does; f = iclip_diff(3*(q0-p0) + (hev ?
		// iclip_diff(p1-q1) : 0)) is three saturating adds of the saturated
		// q0-p0 rather than dav1d's widened usubl/mul/saddw/sqxtn (exact: the
		// partial sums move monotonically toward the clamp, so saturating
		// early never changes the clamped total), and (f1+1)>>1 is a rounding
		// halving add with zero (srshr #1).
		bias := archsimd.BroadcastUint8x16(128)
		hev := maxd.Greater(hV)
		p1c := p1.Xor(bias).BitsToInt8()
		p0c := p0.Xor(bias).BitsToInt8()
		q0c := q0.Xor(bias).BitsToInt8()
		q1c := q1.Xor(bias).BitsToInt8()
		d := q0c.SubSaturated(p0c)
		f := p1c.SubSaturated(q1c).Masked(hev).AddSaturated(d).AddSaturated(d).AddSaturated(d)
		sh3 := archsimd.BroadcastInt8x16(-3)
		f1 := f.AddSaturated(archsimd.BroadcastInt8x16(4)).Shift(sh3)
		f2 := f.AddSaturated(archsimd.BroadcastInt8x16(3)).Shift(sh3)
		outer := f1.Average(archsimd.Int8x16{})
		o0 = p0c.AddSaturated(f2).ToBits().Xor(bias).IfElse(mF4, p0)
		n0 = q0c.SubSaturated(f1).ToBits().Xor(bias).IfElse(mF4, q0)
		mOuter := mF4.And(hev.Not())
		o1 = p1c.AddSaturated(outer).ToBits().Xor(bias).IfElse(mOuter, p1)
		n1 = q1c.SubSaturated(outer).ToBits().Xor(bias).IfElse(mOuter, q1)
	}
	if wd == 6 && lpfAny(mFlat) {
		k, k3 := archsimd.BroadcastUint8x16(32), archsimd.BroadcastUint8x16(96)
		bias8 := archsimd.BroadcastUint16x8(128)
		lP2, hP2 := p2.MulWidenLo(k), p2.HiToLo().MulWidenLo(k.HiToLo())
		lP1, hP1 := p1.MulWidenLo(k), p1.HiToLo().MulWidenLo(k.HiToLo())
		lP0, hP0 := p0.MulWidenLo(k), p0.HiToLo().MulWidenLo(k.HiToLo())
		lQ0, hQ0 := q0.MulWidenLo(k), q0.HiToLo().MulWidenLo(k.HiToLo())
		lQ1, hQ1 := q1.MulWidenLo(k), q1.HiToLo().MulWidenLo(k.HiToLo())
		lQ2, hQ2 := q2.MulWidenLo(k), q2.HiToLo().MulWidenLo(k.HiToLo())
		l3P2, h3P2 := p2.MulWidenLo(k3), p2.HiToLo().MulWidenLo(k3.HiToLo())
		lt, ht := lP1.Add(lP0), hP1.Add(hP0)
		la1, la0, lb0, lb1 := lpfFlat6Half(lP2, lP1, lP0, lQ0, lQ1, lQ2, l3P2.Add(lt.Add(lt)).Add(lQ0.Add(bias8)))
		ha1, ha0, hb0, hb1 := lpfFlat6Half(hP2, hP1, hP0, hQ0, hQ1, hQ2, h3P2.Add(ht.Add(ht)).Add(hQ0.Add(bias8)))
		o1 = la1.ReshapeToUint8s().ConcatOdd(ha1.ReshapeToUint8s()).IfElse(mFlat, o1)
		o0 = la0.ReshapeToUint8s().ConcatOdd(ha0.ReshapeToUint8s()).IfElse(mFlat, o0)
		n0 = lb0.ReshapeToUint8s().ConcatOdd(hb0.ReshapeToUint8s()).IfElse(mFlat, n0)
		n1 = lb1.ReshapeToUint8s().ConcatOdd(hb1.ReshapeToUint8s()).IfElse(mFlat, n1)
	} else if wd == 8 && lpfAny(mFlat) {
		k, k2, k3 := archsimd.BroadcastUint8x16(32), archsimd.BroadcastUint8x16(64), archsimd.BroadcastUint8x16(96)
		bias8 := archsimd.BroadcastUint16x8(128)
		lP3, hP3 := p3.MulWidenLo(k), p3.HiToLo().MulWidenLo(k.HiToLo())
		lP2, hP2 := p2.MulWidenLo(k), p2.HiToLo().MulWidenLo(k.HiToLo())
		lP1, hP1 := p1.MulWidenLo(k), p1.HiToLo().MulWidenLo(k.HiToLo())
		lP0, hP0 := p0.MulWidenLo(k), p0.HiToLo().MulWidenLo(k.HiToLo())
		lQ0, hQ0 := q0.MulWidenLo(k), q0.HiToLo().MulWidenLo(k.HiToLo())
		lQ1, hQ1 := q1.MulWidenLo(k), q1.HiToLo().MulWidenLo(k.HiToLo())
		lQ2, hQ2 := q2.MulWidenLo(k), q2.HiToLo().MulWidenLo(k.HiToLo())
		lQ3, hQ3 := q3.MulWidenLo(k), q3.HiToLo().MulWidenLo(k.HiToLo())
		l3P3, h3P3 := p3.MulWidenLo(k3), p3.HiToLo().MulWidenLo(k3.HiToLo())
		l2P2, h2P2 := p2.MulWidenLo(k2), p2.HiToLo().MulWidenLo(k2.HiToLo())
		la2, la1, la0, lb0, lb1, lb2 := lpfFlat8Half(lP3, lP2, lP1, lP0, lQ0, lQ1, lQ2, lQ3, l3P3.Add(l2P2).Add(lP1.Add(lP0)).Add(lQ0.Add(bias8)))
		ha2, ha1, ha0, hb0, hb1, hb2 := lpfFlat8Half(hP3, hP2, hP1, hP0, hQ0, hQ1, hQ2, hQ3, h3P3.Add(h2P2).Add(hP1.Add(hP0)).Add(hQ0.Add(bias8)))
		o2 = la2.ReshapeToUint8s().ConcatOdd(ha2.ReshapeToUint8s()).IfElse(mFlat, o2)
		o1 = la1.ReshapeToUint8s().ConcatOdd(ha1.ReshapeToUint8s()).IfElse(mFlat, o1)
		o0 = la0.ReshapeToUint8s().ConcatOdd(ha0.ReshapeToUint8s()).IfElse(mFlat, o0)
		n0 = lb0.ReshapeToUint8s().ConcatOdd(hb0.ReshapeToUint8s()).IfElse(mFlat, n0)
		n1 = lb1.ReshapeToUint8s().ConcatOdd(hb1.ReshapeToUint8s()).IfElse(mFlat, n1)
		n2 = lb2.ReshapeToUint8s().ConcatOdd(hb2.ReshapeToUint8s()).IfElse(mFlat, n2)
	}
	if cols {
		if wd < 8 {
			// dav1d transpose_4x16b: byte zips pair p1/p0 and q0/q1, halfword
			// zips gather each row's four bytes; lane k of c_j is row 4j+k.
			a0, a1 := o1.InterleaveLo(o0), o1.InterleaveHi(o0)
			b0, b1 := n0.InterleaveLo(n1), n0.InterleaveHi(n1)
			c0 := a0.ReshapeToUint16s().InterleaveLo(b0.ReshapeToUint16s()).ReshapeToUint32s().BitsToFloat32()
			c1 := a0.ReshapeToUint16s().InterleaveHi(b0.ReshapeToUint16s()).ReshapeToUint32s().BitsToFloat32()
			c2 := a1.ReshapeToUint16s().InterleaveLo(b1.ReshapeToUint16s()).ReshapeToUint32s().BitsToFloat32()
			c3 := a1.ReshapeToUint16s().InterleaveHi(b1.ReshapeToUint16s()).ReshapeToUint32s().BitsToFloat32()
			tv.PutF32(0, 2, c0.GetElem(0))
			tv.PutF32(1, 2, c0.GetElem(1))
			tv.PutF32(2, 2, c0.GetElem(2))
			tv.PutF32(3, 2, c0.GetElem(3))
			tv.PutF32(4, 2, c1.GetElem(0))
			tv.PutF32(5, 2, c1.GetElem(1))
			tv.PutF32(6, 2, c1.GetElem(2))
			tv.PutF32(7, 2, c1.GetElem(3))
			tv.PutF32(8, 2, c2.GetElem(0))
			tv.PutF32(9, 2, c2.GetElem(1))
			tv.PutF32(10, 2, c2.GetElem(2))
			tv.PutF32(11, 2, c2.GetElem(3))
			tv.PutF32(12, 2, c3.GetElem(0))
			tv.PutF32(13, 2, c3.GetElem(1))
			tv.PutF32(14, 2, c3.GetElem(2))
			tv.PutF32(15, 2, c3.GetElem(3))
		} else {
			t0, t1, t2, t3, t4, t5, t6, t7 := lpfTranspose8x8Pairs(p3, o2, o1, o0, n0, n1, n2, q3)
			tv.PutF64(0, 0, t0.ReshapeToUint64s().BitsToFloat64().GetElem(0))
			tv.PutF64(1, 0, t1.ReshapeToUint64s().BitsToFloat64().GetElem(0))
			tv.PutF64(2, 0, t2.ReshapeToUint64s().BitsToFloat64().GetElem(0))
			tv.PutF64(3, 0, t3.ReshapeToUint64s().BitsToFloat64().GetElem(0))
			tv.PutF64(4, 0, t4.ReshapeToUint64s().BitsToFloat64().GetElem(0))
			tv.PutF64(5, 0, t5.ReshapeToUint64s().BitsToFloat64().GetElem(0))
			tv.PutF64(6, 0, t6.ReshapeToUint64s().BitsToFloat64().GetElem(0))
			tv.PutF64(7, 0, t7.ReshapeToUint64s().BitsToFloat64().GetElem(0))
			tv.PutF64(8, 0, t0.ReshapeToUint64s().BitsToFloat64().GetElem(1))
			tv.PutF64(9, 0, t1.ReshapeToUint64s().BitsToFloat64().GetElem(1))
			tv.PutF64(10, 0, t2.ReshapeToUint64s().BitsToFloat64().GetElem(1))
			tv.PutF64(11, 0, t3.ReshapeToUint64s().BitsToFloat64().GetElem(1))
			tv.PutF64(12, 0, t4.ReshapeToUint64s().BitsToFloat64().GetElem(1))
			tv.PutF64(13, 0, t5.ReshapeToUint64s().BitsToFloat64().GetElem(1))
			tv.PutF64(14, 0, t6.ReshapeToUint64s().BitsToFloat64().GetElem(1))
			tv.PutF64(15, 0, t7.ReshapeToUint64s().BitsToFloat64().GetElem(1))
		}
	} else {
		if wd >= 8 {
			o2.StoreArray(tv.Row16(rh - 3))
			n2.StoreArray(tv.Row16(rh + 2))
		}
		o1.StoreArray(tv.Row16(rh - 2))
		o0.StoreArray(tv.Row16(rh - 1))
		n0.StoreArray(tv.Row16(rh))
		n1.StoreArray(tv.Row16(rh + 1))
	}
	return true
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

// lpf8U8 dispatches the eight row taps p3..q3 of v from row r0 (p3), column
// c0, to the width-specialized lpf8U8W instance (wd 4, 6 or 8; the narrower
// widths read only their inner rows).
func lpf8U8(wd int, v unsafeutil.Rows, r0, c0 int, eV, iV, hV archsimd.Uint8x16, m4, mGt4 archsimd.Mask8x16) bool {
	switch wd {
	case 8:
		return lpf8U8W[lpfW8, lpfRows](unsafeutil.BlockAt[lpfR8](v, r0, c0), eV, iV, hV, m4, mGt4)
	case 6:
		return lpf8U8W[lpfW6, lpfRows](unsafeutil.BlockAt[lpfR6](v, r0+1, c0), eV, iV, hV, m4, mGt4)
	default:
		return lpf8U8W[lpfW4, lpfRows](unsafeutil.BlockAt[lpfR4](v, r0+2, c0), eV, iV, hV, m4, mGt4)
	}
}
