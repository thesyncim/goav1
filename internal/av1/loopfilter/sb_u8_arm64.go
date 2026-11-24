// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package loopfilter

import (
	"encoding/binary"
	"math"
	"math/bits"
	"simd/archsimd"
)

// Every tap row is one bounds-checked window of the plane (lpfRow), taken once
// and reused for the store; the drivers have already range-checked the call,
// so the checks never fail. The transposes run through local arrays.

// lpfRow is the sixteen-byte row window of pix at o.
func lpfRow(pix []byte, o int) *[16]byte {
	return (*[16]byte)(pix[o : o+16])
}

// lpfWin4 is the four row windows from b at stride st. Callers pass b and st
// below 1<<32 (int(uint32(x))), which proves every row offset non-negative and
// overflow-free, so each window costs one compare against len(pix).
func lpfWin4(pix []byte, b, st int) (*[16]byte, *[16]byte, *[16]byte, *[16]byte) {
	return lpfRow(pix, b), lpfRow(pix, b+st), lpfRow(pix, b+2*st), lpfRow(pix, b+3*st)
}

// lpfMaxPlane bounds the planes the vector drivers take: every row offset of
// a plane up to 1<<32 bytes fits the 32-bit offsets of lpfWin4. Larger planes
// (beyond any AV1 frame) take the scalar reference.
const lpfMaxPlane = 1 << 32

// lpfPut32 stores a row's p1..q1 (bytes 2..5 of a column-layout window that
// starts at p3); the float lane keeps the store a single STR S.
func lpfPut32(w *[16]byte, v float32) {
	binary.LittleEndian.PutUint32(w[2:6], math.Float32bits(v))
}

// lpfPut64 stores a row's p3..q3 (bytes 0..7 of the window) as one STR D.
func lpfPut64(w *[16]byte, v float64) {
	binary.LittleEndian.PutUint64(w[0:8], math.Float64bits(v))
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

// lpf16U8Wd16 is dav1d's lpf_16_wd16_neon: it filters sixteen lanes whose
// taps are the rows wp6..wq6 around the edge (wq0 is the first row below it)
// and stores the changed rows back; the wd4, flat8in and flat8out lanes run
// in one body, each lane at its own width exactly as the C reference does.
// Widths up to 8 run lpf8U8W, which the transposed vertical path shares.
func lpf16U8Wd16(wp6, wp5, wp4, wp3, wp2, wp1, wp0, wq0, wq1, wq2, wq3, wq4, wq5, wq6 *[16]byte, eV, iV, hV archsimd.Uint8x16, m4, mGt4, m16 archsimd.Mask8x16) bool {
	p3 := archsimd.LoadUint8x16Array(wp3)
	p2 := archsimd.LoadUint8x16Array(wp2)
	p1 := archsimd.LoadUint8x16Array(wp1)
	p0 := archsimd.LoadUint8x16Array(wp0)
	q0 := archsimd.LoadUint8x16Array(wq0)
	q1 := archsimd.LoadUint8x16Array(wq1)
	q2 := archsimd.LoadUint8x16Array(wq2)
	q3 := archsimd.LoadUint8x16Array(wq3)
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
	p6 := archsimd.LoadUint8x16Array(wp6)
	p5 := archsimd.LoadUint8x16Array(wp5)
	p4 := archsimd.LoadUint8x16Array(wp4)
	q4 := archsimd.LoadUint8x16Array(wq4)
	q5 := archsimd.LoadUint8x16Array(wq5)
	q6 := archsimd.LoadUint8x16Array(wq6)
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
		o5, o4, o3, n3, n4, n5 := lpfFlat16U8(p6, p5, p4, p3, p2, p1, p0, q0, q1, q2, q3, q4, q5, q6, mFlat16, &o2, &o1, &o0, &n0, &n1, &n2)
		o5.StoreArray(wp5)
		o4.StoreArray(wp4)
		o3.StoreArray(wp3)
		n3.StoreArray(wq3)
		n4.StoreArray(wq4)
		n5.StoreArray(wq5)
	}
	o2.StoreArray(wp2)
	n2.StoreArray(wq2)
	o1.StoreArray(wp1)
	o0.StoreArray(wp0)
	n0.StoreArray(wq0)
	n1.StoreArray(wq1)
	return true
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
	return archsimd.LoadUint8x16Array(&sbNibbleLanes[n&15]).BitsToInt8().ToMask()
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

// sbLevelPair moves component sh/8 of the two level cells in l to the low
// bytes of two 32-bit lanes (the other bytes zero).
func sbLevelPair(l *[2][4]uint8, sh uint) uint64 {
	x := uint64(binary.LittleEndian.Uint32(l[0][:])) | uint64(binary.LittleEndian.Uint32(l[1][:]))<<32
	return x >> sh & 0x000000ff000000ff
}

// sbLevelCells4 is component sh/8 of the four level cells l, one per 32-bit
// lane.
func sbLevelCells4(l *[4][4]uint8, sh uint) archsimd.Uint32x4 {
	t := [2]uint64{sbLevelPair((*[2][4]uint8)(l[0:2]), sh), sbLevelPair((*[2][4]uint8)(l[2:4]), sh)}
	return archsimd.LoadUint64x2Array(&t).ReshapeToUint32s()
}

// sbLevels4 is component sh/8 of the four level cells from u, one per 32-bit
// lane (dav1d's ld1 of l[0]).
func sbLevels4(lvl [][4]uint8, u int, sh uint) archsimd.Uint32x4 {
	l := (*[4][4]uint8)(lvl[u : u+4])
	t := [2]uint64{sbLevelPair((*[2][4]uint8)(l[0:2]), sh), sbLevelPair((*[2][4]uint8)(l[2:4]), sh)}
	return archsimd.LoadUint64x2Array(&t).ReshapeToUint32s()
}

// sbLevelsPairs is component sh/8 of the (l[-1], l[0]) cell pairs of four
// units stacked ls cells apart from li (dav1d's ld2 lane loads): cur holds
// each unit's l[0], prev its l[-1].
func sbLevelsPairs(lvl [][4]uint8, li, ls int, sh uint) (cur, prev archsimd.Uint32x4) {
	i := li - 1
	x := [2]uint64{sbLevelPair((*[2][4]uint8)(lvl[i:i+2]), sh), sbLevelPair((*[2][4]uint8)(lvl[i+ls:i+ls+2]), sh)}
	i += 2 * ls
	y := [2]uint64{sbLevelPair((*[2][4]uint8)(lvl[i:i+2]), sh), sbLevelPair((*[2][4]uint8)(lvl[i+ls:i+ls+2]), sh)}
	a := archsimd.LoadUint64x2Array(&x).ReshapeToUint32s()
	b := archsimd.LoadUint64x2Array(&y).ReshapeToUint32s()
	return a.ConcatOdd(b), a.ConcatEven(b)
}

// sbLevelsRow loads the levels of four consecutive units (horizontal edges:
// dav1d lpf_v_sb ld1 of l[0] and l[-b4_stride]) with component comp moved to
// the low byte of each lane.
func sbLevelsRow(lvl [][4]uint8, u0, lvlStride, comp int, nAll uint32) (cur, prev archsimd.Uint32x4) {
	if u0-lvlStride >= 0 && u0+4 <= len(lvl) {
		sh := uint(8 * (comp & 3))
		return sbLevels4(lvl, u0, sh), sbLevels4(lvl, u0-lvlStride, sh)
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
func sbLevelsCol(lvl [][4]uint8, li, lvlStride, comp int, nAll uint32) (cur, prev archsimd.Uint32x4) {
	if li-1 >= 0 && li+3*lvlStride < len(lvl) {
		return sbLevelsPairs(lvl, li, lvlStride, uint(8*(comp&3)))
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

// filterSBHorizontal8Edge is the group-by-group driver for calls whose
// samples or levels reach the end of their buffers: each group checks its own
// reach and falls back to the unit-by-unit reference when it would leave pix.
func filterSBHorizontal8Edge(pix []byte, off, stride, rowBytes int, vm [3]uint32, luma bool, lvl [][4]uint8, lvlOff, lvlStride, comp int, lut *SBLUT) {
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
		reach := 4
		if wd == 16 {
			reach = 7
		}
		if 16*g+16 > rowBytes || q0Off-reach*stride < 0 || q0Off+(reach-1)*stride+16 > len(pix) {
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
		lpfRowsU8(pix, q0Off, stride, wd, eV, iV, hV, m4, sbNibbleMask(nGt4), sbNibbleMask(n16))
	}
}

// filterSBVertical8Edge is filterSBHorizontal8Edge for vertical edges.
func filterSBVertical8Edge(pix []byte, off, stride, rowBytes int, vm [3]uint32, luma bool, lvl [][4]uint8, lvlOff, lvlStride, comp int, lut *SBLUT) {
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
		lastRow := q0Off + 15*stride
		if wd == 16 {
			if q0Off-8 < 0 || lastRow+8 > len(pix) {
				sbVerticalScalarGroup(pix, q0Off, stride, vm, sh, luma, lvl, u0, lvlStride, comp, lut)
				continue
			}
		} else if q0Off-4 < 0 || lastRow+4 > len(pix) {
			sbVerticalScalarGroup(pix, q0Off, stride, vm, sh, luma, lvl, u0, lvlStride, comp, lut)
			continue
		}
		cur, prev := sbLevelsCol(lvl, u0, lvlStride, comp, nib)
		eV, iV, hV, m4, ok := sbGroupLimits(cur, prev, nib, sharp0, sharp1)
		if !ok {
			continue
		}
		if wd == 16 {
			lpfVertical16x16U8(pix, q0Off-8, stride, eV, iV, hV, m4, sbNibbleMask(nGt4), sbNibbleMask(n16))
		} else if rowBytes >= 12 && lastRow+12 <= len(pix) {
			p3 := q0Off - 4
			mGt4 := sbNibbleMask(nGt4)
			switch wd {
			case 8:
				lpfVertical16x8U8WideW[lpfW8](pix, p3, stride, eV, iV, hV, m4, mGt4)
			case 6:
				lpfVertical16x8U8WideW[lpfW6](pix, p3, stride, eV, iV, hV, m4, mGt4)
			default:
				lpfVertical16x8U8WideW[lpfW4](pix, p3, stride, eV, iV, hV, m4, mGt4)
			}
		} else {
			lpfVertical16x8U8(pix, q0Off-4, stride, wd, eV, iV, hV, m4, sbNibbleMask(nGt4))
		}
	}
}

// sbLimits is the per-group level prologue of dav1d lpf_func
// (src/arm/64/loopfilter.S), one unit per 32-bit lane: cur and prev hold each
// unit's own and fallback (left or above) level in the low byte of its lane
// with the other bytes zero. L = cur ? cur : prev is replicated over the
// unit's four bytes by a multiply with the bytes-of-one vector that also
// floors I; I = imax(imin(L >> sharp[0], sharp[1]), 1) and E = 2*(L+2)+I come
// from the LUT's replicated sharp vectors, H = L>>4. ok is false when no unit
// of the nibble has a level ("if (!L) continue").
func sbLimits(cur, prev archsimd.Uint32x4, nib uint32, lut *SBLUT) (eV, iV, hV archsimd.Uint8x16, m4 archsimd.Mask8x16, ok bool) {
	one := archsimd.BroadcastUint8x16(1)
	L := prev.IfElse(cur.Equal(archsimd.Uint32x4{}), cur)
	lr := L.Mul(one.ReshapeToUint32s()).ReshapeToUint8s()
	act := archsimd.LoadUint8x16Array(&sbNibbleLanes[nib&15]).AndNot(lr.Equal(archsimd.Uint8x16{}).ToInt8x16().ToBits())
	h := act.ReshapeToUint64s()
	if h.GetElem(0)|h.GetElem(1) == 0 {
		return
	}
	hV = lr.ShiftAllRight(4)
	iV = lr.Shift(archsimd.LoadInt8x16Array(&lut.SharpShift)).Min(archsimd.LoadUint8x16Array(&lut.SharpCap)).Max(one)
	e := lr.Add(archsimd.BroadcastUint8x16(2))
	eV = e.Add(e).Add(iV)
	m4 = act.BitsToInt8().ToMask()
	ok = true
	return
}

// filterSBHorizontal8 is dav1d lpf_v_sb_{y,uv}_8bpc_neon: the three width
// masks shift down one four-unit nibble per group while the sample and level
// cursors advance, each group's levels are one load of l[0] and one of
// l[-b4_stride] from a level cursor already pointing at component comp (the
// C side's &lvl[x][comp]), and the widest set nibble picks the kernel. The
// call's reach is checked once up front, as dav1d trusts its caller; calls
// that touch the end of pix, the row or the level map take the edge driver.
func filterSBHorizontal8(pix []byte, off, stride, rowBytes int, vm [3]uint32, luma bool, lvl [][4]uint8, lvlOff, lvlStride, comp int, lut *SBLUT) {
	vm0, vm1, vm2 := vm[0]|vm[1]|vm[2], vm[1]|vm[2], vm[2]
	end := 16 * ((31-bits.LeadingZeros32(vm0))>>2 + 1)
	reach := 4
	if vm2 != 0 {
		reach = 7
	}
	li, ls4 := 4*lvlOff+comp, 4*lvlStride
	if len(pix) > lpfMaxPlane {
		filterSBHorizontal8Scalar(pix, off, stride, vm, luma, lvl, lvlOff, lvlStride, comp, lut)
		return
	}
	if end > rowBytes || off < reach*stride || off+(reach-1)*stride+end > len(pix) || li < ls4 || li+end > 4*len(lvl) {
		filterSBHorizontal8Edge(pix, off, stride, rowBytes, vm, luma, lvl, lvlOff, lvlStride, comp, lut)
		return
	}
	sh := uint(8 * (comp & 3))
	// Each tap row (and the two level rows) is one window of the call's end
	// columns, taken once; the group loop's bounds prove every group's
	// sixteen-byte (four-cell) window, so the groups pay no checks. Without a
	// wd 16 group the outer rows p6..p4, q4..q6 alias q0's row (never read).
	n, far := end, 0
	if vm2 != 0 {
		far = stride
	}
	rq0 := pix[off:][:n:n]
	rp0 := pix[off-stride:][:n:n]
	rp1 := pix[off-2*stride:][:n:n]
	rp2 := pix[off-3*stride:][:n:n]
	rp3 := pix[off-4*stride:][:n:n]
	rq1 := pix[off+stride:][:n:n]
	rq2 := pix[off+2*stride:][:n:n]
	rq3 := pix[off+3*stride:][:n:n]
	rp4 := pix[off-5*far:][:n:n]
	rp5 := pix[off-6*far:][:n:n]
	rp6 := pix[off-7*far:][:n:n]
	rq4 := pix[off+4*far:][:n:n]
	rq5 := pix[off+5*far:][:n:n]
	rq6 := pix[off+6*far:][:n:n]
	lc := lvl[lvlOff:][: n/4 : n/4]
	lp := lvl[lvlOff-lvlStride:][: n/4 : n/4]
	for x, u := 0, 0; vm0 != 0 && x <= len(rq0)-16 && u <= len(lc)-4 && u <= len(lp)-4; vm0, vm1, vm2, x, u = vm0>>4, vm1>>4, vm2>>4, x+16, u+4 {
		if vm0&15 == 0 {
			continue
		}
		eV, iV, hV, m4, ok := sbLimits(sbLevelCells4((*[4][4]uint8)(lc[u:u+4]), sh), sbLevelCells4((*[4][4]uint8)(lp[u:u+4]), sh), vm0, lut)
		if !ok {
			continue
		}
		wp3, wp2, wp1, wp0 := (*[16]byte)(rp3[x:x+16]), (*[16]byte)(rp2[x:x+16]), (*[16]byte)(rp1[x:x+16]), (*[16]byte)(rp0[x:x+16])
		wq0, wq1, wq2, wq3 := (*[16]byte)(rq0[x:x+16]), (*[16]byte)(rq1[x:x+16]), (*[16]byte)(rq2[x:x+16]), (*[16]byte)(rq3[x:x+16])
		if vm2&15 != 0 {
			wp6, wp5, wp4 := (*[16]byte)(rp6[x:x+16]), (*[16]byte)(rp5[x:x+16]), (*[16]byte)(rp4[x:x+16])
			wq4, wq5, wq6 := (*[16]byte)(rq4[x:x+16]), (*[16]byte)(rq5[x:x+16]), (*[16]byte)(rq6[x:x+16])
			lpf16U8Wd16(wp6, wp5, wp4, wp3, wp2, wp1, wp0, wq0, wq1, wq2, wq3, wq4, wq5, wq6, eV, iV, hV, m4, sbNibbleMask(vm1), sbNibbleMask(vm2))
		} else if vm1&15 != 0 {
			if luma {
				lpf8U8W[lpfW8, lpfRows](nil, 0, 0, wp3, wp2, wp1, wp0, wq0, wq1, wq2, wq3, eV, iV, hV, m4, sbNibbleMask(vm1))
			} else {
				lpf8U8W[lpfW6, lpfRows](nil, 0, 0, nil, wp2, wp1, wp0, wq0, wq1, wq2, nil, eV, iV, hV, m4, sbNibbleMask(vm1))
			}
		} else {
			lpf8U8W[lpfW4, lpfRows](nil, 0, 0, nil, nil, wp1, wp0, wq0, wq1, nil, nil, eV, iV, hV, m4, archsimd.Mask8x16{})
		}
	}
}

// lpfRowsU8 is the per-group kernel choice of filterSBHorizontal8 for the
// edge driver: the width-wd row kernel on the tap rows around q0.
func lpfRowsU8(pix []byte, q0, stride, wd int, eV, iV, hV archsimd.Uint8x16, m4, mGt4, m16 archsimd.Mask8x16) {
	st := int(uint32(stride))
	switch wd {
	case 16:
		b := int(uint32(q0 - 7*stride))
		wp6, wp5, wp4, wp3 := lpfWin4(pix, b, st)
		wp2, wp1, wp0, wq0 := lpfWin4(pix, b+4*st, st)
		wq1, wq2, wq3, wq4 := lpfWin4(pix, b+8*st, st)
		wq5, wq6 := lpfRow(pix, b+12*st), lpfRow(pix, b+13*st)
		lpf16U8Wd16(wp6, wp5, wp4, wp3, wp2, wp1, wp0, wq0, wq1, wq2, wq3, wq4, wq5, wq6, eV, iV, hV, m4, mGt4, m16)
	case 8:
		b := int(uint32(q0 - 4*stride))
		wp3, wp2, wp1, wp0 := lpfWin4(pix, b, st)
		wq0, wq1, wq2, wq3 := lpfWin4(pix, b+4*st, st)
		lpf8U8W[lpfW8, lpfRows](nil, 0, 0, wp3, wp2, wp1, wp0, wq0, wq1, wq2, wq3, eV, iV, hV, m4, mGt4)
	case 6:
		b := int(uint32(q0 - 3*stride))
		wp2, wp1, wp0, wq0 := lpfWin4(pix, b, st)
		wq1, wq2 := lpfRow(pix, b+4*st), lpfRow(pix, b+5*st)
		lpf8U8W[lpfW6, lpfRows](nil, 0, 0, nil, wp2, wp1, wp0, wq0, wq1, wq2, nil, eV, iV, hV, m4, mGt4)
	default:
		wp1, wp0, wq0, wq1 := lpfWin4(pix, int(uint32(q0-2*stride)), st)
		lpf8U8W[lpfW4, lpfRows](nil, 0, 0, nil, nil, wp1, wp0, wq0, wq1, nil, nil, eV, iV, hV, m4, mGt4)
	}
}

// filterSBVertical8 is dav1d lpf_h_sb_{y,uv}_8bpc_neon in the same shape:
// each group reads the (l[-1], l[0]) level pairs of its four units (the ld2
// lane loads) from the component-offset cursor.
func filterSBVertical8(pix []byte, off, stride, rowBytes int, vm [3]uint32, luma bool, lvl [][4]uint8, lvlOff, lvlStride, comp int, lut *SBLUT) {
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
	if rowBytes < 12 || off < reach || off+(16*hiG+15)*stride+12 > len(pix) || li < 4 || li+(4*hiG+3)*ls4+4 > 4*len(lvl) {
		filterSBVertical8Edge(pix, off, stride, rowBytes, vm, luma, lvl, lvlOff, lvlStride, comp, lut)
		return
	}
	sh := uint(8 * (comp & 3))
	stride16 := 16 * stride
	for q0, u := off, lvlOff; vm0 != 0; vm0, vm1, vm2, q0, u = vm0>>4, vm1>>4, vm2>>4, q0+stride16, u+4*lvlStride {
		if vm0&15 == 0 {
			continue
		}
		cur, prev := sbLevelsPairs(lvl, u, lvlStride, sh)
		eV, iV, hV, m4, ok := sbLimits(cur, prev, vm0, lut)
		if !ok {
			continue
		}
		if vm2&15 != 0 {
			lpfVertical16x16U8(pix, q0-8, stride, eV, iV, hV, m4, sbNibbleMask(vm1), sbNibbleMask(vm2))
		} else if vm1&15 != 0 {
			if luma {
				lpfVertical16x8U8WideW[lpfW8](pix, q0-4, stride, eV, iV, hV, m4, sbNibbleMask(vm1))
			} else {
				lpfVertical16x8U8WideW[lpfW6](pix, q0-4, stride, eV, iV, hV, m4, sbNibbleMask(vm1))
			}
		} else {
			lpfVertical16x8U8WideW[lpfW4](pix, q0-4, stride, eV, iV, hV, m4, archsimd.Mask8x16{})
		}
	}
}

func sbVerticalScalarGroup(pix []byte, q0Off, stride int, vm [3]uint32, sh uint, luma bool, lvl [][4]uint8, u0, lvlStride, comp int, lut *SBLUT) {
	sub := vm
	for k := range sub {
		sub[k] = (sub[k] >> sh) & 0xf
	}
	filterSBVertical8Scalar(pix, q0Off, stride, sub, luma, lvl, u0, lvlStride, comp, lut)
}

// lpfVertical16x8U8 filters a vertical edge across sixteen rows using eight
// bytes per row (p3..q3): o is the offset of p3 in the first row. The rows
// are transposed into tap vectors, filtered, and transposed back.
func lpfVertical16x8U8(pix []byte, o, stride int, wd int, eV, iV, hV archsimd.Uint8x16, m4, mGt4 archsimd.Mask8x16) {
	var buf [128]byte
	// Pair row r (low half) with row r+8 (high half), then an 8x8 byte
	// transpose of both halves at once yields one vector per tap.
	var pr [8][2]uint64
	for r := 0; r < 8; r++ {
		a, b := o+r*stride, o+(r+8)*stride
		pr[r][0] = binary.LittleEndian.Uint64(pix[a : a+8])
		pr[r][1] = binary.LittleEndian.Uint64(pix[b : b+8])
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
	if !lpf8U8(wd, (*[16]byte)(buf[0:16]), (*[16]byte)(buf[16:32]), (*[16]byte)(buf[32:48]), (*[16]byte)(buf[48:64]),
		(*[16]byte)(buf[64:80]), (*[16]byte)(buf[80:96]), (*[16]byte)(buf[96:112]), (*[16]byte)(buf[112:128]), eV, iV, hV, m4, mGt4) {
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
		a, b := o+r*stride, o+(r+8)*stride
		binary.LittleEndian.PutUint64(pix[a:a+8], pr[r][0])
		binary.LittleEndian.PutUint64(pix[b:b+8], pr[r][1])
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

// lpfVertical16x16U8 is the wd16 variant: sixteen bytes per row (p7..q7, o
// is the offset of p7 in the first row), a full 16x16 transpose each way
// through buf, where the row kernel sees taps p6..q6 as buffer rows 1..14.
func lpfVertical16x16U8(pix []byte, o, stride int, eV, iV, hV archsimd.Uint8x16, m4, mGt4, m16 archsimd.Mask8x16) {
	var buf [256]byte
	lpfTranspose16x16FromPix(pix, o, stride, &buf)
	if !lpf16U8Wd16((*[16]byte)(buf[16:32]), (*[16]byte)(buf[32:48]), (*[16]byte)(buf[48:64]), (*[16]byte)(buf[64:80]), (*[16]byte)(buf[80:96]), (*[16]byte)(buf[96:112]), (*[16]byte)(buf[112:128]), (*[16]byte)(buf[128:144]), (*[16]byte)(buf[144:160]), (*[16]byte)(buf[160:176]), (*[16]byte)(buf[176:192]), (*[16]byte)(buf[192:208]), (*[16]byte)(buf[208:224]), (*[16]byte)(buf[224:240]), eV, iV, hV, m4, mGt4, m16) {
		return
	}
	lpfTranspose16x16ToPix(&buf, pix, o, stride)
}

// lpfTranspose16x16FromPix loads the sixteen 16-byte rows of pix from o (one
// window each, offsets below 1<<32), transposes the 16x16 byte matrix in
// registers and stores it as the rows of buf.
func lpfTranspose16x16FromPix(pix []byte, o, stride int, buf *[256]byte) {
	b, st := int(uint32(o)), int(uint32(stride))
	w0, w1, w2, w3 := lpfWin4(pix, b, st)
	w4, w5, w6, w7 := lpfWin4(pix, b+4*st, st)
	w8, w9, w10, w11 := lpfWin4(pix, b+8*st, st)
	w12, w13, w14, w15 := lpfWin4(pix, b+12*st, st)
	r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15 := lpfTranspose16x16(
		archsimd.LoadUint8x16Array(w0), archsimd.LoadUint8x16Array(w1), archsimd.LoadUint8x16Array(w2), archsimd.LoadUint8x16Array(w3), archsimd.LoadUint8x16Array(w4), archsimd.LoadUint8x16Array(w5), archsimd.LoadUint8x16Array(w6), archsimd.LoadUint8x16Array(w7), archsimd.LoadUint8x16Array(w8), archsimd.LoadUint8x16Array(w9), archsimd.LoadUint8x16Array(w10), archsimd.LoadUint8x16Array(w11), archsimd.LoadUint8x16Array(w12), archsimd.LoadUint8x16Array(w13), archsimd.LoadUint8x16Array(w14), archsimd.LoadUint8x16Array(w15))
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
// transposed, to the sixteen rows of pix from o.
func lpfTranspose16x16ToPix(buf *[256]byte, pix []byte, o, stride int) {
	r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15 := lpfTranspose16x16(
		archsimd.LoadUint8x16Array((*[16]byte)(buf[0:16])), archsimd.LoadUint8x16Array((*[16]byte)(buf[16:32])), archsimd.LoadUint8x16Array((*[16]byte)(buf[32:48])), archsimd.LoadUint8x16Array((*[16]byte)(buf[48:64])), archsimd.LoadUint8x16Array((*[16]byte)(buf[64:80])), archsimd.LoadUint8x16Array((*[16]byte)(buf[80:96])), archsimd.LoadUint8x16Array((*[16]byte)(buf[96:112])), archsimd.LoadUint8x16Array((*[16]byte)(buf[112:128])), archsimd.LoadUint8x16Array((*[16]byte)(buf[128:144])), archsimd.LoadUint8x16Array((*[16]byte)(buf[144:160])), archsimd.LoadUint8x16Array((*[16]byte)(buf[160:176])), archsimd.LoadUint8x16Array((*[16]byte)(buf[176:192])), archsimd.LoadUint8x16Array((*[16]byte)(buf[192:208])), archsimd.LoadUint8x16Array((*[16]byte)(buf[208:224])), archsimd.LoadUint8x16Array((*[16]byte)(buf[224:240])), archsimd.LoadUint8x16Array((*[16]byte)(buf[240:256])))
	b, st := int(uint32(o)), int(uint32(stride))
	w0, w1, w2, w3 := lpfWin4(pix, b, st)
	w4, w5, w6, w7 := lpfWin4(pix, b+4*st, st)
	w8, w9, w10, w11 := lpfWin4(pix, b+8*st, st)
	w12, w13, w14, w15 := lpfWin4(pix, b+12*st, st)
	r0.StoreArray(w0)
	r1.StoreArray(w1)
	r2.StoreArray(w2)
	r3.StoreArray(w3)
	r4.StoreArray(w4)
	r5.StoreArray(w5)
	r6.StoreArray(w6)
	r7.StoreArray(w7)
	r8.StoreArray(w8)
	r9.StoreArray(w9)
	r10.StoreArray(w10)
	r11.StoreArray(w11)
	r12.StoreArray(w12)
	r13.StoreArray(w13)
	r14.StoreArray(w14)
	r15.StoreArray(w15)
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
// read sixteen bytes at a time from p (p3 of the edge): lpf8U8W in its column
// layout.
func lpfVertical16x8U8WideW[W lpfWidth](pix []byte, o, stride int, eV, iV, hV archsimd.Uint8x16, m4, mGt4 archsimd.Mask8x16) {
	lpf8U8W[W, lpfCols](pix, o, stride, nil, nil, nil, nil, nil, nil, nil, nil, eV, iV, hV, m4, mGt4)
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
func lpf8U8W[W lpfWidth, L lpfLayout](pix []byte, o, stride int, wp3, wp2, wp1, wp0, wq0, wq1, wq2, wq3 *[16]byte, eV, iV, hV archsimd.Uint8x16, m4, mGt4 archsimd.Mask8x16) bool {
	var wdT W
	var lT L
	wd := len(wdT)
	cols := len(lT) == 2
	var p3, p2, p1, p0, q0, q1, q2, q3 archsimd.Uint8x16
	// The row layout gets its tap rows as windows (the caller took them);
	// the column layout takes its sixteen row windows w0..w15 here, once
	// each, and reuses them for the stores. Offsets the drivers keep below
	// 1<<32 make each window one compare.
	var w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15 *[16]byte
	if cols {
		b, st := int(uint32(o)), int(uint32(stride))
		w0, w1, w2, w3 = lpfWin4(pix, b, st)
		w4, w5, w6, w7 = lpfWin4(pix, b+4*st, st)
		w8, w9, w10, w11 = lpfWin4(pix, b+8*st, st)
		w12, w13, w14, w15 = lpfWin4(pix, b+12*st, st)
		r0, r8 := archsimd.LoadUint8x16Array(w0), archsimd.LoadUint8x16Array(w8)
		r1, r9 := archsimd.LoadUint8x16Array(w1), archsimd.LoadUint8x16Array(w9)
		r2, r10 := archsimd.LoadUint8x16Array(w2), archsimd.LoadUint8x16Array(w10)
		r3, r11 := archsimd.LoadUint8x16Array(w3), archsimd.LoadUint8x16Array(w11)
		r4, r12 := archsimd.LoadUint8x16Array(w4), archsimd.LoadUint8x16Array(w12)
		r5, r13 := archsimd.LoadUint8x16Array(w5), archsimd.LoadUint8x16Array(w13)
		r6, r14 := archsimd.LoadUint8x16Array(w6), archsimd.LoadUint8x16Array(w14)
		r7, r15 := archsimd.LoadUint8x16Array(w7), archsimd.LoadUint8x16Array(w15)
		p3, p2, p1, p0, q0, q1, q2, q3 = lpfTranspose8x8Pairs(
			r0.ReshapeToUint64s().InterleaveLo(r8.ReshapeToUint64s()).ReshapeToUint8s(), r1.ReshapeToUint64s().InterleaveLo(r9.ReshapeToUint64s()).ReshapeToUint8s(), r2.ReshapeToUint64s().InterleaveLo(r10.ReshapeToUint64s()).ReshapeToUint8s(), r3.ReshapeToUint64s().InterleaveLo(r11.ReshapeToUint64s()).ReshapeToUint8s(),
			r4.ReshapeToUint64s().InterleaveLo(r12.ReshapeToUint64s()).ReshapeToUint8s(), r5.ReshapeToUint64s().InterleaveLo(r13.ReshapeToUint64s()).ReshapeToUint8s(), r6.ReshapeToUint64s().InterleaveLo(r14.ReshapeToUint64s()).ReshapeToUint8s(), r7.ReshapeToUint64s().InterleaveLo(r15.ReshapeToUint64s()).ReshapeToUint8s())
	} else {
		p1 = archsimd.LoadUint8x16Array(wp1)
		p0 = archsimd.LoadUint8x16Array(wp0)
		q0 = archsimd.LoadUint8x16Array(wq0)
		q1 = archsimd.LoadUint8x16Array(wq1)
		if wd >= 6 {
			p2 = archsimd.LoadUint8x16Array(wp2)
			q2 = archsimd.LoadUint8x16Array(wq2)
		}
		if wd >= 8 {
			p3 = archsimd.LoadUint8x16Array(wp3)
			q3 = archsimd.LoadUint8x16Array(wq3)
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
			lpfPut32(w0, c0.GetElem(0))
			lpfPut32(w1, c0.GetElem(1))
			lpfPut32(w2, c0.GetElem(2))
			lpfPut32(w3, c0.GetElem(3))
			lpfPut32(w4, c1.GetElem(0))
			lpfPut32(w5, c1.GetElem(1))
			lpfPut32(w6, c1.GetElem(2))
			lpfPut32(w7, c1.GetElem(3))
			lpfPut32(w8, c2.GetElem(0))
			lpfPut32(w9, c2.GetElem(1))
			lpfPut32(w10, c2.GetElem(2))
			lpfPut32(w11, c2.GetElem(3))
			lpfPut32(w12, c3.GetElem(0))
			lpfPut32(w13, c3.GetElem(1))
			lpfPut32(w14, c3.GetElem(2))
			lpfPut32(w15, c3.GetElem(3))
		} else {
			t0, t1, t2, t3, t4, t5, t6, t7 := lpfTranspose8x8Pairs(p3, o2, o1, o0, n0, n1, n2, q3)
			lpfPut64(w0, t0.ReshapeToUint64s().BitsToFloat64().GetElem(0))
			lpfPut64(w1, t1.ReshapeToUint64s().BitsToFloat64().GetElem(0))
			lpfPut64(w2, t2.ReshapeToUint64s().BitsToFloat64().GetElem(0))
			lpfPut64(w3, t3.ReshapeToUint64s().BitsToFloat64().GetElem(0))
			lpfPut64(w4, t4.ReshapeToUint64s().BitsToFloat64().GetElem(0))
			lpfPut64(w5, t5.ReshapeToUint64s().BitsToFloat64().GetElem(0))
			lpfPut64(w6, t6.ReshapeToUint64s().BitsToFloat64().GetElem(0))
			lpfPut64(w7, t7.ReshapeToUint64s().BitsToFloat64().GetElem(0))
			lpfPut64(w8, t0.ReshapeToUint64s().BitsToFloat64().GetElem(1))
			lpfPut64(w9, t1.ReshapeToUint64s().BitsToFloat64().GetElem(1))
			lpfPut64(w10, t2.ReshapeToUint64s().BitsToFloat64().GetElem(1))
			lpfPut64(w11, t3.ReshapeToUint64s().BitsToFloat64().GetElem(1))
			lpfPut64(w12, t4.ReshapeToUint64s().BitsToFloat64().GetElem(1))
			lpfPut64(w13, t5.ReshapeToUint64s().BitsToFloat64().GetElem(1))
			lpfPut64(w14, t6.ReshapeToUint64s().BitsToFloat64().GetElem(1))
			lpfPut64(w15, t7.ReshapeToUint64s().BitsToFloat64().GetElem(1))
		}
	} else {
		if wd >= 8 {
			o2.StoreArray(wp2)
			n2.StoreArray(wq2)
		}
		o1.StoreArray(wp1)
		o0.StoreArray(wp0)
		n0.StoreArray(wq0)
		n1.StoreArray(wq1)
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

// lpf8U8 dispatches the row taps p3..q3 to the width-specialized lpf8U8W
// instance (wd 4, 6 or 8).
func lpf8U8(wd int, wp3, wp2, wp1, wp0, wq0, wq1, wq2, wq3 *[16]byte, eV, iV, hV archsimd.Uint8x16, m4, mGt4 archsimd.Mask8x16) bool {
	switch wd {
	case 8:
		return lpf8U8W[lpfW8, lpfRows](nil, 0, 0, wp3, wp2, wp1, wp0, wq0, wq1, wq2, wq3, eV, iV, hV, m4, mGt4)
	case 6:
		return lpf8U8W[lpfW6, lpfRows](nil, 0, 0, wp3, wp2, wp1, wp0, wq0, wq1, wq2, wq3, eV, iV, hV, m4, mGt4)
	default:
		return lpf8U8W[lpfW4, lpfRows](nil, 0, 0, wp3, wp2, wp1, wp0, wq0, wq1, wq2, wq3, eV, iV, hV, m4, mGt4)
	}
}
