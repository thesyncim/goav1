// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package loopfilter

import (
	"encoding/binary"
	"math/bits"
	"simd/archsimd"
)

// Every tap row is one bounds-checked window of the plane (lpfRow), taken
// from 32-bit offsets (the drivers send planes past lpfMaxPlane to the scalar
// reference), so each row costs one compare; the drivers have range-checked
// the call, so the checks never fail.

// sbPairLanes16[b] selects the four uint16 lanes of every unit of a two-unit
// half group whose bit is set in b (unit k owns lanes 4k..4k+3).
var sbPairLanes16 = func() (t [4][8]uint16) {
	for b := range t {
		for k := 0; k < 2; k++ {
			if b&(1<<k) != 0 {
				for i := 0; i < 4; i++ {
					t[b][4*k+i] = 0xffff
				}
			}
		}
	}
	return t
}()

func lpf16Ld(w *[16]byte) archsimd.Uint16x8 {
	return archsimd.LoadUint8x16Array(w).ReshapeToUint16s()
}

// lpfWin2d is lpfWin2 for two rows d bytes apart (b and d below 1<<32): both
// windows end at the second row's end, one compare for the pair.
func lpfWin2d(pix []byte, b, d int) (*[16]byte, *[16]byte) {
	e := b + d + 16
	return (*[16]byte)(pix[b:e]), (*[16]byte)(pix[b+d : e])
}

func lpf16Any(m archsimd.Mask16x8) bool {
	b := m.ToInt16x8().ToBits().ReshapeToUint64s()
	return b.GetElem(0)|b.GetElem(1) != 0
}

// lpf8U16 is the high-bit-depth counterpart of lpf16U8 on eight uint16 lanes
// (two 4-sample units): dav1d's loop_filter() with E/I/H and the flatness
// threshold scaled by bitdepth_min_8 (the bitdepth's lpf16Tab row k),
// filter4 in signed 16-bit lanes with the iclip_diff/iclip_pixel bounds, and
// the flat filters' sums held exactly in uint16 lanes (at most 16*4095+8).
// Taps are rows around q0 (a byte offset into pix), stride in bytes. Vertical
// edges run lpfCols16.
func lpf8U16(pix []byte, q0Off, stride int, wd int, k *lpf16Consts, eV, iV, hV archsimd.Uint16x8, m4, mGt4, m16 archsimd.Mask16x8) {
	// Tap rows p3..q3 from t4 (the p3 row); at wd 16 p6..p4 from t7. Each
	// row pair is one window pair (one compare), taken once and reused by
	// the stores; offsets below 1<<32 (the drivers send larger planes to the
	// scalar reference).
	st, t4 := int(uint32(stride)), int(uint32(q0Off-4*stride))
	w1, w0, v0, v1 := lpfWin4(pix, t4+2*st, st)
	p1 := lpf16Ld(w1)
	p0 := lpf16Ld(w0)
	q0 := lpf16Ld(v0)
	q1 := lpf16Ld(v1)
	var w3, w2, v2, v3 *[16]byte
	d10 := lpf16Abd(p1, p0)
	dq := lpf16Abd(q1, q0)
	maxd := d10.Max(dq)
	t := maxd
	var p2, q2, p3, q3 archsimd.Uint16x8
	if wd >= 6 {
		w2, v2 = lpfWin2d(pix, t4+st, int(uint32(5*stride)))
		p2 = lpf16Ld(w2)
		q2 = lpf16Ld(v2)
		tt := lpf16Abd(p2, p1).Max(lpf16Abd(q2, q1))
		if wd >= 8 {
			w3, v3 = lpfWin2d(pix, t4, int(uint32(7*stride)))
			p3 = lpf16Ld(w3)
			q3 = lpf16Ld(v3)
			tt = tt.Max(lpf16Abd(p3, p2).Max(lpf16Abd(q3, q2)))
		}
		t = t.Max(tt.Masked(mGt4))
	}
	d00 := lpf16Abd(p0, q0)
	sum := d00.Add(d00).Add(lpf16Abd(p1, q1).Shift(archsimd.LoadInt16x8Array(&k.shr1)))
	fm := iV.GreaterEqual(t).And(eV.GreaterEqual(sum)).And(m4)
	if !lpf16Any(fm) {
		return
	}
	fV := archsimd.LoadUint16x8Array(&k.flat)
	mF4 := fm
	var mFlat, mFlat16 archsimd.Mask16x8
	var p6, p5, p4, q4, q5, q6 archsimd.Uint16x8
	var w5, w4, v4, v5 *[16]byte
	if wd >= 6 {
		fl := lpf16Abd(p2, p0).Max(maxd).Max(lpf16Abd(q2, q0))
		if wd >= 8 {
			fl = fl.Max(lpf16Abd(p3, p0)).Max(lpf16Abd(q3, q0))
		}
		mFlat = fV.GreaterEqual(fl).And(mGt4).And(fm)
		mF4 = fm.And(mFlat.Not())
		if wd == 16 {
			t7 := int(uint32(q0Off - 7*stride))
			w6, w5x := lpfWin2(pix, t7, st)
			w4x, v4x := lpfWin2d(pix, t7+2*st, int(uint32(9*stride)))
			v5x, v6 := lpfWin2(pix, t4+9*st, st)
			w5, w4, v4, v5 = w5x, w4x, v4x, v5x
			p6 = lpf16Ld(w6)
			p5 = lpf16Ld(w5)
			p4 = lpf16Ld(w4)
			q4 = lpf16Ld(v4)
			q5 = lpf16Ld(v5)
			q6 = lpf16Ld(v6)
			fo := lpf16Abd(p6, p0).Max(lpf16Abd(p5, p0)).Max(lpf16Abd(p4, p0)).
				Max(lpf16Abd(q4, q0)).Max(lpf16Abd(q5, q0)).Max(lpf16Abd(q6, q0))
			mFlat16 = fV.GreaterEqual(fo).And(m16).And(mFlat)
			mFlat = mFlat.And(mFlat16.Not())
		}
	}

	o1, o0, n0, n1 := p1, p0, q0, q1
	if lpf16Any(mF4) {
		hev := maxd.Greater(hV)
		dmin := archsimd.LoadInt16x8Array(&k.dmin)
		dmax := archsimd.LoadInt16x8Array(&k.dmax)
		zero := archsimd.Int16x8{}
		maxPix := archsimd.LoadInt16x8Array(&k.maxPix)
		p1s, p0s, q0s, q1s := p1.BitsToInt16(), p0.BitsToInt16(), q0.BitsToInt16(), q1.BitsToInt16()
		f := p1s.Sub(q1s).Max(dmin).Min(dmax).Masked(hev)
		f = q0s.Sub(p0s).Mul(archsimd.LoadInt16x8Array(&k.three)).Add(f).Max(dmin).Min(dmax)
		f1 := f.Add(archsimd.LoadInt16x8Array(&k.four)).Min(dmax).Shift(archsimd.LoadInt16x8Array(&k.shr3))
		f2 := f.Add(archsimd.LoadInt16x8Array(&k.three)).Min(dmax).Shift(archsimd.LoadInt16x8Array(&k.shr3))
		o0 = p0s.Add(f2).Max(zero).Min(maxPix).ToBits().IfElse(mF4, p0)
		n0 = q0s.Sub(f1).Max(zero).Min(maxPix).ToBits().IfElse(mF4, q0)
		f3 := f1.Add(archsimd.LoadInt16x8Array(&k.one)).Shift(archsimd.LoadInt16x8Array(&k.shr1))
		mOuter := mF4.And(hev.Not())
		o1 = p1s.Add(f3).Max(zero).Min(maxPix).ToBits().IfElse(mOuter, p1)
		n1 = q1s.Sub(f3).Max(zero).Min(maxPix).ToBits().IfElse(mOuter, q1)
	}

	if wd == 6 {
		if lpf16Any(mFlat) {
			s := p2.Add(p2).Add(p2).Add(p1.Add(p0).Add(p1.Add(p0))).Add(q0).Add(archsimd.LoadUint16x8Array(&k.rnd4))
			o1 = s.Shift(archsimd.LoadInt16x8Array(&k.shr3)).IfElse(mFlat, o1)
			s = s.Add(q0.Add(q1)).Sub(p2.Add(p2))
			o0 = s.Shift(archsimd.LoadInt16x8Array(&k.shr3)).IfElse(mFlat, o0)
			s = s.Add(q1.Add(q2)).Sub(p2.Add(p1))
			n0 = s.Shift(archsimd.LoadInt16x8Array(&k.shr3)).IfElse(mFlat, n0)
			s = s.Add(q2.Add(q2)).Sub(p1.Add(p0))
			n1 = s.Shift(archsimd.LoadInt16x8Array(&k.shr3)).IfElse(mFlat, n1)
		}
		o1.ReshapeToUint8s().StoreArray(w1)
		o0.ReshapeToUint8s().StoreArray(w0)
		n0.ReshapeToUint8s().StoreArray(v0)
		n1.ReshapeToUint8s().StoreArray(v1)
		return
	}

	o2, n2 := p2, q2
	if wd >= 8 && lpf16Any(mFlat) {
		s := p3.Add(p3).Add(p3).Add(p2.Add(p2)).Add(p1.Add(p0)).Add(q0.Add(archsimd.LoadUint16x8Array(&k.rnd4)))
		o2 = s.Shift(archsimd.LoadInt16x8Array(&k.shr3)).IfElse(mFlat, o2)
		s = s.Add(p1.Add(q1)).Sub(p3.Add(p2))
		o1 = s.Shift(archsimd.LoadInt16x8Array(&k.shr3)).IfElse(mFlat, o1)
		s = s.Add(p0.Add(q2)).Sub(p3.Add(p1))
		o0 = s.Shift(archsimd.LoadInt16x8Array(&k.shr3)).IfElse(mFlat, o0)
		s = s.Add(q0.Add(q3)).Sub(p3.Add(p0))
		n0 = s.Shift(archsimd.LoadInt16x8Array(&k.shr3)).IfElse(mFlat, n0)
		s = s.Add(q1.Add(q3)).Sub(p2.Add(q0))
		n1 = s.Shift(archsimd.LoadInt16x8Array(&k.shr3)).IfElse(mFlat, n1)
		s = s.Add(q2.Add(q3)).Sub(p1.Add(q1))
		n2 = s.Shift(archsimd.LoadInt16x8Array(&k.shr3)).IfElse(mFlat, n2)
	}

	if wd == 16 && lpf16Any(mFlat16) {
		m := mFlat16
		p6x2 := p6.Add(p6)
		s := p6.Add(p6x2).Add(p6x2.Add(p6x2))
		s = s.Add(p5.Add(p5)).Add(p4.Add(p4)).Add(p3.Add(p2)).Add(p1.Add(p0)).Add(q0).Add(archsimd.LoadUint16x8Array(&k.rnd8))
		s.Shift(archsimd.LoadInt16x8Array(&k.shr4)).IfElse(m, p5).ReshapeToUint8s().StoreArray(w5)
		s = s.Add(p3.Add(q1)).Sub(p6x2)
		s.Shift(archsimd.LoadInt16x8Array(&k.shr4)).IfElse(m, p4).ReshapeToUint8s().StoreArray(w4)
		s = s.Add(p2.Add(q2)).Sub(p6.Add(p5))
		s.Shift(archsimd.LoadInt16x8Array(&k.shr4)).IfElse(m, p3).ReshapeToUint8s().StoreArray(w3)
		s = s.Add(p1.Add(q3)).Sub(p6.Add(p4))
		o2 = s.Shift(archsimd.LoadInt16x8Array(&k.shr4)).IfElse(m, o2)
		s = s.Add(p0.Add(q4)).Sub(p6.Add(p3))
		o1 = s.Shift(archsimd.LoadInt16x8Array(&k.shr4)).IfElse(m, o1)
		s = s.Add(q0.Add(q5)).Sub(p6.Add(p2))
		o0 = s.Shift(archsimd.LoadInt16x8Array(&k.shr4)).IfElse(m, o0)
		s = s.Add(q1.Add(q6)).Sub(p6.Add(p1))
		n0 = s.Shift(archsimd.LoadInt16x8Array(&k.shr4)).IfElse(m, n0)
		s = s.Add(q2.Add(q6)).Sub(p5.Add(p0))
		n1 = s.Shift(archsimd.LoadInt16x8Array(&k.shr4)).IfElse(m, n1)
		s = s.Add(q3.Add(q6)).Sub(p4.Add(q0))
		n2 = s.Shift(archsimd.LoadInt16x8Array(&k.shr4)).IfElse(m, n2)
		s = s.Add(q4.Add(q6)).Sub(p3.Add(q1))
		s.Shift(archsimd.LoadInt16x8Array(&k.shr4)).IfElse(m, q3).ReshapeToUint8s().StoreArray(v3)
		s = s.Add(q5.Add(q6)).Sub(p2.Add(q2))
		s.Shift(archsimd.LoadInt16x8Array(&k.shr4)).IfElse(m, q4).ReshapeToUint8s().StoreArray(v4)
		s = s.Add(q6.Add(q6)).Sub(p1.Add(q3))
		s.Shift(archsimd.LoadInt16x8Array(&k.shr4)).IfElse(m, q5).ReshapeToUint8s().StoreArray(v5)
	}
	if wd >= 8 {
		o2.ReshapeToUint8s().StoreArray(w2)
		n2.ReshapeToUint8s().StoreArray(v2)
	}
	o1.ReshapeToUint8s().StoreArray(w1)
	o0.ReshapeToUint8s().StoreArray(w0)
	n0.ReshapeToUint8s().StoreArray(v0)
	n1.ReshapeToUint8s().StoreArray(v1)
}

// lpfTranspose8x8U16Regs transposes an 8x8 matrix of uint16 rows.
func lpfTranspose8x8U16Regs(a, b, c, d, e, f, g, h archsimd.Uint16x8) (o0, o1, o2, o3, o4, o5, o6, o7 archsimd.Uint16x8) {
	ab0, ab1 := a.InterleaveEven(b), a.InterleaveOdd(b)
	cd0, cd1 := c.InterleaveEven(d), c.InterleaveOdd(d)
	ef0, ef1 := e.InterleaveEven(f), e.InterleaveOdd(f)
	gh0, gh1 := g.InterleaveEven(h), g.InterleaveOdd(h)
	ac0, ac2 := ab0.ReshapeToUint32s().InterleaveEven(cd0.ReshapeToUint32s()), ab0.ReshapeToUint32s().InterleaveOdd(cd0.ReshapeToUint32s())
	ac1, ac3 := ab1.ReshapeToUint32s().InterleaveEven(cd1.ReshapeToUint32s()), ab1.ReshapeToUint32s().InterleaveOdd(cd1.ReshapeToUint32s())
	eg0, eg2 := ef0.ReshapeToUint32s().InterleaveEven(gh0.ReshapeToUint32s()), ef0.ReshapeToUint32s().InterleaveOdd(gh0.ReshapeToUint32s())
	eg1, eg3 := ef1.ReshapeToUint32s().InterleaveEven(gh1.ReshapeToUint32s()), ef1.ReshapeToUint32s().InterleaveOdd(gh1.ReshapeToUint32s())
	o0 = ac0.ReshapeToUint64s().InterleaveLo(eg0.ReshapeToUint64s()).ReshapeToUint16s()
	o1 = ac1.ReshapeToUint64s().InterleaveLo(eg1.ReshapeToUint64s()).ReshapeToUint16s()
	o2 = ac2.ReshapeToUint64s().InterleaveLo(eg2.ReshapeToUint64s()).ReshapeToUint16s()
	o3 = ac3.ReshapeToUint64s().InterleaveLo(eg3.ReshapeToUint64s()).ReshapeToUint16s()
	o4 = ac0.ReshapeToUint64s().InterleaveHi(eg0.ReshapeToUint64s()).ReshapeToUint16s()
	o5 = ac1.ReshapeToUint64s().InterleaveHi(eg1.ReshapeToUint64s()).ReshapeToUint16s()
	o6 = ac2.ReshapeToUint64s().InterleaveHi(eg2.ReshapeToUint64s()).ReshapeToUint16s()
	o7 = ac3.ReshapeToUint64s().InterleaveHi(eg3.ReshapeToUint64s()).ReshapeToUint16s()
	return
}

// sbHalf16Limits is the per-group prologue of dav1d's 16bpc
// lpf_{v,h}_sb_{y,uv}_16bpc_neon drivers (src/arm/64/loopfilter16.S
// lpf_func) for one two-unit group: lv holds the units' own levels in lanes 0
// and 1 and their fallback (left or above) levels in lanes 2 and 3, component
// in the low byte. L = cur ? cur : prev; I, E and H follow from L and
// Av1FilterLUT.sharp on byte lanes, are widened to the units' four uint16
// lanes (uxtl) and shifted left by bitdepth_min_8; the "wd >= 4" lane mask is
// the vmask pair restricted to L != 0. ok is false when no selected unit has
// a level.
func sbHalf16Limits(lv archsimd.Uint32x4, pair uint32, sharp0 uint64, sharp1 uint8, bd8 uint64) (eV, iV, hV archsimd.Uint16x8, m4 archsimd.Mask16x8, ok bool) {
	zero32 := archsimd.Uint32x4{}
	prev := lv.ReshapeToUint64s().InterleaveHi(lv.ReshapeToUint64s()).ReshapeToUint32s()
	L := prev.IfElse(lv.Equal(zero32), lv)
	lr := L.Mul(archsimd.BroadcastUint32x4(0x01010101)).ReshapeToUint8s()
	lw := lr.ExtendLo8ToUint16()
	act := archsimd.LoadUint16x8Array(&sbPairLanes16[pair&3]).AndNot(lw.Equal(archsimd.Uint16x8{}).ToInt16x8().ToBits())
	h := act.ReshapeToUint64s()
	if h.GetElem(0)|h.GetElem(1) == 0 {
		return
	}
	hB := lr.ShiftAllRight(4)
	iB := lr.ShiftAllRight(sharp0).Min(archsimd.BroadcastUint8x16(sharp1)).Max(archsimd.BroadcastUint8x16(1))
	e := lr.Add(archsimd.BroadcastUint8x16(2))
	eB := e.Add(e).Add(iB)
	hV = hB.ExtendLo8ToUint16().ShiftAllLeft(bd8)
	iV = iB.ExtendLo8ToUint16().ShiftAllLeft(bd8)
	eV = eB.ExtendLo8ToUint16().ShiftAllLeft(bd8)
	m4 = act.BitsToInt16().ToMask()
	ok = true
	return
}

func sbPairMask16(n uint32) archsimd.Mask16x8 {
	return archsimd.LoadUint16x8Array(&sbPairLanes16[n&3]).BitsToInt16().ToMask()
}

// sbLevels16Row loads two consecutive units' levels and their above
// fallbacks (dav1d 16bpc lpf_v_sb ld1 {v0.8b}/{v1.8b}).
func sbLevels16Row(lvl []uint8, u0, lvlStride, comp int, pair uint32) archsimd.Uint32x4 {
	if li := 4*u0 + comp; u0-lvlStride >= 0 && li+8 <= len(lvl) {
		return sbLevels16Pair(lvl, li, 4*lvlStride)
	}
	var v [2]uint64
	{
		for k := 0; k < 2; k++ {
			if pair&(1<<uint(k)) != 0 {
				v[0] |= uint64(lvl[4*(u0+k)+comp]) << (8*comp + 32*k)
				v[1] |= uint64(lvl[4*(u0+k-lvlStride)+comp]) << (8*comp + 32*k)
			}
		}
	}
	return archsimd.LoadUint64x2Array(&v).ReshapeToUint32s().ShiftAllRight(uint64(8 * comp)).And(archsimd.LoadUint32x4Array(&sbLowByteLanes))
}

// sbLevels16Col loads two vertically stacked units' levels and their left
// fallbacks (dav1d 16bpc lpf_h_sb ld2 of the l[-1], l[0] pairs).
func sbLevels16Col(lvl []uint8, li, lvlStride, comp int, pair uint32) archsimd.Uint32x4 {
	if li-1 >= 0 && 4*(li+lvlStride)+comp+4 <= len(lvl) {
		return sbLevels16Pairs(lvl, 4*li+comp, 4*lvlStride)
	}
	var v [2]uint64
	{
		for k := 0; k < 2; k++ {
			if pair&(1<<uint(k)) != 0 {
				v[0] |= uint64(lvl[4*(li+k*lvlStride)+comp]) << (8*comp + 32*k)
				v[1] |= uint64(lvl[4*(li+k*lvlStride-1)+comp]) << (8*comp + 32*k)
			}
		}
	}
	return archsimd.LoadUint64x2Array(&v).ReshapeToUint32s().ShiftAllRight(uint64(8 * comp)).And(archsimd.LoadUint32x4Array(&sbLowByteLanes))
}

// sbLevels16Pair is component comp of two consecutive cells from the byte
// cursor li (lanes 0-1) and of the two cells ls4 bytes above (lanes 2-3).
func sbLevels16Pair(lvl []uint8, li, ls4 int) archsimd.Uint32x4 {
	v := [2]uint64{binary.LittleEndian.Uint64(lvl[li : li+8]), binary.LittleEndian.Uint64(lvl[li-ls4 : li-ls4+8])}
	return archsimd.LoadUint64x2Array(&v).ReshapeToUint32s().And(archsimd.LoadUint32x4Array(&sbLowByteLanes))
}

// sbLevels16Pairs is component comp of the (l[-1], l[0]) cell pairs of two
// units ls4 bytes apart from the byte cursor li, the units' own levels in
// lanes 0-1 and their left fallbacks in lanes 2-3; one window holds both
// pairs (one compare from 32-bit offsets).
func sbLevels16Pairs(lvl []uint8, li, ls4 int) archsimd.Uint32x4 {
	o, l := int(uint32(li-4)), int(uint32(ls4))
	e := o + l + 8
	x0 := binary.LittleEndian.Uint64(lvl[o:e])
	x1 := binary.LittleEndian.Uint64(lvl[o+l : e])
	v := [2]uint64{x0>>32 | x1&^0xffffffff, x0&0xffffffff | x1<<32}
	return archsimd.LoadUint64x2Array(&v).ReshapeToUint32s().And(archsimd.LoadUint32x4Array(&sbLowByteLanes))
}

// filterSBHorizontal16Edge is the group-by-group driver for calls whose
// samples or levels reach the end of their buffers.
func filterSBHorizontal16Edge(pix []byte, off, stride, rowBytes int, bitDepth uint8, vm [3]uint32, luma bool, lvl []uint8, lvlOff, lvlStride, comp int, lut *SBLUT) {
	bd8 := uint(bitDepth - 8)
	k := &lpf16Tab[bd8>>1]
	gt4All := vm[1] | vm[2]
	all := vm[0] | gt4All
	sharp0, sharp1 := uint64(lut.Sharp[0]), lut.Sharp[1]
	for g := 0; g < 16 && all>>(2*uint(g)) != 0; g++ {
		sh := 2 * uint(g)
		pair := (all >> sh) & 3
		if pair == 0 {
			continue
		}
		nGt4 := (gt4All >> sh) & 3
		n16 := (vm[2] >> sh) & 3
		wd := sbWidth(nGt4, n16, luma)
		u0 := lvlOff + 2*g
		q0Off := off + 16*g
		reach := 4
		if wd == 16 {
			reach = 7
		}
		if 16*g+16 > rowBytes || q0Off-reach*stride < 0 || q0Off+(reach-1)*stride+16 > len(pix) {
			sub := vm
			for k := range sub {
				sub[k] = (sub[k] >> sh) & 3
			}
			filterSBHorizontal16Scalar(pix, q0Off, stride, bitDepth, sub, luma, lvl, u0, lvlStride, comp, lut)
			continue
		}
		eV, iV, hV, m4, ok := sbHalf16Limits(sbLevels16Row(lvl, u0, lvlStride, comp, pair), pair, sharp0, sharp1, uint64(bd8))
		if !ok {
			continue
		}
		lpf8U16(pix, q0Off, stride, wd, k, eV, iV, hV, m4, sbPairMask16(nGt4), sbPairMask16(n16))
	}
}

// filterSBVertical16Edge is filterSBHorizontal16Edge for vertical edges.
func filterSBVertical16Edge(pix []byte, off, stride, rowBytes int, bitDepth uint8, vm [3]uint32, luma bool, lvl []uint8, lvlOff, lvlStride, comp int, lut *SBLUT) {
	bd8 := uint(bitDepth - 8)
	k := &lpf16Tab[bd8>>1]
	gt4All := vm[1] | vm[2]
	all := vm[0] | gt4All
	sharp0, sharp1 := uint64(lut.Sharp[0]), lut.Sharp[1]
	for g := 0; g < 16 && all>>(2*uint(g)) != 0; g++ {
		sh := 2 * uint(g)
		pair := (all >> sh) & 3
		if pair == 0 {
			continue
		}
		nGt4 := (gt4All >> sh) & 3
		n16 := (vm[2] >> sh) & 3
		wd := sbWidth(nGt4, n16, luma)
		u0 := lvlOff + 2*g*lvlStride
		q0Off := off + 8*g*stride
		lastRow := q0Off + 7*stride
		reach := 8 // bytes on each side: p3..q3
		if wd == 16 {
			reach = 16 // p7..q7
		}
		if rowBytes < reach || q0Off-reach < 0 || lastRow+reach > len(pix) {
			sub := vm
			for k := range sub {
				sub[k] = (sub[k] >> sh) & 3
			}
			filterSBVertical16Scalar(pix, q0Off, stride, bitDepth, sub, luma, lvl, u0, lvlStride, comp, lut)
			continue
		}
		eV, iV, hV, m4, ok := sbHalf16Limits(sbLevels16Col(lvl, u0, lvlStride, comp, pair), pair, sharp0, sharp1, uint64(bd8))
		if !ok {
			continue
		}
		lpfCols16(pix, q0Off, stride, wd, k, eV, iV, hV, m4, sbPairMask16(nGt4), sbPairMask16(n16))
	}
}

// sbLimits16 is sbHalf16Limits for the dav1d-shaped drivers: lv holds the two
// units' own levels in lanes 0 and 1 and their fallbacks in lanes 2 and 3,
// already in the low byte of each lane; I and E come from the LUT's sharp
// vectors and the widened limits shift left by bitdepth_min_8 through one
// count vector.
func sbLimits16(lv archsimd.Uint32x4, pair uint32, lut *SBLUT, bd8 int16) (eV, iV, hV archsimd.Uint16x8, m4 archsimd.Mask16x8, ok bool) {
	one := archsimd.BroadcastUint8x16(1)
	prev := lv.ReshapeToUint64s().InterleaveHi(lv.ReshapeToUint64s()).ReshapeToUint32s()
	L := prev.IfElse(lv.Equal(archsimd.Uint32x4{}), lv)
	lr := L.Mul(one.ReshapeToUint32s()).ReshapeToUint8s()
	lw := lr.ExtendLo8ToUint16()
	act := archsimd.LoadUint16x8Array(&sbPairLanes16[pair&3]).AndNot(lw.Equal(archsimd.Uint16x8{}).ToInt16x8().ToBits())
	h := act.ReshapeToUint64s()
	if h.GetElem(0)|h.GetElem(1) == 0 {
		return
	}
	hB := lr.ShiftAllRight(4)
	iB := lr.Shift(archsimd.LoadInt8x16Array(&lut.SharpShift)).Min(archsimd.LoadUint8x16Array(&lut.SharpCap)).Max(one)
	e := lr.Add(archsimd.BroadcastUint8x16(2))
	eB := e.Add(e).Add(iB)
	bdV := archsimd.BroadcastInt16x8(bd8)
	hV = hB.ExtendLo8ToUint16().Shift(bdV)
	iV = iB.ExtendLo8ToUint16().Shift(bdV)
	eV = eB.ExtendLo8ToUint16().Shift(bdV)
	m4 = act.BitsToInt16().ToMask()
	ok = true
	return
}

// sbWidth16 is the kernel width from a group's raw width-mask pairs.
func sbWidth16(vm1, vm2 uint32, luma bool) int {
	if vm2&3 != 0 {
		return 16
	}
	if vm1&3 != 0 {
		if luma {
			return 8
		}
		return 6
	}
	return 4
}

// filterSBHorizontal16 is dav1d lpf_v_sb_{y,uv}_16bpc_neon (two units per
// group): the width masks shift down one pair per group while the sample and
// level cursors advance, and each group's levels are one 8-byte load each of
// l[0] and l[-b4_stride] from the component-offset level cursor. The call's
// reach is checked once; calls touching a buffer end take the edge driver.
func filterSBHorizontal16(pix []byte, off, stride, rowBytes int, bitDepth uint8, vm [3]uint32, luma bool, lvl []uint8, lvlOff, lvlStride, comp int, lut *SBLUT) {
	vm0, vm1, vm2 := vm[0]|vm[1]|vm[2], vm[1]|vm[2], vm[2]
	end := 16 * ((31-bits.LeadingZeros32(vm0))>>1 + 1)
	reach := 4
	if vm2 != 0 {
		reach = 7
	}
	li, ls4 := 4*lvlOff+comp, 4*lvlStride
	if len(pix) > lpfMaxPlane {
		filterSBHorizontal16Scalar(pix, off, stride, bitDepth, vm, luma, lvl, lvlOff, lvlStride, comp, lut)
		return
	}
	if end > rowBytes || off < reach*stride || off+(reach-1)*stride+end > len(pix) || li < ls4 || li+end/2 > len(lvl) {
		filterSBHorizontal16Edge(pix, off, stride, rowBytes, bitDepth, vm, luma, lvl, lvlOff, lvlStride, comp, lut)
		return
	}
	bd8 := uint(bitDepth - 8)
	k := &lpf16Tab[bd8>>1]
	for q0 := off; vm0 != 0; vm0, vm1, vm2, q0, li = vm0>>2, vm1>>2, vm2>>2, q0+16, li+8 {
		if vm0&3 == 0 {
			continue
		}
		lv := sbLevels16Pair(lvl, li, ls4)
		eV, iV, hV, m4, ok := sbLimits16(lv, vm0, lut, int16(bd8))
		if !ok {
			continue
		}
		lpf8U16(pix, q0, stride, sbWidth16(vm1, vm2, luma), k, eV, iV, hV, m4, sbPairMask16(vm1), sbPairMask16(vm2))
	}
}

// filterSBVertical16 is dav1d lpf_h_sb_{y,uv}_16bpc_neon in the same shape:
// each group reads the (l[-1], l[0]) level pairs of its two units.
func filterSBVertical16(pix []byte, off, stride, rowBytes int, bitDepth uint8, vm [3]uint32, luma bool, lvl []uint8, lvlOff, lvlStride, comp int, lut *SBLUT) {
	vm0, vm1, vm2 := vm[0]|vm[1]|vm[2], vm[1]|vm[2], vm[2]
	hiG := (31 - bits.LeadingZeros32(vm0)) >> 1
	reach := 8 // bytes on each side: p3..q3
	if vm2 != 0 {
		reach = 16 // p7..q7
	}
	li, ls4 := 4*lvlOff+comp, 4*lvlStride
	if len(pix) > lpfMaxPlane {
		filterSBVertical16Scalar(pix, off, stride, bitDepth, vm, luma, lvl, lvlOff, lvlStride, comp, lut)
		return
	}
	if rowBytes < reach || off < reach || off+(8*hiG+7)*stride+reach > len(pix) || li < 4 || li+(2*hiG+1)*ls4+4 > len(lvl) {
		filterSBVertical16Edge(pix, off, stride, rowBytes, bitDepth, vm, luma, lvl, lvlOff, lvlStride, comp, lut)
		return
	}
	bd8 := uint(bitDepth - 8)
	k := &lpf16Tab[bd8>>1]
	stride8 := 8 * stride
	for q0 := off; vm0 != 0; vm0, vm1, vm2, q0, li = vm0>>2, vm1>>2, vm2>>2, q0+stride8, li+2*ls4 {
		if vm0&3 == 0 {
			continue
		}
		lv := sbLevels16Pairs(lvl, li, ls4)
		eV, iV, hV, m4, ok := sbLimits16(lv, vm0, lut, int16(bd8))
		if !ok {
			continue
		}
		lpfCols16(pix, q0, stride, sbWidth16(vm1, vm2, luma), k, eV, iV, hV, m4, sbPairMask16(vm1), sbPairMask16(vm2))
	}
}
