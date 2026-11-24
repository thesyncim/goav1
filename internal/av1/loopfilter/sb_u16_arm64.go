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

// The horizontal-edge drivers check each call's (or edge group's) sample
// and level footprints once, as unsafeutil.Rows views, and hand each group's
// tap rows to the row kernel as an unsafeutil.Block, exactly as the 8-bit
// drivers do (sb_u8_arm64.go). The vertical-edge kernels (lpfCols16) take
// checked row windows.

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

func lpf16Any(m archsimd.Mask16x8) bool {
	b := m.ToInt16x8().ToBits().ReshapeToUint64s()
	return b.GetElem(0)|b.GetElem(1) != 0
}

// lpf8U16W is the high-bit-depth counterpart of lpf16U8Wd16/lpf8U8W on eight
// uint16 lanes (two 4-sample units): dav1d's loop_filter() with E/I/H and the
// flatness threshold scaled by bitdepth_min_8 (the bitdepth's lpf16Tab row
// k), filter4 in signed 16-bit lanes with the iclip_diff/iclip_pixel bounds,
// and the flat filters' sums held exactly in uint16 lanes (at most
// 16*4095+8). The tap rows are tv's: R's row count picks the width (4, 6 and
// 8 rows run wd 4, 6 and 8, fourteen rows p6..q6 run wd 16), q0 is row
// len(R)/2, and every row is named by a constant index at its load and its
// store (no checks, no inline marks). Vertical edges run lpfCols16.
func lpf8U16W[R unsafeutil.BlockRows](tv unsafeutil.Block[R], k *lpf16Consts, eV, iV, hV archsimd.Uint16x8, m4, mGt4, m16 archsimd.Mask16x8) {
	var rows R
	n := len(rows)
	wd := n
	if n == 14 {
		wd = 16
	}
	q := n / 2 // row of q0
	p1 := archsimd.LoadUint8x16Array(tv.Row16(q - 2)).ReshapeToUint16s()
	p0 := archsimd.LoadUint8x16Array(tv.Row16(q - 1)).ReshapeToUint16s()
	q0 := archsimd.LoadUint8x16Array(tv.Row16(q)).ReshapeToUint16s()
	q1 := archsimd.LoadUint8x16Array(tv.Row16(q + 1)).ReshapeToUint16s()
	maxd := lpf16Abd(p1, p0).Max(lpf16Abd(q1, q0))
	t := maxd
	var p2, q2, p3, q3 archsimd.Uint16x8
	if wd >= 6 {
		p2 = archsimd.LoadUint8x16Array(tv.Row16(q - 3)).ReshapeToUint16s()
		q2 = archsimd.LoadUint8x16Array(tv.Row16(q + 2)).ReshapeToUint16s()
		tt := lpf16Abd(p2, p1).Max(lpf16Abd(q2, q1))
		if wd >= 8 {
			p3 = archsimd.LoadUint8x16Array(tv.Row16(q - 4)).ReshapeToUint16s()
			q3 = archsimd.LoadUint8x16Array(tv.Row16(q + 3)).ReshapeToUint16s()
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
	if wd >= 6 {
		fl := lpf16Abd(p2, p0).Max(maxd).Max(lpf16Abd(q2, q0))
		if wd >= 8 {
			fl = fl.Max(lpf16Abd(p3, p0)).Max(lpf16Abd(q3, q0))
		}
		mFlat = fV.GreaterEqual(fl).And(mGt4).And(fm)
		mF4 = fm.And(mFlat.Not())
		if wd == 16 {
			p6 = archsimd.LoadUint8x16Array(tv.Row16(q - 7)).ReshapeToUint16s()
			p5 = archsimd.LoadUint8x16Array(tv.Row16(q - 6)).ReshapeToUint16s()
			p4 = archsimd.LoadUint8x16Array(tv.Row16(q - 5)).ReshapeToUint16s()
			q4 = archsimd.LoadUint8x16Array(tv.Row16(q + 4)).ReshapeToUint16s()
			q5 = archsimd.LoadUint8x16Array(tv.Row16(q + 5)).ReshapeToUint16s()
			q6 = archsimd.LoadUint8x16Array(tv.Row16(q + 6)).ReshapeToUint16s()
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
		o1.ReshapeToUint8s().StoreArray(tv.Row16(q - 2))
		o0.ReshapeToUint8s().StoreArray(tv.Row16(q - 1))
		n0.ReshapeToUint8s().StoreArray(tv.Row16(q))
		n1.ReshapeToUint8s().StoreArray(tv.Row16(q + 1))
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
		s.Shift(archsimd.LoadInt16x8Array(&k.shr4)).IfElse(m, p5).ReshapeToUint8s().StoreArray(tv.Row16(q - 6))
		s = s.Add(p3.Add(q1)).Sub(p6x2)
		s.Shift(archsimd.LoadInt16x8Array(&k.shr4)).IfElse(m, p4).ReshapeToUint8s().StoreArray(tv.Row16(q - 5))
		s = s.Add(p2.Add(q2)).Sub(p6.Add(p5))
		s.Shift(archsimd.LoadInt16x8Array(&k.shr4)).IfElse(m, p3).ReshapeToUint8s().StoreArray(tv.Row16(q - 4))
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
		s.Shift(archsimd.LoadInt16x8Array(&k.shr4)).IfElse(m, q3).ReshapeToUint8s().StoreArray(tv.Row16(q + 3))
		s = s.Add(q5.Add(q6)).Sub(p2.Add(q2))
		s.Shift(archsimd.LoadInt16x8Array(&k.shr4)).IfElse(m, q4).ReshapeToUint8s().StoreArray(tv.Row16(q + 4))
		s = s.Add(q6.Add(q6)).Sub(p1.Add(q3))
		s.Shift(archsimd.LoadInt16x8Array(&k.shr4)).IfElse(m, q5).ReshapeToUint8s().StoreArray(tv.Row16(q + 5))
	}
	if wd >= 8 {
		o2.ReshapeToUint8s().StoreArray(tv.Row16(q - 3))
		n2.ReshapeToUint8s().StoreArray(tv.Row16(q + 2))
	}
	o1.ReshapeToUint8s().StoreArray(tv.Row16(q - 2))
	o0.ReshapeToUint8s().StoreArray(tv.Row16(q - 1))
	n0.ReshapeToUint8s().StoreArray(tv.Row16(q))
	n1.ReshapeToUint8s().StoreArray(tv.Row16(q + 1))
}

// lpfRowsU16 runs the width-wd row kernel on the tap rows of gv (the wd/2
// rows each side of the edge, seven at wd 16): the edge driver's choice.
func lpfRowsU16(gv unsafeutil.Rows, wd int, k *lpf16Consts, eV, iV, hV archsimd.Uint16x8, m4, mGt4, m16 archsimd.Mask16x8) {
	switch wd {
	case 16:
		lpf8U16W(unsafeutil.BlockAt[lpfR14](gv, 0, 0), k, eV, iV, hV, m4, mGt4, m16)
	case 8:
		lpf8U16W(unsafeutil.BlockAt[lpfR8](gv, 0, 0), k, eV, iV, hV, m4, mGt4, m16)
	case 6:
		lpf8U16W(unsafeutil.BlockAt[lpfR6](gv, 0, 0), k, eV, iV, hV, m4, mGt4, m16)
	default:
		lpf8U16W(unsafeutil.BlockAt[lpfR4](gv, 0, 0), k, eV, iV, hV, m4, mGt4, m16)
	}
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
	m4 = archsimd.Int16x8{}.Greater(act.BitsToInt16())
	ok = true
	return
}

func sbPairMask16(n uint32) archsimd.Mask16x8 {
	return archsimd.Int16x8{}.Greater(archsimd.LoadUint16x8Array(&sbPairLanes16[n&3]).BitsToInt16())
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
		// The group's tap rows: the wd/2 rows each side of the edge (seven
		// at wd 16), sixteen bytes each.
		rh := wd / 2
		if wd == 16 {
			rh = 7
		}
		if 16*g+16 > rowBytes || !unsafeutil.RowsFit(len(pix), q0Off-rh*stride, stride, 16, 2*rh) {
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
		lpfRowsU16(unsafeutil.NewRows(pix, q0Off-rh*stride, stride, 16, 2*rh), wd, k, eV, iV, hV, m4, sbPairMask16(nGt4), sbPairMask16(n16))
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
		reach := 8 // bytes on each side: p3..q3
		if wd == 16 {
			reach = 16 // p7..q7
		}
		if rowBytes < reach || !unsafeutil.RowsFit(len(pix), q0Off-reach, stride, 2*reach, 8) {
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
		lpfCols16(unsafeutil.NewRows(pix, q0Off-reach, stride, 2*reach, 8), 0, reach, wd, k, eV, iV, hV, m4, sbPairMask16(nGt4), sbPairMask16(n16))
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
	m4 = archsimd.Int16x8{}.Greater(act.BitsToInt16())
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
// reach is checked once, as one view of its samples and one of its levels
// (sbRowsU16); calls touching a buffer end take the edge driver.
func filterSBHorizontal16(pix []byte, off, stride, rowBytes int, bitDepth uint8, vm [3]uint32, luma bool, lvl []uint8, lvlOff, lvlStride, comp int, lut *SBLUT) {
	vm0, vm1, vm2 := vm[0]|vm[1]|vm[2], vm[1]|vm[2], vm[2]
	end := 16 * ((31-bits.LeadingZeros32(vm0))>>1 + 1)
	li, ls4 := 4*lvlOff+comp, 4*lvlStride
	if len(pix) > lpfMaxPlane {
		filterSBHorizontal16Scalar(pix, off, stride, bitDepth, vm, luma, lvl, lvlOff, lvlStride, comp, lut)
		return
	}
	if end > rowBytes {
		filterSBHorizontal16Edge(pix, off, stride, rowBytes, bitDepth, vm, luma, lvl, lvlOff, lvlStride, comp, lut)
		return
	}
	k := &lpf16Tab[uint(bitDepth-8)>>1]
	var ok bool
	if vm2 != 0 {
		ok = sbRowsU16[lpfW16](pix, off, stride, end, vm0, vm1, vm2, luma, lvl, li, ls4, lut, k, int16(bitDepth-8))
	} else {
		ok = sbRowsU16[lpfW8](pix, off, stride, end, vm0, vm1, 0, luma, lvl, li, ls4, lut, k, int16(bitDepth-8))
	}
	if !ok {
		filterSBHorizontal16Edge(pix, off, stride, rowBytes, bitDepth, vm, luma, lvl, lvlOff, lvlStride, comp, lut)
	}
}

// sbRowsU16 is filterSBHorizontal16's group loop for calls whose widest group
// is W (lpfW16, or lpfW8 for every narrower call): one view of the call's
// tap rows (the four rows each side of the edge at off, seven at wd 16, n
// bytes each) and one of its two level rows (l[0] at li and l[-b4_stride] ls4
// bytes before it, n/2 bytes) replace the reach checks; every group's
// sixteen-byte (two-unit) block is then a column of the sample view and an
// eight-byte column of the level view, and the narrower kernels of a wider
// call take its inner rows. It reports false, having filtered nothing, when
// either view leaves its slice.
func sbRowsU16[W lpfWidth](pix []byte, off, stride, n int, vm0, vm1, vm2 uint32, luma bool, lvl []uint8, li, ls4 int, lut *SBLUT, k *lpf16Consts, bd8 int16) bool {
	var wdT W
	rh := 4
	if len(wdT) == 16 {
		rh = 7
	}
	if !unsafeutil.RowsFit(len(pix), off-rh*stride, stride, n, 2*rh) || !unsafeutil.RowsFit(len(lvl), li-ls4, ls4, n/2, 2) {
		return false
	}
	v := unsafeutil.NewRows(pix, off-rh*stride, stride, n, 2*rh)
	lv := unsafeutil.NewRows(lvl, li-ls4, ls4, n/2, 2)
	for x := 0; vm0 != 0 && x <= v.Width()-16; vm0, vm1, vm2, x = vm0>>2, vm1>>2, vm2>>2, x+16 {
		if vm0&3 == 0 {
			continue
		}
		// The group's (l[0], l[-b4_stride]) cell pairs, as sbLevels16Pair.
		lp := [2]uint64{binary.LittleEndian.Uint64(lv.Row8(1, x/2)[:]), binary.LittleEndian.Uint64(lv.Row8(0, x/2)[:])}
		lw := archsimd.LoadUint64x2Array(&lp).ReshapeToUint32s().And(archsimd.LoadUint32x4Array(&sbLowByteLanes))
		if eV, iV, hV, m4, ok := sbLimits16(lw, vm0, lut, bd8); !ok {
			continue
		} else if len(wdT) == 16 && vm2&3 != 0 {
			lpf8U16W(unsafeutil.BlockAt[lpfR14](v, 0, x), k, eV, iV, hV, m4, sbPairMask16(vm1), sbPairMask16(vm2))
		} else if vm1&3 == 0 {
			lpf8U16W(unsafeutil.BlockAt[lpfR4](v, rh-2, x), k, eV, iV, hV, m4, archsimd.Mask16x8{}, archsimd.Mask16x8{})
		} else if luma {
			lpf8U16W(unsafeutil.BlockAt[lpfR8](v, rh-4, x), k, eV, iV, hV, m4, sbPairMask16(vm1), archsimd.Mask16x8{})
		} else {
			lpf8U16W(unsafeutil.BlockAt[lpfR6](v, rh-3, x), k, eV, iV, hV, m4, sbPairMask16(vm1), archsimd.Mask16x8{})
		}
	}
	return true
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
	// One view of the call's samples (its 8*(hiG+1) rows, reach bytes each
	// side of the edge) and one of its level pairs (the (l[-1], l[0]) cell
	// pairs of its 2*(hiG+1) units) replace the reach checks; each group is
	// then an eight-row block of the sample view and two rows of the level
	// view.
	if rowBytes < reach || !unsafeutil.RowsFit(len(pix), off-reach, stride, 2*reach, 8*hiG+8) || !unsafeutil.RowsFit(len(lvl), li-4, ls4, 8, 2*hiG+2) {
		filterSBVertical16Edge(pix, off, stride, rowBytes, bitDepth, vm, luma, lvl, lvlOff, lvlStride, comp, lut)
		return
	}
	v := unsafeutil.NewRows(pix, off-reach, stride, 2*reach, 8*hiG+8)
	lv := unsafeutil.NewRows(lvl, li-4, ls4, 8, 2*hiG+2)
	bd8 := int16(bitDepth - 8)
	k := &lpf16Tab[bd8>>1]
	for g := 0; vm0 != 0; vm0, vm1, vm2, g = vm0>>2, vm1>>2, vm2>>2, g+1 {
		if vm0&3 == 0 {
			continue
		}
		// sbLevels16Pairs on the group's two level rows.
		x0, x1 := binary.LittleEndian.Uint64(lv.Row8(2*g, 0)[:]), binary.LittleEndian.Uint64(lv.Row8(2*g+1, 0)[:])
		lp := [2]uint64{x0>>32 | x1&^0xffffffff, x0&0xffffffff | x1<<32}
		lw := archsimd.LoadUint64x2Array(&lp).ReshapeToUint32s().And(archsimd.LoadUint32x4Array(&sbLowByteLanes))
		if eV, iV, hV, m4, ok := sbLimits16(lw, vm0, lut, bd8); ok {
			lpfCols16(v, 8*g, reach, sbWidth16(vm1, vm2, luma), k, eV, iV, hV, m4, sbPairMask16(vm1), sbPairMask16(vm2))
		}
	}
}
