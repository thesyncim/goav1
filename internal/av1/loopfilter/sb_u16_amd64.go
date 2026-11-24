// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && amd64 && !purego

package loopfilter

import (
	"encoding/binary"
	"math/bits"
	"simd/archsimd"

	"github.com/thesyncim/goav1/internal/unsafeutil"
)

// Plane rows are bounds-checked windows from 32-bit offsets, as in
// sb_u8_amd64.go: one compare per row.

// This file ports dav1d's 16bpc AVX2 superblock deblock drivers and FILTER
// macro (src/x86/loopfilter16_avx2.asm lpf_{v,h}_sb_{y,uv}_16bpc) to
// archsimd. One kernel call covers sixteen uint16 samples, four 4x4 units (a
// vmask nibble): horizontal edges filter the rows straight from the plane,
// vertical edges transpose sixteen rows (0..7 in the low 128-bit lane, 8..15
// in the high one) with 8x8 word transposes and back. E/I/H and the flatness
// threshold are scaled by bitdepth_min_8; the narrow filter runs in signed
// 16-bit lanes with the iclip_diff/iclip_pixel bounds, and the flat filters'
// sums fit uint16 exactly (at most 16*4095+8).

// avx16Nibble[n] selects the four uint16 lanes of every unit in the four-unit
// group whose bit is set in n (dav1d 16bpc pb_mask + pcmpeqd).
var avx16Nibble = func() (t [16][16]uint16) {
	for n := range t {
		for k := 0; k < 4; k++ {
			if n&(1<<k) != 0 {
				for i := 0; i < 4; i++ {
					t[n][4*k+i] = 0xffff
				}
			}
		}
	}
	return t
}()

// avx16LevelSel16[c] moves component c of the four units (sixteen level
// bytes, present in both 128-bit lanes) into the low byte of each unit's four
// uint16 lanes, zeroing the high bytes: units 0-1 in lane 0, 2-3 in lane 1.
var avx16LevelSel16 = func() (t [4][32]int8) {
	for c := range t {
		for j := 0; j < 16; j++ {
			t[c][2*j] = int8(4*(j/4) + c)
			t[c][2*j+1] = -1
		}
	}
	return t
}()

// avx16K holds a call's bit-depth constants.
type avx16K struct {
	flat, dmin, dmax, maxPix archsimd.Int16x16
}

func newAVX16K(bd8 uint) avx16K {
	return avx16K{
		flat:   archsimd.BroadcastInt16x16(int16(1) << bd8),
		dmin:   archsimd.BroadcastInt16x16(int16(-128 << bd8)),
		dmax:   archsimd.BroadcastInt16x16(int16(128<<bd8 - 1)),
		maxPix: archsimd.BroadcastInt16x16(int16(256<<bd8 - 1)),
	}
}

func avx16Absd(a, b archsimd.Uint16x16) archsimd.Uint16x16 {
	return a.SubSaturated(b).Or(b.SubSaturated(a))
}

// avx16LE returns all-ones lanes where x <= lim (unsigned).
func avx16LE(x, lim archsimd.Uint16x16) archsimd.Uint16x16 {
	return x.SubSaturated(lim).Equal(archsimd.Uint16x16{}).ToInt16x16().ToBits()
}

// avx16Sel returns a where m is all-ones and b elsewhere (VPBLENDVB).
func avx16Sel(m, a, b archsimd.Uint16x16) archsimd.Uint16x16 {
	return a.IfElse(archsimd.Int16x16{}.Greater(m.BitsToInt16()), b)
}

// avx16Core is dav1d 16bpc FILTER for widths 4, 6 and 8 on sixteen lanes in
// registers: p3..q3 in, the filtered p2..q2 out. mFlat returns the lanes the
// flat filter took; ok is false when no lane passed the filter mask.
func avx16Core[W avxWidth](k *avx16K, p3, p2, p1, p0, q0, q1, q2, q3, eV, iV, hV, m4, mGt4 archsimd.Uint16x16) (o2, o1, o0, n0, n1, n2, mFlat archsimd.Uint16x16, ok bool) {
	var wdT W
	wd := len(wdT)
	o2, o1, o0, n0, n1, n2 = p2, p1, p0, q0, q1, q2
	maxd := avx16Absd(p1, p0).Max(avx16Absd(q1, q0))
	t := maxd
	if wd >= 6 {
		tt := avx16Absd(p2, p1).Max(avx16Absd(q2, q1))
		if wd >= 8 {
			tt = tt.Max(avx16Absd(p3, p2)).Max(avx16Absd(q3, q2))
		}
		t = t.Max(tt.And(mGt4))
	}
	d00 := avx16Absd(p0, q0)
	sum := d00.Add(d00).Add(avx16Absd(p1, q1).ShiftAllRight(1))
	fm := t.SubSaturated(iV).Or(sum.SubSaturated(eV)).Equal(archsimd.Uint16x16{}).ToInt16x16().ToBits().And(m4)
	if fm.IsZero() {
		return
	}
	ok = true
	f4 := fm
	if wd >= 6 {
		fl := avx16Absd(p2, p0).Max(maxd).Max(avx16Absd(q2, q0))
		if wd >= 8 {
			fl = fl.Max(avx16Absd(p3, p0)).Max(avx16Absd(q3, q0))
		}
		mFlat = avx16LE(fl, k.flat.ToBits()).And(mGt4).And(fm)
		f4 = fm.AndNot(mFlat)
	}
	if !f4.IsZero() {
		notHev := avx16LE(maxd, hV)
		var zero archsimd.Int16x16
		p1s, p0s, q0s, q1s := p1.BitsToInt16(), p0.BitsToInt16(), q0.BitsToInt16(), q1.BitsToInt16()
		f := p1s.Sub(q1s).Max(k.dmin).Min(k.dmax).AndNot(notHev.BitsToInt16())
		f = q0s.Sub(p0s).Mul(archsimd.BroadcastInt16x16(3)).Add(f).Max(k.dmin).Min(k.dmax).And(f4.BitsToInt16())
		f1 := f.Add(archsimd.BroadcastInt16x16(4)).Min(k.dmax).ShiftAllRight(3)
		f2 := f.Add(archsimd.BroadcastInt16x16(3)).Min(k.dmax).ShiftAllRight(3)
		o0 = p0s.Add(f2).Max(zero).Min(k.maxPix).ToBits()
		n0 = q0s.Sub(f1).Max(zero).Min(k.maxPix).ToBits()
		f3 := f1.Add(archsimd.BroadcastInt16x16(1)).ShiftAllRight(1).And(notHev.BitsToInt16())
		o1 = p1s.Add(f3).Max(zero).Min(k.maxPix).ToBits()
		n1 = q1s.Sub(f3).Max(zero).Min(k.maxPix).ToBits()
	}
	if wd < 6 || mFlat.IsZero() {
		return
	}
	four := archsimd.BroadcastUint16x16(4)
	if wd == 6 {
		s := p2.Add(p2).Add(p2).Add(p1.Add(p0).Add(p1.Add(p0))).Add(q0).Add(four)
		o1 = avx16Sel(mFlat, s.ShiftAllRight(3), o1)
		s = s.Add(q0.Add(q1)).Sub(p2.Add(p2))
		o0 = avx16Sel(mFlat, s.ShiftAllRight(3), o0)
		s = s.Add(q1.Add(q2)).Sub(p2.Add(p1))
		n0 = avx16Sel(mFlat, s.ShiftAllRight(3), n0)
		s = s.Add(q2.Add(q2)).Sub(p1.Add(p0))
		n1 = avx16Sel(mFlat, s.ShiftAllRight(3), n1)
		return
	}
	s := p3.Add(p3).Add(p3).Add(p2.Add(p2)).Add(p1.Add(p0)).Add(q0.Add(four))
	o2 = avx16Sel(mFlat, s.ShiftAllRight(3), o2)
	s = s.Add(p1.Add(q1)).Sub(p3.Add(p2))
	o1 = avx16Sel(mFlat, s.ShiftAllRight(3), o1)
	s = s.Add(p0.Add(q2)).Sub(p3.Add(p1))
	o0 = avx16Sel(mFlat, s.ShiftAllRight(3), o0)
	s = s.Add(q0.Add(q3)).Sub(p3.Add(p0))
	n0 = avx16Sel(mFlat, s.ShiftAllRight(3), n0)
	s = s.Add(q1.Add(q3)).Sub(p2.Add(q0))
	n1 = avx16Sel(mFlat, s.ShiftAllRight(3), n1)
	s = s.Add(q2.Add(q3)).Sub(p1.Add(q1))
	n2 = avx16Sel(mFlat, s.ShiftAllRight(3), n2)
	return
}

// avx16Lpf16 is 16bpc FILTER 16 on taps t[0..15] = p7..q7: the width-8 core,
// then the 13-tap flat filter on the lanes whose own width is 16 and whose
// outer taps are flat too. The filtered p5..q5 replace t[2..13].
func avx16Lpf16(k *avx16K, t *[16]archsimd.Uint16x16, eV, iV, hV, m4, mGt4, m16 archsimd.Uint16x16) bool {
	o2, o1, o0, n0, n1, n2, mFlat, ok := avx16Core[avxW8](k, t[4], t[5], t[6], t[7], t[8], t[9], t[10], t[11], eV, iV, hV, m4, mGt4)
	if !ok {
		return false
	}
	m := mFlat.And(m16)
	if !m.IsZero() {
		p0, q0 := t[7], t[8]
		fo := avx16Absd(t[1], p0).Max(avx16Absd(t[2], p0)).Max(avx16Absd(t[3], p0)).
			Max(avx16Absd(t[12], q0)).Max(avx16Absd(t[13], q0)).Max(avx16Absd(t[14], q0))
		m = m.And(avx16LE(fo, k.flat.ToBits()))
	}
	if !m.IsZero() {
		p6, p5, p4, p3, p2, p1, p0 := t[1], t[2], t[3], t[4], t[5], t[6], t[7]
		q0, q1, q2, q3, q4, q5, q6 := t[8], t[9], t[10], t[11], t[12], t[13], t[14]
		p6x2 := p6.Add(p6)
		s := p6.Add(p6x2).Add(p6x2.Add(p6x2))
		s = s.Add(p5.Add(p5)).Add(p4.Add(p4)).Add(p3.Add(p2)).Add(p1.Add(p0)).Add(q0).Add(archsimd.BroadcastUint16x16(8))
		t[2] = avx16Sel(m, s.ShiftAllRight(4), p5)
		s = s.Add(p3.Add(q1)).Sub(p6x2)
		t[3] = avx16Sel(m, s.ShiftAllRight(4), p4)
		s = s.Add(p2.Add(q2)).Sub(p6.Add(p5))
		t[4] = avx16Sel(m, s.ShiftAllRight(4), p3)
		s = s.Add(p1.Add(q3)).Sub(p6.Add(p4))
		o2 = avx16Sel(m, s.ShiftAllRight(4), o2)
		s = s.Add(p0.Add(q4)).Sub(p6.Add(p3))
		o1 = avx16Sel(m, s.ShiftAllRight(4), o1)
		s = s.Add(q0.Add(q5)).Sub(p6.Add(p2))
		o0 = avx16Sel(m, s.ShiftAllRight(4), o0)
		s = s.Add(q1.Add(q6)).Sub(p6.Add(p1))
		n0 = avx16Sel(m, s.ShiftAllRight(4), n0)
		s = s.Add(q2.Add(q6)).Sub(p5.Add(p0))
		n1 = avx16Sel(m, s.ShiftAllRight(4), n1)
		s = s.Add(q3.Add(q6)).Sub(p4.Add(q0))
		n2 = avx16Sel(m, s.ShiftAllRight(4), n2)
		s = s.Add(q4.Add(q6)).Sub(p3.Add(q1))
		t[11] = avx16Sel(m, s.ShiftAllRight(4), q3)
		s = s.Add(q5.Add(q6)).Sub(p2.Add(q2))
		t[12] = avx16Sel(m, s.ShiftAllRight(4), q4)
		s = s.Add(q6.Add(q6)).Sub(p1.Add(q3))
		t[13] = avx16Sel(m, s.ShiftAllRight(4), q5)
	}
	t[5], t[6], t[7], t[8], t[9], t[10] = o2, o1, o0, n0, n1, n2
	return true
}

// avx16GroupLimits is the 16bpc FILTER prologue for one four-unit group: lv
// holds the units' own levels (four components per unit) in both 128-bit
// lanes and pv their fallbacks. L = cur ? cur : prev for component comp lands
// in each unit's four uint16 lanes (dav1d pshufb pb_4x1_4x5_4x9_4x13), and
// I = imax(imin(L >> sharp[0], sharp[1]), 1), E = 2*(L+2)+I and H = L>>4 are
// computed there and shifted left by bitdepth_min_8 (2 or 4). Everything
// stays in 256-bit registers: 128-bit values are moved with legacy-SSE MOVUPS
// by the compiler. m4 is the vmask nibble restricted to L != 0.
func avx16GroupLimits(lv, pv archsimd.Uint8x32, sel archsimd.Int8x32, nAll uint32, sharp0 uint64, sharp1 archsimd.Uint16x16, bd8 uint) (eV, iV, hV, m4 archsimd.Uint16x16, ok bool) {
	var zero archsimd.Uint16x16
	c := lv.PermuteOrZeroGrouped(sel).ReshapeToUint16s()
	p := pv.PermuteOrZeroGrouped(sel).ReshapeToUint16s()
	L := c.Or(p.And(c.Equal(zero).ToInt16x16().ToBits()))
	act := avx16Mask(nAll).AndNot(L.Equal(zero).ToInt16x16().ToBits())
	if act.IsZero() {
		return
	}
	lq := L
	switch sharp0 {
	case 0:
	case 1:
		lq = lq.ShiftAllRight(1)
	default:
		lq = lq.ShiftAllRight(2)
	}
	one := archsimd.BroadcastUint16x16(1)
	iV = lq.Min(sharp1).Max(one)
	hV = L.ShiftAllRight(4)
	e := L.Add(one).Add(one)
	eV = e.Add(e).Add(iV)
	if bd8 == 2 {
		hV, iV, eV = hV.ShiftAllLeft(2), iV.ShiftAllLeft(2), eV.ShiftAllLeft(2)
	} else {
		hV, iV, eV = hV.ShiftAllLeft(4), iV.ShiftAllLeft(4), eV.ShiftAllLeft(4)
	}
	m4 = act
	ok = true
	return
}

func avx16Mask(n uint32) archsimd.Uint16x16 {
	return archsimd.LoadUint16x16Array(&avx16Nibble[n&15])
}

func avx16Both(v archsimd.Uint8x16) archsimd.Uint8x32 {
	return archsimd.Uint8x32{}.SetLo(v).SetHi(v)
}

// avx16Levels4 loads the four level cells from u.
func avx16Levels4(lvl []uint8, u int) archsimd.Uint8x16 {
	o := int(uint32(4 * u))
	return archsimd.LoadUint8x16Array((*[16]uint8)(lvl[o : o+16]))
}

// avx16LevelsRow loads four consecutive units' levels and their above
// fallbacks (dav1d 16bpc lpf_v_sb) into both 128-bit lanes.
func avx16LevelsRow(lvl []uint8, u0, lvlStride int, nAll uint32) (cur, prev archsimd.Uint8x32) {
	if u0-lvlStride >= 0 && 4*(u0+4) <= len(lvl) {
		return avx16Both(avx16Levels4(lvl, u0)), avx16Both(avx16Levels4(lvl, u0-lvlStride))
	}
	var c, p [4]uint32
	for k := 0; k < 4; k++ {
		if nAll&(1<<uint(k)) != 0 {
			c[k] = binary.LittleEndian.Uint32(lvl[4*(u0+k):])
			p[k] = binary.LittleEndian.Uint32(lvl[4*(u0+k-lvlStride):])
		}
	}
	return avx16Both(archsimd.LoadUint32x4Array(&c).ReshapeToUint8s()), avx16Both(archsimd.LoadUint32x4Array(&p).ReshapeToUint8s())
}

// avx16LevelsCol gathers four stacked units' levels and their left
// fallbacks (dav1d 16bpc lpf_h_sb) into both 128-bit lanes.
func avx16LevelsCol(lvl []uint8, li, lvlStride int, nAll uint32) (cur, prev archsimd.Uint8x32) {
	var c, p [8]uint32
	full := li-1 >= 0 && 4*(li+3*lvlStride+1) <= len(lvl)
	for k := 0; k < 4; k++ {
		if full || nAll&(1<<uint(k)) != 0 {
			i := li + k*lvlStride
			c[k] = binary.LittleEndian.Uint32(lvl[4*i:])
			p[k] = binary.LittleEndian.Uint32(lvl[4*(i-1):])
			c[k+4], p[k+4] = c[k], p[k]
		}
	}
	return archsimd.LoadUint32x8Array(&c).ReshapeToUint8s(), archsimd.LoadUint32x8Array(&p).ReshapeToUint8s()
}

func sbGroupSub4(vm [3]uint32, sh uint) [3]uint32 {
	for k := range vm {
		vm[k] = (vm[k] >> sh) & 0xf
	}
	return vm
}

func avx16LevelSel(comp int) archsimd.Int8x32 {
	return archsimd.LoadInt8x32Array(&avx16LevelSel16[comp&3])
}

func filterSBHorizontal16(pix []byte, off, stride, rowBytes int, bitDepth uint8, vm [3]uint32, luma bool, lvl []uint8, lvlOff, lvlStride, comp int, lut *SBLUT) {
	if !sbAVX2 || (bitDepth != 10 && bitDepth != 12) || len(pix) > lfMaxPlane {
		filterSBHorizontal16Scalar(pix, off, stride, bitDepth, vm, luma, lvl, lvlOff, lvlStride, comp, lut)
		return
	}
	avx16SBHorizontal(pix, off, stride, rowBytes, bitDepth, vm, luma, lvl, lvlOff, lvlStride, comp, lut)
	archsimd.ClearAVXUpperBits()
}

func filterSBVertical16(pix []byte, off, stride, rowBytes int, bitDepth uint8, vm [3]uint32, luma bool, lvl []uint8, lvlOff, lvlStride, comp int, lut *SBLUT) {
	if !sbAVX2 || (bitDepth != 10 && bitDepth != 12) || len(pix) > lfMaxPlane {
		filterSBVertical16Scalar(pix, off, stride, bitDepth, vm, luma, lvl, lvlOff, lvlStride, comp, lut)
		return
	}
	avx16SBVertical(pix, off, stride, rowBytes, bitDepth, vm, luma, lvl, lvlOff, lvlStride, comp, lut)
	archsimd.ClearAVXUpperBits()
}

func avx16Ld(p *[32]byte) archsimd.Uint16x16 { return archsimd.LoadUint8x32Array(p).ReshapeToUint16s() }

// avx16Horz16 filters the sixteen samples of the fourteen tap rows p6..q6 of
// b (rows 0..13) at width 16 and stores the changed rows p5..q5.
func avx16Horz16(k *avx16K, b unsafeutil.Block32[avxR14], eV, iV, hV, m4, mGt4, m16 archsimd.Uint16x16) {
	var t [16]archsimd.Uint16x16
	t[1] = avx16Ld(b.Row32(0))
	t[2] = avx16Ld(b.Row32(1))
	t[3] = avx16Ld(b.Row32(2))
	t[4] = avx16Ld(b.Row32(3))
	t[5] = avx16Ld(b.Row32(4))
	t[6] = avx16Ld(b.Row32(5))
	t[7] = avx16Ld(b.Row32(6))
	t[8] = avx16Ld(b.Row32(7))
	t[9] = avx16Ld(b.Row32(8))
	t[10] = avx16Ld(b.Row32(9))
	t[11] = avx16Ld(b.Row32(10))
	t[12] = avx16Ld(b.Row32(11))
	t[13] = avx16Ld(b.Row32(12))
	t[14] = avx16Ld(b.Row32(13))
	if !avx16Lpf16(k, &t, eV, iV, hV, m4, mGt4, m16) {
		return
	}
	t[2].ReshapeToUint8s().StoreArray(b.Row32(1))
	t[3].ReshapeToUint8s().StoreArray(b.Row32(2))
	t[4].ReshapeToUint8s().StoreArray(b.Row32(3))
	t[5].ReshapeToUint8s().StoreArray(b.Row32(4))
	t[6].ReshapeToUint8s().StoreArray(b.Row32(5))
	t[7].ReshapeToUint8s().StoreArray(b.Row32(6))
	t[8].ReshapeToUint8s().StoreArray(b.Row32(7))
	t[9].ReshapeToUint8s().StoreArray(b.Row32(8))
	t[10].ReshapeToUint8s().StoreArray(b.Row32(9))
	t[11].ReshapeToUint8s().StoreArray(b.Row32(10))
	t[12].ReshapeToUint8s().StoreArray(b.Row32(11))
	t[13].ReshapeToUint8s().StoreArray(b.Row32(12))
}

// avx16Horz8 is avx16Horz16 at width 8: the tap rows p3..q3 (rows 0..7), the
// changed rows p2..q2 stored.
func avx16Horz8(k *avx16K, b unsafeutil.Block32[avxR8], eV, iV, hV, m4, mGt4 archsimd.Uint16x16) {
	o2, o1, o0, n0, n1, n2, _, ok := avx16Core[avxW8](k, avx16Ld(b.Row32(0)), avx16Ld(b.Row32(1)), avx16Ld(b.Row32(2)), avx16Ld(b.Row32(3)),
		avx16Ld(b.Row32(4)), avx16Ld(b.Row32(5)), avx16Ld(b.Row32(6)), avx16Ld(b.Row32(7)), eV, iV, hV, m4, mGt4)
	if !ok {
		return
	}
	o2.ReshapeToUint8s().StoreArray(b.Row32(1))
	o1.ReshapeToUint8s().StoreArray(b.Row32(2))
	o0.ReshapeToUint8s().StoreArray(b.Row32(3))
	n0.ReshapeToUint8s().StoreArray(b.Row32(4))
	n1.ReshapeToUint8s().StoreArray(b.Row32(5))
	n2.ReshapeToUint8s().StoreArray(b.Row32(6))
}

// avx16Horz6 is avx16Horz16 at width 6: the tap rows p2..q2 (rows 0..5), the
// changed rows p1..q1 stored.
func avx16Horz6(k *avx16K, b unsafeutil.Block32[avxR6], eV, iV, hV, m4, mGt4 archsimd.Uint16x16) {
	p2, q2 := avx16Ld(b.Row32(0)), avx16Ld(b.Row32(5))
	_, o1, o0, n0, n1, _, _, ok := avx16Core[avxW6](k, p2, p2, avx16Ld(b.Row32(1)), avx16Ld(b.Row32(2)), avx16Ld(b.Row32(3)), avx16Ld(b.Row32(4)), q2, q2, eV, iV, hV, m4, mGt4)
	if !ok {
		return
	}
	o1.ReshapeToUint8s().StoreArray(b.Row32(1))
	o0.ReshapeToUint8s().StoreArray(b.Row32(2))
	n0.ReshapeToUint8s().StoreArray(b.Row32(3))
	n1.ReshapeToUint8s().StoreArray(b.Row32(4))
}

// avx16Horz4 is avx16Horz16 at width 4: the tap rows p1..q1 (rows 0..3), all
// four stored.
func avx16Horz4(k *avx16K, b unsafeutil.Block32[avxR4], eV, iV, hV, m4 archsimd.Uint16x16) {
	p1, q1 := avx16Ld(b.Row32(0)), avx16Ld(b.Row32(3))
	_, o1, o0, n0, n1, _, _, ok := avx16Core[avxW4](k, p1, p1, p1, avx16Ld(b.Row32(1)), avx16Ld(b.Row32(2)), q1, q1, q1, eV, iV, hV, m4, archsimd.Uint16x16{})
	if !ok {
		return
	}
	o1.ReshapeToUint8s().StoreArray(b.Row32(0))
	o0.ReshapeToUint8s().StoreArray(b.Row32(1))
	n0.ReshapeToUint8s().StoreArray(b.Row32(2))
	n1.ReshapeToUint8s().StoreArray(b.Row32(3))
}

// avx16HorzGroup runs the width-wd kernel on the sixteen samples (32 bytes)
// at byte column x of v, whose row rh is q0.
func avx16HorzGroup(k *avx16K, v unsafeutil.Rows, rh, x, wd int, eV, iV, hV, m4, mGt4 archsimd.Uint16x16, n16 uint32) {
	switch wd {
	case 16:
		avx16Horz16(k, unsafeutil.Block32At[avxR14](v, rh-7, x), eV, iV, hV, m4, mGt4, avx16Mask(n16))
	case 8:
		avx16Horz8(k, unsafeutil.Block32At[avxR8](v, rh-4, x), eV, iV, hV, m4, mGt4)
	case 6:
		avx16Horz6(k, unsafeutil.Block32At[avxR6](v, rh-3, x), eV, iV, hV, m4, mGt4)
	default:
		avx16Horz4(k, unsafeutil.Block32At[avxR4](v, rh-2, x), eV, iV, hV, m4)
	}
}

// avx16SBHorizontal is dav1d lpf_v_sb_{y,uv}_16bpc: the horizontal edges of
// one row of units, sixteen samples (one vmask nibble) per kernel call. As in
// avxSBHorizontal8, the call's tap rows and its two level rows are checked
// once as views, and every group is a column of them; calls whose views do
// not fit take the group-by-group edge driver. simd counts the groups the
// vector kernel handled.
func avx16SBHorizontal(pix []byte, off, stride, rowBytes int, bitDepth uint8, vm [3]uint32, luma bool, lvl []uint8, lvlOff, lvlStride, comp int, lut *SBLUT) (simd int) {
	all := vm[0] | vm[1] | vm[2]
	if all == 0 {
		return 0
	}
	n := 32 * ((31-bits.LeadingZeros32(all))>>2 + 1)
	if n <= rowBytes {
		switch {
		case vm[2] != 0:
			simd = avx16HorzRows[avxW16](pix, off, stride, n, bitDepth, vm, luma, lvl, lvlOff, lvlStride, comp, lut)
		case vm[1] != 0 && luma:
			simd = avx16HorzRows[avxW8](pix, off, stride, n, bitDepth, vm, luma, lvl, lvlOff, lvlStride, comp, lut)
		case vm[1] != 0:
			simd = avx16HorzRows[avxW6](pix, off, stride, n, bitDepth, vm, luma, lvl, lvlOff, lvlStride, comp, lut)
		default:
			simd = avx16HorzRows[avxW4](pix, off, stride, n, bitDepth, vm, luma, lvl, lvlOff, lvlStride, comp, lut)
		}
		if simd >= 0 {
			return simd
		}
	}
	return avx16SBHorizontalEdge(pix, off, stride, rowBytes, bitDepth, vm, luma, lvl, lvlOff, lvlStride, comp, lut)
}

// avx16HorzRows is avxHorzRows8 for 16-bit samples: one view of the call's
// tap rows (n bytes each, 32 per group) and one of its two level rows (16
// bytes per group); it returns -1, having filtered nothing, when a view
// leaves its slice.
func avx16HorzRows[W avxCallWidth](pix []byte, off, stride, n int, bitDepth uint8, vm [3]uint32, luma bool, lvl []uint8, lvlOff, lvlStride, comp int, lut *SBLUT) (simd int) {
	var wT W
	rh := avxHalfRows(len(wT))
	li, ls4 := 4*lvlOff, 4*lvlStride
	if !unsafeutil.RowsFit(len(pix), off-rh*stride, stride, n, 2*rh) || !unsafeutil.RowsFit(len(lvl), li-ls4, ls4, n/2, 2) {
		return -1
	}
	v := unsafeutil.NewRows(pix, off-rh*stride, stride, n, 2*rh)
	lv := unsafeutil.NewRows(lvl, li-ls4, ls4, n/2, 2)
	bd8 := uint(bitDepth - 8)
	k := newAVX16K(bd8)
	gt4All := vm[1] | vm[2]
	all := vm[0] | gt4All
	sel := avx16LevelSel(comp)
	sharp0, sharp1 := uint64(lut.Sharp[0]), archsimd.BroadcastUint16x16(uint16(lut.Sharp[1]))
	for x := 0; x+32 <= n; x += 32 {
		sh := uint(x) / 8
		nib := (all >> sh) & 0xf
		if nib == 0 {
			continue
		}
		nGt4 := (gt4All >> sh) & 0xf
		n16 := (vm[2] >> sh) & 0xf
		lb := unsafeutil.BlockAt[avxR2](lv, 0, x/2)
		eV, iV, hV, m4, ok := avx16GroupLimits(avx16Both(archsimd.LoadUint8x16Array(lb.Row16(1))), avx16Both(archsimd.LoadUint8x16Array(lb.Row16(0))), sel, nib, sharp0, sharp1, bd8)
		if !ok {
			continue
		}
		simd++
		switch {
		case len(wT) == 16 && n16 != 0:
			avx16Horz16(&k, unsafeutil.Block32At[avxR14](v, rh-7, x), eV, iV, hV, m4, avx16Mask(nGt4), avx16Mask(n16))
		case len(wT) >= 6 && nGt4 != 0 && luma:
			avx16Horz8(&k, unsafeutil.Block32At[avxR8](v, rh-4, x), eV, iV, hV, m4, avx16Mask(nGt4))
		case len(wT) >= 6 && nGt4 != 0:
			avx16Horz6(&k, unsafeutil.Block32At[avxR6](v, rh-3, x), eV, iV, hV, m4, avx16Mask(nGt4))
		default:
			avx16Horz4(&k, unsafeutil.Block32At[avxR4](v, rh-2, x), eV, iV, hV, m4)
		}
	}
	return simd
}

// avx16SBHorizontalEdge is avx16SBHorizontal group by group, for calls that
// reach the end of pix, the row or the level map.
func avx16SBHorizontalEdge(pix []byte, off, stride, rowBytes int, bitDepth uint8, vm [3]uint32, luma bool, lvl []uint8, lvlOff, lvlStride, comp int, lut *SBLUT) (simd int) {
	bd8 := uint(bitDepth - 8)
	k := newAVX16K(bd8)
	gt4All := vm[1] | vm[2]
	all := vm[0] | gt4All
	sel := avx16LevelSel(comp)
	sharp0, sharp1 := uint64(lut.Sharp[0]), archsimd.BroadcastUint16x16(uint16(lut.Sharp[1]))
	for g := 0; g < 8 && all>>(4*uint(g)) != 0; g++ {
		sh := 4 * uint(g)
		nib := (all >> sh) & 0xf
		if nib == 0 {
			continue
		}
		nGt4 := (gt4All >> sh) & 0xf
		n16 := (vm[2] >> sh) & 0xf
		wd := avxSBWidth(nGt4, n16, luma)
		u0 := lvlOff + 4*g
		q0Off := off + 32*g
		rh := avxHalfRows(wd)
		if 32*g+32 > rowBytes || !unsafeutil.RowsFit(len(pix), q0Off-rh*stride, stride, 32, 2*rh) {
			filterSBHorizontal16Scalar(pix, q0Off, stride, bitDepth, sbGroupSub4(vm, sh), luma, lvl, u0, lvlStride, comp, lut)
			continue
		}
		cur, prev := avx16LevelsRow(lvl, u0, lvlStride, nib)
		eV, iV, hV, m4, ok := avx16GroupLimits(cur, prev, sel, nib, sharp0, sharp1, bd8)
		if !ok {
			continue
		}
		simd++
		avx16HorzGroup(&k, unsafeutil.NewRows(pix, q0Off-rh*stride, stride, 32, 2*rh), rh, 0, wd, eV, iV, hV, m4, avx16Mask(nGt4), n16)
	}
	return simd
}

// avx16Rows8 loads the eight-sample rows k of lo (rows 0..7) and hi (rows
// 8..15), row k of hi sharing a register with row k of lo, and transposes
// them into tap vectors.
func avx16Rows8(lo, hi unsafeutil.Block[avxR8]) (t0, t1, t2, t3, t4, t5, t6, t7 archsimd.Uint8x32) {
	return avxTransposeWords(avx16Rows2(lo, hi, 0), avx16Rows2(lo, hi, 1), avx16Rows2(lo, hi, 2), avx16Rows2(lo, hi, 3),
		avx16Rows2(lo, hi, 4), avx16Rows2(lo, hi, 5), avx16Rows2(lo, hi, 6), avx16Rows2(lo, hi, 7))
}

func avx16Rows2(lo, hi unsafeutil.Block[avxR8], k int) archsimd.Uint8x32 {
	return archsimd.Uint8x32{}.SetLo(archsimd.LoadUint8x16Array(lo.Row16(k))).SetHi(archsimd.LoadUint8x16Array(hi.Row16(k)))
}

// avx16StoreRows8 transposes tap vectors back to rows and stores them to lo
// and hi.
func avx16StoreRows8(lo, hi unsafeutil.Block[avxR8], t0, t1, t2, t3, t4, t5, t6, t7 archsimd.Uint8x32) {
	y0, y1, y2, y3, y4, y5, y6, y7 := avxTransposeWords(t0, t1, t2, t3, t4, t5, t6, t7)
	y0.GetLo().StoreArray(lo.Row16(0))
	y0.GetHi().StoreArray(hi.Row16(0))
	y1.GetLo().StoreArray(lo.Row16(1))
	y1.GetHi().StoreArray(hi.Row16(1))
	y2.GetLo().StoreArray(lo.Row16(2))
	y2.GetHi().StoreArray(hi.Row16(2))
	y3.GetLo().StoreArray(lo.Row16(3))
	y3.GetHi().StoreArray(hi.Row16(3))
	y4.GetLo().StoreArray(lo.Row16(4))
	y4.GetHi().StoreArray(hi.Row16(4))
	y5.GetLo().StoreArray(lo.Row16(5))
	y5.GetHi().StoreArray(hi.Row16(5))
	y6.GetLo().StoreArray(lo.Row16(6))
	y6.GetHi().StoreArray(hi.Row16(6))
	y7.GetLo().StoreArray(lo.Row16(7))
	y7.GetHi().StoreArray(hi.Row16(7))
}

func u16v(x archsimd.Uint8x32) archsimd.Uint16x16 { return x.ReshapeToUint16s() }
func u8v(x archsimd.Uint16x16) archsimd.Uint8x32  { return x.ReshapeToUint8s() }

// avx16Vertical8 filters a vertical edge across sixteen rows for widths up to
// 8: lo and hi are rows 0..7 and 8..15, eight samples p3..q3 each, read and
// written.
func avx16Vertical8[W avxWidth](k *avx16K, lo, hi unsafeutil.Block[avxR8], eV, iV, hV, m4, mGt4 archsimd.Uint16x16) {
	t0, t1, t2, t3, t4, t5, t6, t7 := avx16Rows8(lo, hi)
	o2, o1, o0, n0, n1, n2, _, ok := avx16Core[W](k, u16v(t0), u16v(t1), u16v(t2), u16v(t3), u16v(t4), u16v(t5), u16v(t6), u16v(t7), eV, iV, hV, m4, mGt4)
	if !ok {
		return
	}
	avx16StoreRows8(lo, hi, t0, u8v(o2), u8v(o1), u8v(o0), u8v(n0), u8v(n1), u8v(n2), t7)
}

// avx16Vertical16 is the wd16 form: sixteen samples p7..q7 per row, the
// eight p7..p0 in lo/hi and the eight q0..q7 in lo2/hi2.
func avx16Vertical16(k *avx16K, lo, hi, lo2, hi2 unsafeutil.Block[avxR8], eV, iV, hV, m4, mGt4, m16 archsimd.Uint16x16) {
	var t [16]archsimd.Uint16x16
	a0, a1, a2, a3, a4, a5, a6, a7 := avx16Rows8(lo, hi)
	b0, b1, b2, b3, b4, b5, b6, b7 := avx16Rows8(lo2, hi2)
	t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7] = u16v(a0), u16v(a1), u16v(a2), u16v(a3), u16v(a4), u16v(a5), u16v(a6), u16v(a7)
	t[8], t[9], t[10], t[11], t[12], t[13], t[14], t[15] = u16v(b0), u16v(b1), u16v(b2), u16v(b3), u16v(b4), u16v(b5), u16v(b6), u16v(b7)
	if !avx16Lpf16(k, &t, eV, iV, hV, m4, mGt4, m16) {
		return
	}
	avx16StoreRows8(lo, hi, u8v(t[0]), u8v(t[1]), u8v(t[2]), u8v(t[3]), u8v(t[4]), u8v(t[5]), u8v(t[6]), u8v(t[7]))
	avx16StoreRows8(lo2, hi2, u8v(t[8]), u8v(t[9]), u8v(t[10]), u8v(t[11]), u8v(t[12]), u8v(t[13]), u8v(t[14]), u8v(t[15]))
}

// avx16VertGroup runs the width-wd kernel on the sixteen rows of v from row
// r, from byte column c (p7 at wd 16, else p3).
func avx16VertGroup(k *avx16K, v unsafeutil.Rows, r, c, wd int, eV, iV, hV, m4, mGt4 archsimd.Uint16x16, n16 uint32) {
	lo, hi := unsafeutil.BlockAt[avxR8](v, r, c), unsafeutil.BlockAt[avxR8](v, r+8, c)
	switch wd {
	case 16:
		avx16Vertical16(k, lo, hi, unsafeutil.BlockAt[avxR8](v, r, c+16), unsafeutil.BlockAt[avxR8](v, r+8, c+16), eV, iV, hV, m4, mGt4, avx16Mask(n16))
	case 8:
		avx16Vertical8[avxW8](k, lo, hi, eV, iV, hV, m4, mGt4)
	case 6:
		avx16Vertical8[avxW6](k, lo, hi, eV, iV, hV, m4, mGt4)
	default:
		avx16Vertical8[avxW4](k, lo, hi, eV, iV, hV, m4, mGt4)
	}
}

// avx16SBVertical is dav1d lpf_h_sb_{y,uv}_16bpc: the vertical edges of one
// column of units, sixteen rows per kernel call, with arm64's footprint (p3..q3,
// or p7..q7 at width 16). One view of the call's rows and one of its
// (l[-1], l[0]) level pairs replace the per-group checks; calls whose views do
// not fit take the group-by-group edge driver.
func avx16SBVertical(pix []byte, off, stride, rowBytes int, bitDepth uint8, vm [3]uint32, luma bool, lvl []uint8, lvlOff, lvlStride, comp int, lut *SBLUT) (simd int) {
	gt4All := vm[1] | vm[2]
	all := vm[0] | gt4All
	if all == 0 {
		return 0
	}
	hiG := (31 - bits.LeadingZeros32(all)) >> 2
	reach := 8
	if vm[2] != 0 {
		reach = 16
	}
	if rowBytes < reach || !unsafeutil.RowsFit(len(pix), off-reach, stride, 2*reach, 16*hiG+16) || !unsafeutil.RowsFit(len(lvl), 4*lvlOff-4, 4*lvlStride, 8, 4*hiG+4) {
		return avx16SBVerticalEdge(pix, off, stride, rowBytes, bitDepth, vm, luma, lvl, lvlOff, lvlStride, comp, lut)
	}
	v := unsafeutil.NewRows(pix, off-reach, stride, 2*reach, 16*hiG+16)
	lv := unsafeutil.NewRows(lvl, 4*lvlOff-4, 4*lvlStride, 8, 4*hiG+4)
	c16, c8 := reach-16, reach-8 // p7 and p3 byte columns of the view
	bd8 := uint(bitDepth - 8)
	k := newAVX16K(bd8)
	sel := avx16LevelSel(comp)
	sharp0, sharp1 := uint64(lut.Sharp[0]), archsimd.BroadcastUint16x16(uint16(lut.Sharp[1]))
	for g := 0; g <= hiG; g++ {
		sh := 4 * uint(g)
		nib := (all >> sh) & 0xf
		if nib == 0 {
			continue
		}
		nGt4 := (gt4All >> sh) & 0xf
		n16 := (vm[2] >> sh) & 0xf
		cur, prev := avx16LevelPairs(lv, 4*g)
		eV, iV, hV, m4, ok := avx16GroupLimits(cur, prev, sel, nib, sharp0, sharp1, bd8)
		if !ok {
			continue
		}
		simd++
		wd := avxSBWidth(nGt4, n16, luma)
		c := c8
		if wd == 16 {
			c = c16
		}
		avx16VertGroup(&k, v, 16*g, c, wd, eV, iV, hV, m4, avx16Mask(nGt4), n16)
	}
	return simd
}

// avx16LevelPairs splits the four (l[-1], l[0]) cell pairs of rows r..r+3 of
// lv (one per row) into the l[0] cells (cur) and the l[-1] cells (prev), each
// in both 128-bit lanes.
func avx16LevelPairs(lv unsafeutil.Rows, r int) (cur, prev archsimd.Uint8x32) {
	lg := lv.Sub(r, 0, 8, 4)
	le := binary.LittleEndian
	var c, p [8]uint32
	x := le.Uint64(lg.Row8(0, 0)[:])
	c[0], p[0] = uint32(x>>32), uint32(x)
	x = le.Uint64(lg.Row8(1, 0)[:])
	c[1], p[1] = uint32(x>>32), uint32(x)
	x = le.Uint64(lg.Row8(2, 0)[:])
	c[2], p[2] = uint32(x>>32), uint32(x)
	x = le.Uint64(lg.Row8(3, 0)[:])
	c[3], p[3] = uint32(x>>32), uint32(x)
	c[4], c[5], c[6], c[7] = c[0], c[1], c[2], c[3]
	p[4], p[5], p[6], p[7] = p[0], p[1], p[2], p[3]
	return archsimd.LoadUint32x8Array(&c).ReshapeToUint8s(), archsimd.LoadUint32x8Array(&p).ReshapeToUint8s()
}

// avx16SBVerticalEdge is avx16SBVertical group by group, for calls that
// reach the end of pix, the row or the level map.
func avx16SBVerticalEdge(pix []byte, off, stride, rowBytes int, bitDepth uint8, vm [3]uint32, luma bool, lvl []uint8, lvlOff, lvlStride, comp int, lut *SBLUT) (simd int) {
	bd8 := uint(bitDepth - 8)
	k := newAVX16K(bd8)
	gt4All := vm[1] | vm[2]
	all := vm[0] | gt4All
	sel := avx16LevelSel(comp)
	sharp0, sharp1 := uint64(lut.Sharp[0]), archsimd.BroadcastUint16x16(uint16(lut.Sharp[1]))
	for g := 0; g < 8 && all>>(4*uint(g)) != 0; g++ {
		sh := 4 * uint(g)
		nib := (all >> sh) & 0xf
		if nib == 0 {
			continue
		}
		nGt4 := (gt4All >> sh) & 0xf
		n16 := (vm[2] >> sh) & 0xf
		wd := avxSBWidth(nGt4, n16, luma)
		u0 := lvlOff + 4*g*lvlStride
		q0Off := off + 16*g*stride
		reach := 8
		if wd == 16 {
			reach = 16
		}
		if rowBytes < reach || !unsafeutil.RowsFit(len(pix), q0Off-reach, stride, 2*reach, 16) {
			filterSBVertical16Scalar(pix, q0Off, stride, bitDepth, sbGroupSub4(vm, sh), luma, lvl, u0, lvlStride, comp, lut)
			continue
		}
		cur, prev := avx16LevelsCol(lvl, u0, lvlStride, nib)
		eV, iV, hV, m4, ok := avx16GroupLimits(cur, prev, sel, nib, sharp0, sharp1, bd8)
		if !ok {
			continue
		}
		simd++
		avx16VertGroup(&k, unsafeutil.NewRows(pix, q0Off-reach, stride, 2*reach, 16), 0, 0, wd, eV, iV, hV, m4, avx16Mask(nGt4), n16)
	}
	return simd
}
