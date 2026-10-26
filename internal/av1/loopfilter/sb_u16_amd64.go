// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && amd64 && !purego

package loopfilter

import (
	"encoding/binary"
	"math/bits"
	"simd/archsimd"
	"unsafe"
)

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

func avx16Ld(base unsafe.Pointer, off int) archsimd.Uint16x16 {
	return archsimd.LoadUint16x16Array((*[16]uint16)(unsafe.Add(base, off)))
}

func avx16St(base unsafe.Pointer, off int, v archsimd.Uint16x16) {
	v.StoreArray((*[16]uint16)(unsafe.Add(base, off)))
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
	wd := int(unsafe.Sizeof(wdT))
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

// avx16Acc carries the apply counters through a call.
type avx16Acc struct {
	applied uint32
	max     archsimd.Uint16x16
}

func (a *avx16Acc) stats() SBStats {
	var b [16]uint16
	a.max.StoreArray(&b)
	return SBStats{Applied: a.applied, MaxLevel: uint8(max(b[0], b[4], b[8], b[12]))}
}

// avx16GroupLimits is the 16bpc FILTER prologue for one four-unit group: lv
// holds the units' own levels (four components per unit) in both 128-bit
// lanes and pv their fallbacks. L = cur ? cur : prev for component comp lands
// in each unit's four uint16 lanes (dav1d pshufb pb_4x1_4x5_4x9_4x13), and
// I = imax(imin(L >> sharp[0], sharp[1]), 1), E = 2*(L+2)+I and H = L>>4 are
// computed there and shifted left by bitdepth_min_8 (2 or 4). Everything
// stays in 256-bit registers: 128-bit values are moved with legacy-SSE MOVUPS
// by the compiler. m4 is the vmask nibble restricted to L != 0.
func avx16GroupLimits(lv, pv archsimd.Uint8x32, sel archsimd.Int8x32, nAll uint32, sharp0 uint64, sharp1 archsimd.Uint16x16, bd8 uint, acc *avx16Acc) (eV, iV, hV, m4 archsimd.Uint16x16, ok bool) {
	var zero archsimd.Uint16x16
	c := lv.PermuteOrZeroGrouped(sel).ReshapeToUint16s()
	p := pv.PermuteOrZeroGrouped(sel).ReshapeToUint16s()
	L := c.Or(p.And(c.Equal(zero).ToInt16x16().ToBits()))
	act := avx16Mask(nAll).AndNot(L.Equal(zero).ToInt16x16().ToBits())
	if act.IsZero() {
		return
	}
	acc.applied += uint32(bits.OnesCount32(archsimd.Int8x32{}.Greater(act.ReshapeToUint8s().BitsToInt8()).ToBits())) >> 3
	acc.max = acc.max.Max(L.And(act))
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

// avx16LevelsRow loads four consecutive units' levels and their above
// fallbacks (dav1d 16bpc lpf_v_sb) into both 128-bit lanes.
func avx16LevelsRow(lb []byte, lvl [][4]uint8, u0, lvlStride int, nAll uint32) (cur, prev archsimd.Uint8x32) {
	if u0-lvlStride >= 0 && u0+4 <= len(lvl) {
		return avx16Both(archsimd.LoadUint8x16(lb[4*u0:])), avx16Both(archsimd.LoadUint8x16(lb[4*(u0-lvlStride):]))
	}
	var c, p [4]uint32
	for k := 0; k < 4; k++ {
		if nAll&(1<<uint(k)) != 0 {
			c[k] = binary.LittleEndian.Uint32(lvl[u0+k][:])
			p[k] = binary.LittleEndian.Uint32(lvl[u0+k-lvlStride][:])
		}
	}
	return avx16Both(archsimd.LoadUint32x4Array(&c).ReshapeToUint8s()), avx16Both(archsimd.LoadUint32x4Array(&p).ReshapeToUint8s())
}

// avx16LevelsCol gathers four stacked units' levels and their left
// fallbacks (dav1d 16bpc lpf_h_sb) into both 128-bit lanes.
func avx16LevelsCol(lvl [][4]uint8, li, lvlStride int, nAll uint32) (cur, prev archsimd.Uint8x32) {
	var c, p [8]uint32
	full := li-1 >= 0 && li+3*lvlStride < len(lvl)
	for k := 0; k < 4; k++ {
		if full || nAll&(1<<uint(k)) != 0 {
			i := li + k*lvlStride
			c[k] = binary.LittleEndian.Uint32(lvl[i][:])
			p[k] = binary.LittleEndian.Uint32(lvl[i-1][:])
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

func filterSBHorizontal16(pix []byte, off, stride, rowBytes int, bitDepth uint8, vm [3]uint32, luma bool, lvl [][4]uint8, lvlOff, lvlStride, comp int, lut *SBLUT) SBStats {
	if !sbAVX2 || (bitDepth != 10 && bitDepth != 12) {
		return filterSBHorizontal16Scalar(pix, off, stride, bitDepth, vm, luma, lvl, lvlOff, lvlStride, comp, lut)
	}
	st, _ := avx16SBHorizontal(pix, off, stride, rowBytes, bitDepth, vm, luma, lvl, lvlOff, lvlStride, comp, lut)
	archsimd.ClearAVXUpperBits()
	return st
}

func filterSBVertical16(pix []byte, off, stride, rowBytes int, bitDepth uint8, vm [3]uint32, luma bool, lvl [][4]uint8, lvlOff, lvlStride, comp int, lut *SBLUT) SBStats {
	if !sbAVX2 || (bitDepth != 10 && bitDepth != 12) {
		return filterSBVertical16Scalar(pix, off, stride, bitDepth, vm, luma, lvl, lvlOff, lvlStride, comp, lut)
	}
	st, _ := avx16SBVertical(pix, off, stride, rowBytes, bitDepth, vm, luma, lvl, lvlOff, lvlStride, comp, lut)
	archsimd.ClearAVXUpperBits()
	return st
}

// avx16SBHorizontal is dav1d lpf_v_sb_{y,uv}_16bpc: the horizontal edges of
// one row of units, sixteen samples per kernel call. simd counts the groups
// the vector kernel handled.
func avx16SBHorizontal(pix []byte, off, stride, rowBytes int, bitDepth uint8, vm [3]uint32, luma bool, lvl [][4]uint8, lvlOff, lvlStride, comp int, lut *SBLUT) (st SBStats, simd int) {
	var acc avx16Acc
	bd8 := uint(bitDepth - 8)
	k := newAVX16K(bd8)
	gt4All := vm[1] | vm[2]
	all := vm[0] | gt4All
	lb := avxLevelBytes(lvl)
	base := unsafe.Pointer(unsafe.SliceData(pix))
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
		reach := 4
		if wd == 16 {
			reach = 7
		}
		if 32*g+32 > rowBytes || q0Off-reach*stride < 0 || q0Off+(reach-1)*stride+32 > len(pix) {
			st.Add(filterSBHorizontal16Scalar(pix, q0Off, stride, bitDepth, sbGroupSub4(vm, sh), luma, lvl, u0, lvlStride, comp, lut))
			continue
		}
		cur, prev := avx16LevelsRow(lb, lvl, u0, lvlStride, nib)
		eV, iV, hV, m4, ok := avx16GroupLimits(cur, prev, sel, nib, sharp0, sharp1, bd8, &acc)
		if !ok {
			continue
		}
		simd++
		mGt4 := avx16Mask(nGt4)
		if wd == 16 {
			var t [16]archsimd.Uint16x16
			for j := 1; j < 15; j++ {
				t[j] = avx16Ld(base, q0Off+(j-8)*stride)
			}
			if avx16Lpf16(&k, &t, eV, iV, hV, m4, mGt4, avx16Mask(n16)) {
				for j := 2; j < 14; j++ {
					avx16St(base, q0Off+(j-8)*stride, t[j])
				}
			}
			continue
		}
		p1 := avx16Ld(base, q0Off-2*stride)
		p0 := avx16Ld(base, q0Off-stride)
		q0 := avx16Ld(base, q0Off)
		q1 := avx16Ld(base, q0Off+stride)
		switch wd {
		case 8:
			p3 := avx16Ld(base, q0Off-4*stride)
			p2 := avx16Ld(base, q0Off-3*stride)
			q2 := avx16Ld(base, q0Off+2*stride)
			q3 := avx16Ld(base, q0Off+3*stride)
			o2, o1, o0, n0, n1, n2, _, ok := avx16Core[avxW8](&k, p3, p2, p1, p0, q0, q1, q2, q3, eV, iV, hV, m4, mGt4)
			if ok {
				avx16St(base, q0Off-3*stride, o2)
				avx16St(base, q0Off-2*stride, o1)
				avx16St(base, q0Off-stride, o0)
				avx16St(base, q0Off, n0)
				avx16St(base, q0Off+stride, n1)
				avx16St(base, q0Off+2*stride, n2)
			}
		case 6:
			p2 := avx16Ld(base, q0Off-3*stride)
			q2 := avx16Ld(base, q0Off+2*stride)
			_, o1, o0, n0, n1, _, _, ok := avx16Core[avxW6](&k, p2, p2, p1, p0, q0, q1, q2, q2, eV, iV, hV, m4, mGt4)
			if ok {
				avx16St(base, q0Off-2*stride, o1)
				avx16St(base, q0Off-stride, o0)
				avx16St(base, q0Off, n0)
				avx16St(base, q0Off+stride, n1)
			}
		default:
			_, o1, o0, n0, n1, _, _, ok := avx16Core[avxW4](&k, p1, p1, p1, p0, q0, q1, q1, q1, eV, iV, hV, m4, mGt4)
			if ok {
				avx16St(base, q0Off-2*stride, o1)
				avx16St(base, q0Off-stride, o0)
				avx16St(base, q0Off, n0)
				avx16St(base, q0Off+stride, n1)
			}
		}
	}
	st.Add(acc.stats())
	return st, simd
}

// avx16Rows8 loads 8 rows of eight samples (sixteen bytes each, row r and r+8
// sharing a register) from p and transposes them into tap vectors.
func avx16Rows8(p unsafe.Pointer, stride int) (t0, t1, t2, t3, t4, t5, t6, t7 archsimd.Uint8x32) {
	s8 := 8 * stride
	var y [8]archsimd.Uint8x32
	r := p
	for j := range y {
		y[j] = avxRows2(r, s8)
		r = unsafe.Add(r, stride)
	}
	return avxTransposeWords(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7])
}

// avx16StoreRows8 transposes tap vectors back to rows and stores them.
func avx16StoreRows8(p unsafe.Pointer, stride int, t0, t1, t2, t3, t4, t5, t6, t7 archsimd.Uint8x32) {
	var y [8]archsimd.Uint8x32
	y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7] = avxTransposeWords(t0, t1, t2, t3, t4, t5, t6, t7)
	s8 := 8 * stride
	r := p
	for j := range y {
		y[j].GetLo().StoreArray((*[16]uint8)(r))
		y[j].GetHi().StoreArray((*[16]uint8)(unsafe.Add(r, s8)))
		r = unsafe.Add(r, stride)
	}
}

func u16v(x archsimd.Uint8x32) archsimd.Uint16x16 { return x.ReshapeToUint16s() }
func u8v(x archsimd.Uint16x16) archsimd.Uint8x32  { return x.ReshapeToUint8s() }

// avx16Vertical8 filters a vertical edge across sixteen rows for widths up to
// 8: p3 is the byte offset of p3 on the first row (eight samples p3..q3 per
// row are read and written).
func avx16Vertical8[W avxWidth](k *avx16K, base unsafe.Pointer, p3, stride int, eV, iV, hV, m4, mGt4 archsimd.Uint16x16) {
	p := unsafe.Add(base, p3)
	t0, t1, t2, t3, t4, t5, t6, t7 := avx16Rows8(p, stride)
	o2, o1, o0, n0, n1, n2, _, ok := avx16Core[W](k, u16v(t0), u16v(t1), u16v(t2), u16v(t3), u16v(t4), u16v(t5), u16v(t6), u16v(t7), eV, iV, hV, m4, mGt4)
	if !ok {
		return
	}
	avx16StoreRows8(p, stride, t0, u8v(o2), u8v(o1), u8v(o0), u8v(n0), u8v(n1), u8v(n2), t7)
}

// avx16Vertical16 is the wd16 form: sixteen samples p7..q7 per row, p7 the
// byte offset of p7 on the first row.
func avx16Vertical16(k *avx16K, base unsafe.Pointer, p7, stride int, eV, iV, hV, m4, mGt4, m16 archsimd.Uint16x16) {
	p := unsafe.Add(base, p7)
	var t [16]archsimd.Uint16x16
	a0, a1, a2, a3, a4, a5, a6, a7 := avx16Rows8(p, stride)
	b0, b1, b2, b3, b4, b5, b6, b7 := avx16Rows8(unsafe.Add(p, 16), stride)
	t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7] = u16v(a0), u16v(a1), u16v(a2), u16v(a3), u16v(a4), u16v(a5), u16v(a6), u16v(a7)
	t[8], t[9], t[10], t[11], t[12], t[13], t[14], t[15] = u16v(b0), u16v(b1), u16v(b2), u16v(b3), u16v(b4), u16v(b5), u16v(b6), u16v(b7)
	if !avx16Lpf16(k, &t, eV, iV, hV, m4, mGt4, m16) {
		return
	}
	avx16StoreRows8(p, stride, u8v(t[0]), u8v(t[1]), u8v(t[2]), u8v(t[3]), u8v(t[4]), u8v(t[5]), u8v(t[6]), u8v(t[7]))
	avx16StoreRows8(unsafe.Add(p, 16), stride, u8v(t[8]), u8v(t[9]), u8v(t[10]), u8v(t[11]), u8v(t[12]), u8v(t[13]), u8v(t[14]), u8v(t[15]))
}

// avx16SBVertical is dav1d lpf_h_sb_{y,uv}_16bpc: the vertical edges of one
// column of units, sixteen rows per kernel call, with arm64's footprint (p3..q3,
// or p7..q7 at width 16).
func avx16SBVertical(pix []byte, off, stride, rowBytes int, bitDepth uint8, vm [3]uint32, luma bool, lvl [][4]uint8, lvlOff, lvlStride, comp int, lut *SBLUT) (st SBStats, simd int) {
	var acc avx16Acc
	bd8 := uint(bitDepth - 8)
	k := newAVX16K(bd8)
	gt4All := vm[1] | vm[2]
	all := vm[0] | gt4All
	base := unsafe.Pointer(unsafe.SliceData(pix))
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
		lastRow := q0Off + 15*stride
		reach := 8
		if wd == 16 {
			reach = 16
		}
		if rowBytes < reach || q0Off-reach < 0 || lastRow+reach > len(pix) {
			st.Add(filterSBVertical16Scalar(pix, q0Off, stride, bitDepth, sbGroupSub4(vm, sh), luma, lvl, u0, lvlStride, comp, lut))
			continue
		}
		cur, prev := avx16LevelsCol(lvl, u0, lvlStride, nib)
		eV, iV, hV, m4, ok := avx16GroupLimits(cur, prev, sel, nib, sharp0, sharp1, bd8, &acc)
		if !ok {
			continue
		}
		simd++
		mGt4 := avx16Mask(nGt4)
		switch wd {
		case 16:
			avx16Vertical16(&k, base, q0Off-16, stride, eV, iV, hV, m4, mGt4, avx16Mask(n16))
		case 8:
			avx16Vertical8[avxW8](&k, base, q0Off-8, stride, eV, iV, hV, m4, mGt4)
		case 6:
			avx16Vertical8[avxW6](&k, base, q0Off-8, stride, eV, iV, hV, m4, mGt4)
		default:
			avx16Vertical8[avxW4](&k, base, q0Off-8, stride, eV, iV, hV, m4, mGt4)
		}
	}
	st.Add(acc.stats())
	return st, simd
}
