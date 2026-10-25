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

// This file ports dav1d's AVX2 superblock deblock drivers and their FILTER
// macro (src/x86/loopfilter_avx2.asm lpf_{v,h}_sb_{y,uv}_8bpc) to archsimd.
// One kernel call covers 32 samples, eight 4x4 units, which is one byte of
// each vmask word, as dav1d's mask-byte loop does: horizontal edges filter 32
// columns of rows straight from the plane, vertical edges transpose 32 rows
// into tap vectors (rows 0..15 in the low 128-bit lane, 16..31 in the high
// one) and back.
//
// AVX2 archsimd lacks a few of the instructions the assembly uses, so three
// steps are rebuilt from operations it does have, byte-exact:
//   - punpck{l,h}bw: the two rows are first paired into one lane with a 64-bit
//     unpack, and a byte shuffle (VPSHUFB 0,8,1,9,...) interleaves them.
//   - pmaddubsw/pmulhrsw/packuswb in the flat filters: the 16-bit sums run on
//     the even and odd bytes of each 16-bit lane (v & 0xff, v >> 8), and the
//     rounded results are merged back as even | odd<<8. The split is
//     lane-local, so no cross-lane narrowing is needed.
//   - byte shifts: 64-bit shifts plus a mask, as the assembly itself does.
// The narrow (four-tap) filter is dav1d's all-int8 saturating sequence.

// sbAVX2 gates the AVX2 superblock path; the tests flip it to compare both.
var sbAVX2 = archsimd.X86.AVX2()

type (
	avxW4 [4]byte
	avxW6 [6]byte
	avxW8 [8]byte
)

// avxWidth selects a width-specialized kernel instance: each type's size is
// its filter width, so every instance drops the code of the other widths.
type avxWidth interface {
	avxW4 | avxW6 | avxW8
}

// avxConst holds the byte-splat and shuffle constants, loaded from memory as
// the assembly's RODATA.
var avxConst = struct {
	b80, b7f, bf8, b10, b40, b01, b03, b04, b3f, b0f [32]uint8
	w00ff                                            [16]uint16
	unitBits                                         [8]uint32
	zip, unzip                                       [32]int8
	lowLanes                                         [32]int8
	levelSel                                         [4][32]int8
}{
	b80: splat32(0x80), b7f: splat32(0x7f), bf8: splat32(0xf8), b10: splat32(0x10),
	b40: splat32(0x40), b01: splat32(1), b03: splat32(3), b04: splat32(4),
	b3f: splat32(0x3f), b0f: splat32(0x0f),
	w00ff:    [16]uint16{0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff},
	unitBits: [8]uint32{1, 2, 4, 8, 16, 32, 64, 128},
	zip:      [32]int8{0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15, 0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15},
	unzip:    [32]int8{0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15, 0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15},
	levelSel: func() (t [4][32]int8) {
		// dav1d pb_4x1_4x5_4x9_4x13: replicate component c of every unit over
		// the unit's four bytes.
		for c := range t {
			for i := range t[c] {
				t[c][i] = int8(i&^3&15 + c)
			}
		}
		return t
	}(),
}

func splat32(v uint8) (a [32]uint8) {
	for i := range a {
		a[i] = v
	}
	return a
}

func avxB(a *[32]uint8) archsimd.Uint8x32 { return archsimd.LoadUint8x32Array(a) }

func avxAbsd(a, b archsimd.Uint8x32) archsimd.Uint8x32 {
	return a.SubSaturated(b).Or(b.SubSaturated(a))
}

// avxLE returns all-ones lanes where x <= lim (unsigned).
func avxLE(x, lim archsimd.Uint8x32) archsimd.Uint8x32 {
	return x.SubSaturated(lim).Equal(archsimd.Uint8x32{}).ToInt8x32().ToBits()
}

// avxSel returns a where m is all-ones and b elsewhere (VPBLENDVB).
func avxSel(m, a, b archsimd.Uint8x32) archsimd.Uint8x32 {
	return a.IfElse(archsimd.Int8x32{}.Greater(m.BitsToInt8()), b)
}

// avxSra3 is the signed byte shift x >> 3: pand 0xf8, psrlq 3, then the 5-bit
// result is sign-extended with pxor 16, psubb 16.
func avxSra3(x archsimd.Int8x32) archsimd.Int8x32 {
	b10 := avxB(&avxConst.b10)
	return x.ToBits().And(avxB(&avxConst.bf8)).ReshapeToUint64s().ShiftAllRight(3).ReshapeToUint8s().
		Xor(b10).Sub(b10).BitsToInt8()
}

func avxEven(x archsimd.Uint8x32) archsimd.Uint16x16 {
	return x.ReshapeToUint16s().And(archsimd.LoadUint16x16Array(&avxConst.w00ff))
}

func avxOdd(x archsimd.Uint8x32) archsimd.Uint16x16 {
	return x.ReshapeToUint16s().ShiftAllRight(8)
}

// avxJoin merges the even- and odd-byte results (each 0..255) back into bytes.
func avxJoin(even, odd archsimd.Uint16x16) archsimd.Uint8x32 {
	return even.Or(odd.ShiftAllLeft(8)).ReshapeToUint8s()
}

// avxLpfCore is dav1d's FILTER for widths 4, 6 and 8 on 32 lanes held in
// registers: p3..q3 in, the filtered p2..q2 out (p3/q3 never change at these
// widths). mFlat returns the lanes the flat filter took (fm & flat8in on
// lanes wider than 4); ok is false when no lane passed the filter mask.
func avxLpfCore[W avxWidth](p3, p2, p1, p0, q0, q1, q2, q3, eV, iV, hV, m4, mGt4 archsimd.Uint8x32) (o2, o1, o0, n0, n1, n2, mFlat archsimd.Uint8x32, ok bool) {
	var wdT W
	wd := int(unsafe.Sizeof(wdT))
	o2, o1, o0, n0, n1, n2 = p2, p1, p0, q0, q1, q2
	maxd := avxAbsd(p1, p0).Max(avxAbsd(q1, q0))
	t := maxd
	if wd >= 6 {
		tt := avxAbsd(p2, p1).Max(avxAbsd(q2, q1))
		if wd >= 8 {
			tt = tt.Max(avxAbsd(p3, p2)).Max(avxAbsd(q3, q2))
		}
		t = t.Max(tt.And(mGt4))
	}
	d00 := avxAbsd(p0, q0)
	d11 := avxAbsd(p1, q1).ReshapeToUint64s().ShiftAllRight(1).ReshapeToUint8s().And(avxB(&avxConst.b7f))
	sum := d00.AddSaturated(d00).AddSaturated(d11)
	fm := t.SubSaturated(iV).Or(sum.SubSaturated(eV)).Equal(archsimd.Uint8x32{}).ToInt8x32().ToBits().And(m4)
	if fm.IsZero() {
		return
	}
	ok = true
	f4 := fm
	if wd >= 6 {
		fl := avxAbsd(p2, p0).Max(maxd).Max(avxAbsd(q2, q0))
		if wd >= 8 {
			fl = fl.Max(avxAbsd(p3, p0)).Max(avxAbsd(q3, q0))
		}
		mFlat = avxLE(fl, avxB(&avxConst.b01)).And(mGt4).And(fm)
		f4 = fm.AndNot(mFlat)
	}
	if !f4.IsZero() {
		bias := avxB(&avxConst.b80)
		p1s := p1.Xor(bias).BitsToInt8()
		p0s := p0.Xor(bias).BitsToInt8()
		q0s := q0.Xor(bias).BitsToInt8()
		q1s := q1.Xor(bias).BitsToInt8()
		notHev := avxLE(maxd, hV).BitsToInt8()
		f := p1s.SubSaturated(q1s).AndNot(notHev) // iclip_diff(p1-q1) & hev
		d := q0s.SubSaturated(p0s)
		f = f.AddSaturated(d).AddSaturated(d).AddSaturated(d).And(f4.BitsToInt8())
		f1 := avxSra3(f.AddSaturated(avxB(&avxConst.b04).BitsToInt8()))
		f2 := avxSra3(f.AddSaturated(avxB(&avxConst.b03).BitsToInt8()))
		o0 = p0s.AddSaturated(f2).ToBits().Xor(bias)
		n0 = q0s.SubSaturated(f1).ToBits().Xor(bias)
		// (f1+1)>>1 = pavgb(f1^0x80, 0) - 64, applied where !hev.
		outer := f1.ToBits().Xor(bias).Average(archsimd.Uint8x32{}).Sub(avxB(&avxConst.b40)).BitsToInt8().And(notHev)
		o1 = p1s.AddSaturated(outer).ToBits().Xor(bias)
		n1 = q1s.SubSaturated(outer).ToBits().Xor(bias)
	}
	if wd < 6 || mFlat.IsZero() {
		return
	}
	if wd == 6 {
		ap1, ap0, aq0, aq1 := avxFlat6Half(avxEven(p2), avxEven(p1), avxEven(p0), avxEven(q0), avxEven(q1), avxEven(q2))
		bp1, bp0, bq0, bq1 := avxFlat6Half(avxOdd(p2), avxOdd(p1), avxOdd(p0), avxOdd(q0), avxOdd(q1), avxOdd(q2))
		o1 = avxSel(mFlat, avxJoin(ap1, bp1), o1)
		o0 = avxSel(mFlat, avxJoin(ap0, bp0), o0)
		n0 = avxSel(mFlat, avxJoin(aq0, bq0), n0)
		n1 = avxSel(mFlat, avxJoin(aq1, bq1), n1)
		return
	}
	ap2, ap1, ap0, aq0, aq1, aq2 := avxFlat8Half(avxEven(p3), avxEven(p2), avxEven(p1), avxEven(p0), avxEven(q0), avxEven(q1), avxEven(q2), avxEven(q3))
	bp2, bp1, bp0, bq0, bq1, bq2 := avxFlat8Half(avxOdd(p3), avxOdd(p2), avxOdd(p1), avxOdd(p0), avxOdd(q0), avxOdd(q1), avxOdd(q2), avxOdd(q3))
	o2 = avxSel(mFlat, avxJoin(ap2, bp2), o2)
	o1 = avxSel(mFlat, avxJoin(ap1, bp1), o1)
	o0 = avxSel(mFlat, avxJoin(ap0, bp0), o0)
	n0 = avxSel(mFlat, avxJoin(aq0, bq0), n0)
	n1 = avxSel(mFlat, avxJoin(aq1, bq1), n1)
	n2 = avxSel(mFlat, avxJoin(aq2, bq2), n2)
	return
}

// avxFlat6Half is the 5-tap chroma flat filter (dav1d wd6) on sixteen widened
// lanes, as a sliding sum; each output is already rounded and shifted.
func avxFlat6Half(p2, p1, p0, q0, q1, q2 archsimd.Uint16x16) (op1, op0, oq0, oq1 archsimd.Uint16x16) {
	s := p2.Add(p2).Add(p2).Add(p1.Add(p0).Add(p1.Add(p0))).Add(q0).Add(archsimd.BroadcastUint16x16(4))
	op1 = s.ShiftAllRight(3)
	s = s.Add(q0.Add(q1)).Sub(p2.Add(p2))
	op0 = s.ShiftAllRight(3)
	s = s.Add(q1.Add(q2)).Sub(p2.Add(p1))
	oq0 = s.ShiftAllRight(3)
	s = s.Add(q2.Add(q2)).Sub(p1.Add(p0))
	oq1 = s.ShiftAllRight(3)
	return
}

// avxFlat8Half is the 7-tap flat filter (dav1d wd8) on sixteen widened lanes.
func avxFlat8Half(p3, p2, p1, p0, q0, q1, q2, q3 archsimd.Uint16x16) (op2, op1, op0, oq0, oq1, oq2 archsimd.Uint16x16) {
	s := p3.Add(p3).Add(p3).Add(p2.Add(p2)).Add(p1.Add(p0)).Add(q0.Add(archsimd.BroadcastUint16x16(4)))
	op2 = s.ShiftAllRight(3)
	s = s.Add(p1.Add(q1)).Sub(p3.Add(p2))
	op1 = s.ShiftAllRight(3)
	s = s.Add(p0.Add(q2)).Sub(p3.Add(p1))
	op0 = s.ShiftAllRight(3)
	s = s.Add(q0.Add(q3)).Sub(p3.Add(p0))
	oq0 = s.ShiftAllRight(3)
	s = s.Add(q1.Add(q3)).Sub(p2.Add(q0))
	oq1 = s.ShiftAllRight(3)
	s = s.Add(q2.Add(q3)).Sub(p1.Add(q1))
	oq2 = s.ShiftAllRight(3)
	return
}

// avxFlat16Half runs the dav1d sliding-sum form of the 13-tap filter on
// sixteen widened lanes (t[0..13] = p6..q6); out[k] is output p5+k (p5..q5),
// rounded and shifted.
func avxFlat16Half(t *[14]archsimd.Uint16x16, out *[12]archsimd.Uint16x16) {
	p6, p5, p4, p3, p2, p1, p0 := t[0], t[1], t[2], t[3], t[4], t[5], t[6]
	q0, q1, q2, q3, q4, q5, q6 := t[7], t[8], t[9], t[10], t[11], t[12], t[13]
	p6x2 := p6.Add(p6)
	s := p6.Add(p6x2).Add(p6x2.Add(p6x2)) // 7*p6
	s = s.Add(p5.Add(p5)).Add(p4.Add(p4)).Add(p3.Add(p2)).Add(p1.Add(p0)).Add(q0).Add(archsimd.BroadcastUint16x16(8))
	out[0] = s.ShiftAllRight(4)
	s = s.Add(p3.Add(q1)).Sub(p6x2)
	out[1] = s.ShiftAllRight(4)
	s = s.Add(p2.Add(q2)).Sub(p6.Add(p5))
	out[2] = s.ShiftAllRight(4)
	s = s.Add(p1.Add(q3)).Sub(p6.Add(p4))
	out[3] = s.ShiftAllRight(4)
	s = s.Add(p0.Add(q4)).Sub(p6.Add(p3))
	out[4] = s.ShiftAllRight(4)
	s = s.Add(q0.Add(q5)).Sub(p6.Add(p2))
	out[5] = s.ShiftAllRight(4)
	s = s.Add(q1.Add(q6)).Sub(p6.Add(p1))
	out[6] = s.ShiftAllRight(4)
	s = s.Add(q2.Add(q6)).Sub(p5.Add(p0))
	out[7] = s.ShiftAllRight(4)
	s = s.Add(q3.Add(q6)).Sub(p4.Add(q0))
	out[8] = s.ShiftAllRight(4)
	s = s.Add(q4.Add(q6)).Sub(p3.Add(q1))
	out[9] = s.ShiftAllRight(4)
	s = s.Add(q5.Add(q6)).Sub(p2.Add(q2))
	out[10] = s.ShiftAllRight(4)
	s = s.Add(q6.Add(q6)).Sub(p1.Add(q3))
	out[11] = s.ShiftAllRight(4)
}

// avxLpf16 is dav1d FILTER 16 on taps t[0..15] = p7..q7 (t[8] is q0): the
// width-8 core with the 13-tap flat filter on the lanes whose own width is 16
// and whose outer taps are flat too. The filtered p5..q5 replace t[2..13].
func avxLpf16(t *[16]archsimd.Uint8x32, eV, iV, hV, m4, mGt4, m16 archsimd.Uint8x32) bool {
	o2, o1, o0, n0, n1, n2, mFlat, ok := avxLpfCore[avxW8](t[4], t[5], t[6], t[7], t[8], t[9], t[10], t[11], eV, iV, hV, m4, mGt4)
	if !ok {
		return false
	}
	m := mFlat.And(m16)
	if !m.IsZero() {
		p0, q0 := t[7], t[8]
		fo := avxAbsd(t[1], p0).Max(avxAbsd(t[2], p0)).Max(avxAbsd(t[3], p0)).
			Max(avxAbsd(t[12], q0)).Max(avxAbsd(t[13], q0)).Max(avxAbsd(t[14], q0))
		m = m.And(avxLE(fo, avxB(&avxConst.b01)))
	}
	if !m.IsZero() {
		var w [14]archsimd.Uint16x16
		var ev, od [12]archsimd.Uint16x16
		for k := range w {
			w[k] = avxEven(t[k+1])
		}
		avxFlat16Half(&w, &ev)
		for k := range w {
			w[k] = avxOdd(t[k+1])
		}
		avxFlat16Half(&w, &od)
		t[2] = avxSel(m, avxJoin(ev[0], od[0]), t[2])
		t[3] = avxSel(m, avxJoin(ev[1], od[1]), t[3])
		t[4] = avxSel(m, avxJoin(ev[2], od[2]), t[4])
		o2 = avxSel(m, avxJoin(ev[3], od[3]), o2)
		o1 = avxSel(m, avxJoin(ev[4], od[4]), o1)
		o0 = avxSel(m, avxJoin(ev[5], od[5]), o0)
		n0 = avxSel(m, avxJoin(ev[6], od[6]), n0)
		n1 = avxSel(m, avxJoin(ev[7], od[7]), n1)
		n2 = avxSel(m, avxJoin(ev[8], od[8]), n2)
		t[11] = avxSel(m, avxJoin(ev[9], od[9]), t[11])
		t[12] = avxSel(m, avxJoin(ev[10], od[10]), t[12])
		t[13] = avxSel(m, avxJoin(ev[11], od[11]), t[13])
	}
	t[5], t[6], t[7], t[8], t[9], t[10] = o2, o1, o0, n0, n1, n2
	return true
}

// avxAcc carries goav1's apply counters through a superblock call; dav1d
// keeps no such counters.
type avxAcc struct {
	applied uint32
	max     archsimd.Uint8x32
}

func (a *avxAcc) stats() SBStats {
	var b [16]uint8
	a.max.GetLo().Max(a.max.GetHi()).StoreArray(&b)
	return SBStats{Applied: a.applied, MaxLevel: max(b[0], b[4], b[8], b[12])}
}

// avxUnitMask expands a vmask byte to 32 lanes, unit k owning lanes 4k..4k+3
// (dav1d vpbroadcastd + pand pb_mask + pcmpeqd).
func avxUnitMask(n uint32) archsimd.Uint8x32 {
	b := archsimd.LoadUint32x8Array(&avxConst.unitBits)
	return archsimd.BroadcastUint32x8(n).And(b).Equal(b).ToInt32x8().ToBits().ReshapeToUint8s()
}

// avxGroupLimits is the prologue of dav1d's FILTER for one 8-unit group: cur
// and prev hold the units' own and fallback levels (four components per
// unit); L = cur ? cur : prev for component comp, replicated over the unit's
// four lanes, I = imax(imin(L >> sharp[0], sharp[1]), 1), E = 2*(L+2)+I and
// H = L>>4. m4 is the vmask byte restricted to L != 0; ok is false when no
// selected unit has a level.
func avxGroupLimits(cur, prev archsimd.Uint8x32, sel archsimd.Int8x32, nAll uint32, sharp0 uint64, sharp1 archsimd.Uint8x32, acc *avxAcc) (eV, iV, hV, m4 archsimd.Uint8x32, ok bool) {
	var zero archsimd.Uint8x32
	c := cur.PermuteOrZeroGrouped(sel)
	p := prev.PermuteOrZeroGrouped(sel)
	L := c.Or(p.And(c.Equal(zero).ToInt8x32().ToBits()))
	act := avxUnitMask(nAll).AndNot(L.Equal(zero).ToInt8x32().ToBits())
	if act.IsZero() {
		return
	}
	acc.applied += uint32(bits.OnesCount32(archsimd.Int8x32{}.Greater(act.BitsToInt8()).ToBits())) >> 2
	acc.max = acc.max.Max(L.And(act))
	hV = L.ReshapeToUint64s().ShiftAllRight(4).ReshapeToUint8s().And(avxB(&avxConst.b0f))
	iV = L.ReshapeToUint64s().ShiftAllRight(sharp0).ReshapeToUint8s().And(avxB(&avxConst.b3f)).
		Min(sharp1).Max(avxB(&avxConst.b01))
	e := L.Add(archsimd.BroadcastUint8x32(2))
	eV = e.Add(e).Add(iV)
	m4 = act
	ok = true
	return
}

// avxSBWidth is the kernel choice from the raw vmask bytes.
func avxSBWidth(nGt4, n16 uint32, luma bool) int {
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

func avxLevelBytes(lvl [][4]uint8) []byte {
	return unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(lvl))), 4*len(lvl))
}

// avxLevelsRow loads the levels of eight consecutive units (horizontal
// edges: dav1d lpf_v_sb movu of l[0] and l[-b4_stride]).
func avxLevelsRow(lb []byte, lvl [][4]uint8, u0, lvlStride int, nAll uint32) (cur, prev archsimd.Uint8x32) {
	if u0-lvlStride >= 0 && u0+8 <= len(lvl) {
		return archsimd.LoadUint8x32(lb[4*u0:]), archsimd.LoadUint8x32(lb[4*(u0-lvlStride):])
	}
	var c, p [8]uint32
	for k := 0; k < 8; k++ {
		if nAll&(1<<uint(k)) != 0 {
			c[k] = binary.LittleEndian.Uint32(lvl[u0+k][:])
			p[k] = binary.LittleEndian.Uint32(lvl[u0+k-lvlStride][:])
		}
	}
	return archsimd.LoadUint32x8Array(&c).ReshapeToUint8s(), archsimd.LoadUint32x8Array(&p).ReshapeToUint8s()
}

// avxLevelsCol gathers the levels of eight units stacked lvlStride cells
// apart and of their left neighbours (vertical edges: dav1d lpf_h_sb).
func avxLevelsCol(lvl [][4]uint8, li, lvlStride int, nAll uint32) (cur, prev archsimd.Uint8x32) {
	var c, p [8]uint32
	if li-1 >= 0 && li+7*lvlStride < len(lvl) {
		for k := 0; k < 8; k++ {
			i := li + k*lvlStride
			c[k] = binary.LittleEndian.Uint32(lvl[i][:])
			p[k] = binary.LittleEndian.Uint32(lvl[i-1][:])
		}
	} else {
		for k := 0; k < 8; k++ {
			if nAll&(1<<uint(k)) != 0 {
				i := li + k*lvlStride
				c[k] = binary.LittleEndian.Uint32(lvl[i][:])
				p[k] = binary.LittleEndian.Uint32(lvl[i-1][:])
			}
		}
	}
	return archsimd.LoadUint32x8Array(&c).ReshapeToUint8s(), archsimd.LoadUint32x8Array(&p).ReshapeToUint8s()
}

func sbGroupSub(vm [3]uint32, sh uint) [3]uint32 {
	for k := range vm {
		vm[k] = (vm[k] >> sh) & 0xff
	}
	return vm
}

func filterSBHorizontal8(pix []byte, off, stride, rowBytes int, vm [3]uint32, luma bool, lvl [][4]uint8, lvlOff, lvlStride, comp int, lut *SBLUT) SBStats {
	if !sbAVX2 {
		return filterSBHorizontal8Scalar(pix, off, stride, vm, luma, lvl, lvlOff, lvlStride, comp, lut)
	}
	st, _ := avxSBHorizontal8(pix, off, stride, rowBytes, vm, luma, lvl, lvlOff, lvlStride, comp, lut)
	return st
}

func filterSBVertical8(pix []byte, off, stride, rowBytes int, vm [3]uint32, luma bool, lvl [][4]uint8, lvlOff, lvlStride, comp int, lut *SBLUT) SBStats {
	if !sbAVX2 {
		return filterSBVertical8Scalar(pix, off, stride, vm, luma, lvl, lvlOff, lvlStride, comp, lut)
	}
	st, _ := avxSBVertical8(pix, off, stride, rowBytes, vm, luma, lvl, lvlOff, lvlStride, comp, lut)
	return st
}

// avxSBHorizontal8 is dav1d lpf_v_sb_{y,uv}_8bpc: the horizontal edges of one
// row of units, 32 columns per kernel call. Groups that would cross rowBytes,
// the plane or the tap reach take the unit-by-unit reference instead. simd
// counts the groups the vector kernel handled.
func avxSBHorizontal8(pix []byte, off, stride, rowBytes int, vm [3]uint32, luma bool, lvl [][4]uint8, lvlOff, lvlStride, comp int, lut *SBLUT) (st SBStats, simd int) {
	var acc avxAcc
	gt4All := vm[1] | vm[2]
	all := vm[0] | gt4All
	lb := avxLevelBytes(lvl)
	sel := archsimd.LoadInt8x32Array(&avxConst.levelSel[comp&3])
	sharp0, sharp1 := uint64(lut.Sharp[0]), archsimd.BroadcastUint8x32(lut.Sharp[1])
	for g := 0; g < 4 && all>>(8*uint(g)) != 0; g++ {
		sh := 8 * uint(g)
		nib := (all >> sh) & 0xff
		if nib == 0 {
			continue
		}
		nGt4 := (gt4All >> sh) & 0xff
		n16 := (vm[2] >> sh) & 0xff
		wd := avxSBWidth(nGt4, n16, luma)
		u0 := lvlOff + 8*g
		q0Off := off + 32*g
		reach := 4
		if wd == 16 {
			reach = 7
		}
		if 32*g+32 > rowBytes || q0Off-reach*stride < 0 || q0Off+(reach-1)*stride+32 > len(pix) {
			st.Add(filterSBHorizontal8Scalar(pix, q0Off, stride, sbGroupSub(vm, sh), luma, lvl, u0, lvlStride, comp, lut))
			continue
		}
		cur, prev := avxLevelsRow(lb, lvl, u0, lvlStride, nib)
		eV, iV, hV, m4, ok := avxGroupLimits(cur, prev, sel, nib, sharp0, sharp1, &acc)
		if !ok {
			continue
		}
		simd++
		mGt4 := avxUnitMask(nGt4)
		if wd == 16 {
			var t [16]archsimd.Uint8x32
			for k := 1; k < 15; k++ {
				t[k] = archsimd.LoadUint8x32(pix[q0Off+(k-8)*stride:])
			}
			if avxLpf16(&t, eV, iV, hV, m4, mGt4, avxUnitMask(n16)) {
				for k := 2; k < 14; k++ {
					t[k].Store(pix[q0Off+(k-8)*stride:])
				}
			}
			continue
		}
		p1 := archsimd.LoadUint8x32(pix[q0Off-2*stride:])
		p0 := archsimd.LoadUint8x32(pix[q0Off-stride:])
		q0 := archsimd.LoadUint8x32(pix[q0Off:])
		q1 := archsimd.LoadUint8x32(pix[q0Off+stride:])
		switch wd {
		case 8:
			p3 := archsimd.LoadUint8x32(pix[q0Off-4*stride:])
			p2 := archsimd.LoadUint8x32(pix[q0Off-3*stride:])
			q2 := archsimd.LoadUint8x32(pix[q0Off+2*stride:])
			q3 := archsimd.LoadUint8x32(pix[q0Off+3*stride:])
			o2, o1, o0, n0, n1, n2, _, ok := avxLpfCore[avxW8](p3, p2, p1, p0, q0, q1, q2, q3, eV, iV, hV, m4, mGt4)
			if ok {
				o2.Store(pix[q0Off-3*stride:])
				o1.Store(pix[q0Off-2*stride:])
				o0.Store(pix[q0Off-stride:])
				n0.Store(pix[q0Off:])
				n1.Store(pix[q0Off+stride:])
				n2.Store(pix[q0Off+2*stride:])
			}
		case 6:
			p2 := archsimd.LoadUint8x32(pix[q0Off-3*stride:])
			q2 := archsimd.LoadUint8x32(pix[q0Off+2*stride:])
			_, o1, o0, n0, n1, _, _, ok := avxLpfCore[avxW6](p2, p2, p1, p0, q0, q1, q2, q2, eV, iV, hV, m4, mGt4)
			if ok {
				o1.Store(pix[q0Off-2*stride:])
				o0.Store(pix[q0Off-stride:])
				n0.Store(pix[q0Off:])
				n1.Store(pix[q0Off+stride:])
			}
		default:
			_, o1, o0, n0, n1, _, _, ok := avxLpfCore[avxW4](p1, p1, p1, p0, q0, q1, q1, q1, eV, iV, hV, m4, mGt4)
			if ok {
				o1.Store(pix[q0Off-2*stride:])
				o0.Store(pix[q0Off-stride:])
				n0.Store(pix[q0Off:])
				n1.Store(pix[q0Off+stride:])
			}
		}
	}
	st.Add(acc.stats())
	return st, simd
}

// avxTransposeWords transposes, independently in each 128-bit lane, the 8x8
// matrix of 16-bit words held in b0..b7 (punpck{l,h}{wd,dq,qdq}). It is its
// own inverse.
func avxTransposeWords(b0, b1, b2, b3, b4, b5, b6, b7 archsimd.Uint8x32) (t0, t1, t2, t3, t4, t5, t6, t7 archsimd.Uint8x32) {
	c0 := b0.ReshapeToUint16s().InterleaveLoGrouped(b1.ReshapeToUint16s()).ReshapeToUint32s()
	c1 := b0.ReshapeToUint16s().InterleaveHiGrouped(b1.ReshapeToUint16s()).ReshapeToUint32s()
	c2 := b2.ReshapeToUint16s().InterleaveLoGrouped(b3.ReshapeToUint16s()).ReshapeToUint32s()
	c3 := b2.ReshapeToUint16s().InterleaveHiGrouped(b3.ReshapeToUint16s()).ReshapeToUint32s()
	c4 := b4.ReshapeToUint16s().InterleaveLoGrouped(b5.ReshapeToUint16s()).ReshapeToUint32s()
	c5 := b4.ReshapeToUint16s().InterleaveHiGrouped(b5.ReshapeToUint16s()).ReshapeToUint32s()
	c6 := b6.ReshapeToUint16s().InterleaveLoGrouped(b7.ReshapeToUint16s()).ReshapeToUint32s()
	c7 := b6.ReshapeToUint16s().InterleaveHiGrouped(b7.ReshapeToUint16s()).ReshapeToUint32s()
	d0 := c0.InterleaveLoGrouped(c2).ReshapeToUint64s()
	d1 := c0.InterleaveHiGrouped(c2).ReshapeToUint64s()
	d2 := c1.InterleaveLoGrouped(c3).ReshapeToUint64s()
	d3 := c1.InterleaveHiGrouped(c3).ReshapeToUint64s()
	d4 := c4.InterleaveLoGrouped(c6).ReshapeToUint64s()
	d5 := c4.InterleaveHiGrouped(c6).ReshapeToUint64s()
	d6 := c5.InterleaveLoGrouped(c7).ReshapeToUint64s()
	d7 := c5.InterleaveHiGrouped(c7).ReshapeToUint64s()
	t0 = d0.InterleaveLoGrouped(d4).ReshapeToUint8s()
	t1 = d0.InterleaveHiGrouped(d4).ReshapeToUint8s()
	t2 = d1.InterleaveLoGrouped(d5).ReshapeToUint8s()
	t3 = d1.InterleaveHiGrouped(d5).ReshapeToUint8s()
	t4 = d2.InterleaveLoGrouped(d6).ReshapeToUint8s()
	t5 = d2.InterleaveHiGrouped(d6).ReshapeToUint8s()
	t6 = d3.InterleaveLoGrouped(d7).ReshapeToUint8s()
	t7 = d3.InterleaveHiGrouped(d7).ReshapeToUint8s()
	return
}

// avxRows2 loads row r into the low lane and row r+16 into the high lane.
func avxRows2(pix []byte, o, stride int) archsimd.Uint8x32 {
	return archsimd.Uint8x32{}.SetLo(archsimd.LoadUint8x16(pix[o:])).SetHi(archsimd.LoadUint8x16(pix[o+16*stride:]))
}

// avxZipRows pairs rows a and b (their low eight bytes when hi is false, the
// high eight otherwise) and interleaves them bytewise: punpck{l,h}bw a, b.
func avxZipRows(a, b archsimd.Uint8x32, zip archsimd.Int8x32, hi bool) archsimd.Uint8x32 {
	if hi {
		return a.ReshapeToUint64s().InterleaveHiGrouped(b.ReshapeToUint64s()).ReshapeToUint8s().PermuteOrZeroGrouped(zip)
	}
	return a.ReshapeToUint64s().InterleaveLoGrouped(b.ReshapeToUint64s()).ReshapeToUint8s().PermuteOrZeroGrouped(zip)
}

// avxStoreRowPair writes a de-interleaved row pair: the low eight bytes of
// each lane are row 2j (lane 0) / 2j+16 (lane 1), the high eight row 2j+1 /
// 2j+17.
func avxStoreRowPair(pix []byte, o, stride int, a archsimd.Uint8x32) {
	lo, hi := a.GetLo().ReshapeToUint64s(), a.GetHi().ReshapeToUint64s()
	binary.LittleEndian.PutUint64(pix[o:], lo.GetElem(0))
	binary.LittleEndian.PutUint64(pix[o+stride:], lo.GetElem(1))
	binary.LittleEndian.PutUint64(pix[o+16*stride:], hi.GetElem(0))
	binary.LittleEndian.PutUint64(pix[o+17*stride:], hi.GetElem(1))
}

// avxVertical8 filters a vertical edge across 32 rows for widths up to 8
// (dav1d FILTER {4,6,8}, h): p3 is the offset of p3 on the first row; sixteen
// bytes are read per row and the eight p3..q3 bytes are written back.
func avxVertical8[W avxWidth](pix []byte, p3, stride int, eV, iV, hV, m4, mGt4 archsimd.Uint8x32) {
	zip := archsimd.LoadInt8x32Array(&avxConst.zip)
	y0, y1 := avxRows2(pix, p3, stride), avxRows2(pix, p3+stride, stride)
	y2, y3 := avxRows2(pix, p3+2*stride, stride), avxRows2(pix, p3+3*stride, stride)
	y4, y5 := avxRows2(pix, p3+4*stride, stride), avxRows2(pix, p3+5*stride, stride)
	y6, y7 := avxRows2(pix, p3+6*stride, stride), avxRows2(pix, p3+7*stride, stride)
	y8, y9 := avxRows2(pix, p3+8*stride, stride), avxRows2(pix, p3+9*stride, stride)
	y10, y11 := avxRows2(pix, p3+10*stride, stride), avxRows2(pix, p3+11*stride, stride)
	y12, y13 := avxRows2(pix, p3+12*stride, stride), avxRows2(pix, p3+13*stride, stride)
	y14, y15 := avxRows2(pix, p3+14*stride, stride), avxRows2(pix, p3+15*stride, stride)
	t0, t1, t2, t3, t4, t5, t6, t7 := avxTransposeWords(
		avxZipRows(y0, y1, zip, false), avxZipRows(y2, y3, zip, false),
		avxZipRows(y4, y5, zip, false), avxZipRows(y6, y7, zip, false),
		avxZipRows(y8, y9, zip, false), avxZipRows(y10, y11, zip, false),
		avxZipRows(y12, y13, zip, false), avxZipRows(y14, y15, zip, false))
	o2, o1, o0, n0, n1, n2, _, ok := avxLpfCore[W](t0, t1, t2, t3, t4, t5, t6, t7, eV, iV, hV, m4, mGt4)
	if !ok {
		return
	}
	b0, b1, b2, b3, b4, b5, b6, b7 := avxTransposeWords(t0, o2, o1, o0, n0, n1, n2, t7)
	unzip := archsimd.LoadInt8x32Array(&avxConst.unzip)
	avxStoreRowPair(pix, p3, stride, b0.PermuteOrZeroGrouped(unzip))
	avxStoreRowPair(pix, p3+2*stride, stride, b1.PermuteOrZeroGrouped(unzip))
	avxStoreRowPair(pix, p3+4*stride, stride, b2.PermuteOrZeroGrouped(unzip))
	avxStoreRowPair(pix, p3+6*stride, stride, b3.PermuteOrZeroGrouped(unzip))
	avxStoreRowPair(pix, p3+8*stride, stride, b4.PermuteOrZeroGrouped(unzip))
	avxStoreRowPair(pix, p3+10*stride, stride, b5.PermuteOrZeroGrouped(unzip))
	avxStoreRowPair(pix, p3+12*stride, stride, b6.PermuteOrZeroGrouped(unzip))
	avxStoreRowPair(pix, p3+14*stride, stride, b7.PermuteOrZeroGrouped(unzip))
}

// avxVertical16 is the wd16 variant: sixteen bytes per row (p7..q7, p7 is the
// offset of p7 on the first row), a full 16x16 byte transpose per lane each
// way (dav1d TRANSPOSE_16X16B).
func avxVertical16(pix []byte, p7, stride int, eV, iV, hV, m4, mGt4, m16 archsimd.Uint8x32) {
	zip := archsimd.LoadInt8x32Array(&avxConst.zip)
	var y [16]archsimd.Uint8x32
	for k := range y {
		y[k] = avxRows2(pix, p7+k*stride, stride)
	}
	var t [16]archsimd.Uint8x32
	t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7] = avxTransposeWords(
		avxZipRows(y[0], y[1], zip, false), avxZipRows(y[2], y[3], zip, false),
		avxZipRows(y[4], y[5], zip, false), avxZipRows(y[6], y[7], zip, false),
		avxZipRows(y[8], y[9], zip, false), avxZipRows(y[10], y[11], zip, false),
		avxZipRows(y[12], y[13], zip, false), avxZipRows(y[14], y[15], zip, false))
	t[8], t[9], t[10], t[11], t[12], t[13], t[14], t[15] = avxTransposeWords(
		avxZipRows(y[0], y[1], zip, true), avxZipRows(y[2], y[3], zip, true),
		avxZipRows(y[4], y[5], zip, true), avxZipRows(y[6], y[7], zip, true),
		avxZipRows(y[8], y[9], zip, true), avxZipRows(y[10], y[11], zip, true),
		avxZipRows(y[12], y[13], zip, true), avxZipRows(y[14], y[15], zip, true))
	if !avxLpf16(&t, eV, iV, hV, m4, mGt4, m16) {
		return
	}
	var lo, hi [8]archsimd.Uint8x32
	lo[0], lo[1], lo[2], lo[3], lo[4], lo[5], lo[6], lo[7] = avxTransposeWords(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7])
	hi[0], hi[1], hi[2], hi[3], hi[4], hi[5], hi[6], hi[7] = avxTransposeWords(t[8], t[9], t[10], t[11], t[12], t[13], t[14], t[15])
	unzip := archsimd.LoadInt8x32Array(&avxConst.unzip)
	for j := 0; j < 8; j++ {
		l := lo[j].PermuteOrZeroGrouped(unzip).ReshapeToUint64s()
		h := hi[j].PermuteOrZeroGrouped(unzip).ReshapeToUint64s()
		r0 := l.InterleaveLoGrouped(h).ReshapeToUint8s()
		r1 := l.InterleaveHiGrouped(h).ReshapeToUint8s()
		o := p7 + 2*j*stride
		r0.GetLo().Store(pix[o:])
		r0.GetHi().Store(pix[o+16*stride:])
		r1.GetLo().Store(pix[o+stride:])
		r1.GetHi().Store(pix[o+17*stride:])
	}
}

// avxSBVertical8 is dav1d lpf_h_sb_{y,uv}_8bpc: the vertical edges of one
// column of units, 32 rows per kernel call. Groups whose rows do not fit the
// footprint arm64 uses (sixteen bytes from p3, or p7 at width 16) take the
// unit-by-unit reference.
func avxSBVertical8(pix []byte, off, stride, rowBytes int, vm [3]uint32, luma bool, lvl [][4]uint8, lvlOff, lvlStride, comp int, lut *SBLUT) (st SBStats, simd int) {
	var acc avxAcc
	gt4All := vm[1] | vm[2]
	all := vm[0] | gt4All
	sel := archsimd.LoadInt8x32Array(&avxConst.levelSel[comp&3])
	sharp0, sharp1 := uint64(lut.Sharp[0]), archsimd.BroadcastUint8x32(lut.Sharp[1])
	for g := 0; g < 4 && all>>(8*uint(g)) != 0; g++ {
		sh := 8 * uint(g)
		nib := (all >> sh) & 0xff
		if nib == 0 {
			continue
		}
		nGt4 := (gt4All >> sh) & 0xff
		n16 := (vm[2] >> sh) & 0xff
		wd := avxSBWidth(nGt4, n16, luma)
		u0 := lvlOff + 8*g*lvlStride
		q0Off := off + 32*g*stride
		lastRow := q0Off + 31*stride
		if wd == 16 {
			if q0Off-8 < 0 || lastRow+8 > len(pix) {
				st.Add(filterSBVertical8Scalar(pix, q0Off, stride, sbGroupSub(vm, sh), luma, lvl, u0, lvlStride, comp, lut))
				continue
			}
		} else if rowBytes < 12 || q0Off-4 < 0 || lastRow+12 > len(pix) {
			st.Add(filterSBVertical8Scalar(pix, q0Off, stride, sbGroupSub(vm, sh), luma, lvl, u0, lvlStride, comp, lut))
			continue
		}
		cur, prev := avxLevelsCol(lvl, u0, lvlStride, nib)
		eV, iV, hV, m4, ok := avxGroupLimits(cur, prev, sel, nib, sharp0, sharp1, &acc)
		if !ok {
			continue
		}
		simd++
		mGt4 := avxUnitMask(nGt4)
		switch wd {
		case 16:
			avxVertical16(pix, q0Off-8, stride, eV, iV, hV, m4, mGt4, avxUnitMask(n16))
		case 8:
			avxVertical8[avxW8](pix, q0Off-4, stride, eV, iV, hV, m4, mGt4)
		case 6:
			avxVertical8[avxW6](pix, q0Off-4, stride, eV, iV, hV, m4, mGt4)
		default:
			avxVertical8[avxW4](pix, q0Off-4, stride, eV, iV, hV, m4, mGt4)
		}
	}
	st.Add(acc.stats())
	return st, simd
}
