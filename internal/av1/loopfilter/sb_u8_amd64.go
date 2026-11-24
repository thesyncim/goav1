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

// Every plane row is one bounds-checked window, taken from a 32-bit offset
// (the drivers send planes past lfMaxPlane to the scalar reference), which
// proves the offset non-negative and overflow-free: one compare per row. The
// drivers have range-checked every group, so the checks never fail.

// This file ports dav1d's AVX2 superblock deblock drivers and their FILTER
// macro (src/x86/loopfilter_avx2.asm lpf_{v,h}_sb_{y,uv}_8bpc) to archsimd.
// One kernel call covers 32 samples, eight 4x4 units, which is one byte of
// each vmask word, as dav1d's mask-byte loop does: horizontal edges filter 32
// columns of rows straight from the plane, vertical edges transpose 32 rows
// into tap vectors (rows 0..15 in the low 128-bit lane, 16..31 in the high
// one) and back.
//
// AVX2 archsimd lacks a few of the instructions dav1d's assembly uses, so three
// steps are rebuilt from operations it does have, byte-exact:
//   - punpck{l,h}bw: the two rows are first paired into one lane with a 64-bit
//     unpack, and a byte shuffle (VPSHUFB 0,8,1,9,...) interleaves them.
//   - pmaddubsw/pmulhrsw/packuswb in the flat filters: the 16-bit sums run on
//     the even and odd bytes of each 16-bit lane (v & 0xff, v >> 8), and the
//     rounded results are merged back as even | odd<<8. The split is
//     lane-local, so no cross-lane narrowing is needed.
//   - byte shifts: 64-bit shifts plus a mask, as dav1d's assembly does.
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
// dav1d's assembly loads its RODATA.
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
	wd := len(wdT)
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
// sixteen lanes widened from the even or odd bytes of t[1..14] = p6..q6;
// out[k] is output p5+k (p5..q5), rounded and shifted.
func avxFlat16Half(t *[16]archsimd.Uint8x32, odd bool, out *[12]archsimd.Uint16x16) {
	w := func(x archsimd.Uint8x32) archsimd.Uint16x16 {
		if odd {
			return x.ReshapeToUint16s().ShiftAllRight(8)
		}
		return x.ReshapeToUint16s().And(archsimd.LoadUint16x16Array(&avxConst.w00ff))
	}
	p6, p5, p4, p3, p2, p1, p0 := w(t[1]), w(t[2]), w(t[3]), w(t[4]), w(t[5]), w(t[6]), w(t[7])
	q0, q1, q2, q3, q4, q5, q6 := w(t[8]), w(t[9]), w(t[10]), w(t[11]), w(t[12]), w(t[13]), w(t[14])
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
		var ev, od [12]archsimd.Uint16x16
		avxFlat16Half(t, false, &ev)
		avxFlat16Half(t, true, &od)
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
func avxGroupLimits(cur, prev archsimd.Uint8x32, sel archsimd.Int8x32, nAll uint32, sharp0 uint64, sharp1 archsimd.Uint8x32) (eV, iV, hV, m4 archsimd.Uint8x32, ok bool) {
	var zero archsimd.Uint8x32
	c := cur.PermuteOrZeroGrouped(sel)
	p := prev.PermuteOrZeroGrouped(sel)
	L := c.Or(p.And(c.Equal(zero).ToInt8x32().ToBits()))
	act := avxUnitMask(nAll).AndNot(L.Equal(zero).ToInt8x32().ToBits())
	if act.IsZero() {
		return
	}
	hV = L.ReshapeToUint64s().ShiftAllRight(4).ReshapeToUint8s().And(avxB(&avxConst.b0f))
	// sharp[0] is 0..2; immediate shifts keep the count out of a legacy-SSE
	// GPR-to-XMM move.
	lq := L.ReshapeToUint64s()
	switch sharp0 {
	case 0:
	case 1:
		lq = lq.ShiftAllRight(1)
	default:
		lq = lq.ShiftAllRight(2)
	}
	iV = lq.ReshapeToUint8s().And(avxB(&avxConst.b3f)).
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

// avxLevels8 loads the eight level cells from u (dav1d's movu of l[0]): the
// level map holds four bytes per cell, so this is one 32-byte load.
func avxLevels8(lvl []uint8, u int) archsimd.Uint8x32 {
	o := int(uint32(4 * u))
	return archsimd.LoadUint8x32Array((*[32]uint8)(lvl[o : o+32]))
}

// avxLevelsRow loads the levels of eight consecutive units (horizontal
// edges: dav1d lpf_v_sb movu of l[0] and l[-b4_stride]).
func avxLevelsRow(lvl []uint8, u0, lvlStride int, nAll uint32) (cur, prev archsimd.Uint8x32) {
	if u0-lvlStride >= 0 && 4*(u0+8) <= len(lvl) {
		return avxLevels8(lvl, u0), avxLevels8(lvl, u0-lvlStride)
	}
	var c, p [8]uint32
	for k := 0; k < 8; k++ {
		if nAll&(1<<uint(k)) != 0 {
			c[k] = binary.LittleEndian.Uint32(lvl[4*(u0+k):])
			p[k] = binary.LittleEndian.Uint32(lvl[4*(u0+k-lvlStride):])
		}
	}
	return archsimd.LoadUint32x8Array(&c).ReshapeToUint8s(), archsimd.LoadUint32x8Array(&p).ReshapeToUint8s()
}

// avxLevelsCol gathers the levels of eight units stacked lvlStride cells
// apart and of their left neighbours (vertical edges: dav1d lpf_h_sb).
func avxLevelsCol(lvl []uint8, li, lvlStride int, nAll uint32) (cur, prev archsimd.Uint8x32) {
	var c, p [8]uint32
	if li-1 >= 0 && 4*(li+7*lvlStride+1) <= len(lvl) {
		for k := 0; k < 8; k++ {
			i := 4 * (li + k*lvlStride)
			c[k] = binary.LittleEndian.Uint32(lvl[i:])
			p[k] = binary.LittleEndian.Uint32(lvl[i-4:])
		}
	} else {
		for k := 0; k < 8; k++ {
			if nAll&(1<<uint(k)) != 0 {
				i := 4 * (li + k*lvlStride)
				c[k] = binary.LittleEndian.Uint32(lvl[i:])
				p[k] = binary.LittleEndian.Uint32(lvl[i-4:])
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

func filterSBHorizontal8(pix []byte, off, stride, rowBytes int, vm [3]uint32, luma bool, lvl []uint8, lvlOff, lvlStride, comp int, lut *SBLUT) {
	if !sbAVX2 || len(pix) > lfMaxPlane {
		filterSBHorizontal8Scalar(pix, off, stride, vm, luma, lvl, lvlOff, lvlStride, comp, lut)
		return
	}
	avxSBHorizontal8(pix, off, stride, rowBytes, vm, luma, lvl, lvlOff, lvlStride, comp, lut)
	// dav1d's RET: vzeroupper before returning to (legacy-SSE) Go code.
	archsimd.ClearAVXUpperBits()
}

func filterSBVertical8(pix []byte, off, stride, rowBytes int, vm [3]uint32, luma bool, lvl []uint8, lvlOff, lvlStride, comp int, lut *SBLUT) {
	if !sbAVX2 || len(pix) > lfMaxPlane {
		filterSBVertical8Scalar(pix, off, stride, vm, luma, lvl, lvlOff, lvlStride, comp, lut)
		return
	}
	avxSBVertical8(pix, off, stride, rowBytes, vm, luma, lvl, lvlOff, lvlStride, comp, lut)
	// dav1d's RET: vzeroupper before returning to (legacy-SSE) Go code.
	archsimd.ClearAVXUpperBits()
}

// avxR2..avxR16 are the tap-row counts of the kernels' Blocks.
type (
	avxR2  = [2]struct{}
	avxR4  = [4]struct{}
	avxR6  = [6]struct{}
	avxR8  = [8]struct{}
	avxR14 = [14]struct{}
	avxR16 = [16]struct{}
)

// avxHalfRows is the number of tap rows each side of a horizontal edge the
// width-wd kernel reads (seven at wd 16: p6..q6).
func avxHalfRows(wd int) int {
	if wd == 16 {
		return 7
	}
	return wd / 2
}

// avxHorz16 filters the 32 columns of the fourteen tap rows p6..q6 of b
// (rows 0..13) at width 16 and stores the changed rows p5..q5.
func avxHorz16(b unsafeutil.Block32[avxR14], eV, iV, hV, m4, mGt4, m16 archsimd.Uint8x32) {
	var t [16]archsimd.Uint8x32
	t[1] = archsimd.LoadUint8x32Array(b.Row32(0))
	t[2] = archsimd.LoadUint8x32Array(b.Row32(1))
	t[3] = archsimd.LoadUint8x32Array(b.Row32(2))
	t[4] = archsimd.LoadUint8x32Array(b.Row32(3))
	t[5] = archsimd.LoadUint8x32Array(b.Row32(4))
	t[6] = archsimd.LoadUint8x32Array(b.Row32(5))
	t[7] = archsimd.LoadUint8x32Array(b.Row32(6))
	t[8] = archsimd.LoadUint8x32Array(b.Row32(7))
	t[9] = archsimd.LoadUint8x32Array(b.Row32(8))
	t[10] = archsimd.LoadUint8x32Array(b.Row32(9))
	t[11] = archsimd.LoadUint8x32Array(b.Row32(10))
	t[12] = archsimd.LoadUint8x32Array(b.Row32(11))
	t[13] = archsimd.LoadUint8x32Array(b.Row32(12))
	t[14] = archsimd.LoadUint8x32Array(b.Row32(13))
	if !avxLpf16(&t, eV, iV, hV, m4, mGt4, m16) {
		return
	}
	t[2].StoreArray(b.Row32(1))
	t[3].StoreArray(b.Row32(2))
	t[4].StoreArray(b.Row32(3))
	t[5].StoreArray(b.Row32(4))
	t[6].StoreArray(b.Row32(5))
	t[7].StoreArray(b.Row32(6))
	t[8].StoreArray(b.Row32(7))
	t[9].StoreArray(b.Row32(8))
	t[10].StoreArray(b.Row32(9))
	t[11].StoreArray(b.Row32(10))
	t[12].StoreArray(b.Row32(11))
	t[13].StoreArray(b.Row32(12))
}

// avxHorz8 is avxHorz16 at width 8: the tap rows p3..q3 (rows 0..7), the
// changed rows p2..q2 stored.
func avxHorz8(b unsafeutil.Block32[avxR8], eV, iV, hV, m4, mGt4 archsimd.Uint8x32) {
	o2, o1, o0, n0, n1, n2, _, ok := avxLpfCore[avxW8](archsimd.LoadUint8x32Array(b.Row32(0)), archsimd.LoadUint8x32Array(b.Row32(1)),
		archsimd.LoadUint8x32Array(b.Row32(2)), archsimd.LoadUint8x32Array(b.Row32(3)), archsimd.LoadUint8x32Array(b.Row32(4)),
		archsimd.LoadUint8x32Array(b.Row32(5)), archsimd.LoadUint8x32Array(b.Row32(6)), archsimd.LoadUint8x32Array(b.Row32(7)), eV, iV, hV, m4, mGt4)
	if !ok {
		return
	}
	o2.StoreArray(b.Row32(1))
	o1.StoreArray(b.Row32(2))
	o0.StoreArray(b.Row32(3))
	n0.StoreArray(b.Row32(4))
	n1.StoreArray(b.Row32(5))
	n2.StoreArray(b.Row32(6))
}

// avxHorz6 is avxHorz16 at width 6: the tap rows p2..q2 (rows 0..5), the
// changed rows p1..q1 stored.
func avxHorz6(b unsafeutil.Block32[avxR6], eV, iV, hV, m4, mGt4 archsimd.Uint8x32) {
	p2 := archsimd.LoadUint8x32Array(b.Row32(0))
	q2 := archsimd.LoadUint8x32Array(b.Row32(5))
	_, o1, o0, n0, n1, _, _, ok := avxLpfCore[avxW6](p2, p2, archsimd.LoadUint8x32Array(b.Row32(1)), archsimd.LoadUint8x32Array(b.Row32(2)),
		archsimd.LoadUint8x32Array(b.Row32(3)), archsimd.LoadUint8x32Array(b.Row32(4)), q2, q2, eV, iV, hV, m4, mGt4)
	if !ok {
		return
	}
	o1.StoreArray(b.Row32(1))
	o0.StoreArray(b.Row32(2))
	n0.StoreArray(b.Row32(3))
	n1.StoreArray(b.Row32(4))
}

// avxHorz4 is avxHorz16 at width 4: the tap rows p1..q1 (rows 0..3), all
// four stored.
func avxHorz4(b unsafeutil.Block32[avxR4], eV, iV, hV, m4 archsimd.Uint8x32) {
	p1 := archsimd.LoadUint8x32Array(b.Row32(0))
	q1 := archsimd.LoadUint8x32Array(b.Row32(3))
	_, o1, o0, n0, n1, _, _, ok := avxLpfCore[avxW4](p1, p1, p1, archsimd.LoadUint8x32Array(b.Row32(1)), archsimd.LoadUint8x32Array(b.Row32(2)),
		q1, q1, q1, eV, iV, hV, m4, archsimd.Uint8x32{})
	if !ok {
		return
	}
	o1.StoreArray(b.Row32(0))
	o0.StoreArray(b.Row32(1))
	n0.StoreArray(b.Row32(2))
	n1.StoreArray(b.Row32(3))
}

// avxHorzGroup runs the width-wd kernel on the 32 columns at x of v, whose
// row rh is q0 (rh >= avxHalfRows(wd) rows above it, as many below).
func avxHorzGroup(v unsafeutil.Rows, rh, x, wd int, eV, iV, hV, m4, mGt4 archsimd.Uint8x32, n16 uint32) {
	switch wd {
	case 16:
		avxHorz16(unsafeutil.Block32At[avxR14](v, rh-7, x), eV, iV, hV, m4, mGt4, avxUnitMask(n16))
	case 8:
		avxHorz8(unsafeutil.Block32At[avxR8](v, rh-4, x), eV, iV, hV, m4, mGt4)
	case 6:
		avxHorz6(unsafeutil.Block32At[avxR6](v, rh-3, x), eV, iV, hV, m4, mGt4)
	default:
		avxHorz4(unsafeutil.Block32At[avxR4](v, rh-2, x), eV, iV, hV, m4)
	}
}

// avxSBHorizontal8 is dav1d lpf_v_sb_{y,uv}_8bpc: the horizontal edges of one
// row of units, 32 columns (one vmask byte) per kernel call. The call's tap
// rows (avxCallHalfRows each side of the edge, across the n columns up to its
// last group) and its two level rows (l[0] and l[-b4_stride], four bytes per
// unit) are each checked once, as unsafeutil.Rows views; every group is then
// a column of those views, and the narrower kernels of a wider call take
// their inner rows. A call whose views leave pix, the row or the level map
// takes the group-by-group edge driver. simd counts the groups the vector
// kernel handled.
func avxSBHorizontal8(pix []byte, off, stride, rowBytes int, vm [3]uint32, luma bool, lvl []uint8, lvlOff, lvlStride, comp int, lut *SBLUT) (simd int) {
	all := vm[0] | vm[1] | vm[2]
	if all == 0 {
		return 0
	}
	n := 32 * ((31-bits.LeadingZeros32(all))>>3 + 1)
	if n <= rowBytes {
		switch {
		case vm[2] != 0:
			simd = avxHorzRows8[avxW16](pix, off, stride, n, vm, luma, lvl, lvlOff, lvlStride, comp, lut)
		case vm[1] != 0 && luma:
			simd = avxHorzRows8[avxW8](pix, off, stride, n, vm, luma, lvl, lvlOff, lvlStride, comp, lut)
		case vm[1] != 0:
			simd = avxHorzRows8[avxW6](pix, off, stride, n, vm, luma, lvl, lvlOff, lvlStride, comp, lut)
		default:
			simd = avxHorzRows8[avxW4](pix, off, stride, n, vm, luma, lvl, lvlOff, lvlStride, comp, lut)
		}
		if simd >= 0 {
			return simd
		}
	}
	return avxSBHorizontal8Edge(pix, off, stride, rowBytes, vm, luma, lvl, lvlOff, lvlStride, comp, lut)
}

// avxW16 names the width-16 call shape of the generic group loops.
type avxW16 [16]byte

// avxCallWidth is the widest kernel of a call: its size is the width, so each
// instance's tap-row count is a constant.
type avxCallWidth interface {
	avxW4 | avxW6 | avxW8 | avxW16
}

// avxHorzRows8 is avxSBHorizontal8's group loop for calls whose widest group
// is W: one view of the call's tap rows (avxHalfRows(W) each side of the edge
// at off, n bytes each) and one of its two level rows (l[0] and l[-b4_stride],
// four bytes per unit) replace the per-group reach checks; every group's 32
// columns are a column of those views, and the narrower kernels take inner
// rows. It returns -1, having filtered nothing, when a view leaves its slice.
func avxHorzRows8[W avxCallWidth](pix []byte, off, stride, n int, vm [3]uint32, luma bool, lvl []uint8, lvlOff, lvlStride, comp int, lut *SBLUT) (simd int) {
	var wT W
	rh := avxHalfRows(len(wT))
	li, ls4 := 4*lvlOff, 4*lvlStride
	if !unsafeutil.RowsFit(len(pix), off-rh*stride, stride, n, 2*rh) || !unsafeutil.RowsFit(len(lvl), li-ls4, ls4, n, 2) {
		return -1
	}
	v := unsafeutil.NewRows(pix, off-rh*stride, stride, n, 2*rh)
	lv := unsafeutil.NewRows(lvl, li-ls4, ls4, n, 2)
	gt4All := vm[1] | vm[2]
	all := vm[0] | gt4All
	sel := archsimd.LoadInt8x32Array(&avxConst.levelSel[comp&3])
	sharp0, sharp1 := uint64(lut.Sharp[0]), archsimd.BroadcastUint8x32(lut.Sharp[1])
	for x := 0; x+32 <= n; x += 32 {
		sh := uint(x) / 4
		nib := (all >> sh) & 0xff
		if nib == 0 {
			continue
		}
		nGt4 := (gt4All >> sh) & 0xff
		n16 := (vm[2] >> sh) & 0xff
		lb := unsafeutil.Block32At[avxR2](lv, 0, x)
		eV, iV, hV, m4, ok := avxGroupLimits(archsimd.LoadUint8x32Array(lb.Row32(1)), archsimd.LoadUint8x32Array(lb.Row32(0)), sel, nib, sharp0, sharp1)
		if !ok {
			continue
		}
		simd++
		switch {
		case len(wT) == 16 && n16 != 0:
			avxHorz16(unsafeutil.Block32At[avxR14](v, rh-7, x), eV, iV, hV, m4, avxUnitMask(nGt4), avxUnitMask(n16))
		case len(wT) >= 6 && nGt4 != 0 && luma:
			avxHorz8(unsafeutil.Block32At[avxR8](v, rh-4, x), eV, iV, hV, m4, avxUnitMask(nGt4))
		case len(wT) >= 6 && nGt4 != 0:
			avxHorz6(unsafeutil.Block32At[avxR6](v, rh-3, x), eV, iV, hV, m4, avxUnitMask(nGt4))
		default:
			avxHorz4(unsafeutil.Block32At[avxR4](v, rh-2, x), eV, iV, hV, m4)
		}
	}
	return simd
}

// avxSBHorizontal8Edge is avxSBHorizontal8 group by group, for calls that
// reach the end of pix, the row or the level map: each group checks its own
// tap rows and takes the unit-by-unit reference when they leave pix or the
// row.
func avxSBHorizontal8Edge(pix []byte, off, stride, rowBytes int, vm [3]uint32, luma bool, lvl []uint8, lvlOff, lvlStride, comp int, lut *SBLUT) (simd int) {
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
		u0 := lvlOff + 8*g
		q0Off := off + 32*g
		rh := avxHalfRows(wd)
		if 32*g+32 > rowBytes || !unsafeutil.RowsFit(len(pix), q0Off-rh*stride, stride, 32, 2*rh) {
			filterSBHorizontal8Scalar(pix, q0Off, stride, sbGroupSub(vm, sh), luma, lvl, u0, lvlStride, comp, lut)
			continue
		}
		cur, prev := avxLevelsRow(lvl, u0, lvlStride, nib)
		eV, iV, hV, m4, ok := avxGroupLimits(cur, prev, sel, nib, sharp0, sharp1)
		if !ok {
			continue
		}
		simd++
		avxHorzGroup(unsafeutil.NewRows(pix, q0Off-rh*stride, stride, 32, 2*rh), rh, 0, wd, eV, iV, hV, m4, avxUnitMask(nGt4), n16)
	}
	return simd
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

// avxZipRows pairs rows a and b (their low eight bytes when hi is false, the
// high eight otherwise) and interleaves them bytewise: punpck{l,h}bw a, b.
func avxZipRows(a, b archsimd.Uint8x32, zip archsimd.Int8x32, hi bool) archsimd.Uint8x32 {
	if hi {
		return a.ReshapeToUint64s().InterleaveHiGrouped(b.ReshapeToUint64s()).ReshapeToUint8s().PermuteOrZeroGrouped(zip)
	}
	return a.ReshapeToUint64s().InterleaveLoGrouped(b.ReshapeToUint64s()).ReshapeToUint8s().PermuteOrZeroGrouped(zip)
}

// avxRows2 loads the sixteen bytes of row k of lo into the low lane and those
// of row k of hi (sixteen rows further down) into the high lane.
func avxRows2(lo, hi unsafeutil.Block[avxR16], k int) archsimd.Uint8x32 {
	return archsimd.Uint8x32{}.SetLo(archsimd.LoadUint8x16Array(lo.Row16(k))).SetHi(archsimd.LoadUint8x16Array(hi.Row16(k)))
}

// avxVertical8 filters a vertical edge across 32 rows for widths up to 8
// (dav1d FILTER {4,6,8}, h): lo and hi are rows 0..15 and 16..31, each
// sixteen bytes from p3; the eight p3..q3 bytes of every row are written back.
func avxVertical8[W avxWidth](lo, hi unsafeutil.Block[avxR16], eV, iV, hV, m4, mGt4 archsimd.Uint8x32) {
	zip := archsimd.LoadInt8x32Array(&avxConst.zip)
	b0 := avxZipRows(avxRows2(lo, hi, 0), avxRows2(lo, hi, 1), zip, false)
	b1 := avxZipRows(avxRows2(lo, hi, 2), avxRows2(lo, hi, 3), zip, false)
	b2 := avxZipRows(avxRows2(lo, hi, 4), avxRows2(lo, hi, 5), zip, false)
	b3 := avxZipRows(avxRows2(lo, hi, 6), avxRows2(lo, hi, 7), zip, false)
	b4 := avxZipRows(avxRows2(lo, hi, 8), avxRows2(lo, hi, 9), zip, false)
	b5 := avxZipRows(avxRows2(lo, hi, 10), avxRows2(lo, hi, 11), zip, false)
	b6 := avxZipRows(avxRows2(lo, hi, 12), avxRows2(lo, hi, 13), zip, false)
	b7 := avxZipRows(avxRows2(lo, hi, 14), avxRows2(lo, hi, 15), zip, false)
	t0, t1, t2, t3, t4, t5, t6, t7 := avxTransposeWords(b0, b1, b2, b3, b4, b5, b6, b7)
	o2, o1, o0, n0, n1, n2, _, ok := avxLpfCore[W](t0, t1, t2, t3, t4, t5, t6, t7, eV, iV, hV, m4, mGt4)
	if !ok {
		return
	}
	b0, b1, b2, b3, b4, b5, b6, b7 = avxTransposeWords(t0, o2, o1, o0, n0, n1, n2, t7)
	unzip := archsimd.LoadInt8x32Array(&avxConst.unzip)
	avxStoreRowPair(lo, hi, 0, b0.PermuteOrZeroGrouped(unzip))
	avxStoreRowPair(lo, hi, 2, b1.PermuteOrZeroGrouped(unzip))
	avxStoreRowPair(lo, hi, 4, b2.PermuteOrZeroGrouped(unzip))
	avxStoreRowPair(lo, hi, 6, b3.PermuteOrZeroGrouped(unzip))
	avxStoreRowPair(lo, hi, 8, b4.PermuteOrZeroGrouped(unzip))
	avxStoreRowPair(lo, hi, 10, b5.PermuteOrZeroGrouped(unzip))
	avxStoreRowPair(lo, hi, 12, b6.PermuteOrZeroGrouped(unzip))
	avxStoreRowPair(lo, hi, 14, b7.PermuteOrZeroGrouped(unzip))
}

// avxStoreRowPair writes a de-interleaved row pair: the low eight bytes of
// each lane are row k (lane 0, rows of lo) / 16+k (lane 1, rows of hi), the
// high eight rows k+1 / 17+k, each to the first eight bytes of its row.
func avxStoreRowPair(lo, hi unsafeutil.Block[avxR16], k int, a archsimd.Uint8x32) {
	l, h := a.GetLo().ReshapeToUint64s(), a.GetHi().ReshapeToUint64s()
	binary.LittleEndian.PutUint64(lo.Row16(k)[:8], l.GetElem(0))
	binary.LittleEndian.PutUint64(lo.Row16(k + 1)[:8], l.GetElem(1))
	binary.LittleEndian.PutUint64(hi.Row16(k)[:8], h.GetElem(0))
	binary.LittleEndian.PutUint64(hi.Row16(k + 1)[:8], h.GetElem(1))
}

// avxVertical16 is the wd16 variant: sixteen bytes per row (p7..q7 of rows
// 0..15 in lo and 16..31 in hi), a full 16x16 byte transpose per lane each
// way (dav1d TRANSPOSE_16X16B).
func avxVertical16(lo, hi unsafeutil.Block[avxR16], eV, iV, hV, m4, mGt4, m16 archsimd.Uint8x32) {
	zip := archsimd.LoadInt8x32Array(&avxConst.zip)
	var l, h [8]archsimd.Uint8x32
	for j := range l {
		a, b := avxRows2(lo, hi, 2*j), avxRows2(lo, hi, 2*j+1)
		l[j] = avxZipRows(a, b, zip, false)
		h[j] = avxZipRows(a, b, zip, true)
	}
	var t [16]archsimd.Uint8x32
	t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7] = avxTransposeWords(l[0], l[1], l[2], l[3], l[4], l[5], l[6], l[7])
	t[8], t[9], t[10], t[11], t[12], t[13], t[14], t[15] = avxTransposeWords(h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7])
	if !avxLpf16(&t, eV, iV, hV, m4, mGt4, m16) {
		return
	}
	l[0], l[1], l[2], l[3], l[4], l[5], l[6], l[7] = avxTransposeWords(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7])
	h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7] = avxTransposeWords(t[8], t[9], t[10], t[11], t[12], t[13], t[14], t[15])
	unzip := archsimd.LoadInt8x32Array(&avxConst.unzip)
	for j := range l {
		a := l[j].PermuteOrZeroGrouped(unzip).ReshapeToUint64s()
		b := h[j].PermuteOrZeroGrouped(unzip).ReshapeToUint64s()
		r0 := a.InterleaveLoGrouped(b).ReshapeToUint8s()
		r1 := a.InterleaveHiGrouped(b).ReshapeToUint8s()
		r0.GetLo().StoreArray(lo.Row16(2 * j))
		r0.GetHi().StoreArray(hi.Row16(2 * j))
		r1.GetLo().StoreArray(lo.Row16(2*j + 1))
		r1.GetHi().StoreArray(hi.Row16(2*j + 1))
	}
}

// avxVertGroup runs the width-wd kernel on the 32 rows of v from row r,
// sixteen bytes from column c (p7 at wd 16, else p3).
func avxVertGroup(v unsafeutil.Rows, r, c, wd int, eV, iV, hV, m4, mGt4 archsimd.Uint8x32, n16 uint32) {
	lo, hi := unsafeutil.BlockAt[avxR16](v, r, c), unsafeutil.BlockAt[avxR16](v, r+16, c)
	switch wd {
	case 16:
		avxVertical16(lo, hi, eV, iV, hV, m4, mGt4, avxUnitMask(n16))
	case 8:
		avxVertical8[avxW8](lo, hi, eV, iV, hV, m4, mGt4)
	case 6:
		avxVertical8[avxW6](lo, hi, eV, iV, hV, m4, mGt4)
	default:
		avxVertical8[avxW4](lo, hi, eV, iV, hV, m4, mGt4)
	}
}

// avxSBVertical8 is dav1d lpf_h_sb_{y,uv}_8bpc: the vertical edges of one
// column of units, 32 rows per kernel call. One view of the call's rows (its
// 32*(hiG+1) rows, reach bytes left of the edge to twelve right of it) and one
// of its (l[-1], l[0]) level cell pairs replace the per-group reach checks;
// each group is then two sixteen-row blocks of the sample view and an
// eight-row block of the level view. Calls whose views do not fit take the
// group-by-group edge driver.
func avxSBVertical8(pix []byte, off, stride, rowBytes int, vm [3]uint32, luma bool, lvl []uint8, lvlOff, lvlStride, comp int, lut *SBLUT) (simd int) {
	gt4All := vm[1] | vm[2]
	all := vm[0] | gt4All
	if all == 0 {
		return 0
	}
	hiG := (31 - bits.LeadingZeros32(all)) >> 3
	reach := 4
	if vm[2] != 0 {
		reach = 8
	}
	if rowBytes < 12 || !unsafeutil.RowsFit(len(pix), off-reach, stride, reach+12, 32*hiG+32) || !unsafeutil.RowsFit(len(lvl), 4*lvlOff-4, 4*lvlStride, 8, 8*hiG+8) {
		return avxSBVertical8Edge(pix, off, stride, rowBytes, vm, luma, lvl, lvlOff, lvlStride, comp, lut)
	}
	v := unsafeutil.NewRows(pix, off-reach, stride, reach+12, 32*hiG+32)
	lv := unsafeutil.NewRows(lvl, 4*lvlOff-4, 4*lvlStride, 8, 8*hiG+8)
	c16, c8 := reach-8, reach-4 // p7 and p3 columns of the view
	sel := archsimd.LoadInt8x32Array(&avxConst.levelSel[comp&3])
	sharp0, sharp1 := uint64(lut.Sharp[0]), archsimd.BroadcastUint8x32(lut.Sharp[1])
	for g := 0; g <= hiG; g++ {
		sh := 8 * uint(g)
		nib := (all >> sh) & 0xff
		if nib == 0 {
			continue
		}
		nGt4 := (gt4All >> sh) & 0xff
		n16 := (vm[2] >> sh) & 0xff
		cur, prev := avxLevelPairs(lv, 8*g)
		eV, iV, hV, m4, ok := avxGroupLimits(cur, prev, sel, nib, sharp0, sharp1)
		if !ok {
			continue
		}
		simd++
		wd := avxSBWidth(nGt4, n16, luma)
		c := c8
		if wd == 16 {
			c = c16
		}
		avxVertGroup(v, 32*g, c, wd, eV, iV, hV, m4, avxUnitMask(nGt4), n16)
	}
	return simd
}

// avxLevelPairs splits the eight (l[-1], l[0]) cell pairs of rows r..r+7 of
// lv (one per row) into the l[0] cells (cur) and the l[-1] cells (prev).
func avxLevelPairs(lv unsafeutil.Rows, r int) (cur, prev archsimd.Uint8x32) {
	lg := lv.Sub(r, 0, 8, 8)
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
	x = le.Uint64(lg.Row8(4, 0)[:])
	c[4], p[4] = uint32(x>>32), uint32(x)
	x = le.Uint64(lg.Row8(5, 0)[:])
	c[5], p[5] = uint32(x>>32), uint32(x)
	x = le.Uint64(lg.Row8(6, 0)[:])
	c[6], p[6] = uint32(x>>32), uint32(x)
	x = le.Uint64(lg.Row8(7, 0)[:])
	c[7], p[7] = uint32(x>>32), uint32(x)
	return archsimd.LoadUint32x8Array(&c).ReshapeToUint8s(), archsimd.LoadUint32x8Array(&p).ReshapeToUint8s()
}

// avxSBVertical8Edge is avxSBVertical8 group by group, for calls that reach
// the end of pix or the level map: groups whose rows do not fit (sixteen
// bytes from p3, or from p7 at width 16) take the unit-by-unit reference.
func avxSBVertical8Edge(pix []byte, off, stride, rowBytes int, vm [3]uint32, luma bool, lvl []uint8, lvlOff, lvlStride, comp int, lut *SBLUT) (simd int) {
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
		reach := 4
		if wd == 16 {
			reach = 8
		}
		if (wd != 16 && rowBytes < 12) || !unsafeutil.RowsFit(len(pix), q0Off-reach, stride, 16, 32) {
			filterSBVertical8Scalar(pix, q0Off, stride, sbGroupSub(vm, sh), luma, lvl, u0, lvlStride, comp, lut)
			continue
		}
		cur, prev := avxLevelsCol(lvl, u0, lvlStride, nib)
		eV, iV, hV, m4, ok := avxGroupLimits(cur, prev, sel, nib, sharp0, sharp1)
		if !ok {
			continue
		}
		simd++
		avxVertGroup(unsafeutil.NewRows(pix, q0Off-reach, stride, 16, 32), 0, 0, wd, eV, iV, hV, m4, avxUnitMask(nGt4), n16)
	}
	return simd
}
