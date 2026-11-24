// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package loopfilter

import (
	"encoding/binary"
	"simd/archsimd"
)

// The 10/12-bit vertical-edge kernels, dav1d's lpf_h_{4,6,8,16}_8_neon
// (src/arm/64/loopfilter16.S): eight rows are loaded, transposed into one
// vector per tap in registers, filtered by the lpf_8_wd* body written out in
// the same function (a call here would spill every live vector: Go has no
// callee-saved vector registers), and only the columns the filter can have
// changed are transposed back and stored, through dav1d's exits: nothing
// when no lane passed the filter mask, p1..q1 (a 4x8 transpose and one
// 8-byte store per row) when only the wd4 filter ran, p3..q3 (8x8, sixteen
// bytes) when the flat8in filter ran but flat8out did not.
//
// Every row is one bounds-checked window of the plane, taken once and reused
// for the store; the drivers have already range-checked the call, so the
// checks never fail.

// lpf16Consts holds the kernels' vector constants, loaded from memory (one
// LDR each) where an archsimd broadcast or constant shift costs a MOVD, an
// INS and a DUP every use: the bitdepth-scaled iclip_diff/iclip_pixel bounds
// and flatness threshold, the wd4 multipliers, the shift counts (USHL/SSHL by
// a negative count is a right shift) and the flat filters' rounding biases.
type lpf16Consts struct {
	dmin, dmax, maxPix [8]int16
	flat               [8]uint16
	three, four, one   [8]int16
	shr1, shr3, shr4   [8]int16
	rnd4, rnd8         [8]uint16
}

// lpf16Tab is indexed by bitdepth_min_8 / 2 (8, 10 and 12 bits).
var lpf16Tab = func() (t [3]lpf16Consts) {
	for i := range t {
		bd8 := 2 * i
		for j := 0; j < 8; j++ {
			c := &t[i]
			c.dmin[j] = int16(-128 << bd8)
			c.dmax[j] = int16(128<<bd8 - 1)
			c.maxPix[j] = int16(256<<bd8 - 1)
			c.flat[j] = uint16(1) << bd8
			c.three[j], c.four[j], c.one[j] = 3, 4, 1
			c.shr1[j], c.shr3[j], c.shr4[j] = -1, -3, -4
			c.rnd4[j], c.rnd8[j] = 4, 8
		}
	}
	return
}()

// lpf16Abd is |a - b| on 10/12-bit samples (dav1d uabd): the difference of
// two samples below 2^15 is exact in signed 16-bit lanes, so a sub and an abs
// replace the two saturating subtracts and the orr of the unsigned form.
func lpf16Abd(a, b archsimd.Uint16x8) archsimd.Uint16x8 {
	return a.BitsToInt16().Sub(b.BitsToInt16()).Abs().ToBits()
}

// lpf16Transpose4x8 is dav1d transpose_4x8h: the four tap vectors a..d of
// eight rows become four vectors whose 64-bit halves are rows (0, 4), (1, 5),
// (2, 6) and (3, 7), four samples a..d each.
func lpf16Transpose4x8(a, b, c, d archsimd.Uint16x8) (r04, r15, r26, r37 archsimd.Uint64x2) {
	ab0, ab1 := a.InterleaveEven(b).ReshapeToUint32s(), a.InterleaveOdd(b).ReshapeToUint32s()
	cd0, cd1 := c.InterleaveEven(d).ReshapeToUint32s(), c.InterleaveOdd(d).ReshapeToUint32s()
	r04 = ab0.InterleaveEven(cd0).ReshapeToUint64s()
	r26 = ab0.InterleaveOdd(cd0).ReshapeToUint64s()
	r15 = ab1.InterleaveEven(cd1).ReshapeToUint64s()
	r37 = ab1.InterleaveOdd(cd1).ReshapeToUint64s()
	return
}

// lpfCols16W is dav1d lpf_h_{4,6,8}_8_neon for one two-unit group: o is the
// byte offset of q0 in the first of eight rows, p3..q3 sit at o-8..o+8.
func lpfCols16W[W lpfWidth](pix []byte, o, stride int, k *lpf16Consts, eV, iV, hV archsimd.Uint16x8, m4, mGt4 archsimd.Mask16x8) {
	var wdT W
	wd := len(wdT)
	b := o - 8
	w0 := (*[16]byte)(pix[b : b+16])
	b += stride
	w1 := (*[16]byte)(pix[b : b+16])
	b += stride
	w2 := (*[16]byte)(pix[b : b+16])
	b += stride
	w3 := (*[16]byte)(pix[b : b+16])
	b += stride
	w4 := (*[16]byte)(pix[b : b+16])
	b += stride
	w5 := (*[16]byte)(pix[b : b+16])
	b += stride
	w6 := (*[16]byte)(pix[b : b+16])
	b += stride
	w7 := (*[16]byte)(pix[b : b+16])
	p3, p2, p1, p0, q0, q1, q2, q3 := lpfTranspose8x8U16Regs(archsimd.LoadUint8x16Array(w0).ReshapeToUint16s(), archsimd.LoadUint8x16Array(w1).ReshapeToUint16s(), archsimd.LoadUint8x16Array(w2).ReshapeToUint16s(), archsimd.LoadUint8x16Array(w3).ReshapeToUint16s(),
		archsimd.LoadUint8x16Array(w4).ReshapeToUint16s(), archsimd.LoadUint8x16Array(w5).ReshapeToUint16s(), archsimd.LoadUint8x16Array(w6).ReshapeToUint16s(), archsimd.LoadUint8x16Array(w7).ReshapeToUint16s())

	maxd := lpf16Abd(p1, p0).Max(lpf16Abd(q1, q0))
	t := maxd
	if wd >= 6 {
		tt := lpf16Abd(p2, p1).Max(lpf16Abd(q2, q1))
		if wd >= 8 {
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
	mF4 := fm
	var mFlat archsimd.Mask16x8
	anyFlat := false
	if wd >= 6 {
		fl := lpf16Abd(p2, p0).Max(maxd).Max(lpf16Abd(q2, q0))
		if wd >= 8 {
			fl = fl.Max(lpf16Abd(p3, p0)).Max(lpf16Abd(q3, q0))
		}
		mFlat = archsimd.LoadUint16x8Array(&k.flat).GreaterEqual(fl).And(mGt4).And(fm)
		mF4 = fm.And(mFlat.Not())
		anyFlat = lpf16Any(mFlat)
	}

	o2, o1, o0, n0, n1, n2 := p2, p1, p0, q0, q1, q2
	if wd == 4 || lpf16Any(mF4) {
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
	if wd == 6 && anyFlat {
		s := p2.Add(p2).Add(p2).Add(p1.Add(p0).Add(p1.Add(p0))).Add(q0).Add(archsimd.LoadUint16x8Array(&k.rnd4))
		o1 = s.Shift(archsimd.LoadInt16x8Array(&k.shr3)).IfElse(mFlat, o1)
		s = s.Add(q0.Add(q1)).Sub(p2.Add(p2))
		o0 = s.Shift(archsimd.LoadInt16x8Array(&k.shr3)).IfElse(mFlat, o0)
		s = s.Add(q1.Add(q2)).Sub(p2.Add(p1))
		n0 = s.Shift(archsimd.LoadInt16x8Array(&k.shr3)).IfElse(mFlat, n0)
		s = s.Add(q2.Add(q2)).Sub(p1.Add(p0))
		n1 = s.Shift(archsimd.LoadInt16x8Array(&k.shr3)).IfElse(mFlat, n1)
	}
	if wd == 8 && anyFlat {
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

		r0, r1, r2, r3, r4, r5, r6, r7 := lpfTranspose8x8U16Regs(p3, o2, o1, o0, n0, n1, n2, q3)
		r0.ReshapeToUint8s().StoreArray(w0)
		r1.ReshapeToUint8s().StoreArray(w1)
		r2.ReshapeToUint8s().StoreArray(w2)
		r3.ReshapeToUint8s().StoreArray(w3)
		r4.ReshapeToUint8s().StoreArray(w4)
		r5.ReshapeToUint8s().StoreArray(w5)
		r6.ReshapeToUint8s().StoreArray(w6)
		r7.ReshapeToUint8s().StoreArray(w7)
		return
	}
	r04, r15, r26, r37 := lpf16Transpose4x8(o1, o0, n0, n1)
	binary.LittleEndian.PutUint64(w0[4:12], r04.GetElem(0))
	binary.LittleEndian.PutUint64(w1[4:12], r15.GetElem(0))
	binary.LittleEndian.PutUint64(w2[4:12], r26.GetElem(0))
	binary.LittleEndian.PutUint64(w3[4:12], r37.GetElem(0))
	binary.LittleEndian.PutUint64(w4[4:12], r04.GetElem(1))
	binary.LittleEndian.PutUint64(w5[4:12], r15.GetElem(1))
	binary.LittleEndian.PutUint64(w6[4:12], r26.GetElem(1))
	binary.LittleEndian.PutUint64(w7[4:12], r37.GetElem(1))
}

// lpfCols16Wd16 is dav1d lpf_h_16_8_neon for one two-unit group: o is the
// byte offset of q0 in the first of eight rows, p7..q7 sit at o-16..o+16.
func lpfCols16Wd16(pix []byte, o, stride int, k *lpf16Consts, eV, iV, hV archsimd.Uint16x8, m4, mGt4, m16 archsimd.Mask16x8) {
	b := o - 16
	w0 := (*[32]byte)(pix[b : b+32])
	b += stride
	w1 := (*[32]byte)(pix[b : b+32])
	b += stride
	w2 := (*[32]byte)(pix[b : b+32])
	b += stride
	w3 := (*[32]byte)(pix[b : b+32])
	b += stride
	w4 := (*[32]byte)(pix[b : b+32])
	b += stride
	w5 := (*[32]byte)(pix[b : b+32])
	b += stride
	w6 := (*[32]byte)(pix[b : b+32])
	b += stride
	w7 := (*[32]byte)(pix[b : b+32])
	p7, p6, p5, p4, p3, p2, p1, p0 := lpfTranspose8x8U16Regs(
		archsimd.LoadUint8x16Array((*[16]byte)(w0[:16])).ReshapeToUint16s(), archsimd.LoadUint8x16Array((*[16]byte)(w1[:16])).ReshapeToUint16s(), archsimd.LoadUint8x16Array((*[16]byte)(w2[:16])).ReshapeToUint16s(), archsimd.LoadUint8x16Array((*[16]byte)(w3[:16])).ReshapeToUint16s(),
		archsimd.LoadUint8x16Array((*[16]byte)(w4[:16])).ReshapeToUint16s(), archsimd.LoadUint8x16Array((*[16]byte)(w5[:16])).ReshapeToUint16s(), archsimd.LoadUint8x16Array((*[16]byte)(w6[:16])).ReshapeToUint16s(), archsimd.LoadUint8x16Array((*[16]byte)(w7[:16])).ReshapeToUint16s())
	q0, q1, q2, q3, q4, q5, q6, q7 := lpfTranspose8x8U16Regs(
		archsimd.LoadUint8x16Array((*[16]byte)(w0[16:])).ReshapeToUint16s(), archsimd.LoadUint8x16Array((*[16]byte)(w1[16:])).ReshapeToUint16s(), archsimd.LoadUint8x16Array((*[16]byte)(w2[16:])).ReshapeToUint16s(), archsimd.LoadUint8x16Array((*[16]byte)(w3[16:])).ReshapeToUint16s(),
		archsimd.LoadUint8x16Array((*[16]byte)(w4[16:])).ReshapeToUint16s(), archsimd.LoadUint8x16Array((*[16]byte)(w5[16:])).ReshapeToUint16s(), archsimd.LoadUint8x16Array((*[16]byte)(w6[16:])).ReshapeToUint16s(), archsimd.LoadUint8x16Array((*[16]byte)(w7[16:])).ReshapeToUint16s())

	maxd := lpf16Abd(p1, p0).Max(lpf16Abd(q1, q0))
	tt := lpf16Abd(p2, p1).Max(lpf16Abd(q2, q1)).Max(lpf16Abd(p3, p2).Max(lpf16Abd(q3, q2)))
	t := maxd.Max(tt.Masked(mGt4))
	d00 := lpf16Abd(p0, q0)
	sum := d00.Add(d00).Add(lpf16Abd(p1, q1).Shift(archsimd.LoadInt16x8Array(&k.shr1)))
	fm := iV.GreaterEqual(t).And(eV.GreaterEqual(sum)).And(m4)
	if !lpf16Any(fm) {
		return
	}
	fV := archsimd.LoadUint16x8Array(&k.flat)
	fl := lpf16Abd(p2, p0).Max(maxd).Max(lpf16Abd(q2, q0)).Max(lpf16Abd(p3, p0)).Max(lpf16Abd(q3, q0))
	mFlat := fV.GreaterEqual(fl).And(mGt4).And(fm)
	mF4 := fm.And(mFlat.Not())
	fo := lpf16Abd(p6, p0).Max(lpf16Abd(p5, p0)).Max(lpf16Abd(p4, p0)).
		Max(lpf16Abd(q4, q0)).Max(lpf16Abd(q5, q0)).Max(lpf16Abd(q6, q0))
	mFlat16 := fV.GreaterEqual(fo).And(m16).And(mFlat)
	mFlat = mFlat.And(mFlat16.Not())
	anyFlat, anyFlat16 := lpf16Any(mFlat), lpf16Any(mFlat16)

	o2, o1, o0, n0, n1, n2 := p2, p1, p0, q0, q1, q2
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
	if anyFlat {
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
	if anyFlat16 {
		m := mFlat16
		p6x2 := p6.Add(p6)
		s := p6.Add(p6x2).Add(p6x2.Add(p6x2))
		s = s.Add(p5.Add(p5)).Add(p4.Add(p4)).Add(p3.Add(p2)).Add(p1.Add(p0)).Add(q0).Add(archsimd.LoadUint16x8Array(&k.rnd8))
		o5 := s.Shift(archsimd.LoadInt16x8Array(&k.shr4)).IfElse(m, p5)
		s = s.Add(p3.Add(q1)).Sub(p6x2)
		o4 := s.Shift(archsimd.LoadInt16x8Array(&k.shr4)).IfElse(m, p4)
		s = s.Add(p2.Add(q2)).Sub(p6.Add(p5))
		o3 := s.Shift(archsimd.LoadInt16x8Array(&k.shr4)).IfElse(m, p3)
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
		n3 := s.Shift(archsimd.LoadInt16x8Array(&k.shr4)).IfElse(m, q3)
		s = s.Add(q5.Add(q6)).Sub(p2.Add(q2))
		n4 := s.Shift(archsimd.LoadInt16x8Array(&k.shr4)).IfElse(m, q4)
		s = s.Add(q6.Add(q6)).Sub(p1.Add(q3))
		n5 := s.Shift(archsimd.LoadInt16x8Array(&k.shr4)).IfElse(m, q5)

		a0, a1, a2, a3, a4, a5, a6, a7 := lpfTranspose8x8U16Regs(p7, p6, o5, o4, o3, o2, o1, o0)
		a0.ReshapeToUint8s().StoreArray((*[16]byte)(w0[:16]))
		a1.ReshapeToUint8s().StoreArray((*[16]byte)(w1[:16]))
		a2.ReshapeToUint8s().StoreArray((*[16]byte)(w2[:16]))
		a3.ReshapeToUint8s().StoreArray((*[16]byte)(w3[:16]))
		a4.ReshapeToUint8s().StoreArray((*[16]byte)(w4[:16]))
		a5.ReshapeToUint8s().StoreArray((*[16]byte)(w5[:16]))
		a6.ReshapeToUint8s().StoreArray((*[16]byte)(w6[:16]))
		a7.ReshapeToUint8s().StoreArray((*[16]byte)(w7[:16]))
		c0, c1, c2, c3, c4, c5, c6, c7 := lpfTranspose8x8U16Regs(n0, n1, n2, n3, n4, n5, q6, q7)
		c0.ReshapeToUint8s().StoreArray((*[16]byte)(w0[16:]))
		c1.ReshapeToUint8s().StoreArray((*[16]byte)(w1[16:]))
		c2.ReshapeToUint8s().StoreArray((*[16]byte)(w2[16:]))
		c3.ReshapeToUint8s().StoreArray((*[16]byte)(w3[16:]))
		c4.ReshapeToUint8s().StoreArray((*[16]byte)(w4[16:]))
		c5.ReshapeToUint8s().StoreArray((*[16]byte)(w5[16:]))
		c6.ReshapeToUint8s().StoreArray((*[16]byte)(w6[16:]))
		c7.ReshapeToUint8s().StoreArray((*[16]byte)(w7[16:]))
		return
	}
	if anyFlat {
		r0, r1, r2, r3, r4, r5, r6, r7 := lpfTranspose8x8U16Regs(p3, o2, o1, o0, n0, n1, n2, q3)
		r0.ReshapeToUint8s().StoreArray((*[16]byte)(w0[8:24]))
		r1.ReshapeToUint8s().StoreArray((*[16]byte)(w1[8:24]))
		r2.ReshapeToUint8s().StoreArray((*[16]byte)(w2[8:24]))
		r3.ReshapeToUint8s().StoreArray((*[16]byte)(w3[8:24]))
		r4.ReshapeToUint8s().StoreArray((*[16]byte)(w4[8:24]))
		r5.ReshapeToUint8s().StoreArray((*[16]byte)(w5[8:24]))
		r6.ReshapeToUint8s().StoreArray((*[16]byte)(w6[8:24]))
		r7.ReshapeToUint8s().StoreArray((*[16]byte)(w7[8:24]))
		return
	}
	r04, r15, r26, r37 := lpf16Transpose4x8(o1, o0, n0, n1)
	binary.LittleEndian.PutUint64(w0[12:20], r04.GetElem(0))
	binary.LittleEndian.PutUint64(w1[12:20], r15.GetElem(0))
	binary.LittleEndian.PutUint64(w2[12:20], r26.GetElem(0))
	binary.LittleEndian.PutUint64(w3[12:20], r37.GetElem(0))
	binary.LittleEndian.PutUint64(w4[12:20], r04.GetElem(1))
	binary.LittleEndian.PutUint64(w5[12:20], r15.GetElem(1))
	binary.LittleEndian.PutUint64(w6[12:20], r26.GetElem(1))
	binary.LittleEndian.PutUint64(w7[12:20], r37.GetElem(1))
}

// lpfCols16 runs the width-specialized vertical-edge kernel.
func lpfCols16(pix []byte, o, stride, wd int, k *lpf16Consts, eV, iV, hV archsimd.Uint16x8, m4, mGt4, m16 archsimd.Mask16x8) {
	switch wd {
	case 16:
		lpfCols16Wd16(pix, o, stride, k, eV, iV, hV, m4, mGt4, m16)
	case 8:
		lpfCols16W[lpfW8](pix, o, stride, k, eV, iV, hV, m4, mGt4)
	case 6:
		lpfCols16W[lpfW6](pix, o, stride, k, eV, iV, hV, m4, mGt4)
	default:
		lpfCols16W[lpfW4](pix, o, stride, k, eV, iV, hV, m4, mGt4)
	}
}
