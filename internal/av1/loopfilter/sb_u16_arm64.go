// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package loopfilter

import (
	"simd/archsimd"
	"unsafe"
)

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

// sbHalf16 holds one two-unit half group's limits (already scaled by the bit
// depth, each unit's value replicated over its four lanes) and unit masks.
type sbHalf16 struct {
	e, i, h      [2]uint64
	m4, gt4, w16 uint32
}

// sbRep16 replicates a 16-bit value over the four lanes of a uint64.
const sbRep16 = 0x0001000100010001

func (s *sbHalf16) set(k int, L uint8, lut *SBLUT, bd8 uint) {
	s.e[k] = uint64(lut.E[L]) << bd8 * sbRep16
	s.i[k] = uint64(lut.I[L]) << bd8 * sbRep16
	s.h[k] = uint64(L>>4) << bd8 * sbRep16
}

func (s *sbHalf16) vectors() (eV, iV, hV archsimd.Uint16x8, m4, mGt4, m16 archsimd.Mask16x8) {
	eV = archsimd.LoadUint64x2Array(&s.e).ReshapeToUint16s()
	iV = archsimd.LoadUint64x2Array(&s.i).ReshapeToUint16s()
	hV = archsimd.LoadUint64x2Array(&s.h).ReshapeToUint16s()
	m4 = archsimd.LoadUint16x8Array(&sbPairLanes16[s.m4]).BitsToInt16().ToMask()
	mGt4 = archsimd.LoadUint16x8Array(&sbPairLanes16[s.gt4]).BitsToInt16().ToMask()
	m16 = archsimd.LoadUint16x8Array(&sbPairLanes16[s.w16]).BitsToInt16().ToMask()
	return
}

func (s *sbHalf16) width(luma bool) int {
	switch {
	case s.w16 != 0:
		return 16
	case s.gt4 != 0:
		if luma {
			return 8
		}
		return 6
	}
	return 4
}

func lpf16Load(p unsafe.Pointer, off int) archsimd.Uint16x8 {
	return archsimd.LoadUint16x8Array((*[8]uint16)(unsafe.Add(p, off)))
}

func lpf16Store(p unsafe.Pointer, off int, v archsimd.Uint16x8) {
	v.StoreArray((*[8]uint16)(unsafe.Add(p, off)))
}

func lpf16AbsDiff(a, b archsimd.Uint16x8) archsimd.Uint16x8 {
	return a.SubSaturated(b).Or(b.SubSaturated(a))
}

func lpf16Any(m archsimd.Mask16x8) bool {
	b := m.ToInt16x8().ToBits().ReshapeToUint64s()
	return b.GetElem(0)|b.GetElem(1) != 0
}

// lpf8U16 is the high-bit-depth counterpart of lpf16U8 on eight uint16 lanes
// (two 4-sample units): dav1d's loop_filter() with E/I/H and the flatness
// threshold scaled by bd8 = bitDepth-8, filter4 in signed 16-bit lanes with
// the iclip_diff/iclip_pixel bounds, and the flat filters' sums held exactly
// in uint16 lanes (at most 16*4095+8). Taps are rows around q0 at p, stride
// in bytes.
//
//go:nocheckptr
func lpf8U16(p unsafe.Pointer, stride int, wd int, bd8 uint, eV, iV, hV archsimd.Uint16x8, m4, mGt4, m16 archsimd.Mask16x8) {
	p1 := lpf16Load(p, -2*stride)
	p0 := lpf16Load(p, -stride)
	q0 := lpf16Load(p, 0)
	q1 := lpf16Load(p, stride)
	d10 := lpf16AbsDiff(p1, p0)
	dq := lpf16AbsDiff(q1, q0)
	maxd := d10.Max(dq)
	t := maxd
	var p2, q2, p3, q3 archsimd.Uint16x8
	if wd >= 6 {
		p2 = lpf16Load(p, -3*stride)
		q2 = lpf16Load(p, 2*stride)
		tt := lpf16AbsDiff(p2, p1).Max(lpf16AbsDiff(q2, q1))
		if wd >= 8 {
			p3 = lpf16Load(p, -4*stride)
			q3 = lpf16Load(p, 3*stride)
			tt = tt.Max(lpf16AbsDiff(p3, p2).Max(lpf16AbsDiff(q3, q2)))
		}
		t = t.Max(tt.Masked(mGt4))
	}
	d00 := lpf16AbsDiff(p0, q0)
	sum := d00.Add(d00).Add(lpf16AbsDiff(p1, q1).ShiftAllRight(1))
	fm := iV.GreaterEqual(t).And(eV.GreaterEqual(sum)).And(m4)
	if !lpf16Any(fm) {
		return
	}
	fV := archsimd.BroadcastUint16x8(uint16(1) << bd8)
	mF4 := fm
	var mFlat, mFlat16 archsimd.Mask16x8
	var p6, p5, p4, q4, q5, q6 archsimd.Uint16x8
	if wd >= 6 {
		fl := lpf16AbsDiff(p2, p0).Max(maxd).Max(lpf16AbsDiff(q2, q0))
		if wd >= 8 {
			fl = fl.Max(lpf16AbsDiff(p3, p0)).Max(lpf16AbsDiff(q3, q0))
		}
		mFlat = fV.GreaterEqual(fl).And(mGt4).And(fm)
		mF4 = fm.And(mFlat.Not())
		if wd == 16 {
			p6 = lpf16Load(p, -7*stride)
			p5 = lpf16Load(p, -6*stride)
			p4 = lpf16Load(p, -5*stride)
			q4 = lpf16Load(p, 4*stride)
			q5 = lpf16Load(p, 5*stride)
			q6 = lpf16Load(p, 6*stride)
			fo := lpf16AbsDiff(p6, p0).Max(lpf16AbsDiff(p5, p0)).Max(lpf16AbsDiff(p4, p0)).
				Max(lpf16AbsDiff(q4, q0)).Max(lpf16AbsDiff(q5, q0)).Max(lpf16AbsDiff(q6, q0))
			mFlat16 = fV.GreaterEqual(fo).And(m16).And(mFlat)
			mFlat = mFlat.And(mFlat16.Not())
		}
	}

	o1, o0, n0, n1 := p1, p0, q0, q1
	if lpf16Any(mF4) {
		hev := maxd.Greater(hV)
		dmin := archsimd.BroadcastInt16x8(int16(-128 << bd8))
		dmax := archsimd.BroadcastInt16x8(int16(128<<bd8 - 1))
		zero := archsimd.Int16x8{}
		maxPix := archsimd.BroadcastInt16x8(int16(256<<bd8 - 1))
		p1s, p0s, q0s, q1s := p1.BitsToInt16(), p0.BitsToInt16(), q0.BitsToInt16(), q1.BitsToInt16()
		f := p1s.Sub(q1s).Max(dmin).Min(dmax).Masked(hev)
		f = q0s.Sub(p0s).Mul(archsimd.BroadcastInt16x8(3)).Add(f).Max(dmin).Min(dmax)
		f1 := f.Add(archsimd.BroadcastInt16x8(4)).Min(dmax).ShiftAllRight(3)
		f2 := f.Add(archsimd.BroadcastInt16x8(3)).Min(dmax).ShiftAllRight(3)
		o0 = p0s.Add(f2).Max(zero).Min(maxPix).ToBits().IfElse(mF4, p0)
		n0 = q0s.Sub(f1).Max(zero).Min(maxPix).ToBits().IfElse(mF4, q0)
		f3 := f1.Add(archsimd.BroadcastInt16x8(1)).ShiftAllRight(1)
		mOuter := mF4.And(hev.Not())
		o1 = p1s.Add(f3).Max(zero).Min(maxPix).ToBits().IfElse(mOuter, p1)
		n1 = q1s.Sub(f3).Max(zero).Min(maxPix).ToBits().IfElse(mOuter, q1)
	}

	if wd == 6 {
		if lpf16Any(mFlat) {
			s := p2.Add(p2).Add(p2).Add(p1.Add(p0).Add(p1.Add(p0))).Add(q0).Add(archsimd.BroadcastUint16x8(4))
			o1 = s.ShiftAllRight(3).IfElse(mFlat, o1)
			s = s.Add(q0.Add(q1)).Sub(p2.Add(p2))
			o0 = s.ShiftAllRight(3).IfElse(mFlat, o0)
			s = s.Add(q1.Add(q2)).Sub(p2.Add(p1))
			n0 = s.ShiftAllRight(3).IfElse(mFlat, n0)
			s = s.Add(q2.Add(q2)).Sub(p1.Add(p0))
			n1 = s.ShiftAllRight(3).IfElse(mFlat, n1)
		}
		lpf16Store(p, -2*stride, o1)
		lpf16Store(p, -stride, o0)
		lpf16Store(p, 0, n0)
		lpf16Store(p, stride, n1)
		return
	}

	o2, n2 := p2, q2
	if wd >= 8 && lpf16Any(mFlat) {
		s := p3.Add(p3).Add(p3).Add(p2.Add(p2)).Add(p1.Add(p0)).Add(q0.Add(archsimd.BroadcastUint16x8(4)))
		o2 = s.ShiftAllRight(3).IfElse(mFlat, o2)
		s = s.Add(p1.Add(q1)).Sub(p3.Add(p2))
		o1 = s.ShiftAllRight(3).IfElse(mFlat, o1)
		s = s.Add(p0.Add(q2)).Sub(p3.Add(p1))
		o0 = s.ShiftAllRight(3).IfElse(mFlat, o0)
		s = s.Add(q0.Add(q3)).Sub(p3.Add(p0))
		n0 = s.ShiftAllRight(3).IfElse(mFlat, n0)
		s = s.Add(q1.Add(q3)).Sub(p2.Add(q0))
		n1 = s.ShiftAllRight(3).IfElse(mFlat, n1)
		s = s.Add(q2.Add(q3)).Sub(p1.Add(q1))
		n2 = s.ShiftAllRight(3).IfElse(mFlat, n2)
	}

	if wd == 16 && lpf16Any(mFlat16) {
		m := mFlat16
		p6x2 := p6.Add(p6)
		s := p6.Add(p6x2).Add(p6x2.Add(p6x2))
		s = s.Add(p5.Add(p5)).Add(p4.Add(p4)).Add(p3.Add(p2)).Add(p1.Add(p0)).Add(q0).Add(archsimd.BroadcastUint16x8(8))
		lpf16Store(p, -6*stride, s.ShiftAllRight(4).IfElse(m, p5))
		s = s.Add(p3.Add(q1)).Sub(p6x2)
		lpf16Store(p, -5*stride, s.ShiftAllRight(4).IfElse(m, p4))
		s = s.Add(p2.Add(q2)).Sub(p6.Add(p5))
		lpf16Store(p, -4*stride, s.ShiftAllRight(4).IfElse(m, p3))
		s = s.Add(p1.Add(q3)).Sub(p6.Add(p4))
		o2 = s.ShiftAllRight(4).IfElse(m, o2)
		s = s.Add(p0.Add(q4)).Sub(p6.Add(p3))
		o1 = s.ShiftAllRight(4).IfElse(m, o1)
		s = s.Add(q0.Add(q5)).Sub(p6.Add(p2))
		o0 = s.ShiftAllRight(4).IfElse(m, o0)
		s = s.Add(q1.Add(q6)).Sub(p6.Add(p1))
		n0 = s.ShiftAllRight(4).IfElse(m, n0)
		s = s.Add(q2.Add(q6)).Sub(p5.Add(p0))
		n1 = s.ShiftAllRight(4).IfElse(m, n1)
		s = s.Add(q3.Add(q6)).Sub(p4.Add(q0))
		n2 = s.ShiftAllRight(4).IfElse(m, n2)
		s = s.Add(q4.Add(q6)).Sub(p3.Add(q1))
		lpf16Store(p, 3*stride, s.ShiftAllRight(4).IfElse(m, q3))
		s = s.Add(q5.Add(q6)).Sub(p2.Add(q2))
		lpf16Store(p, 4*stride, s.ShiftAllRight(4).IfElse(m, q4))
		s = s.Add(q6.Add(q6)).Sub(p1.Add(q3))
		lpf16Store(p, 5*stride, s.ShiftAllRight(4).IfElse(m, q5))
	}
	if wd >= 8 {
		lpf16Store(p, -3*stride, o2)
		lpf16Store(p, 2*stride, n2)
	}
	lpf16Store(p, -2*stride, o1)
	lpf16Store(p, -stride, o0)
	lpf16Store(p, 0, n0)
	lpf16Store(p, stride, n1)
}

// lpf8U16Regs is lpf8U16 for widths up to 8 on taps held in registers: p3..q3
// in, the filtered p2..q2 out (p3/q3 never change at these widths); ok is
// false when no lane passed the filter mask and nothing changed.
func lpf8U16Regs(wd int, bd8 uint, p3, p2, p1, p0, q0, q1, q2, q3, eV, iV, hV archsimd.Uint16x8, m4, mGt4 archsimd.Mask16x8) (o2, o1, o0, n0, n1, n2 archsimd.Uint16x8, ok bool) {
	o2, o1, o0, n0, n1, n2 = p2, p1, p0, q0, q1, q2
	d10 := lpf16AbsDiff(p1, p0)
	dq := lpf16AbsDiff(q1, q0)
	maxd := d10.Max(dq)
	t := maxd
	if wd >= 6 {
		tt := lpf16AbsDiff(p2, p1).Max(lpf16AbsDiff(q2, q1))
		if wd >= 8 {
			tt = tt.Max(lpf16AbsDiff(p3, p2).Max(lpf16AbsDiff(q3, q2)))
		}
		t = t.Max(tt.Masked(mGt4))
	}
	d00 := lpf16AbsDiff(p0, q0)
	sum := d00.Add(d00).Add(lpf16AbsDiff(p1, q1).ShiftAllRight(1))
	fm := iV.GreaterEqual(t).And(eV.GreaterEqual(sum)).And(m4)
	if !lpf16Any(fm) {
		return
	}
	ok = true
	mF4 := fm
	var mFlat archsimd.Mask16x8
	if wd >= 6 {
		fl := lpf16AbsDiff(p2, p0).Max(maxd).Max(lpf16AbsDiff(q2, q0))
		if wd >= 8 {
			fl = fl.Max(lpf16AbsDiff(p3, p0)).Max(lpf16AbsDiff(q3, q0))
		}
		mFlat = archsimd.BroadcastUint16x8(uint16(1) << bd8).GreaterEqual(fl).And(mGt4).And(fm)
		mF4 = fm.And(mFlat.Not())
	}
	if lpf16Any(mF4) {
		hev := maxd.Greater(hV)
		dmin := archsimd.BroadcastInt16x8(int16(-128 << bd8))
		dmax := archsimd.BroadcastInt16x8(int16(128<<bd8 - 1))
		zero := archsimd.Int16x8{}
		maxPix := archsimd.BroadcastInt16x8(int16(256<<bd8 - 1))
		p1s, p0s, q0s, q1s := p1.BitsToInt16(), p0.BitsToInt16(), q0.BitsToInt16(), q1.BitsToInt16()
		f := p1s.Sub(q1s).Max(dmin).Min(dmax).Masked(hev)
		f = q0s.Sub(p0s).Mul(archsimd.BroadcastInt16x8(3)).Add(f).Max(dmin).Min(dmax)
		f1 := f.Add(archsimd.BroadcastInt16x8(4)).Min(dmax).ShiftAllRight(3)
		f2 := f.Add(archsimd.BroadcastInt16x8(3)).Min(dmax).ShiftAllRight(3)
		o0 = p0s.Add(f2).Max(zero).Min(maxPix).ToBits().IfElse(mF4, p0)
		n0 = q0s.Sub(f1).Max(zero).Min(maxPix).ToBits().IfElse(mF4, q0)
		f3 := f1.Add(archsimd.BroadcastInt16x8(1)).ShiftAllRight(1)
		mOuter := mF4.And(hev.Not())
		o1 = p1s.Add(f3).Max(zero).Min(maxPix).ToBits().IfElse(mOuter, p1)
		n1 = q1s.Sub(f3).Max(zero).Min(maxPix).ToBits().IfElse(mOuter, q1)
	}
	if !lpf16Any(mFlat) {
		return
	}
	if wd == 6 {
		s := p2.Add(p2).Add(p2).Add(p1.Add(p0).Add(p1.Add(p0))).Add(q0).Add(archsimd.BroadcastUint16x8(4))
		o1 = s.ShiftAllRight(3).IfElse(mFlat, o1)
		s = s.Add(q0.Add(q1)).Sub(p2.Add(p2))
		o0 = s.ShiftAllRight(3).IfElse(mFlat, o0)
		s = s.Add(q1.Add(q2)).Sub(p2.Add(p1))
		n0 = s.ShiftAllRight(3).IfElse(mFlat, n0)
		s = s.Add(q2.Add(q2)).Sub(p1.Add(p0))
		n1 = s.ShiftAllRight(3).IfElse(mFlat, n1)
		return
	}
	s := p3.Add(p3).Add(p3).Add(p2.Add(p2)).Add(p1.Add(p0)).Add(q0.Add(archsimd.BroadcastUint16x8(4)))
	o2 = s.ShiftAllRight(3).IfElse(mFlat, o2)
	s = s.Add(p1.Add(q1)).Sub(p3.Add(p2))
	o1 = s.ShiftAllRight(3).IfElse(mFlat, o1)
	s = s.Add(p0.Add(q2)).Sub(p3.Add(p1))
	o0 = s.ShiftAllRight(3).IfElse(mFlat, o0)
	s = s.Add(q0.Add(q3)).Sub(p3.Add(p0))
	n0 = s.ShiftAllRight(3).IfElse(mFlat, n0)
	s = s.Add(q1.Add(q3)).Sub(p2.Add(q0))
	n1 = s.ShiftAllRight(3).IfElse(mFlat, n1)
	s = s.Add(q2.Add(q3)).Sub(p1.Add(q1))
	n2 = s.ShiftAllRight(3).IfElse(mFlat, n2)
	return
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

// lpfTransposeRows8U16 loads eight rows of eight uint16 samples from src
// (byte stride), transposes them and stores eight rows at dst.
//
//go:nocheckptr
func lpfTransposeRows8U16(src unsafe.Pointer, srcStride int, dst unsafe.Pointer, dstStride int) {
	o0, o1, o2, o3, o4, o5, o6, o7 := lpfTranspose8x8U16Regs(
		lpf16Load(src, 0), lpf16Load(src, srcStride), lpf16Load(src, 2*srcStride), lpf16Load(src, 3*srcStride),
		lpf16Load(src, 4*srcStride), lpf16Load(src, 5*srcStride), lpf16Load(src, 6*srcStride), lpf16Load(src, 7*srcStride))
	lpf16Store(dst, 0, o0)
	lpf16Store(dst, dstStride, o1)
	lpf16Store(dst, 2*dstStride, o2)
	lpf16Store(dst, 3*dstStride, o3)
	lpf16Store(dst, 4*dstStride, o4)
	lpf16Store(dst, 5*dstStride, o5)
	lpf16Store(dst, 6*dstStride, o6)
	lpf16Store(dst, 7*dstStride, o7)
}

// lpfVertical8x8U16 filters a vertical edge across eight rows: p points at p3
// of the first row (eight samples p3..q3 per row). The rows are transposed,
// filtered and transposed back in registers.
//
//go:nocheckptr
func lpfVertical8x8U16(p unsafe.Pointer, stride int, wd int, bd8 uint, eV, iV, hV archsimd.Uint16x8, m4, mGt4 archsimd.Mask16x8) {
	p3, p2, p1, p0, q0, q1, q2, q3 := lpfTranspose8x8U16Regs(
		lpf16Load(p, 0), lpf16Load(p, stride), lpf16Load(p, 2*stride), lpf16Load(p, 3*stride),
		lpf16Load(p, 4*stride), lpf16Load(p, 5*stride), lpf16Load(p, 6*stride), lpf16Load(p, 7*stride))
	o2, o1, o0, n0, n1, n2, ok := lpf8U16Regs(wd, bd8, p3, p2, p1, p0, q0, q1, q2, q3, eV, iV, hV, m4, mGt4)
	if !ok {
		return
	}
	r0, r1, r2, r3, r4, r5, r6, r7 := lpfTranspose8x8U16Regs(p3, o2, o1, o0, n0, n1, n2, q3)
	lpf16Store(p, 0, r0)
	lpf16Store(p, stride, r1)
	lpf16Store(p, 2*stride, r2)
	lpf16Store(p, 3*stride, r3)
	lpf16Store(p, 4*stride, r4)
	lpf16Store(p, 5*stride, r5)
	lpf16Store(p, 6*stride, r6)
	lpf16Store(p, 7*stride, r7)
}

// lpfVertical8x16U16 is the wd16 form: sixteen samples p7..q7 per row, p
// points at p7 of the first row.
//
//go:nocheckptr
func lpfVertical8x16U16(p unsafe.Pointer, stride int, bd8 uint, eV, iV, hV archsimd.Uint16x8, m4, mGt4, m16 archsimd.Mask16x8) {
	var buf [16][8]uint16
	lpfTransposeRows8U16(p, stride, unsafe.Pointer(&buf[0]), 16)
	lpfTransposeRows8U16(unsafe.Add(p, 16), stride, unsafe.Pointer(&buf[8]), 16)
	lpf8U16(unsafe.Pointer(&buf[8]), 16, 16, bd8, eV, iV, hV, m4, mGt4, m16)
	lpfTransposeRows8U16(unsafe.Pointer(&buf[0]), 16, p, stride)
	lpfTransposeRows8U16(unsafe.Pointer(&buf[8]), 16, unsafe.Add(p, 16), stride)
}

// sbResolveHalf16 resolves the levels of units 2*half and 2*half+1 of the
// nibble (lvlStep: level-index step between units, prevStep: fallback step).
func sbResolveHalf16(s *sbHalf16, nib uint32, half int, u0, lvlStep, prevStep int, lvl [][4]uint8, comp int, lut *SBLUT, bd8 uint, st *SBStats) {
	for k := 0; k < 2; k++ {
		unit := 2*half + k
		if nib&(1<<uint(unit)) == 0 {
			continue
		}
		li := u0 + unit*lvlStep
		L := lvl[li][comp]
		if L == 0 {
			L = lvl[li-prevStep][comp]
		}
		if L == 0 {
			continue
		}
		s.set(k, L, lut, bd8)
		s.m4 |= 1 << uint(k)
		st.Applied++
		if L > st.MaxLevel {
			st.MaxLevel = L
		}
	}
}

//go:nocheckptr
func filterSBHorizontal16(pix []byte, off, stride, rowBytes int, bitDepth uint8, vm [3]uint32, luma bool, lvl [][4]uint8, lvlOff, lvlStride, comp int, lut *SBLUT) SBStats {
	var st SBStats
	bd8 := uint(bitDepth - 8)
	gt4All := vm[1] | vm[2]
	all := vm[0] | gt4All
	base := unsafe.Pointer(unsafe.SliceData(pix))
	for g := 0; g < 16 && all>>(2*uint(g)) != 0; g++ {
		sh := 2 * uint(g)
		pair := (all >> sh) & 3
		if pair == 0 {
			continue
		}
		var s sbHalf16
		var gst SBStats
		u0 := lvlOff + 2*g
		sbResolveHalf16(&s, pair, 0, u0, 1, lvlStride, lvl, comp, lut, bd8, &gst)
		if s.m4 == 0 {
			continue
		}
		st.Add(gst)
		s.gt4 = (gt4All >> sh) & s.m4
		s.w16 = (vm[2] >> sh) & s.m4
		wd := s.width(luma)
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
			_ = filterSBHorizontal16Scalar(pix, q0Off, stride, bitDepth, sub, luma, lvl, u0, lvlStride, comp, lut)
			continue
		}
		eV, iV, hV, m4, mGt4, m16 := s.vectors()
		lpf8U16(unsafe.Add(base, q0Off), stride, wd, bd8, eV, iV, hV, m4, mGt4, m16)
	}
	return st
}

//go:nocheckptr
func filterSBVertical16(pix []byte, off, stride, rowBytes int, bitDepth uint8, vm [3]uint32, luma bool, lvl [][4]uint8, lvlOff, lvlStride, comp int, lut *SBLUT) SBStats {
	var st SBStats
	bd8 := uint(bitDepth - 8)
	gt4All := vm[1] | vm[2]
	all := vm[0] | gt4All
	base := unsafe.Pointer(unsafe.SliceData(pix))
	for g := 0; g < 16 && all>>(2*uint(g)) != 0; g++ {
		sh := 2 * uint(g)
		pair := (all >> sh) & 3
		if pair == 0 {
			continue
		}
		var s sbHalf16
		var gst SBStats
		u0 := lvlOff + 2*g*lvlStride
		sbResolveHalf16(&s, pair, 0, u0, lvlStride, 1, lvl, comp, lut, bd8, &gst)
		if s.m4 == 0 {
			continue
		}
		st.Add(gst)
		s.gt4 = (gt4All >> sh) & s.m4
		s.w16 = (vm[2] >> sh) & s.m4
		wd := s.width(luma)
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
			_ = filterSBVertical16Scalar(pix, q0Off, stride, bitDepth, sub, luma, lvl, u0, lvlStride, comp, lut)
			continue
		}
		eV, iV, hV, m4, mGt4, m16 := s.vectors()
		if wd == 16 {
			lpfVertical8x16U16(unsafe.Add(base, q0Off-16), stride, bd8, eV, iV, hV, m4, mGt4, m16)
		} else {
			lpfVertical8x8U16(unsafe.Add(base, q0Off-8), stride, wd, bd8, eV, iV, hV, m4, mGt4)
		}
	}
	return st
}
