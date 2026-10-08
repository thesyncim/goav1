// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package transform

// The hybrid cores keep both passes and the 8x8 transpose in vector locals.
// The guarded public bindings preserve the scalar path for wider residuals.
func fwd8ADSTDCTCore(coeff []int32, coeffStride int, residual []int16, residualStride int, scratch []int32) {
	var c0l, c1l, c2l, c3l, c4l, c5l, c6l, c7l, c0h, c1h, c2h, c3h, c4h, c5h, c6h, c7h fwdVec
	{
		g := 0
		rs := residualStride

		x0 := fwdLoadResAt(residual, 0*rs+g&^7, g)
		x1 := fwdLoadResAt(residual, 1*rs+g&^7, g)
		x2 := fwdLoadResAt(residual, 2*rs+g&^7, g)
		x3 := fwdLoadResAt(residual, 3*rs+g&^7, g)
		x4 := fwdLoadResAt(residual, 4*rs+g&^7, g)
		x5 := fwdLoadResAt(residual, 5*rs+g&^7, g)
		x6 := fwdLoadResAt(residual, 6*rs+g&^7, g)
		x7 := fwdLoadResAt(residual, 7*rs+g&^7, g)
		c4 := fwdConst8153
		c12 := fwdConst7839
		c16 := fwdConst7568
		c20 := fwdConst7225
		c28 := fwdConst6333
		c32 := fwdConst5793
		nc32 := fwdConstN5793
		c36 := fwdConst5197
		c44 := fwdConst3862
		c48 := fwdConst3135
		nc48 := fwdConstN3135
		c52 := fwdConst2378
		c60 := fwdConst803
		nc4 := fwdConstN8153
		nc20 := fwdConstN7225
		nc16 := fwdConstN7568
		nc36 := fwdConstN5197
		nc52 := fwdConstN2378

		nx3 := x3.Neg()
		nx7 := x7.Neg()
		nx5 := x5.Neg()
		nx1 := x1.Neg()

		s0 := x0
		s1 := nx7
		s2 := fwdHalfBtf13V(c32, nx3, c32, x4)
		s3 := fwdHalfBtf13V(c32, nx3, nc32, x4)
		s4 := nx1
		s5 := x6
		s6 := fwdHalfBtf13V(c32, x2, c32, nx5)
		s7 := fwdHalfBtf13V(c32, x2, nc32, nx5)

		t0 := s0.Add(s2)
		t1 := s1.Add(s3)
		t2 := s0.Sub(s2)
		t3 := s1.Sub(s3)
		t4 := s4.Add(s6)
		t5 := s5.Add(s7)
		t6 := s4.Sub(s6)
		t7 := s5.Sub(s7)

		s4 = fwdHalfBtf13V(c16, t4, c48, t5)
		s5 = fwdHalfBtf13V(c48, t4, nc16, t5)
		s6 = fwdHalfBtf13V(nc48, t6, c16, t7)
		s7 = fwdHalfBtf13V(c16, t6, c48, t7)

		t4 = t0.Sub(s4)
		t5 = t1.Sub(s5)
		t6 = t2.Sub(s6)
		t7 = t3.Sub(s7)
		t0 = t0.Add(s4)
		t1 = t1.Add(s5)
		t2 = t2.Add(s6)
		t3 = t3.Add(s7)

		s0 = fwdHalfBtf13V(c4, t0, c60, t1)
		s1 = fwdHalfBtf13V(c60, t0, nc4, t1)
		s2 = fwdHalfBtf13V(c20, t2, c44, t3)
		s3 = fwdHalfBtf13V(c44, t2, nc20, t3)
		s4 = fwdHalfBtf13V(c36, t4, c28, t5)
		s5 = fwdHalfBtf13V(c28, t4, nc36, t5)
		s6 = fwdHalfBtf13V(c52, t6, c12, t7)
		s7 = fwdHalfBtf13V(c12, t6, nc52, t7)
		c0l = fwdRoundShift1V(s1)
		c1l = fwdRoundShift1V(s6)
		c2l = fwdRoundShift1V(s3)
		c3l = fwdRoundShift1V(s4)
		c4l = fwdRoundShift1V(s5)
		c5l = fwdRoundShift1V(s2)
		c6l = fwdRoundShift1V(s7)
		c7l = fwdRoundShift1V(s0)
	}
	{
		g := 4
		rs := residualStride

		x0 := fwdLoadResAt(residual, 0*rs+g&^7, g)
		x1 := fwdLoadResAt(residual, 1*rs+g&^7, g)
		x2 := fwdLoadResAt(residual, 2*rs+g&^7, g)
		x3 := fwdLoadResAt(residual, 3*rs+g&^7, g)
		x4 := fwdLoadResAt(residual, 4*rs+g&^7, g)
		x5 := fwdLoadResAt(residual, 5*rs+g&^7, g)
		x6 := fwdLoadResAt(residual, 6*rs+g&^7, g)
		x7 := fwdLoadResAt(residual, 7*rs+g&^7, g)
		c4 := fwdConst8153
		c12 := fwdConst7839
		c16 := fwdConst7568
		c20 := fwdConst7225
		c28 := fwdConst6333
		c32 := fwdConst5793
		nc32 := fwdConstN5793
		c36 := fwdConst5197
		c44 := fwdConst3862
		c48 := fwdConst3135
		nc48 := fwdConstN3135
		c52 := fwdConst2378
		c60 := fwdConst803
		nc4 := fwdConstN8153
		nc20 := fwdConstN7225
		nc16 := fwdConstN7568
		nc36 := fwdConstN5197
		nc52 := fwdConstN2378

		nx3 := x3.Neg()
		nx7 := x7.Neg()
		nx5 := x5.Neg()
		nx1 := x1.Neg()

		s0 := x0
		s1 := nx7
		s2 := fwdHalfBtf13V(c32, nx3, c32, x4)
		s3 := fwdHalfBtf13V(c32, nx3, nc32, x4)
		s4 := nx1
		s5 := x6
		s6 := fwdHalfBtf13V(c32, x2, c32, nx5)
		s7 := fwdHalfBtf13V(c32, x2, nc32, nx5)

		t0 := s0.Add(s2)
		t1 := s1.Add(s3)
		t2 := s0.Sub(s2)
		t3 := s1.Sub(s3)
		t4 := s4.Add(s6)
		t5 := s5.Add(s7)
		t6 := s4.Sub(s6)
		t7 := s5.Sub(s7)

		s4 = fwdHalfBtf13V(c16, t4, c48, t5)
		s5 = fwdHalfBtf13V(c48, t4, nc16, t5)
		s6 = fwdHalfBtf13V(nc48, t6, c16, t7)
		s7 = fwdHalfBtf13V(c16, t6, c48, t7)

		t4 = t0.Sub(s4)
		t5 = t1.Sub(s5)
		t6 = t2.Sub(s6)
		t7 = t3.Sub(s7)
		t0 = t0.Add(s4)
		t1 = t1.Add(s5)
		t2 = t2.Add(s6)
		t3 = t3.Add(s7)

		s0 = fwdHalfBtf13V(c4, t0, c60, t1)
		s1 = fwdHalfBtf13V(c60, t0, nc4, t1)
		s2 = fwdHalfBtf13V(c20, t2, c44, t3)
		s3 = fwdHalfBtf13V(c44, t2, nc20, t3)
		s4 = fwdHalfBtf13V(c36, t4, c28, t5)
		s5 = fwdHalfBtf13V(c28, t4, nc36, t5)
		s6 = fwdHalfBtf13V(c52, t6, c12, t7)
		s7 = fwdHalfBtf13V(c12, t6, nc52, t7)
		c0h = fwdRoundShift1V(s1)
		c1h = fwdRoundShift1V(s6)
		c2h = fwdRoundShift1V(s3)
		c3h = fwdRoundShift1V(s4)
		c4h = fwdRoundShift1V(s5)
		c5h = fwdRoundShift1V(s2)
		c6h = fwdRoundShift1V(s7)
		c7h = fwdRoundShift1V(s0)
	}
	{
		h := 0

		le0 := c0l.InterleaveLo(c1l)
		le1 := c0l.InterleaveHi(c1l)
		le2 := c2l.InterleaveLo(c3l)
		le3 := c2l.InterleaveHi(c3l)
		x0 := fwdI64AsI32(fwdI32AsI64(le0).InterleaveLo(fwdI32AsI64(le2)))
		x1 := fwdI64AsI32(fwdI32AsI64(le0).InterleaveHi(fwdI32AsI64(le2)))
		x2 := fwdI64AsI32(fwdI32AsI64(le1).InterleaveLo(fwdI32AsI64(le3)))
		x3 := fwdI64AsI32(fwdI32AsI64(le1).InterleaveHi(fwdI32AsI64(le3)))

		he0 := c0h.InterleaveLo(c1h)
		he1 := c0h.InterleaveHi(c1h)
		he2 := c2h.InterleaveLo(c3h)
		he3 := c2h.InterleaveHi(c3h)
		x4 := fwdI64AsI32(fwdI32AsI64(he0).InterleaveLo(fwdI32AsI64(he2)))
		x5 := fwdI64AsI32(fwdI32AsI64(he0).InterleaveHi(fwdI32AsI64(he2)))
		x6 := fwdI64AsI32(fwdI32AsI64(he1).InterleaveLo(fwdI32AsI64(he3)))
		x7 := fwdI64AsI32(fwdI32AsI64(he1).InterleaveHi(fwdI32AsI64(he3)))
		c8 := fwdConst8035
		c16 := fwdConst7568
		c24 := fwdConst6811
		c32 := fwdConst5793
		nc32 := fwdConstN5793
		c40 := fwdConst4551
		c48 := fwdConst3135
		c56 := fwdConst1598
		nc8 := fwdConstN8035
		nc16 := fwdConstN7568
		nc40 := fwdConstN4551

		b0 := x0.Add(x7)
		b1 := x1.Add(x6)
		b2 := x2.Add(x5)
		b3 := x3.Add(x4)
		b4 := x3.Sub(x4)
		b5 := x2.Sub(x5)
		b6 := x1.Sub(x6)
		b7 := x0.Sub(x7)

		s0 := b0.Add(b3)
		s1 := b1.Add(b2)
		s2 := b1.Sub(b2)
		s3 := b0.Sub(b3)
		s4 := b4
		s5 := fwdHalfBtf13V(nc32, b5, c32, b6)
		s6 := fwdHalfBtf13V(c32, b6, c32, b5)
		s7 := b7

		b0 = fwdHalfBtf13V(c32, s0, c32, s1)
		b1 = fwdHalfBtf13V(nc32, s1, c32, s0)
		b2 = fwdHalfBtf13V(c48, s2, c16, s3)
		b3 = fwdHalfBtf13V(c48, s3, nc16, s2)
		b4 = s4.Add(s5)
		b5 = s4.Sub(s5)
		b6 = s7.Sub(s6)
		b7 = s7.Add(s6)

		s0 = b0
		s1 = b1
		s2 = b2
		s3 = b3
		s4 = fwdHalfBtf13V(c56, b4, c8, b7)
		s5 = fwdHalfBtf13V(c24, b5, c40, b6)
		s6 = fwdHalfBtf13V(c24, b6, nc40, b5)
		s7 = fwdHalfBtf13V(c56, b7, nc8, b4)
		fwdStoreI32At(coeff, 0*coeffStride+h, s0)
		fwdStoreI32At(coeff, 1*coeffStride+h, s4)
		fwdStoreI32At(coeff, 2*coeffStride+h, s2)
		fwdStoreI32At(coeff, 3*coeffStride+h, s6)
		fwdStoreI32At(coeff, 4*coeffStride+h, s1)
		fwdStoreI32At(coeff, 5*coeffStride+h, s5)
		fwdStoreI32At(coeff, 6*coeffStride+h, s3)
		fwdStoreI32At(coeff, 7*coeffStride+h, s7)
	}
	{
		h := 4

		le0 := c4l.InterleaveLo(c5l)
		le1 := c4l.InterleaveHi(c5l)
		le2 := c6l.InterleaveLo(c7l)
		le3 := c6l.InterleaveHi(c7l)
		x0 := fwdI64AsI32(fwdI32AsI64(le0).InterleaveLo(fwdI32AsI64(le2)))
		x1 := fwdI64AsI32(fwdI32AsI64(le0).InterleaveHi(fwdI32AsI64(le2)))
		x2 := fwdI64AsI32(fwdI32AsI64(le1).InterleaveLo(fwdI32AsI64(le3)))
		x3 := fwdI64AsI32(fwdI32AsI64(le1).InterleaveHi(fwdI32AsI64(le3)))

		he0 := c4h.InterleaveLo(c5h)
		he1 := c4h.InterleaveHi(c5h)
		he2 := c6h.InterleaveLo(c7h)
		he3 := c6h.InterleaveHi(c7h)
		x4 := fwdI64AsI32(fwdI32AsI64(he0).InterleaveLo(fwdI32AsI64(he2)))
		x5 := fwdI64AsI32(fwdI32AsI64(he0).InterleaveHi(fwdI32AsI64(he2)))
		x6 := fwdI64AsI32(fwdI32AsI64(he1).InterleaveLo(fwdI32AsI64(he3)))
		x7 := fwdI64AsI32(fwdI32AsI64(he1).InterleaveHi(fwdI32AsI64(he3)))
		c8 := fwdConst8035
		c16 := fwdConst7568
		c24 := fwdConst6811
		c32 := fwdConst5793
		nc32 := fwdConstN5793
		c40 := fwdConst4551
		c48 := fwdConst3135
		c56 := fwdConst1598
		nc8 := fwdConstN8035
		nc16 := fwdConstN7568
		nc40 := fwdConstN4551

		b0 := x0.Add(x7)
		b1 := x1.Add(x6)
		b2 := x2.Add(x5)
		b3 := x3.Add(x4)
		b4 := x3.Sub(x4)
		b5 := x2.Sub(x5)
		b6 := x1.Sub(x6)
		b7 := x0.Sub(x7)

		s0 := b0.Add(b3)
		s1 := b1.Add(b2)
		s2 := b1.Sub(b2)
		s3 := b0.Sub(b3)
		s4 := b4
		s5 := fwdHalfBtf13V(nc32, b5, c32, b6)
		s6 := fwdHalfBtf13V(c32, b6, c32, b5)
		s7 := b7

		b0 = fwdHalfBtf13V(c32, s0, c32, s1)
		b1 = fwdHalfBtf13V(nc32, s1, c32, s0)
		b2 = fwdHalfBtf13V(c48, s2, c16, s3)
		b3 = fwdHalfBtf13V(c48, s3, nc16, s2)
		b4 = s4.Add(s5)
		b5 = s4.Sub(s5)
		b6 = s7.Sub(s6)
		b7 = s7.Add(s6)

		s0 = b0
		s1 = b1
		s2 = b2
		s3 = b3
		s4 = fwdHalfBtf13V(c56, b4, c8, b7)
		s5 = fwdHalfBtf13V(c24, b5, c40, b6)
		s6 = fwdHalfBtf13V(c24, b6, nc40, b5)
		s7 = fwdHalfBtf13V(c56, b7, nc8, b4)
		fwdStoreI32At(coeff, 0*coeffStride+h, s0)
		fwdStoreI32At(coeff, 1*coeffStride+h, s4)
		fwdStoreI32At(coeff, 2*coeffStride+h, s2)
		fwdStoreI32At(coeff, 3*coeffStride+h, s6)
		fwdStoreI32At(coeff, 4*coeffStride+h, s1)
		fwdStoreI32At(coeff, 5*coeffStride+h, s5)
		fwdStoreI32At(coeff, 6*coeffStride+h, s3)
		fwdStoreI32At(coeff, 7*coeffStride+h, s7)
	}
}
