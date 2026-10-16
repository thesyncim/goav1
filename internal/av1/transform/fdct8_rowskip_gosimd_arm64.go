// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package transform

import "simd/archsimd"

// forwardDCT8x8RowSkipSIMD is fwd8DCTCore with the row-skip bound of
// forwardDCT8x8RowSkipPureGo between its register-resident column and row
// passes: each cNl/cNh pair is column-pass row N, so |cNl|+|cNh| carries that
// row's magnitude sum in four partial lanes.
func forwardDCT8x8RowSkipSIMD(coeffArr *[64]int32, residualArr *[64]int16, thr0, thr int32) uint8 {
	coeff, coeffStride, residual, residualStride := coeffArr[:], 8, residualArr[:], 8
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
		c0l = fwdRoundShift1V(s0)
		c1l = fwdRoundShift1V(s4)
		c2l = fwdRoundShift1V(s2)
		c3l = fwdRoundShift1V(s6)
		c4l = fwdRoundShift1V(s1)
		c5l = fwdRoundShift1V(s5)
		c6l = fwdRoundShift1V(s3)
		c7l = fwdRoundShift1V(s7)
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
		c0h = fwdRoundShift1V(s0)
		c1h = fwdRoundShift1V(s4)
		c2h = fwdRoundShift1V(s2)
		c3h = fwdRoundShift1V(s6)
		c4h = fwdRoundShift1V(s1)
		c5h = fwdRoundShift1V(s5)
		c6h = fwdRoundShift1V(s3)
		c7h = fwdRoundShift1V(s7)
	}
	thrRest := archsimd.BroadcastInt32x4(thr)
	var mask uint8
	if fwdRowSkipGroupLive(c0l.Abs().Add(c0h.Abs()), c1l.Abs().Add(c1h.Abs()),
		c2l.Abs().Add(c2h.Abs()), c3l.Abs().Add(c3h.Abs()), thrRest.SetElem(0, thr0), fwdRowSkipSlack8V) {
		mask |= 1
	}
	if fwdRowSkipGroupLive(c4l.Abs().Add(c4h.Abs()), c5l.Abs().Add(c5h.Abs()),
		c6l.Abs().Add(c6h.Abs()), c7l.Abs().Add(c7h.Abs()), thrRest, fwdRowSkipSlack8V) {
		mask |= 2
	}
	if mask == 0 {
		return 0
	}
	zero := archsimd.BroadcastInt32x4(0)
	if mask&1 == 0 {
		for l := range 8 {
			fwdStoreI32At(coeff, l*coeffStride, zero)
		}
	} else {
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
	if mask&2 == 0 {
		for l := range 8 {
			fwdStoreI32At(coeff, l*coeffStride+4, zero)
		}
	} else {
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
	return mask
}
