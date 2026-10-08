// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && (arm64 || amd64) && !purego

package transform

// Shared Go SIMD forward-transform network. Each 1-D pass is vectorized across
// independent lines: a fwdVec holds one sample index for fwdLanes lines (the
// columns on the column pass, the rows on the row pass), so every butterfly is
// one vector operation. The butterfly bodies mirror fwdDCT8Values and
// fwdADST8Values, and the drivers mirror the PureGo two-pass references, so the
// output is bit-identical to the scalar oracle for every residual the callers'
// guards admit. Intermediates stay in int32 lanes; the guards bound them.

// fwdRound13 and fwdRound12 are the half_btf rounding terms, pre-broadcast so
// each use is one load instead of a broadcast.
var (
	fwdRound13 = fwdBcast(1 << 12)
	fwdRound12 = fwdBcast(1 << 11)
)

// fwdRoundShift1V is fwdRoundShift1Value per lane: (v + 1 + (v>>31)) >> 1.
func fwdRoundShift1V(v fwdVec) fwdVec {
	return fwdShr(v.Add(fwdConst1).Add(fwdShr(v, 31)), 1)
}

// fwdColDCT8 runs the DCT8 column pass for the fwdLanes columns starting at g
// and stores the shift[1]=-1 rounded outputs into buf (row-major, stride 8).
func fwdColDCT8(buf []int32, g int, residual []int16, rs int) {
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
	fwdStoreI32At(buf, 0*8+g, fwdRoundShift1V(s0))
	fwdStoreI32At(buf, 1*8+g, fwdRoundShift1V(s4))
	fwdStoreI32At(buf, 2*8+g, fwdRoundShift1V(s2))
	fwdStoreI32At(buf, 3*8+g, fwdRoundShift1V(s6))
	fwdStoreI32At(buf, 4*8+g, fwdRoundShift1V(s1))
	fwdStoreI32At(buf, 5*8+g, fwdRoundShift1V(s5))
	fwdStoreI32At(buf, 6*8+g, fwdRoundShift1V(s3))
	fwdStoreI32At(buf, 7*8+g, fwdRoundShift1V(s7))
}

// fwdColADST8 is fwdColDCT8 with the ADST8 column transform.
func fwdColADST8(buf []int32, g int, residual []int16, rs int) {
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
	fwdStoreI32At(buf, 0*8+g, fwdRoundShift1V(s1))
	fwdStoreI32At(buf, 1*8+g, fwdRoundShift1V(s6))
	fwdStoreI32At(buf, 2*8+g, fwdRoundShift1V(s3))
	fwdStoreI32At(buf, 3*8+g, fwdRoundShift1V(s4))
	fwdStoreI32At(buf, 4*8+g, fwdRoundShift1V(s5))
	fwdStoreI32At(buf, 5*8+g, fwdRoundShift1V(s2))
	fwdStoreI32At(buf, 6*8+g, fwdRoundShift1V(s7))
	fwdStoreI32At(buf, 7*8+g, fwdRoundShift1V(s0))
}

// fwdRowDCT8 runs the DCT8 row pass for the fwdLanes rows starting at h. bufT
// is the transposed column-pass output: bufT[c*8+k] is the value for row k.
func fwdRowDCT8(coeff []int32, coeffStride int, bufT []int32, h int) {
	x0 := fwdLoadI32At(bufT, 0*8+h)
	x1 := fwdLoadI32At(bufT, 1*8+h)
	x2 := fwdLoadI32At(bufT, 2*8+h)
	x3 := fwdLoadI32At(bufT, 3*8+h)
	x4 := fwdLoadI32At(bufT, 4*8+h)
	x5 := fwdLoadI32At(bufT, 5*8+h)
	x6 := fwdLoadI32At(bufT, 6*8+h)
	x7 := fwdLoadI32At(bufT, 7*8+h)
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

// fwdRowADST8 is fwdRowDCT8 with the ADST8 row transform.
func fwdRowADST8(coeff []int32, coeffStride int, bufT []int32, h int) {
	x0 := fwdLoadI32At(bufT, 0*8+h)
	x1 := fwdLoadI32At(bufT, 1*8+h)
	x2 := fwdLoadI32At(bufT, 2*8+h)
	x3 := fwdLoadI32At(bufT, 3*8+h)
	x4 := fwdLoadI32At(bufT, 4*8+h)
	x5 := fwdLoadI32At(bufT, 5*8+h)
	x6 := fwdLoadI32At(bufT, 6*8+h)
	x7 := fwdLoadI32At(bufT, 7*8+h)
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
	fwdStoreI32At(coeff, 0*coeffStride+h, s1)
	fwdStoreI32At(coeff, 1*coeffStride+h, s6)
	fwdStoreI32At(coeff, 2*coeffStride+h, s3)
	fwdStoreI32At(coeff, 3*coeffStride+h, s4)
	fwdStoreI32At(coeff, 4*coeffStride+h, s5)
	fwdStoreI32At(coeff, 5*coeffStride+h, s2)
	fwdStoreI32At(coeff, 6*coeffStride+h, s7)
	fwdStoreI32At(coeff, 7*coeffStride+h, s0)
}

// forwardDCT8x8SIMD is the 8x8 DCT_DCT kernel (forwardDCT8x8PureGo). It
// assumes the residual is within the 8-bit range checked by the guarded binding.
func forwardDCT8x8SIMD(coeff []int32, coeffStride int, residual []int16, residualStride int) {
	fwd8DCTCore(coeff, coeffStride, residual, residualStride)
}

// forwardDCT8x8SIMDGuarded routes residuals outside the 8-bit range to the
// int64 scalar reference, like the asm bindings it replaces.
func forwardDCT8x8SIMDGuarded(coeff []int32, coeffStride int, residual []int16, residualStride int) {
	if !residualFitsMagnitude(residual, residualStride, 8, 8, 255) {
		forwardDCT8x8PureGo(coeff, coeffStride, residual, residualStride)
		return
	}
	forwardDCT8x8SIMD(coeff, coeffStride, residual, residualStride)
}

// forwardBlock8x8ADSTDCTSIMD is the ADST_DCT kernel: ADST on the columns, DCT
// on the rows (forwardBlock8x8ADSTDCTPureGo).
func forwardBlock8x8ADSTDCTSIMD(coeff []int32, coeffStride int, residual []int16, residualStride int, scratch []int32) {
	_ = scratch[63]
	fwd8ADSTDCTCore(coeff, coeffStride, residual, residualStride, scratch)
}

// forwardBlock8x8DCTADSTSIMD is the DCT_ADST kernel: DCT on the columns, ADST
// on the rows (forwardBlock8x8DCTADSTPureGo).
func forwardBlock8x8DCTADSTSIMD(coeff []int32, coeffStride int, residual []int16, residualStride int, scratch []int32) {
	_ = scratch[63]
	var buf, bufT [64]int32
	for g := 0; g < 8; g += fwdLanes {
		fwdColDCT8(buf[:], g, residual, residualStride)
	}
	fwdTranspose(bufT[:], buf[:], 8)
	for h := 0; h < 8; h += fwdLanes {
		fwdRowADST8(coeff, coeffStride, bufT[:], h)
	}
}

// forwardBlock8x8ADSTADSTSIMD is the ADST_ADST kernel (forwardBlock8x8ADSTADSTPureGo).
func forwardBlock8x8ADSTADSTSIMD(coeff []int32, coeffStride int, residual []int16, residualStride int, scratch []int32) {
	_ = scratch[63]
	var buf, bufT [64]int32
	for g := 0; g < 8; g += fwdLanes {
		fwdColADST8(buf[:], g, residual, residualStride)
	}
	fwdTranspose(bufT[:], buf[:], 8)
	for h := 0; h < 8; h += fwdLanes {
		fwdRowADST8(coeff, coeffStride, bufT[:], h)
	}
}

// forwardBlock8x8ADSTDCTSIMDGuarded and its siblings keep the 8-bit residual
// guard of the asm bindings they replace.
func forwardBlock8x8ADSTDCTSIMDGuarded(coeff []int32, coeffStride int, residual []int16, residualStride int, scratch []int32) {
	if !residualFitsMagnitude(residual, residualStride, 8, 8, 255) {
		forwardBlock8x8ADSTDCTPureGo(coeff, coeffStride, residual, residualStride, scratch)
		return
	}
	forwardBlock8x8ADSTDCTSIMD(coeff, coeffStride, residual, residualStride, scratch)
}

func forwardBlock8x8DCTADSTSIMDGuarded(coeff []int32, coeffStride int, residual []int16, residualStride int, scratch []int32) {
	if !residualFitsMagnitude(residual, residualStride, 8, 8, 255) {
		forwardBlock8x8DCTADSTPureGo(coeff, coeffStride, residual, residualStride, scratch)
		return
	}
	forwardBlock8x8DCTADSTSIMD(coeff, coeffStride, residual, residualStride, scratch)
}

func forwardBlock8x8ADSTADSTSIMDGuarded(coeff []int32, coeffStride int, residual []int16, residualStride int, scratch []int32) {
	if !residualFitsMagnitude(residual, residualStride, 8, 8, 255) {
		forwardBlock8x8ADSTADSTPureGo(coeff, coeffStride, residual, residualStride, scratch)
		return
	}
	forwardBlock8x8ADSTADSTSIMD(coeff, coeffStride, residual, residualStride, scratch)
}
