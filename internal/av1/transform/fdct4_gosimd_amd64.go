// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && amd64 && !purego

package transform

import (
	"simd/archsimd"
	"unsafe"
)

// Pre-broadcast fdct4 twiddles, kept as package-level rodata so each loads with
// one instruction (the same tables the arm64 4x4 kernel uses).
var (
	fdct4W32   = [4]int32{fwdCospi13[32], fwdCospi13[32], fwdCospi13[32], fwdCospi13[32]}
	fdct4Wn32  = [4]int32{-fwdCospi13[32], -fwdCospi13[32], -fwdCospi13[32], -fwdCospi13[32]}
	fdct4W48   = [4]int32{fwdCospi13[48], fwdCospi13[48], fwdCospi13[48], fwdCospi13[48]}
	fdct4W16   = [4]int32{fwdCospi13[16], fwdCospi13[16], fwdCospi13[16], fwdCospi13[16]}
	fdct4Wn16  = [4]int32{-fwdCospi13[16], -fwdCospi13[16], -fwdCospi13[16], -fwdCospi13[16]}
	fdct4Round = [4]int32{1 << 12, 1 << 12, 1 << 12, 1 << 12}
)

// forwardDCT4x4SIMD is the 4x4 DCT_DCT kernel. The first-pass output magnitude is
// at most 46341 and the largest second-pass accumulator at most 2147446036 for
// |residual| <= 2048; larger public int16 inputs use the scalar path.
func forwardDCT4x4SIMD(coeff []int32, coeffStride int, residual []int16, residualStride int) {
	if !residualFitsMagnitude(residual, residualStride, 4, 4, 2048) {
		forwardDCT4x4PureGo(coeff, coeffStride, residual, residualStride)
		return
	}
	forwardDCT4x4SIMDCore(coeff, coeffStride, residual, residualStride)
}

// forwardDCT4x4SIMDCore runs one straight-line pass: widen four residual rows,
// the fdct4 column butterfly, an in-register 4x4 transpose, the fdct4 row
// butterfly, and a raw-pointer store. shift[1]=shift[2]=0 for 4x4.
func forwardDCT4x4SIMDCore(coeff []int32, coeffStride int, residual []int16, residualStride int) {
	loadRow := func(row int) archsimd.Int32x4 {
		// Only the four samples of this row: a full 8-lane load would read past an
		// exact 4x4 residual buffer.
		v, _ := archsimd.LoadInt16x8Part(residual[row*residualStride : row*residualStride+4])
		return v.ExtendLo4ToInt32().ShiftAllLeft(2)
	}
	r0 := loadRow(0)
	r1 := loadRow(1)
	r2 := loadRow(2)
	r3 := loadRow(3)

	w32 := archsimd.LoadInt32x4Array(&fdct4W32)
	wn32 := archsimd.LoadInt32x4Array(&fdct4Wn32)
	w48 := archsimd.LoadInt32x4Array(&fdct4W48)
	w16 := archsimd.LoadInt32x4Array(&fdct4W16)
	wn16 := archsimd.LoadInt32x4Array(&fdct4Wn16)
	round := archsimd.LoadInt32x4Array(&fdct4Round)

	// Column pass: each half-butterfly is (a*w0 + b*w1 + round) >> 13.
	var c0, c1, c2, c3 archsimd.Int32x4
	{
		s0 := r0.Add(r3)
		s1 := r1.Add(r2)
		s2 := r1.Sub(r2)
		s3 := r0.Sub(r3)
		c0 = s1.Mul(w32).Add(s0.Mul(w32).Add(round)).ShiftAllRight(13)
		c2 = s0.Mul(w32).Add(s1.Mul(wn32).Add(round)).ShiftAllRight(13)
		c1 = s3.Mul(w16).Add(s2.Mul(w48).Add(round)).ShiftAllRight(13)
		c3 = s2.Mul(wn16).Add(s3.Mul(w48).Add(round)).ShiftAllRight(13)
	}

	// In-register 4x4 int32 transpose.
	e0 := c0.InterleaveLo(c1)
	e1 := c0.InterleaveHi(c1)
	e2 := c2.InterleaveLo(c3)
	e3 := c2.InterleaveHi(c3)
	t0 := fdctInt64AsInt32Amd(fdctInt32AsInt64Amd(e0).InterleaveLo(fdctInt32AsInt64Amd(e2)))
	t1 := fdctInt64AsInt32Amd(fdctInt32AsInt64Amd(e0).InterleaveHi(fdctInt32AsInt64Amd(e2)))
	t2 := fdctInt64AsInt32Amd(fdctInt32AsInt64Amd(e1).InterleaveLo(fdctInt32AsInt64Amd(e3)))
	t3 := fdctInt64AsInt32Amd(fdctInt32AsInt64Amd(e1).InterleaveHi(fdctInt32AsInt64Amd(e3)))

	// Row pass.
	var o0, o1, o2, o3 archsimd.Int32x4
	{
		s0 := t0.Add(t3)
		s1 := t1.Add(t2)
		s2 := t1.Sub(t2)
		s3 := t0.Sub(t3)
		o0 = s1.Mul(w32).Add(s0.Mul(w32).Add(round)).ShiftAllRight(13)
		o2 = s0.Mul(w32).Add(s1.Mul(wn32).Add(round)).ShiftAllRight(13)
		o1 = s3.Mul(w16).Add(s2.Mul(w48).Add(round)).ShiftAllRight(13)
		o3 = s2.Mul(wn16).Add(s3.Mul(w48).Add(round)).ShiftAllRight(13)
	}

	obase := unsafe.Pointer(unsafe.SliceData(coeff))
	ostep := uintptr(coeffStride) * 4
	o0.StoreArray((*[4]int32)(obase))
	o1.StoreArray((*[4]int32)(unsafe.Add(obase, ostep*1)))
	o2.StoreArray((*[4]int32)(unsafe.Add(obase, ostep*2)))
	o3.StoreArray((*[4]int32)(unsafe.Add(obase, ostep*3)))
}

func fdctInt32AsInt64Amd(v archsimd.Int32x4) archsimd.Int64x2 {
	return v.AsInt64x2()
}

func fdctInt64AsInt32Amd(v archsimd.Int64x2) archsimd.Int32x4 {
	return v.AsInt32x4()
}
