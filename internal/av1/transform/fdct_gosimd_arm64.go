// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package transform

import (
	"simd/archsimd"
	"unsafe"
)

// fdct16NEONCtx carries the kernel arguments; offsets are mirrored by
// #define in fdct16_neon_arm64.s. Buf points at caller-owned 16x16 int32
// scratch for the column-pass output.
type fdct16NEONCtx struct {
	In        unsafe.Pointer
	InStride  int64
	Out       unsafe.Pointer
	OutStride int64
	Buf       unsafe.Pointer
}

//go:noescape
func fdct16x16NEONAsm(ctx *fdct16NEONCtx)

func forwardDCT16x16NEON(coeff []int32, coeffStride int, residual []int16, residualStride int) {
	var buf [256]int32
	ctx := fdct16NEONCtx{
		In:        unsafe.Pointer(&residual[0]),
		InStride:  int64(residualStride),
		Out:       unsafe.Pointer(&coeff[0]),
		OutStride: int64(coeffStride),
		Buf:       unsafe.Pointer(&buf[0]),
	}
	fdct16x16NEONAsm(&ctx)
}

func forwardDCT16x16NEONGuarded(coeff []int32, coeffStride int, residual []int16, residualStride int) {
	if !residualFitsMagnitude(residual, residualStride, 16, 16, 255) {
		forwardDCT16x16PureGo(coeff, coeffStride, residual, residualStride)
		return
	}
	forwardDCT16x16NEON(coeff, coeffStride, residual, residualStride)
}

// fdct32NEONCtx carries the kernel arguments; offsets are mirrored by
// #define in fdct32_neon_arm64.s. Buf points at caller-owned 32x32 int32
// scratch for the column-pass output and Spill at sixteen vectors of
// stage-1 difference staging.
type fdct32NEONCtx struct {
	In        unsafe.Pointer
	InStride  int64
	Out       unsafe.Pointer
	OutStride int64
	Buf       unsafe.Pointer
	Spill     unsafe.Pointer
}

//go:noescape
func fdct32x32NEONAsm(ctx *fdct32NEONCtx)

func forwardDCT32x32NEON(coeff []int32, coeffStride int, residual []int16, residualStride int) {
	if !residualFitsMagnitude(residual, residualStride, 32, 32, 255) {
		forwardDCT32x32PureGo(coeff, coeffStride, residual, residualStride)
		return
	}
	forwardDCT32x32NEONRaw(coeff, coeffStride, residual, residualStride)
}

func forwardDCT32x32NEONRaw(coeff []int32, coeffStride int, residual []int16, residualStride int) {
	var buf [1024]int32
	var spill [64]int32
	ctx := fdct32NEONCtx{
		In:        unsafe.Pointer(&residual[0]),
		InStride:  int64(residualStride),
		Out:       unsafe.Pointer(&coeff[0]),
		OutStride: int64(coeffStride),
		Buf:       unsafe.Pointer(&buf[0]),
		Spill:     unsafe.Pointer(&spill[0]),
	}
	fdct32x32NEONAsm(&ctx)
}

// The 4x4 and 8x8 dispatch slots are Go SIMD; 16x16 and 32x32 keep their
// guarded NEON kernels until they are ported.
var forwardDCT4x4Impl = forwardDCT4x4SIMD
var forwardDCT8x8Impl = forwardDCT8x8SIMDGuarded
var forwardDCT16x16Impl = forwardDCT16x16NEONGuarded
var forwardDCT32x32Impl = forwardDCT32x32NEON
var forwardDCT4x4Trusted8BitImpl = forwardDCT4x4SIMDCore
var forwardDCT8x8Trusted8BitImpl = forwardDCT8x8SIMD
var forwardDCT16x16Trusted8BitImpl = forwardDCT16x16NEON
var forwardDCT32x32Trusted8BitImpl = forwardDCT32x32NEONRaw

// Pre-broadcast fdct4 twiddle vectors, kept as package-level (rodata) arrays so
// each loads with a single instruction instead of a per-call stack fill. This
// mirrors how the hand-written NEON kernels keep their coefficients in a loaded
// constant vector rather than re-broadcasting a scalar every use.
var (
	fdct4W32   = [4]int32{fwdCospi13[32], fwdCospi13[32], fwdCospi13[32], fwdCospi13[32]}
	fdct4Wn32  = [4]int32{-fwdCospi13[32], -fwdCospi13[32], -fwdCospi13[32], -fwdCospi13[32]}
	fdct4W48   = [4]int32{fwdCospi13[48], fwdCospi13[48], fwdCospi13[48], fwdCospi13[48]}
	fdct4W16   = [4]int32{fwdCospi13[16], fwdCospi13[16], fwdCospi13[16], fwdCospi13[16]}
	fdct4Wn16  = [4]int32{-fwdCospi13[16], -fwdCospi13[16], -fwdCospi13[16], -fwdCospi13[16]}
	fdct4Round = [4]int32{1 << 12, 1 << 12, 1 << 12, 1 << 12}
)

func forwardDCT4x4SIMD(coeff []int32, coeffStride int, residual []int16, residualStride int) {
	// One straight-line pass: load 4 rows of int16 residual widened to int32
	// (<<2), fdct4 column butterfly, in-register 4x4 int32 transpose, fdct4 row
	// butterfly, raw-pointer store. shift[1]=shift[2]=0 for 4x4 so there is no
	// inter-pass round-shift; everything stays in Int32x4 registers.
	// The first-pass output magnitude is at most 46341 and the largest
	// second-pass accumulator at most 2147446036 for |residual| <= 2048. Larger
	// public int16 inputs use the scalar path, whose multiply-accumulates are
	// int64.
	if !residualFitsMagnitude(residual, residualStride, 4, 4, 2048) {
		forwardDCT4x4PureGo(coeff, coeffStride, residual, residualStride)
		return
	}
	forwardDCT4x4SIMDCore(coeff, coeffStride, residual, residualStride)
}

func forwardDCT4x4SIMDCore(coeff []int32, coeffStride int, residual []int16, residualStride int) {
	loadRow := func(row int) archsimd.Int32x4 {
		// Load only the four samples used by this kernel. A full Int16x8 load
		// from the final row reads past an exact 4x4 residual buffer.
		v, _ := archsimd.LoadInt16x8Part(residual[row*residualStride : row*residualStride+4])
		return v.ExtendLo4ToInt32().ShiftAllLeft(2)
	}
	r0 := loadRow(0)
	r1 := loadRow(1)
	r2 := loadRow(2)
	r3 := loadRow(3)

	// Load the pre-broadcast fdct4 twiddles (single VLD1/FMOVQ each, straight
	// from rodata; shared by both passes). This is far cheaper than
	// BroadcastInt32x4 of a scalar, which materializes the value in a GPR and
	// does VMOV+VDUP (three instructions plus a preserving move) per twiddle.
	w32 := archsimd.LoadInt32x4Array(&fdct4W32)
	wn32 := archsimd.LoadInt32x4Array(&fdct4Wn32)
	w48 := archsimd.LoadInt32x4Array(&fdct4W48)
	w16 := archsimd.LoadInt32x4Array(&fdct4W16)
	wn16 := archsimd.LoadInt32x4Array(&fdct4Wn16)
	round := archsimd.LoadInt32x4Array(&fdct4Round)

	// --- Column pass (inlined fdct4; output in (t0,t2,t1,t3) order) ---
	// Each half-butterfly is (a*w0 + b*w1 + round) >> 13, built with fused
	// multiply-adds (VMLA) so there is one VMUL + one VMLA per lane instead of
	// two VMUL + two VADD.
	var c0, c1, c2, c3 archsimd.Int32x4
	{
		s0 := r0.Add(r3)
		s1 := r1.Add(r2)
		s2 := r1.Sub(r2)
		s3 := r0.Sub(r3)
		c0 = s1.MulAdd(w32, s0.MulAdd(w32, round)).ShiftAllRight(13)
		c2 = s0.MulAdd(w32, s1.MulAdd(wn32, round)).ShiftAllRight(13)
		c1 = s3.MulAdd(w16, s2.MulAdd(w48, round)).ShiftAllRight(13)
		c3 = s2.MulAdd(wn16, s3.MulAdd(w48, round)).ShiftAllRight(13)
	}

	// In-register 4x4 int32 transpose.
	e0 := c0.InterleaveLo(c1)
	e1 := c0.InterleaveHi(c1)
	e2 := c2.InterleaveLo(c3)
	e3 := c2.InterleaveHi(c3)
	t0 := fdctInt64AsInt32(fdctInt32AsInt64(e0).InterleaveLo(fdctInt32AsInt64(e2)))
	t1 := fdctInt64AsInt32(fdctInt32AsInt64(e0).InterleaveHi(fdctInt32AsInt64(e2)))
	t2 := fdctInt64AsInt32(fdctInt32AsInt64(e1).InterleaveLo(fdctInt32AsInt64(e3)))
	t3 := fdctInt64AsInt32(fdctInt32AsInt64(e1).InterleaveHi(fdctInt32AsInt64(e3)))

	// --- Row pass (inlined fdct4) ---
	var o0, o1, o2, o3 archsimd.Int32x4
	{
		s0 := t0.Add(t3)
		s1 := t1.Add(t2)
		s2 := t1.Sub(t2)
		s3 := t0.Sub(t3)
		o0 = s1.MulAdd(w32, s0.MulAdd(w32, round)).ShiftAllRight(13)
		o2 = s0.MulAdd(w32, s1.MulAdd(wn32, round)).ShiftAllRight(13)
		o1 = s3.MulAdd(w16, s2.MulAdd(w48, round)).ShiftAllRight(13)
		o3 = s2.MulAdd(wn16, s3.MulAdd(w48, round)).ShiftAllRight(13)
	}

	obase := unsafe.Pointer(unsafe.SliceData(coeff))
	ostep := uintptr(coeffStride) * 4
	o0.StoreArray((*[4]int32)(obase))
	o1.StoreArray((*[4]int32)(unsafe.Add(obase, ostep*1)))
	o2.StoreArray((*[4]int32)(unsafe.Add(obase, ostep*2)))
	o3.StoreArray((*[4]int32)(unsafe.Add(obase, ostep*3)))
}

func fdctInt32AsInt64(v archsimd.Int32x4) archsimd.Int64x2 {
	return v.ToBits().ReshapeToUint64s().BitsToInt64()
}

func fdctInt64AsInt32(v archsimd.Int64x2) archsimd.Int32x4 {
	return v.ToBits().ReshapeToUint32s().BitsToInt32()
}
