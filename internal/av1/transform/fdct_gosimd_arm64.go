// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package transform

import (
	"simd/archsimd"
	"unsafe"
)

// fdct8x8NEONCtx carries the kernel arguments; offsets are mirrored by
// #define in fdct_neon_arm64.s. Strides are in elements.
type fdct8x8NEONCtx struct {
	In        unsafe.Pointer
	InStride  int64
	Out       unsafe.Pointer
	OutStride int64
}

//go:noescape
func fdct8x8NEONAsm(ctx *fdct8x8NEONCtx)

//go:noescape
func fdct4x4NEONAsm(ctx *fdct8x8NEONCtx)

func forwardDCT8x8NEON(coeff []int32, coeffStride int, residual []int16, residualStride int) {
	ctx := fdct8x8NEONCtx{
		In:        unsafe.Pointer(&residual[0]),
		InStride:  int64(residualStride),
		Out:       unsafe.Pointer(&coeff[0]),
		OutStride: int64(coeffStride),
	}
	fdct8x8NEONAsm(&ctx)
}

func forwardDCT4x4NEON(coeff []int32, coeffStride int, residual []int16, residualStride int) {
	ctx := fdct8x8NEONCtx{
		In:        unsafe.Pointer(&residual[0]),
		InStride:  int64(residualStride),
		Out:       unsafe.Pointer(&coeff[0]),
		OutStride: int64(coeffStride),
	}
	fdct4x4NEONAsm(&ctx)
}

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

// forwardDCT*Impl are the FDCT dispatch slots under GOEXPERIMENT=simd. The
// 4/8/16 kernels use Go-native SIMD for the column pass. The 32x32 slot stays
// on the existing NEON asm so the experiment build does not regress that size.
var forwardDCT4x4Impl = forwardDCT4x4SIMD
var forwardDCT8x8Impl = forwardDCT8x8SIMD
var forwardDCT16x16Impl = forwardDCT16x16SIMD
var forwardDCT32x32Impl = forwardDCT32x32NEON

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

// bcast16 fills a [8]int16 with v (compile-time constant folded).
func bcast16(v int32) [8]int16 {
	x := int16(v)
	return [8]int16{x, x, x, x, x, x, x, x}
}

// Pre-broadcast fdct8 twiddle vectors (int16), loaded with a single VLD1 in the
// 8x8 kernel rather than re-broadcasting a scalar (VMOV+VDUP) each use.
var (
	fdct8K32  = bcast16(fwdCospi13[32])
	fdct8Kn32 = bcast16(-fwdCospi13[32])
	fdct8K48  = bcast16(fwdCospi13[48])
	fdct8K16  = bcast16(fwdCospi13[16])
	fdct8Kn16 = bcast16(-fwdCospi13[16])
	fdct8K56  = bcast16(fwdCospi13[56])
	fdct8K8   = bcast16(fwdCospi13[8])
	fdct8Kn8  = bcast16(-fwdCospi13[8])
	fdct8K24  = bcast16(fwdCospi13[24])
	fdct8K40  = bcast16(fwdCospi13[40])
	fdct8Kn40 = bcast16(-fwdCospi13[40])
)


func forwardDCT4x4SIMD(coeff []int32, coeffStride int, residual []int16, residualStride int) {
	// One straight-line pass: load 4 rows of int16 residual widened to int32
	// (<<2), fdct4 column butterfly, in-register 4x4 int32 transpose, fdct4 row
	// butterfly, raw-pointer store. shift[1]=shift[2]=0 for 4x4 so there is no
	// inter-pass round-shift; everything stays in Int32x4 registers.
	rbase := unsafe.Pointer(unsafe.SliceData(residual))
	rstep := uintptr(residualStride) * 2
	r0 := archsimd.LoadInt16x8Array((*[8]int16)(rbase)).ExtendLo4ToInt32().ShiftAllLeftConst(2)
	r1 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(rbase, rstep*1))).ExtendLo4ToInt32().ShiftAllLeftConst(2)
	r2 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(rbase, rstep*2))).ExtendLo4ToInt32().ShiftAllLeftConst(2)
	r3 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(rbase, rstep*3))).ExtendLo4ToInt32().ShiftAllLeftConst(2)

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
		c0 = s1.MulAdd(w32, s0.MulAdd(w32, round)).ShiftAllRightConst(13)
		c2 = s0.MulAdd(w32, s1.MulAdd(wn32, round)).ShiftAllRightConst(13)
		c1 = s3.MulAdd(w16, s2.MulAdd(w48, round)).ShiftAllRightConst(13)
		c3 = s2.MulAdd(wn16, s3.MulAdd(w48, round)).ShiftAllRightConst(13)
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
		o0 = s1.MulAdd(w32, s0.MulAdd(w32, round)).ShiftAllRightConst(13)
		o2 = s0.MulAdd(w32, s1.MulAdd(wn32, round)).ShiftAllRightConst(13)
		o1 = s3.MulAdd(w16, s2.MulAdd(w48, round)).ShiftAllRightConst(13)
		o3 = s2.MulAdd(wn16, s3.MulAdd(w48, round)).ShiftAllRightConst(13)
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

func forwardDCT8x8SIMD(coeff []int32, coeffStride int, residual []int16, residualStride int) {
	// Everything below is one straight-line function: the fdct8 butterfly (both
	// passes), the transpose, round-shift, and store are written inline so all
	// intermediate Int16x8 values stay register-resident with no stack arrays
	// and no per-pass CALL/ABI spill. Raw unsafe pointers walk the input/output
	// so there is no per-row bounds check.
	rbase := unsafe.Pointer(unsafe.SliceData(residual))
	rstep := uintptr(residualStride) * 2
	l0 := archsimd.LoadInt16x8Array((*[8]int16)(rbase)).ShiftAllLeftConst(2)
	l1 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(rbase, rstep))).ShiftAllLeftConst(2)
	l2 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(rbase, rstep*2))).ShiftAllLeftConst(2)
	l3 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(rbase, rstep*3))).ShiftAllLeftConst(2)
	l4 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(rbase, rstep*4))).ShiftAllLeftConst(2)
	l5 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(rbase, rstep*5))).ShiftAllLeftConst(2)
	l6 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(rbase, rstep*6))).ShiftAllLeftConst(2)
	l7 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(rbase, rstep*7))).ShiftAllLeftConst(2)

	// Load loop-invariant twiddle vectors from rodata (single VLD1 each; used by
	// both passes) instead of re-broadcasting scalars.
	k32 := archsimd.LoadInt16x8Array(&fdct8K32)
	kn32 := archsimd.LoadInt16x8Array(&fdct8Kn32)
	k48 := archsimd.LoadInt16x8Array(&fdct8K48)
	k16 := archsimd.LoadInt16x8Array(&fdct8K16)
	kn16 := archsimd.LoadInt16x8Array(&fdct8Kn16)
	k56 := archsimd.LoadInt16x8Array(&fdct8K56)
	k8 := archsimd.LoadInt16x8Array(&fdct8K8)
	kn8 := archsimd.LoadInt16x8Array(&fdct8Kn8)
	k24 := archsimd.LoadInt16x8Array(&fdct8K24)
	k40 := archsimd.LoadInt16x8Array(&fdct8K40)
	kn40 := archsimd.LoadInt16x8Array(&fdct8Kn40)

	// --- Column pass (fully inlined butterfly) ---
	var c0, c1, c2, c3, c4, c5, c6, c7 archsimd.Int16x8
	{
		b0 := l0.AddSaturated(l7)
		b1 := l1.AddSaturated(l6)
		b2 := l2.AddSaturated(l5)
		b3 := l3.AddSaturated(l4)
		b4 := l3.SubSaturated(l4)
		b5 := l2.SubSaturated(l5)
		b6 := l1.SubSaturated(l6)
		b7 := l0.SubSaturated(l7)

		s0 := b0.AddSaturated(b3)
		s1 := b1.AddSaturated(b2)
		s2 := b1.SubSaturated(b2)
		s3 := b0.SubSaturated(b3)
		s4 := b4
		s5 := fdctHalfBtf16Vec(kn32, b5, k32, b6, 13)
		s6 := fdctHalfBtf16Vec(k32, b6, k32, b5, 13)
		s7 := b7

		p0 := fdctHalfBtf16Vec(k32, s0, k32, s1, 13)
		p1 := fdctHalfBtf16Vec(kn32, s1, k32, s0, 13)
		p2 := fdctHalfBtf16Vec(k48, s2, k16, s3, 13)
		p3 := fdctHalfBtf16Vec(k48, s3, kn16, s2, 13)
		p4 := s4.AddSaturated(s5)
		p5 := s4.SubSaturated(s5)
		p6 := s7.SubSaturated(s6)
		p7 := s7.AddSaturated(s6)

		c0 = fdctRoundShift1Int16Native(p0)
		c4 = fdctRoundShift1Int16Native(p1)
		c2 = fdctRoundShift1Int16Native(p2)
		c6 = fdctRoundShift1Int16Native(p3)
		c1 = fdctRoundShift1Int16Native(fdctHalfBtf16Vec(k56, p4, k8, p7, 13))
		c5 = fdctRoundShift1Int16Native(fdctHalfBtf16Vec(k24, p5, k40, p6, 13))
		c3 = fdctRoundShift1Int16Native(fdctHalfBtf16Vec(k24, p6, kn40, p5, 13))
		c7 = fdctRoundShift1Int16Native(fdctHalfBtf16Vec(k56, p7, kn8, p4, 13))
	}

	// --- Transpose (out-of-line: isolates its temporaries from the butterfly
	// passes' register pressure) ---
	t0, t1, t2, t3, t4, t5, t6, t7 := fdctTranspose8x8Int16(c0, c1, c2, c3, c4, c5, c6, c7)

	// --- Row pass (fully inlined butterfly) ---
	var o0, o1, o2, o3, o4, o5, o6, o7 archsimd.Int16x8
	{
		b0 := t0.AddSaturated(t7)
		b1 := t1.AddSaturated(t6)
		b2 := t2.AddSaturated(t5)
		b3 := t3.AddSaturated(t4)
		b4 := t3.SubSaturated(t4)
		b5 := t2.SubSaturated(t5)
		b6 := t1.SubSaturated(t6)
		b7 := t0.SubSaturated(t7)

		s0 := b0.AddSaturated(b3)
		s1 := b1.AddSaturated(b2)
		s2 := b1.SubSaturated(b2)
		s3 := b0.SubSaturated(b3)
		s4 := b4
		s5 := fdctHalfBtf16Vec(kn32, b5, k32, b6, 13)
		s6 := fdctHalfBtf16Vec(k32, b6, k32, b5, 13)
		s7 := b7

		p0 := fdctHalfBtf16Vec(k32, s0, k32, s1, 13)
		p1 := fdctHalfBtf16Vec(kn32, s1, k32, s0, 13)
		p2 := fdctHalfBtf16Vec(k48, s2, k16, s3, 13)
		p3 := fdctHalfBtf16Vec(k48, s3, kn16, s2, 13)
		p4 := s4.AddSaturated(s5)
		p5 := s4.SubSaturated(s5)
		p6 := s7.SubSaturated(s6)
		p7 := s7.AddSaturated(s6)

		o0 = p0
		o4 = p1
		o2 = p2
		o6 = p3
		o1 = fdctHalfBtf16Vec(k56, p4, k8, p7, 13)
		o5 = fdctHalfBtf16Vec(k24, p5, k40, p6, 13)
		o3 = fdctHalfBtf16Vec(k24, p6, kn40, p5, 13)
		o7 = fdctHalfBtf16Vec(k56, p7, kn8, p4, 13)
	}

	// --- Store (int16 -> int32 expansion, raw pointer walk) ---
	obase := unsafe.Pointer(unsafe.SliceData(coeff))
	ostep := uintptr(coeffStride) * 4
	fdctStoreInt16x8ToInt32(obase, o0)
	fdctStoreInt16x8ToInt32(unsafe.Add(obase, ostep), o1)
	fdctStoreInt16x8ToInt32(unsafe.Add(obase, ostep*2), o2)
	fdctStoreInt16x8ToInt32(unsafe.Add(obase, ostep*3), o3)
	fdctStoreInt16x8ToInt32(unsafe.Add(obase, ostep*4), o4)
	fdctStoreInt16x8ToInt32(unsafe.Add(obase, ostep*5), o5)
	fdctStoreInt16x8ToInt32(unsafe.Add(obase, ostep*6), o6)
	fdctStoreInt16x8ToInt32(unsafe.Add(obase, ostep*7), o7)
}

// fdctHalfBtf16Vec is the widening butterfly with pre-broadcast twiddle
// vectors; kept tiny so it inlines everywhere.
func fdctHalfBtf16Vec(k0 archsimd.Int16x8, in0 archsimd.Int16x8, k1 archsimd.Int16x8, in1 archsimd.Int16x8, cosBit uint8) archsimd.Int16x8 {
	lo := in0.MulWidenLo(k0).MulWidenLoAdd(in1, k1)
	hi := in0.MulWidenHi(k0).MulWidenHiAdd(in1, k1)
	return lo.ShiftRightRoundNarrow(cosBit).ShiftRightRoundNarrowHi(hi, cosBit)
}


func forwardDCT16x16SIMD(coeff []int32, coeffStride int, residual []int16, residualStride int) {
	// rowsLo/rowsHi hold the column-pass output (32 Int16x8 = the whole 16x16
	// tile). This is the one unavoidable scratch: the 16-point butterfly needs
	// all 16 rows live for a column group, and 32 vectors exceed the 32 V-regs,
	// so the NEON kernel likewise spills its column output to a buf. Everything
	// else threads through registers via value-returning helpers.
	var rowsLo, rowsHi [16]archsimd.Int16x8
	fdct16ColumnPass8SIMD16(&rowsLo, 0, residual, residualStride)
	fdct16ColumnPass8SIMD16(&rowsHi, 8, residual, residualStride)

	obase := unsafe.Pointer(unsafe.SliceData(coeff))
	ostride := uintptr(coeffStride) * 4
	for row := 0; row < 16; row += 8 {
		// Transpose the two 8x8 blocks of this 8-row strip into 16 columns.
		t0, t1, t2, t3, t4, t5, t6, t7 := fdctTranspose8x8Int16(
			rowsLo[row+0], rowsLo[row+1], rowsLo[row+2], rowsLo[row+3],
			rowsLo[row+4], rowsLo[row+5], rowsLo[row+6], rowsLo[row+7],
		)
		t8, t9, t10, t11, t12, t13, t14, t15 := fdctTranspose8x8Int16(
			rowsHi[row+0], rowsHi[row+1], rowsHi[row+2], rowsHi[row+3],
			rowsHi[row+4], rowsHi[row+5], rowsHi[row+6], rowsHi[row+7],
		)
		o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13, o14, o15 := fdct16Butterfly12(
			t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15)

		base := unsafe.Add(obase, uintptr(row)*4)
		fdctStoreInt16x8ToInt32(unsafe.Add(base, ostride*0), o0)
		fdctStoreInt16x8ToInt32(unsafe.Add(base, ostride*1), o1)
		fdctStoreInt16x8ToInt32(unsafe.Add(base, ostride*2), o2)
		fdctStoreInt16x8ToInt32(unsafe.Add(base, ostride*3), o3)
		fdctStoreInt16x8ToInt32(unsafe.Add(base, ostride*4), o4)
		fdctStoreInt16x8ToInt32(unsafe.Add(base, ostride*5), o5)
		fdctStoreInt16x8ToInt32(unsafe.Add(base, ostride*6), o6)
		fdctStoreInt16x8ToInt32(unsafe.Add(base, ostride*7), o7)
		fdctStoreInt16x8ToInt32(unsafe.Add(base, ostride*8), o8)
		fdctStoreInt16x8ToInt32(unsafe.Add(base, ostride*9), o9)
		fdctStoreInt16x8ToInt32(unsafe.Add(base, ostride*10), o10)
		fdctStoreInt16x8ToInt32(unsafe.Add(base, ostride*11), o11)
		fdctStoreInt16x8ToInt32(unsafe.Add(base, ostride*12), o12)
		fdctStoreInt16x8ToInt32(unsafe.Add(base, ostride*13), o13)
		fdctStoreInt16x8ToInt32(unsafe.Add(base, ostride*14), o14)
		fdctStoreInt16x8ToInt32(unsafe.Add(base, ostride*15), o15)
	}
}

func fdctStoreInt16x8ToInt32(base unsafe.Pointer, v archsimd.Int16x8) {
	lo := v.ExtendLo4ToInt32()
	hi := v.HiToLo().ExtendLo4ToInt32()
	lo.StoreArray((*[4]int32)(base))
	hi.StoreArray((*[4]int32)(unsafe.Add(base, 16)))
}

func fdctInt16AsInt32(v archsimd.Int16x8) archsimd.Int32x4 {
	return v.ToBits().ReshapeToUint32s().BitsToInt32()
}

func fdctInt32AsInt16(v archsimd.Int32x4) archsimd.Int16x8 {
	return v.ToBits().ReshapeToUint16s().BitsToInt16()
}

func fdctInt16AsInt64(v archsimd.Int16x8) archsimd.Int64x2 {
	return v.ToBits().ReshapeToUint64s().BitsToInt64()
}

func fdctInt64AsInt16(v archsimd.Int64x2) archsimd.Int16x8 {
	return v.ToBits().ReshapeToUint16s().BitsToInt16()
}

func fdctTranspose8x8Int16(r0, r1, r2, r3, r4, r5, r6, r7 archsimd.Int16x8) (
	archsimd.Int16x8, archsimd.Int16x8, archsimd.Int16x8, archsimd.Int16x8,
	archsimd.Int16x8, archsimd.Int16x8, archsimd.Int16x8, archsimd.Int16x8,
) {
	a0 := r0.InterleaveLo(r1)
	a1 := r0.InterleaveHi(r1)
	a2 := r2.InterleaveLo(r3)
	a3 := r2.InterleaveHi(r3)
	a4 := r4.InterleaveLo(r5)
	a5 := r4.InterleaveHi(r5)
	a6 := r6.InterleaveLo(r7)
	a7 := r6.InterleaveHi(r7)

	b0 := fdctInt32AsInt16(fdctInt16AsInt32(a0).InterleaveLo(fdctInt16AsInt32(a2)))
	b1 := fdctInt32AsInt16(fdctInt16AsInt32(a0).InterleaveHi(fdctInt16AsInt32(a2)))
	b2 := fdctInt32AsInt16(fdctInt16AsInt32(a1).InterleaveLo(fdctInt16AsInt32(a3)))
	b3 := fdctInt32AsInt16(fdctInt16AsInt32(a1).InterleaveHi(fdctInt16AsInt32(a3)))
	b4 := fdctInt32AsInt16(fdctInt16AsInt32(a4).InterleaveLo(fdctInt16AsInt32(a6)))
	b5 := fdctInt32AsInt16(fdctInt16AsInt32(a4).InterleaveHi(fdctInt16AsInt32(a6)))
	b6 := fdctInt32AsInt16(fdctInt16AsInt32(a5).InterleaveLo(fdctInt16AsInt32(a7)))
	b7 := fdctInt32AsInt16(fdctInt16AsInt32(a5).InterleaveHi(fdctInt16AsInt32(a7)))

	o0 := fdctInt64AsInt16(fdctInt16AsInt64(b0).InterleaveLo(fdctInt16AsInt64(b4)))
	o1 := fdctInt64AsInt16(fdctInt16AsInt64(b0).InterleaveHi(fdctInt16AsInt64(b4)))
	o2 := fdctInt64AsInt16(fdctInt16AsInt64(b1).InterleaveLo(fdctInt16AsInt64(b5)))
	o3 := fdctInt64AsInt16(fdctInt16AsInt64(b1).InterleaveHi(fdctInt16AsInt64(b5)))
	o4 := fdctInt64AsInt16(fdctInt16AsInt64(b2).InterleaveLo(fdctInt16AsInt64(b6)))
	o5 := fdctInt64AsInt16(fdctInt16AsInt64(b2).InterleaveHi(fdctInt16AsInt64(b6)))
	o6 := fdctInt64AsInt16(fdctInt16AsInt64(b3).InterleaveLo(fdctInt16AsInt64(b7)))
	o7 := fdctInt64AsInt16(fdctInt16AsInt64(b3).InterleaveHi(fdctInt16AsInt64(b7)))
	return o0, o1, o2, o3, o4, o5, o6, o7
}

func fdct16ColumnPass8SIMD16(output *[16]archsimd.Int16x8, col int, residual []int16, residualStride int) {
	rbase := unsafe.Add(unsafe.Pointer(unsafe.SliceData(residual)), uintptr(col)*2)
	rstep := uintptr(residualStride) * 2
	l0 := archsimd.LoadInt16x8Array((*[8]int16)(rbase)).ShiftAllLeftConst(2)
	l1 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(rbase, rstep*1))).ShiftAllLeftConst(2)
	l2 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(rbase, rstep*2))).ShiftAllLeftConst(2)
	l3 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(rbase, rstep*3))).ShiftAllLeftConst(2)
	l4 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(rbase, rstep*4))).ShiftAllLeftConst(2)
	l5 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(rbase, rstep*5))).ShiftAllLeftConst(2)
	l6 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(rbase, rstep*6))).ShiftAllLeftConst(2)
	l7 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(rbase, rstep*7))).ShiftAllLeftConst(2)
	l8 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(rbase, rstep*8))).ShiftAllLeftConst(2)
	l9 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(rbase, rstep*9))).ShiftAllLeftConst(2)
	l10 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(rbase, rstep*10))).ShiftAllLeftConst(2)
	l11 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(rbase, rstep*11))).ShiftAllLeftConst(2)
	l12 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(rbase, rstep*12))).ShiftAllLeftConst(2)
	l13 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(rbase, rstep*13))).ShiftAllLeftConst(2)
	l14 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(rbase, rstep*14))).ShiftAllLeftConst(2)
	l15 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(rbase, rstep*15))).ShiftAllLeftConst(2)

	o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13, o14, o15 := fdct16Butterfly13(
		l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14, l15)

	output[0] = fdctRoundShift2Int16Native(o0)
	output[1] = fdctRoundShift2Int16Native(o1)
	output[2] = fdctRoundShift2Int16Native(o2)
	output[3] = fdctRoundShift2Int16Native(o3)
	output[4] = fdctRoundShift2Int16Native(o4)
	output[5] = fdctRoundShift2Int16Native(o5)
	output[6] = fdctRoundShift2Int16Native(o6)
	output[7] = fdctRoundShift2Int16Native(o7)
	output[8] = fdctRoundShift2Int16Native(o8)
	output[9] = fdctRoundShift2Int16Native(o9)
	output[10] = fdctRoundShift2Int16Native(o10)
	output[11] = fdctRoundShift2Int16Native(o11)
	output[12] = fdctRoundShift2Int16Native(o12)
	output[13] = fdctRoundShift2Int16Native(o13)
	output[14] = fdctRoundShift2Int16Native(o14)
	output[15] = fdctRoundShift2Int16Native(o15)
}

func fdctAdd16(a, b archsimd.Int16x8) archsimd.Int16x8 {
	return a.AddSaturated(b)
}

func fdctSub16(a, b archsimd.Int16x8) archsimd.Int16x8 {
	return a.SubSaturated(b)
}

// Per-index pre-broadcast twiddle tables for the fdct16 butterflies.
var fdct16Pos13 = [64][8]int16{
	4: bcast16(fwdCospi13[4]),
	8: bcast16(fwdCospi13[8]),
	12: bcast16(fwdCospi13[12]),
	16: bcast16(fwdCospi13[16]),
	20: bcast16(fwdCospi13[20]),
	24: bcast16(fwdCospi13[24]),
	28: bcast16(fwdCospi13[28]),
	32: bcast16(fwdCospi13[32]),
	36: bcast16(fwdCospi13[36]),
	40: bcast16(fwdCospi13[40]),
	44: bcast16(fwdCospi13[44]),
	48: bcast16(fwdCospi13[48]),
	52: bcast16(fwdCospi13[52]),
	56: bcast16(fwdCospi13[56]),
	60: bcast16(fwdCospi13[60]),
}
var fdct16Neg13 = [64][8]int16{
	4: bcast16(-fwdCospi13[4]),
	8: bcast16(-fwdCospi13[8]),
	12: bcast16(-fwdCospi13[12]),
	16: bcast16(-fwdCospi13[16]),
	20: bcast16(-fwdCospi13[20]),
	24: bcast16(-fwdCospi13[24]),
	28: bcast16(-fwdCospi13[28]),
	32: bcast16(-fwdCospi13[32]),
	36: bcast16(-fwdCospi13[36]),
	40: bcast16(-fwdCospi13[40]),
	44: bcast16(-fwdCospi13[44]),
	48: bcast16(-fwdCospi13[48]),
	52: bcast16(-fwdCospi13[52]),
	56: bcast16(-fwdCospi13[56]),
	60: bcast16(-fwdCospi13[60]),
}
var fdct16Pos12 = [64][8]int16{
	4: bcast16(fwdCospi12[4]),
	8: bcast16(fwdCospi12[8]),
	12: bcast16(fwdCospi12[12]),
	16: bcast16(fwdCospi12[16]),
	20: bcast16(fwdCospi12[20]),
	24: bcast16(fwdCospi12[24]),
	28: bcast16(fwdCospi12[28]),
	32: bcast16(fwdCospi12[32]),
	36: bcast16(fwdCospi12[36]),
	40: bcast16(fwdCospi12[40]),
	44: bcast16(fwdCospi12[44]),
	48: bcast16(fwdCospi12[48]),
	52: bcast16(fwdCospi12[52]),
	56: bcast16(fwdCospi12[56]),
	60: bcast16(fwdCospi12[60]),
}
var fdct16Neg12 = [64][8]int16{
	4: bcast16(-fwdCospi12[4]),
	8: bcast16(-fwdCospi12[8]),
	12: bcast16(-fwdCospi12[12]),
	16: bcast16(-fwdCospi12[16]),
	20: bcast16(-fwdCospi12[20]),
	24: bcast16(-fwdCospi12[24]),
	28: bcast16(-fwdCospi12[28]),
	32: bcast16(-fwdCospi12[32]),
	36: bcast16(-fwdCospi12[36]),
	40: bcast16(-fwdCospi12[40]),
	44: bcast16(-fwdCospi12[44]),
	48: bcast16(-fwdCospi12[48]),
	52: bcast16(-fwdCospi12[52]),
	56: bcast16(-fwdCospi12[56]),
	60: bcast16(-fwdCospi12[60]),
}

// fdctHalfBtf16V is fdctHalfBtf16 with pre-broadcast twiddle vectors (loaded
// from rodata) instead of scalar weights, so no VMOV+VDUP per call.
func fdctHalfBtf16V(k0 archsimd.Int16x8, in0 archsimd.Int16x8, k1 archsimd.Int16x8, in1 archsimd.Int16x8, cosBit uint8) archsimd.Int16x8 {
	lo := in0.MulWidenLo(k0).MulWidenLoAdd(in1, k1)
	hi := in0.MulWidenHi(k0).MulWidenHiAdd(in1, k1)
	return lo.ShiftRightRoundNarrow(cosBit).ShiftRightRoundNarrowHi(hi, cosBit)
}

// fdctRoundShift1Int16Native computes ROUND_POWER_OF_TWO_SIGNED(v, 1) =
// (v + 1 + (v>>31)) >> 1 entirely in int16 lanes, matching the int32 reference
// for the whole int16 input domain (the +1 cannot overflow because v>=32767
// would require an unshifted lane, which the FDCT column output never reaches
// for the in-range residuals the SIMD path already assumes). This avoids the
// widen->int32->narrow round trip (VSXTL/VXTN + VSSHR pairs) that dominates the
// column pass; each lane is one SSHR (sign) + ADD + ADD + SSHR.
func fdctRoundShift1Int16Native(v archsimd.Int16x8) archsimd.Int16x8 {
	// (v + 1) folds a VMOVI immediate; the sign term is v>>15 (-1 if negative).
	sign := v.ShiftAllRightConst(15)
	return v.Add(archsimd.BroadcastInt16x8(1)).Add(sign).ShiftAllRightConst(1)
}

// fdctRoundShift2Int16Native computes round_shift(v, 2) = (v + 2) >> 2 in int16
// lanes. This matches the int32 reference across the value range the FDCT
// column pass actually produces (well inside int16, so v+2 does not overflow);
// it removes the widen->int32->narrow round trip.
func fdctRoundShift2Int16Native(v archsimd.Int16x8) archsimd.Int16x8 {
	return v.Add(archsimd.BroadcastInt16x8(2)).ShiftAllRightConst(2)
}

// fdct16Butterfly13/12 are av1_fdct16 with the cosBit shift baked in as a
// compile-time constant. Passing cosBit as a runtime value forced the VSQRSHRN
// round-shift to a per-shift-amount jump table (hundreds of dead branches);
// the constant lets each narrow lower to a single VSQRSHRN #imm.
func fdct16Butterfly13(
	i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15 archsimd.Int16x8,
) (
	archsimd.Int16x8, archsimd.Int16x8, archsimd.Int16x8, archsimd.Int16x8,
	archsimd.Int16x8, archsimd.Int16x8, archsimd.Int16x8, archsimd.Int16x8,
	archsimd.Int16x8, archsimd.Int16x8, archsimd.Int16x8, archsimd.Int16x8,
	archsimd.Int16x8, archsimd.Int16x8, archsimd.Int16x8, archsimd.Int16x8,
) {
	// Stage 1
	a0 := fdctAdd16(i0, i15)
	a1 := fdctAdd16(i1, i14)
	a2 := fdctAdd16(i2, i13)
	a3 := fdctAdd16(i3, i12)
	a4 := fdctAdd16(i4, i11)
	a5 := fdctAdd16(i5, i10)
	a6 := fdctAdd16(i6, i9)
	a7 := fdctAdd16(i7, i8)
	a8 := fdctSub16(i7, i8)
	a9 := fdctSub16(i6, i9)
	a10 := fdctSub16(i5, i10)
	a11 := fdctSub16(i4, i11)
	a12 := fdctSub16(i3, i12)
	a13 := fdctSub16(i2, i13)
	a14 := fdctSub16(i1, i14)
	a15 := fdctSub16(i0, i15)

	// Stage 2
	b0 := fdctAdd16(a0, a7)
	b1 := fdctAdd16(a1, a6)
	b2 := fdctAdd16(a2, a5)
	b3 := fdctAdd16(a3, a4)
	b4 := fdctSub16(a3, a4)
	b5 := fdctSub16(a2, a5)
	b6 := fdctSub16(a1, a6)
	b7 := fdctSub16(a0, a7)
	b8 := a8
	b9 := a9
	b10 := fdctHalfBtf16V(archsimd.LoadInt16x8Array(&fdct16Neg13[32]), a10, archsimd.LoadInt16x8Array(&fdct16Pos13[32]), a13, 13)
	b11 := fdctHalfBtf16V(archsimd.LoadInt16x8Array(&fdct16Neg13[32]), a11, archsimd.LoadInt16x8Array(&fdct16Pos13[32]), a12, 13)
	b12 := fdctHalfBtf16V(archsimd.LoadInt16x8Array(&fdct16Pos13[32]), a12, archsimd.LoadInt16x8Array(&fdct16Pos13[32]), a11, 13)
	b13 := fdctHalfBtf16V(archsimd.LoadInt16x8Array(&fdct16Pos13[32]), a13, archsimd.LoadInt16x8Array(&fdct16Pos13[32]), a10, 13)
	b14 := a14
	b15 := a15

	// Stage 3
	c0 := fdctAdd16(b0, b3)
	c1 := fdctAdd16(b1, b2)
	c2 := fdctSub16(b1, b2)
	c3 := fdctSub16(b0, b3)
	c4 := b4
	c5 := fdctHalfBtf16V(archsimd.LoadInt16x8Array(&fdct16Neg13[32]), b5, archsimd.LoadInt16x8Array(&fdct16Pos13[32]), b6, 13)
	c6 := fdctHalfBtf16V(archsimd.LoadInt16x8Array(&fdct16Pos13[32]), b6, archsimd.LoadInt16x8Array(&fdct16Pos13[32]), b5, 13)
	c7 := b7
	c8 := fdctAdd16(b8, b11)
	c9 := fdctAdd16(b9, b10)
	c10 := fdctSub16(b9, b10)
	c11 := fdctSub16(b8, b11)
	c12 := fdctSub16(b15, b12)
	c13 := fdctSub16(b14, b13)
	c14 := fdctAdd16(b14, b13)
	c15 := fdctAdd16(b15, b12)

	// Stage 4
	d0 := fdctHalfBtf16V(archsimd.LoadInt16x8Array(&fdct16Pos13[32]), c0, archsimd.LoadInt16x8Array(&fdct16Pos13[32]), c1, 13)
	d1 := fdctHalfBtf16V(archsimd.LoadInt16x8Array(&fdct16Neg13[32]), c1, archsimd.LoadInt16x8Array(&fdct16Pos13[32]), c0, 13)
	d2 := fdctHalfBtf16V(archsimd.LoadInt16x8Array(&fdct16Pos13[48]), c2, archsimd.LoadInt16x8Array(&fdct16Pos13[16]), c3, 13)
	d3 := fdctHalfBtf16V(archsimd.LoadInt16x8Array(&fdct16Pos13[48]), c3, archsimd.LoadInt16x8Array(&fdct16Neg13[16]), c2, 13)
	d4 := fdctAdd16(c4, c5)
	d5 := fdctSub16(c4, c5)
	d6 := fdctSub16(c7, c6)
	d7 := fdctAdd16(c7, c6)
	d8 := c8
	d9 := fdctHalfBtf16V(archsimd.LoadInt16x8Array(&fdct16Neg13[16]), c9, archsimd.LoadInt16x8Array(&fdct16Pos13[48]), c14, 13)
	d10 := fdctHalfBtf16V(archsimd.LoadInt16x8Array(&fdct16Neg13[48]), c10, archsimd.LoadInt16x8Array(&fdct16Neg13[16]), c13, 13)
	d11 := c11
	d12 := c12
	d13 := fdctHalfBtf16V(archsimd.LoadInt16x8Array(&fdct16Pos13[48]), c13, archsimd.LoadInt16x8Array(&fdct16Neg13[16]), c10, 13)
	d14 := fdctHalfBtf16V(archsimd.LoadInt16x8Array(&fdct16Pos13[16]), c14, archsimd.LoadInt16x8Array(&fdct16Pos13[48]), c9, 13)
	d15 := c15

	// Stage 5
	e0 := d0
	e1 := d1
	e2 := d2
	e3 := d3
	e4 := fdctHalfBtf16V(archsimd.LoadInt16x8Array(&fdct16Pos13[56]), d4, archsimd.LoadInt16x8Array(&fdct16Pos13[8]), d7, 13)
	e5 := fdctHalfBtf16V(archsimd.LoadInt16x8Array(&fdct16Pos13[24]), d5, archsimd.LoadInt16x8Array(&fdct16Pos13[40]), d6, 13)
	e6 := fdctHalfBtf16V(archsimd.LoadInt16x8Array(&fdct16Pos13[24]), d6, archsimd.LoadInt16x8Array(&fdct16Neg13[40]), d5, 13)
	e7 := fdctHalfBtf16V(archsimd.LoadInt16x8Array(&fdct16Pos13[56]), d7, archsimd.LoadInt16x8Array(&fdct16Neg13[8]), d4, 13)
	e8 := fdctAdd16(d8, d9)
	e9 := fdctSub16(d8, d9)
	e10 := fdctSub16(d11, d10)
	e11 := fdctAdd16(d11, d10)
	e12 := fdctAdd16(d12, d13)
	e13 := fdctSub16(d12, d13)
	e14 := fdctSub16(d15, d14)
	e15 := fdctAdd16(d15, d14)

	// Stage 6
	f8 := fdctHalfBtf16V(archsimd.LoadInt16x8Array(&fdct16Pos13[60]), e8, archsimd.LoadInt16x8Array(&fdct16Pos13[4]), e15, 13)
	f9 := fdctHalfBtf16V(archsimd.LoadInt16x8Array(&fdct16Pos13[28]), e9, archsimd.LoadInt16x8Array(&fdct16Pos13[36]), e14, 13)
	f10 := fdctHalfBtf16V(archsimd.LoadInt16x8Array(&fdct16Pos13[44]), e10, archsimd.LoadInt16x8Array(&fdct16Pos13[20]), e13, 13)
	f11 := fdctHalfBtf16V(archsimd.LoadInt16x8Array(&fdct16Pos13[12]), e11, archsimd.LoadInt16x8Array(&fdct16Pos13[52]), e12, 13)
	f12 := fdctHalfBtf16V(archsimd.LoadInt16x8Array(&fdct16Pos13[12]), e12, archsimd.LoadInt16x8Array(&fdct16Neg13[52]), e11, 13)
	f13 := fdctHalfBtf16V(archsimd.LoadInt16x8Array(&fdct16Pos13[44]), e13, archsimd.LoadInt16x8Array(&fdct16Neg13[20]), e10, 13)
	f14 := fdctHalfBtf16V(archsimd.LoadInt16x8Array(&fdct16Pos13[28]), e14, archsimd.LoadInt16x8Array(&fdct16Neg13[36]), e9, 13)
	f15 := fdctHalfBtf16V(archsimd.LoadInt16x8Array(&fdct16Pos13[60]), e15, archsimd.LoadInt16x8Array(&fdct16Neg13[4]), e8, 13)

	// Output permutation
	return e0, f8, e4, f12, e2, f10, e6, f14, e1, f9, e5, f13, e3, f11, e7, f15
}


func fdct16Butterfly12(
	i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15 archsimd.Int16x8,
) (
	archsimd.Int16x8, archsimd.Int16x8, archsimd.Int16x8, archsimd.Int16x8,
	archsimd.Int16x8, archsimd.Int16x8, archsimd.Int16x8, archsimd.Int16x8,
	archsimd.Int16x8, archsimd.Int16x8, archsimd.Int16x8, archsimd.Int16x8,
	archsimd.Int16x8, archsimd.Int16x8, archsimd.Int16x8, archsimd.Int16x8,
) {
	// Stage 1
	a0 := fdctAdd16(i0, i15)
	a1 := fdctAdd16(i1, i14)
	a2 := fdctAdd16(i2, i13)
	a3 := fdctAdd16(i3, i12)
	a4 := fdctAdd16(i4, i11)
	a5 := fdctAdd16(i5, i10)
	a6 := fdctAdd16(i6, i9)
	a7 := fdctAdd16(i7, i8)
	a8 := fdctSub16(i7, i8)
	a9 := fdctSub16(i6, i9)
	a10 := fdctSub16(i5, i10)
	a11 := fdctSub16(i4, i11)
	a12 := fdctSub16(i3, i12)
	a13 := fdctSub16(i2, i13)
	a14 := fdctSub16(i1, i14)
	a15 := fdctSub16(i0, i15)

	// Stage 2
	b0 := fdctAdd16(a0, a7)
	b1 := fdctAdd16(a1, a6)
	b2 := fdctAdd16(a2, a5)
	b3 := fdctAdd16(a3, a4)
	b4 := fdctSub16(a3, a4)
	b5 := fdctSub16(a2, a5)
	b6 := fdctSub16(a1, a6)
	b7 := fdctSub16(a0, a7)
	b8 := a8
	b9 := a9
	b10 := fdctHalfBtf16V(archsimd.LoadInt16x8Array(&fdct16Neg12[32]), a10, archsimd.LoadInt16x8Array(&fdct16Pos12[32]), a13, 12)
	b11 := fdctHalfBtf16V(archsimd.LoadInt16x8Array(&fdct16Neg12[32]), a11, archsimd.LoadInt16x8Array(&fdct16Pos12[32]), a12, 12)
	b12 := fdctHalfBtf16V(archsimd.LoadInt16x8Array(&fdct16Pos12[32]), a12, archsimd.LoadInt16x8Array(&fdct16Pos12[32]), a11, 12)
	b13 := fdctHalfBtf16V(archsimd.LoadInt16x8Array(&fdct16Pos12[32]), a13, archsimd.LoadInt16x8Array(&fdct16Pos12[32]), a10, 12)
	b14 := a14
	b15 := a15

	// Stage 3
	c0 := fdctAdd16(b0, b3)
	c1 := fdctAdd16(b1, b2)
	c2 := fdctSub16(b1, b2)
	c3 := fdctSub16(b0, b3)
	c4 := b4
	c5 := fdctHalfBtf16V(archsimd.LoadInt16x8Array(&fdct16Neg12[32]), b5, archsimd.LoadInt16x8Array(&fdct16Pos12[32]), b6, 12)
	c6 := fdctHalfBtf16V(archsimd.LoadInt16x8Array(&fdct16Pos12[32]), b6, archsimd.LoadInt16x8Array(&fdct16Pos12[32]), b5, 12)
	c7 := b7
	c8 := fdctAdd16(b8, b11)
	c9 := fdctAdd16(b9, b10)
	c10 := fdctSub16(b9, b10)
	c11 := fdctSub16(b8, b11)
	c12 := fdctSub16(b15, b12)
	c13 := fdctSub16(b14, b13)
	c14 := fdctAdd16(b14, b13)
	c15 := fdctAdd16(b15, b12)

	// Stage 4
	d0 := fdctHalfBtf16V(archsimd.LoadInt16x8Array(&fdct16Pos12[32]), c0, archsimd.LoadInt16x8Array(&fdct16Pos12[32]), c1, 12)
	d1 := fdctHalfBtf16V(archsimd.LoadInt16x8Array(&fdct16Neg12[32]), c1, archsimd.LoadInt16x8Array(&fdct16Pos12[32]), c0, 12)
	d2 := fdctHalfBtf16V(archsimd.LoadInt16x8Array(&fdct16Pos12[48]), c2, archsimd.LoadInt16x8Array(&fdct16Pos12[16]), c3, 12)
	d3 := fdctHalfBtf16V(archsimd.LoadInt16x8Array(&fdct16Pos12[48]), c3, archsimd.LoadInt16x8Array(&fdct16Neg12[16]), c2, 12)
	d4 := fdctAdd16(c4, c5)
	d5 := fdctSub16(c4, c5)
	d6 := fdctSub16(c7, c6)
	d7 := fdctAdd16(c7, c6)
	d8 := c8
	d9 := fdctHalfBtf16V(archsimd.LoadInt16x8Array(&fdct16Neg12[16]), c9, archsimd.LoadInt16x8Array(&fdct16Pos12[48]), c14, 12)
	d10 := fdctHalfBtf16V(archsimd.LoadInt16x8Array(&fdct16Neg12[48]), c10, archsimd.LoadInt16x8Array(&fdct16Neg12[16]), c13, 12)
	d11 := c11
	d12 := c12
	d13 := fdctHalfBtf16V(archsimd.LoadInt16x8Array(&fdct16Pos12[48]), c13, archsimd.LoadInt16x8Array(&fdct16Neg12[16]), c10, 12)
	d14 := fdctHalfBtf16V(archsimd.LoadInt16x8Array(&fdct16Pos12[16]), c14, archsimd.LoadInt16x8Array(&fdct16Pos12[48]), c9, 12)
	d15 := c15

	// Stage 5
	e0 := d0
	e1 := d1
	e2 := d2
	e3 := d3
	e4 := fdctHalfBtf16V(archsimd.LoadInt16x8Array(&fdct16Pos12[56]), d4, archsimd.LoadInt16x8Array(&fdct16Pos12[8]), d7, 12)
	e5 := fdctHalfBtf16V(archsimd.LoadInt16x8Array(&fdct16Pos12[24]), d5, archsimd.LoadInt16x8Array(&fdct16Pos12[40]), d6, 12)
	e6 := fdctHalfBtf16V(archsimd.LoadInt16x8Array(&fdct16Pos12[24]), d6, archsimd.LoadInt16x8Array(&fdct16Neg12[40]), d5, 12)
	e7 := fdctHalfBtf16V(archsimd.LoadInt16x8Array(&fdct16Pos12[56]), d7, archsimd.LoadInt16x8Array(&fdct16Neg12[8]), d4, 12)
	e8 := fdctAdd16(d8, d9)
	e9 := fdctSub16(d8, d9)
	e10 := fdctSub16(d11, d10)
	e11 := fdctAdd16(d11, d10)
	e12 := fdctAdd16(d12, d13)
	e13 := fdctSub16(d12, d13)
	e14 := fdctSub16(d15, d14)
	e15 := fdctAdd16(d15, d14)

	// Stage 6
	f8 := fdctHalfBtf16V(archsimd.LoadInt16x8Array(&fdct16Pos12[60]), e8, archsimd.LoadInt16x8Array(&fdct16Pos12[4]), e15, 12)
	f9 := fdctHalfBtf16V(archsimd.LoadInt16x8Array(&fdct16Pos12[28]), e9, archsimd.LoadInt16x8Array(&fdct16Pos12[36]), e14, 12)
	f10 := fdctHalfBtf16V(archsimd.LoadInt16x8Array(&fdct16Pos12[44]), e10, archsimd.LoadInt16x8Array(&fdct16Pos12[20]), e13, 12)
	f11 := fdctHalfBtf16V(archsimd.LoadInt16x8Array(&fdct16Pos12[12]), e11, archsimd.LoadInt16x8Array(&fdct16Pos12[52]), e12, 12)
	f12 := fdctHalfBtf16V(archsimd.LoadInt16x8Array(&fdct16Pos12[12]), e12, archsimd.LoadInt16x8Array(&fdct16Neg12[52]), e11, 12)
	f13 := fdctHalfBtf16V(archsimd.LoadInt16x8Array(&fdct16Pos12[44]), e13, archsimd.LoadInt16x8Array(&fdct16Neg12[20]), e10, 12)
	f14 := fdctHalfBtf16V(archsimd.LoadInt16x8Array(&fdct16Pos12[28]), e14, archsimd.LoadInt16x8Array(&fdct16Neg12[36]), e9, 12)
	f15 := fdctHalfBtf16V(archsimd.LoadInt16x8Array(&fdct16Pos12[60]), e15, archsimd.LoadInt16x8Array(&fdct16Neg12[4]), e8, 12)

	// Output permutation
	return e0, f8, e4, f12, e2, f10, e6, f14, e1, f9, e5, f13, e3, f11, e7, f15
}

