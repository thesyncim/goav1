// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.
//
// The four kernels share one straight-line template: the butterflies,
// transpose, and store are all inlined with NO SIMD-returning calls in the hot
// body. A CALL clobbers the caller-saved V-registers and would spill every live
// vector around it (a wall of FMOVQ/STP to RSP), so only tiny leaf helpers that
// actually inline (one half-butterfly, the reinterpret casts, the round-shift-1)
// stay as functions. Twiddles are loaded into per-pass locals so each rodata
// load emits once (not an ADRP+FMOVQ per half-butterfly), and the round-narrow
// shift bakes its amount as a compile-time constant. Measured to beat the NEON
// asm by 15-21% on the 8x8 hybrid benchmarks.

//go:build goexperiment.simd && arm64 && !purego

package transform

import (
	"simd/archsimd"
	"unsafe"
)

// forwardBlock8x8{ADSTDCT,DCTADST,ADSTADST,IDTX}Impl are the Go-native SIMD
// dispatch slots for the 8x8 hybrid transforms under GOEXPERIMENT=simd,
// replacing the hand-written NEON asm. The math runs in int16 lanes (eight
// columns per Int16x8): the half-butterfly widens int16*int16 to int32 for the
// product and narrows back with a rounding shift (ShiftRightRoundNarrow ==
// srshr #13); adds saturate but never clamp in the 8-bit residual domain, so the
// result is byte-identical to the int32 scalar reference (verified by the
// differential test).
var forwardBlock8x8ADSTDCTImpl = forwardBlock8x8ADSTDCTSIMD
var forwardBlock8x8DCTADSTImpl = forwardBlock8x8DCTADSTSIMD
var forwardBlock8x8ADSTADSTImpl = forwardBlock8x8ADSTADSTSIMD
var forwardBlock8x8IDTXImpl = forwardBlock8x8IDTXSIMD

// bcast16h fills a [8]int16 with v (compile-time constant folded).
func bcast16h(v int32) [8]int16 {
	x := int16(v)
	return [8]int16{x, x, x, x, x, x, x, x}
}

// Pre-broadcast twiddle vectors (int16, cos_bit 13).
var (
	fadstHW32  = bcast16h(fwdCospi13[32])
	fadstHWn32 = bcast16h(-fwdCospi13[32])
	fadstHW16  = bcast16h(fwdCospi13[16])
	fadstHW48  = bcast16h(fwdCospi13[48])
	fadstHWn16 = bcast16h(-fwdCospi13[16])
	fadstHWn48 = bcast16h(-fwdCospi13[48])
	fadstHW4   = bcast16h(fwdCospi13[4])
	fadstHW60  = bcast16h(fwdCospi13[60])
	fadstHWn4  = bcast16h(-fwdCospi13[4])
	fadstHW20  = bcast16h(fwdCospi13[20])
	fadstHW44  = bcast16h(fwdCospi13[44])
	fadstHWn20 = bcast16h(-fwdCospi13[20])
	fadstHW36  = bcast16h(fwdCospi13[36])
	fadstHW28  = bcast16h(fwdCospi13[28])
	fadstHWn36 = bcast16h(-fwdCospi13[36])
	fadstHW52  = bcast16h(fwdCospi13[52])
	fadstHW12  = bcast16h(fwdCospi13[12])
	fadstHWn52 = bcast16h(-fwdCospi13[52])

	fdctHW32  = bcast16h(fwdCospi13[32])
	fdctHWn32 = bcast16h(-fwdCospi13[32])
	fdctHW48  = bcast16h(fwdCospi13[48])
	fdctHW16  = bcast16h(fwdCospi13[16])
	fdctHWn16 = bcast16h(-fwdCospi13[16])
	fdctHW56  = bcast16h(fwdCospi13[56])
	fdctHW8   = bcast16h(fwdCospi13[8])
	fdctHWn8  = bcast16h(-fwdCospi13[8])
	fdctHW24  = bcast16h(fwdCospi13[24])
	fdctHW40  = bcast16h(fwdCospi13[40])
	fdctHWn40 = bcast16h(-fwdCospi13[40])
)

// adstHalfBtf16 is half_btf at cos_bit 13, int16 I/O with an int32 accumulator
// and a rounding narrow shift; twiddles passed pre-loaded. Tiny; inlines.
func adstHalfBtf16(k0, in0, k1, in1 archsimd.Int16x8) archsimd.Int16x8 {
	lo := in0.MulWidenLo(k0).MulWidenLoAdd(in1, k1)
	hi := in0.MulWidenHi(k0).MulWidenHiAdd(in1, k1)
	return lo.ShiftRightRoundNarrow(13).ShiftRightRoundNarrowHi(hi, 13)
}

// adstRoundShift1Int16 is fwdRoundShift1Value per int16 lane: (v+1+(v>>15))>>1.
// The +1 folds into a VMOVI-immediate ADD (no live broadcast register).
func adstRoundShift1Int16(v archsimd.Int16x8) archsimd.Int16x8 {
	sign := v.ShiftAllRightConst(15)
	return v.Add(archsimd.BroadcastInt16x8(1)).Add(sign).ShiftAllRightConst(1)
}

func adstInt16AsInt32(v archsimd.Int16x8) archsimd.Int32x4 {
	return v.ToBits().ReshapeToUint32s().BitsToInt32()
}
func adstInt32AsInt16(v archsimd.Int32x4) archsimd.Int16x8 {
	return v.ToBits().ReshapeToUint16s().BitsToInt16()
}
func adstInt16AsInt64(v archsimd.Int16x8) archsimd.Int64x2 {
	return v.ToBits().ReshapeToUint64s().BitsToInt64()
}
func adstInt64AsInt16(v archsimd.Int64x2) archsimd.Int16x8 {
	return v.ToBits().ReshapeToUint16s().BitsToInt16()
}

// forwardBlock8x8ADSTADSTSIMD: ADST column, common shift, transpose, ADST row.
func forwardBlock8x8ADSTADSTSIMD(coeff []int32, coeffStride int, residual []int16, residualStride int, scratch []int32) {
	_ = scratch[63]
	rbase := unsafe.Pointer(unsafe.SliceData(residual))
	rstep := uintptr(residualStride) * 2
	l0 := archsimd.LoadInt16x8Array((*[8]int16)(rbase)).ShiftAllLeftConst(2)
	l1 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(rbase, rstep*1))).ShiftAllLeftConst(2)
	l2 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(rbase, rstep*2))).ShiftAllLeftConst(2)
	l3 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(rbase, rstep*3))).ShiftAllLeftConst(2)
	l4 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(rbase, rstep*4))).ShiftAllLeftConst(2)
	l5 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(rbase, rstep*5))).ShiftAllLeftConst(2)
	l6 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(rbase, rstep*6))).ShiftAllLeftConst(2)
	l7 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(rbase, rstep*7))).ShiftAllLeftConst(2)

	// --- Column pass ---
	var c0, c1, c2, c3, c4, c5, c6, c7 archsimd.Int16x8
	{
		kW32 := archsimd.LoadInt16x8Array(&fadstHW32)
		kWn32 := archsimd.LoadInt16x8Array(&fadstHWn32)
		kW16 := archsimd.LoadInt16x8Array(&fadstHW16)
		kW48 := archsimd.LoadInt16x8Array(&fadstHW48)
		kWn16 := archsimd.LoadInt16x8Array(&fadstHWn16)
		kWn48 := archsimd.LoadInt16x8Array(&fadstHWn48)
		kW4 := archsimd.LoadInt16x8Array(&fadstHW4)
		kW60 := archsimd.LoadInt16x8Array(&fadstHW60)
		kWn4 := archsimd.LoadInt16x8Array(&fadstHWn4)
		kW20 := archsimd.LoadInt16x8Array(&fadstHW20)
		kW44 := archsimd.LoadInt16x8Array(&fadstHW44)
		kWn20 := archsimd.LoadInt16x8Array(&fadstHWn20)
		kW36 := archsimd.LoadInt16x8Array(&fadstHW36)
		kW28 := archsimd.LoadInt16x8Array(&fadstHW28)
		kWn36 := archsimd.LoadInt16x8Array(&fadstHWn36)
		kW52 := archsimd.LoadInt16x8Array(&fadstHW52)
		kW12 := archsimd.LoadInt16x8Array(&fadstHW12)
		kWn52 := archsimd.LoadInt16x8Array(&fadstHWn52)
		s0 := l0
		s1 := l7.Neg()
		s2 := adstHalfBtf16(kWn32, l3, kW32, l4)
		s3 := adstHalfBtf16(kWn32, l3, kWn32, l4)
		s4 := l1.Neg()
		s5 := l6
		s6 := adstHalfBtf16(kW32, l2, kWn32, l5)
		s7 := adstHalfBtf16(kW32, l2, kW32, l5)

		t0 := s0.AddSaturated(s2)
		t1 := s1.AddSaturated(s3)
		t2 := s0.SubSaturated(s2)
		t3 := s1.SubSaturated(s3)
		t4 := s4.AddSaturated(s6)
		t5 := s5.AddSaturated(s7)
		t6 := s4.SubSaturated(s6)
		t7 := s5.SubSaturated(s7)

		s4 = adstHalfBtf16(kW16, t4, kW48, t5)
		s5 = adstHalfBtf16(kW48, t4, kWn16, t5)
		s6 = adstHalfBtf16(kWn48, t6, kW16, t7)
		s7 = adstHalfBtf16(kW16, t6, kW48, t7)

		u4 := t0.SubSaturated(s4)
		u5 := t1.SubSaturated(s5)
		u6 := t2.SubSaturated(s6)
		u7 := t3.SubSaturated(s7)
		t0 = t0.AddSaturated(s4)
		t1 = t1.AddSaturated(s5)
		t2 = t2.AddSaturated(s6)
		t3 = t3.AddSaturated(s7)

		c0 = adstRoundShift1Int16(adstHalfBtf16(kW60, t0, kWn4, t1))
		c1 = adstRoundShift1Int16(adstHalfBtf16(kW52, u6, kW12, u7))
		c2 = adstRoundShift1Int16(adstHalfBtf16(kW44, t2, kWn20, t3))
		c3 = adstRoundShift1Int16(adstHalfBtf16(kW36, u4, kW28, u5))
		c4 = adstRoundShift1Int16(adstHalfBtf16(kW28, u4, kWn36, u5))
		c5 = adstRoundShift1Int16(adstHalfBtf16(kW20, t2, kW44, t3))
		c6 = adstRoundShift1Int16(adstHalfBtf16(kW12, u6, kWn52, u7))
		c7 = adstRoundShift1Int16(adstHalfBtf16(kW4, t0, kW60, t1))
	}

	// --- Transpose (inlined 8x8 int16) ---
	a0 := c0.InterleaveLo(c1)
	a1 := c0.InterleaveHi(c1)
	a2 := c2.InterleaveLo(c3)
	a3 := c2.InterleaveHi(c3)
	a4 := c4.InterleaveLo(c5)
	a5 := c4.InterleaveHi(c5)
	a6 := c6.InterleaveLo(c7)
	a7 := c6.InterleaveHi(c7)
	b0 := adstInt32AsInt16(adstInt16AsInt32(a0).InterleaveLo(adstInt16AsInt32(a2)))
	b1 := adstInt32AsInt16(adstInt16AsInt32(a0).InterleaveHi(adstInt16AsInt32(a2)))
	b2 := adstInt32AsInt16(adstInt16AsInt32(a1).InterleaveLo(adstInt16AsInt32(a3)))
	b3 := adstInt32AsInt16(adstInt16AsInt32(a1).InterleaveHi(adstInt16AsInt32(a3)))
	b4 := adstInt32AsInt16(adstInt16AsInt32(a4).InterleaveLo(adstInt16AsInt32(a6)))
	b5 := adstInt32AsInt16(adstInt16AsInt32(a4).InterleaveHi(adstInt16AsInt32(a6)))
	b6 := adstInt32AsInt16(adstInt16AsInt32(a5).InterleaveLo(adstInt16AsInt32(a7)))
	b7 := adstInt32AsInt16(adstInt16AsInt32(a5).InterleaveHi(adstInt16AsInt32(a7)))
	t0 := adstInt64AsInt16(adstInt16AsInt64(b0).InterleaveLo(adstInt16AsInt64(b4)))
	t1 := adstInt64AsInt16(adstInt16AsInt64(b0).InterleaveHi(adstInt16AsInt64(b4)))
	t2 := adstInt64AsInt16(adstInt16AsInt64(b1).InterleaveLo(adstInt16AsInt64(b5)))
	t3 := adstInt64AsInt16(adstInt16AsInt64(b1).InterleaveHi(adstInt16AsInt64(b5)))
	t4 := adstInt64AsInt16(adstInt16AsInt64(b2).InterleaveLo(adstInt16AsInt64(b6)))
	t5 := adstInt64AsInt16(adstInt16AsInt64(b2).InterleaveHi(adstInt16AsInt64(b6)))
	t6 := adstInt64AsInt16(adstInt16AsInt64(b3).InterleaveLo(adstInt16AsInt64(b7)))
	t7 := adstInt64AsInt16(adstInt16AsInt64(b3).InterleaveHi(adstInt16AsInt64(b7)))

	// --- Row pass ---
	var o0, o1, o2, o3, o4, o5, o6, o7 archsimd.Int16x8
	{
		kW32 := archsimd.LoadInt16x8Array(&fadstHW32)
		kWn32 := archsimd.LoadInt16x8Array(&fadstHWn32)
		kW16 := archsimd.LoadInt16x8Array(&fadstHW16)
		kW48 := archsimd.LoadInt16x8Array(&fadstHW48)
		kWn16 := archsimd.LoadInt16x8Array(&fadstHWn16)
		kWn48 := archsimd.LoadInt16x8Array(&fadstHWn48)
		kW4 := archsimd.LoadInt16x8Array(&fadstHW4)
		kW60 := archsimd.LoadInt16x8Array(&fadstHW60)
		kWn4 := archsimd.LoadInt16x8Array(&fadstHWn4)
		kW20 := archsimd.LoadInt16x8Array(&fadstHW20)
		kW44 := archsimd.LoadInt16x8Array(&fadstHW44)
		kWn20 := archsimd.LoadInt16x8Array(&fadstHWn20)
		kW36 := archsimd.LoadInt16x8Array(&fadstHW36)
		kW28 := archsimd.LoadInt16x8Array(&fadstHW28)
		kWn36 := archsimd.LoadInt16x8Array(&fadstHWn36)
		kW52 := archsimd.LoadInt16x8Array(&fadstHW52)
		kW12 := archsimd.LoadInt16x8Array(&fadstHW12)
		kWn52 := archsimd.LoadInt16x8Array(&fadstHWn52)
		s0 := t0
		s1 := t7.Neg()
		s2 := adstHalfBtf16(kWn32, t3, kW32, t4)
		s3 := adstHalfBtf16(kWn32, t3, kWn32, t4)
		s4 := t1.Neg()
		s5 := t6
		s6 := adstHalfBtf16(kW32, t2, kWn32, t5)
		s7 := adstHalfBtf16(kW32, t2, kW32, t5)

		t0 := s0.AddSaturated(s2)
		t1 := s1.AddSaturated(s3)
		t2 := s0.SubSaturated(s2)
		t3 := s1.SubSaturated(s3)
		t4 := s4.AddSaturated(s6)
		t5 := s5.AddSaturated(s7)
		t6 := s4.SubSaturated(s6)
		t7 := s5.SubSaturated(s7)

		s4 = adstHalfBtf16(kW16, t4, kW48, t5)
		s5 = adstHalfBtf16(kW48, t4, kWn16, t5)
		s6 = adstHalfBtf16(kWn48, t6, kW16, t7)
		s7 = adstHalfBtf16(kW16, t6, kW48, t7)

		u4 := t0.SubSaturated(s4)
		u5 := t1.SubSaturated(s5)
		u6 := t2.SubSaturated(s6)
		u7 := t3.SubSaturated(s7)
		t0 = t0.AddSaturated(s4)
		t1 = t1.AddSaturated(s5)
		t2 = t2.AddSaturated(s6)
		t3 = t3.AddSaturated(s7)

		o0 = adstHalfBtf16(kW60, t0, kWn4, t1)
		o1 = adstHalfBtf16(kW52, u6, kW12, u7)
		o2 = adstHalfBtf16(kW44, t2, kWn20, t3)
		o3 = adstHalfBtf16(kW36, u4, kW28, u5)
		o4 = adstHalfBtf16(kW28, u4, kWn36, u5)
		o5 = adstHalfBtf16(kW20, t2, kW44, t3)
		o6 = adstHalfBtf16(kW12, u6, kWn52, u7)
		o7 = adstHalfBtf16(kW4, t0, kW60, t1)
	}

	// --- Store (int16 -> int32 coeff layout) ---
	obase := unsafe.Pointer(unsafe.SliceData(coeff))
	ostep := uintptr(coeffStride) * 4
	{
		lo := o0.ExtendLo4ToInt32()
		hi := o0.HiToLo().ExtendLo4ToInt32()
		lo.StoreArray((*[4]int32)(obase))
		hi.StoreArray((*[4]int32)(unsafe.Add(obase, 16)))
	}
	{
		lo := o1.ExtendLo4ToInt32()
		hi := o1.HiToLo().ExtendLo4ToInt32()
		lo.StoreArray((*[4]int32)(unsafe.Add(obase, ostep*1)))
		hi.StoreArray((*[4]int32)(unsafe.Add(unsafe.Add(obase, ostep*1), 16)))
	}
	{
		lo := o2.ExtendLo4ToInt32()
		hi := o2.HiToLo().ExtendLo4ToInt32()
		lo.StoreArray((*[4]int32)(unsafe.Add(obase, ostep*2)))
		hi.StoreArray((*[4]int32)(unsafe.Add(unsafe.Add(obase, ostep*2), 16)))
	}
	{
		lo := o3.ExtendLo4ToInt32()
		hi := o3.HiToLo().ExtendLo4ToInt32()
		lo.StoreArray((*[4]int32)(unsafe.Add(obase, ostep*3)))
		hi.StoreArray((*[4]int32)(unsafe.Add(unsafe.Add(obase, ostep*3), 16)))
	}
	{
		lo := o4.ExtendLo4ToInt32()
		hi := o4.HiToLo().ExtendLo4ToInt32()
		lo.StoreArray((*[4]int32)(unsafe.Add(obase, ostep*4)))
		hi.StoreArray((*[4]int32)(unsafe.Add(unsafe.Add(obase, ostep*4), 16)))
	}
	{
		lo := o5.ExtendLo4ToInt32()
		hi := o5.HiToLo().ExtendLo4ToInt32()
		lo.StoreArray((*[4]int32)(unsafe.Add(obase, ostep*5)))
		hi.StoreArray((*[4]int32)(unsafe.Add(unsafe.Add(obase, ostep*5), 16)))
	}
	{
		lo := o6.ExtendLo4ToInt32()
		hi := o6.HiToLo().ExtendLo4ToInt32()
		lo.StoreArray((*[4]int32)(unsafe.Add(obase, ostep*6)))
		hi.StoreArray((*[4]int32)(unsafe.Add(unsafe.Add(obase, ostep*6), 16)))
	}
	{
		lo := o7.ExtendLo4ToInt32()
		hi := o7.HiToLo().ExtendLo4ToInt32()
		lo.StoreArray((*[4]int32)(unsafe.Add(obase, ostep*7)))
		hi.StoreArray((*[4]int32)(unsafe.Add(unsafe.Add(obase, ostep*7), 16)))
	}
}

// forwardBlock8x8ADSTDCTSIMD: ADST column, common shift, transpose, DCT row.
func forwardBlock8x8ADSTDCTSIMD(coeff []int32, coeffStride int, residual []int16, residualStride int, scratch []int32) {
	_ = scratch[63]
	rbase := unsafe.Pointer(unsafe.SliceData(residual))
	rstep := uintptr(residualStride) * 2
	l0 := archsimd.LoadInt16x8Array((*[8]int16)(rbase)).ShiftAllLeftConst(2)
	l1 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(rbase, rstep*1))).ShiftAllLeftConst(2)
	l2 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(rbase, rstep*2))).ShiftAllLeftConst(2)
	l3 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(rbase, rstep*3))).ShiftAllLeftConst(2)
	l4 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(rbase, rstep*4))).ShiftAllLeftConst(2)
	l5 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(rbase, rstep*5))).ShiftAllLeftConst(2)
	l6 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(rbase, rstep*6))).ShiftAllLeftConst(2)
	l7 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(rbase, rstep*7))).ShiftAllLeftConst(2)

	// --- Column pass ---
	var c0, c1, c2, c3, c4, c5, c6, c7 archsimd.Int16x8
	{
		kW32 := archsimd.LoadInt16x8Array(&fadstHW32)
		kWn32 := archsimd.LoadInt16x8Array(&fadstHWn32)
		kW16 := archsimd.LoadInt16x8Array(&fadstHW16)
		kW48 := archsimd.LoadInt16x8Array(&fadstHW48)
		kWn16 := archsimd.LoadInt16x8Array(&fadstHWn16)
		kWn48 := archsimd.LoadInt16x8Array(&fadstHWn48)
		kW4 := archsimd.LoadInt16x8Array(&fadstHW4)
		kW60 := archsimd.LoadInt16x8Array(&fadstHW60)
		kWn4 := archsimd.LoadInt16x8Array(&fadstHWn4)
		kW20 := archsimd.LoadInt16x8Array(&fadstHW20)
		kW44 := archsimd.LoadInt16x8Array(&fadstHW44)
		kWn20 := archsimd.LoadInt16x8Array(&fadstHWn20)
		kW36 := archsimd.LoadInt16x8Array(&fadstHW36)
		kW28 := archsimd.LoadInt16x8Array(&fadstHW28)
		kWn36 := archsimd.LoadInt16x8Array(&fadstHWn36)
		kW52 := archsimd.LoadInt16x8Array(&fadstHW52)
		kW12 := archsimd.LoadInt16x8Array(&fadstHW12)
		kWn52 := archsimd.LoadInt16x8Array(&fadstHWn52)
		s0 := l0
		s1 := l7.Neg()
		s2 := adstHalfBtf16(kWn32, l3, kW32, l4)
		s3 := adstHalfBtf16(kWn32, l3, kWn32, l4)
		s4 := l1.Neg()
		s5 := l6
		s6 := adstHalfBtf16(kW32, l2, kWn32, l5)
		s7 := adstHalfBtf16(kW32, l2, kW32, l5)

		t0 := s0.AddSaturated(s2)
		t1 := s1.AddSaturated(s3)
		t2 := s0.SubSaturated(s2)
		t3 := s1.SubSaturated(s3)
		t4 := s4.AddSaturated(s6)
		t5 := s5.AddSaturated(s7)
		t6 := s4.SubSaturated(s6)
		t7 := s5.SubSaturated(s7)

		s4 = adstHalfBtf16(kW16, t4, kW48, t5)
		s5 = adstHalfBtf16(kW48, t4, kWn16, t5)
		s6 = adstHalfBtf16(kWn48, t6, kW16, t7)
		s7 = adstHalfBtf16(kW16, t6, kW48, t7)

		u4 := t0.SubSaturated(s4)
		u5 := t1.SubSaturated(s5)
		u6 := t2.SubSaturated(s6)
		u7 := t3.SubSaturated(s7)
		t0 = t0.AddSaturated(s4)
		t1 = t1.AddSaturated(s5)
		t2 = t2.AddSaturated(s6)
		t3 = t3.AddSaturated(s7)

		c0 = adstRoundShift1Int16(adstHalfBtf16(kW60, t0, kWn4, t1))
		c1 = adstRoundShift1Int16(adstHalfBtf16(kW52, u6, kW12, u7))
		c2 = adstRoundShift1Int16(adstHalfBtf16(kW44, t2, kWn20, t3))
		c3 = adstRoundShift1Int16(adstHalfBtf16(kW36, u4, kW28, u5))
		c4 = adstRoundShift1Int16(adstHalfBtf16(kW28, u4, kWn36, u5))
		c5 = adstRoundShift1Int16(adstHalfBtf16(kW20, t2, kW44, t3))
		c6 = adstRoundShift1Int16(adstHalfBtf16(kW12, u6, kWn52, u7))
		c7 = adstRoundShift1Int16(adstHalfBtf16(kW4, t0, kW60, t1))
	}

	// --- Transpose (inlined 8x8 int16) ---
	a0 := c0.InterleaveLo(c1)
	a1 := c0.InterleaveHi(c1)
	a2 := c2.InterleaveLo(c3)
	a3 := c2.InterleaveHi(c3)
	a4 := c4.InterleaveLo(c5)
	a5 := c4.InterleaveHi(c5)
	a6 := c6.InterleaveLo(c7)
	a7 := c6.InterleaveHi(c7)
	b0 := adstInt32AsInt16(adstInt16AsInt32(a0).InterleaveLo(adstInt16AsInt32(a2)))
	b1 := adstInt32AsInt16(adstInt16AsInt32(a0).InterleaveHi(adstInt16AsInt32(a2)))
	b2 := adstInt32AsInt16(adstInt16AsInt32(a1).InterleaveLo(adstInt16AsInt32(a3)))
	b3 := adstInt32AsInt16(adstInt16AsInt32(a1).InterleaveHi(adstInt16AsInt32(a3)))
	b4 := adstInt32AsInt16(adstInt16AsInt32(a4).InterleaveLo(adstInt16AsInt32(a6)))
	b5 := adstInt32AsInt16(adstInt16AsInt32(a4).InterleaveHi(adstInt16AsInt32(a6)))
	b6 := adstInt32AsInt16(adstInt16AsInt32(a5).InterleaveLo(adstInt16AsInt32(a7)))
	b7 := adstInt32AsInt16(adstInt16AsInt32(a5).InterleaveHi(adstInt16AsInt32(a7)))
	t0 := adstInt64AsInt16(adstInt16AsInt64(b0).InterleaveLo(adstInt16AsInt64(b4)))
	t1 := adstInt64AsInt16(adstInt16AsInt64(b0).InterleaveHi(adstInt16AsInt64(b4)))
	t2 := adstInt64AsInt16(adstInt16AsInt64(b1).InterleaveLo(adstInt16AsInt64(b5)))
	t3 := adstInt64AsInt16(adstInt16AsInt64(b1).InterleaveHi(adstInt16AsInt64(b5)))
	t4 := adstInt64AsInt16(adstInt16AsInt64(b2).InterleaveLo(adstInt16AsInt64(b6)))
	t5 := adstInt64AsInt16(adstInt16AsInt64(b2).InterleaveHi(adstInt16AsInt64(b6)))
	t6 := adstInt64AsInt16(adstInt16AsInt64(b3).InterleaveLo(adstInt16AsInt64(b7)))
	t7 := adstInt64AsInt16(adstInt16AsInt64(b3).InterleaveHi(adstInt16AsInt64(b7)))

	// --- Row pass ---
	var o0, o1, o2, o3, o4, o5, o6, o7 archsimd.Int16x8
	{
		dW32 := archsimd.LoadInt16x8Array(&fdctHW32)
		dWn32 := archsimd.LoadInt16x8Array(&fdctHWn32)
		dW48 := archsimd.LoadInt16x8Array(&fdctHW48)
		dW16 := archsimd.LoadInt16x8Array(&fdctHW16)
		dWn16 := archsimd.LoadInt16x8Array(&fdctHWn16)
		dW56 := archsimd.LoadInt16x8Array(&fdctHW56)
		dW8 := archsimd.LoadInt16x8Array(&fdctHW8)
		dWn8 := archsimd.LoadInt16x8Array(&fdctHWn8)
		dW24 := archsimd.LoadInt16x8Array(&fdctHW24)
		dW40 := archsimd.LoadInt16x8Array(&fdctHW40)
		dWn40 := archsimd.LoadInt16x8Array(&fdctHWn40)
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
		s5 := adstHalfBtf16(dWn32, b5, dW32, b6)
		s6 := adstHalfBtf16(dW32, b6, dW32, b5)
		s7 := b7

		p0 := adstHalfBtf16(dW32, s0, dW32, s1)
		p1 := adstHalfBtf16(dWn32, s1, dW32, s0)
		p2 := adstHalfBtf16(dW48, s2, dW16, s3)
		p3 := adstHalfBtf16(dW48, s3, dWn16, s2)
		p4 := s4.AddSaturated(s5)
		p5 := s4.SubSaturated(s5)
		p6 := s7.SubSaturated(s6)
		p7 := s7.AddSaturated(s6)

		o0 = p0
		o1 = adstHalfBtf16(dW56, p4, dW8, p7)
		o2 = p2
		o3 = adstHalfBtf16(dW24, p6, dWn40, p5)
		o4 = p1
		o5 = adstHalfBtf16(dW24, p5, dW40, p6)
		o6 = p3
		o7 = adstHalfBtf16(dW56, p7, dWn8, p4)
	}

	// --- Store (int16 -> int32 coeff layout) ---
	obase := unsafe.Pointer(unsafe.SliceData(coeff))
	ostep := uintptr(coeffStride) * 4
	{
		lo := o0.ExtendLo4ToInt32()
		hi := o0.HiToLo().ExtendLo4ToInt32()
		lo.StoreArray((*[4]int32)(obase))
		hi.StoreArray((*[4]int32)(unsafe.Add(obase, 16)))
	}
	{
		lo := o1.ExtendLo4ToInt32()
		hi := o1.HiToLo().ExtendLo4ToInt32()
		lo.StoreArray((*[4]int32)(unsafe.Add(obase, ostep*1)))
		hi.StoreArray((*[4]int32)(unsafe.Add(unsafe.Add(obase, ostep*1), 16)))
	}
	{
		lo := o2.ExtendLo4ToInt32()
		hi := o2.HiToLo().ExtendLo4ToInt32()
		lo.StoreArray((*[4]int32)(unsafe.Add(obase, ostep*2)))
		hi.StoreArray((*[4]int32)(unsafe.Add(unsafe.Add(obase, ostep*2), 16)))
	}
	{
		lo := o3.ExtendLo4ToInt32()
		hi := o3.HiToLo().ExtendLo4ToInt32()
		lo.StoreArray((*[4]int32)(unsafe.Add(obase, ostep*3)))
		hi.StoreArray((*[4]int32)(unsafe.Add(unsafe.Add(obase, ostep*3), 16)))
	}
	{
		lo := o4.ExtendLo4ToInt32()
		hi := o4.HiToLo().ExtendLo4ToInt32()
		lo.StoreArray((*[4]int32)(unsafe.Add(obase, ostep*4)))
		hi.StoreArray((*[4]int32)(unsafe.Add(unsafe.Add(obase, ostep*4), 16)))
	}
	{
		lo := o5.ExtendLo4ToInt32()
		hi := o5.HiToLo().ExtendLo4ToInt32()
		lo.StoreArray((*[4]int32)(unsafe.Add(obase, ostep*5)))
		hi.StoreArray((*[4]int32)(unsafe.Add(unsafe.Add(obase, ostep*5), 16)))
	}
	{
		lo := o6.ExtendLo4ToInt32()
		hi := o6.HiToLo().ExtendLo4ToInt32()
		lo.StoreArray((*[4]int32)(unsafe.Add(obase, ostep*6)))
		hi.StoreArray((*[4]int32)(unsafe.Add(unsafe.Add(obase, ostep*6), 16)))
	}
	{
		lo := o7.ExtendLo4ToInt32()
		hi := o7.HiToLo().ExtendLo4ToInt32()
		lo.StoreArray((*[4]int32)(unsafe.Add(obase, ostep*7)))
		hi.StoreArray((*[4]int32)(unsafe.Add(unsafe.Add(obase, ostep*7), 16)))
	}
}

// forwardBlock8x8DCTADSTSIMD: DCT column, common shift, transpose, ADST row.
func forwardBlock8x8DCTADSTSIMD(coeff []int32, coeffStride int, residual []int16, residualStride int, scratch []int32) {
	_ = scratch[63]
	rbase := unsafe.Pointer(unsafe.SliceData(residual))
	rstep := uintptr(residualStride) * 2
	l0 := archsimd.LoadInt16x8Array((*[8]int16)(rbase)).ShiftAllLeftConst(2)
	l1 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(rbase, rstep*1))).ShiftAllLeftConst(2)
	l2 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(rbase, rstep*2))).ShiftAllLeftConst(2)
	l3 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(rbase, rstep*3))).ShiftAllLeftConst(2)
	l4 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(rbase, rstep*4))).ShiftAllLeftConst(2)
	l5 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(rbase, rstep*5))).ShiftAllLeftConst(2)
	l6 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(rbase, rstep*6))).ShiftAllLeftConst(2)
	l7 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(rbase, rstep*7))).ShiftAllLeftConst(2)

	// --- Column pass ---
	var c0, c1, c2, c3, c4, c5, c6, c7 archsimd.Int16x8
	{
		dW32 := archsimd.LoadInt16x8Array(&fdctHW32)
		dWn32 := archsimd.LoadInt16x8Array(&fdctHWn32)
		dW48 := archsimd.LoadInt16x8Array(&fdctHW48)
		dW16 := archsimd.LoadInt16x8Array(&fdctHW16)
		dWn16 := archsimd.LoadInt16x8Array(&fdctHWn16)
		dW56 := archsimd.LoadInt16x8Array(&fdctHW56)
		dW8 := archsimd.LoadInt16x8Array(&fdctHW8)
		dWn8 := archsimd.LoadInt16x8Array(&fdctHWn8)
		dW24 := archsimd.LoadInt16x8Array(&fdctHW24)
		dW40 := archsimd.LoadInt16x8Array(&fdctHW40)
		dWn40 := archsimd.LoadInt16x8Array(&fdctHWn40)
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
		s5 := adstHalfBtf16(dWn32, b5, dW32, b6)
		s6 := adstHalfBtf16(dW32, b6, dW32, b5)
		s7 := b7

		p0 := adstHalfBtf16(dW32, s0, dW32, s1)
		p1 := adstHalfBtf16(dWn32, s1, dW32, s0)
		p2 := adstHalfBtf16(dW48, s2, dW16, s3)
		p3 := adstHalfBtf16(dW48, s3, dWn16, s2)
		p4 := s4.AddSaturated(s5)
		p5 := s4.SubSaturated(s5)
		p6 := s7.SubSaturated(s6)
		p7 := s7.AddSaturated(s6)

		c0 = adstRoundShift1Int16(p0)
		c1 = adstRoundShift1Int16(adstHalfBtf16(dW56, p4, dW8, p7))
		c2 = adstRoundShift1Int16(p2)
		c3 = adstRoundShift1Int16(adstHalfBtf16(dW24, p6, dWn40, p5))
		c4 = adstRoundShift1Int16(p1)
		c5 = adstRoundShift1Int16(adstHalfBtf16(dW24, p5, dW40, p6))
		c6 = adstRoundShift1Int16(p3)
		c7 = adstRoundShift1Int16(adstHalfBtf16(dW56, p7, dWn8, p4))
	}

	// --- Transpose (inlined 8x8 int16) ---
	a0 := c0.InterleaveLo(c1)
	a1 := c0.InterleaveHi(c1)
	a2 := c2.InterleaveLo(c3)
	a3 := c2.InterleaveHi(c3)
	a4 := c4.InterleaveLo(c5)
	a5 := c4.InterleaveHi(c5)
	a6 := c6.InterleaveLo(c7)
	a7 := c6.InterleaveHi(c7)
	b0 := adstInt32AsInt16(adstInt16AsInt32(a0).InterleaveLo(adstInt16AsInt32(a2)))
	b1 := adstInt32AsInt16(adstInt16AsInt32(a0).InterleaveHi(adstInt16AsInt32(a2)))
	b2 := adstInt32AsInt16(adstInt16AsInt32(a1).InterleaveLo(adstInt16AsInt32(a3)))
	b3 := adstInt32AsInt16(adstInt16AsInt32(a1).InterleaveHi(adstInt16AsInt32(a3)))
	b4 := adstInt32AsInt16(adstInt16AsInt32(a4).InterleaveLo(adstInt16AsInt32(a6)))
	b5 := adstInt32AsInt16(adstInt16AsInt32(a4).InterleaveHi(adstInt16AsInt32(a6)))
	b6 := adstInt32AsInt16(adstInt16AsInt32(a5).InterleaveLo(adstInt16AsInt32(a7)))
	b7 := adstInt32AsInt16(adstInt16AsInt32(a5).InterleaveHi(adstInt16AsInt32(a7)))
	t0 := adstInt64AsInt16(adstInt16AsInt64(b0).InterleaveLo(adstInt16AsInt64(b4)))
	t1 := adstInt64AsInt16(adstInt16AsInt64(b0).InterleaveHi(adstInt16AsInt64(b4)))
	t2 := adstInt64AsInt16(adstInt16AsInt64(b1).InterleaveLo(adstInt16AsInt64(b5)))
	t3 := adstInt64AsInt16(adstInt16AsInt64(b1).InterleaveHi(adstInt16AsInt64(b5)))
	t4 := adstInt64AsInt16(adstInt16AsInt64(b2).InterleaveLo(adstInt16AsInt64(b6)))
	t5 := adstInt64AsInt16(adstInt16AsInt64(b2).InterleaveHi(adstInt16AsInt64(b6)))
	t6 := adstInt64AsInt16(adstInt16AsInt64(b3).InterleaveLo(adstInt16AsInt64(b7)))
	t7 := adstInt64AsInt16(adstInt16AsInt64(b3).InterleaveHi(adstInt16AsInt64(b7)))

	// --- Row pass ---
	var o0, o1, o2, o3, o4, o5, o6, o7 archsimd.Int16x8
	{
		kW32 := archsimd.LoadInt16x8Array(&fadstHW32)
		kWn32 := archsimd.LoadInt16x8Array(&fadstHWn32)
		kW16 := archsimd.LoadInt16x8Array(&fadstHW16)
		kW48 := archsimd.LoadInt16x8Array(&fadstHW48)
		kWn16 := archsimd.LoadInt16x8Array(&fadstHWn16)
		kWn48 := archsimd.LoadInt16x8Array(&fadstHWn48)
		kW4 := archsimd.LoadInt16x8Array(&fadstHW4)
		kW60 := archsimd.LoadInt16x8Array(&fadstHW60)
		kWn4 := archsimd.LoadInt16x8Array(&fadstHWn4)
		kW20 := archsimd.LoadInt16x8Array(&fadstHW20)
		kW44 := archsimd.LoadInt16x8Array(&fadstHW44)
		kWn20 := archsimd.LoadInt16x8Array(&fadstHWn20)
		kW36 := archsimd.LoadInt16x8Array(&fadstHW36)
		kW28 := archsimd.LoadInt16x8Array(&fadstHW28)
		kWn36 := archsimd.LoadInt16x8Array(&fadstHWn36)
		kW52 := archsimd.LoadInt16x8Array(&fadstHW52)
		kW12 := archsimd.LoadInt16x8Array(&fadstHW12)
		kWn52 := archsimd.LoadInt16x8Array(&fadstHWn52)
		s0 := t0
		s1 := t7.Neg()
		s2 := adstHalfBtf16(kWn32, t3, kW32, t4)
		s3 := adstHalfBtf16(kWn32, t3, kWn32, t4)
		s4 := t1.Neg()
		s5 := t6
		s6 := adstHalfBtf16(kW32, t2, kWn32, t5)
		s7 := adstHalfBtf16(kW32, t2, kW32, t5)

		t0 := s0.AddSaturated(s2)
		t1 := s1.AddSaturated(s3)
		t2 := s0.SubSaturated(s2)
		t3 := s1.SubSaturated(s3)
		t4 := s4.AddSaturated(s6)
		t5 := s5.AddSaturated(s7)
		t6 := s4.SubSaturated(s6)
		t7 := s5.SubSaturated(s7)

		s4 = adstHalfBtf16(kW16, t4, kW48, t5)
		s5 = adstHalfBtf16(kW48, t4, kWn16, t5)
		s6 = adstHalfBtf16(kWn48, t6, kW16, t7)
		s7 = adstHalfBtf16(kW16, t6, kW48, t7)

		u4 := t0.SubSaturated(s4)
		u5 := t1.SubSaturated(s5)
		u6 := t2.SubSaturated(s6)
		u7 := t3.SubSaturated(s7)
		t0 = t0.AddSaturated(s4)
		t1 = t1.AddSaturated(s5)
		t2 = t2.AddSaturated(s6)
		t3 = t3.AddSaturated(s7)

		o0 = adstHalfBtf16(kW60, t0, kWn4, t1)
		o1 = adstHalfBtf16(kW52, u6, kW12, u7)
		o2 = adstHalfBtf16(kW44, t2, kWn20, t3)
		o3 = adstHalfBtf16(kW36, u4, kW28, u5)
		o4 = adstHalfBtf16(kW28, u4, kWn36, u5)
		o5 = adstHalfBtf16(kW20, t2, kW44, t3)
		o6 = adstHalfBtf16(kW12, u6, kWn52, u7)
		o7 = adstHalfBtf16(kW4, t0, kW60, t1)
	}

	// --- Store (int16 -> int32 coeff layout) ---
	obase := unsafe.Pointer(unsafe.SliceData(coeff))
	ostep := uintptr(coeffStride) * 4
	{
		lo := o0.ExtendLo4ToInt32()
		hi := o0.HiToLo().ExtendLo4ToInt32()
		lo.StoreArray((*[4]int32)(obase))
		hi.StoreArray((*[4]int32)(unsafe.Add(obase, 16)))
	}
	{
		lo := o1.ExtendLo4ToInt32()
		hi := o1.HiToLo().ExtendLo4ToInt32()
		lo.StoreArray((*[4]int32)(unsafe.Add(obase, ostep*1)))
		hi.StoreArray((*[4]int32)(unsafe.Add(unsafe.Add(obase, ostep*1), 16)))
	}
	{
		lo := o2.ExtendLo4ToInt32()
		hi := o2.HiToLo().ExtendLo4ToInt32()
		lo.StoreArray((*[4]int32)(unsafe.Add(obase, ostep*2)))
		hi.StoreArray((*[4]int32)(unsafe.Add(unsafe.Add(obase, ostep*2), 16)))
	}
	{
		lo := o3.ExtendLo4ToInt32()
		hi := o3.HiToLo().ExtendLo4ToInt32()
		lo.StoreArray((*[4]int32)(unsafe.Add(obase, ostep*3)))
		hi.StoreArray((*[4]int32)(unsafe.Add(unsafe.Add(obase, ostep*3), 16)))
	}
	{
		lo := o4.ExtendLo4ToInt32()
		hi := o4.HiToLo().ExtendLo4ToInt32()
		lo.StoreArray((*[4]int32)(unsafe.Add(obase, ostep*4)))
		hi.StoreArray((*[4]int32)(unsafe.Add(unsafe.Add(obase, ostep*4), 16)))
	}
	{
		lo := o5.ExtendLo4ToInt32()
		hi := o5.HiToLo().ExtendLo4ToInt32()
		lo.StoreArray((*[4]int32)(unsafe.Add(obase, ostep*5)))
		hi.StoreArray((*[4]int32)(unsafe.Add(unsafe.Add(obase, ostep*5), 16)))
	}
	{
		lo := o6.ExtendLo4ToInt32()
		hi := o6.HiToLo().ExtendLo4ToInt32()
		lo.StoreArray((*[4]int32)(unsafe.Add(obase, ostep*6)))
		hi.StoreArray((*[4]int32)(unsafe.Add(unsafe.Add(obase, ostep*6), 16)))
	}
	{
		lo := o7.ExtendLo4ToInt32()
		hi := o7.HiToLo().ExtendLo4ToInt32()
		lo.StoreArray((*[4]int32)(unsafe.Add(obase, ostep*7)))
		hi.StoreArray((*[4]int32)(unsafe.Add(unsafe.Add(obase, ostep*7), 16)))
	}
}

// forwardBlock8x8IDTXSIMD is the IDTX identity scale (coeff[c*stride+r]=res[r][c]<<3).
func forwardBlock8x8IDTXSIMD(coeff []int32, coeffStride int, residual []int16, residualStride int, scratch []int32) {
	_ = scratch[63]
	rbase := unsafe.Pointer(unsafe.SliceData(residual))
	rstep := uintptr(residualStride) * 2
	l0 := archsimd.LoadInt16x8Array((*[8]int16)(rbase)).ShiftAllLeftConst(3)
	l1 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(rbase, rstep*1))).ShiftAllLeftConst(3)
	l2 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(rbase, rstep*2))).ShiftAllLeftConst(3)
	l3 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(rbase, rstep*3))).ShiftAllLeftConst(3)
	l4 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(rbase, rstep*4))).ShiftAllLeftConst(3)
	l5 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(rbase, rstep*5))).ShiftAllLeftConst(3)
	l6 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(rbase, rstep*6))).ShiftAllLeftConst(3)
	l7 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(rbase, rstep*7))).ShiftAllLeftConst(3)

	// Transpose the scaled residual and store in coeff layout.
	a0 := l0.InterleaveLo(l1)
	a1 := l0.InterleaveHi(l1)
	a2 := l2.InterleaveLo(l3)
	a3 := l2.InterleaveHi(l3)
	a4 := l4.InterleaveLo(l5)
	a5 := l4.InterleaveHi(l5)
	a6 := l6.InterleaveLo(l7)
	a7 := l6.InterleaveHi(l7)
	b0 := adstInt32AsInt16(adstInt16AsInt32(a0).InterleaveLo(adstInt16AsInt32(a2)))
	b1 := adstInt32AsInt16(adstInt16AsInt32(a0).InterleaveHi(adstInt16AsInt32(a2)))
	b2 := adstInt32AsInt16(adstInt16AsInt32(a1).InterleaveLo(adstInt16AsInt32(a3)))
	b3 := adstInt32AsInt16(adstInt16AsInt32(a1).InterleaveHi(adstInt16AsInt32(a3)))
	b4 := adstInt32AsInt16(adstInt16AsInt32(a4).InterleaveLo(adstInt16AsInt32(a6)))
	b5 := adstInt32AsInt16(adstInt16AsInt32(a4).InterleaveHi(adstInt16AsInt32(a6)))
	b6 := adstInt32AsInt16(adstInt16AsInt32(a5).InterleaveLo(adstInt16AsInt32(a7)))
	b7 := adstInt32AsInt16(adstInt16AsInt32(a5).InterleaveHi(adstInt16AsInt32(a7)))
	o0 := adstInt64AsInt16(adstInt16AsInt64(b0).InterleaveLo(adstInt16AsInt64(b4)))
	o1 := adstInt64AsInt16(adstInt16AsInt64(b0).InterleaveHi(adstInt16AsInt64(b4)))
	o2 := adstInt64AsInt16(adstInt16AsInt64(b1).InterleaveLo(adstInt16AsInt64(b5)))
	o3 := adstInt64AsInt16(adstInt16AsInt64(b1).InterleaveHi(adstInt16AsInt64(b5)))
	o4 := adstInt64AsInt16(adstInt16AsInt64(b2).InterleaveLo(adstInt16AsInt64(b6)))
	o5 := adstInt64AsInt16(adstInt16AsInt64(b2).InterleaveHi(adstInt16AsInt64(b6)))
	o6 := adstInt64AsInt16(adstInt16AsInt64(b3).InterleaveLo(adstInt16AsInt64(b7)))
	o7 := adstInt64AsInt16(adstInt16AsInt64(b3).InterleaveHi(adstInt16AsInt64(b7)))

	obase := unsafe.Pointer(unsafe.SliceData(coeff))
	ostep := uintptr(coeffStride) * 4
	{
		lo := o0.ExtendLo4ToInt32()
		hi := o0.HiToLo().ExtendLo4ToInt32()
		lo.StoreArray((*[4]int32)(obase))
		hi.StoreArray((*[4]int32)(unsafe.Add(obase, 16)))
	}
	{
		lo := o1.ExtendLo4ToInt32()
		hi := o1.HiToLo().ExtendLo4ToInt32()
		lo.StoreArray((*[4]int32)(unsafe.Add(obase, ostep*1)))
		hi.StoreArray((*[4]int32)(unsafe.Add(unsafe.Add(obase, ostep*1), 16)))
	}
	{
		lo := o2.ExtendLo4ToInt32()
		hi := o2.HiToLo().ExtendLo4ToInt32()
		lo.StoreArray((*[4]int32)(unsafe.Add(obase, ostep*2)))
		hi.StoreArray((*[4]int32)(unsafe.Add(unsafe.Add(obase, ostep*2), 16)))
	}
	{
		lo := o3.ExtendLo4ToInt32()
		hi := o3.HiToLo().ExtendLo4ToInt32()
		lo.StoreArray((*[4]int32)(unsafe.Add(obase, ostep*3)))
		hi.StoreArray((*[4]int32)(unsafe.Add(unsafe.Add(obase, ostep*3), 16)))
	}
	{
		lo := o4.ExtendLo4ToInt32()
		hi := o4.HiToLo().ExtendLo4ToInt32()
		lo.StoreArray((*[4]int32)(unsafe.Add(obase, ostep*4)))
		hi.StoreArray((*[4]int32)(unsafe.Add(unsafe.Add(obase, ostep*4), 16)))
	}
	{
		lo := o5.ExtendLo4ToInt32()
		hi := o5.HiToLo().ExtendLo4ToInt32()
		lo.StoreArray((*[4]int32)(unsafe.Add(obase, ostep*5)))
		hi.StoreArray((*[4]int32)(unsafe.Add(unsafe.Add(obase, ostep*5), 16)))
	}
	{
		lo := o6.ExtendLo4ToInt32()
		hi := o6.HiToLo().ExtendLo4ToInt32()
		lo.StoreArray((*[4]int32)(unsafe.Add(obase, ostep*6)))
		hi.StoreArray((*[4]int32)(unsafe.Add(unsafe.Add(obase, ostep*6), 16)))
	}
	{
		lo := o7.ExtendLo4ToInt32()
		hi := o7.HiToLo().ExtendLo4ToInt32()
		lo.StoreArray((*[4]int32)(unsafe.Add(obase, ostep*7)))
		hi.StoreArray((*[4]int32)(unsafe.Add(unsafe.Add(obase, ostep*7), 16)))
	}
}
