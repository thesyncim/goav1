// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

// Go-native SIMD 8-bit-dst CDEF secondary-only block filters (the
// pri_strength == 0 split, dav1d cdef_filter{8,4}_sec_neon). This is a
// deliberate re-test of the old "CDEF can't beat asm from Go" pin: the uint16
// block kernel (filter_gosimd_arm64.go) already beats its asm after the
// immediate-shift/hoisting fixes, and the u8 secondary kernel's data movement
// is entirely contiguous uint16x8 loads (one per tap row), not the strided
// gathers that motivated the pin.
//
// Structure vs cdefFilterBlock{8,4}SecondaryU8NEON (filter_u8_neon_arm64_split.s):
//   - the same clamp-form constrain the asm uses (uabd, ushl by the hoisted
//     negative damping shift, uqsub against the strength for the built-in
//     max(0,..), then clamp diff to [-lim, +lim]) — one op cheaper than the
//     xor-sign form;
//   - the asm accumulates all eight taps with serial MLAs into one register
//     (a ~8x mla-latency dependency chain per row); here the eight constrain
//     results land in four independent Add accumulators, two per tap weight,
//     and the {2,1} secondary tap weights (cdefSecondaryTaps, compile-time
//     constants) fold into one ShiftAllLeftConst(1)+Add at the end instead of
//     eight MLAs;
//   - the 4-wide kernel processes two rows per vector (the two 4-lane rows
//     zip1'd on 64-bit lanes), like the asm's d-register pairs.
//
// Byte-exactness with filterBlockU8PureGo: constrainShifted is reproduced
// lane-wise (identical to the asm's arithmetic, proven by the existing
// differential corpus); the finalize is x + ((8 + sum - (sum<0)) >> 4) with
// the (sum<0) term via an arithmetic ShiftAllRightConst(15). The secondary-
// only split never clips, so no min/max tracking exists. VeryLarge (0x4000)
// halo sentinels produce |diff| large enough that the uqsub saturates the
// limit to zero, exactly as in the reference.

package cdef

import (
	"simd/archsimd"
	"unsafe"
)

// dispatchFilterBlockU8NEON is the goexperiment.simd router: every strength
// split (primary-only, secondary-only, fused) runs the Go-native SIMD kernels
// below. The stock router lives in filter_u8_route_arm64.go.
func dispatchFilterBlockU8NEON(ctx *filterBlockU8NEONCtx, width int, primaryStrength int, secondaryStrength int) {
	if width == 8 {
		switch {
		case primaryStrength != 0 && secondaryStrength == 0:
			cdefFilterBlock8PrimaryU8SIMD(ctx)
		case primaryStrength == 0 && secondaryStrength != 0:
			cdefFilterBlock8SecondaryU8SIMD(ctx)
		default:
			cdefFilterBlock8U8SIMD(ctx)
		}
		return
	}
	switch {
	case primaryStrength != 0 && secondaryStrength == 0:
		cdefFilterBlock4PrimaryU8SIMD(ctx)
	case primaryStrength == 0 && secondaryStrength != 0:
		cdefFilterBlock4SecondaryU8SIMD(ctx)
	default:
		cdefFilterBlock4U8SIMD(ctx)
	}
}

// cdefLoadU16P loads 8 uint16 CDEF samples at a raw pointer as Int16x8
// (samples <= 0x4000, the bit pattern is a non-negative int16).
func cdefLoadU16P(p unsafe.Pointer) archsimd.Int16x8 {
	return archsimd.LoadInt16x8Array((*[8]int16)(p))
}

// cdefFilterBlock8SecondaryU8SIMD is the 8-wide secondary-only kernel.
// The eight tap chains are pasted inline so everything stays
// register-resident; each chain is
//   lim = uqsub(str, |t-x| >> shift);  c = clamp(t-x, -lim, +lim)
// and the four weight-2 results / four weight-1 results accumulate into
// independent pairs.
func cdefFilterBlock8SecondaryU8SIMD(ctx *filterBlockU8NEONCtx) {
	strU := archsimd.BroadcastInt16x8(int16(ctx.secStrength)).ToBits()
	shV := archsimd.BroadcastInt16x8(int16(-ctx.secShift))
	eight := archsimd.BroadcastInt16x8(8)
	// Tap offsets in BYTES within the uint16 input buffer.
	o0 := int(ctx.sec0) * 2
	o1 := int(ctx.sec1) * 2
	o2 := int(ctx.sec2) * 2
	o3 := int(ctx.sec3) * 2
	src := unsafe.Pointer(ctx.input)
	dst := unsafe.Pointer(ctx.dst)
	dstStr := int(ctx.dstStr)
	for h := int(ctx.height); h > 0; h-- {
		x := cdefLoadU16P(src)

		t0 := cdefLoadU16P(unsafe.Add(src, o0))
		t1 := cdefLoadU16P(unsafe.Add(src, -o0))
		t2 := cdefLoadU16P(unsafe.Add(src, o1))
		t3 := cdefLoadU16P(unsafe.Add(src, -o1))
		t4 := cdefLoadU16P(unsafe.Add(src, o2))
		t5 := cdefLoadU16P(unsafe.Add(src, -o2))
		t6 := cdefLoadU16P(unsafe.Add(src, o3))
		t7 := cdefLoadU16P(unsafe.Add(src, -o3))

		l0 := strU.SubSaturated(t0.AbsDiff(x).ToBits().Shift(shV)).BitsToInt16()
		a20 := t0.Sub(x).Min(l0).Max(l0.Neg())
		l1 := strU.SubSaturated(t1.AbsDiff(x).ToBits().Shift(shV)).BitsToInt16()
		a21 := t1.Sub(x).Min(l1).Max(l1.Neg())
		l2 := strU.SubSaturated(t2.AbsDiff(x).ToBits().Shift(shV)).BitsToInt16()
		a20 = a20.Add(t2.Sub(x).Min(l2).Max(l2.Neg()))
		l3 := strU.SubSaturated(t3.AbsDiff(x).ToBits().Shift(shV)).BitsToInt16()
		a21 = a21.Add(t3.Sub(x).Min(l3).Max(l3.Neg()))
		l4 := strU.SubSaturated(t4.AbsDiff(x).ToBits().Shift(shV)).BitsToInt16()
		a10 := t4.Sub(x).Min(l4).Max(l4.Neg())
		l5 := strU.SubSaturated(t5.AbsDiff(x).ToBits().Shift(shV)).BitsToInt16()
		a11 := t5.Sub(x).Min(l5).Max(l5.Neg())
		l6 := strU.SubSaturated(t6.AbsDiff(x).ToBits().Shift(shV)).BitsToInt16()
		a10 = a10.Add(t6.Sub(x).Min(l6).Max(l6.Neg()))
		l7 := strU.SubSaturated(t7.AbsDiff(x).ToBits().Shift(shV)).BitsToInt16()
		a11 = a11.Add(t7.Sub(x).Min(l7).Max(l7.Neg()))

		// sum = 2*(weight-2 taps) + (weight-1 taps); cdefSecondaryTaps is {2,1}.
		sum := a20.Add(a21).ShiftAllLeftConst(1).Add(a10.Add(a11))
		// y = x + ((8 + sum - (sum<0)) >> 4)
		neg := sum.ShiftAllRightConst(15)
		y := x.Add(sum.Add(neg).Add(eight).ShiftAllRightConst(4))
		*(*float64)(dst) = y.SaturateToUint8().ReshapeToFloat64x2().GetElem(0)

		src = unsafe.Add(src, BStride*2)
		dst = unsafe.Add(dst, dstStr)
	}
}

// cdefLoadPairU16P zips the low four uint16 samples of two rows into one
// Int16x8 (lanes 0..3 = row r, lanes 4..7 = row r+1), the vector shape the
// 4-wide kernel filters two rows at a time with. The full-width loads read
// four halo samples past each 4-wide row segment; the CDEF input buffer's
// 8-column horizontal border keeps them in bounds.
func cdefLoadPairU16P(p, q unsafe.Pointer) archsimd.Int16x8 {
	lo := archsimd.LoadInt16x8Array((*[8]int16)(p)).ToBits().ReshapeToUint64s()
	hi := archsimd.LoadInt16x8Array((*[8]int16)(q)).ToBits().ReshapeToUint64s()
	return lo.InterleaveLo(hi).ReshapeToUint16s().BitsToInt16()
}

// cdefFilterBlock4SecondaryU8SIMD is the 4-wide secondary-only kernel: two
// rows per Int16x8 via cdefLoadPairU16P, identical arithmetic to the 8-wide
// kernel, and the narrowed result split into two 4-byte stores. Height is
// even by dispatch contract (the NEON wrapper routes odd heights to pure Go).
func cdefFilterBlock4SecondaryU8SIMD(ctx *filterBlockU8NEONCtx) {
	strU := archsimd.BroadcastInt16x8(int16(ctx.secStrength)).ToBits()
	shV := archsimd.BroadcastInt16x8(int16(-ctx.secShift))
	eight := archsimd.BroadcastInt16x8(8)
	o0 := int(ctx.sec0) * 2
	o1 := int(ctx.sec1) * 2
	o2 := int(ctx.sec2) * 2
	o3 := int(ctx.sec3) * 2
	src := unsafe.Pointer(ctx.input)
	dst := unsafe.Pointer(ctx.dst)
	dstStr := int(ctx.dstStr)
	for h := int(ctx.height); h > 0; h -= 2 {
		src2 := unsafe.Add(src, BStride*2)
		x := cdefLoadPairU16P(src, src2)

		t0 := cdefLoadPairU16P(unsafe.Add(src, o0), unsafe.Add(src2, o0))
		t1 := cdefLoadPairU16P(unsafe.Add(src, -o0), unsafe.Add(src2, -o0))
		t2 := cdefLoadPairU16P(unsafe.Add(src, o1), unsafe.Add(src2, o1))
		t3 := cdefLoadPairU16P(unsafe.Add(src, -o1), unsafe.Add(src2, -o1))
		t4 := cdefLoadPairU16P(unsafe.Add(src, o2), unsafe.Add(src2, o2))
		t5 := cdefLoadPairU16P(unsafe.Add(src, -o2), unsafe.Add(src2, -o2))
		t6 := cdefLoadPairU16P(unsafe.Add(src, o3), unsafe.Add(src2, o3))
		t7 := cdefLoadPairU16P(unsafe.Add(src, -o3), unsafe.Add(src2, -o3))

		l0 := strU.SubSaturated(t0.AbsDiff(x).ToBits().Shift(shV)).BitsToInt16()
		a20 := t0.Sub(x).Min(l0).Max(l0.Neg())
		l1 := strU.SubSaturated(t1.AbsDiff(x).ToBits().Shift(shV)).BitsToInt16()
		a21 := t1.Sub(x).Min(l1).Max(l1.Neg())
		l2 := strU.SubSaturated(t2.AbsDiff(x).ToBits().Shift(shV)).BitsToInt16()
		a20 = a20.Add(t2.Sub(x).Min(l2).Max(l2.Neg()))
		l3 := strU.SubSaturated(t3.AbsDiff(x).ToBits().Shift(shV)).BitsToInt16()
		a21 = a21.Add(t3.Sub(x).Min(l3).Max(l3.Neg()))
		l4 := strU.SubSaturated(t4.AbsDiff(x).ToBits().Shift(shV)).BitsToInt16()
		a10 := t4.Sub(x).Min(l4).Max(l4.Neg())
		l5 := strU.SubSaturated(t5.AbsDiff(x).ToBits().Shift(shV)).BitsToInt16()
		a11 := t5.Sub(x).Min(l5).Max(l5.Neg())
		l6 := strU.SubSaturated(t6.AbsDiff(x).ToBits().Shift(shV)).BitsToInt16()
		a10 = a10.Add(t6.Sub(x).Min(l6).Max(l6.Neg()))
		l7 := strU.SubSaturated(t7.AbsDiff(x).ToBits().Shift(shV)).BitsToInt16()
		a11 = a11.Add(t7.Sub(x).Min(l7).Max(l7.Neg()))

		sum := a20.Add(a21).ShiftAllLeftConst(1).Add(a10.Add(a11))
		neg := sum.ShiftAllRightConst(15)
		y := x.Add(sum.Add(neg).Add(eight).ShiftAllRightConst(4))
		out := y.SaturateToUint8().ReshapeToUint32s()
		*(*uint32)(dst) = out.GetElem(0)
		*(*uint32)(unsafe.Add(dst, dstStr)) = out.GetElem(1)

		src = unsafe.Add(src, 2*BStride*2)
		dst = unsafe.Add(dst, 2*dstStr)
	}
}

// --- primary-only kernels ------------------------------------------------------

// cdefFilterBlock8PrimaryU8SIMD is the 8-wide primary-only kernel (dav1d
// cdef_filter8_pri_neon). Four taps at +-pri0/+-pri1; the per-strength primary
// tap weights ({4,2} or {3,3}, chosen by strength parity) are broadcast once
// and applied as two Muls on the per-pair accumulators instead of four serial
// MLAs. No clipping in this split.
func cdefFilterBlock8PrimaryU8SIMD(ctx *filterBlockU8NEONCtx) {
	strU := archsimd.BroadcastInt16x8(int16(ctx.priStrength)).ToBits()
	shV := archsimd.BroadcastInt16x8(int16(-ctx.priShift))
	tap0V := archsimd.BroadcastInt16x8(int16(ctx.priTap0))
	tap1V := archsimd.BroadcastInt16x8(int16(ctx.priTap1))
	eight := archsimd.BroadcastInt16x8(8)
	o0 := int(ctx.pri0) * 2
	o1 := int(ctx.pri1) * 2
	src := unsafe.Pointer(ctx.input)
	dst := unsafe.Pointer(ctx.dst)
	dstStr := int(ctx.dstStr)
	for h := int(ctx.height); h > 0; h-- {
		x := cdefLoadU16P(src)
		t0 := cdefLoadU16P(unsafe.Add(src, o0))
		t1 := cdefLoadU16P(unsafe.Add(src, -o0))
		t2 := cdefLoadU16P(unsafe.Add(src, o1))
		t3 := cdefLoadU16P(unsafe.Add(src, -o1))

		l0 := strU.SubSaturated(t0.AbsDiff(x).ToBits().Shift(shV)).BitsToInt16()
		a0 := t0.Sub(x).Min(l0).Max(l0.Neg())
		l1 := strU.SubSaturated(t1.AbsDiff(x).ToBits().Shift(shV)).BitsToInt16()
		a1 := t1.Sub(x).Min(l1).Max(l1.Neg())
		l2 := strU.SubSaturated(t2.AbsDiff(x).ToBits().Shift(shV)).BitsToInt16()
		b0 := t2.Sub(x).Min(l2).Max(l2.Neg())
		l3 := strU.SubSaturated(t3.AbsDiff(x).ToBits().Shift(shV)).BitsToInt16()
		b1 := t3.Sub(x).Min(l3).Max(l3.Neg())

		sum := a0.Add(a1).Mul(tap0V).Add(b0.Add(b1).Mul(tap1V))
		neg := sum.ShiftAllRightConst(15)
		y := x.Add(sum.Add(neg).Add(eight).ShiftAllRightConst(4))
		*(*float64)(dst) = y.SaturateToUint8().ReshapeToFloat64x2().GetElem(0)

		src = unsafe.Add(src, BStride*2)
		dst = unsafe.Add(dst, dstStr)
	}
}

// cdefFilterBlock4PrimaryU8SIMD is the 4-wide primary-only kernel: two rows
// per vector via cdefLoadPairU16P, arithmetic as in the 8-wide form.
func cdefFilterBlock4PrimaryU8SIMD(ctx *filterBlockU8NEONCtx) {
	strU := archsimd.BroadcastInt16x8(int16(ctx.priStrength)).ToBits()
	shV := archsimd.BroadcastInt16x8(int16(-ctx.priShift))
	tap0V := archsimd.BroadcastInt16x8(int16(ctx.priTap0))
	tap1V := archsimd.BroadcastInt16x8(int16(ctx.priTap1))
	eight := archsimd.BroadcastInt16x8(8)
	o0 := int(ctx.pri0) * 2
	o1 := int(ctx.pri1) * 2
	src := unsafe.Pointer(ctx.input)
	dst := unsafe.Pointer(ctx.dst)
	dstStr := int(ctx.dstStr)
	for h := int(ctx.height); h > 0; h -= 2 {
		src2 := unsafe.Add(src, BStride*2)
		x := cdefLoadPairU16P(src, src2)
		t0 := cdefLoadPairU16P(unsafe.Add(src, o0), unsafe.Add(src2, o0))
		t1 := cdefLoadPairU16P(unsafe.Add(src, -o0), unsafe.Add(src2, -o0))
		t2 := cdefLoadPairU16P(unsafe.Add(src, o1), unsafe.Add(src2, o1))
		t3 := cdefLoadPairU16P(unsafe.Add(src, -o1), unsafe.Add(src2, -o1))

		l0 := strU.SubSaturated(t0.AbsDiff(x).ToBits().Shift(shV)).BitsToInt16()
		a0 := t0.Sub(x).Min(l0).Max(l0.Neg())
		l1 := strU.SubSaturated(t1.AbsDiff(x).ToBits().Shift(shV)).BitsToInt16()
		a1 := t1.Sub(x).Min(l1).Max(l1.Neg())
		l2 := strU.SubSaturated(t2.AbsDiff(x).ToBits().Shift(shV)).BitsToInt16()
		b0 := t2.Sub(x).Min(l2).Max(l2.Neg())
		l3 := strU.SubSaturated(t3.AbsDiff(x).ToBits().Shift(shV)).BitsToInt16()
		b1 := t3.Sub(x).Min(l3).Max(l3.Neg())

		sum := a0.Add(a1).Mul(tap0V).Add(b0.Add(b1).Mul(tap1V))
		neg := sum.ShiftAllRightConst(15)
		y := x.Add(sum.Add(neg).Add(eight).ShiftAllRightConst(4))
		out := y.SaturateToUint8().ReshapeToUint32s()
		*(*uint32)(dst) = out.GetElem(0)
		*(*uint32)(unsafe.Add(dst, dstStr)) = out.GetElem(1)

		src = unsafe.Add(src, 2*BStride*2)
		dst = unsafe.Add(dst, 2*dstStr)
	}
}

// --- fused primary+secondary kernels -------------------------------------------

// cdefFilterBlock8U8SIMD is the 8-wide fused kernel (dav1d
// cdef_filter8_pri_sec_neon). Twelve taps, two constrain parameter sets, and
// the clip range: min is a plain unsigned Min tree (the VeryLarge sentinel
// never wins a min), max uses the asm's xor-domain trick — taps are XORed
// with 0x4000 so sentinels map to 0 and reals to 0x4000+r, unsigned-Maxed,
// and the tree root is XORed back and Maxed with the center pixel. Where the
// asm folds all twelve weighted taps into one serial MLA chain plus serial
// umin/umax chains, here the sum uses four accumulators (pri pairs by tap
// weight, sec pairs by tap weight), and min/max use two-accumulator trees, so
// no chain is longer than a few ops.
func cdefFilterBlock8U8SIMD(ctx *filterBlockU8NEONCtx) {
	priStrU := archsimd.BroadcastInt16x8(int16(ctx.priStrength)).ToBits()
	secStrU := archsimd.BroadcastInt16x8(int16(ctx.secStrength)).ToBits()
	priShV := archsimd.BroadcastInt16x8(int16(-ctx.priShift))
	secShV := archsimd.BroadcastInt16x8(int16(-ctx.secShift))
	tap0V := archsimd.BroadcastInt16x8(int16(ctx.priTap0))
	tap1V := archsimd.BroadcastInt16x8(int16(ctx.priTap1))
	eight := archsimd.BroadcastInt16x8(8)
	vlU := archsimd.BroadcastInt16x8(VeryLarge).ToBits()
	p0 := int(ctx.pri0) * 2
	p1 := int(ctx.pri1) * 2
	o0 := int(ctx.sec0) * 2
	o1 := int(ctx.sec1) * 2
	o2 := int(ctx.sec2) * 2
	o3 := int(ctx.sec3) * 2
	src := unsafe.Pointer(ctx.input)
	dst := unsafe.Pointer(ctx.dst)
	dstStr := int(ctx.dstStr)
	for h := int(ctx.height); h > 0; h-- {
		x := cdefLoadU16P(src)

		// primary taps
		t0 := cdefLoadU16P(unsafe.Add(src, p0))
		t1 := cdefLoadU16P(unsafe.Add(src, -p0))
		t2 := cdefLoadU16P(unsafe.Add(src, p1))
		t3 := cdefLoadU16P(unsafe.Add(src, -p1))
		mn0 := x.ToBits().Min(t0.ToBits())
		mn1 := t1.ToBits().Min(t2.ToBits())
		mx0 := t0.ToBits().Xor(vlU).Max(t1.ToBits().Xor(vlU))
		mx1 := t2.ToBits().Xor(vlU).Max(t3.ToBits().Xor(vlU))
		mn0 = mn0.Min(t3.ToBits())
		l0 := priStrU.SubSaturated(t0.AbsDiff(x).ToBits().Shift(priShV)).BitsToInt16()
		pa := t0.Sub(x).Min(l0).Max(l0.Neg())
		l1 := priStrU.SubSaturated(t1.AbsDiff(x).ToBits().Shift(priShV)).BitsToInt16()
		pa = pa.Add(t1.Sub(x).Min(l1).Max(l1.Neg()))
		l2 := priStrU.SubSaturated(t2.AbsDiff(x).ToBits().Shift(priShV)).BitsToInt16()
		pb := t2.Sub(x).Min(l2).Max(l2.Neg())
		l3 := priStrU.SubSaturated(t3.AbsDiff(x).ToBits().Shift(priShV)).BitsToInt16()
		pb = pb.Add(t3.Sub(x).Min(l3).Max(l3.Neg()))

		// secondary taps
		s0 := cdefLoadU16P(unsafe.Add(src, o0))
		s1 := cdefLoadU16P(unsafe.Add(src, -o0))
		s2 := cdefLoadU16P(unsafe.Add(src, o1))
		s3 := cdefLoadU16P(unsafe.Add(src, -o1))
		s4 := cdefLoadU16P(unsafe.Add(src, o2))
		s5 := cdefLoadU16P(unsafe.Add(src, -o2))
		s6 := cdefLoadU16P(unsafe.Add(src, o3))
		s7 := cdefLoadU16P(unsafe.Add(src, -o3))
		mn0 = mn0.Min(s0.ToBits()).Min(s2.ToBits()).Min(s4.ToBits()).Min(s6.ToBits())
		mn1 = mn1.Min(s1.ToBits()).Min(s3.ToBits()).Min(s5.ToBits()).Min(s7.ToBits())
		mx0 = mx0.Max(s0.ToBits().Xor(vlU)).Max(s2.ToBits().Xor(vlU)).Max(s4.ToBits().Xor(vlU)).Max(s6.ToBits().Xor(vlU))
		mx1 = mx1.Max(s1.ToBits().Xor(vlU)).Max(s3.ToBits().Xor(vlU)).Max(s5.ToBits().Xor(vlU)).Max(s7.ToBits().Xor(vlU))

		k0 := secStrU.SubSaturated(s0.AbsDiff(x).ToBits().Shift(secShV)).BitsToInt16()
		a20 := s0.Sub(x).Min(k0).Max(k0.Neg())
		k1 := secStrU.SubSaturated(s1.AbsDiff(x).ToBits().Shift(secShV)).BitsToInt16()
		a21 := s1.Sub(x).Min(k1).Max(k1.Neg())
		k2 := secStrU.SubSaturated(s2.AbsDiff(x).ToBits().Shift(secShV)).BitsToInt16()
		a20 = a20.Add(s2.Sub(x).Min(k2).Max(k2.Neg()))
		k3 := secStrU.SubSaturated(s3.AbsDiff(x).ToBits().Shift(secShV)).BitsToInt16()
		a21 = a21.Add(s3.Sub(x).Min(k3).Max(k3.Neg()))
		k4 := secStrU.SubSaturated(s4.AbsDiff(x).ToBits().Shift(secShV)).BitsToInt16()
		a10 := s4.Sub(x).Min(k4).Max(k4.Neg())
		k5 := secStrU.SubSaturated(s5.AbsDiff(x).ToBits().Shift(secShV)).BitsToInt16()
		a11 := s5.Sub(x).Min(k5).Max(k5.Neg())
		k6 := secStrU.SubSaturated(s6.AbsDiff(x).ToBits().Shift(secShV)).BitsToInt16()
		a10 = a10.Add(s6.Sub(x).Min(k6).Max(k6.Neg()))
		k7 := secStrU.SubSaturated(s7.AbsDiff(x).ToBits().Shift(secShV)).BitsToInt16()
		a11 = a11.Add(s7.Sub(x).Min(k7).Max(k7.Neg()))

		sum := pa.Mul(tap0V).Add(pb.Mul(tap1V)).
			Add(a20.Add(a21).ShiftAllLeftConst(1)).
			Add(a10.Add(a11))
		neg := sum.ShiftAllRightConst(15)
		mxReal := mx0.Max(mx1).Xor(vlU).Max(x.ToBits()).BitsToInt16()
		mnReal := mn0.Min(mn1).BitsToInt16()
		y := x.Add(sum.Add(neg).Add(eight).ShiftAllRightConst(4)).
			Min(mxReal).Max(mnReal)
		*(*float64)(dst) = y.SaturateToUint8().ReshapeToFloat64x2().GetElem(0)

		src = unsafe.Add(src, BStride*2)
		dst = unsafe.Add(dst, dstStr)
	}
}

// cdefFilterBlock4U8SIMD is the 4-wide fused kernel: two rows per vector via
// cdefLoadPairU16P, arithmetic identical to the 8-wide fused kernel.
func cdefFilterBlock4U8SIMD(ctx *filterBlockU8NEONCtx) {
	priStrU := archsimd.BroadcastInt16x8(int16(ctx.priStrength)).ToBits()
	secStrU := archsimd.BroadcastInt16x8(int16(ctx.secStrength)).ToBits()
	priShV := archsimd.BroadcastInt16x8(int16(-ctx.priShift))
	secShV := archsimd.BroadcastInt16x8(int16(-ctx.secShift))
	tap0V := archsimd.BroadcastInt16x8(int16(ctx.priTap0))
	tap1V := archsimd.BroadcastInt16x8(int16(ctx.priTap1))
	eight := archsimd.BroadcastInt16x8(8)
	vlU := archsimd.BroadcastInt16x8(VeryLarge).ToBits()
	p0 := int(ctx.pri0) * 2
	p1 := int(ctx.pri1) * 2
	o0 := int(ctx.sec0) * 2
	o1 := int(ctx.sec1) * 2
	o2 := int(ctx.sec2) * 2
	o3 := int(ctx.sec3) * 2
	src := unsafe.Pointer(ctx.input)
	dst := unsafe.Pointer(ctx.dst)
	dstStr := int(ctx.dstStr)
	for h := int(ctx.height); h > 0; h -= 2 {
		src2 := unsafe.Add(src, BStride*2)
		x := cdefLoadPairU16P(src, src2)

		t0 := cdefLoadPairU16P(unsafe.Add(src, p0), unsafe.Add(src2, p0))
		t1 := cdefLoadPairU16P(unsafe.Add(src, -p0), unsafe.Add(src2, -p0))
		t2 := cdefLoadPairU16P(unsafe.Add(src, p1), unsafe.Add(src2, p1))
		t3 := cdefLoadPairU16P(unsafe.Add(src, -p1), unsafe.Add(src2, -p1))
		mn0 := x.ToBits().Min(t0.ToBits())
		mn1 := t1.ToBits().Min(t2.ToBits())
		mx0 := t0.ToBits().Xor(vlU).Max(t1.ToBits().Xor(vlU))
		mx1 := t2.ToBits().Xor(vlU).Max(t3.ToBits().Xor(vlU))
		mn0 = mn0.Min(t3.ToBits())
		l0 := priStrU.SubSaturated(t0.AbsDiff(x).ToBits().Shift(priShV)).BitsToInt16()
		pa := t0.Sub(x).Min(l0).Max(l0.Neg())
		l1 := priStrU.SubSaturated(t1.AbsDiff(x).ToBits().Shift(priShV)).BitsToInt16()
		pa = pa.Add(t1.Sub(x).Min(l1).Max(l1.Neg()))
		l2 := priStrU.SubSaturated(t2.AbsDiff(x).ToBits().Shift(priShV)).BitsToInt16()
		pb := t2.Sub(x).Min(l2).Max(l2.Neg())
		l3 := priStrU.SubSaturated(t3.AbsDiff(x).ToBits().Shift(priShV)).BitsToInt16()
		pb = pb.Add(t3.Sub(x).Min(l3).Max(l3.Neg()))

		s0 := cdefLoadPairU16P(unsafe.Add(src, o0), unsafe.Add(src2, o0))
		s1 := cdefLoadPairU16P(unsafe.Add(src, -o0), unsafe.Add(src2, -o0))
		s2 := cdefLoadPairU16P(unsafe.Add(src, o1), unsafe.Add(src2, o1))
		s3 := cdefLoadPairU16P(unsafe.Add(src, -o1), unsafe.Add(src2, -o1))
		s4 := cdefLoadPairU16P(unsafe.Add(src, o2), unsafe.Add(src2, o2))
		s5 := cdefLoadPairU16P(unsafe.Add(src, -o2), unsafe.Add(src2, -o2))
		s6 := cdefLoadPairU16P(unsafe.Add(src, o3), unsafe.Add(src2, o3))
		s7 := cdefLoadPairU16P(unsafe.Add(src, -o3), unsafe.Add(src2, -o3))
		mn0 = mn0.Min(s0.ToBits()).Min(s2.ToBits()).Min(s4.ToBits()).Min(s6.ToBits())
		mn1 = mn1.Min(s1.ToBits()).Min(s3.ToBits()).Min(s5.ToBits()).Min(s7.ToBits())
		mx0 = mx0.Max(s0.ToBits().Xor(vlU)).Max(s2.ToBits().Xor(vlU)).Max(s4.ToBits().Xor(vlU)).Max(s6.ToBits().Xor(vlU))
		mx1 = mx1.Max(s1.ToBits().Xor(vlU)).Max(s3.ToBits().Xor(vlU)).Max(s5.ToBits().Xor(vlU)).Max(s7.ToBits().Xor(vlU))

		k0 := secStrU.SubSaturated(s0.AbsDiff(x).ToBits().Shift(secShV)).BitsToInt16()
		a20 := s0.Sub(x).Min(k0).Max(k0.Neg())
		k1 := secStrU.SubSaturated(s1.AbsDiff(x).ToBits().Shift(secShV)).BitsToInt16()
		a21 := s1.Sub(x).Min(k1).Max(k1.Neg())
		k2 := secStrU.SubSaturated(s2.AbsDiff(x).ToBits().Shift(secShV)).BitsToInt16()
		a20 = a20.Add(s2.Sub(x).Min(k2).Max(k2.Neg()))
		k3 := secStrU.SubSaturated(s3.AbsDiff(x).ToBits().Shift(secShV)).BitsToInt16()
		a21 = a21.Add(s3.Sub(x).Min(k3).Max(k3.Neg()))
		k4 := secStrU.SubSaturated(s4.AbsDiff(x).ToBits().Shift(secShV)).BitsToInt16()
		a10 := s4.Sub(x).Min(k4).Max(k4.Neg())
		k5 := secStrU.SubSaturated(s5.AbsDiff(x).ToBits().Shift(secShV)).BitsToInt16()
		a11 := s5.Sub(x).Min(k5).Max(k5.Neg())
		k6 := secStrU.SubSaturated(s6.AbsDiff(x).ToBits().Shift(secShV)).BitsToInt16()
		a10 = a10.Add(s6.Sub(x).Min(k6).Max(k6.Neg()))
		k7 := secStrU.SubSaturated(s7.AbsDiff(x).ToBits().Shift(secShV)).BitsToInt16()
		a11 = a11.Add(s7.Sub(x).Min(k7).Max(k7.Neg()))

		sum := pa.Mul(tap0V).Add(pb.Mul(tap1V)).
			Add(a20.Add(a21).ShiftAllLeftConst(1)).
			Add(a10.Add(a11))
		neg := sum.ShiftAllRightConst(15)
		mxReal := mx0.Max(mx1).Xor(vlU).Max(x.ToBits()).BitsToInt16()
		mnReal := mn0.Min(mn1).BitsToInt16()
		y := x.Add(sum.Add(neg).Add(eight).ShiftAllRightConst(4)).
			Min(mxReal).Max(mnReal)
		out := y.SaturateToUint8().ReshapeToUint32s()
		*(*uint32)(dst) = out.GetElem(0)
		*(*uint32)(unsafe.Add(dst, dstStr)) = out.GetElem(1)

		src = unsafe.Add(src, 2*BStride*2)
		dst = unsafe.Add(dst, 2*dstStr)
	}
}
