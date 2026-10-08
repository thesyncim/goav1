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
//     constants) fold into one ShiftAllLeft(1)+Add at the end instead of
//     eight MLAs;
//   - the 4-wide kernel processes two rows per vector (the two 4-lane rows
//     zip1'd on 64-bit lanes), like the asm's d-register pairs.
//
// Byte-exactness with filterBlockU8PureGo: constrainShifted is reproduced
// lane-wise (identical to the asm's arithmetic, proven by the existing
// differential corpus); the finalize is x + ((8 + sum - (sum<0)) >> 4) with
// the (sum<0) term via an arithmetic ShiftAllRight(15). The secondary-
// only split never clips, so no min/max tracking exists. VeryLarge (0x4000)
// halo sentinels produce |diff| large enough that the uqsub saturates the
// limit to zero, exactly as in the reference.

package cdef

import (
	"simd/archsimd"
	"unsafe"
)

const filterBlockU8SplitSIMDEnabled = true

// dispatchFilterBlockU8NEON routes measured primary-only and secondary-only
// winners through Go SIMD. Fused blocks remain on NEON because the Go SIMD
// candidate did not show a repeatable speedup.
func dispatchFilterBlockU8NEON(ctx *filterBlockU8NEONCtx, width int, primaryStrength int, secondaryStrength int) {
	if width == 8 {
		switch {
		case primaryStrength != 0 && secondaryStrength == 0:
			cdefFilterBlock8PrimaryU8SIMD(ctx)
		case primaryStrength == 0 && secondaryStrength != 0:
			cdefFilterBlock8SecondaryU8SIMD(ctx)
		default:
			cdefFilterBlock8U8NEON(ctx)
		}
		return
	}
	switch {
	case primaryStrength != 0 && secondaryStrength == 0:
		cdefFilterBlock4PrimaryU8SIMD(ctx)
	case primaryStrength == 0 && secondaryStrength != 0:
		cdefFilterBlock4SecondaryU8SIMD(ctx)
	default:
		cdefFilterBlock4U8NEON(ctx)
	}
}

// cdefFilterBlock8SecondaryU8SIMD is the 8-wide secondary-only kernel.
// The eight tap chains are pasted inline so everything stays
// register-resident; each chain is
//
//	lim = uqsub(str, |t-x| >> shift);  c = clamp(t-x, -lim, +lim)
//
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

		l0 := strU.SubSaturated(cdefAbsDiffInt16x8(t0, x).ToBits().Shift(shV)).BitsToInt16()
		a20 := t0.Sub(x).Min(l0).Max(l0.Neg())
		l1 := strU.SubSaturated(cdefAbsDiffInt16x8(t1, x).ToBits().Shift(shV)).BitsToInt16()
		a21 := t1.Sub(x).Min(l1).Max(l1.Neg())
		l2 := strU.SubSaturated(cdefAbsDiffInt16x8(t2, x).ToBits().Shift(shV)).BitsToInt16()
		a20 = a20.Add(t2.Sub(x).Min(l2).Max(l2.Neg()))
		l3 := strU.SubSaturated(cdefAbsDiffInt16x8(t3, x).ToBits().Shift(shV)).BitsToInt16()
		a21 = a21.Add(t3.Sub(x).Min(l3).Max(l3.Neg()))
		l4 := strU.SubSaturated(cdefAbsDiffInt16x8(t4, x).ToBits().Shift(shV)).BitsToInt16()
		a10 := t4.Sub(x).Min(l4).Max(l4.Neg())
		l5 := strU.SubSaturated(cdefAbsDiffInt16x8(t5, x).ToBits().Shift(shV)).BitsToInt16()
		a11 := t5.Sub(x).Min(l5).Max(l5.Neg())
		l6 := strU.SubSaturated(cdefAbsDiffInt16x8(t6, x).ToBits().Shift(shV)).BitsToInt16()
		a10 = a10.Add(t6.Sub(x).Min(l6).Max(l6.Neg()))
		l7 := strU.SubSaturated(cdefAbsDiffInt16x8(t7, x).ToBits().Shift(shV)).BitsToInt16()
		a11 = a11.Add(t7.Sub(x).Min(l7).Max(l7.Neg()))

		// sum = 2*(weight-2 taps) + (weight-1 taps); cdefSecondaryTaps is {2,1}.
		sum := a20.Add(a21).ShiftAllLeft(1).Add(a10.Add(a11))
		// y = x + ((8 + sum - (sum<0)) >> 4)
		neg := sum.ShiftAllRight(15)
		y := x.Add(sum.Add(neg).Add(eight).ShiftAllRight(4))
		*(*uint64)(dst) = y.SaturateToUint8().ReshapeToUint64s().GetElem(0)

		if h > 1 {
			src = unsafe.Add(src, BStride*2)
			dst = unsafe.Add(dst, dstStr)
		}
	}
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

		l0 := strU.SubSaturated(cdefAbsDiffInt16x8(t0, x).ToBits().Shift(shV)).BitsToInt16()
		a20 := t0.Sub(x).Min(l0).Max(l0.Neg())
		l1 := strU.SubSaturated(cdefAbsDiffInt16x8(t1, x).ToBits().Shift(shV)).BitsToInt16()
		a21 := t1.Sub(x).Min(l1).Max(l1.Neg())
		l2 := strU.SubSaturated(cdefAbsDiffInt16x8(t2, x).ToBits().Shift(shV)).BitsToInt16()
		a20 = a20.Add(t2.Sub(x).Min(l2).Max(l2.Neg()))
		l3 := strU.SubSaturated(cdefAbsDiffInt16x8(t3, x).ToBits().Shift(shV)).BitsToInt16()
		a21 = a21.Add(t3.Sub(x).Min(l3).Max(l3.Neg()))
		l4 := strU.SubSaturated(cdefAbsDiffInt16x8(t4, x).ToBits().Shift(shV)).BitsToInt16()
		a10 := t4.Sub(x).Min(l4).Max(l4.Neg())
		l5 := strU.SubSaturated(cdefAbsDiffInt16x8(t5, x).ToBits().Shift(shV)).BitsToInt16()
		a11 := t5.Sub(x).Min(l5).Max(l5.Neg())
		l6 := strU.SubSaturated(cdefAbsDiffInt16x8(t6, x).ToBits().Shift(shV)).BitsToInt16()
		a10 = a10.Add(t6.Sub(x).Min(l6).Max(l6.Neg()))
		l7 := strU.SubSaturated(cdefAbsDiffInt16x8(t7, x).ToBits().Shift(shV)).BitsToInt16()
		a11 = a11.Add(t7.Sub(x).Min(l7).Max(l7.Neg()))

		sum := a20.Add(a21).ShiftAllLeft(1).Add(a10.Add(a11))
		neg := sum.ShiftAllRight(15)
		y := x.Add(sum.Add(neg).Add(eight).ShiftAllRight(4))
		out := y.SaturateToUint8().ReshapeToUint32s()
		*(*uint32)(dst) = out.GetElem(0)
		*(*uint32)(unsafe.Add(dst, dstStr)) = out.GetElem(1)

		if h > 2 {
			src = unsafe.Add(src, 2*BStride*2)
			dst = unsafe.Add(dst, 2*dstStr)
		}
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

		l0 := strU.SubSaturated(cdefAbsDiffInt16x8(t0, x).ToBits().Shift(shV)).BitsToInt16()
		a0 := t0.Sub(x).Min(l0).Max(l0.Neg())
		l1 := strU.SubSaturated(cdefAbsDiffInt16x8(t1, x).ToBits().Shift(shV)).BitsToInt16()
		a1 := t1.Sub(x).Min(l1).Max(l1.Neg())
		l2 := strU.SubSaturated(cdefAbsDiffInt16x8(t2, x).ToBits().Shift(shV)).BitsToInt16()
		b0 := t2.Sub(x).Min(l2).Max(l2.Neg())
		l3 := strU.SubSaturated(cdefAbsDiffInt16x8(t3, x).ToBits().Shift(shV)).BitsToInt16()
		b1 := t3.Sub(x).Min(l3).Max(l3.Neg())

		sum := a0.Add(a1).Mul(tap0V).Add(b0.Add(b1).Mul(tap1V))
		neg := sum.ShiftAllRight(15)
		y := x.Add(sum.Add(neg).Add(eight).ShiftAllRight(4))
		*(*uint64)(dst) = y.SaturateToUint8().ReshapeToUint64s().GetElem(0)

		if h > 1 {
			src = unsafe.Add(src, BStride*2)
			dst = unsafe.Add(dst, dstStr)
		}
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

		l0 := strU.SubSaturated(cdefAbsDiffInt16x8(t0, x).ToBits().Shift(shV)).BitsToInt16()
		a0 := t0.Sub(x).Min(l0).Max(l0.Neg())
		l1 := strU.SubSaturated(cdefAbsDiffInt16x8(t1, x).ToBits().Shift(shV)).BitsToInt16()
		a1 := t1.Sub(x).Min(l1).Max(l1.Neg())
		l2 := strU.SubSaturated(cdefAbsDiffInt16x8(t2, x).ToBits().Shift(shV)).BitsToInt16()
		b0 := t2.Sub(x).Min(l2).Max(l2.Neg())
		l3 := strU.SubSaturated(cdefAbsDiffInt16x8(t3, x).ToBits().Shift(shV)).BitsToInt16()
		b1 := t3.Sub(x).Min(l3).Max(l3.Neg())

		sum := a0.Add(a1).Mul(tap0V).Add(b0.Add(b1).Mul(tap1V))
		neg := sum.ShiftAllRight(15)
		y := x.Add(sum.Add(neg).Add(eight).ShiftAllRight(4))
		out := y.SaturateToUint8().ReshapeToUint32s()
		*(*uint32)(dst) = out.GetElem(0)
		*(*uint32)(unsafe.Add(dst, dstStr)) = out.GetElem(1)

		if h > 2 {
			src = unsafe.Add(src, 2*BStride*2)
			dst = unsafe.Add(dst, 2*dstStr)
		}
	}
}
