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

// dispatchFilterBlockU8NEON is the goexperiment.simd router: secondary-only
// blocks run the Go-native SIMD kernels below; every other strength split
// keeps the NEON asm (still compiled in this build). The stock router lives in
// filter_u8_route_arm64.go.
func dispatchFilterBlockU8NEON(ctx *filterBlockU8NEONCtx, width int, primaryStrength int, secondaryStrength int) {
	if primaryStrength == 0 && secondaryStrength != 0 {
		if width == 8 {
			cdefFilterBlock8SecondaryU8SIMD(ctx)
		} else {
			cdefFilterBlock4SecondaryU8SIMD(ctx)
		}
		return
	}
	if width == 8 {
		if secondaryStrength == 0 {
			cdefFilterBlock8PrimaryU8NEON(ctx)
		} else {
			cdefFilterBlock8U8NEON(ctx)
		}
		return
	}
	if secondaryStrength == 0 {
		cdefFilterBlock4PrimaryU8NEON(ctx)
	} else {
		cdefFilterBlock4U8NEON(ctx)
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
