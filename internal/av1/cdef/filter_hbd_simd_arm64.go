// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package cdef

import (
	"simd/archsimd"
	"unsafe"
)

// The uint16-output CDEF block kernels split by strength, as dav1d's
// cdef_filter{4,8}_{pri,sec,pri_sec}_16bpc_neon (src/arm/64/cdef_tmpl.S
// filter_func): each block runs the kernel for the taps it has, so the row
// loops carry no strength branches, the secondary-only and primary-only
// kernels skip the min/max clip (it only applies when both are active), and
// the secondary taps (2, 1) fold to an add instead of two multiplies. The
// arithmetic is cdefFilterBlock8SIMD / cdefFilterBlock4SIMD's, lane for lane.
// The 4-wide block with both strengths keeps the shared kernel.

// cdefFilterBlock16 routes a prepared uint16-output context to the kernel
// for its strengths. At least one strength is nonzero (zero-strength blocks
// are skipped by the caller); width is 8, or 4 with an even height.
func cdefFilterBlock16(ctx *cdefSIMDCtx, width int) {
	switch {
	case !ctx.enablePrimary:
		if width == 8 {
			cdefFilter8Sec16(ctx)
		} else {
			cdefFilter4Sec16(ctx)
		}
	case !ctx.enableSecondary:
		if width == 8 {
			cdefFilter8Pri16(ctx)
		} else {
			cdefFilter4Pri16(ctx)
		}
	case width == 8:
		cdefFilter8PriSec16(ctx)
	default:
		cdefFilterBlockSIMD(ctx, width)
	}
}

// cdefFinish16 is x + ((8 + sum - (sum < 0)) >> 4).
func cdefFinish16(x, sum, eight archsimd.Int16x8, sar15, sar4 cdefShiftCount) archsimd.Int16x8 {
	return x.Add(cdefSarI16(sum.Add(cdefSarI16(sum, sar15)).Add(eight), sar4))
}

// cdefFilter8Sec16 is the 8-wide secondary-only kernel: sum =
// 2 * (c0 + c1 + c2 + c3) + (c4 + c5 + c6 + c7), no clip.
func cdefFilter8Sec16(ctx *cdefSIMDCtx) {
	secStr := archsimd.BroadcastInt16x8(int16(ctx.secStrength)).ToBits()
	secSh := cdefShiftCountOf(ctx.secShift)
	eight := archsimd.BroadcastInt16x8(8)
	sar15 := cdefShiftCountOf(15)
	sar4 := cdefShiftCountOf(4)
	sec0, sec1, sec2, sec3 := ctx.sec0*2, ctx.sec1*2, ctx.sec2*2, ctx.sec3*2
	src := ctx.input
	dst := ctx.dst
	dstStr := ctx.dstStr
	for h := ctx.height; h > 0; h-- {
		x := cdefLoadU16P(src)
		a := cdefConstrain(cdefLoadU16P(unsafe.Add(src, sec0)), x, secStr, secSh).
			Add(cdefConstrain(cdefLoadU16P(unsafe.Add(src, -sec0)), x, secStr, secSh)).
			Add(cdefConstrain(cdefLoadU16P(unsafe.Add(src, sec1)), x, secStr, secSh)).
			Add(cdefConstrain(cdefLoadU16P(unsafe.Add(src, -sec1)), x, secStr, secSh))
		b := cdefConstrain(cdefLoadU16P(unsafe.Add(src, sec2)), x, secStr, secSh).
			Add(cdefConstrain(cdefLoadU16P(unsafe.Add(src, -sec2)), x, secStr, secSh)).
			Add(cdefConstrain(cdefLoadU16P(unsafe.Add(src, sec3)), x, secStr, secSh)).
			Add(cdefConstrain(cdefLoadU16P(unsafe.Add(src, -sec3)), x, secStr, secSh))
		sum := a.Add(a).Add(b)
		cdefFinish16(x, sum, eight, sar15, sar4).ToBits().StoreArray((*[8]uint16)(dst))
		// Advance only while rows remain: a pointer past the last row may
		// leave the allocation (checkptr).
		if h > 1 {
			src = unsafe.Add(src, BStride*2)
			dst = unsafe.Add(dst, dstStr)
		}
	}
}

// cdefFilter4Sec16 is cdefFilter8Sec16 for a 4-wide block, two rows per
// vector (rows r and r+1 in lanes 0..3 and 4..7).
func cdefFilter4Sec16(ctx *cdefSIMDCtx) {
	secStr := archsimd.BroadcastInt16x8(int16(ctx.secStrength)).ToBits()
	secSh := cdefShiftCountOf(ctx.secShift)
	eight := archsimd.BroadcastInt16x8(8)
	sar15 := cdefShiftCountOf(15)
	sar4 := cdefShiftCountOf(4)
	sec0, sec1, sec2, sec3 := ctx.sec0*2, ctx.sec1*2, ctx.sec2*2, ctx.sec3*2
	src := ctx.input
	dst := ctx.dst
	dstStr := ctx.dstStr
	for h := ctx.height; h > 0; h -= 2 {
		src2 := unsafe.Add(src, BStride*2)
		x := cdefLoadPairU16P(src, src2)
		a := cdefConstrain(cdefLoadPairU16P(unsafe.Add(src, sec0), unsafe.Add(src2, sec0)), x, secStr, secSh).
			Add(cdefConstrain(cdefLoadPairU16P(unsafe.Add(src, -sec0), unsafe.Add(src2, -sec0)), x, secStr, secSh)).
			Add(cdefConstrain(cdefLoadPairU16P(unsafe.Add(src, sec1), unsafe.Add(src2, sec1)), x, secStr, secSh)).
			Add(cdefConstrain(cdefLoadPairU16P(unsafe.Add(src, -sec1), unsafe.Add(src2, -sec1)), x, secStr, secSh))
		b := cdefConstrain(cdefLoadPairU16P(unsafe.Add(src, sec2), unsafe.Add(src2, sec2)), x, secStr, secSh).
			Add(cdefConstrain(cdefLoadPairU16P(unsafe.Add(src, -sec2), unsafe.Add(src2, -sec2)), x, secStr, secSh)).
			Add(cdefConstrain(cdefLoadPairU16P(unsafe.Add(src, sec3), unsafe.Add(src2, sec3)), x, secStr, secSh)).
			Add(cdefConstrain(cdefLoadPairU16P(unsafe.Add(src, -sec3), unsafe.Add(src2, -sec3)), x, secStr, secSh))
		sum := a.Add(a).Add(b)
		out := cdefFinish16(x, sum, eight, sar15, sar4).ToBits().ReshapeToUint64s()
		*(*uint64)(dst) = out.GetElem(0)
		*(*uint64)(unsafe.Add(dst, dstStr)) = out.GetElem(1)
		if h > 2 {
			src = unsafe.Add(src, 2*BStride*2)
			dst = unsafe.Add(dst, 2*dstStr)
		}
	}
}

// cdefFilter8Pri16 is the 8-wide primary-only kernel (no clip).
func cdefFilter8Pri16(ctx *cdefSIMDCtx) {
	priStr := archsimd.BroadcastInt16x8(int16(ctx.priStrength)).ToBits()
	priSh := cdefShiftCountOf(ctx.priShift)
	priTap0 := archsimd.BroadcastInt16x8(int16(ctx.priTap0))
	priTap1 := archsimd.BroadcastInt16x8(int16(ctx.priTap1))
	eight := archsimd.BroadcastInt16x8(8)
	sar15 := cdefShiftCountOf(15)
	sar4 := cdefShiftCountOf(4)
	pri0, pri1 := ctx.pri0*2, ctx.pri1*2
	src := ctx.input
	dst := ctx.dst
	dstStr := ctx.dstStr
	for h := ctx.height; h > 0; h-- {
		x := cdefLoadU16P(src)
		sum := cdefConstrain(cdefLoadU16P(unsafe.Add(src, pri0)), x, priStr, priSh).
			Add(cdefConstrain(cdefLoadU16P(unsafe.Add(src, -pri0)), x, priStr, priSh)).Mul(priTap0).
			Add(cdefConstrain(cdefLoadU16P(unsafe.Add(src, pri1)), x, priStr, priSh).
				Add(cdefConstrain(cdefLoadU16P(unsafe.Add(src, -pri1)), x, priStr, priSh)).Mul(priTap1))
		cdefFinish16(x, sum, eight, sar15, sar4).ToBits().StoreArray((*[8]uint16)(dst))
		// Advance only while rows remain: a pointer past the last row may
		// leave the allocation (checkptr).
		if h > 1 {
			src = unsafe.Add(src, BStride*2)
			dst = unsafe.Add(dst, dstStr)
		}
	}
}

// cdefFilter4Pri16 is cdefFilter8Pri16 for a 4-wide block, two rows per
// vector.
func cdefFilter4Pri16(ctx *cdefSIMDCtx) {
	priStr := archsimd.BroadcastInt16x8(int16(ctx.priStrength)).ToBits()
	priSh := cdefShiftCountOf(ctx.priShift)
	priTap0 := archsimd.BroadcastInt16x8(int16(ctx.priTap0))
	priTap1 := archsimd.BroadcastInt16x8(int16(ctx.priTap1))
	eight := archsimd.BroadcastInt16x8(8)
	sar15 := cdefShiftCountOf(15)
	sar4 := cdefShiftCountOf(4)
	pri0, pri1 := ctx.pri0*2, ctx.pri1*2
	src := ctx.input
	dst := ctx.dst
	dstStr := ctx.dstStr
	for h := ctx.height; h > 0; h -= 2 {
		src2 := unsafe.Add(src, BStride*2)
		x := cdefLoadPairU16P(src, src2)
		sum := cdefConstrain(cdefLoadPairU16P(unsafe.Add(src, pri0), unsafe.Add(src2, pri0)), x, priStr, priSh).
			Add(cdefConstrain(cdefLoadPairU16P(unsafe.Add(src, -pri0), unsafe.Add(src2, -pri0)), x, priStr, priSh)).Mul(priTap0).
			Add(cdefConstrain(cdefLoadPairU16P(unsafe.Add(src, pri1), unsafe.Add(src2, pri1)), x, priStr, priSh).
				Add(cdefConstrain(cdefLoadPairU16P(unsafe.Add(src, -pri1), unsafe.Add(src2, -pri1)), x, priStr, priSh)).Mul(priTap1))
		out := cdefFinish16(x, sum, eight, sar15, sar4).ToBits().ReshapeToUint64s()
		*(*uint64)(dst) = out.GetElem(0)
		*(*uint64)(unsafe.Add(dst, dstStr)) = out.GetElem(1)
		if h > 2 {
			src = unsafe.Add(src, 2*BStride*2)
			dst = unsafe.Add(dst, 2*dstStr)
		}
	}
}

// cdefFilter8PriSec16 is the 8-wide kernel with both strengths: the primary
// and secondary sums, then the clip to [min, max] over the taps (the
// sentinel masked out of the max, as in cdefFilterBlock8SIMD).
func cdefFilter8PriSec16(ctx *cdefSIMDCtx) {
	priStr := archsimd.BroadcastInt16x8(int16(ctx.priStrength)).ToBits()
	secStr := archsimd.BroadcastInt16x8(int16(ctx.secStrength)).ToBits()
	priSh := cdefShiftCountOf(ctx.priShift)
	secSh := cdefShiftCountOf(ctx.secShift)
	priTap0 := archsimd.BroadcastInt16x8(int16(ctx.priTap0))
	priTap1 := archsimd.BroadcastInt16x8(int16(ctx.priTap1))
	eight := archsimd.BroadcastInt16x8(8)
	noSentinel := archsimd.BroadcastInt16x8(int16(^VeryLarge))
	sar15 := cdefShiftCountOf(15)
	sar4 := cdefShiftCountOf(4)
	pri0, pri1 := ctx.pri0*2, ctx.pri1*2
	sec0, sec1, sec2, sec3 := ctx.sec0*2, ctx.sec1*2, ctx.sec2*2, ctx.sec3*2
	src := ctx.input
	dst := ctx.dst
	dstStr := ctx.dstStr
	for h := ctx.height; h > 0; h-- {
		x := cdefLoadU16P(src)
		t0 := cdefLoadU16P(unsafe.Add(src, pri0))
		t1 := cdefLoadU16P(unsafe.Add(src, -pri0))
		t2 := cdefLoadU16P(unsafe.Add(src, pri1))
		t3 := cdefLoadU16P(unsafe.Add(src, -pri1))
		sum := cdefConstrain(t0, x, priStr, priSh).Add(cdefConstrain(t1, x, priStr, priSh)).Mul(priTap0).
			Add(cdefConstrain(t2, x, priStr, priSh).Add(cdefConstrain(t3, x, priStr, priSh)).Mul(priTap1))
		// The min and max fold as trees per tap group so their dependency
		// chains stay short.
		mx := t0.And(noSentinel).Max(t1.And(noSentinel)).Max(t2.And(noSentinel).Max(t3.And(noSentinel)))
		mn := t0.Min(t1).Min(t2.Min(t3))
		s0 := cdefLoadU16P(unsafe.Add(src, sec0))
		s1 := cdefLoadU16P(unsafe.Add(src, -sec0))
		s2 := cdefLoadU16P(unsafe.Add(src, sec1))
		s3 := cdefLoadU16P(unsafe.Add(src, -sec1))
		a := cdefConstrain(s0, x, secStr, secSh).Add(cdefConstrain(s1, x, secStr, secSh)).
			Add(cdefConstrain(s2, x, secStr, secSh)).Add(cdefConstrain(s3, x, secStr, secSh))
		mx1 := s0.And(noSentinel).Max(s1.And(noSentinel)).Max(s2.And(noSentinel).Max(s3.And(noSentinel)))
		mn1 := s0.Min(s1).Min(s2.Min(s3))
		s4 := cdefLoadU16P(unsafe.Add(src, sec2))
		s5 := cdefLoadU16P(unsafe.Add(src, -sec2))
		s6 := cdefLoadU16P(unsafe.Add(src, sec3))
		s7 := cdefLoadU16P(unsafe.Add(src, -sec3))
		b := cdefConstrain(s4, x, secStr, secSh).Add(cdefConstrain(s5, x, secStr, secSh)).
			Add(cdefConstrain(s6, x, secStr, secSh)).Add(cdefConstrain(s7, x, secStr, secSh))
		mx2 := s4.And(noSentinel).Max(s5.And(noSentinel)).Max(s6.And(noSentinel).Max(s7.And(noSentinel)))
		mn2 := s4.Min(s5).Min(s6.Min(s7))
		mx = mx.Max(x).Max(mx1.Max(mx2))
		mn = mn.Min(x).Min(mn1.Min(mn2))
		sum = sum.Add(a.Add(a).Add(b))
		cdefFinish16(x, sum, eight, sar15, sar4).Max(mn).Min(mx).ToBits().StoreArray((*[8]uint16)(dst))
		// Advance only while rows remain: a pointer past the last row may
		// leave the allocation (checkptr).
		if h > 1 {
			src = unsafe.Add(src, BStride*2)
			dst = unsafe.Add(dst, dstStr)
		}
	}
}
