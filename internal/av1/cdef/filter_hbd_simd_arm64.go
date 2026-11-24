// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package cdef

import "simd/archsimd"

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
	s0p, s0m, s1p, s1m := ctx.secTaps0()
	s2p, s2m, s3p, s3m := ctx.secTaps1()
	in, base, height := ctx.rows()
	out16, _, do, ds := ctx.outRows()
	for r := 0; r < height; r++ {
		w := cdefRow(in, base+r*BStride)
		x := cdefLdW1(w, cdefWinReach)
		a := cdefConstrain(cdefLdW1(w, s0p), x, secStr, secSh).
			Add(cdefConstrain(cdefLdW1(w, s0m), x, secStr, secSh)).
			Add(cdefConstrain(cdefLdW1(w, s1p), x, secStr, secSh)).
			Add(cdefConstrain(cdefLdW1(w, s1m), x, secStr, secSh))
		b := cdefConstrain(cdefLdW1(w, s2p), x, secStr, secSh).
			Add(cdefConstrain(cdefLdW1(w, s2m), x, secStr, secSh)).
			Add(cdefConstrain(cdefLdW1(w, s3p), x, secStr, secSh)).
			Add(cdefConstrain(cdefLdW1(w, s3m), x, secStr, secSh))
		sum := a.Add(a).Add(b)
		cdefFinish16(x, sum, eight, sar15, sar4).ToBits().StoreArray(cdefOut16(out16, do+r*ds))
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
	s0p, s0m, s1p, s1m := ctx.secTaps0()
	s2p, s2m, s3p, s3m := ctx.secTaps1()
	in, base, height := ctx.rows()
	out16, _, do, ds := ctx.outRows()
	// Counted by row pair: a step-2 counter keeps a second check per window.
	for i := 0; i < height>>1; i++ {
		r := 2 * i
		w0, w1 := cdefRow(in, base+r*BStride), cdefRow(in, base+(r+1)*BStride)
		x := cdefLdPair(w0, w1, cdefWinReach)
		a := cdefConstrain(cdefLdPair(w0, w1, s0p), x, secStr, secSh).
			Add(cdefConstrain(cdefLdPair(w0, w1, s0m), x, secStr, secSh)).
			Add(cdefConstrain(cdefLdPair(w0, w1, s1p), x, secStr, secSh)).
			Add(cdefConstrain(cdefLdPair(w0, w1, s1m), x, secStr, secSh))
		b := cdefConstrain(cdefLdPair(w0, w1, s2p), x, secStr, secSh).
			Add(cdefConstrain(cdefLdPair(w0, w1, s2m), x, secStr, secSh)).
			Add(cdefConstrain(cdefLdPair(w0, w1, s3p), x, secStr, secSh)).
			Add(cdefConstrain(cdefLdPair(w0, w1, s3m), x, secStr, secSh))
		sum := a.Add(a).Add(b)
		out := cdefFinish16(x, sum, eight, sar15, sar4).ToBits().ReshapeToUint64s()
		cdefPutU16x4(out16, do+r*ds, out.GetElem(0))
		cdefPutU16x4(out16, do+(r+1)*ds, out.GetElem(1))
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
	p0p, p0m, p1p, p1m := ctx.priTaps()
	in, base, height := ctx.rows()
	out16, _, do, ds := ctx.outRows()
	for r := 0; r < height; r++ {
		w := cdefRow(in, base+r*BStride)
		x := cdefLdW1(w, cdefWinReach)
		sum := cdefConstrain(cdefLdW1(w, p0p), x, priStr, priSh).
			Add(cdefConstrain(cdefLdW1(w, p0m), x, priStr, priSh)).Mul(priTap0).
			Add(cdefConstrain(cdefLdW1(w, p1p), x, priStr, priSh).
				Add(cdefConstrain(cdefLdW1(w, p1m), x, priStr, priSh)).Mul(priTap1))
		cdefFinish16(x, sum, eight, sar15, sar4).ToBits().StoreArray(cdefOut16(out16, do+r*ds))
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
	p0p, p0m, p1p, p1m := ctx.priTaps()
	in, base, height := ctx.rows()
	out16, _, do, ds := ctx.outRows()
	// Counted by row pair: a step-2 counter keeps a second check per window.
	for i := 0; i < height>>1; i++ {
		r := 2 * i
		w0, w1 := cdefRow(in, base+r*BStride), cdefRow(in, base+(r+1)*BStride)
		x := cdefLdPair(w0, w1, cdefWinReach)
		sum := cdefConstrain(cdefLdPair(w0, w1, p0p), x, priStr, priSh).
			Add(cdefConstrain(cdefLdPair(w0, w1, p0m), x, priStr, priSh)).Mul(priTap0).
			Add(cdefConstrain(cdefLdPair(w0, w1, p1p), x, priStr, priSh).
				Add(cdefConstrain(cdefLdPair(w0, w1, p1m), x, priStr, priSh)).Mul(priTap1))
		out := cdefFinish16(x, sum, eight, sar15, sar4).ToBits().ReshapeToUint64s()
		cdefPutU16x4(out16, do+r*ds, out.GetElem(0))
		cdefPutU16x4(out16, do+(r+1)*ds, out.GetElem(1))
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
	p0p, p0m, p1p, p1m := ctx.priTaps()
	s0p, s0m, s1p, s1m := ctx.secTaps0()
	s2p, s2m, s3p, s3m := ctx.secTaps1()
	in, base, height := ctx.rows()
	out16, _, do, ds := ctx.outRows()
	for r := 0; r < height; r++ {
		w := cdefRow(in, base+r*BStride)
		x := cdefLdW1(w, cdefWinReach)
		t0 := cdefLdW1(w, p0p)
		t1 := cdefLdW1(w, p0m)
		t2 := cdefLdW1(w, p1p)
		t3 := cdefLdW1(w, p1m)
		sum := cdefConstrain(t0, x, priStr, priSh).Add(cdefConstrain(t1, x, priStr, priSh)).Mul(priTap0).
			Add(cdefConstrain(t2, x, priStr, priSh).Add(cdefConstrain(t3, x, priStr, priSh)).Mul(priTap1))
		// The min and max fold as trees per tap group so their dependency
		// chains stay short.
		mx := t0.And(noSentinel).Max(t1.And(noSentinel)).Max(t2.And(noSentinel).Max(t3.And(noSentinel)))
		mn := t0.Min(t1).Min(t2.Min(t3))
		s0 := cdefLdW1(w, s0p)
		s1 := cdefLdW1(w, s0m)
		s2 := cdefLdW1(w, s1p)
		s3 := cdefLdW1(w, s1m)
		a := cdefConstrain(s0, x, secStr, secSh).Add(cdefConstrain(s1, x, secStr, secSh)).
			Add(cdefConstrain(s2, x, secStr, secSh)).Add(cdefConstrain(s3, x, secStr, secSh))
		mx1 := s0.And(noSentinel).Max(s1.And(noSentinel)).Max(s2.And(noSentinel).Max(s3.And(noSentinel)))
		mn1 := s0.Min(s1).Min(s2.Min(s3))
		s4 := cdefLdW1(w, s2p)
		s5 := cdefLdW1(w, s2m)
		s6 := cdefLdW1(w, s3p)
		s7 := cdefLdW1(w, s3m)
		b := cdefConstrain(s4, x, secStr, secSh).Add(cdefConstrain(s5, x, secStr, secSh)).
			Add(cdefConstrain(s6, x, secStr, secSh)).Add(cdefConstrain(s7, x, secStr, secSh))
		mx2 := s4.And(noSentinel).Max(s5.And(noSentinel)).Max(s6.And(noSentinel).Max(s7.And(noSentinel)))
		mn2 := s4.Min(s5).Min(s6.Min(s7))
		mx = mx.Max(x).Max(mx1.Max(mx2))
		mn = mn.Min(x).Min(mn1.Min(mn2))
		sum = sum.Add(a.Add(a).Add(b))
		cdefFinish16(x, sum, eight, sar15, sar4).Max(mn).Min(mx).ToBits().StoreArray(cdefOut16(out16, do+r*ds))
	}
}
