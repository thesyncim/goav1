// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

//go:build goexperiment.simd && amd64 && !purego

package cdef

// 256-bit AVX2 CDEF block kernels on the uint16 tap buffer (sentinel halo),
// ported from dav1d's border path (src/x86/cdef_avx2.asm BORDER_LOAD_BLOCK /
// ACCUMULATE_TAP_WORD / BORDER_ADJUST_PIXEL, and cdef16_avx2.asm, which runs
// every block that way). 16 word lanes hold two 8-pixel rows ([r0 | r1]) for
// 8-wide blocks, or four 4-pixel rows ([r0 r1 | r2 r3]) for 4-wide blocks.
//
// The arithmetic is the 128-bit cdefFilterBlock8SIMD's lane for lane, with
// constrain() as dav1d's pabsw / psrlw / psubusw / pminsw / psignw:
// psignw folds the sign of t-x back onto min(|t-x|, lim) in one op (zero
// when t == x, where the magnitude is zero anyway). dav1d's 8bpc border path
// packs the differences to bytes with packsswb; archsimd has no VPACKSSWB on
// amd64, so the constrain stays in words. Output is byte-identical to
// filterBlockPureGo / filterBlockU8PureGo.

import "simd/archsimd"

// cdefWordLd loads the 8 samples of w at i as Int16x8.
func cdefWordLd(w *[cdefWin1]uint16, i uint) archsimd.Int16x8 {
	return archsimd.LoadUint16x8Array((*[8]uint16)(w[i : i+8])).AsInt16x8()
}

// cdefWordLoad8W packs the 8 samples at window index i of two rows' windows
// as [r0 | r1].
func cdefWordLoad8W(w0, w1 *[cdefWin1]uint16, i uint) archsimd.Int16x16 {
	var z archsimd.Int16x16
	return z.SetLo(cdefWordLd(w0, i)).SetHi(cdefWordLd(w1, i))
}

// cdefWordLoad4W packs the 4 samples at window index i of four rows' windows
// as [r0 r1 | r2 r3] (BORDER_LOAD_BLOCK w=4: movq / movhps /
// vinserti128). The full-width loads read four halo samples past each row
// segment, inside the 8-column border.
func cdefWordLoad4W(w0, w1, w2, w3 *[cdefWin1]uint16, i uint) archsimd.Int16x16 {
	var z archsimd.Int16x16
	ac := z.SetLo(cdefWordLd(w0, i)).SetHi(cdefWordLd(w2, i)).AsUint64x4()
	bd := z.SetLo(cdefWordLd(w1, i)).SetHi(cdefWordLd(w3, i)).AsUint64x4()
	return ac.InterleaveLoGrouped(bd).AsInt16x16()
}

// cdefWordConstrain is constrain() on 16 word lanes.
//
// The shift is VPSRLVD by a hoisted count vector, masked back to words
// (wmask = 0xffff >> sh): VPSRLW takes its count from an xmm the compiler
// rematerializes with a legacy-SSE MOVQ inside the row loop, an SSE/AVX
// transition while the ymm upper halves are live.
func cdefWordConstrain(t, x archsimd.Int16x16, str archsimd.Uint16x16, sh archsimd.Uint32x8, wmask archsimd.Uint16x16) archsimd.Int16x16 {
	d := t.Sub(x)
	a := d.Abs()
	lim := str.SubSaturated(a.AsUint32x8().ShiftRight(sh).AsUint16x16().And(wmask)).AsInt16x16()
	return a.Min(lim).MulSign(d)
}

// cdefWordLowBytes gathers each 128-bit lane's low bytes of its eight words
// into that lane's bytes 0..7 (VPSHUFB).
var cdefWordLowBytes = [32]int8{
	0, 2, 4, 6, 8, 10, 12, 14, -128, -128, -128, -128, -128, -128, -128, -128,
	0, 2, 4, 6, 8, 10, 12, 14, -128, -128, -128, -128, -128, -128, -128, -128,
}

// cdefWordFilter8 filters an 8-wide block two rows per vector. Like the
// 128-bit kernel it selects the strength split with loop-invariant flags.
func cdefWordFilter8(ctx *cdefSIMDCtx) {
	priStr := archsimd.BroadcastUint16x16(uint16(ctx.priStrength))
	secStr := archsimd.BroadcastUint16x16(uint16(ctx.secStrength))
	priSh, secSh := archsimd.BroadcastUint32x8(uint32(ctx.priShift)), archsimd.BroadcastUint32x8(uint32(ctx.secShift))
	priM, secM := archsimd.BroadcastUint16x16(uint16(0xffff>>ctx.priShift)), archsimd.BroadcastUint16x16(uint16(0xffff>>ctx.secShift))
	priTap0 := archsimd.BroadcastInt16x16(int16(ctx.priTap0))
	priTap1 := archsimd.BroadcastInt16x16(int16(ctx.priTap1))
	secTap0 := archsimd.BroadcastInt16x16(int16(ctx.secTap0))
	secTap1 := archsimd.BroadcastInt16x16(int16(ctx.secTap1))
	eight := archsimd.BroadcastInt16x16(8)
	noSentinel := archsimd.BroadcastInt16x16(int16(^VeryLarge))
	priOn, secOn, u8 := ctx.enablePrimary, ctx.enableSecondary, ctx.u8
	clip := priOn && secOn
	p0p, p0m, p1p, p1m := ctx.priTaps()
	s0p, s0m, s1p, s1m := ctx.secTaps0()
	s2p, s2m, s3p, s3m := ctx.secTaps1()
	in, base, height := ctx.rows()
	out16, out8, do, ds := ctx.outRows()
	// Counted by row pair: a step-2 counter keeps a second check per window.
	for i := 0; i < height>>1; i++ {
		r := 2 * i
		w0, w1 := cdefRow(in, base+r*BStride), cdefRow(in, base+(r+1)*BStride)
		x := cdefWordLoad8W(w0, w1, cdefWinReach)
		var sum archsimd.Int16x16
		mx, mn := x, x
		if priOn {
			t0 := cdefWordLoad8W(w0, w1, p0p)
			t1 := cdefWordLoad8W(w0, w1, p0m)
			t2 := cdefWordLoad8W(w0, w1, p1p)
			t3 := cdefWordLoad8W(w0, w1, p1m)
			sum = cdefWordConstrain(t0, x, priStr, priSh, priM).Add(cdefWordConstrain(t1, x, priStr, priSh, priM)).Mul(priTap0).
				Add(cdefWordConstrain(t2, x, priStr, priSh, priM).Add(cdefWordConstrain(t3, x, priStr, priSh, priM)).Mul(priTap1))
			if clip {
				mx = mx.Max(t0.And(noSentinel)).Max(t1.And(noSentinel)).Max(t2.And(noSentinel)).Max(t3.And(noSentinel))
				mn = mn.Min(t0).Min(t1).Min(t2).Min(t3)
			}
		}
		if secOn {
			s0 := cdefWordLoad8W(w0, w1, s0p)
			s1 := cdefWordLoad8W(w0, w1, s0m)
			s2 := cdefWordLoad8W(w0, w1, s1p)
			s3 := cdefWordLoad8W(w0, w1, s1m)
			sum = sum.Add(cdefWordConstrain(s0, x, secStr, secSh, secM).Add(cdefWordConstrain(s1, x, secStr, secSh, secM)).
				Add(cdefWordConstrain(s2, x, secStr, secSh, secM)).Add(cdefWordConstrain(s3, x, secStr, secSh, secM)).Mul(secTap0))
			if clip {
				mx = mx.Max(s0.And(noSentinel)).Max(s1.And(noSentinel)).Max(s2.And(noSentinel)).Max(s3.And(noSentinel))
				mn = mn.Min(s0).Min(s1).Min(s2).Min(s3)
			}
			s4 := cdefWordLoad8W(w0, w1, s2p)
			s5 := cdefWordLoad8W(w0, w1, s2m)
			s6 := cdefWordLoad8W(w0, w1, s3p)
			s7 := cdefWordLoad8W(w0, w1, s3m)
			sum = sum.Add(cdefWordConstrain(s4, x, secStr, secSh, secM).Add(cdefWordConstrain(s5, x, secStr, secSh, secM)).
				Add(cdefWordConstrain(s6, x, secStr, secSh, secM)).Add(cdefWordConstrain(s7, x, secStr, secSh, secM)).Mul(secTap1))
			if clip {
				mx = mx.Max(s4.And(noSentinel)).Max(s5.And(noSentinel)).Max(s6.And(noSentinel)).Max(s7.And(noSentinel))
				mn = mn.Min(s4).Min(s5).Min(s6).Min(s7)
			}
		}
		y := x.Add(sum.Add(sum.ShiftAllRight(15)).Add(eight).ShiftAllRight(4))
		if clip {
			y = y.Max(mn).Min(mx)
		}
		if u8 {
			b := y.AsUint8x32().PermuteOrZeroGrouped(archsimd.LoadInt8x32Array(&cdefWordLowBytes))
			cdefPut8(out8, do+r*ds, b.GetLo().AsUint64x2().GetElem(0))
			cdefPut8(out8, do+(r+1)*ds, b.GetHi().AsUint64x2().GetElem(0))
		} else {
			y.GetLo().AsUint16x8().StoreArray(cdefOut16(out16, do+r*ds))
			y.GetHi().AsUint16x8().StoreArray(cdefOut16(out16, do+(r+1)*ds))
		}
	}
	archsimd.ClearAVXUpperBits()
}

// cdefWordFilter4 filters a 4-wide block four rows per vector. The height is
// a multiple of 4 by dispatch contract.
func cdefWordFilter4(ctx *cdefSIMDCtx) {
	priStr := archsimd.BroadcastUint16x16(uint16(ctx.priStrength))
	secStr := archsimd.BroadcastUint16x16(uint16(ctx.secStrength))
	priSh, secSh := archsimd.BroadcastUint32x8(uint32(ctx.priShift)), archsimd.BroadcastUint32x8(uint32(ctx.secShift))
	priM, secM := archsimd.BroadcastUint16x16(uint16(0xffff>>ctx.priShift)), archsimd.BroadcastUint16x16(uint16(0xffff>>ctx.secShift))
	priTap0 := archsimd.BroadcastInt16x16(int16(ctx.priTap0))
	priTap1 := archsimd.BroadcastInt16x16(int16(ctx.priTap1))
	secTap0 := archsimd.BroadcastInt16x16(int16(ctx.secTap0))
	secTap1 := archsimd.BroadcastInt16x16(int16(ctx.secTap1))
	eight := archsimd.BroadcastInt16x16(8)
	noSentinel := archsimd.BroadcastInt16x16(int16(^VeryLarge))
	priOn, secOn, u8 := ctx.enablePrimary, ctx.enableSecondary, ctx.u8
	clip := priOn && secOn
	p0p, p0m, p1p, p1m := ctx.priTaps()
	s0p, s0m, s1p, s1m := ctx.secTaps0()
	s2p, s2m, s3p, s3m := ctx.secTaps1()
	in, base, height := ctx.rows()
	out16, out8, do, ds := ctx.outRows()
	for i := 0; i < height>>2; i++ {
		r := 4 * i
		w0, w1 := cdefRow(in, base+r*BStride), cdefRow(in, base+(r+1)*BStride)
		w2, w3 := cdefRow(in, base+(r+2)*BStride), cdefRow(in, base+(r+3)*BStride)
		x := cdefWordLoad4W(w0, w1, w2, w3, cdefWinReach)
		var sum archsimd.Int16x16
		mx, mn := x, x
		if priOn {
			t0 := cdefWordLoad4W(w0, w1, w2, w3, p0p)
			t1 := cdefWordLoad4W(w0, w1, w2, w3, p0m)
			t2 := cdefWordLoad4W(w0, w1, w2, w3, p1p)
			t3 := cdefWordLoad4W(w0, w1, w2, w3, p1m)
			sum = cdefWordConstrain(t0, x, priStr, priSh, priM).Add(cdefWordConstrain(t1, x, priStr, priSh, priM)).Mul(priTap0).
				Add(cdefWordConstrain(t2, x, priStr, priSh, priM).Add(cdefWordConstrain(t3, x, priStr, priSh, priM)).Mul(priTap1))
			if clip {
				mx = mx.Max(t0.And(noSentinel)).Max(t1.And(noSentinel)).Max(t2.And(noSentinel)).Max(t3.And(noSentinel))
				mn = mn.Min(t0).Min(t1).Min(t2).Min(t3)
			}
		}
		if secOn {
			s0 := cdefWordLoad4W(w0, w1, w2, w3, s0p)
			s1 := cdefWordLoad4W(w0, w1, w2, w3, s0m)
			s2 := cdefWordLoad4W(w0, w1, w2, w3, s1p)
			s3 := cdefWordLoad4W(w0, w1, w2, w3, s1m)
			sum = sum.Add(cdefWordConstrain(s0, x, secStr, secSh, secM).Add(cdefWordConstrain(s1, x, secStr, secSh, secM)).
				Add(cdefWordConstrain(s2, x, secStr, secSh, secM)).Add(cdefWordConstrain(s3, x, secStr, secSh, secM)).Mul(secTap0))
			if clip {
				mx = mx.Max(s0.And(noSentinel)).Max(s1.And(noSentinel)).Max(s2.And(noSentinel)).Max(s3.And(noSentinel))
				mn = mn.Min(s0).Min(s1).Min(s2).Min(s3)
			}
			s4 := cdefWordLoad4W(w0, w1, w2, w3, s2p)
			s5 := cdefWordLoad4W(w0, w1, w2, w3, s2m)
			s6 := cdefWordLoad4W(w0, w1, w2, w3, s3p)
			s7 := cdefWordLoad4W(w0, w1, w2, w3, s3m)
			sum = sum.Add(cdefWordConstrain(s4, x, secStr, secSh, secM).Add(cdefWordConstrain(s5, x, secStr, secSh, secM)).
				Add(cdefWordConstrain(s6, x, secStr, secSh, secM)).Add(cdefWordConstrain(s7, x, secStr, secSh, secM)).Mul(secTap1))
			if clip {
				mx = mx.Max(s4.And(noSentinel)).Max(s5.And(noSentinel)).Max(s6.And(noSentinel)).Max(s7.And(noSentinel))
				mn = mn.Min(s4).Min(s5).Min(s6).Min(s7)
			}
		}
		y := x.Add(sum.Add(sum.ShiftAllRight(15)).Add(eight).ShiftAllRight(4))
		if clip {
			y = y.Max(mn).Min(mx)
		}
		if u8 {
			b := y.AsUint8x32().PermuteOrZeroGrouped(archsimd.LoadInt8x32Array(&cdefWordLowBytes))
			lo, hi := b.GetLo().AsUint32x4(), b.GetHi().AsUint32x4()
			cdefPut4(out8, do+r*ds, lo.GetElem(0))
			cdefPut4(out8, do+(r+1)*ds, lo.GetElem(1))
			cdefPut4(out8, do+(r+2)*ds, hi.GetElem(0))
			cdefPut4(out8, do+(r+3)*ds, hi.GetElem(1))
		} else {
			lo, hi := y.GetLo().AsUint64x2(), y.GetHi().AsUint64x2()
			cdefPutU16x4(out16, do+r*ds, lo.GetElem(0))
			cdefPutU16x4(out16, do+(r+1)*ds, lo.GetElem(1))
			cdefPutU16x4(out16, do+(r+2)*ds, hi.GetElem(0))
			cdefPutU16x4(out16, do+(r+3)*ds, hi.GetElem(1))
		}
	}
	archsimd.ClearAVXUpperBits()
}

// cdefWordShapeOK reports whether the 256-bit kernels cover a block shape:
// 8 wide with an even height, or 4 wide with a height that is a multiple of 4.
func cdefWordShapeOK(params BlockFilterParams) bool {
	switch params.Width {
	case 8:
		return params.Height&1 == 0
	case 4:
		return params.Height&3 == 0
	}
	return false
}

// filterBlockAVX2 is the uint16-output block filter on AVX2. Shapes the
// 256-bit kernels do not cover go to the 128-bit kernel.
func filterBlockAVX2(dst []uint16, dstStride int, dstOrigin int, input []uint16, inputOrigin int, params BlockFilterParams) {
	if !cdefWordShapeOK(params) {
		filterBlockSIMD(dst, dstStride, dstOrigin, input, inputOrigin, params)
		return
	}
	var ctx cdefSIMDCtx
	ctx.setBlock(dst, nil, dstOrigin, dstStride, input, inputOrigin, params, false)
	if params.Width == 8 {
		cdefWordFilter8(&ctx)
	} else {
		cdefWordFilter4(&ctx)
	}
}

// filterBlockU8AVX2 is the 8-bit-dst block filter on AVX2, under the 8-bit
// contract of filter_u8.go.
func filterBlockU8AVX2(dst []byte, dstStride int, dstOrigin int, input []uint16, inputOrigin int, params BlockFilterParams) {
	if !cdefWordShapeOK(params) {
		filterBlockU8SIMD(dst, dstStride, dstOrigin, input, inputOrigin, params)
		return
	}
	var ctx cdefSIMDCtx
	ctx.setBlock(nil, dst, dstOrigin, dstStride, input, inputOrigin, params, true)
	if params.Width == 8 {
		cdefWordFilter8(&ctx)
	} else {
		cdefWordFilter4(&ctx)
	}
}
