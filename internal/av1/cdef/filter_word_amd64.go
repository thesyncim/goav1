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

import (
	"simd/archsimd"
	"unsafe"
)

func cdefWordLd(p unsafe.Pointer) archsimd.Int16x8 {
	return archsimd.LoadInt16x8Array((*[8]int16)(p))
}

// cdefWordLoad8 packs the 8 samples at p and at p+BStride as [r0 | r1].
func cdefWordLoad8(p unsafe.Pointer) archsimd.Int16x16 {
	var z archsimd.Int16x16
	return z.SetLo(cdefWordLd(p)).SetHi(cdefWordLd(unsafe.Add(p, BStride*2)))
}

// cdefWordLoad4 packs the 4 samples of four rows at p as [r0 r1 | r2 r3]
// (BORDER_LOAD_BLOCK w=4: movq / movhps / vinserti128). The full-width loads
// read four halo samples past each row segment, inside the 8-column border.
func cdefWordLoad4(p unsafe.Pointer) archsimd.Int16x16 {
	var z archsimd.Int16x16
	ac := z.SetLo(cdefWordLd(p)).SetHi(cdefWordLd(unsafe.Add(p, 2*BStride*2))).AsUint64x4()
	bd := z.SetLo(cdefWordLd(unsafe.Add(p, BStride*2))).SetHi(cdefWordLd(unsafe.Add(p, 3*BStride*2))).AsUint64x4()
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
	pri0, pri1 := ctx.pri0*2, ctx.pri1*2
	sec0, sec1, sec2, sec3 := ctx.sec0*2, ctx.sec1*2, ctx.sec2*2, ctx.sec3*2
	src := ctx.input
	dst := ctx.dst
	dstStr := ctx.dstStr
	for h := ctx.height; h > 0; h -= 2 {
		x := cdefWordLoad8(src)
		var sum archsimd.Int16x16
		mx, mn := x, x
		if priOn {
			t0 := cdefWordLoad8(unsafe.Add(src, pri0))
			t1 := cdefWordLoad8(unsafe.Add(src, -pri0))
			t2 := cdefWordLoad8(unsafe.Add(src, pri1))
			t3 := cdefWordLoad8(unsafe.Add(src, -pri1))
			sum = cdefWordConstrain(t0, x, priStr, priSh, priM).Add(cdefWordConstrain(t1, x, priStr, priSh, priM)).Mul(priTap0).
				Add(cdefWordConstrain(t2, x, priStr, priSh, priM).Add(cdefWordConstrain(t3, x, priStr, priSh, priM)).Mul(priTap1))
			if clip {
				mx = mx.Max(t0.And(noSentinel)).Max(t1.And(noSentinel)).Max(t2.And(noSentinel)).Max(t3.And(noSentinel))
				mn = mn.Min(t0).Min(t1).Min(t2).Min(t3)
			}
		}
		if secOn {
			s0 := cdefWordLoad8(unsafe.Add(src, sec0))
			s1 := cdefWordLoad8(unsafe.Add(src, -sec0))
			s2 := cdefWordLoad8(unsafe.Add(src, sec1))
			s3 := cdefWordLoad8(unsafe.Add(src, -sec1))
			sum = sum.Add(cdefWordConstrain(s0, x, secStr, secSh, secM).Add(cdefWordConstrain(s1, x, secStr, secSh, secM)).
				Add(cdefWordConstrain(s2, x, secStr, secSh, secM)).Add(cdefWordConstrain(s3, x, secStr, secSh, secM)).Mul(secTap0))
			if clip {
				mx = mx.Max(s0.And(noSentinel)).Max(s1.And(noSentinel)).Max(s2.And(noSentinel)).Max(s3.And(noSentinel))
				mn = mn.Min(s0).Min(s1).Min(s2).Min(s3)
			}
			s4 := cdefWordLoad8(unsafe.Add(src, sec2))
			s5 := cdefWordLoad8(unsafe.Add(src, -sec2))
			s6 := cdefWordLoad8(unsafe.Add(src, sec3))
			s7 := cdefWordLoad8(unsafe.Add(src, -sec3))
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
			*(*uint64)(dst) = b.GetLo().AsUint64x2().GetElem(0)
			*(*uint64)(unsafe.Add(dst, dstStr)) = b.GetHi().AsUint64x2().GetElem(0)
		} else {
			y.GetLo().StoreArray((*[8]int16)(dst))
			y.GetHi().StoreArray((*[8]int16)(unsafe.Add(dst, dstStr)))
		}
		if h > 2 {
			src = unsafe.Add(src, 2*BStride*2)
			dst = unsafe.Add(dst, 2*dstStr)
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
	pri0, pri1 := ctx.pri0*2, ctx.pri1*2
	sec0, sec1, sec2, sec3 := ctx.sec0*2, ctx.sec1*2, ctx.sec2*2, ctx.sec3*2
	src := ctx.input
	dst := ctx.dst
	dstStr := ctx.dstStr
	for h := ctx.height; h > 0; h -= 4 {
		x := cdefWordLoad4(src)
		var sum archsimd.Int16x16
		mx, mn := x, x
		if priOn {
			t0 := cdefWordLoad4(unsafe.Add(src, pri0))
			t1 := cdefWordLoad4(unsafe.Add(src, -pri0))
			t2 := cdefWordLoad4(unsafe.Add(src, pri1))
			t3 := cdefWordLoad4(unsafe.Add(src, -pri1))
			sum = cdefWordConstrain(t0, x, priStr, priSh, priM).Add(cdefWordConstrain(t1, x, priStr, priSh, priM)).Mul(priTap0).
				Add(cdefWordConstrain(t2, x, priStr, priSh, priM).Add(cdefWordConstrain(t3, x, priStr, priSh, priM)).Mul(priTap1))
			if clip {
				mx = mx.Max(t0.And(noSentinel)).Max(t1.And(noSentinel)).Max(t2.And(noSentinel)).Max(t3.And(noSentinel))
				mn = mn.Min(t0).Min(t1).Min(t2).Min(t3)
			}
		}
		if secOn {
			s0 := cdefWordLoad4(unsafe.Add(src, sec0))
			s1 := cdefWordLoad4(unsafe.Add(src, -sec0))
			s2 := cdefWordLoad4(unsafe.Add(src, sec1))
			s3 := cdefWordLoad4(unsafe.Add(src, -sec1))
			sum = sum.Add(cdefWordConstrain(s0, x, secStr, secSh, secM).Add(cdefWordConstrain(s1, x, secStr, secSh, secM)).
				Add(cdefWordConstrain(s2, x, secStr, secSh, secM)).Add(cdefWordConstrain(s3, x, secStr, secSh, secM)).Mul(secTap0))
			if clip {
				mx = mx.Max(s0.And(noSentinel)).Max(s1.And(noSentinel)).Max(s2.And(noSentinel)).Max(s3.And(noSentinel))
				mn = mn.Min(s0).Min(s1).Min(s2).Min(s3)
			}
			s4 := cdefWordLoad4(unsafe.Add(src, sec2))
			s5 := cdefWordLoad4(unsafe.Add(src, -sec2))
			s6 := cdefWordLoad4(unsafe.Add(src, sec3))
			s7 := cdefWordLoad4(unsafe.Add(src, -sec3))
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
			*(*uint32)(dst) = lo.GetElem(0)
			*(*uint32)(unsafe.Add(dst, dstStr)) = lo.GetElem(1)
			*(*uint32)(unsafe.Add(dst, 2*dstStr)) = hi.GetElem(0)
			*(*uint32)(unsafe.Add(dst, 3*dstStr)) = hi.GetElem(1)
		} else {
			lo, hi := y.GetLo().AsUint64x2(), y.GetHi().AsUint64x2()
			*(*uint64)(dst) = lo.GetElem(0)
			*(*uint64)(unsafe.Add(dst, dstStr)) = lo.GetElem(1)
			*(*uint64)(unsafe.Add(dst, 2*dstStr)) = hi.GetElem(0)
			*(*uint64)(unsafe.Add(dst, 3*dstStr)) = hi.GetElem(1)
		}
		if h > 4 {
			src = unsafe.Add(src, 4*BStride*2)
			dst = unsafe.Add(dst, 4*dstStr)
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
	ctx.setBlock(unsafe.Pointer(&dst[dstOrigin]), dstStride*2, unsafe.Pointer(&input[inputOrigin]), params, false)
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
	ctx.setBlock(unsafe.Pointer(&dst[dstOrigin]), dstStride, unsafe.Pointer(&input[inputOrigin]), params, true)
	if params.Width == 8 {
		cdefWordFilter8(&ctx)
	} else {
		cdefWordFilter4(&ctx)
	}
}
