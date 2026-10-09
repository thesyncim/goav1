// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && (arm64 || amd64) && !purego

package cdef

// Go-native SIMD CDEF block kernels (simd/archsimd, GOEXPERIMENT=simd). One
// row-loop body serves every strength split (primary-only, secondary-only,
// fused primary+secondary) and both output widths (uint16 frames and the 8-bit
// frame plane), selected by per-block flags that are loop-invariant. Each
// kernel is a byte-exact twin of filterBlockPureGo / filterBlockU8PureGo:
//   - constrain() is lane-wise: lim = uqsub(strength, |diff| >> shift), then
//     diff is clamped to [-lim, +lim]. That is identical to constrainShifted
//     because min(max(t-x) ...) folds the sign back in exactly.
//   - tap weights accumulate in int16 lanes; the worst case |sum| is
//     2*(4+2)*pri + 4*(2+1)*sec, at most 3648 at 12-bit, so no lane wraps.
//   - the final x + ((8 + sum - (sum<0)) >> 4) is computed as x + ((sum + (sum>>15) + 8) >> 4)
//     with an arithmetic shift, the same bits as the reference.
//   - when both strengths are active the result clamps to [min, max] over the
//     tap neighbourhood. maxClip skips the VeryLarge (0x4000) sentinel; masking
//     the sentinel to zero before the max fold is equivalent because the
//     running max starts at the center sample, which is never negative.
//     min4 has no sentinel skip, so the minimum uses the raw taps.
//   - the 8-bit store narrows with unsigned saturation. Under the 8-bit
//     contract the value is already in [0, 255], so this equals byte(y).

import (
	"simd/archsimd"
	"unsafe"
)

// cdefSIMDCtx is the per-block context shared by the Go SIMD kernels. Tap
// offsets are signed sample offsets into the uint16 input buffer; dstStr is in
// bytes. dst points at the first output sample (*uint16 or *byte when u8).
type cdefSIMDCtx struct {
	dst    unsafe.Pointer
	input  unsafe.Pointer // input[inputOrigin]
	dstStr int
	height int

	pri0, pri1             int // primary tap offsets (direction+2)
	sec0, sec1, sec2, sec3 int // secondary tap offsets (direction+4, direction)

	priTap0, priTap1 int
	secTap0, secTap1 int

	priStrength, secStrength int
	priShift, secShift       int

	enablePrimary   bool
	enableSecondary bool
	u8              bool
}

// setDirection fills the direction-dependent tap offsets.
func (ctx *cdefSIMDCtx) setDirection(direction int) {
	ctx.pri0 = int(cdefDirections[direction+2][0])
	ctx.pri1 = int(cdefDirections[direction+2][1])
	ctx.sec0 = int(cdefDirections[direction+4][0])
	ctx.sec1 = int(cdefDirections[direction][0])
	ctx.sec2 = int(cdefDirections[direction+4][1])
	ctx.sec3 = int(cdefDirections[direction][1])
}

// setPrimary fills the primary-strength-dependent fields.
func (ctx *cdefSIMDCtx) setPrimary(strength, damping, coeffShift int) {
	priTaps := cdefPrimaryTaps[(strength>>coeffShift)&1]
	ctx.priTap0 = int(priTaps[0])
	ctx.priTap1 = int(priTaps[1])
	ctx.priStrength = strength
	ctx.priShift = constrainShift(strength, damping)
	ctx.enablePrimary = strength != 0
}

// setSecondary fills the secondary-strength-dependent fields.
func (ctx *cdefSIMDCtx) setSecondary(strength, damping int) {
	ctx.secTap0 = int(cdefSecondaryTaps[0])
	ctx.secTap1 = int(cdefSecondaryTaps[1])
	ctx.secStrength = strength
	ctx.secShift = constrainShift(strength, damping)
	ctx.enableSecondary = strength != 0
}

// setBlock fills the full context for one block in place; a by-value return
// would copy the struct through the stack on every block.
func (ctx *cdefSIMDCtx) setBlock(dst unsafe.Pointer, dstStr int, input unsafe.Pointer, params BlockFilterParams, u8 bool) {
	ctx.dst = dst
	ctx.input = input
	ctx.dstStr = dstStr
	ctx.height = int(params.Height)
	ctx.u8 = u8
	ctx.setDirection(int(params.Direction))
	ctx.setPrimary(int(params.PrimaryStrength), int(params.PrimaryDamping), int(params.CoeffShift))
	ctx.setSecondary(int(params.SecondaryStrength), int(params.SecondaryDamping))
}

// cdefFilterBlockSIMD routes a prepared context to the width-specialized
// kernel. width must be 8, or 4 with an even height.
func cdefFilterBlockSIMD(ctx *cdefSIMDCtx, width int) {
	if width == 8 {
		cdefFilterBlock8SIMD(ctx)
		return
	}
	cdefFilterBlock4SIMD(ctx)
}

// filterBlockSIMD is the uint16-output Go SIMD block filter. It routes narrow
// shapes the kernels do not cover (odd 4-wide heights) to the pure-Go
// reference, and otherwise matches filterBlockPureGo sample for sample.
func filterBlockSIMD(dst []uint16, dstStride int, dstOrigin int, input []uint16, inputOrigin int, params BlockFilterParams) {
	if w := int(params.Width); (w != 8 && w != 4) || (w == 4 && params.Height&1 != 0) {
		filterBlockPureGo(dst, dstStride, dstOrigin, input, inputOrigin, params)
		return
	}
	var ctx cdefSIMDCtx
	ctx.setBlock(unsafe.Pointer(&dst[dstOrigin]), dstStride*2, unsafe.Pointer(&input[inputOrigin]), params, false)
	cdefFilterBlockSIMD(&ctx, int(params.Width))
}

// cdefLoadU16P loads 8 uint16 CDEF samples at a raw pointer as Int16x8
// (samples <= 0x4000, the bit pattern is a non-negative int16).
func cdefLoadU16P(p unsafe.Pointer) archsimd.Int16x8 {
	return archsimd.LoadInt16x8Array((*[8]int16)(p))
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

// cdefConstrain is constrain() on eight lanes: t and x are non-negative
// samples, str is the broadcast strength, sh the broadcast-or-scalar right
// shift from cdefShiftCountOf.
func cdefConstrain(t, x archsimd.Int16x8, str archsimd.Uint16x8, sh cdefShiftCount) archsimd.Int16x8 {
	d := t.Sub(x)
	lim := str.SubSaturated(cdefShrU16(d.Abs().ToBits(), sh)).BitsToInt16()
	return d.Min(lim).Max(lim.Neg())
}

// cdefFilterBlock8SIMD filters an 8-wide block, one row per vector.
func cdefFilterBlock8SIMD(ctx *cdefSIMDCtx) {
	zero := archsimd.BroadcastInt16x8(0)
	priStr := archsimd.BroadcastInt16x8(int16(ctx.priStrength)).ToBits()
	secStr := archsimd.BroadcastInt16x8(int16(ctx.secStrength)).ToBits()
	priSh := cdefShiftCountOf(ctx.priShift)
	secSh := cdefShiftCountOf(ctx.secShift)
	priTap0 := archsimd.BroadcastInt16x8(int16(ctx.priTap0))
	priTap1 := archsimd.BroadcastInt16x8(int16(ctx.priTap1))
	secTap0 := archsimd.BroadcastInt16x8(int16(ctx.secTap0))
	secTap1 := archsimd.BroadcastInt16x8(int16(ctx.secTap1))
	eight := archsimd.BroadcastInt16x8(8)
	noSentinel := archsimd.BroadcastInt16x8(int16(^VeryLarge))
	priOn, secOn, u8 := ctx.enablePrimary, ctx.enableSecondary, ctx.u8
	clip := priOn && secOn
	pri0, pri1 := ctx.pri0*2, ctx.pri1*2
	sec0, sec1, sec2, sec3 := ctx.sec0*2, ctx.sec1*2, ctx.sec2*2, ctx.sec3*2
	src := ctx.input
	dst := ctx.dst
	dstStr := ctx.dstStr
	for h := ctx.height; h > 0; h-- {
		x := cdefLoadU16P(src)
		sum := zero
		mx, mn := x, x
		if priOn {
			t0 := cdefLoadU16P(unsafe.Add(src, pri0))
			t1 := cdefLoadU16P(unsafe.Add(src, -pri0))
			t2 := cdefLoadU16P(unsafe.Add(src, pri1))
			t3 := cdefLoadU16P(unsafe.Add(src, -pri1))
			c0 := cdefConstrain(t0, x, priStr, priSh)
			c1 := cdefConstrain(t1, x, priStr, priSh)
			c2 := cdefConstrain(t2, x, priStr, priSh)
			c3 := cdefConstrain(t3, x, priStr, priSh)
			sum = c0.Add(c1).Mul(priTap0).Add(c2.Add(c3).Mul(priTap1))
			if clip {
				mx = mx.Max(t0.And(noSentinel)).Max(t1.And(noSentinel)).Max(t2.And(noSentinel)).Max(t3.And(noSentinel))
				mn = mn.Min(t0).Min(t1).Min(t2).Min(t3)
			}
		}
		if secOn {
			s0 := cdefLoadU16P(unsafe.Add(src, sec0))
			s1 := cdefLoadU16P(unsafe.Add(src, -sec0))
			s2 := cdefLoadU16P(unsafe.Add(src, sec1))
			s3 := cdefLoadU16P(unsafe.Add(src, -sec1))
			s4 := cdefLoadU16P(unsafe.Add(src, sec2))
			s5 := cdefLoadU16P(unsafe.Add(src, -sec2))
			s6 := cdefLoadU16P(unsafe.Add(src, sec3))
			s7 := cdefLoadU16P(unsafe.Add(src, -sec3))
			c0 := cdefConstrain(s0, x, secStr, secSh)
			c1 := cdefConstrain(s1, x, secStr, secSh)
			c2 := cdefConstrain(s2, x, secStr, secSh)
			c3 := cdefConstrain(s3, x, secStr, secSh)
			c4 := cdefConstrain(s4, x, secStr, secSh)
			c5 := cdefConstrain(s5, x, secStr, secSh)
			c6 := cdefConstrain(s6, x, secStr, secSh)
			c7 := cdefConstrain(s7, x, secStr, secSh)
			sum = sum.Add(c0.Add(c1).Add(c2).Add(c3).Mul(secTap0)).Add(c4.Add(c5).Add(c6).Add(c7).Mul(secTap1))
			if clip {
				mx = mx.Max(s0.And(noSentinel)).Max(s1.And(noSentinel)).Max(s2.And(noSentinel)).Max(s3.And(noSentinel))
				mx = mx.Max(s4.And(noSentinel)).Max(s5.And(noSentinel)).Max(s6.And(noSentinel)).Max(s7.And(noSentinel))
				mn = mn.Min(s0).Min(s1).Min(s2).Min(s3)
				mn = mn.Min(s4).Min(s5).Min(s6).Min(s7)
			}
		}
		// y = x + ((8 + sum - (sum<0)) >> 4)
		y := x.Add(sum.Add(sum.ShiftAllRight(15)).Add(eight).ShiftAllRight(4))
		if clip {
			y = y.Max(mn).Min(mx)
		}
		if u8 {
			*(*uint64)(dst) = cdefNarrowU8(y).ReshapeToUint64s().GetElem(0)
		} else {
			y.ToBits().StoreArray((*[8]uint16)(dst))
		}
		if h > 1 {
			src = unsafe.Add(src, BStride*2)
			dst = unsafe.Add(dst, dstStr)
		}
	}
}

// cdefFilterBlock4SIMD filters a 4-wide block, two rows per vector (rows r
// and r+1 in lanes 0..3 and 4..7). The height is even by dispatch contract.
func cdefFilterBlock4SIMD(ctx *cdefSIMDCtx) {
	zero := archsimd.BroadcastInt16x8(0)
	priStr := archsimd.BroadcastInt16x8(int16(ctx.priStrength)).ToBits()
	secStr := archsimd.BroadcastInt16x8(int16(ctx.secStrength)).ToBits()
	priSh := cdefShiftCountOf(ctx.priShift)
	secSh := cdefShiftCountOf(ctx.secShift)
	priTap0 := archsimd.BroadcastInt16x8(int16(ctx.priTap0))
	priTap1 := archsimd.BroadcastInt16x8(int16(ctx.priTap1))
	secTap0 := archsimd.BroadcastInt16x8(int16(ctx.secTap0))
	secTap1 := archsimd.BroadcastInt16x8(int16(ctx.secTap1))
	eight := archsimd.BroadcastInt16x8(8)
	noSentinel := archsimd.BroadcastInt16x8(int16(^VeryLarge))
	priOn, secOn, u8 := ctx.enablePrimary, ctx.enableSecondary, ctx.u8
	clip := priOn && secOn
	pri0, pri1 := ctx.pri0*2, ctx.pri1*2
	sec0, sec1, sec2, sec3 := ctx.sec0*2, ctx.sec1*2, ctx.sec2*2, ctx.sec3*2
	src := ctx.input
	dst := ctx.dst
	dstStr := ctx.dstStr
	for h := ctx.height; h > 0; h -= 2 {
		src2 := unsafe.Add(src, BStride*2)
		x := cdefLoadPairU16P(src, src2)
		sum := zero
		mx, mn := x, x
		if priOn {
			t0 := cdefLoadPairU16P(unsafe.Add(src, pri0), unsafe.Add(src2, pri0))
			t1 := cdefLoadPairU16P(unsafe.Add(src, -pri0), unsafe.Add(src2, -pri0))
			t2 := cdefLoadPairU16P(unsafe.Add(src, pri1), unsafe.Add(src2, pri1))
			t3 := cdefLoadPairU16P(unsafe.Add(src, -pri1), unsafe.Add(src2, -pri1))
			c0 := cdefConstrain(t0, x, priStr, priSh)
			c1 := cdefConstrain(t1, x, priStr, priSh)
			c2 := cdefConstrain(t2, x, priStr, priSh)
			c3 := cdefConstrain(t3, x, priStr, priSh)
			sum = c0.Add(c1).Mul(priTap0).Add(c2.Add(c3).Mul(priTap1))
			if clip {
				mx = mx.Max(t0.And(noSentinel)).Max(t1.And(noSentinel)).Max(t2.And(noSentinel)).Max(t3.And(noSentinel))
				mn = mn.Min(t0).Min(t1).Min(t2).Min(t3)
			}
		}
		if secOn {
			s0 := cdefLoadPairU16P(unsafe.Add(src, sec0), unsafe.Add(src2, sec0))
			s1 := cdefLoadPairU16P(unsafe.Add(src, -sec0), unsafe.Add(src2, -sec0))
			s2 := cdefLoadPairU16P(unsafe.Add(src, sec1), unsafe.Add(src2, sec1))
			s3 := cdefLoadPairU16P(unsafe.Add(src, -sec1), unsafe.Add(src2, -sec1))
			s4 := cdefLoadPairU16P(unsafe.Add(src, sec2), unsafe.Add(src2, sec2))
			s5 := cdefLoadPairU16P(unsafe.Add(src, -sec2), unsafe.Add(src2, -sec2))
			s6 := cdefLoadPairU16P(unsafe.Add(src, sec3), unsafe.Add(src2, sec3))
			s7 := cdefLoadPairU16P(unsafe.Add(src, -sec3), unsafe.Add(src2, -sec3))
			c0 := cdefConstrain(s0, x, secStr, secSh)
			c1 := cdefConstrain(s1, x, secStr, secSh)
			c2 := cdefConstrain(s2, x, secStr, secSh)
			c3 := cdefConstrain(s3, x, secStr, secSh)
			c4 := cdefConstrain(s4, x, secStr, secSh)
			c5 := cdefConstrain(s5, x, secStr, secSh)
			c6 := cdefConstrain(s6, x, secStr, secSh)
			c7 := cdefConstrain(s7, x, secStr, secSh)
			sum = sum.Add(c0.Add(c1).Add(c2).Add(c3).Mul(secTap0)).Add(c4.Add(c5).Add(c6).Add(c7).Mul(secTap1))
			if clip {
				mx = mx.Max(s0.And(noSentinel)).Max(s1.And(noSentinel)).Max(s2.And(noSentinel)).Max(s3.And(noSentinel))
				mx = mx.Max(s4.And(noSentinel)).Max(s5.And(noSentinel)).Max(s6.And(noSentinel)).Max(s7.And(noSentinel))
				mn = mn.Min(s0).Min(s1).Min(s2).Min(s3)
				mn = mn.Min(s4).Min(s5).Min(s6).Min(s7)
			}
		}
		y := x.Add(sum.Add(sum.ShiftAllRight(15)).Add(eight).ShiftAllRight(4))
		if clip {
			y = y.Max(mn).Min(mx)
		}
		if u8 {
			out := cdefNarrowU8(y).ReshapeToUint32s()
			*(*uint32)(dst) = out.GetElem(0)
			*(*uint32)(unsafe.Add(dst, dstStr)) = out.GetElem(1)
		} else {
			out := y.ToBits().ReshapeToUint64s()
			*(*uint64)(dst) = out.GetElem(0)
			*(*uint64)(unsafe.Add(dst, dstStr)) = out.GetElem(1)
		}
		if h > 2 {
			src = unsafe.Add(src, 2*BStride*2)
			dst = unsafe.Add(dst, 2*dstStr)
		}
	}
}

// filterBlockU8SIMD is the 8-bit-dst Go SIMD block filter. It shares the
// uint16 kernel body; the store narrows each row to bytes in place, under the
// 8-bit contract documented in filter_u8.go. It matches filterBlockU8PureGo
// sample for sample.
func filterBlockU8SIMD(dst []byte, dstStride int, dstOrigin int, input []uint16, inputOrigin int, params BlockFilterParams) {
	if w := int(params.Width); (w != 8 && w != 4) || (w == 4 && params.Height&1 != 0) {
		filterBlockU8PureGo(dst, dstStride, dstOrigin, input, inputOrigin, params)
		return
	}
	var ctx cdefSIMDCtx
	ctx.setBlock(unsafe.Pointer(&dst[dstOrigin]), dstStride, unsafe.Pointer(&input[inputOrigin]), params, true)
	cdefFilterBlockSIMD(&ctx, int(params.Width))
}
