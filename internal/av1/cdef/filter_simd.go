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

import "simd/archsimd"

// cdefSIMDCtx is the per-block context shared by the Go SIMD kernels. The
// kernels read the uint16 tap buffer in at io (the block's first sample)
// through one bounds-checked window per row group and write out16 or out8
// (u8) at do, stride dstStr samples.
type cdefSIMDCtx struct {
	in     []uint16
	io     int
	out16  []uint16
	out8   []byte
	do     int
	dstStr int
	height int

	// Tap window indices cdefWinReach +- the direction's tap offsets: the
	// primary taps (direction+2) and the secondary taps (direction+4,
	// direction).
	taps *[12]uint

	priTap0, priTap1 int
	secTap0, secTap1 int

	priStrength, secStrength int
	priShift, secShift       int

	enablePrimary   bool
	enableSecondary bool
	u8              bool
}

// The kernels take each row's taps from one window of the tap buffer that
// starts cdefWinReach samples (two rows and two columns, the farthest tap)
// before the row's first sample and holds an 8-wide row's taps. A tap at
// offset t sits at cdefWinReach + t in the window. (Multi-row kernels take
// one window per row: a window index plus a row stride is not something the
// compiler proves in bounds.)
const (
	cdefWinReach = 2*BStride + 2
	cdefTapMax   = 2 * cdefWinReach
	cdefWin1     = cdefTapMax + 8
)

const errCDEFWindow = "cdef: block outside the tap buffer or destination"

// rows returns the tap buffer, the window base of the block's first row and
// the height, checked so that each row window below is one compare.
func (ctx *cdefSIMDCtx) rows() ([]uint16, int, int) {
	if ctx.io < cdefWinReach || ctx.io > 1<<24 || uint(ctx.height) > 128 {
		panic(errCDEFWindow)
	}
	return ctx.in, ctx.io - cdefWinReach, ctx.height
}

// outRows returns the destinations, the first output index and the stride,
// checked so that each output row costs one compare.
func (ctx *cdefSIMDCtx) outRows() ([]uint16, []byte, int, int) {
	if uint(ctx.do) > 1<<30 || uint(ctx.dstStr) > 1<<20 {
		panic(errCDEFWindow)
	}
	return ctx.out16, ctx.out8, ctx.do, ctx.dstStr
}

// priTaps returns the primary tap window indices, checked against the
// window so the tap loads need no bounds checks.
func (ctx *cdefSIMDCtx) priTaps() (uint, uint, uint, uint) {
	t := ctx.taps
	a, b, c, d := t[0], t[1], t[2], t[3]
	if a > cdefTapMax || b > cdefTapMax || c > cdefTapMax || d > cdefTapMax {
		panic(errCDEFWindow)
	}
	return a, b, c, d
}

// secTaps0 returns the window indices of the weight-2 secondary taps
// (direction+4 and direction, first tap), checked as priTaps.
func (ctx *cdefSIMDCtx) secTaps0() (uint, uint, uint, uint) {
	t := ctx.taps
	a, b, c, d := t[4], t[5], t[6], t[7]
	if a > cdefTapMax || b > cdefTapMax || c > cdefTapMax || d > cdefTapMax {
		panic(errCDEFWindow)
	}
	return a, b, c, d
}

// secTaps1 returns the window indices of the weight-1 secondary taps.
// (Two functions rather than one: eight returns exceed the inlining budget,
// and the checks only help the kernels where they are inlined.)
func (ctx *cdefSIMDCtx) secTaps1() (uint, uint, uint, uint) {
	t := ctx.taps
	a, b, c, d := t[8], t[9], t[10], t[11]
	if a > cdefTapMax || b > cdefTapMax || c > cdefTapMax || d > cdefTapMax {
		panic(errCDEFWindow)
	}
	return a, b, c, d
}

// cdefTapIdx holds, per direction, the tap window indices cdefWinReach +- t
// of the primary taps (direction+2) and the secondary taps (direction+4,
// direction): p0+, p0-, p1+, p1-, s0+, s0-, s1+, s1-, s2+, s2-, s3+, s3-.
var cdefTapIdx = func() (t [8][12]uint) {
	for d := range t {
		offs := [6]int{
			int(cdefDirections[d+2][0]), int(cdefDirections[d+2][1]),
			int(cdefDirections[d+4][0]), int(cdefDirections[d][0]),
			int(cdefDirections[d+4][1]), int(cdefDirections[d][1]),
		}
		for k, o := range offs {
			t[d][2*k], t[d][2*k+1] = uint(cdefWinReach+o), uint(cdefWinReach-o)
		}
	}
	return
}()

// setDirection points the context at the direction's tap window indices.
func (ctx *cdefSIMDCtx) setDirection(direction int) {
	ctx.taps = &cdefTapIdx[direction&7]
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
func (ctx *cdefSIMDCtx) setBlock(out16 []uint16, out8 []byte, do, dstStr int, in []uint16, io int, params BlockFilterParams, u8 bool) {
	ctx.out16, ctx.out8, ctx.do = out16, out8, do
	ctx.in, ctx.io = in, io
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
	ctx.setBlock(dst, nil, dstOrigin, dstStride, input, inputOrigin, params, false)
	cdefFilterBlockSIMD(&ctx, int(params.Width))
}

// cdefLdW1 loads the 8 samples at window index i as Int16x8 (samples <=
// 0x4000, the bit pattern is a non-negative int16).
func cdefLdW1(w *[cdefWin1]uint16, i uint) archsimd.Int16x8 {
	return archsimd.LoadUint16x8Array((*[8]uint16)(w[i : i+8])).BitsToInt16()
}

// cdefRow is the tap window of the row whose window starts at o.
func cdefRow(in []uint16, o int) *[cdefWin1]uint16 {
	return (*[cdefWin1]uint16)(in[o : o+cdefWin1])
}

// cdefRows2 returns the tap windows of two consecutive rows, the window of
// the first starting at o: one bounds check for the pair, the second row's
// window a constant-offset slice of the same array.
func cdefRows2(in []uint16, o int) (*[cdefWin1]uint16, *[cdefWin1]uint16) {
	g := (*[cdefWin1 + BStride]uint16)(in[o : o+cdefWin1+BStride])
	return (*[cdefWin1]uint16)(g[0:cdefWin1]), (*[cdefWin1]uint16)(g[BStride : BStride+cdefWin1])
}

// cdefRows4 is cdefRows2 for four consecutive rows.
func cdefRows4(in []uint16, o int) (*[cdefWin1]uint16, *[cdefWin1]uint16, *[cdefWin1]uint16, *[cdefWin1]uint16) {
	g := (*[cdefWin1 + 3*BStride]uint16)(in[o : o+cdefWin1+3*BStride])
	return (*[cdefWin1]uint16)(g[0:cdefWin1]), (*[cdefWin1]uint16)(g[BStride : BStride+cdefWin1]),
		(*[cdefWin1]uint16)(g[2*BStride : 2*BStride+cdefWin1]), (*[cdefWin1]uint16)(g[3*BStride : 3*BStride+cdefWin1])
}

// cdefLdPair zips the low four samples at window index i of two rows' windows
// into one Int16x8 (lanes 0..3 = row r, lanes 4..7 = row r+1), the vector
// shape the 4-wide kernel filters two rows at a time with. The full-width
// loads read four halo samples past each 4-wide row segment; the CDEF input
// buffer's 8-column horizontal border keeps them in bounds.
func cdefLdPair(w0, w1 *[cdefWin1]uint16, i uint) archsimd.Int16x8 {
	lo := archsimd.LoadUint16x8Array((*[8]uint16)(w0[i : i+8])).ReshapeToUint64s()
	hi := archsimd.LoadUint16x8Array((*[8]uint16)(w1[i : i+8])).ReshapeToUint64s()
	return lo.InterleaveLo(hi).ReshapeToUint16s().BitsToInt16()
}

// cdefOut16 is the 8-sample output row of out at o.
func cdefOut16(out []uint16, o int) *[8]uint16 {
	return (*[8]uint16)(out[o : o+8])
}

// cdefPut8 stores the 8 bytes of v at out[o:].
func cdefPut8(out []byte, o int, v uint64) {
	b := (*[8]byte)(out[o : o+8])
	b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7] = byte(v), byte(v>>8), byte(v>>16), byte(v>>24), byte(v>>32), byte(v>>40), byte(v>>48), byte(v>>56)
}

// cdefPut4 stores the 4 bytes of v at out[o:].
func cdefPut4(out []byte, o int, v uint32) {
	b := (*[4]byte)(out[o : o+4])
	b[0], b[1], b[2], b[3] = byte(v), byte(v>>8), byte(v>>16), byte(v>>24)
}

// cdefPutU16x4 stores the four uint16 samples packed in v at out[o:].
func cdefPutU16x4(out []uint16, o int, v uint64) {
	b := (*[4]uint16)(out[o : o+4])
	b[0], b[1], b[2], b[3] = uint16(v), uint16(v>>16), uint16(v>>32), uint16(v>>48)
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
	sar15 := cdefShiftCountOf(15)
	sar4 := cdefShiftCountOf(4)
	priOn, secOn, u8 := ctx.enablePrimary, ctx.enableSecondary, ctx.u8
	clip := priOn && secOn
	p0p, p0m, p1p, p1m := ctx.priTaps()
	s0p, s0m, s1p, s1m := ctx.secTaps0()
	s2p, s2m, s3p, s3m := ctx.secTaps1()
	in, base, height := ctx.rows()
	out16, out8, do, ds := ctx.outRows()
	for r := 0; r < height; r++ {
		w := cdefRow(in, base+r*BStride)
		x := cdefLdW1(w, cdefWinReach)
		sum := zero
		mx, mn := x, x
		if priOn {
			t0 := cdefLdW1(w, p0p)
			t1 := cdefLdW1(w, p0m)
			t2 := cdefLdW1(w, p1p)
			t3 := cdefLdW1(w, p1m)
			// Each constrain result feeds an Add on its own line: a call that is
			// alone on its line leaves an inline-mark NOP in the row loop.
			sum = cdefConstrain(t0, x, priStr, priSh).Add(cdefConstrain(t1, x, priStr, priSh)).Mul(priTap0).
				Add(cdefConstrain(t2, x, priStr, priSh).Add(cdefConstrain(t3, x, priStr, priSh)).Mul(priTap1))
			if clip {
				mx = mx.Max(t0.And(noSentinel)).Max(t1.And(noSentinel)).Max(t2.And(noSentinel)).Max(t3.And(noSentinel))
				mn = mn.Min(t0).Min(t1).Min(t2).Min(t3)
			}
		}
		if secOn {
			s0 := cdefLdW1(w, s0p)
			s1 := cdefLdW1(w, s0m)
			s2 := cdefLdW1(w, s1p)
			s3 := cdefLdW1(w, s1m)
			// Each half of the secondary taps folds into sum, max and min before the
			// next half loads, so only four taps are live at once.
			sum = sum.Add(cdefConstrain(s0, x, secStr, secSh).Add(cdefConstrain(s1, x, secStr, secSh)).
				Add(cdefConstrain(s2, x, secStr, secSh)).Add(cdefConstrain(s3, x, secStr, secSh)).Mul(secTap0))
			if clip {
				mx = mx.Max(s0.And(noSentinel)).Max(s1.And(noSentinel)).Max(s2.And(noSentinel)).Max(s3.And(noSentinel))
				mn = mn.Min(s0).Min(s1).Min(s2).Min(s3)
			}
			s4 := cdefLdW1(w, s2p)
			s5 := cdefLdW1(w, s2m)
			s6 := cdefLdW1(w, s3p)
			s7 := cdefLdW1(w, s3m)
			sum = sum.Add(cdefConstrain(s4, x, secStr, secSh).Add(cdefConstrain(s5, x, secStr, secSh)).
				Add(cdefConstrain(s6, x, secStr, secSh)).Add(cdefConstrain(s7, x, secStr, secSh)).Mul(secTap1))
			if clip {
				mx = mx.Max(s4.And(noSentinel)).Max(s5.And(noSentinel)).Max(s6.And(noSentinel)).Max(s7.And(noSentinel))
				mn = mn.Min(s4).Min(s5).Min(s6).Min(s7)
			}
		}
		// y = x + ((8 + sum - (sum<0)) >> 4)
		y := x.Add(cdefSarI16(sum.Add(cdefSarI16(sum, sar15)).Add(eight), sar4))
		if clip {
			y = y.Max(mn).Min(mx)
		}
		if u8 {
			cdefPut8(out8, do+r*ds, cdefNarrowU8(y).ReshapeToUint64s().GetElem(0))
		} else {
			y.ToBits().StoreArray(cdefOut16(out16, do+r*ds))
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
	sar15 := cdefShiftCountOf(15)
	sar4 := cdefShiftCountOf(4)
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
		w0, w1 := cdefRows2(in, base+r*BStride)
		x := cdefLdPair(w0, w1, cdefWinReach)
		sum := zero
		mx, mn := x, x
		if priOn {
			t0 := cdefLdPair(w0, w1, p0p)
			t1 := cdefLdPair(w0, w1, p0m)
			t2 := cdefLdPair(w0, w1, p1p)
			t3 := cdefLdPair(w0, w1, p1m)
			sum = cdefConstrain(t0, x, priStr, priSh).Add(cdefConstrain(t1, x, priStr, priSh)).Mul(priTap0).
				Add(cdefConstrain(t2, x, priStr, priSh).Add(cdefConstrain(t3, x, priStr, priSh)).Mul(priTap1))
			if clip {
				mx = mx.Max(t0.And(noSentinel)).Max(t1.And(noSentinel)).Max(t2.And(noSentinel)).Max(t3.And(noSentinel))
				mn = mn.Min(t0).Min(t1).Min(t2).Min(t3)
			}
		}
		if secOn {
			s0 := cdefLdPair(w0, w1, s0p)
			s1 := cdefLdPair(w0, w1, s0m)
			s2 := cdefLdPair(w0, w1, s1p)
			s3 := cdefLdPair(w0, w1, s1m)
			// Each half of the secondary taps folds into sum, max and min before the
			// next half loads, so only four taps are live at once.
			sum = sum.Add(cdefConstrain(s0, x, secStr, secSh).Add(cdefConstrain(s1, x, secStr, secSh)).
				Add(cdefConstrain(s2, x, secStr, secSh)).Add(cdefConstrain(s3, x, secStr, secSh)).Mul(secTap0))
			if clip {
				mx = mx.Max(s0.And(noSentinel)).Max(s1.And(noSentinel)).Max(s2.And(noSentinel)).Max(s3.And(noSentinel))
				mn = mn.Min(s0).Min(s1).Min(s2).Min(s3)
			}
			s4 := cdefLdPair(w0, w1, s2p)
			s5 := cdefLdPair(w0, w1, s2m)
			s6 := cdefLdPair(w0, w1, s3p)
			s7 := cdefLdPair(w0, w1, s3m)
			sum = sum.Add(cdefConstrain(s4, x, secStr, secSh).Add(cdefConstrain(s5, x, secStr, secSh)).
				Add(cdefConstrain(s6, x, secStr, secSh)).Add(cdefConstrain(s7, x, secStr, secSh)).Mul(secTap1))
			if clip {
				mx = mx.Max(s4.And(noSentinel)).Max(s5.And(noSentinel)).Max(s6.And(noSentinel)).Max(s7.And(noSentinel))
				mn = mn.Min(s4).Min(s5).Min(s6).Min(s7)
			}
		}
		y := x.Add(cdefSarI16(sum.Add(cdefSarI16(sum, sar15)).Add(eight), sar4))
		if clip {
			y = y.Max(mn).Min(mx)
		}
		if u8 {
			out := cdefNarrowU8(y).ReshapeToUint32s()
			cdefPut4(out8, do+r*ds, out.GetElem(0))
			cdefPut4(out8, do+(r+1)*ds, out.GetElem(1))
		} else {
			out := y.ToBits().ReshapeToUint64s()
			cdefPutU16x4(out16, do+r*ds, out.GetElem(0))
			cdefPutU16x4(out16, do+(r+1)*ds, out.GetElem(1))
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
	ctx.setBlock(nil, dst, dstOrigin, dstStride, input, inputOrigin, params, true)
	cdefFilterBlockSIMD(&ctx, int(params.Width))
}
