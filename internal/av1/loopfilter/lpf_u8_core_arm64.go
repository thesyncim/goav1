// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package loopfilter

import "simd/archsimd"

// The 8-bit kernel arithmetic, shared by the horizontal (row) and transposed
// vertical (column) kernels. Each helper is an exact rewrite of a dav1d
// lpf_16_wd* step (src/arm/64/loopfilter.S) into the operations archsimd has
// on arm64; the comments name the dav1d instructions each replaces.

// lpfSpread returns max over x in {a, b} of |x - c|, dav1d's two uabd and a
// umax for the flat8in/flat8out tests. archsimd has no UABD (|x - c| costs
// two uqsub and an orr here), but against one centre the maximum distance is
// max(max(a, b) - c, c - min(a, b)) in saturating arithmetic: five
// operations where two emulated uabd and a umax take seven, and each extra
// tap adds a max and a min instead of four.
func lpfSpread(a, b, c archsimd.Uint8x16) archsimd.Uint8x16 {
	return a.Max(b).SubSaturated(c).Max(c.SubSaturated(a.Min(b)))
}

// lpfSpread3 is lpfSpread over three taps.
func lpfSpread3(a, b, d, c archsimd.Uint8x16) archsimd.Uint8x16 {
	return a.Max(b).Max(d).SubSaturated(c).Max(c.SubSaturated(a.Min(b).Min(d)))
}

// The flat filters run on widened lanes pre-scaled by 2^(8-shift) (32 for
// the >>3 filters, 16 for the >>4 one): the widening multiply that scales
// costs what a plain widen does (umull/umull2 against uxtl/uxtl2), the
// rounding bias rides in the accumulator, and the rounded, narrowed output
// byte is then the high byte of each 16-bit lane, one uzp2 per sixteen
// outputs where dav1d spends rshrn + rshrn2. Every intermediate is exact mod
// 2^16 and every output sum is below 2^16 (8*255+4 scaled by 32, 16*255+8 by
// 16), so the high byte is exactly (sum + bias) >> shift.

// lpfNarrowHi packs the high bytes of the low- and high-half sums (uzp2).
func lpfNarrowHi(lo, hi archsimd.Uint16x8) archsimd.Uint8x16 {
	return lo.ReshapeToUint8s().ConcatOdd(hi.ReshapeToUint8s())
}

// lpfFlat6Half is the dav1d wd6 sliding sum (out p1, p0, q0, q1) on scaled
// widened lanes; s is the scaled, biased 3*p2 + 2*p1 + 2*p0 + q0 + 4.
func lpfFlat6Half(P2, P1, P0, Q0, Q1, Q2, s archsimd.Uint16x8) (a1, a0, b0, b1 archsimd.Uint16x8) {
	a1 = s
	s = s.Add(Q0.Add(Q1)).Sub(P2.Add(P2))
	a0 = s
	s = s.Add(Q1.Add(Q2)).Sub(P2.Add(P1))
	b0 = s
	b1 = s.Add(Q2.Add(Q2)).Sub(P1.Add(P0))
	return
}

// lpfFlat8Half is the dav1d wd8 sliding sum (out p2..q2) on scaled widened
// lanes; s is the scaled, biased 3*p3 + 2*p2 + p1 + p0 + q0 + 4.
func lpfFlat8Half(P3, P2, P1, P0, Q0, Q1, Q2, Q3, s archsimd.Uint16x8) (a2, a1, a0, b0, b1, b2 archsimd.Uint16x8) {
	a2 = s
	s = s.Add(P1.Add(Q1)).Sub(P3.Add(P2))
	a1 = s
	s = s.Add(P0.Add(Q2)).Sub(P3.Add(P1))
	a0 = s
	s = s.Add(Q0.Add(Q3)).Sub(P3.Add(P0))
	b0 = s
	s = s.Add(Q1.Add(Q3)).Sub(P2.Add(Q0))
	b1 = s
	b2 = s.Add(Q2.Add(Q3)).Sub(P1.Add(Q1))
	return
}

// lpfWiden widens both halves of x scaled by k (umull/umull2): the scaled
// widen the flat filters accumulate in.
func lpfWiden(x, k archsimd.Uint8x16) (lo, hi archsimd.Uint16x8) {
	return x.MulWidenLo(k), x.HiToLo().MulWidenLo(k.HiToLo())
}
