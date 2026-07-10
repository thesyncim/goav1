// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

// int32 8-wide Go-native SIMD Wiener vertical restoration pass. This is a
// compute-bound 7-tap column convolution (the same shape as the inverse-DCT
// column pass): eight columns are processed per iteration, the seven vertical
// taps are fused widening multiply-accumulated into two int32 lane groups
// (columns 0..3 and 4..7), then the accumulator is round-shifted, clamped to
// [0,max] and narrowed back to uint16.
//
// Fused widening MAC (SMLAL / SMLAL2 via Int32x4.MulWidenLoAdd/HiAdd): every
// temp sample is the horizontal pass's output, clamped to [0,maxClamp] with
// maxClamp <= 32767 for all bit depths (8-bit: 8191, 10/12-bit: 32767). A value
// in [0,32767] has the same bit pattern reinterpreted as a non-negative int16,
// so int16*int16->int32 reproduces the reference's uint16-widened
// int32 product exactly, and the 7-tap int32 sum never overflows. This is the
// identical sample-range invariant the hand-written NEON asm relies on, so the
// kernel is byte-identical to wienerVertical (TestWienerVerticalSIMDMatchesPureGo).
// The two accumulators are seeded so each tap is one SMLAL/SMLAL2 (14 total),
// matching the NEON asm's fused-MAC op count.
//
// Two folds make the inner loop a plain 7-tap MAC (shared with the NEON asm
// wrapper): the libaom center reapplication s3<<WienerFilterBits is folded into
// the center tap (tap3 += 1<<WienerFilterBits), and the rounding bias
// 1<<(round1-1) together with -offset is folded into the accumulator seed so the
// trailing arithmetic right shift reproduces roundPowerOfTwo(sum, round1)
// exactly.
//
// Widths that are not a multiple of 8 (and widths below one 8-lane vector) fall
// back to the scalar wienerVertical reference.

package restoration

import (
	"simd/archsimd"
	"unsafe"

	"github.com/thesyncim/goav1/internal/av1/dsp/cpu"
)

// init binds the Go-native SIMD Wiener vertical pass as the dispatch kernel
// under the goexperiment.simd build (the NEON asm binding in
// wiener_dispatch_arm64.go is excluded there via !goexperiment.simd). The two
// horizontal passes keep the hand-written NEON asm, which is still compiled in
// this build (wiener_neon_arm64.go is tagged arm64 && !purego). Every binding is
// byte-identical to the pure-Go reference.
func init() {
	_ = cpu.Detected // ensure cpu package init runs before this point
	if cpu.Detected.NEON {
		wienerHorizontalImpl = wienerHorizontalNEON
		wienerHorizontalTrustedImpl = wienerHorizontalNEONTrusted
	} else {
		wienerHorizontalImpl = wienerHorizontal
		wienerHorizontalTrustedImpl = wienerHorizontalTrusted
	}
	wienerVerticalImpl = wienerVerticalSIMD
}

// wienerLoadV reinterprets the 8 uint16 samples at pointer p as an Int16x8 (the
// samples are <= 32767, so the bit pattern is a non-negative int16). p is a
// walking pointer advanced 16 bytes per column by the caller, so the load is a
// register-direct FMOVQ with no per-column address arithmetic and no slice bounds
// check. Package-level so it inlines and the vector stays register-resident (no
// heap escape).
func wienerLoadV(p unsafe.Pointer) archsimd.Int16x8 {
	return archsimd.LoadInt16x8Array((*[8]int16)(p))
}

// wienerStoreV completes the [0,max] clamp and narrow for the two round-shifted
// Int32x4 accumulators, then stores 8 uint16 (lanes 0..3 from lo, 4..7 from hi)
// at the walking dst pointer p. This is the NEON asm's exact tail:
// SaturateToUint16 (SQXTUN) narrows lo into the low 4 lanes and folds the lower
// bound (negatives -> 0), SaturateToUint16Hi (SQXTUN2) narrows hi into the high 4
// lanes in place (no shuffle/zip), then a single Uint16x8.Min (UMIN) applies the
// upper bound to all 8 lanes.
//
// Byte-exact with clampInt32(x,0,max) by clamp order-independence (0 <= max):
// SQXTUN maps x<0 -> 0 and saturates x>65535 -> 65535 (>= max), so
// Min(SQXTUN(x), max) == median(x,0,max) == clampInt32(x,0,max) for every case.
func wienerStoreV(p unsafe.Pointer, lo, hi archsimd.Int32x4, maxV archsimd.Uint16x8) {
	out := lo.SaturateToUint16().SaturateToUint16Hi(hi).Min(maxV)
	out.StoreArray((*[8]uint16)(p))
}

// wienerVerticalSIMD is the Go-native SIMD form of wienerVertical. Byte-identical
// to the scalar reference; non-8-aligned widths fall back to it.
func wienerVerticalSIMD(temp []uint16, tempStride int, dst []uint16, dstStride int, width int, height int, filter WienerFilter, bitDepth int, round1 int, max uint16) {
	if width < 8 || width%8 != 0 {
		wienerVertical(temp, tempStride, dst, dstStride, width, height, filter, bitDepth, round1, max)
		return
	}
	offset := int32(1 << (bitDepth + round1 - 1))
	// seed folds -offset and the rounding bias 1<<(round1-1) so the trailing
	// arithmetic shift reproduces roundPowerOfTwo(sum, round1) exactly.
	seed := -offset + roundBias(round1)

	seedV := archsimd.BroadcastInt32x4(seed)
	maxV := archsimd.BroadcastUint16x8(max)
	// Per-lane arithmetic right shift by round1 as a broadcast count vector
	// (VSSHL, negative => shift right). Broadcasting once here hoists the DUP out
	// of the hot loop; round1 is a runtime value so an immediate shift is not
	// available.
	negShiftV := archsimd.BroadcastInt32x4(int32(-round1))
	f0V := archsimd.BroadcastInt16x8(filter[0])
	f1V := archsimd.BroadcastInt16x8(filter[1])
	f2V := archsimd.BroadcastInt16x8(filter[2])
	// The center tap absorbs the s3<<WienerFilterBits center reapplication.
	f3V := archsimd.BroadcastInt16x8(filter[3] + (1 << WienerFilterBits))
	f4V := archsimd.BroadcastInt16x8(filter[4])
	f5V := archsimd.BroadcastInt16x8(filter[5])
	f6V := archsimd.BroadcastInt16x8(filter[6])

	// Base pointers taken once; each row seeds seven walking tap pointers plus a
	// dst pointer that advance 16 bytes (8 uint16) per column iteration. This
	// keeps the hot loop free of slice bounds checks and of per-load address
	// arithmetic: every load is a register-direct FMOVQ, mirroring the NEON asm's
	// post-indexed loads. All accesses are in range: temp holds
	// (height+2*WienerHalfwin)*tempStride uint16 and dst holds at least
	// height*dstStride, both guaranteed by the caller/dispatch validation.
	const elem = 2 // sizeof(uint16)
	tbase := unsafe.Pointer(&temp[0])
	dbase := unsafe.Pointer(&dst[0])
	// Column-strip walk with a SLIDING 7-row register window: each temp row is
	// loaded once and reused across the 7 output rows that read it (the hand asm's
	// trick), instead of reloading all 7 rows per output row. Go renames the slide
	// cleanly. Seed folds -offset + round bias into the SMLAL accumulator init (no
	// trailing add on the critical path).
	for col := 0; col < width; col += 8 {
		cp := unsafe.Add(tbase, col*elem)
		r0 := wienerLoadV(cp)
		r1 := wienerLoadV(unsafe.Add(cp, 1*tempStride*elem))
		r2 := wienerLoadV(unsafe.Add(cp, 2*tempStride*elem))
		r3 := wienerLoadV(unsafe.Add(cp, 3*tempStride*elem))
		r4 := wienerLoadV(unsafe.Add(cp, 4*tempStride*elem))
		r5 := wienerLoadV(unsafe.Add(cp, 5*tempStride*elem))
		nextp := unsafe.Add(cp, 6*tempStride*elem)
		dp := unsafe.Add(dbase, col*elem)
		for row := 0; row < height; row++ {
			r6 := wienerLoadV(nextp)
			lo := seedV.MulWidenLoAdd(r0, f0V).MulWidenLoAdd(r1, f1V).
				MulWidenLoAdd(r2, f2V).MulWidenLoAdd(r3, f3V).
				MulWidenLoAdd(r4, f4V).MulWidenLoAdd(r5, f5V).
				MulWidenLoAdd(r6, f6V)
			hi := seedV.MulWidenHiAdd(r0, f0V).MulWidenHiAdd(r1, f1V).
				MulWidenHiAdd(r2, f2V).MulWidenHiAdd(r3, f3V).
				MulWidenHiAdd(r4, f4V).MulWidenHiAdd(r5, f5V).
				MulWidenHiAdd(r6, f6V)
			lo = lo.Shift(negShiftV)
			hi = hi.Shift(negShiftV)
			wienerStoreV(dp, lo, hi, maxV)
			r0, r1, r2, r3, r4, r5 = r1, r2, r3, r4, r5, r6
			nextp = unsafe.Add(nextp, tempStride*elem)
			dp = unsafe.Add(dp, dstStride*elem)
		}
	}
}
