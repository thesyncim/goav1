// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package restoration

import (
	"simd/archsimd"
	"unsafe"
)

// wienerHorizontalU8SIMD is the Go-native SIMD form of the 8-bit Wiener
// horizontal pass: a 7-tap sliding-window FIR over uint8 source pixels
// producing the libaom-offset uint16 intermediate. Byte-identical to
// wienerHorizontalU8 (TestWienerHorizontalU8SIMDMatchesReference).
//
// Algebraic structure (the op-count cut over the hand asm): AV1 Wiener filters
// are symmetric (f0==f6, f1==f5, f2==f4, enforced by validWienerInfo before
// dispatch and re-checked here), so with the libaom center reapplication
// s3<<WienerFilterBits folded into the center tap (f3' = f3 + 128) the per-pixel
// sum factors into four terms:
//
//	sum = seed + f0*(s0+s6) + f1*(s1+s5) + f2*(s2+s4) + f3'*s3
//
// where seed = offset + 1<<(round0-1) also folds the rounding bias, so the
// trailing shift is a plain arithmetic >>round0. The u8 tap pairs sum to at
// most 510, which fits a non-negative int16 lane, so each term is ONE
// SMLAL/SMLAL2 pair: 8 widening MACs per 8 outputs instead of the asm's 14,
// and a 4-deep accumulator dependency chain instead of 7-deep. Exactness: all
// arithmetic is int32 with |sum| far below 2^31, and int32 add/mul commute, so
// the regrouped sum equals the reference's term-by-term sum bit-for-bit.
//
// Window construction matches the asm exactly: one 16-byte load per 8-output
// group (advancing 8), UXTL/UXTL2 widen, then byte-EXT of the u16 lane pair
// forms the six shifted windows; the tap pairs are three u16 adds. The load
// contract is identical to wienerHorizontalU8NEONAsm: the last load of each
// row reaches up to 2 samples past the 3-pixel border (the extra lanes never
// contribute to any output).
//
// The store tail is one SQSHRUN/SQSHRUN2 pair (ShiftRightSatUnsignedNarrow:
// arithmetic >>round0 + int32->uint16 unsigned-saturating narrow, folding the
// lower clamp) plus one UMIN for the upper clamp — the asm spends four ops
// (2 SSHL + 2 SQXTUN) on the same dataflow. round0 is always WienerRound0Bits
// (=3) for 8-bit input (wienerRounds), kept as a literal so SQSHRUN lowers to
// its immediate form; any other value falls back (defensive, unreachable from
// the public entry).
//
// The main loop runs two independent 8-output groups per iteration (16
// outputs) sharing the middle UXTL2, so the two 4-deep MAC chains pipeline;
// widths that are not a multiple of 8 (or below one vector) fall back to the
// scalar reference exactly like the asm wrapper.
func wienerHorizontalU8SIMD(src []uint8, srcStride int, srcOrigin int, width int, height int, filter WienerFilter, round0 int, temp []uint16) {
	if width < 8 || width%8 != 0 || round0 != WienerRound0Bits ||
		filter[0] != filter[6] || filter[1] != filter[5] || filter[2] != filter[4] {
		wienerHorizontalU8(src, srcStride, srcOrigin, width, height, filter, round0, temp)
		return
	}
	const bitDepth = 8
	limit := int32(1) << (bitDepth + 1 + WienerFilterBits - WienerRound0Bits)
	offset := int32(1) << (bitDepth + WienerFilterBits - 1)
	// seed folds the libaom offset and the rounding bias 1<<(round0-1) so the
	// SQSHRUN immediate shift reproduces roundPowerOfTwo(sum, round0) exactly.
	seedV := archsimd.BroadcastInt32x4(offset + roundBias(WienerRound0Bits))
	maxV := archsimd.BroadcastUint16x8(uint16(limit - 1))
	g0 := archsimd.BroadcastInt16x8(filter[0])
	g1 := archsimd.BroadcastInt16x8(filter[1])
	g2 := archsimd.BroadcastInt16x8(filter[2])
	// The center tap absorbs the s3<<WienerFilterBits center reapplication.
	g3 := archsimd.BroadcastInt16x8(filter[3] + (1 << WienerFilterBits))

	rows := height + 2*WienerHalfwin
	w16 := width &^ 15
	// Row base pointers walk the source tap window (starting at row -3, col -3)
	// and the temp output; the inner loop walks both with post-added cursors so
	// every load/store is a register-direct FMOVQ with no bounds checks.
	srow := unsafe.Pointer(&src[srcOrigin-WienerHalfwin*srcStride-WienerHalfwin])
	drow := unsafe.Pointer(&temp[0])
	for row := 0; row < rows; row++ {
		s := srow
		d := drow
		for col := 0; col < w16; col += 16 {
			v0 := archsimd.LoadUint8x16Array((*[16]uint8)(s))                // s0..s15
			v1 := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(s, 8))) // s8..s23
			l0 := v0.ExtendLo8ToUint16()                                     // s0..s7  (u16)
			m0 := v0.ExtendHi8ToUint16()                                     // s8..s15 (u16)
			h1 := v1.ExtendHi8ToUint16()                                     // s16..s23 (u16)
			lb := l0.ReshapeToUint8s()
			mb := m0.ReshapeToUint8s()
			hb := h1.ReshapeToUint8s()

			// Group A: outputs col..col+7, windows over s0..s13.
			w1a := mb.ConcatShiftBytesRight(lb, 2).ReshapeToUint16s()  // s1..s8
			w2a := mb.ConcatShiftBytesRight(lb, 4).ReshapeToUint16s()  // s2..s9
			w3a := mb.ConcatShiftBytesRight(lb, 6).ReshapeToUint16s()  // s3..s10
			w4a := mb.ConcatShiftBytesRight(lb, 8).ReshapeToUint16s()  // s4..s11
			w5a := mb.ConcatShiftBytesRight(lb, 10).ReshapeToUint16s() // s5..s12
			w6a := mb.ConcatShiftBytesRight(lb, 12).ReshapeToUint16s() // s6..s13
			aA := l0.Add(w6a).AsInt16x8()                              // s_j + s_{j+6} <= 510
			bA := w1a.Add(w5a).AsInt16x8()
			cA := w2a.Add(w4a).AsInt16x8()
			dA := w3a.AsInt16x8()
			loA := seedV.MulWidenLoAdd(aA, g0).MulWidenLoAdd(bA, g1).
				MulWidenLoAdd(cA, g2).MulWidenLoAdd(dA, g3)
			hiA := seedV.MulWidenHiAdd(aA, g0).MulWidenHiAdd(bA, g1).
				MulWidenHiAdd(cA, g2).MulWidenHiAdd(dA, g3)
			outA := loA.ShiftRightSatUnsignedNarrow(WienerRound0Bits).
				ShiftRightSatUnsignedNarrowHi(hiA, WienerRound0Bits).Min(maxV)
			outA.StoreArray((*[8]uint16)(d))

			// Group B: outputs col+8..col+15, windows over s8..s21.
			w1b := hb.ConcatShiftBytesRight(mb, 2).ReshapeToUint16s()  // s9..s16
			w2b := hb.ConcatShiftBytesRight(mb, 4).ReshapeToUint16s()  // s10..s17
			w3b := hb.ConcatShiftBytesRight(mb, 6).ReshapeToUint16s()  // s11..s18
			w4b := hb.ConcatShiftBytesRight(mb, 8).ReshapeToUint16s()  // s12..s19
			w5b := hb.ConcatShiftBytesRight(mb, 10).ReshapeToUint16s() // s13..s20
			w6b := hb.ConcatShiftBytesRight(mb, 12).ReshapeToUint16s() // s14..s21
			aB := m0.Add(w6b).AsInt16x8()
			bB := w1b.Add(w5b).AsInt16x8()
			cB := w2b.Add(w4b).AsInt16x8()
			dB := w3b.AsInt16x8()
			loB := seedV.MulWidenLoAdd(aB, g0).MulWidenLoAdd(bB, g1).
				MulWidenLoAdd(cB, g2).MulWidenLoAdd(dB, g3)
			hiB := seedV.MulWidenHiAdd(aB, g0).MulWidenHiAdd(bB, g1).
				MulWidenHiAdd(cB, g2).MulWidenHiAdd(dB, g3)
			outB := loB.ShiftRightSatUnsignedNarrow(WienerRound0Bits).
				ShiftRightSatUnsignedNarrowHi(hiB, WienerRound0Bits).Min(maxV)
			outB.StoreArray((*[8]uint16)(unsafe.Add(d, 16)))

			s = unsafe.Add(s, 16)
			d = unsafe.Add(d, 32)
		}
		if w16 != width { // trailing width%16 == 8 group
			v0 := archsimd.LoadUint8x16Array((*[16]uint8)(s)) // s0..s15 (needs s0..s13)
			l0 := v0.ExtendLo8ToUint16()
			m0 := v0.ExtendHi8ToUint16()
			lb := l0.ReshapeToUint8s()
			mb := m0.ReshapeToUint8s()
			w1 := mb.ConcatShiftBytesRight(lb, 2).ReshapeToUint16s()
			w2 := mb.ConcatShiftBytesRight(lb, 4).ReshapeToUint16s()
			w3 := mb.ConcatShiftBytesRight(lb, 6).ReshapeToUint16s()
			w4 := mb.ConcatShiftBytesRight(lb, 8).ReshapeToUint16s()
			w5 := mb.ConcatShiftBytesRight(lb, 10).ReshapeToUint16s()
			w6 := mb.ConcatShiftBytesRight(lb, 12).ReshapeToUint16s()
			a := l0.Add(w6).AsInt16x8()
			b := w1.Add(w5).AsInt16x8()
			c := w2.Add(w4).AsInt16x8()
			dd := w3.AsInt16x8()
			lo := seedV.MulWidenLoAdd(a, g0).MulWidenLoAdd(b, g1).
				MulWidenLoAdd(c, g2).MulWidenLoAdd(dd, g3)
			hi := seedV.MulWidenHiAdd(a, g0).MulWidenHiAdd(b, g1).
				MulWidenHiAdd(c, g2).MulWidenHiAdd(dd, g3)
			out := lo.ShiftRightSatUnsignedNarrow(WienerRound0Bits).
				ShiftRightSatUnsignedNarrowHi(hi, WienerRound0Bits).Min(maxV)
			out.StoreArray((*[8]uint16)(d))
		}
		srow = unsafe.Add(srow, srcStride)
		drow = unsafe.Add(drow, width*2)
	}
}
