// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

// Go-native SIMD 8-bit compound (bidirectional) inter-prediction convolve. These
// fill a 16-bit CONV_BUF predictor (un-rounded to COMPOUND_ROUND1_BITS precision)
// that the compound blend later averages/masks. The horizontal (X) pass reuses
// the USMMLA matrix-multiply structure of convolveX8GoSIMD -- the eight taps are
// contiguous in memory, so it beats the I8MM asm the same way the single-
// prediction X pass does -- and only the tail differs: instead of the SQRSHRUN
// round-to-uint8, the int32 half-sum is shim-added, arithmetic-shifted by
// round0-1 and narrowed to int16 CONV_BUF.

package motion

import (
	"simd/archsimd"
	"unsafe"

	"github.com/thesyncim/goav1/internal/av1/dsp/cpu"
	"github.com/thesyncim/goav1/internal/av1/frame"
)

// compoundX8GoSIMD is the Go-native SIMD form of predictInterCompoundRef8ToConvBufX
// for width>=8 (multiple of 8), fully-resident tap windows, even taps and a zero
// end tap (f0==0 -- every regular/smooth phase; the sharp family's nonzero end tap
// falls back). Byte-identical to the pure-Go reference; other shapes route to the
// best asm tier.
//
// The halved-tap USMMLA gives halfsum = fullsum/2 exactly (even taps), and the asm
// folds the CONV_BUF round offset into a shim so that (halfsum + roundShim) >>
// (round0-1) == roundPowerOfTwo3(fullsum) + roundOffset -- the exact uint16 the
// reference stores. roundShim = (roundOffset << (round0-1)) + (1 << (round0-2)).
func compoundX8GoSIMD(out []uint16, ref frame.Plane, refX int, refY int, width int, height int, kernel [filterTaps]int16, roundOffset int) {
	fo := filterTaps/2 - 1
	filter, f0, ok := convolveX8I8MMFilter(kernel)
	if !ok || f0 != 0 || width < 8 || width%8 != 0 ||
		!planeRegionFits(ref, 1, refX-fo, refY, width+filterTaps, height) {
		// Not the SIMD-covered shape: route to the fast asm tier (which has its
		// own width-4 / edge / odd-tap fallbacks), else the pure-Go reference.
		if cpu.Detected.I8MM {
			predictInterCompoundRef8ToConvBufXI8MM(out, ref, refX, refY, width, height, kernel, roundOffset)
		} else {
			predictInterCompoundRef8ToConvBufXNEON(out, ref, refX, refY, width, height, kernel, roundOffset)
		}
		return
	}

	filterV := archsimd.LoadInt8x16Array((*[16]int8)(unsafe.Pointer(&filter[0])))
	permLo := archsimd.LoadUint8x16Array(&convXPermuteLoArr)
	permHi := archsimd.LoadUint8x16Array(&convXPermuteHiArr)
	const round0 = compoundRound0Bits
	// The halved-tap sum fits int16, so the whole CONV_BUF tail runs in the int16
	// domain (as in convolveX8GoSIMD): round(halfsum >> round0-1) then + roundOffset.
	// The +1<<(round0-2) rounding bias is folded into the USMMLA accumulator seed
	// (MatMulUS accumulates, so seeding with the bias instead of 0 is free), leaving
	// a two-op int16 tail (arith shift + add offset); the offset add stays separate
	// so no int16 intermediate overflows (halfsum + roundOffset<<2 would).
	seedV := archsimd.BroadcastInt32x4(1 << (round0 - 2))
	roundOffV := archsimd.BroadcastInt16x8(int16(roundOffset))

	rbase := unsafe.Pointer(&ref.Pix[refY*ref.Stride+refX-fo])
	dbase := unsafe.Pointer(&out[0])
	for y := 0; y < height; y++ {
		sp := unsafe.Add(rbase, y*ref.Stride)
		dp := unsafe.Add(dbase, y*width*2) // uint16 = 2 bytes
		for col := 0; col < width; col += 8 {
			raw := archsimd.LoadUint8x16Array((*[16]uint8)(sp))
			r0 := raw.LookupOrZero(permLo)
			r1 := raw.LookupOrZero(permHi)
			acc0 := seedV.MatMulUS(r0, filterV) // int32 (halfsum + rounding bias), cols 0..3
			acc1 := seedV.MatMulUS(r1, filterV) // cols 4..7
			sumHalf := acc0.TruncToInt16().TruncToInt16Hi(acc1)
			// round(halfsum >> round0-1) + roundOffset, all int16.
			out16 := sumHalf.ShiftAllRightConst(round0 - 1).Add(roundOffV)
			out16.StoreArray((*[8]int16)(dp))

			sp = unsafe.Add(sp, 8)
			dp = unsafe.Add(dp, 8*2)
		}
	}
}
