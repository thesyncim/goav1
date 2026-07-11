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

// compound2D8GoSIMD is the Go-native SIMD form of the 8-bit compound
// both-axes-fractional CONV_BUF convolve (predictInterCompoundRef8ToConvBuf2D).
// The horizontal pass is textually the H pass of convolve2D8GoSIMDIM — the
// intermediate domain (xBias fold, halved-tap USMMLA, round0) is identical for
// single and compound prediction — and the vertical pass is the SMLAL column
// MAC over the sequential int16 intermediate with the compound tail: the
// accumulator seeds with 1<<offsetBits (libaom's CONV_BUF offset domain, not
// dav1d's PREP_BIAS), and one SQRSHRN #COMPOUND_ROUND1_BITS rounds and narrows
// straight to the uint16 CONV_BUF value (no clamp, no offset subtraction —
// the blend consumes the offset domain). The un-rounded sum is non-negative
// and < 2^21, so the saturating narrow is exact. Shapes not covered (width<8,
// non-multiple-of-8, edge-overhanging windows, packed-filter misses,
// offsetBits != 19) route to the I8MM/NEON front doors, which own the W4 tier
// and the emu-edge halo.
func compound2D8GoSIMD(out []uint16, ref frame.Plane, refX int, refY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16, offsetBits int, scratch *CompoundConvolveScratch) {
	foX := filterTaps/2 - 1
	foY := filterTaps/2 - 1
	if offsetBits != 19 || width < 8 || width%8 != 0 ||
		!planeRegionFits(ref, 1, refX-foX, refY-foY, width+filterTaps, height+filterTaps-1) {
		if cpu.Detected.I8MM {
			predictInterCompoundRef8ToConvBuf2DI8MM(out, ref, refX, refY, width, height, xKernel, yKernel, offsetBits, scratch)
		} else {
			predictInterCompoundRef8ToConvBuf2DNEON(out, ref, refX, refY, width, height, xKernel, yKernel, offsetBits, scratch)
		}
		return
	}
	filter, f0, ok := convolveX8I8MMFilter(xKernel)
	if !ok {
		if cpu.Detected.I8MM {
			predictInterCompoundRef8ToConvBuf2DI8MM(out, ref, refX, refY, width, height, xKernel, yKernel, offsetBits, scratch)
		} else {
			predictInterCompoundRef8ToConvBuf2DNEON(out, ref, refX, refY, width, height, xKernel, yKernel, offsetBits, scratch)
		}
		return
	}
	if scratch != nil {
		compound2D8GoSIMDIM(out, ref, refX, refY, width, height, filter, f0, yKernel, &scratch.im8)
		return
	}
	var im compoundIM16
	compound2D8GoSIMDIM(out, ref, refX, refY, width, height, filter, f0, yKernel, &im)
}

func compound2D8GoSIMDIM(out []uint16, ref frame.Plane, refX int, refY int, width int, height int, filter [16]byte, f0 uint8, yKernel [filterTaps]int16, im *compoundIM16) {
	const imStride2D = maxBlockSize
	foX := filterTaps/2 - 1
	foY := filterTaps/2 - 1
	imH := height + filterTaps - 1

	// ---- Horizontal pass: byte ref -> int16 im (USMMLA, halved taps). ----
	// Identical to convolve2D8GoSIMDIM's H pass; see there for the shim math.
	filterV := archsimd.LoadInt8x16Array((*[16]int8)(unsafe.Pointer(&filter[0])))
	permLo := archsimd.LoadUint8x16Array(&convXPermuteLoArr)
	permHi := archsimd.LoadUint8x16Array(&convXPermuteHiArr)
	f0V := archsimd.BroadcastUint8x16(f0)
	const xShim2D = (1 << (8 + filterBits - 2)) + (1 << ((round0Bits - 1) - 1))
	shimHV := archsimd.BroadcastInt16x8(xShim2D)
	zero := archsimd.BroadcastInt32x4(0)

	rbase := unsafe.Pointer(&ref.Pix[(refY-foY)*ref.Stride+refX-foX])
	ibase := unsafe.Pointer(&im[0])
	const e = 2 // bytes per int16 im element
	if f0 == 0 {
		for y := 0; y < imH; y++ {
			sp := unsafe.Add(rbase, y*ref.Stride)
			ip := unsafe.Add(ibase, y*imStride2D*e)
			for col := 0; col < width; col += 8 {
				raw := archsimd.LoadUint8x16Array((*[16]uint8)(sp))
				r0 := raw.LookupOrZero(permLo)
				r1 := raw.LookupOrZero(permHi)
				acc0 := zero.MatMulUS(r0, filterV)
				acc1 := zero.MatMulUS(r1, filterV)
				sumHalf := acc0.TruncToInt16().TruncToInt16Hi(acc1)
				outIM := sumHalf.Add(shimHV).ShiftAllRightConst(2)
				outIM.StoreArray((*[8]int16)(ip))

				sp = unsafe.Add(sp, 8)
				ip = unsafe.Add(ip, 8*e)
			}
		}
	} else {
		for y := 0; y < imH; y++ {
			sp := unsafe.Add(rbase, y*ref.Stride)
			ip := unsafe.Add(ibase, y*imStride2D*e)
			for col := 0; col < width; col += 8 {
				raw := archsimd.LoadUint8x16Array((*[16]uint8)(sp))
				r0 := raw.LookupOrZero(permLo)
				r1 := raw.LookupOrZero(permHi)
				acc0 := zero.MatMulUS(r0, filterV)
				acc1 := zero.MatMulUS(r1, filterV)
				sumHalf := acc0.TruncToInt16().TruncToInt16Hi(acc1)
				tap0 := raw.MulWidenLo(f0V).ConvertToInt16()
				outIM := sumHalf.Sub(tap0).Add(shimHV).ShiftAllRightConst(2)
				outIM.StoreArray((*[8]int16)(ip))

				sp = unsafe.Add(sp, 8)
				ip = unsafe.Add(ip, 8*e)
			}
		}
	}

	// ---- Vertical pass: int16 im -> uint16 CONV_BUF (SMLAL column MAC). ----
	yk0 := archsimd.BroadcastInt16x8(yKernel[0])
	yk1 := archsimd.BroadcastInt16x8(yKernel[1])
	yk2 := archsimd.BroadcastInt16x8(yKernel[2])
	yk3 := archsimd.BroadcastInt16x8(yKernel[3])
	yk4 := archsimd.BroadcastInt16x8(yKernel[4])
	yk5 := archsimd.BroadcastInt16x8(yKernel[5])
	yk6 := archsimd.BroadcastInt16x8(yKernel[6])
	yk7 := archsimd.BroadcastInt16x8(yKernel[7])

	const offsetBits2D = 8 + 2*filterBits - round0Bits // 19, guarded by the caller
	seedV := archsimd.BroadcastInt32x4(1 << offsetBits2D)

	dbase := unsafe.Pointer(&out[0])
	// Four output rows per block: eleven im rows loaded once feed all four
	// 8-tap column MACs (vs eight reloads per single row), the same
	// row-sharing the asm's sliding window gets, without loop-carried vector
	// state. Compound heights are multiples of 4; a defensive single-row tail
	// covers anything else.
	y := 0
	for ; y+4 <= height; y += 4 {
		ip := unsafe.Add(ibase, y*imStride2D*e)
		dp := unsafe.Add(dbase, y*width*2)
		for col := 0; col < width; col += 8 {
			cp := unsafe.Add(ip, col*e)
			c0 := convIMLoad8(cp)
			cp = unsafe.Add(cp, imStride2D*e)
			c1 := convIMLoad8(cp)
			cp = unsafe.Add(cp, imStride2D*e)
			c2 := convIMLoad8(cp)
			cp = unsafe.Add(cp, imStride2D*e)
			c3 := convIMLoad8(cp)
			cp = unsafe.Add(cp, imStride2D*e)
			c4 := convIMLoad8(cp)
			cp = unsafe.Add(cp, imStride2D*e)
			c5 := convIMLoad8(cp)
			cp = unsafe.Add(cp, imStride2D*e)
			c6 := convIMLoad8(cp)
			cp = unsafe.Add(cp, imStride2D*e)
			c7 := convIMLoad8(cp)
			cp = unsafe.Add(cp, imStride2D*e)
			c8 := convIMLoad8(cp)
			cp = unsafe.Add(cp, imStride2D*e)
			c9 := convIMLoad8(cp)
			cp = unsafe.Add(cp, imStride2D*e)
			c10 := convIMLoad8(cp)

			lo := seedV.MulWidenLoAdd(c0, yk0).MulWidenLoAdd(c1, yk1).
				MulWidenLoAdd(c2, yk2).MulWidenLoAdd(c3, yk3).
				MulWidenLoAdd(c4, yk4).MulWidenLoAdd(c5, yk5).
				MulWidenLoAdd(c6, yk6).MulWidenLoAdd(c7, yk7)
			hi := seedV.MulWidenHiAdd(c0, yk0).MulWidenHiAdd(c1, yk1).
				MulWidenHiAdd(c2, yk2).MulWidenHiAdd(c3, yk3).
				MulWidenHiAdd(c4, yk4).MulWidenHiAdd(c5, yk5).
				MulWidenHiAdd(c6, yk6).MulWidenHiAdd(c7, yk7)
			lo.ShiftRightRoundNarrow(compoundRound1Bits).ShiftRightRoundNarrowHi(hi, compoundRound1Bits).
				StoreArray((*[8]int16)(unsafe.Add(dp, col*2)))

			lo = seedV.MulWidenLoAdd(c1, yk0).MulWidenLoAdd(c2, yk1).
				MulWidenLoAdd(c3, yk2).MulWidenLoAdd(c4, yk3).
				MulWidenLoAdd(c5, yk4).MulWidenLoAdd(c6, yk5).
				MulWidenLoAdd(c7, yk6).MulWidenLoAdd(c8, yk7)
			hi = seedV.MulWidenHiAdd(c1, yk0).MulWidenHiAdd(c2, yk1).
				MulWidenHiAdd(c3, yk2).MulWidenHiAdd(c4, yk3).
				MulWidenHiAdd(c5, yk4).MulWidenHiAdd(c6, yk5).
				MulWidenHiAdd(c7, yk6).MulWidenHiAdd(c8, yk7)
			lo.ShiftRightRoundNarrow(compoundRound1Bits).ShiftRightRoundNarrowHi(hi, compoundRound1Bits).
				StoreArray((*[8]int16)(unsafe.Add(dp, (width+col)*2)))

			lo = seedV.MulWidenLoAdd(c2, yk0).MulWidenLoAdd(c3, yk1).
				MulWidenLoAdd(c4, yk2).MulWidenLoAdd(c5, yk3).
				MulWidenLoAdd(c6, yk4).MulWidenLoAdd(c7, yk5).
				MulWidenLoAdd(c8, yk6).MulWidenLoAdd(c9, yk7)
			hi = seedV.MulWidenHiAdd(c2, yk0).MulWidenHiAdd(c3, yk1).
				MulWidenHiAdd(c4, yk2).MulWidenHiAdd(c5, yk3).
				MulWidenHiAdd(c6, yk4).MulWidenHiAdd(c7, yk5).
				MulWidenHiAdd(c8, yk6).MulWidenHiAdd(c9, yk7)
			lo.ShiftRightRoundNarrow(compoundRound1Bits).ShiftRightRoundNarrowHi(hi, compoundRound1Bits).
				StoreArray((*[8]int16)(unsafe.Add(dp, (2*width+col)*2)))

			lo = seedV.MulWidenLoAdd(c3, yk0).MulWidenLoAdd(c4, yk1).
				MulWidenLoAdd(c5, yk2).MulWidenLoAdd(c6, yk3).
				MulWidenLoAdd(c7, yk4).MulWidenLoAdd(c8, yk5).
				MulWidenLoAdd(c9, yk6).MulWidenLoAdd(c10, yk7)
			hi = seedV.MulWidenHiAdd(c3, yk0).MulWidenHiAdd(c4, yk1).
				MulWidenHiAdd(c5, yk2).MulWidenHiAdd(c6, yk3).
				MulWidenHiAdd(c7, yk4).MulWidenHiAdd(c8, yk5).
				MulWidenHiAdd(c9, yk6).MulWidenHiAdd(c10, yk7)
			lo.ShiftRightRoundNarrow(compoundRound1Bits).ShiftRightRoundNarrowHi(hi, compoundRound1Bits).
				StoreArray((*[8]int16)(unsafe.Add(dp, (3*width+col)*2)))
		}
	}
	for ; y < height; y++ {
		ip := unsafe.Add(ibase, y*imStride2D*e)
		dp := unsafe.Add(dbase, y*width*2)
		for col := 0; col < width; col += 8 {
			cp := unsafe.Add(ip, col*e)
			c0 := convIMLoad8(cp)
			c1 := convIMLoad8(unsafe.Add(cp, imStride2D*e))
			c2 := convIMLoad8(unsafe.Add(cp, 2*imStride2D*e))
			c3 := convIMLoad8(unsafe.Add(cp, 3*imStride2D*e))
			c4 := convIMLoad8(unsafe.Add(cp, 4*imStride2D*e))
			c5 := convIMLoad8(unsafe.Add(cp, 5*imStride2D*e))
			c6 := convIMLoad8(unsafe.Add(cp, 6*imStride2D*e))
			c7 := convIMLoad8(unsafe.Add(cp, 7*imStride2D*e))
			lo := seedV.MulWidenLoAdd(c0, yk0).MulWidenLoAdd(c1, yk1).
				MulWidenLoAdd(c2, yk2).MulWidenLoAdd(c3, yk3).
				MulWidenLoAdd(c4, yk4).MulWidenLoAdd(c5, yk5).
				MulWidenLoAdd(c6, yk6).MulWidenLoAdd(c7, yk7)
			hi := seedV.MulWidenHiAdd(c0, yk0).MulWidenHiAdd(c1, yk1).
				MulWidenHiAdd(c2, yk2).MulWidenHiAdd(c3, yk3).
				MulWidenHiAdd(c4, yk4).MulWidenHiAdd(c5, yk5).
				MulWidenHiAdd(c6, yk6).MulWidenHiAdd(c7, yk7)
			lo.ShiftRightRoundNarrow(compoundRound1Bits).ShiftRightRoundNarrowHi(hi, compoundRound1Bits).
				StoreArray((*[8]int16)(unsafe.Add(dp, col*2)))
		}
	}
}
