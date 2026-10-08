// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

// Go-native SIMD 8-bit compound (bidirectional) inter-prediction convolve. These
// fill a 16-bit CONV_BUF predictor (un-rounded to COMPOUND_ROUND1_BITS precision)
// that the compound blend later averages/masks. The horizontal (X) pass uses the
// same official widening operations as convolveX8GoSIMD, then applies the
// reference's rounding and narrows to int16 CONV_BUF.

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
// AV1's even taps allow computing the half-sum exactly. The reference's
// roundPowerOfTwo3 and CONV_BUF offset are applied in separate int32 lanes
// before narrowing, preserving the exact uint16 output.
func compoundX8GoSIMD(out []uint16, ref frame.Plane, refX int, refY int, width int, height int, kernel [filterTaps]int16, roundOffset int) {
	fo := filterTaps/2 - 1
	_, f0, ok := convolveX8I8MMFilter(kernel)
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

	const round0 = compoundRound0Bits
	// The official Go SIMD API has no USMMLA primitive. Compute the full-tap sum
	// with widening NEON multiplies, then apply roundPowerOfTwo3 and the CONV_BUF
	// offset in int32 lanes.
	roundOffV := archsimd.BroadcastInt32x4(int32(roundOffset))

	rbase := unsafe.Pointer(&ref.Pix[refY*ref.Stride+refX-fo])
	dbase := unsafe.Pointer(&out[0])
	for y := 0; y < height; y++ {
		sp := unsafe.Add(rbase, y*ref.Stride)
		dp := unsafe.Add(dbase, y*width*2) // uint16 = 2 bytes
		for col := 0; col < width; col += 8 {
			raw := archsimd.LoadUint8x16Array((*[16]uint8)(sp))
			lo, hi := simdHorizontalConvAcc(raw, kernel, 0)
			lo = simdRoundShiftInt32(lo, round0).Add(roundOffV)
			hi = simdRoundShiftInt32(hi, round0).Add(roundOffV)
			out16 := simdConcatInt16x8(lo.TruncToInt16(), hi.TruncToInt16()).ConvertToUint16()
			out16.StoreArray((*[8]uint16)(dp))

			// Keep the current vector pointers within their backing slices on
			// the final group. Advancing to a one-past pointer is rejected by
			// checkptr even though the next iteration would not dereference it.
			if col+8 < width {
				sp = unsafe.Add(sp, 8)
				dp = unsafe.Add(dp, 8*2)
			}
		}
	}
}

// compound2D8GoSIMD is the Go-native SIMD form of the 8-bit compound
// both-axes-fractional CONV_BUF convolve (predictInterCompoundRef8ToConvBuf2D).
// The horizontal pass uses official widening multiplies to produce the
// intermediate in the same xBias and round0 domain as single prediction. The
// vertical pass uses int16 widening multiplies with an accumulator seed of
// 1<<offsetBits (libaom's CONV_BUF offset domain, not dav1d's PREP_BIAS), then
// rounds and narrows directly to uint16 CONV_BUF. It does not clamp or subtract
// the offset because the blend consumes the offset domain. The un-rounded sum
// is non-negative and < 2^21, so the saturating narrow is exact. Shapes not covered (width<8,
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

	// ---- Horizontal pass: byte ref -> int16 im. ----
	// Reconstruct the even filter from its I8MM packing and apply the reference
	// bias and rounding stages with official NEON widening operations.
	kernel := simdKernelFromI8MMFilter(filter, f0)
	const xBias = 1 << (8 + filterBits - 1)

	rbase := unsafe.Pointer(&ref.Pix[(refY-foY)*ref.Stride+refX-foX])
	ibase := unsafe.Pointer(&im[0])
	const e = 2 // bytes per int16 im element
	for y := 0; y < imH; y++ {
		sp := unsafe.Add(rbase, y*ref.Stride)
		ip := unsafe.Add(ibase, y*imStride2D*e)
		for col := 0; col < width; col += 8 {
			raw := archsimd.LoadUint8x16Array((*[16]uint8)(sp))
			lo, hi := simdHorizontalConvAcc(raw, kernel, xBias)
			outIM := simdRoundShiftNarrowInt32Pair(lo, hi, round0Bits)
			outIM.StoreArray((*[8]int16)(ip))

			if col+8 < width {
				sp = unsafe.Add(sp, 8)
				ip = unsafe.Add(ip, 8*e)
			}
		}
	}

	// ---- Vertical pass: int16 im -> uint16 CONV_BUF. ----
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

			lo := seedV.Add(c0.MulWidenLo(yk0)).Add(c1.MulWidenLo(yk1)).Add(c2.MulWidenLo(yk2)).Add(c3.MulWidenLo(yk3)).Add(c4.MulWidenLo(yk4)).Add(c5.MulWidenLo(yk5)).Add(c6.MulWidenLo(yk6)).Add(c7.MulWidenLo(yk7))
			hi := seedV.Add(c0.HiToLo().MulWidenLo(yk0.HiToLo())).Add(c1.HiToLo().MulWidenLo(yk1.HiToLo())).Add(c2.HiToLo().MulWidenLo(yk2.HiToLo())).Add(c3.HiToLo().MulWidenLo(yk3.HiToLo())).Add(c4.HiToLo().MulWidenLo(yk4.HiToLo())).Add(c5.HiToLo().MulWidenLo(yk5.HiToLo())).Add(c6.HiToLo().MulWidenLo(yk6.HiToLo())).Add(c7.HiToLo().MulWidenLo(yk7.HiToLo()))
			simdRoundShiftNarrowInt32Pair(lo, hi, compoundRound1Bits).
				StoreArray((*[8]int16)(unsafe.Add(dp, col*2)))

			lo = seedV.Add(c1.MulWidenLo(yk0)).Add(c2.MulWidenLo(yk1)).Add(c3.MulWidenLo(yk2)).Add(c4.MulWidenLo(yk3)).Add(c5.MulWidenLo(yk4)).Add(c6.MulWidenLo(yk5)).Add(c7.MulWidenLo(yk6)).Add(c8.MulWidenLo(yk7))
			hi = seedV.Add(c1.HiToLo().MulWidenLo(yk0.HiToLo())).Add(c2.HiToLo().MulWidenLo(yk1.HiToLo())).Add(c3.HiToLo().MulWidenLo(yk2.HiToLo())).Add(c4.HiToLo().MulWidenLo(yk3.HiToLo())).Add(c5.HiToLo().MulWidenLo(yk4.HiToLo())).Add(c6.HiToLo().MulWidenLo(yk5.HiToLo())).Add(c7.HiToLo().MulWidenLo(yk6.HiToLo())).Add(c8.HiToLo().MulWidenLo(yk7.HiToLo()))
			simdRoundShiftNarrowInt32Pair(lo, hi, compoundRound1Bits).
				StoreArray((*[8]int16)(unsafe.Add(dp, (width+col)*2)))

			lo = seedV.Add(c2.MulWidenLo(yk0)).Add(c3.MulWidenLo(yk1)).Add(c4.MulWidenLo(yk2)).Add(c5.MulWidenLo(yk3)).Add(c6.MulWidenLo(yk4)).Add(c7.MulWidenLo(yk5)).Add(c8.MulWidenLo(yk6)).Add(c9.MulWidenLo(yk7))
			hi = seedV.Add(c2.HiToLo().MulWidenLo(yk0.HiToLo())).Add(c3.HiToLo().MulWidenLo(yk1.HiToLo())).Add(c4.HiToLo().MulWidenLo(yk2.HiToLo())).Add(c5.HiToLo().MulWidenLo(yk3.HiToLo())).Add(c6.HiToLo().MulWidenLo(yk4.HiToLo())).Add(c7.HiToLo().MulWidenLo(yk5.HiToLo())).Add(c8.HiToLo().MulWidenLo(yk6.HiToLo())).Add(c9.HiToLo().MulWidenLo(yk7.HiToLo()))
			simdRoundShiftNarrowInt32Pair(lo, hi, compoundRound1Bits).
				StoreArray((*[8]int16)(unsafe.Add(dp, (2*width+col)*2)))

			lo = seedV.Add(c3.MulWidenLo(yk0)).Add(c4.MulWidenLo(yk1)).Add(c5.MulWidenLo(yk2)).Add(c6.MulWidenLo(yk3)).Add(c7.MulWidenLo(yk4)).Add(c8.MulWidenLo(yk5)).Add(c9.MulWidenLo(yk6)).Add(c10.MulWidenLo(yk7))
			hi = seedV.Add(c3.HiToLo().MulWidenLo(yk0.HiToLo())).Add(c4.HiToLo().MulWidenLo(yk1.HiToLo())).Add(c5.HiToLo().MulWidenLo(yk2.HiToLo())).Add(c6.HiToLo().MulWidenLo(yk3.HiToLo())).Add(c7.HiToLo().MulWidenLo(yk4.HiToLo())).Add(c8.HiToLo().MulWidenLo(yk5.HiToLo())).Add(c9.HiToLo().MulWidenLo(yk6.HiToLo())).Add(c10.HiToLo().MulWidenLo(yk7.HiToLo()))
			simdRoundShiftNarrowInt32Pair(lo, hi, compoundRound1Bits).
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
			lo := seedV.Add(c0.MulWidenLo(yk0)).Add(c1.MulWidenLo(yk1)).Add(c2.MulWidenLo(yk2)).Add(c3.MulWidenLo(yk3)).Add(c4.MulWidenLo(yk4)).Add(c5.MulWidenLo(yk5)).Add(c6.MulWidenLo(yk6)).Add(c7.MulWidenLo(yk7))
			hi := seedV.Add(c0.HiToLo().MulWidenLo(yk0.HiToLo())).Add(c1.HiToLo().MulWidenLo(yk1.HiToLo())).Add(c2.HiToLo().MulWidenLo(yk2.HiToLo())).Add(c3.HiToLo().MulWidenLo(yk3.HiToLo())).Add(c4.HiToLo().MulWidenLo(yk4.HiToLo())).Add(c5.HiToLo().MulWidenLo(yk5.HiToLo())).Add(c6.HiToLo().MulWidenLo(yk6.HiToLo())).Add(c7.HiToLo().MulWidenLo(yk7.HiToLo()))
			simdRoundShiftNarrowInt32Pair(lo, hi, compoundRound1Bits).
				StoreArray((*[8]int16)(unsafe.Add(dp, col*2)))
		}
	}
}
