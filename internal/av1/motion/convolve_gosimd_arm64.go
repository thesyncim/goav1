// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

// Go-native SIMD 8-bit motion-compensation convolve kernels. These target the
// #2 decode hotspot (inter prediction) and are byte-identical to the pure-Go
// reference in vector.go, which every variant must match sample for sample.
//
// The vertical (Y) pass is a pure strided-column widening MAC, the same shape
// as the Wiener vertical restoration pass that already beats its hand-written
// NEON asm: eight output columns are processed per iteration, the eight taps
// are fused widening multiply-accumulated (SMLAL / SMLAL2 via
// Int32x4.MulWidenLoAdd / MulWidenHiAdd) into two int32 lane groups, then the
// pair of accumulators is round-shifted-and-narrowed to int16 (SQRSHRN /
// SQRSHRN2 via ShiftRightRoundNarrow / ShiftRightRoundNarrowHi) and clamped to
// [0,255] (SQXTUN via SaturateToUint8). No transpose is needed: the vertical
// convolve reads eight consecutive bytes per row across eight rows, so every
// load is a register-direct byte load and the accumulate chain is straight-line
// register-resident work.

package motion

import (
	"simd/archsimd"
	"unsafe"

	"github.com/thesyncim/goav1/internal/av1/dsp/cpu"
	"github.com/thesyncim/goav1/internal/av1/frame"
)

// convYLoad8 widens 8 consecutive reference bytes at p to an Int16x8. Pixels are
// 0..255, so the zero-extended value is a non-negative int16 whose bit pattern
// reproduces int(byte) exactly; int16*int16->int32 (SMLAL) therefore matches the
// reference's k[i]*int(s[i]) product. p is a walking pointer advanced 8 bytes per
// column iteration, so this lowers to a register-direct 8-byte load with no slice
// bounds check.
func convYLoad8(p unsafe.Pointer) archsimd.Int16x8 {
	return archsimd.LoadUint8x16Array((*[16]uint8)(p)).ExtendLo8ToUint16().ConvertToInt16()
}

// convStore8 narrows the low 8 int16 lanes of v to bytes (SQXTUN) and writes
// them contiguously at raw pointer p as a single 8-byte store. Only the low 8
// bytes are valid; there is no 8-byte narrow store in archsimd, so the byte
// vector's low 64 bits are extracted and stored directly (no slice bounds check,
// no panic path). Mirrors the loopfilter lf8StoreP idiom.
func convStore8(p unsafe.Pointer, v archsimd.Int16x8) {
	*(*float64)(p) = v.SaturateToUint8().ReshapeToFloat64x2().GetElem(0)
}

// convStore8U8 writes the low 8 bytes of an already-narrowed Uint8x16 at raw
// pointer p as a single 8-byte store.
func convStore8U8(p unsafe.Pointer, v archsimd.Uint8x16) {
	*(*float64)(p) = v.ReshapeToFloat64x2().GetElem(0)
}

// convXPermuteLo / convXPermuteHi are the two USMMLA sample-permute index
// vectors (SVT svt_kMatMul8PermuteTbl), loaded once from convolveX8I8MMPermute.
// permLo builds the two 8-byte USMMLA rows for output columns 0..3 (row0 =
// samples[1..8] -> out0, row1 = samples[3..10] -> out2; the staggered filter
// column then yields out1/out3), permHi does the same for output columns 4..7.
var convXPermuteLoArr = *(*[16]uint8)(convolveX8I8MMPermute[0:16])
var convXPermuteHiArr = *(*[16]uint8)(convolveX8I8MMPermute[16:32])

// convolveX8GoSIMD is the Go-native SIMD form of convolveX8PureGo using the
// I8MM matrix-multiply-accumulate (USMMLA via Int32x4.MatMulUS), the same
// algorithm as convolveX8I8MMAsm. For each 8-column group it loads 16 reference
// bytes, permutes them into the two USMMLA operand layouts (TBL via
// LookupOrZero), runs two USMMLA (each producing four staggered 8-tap partial
// sums = outputs 0..3 and 4..7 with the col-shifted filter), narrows to int16
// (XTN/XTN2), folds in tap-0 via a widening multiply-subtract (UMULL+SUB,
// reproducing the asm's UMLSL), adds the round shim, arithmetic-shifts by 6 and
// clamps to [0,255] (SQXTUN). Byte-identical to the reference; kernels whose
// taps do not fit the halved-even / non-positive-tap0 packing, and widths that
// are not a positive multiple of 8, fall back to the scalar reference.
func convolveX8GoSIMD(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, kernel [filterTaps]int16) {
	filter, f0, ok := convolveX8I8MMFilter(kernel)
	if !ok || !(width >= 8 && width%8 == 0) {
		convolveX8PureGo(dst, ref, dstX, dstY, refX, refY, width, height, kernel)
		return
	}
	fo := filterTaps/2 - 1

	filterV := archsimd.LoadInt8x16Array((*[16]int8)(unsafe.Pointer(&filter[0])))
	permLo := archsimd.LoadUint8x16Array(&convXPermuteLoArr)
	permHi := archsimd.LoadUint8x16Array(&convXPermuteHiArr)
	f0V := archsimd.BroadcastUint8x16(f0)
	// Round shim folded into the int16 domain: the asm adds 2 then sqrshrun #6
	// (which adds 1<<5). Fold both: (sum/2 + 34) >> 6.
	const xShim = 2 + (1 << 5)
	shimV := archsimd.BroadcastInt16x8(xShim)
	zero := archsimd.BroadcastInt32x4(0)

	rbase := unsafe.Pointer(&ref.Pix[refY*ref.Stride+refX-fo])
	dbase := unsafe.Pointer(&dst.Pix[dstY*dst.Stride+dstX])
	// f0 == 0 (every regular/smooth filter phase; only the multi-tap-sharp family
	// has a nonzero end tap) drops the tap-0 UMULL+SUB from the hot loop.
	if f0 == 0 {
		for y := 0; y < height; y++ {
			sp := unsafe.Add(rbase, y*ref.Stride)
			dp := unsafe.Add(dbase, y*dst.Stride)
			for col := 0; col < width; col += 8 {
				raw := archsimd.LoadUint8x16Array((*[16]uint8)(sp))
				r0 := raw.LookupOrZero(permLo)
				r1 := raw.LookupOrZero(permHi)
				acc0 := zero.MatMulUS(r0, filterV) // outputs 0..3
				acc1 := zero.MatMulUS(r1, filterV) // outputs 4..7
				sumHalf := acc0.TruncToInt16().TruncToInt16Hi(acc1)
				out := sumHalf.Add(shimV).ShiftAllRightConst(6).SaturateToUint8()
				convStore8U8(dp, out)

				sp = unsafe.Add(sp, 8)
				dp = unsafe.Add(dp, 8)
			}
		}
		return
	}
	for y := 0; y < height; y++ {
		sp := unsafe.Add(rbase, y*ref.Stride)
		dp := unsafe.Add(dbase, y*dst.Stride)
		for col := 0; col < width; col += 8 {
			raw := archsimd.LoadUint8x16Array((*[16]uint8)(sp))
			r0 := raw.LookupOrZero(permLo)
			r1 := raw.LookupOrZero(permHi)
			acc0 := zero.MatMulUS(r0, filterV) // outputs 0..3
			acc1 := zero.MatMulUS(r1, filterV) // outputs 4..7
			// Narrow the two int32x4 accumulators into one int16x8 (XTN/XTN2).
			sumHalf := acc0.TruncToInt16().TruncToInt16Hi(acc1)
			// tap-0 fold: += (k0>>1)*s0 == -(f0*s0). UMULL of the low 8 raw bytes
			// with the broadcast f0, reinterpreted as int16, subtracted.
			tap0 := raw.MulWidenLo(f0V).ConvertToInt16()
			out := sumHalf.Sub(tap0).Add(shimV).ShiftAllRightConst(6).SaturateToUint8()
			convStore8U8(dp, out)

			sp = unsafe.Add(sp, 8)
			dp = unsafe.Add(dp, 8)
		}
	}
}

// convIMLoad8 loads 8 contiguous int16 intermediates at raw pointer p as an
// Int16x8 (register-direct FMOVQ, no bounds check).
func convIMLoad8(p unsafe.Pointer) archsimd.Int16x8 {
	return archsimd.LoadInt16x8Array((*[8]int16)(p))
}

// convolve2D8GoSIMD is the Go-native SIMD form of convolve2D8PureGo (both axes
// fractional) for width>=8, multiple of 8. The horizontal pass reuses the USMMLA
// matrix-multiply structure of convolveX8GoSIMD to produce the int16 intermediate
// (rounded by round0Bits with the folded xBias), and the vertical pass is a pure
// int16 widening SMLAL column MAC over that intermediate -- the Wiener-shaped
// pattern -- with the staged round1 shift, roundOffset subtraction and [0,255]
// clip. It is byte-identical to the reference. The intermediate is caller-owned
// scratch when provided (the hot decode path) or a stack array otherwise.
func convolve2D8GoSIMD(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16) {
	convolve2D8GoSIMDScratch(dst, ref, dstX, dstY, refX, refY, width, height, xKernel, yKernel, nil)
}

func convolve2D8GoSIMDScratch(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16, scratch *ConvolveScratch) {
	// Width check first (before the filter pack) so the narrow shapes skip the
	// filter-pack work entirely. Route width-4 and non-multiple-of-8 to the best
	// asm tier: the I8MM-with-scratch path (fast width-4 4-tap tier + its own
	// NEON/pure-Go fallbacks) when I8MM is present, else NEON.
	if !(width >= 8 && width%8 == 0) {
		if cpu.Detected.I8MM {
			convolve2D8I8MMWithScratch(dst, ref, dstX, dstY, refX, refY, width, height, xKernel, yKernel, scratch)
		} else {
			convolve2D8NEONWithScratch(dst, ref, dstX, dstY, refX, refY, width, height, xKernel, yKernel, scratch)
		}
		return
	}
	filter, f0, ok := convolveX8I8MMFilter(xKernel)
	if !ok {
		if cpu.Detected.I8MM {
			convolve2D8I8MMWithScratch(dst, ref, dstX, dstY, refX, refY, width, height, xKernel, yKernel, scratch)
		} else {
			convolve2D8NEONWithScratch(dst, ref, dstX, dstY, refX, refY, width, height, xKernel, yKernel, scratch)
		}
		return
	}
	const imStride = maxBlockSize
	if scratch != nil {
		convolve2D8GoSIMDIM(dst, ref, dstX, dstY, refX, refY, width, height, filter, f0, yKernel, &scratch.im, imStride)
		return
	}
	var im [(maxBlockSize + filterTaps - 1) * maxBlockSize]int16
	convolve2D8GoSIMDIM(dst, ref, dstX, dstY, refX, refY, width, height, filter, f0, yKernel, &im, imStride)
}

func convolve2D8GoSIMDIM(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, filter [16]byte, f0 uint8, yKernel [filterTaps]int16, im *[(maxBlockSize + filterTaps - 1) * maxBlockSize]int16, imStride int) {
	foX := filterTaps/2 - 1
	foY := filterTaps/2 - 1
	imH := height + filterTaps - 1

	// ---- Horizontal pass: byte ref -> int16 im (USMMLA, halved taps). ----
	filterV := archsimd.LoadInt8x16Array((*[16]int8)(unsafe.Pointer(&filter[0])))
	permLo := archsimd.LoadUint8x16Array(&convXPermuteLoArr)
	permHi := archsimd.LoadUint8x16Array(&convXPermuteHiArr)
	f0V := archsimd.BroadcastUint8x16(f0)
	// im = (sum/2 + xShim2D) >> 2, xShim2D = (1<<(8+FILTER_BITS-2)) + (1<<((ROUND0_BITS-1)-1)) = 8194.
	const xShim2D = (1 << (8 + filterBits - 2)) + (1 << ((round0Bits - 1) - 1))
	shimHV := archsimd.BroadcastInt16x8(xShim2D)
	zero := archsimd.BroadcastInt32x4(0)

	rbase := unsafe.Pointer(&ref.Pix[(refY-foY)*ref.Stride+refX-foX])
	ibase := unsafe.Pointer(&im[0])
	const imElem = 2
	// f0 == 0 (regular/smooth phases) drops the tap-0 UMULL+SUB from the hot loop.
	if f0 == 0 {
		for y := 0; y < imH; y++ {
			sp := unsafe.Add(rbase, y*ref.Stride)
			ip := unsafe.Add(ibase, y*imStride*imElem)
			for col := 0; col < width; col += 8 {
				raw := archsimd.LoadUint8x16Array((*[16]uint8)(sp))
				r0 := raw.LookupOrZero(permLo)
				r1 := raw.LookupOrZero(permHi)
				acc0 := zero.MatMulUS(r0, filterV) // outputs 0..3
				acc1 := zero.MatMulUS(r1, filterV) // outputs 4..7
				sumHalf := acc0.TruncToInt16().TruncToInt16Hi(acc1)
				outIM := sumHalf.Add(shimHV).ShiftAllRightConst(2)
				outIM.StoreArray((*[8]int16)(ip))

				sp = unsafe.Add(sp, 8)
				ip = unsafe.Add(ip, 8*imElem)
			}
		}
	} else {
		for y := 0; y < imH; y++ {
			sp := unsafe.Add(rbase, y*ref.Stride)
			ip := unsafe.Add(ibase, y*imStride*imElem)
			for col := 0; col < width; col += 8 {
				raw := archsimd.LoadUint8x16Array((*[16]uint8)(sp))
				r0 := raw.LookupOrZero(permLo)
				r1 := raw.LookupOrZero(permHi)
				acc0 := zero.MatMulUS(r0, filterV) // outputs 0..3
				acc1 := zero.MatMulUS(r1, filterV) // outputs 4..7
				sumHalf := acc0.TruncToInt16().TruncToInt16Hi(acc1)
				// tap-0 fold (UMULL+SUB == asm's UMLSL), then round shim; ushr #2 is
				// logical but the value is non-negative after the xBias fold, so an
				// arithmetic shift is bit-identical.
				tap0 := raw.MulWidenLo(f0V).ConvertToInt16()
				outIM := sumHalf.Sub(tap0).Add(shimHV).ShiftAllRightConst(2)
				outIM.StoreArray((*[8]int16)(ip))

				sp = unsafe.Add(sp, 8)
				ip = unsafe.Add(ip, 8*imElem)
			}
		}
	}

	// ---- Vertical pass: int16 im -> uint8 dst (SMLAL column MAC). ----
	yk0 := archsimd.BroadcastInt16x8(yKernel[0])
	yk1 := archsimd.BroadcastInt16x8(yKernel[1])
	yk2 := archsimd.BroadcastInt16x8(yKernel[2])
	yk3 := archsimd.BroadcastInt16x8(yKernel[3])
	yk4 := archsimd.BroadcastInt16x8(yKernel[4])
	yk5 := archsimd.BroadcastInt16x8(yKernel[5])
	yk6 := archsimd.BroadcastInt16x8(yKernel[6])
	yk7 := archsimd.BroadcastInt16x8(yKernel[7])

	const offsetBits = 8 + 2*filterBits - round0Bits // 19
	const yBias = 1 << offsetBits
	const roundOffset = (1 << (offsetBits - round1Bits)) + (1 << (offsetBits - round1Bits - 1))
	// Fold -roundOffset into the seed: since roundOffset*(1<<round1Bits) is a
	// multiple of 1<<round1Bits, srshr(sum - roundOffset<<round1Bits, round1Bits)
	// == srshr(sum, round1Bits) - roundOffset. This keeps the subtraction in the
	// wide int32 domain (matching the asm) so the final SQRSHRN+SQXTUN tail is a
	// plain rounding-narrow + uint8 clamp with no int16 underflow hazard.
	const ySeed = yBias - (roundOffset << round1Bits)
	seedV := archsimd.BroadcastInt32x4(ySeed)

	dbase := unsafe.Pointer(&dst.Pix[dstY*dst.Stride+dstX])
	for y := 0; y < height; y++ {
		c0p := unsafe.Add(ibase, (y+0)*imStride*imElem)
		c1p := unsafe.Add(ibase, (y+1)*imStride*imElem)
		c2p := unsafe.Add(ibase, (y+2)*imStride*imElem)
		c3p := unsafe.Add(ibase, (y+3)*imStride*imElem)
		c4p := unsafe.Add(ibase, (y+4)*imStride*imElem)
		c5p := unsafe.Add(ibase, (y+5)*imStride*imElem)
		c6p := unsafe.Add(ibase, (y+6)*imStride*imElem)
		c7p := unsafe.Add(ibase, (y+7)*imStride*imElem)
		dp := unsafe.Add(dbase, y*dst.Stride)
		for col := 0; col < width; col += 8 {
			c0 := convIMLoad8(c0p)
			c1 := convIMLoad8(c1p)
			c2 := convIMLoad8(c2p)
			c3 := convIMLoad8(c3p)
			c4 := convIMLoad8(c4p)
			c5 := convIMLoad8(c5p)
			c6 := convIMLoad8(c6p)
			c7 := convIMLoad8(c7p)

			lo := seedV.MulWidenLoAdd(c0, yk0).MulWidenLoAdd(c1, yk1).
				MulWidenLoAdd(c2, yk2).MulWidenLoAdd(c3, yk3).
				MulWidenLoAdd(c4, yk4).MulWidenLoAdd(c5, yk5).
				MulWidenLoAdd(c6, yk6).MulWidenLoAdd(c7, yk7)
			hi := seedV.MulWidenHiAdd(c0, yk0).MulWidenHiAdd(c1, yk1).
				MulWidenHiAdd(c2, yk2).MulWidenHiAdd(c3, yk3).
				MulWidenHiAdd(c4, yk4).MulWidenHiAdd(c5, yk5).
				MulWidenHiAdd(c6, yk6).MulWidenHiAdd(c7, yk7)

			// srshr #round1Bits with saturating narrow to int16 (SQRSHRN/SQRSHRN2),
			// then [0,255] clamp (SQXTUN). The -roundOffset is folded into the seed.
			narrow := lo.ShiftRightRoundNarrow(round1Bits).ShiftRightRoundNarrowHi(hi, round1Bits)
			convStore8(dp, narrow)

			c0p = unsafe.Add(c0p, 8*imElem)
			c1p = unsafe.Add(c1p, 8*imElem)
			c2p = unsafe.Add(c2p, 8*imElem)
			c3p = unsafe.Add(c3p, 8*imElem)
			c4p = unsafe.Add(c4p, 8*imElem)
			c5p = unsafe.Add(c5p, 8*imElem)
			c6p = unsafe.Add(c6p, 8*imElem)
			c7p = unsafe.Add(c7p, 8*imElem)
			dp = unsafe.Add(dp, 8)
		}
	}
}

// convolveY8GoSIMD is the Go-native SIMD form of convolveY8PureGo for width>=8
// (multiple of 8). Byte-identical to the scalar reference; other widths fall
// back to it. It always runs the full 8-tap MAC; 4-tap kernels merely zero the
// end taps so their contribution vanishes, keeping the result bit-exact.
func convolveY8GoSIMD(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, kernel [filterTaps]int16) {
	if !(width >= 8 && width%8 == 0) {
		convolveY8PureGo(dst, ref, dstX, dstY, refX, refY, width, height, kernel)
		return
	}
	fo := filterTaps/2 - 1
	stride := ref.Stride

	// Broadcast the eight taps once, hoisting the DUPs out of the hot loop.
	k0 := archsimd.BroadcastInt16x8(kernel[0])
	k1 := archsimd.BroadcastInt16x8(kernel[1])
	k2 := archsimd.BroadcastInt16x8(kernel[2])
	k3 := archsimd.BroadcastInt16x8(kernel[3])
	k4 := archsimd.BroadcastInt16x8(kernel[4])
	k5 := archsimd.BroadcastInt16x8(kernel[5])
	k6 := archsimd.BroadcastInt16x8(kernel[6])
	k7 := archsimd.BroadcastInt16x8(kernel[7])
	zero := archsimd.BroadcastInt32x4(0)

	// Base pointer of the top tap row / first output column. Row walks advance by
	// the plane strides; column walks advance 8 bytes per 8-wide group. Every
	// access is in range (the dispatch guarantees the tap window is resident), so
	// there are no per-load bounds checks.
	rbase := unsafe.Pointer(&ref.Pix[(refY-fo)*stride+refX])
	dbase := unsafe.Pointer(&dst.Pix[dstY*dst.Stride+dstX])
	for y := 0; y < height; y++ {
		p0 := unsafe.Add(rbase, y*stride)
		p1 := unsafe.Add(p0, stride)
		p2 := unsafe.Add(p1, stride)
		p3 := unsafe.Add(p2, stride)
		p4 := unsafe.Add(p3, stride)
		p5 := unsafe.Add(p4, stride)
		p6 := unsafe.Add(p5, stride)
		p7 := unsafe.Add(p6, stride)
		dp := unsafe.Add(dbase, y*dst.Stride)
		for col := 0; col < width; col += 8 {
			c0 := convYLoad8(p0)
			c1 := convYLoad8(p1)
			c2 := convYLoad8(p2)
			c3 := convYLoad8(p3)
			c4 := convYLoad8(p4)
			c5 := convYLoad8(p5)
			c6 := convYLoad8(p6)
			c7 := convYLoad8(p7)

			// Columns 0..3 (SMLAL) and 4..7 (SMLAL2), one fused widening MAC per
			// tap into each int32 lane group: 16 SMLAL total.
			lo := zero.MulWidenLoAdd(c0, k0).MulWidenLoAdd(c1, k1).
				MulWidenLoAdd(c2, k2).MulWidenLoAdd(c3, k3).
				MulWidenLoAdd(c4, k4).MulWidenLoAdd(c5, k5).
				MulWidenLoAdd(c6, k6).MulWidenLoAdd(c7, k7)
			hi := zero.MulWidenHiAdd(c0, k0).MulWidenHiAdd(c1, k1).
				MulWidenHiAdd(c2, k2).MulWidenHiAdd(c3, k3).
				MulWidenHiAdd(c4, k4).MulWidenHiAdd(c5, k5).
				MulWidenHiAdd(c6, k6).MulWidenHiAdd(c7, k7)

			// roundPowerOfTwo(sum, filterBits) with signed saturating narrow to
			// int16 (SQRSHRN/SQRSHRN2), then [0,255] clamp (SQXTUN).
			narrow := lo.ShiftRightRoundNarrow(filterBits).ShiftRightRoundNarrowHi(hi, filterBits)
			convStore8(dp, narrow)

			p0 = unsafe.Add(p0, 8)
			p1 = unsafe.Add(p1, 8)
			p2 = unsafe.Add(p2, 8)
			p3 = unsafe.Add(p3, 8)
			p4 = unsafe.Add(p4, 8)
			p5 = unsafe.Add(p5, 8)
			p6 = unsafe.Add(p6, 8)
			p7 = unsafe.Add(p7, 8)
			dp = unsafe.Add(dp, 8)
		}
	}
}
