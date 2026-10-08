// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

// Go-native SIMD 8-bit motion-compensation convolve kernels. They use the
// official Go 1.27 archsimd API and must match the pure-Go reference sample for
// sample. Horizontal paths use guarded int16 multiply-accumulates when the
// filter bounds prove them exact and widening multiplies otherwise; the
// vertical path composes its dot products from official vector operations.
// Dispatch and performance are measured separately from these correctness
// kernels.

package motion

import (
	"simd/archsimd"
	"unsafe"

	"github.com/thesyncim/goav1/internal/av1/dsp/cpu"
	"github.com/thesyncim/goav1/internal/av1/frame"
)

// convStore8 narrows the low 8 int16 lanes of v to bytes (SQXTUN) and writes
// them contiguously at raw pointer p as a single 8-byte store. Only the low 8
// bytes are valid; there is no 8-byte narrow store in archsimd, so the byte
// vector's low 64 bits are extracted and stored directly (no slice bounds check,
// no panic path). Mirrors the loopfilter lf8StoreP idiom.
func convStore8(p unsafe.Pointer, v archsimd.Int16x8) {
	*(*uint64)(p) = v.SaturateToUint8().ReshapeToUint64s().GetElem(0)
}

// convStore8U8 writes the low 8 bytes of an already-narrowed Uint8x16 at raw
// pointer p as a single 8-byte store.
func convStore8U8(p unsafe.Pointer, v archsimd.Uint8x16) {
	*(*uint64)(p) = v.ReshapeToUint64s().GetElem(0)
}

// convolveX8GoSIMD is the Go-native SIMD form of convolveX8PureGo. Bounded
// filters use exact half-coefficient int16 multiply-accumulates; wider packed
// filters use int32 widening multiplies. Both paths preserve the reference's
// staged rounding and saturation. Unsupported kernels and widths fall back to
// the scalar reference.
func convolveX8GoSIMD(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, kernel [filterTaps]int16) {
	_, _, ok := convolveX8I8MMFilter(kernel)
	if !ok || !(width >= 8 && width%8 == 0) {
		convolveX8PureGo(dst, ref, dstX, dstY, refX, refY, width, height, kernel)
		return
	}
	if halfKernel, narrowOK := simdNarrowHorizontalKernel(kernel); narrowOK {
		convolveX8GoSIMDNarrow(dst, ref, dstX, dstY, refX, refY, width, height, halfKernel)
		return
	}
	fo := filterTaps/2 - 1
	k0 := archsimd.BroadcastInt16x8(kernel[0])
	k1 := archsimd.BroadcastInt16x8(kernel[1])
	k2 := archsimd.BroadcastInt16x8(kernel[2])
	k3 := archsimd.BroadcastInt16x8(kernel[3])
	k4 := archsimd.BroadcastInt16x8(kernel[4])
	k5 := archsimd.BroadcastInt16x8(kernel[5])
	k6 := archsimd.BroadcastInt16x8(kernel[6])
	k7 := archsimd.BroadcastInt16x8(kernel[7])

	rbase := unsafe.Pointer(&ref.Pix[refY*ref.Stride+refX-fo])
	dbase := unsafe.Pointer(&dst.Pix[dstY*dst.Stride+dstX])
	for y := 0; y < height; y++ {
		sp := unsafe.Add(rbase, y*ref.Stride)
		dp := unsafe.Add(dbase, y*dst.Stride)
		for col := 0; col < width; col += 8 {
			raw := archsimd.LoadUint8x16Array((*[16]uint8)(sp))
			lo := archsimd.BroadcastInt32x4(0)
			hi := archsimd.BroadcastInt32x4(0)
			lo, hi = simdHorizontalConvMAC(lo, hi, raw.ExtendLo8ToUint16().ConvertToInt16(), k0)
			lo, hi = simdHorizontalConvMAC(lo, hi, raw.ConcatShiftBytesRight(raw, 1).ExtendLo8ToUint16().ConvertToInt16(), k1)
			lo, hi = simdHorizontalConvMAC(lo, hi, raw.ConcatShiftBytesRight(raw, 2).ExtendLo8ToUint16().ConvertToInt16(), k2)
			lo, hi = simdHorizontalConvMAC(lo, hi, raw.ConcatShiftBytesRight(raw, 3).ExtendLo8ToUint16().ConvertToInt16(), k3)
			lo, hi = simdHorizontalConvMAC(lo, hi, raw.ConcatShiftBytesRight(raw, 4).ExtendLo8ToUint16().ConvertToInt16(), k4)
			lo, hi = simdHorizontalConvMAC(lo, hi, raw.ConcatShiftBytesRight(raw, 5).ExtendLo8ToUint16().ConvertToInt16(), k5)
			lo, hi = simdHorizontalConvMAC(lo, hi, raw.ConcatShiftBytesRight(raw, 6).ExtendLo8ToUint16().ConvertToInt16(), k6)
			lo, hi = simdHorizontalConvMAC(lo, hi, raw.ConcatShiftBytesRight(raw, 7).ExtendLo8ToUint16().ConvertToInt16(), k7)
			// Preserve the reference's two separately rounded shifts. The first
			// narrows the intermediate to int16; the second applies the remaining
			// filter shift and clamps to the output byte range.
			stage1 := simdRoundShiftNarrowInt32Pair(lo, hi, round0Bits)
			out := simdRoundShiftNarrowUint8(stage1, filterBits-round0Bits)
			convStore8U8(dp, out)

			if col+8 < width {
				sp = unsafe.Add(sp, 8)
				dp = unsafe.Add(dp, 8)
			}
		}
	}
}

// convolveX8GoSIMDNarrow uses int16 multiply-accumulates only when the half
// filter's L1 coefficient bound proves every partial sum exact. Widening and
// doubling restores the full even-coefficient sum before the reference's two
// separately rounded and saturated stages.
func convolveX8GoSIMDNarrow(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, kernel [filterTaps]int16) {
	fo := filterTaps/2 - 1
	k0 := archsimd.BroadcastInt16x8(kernel[0])
	k1 := archsimd.BroadcastInt16x8(kernel[1])
	k2 := archsimd.BroadcastInt16x8(kernel[2])
	k3 := archsimd.BroadcastInt16x8(kernel[3])
	k4 := archsimd.BroadcastInt16x8(kernel[4])
	k5 := archsimd.BroadcastInt16x8(kernel[5])
	k6 := archsimd.BroadcastInt16x8(kernel[6])
	k7 := archsimd.BroadcastInt16x8(kernel[7])
	zero := archsimd.BroadcastInt16x8(0)
	const collapsedRoundBias = int16(34)
	roundBias := archsimd.BroadcastInt16x8(collapsedRoundBias)
	roundShift := archsimd.BroadcastInt16x8(-6)

	rbase := unsafe.Pointer(&ref.Pix[refY*ref.Stride+refX-fo])
	dbase := unsafe.Pointer(&dst.Pix[dstY*dst.Stride+dstX])
	for y := 0; y < height; y++ {
		sp := unsafe.Add(rbase, y*ref.Stride)
		dp := unsafe.Add(dbase, y*dst.Stride)
		for col := 0; col < width; col += 8 {
			raw := archsimd.LoadUint8x16Array((*[16]uint8)(sp))
			even := raw.ExtendLo8ToUint16().ConvertToInt16().MulAdd(k0, zero)
			odd := raw.ConcatShiftBytesRight(raw, 1).ExtendLo8ToUint16().ConvertToInt16().MulAdd(k1, zero)
			even = raw.ConcatShiftBytesRight(raw, 2).ExtendLo8ToUint16().ConvertToInt16().MulAdd(k2, even)
			odd = raw.ConcatShiftBytesRight(raw, 3).ExtendLo8ToUint16().ConvertToInt16().MulAdd(k3, odd)
			even = raw.ConcatShiftBytesRight(raw, 4).ExtendLo8ToUint16().ConvertToInt16().MulAdd(k4, even)
			odd = raw.ConcatShiftBytesRight(raw, 5).ExtendLo8ToUint16().ConvertToInt16().MulAdd(k5, odd)
			even = raw.ConcatShiftBytesRight(raw, 6).ExtendLo8ToUint16().ConvertToInt16().MulAdd(k6, even)
			odd = raw.ConcatShiftBytesRight(raw, 7).ExtendLo8ToUint16().ConvertToInt16().MulAdd(k7, odd)
			// With full taps equal to 2*half taps, the two reference shifts
			// collapse exactly to floor((halfSum+34)/64). The guard bounds
			// halfSum to [-32640,32640], so the bias and arithmetic shift stay
			// inside int16 and the omitted intermediate saturation cannot trigger.
			out := even.Add(odd).Add(roundBias).Shift(roundShift).SaturateToUint8()
			convStore8U8(dp, out)

			if col+8 < width {
				sp = unsafe.Add(sp, 8)
				dp = unsafe.Add(dp, 8)
			}
		}
	}
}

// convolveX8GoSIMDDispatch routes supported width>=8 blocks to the Go-native
// SIMD kernel and other shapes to the best asm tier (I8MM's width-4 path when
// available, otherwise NEON). Both paths are byte-identical to the reference.
func convolveX8GoSIMDDispatch(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, kernel [filterTaps]int16) {
	if width >= 8 && width%8 == 0 {
		if _, _, ok := convolveX8I8MMFilter(kernel); ok {
			convolveX8GoSIMD(dst, ref, dstX, dstY, refX, refY, width, height, kernel)
			return
		}
	}
	if cpu.Detected.I8MM {
		convolveX8I8MM(dst, ref, dstX, dstY, refX, refY, width, height, kernel)
	} else {
		convolveX8NEON(dst, ref, dstX, dstY, refX, refY, width, height, kernel)
	}
}

// convIMLoad8 loads 8 contiguous int16 intermediates at raw pointer p as an
// Int16x8 (register-direct FMOVQ, no bounds check).
func convIMLoad8(p unsafe.Pointer) archsimd.Int16x8 {
	return archsimd.LoadInt16x8Array((*[8]int16)(p))
}

// convolve2D8GoSIMD is the Go-native SIMD form of convolve2D8PureGo for widths
// that are positive multiples of 8. The horizontal pass produces the int16
// intermediate with the reference's xBias and round0Bits stages. The vertical
// pass uses int16 widening multiplies followed by the staged round1 shift,
// roundOffset subtraction, and [0,255] clip. The intermediate is caller-owned
// scratch when provided or a stack array otherwise.
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

	// ---- Horizontal pass: byte ref -> int16 im. The packed kernel is
	// reconstructed without loss. Filters with a bounded half-coefficient sum
	// use int16 multiply-accumulates; the wide path preserves arbitrary packed
	// filters that exceed that proven range.
	kernel := simdKernelFromI8MMFilter(filter, f0)
	const xBias = 1 << (8 + filterBits - 1)

	rbase := unsafe.Pointer(&ref.Pix[(refY-foY)*ref.Stride+refX-foX])
	ibase := unsafe.Pointer(&im[0])
	const imElem = 2
	if halfKernel, narrowOK := simdNarrowHorizontalKernel(kernel); narrowOK {
		simdHorizontalNarrowToIM(rbase, ibase, width, imH, ref.Stride, imStride, halfKernel, xBias)
	} else {
		k0 := archsimd.BroadcastInt16x8(kernel[0])
		k1 := archsimd.BroadcastInt16x8(kernel[1])
		k2 := archsimd.BroadcastInt16x8(kernel[2])
		k3 := archsimd.BroadcastInt16x8(kernel[3])
		k4 := archsimd.BroadcastInt16x8(kernel[4])
		k5 := archsimd.BroadcastInt16x8(kernel[5])
		k6 := archsimd.BroadcastInt16x8(kernel[6])
		k7 := archsimd.BroadcastInt16x8(kernel[7])
		for y := 0; y < imH; y++ {
			sp := unsafe.Add(rbase, y*ref.Stride)
			ip := unsafe.Add(ibase, y*imStride*imElem)
			for col := 0; col < width; col += 8 {
				raw := archsimd.LoadUint8x16Array((*[16]uint8)(sp))
				lo := archsimd.BroadcastInt32x4(xBias)
				hi := archsimd.BroadcastInt32x4(xBias)
				lo, hi = simdHorizontalConvMAC(lo, hi, raw.ExtendLo8ToUint16().ConvertToInt16(), k0)
				lo, hi = simdHorizontalConvMAC(lo, hi, raw.ConcatShiftBytesRight(raw, 1).ExtendLo8ToUint16().ConvertToInt16(), k1)
				lo, hi = simdHorizontalConvMAC(lo, hi, raw.ConcatShiftBytesRight(raw, 2).ExtendLo8ToUint16().ConvertToInt16(), k2)
				lo, hi = simdHorizontalConvMAC(lo, hi, raw.ConcatShiftBytesRight(raw, 3).ExtendLo8ToUint16().ConvertToInt16(), k3)
				lo, hi = simdHorizontalConvMAC(lo, hi, raw.ConcatShiftBytesRight(raw, 4).ExtendLo8ToUint16().ConvertToInt16(), k4)
				lo, hi = simdHorizontalConvMAC(lo, hi, raw.ConcatShiftBytesRight(raw, 5).ExtendLo8ToUint16().ConvertToInt16(), k5)
				lo, hi = simdHorizontalConvMAC(lo, hi, raw.ConcatShiftBytesRight(raw, 6).ExtendLo8ToUint16().ConvertToInt16(), k6)
				lo, hi = simdHorizontalConvMAC(lo, hi, raw.ConcatShiftBytesRight(raw, 7).ExtendLo8ToUint16().ConvertToInt16(), k7)
				simdRoundShiftNarrowInt32Pair(lo, hi, round0Bits).StoreArray((*[8]int16)(ip))

				if col+8 < width {
					sp = unsafe.Add(sp, 8)
					ip = unsafe.Add(ip, 8*imElem)
				}
			}
		}
	}

	// ---- Vertical pass: int16 im -> uint8 dst using widening MACs. ----
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
	// multiple of 1<<round1Bits, the rounded shift of (sum - roundOffset<<n)
	// equals the rounded shift of sum minus roundOffset. Keeping the subtraction
	// in int32 avoids int16 underflow before the rounding shift and final byte clamp.
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

			lo := seedV.Add(c0.MulWidenLo(yk0)).Add(c1.MulWidenLo(yk1)).
				Add(c2.MulWidenLo(yk2)).Add(c3.MulWidenLo(yk3)).
				Add(c4.MulWidenLo(yk4)).Add(c5.MulWidenLo(yk5)).
				Add(c6.MulWidenLo(yk6)).Add(c7.MulWidenLo(yk7))
			hi := seedV.Add(c0.HiToLo().MulWidenLo(yk0.HiToLo())).Add(c1.HiToLo().MulWidenLo(yk1.HiToLo())).
				Add(c2.HiToLo().MulWidenLo(yk2.HiToLo())).Add(c3.HiToLo().MulWidenLo(yk3.HiToLo())).
				Add(c4.HiToLo().MulWidenLo(yk4.HiToLo())).Add(c5.HiToLo().MulWidenLo(yk5.HiToLo())).
				Add(c6.HiToLo().MulWidenLo(yk6.HiToLo())).Add(c7.HiToLo().MulWidenLo(yk7.HiToLo()))

			// Rounded shift with saturating int16 narrow, then [0,255] clamp. The
			// -roundOffset is folded into the seed.
			narrow := simdRoundShiftNarrowInt32Pair(lo, hi, round1Bits)
			convStore8(dp, narrow)

			if col+8 < width {
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
}

// convYLoadRow8 loads 8 reference bytes at p into the low 8 lanes of a Uint8x16
// (a 16-byte register-direct load; only the low 8 are used by the ZIP transpose,
// the high 8 are the next columns and are discarded). p indexes into the resident
// tap window, so the load never bounds-faults.
func convYLoadRow8(p unsafe.Pointer) archsimd.Uint8x16 {
	return archsimd.LoadUint8x16Array((*[16]uint8)(p))
}

// convolveY8GoSIMDUSDOT follows the transposed dot-product layout used by the
// ARM I8MM kernel: tap rows are interleaved so each output column's taps become
// contiguous, and simdDotProdUS computes the dot products from official Go SIMD
// operations. It produces four output rows per iteration and preserves the
// reference's rounding and byte saturation. Unsupported widths, heights, or
// filters fall back to the I8MM/NEON implementation.
//
// Every AV1 tap is even, so kernel[i]>>1 fits int8 and each dot product is half
// the full convolution. The final rounding shift accounts for that scale.
func convolveY8GoSIMDUSDOT(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, kernel [filterTaps]int16) {
	filter, taps, ok := convolveY8I8MMFilter(kernel)
	if !ok || taps != 8 || !(width >= 8 && width%8 == 0) || height%4 != 0 {
		convolveY8I8MM(dst, ref, dstX, dstY, refX, refY, width, height, kernel)
		return
	}
	// fLo / fHi: the low four taps and high four taps, each 4-byte group replicated
	// across the vector lanes so simdDotProdUS applies the same taps to each column.
	var fLoArr, fHiArr [16]int8
	for g := 0; g < 4; g++ {
		fLoArr[g*4+0] = int8(filter[0])
		fLoArr[g*4+1] = int8(filter[1])
		fLoArr[g*4+2] = int8(filter[2])
		fLoArr[g*4+3] = int8(filter[3])
		fHiArr[g*4+0] = int8(filter[4])
		fHiArr[g*4+1] = int8(filter[5])
		fHiArr[g*4+2] = int8(filter[6])
		fHiArr[g*4+3] = int8(filter[7])
	}
	fLo := archsimd.LoadInt8x16Array(&fLoArr)
	fHi := archsimd.LoadInt8x16Array(&fHiArr)
	zero := archsimd.BroadcastInt32x4(0)

	fo := filterTaps/2 - 1 // 3
	stride := ref.Stride
	dstStride := dst.Stride
	rbase := unsafe.Pointer(&ref.Pix[(refY-fo)*stride+refX])
	dbase := unsafe.Pointer(&dst.Pix[dstY*dstStride+dstX])
	for col := 0; col < width; col += 8 {
		rc := unsafe.Add(rbase, col)
		dp := unsafe.Add(dbase, col)
		y := 0
		// Eight output rows per block from fifteen input rows: transpose all
		// thirteen zRow(j) = zip1(row j, row j+2) once, then emit d0..d7. The block
		// is self-contained (no loop-carried z state), which trades the four-row
		// sliding window's per-iteration PHI-copy shuffle for a couple extra ZIPs and
		// a little z-spill -- a net win (118ns -> 112ns) since the prime amortizes
		// over eight rows and there is no loop-back-edge register shuffle.
		for ; y+8 <= height; y += 8 {
			p := unsafe.Add(rc, y*stride)
			s0 := convYLoadRow8(p)
			p = unsafe.Add(p, stride)
			s1 := convYLoadRow8(p)
			p = unsafe.Add(p, stride)
			s2 := convYLoadRow8(p)
			p = unsafe.Add(p, stride)
			s3 := convYLoadRow8(p)
			p = unsafe.Add(p, stride)
			s4 := convYLoadRow8(p)
			p = unsafe.Add(p, stride)
			s5 := convYLoadRow8(p)
			p = unsafe.Add(p, stride)
			s6 := convYLoadRow8(p)
			p = unsafe.Add(p, stride)
			s7 := convYLoadRow8(p)
			p = unsafe.Add(p, stride)
			s8 := convYLoadRow8(p)
			p = unsafe.Add(p, stride)
			s9 := convYLoadRow8(p)
			p = unsafe.Add(p, stride)
			s10 := convYLoadRow8(p)
			p = unsafe.Add(p, stride)
			s11 := convYLoadRow8(p)
			p = unsafe.Add(p, stride)
			s12 := convYLoadRow8(p)
			p = unsafe.Add(p, stride)
			s13 := convYLoadRow8(p)
			p = unsafe.Add(p, stride)
			s14 := convYLoadRow8(p)
			z0 := s0.InterleaveLo(s2)
			z1 := s1.InterleaveLo(s3)
			z2 := s2.InterleaveLo(s4)
			z3 := s3.InterleaveLo(s5)
			z4 := s4.InterleaveLo(s6)
			z5 := s5.InterleaveLo(s7)
			z6 := s6.InterleaveLo(s8)
			z7 := s7.InterleaveLo(s9)
			z8 := s8.InterleaveLo(s10)
			z9 := s9.InterleaveLo(s11)
			z10 := s10.InterleaveLo(s12)
			z11 := s11.InterleaveLo(s13)
			z12 := s12.InterleaveLo(s14)
			convY8Emit(dp, zero, z0, z1, z4, z5, fLo, fHi)
			convY8Emit(unsafe.Add(dp, dstStride), zero, z1, z2, z5, z6, fLo, fHi)
			convY8Emit(unsafe.Add(dp, 2*dstStride), zero, z2, z3, z6, z7, fLo, fHi)
			convY8Emit(unsafe.Add(dp, 3*dstStride), zero, z3, z4, z7, z8, fLo, fHi)
			convY8Emit(unsafe.Add(dp, 4*dstStride), zero, z4, z5, z8, z9, fLo, fHi)
			convY8Emit(unsafe.Add(dp, 5*dstStride), zero, z5, z6, z9, z10, fLo, fHi)
			convY8Emit(unsafe.Add(dp, 6*dstStride), zero, z6, z7, z10, z11, fLo, fHi)
			convY8Emit(unsafe.Add(dp, 7*dstStride), zero, z7, z8, z11, z12, fLo, fHi)
			if y+8 < height {
				dp = unsafe.Add(dp, 8*dstStride)
			}
		}
		// Four-row tail (height % 8 == 4): eleven input rows -> nine zRow, emit d0..d3.
		if y < height {
			p := unsafe.Add(rc, y*stride)
			s0 := convYLoadRow8(p)
			p = unsafe.Add(p, stride)
			s1 := convYLoadRow8(p)
			p = unsafe.Add(p, stride)
			s2 := convYLoadRow8(p)
			p = unsafe.Add(p, stride)
			s3 := convYLoadRow8(p)
			p = unsafe.Add(p, stride)
			s4 := convYLoadRow8(p)
			p = unsafe.Add(p, stride)
			s5 := convYLoadRow8(p)
			p = unsafe.Add(p, stride)
			s6 := convYLoadRow8(p)
			p = unsafe.Add(p, stride)
			s7 := convYLoadRow8(p)
			p = unsafe.Add(p, stride)
			s8 := convYLoadRow8(p)
			p = unsafe.Add(p, stride)
			s9 := convYLoadRow8(p)
			p = unsafe.Add(p, stride)
			s10 := convYLoadRow8(p)
			z0 := s0.InterleaveLo(s2)
			z1 := s1.InterleaveLo(s3)
			z2 := s2.InterleaveLo(s4)
			z3 := s3.InterleaveLo(s5)
			z4 := s4.InterleaveLo(s6)
			z5 := s5.InterleaveLo(s7)
			z6 := s6.InterleaveLo(s8)
			z7 := s7.InterleaveLo(s9)
			z8 := s8.InterleaveLo(s10)
			convY8Emit(dp, zero, z0, z1, z4, z5, fLo, fHi)
			convY8Emit(unsafe.Add(dp, dstStride), zero, z1, z2, z5, z6, fLo, fHi)
			convY8Emit(unsafe.Add(dp, 2*dstStride), zero, z2, z3, z6, z7, fLo, fHi)
			convY8Emit(unsafe.Add(dp, 3*dstStride), zero, z3, z4, z7, z8, fLo, fHi)
		}
	}
}

// convY8Emit computes one 8-column vertical-convolve output row and stores it.
// za/zb are the low-tap (0..3) window's zRow pair, zc/zd the high-tap (4..7)
// window's; InterleaveLo -> columns 0..3, InterleaveHi -> columns 4..7. The
// helper composes each halved-tap dot product, then the result is rounded,
// narrowed, and clamped to bytes. Keep it small enough to inline in the hot loop.
func convY8Emit(dstp unsafe.Pointer, zero archsimd.Int32x4, za, zb, zc, zd archsimd.Uint8x16, fLo, fHi archsimd.Int8x16) {
	lo := simdDotProdUS(simdDotProdUS(zero, za.InterleaveLo(zb), fLo), zc.InterleaveLo(zd), fHi)
	hi := simdDotProdUS(simdDotProdUS(zero, za.InterleaveHi(zb), fLo), zc.InterleaveHi(zd), fHi)
	packed := simdConcatInt16x8(lo.TruncToInt16(), hi.TruncToInt16())
	convStore8U8(dstp, simdRoundShiftNarrowUint8(packed, 6))
}
