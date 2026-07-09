// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

// Go-native SIMD 8-bit motion-compensation convolve kernels. These target the
// #2 decode hotspot (inter prediction) and are byte-identical to the pure-Go
// reference in vector.go, which every variant must match sample for sample.
//
// The horizontal (X) pass (convolveX8GoSIMD) beats the I8MM asm via USMMLA plus
// the fused SQRSHRUN round-narrow tail and is wired as the dispatch kernel.
//
// The vertical (Y) pass (convolveY8GoSIMDUSDOT) ports SVT's transposed-USDOT
// algorithm: the tap rows are byte-transposed with ZIP so each output column's
// taps land as four contiguous bytes, then two USDOT per lane group accumulate
// the (all-halved) int8 filter's dot products and SQRSHRUN #6 rounds+narrows the
// int16 half-sum. It is ~2x faster than the naive strided widening MAC and
// byte-exact, but still ~1.26x behind the hand-scheduled I8MM asm -- Go's
// instruction scheduler on the transpose-heavy body is the wall -- so Y8 keeps
// the asm tier dispatched and this kernel is retained as the fastest pure-Go
// vertical form.

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
	// Horizontal bias only: the asm adds 2 then sqrshrun #6. SQRSHRUN performs
	// the rounding (its internal +1<<5) and the [0,255] clamp itself, so the
	// shim carries just the +2 and the tail is a single fused round-narrow.
	const xShim = 2
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
				// Fused round-shift-narrow-clamp: sqrshrun #6 == round(+1<<5),
				// arith-shift #6, saturate to [0,255]. Replaces asr #6 + sqxtun.
				out := sumHalf.Add(shimV).ShiftRightRoundNarrowUint8(6)
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
			out := sumHalf.Sub(tap0).Add(shimV).ShiftRightRoundNarrowUint8(6)
			convStore8U8(dp, out)

			sp = unsafe.Add(sp, 8)
			dp = unsafe.Add(dp, 8)
		}
	}
}

// convolveX8GoSIMDDispatch routes the horizontal-8 convolve to the Go-native
// SIMD kernel (which beats the I8MM asm via the fused SQRSHRUN round-narrow tail
// -- one instruction for round-shift+narrow+[0,255]-clamp) for width>=8
// multiple-of-8 blocks whose taps fit the USMMLA packing, and to the best asm
// tier (I8MM's fast width-4 4-tap path, else NEON) for the narrow and
// unsupported shapes. Byte-identical on every shape.
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

// convYLoadRow8 loads 8 reference bytes at p into the low 8 lanes of a Uint8x16
// (a 16-byte register-direct load; only the low 8 are used by the ZIP transpose,
// the high 8 are the next columns and are discarded). p indexes into the resident
// tap window, so the load never bounds-faults.
func convYLoadRow8(p unsafe.Pointer) archsimd.Uint8x16 {
	return archsimd.LoadUint8x16Array((*[16]uint8)(p))
}

// convolveY8GoSIMDUSDOT is the Go-native SIMD form of the vertical 8-tap convolve
// using the transposed-USDOT algorithm of SVT's convolve_y_sr_8tap_neon_i8mm (the
// same one convolveY8I8MMAsm implements): the eight tap rows are byte-transposed
// with ZIP so each output column's taps become four contiguous bytes, then two
// USDOT per lane-group dot them against the (even+odd all halved) int8 filter, and
// the int16 half-sum is round-shifted+narrowed+clamped in one SQRSHRUN #6. Four
// output rows are produced per iteration from eleven input rows, reusing the
// shared ZIP stages. It replaces the naive strided SMLAL MAC, which loses ~2.6x to
// the asm because it issues 16 widening MACs per 8 outputs where USDOT needs four.
// Byte-identical to convolveY8PureGo; shapes it does not cover fall back to asm.
//
// Filter math: every AV1 tap is even, so filter[i] = kernel[i]>>1 fits int8 and
// USDOT accumulates sum(sample*kernel/2) = half the true convolution; SQRSHRUN #6
// then yields round(sum*kernel >> 7) saturated to [0,255] with no tap-0 fixup.
func convolveY8GoSIMDUSDOT(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, kernel [filterTaps]int16) {
	filter, taps, ok := convolveY8I8MMFilter(kernel)
	if !ok || taps != 8 || !(width >= 8 && width%8 == 0) || height%4 != 0 {
		convolveY8I8MM(dst, ref, dstX, dstY, refX, refY, width, height, kernel)
		return
	}
	// fLo / fHi: the low four taps and high four taps, each 4-byte group replicated
	// across all four USDOT lane groups so the plain (non-indexed) USDOT applies the
	// same four taps to every column, matching the asm's usdot ..., v0.4b[0/1].
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
			dp = unsafe.Add(dp, 8*dstStride)
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
// window's; InterleaveLo -> columns 0..3, InterleaveHi -> columns 4..7. Two USDOT
// per lane group accumulate the halved-tap dot products (int16 half-sum), then
// SQRSHRUN #6 rounds+narrows+clamps. Kept tiny so it inlines (no CALL in the hot
// body); verified via the CALL-target dump.
func convY8Emit(dstp unsafe.Pointer, zero archsimd.Int32x4, za, zb, zc, zd archsimd.Uint8x16, fLo, fHi archsimd.Int8x16) {
	lo := zero.DotProdUS(za.InterleaveLo(zb), fLo).DotProdUS(zc.InterleaveLo(zd), fHi)
	hi := zero.DotProdUS(za.InterleaveHi(zb), fLo).DotProdUS(zc.InterleaveHi(zd), fHi)
	convStore8U8(dstp, lo.TruncToInt16().TruncToInt16Hi(hi).ShiftRightRoundNarrowUint8(6))
}
