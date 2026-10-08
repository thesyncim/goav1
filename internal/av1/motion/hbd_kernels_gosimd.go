// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && (amd64 || arm64) && !purego

package motion

// Go-native SIMD high-bit-depth (10/12-bit) inter-prediction kernels. Each
// kernel reads the calling context the HBD wrappers build and computes exactly
// the scalar Go formula of its pure-Go reference (convolve*HighBDPureGo and
// predictInterCompoundRefHighBD*), so output is byte-identical.
//
// Every kernel works on slices, not raw pointers. Sample vectors are loaded
// from the little-endian byte planes with LoadUint8x16 and reshaped to uint16
// lanes; intermediates are []int32 and compound buffers are []uint16. This keeps
// the kernels free of unsafe pointer arithmetic, so no local window, filter or
// intermediate buffer is forced onto the heap when the race or checkptr
// instrumentation is enabled.
//
// The wrappers trim every filter to its nonzero tap span (hbdTapSpan) and start
// the sample slice at the span's first tap. Zero taps contribute nothing, so
// trimming is exact, and a kernel reads exactly the window the scalar reference
// reads.
//
// Lane layout: eight output samples per iteration (two Int32x4 accumulators)
// with a four-sample tail. Tails stage their window through a local array that
// is only sliced, so an eight-lane load never reads past the footprint.

import (
	"simd/archsimd"
)

// hbdTaps16 holds the broadcast coefficients of one filter for the sample-side
// (int16) multiply-widen. Entries at index >= the tap count are zero.
type hbdTaps16 [filterTaps]archsimd.Int16x8

// hbdTaps32 holds the broadcast coefficients for the int32 intermediate.
type hbdTaps32 [filterTaps]archsimd.Int32x4

// hbdTapSpan returns the first tap index and the tap count that cover every
// nonzero coefficient. An all-zero filter yields a single zero tap.
func hbdTapSpan(kernel *[filterTaps]int16) (lo, n int) {
	lo, hi := 0, filterTaps-1
	for lo < hi && kernel[lo] == 0 {
		lo++
	}
	for hi > lo && kernel[hi] == 0 {
		hi--
	}
	return lo, hi - lo + 1
}

// hbdTrimTaps copies the n taps starting at lo into a zero-padded array.
func hbdTrimTaps(kernel *[filterTaps]int16, lo, n int) [filterTaps]int16 {
	var k [filterTaps]int16
	copy(k[:n], kernel[lo:lo+n])
	return k
}

func hbdBroadcastTaps16(k *[filterTaps]int16) hbdTaps16 {
	var t hbdTaps16
	for i := range t {
		t[i] = archsimd.BroadcastInt16x8(k[i])
	}
	return t
}

func hbdBroadcastTaps32(k *[filterTaps]int16) hbdTaps32 {
	var t hbdTaps32
	for i := range t {
		t[i] = archsimd.BroadcastInt32x4(int32(k[i]))
	}
	return t
}

// hbdLoad8 loads eight little-endian uint16 samples from the first 16 bytes of
// pix as signed int16 lanes. HBD samples are at most 4095, so the bit pattern
// is a non-negative int16.
func hbdLoad8(pix []byte) archsimd.Int16x8 {
	return archsimd.LoadUint8x16(pix[:16]).ReshapeToUint16s().BitsToInt16()
}

// hbdSum8 returns the eight-lane tap sum over n taps: lanes 0..3 in lo, lanes
// 4..7 in hi. Tap k reads eight samples starting step bytes after tap k-1.
func hbdSum8(pix []byte, step int, t *hbdTaps16, n int) (lo, hi archsimd.Int32x4) {
	lo = archsimd.BroadcastInt32x4(0)
	hi = lo
	for k := range n {
		lo, hi = hbdMAC8(lo, hi, hbdLoad8(pix[k*step:]), t[k])
	}
	return lo, hi
}

// hbdSum8Int32 is hbdSum8 over the int32 intermediate: tap k reads eight int32
// samples starting step elements after tap k-1.
func hbdSum8Int32(im []int32, step int, t *hbdTaps32, n int) (lo, hi archsimd.Int32x4) {
	lo = archsimd.BroadcastInt32x4(0)
	hi = lo
	for k := range n {
		row := im[k*step:]
		lo = hbdMulAdd32(archsimd.LoadInt32x4(row[:4]), t[k], lo)
		hi = hbdMulAdd32(archsimd.LoadInt32x4(row[4:8]), t[k], hi)
	}
	return lo, hi
}

// hbdSum4Int32 is the four-lane form of hbdSum8Int32.
func hbdSum4Int32(im []int32, step int, t *hbdTaps32, n int) archsimd.Int32x4 {
	sum := archsimd.BroadcastInt32x4(0)
	for k := range n {
		sum = hbdMulAdd32(archsimd.LoadInt32x4(im[k*step:][:4]), t[k], sum)
	}
	return sum
}

// hbdRound returns (v + 2^(n-1)) >> n for n > 0 and v unchanged for n == 0,
// matching roundPowerOfTwo. The shift is an arithmetic right shift.
func hbdRound(v archsimd.Int32x4, n int) archsimd.Int32x4 {
	if n <= 0 {
		return v
	}
	rnd := archsimd.BroadcastInt32x4(int32(1) << (n - 1))
	return v.Add(rnd).ShiftAllRight(uint64(n))
}

// hbdClip clamps to [0, max] (clipPixelHighBD).
func hbdClip(v, zero, maxV archsimd.Int32x4) archsimd.Int32x4 {
	return v.Max(zero).Min(maxV)
}

// hbdFinish2D is the vertical stage of the 2D convolve: round, remove the
// intermediate bias, apply the final bits shift, then clip.
func hbdFinish2D(v archsimd.Int32x4, round1 int, rndOff archsimd.Int32x4, bits int, zero, maxV archsimd.Int32x4) archsimd.Int32x4 {
	v = hbdRound(v, round1).Sub(rndOff)
	return hbdClip(hbdRound(v, bits), zero, maxV)
}

// hbdStorePix8 writes eight clipped HBD samples (little-endian uint16) at the
// start of dst.
func hbdStorePix8(dst []byte, lo, hi archsimd.Int32x4) {
	hbdJoinU16(lo, hi).ReshapeToUint8s().Store(dst[:16])
}

// hbdStorePix4 writes the four low lanes of lo as HBD samples at the start of
// dst.
func hbdStorePix4(dst []byte, lo archsimd.Int32x4) {
	var t [8]uint16
	hbdJoinU16(lo, lo).Store(t[:])
	for i := range 4 {
		dst[2*i] = byte(t[i])
		dst[2*i+1] = byte(t[i] >> 8)
	}
}

// hbdStoreU16x8 writes eight CONV_BUF samples at the start of out.
func hbdStoreU16x8(out []uint16, lo, hi archsimd.Int32x4) {
	hbdJoinU16(lo, hi).Store(out[:8])
}

// hbdStoreU16x4 writes the four low lanes of lo as CONV_BUF samples at the start
// of out.
func hbdStoreU16x4(out []uint16, lo archsimd.Int32x4) {
	var t [8]uint16
	hbdJoinU16(lo, lo).Store(t[:])
	copy(out[:4], t[:4])
}

// convolveXHighBDKernel is the 1D horizontal HBD convolve over the context of
// convolveXHighBDGoSIMD. round0 is the first shift, round1 the final bits shift
// and maxVal the clip bound; the sample slices start at the trimmed tap span.
func convolveXHighBDKernel(ctx *convolveHighBDGoSIMDCtx) {
	n := int(ctx.tapsX)
	taps := hbdBroadcastTaps16(&ctx.kernel)
	zero := archsimd.BroadcastInt32x4(0)
	maxV := archsimd.BroadcastInt32x4(ctx.maxVal)
	round0, bits := int(ctx.round0), int(ctx.round1)
	width, height := int(ctx.width), int(ctx.height)
	for y := range height {
		src := ctx.ref[y*ctx.refStr:]
		dst := ctx.dst[y*ctx.dstStr:]
		x := 0
		for ; x+8 <= width; x += 8 {
			lo, hi := hbdSumRowX(src[x*2:], &taps, n)
			lo = hbdClip(hbdRound(hbdRound(lo, round0), bits), zero, maxV)
			hi = hbdClip(hbdRound(hbdRound(hi, round0), bits), zero, maxV)
			hbdStorePix8(dst[x*2:], lo, hi)
		}
		if x < width {
			var win [32]byte
			copy(win[:2*(n+3)], src[x*2:x*2+2*(n+3)])
			lo, _ := hbdSumRowX(win[:], &taps, n)
			lo = hbdClip(hbdRound(hbdRound(lo, round0), bits), zero, maxV)
			hbdStorePix4(dst[x*2:], lo)
		}
	}
}

// convolveYHighBDKernel is the 1D vertical HBD convolve over the context of
// convolveYHighBDGoSIMD: one FILTER_BITS rounding, then clip.
func convolveYHighBDKernel(ctx *convolveHighBDGoSIMDCtx) {
	n := int(ctx.tapsY)
	taps := hbdBroadcastTaps16(&ctx.kernel)
	zero := archsimd.BroadcastInt32x4(0)
	maxV := archsimd.BroadcastInt32x4(ctx.maxVal)
	round0 := int(ctx.round0)
	width, height := int(ctx.width), int(ctx.height)
	for y := range height {
		src := ctx.ref[y*ctx.refStr:]
		dst := ctx.dst[y*ctx.dstStr:]
		x := 0
		for ; x+8 <= width; x += 8 {
			lo, hi := hbdSum8(src[x*2:], ctx.refStr, &taps, n)
			lo = hbdClip(hbdRound(lo, round0), zero, maxV)
			hi = hbdClip(hbdRound(hi, round0), zero, maxV)
			hbdStorePix8(dst[x*2:], lo, hi)
		}
		if x < width {
			var win [128]byte
			for k := range n {
				copy(win[k*16:k*16+8], src[k*ctx.refStr+x*2:k*ctx.refStr+x*2+8])
			}
			lo, _ := hbdSum8(win[:], 16, &taps, n)
			lo = hbdClip(hbdRound(lo, round0), zero, maxV)
			hbdStorePix4(dst[x*2:], lo)
		}
	}
}

// hbdHorizontalIM computes the biased horizontal pass of the separable 2D
// convolves: rows reference rows, each becoming round0(bias + sum) int32
// intermediate samples at row stride imStr elements.
func hbdHorizontalIM(ref []byte, refStr int, im []int32, imStr int, width, rows int, taps *hbdTaps16, n int, bias archsimd.Int32x4, round0 int) {
	for y := range rows {
		src := ref[y*refStr:]
		row := im[y*imStr:]
		x := 0
		for ; x+8 <= width; x += 8 {
			lo, hi := hbdSumRowX(src[x*2:], taps, n)
			lo = hbdRound(lo.Add(bias), round0)
			hi = hbdRound(hi.Add(bias), round0)
			lo.Store(row[x : x+4])
			hi.Store(row[x+4 : x+8])
		}
		if x < width {
			var win [32]byte
			copy(win[:2*(n+3)], src[x*2:x*2+2*(n+3)])
			lo, _ := hbdSumRowX(win[:], taps, n)
			lo = hbdRound(lo.Add(bias), round0)
			lo.Store(row[x : x+4])
		}
	}
}

// convolve2DHighBDKernel is the separable 2D HBD convolve over the context of
// convolve2DHighBDGoSIMDWithIM. The horizontal pass produces the biased int32
// intermediate; the vertical pass removes the bias with rndOff.
func convolve2DHighBDKernel(ctx *convolveHighBDGoSIMDCtx) {
	nx, ny := int(ctx.tapsX), int(ctx.tapsY)
	xTaps := hbdBroadcastTaps16(&ctx.xKern)
	yTaps := hbdBroadcastTaps32(&ctx.kernel)
	zero := archsimd.BroadcastInt32x4(0)
	maxV := archsimd.BroadcastInt32x4(ctx.maxVal)
	round0, round1, bits := int(ctx.round0), int(ctx.round1), int(ctx.bits)
	xBias := archsimd.BroadcastInt32x4(ctx.xBias)
	yBias := archsimd.BroadcastInt32x4(ctx.yBias)
	rndOff := archsimd.BroadcastInt32x4(ctx.rndOff)
	width, height := int(ctx.width), int(ctx.height)
	imStr := ctx.imStr

	hbdHorizontalIM(ctx.ref, ctx.refStr, ctx.im, imStr, width, height+ny-1, &xTaps, nx, xBias, round0)

	for y := range height {
		dst := ctx.dst[y*ctx.dstStr:]
		col := ctx.im[y*imStr:]
		x := 0
		for ; x+8 <= width; x += 8 {
			lo, hi := hbdSum8Int32(col[x:], imStr, &yTaps, ny)
			lo = hbdFinish2D(lo.Add(yBias), round1, rndOff, bits, zero, maxV)
			hi = hbdFinish2D(hi.Add(yBias), round1, rndOff, bits, zero, maxV)
			hbdStorePix8(dst[x*2:], lo, hi)
		}
		if x < width {
			lo := hbdSum4Int32(col[x:], imStr, &yTaps, ny)
			lo = hbdFinish2D(lo.Add(yBias), round1, rndOff, bits, zero, maxV)
			hbdStorePix4(dst[x*2:], lo)
		}
	}
}

// compoundCopyHighBDKernel is the unfiltered HBD compound copy: each sample is
// scaled to CONV_BUF precision by 1 << shift (shift = 7 - round0) and offset.
// The wrapper guarantees width%8 == 0.
func compoundCopyHighBDKernel(ctx *compoundCopyGoSIMDCtx, shift int) {
	// The scalar reference is uint16(s*scale + roundOffset) with scale a power
	// of two; the same value is the low 16 bits of (s << shift) + roundOffset,
	// so the copy runs in uint16 lanes with no widening.
	off := archsimd.BroadcastUint16x8(uint16(ctx.roundOffset))
	width, height := int(ctx.width), int(ctx.height)
	for y := range height {
		src := ctx.ref[y*ctx.refStr:]
		out := ctx.out[y*width:]
		for x := 0; x < width; x += 8 {
			s := archsimd.LoadUint8x16(src[x*2 : x*2+16]).ReshapeToUint16s()
			s.ShiftAllLeft(uint64(shift)).Add(off).Store(out[x : x+8])
		}
	}
}

// hbdSamplesU8 reinterprets the 16 bytes of v as eight little-endian uint16
// samples in signed int16 lanes (HBD samples are at most 4095).
func hbdSamplesU8(v archsimd.Uint8x16) archsimd.Int16x8 {
	return v.ReshapeToUint16s().BitsToInt16()
}

// hbdSumRowX is hbdSum8 for the horizontal (step 2 byte) taps. It loads two
// 16-byte vectors once and forms each tap's sample vector with a constant byte
// shift, so the eight taps cost two loads instead of eight. The taps past n are
// zero, so the result equals the n-tap sum. Short slices take the generic path.
func hbdSumRowX(pix []byte, t *hbdTaps16, n int) (lo, hi archsimd.Int32x4) {
	if len(pix) < 32 {
		return hbdSum8(pix, 2, t, n)
	}
	a := archsimd.LoadUint8x16(pix[:16])
	b := archsimd.LoadUint8x16(pix[16:32])
	lo = archsimd.BroadcastInt32x4(0)
	hi = lo
	lo, hi = hbdMAC8(lo, hi, hbdSamplesU8(a), t[0])
	lo, hi = hbdMAC8(lo, hi, hbdSamplesU8(b.ConcatShiftBytesRight(a, 2)), t[1])
	lo, hi = hbdMAC8(lo, hi, hbdSamplesU8(b.ConcatShiftBytesRight(a, 4)), t[2])
	lo, hi = hbdMAC8(lo, hi, hbdSamplesU8(b.ConcatShiftBytesRight(a, 6)), t[3])
	lo, hi = hbdMAC8(lo, hi, hbdSamplesU8(b.ConcatShiftBytesRight(a, 8)), t[4])
	lo, hi = hbdMAC8(lo, hi, hbdSamplesU8(b.ConcatShiftBytesRight(a, 10)), t[5])
	lo, hi = hbdMAC8(lo, hi, hbdSamplesU8(b.ConcatShiftBytesRight(a, 12)), t[6])
	lo, hi = hbdMAC8(lo, hi, hbdSamplesU8(b.ConcatShiftBytesRight(a, 14)), t[7])
	return lo, hi
}

// compoundXHighBDKernel is the horizontal HBD compound predictor: the filter
// result is (sum + 2^(round0-1)) >> round0 plus the offset, stored as uint16
// CONV_BUF samples without clipping.
func compoundXHighBDKernel(ctx *compoundFilterGoSIMDCtx) {
	n := int(ctx.taps)
	taps := hbdBroadcastTaps16(&ctx.kernel)
	rndOff := archsimd.BroadcastInt32x4(int32(ctx.roundOffset))
	round0 := int(ctx.round0)
	width, height := int(ctx.width), int(ctx.height)
	for y := range height {
		src := ctx.ref[y*ctx.refStr:]
		out := ctx.out[y*width:]
		x := 0
		for ; x+8 <= width; x += 8 {
			lo, hi := hbdSumRowX(src[x*2:], &taps, n)
			lo = hbdRound(lo, round0).Add(rndOff)
			hi = hbdRound(hi, round0).Add(rndOff)
			hbdStoreU16x8(out[x:], lo, hi)
		}
		if x < width {
			var win [32]byte
			copy(win[:2*(n+3)], src[x*2:x*2+2*(n+3)])
			lo, _ := hbdSumRowX(win[:], &taps, n)
			lo = hbdRound(lo, round0).Add(rndOff)
			hbdStoreU16x4(out[x:], lo)
		}
	}
}

// compoundYHighBDKernel is the vertical HBD compound predictor:
// roundPowerOfTwo7(sum << (7 - round0)) plus the offset.
func compoundYHighBDKernel(ctx *compoundFilterGoSIMDCtx) {
	n := int(ctx.taps)
	taps := hbdBroadcastTaps16(&ctx.kernel)
	scale := archsimd.BroadcastInt32x4(int32(1) << (filterBits - int(ctx.round0)))
	rndOff := archsimd.BroadcastInt32x4(int32(ctx.roundOffset))
	width, height := int(ctx.width), int(ctx.height)
	for y := range height {
		src := ctx.ref[y*ctx.refStr:]
		out := ctx.out[y*width:]
		x := 0
		for ; x+8 <= width; x += 8 {
			lo, hi := hbdSum8(src[x*2:], ctx.refStr, &taps, n)
			lo = hbdRound(lo.Mul(scale), filterBits).Add(rndOff)
			hi = hbdRound(hi.Mul(scale), filterBits).Add(rndOff)
			hbdStoreU16x8(out[x:], lo, hi)
		}
		if x < width {
			var win [128]byte
			for k := range n {
				copy(win[k*16:k*16+8], src[k*ctx.refStr+x*2:k*ctx.refStr+x*2+8])
			}
			lo, _ := hbdSum8(win[:], 16, &taps, n)
			lo = hbdRound(lo.Mul(scale), filterBits).Add(rndOff)
			hbdStoreU16x4(out[x:], lo)
		}
	}
}

// compound2DHighBDKernel is the separable 2D HBD compound predictor. The output
// is roundPowerOfTwo7(yBias + vertical sum) with no further offset.

// blendCompoundAvgHighBDKernel averages two CONV_BUF predictions with distance
// weights: clip(roundPowerOfTwo(((s0*fwd + s1*bck) >> 4) - roundOffset, bits)).
// The wrapper guarantees width%4 == 0.
func blendCompoundAvgHighBDKernel(ctx *compoundBlendGoSIMDCtx, roundOffset, roundBits int) {
	fwd := archsimd.BroadcastInt32x4(int32(ctx.fwd))
	bck := archsimd.BroadcastInt32x4(int32(ctx.bck))
	roundOff := archsimd.BroadcastInt32x4(int32(roundOffset))
	zero := archsimd.BroadcastInt32x4(0)
	maxV := archsimd.BroadcastInt32x4(ctx.maxVal)
	width, height := int(ctx.width), int(ctx.height)
	for y := range height {
		s0 := ctx.src0[y*width:]
		s1 := ctx.src1[y*width:]
		dst := ctx.dst[y*ctx.dstStr:]
		x := 0
		for ; x+8 <= width; x += 8 {
			a := archsimd.LoadUint16x8(s0[x : x+8])
			b := archsimd.LoadUint16x8(s1[x : x+8])
			lo, hi := hbdBlendHalf(a, b, fwd, bck, roundOff, roundBits, zero, maxV)
			hbdStorePix8(dst[x*2:], lo, hi)
		}
		if x < width {
			var a, b [8]uint16
			copy(a[:4], s0[x:x+4])
			copy(b[:4], s1[x:x+4])
			lo, _ := hbdBlendHalf(archsimd.LoadUint16x8(a[:]), archsimd.LoadUint16x8(b[:]), fwd, bck, roundOff, roundBits, zero, maxV)
			hbdStorePix4(dst[x*2:], lo)
		}
	}
}

// hbdBlendHalf blends eight CONV_BUF pairs (lo = first four, hi = last four).
// The weighted sums stay exact in int32: CONV_BUF samples are below 2^17 and
// the distance weights are at most 16.
func hbdBlendHalf(a, b archsimd.Uint16x8, fwd, bck, roundOff archsimd.Int32x4, roundBits int, zero, maxV archsimd.Int32x4) (archsimd.Int32x4, archsimd.Int32x4) {
	a0, a1 := hbdWidenU16(a)
	b0, b1 := hbdWidenU16(b)
	lo := a0.Mul(fwd).Add(b0.Mul(bck)).ShiftAllRight(4).Sub(roundOff)
	hi := a1.Mul(fwd).Add(b1.Mul(bck)).ShiftAllRight(4).Sub(roundOff)
	return hbdClip(hbdRound(lo, roundBits), zero, maxV), hbdClip(hbdRound(hi, roundBits), zero, maxV)
}

// compound2DHighBDKernel is the separable 2D HBD compound predictor. The output
// is roundPowerOfTwo7(yBias + vertical sum) with no further offset.
func compound2DHighBDKernel(ctx *compound2DGoSIMDCtx) {
	nx, ny := int(ctx.tapsX), int(ctx.tapsY)
	xTaps := hbdBroadcastTaps16(&ctx.xKern)
	yTaps := hbdBroadcastTaps32(&ctx.kernel)
	width, height := int(ctx.width), int(ctx.height)
	xBias := archsimd.BroadcastInt32x4(int32(ctx.xBias))
	yBias := archsimd.BroadcastInt32x4(int32(ctx.yBias))

	hbdHorizontalIM(ctx.ref, ctx.refStr, ctx.im, ctx.imStr, width, height+ny-1, &xTaps, nx, xBias, int(ctx.round0))
	compoundVerticalIM(ctx.out, width, height, ctx.im, ctx.imStr, &yTaps, ny, yBias)
}

// compoundVerticalIM is the vertical stage shared by the 2D compound
// predictors: out[y][x] = uint16(roundPowerOfTwo7(yBias + sum_k t[k]*im[y+k][x])).
// im rows are imStr elements apart.
func compoundVerticalIM(out []uint16, width, height int, im []int32, imStr int, yTaps *hbdTaps32, ny int, yBias archsimd.Int32x4) {
	for y := range height {
		row := out[y*width:]
		col := im[y*imStr:]
		x := 0
		for ; x+8 <= width; x += 8 {
			lo, hi := hbdSum8Int32(col[x:], imStr, yTaps, ny)
			lo = hbdRound(lo.Add(yBias), filterBits)
			hi = hbdRound(hi.Add(yBias), filterBits)
			hbdStoreU16x8(row[x:], lo, hi)
		}
		if x < width {
			lo := hbdSum4Int32(col[x:], imStr, yTaps, ny)
			lo = hbdRound(lo.Add(yBias), filterBits)
			hbdStoreU16x4(row[x:], lo)
		}
	}
}

// hbdStoreU8x8 writes eight clipped 8-bit samples (lo and hi already clipped to
// [0, 255]) at the start of dst.
func hbdStoreU8x8(dst []byte, lo, hi archsimd.Int32x4) {
	var t [16]byte
	narrowU8x8(lo, hi).Store(t[:])
	copy(dst[:8], t[:8])
}

// hbdStoreU8x4 writes the four low lanes of lo as 8-bit samples at the start of
// dst.
func hbdStoreU8x4(dst []byte, lo archsimd.Int32x4) {
	var t [16]byte
	narrowU8x8(lo, lo).Store(t[:])
	copy(dst[:4], t[:4])
}
