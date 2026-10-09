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
// Sample vectors are loaded from the little-endian byte planes and reshaped to
// uint16 lanes; intermediates are []int32 and compound buffers are []uint16.
// Sliding vertical loops widen each source row once and reuse it for the next
// output rows. The scalar tails stage exact-width windows so a vector load
// cannot cross the slice end.
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
	"encoding/binary"
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

// hbdLoad4Int32 reads exactly four uint16 samples, including at the end of a
// minimum-size row. The 64-bit broadcast supplies the low four lanes for one
// widening instruction without staging a 16-byte temporary window.
func hbdLoad4Int32(pix []byte) archsimd.Int32x4 {
	return archsimd.BroadcastUint64x2(binary.LittleEndian.Uint64(pix[:8])).ReshapeToUint16s().ExtendLo4ToUint32().BitsToInt32()
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

// hbdRoundShift uses a rounding bias and signed shift amount prepared before
// the pixel loop. A negative amount in Shift is an arithmetic right shift.
func hbdRoundShift(v, bias, shift archsimd.Int32x4, n int) archsimd.Int32x4 {
	return hbdShiftRight(v.Add(bias), shift, n)
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
	round0Bias := archsimd.BroadcastInt32x4(int32(1) << (round0 - 1))
	round0Shift := archsimd.BroadcastInt32x4(int32(-round0))
	bitsBias := archsimd.BroadcastInt32x4(int32(1) << (bits - 1))
	bitsShift := archsimd.BroadcastInt32x4(int32(-bits))
	width, height := int(ctx.width), int(ctx.height)
	for y := range height {
		src := ctx.ref[y*ctx.refStr:]
		dst := ctx.dst[y*ctx.dstStr:]
		x := 0
		for ; x+8 <= width; x += 8 {
			var lo, hi archsimd.Int32x4
			if x*2+31 < len(src) {
				a := archsimd.LoadUint8x16(src[x*2 : x*2+16])
				b := archsimd.LoadUint8x16(src[x*2+16 : x*2+32])
				lo = zero
				hi = zero
				lo, hi = hbdMAC8(lo, hi, hbdSamplesU8(a), taps[0])
				lo, hi = hbdMAC8(lo, hi, hbdSamplesU8(b.ConcatShiftBytesRight(a, 2)), taps[1])
				if n > 2 {
					lo, hi = hbdMAC8(lo, hi, hbdSamplesU8(b.ConcatShiftBytesRight(a, 4)), taps[2])
					lo, hi = hbdMAC8(lo, hi, hbdSamplesU8(b.ConcatShiftBytesRight(a, 6)), taps[3])
				}
				if n > 4 {
					lo, hi = hbdMAC8(lo, hi, hbdSamplesU8(b.ConcatShiftBytesRight(a, 8)), taps[4])
					lo, hi = hbdMAC8(lo, hi, hbdSamplesU8(b.ConcatShiftBytesRight(a, 10)), taps[5])
				}
				if n > 6 {
					lo, hi = hbdMAC8(lo, hi, hbdSamplesU8(b.ConcatShiftBytesRight(a, 12)), taps[6])
					lo, hi = hbdMAC8(lo, hi, hbdSamplesU8(b.ConcatShiftBytesRight(a, 14)), taps[7])
				}
			} else {
				lo, hi = hbdSumRowX(src[x*2:], &taps, n)
			}
			lo = hbdClip(hbdRoundShift(hbdRoundShift(lo, round0Bias, round0Shift, round0), bitsBias, bitsShift, bits), zero, maxV)
			hi = hbdClip(hbdRoundShift(hbdRoundShift(hi, round0Bias, round0Shift, round0), bitsBias, bitsShift, bits), zero, maxV)
			hbdStorePix8(dst[x*2:], lo, hi)
		}
		if x < width {
			var win [32]byte
			copy(win[:2*(n+3)], src[x*2:x*2+2*(n+3)])
			lo, _ := hbdSumRowX(win[:], &taps, n)
			lo = hbdClip(hbdRoundShift(hbdRoundShift(lo, round0Bias, round0Shift, round0), bitsBias, bitsShift, bits), zero, maxV)
			hbdStorePix4(dst[x*2:], lo)
		}
	}
}

// convolveYHighBDKernel is the 1D vertical HBD convolve over the context of
// convolveYHighBDGoSIMD: one FILTER_BITS rounding, then clip.
func convolveYHighBDKernel(ctx *convolveHighBDGoSIMDCtx) {
	n := int(ctx.tapsY)
	zero := archsimd.BroadcastInt32x4(0)
	maxV := archsimd.BroadcastInt32x4(ctx.maxVal)
	round0 := int(ctx.round0)
	roundBias := archsimd.BroadcastInt32x4(int32(1) << (round0 - 1))
	roundShift := archsimd.BroadcastInt32x4(int32(-round0))
	width, height := int(ctx.width), int(ctx.height)
	if width == 4 && (n == 4 || n == 6) {
		convolveYHighBDW4Sliding(ctx, n, roundBias, roundShift, zero, maxV)
		return
	}
	if n == 8 && width >= 8 && width%8 == 0 {
		convolveYHighBD8TapSliding(ctx, roundBias, roundShift, zero, maxV)
		return
	}
	if n == 6 && width >= 8 && width%8 == 0 {
		// Keep the six input rows in registers as the output advances. Only
		// the newly entering row needs to be loaded and widened each time.
		ref, dst := ctx.ref, ctx.dst
		refStr, dstStr := ctx.refStr, ctx.dstStr
		_ = ref[(height+n-2)*refStr+width*2-1]
		_ = dst[(height-1)*dstStr+width*2-1]
		coeff := hbdBroadcastTaps32(&ctx.kernel)
		c0, c1, c2, c3, c4, c5 := coeff[0], coeff[1], coeff[2], coeff[3], coeff[4], coeff[5]
		for x := 0; x < width; x += 8 {
			p := x * 2
			r0l, r0h := hbdWidenU16(archsimd.LoadUint8x16(ref[p : p+16]).ReshapeToUint16s())
			r1l, r1h := hbdWidenU16(archsimd.LoadUint8x16(ref[p+refStr : p+refStr+16]).ReshapeToUint16s())
			r2l, r2h := hbdWidenU16(archsimd.LoadUint8x16(ref[p+2*refStr : p+2*refStr+16]).ReshapeToUint16s())
			r3l, r3h := hbdWidenU16(archsimd.LoadUint8x16(ref[p+3*refStr : p+3*refStr+16]).ReshapeToUint16s())
			r4l, r4h := hbdWidenU16(archsimd.LoadUint8x16(ref[p+4*refStr : p+4*refStr+16]).ReshapeToUint16s())
			r5l, r5h := hbdWidenU16(archsimd.LoadUint8x16(ref[p+5*refStr : p+5*refStr+16]).ReshapeToUint16s())
			for y := 0; y < height; y++ {
				lo := hbdMulAdd32(r0l, c0, roundBias)
				hi := hbdMulAdd32(r0h, c0, roundBias)
				lo = hbdMulAdd32(r1l, c1, lo)
				hi = hbdMulAdd32(r1h, c1, hi)
				lo = hbdMulAdd32(r2l, c2, lo)
				hi = hbdMulAdd32(r2h, c2, hi)
				lo = hbdMulAdd32(r3l, c3, lo)
				hi = hbdMulAdd32(r3h, c3, hi)
				lo = hbdMulAdd32(r4l, c4, lo)
				hi = hbdMulAdd32(r4h, c4, hi)
				lo = hbdMulAdd32(r5l, c5, lo)
				hi = hbdMulAdd32(r5h, c5, hi)
				lo = hbdClip(hbdShiftRight(lo, roundShift, round0), zero, maxV)
				hi = hbdClip(hbdShiftRight(hi, roundShift, round0), zero, maxV)
				hbdStorePix8(dst[y*dstStr+x*2:], lo, hi)
				if y+1 < height {
					r0l, r0h = r1l, r1h
					r1l, r1h = r2l, r2h
					r2l, r2h = r3l, r3h
					r3l, r3h = r4l, r4h
					r4l, r4h = r5l, r5h
					p += refStr
					r5l, r5h = hbdWidenU16(archsimd.LoadUint8x16(ref[p+5*refStr : p+5*refStr+16]).ReshapeToUint16s())
				}
			}
		}
		return
	}
	taps := hbdBroadcastTaps16(&ctx.kernel)
	for y := range height {
		src := ctx.ref[y*ctx.refStr:]
		dst := ctx.dst[y*ctx.dstStr:]
		x := 0
		for ; x+8 <= width; x += 8 {
			lo, hi := hbdSum8(src[x*2:], ctx.refStr, &taps, n)
			lo = hbdClip(hbdRoundShift(lo, roundBias, roundShift, round0), zero, maxV)
			hi = hbdClip(hbdRoundShift(hi, roundBias, roundShift, round0), zero, maxV)
			hbdStorePix8(dst[x*2:], lo, hi)
		}
		if x < width {
			var win [128]byte
			for k := range n {
				copy(win[k*16:k*16+8], src[k*ctx.refStr+x*2:k*ctx.refStr+x*2+8])
			}
			lo, _ := hbdSum8(win[:], 16, &taps, n)
			lo = hbdClip(hbdRoundShift(lo, roundBias, roundShift, round0), zero, maxV)
			hbdStorePix4(dst[x*2:], lo)
		}
	}
}

// convolveYHighBD8TapSliding is the full eight-tap counterpart of the
// register-resident six-tap path above. Sharp filters use every tap.
func convolveYHighBD8TapSliding(ctx *convolveHighBDGoSIMDCtx, roundBias, roundShift, zero, maxV archsimd.Int32x4) {
	width, height := ctx.width, ctx.height
	ref, dst := ctx.ref, ctx.dst
	refStr, dstStr := ctx.refStr, ctx.dstStr
	_ = ref[(height+6)*refStr+width*2-1]
	_ = dst[(height-1)*dstStr+width*2-1]
	coeff := hbdBroadcastTaps32(&ctx.kernel)
	c0, c1, c2, c3 := coeff[0], coeff[1], coeff[2], coeff[3]
	c4, c5, c6, c7 := coeff[4], coeff[5], coeff[6], coeff[7]
	for x := 0; x < width; x += 8 {
		p := x * 2
		r0l, r0h := hbdWidenU16(archsimd.LoadUint8x16(ref[p : p+16]).ReshapeToUint16s())
		r1l, r1h := hbdWidenU16(archsimd.LoadUint8x16(ref[p+refStr : p+refStr+16]).ReshapeToUint16s())
		r2l, r2h := hbdWidenU16(archsimd.LoadUint8x16(ref[p+2*refStr : p+2*refStr+16]).ReshapeToUint16s())
		r3l, r3h := hbdWidenU16(archsimd.LoadUint8x16(ref[p+3*refStr : p+3*refStr+16]).ReshapeToUint16s())
		r4l, r4h := hbdWidenU16(archsimd.LoadUint8x16(ref[p+4*refStr : p+4*refStr+16]).ReshapeToUint16s())
		r5l, r5h := hbdWidenU16(archsimd.LoadUint8x16(ref[p+5*refStr : p+5*refStr+16]).ReshapeToUint16s())
		r6l, r6h := hbdWidenU16(archsimd.LoadUint8x16(ref[p+6*refStr : p+6*refStr+16]).ReshapeToUint16s())
		r7l, r7h := hbdWidenU16(archsimd.LoadUint8x16(ref[p+7*refStr : p+7*refStr+16]).ReshapeToUint16s())
		for y := 0; y < height; y++ {
			lo := hbdMulAdd32(r0l, c0, roundBias)
			hi := hbdMulAdd32(r0h, c0, roundBias)
			lo = hbdMulAdd32(r1l, c1, lo)
			hi = hbdMulAdd32(r1h, c1, hi)
			lo = hbdMulAdd32(r2l, c2, lo)
			hi = hbdMulAdd32(r2h, c2, hi)
			lo = hbdMulAdd32(r3l, c3, lo)
			hi = hbdMulAdd32(r3h, c3, hi)
			lo = hbdMulAdd32(r4l, c4, lo)
			hi = hbdMulAdd32(r4h, c4, hi)
			lo = hbdMulAdd32(r5l, c5, lo)
			hi = hbdMulAdd32(r5h, c5, hi)
			lo = hbdMulAdd32(r6l, c6, lo)
			hi = hbdMulAdd32(r6h, c6, hi)
			lo = hbdMulAdd32(r7l, c7, lo)
			hi = hbdMulAdd32(r7h, c7, hi)
			lo = hbdClip(hbdShiftRight(lo, roundShift, int(ctx.round0)), zero, maxV)
			hi = hbdClip(hbdShiftRight(hi, roundShift, int(ctx.round0)), zero, maxV)
			hbdStorePix8(dst[y*dstStr+x*2:], lo, hi)
			if y+1 < height {
				r0l, r0h = r1l, r1h
				r1l, r1h = r2l, r2h
				r2l, r2h = r3l, r3h
				r3l, r3h = r4l, r4h
				r4l, r4h = r5l, r5h
				r5l, r5h = r6l, r6h
				r6l, r6h = r7l, r7h
				p += refStr
				r7l, r7h = hbdWidenU16(archsimd.LoadUint8x16(ref[p+7*refStr : p+7*refStr+16]).ReshapeToUint16s())
			}
		}
	}
}

// convolveYHighBDW4Sliding dispatches the narrow, exact-window row reuse.
func convolveYHighBDW4Sliding(ctx *convolveHighBDGoSIMDCtx, n int, bias, shift, zero, maxV archsimd.Int32x4) {
	if n == 4 {
		convolveYHighBDW4Sliding4(ctx, bias, shift, zero, maxV)
	} else {
		convolveYHighBDW4Sliding6(ctx, bias, shift, zero, maxV)
	}
}

func convolveYHighBDW4Sliding4(ctx *convolveHighBDGoSIMDCtx, bias, shift, zero, maxV archsimd.Int32x4) {
	height, refStr, dstStr := ctx.height, ctx.refStr, ctx.dstStr
	ref, dst := ctx.ref, ctx.dst
	_ = ref[(height+2)*refStr+7]
	_ = dst[(height-1)*dstStr+7]
	c := hbdBroadcastTaps32(&ctx.kernel)
	c0, c1, c2, c3 := c[0], c[1], c[2], c[3]
	r0, r1, r2, r3 := hbdLoad4Int32(ref), hbdLoad4Int32(ref[1*refStr:]), hbdLoad4Int32(ref[2*refStr:]), hbdLoad4Int32(ref[3*refStr:])
	p := 0
	for y := 0; y < height; y++ {
		sum := hbdMulAdd32(r0, c0, bias)
		sum = hbdMulAdd32(r1, c1, sum)
		sum = hbdMulAdd32(r2, c2, sum)
		sum = hbdMulAdd32(r3, c3, sum)
		sum = hbdClip(hbdShiftRight(sum, shift, int(ctx.round0)), zero, maxV)
		hbdStorePix4(dst[y*dstStr:], sum)
		if y+1 < height {
			r0, r1, r2 = r1, r2, r3
			p += refStr
			r3 = hbdLoad4Int32(ref[p+3*refStr:])
		}
	}
}

func convolveYHighBDW4Sliding6(ctx *convolveHighBDGoSIMDCtx, bias, shift, zero, maxV archsimd.Int32x4) {
	height, refStr, dstStr := ctx.height, ctx.refStr, ctx.dstStr
	ref, dst := ctx.ref, ctx.dst
	_ = ref[(height+4)*refStr+7]
	_ = dst[(height-1)*dstStr+7]
	c := hbdBroadcastTaps32(&ctx.kernel)
	c0, c1, c2, c3, c4, c5 := c[0], c[1], c[2], c[3], c[4], c[5]
	r0, r1, r2, r3, r4, r5 := hbdLoad4Int32(ref), hbdLoad4Int32(ref[1*refStr:]), hbdLoad4Int32(ref[2*refStr:]), hbdLoad4Int32(ref[3*refStr:]), hbdLoad4Int32(ref[4*refStr:]), hbdLoad4Int32(ref[5*refStr:])
	p := 0
	for y := 0; y < height; y++ {
		sum := hbdMulAdd32(r0, c0, bias)
		sum = hbdMulAdd32(r1, c1, sum)
		sum = hbdMulAdd32(r2, c2, sum)
		sum = hbdMulAdd32(r3, c3, sum)
		sum = hbdMulAdd32(r4, c4, sum)
		sum = hbdMulAdd32(r5, c5, sum)
		sum = hbdClip(hbdShiftRight(sum, shift, int(ctx.round0)), zero, maxV)
		hbdStorePix4(dst[y*dstStr:], sum)
		if y+1 < height {
			r0, r1, r2, r3, r4 = r1, r2, r3, r4, r5
			p += refStr
			r5 = hbdLoad4Int32(ref[p+5*refStr:])
		}
	}
}

// hbdHorizontalIM computes the biased horizontal pass of the separable 2D
// convolves: rows reference rows, each becoming round0(bias + sum) int32
// intermediate samples at row stride imStr elements.
func hbdHorizontalIM(ref []byte, refStr int, im []int32, imStr int, width, rows int, taps *hbdTaps16, n int, bias archsimd.Int32x4, round0 int) {
	roundBias := bias.Add(archsimd.BroadcastInt32x4(int32(1) << (round0 - 1)))
	roundShift := archsimd.BroadcastInt32x4(int32(-round0))
	for y := range rows {
		src := ref[y*refStr:]
		row := im[y*imStr:]
		x := 0
		for ; x+8 <= width; x += 8 {
			var lo, hi archsimd.Int32x4
			if x*2+31 < len(src) {
				a := archsimd.LoadUint8x16(src[x*2 : x*2+16])
				b := archsimd.LoadUint8x16(src[x*2+16 : x*2+32])
				lo = archsimd.BroadcastInt32x4(0)
				hi = lo
				lo, hi = hbdMAC8(lo, hi, hbdSamplesU8(a), taps[0])
				lo, hi = hbdMAC8(lo, hi, hbdSamplesU8(b.ConcatShiftBytesRight(a, 2)), taps[1])
				if n > 2 {
					lo, hi = hbdMAC8(lo, hi, hbdSamplesU8(b.ConcatShiftBytesRight(a, 4)), taps[2])
					lo, hi = hbdMAC8(lo, hi, hbdSamplesU8(b.ConcatShiftBytesRight(a, 6)), taps[3])
				}
				if n > 4 {
					lo, hi = hbdMAC8(lo, hi, hbdSamplesU8(b.ConcatShiftBytesRight(a, 8)), taps[4])
					lo, hi = hbdMAC8(lo, hi, hbdSamplesU8(b.ConcatShiftBytesRight(a, 10)), taps[5])
				}
				if n > 6 {
					lo, hi = hbdMAC8(lo, hi, hbdSamplesU8(b.ConcatShiftBytesRight(a, 12)), taps[6])
					lo, hi = hbdMAC8(lo, hi, hbdSamplesU8(b.ConcatShiftBytesRight(a, 14)), taps[7])
				}
			} else {
				lo, hi = hbdSumRowX(src[x*2:], taps, n)
			}
			lo = hbdShiftRight(lo.Add(roundBias), roundShift, round0)
			hi = hbdShiftRight(hi.Add(roundBias), roundShift, round0)
			lo.Store(row[x : x+4])
			hi.Store(row[x+4 : x+8])
		}
		if x < width {
			var win [32]byte
			copy(win[:2*(n+3)], src[x*2:x*2+2*(n+3)])
			lo, _ := hbdSumRowX(win[:], taps, n)
			lo = hbdShiftRight(lo.Add(roundBias), roundShift, round0)
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
	round0, round1 := int(ctx.round0), int(ctx.round1)
	xBias := archsimd.BroadcastInt32x4(ctx.xBias)
	yBias := archsimd.BroadcastInt32x4(ctx.yBias)
	rndOff := archsimd.BroadcastInt32x4(ctx.rndOff)
	verticalBias := yBias.Add(archsimd.BroadcastInt32x4(int32(1) << (round1 - 1)))
	verticalShift := archsimd.BroadcastInt32x4(int32(-round1))
	width, height := int(ctx.width), int(ctx.height)
	imStr := ctx.imStr

	hbdHorizontalIM(ctx.ref, ctx.refStr, ctx.im, imStr, width, height+ny-1, &xTaps, nx, xBias, round0)

	if ny == 6 && width >= 8 && width%8 == 0 {
		// Unroll the six intermediate-row MACs and fold the rounding bias
		// into the first accumulator, keeping both four-lane chains live.
		im, dst := ctx.im, ctx.dst
		dstStr := ctx.dstStr
		c0, c1, c2, c3, c4, c5 := yTaps[0], yTaps[1], yTaps[2], yTaps[3], yTaps[4], yTaps[5]
		for y := 0; y < height; y++ {
			for x := 0; x < width; x += 8 {
				base := y*imStr + x
				lo := hbdMulAdd32(archsimd.LoadInt32x4(im[base:base+4]), c0, verticalBias)
				hi := hbdMulAdd32(archsimd.LoadInt32x4(im[base+4:base+4+4]), c0, verticalBias)
				lo = hbdMulAdd32(archsimd.LoadInt32x4(im[base+1*imStr:base+1*imStr+4]), c1, lo)
				hi = hbdMulAdd32(archsimd.LoadInt32x4(im[base+1*imStr+4:base+1*imStr+4+4]), c1, hi)
				lo = hbdMulAdd32(archsimd.LoadInt32x4(im[base+2*imStr:base+2*imStr+4]), c2, lo)
				hi = hbdMulAdd32(archsimd.LoadInt32x4(im[base+2*imStr+4:base+2*imStr+4+4]), c2, hi)
				lo = hbdMulAdd32(archsimd.LoadInt32x4(im[base+3*imStr:base+3*imStr+4]), c3, lo)
				hi = hbdMulAdd32(archsimd.LoadInt32x4(im[base+3*imStr+4:base+3*imStr+4+4]), c3, hi)
				lo = hbdMulAdd32(archsimd.LoadInt32x4(im[base+4*imStr:base+4*imStr+4]), c4, lo)
				hi = hbdMulAdd32(archsimd.LoadInt32x4(im[base+4*imStr+4:base+4*imStr+4+4]), c4, hi)
				lo = hbdMulAdd32(archsimd.LoadInt32x4(im[base+5*imStr:base+5*imStr+4]), c5, lo)
				hi = hbdMulAdd32(archsimd.LoadInt32x4(im[base+5*imStr+4:base+5*imStr+4+4]), c5, hi)
				lo = hbdClip(hbdShiftRight(lo, verticalShift, round1).Sub(rndOff), zero, maxV)
				hi = hbdClip(hbdShiftRight(hi, verticalShift, round1).Sub(rndOff), zero, maxV)
				hbdStorePix8(dst[y*dstStr+x*2:], lo, hi)
			}
		}
		return
	}
	if ny == 8 && width >= 8 && width%8 == 0 {
		im, dst := ctx.im, ctx.dst
		dstStr := ctx.dstStr
		c0, c1, c2, c3 := yTaps[0], yTaps[1], yTaps[2], yTaps[3]
		c4, c5, c6, c7 := yTaps[4], yTaps[5], yTaps[6], yTaps[7]
		for y := 0; y < height; y++ {
			for x := 0; x < width; x += 8 {
				base := y*imStr + x
				lo := hbdMulAdd32(archsimd.LoadInt32x4(im[base:base+4]), c0, verticalBias)
				hi := hbdMulAdd32(archsimd.LoadInt32x4(im[base+4:base+4+4]), c0, verticalBias)
				lo = hbdMulAdd32(archsimd.LoadInt32x4(im[base+1*imStr:base+1*imStr+4]), c1, lo)
				hi = hbdMulAdd32(archsimd.LoadInt32x4(im[base+1*imStr+4:base+1*imStr+4+4]), c1, hi)
				lo = hbdMulAdd32(archsimd.LoadInt32x4(im[base+2*imStr:base+2*imStr+4]), c2, lo)
				hi = hbdMulAdd32(archsimd.LoadInt32x4(im[base+2*imStr+4:base+2*imStr+4+4]), c2, hi)
				lo = hbdMulAdd32(archsimd.LoadInt32x4(im[base+3*imStr:base+3*imStr+4]), c3, lo)
				hi = hbdMulAdd32(archsimd.LoadInt32x4(im[base+3*imStr+4:base+3*imStr+4+4]), c3, hi)
				lo = hbdMulAdd32(archsimd.LoadInt32x4(im[base+4*imStr:base+4*imStr+4]), c4, lo)
				hi = hbdMulAdd32(archsimd.LoadInt32x4(im[base+4*imStr+4:base+4*imStr+4+4]), c4, hi)
				lo = hbdMulAdd32(archsimd.LoadInt32x4(im[base+5*imStr:base+5*imStr+4]), c5, lo)
				hi = hbdMulAdd32(archsimd.LoadInt32x4(im[base+5*imStr+4:base+5*imStr+4+4]), c5, hi)
				lo = hbdMulAdd32(archsimd.LoadInt32x4(im[base+6*imStr:base+6*imStr+4]), c6, lo)
				hi = hbdMulAdd32(archsimd.LoadInt32x4(im[base+6*imStr+4:base+6*imStr+4+4]), c6, hi)
				lo = hbdMulAdd32(archsimd.LoadInt32x4(im[base+7*imStr:base+7*imStr+4]), c7, lo)
				hi = hbdMulAdd32(archsimd.LoadInt32x4(im[base+7*imStr+4:base+7*imStr+4+4]), c7, hi)
				lo = hbdClip(hbdShiftRight(lo, verticalShift, round1).Sub(rndOff), zero, maxV)
				hi = hbdClip(hbdShiftRight(hi, verticalShift, round1).Sub(rndOff), zero, maxV)
				hbdStorePix8(dst[y*dstStr+x*2:], lo, hi)
			}
		}
		return
	}
	for y := range height {
		dst := ctx.dst[y*ctx.dstStr:]
		col := ctx.im[y*imStr:]
		x := 0
		for ; x+8 <= width; x += 8 {
			lo, hi := hbdSum8Int32(col[x:], imStr, &yTaps, ny)
			lo = hbdClip(hbdShiftRight(lo.Add(verticalBias), verticalShift, round1).Sub(rndOff), zero, maxV)
			hi = hbdClip(hbdShiftRight(hi.Add(verticalBias), verticalShift, round1).Sub(rndOff), zero, maxV)
			hbdStorePix8(dst[x*2:], lo, hi)
		}
		if x < width {
			lo := hbdSum4Int32(col[x:], imStr, &yTaps, ny)
			lo = hbdClip(hbdShiftRight(lo.Add(verticalBias), verticalShift, round1).Sub(rndOff), zero, maxV)
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
	if n <= 2 {
		return lo, hi
	}
	lo, hi = hbdMAC8(lo, hi, hbdSamplesU8(b.ConcatShiftBytesRight(a, 4)), t[2])
	lo, hi = hbdMAC8(lo, hi, hbdSamplesU8(b.ConcatShiftBytesRight(a, 6)), t[3])
	if n <= 4 {
		return lo, hi
	}
	lo, hi = hbdMAC8(lo, hi, hbdSamplesU8(b.ConcatShiftBytesRight(a, 8)), t[4])
	lo, hi = hbdMAC8(lo, hi, hbdSamplesU8(b.ConcatShiftBytesRight(a, 10)), t[5])
	if n <= 6 {
		return lo, hi
	}
	lo, hi = hbdMAC8(lo, hi, hbdSamplesU8(b.ConcatShiftBytesRight(a, 12)), t[6])
	lo, hi = hbdMAC8(lo, hi, hbdSamplesU8(b.ConcatShiftBytesRight(a, 14)), t[7])
	return lo, hi
}

// compoundXHighBDKernel is the horizontal HBD compound predictor: the filter
// result is (sum + 2^(round0-1)) >> round0 plus the offset, stored as uint16
// CONV_BUF samples without clipping.
func compoundXHighBDKernel(ctx *compoundFilterGoSIMDCtx) {
	n := int(ctx.taps)
	if ctx.width == 4 && n == 4 {
		compoundXHighBD4TapW4(ctx)
		return
	}
	taps := hbdBroadcastTaps16(&ctx.kernel)
	rndOff := archsimd.BroadcastInt32x4(int32(ctx.roundOffset))
	zero := archsimd.BroadcastInt32x4(0)
	roundBias := archsimd.BroadcastInt32x4(int32(1) << (int(ctx.round0) - 1))
	roundShift := archsimd.BroadcastInt32x4(int32(-ctx.round0))
	round0 := int(ctx.round0)
	width, height := int(ctx.width), int(ctx.height)
	for y := range height {
		src := ctx.ref[y*ctx.refStr:]
		out := ctx.out[y*width:]
		x := 0
		for ; x+8 <= width; x += 8 {
			var lo, hi archsimd.Int32x4
			if x*2+31 < len(src) {
				a := archsimd.LoadUint8x16(src[x*2 : x*2+16])
				b := archsimd.LoadUint8x16(src[x*2+16 : x*2+32])
				lo = zero
				hi = zero
				lo, hi = hbdMAC8(lo, hi, hbdSamplesU8(a), taps[0])
				lo, hi = hbdMAC8(lo, hi, hbdSamplesU8(b.ConcatShiftBytesRight(a, 2)), taps[1])
				if n > 2 {
					lo, hi = hbdMAC8(lo, hi, hbdSamplesU8(b.ConcatShiftBytesRight(a, 4)), taps[2])
					lo, hi = hbdMAC8(lo, hi, hbdSamplesU8(b.ConcatShiftBytesRight(a, 6)), taps[3])
				}
				if n > 4 {
					lo, hi = hbdMAC8(lo, hi, hbdSamplesU8(b.ConcatShiftBytesRight(a, 8)), taps[4])
					lo, hi = hbdMAC8(lo, hi, hbdSamplesU8(b.ConcatShiftBytesRight(a, 10)), taps[5])
				}
				if n > 6 {
					lo, hi = hbdMAC8(lo, hi, hbdSamplesU8(b.ConcatShiftBytesRight(a, 12)), taps[6])
					lo, hi = hbdMAC8(lo, hi, hbdSamplesU8(b.ConcatShiftBytesRight(a, 14)), taps[7])
				}
			} else {
				lo, hi = hbdSumRowX(src[x*2:], &taps, n)
			}
			lo = hbdRoundShift(lo, roundBias, roundShift, round0).Add(rndOff)
			hi = hbdRoundShift(hi, roundBias, roundShift, round0).Add(rndOff)
			hbdStoreU16x8(out[x:], lo, hi)
		}
		if x < width {
			var win [32]byte
			copy(win[:2*(n+3)], src[x*2:x*2+2*(n+3)])
			lo, _ := hbdSumRowX(win[:], &taps, n)
			lo = hbdRoundShift(lo, roundBias, roundShift, round0).Add(rndOff)
			hbdStoreU16x4(out[x:], lo)
		}
	}
}

// compoundXHighBD4TapW4 computes only the four output lanes. The common path
// can read one vector from resident planes; the exact-minimum slice stages its
// fourteen input bytes before the same vector operations.
func compoundXHighBD4TapW4(ctx *compoundFilterGoSIMDCtx) {
	ref, out := ctx.ref, ctx.out
	height, refStr := ctx.height, ctx.refStr
	_ = ref[(height-1)*refStr+13]
	_ = out[height*4-1]
	t := hbdBroadcastTaps16(&ctx.kernel)
	c0, c1, c2, c3 := t[0], t[1], t[2], t[3]
	round0 := int(ctx.round0)
	bias := archsimd.BroadcastInt32x4(int32(1) << (round0 - 1))
	shift := archsimd.BroadcastInt32x4(int32(-round0))
	off := archsimd.BroadcastInt32x4(ctx.roundOffset)
	for y := 0; y < height; y++ {
		src := ref[y*refStr:]
		var a archsimd.Uint8x16
		if len(src) >= 16 {
			a = archsimd.LoadUint8x16(src[:16])
		} else {
			var win [16]byte
			copy(win[:14], src[:14])
			a = archsimd.LoadUint8x16(win[:])
		}
		sum := hbdMAC4(bias, hbdSamplesU8(a), c0)
		sum = hbdMAC4(sum, hbdSamplesU8(a.ConcatShiftBytesRight(a, 2)), c1)
		sum = hbdMAC4(sum, hbdSamplesU8(a.ConcatShiftBytesRight(a, 4)), c2)
		sum = hbdMAC4(sum, hbdSamplesU8(a.ConcatShiftBytesRight(a, 6)), c3)
		hbdStoreU16x4(out[y*4:], hbdShiftRight(sum, shift, round0).Add(off))
	}
}

// compoundYHighBDKernel is the vertical HBD compound predictor:
// roundPowerOfTwo7(sum << (7 - round0)) plus the offset.
func compoundYHighBDKernel(ctx *compoundFilterGoSIMDCtx) {
	n := int(ctx.taps)
	if ctx.taps == 4 && ctx.width == 4 {
		compoundYHighBD4TapW4Sliding(ctx)
		return
	}
	if ctx.taps == 6 && ctx.width == 4 {
		compoundYHighBD6TapW4Sliding(ctx)
		return
	}
	if ctx.taps == 6 && ctx.width >= 8 && ctx.width%8 == 0 {
		compoundYHighBD6TapSliding(ctx)
		return
	}
	taps := hbdBroadcastTaps16(&ctx.kernel)
	round0 := int(ctx.round0)
	roundBias := archsimd.BroadcastInt32x4(int32(1) << (round0 - 1))
	roundShift := archsimd.BroadcastInt32x4(int32(-round0))
	rndOff := archsimd.BroadcastInt32x4(int32(ctx.roundOffset))
	width, height := int(ctx.width), int(ctx.height)
	for y := range height {
		src := ctx.ref[y*ctx.refStr:]
		out := ctx.out[y*width:]
		x := 0
		for ; x+8 <= width; x += 8 {
			lo, hi := hbdSum8(src[x*2:], ctx.refStr, &taps, n)
			lo = hbdRoundShift(lo, roundBias, roundShift, round0).Add(rndOff)
			hi = hbdRoundShift(hi, roundBias, roundShift, round0).Add(rndOff)
			hbdStoreU16x8(out[x:], lo, hi)
		}
		if x < width {
			var win [128]byte
			for k := range n {
				copy(win[k*16:k*16+8], src[k*ctx.refStr+x*2:k*ctx.refStr+x*2+8])
			}
			lo, _ := hbdSum8(win[:], 16, &taps, n)
			lo = hbdRoundShift(lo, roundBias, roundShift, round0).Add(rndOff)
			hbdStoreU16x4(out[x:], lo)
		}
	}
}

func compoundYHighBD4TapW4Sliding(ctx *compoundFilterGoSIMDCtx) {
	height, refStr := ctx.height, ctx.refStr
	ref, out := ctx.ref, ctx.out
	_ = ref[(height+2)*refStr+7]
	_ = out[height*4-1]
	c := hbdBroadcastTaps32(&ctx.kernel)
	c0, c1, c2, c3 := c[0], c[1], c[2], c[3]
	round0 := int(ctx.round0)
	bias := archsimd.BroadcastInt32x4(int32(1) << (round0 - 1))
	shift := archsimd.BroadcastInt32x4(int32(-round0))
	off := archsimd.BroadcastInt32x4(ctx.roundOffset)
	r0 := hbdLoad4Int32(ref)
	r1 := hbdLoad4Int32(ref[refStr:])
	r2 := hbdLoad4Int32(ref[2*refStr:])
	r3 := hbdLoad4Int32(ref[3*refStr:])
	p := 0
	for y := 0; y < height; y++ {
		sum := hbdMulAdd32(r0, c0, bias)
		sum = hbdMulAdd32(r1, c1, sum)
		sum = hbdMulAdd32(r2, c2, sum)
		sum = hbdMulAdd32(r3, c3, sum)
		hbdStoreU16x4(out[y*4:], hbdShiftRight(sum, shift, round0).Add(off))
		if y+1 < height {
			r0, r1, r2 = r1, r2, r3
			p += refStr
			r3 = hbdLoad4Int32(ref[p+3*refStr:])
		}
	}
}

// compoundYHighBD6TapW4Sliding loads exactly four samples per input row and
// retains the widened rows across outputs. Narrow blocks otherwise spent most
// of their time copying each row into a 16-byte staging window repeatedly.
func compoundYHighBD6TapW4Sliding(ctx *compoundFilterGoSIMDCtx) {
	height, refStr := ctx.height, ctx.refStr
	ref, out := ctx.ref, ctx.out
	_ = ref[(height+4)*refStr+7]
	_ = out[height*4-1]
	c := hbdBroadcastTaps32(&ctx.kernel)
	c0, c1, c2, c3, c4, c5 := c[0], c[1], c[2], c[3], c[4], c[5]
	round0 := int(ctx.round0)
	bias := archsimd.BroadcastInt32x4(int32(1) << (round0 - 1))
	shift := archsimd.BroadcastInt32x4(int32(-round0))
	off := archsimd.BroadcastInt32x4(ctx.roundOffset)
	r0 := hbdLoad4Int32(ref)
	r1 := hbdLoad4Int32(ref[refStr:])
	r2 := hbdLoad4Int32(ref[2*refStr:])
	r3 := hbdLoad4Int32(ref[3*refStr:])
	r4 := hbdLoad4Int32(ref[4*refStr:])
	r5 := hbdLoad4Int32(ref[5*refStr:])
	p := 0
	for y := 0; y < height; y++ {
		sum := hbdMulAdd32(r0, c0, bias)
		sum = hbdMulAdd32(r1, c1, sum)
		sum = hbdMulAdd32(r2, c2, sum)
		sum = hbdMulAdd32(r3, c3, sum)
		sum = hbdMulAdd32(r4, c4, sum)
		sum = hbdMulAdd32(r5, c5, sum)
		hbdStoreU16x4(out[y*4:], hbdShiftRight(sum, shift, round0).Add(off))
		if y+1 < height {
			r0, r1, r2, r3, r4 = r1, r2, r3, r4, r5
			p += refStr
			r5 = hbdLoad4Int32(ref[p+5*refStr:])
		}
	}
}

// compoundYHighBD6TapSliding reuses widened rows across output rows.
func compoundYHighBD6TapSliding(ctx *compoundFilterGoSIMDCtx) {
	width, height := ctx.width, ctx.height
	round0 := ctx.round0
	roundBias := archsimd.BroadcastInt32x4(int32(1) << (round0 - 1))
	roundShift := archsimd.BroadcastInt32x4(int32(-round0))
	rndOff := archsimd.BroadcastInt32x4(ctx.roundOffset)

	// Keep the six input rows in registers as the output advances. Only
	// the newly entering row needs to be loaded and widened each time.
	ref, out := ctx.ref, ctx.out
	refStr := ctx.refStr
	_ = ref[(height+4)*refStr+width*2-1]
	_ = out[height*width-1]
	coeff := hbdBroadcastTaps32(&ctx.kernel)
	c0, c1, c2, c3, c4, c5 := coeff[0], coeff[1], coeff[2], coeff[3], coeff[4], coeff[5]
	for x := 0; x < width; x += 8 {
		p := x * 2
		r0l, r0h := hbdWidenU16(archsimd.LoadUint8x16(ref[p : p+16]).ReshapeToUint16s())
		r1l, r1h := hbdWidenU16(archsimd.LoadUint8x16(ref[p+refStr : p+refStr+16]).ReshapeToUint16s())
		r2l, r2h := hbdWidenU16(archsimd.LoadUint8x16(ref[p+2*refStr : p+2*refStr+16]).ReshapeToUint16s())
		r3l, r3h := hbdWidenU16(archsimd.LoadUint8x16(ref[p+3*refStr : p+3*refStr+16]).ReshapeToUint16s())
		r4l, r4h := hbdWidenU16(archsimd.LoadUint8x16(ref[p+4*refStr : p+4*refStr+16]).ReshapeToUint16s())
		r5l, r5h := hbdWidenU16(archsimd.LoadUint8x16(ref[p+5*refStr : p+5*refStr+16]).ReshapeToUint16s())
		for y := 0; y < height; y++ {
			lo := hbdMulAdd32(r0l, c0, roundBias)
			hi := hbdMulAdd32(r0h, c0, roundBias)
			lo = hbdMulAdd32(r1l, c1, lo)
			hi = hbdMulAdd32(r1h, c1, hi)
			lo = hbdMulAdd32(r2l, c2, lo)
			hi = hbdMulAdd32(r2h, c2, hi)
			lo = hbdMulAdd32(r3l, c3, lo)
			hi = hbdMulAdd32(r3h, c3, hi)
			lo = hbdMulAdd32(r4l, c4, lo)
			hi = hbdMulAdd32(r4h, c4, hi)
			lo = hbdMulAdd32(r5l, c5, lo)
			hi = hbdMulAdd32(r5h, c5, hi)
			lo = hbdShiftRight(lo, roundShift, round0).Add(rndOff)
			hi = hbdShiftRight(hi, roundShift, round0).Add(rndOff)
			hbdStoreU16x8(out[y*width+x:], lo, hi)
			if y+1 < height {
				r0l, r0h = r1l, r1h
				r1l, r1h = r2l, r2h
				r2l, r2h = r3l, r3h
				r3l, r3h = r4l, r4h
				r4l, r4h = r5l, r5h
				p += refStr
				r5l, r5h = hbdWidenU16(archsimd.LoadUint8x16(ref[p+5*refStr : p+5*refStr+16]).ReshapeToUint16s())
			}
		}
	}
}

// compound2DHighBDKernel is the separable 2D HBD compound predictor. The output
// is roundPowerOfTwo7(yBias + vertical sum) with no further offset.

// blendCompoundAvgHighBDKernel averages two CONV_BUF predictions with distance
// weights: clip(roundPowerOfTwo(((s0*fwd + s1*bck) >> 4) - roundOffset, bits)).
// The wrapper guarantees width%4 == 0.
func blendCompoundAvgHighBDKernel(ctx *compoundBlendGoSIMDCtx, roundOffset, roundBits int) {
	roundBiasScalar := 1 << (roundBits - 1)
	if ctx.fwd == 8 && ctx.bck == 8 && roundOffset >= roundBiasScalar && roundOffset-roundBiasScalar <= 65535 {
		// For equal distance weights, ((a*8 + b*8) >> 4) is the
		// floor-average. Saturating subtraction folds the zero clip and
		// rounding bias into a single uint16 operation.
		threshold := archsimd.BroadcastUint16x8(uint16(roundOffset - roundBiasScalar))
		maxV := archsimd.BroadcastUint16x8(uint16(ctx.maxVal))
		shiftAmount := archsimd.BroadcastInt16x8(int16(-roundBits))
		width, height := int(ctx.width), int(ctx.height)
		for y := range height {
			s0 := ctx.src0[y*width:]
			s1 := ctx.src1[y*width:]
			dst := ctx.dst[y*ctx.dstStr:]
			x := 0
			for ; x+8 <= width; x += 8 {
				a := archsimd.LoadUint16x8(s0[x : x+8])
				b := archsimd.LoadUint16x8(s1[x : x+8])
				pix := hbdShiftRightU16(hbdAverageU16(a, b).SubSaturated(threshold), shiftAmount, roundBits).Min(maxV)
				pix.ReshapeToUint8s().Store(dst[x*2 : x*2+16])
			}
			if x < width {
				var a, b [8]uint16
				copy(a[:4], s0[x:x+4])
				copy(b[:4], s1[x:x+4])
				pix := hbdShiftRightU16(hbdAverageU16(archsimd.LoadUint16x8(a[:]), archsimd.LoadUint16x8(b[:])).SubSaturated(threshold), shiftAmount, roundBits).Min(maxV)
				var bytes [16]byte
				pix.ReshapeToUint8s().Store(bytes[:])
				copy(dst[x*2:x*2+8], bytes[:8])
			}
		}
		return
	}
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

func hbdAverageU16(a, b archsimd.Uint16x8) archsimd.Uint16x8 {
	return a.And(b).Add(a.Xor(b).ShiftAllRight(1))
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
	if width == 4 && (ny == 4 || ny == 6) {
		compoundVerticalIMHighBDW4(ctx.out, height, ctx.im, ctx.imStr, &yTaps, ny, yBias)
		return
	}
	if ny == 6 && width >= 8 && width%8 == 0 {
		compoundVerticalIMHighBD6(ctx.out, width, height, ctx.im, ctx.imStr, &yTaps, yBias)
		return
	}
	if ny == 8 && width >= 8 && width%8 == 0 {
		compoundVerticalIMHighBD8(ctx.out, width, height, ctx.im, ctx.imStr, &yTaps, yBias)
		return
	}
	compoundVerticalIM(ctx.out, width, height, ctx.im, ctx.imStr, &yTaps, ny, yBias)
}

// compoundVerticalIMHighBDW4 reuses the intermediate rows of narrow blocks.
func compoundVerticalIMHighBDW4(out []uint16, height int, im []int32, imStr int, taps *hbdTaps32, n int, yBias archsimd.Int32x4) {
	if n == 4 {
		compoundVerticalIMHighBDW4x4(out, height, im, imStr, taps, yBias)
	} else {
		compoundVerticalIMHighBDW4x6(out, height, im, imStr, taps, yBias)
	}
}

func compoundVerticalIMHighBDW4x4(out []uint16, height int, im []int32, imStr int, taps *hbdTaps32, yBias archsimd.Int32x4) {
	_ = im[(height+2)*imStr+3]
	_ = out[height*4-1]
	c0, c1, c2, c3 := taps[0], taps[1], taps[2], taps[3]
	bias := yBias.Add(archsimd.BroadcastInt32x4(1 << (filterBits - 1)))
	shift := archsimd.BroadcastInt32x4(-filterBits)
	r0, r1, r2, r3 := archsimd.LoadInt32x4(im[:4]), archsimd.LoadInt32x4(im[1*imStr:1*imStr+4]), archsimd.LoadInt32x4(im[2*imStr:2*imStr+4]), archsimd.LoadInt32x4(im[3*imStr:3*imStr+4])
	p := 0
	for y := 0; y < height; y++ {
		sum := hbdMulAdd32(r0, c0, bias)
		sum = hbdMulAdd32(r1, c1, sum)
		sum = hbdMulAdd32(r2, c2, sum)
		sum = hbdMulAdd32(r3, c3, sum)
		hbdStoreU16x4(out[y*4:], hbdShiftRight(sum, shift, filterBits))
		if y+1 < height {
			r0, r1, r2 = r1, r2, r3
			p += imStr
			r3 = archsimd.LoadInt32x4(im[p+3*imStr : p+3*imStr+4])
		}
	}
}

func compoundVerticalIMHighBDW4x6(out []uint16, height int, im []int32, imStr int, taps *hbdTaps32, yBias archsimd.Int32x4) {
	_ = im[(height+4)*imStr+3]
	_ = out[height*4-1]
	c0, c1, c2, c3, c4, c5 := taps[0], taps[1], taps[2], taps[3], taps[4], taps[5]
	bias := yBias.Add(archsimd.BroadcastInt32x4(1 << (filterBits - 1)))
	shift := archsimd.BroadcastInt32x4(-filterBits)
	r0, r1, r2, r3, r4, r5 := archsimd.LoadInt32x4(im[:4]), archsimd.LoadInt32x4(im[1*imStr:1*imStr+4]), archsimd.LoadInt32x4(im[2*imStr:2*imStr+4]), archsimd.LoadInt32x4(im[3*imStr:3*imStr+4]), archsimd.LoadInt32x4(im[4*imStr:4*imStr+4]), archsimd.LoadInt32x4(im[5*imStr:5*imStr+4])
	p := 0
	for y := 0; y < height; y++ {
		sum := hbdMulAdd32(r0, c0, bias)
		sum = hbdMulAdd32(r1, c1, sum)
		sum = hbdMulAdd32(r2, c2, sum)
		sum = hbdMulAdd32(r3, c3, sum)
		sum = hbdMulAdd32(r4, c4, sum)
		sum = hbdMulAdd32(r5, c5, sum)
		hbdStoreU16x4(out[y*4:], hbdShiftRight(sum, shift, filterBits))
		if y+1 < height {
			r0, r1, r2, r3, r4 = r1, r2, r3, r4, r5
			p += imStr
			r5 = archsimd.LoadInt32x4(im[p+5*imStr : p+5*imStr+4])
		}
	}
}

// compoundVerticalIMHighBD6 keeps the six vertical coefficients in registers
// and folds the final rounding bias into the first multiply-accumulate. The
// four-lane chains for the two halves of a block can advance independently.
func compoundVerticalIMHighBD6(out []uint16, width, height int, im []int32, imStr int, taps *hbdTaps32, yBias archsimd.Int32x4) {
	_ = im[(height+4)*imStr+width-1]
	_ = out[height*width-1]
	c0, c1, c2 := taps[0], taps[1], taps[2]
	c3, c4, c5 := taps[3], taps[4], taps[5]
	bias := yBias.Add(archsimd.BroadcastInt32x4(1 << (filterBits - 1)))
	shift := archsimd.BroadcastInt32x4(-filterBits)
	for y := 0; y < height; y++ {
		for x := 0; x < width; x += 8 {
			p := y*imStr + x
			lo := hbdMulAdd32(archsimd.LoadInt32x4(im[p:p+4]), c0, bias)
			hi := hbdMulAdd32(archsimd.LoadInt32x4(im[p+4:p+8]), c0, bias)
			lo = hbdMulAdd32(archsimd.LoadInt32x4(im[p+imStr:p+imStr+4]), c1, lo)
			hi = hbdMulAdd32(archsimd.LoadInt32x4(im[p+imStr+4:p+imStr+8]), c1, hi)
			lo = hbdMulAdd32(archsimd.LoadInt32x4(im[p+2*imStr:p+2*imStr+4]), c2, lo)
			hi = hbdMulAdd32(archsimd.LoadInt32x4(im[p+2*imStr+4:p+2*imStr+8]), c2, hi)
			lo = hbdMulAdd32(archsimd.LoadInt32x4(im[p+3*imStr:p+3*imStr+4]), c3, lo)
			hi = hbdMulAdd32(archsimd.LoadInt32x4(im[p+3*imStr+4:p+3*imStr+8]), c3, hi)
			lo = hbdMulAdd32(archsimd.LoadInt32x4(im[p+4*imStr:p+4*imStr+4]), c4, lo)
			hi = hbdMulAdd32(archsimd.LoadInt32x4(im[p+4*imStr+4:p+4*imStr+8]), c4, hi)
			lo = hbdMulAdd32(archsimd.LoadInt32x4(im[p+5*imStr:p+5*imStr+4]), c5, lo)
			hi = hbdMulAdd32(archsimd.LoadInt32x4(im[p+5*imStr+4:p+5*imStr+8]), c5, hi)
			lo = hbdShiftRight(lo, shift, filterBits)
			hi = hbdShiftRight(hi, shift, filterBits)
			hbdStoreU16x8(out[y*width+x:], lo, hi)
		}
	}
}

// compoundVerticalIMHighBD8 keeps the eight sharp-filter taps in registers.
func compoundVerticalIMHighBD8(out []uint16, width, height int, im []int32, imStr int, taps *hbdTaps32, yBias archsimd.Int32x4) {
	_ = im[(height+6)*imStr+width-1]
	_ = out[height*width-1]
	c0, c1, c2, c3 := taps[0], taps[1], taps[2], taps[3]
	c4, c5, c6, c7 := taps[4], taps[5], taps[6], taps[7]
	bias := yBias.Add(archsimd.BroadcastInt32x4(1 << (filterBits - 1)))
	shift := archsimd.BroadcastInt32x4(-filterBits)
	for y := 0; y < height; y++ {
		for x := 0; x < width; x += 8 {
			p := y*imStr + x
			lo := hbdMulAdd32(archsimd.LoadInt32x4(im[p:p+4]), c0, bias)
			hi := hbdMulAdd32(archsimd.LoadInt32x4(im[p+4:p+8]), c0, bias)
			lo = hbdMulAdd32(archsimd.LoadInt32x4(im[p+1*imStr:p+1*imStr+4]), c1, lo)
			hi = hbdMulAdd32(archsimd.LoadInt32x4(im[p+1*imStr+4:p+1*imStr+8]), c1, hi)
			lo = hbdMulAdd32(archsimd.LoadInt32x4(im[p+2*imStr:p+2*imStr+4]), c2, lo)
			hi = hbdMulAdd32(archsimd.LoadInt32x4(im[p+2*imStr+4:p+2*imStr+8]), c2, hi)
			lo = hbdMulAdd32(archsimd.LoadInt32x4(im[p+3*imStr:p+3*imStr+4]), c3, lo)
			hi = hbdMulAdd32(archsimd.LoadInt32x4(im[p+3*imStr+4:p+3*imStr+8]), c3, hi)
			lo = hbdMulAdd32(archsimd.LoadInt32x4(im[p+4*imStr:p+4*imStr+4]), c4, lo)
			hi = hbdMulAdd32(archsimd.LoadInt32x4(im[p+4*imStr+4:p+4*imStr+8]), c4, hi)
			lo = hbdMulAdd32(archsimd.LoadInt32x4(im[p+5*imStr:p+5*imStr+4]), c5, lo)
			hi = hbdMulAdd32(archsimd.LoadInt32x4(im[p+5*imStr+4:p+5*imStr+8]), c5, hi)
			lo = hbdMulAdd32(archsimd.LoadInt32x4(im[p+6*imStr:p+6*imStr+4]), c6, lo)
			hi = hbdMulAdd32(archsimd.LoadInt32x4(im[p+6*imStr+4:p+6*imStr+8]), c6, hi)
			lo = hbdMulAdd32(archsimd.LoadInt32x4(im[p+7*imStr:p+7*imStr+4]), c7, lo)
			hi = hbdMulAdd32(archsimd.LoadInt32x4(im[p+7*imStr+4:p+7*imStr+8]), c7, hi)
			lo = hbdShiftRight(lo, shift, filterBits)
			hi = hbdShiftRight(hi, shift, filterBits)
			hbdStoreU16x8(out[y*width+x:], lo, hi)
		}
	}
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
