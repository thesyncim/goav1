// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && amd64 && !purego

package restoration

import "simd/archsimd"

// AVX2 Go-native SIMD horizontal Wiener passes. Eight output columns are
// computed per iteration in 32-bit lanes (one Int32x8), which is exact for every
// bit depth: the widest intermediate is a symmetric pair sum (<= 8190 for the
// 12-bit domain) times a tap, plus the libaom offset, far inside int32.
//
// The seven taps fold into four products exactly as in the arm64 kernel: the
// Wiener filter is symmetric (f0==f6, f1==f5, f2==f4, enforced by
// validWienerInfo), the libaom center reapplication s3<<WienerFilterBits is
// folded into the center tap (f3' = f3 + 128), and
//
//	sum = seed + f0*(s0+s6) + f1*(s1+s5) + f2*(s2+s4) + f3'*s3
//
// where seed folds the libaom offset and the rounding bias 1<<(round0-1). The
// trailing arithmetic shift and the [0,maxClamp] clamp match
// roundPowerOfTwo + clampInt32 bit for bit.

// wienerHorizontalSIMD is the AVX2 form of wienerHorizontal for the uint16 sample
// domain. It applies the reference validity scan first (every window sample must
// be <= max, else the call fails with false), then runs the kernel.
func wienerHorizontalSIMD(src []uint16, srcStride int, srcOrigin int, width int, height int, filter WienerFilter, bitDepth int, round0 int, max uint16, temp []uint16) bool {
	if width < 8 || width%8 != 0 || !wienerFilterSymmetric(filter) {
		return wienerHorizontal(src, srcStride, srcOrigin, width, height, filter, bitDepth, round0, max, temp)
	}
	for row := -WienerHalfwin; row < height+WienerHalfwin; row++ {
		srcStart := srcOrigin + row*srcStride - WienerHalfwin
		for _, s := range src[srcStart : srcStart+width+2*WienerHalfwin] {
			if s > max {
				return false
			}
		}
	}
	wienerHorizontalSIMDKernel(src, srcStride, srcOrigin, width, height, filter, bitDepth, round0, max, temp)
	return true
}

// wienerHorizontalSIMDTrusted is the uint16 AVX2 horizontal pass without the
// per-sample validity scan, used by the decoder-owned trusted entry.
func wienerHorizontalSIMDTrusted(src []uint16, srcStride int, srcOrigin int, width int, height int, filter WienerFilter, bitDepth int, round0 int, max uint16, temp []uint16) {
	if width < 8 || width%8 != 0 || !wienerFilterSymmetric(filter) {
		wienerHorizontalTrusted(src, srcStride, srcOrigin, width, height, filter, bitDepth, round0, max, temp)
		return
	}
	wienerHorizontalSIMDKernel(src, srcStride, srcOrigin, width, height, filter, bitDepth, round0, max, temp)
}

// wienerHorizontalSIMDKernel is the uint16 AVX2 horizontal pass for widths that
// are a multiple of 8. Each 8-lane window load is exactly eight samples, so the
// slice bounds checks are the only guard needed; no trailing pad is read.
func wienerHorizontalSIMDKernel(src []uint16, srcStride int, srcOrigin int, width int, height int, filter WienerFilter, bitDepth int, round0 int, max uint16, temp []uint16) {
	limit := int32(1) << (bitDepth + 1 + WienerFilterBits - round0)
	offset := int32(1) << (bitDepth + WienerFilterBits - 1)
	seedV := archsimd.BroadcastInt32x8(offset + roundBias(round0))
	zero := archsimd.BroadcastInt32x8(0)
	maxV := archsimd.BroadcastInt32x8(limit - 1)
	f0 := archsimd.BroadcastInt32x8(int32(filter[0]))
	f1 := archsimd.BroadcastInt32x8(int32(filter[1]))
	f2 := archsimd.BroadcastInt32x8(int32(filter[2]))
	// The center tap absorbs the s3<<WienerFilterBits center reapplication.
	f3 := archsimd.BroadcastInt32x8(int32(filter[3]) + (1 << WienerFilterBits))
	shift := uint64(round0)
	rows := height + 2*WienerHalfwin
	for row := 0; row < rows; row++ {
		srow := src[srcOrigin+(row-WienerHalfwin)*srcStride-WienerHalfwin:]
		dstRow := temp[row*width : row*width+width]
		for col := 0; col < width; col += 8 {
			s0 := archsimd.LoadUint16x8(srow[col:]).ExtendToUint32().AsInt32x8()
			s1 := archsimd.LoadUint16x8(srow[col+1:]).ExtendToUint32().AsInt32x8()
			s2 := archsimd.LoadUint16x8(srow[col+2:]).ExtendToUint32().AsInt32x8()
			s3 := archsimd.LoadUint16x8(srow[col+3:]).ExtendToUint32().AsInt32x8()
			s4 := archsimd.LoadUint16x8(srow[col+4:]).ExtendToUint32().AsInt32x8()
			s5 := archsimd.LoadUint16x8(srow[col+5:]).ExtendToUint32().AsInt32x8()
			s6 := archsimd.LoadUint16x8(srow[col+6:]).ExtendToUint32().AsInt32x8()
			sum := seedV.Add(s0.Add(s6).Mul(f0)).Add(s1.Add(s5).Mul(f1)).
				Add(s2.Add(s4).Mul(f2)).Add(s3.Mul(f3))
			out := sum.ShiftAllRight(shift).Max(zero).Min(maxV)
			packClampedInt32x8ToUint16(out).Store(dstRow[col:])
		}
	}
	// dav1d returns through vzeroupper; the Go compiler emits none, and
	// dirty upper ymm state slows the legacy-SSE code that follows.
	archsimd.ClearAVXUpperBits()
}

// wienerHorizontalU8SIMD is the AVX2 form of wienerHorizontalU8 for 8-bit source
// samples. Each 8-output group loads one 16-byte window and derives the six
// shifted windows in-register with ConcatShiftBytesRight against zero; the
// caller must have checked wienerHorizontalU8SIMDCanRun so the 16-byte load
// stays inside the padded source.
func wienerHorizontalU8SIMD(src []uint8, srcStride int, srcOrigin int, width int, height int, filter WienerFilter, round0 int, temp []uint16) {
	if width < 8 || width%8 != 0 || round0 != WienerRound0Bits || !wienerFilterSymmetric(filter) {
		wienerHorizontalU8(src, srcStride, srcOrigin, width, height, filter, round0, temp)
		return
	}
	const bitDepth = 8
	limit := int32(1) << (bitDepth + 1 + WienerFilterBits - WienerRound0Bits)
	offset := int32(1) << (bitDepth + WienerFilterBits - 1)
	// The 16-wide columns run first, in their own loop, so no 128-bit vector
	// of the 8-wide tail below is live across the calls.
	w16 := 0
	if wienerU8Wide16 {
		w16 = width &^ 15
		k16 := newWienerU8Taps(filter, offset+roundBias(WienerRound0Bits), limit-1)
		for row := 0; row < height+2*WienerHalfwin; row++ {
			srow := src[srcOrigin+(row-WienerHalfwin)*srcStride-WienerHalfwin:]
			dstRow := temp[row*width : row*width+width]
			for col := 0; col < w16; col += 16 {
				wienerHorizontalU8x16(srow[col:col+24], dstRow[col:col+16], &k16)
			}
		}
	}
	seedV := archsimd.BroadcastInt32x8(offset + roundBias(WienerRound0Bits))
	zero := archsimd.BroadcastInt32x8(0)
	zeroBytes := archsimd.BroadcastUint8x16(0)
	maxV := archsimd.BroadcastInt32x8(limit - 1)
	f0 := archsimd.BroadcastInt32x8(int32(filter[0]))
	f1 := archsimd.BroadcastInt32x8(int32(filter[1]))
	f2 := archsimd.BroadcastInt32x8(int32(filter[2]))
	f3 := archsimd.BroadcastInt32x8(int32(filter[3]) + (1 << WienerFilterBits))
	shift := uint64(WienerRound0Bits)
	rows := height + 2*WienerHalfwin
	for row := 0; row < rows; row++ {
		srow := src[srcOrigin+(row-WienerHalfwin)*srcStride-WienerHalfwin:]
		dstRow := temp[row*width : row*width+width]
		for col := w16; col < width; col += 8 {
			v0 := archsimd.LoadUint8x16(srow[col:]) // s0..s15 of this group
			// Window k is bytes k..k+7 of v0: shift v0 down by k bytes, zero filling
			// the top (zeroBytes is the high operand, v0 the low one).
			s0 := v0.ExtendLo8ToUint32().AsInt32x8()
			s1 := zeroBytes.ConcatShiftBytesRight(v0, 1).ExtendLo8ToUint32().AsInt32x8()
			s2 := zeroBytes.ConcatShiftBytesRight(v0, 2).ExtendLo8ToUint32().AsInt32x8()
			s3 := zeroBytes.ConcatShiftBytesRight(v0, 3).ExtendLo8ToUint32().AsInt32x8()
			s4 := zeroBytes.ConcatShiftBytesRight(v0, 4).ExtendLo8ToUint32().AsInt32x8()
			s5 := zeroBytes.ConcatShiftBytesRight(v0, 5).ExtendLo8ToUint32().AsInt32x8()
			s6 := zeroBytes.ConcatShiftBytesRight(v0, 6).ExtendLo8ToUint32().AsInt32x8()
			sum := seedV.Add(s0.Add(s6).Mul(f0)).Add(s1.Add(s5).Mul(f1)).
				Add(s2.Add(s4).Mul(f2)).Add(s3.Mul(f3))
			out := sum.ShiftAllRight(shift).Max(zero).Min(maxV)
			packClampedInt32x8ToUint16(out).Store(dstRow[col:])
		}
	}
	// dav1d returns through vzeroupper; the Go compiler emits none, and
	// dirty upper ymm state slows the legacy-SSE code that follows.
	archsimd.ClearAVXUpperBits()
}

// packClampedInt32x8ToUint16 narrows eight int32 lanes already clamped to
// [0,65535] to uint16 with VPACKUSDW (AVX2). Truncating forms (VPMOVDW) need
// AVX-512, so the pack goes through two 128-bit halves.
func packClampedInt32x8ToUint16(v archsimd.Int32x8) archsimd.Uint16x8 {
	return v.GetLo().SaturateToUint16Concat(v.GetHi())
}

// wienerU8Taps holds the dav1d-style pmaddwd tap pairs of the 16-wide 8-bit
// Wiener kernels.
type wienerU8Taps struct {
	f01, f23, f45, f6 [16]int16 // (f0, f1), (f2, f3'), (f4, f5), (f6, 0) pairs
	seed, max         int32
}

func newWienerU8Taps(filter WienerFilter, seed, max int32) wienerU8Taps {
	k := wienerU8Taps{seed: seed, max: max}
	f3 := int16(int32(filter[3]) + (1 << WienerFilterBits))
	for i := 0; i < 16; i += 2 {
		k.f01[i], k.f01[i+1] = int16(filter[0]), int16(filter[1])
		k.f23[i], k.f23[i+1] = int16(filter[2]), f3
		k.f45[i], k.f45[i+1] = int16(filter[4]), int16(filter[5])
		k.f6[i] = int16(filter[6])
	}
	return k
}

// wienerHorizontalU8x16 is the horizontal pass for sixteen outputs, as
// dav1d's wiener_filter7_8bpc AVX2 forms it: the two 128-bit halves hold
// outputs 0..7 and 8..15, the seven windows come from vpalignr on the
// widened samples s0..s15 and s8..s23, the symmetric pair sums go through
// vpmaddwd against the (f0, f1) and (f2, f3') pairs (exact in int32, every
// operand fits int16), and the clamped result packs back with vpackssdw in
// the order the in-lane unpack produced.
func wienerHorizontalU8x16(s []uint8, dst []uint16, k *wienerU8Taps) {
	a := archsimd.LoadUint8x16(s[0:16]).ExtendToUint16().AsUint8x32()
	b := archsimd.LoadUint8x16(s[8:24]).ExtendToUint16().AsUint8x32()
	w := func(sh uint64) archsimd.Int16x16 { return b.ConcatShiftBytesRightGrouped(a, sh).AsInt16x16() }
	p0 := a.AsInt16x16().Add(w(12))
	p1 := w(2).Add(w(10))
	p2 := w(4).Add(w(8))
	p3 := w(6)
	f01 := archsimd.LoadInt16x16Array(&k.f01)
	f23 := archsimd.LoadInt16x16Array(&k.f23)
	seed := archsimd.BroadcastInt32x8(k.seed)
	zero := archsimd.Int32x8{}
	mx := archsimd.BroadcastInt32x8(k.max)
	lo := p0.InterleaveLoGrouped(p1).DotProductPairs(f01).Add(p2.InterleaveLoGrouped(p3).DotProductPairs(f23)).Add(seed)
	hi := p0.InterleaveHiGrouped(p1).DotProductPairs(f01).Add(p2.InterleaveHiGrouped(p3).DotProductPairs(f23)).Add(seed)
	lo = lo.ShiftAllRight(WienerRound0Bits).Max(zero).Min(mx)
	hi = hi.ShiftAllRight(WienerRound0Bits).Max(zero).Min(mx)
	lo.SaturateToInt16ConcatGrouped(hi).AsUint16x16().Store(dst[:16])
	archsimd.ClearAVXUpperBits()
}
