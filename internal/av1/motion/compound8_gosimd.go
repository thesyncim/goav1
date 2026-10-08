// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && (amd64 || arm64) && !purego

package motion

import (
	"simd/archsimd"

	"github.com/thesyncim/goav1/internal/av1/frame"
)

// Go-native SIMD 8-bit compound (CONV_BUF) predictors and distance-weighted
// blend. Each kernel computes the scalar formula of its pure-Go reference
// (predictInterCompoundRef8ToConvBuf*PureGo and blendCompoundAvg8PureGo) with
// the same int32 arithmetic as the HBD kernels: samples widen from bytes, sums
// stay in int32, and the CONV_BUF store is the uint16 cast.
//
// 8-bit compound always uses round0 = compoundRound0Bits (3).

func init() {
	if !hbdSIMDAvailable() {
		return
	}
	blendCompoundAvg8Impl = blendCompoundAvg8GoSIMD
	predictInterCompoundRef8ToConvBufCopyImpl = predictInterCompoundRef8ToConvBufCopyGoSIMD
	predictInterCompoundRef8ToConvBufXImpl = predictInterCompoundRef8ToConvBufXGoSIMD
	predictInterCompoundRef8ToConvBufYImpl = predictInterCompoundRef8ToConvBufYGoSIMD
	predictInterCompoundRef8ToConvBuf2DImpl = predictInterCompoundRef8ToConvBuf2DGoSIMD
}

// u8Bytes16 returns the 16 bytes at the start of pix. A shorter slice is
// staged through a zero-padded window, so loads never leave the slice.
func u8Bytes16(pix []byte) archsimd.Uint8x16 {
	if len(pix) >= 16 {
		return archsimd.LoadUint8x16(pix[:16])
	}
	var w [16]byte
	copy(w[:], pix)
	return archsimd.LoadUint8x16(w[:])
}

// u8Samples8 widens the first eight bytes at pix to int16 sample lanes.
func u8Samples8(pix []byte) archsimd.Int16x8 {
	return u8Bytes16(pix).ExtendLo8ToUint16().BitsToInt16()
}

// u8SumRowX is the eight-lane horizontal tap sum over n taps for byte samples.
// It loads two 16-byte vectors once and forms each tap's sample vector with a
// constant byte shift. The taps past n are zero, so the result is the n-tap sum.
func u8SumRowX(pix []byte, t *hbdTaps16, n int) (lo, hi archsimd.Int32x4) {
	lo = archsimd.BroadcastInt32x4(0)
	hi = lo
	if len(pix) < 32 {
		for k := range n {
			lo, hi = hbdMAC8(lo, hi, u8Samples8(pix[k:]), t[k])
		}
		return lo, hi
	}
	a := archsimd.LoadUint8x16(pix[:16])
	b := archsimd.LoadUint8x16(pix[16:32])
	lo, hi = hbdMAC8(lo, hi, a.ExtendLo8ToUint16().BitsToInt16(), t[0])
	lo, hi = hbdMAC8(lo, hi, b.ConcatShiftBytesRight(a, 1).ExtendLo8ToUint16().BitsToInt16(), t[1])
	lo, hi = hbdMAC8(lo, hi, b.ConcatShiftBytesRight(a, 2).ExtendLo8ToUint16().BitsToInt16(), t[2])
	lo, hi = hbdMAC8(lo, hi, b.ConcatShiftBytesRight(a, 3).ExtendLo8ToUint16().BitsToInt16(), t[3])
	lo, hi = hbdMAC8(lo, hi, b.ConcatShiftBytesRight(a, 4).ExtendLo8ToUint16().BitsToInt16(), t[4])
	lo, hi = hbdMAC8(lo, hi, b.ConcatShiftBytesRight(a, 5).ExtendLo8ToUint16().BitsToInt16(), t[5])
	lo, hi = hbdMAC8(lo, hi, b.ConcatShiftBytesRight(a, 6).ExtendLo8ToUint16().BitsToInt16(), t[6])
	lo, hi = hbdMAC8(lo, hi, b.ConcatShiftBytesRight(a, 7).ExtendLo8ToUint16().BitsToInt16(), t[7])
	return lo, hi
}

// u8SumColY is the eight-lane vertical tap sum over n taps for byte samples:
// tap k reads eight bytes starting step bytes after tap k-1.
func u8SumColY(pix []byte, step int, t *hbdTaps16, n int) (lo, hi archsimd.Int32x4) {
	lo = archsimd.BroadcastInt32x4(0)
	hi = lo
	for k := range n {
		lo, hi = hbdMAC8(lo, hi, u8Samples8(pix[k*step:]), t[k])
	}
	return lo, hi
}

// u8HorizontalIM computes the biased horizontal pass of the 8-bit 2D compound:
// round3(bias + sum) int32 intermediate samples over rows reference rows.
func u8HorizontalIM(ref []byte, refStr int, im []int32, imStr int, width, rows int, taps *hbdTaps16, n int, bias archsimd.Int32x4) {
	for y := range rows {
		src := ref[y*refStr:]
		row := im[y*imStr:]
		x := 0
		for ; x+8 <= width; x += 8 {
			lo, hi := u8SumRowX(src[x:], taps, n)
			lo = hbdRound(lo.Add(bias), compoundRound0Bits)
			hi = hbdRound(hi.Add(bias), compoundRound0Bits)
			lo.Store(row[x : x+4])
			hi.Store(row[x+4 : x+8])
		}
		if x < width {
			var win [32]byte
			copy(win[:], src[x:])
			lo, _ := u8SumRowX(win[:], taps, n)
			lo = hbdRound(lo.Add(bias), compoundRound0Bits)
			lo.Store(row[x : x+4])
		}
	}
}

// compoundCopy8Kernel is the unfiltered 8-bit compound copy: the scalar
// reference is uint16(s*16 + roundOffset), the low 16 bits of (s << 4) + off.
func compoundCopy8Kernel(ctx *compoundCopyGoSIMDCtx) {
	off := archsimd.BroadcastUint16x8(uint16(ctx.roundOffset))
	width, height := int(ctx.width), int(ctx.height)
	for y := range height {
		src := ctx.ref[y*ctx.refStr:]
		out := ctx.out[y*width:]
		x := 0
		for ; x+8 <= width; x += 8 {
			s := u8Bytes16(src[x:]).ExtendLo8ToUint16()
			s.ShiftAllLeft(compoundCopy8Shift).Add(off).Store(out[x : x+8])
		}
		if x < width {
			s := u8Bytes16(src[x:]).ExtendLo8ToUint16()
			var t [8]uint16
			s.ShiftAllLeft(compoundCopy8Shift).Add(off).Store(t[:])
			copy(out[x:width], t[:width-x])
		}
	}
}

// compoundX8Kernel is the horizontal 8-bit compound predictor:
// uint16(roundPowerOfTwo3(sum) + roundOffset).
func compoundX8Kernel(ctx *compoundFilterGoSIMDCtx) {
	n := int(ctx.taps)
	taps := hbdBroadcastTaps16(&ctx.kernel)
	off := archsimd.BroadcastInt32x4(int32(ctx.roundOffset))
	width, height := int(ctx.width), int(ctx.height)
	for y := range height {
		src := ctx.ref[y*ctx.refStr:]
		out := ctx.out[y*width:]
		x := 0
		for ; x+8 <= width; x += 8 {
			lo, hi := u8SumRowX(src[x:], &taps, n)
			lo = hbdRound(lo, compoundRound0Bits).Add(off)
			hi = hbdRound(hi, compoundRound0Bits).Add(off)
			hbdStoreU16x8(out[x:], lo, hi)
		}
		if x < width {
			var win [32]byte
			copy(win[:], src[x:])
			lo, _ := u8SumRowX(win[:], &taps, n)
			lo = hbdRound(lo, compoundRound0Bits).Add(off)
			hbdStoreU16x4(out[x:], lo)
		}
	}
}

// compoundY8Kernel is the vertical 8-bit compound predictor:
// uint16(roundPowerOfTwo7(sum << 4) + roundOffset).
func compoundY8Kernel(ctx *compoundFilterGoSIMDCtx) {
	n := int(ctx.taps)
	taps := hbdBroadcastTaps16(&ctx.kernel)
	scale := archsimd.BroadcastInt32x4(int32(1) << (filterBits - compoundRound0Bits))
	off := archsimd.BroadcastInt32x4(int32(ctx.roundOffset))
	width, height := int(ctx.width), int(ctx.height)
	for y := range height {
		src := ctx.ref[y*ctx.refStr:]
		out := ctx.out[y*width:]
		x := 0
		for ; x+8 <= width; x += 8 {
			lo, hi := u8SumColY(src[x:], ctx.refStr, &taps, n)
			lo = hbdRound(lo.Mul(scale), filterBits).Add(off)
			hi = hbdRound(hi.Mul(scale), filterBits).Add(off)
			hbdStoreU16x8(out[x:], lo, hi)
		}
		if x < width {
			var win [8 * 16]byte
			for k := range n {
				copy(win[k*16:k*16+4], src[k*ctx.refStr+x:])
			}
			lo, _ := u8SumColY(win[:], 16, &taps, n)
			lo = hbdRound(lo.Mul(scale), filterBits).Add(off)
			hbdStoreU16x4(out[x:], lo)
		}
	}
}

// compound2D8Kernel is the separable 8-bit 2D compound predictor. The
// horizontal bias is 2^(8+FILTER_BITS-1) and the vertical bias is yBias.
func compound2D8Kernel(ctx *compound2DGoSIMDCtx) {
	nx, ny := int(ctx.tapsX), int(ctx.tapsY)
	xTaps := hbdBroadcastTaps16(&ctx.xKern)
	yTaps := hbdBroadcastTaps32(&ctx.kernel)
	width, height := int(ctx.width), int(ctx.height)
	xBias := archsimd.BroadcastInt32x4(int32(ctx.xBias))
	yBias := archsimd.BroadcastInt32x4(int32(ctx.yBias))

	u8HorizontalIM(ctx.ref, ctx.refStr, ctx.im, ctx.imStr, width, height+ny-1, &xTaps, nx, xBias)
	compoundVerticalIM(ctx.out, width, height, ctx.im, ctx.imStr, &yTaps, ny, yBias)
}

// blendCompoundAvg8Kernel is the 8-bit distance-weighted compound average:
// clip255(roundPowerOfTwo(((s0*fwd + s1*bck) >> 4) - roundOffset, roundBits)).
// The wrapper guarantees width%4 == 0.
func blendCompoundAvg8Kernel(ctx *compoundBlendGoSIMDCtx, roundOffset, roundBits int) {
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
			hbdStoreU8x8(dst[x:], lo, hi)
		}
		if x < width {
			var a, b [8]uint16
			copy(a[:4], s0[x:x+4])
			copy(b[:4], s1[x:x+4])
			lo, _ := hbdBlendHalf(archsimd.LoadUint16x8(a[:]), archsimd.LoadUint16x8(b[:]), fwd, bck, roundOff, roundBits, zero, maxV)
			hbdStoreU8x4(dst[x:], lo)
		}
	}
}

// predictInterCompoundRef8ToConvBufCopyGoSIMD is the 8-bit compound copy
// predictor (round0 must be compoundRound0Bits).
func predictInterCompoundRef8ToConvBufCopyGoSIMD(out []uint16, ref frame.Plane, refX int, refY int, width int, height int, round0 int, roundOffset int) {
	if round0 != compoundRound0Bits || width%4 != 0 ||
		!planeRegionFits(ref, 1, refX, refY, width, height) {
		predictInterCompoundRef8ToConvBufCopyPureGo(out, ref, refX, refY, width, height, round0, roundOffset)
		return
	}
	ctx := compoundCopyGoSIMDCtx{
		out:         out,
		ref:         ref.Pix[refY*ref.Stride+refX:],
		refStr:      ref.Stride,
		width:       width,
		height:      height,
		roundOffset: int32(roundOffset),
	}
	compoundCopy8Kernel(&ctx)
}

// predictInterCompoundRef8ToConvBufXGoSIMD is the horizontal-only 8-bit
// compound predictor.
func predictInterCompoundRef8ToConvBufXGoSIMD(out []uint16, ref frame.Plane, refX int, refY int, width int, height int, kernel [filterTaps]int16, roundOffset int) {
	fo := filterTaps/2 - 1
	if width%4 != 0 || !planeRegionFits(ref, 1, refX-fo, refY, width+filterTaps, height) {
		predictInterCompoundRef8ToConvBufXPureGo(out, ref, refX, refY, width, height, kernel, roundOffset)
		return
	}
	lo, n := hbdTapSpan(&kernel)
	ctx := compoundFilterGoSIMDCtx{
		out:         out,
		ref:         ref.Pix[refY*ref.Stride+refX-fo+lo:],
		kernel:      hbdTrimTaps(&kernel, lo, n),
		refStr:      ref.Stride,
		width:       width,
		height:      height,
		round0:      compoundRound0Bits,
		roundOffset: int32(roundOffset),
		taps:        n,
	}
	compoundX8Kernel(&ctx)
}

// predictInterCompoundRef8ToConvBufYGoSIMD is the vertical-only 8-bit compound
// predictor.
func predictInterCompoundRef8ToConvBufYGoSIMD(out []uint16, ref frame.Plane, refX int, refY int, width int, height int, kernel [filterTaps]int16, round0 int, roundOffset int) {
	fo := filterTaps/2 - 1
	if round0 != compoundRound0Bits || width%4 != 0 ||
		!planeRegionFits(ref, 1, refX, refY-fo, width, height+filterTaps-1) {
		predictInterCompoundRef8ToConvBufYPureGo(out, ref, refX, refY, width, height, kernel, round0, roundOffset)
		return
	}
	lo, n := hbdTapSpan(&kernel)
	ctx := compoundFilterGoSIMDCtx{
		out:         out,
		ref:         ref.Pix[(refY-fo+lo)*ref.Stride+refX:],
		kernel:      hbdTrimTaps(&kernel, lo, n),
		refStr:      ref.Stride,
		width:       width,
		height:      height,
		round0:      round0,
		roundOffset: int32(roundOffset),
		taps:        n,
	}
	compoundY8Kernel(&ctx)
}

// predictInterCompoundRef8ToConvBuf2DGoSIMD is the separable 8-bit 2D compound
// predictor. The intermediate lives in scratch when provided.
func predictInterCompoundRef8ToConvBuf2DGoSIMD(out []uint16, ref frame.Plane, refX int, refY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16, offsetBits int, scratch *CompoundConvolveScratch) {
	foX := filterTaps/2 - 1
	foY := filterTaps/2 - 1
	if width%4 != 0 || !planeRegionFits(ref, 1, refX-foX, refY-foY, width+filterTaps, height+filterTaps-1) {
		predictInterCompoundRef8ToConvBuf2DPureGo(out, ref, refX, refY, width, height, xKernel, yKernel, offsetBits, scratch)
		return
	}
	var im *compoundIM
	if scratch != nil {
		im = &scratch.im
	} else {
		im = new(compoundIM)
	}
	loX, nX := hbdTapSpan(&xKernel)
	loY, nY := hbdTapSpan(&yKernel)
	ctx := compound2DGoSIMDCtx{
		out:    out,
		ref:    ref.Pix[(refY-foY+loY)*ref.Stride+(refX-foX+loX):],
		kernel: hbdTrimTaps(&yKernel, loY, nY),
		xKern:  hbdTrimTaps(&xKernel, loX, nX),
		refStr: ref.Stride,
		width:  width,
		height: height,
		im:     im[:],
		imStr:  maxBlockSize,
		xBias:  int32(1 << (8 + filterBits - 1)),
		yBias:  int32(1 << offsetBits),
		tapsX:  nX,
		tapsY:  nY,
	}
	compound2D8Kernel(&ctx)
}

// blendCompoundAvg8GoSIMD is the 8-bit distance-weighted compound average.
// The scalar reference is blendCompoundAvg8PureGo.
func blendCompoundAvg8GoSIMD(dst frame.Plane, src0 []uint16, src1 []uint16, dstX int, dstY int, width int, height int, fwdOffset int, bckOffset int, roundOffset int, roundBits int) {
	if width%4 != 0 || roundOffset <= 0 || height <= 0 ||
		fwdOffset < 0 || fwdOffset > 16 || bckOffset < 0 || bckOffset > 16 {
		blendCompoundAvg8PureGo(dst, src0, src1, dstX, dstY, width, height, fwdOffset, bckOffset, roundOffset, roundBits)
		return
	}
	ctx := compoundBlendGoSIMDCtx{
		dst:    dst.Pix[dstY*dst.Stride+dstX:],
		src0:   src0,
		src1:   src1,
		dstStr: dst.Stride,
		width:  width,
		height: height,
		fwd:    fwdOffset,
		bck:    bckOffset,
		maxVal: 255,
	}
	blendCompoundAvg8Kernel(&ctx, roundOffset, roundBits)
}

// compoundCopy8Shift is the 8-bit copy scale exponent: 2*FILTER_BITS -
// compoundRound1Bits - compoundRound0Bits (4), so the copy scales by 16.
const compoundCopy8Shift = 2*filterBits - compoundRound1Bits - compoundRound0Bits
