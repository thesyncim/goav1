// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && (arm64 || amd64) && !purego

package motion

import (
	"encoding/binary"
	"github.com/thesyncim/goav1/internal/av1/frame"
	"simd/archsimd"
)

// Go-native SIMD 8-bit compound (CONV_BUF) predictors and distance-weighted
// blend. Each kernel computes the scalar formula of its pure-Go reference
// (predictInterCompoundRef8ToConvBuf*PureGo and blendCompoundAvg8PureGo).
// AV1 filter taps use halved int16 coefficients for byte samples and an int16
// 2D intermediate; uncommon coefficients retain the int32 fallback.
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
		return archsimd.LoadUint8x16Array((*[16]uint8)(pix[:16]))
	}
	var w [16]byte
	copy(w[:], pix)
	return archsimd.LoadUint8x16Array(&w)
}

// u8Samples8 widens the first eight bytes at pix to int16 sample lanes.
func u8Samples8(pix []byte) archsimd.Int16x8 {
	return u8Bytes16(pix).ExtendLo8ToUint16().BitsToInt16()
}

// u8SumRowHalf reads the full 8-output tap window once. The caller proves the
// halved filter sum fits signed 16-bit lanes. A padded load handles exact
// minimum-length windows at the last row without touching the next row.
func u8SumRowHalf(pix []byte, t *hbdTaps16, n int) archsimd.Int16x8 {
	a := u8Bytes16(pix)
	z := archsimd.BroadcastInt16x8(0)
	s := u8HalfMAC(a.ExtendLo8ToUint16().BitsToInt16(), t[0], z)
	s = u8HalfMAC(a.ConcatShiftBytesRight(a, 1).ExtendLo8ToUint16().BitsToInt16(), t[1], s)
	s = u8HalfMAC(a.ConcatShiftBytesRight(a, 2).ExtendLo8ToUint16().BitsToInt16(), t[2], s)
	s = u8HalfMAC(a.ConcatShiftBytesRight(a, 3).ExtendLo8ToUint16().BitsToInt16(), t[3], s)
	if n > 4 {
		s = u8HalfMAC(a.ConcatShiftBytesRight(a, 4).ExtendLo8ToUint16().BitsToInt16(), t[4], s)
		s = u8HalfMAC(a.ConcatShiftBytesRight(a, 5).ExtendLo8ToUint16().BitsToInt16(), t[5], s)
	}
	if n > 6 {
		s = u8HalfMAC(a.ConcatShiftBytesRight(a, 6).ExtendLo8ToUint16().BitsToInt16(), t[6], s)
		s = u8HalfMAC(a.ConcatShiftBytesRight(a, 7).ExtendLo8ToUint16().BitsToInt16(), t[7], s)
	}
	return s
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
	a := archsimd.LoadUint8x16Array((*[16]uint8)(pix[:16]))
	b := archsimd.LoadUint8x16Array((*[16]uint8)(pix[16:32]))
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
			lo.StoreArray((*[4]int32)(row[x : x+4]))
			hi.StoreArray((*[4]int32)(row[x+4 : x+8]))
		}
		if x < width {
			var win [32]byte
			copy(win[:], src[x:])
			lo, _ := u8SumRowX(win[:], taps, n)
			lo = hbdRound(lo.Add(bias), compoundRound0Bits)
			lo.StoreArray((*[4]int32)(row[x : x+4]))
		}
	}
}

// compoundCopy8Kernel is the unfiltered 8-bit compound copy: the scalar
// reference is uint16(s*16 + roundOffset), the low 16 bits of (s << 4) + off.
func compoundCopy8Kernel(ctx *compoundCopyGoSIMDCtx) {
	off := archsimd.BroadcastUint16x8(uint16(ctx.roundOffset))
	width, height := int(ctx.width), int(ctx.height)
	if width%16 == 0 {
		for y := range height {
			src := ctx.ref[y*ctx.refStr:]
			out := ctx.out[y*width:]
			for x := 0; x < width; x += 16 {
				v := archsimd.LoadUint8x16Array((*[16]uint8)(src[x : x+16]))
				v.ExtendLo8ToUint16().ShiftAllLeft(compoundCopy8Shift).Add(off).StoreArray((*[8]uint16)(out[x : x+8]))
				v.ConcatShiftBytesRight(v, 8).ExtendLo8ToUint16().ShiftAllLeft(compoundCopy8Shift).Add(off).StoreArray((*[8]uint16)(out[x+8 : x+16]))
			}
		}
		return
	}
	for y := range height {
		src := ctx.ref[y*ctx.refStr:]
		out := ctx.out[y*width:]
		x := 0
		for ; x+8 <= width; x += 8 {
			s := u8Bytes16(src[x:]).ExtendLo8ToUint16()
			s.ShiftAllLeft(compoundCopy8Shift).Add(off).StoreArray((*[8]uint16)(out[x : x+8]))
		}
		if x < width {
			s := u8Bytes16(src[x:]).ExtendLo8ToUint16()
			var t [8]uint16
			s.ShiftAllLeft(compoundCopy8Shift).Add(off).StoreArray(&t)
			copy(out[x:width], t[:width-x])
		}
	}
}

// compoundX8Kernel is the horizontal 8-bit compound predictor:
// uint16(roundPowerOfTwo3(sum) + roundOffset).
func compoundX8Kernel(ctx *compoundFilterGoSIMDCtx) {
	if half, ok := u8HalfTaps(&ctx.kernel, 2); ok {
		compoundX8HalfKernel(ctx, &half)
		return
	}
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

func compoundX8HalfKernel(ctx *compoundFilterGoSIMDCtx, taps *hbdTaps16) {
	width, height, n := ctx.width, ctx.height, ctx.taps
	need := (height-1)*ctx.refStr + width + 8
	if width%8 == 0 && ctx.refStr > 0 && need >= 0 && len(ctx.ref) >= need {
		compoundX8HalfResident(ctx)
		return
	}
	round := archsimd.BroadcastInt16x8(2)
	right2 := archsimd.BroadcastInt16x8(-2)
	off := archsimd.BroadcastUint16x8(uint16(ctx.roundOffset))
	for y := range height {
		src := ctx.ref[y*ctx.refStr:]
		out := ctx.out[y*width:]
		x := 0
		for ; x+8 <= width; x += 8 {
			u8Right16(u8SumRowHalf(src[x:], taps, n).Add(round), right2, 2).ToBits().Add(off).StoreArray((*[8]uint16)(out[x : x+8]))
		}
		if x < width {
			var win [16]byte
			copy(win[:], src[x:])
			var tmp [8]uint16
			u8Right16(u8SumRowHalf(win[:], taps, n).Add(round), right2, 2).ToBits().Add(off).StoreArray(&tmp)
			copy(out[x:width], tmp[:width-x])
		}
	}
}

func compoundX8HalfResident(ctx *compoundFilterGoSIMDCtx) {
	c0 := archsimd.BroadcastInt16x8(ctx.kernel[0] / 2)
	c1 := archsimd.BroadcastInt16x8(ctx.kernel[1] / 2)
	c2 := archsimd.BroadcastInt16x8(ctx.kernel[2] / 2)
	c3 := archsimd.BroadcastInt16x8(ctx.kernel[3] / 2)
	c4 := archsimd.BroadcastInt16x8(ctx.kernel[4] / 2)
	c5 := archsimd.BroadcastInt16x8(ctx.kernel[5] / 2)
	c6 := archsimd.BroadcastInt16x8(ctx.kernel[6] / 2)
	c7 := archsimd.BroadcastInt16x8(ctx.kernel[7] / 2)
	round := archsimd.BroadcastInt16x8(2)
	right2 := archsimd.BroadcastInt16x8(-2)
	off := archsimd.BroadcastUint16x8(uint16(ctx.roundOffset))
	// Each row is a width+8 byte window of the len(ctx.ref) >= need extent the
	// dispatching kernel checked; the loop bound proves each 16-byte load.
	for y := range ctx.height {
		out := ctx.out[y*ctx.width:]
		src := ctx.ref[y*ctx.refStr:][: ctx.width+8 : ctx.width+8]
		for x := 0; x <= len(src)-16; x += 8 {
			a := archsimd.LoadUint8x16Array((*[16]byte)(src[x : x+16]))
			sum := u8HalfMAC(a.ExtendLo8ToUint16().BitsToInt16(), c0, round)
			sum = u8HalfMAC(a.ConcatShiftBytesRight(a, 1).ExtendLo8ToUint16().BitsToInt16(), c1, sum)
			sum = u8HalfMAC(a.ConcatShiftBytesRight(a, 2).ExtendLo8ToUint16().BitsToInt16(), c2, sum)
			sum = u8HalfMAC(a.ConcatShiftBytesRight(a, 3).ExtendLo8ToUint16().BitsToInt16(), c3, sum)
			if ctx.taps > 4 {
				sum = u8HalfMAC(a.ConcatShiftBytesRight(a, 4).ExtendLo8ToUint16().BitsToInt16(), c4, sum)
				sum = u8HalfMAC(a.ConcatShiftBytesRight(a, 5).ExtendLo8ToUint16().BitsToInt16(), c5, sum)
			}
			if ctx.taps > 6 {
				sum = u8HalfMAC(a.ConcatShiftBytesRight(a, 6).ExtendLo8ToUint16().BitsToInt16(), c6, sum)
				sum = u8HalfMAC(a.ConcatShiftBytesRight(a, 7).ExtendLo8ToUint16().BitsToInt16(), c7, sum)
			}
			u8Right16(sum, right2, 2).ToBits().Add(off).StoreArray((*[8]uint16)(out[x : x+8]))
		}
	}
}

// compoundY8Kernel is the vertical 8-bit compound predictor:
// uint16(roundPowerOfTwo7(sum << 4) + roundOffset).
func compoundY8Kernel(ctx *compoundFilterGoSIMDCtx) {
	if half, ok := u8HalfTaps(&ctx.kernel, 2); ok {
		compoundY8HalfKernel(ctx, &half)
		return
	}
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

func compoundY8HalfKernel(ctx *compoundFilterGoSIMDCtx, taps *hbdTaps16) {
	width, height, n := ctx.width, ctx.height, ctx.taps
	need := (height+u8LoadedTaps(n)-2)*ctx.refStr + width + 8
	if width%8 == 0 && ctx.refStr > 0 && need >= 0 && len(ctx.ref) >= need {
		compoundY8HalfResident(ctx)
		return
	}
	round := archsimd.BroadcastInt16x8(2)
	right2 := archsimd.BroadcastInt16x8(-2)
	off := archsimd.BroadcastUint16x8(uint16(ctx.roundOffset))
	for y := range height {
		src := ctx.ref[y*ctx.refStr:]
		out := ctx.out[y*width:]
		x := 0
		for ; x+8 <= width; x += 8 {
			u8Right16(u8SumColHalf(src[x:], ctx.refStr, taps, n).Add(round), right2, 2).ToBits().Add(off).StoreArray((*[8]uint16)(out[x : x+8]))
		}
		if x < width {
			var win [8 * 16]byte
			for k := range n {
				copy(win[k*16:k*16+4], src[k*ctx.refStr+x:])
			}
			var tmp [8]uint16
			u8Right16(u8SumColHalf(win[:], 16, taps, n).Add(round), right2, 2).ToBits().Add(off).StoreArray(&tmp)
			copy(out[x:width], tmp[:width-x])
		}
	}
}

func compoundY8HalfResident(ctx *compoundFilterGoSIMDCtx) {
	c0 := archsimd.BroadcastInt16x8(ctx.kernel[0] / 2)
	c1 := archsimd.BroadcastInt16x8(ctx.kernel[1] / 2)
	c2 := archsimd.BroadcastInt16x8(ctx.kernel[2] / 2)
	c3 := archsimd.BroadcastInt16x8(ctx.kernel[3] / 2)
	c4 := archsimd.BroadcastInt16x8(ctx.kernel[4] / 2)
	c5 := archsimd.BroadcastInt16x8(ctx.kernel[5] / 2)
	c6 := archsimd.BroadcastInt16x8(ctx.kernel[6] / 2)
	c7 := archsimd.BroadcastInt16x8(ctx.kernel[7] / 2)
	round := archsimd.BroadcastInt16x8(2)
	right2 := archsimd.BroadcastInt16x8(-2)
	off := archsimd.BroadcastUint16x8(uint16(ctx.roundOffset))
	// Each tap row is a 16-byte window of the len(ctx.ref) >= need extent the
	// dispatching kernel checked.
	ref := ctx.ref
	step := ctx.refStr
	load := func(offset int) archsimd.Int16x8 {
		return archsimd.LoadUint8x16Array((*[16]byte)(ref[offset : offset+16])).ExtendLo8ToUint16().BitsToInt16()
	}
	for y := range ctx.height {
		out := ctx.out[y*ctx.width:]
		for x := 0; x < ctx.width; x += 8 {
			pos := y*step + x
			sum := u8HalfMAC(load(pos), c0, round)
			sum = u8HalfMAC(load(pos+step), c1, sum)
			sum = u8HalfMAC(load(pos+2*step), c2, sum)
			sum = u8HalfMAC(load(pos+3*step), c3, sum)
			if ctx.taps > 4 {
				sum = u8HalfMAC(load(pos+4*step), c4, sum)
				sum = u8HalfMAC(load(pos+5*step), c5, sum)
			}
			if ctx.taps > 6 {
				sum = u8HalfMAC(load(pos+6*step), c6, sum)
				sum = u8HalfMAC(load(pos+7*step), c7, sum)
			}
			u8Right16(sum, right2, 2).ToBits().Add(off).StoreArray((*[8]uint16)(out[x : x+8]))
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

// The wrapper allocates the rounded tap count of intermediate rows; every
// eight-sample vector load below stays inside its 128-sample row.
func compound2D8HalfKernel(ctx *compound2DGoSIMDCtx, im []int16, xTaps *hbdTaps16) {
	nx, ny := ctx.tapsX, ctx.tapsY
	rows := ctx.height + ny - 1
	xRound := archsimd.BroadcastInt16x8(8194)
	right2 := archsimd.BroadcastInt16x8(-2)
	need := (rows-1)*ctx.refStr + ctx.width + 8
	if ctx.width%8 == 0 && ctx.refStr > 0 && need >= 0 && len(ctx.ref) >= need {
		xCtx := convolve8GoSIMDCtx{
			ref: ctx.ref, xKern: ctx.xKern, refStr: ctx.refStr,
			width: ctx.width, height: ctx.height, tapsX: nx, tapsY: ny,
		}
		u8HorizontalHalfIMResident(&xCtx, im)
	} else {
		for y := range rows {
			src := ctx.ref[y*ctx.refStr:]
			row := im[y*maxBlockSize:]
			x := 0
			for ; x+8 <= ctx.width; x += 8 {
				u8Right16(u8SumRowHalf(src[x:], xTaps, nx).Add(xRound), right2, 2).StoreArray((*[8]int16)(row[x : x+8]))
			}
			if x < ctx.width {
				var win [16]byte
				copy(win[:], src[x:])
				var tmp [8]int16
				u8Right16(u8SumRowHalf(win[:], xTaps, nx).Add(xRound), right2, 2).StoreArray(&tmp)
				copy(row[x:ctx.width], tmp[:ctx.width-x])
			}
		}
	}
	c0 := archsimd.BroadcastInt16x8(ctx.kernel[0])
	c1 := archsimd.BroadcastInt16x8(ctx.kernel[1])
	c2 := archsimd.BroadcastInt16x8(ctx.kernel[2])
	c3 := archsimd.BroadcastInt16x8(ctx.kernel[3])
	c4 := archsimd.BroadcastInt16x8(ctx.kernel[4])
	c5 := archsimd.BroadcastInt16x8(ctx.kernel[5])
	c6 := archsimd.BroadcastInt16x8(ctx.kernel[6])
	c7 := archsimd.BroadcastInt16x8(ctx.kernel[7])
	yRound := archsimd.BroadcastInt32x4(int32(ctx.yBias) + 64)
	right7 := archsimd.BroadcastInt32x4(-7)
	// im rows (stride maxBlockSize): the height+taps-1 rows the wrapper supplies.
	load := func(y, x int) archsimd.Int16x8 {
		o := y*maxBlockSize + x
		return archsimd.LoadInt16x8Array((*[8]int16)(im[o : o+8]))
	}
	for y := range ctx.height {
		out := ctx.out[y*ctx.width:]
		for x := 0; x < ctx.width; x += 8 {
			v := load(y, x)
			lo, hi := u8WideMAC8(yRound, yRound, v, c0)
			v = load(y+1, x)
			lo, hi = u8WideMAC8(lo, hi, v, c1)
			v = load(y+2, x)
			lo, hi = u8WideMAC8(lo, hi, v, c2)
			v = load(y+3, x)
			lo, hi = u8WideMAC8(lo, hi, v, c3)
			if ny > 4 {
				v = load(y+4, x)
				lo, hi = u8WideMAC8(lo, hi, v, c4)
				v = load(y+5, x)
				lo, hi = u8WideMAC8(lo, hi, v, c5)
			}
			if ny > 6 {
				v = load(y+6, x)
				lo, hi = u8WideMAC8(lo, hi, v, c6)
				v = load(y+7, x)
				lo, hi = u8WideMAC8(lo, hi, v, c7)
			}
			packed := u8PackU16From32(u8Right32(lo, right7, compoundRound1Bits), u8Right32(hi, right7, compoundRound1Bits))
			if x+8 <= ctx.width {
				packed.StoreArray((*[8]uint16)(out[x : x+8]))
			} else {
				var tmp [8]uint16
				packed.StoreArray(&tmp)
				copy(out[x:ctx.width], tmp[:ctx.width-x])
			}
		}
	}
}

// blendCompoundAvg8Kernel is the 8-bit distance-weighted compound average:
// clip255(roundPowerOfTwo(((s0*fwd + s1*bck) >> 4) - roundOffset, roundBits)).
// The wrapper guarantees width%4 == 0.
func blendCompoundAvg8Kernel(ctx *compoundBlendGoSIMDCtx, roundOffset, roundBits int) {
	if roundBits == 4 && ctx.maxVal == 255 && roundOffset > 0 && roundOffset <= 1<<20 {
		blendCompoundAvg8Fast(ctx, roundOffset)
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
			a := archsimd.LoadUint16x8Array((*[8]uint16)(s0[x : x+8]))
			b := archsimd.LoadUint16x8Array((*[8]uint16)(s1[x : x+8]))
			lo, hi := hbdBlendHalf(a, b, fwd, bck, roundOff, roundBits, zero, maxV)
			hbdStoreU8x8(dst[x:], lo, hi)
		}
		if x < width {
			var a, b [8]uint16
			copy(a[:4], s0[x:x+4])
			copy(b[:4], s1[x:x+4])
			lo, _ := hbdBlendHalf(archsimd.LoadUint16x8Array(&a), archsimd.LoadUint16x8Array(&b), fwd, bck, roundOff, roundBits, zero, maxV)
			hbdStoreU8x4(dst[x:], lo)
		}
	}
}

func blendCompoundAvg8Fast(ctx *compoundBlendGoSIMDCtx, roundOffset int) {
	fwd := archsimd.BroadcastInt32x4(int32(ctx.fwd))
	bck := archsimd.BroadcastInt32x4(int32(ctx.bck))
	bias := archsimd.BroadcastInt32x4(int32((roundOffset - 8) << 4))
	right8 := archsimd.BroadcastInt32x4(-8)
	for y := range ctx.height {
		s0 := ctx.src0[y*ctx.width:]
		s1 := ctx.src1[y*ctx.width:]
		dst := ctx.dst[y*ctx.dstStr:]
		for x := 0; x < ctx.width; x += 8 {
			if x+8 <= ctx.width {
				a := archsimd.LoadUint16x8Array((*[8]uint16)(s0[x : x+8]))
				b := archsimd.LoadUint16x8Array((*[8]uint16)(s1[x : x+8]))
				a0, a1 := u8WidenU16(a)
				b0, b1 := u8WidenU16(b)
				lo := u8MulAdd32(a0, fwd, b0.Mul(bck))
				hi := u8MulAdd32(a1, fwd, b1.Mul(bck))
				lo = u8Right32(lo.Sub(bias), right8, 8)
				hi = u8Right32(hi.Sub(bias), right8, 8)
				u8PackSaturate32(lo, hi).StorePart(dst[x : x+8])
			} else {
				var aa, bb [8]uint16
				copy(aa[:], s0[x:ctx.width])
				copy(bb[:], s1[x:ctx.width])
				a0, _ := u8WidenU16(archsimd.LoadUint16x8Array(&aa))
				b0, _ := u8WidenU16(archsimd.LoadUint16x8Array(&bb))
				lo := u8MulAdd32(a0, fwd, b0.Mul(bck))
				lo = u8Right32(lo.Sub(bias), right8, 8)
				u8PackSaturate32(lo, lo).StorePart(dst[x:ctx.width])
			}
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
	if width%4 != 0 {
		predictInterCompoundRef8ToConvBuf2DPureGo(out, ref, refX, refY, width, height, xKernel, yKernel, offsetBits, scratch)
		return
	}
	if !planeRegionFits(ref, 1, refX-foX, refY-foY, width+filterTaps, height+filterTaps-1) {
		if scratch == nil || !planeRegionFits(ref, 1, 0, 0, ref.Width, ref.Height) {
			predictInterCompoundRef8ToConvBuf2DPureGo(out, ref, refX, refY, width, height, xKernel, yKernel, offsetBits, scratch)
			return
		}
		emu, emuX, emuY := emuEdgeWindow(ref, refX, refY, width, height, &scratch.edge)
		predictInterCompoundRef8ToConvBuf2DGoSIMD(out, emu, emuX, emuY, width, height, xKernel, yKernel, offsetBits, scratch)
		return
	}
	loX, nX := hbdTapSpan(&xKernel)
	loY, nY := hbdTapSpan(&yKernel)
	trimX := hbdTrimTaps(&xKernel, loX, nX)
	trimY := hbdTrimTaps(&yKernel, loY, nY)
	if offsetBits == 19 {
		if half, ok := u8HalfTaps(&trimX, 8194); ok {
			ctx := compound2DGoSIMDCtx{
				out: out, ref: ref.Pix[(refY-foY+loY)*ref.Stride+(refX-foX+loX):],
				kernel: trimY, xKern: trimX, refStr: ref.Stride,
				width: width, height: height, xBias: 1 << (8 + filterBits - 1),
				yBias: 1 << offsetBits, tapsX: nX, tapsY: nY,
			}
			if scratch != nil {
				compound2D8HalfKernel(&ctx, scratch.im8[:], &half)
			} else {
				loadTaps := nY
				if loadTaps < 4 {
					loadTaps = 4
				} else if loadTaps&1 != 0 {
					loadTaps++
				}
				needRows := height + loadTaps - 1
				switch {
				case needRows <= 8:
					var local [8 * maxBlockSize]int16
					compound2D8HalfKernel(&ctx, local[:], &half)
				case needRows <= 32:
					var local [32 * maxBlockSize]int16
					compound2D8HalfKernel(&ctx, local[:], &half)
				default:
					var local compoundIM16
					compound2D8HalfKernel(&ctx, local[:], &half)
				}
			}
			return
		}
	}
	var im *compoundIM
	if scratch != nil {
		im = &scratch.im
	} else {
		im = new(compoundIM)
	}
	ctx := compound2DGoSIMDCtx{
		out:    out,
		ref:    ref.Pix[(refY-foY+loY)*ref.Stride+(refX-foX+loX):],
		kernel: trimY,
		xKern:  trimX,
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

// init binds the high-bit-depth compound predictors to the Go SIMD kernels.
func init() {
	if !hbdSIMDAvailable() {
		return
	}
	blendCompoundAvgHighBDImpl = blendCompoundAvgHighBDGoSIMD
	predictInterCompoundRefHighBDToConvBufCopyResidentImpl = predictInterCompoundRefHighBDToConvBufCopyResidentGoSIMD
	predictInterCompoundRefHighBDToConvBuf2DResidentImpl = predictInterCompoundRefHighBDToConvBuf2DResidentGoSIMD
	predictInterCompoundRefHighBDToConvBuf2DClampedImpl = predictInterCompoundRefHighBDToConvBuf2DClampedGoSIMD
	predictInterCompoundRefHighBDToConvBufXResidentImpl = predictInterCompoundRefHighBDToConvBufXResidentGoSIMD
	predictInterCompoundRefHighBDToConvBufYResidentImpl = predictInterCompoundRefHighBDToConvBufYResidentGoSIMD
}

// compoundCopyGoSIMDCtx is the calling context of the unfiltered HBD
// compound copy. ref starts at the first sample and out is the CONV_BUF.
type compoundCopyGoSIMDCtx struct {
	out         []uint16
	ref         []byte
	refStr      int
	width       int
	height      int
	roundOffset int32
}

// compoundFilterGoSIMDCtx is the calling context of the one-dimensional
// HBD compound kernels. ref starts at the first sample of the trimmed tap span
// and kernel holds the taps zero-padded to filterTaps; taps counts the span.
type compoundFilterGoSIMDCtx struct {
	out         []uint16
	ref         []byte
	kernel      [filterTaps]int16
	refStr      int
	width       int
	height      int
	round0      int
	roundOffset int32
	taps        int
}

// compound2DGoSIMDCtx is the calling context of the separable 2D HBD
// compound kernel. ref starts at the first sample of the trimmed X span and the
// kernels hold their trimmed taps zero-padded to filterTaps.
type compound2DGoSIMDCtx struct {
	out    []uint16
	ref    []byte
	kernel [filterTaps]int16
	xKern  [filterTaps]int16
	refStr int
	width  int
	height int
	im     []int32
	imStr  int
	round0 int
	xBias  int32
	yBias  int32
	tapsX  int
	tapsY  int
}

// compoundBlendGoSIMDCtx is the calling context of the HBD distance
// weighted blend.
type compoundBlendGoSIMDCtx struct {
	dst    []byte // first destination sample (2 bytes/sample)
	src0   []uint16
	src1   []uint16
	dstStr int // destination stride in bytes
	width  int
	height int
	fwd    int
	bck    int
	maxVal int32 // clip upper bound (1<<bd)-1
}

func predictInterCompoundRefHighBDToConvBufCopyResidentGoSIMD(out []uint16, ref frame.Plane, refX int, refY int, width int, height int, round0 int, roundOffset int) {
	if (round0 != compoundRound0Bits && round0 != compoundRound0Bits+2) ||
		width < 8 || width%8 != 0 ||
		!planeRegionFits(ref, 2, refX, refY, width, height) {
		predictInterCompoundRefHighBDToConvBufCopyResidentPureGo(out, ref, refX, refY, width, height, round0, roundOffset)
		return
	}
	ctx := compoundCopyGoSIMDCtx{
		out:         out,
		ref:         ref.Pix[refY*ref.Stride+refX*2:],
		refStr:      ref.Stride,
		width:       width,
		height:      height,
		roundOffset: int32(roundOffset),
	}
	compoundCopyHighBDKernel(&ctx, filterBits-round0)
}

func predictInterCompoundRefHighBDToConvBufXResidentGoSIMD(out []uint16, ref frame.Plane, refX int, refY int, width int, height int, kernel [filterTaps]int16, round0 int, roundOffset int) {
	fo := filterTaps/2 - 1
	if (round0 != compoundRound0Bits && round0 != compoundRound0Bits+2) || width%4 != 0 {
		predictInterCompoundRefHighBDToConvBufXResident(out, ref, refX, refY, width, height, kernel, round0, roundOffset)
		return
	}
	lo, n := hbdTapSpan(&kernel)
	ctx := compoundFilterGoSIMDCtx{
		out:         out,
		ref:         ref.Pix[refY*ref.Stride+(refX-fo+lo)*2:],
		kernel:      hbdTrimTaps(&kernel, lo, n),
		refStr:      ref.Stride,
		width:       width,
		height:      height,
		round0:      round0,
		roundOffset: int32(roundOffset),
		taps:        n,
	}
	compoundXHighBDKernel(&ctx)
}

func predictInterCompoundRefHighBDToConvBufYResidentGoSIMD(out []uint16, ref frame.Plane, refX int, refY int, width int, height int, kernel [filterTaps]int16, round0 int, roundOffset int) {
	fo := filterTaps/2 - 1
	if (round0 != compoundRound0Bits && round0 != compoundRound0Bits+2) || width%4 != 0 {
		predictInterCompoundRefHighBDToConvBufYResident(out, ref, refX, refY, width, height, kernel, round0, roundOffset)
		return
	}
	lo, n := hbdTapSpan(&kernel)
	ctx := compoundFilterGoSIMDCtx{
		out:         out,
		ref:         ref.Pix[(refY-fo+lo)*ref.Stride+refX*2:],
		kernel:      hbdTrimTaps(&kernel, lo, n),
		refStr:      ref.Stride,
		width:       width,
		height:      height,
		round0:      round0,
		roundOffset: int32(roundOffset),
		taps:        n,
	}
	compoundYHighBDKernel(&ctx)
}

func predictInterCompoundRefHighBDToConvBuf2DResidentGoSIMD(out []uint16, ref frame.Plane, refX int, refY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16, round0 int, offsetBits int, bitDepth int, im *compoundIM) {
	foX := filterTaps/2 - 1
	foY := filterTaps/2 - 1
	if (round0 != compoundRound0Bits && round0 != compoundRound0Bits+2) ||
		width < 4 || width%4 != 0 ||
		!planeRegionFits(ref, 2, refX-foX, refY-foY, width+filterTaps, height+filterTaps-1) {
		predictInterCompoundRefHighBDToConvBuf2DResident(out, ref, refX, refY, width, height, xKernel, yKernel, round0, offsetBits, bitDepth, im)
		return
	}
	loX, nX := hbdTapSpan(&xKernel)
	loY, nY := hbdTapSpan(&yKernel)
	ctx := compound2DGoSIMDCtx{
		out:    out,
		ref:    ref.Pix[(refY-foY+loY)*ref.Stride+(refX-foX+loX)*2:],
		kernel: hbdTrimTaps(&yKernel, loY, nY),
		xKern:  hbdTrimTaps(&xKernel, loX, nX),
		refStr: ref.Stride,
		width:  width,
		height: height,
		im:     im[:],
		imStr:  maxBlockSize,
		round0: round0,
		xBias:  int32(1 << (bitDepth + filterBits - 1)),
		yBias:  int32(1 << offsetBits),
		tapsX:  nX,
		tapsY:  nY,
	}
	compound2DHighBDKernel(&ctx)
}

// predictInterCompoundRefHighBDToConvBuf2DClampedGoSIMD handles the
// edge-overhanging HBD compound 2D convolve. Following dav1d's reconstruction
// model (src/recon_tmpl.c mc(), src/mc_tmpl.c emu_edge_c), it materializes the
// clamped tap-window halo once (emuEdgeWindow16) and re-runs the resident Go
// SIMD joint-convolve over the resident window. Bit-identical to the pure-Go
// per-tap-clamping reference (predictInterCompoundRefHighBDToConvBuf2DClamped).
// Only width%4 != 0 shapes, which do not occur for AV1 inter blocks, take
// pure-Go. edge optionally carries the caller-owned halo window so the ~38KB
// buffer is not zero-filled per call; nil keeps per-call stack storage.
func predictInterCompoundRefHighBDToConvBuf2DClampedGoSIMD(out []uint16, ref frame.Plane, refX int, refY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16, round0 int, offsetBits int, bitDepth int, im *compoundIM, edge *emuEdge16Buf) {
	if width < 4 || width%4 != 0 {
		predictInterCompoundRefHighBDToConvBuf2DClamped(out, ref, refX, refY, width, height, xKernel, yKernel, round0, offsetBits, bitDepth, im)
		return
	}
	if edge != nil {
		emu, emuX, emuY := emuEdgeWindow16(ref, refX, refY, width, height, edge)
		predictInterCompoundRefHighBDToConvBuf2DResidentGoSIMD(out, emu, emuX, emuY, width, height, xKernel, yKernel, round0, offsetBits, bitDepth, im)
		return
	}
	var stackEdge emuEdge16Buf
	emu, emuX, emuY := emuEdgeWindow16(ref, refX, refY, width, height, &stackEdge)
	predictInterCompoundRefHighBDToConvBuf2DResidentGoSIMD(out, emu, emuX, emuY, width, height, xKernel, yKernel, round0, offsetBits, bitDepth, im)
}

// blendCompoundAvgHighBDGoSIMD is the HBD distance-weighted compound average.
// The scalar reference is blendCompoundAvgHighBD.
func blendCompoundAvgHighBDGoSIMD(dst frame.Plane, src0 []uint16, src1 []uint16, max uint16, dstX int, dstY int, width int, height int, fwdOffset int, bckOffset int, roundOffset int, roundBits int) {
	if (roundBits != 4 && roundBits != 2) || roundOffset <= 0 ||
		fwdOffset < 0 || fwdOffset > 16 || bckOffset < 0 || bckOffset > 16 ||
		height <= 0 || width%4 != 0 {
		blendCompoundAvgHighBD(dst, src0, src1, max, dstX, dstY, width, height, fwdOffset, bckOffset, roundOffset, roundBits)
		return
	}
	ctx := compoundBlendGoSIMDCtx{
		dst:    dst.Pix[dstY*dst.Stride+dstX*2:],
		src0:   src0,
		src1:   src1,
		dstStr: dst.Stride,
		width:  width,
		height: height,
		fwd:    fwdOffset,
		bck:    bckOffset,
		maxVal: int32(max),
	}
	blendCompoundAvgHighBDKernel(&ctx, roundOffset, roundBits)
}

// Go-native SIMD 8-bit single-prediction convolves (X, Y and 2D, plus the
// clamped and emulated-edge wrappers). The kernels compute the scalar formulas
// of convolveX8PureGo, convolveY8PureGo and convolve2D8PureGo. Even AV1 taps
// use int16 products of halved coefficients; uncommon coefficients use the
// original int32 path. The 2D intermediate is int16 in the common path.

// convolve8GoSIMDCtx is the calling context of the 8-bit convolve kernels. dst
// and ref are byte planes starting at the first destination pixel and the first
// sample of the trimmed tap span; kernel and xKern hold the trimmed taps
// zero-padded to filterTaps.
type convolve8GoSIMDCtx struct {
	dst    []byte
	ref    []byte
	kernel [filterTaps]int16 // vertical kernel (1D Y) or shared single kernel
	xKern  [filterTaps]int16 // 2D only: horizontal kernel
	dstStr int
	refStr int
	width  int
	height int
	tapsX  int // X tap count of the trimmed span (1D X and 2D)
	tapsY  int // Y tap count of the trimmed span (1D Y and 2D)
	im     []int32
	imStr  int // 2D only: intermediate row stride in int32 elements
}

// The single-prediction 8-bit rounding: the first stage is round3 and the final
// stage is round4 (1D) for the 1D kernels. The 2D vertical stage uses the
// libaom offset (yBias, roundOffset) with round1 = 11 and no final shift.
const (
	convolve8Round1Bits  = round1Bits
	convolve8YBias       = 1 << (8 + 2*filterBits - round0Bits)
	convolve8RoundOffset = (1 << (8 + 2*filterBits - round0Bits - round1Bits)) + (1 << (8 + 2*filterBits - round0Bits - round1Bits - 1))
	convolve8Bits        = 2*filterBits - round0Bits - round1Bits
)

// u8HalfTaps keeps the byte filter in signed 16-bit lanes. Every AV1 subpel
// coefficient is even. For a coefficient vector outside that domain, or one
// whose extreme sample sum could overflow int16, use the original wide path.
func u8HalfTaps(k *[filterTaps]int16, bias int) (t hbdTaps16, ok bool) {
	minSum, maxSum := bias, bias
	for i, v := range k {
		if v&1 != 0 {
			return t, false
		}
		h := int(v) / 2
		if h < 0 {
			minSum += 255 * h
		} else {
			maxSum += 255 * h
		}
		t[i] = archsimd.BroadcastInt16x8(int16(h))
	}
	return t, minSum-bias >= -32768 && minSum >= -32768 && maxSum <= 32767
}

// u8SumColHalf evaluates eight independent columns with one 16-bit MLA per
// tap. The caller has already proved that no partial or final sum wraps.
func u8SumColHalf(pix []byte, step int, t *hbdTaps16, n int) archsimd.Int16x8 {
	sum := archsimd.BroadcastInt16x8(0)
	for k := range n {
		sum = u8HalfMAC(u8Samples8(pix[k*step:]), t[k], sum)
	}
	return sum
}

func u8StoreHalf8(dst []byte, v archsimd.Int16x8) {
	var tmp [16]byte
	u8HalfSaturate(v).Store(tmp[:])
	copy(dst[:8], tmp[:8])
}

func convolveX8HalfKernel(ctx *convolve8GoSIMDCtx, taps *hbdTaps16) {
	need := (ctx.height-1)*ctx.refStr + ctx.width + 8
	if ctx.width%8 == 0 && ctx.refStr > 0 && need >= 0 && len(ctx.ref) >= need {
		convolveX8HalfResident(ctx)
		return
	}
	round3 := archsimd.BroadcastInt16x8(2)
	round4 := archsimd.BroadcastInt16x8(8)
	right2 := archsimd.BroadcastInt16x8(-2)
	right4 := archsimd.BroadcastInt16x8(-4)
	for y := range ctx.height {
		src := ctx.ref[y*ctx.refStr:]
		dst := ctx.dst[y*ctx.dstStr:]
		x := 0
		for ; x+8 <= ctx.width; x += 8 {
			sum := u8SumRowHalf(src[x:], taps, ctx.tapsX)
			u8StoreHalf8(dst[x:], u8Right16(u8Right16(sum.Add(round3), right2, 2).Add(round4), right4, 4))
		}
		if x < ctx.width {
			var win [32]byte
			copy(win[:], src[x:])
			sum := u8SumRowHalf(win[:], taps, ctx.tapsX)
			var tmp [16]byte
			u8HalfSaturate(u8Right16(u8Right16(sum.Add(round3), right2, 2).Add(round4), right4, 4)).Store(tmp[:])
			copy(dst[x:x+4], tmp[:4])
		}
	}
}

func convolveX8HalfResident(ctx *convolve8GoSIMDCtx) {
	c0 := archsimd.BroadcastInt16x8(ctx.kernel[0] / 2)
	c1 := archsimd.BroadcastInt16x8(ctx.kernel[1] / 2)
	c2 := archsimd.BroadcastInt16x8(ctx.kernel[2] / 2)
	c3 := archsimd.BroadcastInt16x8(ctx.kernel[3] / 2)
	c4 := archsimd.BroadcastInt16x8(ctx.kernel[4] / 2)
	c5 := archsimd.BroadcastInt16x8(ctx.kernel[5] / 2)
	c6 := archsimd.BroadcastInt16x8(ctx.kernel[6] / 2)
	c7 := archsimd.BroadcastInt16x8(ctx.kernel[7] / 2)
	round3 := archsimd.BroadcastInt16x8(2)
	round4 := archsimd.BroadcastInt16x8(8)
	right2 := archsimd.BroadcastInt16x8(-2)
	right4 := archsimd.BroadcastInt16x8(-4)
	// Each row is a width+8 byte window of the len(ctx.ref) >= need extent the
	// dispatching kernel checked; the loop bound proves each 16-byte load.
	for y := range ctx.height {
		dst := ctx.dst[y*ctx.dstStr:]
		src := ctx.ref[y*ctx.refStr:][: ctx.width+8 : ctx.width+8]
		for x := 0; x <= len(src)-16; x += 8 {
			a := archsimd.LoadUint8x16Array((*[16]byte)(src[x : x+16]))
			sum := u8HalfMAC(a.ExtendLo8ToUint16().BitsToInt16(), c0, round3)
			sum = u8HalfMAC(a.ConcatShiftBytesRight(a, 1).ExtendLo8ToUint16().BitsToInt16(), c1, sum)
			sum = u8HalfMAC(a.ConcatShiftBytesRight(a, 2).ExtendLo8ToUint16().BitsToInt16(), c2, sum)
			sum = u8HalfMAC(a.ConcatShiftBytesRight(a, 3).ExtendLo8ToUint16().BitsToInt16(), c3, sum)
			if ctx.tapsX > 4 {
				sum = u8HalfMAC(a.ConcatShiftBytesRight(a, 4).ExtendLo8ToUint16().BitsToInt16(), c4, sum)
				sum = u8HalfMAC(a.ConcatShiftBytesRight(a, 5).ExtendLo8ToUint16().BitsToInt16(), c5, sum)
			}
			if ctx.tapsX > 6 {
				sum = u8HalfMAC(a.ConcatShiftBytesRight(a, 6).ExtendLo8ToUint16().BitsToInt16(), c6, sum)
				sum = u8HalfMAC(a.ConcatShiftBytesRight(a, 7).ExtendLo8ToUint16().BitsToInt16(), c7, sum)
			}
			u8StoreHalf8(dst[x:], u8Right16(u8Right16(sum, right2, 2).Add(round4), right4, 4))
		}
	}
}

func convolveY8HalfKernel(ctx *convolve8GoSIMDCtx, taps *hbdTaps16) {
	// The resident fast path loads 16 bytes for each eight output columns.
	// Its final load is checked once here; exact-minimum slices use the safe
	// staged-load path below.
	need := (ctx.height+u8LoadedTaps(ctx.tapsY)-2)*ctx.refStr + ctx.width + 8
	if ctx.width%8 == 0 && ctx.refStr > 0 && need >= 0 && len(ctx.ref) >= need {
		convolveY8HalfResident(ctx)
		return
	}
	round := archsimd.BroadcastInt16x8(32)
	right6 := archsimd.BroadcastInt16x8(-6)
	for y := range ctx.height {
		src := ctx.ref[y*ctx.refStr:]
		dst := ctx.dst[y*ctx.dstStr:]
		x := 0
		for ; x+8 <= ctx.width; x += 8 {
			sum := u8SumColHalf(src[x:], ctx.refStr, taps, ctx.tapsY)
			u8StoreHalf8(dst[x:], u8Right16(sum.Add(round), right6, 6))
		}
		if x < ctx.width {
			var win [8 * 16]byte
			for k := range ctx.tapsY {
				copy(win[k*16:k*16+4], src[k*ctx.refStr+x:])
			}
			var tmp [16]byte
			u8HalfSaturate(u8Right16(u8SumColHalf(win[:], 16, taps, ctx.tapsY).Add(round), right6, 6)).Store(tmp[:])
			copy(dst[x:x+4], tmp[:4])
		}
	}
}

func convolveY8HalfResident(ctx *convolve8GoSIMDCtx) {
	c0 := archsimd.BroadcastInt16x8(ctx.kernel[0] / 2)
	c1 := archsimd.BroadcastInt16x8(ctx.kernel[1] / 2)
	c2 := archsimd.BroadcastInt16x8(ctx.kernel[2] / 2)
	c3 := archsimd.BroadcastInt16x8(ctx.kernel[3] / 2)
	c4 := archsimd.BroadcastInt16x8(ctx.kernel[4] / 2)
	c5 := archsimd.BroadcastInt16x8(ctx.kernel[5] / 2)
	c6 := archsimd.BroadcastInt16x8(ctx.kernel[6] / 2)
	c7 := archsimd.BroadcastInt16x8(ctx.kernel[7] / 2)
	round := archsimd.BroadcastInt16x8(32)
	right6 := archsimd.BroadcastInt16x8(-6)
	// Each tap row is a 16-byte window of the len(ctx.ref) >= need extent the
	// dispatching kernel checked.
	ref := ctx.ref
	step := ctx.refStr
	load := func(off int) archsimd.Int16x8 {
		return archsimd.LoadUint8x16Array((*[16]byte)(ref[off : off+16])).ExtendLo8ToUint16().BitsToInt16()
	}
	for y := range ctx.height {
		dst := ctx.dst[y*ctx.dstStr:]
		for x := 0; x < ctx.width; x += 8 {
			off := y*step + x
			sum := u8HalfMAC(load(off), c0, round)
			sum = u8HalfMAC(load(off+step), c1, sum)
			sum = u8HalfMAC(load(off+2*step), c2, sum)
			sum = u8HalfMAC(load(off+3*step), c3, sum)
			if ctx.tapsY > 4 {
				sum = u8HalfMAC(load(off+4*step), c4, sum)
				sum = u8HalfMAC(load(off+5*step), c5, sum)
			}
			if ctx.tapsY > 6 {
				sum = u8HalfMAC(load(off+6*step), c6, sum)
				sum = u8HalfMAC(load(off+7*step), c7, sum)
			}
			u8StoreHalf8(dst[x:], u8Right16(sum, right6, 6))
		}
	}
}

// u8LoadedTaps is the number of rows a resident vertical kernel loads for a
// trimmed span of n taps: its tap chains run in groups of 4, 6 or 8.
func u8LoadedTaps(n int) int {
	switch {
	case n <= 4:
		return 4
	case n <= 6:
		return 6
	}
	return 8
}

// convolveX8Kernel is the 1D horizontal 8-bit convolve: clip(round4(round3(sum))).
func convolveX8Kernel(ctx *convolve8GoSIMDCtx) {
	n := ctx.tapsX
	if half, ok := u8HalfTaps(&ctx.kernel, 10); ok {
		convolveX8HalfKernel(ctx, &half)
		return
	}
	taps := hbdBroadcastTaps16(&ctx.kernel)
	zero := archsimd.BroadcastInt32x4(0)
	maxV := archsimd.BroadcastInt32x4(255)
	width, height := ctx.width, ctx.height
	for y := range height {
		src := ctx.ref[y*ctx.refStr:]
		dst := ctx.dst[y*ctx.dstStr:]
		x := 0
		for ; x+8 <= width; x += 8 {
			lo, hi := u8SumRowX(src[x:], &taps, n)
			lo = hbdClip(hbdRound(hbdRound(lo, round0Bits), filterBits-round0Bits), zero, maxV)
			hi = hbdClip(hbdRound(hbdRound(hi, round0Bits), filterBits-round0Bits), zero, maxV)
			hbdStoreU8x8(dst[x:], lo, hi)
		}
		if x < width {
			var win [32]byte
			copy(win[:], src[x:])
			lo, _ := u8SumRowX(win[:], &taps, n)
			lo = hbdClip(hbdRound(hbdRound(lo, round0Bits), filterBits-round0Bits), zero, maxV)
			hbdStoreU8x4(dst[x:], lo)
		}
	}
}

// convolveY8Kernel is the 1D vertical 8-bit convolve: clip(round7(sum)).
func convolveY8Kernel(ctx *convolve8GoSIMDCtx) {
	n := ctx.tapsY
	if half, ok := u8HalfTaps(&ctx.kernel, 32); ok {
		convolveY8HalfKernel(ctx, &half)
		return
	}
	taps := hbdBroadcastTaps16(&ctx.kernel)
	zero := archsimd.BroadcastInt32x4(0)
	maxV := archsimd.BroadcastInt32x4(255)
	width, height := ctx.width, ctx.height
	for y := range height {
		src := ctx.ref[y*ctx.refStr:]
		dst := ctx.dst[y*ctx.dstStr:]
		x := 0
		for ; x+8 <= width; x += 8 {
			lo, hi := u8SumColY(src[x:], ctx.refStr, &taps, n)
			lo = hbdClip(hbdRound(lo, filterBits), zero, maxV)
			hi = hbdClip(hbdRound(hi, filterBits), zero, maxV)
			hbdStoreU8x8(dst[x:], lo, hi)
		}
		if x < width {
			var win [8 * 16]byte
			for k := range n {
				copy(win[k*16:k*16+4], src[k*ctx.refStr+x:])
			}
			lo, _ := u8SumColY(win[:], 16, &taps, n)
			lo = hbdClip(hbdRound(lo, filterBits), zero, maxV)
			hbdStoreU8x4(dst[x:], lo)
		}
	}
}

// convolve2D8Kernel is the separable 8-bit 2D convolve. The horizontal pass is
// the 8-bit compound horizontal pass (round3 with bias 2^(8+6)); the vertical
// pass rounds with round1, removes roundOffset and clips to the pixel range.
func convolve2D8Kernel(ctx *convolve8GoSIMDCtx) {
	nx, ny := ctx.tapsX, ctx.tapsY
	xTaps := hbdBroadcastTaps16(&ctx.xKern)
	yTaps := hbdBroadcastTaps32(&ctx.kernel)
	zero := archsimd.BroadcastInt32x4(0)
	maxV := archsimd.BroadcastInt32x4(255)
	yBias := archsimd.BroadcastInt32x4(convolve8YBias)
	rndOff := archsimd.BroadcastInt32x4(convolve8RoundOffset)
	xBias := archsimd.BroadcastInt32x4(1 << (8 + filterBits - 1))
	width, height := ctx.width, ctx.height

	u8HorizontalIM(ctx.ref, ctx.refStr, ctx.im, ctx.imStr, width, height+ny-1, &xTaps, nx, xBias)
	for y := range height {
		dst := ctx.dst[y*ctx.dstStr:]
		col := ctx.im[y*ctx.imStr:]
		x := 0
		for ; x+8 <= width; x += 8 {
			lo, hi := hbdSum8Int32(col[x:], ctx.imStr, &yTaps, ny)
			lo = hbdFinish2D(lo.Add(yBias), convolve8Round1Bits, rndOff, convolve8Bits, zero, maxV)
			hi = hbdFinish2D(hi.Add(yBias), convolve8Round1Bits, rndOff, convolve8Bits, zero, maxV)
			hbdStoreU8x8(dst[x:], lo, hi)
		}
		if x < width {
			lo := hbdSum4Int32(col[x:], ctx.imStr, &yTaps, ny)
			lo = hbdFinish2D(lo.Add(yBias), convolve8Round1Bits, rndOff, convolve8Bits, zero, maxV)
			hbdStoreU8x4(dst[x:], lo)
		}
	}
}

// convolve2D8HalfKernel keeps the horizontal intermediate in signed 16-bit
// storage. The vertical products still widen to 32 bits as required by AV1.
// The wrapper supplies at least height+roundedTapCount-1 rows of 128 samples.
// Every 8-sample vertical load stays within that buffer, including width 4.
func convolve2D8HalfKernel(ctx *convolve8GoSIMDCtx, im []int16, xTaps *hbdTaps16) {
	c0 := archsimd.BroadcastInt16x8(ctx.kernel[0])
	c1 := archsimd.BroadcastInt16x8(ctx.kernel[1])
	c2 := archsimd.BroadcastInt16x8(ctx.kernel[2])
	c3 := archsimd.BroadcastInt16x8(ctx.kernel[3])
	c4 := archsimd.BroadcastInt16x8(ctx.kernel[4])
	c5 := archsimd.BroadcastInt16x8(ctx.kernel[5])
	c6 := archsimd.BroadcastInt16x8(ctx.kernel[6])
	c7 := archsimd.BroadcastInt16x8(ctx.kernel[7])
	xRound := archsimd.BroadcastInt16x8(8194) // half of 16384 plus round3 bias
	right2 := archsimd.BroadcastInt16x8(-2)
	yRound := archsimd.BroadcastInt32x4(convolve8YBias + (1 << (round1Bits - 1)))
	right11 := archsimd.BroadcastInt32x4(-round1Bits)
	rndOff := archsimd.BroadcastInt32x4(convolve8RoundOffset)
	zero := archsimd.BroadcastInt32x4(0)
	maxV := archsimd.BroadcastInt32x4(255)
	rows := ctx.height + ctx.tapsY - 1
	need := (rows-1)*ctx.refStr + ctx.width + 8
	if ctx.width%8 == 0 && ctx.refStr > 0 && need >= 0 && len(ctx.ref) >= need {
		u8HorizontalHalfIMResident(ctx, im)
	} else {
		for y := 0; y < rows; y++ {
			src := ctx.ref[y*ctx.refStr:]
			row := im[y*maxBlockSize:]
			x := 0
			for ; x+8 <= ctx.width; x += 8 {
				u8Right16(u8SumRowHalf(src[x:], xTaps, ctx.tapsX).Add(xRound), right2, 2).Store(row[x : x+8])
			}
			if x < ctx.width {
				var win [16]byte
				copy(win[:], src[x:])
				var out [8]int16
				u8Right16(u8SumRowHalf(win[:], xTaps, ctx.tapsX).Add(xRound), right2, 2).Store(out[:])
				copy(row[x:x+4], out[:4])
			}
		}
	}
	// im rows (stride maxBlockSize): the height+taps-1 rows the wrapper supplies.
	load := func(y, x int) archsimd.Int16x8 {
		o := y*maxBlockSize + x
		return archsimd.LoadInt16x8Array((*[8]int16)(im[o : o+8]))
	}
	for y := range ctx.height {
		dst := ctx.dst[y*ctx.dstStr:]
		for x := 0; x < ctx.width; x += 8 {
			v := load(y, x)
			lo, hi := u8WideMAC8(zero, zero, v, c0)
			v = load(y+1, x)
			lo, hi = u8WideMAC8(lo, hi, v, c1)
			v = load(y+2, x)
			lo, hi = u8WideMAC8(lo, hi, v, c2)
			v = load(y+3, x)
			lo, hi = u8WideMAC8(lo, hi, v, c3)
			if ctx.tapsY > 4 {
				v = load(y+4, x)
				lo, hi = u8WideMAC8(lo, hi, v, c4)
				v = load(y+5, x)
				lo, hi = u8WideMAC8(lo, hi, v, c5)
			}
			if ctx.tapsY > 6 {
				v = load(y+6, x)
				lo, hi = u8WideMAC8(lo, hi, v, c6)
				v = load(y+7, x)
				lo, hi = u8WideMAC8(lo, hi, v, c7)
			}
			lo = u8Right32(lo.Add(yRound), right11, round1Bits).Sub(rndOff).Max(zero).Min(maxV)
			hi = u8Right32(hi.Add(yRound), right11, round1Bits).Sub(rndOff).Max(zero).Min(maxV)
			packed := u8PackClipped32(lo, hi)
			var tmp [16]byte
			packed.Store(tmp[:])
			if x+8 <= ctx.width {
				copy(dst[x:x+8], tmp[:8])
			} else {
				copy(dst[x:ctx.width], tmp[:ctx.width-x])
			}
		}
	}
}

func u8HorizontalHalfIMResident(ctx *convolve8GoSIMDCtx, im []int16) {
	c0 := archsimd.BroadcastInt16x8(ctx.xKern[0] / 2)
	c1 := archsimd.BroadcastInt16x8(ctx.xKern[1] / 2)
	c2 := archsimd.BroadcastInt16x8(ctx.xKern[2] / 2)
	c3 := archsimd.BroadcastInt16x8(ctx.xKern[3] / 2)
	c4 := archsimd.BroadcastInt16x8(ctx.xKern[4] / 2)
	c5 := archsimd.BroadcastInt16x8(ctx.xKern[5] / 2)
	c6 := archsimd.BroadcastInt16x8(ctx.xKern[6] / 2)
	c7 := archsimd.BroadcastInt16x8(ctx.xKern[7] / 2)
	xRound := archsimd.BroadcastInt16x8(8194)
	right2 := archsimd.BroadcastInt16x8(-2)
	// Each row is a width+8 byte window of the len(ctx.ref) >= need extent the
	// dispatching kernel checked; the loop bound proves each 16-byte load.
	for y := 0; y < ctx.height+ctx.tapsY-1; y++ {
		row := im[y*maxBlockSize:]
		src := ctx.ref[y*ctx.refStr:][: ctx.width+8 : ctx.width+8]
		for x := 0; x <= len(src)-16; x += 8 {
			a := archsimd.LoadUint8x16Array((*[16]byte)(src[x : x+16]))
			sum := u8HalfMAC(a.ExtendLo8ToUint16().BitsToInt16(), c0, xRound)
			sum = u8HalfMAC(a.ConcatShiftBytesRight(a, 1).ExtendLo8ToUint16().BitsToInt16(), c1, sum)
			sum = u8HalfMAC(a.ConcatShiftBytesRight(a, 2).ExtendLo8ToUint16().BitsToInt16(), c2, sum)
			sum = u8HalfMAC(a.ConcatShiftBytesRight(a, 3).ExtendLo8ToUint16().BitsToInt16(), c3, sum)
			if ctx.tapsX > 4 {
				sum = u8HalfMAC(a.ConcatShiftBytesRight(a, 4).ExtendLo8ToUint16().BitsToInt16(), c4, sum)
				sum = u8HalfMAC(a.ConcatShiftBytesRight(a, 5).ExtendLo8ToUint16().BitsToInt16(), c5, sum)
			}
			if ctx.tapsX > 6 {
				sum = u8HalfMAC(a.ConcatShiftBytesRight(a, 6).ExtendLo8ToUint16().BitsToInt16(), c6, sum)
				sum = u8HalfMAC(a.ConcatShiftBytesRight(a, 7).ExtendLo8ToUint16().BitsToInt16(), c7, sum)
			}
			u8Right16(sum, right2, 2).Store(row[x : x+8])
		}
	}
}

// convolveX8GoSIMD is the 1D horizontal 8-bit convolve over a resident window.
func convolveX8GoSIMD(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, kernel [filterTaps]int16) {
	if width%4 != 0 {
		convolveX8PureGo(dst, ref, dstX, dstY, refX, refY, width, height, kernel)
		return
	}
	fo := filterTaps/2 - 1
	lo, n := hbdTapSpan(&kernel)
	ctx := convolve8GoSIMDCtx{
		dst:    dst.Pix[dstY*dst.Stride+dstX:],
		ref:    ref.Pix[refY*ref.Stride+refX-fo+lo:],
		kernel: hbdTrimTaps(&kernel, lo, n),
		dstStr: dst.Stride,
		refStr: ref.Stride,
		width:  width,
		height: height,
		tapsX:  n,
	}
	convolveX8Kernel(&ctx)
}

// convolveY8GoSIMD is the 1D vertical 8-bit convolve over a resident window.
func convolveY8GoSIMD(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, kernel [filterTaps]int16) {
	if width%4 != 0 {
		convolveY8PureGo(dst, ref, dstX, dstY, refX, refY, width, height, kernel)
		return
	}
	fo := filterTaps/2 - 1
	lo, n := hbdTapSpan(&kernel)
	ctx := convolve8GoSIMDCtx{
		dst:    dst.Pix[dstY*dst.Stride+dstX:],
		ref:    ref.Pix[(refY-fo+lo)*ref.Stride+refX:],
		kernel: hbdTrimTaps(&kernel, lo, n),
		dstStr: dst.Stride,
		refStr: ref.Stride,
		width:  width,
		height: height,
		tapsY:  n,
	}
	convolveY8Kernel(&ctx)
}

// convolve2D8GoSIMD is convolve2D8GoSIMDWithScratch without caller scratch.
func convolve2D8GoSIMD(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16) {
	convolve2D8GoSIMDWithScratch(dst, ref, dstX, dstY, refX, refY, width, height, xKernel, yKernel, nil)
}

// convolve2D8GoSIMDWithScratch is the separable 8-bit 2D convolve. With scratch
// the int32 intermediate lives in the scratch (imHBD) and is not zero-filled
// per call.
func convolve2D8GoSIMDWithScratch(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16, scratch *ConvolveScratch) {
	if width%4 != 0 {
		convolve2D8PureGo(dst, ref, dstX, dstY, refX, refY, width, height, xKernel, yKernel)
		return
	}
	foX := filterTaps/2 - 1
	foY := filterTaps/2 - 1
	loX, nX := hbdTapSpan(&xKernel)
	loY, nY := hbdTapSpan(&yKernel)
	trimX := hbdTrimTaps(&xKernel, loX, nX)
	trimY := hbdTrimTaps(&yKernel, loY, nY)
	if half, ok := u8HalfTaps(&trimX, 8194); ok {
		ctx := convolve8GoSIMDCtx{
			dst:    dst.Pix[dstY*dst.Stride+dstX:],
			ref:    ref.Pix[(refY-foY+loY)*ref.Stride+(refX-foX+loX):],
			kernel: trimY, xKern: trimX, dstStr: dst.Stride, refStr: ref.Stride,
			width: width, height: height, tapsX: nX, tapsY: nY,
		}
		if scratch != nil {
			convolve2D8HalfKernel(&ctx, scratch.im[:], &half)
		} else {
			loadTaps := nY
			if loadTaps < 4 {
				loadTaps = 4
			} else if loadTaps&1 != 0 {
				loadTaps++
			}
			needRows := height + loadTaps - 1
			switch {
			case needRows <= 8:
				var local [8 * maxBlockSize]int16
				convolve2D8HalfKernel(&ctx, local[:], &half)
			case needRows <= 32:
				var local [32 * maxBlockSize]int16
				convolve2D8HalfKernel(&ctx, local[:], &half)
			default:
				var local [(maxBlockSize + filterTaps - 1) * maxBlockSize]int16
				convolve2D8HalfKernel(&ctx, local[:], &half)
			}
		}
		return
	}
	var im []int32
	if scratch != nil {
		im = scratch.imHBD[:]
	} else {
		var local [(maxBlockSize + filterTaps - 1) * maxBlockSize]int32
		im = local[:]
	}
	ctx := convolve8GoSIMDCtx{
		dst:    dst.Pix[dstY*dst.Stride+dstX:],
		ref:    ref.Pix[(refY-foY+loY)*ref.Stride+(refX-foX+loX):],
		kernel: trimY,
		xKern:  trimX,
		dstStr: dst.Stride,
		refStr: ref.Stride,
		width:  width,
		height: height,
		tapsX:  nX,
		tapsY:  nY,
		im:     im,
		imStr:  maxBlockSize,
	}
	convolve2D8Kernel(&ctx)
}

// The *ClampedGoSIMD wrappers handle the edge-block convolves. When the whole tap
// window happens to lie inside the reference plane the clamp is a no-op and the
// result is bit-identical to the non-clamped fast path, so we route to the GoSIMD
// kernel. Otherwise (a tap coordinate genuinely falls off the frame) we fall back
// to the pure-Go clamped reference, which clamps each tap coordinate. This keeps
// the kernel free of per-tap coordinate clamping while still accelerating the
// common case where an "edge" block's halo is actually resident.

func convolveX8ClampedGoSIMD(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, kernel [filterTaps]int16) {
	fo := filterTaps/2 - 1
	neonWidth := width == 4 || (width >= 8 && width%8 == 0)
	// Both the width>=8 and width-4 horizontal kernels physically load one sample
	// past the consumed tap window: the width>=8 loop does a 16-byte ld1 for its
	// last 8-column group, and the width-4 kernel loads bytes 8..11 to fill the
	// slide register. That trailing sample never feeds an output, but it must be
	// resident, so reserve width+filterTaps samples in the fit check.
	haloW := width + filterTaps
	if neonWidth && planeRegionFits(ref, 1, refX-fo, refY, haloW, height) {
		convolveX8GoSIMD(dst, ref, dstX, dstY, refX, refY, width, height, kernel)
		return
	}
	if convolveX8HorizontalEdgeGoSIMD(dst, ref, dstX, dstY, refX, refY, width, height, kernel) {
		return
	}
	convolveX8ClampedPureGo(dst, ref, dstX, dstY, refX, refY, width, height, kernel)
}

func convolveY8ClampedGoSIMD(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, kernel [filterTaps]int16) {
	fo := filterTaps/2 - 1
	neonWidth := width == 4 || (width >= 8 && width%8 == 0)
	if neonWidth && planeRegionFits(ref, 1, refX, refY-fo, width, height+filterTaps-1) {
		convolveY8GoSIMD(dst, ref, dstX, dstY, refX, refY, width, height, kernel)
		return
	}
	if convolveY8VerticalEdgeGoSIMD(dst, ref, dstX, dstY, refX, refY, width, height, kernel) {
		return
	}
	convolveY8ClampedPureGo(dst, ref, dstX, dstY, refX, refY, width, height, kernel)
}

// convolveX8ClampedGoSIMDWithScratch is convolveX8ClampedGoSIMD with a dav1d-style
// emu_edge fast path (src/recon_tmpl.c mc()): when the tap window is not
// resident, the clamped halo is materialized once into the caller scratch and
// the plain X kernel runs over it.
func convolveX8ClampedGoSIMDWithScratch(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, kernel [filterTaps]int16, scratch *ConvolveScratch) {
	fo := filterTaps/2 - 1
	neonWidth := width == 4 || (width >= 8 && width%8 == 0)
	if scratch != nil && neonWidth {
		// The horizontal kernels load one sample past the consumed tap window
		// (16-byte ld1 for width>=8; bytes 8..11 for width-4); reserve
		// width+filterTaps so that trailing sample stays resident.
		haloW := width + filterTaps
		if !planeRegionFits(ref, 1, refX-fo, refY, haloW, height) {
			emu, emuX, emuY := emuEdgeWindow(ref, refX, refY, width, height, &scratch.edge)
			convolveX8Impl(dst, emu, dstX, dstY, emuX, emuY, width, height, kernel)
			return
		}
	}
	convolveX8ClampedGoSIMD(dst, ref, dstX, dstY, refX, refY, width, height, kernel)
}

// convolveY8ClampedGoSIMDWithScratch is convolveY8ClampedGoSIMD with the same
// emu_edge fast path as convolveX8ClampedGoSIMDWithScratch.
func convolveY8ClampedGoSIMDWithScratch(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, kernel [filterTaps]int16, scratch *ConvolveScratch) {
	fo := filterTaps/2 - 1
	neonWidth := width == 4 || (width >= 8 && width%8 == 0)
	if scratch != nil && neonWidth &&
		!planeRegionFits(ref, 1, refX, refY-fo, width, height+filterTaps-1) {
		emu, emuX, emuY := emuEdgeWindow(ref, refX, refY, width, height, &scratch.edge)
		convolveY8Impl(dst, emu, dstX, dstY, emuX, emuY, width, height, kernel)
		return
	}
	convolveY8ClampedGoSIMD(dst, ref, dstX, dstY, refX, refY, width, height, kernel)
}

func convolveX8HorizontalEdgeGoSIMD(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, kernel [filterTaps]int16) bool {
	fo := filterTaps/2 - 1
	if !planeRegionFits(ref, 1, 0, refY, ref.Width, height) {
		return false
	}
	xLo, xHi := clampedXInterior(refX-fo, filterTaps, ref.Width, width)
	if xHi <= xLo {
		return false
	}
	if xLo > 0 {
		convolveX8ClampedPureGo(dst, ref, dstX, dstY, refX, refY, xLo, height, kernel)
	}
	start := xLo
	didGoSIMD := false
	for start < xHi {
		remaining := xHi - start
		if remaining >= 8 {
			chunk := remaining &^ 7
			// clampedXInterior only guarantees the consumed taps are resident;
			// the last 8-column group's 16-byte load reaches one sample further
			// (chunk+filterTaps in total). Shrink the GoSIMD span by whole groups
			// until that physical reach is in-bounds so the kernel never over-reads;
			// any dropped columns fall to the scalar tail below.
			for chunk >= 8 && !planeRegionFits(ref, 1, refX+start-fo, refY, chunk+filterTaps, height) {
				chunk -= 8
			}
			if chunk < 8 {
				break
			}
			convolveX8GoSIMD(dst, ref, dstX+start, dstY, refX+start, refY, chunk, height, kernel)
			start += chunk
			didGoSIMD = true
			continue
		}
		if remaining >= 4 {
			if !planeRegionFits(ref, 1, refX+start-fo, refY, filterTaps+4, height) {
				break
			}
			convolveX8GoSIMD(dst, ref, dstX+start, dstY, refX+start, refY, 4, height, kernel)
			start += 4
			didGoSIMD = true
			continue
		}
		break
	}
	if !didGoSIMD {
		return false
	}
	if start < xHi {
		convolveX8ClampedPureGo(dst, ref, dstX+start, dstY, refX+start, refY, xHi-start, height, kernel)
	}
	if xHi < width {
		convolveX8ClampedPureGo(dst, ref, dstX+xHi, dstY, refX+xHi, refY, width-xHi, height, kernel)
	}
	return true
}

func convolveY8VerticalEdgeGoSIMD(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, kernel [filterTaps]int16) bool {
	if !(width == 4 || (width >= 8 && width%8 == 0)) {
		return false
	}
	if !planeRegionFits(ref, 1, refX, 0, width, ref.Height) {
		return false
	}
	fo := filterTaps/2 - 1
	yLo, yHi := clampedXInterior(refY-fo, filterTaps, ref.Height, height)
	if yHi <= yLo {
		return false
	}
	if yLo > 0 {
		convolveY8ClampedPureGo(dst, ref, dstX, dstY, refX, refY, width, yLo, kernel)
	}
	convolveY8GoSIMD(dst, ref, dstX, dstY+yLo, refX, refY+yLo, width, yHi-yLo, kernel)
	if yHi < height {
		convolveY8ClampedPureGo(dst, ref, dstX, dstY+yHi, refX, refY+yHi, width, height-yHi, kernel)
	}
	return true
}

func convolve2D8ClampedGoSIMD(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16) {
	convolve2D8ClampedGoSIMDWithScratch(dst, ref, dstX, dstY, refX, refY, width, height, xKernel, yKernel, nil)
}

func convolve2D8ClampedGoSIMDWithScratch(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16, scratch *ConvolveScratch) {
	foX := filterTaps/2 - 1
	foY := filterTaps/2 - 1
	neonWidth := width == 4 || (width >= 8 && width%8 == 0)
	// The horizontal pass loads one sample past the consumed tap window (16-byte
	// ld1 for width>=8; bytes 8..11 for width-4); reserve width+filterTaps
	// columns so that trailing sample stays resident.
	haloW := width + filterTaps
	if neonWidth &&
		planeRegionFits(ref, 1, refX-foX, refY-foY, haloW, height+filterTaps-1) {
		convolve2D8GoSIMDWithScratch(dst, ref, dstX, dstY, refX, refY, width, height, xKernel, yKernel, scratch)
		return
	}
	if neonWidth && scratch != nil {
		// dav1d src/recon_tmpl.c mc(): out-of-bounds references materialize
		// the clamped halo once (emu_edge) and run the plain 8tap kernel over
		// the resident window.
		emu, emuX, emuY := emuEdgeWindow(ref, refX, refY, width, height, &scratch.edge)
		convolve2D8GoSIMDWithScratch(dst, emu, dstX, dstY, emuX, emuY, width, height, xKernel, yKernel, scratch)
		return
	}
	if convolve2D8ClampedEdgeSplitGoSIMDWithScratch(dst, ref, dstX, dstY, refX, refY, width, height, xKernel, yKernel, scratch) {
		return
	}
	convolve2D8ClampedPureGoWithScratch(dst, ref, dstX, dstY, refX, refY, width, height, xKernel, yKernel, scratch)
}

func convolve2D8ClampedEdgeSplitGoSIMDWithScratch(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16, scratch *ConvolveScratch) bool {
	foX := filterTaps/2 - 1
	foY := filterTaps/2 - 1
	xLo, xHi := clampedXInterior(refX-foX, filterTaps, ref.Width, width)
	yLo, yHi := clampedXInterior(refY-foY, filterTaps, ref.Height, height)
	if xHi <= xLo || yHi <= yLo {
		return false
	}

	var starts [16]int
	var widths [16]int
	n := 0
	for start := xLo; start < xHi && n < len(starts); {
		remaining := xHi - start
		if remaining >= 8 {
			chunk := remaining &^ 7
			// The 2D horizontal pass loads 16 bytes for its last 8-column group,
			// reaching chunk+filterTaps source samples across the midH+filterTaps-1
			// halo rows. Shrink by whole groups until that physical reach is
			// resident so the kernel never over-reads; dropped columns go to the
			// scalar tail below.
			for chunk >= 8 && !planeRegionFits(ref, 1, refX+start-foX, refY+yLo-foY, chunk+filterTaps, yHi-yLo+filterTaps-1) {
				chunk -= 8
			}
			if chunk < 8 {
				break
			}
			starts[n], widths[n] = start, chunk
			n++
			start += chunk
			continue
		}
		if remaining >= 4 {
			const w4Halo = 4 + filterTaps
			if !planeRegionFits(ref, 1, refX+start-foX, refY+yLo-foY, w4Halo, yHi-yLo+filterTaps-1) {
				break
			}
			starts[n], widths[n] = start, 4
			n++
			start += 4
			continue
		}
		break
	}
	if n == 0 {
		return false
	}

	if yLo > 0 {
		convolve2D8ClampedPureGoWithScratch(dst, ref, dstX, dstY, refX, refY, width, yLo, xKernel, yKernel, scratch)
	}
	midH := yHi - yLo
	if xLo > 0 {
		convolve2D8ClampedPureGoWithScratch(dst, ref, dstX, dstY+yLo, refX, refY+yLo, xLo, midH, xKernel, yKernel, scratch)
	}
	coveredHi := xLo
	for i := 0; i < n; i++ {
		start := starts[i]
		chunk := widths[i]
		convolve2D8GoSIMDWithScratch(dst, ref, dstX+start, dstY+yLo, refX+start, refY+yLo, chunk, midH, xKernel, yKernel, scratch)
		coveredHi = start + chunk
	}
	if coveredHi < width {
		convolve2D8ClampedPureGoWithScratch(dst, ref, dstX+coveredHi, dstY+yLo, refX+coveredHi, refY+yLo, width-coveredHi, midH, xKernel, yKernel, scratch)
	}
	if yHi < height {
		convolve2D8ClampedPureGoWithScratch(dst, ref, dstX, dstY+yHi, refX, refY+yHi, width, height-yHi, xKernel, yKernel, scratch)
	}
	return true
}

// init binds the 8-bit single-prediction convolves to the Go SIMD kernels. The
// scalar defaults in convolve_dispatch.go stay in place when Go SIMD is not
// available.
func init() {
	if !hbdSIMDAvailable() {
		return
	}
	convolveX8Impl = convolveX8GoSIMD
	convolveY8Impl = convolveY8GoSIMD
	convolve2D8Impl = convolve2D8GoSIMD
	convolve2D8WithScratchImpl = convolve2D8GoSIMDWithScratch
	convolveX8ClampedImpl = convolveX8ClampedGoSIMD
	convolveY8ClampedImpl = convolveY8ClampedGoSIMD
	convolve2D8ClampedImpl = convolve2D8ClampedGoSIMD
	convolve2D8ClampedWithScratchImpl = convolve2D8ClampedGoSIMDWithScratch
	convolveX8ClampedWithScratchImpl = convolveX8ClampedGoSIMDWithScratch
	convolveY8ClampedWithScratchImpl = convolveY8ClampedGoSIMDWithScratch
}

// init binds the high-bit-depth convolves to the Go SIMD kernels. The clamped
// variants run the resident kernels over an emulated-edge window.
func init() {
	if !hbdSIMDAvailable() {
		return
	}
	convolveXHighBDImpl = convolveXHighBDGoSIMD
	convolveYHighBDImpl = convolveYHighBDGoSIMD
	convolve2DHighBDImpl = convolve2DHighBDGoSIMD
	convolveXHighBDClampedImpl = convolveXHighBDClampedGoSIMD
	convolveYHighBDClampedImpl = convolveYHighBDClampedGoSIMD
	convolve2DHighBDClampedImpl = convolve2DHighBDClampedGoSIMD
	convolve2DHighBDWithScratchImpl = convolve2DHighBDGoSIMDWithScratch
	convolve2DHighBDClampedWithScratchImpl = convolve2DHighBDClampedGoSIMDWithScratch
}

// Go SIMD high-bit-depth (10/12-bit) convolve wrappers. Samples are
// little-endian uint16; the round-0/round-1 shift amounts are bit-depth
// dependent (highBDRoundBits): round0=3,round1=11 for bd<=10 and round0=5,
// round1=9 for bd==12. The shifts are applied as arithmetic right shifts of
// int32 lanes, so a single code path covers every bit depth.
//
// The kernels handle every width that is a multiple of 4 (the tails of 4 and
// 12-lane blocks use a four-sample window). Widths that are not a multiple of 4
// (none occur in AV1) fall back to pure-Go. Final clipping is a vector
// max(0)/min(max) clamp because the upper bound is bit-depth dependent, not a
// fixed 8-/16-bit saturation.

// convolveHighBDGoSIMDCtx is the calling context of the high-bit-depth convolve
// kernels. dst and ref are little-endian byte planes starting at the first
// destination sample and the first sample of the trimmed tap span; kernel and
// xKern hold the trimmed taps zero-padded to filterTaps.
type convolveHighBDGoSIMDCtx struct {
	dst    []byte
	ref    []byte
	kernel [filterTaps]int16 // vertical kernel (1D Y) or shared single kernel
	xKern  [filterTaps]int16 // 2D only: horizontal kernel
	dstStr int
	refStr int
	width  int
	height int
	im     []int32 // 2D only: int32 intermediate scratch
	imStr  int     // 2D only: intermediate row stride in int32 elements
	round0 int     // round-0 shift bits (1D Y: single FILTER_BITS shift)
	round1 int     // round-1 shift bits (2D vertical) or final "bits" (1D X)
	bits   int     // 2D only: final round shift
	rndOff int32   // 2D only: roundOffset
	xBias  int32   // 2D only: 1 << (bd + FILTER_BITS - 1)
	yBias  int32   // 2D only: 1 << offsetBits
	maxVal int32   // clip upper bound (1<<bd)-1
	tapsX  int     // X tap count of the trimmed span (1D X and 2D)
	tapsY  int     // Y tap count of the trimmed span (1D Y and 2D)
}

func convolveXHighBDGoSIMD(dst frame.Plane, ref frame.Plane, bitDepth uint8, max uint16, dstX int, dstY int, refX int, refY int, width int, height int, kernel [filterTaps]int16) {
	if width < 4 || width%4 != 0 {
		convolveXHighBDPureGo(dst, ref, bitDepth, max, dstX, dstY, refX, refY, width, height, kernel)
		return
	}
	fo := filterTaps/2 - 1
	round0, _ := highBDRoundBits(bitDepth)
	bits := filterBits - round0
	lo, n := hbdTapSpan(&kernel)
	ctx := convolveHighBDGoSIMDCtx{
		dst:    dst.Pix[dstY*dst.Stride+dstX*2:],
		ref:    ref.Pix[refY*ref.Stride+(refX-fo+lo)*2:],
		kernel: hbdTrimTaps(&kernel, lo, n),
		dstStr: dst.Stride,
		refStr: ref.Stride,
		width:  width,
		height: height,
		round0: round0,
		round1: bits,
		maxVal: int32(max),
		tapsX:  n,
	}
	convolveXHighBDKernel(&ctx)
}

func convolveYHighBDGoSIMD(dst frame.Plane, ref frame.Plane, bitDepth uint8, max uint16, dstX int, dstY int, refX int, refY int, width int, height int, kernel [filterTaps]int16) {
	_ = bitDepth
	if width < 4 || width%4 != 0 {
		convolveYHighBDPureGo(dst, ref, bitDepth, max, dstX, dstY, refX, refY, width, height, kernel)
		return
	}
	fo := filterTaps/2 - 1
	lo, n := hbdTapSpan(&kernel)
	ctx := convolveHighBDGoSIMDCtx{
		dst:    dst.Pix[dstY*dst.Stride+dstX*2:],
		ref:    ref.Pix[(refY-fo+lo)*ref.Stride+refX*2:],
		kernel: hbdTrimTaps(&kernel, lo, n),
		dstStr: dst.Stride,
		refStr: ref.Stride,
		width:  width,
		height: height,
		round0: filterBits, // single rounding shift by FILTER_BITS
		maxVal: int32(max),
		tapsY:  n,
	}
	convolveYHighBDKernel(&ctx)
}

// convolve2DHighBDGoSIMDIMStride matches the imStride const in
// convolve2DHighBDPureGo (maxBlockSize) so the vertical pass walks the same rows.
const convolve2DHighBDGoSIMDIMStride = maxBlockSize

func convolve2DHighBDGoSIMD(dst frame.Plane, ref frame.Plane, bitDepth uint8, max uint16, dstX int, dstY int, refX int, refY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16) {
	convolve2DHighBDGoSIMDWithScratch(dst, ref, bitDepth, max, dstX, dstY, refX, refY, width, height, xKernel, yKernel, nil)
}

// convolve2DHighBDGoSIMDWithScratch is convolve2DHighBDGoSIMD with optional
// caller-owned scratch for the int32 intermediate block. With scratch the
// per-call zero-fill of the ~69KB stack array disappears (the kernel overwrites
// every intermediate sample it reads); without scratch the stack array keeps
// the historical behavior for callers outside the framework decode path.
func convolve2DHighBDGoSIMDWithScratch(dst frame.Plane, ref frame.Plane, bitDepth uint8, max uint16, dstX int, dstY int, refX int, refY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16, scratch *ConvolveScratch) {
	if width < 4 || width%4 != 0 {
		convolve2DHighBDPureGoWithScratch(dst, ref, bitDepth, max, dstX, dstY, refX, refY, width, height, xKernel, yKernel, scratch)
		return
	}
	if scratch != nil {
		convolve2DHighBDGoSIMDWithIM(dst, ref, bitDepth, max, dstX, dstY, refX, refY, width, height, xKernel, yKernel, scratch.imHBD[:])
		return
	}
	var im [(maxBlockSize + filterTaps - 1) * convolve2DHighBDGoSIMDIMStride]int32
	convolve2DHighBDGoSIMDWithIM(dst, ref, bitDepth, max, dstX, dstY, refX, refY, width, height, xKernel, yKernel, im[:])
}

func convolve2DHighBDGoSIMDWithIM(dst frame.Plane, ref frame.Plane, bitDepth uint8, max uint16, dstX int, dstY int, refX int, refY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16, im []int32) {
	foX := filterTaps/2 - 1
	foY := filterTaps/2 - 1
	round0, round1 := highBDRoundBits(bitDepth)
	offsetBits := int(bitDepth) + 2*filterBits - round0
	roundOffset := (1 << (offsetBits - round1)) + (1 << (offsetBits - round1 - 1))
	bits := 2*filterBits - round0 - round1
	xBias := 1 << (int(bitDepth) + filterBits - 1)
	yBias := 1 << offsetBits

	loX, nX := hbdTapSpan(&xKernel)
	loY, nY := hbdTapSpan(&yKernel)
	ctx := convolveHighBDGoSIMDCtx{
		dst:    dst.Pix[dstY*dst.Stride+dstX*2:],
		ref:    ref.Pix[(refY-foY+loY)*ref.Stride+(refX-foX+loX)*2:],
		kernel: hbdTrimTaps(&yKernel, loY, nY),
		xKern:  hbdTrimTaps(&xKernel, loX, nX),
		tapsX:  nX,
		tapsY:  nY,
		dstStr: dst.Stride,
		refStr: ref.Stride,
		width:  width,
		height: height,
		im:     im,
		imStr:  convolve2DHighBDGoSIMDIMStride,
		round0: round0,
		round1: round1,
		bits:   bits,
		rndOff: int32(roundOffset),
		xBias:  int32(xBias),
		yBias:  int32(yBias),
		maxVal: int32(max),
	}
	convolve2DHighBDKernel(&ctx)
}

// The HBD clamped wrappers mirror the 8-bit clamped wrappers: when the whole
// tap window is resident the clamp is a no-op and the result is bit-identical
// to the resident kernel. When the window overhangs the plane, dav1d's
// reconstruction (src/recon_tmpl.c mc()) materializes the clamped halo once via
// emu_edge (src/mc_tmpl.c emu_edge_c) and re-runs the resident kernel over it,
// bit-identical to the per-tap-clamping pure-Go reference. Only widths the
// kernels cannot vectorize (width%4 != 0, which does not occur for AV1
// luma/chroma inter blocks) fall back to pure-Go.

func convolveXHighBDClampedGoSIMD(dst frame.Plane, ref frame.Plane, bitDepth uint8, max uint16, dstX int, dstY int, refX int, refY int, width int, height int, kernel [filterTaps]int16) {
	fo := filterTaps/2 - 1
	if width >= 4 && width%4 == 0 &&
		planeRegionFits(ref, 2, refX-fo, refY, width+filterTaps-1, height) {
		convolveXHighBDGoSIMD(dst, ref, bitDepth, max, dstX, dstY, refX, refY, width, height, kernel)
		return
	}
	if width >= 4 && width%4 == 0 {
		var edge emuEdge16Buf
		emu, emuX, emuY := emuEdgeWindow16(ref, refX, refY, width, height, &edge)
		convolveXHighBDGoSIMD(dst, emu, bitDepth, max, dstX, dstY, emuX, emuY, width, height, kernel)
		return
	}
	convolveXHighBDClampedPureGo(dst, ref, bitDepth, max, dstX, dstY, refX, refY, width, height, kernel)
}

func convolveYHighBDClampedGoSIMD(dst frame.Plane, ref frame.Plane, bitDepth uint8, max uint16, dstX int, dstY int, refX int, refY int, width int, height int, kernel [filterTaps]int16) {
	fo := filterTaps/2 - 1
	if width >= 4 && width%4 == 0 &&
		planeRegionFits(ref, 2, refX, refY-fo, width, height+filterTaps-1) {
		convolveYHighBDGoSIMD(dst, ref, bitDepth, max, dstX, dstY, refX, refY, width, height, kernel)
		return
	}
	if width >= 4 && width%4 == 0 {
		var edge emuEdge16Buf
		emu, emuX, emuY := emuEdgeWindow16(ref, refX, refY, width, height, &edge)
		convolveYHighBDGoSIMD(dst, emu, bitDepth, max, dstX, dstY, emuX, emuY, width, height, kernel)
		return
	}
	convolveYHighBDClampedPureGo(dst, ref, bitDepth, max, dstX, dstY, refX, refY, width, height, kernel)
}

func convolve2DHighBDClampedGoSIMD(dst frame.Plane, ref frame.Plane, bitDepth uint8, max uint16, dstX int, dstY int, refX int, refY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16) {
	convolve2DHighBDClampedGoSIMDWithScratch(dst, ref, bitDepth, max, dstX, dstY, refX, refY, width, height, xKernel, yKernel, nil)
}

// convolve2DHighBDClampedGoSIMDWithScratch is convolve2DHighBDClampedGoSIMD with
// optional caller-owned scratch for both the int32 intermediate block and the
// 16bpc emulated-edge window, so hot decode paths zero-fill neither per call.
func convolve2DHighBDClampedGoSIMDWithScratch(dst frame.Plane, ref frame.Plane, bitDepth uint8, max uint16, dstX int, dstY int, refX int, refY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16, scratch *ConvolveScratch) {
	foX := filterTaps/2 - 1
	foY := filterTaps/2 - 1
	if width >= 4 && width%4 == 0 &&
		planeRegionFits(ref, 2, refX-foX, refY-foY, width+filterTaps-1, height+filterTaps-1) {
		convolve2DHighBDGoSIMDWithScratch(dst, ref, bitDepth, max, dstX, dstY, refX, refY, width, height, xKernel, yKernel, scratch)
		return
	}
	if width >= 4 && width%4 == 0 {
		if scratch != nil {
			emu, emuX, emuY := emuEdgeWindow16(ref, refX, refY, width, height, &scratch.edge16)
			convolve2DHighBDGoSIMDWithScratch(dst, emu, bitDepth, max, dstX, dstY, emuX, emuY, width, height, xKernel, yKernel, scratch)
			return
		}
		var edge emuEdge16Buf
		emu, emuX, emuY := emuEdgeWindow16(ref, refX, refY, width, height, &edge)
		convolve2DHighBDGoSIMD(dst, emu, bitDepth, max, dstX, dstY, emuX, emuY, width, height, xKernel, yKernel)
		return
	}
	convolve2DHighBDClampedPureGoWithScratch(dst, ref, bitDepth, max, dstX, dstY, refX, refY, width, height, xKernel, yKernel, scratch)
}

// scaledStep2Half is half a full sample in Q10 stepper units: the step of an
// exact 2:1 upscale, the ratio between spatial layers in SVC streams.
const scaledStep2Half = ScaleSubpelScale / 2

// convolveScale2D8Step2GoSIMD computes convolveScale2D8 for xStep == yStep ==
// 512, an exact 2:1 upscale. The filter phase then alternates between two
// kernels: output column x reads source column xi0 + x/2 (+1 for odd x when
// the odd phase starts one sample later) with kernel phase x&1. A 16-byte load
// and one byte permute per tap therefore gather eight output columns, and an
// alternating coefficient vector applies both phases at once. Each output row
// uses a single vertical kernel, so the vertical pass is an ordinary 8-tap
// filter across columns. Rounding is the scaled formula, so the result is
// bit-identical. It returns false when the taps are not all even or the source
// window does not allow 16-byte loads; the caller then runs the scalar path.
func convolveScale2D8Step2GoSIMD(dst frame.Plane, ref frame.Plane, dstX int, dstY int, width int, height int,
	startX int64, startY int64, xTable, yTable *SubpelKernelTable, imH int, imBuf *scaledIM) bool {
	const fo = filterTaps/2 - 1
	k0 := &xTable[int(scaledSubpel(startX)>>ScaleExtraBits)]
	k1 := &xTable[int(scaledSubpel(startX+scaledStep2Half)>>ScaleExtraBits)]
	for t := range filterTaps {
		if k0[t]&1 != 0 || k1[t]&1 != 0 {
			return false
		}
	}
	xi0 := int(scaledIntFloor(startX)) - fo
	odd := int(scaledIntFloor(startX+scaledStep2Half)) - fo - xi0 // 0 or 1
	startRow := int(scaledIntFloor(startY)) - fo
	w8 := (width + 7) &^ 7
	if xi0 < 0 || startRow < 0 || imH <= 0 ||
		(startRow+imH-1)*ref.Stride+xi0+(w8-8)/2+16 > len(ref.Pix) {
		return false
	}

	// Lane x uses the halved coefficient of phase x&1.
	var coef [filterTaps]archsimd.Int16x8
	for t := range filterTaps {
		var c [8]int16
		for x := 0; x < 8; x += 2 {
			c[x], c[x+1] = k0[t]/2, k1[t]/2
		}
		coef[t] = archsimd.LoadInt16x8Array(&c)
	}
	im := imBuf[:]
	// Each source row is a window of the extent checked above: the
	// (w8-8)/2+16 bytes from xi0 the eight-column loads read.
	rowLen := (w8-8)/2 + 16
	for y := range imH {
		o := (startRow+y)*ref.Stride + xi0
		scaledStep2HorizontalRow(ref.Pix[o:o+rowLen], w8, odd, &coef, im[y*maxBlockSize:])
	}

	baseY := int(scaledIntFloor(startY))
	yRound := archsimd.BroadcastInt32x4(convolve8YBias + (1 << (round1Bits - 1)))
	right11 := archsimd.BroadcastInt32x4(-round1Bits)
	rndOff := archsimd.BroadcastInt32x4(convolve8RoundOffset)
	zero := archsimd.BroadcastInt32x4(0)
	maxV := archsimd.BroadcastInt32x4(255)
	load := func(y, x int) archsimd.Int16x8 {
		o := y*maxBlockSize + x
		return archsimd.LoadInt16x8Array((*[8]int16)(im[o : o+8]))
	}
	yPos := startY
	for y := range height {
		r := int(scaledIntFloor(yPos)) - baseY
		k := &yTable[int(scaledSubpel(yPos)>>ScaleExtraBits)]
		c0 := archsimd.BroadcastInt16x8(k[0])
		c1 := archsimd.BroadcastInt16x8(k[1])
		c2 := archsimd.BroadcastInt16x8(k[2])
		c3 := archsimd.BroadcastInt16x8(k[3])
		c4 := archsimd.BroadcastInt16x8(k[4])
		c5 := archsimd.BroadcastInt16x8(k[5])
		c6 := archsimd.BroadcastInt16x8(k[6])
		c7 := archsimd.BroadcastInt16x8(k[7])
		off := (dstY+y)*dst.Stride + dstX
		dstRow := dst.Pix[off : off+width : off+width]
		for x := 0; x < width; x += 8 {
			lo, hi := u8WideMAC8(zero, zero, load(r, x), c0)
			lo, hi = u8WideMAC8(lo, hi, load(r+1, x), c1)
			lo, hi = u8WideMAC8(lo, hi, load(r+2, x), c2)
			lo, hi = u8WideMAC8(lo, hi, load(r+3, x), c3)
			lo, hi = u8WideMAC8(lo, hi, load(r+4, x), c4)
			lo, hi = u8WideMAC8(lo, hi, load(r+5, x), c5)
			lo, hi = u8WideMAC8(lo, hi, load(r+6, x), c6)
			lo, hi = u8WideMAC8(lo, hi, load(r+7, x), c7)
			lo = u8Right32(lo.Add(yRound), right11, round1Bits).Sub(rndOff).Max(zero).Min(maxV)
			hi = u8Right32(hi.Add(yRound), right11, round1Bits).Sub(rndOff).Max(zero).Min(maxV)
			var tmp [16]byte
			u8PackClipped32(lo, hi).Store(tmp[:])
			copy(dstRow[x:], tmp[:min(8, width-x)])
		}
		yPos += scaledStep2Half
	}
	return true
}

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
	return archsimd.LoadUint8x16Array((*[16]uint8)(pix[:16])).ReshapeToUint16s().BitsToInt16()
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
		lo = hbdMulAdd32(archsimd.LoadInt32x4Array((*[4]int32)(row[:4])), t[k], lo)
		hi = hbdMulAdd32(archsimd.LoadInt32x4Array((*[4]int32)(row[4:8])), t[k], hi)
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
				a := archsimd.LoadUint8x16Array((*[16]uint8)(src[x*2 : x*2+16]))
				b := archsimd.LoadUint8x16Array((*[16]uint8)(src[x*2+16 : x*2+32]))
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
			r0l, r0h := hbdWidenU16(archsimd.LoadUint8x16Array((*[16]uint8)(ref[p : p+16])).ReshapeToUint16s())
			r1l, r1h := hbdWidenU16(archsimd.LoadUint8x16Array((*[16]uint8)(ref[p+refStr : p+refStr+16])).ReshapeToUint16s())
			r2l, r2h := hbdWidenU16(archsimd.LoadUint8x16Array((*[16]uint8)(ref[p+2*refStr : p+2*refStr+16])).ReshapeToUint16s())
			r3l, r3h := hbdWidenU16(archsimd.LoadUint8x16Array((*[16]uint8)(ref[p+3*refStr : p+3*refStr+16])).ReshapeToUint16s())
			r4l, r4h := hbdWidenU16(archsimd.LoadUint8x16Array((*[16]uint8)(ref[p+4*refStr : p+4*refStr+16])).ReshapeToUint16s())
			r5l, r5h := hbdWidenU16(archsimd.LoadUint8x16Array((*[16]uint8)(ref[p+5*refStr : p+5*refStr+16])).ReshapeToUint16s())
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
					r5l, r5h = hbdWidenU16(archsimd.LoadUint8x16Array((*[16]uint8)(ref[p+5*refStr : p+5*refStr+16])).ReshapeToUint16s())
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
		r0l, r0h := hbdWidenU16(archsimd.LoadUint8x16Array((*[16]uint8)(ref[p : p+16])).ReshapeToUint16s())
		r1l, r1h := hbdWidenU16(archsimd.LoadUint8x16Array((*[16]uint8)(ref[p+refStr : p+refStr+16])).ReshapeToUint16s())
		r2l, r2h := hbdWidenU16(archsimd.LoadUint8x16Array((*[16]uint8)(ref[p+2*refStr : p+2*refStr+16])).ReshapeToUint16s())
		r3l, r3h := hbdWidenU16(archsimd.LoadUint8x16Array((*[16]uint8)(ref[p+3*refStr : p+3*refStr+16])).ReshapeToUint16s())
		r4l, r4h := hbdWidenU16(archsimd.LoadUint8x16Array((*[16]uint8)(ref[p+4*refStr : p+4*refStr+16])).ReshapeToUint16s())
		r5l, r5h := hbdWidenU16(archsimd.LoadUint8x16Array((*[16]uint8)(ref[p+5*refStr : p+5*refStr+16])).ReshapeToUint16s())
		r6l, r6h := hbdWidenU16(archsimd.LoadUint8x16Array((*[16]uint8)(ref[p+6*refStr : p+6*refStr+16])).ReshapeToUint16s())
		r7l, r7h := hbdWidenU16(archsimd.LoadUint8x16Array((*[16]uint8)(ref[p+7*refStr : p+7*refStr+16])).ReshapeToUint16s())
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
				r7l, r7h = hbdWidenU16(archsimd.LoadUint8x16Array((*[16]uint8)(ref[p+7*refStr : p+7*refStr+16])).ReshapeToUint16s())
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
				a := archsimd.LoadUint8x16Array((*[16]uint8)(src[x*2 : x*2+16]))
				b := archsimd.LoadUint8x16Array((*[16]uint8)(src[x*2+16 : x*2+32]))
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
			s := archsimd.LoadUint8x16Array((*[16]uint8)(src[x*2 : x*2+16])).ReshapeToUint16s()
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
	a := archsimd.LoadUint8x16Array((*[16]uint8)(pix[:16]))
	b := archsimd.LoadUint8x16Array((*[16]uint8)(pix[16:32]))
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
				a := archsimd.LoadUint8x16Array((*[16]uint8)(src[x*2 : x*2+16]))
				b := archsimd.LoadUint8x16Array((*[16]uint8)(src[x*2+16 : x*2+32]))
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
			a = archsimd.LoadUint8x16Array((*[16]uint8)(src[:16]))
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
		r0l, r0h := hbdWidenU16(archsimd.LoadUint8x16Array((*[16]uint8)(ref[p : p+16])).ReshapeToUint16s())
		r1l, r1h := hbdWidenU16(archsimd.LoadUint8x16Array((*[16]uint8)(ref[p+refStr : p+refStr+16])).ReshapeToUint16s())
		r2l, r2h := hbdWidenU16(archsimd.LoadUint8x16Array((*[16]uint8)(ref[p+2*refStr : p+2*refStr+16])).ReshapeToUint16s())
		r3l, r3h := hbdWidenU16(archsimd.LoadUint8x16Array((*[16]uint8)(ref[p+3*refStr : p+3*refStr+16])).ReshapeToUint16s())
		r4l, r4h := hbdWidenU16(archsimd.LoadUint8x16Array((*[16]uint8)(ref[p+4*refStr : p+4*refStr+16])).ReshapeToUint16s())
		r5l, r5h := hbdWidenU16(archsimd.LoadUint8x16Array((*[16]uint8)(ref[p+5*refStr : p+5*refStr+16])).ReshapeToUint16s())
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
				r5l, r5h = hbdWidenU16(archsimd.LoadUint8x16Array((*[16]uint8)(ref[p+5*refStr : p+5*refStr+16])).ReshapeToUint16s())
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
				a := archsimd.LoadUint16x8Array((*[8]uint16)(s0[x : x+8]))
				b := archsimd.LoadUint16x8Array((*[8]uint16)(s1[x : x+8]))
				pix := hbdShiftRightU16(hbdAverageU16(a, b).SubSaturated(threshold), shiftAmount, roundBits).Min(maxV)
				pix.ReshapeToUint8s().Store(dst[x*2 : x*2+16])
			}
			if x < width {
				var a, b [8]uint16
				copy(a[:4], s0[x:x+4])
				copy(b[:4], s1[x:x+4])
				pix := hbdShiftRightU16(hbdAverageU16(archsimd.LoadUint16x8Array(&a), archsimd.LoadUint16x8Array(&b)).SubSaturated(threshold), shiftAmount, roundBits).Min(maxV)
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
			a := archsimd.LoadUint16x8Array((*[8]uint16)(s0[x : x+8]))
			b := archsimd.LoadUint16x8Array((*[8]uint16)(s1[x : x+8]))
			lo, hi := hbdBlendHalf(a, b, fwd, bck, roundOff, roundBits, zero, maxV)
			hbdStorePix8(dst[x*2:], lo, hi)
		}
		if x < width {
			var a, b [8]uint16
			copy(a[:4], s0[x:x+4])
			copy(b[:4], s1[x:x+4])
			lo, _ := hbdBlendHalf(archsimd.LoadUint16x8Array(&a), archsimd.LoadUint16x8Array(&b), fwd, bck, roundOff, roundBits, zero, maxV)
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
