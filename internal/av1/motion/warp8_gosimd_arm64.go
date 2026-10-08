// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package motion

import (
	"os"
	"simd/archsimd"
	"unsafe"

	"github.com/thesyncim/goav1/internal/av1/dsp/cpu"
	"github.com/thesyncim/goav1/internal/av1/frame"
)

// Go-native SIMD 8-bit warped-motion kernels. The horizontal pass multiplies
// centred byte samples by byte coefficients in int16 lanes. The vertical pass
// forms each output column's eight-tap dot product in int32 lanes. Both compute
// the scalar formulas of warpHorizontal8Resident and warpVertical8Full exactly.

// Each row is 16 bytes so every filter row can be loaded as a full vector.
// Only the first eight lanes are used. The source table itself has eight-byte
// rows, which makes a full vector load at its last row unsafe.
var warp8FilterPadded [warpedPixelPrecShifts*3 + 1][16]int8

func init() {
	for i := range warpedFilterI8 {
		copy(warp8FilterPadded[i][:8], warpedFilterI8[i][:])
	}
}

// warpGoSIMDEnabled selects the Go SIMD warp kernels on arm64. The
// GOAV1_DISABLE_WARP_ASM variable keeps the scalar path for A/B timing; it is
// measurement scaffolding, not a shipped switch.
var warpGoSIMDEnabled = cpu.Detected.NEON && os.Getenv("GOAV1_DISABLE_WARP_ASM") == ""

func warpHorizontal8ResidentDispatch(tmp *warpTmp, ref frame.Plane, ix4, sx4, iy4, sy4, alpha, beta, reduceBitsHoriz, offsetBitsHoriz int) int {
	if warpGoSIMDEnabled {
		return warpHorizontal8ResidentGoSIMD(tmp, ref, ix4, sx4, iy4, sy4, alpha, beta, reduceBitsHoriz, offsetBitsHoriz)
	}
	return warpHorizontal8Resident(tmp, ref, ix4, sx4, iy4, sy4, alpha, beta, reduceBitsHoriz, offsetBitsHoriz)
}

func warpVertical8FullDispatch(dst frame.Plane, tmp *warpTmp, i, j, rowShift, colShift, baseSY, gamma, delta, reduceBitsVert, offsetBitsVert int) {
	if warpGoSIMDEnabled {
		warpVertical8FullGoSIMD(dst, tmp, i, j, rowShift, colShift, baseSY, gamma, delta, reduceBitsVert, offsetBitsVert)
		return
	}
	warpVertical8Full(dst, tmp, i, j, rowShift, colShift, baseSY, gamma, delta, reduceBitsVert, offsetBitsVert)
}

func warpVertical8FullGamma0Dispatch(dst frame.Plane, tmp *warpTmp, i, j, rowShift, colShift, baseSY, delta, reduceBitsVert, offsetBitsVert int) {
	if warpGoSIMDEnabled {
		warpVertical8FullGamma0GoSIMD(dst, tmp, i, j, rowShift, colShift, baseSY, delta, reduceBitsVert, offsetBitsVert)
		return
	}
	warpVertical8FullGamma0(dst, tmp, i, j, rowShift, colShift, baseSY, delta, reduceBitsVert, offsetBitsVert)
}

// warp8FilterIndex is the warpedFilter row for a filter phase: the rounded
// phase plus the precision shift, clamped to the table (as the scalar path).
func warp8FilterIndex(sx int) int {
	offs := roundPowerOfTwo(sx, warpedDiffPrecBits) + warpedPixelPrecShifts
	if offs < 0 {
		return 0
	}
	if offs >= len(warpedFilter) {
		return len(warpedFilter) - 1
	}
	return offs
}

// warpHorizontal8ResidentGoSIMD computes the resident 15x8 intermediate.
// Its routing matches the scalar warpHorizontal8Resident preconditions.
func warpHorizontal8ResidentGoSIMD(tmp *warpTmp, ref frame.Plane, ix4, sx4, iy4, sy4, alpha, beta, reduceBitsHoriz, offsetBitsHoriz int) int {
	if reduceBitsHoriz != round0Bits || offsetBitsHoriz != 8+filterBits-1 ||
		!warpHorizResidentOffsInRange(sx4, alpha, beta) {
		return warpHorizontal8Resident(tmp, ref, ix4, sx4, iy4, sy4, alpha, beta, reduceBitsHoriz, offsetBitsHoriz)
	}
	centre := archsimd.BroadcastUint8x16(128)
	round := archsimd.BroadcastInt16x8(4)
	right := archsimd.BroadcastInt16x8(-3)
	restore := archsimd.BroadcastInt16x8(4096)
	for k := -7; k < 8; k++ {
		rowByte := (iy4+k)*ref.Stride + ix4 - 7
		sx := sx4 + beta*(k+4)
		// The resident 15-sample window fits; u8Bytes16 also handles the
		// exact last-row case where the sixteenth byte is outside ref.Pix.
		a := u8Bytes16(ref.Pix[rowByte:]).Xor(centre)
		f0 := archsimd.LoadInt8x16(warp8FilterPadded[64+((sx+512)>>10)][:])
		f1 := archsimd.LoadInt8x16(warp8FilterPadded[64+((sx+alpha+512)>>10)][:])
		f2 := archsimd.LoadInt8x16(warp8FilterPadded[64+((sx+2*alpha+512)>>10)][:])
		f3 := archsimd.LoadInt8x16(warp8FilterPadded[64+((sx+3*alpha+512)>>10)][:])
		f4 := archsimd.LoadInt8x16(warp8FilterPadded[64+((sx+4*alpha+512)>>10)][:])
		f5 := archsimd.LoadInt8x16(warp8FilterPadded[64+((sx+5*alpha+512)>>10)][:])
		f6 := archsimd.LoadInt8x16(warp8FilterPadded[64+((sx+6*alpha+512)>>10)][:])
		f7 := archsimd.LoadInt8x16(warp8FilterPadded[64+((sx+7*alpha+512)>>10)][:])
		p0 := a.BitsToInt8().MulWidenLo(f0)
		p1 := a.ConcatShiftBytesRight(a, 1).BitsToInt8().MulWidenLo(f1)
		p2 := a.ConcatShiftBytesRight(a, 2).BitsToInt8().MulWidenLo(f2)
		p3 := a.ConcatShiftBytesRight(a, 3).BitsToInt8().MulWidenLo(f3)
		p4 := a.ConcatShiftBytesRight(a, 4).BitsToInt8().MulWidenLo(f4)
		p5 := a.ConcatShiftBytesRight(a, 5).BitsToInt8().MulWidenLo(f5)
		p6 := a.ConcatShiftBytesRight(a, 6).BitsToInt8().MulWidenLo(f6)
		p7 := a.ConcatShiftBytesRight(a, 7).BitsToInt8().MulWidenLo(f7)
		low := p0.ConcatAddPairs(p1).ConcatAddPairs(p2.ConcatAddPairs(p3))
		high := p4.ConcatAddPairs(p5).ConcatAddPairs(p6.ConcatAddPairs(p7))
		low = low.ConcatAddPairs(low)
		high = high.ConcatAddPairs(high)
		out := (k + 7) * warpedIntermediateColumns
		low.Add(round).Shift(right).Add(restore).ExtendLo4ToInt32().Store(tmp[out : out+4])
		high.Add(round).Shift(right).Add(restore).ExtendLo4ToInt32().Store(tmp[out+4 : out+8])
	}
	return sy4
}

// warpVertical8FullGoSIMD computes the vertical block with the same
// routing; the eight output columns of each row are two vectors of four.
func warpVertical8FullGoSIMD(dst frame.Plane, tmp *warpTmp, i, j, rowShift, colShift, baseSY, gamma, delta, reduceBitsVert, offsetBitsVert int) {
	if reduceBitsVert != round1Bits || offsetBitsVert != 8+2*filterBits-round0Bits ||
		!warpVertFullOffsInRange(baseSY, gamma, delta) {
		warpVertical8Full(dst, tmp, i, j, rowShift, colShift, baseSY, gamma, delta, reduceBitsVert, offsetBitsVert)
		return
	}
	warpVerticalGoSIMDCommon(dst, tmp, i, j, rowShift, colShift, baseSY, gamma, delta, reduceBitsVert, offsetBitsVert)
}

// warpVertical8FullGamma0GoSIMD is the gamma-zero vertical kernel: every column
// of a row shares one filter phase.
func warpVertical8FullGamma0GoSIMD(dst frame.Plane, tmp *warpTmp, i, j, rowShift, colShift, baseSY, delta, reduceBitsVert, offsetBitsVert int) {
	if reduceBitsVert != round1Bits || offsetBitsVert != 8+2*filterBits-round0Bits ||
		!warpVertFullOffsInRange(baseSY, 0, delta) {
		warpVertical8FullGamma0(dst, tmp, i, j, rowShift, colShift, baseSY, delta, reduceBitsVert, offsetBitsVert)
		return
	}
	warpVertical8Gamma0Kernel(dst, tmp, i, j, rowShift, colShift, baseSY, delta)
}

// warpVertical8Gamma0Kernel uses one filter for all eight columns. Folding the
// final subtraction into the accumulator seed is exact for the arithmetic
// right shift: (sum >> 11) - 384 == (sum - 384*2048) >> 11.
// The caller has proved that all eight destination columns fit.
// All intermediate pointer loads stay within the 15x8 fixed array.
//
//go:nocheckptr
func warpVertical8Gamma0Kernel(dst frame.Plane, tmp *warpTmp, i, j, rowShift, colShift, baseSY, delta int) {
	const seed = (1 << (8 + 2*filterBits - round0Bits)) + (1 << (round1Bits - 1)) - ((1<<7)+(1<<8))*(1<<round1Bits)
	bias := archsimd.BroadcastInt32x4(seed)
	right := archsimd.BroadcastInt32x4(-round1Bits)
	tmpBase := unsafe.Pointer(&tmp[0])
	load := func(off int) archsimd.Int32x4 {
		return archsimd.LoadInt32x4((*[4]int32)(unsafe.Add(tmpBase, off*4))[:])
	}
	for row := 0; row < 8; row++ {
		coeff := &warpedFilter[warp8FilterIndex(baseSY+delta*row)]
		c0 := archsimd.BroadcastInt32x4(int32(coeff[0]))
		c1 := archsimd.BroadcastInt32x4(int32(coeff[1]))
		c2 := archsimd.BroadcastInt32x4(int32(coeff[2]))
		c3 := archsimd.BroadcastInt32x4(int32(coeff[3]))
		c4 := archsimd.BroadcastInt32x4(int32(coeff[4]))
		c5 := archsimd.BroadcastInt32x4(int32(coeff[5]))
		c6 := archsimd.BroadcastInt32x4(int32(coeff[6]))
		c7 := archsimd.BroadcastInt32x4(int32(coeff[7]))
		p := row * warpedIntermediateColumns
		lo := load(p).MulAdd(c0, bias)
		hi := load(p+4).MulAdd(c0, bias)
		lo = load(p+8).MulAdd(c1, lo)
		hi = load(p+12).MulAdd(c1, hi)
		lo = load(p+16).MulAdd(c2, lo)
		hi = load(p+20).MulAdd(c2, hi)
		lo = load(p+24).MulAdd(c3, lo)
		hi = load(p+28).MulAdd(c3, hi)
		lo = load(p+32).MulAdd(c4, lo)
		hi = load(p+36).MulAdd(c4, hi)
		lo = load(p+40).MulAdd(c5, lo)
		hi = load(p+44).MulAdd(c5, hi)
		lo = load(p+48).MulAdd(c6, lo)
		hi = load(p+52).MulAdd(c6, hi)
		lo = load(p+56).MulAdd(c7, lo)
		hi = load(p+60).MulAdd(c7, hi)
		dstRow := (i+rowShift+row)*dst.Stride + j + colShift
		u8PackSaturate32(u8Right32(lo, right, round1Bits), u8Right32(hi, right, round1Bits)).StorePart(dst.Pix[dstRow : dstRow+8])
	}
}

// warpVerticalGoSIMDCommon computes the 8x8 vertical output block. Eight byte
// filter rows are transposed in registers so each tap vector contains the
// coefficients for eight columns. The int32 horizontal intermediate fits
// signed int16, so widening products preserve the scalar dot product.
// All intermediate pointer loads stay within the 15x8 fixed array.
//
//go:nocheckptr
func warpVerticalGoSIMDCommon(dst frame.Plane, tmp *warpTmp, i, j, rowShift, colShift, baseSY, gamma, delta, reduceBitsVert, offsetBitsVert int) {
	const seed = (1 << (8 + 2*filterBits - round0Bits)) + (1 << (round1Bits - 1)) - ((1<<7)+(1<<8))*(1<<round1Bits)
	bias := archsimd.BroadcastInt32x4(seed)
	right := archsimd.BroadcastInt32x4(-round1Bits)
	tmpBase := unsafe.Pointer(&tmp[0])
	load := func(off int) archsimd.Int16x8 {
		lo := archsimd.LoadInt32x4((*[4]int32)(unsafe.Add(tmpBase, off*4))[:])
		hi := archsimd.LoadInt32x4((*[4]int32)(unsafe.Add(tmpBase, (off+4)*4))[:])
		return u8PackU16From32(lo, hi).BitsToInt16()
	}
	for row := 0; row < 8; row++ {
		sy := baseSY + delta*row + 512
		f0 := archsimd.LoadInt8x16(warp8FilterPadded[64+(sy>>10)][:])
		f1 := archsimd.LoadInt8x16(warp8FilterPadded[64+((sy+gamma)>>10)][:])
		f2 := archsimd.LoadInt8x16(warp8FilterPadded[64+((sy+2*gamma)>>10)][:])
		f3 := archsimd.LoadInt8x16(warp8FilterPadded[64+((sy+3*gamma)>>10)][:])
		f4 := archsimd.LoadInt8x16(warp8FilterPadded[64+((sy+4*gamma)>>10)][:])
		f5 := archsimd.LoadInt8x16(warp8FilterPadded[64+((sy+5*gamma)>>10)][:])
		f6 := archsimd.LoadInt8x16(warp8FilterPadded[64+((sy+6*gamma)>>10)][:])
		f7 := archsimd.LoadInt8x16(warp8FilterPadded[64+((sy+7*gamma)>>10)][:])
		a01 := f0.InterleaveLo(f1).ToBits().ReshapeToUint16s()
		a23 := f2.InterleaveLo(f3).ToBits().ReshapeToUint16s()
		a45 := f4.InterleaveLo(f5).ToBits().ReshapeToUint16s()
		a67 := f6.InterleaveLo(f7).ToBits().ReshapeToUint16s()
		b03lo := a01.InterleaveLo(a23).ReshapeToUint32s()
		b03hi := a01.InterleaveHi(a23).ReshapeToUint32s()
		b47lo := a45.InterleaveLo(a67).ReshapeToUint32s()
		b47hi := a45.InterleaveHi(a67).ReshapeToUint32s()
		c01 := b03lo.InterleaveLo(b47lo).ReshapeToUint8s()
		c23 := b03lo.InterleaveHi(b47lo).ReshapeToUint8s()
		c45 := b03hi.InterleaveLo(b47hi).ReshapeToUint8s()
		c67 := b03hi.InterleaveHi(b47hi).ReshapeToUint8s()
		t0 := c01.BitsToInt8().ExtendLo8ToInt16()
		t1 := c01.ConcatShiftBytesRight(c01, 8).BitsToInt8().ExtendLo8ToInt16()
		t2 := c23.BitsToInt8().ExtendLo8ToInt16()
		t3 := c23.ConcatShiftBytesRight(c23, 8).BitsToInt8().ExtendLo8ToInt16()
		t4 := c45.BitsToInt8().ExtendLo8ToInt16()
		t5 := c45.ConcatShiftBytesRight(c45, 8).BitsToInt8().ExtendLo8ToInt16()
		t6 := c67.BitsToInt8().ExtendLo8ToInt16()
		t7 := c67.ConcatShiftBytesRight(c67, 8).BitsToInt8().ExtendLo8ToInt16()
		p := row * warpedIntermediateColumns
		lo, hi := u8WideMAC8(bias, bias, load(p), t0)
		lo, hi = u8WideMAC8(lo, hi, load(p+8), t1)
		lo, hi = u8WideMAC8(lo, hi, load(p+16), t2)
		lo, hi = u8WideMAC8(lo, hi, load(p+24), t3)
		lo, hi = u8WideMAC8(lo, hi, load(p+32), t4)
		lo, hi = u8WideMAC8(lo, hi, load(p+40), t5)
		lo, hi = u8WideMAC8(lo, hi, load(p+48), t6)
		lo, hi = u8WideMAC8(lo, hi, load(p+56), t7)
		dstRow := (i+rowShift+row)*dst.Stride + j + colShift
		u8PackSaturate32(u8Right32(lo, right, round1Bits), u8Right32(hi, right, round1Bits)).StorePart(dst.Pix[dstRow : dstRow+8])
	}
}
