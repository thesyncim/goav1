// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package motion

import (
	"simd/archsimd"

	"github.com/thesyncim/goav1/internal/av1/frame"
)

// The 16bpc vertical warp pass (dav1d warp_affine_8x8_16bpc_neon, the
// vertical half of src/arm/64/mc16.S warp): eight output rows of eight
// columns, each column's eight-tap filter taken from its own phase. As in
// dav1d the eight filter rows of an output row are transposed in registers,
// so tap m is one vector over the eight columns; the int32 intermediate rows
// multiply-accumulate against those taps widened to int32 lanes (MLA), which
// is exact for any int32 intermediate the horizontal pass produces (the sums
// stay far inside int32). The scalar rounding and the final
// - 2^(bd-1) - 2^bd fold into the accumulator seed: for the arithmetic right
// shift, (s >> r) - k == (s - k*2^r) >> r.

func warpVerticalHighBDFullDispatch(dst frame.Plane, bitDepth uint8, max uint16, tmp *warpTmp, i, j, rowShift, colShift, baseSY, gamma, delta, reduceBitsVert, offsetBitsVert int) {
	if !warpGoSIMDEnabled || !warpVertFullOffsInRange(baseSY, gamma, delta) {
		warpVerticalHighBDFull(dst, bitDepth, max, tmp, i, j, rowShift, colShift, baseSY, gamma, delta, reduceBitsVert, offsetBitsVert)
		return
	}
	warpVerticalHighBDGoSIMD(dst, bitDepth, max, tmp, i+rowShift, j+colShift, baseSY, gamma, delta, reduceBitsVert, offsetBitsVert)
}

func warpVerticalHighBDFullGamma0Dispatch(dst frame.Plane, bitDepth uint8, max uint16, tmp *warpTmp, i, j, rowShift, colShift, baseSY, delta, reduceBitsVert, offsetBitsVert int) {
	if !warpGoSIMDEnabled || !warpVertFullOffsInRange(baseSY, 0, delta) {
		warpVerticalHighBDFullGamma0(dst, bitDepth, max, tmp, i, j, rowShift, colShift, baseSY, delta, reduceBitsVert, offsetBitsVert)
		return
	}
	warpVerticalHighBDGamma0GoSIMD(dst, bitDepth, max, tmp, i+rowShift, j+colShift, baseSY, delta, reduceBitsVert, offsetBitsVert)
}

// warpHBDVertSeed is the accumulator seed of the 16bpc vertical pass:
// 2^offset + 2^(reduce-1) - (2^(bd-1) + 2^bd) * 2^reduce.
func warpHBDVertSeed(bitDepth uint8, reduce, offset int) int32 {
	return int32(1)<<offset + int32(1)<<(reduce-1) - (int32(1)<<(bitDepth-1)+int32(1)<<bitDepth)<<reduce
}

// warpHBDStoreRow clips the eight int32 sums lo|hi (already shifted) to
// [0, max] and stores them as eight 16-bit samples at dst row y, column x.
func warpHBDStoreRow(dst frame.Plane, y, x int, lo, hi, zero, maxV archsimd.Int32x4) {
	v := lo.Max(zero).Min(maxV).ToBits().ReshapeToUint16s().ConcatEven(hi.Max(zero).Min(maxV).ToBits().ReshapeToUint16s())
	off := y*dst.Stride + 2*x
	v.ReshapeToUint8s().StoreArray((*[16]byte)(dst.Pix[off : off+16]))
}

// warpVerticalHighBDGoSIMD computes the 8x8 output block at (x0, y0); the
// caller has checked every filter phase is in range.
func warpVerticalHighBDGoSIMD(dst frame.Plane, bitDepth uint8, max uint16, tmp *warpTmp, y0, x0, baseSY, gamma, delta, reduce, offset int) {
	seed := archsimd.BroadcastInt32x4(warpHBDVertSeed(bitDepth, reduce, offset))
	right := archsimd.BroadcastInt32x4(-int32(reduce))
	zero := archsimd.Int32x4{}
	maxV := archsimd.BroadcastInt32x4(int32(max))
	for row := 0; row < 8; row++ {
		// The eight taps of this output row read tmp rows row..row+7.
		win := (*[8 * warpedIntermediateColumns]int32)(tmp[row*warpedIntermediateColumns:])
		sy := baseSY + delta*row + 512
		f0 := archsimd.LoadInt8x16(warp8FilterPadded[(64+(sy>>10))&255][:])
		f1 := archsimd.LoadInt8x16(warp8FilterPadded[(64+((sy+gamma)>>10))&255][:])
		f2 := archsimd.LoadInt8x16(warp8FilterPadded[(64+((sy+2*gamma)>>10))&255][:])
		f3 := archsimd.LoadInt8x16(warp8FilterPadded[(64+((sy+3*gamma)>>10))&255][:])
		f4 := archsimd.LoadInt8x16(warp8FilterPadded[(64+((sy+4*gamma)>>10))&255][:])
		f5 := archsimd.LoadInt8x16(warp8FilterPadded[(64+((sy+5*gamma)>>10))&255][:])
		f6 := archsimd.LoadInt8x16(warp8FilterPadded[(64+((sy+6*gamma)>>10))&255][:])
		f7 := archsimd.LoadInt8x16(warp8FilterPadded[(64+((sy+7*gamma)>>10))&255][:])
		a01 := f0.InterleaveLo(f1).ToBits().ReshapeToUint16s()
		a23 := f2.InterleaveLo(f3).ToBits().ReshapeToUint16s()
		a45 := f4.InterleaveLo(f5).ToBits().ReshapeToUint16s()
		a67 := f6.InterleaveLo(f7).ToBits().ReshapeToUint16s()
		b03lo := a01.InterleaveLo(a23).ReshapeToUint32s()
		b03hi := a01.InterleaveHi(a23).ReshapeToUint32s()
		b47lo := a45.InterleaveLo(a67).ReshapeToUint32s()
		b47hi := a45.InterleaveHi(a67).ReshapeToUint32s()
		// c01 holds tap 0 of columns 0..7 in bytes 0..7 and tap 1 in bytes
		// 8..15, and so on.
		c01 := b03lo.InterleaveLo(b47lo).ReshapeToUint8s().BitsToInt8()
		c23 := b03lo.InterleaveHi(b47lo).ReshapeToUint8s().BitsToInt8()
		c45 := b03hi.InterleaveLo(b47hi).ReshapeToUint8s().BitsToInt8()
		c67 := b03hi.InterleaveHi(b47hi).ReshapeToUint8s().BitsToInt8()
		lo, hi := seed, seed
		mac := func(m int, t archsimd.Int16x8) {
			lo = archsimd.LoadInt32x4Array((*[4]int32)(win[8*m:8*m+4])).MulAdd(t.ExtendLo4ToInt32(), lo)
			hi = archsimd.LoadInt32x4Array((*[4]int32)(win[8*m+4:8*m+8])).MulAdd(t.HiToLo().ExtendLo4ToInt32(), hi)
		}
		mac(0, c01.ExtendLo8ToInt16())
		mac(1, c01.HiToLo().ExtendLo8ToInt16())
		mac(2, c23.ExtendLo8ToInt16())
		mac(3, c23.HiToLo().ExtendLo8ToInt16())
		mac(4, c45.ExtendLo8ToInt16())
		mac(5, c45.HiToLo().ExtendLo8ToInt16())
		mac(6, c67.ExtendLo8ToInt16())
		mac(7, c67.HiToLo().ExtendLo8ToInt16())
		warpHBDStoreRow(dst, y0+row, x0, lo.Shift(right), hi.Shift(right), zero, maxV)
	}
}

// warpVerticalHighBDGamma0GoSIMD is the gamma-zero vertical pass: one filter
// for all eight columns of a row.
func warpVerticalHighBDGamma0GoSIMD(dst frame.Plane, bitDepth uint8, max uint16, tmp *warpTmp, y0, x0, baseSY, delta, reduce, offset int) {
	seed := archsimd.BroadcastInt32x4(warpHBDVertSeed(bitDepth, reduce, offset))
	right := archsimd.BroadcastInt32x4(-int32(reduce))
	zero := archsimd.Int32x4{}
	maxV := archsimd.BroadcastInt32x4(int32(max))
	for row := 0; row < 8; row++ {
		win := (*[8 * warpedIntermediateColumns]int32)(tmp[row*warpedIntermediateColumns:])
		coeff := &warpedFilter[64+((baseSY+delta*row+512)>>10)]
		lo, hi := seed, seed
		mac := func(m int) {
			c := archsimd.BroadcastInt32x4(int32(coeff[m]))
			lo = archsimd.LoadInt32x4Array((*[4]int32)(win[8*m:8*m+4])).MulAdd(c, lo)
			hi = archsimd.LoadInt32x4Array((*[4]int32)(win[8*m+4:8*m+8])).MulAdd(c, hi)
		}
		mac(0)
		mac(1)
		mac(2)
		mac(3)
		mac(4)
		mac(5)
		mac(6)
		mac(7)
		warpHBDStoreRow(dst, y0+row, x0, lo.Shift(right), hi.Shift(right), zero, maxV)
	}
}
