// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package motion

import (
	"simd/archsimd"
	"unsafe"

	"github.com/thesyncim/goav1/internal/av1/frame"
)

// warpVertical8FullSIMD / warpVertical8FullGamma0SIMD are the Go-native SIMD
// forms of the 8-bit warped-motion vertical pass. Eight destination columns are
// produced per row as one Int16x8; the 8-tap vertical MAC runs in int32 lanes
// with widening multiplies. A single bias fold combines the +offsetBits
// pre-bias and -128-256 output shift. The result is rounded and saturated to
// int16, then saturated to uint8, matching the scalar round and clip stages.
//
// tmp holds the horizontal-pass output. Its magnitude is bounded well inside
// int16 (|tmp| < 2^15: reduce_bits_horiz=3 leaves at most ~6128), so the rows
// are narrowed to Int16x8 once and the widening MACs never overflow int32
// (worst term |127*tmp|*8 + bias stays under ~5M << 2^31). This matches the
// scalar warpVertical8Full* byte-for-byte.

// biasVert folds the vertical-pass constant offset into a single int32 add
// applied to the raw MAC accumulator, before the rounding narrow of
// reduceBitsVert. Scalar does:
//
//	out = clip[0,255]( ((acc + (1<<offsetBitsVert) + (1<<(reduceBitsVert-1))) >> reduceBitsVert) - 128 - 256 )
//
// Because 128+256==384 is subtracted AFTER the arithmetic shift, and
// 384<<reduceBitsVert is an exact multiple of 1<<reduceBitsVert, it folds into
// the pre-shift value: (X>>n) - k == (X - k*2^n)>>n. The vertical SIMD kernels
// add the rounding term to this bias after verifying the supported bit counts.
func biasVert(reduceBitsVert, offsetBitsVert int) int32 {
	const clipOffset = (1 << 7) + (1 << 8)
	return int32((1 << offsetBitsVert) - (clipOffset << uint(reduceBitsVert)))
}

// warpVertical8FullGamma0SIMD is the gamma==0 case: the vertical filter is
// constant across all eight columns of a row (offs depends only on sy, which is
// fixed per row), so it is a straight 8-tap vertical filter applied to eight
// int32 columns at once. An out-of-range offset sends the whole block through
// the scalar path, which leaves each out-of-range row untouched.
func warpVertical8FullGamma0SIMD(dst frame.Plane, tmp *[warpedIntermediateRows * warpedIntermediateColumns]int32, i, j, rowShift, colShift, baseSY, delta, reduceBitsVert, offsetBitsVert int) {
	const expectedOffsetBits = 8 + 2*filterBits - round0Bits
	if reduceBitsVert != round1Bits || offsetBitsVert != expectedOffsetBits || !warpVertFullOffsInRange(baseSY, 0, delta) {
		warpVertical8FullGamma0(dst, tmp, i, j, rowShift, colShift, baseSY, delta, reduceBitsVert, offsetBitsVert)
		return
	}
	// Include the rounding bias before the arithmetic shift. The accumulator
	// bound is far below int32 limits, so this is exact and avoids a second shift
	// plus per-row shift-count broadcasts in the tie-correction sequence.
	bias := archsimd.BroadcastInt32x4(biasVert(round1Bits, expectedOffsetBits) + (1 << (round1Bits - 1)))
	shift := archsimd.BroadcastInt32x4(-round1Bits)

	// Narrow the 15 tmp rows (int32, |tmp|<2^15) to Int16x8 once. Output row k
	// reads the 8-tap window tmp rows (k+4)..(k+11) (scalar tmpRow=(k+4)*8).
	var s [warpedIntermediateRows]archsimd.Int16x8
	for m := 0; m < warpedIntermediateRows; m++ {
		s[m] = loadTmpRow16(tmp, m)
	}

	for k := -4; k < 4; k++ {
		sy := baseSY + delta*(k+4)
		offs := roundPowerOfTwo(sy, warpedDiffPrecBits) + warpedPixelPrecShifts
		base := k + 4
		c := &warpedFilter[offs]
		// Broadcast each tap across all 8 columns; MAC in int32 lanes.
		t0 := archsimd.BroadcastInt16x8(c[0])
		t1 := archsimd.BroadcastInt16x8(c[1])
		t2 := archsimd.BroadcastInt16x8(c[2])
		t3 := archsimd.BroadcastInt16x8(c[3])
		t4 := archsimd.BroadcastInt16x8(c[4])
		t5 := archsimd.BroadcastInt16x8(c[5])
		t6 := archsimd.BroadcastInt16x8(c[6])
		t7 := archsimd.BroadcastInt16x8(c[7])
		lo := s[base+0].MulWidenLo(t0)
		hi := s[base+0].HiToLo().MulWidenLo(t0.HiToLo())
		lo = lo.Add(s[base+1].MulWidenLo(t1))
		hi = hi.Add(s[base+1].HiToLo().MulWidenLo(t1.HiToLo()))
		lo = lo.Add(s[base+2].MulWidenLo(t2))
		hi = hi.Add(s[base+2].HiToLo().MulWidenLo(t2.HiToLo()))
		lo = lo.Add(s[base+3].MulWidenLo(t3))
		hi = hi.Add(s[base+3].HiToLo().MulWidenLo(t3.HiToLo()))
		lo = lo.Add(s[base+4].MulWidenLo(t4))
		hi = hi.Add(s[base+4].HiToLo().MulWidenLo(t4.HiToLo()))
		lo = lo.Add(s[base+5].MulWidenLo(t5))
		hi = hi.Add(s[base+5].HiToLo().MulWidenLo(t5.HiToLo()))
		lo = lo.Add(s[base+6].MulWidenLo(t6))
		hi = hi.Add(s[base+6].HiToLo().MulWidenLo(t6.HiToLo()))
		lo = lo.Add(s[base+7].MulWidenLo(t7))
		hi = hi.Add(s[base+7].HiToLo().MulWidenLo(t7.HiToLo()))

		lo = lo.Add(bias).Shift(shift)
		hi = hi.Add(bias).Shift(shift)
		out := simdConcatInt16x8(lo.SaturateToInt16(), hi.SaturateToInt16())
		dstRow := (i+rowShift+k+4)*dst.Stride + j + colShift
		convStore8(unsafe.Pointer(&dst.Pix[dstRow]), out)
	}
}

// warpVertical8FullSIMD is the general (gamma!=0) case: the filter index offs
// varies per column (sy increments by gamma across columns) and per row (baseSY
// stepped by delta). Eight per-column filters are gathered and transposed into
// per-tap Int16x8 vectors so the same 8-lane MAC pipeline applies; the
// elementwise product s[m]*ftap[m] yields, per lane col, coeffs_col[m]*tmp[m].
//
// The scalar path skips any column whose offs is out of range (leaving dst
// unchanged for that pixel). A single affine bounds check decides whether the
// entire block is safe for SIMD; edge shears fall back to scalar to preserve
// the exact per-pixel skip semantics without a bounds branch in every row.
func warpVertical8FullSIMD(dst frame.Plane, tmp *[warpedIntermediateRows * warpedIntermediateColumns]int32, i, j, rowShift, colShift, baseSY, gamma, delta, reduceBitsVert, offsetBitsVert int) {
	const expectedOffsetBits = 8 + 2*filterBits - round0Bits
	if reduceBitsVert != round1Bits || offsetBitsVert != expectedOffsetBits || !warpVertFullOffsInRange(baseSY, gamma, delta) {
		warpVertical8Full(dst, tmp, i, j, rowShift, colShift, baseSY, gamma, delta, reduceBitsVert, offsetBitsVert)
		return
	}
	// As in the gamma0 kernel, fold roundPowerOfTwo's bias into the existing
	// vertical offset. The bounded MAC accumulator cannot overflow int32.
	bias := archsimd.BroadcastInt32x4(biasVert(round1Bits, expectedOffsetBits) + (1 << (round1Bits - 1)))
	shift := archsimd.BroadcastInt32x4(-round1Bits)

	// Output row k reads the 8-tap window tmp rows (k+4)..(k+11).
	var s [warpedIntermediateRows]archsimd.Int16x8
	for m := 0; m < warpedIntermediateRows; m++ {
		s[m] = loadTmpRow16(tmp, m)
	}

	for k := -4; k < 4; k++ {
		sy := baseSY + delta*(k+4)
		// Gather the 8 per-column filters and transpose into per-tap columns:
		// ftap[m][col] = warpedFilter[offs_col][m].
		var ftap [8][8]int16
		syc := sy
		for col := 0; col < 8; col++ {
			offs := roundPowerOfTwo(syc, warpedDiffPrecBits) + warpedPixelPrecShifts
			c := &warpedFilter[offs]
			ftap[0][col] = c[0]
			ftap[1][col] = c[1]
			ftap[2][col] = c[2]
			ftap[3][col] = c[3]
			ftap[4][col] = c[4]
			ftap[5][col] = c[5]
			ftap[6][col] = c[6]
			ftap[7][col] = c[7]
			syc += gamma
		}
		base := k + 4
		f0 := archsimd.LoadInt16x8Array(&ftap[0])
		f1 := archsimd.LoadInt16x8Array(&ftap[1])
		f2 := archsimd.LoadInt16x8Array(&ftap[2])
		f3 := archsimd.LoadInt16x8Array(&ftap[3])
		f4 := archsimd.LoadInt16x8Array(&ftap[4])
		f5 := archsimd.LoadInt16x8Array(&ftap[5])
		f6 := archsimd.LoadInt16x8Array(&ftap[6])
		f7 := archsimd.LoadInt16x8Array(&ftap[7])
		lo := s[base+0].MulWidenLo(f0)
		hi := s[base+0].HiToLo().MulWidenLo(f0.HiToLo())
		lo = lo.Add(s[base+1].MulWidenLo(f1))
		hi = hi.Add(s[base+1].HiToLo().MulWidenLo(f1.HiToLo()))
		lo = lo.Add(s[base+2].MulWidenLo(f2))
		hi = hi.Add(s[base+2].HiToLo().MulWidenLo(f2.HiToLo()))
		lo = lo.Add(s[base+3].MulWidenLo(f3))
		hi = hi.Add(s[base+3].HiToLo().MulWidenLo(f3.HiToLo()))
		lo = lo.Add(s[base+4].MulWidenLo(f4))
		hi = hi.Add(s[base+4].HiToLo().MulWidenLo(f4.HiToLo()))
		lo = lo.Add(s[base+5].MulWidenLo(f5))
		hi = hi.Add(s[base+5].HiToLo().MulWidenLo(f5.HiToLo()))
		lo = lo.Add(s[base+6].MulWidenLo(f6))
		hi = hi.Add(s[base+6].HiToLo().MulWidenLo(f6.HiToLo()))
		lo = lo.Add(s[base+7].MulWidenLo(f7))
		hi = hi.Add(s[base+7].HiToLo().MulWidenLo(f7.HiToLo()))

		lo = lo.Add(bias).Shift(shift)
		hi = hi.Add(bias).Shift(shift)
		out := simdConcatInt16x8(lo.SaturateToInt16(), hi.SaturateToInt16())
		dstRow := (i+rowShift+k+4)*dst.Stride + j + colShift
		convStore8(unsafe.Pointer(&dst.Pix[dstRow]), out)
	}
}

// loadTmpRow16 loads the 8 int32 columns of intermediate row m and narrows them
// to an Int16x8 (|tmp| < 2^15, so truncation is lossless). Lane col holds
// tmp[m*8+col].
func loadTmpRow16(tmp *[warpedIntermediateRows * warpedIntermediateColumns]int32, m int) archsimd.Int16x8 {
	lo := archsimd.LoadInt32x4Array((*[4]int32)(tmp[m*warpedIntermediateColumns:]))
	hi := archsimd.LoadInt32x4Array((*[4]int32)(tmp[m*warpedIntermediateColumns+4:]))
	return simdConcatInt16x8(lo.TruncToInt16(), hi.TruncToInt16())
}
