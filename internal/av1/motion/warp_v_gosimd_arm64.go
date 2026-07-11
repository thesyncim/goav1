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
// (SMULL/SMLAL via MulWidenLo/Hi), a single bias fold collapses the +offsetBits
// pre-bias, the -128-256 output shift, and the round1 rounding into one add, and
// a rounding narrow (SQRSHRN, ShiftRightRoundNarrow) plus a saturating u8 narrow
// (SQXTUN, SaturateToUint8) fuse the >>reduceBitsVert round with the clipPixel
// clamp to [0,255].
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
// the pre-shift value: (X>>n) - k == (X - k*2^n)>>n. The rounding term
// (1<<(reduceBitsVert-1)) is supplied by ShiftRightRoundNarrow itself, so it is
// NOT included here.
func biasVert(reduceBitsVert, offsetBitsVert int) int32 {
	return int32((1 << offsetBitsVert) - ((1 << 7) + (1 << 8)) << uint(reduceBitsVert))
}

// warpVertical8FullGamma0SIMD is the gamma==0 case: the vertical filter is
// constant across all eight columns of a row (offs depends only on sy, which is
// fixed per row), so it is a straight 8-tap vertical filter applied to eight
// int32 columns at once. An out-of-range offs skips the whole row (dst
// untouched), exactly as the scalar path.
func warpVertical8FullGamma0SIMD(dst frame.Plane, tmp *[warpedIntermediateRows * warpedIntermediateColumns]int32, i, j, rowShift, colShift, baseSY, delta, reduceBitsVert, offsetBitsVert int) {
	bias := archsimd.BroadcastInt32x4(biasVert(reduceBitsVert, offsetBitsVert))
	rb := uint8(reduceBitsVert)

	// Narrow the 15 tmp rows (int32, |tmp|<2^15) to Int16x8 once. Output row k
	// reads the 8-tap window tmp rows (k+4)..(k+11) (scalar tmpRow=(k+4)*8).
	var s [warpedIntermediateRows]archsimd.Int16x8
	for m := 0; m < warpedIntermediateRows; m++ {
		s[m] = loadTmpRow16(tmp, m)
	}

	for k := -4; k < 4; k++ {
		sy := baseSY + delta*(k+4)
		offs := roundPowerOfTwo(sy, warpedDiffPrecBits) + warpedPixelPrecShifts
		if offs < 0 || offs >= len(warpedFilter) {
			continue
		}
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
		hi := s[base+0].MulWidenHi(t0)
		lo = lo.MulWidenLoAdd(s[base+1], t1)
		hi = hi.MulWidenHiAdd(s[base+1], t1)
		lo = lo.MulWidenLoAdd(s[base+2], t2)
		hi = hi.MulWidenHiAdd(s[base+2], t2)
		lo = lo.MulWidenLoAdd(s[base+3], t3)
		hi = hi.MulWidenHiAdd(s[base+3], t3)
		lo = lo.MulWidenLoAdd(s[base+4], t4)
		hi = hi.MulWidenHiAdd(s[base+4], t4)
		lo = lo.MulWidenLoAdd(s[base+5], t5)
		hi = hi.MulWidenHiAdd(s[base+5], t5)
		lo = lo.MulWidenLoAdd(s[base+6], t6)
		hi = hi.MulWidenHiAdd(s[base+6], t6)
		lo = lo.MulWidenLoAdd(s[base+7], t7)
		hi = hi.MulWidenHiAdd(s[base+7], t7)

		lo = lo.Add(bias)
		hi = hi.Add(bias)
		out := lo.ShiftRightRoundNarrow(rb).ShiftRightRoundNarrowHi(hi, rb)
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
// unchanged for that pixel). Rows where every column is in range take the full
// SIMD store; a row with any out-of-range column falls back to the scalar row
// (rare edge shears) to preserve the exact per-pixel skip semantics.
func warpVertical8FullSIMD(dst frame.Plane, tmp *[warpedIntermediateRows * warpedIntermediateColumns]int32, i, j, rowShift, colShift, baseSY, gamma, delta, reduceBitsVert, offsetBitsVert int) {
	bias := archsimd.BroadcastInt32x4(biasVert(reduceBitsVert, offsetBitsVert))
	rb := uint8(reduceBitsVert)

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
		ok := true
		syc := sy
		for col := 0; col < 8; col++ {
			offs := roundPowerOfTwo(syc, warpedDiffPrecBits) + warpedPixelPrecShifts
			if offs < 0 || offs >= len(warpedFilter) {
				ok = false
				break
			}
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
		if !ok {
			// Rare: at least one column out of range. Scalar handles the exact
			// per-pixel skip for just this row.
			warpVertical8FullRowScalar(dst, tmp, i, j, rowShift, colShift, sy, gamma, k, reduceBitsVert, offsetBitsVert)
			continue
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
		hi := s[base+0].MulWidenHi(f0)
		lo = lo.MulWidenLoAdd(s[base+1], f1)
		hi = hi.MulWidenHiAdd(s[base+1], f1)
		lo = lo.MulWidenLoAdd(s[base+2], f2)
		hi = hi.MulWidenHiAdd(s[base+2], f2)
		lo = lo.MulWidenLoAdd(s[base+3], f3)
		hi = hi.MulWidenHiAdd(s[base+3], f3)
		lo = lo.MulWidenLoAdd(s[base+4], f4)
		hi = hi.MulWidenHiAdd(s[base+4], f4)
		lo = lo.MulWidenLoAdd(s[base+5], f5)
		hi = hi.MulWidenHiAdd(s[base+5], f5)
		lo = lo.MulWidenLoAdd(s[base+6], f6)
		hi = hi.MulWidenHiAdd(s[base+6], f6)
		lo = lo.MulWidenLoAdd(s[base+7], f7)
		hi = hi.MulWidenHiAdd(s[base+7], f7)

		lo = lo.Add(bias)
		hi = hi.Add(bias)
		out := lo.ShiftRightRoundNarrow(rb).ShiftRightRoundNarrowHi(hi, rb)
		dstRow := (i+rowShift+k+4)*dst.Stride + j + colShift
		convStore8(unsafe.Pointer(&dst.Pix[dstRow]), out)
	}
}

// warpVertical8FullRowScalar reproduces the scalar warpVertical8Full inner
// column loop for a single row k, including the per-column out-of-range skip.
func warpVertical8FullRowScalar(dst frame.Plane, tmp *[warpedIntermediateRows * warpedIntermediateColumns]int32, i, j, rowShift, colShift, sy, gamma, k, reduceBitsVert, offsetBitsVert int) {
	dstRow := (i+rowShift+k+4)*dst.Stride + j + colShift
	tmpRow := (k + 4) * warpedIntermediateColumns
	for l := -4; l < 4; l++ {
		offs := roundPowerOfTwo(sy, warpedDiffPrecBits) + warpedPixelPrecShifts
		if offs >= 0 && offs < len(warpedFilter) {
			coeffs := warpedFilter[offs]
			col := l + 4
			sum := 1 << offsetBitsVert
			sum += int(coeffs[0]) * int(tmp[tmpRow+0*warpedIntermediateColumns+col])
			sum += int(coeffs[1]) * int(tmp[tmpRow+1*warpedIntermediateColumns+col])
			sum += int(coeffs[2]) * int(tmp[tmpRow+2*warpedIntermediateColumns+col])
			sum += int(coeffs[3]) * int(tmp[tmpRow+3*warpedIntermediateColumns+col])
			sum += int(coeffs[4]) * int(tmp[tmpRow+4*warpedIntermediateColumns+col])
			sum += int(coeffs[5]) * int(tmp[tmpRow+5*warpedIntermediateColumns+col])
			sum += int(coeffs[6]) * int(tmp[tmpRow+6*warpedIntermediateColumns+col])
			sum += int(coeffs[7]) * int(tmp[tmpRow+7*warpedIntermediateColumns+col])
			sum = roundPowerOfTwo(sum, reduceBitsVert)
			dst.Pix[dstRow+l+4] = byte(clipPixel(sum - (1 << 7) - (1 << 8)))
		}
		sy += gamma
	}
}

// loadTmpRow16 loads the 8 int32 columns of intermediate row m and narrows them
// to an Int16x8 (|tmp| < 2^15, so truncation is lossless). Lane col holds
// tmp[m*8+col].
func loadTmpRow16(tmp *[warpedIntermediateRows * warpedIntermediateColumns]int32, m int) archsimd.Int16x8 {
	lo := archsimd.LoadInt32x4Array((*[4]int32)(tmp[m*warpedIntermediateColumns:]))
	hi := archsimd.LoadInt32x4Array((*[4]int32)(tmp[m*warpedIntermediateColumns+4:]))
	return lo.TruncToInt16().TruncToInt16Hi(hi)
}
