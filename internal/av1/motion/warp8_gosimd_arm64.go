// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package motion

import (
	"os"
	"simd/archsimd"

	"github.com/thesyncim/goav1/internal/av1/dsp/cpu"
	"github.com/thesyncim/goav1/internal/av1/frame"
)

// Go-native SIMD 8-bit warped-motion kernels. The horizontal pass produces four
// columns per vector from eight sample loads (one per column) and the paired
// reductions of the high-bit-depth warp; the vertical pass forms each output
// column's eight-tap dot product with int32 lanes. Both compute the scalar
// formulas of warpHorizontal8Resident and warpVertical8Full(Gamma0) exactly.

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

// warpHorizontal8ResidentGoSIMD is warpHorizontal8ResidentNEON's replacement:
// the resident 15x8 intermediate with the same routing as the asm wrapper.
func warpHorizontal8ResidentGoSIMD(tmp *warpTmp, ref frame.Plane, ix4, sx4, iy4, sy4, alpha, beta, reduceBitsHoriz, offsetBitsHoriz int) int {
	if reduceBitsHoriz != round0Bits || offsetBitsHoriz != 8+filterBits-1 ||
		!warpHorizResidentOffsInRange(sx4, alpha, beta) {
		return warpHorizontal8Resident(tmp, ref, ix4, sx4, iy4, sy4, alpha, beta, reduceBitsHoriz, offsetBitsHoriz)
	}
	roundBias := archsimd.BroadcastInt32x4(int32((1 << offsetBitsHoriz) + (1 << (reduceBitsHoriz - 1))))
	roundShift := uint64(reduceBitsHoriz)
	for k := -7; k < 8; k++ {
		rowByte := (iy4 + k) * ref.Stride
		sx := sx4 + beta*(k+4)
		for group := 0; group < warpedIntermediateColumns; group += 4 {
			var s [4]archsimd.Int16x8
			var c [4]archsimd.Int16x8
			for lane := range 4 {
				phase := sx
				sx += alpha
				offs := warp8FilterIndex(phase)
				xByte := rowByte + (ix4-7+group+lane)
				s[lane] = u8Samples8(ref.Pix[xByte:])
				c[lane] = archsimd.LoadInt16x8(warpedFilter[offs][:])
			}
			p0 := s[0].MulWidenLo(c[0])
			p1 := s[1].MulWidenLo(c[1])
			p2 := s[2].MulWidenLo(c[2])
			p3 := s[3].MulWidenLo(c[3])
			q0 := s[0].HiToLo().MulWidenLo(c[0].HiToLo())
			q1 := s[1].HiToLo().MulWidenLo(c[1].HiToLo())
			q2 := s[2].HiToLo().MulWidenLo(c[2].HiToLo())
			q3 := s[3].HiToLo().MulWidenLo(c[3].HiToLo())
			// Each pairwise tree gathers the eight taps of each output lane.
			lo := p0.ConcatAddPairs(p1).ConcatAddPairs(p2.ConcatAddPairs(p3))
			hi := q0.ConcatAddPairs(q1).ConcatAddPairs(q2.ConcatAddPairs(q3))
			out := (k+7)*warpedIntermediateColumns + group
			lo.Add(hi).Add(roundBias).ShiftAllRight(roundShift).Store(tmp[out : out+4])
		}
	}
	return sy4
}

// warpVertical8FullGoSIMD is warpVertical8FullNEON's replacement with the same
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
	warpVerticalGoSIMDCommon(dst, tmp, i, j, rowShift, colShift, baseSY, 0, delta, reduceBitsVert, offsetBitsVert)
}

// warpVerticalGoSIMDCommon computes the 8x8 vertical output block. Output row k
// and column col (0..7) use the filter phase baseSY + delta*(k+4) + gamma*col;
// the eight taps read tmp rows k+4..k+11 at column col. Each vector holds four
// columns, and the coefficient vectors are gathered per lane from warpedFilter.
func warpVerticalGoSIMDCommon(dst frame.Plane, tmp *warpTmp, i, j, rowShift, colShift, baseSY, gamma, delta, reduceBitsVert, offsetBitsVert int) {
	// roundPowerOfTwo(sum, reduceBitsVert) with the bias folded into the seed.
	bias := archsimd.BroadcastInt32x4(int32(1)<<offsetBitsVert + int32(1)<<(reduceBitsVert-1))
	off := archsimd.BroadcastInt32x4(int32((1 << 7) + (1 << 8)))
	zero := archsimd.BroadcastInt32x4(0)
	maxV := archsimd.BroadcastInt32x4(255)
	for k := -4; k < 4; k++ {
		sy := baseSY + delta*(k+4)
		dstRow := (i+rowShift+k+4)*dst.Stride + j + colShift
		for group := 0; group < warpedIntermediateColumns; group += 4 {
			var coef [filterTaps][4]int32
			for lane := range 4 {
				offs := warp8FilterIndex(sy + gamma*(group+lane))
				for t := range filterTaps {
					coef[t][lane] = int32(warpedFilter[offs][t])
				}
			}
			sum := bias
			for t := range filterTaps {
				v := archsimd.LoadInt32x4(tmp[(k+4+t)*warpedIntermediateColumns+group:][:4])
				sum = hbdMulAdd32(archsimd.LoadInt32x4(coef[t][:]), v, sum)
			}
			pix := hbdClip(sum.ShiftAllRight(uint64(reduceBitsVert)).Sub(off), zero, maxV)
			hbdStoreU8x4(dst.Pix[dstRow+group:], pix)
		}
	}
}
