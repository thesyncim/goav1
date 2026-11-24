// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package motion

import (
	"simd/archsimd"

	"github.com/thesyncim/goav1/internal/av1/frame"
)

// SIMD dot products center uint16 samples before signed widening. Every row in
// warpedFilter sums to 128, so this restores the removed 32768*128 from each
// dot while preserving the full uint16 input domain.
const warpHighBDSampleCenterBias = 1 << 22

// warpHorizontalHighBDResidentGoSIMD computes the resident 15x8 tile using
// pairwise vector reductions to produce four horizontal outputs at a time.
func warpHorizontalHighBDResidentGoSIMD(tmp *[warpedIntermediateRows * warpedIntermediateColumns]int32, ref frame.Plane, ix4 int, sx4 int, iy4 int, sy4 int, alpha int, beta int, reduceBitsHoriz int, offsetBitsHoriz int) int {
	// The samples load as bytes, so no alignment is required of ref.
	center := archsimd.BroadcastInt16x8(-1 << 15)
	// All 193 filter rows sum to 128 and have L1 <= 222 (checked in tests).
	// Centered dots are bounded by 7,274,496; the compensation and rounding
	// bias therefore remain safely within int32.
	roundBias := archsimd.BroadcastInt32x4(int32(warpHighBDSampleCenterBias + (1 << offsetBitsHoriz) + (1 << (reduceBitsHoriz - 1))))
	roundShift := archsimd.BroadcastInt32x4(-int32(reduceBitsHoriz))
	for k := -7; k < 8; k++ {
		// The row's 15 taps, columns ix4-7 .. ix4+7 (the caller keeps them
		// inside the plane).
		row := (*[30]byte)(ref.Pix[(iy4+k)*ref.Stride+(ix4-7)*2:])
		sx := sx4 + beta*(k+4)
		for group := 0; group < warpedIntermediateColumns; group += 4 {
			phase0 := sx
			phase1 := phase0 + alpha
			phase2 := phase1 + alpha
			phase3 := phase2 + alpha
			offs0 := warpHorizontalHighBDFilterIndex(phase0)
			offs1 := warpHorizontalHighBDFilterIndex(phase1)
			offs2 := warpHorizontalHighBDFilterIndex(phase2)
			offs3 := warpHorizontalHighBDFilterIndex(phase3)
			sx = phase3 + alpha

			g := (*[22]byte)(row[2*group:])
			s0 := archsimd.LoadUint8x16Array((*[16]byte)(g[0:16])).ReshapeToUint16s().BitsToInt16().Xor(center)
			s1 := archsimd.LoadUint8x16Array((*[16]byte)(g[2:18])).ReshapeToUint16s().BitsToInt16().Xor(center)
			s2 := archsimd.LoadUint8x16Array((*[16]byte)(g[4:20])).ReshapeToUint16s().BitsToInt16().Xor(center)
			s3 := archsimd.LoadUint8x16Array((*[16]byte)(g[6:22])).ReshapeToUint16s().BitsToInt16().Xor(center)
			c0 := archsimd.LoadInt16x8Array(&warpedFilter[offs0])
			c1 := archsimd.LoadInt16x8Array(&warpedFilter[offs1])
			c2 := archsimd.LoadInt16x8Array(&warpedFilter[offs2])
			c3 := archsimd.LoadInt16x8Array(&warpedFilter[offs3])

			p0 := s0.MulWidenLo(c0)
			p1 := s1.MulWidenLo(c1)
			p2 := s2.MulWidenLo(c2)
			p3 := s3.MulWidenLo(c3)
			q0 := s0.HiToLo().MulWidenLo(c0.HiToLo())
			q1 := s1.HiToLo().MulWidenLo(c1.HiToLo())
			q2 := s2.HiToLo().MulWidenLo(c2.HiToLo())
			q3 := s3.HiToLo().MulWidenLo(c3.HiToLo())
			// Each pairwise tree gathers four terms from each output lane.
			lo := p0.ConcatAddPairs(p1).ConcatAddPairs(p2.ConcatAddPairs(p3))
			hi := q0.ConcatAddPairs(q1).ConcatAddPairs(q2.ConcatAddPairs(q3))
			dots := lo.Add(hi)

			out := (k+7)*warpedIntermediateColumns + group
			dots.Add(roundBias).Shift(roundShift).StoreArray((*[4]int32)(tmp[out:]))
		}
	}
	return sy4
}

func warpHorizontalHighBDResidentDispatch(tmp *[warpedIntermediateRows * warpedIntermediateColumns]int32, ref frame.Plane, ix4 int, sx4 int, iy4 int, sy4 int, alpha int, beta int, reduceBitsHoriz int, offsetBitsHoriz int) int {
	return warpHorizontalHighBDResidentGoSIMD(tmp, ref, ix4, sx4, iy4, sy4, alpha, beta, reduceBitsHoriz, offsetBitsHoriz)
}

func warpHorizontalHighBDFilterIndex(sx int) int {
	offs := roundPowerOfTwo(sx, warpedDiffPrecBits) + warpedPixelPrecShifts
	if offs < 0 {
		return 0
	}
	if offs >= len(warpedFilter) {
		return len(warpedFilter) - 1
	}
	return offs
}
