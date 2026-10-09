// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && (amd64 || arm64) && !purego

package filmgrain

import (
	"simd/archsimd"
)

// applyGrainSegmentUseSIMD selects the Go SIMD apply kernel. It is resolved
// once, at package init, before any decoder goroutine starts. It must not be
// mutated concurrently with live decoding.
//
// The dispatch is a plain bool rather than a func pointer so applyGrainSegment
// stays a concrete (non-indirect) call: escape analysis can then see that
// neither implementation retains its slice arguments, which keeps the caller's
// scale scratch buffer on the stack and the apply path zero-alloc.
var applyGrainSegmentUseSIMD bool

func init() {
	applyGrainSegmentUseSIMD = grainSIMDAvailable()
}

func applyGrainSegment(dst []uint16, src []uint16, scale []uint16, grain []int16, scalingShift int, minValue int, maxValue int) {
	if applyGrainSegmentUseSIMD {
		applyGrainSegmentSIMD(dst, src, scale, grain, scalingShift, minValue, maxValue)
		return
	}
	applyGrainSegmentPureGo(dst, src, scale, grain, scalingShift, minValue, maxValue)
}

// applyGrainSegmentSIMD is the Go SIMD twin of applyGrainSegmentPureGo. Eight
// samples per iteration: the scaling values and grain run widen to int32 lanes
// and multiply exactly (scale <= 255 times an int16 fits int32), the rounding
// shift matches roundPowerOfTwo with an arithmetic right shift, the source
// sample is added, and Max/Min reproduce clipInt. The result lies in
// [minValue, maxValue] within [0, 4095], so the narrow to uint16 is exact. The
// final (<8) samples take the scalar reference.
func applyGrainSegmentSIMD(dst []uint16, src []uint16, scale []uint16, grain []int16, scalingShift int, minValue int, maxValue int) {
	n := len(dst)
	// Slicing every operand to n checks each one once; the group loop then
	// addresses full groups with checked array-pointer conversions.
	scl, gr, sr, dl := scale[:n], grain[:n], src[:n], dst[:n]
	round := archsimd.BroadcastInt32x4(int32(1) << (scalingShift - 1))
	shift := uint64(scalingShift)
	minV := archsimd.BroadcastInt32x4(int32(minValue))
	maxV := archsimd.BroadcastInt32x4(int32(maxValue))
	i := 0
	for ; i+8 <= n; i += 8 {
		sLo, sHi := grainExtend8(archsimd.LoadUint16x8Array((*[8]uint16)(scl[i:])).BitsToInt16())
		gLo, gHi := grainExtend8(archsimd.LoadInt16x8Array((*[8]int16)(gr[i:])))
		xLo, xHi := grainExtend8(archsimd.LoadUint16x8Array((*[8]uint16)(sr[i:])).BitsToInt16())
		lo := sLo.Mul(gLo).Add(round).ShiftAllRight(shift).Add(xLo).Max(minV).Min(maxV)
		hi := sHi.Mul(gHi).Add(round).ShiftAllRight(shift).Add(xHi).Max(minV).Min(maxV)
		grainPack8(lo, hi).StoreArray((*[8]uint16)(dl[i:]))
	}
	for ; i < n; i++ {
		noise := roundPowerOfTwo(int(scl[i])*int(gr[i]), scalingShift)
		dl[i] = uint16(clipInt(int(sr[i])+noise, minValue, maxValue))
	}
}
