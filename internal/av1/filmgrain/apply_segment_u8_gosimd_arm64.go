// SPDX-License-Identifier: BSD-2-Clause
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package filmgrain

import (
	"encoding/binary"
	"simd/archsimd"
)

// applyGrainSegmentU8 retains the uint16 path's exact int32 products, signed
// rounding, add and clip, but loads and stores eight source/display bytes.
// Eight-byte accesses stay within the row, including overlap runs of 15/30
// samples. A partial final group is scalar. Exact src/dst alias is supported.
func applyGrainSegmentU8(dst, src []uint8, scale []uint16, grain []int16, scalingShift, minValue, maxValue int) {
	n := len(dst)
	scl, gr, sr := scale[:n], grain[:n], src[:n]
	round := archsimd.BroadcastInt32x4(int32(1) << (scalingShift - 1))
	shift := uint64(scalingShift)
	minV := archsimd.BroadcastInt32x4(int32(minValue))
	maxV := archsimd.BroadcastInt32x4(int32(maxValue))
	i := 0
	for ; i+8 <= n; i += 8 {
		sLo, sHi := grainExtend8(archsimd.LoadUint16x8Array((*[8]uint16)(scl[i:])).BitsToInt16())
		gLo, gHi := grainExtend8(archsimd.LoadInt16x8Array((*[8]int16)(gr[i:])))
		x := archsimd.BroadcastUint64x2(binary.LittleEndian.Uint64(sr[i:])).ReshapeToUint8s().ExtendLo8ToUint16()
		xLo, xHi := grainExtend8(x.BitsToInt16())
		lo := sLo.Mul(gLo).Add(round).ShiftAllRight(shift).Add(xLo).Max(minV).Min(maxV)
		hi := sHi.Mul(gHi).Add(round).ShiftAllRight(shift).Add(xHi).Max(minV).Min(maxV)
		out := grainPack8(lo, hi).SaturateToUint8().ReshapeToUint64s().GetElem(0)
		binary.LittleEndian.PutUint64(dst[i:], out)
	}
	for ; i < n; i++ {
		noise := roundPowerOfTwo(int(scl[i])*int(gr[i]), scalingShift)
		dst[i] = uint8(clipInt(int(sr[i])+noise, minValue, maxValue))
	}
}
