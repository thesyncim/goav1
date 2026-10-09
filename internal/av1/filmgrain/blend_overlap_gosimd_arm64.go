// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package filmgrain

import "simd/archsimd"

// blendGrainRow is the overlap two-tap blend entry point under GOEXPERIMENT=simd
// on arm64; blendGrainRowPureGo remains the bit-exact reference. The call is
// concrete (not a func variable) so the caller's blend scratch buffer stays on
// the stack and the apply path stays zero-alloc.
func blendGrainRow(dst []int16, prev []int16, cur []int16, prevWeight int, curWeight int, grainMin int, grainMax int) {
	blendGrainRowSIMD(dst, prev, cur, prevWeight, curWeight, grainMin, grainMax)
}

// blendGrainRowSIMD is the Go SIMD twin of blendGrainRowPureGo. Eight grain
// samples per iteration: prev and cur widen to int32 lanes, form
// prev*prevWeight+cur*curWeight exactly (grain fits int16 and the weights sum
// to 44 or 45, so the sum fits int32), apply the rounding shift that reproduces
// roundPowerOfTwo(v, 5) with an arithmetic right shift, and clamp to
// [grainMin, grainMax]. The clamped values fit int16, so the narrow is exact.
// The final (<8) samples take the scalar reference.
func blendGrainRowSIMD(dst []int16, prev []int16, cur []int16, prevWeight int, curWeight int, grainMin int, grainMax int) {
	n := len(dst)
	// Slicing every operand to n checks each one once. The group loop then
	// consumes the heads of these slices; its condition proves each holds a
	// full group, so the array-pointer conversions need no bounds checks.
	pr, cr, dl := prev[:n], cur[:n], dst[:n]
	pw := archsimd.BroadcastInt32x4(int32(prevWeight))
	cw := archsimd.BroadcastInt32x4(int32(curWeight))
	round := archsimd.BroadcastInt32x4(1 << 4)
	minV := archsimd.BroadcastInt32x4(int32(grainMin))
	maxV := archsimd.BroadcastInt32x4(int32(grainMax))
	for len(dl) >= 8 && len(pr) >= 8 && len(cr) >= 8 {
		pLo, pHi := grainExtend8(archsimd.LoadInt16x8Array((*[8]int16)(pr)))
		cLo, cHi := grainExtend8(archsimd.LoadInt16x8Array((*[8]int16)(cr)))
		lo := pLo.Mul(pw).Add(cLo.Mul(cw)).Add(round).ShiftAllRight(5).Max(minV).Min(maxV)
		hi := pHi.Mul(pw).Add(cHi.Mul(cw)).Add(round).ShiftAllRight(5).Max(minV).Min(maxV)
		grainPack8(lo, hi).BitsToInt16().StoreArray((*[8]int16)(dl))
		pr, cr, dl = pr[8:], cr[8:], dl[8:]
	}
	for k := range dl {
		v := roundPowerOfTwo(int(pr[k])*prevWeight+int(cr[k])*curWeight, 5)
		dl[k] = int16(clipInt(v, grainMin, grainMax))
	}
}
