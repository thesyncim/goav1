// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && amd64 && !purego

package transform

import "simd/archsimd"

// init binds the AVX2 Go SIMD inverse-transform glue kernels when the CPU
// advertises AVX2. Every operation used here is AVX2 (VPSRAD, VPADDD, VPMINSD,
// VPMAXSD, VPACKSSDW); none lowers to AVX-512.
func init() {
	if invGlueSIMDAvailable() {
		clampRoundImpl = clampRoundSIMD
		narrowStoreImpl = narrowStoreSIMD
	}
}

// invGlueSIMDAvailable reports whether the AVX2 glue kernels may be bound.
func invGlueSIMDAvailable() bool { return archsimd.X86.AVX2() }

// clampRoundSIMD rounds-and-shifts scratch in place (shift > 0) and clamps it to
// [min, max], eight int32 lanes at a time. The rounding uses
// (x>>s) + ((x>>(s-1))&1), which equals roundShift(x, s) for every int32 x and
// cannot overflow. Tails and out-of-range shifts use the pure-Go reference.
func clampRoundSIMD(scratch []int32, shift int, min, max int32) {
	if shift < 0 || shift > 31 {
		clampRoundPureGo(scratch, shift, min, max)
		return
	}
	minV := archsimd.BroadcastInt32x8(min)
	maxV := archsimd.BroadcastInt32x8(max)
	one := archsimd.BroadcastInt32x8(1)
	s := uint64(shift)
	i := 0
	if shift == 0 {
		for ; i+8 <= len(scratch); i += 8 {
			archsimd.LoadInt32x8Array((*[8]int32)(scratch[i:])).
				Max(minV).Min(maxV).
				StoreArray((*[8]int32)(scratch[i:]))
		}
	} else {
		for ; i+8 <= len(scratch); i += 8 {
			v := archsimd.LoadInt32x8Array((*[8]int32)(scratch[i:]))
			r := v.ShiftAllRight(s).Add(v.ShiftAllRight(s - 1).And(one))
			r.Max(minV).Min(maxV).StoreArray((*[8]int32)(scratch[i:]))
		}
	}
	if i < len(scratch) {
		clampRoundPureGo(scratch[i:], shift, min, max)
	}
}

// roundShiftNarrowInt32x4PairToInt16x8 rounding-shifts two int32x4 halves right
// by shift and saturates them to one int16x8 in lane order (lo lanes 0-3, hi
// lanes 4-7). SaturateToInt16Concat is VPACKSSDW (AVX, not AVX-512).
func roundShiftNarrowInt32x4PairToInt16x8(lo, hi archsimd.Int32x4, shift uint64) archsimd.Int16x8 {
	one := archsimd.BroadcastInt32x4(1)
	lo16 := lo.ShiftAllRight(shift).Add(lo.ShiftAllRight(shift - 1).And(one))
	hi16 := hi.ShiftAllRight(shift).Add(hi.ShiftAllRight(shift - 1).And(one))
	return lo16.SaturateToInt16Concat(hi16)
}

// narrowStoreSIMD writes the final residual rows: every int32 lane is
// rounding-shifted right by four and saturated to int16. Width is four or a
// multiple of eight; other widths use the pure-Go reference.
func narrowStoreSIMD(dst []int16, dstStride int, scratch []int32, width, height int) {
	if width != 4 && width%8 != 0 {
		narrowStorePureGo(dst, dstStride, scratch, width, height)
		return
	}
	for row := range height {
		dstLine := dst[row*dstStride : row*dstStride+width : row*dstStride+width]
		srcLine := scratch[row*width : row*width+width : row*width+width]
		if width == 4 {
			v := archsimd.LoadInt32x4Array((*[4]int32)(srcLine))
			var out [8]int16
			roundShiftNarrowInt32x4PairToInt16x8(v, v, 4).StoreArray(&out)
			copy(dstLine, out[:4])
			continue
		}
		for col := 0; col < width; col += 8 {
			lo := archsimd.LoadInt32x4Array((*[4]int32)(srcLine[col:]))
			hi := archsimd.LoadInt32x4Array((*[4]int32)(srcLine[col+4:]))
			roundShiftNarrowInt32x4PairToInt16x8(lo, hi, 4).StoreArray((*[8]int16)(dstLine[col:]))
		}
	}
}
