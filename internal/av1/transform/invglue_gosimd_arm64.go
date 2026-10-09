// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package transform

import "simd/archsimd"

// init binds the NEON-width Go SIMD inverse-transform glue kernels. They are
// bit-exact with clampRoundPureGo and narrowStorePureGo (see hybrid.go); the
// shared differential tests in invglue_gosimd_test.go pin that equivalence.
func init() {
	clampRoundImpl = clampRoundSIMD
	narrowStoreImpl = narrowStoreSIMD
}

// invGlueSIMDAvailable reports whether init bound the SIMD glue kernels. NEON
// is part of the arm64 baseline, so the binding is unconditional.
func invGlueSIMDAvailable() bool { return true }

// invGlueNegShift[s] holds -s in every lane: the variable-shift count that
// shifts right by s. Loading it from a table costs one vector load, where a
// broadcast per call is a lane insert plus a dup.
var invGlueNegShift = func() (t [32][4]int32) {
	for s := range t {
		for l := range t[s] {
			t[s][l] = -int32(s)
		}
	}
	return t
}()

// invGlueRoundVec returns roundShift(x, s) for each lane of v, for s >= 1.
// With y = x>>(s-1) (negS1 = -(s-1) broadcast), roundShift(x, s) == (y+1)>>1:
// the rounded halving add of y and zero (SRHADD) computes that without the
// overflow a plain y+1 would risk at MaxInt32.
func invGlueRoundVec(v, negS1, zero archsimd.Int32x4) archsimd.Int32x4 {
	return v.Shift(negS1).Average(zero)
}

// clampRoundSIMD rounds-and-shifts scratch in place (shift > 0) and clamps it to
// [min, max], eight int32 lanes per iteration. Tails and out-of-range shifts
// use the pure-Go reference.
func clampRoundSIMD(scratch []int32, shift int, min, max int32) {
	if shift < 0 || shift > 31 {
		clampRoundPureGo(scratch, shift, min, max)
		return
	}
	minV := archsimd.BroadcastInt32x4(min)
	maxV := archsimd.BroadcastInt32x4(max)
	n := len(scratch) &^ 7
	if shift == 0 {
		for i := 0; i < n; i += 8 {
			blk := scratch[i : i+8 : i+8]
			a := archsimd.LoadInt32x4Array((*[4]int32)(blk[0:4])).Max(minV).Min(maxV)
			b := archsimd.LoadInt32x4Array((*[4]int32)(blk[4:8])).Max(minV).Min(maxV)
			a.StoreArray((*[4]int32)(blk[0:4]))
			b.StoreArray((*[4]int32)(blk[4:8]))
		}
	} else {
		negS1 := archsimd.LoadInt32x4Array(&invGlueNegShift[shift-1])
		var zero archsimd.Int32x4
		for i := 0; i < n; i += 8 {
			blk := scratch[i : i+8 : i+8]
			a := invGlueRoundVec(archsimd.LoadInt32x4Array((*[4]int32)(blk[0:4])), negS1, zero)
			b := invGlueRoundVec(archsimd.LoadInt32x4Array((*[4]int32)(blk[4:8])), negS1, zero)
			a.Max(minV).Min(maxV).StoreArray((*[4]int32)(blk[0:4]))
			b.Max(minV).Min(maxV).StoreArray((*[4]int32)(blk[4:8]))
		}
	}
	if n < len(scratch) {
		clampRoundPureGo(scratch[n:], shift, min, max)
	}
}

// narrowStoreSIMD writes the final residual rows: every int32 lane is
// rounding-shifted right by four and saturated to int16. Width is four or a
// multiple of eight; other widths use the pure-Go reference.
func narrowStoreSIMD(dst []int16, dstStride int, scratch []int32, width, height int) {
	if width != 4 && width%8 != 0 {
		narrowStorePureGo(dst, dstStride, scratch, width, height)
		return
	}
	neg3 := archsimd.LoadInt32x4Array(&invGlueNegShift[3])
	var zero archsimd.Int32x4
	for row := range height {
		dstLine := dst[row*dstStride : row*dstStride+width : row*dstStride+width]
		srcLine := scratch[row*width : row*width+width : row*width+width]
		if width == 4 {
			v := archsimd.LoadInt32x4Array((*[4]int32)(srcLine))
			var out [8]int16
			narrowRound4ToInt16x8(v, v, neg3, zero).StoreArray(&out)
			copy(dstLine, out[:4])
			continue
		}
		for col := 0; col < width; col += 8 {
			lo := archsimd.LoadInt32x4Array((*[4]int32)(srcLine[col:]))
			hi := archsimd.LoadInt32x4Array((*[4]int32)(srcLine[col+4:]))
			narrowRound4ToInt16x8(lo, hi, neg3, zero).StoreArray((*[8]int16)(dstLine[col:]))
		}
	}
}

// narrowRound4ToInt16x8 rounding-shifts two int32x4 halves right by four and
// saturates them to one int16x8 in lane order (lo lanes 0-3, hi lanes 4-7). It
// is the hoisted-count form of roundShiftNarrowInt32x4ToInt16x8.
func narrowRound4ToInt16x8(lo, hi, neg3, zero archsimd.Int32x4) archsimd.Int16x8 {
	lo16 := invGlueRoundVec(lo, neg3, zero).SaturateToInt16()
	hi16 := invGlueRoundVec(hi, neg3, zero).SaturateToInt16()
	return lo16.ToBits().ReshapeToUint64s().InterleaveLo(hi16.ToBits().ReshapeToUint64s()).ReshapeToUint16s().BitsToInt16()
}
