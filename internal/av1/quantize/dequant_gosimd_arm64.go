// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package quantize

import (
	"simd/archsimd"
	"unsafe"

	"github.com/thesyncim/goav1/internal/av1/dsp/cpu"
)

// dequantColumnSIMD is the Go-native SIMD twin of dequantColumnPureGo on NEON.
// It processes eight coefficients per iteration as two Int32x4 halves and
// handles the n%8 tail with the scalar reference.
//
// Bit-exactness with dequantScalar:
//   - the int16 coefficients are sign-extended to int32 lanes, so |c| (and
//     INT16_MIN) is exact in int32.
//   - the product |c|*scale is a 32-bit multiply; only its low 24 bits survive
//     the mask, and the low 24 bits of an int32 product equal those of the
//     int64 product dequantScalar forms.
//   - the masked product is non-negative, so the arithmetic right shift by
//     txScale (SSHL with a negative count, held in a hoisted vector) equals the
//     scalar shift.
//   - the sign is reapplied as multiplication by clamp(c, -1, 1): -1 for
//     negative c, +1 for positive c, and 0 for c == 0 where the magnitude is
//     already 0. The result is then clamped to [dqMin, dqMax] with Min then
//     Max, as the scalar clamp does.
func dequantColumnSIMD(dst []int32, coeff []int16, scale int32, txScale uint8, dqMin int32, dqMax int32) {
	n := len(coeff)
	if n == 0 {
		return
	}
	_ = dst[n-1]
	blocks8 := n &^ 7
	if blocks8 > 0 {
		scaleV := archsimd.BroadcastInt32x4(scale)
		maskV := archsimd.BroadcastInt32x4(0x00ffffff)
		oneV := archsimd.BroadcastInt32x4(1)
		negOneV := archsimd.BroadcastInt32x4(-1)
		// Negative counts make SSHL shift right; hoisted out of the loop.
		shiftV := archsimd.BroadcastInt32x4(-int32(txScale))
		loV := archsimd.BroadcastInt32x4(dqMin)
		hiV := archsimd.BroadcastInt32x4(dqMax)
		cp := unsafe.Pointer(&coeff[0])
		dp := unsafe.Pointer(&dst[0])
		for i := 0; i < blocks8; i += 8 {
			c := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(cp, 2*i)))
			lo := dequantLane4SIMD(c.ExtendLo4ToInt32(), scaleV, maskV, oneV, negOneV, shiftV, loV, hiV)
			hi := dequantLane4SIMD(c.HiToLo().ExtendLo4ToInt32(), scaleV, maskV, oneV, negOneV, shiftV, loV, hiV)
			lo.StoreArray((*[4]int32)(unsafe.Add(dp, 4*i)))
			hi.StoreArray((*[4]int32)(unsafe.Add(dp, 4*i+16)))
		}
	}
	if blocks8 < n {
		dequantColumnPureGo(dst[blocks8:], coeff[blocks8:], scale, txScale, dqMin, dqMax)
	}
}

// dequantLane4SIMD dequantizes four sign-extended coefficients (see
// dequantColumnSIMD for the exactness argument).
func dequantLane4SIMD(x, scale, mask, one, negOne, shift, lo, hi archsimd.Int32x4) archsimd.Int32x4 {
	sign := x.Min(one).Max(negOne)
	p := x.Abs().Mul(scale).And(mask).Shift(shift)
	return p.Mul(sign).Min(hi).Max(lo)
}

func init() {
	if cpu.Detected.NEON {
		dequantColumnImpl = dequantColumnSIMD
	}
}
