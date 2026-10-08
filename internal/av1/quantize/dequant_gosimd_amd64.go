// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && amd64 && !purego

package quantize

import (
	"simd/archsimd"
	"unsafe"
)

// dequantColumnSIMD is the Go-native SIMD twin of dequantColumnPureGo on AVX2.
// It processes eight coefficients per iteration in one Int32x8 register and
// handles the n%8 tail with the scalar reference. The exactness argument is the
// same as the NEON kernel's (see dequant_gosimd_arm64.go): sign-extend, abs,
// 32-bit multiply masked to 24 bits, arithmetic shift of a non-negative value,
// sign reapplied as (x ^ sign) - sign, then Min/Max clamp.
func dequantColumnSIMD(dst []int32, coeff []int16, scale int32, txScale uint8, dqMin int32, dqMax int32) {
	n := len(coeff)
	if n == 0 {
		return
	}
	_ = dst[n-1]
	blocks8 := n &^ 7
	if blocks8 > 0 {
		scaleV := archsimd.BroadcastInt32x8(scale)
		maskV := archsimd.BroadcastInt32x8(0x00ffffff)
		loV := archsimd.BroadcastInt32x8(dqMin)
		hiV := archsimd.BroadcastInt32x8(dqMax)
		shift := uint64(txScale)
		for i := 0; i < blocks8; i += 8 {
			c := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Pointer(&coeff[i])))
			x := c.ExtendToInt32()
			sign := x.ShiftAllRight(31)
			p := x.Abs().Mul(scaleV).And(maskV).ShiftAllRight(shift)
			p.Xor(sign).Sub(sign).Min(hiV).Max(loV).StoreArray((*[8]int32)(unsafe.Pointer(&dst[i])))
		}
	}
	if blocks8 < n {
		dequantColumnPureGo(dst[blocks8:], coeff[blocks8:], scale, txScale, dqMin, dqMax)
	}
}

func init() {
	if archsimd.X86.AVX2() {
		dequantColumnImpl = dequantColumnSIMD
	}
}
