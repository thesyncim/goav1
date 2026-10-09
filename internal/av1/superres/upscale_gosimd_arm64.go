// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package superres

import "simd/archsimd"

// init binds the Go SIMD super-res row kernel on arm64. NEON (AdvSIMD) is
// mandatory on every arm64 target Go runs on, so the binding is static.
func init() {
	upscaleRowImpl = upscaleRowSIMD
}

// superresPack8 narrows eight int32 pixel sums (lo holds pixels 0..3, hi holds
// pixels 4..7; every value is already in [0, 4095]) to uint16 lanes in pixel
// order. SaturateToInt16 leaves each four-pixel result in the low half of its
// vector, so a 64-bit interleave packs the two halves into one row.
func superresPack8(lo, hi archsimd.Int32x4) archsimd.Uint16x8 {
	l := lo.SaturateToInt16().ConvertToUint16().ReshapeToUint64s()
	h := hi.SaturateToInt16().ConvertToUint16().ReshapeToUint64s()
	return l.InterleaveLo(h).ReshapeToUint16s()
}
