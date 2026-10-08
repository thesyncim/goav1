// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package filmgrain

import "simd/archsimd"

// grainSIMDAvailable reports whether the Go SIMD grain kernels may run. NEON
// (AdvSIMD) is mandatory on every arm64 target Go runs on.
func grainSIMDAvailable() bool {
	return true
}

// grainPack8 narrows eight int32 values (lo holds lanes 0..3, hi holds lanes
// 4..7; every value is already clamped into the int16 range) to uint16 lanes in
// order. SaturateToInt16 leaves each four-lane result in the low half of its
// vector, so a 64-bit interleave packs the two halves into one row.
func grainPack8(lo, hi archsimd.Int32x4) archsimd.Uint16x8 {
	l := lo.SaturateToInt16().ConvertToUint16().ReshapeToUint64s()
	h := hi.SaturateToInt16().ConvertToUint16().ReshapeToUint64s()
	return l.InterleaveLo(h).ReshapeToUint16s()
}
