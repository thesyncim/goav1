// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && (amd64 || arm64) && !purego

package filmgrain

import "simd/archsimd"

// Shared helpers for the Go SIMD grain kernels (simd/archsimd, GOEXPERIMENT=simd).
// Every grain sample fits int16 (pixels are at most 4095, scaling values at most
// 255, grain samples are int16), so the 16-bit lanes are exact inputs and the
// products and sums are formed in int32 lanes, matching the scalar reference.

// grainHi4 moves the high four 16-bit lanes of x into the low four lanes with
// one byte-granular concatenate-shift (EXT / PALIGNR by 8 bytes); the high four
// lanes of the result are not used.
func grainHi4(x archsimd.Int16x8) archsimd.Int16x8 {
	b := x.ToBits().ReshapeToUint8s()
	return b.ConcatShiftBytesRight(b, 8).ReshapeToUint16s().BitsToInt16()
}

// grainExtend8 sign-extends the eight 16-bit lanes of x to two int32 vectors:
// lo holds lanes 0..3 and hi holds lanes 4..7.
func grainExtend8(x archsimd.Int16x8) (lo, hi archsimd.Int32x4) {
	return x.ExtendLo4ToInt32(), grainHi4(x).ExtendLo4ToInt32()
}
