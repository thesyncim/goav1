// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && amd64 && !purego

package prediction

import "simd/archsimd"

// filterIntraPackBytes narrows eight int16 filter outputs, each already clamped
// to [0,255], into the low eight bytes of the result. The narrowing pack
// instructions (VPMOVWB/VPMOVUSWB/VPMOVSWB) and VPERMB are EVEX-only, so the
// bytes are shuffled out of the int16 lanes with the AVX VPSHUFB instead.
func filterIntraPackBytes(out archsimd.Int16x8) archsimd.Uint8x16 {
	return out.AsUint8x16().PermuteOrZero(evenBytesIdx)
}

// filterIntraShift4 arithmetic-shifts each int16 lane right by filterIntraScaleBits
// (VPSRAW with an immediate).
func filterIntraShift4(v archsimd.Int16x8) archsimd.Int16x8 {
	return v.ShiftAllRight(filterIntraScaleBits)
}
