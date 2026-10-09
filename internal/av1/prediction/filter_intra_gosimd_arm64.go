// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package prediction

import "simd/archsimd"

// filterIntraPackBytes narrows eight int16 filter outputs, each already clamped
// to [0,255], into the low eight bytes of the result (UQXTN).
func filterIntraPackBytes(out archsimd.Int16x8) archsimd.Uint8x16 {
	return out.ConvertToUint16().SaturateToUint8()
}

// filterIntraShr4 is the rounding shift (by filterIntraScaleBits) as a per-lane
// shift vector: the arm64 ShiftAllRight lowering re-broadcasts the amount for
// every use, whereas a per-lane Shift against a hoisted vector is one VSSHL.
var filterIntraShr4 = archsimd.BroadcastInt16x8(-filterIntraScaleBits)

// filterIntraShift4 arithmetic-shifts each int16 lane right by filterIntraScaleBits.
func filterIntraShift4(v archsimd.Int16x8) archsimd.Int16x8 {
	return v.Shift(filterIntraShr4)
}
