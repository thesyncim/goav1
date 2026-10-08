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
