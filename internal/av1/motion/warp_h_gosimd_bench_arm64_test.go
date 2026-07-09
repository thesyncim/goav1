// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package motion

import "testing"

// These benchmarks set up a resident 48x48 ref plane with a representative shear
// (alpha!=0, beta!=0) so the per-column-varying filter path is exercised, then
// measure the horizontal pass (scalar reference vs the Go-native SIMD kernel) in
// isolation.
func BenchmarkWarpHorizontal8ResidentScalar(b *testing.B) {
	ref, _ := testPlane(48, 48, 1, 48)
	for i := range ref.Pix {
		ref.Pix[i] = byte((i*61 + (i/48)*29 + 7) & 0xff)
	}
	const reduceBitsHoriz = round0Bits
	const offsetBitsHoriz = 8 + filterBits - 1
	const ix4, iy4 = 20, 20
	var tmp [warpedIntermediateRows * warpedIntermediateColumns]int32
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		warpHorizontal8Resident(&tmp, ref, ix4, 512, iy4, -256, 32, -16, reduceBitsHoriz, offsetBitsHoriz)
	}
	sink32 = tmp[0]
}

func BenchmarkWarpHorizontal8ResidentSIMD(b *testing.B) {
	ref, _ := testPlane(48, 48, 1, 48)
	for i := range ref.Pix {
		ref.Pix[i] = byte((i*61 + (i/48)*29 + 7) & 0xff)
	}
	const reduceBitsHoriz = round0Bits
	const offsetBitsHoriz = 8 + filterBits - 1
	const ix4, iy4 = 20, 20
	var tmp [warpedIntermediateRows * warpedIntermediateColumns]int32
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		warpHorizontal8ResidentSIMD(&tmp, ref, ix4, 512, iy4, -256, 32, -16, reduceBitsHoriz, offsetBitsHoriz)
	}
	sink32 = tmp[0]
}

var sink32 int32
