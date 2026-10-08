// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package motion

import "testing"

// These benchmarks reuse the exact fixture and shear from the NEON benchmarks
// in warp_neon_arm64_test.go so scalar, NEON, and Go SIMD results are comparable.
func BenchmarkWarpHorizontal8ResidentScalar(b *testing.B) {
	ref, ix4, iy4, sx4, sy4, alpha, beta := benchWarpHorizInputs()
	const reduceBitsHoriz = round0Bits
	const offsetBitsHoriz = 8 + filterBits - 1
	var tmp [warpedIntermediateRows * warpedIntermediateColumns]int32
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		warpHorizontal8Resident(&tmp, ref, ix4, sx4, iy4, sy4, alpha, beta, reduceBitsHoriz, offsetBitsHoriz)
	}
	sink32 = tmp[0]
}

func BenchmarkWarpHorizontal8ResidentSIMD(b *testing.B) {
	ref, ix4, iy4, sx4, sy4, alpha, beta := benchWarpHorizInputs()
	const reduceBitsHoriz = round0Bits
	const offsetBitsHoriz = 8 + filterBits - 1
	var tmp [warpedIntermediateRows * warpedIntermediateColumns]int32
	if !warpHorizResidentOffsInRange(sx4, alpha, beta) {
		b.Skip("bench inputs out of range")
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		warpHorizontal8ResidentSIMD(&tmp, ref, ix4, sx4, iy4, sy4, alpha, beta, reduceBitsHoriz, offsetBitsHoriz)
	}
	sink32 = tmp[0]
}

var sink32 int32
