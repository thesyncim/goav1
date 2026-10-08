// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package motion

import "testing"

func BenchmarkWarpVertical8FullGamma0Scalar(b *testing.B) {
	tmp, dst := benchWarpVerticalInputs()
	const reduceBitsVert = round1Bits
	const offsetBitsVert = 8 + 2*filterBits - round0Bits
	b.ResetTimer()
	for range b.N {
		warpVertical8FullGamma0(dst, &tmp, 8, 8, 0, 0, 32768, -64, reduceBitsVert, offsetBitsVert)
	}
}

func BenchmarkWarpVertical8FullGamma0SIMD(b *testing.B) {
	tmp, dst := benchWarpVerticalInputs()
	const reduceBitsVert = round1Bits
	const offsetBitsVert = 8 + 2*filterBits - round0Bits
	if !warpVertFullOffsInRange(32768, 0, -64) {
		b.Skip("bench inputs out of range")
	}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		warpVertical8FullGamma0SIMD(dst, &tmp, 8, 8, 0, 0, 32768, -64, reduceBitsVert, offsetBitsVert)
	}
}

func BenchmarkWarpVertical8FullScalar(b *testing.B) {
	tmp, dst := benchWarpVerticalInputs()
	const reduceBitsVert = round1Bits
	const offsetBitsVert = 8 + 2*filterBits - round0Bits
	b.ResetTimer()
	for range b.N {
		warpVertical8Full(dst, &tmp, 8, 8, 0, 0, 32768, 96, -64, reduceBitsVert, offsetBitsVert)
	}
}

func BenchmarkWarpVertical8FullSIMD(b *testing.B) {
	tmp, dst := benchWarpVerticalInputs()
	const reduceBitsVert = round1Bits
	const offsetBitsVert = 8 + 2*filterBits - round0Bits
	if !warpVertFullOffsInRange(32768, 96, -64) {
		b.Skip("bench inputs out of range")
	}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		warpVertical8FullSIMD(dst, &tmp, 8, 8, 0, 0, 32768, 96, -64, reduceBitsVert, offsetBitsVert)
	}
}
