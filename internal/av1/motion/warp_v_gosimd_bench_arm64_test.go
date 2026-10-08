// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package motion

import "testing"

// benchWarpVTmp is a representative horizontal-pass intermediate buffer for the
// vertical-kernel micro-benchmarks.
func benchWarpVTmp() [warpedIntermediateRows * warpedIntermediateColumns]int32 {
	var tmp [warpedIntermediateRows * warpedIntermediateColumns]int32
	for i := range tmp {
		tmp[i] = int32((i*97+31)%8192 - 4096)
	}
	return tmp
}

func BenchmarkWarpVertical8FullGamma0Scalar(b *testing.B) {
	tmp := benchWarpVTmp()
	dst, _ := testPlane(48, 48, 1, 48)
	const reduceBitsVert = round1Bits
	const offsetBitsVert = 8 + 2*filterBits - round0Bits
	b.ResetTimer()
	for range b.N {
		warpVertical8FullGamma0(dst, &tmp, 16, 16, 1, 2, -700, -64, reduceBitsVert, offsetBitsVert)
	}
}

func BenchmarkWarpVertical8FullGamma0SIMD(b *testing.B) {
	tmp := benchWarpVTmp()
	dst, _ := testPlane(48, 48, 1, 48)
	const reduceBitsVert = round1Bits
	const offsetBitsVert = 8 + 2*filterBits - round0Bits
	b.ResetTimer()
	for range b.N {
		warpVertical8FullGamma0SIMD(dst, &tmp, 16, 16, 1, 2, -700, -64, reduceBitsVert, offsetBitsVert)
	}
}

func BenchmarkWarpVertical8FullScalar(b *testing.B) {
	tmp := benchWarpVTmp()
	dst, _ := testPlane(48, 48, 1, 48)
	const reduceBitsVert = round1Bits
	const offsetBitsVert = 8 + 2*filterBits - round0Bits
	b.ResetTimer()
	for range b.N {
		warpVertical8Full(dst, &tmp, 16, 16, 1, 2, -700, 24, -64, reduceBitsVert, offsetBitsVert)
	}
}

func BenchmarkWarpVertical8FullSIMD(b *testing.B) {
	tmp := benchWarpVTmp()
	dst, _ := testPlane(48, 48, 1, 48)
	const reduceBitsVert = round1Bits
	const offsetBitsVert = 8 + 2*filterBits - round0Bits
	b.ResetTimer()
	for range b.N {
		warpVertical8FullSIMD(dst, &tmp, 16, 16, 1, 2, -700, 24, -64, reduceBitsVert, offsetBitsVert)
	}
}
