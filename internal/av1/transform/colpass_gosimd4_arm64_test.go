// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package transform

import (
	"math/rand"
	"testing"
)

func benchDCT8x4(b *testing.B, fn func([]int32, int, int32, int32)) {
	rng := rand.New(rand.NewSource(9))
	const stride = 4
	buf := make([]int32, 8*stride+8)
	for i := range buf {
		buf[i] = int32(rng.Intn(1<<13) - (1 << 12))
	}
	work := make([]int32, len(buf))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		copy(work, buf)
		fn(work, stride, -(1 << 14), (1<<14)-1)
	}
}

func BenchmarkDCT8x4_ASMCol4(b *testing.B) {
	benchDCT8x4(b, inverseDCT8Col4SIMDAdapter)
}

func BenchmarkDCT8x4_ASMCol2(b *testing.B) {
	benchDCT8x4(b, func(buf []int32, s int, mn, mx int32) {
		inverseDCT8Col2SIMDAdapter(buf, s, mn, mx)
		inverseDCT8Col2SIMDAdapter(buf[2:], s, mn, mx)
	})
}

func benchDCT16x4(b *testing.B, fn func([]int32, int, int32, int32)) {
	rng := rand.New(rand.NewSource(9))
	const stride = 4
	buf := make([]int32, 16*stride+8)
	for i := range buf {
		buf[i] = int32(rng.Intn(1<<13) - (1 << 12))
	}
	work := make([]int32, len(buf))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		copy(work, buf)
		fn(work, stride, -(1 << 14), (1<<14)-1)
	}
}

func BenchmarkDCT16x4_ASMCol4(b *testing.B) {
	benchDCT16x4(b, inverseDCT16Col4SIMDAdapter)
}

func BenchmarkDCT16x4_ASMCol2(b *testing.B) {
	benchDCT16x4(b, func(buf []int32, s int, mn, mx int32) {
		inverseDCT16Col2SIMDAdapter(buf, s, mn, mx)
		inverseDCT16Col2SIMDAdapter(buf[2:], s, mn, mx)
	})
}
