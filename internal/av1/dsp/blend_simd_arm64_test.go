// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package dsp

import (
	"math/rand"
	"testing"
)

func makeBlendCase(rng *rand.Rand, w, h int) (dst, s0, s1 []uint16, mask []uint8) {
	dst = make([]uint16, w*h)
	s0 = make([]uint16, w*h)
	s1 = make([]uint16, w*h)
	mask = make([]uint8, w*h)
	for i := range s0 {
		s0[i] = uint16(rng.Intn(256)) // 8-bit predictions
		s1[i] = uint16(rng.Intn(256))
		mask[i] = uint8(rng.Intn(65)) // valid alpha 0..64
	}
	return
}

func blendArgs(dst, s0, s1 []uint16, mask []uint8, w, h int) blendA64MaskArgs {
	return blendA64MaskArgs{
		dst: dst, dstStride: w, src0: s0, src0Stride: w, src1: s1, src1Stride: w,
		mask: mask, maskStride: w, width: w, height: h, max: 255, subX: false, subY: false,
	}
}

// TestBlendA64MaskSIMDMatchesScalar proves the Go-native-SIMD blend is
// byte-identical to the scalar reference on the 8-bit non-subsampled path.
func TestBlendA64MaskSIMDMatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(0xb1e2))
	widths := []int{16, 32, 48, 64}
	heights := []int{1, 2, 4, 8, 16}
	for iter := 0; iter < 4000; iter++ {
		w := widths[rng.Intn(len(widths))]
		h := heights[rng.Intn(len(heights))]
		dstA, s0, s1, mask := makeBlendCase(rng, w, h)
		dstB := make([]uint16, len(dstA))
		copy(dstB, dstA)
		if !blendA64MaskPureGo(blendArgs(dstA, s0, s1, mask, w, h)) {
			t.Fatalf("scalar returned false (unexpected) iter=%d", iter)
		}
		if !blendA64MaskSIMD(blendArgs(dstB, s0, s1, mask, w, h)) {
			t.Fatalf("simd returned false (unexpected) iter=%d", iter)
		}
		for i := range dstA {
			if dstA[i] != dstB[i] {
				t.Fatalf("blend mismatch iter=%d w=%d h=%d at %d: scalar=%d simd=%d", iter, w, h, i, dstA[i], dstB[i])
			}
		}
	}
}

func benchBlend(b *testing.B, w, h int, fn func(blendA64MaskArgs) bool) {
	rng := rand.New(rand.NewSource(3))
	dst, s0, s1, mask := makeBlendCase(rng, w, h)
	b.SetBytes(int64(w * h))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fn(blendArgs(dst, s0, s1, mask, w, h))
	}
}

func BenchmarkBlend32x32_Scalar(b *testing.B) { benchBlend(b, 32, 32, blendA64MaskPureGo) }
func BenchmarkBlend32x32_SIMD(b *testing.B)   { benchBlend(b, 32, 32, blendA64MaskSIMD) }
func BenchmarkBlend32x32_ASM(b *testing.B)    { benchBlend(b, 32, 32, blendA64MaskImpl) }
func BenchmarkBlend64x64_Scalar(b *testing.B) { benchBlend(b, 64, 64, blendA64MaskPureGo) }
func BenchmarkBlend64x64_SIMD(b *testing.B)   { benchBlend(b, 64, 64, blendA64MaskSIMD) }
func BenchmarkBlend64x64_ASM(b *testing.B)    { benchBlend(b, 64, 64, blendA64MaskImpl) }
