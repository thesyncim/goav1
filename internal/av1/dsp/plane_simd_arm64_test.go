// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package dsp

import (
	"math/rand"
	"testing"
)

// makeResidualCase builds a destination block (height*stride bytes, tight
// residual) plus a matching residual buffer, filled from rng. resExtreme mixes
// in int16 saturation edges so the byte-exactness argument is actually exercised.
func makeResidualCase(rng *rand.Rand, width, height, stride int, resExtreme bool) ([]byte, []int16) {
	dst := make([]byte, height*stride)
	for i := range dst {
		dst[i] = byte(rng.Intn(256))
	}
	res := make([]int16, height*width)
	for i := range res {
		switch {
		case resExtreme && rng.Intn(8) == 0:
			// Saturation edges and near-edges.
			res[i] = []int16{-32768, -32767, -300, -1, 0, 1, 300, 32767}[rng.Intn(8)]
		default:
			res[i] = int16(rng.Intn(1024) - 512)
		}
	}
	return dst, res
}

func mkBlock(pix []byte, stride, width, height int) planeBlock {
	return planeBlock{pix: pix, stride: stride, width: width, height: height, rowBytes: width}
}

// TestAddResidualSIMDMatchesScalar proves the Go-native-SIMD 8-bit residual-add
// is byte-identical to the scalar reference across widths, heights, strides, and
// residual ranges (including int16 saturation edges).
func TestAddResidualSIMDMatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(0x5eed))
	widths := []int{8, 16, 24, 32, 64}
	heights := []int{1, 2, 4, 8, 16}
	for iter := 0; iter < 4000; iter++ {
		w := widths[rng.Intn(len(widths))]
		h := heights[rng.Intn(len(heights))]
		stride := w + rng.Intn(3)*8 // exercise stride != width
		if stride < w {
			stride = w
		}
		dstScalar, res := makeResidualCase(rng, w, h, stride, true)
		dstSIMD := make([]byte, len(dstScalar))
		copy(dstSIMD, dstScalar)

		addResidualPlaneBlockPureGo(mkBlock(dstScalar, stride, w, h), 1, 255, w, res, w)
		addResidualPlaneBlockSIMD(mkBlock(dstSIMD, stride, w, h), 1, 255, w, res, w)

		for i := range dstScalar {
			if dstScalar[i] != dstSIMD[i] {
				t.Fatalf("mismatch iter=%d w=%d h=%d stride=%d at byte %d: scalar=%d simd=%d",
					iter, w, h, stride, i, dstScalar[i], dstSIMD[i])
			}
		}
	}
}

// exhaustive per-pixel check: every (dst, residual) pair through both paths.
func TestAddResidualSIMDExhaustivePixel(t *testing.T) {
	// width 8, height 1, one row; sweep residual over the full int16 range for a
	// fixed dst pattern, and sweep dst 0..255 for representative residuals.
	for _, r := range []int{-32768, -32767, -256, -1, 0, 1, 255, 256, 32766, 32767} {
		res := make([]int16, 8)
		for i := range res {
			res[i] = int16(r)
		}
		for base := 0; base < 256; base++ {
			dstA := make([]byte, 8)
			for i := range dstA {
				dstA[i] = byte((base + i*17) & 0xff)
			}
			dstB := make([]byte, 8)
			copy(dstB, dstA)
			addResidualPlaneBlockPureGo(mkBlock(dstA, 8, 8, 1), 1, 255, 8, res, 8)
			addResidualPlaneBlockSIMD(mkBlock(dstB, 8, 8, 1), 1, 255, 8, res, 8)
			for i := range dstA {
				if dstA[i] != dstB[i] {
					t.Fatalf("pixel mismatch res=%d base=%d i=%d: scalar=%d simd=%d", r, base, i, dstA[i], dstB[i])
				}
			}
		}
	}
}

// Three-way benchmark: scalar reference vs Go-native SIMD vs the production NEON
// asm (addResidualPlaneBlockImpl resolves to the hand-written kernel on arm64).
func benchResidual(b *testing.B, w, h int, fn func(planeBlock, int, uint16, int, []int16, int)) {
	rng := rand.New(rand.NewSource(1))
	stride := w
	dst, res := makeResidualCase(rng, w, h, stride, false)
	work := make([]byte, len(dst))
	b.SetBytes(int64(w * h))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		copy(work, dst)
		fn(mkBlock(work, stride, w, h), 1, 255, w, res, w)
	}
}

func BenchmarkAddResidual32x32_Scalar(b *testing.B) { benchResidual(b, 32, 32, addResidualPlaneBlockPureGo) }
func BenchmarkAddResidual32x32_SIMD(b *testing.B)   { benchResidual(b, 32, 32, addResidualPlaneBlockSIMD) }
func BenchmarkAddResidual32x32_ASM(b *testing.B)    { benchResidual(b, 32, 32, addResidualPlaneBlockImpl) }
func BenchmarkAddResidual64x64_Scalar(b *testing.B) { benchResidual(b, 64, 64, addResidualPlaneBlockPureGo) }
func BenchmarkAddResidual64x64_SIMD(b *testing.B)   { benchResidual(b, 64, 64, addResidualPlaneBlockSIMD) }
func BenchmarkAddResidual64x64_ASM(b *testing.B)    { benchResidual(b, 64, 64, addResidualPlaneBlockImpl) }
func BenchmarkAddResidual16x16_Scalar(b *testing.B) { benchResidual(b, 16, 16, addResidualPlaneBlockPureGo) }
func BenchmarkAddResidual16x16_SIMD(b *testing.B)   { benchResidual(b, 16, 16, addResidualPlaneBlockSIMD) }
func BenchmarkAddResidual16x16_ASM(b *testing.B)    { benchResidual(b, 16, 16, addResidualPlaneBlockImpl) }

// _Naive variants use the first un-tuned 8-wide port, to show the tuning delta.
func BenchmarkAddResidual32x32_Naive(b *testing.B) { benchResidual(b, 32, 32, addResidualPlaneBlockSIMDNaive) }
func BenchmarkAddResidual64x64_Naive(b *testing.B) { benchResidual(b, 64, 64, addResidualPlaneBlockSIMDNaive) }
func BenchmarkAddResidual16x16_Naive(b *testing.B) { benchResidual(b, 16, 16, addResidualPlaneBlockSIMDNaive) }

// Also verify the naive port stays byte-exact.
func TestAddResidualSIMDNaiveMatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(0xd00d))
	for iter := 0; iter < 2000; iter++ {
		w := []int{8, 16, 32, 64}[rng.Intn(4)]
		h := 1 + rng.Intn(16)
		stride := w + rng.Intn(2)*8
		dstA, res := makeResidualCase(rng, w, h, stride, true)
		dstB := make([]byte, len(dstA))
		copy(dstB, dstA)
		addResidualPlaneBlockPureGo(mkBlock(dstA, stride, w, h), 1, 255, w, res, w)
		addResidualPlaneBlockSIMDNaive(mkBlock(dstB, stride, w, h), 1, 255, w, res, w)
		for i := range dstA {
			if dstA[i] != dstB[i] {
				t.Fatalf("naive mismatch iter=%d w=%d h=%d at %d: %d vs %d", iter, w, h, i, dstA[i], dstB[i])
			}
		}
	}
}
