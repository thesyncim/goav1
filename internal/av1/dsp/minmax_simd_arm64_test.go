// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package dsp

import (
	"math/rand"
	"testing"
)

// TestMinMaxAbsDiff8x8SIMDMatchesScalar proves the Go-native-SIMD 8x8 min/max
// abs-diff kernel is identical to the scalar reference across strides and
// content (including all-equal, max-contrast, and random blocks).
func TestMinMaxAbsDiff8x8SIMDMatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(0x33cc))
	for iter := 0; iter < 5000; iter++ {
		aStride := 8 + rng.Intn(4)
		bStride := 8 + rng.Intn(4)
		a := make([]byte, 8*aStride)
		b := make([]byte, 8*bStride)
		switch rng.Intn(4) {
		case 0: // random
			for i := range a {
				a[i] = byte(rng.Intn(256))
			}
			for i := range b {
				b[i] = byte(rng.Intn(256))
			}
		case 1: // identical blocks -> min==max==0
			for i := range a {
				a[i] = byte(rng.Intn(256))
			}
			copy(b, a)
		case 2: // max contrast: a all 0, b all 255
			for i := range a {
				a[i] = 0
			}
			for i := range b {
				b[i] = 255
			}
		default: // small diffs
			for i := range a {
				a[i] = 100
			}
			for i := range b {
				b[i] = byte(100 + rng.Intn(7) - 3)
			}
		}
		wMin, wMax, wErr := minMaxAbsDiff8x8PureGo(a, aStride, b, bStride, 1)
		gMin, gMax, gErr := minMaxAbsDiff8x8SIMD(a, aStride, b, bStride, 1)
		if wErr != gErr || wMin != gMin || wMax != gMax {
			t.Fatalf("iter=%d: scalar(min=%d max=%d err=%v) simd(min=%d max=%d err=%v)",
				iter, wMin, wMax, wErr, gMin, gMax, gErr)
		}
	}
}

func benchMinMax(b *testing.B, fn func([]byte, int, []byte, int, int) (uint16, uint16, error)) {
	rng := rand.New(rand.NewSource(4))
	a := make([]byte, 8*8)
	bb := make([]byte, 8*8)
	for i := range a {
		a[i] = byte(rng.Intn(256))
		bb[i] = byte(rng.Intn(256))
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fn(a, 8, bb, 8, 1)
	}
}

func BenchmarkMinMax8x8_Scalar(b *testing.B) { benchMinMax(b, minMaxAbsDiff8x8PureGo) }
func BenchmarkMinMax8x8_SIMD(b *testing.B)   { benchMinMax(b, minMaxAbsDiff8x8SIMD) }
func BenchmarkMinMax8x8_ASM(b *testing.B)    { benchMinMax(b, minMaxAbsDiff8x8Impl) }
