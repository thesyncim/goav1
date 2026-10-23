// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && (arm64 || amd64) && !purego

package dsp

import (
	"bytes"
	"math/rand"
	"testing"
)

func TestAddConstantPlaneBlock8SIMDMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	residuals := []int16{-32768, -1000, -256, -255, -254, -128, -1, 0, 1, 127, 254, 255, 256, 1000, 32767}
	for _, w := range []int{4, 8, 12, 16, 24, 32, 40, 64} {
		for _, h := range []int{1, 2, 3, 4, 7, 8, 16, 64} {
			for i := 0; i < 24; i++ {
				r := int16(rng.Intn(65536) - 32768)
				if i < len(residuals) {
					r = residuals[i]
				}
				stride := w + rng.Intn(9)
				buf := make([]byte, (h-1)*stride+w+8)
				rng.Read(buf)
				want := append([]byte(nil), buf...)
				got := append([]byte(nil), buf...)
				addConstantPlaneBlock8PureGo(want, stride, w, h, r)
				addConstantPlaneBlock8SIMD(got, stride, w, h, r)
				if !bytes.Equal(got, want) {
					t.Fatalf("w=%d h=%d stride=%d r=%d: SIMD differs from the scalar reference", w, h, stride, r)
				}
			}
		}
	}
}
