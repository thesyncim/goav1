// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && (arm64 || amd64) && !purego

package transform

import (
	"math/rand"
	"runtime"
	"simd/archsimd"
	"testing"
)

func fwd64SIMDAvailable() bool { return runtime.GOARCH == "arm64" || archsimd.X86.AVX2() }

// TestForwardDCT64x64SIMDMatchesScalar differential-tests the SVT-shaped
// 64x64 kernel (highbd_fdct64_N2_x4_neon passes) against the libaom scalar
// reference over 8-bit residuals, extremes included.
func TestForwardDCT64x64SIMDMatchesScalar(t *testing.T) {
	if !fwd64SIMDAvailable() {
		t.Skip("no vector backend")
	}
	rng := rand.New(rand.NewSource(64))
	var residual [64 * 64]int16
	for trial := 0; trial < 400; trial++ {
		switch trial % 5 {
		case 0:
			for i := range residual {
				residual[i] = int16(rng.Intn(511) - 255)
			}
		case 1:
			v := int16(255)
			if trial&8 != 0 {
				v = -255
			}
			for i := range residual {
				residual[i] = v
			}
		case 2:
			for i := range residual {
				if (i/64+i%64+trial)&1 == 0 {
					residual[i] = 255
				} else {
					residual[i] = -255
				}
			}
		case 3:
			for i := range residual {
				residual[i] = int16(rng.Intn(21) - 10)
			}
		default:
			for i := range residual {
				if rng.Intn(2) == 0 {
					residual[i] = 255
				} else {
					residual[i] = -255
				}
			}
		}
		var want, got [32 * 32]int32
		for i := range got {
			got[i] = -12345
		}
		forwardDCT64x64PureGo(want[:], 32, residual[:], 64)
		forwardDCT64x64SIMD(got[:], 32, residual[:], 64)
		if want != got {
			for i := range want {
				if want[i] != got[i] {
					t.Fatalf("trial %d: coeff[col %d][row %d] = %d, want %d", trial, i/32, i%32, got[i], want[i])
				}
			}
		}
	}
}

func BenchmarkForwardDCT64x64SIMD(b *testing.B) {
	if !fwd64SIMDAvailable() {
		b.Skip("no vector backend")
	}
	var residual [64 * 64]int16
	rng := rand.New(rand.NewSource(1))
	for i := range residual {
		residual[i] = int16(rng.Intn(511) - 255)
	}
	var coeff [32 * 32]int32
	b.ResetTimer()
	for b.Loop() {
		forwardDCT64x64SIMD(coeff[:], 32, residual[:], 64)
	}
}
