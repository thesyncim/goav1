// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && (arm64 || amd64) && !purego

package transform

import (
	"math/rand"
	"testing"
)

// TestForwardBlock8x8SIMDMatchesPureGo checks the Go SIMD 8x8 kernels, called
// directly, against the scalar oracle over random 8-bit residuals, the
// extreme sign patterns, and non-packed strides.
func TestForwardBlock8x8SIMDMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(401))
	const resStride, coeffStride = 19, 13
	kernels := []struct {
		name string
		simd func([]int32, int, []int16, int, []int32)
		pure func([]int32, int, []int16, int, []int32)
	}{
		{"DCT_DCT", func(c []int32, cs int, r []int16, rs int, _ []int32) { forwardDCT8x8SIMD(c, cs, r, rs) },
			func(c []int32, cs int, r []int16, rs int, _ []int32) { forwardDCT8x8PureGo(c, cs, r, rs) }},
		{"ADST_DCT", forwardBlock8x8ADSTDCTSIMD, forwardBlock8x8ADSTDCTPureGo},
		{"DCT_ADST", forwardBlock8x8DCTADSTSIMD, forwardBlock8x8DCTADSTPureGo},
		{"ADST_ADST", forwardBlock8x8ADSTADSTSIMD, forwardBlock8x8ADSTADSTPureGo},
	}
	extremes := []func(r, c int) int16{
		func(r, c int) int16 { return 255 },
		func(r, c int) int16 { return -255 },
		func(r, c int) int16 {
			if (r+c)&1 == 0 {
				return 255
			}
			return -255
		},
		func(r, c int) int16 {
			if r&1 == 0 {
				return 255
			}
			return -255
		},
	}
	for _, k := range kernels {
		residual := make([]int16, resStride*8)
		got := make([]int32, coeffStride*8)
		want := make([]int32, coeffStride*8)
		var scratch [64]int32
		check := func(trial int) {
			t.Helper()
			clear(got)
			clear(want)
			k.simd(got, coeffStride, residual, resStride, scratch[:])
			k.pure(want, coeffStride, residual, resStride, scratch[:])
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("%s trial %d: coeff[%d] simd %d want %d", k.name, trial, i, got[i], want[i])
				}
			}
		}
		for _, pat := range extremes {
			for r := range 8 {
				for c := range 8 {
					residual[r*resStride+c] = pat(r, c)
				}
			}
			check(-1)
		}
		for trial := range 3000 {
			for i := range residual {
				residual[i] = int16(rng.Intn(511)) - 255
			}
			check(trial)
		}
	}
}

// TestForwardDCT8x8SIMDZeroAlloc keeps the 8x8 SIMD driver allocation-free.
func TestForwardDCT8x8SIMDZeroAlloc(t *testing.T) {
	var residual [64]int16
	for i := range residual {
		residual[i] = int16(i%511) - 255
	}
	var coeff [64]int32
	var scratch [64]int32
	allocs := testing.AllocsPerRun(100, func() {
		forwardDCT8x8SIMDGuarded(coeff[:], 8, residual[:], 8)
		forwardBlock8x8ADSTDCTSIMDGuarded(coeff[:], 8, residual[:], 8, scratch[:])
	})
	if allocs != 0 {
		t.Fatalf("SIMD 8x8 forward kernels allocated %v objects/run, want 0", allocs)
	}
}
