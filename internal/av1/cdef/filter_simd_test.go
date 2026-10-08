// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && (arm64 || amd64) && !purego

package cdef

import (
	"math/rand"
	"testing"
)

// TestFilterBlockSIMDMatchesPureGo proves the Go SIMD CDEF block filter
// bit-exact with the pure-Go reference across directions, strengths,
// dampings, coefficient shifts, block shapes, and inputs salted with the
// VeryLarge border sentinel.
func TestFilterBlockSIMDMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(97))
	input := make([]uint16, BStride*24)
	for trial := 0; trial < 4000; trial++ {
		maxSample := 255
		coeffShift := rng.Intn(3)
		if coeffShift > 0 {
			maxSample = (1 << (8 + coeffShift)) - 1
		}
		for i := range input {
			if rng.Intn(13) == 0 {
				input[i] = VeryLarge
			} else {
				input[i] = uint16(rng.Intn(maxSample + 1))
			}
		}
		var width, height int
		switch rng.Intn(3) {
		case 0:
			width, height = 8, 8
		case 1:
			width, height = 8, 4
		default:
			width, height = 4, 4
		}
		params := BlockFilterParams{
			PrimaryStrength:   uint8(rng.Intn(16) << coeffShift),
			SecondaryStrength: uint8([]int{0, 1, 2, 4}[rng.Intn(4)] << coeffShift),
			Direction:         uint8(rng.Intn(8)),
			PrimaryDamping:    uint8(3 + coeffShift + rng.Intn(3)),
			SecondaryDamping:  uint8(3 + coeffShift + rng.Intn(3)),
			CoeffShift:        uint8(coeffShift),
			Width:             uint8(width),
			Height:            uint8(height),
		}
		origin := BStride*8 + 16
		want := make([]uint16, 16*16)
		got := make([]uint16, 16*16)
		filterBlockPureGo(want, 16, 0, input, origin, params)
		filterBlockSIMD(got, 16, 0, input, origin, params)
		for i := range want {
			if want[i] != got[i] {
				t.Fatalf("trial %d params=%+v: dst[%d] simd %d want %d", trial, params, i, got[i], want[i])
			}
		}
	}
}

// TestFilterBlockSIMDStrengthCasesMatchPureGo pins the single-strength
// shapes (primary-only and secondary-only), which the kernel serves with the
// same row loop as the fused case but with the other side disabled.
func TestFilterBlockSIMDStrengthCasesMatchPureGo(t *testing.T) {
	cases := []struct {
		name      string
		primary   uint8
		secondary uint8
	}{
		{name: "primary-only", primary: 13, secondary: 0},
		{name: "secondary-only", primary: 0, secondary: 4},
		{name: "fused", primary: 9, secondary: 2},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			shapes := []struct {
				width  int
				height int
			}{
				{width: 8, height: 8},
				{width: 4, height: 8},
				{width: 4, height: 4},
			}
			for coeffShift := uint8(0); coeffShift <= 2; coeffShift++ {
				for direction := uint8(0); direction < 8; direction++ {
					for _, shape := range shapes {
						seed := cdefDeterministicSeed ^
							uint32(direction) ^
							(uint32(coeffShift) << 8) ^
							(uint32(shape.width) << 16) ^
							(uint32(shape.height) << 20)
						input := makeCDEFBlockInput(newCDEFRandom(seed), 8+int(coeffShift), int(direction)&0xf, int(coeffShift)+1)
						params := BlockFilterParams{
							PrimaryStrength:   tc.primary << coeffShift,
							SecondaryStrength: tc.secondary << coeffShift,
							Direction:         direction,
							PrimaryDamping:    4 + coeffShift,
							SecondaryDamping:  4 + coeffShift,
							CoeffShift:        coeffShift,
							Width:             uint8(shape.width),
							Height:            uint8(shape.height),
						}
						origin := cdefBlockOrigin()
						want := make([]uint16, 16*16)
						got := make([]uint16, 16*16)
						filterBlockPureGo(want, 16, 0, input, origin, params)
						filterBlockSIMD(got, 16, 0, input, origin, params)
						for i := range want {
							if want[i] != got[i] {
								t.Fatalf("%s %dx%d coeffShift=%d direction=%d dst[%d] simd %d want %d", tc.name, shape.width, shape.height, coeffShift, direction, i, got[i], want[i])
							}
						}
					}
				}
			}
		})
	}
}

func benchFilterSIMD(b *testing.B, fn func([]uint16, int, int, []uint16, int, BlockFilterParams)) {
	benchFilterSIMDParams(b, BlockFilterParams{
		PrimaryStrength: 9, SecondaryStrength: 2, Direction: 5,
		PrimaryDamping: 5, SecondaryDamping: 4, Width: 8, Height: 8,
	}, fn)
}

func benchFilterSIMDParams(b *testing.B, params BlockFilterParams, fn func([]uint16, int, int, []uint16, int, BlockFilterParams)) {
	input := make([]uint16, BStride*24)
	for i := range input {
		input[i] = uint16(i * 7 % 256)
	}
	dst := make([]uint16, 16*16)
	b.ReportAllocs()
	for b.Loop() {
		fn(dst, 16, 0, input, BStride*8+16, params)
	}
}

func BenchmarkFilterBlock8x8SIMD(b *testing.B)   { benchFilterSIMD(b, filterBlockSIMD) }
func BenchmarkFilterBlock8x8PureGo(b *testing.B) { benchFilterSIMD(b, filterBlockPureGo) }

func BenchmarkFilterBlock8x8PrimaryOnlySIMD(b *testing.B) {
	benchFilterSIMDParams(b, BlockFilterParams{
		PrimaryStrength: 13, SecondaryStrength: 0, Direction: 5,
		PrimaryDamping: 5, SecondaryDamping: 4, Width: 8, Height: 8,
	}, filterBlockSIMD)
}

func BenchmarkFilterBlock8x8SecondaryOnlySIMD(b *testing.B) {
	benchFilterSIMDParams(b, BlockFilterParams{
		PrimaryStrength: 0, SecondaryStrength: 4, Direction: 5,
		PrimaryDamping: 5, SecondaryDamping: 4, Width: 8, Height: 8,
	}, filterBlockSIMD)
}

func BenchmarkFilterBlock4x4PrimaryOnlySIMD(b *testing.B) {
	benchFilterSIMDParams(b, BlockFilterParams{
		PrimaryStrength: 13, SecondaryStrength: 0, Direction: 5,
		PrimaryDamping: 5, SecondaryDamping: 4, Width: 4, Height: 4,
	}, filterBlockSIMD)
}

func BenchmarkFilterBlock4x4SecondaryOnlySIMD(b *testing.B) {
	benchFilterSIMDParams(b, BlockFilterParams{
		PrimaryStrength: 0, SecondaryStrength: 4, Direction: 5,
		PrimaryDamping: 5, SecondaryDamping: 4, Width: 4, Height: 4,
	}, filterBlockSIMD)
}

// TestFilterBlockSIMDClipCorpus targets the clamp path. Uniform random taps
// almost never overshoot, because the fused weights sum to 24 against a
// divisor of 16 and only a near-flat neighbourhood with strong strengths
// exposes the [min, max] clamp. Narrow-spread neighbourhoods with strong
// strengths and injected VeryLarge halo samples hit it directly, including the
// sentinel-skip in the maximum fold.
func TestFilterBlockSIMDClipCorpus(t *testing.T) {
	rng := rand.New(rand.NewSource(0x5ca1ab1e))
	input := make([]uint16, BStride*24)
	origin := cdefBlockOrigin()
	for trial := 0; trial < 6000; trial++ {
		coeffShift := rng.Intn(3)
		maxSample := (1 << (8 + coeffShift)) - 1
		base := rng.Intn(maxSample + 1)
		spread := 1 << rng.Intn(6)
		for i := range input {
			v := min(base+rng.Intn(spread+1), maxSample)
			if rng.Intn(8) == 0 {
				v = VeryLarge
			}
			input[i] = uint16(v)
		}
		var width, height int
		switch rng.Intn(3) {
		case 0:
			width, height = 8, 8
		case 1:
			width, height = 8, 4
		default:
			width, height = 4, 4
		}
		params := BlockFilterParams{
			PrimaryStrength:   uint8(1+rng.Intn(15)) << coeffShift,
			SecondaryStrength: uint8(1+rng.Intn(4)) << coeffShift,
			Direction:         uint8(rng.Intn(8)),
			PrimaryDamping:    uint8(3 + coeffShift + rng.Intn(4)),
			SecondaryDamping:  uint8(3 + coeffShift + rng.Intn(4)),
			CoeffShift:        uint8(coeffShift),
			Width:             uint8(width),
			Height:            uint8(height),
		}
		want := make([]uint16, 16*16)
		got := make([]uint16, 16*16)
		filterBlockPureGo(want, 16, 0, input, origin, params)
		filterBlockSIMD(got, 16, 0, input, origin, params)
		for i := range want {
			if want[i] != got[i] {
				t.Fatalf("trial %d params=%+v spread=%d base=%d: dst[%d] simd %d want %d", trial, params, spread, base, i, got[i], want[i])
			}
		}
	}
}
