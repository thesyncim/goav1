// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

package cdef

import (
	"math/rand"
	"testing"
)

// TestFilterBlockNEONMatchesPureGo proves the NEON CDEF block filter
// bit-exact with the pure-Go reference across directions, strengths,
// dampings, coefficient shifts, block shapes, and inputs salted with the
// VeryLarge border sentinel.
func TestFilterBlockNEONMatchesPureGo(t *testing.T) {
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
		filterBlockNEON(got, 16, 0, input, origin, params)
		for i := range want {
			if want[i] != got[i] {
				t.Fatalf("trial %d params=%+v: dst[%d] neon %d want %d", trial, params, i, got[i], want[i])
			}
		}
	}
}

func TestFilterBlockNEONStrengthCasesMatchPureGo(t *testing.T) {
	cases := []struct {
		name      string
		primary   uint8
		secondary uint8
	}{
		{name: "primary-only", primary: 13, secondary: 0},
		{name: "secondary-only", primary: 0, secondary: 4},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			for coeffShift := uint8(0); coeffShift <= 2; coeffShift++ {
				for direction := uint8(0); direction < 8; direction++ {
					input := makeCDEFBlockInput(newCDEFRandom(cdefDeterministicSeed^uint32(direction)^uint32(coeffShift)<<8), 8+int(coeffShift), int(direction)&0xf, int(coeffShift)+1)
					params := BlockFilterParams{
						PrimaryStrength:   tc.primary << coeffShift,
						SecondaryStrength: tc.secondary << coeffShift,
						Direction:         direction,
						PrimaryDamping:    4 + coeffShift,
						SecondaryDamping:  4 + coeffShift,
						CoeffShift:        coeffShift,
						Width:             8,
						Height:            8,
					}
					origin := cdefBlockOrigin()
					want := make([]uint16, 16*16)
					got := make([]uint16, 16*16)
					filterBlockPureGo(want, 16, 0, input, origin, params)
					filterBlockNEON(got, 16, 0, input, origin, params)
					for i := range want {
						if want[i] != got[i] {
							t.Fatalf("%s coeffShift=%d direction=%d dst[%d] neon %d want %d", tc.name, coeffShift, direction, i, got[i], want[i])
						}
					}
				}
			}
		})
	}
}

func benchFilterNEON(b *testing.B, fn func([]uint16, int, int, []uint16, int, BlockFilterParams)) {
	benchFilterNEONParams(b, BlockFilterParams{
		PrimaryStrength: 9, SecondaryStrength: 2, Direction: 5,
		PrimaryDamping: 5, SecondaryDamping: 4, Width: 8, Height: 8,
	}, fn)
}

func benchFilterNEONParams(b *testing.B, params BlockFilterParams, fn func([]uint16, int, int, []uint16, int, BlockFilterParams)) {
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

func BenchmarkFilterBlock8x8NEON(b *testing.B)   { benchFilterNEON(b, filterBlockNEON) }
func BenchmarkFilterBlock8x8PureGo(b *testing.B) { benchFilterNEON(b, filterBlockPureGo) }

func BenchmarkFilterBlock8x8PrimaryOnlyDispatchNEON(b *testing.B) {
	benchFilterNEONParams(b, BlockFilterParams{
		PrimaryStrength: 13, SecondaryStrength: 0, Direction: 5,
		PrimaryDamping: 5, SecondaryDamping: 4, Width: 8, Height: 8,
	}, filterBlockNEON)
}

func BenchmarkFilterBlock8x8PrimaryOnlyGenericDispatchNEON(b *testing.B) {
	benchFilterNEONParams(b, BlockFilterParams{
		PrimaryStrength: 13, SecondaryStrength: 0, Direction: 5,
		PrimaryDamping: 5, SecondaryDamping: 4, Width: 8, Height: 8,
	}, filterBlockNEONGeneric8)
}

func BenchmarkFilterBlock8x8SecondaryOnlyDispatchNEON(b *testing.B) {
	benchFilterNEONParams(b, BlockFilterParams{
		PrimaryStrength: 0, SecondaryStrength: 4, Direction: 5,
		PrimaryDamping: 5, SecondaryDamping: 4, Width: 8, Height: 8,
	}, filterBlockNEON)
}

func BenchmarkFilterBlock8x8PrimaryOnlyGenericNEON(b *testing.B) {
	benchFilterBlock8Asm(b, BlockFilterParams{
		PrimaryStrength: 13, SecondaryStrength: 0, Direction: 5,
		PrimaryDamping: 5, SecondaryDamping: 4, Width: 8, Height: 8,
	}, cdefFilterBlock8NEON)
}

func BenchmarkFilterBlock8x8PrimaryOnlySplitNEON(b *testing.B) {
	benchFilterBlock8Asm(b, BlockFilterParams{
		PrimaryStrength: 13, SecondaryStrength: 0, Direction: 5,
		PrimaryDamping: 5, SecondaryDamping: 4, Width: 8, Height: 8,
	}, cdefFilterBlock8PrimaryNEON)
}

func BenchmarkFilterBlock8x8SecondaryOnlyGenericNEON(b *testing.B) {
	benchFilterBlock8Asm(b, BlockFilterParams{
		PrimaryStrength: 0, SecondaryStrength: 4, Direction: 5,
		PrimaryDamping: 5, SecondaryDamping: 4, Width: 8, Height: 8,
	}, cdefFilterBlock8NEON)
}

func benchFilterBlock8Asm(b *testing.B, params BlockFilterParams, fn func(*filterBlockNEONCtx)) {
	input := make([]uint16, BStride*24)
	for i := range input {
		input[i] = uint16(i * 7 % 256)
	}
	dst := make([]uint16, 16*16)
	ctx := makeFilterBlockNEONCtx(dst, 16, 0, input, BStride*8+16, params, int(params.PrimaryStrength), int(params.SecondaryStrength))
	b.ReportAllocs()
	for b.Loop() {
		fn(&ctx)
	}
}

func filterBlockNEONGeneric8(dst []uint16, dstStride int, dstOrigin int, input []uint16, inputOrigin int, params BlockFilterParams) {
	ctx := makeFilterBlockNEONCtx(dst, dstStride, dstOrigin, input, inputOrigin, params, int(params.PrimaryStrength), int(params.SecondaryStrength))
	cdefFilterBlock8NEON(&ctx)
}
