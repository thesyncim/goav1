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

func benchFilterNEON(b *testing.B, fn func([]uint16, int, int, []uint16, int, BlockFilterParams)) {
	input := make([]uint16, BStride*24)
	for i := range input {
		input[i] = uint16(i * 7 % 256)
	}
	dst := make([]uint16, 16*16)
	params := BlockFilterParams{
		PrimaryStrength: 9, SecondaryStrength: 2, Direction: 5,
		PrimaryDamping: 5, SecondaryDamping: 4, Width: 8, Height: 8,
	}
	b.ReportAllocs()
	for b.Loop() {
		fn(dst, 16, 0, input, BStride*8+16, params)
	}
}

func BenchmarkFilterBlock8x8NEON(b *testing.B)   { benchFilterNEON(b, filterBlockNEON) }
func BenchmarkFilterBlock8x8PureGo(b *testing.B) { benchFilterNEON(b, filterBlockPureGo) }
