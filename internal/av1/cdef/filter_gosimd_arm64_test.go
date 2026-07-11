// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package cdef

import (
	"math/rand"
	"testing"
)

// TestFilterBlockSIMD16MatchesPureGo checks the int16 8-wide Go SIMD CDEF filter
// is byte-identical to the scalar reference across random strengths, directions,
// dampings, coeff shifts and widths.
func TestFilterBlockSIMD16MatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	input := make([]uint16, BStride*40)
	origin := BStride*8 + 16
	want := make([]uint16, 16*16)
	got := make([]uint16, 16*16)
	for iter := 0; iter < 6000; iter++ {
		bd := []int{8, 10, 12}[rng.Intn(3)]
		coeffShift := bd - 8
		maxv := (1 << bd) - 1
		for i := range input {
			input[i] = uint16(rng.Intn(maxv + 1))
		}
		w := []uint8{8, 16}[rng.Intn(2)]
		h := uint8(8)
		params := BlockFilterParams{
			PrimaryStrength:   uint8(rng.Intn(16) << coeffShift),
			SecondaryStrength: uint8(rng.Intn(4) << coeffShift),
			Direction:         uint8(rng.Intn(8)),
			PrimaryDamping:    uint8(3 + rng.Intn(4) + coeffShift),
			SecondaryDamping:  uint8(3 + rng.Intn(3) + coeffShift),
			CoeffShift:        uint8(coeffShift),
			Width:             w,
			Height:            h,
		}
		for i := range want {
			want[i] = 0
			got[i] = 0
		}
		filterBlockPureGo(want, int(w), 0, input, origin, params)
		filterBlockSIMD16(got, int(w), 0, input, origin, params)
		for i := 0; i < int(w)*int(h); i++ {
			if want[i] != got[i] {
				t.Fatalf("iter=%d i=%d want=%d got=%d params=%+v", iter, i, want[i], got[i], params)
			}
		}
	}
}

func benchFilterBlock(b *testing.B, fn func([]uint16, int, int, []uint16, int, BlockFilterParams)) {
	rng := rand.New(rand.NewSource(7))
	input := make([]uint16, BStride*40)
	for i := range input {
		input[i] = uint16(rng.Intn(256))
	}
	params := BlockFilterParams{
		PrimaryStrength: 12, SecondaryStrength: 2, Direction: 3,
		PrimaryDamping: 5, SecondaryDamping: 4, CoeffShift: 0, Width: 8, Height: 8,
	}
	origin := BStride*8 + 16
	dst := make([]uint16, 8*8)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fn(dst, 8, 0, input, origin, params)
	}
}
func BenchmarkFilterBlock8x8_SIMD(b *testing.B)   { benchFilterBlock(b, filterBlockSIMD16) }
func BenchmarkFilterBlock8x8_NEON(b *testing.B)   { benchFilterBlock(b, filterBlockNEON) }
func BenchmarkFilterBlock8x8_PureGo(b *testing.B) { benchFilterBlock(b, filterBlockPureGo) }
