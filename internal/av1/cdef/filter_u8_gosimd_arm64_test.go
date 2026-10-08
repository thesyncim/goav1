// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package cdef

import "testing"

func compareFilterBlockU8SIMD(t *testing.T, input []uint16, origin int, params BlockFilterParams) {
	t.Helper()
	const dstStride = 16
	want := make([]byte, dstStride*8)
	got := make([]byte, dstStride*8)
	filterBlockU8PureGo(want, dstStride, 0, input, origin, params)
	filterBlockU8Impl(got, dstStride, 0, input, origin, params)
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("w=%d h=%d dir=%d pri=%d sec=%d damp=%d idx=%d got=%d want=%d",
				params.Width, params.Height, params.Direction, params.PrimaryStrength,
				params.SecondaryStrength, params.PrimaryDamping, i, got[i], want[i])
		}
	}
}

func TestCDEFPrimarySecondaryU8DispatchMatchesPureGo(t *testing.T) {
	origin := cdefBlockOrigin()
	shapes := [...]struct{ width, height int }{{8, 8}, {8, 4}, {4, 8}, {4, 4}}
	for _, boundary := range []int{0, 5, 10, 15} {
		input := makeCDEFBlockInput(newCDEFRandom(cdefDeterministicSeed^0x51D51D), 8, boundary, boundary+1)
		for _, shape := range shapes {
			for _, dir := range []int{0, 3, 7} {
				for _, damping := range []int{3, 6} {
					for _, sec := range cdefSecondaryStrengthCorpus(0)[1:] {
						compareFilterBlockU8SIMD(t, input, origin, BlockFilterParams{
							SecondaryStrength: uint8(sec), Direction: uint8(dir),
							PrimaryDamping: uint8(damping), SecondaryDamping: uint8(damping),
							Width: uint8(shape.width), Height: uint8(shape.height),
						})
					}
					for _, pri := range cdefPrimaryStrengthCorpus(0)[1:] {
						compareFilterBlockU8SIMD(t, input, origin, BlockFilterParams{
							PrimaryStrength: uint8(pri), Direction: uint8(dir),
							PrimaryDamping: uint8(damping), SecondaryDamping: uint8(damping),
							Width: uint8(shape.width), Height: uint8(shape.height),
						})
					}
					for _, strengths := range [][2]int{{4, 2}, {15, 4}} {
						compareFilterBlockU8SIMD(t, input, origin, BlockFilterParams{
							PrimaryStrength: uint8(strengths[0]), SecondaryStrength: uint8(strengths[1]),
							Direction: uint8(dir), PrimaryDamping: uint8(damping), SecondaryDamping: uint8(damping),
							Width: uint8(shape.width), Height: uint8(shape.height),
						})
					}
				}
			}
		}
	}
}

func TestCDEFPrimarySecondaryU8Extremes(t *testing.T) {
	origin := cdefBlockOrigin()
	fills := []func(int) uint16{
		func(int) uint16 { return 0 },
		func(int) uint16 { return 255 },
		func(i int) uint16 {
			if i&1 == 0 {
				return 0
			}
			return 255
		},
	}
	for fi, fill := range fills {
		input := make([]uint16, InputBufferSize)
		for i := range input {
			input[i] = fill(i)
		}
		for i := range input {
			row, col := i/BStride, i%BStride
			if row < VerticalBorder || row >= VerticalBorder+8 || col < HorizontalBorder || col >= HorizontalBorder+8 {
				input[i] = VeryLarge
			}
		}
		for dir := 0; dir < 8; dir++ {
			for _, shape := range [...]struct{ width, height int }{{8, 8}, {4, 4}} {
				for _, strengths := range [][2]uint8{{15, 0}, {0, 4}} {
					compareFilterBlockU8SIMD(t, input, origin, BlockFilterParams{
						PrimaryStrength: strengths[0], SecondaryStrength: strengths[1],
						Direction: uint8(dir), PrimaryDamping: 3, SecondaryDamping: 3,
						Width: uint8(shape.width), Height: uint8(shape.height),
					})
				}
			}
		}
		_ = fi
	}
}

func TestCDEFPrimarySecondaryU8UnitDispatchMatchesPureGo(t *testing.T) {
	blocks := []BlockPosition{{BY: 0, BX: 0}}
	for _, boundary := range []int{0, 15} { // exercise both interior and sentinel-border routes
		input := makeCDEFBlockInput(newCDEFRandom(cdefDeterministicSeed^0x77AA55), 8, boundary, boundary+1)
		for _, strengths := range [][2]int{{0, 4}, {15, 0}, {15, 4}} {
			var directions DirectionGrid
			directions[0][0] = 3
			var variances VarianceGrid
			u := unitFilterParams{
				primaryStrength: strengths[0], secondaryStrength: strengths[1], damping: 5,
				bwLog2: 3, bhLog2: 3, blockWidth: 8, blockHeight: 8,
			}
			want := make([]byte, 64)
			got := make([]byte, 64)
			// Keep the unit-path oracle independent of `filterBlockU8Impl` so
			// changing the selected dispatch cannot make both sides agree by
			// calling the same SIMD kernel.
			dir := 0
			if strengths[0] != 0 {
				dir = int(directions[0][0])
			}
			params := BlockFilterParams{
				PrimaryStrength: uint8(strengths[0]), SecondaryStrength: uint8(strengths[1]),
				Direction: uint8(dir), PrimaryDamping: uint8(u.damping), SecondaryDamping: uint8(u.damping),
				Width: uint8(u.blockWidth), Height: uint8(u.blockHeight),
			}
			filterBlockU8PureGo(want, 8, 0, input, cdefBlockOrigin(), params)
			if err := filterUnitBlocksU8NEON(got, 8, input, cdefBlockOrigin(), blocks, &directions, &variances, u); err != nil {
				t.Fatal(err)
			}
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("boundary=%d pri=%d sec=%d byte=%d got=%d want=%d", boundary, strengths[0], strengths[1], i, got[i], want[i])
				}
			}
		}
	}
}

func TestCDEFPrimarySecondaryU8SIMDZeroAlloc(t *testing.T) {
	input := makeCDEFBlockInput(newCDEFRandom(cdefDeterministicSeed), 8, 0, 0)
	dst := make([]byte, 64)
	params := BlockFilterParams{SecondaryStrength: 4, Direction: 3, PrimaryDamping: 5, SecondaryDamping: 5, Width: 8, Height: 8}
	if allocs := testing.AllocsPerRun(50, func() { filterBlockU8Impl(dst, 8, 0, input, cdefBlockOrigin(), params) }); allocs != 0 {
		t.Fatalf("selected secondary kernel allocated %.1f objects/run", allocs)
	}
	blocks := []BlockPosition{{BY: 0, BX: 0}}
	var directions DirectionGrid
	var variances VarianceGrid
	u := unitFilterParams{secondaryStrength: 4, damping: 5, bwLog2: 3, bhLog2: 3, blockWidth: 8, blockHeight: 8}
	if allocs := testing.AllocsPerRun(50, func() {
		if err := filterUnitBlocksU8NEON(dst, 8, input, cdefBlockOrigin(), blocks, &directions, &variances, u); err != nil {
			t.Fatal(err)
		}
	}); allocs != 0 {
		t.Fatalf("selected unit path allocated %.1f objects/run", allocs)
	}
}

var cdefU8SIMDBenchmarkSink byte

func benchCDEFU8SIMD(b *testing.B, width, height int, primary, secondary int) {
	input := makeCDEFBlockInput(newCDEFRandom(cdefDeterministicSeed), 8, 0, 0)
	dst := make([]byte, 8*8)
	params := BlockFilterParams{
		PrimaryStrength: uint8(primary), SecondaryStrength: uint8(secondary), Direction: 4,
		PrimaryDamping: 5, SecondaryDamping: 5, Width: uint8(width), Height: uint8(height),
	}
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		filterBlockU8Impl(dst, 8, 0, input, cdefBlockOrigin(), params)
	}
	cdefU8SIMDBenchmarkSink = dst[0]
}

func BenchmarkCDEFSecondaryU8_8x8_SIMD(b *testing.B) { benchCDEFU8SIMD(b, 8, 8, 0, 4) }
func BenchmarkCDEFSecondaryU8_4x8_SIMD(b *testing.B) { benchCDEFU8SIMD(b, 4, 8, 0, 4) }
func BenchmarkCDEFSecondaryU8_4x4_SIMD(b *testing.B) { benchCDEFU8SIMD(b, 4, 4, 0, 4) }
func BenchmarkCDEFPrimaryU8_8x8_SIMD(b *testing.B)   { benchCDEFU8SIMD(b, 8, 8, 15, 0) }
func BenchmarkCDEFPrimaryU8_4x8_SIMD(b *testing.B)   { benchCDEFU8SIMD(b, 4, 8, 15, 0) }
