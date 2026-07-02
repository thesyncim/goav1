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
						filterBlockNEON(got, 16, 0, input, origin, params)
						for i := range want {
							if want[i] != got[i] {
								t.Fatalf("%s %dx%d coeffShift=%d direction=%d dst[%d] neon %d want %d", tc.name, shape.width, shape.height, coeffShift, direction, i, got[i], want[i])
							}
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

func BenchmarkFilterBlock8x8SecondaryOnlySplitNEON(b *testing.B) {
	benchFilterBlock8Asm(b, BlockFilterParams{
		PrimaryStrength: 0, SecondaryStrength: 4, Direction: 5,
		PrimaryDamping: 5, SecondaryDamping: 4, Width: 8, Height: 8,
	}, cdefFilterBlock8SecondaryNEON)
}

func BenchmarkFilterBlock4x4PrimaryOnlyDispatchNEON(b *testing.B) {
	benchFilterNEONParams(b, BlockFilterParams{
		PrimaryStrength: 13, SecondaryStrength: 0, Direction: 5,
		PrimaryDamping: 5, SecondaryDamping: 4, Width: 4, Height: 4,
	}, filterBlockNEON)
}

func BenchmarkFilterBlock4x4PrimaryOnlyGenericNEON(b *testing.B) {
	benchFilterBlock4Asm(b, BlockFilterParams{
		PrimaryStrength: 13, SecondaryStrength: 0, Direction: 5,
		PrimaryDamping: 5, SecondaryDamping: 4, Width: 4, Height: 4,
	}, cdefFilterBlock4NEON)
}

func BenchmarkFilterBlock4x4PrimaryOnlySplitNEON(b *testing.B) {
	benchFilterBlock4Asm(b, BlockFilterParams{
		PrimaryStrength: 13, SecondaryStrength: 0, Direction: 5,
		PrimaryDamping: 5, SecondaryDamping: 4, Width: 4, Height: 4,
	}, cdefFilterBlock4PrimaryNEON)
}

func BenchmarkFilterBlock4x4SecondaryOnlyDispatchNEON(b *testing.B) {
	benchFilterNEONParams(b, BlockFilterParams{
		PrimaryStrength: 0, SecondaryStrength: 4, Direction: 5,
		PrimaryDamping: 5, SecondaryDamping: 4, Width: 4, Height: 4,
	}, filterBlockNEON)
}

func BenchmarkFilterBlock4x4SecondaryOnlyGenericNEON(b *testing.B) {
	benchFilterBlock4Asm(b, BlockFilterParams{
		PrimaryStrength: 0, SecondaryStrength: 4, Direction: 5,
		PrimaryDamping: 5, SecondaryDamping: 4, Width: 4, Height: 4,
	}, cdefFilterBlock4NEON)
}

func BenchmarkFilterBlock4x4SecondaryOnlySplitNEON(b *testing.B) {
	benchFilterBlock4Asm(b, BlockFilterParams{
		PrimaryStrength: 0, SecondaryStrength: 4, Direction: 5,
		PrimaryDamping: 5, SecondaryDamping: 4, Width: 4, Height: 4,
	}, cdefFilterBlock4SecondaryNEON)
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

func benchFilterBlock4Asm(b *testing.B, params BlockFilterParams, fn func(*filterBlockNEONCtx)) {
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

// filterUnitBlocksReference is the trusted-path oracle for the unit-level
// loop: the same strength/direction derivation as filterUnitBlocksPureGo but
// routed straight to filterBlockPureGo, bypassing every dispatch slot.
func filterUnitBlocksReference(dst []uint16, dstStride int, input []uint16, inputOrigin int, blocks []BlockPosition, directions *DirectionGrid, variances *VarianceGrid, u unitFilterParams) {
	for _, block := range blocks {
		by := int(block.BY)
		bx := int(block.BX)
		strength := u.primaryStrength
		if u.lumaAdjust {
			strength = adjustStrength(u.primaryStrength, variances[by][bx])
		}
		dir := 0
		if u.primaryStrength != 0 {
			dir = int(directions[by][bx])
		}
		srcOrigin := inputOrigin + ((by * BStride) << u.bhLog2) + (bx << u.bwLog2)
		dstOrigin := (by<<u.bhLog2)*dstStride + (bx << u.bwLog2)
		filterBlockPureGo(dst, dstStride, dstOrigin, input, srcOrigin, BlockFilterParams{
			PrimaryStrength:   uint8(strength),
			SecondaryStrength: uint8(u.secondaryStrength),
			Direction:         uint8(dir),
			PrimaryDamping:    uint8(u.damping),
			SecondaryDamping:  uint8(u.damping),
			CoeffShift:        uint8(u.coeffShift),
			Width:             uint8(u.blockWidth),
			Height:            uint8(u.blockHeight),
		})
	}
}

func TestFilterUnitBlocksNEONMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0x11cdef))
	input := make([]uint16, InputBufferSize)
	inputOrigin := VerticalBorder*BStride + HorizontalBorder
	const dstStride = BStride
	for trial := 0; trial < 600; trial++ {
		coeffShift := rng.Intn(3)
		maxSample := (1 << (8 + coeffShift)) - 1
		for i := range input {
			if rng.Intn(13) == 0 {
				input[i] = VeryLarge
			} else {
				input[i] = uint16(rng.Intn(maxSample + 1))
			}
		}
		xDec := rng.Intn(2)
		yDec := rng.Intn(2)
		lumaAdjust := rng.Intn(2) == 0
		if lumaAdjust {
			xDec, yDec = 0, 0
		}
		bwLog2 := 3 - xDec
		bhLog2 := 3 - yDec
		u := unitFilterParams{
			primaryStrength:   rng.Intn(16) << coeffShift,
			secondaryStrength: []int{0, 1, 2, 4}[rng.Intn(4)] << coeffShift,
			damping:           3 + coeffShift + rng.Intn(3),
			coeffShift:        coeffShift,
			bwLog2:            bwLog2,
			bhLog2:            bhLog2,
			blockWidth:        1 << bwLog2,
			blockHeight:       1 << bhLog2,
			lumaAdjust:        lumaAdjust,
		}
		var directions DirectionGrid
		var variances VarianceGrid
		var blocks []BlockPosition
		for by := 0; by < 8; by++ {
			for bx := 0; bx < 8; bx++ {
				directions[by][bx] = uint8(rng.Intn(8))
				variances[by][bx] = int32(rng.Intn(1 << 20))
				if rng.Intn(4) == 0 {
					variances[by][bx] = 0
				}
				if rng.Intn(3) != 0 {
					blocks = append(blocks, BlockPosition{BY: uint8(by), BX: uint8(bx)})
				}
			}
		}
		if len(blocks) == 0 {
			blocks = append(blocks, BlockPosition{})
		}
		want := make([]uint16, dstStride*BlockSize)
		got := make([]uint16, dstStride*BlockSize)
		for i := range want {
			want[i] = 0xdead
			got[i] = 0xdead
		}
		filterUnitBlocksReference(want, dstStride, input, inputOrigin, blocks, &directions, &variances, u)
		if err := filterUnitBlocksNEON(got, dstStride, input, inputOrigin, blocks, &directions, &variances, u, true); err != nil {
			t.Fatal(err)
		}
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("trial=%d u=%+v sample=%d: NEON=%d reference=%d", trial, u, i, got[i], want[i])
			}
		}
	}
}

func TestFilterUnitBlocksNEONZeroAlloc(t *testing.T) {
	input := make([]uint16, InputBufferSize)
	for i := range input {
		input[i] = uint16(i * 29 & 1023)
	}
	inputOrigin := VerticalBorder*BStride + HorizontalBorder
	var directions DirectionGrid
	var variances VarianceGrid
	var blocks []BlockPosition
	for by := 0; by < 8; by++ {
		for bx := 0; bx < 8; bx++ {
			directions[by][bx] = uint8((by + bx) & 7)
			variances[by][bx] = int32(bx * by << 10)
			blocks = append(blocks, BlockPosition{BY: uint8(by), BX: uint8(bx)})
		}
	}
	dst := make([]uint16, BStride*BlockSize)
	u := unitFilterParams{
		primaryStrength:   5,
		secondaryStrength: 2,
		damping:           4,
		bwLog2:            3,
		bhLog2:            3,
		blockWidth:        8,
		blockHeight:       8,
		lumaAdjust:        true,
	}
	allocs := testing.AllocsPerRun(50, func() {
		if err := filterUnitBlocksNEON(dst, BStride, input, inputOrigin, blocks, &directions, &variances, u, true); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("filterUnitBlocksNEON allocates: %v allocs/run", allocs)
	}
}

func benchFilterUnitBlocks(b *testing.B, fn func(dst []uint16, dstStride int, input []uint16, inputOrigin int, blocks []BlockPosition, directions *DirectionGrid, variances *VarianceGrid, u unitFilterParams, trusted bool) error) {
	input := make([]uint16, InputBufferSize)
	for i := range input {
		input[i] = uint16(i * 29 & 1023)
	}
	inputOrigin := VerticalBorder*BStride + HorizontalBorder
	var directions DirectionGrid
	var variances VarianceGrid
	var blocks []BlockPosition
	for by := 0; by < 8; by++ {
		for bx := 0; bx < 8; bx++ {
			directions[by][bx] = uint8((by + bx) & 7)
			variances[by][bx] = int32(bx * by << 10)
			blocks = append(blocks, BlockPosition{BY: uint8(by), BX: uint8(bx)})
		}
	}
	dst := make([]uint16, BStride*BlockSize)
	u := unitFilterParams{
		primaryStrength:   5,
		secondaryStrength: 2,
		damping:           4,
		bwLog2:            3,
		bhLog2:            3,
		blockWidth:        8,
		blockHeight:       8,
		lumaAdjust:        true,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := fn(dst, BStride, input, inputOrigin, blocks, &directions, &variances, u, true); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkFilterUnitBlocksNEON_64Blocks(b *testing.B) {
	benchFilterUnitBlocks(b, filterUnitBlocksNEON)
}

func BenchmarkFilterUnitBlocksPerBlockCtx_64Blocks(b *testing.B) {
	benchFilterUnitBlocks(b, filterUnitBlocksPureGo)
}
