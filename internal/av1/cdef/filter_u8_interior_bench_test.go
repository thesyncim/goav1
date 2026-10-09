// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package cdef

import "testing"

// interiorBenchInput returns a sentinel-free tap buffer and a fused 8x8 block
// with both strengths enabled, the shape the interior kernel serves.
func interiorBenchInput() ([]uint16, BlockFilterParams) {
	input := make([]uint16, BStride*24)
	for i := range input {
		input[i] = uint16(i * 7 % 256)
	}
	return input, BlockFilterParams{
		PrimaryStrength: 9, SecondaryStrength: 2, Direction: 5,
		PrimaryDamping: 5, SecondaryDamping: 4, Width: 8, Height: 8,
	}
}

func BenchmarkFilterBlockU8_8x8_FusedSIMD(b *testing.B) {
	input, params := interiorBenchInput()
	dst := make([]byte, 24*8)
	b.ReportAllocs()
	for b.Loop() {
		filterBlockU8SIMD(dst, 24, 0, input, cdefBlockOrigin(), params)
	}
}

func BenchmarkFilterBlockU8_8x8_InteriorSIMD(b *testing.B) {
	input, params := interiorBenchInput()
	dst := make([]byte, 24*8)
	b.ReportAllocs()
	for b.Loop() {
		filterBlockU8InteriorSIMD(dst, 24, 0, input, cdefBlockOrigin(), params)
	}
}

func benchmarkCDEFUnitU8Interior() ([]uint16, []BlockPosition, *DirectionGrid, *VarianceGrid, []byte) {
	input := make([]uint16, InputBufferSize)
	for i := range input {
		input[i] = uint16((i*37 + i/17) & 0xFF)
	}
	blocks := make([]BlockPosition, 0, 64)
	for by := range 8 {
		for bx := range 8 {
			blocks = append(blocks, BlockPosition{BY: uint8(by), BX: uint8(bx)})
		}
	}
	dirs := new(DirectionGrid)
	vars := new(VarianceGrid)
	for by := range 8 {
		for bx := range 8 {
			dirs[by][bx] = uint8((by*3 + bx*5) & 7)
			vars[by][bx] = 1 << 12
		}
	}
	return input, blocks, dirs, vars, make([]byte, BlockSize*BlockSize)
}

func BenchmarkCDEFUnitU8InteriorScan(b *testing.B) {
	input, full, dirs, vars, dst := benchmarkCDEFUnitU8Interior()
	sparse := []BlockPosition{{BY: 0, BX: 0}, {BY: 0, BX: 7}, {BY: 7, BX: 0}, {BY: 7, BX: 7}}
	single := []BlockPosition{{BY: 3, BX: 3}}
	cases := []struct {
		name       string
		blocks     []BlockPosition
		primary    int
		secondary  int
		lumaAdjust bool
		zeroVars   bool
	}{
		{name: "full64x64", blocks: full, primary: 15, secondary: 4},
		{name: "sparse64x64", blocks: sparse, primary: 15, secondary: 4},
		{name: "single8x8", blocks: single, primary: 15, secondary: 4},
		{name: "full64x64_primaryOnly", blocks: full, primary: 15},
		{name: "full64x64_secondaryOnly", blocks: full, secondary: 4},
		{name: "flatLumaAdjustedPrimaryZero", blocks: full, primary: 15, secondary: 4, lumaAdjust: true, zeroVars: true},
	}
	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			unitVars := vars
			if tc.zeroVars {
				unitVars = new(VarianceGrid)
			}
			u := unitFilterParams{
				primaryStrength: tc.primary, secondaryStrength: tc.secondary,
				damping: 5, bwLog2: 3, bhLog2: 3, blockWidth: 8, blockHeight: 8,
				lumaAdjust: tc.lumaAdjust,
			}
			b.ReportAllocs()
			for b.Loop() {
				if err := filterUnitBlocksU8(dst, BlockSize, input, cdefBlockOrigin(), tc.blocks, dirs, unitVars, u); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func BenchmarkCDEFUnitU8InteriorScanDirect(b *testing.B) {
	input, blocks, _, _, _ := benchmarkCDEFUnitU8Interior()
	for _, tc := range []struct {
		name   string
		blocks []BlockPosition
	}{
		{name: "full64x64", blocks: blocks},
		{name: "sparse64x64", blocks: []BlockPosition{{BY: 0, BX: 0}, {BY: 0, BX: 7}, {BY: 7, BX: 0}, {BY: 7, BX: 7}}},
		{name: "single8x8", blocks: []BlockPosition{{BY: 3, BX: 3}}},
	} {
		b.Run(tc.name+"/"+cdefUnitInteriorScanFlavor, func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				_ = cdefUnitInteriorU8(input, cdefBlockOrigin(), tc.blocks, 3, 3)
			}
		})
		b.Run(tc.name+"/ScalarOracle", func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				_ = cdefUnitInteriorU8ScalarOracle(input, cdefBlockOrigin(), tc.blocks, 3, 3)
			}
		})
	}
}
