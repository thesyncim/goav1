// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package cdef

import (
	"math/rand"
	"testing"
)

func TestFilterUnitBlocksSIMDMatchesPureGo(t *testing.T) {
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
		if err := filterUnitBlocksSIMD(got, dstStride, input, inputOrigin, blocks, &directions, &variances, u, true); err != nil {
			t.Fatal(err)
		}
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("trial=%d u=%+v sample=%d: SIMD=%d reference=%d", trial, u, i, got[i], want[i])
			}
		}
	}
}

func TestFilterUnitBlocksSIMDZeroAlloc(t *testing.T) {
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
		if err := filterUnitBlocksSIMD(dst, BStride, input, inputOrigin, blocks, &directions, &variances, u, true); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("filterUnitBlocksSIMD allocates: %v allocs/run", allocs)
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

func BenchmarkFilterUnitBlocksSIMD_64Blocks(b *testing.B) {
	benchFilterUnitBlocks(b, filterUnitBlocksSIMD)
}

func BenchmarkFilterUnitBlocksPerBlockCtx_64Blocks(b *testing.B) {
	benchFilterUnitBlocks(b, filterUnitBlocksPureGo)
}
