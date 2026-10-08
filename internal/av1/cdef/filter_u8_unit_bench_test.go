// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package cdef

import "testing"

// BenchmarkFilterUnitBlocksU8Dispatch runs the 8-bit unit loop over one full
// 64x64 unit of 8x8 blocks through the resolved per-unit dispatch.
func BenchmarkFilterUnitBlocksU8Dispatch(b *testing.B) {
	input := make([]uint16, InputBufferSize)
	for i := range input {
		input[i] = uint16(i * 29 & 255)
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
	dst := make([]byte, BStride*BlockSize)
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
		if err := filterUnitBlocksU8(dst, BStride, input, inputOrigin, blocks, &directions, &variances, u); err != nil {
			b.Fatal(err)
		}
	}
}
