// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build !(goexperiment.simd && arm64 && !purego)

package prediction

import "github.com/thesyncim/goav1/internal/av1/frame"

// fillWide8, verticalWide8 and horizontalWide8 fill 16, 32 or 64 wide rows:
// one row is built and copied down, which memmove does with wide stores.

func fillWide8(block planeBlock, value byte) {
	first := block.pix[:block.width:block.width]
	frame.FillBytes(first, value)
	copyRowDown8(block, first)
}

func verticalWide8(block planeBlock, top []byte) {
	first := block.pix[:block.width:block.width]
	copy(first, top)
	copyRowDown8(block, first)
}

func horizontalWide8(block planeBlock, left []byte) {
	for row, v := range left {
		frame.FillBytes(block.pix[row*block.stride:][:block.width], v)
	}
}

// copyRowDown8 copies the block's first row into its other rows.
func copyRowDown8(block planeBlock, first []byte) {
	for row := 1; row < block.height; row++ {
		copy(block.pix[row*block.stride:][:block.width], first)
	}
}
