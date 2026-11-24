// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package prediction

import (
	"encoding/binary"

	"github.com/thesyncim/goav1/internal/av1/frame"
)

// 8-bit DC / vertical / horizontal stores. Like dav1d's ipred_dc, ipred_v and
// ipred_h (src/arm/64/ipred.S), the row pattern is built once and each row is
// written with word stores (4 and 8 wide) or the wide stores of
// fillWide8, verticalWide8 and horizontalWide8 (16, 32 and 64 wide) instead of
// a per-row copy call.

const bytes8Splat = 0x0101010101010101

func fillBlock8(block planeBlock, value byte) {
	w, stride := block.width, block.stride
	switch w {
	case 4:
		p := uint32(value) * 0x01010101
		for o, h := 0, block.height; h > 0; h-- {
			binary.LittleEndian.PutUint32(block.pix[o:o+4], p)
			o += stride
		}
	case 8:
		p := uint64(value) * bytes8Splat
		for o, h := 0, block.height; h > 0; h-- {
			binary.LittleEndian.PutUint64(block.pix[o:o+8], p)
			o += stride
		}
	case 16, 32, 64:
		fillWide8(block, value)
	default:
		if block.height == 0 {
			return
		}
		first := block.pix[:w:w]
		frame.FillBytes(first, value)
		for row := 1; row < block.height; row++ {
			copy(block.pix[row*stride:][:w], first)
		}
	}
}

func predictHorizontal8(block planeBlock, left []uint16) {
	w, stride := block.width, block.stride
	left = left[:block.height]
	switch w {
	case 4:
		o := 0
		for _, v := range left {
			binary.LittleEndian.PutUint32(block.pix[o:o+4], uint32(byte(v))*0x01010101)
			o += stride
		}
	case 8:
		o := 0
		for _, v := range left {
			binary.LittleEndian.PutUint64(block.pix[o:o+8], uint64(byte(v))*bytes8Splat)
			o += stride
		}
	default:
		for row, v := range left {
			frame.FillBytes(block.pix[row*stride:][:w], byte(v))
		}
	}
}

func predictVertical8(block planeBlock, above []uint16) {
	w, stride := block.width, block.stride
	above = above[:w]
	switch w {
	case 4, 8:
		var p uint64
		for col, v := range above {
			p |= uint64(byte(v)) << (8 * (col & 7))
		}
		for row := range block.height {
			line := block.pix[row*stride:]
			if w == 4 {
				binary.LittleEndian.PutUint32(line[:4], uint32(p))
			} else {
				binary.LittleEndian.PutUint64(line[:8], p)
			}
		}
	default:
		if block.height == 0 {
			return
		}
		first := block.pix[:w:w]
		for col, v := range above {
			first[col] = byte(v)
		}
		for row := 1; row < block.height; row++ {
			copy(block.pix[row*stride:][:w], first)
		}
	}
}

// predictVertical8Bytes is predictVertical8 with the top row read in place
// from the plane; top holds at least the block width.
func predictVertical8Bytes(block planeBlock, top []byte) {
	w, stride := block.width, block.stride
	switch w {
	case 4:
		p := binary.LittleEndian.Uint32(top)
		for o, h := 0, block.height; h > 0; h-- {
			binary.LittleEndian.PutUint32(block.pix[o:o+4], p)
			o += stride
		}
	case 8:
		p := binary.LittleEndian.Uint64(top)
		for o, h := 0, block.height; h > 0; h-- {
			binary.LittleEndian.PutUint64(block.pix[o:o+8], p)
			o += stride
		}
	default:
		verticalWide8(block, top)
	}
}

// predictHorizontal8Bytes is predictHorizontal8 with the left column as
// pixels.
func predictHorizontal8Bytes(block planeBlock, left []byte) {
	stride := block.stride
	switch block.width {
	case 4:
		o := 0
		for _, v := range left {
			binary.LittleEndian.PutUint32(block.pix[o:o+4], uint32(v)*0x01010101)
			o += stride
		}
	case 8:
		o := 0
		for _, v := range left {
			binary.LittleEndian.PutUint64(block.pix[o:o+8], uint64(v)*bytes8Splat)
			o += stride
		}
	default:
		horizontalWide8(block, left)
	}
}
