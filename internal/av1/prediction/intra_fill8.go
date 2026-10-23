// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package prediction

import (
	"encoding/binary"

	"github.com/thesyncim/goav1/internal/av1/frame"
)

// 8-bit DC / vertical / horizontal stores. Like dav1d's ipred_dc, ipred_v and
// ipred_h (src/arm/64/ipred.S), 4- and 8-wide blocks build the row pattern
// once and write each row with a single word store instead of a per-row copy
// call; wider blocks fill one row and copy it, which memmove already does with
// wide vector stores.

const bytes8Splat = 0x0101010101010101

func fillBlock8(block planeBlock, value byte) {
	w, stride := block.width, block.stride
	switch w {
	case 4:
		p := uint32(value) * 0x01010101
		for row := range block.height {
			binary.LittleEndian.PutUint32(block.pix[row*stride:][:4], p)
		}
	case 8:
		p := uint64(value) * bytes8Splat
		for row := range block.height {
			binary.LittleEndian.PutUint64(block.pix[row*stride:][:8], p)
		}
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
		for row, v := range left {
			binary.LittleEndian.PutUint32(block.pix[row*stride:][:4], uint32(byte(v))*0x01010101)
		}
	case 8:
		for row, v := range left {
			binary.LittleEndian.PutUint64(block.pix[row*stride:][:8], uint64(byte(v))*bytes8Splat)
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
