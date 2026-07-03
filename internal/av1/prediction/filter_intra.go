// Ported from libaom: av1/common/reconintra.c
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

package prediction

import "github.com/thesyncim/goav1/internal/av1/frame"

const filterIntraScaleBits = 4

// FilterIntraMode identifies libaom's FILTER_INTRA_MODE values.
type FilterIntraMode uint8

const (
	FilterIntraModeDC FilterIntraMode = iota
	FilterIntraModeVertical
	FilterIntraModeHorizontal
	FilterIntraModeD157
	FilterIntraModePaeth
	FilterIntraModes
)

var filterIntraTaps = [FilterIntraModes][8][8]int8{
	{
		{-6, 10, 0, 0, 0, 12, 0, 0},
		{-5, 2, 10, 0, 0, 9, 0, 0},
		{-3, 1, 1, 10, 0, 7, 0, 0},
		{-3, 1, 1, 2, 10, 5, 0, 0},
		{-4, 6, 0, 0, 0, 2, 12, 0},
		{-3, 2, 6, 0, 0, 2, 9, 0},
		{-3, 2, 2, 6, 0, 2, 7, 0},
		{-3, 1, 2, 2, 6, 3, 5, 0},
	},
	{
		{-10, 16, 0, 0, 0, 10, 0, 0},
		{-6, 0, 16, 0, 0, 6, 0, 0},
		{-4, 0, 0, 16, 0, 4, 0, 0},
		{-2, 0, 0, 0, 16, 2, 0, 0},
		{-10, 16, 0, 0, 0, 0, 10, 0},
		{-6, 0, 16, 0, 0, 0, 6, 0},
		{-4, 0, 0, 16, 0, 0, 4, 0},
		{-2, 0, 0, 0, 16, 0, 2, 0},
	},
	{
		{-8, 8, 0, 0, 0, 16, 0, 0},
		{-8, 0, 8, 0, 0, 16, 0, 0},
		{-8, 0, 0, 8, 0, 16, 0, 0},
		{-8, 0, 0, 0, 8, 16, 0, 0},
		{-4, 4, 0, 0, 0, 0, 16, 0},
		{-4, 0, 4, 0, 0, 0, 16, 0},
		{-4, 0, 0, 4, 0, 0, 16, 0},
		{-4, 0, 0, 0, 4, 0, 16, 0},
	},
	{
		{-2, 8, 0, 0, 0, 10, 0, 0},
		{-1, 3, 8, 0, 0, 6, 0, 0},
		{-1, 2, 3, 8, 0, 4, 0, 0},
		{0, 1, 2, 3, 8, 2, 0, 0},
		{-1, 4, 0, 0, 0, 3, 10, 0},
		{-1, 3, 4, 0, 0, 4, 6, 0},
		{-1, 2, 3, 4, 0, 4, 4, 0},
		{-1, 2, 2, 3, 4, 3, 3, 0},
	},
	{
		{-12, 14, 0, 0, 0, 14, 0, 0},
		{-10, 0, 14, 0, 0, 12, 0, 0},
		{-9, 0, 0, 14, 0, 11, 0, 0},
		{-8, 0, 0, 0, 14, 10, 0, 0},
		{-10, 12, 0, 0, 0, 0, 14, 0},
		{-9, 1, 12, 0, 0, 0, 12, 0},
		{-8, 0, 0, 12, 0, 1, 11, 0},
		{-7, 0, 0, 1, 12, 1, 9, 0},
	},
}

// PredictFilterIntraPlaneBlock writes libaom-compatible filter-intra
// prediction. The AV1 filter-intra predictor is defined for transform blocks up
// to 32x32 whose width is a multiple of four and height is a multiple of two.
func PredictFilterIntraPlaneBlock(dst frame.Plane, bytesPerSample int, bitDepth uint8, x int, y int, width int, height int, mode FilterIntraMode, edges IntraEdges) error {
	return PredictFilterIntraPlaneBlockWithExtent(dst, bytesPerSample, bitDepth, x, y, width, height, width, height, mode, edges)
}

// PredictFilterIntraPlaneBlockWithExtent writes the visible sub-rectangle of a
// filter-intra predictor whose coded transform extent may continue past the
// frame boundary.
func PredictFilterIntraPlaneBlockWithExtent(dst frame.Plane, bytesPerSample int, bitDepth uint8, x int, y int, width int, height int, predWidth int, predHeight int, mode FilterIntraMode, edges IntraEdges) error {
	block, err := planeBlockWindow(dst, bytesPerSample, x, y, width, height)
	if err != nil {
		return err
	}
	if predWidth < width || predHeight < height ||
		predWidth > 32 || predHeight > 32 ||
		predWidth%4 != 0 || predHeight%2 != 0 ||
		mode >= FilterIntraModes {
		return ErrInvalidPrediction
	}
	max, err := sampleMax(bytesPerSample, bitDepth)
	if err != nil {
		return err
	}
	if err := validateIntraDirectionalEdges(predWidth, predHeight, max, edges, true); err != nil {
		return err
	}
	if width == predWidth && height == predHeight {
		predictFilterIntraBlockDirect(block, bytesPerSample, width, height, mode, edges, max)
		return nil
	}

	var buffer [33][33]uint16
	for row := range predHeight {
		buffer[row+1][0] = edges.Left[row]
	}
	buffer[0][0] = edges.AboveLeft
	copy(buffer[0][1:predWidth+1], edges.Above[:predWidth])

	taps := filterIntraTaps[mode]
	for row := 1; row < predHeight+1; row += 2 {
		for col := 1; col < predWidth+1; col += 4 {
			p0 := int(buffer[row-1][col-1])
			p1 := int(buffer[row-1][col])
			p2 := int(buffer[row-1][col+1])
			p3 := int(buffer[row-1][col+2])
			p4 := int(buffer[row-1][col+3])
			p5 := int(buffer[row][col-1])
			p6 := int(buffer[row+1][col-1])
			for k := range 8 {
				tap := taps[k]
				sum := int(tap[0])*p0 + int(tap[1])*p1 + int(tap[2])*p2 +
					int(tap[3])*p3 + int(tap[4])*p4 + int(tap[5])*p5 +
					int(tap[6])*p6
				sample := roundPowerOfTwo(sum, filterIntraScaleBits)
				if sample < 0 {
					sample = 0
				} else if sample > int(max) {
					sample = int(max)
				}
				buffer[row+(k>>2)][col+(k&3)] = uint16(sample)
			}
		}
	}

	storeFilterIntraBlock(block, bytesPerSample, width, height, &buffer)
	return nil
}

func predictFilterIntraBlockDirect(block planeBlock, bytesPerSample int, width int, height int, mode FilterIntraMode, edges IntraEdges, max uint16) {
	switch bytesPerSample {
	case 1:
		predictFilterIntra8Impl(block, width, height, mode, edges, int(max))
	case 2:
		predictFilterIntraBlockDirect16(block, width, height, mode, edges, int(max))
	}
}

func predictFilterIntraBlockDirect8(block planeBlock, width int, height int, mode FilterIntraMode, edges IntraEdges, max int) {
	taps := filterIntraTaps[mode]
	for row := 0; row < height; row += 2 {
		row0 := block.pix[row*block.stride : row*block.stride+width : row*block.stride+width]
		row1 := block.pix[(row+1)*block.stride : (row+1)*block.stride+width : (row+1)*block.stride+width]
		for col := 0; col < width; col += 4 {
			p0 := int(edges.AboveLeft)
			if col > 0 {
				if row == 0 {
					p0 = int(edges.Above[col-1])
				} else {
					p0 = int(block.pix[(row-1)*block.stride+col-1])
				}
			} else if row > 0 {
				p0 = int(edges.Left[row-1])
			}
			var p1, p2, p3, p4 int
			if row == 0 {
				p1 = int(edges.Above[col+0])
				p2 = int(edges.Above[col+1])
				p3 = int(edges.Above[col+2])
				p4 = int(edges.Above[col+3])
			} else {
				top := block.pix[(row-1)*block.stride+col : (row-1)*block.stride+col+4 : (row-1)*block.stride+col+4]
				p1 = int(top[0])
				p2 = int(top[1])
				p3 = int(top[2])
				p4 = int(top[3])
			}
			p5 := int(edges.Left[row])
			p6 := int(edges.Left[row+1])
			if col > 0 {
				p5 = int(row0[col-1])
				p6 = int(row1[col-1])
			}
			for k := range 8 {
				tap := taps[k]
				sum := int(tap[0])*p0 + int(tap[1])*p1 + int(tap[2])*p2 +
					int(tap[3])*p3 + int(tap[4])*p4 + int(tap[5])*p5 +
					int(tap[6])*p6
				sample := roundPowerOfTwo(sum, filterIntraScaleBits)
				if sample < 0 {
					sample = 0
				} else if sample > max {
					sample = max
				}
				if k < 4 {
					row0[col+k] = byte(sample)
				} else {
					row1[col+k-4] = byte(sample)
				}
			}
		}
	}
}

func predictFilterIntraBlockDirect16(block planeBlock, width int, height int, mode FilterIntraMode, edges IntraEdges, max int) {
	taps := filterIntraTaps[mode]
	rowBytes := width << 1
	for row := 0; row < height; row += 2 {
		row0 := block.pix[row*block.stride : row*block.stride+rowBytes : row*block.stride+rowBytes]
		row1 := block.pix[(row+1)*block.stride : (row+1)*block.stride+rowBytes : (row+1)*block.stride+rowBytes]
		for col := 0; col < width; col += 4 {
			p0 := int(edges.AboveLeft)
			if col > 0 {
				if row == 0 {
					p0 = int(edges.Above[col-1])
				} else {
					p0 = loadFilterIntra16(block.pix[(row-1)*block.stride:], col-1)
				}
			} else if row > 0 {
				p0 = int(edges.Left[row-1])
			}
			var p1, p2, p3, p4 int
			if row == 0 {
				p1 = int(edges.Above[col+0])
				p2 = int(edges.Above[col+1])
				p3 = int(edges.Above[col+2])
				p4 = int(edges.Above[col+3])
			} else {
				top := block.pix[(row-1)*block.stride:]
				p1 = loadFilterIntra16(top, col+0)
				p2 = loadFilterIntra16(top, col+1)
				p3 = loadFilterIntra16(top, col+2)
				p4 = loadFilterIntra16(top, col+3)
			}
			p5 := int(edges.Left[row])
			p6 := int(edges.Left[row+1])
			if col > 0 {
				p5 = loadFilterIntra16(row0, col-1)
				p6 = loadFilterIntra16(row1, col-1)
			}
			for k := range 8 {
				tap := taps[k]
				sum := int(tap[0])*p0 + int(tap[1])*p1 + int(tap[2])*p2 +
					int(tap[3])*p3 + int(tap[4])*p4 + int(tap[5])*p5 +
					int(tap[6])*p6
				sample := roundPowerOfTwo(sum, filterIntraScaleBits)
				if sample < 0 {
					sample = 0
				} else if sample > max {
					sample = max
				}
				if k < 4 {
					storeFilterIntra16(row0, col+k, uint16(sample))
				} else {
					storeFilterIntra16(row1, col+k-4, uint16(sample))
				}
			}
		}
	}
}

func loadFilterIntra16(row []byte, col int) int {
	offset := col << 1
	return int(uint16(row[offset]) | uint16(row[offset+1])<<8)
}

func storeFilterIntra16(row []byte, col int, value uint16) {
	offset := col << 1
	row[offset] = byte(value)
	row[offset+1] = byte(value >> 8)
}

func storeFilterIntraBlock(block planeBlock, bytesPerSample int, width int, height int, buffer *[33][33]uint16) {
	switch bytesPerSample {
	case 1:
		for row := range height {
			dst := block.pix[row*block.stride:]
			src := buffer[row+1][1:]
			for col := range width {
				dst[col] = byte(src[col])
			}
		}
	case 2:
		for row := range height {
			dst := block.pix[row*block.stride:]
			src := buffer[row+1][1:]
			for col := range width {
				value := src[col]
				offset := col << 1
				dst[offset] = byte(value)
				dst[offset+1] = byte(value >> 8)
			}
		}
	}
}
