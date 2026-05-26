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

	var buffer [33][33]uint16
	for row := 0; row < predHeight; row++ {
		buffer[row+1][0] = edges.Left[row]
	}
	buffer[0][0] = edges.AboveLeft
	copy(buffer[0][1:predWidth+1], edges.Above[:predWidth])

	for row := 1; row < predHeight+1; row += 2 {
		for col := 1; col < predWidth+1; col += 4 {
			p := [7]uint16{
				buffer[row-1][col-1],
				buffer[row-1][col],
				buffer[row-1][col+1],
				buffer[row-1][col+2],
				buffer[row-1][col+3],
				buffer[row][col-1],
				buffer[row+1][col-1],
			}
			for k := 0; k < 8; k++ {
				sum := 0
				for i := 0; i < len(p); i++ {
					sum += int(filterIntraTaps[mode][k][i]) * int(p[i])
				}
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

	for row := 0; row < height; row++ {
		for col := 0; col < width; col++ {
			setBlockSample(block, bytesPerSample, row, col, buffer[row+1][col+1])
		}
	}
	return nil
}
