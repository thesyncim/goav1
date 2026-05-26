// Ported from libaom: av1/common/cdef_block.c
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

package cdef

import "errors"

var ErrInvalidCDEF = errors.New("cdef: invalid input")

// FindDirection ports libaom's cdef_find_dir_c. img must contain an 8x8 block
// addressed with stride. coeffShift is bitDepth-8 for 8/10/12-bit input.
func FindDirection(img []uint16, stride int, coeffShift int) (int, int32, error) {
	if stride < 8 || coeffShift < 0 || coeffShift > 4 || !blockFits(len(img), stride, 8, 8) {
		return 0, 0, ErrInvalidCDEF
	}
	var cost [8]int32
	var partial [8][15]int
	divTable := [...]int{0, 840, 420, 280, 210, 168, 140, 120, 105}
	for i := range 8 {
		for j := range 8 {
			x := int(img[i*stride+j]>>coeffShift) - 128
			partial[0][i+j] += x
			partial[1][i+j/2] += x
			partial[2][i] += x
			partial[3][3+i-j/2] += x
			partial[4][7+i-j] += x
			partial[5][3-i/2+j] += x
			partial[6][j] += x
			partial[7][i/2+j] += x
		}
	}
	for i := range 8 {
		cost[2] += int32(partial[2][i] * partial[2][i])
		cost[6] += int32(partial[6][i] * partial[6][i])
	}
	cost[2] *= int32(divTable[8])
	cost[6] *= int32(divTable[8])
	for i := range 7 {
		cost[0] += int32((partial[0][i]*partial[0][i] + partial[0][14-i]*partial[0][14-i]) * divTable[i+1])
		cost[4] += int32((partial[4][i]*partial[4][i] + partial[4][14-i]*partial[4][14-i]) * divTable[i+1])
	}
	cost[0] += int32(partial[0][7] * partial[0][7] * divTable[8])
	cost[4] += int32(partial[4][7] * partial[4][7] * divTable[8])
	for i := 1; i < 8; i += 2 {
		for j := range 5 {
			cost[i] += int32(partial[i][3+j] * partial[i][3+j])
		}
		cost[i] *= int32(divTable[8])
		for j := range 3 {
			cost[i] += int32((partial[i][j]*partial[i][j] + partial[i][10-j]*partial[i][10-j]) * divTable[2*j+2])
		}
	}
	bestCost := int32(0)
	bestDir := 0
	for i := range 8 {
		if cost[i] > bestCost {
			bestCost = cost[i]
			bestDir = i
		}
	}
	variance := (bestCost - cost[(bestDir+4)&7]) >> 10
	return bestDir, variance, nil
}

// FindDirectionDual runs FindDirection for two adjacent 8x8 blocks.
func FindDirectionDual(img1 []uint16, img2 []uint16, stride int, coeffShift int) (int, int32, int, int32, error) {
	dir1, var1, err := FindDirection(img1, stride, coeffShift)
	if err != nil {
		return 0, 0, 0, 0, err
	}
	dir2, var2, err := FindDirection(img2, stride, coeffShift)
	if err != nil {
		return 0, 0, 0, 0, err
	}
	return dir1, var1, dir2, var2, nil
}

func blockFits(length int, stride int, width int, height int) bool {
	if stride <= 0 || width <= 0 || height <= 0 || stride < width {
		return false
	}
	lastRow, ok := checkedMul(height-1, stride)
	if !ok {
		return false
	}
	needed, ok := checkedAdd(lastRow, width)
	return ok && needed <= length
}

func checkedAdd(a int, b int) (int, bool) {
	c := a + b
	if c < a {
		return 0, false
	}
	return c, true
}

func checkedMul(a int, b int) (int, bool) {
	if a < 0 || b < 0 {
		return 0, false
	}
	if a != 0 && b > int(^uint(0)>>1)/a {
		return 0, false
	}
	return a * b, true
}
