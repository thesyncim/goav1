// Ported from libaom: av1/common/cdef_block.c
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

package cdef

import "errors"

var ErrInvalidCDEF = errors.New("cdef: invalid input")

var cdefDirectionDivTable = [...]int32{0, 840, 420, 280, 210, 168, 140, 120, 105}

// FindDirection ports libaom's cdef_find_dir_c. img must contain an 8x8 block
// addressed with stride. coeffShift is bitDepth-8 for 8/10/12-bit input.
func FindDirection(img []uint16, stride int, coeffShift int) (int, int32, error) {
	if stride < 8 || coeffShift < 0 || coeffShift > 4 || !blockFits(len(img), stride, 8, 8) {
		return 0, 0, ErrInvalidCDEF
	}
	dir, variance := findDirectionUnchecked(img, stride, coeffShift)
	return dir, variance, nil
}

func findDirectionUnchecked(img []uint16, stride int, coeffShift int) (int, int32) {
	var cost [8]int32
	var partial [8][15]int32
	for i := range 8 {
		rowBase := i * stride
		iHalf := i >> 1
		for j := range 8 {
			x := int32(img[rowBase+j]>>uint(coeffShift)) - 128
			partial[0][i+j] += x
			partial[1][i+(j>>1)] += x
			partial[2][i] += x
			partial[3][3+i-(j>>1)] += x
			partial[4][7+i-j] += x
			partial[5][3-iHalf+j] += x
			partial[6][j] += x
			partial[7][iHalf+j] += x
		}
	}
	for i := range 8 {
		cost[2] += partial[2][i] * partial[2][i]
		cost[6] += partial[6][i] * partial[6][i]
	}
	cost[2] *= cdefDirectionDivTable[8]
	cost[6] *= cdefDirectionDivTable[8]
	for i := range 7 {
		cost[0] += (partial[0][i]*partial[0][i] + partial[0][14-i]*partial[0][14-i]) * cdefDirectionDivTable[i+1]
		cost[4] += (partial[4][i]*partial[4][i] + partial[4][14-i]*partial[4][14-i]) * cdefDirectionDivTable[i+1]
	}
	cost[0] += partial[0][7] * partial[0][7] * cdefDirectionDivTable[8]
	cost[4] += partial[4][7] * partial[4][7] * cdefDirectionDivTable[8]
	for i := 1; i < 8; i += 2 {
		for j := range 5 {
			cost[i] += partial[i][3+j] * partial[i][3+j]
		}
		cost[i] *= cdefDirectionDivTable[8]
		for j := range 3 {
			cost[i] += (partial[i][j]*partial[i][j] + partial[i][10-j]*partial[i][10-j]) * cdefDirectionDivTable[2*j+2]
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
	return bestDir, variance
}

// FindDirectionDual runs FindDirection for two adjacent 8x8 blocks.
func FindDirectionDual(img1 []uint16, img2 []uint16, stride int, coeffShift int) (int, int32, int, int32, error) {
	if stride < 8 || coeffShift < 0 || coeffShift > 4 ||
		!blockFits(len(img1), stride, 8, 8) ||
		!blockFits(len(img2), stride, 8, 8) {
		return 0, 0, 0, 0, ErrInvalidCDEF
	}
	dir1, var1, dir2, var2 := findDirectionDualUnchecked(img1, img2, stride, coeffShift)
	return dir1, var1, dir2, var2, nil
}

func findDirectionDualUnchecked(img1 []uint16, img2 []uint16, stride int, coeffShift int) (int, int32, int, int32) {
	var partial1 [8][15]int32
	var partial2 [8][15]int32
	for i := range 8 {
		rowBase := i * stride
		iHalf := i >> 1
		for j := range 8 {
			x1 := int32(img1[rowBase+j]>>uint(coeffShift)) - 128
			x2 := int32(img2[rowBase+j]>>uint(coeffShift)) - 128
			diag0 := i + j
			diag1 := i + (j >> 1)
			diag3 := 3 + i - (j >> 1)
			diag4 := 7 + i - j
			diag5 := 3 - iHalf + j
			diag7 := iHalf + j

			partial1[0][diag0] += x1
			partial1[1][diag1] += x1
			partial1[2][i] += x1
			partial1[3][diag3] += x1
			partial1[4][diag4] += x1
			partial1[5][diag5] += x1
			partial1[6][j] += x1
			partial1[7][diag7] += x1

			partial2[0][diag0] += x2
			partial2[1][diag1] += x2
			partial2[2][i] += x2
			partial2[3][diag3] += x2
			partial2[4][diag4] += x2
			partial2[5][diag5] += x2
			partial2[6][j] += x2
			partial2[7][diag7] += x2
		}
	}
	dir1, var1 := finishDirection(&partial1)
	dir2, var2 := finishDirection(&partial2)
	return dir1, var1, dir2, var2
}

func finishDirection(partial *[8][15]int32) (int, int32) {
	c0 := finishDirectionDiagonalCost(&partial[0])
	c1 := finishDirectionOddCost(&partial[1])
	c2 := finishDirectionStraightCost(&partial[2])
	c3 := finishDirectionOddCost(&partial[3])
	c4 := finishDirectionDiagonalCost(&partial[4])
	c5 := finishDirectionOddCost(&partial[5])
	c6 := finishDirectionStraightCost(&partial[6])
	c7 := finishDirectionOddCost(&partial[7])

	bestDir := 0
	bestCost := c0
	opposite := c4
	if c1 > bestCost {
		bestDir, bestCost, opposite = 1, c1, c5
	}
	if c2 > bestCost {
		bestDir, bestCost, opposite = 2, c2, c6
	}
	if c3 > bestCost {
		bestDir, bestCost, opposite = 3, c3, c7
	}
	if c4 > bestCost {
		bestDir, bestCost, opposite = 4, c4, c0
	}
	if c5 > bestCost {
		bestDir, bestCost, opposite = 5, c5, c1
	}
	if c6 > bestCost {
		bestDir, bestCost, opposite = 6, c6, c2
	}
	if c7 > bestCost {
		bestDir, bestCost, opposite = 7, c7, c3
	}
	return bestDir, (bestCost - opposite) >> 10
}

func finishDirectionStraightCost(partial *[15]int32) int32 {
	sum := partial[0]*partial[0] +
		partial[1]*partial[1] +
		partial[2]*partial[2] +
		partial[3]*partial[3] +
		partial[4]*partial[4] +
		partial[5]*partial[5] +
		partial[6]*partial[6] +
		partial[7]*partial[7]
	return sum * cdefDirectionDivTable[8]
}

func finishDirectionDiagonalCost(partial *[15]int32) int32 {
	return (partial[0]*partial[0]+partial[14]*partial[14])*cdefDirectionDivTable[1] +
		(partial[1]*partial[1]+partial[13]*partial[13])*cdefDirectionDivTable[2] +
		(partial[2]*partial[2]+partial[12]*partial[12])*cdefDirectionDivTable[3] +
		(partial[3]*partial[3]+partial[11]*partial[11])*cdefDirectionDivTable[4] +
		(partial[4]*partial[4]+partial[10]*partial[10])*cdefDirectionDivTable[5] +
		(partial[5]*partial[5]+partial[9]*partial[9])*cdefDirectionDivTable[6] +
		(partial[6]*partial[6]+partial[8]*partial[8])*cdefDirectionDivTable[7] +
		partial[7]*partial[7]*cdefDirectionDivTable[8]
}

func finishDirectionOddCost(partial *[15]int32) int32 {
	return (partial[3]*partial[3]+
		partial[4]*partial[4]+
		partial[5]*partial[5]+
		partial[6]*partial[6]+
		partial[7]*partial[7])*cdefDirectionDivTable[8] +
		(partial[0]*partial[0]+partial[10]*partial[10])*cdefDirectionDivTable[2] +
		(partial[1]*partial[1]+partial[9]*partial[9])*cdefDirectionDivTable[4] +
		(partial[2]*partial[2]+partial[8]*partial[8])*cdefDirectionDivTable[6]
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
