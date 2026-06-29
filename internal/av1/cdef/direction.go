// Ported from libaom: av1/common/cdef_block.c
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

package cdef

import "errors"

var ErrInvalidCDEF = errors.New("cdef: invalid input")

var cdefDirectionDivTable = [...]int32{0, 840, 420, 280, 210, 168, 140, 120, 105}

type findDirectionFunc func([]uint16, int, int) (int, int32)
type findDirectionDualFunc func([]uint16, []uint16, int, int) (int, int32, int, int32)

var (
	findDirectionImpl     findDirectionFunc     = findDirectionScalar
	findDirectionDualImpl findDirectionDualFunc = findDirectionDualScalar
)

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
	return findDirectionImpl(img, stride, coeffShift)
}

func findDirectionScalar(img []uint16, stride int, coeffShift int) (int, int32) {
	if coeffShift == 0 {
		return findDirection8Unchecked(img, stride)
	}
	var partial [8][15]int32
	shift := uint(coeffShift)
	for i := range 8 {
		rowBase := i * stride
		iHalf := i >> 1
		for j := range 8 {
			x := int32(img[rowBase+j]>>shift) - 128
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
	return finishDirection(&partial)
}

func findDirection8Unchecked(img []uint16, stride int) (int, int32) {
	var partial [8][15]int32
	for i := range 8 {
		rowBase := i * stride
		iHalf := i >> 1
		for j := range 8 {
			x := int32(img[rowBase+j]) - 128
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
	return finishDirection(&partial)
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
	return findDirectionDualImpl(img1, img2, stride, coeffShift)
}

func findDirectionDualScalar(img1 []uint16, img2 []uint16, stride int, coeffShift int) (int, int32, int, int32) {
	if coeffShift == 0 {
		return findDirectionDual8Unchecked(img1, img2, stride)
	}
	var partial1 [8][15]int32
	var partial2 [8][15]int32
	shift := uint(coeffShift)
	for i := range 8 {
		rowBase := i * stride
		iHalf := i >> 1
		row1 := img1[rowBase : rowBase+8]
		row2 := img2[rowBase : rowBase+8]

		x1 := int32(row1[0]>>shift) - 128
		x2 := int32(row2[0]>>shift) - 128
		partial1[0][i] += x1
		partial1[1][i] += x1
		partial1[2][i] += x1
		partial1[3][i+3] += x1
		partial1[4][i+7] += x1
		partial1[5][3-iHalf] += x1
		partial1[6][0] += x1
		partial1[7][iHalf] += x1
		partial2[0][i] += x2
		partial2[1][i] += x2
		partial2[2][i] += x2
		partial2[3][i+3] += x2
		partial2[4][i+7] += x2
		partial2[5][3-iHalf] += x2
		partial2[6][0] += x2
		partial2[7][iHalf] += x2

		x1 = int32(row1[1]>>shift) - 128
		x2 = int32(row2[1]>>shift) - 128
		partial1[0][i+1] += x1
		partial1[1][i] += x1
		partial1[2][i] += x1
		partial1[3][i+3] += x1
		partial1[4][i+6] += x1
		partial1[5][4-iHalf] += x1
		partial1[6][1] += x1
		partial1[7][iHalf+1] += x1
		partial2[0][i+1] += x2
		partial2[1][i] += x2
		partial2[2][i] += x2
		partial2[3][i+3] += x2
		partial2[4][i+6] += x2
		partial2[5][4-iHalf] += x2
		partial2[6][1] += x2
		partial2[7][iHalf+1] += x2

		x1 = int32(row1[2]>>shift) - 128
		x2 = int32(row2[2]>>shift) - 128
		partial1[0][i+2] += x1
		partial1[1][i+1] += x1
		partial1[2][i] += x1
		partial1[3][i+2] += x1
		partial1[4][i+5] += x1
		partial1[5][5-iHalf] += x1
		partial1[6][2] += x1
		partial1[7][iHalf+2] += x1
		partial2[0][i+2] += x2
		partial2[1][i+1] += x2
		partial2[2][i] += x2
		partial2[3][i+2] += x2
		partial2[4][i+5] += x2
		partial2[5][5-iHalf] += x2
		partial2[6][2] += x2
		partial2[7][iHalf+2] += x2

		x1 = int32(row1[3]>>shift) - 128
		x2 = int32(row2[3]>>shift) - 128
		partial1[0][i+3] += x1
		partial1[1][i+1] += x1
		partial1[2][i] += x1
		partial1[3][i+2] += x1
		partial1[4][i+4] += x1
		partial1[5][6-iHalf] += x1
		partial1[6][3] += x1
		partial1[7][iHalf+3] += x1
		partial2[0][i+3] += x2
		partial2[1][i+1] += x2
		partial2[2][i] += x2
		partial2[3][i+2] += x2
		partial2[4][i+4] += x2
		partial2[5][6-iHalf] += x2
		partial2[6][3] += x2
		partial2[7][iHalf+3] += x2

		x1 = int32(row1[4]>>shift) - 128
		x2 = int32(row2[4]>>shift) - 128
		partial1[0][i+4] += x1
		partial1[1][i+2] += x1
		partial1[2][i] += x1
		partial1[3][i+1] += x1
		partial1[4][i+3] += x1
		partial1[5][7-iHalf] += x1
		partial1[6][4] += x1
		partial1[7][iHalf+4] += x1
		partial2[0][i+4] += x2
		partial2[1][i+2] += x2
		partial2[2][i] += x2
		partial2[3][i+1] += x2
		partial2[4][i+3] += x2
		partial2[5][7-iHalf] += x2
		partial2[6][4] += x2
		partial2[7][iHalf+4] += x2

		x1 = int32(row1[5]>>shift) - 128
		x2 = int32(row2[5]>>shift) - 128
		partial1[0][i+5] += x1
		partial1[1][i+2] += x1
		partial1[2][i] += x1
		partial1[3][i+1] += x1
		partial1[4][i+2] += x1
		partial1[5][8-iHalf] += x1
		partial1[6][5] += x1
		partial1[7][iHalf+5] += x1
		partial2[0][i+5] += x2
		partial2[1][i+2] += x2
		partial2[2][i] += x2
		partial2[3][i+1] += x2
		partial2[4][i+2] += x2
		partial2[5][8-iHalf] += x2
		partial2[6][5] += x2
		partial2[7][iHalf+5] += x2

		x1 = int32(row1[6]>>shift) - 128
		x2 = int32(row2[6]>>shift) - 128
		partial1[0][i+6] += x1
		partial1[1][i+3] += x1
		partial1[2][i] += x1
		partial1[3][i] += x1
		partial1[4][i+1] += x1
		partial1[5][9-iHalf] += x1
		partial1[6][6] += x1
		partial1[7][iHalf+6] += x1
		partial2[0][i+6] += x2
		partial2[1][i+3] += x2
		partial2[2][i] += x2
		partial2[3][i] += x2
		partial2[4][i+1] += x2
		partial2[5][9-iHalf] += x2
		partial2[6][6] += x2
		partial2[7][iHalf+6] += x2

		x1 = int32(row1[7]>>shift) - 128
		x2 = int32(row2[7]>>shift) - 128
		partial1[0][i+7] += x1
		partial1[1][i+3] += x1
		partial1[2][i] += x1
		partial1[3][i] += x1
		partial1[4][i] += x1
		partial1[5][10-iHalf] += x1
		partial1[6][7] += x1
		partial1[7][iHalf+7] += x1
		partial2[0][i+7] += x2
		partial2[1][i+3] += x2
		partial2[2][i] += x2
		partial2[3][i] += x2
		partial2[4][i] += x2
		partial2[5][10-iHalf] += x2
		partial2[6][7] += x2
		partial2[7][iHalf+7] += x2
	}
	dir1, var1 := finishDirection(&partial1)
	dir2, var2 := finishDirection(&partial2)
	return dir1, var1, dir2, var2
}

func findDirectionDual8Unchecked(img1 []uint16, img2 []uint16, stride int) (int, int32, int, int32) {
	var partial1 [8][15]int32
	var partial2 [8][15]int32
	for i := range 8 {
		rowBase := i * stride
		iHalf := i >> 1
		row1 := img1[rowBase : rowBase+8]
		row2 := img2[rowBase : rowBase+8]

		x1 := int32(row1[0]) - 128
		x2 := int32(row2[0]) - 128
		partial1[0][i] += x1
		partial1[1][i] += x1
		partial1[2][i] += x1
		partial1[3][i+3] += x1
		partial1[4][i+7] += x1
		partial1[5][3-iHalf] += x1
		partial1[6][0] += x1
		partial1[7][iHalf] += x1
		partial2[0][i] += x2
		partial2[1][i] += x2
		partial2[2][i] += x2
		partial2[3][i+3] += x2
		partial2[4][i+7] += x2
		partial2[5][3-iHalf] += x2
		partial2[6][0] += x2
		partial2[7][iHalf] += x2

		x1 = int32(row1[1]) - 128
		x2 = int32(row2[1]) - 128
		partial1[0][i+1] += x1
		partial1[1][i] += x1
		partial1[2][i] += x1
		partial1[3][i+3] += x1
		partial1[4][i+6] += x1
		partial1[5][4-iHalf] += x1
		partial1[6][1] += x1
		partial1[7][iHalf+1] += x1
		partial2[0][i+1] += x2
		partial2[1][i] += x2
		partial2[2][i] += x2
		partial2[3][i+3] += x2
		partial2[4][i+6] += x2
		partial2[5][4-iHalf] += x2
		partial2[6][1] += x2
		partial2[7][iHalf+1] += x2

		x1 = int32(row1[2]) - 128
		x2 = int32(row2[2]) - 128
		partial1[0][i+2] += x1
		partial1[1][i+1] += x1
		partial1[2][i] += x1
		partial1[3][i+2] += x1
		partial1[4][i+5] += x1
		partial1[5][5-iHalf] += x1
		partial1[6][2] += x1
		partial1[7][iHalf+2] += x1
		partial2[0][i+2] += x2
		partial2[1][i+1] += x2
		partial2[2][i] += x2
		partial2[3][i+2] += x2
		partial2[4][i+5] += x2
		partial2[5][5-iHalf] += x2
		partial2[6][2] += x2
		partial2[7][iHalf+2] += x2

		x1 = int32(row1[3]) - 128
		x2 = int32(row2[3]) - 128
		partial1[0][i+3] += x1
		partial1[1][i+1] += x1
		partial1[2][i] += x1
		partial1[3][i+2] += x1
		partial1[4][i+4] += x1
		partial1[5][6-iHalf] += x1
		partial1[6][3] += x1
		partial1[7][iHalf+3] += x1
		partial2[0][i+3] += x2
		partial2[1][i+1] += x2
		partial2[2][i] += x2
		partial2[3][i+2] += x2
		partial2[4][i+4] += x2
		partial2[5][6-iHalf] += x2
		partial2[6][3] += x2
		partial2[7][iHalf+3] += x2

		x1 = int32(row1[4]) - 128
		x2 = int32(row2[4]) - 128
		partial1[0][i+4] += x1
		partial1[1][i+2] += x1
		partial1[2][i] += x1
		partial1[3][i+1] += x1
		partial1[4][i+3] += x1
		partial1[5][7-iHalf] += x1
		partial1[6][4] += x1
		partial1[7][iHalf+4] += x1
		partial2[0][i+4] += x2
		partial2[1][i+2] += x2
		partial2[2][i] += x2
		partial2[3][i+1] += x2
		partial2[4][i+3] += x2
		partial2[5][7-iHalf] += x2
		partial2[6][4] += x2
		partial2[7][iHalf+4] += x2

		x1 = int32(row1[5]) - 128
		x2 = int32(row2[5]) - 128
		partial1[0][i+5] += x1
		partial1[1][i+2] += x1
		partial1[2][i] += x1
		partial1[3][i+1] += x1
		partial1[4][i+2] += x1
		partial1[5][8-iHalf] += x1
		partial1[6][5] += x1
		partial1[7][iHalf+5] += x1
		partial2[0][i+5] += x2
		partial2[1][i+2] += x2
		partial2[2][i] += x2
		partial2[3][i+1] += x2
		partial2[4][i+2] += x2
		partial2[5][8-iHalf] += x2
		partial2[6][5] += x2
		partial2[7][iHalf+5] += x2

		x1 = int32(row1[6]) - 128
		x2 = int32(row2[6]) - 128
		partial1[0][i+6] += x1
		partial1[1][i+3] += x1
		partial1[2][i] += x1
		partial1[3][i] += x1
		partial1[4][i+1] += x1
		partial1[5][9-iHalf] += x1
		partial1[6][6] += x1
		partial1[7][iHalf+6] += x1
		partial2[0][i+6] += x2
		partial2[1][i+3] += x2
		partial2[2][i] += x2
		partial2[3][i] += x2
		partial2[4][i+1] += x2
		partial2[5][9-iHalf] += x2
		partial2[6][6] += x2
		partial2[7][iHalf+6] += x2

		x1 = int32(row1[7]) - 128
		x2 = int32(row2[7]) - 128
		partial1[0][i+7] += x1
		partial1[1][i+3] += x1
		partial1[2][i] += x1
		partial1[3][i] += x1
		partial1[4][i] += x1
		partial1[5][10-iHalf] += x1
		partial1[6][7] += x1
		partial1[7][iHalf+7] += x1
		partial2[0][i+7] += x2
		partial2[1][i+3] += x2
		partial2[2][i] += x2
		partial2[3][i] += x2
		partial2[4][i] += x2
		partial2[5][10-iHalf] += x2
		partial2[6][7] += x2
		partial2[7][iHalf+7] += x2
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
