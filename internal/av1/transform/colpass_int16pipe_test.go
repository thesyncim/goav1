// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package transform

import (
	"math/rand"
	"testing"
)

func TestInt16ColumnSIMDInputGuardBounds(t *testing.T) {
	for _, height := range []int{8, 16, 32, 64} {
		limit := int16ColumnSIMDInputBound(height)
		if limit == 0 {
			t.Fatalf("height=%d has no SIMD bound", height)
		}
		full := make([]int16, 8*height)
		full[0], full[1] = int16(limit), int16(-limit)
		if !int16ColumnSIMDInputSafe(full, 8, height, minInt16, maxInt16) {
			t.Fatalf("height=%d rejected certified boundary", height)
		}
		for _, value := range []int32{limit + 1, -limit - 1, minInt16, maxInt16} {
			full[0] = int16(value)
			if int16ColumnSIMDInputSafe(full, 8, height, minInt16, maxInt16) {
				t.Fatalf("height=%d accepted value=%d outside certified interval", height, value)
			}
		}
	}

	input := make([]int16, 8*8)
	input[0], input[4*8] = 30000, 30000
	if int16ColumnSIMDInputSafe(input, 8, 8, minInt16, maxInt16) {
		t.Fatal("accepted DCT8 even-butterfly input that overflows an int16 pre-rotation sum")
	}
	if int16ColumnSIMDInputSafe(input, 8, 8, 1, maxInt16) {
		t.Fatal("accepted a clamp interval that excludes zero")
	}
}

func TestInverseBlockBitDepthFullRangeMatchesScalarColumnPath(t *testing.T) {
	rng := rand.New(rand.NewSource(0x1632dc7))
	for _, height := range []int{8, 16, 32, 64} {
		size := Size{Width: uint8(height), Height: uint8(height)}
		width := height
		for _, pattern := range []string{"random-int16", "alternating-extrema"} {
			coeff := make([]int32, width*height)
			for i := range coeff {
				switch pattern {
				case "random-int16":
					coeff[i] = int32(int16(rng.Uint32()))
				case "alternating-extrema":
					if i%2 == 0 {
						coeff[i] = maxInt16
					} else {
						coeff[i] = minInt16
					}
				}
			}
			compareInverseBlockBitDepthToScalar(t, size, TypeDCTDCT, coeff)
		}
	}

	// TypeVDCT leaves each row as an identity transform. These coefficients
	// produce pre-column values of +16384 at c0 and c4 after the 8x8 mid-pass,
	// crossing the int16 SIMD DCT8 sum boundary while remaining valid int32 input.
	coeff := make([]int32, 8*8)
	coeff[0], coeff[4] = 1<<14, 1<<14
	size := Size{Width: 8, Height: 8}
	if int16ColumnFast {
		rowMin, rowMax, colMin, colMax, _ := stageRangeBounds(8)
		usedSIMD, err := inverseSeparableBlockClampedRowsToScratch(
			coeff, 8, make([]int32, 8*8), size, TypeVDCT,
			rowMin, rowMax, colMin, colMax, 0, make([]int16, 8*8), false,
		)
		if err != nil {
			t.Fatal(err)
		}
		if usedSIMD {
			t.Fatal("SIMD column path accepted pre-column values outside its certified interval")
		}
	}
	compareInverseBlockBitDepthToScalar(t, size, TypeVDCT, coeff)
}

func compareInverseBlockBitDepthToScalar(t *testing.T, size Size, typ Type, coeff []int32) {
	t.Helper()
	width, height := int(size.Width), int(size.Height)
	got := make([]int16, width*height)
	gotScratch := make([]int32, width*height)
	if err := InverseBlockBitDepth(got, width, coeff, height, gotScratch, size, typ, 8); err != nil {
		t.Fatalf("InverseBlockBitDepth size=%+v type=%d: %v", size, typ, err)
	}
	want := inverseBlockBitDepthScalarReference(coeff, height, size, typ)
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("size=%+v type=%d index=%d SIMD-dispatch=%d scalar=%d", size, typ, i, got[i], want[i])
		}
	}
}

// inverseBlockBitDepthScalarReference spells out the two passes with the
// int32 scalar kernels. It deliberately bypasses the int16 column dispatcher
// so it can serve as an independent oracle for its range guard and fallback.
func inverseBlockBitDepthScalarReference(coeff []int32, coeffStride int, size Size, typ Type) []int16 {
	width, height := int(size.Width), int(size.Height)
	rowMin, rowMax, colMin, colMax, _ := stageRangeBounds(8)
	idx := sizeIndex(size)
	coeffSize := adjustedScanSizeTable[idx]
	coeffW, coeffH := int(coeffSize.Width), int(coeffSize.Height)
	shift := int(sizeShiftTable[idx])
	vertical, horizontal, _ := typ.tx1DTypes()
	scratch := make([]int32, width*height)
	stageTransposeClampScalar(scratch, width, coeff, coeffStride, coeffH, coeffW, size.IsRect2(), rowMin, rowMax)
	for row := 0; row < coeffH; row++ {
		line := scratch[row*width : row*width+width]
		switch horizontal {
		case tx1DDCT:
			inverseDCT1D(line, 1, width, rowMin, rowMax)
		case tx1DIdentity:
			inverseIdentity1DRow(line, width)
		}
	}
	for i, value := range scratch {
		if shift > 0 {
			value = int32(clipRange(roundShift(int64(value), shift), colMin, colMax))
		} else {
			value = clipRange(int64(value), colMin, colMax)
		}
		scratch[i] = value
	}
	for col := 0; col < width; col++ {
		switch vertical {
		case tx1DDCT:
			inverseDCT1D(scratch[col:], width, height, colMin, colMax)
		case tx1DIdentity:
			inverseIdentity1D(scratch[col:], width, height)
		}
	}
	dst := make([]int16, width*height)
	for i, value := range scratch {
		dst[i] = clipInt16(int32(roundShift(int64(value), 4)))
	}
	return dst
}
