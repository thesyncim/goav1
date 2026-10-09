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
		if height == 8 {
			full[0], full[1] = minInt16, maxInt16
			if !int16ColumnSIMDInputSafe(full, 8, height, minInt16, maxInt16) {
				t.Fatal("DCT8 rejected the full int16 input range")
			}
			full[0], full[1] = 30000, 30000
			if !int16ColumnSIMDInputSafe(full, 8, height, minInt16, maxInt16) {
				t.Fatal("DCT8 rejected a full-range even-butterfly input")
			}
			if !int16ColumnSIMDInputSafe(full, 8, height, 1, maxInt16) {
				t.Fatal("DCT8 rejected a valid positive-only int16 clamp")
			}
			continue
		}
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
	if !int16ColumnSIMDInputSafe(input, 8, 8, minInt16, maxInt16) {
		t.Fatal("DCT8 rejected an exact wide pre-rotation sum")
	}
	if int16ColumnSIMDInputSafe(make([]int16, 8*16), 8, 16, 1, maxInt16) {
		t.Fatal("accepted a deeper DCT clamp interval that excludes zero")
	}
}

func TestInverseBlockBitDepth8BenchmarkFixturesSelectExpectedColumnPath(t *testing.T) {
	if !int16ColumnFast {
		t.Skip("SIMD int16 column path is not bound")
	}
	for _, side := range []int{8, 16, 32, 64} {
		for _, pattern := range []string{"bounded", "high-range"} {
			t.Run(inverseBenchSizeName(side)+"/"+pattern, func(t *testing.T) {
				coeff := inverseColumnBenchmarkCoefficients(side, pattern)
				size := Size{Width: uint8(side), Height: uint8(side)}
				scratch := make([]int32, side*side)
				col16 := make([]int16, side*side)
				rowMin, rowMax, colMin, colMax, ok := stageRangeBounds(8)
				if !ok {
					t.Fatal("missing 8-bit stage bounds")
				}
				usedInt16, err := inverseSeparableBlockClampedRowsToScratch(
					coeff, side, scratch, size, TypeDCTDCT,
					rowMin, rowMax, colMin, colMax, 0, col16, false,
				)
				if err != nil {
					t.Fatal(err)
				}
				wantInt16 := pattern == "bounded"
				if usedInt16 != wantInt16 {
					t.Fatalf("usedInt16=%t want %t", usedInt16, wantInt16)
				}
			})
		}
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
	// crossing the profitability threshold. The exact widened kernel supports
	// this range, but production dispatch should use the faster int32/NEON path.
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
			t.Fatal("high-magnitude DCT8 fixture did not take the faster int32 path")
		}
	}
	compareInverseBlockBitDepthToScalar(t, size, TypeVDCT, coeff)

	// DCT16 still uses its certified narrow range. This identity-row block
	// produces an input above the DCT16 SIMD bound and must take the int32 path.
	coeff16 := make([]int32, 16*16)
	coeff16[0], coeff16[8] = 12000, 12000
	size16 := Size{Width: 16, Height: 16}
	if int16ColumnFast {
		rowMin, rowMax, colMin, colMax, _ := stageRangeBounds(8)
		usedSIMD, err := inverseSeparableBlockClampedRowsToScratch(
			coeff16, 16, make([]int32, 16*16), size16, TypeVDCT,
			rowMin, rowMax, colMin, colMax, 0, make([]int16, 16*16), false,
		)
		if err != nil {
			t.Fatal(err)
		}
		if usedSIMD {
			t.Fatal("DCT16 SIMD path accepted values outside its certified interval")
		}
	}
	compareInverseBlockBitDepthToScalar(t, size16, TypeVDCT, coeff16)
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
