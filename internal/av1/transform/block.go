// Ported from libaom:
//   av1/common/av1_inv_txfm2d.c
//   av1/common/enums.h
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

package transform

// Type identifies the AV1 inverse transform kind for a residual block.
type Type uint8

const (
	TypeDCTDCT Type = iota
	TypeADSTDCT
	TypeDCTADST
	TypeADSTADST
	TypeFlipADSTDCT
	TypeDCTFlipADST
	TypeFlipADSTFlipADST
	TypeADSTFlipADST
	TypeFlipADSTADST
	TypeIDTX
	TypeVDCT
	TypeHDCT
	TypeVADST
	TypeHADST
	TypeVFlipADST
	TypeHFlipADST
	TypeCount
)

var typeClasses = [TypeCount]Class{
	TypeDCTDCT:           Class2D,
	TypeADSTDCT:          Class2D,
	TypeDCTADST:          Class2D,
	TypeADSTADST:         Class2D,
	TypeFlipADSTDCT:      Class2D,
	TypeDCTFlipADST:      Class2D,
	TypeFlipADSTFlipADST: Class2D,
	TypeADSTFlipADST:     Class2D,
	TypeFlipADSTADST:     Class2D,
	TypeIDTX:             Class2D,
	TypeVDCT:             ClassVert,
	TypeHDCT:             ClassHoriz,
	TypeVADST:            ClassVert,
	TypeHADST:            ClassHoriz,
	TypeVFlipADST:        ClassVert,
	TypeHFlipADST:        ClassHoriz,
}

// Valid reports whether t names an AV1 transform type.
func (t Type) Valid() bool {
	return t < TypeCount
}

func (t Type) Class() (Class, error) {
	if !t.Valid() {
		return 0, ErrInvalidTransform
	}
	return typeClasses[t], nil
}

// Supported reports whether t can currently be applied to size.
func (t Type) Supported(size Size) bool {
	switch t {
	case TypeIDTX:
		return identityBlockSupported(size)
	default:
		vertical, horizontal, ok := t.tx1DTypes()
		return ok &&
			size.Valid() &&
			tx1DSupported(horizontal, int(size.Width)) &&
			tx1DSupported(vertical, int(size.Height))
	}
}

// ScratchLenForType returns the number of int32 scratch values needed by t.
func ScratchLenForType(t Type, size Size) (int, error) {
	if !t.Supported(size) {
		return 0, ErrInvalidTransform
	}
	if t == TypeIDTX {
		return 0, nil
	}
	return int(size.Width) * int(size.Height), nil
}

// InverseBlock writes a transform residual block to dst using t. IDTX keeps
// its zero-scratch direct path; the other supported separable transforms use
// caller-provided scratch. The 8-bit stage-range clamps are used; high
// bit-depth callers should use InverseBlockBitDepth.
func InverseBlock(dst []int16, dstStride int, coeff []int32, coeffStride int, scratch []int32, size Size, t Type) error {
	if t == TypeIDTX {
		return InverseIdentityBlock(dst, dstStride, coeff, coeffStride, size)
	}
	if !t.Supported(size) {
		return ErrInvalidTransform
	}
	return inverseSeparableBlock(dst, dstStride, coeff, coeffStride, scratch, size, t)
}

// InverseBlockBitDepth writes a transform residual block to dst using t and
// the libaom inverse-transform stage-range clamps for the given pixel
// bitDepth. Supported bitDepth values are 8, 10, and 12. The row-input clamp
// is bd+8 bits and the column-input clamp is max(bd+6, 16) bits, mirroring
// clamp_buf() and av1_gen_inv_stage_range() in
// av1/common/av1_inv_txfm2d.c. Behaviour at bitDepth==8 is identical to
// InverseBlock.
func InverseBlockBitDepth(dst []int16, dstStride int, coeff []int32, coeffStride int, scratch []int32, size Size, t Type, bitDepth uint8) error {
	rowMin, rowMax, colMin, colMax, ok := stageRangeBounds(bitDepth)
	if !ok {
		return ErrInvalidTransform
	}
	if t == TypeIDTX {
		return inverseIdentityBlockClamped(dst, dstStride, coeff, coeffStride, size, rowMin, rowMax, colMin, colMax)
	}
	if !t.Supported(size) {
		return ErrInvalidTransform
	}
	return inverseSeparableBlockClamped(dst, dstStride, coeff, coeffStride, scratch, size, t, rowMin, rowMax, colMin, colMax)
}

// InverseDCTDCOnlyBlockBitDepth writes the residual for a DCT_DCT block whose
// only non-zero coefficient is DC. It uses the same 1D DCT kernels and stage
// clamps as InverseBlockBitDepth, but evaluates only the single horizontal and
// vertical DC basis before filling the block.
func InverseDCTDCOnlyBlockBitDepth(dst []int16, dstStride int, dc int32, scratch []int32, size Size, bitDepth uint8) error {
	rowMin, rowMax, colMin, colMax, ok := stageRangeBounds(bitDepth)
	if !ok {
		return ErrInvalidTransform
	}
	idx := sizeIndex(size)
	if idx < 0 || !sizeValidTable[idx] {
		return ErrInvalidTransform
	}
	width := int(size.Width)
	height := int(size.Height)
	if dstStride < width || len(scratch) < width+height || !blockFits(len(dst), dstStride, width, height) {
		return ErrInvalidTransform
	}
	if size.IsRect2() {
		dc = rect2Scale(dc)
	}

	row := scratch[:width:width]
	clear(row)
	row[0] = clipRange(int64(dc), rowMin, rowMax)
	inverse1DRow(row, width, tx1DDCT, rowMin, rowMax)

	v := row[0]
	if shift := int(sizeShiftTable[idx]); shift > 0 {
		v = clipRange(roundShift(int64(v), shift), colMin, colMax)
	} else {
		v = clipRange(int64(v), colMin, colMax)
	}

	col := scratch[width : width+height : width+height]
	clear(col)
	col[0] = v
	inverse1D(col, 1, height, tx1DDCT, colMin, colMax)

	for rowIndex := 0; rowIndex < height; rowIndex++ {
		sample := clipInt16(int32(roundShift(int64(col[rowIndex]), 4)))
		dstLine := dst[rowIndex*dstStride : rowIndex*dstStride+width : rowIndex*dstStride+width]
		for colIndex := range dstLine {
			dstLine[colIndex] = sample
		}
	}
	return nil
}

// stageRangeBounds returns the libaom inverse-transform stage clamp bounds for
// bitDepth. The row bound clamps to bd+8 bits; the column bound clamps to
// max(bd+6, 16) bits.
func stageRangeBounds(bitDepth uint8) (rowMin int32, rowMax int32, colMin int32, colMax int32, ok bool) {
	switch bitDepth {
	case 8, 10, 12:
	default:
		return 0, 0, 0, 0, false
	}
	rowBits := int(bitDepth) + 8
	colBits := max(int(bitDepth)+6, 16)
	rowMin = -int32(1) << uint(rowBits-1)
	rowMax = int32(1)<<uint(rowBits-1) - 1
	colMin = -int32(1) << uint(colBits-1)
	colMax = int32(1)<<uint(colBits-1) - 1
	return rowMin, rowMax, colMin, colMax, true
}
