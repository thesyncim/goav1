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
			tx1DSupported(horizontal, size.Width) &&
			tx1DSupported(vertical, size.Height)
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
	return size.Width * size.Height, nil
}

// InverseBlock writes a transform residual block to dst using t. IDTX keeps
// its zero-scratch direct path; the other supported separable transforms use
// caller-provided scratch.
func InverseBlock(dst []int16, dstStride int, coeff []int32, coeffStride int, scratch []int32, size Size, t Type) error {
	if t == TypeIDTX {
		return InverseIdentityBlock(dst, dstStride, coeff, coeffStride, size)
	}
	if !t.Supported(size) {
		return ErrInvalidTransform
	}
	return inverseSeparableBlock(dst, dstStride, coeff, coeffStride, scratch, size, t)
}
