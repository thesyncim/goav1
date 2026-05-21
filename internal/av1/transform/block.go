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
	case TypeDCTDCT:
		return dctBlockSupported(size)
	case TypeIDTX:
		return identityBlockSupported(size)
	default:
		return false
	}
}

// ScratchLenForType returns the number of int32 scratch values needed by t.
func ScratchLenForType(t Type, size Size) (int, error) {
	if !t.Supported(size) {
		return 0, ErrInvalidTransform
	}
	if t == TypeDCTDCT {
		return size.Width * size.Height, nil
	}
	return 0, nil
}

// InverseBlock writes a transform residual block to dst using t. DCT_DCT uses
// caller-provided scratch; IDTX does not need scratch.
func InverseBlock(dst []int16, dstStride int, coeff []int32, coeffStride int, scratch []int32, size Size, t Type) error {
	switch t {
	case TypeDCTDCT:
		return InverseDCTBlock(dst, dstStride, coeff, coeffStride, scratch, size)
	case TypeIDTX:
		return InverseIdentityBlock(dst, dstStride, coeff, coeffStride, size)
	default:
		return ErrInvalidTransform
	}
}
