package goav1

import internaltransform "github.com/thesyncim/goav1/internal/av1/transform"

// TransformSize is a {Width, Height} pair describing a transform block's
// dimensions in samples.
type TransformSize = internaltransform.Size

// TransformType is one of the AV1 inverse transform kernels (DCT, ADST,
// FLIPADST, IDTX, or a separable combination).
type TransformType = internaltransform.Type

// TransformClass groups TransformTypes by their 1D direction (2D, horizontal,
// or vertical only).
type TransformClass = internaltransform.Class

// TransformScanMode selects the coefficient scan order applied during entropy
// decode and inverse transform.
type TransformScanMode = internaltransform.ScanMode

// AV1 transform kernel constants. Each TransformType* identifies the AV1
// separable 2D transform applied during inverse-transform.
//
// TransformClass2D / Horiz / Vert classify a TransformType by its 1D
// direction.
//
// TransformScanMode* values select the coefficient scan order for the
// entropy and inverse-transform stages.
const (
	TransformTypeDCTDCT           TransformType = internaltransform.TypeDCTDCT
	TransformTypeADSTDCT          TransformType = internaltransform.TypeADSTDCT
	TransformTypeDCTADST          TransformType = internaltransform.TypeDCTADST
	TransformTypeADSTADST         TransformType = internaltransform.TypeADSTADST
	TransformTypeFlipADSTDCT      TransformType = internaltransform.TypeFlipADSTDCT
	TransformTypeDCTFlipADST      TransformType = internaltransform.TypeDCTFlipADST
	TransformTypeFlipADSTFlipADST TransformType = internaltransform.TypeFlipADSTFlipADST
	TransformTypeADSTFlipADST     TransformType = internaltransform.TypeADSTFlipADST
	TransformTypeFlipADSTADST     TransformType = internaltransform.TypeFlipADSTADST
	TransformTypeIDTX             TransformType = internaltransform.TypeIDTX
	TransformTypeVDCT             TransformType = internaltransform.TypeVDCT
	TransformTypeHDCT             TransformType = internaltransform.TypeHDCT
	TransformTypeVADST            TransformType = internaltransform.TypeVADST
	TransformTypeHADST            TransformType = internaltransform.TypeHADST
	TransformTypeVFlipADST        TransformType = internaltransform.TypeVFlipADST
	TransformTypeHFlipADST        TransformType = internaltransform.TypeHFlipADST

	TransformClass2D    TransformClass = internaltransform.Class2D
	TransformClassHoriz TransformClass = internaltransform.ClassHoriz
	TransformClassVert  TransformClass = internaltransform.ClassVert

	TransformScanModeZigZag  TransformScanMode = internaltransform.ScanModeZigZag
	TransformScanModeColDiag TransformScanMode = internaltransform.ScanModeColDiag
	TransformScanModeRowDiag TransformScanMode = internaltransform.ScanModeRowDiag
	TransformScanModeCol1D   TransformScanMode = internaltransform.ScanModeCol1D
	TransformScanModeRow1D   TransformScanMode = internaltransform.ScanModeRow1D
)

// ErrTransformInvalidTransform is returned by the transform helpers when the
// requested (size, type, class, or scan-mode) combination is not defined by
// the AV1 specification.
var ErrTransformInvalidTransform = internaltransform.ErrInvalidTransform

// TransformSizeFromDimensions returns the canonical TransformSize for the
// supplied width and height in samples.
func TransformSizeFromDimensions(width int, height int) (TransformSize, error) {
	return internaltransform.SizeFromDimensions(width, height)
}

// TransformScanSize returns the scan-table size used for coefficient scans
// of the supplied transform block size.
func TransformScanSize(size TransformSize) (TransformSize, error) {
	return internaltransform.ScanSize(size)
}

// TransformDefaultScanMode returns the default scan mode selected by the AV1
// specification for the supplied block size and transform class.
func TransformDefaultScanMode(size TransformSize, class TransformClass) (TransformScanMode, error) {
	return internaltransform.DefaultScanMode(size, class)
}

// FillTransformDefaultScan populates the forward and inverse scan tables for
// the AV1 default scan order corresponding to (size, class). The caller-owned
// scan and inverse slices must each contain size.Width*size.Height entries.
func FillTransformDefaultScan(scan []int16, inverse []int16, size TransformSize, class TransformClass) error {
	return internaltransform.FillDefaultScan(scan, inverse, size, class)
}

// FillTransformScanOrder populates the forward and inverse scan tables for an
// explicit TransformScanMode. The caller-owned scan and inverse slices must
// each contain size.Width*size.Height entries.
func FillTransformScanOrder(scan []int16, inverse []int16, size TransformSize, mode TransformScanMode) error {
	return internaltransform.FillScanOrder(scan, inverse, size, mode)
}

// TransformScratchLenForType reports the number of int32 scratch words
// required by InverseTransformBlock for the given (typ, size).
func TransformScratchLenForType(typ TransformType, size TransformSize) (int, error) {
	return internaltransform.ScratchLenForType(typ, size)
}

// InverseTransformBlock applies the AV1 inverse transform of (typ, size) to
// coeff, writing residual samples into dst. dstStride and coeffStride are
// sample counts; scratch is caller-owned and must be sized via
// TransformScratchLenForType.
func InverseTransformBlock(dst []int16, dstStride int, coeff []int32, coeffStride int, scratch []int32, size TransformSize, typ TransformType) error {
	return internaltransform.InverseBlock(dst, dstStride, coeff, coeffStride, scratch, size, typ)
}
