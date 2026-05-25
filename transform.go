package goav1

import internaltransform "github.com/thesyncim/goav1/internal/av1/transform"

type TransformSize = internaltransform.Size
type TransformType = internaltransform.Type
type TransformClass = internaltransform.Class
type TransformScanMode = internaltransform.ScanMode

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

var ErrTransformInvalidTransform = internaltransform.ErrInvalidTransform

func TransformSizeFromDimensions(width int, height int) (TransformSize, error) {
	return internaltransform.SizeFromDimensions(width, height)
}

func TransformScanSize(size TransformSize) (TransformSize, error) {
	return internaltransform.ScanSize(size)
}

func TransformDefaultScanMode(size TransformSize, class TransformClass) (TransformScanMode, error) {
	return internaltransform.DefaultScanMode(size, class)
}

func FillTransformDefaultScan(scan []int16, inverse []int16, size TransformSize, class TransformClass) error {
	return internaltransform.FillDefaultScan(scan, inverse, size, class)
}

func FillTransformScanOrder(scan []int16, inverse []int16, size TransformSize, mode TransformScanMode) error {
	return internaltransform.FillScanOrder(scan, inverse, size, mode)
}

func TransformScratchLenForType(typ TransformType, size TransformSize) (int, error) {
	return internaltransform.ScratchLenForType(typ, size)
}

func InverseTransformBlock(dst []int16, dstStride int, coeff []int32, coeffStride int, scratch []int32, size TransformSize, typ TransformType) error {
	return internaltransform.InverseBlock(dst, dstStride, coeff, coeffStride, scratch, size, typ)
}
