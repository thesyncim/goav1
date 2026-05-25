package goav1

import internalquantize "github.com/thesyncim/goav1/internal/av1/quantize"

type QuantizerPlane = internalquantize.Plane
type Quantizer = internalquantize.Quantizer

const (
	QuantizerMaxQIndex = internalquantize.MaxQIndex

	QuantizerPlaneY QuantizerPlane = internalquantize.PlaneY
	QuantizerPlaneU QuantizerPlane = internalquantize.PlaneU
	QuantizerPlaneV QuantizerPlane = internalquantize.PlaneV
)

var ErrQuantizeInvalidQuantizer = internalquantize.ErrInvalidQuantizer

func ClampQIndex(qIndex uint8, delta int16) uint8 {
	return internalquantize.ClampQIndex(qIndex, delta)
}

func DCQuant(qIndex uint8, delta int16, bitDepth uint8) (int32, error) {
	return internalquantize.DCQuant(qIndex, delta, bitDepth)
}

func ACQuant(qIndex uint8, delta int16, bitDepth uint8) (int32, error) {
	return internalquantize.ACQuant(qIndex, delta, bitDepth)
}

func PlaneQuantizer(params QuantizationParams, qIndex uint8, bitDepth uint8, plane QuantizerPlane) (Quantizer, error) {
	return internalquantize.PlaneQuantizer(params, qIndex, bitDepth, plane)
}

func TransformScale(width int, height int) (uint8, error) {
	return internalquantize.TransformScale(width, height)
}

func DequantizeBlock(dst []int32, dstStride int, coeff []int16, coeffStride int, width int, height int, q Quantizer) error {
	return internalquantize.DequantizeBlock(dst, dstStride, coeff, coeffStride, width, height, q)
}

func DequantizeBlockScaled(dst []int32, dstStride int, coeff []int16, coeffStride int, width int, height int, q Quantizer, txScale uint8) error {
	return internalquantize.DequantizeBlockScaled(dst, dstStride, coeff, coeffStride, width, height, q, txScale)
}
