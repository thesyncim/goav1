package goav1

import internalquantize "github.com/thesyncim/goav1/internal/av1/quantize"

// QuantizerPlane selects between the luma and chroma quantization parameter
// sets used by AV1.
type QuantizerPlane = internalquantize.Plane

// Quantizer holds per-plane DC/AC quantizers, including any AV1 quantization
// matrix (qmatrix) levels in effect. Instances are produced by PlaneQuantizer.
type Quantizer = internalquantize.Quantizer

// QuantizerMaxQIndex is the maximum AV1 base_q_idx value (255).
//
// QuantizerPlaneY, QuantizerPlaneU, and QuantizerPlaneV identify the luma,
// U-chroma, and V-chroma quantizer planes respectively.
const (
	QuantizerMaxQIndex = internalquantize.MaxQIndex

	QuantizerPlaneY QuantizerPlane = internalquantize.PlaneY
	QuantizerPlaneU QuantizerPlane = internalquantize.PlaneU
	QuantizerPlaneV QuantizerPlane = internalquantize.PlaneV
)

// ErrQuantizeInvalidQuantizer is returned by the quantizer helpers when their
// arguments fall outside the AV1 spec ranges.
var ErrQuantizeInvalidQuantizer = internalquantize.ErrInvalidQuantizer

// ClampQIndex returns qIndex shifted by delta and clamped to [0,
// QuantizerMaxQIndex]. It implements the AV1 delta-q / segment-q clamping
// rules used while building per-block quantizers.
func ClampQIndex(qIndex uint8, delta int16) uint8 {
	return internalquantize.ClampQIndex(qIndex, delta)
}

// DCQuant returns the AV1 DC quantizer for (qIndex + delta) at bitDepth (8,
// 10, or 12). It is the building block used by PlaneQuantizer.
func DCQuant(qIndex uint8, delta int16, bitDepth uint8) (int32, error) {
	return internalquantize.DCQuant(qIndex, delta, bitDepth)
}

// ACQuant returns the AV1 AC quantizer for (qIndex + delta) at bitDepth (8,
// 10, or 12). It is the building block used by PlaneQuantizer.
func ACQuant(qIndex uint8, delta int16, bitDepth uint8) (int32, error) {
	return internalquantize.ACQuant(qIndex, delta, bitDepth)
}

// PlaneQuantizer assembles the full set of DC/AC quantizers and qmatrix
// levels for one plane, given the frame quantization parameters, the
// per-block qIndex, and the sequence bitDepth.
func PlaneQuantizer(params QuantizationParams, qIndex uint8, bitDepth uint8, plane QuantizerPlane) (Quantizer, error) {
	return internalquantize.PlaneQuantizer(params, qIndex, bitDepth, plane)
}

// TransformScale returns the AV1 transform scale (in bits) for a transform
// block of the supplied width and height.
func TransformScale(width int, height int) (uint8, error) {
	return internalquantize.TransformScale(width, height)
}

// DequantizeBlock dequantizes coeff into dst using the supplied Quantizer.
// dstStride and coeffStride are sample counts (not bytes). The block geometry
// must match the layout encoded in q.
func DequantizeBlock(dst []int32, dstStride int, coeff []int16, coeffStride int, width int, height int, q Quantizer) error {
	return internalquantize.DequantizeBlock(dst, dstStride, coeff, coeffStride, width, height, q)
}

// DequantizeBlockScaled is DequantizeBlock with an explicit caller-supplied
// transform scale (typically obtained from TransformScale) instead of using
// the default scale implied by the block geometry.
func DequantizeBlockScaled(dst []int32, dstStride int, coeff []int16, coeffStride int, width int, height int, q Quantizer, txScale uint8) error {
	return internalquantize.DequantizeBlockScaled(dst, dstStride, coeff, coeffStride, width, height, q, txScale)
}
