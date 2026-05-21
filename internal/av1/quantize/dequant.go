package quantize

import "github.com/thesyncim/goav1/internal/av1/parser"

const MaxQIndex = QIndexRange - 1

// Plane identifies the component whose quantization deltas should be used.
type Plane uint8

const (
	PlaneY Plane = iota
	PlaneU
	PlaneV
)

// Quantizer stores the DC and AC dequantization steps for one transform block.
type Quantizer struct {
	DC int32
	AC int32
}

// ClampQIndex applies an AV1 signed qindex delta and clips the result to the
// normative qindex range.
func ClampQIndex(qIndex uint8, delta int16) uint8 {
	q := int(qIndex) + int(delta)
	if q < 0 {
		return 0
	}
	if q > MaxQIndex {
		return MaxQIndex
	}
	return uint8(q)
}

// DCQuant returns the AV1 DC dequantization step for qIndex+delta.
func DCQuant(qIndex uint8, delta int16, bitDepth uint8) (int32, error) {
	table, err := bitDepthTable(bitDepth)
	if err != nil {
		return 0, err
	}
	return dequantLookup[table][ClampQIndex(qIndex, delta)][0], nil
}

// ACQuant returns the AV1 AC dequantization step for qIndex+delta.
func ACQuant(qIndex uint8, delta int16, bitDepth uint8) (int32, error) {
	table, err := bitDepthTable(bitDepth)
	if err != nil {
		return 0, err
	}
	return dequantLookup[table][ClampQIndex(qIndex, delta)][1], nil
}

// PlaneQuantizer derives DC/AC dequantization steps for one frame plane using
// parsed quantization_params() deltas and the supplied segment/current qindex.
func PlaneQuantizer(params parser.QuantizationParams, qIndex uint8, bitDepth uint8, plane Plane) (Quantizer, error) {
	var dcDelta int16
	var acDelta int16
	switch plane {
	case PlaneY:
		dcDelta = params.YDCDelta
	case PlaneU:
		dcDelta = params.UDCDelta
		acDelta = params.UACDelta
	case PlaneV:
		dcDelta = params.VDCDelta
		acDelta = params.VACDelta
	default:
		return Quantizer{}, ErrInvalidQuantizer
	}

	dc, err := DCQuant(qIndex, dcDelta, bitDepth)
	if err != nil {
		return Quantizer{}, err
	}
	ac, err := ACQuant(qIndex, acDelta, bitDepth)
	if err != nil {
		return Quantizer{}, err
	}
	return Quantizer{DC: dc, AC: ac}, nil
}

// DequantizeBlock multiplies quantized transform coefficients into dst. The
// DC coefficient at (0,0) uses q.DC; every other coefficient uses q.AC.
func DequantizeBlock(dst []int32, dstStride int, coeff []int16, coeffStride int, width int, height int, q Quantizer) error {
	if q.DC <= 0 || q.AC <= 0 ||
		width <= 0 ||
		height <= 0 ||
		dstStride < width ||
		coeffStride < width {
		return ErrInvalidQuantizer
	}
	if !blockFits(len(dst), dstStride, width, height) ||
		!blockFits(len(coeff), coeffStride, width, height) {
		return ErrInvalidQuantizer
	}

	for row := 0; row < height; row++ {
		dstLine := dst[row*dstStride : row*dstStride+width]
		coeffLine := coeff[row*coeffStride : row*coeffStride+width]
		for col := 0; col < width; col++ {
			scale := q.AC
			if row == 0 && col == 0 {
				scale = q.DC
			}
			dstLine[col] = int32(coeffLine[col]) * scale
		}
	}
	return nil
}

func bitDepthTable(bitDepth uint8) (int, error) {
	switch bitDepth {
	case 8:
		return 0, nil
	case 10:
		return 1, nil
	case 12:
		return 2, nil
	default:
		return 0, ErrInvalidQuantizer
	}
}

func blockFits(length int, stride int, width int, height int) bool {
	if stride <= 0 || width <= 0 || height <= 0 {
		return false
	}
	lastRowOffset, ok := checkedMul(height-1, stride)
	if !ok {
		return false
	}
	needed, ok := checkedAdd(lastRowOffset, width)
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
