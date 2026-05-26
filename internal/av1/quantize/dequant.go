// Ported from libaom:
//   av1/common/quant_common.c
//   av1/decoder/decodetxb.c
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

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

// DequantizeBlock multiplies quantized transform coefficients into dst. Both
// coeff and dst use AV1 coefficient order: coeff_idx = col * stride + row. The
// DC coefficient at (0,0) uses q.DC; every other coefficient uses q.AC.
func DequantizeBlock(dst []int32, dstStride int, coeff []int16, coeffStride int, width int, height int, q Quantizer) error {
	return DequantizeBlockScaled(dst, dstStride, coeff, coeffStride, width, height, q, 0)
}

// TransformScale ports libaom's av1_get_tx_scale() for a validated AV1
// transform shape. Callers should pass the coded transform dimensions, not the
// adjusted coefficient scan size used by 64-wide/high transforms.
func TransformScale(width int, height int) (uint8, error) {
	if width <= 0 || height <= 0 {
		return 0, ErrInvalidQuantizer
	}
	pixels, ok := checkedMul(width, height)
	if !ok {
		return 0, ErrInvalidQuantizer
	}
	var scale uint8
	if pixels > 256 {
		scale++
	}
	if pixels > 1024 {
		scale++
	}
	return scale, nil
}

// DequantizeBlockScaled is DequantizeBlock with libaom's transform scale
// applied after the DC/AC dequant multiply. The scale is the value returned by
// TransformScale for the coded transform size. The 8-bit dq_coeff clamp is
// used; high bit-depth callers should use DequantizeBlockScaledBitDepth.
func DequantizeBlockScaled(dst []int32, dstStride int, coeff []int16, coeffStride int, width int, height int, q Quantizer, txScale uint8) error {
	return dequantizeBlockScaled(dst, dstStride, coeff, coeffStride, width, height, q, txScale, nil, 8)
}

// DequantizeBlockScaledBitDepth is DequantizeBlockScaled with the bit-depth-
// dependent post-shift clamp libaom applies in decode_coefs():
// |dq_coeff| <= (1 << (7 + bd)) - 1, with the symmetric negative bound
// extending one further. See av1/decoder/decodetxb.c:312 in libaom.
func DequantizeBlockScaledBitDepth(dst []int32, dstStride int, coeff []int16, coeffStride int, width int, height int, q Quantizer, txScale uint8, bitDepth uint8) error {
	return dequantizeBlockScaled(dst, dstStride, coeff, coeffStride, width, height, q, txScale, nil, bitDepth)
}

// DequantizeBlockScaledQMatrix applies libaom's inverse quantization matrix
// weighting before the transform-size dequant shift. iqMatrix is indexed by
// libaom's row-major coefficient position and must cover width*height entries.
// The 8-bit dq_coeff clamp is used; high bit-depth callers should use
// DequantizeBlockScaledQMatrixBitDepth.
func DequantizeBlockScaledQMatrix(dst []int32, dstStride int, coeff []int16, coeffStride int, width int, height int, q Quantizer, txScale uint8, iqMatrix []uint16) error {
	return dequantizeBlockScaledQMatrix(dst, dstStride, coeff, coeffStride, width, height, q, txScale, iqMatrix, 8)
}

// DequantizeBlockScaledQMatrixBitDepth is DequantizeBlockScaledQMatrix with
// the bit-depth-dependent post-shift clamp libaom applies in decode_coefs().
func DequantizeBlockScaledQMatrixBitDepth(dst []int32, dstStride int, coeff []int16, coeffStride int, width int, height int, q Quantizer, txScale uint8, iqMatrix []uint16, bitDepth uint8) error {
	return dequantizeBlockScaledQMatrix(dst, dstStride, coeff, coeffStride, width, height, q, txScale, iqMatrix, bitDepth)
}

func dequantizeBlockScaledQMatrix(dst []int32, dstStride int, coeff []int16, coeffStride int, width int, height int, q Quantizer, txScale uint8, iqMatrix []uint16, bitDepth uint8) error {
	if width <= 0 || height <= 0 {
		return ErrInvalidQuantizer
	}
	// Use checkedMul before the slice-length compare so a hostile caller
	// cannot trigger silent integer overflow on width*height.
	samples, ok := checkedMul(width, height)
	if !ok || len(iqMatrix) < samples {
		return ErrInvalidQuantizer
	}
	return dequantizeBlockScaled(dst, dstStride, coeff, coeffStride, width, height, q, txScale, iqMatrix, bitDepth)
}

func dequantizeBlockScaled(dst []int32, dstStride int, coeff []int16, coeffStride int, width int, height int, q Quantizer, txScale uint8, iqMatrix []uint16, bitDepth uint8) error {
	if q.DC <= 0 || q.AC <= 0 ||
		txScale > 2 ||
		width <= 0 ||
		height <= 0 ||
		dstStride < height ||
		coeffStride < height {
		return ErrInvalidQuantizer
	}
	if !coeffBlockFits(len(dst), dstStride, width, height) ||
		!coeffBlockFits(len(coeff), coeffStride, width, height) {
		return ErrInvalidQuantizer
	}
	dqMin, dqMax, ok := dqCoeffBounds(bitDepth)
	if !ok {
		return ErrInvalidQuantizer
	}

	for col := range width {
		dstCol := dst[col*dstStride : col*dstStride+height]
		coeffCol := coeff[col*coeffStride : col*coeffStride+height]
		for row := range height {
			scale := q.AC
			if row == 0 && col == 0 {
				scale = q.DC
			}
			if iqMatrix != nil {
				scale = (int32(iqMatrix[row*width+col])*scale + (1 << (qmBits - 1))) >> qmBits
			}
			level := int32(coeffCol[row])
			negative := level < 0
			if negative {
				level = -level
			}
			// libaom: dq_coeff = (int32)((int64)level * dqv & 0xffffff). The 24-bit
			// mask saturates the product before the txScale shift to match
			// libaom's tran_low_t bitwidth assumption.
			product := (level * scale) & 0xffffff
			level = product >> txScale
			if negative {
				level = -level
			}
			// libaom decode_coefs() (av1/decoder/decodetxb.c:312) clamps the
			// dequantized coefficient to ±(1<<(7+bd)) — 16 bits for 8-bit, 18
			// bits for 10-bit, 20 bits for 12-bit. Without this clamp the
			// inverse transform sees values libaom's tran_low_t never carries.
			if level < dqMin {
				level = dqMin
			} else if level > dqMax {
				level = dqMax
			}
			dstCol[row] = level
		}
	}
	return nil
}

// dqCoeffBounds returns the post-shift dequantized-coefficient clamp range that
// libaom applies in decode_coefs() for the given bitDepth. Supported bitDepth
// values are 8, 10, and 12.
func dqCoeffBounds(bitDepth uint8) (min int32, max int32, ok bool) {
	switch bitDepth {
	case 8, 10, 12:
	default:
		return 0, 0, false
	}
	bits := uint(7 + bitDepth)
	max = int32(1)<<bits - 1
	min = -int32(1) << bits
	return min, max, true
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

func coeffBlockFits(length int, stride int, width int, height int) bool {
	if stride <= 0 || width <= 0 || height <= 0 {
		return false
	}
	lastColOffset, ok := checkedMul(width-1, stride)
	if !ok {
		return false
	}
	needed, ok := checkedAdd(lastColOffset, height)
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
