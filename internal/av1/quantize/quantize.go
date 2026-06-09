package quantize

// quantize.go is the encoder's forward quantizer, the inverse of this
// package's dequantization path. The rule is plain truncation toward zero of
// the transform-domain magnitude scaled by the DC/AC step:
//
//	|q| = (|c| << txScale) / scale
//
// which inverts dequantScalar's (|q| * scale) >> txScale to within one step.
// Rate-distortion-optimised rounding (libaom quantize_b's zbin/round shaping)
// is a later quality refinement; any deterministic rule here yields a valid
// bitstream because the encoder reconstructs through the same dequant the
// decoder runs.

// QuantizeBlockScaled quantizes transform-domain coefficients into qcoeff.
// Both buffers use AV1 coefficient order (coeff_idx = col*stride + row); the
// DC coefficient at (0,0) uses q.DC and every other position q.AC, matching
// DequantizeBlockScaled. txScale is TransformScale of the coded transform.
func QuantizeBlockScaled(qcoeff []int16, qStride int, coeff []int32, coeffStride int, width int, height int, q Quantizer, txScale uint8) error {
	if q.DC <= 0 || q.AC <= 0 ||
		txScale > 2 ||
		width <= 0 ||
		height <= 0 ||
		qStride < height ||
		coeffStride < height {
		return ErrInvalidQuantizer
	}
	if !coeffBlockFits(len(qcoeff), qStride, width, height) ||
		!coeffBlockFits(len(coeff), coeffStride, width, height) {
		return ErrInvalidQuantizer
	}
	for col := range width {
		qCol := qcoeff[col*qStride : col*qStride+height]
		cCol := coeff[col*coeffStride : col*coeffStride+height]
		for row := range height {
			scale := q.AC
			if col == 0 && row == 0 {
				scale = q.DC
			}
			qCol[row] = quantizeScalar(cCol[row], scale, txScale)
		}
	}
	return nil
}

// quantizeScalar truncates one transform-domain coefficient to its quantized
// level. The int16 clamp keeps the level inside the coefficient writer's
// domain; dequantScalar's 24-bit product mask is unreachable for in-range
// levels at supported bit depths.
func quantizeScalar(coeff int32, scale int32, txScale uint8) int16 {
	level := coeff
	negative := level < 0
	if negative {
		level = -level
	}
	level = int32((int64(level) << txScale) / int64(scale))
	const maxLevel = int32(^uint16(0) >> 1)
	if level > maxLevel {
		level = maxLevel
	}
	if negative {
		level = -level
	}
	return int16(level)
}
