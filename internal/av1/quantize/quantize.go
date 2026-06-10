package quantize

// quantize.go is the encoder's forward quantizer, the inverse of this
// package's dequantization path. QuantizeBlockScaled is plain truncation
// toward zero (|q| = (|c| << txScale) / scale, inverting dequantScalar to
// within one step); QuantizeBlockScaledFP is the libaom realtime rule
// (av1_quantize_fp_no_qmatrix: half-step rounding offset with a one-step
// deadzone), which trades slightly more rate for noticeably less rounding
// error. Any deterministic rule yields a valid bitstream because the encoder
// reconstructs through the same dequant the decoder runs.

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

// QuantizeBlockScaledFP quantizes with libaom's realtime rounding, the port
// of av1_quantize_fp_no_qmatrix (av1/encoder/av1_quantize.c) minus the
// scan/eob bookkeeping (callers walk coefficient order directly):
//
//	quant_fp = (1 << 16) / dequant            (av1_build_quantizer)
//	round_fp = (64 * dequant) >> 7            (qrounding_factor_fp, sharpness 0)
//	rounding = ROUND_POWER_OF_TWO(round_fp, txScale)
//	skip when (|c| << (1 + txScale)) < dequant
//	level    = ((|c| + rounding) * quant_fp) >> (16 - txScale)
func QuantizeBlockScaledFP(qcoeff []int16, qStride int, coeff []int32, coeffStride int, width int, height int, q Quantizer, txScale uint8) error {
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
	// Contiguous square blocks take the vector kernel when available.
	if width == height && qStride == height && coeffStride == height &&
		quantizeFPBlockImpl != nil && quantizeFPBlockImpl(qcoeff[:width*height], coeff[:width*height], width, q, txScale) {
		return nil
	}
	quantDC := int64(1<<16) / int64(q.DC)
	quantAC := int64(1<<16) / int64(q.AC)
	roundDC := roundPowerOfTwo((64*int32(q.DC))>>7, txScale)
	roundAC := roundPowerOfTwo((64*int32(q.AC))>>7, txScale)
	for col := range width {
		qCol := qcoeff[col*qStride : col*qStride+height]
		cCol := coeff[col*coeffStride : col*coeffStride+height]
		for row := range height {
			scale, quant, round := q.AC, quantAC, roundAC
			if col == 0 && row == 0 {
				scale, quant, round = q.DC, quantDC, roundDC
			}
			qCol[row] = quantizeScalarFP(cCol[row], scale, quant, round, txScale)
		}
	}
	return nil
}

// quantizeScalarFP applies the fp rounding rule to one coefficient.
func quantizeScalarFP(coeff int32, dequant int32, quant int64, round int32, txScale uint8) int16 {
	negative := coeff < 0
	abs := coeff
	if negative {
		abs = -abs
	}
	if abs<<(1+txScale) < dequant {
		return 0
	}
	a := int64(abs) + int64(round)
	const maxInt16 = int64(^uint16(0) >> 1)
	if a > maxInt16 {
		a = maxInt16
	}
	level := (a * quant) >> (16 - txScale)
	if level > maxInt16 {
		level = maxInt16
	}
	if negative {
		level = -level
	}
	return int16(level)
}

// quantizeFPBlockImpl, when non-nil, quantizes one contiguous square block
// with the fp rule and reports whether it handled the input; architectures
// without a kernel leave it nil. Implementations must be bit-exact with the
// scalar rule.
var quantizeFPBlockImpl func(qcoeff []int16, coeff []int32, n int, q Quantizer, txScale uint8) bool
