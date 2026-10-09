// Ported from libaom: av1/encoder/av1_quantize.c
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

package quantize

// FPQuantizer stores libaom's fixed-point quantization parameters for DC and
// AC coefficients.
type FPQuantizer struct {
	Quant    [2]int16
	Dequant  [2]int16
	Round    [2]int16
	LogScale uint8
}

// QuantizeFPNoQMatrix applies libaom's av1_quantize_fp_no_qmatrix scalar path
// and writes qcoeff/dqcoeff in coefficient raster order.
func QuantizeFPNoQMatrix(qcoeff []int32, dqcoeff []int32, coeff []int32, scan []int16, q FPQuantizer) (int, error) {
	count := len(scan)
	if count == 0 ||
		len(coeff) < count ||
		len(qcoeff) < count ||
		len(dqcoeff) < count ||
		q.LogScale > 15 ||
		q.Quant[0] <= 0 ||
		q.Quant[1] <= 0 ||
		q.Dequant[0] <= 0 ||
		q.Dequant[1] <= 0 ||
		q.Round[0] < 0 ||
		q.Round[1] < 0 {
		return 0, ErrInvalidQuantizer
	}
	for _, rc := range scan {
		if rc < 0 || int(rc) >= count {
			return 0, ErrInvalidQuantizer
		}
	}

	if quantizeFPNoQMatrixImpl != nil {
		if eob, ok := quantizeFPNoQMatrixImpl(qcoeff[:count], dqcoeff[:count], coeff[:count], scan, q); ok {
			return eob, nil
		}
	}

	for i := range count {
		qcoeff[i] = 0
		dqcoeff[i] = 0
	}

	rounding := [2]int32{
		roundPowerOfTwo(int32(q.Round[0]), q.LogScale),
		roundPowerOfTwo(int32(q.Round[1]), q.LogScale),
	}
	eob := 0
	for i, rawRC := range scan {
		rc := int(rawRC)
		idx := 0
		if rc != 0 {
			idx = 1
		}

		qc, dqc, nonzero := quantizeFPNoQMatrixScalarCoeff(coeff[rc], idx, q, rounding)
		qcoeff[rc] = qc
		dqcoeff[rc] = dqc
		if nonzero {
			eob = i + 1
		}
	}
	return eob, nil
}

func quantizeFPNoQMatrixScalarCoeff(coeff int32, idx int, q FPQuantizer, rounding [2]int32) (int32, int32, bool) {
	absCoeff := int64(coeff)
	negative := absCoeff < 0
	if negative {
		absCoeff = -absCoeff
	}

	thresh := int64(q.Dequant[idx])
	if (absCoeff << (1 + q.LogScale)) < thresh {
		return 0, 0, false
	}
	absCoeff += int64(rounding[idx])
	absCoeff = clampInt64(absCoeff, minInt16, maxInt16)
	tmp := int32((absCoeff * int64(q.Quant[idx])) >> (16 - q.LogScale))
	if tmp == 0 {
		return 0, 0, false
	}
	qc := tmp
	dqc := (tmp * int32(q.Dequant[idx])) >> q.LogScale
	if negative {
		qc = -qc
		dqc = -dqc
	}
	return qc, dqc, true
}

// quantizeFPNoQMatrixImpl, when non-nil, may handle QuantizeFPNoQMatrix after
// public input validation. It returns ok=false for scan shapes it does not
// support, leaving the scalar path to produce the canonical result.
var quantizeFPNoQMatrixImpl func(qcoeff []int32, dqcoeff []int32, coeff []int32, scan []int16, q FPQuantizer) (eob int, ok bool)

func roundPowerOfTwo(v int32, bits uint8) int32 {
	if bits == 0 {
		return v
	}
	return (v + (1 << (bits - 1))) >> bits
}

func clampInt64(v int64, lo int64, hi int64) int64 {
	return min(max(v, lo), hi)
}

const (
	minInt16 = -1 << 15
	maxInt16 = 1<<15 - 1
)
