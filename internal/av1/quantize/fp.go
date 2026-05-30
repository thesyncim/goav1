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

		absCoeff := int64(coeff[rc])
		negative := absCoeff < 0
		if negative {
			absCoeff = -absCoeff
		}

		tmp := int32(0)
		thresh := int64(q.Dequant[idx])
		if (absCoeff << (1 + q.LogScale)) >= thresh {
			absCoeff += int64(rounding[idx])
			absCoeff = clampInt64(absCoeff, minInt16, maxInt16)
			tmp = int32((absCoeff * int64(q.Quant[idx])) >> (16 - q.LogScale))
			if tmp != 0 {
				qc := tmp
				dqc := (tmp * int32(q.Dequant[idx])) >> q.LogScale
				if negative {
					qc = -qc
					dqc = -dqc
				}
				qcoeff[rc] = qc
				dqcoeff[rc] = dqc
			}
		}
		if tmp != 0 {
			eob = i + 1
		}
	}
	return eob, nil
}

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
