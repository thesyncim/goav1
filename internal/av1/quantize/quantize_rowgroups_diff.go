package quantize

import "math/bits"

// TryQuantizeBRowGroupsDiffTrusted combines QuantizeBRowGroupsTrusted with
// the encoder's following skip-minus-code distortion and coarse-rate pass.
// It accepts the same trusted live-row proof and contiguous n=8/16/32 layout,
// plus a trusted live-column bound: every coefficient of the columns at or
// past cols (1 <= cols <= n) must also lie below its zero bin, and those
// columns are not read (SVT's N2/N4 forward transforms leave them zero).
// The caller must still use the original path when handled is false; in that
// case nothing has been written. In particular, overlapping coefficient and
// level storage keeps the original two-pass ordering.
//
// The difference uses AC dequantization followed by the same DC correction as
// the encoder's SIMD statistics. Rate is sum((2+bits.Len16(abs(level)))<<9)
// for nonzero levels, so rate==0 is exactly the all-zero result. This helper
// never replaces the absolute distortion needed by transform-type trials.
//
// eob is the end of block quantize_b reports: one past the largest scan
// position iscan[pos] of a nonzero level (iscan covers the n*n block).
func TryQuantizeBRowGroupsDiffTrusted(qcoeff []int16, coeff []int32, n int, live uint8, cols int, q Quantizer, txScale uint8, iscan []int16) (diff, rate int64, eob int, handled bool) {
	if quantizeBRowGroupsDiffImpl == nil || len(iscan) < n*n {
		return 0, 0, 0, false
	}
	return quantizeBRowGroupsDiffImpl(qcoeff, coeff, n, live, cols, q, txScale, iscan)
}

var quantizeBRowGroupsDiffImpl func([]int16, []int32, int, uint8, int, Quantizer, uint8, []int16) (int64, int64, int, bool)

// rowGroupDCEOB folds the DC level into an end of block taken over the AC
// rule's levels: the DC sits at scan position 0, so it decides the result
// only when no AC level survived.
func rowGroupDCEOB(eob int, dc int16) int {
	if eob > 1 {
		return eob
	}
	if dc != 0 {
		return 1
	}
	return 0
}

// rowGroupLevelRate is one level's coarse rate: 2 + bits.Len16(|v|) when
// non-zero.
func rowGroupLevelRate(v int16) int64 {
	if v == 0 {
		return 0
	}
	if v < 0 {
		v = -v
	}
	return int64(2 + bits.Len16(uint16(v)))
}
