package quantize

import "math/bits"

// TryQuantizeBRowGroupsDiffTrusted combines QuantizeBRowGroupsTrusted with
// the encoder's following skip-minus-code distortion and coarse-rate pass.
// It accepts the same trusted live-row proof and contiguous n=8/16/32 layout.
// The caller must still use the original path when handled is false; in that
// case nothing has been written. In particular, overlapping coefficient and
// level storage keeps the original two-pass ordering.
//
// The difference uses AC dequantization followed by the same DC correction as
// the encoder's SIMD statistics. Rate is sum((2+bits.Len16(abs(level)))<<9)
// for nonzero levels, so rate==0 is exactly the all-zero result. This helper
// never replaces the absolute distortion needed by transform-type trials.
func TryQuantizeBRowGroupsDiffTrusted(qcoeff []int16, coeff []int32, n int, live uint8, q Quantizer, txScale uint8) (diff, rate int64, handled bool) {
	if quantizeBRowGroupsDiffImpl == nil {
		return 0, 0, false
	}
	return quantizeBRowGroupsDiffImpl(qcoeff, coeff, n, live, q, txScale)
}

var quantizeBRowGroupsDiffImpl func([]int16, []int32, int, uint8, Quantizer, uint8) (int64, int64, bool)

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
