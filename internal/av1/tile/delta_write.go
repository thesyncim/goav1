package tile

import "github.com/thesyncim/goav1/internal/av1/entropy"

// WriteSignedDelta codes one delta_qindex / delta_lf value, the forward of
// entropy.Reader.ReadSignedDelta and the port of libaom's write_delta_qindex
// (av1/encoder/bitstream.c): the clamped magnitude through the
// DELTA_Q_SMALL-symbol CDF, an escape coded as 3 bits of (bit count - 1) and
// the magnitude's low bits, then the sign.
func WriteSignedDelta(w *entropy.Writer, cdf *entropy.CDF, delta int) error {
	if w == nil || cdf == nil || cdf.Symbols() != entropy.DeltaSmall+1 {
		return ErrInvalidDecodeState
	}
	abs := delta
	if abs < 0 {
		abs = -abs
	}
	small := abs
	if small > entropy.DeltaSmall {
		small = entropy.DeltaSmall
	}
	w.WriteCDF(cdf, small)
	if abs >= entropy.DeltaSmall {
		// abs = tail + (1 << remBits) + 1 with remBits in [1, 8].
		v := abs - 1
		remBits := 0
		for (v >> (remBits + 1)) != 0 {
			remBits++
		}
		if remBits < 1 || remBits > 8 {
			return ErrInvalidDecodeState
		}
		w.WriteLiteral(uint32(remBits-1), 3)
		w.WriteLiteral(uint32(v-(1<<remBits)), remBits)
	}
	if abs != 0 {
		sign := 0
		if delta < 0 {
			sign = 1
		}
		w.WriteBit(sign)
	}
	return nil
}
