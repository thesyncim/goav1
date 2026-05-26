// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package rtp

// AggregationHeader is the first byte of every AV1 RTP payload.
type AggregationHeader struct {
	// ContinuesPrevious is the Z bit.
	ContinuesPrevious bool

	// ContinuesNext is the Y bit.
	ContinuesNext bool

	// ElementCount is the W field. Zero means all OBU elements are length
	// delimited. Values 1..3 mean the packet contains exactly that many OBU
	// elements and the last element omits its length field.
	ElementCount uint8

	// StartsNewCodedVideoSequence is the N bit.
	StartsNewCodedVideoSequence bool
}

func ParseAggregationHeader(src []byte) (AggregationHeader, int, error) {
	if len(src) < 1 {
		return AggregationHeader{}, 0, ErrShortPayload
	}

	b := src[0]
	if b&0x07 != 0 {
		return AggregationHeader{}, 0, ErrReservedBit
	}

	h := AggregationHeader{
		ContinuesPrevious:           b&0x80 != 0,
		ContinuesNext:               b&0x40 != 0,
		ElementCount:                (b >> 4) & 0x03,
		StartsNewCodedVideoSequence: b&0x08 != 0,
	}
	if h.StartsNewCodedVideoSequence && h.ContinuesPrevious {
		return AggregationHeader{}, 0, ErrInvalidAggregationHeader
	}
	return h, 1, nil
}

func (h AggregationHeader) Byte() (byte, error) {
	if h.ElementCount > 3 {
		return 0, ErrInvalidElementCount
	}
	if h.StartsNewCodedVideoSequence && h.ContinuesPrevious {
		return 0, ErrInvalidAggregationHeader
	}

	var b byte
	if h.ContinuesPrevious {
		b |= 0x80
	}
	if h.ContinuesNext {
		b |= 0x40
	}
	b |= h.ElementCount << 4
	if h.StartsNewCodedVideoSequence {
		b |= 0x08
	}
	return b, nil
}

func PutAggregationHeader(dst []byte, h AggregationHeader) (int, error) {
	if len(dst) < 1 {
		return 0, ErrShortBuffer
	}
	b, err := h.Byte()
	if err != nil {
		return 0, err
	}
	dst[0] = b
	return 1, nil
}
