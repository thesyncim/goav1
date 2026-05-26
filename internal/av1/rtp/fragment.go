package rtp

// PutFragment writes one AV1 RTP payload containing a fragment of obu.
//
// offset is the starting byte inside obu. The returned nextOffset should be fed
// back into the next call while more is true. The output uses W=1 because the
// payload contains exactly one OBU element and can infer its length.
func PutFragment(dst []byte, obu []byte, offset int, mtu int, startsNewCodedVideoSequence bool) (n int, nextOffset int, more bool, err error) {
	if mtu < 2 || len(dst) < mtu {
		return 0, offset, false, ErrMTUTooSmall
	}
	if offset < 0 || offset >= len(obu) {
		return 0, offset, false, ErrShortPayload
	}

	maxData := mtu - 1
	remaining := len(obu) - offset
	payloadLen := min(remaining, maxData)

	nextOffset = offset + payloadLen
	more = nextOffset < len(obu)
	header := AggregationHeader{
		ContinuesPrevious:           offset > 0,
		ContinuesNext:               more,
		ElementCount:                1,
		StartsNewCodedVideoSequence: startsNewCodedVideoSequence && offset == 0,
	}

	if _, err := PutAggregationHeader(dst, header); err != nil {
		return 0, offset, false, err
	}
	copy(dst[1:], obu[offset:nextOffset])
	return 1 + payloadLen, nextOffset, more, nil
}
