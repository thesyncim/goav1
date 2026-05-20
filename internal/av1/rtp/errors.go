package rtp

import "errors"

var (
	ErrShortPayload             = errors.New("rtpav1: short payload")
	ErrReservedBit              = errors.New("rtpav1: reserved bit set")
	ErrInvalidAggregationHeader = errors.New("rtpav1: invalid aggregation header")
	ErrInvalidElementCount      = errors.New("rtpav1: invalid element count")
	ErrShortBuffer              = errors.New("rtpav1: short buffer")
	ErrZeroLengthElement        = errors.New("rtpav1: zero-length OBU element")
	ErrUnexpectedContinuation   = errors.New("rtpav1: unexpected OBU continuation")
	ErrFragmentInterrupted      = errors.New("rtpav1: fragmented OBU interrupted")
	ErrMTUTooSmall              = errors.New("rtpav1: MTU too small")
)
