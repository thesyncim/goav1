package webrtcav1

import internalrtp "github.com/thesyncim/goav1/internal/av1/rtp"

type AggregationHeader = internalrtp.AggregationHeader
type Element = internalrtp.Element
type PayloadIterator = internalrtp.Iterator
type FragmentReassembler = internalrtp.FragmentReassembler

var (
	ErrShortPayload             = internalrtp.ErrShortPayload
	ErrReservedBit              = internalrtp.ErrReservedBit
	ErrInvalidAggregationHeader = internalrtp.ErrInvalidAggregationHeader
	ErrInvalidElementCount      = internalrtp.ErrInvalidElementCount
	ErrShortBuffer              = internalrtp.ErrShortBuffer
	ErrZeroLengthElement        = internalrtp.ErrZeroLengthElement
	ErrUnexpectedContinuation   = internalrtp.ErrUnexpectedContinuation
	ErrFragmentInterrupted      = internalrtp.ErrFragmentInterrupted
	ErrMTUTooSmall              = internalrtp.ErrMTUTooSmall
)

func ParseAggregationHeader(src []byte) (AggregationHeader, int, error) {
	return internalrtp.ParseAggregationHeader(src)
}

func PutAggregationHeader(dst []byte, header AggregationHeader) (int, error) {
	return internalrtp.PutAggregationHeader(dst, header)
}

func NewPayloadIterator(payload []byte) (PayloadIterator, error) {
	return internalrtp.NewIterator(payload)
}

func PutPayload(dst []byte, header AggregationHeader, elements []Element) (int, error) {
	return internalrtp.PutPayload(dst, header, elements)
}

func PutFragment(dst []byte, obu []byte, offset int, mtu int, startsNewCodedVideoSequence bool) (n int, nextOffset int, more bool, err error) {
	return internalrtp.PutFragment(dst, obu, offset, mtu, startsNewCodedVideoSequence)
}
