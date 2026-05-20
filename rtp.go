package goav1

import internalrtp "github.com/thesyncim/goav1/internal/av1/rtp"

type RTPAggregationHeader = internalrtp.AggregationHeader
type RTPElement = internalrtp.Element
type RTPPayloadIterator = internalrtp.Iterator
type RTPFragmentReassembler = internalrtp.FragmentReassembler
type RTPDepacketizer = internalrtp.Depacketizer
type RTPObuSpan = internalrtp.OBUSpan

var (
	ErrRTPShortPayload             = internalrtp.ErrShortPayload
	ErrRTPReservedBit              = internalrtp.ErrReservedBit
	ErrRTPInvalidAggregationHeader = internalrtp.ErrInvalidAggregationHeader
	ErrRTPInvalidElementCount      = internalrtp.ErrInvalidElementCount
	ErrRTPShortBuffer              = internalrtp.ErrShortBuffer
	ErrRTPZeroLengthElement        = internalrtp.ErrZeroLengthElement
	ErrRTPUnexpectedContinuation   = internalrtp.ErrUnexpectedContinuation
	ErrRTPFragmentInterrupted      = internalrtp.ErrFragmentInterrupted
	ErrRTPMTUTooSmall              = internalrtp.ErrMTUTooSmall
)

func ParseRTPAggregationHeader(src []byte) (RTPAggregationHeader, int, error) {
	return internalrtp.ParseAggregationHeader(src)
}

func PutRTPAggregationHeader(dst []byte, header RTPAggregationHeader) (int, error) {
	return internalrtp.PutAggregationHeader(dst, header)
}

func NewRTPPayloadIterator(payload []byte) (RTPPayloadIterator, error) {
	return internalrtp.NewIterator(payload)
}

func PutRTPPayload(dst []byte, header RTPAggregationHeader, elements []RTPElement) (int, error) {
	return internalrtp.PutPayload(dst, header, elements)
}

func PutRTPFragment(dst []byte, obu []byte, offset int, mtu int, startsNewCodedVideoSequence bool) (n int, nextOffset int, more bool, err error) {
	return internalrtp.PutFragment(dst, obu, offset, mtu, startsNewCodedVideoSequence)
}
