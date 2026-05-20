package goav1

import internalparser "github.com/thesyncim/goav1/internal/av1/parser"

type SequenceHeader = internalparser.SequenceHeader
type TimingInfo = internalparser.TimingInfo
type DecoderModelInfo = internalparser.DecoderModelInfo
type OperatingPoint = internalparser.OperatingPoint
type ColorConfig = internalparser.ColorConfig

var ErrInvalidSequenceHeader = internalparser.ErrInvalidSequenceHeader

func ParseSequenceHeader(payload []byte) (SequenceHeader, error) {
	return internalparser.ParseSequenceHeader(payload)
}
