package goav1

import internaldecoder "github.com/thesyncim/goav1/internal/av1/decoder"

type DecoderStream = internaldecoder.Stream
type DecoderEvent = internaldecoder.Event
type DecoderEventKind = internaldecoder.EventKind

const (
	DecoderEventIgnored              DecoderEventKind = internaldecoder.EventIgnored
	DecoderEventSequenceHeader       DecoderEventKind = internaldecoder.EventSequenceHeader
	DecoderEventTemporalDelimiter    DecoderEventKind = internaldecoder.EventTemporalDelimiter
	DecoderEventFrameHeader          DecoderEventKind = internaldecoder.EventFrameHeader
	DecoderEventRedundantFrameHeader DecoderEventKind = internaldecoder.EventRedundantFrameHeader
	DecoderEventFrame                DecoderEventKind = internaldecoder.EventFrame
	DecoderEventTileGroup            DecoderEventKind = internaldecoder.EventTileGroup
	DecoderEventMetadata             DecoderEventKind = internaldecoder.EventMetadata
	DecoderEventTileList             DecoderEventKind = internaldecoder.EventTileList
	DecoderEventPadding              DecoderEventKind = internaldecoder.EventPadding
	DecoderEventReserved             DecoderEventKind = internaldecoder.EventReserved
)

var (
	ErrDecoderMissingSequenceHeader = internaldecoder.ErrMissingSequenceHeader
	ErrDecoderMissingFrameHeader    = internaldecoder.ErrMissingFrameHeader
	ErrDecoderEventBufferTooSmall   = internaldecoder.ErrEventBufferTooSmall
)
