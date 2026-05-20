package decoder

import (
	"github.com/thesyncim/goav1/internal/av1/obu"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/rtp"
)

type EventKind uint8

const (
	EventIgnored EventKind = iota
	EventSequenceHeader
	EventTemporalDelimiter
	EventFrameHeader
	EventRedundantFrameHeader
	EventFrame
	EventTileGroup
	EventMetadata
	EventTileList
	EventPadding
	EventReserved
)

type Event struct {
	Kind EventKind
	Type obu.Type
	Unit obu.Unit

	TemporalID uint8
	SpatialID  uint8

	NewCodedVideoSequence      bool
	NewTemporalUnit            bool
	OperatingParametersChanged bool

	SequenceHeader parser.SequenceHeader
}

type Stream struct {
	sequence     parser.SequenceHeader
	haveSequence bool

	haveFrameHeader bool
	tileGroups      uint32

	rtp rtp.Depacketizer
}

func (s *Stream) Reset() {
	*s = Stream{}
}

func (s *Stream) HasSequenceHeader() bool {
	return s.haveSequence
}

func (s *Stream) SequenceHeader() (parser.SequenceHeader, bool) {
	return s.sequence, s.haveSequence
}

func (s *Stream) InRTPFragment() bool {
	return s.rtp.InFragment()
}

func (s *Stream) PushLowOverhead(src []byte, events []Event) (int, error) {
	it := obu.NewLowOverheadIterator(src)
	count := 0
	for {
		unit, ok, err := it.Next()
		if err != nil {
			return count, err
		}
		if !ok {
			return count, nil
		}
		if count >= len(events) {
			return count, ErrEventBufferTooSmall
		}
		events[count], err = s.PushUnit(unit, false)
		if err != nil {
			return count, err
		}
		count++
	}
}

func (s *Stream) PushOBU(raw []byte, newCodedVideoSequence bool) (Event, error) {
	unit, err := obu.ParseElement(raw)
	if err != nil {
		return Event{}, err
	}
	return s.PushUnit(unit, newCodedVideoSequence)
}

func (s *Stream) PushRTPPayload(dst []byte, used int, spans []rtp.OBUSpan, events []Event, payload []byte) (int, int, error) {
	newUsed, spanCount, _, err := s.rtp.Push(dst, used, spans, payload)
	if err != nil {
		return newUsed, 0, err
	}
	if spanCount > len(events) {
		return newUsed, 0, ErrEventBufferTooSmall
	}

	for i := 0; i < spanCount; i++ {
		span := spans[i]
		end := span.Offset + span.Length
		if span.Offset < 0 || span.Length < 0 || end < span.Offset || end > newUsed {
			return newUsed, i, rtp.ErrShortBuffer
		}
		unit, err := obu.ParseElement(dst[span.Offset:end])
		if err != nil {
			return newUsed, i, err
		}
		events[i], err = s.PushUnit(unit, span.NewSequence)
		if err != nil {
			return newUsed, i, err
		}
	}
	return newUsed, spanCount, nil
}

func (s *Stream) PushUnit(unit obu.Unit, newCodedVideoSequence bool) (Event, error) {
	event := Event{
		Type:                  unit.Header.Type,
		Unit:                  unit,
		TemporalID:            unit.Header.TemporalID,
		SpatialID:             unit.Header.SpatialID,
		NewCodedVideoSequence: newCodedVideoSequence,
	}

	switch unit.Header.Type {
	case obu.TypeSequenceHeader:
		seq, err := parser.ParseSequenceHeader(unit.Payload)
		if err != nil {
			return Event{}, err
		}

		event.Kind = EventSequenceHeader
		event.SequenceHeader = seq
		if !s.haveSequence || !sameSequenceExceptOperatingParameters(s.sequence, seq) {
			event.NewCodedVideoSequence = true
			s.haveFrameHeader = false
			s.tileGroups = 0
		} else if s.sequence != seq {
			event.OperatingParametersChanged = true
		}
		s.sequence = seq
		s.haveSequence = true
		return event, nil

	case obu.TypeTemporalDelimiter:
		event.Kind = EventTemporalDelimiter
		event.NewTemporalUnit = true
		s.haveFrameHeader = false
		s.tileGroups = 0
		return event, nil

	case obu.TypeRedundantFrameHeader:
		if s.haveFrameHeader {
			event.Kind = EventIgnored
			return event, nil
		}
		event.Kind = EventRedundantFrameHeader
		return s.acceptFrameHeader(event)

	case obu.TypeFrameHeader:
		event.Kind = EventFrameHeader
		return s.acceptFrameHeader(event)

	case obu.TypeFrame:
		if !s.haveSequence {
			return Event{}, ErrMissingSequenceHeader
		}
		event.Kind = EventFrame
		s.haveFrameHeader = true
		s.tileGroups = 1
		return event, nil

	case obu.TypeTileGroup:
		if !s.haveFrameHeader {
			return Event{}, ErrMissingFrameHeader
		}
		event.Kind = EventTileGroup
		s.tileGroups++
		return event, nil

	case obu.TypeMetadata:
		event.Kind = EventMetadata
		return event, nil

	case obu.TypeTileList:
		event.Kind = EventTileList
		return event, nil

	case obu.TypePadding:
		event.Kind = EventPadding
		return event, nil
	}

	event.Kind = EventReserved
	return event, nil
}

func (s *Stream) acceptFrameHeader(event Event) (Event, error) {
	if !s.haveSequence {
		return Event{}, ErrMissingSequenceHeader
	}
	s.haveFrameHeader = true
	s.tileGroups = 0
	return event, nil
}

func sameSequenceExceptOperatingParameters(a parser.SequenceHeader, b parser.SequenceHeader) bool {
	clearOperatingParameters(&a)
	clearOperatingParameters(&b)
	return a == b
}

func clearOperatingParameters(s *parser.SequenceHeader) {
	for i := 0; i < len(s.OperatingPoints); i++ {
		s.OperatingPoints[i].DecoderBufferDelay = 0
		s.OperatingPoints[i].EncoderBufferDelay = 0
		s.OperatingPoints[i].LowDelayMode = false
	}
}
