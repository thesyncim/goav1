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
	FrameHeader    parser.FrameHeaderPrefix
	FrameSize      parser.FrameSize
	TileInfo       parser.TileInfo
	Quantization   parser.QuantizationParams
	Segmentation   parser.SegmentationParams
	Delta          parser.DeltaParams
	LoopFilter     parser.LoopFilterParams
	CDEF           parser.CDEFParams
	Restoration    parser.RestorationParams
	TransformRef   parser.TransformReferenceParams
	SkipMode       parser.SkipModeParams
	FrameMode      parser.FrameModeParams
	GlobalMotion   parser.GlobalMotionParams
	FilmGrain      parser.FilmGrainParams
}

type Stream struct {
	sequence     parser.SequenceHeader
	haveSequence bool

	haveFrameHeader bool
	tileGroups      uint32

	references parser.ReferenceState
	rtp        rtp.Depacketizer
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
			s.references.Reset()
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
		frameHeader, err := parser.ParseFrameHeaderPrefix(unit.Payload, s.sequence)
		if err != nil {
			return Event{}, err
		}
		event.Kind = EventFrame
		event.FrameHeader = frameHeader
		if !frameHeader.ShowExistingFrame {
			frameSize, err := parser.ParseFrameSize(unit.Payload, s.sequence, frameHeader, &s.references, event.TemporalID, event.SpatialID)
			if err != nil {
				return Event{}, err
			}
			tileInfo, err := parser.ParseTileInfo(unit.Payload, s.sequence, frameHeader, frameSize)
			if err != nil {
				return Event{}, err
			}
			quant, err := parser.ParseQuantizationParams(unit.Payload, s.sequence, tileInfo)
			if err != nil {
				return Event{}, err
			}
			segmentation, err := parser.ParseSegmentationParams(unit.Payload, frameHeader, quant, nil)
			if err != nil {
				return Event{}, err
			}
			delta, err := parser.ParseDeltaParams(unit.Payload, frameSize, quant, segmentation)
			if err != nil {
				return Event{}, err
			}
			loopFilter, err := parser.ParseLoopFilterParams(unit.Payload, s.sequence, frameHeader, frameSize, segmentation, delta, nil)
			if err != nil {
				return Event{}, err
			}
			cdef, err := parser.ParseCDEFParams(unit.Payload, s.sequence, frameSize, segmentation, loopFilter)
			if err != nil {
				return Event{}, err
			}
			restoration, err := parser.ParseRestorationParams(unit.Payload, s.sequence, frameSize, segmentation, cdef)
			if err != nil {
				return Event{}, err
			}
			transformRef, err := parser.ParseTransformReferenceParams(unit.Payload, frameHeader, segmentation, restoration)
			if err != nil {
				return Event{}, err
			}
			skipMode, err := parser.ParseSkipModeParams(unit.Payload, s.sequence, frameHeader, frameSize, &s.references, transformRef)
			if err != nil {
				return Event{}, err
			}
			frameMode, err := parser.ParseFrameModeParams(unit.Payload, s.sequence, frameHeader, skipMode)
			if err != nil {
				return Event{}, err
			}
			globalMotion, err := parser.ParseGlobalMotionParams(unit.Payload, frameHeader, frameSize, tileInfo, &s.references, frameMode)
			if err != nil {
				return Event{}, err
			}
			filmGrain, err := parser.ParseFilmGrainParams(unit.Payload, s.sequence, frameHeader, frameSize, &s.references, globalMotion)
			if err != nil {
				return Event{}, err
			}
			event.FrameSize = frameSize
			event.TileInfo = tileInfo
			event.Quantization = quant
			event.Segmentation = segmentation
			event.Delta = delta
			event.LoopFilter = loopFilter
			event.CDEF = cdef
			event.Restoration = restoration
			event.TransformRef = transformRef
			event.SkipMode = skipMode
			event.FrameMode = frameMode
			event.GlobalMotion = globalMotion
			event.FilmGrain = filmGrain
			s.references.UpdateWithFrameState(frameHeader, frameSize, globalMotion, filmGrain)
		}
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
	frameHeader, err := parser.ParseFrameHeaderPrefix(event.Unit.Payload, s.sequence)
	if err != nil {
		return Event{}, err
	}
	event.FrameHeader = frameHeader
	if !frameHeader.ShowExistingFrame {
		frameSize, err := parser.ParseFrameSize(event.Unit.Payload, s.sequence, frameHeader, &s.references, event.TemporalID, event.SpatialID)
		if err != nil {
			return Event{}, err
		}
		tileInfo, err := parser.ParseTileInfo(event.Unit.Payload, s.sequence, frameHeader, frameSize)
		if err != nil {
			return Event{}, err
		}
		quant, err := parser.ParseQuantizationParams(event.Unit.Payload, s.sequence, tileInfo)
		if err != nil {
			return Event{}, err
		}
		segmentation, err := parser.ParseSegmentationParams(event.Unit.Payload, frameHeader, quant, nil)
		if err != nil {
			return Event{}, err
		}
		delta, err := parser.ParseDeltaParams(event.Unit.Payload, frameSize, quant, segmentation)
		if err != nil {
			return Event{}, err
		}
		loopFilter, err := parser.ParseLoopFilterParams(event.Unit.Payload, s.sequence, frameHeader, frameSize, segmentation, delta, nil)
		if err != nil {
			return Event{}, err
		}
		cdef, err := parser.ParseCDEFParams(event.Unit.Payload, s.sequence, frameSize, segmentation, loopFilter)
		if err != nil {
			return Event{}, err
		}
		restoration, err := parser.ParseRestorationParams(event.Unit.Payload, s.sequence, frameSize, segmentation, cdef)
		if err != nil {
			return Event{}, err
		}
		transformRef, err := parser.ParseTransformReferenceParams(event.Unit.Payload, frameHeader, segmentation, restoration)
		if err != nil {
			return Event{}, err
		}
		skipMode, err := parser.ParseSkipModeParams(event.Unit.Payload, s.sequence, frameHeader, frameSize, &s.references, transformRef)
		if err != nil {
			return Event{}, err
		}
		frameMode, err := parser.ParseFrameModeParams(event.Unit.Payload, s.sequence, frameHeader, skipMode)
		if err != nil {
			return Event{}, err
		}
		globalMotion, err := parser.ParseGlobalMotionParams(event.Unit.Payload, frameHeader, frameSize, tileInfo, &s.references, frameMode)
		if err != nil {
			return Event{}, err
		}
		filmGrain, err := parser.ParseFilmGrainParams(event.Unit.Payload, s.sequence, frameHeader, frameSize, &s.references, globalMotion)
		if err != nil {
			return Event{}, err
		}
		event.FrameSize = frameSize
		event.TileInfo = tileInfo
		event.Quantization = quant
		event.Segmentation = segmentation
		event.Delta = delta
		event.LoopFilter = loopFilter
		event.CDEF = cdef
		event.Restoration = restoration
		event.TransformRef = transformRef
		event.SkipMode = skipMode
		event.FrameMode = frameMode
		event.GlobalMotion = globalMotion
		event.FilmGrain = filmGrain
		s.references.UpdateWithFrameState(frameHeader, frameSize, globalMotion, filmGrain)
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
