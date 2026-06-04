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
	EventExistingFrame
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

	SequenceHeader      parser.SequenceHeader
	FrameHeader         parser.FrameHeaderPrefix
	FrameSize           parser.FrameSize
	TileInfo            parser.TileInfo
	Quantization        parser.QuantizationParams
	Segmentation        parser.SegmentationParams
	Delta               parser.DeltaParams
	LoopFilter          parser.LoopFilterParams
	CDEF                parser.CDEFParams
	Restoration         parser.RestorationParams
	TransformRef        parser.TransformReferenceParams
	SkipMode            parser.SkipModeParams
	FrameMode           parser.FrameModeParams
	GlobalMotion        parser.GlobalMotionParams
	FilmGrain           parser.FilmGrainParams
	ReferenceOrderHints [parser.InterRefsPerFrame]uint8
	TileGroup           parser.TileGroup
	ExistingFrame       parser.ReferenceFrame

	// Metadata is populated when Kind == EventMetadata and the Metadata OBU
	// payload could be parsed per AV1 spec section 5.8. When parsing fails the
	// stream still emits the EventMetadata event with the parse error captured
	// in MetadataErr so callers can decide whether to abort or skip.
	Metadata    obu.Metadata
	MetadataErr error

	// TileList is populated when Kind == EventTileList and the OBU_TILE_LIST
	// payload could be parsed per AV1 spec section 5.11.1. Entries aliases the
	// scratch buffer returned by Stream.tileListEntries() (or the caller-owned
	// PushUnitWithTileListScratch slice). When parsing fails the event is
	// still emitted with the underlying error in TileListErr so callers can
	// decide whether to abort or skip.
	TileList    parser.TileList
	TileListErr error
}

type frameState struct {
	SequenceHeader      parser.SequenceHeader
	FrameHeader         parser.FrameHeaderPrefix
	FrameSize           parser.FrameSize
	TileInfo            parser.TileInfo
	Quantization        parser.QuantizationParams
	Segmentation        parser.SegmentationParams
	Delta               parser.DeltaParams
	LoopFilter          parser.LoopFilterParams
	CDEF                parser.CDEFParams
	Restoration         parser.RestorationParams
	TransformRef        parser.TransformReferenceParams
	SkipMode            parser.SkipModeParams
	FrameMode           parser.FrameModeParams
	GlobalMotion        parser.GlobalMotionParams
	FilmGrain           parser.FilmGrainParams
	ReferenceOrderHints [parser.InterRefsPerFrame]uint8
}

type Stream struct {
	sequence     parser.SequenceHeader
	haveSequence bool

	haveFrameHeader bool
	tileGroups      uint32
	tileInfo        parser.TileInfo
	nextTile        uint16
	frame           frameState

	references parser.ReferenceState
	rtp        rtp.Depacketizer

	// tileListScratch is reused between EventTileList emissions to avoid
	// per-OBU allocations. It grows on demand and shrinks back to length zero
	// after each parse so the slice header is reset between events.
	tileListScratch []parser.TileListEntry
}

func (s *Stream) Reset() {
	*s = Stream{}
}

func (s *Stream) ResetRTP() {
	s.rtp.Reset()
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

// PushRTPPayloadSize validates one RTP payload against the stream's current RTP
// fragment state and reports the dst bytes and event slots PushRTPPayload would
// need. It does not mutate the stream.
func (s *Stream) PushRTPPayloadSize(used int, payload []byte) (newUsed int, eventCount int, err error) {
	newUsed, eventCount, _, err = s.rtp.PushSize(used, payload)
	return newUsed, eventCount, err
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
		if err := s.PushUnitInto(&events[count], unit, false); err != nil {
			return count, err
		}
		count++
	}
}

func (s *Stream) PushOBUInto(event *Event, raw []byte, newCodedVideoSequence bool) error {
	unit, err := obu.ParseElement(raw)
	if err != nil {
		return err
	}
	return s.PushUnitInto(event, unit, newCodedVideoSequence)
}

func (s *Stream) PushRTPPayload(dst []byte, used int, spans []rtp.OBUSpan, events []Event, payload []byte) (int, int, error) {
	newUsed, spanCount, _, err := s.rtp.Push(dst, used, spans, payload)
	if err != nil {
		return newUsed, 0, err
	}
	if spanCount > len(events) {
		return newUsed, 0, ErrEventBufferTooSmall
	}

	for i := range spanCount {
		span := spans[i]
		end := span.Offset + span.Length
		if span.Offset < 0 || span.Length < 0 || end < span.Offset || end > newUsed {
			return newUsed, i, rtp.ErrShortBuffer
		}
		unit, err := obu.ParseElement(dst[span.Offset:end])
		if err != nil {
			return newUsed, i, err
		}
		if err := s.PushUnitInto(&events[i], unit, span.NewSequence); err != nil {
			return newUsed, i, err
		}
	}
	return newUsed, spanCount, nil
}

func (s *Stream) PushUnitInto(event *Event, unit obu.Unit, newCodedVideoSequence bool) error {
	if event == nil {
		return ErrEventBufferTooSmall
	}
	*event = Event{
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
			return err
		}

		event.Kind = EventSequenceHeader
		event.SequenceHeader = seq
		if event.NewCodedVideoSequence || !s.haveSequence || !sameSequenceExceptOperatingParameters(s.sequence, seq) {
			event.NewCodedVideoSequence = true
			s.clearPendingFrame()
			s.references.Reset()
		} else if s.sequence != seq {
			event.OperatingParametersChanged = true
		}
		s.sequence = seq
		s.haveSequence = true
		return nil

	case obu.TypeTemporalDelimiter:
		event.Kind = EventTemporalDelimiter
		event.NewTemporalUnit = true
		s.clearPendingFrame()
		return nil

	case obu.TypeRedundantFrameHeader:
		if s.haveFrameHeader {
			event.Kind = EventIgnored
			return nil
		}
		event.Kind = EventRedundantFrameHeader
		return s.acceptFrameHeader(event)

	case obu.TypeFrameHeader:
		event.Kind = EventFrameHeader
		return s.acceptFrameHeader(event)

	case obu.TypeFrame:
		if !s.haveSequence {
			return ErrMissingSequenceHeader
		}
		event.SequenceHeader = s.sequence
		frameHeader, err := parser.ParseFrameHeaderPrefix(unit.Payload, s.sequence)
		if err != nil {
			return err
		}
		event.Kind = EventFrame
		event.FrameHeader = frameHeader
		if frameHeader.ShowExistingFrame {
			return parser.ErrInvalidFrameHeader
		}
		if !frameHeader.ShowExistingFrame {
			frameSize, err := parser.ParseFrameSize(unit.Payload, s.sequence, frameHeader, &s.references, event.TemporalID, event.SpatialID)
			if err != nil {
				return err
			}
			tileInfo, err := parser.ParseTileInfo(unit.Payload, s.sequence, frameHeader, frameSize)
			if err != nil {
				return err
			}
			quant, err := parser.ParseQuantizationParams(unit.Payload, s.sequence, tileInfo)
			if err != nil {
				return err
			}
			segmentation, err := parser.ParseSegmentationParams(unit.Payload, frameHeader, quant, s.previousSegmentationData(frameHeader, frameSize))
			if err != nil {
				return err
			}
			delta, err := parser.ParseDeltaParams(unit.Payload, frameSize, quant, segmentation)
			if err != nil {
				return err
			}
			loopFilter, err := parser.ParseLoopFilterParams(unit.Payload, s.sequence, frameHeader, frameSize, segmentation, delta, s.previousLoopFilterDeltas(frameHeader, frameSize))
			if err != nil {
				return err
			}
			cdef, err := parser.ParseCDEFParams(unit.Payload, s.sequence, frameSize, segmentation, loopFilter)
			if err != nil {
				return err
			}
			restoration, err := parser.ParseRestorationParams(unit.Payload, s.sequence, frameSize, segmentation, cdef)
			if err != nil {
				return err
			}
			transformRef, err := parser.ParseTransformReferenceParams(unit.Payload, frameHeader, segmentation, restoration)
			if err != nil {
				return err
			}
			skipMode, err := parser.ParseSkipModeParams(unit.Payload, s.sequence, frameHeader, frameSize, &s.references, transformRef)
			if err != nil {
				return err
			}
			frameMode, err := parser.ParseFrameModeParams(unit.Payload, s.sequence, frameHeader, skipMode)
			if err != nil {
				return err
			}
			globalMotion, err := parser.ParseGlobalMotionParams(unit.Payload, frameHeader, frameSize, tileInfo, &s.references, frameMode)
			if err != nil {
				return err
			}
			filmGrain, err := parser.ParseFilmGrainParams(unit.Payload, s.sequence, frameHeader, frameSize, &s.references, globalMotion)
			if err != nil {
				return err
			}
			tileGroup, err := parser.ParseTileGroupHeader(unit.Payload, tileInfo, int(filmGrain.BitsRead), 0, true)
			if err != nil {
				return err
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
			event.ReferenceOrderHints = referenceOrderHints(frameSize, &s.references)
			event.TileGroup = tileGroup
			s.references.UpdateWithDecodeState(frameHeader, frameSize, segmentation.Data, loopFilter.Deltas, globalMotion, filmGrain)
			s.tileInfo = tileInfo
			s.nextTile = tileGroup.NextTile
			s.storeFrameState(event)
			s.haveFrameHeader = !tileGroup.Final
			s.tileGroups = 1
			if tileGroup.Final {
				s.clearPendingFrame()
			}
			return nil
		}
		s.haveFrameHeader = true
		s.tileGroups = 0
		s.nextTile = 0
		return nil

	case obu.TypeTileGroup:
		if !s.haveFrameHeader {
			return ErrMissingFrameHeader
		}
		tileGroup, err := parser.ParseTileGroupHeader(unit.Payload, s.tileInfo, 0, s.nextTile, false)
		if err != nil {
			return err
		}
		event.Kind = EventTileGroup
		s.applyFrameState(event)
		event.TileGroup = tileGroup
		s.tileGroups++
		s.nextTile = tileGroup.NextTile
		if tileGroup.Final {
			s.clearPendingFrame()
		}
		return nil

	case obu.TypeMetadata:
		event.Kind = EventMetadata
		meta, mErr := obu.ParseMetadata(unit.Payload)
		event.Metadata = meta
		event.MetadataErr = mErr
		return nil

	case obu.TypeTileList:
		event.Kind = EventTileList
		list, err := parser.ParseTileListOBU(unit.Payload, s.tileListScratch[:0])
		if err != nil {
			event.TileListErr = err
			return nil
		}
		// Re-anchor the scratch slice on the buffer the parser returned so a
		// subsequent EventTileList reuses it without re-allocating.
		s.tileListScratch = list.Entries[:len(list.Entries):cap(list.Entries)]
		event.TileList = list
		return nil

	case obu.TypePadding:
		event.Kind = EventPadding
		return nil
	}

	event.Kind = EventReserved
	return nil
}

func (s *Stream) acceptFrameHeader(event *Event) error {
	if !s.haveSequence {
		return ErrMissingSequenceHeader
	}
	event.SequenceHeader = s.sequence
	frameHeader, err := parser.ParseFrameHeaderPrefix(event.Unit.Payload, s.sequence)
	if err != nil {
		return err
	}
	event.FrameHeader = frameHeader
	if frameHeader.ShowExistingFrame {
		return s.acceptExistingFrame(event)
	}

	frameSize, err := parser.ParseFrameSize(event.Unit.Payload, s.sequence, frameHeader, &s.references, event.TemporalID, event.SpatialID)
	if err != nil {
		return err
	}
	tileInfo, err := parser.ParseTileInfo(event.Unit.Payload, s.sequence, frameHeader, frameSize)
	if err != nil {
		return err
	}
	quant, err := parser.ParseQuantizationParams(event.Unit.Payload, s.sequence, tileInfo)
	if err != nil {
		return err
	}
	segmentation, err := parser.ParseSegmentationParams(event.Unit.Payload, frameHeader, quant, s.previousSegmentationData(frameHeader, frameSize))
	if err != nil {
		return err
	}
	delta, err := parser.ParseDeltaParams(event.Unit.Payload, frameSize, quant, segmentation)
	if err != nil {
		return err
	}
	loopFilter, err := parser.ParseLoopFilterParams(event.Unit.Payload, s.sequence, frameHeader, frameSize, segmentation, delta, s.previousLoopFilterDeltas(frameHeader, frameSize))
	if err != nil {
		return err
	}
	cdef, err := parser.ParseCDEFParams(event.Unit.Payload, s.sequence, frameSize, segmentation, loopFilter)
	if err != nil {
		return err
	}
	restoration, err := parser.ParseRestorationParams(event.Unit.Payload, s.sequence, frameSize, segmentation, cdef)
	if err != nil {
		return err
	}
	transformRef, err := parser.ParseTransformReferenceParams(event.Unit.Payload, frameHeader, segmentation, restoration)
	if err != nil {
		return err
	}
	skipMode, err := parser.ParseSkipModeParams(event.Unit.Payload, s.sequence, frameHeader, frameSize, &s.references, transformRef)
	if err != nil {
		return err
	}
	frameMode, err := parser.ParseFrameModeParams(event.Unit.Payload, s.sequence, frameHeader, skipMode)
	if err != nil {
		return err
	}
	globalMotion, err := parser.ParseGlobalMotionParams(event.Unit.Payload, frameHeader, frameSize, tileInfo, &s.references, frameMode)
	if err != nil {
		return err
	}
	filmGrain, err := parser.ParseFilmGrainParams(event.Unit.Payload, s.sequence, frameHeader, frameSize, &s.references, globalMotion)
	if err != nil {
		return err
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
	event.ReferenceOrderHints = referenceOrderHints(frameSize, &s.references)
	s.references.UpdateWithDecodeState(frameHeader, frameSize, segmentation.Data, loopFilter.Deltas, globalMotion, filmGrain)
	s.tileInfo = tileInfo
	s.nextTile = 0
	s.storeFrameState(event)
	s.haveFrameHeader = true
	s.tileGroups = 0
	return nil
}

func (s *Stream) acceptExistingFrame(event *Event) error {
	if event.NewCodedVideoSequence {
		return parser.ErrInvalidFrameHeader
	}
	event.SequenceHeader = s.sequence

	ref := s.references.Frames[event.FrameHeader.ExistingFrameIdx]
	if !ref.Valid {
		return parser.ErrReferenceFrameNeeded
	}
	if s.sequence.FrameIDNumbersPresent && ref.FrameID != event.FrameHeader.FrameID {
		return parser.ErrInvalidFrameHeader
	}
	if !ref.ShowableFrame {
		return parser.ErrInvalidFrameHeader
	}

	event.Kind = EventExistingFrame
	event.ExistingFrame = ref
	event.FrameSize = ref.Size
	event.GlobalMotion = ref.GlobalMotion
	event.FilmGrain = ref.FilmGrain

	s.clearPendingFrame()

	if ref.FrameType == parser.FrameTypeKey {
		ref.ShowableFrame = false
		for i := range parser.RefFrames {
			s.references.Frames[i] = ref
		}
	}
	return nil
}

func (s *Stream) clearPendingFrame() {
	s.haveFrameHeader = false
	s.tileGroups = 0
	s.tileInfo = parser.TileInfo{}
	s.nextTile = 0
	s.frame = frameState{}
}

func (s *Stream) storeFrameState(event *Event) {
	s.frame = frameState{
		SequenceHeader:      event.SequenceHeader,
		FrameHeader:         event.FrameHeader,
		FrameSize:           event.FrameSize,
		TileInfo:            event.TileInfo,
		Quantization:        event.Quantization,
		Segmentation:        event.Segmentation,
		Delta:               event.Delta,
		LoopFilter:          event.LoopFilter,
		CDEF:                event.CDEF,
		Restoration:         event.Restoration,
		TransformRef:        event.TransformRef,
		SkipMode:            event.SkipMode,
		FrameMode:           event.FrameMode,
		GlobalMotion:        event.GlobalMotion,
		FilmGrain:           event.FilmGrain,
		ReferenceOrderHints: event.ReferenceOrderHints,
	}
}

func (s *Stream) applyFrameState(event *Event) {
	event.SequenceHeader = s.frame.SequenceHeader
	event.FrameHeader = s.frame.FrameHeader
	event.FrameSize = s.frame.FrameSize
	event.TileInfo = s.frame.TileInfo
	event.Quantization = s.frame.Quantization
	event.Segmentation = s.frame.Segmentation
	event.Delta = s.frame.Delta
	event.LoopFilter = s.frame.LoopFilter
	event.CDEF = s.frame.CDEF
	event.Restoration = s.frame.Restoration
	event.TransformRef = s.frame.TransformRef
	event.SkipMode = s.frame.SkipMode
	event.FrameMode = s.frame.FrameMode
	event.GlobalMotion = s.frame.GlobalMotion
	event.FilmGrain = s.frame.FilmGrain
	event.ReferenceOrderHints = s.frame.ReferenceOrderHints
}

func referenceOrderHints(size parser.FrameSize, refs *parser.ReferenceState) [parser.InterRefsPerFrame]uint8 {
	var hints [parser.InterRefsPerFrame]uint8
	if refs == nil {
		return hints
	}
	for i := range parser.InterRefsPerFrame {
		slot := size.RefFrameIdx[i]
		if slot < parser.RefFrames {
			hints[i] = refs.Frames[slot].OrderHint
		}
	}
	return hints
}

func (s *Stream) previousSegmentationData(prefix parser.FrameHeaderPrefix, size parser.FrameSize) *parser.SegmentationData {
	if prefix.PrimaryRefFrame == parser.PrimaryRefNone || prefix.PrimaryRefFrame >= parser.InterRefsPerFrame {
		return nil
	}
	slot := size.RefFrameIdx[prefix.PrimaryRefFrame]
	if slot >= parser.RefFrames {
		return nil
	}
	ref := &s.references.Frames[slot]
	if !ref.Valid {
		return nil
	}
	return &ref.Segmentation
}

func (s *Stream) previousLoopFilterDeltas(prefix parser.FrameHeaderPrefix, size parser.FrameSize) *parser.LoopFilterDeltas {
	if prefix.PrimaryRefFrame == parser.PrimaryRefNone || prefix.PrimaryRefFrame >= parser.InterRefsPerFrame {
		return nil
	}
	slot := size.RefFrameIdx[prefix.PrimaryRefFrame]
	if slot >= parser.RefFrames {
		return nil
	}
	ref := &s.references.Frames[slot]
	if !ref.Valid {
		return nil
	}
	return &ref.LoopFilterDeltas
}

func sameSequenceExceptOperatingParameters(a parser.SequenceHeader, b parser.SequenceHeader) bool {
	clearOperatingParameters(&a)
	clearOperatingParameters(&b)
	return a == b
}

func clearOperatingParameters(s *parser.SequenceHeader) {
	for i := range len(s.OperatingPoints) {
		s.OperatingPoints[i].DecoderBufferDelay = 0
		s.OperatingPoints[i].EncoderBufferDelay = 0
		s.OperatingPoints[i].LowDelayMode = false
	}
}
