package goav1

import (
	"errors"
	"fmt"
)

const layeredDecoderMaxFrameFormats = EncoderWebRTCMaxSpatialLayers + 1

// LayeredDecoder is the high-level receive wrapper for AV1 streams whose
// reference slots span more than one coded frame size, such as WebRTC
// shared-reference SVC. It owns a FrameLayerPool, one frame-work state per
// spatial layer, shared reference slots, and shared frame contexts, then drives
// the same residual/event decode path exposed by the lower-level public APIs.
//
// The *Frame values returned by DecodeNext, DecodeAll, DecodeRTPPayload, and
// DecodeRTPPayloadAfterLoss alias decoder-owned layer-pool surfaces. They
// remain valid only until the next decode call, Reset, or Close unless the AV1
// reference map still retains the surface internally.
//
// A LayeredDecoder is not safe for concurrent use; serialize calls.
type LayeredDecoder struct {
	workerPool *TileWorkerPool

	stream        DecoderStream
	refs          DecoderSurfaceReferences
	states        [EncoderWebRTCMaxSpatialLayers]DecoderFrameWorkState
	frameContexts DecoderSharedFrameContextStore
	stats         DecoderFrameWorkTileResidualStats
	sideData      DecoderFrameWorkSideData
	batch         DecoderFrameWorkBatchResidualRunner
	hasBatch      bool

	layerPool FrameLayerPool
	adapter   DecoderFrameLayerPool

	referenceSurfaces [InterRefsPerFrame]int
	referenceFrames   [InterRefsPerFrame]*Frame
	releases          [RefFrames]int

	scratch    DecoderFrameWorkResidualStreamScratch
	postFilter DecoderFrameWorkReusableSupportedPostFilterRunner

	payloadKind   decoderPayloadKind
	payloadSource decoderPayloadSource
	payloadBuf    []byte
	rtpUsed       int
	sequence      SequenceHeader

	next   int
	closed bool

	visible       []*Frame
	visibleOne    [1]*Frame
	visibleMeta   []LayeredFrame
	shownHeld     []layeredShownSurface
	shownHeldBuf  [2 * (RefFrames + 1)]layeredShownSurface
	outputSurface []layeredShownSurface
	outputSurfBuf [2 * (RefFrames + 1)]layeredShownSurface
}

// LayeredFrame is one visible output from LayeredDecoder with the AV1 layer
// metadata parsed from the OBU stream. RTP dependency-descriptor state, active
// decode-target masks, RTP timestamps, and marker bits are carried outside the
// AV1 RTP payload body and should be read through ParseRTPDependencyDescriptor
// or the caller's RTP stack.
type LayeredFrame struct {
	// Frame is the decoded output. It has the same lifetime as frames returned by
	// DecodeNext: it aliases decoder-owned layer-pool storage.
	Frame *Frame
	// TemporalID and SpatialID are the AV1 OBU extension layer IDs.
	TemporalID uint8
	SpatialID  uint8
	// FrameType is the AV1 frame_type for coded frames or the referenced frame's
	// frame_type for show_existing_frame outputs.
	FrameType FrameType
	// CodedKeyframe reports whether this output was coded as an AV1 key frame.
	// In WebRTC SVC key pictures, enhancement layers may be inter-coded even when
	// the picture is a key picture at the RTP/dependency-descriptor level.
	CodedKeyframe bool
	// ShowExistingFrame reports whether the output came from show_existing_frame.
	ShowExistingFrame bool
	// ShowFrame and ShowableFrame mirror the parsed AV1 frame-header visibility
	// flags for coded frames. ShowFrame is true for show_existing_frame outputs.
	ShowFrame     bool
	ShowableFrame bool
	// FrameSize is the parsed AV1 frame_size metadata for the output.
	FrameSize FrameSize
}

type layeredShownSurface struct {
	id int
}

// NewLayeredDecoder builds a LayeredDecoder from complete, ordered AV1
// low-overhead temporal-unit payloads. Use NewDecoder for ordinary single
// decode chains; use LayeredDecoder when a stream can reference surfaces across
// spatial layers or coded frame sizes.
func NewLayeredDecoder(payloads [][]byte, opts ...Option) (*LayeredDecoder, error) {
	return newLayeredDecoderFromPayloadSourceKind(newSliceDecoderPayloadSource(payloads), decoderPayloadLowOverhead, opts...)
}

// NewLayeredDecoderFromRTPPayloads builds a LayeredDecoder from ordered AV1 RTP
// payload bodies with RTP headers and extensions already removed. It is the
// high-level RTP receive path for shared-reference WebRTC SVC streams. DecodeNext
// consumes one RTP payload per call; fragmented frames can return an empty
// frame slice with ok=true until the packet that completes the frame arrives.
//
// The same decoder can drive live payloads via DecodeRTPPayload and
// DecodeRTPPayloadAfterLoss. The construction payloads size reusable arenas, so
// later live payloads must fit those planned RTP/event capacities.
func NewLayeredDecoderFromRTPPayloads(payloads [][]byte, opts ...Option) (*LayeredDecoder, error) {
	return newLayeredDecoderFromPayloadSourceKind(newSliceDecoderPayloadSource(payloads), decoderPayloadRTP, opts...)
}

func newLayeredDecoderFromPayloadSourceKind(source decoderPayloadSource, kind decoderPayloadKind, opts ...Option) (*LayeredDecoder, error) {
	cfg := resolveConfig(opts)
	if cfg.workers < 1 {
		return nil, fmt.Errorf("goav1: workers must be >= 1, got %d", cfg.workers)
	}
	if source.len() == 0 {
		return nil, errors.New("goav1: no payloads to decode")
	}

	workerPool, err := NewTileWorkerPool(cfg.workers)
	if err != nil {
		return nil, fmt.Errorf("goav1: worker pool: %w", err)
	}

	probeEvents := make([]DecoderEvent, source.len()*16+64)
	probeSpans := make([]TileSpan, MaxTiles)
	probeJobs := make([]TileJob, MaxTiles)
	probeBatches := make([]TileBatch, MaxTiles)
	probePayload := source.makeProbeBuffer()

	var plan DecoderFrameWorkResidualStreamPlan
	switch kind {
	case decoderPayloadLowOverhead:
		plan, err = decoderFrameWorkResidualLowOverheadStreamsPlanFromSource(
			DecoderStream{}, source, cfg.workers, probePayload, probeEvents, probeSpans, probeJobs, probeBatches)
	case decoderPayloadRTP:
		plan, err = decoderFrameWorkResidualRTPPayloadsStreamPlanFromSource(
			DecoderStream{}, source, cfg.workers, probeEvents, probeSpans, probeJobs, probeBatches)
	default:
		err = ErrDecoderInvalidFrameWorkState
	}
	if err != nil {
		workerPool.Close()
		return nil, fmt.Errorf("goav1: stream plan: %w", err)
	}
	if !plan.HasEvent() {
		workerPool.Close()
		return nil, errors.New("goav1: stream plan did not identify a bind event")
	}

	formats, err := decoderLayeredCodedFormatsFromSource(source, kind, 64, probePayload)
	if err != nil {
		workerPool.Close()
		return nil, fmt.Errorf("goav1: layer formats: %w", err)
	}
	if len(formats) == 0 {
		workerPool.Close()
		return nil, errors.New("goav1: stream plan did not identify a frame format")
	}

	var postArena DecoderFrameWorkPostFilterRequestScratchSize
	switch kind {
	case decoderPayloadLowOverhead:
		postArena, err = decoderPostFilterScratchArenaUpperBoundFromSource(source, 64, probeEvents, probePayload)
	case decoderPayloadRTP:
		postArena, err = decoderPostFilterScratchArenaUpperBoundFromRTPSource(source, 64)
	}
	if err != nil {
		workerPool.Close()
		return nil, fmt.Errorf("goav1: postfilter scratch plan: %w", err)
	}

	var arenaSize decoderArenaSize
	if err := arenaSize.addStream(plan.Size); err != nil {
		workerPool.Close()
		return nil, fmt.Errorf("goav1: stream arena size: %w", err)
	}
	if err := arenaSize.addPostFilter(postArena); err != nil {
		workerPool.Close()
		return nil, fmt.Errorf("goav1: postfilter arena size: %w", err)
	}
	if err := arenaSize.addBytes(source.payloadBufferLen()); err != nil {
		workerPool.Close()
		return nil, fmt.Errorf("goav1: payload buffer size: %w", err)
	}
	arena := newDecoderArena(arenaSize)

	d := &LayeredDecoder{
		workerPool:    workerPool,
		payloadKind:   kind,
		payloadSource: source,
		payloadBuf:    arena.takeBytes(source.payloadBufferLen()),
	}
	d.scratch = newDecoderStreamScratch(plan.Size, &arena)
	if cap(d.scratch.Outputs) > 0 {
		d.visible = d.scratch.Outputs[:0]
		d.visibleMeta = make([]LayeredFrame, 0, cap(d.scratch.Outputs))
	} else {
		d.visible = d.visibleOne[:0]
		d.visibleMeta = make([]LayeredFrame, 0, cap(d.visibleOne))
	}
	d.shownHeld = d.shownHeldBuf[:0]
	d.outputSurface = d.outputSurfBuf[:0]
	d.postFilter.size = postArena
	d.postFilter.runner.Scratch = decoderPostFilterScratchFromArena(postArena, &arena)

	d.layerPool, err = BindFrameLayerPool(
		make([]FramePool, layeredDecoderMaxFrameFormats),
		make([]FrameFormat, layeredDecoderMaxFrameFormats),
		make([]bool, layeredDecoderMaxFrameFormats),
		256,
		FrameLayerFactoryFunc(func(format FrameFormat) (FramePool, error) {
			_, backingSize, err := FramePoolRequiredSize(format, RefFrames+1)
			if err != nil {
				return FramePool{}, err
			}
			return BindFramePool(make([]byte, backingSize), format,
				make([]Frame, RefFrames+1),
				make([]int, RefFrames+1),
				make([]bool, RefFrames+1))
		}),
	)
	if err != nil {
		workerPool.Close()
		return nil, fmt.Errorf("goav1: layer pool: %w", err)
	}
	for _, format := range formats {
		if _, err := d.layerPool.SubPool(format); err != nil {
			workerPool.Close()
			return nil, fmt.Errorf("goav1: bind layer format: %w", err)
		}
	}
	d.adapter = NewDecoderFrameLayerPool(&d.layerPool)

	if plan.Size.Event.Runner.Workers != 0 {
		d.batch, err = BindDecoderFrameWorkBatchResidualRunner(plan.Size.Event.Runner, d.scratch.Event.Runner)
		if err != nil {
			workerPool.Close()
			return nil, fmt.Errorf("goav1: bind residual runner: %w", err)
		}
		d.hasBatch = true
	}
	return d, nil
}

func decoderLayeredCodedFormatsFromSource(source decoderPayloadSource, kind decoderPayloadKind, align int, payloadBuf []byte) ([]FrameFormat, error) {
	var formats []FrameFormat
	switch kind {
	case decoderPayloadLowOverhead:
		var stream DecoderStream
		var events []DecoderEvent
		err := source.forEachPayload(payloadBuf, func(payload []byte) error {
			eventCount, err := decoderFrameWorkResidualLowOverheadEventLen(payload)
			if err != nil {
				return err
			}
			if len(events) < eventCount {
				events = make([]DecoderEvent, eventCount)
			}
			sequence, _ := stream.SequenceHeader()
			count, err := stream.PushLowOverhead(payload, events[:eventCount])
			if err != nil {
				return err
			}
			for i := range count {
				event := events[i]
				sequence = decoderFrameWorkResidualEventSequence(sequence, event)
				formats, err = decoderLayeredAppendCodedFormat(formats, sequence, event, align)
				if err != nil {
					return err
				}
			}
			return nil
		})
		return formats, err
	case decoderPayloadRTP:
		err := decoderForEachRTPEventFromSource(source, func(sequence SequenceHeader, event DecoderEvent) error {
			var err error
			formats, err = decoderLayeredAppendCodedFormat(formats, sequence, event, align)
			return err
		})
		return formats, err
	default:
		return nil, ErrDecoderInvalidFrameWorkState
	}
}

func decoderLayeredAppendCodedFormat(formats []FrameFormat, sequence SequenceHeader, event DecoderEvent, align int) ([]FrameFormat, error) {
	if !decoderFrameWorkResidualEventBindCandidate(event) {
		return formats, nil
	}
	format, err := FrameCodedFormatFromHeaders(sequence, event.FrameSize, align)
	if err != nil {
		return formats, err
	}
	for _, existing := range formats {
		if existing == format {
			return formats, nil
		}
	}
	return append(formats, format), nil
}

// DecodeNext decodes the next construction payload. For
// NewLayeredDecoderFromRTPPayloads this is one RTP payload body; fragmented RTP
// payloads may return an empty frame slice with ok=true.
func (d *LayeredDecoder) DecodeNext() (frames []*Frame, ok bool, err error) {
	outputs, ok, err := d.DecodeNextWithMetadata()
	if err != nil || !ok {
		return nil, ok, err
	}
	return d.framesFromLayered(outputs), true, nil
}

// DecodeNextWithMetadata decodes the next construction payload and returns every
// visible frame together with its parsed AV1 layer metadata. For
// NewLayeredDecoderFromRTPPayloads this consumes one RTP payload body per call;
// fragmented RTP frames can return an empty slice with ok=true until the packet
// that completes the frame arrives.
func (d *LayeredDecoder) DecodeNextWithMetadata() (frames []LayeredFrame, ok bool, err error) {
	if d == nil || d.payloadSource.len() == 0 {
		return nil, false, errors.New("goav1: layered decoder is not initialized")
	}
	if d.closed {
		return nil, false, errors.New("goav1: layered decoder closed")
	}
	if d.next >= d.payloadSource.len() {
		return nil, false, nil
	}
	i := d.next
	d.next++
	payload, err := d.payloadSource.payloadAt(i, d.payloadBuf)
	if err != nil {
		return nil, false, fmt.Errorf("goav1: payload %d: %w", i, err)
	}
	frames, err = d.decodePayloadWithMetadata(payload, false)
	if err != nil {
		return nil, false, fmt.Errorf("goav1: payload %d: %w", i, err)
	}
	return frames, true, nil
}

// DecodeRTPPayload decodes one caller-supplied AV1 RTP payload body using an
// RTP decoder constructed by NewLayeredDecoderFromRTPPayloads.
func (d *LayeredDecoder) DecodeRTPPayload(payload []byte) ([]*Frame, error) {
	outputs, err := d.DecodeRTPPayloadWithMetadata(payload)
	if err != nil {
		return nil, err
	}
	return d.framesFromLayered(outputs), nil
}

// DecodeRTPPayloadWithMetadata decodes one caller-supplied AV1 RTP payload body
// using an RTP decoder constructed by NewLayeredDecoderFromRTPPayloads and
// returns visible frames with parsed AV1 layer metadata.
func (d *LayeredDecoder) DecodeRTPPayloadWithMetadata(payload []byte) ([]LayeredFrame, error) {
	if d == nil || d.payloadKind != decoderPayloadRTP {
		return nil, errors.New("goav1: layered decoder is not initialized for RTP payloads")
	}
	return d.decodePayloadWithMetadata(payload, false)
}

// DecodeRTPPayloadAfterLoss clears retained RTP fragment bytes, then decodes
// one caller-supplied AV1 RTP payload body while preserving parser sequence and
// reference state.
func (d *LayeredDecoder) DecodeRTPPayloadAfterLoss(payload []byte) ([]*Frame, error) {
	outputs, err := d.DecodeRTPPayloadAfterLossWithMetadata(payload)
	if err != nil {
		return nil, err
	}
	return d.framesFromLayered(outputs), nil
}

// DecodeRTPPayloadAfterLossWithMetadata clears retained RTP fragment bytes,
// decodes one caller-supplied AV1 RTP payload body, and returns visible frames
// with parsed AV1 layer metadata while preserving parser sequence and reference
// state.
func (d *LayeredDecoder) DecodeRTPPayloadAfterLossWithMetadata(payload []byte) ([]LayeredFrame, error) {
	if d == nil || d.payloadKind != decoderPayloadRTP {
		return nil, errors.New("goav1: layered decoder is not initialized for RTP payloads")
	}
	return d.decodePayloadWithMetadata(payload, true)
}

func (d *LayeredDecoder) decodePayload(payload []byte, afterLoss bool) ([]*Frame, error) {
	outputs, err := d.decodePayloadWithMetadata(payload, afterLoss)
	if err != nil {
		return nil, err
	}
	return d.framesFromLayered(outputs), nil
}

func (d *LayeredDecoder) decodePayloadWithMetadata(payload []byte, afterLoss bool) ([]LayeredFrame, error) {
	if d == nil {
		return nil, errors.New("goav1: layered decoder is not initialized")
	}
	if d.closed {
		return nil, errors.New("goav1: layered decoder closed")
	}
	count, err := d.pushPayload(payload, afterLoss)
	if err != nil {
		return nil, err
	}
	events := d.scratch.Events[:count]
	if err := d.prepareForEvents(events); err != nil {
		return nil, err
	}
	out, err := d.runEvents(events)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (d *LayeredDecoder) prepareForEvents(events []DecoderEvent) error {
	if decoderLayeredEventsStartNewCodedVideoSequence(events) {
		d.releaseShownSurfaces()
		next := d.refs
		count, err := next.ReleaseAll(d.releases[:])
		if err != nil {
			return err
		}
		if count != 0 {
			if err := d.adapter.ReleaseFrameSurfaces(d.releases[:count]); err != nil {
				return err
			}
		}
		d.refs = next
		for i := range d.states {
			d.states[i].Reset()
		}
		d.frameContexts.Reset()
		d.shownHeld = d.shownHeld[:0]
		return nil
	}
	if decoderLayeredEventsStartNewPicture(events) {
		d.releaseShownSurfaces()
	}
	return nil
}

func (d *LayeredDecoder) pushPayload(payload []byte, afterLoss bool) (int, error) {
	switch d.payloadKind {
	case decoderPayloadLowOverhead:
		if afterLoss {
			return 0, ErrDecoderInvalidFrameWorkState
		}
		eventCount, err := decoderFrameWorkResidualLowOverheadEventLen(payload)
		if err != nil {
			return 0, err
		}
		if len(d.scratch.Events) < eventCount {
			return 0, ErrDecoderEventBufferTooSmall
		}
		return d.stream.PushLowOverhead(payload, d.scratch.Events[:eventCount])
	case decoderPayloadRTP:
		if afterLoss {
			d.stream.ResetRTP()
			d.rtpUsed = 0
		}
		plannedUsed, eventCount, err := d.stream.PushRTPPayloadSize(d.rtpUsed, payload)
		if err != nil {
			return 0, err
		}
		if len(d.scratch.RTPBuffer) < plannedUsed || len(d.scratch.RTPSpans) < eventCount {
			return 0, ErrRTPShortBuffer
		}
		if len(d.scratch.Events) < eventCount {
			return 0, ErrDecoderEventBufferTooSmall
		}
		used, count, err := d.stream.PushRTPPayload(
			d.scratch.RTPBuffer[:plannedUsed],
			d.rtpUsed,
			d.scratch.RTPSpans[:eventCount],
			d.scratch.Events[:eventCount],
			payload,
		)
		if err != nil {
			return 0, err
		}
		if used != plannedUsed || count != eventCount {
			return 0, ErrDecoderInvalidFrameWorkState
		}
		d.rtpUsed = used
		if !d.stream.InRTPFragment() {
			d.rtpUsed = 0
		}
		return count, nil
	default:
		return 0, ErrDecoderInvalidFrameWorkState
	}
}

func (d *LayeredDecoder) runEvents(events []DecoderEvent) ([]LayeredFrame, error) {
	out := d.visibleMeta[:0]
	d.outputSurface = d.outputSurfBuf[:0]
	for i := range events {
		event := events[i]
		d.sequence = decoderFrameWorkResidualEventSequence(d.sequence, event)
		state, err := d.stateForEvent(event)
		if err != nil {
			return out, err
		}
		framePool, err := d.framePoolForEvent(d.sequence, event)
		if err != nil {
			return out, err
		}
		sideData, err := d.sideDataForEvent(d.sequence, event)
		if err != nil {
			return out, err
		}
		batchRunner := d.batchRunnerForEvent(event)
		globalSurface := func(local int) int {
			if framePool == nil {
				return -1
			}
			return DecoderLayerPoolGlobalSurfaceID(&d.layerPool, framePool, local)
		}
		result, err := RunDecoderFrameWorkEventWithResidualRunner(DecoderFrameWorkResidualEventRequest{
			State:             state,
			Refs:              &d.refs,
			FramePool:         framePool,
			Sequence:          d.sequence,
			Event:             event,
			Align:             64,
			ReferenceSurfaces: d.referenceSurfaces[:],
			ReferenceFrames:   d.referenceFrames[:],
			Workers:           d.workerPool.WorkerCount(),
			Spans:             d.scratch.Event.Spans,
			Jobs:              d.scratch.Event.Jobs,
			Batches:           d.scratch.Event.Batches,
			Releases:          d.releases[:],
			WorkerPool:        d.workerPool,
			Runner:            batchRunner,
			SideData:          sideData,
			PostRunner:        &d.postFilter,
			Stats:             &d.stats,
			External: DecoderFrameWorkExternalReferenceRuntime{
				Provider:      d.adapter,
				GlobalSurface: globalSurface,
				Releaser:      d.adapter,
				FrameContexts: &d.frameContexts,
			},
		})
		if err != nil {
			return out, err
		}
		if !DecoderEventOutputsFrame(event) {
			continue
		}
		if result.Output == nil {
			return out, ErrDecoderInvalidFrameWorkState
		}
		surface, err := d.outputSurfaceForResult(framePool, result)
		if err != nil {
			return out, err
		}
		out = append(out, layeredFrameFromEvent(result.Output, event))
		d.outputSurface = append(d.outputSurface, layeredShownSurface{id: surface})
	}
	d.trackShownSurfaces(d.outputSurface)
	d.visibleMeta = out
	return out, nil
}

func layeredFrameFromEvent(frame *Frame, event DecoderEvent) LayeredFrame {
	frameType := event.FrameHeader.FrameType
	showable := event.FrameHeader.ShowableFrame
	if event.FrameHeader.ShowExistingFrame {
		frameType = event.ExistingFrame.FrameType
		showable = event.ExistingFrame.ShowableFrame
	}
	return LayeredFrame{
		Frame:             frame,
		TemporalID:        event.TemporalID,
		SpatialID:         event.SpatialID,
		FrameType:         frameType,
		CodedKeyframe:     !event.FrameHeader.ShowExistingFrame && event.FrameHeader.FrameType == FrameTypeKey,
		ShowExistingFrame: event.FrameHeader.ShowExistingFrame,
		ShowFrame:         event.FrameHeader.ShowFrame || event.FrameHeader.ShowExistingFrame,
		ShowableFrame:     showable,
		FrameSize:         event.FrameSize,
	}
}

func (d *LayeredDecoder) framesFromLayered(outputs []LayeredFrame) []*Frame {
	out := d.visible[:0]
	for i := range outputs {
		out = append(out, outputs[i].Frame)
	}
	d.visible = out
	return out
}

func (d *LayeredDecoder) stateForEvent(event DecoderEvent) (*DecoderFrameWorkState, error) {
	if event.SpatialID >= EncoderWebRTCMaxSpatialLayers {
		return nil, ErrDecoderInvalidFrameWorkState
	}
	return &d.states[event.SpatialID], nil
}

func (d *LayeredDecoder) framePoolForEvent(sequence SequenceHeader, event DecoderEvent) (*FramePool, error) {
	switch event.Kind {
	case DecoderEventFrameHeader, DecoderEventFrame, DecoderEventTileGroup:
		format, err := FrameCodedFormatFromHeaders(sequence, event.FrameSize, 64)
		if err != nil {
			return nil, err
		}
		return d.layerPool.SubPool(format)
	default:
		return nil, nil
	}
}

func (d *LayeredDecoder) sideDataForEvent(sequence SequenceHeader, event DecoderEvent) (*DecoderFrameWorkSideData, error) {
	if !decoderFrameWorkResidualEventNeedsSideData(event) {
		return nil, nil
	}
	side, err := BindDecoderFrameWorkResidualEventSideData(sequence, event, d.scratch.SideData)
	if err != nil {
		return nil, err
	}
	d.sideData = side
	return &d.sideData, nil
}

func (d *LayeredDecoder) batchRunnerForEvent(event DecoderEvent) *DecoderFrameWorkBatchResidualRunner {
	if !decoderFrameWorkResidualEventHasTilePayload(event) || !d.hasBatch {
		return nil
	}
	return &d.batch
}

func (d *LayeredDecoder) outputSurfaceForResult(framePool *FramePool, result DecoderFrameWorkEventResult) (int, error) {
	switch result.Step.Kind {
	case DecoderFrameWorkStepShowExisting:
		if result.Step.ShowExisting.Surface < 0 {
			return -1, ErrDecoderInvalidSurfaceReference
		}
		return result.Step.ShowExisting.Surface, nil
	case DecoderFrameWorkStepBegin:
		return DecoderLayerPoolGlobalSurfaceID(&d.layerPool, framePool, result.Step.Begin.Surface), nil
	case DecoderFrameWorkStepTile:
		return DecoderLayerPoolGlobalSurfaceID(&d.layerPool, framePool, result.Step.Tile.Surface), nil
	default:
		return -1, ErrDecoderInvalidFrameWorkState
	}
}

func (d *LayeredDecoder) trackShownSurfaces(surfaces []layeredShownSurface) {
	for _, surface := range surfaces {
		if surface.id < 0 {
			continue
		}
		dup := false
		for _, held := range d.shownHeld {
			if held.id == surface.id {
				dup = true
				break
			}
		}
		if !dup && len(d.shownHeld) < cap(d.shownHeldBuf) {
			d.shownHeld = append(d.shownHeld, surface)
		}
	}
}

func decoderLayeredEventsStartNewPicture(events []DecoderEvent) bool {
	for i := range events {
		if events[i].NewTemporalUnit || events[i].NewCodedVideoSequence {
			return true
		}
		if decoderFrameWorkResidualEventBindCandidate(events[i]) && events[i].SpatialID == 0 {
			return true
		}
	}
	return false
}

func decoderLayeredEventsStartNewCodedVideoSequence(events []DecoderEvent) bool {
	for i := range events {
		if events[i].NewCodedVideoSequence {
			return true
		}
	}
	return false
}

func (d *LayeredDecoder) releaseShownSurfaces() {
	for _, surface := range d.shownHeld {
		held := false
		for slot := 0; slot < RefFrames; slot++ {
			ref, ok := d.refs.ReferenceSlot(slot)
			if ok && ref == surface.id {
				held = true
				break
			}
		}
		if !held {
			_ = d.layerPool.Release(surface.id)
		}
	}
	d.shownHeld = d.shownHeld[:0]
}

// DecodeAll decodes every remaining construction payload and returns all
// visible frames in display order. As with Decoder.DecodeAll, returned frames
// alias reused decoder-owned surfaces; callers that need to retain pixels
// across payloads should copy them out per DecodeNext call.
func (d *LayeredDecoder) DecodeAll() ([]*Frame, error) {
	outputs, err := d.DecodeAllWithMetadata()
	if err != nil {
		return nil, err
	}
	return d.framesFromLayered(outputs), nil
}

// DecodeAllWithMetadata decodes every remaining construction payload and
// returns all visible frames in display order with parsed AV1 layer metadata. As
// with DecodeAll, returned frames alias reused decoder-owned surfaces; callers
// that need to retain pixels across payloads should copy them out per DecodeNext
// call.
func (d *LayeredDecoder) DecodeAllWithMetadata() ([]LayeredFrame, error) {
	if d == nil || d.payloadSource.len() == 0 {
		return nil, errors.New("goav1: layered decoder is not initialized")
	}
	if d.closed {
		return nil, errors.New("goav1: layered decoder closed")
	}
	var frames []LayeredFrame
	for {
		got, ok, err := d.DecodeNextWithMetadata()
		if err != nil {
			return frames, err
		}
		if !ok {
			break
		}
		frames = append(frames, got...)
	}
	return frames, nil
}

// Reset rewinds the LayeredDecoder to its initial state so the same bound
// payloads can be decoded again from the start without reallocating scratch.
func (d *LayeredDecoder) Reset() error {
	if d == nil || d.payloadSource.len() == 0 {
		return errors.New("goav1: layered decoder is not initialized")
	}
	if d.closed {
		return errors.New("goav1: layered decoder closed")
	}
	d.layerPool.Reset()
	d.refs.Reset()
	for i := range d.states {
		d.states[i].Reset()
	}
	d.frameContexts.Reset()
	d.stats = DecoderFrameWorkTileResidualStats{}
	d.stream.Reset()
	d.rtpUsed = 0
	d.sequence = SequenceHeader{}
	d.next = 0
	d.visible = d.visible[:0]
	d.visibleMeta = d.visibleMeta[:0]
	d.shownHeld = d.shownHeld[:0]
	d.outputSurface = d.outputSurface[:0]
	return nil
}

// Close releases the worker goroutine pool owned by the LayeredDecoder. After
// Close, the decoder must not be used. Close is idempotent.
func (d *LayeredDecoder) Close() {
	if d == nil || d.closed {
		return
	}
	d.closed = true
	if d.workerPool != nil {
		d.workerPool.Close()
		d.workerPool = nil
	}
}
