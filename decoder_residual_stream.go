package goav1

// DecoderFrameWorkResidualStreamRunner connects the byte-stream/RTP event
// parser to the residual event runner using caller-owned buffers.
type DecoderFrameWorkResidualStreamRunner struct {
	Stream      *DecoderStream
	EventRunner DecoderFrameWorkResidualEventRunner

	Events          []DecoderEvent
	SideDataScratch DecoderFrameWorkSideDataScratch

	RTPBuffer []byte
	RTPSpans  []RTPObuSpan
	RTPUsed   int
}

// DecoderFrameWorkResidualStreamResult reports parser and decode work completed
// by a stream-runner call.
type DecoderFrameWorkResidualStreamResult struct {
	EventCount int
	RTPUsed    int
	Run        DecoderFrameWorkResidualEventsResult
}

// DecoderFrameWorkResidualStreamScratchSize reports caller-owned stream parser
// and residual-event scratch needed for one stream-runner call.
type DecoderFrameWorkResidualStreamScratchSize struct {
	Events    int
	RTPBuffer int
	RTPSpans  int
	Event     DecoderFrameWorkResidualEventScratchSize
}

// DecoderFrameWorkResidualStreamScratch carries typed caller-owned parser and
// event side-data arenas for BindDecoderFrameWorkResidualStreamRunner.
type DecoderFrameWorkResidualStreamScratch struct {
	Events   []DecoderEvent
	SideData DecoderFrameWorkSideDataScratch

	RTPBuffer []byte
	RTPSpans  []RTPObuSpan
}

// Max returns per-arena maximum lengths for reusable residual stream scratch.
func (s DecoderFrameWorkResidualStreamScratchSize) Max(other DecoderFrameWorkResidualStreamScratchSize) DecoderFrameWorkResidualStreamScratchSize {
	return DecoderFrameWorkResidualStreamScratchSize{
		Events:    max(s.Events, other.Events),
		RTPBuffer: max(s.RTPBuffer, other.RTPBuffer),
		RTPSpans:  max(s.RTPSpans, other.RTPSpans),
		Event:     s.Event.Max(other.Event),
	}
}

// BindDecoderFrameWorkResidualStreamRunner binds a stream runner from
// caller-owned parser and side-data scratch. The eventRunner is owned by the
// caller and is copied into the returned stream runner.
func BindDecoderFrameWorkResidualStreamRunner(size DecoderFrameWorkResidualStreamScratchSize, stream *DecoderStream, eventRunner DecoderFrameWorkResidualEventRunner, scratch DecoderFrameWorkResidualStreamScratch) (DecoderFrameWorkResidualStreamRunner, error) {
	if stream == nil {
		return DecoderFrameWorkResidualStreamRunner{}, ErrDecoderInvalidFrameWorkState
	}
	if decoderFrameWorkResidualScratchTooShort(scratch.Events, size.Events) {
		return DecoderFrameWorkResidualStreamRunner{}, ErrDecoderEventBufferTooSmall
	}
	if decoderFrameWorkResidualScratchTooShort(scratch.RTPBuffer, size.RTPBuffer) ||
		decoderFrameWorkResidualScratchTooShort(scratch.RTPSpans, size.RTPSpans) {
		return DecoderFrameWorkResidualStreamRunner{}, ErrRTPShortBuffer
	}
	if decoderFrameWorkResidualStreamSideDataScratchTooShort(scratch.SideData, size.Event.SideData) {
		return DecoderFrameWorkResidualStreamRunner{}, ErrFrameShortBuffer
	}
	return DecoderFrameWorkResidualStreamRunner{
		Stream:      stream,
		EventRunner: eventRunner,
		Events:      scratch.Events[:size.Events],
		SideDataScratch: DecoderFrameWorkSideDataScratch{
			CDEFIndexMap:             scratch.SideData.CDEFIndexMap[:size.Event.SideData.CDEFIndexMap],
			CDEFReadMap:              scratch.SideData.CDEFReadMap[:size.Event.SideData.CDEFReadMap],
			LoopFilterMap:            scratch.SideData.LoopFilterMap[:size.Event.SideData.LoopFilterMap],
			RestorationRecords:       scratch.SideData.RestorationRecords[:size.Event.SideData.RestorationRecords],
			RestorationBoundaryAbove: scratch.SideData.RestorationBoundaryAbove[:size.Event.SideData.RestorationBoundaryAbove],
			RestorationBoundaryBelow: scratch.SideData.RestorationBoundaryBelow[:size.Event.SideData.RestorationBoundaryBelow],
		},
		RTPBuffer: scratch.RTPBuffer[:size.RTPBuffer],
		RTPSpans:  scratch.RTPSpans[:size.RTPSpans],
	}, nil
}

// DecoderFrameWorkResidualLowOverheadStreamScratchLen parses a low-overhead
// OBU buffer through a copy of stream and reports caller-owned scratch needed to
// run its residual events. stream is passed by value so the live decoder stream
// is not mutated by sizing.
func DecoderFrameWorkResidualLowOverheadStreamScratchLen(stream DecoderStream, src []byte, workers int, events []DecoderEvent, spans []TileSpan, jobs []TileJob, batches []TileBatch) (DecoderFrameWorkResidualStreamScratchSize, error) {
	eventCount, err := decoderFrameWorkResidualLowOverheadEventLen(src)
	size := DecoderFrameWorkResidualStreamScratchSize{Events: eventCount}
	if err != nil {
		return size, err
	}
	if len(events) < eventCount {
		return size, ErrDecoderEventBufferTooSmall
	}
	count, err := stream.PushLowOverhead(src, events[:eventCount])
	if err != nil {
		return size, err
	}
	sequence, _ := stream.SequenceHeader()
	eventSize, err := DecoderFrameWorkResidualEventsScratchLen(sequence, events[:count], workers, spans, jobs, batches)
	if err != nil {
		return size, err
	}
	size.Event = eventSize
	return size, nil
}

// DecoderFrameWorkResidualRTPPayloadStreamScratchLen validates one AV1 RTP
// payload against a copy of stream and reports caller-owned RTP/event/residual
// scratch needed to run completed residual events. If used is non-zero, rtpBuffer
// must contain the preserved fragment bytes from the live stream runner.
func DecoderFrameWorkResidualRTPPayloadStreamScratchLen(stream DecoderStream, used int, payload []byte, workers int, rtpBuffer []byte, rtpSpans []RTPObuSpan, events []DecoderEvent, spans []TileSpan, jobs []TileJob, batches []TileBatch) (DecoderFrameWorkResidualStreamScratchSize, error) {
	plannedUsed, eventCount, err := stream.PushRTPPayloadSize(used, payload)
	size := DecoderFrameWorkResidualStreamScratchSize{
		Events:    eventCount,
		RTPBuffer: plannedUsed,
		RTPSpans:  eventCount,
	}
	if err != nil {
		return size, err
	}
	if len(rtpBuffer) < plannedUsed || len(rtpSpans) < eventCount {
		return size, ErrRTPShortBuffer
	}
	if len(events) < eventCount {
		return size, ErrDecoderEventBufferTooSmall
	}

	actualUsed, count, err := stream.PushRTPPayload(rtpBuffer[:plannedUsed], used, rtpSpans[:eventCount], events[:eventCount], payload)
	if err != nil {
		return size, err
	}
	if actualUsed != plannedUsed || count != eventCount {
		return size, ErrDecoderInvalidFrameWorkState
	}
	sequence, _ := stream.SequenceHeader()
	eventSize, err := DecoderFrameWorkResidualEventsScratchLen(sequence, events[:count], workers, spans, jobs, batches)
	if err != nil {
		return size, err
	}
	size.Event = eventSize
	return size, nil
}

// RunLowOverhead parses a low-overhead OBU buffer into caller-owned event
// scratch and immediately runs the parsed events through EventRunner.
func (r *DecoderFrameWorkResidualStreamRunner) RunLowOverhead(src []byte, post DecoderFrameWorkPostFilterFunc) (DecoderFrameWorkResidualStreamResult, error) {
	return r.runLowOverhead(src, post, nil)
}

// RunLowOverheadWithPostFilterRunner is RunLowOverhead using a direct
// postfilter runner instead of a postfilter callback.
func (r *DecoderFrameWorkResidualStreamRunner) RunLowOverheadWithPostFilterRunner(src []byte, post DecoderFrameWorkPostFilterRunner) (DecoderFrameWorkResidualStreamResult, error) {
	return r.runLowOverhead(src, nil, post)
}

// RunRTPPayload depacketizes one AV1 RTP payload into caller-owned OBU/event
// scratch and immediately runs any completed parsed events. RTPUsed is retained
// only while the underlying stream is inside an OBU fragment.
func (r *DecoderFrameWorkResidualStreamRunner) RunRTPPayload(payload []byte, post DecoderFrameWorkPostFilterFunc) (DecoderFrameWorkResidualStreamResult, error) {
	return r.runRTPPayload(payload, post, nil)
}

// RunRTPPayloadWithPostFilterRunner is RunRTPPayload using a direct postfilter
// runner instead of a postfilter callback.
func (r *DecoderFrameWorkResidualStreamRunner) RunRTPPayloadWithPostFilterRunner(payload []byte, post DecoderFrameWorkPostFilterRunner) (DecoderFrameWorkResidualStreamResult, error) {
	return r.runRTPPayload(payload, nil, post)
}

func (r *DecoderFrameWorkResidualStreamRunner) runLowOverhead(src []byte, post DecoderFrameWorkPostFilterFunc, postRunner DecoderFrameWorkPostFilterRunner) (DecoderFrameWorkResidualStreamResult, error) {
	if r == nil || r.Stream == nil {
		return DecoderFrameWorkResidualStreamResult{}, ErrDecoderInvalidFrameWorkState
	}
	if post != nil && postRunner != nil {
		return DecoderFrameWorkResidualStreamResult{}, ErrDecoderInvalidFrameWorkState
	}
	count, err := r.Stream.PushLowOverhead(src, r.Events)
	if err != nil {
		return DecoderFrameWorkResidualStreamResult{EventCount: count}, err
	}
	run, err := r.runParsedEvents(count, post, postRunner)
	if err != nil {
		return DecoderFrameWorkResidualStreamResult{EventCount: count, Run: run}, err
	}
	return DecoderFrameWorkResidualStreamResult{EventCount: count, Run: run}, nil
}

func (r *DecoderFrameWorkResidualStreamRunner) runRTPPayload(payload []byte, post DecoderFrameWorkPostFilterFunc, postRunner DecoderFrameWorkPostFilterRunner) (DecoderFrameWorkResidualStreamResult, error) {
	if r == nil || r.Stream == nil {
		return DecoderFrameWorkResidualStreamResult{}, ErrDecoderInvalidFrameWorkState
	}
	if post != nil && postRunner != nil {
		return DecoderFrameWorkResidualStreamResult{}, ErrDecoderInvalidFrameWorkState
	}
	used, count, err := r.Stream.PushRTPPayload(r.RTPBuffer, r.RTPUsed, r.RTPSpans, r.Events, payload)
	r.RTPUsed = used
	if err != nil {
		return DecoderFrameWorkResidualStreamResult{EventCount: count, RTPUsed: r.RTPUsed}, err
	}
	run, err := r.runParsedEvents(count, post, postRunner)
	if !r.Stream.InRTPFragment() {
		r.RTPUsed = 0
	}
	if err != nil {
		return DecoderFrameWorkResidualStreamResult{EventCount: count, RTPUsed: r.RTPUsed, Run: run}, err
	}
	return DecoderFrameWorkResidualStreamResult{EventCount: count, RTPUsed: r.RTPUsed, Run: run}, nil
}

func (r *DecoderFrameWorkResidualStreamRunner) runParsedEvents(count int, post DecoderFrameWorkPostFilterFunc, postRunner DecoderFrameWorkPostFilterRunner) (DecoderFrameWorkResidualEventsResult, error) {
	if count < 0 || count > len(r.Events) {
		return DecoderFrameWorkResidualEventsResult{}, ErrDecoderEventBufferTooSmall
	}
	sequence, _ := r.Stream.SequenceHeader()
	if postRunner != nil {
		return r.EventRunner.RunEventsWithPostFilterRunner(sequence, r.Events[:count], r.SideDataScratch, postRunner)
	}
	return r.EventRunner.RunEvents(sequence, r.Events[:count], r.SideDataScratch, post)
}

func decoderFrameWorkResidualLowOverheadEventLen(src []byte) (int, error) {
	it := NewLowOverheadIterator(src)
	count := 0
	for {
		_, ok, err := it.Next()
		if err != nil {
			return count, err
		}
		if !ok {
			return count, nil
		}
		count++
	}
}

func decoderFrameWorkResidualStreamSideDataScratchTooShort(scratch DecoderFrameWorkSideDataScratch, size DecoderFrameWorkSideDataScratchSize) bool {
	return decoderFrameWorkPostFilterScratchTooShort(scratch.CDEFIndexMap, size.CDEFIndexMap) ||
		decoderFrameWorkPostFilterScratchTooShort(scratch.CDEFReadMap, size.CDEFReadMap) ||
		decoderFrameWorkPostFilterScratchTooShort(scratch.LoopFilterMap, size.LoopFilterMap) ||
		decoderFrameWorkPostFilterScratchTooShort(scratch.RestorationRecords, size.RestorationRecords) ||
		decoderFrameWorkPostFilterScratchTooShort(scratch.RestorationBoundaryAbove, size.RestorationBoundaryAbove) ||
		decoderFrameWorkPostFilterScratchTooShort(scratch.RestorationBoundaryBelow, size.RestorationBoundaryBelow)
}
