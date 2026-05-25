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

// DecoderFrameWorkResidualStreamPlan reports both the caller-owned scratch
// required for a residual stream-runner call and the sequence/event pair callers
// should use to bind a reusable residual event runner for that stream shape.
type DecoderFrameWorkResidualStreamPlan struct {
	Size DecoderFrameWorkResidualStreamScratchSize
	Bind DecoderFrameWorkResidualEventBindPlan
}

// HasEvent reports whether Bind identifies an event that needs residual-event
// binding or can emit a decoded/display output.
func (p DecoderFrameWorkResidualStreamPlan) HasEvent() bool {
	return p.Bind.HasEvent()
}

// DecoderFrameWorkResidualStreamScratchSize reports caller-owned stream parser
// and residual-event scratch needed for one stream-runner call.
type DecoderFrameWorkResidualStreamScratchSize struct {
	Events    int
	RTPBuffer int
	RTPSpans  int
	Event     DecoderFrameWorkResidualEventScratchSize
}

// DecoderFrameWorkResidualStreamScratch carries typed caller-owned parser,
// residual event, side-data, and output arenas for residual stream runners.
type DecoderFrameWorkResidualStreamScratch struct {
	Events   []DecoderEvent
	Event    DecoderFrameWorkResidualEventScratch
	SideData DecoderFrameWorkSideDataScratch
	Outputs  []*Frame

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

// BindDecoderFrameWorkResidualStreamEventRunner binds a complete residual
// stream runner from caller-owned stream and event scratch. It first binds the
// nested residual event runner, then binds parser/RTP scratch and stream output
// slots around it.
func BindDecoderFrameWorkResidualStreamEventRunner(size DecoderFrameWorkResidualStreamScratchSize, stream *DecoderStream, sequence SequenceHeader, event DecoderEvent, runtime DecoderFrameWorkResidualEventRuntime, scratch DecoderFrameWorkResidualStreamScratch, batchRunner *DecoderFrameWorkBatchResidualRunner) (DecoderFrameWorkResidualStreamRunner, DecoderFrameWorkSideData, error) {
	if stream == nil {
		return DecoderFrameWorkResidualStreamRunner{}, DecoderFrameWorkSideData{}, ErrDecoderInvalidFrameWorkState
	}
	if err := decoderFrameWorkResidualStreamScratchErr(size, scratch); err != nil {
		return DecoderFrameWorkResidualStreamRunner{}, DecoderFrameWorkSideData{}, err
	}
	eventRunner, side, err := BindDecoderFrameWorkResidualEventRunner(size.Event, sequence, event, runtime, scratch.Event, batchRunner)
	if err != nil {
		return DecoderFrameWorkResidualStreamRunner{}, DecoderFrameWorkSideData{}, err
	}
	streamRunner, err := BindDecoderFrameWorkResidualStreamRunner(size, stream, eventRunner, scratch)
	if err != nil {
		return DecoderFrameWorkResidualStreamRunner{}, DecoderFrameWorkSideData{}, err
	}
	return streamRunner, side, nil
}

// BindDecoderFrameWorkResidualStreamRunner binds a stream runner from
// caller-owned parser and side-data scratch. The eventRunner is owned by the
// caller and is copied into the returned stream runner.
func BindDecoderFrameWorkResidualStreamRunner(size DecoderFrameWorkResidualStreamScratchSize, stream *DecoderStream, eventRunner DecoderFrameWorkResidualEventRunner, scratch DecoderFrameWorkResidualStreamScratch) (DecoderFrameWorkResidualStreamRunner, error) {
	if stream == nil {
		return DecoderFrameWorkResidualStreamRunner{}, ErrDecoderInvalidFrameWorkState
	}
	if err := decoderFrameWorkResidualStreamScratchErr(size, scratch); err != nil {
		return DecoderFrameWorkResidualStreamRunner{}, err
	}
	if scratch.Outputs != nil {
		eventRunner.Outputs = scratch.Outputs[:size.Event.Outputs]
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

func decoderFrameWorkResidualStreamScratchErr(size DecoderFrameWorkResidualStreamScratchSize, scratch DecoderFrameWorkResidualStreamScratch) error {
	if decoderFrameWorkResidualScratchTooShort(scratch.Events, size.Events) {
		return ErrDecoderEventBufferTooSmall
	}
	if decoderFrameWorkResidualScratchTooShort(scratch.RTPBuffer, size.RTPBuffer) ||
		decoderFrameWorkResidualScratchTooShort(scratch.RTPSpans, size.RTPSpans) {
		return ErrRTPShortBuffer
	}
	if decoderFrameWorkResidualStreamSideDataScratchTooShort(scratch.SideData, size.Event.SideData) {
		return ErrFrameShortBuffer
	}
	if scratch.Outputs != nil && decoderFrameWorkResidualScratchTooShort(scratch.Outputs, size.Event.Outputs) {
		return ErrFrameShortBuffer
	}
	return nil
}

// DecoderFrameWorkResidualLowOverheadStreamScratchLen parses a low-overhead
// OBU buffer through a copy of stream and reports caller-owned scratch needed to
// run its residual events. stream is passed by value so the live decoder stream
// is not mutated by sizing.
func DecoderFrameWorkResidualLowOverheadStreamScratchLen(stream DecoderStream, src []byte, workers int, events []DecoderEvent, spans []TileSpan, jobs []TileJob, batches []TileBatch) (DecoderFrameWorkResidualStreamScratchSize, error) {
	plan, err := DecoderFrameWorkResidualLowOverheadStreamPlan(stream, src, workers, events, spans, jobs, batches)
	return plan.Size, err
}

// DecoderFrameWorkResidualLowOverheadStreamPlan parses a low-overhead OBU
// buffer through a copy of stream and reports both caller-owned scratch and the
// bind context for a reusable residual stream runner. stream is passed by value
// so the live decoder stream is not mutated by planning.
func DecoderFrameWorkResidualLowOverheadStreamPlan(stream DecoderStream, src []byte, workers int, events []DecoderEvent, spans []TileSpan, jobs []TileJob, batches []TileBatch) (DecoderFrameWorkResidualStreamPlan, error) {
	eventCount, err := decoderFrameWorkResidualLowOverheadEventLen(src)
	plan := DecoderFrameWorkResidualStreamPlan{
		Size: DecoderFrameWorkResidualStreamScratchSize{Events: eventCount},
	}
	if err != nil {
		return plan, err
	}
	if len(events) < eventCount {
		return plan, ErrDecoderEventBufferTooSmall
	}
	sequence, _ := stream.SequenceHeader()
	count, err := stream.PushLowOverhead(src, events[:eventCount])
	if err != nil {
		return plan, err
	}
	eventSize, err := DecoderFrameWorkResidualEventsScratchLen(sequence, events[:count], workers, spans, jobs, batches)
	if err != nil {
		return plan, err
	}
	plan.Size.Event = eventSize
	plan.Bind = DecoderFrameWorkResidualEventsBindPlan(sequence, events[:count])
	return plan, nil
}

// DecoderFrameWorkResidualRTPPayloadStreamScratchLen validates one AV1 RTP
// payload against a copy of stream and reports caller-owned RTP/event/residual
// scratch needed to run completed residual events. If used is non-zero, rtpBuffer
// must contain the preserved fragment bytes from the live stream runner.
func DecoderFrameWorkResidualRTPPayloadStreamScratchLen(stream DecoderStream, used int, payload []byte, workers int, rtpBuffer []byte, rtpSpans []RTPObuSpan, events []DecoderEvent, spans []TileSpan, jobs []TileJob, batches []TileBatch) (DecoderFrameWorkResidualStreamScratchSize, error) {
	plan, err := DecoderFrameWorkResidualRTPPayloadStreamPlan(stream, used, payload, workers, rtpBuffer, rtpSpans, events, spans, jobs, batches)
	return plan.Size, err
}

// DecoderFrameWorkResidualRTPPayloadStreamPlan validates one AV1 RTP payload
// against a copy of stream and reports both caller-owned RTP/event/residual
// scratch and the bind context for completed residual events. If used is
// non-zero, rtpBuffer must contain the preserved fragment bytes from the live
// stream runner.
func DecoderFrameWorkResidualRTPPayloadStreamPlan(stream DecoderStream, used int, payload []byte, workers int, rtpBuffer []byte, rtpSpans []RTPObuSpan, events []DecoderEvent, spans []TileSpan, jobs []TileJob, batches []TileBatch) (DecoderFrameWorkResidualStreamPlan, error) {
	plannedUsed, eventCount, err := stream.PushRTPPayloadSize(used, payload)
	plan := DecoderFrameWorkResidualStreamPlan{
		Size: DecoderFrameWorkResidualStreamScratchSize{
			Events:    eventCount,
			RTPBuffer: plannedUsed,
			RTPSpans:  eventCount,
		},
	}
	if err != nil {
		return plan, err
	}
	if len(rtpBuffer) < plannedUsed || len(rtpSpans) < eventCount {
		return plan, ErrRTPShortBuffer
	}
	if len(events) < eventCount {
		return plan, ErrDecoderEventBufferTooSmall
	}

	sequence, _ := stream.SequenceHeader()
	actualUsed, count, err := stream.PushRTPPayload(rtpBuffer[:plannedUsed], used, rtpSpans[:eventCount], events[:eventCount], payload)
	if err != nil {
		return plan, err
	}
	if actualUsed != plannedUsed || count != eventCount {
		return plan, ErrDecoderInvalidFrameWorkState
	}
	eventSize, err := DecoderFrameWorkResidualEventsScratchLen(sequence, events[:count], workers, spans, jobs, batches)
	if err != nil {
		return plan, err
	}
	plan.Size.Event = eventSize
	plan.Bind = DecoderFrameWorkResidualEventsBindPlan(sequence, events[:count])
	return plan, nil
}

// DecoderFrameWorkResidualRTPPayloadsStreamScratchLen validates an ordered AV1
// RTP payload batch against a copy of stream and reports reusable max scratch
// needed by RunRTPPayloads. If used is non-zero, rtpBuffer must contain the
// preserved fragment bytes from the live stream runner.
func DecoderFrameWorkResidualRTPPayloadsStreamScratchLen(stream DecoderStream, used int, payloads [][]byte, workers int, rtpBuffer []byte, rtpSpans []RTPObuSpan, events []DecoderEvent, spans []TileSpan, jobs []TileJob, batches []TileBatch) (DecoderFrameWorkResidualStreamScratchSize, error) {
	plan, err := DecoderFrameWorkResidualRTPPayloadsStreamPlan(stream, used, payloads, workers, rtpBuffer, rtpSpans, events, spans, jobs, batches)
	return plan.Size, err
}

// DecoderFrameWorkResidualRTPPayloadsStreamPlan validates an ordered AV1 RTP
// payload batch against a copy of stream and reports reusable max scratch plus
// the last bind-relevant event across the payload batch. If used is non-zero,
// rtpBuffer must contain the preserved fragment bytes from the live stream
// runner.
func DecoderFrameWorkResidualRTPPayloadsStreamPlan(stream DecoderStream, used int, payloads [][]byte, workers int, rtpBuffer []byte, rtpSpans []RTPObuSpan, events []DecoderEvent, spans []TileSpan, jobs []TileJob, batches []TileBatch) (DecoderFrameWorkResidualStreamPlan, error) {
	var plan DecoderFrameWorkResidualStreamPlan
	outputs := 0
	for i := range payloads {
		plannedUsed, eventCount, err := stream.PushRTPPayloadSize(used, payloads[i])
		nextSize := DecoderFrameWorkResidualStreamScratchSize{
			Events:    eventCount,
			RTPBuffer: plannedUsed,
			RTPSpans:  eventCount,
		}
		plan.Size = plan.Size.Max(nextSize)
		if err != nil {
			return plan, err
		}
		if len(rtpBuffer) < plannedUsed || len(rtpSpans) < eventCount {
			return plan, ErrRTPShortBuffer
		}
		if len(events) < eventCount {
			return plan, ErrDecoderEventBufferTooSmall
		}

		sequence, _ := stream.SequenceHeader()
		actualUsed, count, err := stream.PushRTPPayload(rtpBuffer[:plannedUsed], used, rtpSpans[:eventCount], events[:eventCount], payloads[i])
		if err != nil {
			return plan, err
		}
		if actualUsed != plannedUsed || count != eventCount {
			return plan, ErrDecoderInvalidFrameWorkState
		}
		eventSize, err := DecoderFrameWorkResidualEventsScratchLen(sequence, events[:count], workers, spans, jobs, batches)
		if err != nil {
			return plan, err
		}
		nextBind := DecoderFrameWorkResidualEventsBindPlan(sequence, events[:count])
		if nextBind.HasEvent() {
			plan.Bind = nextBind
		} else if !plan.Bind.HasEvent() {
			plan.Bind.Sequence = nextBind.Sequence
		}
		nextSize.Event = eventSize
		outputs += eventSize.Outputs
		plan.Size = plan.Size.Max(nextSize)
		plan.Size.Event.Outputs = outputs
		used = actualUsed
		if !stream.InRTPFragment() {
			used = 0
		}
	}
	return plan, nil
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

// RunRTPPayloads depacketizes and runs an ordered batch of AV1 RTP payloads,
// preserving RTP fragment state across payloads and aggregating completed event
// work into one result.
func (r *DecoderFrameWorkResidualStreamRunner) RunRTPPayloads(payloads [][]byte, post DecoderFrameWorkPostFilterFunc) (DecoderFrameWorkResidualStreamResult, error) {
	return r.runRTPPayloads(payloads, post, nil)
}

// RunRTPPayloadsWithPostFilterRunner is RunRTPPayloads using a direct
// postfilter runner instead of a postfilter callback.
func (r *DecoderFrameWorkResidualStreamRunner) RunRTPPayloadsWithPostFilterRunner(payloads [][]byte, post DecoderFrameWorkPostFilterRunner) (DecoderFrameWorkResidualStreamResult, error) {
	return r.runRTPPayloads(payloads, nil, post)
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
	run, err := r.runParsedEvents(count, 0, post, postRunner)
	if err != nil {
		return DecoderFrameWorkResidualStreamResult{EventCount: count, Run: run}, err
	}
	return DecoderFrameWorkResidualStreamResult{EventCount: count, Run: run}, nil
}

func (r *DecoderFrameWorkResidualStreamRunner) runRTPPayload(payload []byte, post DecoderFrameWorkPostFilterFunc, postRunner DecoderFrameWorkPostFilterRunner) (DecoderFrameWorkResidualStreamResult, error) {
	return r.runRTPPayloadWithOutputOffset(payload, 0, post, postRunner)
}

func (r *DecoderFrameWorkResidualStreamRunner) runRTPPayloadWithOutputOffset(payload []byte, outputOffset int, post DecoderFrameWorkPostFilterFunc, postRunner DecoderFrameWorkPostFilterRunner) (DecoderFrameWorkResidualStreamResult, error) {
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
	run, err := r.runParsedEvents(count, outputOffset, post, postRunner)
	if !r.Stream.InRTPFragment() {
		r.RTPUsed = 0
	}
	if err != nil {
		return DecoderFrameWorkResidualStreamResult{EventCount: count, RTPUsed: r.RTPUsed, Run: run}, err
	}
	return DecoderFrameWorkResidualStreamResult{EventCount: count, RTPUsed: r.RTPUsed, Run: run}, nil
}

func (r *DecoderFrameWorkResidualStreamRunner) runRTPPayloads(payloads [][]byte, post DecoderFrameWorkPostFilterFunc, postRunner DecoderFrameWorkPostFilterRunner) (DecoderFrameWorkResidualStreamResult, error) {
	if r == nil || r.Stream == nil {
		return DecoderFrameWorkResidualStreamResult{}, ErrDecoderInvalidFrameWorkState
	}
	if post != nil && postRunner != nil {
		return DecoderFrameWorkResidualStreamResult{}, ErrDecoderInvalidFrameWorkState
	}
	var result DecoderFrameWorkResidualStreamResult
	for i := range payloads {
		next, err := r.runRTPPayloadWithOutputOffset(payloads[i], result.Run.OutputCount, post, postRunner)
		decoderFrameWorkAccumulateResidualStreamResult(&result, next)
		decoderFrameWorkResidualStreamBindResultOutputs(&result, r.EventRunner.Outputs)
		if err != nil {
			return result, err
		}
	}
	return result, nil
}

func (r *DecoderFrameWorkResidualStreamRunner) runParsedEvents(count int, outputOffset int, post DecoderFrameWorkPostFilterFunc, postRunner DecoderFrameWorkPostFilterRunner) (DecoderFrameWorkResidualEventsResult, error) {
	if count < 0 || count > len(r.Events) {
		return DecoderFrameWorkResidualEventsResult{}, ErrDecoderEventBufferTooSmall
	}
	eventRunner := r.EventRunner
	if eventRunner.Outputs != nil {
		if outputOffset < 0 || outputOffset > len(eventRunner.Outputs) {
			return DecoderFrameWorkResidualEventsResult{}, ErrFrameShortBuffer
		}
		eventRunner.Outputs = eventRunner.Outputs[outputOffset:]
	}
	sequence, _ := r.Stream.SequenceHeader()
	if postRunner != nil {
		return eventRunner.RunEventsWithPostFilterRunner(sequence, r.Events[:count], r.SideDataScratch, postRunner)
	}
	return eventRunner.RunEvents(sequence, r.Events[:count], r.SideDataScratch, post)
}

func decoderFrameWorkAccumulateResidualStreamResult(total *DecoderFrameWorkResidualStreamResult, next DecoderFrameWorkResidualStreamResult) {
	total.EventCount += next.EventCount
	total.RTPUsed = next.RTPUsed
	decoderFrameWorkAccumulateResidualEventsResult(&total.Run, next.Run)
}

func decoderFrameWorkAccumulateResidualEventsResult(total *DecoderFrameWorkResidualEventsResult, next DecoderFrameWorkResidualEventsResult) {
	total.Count += next.Count
	if next.Count > 0 {
		total.Last = next.Last
	}
	total.ExecutedTileWork += next.ExecutedTileWork
	total.CompletedFrames += next.CompletedFrames
	total.OutputCount += next.OutputCount
	total.ReleaseCount += next.ReleaseCount
	decoderFrameWorkAccumulateResidualStats(&total.Stats, next.Stats)
}

func decoderFrameWorkResidualStreamBindResultOutputs(result *DecoderFrameWorkResidualStreamResult, outputs []*Frame) {
	if outputs == nil {
		return
	}
	result.Run.Outputs = outputs[:result.Run.OutputCount]
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
