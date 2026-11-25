package goav1

import (
	internaldecoder "github.com/thesyncim/goav1/internal/av1/decoder"
	internalframe "github.com/thesyncim/goav1/internal/av1/frame"
	internalobu "github.com/thesyncim/goav1/internal/av1/obu"
)

// decoderFrameWorkResidualStreamRunner connects the byte-stream/RTP event
// parser to the residual event runner using caller-owned buffers.
type decoderFrameWorkResidualStreamRunner struct {
	Stream      *internaldecoder.Stream
	EventRunner decoderFrameWorkResidualEventRunner

	Events          []internaldecoder.Event
	SideDataScratch decoderFrameWorkSideDataScratch

	RTPBuffer []byte
	RTPSpans  []RTPObuSpan
	RTPUsed   int
}

// decoderFrameWorkResidualStreamRunnerState is a cheap snapshot of a bound
// stream runner's parser state and reusable arena capacities.
type decoderFrameWorkResidualStreamRunnerState struct {
	Bound             bool
	HasSequenceHeader bool
	InRTPFragment     bool
	RTPUsed           int

	EventCapacity     int
	RTPBufferCapacity int
	RTPSpanCapacity   int
	OutputCapacity    int
}

// decoderFrameWorkResidualStreamResult reports parser and decode work completed
// by a stream-runner call.
type decoderFrameWorkResidualStreamResult struct {
	EventCount int
	RTPUsed    int
	Run        decoderFrameWorkResidualEventsResult
}

// Accumulate adds next into r using the same result semantics as
// decoderFrameWorkResidualStreamRunner.RunRTPPayloads. Event counts are summed,
// RTPUsed is replaced by the newest retained-fragment length, and Run is
// accumulated.
func (r *decoderFrameWorkResidualStreamResult) Accumulate(next decoderFrameWorkResidualStreamResult) error {
	if r == nil {
		return internaldecoder.ErrInvalidFrameWorkState
	}
	decoderFrameWorkAccumulateResidualStreamResult(r, next)
	return nil
}

// BindOutputs aliases Run.Outputs to the first Run.OutputCount slots in
// outputs.
func (r *decoderFrameWorkResidualStreamResult) BindOutputs(outputs []*Frame) error {
	if r == nil {
		return internaldecoder.ErrInvalidFrameWorkState
	}
	return r.Run.BindOutputs(outputs)
}

// decoderFrameWorkResidualStreamPlan reports both the caller-owned scratch
// required for a residual stream-runner call and the sequence/event pair callers
// should use to bind a reusable residual event runner for that stream shape.
type decoderFrameWorkResidualStreamPlan struct {
	Size decoderFrameWorkResidualStreamScratchSize
	Bind decoderFrameWorkResidualEventBindPlan
}

// HasEvent reports whether Bind identifies an event that needs residual-event
// binding or can emit a decoded/display output.
func (p decoderFrameWorkResidualStreamPlan) HasEvent() bool {
	return p.Bind.HasEvent()
}

// Max returns a reusable plan large enough for either stream plan. If both
// plans have bind-relevant events, the bind context from other is retained so
// callers can accumulate plans in decode order and bind against the latest event
// shape.
func (p decoderFrameWorkResidualStreamPlan) Max(other decoderFrameWorkResidualStreamPlan) decoderFrameWorkResidualStreamPlan {
	return decoderFrameWorkResidualStreamPlan{
		Size: p.Size.Max(other.Size),
		Bind: decoderFrameWorkResidualStreamBindPlanMax(
			p.Bind,
			other.Bind,
		),
	}
}

// decoderFrameWorkResidualStreamScratchSize reports caller-owned stream parser
// and residual-event scratch needed for one stream-runner call.
type decoderFrameWorkResidualStreamScratchSize struct {
	Events    int
	RTPBuffer int
	RTPSpans  int
	Event     decoderFrameWorkResidualEventScratchSize
}

// decoderFrameWorkResidualStreamScratch carries typed caller-owned parser,
// residual event, side-data, and output arenas for residual stream runners.
type decoderFrameWorkResidualStreamScratch struct {
	Events   []internaldecoder.Event
	Event    decoderFrameWorkResidualEventScratch
	SideData decoderFrameWorkSideDataScratch
	Outputs  []*Frame

	RTPBuffer []byte
	RTPSpans  []RTPObuSpan
}

// Check reports whether scratch is large enough for size without binding or
// mutating any caller-owned storage.
func (s decoderFrameWorkResidualStreamScratch) Check(size decoderFrameWorkResidualStreamScratchSize) error {
	if err := decoderFrameWorkResidualStreamScratchErr(size, s); err != nil {
		return err
	}
	return s.Event.Check(size.Event)
}

// Max returns per-arena maximum lengths for reusable residual stream scratch.
func (s decoderFrameWorkResidualStreamScratchSize) Max(other decoderFrameWorkResidualStreamScratchSize) decoderFrameWorkResidualStreamScratchSize {
	return decoderFrameWorkResidualStreamScratchSize{
		Events:    max(s.Events, other.Events),
		RTPBuffer: max(s.RTPBuffer, other.RTPBuffer),
		RTPSpans:  max(s.RTPSpans, other.RTPSpans),
		Event:     s.Event.Max(other.Event),
	}
}

// State returns a cheap snapshot of parser state and runner arena capacities.
// A nil or unbound runner returns the zero state.
func (r *decoderFrameWorkResidualStreamRunner) State() decoderFrameWorkResidualStreamRunnerState {
	if r == nil || r.Stream == nil {
		return decoderFrameWorkResidualStreamRunnerState{}
	}
	return decoderFrameWorkResidualStreamRunnerState{
		Bound:             true,
		HasSequenceHeader: r.Stream.HasSequenceHeader(),
		InRTPFragment:     r.Stream.InRTPFragment(),
		RTPUsed:           r.RTPUsed,
		EventCapacity:     len(r.Events),
		RTPBufferCapacity: len(r.RTPBuffer),
		RTPSpanCapacity:   len(r.RTPSpans),
		OutputCapacity:    len(r.EventRunner.Outputs),
	}
}

// SequenceHeader returns the currently accepted sequence header, if the stream
// runner is bound and has seen one.
func (r *decoderFrameWorkResidualStreamRunner) SequenceHeader() (SequenceHeader, bool) {
	if r == nil || r.Stream == nil {
		return SequenceHeader{}, false
	}
	return r.Stream.SequenceHeader()
}

// Reset clears the bound parser stream and any retained RTP fragment bytes so
// the runner can be reused for an independent decode stream.
func (r *decoderFrameWorkResidualStreamRunner) Reset() error {
	if r == nil || r.Stream == nil {
		return internaldecoder.ErrInvalidFrameWorkState
	}
	r.Stream.Reset()
	r.RTPUsed = 0
	return nil
}

// ResetRTP clears only retained RTP fragment bytes while preserving parser
// sequence/reference state. Callers can use this after packet loss or jitter
// buffer resynchronization.
func (r *decoderFrameWorkResidualStreamRunner) ResetRTP() error {
	if r == nil || r.Stream == nil {
		return internaldecoder.ErrInvalidFrameWorkState
	}
	r.Stream.ResetRTP()
	r.RTPUsed = 0
	return nil
}

// bindDecoderFrameWorkResidualStreamEventRunner binds a complete residual
// stream runner from caller-owned stream and event scratch. It first binds the
// nested residual event runner, then binds parser/RTP scratch and stream output
// slots around it.
func bindDecoderFrameWorkResidualStreamEventRunner(size decoderFrameWorkResidualStreamScratchSize, stream *internaldecoder.Stream, sequence SequenceHeader, event *internaldecoder.Event, runtime decoderFrameWorkResidualEventRuntime, scratch decoderFrameWorkResidualStreamScratch, batchRunner *decoderFrameWorkBatchResidualRunner) (decoderFrameWorkResidualStreamRunner, decoderFrameWorkSideData, error) {
	return bindDecoderFrameWorkResidualStreamEventRunnerWith(size, stream, sequence, event, runtime, scratch, batchRunner, false)
}

func bindDecoderFrameWorkResidualStreamEventRunnerWith(size decoderFrameWorkResidualStreamScratchSize, stream *internaldecoder.Stream, sequence SequenceHeader, event *internaldecoder.Event, runtime decoderFrameWorkResidualEventRuntime, scratch decoderFrameWorkResidualStreamScratch, batchRunner *decoderFrameWorkBatchResidualRunner, maskOnlyLoopFilter bool) (decoderFrameWorkResidualStreamRunner, decoderFrameWorkSideData, error) {
	if stream == nil {
		return decoderFrameWorkResidualStreamRunner{}, decoderFrameWorkSideData{}, internaldecoder.ErrInvalidFrameWorkState
	}
	if err := decoderFrameWorkResidualStreamScratchErr(size, scratch); err != nil {
		return decoderFrameWorkResidualStreamRunner{}, decoderFrameWorkSideData{}, err
	}
	eventRunner, side, err := bindDecoderFrameWorkResidualEventRunnerWith(size.Event, sequence, event, runtime, scratch.Event, batchRunner, maskOnlyLoopFilter)
	if err != nil {
		return decoderFrameWorkResidualStreamRunner{}, decoderFrameWorkSideData{}, err
	}
	streamRunner, err := bindDecoderFrameWorkResidualStreamRunner(size, stream, eventRunner, scratch)
	if err != nil {
		return decoderFrameWorkResidualStreamRunner{}, decoderFrameWorkSideData{}, err
	}
	return streamRunner, side, nil
}

// bindDecoderFrameWorkResidualStreamPlanRunner binds a complete residual stream
// runner directly from a stream plan.
func bindDecoderFrameWorkResidualStreamPlanRunner(plan decoderFrameWorkResidualStreamPlan, stream *internaldecoder.Stream, runtime decoderFrameWorkResidualEventRuntime, scratch decoderFrameWorkResidualStreamScratch, batchRunner *decoderFrameWorkBatchResidualRunner) (decoderFrameWorkResidualStreamRunner, decoderFrameWorkSideData, error) {
	return bindDecoderFrameWorkResidualStreamEventRunnerWith(plan.Size, stream, plan.Bind.Sequence, &plan.Bind.Event, runtime, scratch, batchRunner, false)
}

func bindDecoderFrameWorkResidualStreamPlanRunnerWith(plan decoderFrameWorkResidualStreamPlan, stream *internaldecoder.Stream, runtime decoderFrameWorkResidualEventRuntime, scratch decoderFrameWorkResidualStreamScratch, batchRunner *decoderFrameWorkBatchResidualRunner, maskOnlyLoopFilter bool) (decoderFrameWorkResidualStreamRunner, decoderFrameWorkSideData, error) {
	return bindDecoderFrameWorkResidualStreamEventRunnerWith(plan.Size, stream, plan.Bind.Sequence, &plan.Bind.Event, runtime, scratch, batchRunner, maskOnlyLoopFilter)
}

// bindDecoderFrameWorkResidualStreamRunner binds a stream runner from
// caller-owned parser and side-data scratch. The eventRunner is owned by the
// caller and is copied into the returned stream runner.
func bindDecoderFrameWorkResidualStreamRunner(size decoderFrameWorkResidualStreamScratchSize, stream *internaldecoder.Stream, eventRunner decoderFrameWorkResidualEventRunner, scratch decoderFrameWorkResidualStreamScratch) (decoderFrameWorkResidualStreamRunner, error) {
	if stream == nil {
		return decoderFrameWorkResidualStreamRunner{}, internaldecoder.ErrInvalidFrameWorkState
	}
	if err := decoderFrameWorkResidualStreamScratchErr(size, scratch); err != nil {
		return decoderFrameWorkResidualStreamRunner{}, err
	}
	if scratch.Outputs != nil {
		eventRunner.Outputs = scratch.Outputs[:size.Event.Outputs]
	}
	return decoderFrameWorkResidualStreamRunner{
		Stream:      stream,
		EventRunner: eventRunner,
		Events:      scratch.Events[:size.Events],
		SideDataScratch: decoderFrameWorkSideDataScratch{
			CDEFIndexMap:             scratch.SideData.CDEFIndexMap[:size.Event.SideData.CDEFIndexMap],
			CDEFReadMap:              scratch.SideData.CDEFReadMap[:size.Event.SideData.CDEFReadMap],
			LoopFilterMap:            scratch.SideData.LoopFilterMap[:size.Event.SideData.LoopFilterMap],
			LoopFilterMasks:          decoderFrameWorkResliceOrNil(scratch.SideData.LoopFilterMasks, size.Event.SideData.LoopFilterMasks),
			LoopFilterLevelCache:     decoderFrameWorkResliceOrNil(scratch.SideData.LoopFilterLevelCache, size.Event.SideData.LoopFilterLevelCache),
			RestorationRecords:       scratch.SideData.RestorationRecords[:size.Event.SideData.RestorationRecords],
			RestorationBoundaryAbove: scratch.SideData.RestorationBoundaryAbove[:size.Event.SideData.RestorationBoundaryAbove],
			RestorationBoundaryBelow: scratch.SideData.RestorationBoundaryBelow[:size.Event.SideData.RestorationBoundaryBelow],
		},
		RTPBuffer: scratch.RTPBuffer[:size.RTPBuffer],
		RTPSpans:  scratch.RTPSpans[:size.RTPSpans],
	}, nil
}

func decoderFrameWorkResidualStreamScratchErr(size decoderFrameWorkResidualStreamScratchSize, scratch decoderFrameWorkResidualStreamScratch) error {
	if decoderFrameWorkResidualScratchTooShort(scratch.Events, size.Events) {
		return ErrDecoderEventBufferTooSmall
	}
	if decoderFrameWorkResidualScratchTooShort(scratch.RTPBuffer, size.RTPBuffer) ||
		decoderFrameWorkResidualScratchTooShort(scratch.RTPSpans, size.RTPSpans) {
		return ErrRTPShortBuffer
	}
	if decoderFrameWorkResidualStreamSideDataScratchTooShort(scratch.SideData, size.Event.SideData) {
		return internalframe.ErrShortBuffer
	}
	if scratch.Outputs != nil && decoderFrameWorkResidualScratchTooShort(scratch.Outputs, size.Event.Outputs) {
		return internalframe.ErrShortBuffer
	}
	return nil
}

// RunLowOverhead parses a low-overhead OBU buffer into caller-owned event
// scratch and immediately runs the parsed events through EventRunner.
func (r *decoderFrameWorkResidualStreamRunner) RunLowOverhead(src []byte, post internaldecoder.FrameWorkPostFilterFunc) (decoderFrameWorkResidualStreamResult, error) {
	return r.runLowOverhead(src, post, nil)
}

// RunLowOverheadWithPostFilterRunner is RunLowOverhead using a direct
// postfilter runner instead of a postfilter callback.
func (r *decoderFrameWorkResidualStreamRunner) RunLowOverheadWithPostFilterRunner(src []byte, post internaldecoder.FrameWorkPostFilterRunner) (decoderFrameWorkResidualStreamResult, error) {
	return r.runLowOverhead(src, nil, post)
}

// RunLowOverheadInto parses and runs one low-overhead OBU buffer, then appends
// its counters and outputs into result.
func (r *decoderFrameWorkResidualStreamRunner) RunLowOverheadInto(result *decoderFrameWorkResidualStreamResult, src []byte, post internaldecoder.FrameWorkPostFilterFunc) error {
	return r.runLowOverheadInto(result, src, post, nil)
}

// RunLowOverheadIntoWithPostFilterRunner is RunLowOverheadInto using a direct
// postfilter runner instead of a postfilter callback.
func (r *decoderFrameWorkResidualStreamRunner) RunLowOverheadIntoWithPostFilterRunner(result *decoderFrameWorkResidualStreamResult, src []byte, post internaldecoder.FrameWorkPostFilterRunner) error {
	return r.runLowOverheadInto(result, src, nil, post)
}

// RunLowOverheads parses and runs an ordered batch of low-overhead OBU buffers
// and aggregates completed event work into one result.
func (r *decoderFrameWorkResidualStreamRunner) RunLowOverheads(srcs [][]byte, post internaldecoder.FrameWorkPostFilterFunc) (decoderFrameWorkResidualStreamResult, error) {
	return r.runLowOverheads(srcs, post, nil)
}

// RunLowOverheadsWithPostFilterRunner is RunLowOverheads using a direct
// postfilter runner instead of a postfilter callback.
func (r *decoderFrameWorkResidualStreamRunner) RunLowOverheadsWithPostFilterRunner(srcs [][]byte, post internaldecoder.FrameWorkPostFilterRunner) (decoderFrameWorkResidualStreamResult, error) {
	return r.runLowOverheads(srcs, nil, post)
}

// RunLowOverheadsInto parses and runs an ordered batch of low-overhead OBU
// buffers, then appends their counters and outputs into result.
func (r *decoderFrameWorkResidualStreamRunner) RunLowOverheadsInto(result *decoderFrameWorkResidualStreamResult, srcs [][]byte, post internaldecoder.FrameWorkPostFilterFunc) error {
	return r.runLowOverheadsInto(result, srcs, post, nil)
}

// RunLowOverheadsIntoWithPostFilterRunner is RunLowOverheadsInto using a
// direct postfilter runner instead of a postfilter callback.
func (r *decoderFrameWorkResidualStreamRunner) RunLowOverheadsIntoWithPostFilterRunner(result *decoderFrameWorkResidualStreamResult, srcs [][]byte, post internaldecoder.FrameWorkPostFilterRunner) error {
	return r.runLowOverheadsInto(result, srcs, nil, post)
}

// RunRTPPayload depacketizes one AV1 RTP payload into caller-owned OBU/event
// scratch and immediately runs any completed parsed events. RTPUsed is retained
// only while the underlying stream is inside an OBU fragment.
func (r *decoderFrameWorkResidualStreamRunner) RunRTPPayload(payload []byte, post internaldecoder.FrameWorkPostFilterFunc) (decoderFrameWorkResidualStreamResult, error) {
	return r.runRTPPayload(payload, post, nil)
}

// RunRTPPayloadWithPostFilterRunner is RunRTPPayload using a direct postfilter
// runner instead of a postfilter callback.
func (r *decoderFrameWorkResidualStreamRunner) RunRTPPayloadWithPostFilterRunner(payload []byte, post internaldecoder.FrameWorkPostFilterRunner) (decoderFrameWorkResidualStreamResult, error) {
	return r.runRTPPayload(payload, nil, post)
}

// RunRTPPayloadAfterLoss clears any retained RTP fragment bytes, then
// depacketizes and runs one AV1 RTP payload. Parser sequence/reference state is
// preserved, matching ResetRTP.
func (r *decoderFrameWorkResidualStreamRunner) RunRTPPayloadAfterLoss(payload []byte, post internaldecoder.FrameWorkPostFilterFunc) (decoderFrameWorkResidualStreamResult, error) {
	return r.runRTPPayloadAfterLoss(payload, post, nil)
}

// RunRTPPayloadAfterLossWithPostFilterRunner is RunRTPPayloadAfterLoss using a
// direct postfilter runner instead of a postfilter callback.
func (r *decoderFrameWorkResidualStreamRunner) RunRTPPayloadAfterLossWithPostFilterRunner(payload []byte, post internaldecoder.FrameWorkPostFilterRunner) (decoderFrameWorkResidualStreamResult, error) {
	return r.runRTPPayloadAfterLoss(payload, nil, post)
}

// RunRTPPayloadInto depacketizes and runs one AV1 RTP payload, then appends its
// counters and outputs into result. This is the packet-by-packet equivalent of
// RunRTPPayloads for callers that drive decoding from a realtime jitter buffer.
func (r *decoderFrameWorkResidualStreamRunner) RunRTPPayloadInto(result *decoderFrameWorkResidualStreamResult, payload []byte, post internaldecoder.FrameWorkPostFilterFunc) error {
	return r.runRTPPayloadInto(result, payload, post, nil)
}

// RunRTPPayloadIntoWithPostFilterRunner is RunRTPPayloadInto using a direct
// postfilter runner instead of a postfilter callback.
func (r *decoderFrameWorkResidualStreamRunner) RunRTPPayloadIntoWithPostFilterRunner(result *decoderFrameWorkResidualStreamResult, payload []byte, post internaldecoder.FrameWorkPostFilterRunner) error {
	return r.runRTPPayloadInto(result, payload, nil, post)
}

// RunRTPPayloadAfterLossInto clears any retained RTP fragment bytes, then
// depacketizes and runs one AV1 RTP payload and appends its counters and outputs
// into result.
func (r *decoderFrameWorkResidualStreamRunner) RunRTPPayloadAfterLossInto(result *decoderFrameWorkResidualStreamResult, payload []byte, post internaldecoder.FrameWorkPostFilterFunc) error {
	return r.runRTPPayloadAfterLossInto(result, payload, post, nil)
}

// RunRTPPayloadAfterLossIntoWithPostFilterRunner is
// RunRTPPayloadAfterLossInto using a direct postfilter runner instead of a
// postfilter callback.
func (r *decoderFrameWorkResidualStreamRunner) RunRTPPayloadAfterLossIntoWithPostFilterRunner(result *decoderFrameWorkResidualStreamResult, payload []byte, post internaldecoder.FrameWorkPostFilterRunner) error {
	return r.runRTPPayloadAfterLossInto(result, payload, nil, post)
}

// RunRTPPayloads depacketizes and runs an ordered batch of AV1 RTP payloads,
// preserving RTP fragment state across payloads and aggregating completed event
// work into one result.
func (r *decoderFrameWorkResidualStreamRunner) RunRTPPayloads(payloads [][]byte, post internaldecoder.FrameWorkPostFilterFunc) (decoderFrameWorkResidualStreamResult, error) {
	return r.runRTPPayloads(payloads, post, nil)
}

// RunRTPPayloadsWithPostFilterRunner is RunRTPPayloads using a direct
// postfilter runner instead of a postfilter callback.
func (r *decoderFrameWorkResidualStreamRunner) RunRTPPayloadsWithPostFilterRunner(payloads [][]byte, post internaldecoder.FrameWorkPostFilterRunner) (decoderFrameWorkResidualStreamResult, error) {
	return r.runRTPPayloads(payloads, nil, post)
}

// RunRTPPayloadsInto depacketizes and runs an ordered batch of AV1 RTP
// payloads, then appends their counters and outputs into result.
func (r *decoderFrameWorkResidualStreamRunner) RunRTPPayloadsInto(result *decoderFrameWorkResidualStreamResult, payloads [][]byte, post internaldecoder.FrameWorkPostFilterFunc) error {
	return r.runRTPPayloadsInto(result, payloads, post, nil)
}

// RunRTPPayloadsIntoWithPostFilterRunner is RunRTPPayloadsInto using a direct
// postfilter runner instead of a postfilter callback.
func (r *decoderFrameWorkResidualStreamRunner) RunRTPPayloadsIntoWithPostFilterRunner(result *decoderFrameWorkResidualStreamResult, payloads [][]byte, post internaldecoder.FrameWorkPostFilterRunner) error {
	return r.runRTPPayloadsInto(result, payloads, nil, post)
}

func (r *decoderFrameWorkResidualStreamRunner) runLowOverhead(src []byte, post internaldecoder.FrameWorkPostFilterFunc, postRunner internaldecoder.FrameWorkPostFilterRunner) (decoderFrameWorkResidualStreamResult, error) {
	return r.runLowOverheadWithOutputOffset(src, 0, post, postRunner)
}

func (r *decoderFrameWorkResidualStreamRunner) runLowOverheadInto(result *decoderFrameWorkResidualStreamResult, src []byte, post internaldecoder.FrameWorkPostFilterFunc, postRunner internaldecoder.FrameWorkPostFilterRunner) error {
	if result == nil {
		return internaldecoder.ErrInvalidFrameWorkState
	}
	next, err := r.runLowOverheadWithOutputOffset(src, result.Run.OutputCount, post, postRunner)
	if accErr := result.Accumulate(next); accErr != nil {
		return accErr
	}
	decoderFrameWorkResidualStreamBindResultOutputs(result, r.EventRunner.Outputs)
	return err
}

func (r *decoderFrameWorkResidualStreamRunner) runLowOverheads(srcs [][]byte, post internaldecoder.FrameWorkPostFilterFunc, postRunner internaldecoder.FrameWorkPostFilterRunner) (decoderFrameWorkResidualStreamResult, error) {
	if r == nil || r.Stream == nil {
		return decoderFrameWorkResidualStreamResult{}, internaldecoder.ErrInvalidFrameWorkState
	}
	if post != nil && postRunner != nil {
		return decoderFrameWorkResidualStreamResult{}, internaldecoder.ErrInvalidFrameWorkState
	}
	var result decoderFrameWorkResidualStreamResult
	for i := range srcs {
		next, err := r.runLowOverheadWithOutputOffset(srcs[i], result.Run.OutputCount, post, postRunner)
		decoderFrameWorkAccumulateResidualStreamResult(&result, next)
		decoderFrameWorkResidualStreamBindResultOutputs(&result, r.EventRunner.Outputs)
		if err != nil {
			return result, err
		}
	}
	return result, nil
}

func (r *decoderFrameWorkResidualStreamRunner) runLowOverheadsInto(result *decoderFrameWorkResidualStreamResult, srcs [][]byte, post internaldecoder.FrameWorkPostFilterFunc, postRunner internaldecoder.FrameWorkPostFilterRunner) error {
	if result == nil || r == nil || r.Stream == nil {
		return internaldecoder.ErrInvalidFrameWorkState
	}
	if post != nil && postRunner != nil {
		return internaldecoder.ErrInvalidFrameWorkState
	}
	for i := range srcs {
		if err := r.runLowOverheadInto(result, srcs[i], post, postRunner); err != nil {
			return err
		}
	}
	return nil
}

func (r *decoderFrameWorkResidualStreamRunner) runLowOverheadWithOutputOffset(src []byte, outputOffset int, post internaldecoder.FrameWorkPostFilterFunc, postRunner internaldecoder.FrameWorkPostFilterRunner) (decoderFrameWorkResidualStreamResult, error) {
	if r == nil || r.Stream == nil {
		return decoderFrameWorkResidualStreamResult{}, internaldecoder.ErrInvalidFrameWorkState
	}
	if post != nil && postRunner != nil {
		return decoderFrameWorkResidualStreamResult{}, internaldecoder.ErrInvalidFrameWorkState
	}
	count, err := r.Stream.PushLowOverhead(src, r.Events)
	if err != nil {
		return decoderFrameWorkResidualStreamResult{EventCount: count}, err
	}
	run, err := r.runParsedEvents(count, outputOffset, post, postRunner)
	if err != nil {
		return decoderFrameWorkResidualStreamResult{EventCount: count, Run: run}, err
	}
	return decoderFrameWorkResidualStreamResult{EventCount: count, Run: run}, nil
}

func (r *decoderFrameWorkResidualStreamRunner) runRTPPayload(payload []byte, post internaldecoder.FrameWorkPostFilterFunc, postRunner internaldecoder.FrameWorkPostFilterRunner) (decoderFrameWorkResidualStreamResult, error) {
	return r.runRTPPayloadWithOutputOffset(payload, 0, post, postRunner)
}

func (r *decoderFrameWorkResidualStreamRunner) runRTPPayloadAfterLoss(payload []byte, post internaldecoder.FrameWorkPostFilterFunc, postRunner internaldecoder.FrameWorkPostFilterRunner) (decoderFrameWorkResidualStreamResult, error) {
	if err := r.ResetRTP(); err != nil {
		return decoderFrameWorkResidualStreamResult{}, err
	}
	return r.runRTPPayload(payload, post, postRunner)
}

func (r *decoderFrameWorkResidualStreamRunner) runRTPPayloadInto(result *decoderFrameWorkResidualStreamResult, payload []byte, post internaldecoder.FrameWorkPostFilterFunc, postRunner internaldecoder.FrameWorkPostFilterRunner) error {
	if result == nil {
		return internaldecoder.ErrInvalidFrameWorkState
	}
	next, err := r.runRTPPayloadWithOutputOffset(payload, result.Run.OutputCount, post, postRunner)
	if accErr := result.Accumulate(next); accErr != nil {
		return accErr
	}
	decoderFrameWorkResidualStreamBindResultOutputs(result, r.EventRunner.Outputs)
	return err
}

func (r *decoderFrameWorkResidualStreamRunner) runRTPPayloadAfterLossInto(result *decoderFrameWorkResidualStreamResult, payload []byte, post internaldecoder.FrameWorkPostFilterFunc, postRunner internaldecoder.FrameWorkPostFilterRunner) error {
	if result == nil {
		return internaldecoder.ErrInvalidFrameWorkState
	}
	if err := r.ResetRTP(); err != nil {
		return err
	}
	return r.runRTPPayloadInto(result, payload, post, postRunner)
}

func (r *decoderFrameWorkResidualStreamRunner) runRTPPayloadWithOutputOffset(payload []byte, outputOffset int, post internaldecoder.FrameWorkPostFilterFunc, postRunner internaldecoder.FrameWorkPostFilterRunner) (decoderFrameWorkResidualStreamResult, error) {
	if r == nil || r.Stream == nil {
		return decoderFrameWorkResidualStreamResult{}, internaldecoder.ErrInvalidFrameWorkState
	}
	if post != nil && postRunner != nil {
		return decoderFrameWorkResidualStreamResult{}, internaldecoder.ErrInvalidFrameWorkState
	}
	used, count, err := r.Stream.PushRTPPayload(r.RTPBuffer, r.RTPUsed, r.RTPSpans, r.Events, payload)
	r.RTPUsed = used
	if err != nil {
		return decoderFrameWorkResidualStreamResult{EventCount: count, RTPUsed: r.RTPUsed}, err
	}
	run, err := r.runParsedEvents(count, outputOffset, post, postRunner)
	if !r.Stream.InRTPFragment() {
		r.RTPUsed = 0
	}
	if err != nil {
		return decoderFrameWorkResidualStreamResult{EventCount: count, RTPUsed: r.RTPUsed, Run: run}, err
	}
	return decoderFrameWorkResidualStreamResult{EventCount: count, RTPUsed: r.RTPUsed, Run: run}, nil
}

func (r *decoderFrameWorkResidualStreamRunner) runRTPPayloads(payloads [][]byte, post internaldecoder.FrameWorkPostFilterFunc, postRunner internaldecoder.FrameWorkPostFilterRunner) (decoderFrameWorkResidualStreamResult, error) {
	if r == nil || r.Stream == nil {
		return decoderFrameWorkResidualStreamResult{}, internaldecoder.ErrInvalidFrameWorkState
	}
	if post != nil && postRunner != nil {
		return decoderFrameWorkResidualStreamResult{}, internaldecoder.ErrInvalidFrameWorkState
	}
	var result decoderFrameWorkResidualStreamResult
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

func (r *decoderFrameWorkResidualStreamRunner) runRTPPayloadsInto(result *decoderFrameWorkResidualStreamResult, payloads [][]byte, post internaldecoder.FrameWorkPostFilterFunc, postRunner internaldecoder.FrameWorkPostFilterRunner) error {
	if result == nil || r == nil || r.Stream == nil {
		return internaldecoder.ErrInvalidFrameWorkState
	}
	if post != nil && postRunner != nil {
		return internaldecoder.ErrInvalidFrameWorkState
	}
	for i := range payloads {
		if err := r.runRTPPayloadInto(result, payloads[i], post, postRunner); err != nil {
			return err
		}
	}
	return nil
}

func (r *decoderFrameWorkResidualStreamRunner) runParsedEvents(count int, outputOffset int, post internaldecoder.FrameWorkPostFilterFunc, postRunner internaldecoder.FrameWorkPostFilterRunner) (decoderFrameWorkResidualEventsResult, error) {
	if count < 0 || count > len(r.Events) {
		return decoderFrameWorkResidualEventsResult{}, ErrDecoderEventBufferTooSmall
	}
	eventRunner := r.EventRunner
	eventRunner.activeMeta = r.Stream.ActiveMetadata()
	if eventRunner.Outputs != nil {
		if outputOffset < 0 || outputOffset > len(eventRunner.Outputs) {
			return decoderFrameWorkResidualEventsResult{}, internalframe.ErrShortBuffer
		}
		eventRunner.Outputs = eventRunner.Outputs[outputOffset:]
	}
	sequence, _ := r.Stream.SequenceHeader()
	if postRunner != nil {
		return eventRunner.RunEventsWithPostFilterRunner(sequence, r.Events[:count], r.SideDataScratch, postRunner)
	}
	return eventRunner.RunEvents(sequence, r.Events[:count], r.SideDataScratch, post)
}

func decoderFrameWorkAccumulateResidualStreamResult(total *decoderFrameWorkResidualStreamResult, next decoderFrameWorkResidualStreamResult) {
	total.EventCount += next.EventCount
	total.RTPUsed = next.RTPUsed
	decoderFrameWorkAccumulateResidualEventsResult(&total.Run, next.Run)
}

func decoderFrameWorkAccumulateResidualEventsResult(total *decoderFrameWorkResidualEventsResult, next decoderFrameWorkResidualEventsResult) {
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

func decoderFrameWorkResidualStreamBindResultOutputs(result *decoderFrameWorkResidualStreamResult, outputs []*Frame) {
	if outputs == nil {
		return
	}
	result.Run.Outputs = outputs[:result.Run.OutputCount]
}

func decoderFrameWorkResidualStreamBindPlanMax(a decoderFrameWorkResidualEventBindPlan, b decoderFrameWorkResidualEventBindPlan) decoderFrameWorkResidualEventBindPlan {
	if b.HasEvent() {
		return b
	}
	if a.HasEvent() {
		return a
	}
	if b.EventIndex >= 0 {
		return b
	}
	return a
}

func decoderFrameWorkResidualLowOverheadEventLen(src []byte) (int, error) {
	it := internalobu.NewLowOverheadIterator(src)
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

func decoderFrameWorkResidualStreamSideDataScratchTooShort(scratch decoderFrameWorkSideDataScratch, size decoderFrameWorkSideDataScratchSize) bool {
	return decoderFrameWorkPostFilterScratchTooShort(scratch.CDEFIndexMap, size.CDEFIndexMap) ||
		decoderFrameWorkPostFilterScratchTooShort(scratch.CDEFReadMap, size.CDEFReadMap) ||
		decoderFrameWorkPostFilterScratchTooShort(scratch.LoopFilterMap, size.LoopFilterMap) ||
		decoderFrameWorkPostFilterScratchTooShort(scratch.RestorationRecords, size.RestorationRecords) ||
		decoderFrameWorkPostFilterScratchTooShort(scratch.RestorationBoundaryAbove, size.RestorationBoundaryAbove) ||
		decoderFrameWorkPostFilterScratchTooShort(scratch.RestorationBoundaryBelow, size.RestorationBoundaryBelow)
}

// decoderFrameWorkResliceOrNil slices s to length n when the caller provisioned
// at least n elements, else returns nil. It keeps the loop-filter mask scratch
// OPTIONAL through the stream-runner rebind: a caller that omits mask storage
// gets nil (edge-list loop filter) instead of a slice-bounds panic.
func decoderFrameWorkResliceOrNil[T any](s []T, n int) []T {
	if n <= 0 || len(s) < n {
		return nil
	}
	return s[:n]
}
