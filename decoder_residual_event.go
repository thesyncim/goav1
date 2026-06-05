package goav1

import (
	internaldecoder "github.com/thesyncim/goav1/internal/av1/decoder"
	internalthreading "github.com/thesyncim/goav1/internal/av1/threading"
)

// DecoderFrameWorkResidualEventRequest carries caller-owned state for running
// one decoder frame-work event with DecoderFrameWorkBatchResidualRunner.
type DecoderFrameWorkResidualEventRequest struct {
	State     *DecoderFrameWorkState
	Refs      *DecoderSurfaceReferences
	FramePool *FramePool

	Sequence SequenceHeader
	Event    DecoderEvent
	Align    int

	ReferenceSurfaces []int
	ReferenceFrames   []*Frame

	Workers int
	Spans   []TileSpan
	Jobs    []TileJob
	Batches []TileBatch

	Releases   []int
	WorkerPool *TileWorkerPool

	Runner   *DecoderFrameWorkBatchResidualRunner
	SideData *DecoderFrameWorkSideData
	Post     DecoderFrameWorkPostFilterFunc

	PostRunner DecoderFrameWorkPostFilterRunner
	Stats      *DecoderFrameWorkTileResidualStats

	External DecoderFrameWorkExternalReferenceRuntime
}

// DecoderFrameWorkResidualEventRunner stores the stable caller-owned buffers
// and pools used to run decoder frame-work events with a residual batch runner.
type DecoderFrameWorkResidualEventRunner struct {
	State     *DecoderFrameWorkState
	Refs      *DecoderSurfaceReferences
	FramePool *FramePool
	Align     int

	ReferenceSurfaces []int
	ReferenceFrames   []*Frame

	Workers int
	Spans   []TileSpan
	Jobs    []TileJob
	Batches []TileBatch

	Releases   []int
	WorkerPool *TileWorkerPool

	BatchRunner *DecoderFrameWorkBatchResidualRunner
	SideData    *DecoderFrameWorkSideData
	Stats       *DecoderFrameWorkTileResidualStats
	Outputs     []*Frame

	External DecoderFrameWorkExternalReferenceRuntime

	sideRunner DecoderFrameWorkSideDataRunner
}

// DecoderFrameWorkResidualEventsResult summarizes a batch of parsed decoder
// events run through a residual event runner. CompletedFrames counts decoded
// final tile frames; OutputCount counts display outputs, including
// show-existing-frame events that reuse a retained reference. Outputs aliases
// the runner's caller-owned output slots when they are provided.
type DecoderFrameWorkResidualEventsResult struct {
	Count            int
	Last             DecoderFrameWorkEventResult
	ExecutedTileWork int
	CompletedFrames  int
	OutputCount      int
	Outputs          []*Frame
	ReleaseCount     int
	Stats            DecoderFrameWorkTileResidualStats
}

// Accumulate adds next into r using the same result semantics as
// DecoderFrameWorkResidualEventRunner.RunEvents. The most recent non-empty
// event becomes Last, counters and residual stats are summed, and Outputs is
// left unchanged so callers can bind it to their own output arena.
func (r *DecoderFrameWorkResidualEventsResult) Accumulate(next DecoderFrameWorkResidualEventsResult) error {
	if r == nil {
		return ErrDecoderInvalidFrameWorkState
	}
	decoderFrameWorkAccumulateResidualEventsResult(r, next)
	return nil
}

// BindOutputs aliases Outputs to the first OutputCount slots in outputs.
func (r *DecoderFrameWorkResidualEventsResult) BindOutputs(outputs []*Frame) error {
	if r == nil {
		return ErrDecoderInvalidFrameWorkState
	}
	if len(outputs) < r.OutputCount {
		return ErrFrameShortBuffer
	}
	r.Outputs = outputs[:r.OutputCount]
	return nil
}

// DecoderFrameWorkResidualEventRuntime groups stable decoder state that is not
// carved out of residual-event scratch.
type DecoderFrameWorkResidualEventRuntime struct {
	State     *DecoderFrameWorkState
	Refs      *DecoderSurfaceReferences
	FramePool *FramePool
	Align     int

	ReferenceSurfaces []int
	ReferenceFrames   []*Frame
	Releases          []int

	WorkerPool *TileWorkerPool
	SideData   *DecoderFrameWorkSideData
	Stats      *DecoderFrameWorkTileResidualStats
	Outputs    []*Frame

	External DecoderFrameWorkExternalReferenceRuntime
}

// DecoderFrameWorkExternalReferenceRuntime configures the residual event
// runner to resolve and release references through a caller-owned global
// surface namespace. This is used by decode paths whose publishable reference
// surface may differ from the coded reconstruction surface, such as super-res
// publication into an upscaled output pool.
type DecoderFrameWorkExternalReferenceRuntime struct {
	Provider      DecoderFrameSurfaceProvider
	GlobalSurface func(local int) int
	Releaser      DecoderFrameSurfaceReleaser
	FrameContexts *DecoderSharedFrameContextStore
	PostPublisher DecoderFrameWorkPostFilterGlobalSurfacePublisher
}

// Enabled reports whether the runtime should use external reference
// resolution.
func (r DecoderFrameWorkExternalReferenceRuntime) Enabled() bool {
	return r.Provider != nil
}

// DecoderFrameWorkResidualEventBindPlan reports the sequence/event context
// callers should use when binding a reusable residual event runner for a parsed
// event list. EventIndex is -1 when the list has no event that needs residual
// event binding; Sequence still reflects sequence headers seen in the list.
type DecoderFrameWorkResidualEventBindPlan struct {
	Sequence   SequenceHeader
	Event      DecoderEvent
	EventIndex int
}

// HasEvent reports whether Event/EventIndex refer to a bind-relevant parsed
// event.
func (p DecoderFrameWorkResidualEventBindPlan) HasEvent() bool {
	return p.EventIndex >= 0 && decoderFrameWorkResidualEventBindCandidate(p.Event)
}

// DecoderFrameWorkResidualEventScratchSize reports the caller-owned scratch
// needed to run one residual event and capture its postfilter side data.
type DecoderFrameWorkResidualEventScratchSize struct {
	Runner   DecoderFrameWorkBatchResidualRunnerScratchSize
	SideData DecoderFrameWorkSideDataScratchSize
	Plan     DecoderTileWorkPlan
	Outputs  int
}

// DecoderFrameWorkResidualEventScratch carries typed caller-owned arenas for
// BindDecoderFrameWorkResidualEventRunner.
type DecoderFrameWorkResidualEventScratch struct {
	Runner   DecoderFrameWorkBatchResidualRunnerScratch
	SideData DecoderFrameWorkSideDataScratch

	Spans   []TileSpan
	Jobs    []TileJob
	Batches []TileBatch
}

// Check reports whether scratch is large enough for size without binding or
// mutating any caller-owned storage.
func (s DecoderFrameWorkResidualEventScratch) Check(size DecoderFrameWorkResidualEventScratchSize) error {
	if decoderFrameWorkResidualScratchTooShort(s.Spans, size.Plan.SpanCount) ||
		decoderFrameWorkResidualScratchTooShort(s.Jobs, size.Plan.JobCount) ||
		decoderFrameWorkResidualScratchTooShort(s.Batches, size.Plan.BatchCount) {
		return ErrFrameShortBuffer
	}
	if decoderFrameWorkResidualStreamSideDataScratchTooShort(s.SideData, size.SideData) {
		return ErrFrameShortBuffer
	}
	if err := decoderFrameWorkBatchResidualRunnerScratchErr(size.Runner, s.Runner); err != nil {
		return err
	}
	return nil
}

// Max returns per-arena maximum lengths and plan counts for reusable residual
// event scratch.
func (s DecoderFrameWorkResidualEventScratchSize) Max(other DecoderFrameWorkResidualEventScratchSize) DecoderFrameWorkResidualEventScratchSize {
	return DecoderFrameWorkResidualEventScratchSize{
		Runner:   s.Runner.Max(other.Runner),
		SideData: s.SideData.Max(other.SideData),
		Plan: DecoderTileWorkPlan{
			SpanCount:  max(s.Plan.SpanCount, other.Plan.SpanCount),
			JobCount:   max(s.Plan.JobCount, other.Plan.JobCount),
			BatchCount: max(s.Plan.BatchCount, other.Plan.BatchCount),
		},
		Outputs: max(s.Outputs, other.Outputs),
	}
}

// DecoderFrameWorkResidualEventsBindPlan scans a parsed event list and returns
// the sequence/event pair appropriate for BindDecoderFrameWorkResidualEventRunner
// or BindDecoderFrameWorkResidualStreamEventRunner. The returned event is the
// last event that can require residual side data, tile payload runner state, or
// output collection; this matches the max-scratch event-list sizing contract.
func DecoderFrameWorkResidualEventsBindPlan(sequence SequenceHeader, events []DecoderEvent) DecoderFrameWorkResidualEventBindPlan {
	plan := DecoderFrameWorkResidualEventBindPlan{
		Sequence:   sequence,
		EventIndex: -1,
	}
	for i := range events {
		event := &events[i]
		sequence = decoderFrameWorkResidualEventSequencePtr(sequence, event)
		if decoderFrameWorkResidualEventBindCandidatePtr(event) {
			plan.Sequence = sequence
			plan.Event = *event
			plan.EventIndex = i
		}
	}
	if plan.EventIndex < 0 {
		plan.Sequence = sequence
	}
	return plan
}

// BindDecoderFrameWorkResidualEventRunner binds a reusable event runner and
// frame-level side data from caller-owned scratch. batchRunner is caller-owned
// storage for the nested residual batch runner; the returned event runner keeps
// a pointer to it.
func BindDecoderFrameWorkResidualEventRunner(size DecoderFrameWorkResidualEventScratchSize, sequence SequenceHeader, event DecoderEvent, runtime DecoderFrameWorkResidualEventRuntime, scratch DecoderFrameWorkResidualEventScratch, batchRunner *DecoderFrameWorkBatchResidualRunner) (DecoderFrameWorkResidualEventRunner, DecoderFrameWorkSideData, error) {
	if size.Runner.Workers != 0 && batchRunner == nil {
		return DecoderFrameWorkResidualEventRunner{}, DecoderFrameWorkSideData{}, ErrThreadingInvalidBatch
	}
	if decoderFrameWorkResidualScratchTooShort(scratch.Spans, size.Plan.SpanCount) ||
		decoderFrameWorkResidualScratchTooShort(scratch.Jobs, size.Plan.JobCount) ||
		decoderFrameWorkResidualScratchTooShort(scratch.Batches, size.Plan.BatchCount) {
		return DecoderFrameWorkResidualEventRunner{}, DecoderFrameWorkSideData{}, ErrFrameShortBuffer
	}

	if size.Runner.Workers != 0 {
		boundRunner, err := BindDecoderFrameWorkBatchResidualRunner(size.Runner, scratch.Runner)
		if err != nil {
			return DecoderFrameWorkResidualEventRunner{}, DecoderFrameWorkSideData{}, err
		}
		*batchRunner = boundRunner
	}
	var side DecoderFrameWorkSideData
	if decoderFrameWorkResidualEventNeedsSideData(event) {
		var err error
		side, err = BindDecoderFrameWorkResidualEventSideData(sequence, event, scratch.SideData)
		if err != nil {
			return DecoderFrameWorkResidualEventRunner{}, DecoderFrameWorkSideData{}, err
		}
	}
	runner := DecoderFrameWorkResidualEventRunner{
		State:             runtime.State,
		Refs:              runtime.Refs,
		FramePool:         runtime.FramePool,
		Align:             runtime.Align,
		ReferenceSurfaces: runtime.ReferenceSurfaces,
		ReferenceFrames:   runtime.ReferenceFrames,
		Workers:           size.Runner.Workers,
		Spans:             scratch.Spans[:size.Plan.SpanCount],
		Jobs:              scratch.Jobs[:size.Plan.JobCount],
		Batches:           scratch.Batches[:size.Plan.BatchCount],
		Releases:          runtime.Releases,
		WorkerPool:        runtime.WorkerPool,
		BatchRunner:       batchRunner,
		SideData:          runtime.SideData,
		Stats:             runtime.Stats,
		Outputs:           runtime.Outputs,
		External:          runtime.External,
	}
	if runner.SideData != nil {
		runner.sideRunner = decoderFrameWorkResidualSideDataRunner{
			SideData:    runner.SideData,
			BatchRunner: runner.BatchRunner,
		}
	}
	return runner, side, nil
}

// Run plans and executes one decoder frame-work event using the runner's
// stable caller-owned state. The sequence, event, side-data, and postfilter
// callback remain per-event inputs.
func (r DecoderFrameWorkResidualEventRunner) Run(sequence SequenceHeader, event DecoderEvent, side *DecoderFrameWorkSideData, post DecoderFrameWorkPostFilterFunc) (DecoderFrameWorkEventResult, error) {
	return runDecoderFrameWorkEventWithResidualRunner(DecoderFrameWorkResidualEventRequest{
		State:             r.State,
		Refs:              r.Refs,
		FramePool:         r.FramePool,
		Sequence:          sequence,
		Event:             event,
		Align:             r.Align,
		ReferenceSurfaces: r.ReferenceSurfaces,
		ReferenceFrames:   r.ReferenceFrames,
		Workers:           r.Workers,
		Spans:             r.Spans,
		Jobs:              r.Jobs,
		Batches:           r.Batches,
		Releases:          r.Releases,
		WorkerPool:        r.WorkerPool,
		Runner:            r.BatchRunner,
		SideData:          side,
		Post:              post,
		Stats:             r.Stats,
		External:          r.External,
	}, r.boundSideDataRunner(side))
}

// RunWithPostFilterRunner plans and executes one decoder frame-work event using
// direct residual and postfilter runners, avoiding callback method values in the
// final-frame path.
func (r DecoderFrameWorkResidualEventRunner) RunWithPostFilterRunner(sequence SequenceHeader, event DecoderEvent, side *DecoderFrameWorkSideData, post DecoderFrameWorkPostFilterRunner) (DecoderFrameWorkEventResult, error) {
	return runDecoderFrameWorkEventWithResidualRunner(DecoderFrameWorkResidualEventRequest{
		State:             r.State,
		Refs:              r.Refs,
		FramePool:         r.FramePool,
		Sequence:          sequence,
		Event:             event,
		Align:             r.Align,
		ReferenceSurfaces: r.ReferenceSurfaces,
		ReferenceFrames:   r.ReferenceFrames,
		Workers:           r.Workers,
		Spans:             r.Spans,
		Jobs:              r.Jobs,
		Batches:           r.Batches,
		Releases:          r.Releases,
		WorkerPool:        r.WorkerPool,
		Runner:            r.BatchRunner,
		SideData:          side,
		PostRunner:        post,
		Stats:             r.Stats,
		External:          r.External,
	}, r.boundSideDataRunner(side))
}

func (r DecoderFrameWorkResidualEventRunner) boundSideDataRunner(side *DecoderFrameWorkSideData) DecoderFrameWorkSideDataRunner {
	if side == nil || side != r.SideData {
		return nil
	}
	return r.sideRunner
}

// RunEvents plans and executes parsed decoder events in order. r.SideData must
// point at caller-owned storage; it is rebound from caller-owned scratch for
// each frame-work event that can produce or consume residual side maps. If
// r.Stats is set, it receives aggregate stats for the whole event slice after a
// successful run.
func (r DecoderFrameWorkResidualEventRunner) RunEvents(sequence SequenceHeader, events []DecoderEvent, sideScratch DecoderFrameWorkSideDataScratch, post DecoderFrameWorkPostFilterFunc) (DecoderFrameWorkResidualEventsResult, error) {
	return r.runEvents(sequence, events, sideScratch, post, nil)
}

// RunEventsWithPostFilterRunner is RunEvents using a direct postfilter runner
// instead of a postfilter callback.
func (r DecoderFrameWorkResidualEventRunner) RunEventsWithPostFilterRunner(sequence SequenceHeader, events []DecoderEvent, sideScratch DecoderFrameWorkSideDataScratch, post DecoderFrameWorkPostFilterRunner) (DecoderFrameWorkResidualEventsResult, error) {
	return r.runEvents(sequence, events, sideScratch, nil, post)
}

// DecoderFrameWorkResidualEventScratchLen plans event tile work into
// caller-owned spans/jobs/batches and reports both residual-runner scratch and
// frame-level side-data scratch for the event.
func DecoderFrameWorkResidualEventScratchLen(sequence SequenceHeader, event DecoderEvent, workers int, spans []TileSpan, jobs []TileJob, batches []TileBatch) (DecoderFrameWorkResidualEventScratchSize, error) {
	return decoderFrameWorkResidualEventScratchLenPtr(sequence, &event, workers, spans, jobs, batches)
}

func decoderFrameWorkResidualEventScratchLenPtr(sequence SequenceHeader, event *DecoderEvent, workers int, spans []TileSpan, jobs []TileJob, batches []TileBatch) (DecoderFrameWorkResidualEventScratchSize, error) {
	size := DecoderFrameWorkResidualEventScratchSize{
		Outputs: decoderFrameWorkResidualEventOutputLenPtr(event),
	}
	if decoderFrameWorkResidualEventHasTilePayloadPtr(event) {
		runner, plan, err := decoderFrameWorkResidualEventRunnerScratchLenPtr(sequence, event, workers, spans, jobs, batches)
		if err != nil {
			return DecoderFrameWorkResidualEventScratchSize{}, err
		}
		size.Runner = runner
		size.Plan = plan
	}
	if decoderFrameWorkResidualEventNeedsSideDataPtr(event) {
		sideData, err := decoderFrameWorkResidualEventSideDataScratchLenPtr(sequence, event)
		if err != nil {
			return DecoderFrameWorkResidualEventScratchSize{}, err
		}
		size.SideData = sideData
	}
	return size, nil
}

// DecoderFrameWorkResidualEventsScratchLen reports reusable scratch for the
// largest tile-bearing event and largest frame-work side-data shape in events.
func DecoderFrameWorkResidualEventsScratchLen(sequence SequenceHeader, events []DecoderEvent, workers int, spans []TileSpan, jobs []TileJob, batches []TileBatch) (DecoderFrameWorkResidualEventScratchSize, error) {
	var size DecoderFrameWorkResidualEventScratchSize
	for i := range events {
		event := &events[i]
		sequence = decoderFrameWorkResidualEventSequencePtr(sequence, event)
		size.Outputs += decoderFrameWorkResidualEventOutputLenPtr(event)
		if decoderFrameWorkResidualEventNeedsSideDataPtr(event) {
			side, err := decoderFrameWorkResidualEventSideDataScratchLenPtr(sequence, event)
			if err != nil {
				return DecoderFrameWorkResidualEventScratchSize{}, err
			}
			size.SideData = size.SideData.Max(side)
		}
		if decoderFrameWorkResidualEventHasTilePayloadPtr(event) {
			runner, plan, err := decoderFrameWorkResidualEventRunnerScratchLenPtr(sequence, event, workers, spans, jobs, batches)
			if err != nil {
				return DecoderFrameWorkResidualEventScratchSize{}, err
			}
			size.Runner = size.Runner.Max(runner)
			size.Plan = decoderFrameWorkResidualEventPlanMax(size.Plan, plan)
		}
	}
	return size, nil
}

// DecoderFrameWorkResidualEventRunnerScratchLen plans event tile work into
// caller-owned spans/jobs/batches and reports the residual runner scratch
// required to execute the planned event.
func DecoderFrameWorkResidualEventRunnerScratchLen(sequence SequenceHeader, event DecoderEvent, workers int, spans []TileSpan, jobs []TileJob, batches []TileBatch) (DecoderFrameWorkBatchResidualRunnerScratchSize, DecoderTileWorkPlan, error) {
	return decoderFrameWorkResidualEventRunnerScratchLenPtr(sequence, &event, workers, spans, jobs, batches)
}

func decoderFrameWorkResidualEventRunnerScratchLenPtr(sequence SequenceHeader, event *DecoderEvent, workers int, spans []TileSpan, jobs []TileJob, batches []TileBatch) (DecoderFrameWorkBatchResidualRunnerScratchSize, DecoderTileWorkPlan, error) {
	plan, err := planDecoderTileWorkPtr(event, workers, spans, jobs, batches)
	if err != nil {
		return DecoderFrameWorkBatchResidualRunnerScratchSize{}, DecoderTileWorkPlan{}, err
	}
	batch := DecoderFrameWorkBatch{
		FrameWorkFrameContext: decoderFrameWorkResidualEventContextPtr(sequence, event),
		Jobs:                  jobs[:plan.JobCount],
	}
	size, err := DecoderFrameWorkBatchResidualRunnerScratchLen(batch, workers)
	if err != nil {
		return DecoderFrameWorkBatchResidualRunnerScratchSize{}, DecoderTileWorkPlan{}, err
	}
	return size, plan, nil
}

// DecoderFrameWorkResidualEventSideDataScratchLen reports frame-level side-data
// scratch for the event's frame size, CDEF, and restoration parameters.
func DecoderFrameWorkResidualEventSideDataScratchLen(sequence SequenceHeader, event DecoderEvent) (DecoderFrameWorkSideDataScratchSize, error) {
	return decoderFrameWorkResidualEventSideDataScratchLenPtr(sequence, &event)
}

func decoderFrameWorkResidualEventSideDataScratchLenPtr(sequence SequenceHeader, event *DecoderEvent) (DecoderFrameWorkSideDataScratchSize, error) {
	return DecoderFrameWorkSideDataScratchLen(sequence, event.FrameSize, event.CDEF, event.Restoration)
}

// BindDecoderFrameWorkResidualEventSideData binds frame-level side data using
// the event's frame size, CDEF, and restoration parameters.
func BindDecoderFrameWorkResidualEventSideData(sequence SequenceHeader, event DecoderEvent, scratch DecoderFrameWorkSideDataScratch) (DecoderFrameWorkSideData, error) {
	return bindDecoderFrameWorkResidualEventSideDataPtr(sequence, &event, scratch)
}

func bindDecoderFrameWorkResidualEventSideDataPtr(sequence SequenceHeader, event *DecoderEvent, scratch DecoderFrameWorkSideDataScratch) (DecoderFrameWorkSideData, error) {
	return BindDecoderFrameWorkSideData(sequence, event.FrameSize, event.CDEF, event.Restoration, scratch)
}

func (r DecoderFrameWorkResidualEventRunner) runEvents(sequence SequenceHeader, events []DecoderEvent, sideScratch DecoderFrameWorkSideDataScratch, post DecoderFrameWorkPostFilterFunc, postRunner DecoderFrameWorkPostFilterRunner) (DecoderFrameWorkResidualEventsResult, error) {
	if post != nil && postRunner != nil {
		return DecoderFrameWorkResidualEventsResult{}, ErrDecoderInvalidFrameWorkState
	}
	if r.Outputs != nil && len(r.Outputs) < decoderFrameWorkResidualEventsOutputLen(events) {
		return DecoderFrameWorkResidualEventsResult{}, ErrFrameShortBuffer
	}
	var result DecoderFrameWorkResidualEventsResult
	if r.Outputs != nil {
		result.Outputs = r.Outputs[:0]
	}
	single := r
	single.Stats = nil
	for i := range events {
		sequence = decoderFrameWorkResidualEventSequence(sequence, events[i])

		sidePtr := (*DecoderFrameWorkSideData)(nil)
		if decoderFrameWorkResidualEventNeedsSideData(events[i]) {
			if r.SideData == nil {
				return result, ErrDecoderInvalidFrameWorkState
			}
			sidePtr = r.SideData
			var err error
			*sidePtr, err = BindDecoderFrameWorkResidualEventSideData(sequence, events[i], sideScratch)
			if err != nil {
				return result, err
			}
		}

		var step DecoderFrameWorkEventResult
		var err error
		if postRunner != nil {
			step, err = single.RunWithPostFilterRunner(sequence, events[i], sidePtr, postRunner)
		} else {
			step, err = single.Run(sequence, events[i], sidePtr, post)
		}
		if err != nil {
			return result, err
		}
		eventStats, err := decoderFrameWorkResidualRunnerStats(single.BatchRunner)
		if err != nil {
			return result, err
		}

		result.Count++
		result.Last = step
		if step.Run.ExecutedTileWork {
			result.ExecutedTileWork++
		}
		if decoderFrameWorkResidualResultOutputsFrame(events[i], step) {
			if r.Outputs != nil {
				if result.OutputCount >= len(r.Outputs) {
					return result, ErrFrameShortBuffer
				}
				r.Outputs[result.OutputCount] = step.Output
			}
			result.OutputCount++
			if r.Outputs != nil {
				result.Outputs = r.Outputs[:result.OutputCount]
			}
		}
		if step.Run.CompletedFrame {
			result.CompletedFrames++
		}
		result.ReleaseCount += int(step.Run.ReleaseCount)
		decoderFrameWorkAccumulateResidualStats(&result.Stats, eventStats)
	}
	if r.Stats != nil {
		*r.Stats = result.Stats
	}
	return result, nil
}

func decoderFrameWorkResidualEventsOutputLen(events []DecoderEvent) int {
	count := 0
	for i := range events {
		count += decoderFrameWorkResidualEventOutputLen(events[i])
	}
	return count
}

func decoderFrameWorkResidualEventOutputLen(event DecoderEvent) int {
	if DecoderEventOutputsFrame(event) {
		return 1
	}
	return 0
}

func decoderFrameWorkResidualEventOutputLenPtr(event *DecoderEvent) int {
	if event == nil {
		return 0
	}
	return decoderFrameWorkResidualEventOutputLen(*event)
}

// RunDecoderFrameWorkEventWithResidualRunner plans and executes one decoder
// frame-work event using a caller-owned residual runner. If SideData is set, it
// is attached to both the active frame-work state and the residual runner after
// planning and before tile work runs.
func RunDecoderFrameWorkEventWithResidualRunner(req DecoderFrameWorkResidualEventRequest) (DecoderFrameWorkEventResult, error) {
	return runDecoderFrameWorkEventWithResidualRunner(req, nil)
}

func runDecoderFrameWorkEventWithResidualRunner(req DecoderFrameWorkResidualEventRequest, sideRunner DecoderFrameWorkSideDataRunner) (DecoderFrameWorkEventResult, error) {
	if decoderFrameWorkResidualEventHasTilePayload(req.Event) && req.Runner == nil {
		return DecoderFrameWorkEventResult{}, ErrThreadingInvalidBatch
	}
	if req.Post != nil && req.PostRunner != nil {
		return DecoderFrameWorkEventResult{}, ErrDecoderInvalidFrameWorkState
	}
	event := req.Event
	event.SequenceHeader = req.Sequence

	if event.Kind == DecoderEventTileList {
		if event.TileListErr != nil {
			return DecoderFrameWorkEventResult{}, event.TileListErr
		}
		return DecoderFrameWorkEventResult{}, ErrDecoderUnsupportedTileList
	}

	if req.Runner != nil {
		if err := req.Runner.ResetStats(); err != nil {
			return DecoderFrameWorkEventResult{}, err
		}
	}
	if req.Stats != nil {
		*req.Stats = DecoderFrameWorkTileResidualStats{}
	}

	if req.External.Enabled() {
		return runDecoderFrameWorkEventWithExternalResidualRunner(req, event, sideRunner)
	}

	step, output, err := req.State.PlanEvent(req.Refs, req.FramePool, req.Sequence, event, req.Align, req.ReferenceSurfaces, req.Workers, req.Spans, req.Jobs, req.Batches, req.Releases)
	if err != nil {
		return DecoderFrameWorkEventResult{}, err
	}

	tile, referenceCount, hasTile, err := decoderFrameWorkEventStepTile(step)
	if err != nil {
		return DecoderFrameWorkEventResult{}, err
	}
	if hasTile && req.SideData != nil {
		if tile.JobCount != 0 {
			if err := SetDecoderFrameWorkBatchResidualRunnerSideData(req.Runner, *req.SideData); err != nil {
				return DecoderFrameWorkEventResult{}, err
			}
		}
		if err := SetDecoderFrameWorkSideData(req.State, *req.SideData); err != nil {
			return DecoderFrameWorkEventResult{}, err
		}
	}

	if step.Kind == DecoderFrameWorkStepShowExisting {
		output, err = req.FramePool.Frame(step.ShowExisting.Surface)
		if err != nil {
			return DecoderFrameWorkEventResult{}, err
		}
	}

	var references []*Frame
	if hasTile && tile.JobCount != 0 {
		if output == nil {
			surface, err := decoderFrameWorkEventStepSurface(step)
			if err != nil {
				return DecoderFrameWorkEventResult{}, err
			}
			output, err = req.FramePool.Frame(surface)
			if err != nil {
				return DecoderFrameWorkEventResult{}, err
			}
		}
		if referenceCount != 0 {
			referenceCountN := int(referenceCount)
			if len(req.ReferenceSurfaces) < referenceCountN {
				return DecoderFrameWorkEventResult{}, ErrDecoderSurfaceReferenceBufferTooSmall
			}
			if step.Kind == DecoderFrameWorkStepTile {
				count, err := req.Refs.FrameReferences(event, req.ReferenceSurfaces)
				if err != nil {
					return DecoderFrameWorkEventResult{}, err
				}
				if count != referenceCountN {
					return DecoderFrameWorkEventResult{}, ErrDecoderInvalidFrameWorkStep
				}
			}
			count, err := ResolveDecoderFrameReferences(req.FramePool, req.ReferenceSurfaces[:referenceCountN], req.ReferenceFrames)
			if err != nil {
				return DecoderFrameWorkEventResult{}, err
			}
			if count != referenceCountN {
				return DecoderFrameWorkEventResult{}, ErrDecoderInvalidFrameWorkStep
			}
			references = req.ReferenceFrames[:referenceCountN]
		}
	}

	var run DecoderFrameWorkStepResult
	if req.WorkerPool != nil && req.WorkerPool.WorkerCount() == 1 {
		run, err = runDecoderFrameWorkEventSingleWorkerResidualRunner(req, event, step, output, references)
	} else if req.PostRunner != nil {
		run, err = req.State.RunStepWithPayloadContextAndPostFilterRunners(req.Refs, req.FramePool, event, step, req.WorkerPool, output, references, event.Unit.Payload, req.Jobs, req.Batches, req.Releases, req.Runner, req.PostRunner)
	} else {
		run, err = req.State.RunStepWithPayloadContextAndPostFilterRunner(req.Refs, req.FramePool, event, step, req.WorkerPool, output, references, event.Unit.Payload, req.Jobs, req.Batches, req.Releases, req.Runner, req.Post)
	}
	stats, statsErr := decoderFrameWorkResidualRunnerStats(req.Runner)
	if req.Stats != nil {
		*req.Stats = stats
	}
	if err != nil {
		return DecoderFrameWorkEventResult{}, err
	}
	if statsErr != nil {
		return DecoderFrameWorkEventResult{}, statsErr
	}
	output = decoderFrameWorkResidualPostFilterOutput(output, run, req.PostRunner)
	return DecoderFrameWorkEventResult{
		Step:           step,
		Output:         output,
		ReferenceCount: referenceCount,
		Run:            run,
	}, nil
}

func runDecoderFrameWorkEventSingleWorkerResidualRunner(req DecoderFrameWorkResidualEventRequest, event DecoderEvent, step DecoderFrameWorkStep, output *Frame, references []*Frame) (DecoderFrameWorkStepResult, error) {
	prepared, err := internaldecoder.PrepareFrameWorkStepWithPayloadContext(req.State, event, step, output, references, event.Unit.Payload, req.Jobs, req.Batches)
	if err != nil {
		return DecoderFrameWorkStepResult{}, err
	}
	if !prepared.HasTileWork {
		return internaldecoder.CompleteFrameWorkPreparedPayloadStep(req.State, req.Refs, req.FramePool, event, step, prepared, output, req.Releases, false, req.Post)
	}
	batches := req.Batches[:prepared.BatchCount]
	jobs := req.Jobs[:prepared.JobCount]
	if err := internalthreading.ValidateSingleWorkerFrameWorkBatches(req.WorkerPool, batches, jobs); err != nil {
		return DecoderFrameWorkStepResult{}, err
	}
	var firstErr error
	for i := range batches {
		batch := batches[i]
		batchJobs, err := batch.JobSlice(jobs)
		if err != nil {
			return DecoderFrameWorkStepResult{}, err
		}
		ctx := prepared.Base
		ctx.Batch = batch
		ctx.Jobs = batchJobs
		if err := req.Runner.RunPtr(&ctx); firstErr == nil && err != nil {
			firstErr = err
		}
	}
	if firstErr != nil {
		return DecoderFrameWorkStepResult{}, firstErr
	}
	if req.PostRunner != nil {
		return internaldecoder.CompleteFrameWorkPreparedPayloadStepRunner(req.State, req.Refs, req.FramePool, event, step, prepared, output, req.Releases, true, req.PostRunner)
	}
	return internaldecoder.CompleteFrameWorkPreparedPayloadStep(req.State, req.Refs, req.FramePool, event, step, prepared, output, req.Releases, true, req.Post)
}

func runDecoderFrameWorkEventWithExternalResidualRunner(req DecoderFrameWorkResidualEventRequest, event DecoderEvent, sideRunner DecoderFrameWorkSideDataRunner) (DecoderFrameWorkEventResult, error) {
	postRunner := req.PostRunner
	if req.Post != nil {
		postRunner = decoderFrameWorkResidualPostFuncRunner{fn: req.Post}
	}
	if sideRunner == nil && req.SideData != nil {
		sideRunner = decoderFrameWorkResidualSideDataRunner{
			SideData:    req.SideData,
			BatchRunner: req.Runner,
		}
	}
	result, err := req.State.RunEventWithContextAndExternalReferencesWithPostPublisher(
		req.Refs,
		req.FramePool,
		req.Sequence,
		event,
		req.Align,
		req.ReferenceSurfaces,
		req.ReferenceFrames,
		req.Workers,
		req.Spans,
		req.Jobs,
		req.Batches,
		req.Releases,
		req.WorkerPool,
		req.External.Provider,
		req.External.GlobalSurface,
		req.External.Releaser,
		req.External.FrameContexts,
		sideRunner,
		req.Runner,
		postRunner,
		req.External.PostPublisher,
	)
	stats, statsErr := decoderFrameWorkResidualRunnerStats(req.Runner)
	if req.Stats != nil {
		*req.Stats = stats
	}
	if err != nil {
		return DecoderFrameWorkEventResult{}, err
	}
	if statsErr != nil {
		return DecoderFrameWorkEventResult{}, statsErr
	}
	result.Output = decoderFrameWorkResidualPostFilterOutput(result.Output, result.Run, postRunner)
	return result, nil
}

type decoderFrameWorkResidualPostFuncRunner struct {
	fn DecoderFrameWorkPostFilterFunc
}

func (r decoderFrameWorkResidualPostFuncRunner) Apply(ctx DecoderFrameWorkPostFilterContext) error {
	if r.fn == nil {
		return nil
	}
	return r.fn(ctx)
}

type decoderFrameWorkResidualSideDataRunner struct {
	SideData    *DecoderFrameWorkSideData
	BatchRunner *DecoderFrameWorkBatchResidualRunner
}

func (r decoderFrameWorkResidualSideDataRunner) BindFrameWorkSideData(state *DecoderFrameWorkState, _ DecoderFrameWorkBatch) error {
	if r.SideData == nil {
		return nil
	}
	if r.BatchRunner != nil {
		if err := SetDecoderFrameWorkBatchResidualRunnerSideData(r.BatchRunner, *r.SideData); err != nil {
			return err
		}
	}
	return SetDecoderFrameWorkSideData(state, *r.SideData)
}

func decoderFrameWorkResidualRunnerStats(runner *DecoderFrameWorkBatchResidualRunner) (DecoderFrameWorkTileResidualStats, error) {
	if runner == nil {
		return DecoderFrameWorkTileResidualStats{}, nil
	}
	return runner.TotalStats()
}

func decoderFrameWorkResidualResultOutputsFrame(event DecoderEvent, result DecoderFrameWorkEventResult) bool {
	if result.Output == nil {
		return false
	}
	if result.Step.Kind == DecoderFrameWorkStepShowExisting {
		return true
	}
	// A completed coded frame is only emitted to the output stream when its
	// header sets show_frame. Not-shown frames (e.g. alt-ref) are decoded for
	// their reference-buffer side effects but surface later via
	// show_existing_frame, so they must stay out of Outputs. DecoderEventOutputsFrame
	// also gates the per-event output-slot sizing, keeping the two in lockstep.
	return result.Run.CompletedFrame && DecoderEventOutputsFrame(event)
}

func decoderFrameWorkResidualPostFilterOutput(output *Frame, run DecoderFrameWorkStepResult, post DecoderFrameWorkPostFilterRunner) *Frame {
	if !run.CompletedFrame || post == nil {
		return output
	}
	provider, ok := post.(DecoderFrameWorkPostFilterOutputProvider)
	if !ok {
		return output
	}
	postOutput, ok := provider.PostFilterOutput()
	if !ok || postOutput == nil {
		return output
	}
	return postOutput
}

func decoderFrameWorkResidualEventSequence(sequence SequenceHeader, event DecoderEvent) SequenceHeader {
	return decoderFrameWorkResidualEventSequencePtr(sequence, &event)
}

func decoderFrameWorkResidualEventSequencePtr(sequence SequenceHeader, event *DecoderEvent) SequenceHeader {
	if event == nil {
		return sequence
	}
	switch event.Kind {
	case DecoderEventSequenceHeader:
		if decoderFrameWorkResidualEventHasSequencePtr(event) {
			return event.SequenceHeader
		}
		return sequence
	case DecoderEventFrameHeader,
		DecoderEventRedundantFrameHeader,
		DecoderEventFrame,
		DecoderEventTileGroup,
		DecoderEventExistingFrame:
		if decoderFrameWorkResidualEventHasSequencePtr(event) {
			return event.SequenceHeader
		}
		return sequence
	default:
		return sequence
	}
}

func decoderFrameWorkResidualEventHasSequence(event DecoderEvent) bool {
	return decoderFrameWorkResidualEventHasSequencePtr(&event)
}

func decoderFrameWorkResidualEventHasSequencePtr(event *DecoderEvent) bool {
	if event == nil {
		return false
	}
	return event.SequenceHeader.ColorConfig.BitDepth != 0
}

func decoderFrameWorkResidualEventNeedsSideData(event DecoderEvent) bool {
	return decoderFrameWorkResidualEventNeedsSideDataPtr(&event)
}

func decoderFrameWorkResidualEventNeedsSideDataPtr(event *DecoderEvent) bool {
	if event == nil {
		return false
	}
	switch event.Kind {
	case DecoderEventFrameHeader, DecoderEventFrame, DecoderEventTileGroup:
		return true
	default:
		return false
	}
}

func decoderFrameWorkResidualEventHasTilePayload(event DecoderEvent) bool {
	return decoderFrameWorkResidualEventHasTilePayloadPtr(&event)
}

func decoderFrameWorkResidualEventHasTilePayloadPtr(event *DecoderEvent) bool {
	if event == nil {
		return false
	}
	switch event.Kind {
	case DecoderEventFrame, DecoderEventTileGroup:
		return true
	default:
		return false
	}
}

func decoderFrameWorkResidualEventBindCandidate(event DecoderEvent) bool {
	return decoderFrameWorkResidualEventBindCandidatePtr(&event)
}

func decoderFrameWorkResidualEventBindCandidatePtr(event *DecoderEvent) bool {
	return decoderFrameWorkResidualEventNeedsSideDataPtr(event) ||
		decoderFrameWorkResidualEventHasTilePayloadPtr(event) ||
		(event != nil && DecoderEventOutputsFrame(*event))
}

func decoderFrameWorkResidualEventPlanMax(a DecoderTileWorkPlan, b DecoderTileWorkPlan) DecoderTileWorkPlan {
	return DecoderTileWorkPlan{
		SpanCount:  max(a.SpanCount, b.SpanCount),
		JobCount:   max(a.JobCount, b.JobCount),
		BatchCount: max(a.BatchCount, b.BatchCount),
	}
}

func decoderFrameWorkEventStepTile(step DecoderFrameWorkStep) (DecoderTileWorkPlan, uint8, bool, error) {
	switch step.Kind {
	case DecoderFrameWorkStepIgnored, DecoderFrameWorkStepDropped, DecoderFrameWorkStepShowExisting:
		return DecoderTileWorkPlan{}, 0, false, nil
	case DecoderFrameWorkStepBegin:
		return step.Begin.Tile, step.Begin.ReferenceCount, true, nil
	case DecoderFrameWorkStepTile:
		return step.Tile.Tile, step.Tile.ReferenceCount, true, nil
	default:
		return DecoderTileWorkPlan{}, 0, false, ErrDecoderInvalidTileWork
	}
}

func decoderFrameWorkResidualEventContext(sequence SequenceHeader, event DecoderEvent) DecoderFrameWorkFrameContext {
	return decoderFrameWorkResidualEventContextPtr(sequence, &event)
}

func decoderFrameWorkResidualEventContextPtr(sequence SequenceHeader, event *DecoderEvent) DecoderFrameWorkFrameContext {
	if event == nil {
		return DecoderFrameWorkFrameContext{
			Sequence: DecoderFrameWorkSequenceContextFromHeader(sequence),
		}
	}
	return DecoderFrameWorkFrameContext{
		Sequence:            DecoderFrameWorkSequenceContextFromHeader(sequence),
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

func decoderFrameWorkEventStepSurface(step DecoderFrameWorkStep) (int, error) {
	switch step.Kind {
	case DecoderFrameWorkStepBegin:
		return step.Begin.Surface, nil
	case DecoderFrameWorkStepTile:
		return step.Tile.Surface, nil
	case DecoderFrameWorkStepShowExisting:
		return step.ShowExisting.Surface, nil
	default:
		return -1, ErrDecoderInvalidFrameWorkStep
	}
}
