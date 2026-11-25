package goav1

import (
	"fmt"

	internalframe "github.com/thesyncim/goav1/internal/av1/frame"
	internalparser "github.com/thesyncim/goav1/internal/av1/parser"
	internaltile "github.com/thesyncim/goav1/internal/av1/tile"

	internaldecoder "github.com/thesyncim/goav1/internal/av1/decoder"
	internalthreading "github.com/thesyncim/goav1/internal/av1/threading"
)

// decoderFrameWorkResidualEventRequest carries caller-owned state for running
// one decoder frame-work event with decoderFrameWorkBatchResidualRunner.
type decoderFrameWorkResidualEventRequest struct {
	State     *internaldecoder.FrameWorkState
	Refs      *internaldecoder.SurfaceReferences
	FramePool *internalframe.Pool
	// TileListOutputPool optionally receives display-only tile-list mosaics and
	// temporary tile-list entry reconstructions. When nil, FramePool is used for
	// both entry reconstruction and the final mosaic output.
	TileListOutputPool *internalframe.Pool

	Sequence SequenceHeader
	Event    internaldecoder.Event
	Align    int

	ReferenceSurfaces []int
	ReferenceFrames   []*Frame

	Workers int
	Spans   []internalparser.TileSpan
	Jobs    []internaltile.Job
	Batches []internalthreading.Batch

	Releases   []int
	WorkerPool *internalthreading.Pool

	Runner   *decoderFrameWorkBatchResidualRunner
	SideData *decoderFrameWorkSideData
	Post     internaldecoder.FrameWorkPostFilterFunc

	PostRunner internaldecoder.FrameWorkPostFilterRunner
	Stats      *internalthreading.FrameWorkTileResidualStats

	External decoderFrameWorkExternalReferenceRuntime
}

// decoderFrameWorkResidualEventRunner stores the stable caller-owned buffers
// and pools used to run decoder frame-work events with a residual batch runner.
type decoderFrameWorkResidualEventRunner struct {
	State     *internaldecoder.FrameWorkState
	Refs      *internaldecoder.SurfaceReferences
	FramePool *internalframe.Pool
	// TileListOutputPool optionally receives display-only tile-list mosaics and
	// temporary tile-list entry reconstructions. When nil, FramePool is used for
	// both entry reconstruction and the final mosaic output.
	TileListOutputPool *internalframe.Pool
	Align              int

	ReferenceSurfaces []int
	ReferenceFrames   []*Frame

	Workers int
	Spans   []internalparser.TileSpan
	Jobs    []internaltile.Job
	Batches []internalthreading.Batch

	Releases   []int
	WorkerPool *internalthreading.Pool

	BatchRunner *decoderFrameWorkBatchResidualRunner
	SideData    *decoderFrameWorkSideData
	Stats       *internalthreading.FrameWorkTileResidualStats
	Outputs     []*Frame

	External decoderFrameWorkExternalReferenceRuntime

	sideRunner internaldecoder.FrameWorkSideDataRunner

	compactState *internalthreading.FrameWorkLoopFilterCompactState
	// lfLevels, when set, is bound per frame so the tile walk writes the
	// loop-filter level cache at decode time (dav1d_create_lf_mask_*).
	lfLevels *internalthreading.FrameWorkLoopFilterLevelTable
	// lfTileEdges collects dav1d's tile-boundary context backups so multi-tile
	// masks are fixed without re-deriving neighbours from the record map.
	lfTileEdges *internalthreading.FrameWorkLoopFilterTileEdges
	// lfNoSkip holds dav1d's CDEF noskip_mask written by the tile walk.
	lfNoSkip          *internalthreading.FrameWorkLoopFilterNoSkip
	compactNoSkip     []byte
	compactPostOwner  *decoderFrameWorkReusableSupportedPostFilterRunner
	compactBatchOwner *decoderFrameWorkBatchResidualRunner
	compactSideOwner  *decoderFrameWorkSideData

	// fp, when set, runs frame-threadable events through the frame-threading
	// pipeline and drains it before any other event; outputs then leave through
	// the pipeline in submission order instead of Outputs.
	fp *decoderFramePipeline

	activeMeta FrameMetadata
}

// decoderFrameWorkResidualEventsResult summarizes a batch of parsed decoder
// events run through a residual event runner. CompletedFrames counts decoded
// final tile frames; OutputCount counts display outputs, including
// show-existing-frame events that reuse a retained reference. Outputs aliases
// the runner's caller-owned output slots when they are provided.
type decoderFrameWorkResidualEventsResult struct {
	Count            int
	Last             internaldecoder.FrameWorkEventResult
	ExecutedTileWork int
	CompletedFrames  int
	OutputCount      int
	Outputs          []*Frame
	ReleaseCount     int
	Stats            internalthreading.FrameWorkTileResidualStats
}

// Accumulate adds next into r using the same result semantics as
// decoderFrameWorkResidualEventRunner.RunEvents. The most recent non-empty
// event becomes Last, counters and residual stats are summed, and Outputs is
// left unchanged so callers can bind it to their own output arena.
func (r *decoderFrameWorkResidualEventsResult) Accumulate(next decoderFrameWorkResidualEventsResult) error {
	if r == nil {
		return internaldecoder.ErrInvalidFrameWorkState
	}
	decoderFrameWorkAccumulateResidualEventsResult(r, next)
	return nil
}

// BindOutputs aliases Outputs to the first OutputCount slots in outputs.
func (r *decoderFrameWorkResidualEventsResult) BindOutputs(outputs []*Frame) error {
	if r == nil {
		return internaldecoder.ErrInvalidFrameWorkState
	}
	if len(outputs) < r.OutputCount {
		return internalframe.ErrShortBuffer
	}
	r.Outputs = outputs[:r.OutputCount]
	return nil
}

// decoderFrameWorkResidualEventRuntime groups stable decoder state that is not
// carved out of residual-event scratch.
type decoderFrameWorkResidualEventRuntime struct {
	State     *internaldecoder.FrameWorkState
	Refs      *internaldecoder.SurfaceReferences
	FramePool *internalframe.Pool
	// TileListOutputPool optionally receives display-only tile-list mosaics and
	// temporary tile-list entry reconstructions. When nil, FramePool is used for
	// both entry reconstruction and the final mosaic output.
	TileListOutputPool *internalframe.Pool
	Align              int

	ReferenceSurfaces []int
	ReferenceFrames   []*Frame
	Releases          []int

	WorkerPool *internalthreading.Pool
	SideData   *decoderFrameWorkSideData
	Stats      *internalthreading.FrameWorkTileResidualStats
	Outputs    []*Frame

	External decoderFrameWorkExternalReferenceRuntime
}

// decoderFrameWorkExternalReferenceRuntime configures the residual event
// runner to resolve and release references through a caller-owned global
// surface namespace. This is used by decode paths whose publishable reference
// surface may differ from the coded reconstruction surface, such as super-res
// publication into an upscaled output pool.
type decoderFrameWorkExternalReferenceRuntime struct {
	Provider      internaldecoder.FrameSurfaceProvider
	GlobalSurface func(local int) int
	Releaser      internaldecoder.FrameSurfaceReleaser
	FrameContexts *internaldecoder.SharedFrameContextStore
	PostPublisher internaldecoder.FrameWorkPostFilterGlobalSurfacePublisher
}

// Enabled reports whether the runtime should use external reference
// resolution.
func (r decoderFrameWorkExternalReferenceRuntime) Enabled() bool {
	return r.Provider != nil
}

// decoderFrameWorkResidualEventBindPlan reports the sequence/event context
// callers should use when binding a reusable residual event runner for a parsed
// event list. EventIndex is -1 when the list has no event that needs residual
// event binding; Sequence still reflects sequence headers seen in the list.
type decoderFrameWorkResidualEventBindPlan struct {
	Sequence   SequenceHeader
	Event      internaldecoder.Event
	EventIndex int
	// Dropped reports that Event is the header of a frame dropped incomplete
	// (restored to its parsed kind); it binds only a stream that decodes no
	// frame.
	Dropped bool
}

// prefer reports whether next should replace p as a stream's bind plan.
func (p decoderFrameWorkResidualEventBindPlan) prefer(next decoderFrameWorkResidualEventBindPlan) bool {
	return next.HasEvent() && (!next.Dropped || !p.HasEvent() || p.Dropped)
}

// HasEvent reports whether Event/EventIndex refer to a bind-relevant parsed
// event.
func (p decoderFrameWorkResidualEventBindPlan) HasEvent() bool {
	return p.EventIndex >= 0 && decoderFrameWorkResidualEventBindCandidate(&p.Event)
}

// decoderFrameWorkResidualEventScratchSize reports the caller-owned scratch
// needed to run one residual event and capture its postfilter side data.
type decoderFrameWorkResidualEventScratchSize struct {
	Runner   decoderFrameWorkBatchResidualRunnerScratchSize
	SideData decoderFrameWorkSideDataScratchSize
	Plan     internaldecoder.TileWorkPlan
	Outputs  int
}

// decoderFrameWorkResidualEventScratch carries typed caller-owned arenas for
// bindDecoderFrameWorkResidualEventRunner.
type decoderFrameWorkResidualEventScratch struct {
	Runner   decoderFrameWorkBatchResidualRunnerScratch
	SideData decoderFrameWorkSideDataScratch

	Spans   []internalparser.TileSpan
	Jobs    []internaltile.Job
	Batches []internalthreading.Batch
}

// Check reports whether scratch is large enough for size without binding or
// mutating any caller-owned storage.
func (s decoderFrameWorkResidualEventScratch) Check(size decoderFrameWorkResidualEventScratchSize) error {
	if decoderFrameWorkResidualScratchTooShort(s.Spans, size.Plan.SpanCount) ||
		decoderFrameWorkResidualScratchTooShort(s.Jobs, size.Plan.JobCount) ||
		decoderFrameWorkResidualScratchTooShort(s.Batches, size.Plan.BatchCount) {
		return internalframe.ErrShortBuffer
	}
	if decoderFrameWorkResidualStreamSideDataScratchTooShort(s.SideData, size.SideData) {
		return internalframe.ErrShortBuffer
	}
	if err := decoderFrameWorkBatchResidualRunnerScratchErr(size.Runner, s.Runner); err != nil {
		return err
	}
	return nil
}

// Max returns per-arena maximum lengths and plan counts for reusable residual
// event scratch.
func (s decoderFrameWorkResidualEventScratchSize) Max(other decoderFrameWorkResidualEventScratchSize) decoderFrameWorkResidualEventScratchSize {
	return decoderFrameWorkResidualEventScratchSize{
		Runner:   s.Runner.Max(other.Runner),
		SideData: s.SideData.Max(other.SideData),
		Plan: internaldecoder.TileWorkPlan{
			SpanCount:  max(s.Plan.SpanCount, other.Plan.SpanCount),
			JobCount:   max(s.Plan.JobCount, other.Plan.JobCount),
			BatchCount: max(s.Plan.BatchCount, other.Plan.BatchCount),
		},
		Outputs: max(s.Outputs, other.Outputs),
	}
}

// decoderFrameWorkResidualEventsBindPlan scans a parsed event list and returns
// the sequence/event pair appropriate for bindDecoderFrameWorkResidualEventRunner
// or bindDecoderFrameWorkResidualStreamEventRunner. The returned event is the
// last event that can require residual side data, tile payload runner state, or
// output collection; this matches the max-scratch event-list sizing contract.
func decoderFrameWorkResidualEventsBindPlan(sequence SequenceHeader, events []internaldecoder.Event) decoderFrameWorkResidualEventBindPlan {
	plan := decoderFrameWorkResidualEventBindPlan{
		Sequence:   sequence,
		EventIndex: -1,
	}
	for i := range events {
		event := &events[i]
		sequence = decoderFrameWorkResidualEventSequencePtr(sequence, event)
		dropped := event.DroppedKind != internaldecoder.EventIgnored
		if decoderFrameWorkResidualEventBindCandidatePtr(event) && (!dropped || plan.EventIndex < 0 || plan.Dropped) {
			plan.Sequence = sequence
			plan.Event = *event
			plan.EventIndex = i
			plan.Dropped = dropped
			if dropped {
				plan.Event.Kind = event.DroppedKind
			}
		}
	}
	if plan.EventIndex < 0 {
		plan.Sequence = sequence
	}
	return plan
}

// bindDecoderFrameWorkResidualEventRunner binds a reusable event runner and
// frame-level side data from caller-owned scratch. batchRunner is caller-owned
// storage for the nested residual batch runner; the returned event runner keeps
// a pointer to it.
func bindDecoderFrameWorkResidualEventRunner(size decoderFrameWorkResidualEventScratchSize, sequence SequenceHeader, event *internaldecoder.Event, runtime decoderFrameWorkResidualEventRuntime, scratch decoderFrameWorkResidualEventScratch, batchRunner *decoderFrameWorkBatchResidualRunner) (decoderFrameWorkResidualEventRunner, decoderFrameWorkSideData, error) {
	return bindDecoderFrameWorkResidualEventRunnerWith(size, sequence, event, runtime, scratch, batchRunner, false)
}

func bindDecoderFrameWorkResidualEventRunnerWith(size decoderFrameWorkResidualEventScratchSize, sequence SequenceHeader, event *internaldecoder.Event, runtime decoderFrameWorkResidualEventRuntime, scratch decoderFrameWorkResidualEventScratch, batchRunner *decoderFrameWorkBatchResidualRunner, maskOnlyLoopFilter bool) (decoderFrameWorkResidualEventRunner, decoderFrameWorkSideData, error) {
	if size.Runner.Workers != 0 && batchRunner == nil {
		return decoderFrameWorkResidualEventRunner{}, decoderFrameWorkSideData{}, internalthreading.ErrInvalidBatch
	}
	if decoderFrameWorkResidualScratchTooShort(scratch.Spans, size.Plan.SpanCount) ||
		decoderFrameWorkResidualScratchTooShort(scratch.Jobs, size.Plan.JobCount) ||
		decoderFrameWorkResidualScratchTooShort(scratch.Batches, size.Plan.BatchCount) {
		return decoderFrameWorkResidualEventRunner{}, decoderFrameWorkSideData{}, internalframe.ErrShortBuffer
	}

	if size.Runner.Workers != 0 {
		boundRunner, err := bindDecoderFrameWorkBatchResidualRunner(size.Runner, scratch.Runner)
		if err != nil {
			return decoderFrameWorkResidualEventRunner{}, decoderFrameWorkSideData{}, err
		}
		*batchRunner = boundRunner
	}
	var side decoderFrameWorkSideData
	if decoderFrameWorkResidualEventNeedsSideData(event) {
		var err error
		side, err = bindDecoderFrameWorkSideDataWith(sequence, event.FrameSize, event.CDEF, event.Restoration, scratch.SideData, nil, nil, nil, maskOnlyLoopFilter)
		if err != nil {
			return decoderFrameWorkResidualEventRunner{}, decoderFrameWorkSideData{}, err
		}
	}
	runner := decoderFrameWorkResidualEventRunner{
		State:              runtime.State,
		Refs:               runtime.Refs,
		FramePool:          runtime.FramePool,
		TileListOutputPool: runtime.TileListOutputPool,
		Align:              runtime.Align,
		ReferenceSurfaces:  runtime.ReferenceSurfaces,
		ReferenceFrames:    runtime.ReferenceFrames,
		Workers:            size.Runner.Workers,
		Spans:              scratch.Spans[:size.Plan.SpanCount],
		Jobs:               scratch.Jobs[:size.Plan.JobCount],
		Batches:            scratch.Batches[:size.Plan.BatchCount],
		Releases:           runtime.Releases,
		WorkerPool:         runtime.WorkerPool,
		BatchRunner:        batchRunner,
		SideData:           runtime.SideData,
		Stats:              runtime.Stats,
		Outputs:            runtime.Outputs,
		External:           runtime.External,
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
func (r decoderFrameWorkResidualEventRunner) Run(sequence SequenceHeader, event *internaldecoder.Event, side *decoderFrameWorkSideData, post internaldecoder.FrameWorkPostFilterFunc) (internaldecoder.FrameWorkEventResult, error) {
	return runDecoderFrameWorkEventWithSideRunner(decoderFrameWorkResidualEventRequest{
		State:              r.State,
		Refs:               r.Refs,
		FramePool:          r.FramePool,
		TileListOutputPool: r.TileListOutputPool,
		Sequence:           sequence,
		Event:              *event,
		Align:              r.Align,
		ReferenceSurfaces:  r.ReferenceSurfaces,
		ReferenceFrames:    r.ReferenceFrames,
		Workers:            r.Workers,
		Spans:              r.Spans,
		Jobs:               r.Jobs,
		Batches:            r.Batches,
		Releases:           r.Releases,
		WorkerPool:         r.WorkerPool,
		Runner:             r.BatchRunner,
		SideData:           side,
		Post:               post,
		Stats:              r.Stats,
		External:           r.External,
	}, r.boundSideDataRunner(side))
}

// RunWithPostFilterRunner plans and executes one decoder frame-work event using
// direct residual and postfilter runners, avoiding callback method values in the
// final-frame path.
func (r decoderFrameWorkResidualEventRunner) RunWithPostFilterRunner(sequence SequenceHeader, event *internaldecoder.Event, side *decoderFrameWorkSideData, post internaldecoder.FrameWorkPostFilterRunner) (internaldecoder.FrameWorkEventResult, error) {
	return runDecoderFrameWorkEventWithSideRunner(decoderFrameWorkResidualEventRequest{
		State:              r.State,
		Refs:               r.Refs,
		FramePool:          r.FramePool,
		TileListOutputPool: r.TileListOutputPool,
		Sequence:           sequence,
		Event:              *event,
		Align:              r.Align,
		ReferenceSurfaces:  r.ReferenceSurfaces,
		ReferenceFrames:    r.ReferenceFrames,
		Workers:            r.Workers,
		Spans:              r.Spans,
		Jobs:               r.Jobs,
		Batches:            r.Batches,
		Releases:           r.Releases,
		WorkerPool:         r.WorkerPool,
		Runner:             r.BatchRunner,
		SideData:           side,
		PostRunner:         post,
		Stats:              r.Stats,
		External:           r.External,
	}, r.boundSideDataRunner(side))
}

func (r decoderFrameWorkResidualEventRunner) boundSideDataRunner(side *decoderFrameWorkSideData) internaldecoder.FrameWorkSideDataRunner {
	if side == nil || side != r.SideData {
		return nil
	}
	if owned, ok := r.sideRunner.(decoderFrameWorkResidualSideDataRunner); ok && owned.SideData != side {
		return nil
	}
	return r.sideRunner
}

// RunEvents plans and executes parsed decoder events in order. r.SideData must
// point at caller-owned storage; it is rebound from caller-owned scratch for
// each frame-work event that can produce or consume residual side maps. If
// r.Stats is set, it receives aggregate stats for the whole event slice after a
// successful run.
func (r decoderFrameWorkResidualEventRunner) RunEvents(sequence SequenceHeader, events []internaldecoder.Event, sideScratch decoderFrameWorkSideDataScratch, post internaldecoder.FrameWorkPostFilterFunc) (decoderFrameWorkResidualEventsResult, error) {
	return r.runEvents(sequence, events, sideScratch, post, nil)
}

// RunEventsWithPostFilterRunner is RunEvents using a direct postfilter runner
// instead of a postfilter callback.
func (r decoderFrameWorkResidualEventRunner) RunEventsWithPostFilterRunner(sequence SequenceHeader, events []internaldecoder.Event, sideScratch decoderFrameWorkSideDataScratch, post internaldecoder.FrameWorkPostFilterRunner) (decoderFrameWorkResidualEventsResult, error) {
	return r.runEvents(sequence, events, sideScratch, nil, post)
}

// decoderFrameWorkResidualEventsScratchLen reports reusable scratch for the
// largest tile-bearing event and largest frame-work side-data shape in events.
func decoderFrameWorkResidualEventsScratchLen(sequence SequenceHeader, events []internaldecoder.Event, workers int, spans []internalparser.TileSpan, jobs []internaltile.Job, batches []internalthreading.Batch) (decoderFrameWorkResidualEventScratchSize, error) {
	var size decoderFrameWorkResidualEventScratchSize
	for i := range events {
		event := &events[i]
		sequence = decoderFrameWorkResidualEventSequencePtr(sequence, event)
		size.Outputs += decoderFrameWorkResidualEventOutputLenPtr(event)
		sideEvent := event
		if event.DroppedKind != internaldecoder.EventIgnored {
			// The header of a frame dropped incomplete can bind the runner
			// (decoderFrameWorkResidualEventsBindPlan), so its side data is
			// sized; its tiles are never split or decoded.
			restored := *event
			restored.Kind = event.DroppedKind
			sideEvent = &restored
		}
		if decoderFrameWorkResidualEventNeedsSideDataPtr(sideEvent) {
			side, err := decoderFrameWorkResidualEventSideDataScratchLenPtr(sequence, sideEvent)
			if err != nil {
				return decoderFrameWorkResidualEventScratchSize{}, err
			}
			size.SideData = size.SideData.Max(side)
		}
		if event.Kind == internaldecoder.EventTileList {
			runner, plan, err := decoderFrameWorkResidualTileListRunnerScratchLenPtr(sequence, event, workers, spans, jobs, batches)
			if err != nil {
				return decoderFrameWorkResidualEventScratchSize{}, err
			}
			size.Runner = size.Runner.Max(runner)
			size.Plan = decoderFrameWorkResidualEventPlanMax(size.Plan, plan)
		} else if decoderFrameWorkResidualEventHasTilePayloadPtr(event) {
			runner, plan, err := decoderFrameWorkResidualEventRunnerScratchLenPtr(sequence, event, workers, spans, jobs, batches)
			if err != nil {
				return decoderFrameWorkResidualEventScratchSize{}, err
			}
			size.Runner = size.Runner.Max(runner)
			size.Plan = decoderFrameWorkResidualEventPlanMax(size.Plan, plan)
		}
	}
	return size, nil
}

func decoderFrameWorkResidualEventRunnerScratchLenPtr(sequence SequenceHeader, event *internaldecoder.Event, workers int, spans []internalparser.TileSpan, jobs []internaltile.Job, batches []internalthreading.Batch) (decoderFrameWorkBatchResidualRunnerScratchSize, internaldecoder.TileWorkPlan, error) {
	if event != nil && event.Kind == internaldecoder.EventTileList {
		return decoderFrameWorkResidualTileListRunnerScratchLenPtr(sequence, event, workers, spans, jobs, batches)
	}
	plan, err := planDecoderTileWorkPtr(event, workers, spans, jobs, batches)
	if err != nil {
		return decoderFrameWorkBatchResidualRunnerScratchSize{}, internaldecoder.TileWorkPlan{}, err
	}
	batch := internaldecoder.FrameWorkBatch{
		FrameWorkFrameContext: decoderFrameWorkResidualEventContextPtr(sequence, event),
		Jobs:                  jobs[:plan.JobCount],
	}
	// A batch's worker index is its position in the plan (BuildBatches), so
	// the runner needs per-worker state for the planned batches only, not for
	// every configured worker: a single-tile stream decoded with 8 workers
	// keeps one tile scratch per frame context instead of eight.
	size, err := decoderFrameWorkBatchResidualRunnerScratchLen(batch, max(1, min(workers, plan.BatchCount)))
	if err != nil {
		return decoderFrameWorkBatchResidualRunnerScratchSize{}, internaldecoder.TileWorkPlan{}, err
	}
	return size, plan, nil
}

func decoderFrameWorkResidualTileListRunnerScratchLenPtr(sequence SequenceHeader, event *internaldecoder.Event, workers int, spans []internalparser.TileSpan, jobs []internaltile.Job, batches []internalthreading.Batch) (decoderFrameWorkBatchResidualRunnerScratchSize, internaldecoder.TileWorkPlan, error) {
	if event == nil || event.Kind != internaldecoder.EventTileList {
		return decoderFrameWorkBatchResidualRunnerScratchSize{}, internaldecoder.TileWorkPlan{}, internaldecoder.ErrInvalidSurfaceEvent
	}
	if event.TileListErr != nil {
		return decoderFrameWorkBatchResidualRunnerScratchSize{}, internaldecoder.TileWorkPlan{}, event.TileListErr
	}
	tileCount := event.TileList.TileCount()
	if tileCount <= 0 || tileCount != len(event.TileList.Entries) {
		return decoderFrameWorkBatchResidualRunnerScratchSize{}, internaldecoder.TileWorkPlan{}, internalparser.ErrTileListInvalidTileCount
	}
	var size decoderFrameWorkBatchResidualRunnerScratchSize
	var maxPlan internaldecoder.TileWorkPlan
	probe := *event
	probe.SequenceHeader = sequence
	for i := 0; i < tileCount; i++ {
		nextSize, plan, err := decoderTileListEntryDecodeRunnerScratchLen(&probe, i, workers, spans, jobs, batches)
		if err != nil {
			return decoderFrameWorkBatchResidualRunnerScratchSize{}, internaldecoder.TileWorkPlan{}, err
		}
		size = size.Max(nextSize)
		maxPlan = decoderFrameWorkResidualEventPlanMax(maxPlan, plan)
	}
	return size, maxPlan, nil
}

func decoderFrameWorkResidualEventSideDataScratchLenPtr(sequence SequenceHeader, event *internaldecoder.Event) (decoderFrameWorkSideDataScratchSize, error) {
	return decoderFrameWorkSideDataScratchLen(sequence, event.FrameSize, event.CDEF, event.Restoration)
}

// bindDecoderFrameWorkResidualEventSideData binds frame-level side data using
// the event's frame size, CDEF, and restoration parameters.
func bindDecoderFrameWorkResidualEventSideData(sequence SequenceHeader, event *internaldecoder.Event, scratch decoderFrameWorkSideDataScratch) (decoderFrameWorkSideData, error) {
	return bindDecoderFrameWorkResidualEventSideDataPtr(sequence, event, scratch)
}

func bindDecoderFrameWorkResidualEventSideDataPtr(sequence SequenceHeader, event *internaldecoder.Event, scratch decoderFrameWorkSideDataScratch) (decoderFrameWorkSideData, error) {
	return bindDecoderFrameWorkSideData(sequence, event.FrameSize, event.CDEF, event.Restoration, scratch)
}

func (r decoderFrameWorkResidualEventRunner) runEvents(sequence SequenceHeader, events []internaldecoder.Event, sideScratch decoderFrameWorkSideDataScratch, post internaldecoder.FrameWorkPostFilterFunc, postRunner internaldecoder.FrameWorkPostFilterRunner) (result decoderFrameWorkResidualEventsResult, runErr error) {
	defer func() {
		if runErr != nil {
			internalthreading.InvalidateFrameWorkLoopFilterCompactState(r.compactState)
		}
	}()
	if post != nil && postRunner != nil {
		return decoderFrameWorkResidualEventsResult{}, internaldecoder.ErrInvalidFrameWorkState
	}
	if r.Outputs != nil && len(r.Outputs) < decoderFrameWorkResidualEventsOutputLen(events) {
		return decoderFrameWorkResidualEventsResult{}, internalframe.ErrShortBuffer
	}
	if r.Outputs != nil {
		result.Outputs = r.Outputs[:0]
	}
	single := r
	single.Stats = nil
	for i := range events {
		sequence = decoderFrameWorkResidualEventSequence(sequence, &events[i])

		if r.fp != nil {
			if r.fp.eligible(&events[i]) {
				step, err := r.fp.runFrame(sequence, &events[i])
				if err != nil {
					return result, err
				}
				result.Count++
				result.Last = step
				result.ExecutedTileWork++
				result.CompletedFrames++
				result.ReleaseCount += int(step.Run.ReleaseCount)
				continue
			}
			if r.fp.deferHeader(events, i) {
				result.Count++
				continue
			}
			if r.fp.showExistingEligible(&events[i]) {
				step, err := r.fp.showExisting(sequence, &events[i])
				if err != nil {
					return result, err
				}
				result.Count++
				result.Last = step
				continue
			}
			if decoderFramePipelineEventNeedsDrain(&events[i]) {
				r.fp.drain()
			}
			if events[i].NewCodedVideoSequence {
				if err := r.fp.resetReferences(); err != nil {
					return result, err
				}
			}
		}

		sidePtr := (*decoderFrameWorkSideData)(nil)
		if decoderFrameWorkResidualEventNeedsSideData(&events[i]) && !decoderEventContinuesFrame(&events[i]) {
			if r.SideData == nil {
				return result, internaldecoder.ErrInvalidFrameWorkState
			}
			sidePtr = r.SideData
			var err error
			compact := r.compactState
			if compact != nil && !r.ownsCompactLoopFilter(sequence, postRunner, &events[i]) {
				internalthreading.DeactivateFrameWorkLoopFilterCompactState(compact)
				if r.BatchRunner != nil {
					r.BatchRunner.compactBoundSide = nil
				}
				compact = nil
			}
			if compact != nil {
				internalthreading.InvalidateFrameWorkLoopFilterCompactState(compact)
				*sidePtr, err = bindDecoderFrameWorkSideDataWith(sequence, events[i].FrameSize, events[i].CDEF, events[i].Restoration, sideScratch, compact, r.compactNoSkip, &events[i], false)
			} else {
				*sidePtr, err = bindDecoderFrameWorkSideDataWith(sequence, events[i].FrameSize, events[i].CDEF, events[i].Restoration, sideScratch, nil, nil, nil,
					r.lfLevels != nil && r.lfTileEdges != nil && r.lfNoSkip != nil)
				if err == nil && r.lfLevels != nil && sidePtr.LoopFilterMasks.Valid() {
					r.lfLevels.Bind(events[i].LoopFilter, events[i].Segmentation, events[i].Delta)
					sidePtr.LoopFilterMasks.BindLevels(r.lfLevels)
					sbLog2 := 4
					if events[i].SequenceHeader.Use128x128Superblock {
						sbLog2 = 5
					}
					sidePtr.LoopFilterMasks.BindTileEdges(r.lfTileEdges, events[i].TileInfo, sbLog2)
					sidePtr.LoopFilterMasks.BindNoSkip(r.lfNoSkip)
				}
			}
			if err != nil {
				return result, err
			}
			if compact != nil && r.BatchRunner != nil {
				r.BatchRunner.compactBoundSide = sidePtr
			}
		}

		var step internaldecoder.FrameWorkEventResult
		var err error
		if postRunner != nil {
			step, err = single.RunWithPostFilterRunner(sequence, &events[i], sidePtr, postRunner)
		} else {
			step, err = single.Run(sequence, &events[i], sidePtr, post)
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
		if decoderFrameWorkResidualResultOutputsFrame(&events[i], step) {
			populateFrameDisplayInfo(step.Output, sequence, &events[i], r.activeMeta)
			if r.fp != nil {
				r.fp.ready = append(r.fp.ready, step.Output)
			}
			if r.Outputs != nil {
				if result.OutputCount >= len(r.Outputs) {
					return result, internalframe.ErrShortBuffer
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

func (r decoderFrameWorkResidualEventRunner) ownsCompactLoopFilter(sequence SequenceHeader, post internaldecoder.FrameWorkPostFilterRunner, event *internaldecoder.Event) bool {
	if event == nil {
		return false
	}
	ownedSide, sideOK := r.sideRunner.(decoderFrameWorkResidualSideDataRunner)
	if !sideOK || ownedSide.SideData == nil || ownedSide.SideData != r.SideData || ownedSide.BatchRunner != r.BatchRunner {
		return false
	}
	payloadEvent := event.Kind == internaldecoder.EventFrame || event.Kind == internaldecoder.EventTileGroup
	kindEligible := event.Kind == internaldecoder.EventFrameHeader || payloadEvent
	if payloadEvent && (event.TileGroup.TileCount != 1 || !event.TileGroup.Final) {
		return false
	}
	if r.compactState == nil || r.BatchRunner == nil || r.compactBatchOwner == nil ||
		r.BatchRunner != r.compactBatchOwner || r.BatchRunner.compactOwner != r.BatchRunner ||
		r.SideData == nil || r.compactSideOwner == nil || r.SideData != r.compactSideOwner ||
		r.BatchRunner.compactSideOwner != r.compactSideOwner ||
		r.Workers != 1 || r.External.Enabled() || !kindEligible ||
		sequence.ColorConfig.BitDepth != 8 || event.FrameSize.SuperResEnabled || event.FrameSize.AllowIntrabc ||
		event.TileInfo.Cols != 1 || event.TileInfo.Rows != 1 ||
		!(event.LoopFilter.LevelY[0] != 0 || event.LoopFilter.LevelY[1] != 0) {
		return false
	}
	if r.WorkerPool != nil && r.WorkerPool.WorkerCount() != 1 {
		return false
	}
	owner, ok := post.(*decoderFrameWorkReusableSupportedPostFilterRunner)
	return ok && owner == r.compactPostOwner && r.compactPostOwner != nil && len(r.compactNoSkip) != 0
}

func decoderFrameWorkResidualEventsOutputLen(events []internaldecoder.Event) int {
	count := 0
	for i := range events {
		count += decoderFrameWorkResidualEventOutputLen(&events[i])
	}
	return count
}

func decoderFrameWorkResidualEventOutputLen(event *internaldecoder.Event) int {
	if internaldecoder.EventOutputsFrame(event) {
		return 1
	}
	return 0
}

func decoderFrameWorkResidualEventOutputLenPtr(event *internaldecoder.Event) int {
	if event == nil {
		return 0
	}
	return decoderFrameWorkResidualEventOutputLen(&*event)
}

func runDecoderFrameWorkEventWithSideRunner(req decoderFrameWorkResidualEventRequest, sideRunner internaldecoder.FrameWorkSideDataRunner) (result internaldecoder.FrameWorkEventResult, runErr error) {
	defer func() {
		if runErr != nil && req.Runner != nil && req.Runner.compactOwner == req.Runner {
			internalthreading.InvalidateFrameWorkLoopFilterCompactState(req.Runner.compactState)
		}
	}()
	if decoderFrameWorkResidualEventHasTilePayload(&req.Event) && req.Runner == nil {
		return internaldecoder.FrameWorkEventResult{}, internalthreading.ErrInvalidBatch
	}
	if req.Post != nil && req.PostRunner != nil {
		return internaldecoder.FrameWorkEventResult{}, internaldecoder.ErrInvalidFrameWorkState
	}
	event := req.Event
	event.SequenceHeader = req.Sequence

	if event.Kind == internaldecoder.EventTileList {
		if event.TileListErr != nil {
			return internaldecoder.FrameWorkEventResult{}, event.TileListErr
		}
		if _, err := internalparser.TileListOutputGeometryForGrid(event.TileList, event.TileInfo, event.SequenceHeader.Use128x128Superblock); err != nil {
			return internaldecoder.FrameWorkEventResult{}, err
		}
		if req.Runner == nil {
			return internaldecoder.FrameWorkEventResult{}, internalthreading.ErrInvalidBatch
		}
		if err := req.Runner.ResetStats(); err != nil {
			return internaldecoder.FrameWorkEventResult{}, err
		}
		if req.Stats != nil {
			*req.Stats = internalthreading.FrameWorkTileResidualStats{}
		}
		result, err := runDecoderFrameWorkTileListWithResidualRunner(req, &event)
		stats, statsErr := decoderFrameWorkResidualRunnerStats(req.Runner)
		if req.Stats != nil {
			*req.Stats = stats
		}
		if err != nil {
			return internaldecoder.FrameWorkEventResult{}, err
		}
		if statsErr != nil {
			return internaldecoder.FrameWorkEventResult{}, statsErr
		}
		return result, nil
	}

	if req.Runner != nil {
		if err := req.Runner.ResetStats(); err != nil {
			return internaldecoder.FrameWorkEventResult{}, err
		}
	}
	if req.Stats != nil {
		*req.Stats = internalthreading.FrameWorkTileResidualStats{}
	}

	if req.External.Enabled() {
		return runDecoderFrameWorkEventWithExternalResidualRunner(req, &event, sideRunner)
	}

	step, output, err := req.State.PlanEvent(req.Refs, req.FramePool, req.Sequence, &event, req.Align, req.ReferenceSurfaces, req.Workers, req.Spans, req.Jobs, req.Batches, req.Releases)
	if err != nil {
		return internaldecoder.FrameWorkEventResult{}, err
	}

	tile, referenceCount, hasTile, err := decoderFrameWorkEventStepTile(step)
	if err != nil {
		return internaldecoder.FrameWorkEventResult{}, err
	}
	if hasTile && req.SideData != nil {
		if decoderFrameWorkResidualSideRunnerOwnsCompact(sideRunner, req.Runner, req.SideData) {
			if err := sideRunner.BindFrameWorkSideData(req.State, internaldecoder.FrameWorkBatch{}); err != nil {
				return internaldecoder.FrameWorkEventResult{}, err
			}
		} else {
			if tile.JobCount != 0 {
				if err := setDecoderFrameWorkBatchResidualRunnerSideData(req.Runner, *req.SideData); err != nil {
					return internaldecoder.FrameWorkEventResult{}, err
				}
			}
			if err := setDecoderFrameWorkSideData(req.State, *req.SideData); err != nil {
				return internaldecoder.FrameWorkEventResult{}, err
			}
		}
	}

	if step.Kind == internaldecoder.FrameWorkStepShowExisting {
		output, err = req.FramePool.Frame(step.ShowExisting.Surface)
		if err != nil {
			return internaldecoder.FrameWorkEventResult{}, err
		}
	}

	var references []*Frame
	if hasTile && tile.JobCount != 0 {
		if output == nil {
			surface, err := decoderFrameWorkEventStepSurface(step)
			if err != nil {
				return internaldecoder.FrameWorkEventResult{}, err
			}
			output, err = req.FramePool.Frame(surface)
			if err != nil {
				return internaldecoder.FrameWorkEventResult{}, err
			}
		}
		if referenceCount != 0 {
			referenceCountN := int(referenceCount)
			if len(req.ReferenceSurfaces) < referenceCountN {
				return internaldecoder.FrameWorkEventResult{}, internaldecoder.ErrSurfaceReferenceBufferTooSmall
			}
			if step.Kind == internaldecoder.FrameWorkStepTile {
				count, err := req.Refs.FrameReferences(&event, req.ReferenceSurfaces)
				if err != nil {
					return internaldecoder.FrameWorkEventResult{}, err
				}
				if count != referenceCountN {
					return internaldecoder.FrameWorkEventResult{}, internaldecoder.ErrInvalidFrameWorkStep
				}
			}
			count, err := internaldecoder.ResolveFrameReferences(req.FramePool, req.ReferenceSurfaces[:referenceCountN], req.ReferenceFrames)
			if err != nil {
				return internaldecoder.FrameWorkEventResult{}, err
			}
			if count != referenceCountN {
				return internaldecoder.FrameWorkEventResult{}, internaldecoder.ErrInvalidFrameWorkStep
			}
			references = req.ReferenceFrames[:referenceCountN]
		}
	}

	var run internaldecoder.FrameWorkStepResult
	if req.WorkerPool != nil && req.WorkerPool.WorkerCount() == 1 {
		var compact *internalthreading.FrameWorkLoopFilterCompactState
		if decoderFrameWorkResidualSideRunnerOwnsCompact(sideRunner, req.Runner, req.SideData) {
			compact = req.Runner.compactState
		}
		run, err = runDecoderFrameWorkEventSingleWorkerResidualRunner(req, &event, step, output, references, compact)
	} else if req.PostRunner != nil {
		run, err = req.State.RunStepWithPayloadContextAndPostFilterRunners(req.Refs, req.FramePool, &event, step, req.WorkerPool, output, references, event.Unit.Payload, req.Jobs, req.Batches, req.Releases, req.Runner, req.PostRunner)
	} else {
		run, err = req.State.RunStepWithPayloadContextAndPostFilterRunner(req.Refs, req.FramePool, &event, step, req.WorkerPool, output, references, event.Unit.Payload, req.Jobs, req.Batches, req.Releases, req.Runner, req.Post)
	}
	stats, statsErr := decoderFrameWorkResidualRunnerStats(req.Runner)
	if req.Stats != nil {
		*req.Stats = stats
	}
	if err != nil {
		return internaldecoder.FrameWorkEventResult{}, err
	}
	if statsErr != nil {
		return internaldecoder.FrameWorkEventResult{}, statsErr
	}
	output = decoderFrameWorkResidualPostFilterOutput(output, run, req.PostRunner)
	if output, err = decoderFrameWorkShowExistingFilmGrain(&event, step, output, req.PostRunner); err != nil {
		return internaldecoder.FrameWorkEventResult{}, err
	}
	return internaldecoder.FrameWorkEventResult{
		Step:           step,
		Output:         output,
		ReferenceCount: referenceCount,
		Run:            run,
	}, nil
}

func runDecoderFrameWorkEventSingleWorkerResidualRunner(req decoderFrameWorkResidualEventRequest, event *internaldecoder.Event, step internaldecoder.FrameWorkStep, output *Frame, references []*Frame, compact *internalthreading.FrameWorkLoopFilterCompactState) (internaldecoder.FrameWorkStepResult, error) {
	prepared, err := internaldecoder.PrepareFrameWorkStepWithPayloadContext(req.State, event, step, output, references, event.Unit.Payload, req.Jobs, req.Batches)
	if err != nil {
		return internaldecoder.FrameWorkStepResult{}, err
	}
	if !prepared.HasTileWork {
		return internaldecoder.CompleteFrameWorkPreparedPayloadStep(req.State, req.Refs, req.FramePool, event, step, prepared, output, req.Releases, false, req.Post)
	}
	batches := req.Batches[:prepared.BatchCount]
	jobs := req.Jobs[:prepared.JobCount]
	if err := internalthreading.ValidateSingleWorkerFrameWorkBatches(req.WorkerPool, batches, jobs); err != nil {
		return internaldecoder.FrameWorkStepResult{}, err
	}
	var firstErr error
	for i := range batches {
		batch := batches[i]
		batchJobs, err := batch.JobSlice(jobs)
		if err != nil {
			return internaldecoder.FrameWorkStepResult{}, err
		}
		ctx := prepared.Base
		ctx.Batch = batch
		ctx.Jobs = batchJobs
		if err := req.Runner.RunPtr(&ctx); firstErr == nil && err != nil {
			firstErr = err
		}
	}
	if firstErr != nil {
		return internaldecoder.FrameWorkStepResult{}, firstErr
	}
	if req.PostRunner != nil {
		if compact != nil {
			return internaldecoder.CompleteFrameWorkPreparedPayloadStepRunnerOwned(req.State, req.Refs, req.FramePool, event, step, prepared, output, req.Releases, true, req.PostRunner, compact)
		}
		return internaldecoder.CompleteFrameWorkPreparedPayloadStepRunner(req.State, req.Refs, req.FramePool, event, step, prepared, output, req.Releases, true, req.PostRunner)
	}
	return internaldecoder.CompleteFrameWorkPreparedPayloadStep(req.State, req.Refs, req.FramePool, event, step, prepared, output, req.Releases, true, req.Post)
}

func runDecoderFrameWorkEventWithExternalResidualRunner(req decoderFrameWorkResidualEventRequest, event *internaldecoder.Event, sideRunner internaldecoder.FrameWorkSideDataRunner) (internaldecoder.FrameWorkEventResult, error) {
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
		decoderResolvePostPublisher(req.External.PostPublisher, postRunner),
	)
	stats, statsErr := decoderFrameWorkResidualRunnerStats(req.Runner)
	if req.Stats != nil {
		*req.Stats = stats
	}
	if err != nil {
		return internaldecoder.FrameWorkEventResult{}, err
	}
	if statsErr != nil {
		return internaldecoder.FrameWorkEventResult{}, statsErr
	}
	result.Output = decoderFrameWorkResidualPostFilterOutput(result.Output, result.Run, postRunner)
	if result.Output, err = decoderFrameWorkShowExistingFilmGrain(event, result.Step, result.Output, postRunner); err != nil {
		return internaldecoder.FrameWorkEventResult{}, err
	}
	return result, nil
}

func runDecoderFrameWorkTileListWithResidualRunner(req decoderFrameWorkResidualEventRequest, event *internaldecoder.Event) (internaldecoder.FrameWorkEventResult, error) {
	if event.TileListErr != nil {
		return internaldecoder.FrameWorkEventResult{}, event.TileListErr
	}
	if req.FramePool == nil {
		return internaldecoder.FrameWorkEventResult{}, internalframe.ErrInvalidPool
	}
	if req.WorkerPool == nil {
		return internaldecoder.FrameWorkEventResult{}, internalthreading.ErrInvalidWorkerCount
	}
	geometry, err := internalparser.TileListOutputGeometryForGrid(event.TileList, event.TileInfo, event.SequenceHeader.Use128x128Superblock)
	if err != nil {
		return internaldecoder.FrameWorkEventResult{}, err
	}
	if _, err := tileListOutputFrameFormat(event.SequenceHeader, geometry, req.Align); err != nil {
		return internaldecoder.FrameWorkEventResult{}, fmt.Errorf("tile-list output format: %w", err)
	}

	outputPool := req.TileListOutputPool
	if outputPool == nil {
		outputPool = req.FramePool
	}
	entryPool := outputPool
	outputSurface, output, err := outputPool.Acquire()
	if err != nil {
		return internaldecoder.FrameWorkEventResult{}, fmt.Errorf("tile-list output acquire: %w", err)
	}
	outputPublished := false
	defer func() {
		if !outputPublished {
			_ = outputPool.Release(outputSurface)
		}
	}()

	entrySurface, entryOutput, err := entryPool.Acquire()
	if err != nil {
		return internaldecoder.FrameWorkEventResult{}, fmt.Errorf("tile-list entry acquire: %w", err)
	}
	entryReleased := false
	defer func() {
		if !entryReleased {
			_ = entryPool.Release(entrySurface)
		}
	}()

	var anchors [internalparser.TileListMaxExternalReferences]*Frame
	if _, err := decoderFrameWorkResolveTileListAnchorFrames(req, event, anchors[:]); err != nil {
		return internaldecoder.FrameWorkEventResult{}, fmt.Errorf("tile-list anchor resolve: %w", err)
	}
	var references [internalparser.InterRefsPerFrame]*Frame
	var lastStep internaldecoder.FrameWorkStep
	var referenceCount uint8
	executed := false
	tileCount := event.TileList.TileCount()
	for i := 0; i < tileCount; i++ {
		for j := range references {
			references[j] = nil
		}
		plan, err := internaldecoder.PlanTileListEntryDecode(event, i, entrySurface, anchors[:], references[:], req.Workers, req.Spans, req.Jobs, req.Batches)
		if err != nil {
			return internaldecoder.FrameWorkEventResult{}, fmt.Errorf("tile-list entry plan %d: %w", i, err)
		}
		prepared, err := internaldecoder.PrepareFrameWorkStepWithPayloadContext(nil, &plan.Event, plan.Step, entryOutput, references[:plan.ReferenceCount], plan.Payload, req.Jobs, req.Batches)
		if err != nil {
			return internaldecoder.FrameWorkEventResult{}, fmt.Errorf("tile-list entry prepare %d: %w", i, err)
		}
		entryExecuted, err := runDecoderFrameWorkTileListPreparedStep(req, prepared)
		if err != nil {
			return internaldecoder.FrameWorkEventResult{}, fmt.Errorf("tile-list entry run %d: %w", i, err)
		}
		if err := copyTileListEntryToOutputFrame(output, entryOutput, event.TileList, plan.Geometry, i); err != nil {
			return internaldecoder.FrameWorkEventResult{}, fmt.Errorf("tile-list entry copy %d: %w", i, err)
		}
		lastStep = plan.Step
		referenceCount = plan.ReferenceCount
		executed = executed || entryExecuted
	}
	if err := entryPool.Release(entrySurface); err != nil {
		return internaldecoder.FrameWorkEventResult{}, fmt.Errorf("tile-list entry release: %w", err)
	}
	entryReleased = true
	outputPublished = true
	return internaldecoder.FrameWorkEventResult{
		Step:           lastStep,
		Output:         output,
		ReferenceCount: referenceCount,
		Run: internaldecoder.FrameWorkStepResult{
			ExecutedTileWork: executed,
		},
	}, nil
}

func runDecoderFrameWorkTileListPreparedStep(req decoderFrameWorkResidualEventRequest, prepared internaldecoder.FrameWorkPreparedPayloadStep) (bool, error) {
	if !prepared.HasTileWork {
		return false, nil
	}
	batches := req.Batches[:prepared.BatchCount]
	jobs := req.Jobs[:prepared.JobCount]
	if req.WorkerPool.WorkerCount() == 1 {
		if err := internalthreading.ValidateSingleWorkerFrameWorkBatches(req.WorkerPool, batches, jobs); err != nil {
			return false, err
		}
		var firstErr error
		for i := range batches {
			batch := batches[i]
			batchJobs, err := batch.JobSlice(jobs)
			if err != nil {
				return false, err
			}
			ctx := prepared.Base
			ctx.Batch = batch
			ctx.Jobs = batchJobs
			if err := req.Runner.RunPtr(&ctx); firstErr == nil && err != nil {
				firstErr = err
			}
		}
		if firstErr != nil {
			return false, firstErr
		}
		return true, nil
	}
	if err := req.WorkerPool.ExecuteFrameWorkRunner(batches, jobs, prepared.Base, req.Runner); err != nil {
		return false, err
	}
	return true, nil
}

func decoderFrameWorkResolveTileListAnchorFrames(req decoderFrameWorkResidualEventRequest, event *internaldecoder.Event, dst []*Frame) (int, error) {
	tileCount := event.TileList.TileCount()
	if tileCount <= 0 || tileCount > internalparser.TileListMaxTiles || tileCount != len(event.TileList.Entries) {
		return 0, internalparser.ErrTileListInvalidTileCount
	}
	var used [internalparser.TileListMaxExternalReferences]bool
	maxAnchor := -1
	for i := range event.TileList.Entries {
		anchor := int(event.TileList.Entries[i].AnchorFrameIdx)
		if anchor < 0 || anchor >= internalparser.TileListMaxExternalReferences {
			return 0, internalparser.ErrTileListInvalidAnchorIndex
		}
		used[anchor] = true
		if anchor > maxAnchor {
			maxAnchor = anchor
		}
	}
	if maxAnchor < 0 {
		return 0, internalparser.ErrTileListInvalidTileCount
	}
	count := maxAnchor + 1
	if len(dst) < count {
		return 0, internaldecoder.ErrSurfaceReferenceBufferTooSmall
	}
	for i := 0; i < count; i++ {
		dst[i] = nil
	}
	for anchor := 0; anchor < count; anchor++ {
		if !used[anchor] {
			continue
		}
		surface, err := decoderFrameWorkTileListAnchorSurface(req, anchor)
		if err != nil {
			return 0, err
		}
		frame, err := decoderFrameWorkTileListAnchorFrame(req, surface)
		if err != nil {
			return 0, err
		}
		dst[anchor] = frame
	}
	return count, nil
}

func decoderFrameWorkTileListAnchorSurface(req decoderFrameWorkResidualEventRequest, anchor int) (int, error) {
	if req.Refs != nil && anchor < internalparser.RefFrames {
		if surface, ok := req.Refs.ReferenceSlot(anchor); ok {
			return surface, nil
		}
	}
	if req.External.Enabled() {
		if anchor >= len(req.ReferenceSurfaces) {
			return -1, internaldecoder.ErrSurfaceReferenceBufferTooSmall
		}
		surface := req.ReferenceSurfaces[anchor]
		if surface < 0 {
			return -1, internaldecoder.ErrInvalidSurfaceReference
		}
		return surface, nil
	}
	if anchor >= len(req.ReferenceSurfaces) {
		return -1, internaldecoder.ErrSurfaceReferenceBufferTooSmall
	}
	surface := req.ReferenceSurfaces[anchor]
	if surface < 0 {
		return -1, internaldecoder.ErrInvalidSurfaceReference
	}
	return surface, nil
}

func decoderFrameWorkTileListAnchorFrame(req decoderFrameWorkResidualEventRequest, surface int) (*Frame, error) {
	if req.External.Enabled() {
		frame, err := req.External.Provider.FrameSurface(surface)
		if err != nil {
			return nil, err
		}
		if frame == nil {
			return nil, internaldecoder.ErrInvalidSurfaceReference
		}
		return frame, nil
	}
	if req.FramePool == nil {
		return nil, internalframe.ErrInvalidPool
	}
	frame, err := req.FramePool.Frame(surface)
	if err != nil {
		return nil, err
	}
	if frame == nil {
		return nil, internaldecoder.ErrInvalidSurfaceReference
	}
	return frame, nil
}

type decoderFrameWorkResidualPostFuncRunner struct {
	fn internaldecoder.FrameWorkPostFilterFunc
}

func (r decoderFrameWorkResidualPostFuncRunner) Apply(ctx internaldecoder.FrameWorkPostFilterContext) error {
	if r.fn == nil {
		return nil
	}
	return r.fn(ctx)
}

type decoderFrameWorkResidualSideDataRunner struct {
	SideData    *decoderFrameWorkSideData
	BatchRunner *decoderFrameWorkBatchResidualRunner
}

func decoderFrameWorkResidualSideRunnerOwnsCompact(sideRunner internaldecoder.FrameWorkSideDataRunner, batchRunner *decoderFrameWorkBatchResidualRunner, side *decoderFrameWorkSideData) bool {
	r, ok := sideRunner.(decoderFrameWorkResidualSideDataRunner)
	if !ok || side == nil || r.SideData != side || batchRunner == nil || r.BatchRunner != batchRunner ||
		batchRunner.compactOwner != batchRunner || batchRunner.compactState == nil ||
		batchRunner.compactSideOwner == nil || r.SideData != batchRunner.compactSideOwner ||
		batchRunner.compactBoundSide != side {
		return false
	}
	return internalthreading.FrameWorkLoopFilterCompactStateActive(batchRunner.compactState)
}

func (r decoderFrameWorkResidualSideDataRunner) BindFrameWorkSideData(state *internaldecoder.FrameWorkState, _ internaldecoder.FrameWorkBatch) error {
	if r.SideData == nil {
		return nil
	}
	if r.BatchRunner != nil && r.BatchRunner.compactOwner == r.BatchRunner &&
		r.BatchRunner.compactSideOwner != nil && r.SideData == r.BatchRunner.compactSideOwner &&
		r.BatchRunner.compactBoundSide == r.SideData &&
		internalthreading.FrameWorkLoopFilterCompactStateActive(r.BatchRunner.compactState) {
		compact := r.BatchRunner.compactState
		if err := internaldecoder.SetOwnedFrameWorkLoopFilterSideData(state, r.SideData.CDEFIndexMap, r.SideData.LoopFilterMasks, compact, r.SideData.RestorationFrameBuffers); err != nil {
			return err
		}
		return r.BatchRunner.setOwnedCompactSideData(r.SideData)
	}
	if r.BatchRunner != nil {
		if err := setDecoderFrameWorkBatchResidualRunnerSideData(r.BatchRunner, *r.SideData); err != nil {
			return err
		}
	}
	return setDecoderFrameWorkSideData(state, *r.SideData)
}

func decoderFrameWorkResidualRunnerStats(runner *decoderFrameWorkBatchResidualRunner) (internalthreading.FrameWorkTileResidualStats, error) {
	if runner == nil {
		return internalthreading.FrameWorkTileResidualStats{}, nil
	}
	return runner.TotalStats()
}

func populateFrameDisplayInfo(dst *Frame, seq SequenceHeader, event *internaldecoder.Event, meta FrameMetadata) {
	if dst == nil || event == nil {
		return
	}
	cc := seq.ColorConfig
	dst.Color = ColorInfo{
		Primaries:               cc.ColorPrimaries,
		TransferCharacteristics: cc.TransferCharacteristics,
		MatrixCoefficients:      cc.MatrixCoefficients,
		ChromaSamplePosition:    cc.ChromaSamplePosition,
		ColorRange:              cc.ColorRange,
		RenderWidth:             int(event.FrameSize.RenderWidth),
		RenderHeight:            int(event.FrameSize.RenderHeight),
	}
	dst.Metadata = meta
}

func decoderFrameWorkResidualResultOutputsFrame(event *internaldecoder.Event, result internaldecoder.FrameWorkEventResult) bool {
	if result.Output == nil || event.SuppressDisplay {
		return false
	}
	if result.Step.Kind == internaldecoder.FrameWorkStepShowExisting {
		return true
	}
	if event.Kind == internaldecoder.EventTileList && event.TileListErr == nil {
		return true
	}
	// A completed coded frame is only emitted to the output stream when its
	// header sets show_frame. Not-shown frames (e.g. alt-ref) are decoded for
	// their reference-buffer side effects but surface later via
	// show_existing_frame, so they must stay out of Outputs. DecoderEventOutputsFrame
	// also gates the per-event output-slot sizing, keeping the two in lockstep.
	return result.Run.CompletedFrame && internaldecoder.EventOutputsFrame(event)
}

func decoderFrameWorkResidualPostFilterOutput(output *Frame, run internaldecoder.FrameWorkStepResult, post internaldecoder.FrameWorkPostFilterRunner) *Frame {
	if !run.CompletedFrame || post == nil {
		return output
	}
	provider := decoderPostFilterOutputProviderOf(post)
	if provider == nil {
		return output
	}
	postOutput, ok := provider.PostFilterOutput()
	if !ok || postOutput == nil {
		return output
	}
	return postOutput
}

func decoderFrameWorkResidualEventSequence(sequence SequenceHeader, event *internaldecoder.Event) SequenceHeader {
	return decoderFrameWorkResidualEventSequencePtr(sequence, event)
}

func decoderFrameWorkResidualEventSequencePtr(sequence SequenceHeader, event *internaldecoder.Event) SequenceHeader {
	if event == nil {
		return sequence
	}
	switch event.Kind {
	case internaldecoder.EventSequenceHeader:
		if decoderFrameWorkResidualEventHasSequencePtr(event) {
			return event.SequenceHeader
		}
		return sequence
	case internaldecoder.EventFrameHeader,
		internaldecoder.EventRedundantFrameHeader,
		internaldecoder.EventFrame,
		internaldecoder.EventTileGroup,
		internaldecoder.EventExistingFrame:
		if decoderFrameWorkResidualEventHasSequencePtr(event) {
			return event.SequenceHeader
		}
		return sequence
	default:
		return sequence
	}
}

func decoderFrameWorkResidualEventHasSequencePtr(event *internaldecoder.Event) bool {
	if event == nil {
		return false
	}
	return event.SequenceHeader.ColorConfig.BitDepth != 0
}

func decoderFrameWorkResidualEventNeedsSideData(event *internaldecoder.Event) bool {
	return decoderFrameWorkResidualEventNeedsSideDataPtr(event)
}

func decoderFrameWorkResidualEventNeedsSideDataPtr(event *internaldecoder.Event) bool {
	if event == nil {
		return false
	}
	switch event.Kind {
	case internaldecoder.EventFrameHeader, internaldecoder.EventFrame, internaldecoder.EventTileGroup:
		return true
	default:
		return false
	}
}

func decoderFrameWorkResidualEventHasTilePayload(event *internaldecoder.Event) bool {
	return decoderFrameWorkResidualEventHasTilePayloadPtr(event)
}

func decoderFrameWorkResidualEventHasTilePayloadPtr(event *internaldecoder.Event) bool {
	if event == nil {
		return false
	}
	switch event.Kind {
	case internaldecoder.EventFrame, internaldecoder.EventTileGroup:
		return true
	default:
		return false
	}
}

func decoderFrameWorkResidualEventBindCandidate(event *internaldecoder.Event) bool {
	return decoderFrameWorkResidualEventBindCandidatePtr(event)
}

// decoderFrameWorkResidualEventBindCandidatePtr reports whether event sizes or
// binds the decoder. The header of a frame dropped incomplete still does, as
// it did before the frame was dropped.
func decoderFrameWorkResidualEventBindCandidatePtr(event *internaldecoder.Event) bool {
	if event != nil && event.DroppedKind != internaldecoder.EventIgnored {
		return true
	}
	return decoderFrameWorkResidualEventNeedsSideDataPtr(event) ||
		decoderFrameWorkResidualEventHasTilePayloadPtr(event) ||
		(event != nil && internaldecoder.EventOutputsFrame(&*event))
}

func decoderFrameWorkResidualEventPlanMax(a internaldecoder.TileWorkPlan, b internaldecoder.TileWorkPlan) internaldecoder.TileWorkPlan {
	return internaldecoder.TileWorkPlan{
		SpanCount:  max(a.SpanCount, b.SpanCount),
		JobCount:   max(a.JobCount, b.JobCount),
		BatchCount: max(a.BatchCount, b.BatchCount),
	}
}

func decoderFrameWorkEventStepTile(step internaldecoder.FrameWorkStep) (internaldecoder.TileWorkPlan, uint8, bool, error) {
	switch step.Kind {
	case internaldecoder.FrameWorkStepIgnored, internaldecoder.FrameWorkStepDropped, internaldecoder.FrameWorkStepShowExisting:
		return internaldecoder.TileWorkPlan{}, 0, false, nil
	case internaldecoder.FrameWorkStepBegin:
		return step.Begin.Tile, step.Begin.ReferenceCount, true, nil
	case internaldecoder.FrameWorkStepTile:
		return step.Tile.Tile, step.Tile.ReferenceCount, true, nil
	default:
		return internaldecoder.TileWorkPlan{}, 0, false, internaldecoder.ErrInvalidTileWork
	}
}

func decoderFrameWorkResidualEventContext(sequence SequenceHeader, event *internaldecoder.Event) internaldecoder.FrameWorkFrameContext {
	return decoderFrameWorkResidualEventContextPtr(sequence, event)
}

// decoderFrameWorkResidualEventContextPtr builds the context used by scratch
// sizing. Scaled-reference policy does not affect those lengths, so keep this
// conversion independent of environment lookups.
func decoderFrameWorkResidualEventContextPtr(sequence SequenceHeader, event *internaldecoder.Event) internaldecoder.FrameWorkFrameContext {
	if event == nil {
		return internaldecoder.FrameWorkFrameContext{
			Sequence: internalthreading.FrameWorkSequenceContextFromHeader(sequence),
		}
	}
	return internaldecoder.FrameWorkFrameContext{
		Sequence:            internalthreading.FrameWorkSequenceContextFromHeader(sequence),
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

func decoderFrameWorkEventStepSurface(step internaldecoder.FrameWorkStep) (int, error) {
	switch step.Kind {
	case internaldecoder.FrameWorkStepBegin:
		return step.Begin.Surface, nil
	case internaldecoder.FrameWorkStepTile:
		return step.Tile.Surface, nil
	case internaldecoder.FrameWorkStepShowExisting:
		return step.ShowExisting.Surface, nil
	default:
		return -1, internaldecoder.ErrInvalidFrameWorkStep
	}
}

// decoderNoGlobalSurfacePublisher reports that the post-filter runner never
// publishes a global surface.
type decoderNoGlobalSurfacePublisher struct{}

func (decoderNoGlobalSurfacePublisher) PublishedFrameWorkGlobalSurface() (int, bool) {
	return -1, false
}

// decoderResolvePostPublisher resolves the global-surface publisher for the
// decoder's own post-filter runners with a concrete type switch. Left nil, the
// event runner falls back to an interface-to-interface type assertion on every
// frame, and the runtime's per-site assertion cache grows (allocating) at
// random times while it warms up, which made the zero-alloc decode benchmarks
// report an occasional 48 B allocation. Unknown runner types keep that
// fallback.
func decoderResolvePostPublisher(publisher internaldecoder.FrameWorkPostFilterGlobalSurfacePublisher, post internaldecoder.FrameWorkPostFilterRunner) internaldecoder.FrameWorkPostFilterGlobalSurfacePublisher {
	if publisher != nil {
		return publisher
	}
	switch r := post.(type) {
	case nil:
		return nil
	case *decoderExternalPostFilterRunner:
		return r
	case *decoderFrameWorkReusableSupportedPostFilterRunner, *decoderFrameWorkSupportedPostFilterScratchRunner, decoderFrameWorkResidualPostFuncRunner:
		return decoderNoGlobalSurfacePublisher{}
	}
	return nil
}

// decoderPostFilterOutputProviderOf returns the display-output provider of the
// decoder's own post-filter runners through a concrete type switch, nil when
// post has none. An interface-to-interface assertion here would warm the
// runtime's per-site assertion cache, which allocates at random calls until
// every runner type seen at the site is cached.
func decoderPostFilterOutputProviderOf(post internaldecoder.FrameWorkPostFilterRunner) decoderFrameWorkPostFilterOutputProvider {
	switch r := post.(type) {
	case *decoderFrameWorkReusableSupportedPostFilterRunner:
		return r
	case *decoderFrameWorkSupportedPostFilterScratchRunner:
		return r
	case *decoderExternalPostFilterRunner:
		return r
	}
	return nil
}

// decoderFrameWorkShowExistingFilmGrain applies film grain to a
// show_existing_frame output. dav1d applies grain to every output picture whose
// frame header carries apply_grain (output_image -> dav1d_apply_grain), and a
// shown existing frame carries the film grain parameters, seed included, that
// were stored with it (load_grain_params). The reconstructed reference stays
// untouched; only its in-loop filters already ran, so every other stage is
// marked complete.
func decoderFrameWorkShowExistingFilmGrain(event *internaldecoder.Event, step internaldecoder.FrameWorkStep, output *Frame, post internaldecoder.FrameWorkPostFilterRunner) (*Frame, error) {
	if step.Kind != internaldecoder.FrameWorkStepShowExisting {
		return output, nil
	}
	return decoderFrameWorkOutputFilmGrain(event, step, output, post)
}

// decoderFrameWorkOutputFilmGrain applies film grain to an output picture
// whose in-loop filters already ran (dav1d output_image ->
// dav1d_apply_grain): post synthesizes the grain into its own display
// surface, and output stays the clean reference.
func decoderFrameWorkOutputFilmGrain(event *internaldecoder.Event, step internaldecoder.FrameWorkStep, output *Frame, post internaldecoder.FrameWorkPostFilterRunner) (*Frame, error) {
	if output == nil || post == nil || !event.FilmGrain.Apply {
		return output, nil
	}
	provider := decoderPostFilterOutputProviderOf(post)
	if provider == nil {
		return output, nil
	}
	ctx := internaldecoder.FrameWorkPostFilterContext{Event: *event, Step: step, Output: output}
	ctx = ctx.WithCompletedPostFilters(internaldecoder.FrameWorkPostFilterLoopFilter | internaldecoder.FrameWorkPostFilterCDEF |
		internaldecoder.FrameWorkPostFilterSuperRes | internaldecoder.FrameWorkPostFilterLoopRestoration)
	if err := post.Apply(ctx); err != nil {
		return nil, err
	}
	if grained, ok := provider.PostFilterOutput(); ok && grained != nil {
		return grained, nil
	}
	return output, nil
}

// decoderEventContinuesFrame reports whether event is a later tile group of a
// frame whose earlier tile groups already ran: its blocks add to the frame's
// CDEF index map, loop-filter masks and restoration records, so they must not
// be rebound (which resets them). dav1d likewise keeps one frame context across
// every tile group of a frame (src/decode.c dav1d_submit_frame runs once all
// tile data arrived).
func decoderEventContinuesFrame(event *internaldecoder.Event) bool {
	return event != nil && event.Kind == internaldecoder.EventTileGroup && event.TileGroup.StartTile != 0
}
